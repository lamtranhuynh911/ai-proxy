"""
Anthropic-compatible proxy for Databricks AI Gateway.
"""

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import dotenv
dotenv.load_dotenv()

# ---------------------------------------------------------------------------
DATABRICKS_GATEWAY_PATH = "/ai-gateway/anthropic/v1/messages"
DATABRICKS_HOST: Optional[str] = None
DATABRICKS_CLIENT_ID: Optional[str] = None
DATABRICKS_CLIENT_SECRET: Optional[str] = None

ENVIRONMENT_PREFIXES = {
    "d": "D_",
    "q": "Q_",
    "p": "P_",
}

MODEL_MAP: Dict[str, str] = {
    "claude-opus-5":              "system.ai.claude-opus-5",
    "claude-opus-4-7":            "system.ai.claude-opus-4-7",
    "claude-opus-4-8":            "system.ai.claude-opus-4-8",
    "claude-opus-4-6":            "system.ai.claude-opus-4-6",
    "claude-opus-4-5":            "system.ai.claude-opus-4-5",
    "claude-opus-4-1":            "system.ai.claude-opus-4-1",
    "claude-sonnet-5":            "system.ai.claude-sonnet-5",
    "claude-sonnet-4-6":          "system.ai.claude-sonnet-4-6",
    "claude-sonnet-4-5":          "system.ai.claude-sonnet-4-5",
    "claude-sonnet-4":            "system.ai.claude-sonnet-4",
    "claude-haiku-4-5":           "system.ai.claude-haiku-4-5",
}
DEFAULT_MODEL = "system.ai.claude-haiku-4-5"

LISTEN_HOST = "0.0.0.0"  # bind to all interfaces so WSL/other hosts can reach it
LISTEN_PORT = 8786
TOKEN_TTL_SECONDS = 55 * 60
TOKEN_REFRESH_SKEW_SECONDS = 60

# Fields at the top-level of an Anthropic request that Databricks' gateway
# is known to reject. Strip them defensively.
STRIP_TOP_LEVEL = {
    "metadata",
    "service_tier",
    "top_k",           # sometimes rejected
    "thinking",        # extended thinking may not be supported via gateway
    "mcp_servers",
}

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("db-anthropic-proxy")

_token: Optional[str] = None
_token_fetched_at: float = 0.0
_token_expires_in: int = TOKEN_TTL_SECONDS


def select_environment(environment: str) -> None:
    """Load Databricks settings for the selected environment."""
    global DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET

    environment = environment.strip().lower()
    prefix = ENVIRONMENT_PREFIXES.get(environment)
    if prefix is None:
        raise ValueError("Environment must be one of: d, q, p")

    settings = {
        "host": os.environ.get(f"{prefix}DATABRICKS_HOST"),
        "client_id": os.environ.get(f"{prefix}DATABRICKS_CLIENT_ID"),
        "client_secret": os.environ.get(f"{prefix}DATABRICKS_CLIENT_SECRET"),
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing {prefix}DATABRICKS settings: {', '.join(missing)}"
        )

    DATABRICKS_HOST = settings["host"]
    DATABRICKS_CLIENT_ID = settings["client_id"]
    DATABRICKS_CLIENT_SECRET = settings["client_secret"]
    log.info("Using Databricks environment %s.", environment)


def _strip_model_date_suffix(model: str) -> str:
    """Remove date suffix (e.g., -20241022) from model names."""
    return re.sub(r"-\d{8}$", "", model)


def _fetch_token() -> str:
    global _token, _token_fetched_at, _token_expires_in
    if not DATABRICKS_HOST or not DATABRICKS_CLIENT_ID or not DATABRICKS_CLIENT_SECRET:
        raise RuntimeError(
            "Databricks environment must be selected before authentication"
        )

    token_url = f"{DATABRICKS_HOST.rstrip('/')}/oidc/v1/token"
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=15.0)) as client:
        response = client.post(
            token_url,
            auth=(DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET),
            data={"grant_type": "client_credentials", "scope": "all-apis"},
        )
        response.raise_for_status()
        token_response = response.json()

    _token = token_response["access_token"]
    _token_expires_in = int(token_response.get("expires_in", TOKEN_TTL_SECONDS))
    _token_fetched_at = time.time()
    log.info("Refreshed Databricks service-principal token.")
    return _token


def get_token(force_refresh: bool = False) -> str:
    global _token, _token_fetched_at
    now = time.time()
    refresh_after = min(TOKEN_TTL_SECONDS, _token_expires_in) - TOKEN_REFRESH_SKEW_SECONDS
    if force_refresh or _token is None or (now - _token_fetched_at) > refresh_after:
        return _fetch_token()
    return _token


app = FastAPI(title="Databricks -> Anthropic proxy")
_http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))


def _strip_cache_control(obj: Any) -> Any:
    """Recursively remove 'cache_control' keys (prompt caching) from the body."""
    if isinstance(obj, dict):
        obj.pop("cache_control", None)
        for v in obj.values():
            _strip_cache_control(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_cache_control(v)
    return obj


def sanitize_body(body: Dict[str, Any]) -> Dict[str, Any]:
    # Model rewrite
    incoming = body.get("model")
    if not incoming:
        body["model"] = DEFAULT_MODEL
    else:
        # Strip date suffix (e.g., -20241022) from model names
        model_base = _strip_model_date_suffix(incoming)
        
        if model_base in MODEL_MAP:
            body["model"] = MODEL_MAP[model_base]
        elif not model_base.startswith("system.ai."):
            log.warning("Unknown model %r -> %s", incoming, DEFAULT_MODEL)
            body["model"] = DEFAULT_MODEL
        else:
            body["model"] = model_base

    # Strip unsupported top-level fields
    for k in list(body.keys()):
        if k in STRIP_TOP_LEVEL:
            log.info("Stripping unsupported field: %s", k)
            body.pop(k, None)

    # Strip prompt-caching markers everywhere
    _strip_cache_control(body)

    # Ensure max_tokens exists (Anthropic requires it)
    body.setdefault("max_tokens", 4096)

    # Some clients send system as a list of blocks with cache_control; that's fine
    # after stripping. But if 'system' is an empty list, drop it.
    if isinstance(body.get("system"), list) and not body["system"]:
        body.pop("system")

    return body


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": n, "type": "model"} for n in MODEL_MAP], "has_more": False}


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
    except Exception as e:
        return JSONResponse({"error": {"type": "invalid_request_error",
                                       "message": f"Invalid JSON: {e}"}},
                            status_code=400)

    # ---- log what the client sent (trimmed) ----
    try:
        preview = json.dumps({k: (v if k != "messages" else f"<{len(v)} msgs>")
                              for k, v in body.items()})
    except Exception:
        preview = "<unserializable>"
    log.info("Incoming request: %s", preview)

    body = sanitize_body(body)
    stream = bool(body.pop("stream", False))  # we'll re-add below only if we forward it

    url = f"{DATABRICKS_HOST}{DATABRICKS_GATEWAY_PATH}"
    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    # Deliberately do NOT forward: anthropic-beta, x-api-key, authorization from client

    log.info("Forwarding model=%s stream=%s keys=%s",
             body.get("model"), stream, list(body.keys()))

    if stream:
        body["stream"] = True

        async def event_stream():
            try:
                async with _http.stream("POST", url, headers=headers, json=body) as r:
                    if r.status_code == 401:
                        await r.aread()
                        headers["Authorization"] = f"Bearer {get_token(force_refresh=True)}"
                        async with _http.stream("POST", url, headers=headers, json=body) as retry:
                            if retry.status_code >= 400:
                                r = retry
                            else:
                                async for chunk in retry.aiter_raw():
                                    if chunk:
                                        yield chunk
                                return
                    if r.status_code >= 400:
                        err = await r.aread()
                        log.error("Upstream %d: %s", r.status_code, err.decode("utf-8", "replace"))
                        # Emit an SSE error event so the client sees something useful
                        msg = err.decode("utf-8", "replace")
                        yield (f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'api_error','message':msg}})}\n\n").encode()
                        return
                    async for chunk in r.aiter_raw():
                        if chunk:
                            yield chunk
            except httpx.HTTPError as e:
                log.exception("Upstream stream error")
                yield (f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'api_error','message':str(e)}})}\n\n").encode()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    try:
        r = await _http.post(url, headers=headers, json=body)
        if r.status_code == 401:
            headers["Authorization"] = f"Bearer {get_token(force_refresh=True)}"
            r = await _http.post(url, headers=headers, json=body)
    except httpx.HTTPError as e:
        log.exception("Upstream error")
        return JSONResponse({"error": {"type": "api_error", "message": str(e)}},
                            status_code=502)

    if r.status_code >= 400:
        log.error("Upstream %d body: %s", r.status_code, r.text)

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


@app.on_event("shutdown")
async def _shutdown():
    await _http.aclose()


if __name__ == "__main__":
    try:
        environment = input("Select Databricks environment (d=dev, q=qa, p=prod): ")
        select_environment(environment)
        get_token()
    except Exception as e:
        log.error("Startup failed: %s", e)
        raise
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)