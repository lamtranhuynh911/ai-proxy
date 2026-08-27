"""
Anthropic-compatible proxy for Databricks AI Gateway.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from databricks.sdk.core import Config
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# ---------------------------------------------------------------------------
DATABRICKS_HOST = "https://adb-4482605715640778.18.azuredatabricks.net"
DATABRICKS_PROFILE = "adb-4482605715640778"
DATABRICKS_GATEWAY_PATH = "/ai-gateway/anthropic/v1/messages"

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
LISTEN_PORT = 8787
TOKEN_TTL_SECONDS = 55 * 60

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

_cfg = Config(host=DATABRICKS_HOST, profile=DATABRICKS_PROFILE)
_token: Optional[str] = None
_token_fetched_at: float = 0.0


def _strip_model_date_suffix(model: str) -> str:
    """Remove date suffix (e.g., -20241022) from model names."""
    return re.sub(r"-\d{8}$", "", model)


def get_token() -> str:
    global _token, _token_fetched_at
    now = time.time()
    if _token is None or (now - _token_fetched_at) > TOKEN_TTL_SECONDS:
        auth = _cfg.authenticate()
        _token = auth["Authorization"].split(" ", 1)[1]
        _token_fetched_at = now
        log.info("Refreshed Databricks token.")
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
        get_token()
    except Exception as e:
        log.error("Auth failed: %s. Run `databricks auth login --host %s`.",
                  e, DATABRICKS_HOST)
        raise
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)

#databricks auth login --host https://adb-4482605715640778.18.azuredatabricks.net