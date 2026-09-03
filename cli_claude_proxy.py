"""Anthropic proxy for Claude Code, authenticated via the Databricks CLI.

Uses `databricks auth token -p <profile>` so no client secrets are needed.
"""

import argparse
import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
DATABRICKS_GATEWAY_PATH = "/ai-gateway/anthropic/v1/messages"
DATABRICKS_CLI = "databricks"

DATABRICKS_HOST: Optional[str] = None
DATABRICKS_PROFILE: Optional[str] = None

MODEL_MAP: Dict[str, str] = {
    "claude-opus-5":     "system.ai.claude-opus-5",
    "claude-opus-4-8":   "system.ai.claude-opus-4-8",
    "claude-opus-4-7":   "system.ai.claude-opus-4-7",
    "claude-opus-4-6":   "system.ai.claude-opus-4-6",
    "claude-opus-4-5":   "system.ai.claude-opus-4-5",
    "claude-opus-4-1":   "system.ai.claude-opus-4-1",
    "claude-sonnet-5":   "system.ai.claude-sonnet-5",
    "claude-sonnet-4-6": "system.ai.claude-sonnet-4-6",
    "claude-sonnet-4-5": "system.ai.claude-sonnet-4-5",
    "claude-sonnet-4":   "system.ai.claude-sonnet-4",
    "claude-haiku-4-5":  "system.ai.claude-haiku-4-5",
}
DEFAULT_MODEL = "system.ai.claude-haiku-4-5"

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8786
TOKEN_TTL_SECONDS = 55 * 60
TOKEN_REFRESH_SKEW_SECONDS = 120

STRIP_TOP_LEVEL = {
    "metadata",
    "service_tier",
    "top_k",
    "mcp_servers",
    "context_management",
}

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cli-dbx-proxy")

_token: Optional[str] = None
_token_expiry: float = 0.0


# --------------------------- CLI auth --------------------------------------
def _run_cli(args: List[str]) -> str:
    proc = subprocess.run(
        [DATABRICKS_CLI, *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{DATABRICKS_CLI} {' '.join(args)}` failed: "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return proc.stdout


def list_profiles() -> List[Dict[str, Any]]:
    data = json.loads(_run_cli(["auth", "profiles", "-o", "json"]))
    return data.get("profiles", [])


def select_profile(profile: Optional[str]) -> None:
    """Resolve the CLI profile and its workspace host."""
    global DATABRICKS_HOST, DATABRICKS_PROFILE

    profiles = list_profiles()
    if not profiles:
        raise RuntimeError("No Databricks CLI profiles found. Run `databricks auth login`.")

    if profile:
        match = next((p for p in profiles if p.get("name") == profile), None)
        if match is None:
            names = ", ".join(p.get("name", "?") for p in profiles)
            raise RuntimeError(f"Profile {profile!r} not found. Available: {names}")
    else:
        match = next((p for p in profiles if p.get("valid")), None)
        if match is None:
            raise RuntimeError("No valid profile. Run `databricks auth login`.")
        log.info("No --profile given; using first valid profile.")

    if not match.get("valid"):
        log.warning("Profile %s is marked invalid; attempting anyway.", match.get("name"))

    DATABRICKS_PROFILE = match["name"]
    DATABRICKS_HOST = str(match["host"]).rstrip("/")
    log.info("Using profile %s (%s).", DATABRICKS_PROFILE, DATABRICKS_HOST)


def _parse_expiry(value: Any) -> float:
    """Parse the CLI's RFC3339 expiry into an epoch timestamp."""
    if not isinstance(value, str) or not value:
        return time.time() + TOKEN_TTL_SECONDS
    try:
        text = value.replace("Z", "+00:00")
        text = re.sub(r"\.(\d{6})\d+", r".\1", text)  # trim ns -> us
        return datetime.fromisoformat(text).astimezone(timezone.utc).timestamp()
    except ValueError:
        return time.time() + TOKEN_TTL_SECONDS


def _fetch_token() -> str:
    global _token, _token_expiry
    if not DATABRICKS_PROFILE:
        raise RuntimeError("Profile must be selected before authentication")

    payload = json.loads(_run_cli(["auth", "token", "-p", DATABRICKS_PROFILE]))
    _token = payload["access_token"]
    _token_expiry = _parse_expiry(payload.get("expiry"))
    log.info("Refreshed token via Databricks CLI (profile=%s).", DATABRICKS_PROFILE)
    return _token


def get_token(force_refresh: bool = False) -> str:
    if force_refresh or _token is None or time.time() >= (_token_expiry - TOKEN_REFRESH_SKEW_SECONDS):
        return _fetch_token()
    return _token


# --------------------------- request shaping -------------------------------
app = FastAPI(title="Databricks CLI -> Claude Code proxy")
_http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))


def _strip_model_date_suffix(model: str) -> str:
    return re.sub(r"-\d{8}$", "", model)


def _strip_cache_control(obj: Any) -> Any:
    if isinstance(obj, dict):
        obj.pop("cache_control", None)
        for v in obj.values():
            _strip_cache_control(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_cache_control(v)
    return obj


def sanitize_body(body: Dict[str, Any]) -> Dict[str, Any]:
    incoming = body.get("model")
    if not incoming:
        body["model"] = DEFAULT_MODEL
    else:
        model_base = _strip_model_date_suffix(incoming)
        if model_base in MODEL_MAP:
            body["model"] = MODEL_MAP[model_base]
        elif not model_base.startswith("system.ai."):
            log.warning("Unknown model %r -> %s", incoming, DEFAULT_MODEL)
            body["model"] = DEFAULT_MODEL
        else:
            body["model"] = model_base

    for k in list(body.keys()):
        if k in STRIP_TOP_LEVEL:
            log.info("Stripping unsupported field: %s", k)
            body.pop(k, None)

    _strip_cache_control(body)
    body.setdefault("max_tokens", 4096)

    if isinstance(body.get("system"), list) and not body["system"]:
        body.pop("system")

    return body


# --------------------------- routes ----------------------------------------
@app.get("/health")
async def health():
    return {"ok": True, "profile": DATABRICKS_PROFILE, "host": DATABRICKS_HOST}


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": n, "type": "model"} for n in MODEL_MAP], "has_more": False}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": f"Invalid JSON: {e}"}},
            status_code=400,
        )
    text = json.dumps(body, ensure_ascii=False)
    return {"input_tokens": max(1, len(text) // 4)}


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        raw = await request.body()
        body = json.loads(raw) if raw else {}
    except Exception as e:
        return JSONResponse(
            {"error": {"type": "invalid_request_error", "message": f"Invalid JSON: {e}"}},
            status_code=400,
        )

    body = sanitize_body(body)
    stream = bool(body.pop("stream", False))

    url = f"{DATABRICKS_HOST}{DATABRICKS_GATEWAY_PATH}"
    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    log.info("Forwarding model=%s stream=%s keys=%s", body.get("model"), stream, list(body.keys()))

    if stream:
        body["stream"] = True

        async def event_stream():
            def sse_error(msg: str) -> bytes:
                payload = {"type": "error", "error": {"type": "api_error", "message": msg}}
                return f"event: error\ndata: {json.dumps(payload)}\n\n".encode()

            try:
                for attempt in (0, 1):
                    async with _http.stream("POST", url, headers=headers, json=body) as r:
                        if r.status_code == 401 and attempt == 0:
                            await r.aread()
                            headers["Authorization"] = f"Bearer {get_token(force_refresh=True)}"
                            continue
                        if r.status_code >= 400:
                            err = (await r.aread()).decode("utf-8", "replace")
                            log.error("Upstream %d: %s", r.status_code, err)
                            yield sse_error(err)
                            return
                        async for chunk in r.aiter_raw():
                            if chunk:
                                yield chunk
                        return
            except httpx.HTTPError as e:
                log.exception("Upstream stream error")
                yield sse_error(str(e))

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        r = await _http.post(url, headers=headers, json=body)
        if r.status_code == 401:
            headers["Authorization"] = f"Bearer {get_token(force_refresh=True)}"
            r = await _http.post(url, headers=headers, json=body)
    except httpx.HTTPError as e:
        log.exception("Upstream error")
        return JSONResponse({"error": {"type": "api_error", "message": str(e)}}, status_code=502)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose Databricks AI Gateway as an Anthropic endpoint using Databricks CLI auth."
    )
    parser.add_argument("-p", "--profile", help="Databricks CLI profile name (default: first valid).")
    parser.add_argument("--host", default=LISTEN_HOST, help="Local bind address.")
    parser.add_argument("--port", type=int, default=LISTEN_PORT, help="Local port.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        select_profile(args.profile)
        get_token()
    except Exception as e:
        log.error("Startup failed: %s", e)
        raise
    uvicorn.run(app, host=args.host, port=args.port)
