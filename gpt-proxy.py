import os
import time
import json
import logging
from typing import Any, Optional
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import httpx
from databricks.sdk.core import Config

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("databricks_proxy")

# --- CẤU HÌNH DATABRICKS ---
DATABRICKS_HOST = "https://adb-3423969507737722.2.azuredatabricks.net"
DATABRICKS_PROFILE = os.environ.get("DATABRICKS_PROFILE", "adb-3423969507737722")
TOKEN_TTL_SECONDS = 3000

_cfg = Config(host=DATABRICKS_HOST, profile=DATABRICKS_PROFILE)
_token: Optional[str] = None
_token_fetched_at: float = 0.0

def get_token() -> str:
    global _token, _token_fetched_at
    now = time.time()
    if _token is None or (now - _token_fetched_at) > TOKEN_TTL_SECONDS:
        try:
            auth = _cfg.authenticate()
            _token = auth["Authorization"].split(" ", 1)[1]
            _token_fetched_at = now
            log.info("Refreshed Databricks token successfully.")
        except Exception as e:
            log.error(f"Failed to fetch Databricks token: {e}")
            raise e
    return _token

# --- FASTAPI APP ---
app = FastAPI(title="Databricks OpenAI Proxy for Zoo Code")
http_client = httpx.AsyncClient(timeout=120.0)


def _content_to_text(value: Any) -> Any:
    """Convert structured content blocks to the string clients expect."""
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, list):
        parts = [_content_to_text(item) for item in value]
        return "".join(part for part in parts if isinstance(part, str))
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        for key in ("summary", "content", "output_text", "value"):
            if key in value:
                text = _content_to_text(value[key])
                if isinstance(text, str):
                    return text
        return ""
    return str(value)


def _normalize_response(payload: Any) -> Any:
    """Normalize OpenAI message content without changing other response fields."""
    if isinstance(payload, dict):
        normalized = {key: _normalize_response(value) for key, value in payload.items()}
        for key in ("content", "output_text"):
            if key in normalized:
                normalized[key] = _content_to_text(normalized[key])
        return normalized
    if isinstance(payload, list):
        return [_normalize_response(item) for item in payload]
    return payload


def _normalize_sse_line(line: bytes) -> bytes:
    if not line.startswith(b"data:"):
        return line
    prefix, raw_data = line.split(b":", 1)
    data = raw_data.strip()
    if not data or data == b"[DONE]":
        return line
    try:
        payload = _normalize_response(json.loads(data))
        return prefix + b": " + json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except (json.JSONDecodeError, TypeError):
        return line


async def _normalized_stream(response: httpx.Response):
    async for line in response.aiter_lines():
        yield _normalize_sse_line(line.encode("utf-8")) + b"\n"

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_requests(path: str, request: Request):
    try:
        # 1. Chuẩn hóa path
        target_path = path
        if target_path.startswith("v1/"):
            target_path = target_path[3:]

        target_url = f"{DATABRICKS_HOST}/serving-endpoints/{target_path}"
        body = await request.body()

        # 2. Xử lý và làm sạch Body Payload
        if request.method == "POST" and len(body) > 0:
            try:
                body_data = json.loads(body)
                
                # Ép buộc dùng model agent_endpoint
                body_data["model"] = "alert_summary_agent"
                
                # --- XÓA CÁC THAM SỐ GÂY LỖI 400 ---
                body_data.pop("stream_options", None)
                body_data.pop("parallel_tool_calls", None) # <--- ĐÃ THÊM DÒNG NÀY ĐỂ FIX LỖI
                
                # (Dự phòng) Nếu Databricks vẫn báo lỗi unknown field "tools" hoặc "tool_choice"
                # thì bạn hãy bỏ comment 2 dòng dưới đây:
                # body_data.pop("tools", None)
                # body_data.pop("tool_choice", None)

                body = json.dumps(body_data).encode("utf-8")
                log.info(f"Forwarding payload keys: {list(body_data.keys())}")
            except json.JSONDecodeError:
                pass

        # 3. CHỈ SỬ DỤNG HEADERS CẦN THIẾT
        headers = {
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }

        req = http_client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params
        )

        log.info(f"Requesting Databricks: {target_url}")

        response = await http_client.send(req, stream=True)

        # 4. CHẶN VÀ ĐỌC LỖI RÕ RÀNG NẾU KHÁC 200 (NHƯ LỖI 400)
        if response.status_code != 200:
            await response.aread() 
            error_msg = response.text
            log.error(f"🔥 LỖI TỪ DATABRICKS [{response.status_code}]: {error_msg}")
            return Response(content=error_msg, status_code=response.status_code, media_type="application/json")

        # Normalize structured message content before Zoo Code renders it.
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            raw_response = await response.aread()
            try:
                normalized = _normalize_response(json.loads(raw_response))
                return Response(
                    content=json.dumps(normalized, ensure_ascii=False).encode("utf-8"),
                    status_code=response.status_code,
                    media_type="application/json",
                )
            except (json.JSONDecodeError, TypeError):
                return Response(
                    content=raw_response,
                    status_code=response.status_code,
                    media_type=content_type or "application/octet-stream",
                )

        # Stream normalized SSE data back to Zoo Code.
        return StreamingResponse(
            _normalized_stream(response),
            status_code=response.status_code,
            headers={
                k: v for k, v in response.headers.items() 
                if k.lower() not in ("content-length", "transfer-encoding", "content-encoding", "date", "server")
            },
            background=BackgroundTask(response.aclose)
        )

    except Exception as e:
        log.error(f"Proxy internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Khởi động Databricks Local Proxy tại: http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)