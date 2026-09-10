#!/home/huynhlamtran/projects/claude_wsl/venv/bin/python3
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent, Tool
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
import uuid
import os
import sys
import hashlib
import threading
import time

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"

# Thư mục workspace hiện tại: Claude Code / VS Code khởi chạy MCP server với cwd
# đặt tại thư mục workspace đang mở, nên mỗi workspace tự có server + theo dõi riêng.
WORKSPACE_DIR = os.path.abspath(
    os.environ.get("MCP_WATCH_DIR") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
)
IGNORED_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules"}
DEBOUNCE_SECONDS = 1.5


def _collection_name_for(workspace_dir: str) -> str:
    """Mỗi workspace có collection riêng để kết quả tìm kiếm không lẫn giữa các project."""
    slug = os.path.basename(workspace_dir.rstrip(os.sep)) or "root"
    slug = "".join(c if c.isalnum() else "_" for c in slug)
    digest = hashlib.sha1(workspace_dir.encode("utf-8")).hexdigest()[:8]
    return f"codebase_{slug}_{digest}"


COLLECTION_NAME = _collection_name_for(WORKSPACE_DIR)

# Khởi tạo MCP Server và kết nối Qdrant
mcp = MCPServer("Local Codebase RAG")
qdrant = QdrantClient(url=QDRANT_URL)

# Tạo Collection trên Qdrant nếu chưa tồn tại (riêng cho workspace này)
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

def get_embedding(text: str) -> list[float]:
    """Gọi API của Ollama để biến đoạn text thành Vector"""
    response = requests.post(OLLAMA_API_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": text
    })
    response.raise_for_status()
    return response.json()["embedding"]

# ==========================================
# KHAI BÁO CÁC CÔNG CỤ (TOOLS) CHO CLAUDE CODE
# ==========================================

def _index_file(filepath: str) -> str:
    """Đọc, cắt nhỏ bằng LangChain và lập chỉ mục (index) file Python vào Qdrant."""
    if not os.path.exists(filepath):
        return f"Error: File not found at path: {filepath}"

    try:
        # Đọc nội dung file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Khởi tạo bộ cắt file thông minh của LangChain dành riêng cho Python
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=200
        )

        # Thực hiện băm nhỏ mã nguồn
        chunks = python_splitter.split_text(content)

        # Xóa các điểm cũ của file này trước khi nạp lại (tránh trùng lặp khi re-index)
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="filepath", match=MatchValue(value=filepath))]
            ),
        )

        # Đóng gói từng chunk thành Vector Point và nạp vào Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            vector = get_embedding(chunk)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "filepath": filepath,
                        "chunk_index": i + 1,
                        "total_chunks": len(chunks),
                        "content": chunk
                    }
                )
            )

        # Upsert vào Database
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return f"Successfully indexed file: {filepath}\nDetails: Split into {len(chunks)} chunks while preserving Python structure."
    except Exception as e:
        return f"Error indexing file {filepath}: {str(e)}"


@mcp.tool()
def index_python_file(filepath: str) -> str:
    """
    Công cụ để đọc, cắt nhỏ bằng LangChain và lập chỉ mục (index) file Python vào Qdrant.
    Hãy gọi công cụ này khi cần đưa thêm kiến thức từ một file code vào Database.
    """
    return _index_file(filepath)

@mcp.tool()
def semantic_search(query: str, limit: int = 5) -> str:
    """
    Search the workspace codebase by meaning.
    Use this tool before answering questions about code locations, implementation details,
    or behavior when the relevant files are not already known. Pass the user's request as
    the query, for example: 'login authentication' or 'tax calculation logic'.
    """
    try:
        # Nhúng câu hỏi thành vector
        query_vector = get_embedding(query)

        # Dò tìm các vector khớp nhất trong Qdrant
        result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        )
        hits = result.points

        if not hits:
            return "No relevant code was found in the database."

        # Chuẩn bị văn bản trả về cho Claude Code đọc
        result_text = f"Found {len(hits)} relevant code chunks for '{query}':\n\n"
        for hit in hits:
            filepath = hit.payload.get("filepath", "Unknown")
            chunk_index = hit.payload.get("chunk_index", "?")
            total_chunks = hit.payload.get("total_chunks", "?")
            content = hit.payload.get("content", "")
            score = round(hit.score, 3)

            result_text += f"--- File: {filepath} (Chunk {chunk_index}/{total_chunks}) | Relevance: {score} ---\n"
            result_text += f"```python\n{content}\n```\n\n"

        return result_text
    except Exception as e:
        return f"Error while performing search: {str(e)}"

# ==========================================
# TỰ ĐỘNG INDEX KHI FILE ĐƯỢC LƯU (WATCHDOG)
# ==========================================

class PythonSaveHandler(FileSystemEventHandler):
    """Lắng nghe sự kiện lưu file .py và tự động index lại, có debounce."""

    def __init__(self):
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _is_ignored(self, path: str) -> bool:
        parts = os.path.normpath(path).split(os.sep)
        return any(part in IGNORED_DIR_NAMES for part in parts)

    def _schedule_index(self, filepath: str):
        if not filepath.endswith(".py") or self._is_ignored(filepath):
            return

        with self._lock:
            existing = self._timers.get(filepath)
            if existing:
                existing.cancel()

            timer = threading.Timer(DEBOUNCE_SECONDS, self._run_index, args=(filepath,))
            timer.daemon = True
            self._timers[filepath] = timer
            timer.start()

    def _run_index(self, filepath: str):
        with self._lock:
            self._timers.pop(filepath, None)
        if os.path.exists(filepath):
            result = _index_file(filepath)
            print(f"[watchdog] {result}", file=sys.stderr, flush=True)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_index(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_index(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._schedule_index(event.dest_path)


def bulk_index_workspace():
    """Quét toàn bộ WORKSPACE_DIR khi server khởi động và index các file .py hiện có."""
    indexed = 0
    for root, dirs, files in os.walk(WORKSPACE_DIR):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIR_NAMES]
        for name in files:
            if name.endswith(".py"):
                filepath = os.path.join(root, name)
                result = _index_file(filepath)
                if result.startswith("Successfully"):
                    indexed += 1
                else:
                    print(f"[startup-index] {result}", file=sys.stderr, flush=True)
    print(f"[startup-index] Successfully indexed {indexed} .py files in workspace: {WORKSPACE_DIR}", file=sys.stderr, flush=True)


def start_watcher():
    """Khởi động Observer theo dõi WORKSPACE_DIR (workspace hiện tại) trong một thread nền."""
    if not os.path.isdir(WORKSPACE_DIR):
        print(f"[watchdog] Skipping watch: directory not found: {WORKSPACE_DIR}", file=sys.stderr)
        return

    handler = PythonSaveHandler()
    observer = Observer()
    observer.schedule(handler, WORKSPACE_DIR, recursive=True)
    observer.start()
    print(f"[watchdog] Workspace: {WORKSPACE_DIR} (collection: {COLLECTION_NAME})", file=sys.stderr)


if __name__ == "__main__":
    start_watcher()
    threading.Thread(target=bulk_index_workspace, daemon=True).start()
    mcp.run()
