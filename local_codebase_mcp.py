#!/home/huynhlamtran/projects/claude_wsl/venv/bin/python3
from mcp.server.mcpserver import MCPServer
import os
import sys

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

# The server only does real work when Claude Code was started from one of these entrypoints.
# "claude-vscode" is the VS Code extension; a terminal session reports "cli" / "sdk-cli".
# Override with MCP_ALLOWED_ENTRYPOINTS="claude-vscode,cli" (or "*" for every client).
ENTRYPOINT = os.environ.get("CLAUDE_CODE_ENTRYPOINT", "")
ALLOWED_ENTRYPOINTS = {
    e.strip()
    for e in os.environ.get("MCP_ALLOWED_ENTRYPOINTS", "claude-vscode").split(",")
    if e.strip()
}


def _idle_reason() -> str:
    """Why this server should do nothing (empty string = run normally)."""
    if "*" not in ALLOWED_ENTRYPOINTS and ENTRYPOINT not in ALLOWED_ENTRYPOINTS:
        return (
            f"launched from '{ENTRYPOINT or 'unknown'}', only {sorted(ALLOWED_ENTRYPOINTS)} "
            "enabled (set MCP_ALLOWED_ENTRYPOINTS to change)"
        )
    if not os.path.isdir(WORKSPACE_DIR):
        return f"workspace directory not found: {WORKSPACE_DIR}"
    # Indexing a whole home/root directory would embed thousands of unrelated files.
    too_broad = {os.path.realpath(os.sep), os.path.realpath(os.path.expanduser("~"))}
    if os.path.realpath(WORKSPACE_DIR) in too_broad:
        return f"refusing to index {WORKSPACE_DIR} (home or root directory)"
    return ""


# Idle mode: still answer the MCP handshake (so the client sees a healthy server with no tools),
# but skip the heavy imports, Qdrant/Ollama calls, the watcher and the startup index.
if __name__ == "__main__" and _idle_reason():
    print(f"[local-codebase] Idle: {_idle_reason()}", file=sys.stderr, flush=True)
    MCPServer("Local Codebase RAG").run()
    raise SystemExit(0)

from qdrant_client import QdrantClient  # noqa: E402  (deliberately after the idle check)
from qdrant_client.models import (  # noqa: E402
    PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, PayloadSchemaType,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language  # noqa: E402
from watchdog.observers import Observer  # noqa: E402
from watchdog.events import FileSystemEventHandler  # noqa: E402
import fnmatch  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import requests  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_FILE_BYTES = 256 * 1024
DEBOUNCE_SECONDS = 1.5
BACKEND_RETRIES = 20       # Qdrant may still be starting when VS Code opens (e.g. after a reboot)
BACKEND_RETRY_SECONDS = 15

# Bump the leading number whenever chunking or the language/extension maps change: it is mixed
# into every file hash, so all files get re-embedded once instead of being skipped as unchanged.
INDEX_VERSION = f"2:{OLLAMA_MODEL}:{CHUNK_SIZE}:{CHUNK_OVERLAP}"

IGNORED_DIR_NAMES = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "site-packages",
    "dist", "build", "target", ".next", ".cache", ".idea", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

# Extensions with a language-aware splitter (split on classes/functions/headings first).
LANGUAGE_BY_EXT = {
    ".py": Language.PYTHON, ".pyi": Language.PYTHON,
    ".js": Language.JS, ".jsx": Language.JS, ".mjs": Language.JS, ".cjs": Language.JS,
    ".ts": Language.TS, ".tsx": Language.TS, ".mts": Language.TS, ".cts": Language.TS,
    ".go": Language.GO, ".rs": Language.RUST, ".java": Language.JAVA,
    ".kt": Language.KOTLIN, ".kts": Language.KOTLIN, ".scala": Language.SCALA,
    ".c": Language.C, ".h": Language.CPP, ".cc": Language.CPP, ".cpp": Language.CPP,
    ".cxx": Language.CPP, ".hpp": Language.CPP, ".cs": Language.CSHARP,
    ".rb": Language.RUBY, ".php": Language.PHP, ".swift": Language.SWIFT,
    ".lua": Language.LUA, ".pl": Language.PERL, ".pm": Language.PERL, ".hs": Language.HASKELL,
    ".ex": Language.ELIXIR, ".exs": Language.ELIXIR, ".r": Language.R,
    ".ps1": Language.POWERSHELL, ".sol": Language.SOL, ".proto": Language.PROTO,
    ".md": Language.MARKDOWN, ".mdx": Language.MARKDOWN, ".rst": Language.RST,
    ".tex": Language.LATEX, ".html": Language.HTML, ".htm": Language.HTML,
}
# Text formats indexed with the generic splitter.
PLAIN_EXTS = {
    ".sh", ".bash", ".zsh", ".sql", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".css", ".scss", ".less", ".vue", ".svelte", ".txt", ".gradle", ".tf",
    ".graphql", ".gql", ".xml",
}
PLAIN_NAMES = {"dockerfile", "makefile", "jenkinsfile", "rakefile", "gemfile", "procfile"}
# Never index: lockfiles, generated bundles, and anything that looks like a credential.
SKIP_NAME_PATTERNS = (
    "*.lock", "package-lock.json", "pnpm-lock.yaml", "*.min.js", "*.min.css", "*.map",
    ".env", ".env.*", "*.env", "*.pem", "*.key", "*.pfx", "*.p12", "id_rsa*", "id_ed25519*",
    "credentials*", "secrets*", "*.tfstate", "*.tfvars",
)


def _collection_name_for(workspace_dir: str) -> str:
    """Mỗi workspace có collection riêng để kết quả tìm kiếm không lẫn giữa các project."""
    slug = os.path.basename(workspace_dir.rstrip(os.sep)) or "root"
    slug = "".join(c if c.isalnum() else "_" for c in slug)
    digest = hashlib.sha1(workspace_dir.encode("utf-8")).hexdigest()[:8]
    return f"codebase_{slug}_{digest}"


COLLECTION_NAME = _collection_name_for(WORKSPACE_DIR)

# Khởi tạo MCP Server (kết nối Qdrant được tạo lười ở get_qdrant)
mcp = MCPServer("Local Codebase RAG")

_qdrant: QdrantClient | None = None
_qdrant_lock = threading.Lock()
_index_lock = threading.Lock()           # serialises indexing so hashes and Qdrant stay consistent
_indexed_hashes: dict[str, str] = {}     # filepath -> hash of the content currently in Qdrant
_initial_index_done = threading.Event()


def get_qdrant() -> QdrantClient:
    """Connect once; create the collection on first use and load the per-file hashes."""
    global _qdrant
    with _qdrant_lock:
        if _qdrant is None:
            client = QdrantClient(url=QDRANT_URL)
            # Tạo Collection trên Qdrant nếu chưa tồn tại (riêng cho workspace này)
            if not client.collection_exists(COLLECTION_NAME):
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
            # Keeps per-file deletes fast on big collections (no-op if it already exists).
            client.create_payload_index(COLLECTION_NAME, "filepath", PayloadSchemaType.KEYWORD)
            _indexed_hashes.update(_load_indexed_hashes(client))
            _qdrant = client
        return _qdrant


def _load_indexed_hashes(client: QdrantClient) -> dict[str, str]:
    """Read filepath -> file_hash for everything already in the collection (payload only)."""
    hashes: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME, limit=1000, offset=offset,
            with_payload=["filepath", "file_hash"], with_vectors=False,
        )
        for point in points:
            hashes[point.payload["filepath"]] = point.payload.get("file_hash", "")
        if offset is None:
            return hashes


def get_embedding(text: str) -> list[float]:
    """Gọi API của Ollama để biến đoạn text thành Vector"""
    response = requests.post(OLLAMA_API_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": text
    }, timeout=120)
    response.raise_for_status()
    return response.json()["embedding"]

# ==========================================
# NHẬN DIỆN FILE CẦN INDEX
# ==========================================

def _is_indexable_path(path: str) -> bool:
    """Path-only checks (no disk access): not in an ignored directory, right type, not a secret."""
    parts = os.path.normpath(path).split(os.sep)
    if any(part in IGNORED_DIR_NAMES for part in parts):
        return False
    name = parts[-1].lower()
    if any(fnmatch.fnmatch(name, pattern) for pattern in SKIP_NAME_PATTERNS):
        return False
    return (
        os.path.splitext(name)[1] in LANGUAGE_BY_EXT
        or os.path.splitext(name)[1] in PLAIN_EXTS
        or name in PLAIN_NAMES
    )


def _is_candidate(path: str) -> bool:
    return _is_indexable_path(path) and os.path.isfile(path) and not os.path.islink(path)


def _is_in_workspace(path: str) -> bool:
    return os.path.commonpath([WORKSPACE_DIR, path]) == WORKSPACE_DIR


def _is_git_ignored(path: str) -> bool:
    """True if .gitignore excludes the path (False outside a git repo)."""
    try:
        return subprocess.run(
            ["git", "-C", WORKSPACE_DIR, "check-ignore", "-q", "--", path],
            capture_output=True, timeout=10,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _list_workspace_files() -> list[str]:
    """Files worth indexing. In a git repo this honours .gitignore; otherwise walk the tree."""
    try:
        out = subprocess.run(
            ["git", "-C", WORKSPACE_DIR, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True, timeout=60,
        ).stdout
        files = [os.path.join(WORKSPACE_DIR, p) for p in out.decode("utf-8", "replace").split("\0") if p]
    except (OSError, subprocess.SubprocessError):
        files = []
        for root, dirs, names in os.walk(WORKSPACE_DIR):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIR_NAMES]
            files.extend(os.path.join(root, name) for name in names)
    return [f for f in files if _is_candidate(f)]


def _read_text(filepath: str) -> tuple[str, str] | None:
    """Return (text, content hash), or None for files that are too large or not UTF-8 text."""
    if os.path.getsize(filepath) > MAX_FILE_BYTES:
        return None
    with open(filepath, "rb") as f:
        data = f.read()
    if b"\0" in data[:8192]:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    digest = hashlib.sha256(INDEX_VERSION.encode("utf-8") + b"\0" + data).hexdigest()
    return text, digest


@functools.lru_cache(maxsize=None)
def _splitter_for(ext: str) -> RecursiveCharacterTextSplitter:
    """Language-aware splitter when we know the language, generic text splitter otherwise."""
    language = LANGUAGE_BY_EXT.get(ext)
    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ==========================================
# INDEX / XÓA FILE TRONG QDRANT
# ==========================================

def _delete_file_points(client: QdrantClient, filepath: str) -> None:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="filepath", match=MatchValue(value=filepath))]
        ),
    )


def _remove_file(filepath: str) -> str:
    """Drop a file's chunks from the index. Returns "" if it was never indexed."""
    with _index_lock:
        try:
            client = get_qdrant()
            if filepath not in _indexed_hashes:
                return ""
            _delete_file_points(client, filepath)
            _indexed_hashes.pop(filepath, None)
            return f"Removed from index: {filepath}"
        except Exception as e:
            return f"Error removing {filepath} from index: {str(e)}"


def _index_file(filepath: str) -> str:
    """Đọc, cắt nhỏ bằng LangChain và lập chỉ mục (index) một file vào Qdrant, bỏ qua nếu không đổi."""
    filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        return f"Error: File not found at path: {filepath}"
    if not _is_in_workspace(filepath):
        return f"Skipped (outside the workspace {WORKSPACE_DIR}): {filepath}"
    if not _is_candidate(filepath):
        return f"Skipped (unsupported type, ignored directory, secret-looking name or symlink): {filepath}"

    with _index_lock:
        try:
            client = get_qdrant()
            loaded = _read_text(filepath)
            if loaded is None:
                return f"Skipped (binary or larger than {MAX_FILE_BYTES // 1024} KB): {filepath}"
            content, digest = loaded

            # Nội dung không đổi kể từ lần index trước -> không cần nhúng lại
            if _indexed_hashes.get(filepath) == digest:
                return f"Unchanged, already indexed: {filepath}"

            ext = os.path.splitext(filepath)[1].lower()
            chunks = _splitter_for(ext).split_text(content)
            if not chunks:
                _delete_file_points(client, filepath)
                _indexed_hashes.pop(filepath, None)
                return f"Skipped (empty file): {filepath}"

            # Embed everything first: if Ollama fails midway the previous index of the file survives.
            vectors = [get_embedding(chunk) for chunk in chunks]
            language = ext.lstrip(".") or os.path.basename(filepath).lower()

            # Xóa các điểm cũ của file này trước khi nạp lại (tránh trùng lặp khi re-index)
            _delete_file_points(client, filepath)
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "filepath": filepath,
                            "chunk_index": i + 1,
                            "total_chunks": len(chunks),
                            "content": chunk,
                            "language": language,
                            "file_hash": digest,
                        },
                    )
                    for i, (chunk, vector) in enumerate(zip(chunks, vectors))
                ],
            )
            _indexed_hashes[filepath] = digest

            return f"Successfully indexed file: {filepath}\nDetails: Split into {len(chunks)} chunks."
        except Exception as e:
            return f"Error indexing file {filepath}: {str(e)}"

# ==========================================
# KHAI BÁO CÁC CÔNG CỤ (TOOLS) CHO CLAUDE CODE
# ==========================================

@mcp.tool()
def index_file(filepath: str) -> str:
    """
    Index (or re-index) one workspace file into the search database.
    Source code, docs and config files are indexed automatically at startup and whenever they are
    saved, so this is only needed to force a refresh. Unchanged files are skipped.
    """
    return _index_file(filepath)

@mcp.tool()
def semantic_search(query: str, limit: int = 5) -> str:
    """
    Search the workspace codebase by meaning (source code, docs and config files).
    Use this tool before answering questions about code locations, implementation details,
    or behavior when the relevant files are not already known. Pass the user's request as
    the query, for example: 'login authentication' or 'tax calculation logic'.
    """
    try:
        client = get_qdrant()

        # Nhúng câu hỏi thành vector
        query_vector = get_embedding(query)

        # Dò tìm các vector khớp nhất trong Qdrant
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        )
        hits = result.points

        note = ""
        if not _initial_index_done.is_set():
            note = "(Note: the startup index is still running, so results may be incomplete.)\n\n"

        if not hits:
            return note + "No relevant code was found in the database."

        # Chuẩn bị văn bản trả về cho Claude Code đọc
        result_text = note + f"Found {len(hits)} relevant code chunks for '{query}':\n\n"
        for hit in hits:
            filepath = hit.payload.get("filepath", "Unknown")
            chunk_index = hit.payload.get("chunk_index", "?")
            total_chunks = hit.payload.get("total_chunks", "?")
            content = hit.payload.get("content", "")
            language = hit.payload.get("language", "")
            score = round(hit.score, 3)

            fence = "````" if "```" in content else "```"
            result_text += f"--- File: {filepath} (Chunk {chunk_index}/{total_chunks}) | Relevance: {score} ---\n"
            result_text += f"{fence}{language}\n{content}\n{fence}\n\n"

        return result_text
    except Exception as e:
        return f"Error while performing search: {str(e)}"

# ==========================================
# TỰ ĐỘNG INDEX KHI FILE ĐƯỢC LƯU (WATCHDOG)
# ==========================================

class WorkspaceSaveHandler(FileSystemEventHandler):
    """Lắng nghe sự kiện lưu/xóa file và đồng bộ lại index, có debounce."""

    def __init__(self):
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _schedule_sync(self, filepath: str):
        if not _is_indexable_path(filepath):
            return

        with self._lock:
            existing = self._timers.get(filepath)
            if existing:
                existing.cancel()

            timer = threading.Timer(DEBOUNCE_SECONDS, self._run_sync, args=(filepath,))
            timer.daemon = True
            self._timers[filepath] = timer
            timer.start()

    def _run_sync(self, filepath: str):
        with self._lock:
            self._timers.pop(filepath, None)
        if os.path.exists(filepath):
            if _is_git_ignored(filepath):
                return
            result = _index_file(filepath)
        else:
            result = _remove_file(filepath)  # deleted, or the old side of a rename
        if result and not result.startswith(("Unchanged", "Skipped")):
            print(f"[watchdog] {result}", file=sys.stderr, flush=True)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_sync(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_sync(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._schedule_sync(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._schedule_sync(event.src_path)
            self._schedule_sync(event.dest_path)


def bulk_index_workspace():
    """Quét WORKSPACE_DIR khi server khởi động: index file mới/đã đổi, bỏ qua file không đổi, dọn file đã xóa."""
    for _ in range(BACKEND_RETRIES):
        try:
            get_qdrant()
            break
        except Exception as e:
            print(f"[startup-index] Qdrant not ready ({e}); retrying in {BACKEND_RETRY_SECONDS}s", file=sys.stderr, flush=True)
            time.sleep(BACKEND_RETRY_SECONDS)
    else:
        print("[startup-index] Giving up: Qdrant is unreachable. Searches will report an error until it is up.", file=sys.stderr, flush=True)
        return

    files = _list_workspace_files()
    counts = {"indexed": 0, "unchanged": 0, "skipped": 0, "failed": 0}
    for filepath in files:
        result = _index_file(filepath)
        if result.startswith("Successfully"):
            counts["indexed"] += 1
        elif result.startswith("Unchanged"):
            counts["unchanged"] += 1
        elif result.startswith("Skipped"):
            counts["skipped"] += 1
        else:
            counts["failed"] += 1
            print(f"[startup-index] {result}", file=sys.stderr, flush=True)

    # Files that were deleted (or no longer qualify) since the last run. An empty listing is
    # treated as a failed scan rather than "everything was deleted".
    removed = 0
    if files:
        current = set(files)
        with _index_lock:
            stale = [p for p in _indexed_hashes if p not in current]
        removed = sum(1 for p in stale if _remove_file(p).startswith("Removed"))

    _initial_index_done.set()
    print(
        f"[startup-index] Done in {WORKSPACE_DIR}: {counts['indexed']} indexed, "
        f"{counts['unchanged']} unchanged, {counts['skipped']} skipped, "
        f"{counts['failed']} failed, {removed} removed",
        file=sys.stderr, flush=True,
    )


def start_watcher():
    """Khởi động Observer theo dõi WORKSPACE_DIR (workspace hiện tại) trong một thread nền."""
    handler = WorkspaceSaveHandler()
    observer = Observer()
    observer.schedule(handler, WORKSPACE_DIR, recursive=True)
    observer.start()
    print(f"[watchdog] Workspace: {WORKSPACE_DIR} (collection: {COLLECTION_NAME})", file=sys.stderr)


if __name__ == "__main__":
    start_watcher()
    threading.Thread(target=bulk_index_workspace, daemon=True).start()
    mcp.run()
