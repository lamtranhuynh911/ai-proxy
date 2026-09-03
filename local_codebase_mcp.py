#!/home/huynhlamtran/projects/claude_wsl/venv/bin/python3 #[cite: 1]
from mcp.server.mcpserver import MCPServer #[cite: 1]
from mcp.types import TextContent, Tool #[cite: 1]
from qdrant_client import QdrantClient #[cite: 1]
from qdrant_client.models import PointStruct, VectorParams, Distance #[cite: 1]
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language #[cite: 1]
import requests #[cite: 1]
import uuid #[cite: 1]
import os #[cite: 1]
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/embeddings" #[cite: 1]
OLLAMA_MODEL = "nomic-embed-text:latest" #[cite: 1]
QDRANT_URL = "http://localhost:6333" #[cite: 1]
COLLECTION_NAME = "local_python_codebase" #[cite: 1]
WATCH_DIRECTORY = "."  # Root directory for watchdog monitoring (defaults to the current directory)

# Initialize the MCP server and connect to Qdrant[cite: 1]
mcp = MCPServer("Local Codebase RAG") #[cite: 1]
qdrant = QdrantClient(url=QDRANT_URL) #[cite: 1]

# Create the Qdrant collection if it does not already exist[cite: 1]
if not qdrant.collection_exists(COLLECTION_NAME): #[cite: 1]
    qdrant.create_collection( #[cite: 1]
        collection_name=COLLECTION_NAME, #[cite: 1]
        vectors_config=VectorParams(size=768, distance=Distance.COSINE), #[cite: 1]
    ) #[cite: 1]

def get_embedding(text: str) -> list[float]: #[cite: 1]
    """Call the Ollama API to convert text into a vector.""" #[cite: 1]
    response = requests.post(OLLAMA_API_URL, json={ #[cite: 1]
        "model": OLLAMA_MODEL, #[cite: 1]
        "prompt": text #[cite: 1]
    }) #[cite: 1]
    response.raise_for_status() #[cite: 1]
    return response.json()["embedding"] #[cite: 1]

# ==========================================
# DECLARE TOOLS FOR CLAUDE CODE
# ==========================================

@mcp.tool() #[cite: 1]
def index_python_file(filepath: str) -> str: #[cite: 1]
    """
    Read, split with LangChain, and index a Python file in Qdrant.[cite: 1]
    Call this tool when you need to add code file knowledge to the database.[cite: 1]
    """
    if not os.path.exists(filepath): #[cite: 1]
        return f"❌ Error: File not found at path: {filepath}" #[cite: 1]

    try:
        # Read the file contents[cite: 1]
        with open(filepath, 'r', encoding='utf-8') as f: #[cite: 1]
            content = f.read() #[cite: 1]

        # Initialize LangChain's Python-aware text splitter[cite: 1]
        python_splitter = RecursiveCharacterTextSplitter.from_language( #[cite: 1]
            language=Language.PYTHON, #[cite: 1]
            chunk_size=1000, #[cite: 1]
            chunk_overlap=200 #[cite: 1]
        ) #[cite: 1]

        # Split the source code into chunks[cite: 1]
        chunks = python_splitter.split_text(content) #[cite: 1]

        # Package each chunk as a vector point and load it into Qdrant[cite: 1]
        points = [] #[cite: 1]
        for i, chunk in enumerate(chunks): #[cite: 1]
            vector = get_embedding(chunk) #[cite: 1]
            points.append( #[cite: 1]
                PointStruct( #[cite: 1]
                    id=str(uuid.uuid4()), #[cite: 1]
                    vector=vector, #[cite: 1]
                    payload={ #[cite: 1]
                        "filepath": filepath, #[cite: 1]
                        "chunk_index": i + 1, #[cite: 1]
                        "total_chunks": len(chunks), #[cite: 1]
                        "content": chunk #[cite: 1]
                    } #[cite: 1]
                ) #[cite: 1]
            ) #[cite: 1]

        # Upsert into the database[cite: 1]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points) #[cite: 1]

        return f"✅ Successfully indexed file: {filepath}\nDetails: Split into {len(chunks)} chunks while preserving Python structure." #[cite: 1]
    except Exception as e: #[cite: 1]
        return f"❌ Error indexing file {filepath}: {str(e)}" #[cite: 1]

@mcp.tool() #[cite: 1]
def semantic_search(query: str, limit: int = 5) -> str: #[cite: 1]
    """
    Perform semantic search across the codebase.[cite: 1]
    Example queries: 'login authentication handler' or 'tax calculation logic'.[cite: 1]
    """
    try:
        # Embed the query as a vector[cite: 1]
        query_vector = get_embedding(query) #[cite: 1]

        # Find the closest matching vectors in Qdrant[cite: 1]
        hits = qdrant.search( #[cite: 1]
            collection_name=COLLECTION_NAME, #[cite: 1]
            query_vector=query_vector, #[cite: 1]
            limit=limit #[cite: 1]
        ) #[cite: 1]

        if not hits: #[cite: 1]
            return "No related code was found in the database." #[cite: 1]

        # Prepare the result text for Claude Code[cite: 1]
        result_text = f"🔍 Found {len(hits)} code chunks related to '{query}':\n\n" #[cite: 1]
        for hit in hits: #[cite: 1]
            filepath = hit.payload.get("filepath", "Unknown") #[cite: 1]
            chunk_index = hit.payload.get("chunk_index", "?") #[cite: 1]
            total_chunks = hit.payload.get("total_chunks", "?") #[cite: 1]
            content = hit.payload.get("content", "") #[cite: 1]
            score = round(hit.score, 3) #[cite: 1]

            result_text += f"--- 📄 File: {filepath} (Chunk {chunk_index}/{total_chunks}) | 🎯 Similarity: {score} ---\n" #[cite: 1]
            result_text += f"```python\n{content}\n```\n\n" #[cite: 1]

        return result_text #[cite: 1]
    except Exception as e: #[cite: 1]
        return f"❌ Search error: {str(e)}" #[cite: 1]

# ==========================================
# WATCHDOG AUTO-INDEXING SYSTEM
# ==========================================

class CodebaseEventHandler(FileSystemEventHandler):
    """Listen for file system changes and automatically call the MCP tool."""
    
    def on_modified(self, event):
        # Monitor only .py files and ignore changes to this MCP file
        if not event.is_directory and event.src_path.endswith(".py"):
            if os.path.basename(event.src_path) != "local_codebase_mcp.py":
                print(f"[Watchdog] 🔄 File change detected; automatically indexing: {event.src_path}")
                index_python_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".py"):
            if os.path.basename(event.src_path) != "local_codebase_mcp.py":
                print(f"[Watchdog] ✨ New file detected; automatically indexing: {event.src_path}")
                index_python_file(event.src_path)

def start_watchdog(path):
    """Start the file-watching service in a separate thread."""
    event_handler = CodebaseEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"👀 Watchdog RAG auto-indexer is running in the background; monitoring: {os.path.abspath(path)}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__": #[cite: 1]
    # Run Watchdog in a background thread so it does not block mcp.run()
    watch_thread = threading.Thread(target=start_watchdog, args=(WATCH_DIRECTORY,), daemon=True)
    watch_thread.start()
    
    # Start the Claude Code server
    mcp.run() #[cite: 1]