#!/home/huynhlamtran/projects/claude_wsl/venv/bin/python3
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent, Tool
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import requests
import uuid
import os

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "local_python_codebase"

# Initialize the MCP server and connect to Qdrant
mcp = MCPServer("Local Codebase RAG")
qdrant = QdrantClient(url=QDRANT_URL)

# Create the Qdrant collection if it does not already exist
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

def get_embedding(text: str) -> list[float]:
    """Call the Ollama API to convert text into a vector."""
    response = requests.post(OLLAMA_API_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": text
    })
    response.raise_for_status()
    return response.json()["embedding"]

# ==========================================
# DECLARE TOOLS FOR CLAUDE CODE
# ==========================================

@mcp.tool()
def index_python_file(filepath: str) -> str:
    """
    Read, split with LangChain, and index a Python file in Qdrant.
    Call this tool when you need to add code file knowledge to the database.
    """
    if not os.path.exists(filepath):
        return f"❌ Error: File not found at path: {filepath}"

    try:
        # Read the file contents
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Initialize LangChain's Python-aware text splitter
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=200
        )

        # Split the source code into chunks
        chunks = python_splitter.split_text(content)

        # Package each chunk as a vector point and load it into Qdrant
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

        # Upsert into the database
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return f"✅ Successfully indexed file: {filepath}\nDetails: Split into {len(chunks)} chunks while preserving Python structure."
    except Exception as e:
        return f"❌ Error indexing file {filepath}: {str(e)}"

@mcp.tool()
def semantic_search(query: str, limit: int = 5) -> str:
    """
    Perform semantic search across the codebase.
    Example queries: 'login authentication handler' or 'tax calculation logic'.
    """
    try:
        # Embed the query as a vector
        query_vector = get_embedding(query)

        # Find the closest matching vectors in Qdrant
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )

        if not hits:
            return "No related code was found in the database."

        # Prepare the result text for Claude Code
        result_text = f"🔍 Found {len(hits)} code chunks related to '{query}':\n\n"
        for hit in hits:
            filepath = hit.payload.get("filepath", "Unknown")
            chunk_index = hit.payload.get("chunk_index", "?")
            total_chunks = hit.payload.get("total_chunks", "?")
            content = hit.payload.get("content", "")
            score = round(hit.score, 3)

            result_text += f"--- 📄 File: {filepath} (Chunk {chunk_index}/{total_chunks}) | 🎯 Similarity: {score} ---\n"
            result_text += f"```python\n{content}\n```\n\n"

        return result_text
    except Exception as e:
        return f"❌ Search error: {str(e)}"

if __name__ == "__main__":
    mcp.run()
