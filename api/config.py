import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _project_path(name: str) -> str:
    path = Path(name)
    return str(path if path.is_absolute() else ROOT_DIR / path)

# --- LLM / llama.cpp ---
LLAMA_SERVER_EXE: str = _project_path(
    os.getenv("LLAMA_SERVER_EXE", r"llama.cpp\build\bin\Release\llama-server.exe")
)
LLAMA_SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
MODEL_PATH: str = _project_path(os.path.join("Models", os.getenv("LLAMA_MODEL", "model.gguf")))
CTX_SIZE: int = int(os.getenv("LLAMA_CTX_SIZE", "4096"))
N_GPU_LAYERS: int = int(os.getenv("LLAMA_N_GPU_LAYERS", "999"))
THREADS: int = int(os.getenv("LLAMA_THREADS", "12"))
PORT: int = int(os.getenv("PORT", "8080"))

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Chroma ---
CHROMA_DIR: str = _project_path(os.getenv("CHROMA_DIR", "chroma_db"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "local_rag_docs")

# --- Ingestion ---
DO_INGEST: bool = os.getenv("DO_INGEST", "False").lower() in ("1", "true", "yes")
PDF_IN: str = _project_path(os.getenv("PDF_IN", "data_in/document.pdf"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval ---
TOP_K: int = int(os.getenv("TOP_K", "3"))
CANDIDATE_POOL: int = int(os.getenv("CANDIDATE_POOL", "25"))
MMR_DIVERSITY: float = float(os.getenv("MMR_DIVERSITY", "0.5"))

# --- Prompt / Contexte ---
MAX_CHARS_TOTAL: int = int(os.getenv("MAX_CHARS_TOTAL", "3000"))
PER_SNIPPET: int = int(os.getenv("PER_SNIPPET", "-1"))

# --- Génération ---
N_PREDICT: int = int(os.getenv("N_PREDICT", "1000"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P: float = float(os.getenv("TOP_P", "0.9"))
REPEAT_PENALTY: float = float(os.getenv("REPEAT_PENALTY", "1.15"))
STOP_TOKENS: list[str] = ["<|assistant|>", "<|user|>", "</s>", "Instruction", "Solution", "[REPONSE]", "[REPONDRE]"]

WEB_DIR: str = _project_path(os.getenv("WEB_DIR", "web"))
