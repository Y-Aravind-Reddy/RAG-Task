"""Central settings loaded from .env (no API key needed to import this)."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "documents")
SUPABASE_QUERY = os.getenv("SUPABASE_QUERY", "match_documents")

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


@lru_cache(maxsize=1)
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file.")
    from supabase import create_client

    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_vectorstore():
    from langchain_community.vectorstores import SupabaseVectorStore

    return SupabaseVectorStore(
        client=get_supabase_client(),
        embedding=get_embeddings(),
        table_name=SUPABASE_TABLE,
        query_name=SUPABASE_QUERY,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
