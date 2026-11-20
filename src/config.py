from langchain_groq import ChatGroq
from src.settings import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    get_vectorstore,
)

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

# Initialize The LLM
llm = ChatGroq(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=GROQ_API_KEY,
)

_retriever = None


def get_retriever():
    """Create the Supabase retriever once and reuse it."""
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K})
    return _retriever


def reload_retriever():
    """Call after rebuilding the vectorstore."""
    global _retriever
    _retriever = None
    return get_retriever()
