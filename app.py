"""Streamlit frontend for the Conversational RAG chatbot.

Run with:  streamlit run app.py
"""
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.settings import DATA_DIR, SUPPORTED_EXTENSIONS

st.set_page_config(page_title="Policy Assistant", page_icon="📄", layout="wide")

HISTORY_WINDOW = 10  # messages sent to the LLM as chat history
INTENT_BADGE = {"DOCUMENT": "📄 Document", "HISTORY": "🕘 History", "GENERAL": "💬 General", "OUT_OF_SCOPE": "🚫 Out of scope"}
EXAMPLES = [
    "How many sick leaves do I get?",
    "Can earned leave be carried forward?",
    "How soon must a data breach be reported?",
    "How long is job applicant data kept?",
]


# ---------- Backend loading ----------
@st.cache_resource(show_spinner="Loading models and vector index...")
def load_backend():
    from src import config, rag_pipeline

    config.get_retriever()  # fail early if the index is missing
    return config, rag_pipeline


def rebuild_index():
    from src.ingestion import build_vectorstore

    num_files, num_chunks = build_vectorstore()
    config, _ = load_backend()
    config.reload_retriever()
    return num_files, num_chunks


# ---------- Session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "meta"}]


def chat_history():
    msgs = []
    for m in st.session_state.messages[-HISTORY_WINDOW:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    return msgs


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📄 Policy Assistant")
    st.caption("Conversational RAG over your company documents")

    st.subheader("Documents")
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS) if DATA_DIR.exists() else []
    if files:
        for f in files:
            st.markdown(f"- `{f.name}`")
    else:
        st.info("No documents in the data folder yet.")

    uploads = st.file_uploader(
        "Add documents",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if st.button("Save & rebuild index", type="primary", use_container_width=True, disabled=not uploads and not files):
        DATA_DIR.mkdir(exist_ok=True)
        for up in uploads or []:
            (DATA_DIR / up.name).write_bytes(up.getbuffer())
        with st.spinner("Rebuilding vector index..."):
            try:
                n_files, n_chunks = rebuild_index()
                st.success(f"Indexed {n_files} file(s) into {n_chunks} chunks.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")
        st.rerun()

    st.divider()
    show_details = st.toggle("Show retrieval details", value=True)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Session: `{st.session_state.session_id}`")


# ---------- Load backend ----------
try:
    config, rag_pipeline = load_backend()
except FileNotFoundError:
    st.warning("No vector index found. Add documents in the sidebar and click **Save & rebuild index**.")
    st.stop()
except RuntimeError as e:  # e.g. missing GROQ_API_KEY
    st.error(str(e))
    st.stop()


# ---------- Chat UI ----------
def render_meta(meta):
    if not meta or not show_details:
        return
    st.caption(INTENT_BADGE.get(meta.get("intent"), meta.get("intent", "")))
    if meta.get("rewritten_question"):
        st.caption(f"🔎 Searched for: *{meta['rewritten_question']}*")
    if meta.get("sources"):
        with st.expander(f"Sources ({len(meta['sources'])})"):
            for i, s in enumerate(meta["sources"], 1):
                page = f", page {s['page']}" if s.get("page") else ""
                st.markdown(f"**{i}. {s['source']}{page}**")
                st.text(s["content"])


st.header("Ask about company policies")

if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_meta(m.get("meta"))

question = st.chat_input("Ask a question about your documents...")
question = question or st.session_state.pop("pending_question", None)

if question:
    history = chat_history()  # history before this question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag_pipeline.ask(question, history)
                answer, meta = result["answer"], result
            except Exception as e:
                answer, meta = f"⚠️ Something went wrong: {e}", None
        st.markdown(answer)
        render_meta(meta)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
