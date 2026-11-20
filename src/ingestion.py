"""Load documents from the data folder and upload them to the Supabase vectorstore."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    SUPABASE_TABLE,
    get_supabase_client,
    get_vectorstore,
)

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
}


def load_documents(data_dir: Path = DATA_DIR, verbose: bool = False):
    documents = []
    for file_path in sorted(Path(data_dir).iterdir()):
        loader_cls = LOADERS.get(file_path.suffix.lower())
        if loader_cls is None:
            continue
        docs = loader_cls(str(file_path)).load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
        documents.extend(docs)
        if verbose:
            print(f"Loaded {file_path.name}: {len(docs)} page(s)")
    return documents


def build_vectorstore(data_dir: Path = DATA_DIR, verbose: bool = False):
    """Rebuild the vectorstore from every supported file. Returns (num_documents, num_chunks)."""
    documents = load_documents(data_dir, verbose=verbose)
    if not documents:
        raise ValueError(f"No .pdf/.txt/.md files found in '{data_dir}'.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    # Full rebuild: clear existing rows, then insert fresh chunks.
    client = get_supabase_client()
    client.table(SUPABASE_TABLE).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    get_vectorstore().add_documents(chunks)
    if verbose:
        print(f"Uploaded {len(chunks)} chunks to Supabase table '{SUPABASE_TABLE}'")

    num_files = len({d.metadata["source"] for d in documents})
    return num_files, len(chunks)
