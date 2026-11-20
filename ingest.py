from src.ingestion import build_vectorstore

if __name__ == "__main__":
    num_files, num_chunks = build_vectorstore(verbose=True)
    print(f"Indexed {num_files} file(s) into {num_chunks} chunks.")
    print("Vector database saved successfully.")
