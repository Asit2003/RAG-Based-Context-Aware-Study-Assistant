# app.py
import os
from utils import load_all_pdfs, flatten_chunks
from embedder import chunk_all_docs, get_embeddings
from vector_store import FAISSStore
from retriever import retrieve_relevant_chunks, format_context
from qa_engine import ask_gpt4



DATA_DIR = "data"

def build_index():
    docs = load_all_pdfs(DATA_DIR)
    if not docs:
        raise RuntimeError(f"No PDFs found in {os.path.abspath(DATA_DIR)}.")
    chunks_by_doc = chunk_all_docs(docs)
    texts, metas = flatten_chunks(chunks_by_doc)
    embeddings = get_embeddings(texts)
    dim = len(embeddings[0])
    store = FAISSStore(dim=dim)
    store.add(embeddings, texts, metas)
    return store

def main():
    print("Loading documents & building FAISS index...")
    store = build_index()
    print(f"Indexed {store.size} chunks from PDFs in '{DATA_DIR}'.")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Ask a question: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        results = retrieve_relevant_chunks(q, store, k=5)
        if not results:
            print("No relevant context found. Try rephrasing.\n")
            continue
        ctx = format_context(results)
        ans = ask_gpt4(q, ctx)
        print("\nAnswer:\n", ans, "\n")

if __name__ == "__main__":
    main()
