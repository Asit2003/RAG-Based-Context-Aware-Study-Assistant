# retriever.py
from typing import List, Dict, Any
from embedder import get_embeddings
from vector_store import FAISSStore

def retrieve_relevant_chunks(query: str, store: FAISSStore, k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = get_embeddings([query])[0]
    return store.search(query_embedding, k=k)

def format_context(results: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    """
    Build a context string grouped by doc, trimmed by max_chars.
    """
    grouped = {}
    for r in results:
        d = r["metadata"]["doc_id"]
        grouped.setdefault(d, []).append(r["text"])
    blocks = []
    for doc_id, texts in grouped.items():
        blocks.append(f"### Source: {doc_id}\n" + "\n---\n".join(texts))
    ctx = "\n\n".join(blocks)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n...[truncated]..."
    return ctx
