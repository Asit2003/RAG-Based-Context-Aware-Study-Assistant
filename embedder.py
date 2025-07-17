# embedder.py
import math
from typing import List, Dict, Any
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your key before import or via env var
# openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # cost-efficient; upgrade if needed

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)

def chunk_doc(doc_id: str, text: str, source_path: str) -> List[Dict[str, Any]]:
    """
    Returns list of chunk dicts: {"doc_id","chunk_id","content","source_path","char_start","char_end"}
    """
    splits = _splitter.split_text(text)
    chunks = []
    offset = 0
    for i, s in enumerate(splits):
        # Find s in text starting at current offset for rough char range (fallback if not found)
        idx = text.find(s, offset)
        if idx == -1:
            idx = offset
        char_start = idx
        char_end = idx + len(s)
        offset = char_end
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": i,
            "content": s,
            "source_path": source_path,
            "char_start": char_start,
            "char_end": char_end
        })
    return chunks

def chunk_all_docs(docs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {d["doc_id"]: chunk_doc(d["doc_id"], d["text"], d["path"]) for d in docs}

def get_embeddings(
    texts: List[str],
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Batch to reduce API overhead.
    """
    all_embeds: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.embeddings.create(model=model, input=batch)
        # API returns embeddings per input in order
        all_embeds.extend([d.embedding for d in resp.data])
    return all_embeds
