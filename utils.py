# utils.py
import os
import glob
import fitz  # PyMuPDF
from typing import List, Dict, Any

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    return "\n".join(text_parts)

def load_all_pdfs(data_dir: str) -> List[Dict[str, Any]]:
    """
    Scan data_dir recursively for PDFs.
    Returns: [{"doc_id": "notes.pdf", "path": "/abs/path/notes.pdf", "text": "..."}]
    Skips zero-length results.
    """
    pdf_paths = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    docs = []
    for p in pdf_paths:
        txt = extract_text_from_pdf(p).strip()
        if txt:
            docs.append({
                "doc_id": os.path.basename(p),
                "path": os.path.abspath(p),
                "text": txt
            })
    return docs

def flatten_chunks(chunks_by_doc: Dict[str, List[Dict[str, Any]]]):
    """
    Turn {doc_id: [chunk_dict,...]} into parallel lists for easier embedding/indexing.
    Returns: texts, metadatas
    """
    texts, metas = [], []
    for doc_id, clist in chunks_by_doc.items():
        for ch in clist:
            texts.append(ch["content"])
            metas.append({
                "doc_id": doc_id,
                "chunk_id": ch["chunk_id"],
                "source_path": ch["source_path"],
                "char_start": ch["char_start"],
                "char_end": ch["char_end"]
            })
    return texts, metas
