# vector_store.py
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self._metas: List[Dict[str, Any]] = []
        self._texts: List[str] = []

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]]):
        if not embeddings:
            return
        arr = np.array(embeddings, dtype="float32")
        self.index.add(arr)
        self._texts.extend(texts)
        self._metas.extend(metadatas)

    def search(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        if self.size == 0:
            return []
        q = np.array([query_embedding], dtype="float32")
        D, I = self.index.search(q, min(k, self.size))
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({
                "text": self._texts[idx],
                "metadata": self._metas[idx],
                "score": float(dist)
            })
        return results

    # Optional persistence
    def save(self, path: str):
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, texts: List[str], metadatas: List[Dict[str, Any]]):
        idx = faiss.read_index(path)
        store = cls(dim=idx.d)
        store.index = idx
        store._texts = texts
        store._metas = metadatas
        return store
