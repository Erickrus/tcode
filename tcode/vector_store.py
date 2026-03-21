from __future__ import annotations
from typing import List, Dict, Any, Optional
import time

# Simple in-memory vector store placeholder for MVP
# It does substring-based retrieval; not semantic embeddings yet.

class InMemoryVectorStore:
    def __init__(self):
        # session_id -> list of docs
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def add_documents(self, session_id: str, docs: List[Dict[str, Any]]):
        # each doc: {'id': str, 'text': str, 'meta': {...}}
        bucket = self._store.setdefault(session_id, [])
        for d in docs:
            entry = dict(d)
            entry.setdefault('created_at', int(time.time()))
            bucket.append(entry)

    def query(self, session_id: str, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
        bucket = self._store.get(session_id, [])
        ql = q.lower()
        scored = []
        for d in bucket:
            text = d.get('text', '') or ''
            score = text.lower().count(ql)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:top_k]]

# global default
_default_store: Optional[InMemoryVectorStore] = None

def get_default_store() -> InMemoryVectorStore:
    global _default_store
    if _default_store is None:
        _default_store = InMemoryVectorStore()
    return _default_store
