"""检索器：文本切分 + 向量召回（in-memory 余弦）。"""
import json
import math
from typing import List, Tuple

from sqlmodel import Session, select

from app.embedder import Embedder
from app.models import Chunk


def chunk_text(text: str, size: int = 300, overlap: int = 50) -> List[str]:
    """按字符长度滑窗切分（中文友好）。"""
    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class Retriever:
    def __init__(self, embedder: Embedder, top_k: int = 3) -> None:
        self.embedder = embedder
        self.top_k = top_k

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed(query)

    def search(self, session: Session, query_vec: List[float], top_k: int = None) -> List[Tuple[float, Chunk]]:
        k = top_k or self.top_k
        chunks = session.exec(select(Chunk)).all()
        scored = []
        for c in chunks:
            vec = json.loads(c.embedding)
            scored.append((cosine(query_vec, vec), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
