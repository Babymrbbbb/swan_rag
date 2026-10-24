"""检索器：双路混合检索（倒排关键词 + 向量召回）→ RRF 融合。

架构演进（完整双轨）：
- 路1 · 向量召回：语义匹配（处理口语化/换说法问题）。mock 向量离线可复现；真实 embedding 可用时切换。
- 路2 · 倒排关键词召回：BM25-lite（词频 + IDF + 业务子库加权），精确匹配业务词（处理术语/编号/规范问法）。
- 融合：RRF（Reciprocal Rank Fusion）合并两路结果，去重加权，取 top-k。

设计动机：单路向量在垂直术语场景可能漏精确命中；单路倒排在口语化场景会漏语义。
双路并行取长补短——这正是 llm-wiki「词频饱和 + 业务子库加权」生产检索的通用化实现。
"""
import json
import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple

from sqlmodel import Session, select

from app.embedder import Embedder
from app.models import Chunk

# ---------- 文本处理 ----------

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


# 高频中文虚词/标点：倒排索引分词时过滤（与 MockEmbedder 同一套停用词思路）。
_STOP = set(
    "的是在有着与和以为其也将并而之及等个们都可所被把要对中内上下面后前这那"
    "或就也很更最再没无此从向当该各它她你我又但因若由、，。：；！？（）《》"
    "「」*_~#@%&$=+-/\\| "
)


def _tokenize(text: str) -> List[str]:
    """中英文通用确定性分词：单字 + 二元组，过滤停用词与标点。"""
    text = text.lower()
    chars = [c for c in text if c not in _STOP]
    toks = list(chars)
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i + 1])
    return toks


# ---------- BM25-lite 倒排索引 ----------

class InvertedIndex:
    """内存倒排索引：term -> {chunk_id: tf}，支持 IDF 加权与 BM25-lite 打分。"""

    def __init__(self, chunks: List[Chunk]) -> None:
        self.doc_len = len(chunks)
        self.df: Counter[str] = Counter()                          # term -> 包含它的 chunk 数
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)  # term -> {chunk_id: tf}
        self.chunk_len: dict[int, int] = {}                        # chunk_id -> 词数
        for c in chunks:
            toks = _tokenize(c.text)
            self.chunk_len[c.id] = len(toks)
            for term, cnt in Counter(toks).items():
                self.postings[term][c.id] = cnt
                self.df[term] += 1

    def idf(self, term: str) -> float:
        """IDF：稀有词权重高，泛词被压低（llm-wiki 的 IDF 降权思想）。"""
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.doc_len - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Tuple[float, int]]:
        """BM25-lite：对查询词累加 IDF × tf 饱和，返回 (score, chunk_id)。"""
        terms = _tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        avg_len = (sum(self.chunk_len.values()) / len(self.chunk_len)) if self.chunk_len else 1.0
        for term in set(terms):  # 同一词只算一次（防止长问题刷分）
            idf = self.idf(term)
            if idf == 0:
                continue
            for cid, tf in self.postings.get(term, {}).items():
                dl = self.chunk_len.get(cid, 1)
                denom = tf + k1 * (1 - b + b * dl / avg_len)
                scores[cid] += idf * (tf * (k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------- 向量召回 ----------

def cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------- RRF 融合 ----------

def rrf_fuse(ranked_lists: List[List[Tuple[float, int]]], k: int = 60, top_k: int = 3) -> List[Tuple[float, int]]:
    """Reciprocal Rank Fusion：多路排序融合。

    对每个 chunk，累加 1/(k + rank)，rank 为该路中的位次（从 1 开始）。
    这样两路都靠前的 chunk 得分最高，兼顾精确与语义。
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (_score, cid) in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # 统一返回 (score, chunk_id) 顺序，与 search 解包一致
    return [(float(score), cid) for cid, score in fused[:top_k]]


# ---------- 检索器 ----------

class Retriever:
    def __init__(self, embedder: Embedder, top_k: int = 3, use_hybrid: bool = True) -> None:
        self.embedder = embedder
        self.top_k = top_k
        self.use_hybrid = use_hybrid

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed(query)

    def search(
        self,
        session: Session,
        query_vec: List[float],
        top_k: int = None,
        query_text: str = "",
        use_hybrid: bool = None,
    ) -> List[Tuple[float, Chunk]]:
        """双路检索入口。

        参数：
        - query_vec: 问题向量（必传）
        - query_text: 问题原文（混合检索的关键词路需要；为空则退化为纯向量）
        - use_hybrid: 是否启用双路（默认取 self.use_hybrid）
        """
        k = top_k or self.top_k
        hybrid = self.use_hybrid if use_hybrid is None else use_hybrid
        chunks = session.exec(select(Chunk)).all()
        if not chunks:
            return []

        # 路1：向量余弦
        vec_ranked: List[Tuple[float, int]] = []
        for c in chunks:
            try:
                vec = json.loads(c.embedding)
            except Exception:
                vec = [0.0]
            vec_ranked.append((cosine(query_vec, vec), c.id))
        vec_ranked.sort(key=lambda x: x[1], reverse=True)  # 按 id 稳定
        vec_ranked.sort(key=lambda x: x[0], reverse=True)
        vec_top = vec_ranked[: max(k * 3, 5)]

        # 路2：倒排关键词（query_text 非空且启用混合时）
        kw_top: List[Tuple[float, int]] = []
        if hybrid and query_text.strip():
            index = InvertedIndex(chunks)
            kw_top = index.search(query_text, top_k=max(k * 3, 5))

        # 融合
        if hybrid and kw_top:
            fused = rrf_fuse([vec_top, kw_top], top_k=k)
        else:
            fused = vec_top[:k]

        # 映射回 Chunk
        by_id = {c.id: c for c in chunks}
        result = []
        for score, cid in fused:
            if cid in by_id:
                result.append((float(score), by_id[cid]))
        return result