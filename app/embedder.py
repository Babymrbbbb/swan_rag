"""向量化（Embedder）抽象层。

架构要点（ADR-001 续）：与 LLM 同理，向量化也抽象成接口。
- MockEmbedder：基于字符/二元组哈希的确定性向量，纯 Python、离线、可复现，
  用于本地演示与评测（中文按字/二元组切分，避免英文空格分词对中文失效）。
- RealEmbedder：对接 OpenAI 兼容 embeddings 接口。
"""
import hashlib
import json
from abc import ABC, abstractmethod

import httpx

from app.config import settings


# 高频中文虚词/标点：mock 向量过滤它们，让内容词主导相似度（贴近真实嵌入语义）。
_STOP = set(
    "的是在有着与和以为其也将并而之及等个们都可所被把要对中内上下面后前这那"
    "或就也很更最再没无此从向当该各它她你我又但因若由、，。：；！？（）《》"
    "「」*_~#@%&$=+-/\\| "
)


def _tokens(text: str) -> list:
    """中英文通用的确定性分词：单字 + 二元组，过滤高频虚词与标点。"""
    text = text.lower()
    chars = [c for c in text if c not in _STOP]
    toks = chars[:]
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i + 1])
    return toks


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list:
        """返回定长浮点向量。"""


class MockEmbedder(Embedder):
    def __init__(self) -> None:
        self.dim = settings.embed_dim

    def embed(self, text: str) -> list:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class RealEmbedder(Embedder):
    def __init__(self) -> None:
        self.base_url = settings.embedder_base_url.rstrip("/")
        self.api_key = settings.embedder_api_key or settings.llm_api_key
        self.model = settings.embedder_model

    def embed(self, text: str) -> list:
        if not self.api_key:
            raise RuntimeError("embedder_api_key 未配置，无法调用真实向量化；请设置 RAG_EMBEDDER_API_KEY")
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


# 进程级单例缓存：避免每次 /chat 都新建 embedder（2026-10-30 压测优化）。
_embedder_cache: dict = {}


def get_embedder() -> Embedder:
    provider = settings.embedder_provider
    if provider in _embedder_cache:
        return _embedder_cache[provider]
    if provider == "real":
        embedder: Embedder = RealEmbedder()
    else:
        embedder = MockEmbedder()
    _embedder_cache[provider] = embedder
    return embedder
