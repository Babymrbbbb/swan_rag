"""请求/响应模型（非表模型，纯 Pydantic）。入参/出参分离。"""
from typing import List, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    title: str
    content: str
    source: str = "manual"


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class ChatCitation(BaseModel):
    document_id: int
    title: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[ChatCitation]
    latency_ms: int
    model: str


class MetricsResponse(BaseModel):
    requests: int
    chat_requests: int
    avg_latency_ms: float
    total_tokens: int


class EvalResult(BaseModel):
    question: str
    expected_doc: str
    retrieved_top1_doc: str
    retrieval_hit: bool
    faithfulness_hit: bool
    answer: str


class EvalReport(BaseModel):
    total: int
    retrieval_hit_rate: float
    faithfulness_rate: float
    note: str
    details: List[EvalResult]
