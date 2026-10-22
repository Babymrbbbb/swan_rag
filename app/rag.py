"""RAG 管线编排：ingest（灌库）与 chat（问答）。"""
import json
import time
from typing import List, Tuple

from sqlmodel import Session, select

from app.config import settings
from app.embedder import Embedder, get_embedder
from app.llm import LLMProvider, get_llm_provider
from app.metrics import metrics
from app.models import Chunk, Document, QA
from app.retriever import Retriever, chunk_text
from app.schemas import ChatCitation

SYSTEM_PROMPT = (
    "你是「天鹅公主 SWAN NOBLE」品牌的 AI 知识助手。"
    "只能依据下面提供的资料回答用户问题，不得编造资料之外的信息；"
    "若资料不足以回答，请如实说明，并列出你参考的资料来源。"
)


def ingest(
    session: Session,
    title: str,
    content: str,
    source: str = "manual",
    embedder: Embedder = None,
) -> Document:
    embedder = embedder or get_embedder()
    doc = Document(title=title, source=source, content=content)
    session.add(doc)
    session.commit()
    session.refresh(doc)

    chunks = chunk_text(content, settings.chunk_size, settings.chunk_overlap)
    for seq, ch in enumerate(chunks):
        vec = embedder.embed(ch)
        session.add(
            Chunk(document_id=doc.id, seq=seq, text=ch, embedding=json.dumps(vec))
        )
    session.commit()
    return doc


def chat(
    session: Session,
    question: str,
    top_k: int = None,
    llm: LLMProvider = None,
    embedder: Embedder = None,
    retriever: Retriever = None,
) -> Tuple[str, List[ChatCitation], int, int]:
    llm = llm or get_llm_provider()
    embedder = embedder or get_embedder()
    retriever = retriever or Retriever(embedder, settings.top_k)

    t0 = time.time()
    qvec = embedder.embed(question)
    hits = retriever.search(session, qvec, top_k)

    doc_titles = {d.id: d.title for d in session.exec(select(Document)).all()}
    context = "\n\n".join(
        f"[资料{i+1}]（来自《{doc_titles.get(chunk.document_id, '未知')}》）\n{chunk.text}"
        for i, (_score, chunk) in enumerate(hits)
    )
    user_msg = f"用户问题：{question}\n\n可参考资料：\n{context}"

    answer = llm.complete(SYSTEM_PROMPT, user_msg)

    citations = [
        ChatCitation(
            document_id=chunk.document_id,
            title=doc_titles.get(chunk.document_id, "未知"),
            snippet=chunk.text[:120],
        )
        for _score, chunk in hits
    ]

    latency = int((time.time() - t0) * 1000)
    # token 近似（mock 与真实通用）：字符数 / 4
    approx_tokens = (len(answer) + len(user_msg)) // 4
    metrics.record_request()
    metrics.record_chat(latency, approx_tokens)

    # 落库审计
    qa = QA(
        question=question,
        answer=answer,
        citations=json.dumps([c.model_dump() for c in citations], ensure_ascii=False),
    )
    session.add(qa)
    session.commit()

    return answer, citations, latency, approx_tokens
