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
from app.wiki import WikiPage, build_wiki

SYSTEM_PROMPT = (
    "你是「天鹅公主 SWAN NOBLE」品牌的 AI 知识助手。"
    "只能依据下面提供的资料回答用户问题，不得编造资料之外的信息；"
    "若资料不足以回答，请如实说明，并列出你参考的资料来源。"
)


def _wiki_context(wiki_pages: List[WikiPage]) -> Tuple[str, List[ChatCitation]]:
    """把聚合页转成上下文与引用（来源列表）。"""
    blocks = []
    citations = []
    for i, page in enumerate(wiki_pages, start=1):
        blocks.append(
            f"[资料{i}]（来自《{page.title}》聚合页）\n"
            f"摘要：{page.summary}\n"
            f"关键事实：{'；'.join(page.facts)}\n"
            f"来源文档：{'、'.join(page.sources)}"
        )
        citations.append(
            ChatCitation(
                document_id=-1,  # 聚合页无 document_id
                title=page.title,
                snippet=page.summary[:120],
            )
        )
    return "\n\n".join(blocks), citations


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
    use_wiki: bool = True,
) -> Tuple[str, List[ChatCitation], int, int]:
    llm = llm or get_llm_provider()
    embedder = embedder or get_embedder()
    retriever = retriever or Retriever(embedder, settings.top_k)

    t0 = time.time()

    # 聚合层优先：命中聚合页则用聚合上下文（raw→wiki 双轨的 wiki 层）
    wiki_hits: List[WikiPage] = []
    if use_wiki:
        wiki = build_wiki()
        wiki_hits = wiki.search(question)
        wiki.save_index()  # 顺带落盘聚合索引

    # RAG 常规路径：混合检索召回原始 chunk
    qvec = embedder.embed(question)
    hits = retriever.search(session, qvec, top_k, query_text=question, use_hybrid=True)

    doc_titles = {d.id: d.title for d in session.exec(select(Document)).all()}
    chunk_context = "\n\n".join(
        f"[资料{i+1}]（来自《{doc_titles.get(chunk.document_id, '未知')}》）\n{chunk.text}"
        for i, (_score, chunk) in enumerate(hits)
    )

    # 上下文 = 聚合页（若有优先）+ 原始 chunk（补充细节）
    contexts = []
    citations: List[ChatCitation] = []
    if wiki_hits:
        wiki_ctx, wiki_cits = _wiki_context(wiki_hits)
        contexts.append(wiki_ctx)
        citations.extend(wiki_cits)
    if chunk_context.strip():
        contexts.append(chunk_context)
        for i, (_score, chunk) in enumerate(hits):
            citations.append(
                ChatCitation(
                    document_id=chunk.document_id,
                    title=doc_titles.get(chunk.document_id, "未知"),
                    snippet=chunk.text[:120],
                )
            )

    context = "\n\n".join(contexts)
    user_msg = f"用户问题：{question}\n\n可参考资料：\n{context}"

    answer = llm.complete(SYSTEM_PROMPT, user_msg)

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
