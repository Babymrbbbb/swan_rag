"""评测闭环：基于 golden set 跑 召回命中率 + 忠实度。

架构意义（ADR-002）：AI 应用必须有"质量闭环"。离线用 MockProvider 也能跑通评测，
证明管线（切分→向量化→召回→生成）正确；接入真实 LLM 后，faithfulness 才有真实意义。
"""
import json
from pathlib import Path
from typing import List

from sqlmodel import Session

from app.embedder import get_embedder
from app.llm import get_llm_provider
from app.rag import chat as rag_chat
from app.retriever import Retriever
from app.schemas import EvalReport, EvalResult

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_golden.json"


def run_eval(session: Session) -> EvalReport:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    llm = get_llm_provider()
    embedder = get_embedder()
    retriever = Retriever(embedder, 3)

    details: List[EvalResult] = []
    retrieval_hits = 0
    faithfulness_hits = 0

    for case in golden:
        question = case["question"]
        expected_doc = case["expected_doc"]
        expected_hint = case.get("expected_hint", "")

        answer, citations, _latency, _tok = rag_chat(
            session, question, top_k=3, llm=llm, embedder=embedder, retriever=retriever
        )
        retrieved_titles = [c.title for c in citations]
        retrieval_hit = expected_doc in retrieved_titles

        # 忠实度（检索充分性）：期望事实是否出现在「生成的回答」或「召回的 top-k 原文」中。
        # 这是 RAG 质量闭环的合理指标；接入真实 LLM 后应使用 LLM-Judge / NLI 度量生成答案本身的忠实度。
        qvec = embedder.embed(question)
        hits = retriever.search(session, qvec, 3)
        context_full = "\n".join(ch.text for _s, ch in hits)
        faithfulness_hit = bool(expected_hint) and (
            expected_hint in answer or expected_hint in context_full
        )

        if retrieval_hit:
            retrieval_hits += 1
        if faithfulness_hit:
            faithfulness_hits += 1

        details.append(
            EvalResult(
                question=question,
                expected_doc=expected_doc,
                retrieved_top1_doc=retrieved_titles[0] if retrieved_titles else "",
                retrieval_hit=retrieval_hit,
                faithfulness_hit=faithfulness_hit,
                answer=answer[:200],
            )
        )

    total = len(golden)
    return EvalReport(
        total=total,
        retrieval_hit_rate=round(retrieval_hits / total, 3) if total else 0.0,
        faithfulness_rate=round(faithfulness_hits / total, 3) if total else 0.0,
        note=(
            "当前为离线 MockProvider：召回命中率验证检索与切分正确；"
            "faithfulness 仅在接入真实 LLM 后具备真实意义。"
        ),
        details=details,
    )


if __name__ == "__main__":
    from app.database import engine, init_db

    init_db()
    with Session(engine) as s:
        report = run_eval(s)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
