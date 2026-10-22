"""FastAPI 入口：路由层 + Depends 注入 + 全局鉴权 + 统一错误信封 + lifespan。"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status as http_status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.config import settings
from app.database import get_session, init_db
from app.embedder import get_embedder
from app.eval import run_eval
from app.llm import get_llm_provider
from app.metrics import metrics
from app.rag import chat as rag_chat, ingest
from app.retriever import Retriever, chunk_text
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    MetricsResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：建表（生产应改用 Alembic 迁移）
    init_db()
    yield


async def get_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """全局依赖：所有接口强制带 X-API-Key，缺失/错误返回 401 统一信封。"""
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED, detail="无效或缺失 API Key"
        )
    return x_api_key


app = FastAPI(
    title="天鹅公主 RAG 知识问答服务",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(get_api_key)],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "请求体校验失败",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm": settings.llm_provider,
        "embedder": settings.embedder_provider,
        "model": settings.llm_model,
    }


@app.post("/ingest")
def ingest_doc(req: IngestRequest, session: Session = Depends(get_session)):
    embedder = get_embedder()
    doc = ingest(session, req.title, req.content, req.source, embedder=embedder)
    n_chunks = len(chunk_text(req.content, settings.chunk_size, settings.chunk_overlap))
    return {"id": doc.id, "title": doc.title, "chunks": n_chunks}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, session: Session = Depends(get_session)):
    llm = get_llm_provider()
    embedder = get_embedder()
    retriever = Retriever(embedder, settings.top_k)
    answer, citations, latency, _tokens = rag_chat(
        session, req.question, top_k=req.top_k, llm=llm, embedder=embedder, retriever=retriever
    )
    return ChatResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency,
        model=f"{settings.llm_provider}:{settings.llm_model}",
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    return MetricsResponse(
        requests=metrics.requests,
        chat_requests=metrics.chat_requests,
        avg_latency_ms=round(metrics.avg_latency, 2),
        total_tokens=metrics.total_tokens,
    )


@app.post("/eval")
def eval_endpoint(session: Session = Depends(get_session)):
    return run_eval(session)
