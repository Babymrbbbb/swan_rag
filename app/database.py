"""数据库连接与会话：sqlite + SQLModel，Depends 注入。"""
from sqlmodel import SQLModel, create_engine, Session

from app.config import settings

sqlite_url = f"sqlite:///./{settings.rag_db_file}"
engine = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})


def init_db() -> None:
    """建表（仅本地兜底；生产应使用 Alembic 迁移）。"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI 依赖：每次请求一个 Session，自动关闭。"""
    with Session(engine) as session:
        yield session
