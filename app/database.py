"""数据库连接与会话：sqlite + SQLModel，Depends 注入。

并发压测优化（2026-10-30）：SQLite 默认 rollback journal 模式在并发写时
单写者串行 + 每次 commit 都 fsync，导致 /chat 高并发下 QA 审计落库互相锁等待。
这里改为 WAL + synchronous=NORMAL + busy_timeout：
- WAL：读写不互斥，写事务追加到 wal 文件，大幅降低锁竞争；
- synchronous=NORMAL：WAL 模式下崩溃不损坏库，只是可能丢最近提交（审计可接受）；
- busy_timeout=5000：并发写排队而非立即报 database is locked。
"""
from sqlalchemy import event
from sqlmodel import SQLModel, create_engine, Session

from app.config import settings

sqlite_url = f"sqlite:///./{settings.rag_db_file}"
engine = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """每个连接建立时设置 SQLite 并发优化 PRAGMA。"""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def init_db() -> None:
    """建表（仅本地兜底；生产应使用 Alembic 迁移）。"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI 依赖：每次请求一个 Session，自动关闭。"""
    with Session(engine) as session:
        yield session
