"""配置分离：pydantic-settings，所有变量以 RAG_ 前缀从环境变量/.env 读取。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 数据库
    rag_db_file: str = "swan_rag.db"

    # 全局鉴权
    api_key: str = "dev-secret-key"

    # LLM 供应商（mock | real）
    llm_provider: str = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    # 关闭推理模型的思考链（reasoning），让其把最终答案直接写入 content 字段
    llm_disable_reasoning: bool = True

    # 向量化供应商（mock | real）
    embedder_provider: str = "mock"
    embedder_base_url: str = "https://api.openai.com/v1"
    embedder_api_key: str = ""
    embedder_model: str = "text-embedding-3-small"
    embed_dim: int = 256

    # RAG 参数
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 3

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RAG_", extra="ignore"
    )


settings = Settings()
