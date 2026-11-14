"""Application settings read from the environment."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and the repo root `.env` if present.

    Attributes:
        database_url: SQLAlchemy URL of the Postgres database.
        voyage_api_key: Voyage API key, required to embed.
        embedding_model: Voyage embedding model.
        embedding_dimension: Embedding length, matching the `chunks.embedding` column.
        embedding_batch_size: Texts per embedding request.
        vector_top_k: Vector results fused.
        lexical_top_k: Lexical results fused.
        rrf_k: Reciprocal rank fusion constant.
        candidate_limit: Candidates returned by hybrid search.
        rerank_model: Voyage rerank model.
        rerank_top_n: Candidates kept after reranking.
        langsmith_api_key: LangSmith API key, evaluation results are not uploaded without it.
        langsmith_project: LangSmith project that experiments are named after.
    """

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str = "postgresql+psycopg://momentum:momentum@localhost:5432/momentum"
    voyage_api_key: str | None = None
    embedding_model: str = "voyage-4"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 128
    vector_top_k: int = 50
    lexical_top_k: int = 50
    rrf_k: int = 60
    candidate_limit: int = 30
    rerank_model: str = "rerank-3-lite"
    rerank_top_n: int = 10
    langsmith_api_key: str | None = None
    langsmith_project: str = "momentum"


settings = Settings()
