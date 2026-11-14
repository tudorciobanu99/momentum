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
    """

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str = "postgresql+psycopg://momentum:momentum@localhost:5432/momentum"
    voyage_api_key: str | None = None
    embedding_model: str = "voyage-4"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 128


settings = Settings()
