"""Application settings read from the environment."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration.

    Attributes:
        database_url: SQLAlchemy URL of the Postgres database.
    """

    database_url: str = "postgresql+psycopg://momentum:momentum@localhost:5432/momentum"


settings = Settings()
