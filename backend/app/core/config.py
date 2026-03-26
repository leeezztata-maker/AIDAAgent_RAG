from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Data Analysis Agent API"
    api_prefix: str = "/api/v1"
    app_env: str = Field(default="development", alias="APP_ENV")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=256, alias="EMBEDDING_DIMENSION")
    retrieval_top_k: int = Field(default=6, alias="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float = Field(default=0.15, alias="RETRIEVAL_SCORE_THRESHOLD")
    rag_context_char_limit: int = Field(default=3000, alias="RAG_CONTEXT_CHAR_LIMIT")
    evaluation_output_dir_name: str = "exports"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    return settings
