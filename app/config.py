from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always load .env from the project root, regardless of the directory the app is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "RagEval"
    app_env: str = "development"
    debug: bool = False
    data_dir: Path = PROJECT_ROOT / "data"

    # Vector store: if QDRANT_URL is empty, an embedded on-disk Qdrant is used (no server needed).
    qdrant_url: str = ""
    qdrant_collection: str = "pdf_chunks"

    # OpenAI generation (key comes only from the environment / .env).
    openai_api_key: str = ""
    openai_model: str = ""
    openai_api_base: str = "https://api.openai.com/v1"

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_dimension: int = 384
    embedding_version: str = "bge-small-en-v1.5"
    embedding_batch_size: int = Field(default=64, ge=1, le=512)
    re_rank_enabled: bool = True
    re_rank_model_name: str = "BAAI/bge-reranker-base"
    generation_timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_upload_size_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    generation_retry_count: int = Field(default=3, ge=0, le=6)
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 80
    prompt_token_budget: int = Field(default=3000, ge=512, le=100000)
    duplicate_upload_policy: Literal["reject", "replace", "allow"] = "reject"
    retrieval_top_k_default: int = Field(default=5, ge=1, le=50)
    rerank_candidate_k: int = Field(default=20, ge=1, le=100)
    retrieval_similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_structural_filter_enabled: bool = False

    @property
    def qdrant_local_path(self) -> Path:
        return self.data_dir / "qdrant"

    def ensure_data_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_data_dirs()
    return settings
