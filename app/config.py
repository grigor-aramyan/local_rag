from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".text", ".html", ".htm"})


class Settings(BaseSettings):
    """Runtime configuration, read from the environment.

    Retrieval knobs default to the brief's starting point (500/50/50/5). Per the
    brief these are the highest-leverage values in the system — move them against
    the eval harness, not by intuition.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    lancedb_path: Path = Path("/data/lancedb")
    documents_path: Path = Path("/data/documents")
    jobs_db_path: Path = Path("/data/lancedb/jobs.db")
    table_name: str = "chunks"

    # Uploads
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    allowed_extensions: frozenset[str] = DEFAULT_EXTENSIONS

    # Models — baked into the image at /models; changing these needs a rebuild.
    # fastembed resolves its cache from FASTEMBED_CACHE_PATH, but pass it
    # explicitly rather than depending on an env var surviving the image.
    model_cache_path: Path = Path("/models")
    # fastembed's HuggingFace path raises under HF_HUB_OFFLINE instead of
    # falling back to the cache, so the baked weights are only found when this
    # is set. Turn it off only to let a model download at runtime.
    model_local_files_only: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, gt=0)
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Chunking. Sizes are in the embedder's own tokens, not words or characters,
    # so chunk_size must stay under the model's context length (512 for bge-small)
    # or every chunk loses its tail to truncation.
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    # Chunks per ONNX batch. Kept well under fastembed's default of 256: a batch
    # of 500-token chunks is a lot of resident memory in a ~1 GB container.
    embed_batch_size: int = Field(default=32, gt=0)

    # Retrieval
    top_k: int = Field(default=50, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)
    ann_index_threshold: int = Field(default=100_000, gt=0)
    # Toggleable per the brief: measure whether reranking helps before treating it
    # as mandatory in the hot path.
    rerank_enabled: bool = True

    # Generation — the Anthropic API. Only generation is remote; embedding and
    # reranking stay local so the corpus never leaves the container.
    anthropic_api_key: SecretStr | None = None
    anthropic_base_url: str | None = None
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = Field(default=4_096, gt=0)
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"

    @model_validator(mode="after")
    def _check_coherence(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); equal or larger never advances."
            )
        if self.rerank_top_n > self.top_k:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) cannot exceed top_k "
                f"({self.top_k}); the reranker only sees what retrieval returned."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
