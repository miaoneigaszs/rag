"""配置对象与环境变量加载逻辑。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv_if_available(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    """在安装了 python-dotenv 时显式加载 .env。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(dotenv_path=dotenv_path, override=override))


@dataclass
class EmbeddingConfig:
    """Embedding 服务配置，支持 OpenAI 官方与兼容代理。"""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "text-embedding-3-small"

    proxy_api_key: str = field(default_factory=lambda: os.getenv("PROXY_API_KEY", ""))
    proxy_base_url: str = field(
        default_factory=lambda: os.getenv("PROXY_BASE_URL", "https://api.siliconflow.cn/v1")
    )
    proxy_model: str = field(
        default_factory=lambda: os.getenv("PROXY_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
    )

    provider: str = field(default_factory=lambda: os.getenv("EMBED_PROVIDER", "proxy"))
    dimension: int = field(default_factory=lambda: int(os.getenv("EMBED_DIM", "1024")))
    batch_size: int = 32
    max_input_chars: int = field(default_factory=lambda: int(os.getenv("EMBED_MAX_CHARS", "450")))

    def __post_init__(self) -> None:
        if self.provider not in ("openai", "proxy"):
            raise ValueError(f"EmbeddingConfig.provider 只能是 'openai' 或 'proxy': {self.provider!r}")
        if self.dimension <= 0:
            raise ValueError(f"EmbeddingConfig.dimension 必须为正整数: {self.dimension}")
        if self.batch_size <= 0:
            raise ValueError(f"EmbeddingConfig.batch_size 必须为正整数: {self.batch_size}")
        if self.max_input_chars <= 0:
            raise ValueError(
                f"EmbeddingConfig.max_input_chars 必须为正整数: {self.max_input_chars}"
            )


@dataclass
class RerankerConfig:
    """Reranker API 配置。"""

    api_key: str = field(
        default_factory=lambda: os.getenv("RERANKER_API_KEY", os.getenv("PROXY_API_KEY", ""))
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1")
    )
    model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    )
    top_n: int = 5
    trust_env: bool = field(default_factory=lambda: _env_bool("RERANKER_TRUST_ENV", False))

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError(f"RerankerConfig.top_n 必须为正整数: {self.top_n}")


@dataclass
class QdrantConfig:
    """Qdrant 连接配置。"""

    mode: str = field(default_factory=lambda: os.getenv("QDRANT_MODE", "local"))
    path: str = field(default_factory=lambda: os.getenv("QDRANT_PATH", "./qdrant_data"))
    host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_name: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "rag_docs"))

    def __post_init__(self) -> None:
        if self.mode not in ("local", "docker", "cloud"):
            raise ValueError(
                f"QdrantConfig.mode 只能是 'local'/'docker'/'cloud': {self.mode!r}"
            )


@dataclass
class ChunkConfig:
    """切块与 Contextual Retrieval 配置。"""

    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_size: int = 50

    rag_mode: str = field(default_factory=lambda: os.getenv("RAG_MODE", "basic"))
    use_contextual_retrieval: bool = field(
        default_factory=lambda: _env_bool("USE_CONTEXTUAL_RETRIEVAL", False)
    )
    context_model: str = field(default_factory=lambda: os.getenv("CONTEXT_MODEL", ""))
    context_max_tokens: int = 100
    contextual_max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CONTEXTUAL_MAX_CONCURRENCY", "8"))
    )
    contextual_cache_size: int = field(
        default_factory=lambda: int(os.getenv("CONTEXTUAL_CACHE_SIZE", "2048"))
    )
    contextual_cache_backend: str = field(
        default_factory=lambda: os.getenv("CONTEXTUAL_CACHE_BACKEND", "disk")
    )
    contextual_cache_dir: str = field(
        default_factory=lambda: os.getenv("CONTEXTUAL_CACHE_DIR", "./contextual_cache")
    )
    contextual_cache_redis_url: str = field(
        default_factory=lambda: os.getenv("CONTEXTUAL_CACHE_REDIS_URL", "redis://localhost:6379/0")
    )
    contextual_cache_ttl: int = field(
        default_factory=lambda: int(os.getenv("CONTEXTUAL_CACHE_TTL", "0"))
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"ChunkConfig.chunk_size 必须为正整数: {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"ChunkConfig.chunk_overlap 不能为负数: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("ChunkConfig.chunk_overlap 必须小于 chunk_size")
        if self.rag_mode not in ("basic", "advanced"):
            raise ValueError(f"ChunkConfig.rag_mode 只能是 'basic' 或 'advanced': {self.rag_mode!r}")
        if self.contextual_cache_backend not in ("memory", "disk", "redis"):
            raise ValueError(
                "ChunkConfig.contextual_cache_backend 只能是 'memory'/'disk'/'redis': "
                f"{self.contextual_cache_backend!r}"
            )


@dataclass
class RAGConfig:
    """RAG 总配置。"""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)

    fetch_k_multiplier: int = 5
    rrf_k: int = 60
    score_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.fetch_k_multiplier <= 0:
            raise ValueError("RAGConfig.fetch_k_multiplier 必须为正整数")
        if self.rrf_k <= 0:
            raise ValueError("RAGConfig.rrf_k 必须为正整数")
        if self.score_threshold < 0:
            raise ValueError("RAGConfig.score_threshold 不能为负数")

    @classmethod
    def from_env(
        cls,
        *,
        load_dotenv_file: bool = True,
        dotenv_path: Optional[str] = None,
        override_env: bool = False,
    ) -> "RAGConfig":
        """按需加载 .env，再构建配置对象。"""
        if load_dotenv_file:
            load_dotenv_if_available(dotenv_path=dotenv_path, override=override_env)
        return cls()
