"""
rag/config.py
=============
所有配置 dataclass，通过环境变量或直接赋值驱动。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    # 自动搜索当前目录或父目录中的 .env 文件并加载
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Embedding 接口配置（兼容 OpenAI API 规范的任意服务）"""

    # 官方 OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "text-embedding-3-small"

    # 中转站（SiliconFlow / OneAPI / 智谱 等）
    proxy_api_key: str = field(default_factory=lambda: os.getenv("PROXY_API_KEY", ""))
    proxy_base_url: str = field(
        default_factory=lambda: os.getenv("PROXY_BASE_URL", "https://api.siliconflow.cn/v1")
    )
    proxy_model: str = field(
        default_factory=lambda: os.getenv("PROXY_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
    )

    # 运行时选择："openai" 或 "proxy"
    provider: str = field(default_factory=lambda: os.getenv("EMBED_PROVIDER", "proxy"))

    # 向量维度
    dimension: int = field(default_factory=lambda: int(os.getenv("EMBED_DIM", "1024")))

    # 批次大小
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.provider not in ("openai", "proxy"):
            raise ValueError(f"EmbeddingConfig.provider 必须为 'openai' 或 'proxy'，当前值: {self.provider!r}")
        if self.dimension <= 0:
            raise ValueError(f"EmbeddingConfig.dimension 必须为正整数，当前值: {self.dimension}")
        if self.batch_size <= 0:
            raise ValueError(f"EmbeddingConfig.batch_size 必须为正整数，当前值: {self.batch_size}")


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

@dataclass
class RerankerConfig:
    """Reranker API 配置（兼容 OpenAI 规范的 rerank 接口）"""

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

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError(f"RerankerConfig.top_n 必须为正整数，当前值: {self.top_n}")


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

@dataclass
class QdrantConfig:
    """
    Qdrant 连接配置。

    三选一：
      mode="local"  → 本地文件（无需 Docker），path 指定目录
      mode="docker" → 本机 Docker，host+port
      mode="cloud"  → Qdrant Cloud，url+api_key
    """

    mode: str = field(default_factory=lambda: os.getenv("QDRANT_MODE", "local"))
    path: str = field(default_factory=lambda: os.getenv("QDRANT_PATH", "./qdrant_data"))
    host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "rag_docs")
    )

    def __post_init__(self) -> None:
        if self.mode not in ("local", "docker", "cloud"):
            raise ValueError(
                f"QdrantConfig.mode 必须为 'local'/'docker'/'cloud'，当前值: {self.mode!r}"
            )


# ---------------------------------------------------------------------------
# Chunk / Contextual
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """切块策略 + Contextual Retrieval 配置"""

    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_size: int = 50

    # Contextual Retrieval（Anthropic 2024）
    use_contextual_retrieval: bool = True
    context_model: str = field(default_factory=lambda: os.getenv("CONTEXT_MODEL", ""))
    context_max_tokens: int = 100
    contextual_max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CONTEXTUAL_MAX_CONCURRENCY", "8"))
    )
    contextual_cache_size: int = field(
        default_factory=lambda: int(os.getenv("CONTEXTUAL_CACHE_SIZE", "2048"))
    )

    # 缓存后端："memory" / "disk" / "redis"
    contextual_cache_backend: str = field(
        default_factory=lambda: os.getenv("CONTEXTUAL_CACHE_BACKEND", "disk")
    )
    contextual_cache_dir: str = field(
        default_factory=lambda: os.getenv("CONTEXTUAL_CACHE_DIR", "./contextual_cache")
    )
    contextual_cache_redis_url: str = field(
        default_factory=lambda: os.getenv(
            "CONTEXTUAL_CACHE_REDIS_URL", "redis://localhost:6379/0"
        )
    )
    contextual_cache_ttl: int = field(
        default_factory=lambda: int(os.getenv("CONTEXTUAL_CACHE_TTL", "0"))
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"ChunkConfig.chunk_size 必须为正整数，当前值: {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"ChunkConfig.chunk_overlap 不能为负数，当前值: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("ChunkConfig.chunk_overlap 必须小于 chunk_size")
        if self.contextual_cache_backend not in ("memory", "disk", "redis"):
            raise ValueError(
                f"ChunkConfig.contextual_cache_backend 必须为 'memory'/'disk'/'redis'，"
                f"当前值: {self.contextual_cache_backend!r}"
            )


# ---------------------------------------------------------------------------
# 顶层汇总
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """顶层配置汇总"""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)

    # 检索参数
    fetch_k_multiplier: int = 5   # 海选倍率：top_k * fetch_k_multiplier 进入 rerank
    rrf_k: int = 60               # RRF 平滑常数
    score_threshold: float = 0.0  # 最低相似度阈值（0 = 不过滤）
