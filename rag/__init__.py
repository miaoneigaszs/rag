"""
rag
===
工业级 RAG Pipeline。

快速开始：
    from rag import create_rag_engine

    engine = create_rag_engine(
        embed_api_key="sf-xxxxx",
        reranker_api_key="sf-xxxxx",
    )
    engine.startup_sync()
    engine.index_file("document.pdf")
    results = engine.retrieve("如何配置环境变量？")
"""

from .config import ChunkConfig, EmbeddingConfig, QdrantConfig, RAGConfig, RerankerConfig
from .engine import RAGEngine, create_rag_engine
from .models import DocumentChunk

__all__ = [
    # 引擎
    "RAGEngine",
    "create_rag_engine",
    # 配置
    "RAGConfig",
    "EmbeddingConfig",
    "RerankerConfig",
    "QdrantConfig",
    "ChunkConfig",
    # 数据结构
    "DocumentChunk",
]
