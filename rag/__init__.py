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
from .evaluation import RetrievalEvalCase, RetrievalEvalResult, RetrievalEvalSummary
from .models import DocumentChunk


def create_rag_engine(*args, **kwargs):
    from .engine import create_rag_engine as _create_rag_engine

    return _create_rag_engine(*args, **kwargs)


def evaluate_engine(*args, **kwargs):
    from .evaluation import evaluate_engine as _evaluate_engine

    return _evaluate_engine(*args, **kwargs)


def evaluate_retriever(*args, **kwargs):
    from .evaluation import evaluate_retriever as _evaluate_retriever

    return _evaluate_retriever(*args, **kwargs)


def __getattr__(name: str):
    if name == "RAGEngine":
        from .engine import RAGEngine

        return RAGEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RAGEngine",
    "create_rag_engine",
    "evaluate_engine",
    "evaluate_retriever",
    "RAGConfig",
    "EmbeddingConfig",
    "RerankerConfig",
    "QdrantConfig",
    "ChunkConfig",
    "DocumentChunk",
    "RetrievalEvalCase",
    "RetrievalEvalResult",
    "RetrievalEvalSummary",
]
