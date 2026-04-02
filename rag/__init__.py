"""
rag
===
Industrial RAG SDK with optional API and adapter layers.
"""

from .config import ChunkConfig, EmbeddingConfig, QdrantConfig, RAGConfig, RerankerConfig
from .evaluation import (
    RetrievalEvalCase,
    RetrievalEvalResult,
    RetrievalEvalSummary,
    load_eval_cases,
)
from .models import (
    DeleteOptions,
    DeleteResult,
    DeleteTarget,
    DocumentChunk,
    DocumentSource,
    HealthStatus,
    IndexOptions,
    IndexRequest,
    IndexResult,
    RetrieveOptions,
    RetrieveResult,
    RetrievedItem,
    SearchRequest,
)


def create_rag_engine(*args, **kwargs):
    from .engine import create_rag_engine as _create_rag_engine

    return _create_rag_engine(*args, **kwargs)


def create_sdk(*args, **kwargs):
    from .service import KnowledgeSDK

    return KnowledgeSDK(*args, **kwargs)


def create_knowledge_service(*args, **kwargs):
    from .service import AgentKnowledgeService

    return AgentKnowledgeService(*args, **kwargs)


def create_api_app(*args, **kwargs):
    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)


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
    if name == "AgentKnowledgeService":
        from .service import AgentKnowledgeService

        return AgentKnowledgeService
    if name == "KnowledgeSDK":
        from .service import KnowledgeSDK

        return KnowledgeSDK
    if name == "MCPAdapter":
        from .mcp import MCPAdapter

        return MCPAdapter
    if name == "KnowledgeSkill":
        from .skill import KnowledgeSkill

        return KnowledgeSkill
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RAGEngine",
    "AgentKnowledgeService",
    "KnowledgeSDK",
    "MCPAdapter",
    "KnowledgeSkill",
    "create_rag_engine",
    "create_sdk",
    "create_knowledge_service",
    "create_api_app",
    "evaluate_engine",
    "evaluate_retriever",
    "load_eval_cases",
    "RAGConfig",
    "EmbeddingConfig",
    "RerankerConfig",
    "QdrantConfig",
    "ChunkConfig",
    "DocumentChunk",
    "DocumentSource",
    "IndexRequest",
    "SearchRequest",
    "DeleteTarget",
    "IndexOptions",
    "RetrieveOptions",
    "DeleteOptions",
    "IndexResult",
    "RetrieveResult",
    "DeleteResult",
    "RetrievedItem",
    "HealthStatus",
    "RetrievalEvalCase",
    "RetrievalEvalResult",
    "RetrievalEvalSummary",
]
