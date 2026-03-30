"""RAGEngine 观测与评测测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import rag.engine as engine_module
from rag.config import ChunkConfig, RAGConfig, RerankerConfig
from rag.evaluation import RetrievalEvalCase, evaluate_engine, evaluate_retriever


class TestRAGEngineIndexing:
    def test_reindex_replaces_existing_source_path(self, tmp_path):
        file_path = tmp_path / "doc.md"
        file_path.write_text("# 标题\n\n正文", encoding="utf-8")

        with (
            patch.object(engine_module, "EmbeddingService") as embedder_cls,
            patch.object(engine_module, "QdrantVectorStore") as vector_store_cls,
            patch.object(engine_module, "DocumentParser") as parser_cls,
            patch.object(engine_module, "HierarchicalMarkdownSplitter") as splitter_cls,
        ):
            embedder = MagicMock()
            embedder.embed_all.return_value = [[0.1, 0.2]]
            embedder_cls.return_value = embedder

            vector_store = MagicMock()
            vector_store.doc_exists.return_value = False
            vector_store.collection_info.return_value = {}
            vector_store_cls.return_value = vector_store

            parser = MagicMock()
            parser.parse.return_value = ("# 标题\n\n正文", "markdown")
            parser_cls.return_value = parser

            splitter = MagicMock()
            splitter.split.return_value = [
                {
                    "content": "正文",
                    "heading_path": ["标题"],
                    "chunk_index": 0,
                    "section_index": 0,
                }
            ]
            splitter_cls.return_value = splitter

            engine = engine_module.RAGEngine(
                RAGConfig(
                    reranker=RerankerConfig(api_key=""),
                    chunk=ChunkConfig(use_contextual_retrieval=False),
                )
            )
            engine.index_file(str(file_path), force_reindex=True)

            resolved = str(Path(file_path).resolve())
            vector_store.delete_by_source_path.assert_called_once_with(resolved)
            vector_store.upsert.assert_called_once()

    def test_delete_file_prefers_source_path_for_path_like_input(self):
        with (
            patch.object(engine_module, "EmbeddingService"),
            patch.object(engine_module, "QdrantVectorStore") as vector_store_cls,
            patch.object(engine_module, "DocumentParser"),
            patch.object(engine_module, "HierarchicalMarkdownSplitter"),
        ):
            vector_store = MagicMock()
            vector_store.collection_info.return_value = {}
            vector_store_cls.return_value = vector_store

            engine = engine_module.RAGEngine(
                RAGConfig(
                    reranker=RerankerConfig(api_key=""),
                    chunk=ChunkConfig(use_contextual_retrieval=False),
                )
            )
            engine.delete_file("docs/example.pdf")

            vector_store.delete_by_source_path.assert_called_once()
            vector_store.delete_by_source_file.assert_not_called()

    def test_index_file_records_observability(self, tmp_path):
        file_path = tmp_path / "doc.md"
        file_path.write_text("# 标题\n\n正文", encoding="utf-8")

        with (
            patch.object(engine_module, "EmbeddingService") as embedder_cls,
            patch.object(engine_module, "QdrantVectorStore") as vector_store_cls,
            patch.object(engine_module, "DocumentParser") as parser_cls,
            patch.object(engine_module, "HierarchicalMarkdownSplitter") as splitter_cls,
        ):
            embedder = MagicMock()
            embedder.embed_all.return_value = [[0.1, 0.2]]
            embedder_cls.return_value = embedder

            vector_store = MagicMock()
            vector_store.doc_exists.return_value = False
            vector_store.collection_info.return_value = {}
            vector_store_cls.return_value = vector_store

            parser = MagicMock()
            parser.parse.return_value = ("# 标题\n\n正文", "markdown")
            parser_cls.return_value = parser

            splitter = MagicMock()
            splitter.split.return_value = [
                {
                    "content": "正文",
                    "heading_path": ["标题"],
                    "chunk_index": 0,
                    "section_index": 0,
                }
            ]
            splitter_cls.return_value = splitter

            engine = engine_module.RAGEngine(
                RAGConfig(
                    reranker=RerankerConfig(api_key=""),
                    chunk=ChunkConfig(use_contextual_retrieval=False),
                )
            )
            engine.index_file(str(file_path), force_reindex=True)

            stats = engine.get_last_index_stats()
            assert stats["status"] == "ok"
            assert stats["chunk_count"] == 1
            assert stats["source_path"] == str(Path(file_path).resolve())
            assert stats["embed_ms"] >= 0.0
            assert stats["upsert_ms"] >= 0.0

    def test_index_file_returns_error_when_upsert_fails(self, tmp_path):
        file_path = tmp_path / "doc.md"
        file_path.write_text("# 标题\n\n正文", encoding="utf-8")

        with (
            patch.object(engine_module, "EmbeddingService") as embedder_cls,
            patch.object(engine_module, "QdrantVectorStore") as vector_store_cls,
            patch.object(engine_module, "DocumentParser") as parser_cls,
            patch.object(engine_module, "HierarchicalMarkdownSplitter") as splitter_cls,
        ):
            embedder = MagicMock()
            embedder.embed_all.return_value = [[0.1, 0.2]]
            embedder_cls.return_value = embedder

            vector_store = MagicMock()
            vector_store.doc_exists.return_value = False
            vector_store.collection_info.return_value = {}
            vector_store.upsert.side_effect = RuntimeError("qdrant unavailable")
            vector_store_cls.return_value = vector_store

            parser = MagicMock()
            parser.parse.return_value = ("# 标题\n\n正文", "markdown")
            parser_cls.return_value = parser

            splitter = MagicMock()
            splitter.split.return_value = [
                {
                    "content": "正文",
                    "heading_path": ["标题"],
                    "chunk_index": 0,
                    "section_index": 0,
                }
            ]
            splitter_cls.return_value = splitter

            engine = engine_module.RAGEngine(
                RAGConfig(
                    reranker=RerankerConfig(api_key=""),
                    chunk=ChunkConfig(use_contextual_retrieval=False),
                )
            )
            result = engine.index_file(str(file_path), force_reindex=True)
            stats = engine.get_last_index_stats()

            assert result["status"] == "error"
            assert "Upsert 失败" in result["error"]
            assert stats["status"] == "error"
            assert "Upsert 失败" in stats["error"]


class TestRAGEngineRetrievalObservability:
    def test_retrieve_records_observability(self):
        with (
            patch.object(engine_module, "EmbeddingService") as embedder_cls,
            patch.object(engine_module, "QdrantVectorStore") as vector_store_cls,
            patch.object(engine_module, "DocumentParser"),
            patch.object(engine_module, "HierarchicalMarkdownSplitter"),
        ):
            embedder = MagicMock()
            embedder.embed_single.return_value = [0.1, 0.2]
            embedder_cls.return_value = embedder

            vector_store = MagicMock()
            vector_store.collection_info.return_value = {}
            vector_store.search_dense.return_value = [
                {
                    "id": "doc-1",
                    "score": 0.9,
                    "payload": {
                        "doc_id": "doc-1",
                        "content": "答案",
                        "source_file": "a.md",
                        "source_path": "/tmp/a.md",
                    },
                }
            ]
            vector_store.search_sparse.return_value = []
            vector_store_cls.return_value = vector_store

            engine = engine_module.RAGEngine(
                RAGConfig(
                    reranker=RerankerConfig(api_key=""),
                    chunk=ChunkConfig(use_contextual_retrieval=False),
                )
            )
            results = engine.retrieve("什么是答案", top_k=3)

            stats = engine.get_last_retrieval_stats()
            assert len(results) == 1
            assert stats["mode"] == "sync"
            assert stats["top_k"] == 3
            assert stats["dense_hit_count"] == 1
            assert stats["sparse_hit_count"] == 0
            assert stats["fused_hit_count"] == 1
            assert stats["result_count"] == 1
            assert stats["used_rerank"] is False
            assert stats["result_doc_ids"] == ["doc-1"]


class FakeEvalEngine:
    def __init__(self):
        self._last_stats = {}
        self._responses = {
            "命中问题": [
                {"doc_id": "doc-1", "source_path": "/tmp/a.md", "source_file": "a.md"},
                {"doc_id": "doc-2", "source_path": "/tmp/b.md", "source_file": "b.md"},
            ],
            "未命中问题": [
                {"doc_id": "doc-x", "source_path": "/tmp/x.md", "source_file": "x.md"},
            ],
        }

    def retrieve(self, query, top_k=5, **kwargs):
        self._last_stats = {"top_k": top_k, "query": query, "kwargs": kwargs}
        return self._responses.get(query, [])[:top_k]

    def get_last_retrieval_stats(self):
        return dict(self._last_stats)


class TestEvaluation:
    def test_evaluate_retriever_computes_metrics(self):
        cases = [
            RetrievalEvalCase(query="命中问题", expected_ids=["doc-1"]),
            RetrievalEvalCase(query="未命中问题", expected_ids=["doc-2"]),
        ]

        def retriever(query, top_k=5, **kwargs):
            mapping = {
                "命中问题": [
                    {"doc_id": "doc-1", "source_path": "/tmp/a.md", "source_file": "a.md"},
                    {"doc_id": "doc-2", "source_path": "/tmp/b.md", "source_file": "b.md"},
                ],
                "未命中问题": [
                    {"doc_id": "doc-x", "source_path": "/tmp/x.md", "source_file": "x.md"},
                ],
            }
            return mapping[query][:top_k]

        summary = evaluate_retriever(retriever, cases, top_k=2)
        assert summary.total_queries == 2
        assert summary.hit_count == 1
        assert summary.hit_rate_at_k == 0.5
        assert summary.recall_at_k == 0.5
        assert summary.mrr_at_k == 0.5
        assert summary.average_first_hit_rank == 1.0

    def test_evaluate_engine_captures_last_retrieval_stats(self):
        engine = FakeEvalEngine()
        cases = [
            RetrievalEvalCase(query="命中问题", expected_ids=["doc-1"], metadata={"bucket": "smoke"}),
        ]

        summary = evaluate_engine(engine, cases, top_k=1)
        detail = summary.details[0]
        assert summary.hit_rate_at_k == 1.0
        assert detail.hit is True
        assert detail.first_hit_rank == 1
        assert detail.retrieval_stats["top_k"] == 1
        assert detail.metadata["bucket"] == "smoke"
