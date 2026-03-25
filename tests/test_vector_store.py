"""SparseEncoder 与 RRF 融合测试。"""

from unittest.mock import MagicMock, patch

import pytest

from rag.config import ChunkConfig, RAGConfig, RerankerConfig
from rag.vector_store import SparseEncoder
import rag.engine as engine_module


class TestSparseEncoder:
    @pytest.fixture()
    def encoder(self, tmp_path) -> SparseEncoder:
        return SparseEncoder(idf_path=str(tmp_path / "bm25_idf"))

    def test_empty_text_returns_empty(self, encoder):
        indices, values = encoder.encode("")
        assert indices == []
        assert values == []

    def test_output_lengths_match(self, encoder):
        indices, values = encoder.encode("hello world hello")
        assert len(indices) == len(values)

    def test_no_duplicate_indices(self, encoder):
        indices, _ = encoder.encode("alpha beta alpha beta alpha")
        assert len(indices) == len(set(indices))

    def test_values_in_unit_range(self, encoder):
        _, values = encoder.encode("some test content here")
        assert all(0.0 < value <= 1.0 for value in values)

    def test_max_freq_token_has_weight_one(self, encoder):
        _, values = encoder.encode("foo foo foo bar")
        assert max(values) == pytest.approx(1.0)

    def test_repeated_word_single_entry(self, encoder):
        indices, _ = encoder.encode("cat cat cat cat")
        assert len(indices) == 1

    def test_chinese_tokenization(self, encoder):
        indices, values = encoder.encode("中文分词测试文本")
        assert len(indices) > 0
        assert all(value > 0 for value in values)

    def test_different_texts_different_vectors(self, encoder):
        idx1, _ = encoder.encode("苹果 香蕉")
        idx2, _ = encoder.encode("数据库 检索")
        assert set(idx1) != set(idx2)

    def test_same_text_same_vector(self, encoder):
        idx1, val1 = encoder.encode("一致性测试")
        idx2, val2 = encoder.encode("一致性测试")
        assert sorted(zip(idx1, val1)) == sorted(zip(idx2, val2))

    def test_whitespace_only_returns_empty(self, encoder):
        indices, values = encoder.encode("   \t\r\n  ")
        assert indices == []
        assert values == []

    def test_remove_idf_reverts_doc_stats(self, encoder):
        encoder.update_idf(["alpha beta", "beta gamma"])
        assert encoder._doc_count == 2
        encoder.remove_idf(["alpha beta"])
        assert encoder._doc_count == 1
        assert encoder._doc_freq.get("alpha", 0) == 0
        assert encoder._doc_freq.get("beta", 0) == 1


class TestRRFFusion:
    @pytest.fixture()
    def engine(self):
        with (
            patch.object(engine_module, "EmbeddingService"),
            patch.object(engine_module, "QdrantVectorStore") as mock_vs_cls,
            patch.object(engine_module, "DocumentParser"),
            patch.object(engine_module, "HierarchicalMarkdownSplitter"),
        ):
            mock_vs = MagicMock()
            mock_vs.collection_info.return_value = {}
            mock_vs.fetch_by_ids.return_value = []
            mock_vs_cls.return_value = mock_vs

            cfg = RAGConfig(
                reranker=RerankerConfig(api_key=""),
                chunk=ChunkConfig(use_contextual_retrieval=False),
            )
            engine = engine_module.RAGEngine(cfg)
            engine.vector_store = mock_vs
            return engine

    def test_dense_only_returns_dense_order(self, engine):
        dense = [
            {"id": "a", "score": 0.9, "payload": {"content": "A"}},
            {"id": "b", "score": 0.8, "payload": {"content": "B"}},
        ]
        fused = engine._rrf_fusion(dense, [], top_k=5)
        assert [item["id"] for item in fused] == ["a", "b"]

    def test_sparse_only_returns_sparse_order(self, engine):
        engine.vector_store.fetch_by_ids.return_value = [
            {"id": "x", "payload": {"content": "X"}},
            {"id": "y", "payload": {"content": "Y"}},
        ]
        fused = engine._rrf_fusion([], [("x", 2.5), ("y", 1.0)], top_k=5)
        assert [item["id"] for item in fused] == ["x", "y"]

    def test_top_k_respected(self, engine):
        dense = [
            {"id": str(i), "score": float(10 - i), "payload": {"content": f"C{i}"}}
            for i in range(10)
        ]
        assert len(engine._rrf_fusion(dense, [], top_k=3)) == 3

    def test_result_ranked_first_in_both_gets_highest_rrf(self, engine):
        dense = [
            {"id": "star", "score": 0.99, "payload": {"content": "star"}},
            {"id": "other", "score": 0.50, "payload": {"content": "other"}},
        ]
        sparse = [("star", 5.0), ("other", 1.0)]
        fused = engine._rrf_fusion(dense, sparse, top_k=5)
        assert fused[0]["id"] == "star"

    def test_rrf_score_formula(self, engine):
        dense = [{"id": "doc", "score": 1.0, "payload": {"content": "X"}}]
        fused = engine._rrf_fusion(dense, [], top_k=1)
        expected = 1.0 / (engine.cfg.rrf_k + 1)
        assert fused[0]["rrf_score"] == pytest.approx(expected)

    def test_empty_inputs_return_empty(self, engine):
        assert engine._rrf_fusion([], [], top_k=5) == []
