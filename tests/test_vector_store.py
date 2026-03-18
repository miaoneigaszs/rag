"""
tests/test_vector_store.py
==========================
SparseEncoder 和 RRF 融合逻辑测试（不依赖 Qdrant 服务）。
"""

import pytest

from rag.vector_store import SparseEncoder


class TestSparseEncoder:
    @pytest.fixture()
    def encoder(self) -> SparseEncoder:
        return SparseEncoder()

    def test_empty_text_returns_empty(self, encoder):
        indices, values = encoder.encode("")
        assert indices == []
        assert values == []

    def test_output_lengths_match(self, encoder):
        indices, values = encoder.encode("hello world hello")
        assert len(indices) == len(values)

    def test_no_duplicate_indices(self, encoder):
        indices, _ = encoder.encode("重复 重复 内容 内容 测试")
        assert len(indices) == len(set(indices))

    def test_values_in_unit_range(self, encoder):
        _, values = encoder.encode("some test content here")
        for v in values:
            assert 0.0 < v <= 1.0

    def test_max_freq_token_has_weight_one(self, encoder):
        """出现频率最高的词权重应为 1.0。"""
        # "foo foo foo bar" 中 foo 出现 3 次，bar 1 次
        indices, values = encoder.encode("foo foo foo bar")
        assert max(values) == pytest.approx(1.0)

    def test_repeated_word_single_entry(self, encoder):
        """同一个词不管出现多少次，indices 中只应有一个条目。"""
        indices, _ = encoder.encode("cat cat cat cat")
        assert len(indices) == 1

    def test_chinese_tokenization(self, encoder):
        """中文文本应正确编码（有 jieba 时用 jieba，否则用正则）。"""
        indices, values = encoder.encode("自然语言处理技术")
        assert len(indices) > 0
        assert all(v > 0 for v in values)

    def test_different_texts_different_vectors(self, encoder):
        """不同文本应产生不同的稀疏向量。"""
        idx1, _ = encoder.encode("北京天气")
        idx2, _ = encoder.encode("上海美食")
        assert set(idx1) != set(idx2)

    def test_same_text_same_vector(self, encoder):
        """相同文本应产生完全相同的稀疏向量（确定性）。"""
        idx1, val1 = encoder.encode("测试文本内容")
        idx2, val2 = encoder.encode("测试文本内容")
        assert sorted(zip(idx1, val1)) == sorted(zip(idx2, val2))

    def test_whitespace_only_returns_empty(self, encoder):
        indices, values = encoder.encode("   \t\n  ")
        assert indices == []
        assert values == []


class TestRRFFusion:
    """
    RRF 融合逻辑测试（通过 RAGEngine._rrf_fusion 内部方法）。
    使用 Mock 替代真实 Qdrant，专注测试算法正确性。
    """

    @pytest.fixture()
    def engine(self, monkeypatch):
        """构建一个最小化的 RAGEngine，mock 掉所有 IO 依赖。"""
        from unittest.mock import MagicMock, patch

        with patch("rag.engine.EmbeddingService"), \
             patch("rag.engine.QdrantVectorStore") as mock_vs_cls, \
             patch("rag.engine.DocumentParser"), \
             patch("rag.engine.HierarchicalMarkdownSplitter"):

            mock_vs = MagicMock()
            mock_vs.collection_info.return_value = {}
            mock_vs.fetch_by_ids.return_value = []
            mock_vs_cls.return_value = mock_vs

            from rag.config import RAGConfig
            from rag.engine import RAGEngine
            engine = RAGEngine(RAGConfig())
            engine.vector_store = mock_vs
            return engine

    def test_dense_only_returns_dense_order(self, engine):
        """仅有 dense 结果时，RRF 应保留其排序。"""
        dense = [
            {"id": "a", "score": 0.9, "payload": {"content": "A"}},
            {"id": "b", "score": 0.8, "payload": {"content": "B"}},
        ]
        fused = engine._rrf_fusion(dense, [], top_k=5)
        assert [r["id"] for r in fused] == ["a", "b"]

    def test_sparse_only_returns_sparse_order(self, engine):
        """仅有 sparse 结果时，RRF 应保留其排序。"""
        engine.vector_store.fetch_by_ids.return_value = [
            {"id": "x", "payload": {"content": "X"}},
            {"id": "y", "payload": {"content": "Y"}},
        ]
        sparse = [("x", 2.5), ("y", 1.0)]
        fused = engine._rrf_fusion([], sparse, top_k=5)
        assert [r["id"] for r in fused] == ["x", "y"]

    def test_top_k_respected(self, engine):
        dense = [
            {"id": str(i), "score": float(10 - i), "payload": {"content": f"C{i}"}}
            for i in range(10)
        ]
        fused = engine._rrf_fusion(dense, [], top_k=3)
        assert len(fused) == 3

    def test_result_ranked_first_in_both_gets_highest_rrf(self, engine):
        """同时在 dense 和 sparse 排名第一的文档应有最高 RRF 分。"""
        dense = [
            {"id": "star", "score": 0.99, "payload": {"content": "star"}},
            {"id": "other", "score": 0.50, "payload": {"content": "other"}},
        ]
        sparse = [("star", 5.0), ("other", 1.0)]
        fused = engine._rrf_fusion(dense, sparse, top_k=5)
        assert fused[0]["id"] == "star"

    def test_rrf_score_formula(self, engine):
        """验证 RRF 分数公式：1/(k+rank)。"""
        dense = [{"id": "doc", "score": 1.0, "payload": {"content": "X"}}]
        fused = engine._rrf_fusion(dense, [], top_k=1)
        expected = 1.0 / (engine.cfg.rrf_k + 1)
        assert fused[0]["rrf_score"] == pytest.approx(expected)

    def test_empty_inputs_return_empty(self, engine):
        assert engine._rrf_fusion([], [], top_k=5) == []
