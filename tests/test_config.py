"""
tests/test_config.py
====================
配置类的校验逻辑测试。
"""

import pytest

from rag.config import ChunkConfig, EmbeddingConfig, QdrantConfig, RAGConfig, RerankerConfig


class TestEmbeddingConfig:
    def test_default_values(self):
        cfg = EmbeddingConfig(proxy_api_key="key")
        assert cfg.provider == "proxy"
        assert cfg.dimension == 1024
        assert cfg.batch_size == 32

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            EmbeddingConfig(provider="unknown")

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            EmbeddingConfig(dimension=0)

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingConfig(batch_size=-1)


class TestRerankerConfig:
    def test_invalid_top_n_raises(self):
        with pytest.raises(ValueError, match="top_n"):
            RerankerConfig(top_n=0)


class TestQdrantConfig:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            QdrantConfig(mode="ftp")

    def test_valid_modes(self):
        for mode in ("local", "docker", "cloud"):
            cfg = QdrantConfig(mode=mode)
            assert cfg.mode == mode


class TestChunkConfig:
    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(chunk_size=100, chunk_overlap=100)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkConfig(chunk_size=0)

    def test_invalid_cache_backend_raises(self):
        with pytest.raises(ValueError, match="contextual_cache_backend"):
            ChunkConfig(contextual_cache_backend="s3")

    def test_valid_cache_backends(self):
        for backend in ("memory", "disk", "redis"):
            cfg = ChunkConfig(contextual_cache_backend=backend)
            assert cfg.contextual_cache_backend == backend

    def test_default_values(self):
        cfg = ChunkConfig()
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 150
        assert cfg.min_chunk_size == 50


class TestRAGConfig:
    def test_default_construction(self):
        cfg = RAGConfig()
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.reranker, RerankerConfig)
        assert isinstance(cfg.qdrant, QdrantConfig)
        assert isinstance(cfg.chunk, ChunkConfig)
        assert cfg.rrf_k == 60
        assert cfg.fetch_k_multiplier == 5
