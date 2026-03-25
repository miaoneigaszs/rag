"""Contextual Retrieval 缓存相关测试。"""

import pytest

from rag.config import ChunkConfig
from rag.contextual import MemoryCacheBackend, build_cache_backend


class TestMemoryCacheBackend:
    @pytest.fixture()
    def cache(self) -> MemoryCacheBackend:
        return MemoryCacheBackend(max_size=5)

    def test_set_and_get(self, cache):
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_missing_key_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_overwrite_existing_key(self, cache):
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"

    def test_eviction_when_full(self, cache):
        for i in range(6):
            cache.set(f"k{i}", f"v{i}")
        assert len(cache._cache) <= cache._max_size

    def test_close_is_noop(self, cache):
        cache.close()


class TestBuildCacheBackend:
    def test_memory_backend(self):
        backend = build_cache_backend(ChunkConfig(contextual_cache_backend="memory"))
        assert isinstance(backend, MemoryCacheBackend)
        backend.close()

    def test_disk_backend_requires_diskcache(self, tmp_path, monkeypatch):
        import rag.contextual as ctx_module
        from rag.contextual import DiskCacheBackend

        monkeypatch.setattr(ctx_module, "_HAS_DISKCACHE", False)
        with pytest.raises(ImportError, match="diskcache"):
            DiskCacheBackend(cache_dir=str(tmp_path / "cache"))

    def test_redis_backend_requires_redis(self, monkeypatch):
        import rag.contextual as ctx_module
        from rag.contextual import RedisCacheBackend

        monkeypatch.setattr(ctx_module, "_HAS_REDIS", False)
        with pytest.raises(ImportError, match="redis"):
            RedisCacheBackend(redis_url="redis://localhost:6379/0")


class TestCacheKeyDeterminism:
    def test_same_content_same_key(self):
        from rag.contextual import ContextualRetrieval

        key1 = ContextualRetrieval._cache_key("章节内容", "片段A")
        key2 = ContextualRetrieval._cache_key("章节内容", "片段A")
        assert key1 == key2

    def test_different_content_different_key(self):
        from rag.contextual import ContextualRetrieval

        key1 = ContextualRetrieval._cache_key("章节A", "片段")
        key2 = ContextualRetrieval._cache_key("章节B", "片段")
        assert key1 != key2

    def test_key_is_hex_string(self):
        from rag.contextual import ContextualRetrieval

        key = ContextualRetrieval._cache_key("章节", "片段")
        assert all(char in "0123456789abcdef" for char in key)
        assert len(key) == 64
