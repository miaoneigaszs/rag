"""
tests/test_contextual.py
========================
Contextual Retrieval 缓存后端测试（不依赖 LLM API）。
"""

import pytest

from rag.contextual import MemoryCacheBackend, build_cache_backend
from rag.config import ChunkConfig


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
        """超过 max_size 时应淘汰旧条目。"""
        for i in range(6):  # max_size=5，写第 6 条时触发淘汰
            cache.set(f"k{i}", f"v{i}")
        # 不应超过 max_size（淘汰后 + 新增）
        assert len(cache._cache) <= cache._max_size

    def test_close_is_noop(self, cache):
        cache.close()  # 不应抛出异常


class TestBuildCacheBackend:
    def test_memory_backend(self, tmp_path):
        cfg = ChunkConfig(contextual_cache_backend="memory")
        backend = build_cache_backend(cfg)
        assert isinstance(backend, MemoryCacheBackend)
        backend.close()

    def test_disk_backend_requires_diskcache(self, tmp_path, monkeypatch):
        import rag.contextual as ctx_module
        monkeypatch.setattr(ctx_module, "_HAS_DISKCACHE", False)

        cfg = ChunkConfig(
            contextual_cache_backend="disk",
            contextual_cache_dir=str(tmp_path / "cache"),
        )
        from rag.contextual import DiskCacheBackend
        with pytest.raises(ImportError, match="diskcache"):
            DiskCacheBackend(cache_dir=str(tmp_path / "cache"))

    def test_redis_backend_requires_redis(self, monkeypatch):
        import rag.contextual as ctx_module
        monkeypatch.setattr(ctx_module, "_HAS_REDIS", False)

        from rag.contextual import RedisCacheBackend
        with pytest.raises(ImportError, match="redis"):
            RedisCacheBackend(redis_url="redis://localhost:6379/0")


class TestCacheKeyDeterminism:
    def test_same_content_same_key(self):
        from rag.contextual import ContextualRetrieval
        key1 = ContextualRetrieval._cache_key("相同内容")
        key2 = ContextualRetrieval._cache_key("相同内容")
        assert key1 == key2

    def test_different_content_different_key(self):
        from rag.contextual import ContextualRetrieval
        key1 = ContextualRetrieval._cache_key("内容A")
        key2 = ContextualRetrieval._cache_key("内容B")
        assert key1 != key2

    def test_key_is_hex_string(self):
        from rag.contextual import ContextualRetrieval
        key = ContextualRetrieval._cache_key("测试")
        assert all(c in "0123456789abcdef" for c in key)
        assert len(key) == 64  # SHA-256 hex 长度
