"""Contextual Retrieval 与缓存后端实现。"""

from __future__ import annotations

import abc
import hashlib
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List, Optional

from tqdm import tqdm

from .config import ChunkConfig, RAGConfig

if TYPE_CHECKING:
    from .models import DocumentChunk

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore

try:
    import diskcache as _diskcache
    _HAS_DISKCACHE = True
except ImportError:
    _HAS_DISKCACHE = False

try:
    import redis as _redis_lib
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class ContextualCacheBackend(abc.ABC):
    """Contextual Retrieval 缓存后端接口。"""

    @abc.abstractmethod
    def get(self, key: str) -> Optional[str]:
        ...

    @abc.abstractmethod
    def set(self, key: str, value: str) -> None:
        ...

    def close(self) -> None:
        """释放后端资源。"""


class MemoryCacheBackend(ContextualCacheBackend):
    """基于内存字典的轻量缓存。"""

    def __init__(self, max_size: int = 2048) -> None:
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._max_size = max_size

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            if len(self._cache) >= self._max_size:
                evict = max(1, self._max_size // 10)
                for old_key in list(self._cache.keys())[:evict]:
                    del self._cache[old_key]
            self._cache[key] = value


class DiskCacheBackend(ContextualCacheBackend):
    """基于 diskcache 的磁盘缓存。"""

    def __init__(self, cache_dir: str, size_limit_mb: int = 512) -> None:
        if not _HAS_DISKCACHE:
            raise ImportError("缺少 diskcache 依赖: pip install diskcache")
        from pathlib import Path

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = _diskcache.Cache(cache_dir, size_limit=size_limit_mb * 1024 * 1024)
        logger.info(f"[ContextualCache] DiskCache 已启用: {cache_dir}")

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)  # type: ignore[return-value]

    def set(self, key: str, value: str) -> None:
        self._cache.set(key, value)

    def close(self) -> None:
        self._cache.close()


class RedisCacheBackend(ContextualCacheBackend):
    """基于 Redis 的共享缓存。"""

    _KEY_PREFIX = "ctx_retrieval:"

    def __init__(self, redis_url: str, ttl: int = 0) -> None:
        if not _HAS_REDIS:
            raise ImportError("缺少 redis 依赖: pip install redis")
        self._client = _redis_lib.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        self._client.ping()
        logger.info(f"[ContextualCache] Redis 已连接: {redis_url}, TTL={ttl}s")

    def get(self, key: str) -> Optional[str]:
        return self._client.get(f"{self._KEY_PREFIX}{key}")  # type: ignore[return-value]

    def set(self, key: str, value: str) -> None:
        full_key = f"{self._KEY_PREFIX}{key}"
        if self._ttl > 0:
            self._client.setex(full_key, self._ttl, value)
        else:
            self._client.set(full_key, value)

    def close(self) -> None:
        self._client.close()


def build_cache_backend(cfg: ChunkConfig) -> ContextualCacheBackend:
    """根据配置构建缓存后端。"""
    backend = cfg.contextual_cache_backend
    if backend == "redis":
        return RedisCacheBackend(
            redis_url=cfg.contextual_cache_redis_url,
            ttl=cfg.contextual_cache_ttl,
        )
    if backend == "disk":
        return DiskCacheBackend(cache_dir=cfg.contextual_cache_dir)
    return MemoryCacheBackend(max_size=cfg.contextual_cache_size)


class ContextualRetrieval:
    """基于章节上下文为 chunk 生成前缀摘要。"""

    _PROMPT_VERSION = "v3"
    _PROMPT_TEMPLATE = (
        "请阅读下面的章节内容，概括它与目标 chunk 的关系。\n"
        "---\n"
        "{section_text}\n"
        "---\n\n"
        "目标 chunk 内容如下：\n"
        "---\n"
        "{chunk_content}\n"
        "---\n\n"
        "请用 1-2 句话生成可直接拼接到 chunk 前面的上下文摘要，不要重复原文。"
    )
    _SECTION_TEXT_LIMIT = 3000

    def __init__(self, cfg: RAGConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("启用 Contextual Retrieval 需要安装 openai") from exc

        embed_cfg = cfg.embedding
        chunk_cfg = cfg.chunk

        if embed_cfg.provider == "openai":
            self._client = OpenAI(
                api_key=embed_cfg.openai_api_key,
                base_url=embed_cfg.openai_base_url,
            )
            self._model = chunk_cfg.context_model or "gpt-4o-mini"
        else:
            self._client = OpenAI(
                api_key=embed_cfg.proxy_api_key,
                base_url=embed_cfg.proxy_base_url,
            )
            self._model = chunk_cfg.context_model or "Qwen/Qwen2.5-7B-Instruct"

        self._max_tokens = chunk_cfg.context_max_tokens
        self._max_concurrency = chunk_cfg.contextual_max_concurrency
        self._cache = build_cache_backend(chunk_cfg)

        logger.info(
            f"[ContextualRetrieval] 已启用，model={self._model}, "
            f"max_concurrency={self._max_concurrency}, "
            f"cache_backend={chunk_cfg.contextual_cache_backend}"
        )

    @classmethod
    def _cache_key(cls, section_text: str, chunk_content: str, model: str = "") -> str:
        """使用完整 section/chunk 内容生成稳定 cache key。"""
        raw = "\n".join((cls._PROMPT_VERSION, model, section_text, chunk_content))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate_context(self, section_text: str, chunk_content: str) -> str:
        """为单个 chunk 生成上下文前缀。"""
        key = self._cache_key(section_text, chunk_content, self._model)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        prompt = self._PROMPT_TEMPLATE.format(
            section_text=section_text[: self._SECTION_TEXT_LIMIT],
            chunk_content=chunk_content[:500],
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            result = (resp.choices[0].message.content or "").strip()
            self._cache.set(key, result)
            return result
        except Exception as exc:
            logger.warning(f"[ContextualRetrieval] 生成上下文失败: {exc}")
            return ""

    def enrich_chunks(
        self,
        chunks: List["DocumentChunk"],
        max_workers: Optional[int] = None,
    ) -> List["DocumentChunk"]:
        """按 section 聚合上下文，并为每个 chunk 生成 context_prefix。"""
        workers = max_workers or self._max_concurrency
        sem = threading.Semaphore(self._max_concurrency)

        section_text_map: Dict[int, str] = defaultdict(str)
        sorted_chunks = sorted(chunks, key=lambda c: (c.section_index, c.chunk_index))
        for chunk in sorted_chunks:
            idx = chunk.section_index
            if idx == -1:
                continue
            if chunk.heading_str and not section_text_map[idx]:
                section_text_map[idx] = f"# {chunk.heading_str}\n"
            section_text_map[idx] += chunk.content + "\n\n"

        def _get_section_text(chunk: "DocumentChunk") -> str:
            if chunk.section_index == -1:
                return chunk.heading_str or ""
            return section_text_map.get(chunk.section_index, "")

        section_texts = [_get_section_text(chunk) for chunk in chunks]
        keys = [self._cache_key(section_text, chunk.content, self._model) for section_text, chunk in zip(section_texts, chunks)]
        cache_hits = sum(self._cache.get(key) is not None for key in keys)

        logger.info(
            f"[ContextualRetrieval] 处理 {len(chunks)} 个 chunk，"
            f"缓存命中 {cache_hits}，LLM 生成 {len(chunks) - cache_hits}，workers={workers}"
        )

        def _guarded_generate(chunk: "DocumentChunk", section_text: str) -> str:
            with sem:
                return self.generate_context(section_text, chunk.content)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_guarded_generate, chunk, section_text): idx
                for idx, (chunk, section_text) in enumerate(zip(chunks, section_texts))
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="ContextualRetrieval"):
                idx = futures[future]
                try:
                    chunks[idx].context_prefix = future.result()
                except Exception as exc:
                    logger.warning(f"[ContextualRetrieval] chunk[{idx}] 处理失败: {exc}")

        return chunks

    def close(self) -> None:
        """释放缓存后端连接。"""
        self._cache.close()
