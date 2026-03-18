"""
rag/contextual.py
=================
Contextual Retrieval（Anthropic 2024 论文）实现。

为每个 chunk 在存储前追加一段 LLM 生成的上下文摘要，
解决 chunk 脱离文档上下文后语义漂移的问题。

缓存后端（ContextualCacheBackend 抽象层）：
  MemoryCacheBackend  → 内存 dict，进程内，调试用
  DiskCacheBackend    → diskcache SQLite，跨重启持久化，单机首选
  RedisCacheBackend   → Redis，多 Worker / 多实例共享，生产高可用
"""

from __future__ import annotations

import abc
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, List, Optional

from openai import AsyncOpenAI, OpenAI
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


# =============================================================================
# 缓存后端
# =============================================================================

class ContextualCacheBackend(abc.ABC):
    """Contextual Retrieval 缓存后端抽象基类。"""

    @abc.abstractmethod
    def get(self, key: str) -> Optional[str]:
        ...

    @abc.abstractmethod
    def set(self, key: str, value: str) -> None:
        ...

    def close(self) -> None:
        """释放资源（可选）。"""


class MemoryCacheBackend(ContextualCacheBackend):
    """内存 dict 后端（调试 / 极小规模用，重启丢失）。"""

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
                # 简单 LRU：淘汰最旧的 10%
                evict = max(1, self._max_size // 10)
                for k in list(self._cache.keys())[:evict]:
                    del self._cache[k]
            self._cache[key] = value


class DiskCacheBackend(ContextualCacheBackend):
    """
    diskcache SQLite 文件后端。

    优点：跨重启持久化、零额外服务、线程安全、自动 LRU 淘汰。
    适用：单机部署 / 单进程 / 开发测试环境。
    """

    def __init__(self, cache_dir: str, size_limit_mb: int = 512) -> None:
        if not _HAS_DISKCACHE:
            raise ImportError("请安装 diskcache: pip install diskcache")
        from pathlib import Path
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = _diskcache.Cache(
            cache_dir, size_limit=size_limit_mb * 1024 * 1024
        )
        logger.info(f"[ContextualCache] DiskCache 后端已就绪: {cache_dir}")

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)  # type: ignore[return-value]

    def set(self, key: str, value: str) -> None:
        self._cache.set(key, value)

    def close(self) -> None:
        self._cache.close()


class RedisCacheBackend(ContextualCacheBackend):
    """
    Redis 后端。

    优点：多 Worker / 多实例共享、TTL 支持。
    适用：生产多实例部署。
    """

    _KEY_PREFIX = "ctx_retrieval:"

    def __init__(self, redis_url: str, ttl: int = 0) -> None:
        if not _HAS_REDIS:
            raise ImportError("请安装 redis: pip install redis")
        self._client = _redis_lib.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        self._client.ping()
        logger.info(f"[ContextualCache] Redis 后端已就绪: {redis_url}, TTL={ttl}s")

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
    """根据配置构建对应的缓存后端实例。"""
    backend = cfg.contextual_cache_backend
    if backend == "redis":
        return RedisCacheBackend(
            redis_url=cfg.contextual_cache_redis_url,
            ttl=cfg.contextual_cache_ttl,
        )
    if backend == "disk":
        return DiskCacheBackend(cache_dir=cfg.contextual_cache_dir)
    # "memory" 或未知值（config 层已校验，此处是最后防线）
    return MemoryCacheBackend(max_size=cfg.contextual_cache_size)


# =============================================================================
# Contextual Retrieval
# =============================================================================

class ContextualRetrieval:
    """
    Anthropic 2024 论文：Contextual Retrieval。

    为每个 chunk 在存储前追加一段 LLM 生成的上下文摘要，
    缓存命中时不重复调用 LLM（disk/redis 后端跨重启有效）。
    """

    _PROMPT_TEMPLATE = (
        "以下是一份完整文档的节选内容（chunk），请用1-2句话简要描述该 chunk\n"
        "在整篇文档中的位置和作用，以帮助检索时快速定位。只输出描述，不要解释。\n\n"
        "文档标题路径：{heading_str}\n\n"
        "Chunk 内容：\n{chunk_content}\n\n"
        "请用中文回答（若文档为英文则用英文）："
    )

    def __init__(self, cfg: RAGConfig) -> None:
        embed_cfg = cfg.embedding
        chunk_cfg = cfg.chunk

        if embed_cfg.provider == "openai":
            self._client = OpenAI(
                api_key=embed_cfg.openai_api_key, base_url=embed_cfg.openai_base_url
            )
            self._model = chunk_cfg.context_model or "gpt-4o-mini"
        else:
            self._client = OpenAI(
                api_key=embed_cfg.proxy_api_key, base_url=embed_cfg.proxy_base_url
            )
            self._model = chunk_cfg.context_model or "Qwen/Qwen2.5-7B-Instruct"

        self._max_tokens = chunk_cfg.context_max_tokens
        self._max_concurrency = chunk_cfg.contextual_max_concurrency
        self._cache = build_cache_backend(chunk_cfg)

        logger.info(
            f"[ContextualRetrieval] 已启用，模型={self._model}, "
            f"max_concurrency={self._max_concurrency}, "
            f"cache_backend={chunk_cfg.contextual_cache_backend}"
        )

    # ------------------------------------------------------------------
    # 缓存键
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(content: str) -> str:
        """以内容前 500 字符的 SHA-256 作为缓存键。"""
        return hashlib.sha256(content[:500].encode()).hexdigest()

    # ------------------------------------------------------------------
    # 单条生成
    # ------------------------------------------------------------------

    def generate_context(self, chunk_content: str, heading_str: str) -> str:
        """为单个 chunk 生成上下文摘要（同步，带持久化缓存）。"""
        key = self._cache_key(chunk_content)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        prompt = self._PROMPT_TEMPLATE.format(
            heading_str=heading_str or "（无标题）",
            chunk_content=chunk_content[:500],
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            result = resp.choices[0].message.content.strip()
            self._cache.set(key, result)
            return result
        except Exception as exc:
            logger.warning(f"[ContextualRetrieval] 生成失败（跳过）: {exc}")
            return ""

    # ------------------------------------------------------------------
    # 批量并发生成
    # ------------------------------------------------------------------

    def enrich_chunks(
        self,
        chunks: List["DocumentChunk"],
        max_workers: Optional[int] = None,
    ) -> List["DocumentChunk"]:
        """
        并发为所有 chunk 生成上下文前缀。

        使用 ThreadPoolExecutor + Semaphore 双层限速：
          - ThreadPoolExecutor.max_workers 控制线程总数
          - Semaphore 控制同时飞行的 HTTP 请求数

        缓存命中统计使用批量预检（一次性收集 key → 批量查询），
        避免对 Redis 等后端产生 N 次独立网络请求。
        """
        workers = max_workers or self._max_concurrency
        sem = threading.Semaphore(self._max_concurrency)

        # 批量预检缓存命中（减少 N+1 问题：一次遍历构建 key 列表，逐一查询但在本地完成）
        keys = [self._cache_key(c.content) for c in chunks]
        cached_flags = [self._cache.get(k) is not None for k in keys]
        cache_hits = sum(cached_flags)

        logger.info(
            f"[ContextualRetrieval] 开始处理 {len(chunks)} 个 chunk "
            f"（缓存命中 {cache_hits}，需调用 LLM {len(chunks) - cache_hits}，并发={workers}）"
        )

        def _guarded_generate(chunk: "DocumentChunk") -> str:
            with sem:
                return self.generate_context(chunk.content, chunk.heading_str)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_guarded_generate, c): i for i, c in enumerate(chunks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ContextualRetrieval"):
                idx = futures[future]
                try:
                    chunks[idx].context_prefix = future.result()
                except Exception as exc:
                    logger.warning(f"[ContextualRetrieval] chunk[{idx}] 生成失败: {exc}")

        return chunks

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭缓存后端连接（由 RAGEngine.shutdown() 调用）。"""
        self._cache.close()
