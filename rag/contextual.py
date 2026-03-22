"""
rag/contextual.py
=================
Contextual Retrieval（Anthropic 2024 论文）实现。

改进版：使用 section 级上下文（同 section 所有 chunk 拼合）替代单 chunk + 标题路径，
解决原版"局部视野"问题，同时避免全文传入的高成本。

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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List, Optional

from openai import OpenAI
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
                evict = max(1, self._max_size // 10)
                for k in list(self._cache.keys())[:evict]:
                    del self._cache[k]
            self._cache[key] = value


class DiskCacheBackend(ContextualCacheBackend):
    """diskcache SQLite 文件后端，跨重启持久化，单机首选。"""

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
    """Redis 后端，多 Worker / 多实例共享，TTL 支持，生产高可用。"""

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
    return MemoryCacheBackend(max_size=cfg.contextual_cache_size)


# =============================================================================
# Contextual Retrieval
# =============================================================================

class ContextualRetrieval:
    """
    改进版 Contextual Retrieval：section 级上下文生成。

    原版问题：只把 heading_path + 单个 chunk 传给 LLM，LLM 没有全局视野，
    生成的 context_prefix 与直接用 heading_str 差别不大。

    改进策略：
      1. 按 section_index 将同一文档的 chunk 分组
      2. 把同 section 内所有 chunk 拼合成"局部文档"（通常 500~3000 字）
      3. LLM 以"局部文档 + 当前 chunk"为输入生成上下文摘要
      → LLM 获得了 section 级全局视野，成本远低于全文方案

    缓存键 = hash(section_text + chunk_content)，section 内容变化时自动失效。
    """

    # 输入：section 全文 + 当前 chunk，要求生成一句上下文描述
    _PROMPT_TEMPLATE = (
        "以下是一个文档章节的完整内容（section）：\n"
        "---\n"
        "{section_text}\n"
        "---\n\n"
        "现在请看该章节中的一个片段（chunk）：\n"
        "---\n"
        "{chunk_content}\n"
        "---\n\n"
        "请结合章节全文，用1-2句话描述这个片段在章节中的位置和作用，"
        "使其在被单独检索时依然语义清晰。只输出描述，不要解释或重复原文。\n"
        "请用中文回答（若文档为英文则用英文）："
    )

    # section 文本截断上限（约 3000 字），防止单 section 过长导致费用失控
    _SECTION_TEXT_LIMIT = 3000

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
            f"[ContextualRetrieval] 已启用（section 级上下文），模型={self._model}, "
            f"max_concurrency={self._max_concurrency}, "
            f"cache_backend={chunk_cfg.contextual_cache_backend}"
        )

    # ------------------------------------------------------------------
    # 缓存键
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(section_text: str, chunk_content: str) -> str:
        """缓存键 = hash(section_text前200字 + chunk_content前500字)。"""
        raw = f"{section_text[:200]}::{chunk_content[:500]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 单条生成
    # ------------------------------------------------------------------

    def generate_context(self, section_text: str, chunk_content: str) -> str:
        """
        为单个 chunk 生成上下文摘要（同步，带持久化缓存）。

        Args:
            section_text  : 当前 chunk 所在 section 的完整文本（已截断）
            chunk_content : 当前 chunk 的原始内容
        """
        key = self._cache_key(section_text, chunk_content)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        prompt = self._PROMPT_TEMPLATE.format(
            section_text=section_text[:self._SECTION_TEXT_LIMIT],
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
    # 批量并发生成（section 级）
    # ------------------------------------------------------------------

    def enrich_chunks(
        self,
        chunks: List["DocumentChunk"],
        max_workers: Optional[int] = None,
    ) -> List["DocumentChunk"]:
        """
        按 section_index 分组，为每个 chunk 生成 section 级上下文前缀。

        流程：
          1. 按 section_index 聚合，拼合同 section 所有 chunk 为 section_text
          2. 批量预检缓存命中，统计需要调用 LLM 的数量
          3. ThreadPoolExecutor + Semaphore 并发生成，保持原始顺序写回

        section_index == -1 的 chunk（合并短节产生）降级为只用 heading_str。
        """
        workers = max_workers or self._max_concurrency
        sem = threading.Semaphore(self._max_concurrency)

        # ── 1. 按 section_index 聚合 section 文本 ─────────────────────────
        # section_text_map: section_index → 该 section 所有 chunk 内容拼合
        section_text_map: Dict[int, str] = defaultdict(str)
        for chunk in chunks:
            idx = chunk.section_index
            if idx != -1:
                # 按 chunk_index 顺序拼合，用双换行分隔（heading_str作为section标题首行）
                section_text_map[idx]  # 预占 key，保证 defaultdict 有序填充

        # 先按 chunk_index 排序，保证拼合顺序正确
        sorted_chunks = sorted(chunks, key=lambda c: (c.section_index, c.chunk_index))
        for chunk in sorted_chunks:
            idx = chunk.section_index
            if idx != -1:
                prefix = f"【{chunk.heading_str}】\n" if chunk.heading_str and not section_text_map[idx] else ""
                section_text_map[idx] += prefix + chunk.content + "\n\n"

        # ── 2. 为每个 chunk 确定其 section_text ──────────────────────────
        def _get_section_text(chunk: "DocumentChunk") -> str:
            if chunk.section_index == -1:
                # 降级：没有 section 信息，用 heading_str 作为最小上下文
                return chunk.heading_str or ""
            return section_text_map.get(chunk.section_index, "")

        # ── 3. 批量预检缓存 ────────────────────────────────────────────────
        section_texts = [_get_section_text(c) for c in chunks]
        keys = [self._cache_key(st, c.content) for st, c in zip(section_texts, chunks)]
        cached_flags = [self._cache.get(k) is not None for k in keys]
        cache_hits = sum(cached_flags)

        logger.info(
            f"[ContextualRetrieval] 开始处理 {len(chunks)} 个 chunk "
            f"（缓存命中 {cache_hits}，需调用 LLM {len(chunks) - cache_hits}，并发={workers}）"
        )

        # ── 4. 并发生成 ────────────────────────────────────────────────────
        def _guarded_generate(chunk: "DocumentChunk", section_text: str) -> str:
            with sem:
                return self.generate_context(section_text, chunk.content)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_guarded_generate, c, st): i
                for i, (c, st) in enumerate(zip(chunks, section_texts))
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="ContextualRetrieval"
            ):
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