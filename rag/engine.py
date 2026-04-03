"""RAG 主引擎。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .chunker import HierarchicalMarkdownSplitter
from .config import RAGConfig
from .contextual import ContextualRetrieval
from .embedder import EmbeddingService
from .models import DocumentChunk
from .parser import DocumentParser
from .reranker import APIReranker
from .vector_store import QdrantVectorStore

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore


_LOGICAL_SOURCE_URI_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

class RAGEngine:
    """负责索引、检索、融合与重排的统一入口。"""

    def __init__(self, cfg: Optional[RAGConfig] = None) -> None:
        self.cfg = cfg or RAGConfig.from_env()

        self.parser = DocumentParser()
        self.splitter = HierarchicalMarkdownSplitter(self.cfg.chunk)
        self.embedder = EmbeddingService(self.cfg.embedding)
        self.vector_store = QdrantVectorStore(self.cfg.qdrant, self.cfg.embedding.dimension)
        self.reranker: Optional[APIReranker] = (
            APIReranker(self.cfg.reranker) if self.cfg.reranker.api_key else None
        )
        self.contextual = self._build_contextual()
        self._started = False
        self._last_index_stats: Dict[str, Any] = {}
        self._last_retrieval_stats: Dict[str, Any] = {}

    def _build_contextual(self) -> Optional[ContextualRetrieval]:
        """上下文增强功能，如果启用，将基于章节内容为chunk添加上下文摘要"""
        if not self.cfg.chunk.use_contextual_retrieval:
            return None
        api_key = (
            self.cfg.embedding.openai_api_key
            if self.cfg.embedding.provider == "openai"
            else self.cfg.embedding.proxy_api_key
        )
        if not api_key:
            logger.warning("[RAGEngine] 已启用 Contextual Retrieval，但缺少 API Key，自动跳过")
            return None
        return ContextualRetrieval(self.cfg)

    def get_last_index_stats(self) -> Dict[str, Any]:
        """返回最近一次索引过程的观测数据。"""
        return deepcopy(self._last_index_stats)

    def get_last_retrieval_stats(self) -> Dict[str, Any]:
        """返回最近一次检索过程的观测数据。"""
        return deepcopy(self._last_retrieval_stats)

    def reset_observability(self) -> None:
        """清空最近一次索引/检索观测数据。"""
        self._last_index_stats = {}
        self._last_retrieval_stats = {}

    def _store_index_stats(self, stats: Dict[str, Any]) -> None:
        """
        存储索引过程的观测数据。

        Args:
            stats (Dict[str, Any]): 索引过程中的观测数据，如各种耗时、文档信息、chunk数量等。
        """
        stats = dict(stats)
        stats.setdefault("total_ms", 0.0)
        self._last_index_stats = stats
        logger.info(
            "[IndexMetrics] status=%s file=%s chunks=%s total_ms=%.2f parse_ms=%.2f chunk_ms=%.2f embed_ms=%.2f upsert_ms=%.2f",
            stats.get("status", "unknown"),
            stats.get("source_file", ""),
            stats.get("chunk_count", 0),
            stats.get("total_ms", 0.0),
            stats.get("parse_ms", 0.0),
            stats.get("chunk_ms", 0.0),
            stats.get("embed_ms", 0.0),
            stats.get("upsert_ms", 0.0),
        )

    def _store_retrieval_stats(self, stats: Dict[str, Any]) -> None:
        stats = dict(stats)
        stats.setdefault("total_ms", 0.0)
        self._last_retrieval_stats = stats
        logger.info(
            "[RetrieveMetrics] mode=%s query_len=%s top_k=%s dense=%s sparse=%s fused=%s returned=%s total_ms=%.2f rerank=%s advanced=%s",
            stats.get("mode", "sync"),
            stats.get("query_length", 0),
            stats.get("top_k", 0),
            stats.get("dense_hit_count", 0),
            stats.get("sparse_hit_count", 0),
            stats.get("fused_hit_count", 0),
            stats.get("result_count", 0),
            stats.get("total_ms", 0.0),
            stats.get("used_rerank", False),
            stats.get("used_advanced_expansion", False),
        )

    # 目前startup函数是冗余的，即不必要的，因为在创建RAGEngine实例时，自动创建了qdrant客户端，并确保了集合存在或被重新创建（_ensure_collection），未来startup可以做更多的预热或健康检查等启动逻辑。
    async def startup(self) -> None:
        if self._started:
            logger.warning("[RAGEngine] startup() 被重复调用，已忽略")
            return
        logger.info("[RAGEngine] 启动中...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.vector_store.collection_info)
        # collection_info函数会调用创建的qdrant客户端，返回collection中向量的数量
        self._started = True
        logger.info("[RAGEngine] 启动完成")

    def startup_sync(self) -> None:
        if self._started:
            return
        self.vector_store.collection_info()
        self._started = True
        logger.info("[RAGEngine] 同步启动完成")

    async def shutdown(self) -> None:
        logger.info("[RAGEngine] 正在关闭...")

        if self.contextual:
            try:
                self.contextual.close() # 这里关闭的是上下文增强功能的缓存，如redis
                logger.info("[RAGEngine] Contextual cache 已关闭")
            except Exception as exc:
                logger.warning(f"[RAGEngine] 关闭 Contextual cache 失败: {exc}")

        if self.reranker:
            await self.reranker.close()
            logger.info("[RAGEngine] Reranker HTTP 客户端已关闭")

        self._started = False
        logger.info("[RAGEngine] 已关闭")

    def index_file(
        self,
        file_path: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        display_source_name: Optional[str] = None,
        display_source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """索引单个文件。"""
        total_start = perf_counter() # 记录索引开始时间，perf_counter()比time.time()更精确，单位是秒
        resolved_input_path = str(Path(file_path).resolve())
        source_path = display_source_path or resolved_input_path
        source_file = display_source_name or Path(source_path).name or Path(resolved_input_path).name
        upload_time = datetime.now(timezone.utc).isoformat()
        stats: Dict[str, Any] = {
            "source_file": source_file,
            "source_path": source_path,
            "input_path": resolved_input_path,
            "force_reindex": force_reindex,
            "contextual_enabled": bool(self.contextual),
        } # input_path专门用来读，source_path是持久化存储的文件路径

        doc_id_start = perf_counter()  # 计算doc_id开始的时间
        doc_id = self._compute_doc_id(source_path) # 使用哈希函数计算doc_id，生成文档的唯一标识符
        stats["doc_id_ms"] = (perf_counter() - doc_id_start) * 1000 # 计算doc_id的耗时，单位是毫秒
        stats["doc_id"] = doc_id

        content_hash_start = perf_counter()
        content_hash = self._compute_content_hash(resolved_input_path) # 计算整个文档内容的哈希值
        stats["content_hash_ms"] = (perf_counter() - content_hash_start) * 1000
        stats["content_hash"] = content_hash

        exists_start = perf_counter()
        already_exists = self.vector_store.doc_exists(doc_id) # 检查文档是否已索引入向量数据库
        stats["doc_exists_check_ms"] = (perf_counter() - exists_start) * 1000
        stats["doc_exists"] = already_exists
        if not force_reindex and already_exists: # 如果没有设置强制重新索引，并且文档已经存在，跳过索引
            logger.info(f"[Index] 已跳过重复文件: {source_file} (doc_id={doc_id[:8]}...)")
            result = {
                "status": "skipped",
                "doc_id": doc_id,
                "content_hash": content_hash,
                "chunks": 0,
                "source_file": source_file,
                "source_path": source_path,
            }
            stats.update({
                "status": "skipped",
                "chunk_count": 0,
                "total_ms": (perf_counter() - total_start) * 1000, # 计算索引的总耗时
            })
            self._store_index_stats(stats)
            return result

        # 文档没有被索引过，开始解析文档内容
        logger.info(f"[Index] 开始解析: {source_file}")
        parse_start = perf_counter()
        try:
            md_text, file_type = self.parser.parse(resolved_input_path)
        except Exception as exc:
            logger.error(f"[Index] 解析失败: {exc}")
            stats.update({
                "status": "error",
                "error": str(exc),
                "parse_ms": (perf_counter() - parse_start) * 1000,
                "total_ms": (perf_counter() - total_start) * 1000,
            })
            self._store_index_stats(stats)
            return {
                "status": "error",
                "error": str(exc),
                "source_file": source_file,
                "source_path": source_path,
            }
        stats["parse_ms"] = (perf_counter() - parse_start) * 1000
        stats["file_type"] = file_type
        stats["parsed_chars"] = len(md_text)

        if not md_text.strip():
            logger.warning(f"[Index] 文件内容为空: {source_file}")
            stats.update({
                "status": "error",
                "error": "解析结果为空",
                "chunk_count": 0,
                "total_ms": (perf_counter() - total_start) * 1000,
            })
            self._store_index_stats(stats)
            return {
                "status": "error",
                "error": "解析结果为空",
                "source_file": source_file,
                "source_path": source_path,
            }

        # 解析完成后，开始进行文档切块
        logger.info(f"[Index] 开始切块: {source_file}")
        
        chunk_start = perf_counter()
        raw_chunks: List[Dict[str, Any]] = self.splitter.split(md_text, source_file=source_file) # 这里虽然上传了source_file，但当前切块器并未使用它
        stats["chunk_ms"] = (perf_counter() - chunk_start) * 1000
        stats["raw_chunk_count"] = len(raw_chunks)
        if not raw_chunks:
            stats.update({
                "status": "error",
                "error": "切块结果为空",
                "chunk_count": 0,
                "total_ms": (perf_counter() - total_start) * 1000,
            })
            self._store_index_stats(stats)
            return {
                "status": "error",
                "error": "切块结果为空",
                "source_file": source_file,
                "source_path": source_path,
            }

        chunks: List[DocumentChunk] = [
            DocumentChunk.create( # 这个文档快里使用UUID创建了chunk_id
                doc_id=doc_id,
                content=raw["content"],
                source_file=source_file,
                source_path=source_path,
                file_type=file_type,
                heading_path=raw["heading_path"],
                chunk_index=raw["chunk_index"],
                section_index=raw.get("section_index", -1),
                upload_time=upload_time,
                extra_meta=extra_meta or {}, # 用户上传文档时添加的额外信息
            )
            for raw in raw_chunks
        ]
        stats["chunk_count"] = len(chunks) # 其实这里chunk_count和raw_chunk_count一定是相等的

        contextual_start = perf_counter()
        if self.contextual:
            chunks = self.contextual.enrich_chunks(chunks) # 此时每个chunk里增加了一个属性context_prefix，描述了上下文信息
        stats["contextual_ms"] = (perf_counter() - contextual_start) * 1000

        # 切块完成，下一步进行向量嵌入
        logger.info(f"[Index] 开始向量嵌入: {source_file}")
        texts_for_embed = [chunk.full_text_for_embed for chunk in chunks] # 这里对chunk进行了丰富，包含了标题、上下文摘要（如果有context_prefix）和chunk原始内容，按照顺序拼接。
        stats["embedded_text_count"] = len(texts_for_embed)
        embed_start = perf_counter()
        try:
            dense_vectors: List[List[float]] = self.embedder.embed_all(texts_for_embed)
        except Exception as exc:
            logger.error(f"[Index] Embedding 失败: {exc}")
            stats.update({
                "status": "error",
                "error": f"Embedding 失败: {exc}",
                "embed_ms": (perf_counter() - embed_start) * 1000,
                "total_ms": (perf_counter() - total_start) * 1000,
            })
            self._store_index_stats(stats)
            return {
                "status": "error",
                "error": f"Embedding 失败: {exc}",
                "source_file": source_file,
                "source_path": source_path,
            }
        stats["embed_ms"] = (perf_counter() - embed_start) * 1000
        stats["dense_vector_count"] = len(dense_vectors)

        # 向量嵌入完成，下一步进行向量存储
        logger.info(f"[Index] 开始向量存储: {source_file}")
        upsert_start = perf_counter()
        self.vector_store.delete_by_source_path(source_path) # 设置了force_reindex=True，这里删除旧的向量
        self.vector_store.upsert(chunks, dense_vectors)
        stats["upsert_ms"] = (perf_counter() - upsert_start) * 1000

        result = {
            "status": "ok",
            "doc_id": doc_id,
            "content_hash": content_hash,
            "source_file": source_file,
            "source_path": source_path,
            "file_type": file_type,
            "chunks": len(chunks),
        }
        stats.update({
            "status": "ok",
            "total_ms": (perf_counter() - total_start) * 1000,
        })
        self._store_index_stats(stats)
        logger.info(f"[Index] 完成: {source_file}, chunks={len(chunks)}, type={file_type}")
        return result

    def index_directory(
        self,
        dir_path: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        glob_pattern: str = "**/*", # **/* 表示递归目录所有文件，包括嵌套目录下的文件
        force_reindex: bool = False,
    ) -> List[Dict[str, Any]]:
        """批量索引目录下的所有支持文件。"""
        supported_exts = self.parser.supported_extensions
        files = [
            path
            for path in Path(dir_path).glob(glob_pattern)
            if path.is_file() and path.suffix.lower() in supported_exts
        ]
        logger.info(f"[Index] 共发现 {len(files)} 个待索引文件...")
        results = []
        for file in tqdm(files, desc="IndexDir"):
            results.append(
                self.index_file(str(file), extra_meta=extra_meta, force_reindex=force_reindex)
            )
        return results

    def delete_file(self, file_identifier: str) -> None:
        """按逻辑 source_path 优先删除；必要时兼容旧的 basename 删除。"""
        if self._is_logical_source_path(file_identifier):
            self.vector_store.delete_by_source_path(file_identifier)
            logger.info(f"[Delete] 已按 logical source_path 删除: {file_identifier}")
            return

        candidate = Path(file_identifier)
        looks_like_path = candidate.is_absolute() or candidate.parent != Path(".")
        if looks_like_path:
            source_path = str(candidate.resolve(strict=False))
            self.vector_store.delete_by_source_path(source_path)
            logger.info(f"[Delete] 已按 source_path 删除: {source_path}")
            return

        self.vector_store.delete_by_source_file(file_identifier)
        logger.warning(f"[Delete] 已按 legacy source_file 删除: {file_identifier}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """执行 Dense + Sparse 检索，经 RRF 融合后可选重排。"""
        total_start = perf_counter()
        fetch_k = top_k * self.cfg.fetch_k_multiplier # 检索相关度最高的top_k个向量
        stats: Dict[str, Any] = {
            "mode": "sync",
            "query": query,
            "query_length": len(query),
            "top_k": top_k,
            "fetch_k": fetch_k,
            "score_threshold": score_threshold or self.cfg.score_threshold,
            "filter_conditions": deepcopy(filter_conditions) if filter_conditions else None, # 深拷贝，避免修改原始参数
            "skip_rerank": skip_rerank,
            "reranker_enabled": bool(self.reranker),
            "rag_mode": self.cfg.chunk.rag_mode,
        }

        embed_start = perf_counter()
        query_vec = self.embedder.embed_single(query) # 对查询进行稠密向量嵌入
        stats["query_embed_ms"] = (perf_counter() - embed_start) * 1000

        dense_start = perf_counter()
        dense_results = self.vector_store.search_dense(
            query_vector=query_vec,
            top_k=fetch_k,
            score_threshold=score_threshold or self.cfg.score_threshold,
            filter_conditions=filter_conditions,
        )
        stats["dense_ms"] = (perf_counter() - dense_start) * 1000
        stats["dense_hit_count"] = len(dense_results)

        sparse_start = perf_counter()
        sparse_results_raw = self.vector_store.search_sparse( # 检索top_k个稀疏向量
            query=query,
            top_k=fetch_k,
            filter_conditions=filter_conditions,
        )
        stats["sparse_ms"] = (perf_counter() - sparse_start) * 1000
        stats["sparse_hit_count"] = len(sparse_results_raw)
        sparse_results: List[Tuple[str, float]] = [
            (result["id"], result["score"]) for result in sparse_results_raw
        ] # 这里只保留检索结果的id和score

        fusion_start = perf_counter()
        fused = self._rrf_fusion(dense_results, sparse_results, fetch_k) # 对_dense_results和_sparse_results进行RRF融合
        stats["fusion_ms"] = (perf_counter() - fusion_start) * 1000
        stats["fused_hit_count"] = len(fused)
        if not fused:
            stats.update({
                "used_rerank": False,
                "used_advanced_expansion": False,
                "result_count": 0,
                "result_doc_ids": [],
                "total_ms": (perf_counter() - total_start) * 1000,
            })
            self._store_retrieval_stats(stats)
            return []

        # 进行重排
        rerank_start = perf_counter()
        used_rerank = False
        if not skip_rerank and self.reranker and len(fused) > 1:
            fused = self._rerank(query, fused, top_k)
            used_rerank = True
        else:
            fused = fused[:top_k]
        stats["rerank_ms"] = (perf_counter() - rerank_start) * 1000
        stats["used_rerank"] = used_rerank

        # 格式化检索结果
        format_start = perf_counter()
        results = self._format_results(fused)
        stats["format_ms"] = (perf_counter() - format_start) * 1000

        # 扩展语义节，如果配置为advanced模式，为每个chunk添加章节（同一section）的上下文信息
        section_expand_start = perf_counter()
        used_advanced_expansion = False
        if self.cfg.chunk.rag_mode == "advanced":
            results = self._expand_sections(results)
            used_advanced_expansion = True
        stats["section_expand_ms"] = (perf_counter() - section_expand_start) * 1000
        stats["used_advanced_expansion"] = used_advanced_expansion
        stats["result_count"] = len(results)
        stats["result_doc_ids"] = [result.get("doc_id", "") for result in results]
        stats["result_source_paths"] = [result.get("source_path", "") for result in results]
        stats["total_ms"] = (perf_counter() - total_start) * 1000
        self._store_retrieval_stats(stats)
        return results

    async def index_file_async(self, file_path: str, **kwargs: Any) -> Dict[str, Any]:
        """异步索引文件，返回索引结果。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.index_file(file_path, **kwargs))

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """异步执行 Dense + Sparse 检索、融合与重排。"""
        total_start = perf_counter()
        fetch_k = top_k * self.cfg.fetch_k_multiplier
        stats: Dict[str, Any] = {
            "mode": "async",
            "query": query,
            "query_length": len(query),
            "top_k": top_k,
            "fetch_k": fetch_k,
            "score_threshold": score_threshold or self.cfg.score_threshold,
            "filter_conditions": deepcopy(filter_conditions) if filter_conditions else None,
            "skip_rerank": skip_rerank,
            "reranker_enabled": bool(self.reranker),
            "rag_mode": self.cfg.chunk.rag_mode,
        }

        embed_start = perf_counter()
        query_vec = await self.embedder.embed_single_async(query)
        stats["query_embed_ms"] = (perf_counter() - embed_start) * 1000

        async def _dense() -> List[Dict[str, Any]]:
            """异步密集检索，返回密集检索结果。"""
            return await self.vector_store.async_search_dense(
                query_vector=query_vec,
                top_k=fetch_k,
                score_threshold=score_threshold or self.cfg.score_threshold,
                filter_conditions=filter_conditions,
            )

        async def _sparse() -> List[Dict[str, Any]]:
            """异步稀疏检索，返回稀疏检索结果。"""
            return await self.vector_store.async_search_sparse(
                query=query,
                top_k=fetch_k,
                filter_conditions=filter_conditions,
            )

        search_start = perf_counter()
        dense_results, sparse_results_raw = await asyncio.gather(_dense(), _sparse())
        stats["search_ms"] = (perf_counter() - search_start) * 1000
        stats["dense_hit_count"] = len(dense_results)
        stats["sparse_hit_count"] = len(sparse_results_raw)
        sparse_results: List[Tuple[str, float]] = [
            (result["id"], result["score"]) for result in sparse_results_raw
        ]

        fusion_start = perf_counter()
        fused = self._rrf_fusion(dense_results, sparse_results, fetch_k)
        stats["fusion_ms"] = (perf_counter() - fusion_start) * 1000
        stats["fused_hit_count"] = len(fused)
        if not fused:
            stats.update({
                "used_rerank": False,
                "used_advanced_expansion": False,
                "result_count": 0,
                "result_doc_ids": [],
                "total_ms": (perf_counter() - total_start) * 1000,
            })
            self._store_retrieval_stats(stats)
            return []

        rerank_start = perf_counter()
        used_rerank = False
        if not skip_rerank and self.reranker and len(fused) > 1:
            fused = await self._async_rerank(query, fused, top_k)
            used_rerank = True
        else:
            fused = fused[:top_k]
        stats["rerank_ms"] = (perf_counter() - rerank_start) * 1000
        stats["used_rerank"] = used_rerank

        format_start = perf_counter()
        results = self._format_results(fused)
        stats["format_ms"] = (perf_counter() - format_start) * 1000

        section_expand_start = perf_counter()
        used_advanced_expansion = False
        if self.cfg.chunk.rag_mode == "advanced":
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self._expand_sections, results)
            used_advanced_expansion = True
        stats["section_expand_ms"] = (perf_counter() - section_expand_start) * 1000
        stats["used_advanced_expansion"] = used_advanced_expansion
        stats["result_count"] = len(results)
        stats["result_doc_ids"] = [result.get("doc_id", "") for result in results]
        stats["result_source_paths"] = [result.get("source_path", "") for result in results]
        stats["total_ms"] = (perf_counter() - total_start) * 1000
        self._store_retrieval_stats(stats)
        return results

    def _expand_sections(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 doc_id + section_index 将命中 chunk 扩展为完整 section 文本。"""
        section_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for result in results:
            doc_id = result.get("doc_id", "")
            section_index = result.get("section_index", -1)
            if section_index == -1 or not doc_id:
                result["section_context"] = result["content"]
                result["section_chunk_count"] = 1
                continue

            cache_key = (doc_id, section_index)
            if cache_key not in section_cache:
                section_cache[cache_key] = self.vector_store.fetch_by_section(doc_id, section_index)

            section_chunks = section_cache[cache_key]
            if not section_chunks:
                result["section_context"] = result["content"]
                result["section_chunk_count"] = 1
                continue

            result["section_context"] = "\n\n".join(
                chunk["payload"].get("content", "") for chunk in section_chunks
            )
            result["section_chunk_count"] = len(section_chunks)

        logger.debug(f"[Retrieve] advanced 模式共扩展 {len(section_cache)} 个 section")
        return results

    def _rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Tuple[str, float]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """将密集检索结果和稀疏检索结果进行融合，返回融合后的结果。"""
        rrf_k = self.cfg.rrf_k
        scores: Dict[str, float] = {}
        id_to_payload: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            point_id = item["id"]
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (rrf_k + rank)
            payload = dict(item["payload"])
            payload["_dense_score"] = item["score"]
            id_to_payload[point_id] = payload

        missing_ids = [point_id for point_id, _ in sparse_results if point_id not in id_to_payload]
        if missing_ids:
            for match in self.vector_store.fetch_by_ids(missing_ids):
                id_to_payload[match["id"]] = match["payload"]

        for rank, (point_id, _bm25_score) in enumerate(sparse_results, start=1):
            if point_id in id_to_payload:
                scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (rrf_k + rank)

        sorted_ids = sorted(scores, key=lambda point_id: scores[point_id], reverse=True)[:top_k]
        return [
            {
                "id": point_id,
                "rrf_score": scores[point_id],
                "payload": id_to_payload.get(point_id, {}),
            }
            for point_id in sorted_ids
        ]

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """对候选文档进行重排。"""
        assert self.reranker is not None
        documents = self._build_rerank_docs(candidates)
        rerank_results = self.reranker.rerank(query, documents, top_n=top_k)
        if not rerank_results:
            logger.warning(f"[Reranker] 重排失败，query='{query[:30]}...'，回退 RRF 结果")
            return candidates[:top_k]
        return self._apply_rerank(candidates, rerank_results)

    async def _async_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """异步对候选文档进行重排。"""
        assert self.reranker is not None
        documents = self._build_rerank_docs(candidates)
        rerank_results = await self.reranker.async_rerank(query, documents, top_n=top_k)
        if not rerank_results:
            logger.warning(f"[Reranker] 异步重排失败，query='{query[:30]}...'，回退 RRF 结果")
            return candidates[:top_k]
        return self._apply_rerank(candidates, rerank_results)

    @staticmethod
    def _build_rerank_docs(candidates: List[Dict[str, Any]]) -> List[str]:
        """为候选文档添加标题前缀，提高重排效果。"""
        docs = []
        for item in candidates:
            heading_str = item["payload"].get("heading_str", "")
            content = item["payload"].get("content", "")
            docs.append(f"[{heading_str}]\n{content}" if heading_str else content)
        return docs

    @staticmethod
    def _apply_rerank(
        candidates: List[Dict[str, Any]],
        rerank_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """应用重排结果到候选文档，返回重排后的结果。"""
        reranked = []
        for result in rerank_results:
            idx = result.get("index", 0)
            if idx < len(candidates):
                item = candidates[idx].copy()
                item["rerank_score"] = result.get("relevance_score", 0.0)
                reranked.append(item)
        return reranked

    @staticmethod
    def _format_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化检索结果，保留必要的字段。"""
        output = []
        for item in items:
            payload = item.get("payload", {})
            output.append(
                {
                    "doc_id": payload.get("doc_id", ""),
                    "content": payload.get("content", ""),
                    "context_prefix": payload.get("context_prefix", ""),
                    "source_file": payload.get("source_file", ""),
                    "source_path": payload.get("source_path", ""),
                    "heading_str": payload.get("heading_str", ""),
                    "heading_path": payload.get("heading_path", []),
                    "chunk_index": payload.get("chunk_index", 0),
                    "section_index": payload.get("section_index", -1),
                    "upload_time": payload.get("upload_time", ""),
                    "score": item.get("rerank_score", item.get("rrf_score", 0.0)),
                    "rrf_score": item.get("rrf_score", 0.0),
                    "dense_score": payload.get("_dense_score", 0.0),
                }
            )
        return output

    def format_results_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """格式化检索结果，准备输入到 LLM。"""
        if not results:
            return "未检索到任何结果。"
        parts = []
        for index, result in enumerate(results, start=1):
            header = f"[{index}] 来源: {result['source_file']}"
            if result["heading_str"]:
                header += f" | 标题: {result['heading_str']}"
            body = result.get("section_context") or result["content"]
            parts.append(f"{header}\n{body}")
        return "\n\n---\n\n".join(parts)

    def collection_stats(self) -> Dict[str, Any]:
        """获取当前索引的统计信息。"""
        return self.vector_store.collection_info()

    def list_source_paths_by_source_file(self, source_file: str) -> List[str]:
        """根据 source文件路径获取所有文档路径。"""
        return self.vector_store.list_source_paths_by_source_file(source_file)

    @staticmethod
    def _is_logical_source_path(file_identifier: str) -> bool:
        """判断文件标识是否为逻辑源路径。"""
        return bool(_LOGICAL_SOURCE_URI_RE.match(file_identifier))

    @staticmethod
    def _compute_doc_id(source_path: str) -> str:
        """基于稳定 source_path 计算文档资源标识。"""
        return hashlib.sha256(source_path.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_content_hash(file_path: str) -> str:
        """基于文件内容计算稳定内容哈希，用于观测与幂等分析。"""
        digest = hashlib.sha256()
        with open(file_path, "rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()


def create_rag_engine(
    embed_provider: str = "proxy",
    embed_api_key: str = "",
    embed_base_url: str = "https://api.siliconflow.cn/v1",
    embed_model: str = "BAAI/bge-large-zh-v1.5",
    embed_dim: int = 1024,
    reranker_api_key: str = "",
    reranker_base_url: str = "https://api.siliconflow.cn/v1",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    qdrant_mode: str = "local",
    qdrant_path: str = "./qdrant_data",
    qdrant_collection: str = "rag_docs",
    use_contextual_retrieval: bool = False,
    rag_mode: str = "basic",
    **kwargs: Any,
) -> RAGEngine:
    from .config import ChunkConfig, EmbeddingConfig, QdrantConfig, RAGConfig, RerankerConfig

    cfg = RAGConfig(
        embedding=EmbeddingConfig(
            provider=embed_provider,
            proxy_api_key=embed_api_key,
            proxy_base_url=embed_base_url,
            proxy_model=embed_model,
            openai_api_key=embed_api_key,
            dimension=embed_dim,
        ),
        reranker=RerankerConfig(
            api_key=reranker_api_key,
            base_url=reranker_base_url,
            model=reranker_model,
        ),
        qdrant=QdrantConfig(
            mode=qdrant_mode,
            path=qdrant_path,
            collection_name=qdrant_collection,
        ),
        chunk=ChunkConfig(
            use_contextual_retrieval=use_contextual_retrieval,
            rag_mode=rag_mode,
            **{key: value for key, value in kwargs.items() if key in ChunkConfig.__dataclass_fields__},
        ),
    )
    return RAGEngine(cfg)