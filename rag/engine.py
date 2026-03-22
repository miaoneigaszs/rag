"""
rag/engine.py
=============
RAG 主引擎。

入库流程：
  File → Parser → Splitter → [ContextualRetrieval] → Embed → Qdrant(dense + sparse)

检索流程（同步）：
  Query → Embed → Dense(Qdrant) ──┐
               → Sparse(Qdrant) ──┴→ RRF → [Reranker] → TopK → [Section扩展]

检索流程（异步）：
  Query → Embed(async) → Dense(AsyncQdrant) ──┐  ← asyncio.gather
                       → Sparse(AsyncQdrant) ──┴→ RRF → Reranker(async) → [Section扩展]

RAG 模式（通过 ChunkConfig.rag_mode 配置）：
  "basic"    : 普通 RAG，返回命中 chunk 本身
  "advanced" : 高级 RAG，检索后按 section_index 拉取同 section 所有 chunk，
               拼合为完整 section 文本作为 LLM 上下文（Parent Document Retrieval）

生命周期管理：
  框架集成：
    FastAPI  → @asynccontextmanager lifespan：await engine.startup() / await engine.shutdown()
    脚本     → engine.startup_sync() / asyncio.run(engine.shutdown())
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
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


class RAGEngine:
    """
    工业级 RAG 主引擎。

    使用示例（脚本）：
        engine = create_rag_engine(
            embed_api_key="...",
            reranker_api_key="...",
            rag_mode="advanced",        # 开启高级 RAG
        )
        engine.startup_sync()
        engine.index_file("doc.pdf")
        hits = engine.retrieve("如何配置环境变量？")
        asyncio.run(engine.shutdown())

    使用示例（FastAPI）：
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await engine.startup()
            yield
            await engine.shutdown()
    """

    def __init__(self, cfg: Optional[RAGConfig] = None) -> None:
        self.cfg = cfg or RAGConfig()

        self.parser = DocumentParser()
        self.splitter = HierarchicalMarkdownSplitter(self.cfg.chunk)
        self.embedder = EmbeddingService(self.cfg.embedding)
        self.vector_store = QdrantVectorStore(self.cfg.qdrant, self.cfg.embedding.dimension)
        self.reranker: Optional[APIReranker] = (
            APIReranker(self.cfg.reranker) if self.cfg.reranker.api_key else None
        )
        self.contextual: Optional[ContextualRetrieval] = (
            ContextualRetrieval(self.cfg) if self.cfg.chunk.use_contextual_retrieval else None
        )
        self._started = False

    # =========================================================================
    # 生命周期
    # =========================================================================

    async def startup(self) -> None:
        if self._started:
            logger.warning("[RAGEngine] startup() 已被调用过，跳过重复初始化")
            return
        logger.info("[RAGEngine] 启动中...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.vector_store.collection_info)
        self._started = True
        logger.info("[RAGEngine] 启动完成")

    def startup_sync(self) -> None:
        if self._started:
            return
        self.vector_store.collection_info()
        self._started = True
        logger.info("[RAGEngine] 同步启动完成")

    async def shutdown(self) -> None:
        logger.info("[RAGEngine] 开始关闭...")

        if self.contextual:
            try:
                self.contextual.close()
                logger.info("[RAGEngine] Contextual 缓存已关闭")
            except Exception as exc:
                logger.warning(f"[RAGEngine] 关闭 Contextual 缓存失败: {exc}")

        if self.reranker:
            await self.reranker.close()
            logger.info("[RAGEngine] Reranker HTTP 连接池已关闭")

        self._started = False
        logger.info("[RAGEngine] 已关闭")

    # =========================================================================
    # 核心入库接口
    # =========================================================================

    def index_file(
        self,
        file_path: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        索引单个文件。

        Returns:
            {"status": "ok"|"skipped"|"error", "chunks": int, "doc_id": str, ...}
        """
        file_path = str(Path(file_path).resolve())
        source_file = Path(file_path).name
        upload_time = datetime.now(timezone.utc).isoformat()

        doc_id = self._compute_doc_hash(file_path)
        if not force_reindex and self.vector_store.doc_exists(doc_id):
            logger.info(f"[Index] 文件已存在，跳过: {source_file} (doc_id={doc_id[:8]}...)")
            return {"status": "skipped", "doc_id": doc_id, "chunks": 0, "source_file": source_file}

        logger.info(f"[Index] 开始处理: {source_file}")

        try:
            md_text, file_type = self.parser.parse(file_path)
        except Exception as exc:
            logger.error(f"[Index] 解析失败: {exc}")
            return {"status": "error", "error": str(exc), "source_file": source_file}

        if not md_text.strip():
            logger.warning(f"[Index] 解析结果为空: {source_file}")
            return {"status": "error", "error": "文档内容为空", "source_file": source_file}

        raw_chunks = self.splitter.split(md_text, source_file=source_file)
        if not raw_chunks:
            return {"status": "error", "error": "切块结果为空", "source_file": source_file}

        chunks: List[DocumentChunk] = [
            DocumentChunk.create(
                doc_id=doc_id,
                content=raw["content"],
                source_file=source_file,
                file_type=file_type,
                heading_path=raw["heading_path"],
                chunk_index=raw["chunk_index"],
                section_index=raw.get("section_index", -1),
                upload_time=upload_time,
                extra_meta=extra_meta or {},
            )
            for raw in raw_chunks
        ]

        if self.contextual:
            chunks = self.contextual.enrich_chunks(chunks)

        texts_for_embed = [c.full_text_for_embed for c in chunks]
        try:
            dense_vectors = self.embedder.embed_all(texts_for_embed)
        except Exception as exc:
            logger.error(f"[Index] Embedding 失败: {exc}")
            return {"status": "error", "error": f"Embedding 失败: {exc}", "source_file": source_file}

        self.vector_store.upsert(chunks, dense_vectors)

        result = {
            "status": "ok",
            "doc_id": doc_id,
            "source_file": source_file,
            "file_type": file_type,
            "chunks": len(chunks),
        }
        logger.info(f"[Index] 完成: {source_file}, chunks={len(chunks)}, type={file_type}")
        return result

    def index_directory(
        self,
        dir_path: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        glob_pattern: str = "**/*",
        force_reindex: bool = False,
    ) -> List[Dict[str, Any]]:
        """批量索引目录下所有支持的文件。"""
        supported_exts = self.parser.supported_extensions
        files = [
            p
            for p in Path(dir_path).glob(glob_pattern)
            if p.is_file() and p.suffix.lower() in supported_exts
        ]
        logger.info(f"[Index] 发现 {len(files)} 个文件，开始批量处理...")
        results = []
        for f in tqdm(files, desc="IndexDir"):
            results.append(
                self.index_file(str(f), extra_meta=extra_meta, force_reindex=force_reindex)
            )
        return results

    def delete_file(self, source_file: str) -> None:
        """删除某文件的所有 chunk。"""
        self.vector_store.delete_by_source_file(source_file)
        logger.info(f"[Delete] 已删除: {source_file}")

    # =========================================================================
    # 核心检索接口（同步）
    # =========================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        主检索接口（同步）：Dense + Sparse → RRF → Reranker → TopK → [Section扩展]。

        basic 模式   : 直接返回 top_k 个命中 chunk
        advanced 模式: 对每个命中 chunk 拉取其所在 section 的全部 chunk，
                       拼合为 section_context 字段追加到结果中，
                       LLM 调用时使用 section_context 而非单个 content
        """
        fetch_k = top_k * self.cfg.fetch_k_multiplier

        query_vec = self.embedder.embed_single(query)
        dense_results = self.vector_store.search_dense(
            query_vector=query_vec,
            top_k=fetch_k,
            score_threshold=score_threshold or self.cfg.score_threshold,
            filter_conditions=filter_conditions,
        )
        sparse_results = self.vector_store.search_sparse(
            query=query,
            top_k=fetch_k,
            filter_conditions=filter_conditions,
        )

        fused = self._rrf_fusion(dense_results, sparse_results, fetch_k)
        if not fused:
            return []

        if not skip_rerank and self.reranker and len(fused) > 1:
            fused = self._rerank(query, fused, top_k)
        else:
            fused = fused[:top_k]

        results = self._format_results(fused)

        # ── 高级 RAG：section 扩展 ────────────────────────────────────────
        if self.cfg.chunk.rag_mode == "advanced":
            results = self._expand_sections(results)

        return results

    # =========================================================================
    # 核心检索接口（异步）
    # =========================================================================

    async def index_file_async(self, file_path: str, **kwargs: Any) -> Dict[str, Any]:
        """异步索引（CPU/IO 密集，用线程池隔离，不阻塞事件循环）。"""
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
        """
        主检索接口（异步）：Dense 和 Sparse 并发，Reranker 异步。
        advanced 模式下 section 扩展通过线程池隔离（Qdrant scroll 为同步调用）。
        """
        fetch_k = top_k * self.cfg.fetch_k_multiplier

        query_vec = await self.embedder.embed_single_async(query)

        async def _dense() -> List[Dict[str, Any]]:
            return await self.vector_store.async_search_dense(
                query_vector=query_vec,
                top_k=fetch_k,
                score_threshold=score_threshold or self.cfg.score_threshold,
                filter_conditions=filter_conditions,
            )

        async def _sparse() -> List[Dict[str, Any]]:
            return await self.vector_store.async_search_sparse(
                query=query,
                top_k=fetch_k,
                filter_conditions=filter_conditions,
            )

        dense_results, sparse_results_raw = await asyncio.gather(_dense(), _sparse())

        sparse_results: List[Tuple[str, float]] = [
            (r["id"], r["score"]) for r in sparse_results_raw
        ]

        fused = self._rrf_fusion(dense_results, sparse_results, fetch_k)
        if not fused:
            return []

        if not skip_rerank and self.reranker and len(fused) > 1:
            fused = await self._async_rerank(query, fused, top_k)
        else:
            fused = fused[:top_k]

        results = self._format_results(fused)

        # ── 高级 RAG：section 扩展（线程池隔离同步 scroll）────────────────
        if self.cfg.chunk.rag_mode == "advanced":
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self._expand_sections, results)

        return results

    # =========================================================================
    # Section 扩展（高级 RAG 核心）
    # =========================================================================

    def _expand_sections(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对每个检索结果，按 source_file + section_index 拉取同 section 的所有 chunk，
        拼合为 section_context 字段追加到结果中。

        去重策略：同一 (source_file, section_index) 只拉取一次，避免重复 scroll。

        结果新增字段：
          section_context : str   同 section 所有 chunk 按顺序拼合的完整文本
                                  （LLM 调用时替代单个 content 使用）
          section_chunk_count : int  该 section 共有多少个 chunk
        """
        # 缓存已拉取的 section，避免重复 scroll
        section_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

        for result in results:
            source_file = result.get("source_file", "")
            section_index = result.get("section_index", -1)

            # section_index == -1 表示无 section 信息，跳过扩展
            if section_index == -1:
                result["section_context"] = result["content"]
                result["section_chunk_count"] = 1
                continue

            cache_key = (source_file, section_index)
            if cache_key not in section_cache:
                section_chunks = self.vector_store.fetch_by_section(source_file, section_index)
                section_cache[cache_key] = section_chunks

            section_chunks = section_cache[cache_key]

            if not section_chunks:
                result["section_context"] = result["content"]
                result["section_chunk_count"] = 1
            else:
                # 拼合 section 内所有 chunk，保持原始顺序
                section_text = "\n\n".join(
                    c["payload"].get("content", "") for c in section_chunks
                )
                result["section_context"] = section_text
                result["section_chunk_count"] = len(section_chunks)

        logger.debug(
            f"[Retrieve] advanced 模式 section 扩展完成，"
            f"涉及 {len(section_cache)} 个唯一 section"
        )
        return results

    # =========================================================================
    # RRF 融合
    # =========================================================================

    def _rrf_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Tuple[str, float]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion。score(doc) = Σ 1 / (k + rank_i)"""
        rrf_k = self.cfg.rrf_k
        scores: Dict[str, float] = {}
        id_to_payload: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            id_to_payload[doc_id] = item["payload"]
            id_to_payload[doc_id]["_dense_score"] = item["score"]

        missing_ids = [sid for sid, _ in sparse_results if sid not in id_to_payload]
        if missing_ids:
            for m in self.vector_store.fetch_by_ids(missing_ids):
                id_to_payload[m["id"]] = m["payload"]

        for rank, (doc_id, _bm25_score) in enumerate(sparse_results, start=1):
            if doc_id in id_to_payload:
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [
            {
                "id": doc_id,
                "rrf_score": scores[doc_id],
                "payload": id_to_payload.get(doc_id, {}),
            }
            for doc_id in sorted_ids
        ]

    # =========================================================================
    # Rerank（同步 / 异步）
    # =========================================================================

    def _rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        assert self.reranker is not None
        documents = self._build_rerank_docs(candidates)
        rerank_results = self.reranker.rerank(query, documents, top_n=top_k)
        if not rerank_results:
            logger.warning(
                f"[Reranker] 返回空结果（query='{query[:30]}...'），降级为 RRF 排序。"
            )
            return candidates[:top_k]
        return self._apply_rerank(candidates, rerank_results)

    async def _async_rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        assert self.reranker is not None
        documents = self._build_rerank_docs(candidates)
        rerank_results = await self.reranker.async_rerank(query, documents, top_n=top_k)
        if not rerank_results:
            logger.warning(
                f"[Reranker] 异步返回空结果（query='{query[:30]}...'），降级为 RRF 排序。"
            )
            return candidates[:top_k]
        return self._apply_rerank(candidates, rerank_results)

    @staticmethod
    def _build_rerank_docs(candidates: List[Dict[str, Any]]) -> List[str]:
        """将候选集格式化为 Reranker 接受的文档列表。"""
        docs = []
        for item in candidates:
            heading_str = item["payload"].get("heading_str", "")
            content = item["payload"].get("content", "")
            docs.append(f"[{heading_str}]\n{content}" if heading_str else content)
        return docs

    @staticmethod
    def _apply_rerank(
        candidates: List[Dict[str, Any]], rerank_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        reranked = []
        for r in rerank_results:
            idx = r.get("index", 0)
            if idx < len(candidates):
                item = candidates[idx].copy()
                item["rerank_score"] = r.get("relevance_score", 0.0)
                reranked.append(item)
        return reranked

    # =========================================================================
    # 格式化输出
    # =========================================================================

    @staticmethod
    def _format_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将内部结果格式化为统一输出结构。"""
        output = []
        for item in items:
            payload = item.get("payload", {})
            output.append(
                {
                    "content": payload.get("content", ""),
                    "context_prefix": payload.get("context_prefix", ""),
                    "source_file": payload.get("source_file", ""),
                    "heading_str": payload.get("heading_str", ""),
                    "heading_path": payload.get("heading_path", []),
                    "chunk_index": payload.get("chunk_index", 0),
                    "section_index": payload.get("section_index", -1),
                    "upload_time": payload.get("upload_time", ""),
                    "score": item.get("rerank_score", item.get("rrf_score", 0.0)),
                    "rrf_score": item.get("rrf_score", 0.0),
                    "dense_score": payload.get("_dense_score", 0.0),
                    # section_context 由 _expand_sections 填充（advanced 模式）
                }
            )
        return output

    def format_results_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """
        将检索结果格式化为适合送入 LLM 的上下文字符串。

        advanced 模式下优先使用 section_context（完整 section 文本），
        basic 模式下使用 content（单个 chunk）。
        """
        if not results:
            return "未检索到相关内容。"
        parts = []
        for i, r in enumerate(results, start=1):
            header = f"[{i}] 文件: {r['source_file']}"
            if r["heading_str"]:
                header += f" | 章节: {r['heading_str']}"
            header += f" | 相关度: {r['score']:.4f}"

            # advanced 模式使用完整 section 文本，basic 模式使用单 chunk
            body = r.get("section_context") or r["content"]
            parts.append(f"{header}\n{body}")
        return "\n\n---\n\n".join(parts)

    # =========================================================================
    # 工具方法
    # =========================================================================

    def collection_stats(self) -> Dict[str, Any]:
        return self.vector_store.collection_info()

    @staticmethod
    def _compute_doc_hash(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# =============================================================================
# 便捷工厂函数
# =============================================================================

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
    """
    快速构建 RAGEngine 的工厂函数。

    示例（普通 RAG）：
        engine = create_rag_engine(embed_api_key="sf-xxxxx")

    示例（高级 RAG，section 扩展 + Contextual Retrieval）：
        engine = create_rag_engine(
            embed_api_key="sf-xxxxx",
            reranker_api_key="sf-xxxxx",
            rag_mode="advanced",
            use_contextual_retrieval=True,
        )
    """
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
            **{k: v for k, v in kwargs.items() if k in ChunkConfig.__dataclass_fields__},
        ),
    )
    return RAGEngine(cfg)