"""
rag/vector_store.py
===================
Qdrant 向量存储封装。

核心改动（替换 BM25）：
  原方案：内存级 rank_bm25，50k 上限，启动需 scroll 全量，删除需全量重建。
  新方案：Qdrant 原生 Sparse Vector（SPLADE / BM25 稀疏模型），
          持久化、可扩展、与 Dense 同库，零额外服务。

稀疏向量说明：
  - 通过 SparseVectorParams 在同一 Collection 中注册稀疏索引。
  - 入库时同时写入 dense vector（名称 "dense"）和 sparse vector（名称 "sparse"）。
  - 检索时分别做 Dense 搜索和 Sparse 搜索，再在 RAGEngine 层做 RRF 融合。
  - 稀疏向量的稀疏表示由 SparseEncoder 生成（见下方）。

支持三种部署模式（通过 QdrantConfig.mode 配置）：
  "local"  → QdrantClient(path=...)  本地文件，无需 Docker
  "docker" → QdrantClient(host=..., port=...)  本机 Docker
  "cloud"  → QdrantClient(url=..., api_key=...)  Qdrant Cloud
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from .config import QdrantConfig
from .models import DocumentChunk

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


# =============================================================================
# 稀疏编码器（替代 rank_bm25）
# =============================================================================

class SparseEncoder:
    """
    轻量级稀疏向量编码器（BM25 词频统计变体）。

    将文本转为 {token_id: weight} 的稀疏表示，直接写入 Qdrant Sparse Vector。

    说明：
      这里使用词频（TF）+ 简单 IDF 近似作为权重，适合单机场景。
      生产环境可替换为 SPLADE / 专用稀疏模型（如 naver/splade-cocondenser-ensembledistil），
      只需改写 encode() 方法，接口不变。

    token_id 映射：使用 Python hash() 对 token 字符串取正整数，
    碰撞概率极低（2^61 空间），Qdrant Sparse Vector 对 id 范围无限制。
    """

    def __init__(self) -> None:
        self._idf_cache: Dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """分词：中文用 jieba，其余用正则词级切分。"""
        if _HAS_JIEBA:
            return [t for t in jieba.cut(text) if t.strip()]
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _token_id(token: str) -> int:
        """将 token 字符串映射到非负整数 id（使用稳定哈希）。"""
        import hashlib
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % (2**31)

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """
        将文本编码为稀疏向量 (indices, values)。

        Returns:
            indices : token id 列表（无重复）
            values  : 对应权重（词频，归一化到 [0, 1]）
        """
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        max_freq = max(tf.values())
        seen_ids: Dict[int, float] = {}
        for token, freq in tf.items():
            tid = self._token_id(token)
            weight = freq / max_freq
            if tid in seen_ids:
                seen_ids[tid] = max(seen_ids[tid], weight)
            else:
                seen_ids[tid] = weight

        indices = list(seen_ids.keys())
        values = list(seen_ids.values())
        return indices, values


# =============================================================================
# Qdrant 向量存储
# =============================================================================

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorStore:
    """
    Qdrant 向量存储封装（Dense + Sparse 双路）。

    同时持有同步 client 和异步 async_client：
      - 同步 client：入库、管理操作
      - 异步 client：async_search_dense / async_search_sparse（供检索路径使用）
    """

    def __init__(self, cfg: QdrantConfig, embed_dim: int) -> None:
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.collection = cfg.collection_name
        self._client = self._build_client(cfg)
        self._async_client = self._build_async_client(cfg)
        self._sparse_encoder = SparseEncoder()
        self._ensure_collection()

    # ------------------------------------------------------------------
    # 客户端构建
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(cfg: QdrantConfig) -> QdrantClient:
        if cfg.mode == "local":
            Path(cfg.path).mkdir(parents=True, exist_ok=True)
            logger.info(f"[Qdrant] 本地文件模式: {cfg.path}")
            return QdrantClient(path=cfg.path)
        if cfg.mode == "docker":
            logger.info(f"[Qdrant] Docker 模式: {cfg.host}:{cfg.port}")
            return QdrantClient(host=cfg.host, port=cfg.port)
        logger.info(f"[Qdrant] Cloud 模式: {cfg.url}")
        return QdrantClient(url=cfg.url, api_key=cfg.api_key)

    @staticmethod
    def _build_async_client(cfg: QdrantConfig) -> AsyncQdrantClient:
        if cfg.mode == "local":
            return AsyncQdrantClient(path=cfg.path)
        if cfg.mode == "docker":
            return AsyncQdrantClient(host=cfg.host, port=cfg.port)
        return AsyncQdrantClient(url=cfg.url, api_key=cfg.api_key)

    # ------------------------------------------------------------------
    # Collection 初始化
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """如果 collection 不存在则创建，存在则复用。"""
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection in existing:
            logger.info(f"[Qdrant] 复用 collection: {self.collection}")
            return

        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(
                    size=self.embed_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
        )

        # payload 索引：常用过滤字段 + section 扩展检索所需字段
        for field_name, schema in [
            ("source_file", PayloadSchemaType.KEYWORD),
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("heading_str", PayloadSchemaType.KEYWORD),
            ("section_index", PayloadSchemaType.INTEGER),   # 高级 RAG section 扩展检索
        ]:
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema,
            )

        logger.info(
            f"[Qdrant] 创建 collection: {self.collection}, "
            f"dense_dim={self.embed_dim}, sparse=enabled"
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def upsert(
        self,
        chunks: List[DocumentChunk],
        dense_vectors: List[List[float]],
        batch_size: int = 64,
    ) -> None:
        """
        批量 upsert（同时写入 dense + sparse vector）。

        Args:
            chunks        : DocumentChunk 列表
            dense_vectors : 与 chunks 一一对应的 dense 向量
            batch_size    : 每批写入条数
        """
        points: List[PointStruct] = []
        for chunk, dense_vec in zip(chunks, dense_vectors):
            sparse_indices, sparse_values = self._sparse_encoder.encode(chunk.content)
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        _DENSE_VECTOR_NAME: dense_vec,
                        _SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_indices, values=sparse_values
                        ),
                    },
                    payload=chunk.to_payload(),
                )
            )

        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self.collection, points=points[i : i + batch_size]
            )

        logger.info(f"[Qdrant] upsert 完成，共 {len(points)} 条（dense + sparse）")

    # ------------------------------------------------------------------
    # 同步检索
    # ------------------------------------------------------------------

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Dense 向量检索（同步）。"""
        results = self._client.search(
            collection_name=self.collection,
            query_vector=NamedVector(name=_DENSE_VECTOR_NAME, vector=query_vector),
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=self._build_filter(filter_conditions),
            with_payload=True,
        )
        return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]

    def search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Sparse 向量检索（同步）。"""
        indices, values = self._sparse_encoder.encode(query)
        if not indices:
            return []
        results = self._client.search(
            collection_name=self.collection,
            query_vector=NamedSparseVector(
                name=_SPARSE_VECTOR_NAME,
                vector=SparseVector(indices=indices, values=values),
            ),
            limit=top_k,
            query_filter=self._build_filter(filter_conditions),
            with_payload=True,
        )
        return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]

    def fetch_by_section(
        self,
        source_file: str,
        section_index: int,
    ) -> List[Dict[str, Any]]:
        """
        高级 RAG：按 source_file + section_index 拉取该 section 的所有 chunk，
        按 chunk_index 升序排列，用于检索后扩展上下文。

        Args:
            source_file   : 文件名（payload 中的 source_file 字段）
            section_index : 语义节序号（payload 中的 section_index 字段）

        Returns:
            按 chunk_index 升序排列的 chunk payload 列表
        """
        if section_index == -1:
            return []

        results, _ = self._client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source_file", match=MatchValue(value=source_file)),
                    FieldCondition(key="section_index", match=MatchValue(value=section_index)),
                ]
            ),
            limit=200,          # 单 section 不应超过 200 个 chunk
            with_payload=True,
            with_vectors=False,
        )
        # 按 chunk_index 升序排列，保证文本顺序
        sorted_results = sorted(
            results,
            key=lambda r: r.payload.get("chunk_index", 0) if r.payload else 0,
        )
        return [{"id": str(r.id), "payload": r.payload} for r in sorted_results]

    # ------------------------------------------------------------------
    # 异步检索
    # ------------------------------------------------------------------

    async def async_search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Dense 向量检索（异步，使用 AsyncQdrantClient）。"""
        results = await self._async_client.search(
            collection_name=self.collection,
            query_vector=NamedVector(name=_DENSE_VECTOR_NAME, vector=query_vector),
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=self._build_filter(filter_conditions),
            with_payload=True,
        )
        return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]

    async def async_search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Sparse 向量检索（异步）。"""
        indices, values = self._sparse_encoder.encode(query)
        if not indices:
            return []
        results = await self._async_client.search(
            collection_name=self.collection,
            query_vector=NamedSparseVector(
                name=_SPARSE_VECTOR_NAME,
                vector=SparseVector(indices=indices, values=values),
            ),
            limit=top_k,
            query_filter=self._build_filter(filter_conditions),
            with_payload=True,
        )
        return [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in results]

    # ------------------------------------------------------------------
    # 管理接口
    # ------------------------------------------------------------------

    def fetch_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """按 ID 批量拉取 payload。"""
        results = self._client.retrieve(
            collection_name=self.collection, ids=ids, with_payload=True
        )
        return [{"id": str(r.id), "payload": r.payload} for r in results]

    def doc_exists(self, doc_id: str) -> bool:
        """检查某 doc_id 的文档是否已入库（用于去重）。"""
        results, _ = self._client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=False,
        )
        return len(results) > 0

    def delete_by_doc_id(self, doc_id: str) -> None:
        """删除某文档的所有 chunk（按 doc_id）。"""
        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        logger.info(f"[Qdrant] 删除 doc_id={doc_id[:8]}...")

    def delete_by_source_file(self, source_file: str) -> None:
        """按文件名删除所有相关 chunk。"""
        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
        )
        logger.info(f"[Qdrant] 删除 source_file={source_file}")

    def collection_info(self) -> Dict[str, Any]:
        """返回 collection 基本统计信息。"""
        info = self._client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(conditions: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """将简单 dict 条件转为 Qdrant Filter 对象。"""
        if not conditions:
            return None
        must_clauses = []
        for key, value in conditions.items():
            if isinstance(value, list):
                must_clauses.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                must_clauses.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must_clauses) if must_clauses else None