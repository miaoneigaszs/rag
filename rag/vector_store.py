"""
rag/vector_store.py
===================
Qdrant 向量存储封装。

稀疏编码器：
  原方案：手写 TF 归一化，无 IDF，停用词和关键词权重同级，精度低。
  新方案：纯 Python 内置的工业级 BM25 实现（TF 饱和截断 + 真实 IDF 加权）。
          利用 shelve 库将全局 IDF 统计（文件级词频映射）持久化到磁盘，
          支持冷启动热加载，增量更新。无需依赖外部 BM25 模型或库。
          可选依赖：pip install jieba （用于更精准的中文分词）

稀疏向量说明：
  - 通过 SparseVectorParams 在同一 Collection 中注册稀疏索引。
  - 入库时同时写入 dense vector（名称 "dense"）和 sparse vector（名称 "sparse"）。
  - 检索时分别做 Dense 搜索和 Sparse 搜索，再在 RAGEngine 层做 RRF 融合。
  - token_id 映射：MD5 哈希到 2^31 空间，与原方案一致，无需重建 collection。

支持三种部署模式（通过 QdrantConfig.mode 配置）：
  "local"  → QdrantClient(path=...)  本地文件，无需 Docker
  "docker" → QdrantClient(host=..., port=...)  本机 Docker
  "cloud"  → QdrantClient(url=..., api_key=...)  Qdrant Cloud
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shelve
import threading
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
# 稀疏编码器（BM25S 升级版）
# =============================================================================

class SparseEncoder:
    """
    BM25 稀疏向量编码器。

    核心改进（对比原 TF 方案）：
      - 真实 IDF 加权：doc_freq 越高的词权重越低，停用词自动降权
      - BM25 TF 饱和：长文档中高频词不再线性增长，更鲁棒
      - IDF 状态持久化：通过 shelve 文件跨重启保存，增量更新

    BM25 公式：
      score(t, d) = IDF(t) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d| / avgdl))
      IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)

    参数：
      k1   : TF 饱和系数，默认 1.5（范围 1.2~2.0）
      b    : 文档长度归一化系数，默认 0.75
      idf_path : IDF 状态持久化路径（shelve 文件前缀），通过环境变量
                 BM25_IDF_PATH 配置，默认 ./bm25_idf

    token_id 映射：MD5 哈希到 2^31，与原方案一致，Qdrant collection 无需重建。
    """

    # BM25 超参数
    K1: float = 1.5
    B: float = 0.75

    def __init__(self, idf_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._idf_path = idf_path or os.getenv("BM25_IDF_PATH", "./bm25_idf")

        # 从磁盘加载持久化 IDF 状态
        self._doc_freq: Dict[str, int] = {}   # token → 包含该 token 的文档数
        self._doc_count: int = 0              # 已索引文档总数
        self._avg_doc_len: float = 0.0        # 平均文档长度（词数）
        self._total_doc_len: int = 0          # 所有文档词数之和
        self._load_idf_state()

        logger.info(
            f"[SparseEncoder] BM25 编码器已就绪，"
            f"已索引文档数={self._doc_count}, idf_path={self._idf_path}"
        )

    # ------------------------------------------------------------------
    # 分词 & token_id
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """分词：中文用 jieba，其余用正则词级切分，过滤纯数字和单字符停用词。"""
        if _HAS_JIEBA:
            tokens = [t for t in jieba.cut(text) if t.strip()]
        else:
            tokens = re.findall(r"\w+", text.lower())
        # 过滤长度为 1 的无意义 token（标点残留等）
        return [t for t in tokens if len(t) > 1]

    @staticmethod
    def _token_id(token: str) -> int:
        """token 字符串 → 非负整数 id（MD5 稳定哈希，与原方案一致）。"""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % (2**31)

    # ------------------------------------------------------------------
    # IDF 状态持久化
    # ------------------------------------------------------------------

    def _load_idf_state(self) -> None:
        """从 shelve 文件加载 IDF 状态，文件不存在则初始化为空。"""
        try:
            with shelve.open(self._idf_path, flag="c") as db:
                self._doc_freq = dict(db.get("doc_freq", {}))
                self._doc_count = int(db.get("doc_count", 0))
                self._total_doc_len = int(db.get("total_doc_len", 0))
                self._avg_doc_len = (
                    self._total_doc_len / self._doc_count
                    if self._doc_count > 0 else 0.0
                )
        except Exception as exc:
            logger.warning(f"[SparseEncoder] 加载 IDF 状态失败，从空状态启动: {exc}")
            self._doc_freq = {}
            self._doc_count = 0
            self._total_doc_len = 0
            self._avg_doc_len = 0.0

    def _save_idf_state(self) -> None:
        """将当前 IDF 状态持久化到 shelve 文件。"""
        try:
            with shelve.open(self._idf_path, flag="c") as db:
                db["doc_freq"] = self._doc_freq
                db["doc_count"] = self._doc_count
                db["total_doc_len"] = self._total_doc_len
        except Exception as exc:
            logger.warning(f"[SparseEncoder] 持久化 IDF 状态失败: {exc}")

    # ------------------------------------------------------------------
    # IDF 增量更新（入库时调用）
    # ------------------------------------------------------------------

    def update_idf(self, texts: List[str]) -> None:
        """
        增量更新 IDF 统计，在批量入库前调用。

        Args:
            texts : 本批次所有文档的原始文本列表
        """
        with self._lock:
            for text in texts:
                tokens = self._tokenize(text)
                if not tokens:
                    continue
                # 每个 token 在本文档中只计一次 doc_freq
                for token in set(tokens):
                    self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
                self._doc_count += 1
                self._total_doc_len += len(tokens)

            self._avg_doc_len = (
                self._total_doc_len / self._doc_count
                if self._doc_count > 0 else 0.0
            )
            self._save_idf_state()

        logger.debug(
            f"[SparseEncoder] IDF 更新完成，文档总数={self._doc_count}, "
            f"词表大小={len(self._doc_freq)}, 平均文档长度={self._avg_doc_len:.1f}"
        )

    # ------------------------------------------------------------------
    # IDF 计算
    # ------------------------------------------------------------------

    def _idf(self, token: str) -> float:
        """
        Robertson-Spärck Jones IDF（BM25 标准公式）：
          ln((N - df + 0.5) / (df + 0.5) + 1)

        冷启动（doc_count == 0）时退化为 1.0，保证系统可用。
        """
        if self._doc_count == 0:
            return 1.0
        df = self._doc_freq.get(token, 0)
        N = self._doc_count
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    # ------------------------------------------------------------------
    # 编码（入库 & 查询共用）
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """
        将文本编码为 BM25 稀疏向量 (indices, values)。

        入库时：权重 = BM25(TF, IDF, doc_len, avgdl)
        查询时：因为查询通常很短（<20 词），avgdl 归一化影响可忽略，
                直接用 IDF 作为权重（等价于 BM25 查询侧标准做法）。

        Returns:
            indices : token id 列表（无重复）
            values  : 对应 BM25 权重（已归一化到 [0, 1]）
        """
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        doc_len = len(tokens)
        avgdl = self._avg_doc_len if self._avg_doc_len > 0 else doc_len

        # TF 统计
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        # BM25 权重计算
        seen_ids: Dict[int, float] = {}
        for token, freq in tf.items():
            idf = self._idf(token)
            # BM25 TF 饱和
            tf_norm = (freq * (self.K1 + 1)) / (
                freq + self.K1 * (1 - self.B + self.B * doc_len / avgdl)
            )
            weight = idf * tf_norm
            tid = self._token_id(token)
            # token_id 碰撞时取最大权重
            if tid in seen_ids:
                seen_ids[tid] = max(seen_ids[tid], weight)
            else:
                seen_ids[tid] = weight

        if not seen_ids:
            return [], []

        # 归一化到 [0, 1]（保持各 token 相对大小，方便 Qdrant 内积计算）
        max_w = max(seen_ids.values())
        indices = list(seen_ids.keys())
        values = [w / max_w for w in seen_ids.values()]
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
        idf_path = os.getenv(
            "BM25_IDF_PATH",
            str(Path(cfg.path) / "bm25_idf") if cfg.mode == "local" else "./bm25_idf"
        )
        self._sparse_encoder = SparseEncoder(idf_path=idf_path)
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
        # 入库前先更新 IDF 统计（BM25 需要语料级 IDF，增量更新后持久化）
        self._sparse_encoder.update_idf([c.content for c in chunks])

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