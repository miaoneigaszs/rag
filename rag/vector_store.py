"""Qdrant 向量存储与 BM25 稀疏编码。"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import shelve
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
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
    )
    _HAS_QDRANT = True
except ImportError:
    AsyncQdrantClient = None  # type: ignore[assignment]
    QdrantClient = None  # type: ignore[assignment]
    Distance = FieldCondition = Filter = HnswConfigDiff = MatchAny = MatchValue = None  # type: ignore[assignment]
    NamedSparseVector = NamedVector = OptimizersConfigDiff = PayloadSchemaType = None  # type: ignore[assignment]
    PointStruct = SparseIndexParams = SparseVector = SparseVectorParams = VectorParams = None  # type: ignore[assignment]
    _HAS_QDRANT = False

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


class SparseEncoder:
    """支持持久化 IDF 状态的轻量 BM25 编码器。"""

    K1: float = 1.5
    B: float = 0.75

    def __init__(self, idf_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._idf_path = idf_path or os.getenv("BM25_IDF_PATH", "./bm25_idf")
        self._doc_freq: Dict[str, int] = {}
        self._doc_count = 0
        self._avg_doc_len = 0.0
        self._total_doc_len = 0
        self._load_idf_state()
        logger.info(
            f"[SparseEncoder] 已加载 BM25 统计，doc_count={self._doc_count}, idf_path={self._idf_path}"
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if _HAS_JIEBA:
            tokens = [token for token in jieba.cut(text) if token.strip()]
        else:
            tokens = re.findall(r"\w+", text.lower())
        return [token for token in tokens if len(token) > 1]

    @staticmethod
    def _token_id(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % (2**31)

    def _load_idf_state(self) -> None:
        try:
            with shelve.open(self._idf_path, flag="c") as db:
                self._doc_freq = dict(db.get("doc_freq", {}))
                self._doc_count = int(db.get("doc_count", 0))
                self._total_doc_len = int(db.get("total_doc_len", 0))
                self._avg_doc_len = (
                    self._total_doc_len / self._doc_count if self._doc_count > 0 else 0.0
                )
        except Exception as exc:
            logger.warning(f"[SparseEncoder] 加载 IDF 状态失败，将从空状态开始: {exc}")
            self._doc_freq = {}
            self._doc_count = 0
            self._total_doc_len = 0
            self._avg_doc_len = 0.0

    def _save_idf_state(self) -> None:
        try:
            with shelve.open(self._idf_path, flag="c") as db:
                db["doc_freq"] = self._doc_freq
                db["doc_count"] = self._doc_count
                db["total_doc_len"] = self._total_doc_len
        except Exception as exc:
            logger.warning(f"[SparseEncoder] 保存 IDF 状态失败: {exc}")

    def _apply_texts(self, texts: List[str], direction: int) -> None:
        for text in texts:
            tokens = self._tokenize(text)
            if not tokens:
                continue
            for token in set(tokens):
                new_value = self._doc_freq.get(token, 0) + direction
                if new_value <= 0:
                    self._doc_freq.pop(token, None)
                else:
                    self._doc_freq[token] = new_value
            self._doc_count = max(0, self._doc_count + direction)
            self._total_doc_len = max(0, self._total_doc_len + direction * len(tokens))
        self._avg_doc_len = self._total_doc_len / self._doc_count if self._doc_count > 0 else 0.0
        self._save_idf_state()

    def update_idf(self, texts: List[str]) -> None:
        """更新 IDF 统计信息。"""
        with self._lock:
            self._apply_texts(texts, direction=1)
        logger.debug(
            f"[SparseEncoder] IDF 已更新，doc_count={self._doc_count}, vocab={len(self._doc_freq)}, avgdl={self._avg_doc_len:.1f}"
        )

    def remove_idf(self, texts: List[str]) -> None:
        """回滚（移除）给定文本对 IDF 的影响。"""
        with self._lock:
            self._apply_texts(texts, direction=-1)
        logger.debug(
            f"[SparseEncoder] IDF 已回滚，doc_count={self._doc_count}, vocab={len(self._doc_freq)}, avgdl={self._avg_doc_len:.1f}"
        )

    def _idf(self, token: str) -> float:
        if self._doc_count == 0:
            return 1.0
        df = self._doc_freq.get(token, 0)
        return math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """将文本编码为稀疏向量 (indices, values)。"""
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        doc_len = len(tokens)
        avgdl = self._avg_doc_len if self._avg_doc_len > 0 else doc_len
        term_freq: Dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        seen_ids: Dict[int, float] = {}
        for token, freq in term_freq.items():
            idf = self._idf(token)
            tf_norm = (freq * (self.K1 + 1)) / (
                freq + self.K1 * (1 - self.B + self.B * doc_len / avgdl)
            )
            weight = idf * tf_norm
            token_id = self._token_id(token)
            seen_ids[token_id] = max(seen_ids.get(token_id, 0.0), weight)

        if not seen_ids:
            return [], []

        max_weight = max(seen_ids.values())
        indices = list(seen_ids.keys())
        values = [weight / max_weight for weight in seen_ids.values()]
        return indices, values


_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorStore:
    """Qdrant 向量存储，支持 dense + sparse 双路检索。"""

    def __init__(self, cfg: QdrantConfig, embed_dim: int) -> None:
        if not _HAS_QDRANT:
            raise ImportError("缺少 qdrant-client 依赖: pip install qdrant-client")

        self.cfg = cfg
        self.embed_dim = embed_dim
        self.collection = cfg.collection_name
        self._client = self._build_client(cfg)
        self._async_client = self._build_async_client(cfg)
        idf_path = os.getenv(
            "BM25_IDF_PATH",
            str(Path(cfg.path) / "bm25_idf") if cfg.mode == "local" else "./bm25_idf",
        )
        self._sparse_encoder = SparseEncoder(idf_path=idf_path)
        self._ensure_collection()

    @staticmethod
    def _build_client(cfg: QdrantConfig):
        if cfg.mode == "local":
            Path(cfg.path).mkdir(parents=True, exist_ok=True)
            logger.info(f"[Qdrant] 使用本地模式: {cfg.path}")
            return QdrantClient(path=cfg.path)
        if cfg.mode == "docker":
            logger.info(f"[Qdrant] 使用 Docker 模式: {cfg.host}:{cfg.port}")
            return QdrantClient(host=cfg.host, port=cfg.port)
        logger.info(f"[Qdrant] 使用 Cloud 模式: {cfg.url}")
        return QdrantClient(url=cfg.url, api_key=cfg.api_key)

    @staticmethod
    def _build_async_client(cfg: QdrantConfig):
        if cfg.mode == "local":
            return AsyncQdrantClient(path=cfg.path)
        if cfg.mode == "docker":
            return AsyncQdrantClient(host=cfg.host, port=cfg.port)
        return AsyncQdrantClient(url=cfg.url, api_key=cfg.api_key)

    def _ensure_collection(self) -> None:
        existing = {collection.name for collection in self._client.get_collections().collections}
        if self.collection in existing:
            logger.info(f"[Qdrant] 复用已有 collection: {self.collection}")
            return

        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(size=self.embed_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
        )

        for field_name, schema in [
            ("source_file", PayloadSchemaType.KEYWORD),
            ("source_path", PayloadSchemaType.KEYWORD),
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("heading_str", PayloadSchemaType.KEYWORD),
            ("section_index", PayloadSchemaType.INTEGER),
        ]:
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema,
            )

        logger.info(
            f"[Qdrant] 已创建 collection: {self.collection}, dense_dim={self.embed_dim}, sparse=enabled"
        )

    def upsert(
        self,
        chunks: List[DocumentChunk],
        dense_vectors: List[List[float]],
        batch_size: int = 64,
    ) -> None:
        """执行向量的插入或更新。"""
        self._sparse_encoder.update_idf([chunk.content for chunk in chunks])

        points: List[Any] = []
        for chunk, dense_vec in zip(chunks, dense_vectors):
            sparse_indices, sparse_values = self._sparse_encoder.encode(chunk.content)
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        _DENSE_VECTOR_NAME: dense_vec,
                        _SPARSE_VECTOR_NAME: SparseVector(indices=sparse_indices, values=sparse_values),
                    },
                    payload=chunk.to_payload(),
                )
            )

        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
            )

        logger.info(f"[Qdrant] 已 upsert {len(points)} 个点（dense + sparse）")

    def _format_query_results(self, results: Any) -> List[Dict[str, Any]]:
        points = getattr(results, "points", results)
        return [{"id": str(result.id), "score": result.score, "payload": result.payload} for result in points]

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行 Dense 检索，可选过滤条件。"""
        if hasattr(self._client, "search"):
            results = self._client.search(
                collection_name=self.collection,
                query_vector=NamedVector(name=_DENSE_VECTOR_NAME, vector=query_vector),
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=self._build_filter(filter_conditions),
                with_payload=True,
            )
        else:
            results = self._client.query_points(
                collection_name=self.collection,
                query=query_vector,
                using=_DENSE_VECTOR_NAME,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=self._build_filter(filter_conditions),
                with_payload=True,
            )
        return self._format_query_results(results)

    def search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行 Sparse 检索，可选过滤条件。"""
        indices, values = self._sparse_encoder.encode(query)
        if not indices:
            return []
        if hasattr(self._client, "search"):
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
        else:
            results = self._client.query_points(
                collection_name=self.collection,
                query=SparseVector(indices=indices, values=values),
                using=_SPARSE_VECTOR_NAME,
                limit=top_k,
                query_filter=self._build_filter(filter_conditions),
                with_payload=True,
            )
        return self._format_query_results(results)

    def fetch_by_section(self, doc_id: str, section_index: int) -> List[Dict[str, Any]]:
        """根据文档 ID 和章节索引获取当前章节下的所有文本块。"""
        if section_index == -1:
            return []
        results = self._scroll_all(
            Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    FieldCondition(key="section_index", match=MatchValue(value=section_index)),
                ]
            ),
            with_payload=True,
        )
        sorted_results = sorted(
            results,
            key=lambda result: result.payload.get("chunk_index", 0) if result.payload else 0,
        )
        return [{"id": str(result.id), "payload": result.payload} for result in sorted_results]

    async def async_search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if hasattr(self._async_client, "search"):
            results = await self._async_client.search(
                collection_name=self.collection,
                query_vector=NamedVector(name=_DENSE_VECTOR_NAME, vector=query_vector),
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=self._build_filter(filter_conditions),
                with_payload=True,
            )
        else:
            results = await self._async_client.query_points(
                collection_name=self.collection,
                query=query_vector,
                using=_DENSE_VECTOR_NAME,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                query_filter=self._build_filter(filter_conditions),
                with_payload=True,
            )
        return self._format_query_results(results)

    async def async_search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        indices, values = self._sparse_encoder.encode(query)
        if not indices:
            return []
        if hasattr(self._async_client, "search"):
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
        else:
            results = await self._async_client.query_points(
                collection_name=self.collection,
                query=SparseVector(indices=indices, values=values),
                using=_SPARSE_VECTOR_NAME,
                limit=top_k,
                query_filter=self._build_filter(filter_conditions),
                with_payload=True,
            )
        return self._format_query_results(results)

    def fetch_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """根据点 ID 列表批量获取对应的 payload 数据。"""
        results = self._client.retrieve(collection_name=self.collection, ids=ids, with_payload=True)
        return [{"id": str(result.id), "payload": result.payload} for result in results]

    def list_source_paths_by_source_file(self, source_file: str) -> List[str]:
        """根据source_file查询所有source_path。"""
        points = self._scroll_all(
            Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]),
            with_payload=True,
        )
        source_paths = {str(point.payload.get("source_path", "")) for point in points if point.payload}
        return sorted(path for path in source_paths if path)

    def doc_exists(self, doc_id: str) -> bool:
        """在向量数据库中检索doc_id对应的向量，来判断该文档是否已索引入向量数据库。"""
        results, _ = self._client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            limit=1,
            with_payload=False,
        )
        return len(results) > 0

    def delete_by_doc_id(self, doc_id: str) -> None:
        """删除指定 doc_id 对应的所有向量。"""
        self._delete_by_filter(
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            log_label=f"doc_id={doc_id[:8]}...",
        )

    def delete_by_source_path(self, source_path: str) -> None:
        """删除指定source_path对应的向量。"""
        self._delete_by_filter(
            Filter(must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]),
            log_label=f"source_path={source_path}",
        )

    def delete_by_source_file(self, source_file: str) -> None:
        self._delete_by_filter(
            Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]),
            log_label=f"source_file={source_file}",
        )

    def collection_info(self) -> Dict[str, Any]:
        """获取向量数据库中指定集合的统计信息。"""
        info = self._client.get_collection(self.collection)
        vectors_count = getattr(info, "vectors_count", None)
        if vectors_count is None:
            vectors_count = getattr(info, "indexed_vectors_count", None)
        if vectors_count is None:
            vectors_count = getattr(info, "points_count", 0)
        return {
            "vectors_count": vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def _delete_by_filter(self, qdrant_filter, log_label: str) -> None:
        points = self._scroll_all(qdrant_filter, with_payload=True)
        texts = [point.payload.get("content", "") for point in points if point.payload]
        if texts:
            self._sparse_encoder.remove_idf(texts)
        self._client.delete(collection_name=self.collection, points_selector=qdrant_filter)
        logger.info(f"[Qdrant] 已删除 {log_label} 命中的 {len(points)} 个点")

    def _scroll_all(self, qdrant_filter, with_payload: bool) -> List[Any]:
        all_results = []
        offset = None
        while True:
            results, next_page_offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=qdrant_filter,
                limit=256,
                with_payload=with_payload,
                with_vectors=False,
                offset=offset,
            )
            all_results.extend(results)
            if next_page_offset is None:
                break
            offset = next_page_offset
        return all_results

    @staticmethod
    def _build_filter(conditions: Optional[Dict[str, Any]]):
        if not conditions:
            return None
        must_clauses = []
        for key, value in conditions.items():
            if isinstance(value, list):
                must_clauses.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                must_clauses.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must_clauses) if must_clauses else None
