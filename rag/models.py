"""数据模型定义。"""

from __future__ import annotations

import re
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Dict, Iterator, List, Optional

_DEFAULT_NAMESPACE = "default"
_ALLOWED_REINDEX_STRATEGIES = frozenset({"skip_existing", "force"})
_UPLOAD_ORIGIN_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
_RESERVED_INDEX_METADATA_KEYS = frozenset(
    {
        "sdk_source_kind",
        "sdk_source_origin",
        "sdk_source_name",
        "sdk_source_path",
    }
)


def _normalize_namespace(namespace: Optional[str]) -> str:
    """规范化处理命名空间字符串，避免空值。"""
    candidate = (namespace or _DEFAULT_NAMESPACE).strip()
    return candidate or _DEFAULT_NAMESPACE


def _validate_index_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata 必须是 dict")
    invalid_keys = [key for key in metadata if not isinstance(key, str) or not key.strip()]
    if invalid_keys:
        raise ValueError("metadata 键必须是非空字符串")
    conflicts = _RESERVED_INDEX_METADATA_KEYS & set(metadata)
    if conflicts:
        conflict_keys = ", ".join(sorted(conflicts))
        raise ValueError(f"metadata 不能覆盖保留字段: {conflict_keys}")
    return dict(metadata)


@dataclass(frozen=True)
class MaterializedDocumentSource:
    """已准备好被引擎索引的物化文档源。"""

    file_path: str
    display_source_name: str
    display_source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentSource:
    """处理 path / text / bytes 类型的稳定 SDK 文档源输入模型。"""

    path: Optional[str] = None
    text: Optional[str] = None
    data: Optional[bytes] = None
    source_name: Optional[str] = None
    upload_origin: str = "sdk"

    def __post_init__(self) -> None:
        provided_count = sum(value is not None for value in (self.path, self.text, self.data))
        if provided_count != 1:
            raise ValueError("DocumentSource 必须且只能提供 path、text、data 其中一种")

        normalized_origin = (self.upload_origin or "").strip()
        if not normalized_origin:
            raise ValueError("upload_origin 不能为空")
        if not _UPLOAD_ORIGIN_RE.match(normalized_origin):
            raise ValueError("upload_origin 仅支持字母、数字、点、下划线和连字符")
        object.__setattr__(self, "upload_origin", normalized_origin)

        if self.path is not None:
            resolved_path = Path(self.path).expanduser().resolve(strict=True)
            if not resolved_path.is_file():
                raise ValueError(f"path 必须指向文件: {resolved_path}")
            object.__setattr__(self, "path", str(resolved_path))
            object.__setattr__(self, "source_name", resolved_path.name)
            return

        normalized_name = self._normalize_source_name(
            self.source_name,
            default_name="inline.md" if self.text is not None else None,
        )
        object.__setattr__(self, "source_name", normalized_name)

        if self.data is not None and not normalized_name:
            raise ValueError("bytes 源必须提供 source_name")

    @classmethod
    def from_path(cls, path: str, *, upload_origin: str = "path") -> "DocumentSource":
        return cls(path=path, upload_origin=upload_origin)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source_name: str = "inline.md",
        upload_origin: str = "text",
    ) -> "DocumentSource":
        return cls(text=text, source_name=source_name, upload_origin=upload_origin)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        source_name: str,
        upload_origin: str = "bytes",
    ) -> "DocumentSource":
        return cls(data=data, source_name=source_name, upload_origin=upload_origin)

    @staticmethod
    def _normalize_source_name(source_name: Optional[str], *, default_name: Optional[str]) -> str:
        candidate = (source_name or default_name or "").strip()
        if not candidate:
            raise ValueError("source_name 不能为空")
        normalized_name = PurePath(candidate).name
        if normalized_name != candidate:
            raise ValueError("source_name 不能包含目录层级")
        if normalized_name in {".", ".."}:
            raise ValueError("source_name 非法")
        if not Path(normalized_name).suffix:
            raise ValueError("source_name 必须包含文件扩展名")
        return normalized_name

    @property
    def kind(self) -> str:
        if self.path is not None:
            return "path"
        if self.text is not None:
            return "text"
        return "bytes"

    @property
    def effective_name(self) -> str:
        if self.path is not None:
            return Path(self.path).name
        return self.source_name or "document"

    @property
    def display_path(self) -> str:
        if self.path is not None:
            return str(Path(self.path))
        return f"{self.kind}://{self.upload_origin}/{self.effective_name}"

    def metadata_fields(self) -> Dict[str, Any]:
        return {
            "sdk_source_kind": self.kind,
            "sdk_source_origin": self.upload_origin,
            "sdk_source_name": self.effective_name,
            "sdk_source_path": self.display_path,
        }

    @contextmanager
    def materialize(self) -> Iterator[MaterializedDocumentSource]:
        if self.path is not None:
            yield MaterializedDocumentSource(
                file_path=self.path,
                display_source_name=self.effective_name,
                display_source_path=self.display_path,
                metadata=self.metadata_fields(),
            )
            return

        with tempfile.TemporaryDirectory(prefix="rag-sdk-") as temp_dir:
            temp_path = Path(temp_dir) / self.effective_name
            if self.text is not None:
                temp_path.write_text(self.text, encoding="utf-8", newline="\n")
            else:
                temp_path.write_bytes(self.data or b"")
            yield MaterializedDocumentSource(
                file_path=str(temp_path),
                display_source_name=self.effective_name,
                display_source_path=self.display_path,
                metadata=self.metadata_fields(),
            )


@dataclass(frozen=True)
class IndexRequest:
    """稳定的 SDK 索引请求模型。"""

    source: DocumentSource
    namespace: str = _DEFAULT_NAMESPACE
    metadata: Dict[str, Any] = field(default_factory=dict)
    reindex_strategy: str = "skip_existing"

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _normalize_namespace(self.namespace))
        object.__setattr__(self, "metadata", _validate_index_metadata(self.metadata))
        strategy = (self.reindex_strategy or "skip_existing").strip().lower()
        if strategy not in _ALLOWED_REINDEX_STRATEGIES:
            raise ValueError(
                "reindex_strategy 仅支持 skip_existing 或 force"
            )
        object.__setattr__(self, "reindex_strategy", strategy)

    @property
    def force_reindex(self) -> bool:
        return self.reindex_strategy == "force"


@dataclass(frozen=True)
class SearchRequest:
    """稳定的 SDK 检索请求模型。"""

    query: str
    namespace: str = _DEFAULT_NAMESPACE
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    score_threshold: float = 0.0
    skip_rerank: bool = False

    def __post_init__(self) -> None:
        normalized_query = self.query.strip()
        if not normalized_query:
            raise ValueError("query 不能为空")
        if self.top_k < 1:
            raise ValueError("top_k 必须大于等于 1")
        if self.score_threshold < 0:
            raise ValueError("score_threshold 不能小于 0")
        object.__setattr__(self, "query", normalized_query)
        object.__setattr__(self, "namespace", _normalize_namespace(self.namespace))
        object.__setattr__(self, "filters", dict(self.filters) if self.filters else None)


@dataclass
class DocumentChunk:
    """单个索引块及其检索元数据。"""

    chunk_id: str
    doc_id: str
    content: str
    context_prefix: str = ""
    source_file: str = ""
    source_path: str = ""
    file_type: str = ""
    heading_path: List[str] = field(default_factory=list)
    chunk_index: int = 0
    section_index: int = -1
    char_count: int = 0
    upload_time: str = ""
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    _RESERVED_EXTRA_META_KEYS = {
        "chunk_id",
        "doc_id",
        "content",
        "context_prefix",
        "source_file",
        "source_path",
        "file_type",
        "heading_path",
        "heading_str",
        "chunk_index",
        "section_index",
        "char_count",
        "upload_time",
    }

    def __post_init__(self) -> None:
        self.char_count = len(self.content)
        conflicts = self._RESERVED_EXTRA_META_KEYS & set(self.extra_meta)
        if conflicts:
            conflict_keys = ", ".join(sorted(conflicts))
            raise ValueError(f"extra_meta 不能覆盖保留字段: {conflict_keys}")

    @classmethod
    def create(
        cls,
        doc_id: str,
        content: str,
        source_file: str = "",
        source_path: str = "",
        file_type: str = "",
        heading_path: List[str] | None = None,
        chunk_index: int = 0,
        section_index: int = -1,
        upload_time: str = "",
        extra_meta: Dict[str, Any] | None = None,
    ) -> "DocumentChunk":
        """创建 chunk，并为其分配新的 UUID。"""
        return cls(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            content=content,
            source_file=source_file,
            source_path=source_path,
            file_type=file_type,
            heading_path=heading_path or [],
            chunk_index=chunk_index,
            section_index=section_index,
            char_count=0,
            upload_time=upload_time,
            extra_meta=dict(extra_meta or {}),
        )

    @property
    def full_text_for_embed(self) -> str:
        """返回用于 embedding 的完整文本，包含标题、上下文摘要和chunk原始内容。"""
        parts: List[str] = []
        if self.heading_path:
            parts.append(self.heading_str)
        if self.context_prefix:
            parts.append(self.context_prefix)
        parts.append(self.content)
        return "\n\n".join(parts)

    @property
    def heading_str(self) -> str:
        """将标题路径格式化为单行字符串。"""
        return " > ".join(self.heading_path)

    def to_payload(self) -> Dict[str, Any]:
        """转换为 Qdrant 可持久化的 payload。"""
        payload = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "context_prefix": self.context_prefix,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "heading_path": self.heading_path,
            "heading_str": self.heading_str,
            "chunk_index": self.chunk_index,
            "section_index": self.section_index,
            "char_count": self.char_count,
            "upload_time": self.upload_time,
        }
        payload.update(self.extra_meta)
        return payload


@dataclass(frozen=True)
class IndexOptions:
    """为旧版 SDK 索引参数预留的兼容选项配置。"""

    namespace: str = _DEFAULT_NAMESPACE
    extra_meta: Dict[str, Any] = field(default_factory=dict)
    force_reindex: bool = False


@dataclass(frozen=True)
class RetrieveOptions:
    """为旧版 SDK 检索参数预留的兼容选项配置。"""

    namespace: str = _DEFAULT_NAMESPACE
    top_k: int = 5
    filter_conditions: Optional[Dict[str, Any]] = None
    skip_rerank: bool = False
    score_threshold: float = 0.0


@dataclass(frozen=True)
class DeleteTarget:
    """通过 source_path 进行明确指向的 SDK 稳定删除目标。"""

    source_path: str

    def __post_init__(self) -> None:
        normalized = self.source_path.strip()
        if not normalized:
            raise ValueError("source_path 不能为空")
        object.__setattr__(self, "source_path", normalized)

    @classmethod
    def from_source_path(cls, source_path: str) -> "DeleteTarget":
        return cls(source_path=source_path)

    @classmethod
    def from_source(cls, source: DocumentSource) -> "DeleteTarget":
        return cls(source_path=source.display_path)


@dataclass(frozen=True)
class DeleteOptions:
    """通过 SDK 进行文档删除操作的相关配置选项。"""

    namespace: str = _DEFAULT_NAMESPACE


@dataclass(frozen=True)
class RetrievedItem:
    """由 SDK 统一返回的数据结果项类型，包含了命中信息的详情结构。"""

    doc_id: str = ""
    content: str = ""
    context_prefix: str = ""
    source_file: str = ""
    source_path: str = ""
    heading_str: str = ""
    heading_path: List[str] = field(default_factory=list)
    chunk_index: int = 0
    section_index: int = -1
    upload_time: str = ""
    score: float = 0.0
    rrf_score: float = 0.0
    dense_score: float = 0.0
    section_context: str = ""
    section_chunk_count: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievedItem":
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            content=str(payload.get("content", "")),
            context_prefix=str(payload.get("context_prefix", "")),
            source_file=str(payload.get("source_file", "")),
            source_path=str(payload.get("source_path", "")),
            heading_str=str(payload.get("heading_str", "")),
            heading_path=list(payload.get("heading_path", []) or []),
            chunk_index=int(payload.get("chunk_index", 0) or 0),
            section_index=int(payload.get("section_index", -1) or -1),
            upload_time=str(payload.get("upload_time", "")),
            score=float(payload.get("score", 0.0) or 0.0),
            rrf_score=float(payload.get("rrf_score", 0.0) or 0.0),
            dense_score=float(payload.get("dense_score", 0.0) or 0.0),
            section_context=str(payload.get("section_context", "")),
            section_chunk_count=int(payload.get("section_chunk_count", 0) or 0),
        )


@dataclass(frozen=True)
class IndexResult:
    """文档索引操作的结果类型。"""
    namespace: str
    raw: Dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.raw.get("status", ""))

    @property
    def source_path(self) -> str:
        return str(self.raw.get("source_path", ""))

    @property
    def chunks(self) -> int:
        return int(self.raw.get("chunks", 0) or 0)


@dataclass(frozen=True)
class DeleteResult:
    """文档删除操作的结果类型。"""
    namespace: str
    raw: Dict[str, Any]

    @property
    def deleted(self) -> bool:
        return bool(self.raw.get("deleted", False))

    @property
    def source_path(self) -> str:
        if "source_path" in self.raw:
            return str(self.raw.get("source_path", ""))
        resolved = self.resolved_source_paths
        return resolved[0] if len(resolved) == 1 else ""

    @property
    def resolved_source_paths(self) -> List[str]:
        values = self.raw.get("resolved_source_paths", []) or []
        return [str(value) for value in values if str(value)]

    @property
    def request_identifier(self) -> str:
        return str(self.raw.get("request_identifier", self.raw.get("file_identifier", "")))

    @property
    def file_identifier(self) -> str:
        return self.request_identifier


@dataclass(frozen=True)
class RetrieveResult:
    """文档检索操作的结果类型，包含了格式化后的查询结果和上下文。"""
    namespace: str
    query: str
    items: List[RetrievedItem]
    formatted_context: str
    raw: Dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.items)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrieveResult":
        return cls(
            namespace=str(payload.get("namespace", _DEFAULT_NAMESPACE)),
            query=str(payload.get("query", "")),
            items=[RetrievedItem.from_dict(item) for item in payload.get("results", []) or []],
            formatted_context=str(payload.get("formatted_context", "")),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class HealthStatus:
    """服务的健康状态及集合统计信息模型。"""
    namespace: str
    status: str
    collection: Optional[Dict[str, Any]] = None
    known_namespaces: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HealthStatus":
        return cls(
            namespace=str(payload.get("namespace", _DEFAULT_NAMESPACE)),
            status=str(payload.get("status", "unknown")),
            collection=payload.get("collection"),
            known_namespaces=list(payload.get("known_namespaces", []) or []),
        )



