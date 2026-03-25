"""数据模型定义。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List


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
        """返回用于 embedding 的完整文本。"""
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
