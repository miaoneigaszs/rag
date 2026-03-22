"""
rag/models.py
=============
核心数据结构。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DocumentChunk:
    """单个文档切块，贯穿解析 → 存储 → 检索全流程。"""

    chunk_id: str
    doc_id: str
    content: str
    context_prefix: str = ""
    source_file: str = ""
    file_type: str = ""
    heading_path: List[str] = field(default_factory=list)
    chunk_index: int = 0
    section_index: int = -1          # 所属语义节序号，-1 表示未知；用于高级 RAG section 扩展
    char_count: int = 0
    upload_time: str = ""
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        doc_id: str,
        content: str,
        source_file: str = "",
        file_type: str = "",
        heading_path: List[str] | None = None,
        chunk_index: int = 0,
        section_index: int = -1,
        upload_time: str = "",
        extra_meta: Dict[str, Any] | None = None,
    ) -> "DocumentChunk":
        """自动生成 chunk_id 的工厂方法，避免调用方手动 uuid.uuid4()。"""
        return cls(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            content=content,
            source_file=source_file,
            file_type=file_type,
            heading_path=heading_path or [],
            chunk_index=chunk_index,
            section_index=section_index,
            char_count=len(content),
            upload_time=upload_time,
            extra_meta=extra_meta or {},
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def full_text_for_embed(self) -> str:
        """用于生成向量的文本：标题路径 + 上下文前缀 + 内容。"""
        parts = []
        if self.heading_path:
            parts.append(self.heading_str)
        if self.context_prefix:
            parts.append(self.context_prefix)
        parts.append(self.content)
        return "\n\n".join(parts)

    @property
    def heading_str(self) -> str:
        """标题路径拼接为可读字符串，如 '第一章 > 1.1 背景'。"""
        return " > ".join(self.heading_path)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_payload(self) -> Dict[str, Any]:
        """转为 Qdrant payload（所有字段需可 JSON 序列化）。"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "context_prefix": self.context_prefix,
            "source_file": self.source_file,
            "file_type": self.file_type,
            "heading_path": self.heading_path,
            "heading_str": self.heading_str,
            "chunk_index": self.chunk_index,
            "section_index": self.section_index,   # 高级 RAG section 扩展检索依赖此字段
            "char_count": self.char_count,
            "upload_time": self.upload_time,
            **self.extra_meta,
        }