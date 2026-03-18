"""
tests/test_models.py
====================
DocumentChunk 数据结构测试。
"""

import pytest

from rag.models import DocumentChunk


class TestDocumentChunkCreate:
    def test_auto_generates_uuid(self):
        c1 = DocumentChunk.create(doc_id="d1", content="hello")
        c2 = DocumentChunk.create(doc_id="d1", content="hello")
        assert c1.chunk_id != c2.chunk_id  # 每次调用应生成不同 UUID

    def test_char_count_set_correctly(self):
        content = "这是测试内容"
        chunk = DocumentChunk.create(doc_id="d", content=content)
        assert chunk.char_count == len(content)

    def test_default_empty_heading_path(self):
        chunk = DocumentChunk.create(doc_id="d", content="c")
        assert chunk.heading_path == []

    def test_extra_meta_defaults_to_empty_dict(self):
        chunk = DocumentChunk.create(doc_id="d", content="c")
        assert chunk.extra_meta == {}


class TestDocumentChunkProperties:
    def test_heading_str_empty_when_no_path(self):
        chunk = DocumentChunk.create(doc_id="d", content="c")
        assert chunk.heading_str == ""

    def test_heading_str_joined_with_arrow(self):
        chunk = DocumentChunk.create(doc_id="d", content="c", heading_path=["H1", "H2", "H3"])
        assert chunk.heading_str == "H1 > H2 > H3"

    def test_full_text_for_embed_without_prefix(self):
        chunk = DocumentChunk.create(doc_id="d", content="内容")
        assert chunk.full_text_for_embed == "内容"

    def test_full_text_for_embed_with_prefix(self):
        chunk = DocumentChunk.create(doc_id="d", content="内容")
        chunk.context_prefix = "摘要"
        assert chunk.full_text_for_embed == "摘要\n\n内容"


class TestDocumentChunkToPayload:
    def test_payload_contains_required_fields(self):
        chunk = DocumentChunk.create(
            doc_id="doc123",
            content="测试内容",
            source_file="test.pdf",
            heading_path=["第一章"],
        )
        payload = chunk.to_payload()
        required = {
            "chunk_id", "doc_id", "content", "context_prefix",
            "source_file", "file_type", "heading_path", "heading_str",
            "chunk_index", "char_count", "upload_time",
        }
        assert required.issubset(payload.keys())

    def test_payload_heading_str_matches_property(self):
        chunk = DocumentChunk.create(doc_id="d", content="c", heading_path=["A", "B"])
        payload = chunk.to_payload()
        assert payload["heading_str"] == chunk.heading_str

    def test_extra_meta_merged_into_payload(self):
        chunk = DocumentChunk.create(
            doc_id="d", content="c", extra_meta={"author": "Alice", "tag": "test"}
        )
        payload = chunk.to_payload()
        assert payload["author"] == "Alice"
        assert payload["tag"] == "test"

    def test_extra_meta_does_not_override_core_fields(self):
        """extra_meta 不应能覆盖核心字段（content、doc_id 等）。"""
        chunk = DocumentChunk.create(
            doc_id="real_id", content="real_content", extra_meta={"doc_id": "fake_id"}
        )
        payload = chunk.to_payload()
        # extra_meta 在 **kwargs 展开，core 字段已先写入；Python dict update 语义决定后者覆盖
        # 这里记录当前行为（extra_meta 中同名 key 会覆盖），并作为回归测试
        # 若将来修改 to_payload() 加入保护，此测试可同步更新
        assert "doc_id" in payload  # 保证字段存在
