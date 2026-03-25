"""DocumentChunk 测试。"""

import pytest

from rag.models import DocumentChunk


class TestDocumentChunkCreate:
    def test_auto_generates_uuid(self):
        chunk1 = DocumentChunk.create(doc_id="d1", content="hello")
        chunk2 = DocumentChunk.create(doc_id="d1", content="hello")
        assert chunk1.chunk_id != chunk2.chunk_id

    def test_char_count_set_correctly(self):
        content = "中文内容"
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
        chunk = DocumentChunk.create(doc_id="d", content="正文")
        assert chunk.full_text_for_embed == "正文"

    def test_full_text_for_embed_with_prefix(self):
        chunk = DocumentChunk.create(doc_id="d", content="正文")
        chunk.context_prefix = "摘要"
        assert chunk.full_text_for_embed == "摘要\n\n正文"

    def test_full_text_for_embed_with_heading_and_prefix(self):
        chunk = DocumentChunk.create(doc_id="d", content="正文", heading_path=["一级标题"])
        chunk.context_prefix = "摘要"
        assert chunk.full_text_for_embed == "一级标题\n\n摘要\n\n正文"


class TestDocumentChunkToPayload:
    def test_payload_contains_required_fields(self):
        chunk = DocumentChunk.create(
            doc_id="doc123",
            content="测试内容",
            source_file="test.pdf",
            source_path="/abs/test.pdf",
            heading_path=["测试标题"],
        )
        payload = chunk.to_payload()
        required = {
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
        assert required.issubset(payload.keys())

    def test_payload_heading_str_matches_property(self):
        chunk = DocumentChunk.create(doc_id="d", content="c", heading_path=["A", "B"])
        payload = chunk.to_payload()
        assert payload["heading_str"] == chunk.heading_str

    def test_extra_meta_merged_into_payload(self):
        chunk = DocumentChunk.create(
            doc_id="d",
            content="c",
            extra_meta={"author": "Alice", "tag": "test"},
        )
        payload = chunk.to_payload()
        assert payload["author"] == "Alice"
        assert payload["tag"] == "test"

    def test_extra_meta_cannot_override_core_fields(self):
        with pytest.raises(ValueError, match="保留字段"):
            DocumentChunk.create(
                doc_id="real_id",
                content="real_content",
                extra_meta={"doc_id": "fake_id"},
            )
