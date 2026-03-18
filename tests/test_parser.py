"""
tests/test_parser.py
====================
DocumentParser 路由逻辑测试（不依赖 Docling / Unstructured 服务）。
"""

import tempfile
from pathlib import Path

import pytest

from rag.parser import DocumentParser


@pytest.fixture()
def parser() -> DocumentParser:
    return DocumentParser()


class TestPlainTextParsing:
    def test_txt_file(self, parser, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello, World!", encoding="utf-8")
        text, file_type = parser.parse(str(f))
        assert file_type == "plain"
        assert "Hello, World!" in text

    def test_md_file(self, parser, tmp_path):
        f = tmp_path / "sample.md"
        f.write_text("# Title\n\nContent here.", encoding="utf-8")
        text, file_type = parser.parse(str(f))
        assert file_type == "plain"
        assert "Title" in text

    def test_json_file(self, parser, tmp_path):
        f = tmp_path / "sample.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        text, file_type = parser.parse(str(f))
        assert file_type == "plain"
        assert "value" in text

    def test_utf8_chinese_content(self, parser, tmp_path):
        f = tmp_path / "chinese.txt"
        f.write_text("这是中文内容测试。", encoding="utf-8")
        text, _ = parser.parse(str(f))
        assert "中文" in text

    def test_encoding_errors_replaced(self, parser, tmp_path):
        f = tmp_path / "bad_encoding.txt"
        f.write_bytes(b"\xff\xfe\x00hello")  # 非 UTF-8 字节
        text, _ = parser.parse(str(f))
        assert isinstance(text, str)  # 不应抛出异常


class TestFileNotFound:
    def test_raises_file_not_found(self, parser):
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            parser.parse("/nonexistent/path/to/file.txt")


class TestSupportedExtensions:
    def test_supported_extensions_is_frozenset(self, parser):
        exts = parser.supported_extensions
        assert isinstance(exts, frozenset)

    def test_common_extensions_supported(self, parser):
        exts = parser.supported_extensions
        for ext in (".pdf", ".docx", ".txt", ".md", ".csv"):
            assert ext in exts


class TestDoclingFallback:
    def test_fallback_to_unstructured_when_docling_fails(self, parser, tmp_path, monkeypatch):
        """当 Docling 解析失败时，应自动降级到 Unstructured。"""
        import rag.parser as parser_module

        # Mock Docling 可用但解析失败
        monkeypatch.setattr(parser_module, "_HAS_DOCLING", True)
        monkeypatch.setattr(parser_module, "_HAS_UNSTRUCTURED", True)

        def _fail_docling(self, _path):
            return None

        def _succeed_unstructured(self, _path):
            return "Unstructured fallback content"

        monkeypatch.setattr(DocumentParser, "_parse_with_docling", _fail_docling)
        monkeypatch.setattr(DocumentParser, "_parse_with_unstructured", _succeed_unstructured)

        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-fake")
        text, file_type = parser.parse(str(f))
        assert file_type == "unstructured"
        assert "fallback" in text

    def test_raises_when_all_parsers_fail(self, parser, tmp_path, monkeypatch):
        """当所有解析器均失败时，应抛出 RuntimeError。"""
        import rag.parser as parser_module

        monkeypatch.setattr(parser_module, "_HAS_DOCLING", True)
        monkeypatch.setattr(parser_module, "_HAS_UNSTRUCTURED", True)
        monkeypatch.setattr(DocumentParser, "_parse_with_docling", lambda s, p: None)
        monkeypatch.setattr(DocumentParser, "_parse_with_unstructured", lambda s, p: None)

        f = tmp_path / "broken.pdf"
        f.write_bytes(b"broken")
        with pytest.raises(RuntimeError, match="所有解析器"):
            parser.parse(str(f))
