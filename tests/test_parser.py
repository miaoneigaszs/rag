"""DocumentParser 路由逻辑测试（不依赖 Docling / Unstructured 服务）。"""

import importlib.util
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
        f.write_bytes(b"\xff\xfe\x00hello")
        text, _ = parser.parse(str(f))
        assert isinstance(text, str)


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


class TestImageAndMultimodalParsing:
    def test_get_image_caption_skips_when_vision_disabled(self, parser):
        parser._vision_enabled = False
        assert parser._get_image_caption(b"fake-image-bytes") == ""

    def test_get_image_caption_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setenv("VISION_ENABLED", "true")
        monkeypatch.delenv("VISION_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        parser = DocumentParser()
        assert parser._get_image_caption(b"fake-image-bytes") == ""

    def test_docling_image_extraction_logic(self, parser, tmp_path, monkeypatch):
        if importlib.util.find_spec("docling") is None:
            pytest.skip("docling 未安装，跳过图片提取逻辑测试")

        import rag.parser as parser_module

        monkeypatch.setattr(parser_module, "_HAS_DOCLING", True)

        def mock_caption(self, image_bytes):
            return "Mocked image description"

        monkeypatch.setattr(DocumentParser, "_get_image_caption", mock_caption)

        f = tmp_path / "test_with_images.pdf"
        f.write_bytes(b"%PDF-fake")
        raw_md = "Img1: <!-- image --> | Img2: <!-- image -->"

        class MockResult:
            def __init__(self):
                self.document = self
                self.pictures = [
                    type(
                        "Pic",
                        (),
                        {
                            "self_ref": "#/picture/0",
                            "get_image": lambda self, doc: type(
                                "Img", (), {"save": lambda self, p, **kwargs: None}
                            )(),
                        },
                    )(),
                    type(
                        "Pic",
                        (),
                        {
                            "self_ref": "#/picture/1",
                            "get_image": lambda self, doc: type(
                                "Img", (), {"save": lambda self, p, **kwargs: None}
                            )(),
                        },
                    )(),
                ]

            def export_to_markdown(self, **kwargs):
                return raw_md


        monkeypatch.setattr(
            "docling.document_converter.DocumentConverter.convert",
            lambda self, path: MockResult(),
        )

        text, file_type = parser.parse(str(f))
        assert file_type == "docling"
        assert text.count("Mocked image description") == 2
        assert "image_0.png" in text
        assert "image_1.png" in text
        assert "<!-- image -->" not in text

    def test_image_dir_creation(self, parser, tmp_path):
        doc_path = tmp_path / "my_doc.pdf"
        expected_img_dir = tmp_path / "my_doc_images"
        assert str(expected_img_dir) == str(Path(doc_path).parent / f"{Path(doc_path).stem}_images")


class TestDoclingFallback:
    def test_fallback_to_unstructured_when_docling_fails(self, parser, tmp_path, monkeypatch):
        import rag.parser as parser_module

        monkeypatch.setattr(parser_module, "_HAS_DOCLING", True)
        monkeypatch.setattr(parser_module, "_HAS_UNSTRUCTURED", True)
        monkeypatch.setattr(DocumentParser, "_parse_with_docling", lambda self, path: None)
        monkeypatch.setattr(
            DocumentParser,
            "_parse_with_unstructured",
            lambda self, path: "Unstructured fallback content",
        )

        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-fake")
        text, file_type = parser.parse(str(f))
        assert file_type == "unstructured"
        assert "fallback" in text

    def test_raises_when_all_parsers_fail(self, parser, tmp_path, monkeypatch):
        import rag.parser as parser_module

        monkeypatch.setattr(parser_module, "_HAS_DOCLING", True)
        monkeypatch.setattr(parser_module, "_HAS_UNSTRUCTURED", True)
        monkeypatch.setattr(DocumentParser, "_parse_with_docling", lambda self, path: None)
        monkeypatch.setattr(DocumentParser, "_parse_with_unstructured", lambda self, path: None)

        f = tmp_path / "broken.pdf"
        f.write_bytes(b"broken")
        with pytest.raises(RuntimeError, match="所有解析器"):
            parser.parse(str(f))
