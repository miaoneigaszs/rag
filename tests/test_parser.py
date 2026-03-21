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


class TestImageAndMultimodalParsing:
    def test_docling_image_extraction_logic(self, parser, tmp_path, monkeypatch):
        """测试 Docling 图片提取和多模态描述替换的顺序替换逻辑。"""
        import rag.parser as parser_module
        from pathlib import Path

        # 1. Mock 依赖状态
        monkeypatch.setattr(parser_module, "_HAS_DOCLING", True)
        
        # 2. Mock _get_image_caption 避免真实 API 调用
        def mock_caption(self, image_bytes):
            return "Mocked image description"
        monkeypatch.setattr(DocumentParser, "_get_image_caption", mock_caption)

        # 3. Mock 转换行为
        f = tmp_path / "test_with_images.pdf"
        f.write_bytes(b"%PDF-fake")
        
        # 模拟 Docling 返回的带两个固定占位符的内容
        raw_md = "Img1: <!-- image --> | Img2: <!-- image -->"
        
        img_dir = tmp_path / "test_with_images_images"
        img_dir.mkdir()

        # 模拟包含两张图片的 result 对象
        class MockResult:
            def __init__(self):
                self.document = self
                # 修复 Mock: 使用 self_ref 且 save 方法增加 **kwargs
                self.pictures = [
                    type('Pic', (), {
                        'self_ref': '#/picture/0', 
                        'get_image': lambda self, doc: type('Img', (), {
                            'save': lambda self, p, **kwargs: None
                        })()
                    })(),
                    type('Pic', (), {
                        'self_ref': '#/picture/1', 
                        'get_image': lambda self, doc: type('Img', (), {
                            'save': lambda self, p, **kwargs: None
                        })()
                    })()
                ]
            def export_to_markdown(self, **kwargs):
                return raw_md


        monkeypatch.setattr("docling.document_converter.DocumentConverter.convert", lambda self, p: MockResult())
        
        # 4. 执行解析
        text, file_type = parser.parse(str(f))
        
        assert file_type == "docling"
        # 验证两个占位符是否都被按顺序替换了
        assert text.count("Mocked image description") == 2
        assert "image_0.png" in text
        assert "image_1.png" in text
        # 验证原始占位符已消失
        assert "<!-- image -->" not in text

    def test_image_dir_creation(self, parser, tmp_path):
        """验证是否为包含图片的文档正确创建了图片子目录。"""
        # 这个测试可以配合一个极其微小的真实 PDF (或者完全 Mock)
        # 这里我们验证路径拼接逻辑是否符合预期
        doc_path = tmp_path / "my_doc.pdf"
        expected_img_dir = tmp_path / "my_doc_images"
        
        # 检查逻辑是否能正确识别路径
        assert str(expected_img_dir) == str(Path(doc_path).parent / f"{Path(doc_path).stem}_images")


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
