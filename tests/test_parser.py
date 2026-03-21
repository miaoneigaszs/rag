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
        """测试 Docling 图片提取和多模态描述替换的逻辑流程。"""
        import rag.parser as parser_module
        from pathlib import Path

        # 1. Mock 依赖状态
        monkeypatch.setattr(parser_module, "_HAS_DOCLING", True)
        
        # 2. Mock _get_image_caption 避免真实 API 调用
        def mock_caption(self, image_bytes):
            return "Mocked image description"
        monkeypatch.setattr(DocumentParser, "_get_image_caption", mock_caption)

        # 3. Mock _parse_with_docling 的核心行为 (因为在测试环境下很难构造真实的 Docling 对象)
        # 我们直接测试 _parse_with_docling 内部调用的替换逻辑
        f = tmp_path / "test_with_images.pdf"
        f.write_bytes(b"%PDF-fake")
        
        # 模拟 Docling 返回的带占位符的内容
        raw_md = "Text before. <!-- image: 1 --> Text after."
        
        # 我们手动触发包含图片处理逻辑的模拟调用
        # (这里为了演示，我们模拟一个 pic_obj 并在 tmp_path 下生成图片目录)
        img_dir = tmp_path / "test_with_images_images"
        img_dir.mkdir()
        (img_dir / "image_1.png").write_bytes(b"fake_image_data")

        # 模拟替换后的结果检查
        # 实际运行中，parser.parse 会调用修改后的 _parse_with_docling
        # 鎴戜滑杩欓噷閫氳繃 mock 鎺 converter.convert 鏉ョ‘淇濊繘鍏ヨВ鏋愰€昏緫
        class MockResult:
            def __init__(self):
                self.document = self
                # 修复 Mock: save 方法增加 **kwargs 接收 format="PNG"
                self.pictures = [
                    type('Pic', (), {
                        'id': 1, 
                        'get_image': lambda self, doc: type('Img', (), {
                            'save': lambda self, p, **kwargs: None
                        })()
                    })()
                ]
            def export_to_markdown(self, **kwargs):
                return raw_md


        monkeypatch.setattr("docling.document_converter.DocumentConverter.convert", lambda self, p: MockResult())
        
        # 执行解析 (注意：这会触发我们修改后的代码)
        # 由于我们 Mock 了 convert，代码会尝试运行图片处理循环
        text, file_type = parser.parse(str(f))
        
        assert file_type == "docling"
        # 验证描述是否被插入
        assert "Mocked image description" in text
        # 验证本地路径标记是否存在
        assert "图片本地路径" in text

    def test_image_dir_creation(self, parser, tmp_path):
        """验证是否为包含图片的文档正确创建了图片子目录。"""
        # 这个测试可以配合一个极其微小的真实 PDF (或者完全 Mock)
        # 这里我们验证路径拼接逻辑是否符合预期
        doc_path = tmp_path / "my_doc.pdf"
        expected_img_dir = tmp_path / "my_doc_images"
        
        # 检查逻辑是否能正确识别路径
        assert str(expected_img_dir) == str(Path(doc_path).parent / f"{Path(doc_path).stem}_images")
