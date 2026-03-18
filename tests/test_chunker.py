"""
tests/test_chunker.py
=====================
HierarchicalMarkdownSplitter 的切块逻辑测试。
"""

import pytest

from rag.chunker import HierarchicalMarkdownSplitter
from rag.config import ChunkConfig


@pytest.fixture()
def default_splitter() -> HierarchicalMarkdownSplitter:
    return HierarchicalMarkdownSplitter(ChunkConfig(chunk_size=200, chunk_overlap=30, min_chunk_size=10))


@pytest.fixture()
def tight_splitter() -> HierarchicalMarkdownSplitter:
    """极小 chunk_size，用于测试递归分割。"""
    return HierarchicalMarkdownSplitter(ChunkConfig(chunk_size=50, chunk_overlap=10, min_chunk_size=5))


class TestSplitByHeadings:
    def test_single_section(self, default_splitter):
        text = "# 标题\n\n这是内容。"
        chunks = default_splitter.split(text)
        assert len(chunks) == 1
        assert "标题" in chunks[0]["content"]
        assert chunks[0]["heading_path"] == ["标题"]

    def test_multiple_sections(self, default_splitter):
        text = "# H1\n\n内容A\n\n## H2\n\n内容B"
        chunks = default_splitter.split(text)
        assert len(chunks) == 2
        assert chunks[0]["heading_path"] == ["H1"]
        assert chunks[1]["heading_path"] == ["H1", "H2"]

    def test_nested_headings_path(self, default_splitter):
        text = "# 章\n\n## 节\n\n### 小节\n\n内容"
        chunks = default_splitter.split(text)
        # 最深层 chunk 应包含完整路径
        deepest = chunks[-1]
        assert deepest["heading_path"] == ["章", "节", "小节"]

    def test_heading_level_reset(self, default_splitter):
        text = "# H1\n\n## H2\n\n# 新H1\n\n内容"
        chunks = default_splitter.split(text)
        last = chunks[-1]
        # "新H1" 下不应带着旧的 H2
        assert last["heading_path"] == ["新H1"]

    def test_no_headings(self, default_splitter):
        text = "纯文本内容，没有任何 Markdown 标题。"
        chunks = default_splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["heading_path"] == []

    def test_empty_text_returns_empty(self, default_splitter):
        assert default_splitter.split("") == []

    def test_short_content_filtered(self, default_splitter):
        """低于 min_chunk_size 的节应被过滤。"""
        cfg = ChunkConfig(chunk_size=200, chunk_overlap=30, min_chunk_size=100)
        splitter = HierarchicalMarkdownSplitter(cfg)
        text = "# H1\n\n短。"  # "短。" < 100 字符
        chunks = splitter.split(text)
        # "# H1\n\n短。" 整体也很短，取决于实现；主要验证 min_chunk_size 生效
        for chunk in chunks:
            assert len(chunk["content"]) >= cfg.min_chunk_size


class TestChunkIndex:
    def test_chunk_index_sequential(self, default_splitter):
        text = "\n\n".join(f"# 节{i}\n\n{'内容' * 5}" for i in range(5))
        chunks = default_splitter.split(text)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i


class TestRecursiveSplit:
    def test_oversized_section_gets_split(self, tight_splitter):
        """超大节应被递归分割为多个 chunk。"""
        long_content = "这是一段很长的内容。" * 20  # ~200 字符，远超 chunk_size=50
        text = f"# 标题\n\n{long_content}"
        chunks = tight_splitter.split(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["content"]) <= tight_splitter.cfg.chunk_size * 2  # 允许少量超出（重叠）

    def test_all_chunks_have_same_heading(self, tight_splitter):
        """同一节拆出的多个 chunk 应共享相同的 heading_path。"""
        long_content = "测试内容单元。" * 30
        text = f"# 统一标题\n\n{long_content}"
        chunks = tight_splitter.split(text)
        assert all(c["heading_path"] == ["统一标题"] for c in chunks)

    def test_overlap_creates_continuity(self, tight_splitter):
        """相邻 chunk 之间应有内容重叠（overlap > 0）。"""
        long_content = "ABCDEFGH " * 20
        text = f"# 标题\n\n{long_content}"
        chunks = tight_splitter.split(text)
        if len(chunks) > 1:
            # 后一个 chunk 的开头应该包含前一个 chunk 的部分尾部
            end_of_first = chunks[0]["content"][-tight_splitter.cfg.chunk_overlap:]
            # 验证有实质性重叠（至少有一个公共词）
            tokens_end = set(end_of_first.split())
            tokens_start = set(chunks[1]["content"].split())
            assert tokens_end & tokens_start, "相邻 chunk 之间没有发现重叠内容"


class TestCharCount:
    def test_char_count_matches_content_length(self, default_splitter):
        text = "# 标题\n\n" + "测试内容。" * 10
        chunks = default_splitter.split(text)
        for chunk in chunks:
            assert chunk["char_count"] == len(chunk["content"])
