"""HierarchicalMarkdownSplitter 测试。"""

import pytest

from rag.chunker import HierarchicalMarkdownSplitter
from rag.config import ChunkConfig


@pytest.fixture()
def default_splitter() -> HierarchicalMarkdownSplitter:
    return HierarchicalMarkdownSplitter(
        ChunkConfig(chunk_size=200, chunk_overlap=30, min_chunk_size=10)
    )


@pytest.fixture()
def tight_splitter() -> HierarchicalMarkdownSplitter:
    return HierarchicalMarkdownSplitter(
        ChunkConfig(chunk_size=50, chunk_overlap=10, min_chunk_size=5)
    )


class TestSplitByHeadings:
    def test_single_section(self, default_splitter):
        text = "# 标题\n\n这是正文"
        chunks = default_splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "这是正文"
        assert chunks[0]["heading_path"] == ["标题"]

    def test_multiple_sections(self, default_splitter):
        text = "# H1\n\n段落A\n\n## H2\n\n段落B"
        chunks = default_splitter.split(text)
        assert len(chunks) == 2
        assert chunks[0]["heading_path"] == ["H1"]
        assert chunks[1]["heading_path"] == ["H1", "H2"]

    def test_nested_headings_path(self, default_splitter):
        text = "# 一级\n\n## 二级\n\n### 三级\n\n正文"
        chunks = default_splitter.split(text)
        assert chunks[-1]["heading_path"] == ["一级", "二级", "三级"]

    def test_heading_level_reset(self, default_splitter):
        text = "# H1\n\n## H2\n\n# 新H1\n\n正文"
        chunks = default_splitter.split(text)
        assert chunks[-1]["heading_path"] == ["新H1"]

    def test_no_headings(self, default_splitter):
        chunks = default_splitter.split("这是一段没有 Markdown 标题的内容。")
        assert len(chunks) == 1
        assert chunks[0]["heading_path"] == []

    def test_empty_text_returns_empty(self, default_splitter):
        assert default_splitter.split("") == []

    def test_short_whole_section_is_preserved(self):
        splitter = HierarchicalMarkdownSplitter(
            ChunkConfig(chunk_size=200, chunk_overlap=30, min_chunk_size=100)
        )
        chunks = splitter.split("# H1\n\n短文本")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "短文本"
        assert chunks[0]["section_index"] == 0


class TestChunkIndex:
    def test_chunk_index_sequential(self, default_splitter):
        text = "\n\n".join(f"# 标题{i}\n\n{'正文' * 5}" for i in range(5))
        chunks = default_splitter.split(text)
        for index, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == index


class TestRecursiveSplit:
    def test_oversized_section_gets_split(self, tight_splitter):
        long_content = "这是很长的内容" * 20
        chunks = tight_splitter.split(f"# 标题\n\n{long_content}")
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["content"]) <= tight_splitter.cfg.chunk_size * 2

    def test_all_chunks_have_same_heading(self, tight_splitter):
        long_content = "重复内容" * 30
        chunks = tight_splitter.split(f"# 总标题\n\n{long_content}")
        assert all(chunk["heading_path"] == ["总标题"] for chunk in chunks)

    def test_overlap_creates_continuity(self, tight_splitter):
        long_content = "ABCDEFGH " * 20
        chunks = tight_splitter.split(f"# 标题\n\n{long_content}")
        if len(chunks) > 1:
            end_of_first = chunks[0]["content"][-tight_splitter.cfg.chunk_overlap :]
            tokens_end = set(end_of_first.split())
            tokens_start = set(chunks[1]["content"].split())
            assert tokens_end & tokens_start


class TestCharCount:
    def test_char_count_matches_content_length(self, default_splitter):
        chunks = default_splitter.split("# 标题\n\n" + "测试内容" * 10)
        for chunk in chunks:
            assert chunk["char_count"] == len(chunk["content"])
