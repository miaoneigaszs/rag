"""
rag/chunker.py
==============
层级感知的 Markdown 切块器。

策略（两阶段）：
  阶段一：按 Markdown 标题（# ~ ######）切出语义节，保留完整标题路径。
  阶段二：对超大节递归字符分割，保留重叠以维持上下文。
          分隔符优先级：段落 > 换行 > 中文句号 > 英文句号 > 空格 > 字符
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .config import ChunkConfig


class HierarchicalMarkdownSplitter:
    """层级感知的 Markdown 切块器。"""

    SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, cfg: ChunkConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def split(self, text: str, source_file: str = "") -> List[Dict[str, Any]]:
        """
        切块入口。

        Returns:
            List[dict]，每个 dict 包含：
              content       : str
              heading_path  : List[str]
              chunk_index   : int        全局线性序号
              section_index : int        所属语义节序号（用于溯源）
              char_count    : int
        """
        raw_sections = self._split_by_headings(text)
        chunks: List[Dict[str, Any]] = []

        for section_index, section in enumerate(raw_sections):
            content = section["content"].strip()
            if not content:
                continue

            # ── 短节：合并到前一个 chunk，而不是丢弃 ──────────────────────
            if len(content) < self.cfg.min_chunk_size:
                if chunks:
                    chunks[-1]["content"] += "\n\n" + content
                    chunks[-1]["char_count"] = len(chunks[-1]["content"])
                else:
                    # 没有前一个 chunk，宁可保留短内容也不丢弃
                    chunks.append(
                        {
                            "content": content,
                            "heading_path": section["heading_path"],
                            "section_index": section_index,
                            "char_count": len(content),
                        }
                    )
                continue

            # ── 正常节：直接入块或递归分割 ────────────────────────────────
            if len(content) <= self.cfg.chunk_size:
                chunks.append(
                    {
                        "content": content,
                        "heading_path": section["heading_path"],
                        "section_index": section_index,
                        "char_count": len(content),
                    }
                )
            else:
                for sub in self._recursive_split(content, self.SEPARATORS):
                    sub = sub.strip()
                    if len(sub) >= self.cfg.min_chunk_size:
                        chunks.append(
                            {
                                "content": sub,
                                "heading_path": section["heading_path"],
                                "section_index": section_index,
                                "char_count": len(sub),
                            }
                        )

        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i

        return chunks

    # ------------------------------------------------------------------
    # 阶段一：按标题切节
    # ------------------------------------------------------------------

    def _split_by_headings(self, text: str) -> List[Dict[str, Any]]:
        """按 Markdown 标题行扫描，切出各语义节。"""
        lines = text.split("\n")
        heading_stack: List[str] = [""] * 6
        current_lines: List[str] = []
        sections: List[Dict[str, Any]] = []

        def _flush() -> None:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    {
                        "content": content,
                        "heading_path": [h for h in heading_stack if h],
                    }
                )

        for line in lines:
            m = self.HEADING_PATTERN.match(line)
            if m:
                _flush()
                current_lines = []
                level = len(m.group(1))
                title = m.group(2).strip()
                heading_stack[level - 1] = title
                for i in range(level, 6):
                    heading_stack[i] = ""
                # 修复：标题已存入 heading_path，内容里不再重复添加标题行
            else:
                current_lines.append(line)

        _flush()
        return sections

    # ------------------------------------------------------------------
    # 阶段二：递归字符分割
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归字符分割，附带重叠窗口。"""
        if len(text) <= self.cfg.chunk_size:
            return [text] if text.strip() else []

        # 选第一个在文本中出现的分隔符
        sep = ""
        for s in separators:
            if s == "" or s in text:
                sep = s
                break

        parts = text.split(sep) if sep else list(text)
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(sep)
            if current_len + part_len > self.cfg.chunk_size and current:
                chunk_str = sep.join(current)
                if chunk_str.strip():
                    chunks.append(chunk_str)
                # 保留重叠尾部：用 continue 而非 break，尽量填满 overlap 窗口
                kept: List[str] = []
                overlap_acc = 0
                for p in reversed(current):
                    if overlap_acc + len(p) + len(sep) <= self.cfg.chunk_overlap:
                        kept.insert(0, p)
                        overlap_acc += len(p) + len(sep)
                    else:
                        break  # 保证重叠内容是原文末尾的连续片段
                current = kept
                current_len = overlap_acc

            current.append(part)
            current_len += part_len

        if current:
            chunk_str = sep.join(current)
            if chunk_str.strip():
                chunks.append(chunk_str)

        # 递归处理仍然过大的 chunk
        next_seps = separators[1:] if len(separators) > 1 else [""]
        final: List[str] = []
        for c in chunks:
            if len(c) > self.cfg.chunk_size and next_seps != [""]:
                final.extend(self._recursive_split(c, next_seps))
            else:
                final.append(c)

        return final