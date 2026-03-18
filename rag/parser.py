"""
rag/parser.py
=============
文档解析层：Docling → Unstructured → 纯文本，三级路由 + 优雅降级。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore

try:
    from docling.document_converter import DocumentConverter
    _HAS_DOCLING = True
except ImportError:
    _HAS_DOCLING = False

try:
    from unstructured.partition.auto import partition
    _HAS_UNSTRUCTURED = True
except ImportError:
    _HAS_UNSTRUCTURED = False


class DocumentParser:
    """
    文档解析器路由。

    路由规则：
      PDF / DOCX / HTML / PPTX → 优先 Docling（结构感知，原生 Markdown 输出）
                                 Docling 失败时降级 → Unstructured
      EML / MSG / XLSX / CSV 等 → Unstructured（元素级解析）
      TXT / MD / JSON 等        → 直接读取（UTF-8）

    返回值：(markdown_text, file_type)
      file_type ∈ {"docling", "unstructured", "plain"}
    """

    DOCLING_TYPES = frozenset({".pdf", ".docx", ".doc", ".html", ".htm", ".pptx"})
    UNSTRUCTURED_TYPES = frozenset(
        {".eml", ".msg", ".xlsx", ".xls", ".csv", ".rst", ".rtf", ".odt", ".epub"}
    )
    PLAIN_TEXT_TYPES = frozenset(
        {".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".log"}
    )

    @property
    def supported_extensions(self) -> frozenset[str]:
        return self.DOCLING_TYPES | self.UNSTRUCTURED_TYPES | self.PLAIN_TEXT_TYPES

    def parse(self, file_path: str) -> Tuple[str, str]:
        """
        解析文档，返回 (markdown_text, file_type)。

        Raises:
            FileNotFoundError: 文件不存在。
            RuntimeError: 所有解析器均失败。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()

        # ── 1. 纯文本直读 ───────────────────────────────────────────────────
        if ext in self.PLAIN_TEXT_TYPES:
            text = path.read_text(encoding="utf-8", errors="replace")
            logger.debug(f"[Parser] plain 读取: {path.name}, 长度={len(text)}")
            return text, "plain"

        # ── 2. Docling（PDF / DOCX / HTML / PPTX）──────────────────────────
        if ext in self.DOCLING_TYPES:
            if not _HAS_DOCLING:
                logger.warning("[Parser] Docling 未安装，跳过（pip install docling）")
            else:
                result = self._parse_with_docling(file_path)
                if result is not None:
                    return result, "docling"
                logger.warning(f"[Parser] Docling 解析失败，降级到 Unstructured: {path.name}")

        # ── 3. Unstructured（扩展格式 + 降级）──────────────────────────────
        if not _HAS_UNSTRUCTURED:
            raise RuntimeError(
                f"无法解析 {path.name}：Docling 不可用或失败，且 Unstructured 未安装。\n"
                "请执行: pip install 'unstructured[all-docs]'"
            )

        result = self._parse_with_unstructured(file_path)
        if result is not None:
            return result, "unstructured"

        raise RuntimeError(
            f"所有解析器均失败，无法处理文件: {file_path}"
        )

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _parse_with_docling(self, file_path: str) -> Optional[str]:
        """使用 Docling 解析，返回 Markdown 文本；失败时返回 None。"""
        try:
            converter = DocumentConverter()
            result = converter.convert(file_path)
            md_text = result.document.export_to_markdown()
            logger.debug(f"[Docling] 解析成功: {Path(file_path).name}, 长度={len(md_text)}")
            return md_text
        except Exception as exc:
            logger.warning(f"[Docling] 解析异常: {exc}")
            return None

    def _parse_with_unstructured(self, file_path: str) -> Optional[str]:
        """
        使用 Unstructured 解析，将语义元素重组为带 Markdown 标题的文本；
        失败时返回 None。
        """
        try:
            elements = partition(filename=file_path, strategy="hi_res")
            lines: list[str] = []

            for el in elements:
                el_type = type(el).__name__
                text = str(el).strip()

                if el_type in ("Title", "Header"):
                    level = int(getattr(el.metadata, "category_depth", 1) or 1)
                    level = max(1, min(level, 6))
                    lines.append(f"{'#' * level} {text}")

                elif el_type == "Table":
                    html_repr = getattr(el.metadata, "text_as_html", None)
                    lines.append(f"\n{html_repr or text}\n")

                elif el_type == "ListItem":
                    lines.append(f"- {text}")

                else:
                    if text:
                        lines.append(text)

                lines.append("")  # 元素间空行

            md_text = "\n".join(lines)
            logger.debug(
                f"[Unstructured] 解析成功: {Path(file_path).name}, 元素数={len(elements)}"
            )
            return md_text

        except Exception as exc:
            logger.warning(f"[Unstructured] 解析异常: {exc}")
            return None
