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
        path = Path(file_path).resolve()
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

    def _get_image_caption(self, image_bytes: bytes) -> str:
        """调用多模态模型生成图片描述"""
        try:
            import os
            import base64
            from openai import OpenAI

            api_key = os.getenv("VISION_API_KEY", os.getenv("OPENAI_API_KEY"))
            base_url = os.getenv("VISION_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
            model = os.getenv("VISION_MODEL", "gpt-4o-mini")

            if not api_key:
                return "未配置 Vision API Key，跳过图片描述。"

            client = OpenAI(api_key=api_key, base_url=base_url)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请详细描述这张图片的内容，包括其中的关键文字、图表趋势、表格数据以及任何逻辑关系。"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content or "未提取到有效描述"
        except Exception as e:
            logger.warning(f"多模态模型调用失败: {e}")
            return "图片描述提取失败"

    def _parse_with_docling(self, file_path: str) -> Optional[str]:
        """使用 Docling 解析，返回 Markdown 文本；支持图片提取和多模态描述。"""
        # Docling 导出的图片占位符固定为该字符串，不含任何 ID
        IMAGE_PLACEHOLDER = "<!-- image -->"

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat

            # 配置 PDF 提取图片
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = 1.0
            pipeline_options.generate_page_images = False
            pipeline_options.generate_picture_images = True

            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML, InputFormat.PPTX],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            result = converter.convert(file_path)

            # 导出 Markdown，优先使用 PLACEHOLDER 模式
            try:
                from docling_core.types.doc.base import ImageRefMode
                md_text = result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
            except ImportError:
                md_text = result.document.export_to_markdown()

            # ── 图片处理：按顺序替换占位符 ──────────────────────────────────
            # 占位符格式为固定的 <!-- image -->，不含 ID，必须按遍历顺序
            # 逐个替换（str.replace 第三参数 count=1），保证位置一一对应。
            if hasattr(result.document, "pictures") and result.document.pictures:
                image_dir = Path(file_path).parent / f"{Path(file_path).stem}_images"
                image_dir.mkdir(parents=True, exist_ok=True)

                for i, pic_obj in enumerate(result.document.pictures):
                    try:
                        # self_ref 形如 "#/pictures/0"，取最后一段作为文件名
                        p_id = getattr(pic_obj, "self_ref", f"picture_{i}").split("/")[-1]
                        img = pic_obj.get_image(result.document)

                        if img:
                            img_filename = f"image_{p_id}.png"
                            img_path = image_dir / img_filename
                            img.save(img_path)

                            import io
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            img_bytes = buffered.getvalue()

                            caption = self._get_image_caption(img_bytes)
                            replacement = (
                                f"\n\n> 🖼️ **图片内容信息** ({img_filename}):\n"
                                f"> {caption}\n"
                                f"> (图片本地路径: `{img_path.resolve()}`)\n\n"
                            )
                        else:
                            # 取到了 PictureItem 但无图像数据，用空白替换占位符
                            replacement = ""

                        if IMAGE_PLACEHOLDER in md_text:
                            # count=1：只替换第一个，保证顺序与 pictures 列表一致
                            md_text = md_text.replace(IMAGE_PLACEHOLDER, replacement, 1)
                        else:
                            # 占位符已耗尽（文档图片数 > 占位符数），追加到末尾
                            if replacement:
                                md_text += replacement

                    except Exception as img_exc:
                        logger.warning(f"[Docling] 处理图片 {i} 失败: {img_exc}")
                        # 失败时也要消耗掉对应占位符，避免后续图片错位
                        md_text = md_text.replace(
                            IMAGE_PLACEHOLDER,
                            f"\n\n> ⚠️ 图片 {i} 处理失败\n\n",
                            1,
                        )

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