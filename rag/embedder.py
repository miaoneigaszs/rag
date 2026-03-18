"""
rag/embedder.py
===============
Embedding 服务：支持 OpenAI 官方 和 OpenAI 兼容中转站（SiliconFlow 等）。
"""

from __future__ import annotations

import logging
from typing import List

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

from .config import EmbeddingConfig

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore


class EmbeddingService:
    """
    统一 Embedding 接口。

    通过 EmbeddingConfig.provider 切换：
      "openai" → openai_api_key + openai_base_url + openai_model
      "proxy"  → proxy_api_key  + proxy_base_url  + proxy_model
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        if cfg.provider == "openai":
            self._client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
            self._async_client = AsyncOpenAI(
                api_key=cfg.openai_api_key, base_url=cfg.openai_base_url
            )
            self._model = cfg.openai_model
        else:
            self._client = OpenAI(api_key=cfg.proxy_api_key, base_url=cfg.proxy_base_url)
            self._async_client = AsyncOpenAI(
                api_key=cfg.proxy_api_key, base_url=cfg.proxy_base_url
            )
            self._model = cfg.proxy_model

        logger.info(f"[Embedding] provider={cfg.provider}, model={self._model}, dim={cfg.dimension}")

    # ------------------------------------------------------------------
    # 同步接口
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """同步批量生成向量（含指数退避重试）。"""
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_all(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """对大量文本分批处理，附进度条。"""
        all_vectors: List[List[float]] = []
        batches = [
            texts[i : i + self.cfg.batch_size]
            for i in range(0, len(texts), self.cfg.batch_size)
        ]
        for batch in tqdm(batches, desc="Embedding", disable=not show_progress):
            all_vectors.extend(self.embed_batch(batch))
        return all_vectors

    def embed_single(self, text: str) -> List[float]:
        """同步单条 Embedding 快捷方法。"""
        return self.embed_batch([text])[0]

    # ------------------------------------------------------------------
    # 异步接口
    # ------------------------------------------------------------------

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """异步批量生成向量（原生 AsyncOpenAI，不占线程）。"""
        resp = await self._async_client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    async def embed_single_async(self, text: str) -> List[float]:
        """异步单条 Embedding 快捷方法。"""
        results = await self.embed_batch_async([text])
        return results[0]
