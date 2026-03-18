"""
rag/reranker.py
===============
Reranker 封装：调用 OpenAI 兼容的 /rerank 接口（SiliconFlow / Cohere / Jina 等）。

修复项：
  - 提供显式 close() 方法，RAGEngine.shutdown() 通过接口关闭，
    不再直接访问私有属性 _http / _async_http。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, reraise, stop_after_attempt, wait_exponential

from .config import RerankerConfig

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore


class APIReranker:
    """
    调用 OpenAI 兼容的 Rerank API。

    接口规范：
      POST /rerank
      Body: {"model": "...", "query": "...", "documents": [...], "top_n": N}
      Response: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}

    同时持有同步和异步 httpx 客户端，供不同检索路径选择。
    """

    def __init__(self, cfg: RerankerConfig) -> None:
        self.cfg = cfg
        _headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        self._http = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            headers=_headers,
            timeout=30.0,
        )
        self._async_http = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers=_headers,
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # 同步 rerank
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        reraise=True,
    )
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        同步精排，返回 [{"index": int, "relevance_score": float}, ...]，按 score 降序。

        失败时记录 warning 并返回空列表（由调用方降级为 RRF 排序）。
        """
        top_n = top_n or self.cfg.top_n
        try:
            resp = self._http.post(
                "/rerank",
                json={
                    "model": self.cfg.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        except Exception as exc:
            logger.warning(f"[Reranker] 同步 API 调用失败: {exc}")
            return []

    # ------------------------------------------------------------------
    # 异步 rerank
    # ------------------------------------------------------------------

    async def async_rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """异步精排，供 retrieve_async 使用。"""
        top_n = top_n or self.cfg.top_n
        try:
            resp = await self._async_http.post(
                "/rerank",
                json={
                    "model": self.cfg.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        except Exception as exc:
            logger.warning(f"[Reranker] 异步 API 调用失败: {exc}")
            return []

    # ------------------------------------------------------------------
    # 生命周期（修复：封装关闭逻辑，不暴露私有属性）
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """关闭同步和异步 httpx 连接池（由 RAGEngine.shutdown() 调用）。"""
        try:
            self._http.close()
        except Exception as exc:
            logger.warning(f"[Reranker] 关闭同步连接池失败: {exc}")
        try:
            await self._async_http.aclose()
        except Exception as exc:
            logger.warning(f"[Reranker] 关闭异步连接池失败: {exc}")
