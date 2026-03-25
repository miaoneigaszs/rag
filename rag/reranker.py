"""Reranker API 客户端封装。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import RerankerConfig

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)  # type: ignore


class APIReranker:
    """兼容 OpenAI 风格 `/rerank` 接口的轻量客户端。"""

    def __init__(self, cfg: RerankerConfig) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("缺少 httpx 依赖: pip install httpx") from exc

        self.cfg = cfg
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        self._http = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=30.0,
            trust_env=cfg.trust_env,
        )
        self._async_http = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=30.0,
            trust_env=cfg.trust_env,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """同步调用 rerank 接口。"""
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
            return sorted(results, key=lambda item: item.get("relevance_score", 0), reverse=True)
        except Exception as exc:
            logger.warning(f"[Reranker] 同步调用失败: {exc}")
            return []

    async def async_rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """异步调用 rerank 接口。"""
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
            return sorted(results, key=lambda item: item.get("relevance_score", 0), reverse=True)
        except Exception as exc:
            logger.warning(f"[Reranker] 异步调用失败: {exc}")
            return []

    async def close(self) -> None:
        """关闭同步与异步 HTTP 客户端。"""
        try:
            self._http.close()
        except Exception as exc:
            logger.warning(f"[Reranker] 关闭同步客户端失败: {exc}")
        try:
            await self._async_http.aclose()
        except Exception as exc:
            logger.warning(f"[Reranker] 关闭异步客户端失败: {exc}")
