"""Skill-style adapter above the knowledge service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .service import AgentKnowledgeService


class KnowledgeSkill:
    """Reusable skill wrapper for a specific retrieval workflow."""

    def __init__(
        self,
        *,
        name: str = "retrieve_knowledge",
        description: str = "Retrieve private knowledge for an agent workflow",
        service: Optional[AgentKnowledgeService] = None,
        default_namespace: str = "default",
        default_top_k: int = 5,
        default_filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.service = service or AgentKnowledgeService()
        self.default_namespace = default_namespace
        self.default_top_k = default_top_k
        self.default_filter_conditions = default_filter_conditions or {}

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": {
                "query": {"type": "string", "required": True},
                "namespace": {"type": "string", "required": False, "default": self.default_namespace},
                "top_k": {"type": "integer", "required": False, "default": self.default_top_k},
                "filter_conditions": {
                    "type": "object",
                    "required": False,
                    "default": self.default_filter_conditions,
                },
                "skip_rerank": {"type": "boolean", "required": False, "default": False},
                "score_threshold": {"type": "number", "required": False, "default": 0.0},
            },
        }

    async def run(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        return await self.service.retrieve(
            query,
            namespace=namespace or self.default_namespace,
            top_k=top_k or self.default_top_k,
            filter_conditions=filter_conditions or dict(self.default_filter_conditions),
            skip_rerank=skip_rerank,
            score_threshold=score_threshold,
        )
