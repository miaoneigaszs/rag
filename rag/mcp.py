"""Lightweight MCP-style adapter above the knowledge service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .service import AgentKnowledgeService


class MCPAdapter:
    """Expose the knowledge service as MCP-friendly tools."""

    def __init__(self, service: Optional[AgentKnowledgeService] = None) -> None:
        self.service = service or AgentKnowledgeService()

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "index_document",
                "description": "Index a document into the knowledge service",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "namespace": {"type": "string"},
                        "extra_meta": {"type": "object"},
                        "force_reindex": {"type": "boolean"},
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "retrieve_knowledge",
                "description": "Retrieve relevant knowledge for a query",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "namespace": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                        "filter_conditions": {"type": "object"},
                        "skip_rerank": {"type": "boolean"},
                        "score_threshold": {"type": "number", "minimum": 0},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "delete_document",
                "description": "Delete a previously indexed document by source path or legacy basename",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_identifier": {"type": "string", "description": "source_path or legacy basename"},
                        "namespace": {"type": "string"},
                    },
                    "required": ["file_identifier"],
                },
            },
            {
                "name": "get_retrieval_stats",
                "description": "Get observability stats from the last retrieval",
                "input_schema": {
                    "type": "object",
                    "properties": {"namespace": {"type": "string"}},
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = arguments or {}

        if name == "index_document":
            return await self.service.index_document(
                payload["file_path"],
                namespace=payload.get("namespace"),
                extra_meta=payload.get("extra_meta"),
                force_reindex=payload.get("force_reindex", False),
            )

        if name == "retrieve_knowledge":
            return await self.service.retrieve(
                payload["query"],
                namespace=payload.get("namespace"),
                top_k=payload.get("top_k", 5),
                filter_conditions=payload.get("filter_conditions"),
                skip_rerank=payload.get("skip_rerank", False),
                score_threshold=payload.get("score_threshold", 0.0),
            )

        if name == "delete_document":
            return await self.service.delete_document(
                payload["file_identifier"],
                namespace=payload.get("namespace"),
            )

        if name == "get_retrieval_stats":
            return self.service.get_last_retrieval_stats(namespace=payload.get("namespace"))

        raise KeyError(f"Unknown MCP tool: {name}")
