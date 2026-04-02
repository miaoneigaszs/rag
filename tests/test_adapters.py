"""MCP and skill adapter tests."""

import asyncio

import pytest

from rag.mcp import MCPAdapter
from rag.service import AgentKnowledgeService
from rag.skill import KnowledgeSkill


class FakeEngine:
    def __init__(self):
        self.last_retrieval_stats = {"result_count": 1}

    def index_file(self, file_path, extra_meta=None, force_reindex=False):
        return {"status": "ok", "source_path": file_path, "extra_meta": extra_meta or {}, "force_reindex": force_reindex}

    def delete_file(self, file_identifier):
        self.deleted = file_identifier

    def retrieve(self, query, top_k=5, filter_conditions=None, skip_rerank=False, score_threshold=0.0):
        return [
            {
                "doc_id": "doc-1",
                "content": query,
                "source_file": "doc.md",
                "source_path": "/tmp/doc.md",
                "heading_str": "标题",
                "score": 0.8,
            }
        ][:top_k]

    def format_results_for_llm(self, results):
        return f"ctx:{len(results)}"

    def get_last_index_stats(self):
        return {"status": "ok"}

    def get_last_retrieval_stats(self):
        return dict(self.last_retrieval_stats)


@pytest.fixture()
def service() -> AgentKnowledgeService:
    return AgentKnowledgeService(engine=FakeEngine())


class TestMCPAdapter:
    def test_lists_expected_tools(self, service):
        adapter = MCPAdapter(service=service)

        tool_names = {tool["name"] for tool in adapter.list_tools()}
        assert tool_names == {
            "index_document",
            "retrieve_knowledge",
            "delete_document",
            "get_retrieval_stats",
        }

    def test_retrieve_tool(self, service):
        adapter = MCPAdapter(service=service)

        result = asyncio.run(adapter.call_tool("retrieve_knowledge", {"query": "你好", "top_k": 1}))

        assert result["count"] == 1
        assert result["formatted_context"] == "ctx:1"


class TestKnowledgeSkill:
    def test_manifest(self, service):
        skill = KnowledgeSkill(service=service, default_top_k=3)

        manifest = skill.manifest()
        assert manifest["name"] == "retrieve_knowledge"
        assert manifest["inputs"]["top_k"]["default"] == 3

    def test_run(self, service):
        skill = KnowledgeSkill(service=service, default_top_k=2)

        result = asyncio.run(skill.run("什么是答案"))

        assert result["count"] == 1
        assert result["formatted_context"] == "ctx:1"
