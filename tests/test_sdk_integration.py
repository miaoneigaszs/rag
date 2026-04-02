"""SDK integration tests against a real Docker Qdrant instance."""

import asyncio

import pytest

from rag import DeleteOptions, DeleteTarget, DocumentSource, IndexRequest, SearchRequest, create_sdk
from tests.integration_helpers import DeterministicEmbeddingService, patch_docker_qdrant_clients

pytestmark = [pytest.mark.integration, pytest.mark.docker_qdrant]


class TestKnowledgeSDKDockerIntegration:
    def test_bytes_source_round_trip_docker(self, docker_qdrant_config_factory, sample_markdown):
        import rag.engine as engine_module

        cfg, namespace = docker_qdrant_config_factory(suffix="bytes_roundtrip", namespace="bytes-docs")

        import rag.vector_store as vector_store_module

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(engine_module, "EmbeddingService", DeterministicEmbeddingService)
            patch_docker_qdrant_clients(monkeypatch, vector_store_module)
            sdk = create_sdk(base_config=cfg)
            sdk.startup_sync()
            try:
                index_result = sdk.index(
                    IndexRequest(
                        source=DocumentSource.from_bytes(
                            sample_markdown.encode("utf-8"),
                            source_name="bytes-guide.md",
                            upload_origin="integration-bytes",
                        ),
                        namespace=namespace,
                    )
                )
                before_delete = sdk.search(SearchRequest(query="pytest", namespace=namespace, top_k=8))
                delete_result = sdk.delete(
                    DeleteTarget.from_source_path(index_result.source_path),
                    DeleteOptions(namespace=namespace),
                )
                after_delete = sdk.search(SearchRequest(query="pytest", namespace=namespace, top_k=8))
            finally:
                sdk.shutdown_sync()

        assert before_delete.count > 0
        assert any(item.source_path == index_result.source_path for item in before_delete.items)
        assert delete_result.source_path == index_result.source_path
        assert after_delete.count == 0

    def test_same_content_different_logical_sources_coexist_docker(self, docker_qdrant_config_factory, sample_markdown):
        import rag.engine as engine_module

        cfg, namespace = docker_qdrant_config_factory(suffix="logical_identity", namespace="identity-docs")

        import rag.vector_store as vector_store_module

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(engine_module, "EmbeddingService", DeterministicEmbeddingService)
            patch_docker_qdrant_clients(monkeypatch, vector_store_module)
            sdk = create_sdk(base_config=cfg)
            sdk.startup_sync()
            try:
                first = sdk.index(
                    IndexRequest(
                        source=DocumentSource.from_text(
                            sample_markdown,
                            source_name="guide-a.md",
                            upload_origin="source-a",
                        ),
                        namespace=namespace,
                    )
                )
                second = sdk.index(
                    IndexRequest(
                        source=DocumentSource.from_text(
                            sample_markdown,
                            source_name="guide-b.md",
                            upload_origin="source-b",
                        ),
                        namespace=namespace,
                    )
                )
                before_delete = sdk.search(SearchRequest(query="pytest", namespace=namespace, top_k=12))
                first_delete = sdk.delete(
                    DeleteTarget.from_source_path(first.source_path),
                    DeleteOptions(namespace=namespace),
                )
                after_first_delete = sdk.search(SearchRequest(query="pytest", namespace=namespace, top_k=12))
                second_delete = sdk.delete(
                    DeleteTarget.from_source_path(second.source_path),
                    DeleteOptions(namespace=namespace),
                )
                after_second_delete = sdk.search(SearchRequest(query="pytest", namespace=namespace, top_k=12))
            finally:
                sdk.shutdown_sync()

        before_paths = {item.source_path for item in before_delete.items}
        after_first_paths = {item.source_path for item in after_first_delete.items}

        assert first.source_path != second.source_path
        assert first_delete.source_path == first.source_path
        assert second_delete.source_path == second.source_path
        assert first.source_path in before_paths
        assert second.source_path in before_paths
        assert first.source_path not in after_first_paths
        assert second.source_path in after_first_paths
        assert after_second_delete.count == 0

    def test_async_bytes_source_round_trip_docker(self, docker_qdrant_config_factory, sample_markdown):
        import rag.engine as engine_module

        cfg, namespace = docker_qdrant_config_factory(suffix="bytes_async", namespace="async-bytes-docs")

        async def run_round_trip():
            sdk = create_sdk(base_config=cfg)
            await sdk.startup()
            try:
                index_result = await sdk.aindex(
                    IndexRequest(
                        source=DocumentSource.from_bytes(
                            sample_markdown.encode("utf-8"),
                            source_name="async-bytes-guide.md",
                            upload_origin="integration-bytes-async",
                        ),
                        namespace=namespace,
                    )
                )
                before_delete = await sdk.asearch(
                    SearchRequest(query="Docker Compose", namespace=namespace, top_k=8)
                )
                delete_result = await sdk.adelete(
                    DeleteTarget.from_source_path(index_result.source_path),
                    DeleteOptions(namespace=namespace),
                )
                after_delete = await sdk.asearch(
                    SearchRequest(query="Docker Compose", namespace=namespace, top_k=8)
                )
                return index_result, before_delete, delete_result, after_delete
            finally:
                await sdk.shutdown()

        import rag.vector_store as vector_store_module

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(engine_module, "EmbeddingService", DeterministicEmbeddingService)
            patch_docker_qdrant_clients(monkeypatch, vector_store_module)
            index_result, before_delete, delete_result, after_delete = asyncio.run(run_round_trip())

        assert before_delete.count > 0
        assert any(item.source_path == index_result.source_path for item in before_delete.items)
        assert delete_result.source_path == index_result.source_path
        assert after_delete.count == 0
