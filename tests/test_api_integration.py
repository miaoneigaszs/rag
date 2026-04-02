"""API integration tests against a real Docker Qdrant instance."""

from fastapi.testclient import TestClient
import pytest

from rag.api import create_app
from rag.service import AgentKnowledgeService
from tests.integration_helpers import DeterministicEmbeddingService, patch_docker_qdrant_clients

pytestmark = [pytest.mark.integration, pytest.mark.docker_qdrant]


def create_api_integration_client(tmp_path, docker_qdrant_config_factory, monkeypatch, *, suffix: str, api_key: str | None = None) -> tuple[TestClient, object]:
    import rag.engine as engine_module
    import rag.vector_store as vector_store_module

    cfg, _ = docker_qdrant_config_factory(suffix=suffix)
    monkeypatch.setattr(engine_module, "EmbeddingService", DeterministicEmbeddingService)
    patch_docker_qdrant_clients(monkeypatch, vector_store_module)

    service = AgentKnowledgeService(base_config=cfg)
    app = create_app(service=service, api_key=api_key, upload_dir=str(tmp_path / "uploads"))
    return TestClient(app), tmp_path / "uploads"


class TestAPIDockerIntegration:
    def test_upload_retrieve_delete_by_source_path_respects_namespace_precedence(
        self,
        tmp_path,
        docker_qdrant_config_factory,
        sample_markdown,
        monkeypatch,
    ):
        client, upload_dir = create_api_integration_client(
            tmp_path,
            docker_qdrant_config_factory,
            monkeypatch,
            suffix="api_source_path",
        )

        with client:
            upload_response = client.post(
                "/documents/upload",
                headers={"X-Namespace": "header-space"},
                data={"namespace": "body-space", "extra_meta_json": '{"bucket": "api"}'},
                files={"file": ("handbook.md", sample_markdown.encode("utf-8"), "text/markdown")},
            )
            upload_payload = upload_response.json()
            source_path = upload_payload["data"]["source_path"]

            retrieve_hit = client.post(
                "/retrieve",
                headers={"X-Namespace": "body-space"},
                json={"query": "pytest", "top_k": 8},
            )
            retrieve_miss = client.post(
                "/retrieve",
                headers={"X-Namespace": "header-space"},
                json={"query": "pytest", "top_k": 8},
            )
            delete_response = client.post(
                "/documents/delete",
                headers={"X-Namespace": "header-space"},
                json={"file_identifier": source_path, "namespace": "body-space"},
            )
            retrieve_after_delete = client.post(
                "/retrieve",
                headers={"X-Namespace": "body-space"},
                json={"query": "pytest", "top_k": 8},
            )

        assert upload_response.status_code == 200
        assert upload_payload["meta"]["namespace"] == "body-space"
        assert source_path == "bytes://api-upload/handbook.md"
        assert retrieve_hit.status_code == 200
        assert retrieve_hit.json()["data"]["count"] > 0
        assert any(item["source_path"] == source_path for item in retrieve_hit.json()["data"]["results"])
        assert retrieve_miss.status_code == 200
        assert retrieve_miss.json()["data"]["count"] == 0
        assert delete_response.status_code == 200
        assert delete_response.json()["data"]["source_path"] == source_path
        assert retrieve_after_delete.status_code == 200
        assert retrieve_after_delete.json()["data"]["count"] == 0
        assert upload_dir.exists()
        assert list(upload_dir.iterdir()) == []

    def test_upload_retrieve_delete_by_legacy_basename_uses_header_namespace(
        self,
        tmp_path,
        docker_qdrant_config_factory,
        sample_markdown,
        monkeypatch,
    ):
        client, upload_dir = create_api_integration_client(
            tmp_path,
            docker_qdrant_config_factory,
            monkeypatch,
            suffix="api_legacy_delete",
        )

        with client:
            upload_response = client.post(
                "/documents/upload",
                headers={"X-Namespace": "legacy-space"},
                files={"file": ("legacy-guide.md", sample_markdown.encode("utf-8"), "text/markdown")},
            )
            upload_payload = upload_response.json()

            retrieve_before_delete = client.post(
                "/retrieve",
                headers={"X-Namespace": "legacy-space"},
                json={"query": "Docker Compose", "top_k": 8},
            )
            delete_response = client.post(
                "/documents/delete",
                headers={"X-Namespace": "legacy-space"},
                json={"file_identifier": "legacy-guide.md"},
            )
            retrieve_after_delete = client.post(
                "/retrieve",
                headers={"X-Namespace": "legacy-space"},
                json={"query": "Docker Compose", "top_k": 8},
            )

        assert upload_response.status_code == 200
        assert upload_payload["meta"]["namespace"] == "legacy-space"
        assert upload_payload["data"]["source_path"] == "bytes://api-upload/legacy-guide.md"
        assert retrieve_before_delete.status_code == 200
        assert retrieve_before_delete.json()["data"]["count"] > 0
        assert delete_response.status_code == 200
        assert delete_response.json()["data"]["request_identifier"] == "legacy-guide.md"
        assert delete_response.json()["data"]["source_path"] == "bytes://api-upload/legacy-guide.md"
        assert delete_response.json()["data"]["resolved_source_paths"] == ["bytes://api-upload/legacy-guide.md"]
        assert retrieve_after_delete.status_code == 200
        assert retrieve_after_delete.json()["data"]["count"] == 0
        assert upload_dir.exists()
        assert list(upload_dir.iterdir()) == []


    def test_api_key_authenticated_round_trip_and_health_anonymous(
        self,
        tmp_path,
        docker_qdrant_config_factory,
        sample_markdown,
        monkeypatch,
    ):
        client, upload_dir = create_api_integration_client(
            tmp_path,
            docker_qdrant_config_factory,
            monkeypatch,
            suffix="api_key_roundtrip",
            api_key="secret-key",
        )

        with client:
            health_response = client.get("/health")
            unauthorized_upload = client.post(
                "/documents/upload",
                files={"file": ("auth-guide.md", sample_markdown.encode("utf-8"), "text/markdown")},
            )
            upload_response = client.post(
                "/documents/upload",
                headers={"X-API-Key": "secret-key", "X-Namespace": "auth-space"},
                files={"file": ("auth-guide.md", sample_markdown.encode("utf-8"), "text/markdown")},
            )
            source_path = upload_response.json()["data"]["source_path"]
            retrieve_response = client.post(
                "/retrieve",
                headers={"X-API-Key": "secret-key", "X-Namespace": "auth-space"},
                json={"query": "pytest", "top_k": 8},
            )
            delete_response = client.post(
                "/documents/delete",
                headers={"X-API-Key": "secret-key", "X-Namespace": "auth-space"},
                json={"file_identifier": source_path},
            )
            retrieve_after_delete = client.post(
                "/retrieve",
                headers={"X-API-Key": "secret-key", "X-Namespace": "auth-space"},
                json={"query": "pytest", "top_k": 8},
            )

        assert health_response.status_code == 200
        assert unauthorized_upload.status_code == 401
        assert upload_response.status_code == 200
        assert retrieve_response.status_code == 200
        assert retrieve_response.json()["data"]["count"] > 0
        assert delete_response.status_code == 200
        assert delete_response.json()["data"]["source_path"] == source_path
        assert retrieve_after_delete.status_code == 200
        assert retrieve_after_delete.json()["data"]["count"] == 0
        assert upload_dir.exists()
        assert list(upload_dir.iterdir()) == []

    def test_bearer_authenticated_round_trip(
        self,
        tmp_path,
        docker_qdrant_config_factory,
        sample_markdown,
        monkeypatch,
    ):
        client, upload_dir = create_api_integration_client(
            tmp_path,
            docker_qdrant_config_factory,
            monkeypatch,
            suffix="bearer_roundtrip",
            api_key="bearer-secret",
        )

        with client:
            upload_response = client.post(
                "/documents/upload",
                headers={"Authorization": "Bearer bearer-secret", "X-Namespace": "bearer-space"},
                files={"file": ("bearer-guide.md", sample_markdown.encode("utf-8"), "text/markdown")},
            )
            source_path = upload_response.json()["data"]["source_path"]
            retrieve_response = client.post(
                "/retrieve",
                headers={"Authorization": "Bearer bearer-secret", "X-Namespace": "bearer-space"},
                json={"query": "Docker Compose", "top_k": 8},
            )
            delete_response = client.post(
                "/documents/delete",
                headers={"Authorization": "Bearer bearer-secret", "X-Namespace": "bearer-space"},
                json={"file_identifier": source_path},
            )
            retrieve_after_delete = client.post(
                "/retrieve",
                headers={"Authorization": "Bearer bearer-secret", "X-Namespace": "bearer-space"},
                json={"query": "Docker Compose", "top_k": 8},
            )

        assert upload_response.status_code == 200
        assert retrieve_response.status_code == 200
        assert retrieve_response.json()["data"]["count"] > 0
        assert delete_response.status_code == 200
        assert delete_response.json()["data"]["source_path"] == source_path
        assert retrieve_after_delete.status_code == 200
        assert retrieve_after_delete.json()["data"]["count"] == 0
        assert upload_dir.exists()
        assert list(upload_dir.iterdir()) == []
