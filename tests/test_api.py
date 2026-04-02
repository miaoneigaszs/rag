"""API layer tests."""

from pathlib import Path

from fastapi.openapi.models import OpenAPI
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.service import AgentKnowledgeService


def _resolve_schema_ref(openapi_schema, schema_or_ref):
    if not isinstance(schema_or_ref, dict):
        return schema_or_ref
    ref = schema_or_ref.get("$ref")
    if not ref:
        return schema_or_ref
    prefix = "#/components/schemas/"
    assert ref.startswith(prefix)
    return openapi_schema["components"]["schemas"][ref[len(prefix):]]


def _response_json_schema(openapi_schema, operation):
    response = operation["responses"]["200"]["content"]["application/json"]["schema"]
    return _resolve_schema_ref(openapi_schema, response)


def _find_component_schema_with_fields(openapi_schema, required_fields):
    required = set(required_fields)
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        properties = set((schema.get("properties") or {}).keys())
        if required.issubset(properties):
            return schema
    raise AssertionError(f"No component schema contains fields: {sorted(required)}")


class FakeEngine:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.index_calls = []
        self.retrieve_calls = []
        self.last_index_stats = {"status": "ok", "chunk_count": 2, "namespace": namespace}
        self.last_retrieval_stats = {"result_count": 1, "top_k": 3, "namespace": namespace}

    def collection_stats(self):
        return {"points_count": 2, "status": "green", "collection": f"rag_docs__{self.namespace}"}

    def index_file(
        self,
        file_path,
        extra_meta=None,
        force_reindex=False,
        display_source_name=None,
        display_source_path=None,
    ):
        file_exists = Path(file_path).exists()
        self.index_calls.append(
            (file_path, extra_meta or {}, force_reindex, file_exists, display_source_name, display_source_path)
        )
        return {
            "status": "ok",
            "source_file": display_source_name or Path(file_path).name,
            "source_path": display_source_path or file_path,
            "chunks": 2,
            "namespace": self.namespace,
        }

    def delete_file(self, file_identifier):
        self.deleted = file_identifier

    def retrieve(self, query, top_k=5, filter_conditions=None, skip_rerank=False, score_threshold=0.0):
        self.retrieve_calls.append((query, top_k, filter_conditions, skip_rerank, score_threshold))
        return [
            {
                "doc_id": f"doc-{self.namespace}",
                "content": f"答案:{self.namespace}",
                "source_file": "a.md",
                "source_path": f"/tmp/{self.namespace}/a.md",
                "heading_str": "标题",
                "score": 0.9,
            }
        ]

    def format_results_for_llm(self, results):
        return f"formatted:{self.namespace}:{len(results)}"

    def get_last_index_stats(self):
        return dict(self.last_index_stats)

    def get_last_retrieval_stats(self):
        return dict(self.last_retrieval_stats)


def create_test_client(tmp_path, api_key=None):
    created = {}

    def factory(namespace: str):
        engine = FakeEngine(namespace)
        created[namespace] = engine
        return engine

    service = AgentKnowledgeService(engine_factory=factory)
    app = create_app(service=service, api_key=api_key, upload_dir=str(tmp_path / "uploads"))
    return TestClient(app), service, created, tmp_path / "uploads"


class TestAPI:
    def test_health(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path)
        with client:
            response = client.get("/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["data"]["status"] == "ok"
        assert payload["data"]["namespace"] == "default"

    def test_index_document(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path)
        with client:
            response = client.post(
                "/documents/index",
                json={
                    "file_path": "docs/a.md",
                    "namespace": "tenant-a",
                    "extra_meta": {"tenant": "demo"},
                    "force_reindex": True,
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["data"]["status"] == "ok"
        assert payload["meta"]["index_stats"]["namespace"] == "tenant-a"

    def test_retrieve_uses_namespace_header(self, tmp_path):
        client, _, created, _ = create_test_client(tmp_path)
        with client:
            response = client.post(
                "/retrieve",
                headers={"X-Namespace": "kb-alpha"},
                json={"query": "什么是答案", "top_k": 3, "filter_conditions": {"source_file": "a.md"}},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["data"]["namespace"] == "kb-alpha"
        assert payload["data"]["formatted_context"] == "formatted:kb-alpha:1"
        assert "kb-alpha" in created

    def test_run_evaluation_with_inline_cases(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path)
        with client:
            response = client.post(
                "/evaluation/run",
                json={
                    "namespace": "eval-a",
                    "top_k": 1,
                    "cases": [
                        {
                            "query": "什么是答案",
                            "expected_ids": ["doc-eval-a"],
                            "metadata": {"bucket": "smoke"},
                        }
                    ],
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["data"]["hit_rate_at_k"] == 1.0
        assert payload["data"]["namespace"] == "eval-a"
        assert payload["data"]["details"][0]["metadata"]["bucket"] == "smoke"

    def test_auth_required_when_api_key_configured(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path, api_key="secret-key")
        with client:
            unauthorized = client.post("/retrieve", json={"query": "什么是答案"})
            health = client.get("/health")
            authorized = client.post(
                "/retrieve",
                headers={"X-API-Key": "secret-key"},
                json={"query": "什么是答案"},
            )

        assert unauthorized.status_code == 401
        assert health.status_code == 200
        assert authorized.status_code == 200

    def test_upload_document_indexes_and_cleans_temp_file(self, tmp_path):
        client, _, created, upload_dir = create_test_client(tmp_path)
        with client:
            response = client.post(
                "/documents/upload",
                data={"namespace": "uploads-a", "force_reindex": "true", "extra_meta_json": '{"source": "form"}'},
                files={"file": ("kb.md", b"# title\n\nbody", "text/markdown")},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["meta"]["namespace"] == "uploads-a"
        assert payload["data"]["source_path"] == "bytes://api-upload/kb.md"
        engine = created["uploads-a"]
        assert engine.index_calls[0][1]["sdk_source_path"] == "bytes://api-upload/kb.md"
        assert engine.index_calls[0][3] is True
        assert engine.index_calls[0][4] == "kb.md"
        assert engine.index_calls[0][5] == "bytes://api-upload/kb.md"
        assert upload_dir.exists()
        assert list(upload_dir.iterdir()) == []

    def test_upload_document_rejects_reserved_metadata_keys(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path)
        with client:
            response = client.post(
                "/documents/upload",
                data={"extra_meta_json": '{"sdk_source_path": "bad"}'},
                files={"file": ("kb.md", b"# title\n\nbody", "text/markdown")},
            )

        assert response.status_code == 400
        payload = response.json()
        assert payload["ok"] is False
        assert "metadata 不能覆盖保留字段" in payload["error"]



    def test_docs_and_redoc_are_public_without_api_key(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path, api_key=None)
        with client:
            docs_response = client.get("/docs")
            redoc_response = client.get("/redoc")

        assert docs_response.status_code == 200
        assert "/openapi.json" in docs_response.text
        assert redoc_response.status_code == 200
        assert "/openapi.json" in redoc_response.text

    def test_docs_and_redoc_require_auth_when_api_key_configured(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path, api_key="secret-key")
        with client:
            docs_response = client.get("/docs")
            redoc_response = client.get("/redoc")

        assert docs_response.status_code == 401
        assert redoc_response.status_code == 401

    def test_docs_and_redoc_render_with_api_key_or_bearer_auth(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path, api_key="secret-key")
        with client:
            docs_response = client.get("/docs", headers={"X-API-Key": "secret-key"})
            redoc_response = client.get("/redoc", headers={"Authorization": "Bearer secret-key"})

        assert docs_response.status_code == 200
        assert "/openapi.json" in docs_response.text
        assert redoc_response.status_code == 200
        assert "/openapi.json" in redoc_response.text

    def test_openapi_omits_security_when_api_key_not_configured(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path, api_key=None)
        with client:
            response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        paths = schema["paths"]
        assert "securitySchemes" not in schema.get("components", {})
        assert "security" not in paths["/retrieve"]["post"]
        assert "security" not in paths["/documents/upload"]["post"]
        assert "security" not in paths["/documents/delete"]["post"]
        assert "security" not in paths["/health"]["get"] or not paths["/health"]["get"]["security"]

    def test_openapi_requires_auth_when_api_key_configured(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path, api_key="secret-key")
        with client:
            response = client.get("/openapi.json")

        assert response.status_code == 401


    def test_openapi_schema_is_consumable_by_fastapi_model(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path, api_key="secret-key")
        with client:
            response = client.get("/openapi.json", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 200
        raw_schema = response.json()
        parsed_schema = OpenAPI.model_validate(raw_schema)
        round_tripped = parsed_schema.model_dump(mode="json", by_alias=True, exclude_none=True)

        assert round_tripped["openapi"].startswith("3.")
        assert parsed_schema.components is not None
        assert parsed_schema.components.securitySchemes is not None
        assert {"ApiKeyAuth", "BearerAuth"}.issubset(parsed_schema.components.securitySchemes.keys())

        upload_path_item = parsed_schema.paths["/documents/upload"]
        delete_path_item = parsed_schema.paths["/documents/delete"]
        upload_post = upload_path_item["post"] if isinstance(upload_path_item, dict) else upload_path_item.post
        delete_post = delete_path_item["post"] if isinstance(delete_path_item, dict) else delete_path_item.post
        upload_request_body = upload_post["requestBody"] if isinstance(upload_post, dict) else upload_post.requestBody
        upload_security = upload_post["security"] if isinstance(upload_post, dict) else upload_post.security
        delete_security = delete_post["security"] if isinstance(delete_post, dict) else delete_post.security
        upload_content = upload_request_body["content"] if isinstance(upload_request_body, dict) else upload_request_body.content

        assert upload_request_body is not None
        assert "multipart/form-data" in upload_content
        assert upload_security is not None and len(upload_security) == 2
        assert delete_security is not None and len(delete_security) == 2

        upload_operation = raw_schema["paths"]["/documents/upload"]["post"]
        delete_operation = raw_schema["paths"]["/documents/delete"]["post"]
        upload_response_schema = _response_json_schema(raw_schema, upload_operation)
        delete_response_schema = _response_json_schema(raw_schema, delete_operation)
        assert "data" in upload_response_schema["properties"]
        assert "data" in delete_response_schema["properties"]

    def test_openapi_schema_documents_upload_and_delete_contracts(self, tmp_path):
        client, _, _, _ = create_test_client(tmp_path, api_key="secret-key")
        with client:
            response = client.get("/openapi.json", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 200
        schema = response.json()
        paths = schema["paths"]

        upload_operation = paths["/documents/upload"]["post"]
        delete_operation = paths["/documents/delete"]["post"]
        retrieve_operation = paths["/retrieve"]["post"]
        health_operation = paths["/health"]["get"]

        security_schemes = schema["components"]["securitySchemes"]
        assert {"ApiKeyAuth", "BearerAuth"}.issubset(security_schemes.keys())

        assert upload_operation["summary"] == "Upload and index a document"
        assert "bytes://api-upload/<filename>" in upload_operation["description"]
        assert "sdk_source_*" in upload_operation["description"]

        assert delete_operation["summary"] == "Delete indexed resources"
        assert "request_identifier" in delete_operation["description"]
        assert "resolved_source_paths" in delete_operation["description"]
        assert "source_path is only present when exactly one canonical resource matched" in delete_operation["description"]

        upload_request_body = upload_operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        upload_request_schema = _resolve_schema_ref(schema, upload_request_body)
        extra_meta_property = upload_request_schema["properties"]["extra_meta_json"]
        assert "sdk_source_*" in extra_meta_property["description"]

        upload_response_schema = _response_json_schema(schema, upload_operation)
        upload_data_property = _resolve_schema_ref(schema, upload_response_schema["properties"]["data"])
        assert {"source_path", "source_file", "chunks"}.issubset(upload_data_property["properties"].keys())

        delete_response_schema = _response_json_schema(schema, delete_operation)
        delete_data_property = _resolve_schema_ref(schema, delete_response_schema["properties"]["data"])
        assert {"source_path", "request_identifier", "resolved_source_paths"}.issubset(delete_data_property["properties"].keys())

        expected_security = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
        assert upload_operation["security"] == expected_security
        assert delete_operation["security"] == expected_security
        assert retrieve_operation["security"] == expected_security
        assert "security" not in health_operation or not health_operation["security"]
