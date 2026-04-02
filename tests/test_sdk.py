"""SDK-oriented service tests."""

import asyncio
from pathlib import Path

import pytest

from rag import (
    DeleteOptions,
    DeleteResult,
    DeleteTarget,
    DocumentSource,
    HealthStatus,
    IndexOptions,
    IndexRequest,
    IndexResult,
    KnowledgeSDK,
    RetrieveOptions,
    RetrieveResult,
    SearchRequest,
    create_sdk,
)
from rag.config import ChunkConfig, EmbeddingConfig, QdrantConfig, RAGConfig, RerankerConfig
from rag.service import AgentKnowledgeService


class FakeEngine:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.indexed = []
        self.retrieved = []
        self.deleted = []
        self.started = False
        self.stopped = False

    def startup_sync(self):
        self.started = True

    def shutdown(self):
        self.stopped = True

    def collection_stats(self):
        return {"collection": f"rag_docs__{self.namespace}"}

    def index_file(
        self,
        file_path,
        extra_meta=None,
        force_reindex=False,
        display_source_name=None,
        display_source_path=None,
    ):
        path = Path(file_path)
        file_content = path.read_bytes() if path.exists() else b""
        record = {
            "file_path": str(path),
            "extra_meta": dict(extra_meta or {}),
            "force_reindex": force_reindex,
            "display_source_name": display_source_name,
            "display_source_path": display_source_path,
            "file_content": file_content,
        }
        self.indexed.append(record)
        return {
            "status": "ok",
            "source_path": display_source_path or str(path),
            "source_file": display_source_name or path.name,
            "chunks": 2,
            "namespace": self.namespace,
        }

    async def index_file_async(
        self,
        file_path,
        extra_meta=None,
        force_reindex=False,
        display_source_name=None,
        display_source_path=None,
    ):
        return self.index_file(
            file_path,
            extra_meta=extra_meta,
            force_reindex=force_reindex,
            display_source_name=display_source_name,
            display_source_path=display_source_path,
        )

    def retrieve(self, query, top_k=5, filter_conditions=None, skip_rerank=False, score_threshold=0.0):
        self.retrieved.append((query, top_k, filter_conditions, skip_rerank, score_threshold))
        return [
            {
                "doc_id": f"doc-{self.namespace}",
                "content": query,
                "source_file": "a.md",
                "source_path": str(Path("/tmp") / self.namespace / "a.md"),
                "heading_str": "标题",
                "heading_path": ["手册", "假期"],
                "score": 0.9,
                "rrf_score": 0.8,
                "dense_score": 0.7,
                "section_context": "section",
                "section_chunk_count": 1,
            }
        ]

    async def retrieve_async(self, query, top_k=5, filter_conditions=None, skip_rerank=False, score_threshold=0.0):
        return self.retrieve(
            query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            skip_rerank=skip_rerank,
            score_threshold=score_threshold,
        )

    def delete_file(self, file_identifier):
        self.deleted.append(file_identifier)

    async def delete_file_async(self, file_identifier):
        self.delete_file(file_identifier)

    def format_results_for_llm(self, results):
        return f"ctx:{self.namespace}:{len(results)}"

    def get_last_index_stats(self):
        return {"status": "ok", "namespace": self.namespace}

    def get_last_retrieval_stats(self):
        return {"status": "ok", "namespace": self.namespace}


class AsyncLocalClientProxy:
    def __init__(self, client):
        self._client = client

    async def query_points(self, **kwargs):
        return self._client.query_points(**kwargs)


class LocalTestEmbeddingService:
    def __init__(self, cfg):
        self.dimension = cfg.dimension

    def _embed_text(self, text: str):
        vector = [0.0] * self.dimension
        for index, byte in enumerate(text.encode("utf-8")):
            vector[index % self.dimension] += byte / 255.0
        return vector

    def embed_all(self, texts):
        return [self._embed_text(text) for text in texts]

    def embed_single(self, text):
        return self._embed_text(text)

    async def embed_single_async(self, text):
        return self._embed_text(text)


def make_local_test_config(tmp_path, collection_name: str) -> RAGConfig:
    return RAGConfig(
        embedding=EmbeddingConfig(provider="proxy", proxy_api_key="", dimension=16),
        reranker=RerankerConfig(api_key=""),
        qdrant=QdrantConfig(mode="local", path=str(tmp_path / "qdrant"), collection_name=collection_name),
        chunk=ChunkConfig(use_contextual_retrieval=False, rag_mode="basic"),
    )


class TestKnowledgeSDK:
    def test_low_level_service_methods_still_work(self):
        created = {}

        def factory(namespace: str):
            engine = FakeEngine(namespace)
            created[namespace] = engine
            return engine

        sdk = KnowledgeSDK(engine_factory=factory)
        sdk.startup_sync()

        index_result = sdk.index_document_sync("docs/a.md", namespace="hr", extra_meta={"tenant": "a"})
        retrieve_result = sdk.retrieve_sync("年假怎么计算", namespace="hr", top_k=3)
        delete_result = sdk.delete_document_sync("docs/a.md", namespace="hr")

        assert index_result["namespace"] == "hr"
        assert retrieve_result["namespace"] == "hr"
        assert retrieve_result["formatted_context"] == "ctx:hr:1"
        assert delete_result["namespace"] == "hr"
        assert "hr" in created
        assert sdk.health("hr")["namespace"] == "hr"

    def test_namespace_isolation(self):
        created = {}

        def factory(namespace: str):
            engine = FakeEngine(namespace)
            created[namespace] = engine
            return engine

        sdk = AgentKnowledgeService(engine_factory=factory)
        sdk.index_document_sync("docs/a.md", namespace="legal")
        sdk.index_document_sync("docs/b.md", namespace="finance")

        assert set(created.keys()) == {"legal", "finance"}
        assert created["legal"].indexed[0]["file_path"].endswith("docs\\a.md") or created["legal"].indexed[0]["file_path"].endswith("docs/a.md")
        assert created["finance"].indexed[0]["file_path"].endswith("docs\\b.md") or created["finance"].indexed[0]["file_path"].endswith("docs/b.md")

    def test_create_sdk_export(self):
        sdk = create_sdk(engine_factory=lambda namespace: FakeEngine(namespace))

        assert isinstance(sdk, KnowledgeSDK)

    def test_index_request_with_path_source(self, tmp_path):
        created = {}
        doc_path = tmp_path / "policy.md"
        doc_path.write_text("# 假期制度", encoding="utf-8")

        def factory(namespace: str):
            engine = FakeEngine(namespace)
            created[namespace] = engine
            return engine

        sdk = KnowledgeSDK(engine_factory=factory)
        result = sdk.index(
            IndexRequest(
                source=DocumentSource.from_path(str(doc_path), upload_origin="local-sync"),
                namespace="hr",
                metadata={"tenant": "internal"},
                reindex_strategy="force",
            )
        )

        call = created["hr"].indexed[0]
        assert isinstance(result, IndexResult)
        assert result.namespace == "hr"
        assert result.source_path == str(doc_path.resolve())
        assert call["extra_meta"]["tenant"] == "internal"
        assert call["extra_meta"]["sdk_source_kind"] == "path"
        assert call["extra_meta"]["sdk_source_origin"] == "local-sync"
        assert call["display_source_name"] == "policy.md"
        assert call["display_source_path"] == str(doc_path.resolve())
        assert call["force_reindex"] is True

    def test_index_request_with_text_source(self):
        created = {}

        def factory(namespace: str):
            engine = FakeEngine(namespace)
            created[namespace] = engine
            return engine

        sdk = KnowledgeSDK(engine_factory=factory)
        result = sdk.index(
            IndexRequest(
                source=DocumentSource.from_text(
                    "# 手册\n\n这是文本输入",
                    source_name="handbook.md",
                    upload_origin="manual-upload",
                ),
                namespace="ops",
                metadata={"tenant": "alpha"},
            )
        )

        call = created["ops"].indexed[0]
        assert result.source_path == "text://manual-upload/handbook.md"
        assert call["display_source_path"] == "text://manual-upload/handbook.md"
        assert call["display_source_name"] == "handbook.md"
        assert call["file_content"].decode("utf-8") == "# 手册\n\n这是文本输入"
        assert call["extra_meta"]["sdk_source_kind"] == "text"
        assert call["extra_meta"]["sdk_source_name"] == "handbook.md"
        assert not Path(call["file_path"]).exists()

    def test_index_request_with_bytes_source(self):
        created = {}

        def factory(namespace: str):
            engine = FakeEngine(namespace)
            created[namespace] = engine
            return engine

        sdk = KnowledgeSDK(engine_factory=factory)
        result = sdk.index(
            IndexRequest(
                source=DocumentSource.from_bytes(
                    b"{\"name\": \"rag\"}",
                    source_name="config.json",
                    upload_origin="sdk-binary",
                ),
                namespace="platform",
            )
        )

        call = created["platform"].indexed[0]
        assert result.source_path == "bytes://sdk-binary/config.json"
        assert call["file_content"] == b"{\"name\": \"rag\"}"
        assert call["extra_meta"]["sdk_source_kind"] == "bytes"

    def test_invalid_metadata_is_rejected(self):
        with pytest.raises(ValueError, match="metadata 不能覆盖保留字段"):
            IndexRequest(
                source=DocumentSource.from_text("hello", source_name="note.md"),
                metadata={"sdk_source_kind": "manual"},
            )

    def test_invalid_path_is_rejected(self, tmp_path):
        missing_path = tmp_path / "missing.md"
        with pytest.raises(FileNotFoundError):
            DocumentSource.from_path(str(missing_path))

    def test_legacy_index_signature_still_works(self, tmp_path):
        created = {}
        doc_path = tmp_path / "legacy.md"
        doc_path.write_text("legacy", encoding="utf-8")

        def factory(namespace: str):
            engine = FakeEngine(namespace)
            created[namespace] = engine
            return engine

        sdk = KnowledgeSDK(engine_factory=factory)
        result = sdk.index(
            str(doc_path),
            IndexOptions(namespace="legacy", extra_meta={"tenant": "beta"}, force_reindex=True),
        )

        call = created["legacy"].indexed[0]
        assert result.namespace == "legacy"
        assert call["extra_meta"]["tenant"] == "beta"
        assert call["extra_meta"]["sdk_source_kind"] == "path"
        assert call["force_reindex"] is True

    def test_search_request_and_legacy_wrapper(self):
        created = {}

        def factory(namespace: str):
            engine = FakeEngine(namespace)
            created[namespace] = engine
            return engine

        sdk = KnowledgeSDK(engine_factory=factory)

        typed_result = sdk.search(
            SearchRequest(
                query="年假怎么计算？",
                namespace="hr",
                top_k=3,
                filters={"source_file": "a.md"},
                score_threshold=0.2,
                skip_rerank=True,
            )
        )
        legacy_result = sdk.search(
            "如何回滚？",
            RetrieveOptions(namespace="ops", top_k=2),
        )

        assert isinstance(typed_result, RetrieveResult)
        assert typed_result.namespace == "hr"
        assert typed_result.count == 1
        assert created["hr"].retrieved[0] == (
            "年假怎么计算？",
            3,
            {"source_file": "a.md"},
            True,
            0.2,
        )
        assert isinstance(legacy_result, RetrieveResult)
        assert legacy_result.namespace == "ops"
        assert created["ops"].retrieved[0] == ("如何回滚？", 2, None, False, 0.0)

    def test_delete_target_uses_logical_source_path(self):
        created = {}

        def factory(namespace: str):
            engine = FakeEngine(namespace)
            created[namespace] = engine
            return engine

        sdk = KnowledgeSDK(engine_factory=factory)
        result = sdk.delete(
            DeleteTarget.from_source_path("text://manual-upload/handbook.md"),
            DeleteOptions(namespace="ops"),
        )

        assert isinstance(result, DeleteResult)
        assert result.namespace == "ops"
        assert result.source_path == "text://manual-upload/handbook.md"
        assert result.file_identifier == "text://manual-upload/handbook.md"
        assert created["ops"].deleted == ["text://manual-upload/handbook.md"]

    def test_typed_async_sdk_methods(self):
        sdk = KnowledgeSDK(engine_factory=lambda namespace: FakeEngine(namespace))

        async def run():
            await sdk.startup()
            index_result = await sdk.aindex(
                IndexRequest(
                    source=DocumentSource.from_text("async", source_name="async.md"),
                    namespace="ops",
                )
            )
            retrieve_result = await sdk.asearch(SearchRequest(query="如何回滚？", namespace="ops", top_k=2))
            delete_result = await sdk.adelete(
                DeleteTarget.from_source_path("text://manual-upload/async.md"),
                DeleteOptions(namespace="ops"),
            )
            await sdk.shutdown()
            return index_result, retrieve_result, delete_result

        index_result, retrieve_result, delete_result = asyncio.run(run())

        assert isinstance(index_result, IndexResult)
        assert index_result.namespace == "ops"
        assert isinstance(retrieve_result, RetrieveResult)
        assert retrieve_result.namespace == "ops"
        assert retrieve_result.items[0].doc_id == "doc-ops"
        assert isinstance(delete_result, DeleteResult)
        assert delete_result.namespace == "ops"
        assert delete_result.source_path == "text://manual-upload/async.md"

    def test_health_return_type(self):
        sdk = KnowledgeSDK(engine_factory=lambda namespace: FakeEngine(namespace))
        health = sdk.get_health("hr")

        assert isinstance(health, HealthStatus)
        assert health.namespace == "hr"


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant.*:UserWarning")
class TestKnowledgeSDKRoundTrip:
    def test_sync_text_source_round_trip_local(self, tmp_path, sample_markdown):
        import rag.engine as engine_module

        cfg = make_local_test_config(tmp_path, "sdk_roundtrip_sync")

        import rag.vector_store as vector_store_module

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(engine_module, "EmbeddingService", LocalTestEmbeddingService)
            monkeypatch.setattr(
                vector_store_module.QdrantVectorStore,
                "_build_async_client",
                lambda self, cfg: AsyncLocalClientProxy(self._client),
            )
            sdk = create_sdk(base_config=cfg)
            sdk.startup_sync()
            try:
                index_result = sdk.index(
                    IndexRequest(
                        source=DocumentSource.from_text(
                            sample_markdown,
                            source_name="guide.md",
                            upload_origin="integration-sync",
                        ),
                        namespace="docs",
                    )
                )
                before_delete = sdk.search(SearchRequest(query="pytest", namespace="docs", top_k=3))
                delete_result = sdk.delete(
                    DeleteTarget.from_source_path(index_result.source_path),
                    DeleteOptions(namespace="docs"),
                )
                after_delete = sdk.search(SearchRequest(query="pytest", namespace="docs", top_k=3))
            finally:
                sdk.shutdown_sync()

        assert before_delete.count > 0
        assert before_delete.items[0].source_path == index_result.source_path
        assert delete_result.source_path == index_result.source_path
        assert after_delete.count == 0

    def test_async_text_source_round_trip_local(self, tmp_path, sample_markdown):
        import rag.engine as engine_module

        cfg = make_local_test_config(tmp_path, "sdk_roundtrip_async")

        async def run_round_trip():
            sdk = create_sdk(base_config=cfg)
            await sdk.startup()
            try:
                index_result = await sdk.aindex(
                    IndexRequest(
                        source=DocumentSource.from_text(
                            sample_markdown,
                            source_name="async-guide.md",
                            upload_origin="integration-async",
                        ),
                        namespace="async-docs",
                    )
                )
                before_delete = await sdk.asearch(
                    SearchRequest(query="Docker Compose", namespace="async-docs", top_k=3)
                )
                delete_result = await sdk.adelete(
                    DeleteTarget.from_source_path(index_result.source_path),
                    DeleteOptions(namespace="async-docs"),
                )
                after_delete = await sdk.asearch(
                    SearchRequest(query="Docker Compose", namespace="async-docs", top_k=3)
                )
                return index_result, before_delete, delete_result, after_delete
            finally:
                await sdk.shutdown()

        import rag.vector_store as vector_store_module

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(engine_module, "EmbeddingService", LocalTestEmbeddingService)
            monkeypatch.setattr(
                vector_store_module.QdrantVectorStore,
                "_build_async_client",
                lambda self, cfg: AsyncLocalClientProxy(self._client),
            )
            index_result, before_delete, delete_result, after_delete = asyncio.run(run_round_trip())

        assert before_delete.count > 0
        assert delete_result.source_path == index_result.source_path
        assert after_delete.count == 0
