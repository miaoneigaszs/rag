"""Shared helpers for Docker-backed integration tests."""

from qdrant_client import AsyncQdrantClient, QdrantClient


class DeterministicEmbeddingService:
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


def patch_docker_qdrant_clients(monkeypatch, vector_store_module):
    monkeypatch.setattr(
        vector_store_module.QdrantVectorStore,
        "_build_client",
        staticmethod(lambda cfg: QdrantClient(host=cfg.host, port=cfg.port, trust_env=False, check_compatibility=False)),
    )
    monkeypatch.setattr(
        vector_store_module.QdrantVectorStore,
        "_build_async_client",
        staticmethod(lambda cfg: AsyncQdrantClient(host=cfg.host, port=cfg.port, trust_env=False, check_compatibility=False)),
    )
