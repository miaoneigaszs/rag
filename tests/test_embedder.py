from rag.config import EmbeddingConfig
from rag.embedder import EmbeddingService


def test_prepare_text_truncates_to_max_input_chars():
    service = EmbeddingService.__new__(EmbeddingService)
    service.cfg = EmbeddingConfig(max_input_chars=10)

    text = "0123456789abcdef"

    assert service._prepare_text(text) == "0123456789"


def test_embed_batch_truncates_long_inputs_before_request():
    service = EmbeddingService.__new__(EmbeddingService)
    service.cfg = EmbeddingConfig(max_input_chars=5)

    captured = {}

    class _Embeddings:
        def create(self, *, model, input):
            captured["model"] = model
            captured["input"] = input

            class _Item:
                embedding = [0.1, 0.2]

            class _Response:
                data = [_Item(), _Item()]

            return _Response()

    class _Client:
        embeddings = _Embeddings()

    service._client = _Client()
    service._model = "test-model"

    vectors = service.embed_batch(["abcdef", "xyz"])

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert captured["model"] == "test-model"
    assert captured["input"] == ["abcde", "xyz"]
