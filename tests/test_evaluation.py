"""检索评测模块测试。"""

from rag.evaluation import RetrievalEvalCase, evaluate_retriever


def test_expected_ids_can_match_source_path_and_source_file():
    cases = [
        RetrievalEvalCase(query="q", expected_ids=["/tmp/a.md", "a.md"]),
    ]

    def retriever(query, top_k=5, **kwargs):
        return [
            {"doc_id": "doc-1", "source_path": "/tmp/a.md", "source_file": "a.md"},
        ]

    summary = evaluate_retriever(retriever, cases, top_k=1)
    detail = summary.details[0]
    assert detail.hit is True
    assert detail.recall == 1.0
    assert detail.first_hit_rank == 1


def test_empty_cases_return_zero_metrics():
    summary = evaluate_retriever(lambda query, top_k=5, **kwargs: [], [], top_k=3)
    assert summary.total_queries == 0
    assert summary.hit_rate_at_k == 0.0
    assert summary.recall_at_k == 0.0
    assert summary.mrr_at_k == 0.0
    assert summary.average_first_hit_rank is None
