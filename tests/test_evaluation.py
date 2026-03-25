"""检索评测模块测试。"""

import json
from pathlib import Path

from rag.evaluation import RetrievalEvalCase, evaluate_retriever, load_eval_cases


def test_expected_ids_can_match_relative_source_path():
    cases = [
        RetrievalEvalCase(query="q", expected_ids=["docs/fastapi_docs/async.md"]),
    ]

    def retriever(query, top_k=5, **kwargs):
        return [
            {
                "doc_id": "doc-1",
                "source_path": "D:/learnsomething/rag/docs/fastapi_docs/async.md",
                "source_file": "async.md",
            },
        ]

    summary = evaluate_retriever(retriever, cases, top_k=1)
    detail = summary.details[0]
    assert detail.hit is True
    assert detail.recall == 1.0
    assert detail.first_hit_rank == 1


def test_expected_heading_is_checked_when_provided():
    cases = [
        RetrievalEvalCase(
            query="q",
            expected_ids=["docs/fastapi_docs/async.md"],
            expected_heading="赶时间吗？",
        ),
    ]

    def retriever(query, top_k=5, **kwargs):
        return [
            {
                "doc_id": "doc-1",
                "source_path": "D:/learnsomething/rag/docs/fastapi_docs/async.md",
                "source_file": "async.md",
                "heading_str": "赶时间吗？",
            },
        ]

    summary = evaluate_retriever(retriever, cases, top_k=1)
    detail = summary.details[0]
    assert detail.hit is True
    assert detail.heading_hit is True
    assert detail.heading_first_hit_rank == 1
    assert summary.heading_case_count == 1
    assert summary.heading_hit_rate_at_k == 1.0


def test_top_level_bucket_is_normalized_into_metadata(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "eval_001",
                "query": "q",
                "expected_ids": ["docs/fastapi_docs/async.md"],
                "expected_heading": "赶时间吗？",
                "difficulty": "easy",
                "bucket": "technical_explanation",
                "multi_hop": False,
                "ground_truth": "答案",
                "metadata": {"notes": "note"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    case = load_eval_cases(dataset)[0]
    assert case.expected_heading == "赶时间吗？"
    assert case.metadata["bucket"] == "technical_explanation"
    assert case.metadata["id"] == "eval_001"
    assert case.metadata["difficulty"] == "easy"
    assert case.metadata["ground_truth"] == "答案"


def test_empty_cases_return_zero_metrics():
    summary = evaluate_retriever(lambda query, top_k=5, **kwargs: [], [], top_k=3)
    assert summary.total_queries == 0
    assert summary.hit_rate_at_k == 0.0
    assert summary.recall_at_k == 0.0
    assert summary.mrr_at_k == 0.0
    assert summary.average_first_hit_rank is None
    assert summary.heading_hit_rate_at_k == 0.0
