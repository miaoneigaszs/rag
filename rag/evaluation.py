"""RAG 检索评测工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class RetrievalEvalCase:
    """单条检索评测样本。"""

    query: str
    expected_ids: Sequence[str]
    expected_heading: str = ""
    filter_conditions: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvalResult:
    """单条 query 的评测结果。"""

    query: str
    expected_ids: List[str]
    expected_heading: str
    matched_ids: List[str]
    observed_ids: List[List[str]]
    observed_headings: List[str]
    first_hit_rank: Optional[int]
    heading_first_hit_rank: Optional[int]
    hit: bool
    heading_hit: bool
    recall: float
    retrieved_count: int
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvalSummary:
    """整组 query 的聚合评测结果。"""

    total_queries: int
    hit_count: int
    hit_rate_at_k: float
    recall_at_k: float
    mrr_at_k: float
    average_first_hit_rank: Optional[float]
    heading_case_count: int
    heading_hit_count: int
    heading_hit_rate_at_k: float
    top_k: int
    details: List[RetrievalEvalResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "hit_count": self.hit_count,
            "hit_rate_at_k": self.hit_rate_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr_at_k": self.mrr_at_k,
            "average_first_hit_rank": self.average_first_hit_rank,
            "heading_case_count": self.heading_case_count,
            "heading_hit_count": self.heading_hit_count,
            "heading_hit_rate_at_k": self.heading_hit_rate_at_k,
            "top_k": self.top_k,
            "details": [
                {
                    "query": detail.query,
                    "expected_ids": detail.expected_ids,
                    "expected_heading": detail.expected_heading,
                    "matched_ids": detail.matched_ids,
                    "observed_ids": detail.observed_ids,
                    "observed_headings": detail.observed_headings,
                    "first_hit_rank": detail.first_hit_rank,
                    "heading_first_hit_rank": detail.heading_first_hit_rank,
                    "hit": detail.hit,
                    "heading_hit": detail.heading_hit,
                    "recall": detail.recall,
                    "retrieved_count": detail.retrieved_count,
                    "retrieval_stats": detail.retrieval_stats,
                    "metadata": detail.metadata,
                }
                for detail in self.details
            ],
        }


def _normalize_identifier(identifier: str) -> str:
    return identifier.strip().replace("\\", "/").lstrip("./")


def _normalize_expected_ids(expected_ids: Sequence[str]) -> List[str]:
    return [identifier for identifier in dict.fromkeys(expected_ids) if identifier]


def _identifier_matches(expected: str, candidate: str) -> bool:
    expected_norm = _normalize_identifier(expected).lower()
    candidate_norm = _normalize_identifier(candidate).lower()
    if not expected_norm or not candidate_norm:
        return False
    if expected_norm == candidate_norm:
        return True
    if "/" in expected_norm:
        return candidate_norm.endswith(f"/{expected_norm}")
    return candidate_norm.split("/")[-1] == expected_norm


def _heading_matches(expected_heading: str, observed_heading: str) -> bool:
    return expected_heading.strip() == observed_heading.strip()


def _result_identifiers(result: Dict[str, Any]) -> List[str]:
    identifiers: List[str] = []
    for key in ("doc_id", "source_path", "source_file"):
        value = result.get(key)
        if isinstance(value, str) and value:
            identifiers.append(value)
    return identifiers


def _get_stats_from_provider(stats_provider: Optional[Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    if stats_provider is None:
        return {}
    try:
        stats = stats_provider()
    except Exception:
        return {}
    return dict(stats or {})


def load_eval_cases(path: str | Path) -> List[RetrievalEvalCase]:
    """从 JSONL 文件加载评测样本，并兼容旧版字段布局。"""
    dataset_path = Path(path)
    cases: List[RetrievalEvalCase] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        metadata = dict(item.get("metadata") or {})
        for key in ("id", "difficulty", "ground_truth", "multi_hop"):
            if key in item and key not in metadata:
                metadata[key] = item[key]
        if item.get("bucket") and "bucket" not in metadata:
            metadata["bucket"] = item["bucket"]

        expected_ids = item.get("expected_ids", [])
        if isinstance(expected_ids, str):
            expected_ids = [expected_ids]

        cases.append(
            RetrievalEvalCase(
                query=item["query"],
                expected_ids=expected_ids,
                expected_heading=item.get("expected_heading", ""),
                filter_conditions=item.get("filter_conditions"),
                metadata=metadata,
            )
        )
    return cases


def evaluate_retriever(
    retriever: Callable[..., List[Dict[str, Any]]],
    cases: Iterable[RetrievalEvalCase],
    *,
    top_k: int = 5,
    retrieve_kwargs: Optional[Dict[str, Any]] = None,
    stats_provider: Optional[Callable[[], Dict[str, Any]]] = None,
) -> RetrievalEvalSummary:
    """对任意 retriever 进行 top-k 检索评测。"""
    details: List[RetrievalEvalResult] = []
    hit_count = 0
    heading_case_count = 0
    heading_hit_count = 0
    recall_sum = 0.0
    mrr_sum = 0.0
    first_hit_ranks: List[int] = []

    for case in cases:
        expected_ids = _normalize_expected_ids(case.expected_ids)
        expected_heading = case.expected_heading or ""
        kwargs = dict(retrieve_kwargs or {})
        if case.filter_conditions is not None:
            kwargs["filter_conditions"] = case.filter_conditions
        results = retriever(case.query, top_k=top_k, **kwargs)

        observed_ids = [_result_identifiers(result) for result in results]
        observed_headings = [str(result.get("heading_str", "")) for result in results]
        matched_ids: List[str] = []
        first_hit_rank: Optional[int] = None
        heading_first_hit_rank: Optional[int] = None

        for rank, (identifiers, heading) in enumerate(zip(observed_ids, observed_headings), start=1):
            matched_now = [
                expected_id
                for expected_id in expected_ids
                if any(_identifier_matches(expected_id, identifier) for identifier in identifiers)
            ]
            for identifier in matched_now:
                if identifier not in matched_ids:
                    matched_ids.append(identifier)
            if matched_now and first_hit_rank is None:
                first_hit_rank = rank
            if matched_now and expected_heading and _heading_matches(expected_heading, heading):
                if heading_first_hit_rank is None:
                    heading_first_hit_rank = rank

        hit = first_hit_rank is not None
        heading_hit = heading_first_hit_rank is not None if expected_heading else False
        recall = len(matched_ids) / len(expected_ids) if expected_ids else 0.0

        if hit:
            hit_count += 1
            mrr_sum += 1.0 / first_hit_rank  # type: ignore[arg-type]
            first_hit_ranks.append(first_hit_rank)  # type: ignore[arg-type]
        if expected_heading:
            heading_case_count += 1
            if heading_hit:
                heading_hit_count += 1
        recall_sum += recall

        details.append(
            RetrievalEvalResult(
                query=case.query,
                expected_ids=expected_ids,
                expected_heading=expected_heading,
                matched_ids=matched_ids,
                observed_ids=observed_ids,
                observed_headings=observed_headings,
                first_hit_rank=first_hit_rank,
                heading_first_hit_rank=heading_first_hit_rank,
                hit=hit,
                heading_hit=heading_hit,
                recall=recall,
                retrieved_count=len(results),
                retrieval_stats=_get_stats_from_provider(stats_provider),
                metadata=dict(case.metadata),
            )
        )

    total_queries = len(details)
    avg_rank = sum(first_hit_ranks) / len(first_hit_ranks) if first_hit_ranks else None
    return RetrievalEvalSummary(
        total_queries=total_queries,
        hit_count=hit_count,
        hit_rate_at_k=hit_count / total_queries if total_queries else 0.0,
        recall_at_k=recall_sum / total_queries if total_queries else 0.0,
        mrr_at_k=mrr_sum / total_queries if total_queries else 0.0,
        average_first_hit_rank=avg_rank,
        heading_case_count=heading_case_count,
        heading_hit_count=heading_hit_count,
        heading_hit_rate_at_k=(heading_hit_count / heading_case_count if heading_case_count else 0.0),
        top_k=top_k,
        details=details,
    )


def evaluate_engine(
    engine: Any,
    cases: Iterable[RetrievalEvalCase],
    *,
    top_k: int = 5,
    retrieve_kwargs: Optional[Dict[str, Any]] = None,
) -> RetrievalEvalSummary:
    """使用 RAGEngine 的 `retrieve()` 与其观测数据执行评测。"""
    stats_provider = None
    if hasattr(engine, "get_last_retrieval_stats"):
        stats_provider = engine.get_last_retrieval_stats
    return evaluate_retriever(
        engine.retrieve,
        cases,
        top_k=top_k,
        retrieve_kwargs=retrieve_kwargs,
        stats_provider=stats_provider,
    )
