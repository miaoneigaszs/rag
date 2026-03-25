"""RAG 检索评测工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class RetrievalEvalCase:
    """单条检索评测样本。"""

    query: str
    expected_ids: Sequence[str]
    filter_conditions: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvalResult:
    """单条 query 的评测结果。"""

    query: str
    expected_ids: List[str]
    matched_ids: List[str]
    observed_ids: List[List[str]]
    first_hit_rank: Optional[int]
    hit: bool
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
            "top_k": self.top_k,
            "details": [
                {
                    "query": detail.query,
                    "expected_ids": detail.expected_ids,
                    "matched_ids": detail.matched_ids,
                    "observed_ids": detail.observed_ids,
                    "first_hit_rank": detail.first_hit_rank,
                    "hit": detail.hit,
                    "recall": detail.recall,
                    "retrieved_count": detail.retrieved_count,
                    "retrieval_stats": detail.retrieval_stats,
                    "metadata": detail.metadata,
                }
                for detail in self.details
            ],
        }


def _normalize_expected_ids(expected_ids: Sequence[str]) -> List[str]:
    return [identifier for identifier in dict.fromkeys(expected_ids) if identifier]


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
    recall_sum = 0.0
    mrr_sum = 0.0
    first_hit_ranks: List[int] = []

    for case in cases:
        expected_ids = _normalize_expected_ids(case.expected_ids)
        kwargs = dict(retrieve_kwargs or {})
        if case.filter_conditions is not None:
            kwargs["filter_conditions"] = case.filter_conditions
        results = retriever(case.query, top_k=top_k, **kwargs)

        observed_ids = [_result_identifiers(result) for result in results]
        matched_ids: List[str] = []
        first_hit_rank: Optional[int] = None
        for rank, identifiers in enumerate(observed_ids, start=1):
            matched_now = [identifier for identifier in identifiers if identifier in expected_ids]
            for identifier in matched_now:
                if identifier not in matched_ids:
                    matched_ids.append(identifier)
            if matched_now and first_hit_rank is None:
                first_hit_rank = rank

        hit = first_hit_rank is not None
        recall = len(matched_ids) / len(expected_ids) if expected_ids else 0.0
        if hit:
            hit_count += 1
            mrr_sum += 1.0 / first_hit_rank  # type: ignore[arg-type]
            first_hit_ranks.append(first_hit_rank)  # type: ignore[arg-type]
        recall_sum += recall

        details.append(
            RetrievalEvalResult(
                query=case.query,
                expected_ids=expected_ids,
                matched_ids=matched_ids,
                observed_ids=observed_ids,
                first_hit_rank=first_hit_rank,
                hit=hit,
                recall=recall,
                retrieved_count=len(results),
                retrieval_stats=_get_stats_from_provider(stats_provider),
                metadata=dict(case.metadata),
            )
        )

    total_queries = len(details)
    avg_rank = (
        sum(first_hit_ranks) / len(first_hit_ranks) if first_hit_ranks else None
    )
    return RetrievalEvalSummary(
        total_queries=total_queries,
        hit_count=hit_count,
        hit_rate_at_k=hit_count / total_queries if total_queries else 0.0,
        recall_at_k=recall_sum / total_queries if total_queries else 0.0,
        mrr_at_k=mrr_sum / total_queries if total_queries else 0.0,
        average_first_hit_rank=avg_rank,
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
