"""RAG 引擎及其集成的服务门面层。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
from copy import deepcopy
from os import PathLike
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Optional

from . import RetrievalEvalCase, evaluate_engine, load_eval_cases
from .config import RAGConfig
from .engine import RAGEngine
from .models import (
    DeleteOptions,
    DeleteResult,
    DeleteTarget,
    DocumentSource,
    HealthStatus,
    IndexOptions,
    IndexRequest,
    IndexResult,
    RetrieveOptions,
    RetrieveResult,
    SearchRequest,
)

_DEFAULT_NAMESPACE = "default"
_NAMESPACE_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]+")

async def maybe_await(value: Any) -> Any:
    """统一处理同步值和异步协程，如果值是可等待对象则等待执行，否则直接返回。"""
    if inspect.isawaitable(value):
        return await value
    return value

class AgentKnowledgeService:
    """RAG 核心引擎之上的稳定服务边界。

    该类以 SDK 为优先设计：同步方法用于直接库调用，
    异步方法支持 API 适配器和异步应用程序。
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        engine_factory: Optional[Callable[[str], Any]] = None,
        default_namespace: str = _DEFAULT_NAMESPACE,
        base_config: Optional[RAGConfig] = None,
    ) -> None:
        """初始化服务实例。

        Args:
            engine: 可选的预创建引擎实例，将被关联到默认命名空间
            engine_factory: 自定义引擎工厂函数，用于按需创建引擎
            default_namespace: 默认命名空间名称
            base_config: 基础配置，所有引擎将继承此配置
        """
        self.default_namespace = default_namespace or _DEFAULT_NAMESPACE
        self._base_config = deepcopy(base_config) if base_config is not None else RAGConfig.from_env()
        self._engine_factory = engine_factory or self._default_engine_factory
        self._engines: Dict[str, Any] = {}
        self._started = False
        self._lock = RLock()

        if engine is not None:
            normalized = self.normalize_namespace(self.default_namespace)
            self._engines[normalized] = engine

    @staticmethod
    def normalize_namespace(namespace: Optional[str]) -> str:
        """Qdrant collection只支持字母、数字、下划线、连字符，并且长度有限制，这里对命名空间进行规范化处理。"""
        raw = (namespace or _DEFAULT_NAMESPACE).strip()
        if not raw:
            raw = _DEFAULT_NAMESPACE
        safe = _NAMESPACE_SANITIZE_RE.sub("-", raw).strip("-_") or _DEFAULT_NAMESPACE
        if len(safe) <= 40:
            return safe.lower()
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
        return f"{safe[:31].lower()}-{digest}"

    def _resolve_namespace(self, namespace: Optional[str]) -> str:
        """这里做了一个薄封装层，规范化命名空间。"""
        return self.normalize_namespace(namespace or self.default_namespace)

    def _build_collection_name(self, namespace: str) -> str:
        """构建安全的 collection 名称，确保总长度不超过 Qdrant 的 63 字符限制。"""
        base_name = self._base_config.qdrant.collection_name
        safe_namespace = self.normalize_namespace(namespace)

        # 验证 base_name 长度（预留空间给 "__namespace"）
        if len(base_name) > 50:
            raise ValueError(
                f"collection_name '{base_name}' 过长（最多50字符），"
                f"当前 {len(base_name)} 字符"
            )

        # 默认命名空间：直接返回 base_name
        if safe_namespace == self.normalize_namespace(self.default_namespace):
            return base_name

        # 组合名称
        candidate = f"{base_name}__{safe_namespace}"

        # 最终检查：如果超过 63 字符，用哈希压缩 namespace
        if len(candidate) > 63:
            namespace_hash = hashlib.md5(safe_namespace.encode()).hexdigest()[:8]
            candidate = f"{base_name}__{namespace_hash}"

        return candidate

    def _default_engine_factory(self, namespace: str) -> Any:
        """默认引擎工厂函数，返回 RAG 引擎实例。"""
        cfg = deepcopy(self._base_config)
        cfg.qdrant.collection_name = self._build_collection_name(namespace)
        return RAGEngine(cfg)

    def _get_engine(self, namespace: Optional[str] = None) -> Any:
        """获取指定命名空间的引擎实例，支持懒加载和线程安全。

        如果引擎不存在则自动创建，如果服务已启动则自动启动新创建的引擎。
        """
        normalized = self._resolve_namespace(namespace)
        with self._lock:
            engine = self._engines.get(normalized)
            if engine is None:
                engine = self._engine_factory(normalized)
                self._engines[normalized] = engine
                should_start = self._started
            else:
                should_start = False

        if should_start:
            startup = getattr(engine, "startup_sync", None)
            if startup is not None:
                startup()
            else:
                startup_async = getattr(engine, "startup", None)
                if startup_async is not None:
                    raise RuntimeError("Async-only engine created after service startup is not supported")
        return engine

    def get_known_namespaces(self) -> list[str]:
        """获取所有已创建引擎的命名空间列表。"""
        return sorted(self._engines.keys())

    async def startup(self) -> None:
        """异步启动服务，启动所有已创建的引擎实例。"""
        self._started = True
        for engine in list(self._engines.values()):
            startup = getattr(engine, "startup", None)
            if startup is not None:
                await maybe_await(startup())
                continue

            startup_sync = getattr(engine, "startup_sync", None)
            if startup_sync is not None:
                startup_sync()

    def startup_sync(self) -> None:
        """同步启动服务，启动所有已创建的引擎实例。"""
        self._started = True
        for engine in list(self._engines.values()):
            startup_sync = getattr(engine, "startup_sync", None)
            if startup_sync is not None:
                startup_sync()
                continue

            startup = getattr(engine, "startup", None)
            if startup is not None:
                asyncio.run(maybe_await(startup()))

    async def shutdown(self) -> None:
        """异步关闭服务，关闭所有引擎实例并释放资源。"""
        for engine in list(self._engines.values()):
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                await maybe_await(shutdown())
        self._started = False

    def shutdown_sync(self) -> None:
        """同步关闭服务，关闭所有引擎实例并释放资源。"""
        for engine in list(self._engines.values()):
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                asyncio.run(maybe_await(shutdown()))
        self._started = False

    async def _call_engine(self, namespace: Optional[str], async_name: str, sync_name: str, *args: Any, **kwargs: Any) -> Any:
        engine = self._get_engine(namespace)
        method = getattr(engine, async_name, None)
        if method is not None:
            return await maybe_await(method(*args, **kwargs))
        return getattr(engine, sync_name)(*args, **kwargs)

    def _call_engine_sync(self, namespace: Optional[str], sync_name: str, *args: Any, **kwargs: Any) -> Any:
        engine = self._get_engine(namespace)
        return getattr(engine, sync_name)(*args, **kwargs)

    def health(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """获取向量数据库健康状态信息。"""
        engine = self._get_engine(namespace)
        info_fn = getattr(engine, "collection_stats", None)
        collection = info_fn() if info_fn is not None else None
        return {
            "status": "ok",
            "namespace": self._resolve_namespace(namespace),
            "collection": collection,
            "known_namespaces": self.get_known_namespaces(),
        }

    async def index_document(
        self,
        file_path: str,
        *,
        namespace: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        display_source_name: Optional[str] = None,
        display_source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """异步索引文档，返回索引结果。"""
        engine_kwargs: Dict[str, Any] = {
            "extra_meta": extra_meta or {},
            "force_reindex": force_reindex,
        }
        if display_source_name is not None:
            engine_kwargs["display_source_name"] = display_source_name
        if display_source_path is not None:
            engine_kwargs["display_source_path"] = display_source_path

        result = await self._call_engine(namespace, "index_file_async", "index_file", file_path, **engine_kwargs)
        result_dict = dict(result)
        result_dict.setdefault("namespace", self._resolve_namespace(namespace))
        return result_dict

    def index_document_sync(
        self,
        file_path: str,
        *,
        namespace: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
        display_source_name: Optional[str] = None,
        display_source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """同步索引文档，返回索引结果格式化字典。"""
        engine_kwargs: Dict[str, Any] = {
            "extra_meta": extra_meta or {},
            "force_reindex": force_reindex,
        }
        if display_source_name is not None:
            engine_kwargs["display_source_name"] = display_source_name
        if display_source_path is not None:
            engine_kwargs["display_source_path"] = display_source_path

        result = self._call_engine_sync(namespace, "index_file", file_path, **engine_kwargs)
        result_dict = dict(result)
        result_dict.setdefault("namespace", self._resolve_namespace(namespace))
        return result_dict

    def _resolve_deleted_source_paths(self, engine: Any, file_identifier: str) -> list[str]:
        """根据文件标识符解析所有可能的source_path。"""
        candidate = file_identifier.strip()
        if not candidate:
            return []

        if RAGEngine._is_logical_source_path(candidate):
            return [candidate]

        path_obj = Path(candidate)
        looks_like_path = path_obj.is_absolute() or path_obj.parent != Path(".")
        if looks_like_path:
            return [str(path_obj.resolve(strict=False))]

        list_paths = getattr(engine, "list_source_paths_by_source_file", None)
        if list_paths is None:
            return []
        return list(list_paths(candidate) or [])

    def _build_delete_result(
        self,
        *,
        file_identifier: str, # 是用户传入的要删除的文件，可以是逻辑路径、物理路径或文件名
        resolved_source_paths: list[str], # 解析后的所有可能的source_path
        namespace: Optional[str], # 命名空间
    ) -> Dict[str, Any]:
        """构建删除结果字典。如果解析后的路径只有一个，就添加到结果中，如果有多个或者0个解析路径，就不添加source_path字段。但是resolved_source_paths字段会区分这种情况。"""
        primary_source_path = resolved_source_paths[0] if len(resolved_source_paths) == 1 else ""
        result = {
            "deleted": True,
            "file_identifier": file_identifier,
            "request_identifier": file_identifier,
            "resolved_source_paths": resolved_source_paths,
            "namespace": self._resolve_namespace(namespace),
        }
        if primary_source_path:
            result["source_path"] = primary_source_path
        return result

    async def delete_document(self, file_identifier: str, *, namespace: Optional[str] = None) -> Dict[str, Any]:
        """异步删除指定文档。

        Args:
            file_identifier: 文件标识符，可以是逻辑路径、物理路径或文件名，如果上传的是文件名，会根据文件名查询所有路径的source_path并删除。
            namespace: 命名空间

        Returns:
            删除结果字典
        """
        engine = self._get_engine(namespace)
        resolved_source_paths = self._resolve_deleted_source_paths(engine, file_identifier)
        await self._call_engine(namespace, "delete_file_async", "delete_file", file_identifier)
        return self._build_delete_result(
            file_identifier=file_identifier,
            resolved_source_paths=resolved_source_paths,
            namespace=namespace,
        )

    def delete_document_sync(self, file_identifier: str, *, namespace: Optional[str] = None) -> Dict[str, Any]:
        """同步删除指定文档。
        
        Args:
            file_identifier: 文件标识符，可以是逻辑路径、物理路径或文件名。
            namespace: 命名空间
            
        Returns:
            删除结果字典
        """
        engine = self._get_engine(namespace)
        resolved_source_paths = self._resolve_deleted_source_paths(engine, file_identifier)
        self._call_engine_sync(namespace, "delete_file", file_identifier)
        return self._build_delete_result(
            file_identifier=file_identifier,
            resolved_source_paths=resolved_source_paths,
            namespace=namespace,
        )

    async def retrieve(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """异步检索相关文档。

        Args:
            query: 查询文本
            namespace: 命名空间
            top_k: 返回结果数量
            filter_conditions: 过滤条件
            skip_rerank: 是否跳过重排序
            score_threshold: 分数阈值

        Returns:
            检索结果字典，包含 results 和 formatted_context
        """
        engine = self._get_engine(namespace)
        results = await self._call_engine(
            namespace,
            "retrieve_async",
            "retrieve",
            query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            skip_rerank=skip_rerank,
            score_threshold=score_threshold,
        )
        return {
            "namespace": self._resolve_namespace(namespace),
            "query": query,
            "count": len(results),
            "results": results,
            "formatted_context": engine.format_results_for_llm(results),
        }

    def retrieve_sync(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """同步检索相关文档。

        Args:
            query: 查询文本
            namespace: 命名空间
            top_k: 返回结果数量
            filter_conditions: 过滤条件
            skip_rerank: 是否跳过重排序
            score_threshold: 分数阈值

        Returns:
            检索结果字典，包含 results 和 formatted_context
        """
        engine = self._get_engine(namespace)
        results = self._call_engine_sync(
            namespace,
            "retrieve",
            query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            skip_rerank=skip_rerank,
            score_threshold=score_threshold,
        )
        return {
            "namespace": self._resolve_namespace(namespace),
            "query": query,
            "count": len(results),
            "results": results,
            "formatted_context": engine.format_results_for_llm(results),
        }

    def get_last_index_stats(self, *, namespace: Optional[str] = None) -> Dict[str, Any]:
        """获取指定命名空间最后一次索引操作的统计信息。"""
        engine = self._get_engine(namespace)
        return dict(engine.get_last_index_stats())

    def get_last_retrieval_stats(self, *, namespace: Optional[str] = None) -> Dict[str, Any]:
        """获取指定命名空间最后一次检索操作的统计信息。"""
        engine = self._get_engine(namespace)
        return dict(engine.get_last_retrieval_stats())

    def run_evaluation(
        self,
        *,
        namespace: Optional[str] = None,
        dataset_path: Optional[str] = None,
        cases: Optional[Iterable[RetrievalEvalCase]] = None,
        top_k: int = 5,
        retrieve_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """运行检索评估。

        Args:
            namespace: 命名空间
            dataset_path: 评估数据集路径
            cases: 评估用例列表
            top_k: 检索数量
            retrieve_kwargs: 检索参数

        Returns:
            评估结果字典
        """
        eval_cases = list(cases or [])
        if dataset_path:
            eval_cases = load_eval_cases(dataset_path)
        engine = self._get_engine(namespace)
        summary = evaluate_engine(
            engine,
            eval_cases,
            top_k=top_k,
            retrieve_kwargs=retrieve_kwargs,
        )
        result = summary.to_dict()
        result["namespace"] = self._resolve_namespace(namespace)
        return result


class KnowledgeSDK(AgentKnowledgeService):
    """SDK 导向的门面类，提供类型化请求、兼容性包装器和类型化结果。"""

    @staticmethod
    def _merge_index_metadata(request: IndexRequest) -> Dict[str, Any]:
        """合并索引请求的元数据和文档源的元数据字段。"""
        metadata = dict(request.metadata)
        metadata.update(request.source.metadata_fields())
        return metadata

    def _coerce_index_request(
        self,
        request_or_path: IndexRequest | DocumentSource | str | PathLike[str],
        options: Optional[IndexOptions],
    ) -> IndexRequest:
        """将多种输入形式统一转换为 IndexRequest 对象。

        支持传入 IndexRequest、DocumentSource、字符串路径或 PathLike 对象。
        """
        if isinstance(request_or_path, IndexRequest):
            return request_or_path

        opts = options or IndexOptions(namespace=self.default_namespace)
        if isinstance(request_or_path, DocumentSource):
            source = request_or_path
        else:
            source = DocumentSource.from_path(str(request_or_path))

        return IndexRequest(
            source=source,
            namespace=opts.namespace,
            metadata=opts.extra_meta,
            reindex_strategy="force" if opts.force_reindex else "skip_existing",
        )

    def _coerce_search_request(
        self,
        request_or_query: SearchRequest | str,
        options: Optional[RetrieveOptions],
    ) -> SearchRequest:
        if isinstance(request_or_query, SearchRequest):
            return request_or_query

        opts = options or RetrieveOptions(namespace=self.default_namespace)
        return SearchRequest(
            query=str(request_or_query),
            namespace=opts.namespace,
            top_k=opts.top_k,
            filters=opts.filter_conditions,
            score_threshold=opts.score_threshold,
            skip_rerank=opts.skip_rerank,
        )

    def _coerce_delete_target(self, target_or_identifier: DeleteTarget | str) -> str:
        """将删除目标统一转换为字符串标识符。"""
        if isinstance(target_or_identifier, DeleteTarget):
            return target_or_identifier.source_path
        return str(target_or_identifier)

    def get_health(self, namespace: Optional[str] = None) -> HealthStatus:
        """获取服务健康状态，返回类型化的 HealthStatus 对象。"""
        return HealthStatus.from_dict(self.health(namespace))

    def index(
        self,
        request_or_path: IndexRequest | DocumentSource | str | PathLike[str],
        options: Optional[IndexOptions] = None,
    ) -> IndexResult:
        """同步索引文档，支持多种输入形式，返回类型化的 IndexResult。如果传入的是字符串，会写入一个临时文件，索引完成后删除临时文件。"""
        request = self._coerce_index_request(request_or_path, options)
        with request.source.materialize() as materialized:
            raw = self.index_document_sync(
                materialized.file_path,
                namespace=request.namespace,
                extra_meta=self._merge_index_metadata(request),
                force_reindex=request.force_reindex,
                display_source_name=materialized.display_source_name,
                display_source_path=materialized.display_source_path,
            )
        return IndexResult(namespace=self._resolve_namespace(request.namespace), raw=raw)

    async def aindex(
        self,
        request_or_path: IndexRequest | DocumentSource | str | PathLike[str],
        options: Optional[IndexOptions] = None,
    ) -> IndexResult:
        """异步索引文档，支持多种输入形式，返回类型化的 IndexResult。"""
        request = self._coerce_index_request(request_or_path, options)
        with request.source.materialize() as materialized:
            raw = await self.index_document(
                materialized.file_path,
                namespace=request.namespace,
                extra_meta=self._merge_index_metadata(request),
                force_reindex=request.force_reindex,
                display_source_name=materialized.display_source_name,
                display_source_path=materialized.display_source_path,
            )
        return IndexResult(namespace=self._resolve_namespace(request.namespace), raw=raw)

    def search(
        self,
        request_or_query: SearchRequest | str,
        options: Optional[RetrieveOptions] = None,
    ) -> RetrieveResult:
        """同步检索文档，支持多种输入形式，返回类型化的 RetrieveResult。"""
        request = self._coerce_search_request(request_or_query, options)
        raw = self.retrieve_sync(
            request.query,
            namespace=request.namespace,
            top_k=request.top_k,
            filter_conditions=request.filters,
            skip_rerank=request.skip_rerank,
            score_threshold=request.score_threshold,
        )
        return RetrieveResult.from_dict(raw)

    async def asearch(
        self,
        request_or_query: SearchRequest | str,
        options: Optional[RetrieveOptions] = None,
    ) -> RetrieveResult:
        """异步检索文档，支持多种输入形式，返回类型化的 RetrieveResult。"""
        request = self._coerce_search_request(request_or_query, options)
        raw = await self.retrieve(
            request.query,
            namespace=request.namespace,
            top_k=request.top_k,
            filter_conditions=request.filters,
            skip_rerank=request.skip_rerank,
            score_threshold=request.score_threshold,
        )
        return RetrieveResult.from_dict(raw)

    def delete(self, target_or_identifier: DeleteTarget | str, options: Optional[DeleteOptions] = None) -> DeleteResult:
        """同步删除文档，支持多种输入形式，返回类型化的 DeleteResult。"""
        opts = options or DeleteOptions(namespace=self.default_namespace)
        file_identifier = self._coerce_delete_target(target_or_identifier)
        raw = self.delete_document_sync(file_identifier, namespace=opts.namespace)
        return DeleteResult(namespace=self._resolve_namespace(opts.namespace), raw=raw)

    async def adelete(self, target_or_identifier: DeleteTarget | str, options: Optional[DeleteOptions] = None) -> DeleteResult:
        """异步删除文档，支持多种输入形式，返回类型化的 DeleteResult。"""
        opts = options or DeleteOptions(namespace=self.default_namespace)
        file_identifier = self._coerce_delete_target(target_or_identifier)
        raw = await self.delete_document(file_identifier, namespace=opts.namespace)
        return DeleteResult(namespace=self._resolve_namespace(opts.namespace), raw=raw)
