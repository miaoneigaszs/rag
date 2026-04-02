"""FastAPI service layer for the RAG engine."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from fastapi import FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from .evaluation import RetrievalEvalCase
from .models import DocumentSource, _validate_index_metadata
from .service import AgentKnowledgeService

_DEFAULT_NAMESPACE = "default"
_SKIP_AUTH_PATHS = {"/health"}


class APIError(RuntimeError):
    """
    API 错误类，当 API 请求出现无效或语义上的错误时抛出。
    """
    """Raised when the API request is semantically invalid."""


class IndexDocumentRequest(BaseModel):
    """
    文档索引请求模型。
    用于接收需要被索引的文件路径及相关元数据。
    """
    file_path: str = Field(..., description="要索引的绝对或相对文件路径")
    namespace: Optional[str] = None
    extra_meta: Dict[str, Any] = Field(default_factory=dict)
    force_reindex: bool = False


class DeleteDocumentRequest(BaseModel):
    """
    删除文档请求模型。
    用于接收需要被删除的文件标识及所属命名空间。
    """
    file_identifier: str = Field(..., description="要删除的目标源路径或旧版文件名")
    namespace: Optional[str] = None


class RetrieveRequest(BaseModel):
    """
    文档检索请求模型。
    用于指定检索的查询条件、过滤参数及相关配置项。
    """
    query: str = Field(..., min_length=1)
    namespace: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)
    filter_conditions: Dict[str, Any] = Field(default_factory=dict)
    skip_rerank: bool = False
    score_threshold: float = Field(default=0.0, ge=0.0)


class EvaluationCaseInput(BaseModel):
    """
    评测用例输入模型。
    表示单个检索评测预期和配置。
    """
    query: str = Field(..., min_length=1)
    expected_ids: List[str] = Field(default_factory=list)
    expected_heading: str = ""
    filter_conditions: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


T = TypeVar("T")


class SuccessResponse(BaseModel, Generic[T]):
    """
    统一标准的成功响应包装模型。
    泛型 T 对应返回的数据实体类型。
    """
    ok: Literal[True] = True
    data: T
    error: None = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class IndexDocumentResponseData(BaseModel):
    """文档索引响应的数据模型。"""
    status: str = Field(description="索引状态，例如 ok、skipped 或 error")
    doc_id: Optional[str] = Field(default=None, description="由规范化 source_path 派生得到的稳定文档标识")
    content_hash: Optional[str] = Field(default=None, description="保留的内容哈希，供可观测性及幂等性分析使用")
    source_file: str = Field(default="", description="记录于被索引分块上的显示文件名")
    source_path: str = Field(description="将在后续的检索与删除调用的标准规范资源标识")
    file_type: Optional[str] = Field(default=None, description="检测到的解析器对应的文件类型")
    chunks: int = Field(default=0, description="当前文档被生成的块（chunk）数量")
    namespace: Optional[str] = Field(default=None, description="接收并处理此处索引的最终命名空间")


class IndexStatsMeta(BaseModel):
    """索引统计元数据模型。"""
    namespace: str = Field(description="当前请求被解析到的目标命名空间")
    index_stats: Dict[str, Any] = Field(default_factory=dict, description="最近生成的引擎侧索引可观测性快照")


class DeleteDocumentResponseData(BaseModel):
    """删除文档响应的数据模型。"""
    deleted: bool = Field(description="指明该删除操作是否已成功执行")
    request_identifier: str = Field(description="作为响应回传展现给客户端时的请求删除标识")
    resolved_source_paths: List[str] = Field(default_factory=list, description="删除操作背后真正定位到并作用的规范资源标识列表")
    source_path: Optional[str] = Field(default=None, description="如果仅仅匹配上唯一的资源时展示其规范 source_path；模糊删除多条数据时这里留空")
    file_identifier: str = Field(description="request_identifier 的兼容备选别名")
    namespace: str = Field(description="真正执行该删除动作所在的命名空间")


class EvaluationRunRequest(BaseModel):
    """
    运行评测的请求模型。
    可选择提供数据集路径或显式传入评测用例集合。
    """
    dataset_path: Optional[str] = None
    namespace: Optional[str] = None
    cases: List[EvaluationCaseInput] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieve_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source(self) -> "EvaluationRunRequest":
        if not self.dataset_path and not self.cases:
            raise ValueError("Either dataset_path or cases must be provided")
        return self


def _ok(data: Any, *, meta: Optional[Dict[str, Any]] = None, status_code: int = 200) -> JSONResponse:
    """包装成功的 JSON 响应。"""
    return JSONResponse(
        status_code=status_code,
        content={"ok": True, "data": data, "error": None, "meta": meta or {}},
    )


def _resolve_namespace(explicit_namespace: Optional[str], header_namespace: Optional[str]) -> str:
    """解析命名空间，优先使用 explicit_namespace。"""
    candidate = explicit_namespace.strip() if explicit_namespace else None
    header_candidate = header_namespace.strip() if header_namespace else None
    return AgentKnowledgeService.normalize_namespace(candidate or header_candidate or _DEFAULT_NAMESPACE)


def _resolve_api_key(request: Request) -> Optional[str]:
    """从请求头提取 API 鉴权。"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return request.headers.get("X-API-Key")


def _write_upload_bytes_to_temp(data: bytes, *, source_name: str, upload_dir: Path) -> str:
    """将接收的文件字节流写入本地临时文件。"""
    suffix = Path(source_name or "upload.bin").suffix
    upload_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="rag-upload-", suffix=suffix, dir=str(upload_dir))
    os.close(fd)
    Path(temp_path).write_bytes(data)
    return temp_path


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """服务生命周期管理，在 FastAPI 启动和关闭时调用对应的 RAG 服务生命周期。"""
    service = app.state.service
    await service.startup()
    try:
        yield
    finally:
        await service.shutdown()


async def _get_service(request: Request) -> AgentKnowledgeService:
    return request.app.state.service


def _install_openapi_security(app: FastAPI, *, enabled: bool) -> None:
    """如果配置了 API Key，开启 OpenAPI (Swagger) 页面的安全性。"""
    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        if enabled:
            components = schema.setdefault("components", {})
            security_schemes = components.setdefault("securitySchemes", {})
            security_schemes["ApiKeyAuth"] = {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "Provide the configured API key via the X-API-Key header.",
            }
            security_schemes["BearerAuth"] = {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "API key",
                "description": "Provide the configured API key via Authorization: Bearer <token>.",
            }

            for path, path_item in schema.get("paths", {}).items():
                if path in _SKIP_AUTH_PATHS:
                    continue
                for method, operation in path_item.items():
                    if method.lower() not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                        continue
                    operation["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi


async def _load_eval_cases_from_request(payload: EvaluationRunRequest) -> List[RetrievalEvalCase]:
    if payload.dataset_path:
        return []

    return [
        RetrievalEvalCase(
            query=case.query,
            expected_ids=case.expected_ids,
            expected_heading=case.expected_heading,
            filter_conditions=case.filter_conditions,
            metadata=case.metadata,
        )
        for case in payload.cases
    ]


def create_app(
    *,
    service: Optional[AgentKnowledgeService] = None,
    engine: Any = None,
    api_key: Optional[str] = None,
    upload_dir: Optional[str] = None,
) -> FastAPI:
    """
    创建并初始化 FastAPI 实例应用。
    
    注册各类中间件、异常处理，并挂载 RAG 核心 API 路由。
    """
    service = service or AgentKnowledgeService(engine=engine)
    resolved_api_key = api_key if api_key is not None else os.getenv("RAG_API_KEY")
    resolved_upload_dir = Path(upload_dir or os.getenv("RAG_UPLOAD_DIR", ".rag_uploads")).resolve()

    app = FastAPI(
        title="Agent Knowledge Access Service",
        version="0.1.0",
        description="HTTP service layer for indexing, retrieval, observability, and evaluation.",
        lifespan=_lifespan,
    )
    app.state.service = service
    app.state.api_key = resolved_api_key
    app.state.upload_dir = resolved_upload_dir
    _install_openapi_security(app, enabled=bool(resolved_api_key))

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        configured_api_key = request.app.state.api_key
        if not configured_api_key or request.url.path in _SKIP_AUTH_PATHS:
            return await call_next(request)

        provided_api_key = _resolve_api_key(request)
        if provided_api_key != configured_api_key:
            return JSONResponse(
                status_code=401,
                content={"ok": False, "data": None, "error": "Unauthorized", "meta": {}},
            )
        return await call_next(request)

    @app.exception_handler(APIError)
    async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "data": None, "error": str(exc), "meta": {}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"ok": False, "data": None, "error": "Validation failed", "meta": {"details": exc.errors()}},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "data": None, "error": str(exc), "meta": {}},
        )

    @app.get("/health")
    async def health(
        request: Request,
        namespace: Optional[str] = Query(default=None),
        x_namespace: Optional[str] = Header(default=None, alias="X-Namespace"),
    ) -> JSONResponse:
        knowledge_service = await _get_service(request)
        resolved_namespace = _resolve_namespace(namespace, x_namespace)
        return _ok(knowledge_service.health(resolved_namespace))

    @app.post(
        "/documents/index",
        response_model=SuccessResponse[IndexDocumentResponseData],
        summary="Index a file by path",
        description="Index a server-side file path into the resolved namespace. The returned data includes the file's canonical source_path for later retrieval and deletion.",
    )
    async def index_document(
        payload: IndexDocumentRequest,
        request: Request,
        x_namespace: Optional[str] = Header(default=None, alias="X-Namespace"),
    ) -> JSONResponse:
        knowledge_service = await _get_service(request)
        resolved_namespace = _resolve_namespace(payload.namespace, x_namespace)
        result = await knowledge_service.index_document(
            payload.file_path,
            namespace=resolved_namespace,
            extra_meta=payload.extra_meta,
            force_reindex=payload.force_reindex,
        )
        return _ok(result, meta={"index_stats": knowledge_service.get_last_index_stats(namespace=resolved_namespace)})

    @app.post(
        "/documents/upload",
        response_model=SuccessResponse[IndexDocumentResponseData],
        summary="Upload and index a document",
        description="Upload a document and index it in the target namespace. The resulting canonical source_path uses the fixed bytes://api-upload/<filename> form. Any metadata keys with sdk_source_* prefixes are rejected with HTTP 400.",
    )
    async def upload_document(
        request: Request,
        file: UploadFile = File(..., description="待索引的二进制文档本身"),
        namespace: Optional[str] = Form(default=None, description="请求发送的命名空间字段（支持隐式配置）。如果在 Body 中设置，它的权重大于 X-Namespace。"),
        force_reindex: bool = Form(default=False, description="即便已经判定完全一样的资源源已存在，依然强制发起对它的更新覆盖写入"),
        extra_meta_json: str = Form(default="{}", description="属于用户个人的附带元数据的 JSON 对象。请注意保留 sdk_source_* 在内的字段不可使用。"),
        x_namespace: Optional[str] = Header(default=None, alias="X-Namespace"),
    ) -> JSONResponse:
        knowledge_service = await _get_service(request)
        resolved_namespace = _resolve_namespace(namespace, x_namespace)
        try:
            extra_meta = json.loads(extra_meta_json or "{}")
            if not isinstance(extra_meta, dict):
                raise APIError("extra_meta_json must decode to an object")
        except json.JSONDecodeError as exc:
            raise APIError(f"Invalid extra_meta_json: {exc}") from exc

        file_bytes = await file.read()
        raw_source_name = ((file.filename or "upload.bin").replace("\\", "/").split("/")[-1].strip() or "upload.bin")
        try:
            upload_source = DocumentSource.from_bytes(
                file_bytes,
                source_name=raw_source_name,
                upload_origin="api-upload",
            )
        except ValueError as exc:
            raise APIError(str(exc)) from exc

        try:
            validated_extra_meta = _validate_index_metadata(extra_meta)
        except (TypeError, ValueError) as exc:
            raise APIError(str(exc)) from exc

        merged_extra_meta = dict(validated_extra_meta)
        merged_extra_meta.update(upload_source.metadata_fields())
        temp_path = _write_upload_bytes_to_temp(
            file_bytes,
            source_name=upload_source.effective_name,
            upload_dir=request.app.state.upload_dir,
        )
        try:
            result = await knowledge_service.index_document(
                temp_path,
                namespace=resolved_namespace,
                extra_meta=merged_extra_meta,
                force_reindex=force_reindex,
                display_source_name=upload_source.effective_name,
                display_source_path=upload_source.display_path,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return _ok(
            result,
            meta={
                "namespace": resolved_namespace,
                "index_stats": knowledge_service.get_last_index_stats(namespace=resolved_namespace),
            },
        )

    @app.post(
        "/documents/delete",
        response_model=SuccessResponse[DeleteDocumentResponseData],
        summary="Delete indexed resources",
        description="Delete indexed resources using a strict source_path or legacy basename. request_identifier echoes the original client input, resolved_source_paths lists every canonical resource matched for deletion, and source_path is only present when exactly one canonical resource matched.",
    )
    async def delete_document(
        payload: DeleteDocumentRequest,
        request: Request,
        x_namespace: Optional[str] = Header(default=None, alias="X-Namespace"),
    ) -> JSONResponse:
        knowledge_service = await _get_service(request)
        resolved_namespace = _resolve_namespace(payload.namespace, x_namespace)
        result = await knowledge_service.delete_document(payload.file_identifier, namespace=resolved_namespace)
        return _ok(result)

    @app.post("/retrieve")
    async def retrieve(
        payload: RetrieveRequest,
        request: Request,
        x_namespace: Optional[str] = Header(default=None, alias="X-Namespace"),
    ) -> JSONResponse:
        knowledge_service = await _get_service(request)
        resolved_namespace = _resolve_namespace(payload.namespace, x_namespace)
        result = await knowledge_service.retrieve(
            payload.query,
            namespace=resolved_namespace,
            top_k=payload.top_k,
            filter_conditions=payload.filter_conditions or None,
            skip_rerank=payload.skip_rerank,
            score_threshold=payload.score_threshold,
        )
        return _ok(result, meta={"retrieval_stats": knowledge_service.get_last_retrieval_stats(namespace=resolved_namespace)})

    @app.get("/metrics/last_index")
    async def last_index_metrics(
        request: Request,
        namespace: Optional[str] = Query(default=None),
        x_namespace: Optional[str] = Header(default=None, alias="X-Namespace"),
    ) -> JSONResponse:
        knowledge_service = await _get_service(request)
        resolved_namespace = _resolve_namespace(namespace, x_namespace)
        return _ok(knowledge_service.get_last_index_stats(namespace=resolved_namespace))

    @app.get("/metrics/last_retrieval")
    async def last_retrieval_metrics(
        request: Request,
        namespace: Optional[str] = Query(default=None),
        x_namespace: Optional[str] = Header(default=None, alias="X-Namespace"),
    ) -> JSONResponse:
        knowledge_service = await _get_service(request)
        resolved_namespace = _resolve_namespace(namespace, x_namespace)
        return _ok(knowledge_service.get_last_retrieval_stats(namespace=resolved_namespace))

    @app.post("/evaluation/run")
    async def run_evaluation(
        payload: EvaluationRunRequest,
        request: Request,
        x_namespace: Optional[str] = Header(default=None, alias="X-Namespace"),
    ) -> JSONResponse:
        knowledge_service = await _get_service(request)
        resolved_namespace = _resolve_namespace(payload.namespace, x_namespace)
        cases = await _load_eval_cases_from_request(payload)
        summary = knowledge_service.run_evaluation(
            namespace=resolved_namespace,
            dataset_path=payload.dataset_path,
            cases=cases,
            top_k=payload.top_k,
            retrieve_kwargs=payload.retrieve_kwargs or None,
        )
        return _ok(summary)

    return app
