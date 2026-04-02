# AtriNexus RAG SDK

一个面向工程使用的 RAG 项目，主打 **SDK 优先**。它把文档解析、分层切块、Dense + Sparse 混合检索、RRF 融合、可选 rerank、上下文扩展和可观测性封装成一条稳定链路，同时也提供 FastAPI 服务层和 Agent 适配层。

## 核心能力

- 文档解析：支持 Markdown、PDF、DOCX、PPTX、HTML 等输入，优先走结构化解析，再按需降级。
- 分层切块：按标题层级组织 chunk，保留 heading path 和 section 结构。
- 混合检索：Dense 向量检索 + Sparse/BM25 检索 + RRF 融合。
- 可选重排：支持 reranker 提升最终 Top-K 质量。
- 上下文扩展：支持 `advanced` RAG 模式和 contextual retrieval。
- 可观测性：索引和检索都有最近一次统计信息可查。
- 多入口：SDK、HTTP API、MCP Adapter、Skill Adapter。

## 安装

### 最小安装

适合阅读代码、跑单元测试、做本地开发。

```bash
pip install -e .
pip install -e ".[dev]"
```

### 运行检索链路

适合本地脚本或业务代码直接调用 SDK。

```bash
pip install -e ".[runtime,cache]"
```

### 启动 HTTP API

适合把 RAG 作为独立服务提供给前端、多语言系统或其他服务调用。

```bash
pip install -e ".[service]"
```

### 全量安装

需要解析、检索、服务、测试能力时使用。

```bash
pip install -e ".[all]"
```

## 环境变量

复制示例文件并按实际模型服务商修改：

```bash
cp .env.example .env
```

最小可运行配置通常至少需要：

```env
EMBED_PROVIDER=proxy
PROXY_API_KEY=your_key
RERANKER_API_KEY=your_key
QDRANT_MODE=local
QDRANT_PATH=./qdrant_data
```

常见可选项：

- `EMBED_PROVIDER=openai` 时使用 OpenAI embedding。
- `USE_CONTEXTUAL_RETRIEVAL=true` 可开启上下文增强。
- `RAG_MODE=advanced` 可启用 section 级扩展。
- `CONTEXTUAL_CACHE_BACKEND` 可选 `memory`、`disk`、`redis`。

## SDK 快速开始

```python
from rag import create_sdk, DocumentSource, IndexRequest, SearchRequest

sdk = create_sdk()
sdk.startup_sync()

index_result = sdk.index(
    IndexRequest(
        source=DocumentSource.from_text(
            """# 年假规则

员工每年有 10 天年假。""",
            source_name="policy.md",
        ),
        namespace="hr",
    )
)

result = sdk.search(SearchRequest(query="年假怎么计算？", namespace="hr", top_k=3))
print(result.formatted_context)
```

## HTTP API

如果你要把项目作为服务启动：

```bash
python main.py
```

也可以直接用 uvicorn：

```bash
uvicorn rag.api:create_app --factory --host 127.0.0.1 --port 8000
```

常用接口：

- `POST /documents/index`
- `POST /documents/upload`
- `POST /documents/delete`
- `POST /retrieve`
- `GET /metrics/last_index`
- `GET /metrics/last_retrieval`
- `POST /evaluation/run`

## 目录结构

```text
.
├── rag/
├── tests/
├── .github/workflows/
├── main.py
├── pyproject.toml
├── requirements*.txt
└── README.md
```

## 测试

```bash
python -m compileall rag tests
pytest -q
```

如果你要跑完整测试集，本地也建议安装 `dev + runtime + service`：

```bash
pip install -e ".[dev,runtime,service]"
```

CI 里同样会安装这些依赖，再运行 `pytest -q`。

## 评测

如果你有自己的评测 JSONL 数据集，可以直接调用：

```python
from rag import create_rag_engine, evaluate_engine, load_eval_cases

engine = create_rag_engine()
engine.startup_sync()

cases = load_eval_cases("path/to/eval_dataset.jsonl")
summary = evaluate_engine(engine, cases, top_k=5)
print(summary.to_dict())
```

## 版本管理建议

这个仓库建议按功能分支开发：先从 `main` 拉出分支，改完后先看 `git status` 和 `git diff --stat`，再提交、推送、合并。

如果你不确定某个文件是否该提交，优先把它归类为：源码、可复现配置、生成物或本地缓存。前两类可以进仓库，后两类不要进。
