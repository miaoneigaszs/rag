# Industrial RAG Pipeline

一个面向工程演进的 RAG 项目，包含分层切块、Dense + Sparse 混合检索、RRF 融合、可选 rerank、Contextual Retrieval、以及 Docling / Unstructured 解析降级链路。

更多工程化细节见 [docs/工程化改造说明.md](docs/工程化改造说明.md)。

## 安装分层

### 1. 最小运行依赖

适合本地开发、单元测试、阅读代码。

```bash
pip install -e .
```

### 2. 主检索链路

适合实际运行 embedding + Qdrant 检索。

```bash
pip install -e ".[runtime,cache]"
```

### 3. 完整功能

包含文档解析、缓存后端、开发测试工具。

```bash
pip install -e ".[all]"
```

### 4. requirements 文件

如果你更习惯 `pip -r`：

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-all.txt
```

## Extras 说明

- `runtime`: `openai`、`httpx`、`qdrant-client`
- `cache`: `diskcache`、`redis`
- `parsing`: `docling`、`unstructured[all-docs]` 及其相关依赖
- `dev`: `pytest`、`pytest-cov`、`ruff`、`pre-commit`
- `all`: 全部功能与开发依赖

## 环境变量

复制 `.env.example` 后按需修改：

```bash
cp .env.example .env
```

最小可运行配置通常至少需要：

```bash
EMBED_PROVIDER=proxy
PROXY_API_KEY=sf-your-key
RERANKER_API_KEY=sf-your-key
QDRANT_MODE=local
QDRANT_PATH=./qdrant_data
```

说明：

- 如果使用 OpenAI 官方 embedding，把 `EMBED_PROVIDER` 改成 `openai` 并设置 `OPENAI_API_KEY`。
- `Contextual Retrieval` 默认关闭，开启后通常还需要 `openai` 相关能力。
- 多模态图片描述能力使用 `VISION_*` 变量。

## 快速开始

```python
from rag import create_rag_engine

engine = create_rag_engine(
    embed_api_key="sf-xxxxx",
    reranker_api_key="sf-xxxxx",
    rag_mode="advanced",
    use_contextual_retrieval=True,
)
engine.startup_sync()

engine.index_file("document.pdf")
results = engine.retrieve("如何配置环境变量？", top_k=5)
print(engine.format_results_for_llm(results))
```

## 目录结构

```text
.
├── rag/
│   ├── __init__.py
│   ├── chunker.py
│   ├── config.py
│   ├── contextual.py
│   ├── embedder.py
│   ├── engine.py
│   ├── models.py
│   ├── parser.py
│   ├── reranker.py
│   └── vector_store.py
├── tests/
├── docs/
├── .env.example
├── .pre-commit-config.yaml
├── pyproject.toml
└── .github/workflows/ci.yml
```

## 代码质量

仓库现在提供最小可用的 `ruff + pre-commit` 基线，先覆盖最常见的问题：

- import 排序
- 明显的语法/未定义变量错误
- 基础 bug 风险检查
- 行尾空格、缺失 EOF newline、YAML/TOML 基本校验

安装与使用：

```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
ruff check .
```

说明：

- 当前只接入 `ruff check --fix`，还没有启用 `ruff format`。
- 这是刻意保守的“半步”方案，先把静态检查接入，再决定是否统一格式化全仓库。

## 效果评测

仓库现在提供一个轻量评测入口，可以直接对小型 query 集合计算 `hit_rate@k`、`recall@k` 和 `MRR@k`。

```python
from rag import evaluate_engine, load_eval_cases

cases = load_eval_cases("docs/eval_dataset_v4.jsonl")
summary = evaluate_engine(engine, cases, top_k=5)
print(summary.to_dict())
```

如果你同时想看最近一次检索链路的耗时与召回量：

```python
stats = engine.get_last_retrieval_stats()
print(stats)
```

## 测试

本地执行：

```bash
python -m compileall rag tests
pytest -q
```

当前基线：

- `78 passed`
- `1 skipped`

跳过项为 `docling` 可选依赖相关测试，未安装时跳过属于预期行为。

## CI

仓库已提供 GitHub Actions 工作流：

- 安装 `.[dev]`
- 执行 `python -m compileall rag tests`
- 执行 `pytest -q`

工作流文件见 [.github/workflows/ci.yml](.github/workflows/ci.yml)。
