# Industrial RAG Pipeline v2.0

工业级 RAG Pipeline
## 目录结构

```
.
├── rag/
│   ├── __init__.py        # 公开 API
│   ├── config.py          # 所有配置 dataclass（含 __post_init__ 校验）
│   ├── models.py          # DocumentChunk 数据结构
│   ├── parser.py          # 文档解析层（Docling → Unstructured → 纯文本）
│   ├── chunker.py         # 层级感知 Markdown 切块器
│   ├── embedder.py        # Embedding 服务（sync + async）
│   ├── contextual.py      # Contextual Retrieval + 三种缓存后端
│   ├── vector_store.py    # Qdrant 封装（Dense + Sparse 双路）
│   ├── reranker.py        # Reranker API 封装
│   └── engine.py          # RAGEngine 主引擎 + create_rag_engine 工厂
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_chunker.py
│   ├── test_parser.py
│   ├── test_vector_store.py
│   └── test_contextual.py
└── pyproject.toml
```

## 核心改动（相比 v1.x 单文件版）

### 1. 模块拆分
原单文件 2000+ 行拆分为 8 个职责单一的模块，每个模块可独立测试和复用。

### 2. BM25 → Qdrant Sparse Vector
| 维度 | v1.x BM25 | v2.0 Sparse Vector |
|------|-----------|-------------------|
| 存储 | 内存，重启丢失 | Qdrant 持久化 |
| 上限 | 50k chunk | 无上限 |
| 启动 | 需 scroll 全量重建 | 即时可用 |
| 删除 | 需全量重建索引 | 直接删除点 |
| 分布式 | 不支持 | Qdrant 原生支持 |
| 额外服务 | 无（但有内存限制） | 无（复用已有 Qdrant） |

### 3. 代码质量修复
- `_bm25_remove_by_source` 空实现残留已删除（BM25 整体替换）
- `reranker._http` 私有属性直接访问 → `reranker.close()` 封装接口
- `asyncio.get_event_loop()` → `asyncio.get_running_loop()`
- 所有配置类加入 `__post_init__` 校验，配置错误在构造时即抛出
- `DocumentChunk.create()` 工厂方法，避免调用方手动 `uuid.uuid4()`
- Contextual 缓存命中统计从 N+1 查询改为批量预检

### 4. 测试覆盖
新增 pytest 测试套件，覆盖：
- 配置校验（正常值 + 边界值 + 非法值）
- 切块逻辑（标题路径、递归分割、重叠、过滤）
- DocumentChunk 数据结构（工厂方法、属性、序列化）
- SparseEncoder（空文本、去重、权重范围、确定性）
- RRF 融合算法（公式验证、top_k、边界）
- DocumentParser 路由（纯文本、降级、FileNotFoundError）
- 缓存后端（读写、淘汰、后端切换）

## 安装

```bash
# 基础（不含文档解析器）
pip install -e .

# 完整安装
pip install -e ".[all]"

# 仅开发依赖
pip install -e ".[dev]"
```

## 快速开始

```python
from rag import create_rag_engine

engine = create_rag_engine(
    embed_api_key="sf-xxxxx",        # SiliconFlow / OpenAI key
    reranker_api_key="sf-xxxxx",
)
engine.startup_sync()

# 索引文档
engine.index_file("document.pdf")

# 检索
results = engine.retrieve("如何配置环境变量？", top_k=5)
print(engine.format_results_for_llm(results))

# 关闭
import asyncio
asyncio.run(engine.shutdown())
```

## FastAPI 集成

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from rag import create_rag_engine

engine = create_rag_engine(embed_api_key="...", reranker_api_key="...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.startup()
    yield
    await engine.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/retrieve")
async def retrieve(query: str, top_k: int = 5):
    results = await engine.retrieve_async(query=query, top_k=top_k)
    return engine.format_results_for_llm(results)
```

## 运行测试

```bash
# 运行全部测试
pytest

# 带覆盖率报告
pytest --cov=rag --cov-report=term-missing

# 仅运行某个模块的测试
pytest tests/test_chunker.py -v
```

## 环境变量配置

```bash
# Embedding（SiliconFlow）
EMBED_PROVIDER=proxy
PROXY_API_KEY=sf-your-key
PROXY_BASE_URL=https://api.siliconflow.cn/v1
PROXY_EMBED_MODEL=BAAI/bge-large-zh-v1.5
EMBED_DIM=1024

# Embedding（OpenAI）
EMBED_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
EMBED_DIM=1536

# Reranker
RERANKER_API_KEY=sf-your-key
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Qdrant（本地文件，默认）
QDRANT_MODE=local
QDRANT_PATH=./qdrant_data

# Contextual Retrieval 缓存
CONTEXTUAL_CACHE_BACKEND=disk   # memory / disk / redis
CONTEXTUAL_CACHE_DIR=./ctx_cache
```
