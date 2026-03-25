"""pytest 公共 fixture。"""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_markdown() -> str:
    """用于切块与解析测试的示例 Markdown。"""
    return """# Python 工程实践

## 项目结构设计

### 1.1 推荐目录

一个清晰的项目结构能降低协作成本，也能减少后期维护风险。

```
project/
├── src/
├── tests/
├── docs/
└── pyproject.toml
```

合理的目录布局也有助于 CI、打包和部署流水线保持稳定。

### 1.2 配置管理

建议使用 python-dotenv 只在显式入口加载环境变量。

如果项目规模继续增大，也可以逐步演进到 pydantic-settings。

## 异步与并发

### 2.1 IO 密集任务

Python asyncio 很适合处理 IO 密集场景，例如网络请求与向量检索。

如果遇到 CPU 密集型任务，则应考虑 multiprocessing 或 ProcessPoolExecutor。

### 2.2 数值计算

NumPy 与 PyTorch 在数值计算场景里都很常见。
Embedding 相关流程通常还会引入额外的批处理与缓存策略。

## 测试与部署建议

### 3.1 单元测试

建议同时使用 pytest 与 pytest-asyncio 来覆盖同步和异步路径。

覆盖率目标可以先做到 80%，没有必要盲目追求 100%。

### 3.2 集成测试

可以配合 Docker Compose 搭建稳定的本地依赖环境。

如果需要更强隔离，也可以引入 testcontainers-python 管理 Qdrant、Redis 等组件。
"""


@pytest.fixture(scope="session")
def sample_plain_text() -> str:
    """用于基础文本切块测试的纯文本样例。"""
    return (
        "这是一段没有 Markdown 标题的普通文本。"
        "它应该被当作一个整体 section 处理。"
        "如果内容继续变长，再由递归切块逻辑进一步拆分。"
    )
