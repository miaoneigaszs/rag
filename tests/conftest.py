"""
tests/conftest.py
=================
pytest 共享 fixture。
"""

import pytest


@pytest.fixture(scope="session")
def sample_markdown() -> str:
    """多层级 Markdown 文档，供多个测试复用。"""
    return """# Python 工程实践指南

## 第一章：项目结构

### 1.1 目录规范

推荐使用如下标准目录结构：

```
project/
├── src/
├── tests/
├── docs/
└── pyproject.toml
```

良好的目录结构有助于团队协作和代码维护。

### 1.2 配置管理

使用 python-dotenv 管理环境变量，避免将敏感信息提交到版本控制系统。

推荐使用 pydantic BaseSettings 进行配置校验，确保配置的类型安全。

## 第二章：性能优化

### 2.1 异步编程

Python asyncio 适合 IO 密集型任务（网络请求、文件读写）。

对于 CPU 密集型任务，应使用 multiprocessing 或 ProcessPoolExecutor。

### 2.2 向量化计算

NumPy 和 PyTorch 提供了高效的批量矩阵运算，
在处理 Embedding 向量时应尽量使用批量接口，避免逐条调用。

## 第三章：测试与质量保障

### 3.1 单元测试

使用 pytest 框架，配合 pytest-asyncio 处理异步测试用例。

测试覆盖率建议不低于 80%，核心业务逻辑应达到 100%。

### 3.2 集成测试

使用 Docker Compose 搭建测试环境，确保测试环境与生产环境一致。

可使用 testcontainers-python 在测试中动态启动依赖服务（如 Qdrant、Redis）。
"""


@pytest.fixture(scope="session")
def sample_plain_text() -> str:
    """无标题的纯文本，用于测试无结构文档的处理。"""
    return (
        "这是一段没有任何 Markdown 格式的纯文本内容。"
        "它用于测试切块器在面对无结构文档时的行为。"
        "切块器应该能够正常处理这种情况，不依赖标题进行分割。"
    )
