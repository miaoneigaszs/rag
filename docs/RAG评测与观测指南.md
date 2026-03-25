# RAG 评测与观测指南

## 目标

这份文档用于把当前仓库的 RAG 优化过程从“凭感觉调参”变成“带指标迭代”。

当前仓库已经提供两类基础能力：

1. 检索链路观测
   - `engine.get_last_index_stats()`
   - `engine.get_last_retrieval_stats()`
2. 轻量评测入口
   - `rag.evaluate_engine()`
   - `rag.evaluate_retriever()`

## 你现在能直接看什么

### 1. 最近一次索引观测

```python
stats = engine.get_last_index_stats()
print(stats)
```

重点字段：

- `parse_ms`
- `chunk_ms`
- `contextual_ms`
- `embed_ms`
- `upsert_ms`
- `chunk_count`
- `file_type`
- `parsed_chars`

### 2. 最近一次检索观测

```python
stats = engine.get_last_retrieval_stats()
print(stats)
```

重点字段：

- `query_embed_ms`
- `dense_ms`
- `sparse_ms`
- `fusion_ms`
- `rerank_ms`
- `section_expand_ms`
- `dense_hit_count`
- `sparse_hit_count`
- `fused_hit_count`
- `result_count`
- `used_rerank`
- `used_advanced_expansion`
- `result_doc_ids`
- `result_source_paths`

## 如何构造评测集

推荐先做一个小而准的评测集，不要一上来追求大。

建议第一版：

- 10 到 30 个 query
- 每个 query 至少有 1 个明确正确答案文档
- 最好覆盖不同问题类型

建议分桶：

- `factoid`: 事实型问题
- `procedure`: 步骤型问题
- `config`: 配置型问题
- `summary`: 总结型问题
- `cross_section`: 需要 section 扩展的问题

仓库里已附带模板文件：

- [eval_dataset_template.jsonl](D:/learnsomething/rag/docs/eval_dataset_template.jsonl)

## 模板格式

每行一个 JSON：

```json
{"query": "如何配置环境变量？", "expected_ids": ["/abs/path/doc.md"], "metadata": {"bucket": "config"}}
```

字段说明：

- `query`: 用户问题
- `expected_ids`: 期望命中的标识，可填 `doc_id`、`source_path` 或 `source_file`
- `metadata`: 可选，用于分桶分析

推荐优先使用 `source_path`，因为最稳定。

## 评测示例

```python
import json
from pathlib import Path
from rag import RetrievalEvalCase, evaluate_engine

cases = []
for line in Path("docs/eval_dataset_template.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    cases.append(
        RetrievalEvalCase(
            query=item["query"],
            expected_ids=item["expected_ids"],
            metadata=item.get("metadata", {}),
        )
    )

summary = evaluate_engine(engine, cases, top_k=5)
print(summary.to_dict())
```

## 当前默认关注指标

第一阶段先盯这三个：

1. `hit_rate@5`
   - 前 5 条结果里是否至少命中一个正确文档
2. `recall@5`
   - 如果问题对应多个正确文档，前 5 条覆盖了多少
3. `MRR@5`
   - 第一个正确结果排得是否足够靠前

## 建议的实验顺序

在这个项目里，建议按以下顺序比较：

1. `basic` vs `advanced`
2. `skip_rerank=True` vs `False`
3. `use_contextual_retrieval=False` vs `True`
4. 不同 `chunk_size / chunk_overlap`

每次只改一个变量，否则结论不稳。

## 我最需要你提供的帮助

如果你要我继续把效果往上做，最有价值的是你给我这两类真实材料：

1. 用于索引的真实文档
   - 3 到 10 份就够
   - 最好包含你实际关心的 PDF / Markdown / DOCX
2. 真实问题清单
   - 10 到 30 条
   - 每条问题最好标明正确文档路径，或者告诉我答案在哪份文档里

如果你愿意，下一步可以直接把这些文档和问题给我，我会基于这套模板帮你落第一版真实评测集，并开始做效果优化。
