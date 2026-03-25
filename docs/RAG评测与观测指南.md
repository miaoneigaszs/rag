# RAG 评测与观测指南

## 目标

这份文档用于把当前仓库的 RAG 优化过程从“凭感觉调参”变成“带指标迭代”。

当前仓库已经提供三类基础能力：

1. 检索链路观测
   - `engine.get_last_index_stats()`
   - `engine.get_last_retrieval_stats()`
2. 轻量评测入口
   - `rag.evaluate_engine()`
   - `rag.evaluate_retriever()`
3. JSONL 数据集加载
   - `rag.load_eval_cases()`

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

- `fact_extraction`: 事实提取
- `technical_explanation`: 技术解释
- `complex_condition`: 条件型规则
- `multi_fact`: 多事实提取
- `table_extraction`: 表格数值提取
- `financial_extraction`: 财报/金融 PDF 抽取

仓库里现在有两份参考文件：

- [eval_dataset_template.jsonl](D:/learnsomething/rag/docs/eval_dataset_template.jsonl)
- [eval_dataset_v4.jsonl](D:/learnsomething/rag/docs/eval_dataset_v4.jsonl)

## 推荐 schema

每行一个 JSON：

```json
{
  "id": "eval_001",
  "query": "如何配置环境变量？",
  "expected_ids": ["docs/fastapi_docs/environment-variables.md"],
  "expected_heading": "环境变量",
  "ground_truth": "......",
  "difficulty": "easy",
  "multi_hop": false,
  "metadata": {
    "bucket": "technical_explanation",
    "notes": "测试基础配置说明",
    "doc_type": "技术文档"
  }
}
```

字段说明：

- `id`: 样本唯一编号
- `query`: 用户问题
- `expected_ids`: 期望命中的标识，推荐使用相对仓库根目录的稳定路径
- `expected_heading`: 可选，用于评测是否命中正确章节
- `ground_truth`: 可选，后续用于答案质量评测
- `difficulty`: 可选，样本难度标签
- `multi_hop`: 可选，是否跨段/跨节
- `metadata.bucket`: 推荐的分桶字段

不再推荐只写 basename，例如 `README.md` 或 `async.md`，因为很容易串档。

## 评测示例

```python
from rag import evaluate_engine, load_eval_cases

cases = load_eval_cases("docs/eval_dataset_v4.jsonl")
summary = evaluate_engine(engine, cases, top_k=5)
print(summary.to_dict())
```

## 当前默认关注指标

第一阶段先盯这四个：

1. `hit_rate@5`
   - 前 5 条结果里是否至少命中一个正确文档
2. `recall@5`
   - 如果问题对应多个正确文档，前 5 条覆盖了多少
3. `MRR@5`
   - 第一个正确结果排得是否足够靠前
4. `heading_hit_rate@5`
   - 如果样本给了 `expected_heading`，是否命中了正确章节

## 建议的实验顺序

在这个项目里，建议按以下顺序比较：

1. `basic` vs `advanced`
2. `skip_rerank=True` vs `False`
3. `use_contextual_retrieval=False` vs `True`
4. 不同 `chunk_size / chunk_overlap`

每次只改一个变量，否则结论不稳。

## 你后续最值得补的数据

如果你要继续让我提升效果，最有价值的是：

1. 把 `ground_truth` 里的占位语句补成真实答案
2. 对关键样本补 `expected_heading`
3. 再加 10 到 30 条真实业务 query

这样下一步就能从“检索命中评测”继续升级到“回答质量评测”。
