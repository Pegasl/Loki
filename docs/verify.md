# Verify 与自动修复

工作流的 verify 阶段按以下顺序执行：全量检查 → 按来源修复一轮 → 全量复检。
检查通过时不会调用模型；复检失败时停止工作流，不进入 retrieve，也不自动重跑 fix。
其他工作流阶段仍保留原来的重试策略。

自动修复处理 frontmatter 格式、空字段/正文、缺失的 Source Summary、Concept、Entity，
以及与登记信息不一致的 Summary 字段。fix 会收到该来源的原文、登记信息和本轮错误列表。
没有可靠 Concept/Entity 时，可以生成说明未识别到可靠条目的 placeholder，不能编造事实。

原文缺失或哈希变化、登记损坏或重复、目录归属不明、多余 Summary、文件读取失败等问题
不会交给 fix。一个来源只要存在这类错误，就跳过该来源的修复；其他可修复来源继续处理。
这些问题仍会使最终验证失败。

fix 只能读当前来源目录，写入报错的 Markdown 文件，或在缺失的 concepts/entities 类别下
创建页面。不能修改 index.md、原文、登记表、aliases 或其他来源，也不能删除文件。
每个来源最多执行一轮 fix，每轮最多 12 次模型调用，模型客户端不自动重试。
修复直接保存文件；中途失败时保留已写入内容，再由全量复检决定是否通过。

日志保存在 `wiki/workflow-log/verify-log.md`，依次记录 `verify` 的原始错误、`fix`
的修改文件及执行结果、`recheck` 的最终结果。fix 的 `completed` 只代表 agent 执行结束。
终端复用现有 agent/tool 进度事件。

## 使用

原有 Python 接口默认只检查并写日志：

```python
from llm_wiki.wiki import wiki_verify

wiki_verify(".")
wiki_verify(".", fix=True)  # 显式启用一轮自动修复
```

工作流默认启用修复。模型沿用 ingest 的环境变量：`MODEL_NAME`、`OPENAI_API_KEY`、
`OPENAI_BASE_URL`，运行前需要将它们设置到进程环境中。

工作流依据保存的状态跳过已完成的阶段。

## 验证实现

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

测试使用模拟模型和临时目录，不调用实际模型或修改现有 Wiki 内容。
