# Loki

Loki 是一个基于 Python、LangGraph 和大语言模型的本地知识库工具。它将原始文本整理为 Markdown Wiki，生成来源摘要、概念和实体页面，再结合 QMD 检索与 Wiki 文件检索，在终端中提供带来源引用的多轮问答。

知识文件保存在本地；整理和问答使用配置的模型 API，相关原文和检索内容会发送到该服务。

## 工作流程

```text
raw/ 原始资料
  → 初始化目录
  → 登记来源、维护别名、生成 Wiki 页面
  → 校验页面并尝试修复
  → 生成目录索引、更新 QMD 索引和向量
  → 终端交互问答
```

- **来源管理**：递归登记 `raw/` 下的 `.md`、`.txt` 文件，为每个来源分配 SrcID 并记录 SHA256。
- **知识整理**：每个来源拥有独立目录，保存来源摘要、概念和实体等页面；共享 `aliases.json` 维护名称与别名。
- **校验与修复**：检查来源与页面结构，对可修复的问题调用模型修复，然后重新校验。
- **双路检索**：并行运行 QMD 检索和 Wiki 检索，汇总证据后生成回答，并按需读取补充文件。
- **进度与续跑**：终端显示阶段、模型及工具执行进度；已完成的准备阶段写入状态文件，下次启动时跳过。

## 环境准备

当前本地环境使用 Python 3.13.7。代码直接导入 `readline`，建议在 macOS 或 Linux 上运行。

项目尚未提供 `pyproject.toml`、依赖清单或安装后的 CLI 命令，需要通过 `PYTHONPATH=src` 运行模块。在项目根目录创建虚拟环境并安装依赖：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install langgraph==1.2.9 langchain-core==1.6.0 langchain-openai==1.5.1 PyYAML==6.0.3
```

以上版本来自当前本地环境，并非完整的依赖锁文件。

### QMD

完整流程还需要 PATH 中可执行的 `qmd`。当前实现依赖以下命令接口，请先确认所安装的 QMD 支持这些命令：

```text
qmd init
qmd update
qmd embed
qmd query "expand: 查询内容" --format json -n 5 -c wiki -c raw
```

`qmd init` 必须能在项目中创建 `.qmd/index.yaml`（或 `index.yml`）和 `.qmd/index.sqlite`。当前仓库未包含 QMD 安装脚本或版本锁定信息。

索引阶段固定使用以下本地模型路径，运行前需要自行准备对应文件：

```text
~/.cache/qmd/models/
├── Qwen3-Embedding-0.6B-Q8_0.gguf
├── qmd-query-expansion-1.7B-q4_k_m.gguf
└── qwen3-reranker-0.6b-q8_0.gguf
```

模型路径在 `src/llm_wiki/workflow.py` 的 `index()` 中配置。该阶段会设置 QMD 的 `raw`、`wiki` 集合和模型配置，然后执行更新与向量化。

## 快速开始

### 1. 配置模型

模型服务需要兼容 OpenAI Chat Completions 接口并支持工具调用。在当前终端设置：

```bash
export MODEL_NAME="your-model-name"
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"
```

程序不会自动加载 `.env`。如果使用 `.env` 保存配置，需要在启动前自行将其中的变量加载到环境中。

可选变量 `LOKI_STREAM=0` 用于关闭模型流式输出；默认开启。

### 2. 放入资料

在项目根目录创建 `raw/`，放入 UTF-8 编码的 Markdown 或纯文本文件，也可以使用子目录：

```bash
mkdir -p raw
cp /path/to/your-notes.md raw/
```

当前登记流程只处理 `.md` 和 `.txt`，PDF、Word 等资料需要先转换为文本。

### 3. 启动

始终在项目根目录运行：

```bash
source .venv/bin/activate
PYTHONPATH=src python -m llm_wiki.workflow
```

首次启动会完成整理、校验和索引，然后显示问题输入提示。例如：

```text
请输入问题（退出 / exit / quit 结束）：这些资料的核心观点是什么？
```

可以继续追问；输入 `退出`、`exit` 或 `quit` 结束。对话历史仅保存在本次进程的内存中，重启后不会恢复。

## 新增资料与重新处理

准备阶段的完成状态保存在 `wiki/state.json`，启动时不会自动检测新增资料并重新执行已完成的阶段。

新增文件后，将 `ingest`、`verify`、`index` 设为 `false`，再启动程序：

```json
{
  "init": true,
  "ingest": false,
  "verify": false,
  "index": false
}
```

登记流程会发现新文件，整理阶段仅处理 `sources.json` 中 `ingest` 为 `false` 的来源。若只是修改了生成的 Wiki 页面，可仅重置 `verify` 和 `index`。

已登记的原始文件发生内容变化时，SHA256 检查会报错，不会自动覆盖已有登记。建议将更新内容保存为新的来源文件；仅重置阶段状态不能绕过哈希检查。

## 目录结构

```text
.
├── src/llm_wiki/
│   ├── workflow.py          # 工作流编排、状态保存和终端入口
│   ├── wiki.py              # 初始化、登记、整理、校验和目录索引
│   ├── wiki_fix.py          # 校验问题的模型修复
│   ├── wiki_query.py        # QMD 与 Wiki 双路检索
│   ├── wiki_agent.py        # 问答 Agent 和受限文件读取
│   ├── tool_execution.py    # 工具执行与进度事件
│   ├── model_stream.py      # 模型流式输出处理
│   ├── progress.py          # 进度报告
│   └── terminal_activity.py # 终端显示
├── raw/                     # 用户提供的原始文本
├── wiki/                    # 生成的知识库
│   ├── sources.json         # 来源登记、哈希与整理状态
│   ├── aliases.json         # 名称与别名
│   ├── state.json           # 准备阶段完成状态
│   ├── index.md             # Wiki 导航索引
│   ├── workflow-log/        # 整理、校验等流程日志
│   └── srcid-xxxxxx/
│       ├── source-summary-xxxxxx.md
│       ├── concepts/
│       └── entities/
└── .qmd/                    # 本地检索配置与索引
```

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `No module named llm_wiki` | 在项目根目录使用 `PYTHONPATH=src python -m llm_wiki.workflow`。 |
| 缺少 `MODEL_NAME` 等环境变量 | 在运行程序的同一终端中导出三个必需变量；`.env` 不会自动加载。 |
| 找不到 `qmd` 或索引失败 | 检查 QMD 命令接口、模型文件路径及 `.qmd/` 配置。 |
| 新资料没有被处理 | 重置 `wiki/state.json` 中的整理、校验和索引状态。 |
| 来源 SHA256 变化 | 保留原始资料不变，将修订内容作为新文件加入。 |
| 校验失败 | 查看 `wiki/workflow-log/verify-log.md`；自动修复后仍不通过时，需要根据日志处理具体问题。 |
