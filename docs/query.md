# Wiki 交互问答

在项目根目录设置既有 `MODEL_NAME`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`
环境变量后运行：

```sh
PYTHONPATH=src .venv/bin/python -m llm_wiki.workflow
```

完成 `init → ingest → verify → index` 后进入 `ask → agent ⇄ tools → ask`。
输入 `退出`、`exit`、`quit`，或按 Ctrl+C / Ctrl+D 结束。
空输入重新等待问题；本次会话保留上下文，可以连续追问。

每轮 Prompt 注入当前 `wiki/index.md`。相关问题优先调用 `wiki_retrieve`，
结合 QMD 和 Wiki 检索报告回答并引用文件来源。索引描述只用于导航。
每轮最多 30 次模型调用，格式修复也计入预算；失败后可继续提问。

文件工具使用相对项目根目录的路径：

- `read_file(path)`：读取 `raw/` 或 `wiki/` 内 UTF-8 文本。
- `write_file(path, content)`：创建或更新 `wiki/query/` 内 Markdown。

工具拒绝绝对路径、`..` 和符号链接，写入拒绝更新硬链接文件。
相关问答完成后由程序自动新增带 UTC 时间戳和随机标识的 Markdown 归档。
混合问题只保存相关子问题与对应答案；无关问题、检索失败和未完成回答不归档。
归档失败不丢弃已生成的答案，会显示保存失败提示。
历史归档不作为检索证据，也不会触发每轮重新索引。

`wiki/state.json` 只保存四个准备阶段的完成标志。旧 `retrieve` 标志会被忽略，
不会跳过问答。会话历史保存在内存，退出后不恢复。

程序接入时使用 `build_graph(checkpointer=InMemorySaver())`，调用时提供
`configurable.thread_id` 和足够的 `recursion_limit`（默认 30 轮模型调用推荐 100）。
收到 `__interrupt__` 后用 `Command(resume=用户输入)` 和相同配置恢复图。
每次恢复有独立的图步数预算。
