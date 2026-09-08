import os
import threading
import time
from typing import Callable, Protocol


class ProgressReporter(Protocol):
    def question_started(self) -> None: ...

    def answer_ready(self) -> None: ...

    def question_finished(self, status: str = "回答完成") -> None: ...

    def model_started(self, agent: str) -> None: ...

    def model_finished(self, agent: str) -> None: ...

    def model_failed(self, agent: str) -> None: ...

    def commentary_delta(self, text: str) -> None: ...

    def commentary_finished(self) -> None: ...

    def stage_skipped(self, stage: str) -> None: ...

    def sources_discovered(self, sources: dict[str, str]) -> None: ...

    def tool_context(self, call_id: str, path: str) -> None: ...

    def workflow_started(self) -> None: ...

    def workflow_completed(self) -> None: ...

    def stage_started(self, stage: str, attempt: int) -> None: ...

    def stage_succeeded(self, stage: str) -> None: ...

    def stage_failed(self, stage: str, error: BaseException | None = None) -> None: ...

    def stage_retrying(self, stage: str, next_attempt: int) -> None: ...

    def no_pending_sources(self) -> None: ...

    def source_started(self, index: int, total: int, srcid: str) -> None: ...

    def source_succeeded(self, index: int, total: int, srcid: str) -> None: ...

    def source_failed(self, index: int, total: int, srcid: str) -> None: ...

    def agent_started(self, agent: str) -> None: ...

    def agent_succeeded(self, agent: str) -> None: ...

    def agent_failed(self, agent: str, error: BaseException | None = None) -> None: ...

    def node_started(self, node: str) -> None: ...

    def node_succeeded(self, node: str) -> None: ...

    def node_failed(self, node: str, error: BaseException | None = None) -> None: ...

    def tool_started(self, tool_name: str, call_id: str) -> None: ...

    def tool_succeeded(self, tool_name: str, call_id: str) -> None: ...

    def tool_failed(
        self,
        tool_name: str,
        call_id: str,
        error: BaseException | None = None,
    ) -> None: ...


class NullProgressReporter:
    def workflow_started(self) -> None:
        pass

    def workflow_completed(self) -> None:
        pass

    def stage_started(self, stage: str, attempt: int) -> None:
        pass

    def stage_succeeded(self, stage: str) -> None:
        pass

    def stage_failed(self, stage: str, error: BaseException | None = None) -> None:
        pass

    def stage_retrying(self, stage: str, next_attempt: int) -> None:
        pass

    def no_pending_sources(self) -> None:
        pass

    def source_started(self, index: int, total: int, srcid: str) -> None:
        pass

    def source_succeeded(self, index: int, total: int, srcid: str) -> None:
        pass

    def source_failed(self, index: int, total: int, srcid: str) -> None:
        pass

    def agent_started(self, agent: str) -> None:
        pass

    def agent_succeeded(self, agent: str) -> None:
        pass

    def agent_failed(self, agent: str, error: BaseException | None = None) -> None:
        pass

    def node_started(self, node: str) -> None:
        pass

    def node_succeeded(self, node: str) -> None:
        pass

    def node_failed(self, node: str, error: BaseException | None = None) -> None:
        pass

    def tool_started(self, tool_name: str, call_id: str) -> None:
        pass

    def tool_succeeded(self, tool_name: str, call_id: str) -> None:
        pass

    def tool_failed(
        self,
        tool_name: str,
        call_id: str,
        error: BaseException | None = None,
    ) -> None:
        pass


class TerminalProgressReporter(NullProgressReporter):
    """Translate workflow events into a chronological activity transcript."""

    STAGES = {"init": "初始化", "ingest": "资料入库", "verify": "校验与修复", "index": "构建索引"}
    AGENTS = {"alias": "整理术语与别名", "content": "生成知识页面", "ask": "组织回答",
              "search": "检索相关资料", "qmd_query": "查询搜索索引", "fix": "修复知识页面"}
    TOOLS = {"read_file": "读取文件", "write_file": "写入文件", "list_files": "查看目录",
             "mkdir": "创建目录", "read_aliases": "读取别名表", "write_aliases": "更新别名表",
             "wiki_retrieve": "检索相关资料", "qmd_query": "查询搜索索引"}
    FAST_TOOLS = {"read_file", "write_file", "list_files", "mkdir", "read_aliases", "write_aliases"}

    def __init__(self, *, live=None, clock=time.monotonic, stream_models=None, stream=None):
        from .terminal_activity import ActivityLog
        self.log = ActivityLog(live=live, clock=clock, stream=stream)
        self.clock = clock
        self.stream_models = os.environ.get("LOKI_STREAM", "1") != "0" if stream_models is None else stream_models
        self.lock = threading.RLock()
        self.started = clock()
        self.question_started_at = None
        self.stages = dict.fromkeys(self.STAGES, "待执行")
        self.names, self.results, self.targets = {}, {}, {}
        self.active_tools, self.models, self.tool_counts = {}, {}, {}
        self.total = 0
        self.current = ""

    @staticmethod
    def _display(value):
        return " ".join("".join(c if c.isprintable() else " " for c in str(value)).split())

    def _summary(self):
        with self.lock:
            return f"成功 {sum(v == '成功' for v in self.results.values())} 篇 · 失败 {sum(v == '失败' for v in self.results.values())} 篇"

    def close(self):
        self.log.close()

    def workflow_started(self):
        self.started = self.clock()
        self.log.line("开始更新知识库")

    def preparation_completed(self):
        self.close()
        if all(status == "本次跳过" for status in self.stages.values()):
            self.log.line("知识库此前已完成，本次跳过，可直接提问")
        else:
            self.log.line(f"知识库准备完成 · {self._summary()} · 总耗时 {self.clock() - self.started:.1f} 秒")

    def workflow_completed(self):
        self.close()
        self.log.line("会话已结束")

    def workflow_failed(self, error=None):
        self.close()
        self.log.line(f"任务已停止 · {self._summary()} · 总耗时 {self.clock() - self.started:.1f} 秒")

    def stage_skipped(self, stage):
        self.stages[stage] = "本次跳过"
        self.log.line(f"↷ {self.STAGES.get(stage, stage)}：此前已完成，本次跳过")

    def stage_started(self, stage, attempt):
        self.stages[stage] = "进行中"
        self.current = ""
        label = self.STAGES.get(stage, self._display(stage))
        self.log.line(f"{label}：开始" + (f"（第 {attempt} 次尝试）" if attempt > 1 else ""))

    def stage_succeeded(self, stage):
        self.stages[stage] = "已完成"
        self.current = ""
        self.log.line(f"✓ {self.STAGES.get(stage, stage)}：阶段完成")

    def stage_failed(self, stage, error=None):
        self.stages[stage] = "失败"
        self.log.line(f"✗ {self.STAGES.get(stage, stage)}：本次尝试失败"
                      + (f"（{type(error).__name__}）" if error else ""))

    def stage_retrying(self, stage, next_attempt):
        self.stages[stage] = "重试中"
        self.log.line(f"↻ {self.STAGES.get(stage, stage)}：正在重试，第 {next_attempt} 次尝试")

    def sources_discovered(self, sources):
        self.names.update(sources)
        self.total = len(self.names)
        self.log.line(f"本次共有 {self.total} 篇资料待处理")

    def no_pending_sources(self):
        self.log.line("没有待入库资料")

    def source_started(self, index, total, srcid):
        self.total = max(self.total, total)
        self.results[srcid] = "处理中"
        self.current = f"《{self._display(self.names.get(srcid, srcid))}》 · 本轮第 {index}/{total} 篇"
        self.log.line(f"正在处理 {self.current} · {self._summary()}")

    def source_succeeded(self, index, total, srcid):
        self.results[srcid], self.current = "成功", ""
        self.log.line(f"✓ 《{self._display(self.names.get(srcid, srcid))}》：资料处理完成 · {self._summary()}")

    def source_failed(self, index, total, srcid):
        self.results[srcid], self.current = "失败", ""
        self.log.line(f"✗ 《{self._display(self.names.get(srcid, srcid))}》：资料处理失败 · {self._summary()}")

    def question_started(self):
        self.close()
        with self.lock:
            self.question_started_at = self.clock()
            self.targets.clear()
            self.active_tools.clear()
            self.models.clear()
            self.tool_counts.clear()
        self.log.line("开始本轮问答")

    def answer_ready(self):
        self.close()

    def question_finished(self, status="回答完成"):
        self.close()
        with self.lock:
            if self.question_started_at is None:
                return
            elapsed = self.clock() - self.question_started_at
            self.question_started_at = None
            counts = " · ".join(f"{self.TOOLS.get(k, k)} {v} 次" for k, v in self.tool_counts.items())
            self.active_tools.clear()
            self.models.clear()
            self.targets.clear()
        self.log.line(f"{status} · 本轮耗时 {elapsed:.1f} 秒" + (f" · {counts}" if counts else ""))

    def model_started(self, agent):
        key = ("model", threading.get_ident(), agent)
        with self.lock:
            self.models[key] = self.clock()
        self.log.begin(key, f"Agent 思考中 · {self.AGENTS.get(agent, self._display(agent))}")

    def model_finished(self, agent):
        key = ("model", threading.get_ident(), agent)
        with self.lock:
            started = self.models.pop(key, self.clock())
        # Keep timing in plain logs; live history is prose and tool results.
        message = None if self.log.live else f"{self.AGENTS.get(agent, agent)}：本轮模型已返回 · 耗时 {self.clock() - started:.1f} 秒"
        self.log.end_text()
        self.log.finish(key, message)

    def model_failed(self, agent):
        key = ("model", threading.get_ident(), agent)
        with self.lock:
            self.models.pop(key, None)
        self.log.end_text()
        self.log.finish(key)

    def commentary_delta(self, text):
        self.log.text(text)

    def agent_commentary_delta(self, agent, text):
        self.log.text(text, owner=None if agent == "ask" else self.AGENTS.get(agent, agent))

    def answer_delta(self, text):
        self.log.text(text)

    def answer_incomplete(self):
        self.log.line("回答未完成。")

    def commentary_finished(self):
        self.log.end_text()
        self.log.refresh()

    def agent_started(self, agent):
        pass

    def agent_succeeded(self, agent):
        if agent == "ask":
            self.question_finished()

    def agent_failed(self, agent, error=None):
        if agent == "ask":
            self.question_finished("回答失败")
        else:
            self.log.line(f"✗ {self.AGENTS.get(agent, agent)}：模型调用失败")

    def node_failed(self, node, error=None):
        self.log.line("✗ 处理步骤失败")

    def tool_context(self, call_id, path):
        with self.lock:
            self.targets[(threading.get_ident(), call_id)] = self._display(path)

    def tool_started(self, tool_name, call_id):
        key = ("tool", threading.get_ident(), call_id)
        with self.lock:
            target = self.targets.pop((threading.get_ident(), call_id), "")
            label = self.TOOLS.get(tool_name, f"执行工具 {self._display(tool_name)}")
            label += f"：{target}" if target else ""
            self.active_tools[key] = (label, self.clock())
        self.log.begin(key, "正在" + label, log=tool_name not in self.FAST_TOOLS)

    def _tool_finished(self, tool_name, call_id, success):
        key = ("tool", threading.get_ident(), call_id)
        with self.lock:
            label, started = self.active_tools.pop(key, (self.TOOLS.get(tool_name, tool_name), self.clock()))
            if success:
                self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1
        elapsed = self.clock() - started
        duration = f" · 耗时 {elapsed:.1f} 秒" if elapsed >= 1 else ""
        status = "操作完成" if success else "操作失败，交回模型处理"
        self.log.finish(key, f"{'✓' if success else '✗'} {label}：{status}{duration}")

    def tool_succeeded(self, tool_name, call_id):
        self._tool_finished(tool_name, call_id, True)

    def tool_failed(self, tool_name, call_id, error=None):
        self._tool_finished(tool_name, call_id, False)


def report(callback: Callable[..., None], *args: object) -> None:
    try:
        callback(*args)
    except Exception:
        pass


def report_event(reporter: object, event: str, *args: object) -> None:
    try:
        callback = getattr(reporter, event)
    except Exception:
        return
    report(callback, *args)


def invoke_model(reporter, agent, model, messages):
    from .model_stream import invoke
    return invoke(reporter, agent, model, messages)
