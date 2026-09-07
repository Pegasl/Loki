import os
import sys
import shutil
import threading
import time
import unicodedata
from typing import Callable, Protocol


class ProgressReporter(Protocol):
    def question_started(self) -> None: ...

    def answer_ready(self) -> None: ...

    def question_finished(self, status: str = "回答完成") -> None: ...

    def model_started(self, agent: str) -> None: ...

    def model_finished(self, agent: str) -> None: ...

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
    """Human-readable events and live preparation/question progress."""

    STAGES = {"init": "初始化", "ingest": "资料入库", "verify": "校验与修复", "index": "构建索引"}
    AGENTS = {"alias": "整理术语与别名", "content": "生成知识页面",
              "ask": "组织回答", "search": "检索相关资料", "qmd_query": "查询搜索索引",
              "fix": "修复知识页面"}
    TOOLS = {"read_file": "读取文件", "write_file": "写入文件", "list_files": "查看目录",
             "mkdir": "创建目录", "read_aliases": "读取别名表", "write_aliases": "更新别名表",
             "wiki_retrieve": "检索相关资料", "qmd_query": "查询搜索索引"}

    def __init__(self, *, live=None, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.action_started = self.started
        self.stages = dict.fromkeys(self.STAGES, "待执行")
        self.names = {}
        self.results = {}
        self.total = 0
        self.current = ""
        self.action = "准备开始"
        self.waiting = False
        self.targets = {}
        self.active_tools = {}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.lines = 0
        self.question_started_at = None
        self.model_started_at = {}
        self.tool_counts = {}
        self.live = (sys.stdout.isatty() and sys.stderr.isatty()
                     and os.environ.get("TERM") != "dumb") if live is None else live
        self.stream = sys.stderr

    @staticmethod
    def _display(value):
        return " ".join("".join(c if c.isprintable() else " " for c in str(value)).split())

    @staticmethod
    def _write(message):
        print(message, flush=True)

    def _summary(self):
        return f"成功 {sum(v == '成功' for v in self.results.values())} 篇 · 失败 {sum(v == '失败' for v in self.results.values())} 篇"

    def _panel(self):
        elapsed = int(self.clock() - self.started)
        action = self.action
        waited = int(self.clock() - self.action_started)
        wait_label = f"已等待 {waited} 秒" if waited >= 1 else "执行中"
        if self.waiting and waited >= 1:
            action += f" · {wait_label}"
        if self.question_started_at is not None:
            return [
                f"本轮问答 · 已用时 {int(self.clock() - self.question_started_at)} 秒",
                self.action,
                wait_label if self.waiting else "正在继续处理",
                self._tool_summary(),
            ]
        stages = [f"{label}（{self.stages[key]}）" for key, label in self.STAGES.items()]
        return [
            f"知识库更新 · 已用时 {elapsed} 秒",
            " → ".join(stages[:2]),
            " → ".join(stages[2:]),
            f"资料：共 {self.total} 篇 · {self._summary()}",
            f"处理中 {sum(v == '处理中' for v in self.results.values())} 篇 · 待处理 {self.total - len(self.results)} 篇",
            f"当前：{self.current or '—'}",
            f"动作：{action}",
        ]

    @staticmethod
    def _clip(text, width):
        result, used = "", 0
        for char in text:
            size = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
            if used + size > width:
                break
            result += char
            used += size
        return result

    def _clear(self):
        if self.lines:
            self.stream.write(f"\x1b[{self.lines}A\r\x1b[J")
            self.lines = 0

    def _draw(self):
        self._clear()
        width = max(1, shutil.get_terminal_size().columns - 1)
        rows = self._panel()
        self.stream.write("\n".join(self._clip(row, width) for row in rows) + "\n")
        self.stream.flush()
        self.lines = len(rows)

    def _tick(self):
        while not self.stop.wait(1):
            with self.lock:
                self._draw()

    def _emit(self, message, *, waiting=False, log=True):
        with self.lock:
            self.action = self._display(message)
            self.waiting = waiting
            self.action_started = self.clock()
            if self.thread is not None:
                self._draw()
            elif log:
                self._write(self.action)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        with self.lock:
            self._clear()
            self.stream.flush()

    def workflow_started(self):
        self.started = self.clock()
        self._emit("开始更新知识库")
        self._start_live()

    def _start_live(self):
        if self.live and self.thread is None:
            self.stop.clear()
            self.thread = threading.Thread(target=self._tick, daemon=True)
            with self.lock:
                self._draw()
            self.thread.start()

    def preparation_completed(self):
        self.close()
        if all(status == "本次跳过" for status in self.stages.values()):
            self._emit("知识库此前已完成，本次跳过，可直接提问")
            return
        self._emit(f"知识库准备完成 · {self._summary()} · 总耗时 {int(self.clock() - self.started)} 秒")

    def question_started(self):
        self.close()
        self.question_started_at = self.clock()
        self.targets.clear()
        self.active_tools.clear()
        self.tool_counts.clear()
        self.model_started_at.clear()
        self.action = "准备处理问题"
        self.waiting = False
        self._start_live()
        self._emit("开始本轮问答")

    def answer_ready(self):
        # Stop drawing before the answer or archive messages write to stdout.
        self.close()

    def question_finished(self, status="回答完成"):
        self.close()
        if self.question_started_at is None:
            return
        elapsed = self.clock() - self.question_started_at
        self.question_started_at = None
        self._emit(f"{status} · 本轮耗时 {elapsed:.1f} 秒 · {self._tool_summary()}")
        self.active_tools.clear()
        self.targets.clear()
        self.model_started_at.clear()

    def _tool_summary(self):
        labels = {"list_files": "查看目录", "read_file": "读取文件", "qmd_query": "索引查询"}
        items = [f"{label} {self.tool_counts[key]} 次" for key, label in labels.items()
                 if self.tool_counts.get(key)]
        return "已完成：" + " · ".join(items) if items else "暂无已完成的检索或文件读取"

    def model_started(self, agent):
        with self.lock:
            self.model_started_at[agent] = self.clock()
            label = self.AGENTS.get(agent, self._display(agent))
            self._emit(f"Agent 思考中 · {label}", waiting=True)

    def model_finished(self, agent):
        with self.lock:
            started = self.model_started_at.pop(agent, None)
            elapsed = self.clock() - started if started is not None else 0
            self._emit(f"{self.AGENTS.get(agent, agent)}：本轮模型已返回 · 耗时 {elapsed:.1f} 秒")

    def workflow_completed(self):
        self.close()
        self._emit("会话已结束")

    def workflow_failed(self, error=None):
        self.close()
        self._emit(f"任务已停止 · {self._summary()} · 总耗时 {int(self.clock() - self.started)} 秒"
                   + (f" · {type(error).__name__}" if error else ""))

    def stage_skipped(self, stage):
        self.stages[stage] = "本次跳过"
        self._emit(f"{self.STAGES.get(stage, stage)}：此前已完成，本次跳过")

    def stage_started(self, stage, attempt):
        self.stages[stage] = "进行中"
        self.current = ""
        # The index subprocess owns its terminal output; do not redraw over it.
        if stage == "index":
            self.close()
        label = self.STAGES.get(stage, self._display(stage))
        self._emit(f"{label}：开始" + (f"（第 {attempt} 次尝试）" if attempt > 1 else ""), waiting=True)

    def stage_succeeded(self, stage):
        self.stages[stage] = "已完成"
        self.current = ""
        self._emit(f"{self.STAGES.get(stage, stage)}：阶段完成")

    def stage_failed(self, stage, error=None):
        self.stages[stage] = "失败"
        self._emit(f"{self.STAGES.get(stage, stage)}：本次尝试失败"
                   + (f"（{type(error).__name__}）" if error else ""))

    def stage_retrying(self, stage, next_attempt):
        self.stages[stage] = "重试中"
        self._emit(f"{self.STAGES.get(stage, stage)}：正在重试，第 {next_attempt} 次尝试")

    def sources_discovered(self, sources):
        self.names.update(sources)
        self.total = len(self.names)
        self._emit(f"本次共有 {self.total} 篇资料待处理")

    def no_pending_sources(self):
        self._emit("没有待入库资料")

    def source_started(self, index, total, srcid):
        self.total = max(self.total, total)
        self.results[srcid] = "处理中"
        self.current = f"《{self._display(self.names.get(srcid, srcid))}》 · 本轮第 {index}/{total} 篇"
        self._emit(f"正在处理 {self.current} · {self._summary()}", waiting=True)

    def source_succeeded(self, index, total, srcid):
        self.results[srcid] = "成功"
        self.current = ""
        self._emit(f"《{self._display(self.names.get(srcid, srcid))}》：资料处理完成 · {self._summary()}")

    def source_failed(self, index, total, srcid):
        self.results[srcid] = "失败"
        self.current = ""
        self._emit(f"《{self._display(self.names.get(srcid, srcid))}》：资料处理失败 · {self._summary()}")

    def agent_started(self, agent):
        self._emit(f"开始{self.AGENTS.get(agent, self._display(agent))}")

    def agent_succeeded(self, agent):
        if agent == "ask":
            self.question_finished()
            return
        status = "本轮模型已返回" if agent in ("alias", "content") else "任务完成"
        self._emit(f"{self.AGENTS.get(agent, self._display(agent))}：{status}")

    def agent_failed(self, agent, error=None):
        if agent == "ask":
            self.question_finished("回答失败")
            return
        self._emit(f"{self.AGENTS.get(agent, self._display(agent))}：模型调用失败")

    def node_started(self, node):
        pass

    def node_succeeded(self, node):
        pass

    def node_failed(self, node, error=None):
        self._emit("处理步骤失败")

    def tool_context(self, call_id, path):
        with self.lock:
            self.targets[call_id] = self._display(path)

    def tool_started(self, tool_name, call_id):
        with self.lock:
            label = self.TOOLS.get(tool_name, f"执行工具 {self._display(tool_name)}")
            target = self.targets.pop(call_id, "")
            label += f"：{target}" if target else ""
            self.active_tools[call_id] = (label, self.clock())
            self._emit("正在" + label, waiting=True,
                       log=tool_name not in ("list_files", "read_file", "write_file", "mkdir", "read_aliases", "write_aliases"))

    def _tool_finished(self, tool_name, call_id, status):
        label, started = self.active_tools.pop(call_id, (self.TOOLS.get(tool_name, self._display(tool_name)), self.clock()))
        elapsed = self.clock() - started
        duration = f" · 耗时 {elapsed:.1f} 秒" if elapsed >= 1 else ""
        self._emit(f"{label}：{status}{duration}")

    def tool_succeeded(self, tool_name, call_id):
        with self.lock:
            self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1
            self._tool_finished(tool_name, call_id, "操作完成")

    def tool_failed(self, tool_name, call_id, error=None):
        with self.lock:
            self._tool_finished(tool_name, call_id, "操作失败，交回模型处理")


def report(callback: Callable[..., None], *args: object) -> None:
    try:
        callback(*args)
    except Exception:
        pass


def invoke_model(reporter, agent, model, messages):
    """Report each actual model call, including calls after tool execution."""
    report_event(reporter, "model_started", agent)
    response = model.invoke(messages)
    report_event(reporter, "model_finished", agent)
    return response


def report_event(reporter: object, event: str, *args: object) -> None:
    try:
        callback = getattr(reporter, event)
    except Exception:
        return
    report(callback, *args)
