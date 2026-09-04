from typing import Callable, Protocol


class ProgressReporter(Protocol):
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
    @staticmethod
    def _display(value: object) -> str:
        return " ".join(str(value).splitlines())

    @staticmethod
    def _write(message: str) -> None:
        print(message, flush=True)

    def workflow_started(self) -> None:
        self._write("[workflow] START")

    def workflow_completed(self) -> None:
        self._write("[workflow] COMPLETE")

    def stage_started(self, stage: str, attempt: int) -> None:
        self._write(f"[{self._display(stage)}] START attempt={attempt}")

    def stage_succeeded(self, stage: str) -> None:
        self._write(f"[{self._display(stage)}] OK")

    def stage_failed(self, stage: str, error: BaseException | None = None) -> None:
        message = f"[{self._display(stage)}] FAILED"
        if error is not None:
            message += f" error={self._display(error)}"
        self._write(message)

    def stage_retrying(self, stage: str, next_attempt: int) -> None:
        self._write(f"[{self._display(stage)}] RETRY attempt={next_attempt}")

    def no_pending_sources(self) -> None:
        self._write("[ingest] NO PENDING SOURCES")

    def source_started(self, index: int, total: int, srcid: str) -> None:
        self._source_status(index, total, srcid, "START")

    def source_succeeded(self, index: int, total: int, srcid: str) -> None:
        self._source_status(index, total, srcid, "OK")

    def source_failed(self, index: int, total: int, srcid: str) -> None:
        self._source_status(index, total, srcid, "FAILED")

    def _source_status(self, index: int, total: int, srcid: str, status: str) -> None:
        self._write(
            f"[ingest] SOURCE {index}/{total} {status} srcid={self._display(srcid)}"
        )

    def agent_started(self, agent: str) -> None:
        self._write(f"[agent:{self._display(agent)}] START")

    def agent_succeeded(self, agent: str) -> None:
        self._write(f"[agent:{self._display(agent)}] OK")

    def agent_failed(self, agent: str, error: BaseException | None = None) -> None:
        self._write(f"[agent:{self._display(agent)}] FAILED")

    def node_started(self, node: str) -> None:
        self._write(f"[node:{self._display(node)}] START")

    def node_succeeded(self, node: str) -> None:
        self._write(f"[node:{self._display(node)}] OK")

    def node_failed(self, node: str, error: BaseException | None = None) -> None:
        self._write(f"[node:{self._display(node)}] FAILED")

    def tool_started(self, tool_name: str, call_id: str) -> None:
        self._tool_status(tool_name, call_id, "START")

    def tool_succeeded(self, tool_name: str, call_id: str) -> None:
        self._tool_status(tool_name, call_id, "OK")

    def tool_failed(
        self,
        tool_name: str,
        call_id: str,
        error: BaseException | None = None,
    ) -> None:
        self._tool_status(tool_name, call_id, "FAILED")

    def _tool_status(self, tool_name: str, call_id: str, status: str) -> None:
        self._write(
            f"[tool] {status} name={self._display(tool_name)} "
            f"call_id={self._display(call_id)}"
        )


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
