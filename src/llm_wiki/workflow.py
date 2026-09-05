import json
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from .progress import (
    NullProgressReporter,
    ProgressReporter,
    TerminalProgressReporter,
    report,
)
from .wiki import wiki_ingest, wiki_init, wiki_register, wiki_verify


class NodeFailedError(RuntimeError):
    """A workflow node returned an unsuccessful result."""


class WikiState(TypedDict):
    init: bool
    ingest: bool
    verify: bool
    retrieve: bool


root = Path(".").expanduser().resolve()
raw = root / "raw"
wiki = root / "wiki"
log = wiki / "log.md"
aliases = wiki / "aliases.json"
sources = wiki / "sources.json"
state_file = wiki / "state.json"


def save_state(state: WikiState) -> None:
    with open(state_file, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def init(state: WikiState) -> dict[str, bool]:
    return {"init": wiki_init()}


def ingest(
    state: WikiState, reporter: ProgressReporter | None = None
) -> dict[str, bool]:
    progress = reporter or NullProgressReporter()
    if not wiki_register():
        return {"ingest": False}

    try:
        registrations = json.loads(sources.read_text(encoding="utf-8"))
        if not isinstance(registrations, list):
            return {"ingest": False}

        pending_srcids = []
        for registration in registrations:
            if not isinstance(registration, dict):
                return {"ingest": False}
            if registration.get("ingest") is False:
                srcid = registration.get("srcid")
                if not isinstance(srcid, str) or not srcid:
                    return {"ingest": False}
                pending_srcids.append(srcid)
    except (OSError, json.JSONDecodeError):
        return {"ingest": False}

    if not pending_srcids:
        report(progress.no_pending_sources)

    results = []
    total = len(pending_srcids)
    for index, srcid in enumerate(pending_srcids, start=1):
        report(progress.source_started, index, total, srcid)
        try:
            result = wiki_ingest(srcid, reporter=progress)
        except Exception:
            report(progress.source_failed, index, total, srcid)
            raise
        if result:
            report(progress.source_succeeded, index, total, srcid)
        else:
            report(progress.source_failed, index, total, srcid)
        results.append(result)

    return {"ingest": all(results)}


def verify(state: WikiState) -> dict[str, bool]:
    return {"verify": wiki_verify()}


def retrieve(state: WikiState) -> dict[str, bool]:
    # if
    #     return {"retrieve": True}
    # else:
    #     return {"retrieve": False}
    return {"retrieve": True}


# def llm_call(state: WikiState) -> WikiState:
#     new_state = state.copy()
#     new_state["retrieve"] = True
#     save_state(new_state)
#     return new_state


def build_graph(reporter: ProgressReporter | None = None):
    progress = reporter or NullProgressReporter()
    retry_policy = RetryPolicy(max_attempts=6, retry_on=NodeFailedError)

    def wrap_stage(stage: str, function: Callable[[WikiState], dict[str, bool]]):
        def run(state: WikiState, runtime: Runtime) -> dict[str, bool]:
            if state.get(stage) is True:
                return {}
            attempt = runtime.execution_info.node_attempt
            if attempt > 1:
                report(progress.stage_retrying, stage, attempt)
            report(progress.stage_started, stage, attempt)
            try:
                result = function(state)
            except Exception as error:
                report(progress.stage_failed, stage, error)
                raise
            if result.get(stage) is not True:
                report(progress.stage_failed, stage)
                raise NodeFailedError(f"Node {stage!r} returned an unsuccessful result")
            report(progress.stage_succeeded, stage)
            save_state({**state, **result})
            return result

        return run

    def run_ingest(state: WikiState) -> dict[str, bool]:
        return ingest(state, progress)

    builder = StateGraph(WikiState)
    builder.add_node("init", wrap_stage("init", init), retry_policy=retry_policy)
    builder.add_node("ingest", wrap_stage("ingest", run_ingest), retry_policy=retry_policy)
    builder.add_node("verify", wrap_stage("verify", verify), retry_policy=retry_policy)
    builder.add_node("retrieve", wrap_stage("retrieve", retrieve), retry_policy=retry_policy)

    builder.add_edge(START, "init")
    builder.add_edge("init", "ingest")
    builder.add_edge("ingest", "verify")
    builder.add_edge("verify", "retrieve")
    builder.add_edge("retrieve", END)
    return builder.compile()


wiki_graph = build_graph()


def main() -> None:
    try:
        with open(state_file, "r", encoding="utf-8") as file:
            initial_state: WikiState = json.load(file)
    except FileNotFoundError:
        initial_state = {
            "init": False,
            "ingest": False,
            "verify": False,
            "retrieve": False,
        }
        wiki.mkdir(exist_ok=True)
        state_file.write_text(json.dumps(initial_state), encoding="utf-8")

    reporter = TerminalProgressReporter()
    report(reporter.workflow_started)
    result = build_graph(reporter).invoke(initial_state)
    save_state(result)
    report(reporter.workflow_completed)


if __name__ == "__main__":
    main()
