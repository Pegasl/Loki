import json
import os
import subprocess
from pathlib import Path
from typing import Callable, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt
from langgraph.checkpoint.memory import InMemorySaver

from .wiki_agent import agent_nodes

from .progress import (
    NullProgressReporter,
    ProgressReporter,
    TerminalProgressReporter,
    report,
    report_event,
)
from .wiki import wiki_index, wiki_ingest, wiki_init, wiki_register, wiki_verify


class NodeFailedError(RuntimeError):
    """A workflow node returned an unsuccessful result."""


class WikiState(TypedDict, total=False):
    init: bool
    ingest: bool
    verify: bool
    index: bool
    question: str
    messages: list
    history: list
    model_turns: int
    retrieval_attempted: bool
    retrieval_succeeded: bool
    next: str


root = Path(".").expanduser().resolve()
raw = root / "raw"
wiki = root / "wiki"
log = wiki / "log.md"
aliases = wiki / "aliases.json"
sources = wiki / "sources.json"
state_file = wiki / "state.json"


def save_state(state: WikiState) -> None:
    with open(state_file, "w", encoding="utf-8") as file:
        json.dump({stage: state.get(stage, False) for stage in ("init", "ingest", "verify", "index")},
                  file, ensure_ascii=False, indent=2)


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
    report_event(progress, "sources_discovered", {
        registration["srcid"]: registration.get("filename") or registration["srcid"]
        for registration in registrations if registration.get("srcid") in pending_srcids
    })
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


def verify(
    state: WikiState, reporter: ProgressReporter | None = None
) -> dict[str, bool]:
    return {"verify": wiki_verify(fix=True, reporter=reporter)}


def index(state: WikiState) -> dict[str, bool]:
    if not wiki_index():
        return {"index": False}
    model_dir = Path.home() / ".cache" / "qmd" / "models"
    models = {
        "embed": str(model_dir / "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        "generate": str(model_dir / "qmd-query-expansion-1.7B-q4_k_m.gguf"),
        "rerank": str(model_dir / "qwen3-reranker-0.6b-q8_0.gguf"),
    }
    # QMD resolves its project from PWD, which cwd alone does not update.
    qmd_env = {
        **os.environ,
        "PWD": str(root),
        **{f"QMD_{role.upper()}_MODEL": path for role, path in models.items()},
    }
    qmd = root / ".qmd"
    config_path = qmd / "index.yaml"
    if not config_path.is_file():
        config_path = qmd / "index.yml"
    if not config_path.is_file() or not (qmd / "index.sqlite").is_file():
        subprocess.run(["qmd", "init"], cwd=root, env=qmd_env, check=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    collections = {
        name: {"path": name, "pattern": "**/*.{md,txt}", "includeByDefault": True}
        for name in ("raw", "wiki")
    }
    if config.get("collections") != collections or config.get("models") != models:
        config["collections"] = collections
        config["models"] = models
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    subprocess.run(["qmd", "update"], cwd=root, env=qmd_env, check=True)
    subprocess.run(["qmd", "embed"], cwd=root, env=qmd_env, check=True)
    return {"index": True}


def ask(state: WikiState) -> dict:
    question = interrupt("请输入问题（退出 / exit / quit 结束）：")
    question = question.strip()
    if question.lower() in {"退出", "exit", "quit"}:
        return {"next": END}
    if not question:
        return {"next": "ask"}
    return {"question": question, "messages": [], "model_turns": 0,
            "retrieval_attempted": False, "retrieval_succeeded": False, "next": "agent"}


def build_graph(reporter: ProgressReporter | None = None, *, checkpointer=None,
                output=print, max_turns: int = 30):
    progress = reporter or NullProgressReporter()
    retry_policy = RetryPolicy(max_attempts=6, retry_on=NodeFailedError)

    def wrap_stage(stage: str, function: Callable[[WikiState], dict[str, bool]]):
        def run(state: WikiState, runtime: Runtime) -> dict[str, bool]:
            if state.get(stage) is True:
                report_event(progress, "stage_skipped", stage)
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

    def run_verify(state: WikiState) -> dict[str, bool]:
        return verify(state, progress)

    builder = StateGraph(WikiState)
    builder.add_node("init", wrap_stage("init", init), retry_policy=retry_policy)
    builder.add_node("ingest", wrap_stage("ingest", run_ingest), retry_policy=retry_policy)
    builder.add_node("verify", wrap_stage("verify", run_verify),
                     retry_policy=RetryPolicy(max_attempts=1))
    builder.add_node("index", wrap_stage("index", index), retry_policy=retry_policy)
    agent, tools = agent_nodes(root, progress, output, max_turns=max_turns)
    builder.add_node("ask", ask)
    builder.add_node("agent", agent)
    builder.add_node("tools", tools)

    builder.add_edge(START, "init")
    builder.add_edge("init", "ingest")
    builder.add_edge("ingest", "verify")
    builder.add_edge("verify", "index")
    builder.add_edge("index", "ask")
    builder.add_conditional_edges("ask", lambda state: state["next"], ["ask", "agent", END])
    builder.add_conditional_edges("agent", lambda state: state["next"], ["ask", "agent", "tools"])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


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
            "index": False,
        }
        wiki.mkdir(exist_ok=True)
        state_file.write_text(json.dumps(initial_state), encoding="utf-8")

    reporter = TerminalProgressReporter()
    report(reporter.workflow_started)
    config = {"configurable": {"thread_id": "wiki-cli"}, "recursion_limit": 100}
    try:
        graph = build_graph(reporter, checkpointer=InMemorySaver())
        result = graph.invoke(initial_state, config)
    except BaseException as error:
        report_event(reporter, "workflow_failed", error)
        raise
    report_event(reporter, "preparation_completed")
    while result.get("__interrupt__"):
        try:
            question = input(result["__interrupt__"][0].value)
        except (EOFError, KeyboardInterrupt):
            print()
            question = "exit"
        try:
            result = graph.invoke(Command(resume=question), config)
        except KeyboardInterrupt:
            print("\n已退出。")
            break
    save_state(result)
    report(reporter.workflow_completed)


if __name__ == "__main__":
    main()
