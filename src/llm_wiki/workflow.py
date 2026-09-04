import json
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .progress import (
    NullProgressReporter,
    ProgressReporter,
    TerminalProgressReporter,
    report,
)
from .wiki import wiki_ingest, wiki_init, wiki_register


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
    # if
    #     return {"verify": True}
    # else:
    #     return {"verify": False}
    return {"verify": True}


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


def next_init(state: WikiState) -> str:
    if state["init"]:
        save_state(state)
        return "next"
    return "retry"


def next_ingest(state: WikiState) -> str:
    if state["ingest"]:
        save_state(state)
        return "next"
    return "retry"


def next_verify(state: WikiState) -> str:
    if state["verify"]:
        save_state(state)
        return "next"
    return "retry"


def next_retrieve(state: WikiState) -> str:
    if state["retrieve"]:
        save_state(state)
        return "next"
    return "retry"


def build_graph(reporter: ProgressReporter | None = None):
    progress = reporter or NullProgressReporter()
    attempts = {"init": 0, "ingest": 0, "verify": 0, "retrieve": 0}

    def wrap_stage(stage: str, function: Callable[[WikiState], dict[str, bool]]):
        def run(state: WikiState) -> dict[str, bool]:
            attempts[stage] += 1
            report(progress.stage_started, stage, attempts[stage])
            try:
                result = function(state)
            except Exception as error:
                report(progress.stage_failed, stage, error)
                raise
            if result.get(stage) is True:
                report(progress.stage_succeeded, stage)
            else:
                report(progress.stage_failed, stage)
            return result

        return run

    def run_ingest(state: WikiState) -> dict[str, bool]:
        return ingest(state, progress)

    def route(stage: str, function: Callable[[WikiState], str]):
        def choose(state: WikiState) -> str:
            destination = function(state)
            if destination == "retry":
                report(progress.stage_retrying, stage, attempts[stage] + 1)
            return destination

        return choose

    builder = StateGraph(WikiState)
    builder.add_node("init", wrap_stage("init", init))
    builder.add_node("ingest", wrap_stage("ingest", run_ingest))
    builder.add_node("verify", wrap_stage("verify", verify))
    builder.add_node("retrieve", wrap_stage("retrieve", retrieve))

    builder.add_edge(START, "init")
    builder.add_conditional_edges(
        "init",
        route("init", next_init),
        {"next": "ingest", "retry": "init"},
    )
    builder.add_conditional_edges(
        "ingest",
        route("ingest", next_ingest),
        {"next": "verify", "retry": "ingest"},
    )
    builder.add_conditional_edges(
        "verify",
        route("verify", next_verify),
        {"next": "retrieve", "retry": "verify"},
    )
    builder.add_conditional_edges(
        "retrieve",
        route("retrieve", next_retrieve),
        {"next": END, "retry": "retrieve"},
    )
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
