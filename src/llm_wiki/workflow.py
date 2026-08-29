import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .wiki import wiki_init, wiki_register


class WikiState(TypedDict):
    init: bool
    ingest: bool
    verify: bool
    retrieve: bool


STATE_FILE = "wiki/wiki-state.json"


def save_state(state: WikiState) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def init(state: WikiState) -> dict[str, bool]:
    return {"init": wiki_init()}


def ingest(state: WikiState) -> dict[str, bool]:
    wiki_register()
    # if 
    #     return {"ingest": True}
    # else:
    #     return {"ingest": False}
    return {"ingest": True}


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


builder = StateGraph(WikiState)

builder.add_node("init", init)
builder.add_node("ingest", ingest)
builder.add_node("verify", verify)
builder.add_node("retrieve", retrieve)

builder.add_edge(START, "init")

builder.add_conditional_edges(
    "init",
    next_init,
    {"next": "ingest", "retry": "init"},
)
builder.add_conditional_edges(
    "ingest",
    next_ingest,
    {"next": "verify", "retry": "ingest"},
)
builder.add_conditional_edges(
    "verify",
    next_verify,
    {"next": "retrieve", "retry": "verify"},
)
builder.add_edge("retrieve", END)

wiki_graph = builder.compile()


def main() -> None:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            initial_state: WikiState = json.load(file)
    except FileNotFoundError:
        initial_state = {
            "init": False,
            "ingest": False,
            "verify": False,
            "retrieve": False,
        }
        save_state(initial_state)
    result = wiki_graph.invoke(initial_state)
    save_state(result)

if __name__ == "__main__":
    main()
