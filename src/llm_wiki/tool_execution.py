"""Shared ToolNode configuration for the Wiki agents."""

from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from .progress import report_event


def wiki_tool_node(tools, progress):
    """Return tool errors to the model and serialize filesystem operations."""
    def report_call(request, execute):
        name = request.tool_call["name"]
        call_id = request.tool_call["id"]
        report_event(progress, "tool_started", name, call_id)
        try:
            result = execute(request)
        except Exception as error:
            report_event(progress, "tool_failed", name, call_id, error)
            raise
        if result.status == "error":
            report_event(progress, "tool_failed", name, call_id, RuntimeError(str(result.content)))
        else:
            report_event(progress, "tool_succeeded", name, call_id)
        return result

    node = ToolNode(tools, handle_tool_errors=True, wrap_tool_call=report_call)

    def execute_tools(state: MessagesState, config: RunnableConfig):
        # Enforce this at invocation time: parent graph config overrides bindings.
        return node.invoke(state, {**config, "max_concurrency": 1})

    return execute_tools
