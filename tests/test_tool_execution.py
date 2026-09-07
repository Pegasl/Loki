import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from llm_wiki.progress import ProgressReporter
from llm_wiki.tool_execution import wiki_tool_node


class ToolExecutionTests(unittest.TestCase):
    def test_batch_errors_preserve_results_and_sequential_filesystem_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"

            @tool
            def write_value(value: str) -> str:
                """Replace the stored value."""
                path.write_text(value)
                return value

            @tool
            def read_value() -> str:
                """Read the stored value."""
                return path.read_text()

            reporter = Mock(spec=ProgressReporter)
            builder = StateGraph(MessagesState)
            builder.add_node("tools", wiki_tool_node([write_value, read_value], reporter))
            builder.add_edge(START, "tools")
            builder.add_edge("tools", END)
            calls = [
                ("unknown", {}, "unknown"),
                ("write_value", {}, "missing-argument"),
                ("read_value", {}, "missing-file"),
                ("write_value", {"value": "first"}, "write-first"),
                ("read_value", {}, "read-first"),
                ("write_value", {"value": "second"}, "write-second"),
                ("read_value", {}, "read-second"),
            ]
            result = builder.compile().invoke({"messages": [AIMessage(
                content="", tool_calls=[
                    {"name": name, "args": args, "id": call_id, "type": "tool_call"}
                    for name, args, call_id in calls
                ],
            )]}, config={"max_concurrency": 8})
            messages = result["messages"][1:]
            self.assertEqual([m.tool_call_id for m in messages], [c[2] for c in calls])
            self.assertEqual([m.status for m in messages], ["error"] * 3 + ["success"] * 4)
            self.assertEqual(messages[4].content, "first")
            self.assertEqual(messages[6].content, "second")
            expected = []
            for index, (name, _, call_id) in enumerate(calls):
                expected.extend([
                    ("tool_started", (name, call_id)),
                    ("tool_failed" if index < 3 else "tool_succeeded", (name, call_id)),
                ])
            self.assertEqual(
                [(event[0], event.args[:2]) for event in reporter.mock_calls], expected
            )
