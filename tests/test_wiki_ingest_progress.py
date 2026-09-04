import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

from llm_wiki.progress import NullProgressReporter, TerminalProgressReporter
from llm_wiki.wiki import wiki_ingest


class RecordingReporter(NullProgressReporter):
    def __init__(self) -> None:
        self.events = []

    def agent_started(self, agent: str) -> None:
        self.events.append(("agent", agent, "START"))

    def agent_succeeded(self, agent: str) -> None:
        self.events.append(("agent", agent, "OK"))

    def agent_failed(self, agent: str, error=None) -> None:
        self.events.append(("agent", agent, "FAILED"))

    def node_started(self, node: str) -> None:
        self.events.append(("node", node, "START"))

    def node_succeeded(self, node: str) -> None:
        self.events.append(("node", node, "OK"))

    def tool_started(self, tool_name: str, call_id: str) -> None:
        self.events.append(("tool", tool_name, call_id, "START"))

    def tool_succeeded(self, tool_name: str, call_id: str) -> None:
        self.events.append(("tool", tool_name, call_id, "OK"))

    def tool_failed(self, tool_name: str, call_id: str, error=None) -> None:
        self.events.append(("tool", tool_name, call_id, "FAILED"))


class FakeBoundModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages):
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeChatModel:
    def __init__(self, **kwargs) -> None:
        self.bind_count = 0

    def bind_tools(self, tools):
        self.bind_count += 1
        if self.bind_count == 1:
            return FakeBoundModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_aliases",
                                "args": {},
                                "id": "alias-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="aliases complete"),
                ]
            )
        return FakeBoundModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "path": "concepts/topic.md",
                                "content": "---\ntype: Concept\ndescription: Topic\n---\n",
                            },
                            "id": "content-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="content complete"),
            ]
        )


class FailingChatModel(FakeChatModel):
    def bind_tools(self, tools):
        self.bind_count += 1
        if self.bind_count == 1:
            return FakeBoundModel([RuntimeError("secret model response")])
        return FakeBoundModel([])


class UnknownToolChatModel(FakeChatModel):
    def bind_tools(self, tools):
        self.bind_count += 1
        if self.bind_count == 1:
            return FakeBoundModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "unknown_tool",
                                "args": {},
                                "id": "unknown-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            )
        return FakeBoundModel([])


class LegacyReporter:
    def source_started(self, index: int, total: int, srcid: str) -> None:
        pass


class WikiIngestProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "raw").mkdir()
        document = self.root / "wiki" / "source"
        document.mkdir(parents=True)
        (self.root / "wiki" / "workflow-log").mkdir()
        (self.root / "raw" / "source.txt").write_text("source text", encoding="utf-8")
        (self.root / "wiki" / "aliases.json").write_text(
            '{"version": 1, "entries": []}', encoding="utf-8"
        )
        (self.root / "wiki" / "sources.json").write_text(
            json.dumps(
                [
                    {
                        "srcid": "srcid-test",
                        "filename": "source.txt",
                        "sha256": "hash",
                        "ingest": False,
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.cwd_patch = patch("pathlib.Path.cwd", return_value=self.root)
        self.cwd_patch.start()
        self.env_patch = patch.dict(
            os.environ,
            {
                "MODEL_NAME": "fake",
                "OPENAI_API_KEY": "fake",
                "OPENAI_BASE_URL": "https://example.invalid",
            },
        )
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.cwd_patch.stop()
        self.temp_dir.cleanup()

    def test_reports_agent_node_and_tool_events_in_execution_order(self) -> None:
        reporter = RecordingReporter()

        with patch("langchain_openai.ChatOpenAI", FakeChatModel):
            result = wiki_ingest("srcid-test", reporter=reporter)

        self.assertTrue(result)
        self.assertEqual(
            reporter.events,
            [
                ("agent", "alias", "START"),
                ("agent", "alias", "OK"),
                ("tool", "read_aliases", "alias-call", "START"),
                ("tool", "read_aliases", "alias-call", "OK"),
                ("agent", "alias", "START"),
                ("agent", "alias", "OK"),
                ("node", "prepare_content", "START"),
                ("node", "prepare_content", "OK"),
                ("agent", "content", "START"),
                ("agent", "content", "OK"),
                ("tool", "write_file", "content-call", "START"),
                ("tool", "write_file", "content-call", "OK"),
                ("agent", "content", "START"),
                ("agent", "content", "OK"),
            ],
        )

    def test_uses_collision_fallback_document(self) -> None:
        document = self.root / "wiki" / "source"
        fallback = self.root / "wiki" / "source-srcid-test"
        document.rename(fallback)
        document.mkdir()

        with patch("langchain_openai.ChatOpenAI", FakeChatModel):
            result = wiki_ingest("srcid-test")

        self.assertTrue(result)
        self.assertTrue((fallback / "concepts" / "topic.md").is_file())
        self.assertFalse((document / "concepts" / "topic.md").exists())

    def test_agent_failure_is_reported_and_preserves_bool_api(self) -> None:
        reporter = RecordingReporter()

        with patch("langchain_openai.ChatOpenAI", FailingChatModel):
            result = wiki_ingest("srcid-test", reporter=reporter)

        self.assertFalse(result)
        self.assertEqual(
            reporter.events,
            [("agent", "alias", "START"), ("agent", "alias", "FAILED")],
        )

    def test_reporter_failure_does_not_change_ingest_result(self) -> None:
        reporter = RecordingReporter()
        reporter.agent_started = lambda agent: (_ for _ in ()).throw(
            RuntimeError("reporter failed")
        )

        with patch("langchain_openai.ChatOpenAI", FakeChatModel):
            result = wiki_ingest("srcid-test", reporter=reporter)

        self.assertTrue(result)

    def test_unknown_tool_reports_started_then_failed(self) -> None:
        reporter = RecordingReporter()

        with patch("langchain_openai.ChatOpenAI", UnknownToolChatModel):
            result = wiki_ingest("srcid-test", reporter=reporter)

        self.assertFalse(result)
        self.assertEqual(
            reporter.events[-2:],
            [
                ("tool", "unknown_tool", "unknown-call", "START"),
                ("tool", "unknown_tool", "unknown-call", "FAILED"),
            ],
        )

    def test_legacy_reporter_without_inner_methods_remains_compatible(self) -> None:
        with patch("langchain_openai.ChatOpenAI", FakeChatModel):
            result = wiki_ingest("srcid-test", reporter=LegacyReporter())

        self.assertTrue(result)

    def test_terminal_inner_failures_do_not_print_sensitive_error_details(self) -> None:
        reporter = TerminalProgressReporter()

        with patch("builtins.print") as print_mock:
            reporter.agent_failed("alias", RuntimeError("secret prompt"))
            reporter.tool_failed(
                "write_file", "call-1", RuntimeError("secret tool args")
            )

        output = "\n".join(str(item) for item in print_mock.call_args_list)
        self.assertIn("[agent:alias] FAILED", output)
        self.assertIn("[tool] FAILED name=write_file call_id=call-1", output)
        self.assertNotIn("secret prompt", output)
        self.assertNotIn("secret tool args", output)


if __name__ == "__main__":
    unittest.main()
