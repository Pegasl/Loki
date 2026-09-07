import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

from llm_wiki.wiki_query import _search_tools, qmd_query, wiki_search


class WikiSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.wiki = self.root / "wiki"
        self.pages = {
            "index.md": "- paper: phase transitions\n",
            "paper/source-summary-123.md": "A study of strain.\n",
            "paper/concepts/index.md": "- strain.md: strain effects\n",
            "paper/concepts/strain.md": "Strain changes stability.\n",
            "paper/entities/index.md": "",
        }
        for name, content in self.pages.items():
            path = self.wiki / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def test_read_order_and_actual_source_tracking(self):
        sources = {}
        listing, read = _search_tools(self.wiki, sources)
        with self.assertRaisesRegex(ValueError, "index.md first"):
            read.invoke({"path": "paper/source-summary-123.md"})
        read.invoke({"path": "index.md"})
        self.assertIn("paper/", listing.invoke({}))
        with self.assertRaisesRegex(ValueError, "summary"):
            read.invoke({"path": "paper/concepts/index.md"})
        read.invoke({"path": "paper/source-summary-123.md"})
        with self.assertRaisesRegex(ValueError, "index.md before"):
            read.invoke({"path": "paper/concepts/strain.md"})
        read.invoke({"path": "paper/concepts/index.md"})
        page = json.loads(read.invoke({"path": "paper/concepts/strain.md"}))
        self.assertEqual(page["citation"], "S2")
        read.invoke({"path": "paper/concepts/strain.md"})
        self.assertEqual(len(sources), 2)
        with self.assertRaises(FileNotFoundError):
            read.invoke({"path": "paper/concepts/missing.md"})
        self.assertEqual(len(sources), 2)

    def test_rejects_escape_and_symlink(self):
        _, read = _search_tools(self.wiki, {})
        (self.wiki / "alias.md").symlink_to(self.wiki / "index.md")
        for path in ("../secret.md", str(self.wiki / "index.md"), "alias.md"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                read.invoke({"path": path})

    def test_search_excludes_archived_answers(self):
        query = self.wiki / "query"
        query.mkdir()
        (query / "answer.md").write_text("Old answer")
        listing, read = _search_tools(self.wiki, {})
        read.invoke({"path": "index.md"})
        self.assertNotIn("query/", listing.invoke({}))
        with self.assertRaisesRegex(ValueError, "not primary"):
            read.invoke({"path": "query/answer.md"})

    @patch.dict(os.environ, {"MODEL_NAME": "test", "OPENAI_API_KEY": "test", "OPENAI_BASE_URL": "http://localhost"})
    @patch("langchain_openai.ChatOpenAI")
    def test_agent_report_uses_read_sources_and_repairs_bad_citation(self, chat):
        paths = ["paper/source-summary-123.md", "paper/concepts/index.md",
                 "paper/concepts/strain.md", "paper/entities/index.md"]
        chat.return_value.bind_tools.return_value.invoke.side_effect = [
            AIMessage(content="", tool_calls=[{
                "name": "read_file", "args": {"path": path}, "id": str(i),
                "type": "tool_call",
            } for i, path in enumerate(paths)]),
            AIMessage(content="Strain changes stability. [S99]"),
            AIMessage(content="Strain changes stability. [S2]"),
        ]
        report = wiki_search("What does strain do?", self.root)
        self.assertIn("Strain changes stability. [S2]", report)
        self.assertIn("wiki/paper/concepts/strain.md", report)
        self.assertNotIn("[S99]", report)
        for path, content in self.pages.items():
            self.assertEqual((self.wiki / path).read_text(), content)

    @patch.dict(os.environ, {"MODEL_NAME": "test", "OPENAI_API_KEY": "test", "OPENAI_BASE_URL": "http://localhost"})
    @patch("langchain_openai.ChatOpenAI")
    def test_no_match_and_turn_limit(self, chat):
        model = chat.return_value.bind_tools.return_value
        model.invoke.return_value = AIMessage(content="当前 Wiki 无匹配证据。")
        self.assertEqual(wiki_search("unrelated", self.root), "当前 Wiki 无匹配证据。")
        model.invoke.side_effect = lambda messages: AIMessage(content="", tool_calls=[{
            "name": "read_file", "args": {"path": "index.md"},
            "id": "repeat", "type": "tool_call",
        }])
        with self.assertRaisesRegex(RuntimeError, "exceeded 2"):
            wiki_search("strain", self.root, max_turns=2)

    def test_invalid_question_and_missing_index(self):
        with self.assertRaises(ValueError):
            wiki_search("   ", self.root)
        with self.assertRaises(FileNotFoundError):
            wiki_search("strain", self.root / "missing")


class QmdQueryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "wiki").mkdir()
        (self.root / "wiki/index.md").write_text("- paper: strain effects", encoding="utf-8")
        (self.root / "wiki/aliases.json").write_text(json.dumps({
            "entries": [{"canonical": "Transition metal dichalcogenides", "aliases": ["TMDs"]}],
        }), encoding="utf-8")
        env = patch.dict(os.environ, {"MODEL_NAME": "test", "OPENAI_API_KEY": "test",
                                      "OPENAI_BASE_URL": "http://localhost"})
        env.start()
        self.addCleanup(env.stop)
        chat = patch("langchain_openai.ChatOpenAI")
        self.model = chat.start().return_value
        self.addCleanup(chat.stop)
        run = patch("llm_wiki.wiki_query.subprocess.run")
        self.run = run.start()
        self.addCleanup(run.stop)
        self.plan = AIMessage(content=json.dumps({"queries": ["TMDs strain?", "Transition metal dichalcogenides strain?"]}))
        self.hit = {"file": "qmd://wiki/paper/concepts/strain.md", "snippet": "Strain changes stability."}
        self.run.return_value.stdout = json.dumps([self.hit])

    def response(self, sources=None):
        return AIMessage(content=json.dumps({"findings": [
            {"text": "Strain changes stability.", "sources": sources or ["S1"]}], "gaps": []}))

    def test_plan_context_multiple_queries_deduplication_and_citation_repair(self):
        self.model.invoke.side_effect = [self.plan, self.response(["S99"]), self.response()]
        report = qmd_query("TMDs?", self.root)
        context = json.loads(self.model.invoke.call_args_list[0].args[0][1].content)
        self.assertIn("TMDs", str(context["aliases"]))
        self.assertIn("strain effects", context["wiki_index"])
        self.assertEqual(self.run.call_count, 2)
        for call in self.run.call_args_list:
            self.assertEqual(call.args[0][:2], ["qmd", "query"])
            self.assertNotIn("--full", call.args[0])
            self.assertEqual(call.kwargs["cwd"], self.root.resolve())
            self.assertEqual(call.kwargs["env"]["PWD"], str(self.root.resolve()))
            self.assertTrue(call.kwargs["check"])
            self.assertNotIn("shell", call.kwargs)
        self.assertIn("Strain changes stability. [S1]", report)
        self.assertEqual(report.count("- [S1]"), 1)
        self.assertNotIn("S99", report)

    def test_empty_results_never_ask_model_to_answer(self):
        self.model.invoke.return_value = self.plan
        self.run.return_value.stdout = "[]"
        report = qmd_query("unrelated", self.root)
        self.assertIn("未找到", report)
        self.assertEqual(self.model.invoke.call_count, 1)
        self.assertNotIn("来源文件", report)

    def test_irrelevant_hits_are_not_forced_into_answer(self):
        self.model.invoke.side_effect = [self.plan, AIMessage(content='{"findings": [], "gaps": []}')]
        report = qmd_query("unrelated", self.root)
        self.assertIn("未找到", report)
        self.assertNotIn("来源文件", report)

    def test_index_only_is_not_evidence(self):
        self.model.invoke.return_value = self.plan
        self.run.return_value.stdout = json.dumps([{"file": "qmd://wiki/index.md", "snippet": "Navigation"}])
        self.assertIn("未找到", qmd_query("TMDs", self.root))
        self.assertEqual(self.model.invoke.call_count, 1)

    def test_archived_answers_are_not_evidence(self):
        self.model.invoke.return_value = self.plan
        self.run.return_value.stdout = json.dumps([
            {"file": "qmd://wiki/query/answer.md", "snippet": "Previously generated answer"},
        ])
        self.assertIn("未找到", qmd_query("TMDs", self.root))
        self.assertEqual(self.model.invoke.call_count, 1)

    def test_command_errors_propagate(self):
        self.model.invoke.return_value = self.plan
        for error in (FileNotFoundError("qmd"), subprocess.CalledProcessError(1, "qmd"),
                      subprocess.TimeoutExpired("qmd", 180)):
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                self.run.side_effect = error
                qmd_query("TMDs", self.root)

    def test_malformed_results_fail(self):
        self.model.invoke.return_value = self.plan
        for output in ("invalid", "{}", '[{"file": "qmd://wiki/a.md"}]',
                       '[{"file": "qmd://wiki/../../secret", "snippet": "x"}]'):
            self.run.return_value.stdout = output
            with self.subTest(output=output), self.assertRaises(ValueError):
                qmd_query("TMDs", self.root)

    def test_missing_metadata_fails_before_model_or_search(self):
        (self.root / "wiki/aliases.json").unlink()
        with self.assertRaises(FileNotFoundError):
            qmd_query("TMDs", self.root)
        self.model.invoke.assert_not_called()
        self.run.assert_not_called()

    def test_invalid_plan_is_bounded(self):
        self.model.invoke.return_value = AIMessage(content='{"queries": ["same", "same"]}')
        with self.assertRaisesRegex(RuntimeError, "exceeded 2"):
            qmd_query("TMDs", self.root, max_turns=2)
        self.run.assert_not_called()

    def test_invalid_arguments(self):
        for kwargs in ({"question": " "}, {"question": "x", "max_turns": True},
                       {"question": "x", "timeout": 0}, {"question": "x", "timeout": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                qmd_query(root=self.root, **kwargs)


if __name__ == "__main__":
    unittest.main()
