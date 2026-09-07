import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from llm_wiki import workflow
from llm_wiki.wiki_agent import archive_qa, file_tools


ENV = {"MODEL_NAME": "test", "OPENAI_API_KEY": "test", "OPENAI_BASE_URL": "http://localhost"}
COMPLETE = {name: True for name in ("init", "ingest", "verify", "index")}


def final(answer="普通答案", question=None, related_answer=None):
    return AIMessage(content=json.dumps({"answer": answer, "wiki_qa": None if question is None else
        {"question": question, "answer": related_answer or answer}}, ensure_ascii=False))


def call(name, args, identifier="call"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": identifier}])


class WikiAgentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        (self.root / "wiki").mkdir()
        (self.root / "raw").mkdir()
        (self.root / "wiki/index.md").write_text("- paper: strain transitions", encoding="utf-8")
        (self.root / "raw/paper.txt").write_text("原始证据", encoding="utf-8")
        for patcher in (patch.object(workflow, "root", self.root), patch.dict(os.environ, ENV)):
            patcher.start()
            self.addCleanup(patcher.stop)
        chat = patch("langchain_openai.ChatOpenAI")
        self.model = chat.start().return_value.bind_tools.return_value
        self.addCleanup(chat.stop)
        retrieve = patch("llm_wiki.wiki_query._retrieve_report", return_value="证据 [source](wiki/paper.md)")
        self.retrieve = retrieve.start()
        self.addCleanup(retrieve.stop)
        self.output = Mock()
        self.config = {"configurable": {"thread_id": "test"}, "recursion_limit": 100}

    def graph(self, max_turns=30):
        graph = workflow.build_graph(checkpointer=InMemorySaver(), output=self.output, max_turns=max_turns)
        result = graph.invoke(COMPLETE, self.config)
        self.assertTrue(result["__interrupt__"])
        return graph

    def resume(self, graph, question):
        return graph.invoke(Command(resume=question), self.config)

    def archives(self):
        return list((self.root / "wiki/query").glob("*.md"))

    def test_multiturn_related_mixed_and_unrelated(self):
        answer = "应变影响稳定性。[来源](wiki/paper.md)"
        self.model.invoke.side_effect = [
            call("wiki_retrieve", {"question": "应变有什么作用？"}),
            final(answer, "应变有什么作用？"),
            call("wiki_retrieve", {"question": "应变如何影响稳定性？"}),
            final(answer + " 另一个无关回答。", "应变如何影响稳定性？", answer),
            final("你好！"),
        ]
        graph = self.graph()
        self.resume(graph, "应变有什么作用？")
        self.resume(graph, "为什么？顺便打个招呼。")
        result = self.resume(graph, "你好")
        self.assertEqual(len(self.archives()), 2)
        for path in self.archives():
            content = path.read_text()
            self.assertIn(answer, content)
            self.assertNotIn("无关回答", content)
        messages = self.model.invoke.call_args_list[2].args[0]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIn("strain transitions", messages[0].content)
        self.assertEqual(messages[1].content, "应变有什么作用？")
        self.assertEqual(messages[2].content, answer)
        self.assertEqual(len(result["history"]), 6)
        self.assertEqual(self.retrieve.call_count, 2)
        self.assertEqual(self.retrieve.call_args.kwargs["root"], self.root)
        self.assertNotIn("__interrupt__", self.resume(graph, "exit"))

    def test_empty_input_and_all_exit_commands(self):
        for command in ("退出", " EXIT ", "quit"):
            graph = self.graph()
            self.assertTrue(self.resume(graph, "   ")["__interrupt__"])
            self.assertNotIn("__interrupt__", self.resume(graph, command))
        self.model.invoke.assert_not_called()
        self.assertFalse(self.archives())

    def test_repair_missing_retrieval_and_invalid_json(self):
        self.model.invoke.side_effect = [final("答案", "应变？"),
            call("wiki_retrieve", {"question": "应变？"}), AIMessage(content="invalid"), final("答案", "应变？")]
        self.resume(self.graph(), "应变？")
        self.retrieve.assert_called_once()
        self.assertEqual(len(self.archives()), 1)

    def test_failed_retrieval_does_not_archive_and_next_question_works(self):
        self.retrieve.side_effect = RuntimeError("retrieval unavailable")
        self.model.invoke.side_effect = [call("wiki_retrieve", {"question": "应变？"}),
            final("检索失败，无法取得证据。", "应变？"), final("你好")]
        graph = self.graph()
        self.resume(graph, "应变？")
        self.assertFalse(self.archives())
        result = self.resume(graph, "你好")
        self.assertEqual(result["history"][-1].content, "你好")
        tool_message = self.model.invoke.call_args_list[1].args[0][-1]
        self.assertEqual(tool_message.status, "error")

    def test_turn_limit_and_model_error_recover(self):
        self.model.invoke.side_effect = [AIMessage(content="invalid"), AIMessage(content="invalid"),
                                         RuntimeError("model offline"), final("你好")]
        graph = self.graph(max_turns=2)
        for question in ("first", "second", "third"):
            self.assertTrue(self.resume(graph, question)["__interrupt__"])
        self.assertFalse(self.archives())
        text = "\n".join(c.args[0] for c in self.output.call_args_list)
        self.assertIn("exceeded 2", text)
        self.assertIn("model offline", text)
        self.assertIn("你好", text)

    def test_file_calls_execute_and_errors_return_to_agent(self):
        self.model.invoke.side_effect = [
            call("wiki_retrieve", {"question": "应变？"}),
            call("read_file", {"path": "raw/paper.txt"}),
            call("write_file", {"path": "wiki/index.md", "content": "forbidden"}),
            call("write_file", {"path": "wiki/query/note.md", "content": "相关笔记"}),
            final("已保存笔记。", "记录应变笔记"),
        ]
        self.resume(self.graph(), "记录应变笔记")
        self.assertEqual((self.root / "wiki/query/note.md").read_text(), "相关笔记")
        self.assertIn("strain transitions", (self.root / "wiki/index.md").read_text())
        self.assertEqual(self.model.invoke.call_args_list[2].args[0][-1].content, "原始证据")
        self.assertEqual(self.model.invoke.call_args_list[3].args[0][-1].status, "error")

    def test_each_resume_gets_fresh_step_budget_and_index(self):
        self.model.invoke.return_value = final("你好")
        graph = self.graph()
        for number in range(40):
            (self.root / "wiki/index.md").write_text(f"index version {number}")
            self.assertTrue(self.resume(graph, "你好")["__interrupt__"])
            self.assertIn(f"index version {number}", self.model.invoke.call_args.args[0][0].content)
        self.assertEqual(self.model.invoke.call_count, 40)

    def test_missing_index_recovers_after_file_restored(self):
        graph = self.graph()
        (self.root / "wiki/index.md").unlink()
        self.resume(graph, "应变？")
        self.model.invoke.assert_not_called()
        (self.root / "wiki/index.md").write_text("restored")
        self.model.invoke.return_value = final("你好")
        self.resume(graph, "你好")
        self.model.invoke.assert_called_once()

    def test_archive_failure_preserves_answer(self):
        self.model.invoke.side_effect = [call("wiki_retrieve", {"question": "应变？"}), final("答案", "应变？")]
        with patch("llm_wiki.wiki_agent.archive_qa", side_effect=OSError("disk full")):
            result = self.resume(self.graph(), "应变？")
        self.assertEqual(result["history"][-1].content, "答案")
        self.assertIn("问答保存失败", self.output.call_args_list[-1].args[0])

    def test_file_tools_and_boundaries(self):
        read, write = file_tools(self.root)
        self.assertEqual(read.invoke({"path": "raw/paper.txt"}), "原始证据")
        write.invoke({"path": "wiki/query/nested/answer.md", "content": "first"})
        write.invoke({"path": "wiki/query/nested/answer.md", "content": "second"})
        self.assertEqual(read.invoke({"path": "wiki/query/nested/answer.md"}), "second")
        for path in ("../secret", ".env", "/etc/passwd", "wiki/../.env"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                read.invoke({"path": path})
        for path in ("wiki/index.md", "raw/paper.md", "wiki/query/../index.md", "wiki/query/x.txt",
                     str(self.root / "wiki/query/x.md"), "wiki/query-other/x.md"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                write.invoke({"path": path, "content": "bad"})
        (self.root / "wiki/query/link").symlink_to(self.root / "raw", target_is_directory=True)
        (self.root / "wiki/link.txt").symlink_to(self.root / "raw/paper.txt")
        with self.assertRaises(ValueError):
            write.invoke({"path": "wiki/query/link/x.md", "content": "bad"})
        with self.assertRaises(ValueError):
            read.invoke({"path": "wiki/link.txt"})
        with self.assertRaises(FileNotFoundError):
            read.invoke({"path": "wiki/missing.md"})
        (self.root / "raw/binary").write_bytes(b"\xff")
        with self.assertRaises(UnicodeError):
            read.invoke({"path": "raw/binary"})
        os.link(self.root / "raw/paper.txt", self.root / "wiki/query/hard.md")
        with self.assertRaises(ValueError):
            write.invoke({"path": "wiki/query/hard.md", "content": "bad"})
        self.assertEqual((self.root / "raw/paper.txt").read_text(), "原始证据")

    def test_archive_unique_and_query_root_symlink_rejected(self):
        qa = {"question": "应变？", "answer": "答案"}
        first = archive_qa(self.root, qa)
        second = archive_qa(self.root, qa)
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.archives()), 2)
        for path in self.archives():
            path.unlink()
        (self.root / "wiki/query").rmdir()
        (self.root / "wiki/query").symlink_to(self.root / "raw", target_is_directory=True)
        with self.assertRaises(ValueError):
            archive_qa(self.root, qa)

    def test_main_handles_eof_and_keyboard_interrupt(self):
        for error in (EOFError, KeyboardInterrupt):
            with self.subTest(error=error), patch.object(workflow, "state_file", self.root / "wiki/state.json"), \
                 patch.object(workflow, "save_state") as save, patch("builtins.input", side_effect=error), \
                 patch("builtins.print"):
                (self.root / "wiki/state.json").write_text(json.dumps({**COMPLETE, "retrieve": True}))
                workflow.main()
                save.assert_called_once()
                self.assertNotIn("__interrupt__", save.call_args.args[0])
