import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from llm_wiki.progress import ProgressReporter
from llm_wiki.wiki_fix import _fix_tools

from llm_wiki.wiki import wiki_index, wiki_register, wiki_verify


class WikiVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.raw = self.root / "raw"
        self.wiki = self.root / "wiki"
        self.log = self.wiki / "workflow-log" / "verify-log.md"
        self.raw.mkdir()
        self.wiki.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_source(
        self,
        filename: str,
        srcid: str,
        *,
        document_name: str | None = None,
    ) -> tuple[dict[str, object], Path]:
        source = self.raw / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"content for {filename}\n", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        registration = {
            "filename": filename,
            "srcid": srcid,
            "sha256": digest,
            "ingest": True,
        }
        document = self.wiki / (document_name or Path(filename).stem)
        (document / "concepts").mkdir(parents=True)
        (document / "entities").mkdir()
        token = srcid.removeprefix("srcid-")
        summary = document / f"source-summary-{token}.md"
        source_path = os.path.relpath(source, document).replace(os.sep, "/")
        summary.write_text(
            "\n".join(
                [
                    "---",
                    'type: "Source Summary"',
                    'description: "A source summary"',
                    f'source: "{source_path}"',
                    f'srcid: "{srcid}"',
                    f'source_hash: "{digest}"',
                    "---",
                    "Summary body.",
                ]
            ),
            encoding="utf-8",
        )
        (document / "concepts" / "topic.md").write_text(
            '---\ntype: "Concept"\ndescription: "A topic"\n---\nConcept body.\n',
            encoding="utf-8",
        )
        (document / "entities" / "person.md").write_text(
            '---\ntype: "Entity"\ndescription: "A person"\n---\nEntity body.\n',
            encoding="utf-8",
        )
        return registration, document

    def write_sources(self, registrations: object) -> None:
        (self.wiki / "sources.json").write_text(
            json.dumps(registrations), encoding="utf-8"
        )

    def test_valid_source_passes_and_is_logged(self) -> None:
        registration, _ = self.add_source("notes/topic.txt", "srcid-abc123")
        self.write_sources([registration])

        self.assertTrue(wiki_verify(self.root))
        log = self.log.read_text(encoding="utf-8")
        self.assertIn("verify | notes/topic.txt", log)
        self.assertIn("- Status: `passed`", log)

    def test_generated_indexes_do_not_fail_verification(self) -> None:
        registration, _ = self.add_source("topic.txt", "srcid-abc123")
        self.write_sources([registration])
        self.assertTrue(wiki_index(self.root))
        self.assertTrue(wiki_verify(self.root))

    def test_summary_registration_mismatches_are_all_logged(self) -> None:
        registration, document = self.add_source("topic.txt", "srcid-abc123")
        summary = document / "source-summary-abc123.md"
        summary.write_text(
            summary.read_text(encoding="utf-8")
            .replace('type: "Source Summary"', 'type: "Other"')
            .replace('source: "../../raw/topic.txt"', 'source: "wrong.txt"')
            .replace('srcid: "srcid-abc123"', 'srcid: "srcid-wrong"')
            .replace(
                f'source_hash: "{registration["sha256"]}"',
                'source_hash: "wrong"',
            ),
            encoding="utf-8",
        )
        self.write_sources([registration])

        self.assertFalse(wiki_verify(self.root))
        log = self.log.read_text(encoding="utf-8")
        for field in ("type", "source", "srcid", "source_hash"):
            self.assertIn(f"{field} does not match registration", log)

    def test_checks_every_markdown_and_required_page_types(self) -> None:
        registration, document = self.add_source("topic.txt", "srcid-abc123")
        (document / "concepts" / "topic.md").write_text(
            "---\ntype: \ndescription: \n---\n", encoding="utf-8"
        )
        (document / "entities" / "person.md").write_text(
            "description: absent frontmatter", encoding="utf-8"
        )
        (document / "broken.md").write_text(
            "---\ntype: Note\ndescription: open", encoding="utf-8"
        )
        self.write_sources([registration])

        self.assertFalse(wiki_verify(self.root))
        log = self.log.read_text(encoding="utf-8")
        self.assertIn("concepts/topic.md: type is empty", log)
        self.assertIn("concepts/topic.md: description is empty", log)
        self.assertIn("concepts/topic.md: body is empty", log)
        self.assertIn("entities/person.md: missing frontmatter", log)
        self.assertIn("broken.md: frontmatter is not closed", log)
        self.assertIn("no Concept page", log)
        self.assertIn("no Entity page", log)

    def test_rejects_unclosed_single_quoted_frontmatter_value(self) -> None:
        registration, document = self.add_source("topic.txt", "srcid-abc123")
        (document / "concepts" / "topic.md").write_text(
            "---\ntype: Concept\ndescription: 'not closed\n---\nBody.\n",
            encoding="utf-8",
        )
        self.write_sources([registration])

        self.assertFalse(wiki_verify(self.root))
        self.assertIn(
            "concepts/topic.md: invalid quoted value for description",
            self.log.read_text(encoding="utf-8"),
        )

    def test_missing_fallback_summary_is_not_reported_as_ambiguous(self) -> None:
        registration, document = self.add_source(
            "nested/topic.txt",
            "srcid-abc123",
            document_name="topic-srcid-abc123",
        )
        (document / "source-summary-abc123.md").unlink()
        (self.wiki / "topic").mkdir()
        self.write_sources([registration])

        self.assertFalse(wiki_verify(self.root))
        log = self.log.read_text(encoding="utf-8")
        self.assertIn("missing Source Summary", log)
        self.assertNotIn("Wiki directory is ambiguous", log)

    def test_missing_summary_and_ambiguous_document_are_logged(self) -> None:
        registration, document = self.add_source("topic.txt", "srcid-abc123")
        (document / "source-summary-abc123.md").unlink()
        self.write_sources([registration])

        self.assertFalse(wiki_verify(self.root))
        self.assertIn("missing Source Summary", self.log.read_text(encoding="utf-8"))

        fallback = self.wiki / "topic-srcid-abc123"
        fallback.mkdir()
        (document / "source-summary-abc123.md").write_text("summary", encoding="utf-8")
        (fallback / "source-summary-abc123.md").write_text("summary", encoding="utf-8")

        self.assertFalse(wiki_verify(self.root))
        self.assertIn(
            "multiple Wiki directories match",
            self.log.read_text(encoding="utf-8"),
        )

    def test_continues_after_failure_and_logs_later_success(self) -> None:
        first, first_document = self.add_source("first.txt", "srcid-111111")
        second, _ = self.add_source("second.txt", "srcid-222222")
        (first_document / "concepts" / "topic.md").unlink()
        self.write_sources([first, second])

        self.assertFalse(wiki_verify(self.root))
        log = self.log.read_text(encoding="utf-8")
        self.assertIn("verify | first.txt", log)
        self.assertIn("verify | second.txt", log)
        second_event = log.split("verify | second.txt", 1)[1]
        self.assertIn("- Status: `passed`", second_event)

    def test_reports_invalid_registrations_and_unregistered_raw(self) -> None:
        (self.raw / "orphan.md").write_text("orphan", encoding="utf-8")
        self.write_sources(["invalid", {"filename": "missing.txt"}])

        self.assertFalse(wiki_verify(self.root))
        log = self.log.read_text(encoding="utf-8")
        self.assertIn("source registration must be an object", log)
        self.assertIn("registration srcid must be", log)
        self.assertIn("orphan.md", log)
        self.assertIn("raw file is not registered", log)

    def test_reports_malformed_and_non_list_sources_json(self) -> None:
        sources = self.wiki / "sources.json"
        for content in ("not json", "{}"):
            with self.subTest(content=content):
                sources.write_text(content, encoding="utf-8")
                self.assertFalse(wiki_verify(self.root))
        log = self.log.read_text(encoding="utf-8")
        self.assertIn("Expecting value", log)
        self.assertIn("must contain a JSON list", log)

    def test_validates_raw_path_and_digest(self) -> None:
        registration, _ = self.add_source("topic.txt", "srcid-abc123")
        registration["sha256"] = "0" * 64
        escaped = {
            "filename": "../outside.txt",
            "srcid": "srcid-escaped",
            "sha256": "0" * 64,
        }
        self.write_sources([registration, escaped])

        self.assertFalse(wiki_verify(self.root))
        log = self.log.read_text(encoding="utf-8")
        self.assertIn("raw SHA256 does not match", log)
        self.assertIn("escapes raw directory", log)

    def test_uses_collision_fallback_directory(self) -> None:
        registration, _ = self.add_source(
            "nested/topic.txt",
            "srcid-abc123",
            document_name="topic-srcid-abc123",
        )
        (self.wiki / "topic").mkdir()
        self.write_sources([registration])

        self.assertTrue(wiki_verify(self.root))

    def prepare_repair(self):
        registration, document = self.add_source("topic.txt", "srcid-abc123")
        self.write_sources([registration])
        page = document / "concepts" / "topic.md"
        original = page.read_text(encoding="utf-8")
        page.write_text(original.replace('description: "A topic"', 'description: ""'), encoding="utf-8")
        return registration, document, page, original

    def mock_model(self, responses):
        model = Mock()
        model.bind_tools.return_value.invoke.side_effect = responses
        self.enterContext(patch.dict(os.environ, {
            "MODEL_NAME": "fake", "OPENAI_API_KEY": "fake",
            "OPENAI_BASE_URL": "https://example.invalid",
        }))
        constructor = self.enterContext(patch("langchain_openai.ChatOpenAI", return_value=model))
        return model.bind_tools.return_value, constructor

    @staticmethod
    def write_response(path, content):
        return AIMessage(content="", tool_calls=[{
            "name": "write_file", "args": {"path": path, "content": content},
            "id": "fix-write", "type": "tool_call",
        }])

    def test_fix_repairs_page_then_rechecks_and_records_progress(self) -> None:
        _, _, page, original = self.prepare_repair()
        reporter = Mock(spec=ProgressReporter)
        model, constructor = self.mock_model([
            self.write_response("concepts/topic.md", original), AIMessage(content="done"),
        ])
        self.assertTrue(wiki_verify(self.root, fix=True, reporter=reporter))
        self.assertEqual(page.read_text(), original)
        self.assertEqual(model.invoke.call_count, 2)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        reporter.agent_started.assert_called_once_with("fix")
        reporter.agent_succeeded.assert_called_once_with("fix")
        reporter.tool_succeeded.assert_called_once_with("write_file", "fix-write")
        log = self.log.read_text()
        self.assertIn("concepts/topic.md: description is empty", log)
        self.assertIn("fix | topic.txt", log)
        self.assertIn("Written: `concepts/topic.md`", log)
        self.assertIn("recheck | topic.txt", log)
        self.assertIn("Status: `passed`", log.split("recheck |", 1)[1])
        prompt = json.loads(model.invoke.call_args.args[0][1].content)
        self.assertEqual(prompt["issues"][0]["file"], "concepts/topic.md")
        self.assertEqual(prompt["original_source"], "content for topic.txt\n")

    def test_agent_claiming_success_does_not_pass_or_retry(self) -> None:
        self.prepare_repair()
        model, _ = self.mock_model([AIMessage(content="Everything is fixed")])
        self.assertFalse(wiki_verify(self.root, fix=True))
        self.assertEqual(model.invoke.call_count, 1)
        self.assertIn("Status: `failed`", self.log.read_text().split("recheck |", 1)[1])

    def test_default_verify_does_not_call_fix(self) -> None:
        self.prepare_repair()
        with patch("llm_wiki.wiki_fix.fix_source") as fix:
            self.assertFalse(wiki_verify(self.root))
        fix.assert_not_called()

    def test_valid_source_does_not_call_fix(self) -> None:
        _, _, page, original = self.prepare_repair()
        page.write_text(original)
        with patch("llm_wiki.wiki_fix.fix_source") as fix:
            self.assertTrue(wiki_verify(self.root, fix=True))
        fix.assert_not_called()

    def test_changed_raw_blocks_repair_even_with_page_errors(self) -> None:
        self.prepare_repair()
        (self.raw / "topic.txt").write_text("changed source")
        with patch("llm_wiki.wiki_fix.fix_source") as fix:
            self.assertFalse(wiki_verify(self.root, fix=True))
        fix.assert_not_called()

    def test_missing_raw_and_broken_registry_do_not_call_fix(self) -> None:
        self.prepare_repair()
        (self.raw / "topic.txt").unlink()
        with patch("llm_wiki.wiki_fix.fix_source") as fix:
            self.assertFalse(wiki_verify(self.root, fix=True))
            (self.wiki / "sources.json").write_text("not JSON")
            self.assertFalse(wiki_verify(self.root, fix=True))
        fix.assert_not_called()

    def test_ambiguous_directory_and_duplicate_registration_do_not_call_fix(self) -> None:
        registration, document, _, _ = self.prepare_repair()
        self.write_sources([registration, registration])
        with patch("llm_wiki.wiki_fix.fix_source") as fix:
            self.assertFalse(wiki_verify(self.root, fix=True))
            self.write_sources([registration])
            other = self.wiki / "topic-srcid-abc123"
            other.mkdir()
            (other / "source-summary-abc123.md").write_text(
                (document / "source-summary-abc123.md").read_text()
            )
            self.assertFalse(wiki_verify(self.root, fix=True))
        fix.assert_not_called()

    def test_shared_or_symlinked_source_directory_blocks_fix(self) -> None:
        first, document, _, _ = self.prepare_repair()
        summary = document / "source-summary-abc123.md"
        summary.unlink()
        nested_raw = self.raw / "nested" / "topic.txt"
        nested_raw.parent.mkdir()
        nested_raw.write_text("second source")
        second = {"filename": "nested/topic.txt", "srcid": "srcid-second",
                  "sha256": hashlib.sha256(nested_raw.read_bytes()).hexdigest()}
        self.write_sources([first, second])
        with patch("llm_wiki.wiki_fix.fix_source") as fix:
            self.assertFalse(wiki_verify(self.root, fix=True))
            self.assertIn("shared by multiple registrations", self.log.read_text())
            self.write_sources([first])
            actual = self.wiki / "other-source"
            document.rename(actual)
            document.symlink_to(actual, target_is_directory=True)
            self.assertFalse(wiki_verify(self.root, fix=True))
        fix.assert_not_called()

    def test_fix_can_restore_missing_summary_and_entity(self) -> None:
        registration, document = self.add_source("topic.txt", "srcid-abc123")
        self.write_sources([registration])
        summary = document / "source-summary-abc123.md"
        summary_text = summary.read_text()
        summary.unlink()
        entity = document / "entities" / "person.md"
        entity_text = entity.read_text()
        entity.unlink()
        entity.parent.rmdir()
        self.mock_model([
            self.write_response(summary.name, summary_text),
            self.write_response("entities/person.md", entity_text),
            AIMessage(content="done"),
        ])
        self.assertTrue(wiki_verify(self.root, fix=True))
        self.assertEqual(summary.read_text(), summary_text + "\n")
        self.assertEqual(entity.read_text(), entity_text)

    def test_fix_failure_continues_to_other_source_and_full_recheck(self) -> None:
        first, _, _, _ = self.prepare_repair()
        second, document = self.add_source("second.txt", "srcid-second")
        self.write_sources([first, second])
        page = document / "entities" / "person.md"
        original = page.read_text()
        page.write_text(original.replace('description: "A person"', 'description: ""'))
        reporter = Mock(spec=ProgressReporter)
        model, _ = self.mock_model([
            RuntimeError("model failed"),
            self.write_response("entities/person.md", original), AIMessage(content="done"),
        ])
        self.assertFalse(wiki_verify(self.root, fix=True, reporter=reporter))
        self.assertEqual(model.invoke.call_count, 3)
        reporter.agent_failed.assert_called_once()
        self.assertEqual(page.read_text(), original)
        log = self.log.read_text()
        self.assertIn("recheck | topic.txt", log)
        self.assertIn("recheck | second.txt", log)

    def test_fix_corrects_tool_error_and_rechecks(self) -> None:
        _, _, page, original = self.prepare_repair()
        model, _ = self.mock_model([
            self.write_response("concepts/unreported.md", "invalid target"),
            self.write_response("concepts/topic.md", original),
            AIMessage(content="done"),
        ])
        reporter = Mock(spec=ProgressReporter)
        self.assertTrue(wiki_verify(self.root, fix=True, reporter=reporter))
        feedback = model.invoke.call_args_list[1].args[0][-1]
        self.assertEqual(feedback.status, "error")
        self.assertEqual(feedback.tool_call_id, "fix-write")
        self.assertIn("Only reported files", feedback.content)
        self.assertEqual(page.read_text(), original)
        reporter.tool_failed.assert_called_once()
        reporter.agent_failed.assert_not_called()

    def test_fix_model_turns_are_bounded(self) -> None:
        self.prepare_repair()
        response = AIMessage(content="", tool_calls=[{
            "name": "list_files", "args": {}, "id": "list", "type": "tool_call",
        }])
        model, _ = self.mock_model([response.model_copy(deep=True) for _ in range(12)])
        self.assertFalse(wiki_verify(self.root, fix=True))
        self.assertEqual(model.invoke.call_count, 12)
        self.assertIn("exceeded 12 model turns", self.log.read_text())

    def test_fix_can_finish_on_twelfth_model_turn(self) -> None:
        _, _, page, original = self.prepare_repair()
        responses = [AIMessage(content="", tool_calls=[{
            "name": "list_files", "args": {}, "id": f"list-{turn}", "type": "tool_call",
        }]) for turn in range(10)]
        model, _ = self.mock_model([
            *responses, self.write_response("concepts/topic.md", original),
            AIMessage(content="done"),
        ])
        self.assertTrue(wiki_verify(self.root, fix=True))
        self.assertEqual(model.invoke.call_count, 12)
        self.assertEqual(page.read_text(), original)

    def test_fix_tools_reject_unrelated_files_and_path_escape(self) -> None:
        _, document, _, _ = self.prepare_repair()
        outside = self.root / "outside.md"
        outside.write_text("keep")
        (document / "concepts" / "linked.md").symlink_to(outside)
        written = []
        tools = {item.name: item for item in _fix_tools(document, [
            {"file": "concepts/topic.md"}, {"file": "concepts/linked.md"},
            {"file": "entities/"}, {"file": "index.md"},
        ], written)}
        for path in ("../../raw/topic.txt", "../sources.json", "../aliases.json",
                     "index.md", "entities/person.md", "concepts/unrelated.md",
                     "concepts/linked.md", str(outside)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                tools["write_file"].invoke({"path": path, "content": "bad"})
        with self.assertRaises(ValueError):
            tools["read_file"].invoke({"path": "concepts/linked.md"})
        with self.assertRaises(ValueError):
            tools["mkdir"].invoke({"path": "findings"})
        self.assertEqual(outside.read_text(), "keep")
        self.assertEqual(written, [])

    def test_register_writes_source_hash_without_algorithm_prefix(self) -> None:
        source = self.raw / "topic.txt"
        source.write_text("registered source\n", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()

        self.assertTrue(wiki_register(self.root))

        registration = json.loads(
            (self.wiki / "sources.json").read_text(encoding="utf-8")
        )[0]
        summary = (
            self.wiki / source.stem
            / f"source-summary-{registration['srcid'].removeprefix('srcid-')}.md"
        )
        summary_text = summary.read_text(encoding="utf-8")

        self.assertIn(f'source_hash: "{digest}"', summary_text)
        self.assertNotIn(f'source_hash: "sha256:{digest}"', summary_text)


if __name__ == "__main__":
    unittest.main()
