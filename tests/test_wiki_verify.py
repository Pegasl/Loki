import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_wiki.wiki import wiki_register, wiki_verify


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
