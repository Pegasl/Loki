import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch

from llm_wiki import workflow


STATE = {"init": False, "ingest": False, "verify": False, "retrieve": False}


class WorkflowProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sources = Path(self.temp_dir.name) / "sources.json"
        self.sources_patch = patch.object(workflow, "sources", self.sources)
        self.sources_patch.start()

    def tearDown(self) -> None:
        self.sources_patch.stop()
        self.temp_dir.cleanup()

    def write_sources(self, registrations: object) -> None:
        self.sources.write_text(json.dumps(registrations), encoding="utf-8")

    def test_ingest_reports_each_source_and_continues_after_failure(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        self.write_sources(
            [
                {"srcid": "first", "ingest": False},
                {"srcid": "second", "ingest": False},
                {"srcid": "done", "ingest": True},
            ]
        )

        with (
            patch.object(workflow, "wiki_register", return_value=True),
            patch.object(workflow, "wiki_ingest", side_effect=[True, False]) as ingest,
        ):
            result = workflow.ingest(STATE, reporter)

        self.assertEqual(result, {"ingest": False})
        self.assertEqual(
            ingest.call_args_list,
            [call("first", reporter=reporter), call("second", reporter=reporter)],
        )
        reporter.source_started.assert_has_calls(
            [call(1, 2, "first"), call(2, 2, "second")]
        )
        reporter.source_succeeded.assert_called_once_with(1, 2, "first")
        reporter.source_failed.assert_called_once_with(2, 2, "second")

    def test_ingest_reports_no_pending_sources(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        self.write_sources([{"srcid": "done", "ingest": True}])

        with (
            patch.object(workflow, "wiki_register", return_value=True),
            patch.object(workflow, "wiki_ingest") as ingest,
        ):
            result = workflow.ingest(STATE, reporter)

        self.assertEqual(result, {"ingest": True})
        reporter.no_pending_sources.assert_called_once_with()
        ingest.assert_not_called()

    def test_invalid_registration_does_not_report_source_progress(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        self.write_sources([{"srcid": "", "ingest": False}])

        with patch.object(workflow, "wiki_register", return_value=True):
            result = workflow.ingest(STATE, reporter)

        self.assertEqual(result, {"ingest": False})
        reporter.source_started.assert_not_called()
        reporter.no_pending_sources.assert_not_called()

    def test_ingest_reports_source_failure_and_propagates_exception(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        self.write_sources([{"srcid": "broken", "ingest": False}])

        with (
            patch.object(workflow, "wiki_register", return_value=True),
            patch.object(workflow, "wiki_ingest", side_effect=RuntimeError("boom")),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                workflow.ingest(STATE, reporter)

        reporter.source_failed.assert_called_once_with(1, 1, "broken")
        reporter.source_succeeded.assert_not_called()

    def test_terminal_reporter_prints_single_lines_and_flushes(self) -> None:
        reporter = workflow.TerminalProgressReporter()
        output = io.StringIO()

        with redirect_stdout(output):
            reporter.stage_started("ingest", 1)
            reporter.source_started(1, 2, "source\nname")
            reporter.stage_failed("ingest", RuntimeError("bad\ninput"))

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "[ingest] START attempt=1",
                "[ingest] SOURCE 1/2 START srcid=source name",
                "[ingest] FAILED error=bad input",
            ],
        )

        with patch("builtins.print") as print_mock:
            reporter.workflow_started()
        print_mock.assert_called_once_with("[workflow] START", flush=True)

    def test_reporter_exception_does_not_change_ingest_result(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        reporter.source_started.side_effect = RuntimeError("display failed")
        self.write_sources([{"srcid": "source", "ingest": False}])

        with (
            patch.object(workflow, "wiki_register", return_value=True),
            patch.object(workflow, "wiki_ingest", return_value=True),
        ):
            result = workflow.ingest(STATE, reporter)

        self.assertEqual(result, {"ingest": True})
        reporter.source_succeeded.assert_called_once_with(1, 1, "source")

    def test_verify_returns_wiki_verify_result(self) -> None:
        with patch.object(workflow, "wiki_verify", return_value=False) as verify:
            result = workflow.verify(STATE)

        self.assertEqual(result, {"verify": False})
        verify.assert_called_once_with()

    def test_graph_reports_stage_lifecycle(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        state_path = Path(self.temp_dir.name) / "state.json"
        self.write_sources([])

        with (
            patch.object(workflow, "state_file", state_path),
            patch.object(workflow, "wiki_init", return_value=True),
            patch.object(workflow, "wiki_register", return_value=True),
            patch.object(workflow, "wiki_verify", return_value=True),
        ):
            result = workflow.build_graph(reporter).invoke(STATE)

        self.assertEqual(
            result,
            {"init": True, "ingest": True, "verify": True, "retrieve": True},
        )
        self.assertEqual(
            reporter.stage_started.call_args_list,
            [
                call("init", 1),
                call("ingest", 1),
                call("verify", 1),
                call("retrieve", 1),
            ],
        )
        self.assertEqual(
            reporter.stage_succeeded.call_args_list,
            [call("init"), call("ingest"), call("verify"), call("retrieve")],
        )
        reporter.stage_failed.assert_not_called()
        reporter.stage_retrying.assert_not_called()

    def test_graph_reports_retry_attempt(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        state_path = Path(self.temp_dir.name) / "state.json"
        self.write_sources([])

        with (
            patch.object(workflow, "state_file", state_path),
            patch.object(workflow, "wiki_init", side_effect=[False, True]),
            patch.object(workflow, "wiki_register", return_value=True),
            patch.object(workflow, "wiki_verify", return_value=True),
        ):
            workflow.build_graph(reporter).invoke(STATE)

        self.assertEqual(
            reporter.stage_started.call_args_list[:2],
            [call("init", 1), call("init", 2)],
        )
        reporter.stage_failed.assert_called_once_with("init")
        reporter.stage_retrying.assert_called_once_with("init", 2)

    def test_graph_retries_retrieve_before_completing(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        state_path = Path(self.temp_dir.name) / "state.json"
        self.write_sources([])

        with (
            patch.object(workflow, "state_file", state_path),
            patch.object(workflow, "wiki_init", return_value=True),
            patch.object(workflow, "wiki_register", return_value=True),
            patch.object(workflow, "wiki_verify", return_value=True),
            patch.object(workflow, "retrieve", side_effect=[{"retrieve": False}, {"retrieve": True}]),
        ):
            result = workflow.build_graph(reporter).invoke(STATE)

        self.assertTrue(result["retrieve"])
        self.assertEqual(
            [item for item in reporter.stage_started.call_args_list if item.args[0] == "retrieve"],
            [call("retrieve", 1), call("retrieve", 2)],
        )
        reporter.stage_retrying.assert_called_once_with("retrieve", 2)


if __name__ == "__main__":
    unittest.main()
