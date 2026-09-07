import io
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import ANY, Mock, call, patch

from llm_wiki import workflow


STATE = {
    "init": False, "ingest": False, "verify": False,
    "index": False,
}


class WorkflowProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        ask_patch = patch.object(workflow, "ask", side_effect=lambda state: {"next": workflow.END})
        ask_patch.start()
        self.addCleanup(ask_patch.stop)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sources = Path(self.temp_dir.name) / "sources.json"
        self.sources_patch = patch.object(workflow, "sources", self.sources)
        self.sources_patch.start()
        self.root = Path(self.temp_dir.name)
        self.root_patch = patch.object(workflow, "root", self.root)
        self.root_patch.start()
        self.qmd = self.root / ".qmd"
        self.qmd.mkdir()
        (self.qmd / "index.yml").write_text("collections: {}\n")
        (self.qmd / "index.sqlite").touch()
        model_dir = Path.home() / ".cache" / "qmd" / "models"
        self.models = {
            "embed": str(model_dir / "Qwen3-Embedding-0.6B-Q8_0.gguf"),
            "generate": str(model_dir / "qmd-query-expansion-1.7B-q4_k_m.gguf"),
            "rerank": str(model_dir / "qwen3-reranker-0.6b-q8_0.gguf"),
        }

    def tearDown(self) -> None:
        self.sources_patch.stop()
        self.root_patch.stop()
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
        verify.assert_called_once_with(fix=True, reporter=None)

    def test_graph_skips_completed_stages(self) -> None:
        for completed in (set(STATE), {"init", "ingest"}, {"verify"}):
            with self.subTest(completed=completed), ExitStack() as stack:
                save = stack.enter_context(patch.object(workflow, "save_state"))
                nodes = {
                    stage: stack.enter_context(patch.object(
                        workflow, stage, return_value={stage: True}
                    ))
                    for stage in STATE
                }
                reporter = Mock(spec=workflow.ProgressReporter)
                result = workflow.build_graph(reporter).invoke({
                    stage: stage in completed for stage in STATE
                })
                self.assertTrue(all(result[stage] for stage in STATE))
                for stage, node in nodes.items():
                    self.assertEqual(node.call_count, int(stage not in completed))
                self.assertEqual(reporter.stage_started.call_args_list, [
                    call(stage, 1) for stage in STATE if stage not in completed
                ])
                self.assertEqual(save.call_count, len(STATE) - len(completed))

    def test_index_runs_embed_and_propagates_failure(self) -> None:
        with (
            patch.object(workflow, "wiki_index", return_value=True),
            patch.object(workflow.subprocess, "run") as run,
        ):
            self.assertEqual(workflow.index(STATE), {"index": True})
        self.assertEqual(run.call_args_list, [
            call(["qmd", "update"], cwd=workflow.root, env=ANY, check=True),
            call(["qmd", "embed"], cwd=workflow.root, env=ANY, check=True),
        ])
        self.assertEqual(run.call_args.kwargs["env"]["PWD"], str(self.root))
        for error in (FileNotFoundError("qmd"), subprocess.CalledProcessError(1, "qmd")):
            with (
                self.subTest(error=error),
                patch.object(workflow, "wiki_index", return_value=True),
                patch.object(workflow.subprocess, "run", side_effect=error),
            ):
                with self.assertRaises(type(error)):
                    workflow.index(STATE)

    def test_index_failure_skips_embed(self) -> None:
        with (
            patch.object(workflow, "wiki_index", return_value=False),
            patch.object(workflow.subprocess, "run") as run,
        ):
            self.assertEqual(workflow.index(STATE), {"index": False})
        run.assert_not_called()

    def test_old_completed_state_still_runs_index_then_embed(self) -> None:
        old_state = {stage: True for stage in ("init", "ingest", "verify", "retrieve")}
        calls = Mock()
        with (
            patch.object(workflow, "save_state"),
            patch.object(workflow, "wiki_index", return_value=True) as index,
            patch.object(workflow.subprocess, "run") as run,
        ):
            calls.attach_mock(index, "index")
            calls.attach_mock(run, "embed")
            result = workflow.build_graph().invoke(old_state)
        self.assertTrue(result["index"])
        self.assertNotIn("embed", result)
        self.assertEqual(calls.mock_calls, [
            call.index(),
            call.embed(["qmd", "update"], cwd=workflow.root, env=ANY, check=True),
            call.embed(["qmd", "embed"], cwd=workflow.root, env=ANY, check=True),
        ])

    def test_index_initializes_missing_local_config_or_database(self) -> None:
        for missing in ("index.yml", "index.sqlite"):
            with self.subTest(missing=missing):
                (self.qmd / missing).unlink()

                def run_command(command, **kwargs):
                    for role, path in self.models.items():
                        self.assertEqual(kwargs["env"][f"QMD_{role.upper()}_MODEL"], path)
                    if command == ["qmd", "init"]:
                        (self.qmd / "index.yml").write_text("collections: {}\n")
                        (self.qmd / "index.sqlite").touch()

                with (
                    patch.object(workflow, "wiki_index", return_value=True),
                    patch.object(workflow.subprocess, "run", side_effect=run_command) as run,
                ):
                    self.assertEqual(workflow.index(STATE), {"index": True})
                self.assertEqual(run.call_args_list, [
                    call(["qmd", command], cwd=self.root, env=ANY, check=True)
                    for command in ("init", "update", "embed")
                ])

    def test_index_limits_collections_and_sets_cached_models_for_both_config_extensions(self) -> None:
        for name in ("index.yml", "index.yaml"):
            with self.subTest(name=name):
                config_path = self.qmd / name
                config_path.write_text(
                    "models:\n  embed: custom.gguf\ncollections:\n  all:\n    path: .\n"
                )
                with (
                    patch.object(workflow, "wiki_index", return_value=True),
                    patch.object(workflow.subprocess, "run"),
                ):
                    self.assertEqual(workflow.index(STATE), {"index": True})
                config = workflow.yaml.safe_load(config_path.read_text())
                self.assertEqual(config["models"], self.models)
                self.assertEqual(config["collections"], {
                    directory: {
                        "path": directory, "pattern": "**/*.{md,txt}",
                        "includeByDefault": True,
                    }
                    for directory in ("raw", "wiki")
                })

    def test_index_replaces_models_even_when_collections_already_match(self) -> None:
        with (
            patch.object(workflow, "wiki_index", return_value=True),
            patch.object(workflow.subprocess, "run"),
        ):
            workflow.index(STATE)
            config_path = self.qmd / "index.yml"
            config = workflow.yaml.safe_load(config_path.read_text())
            config["models"] = {"embed": "old-model.gguf"}
            config_path.write_text(workflow.yaml.safe_dump(config))
            workflow.index(STATE)
        self.assertEqual(
            workflow.yaml.safe_load(config_path.read_text())["models"], self.models
        )

    def test_qmd_update_failure_prevents_embed(self) -> None:
        with (
            patch.object(workflow, "wiki_index", return_value=True),
            patch.object(workflow.subprocess, "run", side_effect=
                         subprocess.CalledProcessError(1, ["qmd", "update"])) as run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.index(STATE)
        run.assert_called_once_with(["qmd", "update"], cwd=self.root, env=ANY, check=True)

    def test_native_retries_stop_and_verify_is_attempted_only_once(self) -> None:
        real_policy = workflow.RetryPolicy
        for failed_stage in STATE:
            with self.subTest(stage=failed_stage), ExitStack() as stack:
                stack.enter_context(patch.object(workflow, "save_state"))
                stack.enter_context(patch.object(
                    workflow, "RetryPolicy",
                    side_effect=lambda **kwargs: real_policy(**kwargs)._replace(
                        initial_interval=0, jitter=False
                    ),
                ))
                nodes = {
                    stage: stack.enter_context(patch.object(
                        workflow, stage, return_value={stage: stage != failed_stage}
                    ))
                    for stage in STATE
                }
                reporter = Mock(spec=workflow.ProgressReporter)
                graph = workflow.build_graph(reporter)
                for _ in range(2):
                    for node in nodes.values():
                        node.reset_mock()
                    reporter.reset_mock()
                    with self.assertRaises(workflow.NodeFailedError):
                        graph.invoke(STATE)
                    attempts = 1 if failed_stage == "verify" else 6
                    self.assertEqual(nodes[failed_stage].call_count, attempts)
                    self.assertEqual(reporter.stage_retrying.call_args_list, [
                        call(failed_stage, attempt) for attempt in range(2, attempts + 1)
                    ])
                    stages = list(STATE)
                    for stage in stages[stages.index(failed_stage) + 1:]:
                        nodes[stage].assert_not_called()

    def test_native_retries_can_succeed_on_last_attempt(self) -> None:
        real_policy = workflow.RetryPolicy
        with ExitStack() as stack:
            stack.enter_context(patch.object(workflow, "save_state"))
            stack.enter_context(patch.object(
                workflow, "RetryPolicy",
                side_effect=lambda **kwargs: real_policy(**kwargs)._replace(
                    initial_interval=0, jitter=False
                ),
            ))
            nodes = {
                stage: stack.enter_context(patch.object(
                    workflow, stage,
                    side_effect=([{stage: False}] * 5 if stage != "verify" else []) + [{stage: True}],
                ))
                for stage in STATE
            }
            result = workflow.build_graph().invoke(STATE)
        self.assertTrue(all(result[stage] for stage in STATE))
        for stage, node in nodes.items():
            self.assertEqual(node.call_count, 1 if stage == "verify" else 6)

    def test_graph_reports_stage_lifecycle(self) -> None:
        reporter = Mock(spec=workflow.ProgressReporter)
        state_path = Path(self.temp_dir.name) / "state.json"
        self.write_sources([])

        with (
            patch.object(workflow, "state_file", state_path),
            patch.object(workflow, "wiki_init", return_value=True),
            patch.object(workflow, "wiki_register", return_value=True),
            patch.object(workflow, "wiki_verify", return_value=True),
            patch.object(workflow, "wiki_index", return_value=True),
            patch.object(workflow.subprocess, "run"),
        ):
            result = workflow.build_graph(reporter).invoke(STATE)

        self.assertEqual(
            {stage: result[stage] for stage in STATE},
            {stage: True for stage in STATE},
        )
        self.assertEqual(
            reporter.stage_started.call_args_list,
            [
                call("init", 1),
                call("ingest", 1),
                call("verify", 1),
                call("index", 1),
            ],
        )
        self.assertEqual(
            reporter.stage_succeeded.call_args_list,
            [call(stage) for stage in STATE],
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
            patch.object(workflow, "wiki_index", return_value=True),
            patch.object(workflow.subprocess, "run"),
        ):
            workflow.build_graph(reporter).invoke(STATE)

        self.assertEqual(
            reporter.stage_started.call_args_list[:2],
            [call("init", 1), call("init", 2)],
        )
        reporter.stage_failed.assert_called_once_with("init")
        reporter.stage_retrying.assert_called_once_with("init", 2)

    def test_completed_old_state_still_enters_ask(self):
        with patch.object(workflow, "ask", return_value={"next": workflow.END}) as ask:
            result = workflow.build_graph().invoke({**{stage: True for stage in STATE}, "retrieve": True})
        ask.assert_called_once()
        self.assertNotIn("retrieve", result)

    def test_save_state_excludes_conversation(self):
        state_path = self.root / "state.json"
        with patch.object(workflow, "state_file", state_path):
            workflow.save_state({**STATE, "messages": [object()], "retrieve": True})
        self.assertEqual(json.loads(state_path.read_text()), STATE)


if __name__ == "__main__":
    unittest.main()
