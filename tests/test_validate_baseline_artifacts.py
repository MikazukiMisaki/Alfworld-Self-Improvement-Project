from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.validate_baseline_artifacts import (
    ArtifactValidationError,
    main,
    validate_baseline_artifacts,
)


class BaselineArtifactValidatorTests(unittest.TestCase):
    def test_zero_reward_grounded_run_passes(self) -> None:
        with self._run_directory() as run_directory:
            summary = validate_baseline_artifacts(
                run_directory, expected_git_revision="abc123"
            )

        self.assertEqual(summary.reward, 0.0)
        self.assertEqual(summary.action_selection_mode, "free_form_validated")
        self.assertEqual(summary.parser_statuses, ("grounded",))
        self.assertEqual(summary.first_parsed_action, "look")

    def test_parser_failure_fails(self) -> None:
        artifacts = self._artifacts()
        artifacts["trajectory"]["metadata"]["termination_reason"] = "parser_failure"
        with self._run_directory(artifacts) as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "parser_failure"):
                validate_baseline_artifacts(run_directory)

    def test_invalid_action_rate_one_fails(self) -> None:
        artifacts = self._artifacts()
        artifacts["metrics"]["invalid_action_rate"] = 1.0
        with self._run_directory(artifacts) as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "invalid_action_rate"):
                validate_baseline_artifacts(run_directory)

    def test_repeated_max_token_saturation_fails(self) -> None:
        artifacts = self._artifacts(step_count=2)
        for step in artifacts["trajectory"]["steps"]:
            step["token_statistics"]["generated_tokens"] = 32
        artifacts["metrics"]["mean_generated_tokens"] = 64.0
        with self._run_directory(artifacts) as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "token cap"):
                validate_baseline_artifacts(run_directory)

    def test_one_max_token_saturation_does_not_fail(self) -> None:
        artifacts = self._artifacts()
        artifacts["trajectory"]["steps"][0]["token_statistics"]["generated_tokens"] = 32
        artifacts["metrics"]["mean_generated_tokens"] = 32.0
        with self._run_directory(artifacts) as run_directory:
            summary = validate_baseline_artifacts(run_directory)
        self.assertEqual(summary.generated_tokens, 32.0)

    def test_missing_artifact_fails(self) -> None:
        with self._run_directory() as run_directory:
            (run_directory / "metrics.json").unlink()
            with self.assertRaisesRegex(ArtifactValidationError, "missing or empty"):
                validate_baseline_artifacts(run_directory)

    def test_incorrect_configuration_fails(self) -> None:
        artifacts = self._artifacts()
        artifacts["manifest"]["resolved_config"]["model"]["enable_thinking"] = True
        with self._run_directory(artifacts) as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "enable_thinking"):
                validate_baseline_artifacts(run_directory)

    def test_missing_explicit_pipeline_version_fails(self) -> None:
        artifacts = self._artifacts()
        del artifacts["manifest"]["pipeline_version"]
        with self._run_directory(artifacts) as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "pipeline_version"):
                validate_baseline_artifacts(run_directory)

    def test_git_revision_mismatch_fails(self) -> None:
        with self._run_directory() as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "git revision mismatch"):
                validate_baseline_artifacts(
                    run_directory, expected_git_revision="different"
                )

    def test_missing_git_revision_passes_when_not_expected(self) -> None:
        artifacts = self._artifacts()
        artifacts["manifest"]["git_revision"] = None
        with self._run_directory(artifacts) as run_directory:
            summary = validate_baseline_artifacts(run_directory)
        self.assertIsNone(summary.git_revision)

    def test_missing_git_revision_fails_when_expected(self) -> None:
        artifacts = self._artifacts()
        artifacts["manifest"]["git_revision"] = None
        with self._run_directory(artifacts) as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "git revision mismatch"):
                validate_baseline_artifacts(
                    run_directory, expected_git_revision="abc123"
                )

    def test_invalid_git_revision_type_fails(self) -> None:
        artifacts = self._artifacts()
        artifacts["manifest"]["git_revision"] = 123
        with self._run_directory(artifacts) as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "must be a string or null"):
                validate_baseline_artifacts(run_directory)

    def test_indexed_artifact_mapping_passes(self) -> None:
        with self._run_directory(self._indexed_artifacts()) as run_directory:
            summary = validate_baseline_artifacts(run_directory)
        self.assertEqual(summary.action_selection_mode, "indexed_admissible")

    def test_indexed_artifact_mapping_mismatch_fails(self) -> None:
        artifacts = self._indexed_artifacts()
        artifacts["trajectory"]["steps"][0]["metadata"]["action_selection"][
            "id_to_command"
        ]["A000"] = "inventory"
        with self._run_directory(artifacts) as run_directory:
            with self.assertRaisesRegex(ArtifactValidationError, "does not preserve"):
                validate_baseline_artifacts(run_directory)

    def test_cli_prints_pass_as_final_status(self) -> None:
        with self._run_directory() as run_directory:
            stdout = io.StringIO()
            with patch("sys.argv", ["validator", str(run_directory)]):
                with redirect_stdout(stdout):
                    exit_code = main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().splitlines()[-1], "BASELINE_SMOKE_PASS")

    def test_cli_prints_fail_as_final_status(self) -> None:
        with self._run_directory() as run_directory:
            (run_directory / "metrics.json").unlink()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("sys.argv", ["validator", str(run_directory)]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = main()
        self.assertEqual(exit_code, 1)
        self.assertIn("validation_error", stderr.getvalue())
        self.assertEqual(stdout.getvalue().splitlines()[-1], "BASELINE_SMOKE_FAIL")

    @staticmethod
    def _artifacts(step_count: int = 1) -> dict[str, dict]:
        steps = []
        for index in range(step_count):
            steps.append(
                {
                    "index": index,
                    "observation": "room",
                    "action": "look",
                    "model_output": "Action: look",
                    "reward": 0.0,
                    "done": False,
                    "timestamp": "2026-08-09T00:00:00+00:00",
                    "token_statistics": {
                        "generated_tokens": 4,
                        "mean_token_log_probability": -0.1,
                        "mean_token_entropy": 0.2,
                    },
                    "reasoning": None,
                    "parser_status": "grounded",
                    "valid_actions": ["look", "inventory"],
                    "action_valid": True,
                    "metadata": {
                        "action_selection_mode": "free_form_validated",
                        "action_selection": {
                            "action_selection_mode": "free_form_validated",
                            "raw_model_output": "Action: look",
                            "parsed_action_id": None,
                            "selected_index": None,
                            "mapped_environment_command": "look",
                            "id_to_command": {},
                            "selection_status": "not_applicable",
                            "failure_reason": None,
                        },
                        "debug": {"invalid_action_reason": None},
                    },
                }
            )
        return {
            "manifest": {
                "run_id": "smoke-run",
                "created_at": "2026-08-09T00:00:00+00:00",
                "model_version": "Qwen/Qwen3-8B",
                "environment": "alfworld_text",
                "resolved_config": {
                    "collection": {
                        "episodes": 1,
                        "max_steps": 50,
                    },
                    "environment": {
                        "config_path": "/tmp/base_config.yaml",
                        "data_path": "/tmp/alfworld_data",
                        "split": "valid_seen",
                    },
                    "model": {
                        "model_id": "Qwen/Qwen3-8B",
                        "pipeline_version": "free_form_v1",
                        "device": "auto",
                        "dtype": "bfloat16",
                        "enable_thinking": False,
                        "action_selection": {"mode": "free_form_validated"},
                        "generation": {
                            "max_new_tokens": 32,
                            "do_sample": False,
                            "temperature": None,
                            "top_p": None,
                        },
                    },
                },
                "seed_schedule": [42],
                "git_revision": "abc123",
                "pipeline_version": "free_form_v1",
                "action_selection_mode": "free_form_validated",
                "split": "valid_seen",
                "metadata": {},
            },
            "metrics": {
                "episodes": 1,
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "mean_episode_length": float(step_count),
                "parser_failure_rate": 0.0,
                "inadmissible_candidate_rate": 0.0,
                "invalid_action_rate": 0.0,
                "selection_failure_rate": 0.0,
                "malformed_id_rate": 0.0,
                "out_of_range_id_rate": 0.0,
                "mean_generated_tokens": float(4 * step_count),
            },
            "trajectory": {
                "trajectory_id": "trajectory-1",
                "task": {
                    "task_id": "task/trial",
                    "text": "look around",
                    "split": "valid_seen",
                    "family": None,
                    "metadata": {},
                },
                "model_version": "Qwen/Qwen3-8B",
                "seed": 42,
                "initial_observation": "room",
                "steps": steps,
                "started_at": "2026-08-09T00:00:00+00:00",
                "completed_at": "2026-08-09T00:00:01+00:00",
                "truncated": True,
                "metadata": {
                    "max_steps": 50,
                    "termination_reason": "max_steps",
                },
            },
        }

    def _indexed_artifacts(self) -> dict[str, dict]:
        artifacts = self._artifacts()
        artifacts["manifest"]["resolved_config"]["model"]["action_selection"][
            "mode"
        ] = "indexed_admissible"
        artifacts["manifest"]["resolved_config"]["model"][
            "pipeline_version"
        ] = "indexed_v1"
        artifacts["manifest"]["pipeline_version"] = "indexed_v1"
        artifacts["manifest"]["action_selection_mode"] = "indexed_admissible"
        step = artifacts["trajectory"]["steps"][0]
        step["model_output"] = "Action-ID: A000"
        step["metadata"]["action_selection_mode"] = "indexed_admissible"
        step["metadata"]["action_selection"] = {
            "action_selection_mode": "indexed_admissible",
            "raw_model_output": "Action-ID: A000",
            "parsed_action_id": "A000",
            "selected_index": 0,
            "mapped_environment_command": "look",
            "id_to_command": {"A000": "look", "A001": "inventory"},
            "selection_status": "selected",
            "failure_reason": None,
        }
        return artifacts

    class _RunDirectory:
        def __init__(self, artifacts: dict[str, dict]) -> None:
            self._artifacts = artifacts
            self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            self._temporary_directory = tempfile.TemporaryDirectory()
            root = Path(self._temporary_directory.name)
            run_directory = root / "smoke-run"
            run_directory.mkdir()
            (run_directory / "run_manifest.json").write_text(
                json.dumps(self._artifacts["manifest"]), encoding="utf-8"
            )
            (run_directory / "metrics.json").write_text(
                json.dumps(self._artifacts["metrics"]), encoding="utf-8"
            )
            (run_directory / "trajectory.jsonl").write_text(
                json.dumps(self._artifacts["trajectory"]) + "\n", encoding="utf-8"
            )
            return run_directory

        def __exit__(self, *args: object) -> None:
            assert self._temporary_directory is not None
            self._temporary_directory.cleanup()

    def _run_directory(
        self, artifacts: dict[str, dict] | None = None
    ) -> _RunDirectory:
        return self._RunDirectory(copy.deepcopy(artifacts or self._artifacts()))
