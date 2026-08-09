#!/usr/bin/env python3
"""Validate and summarize one Sprint 1.5 baseline smoke-test run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = ("run_manifest.json", "trajectory.jsonl", "metrics.json")
EXPECTED_MODEL = "Qwen/Qwen3-8B"
EXPECTED_ENVIRONMENT = "alfworld_text"
EXPECTED_MAX_NEW_TOKENS = 32
MAX_REPEATED_SATURATIONS = 2


class ArtifactValidationError(ValueError):
    """Raised when baseline smoke artifacts violate the approved contract."""


@dataclass(frozen=True)
class BaselineSmokeSummary:
    """Concise fields printed after a baseline artifact validation."""

    run_id: str
    git_revision: str
    task_id: str
    termination_reason: str
    steps: int
    success_rate: float
    reward: float
    invalid_action_rate: float
    generated_tokens: float
    parser_statuses: tuple[str, ...]
    first_raw_output: str
    first_parsed_action: str


def validate_baseline_artifacts(
    run_directory: Path,
    *,
    expected_git_revision: str | None = None,
) -> BaselineSmokeSummary:
    """Validate one run directory and return its smoke-test summary.

    A zero reward is intentionally not an error. This validator checks runtime,
    artifact, grounding, and provenance health rather than task-solving quality.
    """
    run_directory = run_directory.resolve()
    if not run_directory.is_dir():
        raise ArtifactValidationError(f"run directory does not exist: {run_directory}")
    for filename in REQUIRED_ARTIFACTS:
        path = run_directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ArtifactValidationError(f"missing or empty artifact: {filename}")

    manifest = _read_json_object(run_directory / "run_manifest.json")
    metrics = _read_json_object(run_directory / "metrics.json")
    trajectories = _read_jsonl_objects(run_directory / "trajectory.jsonl")

    if len(trajectories) != 1:
        raise ArtifactValidationError(
            f"expected exactly one trajectory, found {len(trajectories)}"
        )
    trajectory = trajectories[0]

    run_id = _required_string(manifest, "run_id", "run manifest")
    if run_id != run_directory.name:
        raise ArtifactValidationError(
            f"run_id {run_id!r} does not match directory {run_directory.name!r}"
        )
    git_revision = _required_string(manifest, "git_revision", "run manifest")
    if expected_git_revision is not None and git_revision != expected_git_revision:
        raise ArtifactValidationError(
            f"git revision mismatch: manifest={git_revision}, expected={expected_git_revision}"
        )
    if manifest.get("environment") != EXPECTED_ENVIRONMENT:
        raise ArtifactValidationError(
            f"unexpected environment: {manifest.get('environment')!r}"
        )
    if manifest.get("model_version") != EXPECTED_MODEL:
        raise ArtifactValidationError(
            f"unexpected model version: {manifest.get('model_version')!r}"
        )

    resolved_config = _required_mapping(manifest, "resolved_config", "run manifest")
    collection_config = _required_mapping(
        resolved_config, "collection", "resolved config"
    )
    environment_config = _required_mapping(
        resolved_config, "environment", "resolved config"
    )
    model_config = _required_mapping(resolved_config, "model", "resolved config")
    generation_config = _required_mapping(model_config, "generation", "model config")

    if collection_config.get("episodes") != 1:
        raise ArtifactValidationError("resolved collection config must contain episodes=1")
    if model_config.get("enable_thinking") is not False:
        raise ArtifactValidationError("resolved model config must contain enable_thinking=false")
    max_new_tokens = generation_config.get("max_new_tokens")
    if max_new_tokens != EXPECTED_MAX_NEW_TOKENS:
        raise ArtifactValidationError(
            f"resolved max_new_tokens must be {EXPECTED_MAX_NEW_TOKENS}, got {max_new_tokens!r}"
        )
    if model_config.get("model_id") != EXPECTED_MODEL:
        raise ArtifactValidationError("resolved model_id does not match manifest model")
    if environment_config.get("split") != "valid_seen":
        raise ArtifactValidationError("smoke run must use the valid_seen split")
    for key in ("config_path", "data_path"):
        value = _required_string(environment_config, key, "environment config")
        if "$" in value:
            raise ArtifactValidationError(f"environment {key} was not resolved: {value!r}")

    seed_schedule = manifest.get("seed_schedule")
    if not isinstance(seed_schedule, list) or len(seed_schedule) != 1:
        raise ArtifactValidationError("run manifest must contain exactly one seed")
    if metrics.get("episodes") != 1:
        raise ArtifactValidationError("metrics must contain episodes=1")

    task = _required_mapping(trajectory, "task", "trajectory")
    task_id = _required_string(task, "task_id", "trajectory task")
    if task.get("split") != environment_config.get("split"):
        raise ArtifactValidationError("trajectory split does not match resolved config")
    if trajectory.get("model_version") != manifest.get("model_version"):
        raise ArtifactValidationError("trajectory model version does not match manifest")
    if trajectory.get("seed") != seed_schedule[0]:
        raise ArtifactValidationError("trajectory seed does not match manifest seed schedule")

    metadata = _required_mapping(trajectory, "metadata", "trajectory")
    termination_reason = _required_string(metadata, "termination_reason", "trajectory metadata")
    if termination_reason == "parser_failure":
        raise ArtifactValidationError("trajectory terminated with parser_failure")

    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ArtifactValidationError("trajectory must contain at least one step")

    parser_statuses: set[str] = set()
    saturated_steps = 0
    generated_token_total = 0
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ArtifactValidationError(f"step {index} is not a JSON object")
        parser_status = _required_string(step, "parser_status", f"step {index}")
        parser_statuses.add(parser_status)
        if parser_status != "grounded":
            raise ArtifactValidationError(
                f"step {index} has non-grounded parser status {parser_status!r}"
            )
        if step.get("action_valid") is not True:
            raise ArtifactValidationError(f"step {index} is not marked action_valid=true")
        _required_string(step, "action", f"step {index}")
        if not isinstance(step.get("model_output"), str):
            raise ArtifactValidationError(f"step {index} is missing model_output")
        token_statistics = _required_mapping(
            step, "token_statistics", f"step {index}"
        )
        token_count = token_statistics.get("generated_tokens")
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 1:
            raise ArtifactValidationError(
                f"step {index} has invalid generated_tokens {token_count!r}"
            )
        generated_token_total += token_count
        if token_count >= max_new_tokens:
            saturated_steps += 1

    if saturated_steps >= MAX_REPEATED_SATURATIONS:
        raise ArtifactValidationError(
            f"generation reached the {max_new_tokens}-token cap on "
            f"{saturated_steps} steps"
        )

    invalid_action_rate = _required_number(
        metrics, "invalid_action_rate", "metrics"
    )
    if not 0.0 <= invalid_action_rate <= 1.0:
        raise ArtifactValidationError("invalid_action_rate must be between 0 and 1")
    if invalid_action_rate >= 1.0:
        raise ArtifactValidationError("invalid_action_rate is 1.0")
    success_rate = _required_number(metrics, "success_rate", "metrics")
    if not 0.0 <= success_rate <= 1.0:
        raise ArtifactValidationError("success_rate must be between 0 and 1")
    reward = _required_number(metrics, "mean_reward", "metrics")
    generated_tokens = _required_number(metrics, "mean_generated_tokens", "metrics")
    mean_episode_length = _required_number(metrics, "mean_episode_length", "metrics")
    if mean_episode_length != len(steps):
        raise ArtifactValidationError(
            "metrics mean_episode_length does not match the one-episode trajectory"
        )
    if generated_tokens != generated_token_total:
        raise ArtifactValidationError(
            "metrics mean_generated_tokens does not match the one-episode trajectory"
        )

    first_step = steps[0]
    return BaselineSmokeSummary(
        run_id=run_id,
        git_revision=git_revision,
        task_id=task_id,
        termination_reason=termination_reason,
        steps=len(steps),
        success_rate=success_rate,
        reward=reward,
        invalid_action_rate=invalid_action_rate,
        generated_tokens=generated_tokens,
        parser_statuses=tuple(sorted(parser_statuses)),
        first_raw_output=first_step["model_output"],
        first_parsed_action=first_step["action"],
    )


def print_summary(summary: BaselineSmokeSummary) -> None:
    """Print the approved concise smoke-test summary."""
    print(f"run_id: {summary.run_id}")
    print(f"git_revision: {summary.git_revision}")
    print(f"task_id: {summary.task_id}")
    print(f"termination_reason: {summary.termination_reason}")
    print(f"steps: {summary.steps}")
    print(f"success_rate: {summary.success_rate}")
    print(f"reward: {summary.reward}")
    print(f"invalid_action_rate: {summary.invalid_action_rate}")
    print(f"generated_tokens: {summary.generated_tokens}")
    print(f"parser_statuses: {', '.join(summary.parser_statuses)}")
    print(f"first_raw_output: {summary.first_raw_output!r}")
    print(f"first_parsed_action: {summary.first_parsed_action!r}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ArtifactValidationError(f"cannot read {path.name}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactValidationError(
                f"invalid JSON in {path.name} line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ArtifactValidationError(
                f"{path.name} line {line_number} must contain a JSON object"
            )
        objects.append(value)
    return objects


def _required_mapping(
    mapping: dict[str, Any], key: str, context: str
) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{context} is missing object field {key!r}")
    return value


def _required_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(f"{context} is missing string field {key!r}")
    return value


def _required_number(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArtifactValidationError(f"{context} is missing numeric field {key!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactValidationError(f"{context} field {key!r} must be finite")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a one-episode baseline smoke run."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--expected-git-revision",
        help="Require the manifest to match the wrapper's pre-run Git revision.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the artifact validator CLI and emit one final status marker."""
    arguments = _arguments()
    try:
        summary = validate_baseline_artifacts(
            arguments.run_directory,
            expected_git_revision=arguments.expected_git_revision,
        )
    except Exception as error:
        print(f"validation_error: {error}", file=sys.stderr)
        print("BASELINE_SMOKE_FAIL")
        return 1
    print_summary(summary)
    print("BASELINE_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
