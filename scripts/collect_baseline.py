#!/usr/bin/env python3
"""Collect reproducible ALFWorld baseline trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow direct script execution from the repository root without an editable
# package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.alfworld import AlfWorldConfig, AlfWorldTextEnvironment
from evaluation.evaluator import evaluate
from models.policy import GenerationOptions
from models.qwen import QwenPolicy, QwenPolicyConfig
from trajectory.store import JsonlTrajectoryStore
from trajectory.provenance import pipeline_for_action_selection_mode
from trajectory.trajectory import RunManifest


def main() -> None:
    """Parse configuration, collect episodes, and persist manifest and JSONL."""
    arguments = _arguments()
    project_root = Path(__file__).resolve().parents[1]
    collection_config = _read_yaml(_resolve(project_root, arguments.config))
    if arguments.episodes is not None:
        collection_config["episodes"] = arguments.episodes
    if arguments.output_dir is not None:
        collection_config["output_dir"] = arguments.output_dir

    environment_config = _read_yaml(_resolve(project_root, collection_config["environment_config"]))
    model_config = _read_yaml(_resolve(project_root, collection_config["model_config"]))
    environment = _environment(environment_config, project_root)
    policy = _policy(model_config)
    action_selection_mode = policy.action_selection_mode
    pipeline_version = str(model_config.get("pipeline_version", ""))
    context_config = model_config.get("history_context", {})
    context_mode = str(context_config.get("mode", "full_raw"))
    expected_pipeline = pipeline_for_action_selection_mode(
        action_selection_mode, context_mode
    )
    if pipeline_version != expected_pipeline:
        raise ValueError(
            "model pipeline_version must match action_selection.mode: "
            f"expected {expected_pipeline!r}, got {pipeline_version!r}"
        )
    episodes = int(collection_config["episodes"])
    seed_start = int(collection_config["seed_start"])
    seeds = tuple(seed_start + index for index in range(episodes))
    run_directory = _run_directory(project_root, collection_config["output_dir"], arguments.run_name)
    manifest = RunManifest(
        run_id=run_directory.name,
        created_at=_timestamp(),
        model_version=policy.model_version,
        environment="alfworld_text",
        resolved_config={
            "collection": collection_config,
            "environment": environment_config,
            "model": model_config,
        },
        seed_schedule=seeds,
        git_revision=_git_revision(project_root),
        pipeline_version=pipeline_version,
        action_selection_mode=action_selection_mode,
        split=environment.split,
    )
    _write_json(run_directory / "run_manifest.json", manifest.to_dict())

    report = evaluate(
        environment_factory=lambda: AlfWorldTextEnvironment(environment),
        policy=policy,
        seeds=seeds,
        max_steps=int(collection_config["max_steps"]),
    )
    store = JsonlTrajectoryStore(run_directory / "trajectory.jsonl")
    for trajectory in report.trajectories:
        store.append(trajectory)
    _write_json(run_directory / "metrics.json", report.metrics.__dict__)
    print(json.dumps(report.metrics.__dict__, indent=2, sort_keys=True))
    print(f"Saved {len(report.trajectories)} trajectories to {run_directory}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a Qwen3-8B ALFWorld baseline.")
    parser.add_argument("--config", default="configs/collection/baseline.yaml")
    parser.add_argument("--episodes", type=int, help="Override the configured episode count.")
    parser.add_argument("--output-dir", help="Override the configured output directory.")
    parser.add_argument("--run-name", help="Optional unique output subdirectory name.")
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Configuration loading requires PyYAML.") from error
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping in {path}")
    return _expand_environment(data)


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _resolve(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _environment(config: dict[str, Any], project_root: Path) -> AlfWorldConfig:
    config_path = str(config.get("config_path", ""))
    if not config_path or "$" in config_path:
        raise ValueError("set ALFWORLD_CONFIG_PATH to an ALFWorld base_config.yaml path")
    data_path = str(config.get("data_path", ""))
    return AlfWorldConfig(
        config_path=_resolve(project_root, config_path),
        data_path=_resolve(project_root, data_path) if data_path and "$" not in data_path else None,
        split=str(config.get("split", "valid_seen")),
        batch_size=int(config.get("batch_size", 1)),
    )


def _policy(config: dict[str, Any]) -> QwenPolicy:
    generation_config = config.get("generation", {})
    generation = GenerationOptions(
        max_new_tokens=int(generation_config.get("max_new_tokens", 64)),
        do_sample=bool(generation_config.get("do_sample", False)),
        temperature=generation_config.get("temperature"),
        top_p=generation_config.get("top_p"),
    )
    context_config = config.get("history_context", {})
    return QwenPolicy(
        QwenPolicyConfig(
            model_id=str(config["model_id"]),
            device=str(config.get("device", "auto")),
            dtype=str(config.get("dtype", "bfloat16")),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            enable_thinking=bool(config.get("enable_thinking", False)),
            action_selection_mode=str(
                config.get("action_selection", {}).get(
                    "mode", "free_form_validated"
                )
            ),
            history_context_mode=str(context_config.get("mode", "full_raw")),
            history_window=(
                int(context_config["window"])
                if context_config.get("window") is not None
                else None
            ),
            generation=generation,
        )
    )


def _run_directory(project_root: Path, output_dir: str, run_name: str | None) -> Path:
    name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _resolve(project_root, output_dir) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
