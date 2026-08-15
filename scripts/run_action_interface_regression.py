#!/usr/bin/env python3
"""Run or compare a small matched free-form versus indexed regression suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.regression import (  # noqa: E402
    compare_trajectory_sets,
    render_comparison_markdown,
    validate_interface_config_equivalence,
)
from trajectory.provenance import (  # noqa: E402
    ProvenanceRequirement,
    load_run_trajectories,
)


FREE_CONFIG = PROJECT_ROOT / "configs/collection/baseline.yaml"
INDEXED_CONFIG = PROJECT_ROOT / "configs/collection/baseline_indexed.yaml"


def main() -> int:
    """Dispatch config checking, offline comparison, or a bounded server run."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-configs")

    compare = subparsers.add_parser("compare")
    compare.add_argument("free_form_run", type=Path)
    compare.add_argument("indexed_run", type=Path)
    compare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--episodes", type=int, default=1)
    run.add_argument("--run-name", required=True)
    run.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/runtime/regression",
    )
    arguments = parser.parse_args()
    _validate_configs()
    if arguments.command == "check-configs":
        print("ACTION_INTERFACE_CONFIGS_MATCH")
        return 0
    if arguments.command == "compare":
        _compare(arguments.free_form_run, arguments.indexed_run, arguments.output)
        return 0
    if arguments.episodes < 1 or arguments.episodes > 10:
        raise SystemExit("episodes must be between 1 and 10")
    pair_root = arguments.output_root / arguments.run_name
    if pair_root.exists():
        raise SystemExit(f"output already exists: {pair_root}")
    pair_root.mkdir(parents=True)
    free_run = _collect(FREE_CONFIG, arguments.episodes, pair_root, "free_form")
    indexed_run = _collect(INDEXED_CONFIG, arguments.episodes, pair_root, "indexed")
    _compare(free_run, indexed_run, pair_root / "comparison.json")
    print(f"saved matched regression to {pair_root}")
    return 0


def _validate_configs() -> None:
    free_collection = _read_yaml(FREE_CONFIG)
    indexed_collection = _read_yaml(INDEXED_CONFIG)
    environment = _read_yaml(_resolve(free_collection["environment_config"]))
    if environment != _read_yaml(_resolve(indexed_collection["environment_config"])):
        raise ValueError("environment configs differ")
    validate_interface_config_equivalence(
        free_collection,
        indexed_collection,
        _read_yaml(_resolve(free_collection["model_config"])),
        _read_yaml(_resolve(indexed_collection["model_config"])),
        environment,
    )


def _collect(config: Path, episodes: int, output_root: Path, name: str) -> Path:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/collect_baseline.py"),
        "--config",
        str(config),
        "--episodes",
        str(episodes),
        "--output-dir",
        str(output_root),
        "--run-name",
        name,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return output_root / name


def _compare(free_run: Path, indexed_run: Path, output: Path) -> None:
    reference_manifest, free_trajectories = load_run_trajectories(
        free_run,
        requirement=ProvenanceRequirement(
            pipeline_versions=frozenset({"legacy_v1", "free_form_v1"}),
            action_selection_modes=frozenset({"free_text", "free_form_validated"}),
            splits=frozenset({"valid_seen"}),
        ),
    )
    _, indexed_trajectories = load_run_trajectories(
        indexed_run,
        requirement=ProvenanceRequirement.one(
            pipeline_version="indexed_v1",
            action_selection_mode="indexed_admissible",
            split="valid_seen",
        ),
    )
    report = compare_trajectory_sets(
        free_trajectories,
        indexed_trajectories,
        reference_pipeline=reference_manifest["pipeline_version"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(
        render_comparison_markdown(report), encoding="utf-8"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
