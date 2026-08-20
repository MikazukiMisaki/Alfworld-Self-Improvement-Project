#!/usr/bin/env python3
"""Collect the preregistered Sprint 2C frozen-H4 trajectory pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from env.alfworld import AlfWorldTextEnvironment  # noqa: E402
from evaluation.metrics import EvaluationMetrics  # noqa: E402
from recovery.corpus import (  # noqa: E402
    INITIAL_POOL_SIZE,
    MINIMUM_INITIAL_SUCCESSES,
    executed_schedule,
    sha256_file,
    validate_pool_schedule,
)
from scripts.collect_baseline import (  # noqa: E402
    _environment,
    _policy,
    _read_yaml,
    _timestamp,
    _write_json,
)
from trajectory.collector import collect_episode  # noqa: E402
from trajectory.store import JsonlTrajectoryStore  # noqa: E402
from trajectory.trajectory import RunManifest  # noqa: E402


DEFAULT_SCHEDULE = PROJECT_ROOT / "configs/corpus/sprint2c_h4_pool_schedule.json"


class ScheduledEnvironment:
    """Fail immediately if ALFWorld reset order differs from the schedule."""

    def __init__(self, environment: AlfWorldTextEnvironment) -> None:
        self.environment = environment
        self.expected_task_id: str | None = None

    def reset(self, *, seed: int | None = None):
        result = self.environment.reset(seed=seed)
        if result.task.task_id != self.expected_task_id:
            raise RuntimeError(
                f"scheduled task mismatch: expected {self.expected_task_id!r}, "
                f"got {result.task.task_id!r}"
            )
        return result

    def step(self, action: str):
        return self.environment.step(action)

    def get_task(self):
        return self.environment.get_task()

    def get_valid_actions(self):
        return self.environment.get_valid_actions()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite output: {arguments.output}")
    revision = _git_revision()
    if revision != arguments.expected_revision:
        raise SystemExit(f"unexpected Git revision: {revision}")
    if _git_status():
        raise SystemExit("tracked worktree must be clean before H4 inference")

    schedule_path = arguments.schedule.resolve()
    schedule_hash = _verify_checksum(schedule_path)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    validate_pool_schedule(schedule)
    collection = _read_yaml(PROJECT_ROOT / "configs/collection/baseline_indexed_h4.yaml")
    environment_config = _read_yaml(PROJECT_ROOT / collection["environment_config"])
    model_config = _read_yaml(PROJECT_ROOT / collection["model_config"])
    _validate_frozen_h4(model_config, environment_config)

    raw_environment = AlfWorldTextEnvironment(
        _environment(environment_config, PROJECT_ROOT)
    )
    for seed in schedule["burn_in_seeds"]:
        raw_environment.reset(seed=int(seed))
    environment = ScheduledEnvironment(raw_environment)
    policy = _policy(model_config)
    arguments.output.mkdir(parents=True)
    _write_json(
        arguments.output / "collection_plan.json",
        {
            "schema_version": "sprint2c_h4_collection_plan_v1",
            "created_at": _timestamp(),
            "git_revision": revision,
            "schedule_sha256": schedule_hash,
            "initial_episode_count": INITIAL_POOL_SIZE,
            "extension_gate_below_success_count": MINIMUM_INITIAL_SUCCESSES,
            "model_inference": True,
        },
    )
    model_load_started = time.perf_counter()
    policy._load()
    model_load_seconds = time.perf_counter() - model_load_started

    trajectories = []
    timings = []
    store = JsonlTrajectoryStore(arguments.output / "trajectory.jsonl")
    run_started = time.perf_counter()
    initial_success_count = 0
    for item in schedule["episodes"]:
        index = int(item["schedule_index"])
        if index == INITIAL_POOL_SIZE:
            initial_success_count = sum(t.succeeded for t in trajectories)
            if initial_success_count >= MINIMUM_INITIAL_SUCCESSES:
                break
            print(
                json.dumps(
                    {
                        "extension_triggered": True,
                        "initial_success_count": initial_success_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        environment.expected_task_id = item["task_id"]
        episode_started = time.perf_counter()
        trajectory = collect_episode(
            environment,
            policy,
            max_steps=50,
            seed=int(item["seed"]),
        )
        elapsed = time.perf_counter() - episode_started
        trajectory = replace(
            trajectory,
            metadata={
                **trajectory.metadata,
                "schedule_index": index,
                "reset_order_position": item["reset_order_position"],
                "pool_block": item["pool_block"],
                "task_family": item["task_family"],
                "episode_inference_seconds": elapsed,
            },
        )
        store.append(trajectory)
        trajectories.append(trajectory)
        timings.append(elapsed)
        print(
            json.dumps(
                {
                    "schedule_index": index,
                    "seed": item["seed"],
                    "task_id": item["task_id"],
                    "success": trajectory.succeeded,
                    "reward": trajectory.total_reward,
                    "steps": trajectory.episode_length,
                    "termination": trajectory.metadata["termination_reason"],
                    "episode_inference_seconds": round(elapsed, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(trajectories) < INITIAL_POOL_SIZE:
        raise RuntimeError("initial H4 pool did not complete")
    initial_success_count = sum(
        trajectory.succeeded for trajectory in trajectories[:INITIAL_POOL_SIZE]
    )
    expected = executed_schedule(schedule, initial_success_count)
    if len(expected) != len(trajectories):
        raise RuntimeError("conditional extension execution violated the schedule")
    metrics = EvaluationMetrics.from_trajectories(trajectories)
    resolved_collection = dict(collection)
    resolved_collection.update(
        episodes=len(trajectories),
        seed_start=2000,
        max_steps=50,
        output_dir=str(arguments.output),
        schedule_sha256=schedule_hash,
        environment_burn_in_resets=len(schedule["burn_in_seeds"]),
        conditional_extension_executed=len(trajectories) > INITIAL_POOL_SIZE,
    )
    manifest = RunManifest(
        run_id=arguments.output.name,
        created_at=_timestamp(),
        model_version=model_config["model_id"],
        environment="alfworld_text",
        resolved_config={
            "collection": resolved_collection,
            "environment": environment_config,
            "model": model_config,
        },
        seed_schedule=tuple(int(item["seed"]) for item in expected),
        git_revision=revision,
        pipeline_version="indexed_bounded_context_v1",
        action_selection_mode="indexed_admissible",
        split="valid_seen",
        metadata={
            "sprint2c_corpus_source": True,
            "frozen_h4_protocol_commit": schedule["frozen_h4_protocol_commit"],
            "schedule_sha256": schedule_hash,
            "initial_success_count": initial_success_count,
            "conditional_extension_executed": len(trajectories) > INITIAL_POOL_SIZE,
        },
    )
    _write_json(arguments.output / "run_manifest.json", manifest.to_dict())
    _write_json(arguments.output / "metrics.json", metrics.__dict__)
    _write_json(
        arguments.output / "collection_summary.json",
        {
            "schema_version": "sprint2c_h4_collection_summary_v1",
            "completed_at": _timestamp(),
            "episodes": len(trajectories),
            "initial_success_count": initial_success_count,
            "total_success_count": sum(t.succeeded for t in trajectories),
            "extension_executed": len(trajectories) > INITIAL_POOL_SIZE,
            "model_load_seconds": model_load_seconds,
            "total_episode_inference_seconds": sum(timings),
            "total_collection_wall_seconds": time.perf_counter() - run_started,
            "trajectory_sha256": sha256_file(arguments.output / "trajectory.jsonl"),
        },
    )
    print(json.dumps(metrics.__dict__, indent=2, sort_keys=True), flush=True)
    return 0


def _validate_frozen_h4(
    model_config: dict[str, object], environment_config: dict[str, object]
) -> None:
    expected = {
        "model_id": "Qwen/Qwen3-8B",
        "device": "auto",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "enable_thinking": False,
        "pipeline_version": "indexed_bounded_context_v1",
        "action_selection": {"mode": "indexed_admissible"},
        "history_context": {"mode": "bounded_recent_state", "window": 4},
        "generation": {
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        },
    }
    if model_config != expected:
        raise RuntimeError("frozen H4 model configuration changed")
    if environment_config.get("split") != "valid_seen":
        raise RuntimeError("Sprint 2C pool requires valid_seen")


def _verify_checksum(path: Path) -> str:
    fields = path.with_suffix(".sha256").read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError("invalid schedule checksum file")
    actual = sha256_file(path)
    if actual != fields[0]:
        raise RuntimeError("schedule checksum mismatch")
    return actual


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
