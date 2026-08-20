#!/usr/bin/env python3
"""Validate deterministic ALFWorld reconstruction from recorded action prefixes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.alfworld import AlfWorldConfig, AlfWorldTextEnvironment  # noqa: E402
from env.replay import replay_prefix, repeated_replay_equal  # noqa: E402
from trajectory.provenance import (  # noqa: E402
    ProvenanceRequirement,
    load_run_trajectories,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="SEED:PREFIX",
        help="Replay PREFIX actions from the trajectory recorded with SEED.",
    )
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite existing report: {arguments.output}")

    manifest, trajectories = load_run_trajectories(
        arguments.run_dir,
        requirement=ProvenanceRequirement.one(
            pipeline_version="indexed_bounded_context_v1",
            action_selection_mode="indexed_admissible",
            split="valid_seen",
        ),
    )
    schedule = _read_object(arguments.schedule)
    cases = _parse_cases(arguments.case)
    trajectory_by_seed = {int(item["seed"]): item for item in trajectories}
    schedule_episodes = schedule.get("episodes")
    if not isinstance(schedule_episodes, list):
        raise SystemExit("schedule is missing episodes")
    _validate_schedule(manifest, schedule_episodes, trajectory_by_seed)
    for seed in cases:
        if seed not in trajectory_by_seed:
            raise SystemExit(f"seed {seed} is absent from the trajectory run")

    repeats = [
        _run_schedule_pass(manifest, schedule, trajectory_by_seed, cases)
        for _ in range(2)
    ]
    results = []
    for seed, prefix_length in cases.items():
        first = repeats[0][seed]
        second = repeats[1][seed]
        repeat_comparison = repeated_replay_equal(first, second)
        results.append(
            {
                "seed": seed,
                "task_id": first["task_id"],
                "prefix_length": prefix_length,
                "first_replay": first,
                "second_replay": second,
                "repeat_comparison": repeat_comparison,
                "accepted": first["exact_public_reconstruction"]
                and second["exact_public_reconstruction"]
                and all(
                    value
                    for value in repeat_comparison.values()
                    if value is not None
                ),
            }
        )

    report = {
        "schema_version": "prefix_replay_validation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(arguments.run_dir.resolve()),
        "source_git_revision": manifest["git_revision"],
        "validator_git_revision": _git_revision(),
        "pipeline_version": manifest["pipeline_version"],
        "split": manifest["split"],
        "burn_in_seeds": schedule.get("burn_in_seeds", []),
        "model_inference_used": False,
        "state_contract": {
            "original_public_state": [
                "task_id",
                "observation",
                "ordered_admissible_actions",
                "admissible_action_set",
                "reward",
                "done",
                "truncated",
            ],
            "hidden_state_note": (
                "The source trajectories did not record hidden state. Hidden-state "
                "hashes therefore test equality between two independent replays only."
            ),
        },
        "cases": results,
        "aggregate": _aggregate(results),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], sort_keys=True))
    print(f"saved replay validation to {arguments.output}")
    return 0 if report["aggregate"]["all_accepted"] else 1


def _run_schedule_pass(
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    trajectory_by_seed: dict[int, dict[str, Any]],
    cases: dict[int, int],
) -> dict[int, dict[str, Any]]:
    environment = AlfWorldTextEnvironment(_environment_config(manifest))
    for seed in schedule.get("burn_in_seeds", []):
        environment.reset(seed=int(seed))
    results: dict[int, dict[str, Any]] = {}
    for episode in schedule["episodes"]:
        seed = int(episode["seed"])
        reset = environment.reset(seed=seed)
        if reset.task.task_id != episode["task_id"]:
            raise RuntimeError(
                f"schedule task mismatch for seed {seed}: "
                f"{reset.task.task_id!r} != {episode['task_id']!r}"
            )
        if seed in cases:
            results[seed] = replay_prefix(
                environment, reset, trajectory_by_seed[seed], cases[seed]
            )
        if len(results) == len(cases):
            break
    return results


def _environment_config(manifest: dict[str, Any]) -> AlfWorldConfig:
    config = manifest["resolved_config"]["environment"]
    return AlfWorldConfig(
        config_path=Path(config["config_path"]),
        split=str(config["split"]),
        batch_size=int(config["batch_size"]),
        data_path=Path(config["data_path"]) if config.get("data_path") else None,
    )


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    accepted = sum(result["accepted"] for result in results)
    observations = sum(
        result["first_replay"]["target"]["observation_equal"]
        and result["second_replay"]["target"]["observation_equal"]
        for result in results
    )
    action_sets = sum(
        result["first_replay"]["target"]["admissible_set_equal"]
        and result["second_replay"]["target"]["admissible_set_equal"]
        for result in results
    )
    rewards = sum(
        result["first_replay"]["histories"]["reward_history_equal"]
        and result["second_replay"]["histories"]["reward_history_equal"]
        for result in results
    )
    terminal = sum(
        result["first_replay"]["histories"]["done_history_equal"]
        and result["second_replay"]["histories"]["done_history_equal"]
        and result["first_replay"]["histories"]["truncated_history_equal"]
        and result["second_replay"]["histories"]["truncated_history_equal"]
        for result in results
    )
    task_selection = sum(
        result["first_replay"]["task_id_equal"]
        and result["second_replay"]["task_id_equal"]
        and result["repeat_comparison"]["task_id_equal"]
        for result in results
    )
    discrepancies = [
        result["seed"]
        for result in results
        if not all(
            value
            for value in result["repeat_comparison"].values()
            if value is not None
        )
    ]
    return {
        "case_count": count,
        "accepted_count": accepted,
        "exact_reconstruction_success_rate": accepted / count,
        "observation_equality_count": observations,
        "admissible_action_set_equality_count": action_sets,
        "reward_history_equality_count": rewards,
        "terminal_truncation_equality_count": terminal,
        "deterministic_task_selection_count": task_selection,
        "stochastic_discrepancy_seeds": discrepancies,
        "all_accepted": accepted == count,
    }


def _parse_cases(values: list[str]) -> dict[int, int]:
    cases: dict[int, int] = {}
    for value in values:
        try:
            seed_text, prefix_text = value.split(":", 1)
            seed, prefix = int(seed_text), int(prefix_text)
        except ValueError as error:
            raise SystemExit(f"invalid --case {value!r}; expected SEED:PREFIX") from error
        if seed in cases:
            raise SystemExit(f"duplicate case seed: {seed}")
        cases[seed] = prefix
    return cases


def _validate_schedule(
    manifest: dict[str, Any],
    episodes: list[Any],
    trajectory_by_seed: dict[int, dict[str, Any]],
) -> None:
    scheduled_seeds = [int(item["seed"]) for item in episodes]
    if scheduled_seeds != [int(seed) for seed in manifest["seed_schedule"]]:
        raise SystemExit("schedule seeds do not match the run manifest")
    for item in episodes:
        seed = int(item["seed"])
        trajectory = trajectory_by_seed.get(seed)
        if trajectory is None or trajectory.get("task", {}).get("task_id") != item["task_id"]:
            raise SystemExit(f"schedule and trajectory task identity differ for seed {seed}")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
