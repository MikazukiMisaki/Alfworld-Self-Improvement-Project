#!/usr/bin/env python3
"""Run the frozen five-prefix Sprint 2B fixed-recovery pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from env.alfworld import AlfWorldTextEnvironment  # noqa: E402
from env.replay import replay_prefix, repeated_replay_equal  # noqa: E402
from recovery.fixed import (  # noqa: E402
    RECOVERY_INSTRUCTIONS,
    RECOVERY_OPERATOR_VERSION,
    RECOVERY_PROMPT_VERSION,
    FixedRecoveryOperator,
)
from recovery.pilot import prefix_hash, run_branch  # noqa: E402
from scripts.collect_baseline import _environment, _policy  # noqa: E402
from trajectory.provenance import (  # noqa: E402
    ProvenanceRequirement,
    load_run_trajectories,
)


DEFAULT_CONFIG = PROJECT_ROOT / (
    "configs/experiments/sprint2b_revised_recovery_development.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite output: {arguments.output}")
    if _git_status():
        raise SystemExit("tracked worktree must be clean before the fixed pilot")

    pilot_config = _read_object(arguments.config)
    source = pilot_config["source"]
    schedule_path = PROJECT_ROOT / source["schedule"]
    run_path = PROJECT_ROOT / source["formal_run"]
    analysis_path = PROJECT_ROOT / source["analysis"]
    _require_hash(schedule_path, source["schedule_sha256"])
    _require_hash(run_path / "trajectory.jsonl", source["trajectory_sha256"])
    _require_hash(analysis_path, source["analysis_sha256"])
    if source.get("old_operator_report"):
        _require_hash(
            PROJECT_ROOT / source["old_operator_report"],
            source["old_operator_report_sha256"],
        )
    schedule = _read_object(schedule_path)
    analysis = _read_object(analysis_path)
    manifest, trajectories = load_run_trajectories(
        run_path,
        requirement=ProvenanceRequirement.one(
            pipeline_version="indexed_bounded_context_v1",
            action_selection_mode="indexed_admissible",
            split="valid_seen",
        ),
    )
    _validate_frozen_configuration(pilot_config, manifest)
    _validate_prefixes(pilot_config, schedule, analysis, trajectories)

    arguments.output.mkdir(parents=True)
    selected_manifest_bytes = arguments.config.read_bytes()
    (arguments.output / "selected_prefix_manifest.json").write_bytes(
        selected_manifest_bytes
    )
    selected_manifest_hash = hashlib.sha256(selected_manifest_bytes).hexdigest()
    run_manifest = {
        "schema_version": "fixed_recovery_pilot_run_v1",
        "created_at": _timestamp(),
        "git_revision": _git_revision(),
        "selected_prefix_manifest_sha256": selected_manifest_hash,
        "source_schedule_sha256": source["schedule_sha256"],
        "source_trajectory_sha256": source["trajectory_sha256"],
        "recovery_operator_version": RECOVERY_OPERATOR_VERSION,
        "recovery_prompt_version": RECOVERY_PROMPT_VERSION,
        "recovery_instructions_sha256": hashlib.sha256(
            RECOVERY_INSTRUCTIONS.encode("utf-8")
        ).hexdigest(),
        "model_inference": True,
        "pair_count": 5,
    }
    _write_json(arguments.output / "run_manifest.json", run_manifest)

    trajectory_by_seed = {int(item["seed"]): item for item in trajectories}
    prefix_by_seed = {
        int(item["seed"]): item for item in pilot_config["prefixes"]
    }
    environment_config = manifest["resolved_config"]["environment"]
    continue_environment = AlfWorldTextEnvironment(
        _environment(environment_config, PROJECT_ROOT)
    )
    recover_environment = AlfWorldTextEnvironment(
        _environment(environment_config, PROJECT_ROOT)
    )
    for seed in schedule["burn_in_seeds"]:
        continue_environment.reset(seed=int(seed))
        recover_environment.reset(seed=int(seed))

    policy = _policy(manifest["resolved_config"]["model"])
    model_load_started = time.perf_counter()
    policy._load()
    model_load_seconds = time.perf_counter() - model_load_started
    recovery = FixedRecoveryOperator(policy)
    pairs: list[dict[str, Any]] = []
    pairs_path = arguments.output / "pairs.jsonl"

    for scheduled in schedule["episodes"]:
        seed = int(scheduled["seed"])
        continue_reset = continue_environment.reset(seed=seed)
        recover_reset = recover_environment.reset(seed=seed)
        expected_task_id = scheduled["task_id"]
        if (
            continue_reset.task.task_id != expected_task_id
            or recover_reset.task.task_id != expected_task_id
        ):
            raise RuntimeError(f"task selection mismatch at schedule seed {seed}")
        if seed not in prefix_by_seed:
            continue

        selected = prefix_by_seed[seed]
        trajectory = trajectory_by_seed[seed]
        action_count = int(selected["replayed_action_count"])
        continue_replay = replay_prefix(
            continue_environment, continue_reset, trajectory, action_count
        )
        recover_replay = replay_prefix(
            recover_environment, recover_reset, trajectory, action_count
        )
        branch_equality = repeated_replay_equal(continue_replay, recover_replay)
        if (
            not continue_replay["exact_public_reconstruction"]
            or not recover_replay["exact_public_reconstruction"]
            or not all(
                value for value in branch_equality.values() if value is not None
            )
        ):
            raise RuntimeError(f"prefix reconstruction mismatch for seed {seed}")

        branch_observation = str(continue_replay["target"]["observation"])
        branch_actions = tuple(
            str(action)
            for action in continue_replay["target"]["admissible_actions"]
        )
        recover_actions = tuple(
            str(action)
            for action in recover_replay["target"]["admissible_actions"]
        )
        if branch_actions != recover_actions:
            raise RuntimeError(f"branch admissible actions differ for seed {seed}")
        remaining_horizon = int(pilot_config["baseline"]["max_episode_steps"]) - action_count

        continue_branch = run_branch(
            continue_environment,
            continue_reset,
            policy,
            trajectory,
            action_count=action_count,
            branch_observation=branch_observation,
            branch_valid_actions=branch_actions,
            remaining_horizon=remaining_horizon,
            first_decision=policy.act,
        )
        recovery_decisions = []

        def recover_once(request):
            decision = recovery.act(request)
            recovery_decisions.append(decision)
            return decision.as_action_decision(policy.model_version)

        recover_branch = run_branch(
            recover_environment,
            recover_reset,
            policy,
            trajectory,
            action_count=action_count,
            branch_observation=str(recover_replay["target"]["observation"]),
            branch_valid_actions=recover_actions,
            remaining_horizon=remaining_horizon,
            first_decision=recover_once,
        )
        if len(recovery_decisions) != 1 or recover_branch["recovery_calls"] != 1:
            raise RuntimeError(f"recovery call-count violation for seed {seed}")
        recovery_decision = recovery_decisions[0]
        raw_value = recover_branch["return"] - continue_branch["return"]
        pair = {
            "task_id": expected_task_id,
            "schedule_index": int(scheduled["schedule_index"]),
            "seed": seed,
            "prefix_step": int(selected["prefix_step"]),
            "replayed_action_count": action_count,
            "prefix_sha256": prefix_hash(trajectory, action_count),
            "reconstruction_provenance": {
                "source_git_revision": manifest["git_revision"],
                "source_schedule_sha256": source["schedule_sha256"],
                "source_trajectory_sha256": source["trajectory_sha256"],
                "selected_prefix_manifest_sha256": selected_manifest_hash,
                "continue_exact": continue_replay["exact_public_reconstruction"],
                "recover_exact": recover_replay["exact_public_reconstruction"],
                "branch_equality": branch_equality,
                "continue_hidden_state": continue_replay["hidden_state"],
                "recover_hidden_state": recover_replay["hidden_state"],
            },
            "branch_validation": {
                "identical_task_identity": continue_reset.task.task_id
                == recover_reset.task.task_id,
                "identical_observation": branch_observation
                == recover_replay["target"]["observation"],
                "identical_admissible_actions": branch_actions == recover_actions,
                "identical_remaining_horizon": continue_branch["remaining_horizon"]
                == recover_branch["remaining_horizon"],
                "same_base_policy_after_first_action": True,
                "recovery_call_count": len(recovery_decisions),
                "fallback_or_repair_used": False,
            },
            "continue_first_action": continue_branch["first_action"],
            "original_frozen_continuation_action": trajectory["steps"][action_count][
                "action"
            ],
            "recovery": {
                "diagnosis": recovery_decision.diagnosis,
                "action_id": recovery_decision.action_id,
                "mapped_command": recovery_decision.action or None,
                "status": recovery_decision.status,
                "failure_reason": recovery_decision.failure_reason,
                "raw_output": recovery_decision.raw_output,
                "generated_tokens": recovery_decision.token_statistics.generated_tokens,
                "input_tokens": recovery_decision.token_statistics.input_tokens,
                "latency_seconds": recovery_decision.latency_seconds,
                "diagnosis_word_count": recovery_decision.diagnosis_word_count,
                "diagnosis_length_valid": recovery_decision.diagnosis_length_valid,
                "output_complete": recovery_decision.output_complete,
                "token_cap_reached": recovery_decision.token_cap_reached,
                "diagnosis_action_consistency": {
                    "annotation": None,
                    "used_as_execution_rule": False,
                },
                "prompt": recovery_decision.prompt,
            },
            "continue": continue_branch,
            "recover": recover_branch,
            "v_raw": raw_value,
            "classification": (
                "beneficial" if raw_value > 0 else "harmful" if raw_value < 0 else "neutral"
            ),
        }
        pairs.append(pair)
        with pairs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(pair, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "seed": seed,
                    "classification": pair["classification"],
                    "continue_return": continue_branch["return"],
                    "recover_return": recover_branch["return"],
                    "recovery_status": recovery_decision.status,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(pairs) != 5:
        raise RuntimeError(f"expected five pairs, produced {len(pairs)}")
    report = {
        **run_manifest,
        "completed_at": _timestamp(),
        "model_load_seconds": model_load_seconds,
        "pairs": pairs,
        "aggregate": _aggregate(pairs),
    }
    _write_json(arguments.output / "report.json", report)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


def _validate_frozen_configuration(
    pilot_config: dict[str, Any], manifest: dict[str, Any]
) -> None:
    baseline = pilot_config["baseline"]
    model = manifest["resolved_config"]["model"]
    expected = {
        "model_id": baseline["model_id"],
        "device": "auto",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "enable_thinking": baseline["enable_thinking"],
        "pipeline_version": baseline["pipeline_version"],
        "action_selection": {"mode": baseline["action_selection_mode"]},
        "history_context": {
            "mode": baseline["history_context_mode"],
            "window": baseline["history_window"],
        },
        "generation": {
            "max_new_tokens": baseline["max_new_tokens"],
            "do_sample": baseline["do_sample"],
            "temperature": None,
            "top_p": None,
        },
    }
    if model != expected:
        raise RuntimeError("formal run does not use the frozen H4 model configuration")
    if manifest["git_revision"] != pilot_config["source"]["baseline_git_revision"]:
        raise RuntimeError("formal run baseline revision does not match pilot config")
    recovery = pilot_config["recovery"]
    if recovery != {
        "operator_version": RECOVERY_OPERATOR_VERSION,
        "prompt_version": RECOVERY_PROMPT_VERSION,
        "model_id": "Qwen/Qwen3-8B",
        "max_calls_per_recover_branch": 1,
        "enable_thinking": False,
        "do_sample": False,
        "max_new_tokens": 32,
        "maximum_diagnosis_words": 12,
        "retry_on_failure": False,
        "fallback_or_repair": False,
    }:
        raise RuntimeError("pilot config does not match the revised recovery operator")
    if (
        pilot_config.get("development_prefixes") is not True
        or pilot_config.get("eligible_as_held_out_evidence") is not False
    ):
        raise RuntimeError("reused recovery prefixes must be marked development-only")


def _validate_prefixes(
    pilot_config: dict[str, Any],
    schedule: dict[str, Any],
    analysis: dict[str, Any],
    trajectories: tuple[dict[str, Any], ...],
) -> None:
    scheduled = {int(item["seed"]): item for item in schedule["episodes"]}
    analyzed = {int(item["seed"]): item for item in analysis["episodes"]}
    recorded = {int(item["seed"]): item for item in trajectories}
    prefixes = pilot_config.get("prefixes", [])
    if len(prefixes) != 5 or len({item["seed"] for item in prefixes}) != 5:
        raise RuntimeError("pilot must contain exactly five unique prefixes")
    for selected in prefixes:
        seed = int(selected["seed"])
        schedule_item = scheduled[seed]
        analysis_item = analyzed[seed]
        trajectory = recorded[seed]
        prefix_step = int(selected["prefix_step"])
        if selected["replayed_action_count"] != prefix_step + 1:
            raise RuntimeError(f"prefix action-count mismatch for seed {seed}")
        if analysis_item.get("recoverable_prefix", {}).get("prefix_step") != prefix_step:
            raise RuntimeError(f"formal annotated prefix differs for seed {seed}")
        expected = (schedule_item["schedule_index"], schedule_item["task_id"])
        configured = (selected["schedule_index"], selected["task_id"])
        recorded_key = (
            trajectory["metadata"]["schedule_index"],
            trajectory["task"]["task_id"],
        )
        if configured != expected or recorded_key != expected:
            raise RuntimeError(f"prefix provenance mismatch for seed {seed}")


def _aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    classes = {name: 0 for name in ("beneficial", "neutral", "harmful")}
    for pair in pairs:
        classes[pair["classification"]] += 1
    latencies = [pair["recovery"]["latency_seconds"] for pair in pairs]
    tokens = [pair["recovery"]["generated_tokens"] for pair in pairs]
    return {
        "pair_count": len(pairs),
        **{f"{name}_count": count for name, count in classes.items()},
        "continue_fail_recover_success_count": sum(
            not pair["continue"]["success"] and pair["recover"]["success"]
            for pair in pairs
        ),
        "continue_success_recover_fail_count": sum(
            pair["continue"]["success"] and not pair["recover"]["success"]
            for pair in pairs
        ),
        "mean_recovery_generated_tokens": sum(tokens) / len(tokens),
        "total_recovery_generated_tokens": sum(tokens),
        "mean_recovery_latency_seconds": sum(latencies) / len(latencies),
        "total_recovery_latency_seconds": sum(latencies),
        "selection_failure_count": sum(
            pair["recovery"]["status"] != "selected" for pair in pairs
        ),
        "malformed_output_count": sum(
            pair["recovery"]["status"]
            in {"malformed_recovery", "truncated_recovery"}
            for pair in pairs
        ),
        "complete_output_count": sum(
            pair["recovery"]["output_complete"] for pair in pairs
        ),
        "diagnosis_length_valid_count": sum(
            pair["recovery"]["diagnosis_length_valid"] for pair in pairs
        ),
        "repeated_continuation_action_count": sum(
            pair["recovery"]["mapped_command"] == pair["continue_first_action"]
            for pair in pairs
        ),
        "value_is_heterogeneous": len({pair["classification"] for pair in pairs}) > 1,
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _require_hash(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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
