#!/usr/bin/env python3
"""Collect the frozen 120-state Sprint 2C paired recovery corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from env.alfworld import AlfWorldTextEnvironment  # noqa: E402
from env.replay import replay_prefix, repeated_replay_equal  # noqa: E402
from recovery.corpus import (  # noqa: E402
    DECISION_FEATURES,
    eligible_states,
    sha256_file,
    validate_prefix_manifest,
)
from recovery.pilot import prefix_hash, run_branch  # noqa: E402
from recovery.two_stage import STAGE_ONE_INSTRUCTIONS, STAGE_ONE_PROMPT_VERSION  # noqa: E402
from recovery.two_stage_v2 import (  # noqa: E402
    RECOVERY_OPERATOR_VERSION,
    STAGE_TWO_INSTRUCTIONS,
    STAGE_TWO_PROMPT_VERSION,
    TwoStageRecoveryV2Operator,
)
from scripts.collect_baseline import _environment, _policy  # noqa: E402
from trajectory.provenance import (  # noqa: E402
    ProvenanceRequirement,
    load_run_trajectories,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "configs/corpus/sprint2c_paired_prefix_manifest.json"
RECORD_SCHEMA = PROJECT_ROOT / "configs/corpus/sprint2c_paired_record_schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite output: {arguments.output}")
    if _git_revision() != arguments.expected_revision:
        raise SystemExit("paired collector Git revision does not match preregistration")
    if _git_status():
        raise SystemExit("tracked worktree must be clean before recovery inference")

    manifest_path = arguments.manifest.resolve()
    manifest_hash = _verify_checksum(manifest_path)
    manifest = _read_object(manifest_path)
    validate_prefix_manifest(manifest)
    _validate_record_schema(_read_object(RECORD_SCHEMA))
    source = manifest["source"]
    pool = PROJECT_ROOT / source["h4_pool"]
    _require_hash(pool / "trajectory.jsonl", source["h4_trajectory_sha256"])
    _require_hash(pool / "run_manifest.json", source["h4_run_manifest_sha256"])
    pool_manifest, trajectories_tuple = load_run_trajectories(
        pool,
        requirement=ProvenanceRequirement.one(
            pipeline_version="indexed_bounded_context_v1",
            action_selection_mode="indexed_admissible",
            split="valid_seen",
        ),
    )
    trajectories = list(trajectories_tuple)
    _validate_frozen_h4(pool_manifest)
    _validate_prefixes_against_pool(manifest, trajectories)
    schedule_path = PROJECT_ROOT / "configs/corpus/sprint2c_h4_pool_schedule.json"
    _require_hash(schedule_path, source["h4_schedule_sha256"])
    schedule = _read_object(schedule_path)
    executed_count = len(trajectories)
    schedule_episodes = schedule["episodes"][:executed_count]

    arguments.output.mkdir(parents=True)
    (arguments.output / "selected_prefix_manifest.json").write_bytes(
        manifest_path.read_bytes()
    )
    run_manifest = {
        "schema_version": "sprint2c_paired_corpus_run_v1",
        "created_at": _timestamp(),
        "git_revision": _git_revision(),
        "frozen_recovery_protocol_commit": manifest[
            "frozen_recovery_protocol_commit"
        ],
        "selected_prefix_manifest_sha256": manifest_hash,
        "record_schema_sha256": sha256_file(RECORD_SCHEMA),
        "source_h4_trajectory_sha256": source["h4_trajectory_sha256"],
        "source_h4_schedule_sha256": source["h4_schedule_sha256"],
        "recovery_operator_version": RECOVERY_OPERATOR_VERSION,
        "stage_one_prompt_version": STAGE_ONE_PROMPT_VERSION,
        "stage_two_prompt_version": STAGE_TWO_PROMPT_VERSION,
        "stage_one_instructions_sha256": _text_hash(STAGE_ONE_INSTRUCTIONS),
        "stage_two_instructions_sha256": _text_hash(STAGE_TWO_INSTRUCTIONS),
        "pair_count": 120,
        "model_inference": True,
        "selector_training": False,
        "frozen_base_policy": {
            "model_id": "Qwen/Qwen3-8B",
            "pipeline_version": "indexed_bounded_context_v1",
            "history_window": 4,
            "thinking": False,
            "decoding": "greedy",
            "max_new_tokens": 32,
            "action_selection_mode": "indexed_admissible",
        },
        "frozen_recovery": {
            "stage_one_max_new_tokens": 40,
            "stage_two_max_new_tokens": 16,
            "stage_two_output_contract": "^A\\d{3}$",
            "diagnosis_calls": 1,
            "action_selection_calls": 1,
            "environment_actions": 1,
            "retry_fallback_or_repair": False,
        },
    }
    _write_json(arguments.output / "run_manifest.json", run_manifest)

    environment_config = pool_manifest["resolved_config"]["environment"]
    environments = [
        AlfWorldTextEnvironment(_environment(environment_config, PROJECT_ROOT))
        for _ in range(4)
    ]
    for seed in schedule["burn_in_seeds"]:
        for environment in environments:
            environment.reset(seed=int(seed))
    policy = _policy(pool_manifest["resolved_config"]["model"])
    model_load_started = time.perf_counter()
    policy._load()
    model_load_seconds = time.perf_counter() - model_load_started
    recovery = TwoStageRecoveryV2Operator(policy)

    trajectory_by_schedule = {
        int(item["metadata"]["schedule_index"]): item for item in trajectories
    }
    prefixes_by_schedule: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prefix in manifest["prefixes"]:
        prefixes_by_schedule[int(prefix["schedule_index"])].append(prefix)
    for values in prefixes_by_schedule.values():
        values.sort(key=lambda item: int(item["action_count"]))

    pairs = []
    pairs_path = arguments.output / "pairs.jsonl"
    for scheduled in schedule_episodes:
        resets = [
            environment.reset(seed=int(scheduled["seed"]))
            for environment in environments
        ]
        if any(reset.task.task_id != scheduled["task_id"] for reset in resets):
            raise RuntimeError(
                f"task selection mismatch at schedule {scheduled['schedule_index']}"
            )
        scheduled_prefixes = prefixes_by_schedule.get(
            int(scheduled["schedule_index"]), []
        )
        if not scheduled_prefixes:
            continue
        trajectory = trajectory_by_schedule[int(scheduled["schedule_index"])]
        for slot, prefix in enumerate(scheduled_prefixes):
            continue_environment = environments[slot * 2]
            recover_environment = environments[slot * 2 + 1]
            continue_reset = resets[slot * 2]
            recover_reset = resets[slot * 2 + 1]
            pair = _run_pair(
                prefix,
                trajectory,
                continue_environment,
                recover_environment,
                continue_reset,
                recover_reset,
                policy,
                recovery,
                source,
                manifest_hash,
            )
            pairs.append(pair)
            with pairs_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(pair, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "prefix_id": pair["prefix_id"],
                        "seed": pair["seed"],
                        "classification": pair["classification"],
                        "continue_return": pair["continue_return"],
                        "recover_return": pair["recover_return"],
                        "stage_one_status": pair["recovery"]["stage_one"]["status"],
                        "stage_two_status": pair["recovery"]["stage_two"][
                            "selection_status"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(pairs) != 120:
        raise RuntimeError(f"expected 120 paired states, collected {len(pairs)}")
    report = {
        **run_manifest,
        "completed_at": _timestamp(),
        "model_load_seconds": model_load_seconds,
        "pairs": pairs,
        "aggregate": aggregate_pairs(pairs),
    }
    _write_json(arguments.output / "report.json", report)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)
    return 0


def _run_pair(
    prefix: dict[str, Any],
    trajectory: dict[str, Any],
    continue_environment: Any,
    recover_environment: Any,
    continue_reset: Any,
    recover_reset: Any,
    policy: Any,
    recovery: TwoStageRecoveryV2Operator,
    source: dict[str, Any],
    manifest_hash: str,
) -> dict[str, Any]:
    action_count = int(prefix["action_count"])
    continue_replay = replay_prefix(
        continue_environment, continue_reset, trajectory, action_count
    )
    recover_replay = replay_prefix(
        recover_environment, recover_reset, trajectory, action_count
    )
    equality = repeated_replay_equal(continue_replay, recover_replay)
    if (
        not continue_replay["exact_public_reconstruction"]
        or not recover_replay["exact_public_reconstruction"]
        or not all(value is True for value in equality.values())
    ):
        raise RuntimeError(f"prefix reconstruction failed for {prefix['prefix_id']}")
    if prefix_hash(trajectory, action_count) != prefix["prefix_sha256"]:
        raise RuntimeError(f"prefix hash changed for {prefix['prefix_id']}")
    observation = str(continue_replay["target"]["observation"])
    actions = tuple(
        str(value) for value in continue_replay["target"]["admissible_actions"]
    )
    recover_actions = tuple(
        str(value) for value in recover_replay["target"]["admissible_actions"]
    )
    if actions != recover_actions:
        raise RuntimeError(f"admissible actions differ for {prefix['prefix_id']}")
    remaining_horizon = int(prefix["remaining_horizon"])
    continue_branch = run_branch(
        continue_environment,
        continue_reset,
        policy,
        trajectory,
        action_count=action_count,
        branch_observation=observation,
        branch_valid_actions=actions,
        remaining_horizon=remaining_horizon,
        first_decision=policy.act,
    )
    if continue_branch["first_action"] != prefix["original_h4_continue_action"]:
        raise RuntimeError(f"H4 continuation changed for {prefix['prefix_id']}")
    source_suffix = trajectory["steps"][
        action_count : action_count + remaining_horizon
    ]
    source_actions = [str(step["action"]) for step in source_suffix]
    source_return = sum(float(step["reward"]) for step in source_suffix)
    if [step["action"] for step in continue_branch["steps"]] != source_actions:
        raise RuntimeError(f"H4 continuation suffix changed for {prefix['prefix_id']}")
    if float(continue_branch["return"]) != source_return:
        raise RuntimeError(f"H4 continuation return changed for {prefix['prefix_id']}")
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
        raise RuntimeError(f"recovery call count changed for {prefix['prefix_id']}")
    decision = recovery_decisions[0]
    environment_action_count = int(
        bool(recover_branch["steps"])
        and recover_branch["steps"][0]["failure_reason"] is None
    )
    continue_return = float(continue_branch["return"])
    recover_return = float(recover_branch["return"])
    value = recover_return - continue_return
    continue_progress = _progress_record(continue_branch)
    recover_progress = _progress_record(recover_branch)
    return {
        "schema_version": "sprint2c_paired_state_v1",
        "prefix_id": prefix["prefix_id"],
        "episode_group_id": prefix["episode_group_id"],
        "task_id": prefix["task_id"],
        "task_family": prefix["task_family"],
        "schedule_index": prefix["schedule_index"],
        "seed": prefix["seed"],
        "prefix_step": prefix["prefix_step"],
        "action_count": action_count,
        "remaining_horizon": remaining_horizon,
        "prefix_sha256": prefix_hash(trajectory, action_count),
        "state_fingerprint": prefix["state_fingerprint"],
        "sampling_stratum": prefix["sampling_stratum"],
        "stage_bucket": prefix["stage_bucket"],
        "decision_features": prefix["decision_features"],
        "reconstruction_provenance": {
            "source_h4_git_revision": source["h4_pool_git_revision"],
            "source_h4_trajectory_sha256": source["h4_trajectory_sha256"],
            "selected_prefix_manifest_sha256": manifest_hash,
            "continue_exact": continue_replay["exact_public_reconstruction"],
            "recover_exact": recover_replay["exact_public_reconstruction"],
            "branch_equality": equality,
            "continue_hidden_state": continue_replay["hidden_state"],
            "recover_hidden_state": recover_replay["hidden_state"],
        },
        "branch_validation": {
            "identical_task_identity": continue_reset.task.task_id
            == recover_reset.task.task_id,
            "identical_observation": observation
            == recover_replay["target"]["observation"],
            "identical_admissible_actions": actions == recover_actions,
            "identical_remaining_horizon": continue_branch["remaining_horizon"]
            == recover_branch["remaining_horizon"],
            "same_base_policy_after_first_action": True,
            "recovery_intervention_count": len(recovery_decisions),
            "recovery_model_call_count": decision.model_call_count,
            "recovery_environment_action_count": environment_action_count,
            "fallback_retry_or_repair_used": False,
            "continue_actions_match_source_suffix": True,
            "continue_return_matches_source_suffix": True,
        },
        "continue_return": continue_return,
        "recover_return": recover_return,
        "v_raw": value,
        "classification": (
            "beneficial" if value > 0 else "harmful" if value < 0 else "neutral"
        ),
        "continue_failure_label": 1 - continue_return,
        "continue": continue_branch,
        "recover": recover_branch,
        "recovery": {
            "status": decision.status,
            "failure_reason": decision.failure_reason,
            "stage_one": _stage_one_record(decision.stage_one),
            "stage_two": _stage_two_record(decision.stage_two),
            "cost": _cost_record(decision),
        },
        "post_run_analysis": {
            "continue_progress": continue_progress,
            "recover_progress": recover_progress,
            "loop_change": _loop_change(continue_branch, recover_branch),
        },
        "qualitative_annotation": None,
    }


def aggregate_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate labels, protocol reliability, costs, and fixed groupings."""
    stage_one = [item["recovery"]["stage_one"] for item in pairs]
    stage_two = [item["recovery"]["stage_two"] for item in pairs]
    costs = [item["recovery"]["cost"] for item in pairs]
    return {
        "pair_count": len(pairs),
        "protocol": {
            "reconstruction_success_count": sum(
                item["reconstruction_provenance"]["continue_exact"]
                and item["reconstruction_provenance"]["recover_exact"]
                for item in pairs
            ),
            "stage_one_valid_count": sum(item["status"] == "valid" for item in stage_one),
            "stage_one_incomplete_count": sum(
                not item["output_complete"] for item in stage_one
            ),
            "stage_one_token_cap_reached_count": sum(
                item["token_cap_reached"] for item in stage_one
            ),
            "stage_two_selected_count": sum(
                item["selection_status"] == "selected" for item in stage_two
            ),
            "stage_two_malformed_count": sum(
                item["selection_status"] == "malformed_id" for item in stage_two
            ),
            "stage_two_out_of_range_count": sum(
                item["selection_status"] == "out_of_range_id" for item in stage_two
            ),
            "stage_two_incomplete_count": sum(
                not item["output_complete"] for item in stage_two
            ),
            "mapping_failure_count": sum(
                item["recovery"]["status"] == "selected"
                and item["branch_validation"]["recovery_environment_action_count"]
                != 1
                for item in pairs
            ),
            "recovery_action_not_executed_count": sum(
                item["branch_validation"]["recovery_environment_action_count"] != 1
                for item in pairs
            ),
        },
        "labels": _label_summary(pairs),
        "cost": {
            "mean_stage_one_input_tokens": _mean(
                item["stage_one_input_tokens"] for item in costs
            ),
            "mean_stage_one_generated_tokens": _mean(
                item["stage_one_generated_tokens"] for item in costs
            ),
            "mean_stage_one_latency_seconds": _mean(
                item["stage_one_latency_seconds"] for item in costs
            ),
            "mean_stage_two_input_tokens": _mean(
                item["stage_two_input_tokens"] for item in costs
            ),
            "mean_stage_two_generated_tokens": _mean(
                item["stage_two_generated_tokens"] for item in costs
            ),
            "mean_stage_two_latency_seconds": _mean(
                item["stage_two_latency_seconds"] for item in costs
            ),
            "mean_total_input_tokens": _mean(item["total_input_tokens"] for item in costs),
            "mean_total_generated_tokens": _mean(
                item["total_generated_tokens"] for item in costs
            ),
            "mean_total_token_count": _mean(item["total_token_count"] for item in costs),
            "mean_sequential_latency_seconds": _mean(
                item["sequential_latency_seconds"] for item in costs
            ),
        },
        "by_sampling_stratum": _grouped_labels(pairs, "sampling_stratum"),
        "by_task_family": _grouped_labels(pairs, "task_family"),
        "by_stage_bucket": _grouped_labels(pairs, "stage_bucket"),
        "by_loop_status": _grouped_labels(
            pairs,
            lambda item: (
                "past_loop"
                if item["decision_features"]["recent_adjacent_repeat_indicator"]
                or item["decision_features"]["recent_two_cycle_indicator"]
                else "non_loop"
            ),
        ),
        "by_entropy_quantile": _grouped_labels(
            pairs, lambda item: item["decision_features"]["entropy_quantile"]
        ),
        "by_continue_outcome": _grouped_labels(
            pairs,
            lambda item: "continue_success"
            if item["continue_return"] > 0
            else "continue_failure",
        ),
        "loop_changes": dict(
            sorted(Counter(item["post_run_analysis"]["loop_change"] for item in pairs).items())
        ),
    }


def _label_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(item["classification"] for item in pairs)
    histogram = Counter(str(item["v_raw"]) for item in pairs)
    return {
        "beneficial_count": classes["beneficial"],
        "neutral_count": classes["neutral"],
        "harmful_count": classes["harmful"],
        "v_raw_histogram": dict(sorted(histogram.items())),
        "mean_v_raw": _mean(item["v_raw"] for item in pairs),
        "continue_fail_recover_success_count": sum(
            item["continue_return"] == 0 and item["recover_return"] > 0
            for item in pairs
        ),
        "continue_success_recover_fail_count": sum(
            item["continue_return"] > 0 and item["recover_return"] == 0
            for item in pairs
        ),
    }


def _grouped_labels(pairs: list[dict[str, Any]], key: Any) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pairs:
        value = key(item) if callable(key) else item[key]
        grouped[str(value)].append(item)
    return {
        name: {"count": len(values), **_label_summary(values)}
        for name, values in sorted(grouped.items())
    }


def _stage_one_record(stage: Any) -> dict[str, Any]:
    return {
        "prompt": stage.generation.prompt,
        "raw_output": stage.generation.raw_output,
        "diagnosis": stage.diagnosis,
        "subgoal": stage.subgoal,
        "status": stage.status,
        "failure_reason": stage.failure_reason,
        "diagnosis_word_count": stage.diagnosis_word_count,
        "subgoal_word_count": stage.subgoal_word_count,
        **_generation_record(stage.generation),
    }


def _stage_two_record(stage: Any) -> dict[str, Any]:
    return {
        "prompt": stage.generation.prompt,
        "raw_output": stage.generation.raw_output,
        "action_id": stage.action_id,
        "mapped_command": stage.action or None,
        "selection_status": stage.status,
        "failure_reason": stage.failure_reason,
        "id_to_command": stage.id_to_command,
        **_generation_record(stage.generation),
    }


def _generation_record(generation: Any) -> dict[str, Any]:
    return {
        "input_tokens": generation.token_statistics.input_tokens,
        "generated_tokens": generation.token_statistics.generated_tokens,
        "latency_seconds": generation.latency_seconds,
        "output_complete": generation.output_complete,
        "token_cap_reached": generation.token_cap_reached,
        "max_new_tokens": generation.max_new_tokens,
    }


def _cost_record(decision: Any) -> dict[str, Any]:
    one = decision.stage_one.generation
    two = decision.stage_two.generation
    one_input = int(one.token_statistics.input_tokens or 0)
    two_input = int(two.token_statistics.input_tokens or 0)
    one_generated = int(one.token_statistics.generated_tokens)
    two_generated = int(two.token_statistics.generated_tokens)
    return {
        "stage_one_input_tokens": one_input,
        "stage_one_generated_tokens": one_generated,
        "stage_one_latency_seconds": one.latency_seconds,
        "stage_two_input_tokens": two_input,
        "stage_two_generated_tokens": two_generated,
        "stage_two_latency_seconds": two.latency_seconds,
        "total_input_tokens": one_input + two_input,
        "total_generated_tokens": one_generated + two_generated,
        "total_token_count": one_input + two_input + one_generated + two_generated,
        "sequential_latency_seconds": one.latency_seconds + two.latency_seconds,
    }


def _progress_record(branch: dict[str, Any]) -> dict[str, Any]:
    steps = branch["steps"]
    observations = [str(step["observation"]) for step in steps]
    rewards = [float(step["reward"]) for step in steps]
    return {
        "success": branch["success"],
        "return": branch["return"],
        "distinct_observation_count": len(set(observations)),
        "observation_change_count": sum(
            first != second for first, second in zip(observations, observations[1:])
        ),
        "first_positive_reward_offset": next(
            (index for index, reward in enumerate(rewards) if reward > 0), None
        ),
    }


def _loop_change(continue_branch: dict[str, Any], recover_branch: dict[str, Any]) -> str:
    def severity(branch: dict[str, Any]) -> tuple[int, int]:
        loops = branch["loop_indicators"]
        return int(loops["two_cycle_events"]), int(loops["adjacent_repeat_events"])

    continued = severity(continue_branch)
    recovered = severity(recover_branch)
    if recovered < continued:
        return "reduced"
    if recovered > continued:
        return "increased"
    return "unchanged"


def _validate_frozen_h4(manifest: dict[str, Any]) -> None:
    model = manifest["resolved_config"]["model"]
    if model != {
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
    }:
        raise RuntimeError("source pool does not use frozen H4")


def _validate_record_schema(schema: dict[str, Any]) -> None:
    if schema.get("schema_version") != "sprint2c_paired_record_schema_v1":
        raise RuntimeError("paired record schema version changed")
    if schema.get("record_schema") != "sprint2c_paired_state_v1":
        raise RuntimeError("paired state schema version changed")
    if schema.get("group_split_key") != "episode_group_id":
        raise RuntimeError("episode grouping key changed")
    if set(schema.get("selector_feature_allowlist", [])) != DECISION_FEATURES:
        raise RuntimeError("selector feature allowlist changed")


def _validate_prefixes_against_pool(
    manifest: dict[str, Any], trajectories: list[dict[str, Any]]
) -> None:
    states = {
        state["prefix_sha256"]: state
        for trajectory in trajectories
        for state in eligible_states(trajectory)
    }
    for prefix in manifest["prefixes"]:
        state = states.get(prefix["prefix_sha256"])
        if state is None:
            raise RuntimeError(f"missing source state {prefix['prefix_id']}")
        for key in (
            "episode_group_id",
            "schedule_index",
            "seed",
            "task_id",
            "task_family",
            "action_count",
            "remaining_horizon",
            "state_fingerprint",
            "original_h4_continue_action",
            "valid_actions",
        ):
            if prefix[key] != state[key]:
                raise RuntimeError(
                    f"source state field {key} changed for {prefix['prefix_id']}"
                )
        if not _decision_features_equal(
            prefix["decision_features"],
            state["decision_features"],
            manifest["selection_protocol"]["entropy_grouping_thresholds"],
        ):
            raise RuntimeError(
                f"source state field decision_features changed for {prefix['prefix_id']}"
            )


def _decision_features_equal(
    selected: dict[str, Any],
    source: dict[str, Any],
    thresholds: dict[str, float],
) -> bool:
    source_with_quantile = dict(source)
    entropy = float(source_with_quantile["h4_decision_token_entropy"])
    if entropy < thresholds["q25"]:
        quantile = "Q1"
    elif entropy < thresholds["q50"]:
        quantile = "Q2"
    elif entropy < thresholds["q75"]:
        quantile = "Q3"
    else:
        quantile = "Q4"
    source_with_quantile["entropy_quantile"] = quantile
    return selected == source_with_quantile


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _verify_checksum(path: Path) -> str:
    fields = path.with_suffix(".sha256").read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError("invalid prefix-manifest checksum file")
    _require_hash(path, fields[0])
    return fields[0]


def _require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: Any) -> float:
    collected = list(values)
    return sum(collected) / len(collected)


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _git_status() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
