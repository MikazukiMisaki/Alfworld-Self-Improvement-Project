#!/usr/bin/env python3
"""Run the frozen ten-prefix R2 two-stage recovery development pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from env.alfworld import AlfWorldTextEnvironment  # noqa: E402
from env.replay import replay_prefix, repeated_replay_equal  # noqa: E402
from recovery.pilot import prefix_hash, run_branch  # noqa: E402
from recovery.two_stage import (  # noqa: E402
    STAGE_ONE_INSTRUCTIONS,
    STAGE_ONE_MAX_NEW_TOKENS,
    STAGE_ONE_PROMPT_VERSION,
    STAGE_TWO_MAX_NEW_TOKENS,
    GenerationRecord,
)
from recovery.two_stage import (  # noqa: E402
    RECOVERY_OPERATOR_VERSION as V1_OPERATOR_VERSION,
)
from recovery.two_stage import STAGE_TWO_INSTRUCTIONS as V1_STAGE_TWO_INSTRUCTIONS  # noqa: E402
from recovery.two_stage import STAGE_TWO_PROMPT_VERSION as V1_STAGE_TWO_PROMPT_VERSION  # noqa: E402
from recovery.two_stage import TwoStageRecoveryOperator as V1_OPERATOR  # noqa: E402
from recovery.two_stage_v2 import (  # noqa: E402
    RECOVERY_OPERATOR_VERSION as V2_OPERATOR_VERSION,
)
from recovery.two_stage_v2 import STAGE_TWO_INSTRUCTIONS as V2_STAGE_TWO_INSTRUCTIONS  # noqa: E402
from recovery.two_stage_v2 import STAGE_TWO_PROMPT_VERSION as V2_STAGE_TWO_PROMPT_VERSION  # noqa: E402
from recovery.two_stage_v2 import TwoStageRecoveryV2Operator as V2_OPERATOR  # noqa: E402
from scripts.collect_baseline import _environment, _policy  # noqa: E402
from trajectory.provenance import (  # noqa: E402
    ProvenanceRequirement,
    load_run_trajectories,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/r2_two_stage_development.json"
EXPECTED_CATEGORIES = {
    "strict_abab_loop",
    "adjacent_repeat_stall",
    "wrong_object_off_target",
    "wrong_tool_or_state_progression",
    "ordinary_non_loop_failure",
}
V1_FORBIDDEN_SEEDS = {1005, 1009, 1010, 1022, 1027}
V2_FORBIDDEN_SEEDS = V1_FORBIDDEN_SEEDS | {
    1000,
    1001,
    1002,
    1004,
    1007,
    1011,
    1013,
    1018,
    1019,
    1024,
}


@dataclass(frozen=True)
class RecoveryProtocol:
    run_schema_version: str
    operator_version: str
    stage_two_prompt_version: str
    stage_two_instructions: str
    operator_class: type
    forbidden_seeds: set[int]
    output_contract: str
    parser_regex: str | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite output: {arguments.output}")
    if _git_status():
        raise SystemExit("tracked worktree must be clean before the R2 pilot")

    config_path = arguments.config.resolve()
    manifest_hash = _verify_frozen_manifest(config_path)
    pilot_config = _read_object(config_path)
    protocol = _protocol(pilot_config)
    source = pilot_config["source"]
    schedule_path = PROJECT_ROOT / source["schedule"]
    run_path = PROJECT_ROOT / source["formal_run"]
    analysis_path = PROJECT_ROOT / source["analysis"]
    _require_hash(schedule_path, source["schedule_sha256"])
    _require_hash(run_path / "trajectory.jsonl", source["trajectory_sha256"])
    _require_hash(analysis_path, source["analysis_sha256"])
    for report_key in ("joint_v2_report", "r2_v1_report"):
        if report_key in source:
            _require_hash(
                PROJECT_ROOT / source[report_key],
                source[f"{report_key}_sha256"],
            )
    schedule = _read_object(schedule_path)
    analysis = _read_object(analysis_path)
    baseline_manifest, trajectories = load_run_trajectories(
        run_path,
        requirement=ProvenanceRequirement.one(
            pipeline_version="indexed_bounded_context_v1",
            action_selection_mode="indexed_admissible",
            split="valid_seen",
        ),
    )
    _validate_frozen_configuration(pilot_config, baseline_manifest, protocol)
    _validate_prefixes(pilot_config, schedule, analysis, trajectories, protocol)

    arguments.output.mkdir(parents=True)
    manifest_bytes = config_path.read_bytes()
    (arguments.output / "selected_prefix_manifest.json").write_bytes(manifest_bytes)
    run_manifest = {
        "schema_version": protocol.run_schema_version,
        "created_at": _timestamp(),
        "protocol_git_revision": _git_revision(),
        "selected_prefix_manifest_sha256": manifest_hash,
        "source_schedule_sha256": source["schedule_sha256"],
        "source_trajectory_sha256": source["trajectory_sha256"],
        "recovery_operator_version": protocol.operator_version,
        "stage_one_prompt_version": STAGE_ONE_PROMPT_VERSION,
        "stage_two_prompt_version": protocol.stage_two_prompt_version,
        "stage_one_instructions_sha256": _text_hash(STAGE_ONE_INSTRUCTIONS),
        "stage_two_instructions_sha256": _text_hash(protocol.stage_two_instructions),
        "stage_one_max_new_tokens": STAGE_ONE_MAX_NEW_TOKENS,
        "stage_two_max_new_tokens": STAGE_TWO_MAX_NEW_TOKENS,
        "model_inference": True,
        "pair_count": 10,
        "development_only": True,
    }
    _write_json(arguments.output / "run_manifest.json", run_manifest)

    trajectory_by_seed = {int(item["seed"]): item for item in trajectories}
    prefix_by_seed = {int(item["seed"]): item for item in pilot_config["prefixes"]}
    environment_config = baseline_manifest["resolved_config"]["environment"]
    continue_environment = AlfWorldTextEnvironment(
        _environment(environment_config, PROJECT_ROOT)
    )
    recover_environment = AlfWorldTextEnvironment(
        _environment(environment_config, PROJECT_ROOT)
    )
    for seed in schedule["burn_in_seeds"]:
        continue_environment.reset(seed=int(seed))
        recover_environment.reset(seed=int(seed))

    policy = _policy(baseline_manifest["resolved_config"]["model"])
    model_load_started = time.perf_counter()
    policy._load()
    model_load_seconds = time.perf_counter() - model_load_started
    recovery = protocol.operator_class(policy)
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
            or not all(value for value in branch_equality.values() if value is not None)
        ):
            raise RuntimeError(f"prefix reconstruction mismatch for seed {seed}")

        branch_observation = str(continue_replay["target"]["observation"])
        branch_actions = tuple(
            str(action) for action in continue_replay["target"]["admissible_actions"]
        )
        recover_actions = tuple(
            str(action) for action in recover_replay["target"]["admissible_actions"]
        )
        if branch_actions != recover_actions:
            raise RuntimeError(f"branch admissible actions differ for seed {seed}")
        remaining_horizon = int(selected["remaining_horizon"])

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
        if continue_branch["first_action"] != selected[
            "original_frozen_continuation_action"
        ]:
            raise RuntimeError(f"frozen H4 continuation mismatch for seed {seed}")

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
            raise RuntimeError(f"recovery intervention-count violation for seed {seed}")
        decision = recovery_decisions[0]
        if decision.model_call_count != 2:
            raise RuntimeError(f"recovery model-call violation for seed {seed}")
        environment_action_count = int(
            bool(recover_branch["steps"])
            and recover_branch["steps"][0]["failure_reason"] is None
        )
        if decision.status == "selected" and environment_action_count != 1:
            raise RuntimeError(f"recovery action-count violation for seed {seed}")

        raw_value = recover_branch["return"] - continue_branch["return"]
        pair = {
            "task_id": expected_task_id,
            "task_family": selected["task_family"],
            "schedule_index": int(scheduled["schedule_index"]),
            "seed": seed,
            "prefix_step": int(selected["prefix_step"]),
            "replayed_action_count": action_count,
            "remaining_horizon": remaining_horizon,
            "prefix_sha256": prefix_hash(trajectory, action_count),
            "selection_categories": selected["selection_categories"],
            "selection_reason": selected["selection_reason"],
            "reconstruction_provenance": {
                "source_baseline_git_revision": baseline_manifest["git_revision"],
                "source_schedule_sha256": source["schedule_sha256"],
                "source_trajectory_sha256": source["trajectory_sha256"],
                "selected_prefix_manifest_sha256": manifest_hash,
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
                "recovery_intervention_count": len(recovery_decisions),
                "recovery_model_call_count": decision.model_call_count,
                "recovery_environment_action_count": environment_action_count,
                "fallback_retry_or_repair_used": False,
            },
            "continue": {
                **continue_branch,
                "original_frozen_continuation_action": selected[
                    "original_frozen_continuation_action"
                ],
            },
            "recovery": {
                "status": decision.status,
                "failure_reason": decision.failure_reason,
                "stage_one": _stage_one_record(decision.stage_one),
                "stage_two": _stage_two_record(decision.stage_two),
                "cost": {
                    "stage_one_generated_tokens": decision.stage_one.generation.token_statistics.generated_tokens,
                    "stage_one_latency_seconds": decision.stage_one.generation.latency_seconds,
                    "stage_two_generated_tokens": decision.stage_two.generation.token_statistics.generated_tokens,
                    "stage_two_latency_seconds": decision.stage_two.generation.latency_seconds,
                    "total_generated_tokens": (
                        decision.stage_one.generation.token_statistics.generated_tokens
                        + decision.stage_two.generation.token_statistics.generated_tokens
                    ),
                    "sequential_total_latency_seconds": (
                        decision.stage_one.generation.latency_seconds
                        + decision.stage_two.generation.latency_seconds
                    ),
                },
            },
            "recover": recover_branch,
            "v_raw": raw_value,
            "classification": (
                "beneficial" if raw_value > 0 else "harmful" if raw_value < 0 else "neutral"
            ),
            "post_run_qualitative_annotation": {
                "diagnosis_correctness": None,
                "subgoal_quality": None,
                "diagnosis_subgoal_to_action_agreement": None,
                "downstream_effectiveness": None,
                "used_during_execution": False,
            },
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
                    "stage_one_status": decision.stage_one.status,
                    "stage_two_status": decision.stage_two.status,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(pairs) != 10:
        raise RuntimeError(f"expected ten pairs, produced {len(pairs)}")
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


def _generation_record(generation: GenerationRecord) -> dict[str, Any]:
    return {
        "input_tokens": generation.token_statistics.input_tokens,
        "generated_tokens": generation.token_statistics.generated_tokens,
        "latency_seconds": generation.latency_seconds,
        "output_complete": generation.output_complete,
        "token_cap_reached": generation.token_cap_reached,
        "max_new_tokens": generation.max_new_tokens,
    }


def _aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(pair["classification"] for pair in pairs)
    stage_one = [pair["recovery"]["stage_one"] for pair in pairs]
    stage_two = [pair["recovery"]["stage_two"] for pair in pairs]
    costs = [pair["recovery"]["cost"] for pair in pairs]
    return {
        "pair_count": len(pairs),
        "beneficial_count": classes["beneficial"],
        "neutral_count": classes["neutral"],
        "harmful_count": classes["harmful"],
        "continue_fail_recover_success_count": sum(
            not pair["continue"]["success"] and pair["recover"]["success"]
            for pair in pairs
        ),
        "continue_success_recover_fail_count": sum(
            pair["continue"]["success"] and not pair["recover"]["success"]
            for pair in pairs
        ),
        "stage_one_valid_count": sum(item["status"] == "valid" for item in stage_one),
        "stage_one_complete_count": sum(item["output_complete"] for item in stage_one),
        "stage_one_truncated_count": sum(item["token_cap_reached"] for item in stage_one),
        "stage_two_selected_count": sum(
            item["selection_status"] == "selected" for item in stage_two
        ),
        "stage_two_complete_count": sum(item["output_complete"] for item in stage_two),
        "stage_two_truncated_count": sum(item["token_cap_reached"] for item in stage_two),
        "mean_stage_one_generated_tokens": _mean(
            item["stage_one_generated_tokens"] for item in costs
        ),
        "mean_stage_two_generated_tokens": _mean(
            item["stage_two_generated_tokens"] for item in costs
        ),
        "mean_total_recovery_generated_tokens": _mean(
            item["total_generated_tokens"] for item in costs
        ),
        "total_recovery_generated_tokens": sum(
            item["total_generated_tokens"] for item in costs
        ),
        "mean_sequential_total_latency_seconds": _mean(
            item["sequential_total_latency_seconds"] for item in costs
        ),
        "total_sequential_recovery_latency_seconds": sum(
            item["sequential_total_latency_seconds"] for item in costs
        ),
        "repeated_continuation_action_count": sum(
            pair["recovery"]["stage_two"]["mapped_command"]
            == pair["continue"]["first_action"]
            for pair in pairs
        ),
        "continue_branch_loop_count": sum(
            pair["continue"]["loop_indicators"]["has_adjacent_repeat"]
            or pair["continue"]["loop_indicators"]["has_two_cycle"]
            for pair in pairs
        ),
        "recover_branch_loop_count": sum(
            pair["recover"]["loop_indicators"]["has_adjacent_repeat"]
            or pair["recover"]["loop_indicators"]["has_two_cycle"]
            for pair in pairs
        ),
        "value_is_heterogeneous": len(classes) > 1,
    }


def _protocol(pilot_config: dict[str, Any]) -> RecoveryProtocol:
    operator_version = pilot_config.get("recovery", {}).get("operator_version")
    if operator_version == V1_OPERATOR_VERSION:
        return RecoveryProtocol(
            run_schema_version="r2_two_stage_pilot_run_v1",
            operator_version=V1_OPERATOR_VERSION,
            stage_two_prompt_version=V1_STAGE_TWO_PROMPT_VERSION,
            stage_two_instructions=V1_STAGE_TWO_INSTRUCTIONS,
            operator_class=V1_OPERATOR,
            forbidden_seeds=V1_FORBIDDEN_SEEDS,
            output_contract="Action-ID: Axxx",
        )
    if operator_version == V2_OPERATOR_VERSION:
        return RecoveryProtocol(
            run_schema_version="r2_two_stage_pilot_run_v2",
            operator_version=V2_OPERATOR_VERSION,
            stage_two_prompt_version=V2_STAGE_TWO_PROMPT_VERSION,
            stage_two_instructions=V2_STAGE_TWO_INSTRUCTIONS,
            operator_class=V2_OPERATOR,
            forbidden_seeds=V2_FORBIDDEN_SEEDS,
            output_contract="Axxx",
            parser_regex=r"^A\d{3}$",
        )
    raise RuntimeError(f"unsupported two-stage operator: {operator_version}")


def _validate_frozen_configuration(
    pilot_config: dict[str, Any],
    manifest: dict[str, Any],
    protocol: RecoveryProtocol,
) -> None:
    baseline = pilot_config["baseline"]
    model = manifest["resolved_config"]["model"]
    expected_model = {
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
    if model != expected_model:
        raise RuntimeError("formal run does not use the frozen H4 model configuration")
    if manifest["git_revision"] != pilot_config["source"]["baseline_git_revision"]:
        raise RuntimeError("formal run baseline revision does not match pilot config")
    expected_recovery = {
        "operator_version": protocol.operator_version,
        "model_id": "Qwen/Qwen3-8B",
        "model_calls_per_intervention": 2,
        "environment_actions_per_intervention": 1,
        "enable_thinking": False,
        "do_sample": False,
        "stage_one": {
            "prompt_version": STAGE_ONE_PROMPT_VERSION,
            "max_new_tokens": STAGE_ONE_MAX_NEW_TOKENS,
            "maximum_diagnosis_words": 12,
            "maximum_subgoal_words": 12,
        },
        "stage_two": {
            "prompt_version": protocol.stage_two_prompt_version,
            "max_new_tokens": STAGE_TWO_MAX_NEW_TOKENS,
            "output_contract": protocol.output_contract,
        },
        "retry_on_failure": False,
        "fallback_or_repair": False,
        "oracle_state": False,
        "memory": False,
    }
    if protocol.parser_regex is not None:
        expected_recovery["stage_two"]["parser_regex"] = protocol.parser_regex
        expected_recovery.update(
            {
                "permissive_dual_format_parsing": False,
                "constrained_decoding": False,
                "candidate_logprob_ranking": False,
            }
        )
    if pilot_config["recovery"] != expected_recovery:
        raise RuntimeError("pilot config does not match the frozen R2 operator")
    if (
        pilot_config.get("frozen_before_execution") is not True
        or pilot_config.get("development_prefixes") is not True
        or pilot_config.get("eligible_as_held_out_evidence") is not False
    ):
        raise RuntimeError("R2 prefixes must be frozen development-only evidence")


def _validate_prefixes(
    pilot_config: dict[str, Any],
    schedule: dict[str, Any],
    analysis: dict[str, Any],
    trajectories: tuple[dict[str, Any], ...],
    protocol: RecoveryProtocol,
) -> None:
    scheduled = {int(item["seed"]): item for item in schedule["episodes"]}
    analyzed = {int(item["seed"]): item for item in analysis["episodes"]}
    recorded = {int(item["seed"]): item for item in trajectories}
    prefixes = pilot_config.get("prefixes", [])
    seeds = {int(item["seed"]) for item in prefixes}
    if len(prefixes) != 10 or len(seeds) != 10 or pilot_config["prefix_count"] != 10:
        raise RuntimeError("R2 pilot must contain exactly ten unique prefixes")
    if seeds & protocol.forbidden_seeds:
        raise RuntimeError("R2 manifest reuses a prior recovery-development seed")
    if set(pilot_config["excluded_prior_recovery_seeds"]) != protocol.forbidden_seeds:
        raise RuntimeError("prior recovery-development exclusions are incomplete")
    category_counts: Counter[str] = Counter()
    task_families = set()
    for selected in prefixes:
        seed = int(selected["seed"])
        schedule_item = scheduled[seed]
        analysis_item = analyzed[seed]
        trajectory = recorded[seed]
        action_count = int(selected["replayed_action_count"])
        if selected["prefix_step"] != action_count - 1:
            raise RuntimeError(f"prefix action-count mismatch for seed {seed}")
        if selected["remaining_horizon"] != 50 - action_count:
            raise RuntimeError(f"remaining-horizon mismatch for seed {seed}")
        expected = (schedule_item["schedule_index"], schedule_item["task_id"])
        configured = (selected["schedule_index"], selected["task_id"])
        recorded_key = (
            trajectory["metadata"]["schedule_index"],
            trajectory["task"]["task_id"],
        )
        if configured != expected or recorded_key != expected:
            raise RuntimeError(f"prefix provenance mismatch for seed {seed}")
        if analysis_item["success"]:
            raise RuntimeError(f"seed {seed} is not a failed H4 trajectory")
        if selected["task_family"] != analysis_item["task_family"]:
            raise RuntimeError(f"task-family mismatch for seed {seed}")
        if selected["original_frozen_continuation_action"] != trajectory["steps"][
            action_count
        ]["action"]:
            raise RuntimeError(f"continuation-action mismatch for seed {seed}")
        if selected["prefix_sha256"] != prefix_hash(trajectory, action_count):
            raise RuntimeError(f"prefix hash mismatch for seed {seed}")
        categories = set(selected["selection_categories"])
        if not categories or not categories <= EXPECTED_CATEGORIES:
            raise RuntimeError(f"invalid selection category for seed {seed}")
        category_counts.update(categories)
        task_families.add(selected["task_family"])
    if any(category_counts[category] < 2 for category in EXPECTED_CATEGORIES):
        raise RuntimeError("R2 manifest does not cover two prefixes per target category")
    if len(task_families) < 5:
        raise RuntimeError("R2 manifest does not cover enough task families")


def _verify_frozen_manifest(path: Path) -> str:
    checksum_path = path.with_suffix(".sha256")
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError(f"invalid frozen-manifest checksum file: {checksum_path}")
    _require_hash(path, fields[0])
    return fields[0]


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
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
