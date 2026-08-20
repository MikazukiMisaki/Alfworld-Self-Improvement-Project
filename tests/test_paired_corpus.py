from __future__ import annotations

import pytest

from scripts.collect_sprint2c_h4_pool import _validate_frozen_h4
from recovery.corpus import (
    POOL_BURN_IN_SEEDS,
    executed_schedule,
    select_prefixes,
    state_fingerprint,
    task_family_from_id,
    validate_prefix_manifest,
    validate_pool_schedule,
)
from scripts.collect_sprint2c_paired_corpus import (
    _validate_record_schema,
    aggregate_pairs,
)
from scripts.analyze_sprint2c_corpus import _gate


def schedule() -> dict:
    episodes = []
    families = [
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
    ]
    for index in range(300):
        family = families[index % len(families)]
        episodes.append(
            {
                "schedule_index": index,
                "reset_order_position": index + len(POOL_BURN_IN_SEEDS),
                "seed": 2000 + index,
                "pool_block": "initial" if index < 200 else "extension",
                "split": "valid_seen",
                "task_id": f"{family}-Object-None-Receptacle-{index}/trial",
                "task_family": family,
            }
        )
    return {
        "schema_version": "sprint2c_h4_pool_schedule_v1",
        "split": "valid_seen",
        "burn_in_seeds": POOL_BURN_IN_SEEDS,
        "reset_offset_rule": {
            "prior_formal_burn_in_resets": 3,
            "valid_seen_game_count": 140,
            "offset": 40,
            "total_burn_in_resets": len(POOL_BURN_IN_SEEDS),
        },
        "extension_gate": {
            "execute_extension_if_initial_success_count_below": 10,
            "initial_episode_count": 200,
            "extension_episode_count": 100,
            "maximum_total_episodes": 300,
            "adaptive_changes_after_initial_pool": False,
        },
        "episodes": episodes,
    }


def test_pool_schedule_and_conditional_extension_are_fixed() -> None:
    value = schedule()
    validate_pool_schedule(value)

    assert len(executed_schedule(value, initial_success_count=10)) == 200
    assert len(executed_schedule(value, initial_success_count=9)) == 300


def test_pool_schedule_rejects_reset_order_or_task_family_changes() -> None:
    reset_changed = schedule()
    reset_changed["episodes"][7]["reset_order_position"] = 99
    with pytest.raises(ValueError, match="reset-order"):
        validate_pool_schedule(reset_changed)

    family_changed = schedule()
    family_changed["episodes"][7]["task_family"] = "unknown"
    with pytest.raises(ValueError, match="task-family"):
        validate_pool_schedule(family_changed)


def test_task_family_parser_rejects_unknown_families() -> None:
    assert (
        task_family_from_id("pick_and_place_simple-Book-None-Sofa-1/trial")
        == "pick_and_place_simple"
    )
    with pytest.raises(ValueError, match="unknown"):
        task_family_from_id("unsupported-Book-None-Sofa-1/trial")


def test_h4_pool_rejects_any_model_or_split_change() -> None:
    model = {
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
    _validate_frozen_h4(model, {"split": "valid_seen"})

    changed = {**model, "enable_thinking": True}
    with pytest.raises(RuntimeError, match="configuration changed"):
        _validate_frozen_h4(changed, {"split": "valid_seen"})
    with pytest.raises(RuntimeError, match="valid_seen"):
        _validate_frozen_h4(model, {"split": "valid_unseen"})


def test_prefix_selection_is_deterministic_balanced_and_group_bounded() -> None:
    trajectories = []
    for index in range(12):
        trajectories.append(_trajectory(index, success=True, loop=False))
    for index in range(12, 32):
        trajectories.append(_trajectory(index, success=False, loop=True))
    for index in range(32, 82):
        trajectories.append(_trajectory(index, success=False, loop=False))

    selected, metadata = select_prefixes(
        trajectories, excluded_state_fingerprints=set()
    )
    repeated, repeated_metadata = select_prefixes(
        trajectories, excluded_state_fingerprints=set()
    )
    manifest = {
        "schema_version": "sprint2c_paired_prefix_manifest_v1",
        "selection_protocol": {
            "neutral_annotation_priority": [f"P{index:03d}" for index in range(120)]
        },
        "prefixes": selected,
    }
    validate_prefix_manifest(manifest)

    assert [item["prefix_sha256"] for item in selected] == [
        item["prefix_sha256"] for item in repeated
    ]
    assert metadata == repeated_metadata
    assert max(
        sum(item["episode_group_id"] == group for item in selected)
        for group in {item["episode_group_id"] for item in selected}
    ) <= 2
    assert metadata["uncertainty_selection_rule"] == (
        "entropy_at_or_above_within_episode_non_loop_failed_q75"
    )


def test_state_fingerprint_ignores_nominal_seed() -> None:
    first = _trajectory(1, success=False, loop=False)
    second = {**first, "seed": 9999}

    assert state_fingerprint(first, 4) == state_fingerprint(second, 4)


def test_manifest_rejects_feature_leakage_or_annotation_order_drift() -> None:
    trajectories = []
    for index in range(12):
        trajectories.append(_trajectory(index, success=True, loop=False))
    for index in range(12, 32):
        trajectories.append(_trajectory(index, success=False, loop=True))
    for index in range(32, 82):
        trajectories.append(_trajectory(index, success=False, loop=False))
    selected, _ = select_prefixes(trajectories, excluded_state_fingerprints=set())
    manifest = {
        "schema_version": "sprint2c_paired_prefix_manifest_v1",
        "selection_protocol": {
            "neutral_annotation_priority": [f"P{index:03d}" for index in range(120)]
        },
        "prefixes": selected,
    }
    manifest["prefixes"][0]["decision_features"]["future_reward"] = 1
    with pytest.raises(ValueError, match="feature schema"):
        validate_prefix_manifest(manifest)

    manifest["prefixes"][0]["decision_features"].pop("future_reward")
    manifest["selection_protocol"]["neutral_annotation_priority"] = ["P000"]
    with pytest.raises(ValueError, match="neutral annotation"):
        validate_prefix_manifest(manifest)


def test_pair_aggregation_keeps_labels_protocol_and_cost_separate() -> None:
    pairs = [
        _pair(continue_return=0.0, recover_return=1.0, loop_change="reduced"),
        _pair(continue_return=1.0, recover_return=0.0, loop_change="increased"),
        _pair(continue_return=0.0, recover_return=0.0, loop_change="unchanged"),
    ]

    aggregate = aggregate_pairs(pairs)

    assert aggregate["labels"]["beneficial_count"] == 1
    assert aggregate["labels"]["harmful_count"] == 1
    assert aggregate["labels"]["neutral_count"] == 1
    assert aggregate["labels"]["continue_fail_recover_success_count"] == 1
    assert aggregate["labels"]["continue_success_recover_fail_count"] == 1
    assert aggregate["protocol"]["mapping_failure_count"] == 0
    assert aggregate["cost"]["mean_total_generated_tokens"] == 6
    assert aggregate["loop_changes"] == {
        "increased": 1,
        "reduced": 1,
        "unchanged": 1,
    }


def test_record_schema_keeps_grouping_and_features_frozen() -> None:
    from pathlib import Path
    import json

    schema = json.loads(
        Path("configs/corpus/sprint2c_paired_record_schema.json").read_text()
    )
    _validate_record_schema(schema)

    schema["selector_feature_allowlist"].append("continue_return")
    with pytest.raises(RuntimeError, match="allowlist"):
        _validate_record_schema(schema)


def test_corpus_gate_requires_label_count_and_episode_family_diversity() -> None:
    sparse = [
        {"episode_group_id": f"episode-{index}", "task_family": "family-a"}
        for index in range(2)
    ]
    enough_but_concentrated = [
        {"episode_group_id": f"episode-{index // 2}", "task_family": "family-a"}
        for index in range(8)
    ]
    enough_and_diverse = [
        {
            "episode_group_id": f"episode-{index // 2}",
            "task_family": f"family-{index % 2}",
        }
        for index in range(8)
    ]

    assert _gate(sparse) == "C_RECOVERY_OPPORTUNITY_TOO_SPARSE"
    assert _gate(enough_but_concentrated).startswith("B_")
    assert _gate(enough_and_diverse) == "A_LABEL_DENSITY_SUFFICIENT"


def _pair(
    *, continue_return: float, recover_return: float, loop_change: str
) -> dict:
    value = recover_return - continue_return
    classification = "beneficial" if value > 0 else "harmful" if value < 0 else "neutral"
    cost = {
        "stage_one_input_tokens": 100,
        "stage_one_generated_tokens": 4,
        "stage_one_latency_seconds": 0.1,
        "stage_two_input_tokens": 120,
        "stage_two_generated_tokens": 2,
        "stage_two_latency_seconds": 0.2,
        "total_input_tokens": 220,
        "total_generated_tokens": 6,
        "total_token_count": 226,
        "sequential_latency_seconds": 0.3,
    }
    return {
        "classification": classification,
        "v_raw": value,
        "continue_return": continue_return,
        "recover_return": recover_return,
        "sampling_stratum": "random_non_loop_failed",
        "task_family": "pick_and_place_simple",
        "stage_bucket": "middle",
        "decision_features": {
            "recent_adjacent_repeat_indicator": False,
            "recent_two_cycle_indicator": False,
            "entropy_quantile": "Q2",
        },
        "reconstruction_provenance": {
            "continue_exact": True,
            "recover_exact": True,
        },
        "branch_validation": {"recovery_environment_action_count": 1},
        "recovery": {
            "stage_one": {
                "status": "valid",
                "output_complete": True,
                "token_cap_reached": False,
            },
            "stage_two": {
                "selection_status": "selected",
                "output_complete": True,
            },
            "cost": cost,
        },
        "post_run_analysis": {"loop_change": loop_change},
    }


def _trajectory(index: int, *, success: bool, loop: bool) -> dict:
    families = sorted(
        [
            "pick_and_place_simple",
            "pick_clean_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_heat_then_place_in_recep",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
        ]
    )
    family = families[index % len(families)]
    length = 12 if success else 50
    steps = []
    for step_index in range(length):
        action = (
            ("look" if step_index % 2 == 0 else "inventory")
            if loop
            else f"synthetic action {index} {step_index}"
        )
        steps.append(
            {
                "index": step_index,
                "observation": f"observation {index} {step_index}",
                "action": action,
                "valid_actions": [action],
                "done": success and step_index == length - 1,
                "reward": float(success and step_index == length - 1),
                "token_statistics": {
                    "generated_tokens": 5,
                    "input_tokens": 100 + step_index,
                    "mean_token_entropy": index / 100 + step_index / 10000,
                    "mean_token_log_probability": -0.1,
                },
                "metadata": {
                    "policy": {
                        "action_selection": {
                            "selection_status": "selected",
                            "parsed_action_id": "A000",
                            "mapped_environment_command": action,
                        }
                    }
                },
            }
        )
    return {
        "trajectory_id": f"trajectory-{index}",
        "seed": 2000 + index,
        "task": {
            "task_id": f"{family}-Object-None-Receptacle-{index}/trial",
            "text": f"complete synthetic task {index}",
        },
        "metadata": {"schedule_index": index, "task_family": family},
        "steps": steps,
    }
