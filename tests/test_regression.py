from __future__ import annotations

import unittest

import yaml

from evaluation.regression import (
    RegressionComparisonError,
    compare_trajectory_sets,
    compare_trajectories,
    parse_task_targets,
    validate_indexed_context_config_equivalence,
    validate_interface_config_equivalence,
)


class RegressionHarnessTests(unittest.TestCase):
    def test_portable_configs_differ_only_by_action_interface(self) -> None:
        free_collection = self._yaml("configs/collection/baseline.yaml")
        indexed_collection = self._yaml("configs/collection/baseline_indexed.yaml")
        environment = self._yaml(free_collection["environment_config"])
        validate_interface_config_equivalence(
            free_collection,
            indexed_collection,
            self._yaml(free_collection["model_config"]),
            self._yaml(indexed_collection["model_config"]),
            environment,
        )

    def test_config_mismatch_is_rejected(self) -> None:
        collection = {
            "model_config": "free.yaml",
            "episodes": 1,
            "max_steps": 50,
        }
        indexed_collection = dict(collection, model_config="indexed.yaml", max_steps=49)
        free_model = self._model("free_form_v1", "free_form_validated")
        indexed_model = self._model("indexed_v1", "indexed_admissible")
        with self.assertRaisesRegex(RegressionComparisonError, "collection configs"):
            validate_interface_config_equivalence(
                collection,
                indexed_collection,
                free_model,
                indexed_model,
                {"split": "valid_seen"},
            )

    def test_indexed_context_configs_differ_only_by_pre_registered_context(self) -> None:
        h0_collection = self._yaml("configs/collection/baseline_indexed.yaml")
        hk_collection = self._yaml("configs/collection/baseline_indexed_h4.yaml")
        environment = self._yaml(h0_collection["environment_config"])
        validate_indexed_context_config_equivalence(
            h0_collection,
            hk_collection,
            self._yaml(h0_collection["model_config"]),
            self._yaml(hk_collection["model_config"]),
            environment,
            expected_window=4,
        )

    def test_indexed_context_comparison_records_both_pipeline_versions(self) -> None:
        h0 = self._trajectory("task-a", 42, ["look"])
        hk = self._trajectory("task-a", 42, ["inventory"])
        report = compare_trajectory_sets(
            [h0],
            [hk],
            reference_pipeline="indexed_v1",
            candidate_pipeline="indexed_bounded_context_v1",
        )
        self.assertEqual(
            report["comparison"],
            "indexed_v1_vs_indexed_bounded_context_v1",
        )
        self.assertEqual(report["candidate_pipeline"], "indexed_bounded_context_v1")

    def test_task_or_seed_mismatch_is_rejected(self) -> None:
        free = self._trajectory("task-a", 42, ["look"])
        indexed = self._trajectory("task-b", 42, ["look"])
        with self.assertRaisesRegex(RegressionComparisonError, "task/seed mismatch"):
            compare_trajectory_sets([free], [indexed])

    def test_step_comparison_finds_earliest_divergence_and_progress(self) -> None:
        task_id = "pick_heat_then_place_in_recep-Tomato-None-DiningTable-16/trial"
        free = self._trajectory(
            task_id,
            42,
            ["go to fridge 1", "take tomato 1 from fridge 1", "heat tomato 1 with microwave 1"],
        )
        indexed = self._trajectory(
            task_id,
            42,
            ["go to fridge 1", "go to stoveburner 1", "go to stoveburner 1"],
        )
        report = compare_trajectories(free, indexed)
        self.assertEqual(report["earliest_divergence_step"], 1)
        self.assertIn(
            "target_pickup_attempt", report["steps"][1]["reference_progress"]
        )
        self.assertIn(
            "required_transformation_attempt",
            report["steps"][2]["reference_progress"],
        )
        self.assertIn("repeated_action", report["steps"][2]["indexed_progress"])

    def test_task_target_parser_does_not_invent_unknown_fields(self) -> None:
        targets = parse_task_targets("unknown")
        self.assertIsNone(targets.target_object)
        self.assertIsNone(targets.transformation)
        self.assertIsNone(targets.destination)

    @staticmethod
    def _yaml(path: str) -> dict:
        with open(path, encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        assert isinstance(value, dict)
        return value

    @staticmethod
    def _model(pipeline: str, mode: str) -> dict:
        return {
            "model_id": "Qwen/Qwen3-8B",
            "pipeline_version": pipeline,
            "action_selection": {"mode": mode},
            "generation": {"max_new_tokens": 32},
        }

    @staticmethod
    def _trajectory(task_id: str, seed: int, actions: list[str]) -> dict:
        return {
            "task": {
                "task_id": task_id,
                "text": "synthetic task",
                "split": "valid_seen",
            },
            "seed": seed,
            "metadata": {"termination_reason": "max_steps"},
            "steps": [
                {
                    "index": index,
                    "observation": "tomato is visible",
                    "action": action,
                    "action_valid": True,
                    "reward": 0.0,
                    "done": False,
                }
                for index, action in enumerate(actions)
            ],
        }
