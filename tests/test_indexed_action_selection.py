from __future__ import annotations

import unittest

from env.base import ResetResult, Task, Transition
from evaluation.metrics import EvaluationMetrics
from models.action_parser import action_id_mapping, parse_action, parse_action_id
from models.policy import ActionRequest
from models.qwen import QwenPolicy, QwenPolicyConfig
from trajectory.collector import collect_episode


class IndexedActionSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = (
            "examine stoveburner 1",
            "take pan 1 from stoveburner 3",
            "go to fridge 1",
        )

    def test_valid_id_recovers_exact_environment_command(self) -> None:
        parsed = parse_action_id("Action-ID: A001", self.actions)
        self.assertEqual(parsed.status, "selected")
        self.assertEqual(parsed.action_id, "A001")
        self.assertEqual(parsed.selected_index, 1)
        self.assertEqual(parsed.action, "take pan 1 from stoveburner 3")

    def test_malformed_id_is_rejected_without_repair(self) -> None:
        parsed = parse_action_id("Action-ID: A1", self.actions)
        self.assertEqual(parsed.status, "malformed_id")
        self.assertEqual(parsed.action, "")
        self.assertIsNone(parsed.action_id)

    def test_free_form_command_is_never_accepted_in_indexed_mode(self) -> None:
        parsed = parse_action_id("Action: go to fridge 1", self.actions)
        self.assertEqual(parsed.status, "malformed_id")
        self.assertEqual(parsed.action, "")

    def test_out_of_range_id_is_rejected_without_fallback(self) -> None:
        parsed = parse_action_id("Action-ID: A999", self.actions)
        self.assertEqual(parsed.status, "out_of_range_id")
        self.assertEqual(parsed.action_id, "A999")
        self.assertEqual(parsed.selected_index, 999)
        self.assertEqual(parsed.action, "")

    def test_duplicate_output_is_ambiguous(self) -> None:
        parsed = parse_action_id(
            "Action-ID: A001\nAction-ID: A002", self.actions
        )
        self.assertEqual(parsed.status, "ambiguous_id")
        self.assertEqual(parsed.action, "")

    def test_duplicate_output_on_one_line_is_ambiguous(self) -> None:
        parsed = parse_action_id(
            "Action-ID: A001 Action-ID: A002", self.actions
        )
        self.assertEqual(parsed.status, "ambiguous_id")
        self.assertEqual(parsed.action, "")

    def test_mapping_is_deterministic_and_preserves_environment_order(self) -> None:
        expected = {
            "A000": "examine stoveburner 1",
            "A001": "take pan 1 from stoveburner 3",
            "A002": "go to fridge 1",
        }
        self.assertEqual(action_id_mapping(self.actions), expected)
        self.assertEqual(action_id_mapping(self.actions), expected)

    def test_indexed_decision_logs_auditable_mapping(self) -> None:
        policy = QwenPolicy(
            QwenPolicyConfig(
                model_id="test-model",
                action_selection_mode="indexed_admissible",
            )
        )
        request = ActionRequest(
            task=Task("task", "put a cool pan in stoveburner", "valid_seen"),
            observation="room",
            history=(),
            valid_actions=self.actions,
        )

        class Generated:
            scores = ()

        decision = policy._indexed_decision(
            request,
            policy._indexed_prompt(request),
            "Action-ID: A001",
            Generated(),
            None,
        )
        selection = decision.metadata["action_selection"]
        self.assertEqual(decision.action, self.actions[1])
        self.assertEqual(decision.parser_status, "grounded")
        self.assertEqual(selection["parsed_action_id"], "A001")
        self.assertEqual(selection["selected_index"], 1)
        self.assertEqual(selection["mapped_environment_command"], self.actions[1])
        self.assertEqual(selection["id_to_command"], action_id_mapping(self.actions))
        self.assertEqual(selection["selection_status"], "selected")
        self.assertIsNone(selection["failure_reason"])

    def test_b0_parser_behavior_is_unchanged(self) -> None:
        parsed = parse_action(
            "Action: take pan 1 from stoveburner 3", self.actions
        )
        self.assertEqual(parsed.status, "grounded")
        self.assertEqual(parsed.action, "take pan 1 from stoveburner 3")


class IndexedCollectorTests(unittest.TestCase):
    class Environment:
        def __init__(self) -> None:
            self.executed_actions: list[str] = []
            self.task = Task("task", "look around", "valid_seen")

        def reset(self, *, seed: int | None = None) -> ResetResult:
            return ResetResult("room", self.task, ("look", "inventory"))

        def step(self, action: str) -> Transition:
            self.executed_actions.append(action)
            return Transition("done", 0.0, True, False, ("look", "inventory"))

        def get_task(self) -> Task:
            return self.task

        def get_valid_actions(self) -> tuple[str, ...]:
            return ("look", "inventory")

    class Policy:
        model_version = "test-model"

        def __init__(self, raw_output: str = "Action-ID: A001") -> None:
            self.raw_output = raw_output

        def act(self, request: ActionRequest):
            policy = QwenPolicy(
                QwenPolicyConfig(
                    model_id=self.model_version,
                    action_selection_mode="indexed_admissible",
                )
            )

            class Generated:
                scores = ()

            return policy._indexed_decision(
                request,
                policy._indexed_prompt(request),
                self.raw_output,
                Generated(),
                None,
            )

    def test_collector_executes_only_mapped_environment_action(self) -> None:
        environment = self.Environment()
        trajectory = collect_episode(environment, self.Policy(), max_steps=1, seed=42)
        self.assertEqual(environment.executed_actions, ["inventory"])
        step = trajectory.steps[0]
        self.assertEqual(step.action, "inventory")
        self.assertEqual(step.metadata["action_selection_mode"], "indexed_admissible")
        self.assertEqual(
            step.metadata["action_selection"]["id_to_command"],
            {"A000": "look", "A001": "inventory"},
        )

    def test_b1_metrics_do_not_increment_b0_invalid_action_rate(self) -> None:
        environment = self.Environment()
        trajectory = collect_episode(environment, self.Policy(), max_steps=1, seed=42)
        metrics = EvaluationMetrics.from_trajectories([trajectory])
        self.assertEqual(metrics.invalid_action_rate, 0.0)
        self.assertEqual(metrics.selection_failure_rate, 0.0)
        self.assertEqual(metrics.malformed_id_rate, 0.0)
        self.assertEqual(metrics.out_of_range_id_rate, 0.0)

    def test_malformed_id_fails_closed_and_uses_only_b1_metrics(self) -> None:
        environment = self.Environment()
        trajectory = collect_episode(
            environment,
            self.Policy("Action-ID: A1"),
            max_steps=1,
            seed=42,
        )
        self.assertEqual(environment.executed_actions, [])
        self.assertEqual(trajectory.metadata["termination_reason"], "selection_failure")
        self.assertEqual(
            trajectory.steps[0].metadata["action_selection"]["selection_status"],
            "malformed_id",
        )
        metrics = EvaluationMetrics.from_trajectories([trajectory])
        self.assertEqual(metrics.selection_failure_rate, 1.0)
        self.assertEqual(metrics.malformed_id_rate, 1.0)
        self.assertEqual(metrics.out_of_range_id_rate, 0.0)
        self.assertEqual(metrics.invalid_action_rate, 0.0)

    def test_out_of_range_id_fails_closed_and_uses_only_b1_metrics(self) -> None:
        environment = self.Environment()
        trajectory = collect_episode(
            environment,
            self.Policy("Action-ID: A999"),
            max_steps=1,
            seed=42,
        )
        self.assertEqual(environment.executed_actions, [])
        metrics = EvaluationMetrics.from_trajectories([trajectory])
        self.assertEqual(metrics.selection_failure_rate, 1.0)
        self.assertEqual(metrics.malformed_id_rate, 0.0)
        self.assertEqual(metrics.out_of_range_id_rate, 1.0)
        self.assertEqual(metrics.invalid_action_rate, 0.0)
