from __future__ import annotations

import unittest
from typing import Any

from env.base import ResetResult, Task, Transition
from evaluation.evaluator import evaluate
from evaluation.metrics import EvaluationMetrics
from models.policy import ActionDecision, ActionRequest, TokenStatistics
from trajectory.collector import collect_episode


class Environment:
    def __init__(self) -> None:
        self.task = Task("task-1", "choose good", "valid_seen", "toy")

    def reset(self, *, seed: int | None = None) -> ResetResult:
        return ResetResult("start", self.task, ("good", "bad"), {"seed": seed})

    def step(self, action: str) -> Transition:
        return Transition(
            observation="done",
            reward=1.0 if action == "good" else 0.0,
            done=True,
            truncated=False,
            valid_actions=("good", "bad"),
            metadata={"success": action == "good"},
        )

    def get_task(self) -> Task:
        return self.task

    def get_valid_actions(self) -> tuple[str, ...]:
        return ("good", "bad")


class Policy:
    model_version = "test-policy"

    def __init__(self, action: str) -> None:
        self.action = action

    def act(self, request: ActionRequest) -> ActionDecision:
        return ActionDecision(
            action=self.action,
            raw_output=f"Action: {self.action}",
            parser_status="grounded",
            model_version=self.model_version,
            token_statistics=TokenStatistics(1, -0.1, 0.2),
        )


class PipelineTests(unittest.TestCase):
    def test_collection_preserves_baseline_provenance(self) -> None:
        trajectory = collect_episode(Environment(), Policy("good"), max_steps=3, seed=7)
        step = trajectory.steps[0]
        self.assertEqual(trajectory.task.task_id, "task-1")
        self.assertEqual(trajectory.task.split, "valid_seen")
        self.assertEqual(trajectory.model_version, "test-policy")
        self.assertEqual(trajectory.seed, 7)
        self.assertEqual(step.index, 0)
        self.assertEqual(step.observation, "start")
        self.assertEqual(step.action, "good")
        self.assertEqual(step.model_output, "Action: good")
        self.assertEqual(step.token_statistics.generated_tokens, 1)
        self.assertTrue(step.done)
        self.assertTrue(trajectory.succeeded)

    def test_metrics_include_length_actions_and_tokens(self) -> None:
        trajectory = collect_episode(Environment(), Policy("good"), max_steps=3, seed=7)
        metrics = EvaluationMetrics.from_trajectories([trajectory])
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.mean_episode_length, 1.0)
        self.assertEqual(metrics.invalid_action_rate, 0.0)
        self.assertEqual(metrics.mean_generated_tokens, 1.0)

    def test_evaluation_reuses_one_environment(self) -> None:
        instances: list[Environment] = []

        def factory() -> Environment:
            instance = Environment()
            instances.append(instance)
            return instance

        report = evaluate(factory, Policy("good"), seeds=(1, 2), max_steps=1)
        self.assertEqual(len(report.trajectories), 2)
        self.assertEqual(len(instances), 1)
