from __future__ import annotations

import unittest
from typing import Any

from alfworld_research.evaluation.metrics import EvaluationMetrics
from alfworld_research.preference.builder import build_pair
from alfworld_research.trajectory.collector import collect_episode


class Environment:
    def reset(self, *, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        return "start", {"seed": seed}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        return "done", 1.0 if action == "good" else 0.0, True, {"success": action == "good"}


class Policy:
    def __init__(self, action: str) -> None:
        self.action = action

    def act(self, observation: str, history: tuple[str, ...]) -> str:
        return self.action


class PipelineTests(unittest.TestCase):
    def test_collection_and_metrics(self) -> None:
        trajectory = collect_episode(Environment(), Policy("good"), seed=7)
        self.assertEqual(trajectory.total_reward, 1.0)
        self.assertTrue(trajectory.succeeded)
        self.assertEqual(EvaluationMetrics.from_trajectories([trajectory]).success_rate, 1.0)

    def test_preference_ranks_higher_reward(self) -> None:
        good = collect_episode(Environment(), Policy("good"))
        bad = collect_episode(Environment(), Policy("bad"))
        example = build_pair(bad, good)
        self.assertEqual(example.chosen, "good")
        self.assertEqual(example.rejected, "bad")
