from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from env.alfworld import AlfWorldConfig, AlfWorldTextEnvironment
from env.base import Task
from trajectory.store import JsonlTrajectoryStore
from trajectory.trajectory import Step, Trajectory


class StoreAndEnvironmentTests(unittest.TestCase):
    def test_jsonl_store_writes_canonical_trajectory(self) -> None:
        trajectory = Trajectory(
            trajectory_id="trajectory-1",
            task=Task("task-1", "test task", "valid_seen"),
            model_version="test-model",
            seed=1,
            initial_observation="start",
            steps=(
                Step(
                    index=0,
                    observation="start",
                    action="go to fridge 1",
                    model_output="Action: go to fridge 1",
                    reward=0.0,
                    done=False,
                    timestamp="2026-01-01T00:00:00+00:00",
                ),
            ),
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            truncated=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            JsonlTrajectoryStore(path).append(trajectory)
            record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["task"]["task_id"], "task-1")
        self.assertEqual(record["steps"][0]["model_output"], "Action: go to fridge 1")
        self.assertEqual(record["model_version"], "test-model")

    def test_alfworld_adapter_normalizes_task_metadata_without_runtime(self) -> None:
        environment = AlfWorldTextEnvironment(AlfWorldConfig(Path("unused.yaml"), split="valid_unseen"))
        task = environment._task_from(
            "Welcome\nYour task is to: heat tomato 1 with microwave 1.\n",
            {"extra_game_info": {"game_file": "/tmp/game_001.json"}},
        )
        self.assertEqual(task.task_id, "game_001")
        self.assertEqual(task.text, "heat tomato 1 with microwave 1")
        self.assertEqual(task.split, "valid_unseen")
