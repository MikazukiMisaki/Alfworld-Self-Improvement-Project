from __future__ import annotations

import unittest

from env.base import Task
from models.policy import ActionRequest
from models.qwen import QwenPolicy


class QwenPromptTests(unittest.TestCase):
    def test_prompt_preserves_goal_and_prevents_unrelated_manipulation(self) -> None:
        prompt = QwenPolicy._prompt(
            ActionRequest(
                task=Task("task-1", "find two book and put them in desk", "valid_seen"),
                observation="You are at shelf 1.",
                history=(("You arrive at shelf 1.", "examine shelf 1"),),
                valid_actions=("go to drawer 1", "take bowl 2 from shelf 1"),
            )
        )
        self.assertIn("Keep pursuing the stated task", prompt)
        self.assertIn("Only take, move, heat, cool, or clean an object required by the task", prompt)
        self.assertIn("Do not alternate between locations", prompt)
        self.assertIn("Task: find two book and put them in desk", prompt)
        self.assertIn("- go to drawer 1", prompt)
