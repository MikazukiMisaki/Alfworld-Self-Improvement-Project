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

    def test_indexed_prompt_uses_deterministic_ids_and_strict_protocol(self) -> None:
        prompt = QwenPolicy._indexed_prompt(
            ActionRequest(
                task=Task("task-1", "look around", "valid_seen"),
                observation="You are in a room.",
                history=(),
                valid_actions=("look", "inventory"),
            )
        )
        self.assertIn("Return exactly one line in this format:\nAction-ID: Axyz", prompt)
        self.assertIn("[A000] look", prompt)
        self.assertIn("[A001] inventory", prompt)
        self.assertNotIn("Return exactly one line in this format: Action: <command>", prompt)
