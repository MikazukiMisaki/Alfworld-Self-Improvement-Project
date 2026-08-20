from __future__ import annotations

import unittest

from env.base import Task
from models.policy import ActionRequest
from models.qwen import QwenPolicy, QwenPolicyConfig


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

    def test_bounded_prompt_keeps_only_four_exact_action_result_transitions(self) -> None:
        request = ActionRequest(
            task=Task("task-1", "heat the tomato and put it on the table", "valid_seen"),
            observation="You are at microwave 1.",
            history=tuple(
                (f"observation-{index}", f"action-{index}") for index in range(6)
            ),
            valid_actions=("inventory", "go to fridge 1"),
        )
        prompt = QwenPolicy._indexed_bounded_prompt(request, 4)
        self.assertIn("Task goal:\nheat the tomato and put it on the table", prompt)
        self.assertIn("Current observation:\nYou are at microwave 1.", prompt)
        self.assertIn("Action: action-2\nResult: observation-3", prompt)
        self.assertIn("Action: action-5\nResult: You are at microwave 1.", prompt)
        self.assertNotIn("action-0", prompt)
        self.assertNotIn("action-1", prompt)
        self.assertIn("[A001] go to fridge 1", prompt)

    def test_bounded_prompt_labels_fresh_inventory_result_when_available(self) -> None:
        request = ActionRequest(
            task=Task("task-1", "look around", "valid_seen"),
            observation="You are carrying: tomato 1.",
            history=(("You are in the kitchen.", "inventory"),),
            valid_actions=("look",),
        )
        prompt = QwenPolicy._indexed_bounded_prompt(request, 4)
        self.assertIn("Current inventory:\nYou are carrying: tomato 1.", prompt)

    def test_bounded_context_configuration_is_explicit_and_validated(self) -> None:
        config = QwenPolicyConfig(
            model_id="test",
            action_selection_mode="indexed_admissible",
            history_context_mode="bounded_recent_state",
            history_window=4,
        )
        self.assertEqual(config.history_window, 4)
        with self.assertRaisesRegex(ValueError, "positive history_window"):
            QwenPolicyConfig(
                model_id="test",
                action_selection_mode="indexed_admissible",
                history_context_mode="bounded_recent_state",
            )
