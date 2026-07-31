from __future__ import annotations

import unittest

from models.action_parser import is_valid_action, normalize_action, parse_action


class ActionParserTests(unittest.TestCase):
    def test_parses_valid_alfworld_action(self) -> None:
        parsed = parse_action("Action: open fridge 1", ("open fridge 1", "go to sinkbasin 1"))
        self.assertEqual(parsed.action, "open fridge 1")
        self.assertEqual(parsed.status, "grounded")

    def test_parses_thought_plus_action_output(self) -> None:
        parsed = parse_action(
            "Thought: open the refrigerator first\nAction: open fridge 1",
            ("open fridge 1", "go to sinkbasin 1"),
        )
        self.assertEqual(parsed.action, "open fridge 1")
        self.assertEqual(parsed.status, "grounded")
        self.assertEqual(parsed.reasoning, "open the refrigerator first")

    def test_rejects_malformed_output_instead_of_reading_reasoning_as_action(self) -> None:
        parsed = parse_action("<think>go to fridge 1</think>", ("go to fridge 1",))
        self.assertEqual(parsed.action, "")
        self.assertEqual(parsed.status, "not_admissible")
        self.assertEqual(parsed.invalid_reason, "not in valid actions")

    def test_normalizes_action_before_grounding(self) -> None:
        parsed = parse_action("ACTION:   Go  To  Fridge  1", ("go to fridge 1",))
        self.assertEqual(normalize_action("  Go  To Fridge 1 "), "go to fridge 1")
        self.assertEqual(parsed.action, "go to fridge 1")

    def test_valid_action_membership(self) -> None:
        valid_actions = ("go to fridge 1", "open fridge 1")
        self.assertTrue(is_valid_action("Go to fridge 1", valid_actions))
        self.assertFalse(is_valid_action("go to oven 1", valid_actions))

    def test_fallback_behavior_never_invents_a_command(self) -> None:
        parsed = parse_action("Thought: I should go to fridge 1", ("go to fridge 1",))
        self.assertEqual(parsed.action, "")
        self.assertEqual(parsed.status, "missing_action")
