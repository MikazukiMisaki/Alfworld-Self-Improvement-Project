from __future__ import annotations

import unittest

from models.action_parser import parse_action


class ActionParserTests(unittest.TestCase):
    def test_parses_grounded_action_after_reasoning(self) -> None:
        parsed = parse_action(
            "Thought: open the refrigerator first\nAction: open fridge 1",
            ("open fridge 1", "go to sinkbasin 1"),
        )
        self.assertEqual(parsed.action, "open fridge 1")
        self.assertEqual(parsed.status, "grounded")
        self.assertEqual(parsed.reasoning, "open the refrigerator first")

    def test_never_invents_a_fallback_action(self) -> None:
        parsed = parse_action("", ("go to fridge 1",))
        self.assertEqual(parsed.action, "")
        self.assertEqual(parsed.status, "empty")
