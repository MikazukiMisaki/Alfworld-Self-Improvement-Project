"""Strict parsing and grounding helpers for textual environment actions."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_PREFIX = re.compile(r"^\s*(?:action|command)\s*:\s*", re.IGNORECASE)
_THOUGHT_PREFIX = re.compile(r"^\s*(?:thought|reasoning)\s*:\s*", re.IGNORECASE)
_ACTION_ID_LINE = re.compile(r"^Action-ID:\s*(A\d{3})$")
_ACTION_ID_ANYWHERE = re.compile(r"Action-ID:\s*(A\d{3})")


@dataclass(frozen=True)
class ParsedAction:
    """A parsed action plus enough context to audit a rejection."""

    action: str
    status: str
    reasoning: str | None = None
    candidate: str | None = None
    invalid_reason: str | None = None


@dataclass(frozen=True)
class ParsedActionId:
    """Strict indexed-selection result mapped to an environment-owned command."""

    action: str
    action_id: str | None
    selected_index: int | None
    status: str
    invalid_reason: str | None = None


def normalize_action(action: str) -> str:
    """Canonicalize harmless formatting differences without changing meaning."""
    return " ".join(action.strip().strip("'\"").split()).casefold()


def is_valid_action(action: str, valid_actions: tuple[str, ...] | None) -> bool | None:
    """Check admissible-command membership using the same normalization as parsing."""
    if valid_actions is None:
        return None
    return bool(action) and normalize_action(action) in {
        normalize_action(valid_action) for valid_action in valid_actions
    }


def parse_action(output: str, valid_actions: tuple[str, ...] | None = None) -> ParsedAction:
    """Accept an explicit command or a single bare admissible command.

    The parser deliberately never extracts an action merely because it appears
    inside reasoning. A malformed response produces an empty action rather
    than an invented fallback command.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    reasoning = next(
        (_THOUGHT_PREFIX.sub("", line).strip() for line in lines if _THOUGHT_PREFIX.match(line)),
        None,
    )
    explicit = [
        _ACTION_PREFIX.sub("", line).strip(" '\"")
        for line in lines
        if _ACTION_PREFIX.match(line)
    ]
    bare = [
        line.strip(" '\"")
        for line in lines
        if not _THOUGHT_PREFIX.match(line) and not _ACTION_PREFIX.match(line)
    ]
    candidates = explicit if explicit else bare if len(bare) == 1 else []
    if not candidates:
        return ParsedAction("", "missing_action", reasoning, invalid_reason="no explicit action")

    candidate = candidates[0]
    if not candidate:
        return ParsedAction("", "empty_action", reasoning, candidate, "empty action value")
    if valid_actions is None:
        return ParsedAction(candidate, "unvalidated", reasoning, candidate)

    canonical_actions = {normalize_action(action): action for action in valid_actions}
    canonical = canonical_actions.get(normalize_action(candidate))
    if canonical is None:
        return ParsedAction("", "not_admissible", reasoning, candidate, "not in valid actions")
    return ParsedAction(canonical, "grounded", reasoning, candidate)


def action_id_mapping(valid_actions: tuple[str, ...]) -> dict[str, str]:
    """Assign stable ephemeral IDs while preserving environment action order."""
    if len(valid_actions) > 1000:
        raise ValueError("indexed action selection supports at most 1000 actions")
    return {f"A{index:03d}": action for index, action in enumerate(valid_actions)}


def parse_action_id(output: str, valid_actions: tuple[str, ...] | None) -> ParsedActionId:
    """Parse exactly one ``Action-ID: Axyz`` line without repair or fallback."""
    if valid_actions is None or not valid_actions:
        return ParsedActionId(
            "", None, None, "malformed_id", "no admissible actions available"
        )

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    embedded_ids = _ACTION_ID_ANYWHERE.findall(output)
    if len(embedded_ids) > 1:
        return ParsedActionId(
            "", None, None, "ambiguous_id", "multiple action IDs"
        )
    if len(lines) != 1:
        return ParsedActionId(
            "", None, None, "malformed_id", "expected exactly one output line"
        )

    match = _ACTION_ID_LINE.fullmatch(lines[0])
    if match is None:
        return ParsedActionId(
            "", None, None, "malformed_id", "expected exactly Action-ID: Axyz"
        )

    action_id = match.group(1)
    selected_index = int(action_id[1:])
    mapping = action_id_mapping(valid_actions)
    action = mapping.get(action_id)
    if action is None:
        return ParsedActionId(
            "",
            action_id,
            selected_index,
            "out_of_range_id",
            "action ID is outside the admissible-action range",
        )
    return ParsedActionId(action, action_id, selected_index, "selected")
