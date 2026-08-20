"""R2 two-stage recovery with a strict bare-ID Stage-2 contract."""

from __future__ import annotations

import re

from models.action_parser import action_id_mapping
from models.policy import ActionRequest

from .two_stage import (
    STAGE_ONE_INSTRUCTIONS,
    STAGE_ONE_MAX_NEW_TOKENS,
    STAGE_ONE_PROMPT_VERSION,
    STAGE_TWO_MAX_NEW_TOKENS,
    GenerationRecord,
    StageTwoDecision,
    TwoStageRecoveryDecision,
    TwoStageRecoveryOperator,
    _bounded_context,
    parse_stage_one_output,
    stage_one_prompt,
)


RECOVERY_OPERATOR_VERSION = "two_stage_bounded_recovery_v2"
STAGE_TWO_PROMPT_VERSION = "two-stage-subgoal-bare-action-id-v2"
STAGE_TWO_INSTRUCTIONS = (
    "Select the one currently available action that best executes the frozen "
    "immediate subgoal.\n"
    "Return exactly one bare action ID in this format:\n"
    "Axyz\n"
    "Choose one ID exactly as written in the indexed admissible-action mapping.\n"
    "Do not output a label, command, explanation, reasoning, extra line, or "
    "<think> tag."
)
_BARE_ACTION_ID = re.compile(r"A\d{3}")


class TwoStageRecoveryV2Operator(TwoStageRecoveryOperator):
    """Use unchanged Stage 1 followed by one strict bare-ID selection call."""

    def act(self, request: ActionRequest) -> TwoStageRecoveryDecision:
        """Make exactly two generations and return one fail-closed action."""
        stage_one = parse_stage_one_output(
            self._generate(stage_one_prompt(request), STAGE_ONE_MAX_NEW_TOKENS)
        )
        stage_two = parse_bare_stage_two_output(
            self._generate(
                stage_two_prompt(
                    request,
                    diagnosis=stage_one.diagnosis,
                    subgoal=stage_one.subgoal,
                    stage_one_valid=stage_one.status == "valid",
                ),
                STAGE_TWO_MAX_NEW_TOKENS,
            ),
            request.valid_actions,
        )
        if stage_one.status != "valid":
            return TwoStageRecoveryDecision(
                stage_one,
                stage_two,
                "",
                "stage_one_failure",
                stage_one.failure_reason,
                operator_version=RECOVERY_OPERATOR_VERSION,
                stage_two_prompt_version=STAGE_TWO_PROMPT_VERSION,
            )
        if stage_two.status != "selected":
            return TwoStageRecoveryDecision(
                stage_one,
                stage_two,
                "",
                stage_two.status,
                stage_two.failure_reason,
                operator_version=RECOVERY_OPERATOR_VERSION,
                stage_two_prompt_version=STAGE_TWO_PROMPT_VERSION,
            )
        return TwoStageRecoveryDecision(
            stage_one,
            stage_two,
            stage_two.action,
            "selected",
            None,
            operator_version=RECOVERY_OPERATOR_VERSION,
            stage_two_prompt_version=STAGE_TWO_PROMPT_VERSION,
        )


def stage_two_prompt(
    request: ActionRequest,
    *,
    diagnosis: str | None,
    subgoal: str | None,
    stage_one_valid: bool,
) -> str:
    """Render the unchanged decision context with the bare-ID contract."""
    mapping = action_id_mapping(request.valid_actions or ())
    actions = "\n".join(
        f"[{action_id}] {command}" for action_id, command in mapping.items()
    ) or "No admissible action list is available."
    frozen_diagnosis = diagnosis if stage_one_valid else "(invalid Stage-1 output)"
    frozen_subgoal = subgoal if stage_one_valid else "(invalid Stage-1 output)"
    return (
        f"{STAGE_TWO_INSTRUCTIONS}\n\n"
        f"{_bounded_context(request)}\n\n"
        "Frozen Stage-1 diagnosis:\n"
        f"{frozen_diagnosis}\n\n"
        "Frozen Stage-1 immediate subgoal:\n"
        f"{frozen_subgoal}\n\n"
        f"Indexed admissible-action mapping:\n{actions}\n\n"
        "Your response:"
    )


def parse_bare_stage_two_output(
    generation: GenerationRecord, valid_actions: tuple[str, ...] | None
) -> StageTwoDecision:
    """Accept exactly one in-range bare ID and map it fail-closed."""
    mapping = action_id_mapping(valid_actions or ())
    if not generation.output_complete:
        return StageTwoDecision(
            None,
            "",
            "truncated_stage_two",
            "Stage 2 did not complete before its token limit",
            mapping,
            generation,
        )
    if _BARE_ACTION_ID.fullmatch(generation.raw_output) is None:
        return StageTwoDecision(
            None,
            "",
            "malformed_id",
            "expected exactly one bare action ID matching A followed by three digits",
            mapping,
            generation,
        )
    action_id = generation.raw_output
    action = mapping.get(action_id)
    if action is None:
        return StageTwoDecision(
            action_id,
            "",
            "out_of_range_id",
            "action ID is outside the admissible-action range",
            mapping,
            generation,
        )
    return StageTwoDecision(
        action_id,
        action,
        "selected",
        None,
        mapping,
        generation,
    )
