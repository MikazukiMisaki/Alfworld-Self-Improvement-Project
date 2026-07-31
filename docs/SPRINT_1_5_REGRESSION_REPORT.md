# Sprint 1.5: Baseline Debug and Validation Report

## Scope

This report compares the legacy runner at
`AlfWorldLegacy/src/envs/batch_run_alfworld.py` with the Sprint 1 baseline.
It covers execution and parsing reliability only. It does not introduce
reflection, recovery, selection, memory, or training.

## Observed regression

`smoke-a40-005` was invalid as a baseline result:

- invalid-action rate: 1.0;
- 50 environment steps;
- 3,200 generated tokens (50 x the 64-token cap);
- reward and success: 0.

The recorded raw output began with `<think>` and was truncated before it
contained a command. The collector sent that string to ALFWorld on every step.
This is a policy/pipeline failure, not evidence about task-solving ability.

`smoke-a40-006` provides a narrow post-fix diagnostic: it used grounded
commands on every step and did not repeatedly hit the token cap. It still
failed the task due to poor search and looping behaviour, which is a separate
agent-quality result rather than a parser regression.

## Legacy versus new runner

| Area | Legacy `batch_run_alfworld.py` | Sprint 1 before 1.5 | Sprint 1.5 rule |
|---|---|---|---|
| Model | Qwen3-VL-8B-Instruct via processor | Text Qwen3-8B via tokenizer | Text Qwen3-8B remains the baseline |
| Prompt | Goal rules, one short example, asks for `Thought` + `Action` | Task, history, current observation, and full admissible-action list | Strict single `Action:` command plus goal-persistence rules |
| Chat template | VL processor template, no explicit thinking control | Text tokenizer template; initially relied on Qwen3 default | `enable_thinking: false` explicitly passed |
| Admissible actions | Not injected as a complete list | Injected on every step | Retained and stored per step |
| Parse | First `Action:` line, otherwise last output line | Could accept any line or a valid action mentioned inside output | Explicit action or one bare command only; reasoning is never an action |
| Normalization | None | Case-insensitive matching only | Whitespace, surrounding quotes, and case normalized before membership check |
| Parse fallback | Uses final output line even if malformed | Could send an ungrounded string to ALFWorld | No invented fallback; malformed output is logged and episode stops |
| Invalid detection | None before `env.step` | Exact membership recorded after `env.step` | Normalized membership before `env.step`; invalid output never reaches environment |
| Length/stopping | 128 generated tokens and 15 environment steps | 64 generated tokens and 50 steps | 32 generated tokens; malformed output stops after one auditable step |
| Split | ALFWorld `train` | Canonical `valid_seen` translated to ALFWorld eval-in-distribution | Same canonical split translation |

## Root causes

1. **Qwen3 thinking mode was not explicitly disabled.** Its default chat
   template began an internal `<think>` block. A 64-token cap ended generation
   before a final command was emitted.
2. **The parser treated arbitrary non-empty lines as candidates.** A truncated
   reasoning string could become an action candidate; its embedded-action scan
   could also mistake reasoning text for a command.
3. **The collector called `environment.step()` even after parsing failed.**
   This converted one model-generation failure into 50 invalid actions and
   obscured the cause in aggregate metrics.
4. **Task provenance was incomplete for the runtime's `extra.gamefile` info
   variant.** This did not cause invalid actions, but made diagnosis weaker.

## Sprint 1.5 changes

- Disable Qwen3 thinking mode and constrain output to 32 tokens.
- Normalize actions before valid-action membership checks.
- Accept only an explicit `Action:`/`Command:` line, or a single bare command.
- Reject malformed or non-admissible text without selecting a fallback action.
- Stop a malformed episode before calling ALFWorld and mark it
  `termination_reason: parser_failure`.
- Persist a structured debug block on every step: raw output, parsed action,
  valid actions, parser status, invalid-action reason, and generated-token
  count.
- Preserve game-file-derived task IDs and reuse the initialized ALFWorld
  environment across episodes.

## Validation protocol

Run 3--5 fixed-seed episodes on the A40 after syncing this revision:

```bash
python scripts/collect_baseline.py --episodes 5 --run-name sprint1-5-a40
```

Inspect `metrics.json` and `trajectory.jsonl`. Sprint 1.5 passes only if:

- invalid-action rate is not 1.0;
- generated-token counts are usually below 32;
- most parser statuses are `grounded`;
- every non-grounded step contains `metadata.debug.invalid_action_reason`;
- at least one full episode has no parser-induced invalid action.

The A40 smoke run is still required; local unit tests validate contracts but
cannot load ALFWorld or Qwen3-8B in this checkout.
