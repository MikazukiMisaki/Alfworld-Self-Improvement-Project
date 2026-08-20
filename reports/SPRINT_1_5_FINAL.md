# Sprint 1.5 Final Baseline Report

## 1. Purpose

Sprint 1.5 stabilized the Qwen3-8B ALFWorld baseline and separated action
grounding failures from policy planning failures. It tested whether indexed
admissible selection improves grounding and whether bounded recent-state
context preserves a credible task-solving signal. H4 is a baseline
stabilization choice, not the paper's main contribution.

## 2. B0: free-form validated baseline

B0 is `free_form_v1` / `free_form_validated`. The model emits one command,
which is parsed and checked against the current environment-owned admissible
set. Invalid output fails closed without repair, substitution, resampling, or
fallback execution.

## 3. B1: indexed grounding

B1 assigns deterministic ephemeral IDs (`A000`, `A001`, ...) to the current
ALFWorld admissible actions. The model returns exactly one `Action-ID: Axxx`;
the selected ID maps back to the exact environment command. Free-form model
text is never executed in indexed mode, and malformed or out-of-range IDs fail
closed.

## 4. H0: unbounded full history

H0 is the original `indexed_v1` prompt. Every prior observation/action pair
remains in context indefinitely. In the three exploratory episodes, model
input grew to 2,133--2,847 tokens by step 50 and all episodes developed severe
repetition or two-cycles.

## 5. H4: bounded recent-state context

H4 is `indexed_bounded_context_v1`. It preserves the task goal, current
observation, current inventory when ALFWorld exposes it, the four most recent
exact action/result transitions, and the current action-ID mapping. It adds no
reflection, recovery, memory, planner, repair, resampling, or task-specific
rules.

## 6. Exploratory H0/H4 diagnostic

The matched development diagnostic used seeds 42, 43, and 44. H0 solved 0/3;
H4 solved 1/3. H4 reduced mean input tokens per step by approximately 55% and
improved target-object adherence, but two failed H4 episodes still entered
take/place cycles. This three-task result was exploratory and is not combined
with formal validation. No statistical significance is claimed.

## 7. Formal H4 validation

The frozen validation used 30 unique `valid_seen` tasks, seeds 1000--1029, and
all six task families. The schedule excluded the development seeds and task
IDs.

| Metric | Result |
|---|---:|
| Success | 2/30 (6.67%) |
| Mean reward | 0.0667 |
| Mean episode length | 47.0 |
| Episodes reaching 50 steps | 28/30 |
| Mean input tokens per step | 686.03 |
| Mean generated tokens per episode | 376 |
| Total episode collection time | 776.14 seconds |

The two successes were a four-step spray-bottle placement and a six-step
vase/desk-lamp task. Formal loop indicators occurred in 23/30 episodes.

## 8. Failure taxonomy

Labels overlap across the 28 failures.

| Failure label | Count |
|---|---:|
| State-progression failure | 28 |
| Target-object identification failure | 23 |
| Off-target manipulation | 22 |
| Strict A-B-A-B deadlock | 13 |
| Navigation/exploration failure | 12 |
| Adjacent repetition | 11 |
| Wrong transformation/tool choice | 10 |
| Interface failure | 0 |

Seeds 1005, 1009, 1010, 1022, and 1027 contain strong candidate prefixes for
future replay and paired-continuation research. They are annotations only and
were not used to implement or run recovery.

## 9. Interface validation

All 1,410 formal selections were valid. There were zero malformed IDs,
out-of-range IDs, selection failures, mapping failures, inadmissible
executions, or output-token saturations. The indexed interface is technically
reliable; observed failures are primarily planning, state progression,
exploration, and loop failures.

## 10. Known limitations

- The formal sample contains only 30 `valid_seen` tasks and no `valid_unseen`
  estimate.
- The 6.67% success estimate is low and has substantial sampling uncertainty.
- Transformation and two-object task families had no formal successes.
- Strict adjacent/A-B-A-B indicators undercount longer periodic behavior.
- Failure labels combine automatic signals with documented qualitative review.
- H0/H4 development results are too small for significance claims.

## 11. Gate decision

**SPRINT 1.5 PASSED / H4 FROZEN.**

H4 has a non-zero held-out task-solving signal, exact action grounding,
auditable trajectories, and meaningful failure states for replay research.
Further baseline tuning is prohibited unless Sprint 2A replay validation
reveals a genuine implementation defect.

## 12. Exact frozen pipeline configuration

- Model: `Qwen/Qwen3-8B`
- Pipeline: `indexed_bounded_context_v1`
- Action interface: `indexed_admissible`
- Context: bounded goal-preserving recent state, `k=4`
- Thinking: disabled
- Decoding: greedy, `do_sample=false`
- Maximum generated tokens: 32
- Maximum episode steps: 50
- Environment: ALFWorld `valid_seen`
- No recovery, reflection, memory, repair, resampling, fallback planner, DPO,
  SFT, PPO, or GRPO

## 13. Frozen provenance

- Implementation commit:
  `08bc212d7b2ed1e38665f4b3049638ffe3a60005`
- Schedule SHA-256:
  `ee79fae0c62c618c4495c5c9ff45b8a95a463e7775ee82d55c922418a9273441`
- Server artifact path:
  `artifacts/runtime/validation/sprint1-5-h4-formal-20260819T191000Z/`
- Tracked machine-readable summary:
  `reports/sprint1_5_formal_summary.json`

## 14. Freeze boundary

Sprint 2 had not begun when this baseline was frozen. The next authorized
milestone is Sprint 2A replay/reconstruction validation only. Recovery,
reflection, branching experiments, and selector implementation remain gated
on deterministic replay evidence.
