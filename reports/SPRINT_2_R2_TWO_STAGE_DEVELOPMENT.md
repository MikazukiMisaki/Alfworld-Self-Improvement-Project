# Sprint 2 R2 Two-Stage Recovery Development Pilot

## Decision

**R2-TWO-STAGE DOES NOT SOLVE THE REASONING-TO-ACTION GAP.**

The frozen ten-prefix development pilot does not pass the collection gate. Stage 1
was structurally reliable, but Stage 2 failed the exact action-selection contract
on all ten calls. No recovery action was executed, there was no non-zero recovery
benefit, and the operator must not be frozen for paired-corpus collection.

This result does not authorize a selector, training, prompt tuning, retry, fallback,
repair, memory, or another run on these ten development prefixes.

## Frozen Protocol

- Protocol implementation commit: `4c032f9c061ddf5938d7209194d93769a4aee97f`
- Base policy: frozen `Qwen/Qwen3-8B` H4, `indexed_bounded_context_v1`, `k=4`
- Base generation: thinking disabled, greedy, `max_new_tokens=32`
- Stage 1: one call, thinking disabled, greedy, `max_new_tokens=40`
- Stage 1 contract: one diagnosis and one immediate subgoal, each at most 12 words
- Stage 2: one call, thinking disabled, greedy, `max_new_tokens=16`
- Stage 2 contract: exactly `Action-ID: Axxx`
- Intervention: one Stage-1 call plus one Stage-2 call at one branch point
- After a valid intervention action: resume the unchanged H4 policy
- No retry, fallback, repair, oracle state, memory, selector, or training

The suggested 40/16 budgets were frozen before inference. Forty tokens provide
room for the two bounded Stage-1 fields; sixteen tokens are ample for the one-line
Stage-2 contract. Neither budget was changed after observing outputs.

## Preregistration And Replay

The manifest was written, checksummed, tested, and committed before model loading:

- Manifest: `configs/experiments/r2_two_stage_development.json`
- Manifest SHA-256: `a317c2ba58f3b6ec5ace04ea6d1b2da4defa7662a7519ec961bc8e6c5c24b444`
- Prefixes: exactly 10 fresh failed H4 trajectories across six task families
- Excluded prior development seeds: `1005`, `1009`, `1010`, `1022`, `1027`
- Composition: two prefixes in each preregistered failure category
- Formal H4 trajectory SHA-256: `48ddb86e33957549d34f8c449870c027f73100d64cdee642c04d98a625e7223b`
- Formal schedule SHA-256: `ee79fae0c62c618c4495c5c9ff45b8a95a463e7775ee82d55c922418a9273441`

The runner preserved the formal reset/burn-in order and verified task identity,
prefix reconstruction, hidden/public replay state, observation, admissible actions,
remaining horizon, and original deterministic H4 continuation action. All checks
passed for all ten pairs.

## Aggregate Results

| Measure | Result |
|---|---:|
| Pairs | 10 |
| Stage-1 valid / complete / truncated | 10 / 10 / 0 |
| Stage-2 selected / complete / truncated | 0 / 10 / 0 |
| Stage-2 malformed | 10 |
| Recovery environment actions executed | 0 |
| Beneficial / neutral / harmful | 0 / 10 / 0 |
| Continue-fail / recover-success | 0 |
| Continue-success / recover-fail | 0 |
| Mean Stage-1 input / generated tokens | 526.3 / 29.9 |
| Mean Stage-2 input / generated tokens | 671.2 / 5.0 |
| Mean total generated tokens | 34.9 |
| Mean sequential recovery latency | 1.410 s |
| Total generated tokens / sequential latency | 349 / 14.102 s |
| Mean continue / recover remaining length | 42.4 / 1.0 |
| Continue branches with loop indicators | 7 / 10 |
| Recover branches with loop indicators | 0 / 10, not interpretable |

Every Stage-2 generation was a unique bare token such as `A002`, rather than the
required `Action-ID: A002`. This was not truncation: all outputs were complete and
used five generated tokens. The strict parser correctly failed closed. The one-step
recovery records are selection-failure records, so their zero loop counts are not
evidence that recovery broke loops.

## Per-Prefix Results

All continue and recovery returns were `0`, all branches were unsuccessful, and
all pairs were neutral. The mapped command column is post-run qualitative analysis
of the bare token only; it was not parsed, selected, or executed.

| Seed | Schedule / prefix | Category | Continue action | Stage-1 diagnosis | Stage-1 subgoal | Bare token -> analytical command | Continue / recover length | Tokens / latency |
|---:|---:|---|---|---|---|---|---:|---:|
| 1000 | 0 / 4 | strict A-B-A-B | `take keychain 2 from sofa 1` | Book not in inventory or location | Find and take book from available location | `A000` -> `examine sofa 1` | 45 / 1 | 26 / 1.038 s |
| 1001 | 1 / 3 | wrong object | `go to countertop 1` | Potato 2 is cooled but not a tomato. | Find a tomato and cool it with fridge 1. | `A004` -> `go to cabinet 1` | 46 / 1 | 34 / 1.409 s |
| 1002 | 2 / 18 | strict A-B-A-B | `take pot 2 from cabinet 1` | Pot 2 is repeatedly moved between cabinet 1 and inventory. | Put pot 2 in a different cabinet to avoid cycle. | `A033` -> `take pot 2 from cabinet 1` | 31 / 1 | 38 / 1.535 s |
| 1004 | 4 / 4 | adjacent stall | `examine desk 1` | Agent repeatedly examines desk 1 without finding keychain or desklamp | Check drawers and shelves for keychain and desklamp | `A002` -> `go to drawer 1` | 45 / 1 | 38 / 1.483 s |
| 1007 | 7 / 2 | state progression | `go to fridge 1` | Lettuce not cleaned or located, fridge not accessed yet | Find and clean lettuce then put it in fridge | `A033` -> `go to fridge 1` | 47 / 1 | 34 / 1.416 s |
| 1011 | 11 / 4 | ordinary non-loop | `move apple 1 to microwave 1` | Apple is in inventory, but tomato is not found. | Find and pick up a cool tomato from the fridge or countertop. | `A020` -> `go to countertop 1` | 45 / 1 | 38 / 1.552 s |
| 1013 | 13 / 4 | wrong object | `move statue 1 to shelf 2` | Statue 1 is in inventory, not winebottle. | Find and pick up a winebottle from a shelf or cabinet. | `A026` -> `go to shelf 1` | 45 / 1 | 39 / 1.580 s |
| 1018 | 18 / 9 | ordinary non-loop | `take cloth 1 from toilet 1` | Only one cloth is at the toilet, need to get another spraybottle. | Take cloth 1 from toilet 1. | `A013` -> `take cloth 1 from toilet 1` | 40 / 1 | 38 / 1.464 s |
| 1019 | 19 / 8 | adjacent stall | `look` | Unable to locate tomato or heat source for cooking | Go to fridge 1 to check for tomato availability | `A019` -> `go to fridge 1` | 41 / 1 | 32 / 1.283 s |
| 1024 | 24 / 10 | state progression | `go to stoveburner 4` | Pot is picked up but not heated yet. | Heat the pot on stoveburner 2. | `A040` -> `move pot 1 to stoveburner 2` | 39 / 1 | 32 / 1.342 s |

Continue-loop counts (adjacent / two-cycle) were respectively: seed 1000 `0/42`,
1001 `9/0`, 1002 `0/28`, 1004 `44/42`, 1007 `38/35`, 1011 `0/0`,
1013 `0/0`, 1018 `0/0`, 1019 `40/38`, and 1024 `0/32`.

## Qualitative Annotation

Post-run annotations were performed only after all pairs completed and did not
alter execution:

| Dimension | Result |
|---|---:|
| Diagnosis correctness | 6 correct / 2 partial / 2 incorrect |
| Subgoal quality | 1 good / 4 partial / 5 poor |
| Bare-ID-to-subgoal agreement | 5 direct / 5 partial / 0 disagreement |
| Downstream effectiveness | 0 assessable / 10 blocked by selection failure |

The architecture appears to improve semantic linkage between the textual subgoal
and the model's intended ID relative to the old joint operator, but this is only a
post-run reading of malformed bare tokens. It cannot be counted as valid action
selection or causal recovery. Stage 1 also remained weak on immediacy: half of the
subgoals were poor, and two diagnoses continued off-target object reasoning.

## Historical Architecture Comparison

This is developmental architecture evidence, not a controlled head-to-head test,
because the prefix sets differ.

| Diagnostic | Joint v2 historical set | R2 two-stage fresh set |
|---|---:|---:|
| Prefixes | 5 | 10 |
| Structured diagnosis validity | 5 / 5 | 10 / 10 |
| Valid action selection | 5 / 5 | 0 / 10 |
| Diagnosis/subgoal-to-action agreement | 1 / 5 | 5 direct + 5 partial, analytical only |
| Beneficial / neutral / harmful | 1 / 4 / 0 | 0 / 10 / 0 |
| Mean generated tokens | 23.6 | 34.9 |
| Mean latency | 0.947 s | 1.410 s |

R2 separates diagnosis from action intent, and the bare IDs generally align with
the subgoals. Nevertheless, the intervention is scientifically unusable in its
frozen form because the second stage has zero protocol reliability and therefore
zero assessable downstream interventions.

## Artifacts

Raw runtime artifacts remain ignored under
`artifacts/runtime/recovery/r2-two-stage-development-20260820T110200Z/`.

- `selected_prefix_manifest.json`: `a317c2ba58f3b6ec5ace04ea6d1b2da4defa7662a7519ec961bc8e6c5c24b444`
- `run_manifest.json`: `98940749b7c9a1c8da5fb7994c49cdfa44f1e9e6f9563fe1412859fabf316edb`
- `pairs.jsonl`: `1d4c3ad5bb5d155264b88af2c67a0f7739a05fa1bac264e97417a80f37ca855d`
- `report.json`: `0cf897975e342799fa7c8312d7f34fa5f8c1d6f846c4a46d125bfd4c702742ac`
- `qualitative_annotations.json`: `22c5b32f0e5266c12d9f8ee7d2686babc13ffad5b3e6e8ebede728084ca3eb16`

The development set is now consumed. It must not be tuned on again or presented as
held-out evaluation evidence.

## Verification

- Full unit suite after the pilot: **82 passed**
- Frozen-manifest validation: 10 exact prefixes across six task families
- GPU used: NVIDIA A40
- Tracked runtime artifacts: none; `artifacts/runtime/` remains ignored
