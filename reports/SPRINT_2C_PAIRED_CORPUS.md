# Sprint 2C Paired Recovery Corpus

Status: complete. No selector was implemented or trained.

Recommendation: **C. RECOVERY OPPORTUNITY IS TOO SPARSE FOR CURRENT SELECTOR
STUDY.** Exactly 2 of 120 preregistered states had positive raw intervention
value. The two positives came from different episodes and task families, but
the count does not meet the gate minimum of three for another collection batch
or eight for selector development.

The corpus was deliberately stratified. Its overall beneficial fraction must
not be interpreted as the natural on-policy recovery rate.

## Frozen Protocol

- Base: `Qwen/Qwen3-8B`, `indexed_bounded_context_v1`, `k=4`, thinking off,
  greedy decoding, 32-token budget, indexed admissible actions.
- Recovery: R2-two-stage-v2 at `1b84f922a39f58c3b6ed2ba31cb29d896486b0d0`.
- Stage 1: frozen prompt, greedy, 40-token budget.
- Stage 2: frozen prompt, greedy, 16-token budget, exact bare-ID contract
  `^A\d{3}$`.
- One Stage-1 call and one Stage-2 call per recover branch; no retry, repair,
  fallback, memory, candidate scoring, second intervention, or selector.
- Exact inference revision:
  `2242534f86a40f7ff53010fd68eada1a5895a117`.

## H4 Source Pool

The committed schedule contained 200 initial episodes plus one conditional
100-episode extension. The initial pool produced 21 successes, so the extension
did not run.

| Measure | Result |
| --- | ---: |
| Episodes | 200 |
| Successes | 21 (10.5%) |
| Environment-terminal episodes | 194 |
| Fail-closed H4 selection-failure episodes | 6 |
| Invalid/inadmissible executions | 0 |
| Schedule SHA-256 | `1066ab7211afc0e4237023f0c5d1c28d31b558602ec90ad0dc49f06062b08863` |
| Trajectory SHA-256 | `2ea24e3ff988da75be6529f59cdbd890a9282f77d35baf2c0385eca85c58bcc4` |

All 200 schedule positions, nominal seeds 2000-2199, reset-order positions,
task IDs, and task families matched the preregistered schedule.

## Frozen Prefixes

Manifest SHA-256:
`5b3ee62ee1e5cd038ab4f8cd7865378fb1318794511b9505c7c06b3301bf3f2d`.

The manifest contains exactly 120 unique states from 79 episode groups, with a
maximum of two states per source episode. It covers all six task families and
early, middle, and late decision positions. The frozen strata are 20 successful
continuations, 30 past-loop failures, 30 high-uncertainty non-loop failures,
and 40 random non-loop failures. Entropy was numerically non-degenerate, so the
predefined uncertainty rule was used without fallback.

## Protocol Results

| Check | Result |
| --- | ---: |
| Exact branch reconstructions | 120/120 |
| Exact continue-branch H4 suffixes | 120/120 |
| Valid Stage-1 outputs | 117/120 |
| Incomplete Stage-1 outputs | 3/120 |
| Stage-1 outputs reaching 40 tokens | 4/120 |
| Valid Stage-2 bare IDs | 120/120 |
| Malformed / out-of-range / truncated Stage-2 outputs | 0 / 0 / 0 |
| ID-to-command mapping failures | 0 |
| Recovery actions not executed | 3 |

The three unexecuted actions were fail-closed Stage-1 truncations: one at
`P046` and two clustered in the same source episode at `P061` and `P067`.
Stage 2 still produced valid mapped IDs, but the joint operator correctly
withheld execution because Stage 1 was invalid. A fourth Stage-1 output reached
40 tokens at `P096` but was complete and valid.

The original ignored `report.json` counted the three withheld actions as
mapping failures. That was a summary-only bug. Raw pair records were not
modified; recomputation from them gives zero mapping failures. The correction
is implemented at `1ddb7bff5e95afb07d5af01b5f00b8ee5ee218c1` and recorded in
the corrected analysis artifact.

## Label Distribution

| Label | Count |
| --- | ---: |
| Beneficial (`V_raw=+1`) | 2 |
| Neutral (`V_raw=0`) | 115 |
| Harmful (`V_raw=-1`) | 3 |
| Continue fail / recover success | 2 |
| Continue success / recover fail | 3 |

Mean `V_raw` was -0.0083. Among 100 continuation failures, two recovered and
98 remained failures. Among 20 continuation successes, 17 remained successes
and three were harmed.

## Breakdown

Counts below are beneficial / neutral / harmful.

| Sampling stratum | B / N / H |
| --- | ---: |
| Successful continuation | 0 / 17 / 3 |
| Past-loop failure | 1 / 29 / 0 |
| High-uncertainty non-loop failure | 1 / 29 / 0 |
| Random non-loop failure | 0 / 40 / 0 |

| Task family | B / N / H |
| --- | ---: |
| Look-at-object-in-light | 1 / 15 / 0 |
| Pick-and-place | 1 / 20 / 1 |
| Clean-then-place | 0 / 21 / 1 |
| Cool-then-place | 0 / 18 / 0 |
| Heat-then-place | 0 / 20 / 0 |
| Pick-two-and-place | 0 / 21 / 1 |

| Group | B / N / H |
| --- | ---: |
| Early / middle / late | 1/57/3 / 0/38/0 / 1/20/0 |
| Past loop / non-loop | 1/29/0 / 1/86/3 |
| Entropy Q1 / Q2 | 0/25/1 / 0/27/1 |
| Entropy Q3 / Q4 | 1/20/0 / 1/43/1 |

Recovery reduced the recorded downstream loop count in 37 pairs, left it
unchanged in 66, and increased it in 17. A coarse observation-diversity proxy
was higher after recovery in 57 pairs, equal in 47, and lower in 16. These are
behavioral descriptors, not task-success substitutes.

## Recovery Cost

| Cost | Stage 1 | Stage 2 | Combined |
| --- | ---: | ---: | ---: |
| Mean input tokens | 517.225 | 648.583 | 1165.808 |
| Mean generated tokens | 29.767 | 5.000 | 34.767 |
| Mean latency, seconds | 1.129 | 0.277 | 1.406 sequential |

Across 120 interventions, recovery generated 4,172 tokens and used 168.686
seconds of sequential model latency. No cost penalty or lambda was selected.

## Qualitative Review

All five non-neutral cases and the preregistered 20-state neutral sample were
annotated after the run. Diagnoses were judged correct in 20 cases, partial in
four, and semantically correct but truncated in one. Subgoal-to-action agreement
was direct in 17, partial in five, and disagreement in three.

Beneficial cases:

- `P050`, seed 2157: diagnosis and subgoal correctly called for using the desk
  lamp. `use desklamp 1` completed the task immediately and broke the loop.
- `P033`, seed 2034: the diagnosis correctly identified repeated examination,
  but the stated sidetable subgoal disagreed with `go to bed 1`. The action was
  nevertheless causally correct: H4 found the book and completed the task.

Harmful cases:

- `P007`, seed 2061: searching for the second tissuebox before placing the held
  first tissuebox disrupted a successful ordering and introduced a loop.
- `P009`, seed 2036: an unsupported bathtub search diverted a successful spray
  bottle trajectory into handling unrelated objects.
- `P018`, seed 2070: going to the sink before taking the ladle caused H4 to
  clean a cup and oscillate instead of completing the successful plan.

The neutral review contained many semantically reasonable searches that
increased exploration without producing target progress. One notable case,
`P113`, found and picked up the target dish sponge but still failed to clean and
place it. The review therefore supports a long-horizon follow-through weakness,
not merely an action-grounding issue.

## Provenance

- R2-v2 protocol: `1b84f922a39f58c3b6ed2ba31cb29d896486b0d0`
- R2-v2 development result: `89ecf910d3febcd2c4556f268cff0132b1070ad6`
- H4 schedule commit: `8fd6c14b3a818e115dbae5e2bff71b00705adaf3`
- Corpus protocol commit: `80f0237520cc9cbf94357a703d7670d553582c7d`
- Prefix manifest commit: `761683aad6be88a44f11ea26549a27876a03ce82`
- Exact inference commit: `2242534f86a40f7ff53010fd68eada1a5895a117`
- Raw paired JSONL SHA-256:
  `785ffc5712b97fafc4ee9965e52eacef4bc390f7dfe4e8e227cc3cf38761f831`
- Original runtime report SHA-256:
  `cba79ec6664aefed166e66915d97091bfef03d6fac5449dcda5e8ae287f36153`
- Corrected analysis SHA-256:
  `4f70a08bec154298d23db99fa0fcf302f088e100375f773de8e8d4b6e74fdf5c`
- Qualitative annotations SHA-256:
  `ce30cf8a1a250fc2bb39bd6274975f5790f03de5c37bd49ced52df947af9296a`

Raw H4 trajectories, paired branches, prompts, generations, and detailed
annotations remain under ignored `artifacts/runtime`.
