# Sprint 2 R2 Two-Stage V2 Development Pilot

## Decision

**FREEZE R2-TWO-STAGE-V2 FOR LARGER PAIRED CORPUS.**

The protocol-level bare-ID revision passed its development gate. Both stages were
operationally valid in all ten fresh pairs, one recovery changed a failed continuation
into task success, no recovery was harmful, and intervention value was heterogeneous.
The frozen operator is suitable for larger paired-corpus collection, not yet for a
claim about population-level recovery effectiveness.

Do not tune this operator on these ten consumed prefixes. This decision does not
authorize a selector, candidate scoring, constrained decoding, memory, training,
retry, fallback, repair, or changes to H4.

## Frozen Protocol

- Pre-inference protocol commit: `1b84f922a39f58c3b6ed2ba31cb29d896486b0d0`
- Frozen H4: `Qwen/Qwen3-8B`, `indexed_bounded_context_v1`, `k=4`
- H4 generation: thinking disabled, greedy, `max_new_tokens=32`
- Stage 1: unchanged v1 prompt and parser, greedy, `max_new_tokens=40`
- Stage 2: unchanged semantic context, greedy, `max_new_tokens=16`
- Sole v2 protocol change: exact bare-ID response `^A\d{3}$`
- Grounding: the exact ID must exist in the current environment-owned mapping
- Intervention: one Stage-1 call, one Stage-2 call, one mapped action, then frozen H4
- No retry, dual-format parser, fallback, repair, constrained decoding, or ranking

Tests assert that Stage 1 uses the same v1 prompt function and that the complete
Stage-2 decision context after the output instructions is unchanged from v1.

## Preregistration And Replay

The ten-prefix manifest was fixed, checksummed, tested, and committed before model
loading:

- Manifest SHA-256: `2ca66693af3e7b1e4650d747aee07fe32d68c5204e06178718086abc12aa22cf`
- Prefixes: exactly 10 previously unused failed H4 trajectories
- Coverage: five task families and varied loop, stall, off-target, transformation,
  state-progression, and ordinary failure contexts
- Permanently excluded prior development seeds: all five joint-recovery seeds and
  all ten R2-v1 seeds
- Formal H4 trajectory SHA-256: `48ddb86e33957549d34f8c449870c027f73100d64cdee642c04d98a625e7223b`
- Formal schedule SHA-256: `ee79fae0c62c618c4495c5c9ff45b8a95a463e7775ee82d55c922418a9273441`

All pairs passed exact task, schedule position, replayed prefix, observation,
admissible-action mapping, hidden/public reconstruction, remaining-horizon, and
deterministic H4 continuation checks. Every recover branch made exactly two model
calls and exactly one mapped environment action, followed by unchanged H4.

## Protocol Results

| Measure | Result |
|---|---:|
| Stage-1 valid | 10 / 10 |
| Stage-1 malformed / incomplete | 0 / 0 |
| Stage-1 token-cap reached | 1 / 10, complete with EOS |
| Stage-2 valid in-range bare ID | 10 / 10 |
| Stage-2 malformed / out-of-range | 0 / 0 |
| Stage-2 incomplete / token-cap reached | 0 / 0 |
| Exact mapped recovery actions executed | 10 / 10 |
| Mapping failures | 0 |

The raw runner aggregate used the name `stage_one_truncated_count` for token-cap
hits. Detailed generation records show that the one cap-reaching output ended with
EOS, was complete, and passed the Stage-1 schema. True Stage-1 truncations were zero.
The post-run reporting code now records incomplete outputs and cap hits separately;
no inference was rerun.

## Causal Outcomes

| Measure | Result |
|---|---:|
| Beneficial / neutral / harmful | 1 / 9 / 0 |
| Continue-fail / recover-success | 1 |
| Continue-success / recover-fail | 0 |
| `V_raw` distribution | `+1`: 1, `0`: 9 |
| Mean `V_raw` | 0.1 |
| Mean continue / recover remaining length | 41.3 / 37.4 |
| Continue / recover branches with loop indicators | 8 / 7 |

The positive case was seed `1003`. H4 continued its dishsponge take/move cycle for
44 remaining steps and failed. Recovery selected `A023`, exactly mapped to
`open cabinet 2`; resumed H4 then found the soap bottle and completed the task in
five remaining actions. This is a genuine `continue=0`, `recover=1` causal pair.

## Per-Prefix Outcomes

Loop columns show adjacent-repeat / two-cycle event counts over each remaining
branch.

| Seed | Context | Continue action | Recovery ID -> command | C/R return | C/R success | C/R length | C loops -> R loops |
|---:|---|---|---|---:|---|---:|---|
| 1003 | strict A-B-A-B | `take dishsponge 1 from cabinet 1` | `A023` -> `open cabinet 2` | 0 / 1 | no / yes | 44 / 5 | 0/41 -> 0/0 |
| 1008 | adjacent stall | `examine towel 1` | `A015` -> `go to towelholder 1` | 0 / 0 | no / no | 39 / 39 | 19/0 -> 0/30 |
| 1012 | ordinary non-loop | `go to desk 1` | `A009` -> `go to shelf 2` | 0 / 0 | no / no | 45 / 45 | 0/0 -> 0/0 |
| 1014 | off-target transformation | `examine diningtable 1` | `A000` -> `examine diningtable 1` | 0 / 0 | no / no | 38 / 38 | 1/31 -> 1/31 |
| 1015 | off-target objects | `examine dresser 1` | `A000` -> `examine dresser 1` | 0 / 0 | no / no | 45 / 45 | 0/6 -> 0/6 |
| 1017 | strict navigation A-B-A-B | `go to stoveburner 1` | `A002` -> `examine microwave 1` | 0 / 0 | no / no | 42 / 42 | 0/39 -> 0/25 |
| 1020 | adjacent stall | `examine dresser 1` | `A000` -> `examine dresser 1` | 0 / 0 | no / no | 44 / 44 | 43/41 -> 43/41 |
| 1021 | adjacent stall | `examine cabinet 3` | `A000` -> `close cabinet 3` | 0 / 0 | no / no | 35 / 35 | 34/32 -> 2/2 |
| 1023 | wrong target cleaning | `go to countertop 1` | `A046` -> `go to fridge 1` | 0 / 0 | no / no | 38 / 38 | 0/31 -> 0/32 |
| 1025 | ordinary non-loop | `take book 1 from bed 1` | `A001` -> `go to drawer 1` | 0 / 0 | no / no | 43 / 43 | 0/0 -> 0/0 |

Loop behavior improved in three pairs, shifted into a different loop in one,
worsened slightly in one, and was unchanged in five. Seed `1021` showed meaningful
state progress without success: closing the stalled cabinet led to broader cabinet
exploration and reduced both loop counts from `34/32` to `2/2`.

## Semantic Annotation

Annotations were performed after all ten branches completed and never influenced
execution.

| Dimension | Result |
|---|---:|
| Diagnosis correctness | 9 correct / 1 partial / 0 incorrect |
| Immediate-subgoal quality | 3 good / 4 partial / 3 poor |
| Subgoal-to-action agreement | 5 direct / 2 partial / 3 disagreement |
| Downstream progress | 1 task success / 1 meaningful / 5 temporary / 3 none |

The model reliably recognized the immediate deficit, but subgoal precision remains
mixed. Three selections did not implement the stated subgoal, and three subgoals
were too broad, mechanism-confused, or insufficiently immediate. These limitations
should remain frozen and measured in the larger corpus rather than tuned on this
development set.

## Cost

| Cost | Stage 1 | Stage 2 | Total intervention |
|---|---:|---:|---:|
| Mean input tokens | 498.8 | 627.2 | 1,126.0 |
| Mean generated tokens | 28.7 | 5.0 | 33.7 |
| Total generated tokens | 287 | 50 | 337 |
| Mean latency | 1.084 s | 0.271 s | 1.354 s sequential |
| Total latency | 10.837 s | 2.707 s | 13.543 s sequential |

## Interpretation

R2-v1 and R2-v2 used different consumed development sets, so their task outcomes
must not be compared statistically. At the protocol level, v2 resolved the exact
failure it was designed to address: ten unique bare IDs became ten valid grounded
selections with no permissive parsing or repair.

Option 3 is not warranted because the simplified generative action interface was
operationally reliable in 10/10 cases. Option 2 is not selected because the fresh
pilot contains non-zero causal benefit, no harmful case, mostly correct diagnoses,
and useful heterogeneity for paired-corpus collection. The larger corpus is needed
to estimate how often the moderate semantic weaknesses matter.

## Artifacts

Raw runtime artifacts remain ignored under
`artifacts/runtime/recovery/r2-two-stage-v2-development-20260820T114458Z/`.

- `selected_prefix_manifest.json`: `2ca66693af3e7b1e4650d747aee07fe32d68c5204e06178718086abc12aa22cf`
- `run_manifest.json`: `819140f4d09fefe2bbdb03156e5a1f2c4a8bbc17ae09d18f3705a2293a5876c6`
- `pairs.jsonl`: `9d6666642904fee1183af5156e276c1625d207ad394b7c31ab6d9908056e3dc5`
- `report.json`: `d5181f5a96b56e7cf5838cb8d2b161407635329c49ec097297a36b307ee601e4`
- `qualitative_annotations.json`: `5b20576d91caa89734827a1c8e3e6790a309ee29bf8258f7172384e8125e8feb`

These ten seeds are now consumed development evidence and are permanently ineligible
for tuning or held-out evaluation claims.

## Verification

- Full unit suite after the pilot: **94 passed**
- Frozen-manifest validation: 10 exact unused prefixes across five task families
- GPU used: NVIDIA A40
- Tracked runtime artifacts: none; `artifacts/runtime/` remains ignored
