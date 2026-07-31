# Implementation Phases

## Scope rule

Implement only what is necessary to answer the paper question: whether a selector can allocate one fixed recovery intervention better than matched alternatives. The phases below are gates, not a promise to build a general agent platform.

Do not implement DPO, memory retrieval, PPO/GRPO, additional environments, vector search, planners, or counterfactual credit assignment beyond the paired branch protocol.

## Phase 1 — Baseline reproduction

### Goal

Produce a frozen, action-grounded Qwen3-8B ALFWorld baseline on fixed seen and unseen validation task sets.

### Minimum work

- ALFWorld environment adapter with reset, step, task, and valid-action access.
- Qwen policy boundary with prompt, generation metadata, and action parser.
- Canonical trajectory and run-manifest records.
- Deterministic or documented stochastic evaluation runner.
- Metrics: success, reward, length, invalid action, tokens, and latency.

### Deliverable and gate

A reproducible baseline report with trajectories and resolved configurations. Proceed only when the same task/seed schedule is stable enough for paired comparisons.

## Phase 2 — Trajectory collection and branch labels

### Goal

Create a small, high-quality intervention-value dataset from paired continuations.

### Minimum work

- Candidate-state sampler and prefix persistence/replay or deterministic reconstruction.
- Continue and one-recovery branch executor with identical remaining horizon.
- Immutable branch-pair record, data split enforcement, and provenance checks.
- Simple branch-quality audit and label summary.

### Deliverable and gate

A pilot corpus containing beneficial, neutral, and harmful recovery examples across multiple task families. Proceed only if one fixed recovery intervention beats or differs meaningfully from matched extra generation on at least a development subset.

## Phase 3 — Fixed reflection intervention

### Goal

Establish whether the recovery operator itself is useful before attempting selection.

### Minimum work

- One frozen structured recovery prompt: short diagnosis plus one replacement action.
- Same valid-action handling and token limit across all recovery baselines.
- No-recovery, unconditional recovery, random matched-rate recovery, and raw-entropy threshold baselines.
- Recovery-effect analysis by failure category.

### Deliverable and gate

A development result showing measurable heterogeneity: recovery helps some states and is neutral or harmful on others. If recovery is uniformly ineffective, stop and revise the recovery operator once; if it remains ineffective, pivot the thesis to reflection-quality evaluation rather than building a selector.

## Phase 4 — Recovery-aware selector

### Goal

Predict and rank marginal intervention value under a limited budget.

### Minimum work

- Pre-action feature extraction: uncertainty, parser/action validity signals, observation repetition, step index, and compact history features.
- Simple calibrated selector; begin with transparent methods before a learned sequence model.
- Feature-only, entropy-only, failure-probability, and combined selector variants.
- Budgeted decision rule plus oracle analysis.

### Deliverable and gate

On held-out development tasks, the selector ranks beneficial interventions above random and raw entropy, with documented calibration and a nontrivial gap to the oracle. Freeze features, calibration, prompt, and budget after this gate.

## Phase 5 — Final evaluation and paper evidence

### Goal

Produce the locked final comparison on valid_seen and valid_unseen.

### Minimum work

- Run B0–B6 and the proposed method on fixed task/seed pairs.
- Compute paired confidence intervals, cost metrics, selector utility, oracle regret, and failure-category results.
- Generate tables, figures, and auditable case studies.
- Run a leakage/provenance audit and document limitations.

### Deliverable

A paper-ready result package containing:
- the main fixed-budget performance/cost table;
- selector and oracle utility table;
- budget curve and calibration figures;
- seen/unseen and failure-category ablations;
- negative/harmful-intervention cases;
- reproducibility appendix inputs: task lists, seeds, prompts, resolved configurations, and artifact manifests.

## Six-month scheduling

| Month | Primary phase | Required decision |
|---|---|---|
| 1 | Phase 1 | Is the baseline trustworthy? |
| 2 | Phase 2 | Does paired recovery have measurable opportunity? |
| 3 | Phase 3 | Is the fixed intervention worth selecting? |
| 4 | Phase 4 | Does selection beat random and entropy on development? |
| 5 | Phase 5 | Lock and run final tests |
| 6 | Analysis and paper | Write; do not add methods |

## Stop conditions

- If action grounding alone explains all gains, publish it as a baseline finding and do not claim self-improvement.
- If recovery has no positive opportunity, pivot to measuring reflection quality/recoverability rather than training a selector.
- If the selector does not beat random selection, report the negative result or simplify to an uncertainty-analysis paper.
- If compute prevents multiple seeds or fixed task lists, reduce methods and tasks before weakening the evaluation protocol.

