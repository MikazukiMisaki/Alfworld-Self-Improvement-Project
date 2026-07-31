# Six-Month Research Roadmap

## Purpose and decision

This roadmap turns the thesis goal—**self-improvement of a small language model in an interactive environment**—into a focused empirical paper. It prioritizes one falsifiable contribution over a broad system claim.

**Recommended primary direction: uncertainty-guided selective self-improvement.** The central question is not whether reflection or DPO can help in general, but whether an 8B policy improves more efficiently when it spends reflection, retrieval, and preference-data budget only on decisions predicted to be risky.

The legacy implementation already contains entropy gating and reflection ideas, but has not defined, calibrated, or ablated them. This creates a feasible and well-grounded six-month contribution. The project will report negative results if uncertainty does not predict useful interventions.

## Milestones at a glance

| Month | Phase | Research deliverable | Engineering boundary |
|---|---|---|---|
| 1 | Foundation | Reproducible ALFWorld baseline and frozen data audit | Minimum trajectory/evaluation infrastructure |
| 2 | Baseline analysis | Failure taxonomy and intervention targets | Log importer and analysis tooling |
| 3 | Contribution selection | Pre-registered hypotheses and pilot results | Selective-gating interfaces |
| 4 | Method development | Full method and ablations | Modules required by the method |
| 5 | Experiments | Main results, robustness, and error analysis | Reproducible experiment runner |
| 6 | Paper | Draft, figures, and artifact package | Cleanup only for reproducibility |

## Phase 0 — Research foundation (weeks 1–4)

### Goal

Establish a reproducible baseline before any self-improvement claim.

### Work

- Pin ALFWorld/data version, Qwen3-8B revision, inference-library versions, hardware, prompt, decoding parameters, and seeds.
- Use ALFWorld `train` only for development and collection; reserve `valid_seen` and `valid_unseen` for reported evaluation.
- Define one canonical trajectory format and import legacy logs as a frozen, provenance-labelled dataset.
- Implement and validate a plain Qwen3-8B policy with action grounding. Compare free generation and admissible-action-constrained selection only if both are reported.
- Define a fixed task list, episode budget, at least three seeds where compute permits, and a run manifest for every result.

### Exit criteria and outputs

- A baseline table with success rate, reward, episode length, invalid-action rate, and cost/latency on seen and unseen validation splits.
- Legacy-data audit: 210 logged episodes, 28 terminal successes, and explicit data-quality flags for malformed and duplicate DPO examples.
- A baseline reproduction note stating exactly which legacy behavior was and was not reproduced.

## Phase 1 — Baseline analysis (weeks 5–8)

### Goal

Understand why current self-improvement attempts fail or help before training a new method.

### Studies

1. **Qwen3-8B behavior.** Characterize success by task family, horizon, receptacle state, invalid action, repeated observation, and action grounding.
2. **Reflection effectiveness.** Compare no reflection, unconditional post-failure reflection, and oracle reflection injection on a fixed retry budget. Score reflections for error localization and whether their suggested correction is executable.
3. **DPO improvement.** Compare base model, legacy-data DPO (audit-labelled), and a cleaned preference baseline. Evaluate the trained adapter in the rollout policy; training loss alone is not a result.
4. **Failure patterns.** Produce a mutually exclusive taxonomy: task parsing, state tracking, invalid syntax, precondition violation, navigation/exploration, object-identity error, loop/deadlock, and incorrect recovery.

### Expected outputs

- Baseline and retry results with confidence intervals.
- A labelled failure sample and taxonomy, with inter-annotator agreement if a second annotator is available; otherwise state the single-annotator limitation.
- Three research questions selected by observed bottlenecks rather than intuition.

## Phase 2 — Research contribution selection (weeks 9–10)

| Candidate | Novelty relative to prior work | Difficulty | Experimental requirement | Six-month verdict |
|---|---|---|---|---|
| Step-level credit assignment | Strong if counterfactual action attribution is accurate; related to token-level DPO/Q-function views | High | Action-level labels and robust causal controls | Secondary/stretch goal |
| Reflection-quality evaluation | Useful measurement contribution; reflection methods are established | Medium | Human or proxy labels plus intervention tests | Strong supporting analysis |
| Uncertainty-guided self-improvement | Selective intervention bridges raw uncertainty and useful corrective action | Medium | Calibrated labels and matched intervention budget | **Primary recommendation** |
| Better preference construction | Practical and likely effective, but crowded | Medium | Comparable positive/negative trajectories and DPO runs | Supporting mechanism |
| Memory-based self-improvement | Closely related to Reflexion-style episodic verbal memory | Medium | Retrieval controls and task-split controls | Supporting baseline |

### Related-work positioning

Reflection and verbal memory have already shown that language agents can improve through trial feedback without weight updates ([Reflexion](https://arxiv.org/abs/2303.11366)). Iterative self-feedback is also established ([Self-Refine](https://arxiv.org/abs/2303.17651)). DPO provides a lightweight preference-optimization baseline ([Rafailov et al.](https://arxiv.org/abs/2305.18290)); token-level views motivate credit-assignment questions ([Rafailov et al., 2024](https://arxiv.org/abs/2404.12358)).

The paper should not claim to invent reflection, memory, DPO, or entropy. Its claim should be that **calibrated uncertainty can decide when a small interactive policy should invoke a bounded corrective mechanism**, and that this selection improves the benefit/cost trade-off.

### Decision gate

Proceed only if a pre-intervention signal predicts at least one operational target—invalid action, deadlock, or eventual failure—on a held-out development set better than step index, action length, and repeated observation. Otherwise pivot to the reflection-quality evaluation study.

## Phase 3 — Method development (weeks 11–16)

### Algorithm: Selective Reflect-and-Recover (SRR)

At each step, the policy emits an action and uncertainty features. A calibrated gate estimates whether continuing unaided is risky. If it fires, the agent receives one bounded corrective intervention: retrieve a provenance-backed failure lesson, generate a structured recovery suggestion, and choose a valid action once more. The gate cannot see future reward.

The offline variant constructs preference examples only from high-risk or observed-error steps, with a corrected alternative and explicit ranking reason. This permits a clean comparison between uniform DPO and selective DPO.

### Required modules

- Environment adapter and action validator.
- Generation trace collector: token scores, action candidates, latency, and decoding settings.
- Uncertainty estimators: token entropy first; optional self-consistency or action-distribution margin if compute permits.
- Calibrator and gate policy.
- Structured reflection/recovery generator and memory retriever.
- Preference builder with source trajectory IDs and ranking rationale.
- Evaluator and run-artifact store.

### Ablations

- No intervention; unconditional intervention; random intervention at matched rate; oracle failure gate; SRR gate.
- Entropy versus repeated-observation versus combined gate features.
- Reflection only, retrieval only, and reflection plus retrieval.
- Uniform DPO versus selective DPO, with equal numbers of preference examples.
- Seen versus unseen and low- versus high-horizon tasks.

## Phase 4 — Experiments (weeks 17–21)

### Baselines

- Qwen3-8B direct policy.
- Action-grounded Qwen3-8B policy.
- Unconditional Reflexion-style retry/memory baseline.
- Legacy-style entropy-threshold baseline, clearly marked as uncalibrated.
- Standard DPO on an equal-size cleaned preference dataset.

### Datasets and metrics

- ALFWorld `train` for collection/development and `valid_seen`/ `valid_unseen` for evaluation, stratified by task family.
- Primary: success rate and mean reward.
- Efficiency: episode length, intervention count/rate, tokens, latency, and preference examples per improvement point.
- Diagnostic: invalid-action rate, deadlock rate, recovery success after a gate, reflection usefulness, calibration error/Brier score/AUROC, and generalization gap.

### Planned tables and figures

1. Main performance and intervention-cost table, seen/unseen splits.
2. Gate prediction/calibration table and reliability diagram.
3. Method ablation table at matched intervention budget.
4. Learning curve: success versus collected trajectories/preference examples.
5. Failure-taxonomy transition plot before and after SRR.
6. Three case studies: successful recovery, unnecessary intervention, and confident failure.

## Phase 5 — Paper preparation (weeks 22–26)

### Candidate title

**When Should Small Language Agents Reflect? Uncertainty-Guided Self-Improvement in Interactive Environments**

### Abstract outline

1. Small agents need sample- and compute-efficient improvement in sequential environments.
2. Existing reflection and preference methods commonly intervene uniformly.
3. Introduce SRR: a calibrated, bounded gate for reflection/retrieval and selective preference construction.
4. Evaluate on ALFWorld with fixed seen/unseen protocols and cost accounting.
5. Report improvement, ablations, calibration, and failure-mode evidence.

### Contribution list

- A controlled formulation of selective self-improvement for language agents.
- A calibrated uncertainty-to-intervention method and matched-budget ablations.
- A provenance-rich ALFWorld trajectory/preference protocol for small-model self-improvement.
- Analysis of when intervention helps, harms, and fails to generalize.

### Paper structure

Introduction; related work; problem formulation; SRR method; experimental protocol; results; analysis and limitations; conclusion; reproducibility appendix with prompts, configurations, task lists, and data lineage.

## Scope controls

ALFWorld is the paper environment. WebShop, ScienceWorld, MiniGrid, and Minecraft are interface-validation targets, not required evaluation suites for this six-month paper. PPO/GRPO, hidden-state probes, full counterfactual credit assignment, and multi-environment claims are deferred unless the primary result is complete early.

