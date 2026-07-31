# Research Questions

## Framing

The legacy project demonstrates a useful observation: small language-model agents can generate failures, reflections, memories, and preference examples from interaction. It does not establish which of these causes improvement, at what granularity it acts, or whether the available signals are reliable.

The questions below are designed to produce publishable evidence on ALFWorld within six months. They draw on reflection/memory work such as [Reflexion](https://arxiv.org/abs/2303.11366), iterative self-feedback work such as [Self-Refine](https://arxiv.org/abs/2303.17651), and preference optimization via [DPO](https://arxiv.org/abs/2305.18290). They are intentionally narrower than a claim of general self-improving intelligence.

## RQ1 — Why does self-improvement work when it works?

### Motivation

An outcome improvement can arise from several mechanisms: better task understanding, action grounding, recovery after local error, extra inference budget, leakage from repeated tasks, or genuine parameter change. Treating all of these as “self-improvement” obscures the thesis question.

### Current limitation

The legacy pipeline mixes admissible actions, prompt changes, entropy-triggered retries, memory, synthetic DPO negatives, and changed scripts. It cannot attribute an observed performance difference to one mechanism.

### Hypothesis

Most near-term gains for Qwen3-8B in ALFWorld will come from preventing or recovering from local action/precondition failures, rather than from globally improved long-horizon planning.

### Experiment design

- Fix the policy, prompt, task set, and action interface.
- Compare direct execution, one extra unconditioned generation, action grounding, reflection/retrieval intervention, and DPO.
- Label outcomes by the failure taxonomy and report which categories change.
- Match inference-token and intervention budgets across methods.
- Test on validation tasks that were not used to construct memory or preference data.

**Falsifier:** gains remain after action grounding is removed, do not concentrate in recoverable categories, and exceed what matched extra generation explains.

## RQ2 — When does reflection help?

### Motivation

Reflections can be insightful but irrelevant, incorrect, or redundant. A fluent explanation is not evidence that it improves an embodied policy.

### Current limitation

Legacy lessons are one mutable rule per task, generated from text logs and not scored for grounding, novelty, or causal usefulness. The system lacks no-reflection and random-reflection controls.

### Hypothesis

Reflection helps primarily after an externally observable local failure (invalid action, repeated observation, unmet precondition), and is less useful when the failure is due to missing exploration or an incorrect task-state model.

### Experiment design

- Generate structured reflections with cited step evidence and a proposed correction.
- Compare no reflection, unconditional reflection, reflection after failure, random retrieved reflection, and oracle selected reflection.
- Define usefulness as conditional recovery success on a matched retry, not language-model judge preference alone.
- Have a small human-labelled sample assess error localization and actionability; correlate those labels with recovery.

**Falsifier:** unconditional reflection is equally effective, or reflection quality has no relationship to recovery.

## RQ3 — How can small models improve under limited data and compute?

### Motivation

An 8B model cannot afford unlimited online exploration, repeated retries, or large-scale preference training. The practical question is how to allocate a finite self-improvement budget.

### Current limitation

The legacy DPO set has 192 examples, 97 exact duplicates, generic synthetic negatives, and no source lineage. Its training run is disconnected from a controlled evaluation of the resulting adapter.

### Hypothesis

Selective, provenance-backed preference examples from error-prone/recoverable steps improve data efficiency relative to uniform preference construction of equal size.

### Experiment design

- Build a cleaned preference dataset whose examples retain chosen/rejected source IDs and a ranking reason.
- Compare no training, uniform DPO, random subset DPO, and uncertainty-selected DPO at equal data and optimization budgets.
- Plot success against preference-example count and against rollout-collection cost.
- Separate seen and unseen task families and audit duplicates/leakage before training.

**Falsifier:** uniform DPO matches or exceeds selective DPO at all data budgets, or gains vanish on held-out task families.

## RQ4 — How should failures be attributed to previous actions?

### Motivation

An episode-level reward says that something went wrong, but does not say which action created the error or whether a later action could have repaired it. This limits reflection quality and preference construction.

### Current limitation

The legacy data treats successful actions as chosen and invents generic rejected actions. It does not distinguish causal errors, harmless detours, and recoverable actions.

### Hypothesis

Simple, observable step labels—invalid action, no state change, repeated observation, and violated precondition—provide useful credit-assignment signals, even without expensive full counterfactual replay.

### Experiment design

- Annotate each step with observable signals and failure-taxonomy labels.
- Compare episode-level versus step-selected reflection and preference examples.
- For a small stratified subset, run bounded counterfactual alternatives from saved/recreated states when ALFWorld permits; use this as a validation set for heuristic attribution.
- Evaluate whether step-level selection improves recovery prediction or data efficiency.

**Falsifier:** observable step labels do not predict recovery or agree with counterfactual outcomes better than a step-index baseline.

This is a strong secondary contribution but should not become the primary method unless the environment supports reliable replay early in the project.

## RQ5 — Can uncertainty guide self-improvement?

### Motivation

Uncertainty is valuable only if it predicts an actionable need for intervention. The legacy system computes token entropy and gates retrieval with manually chosen thresholds, but has no calibration target or budget-matched control.

### Current limitation

Token entropy may reflect decoding temperature, vocabulary ambiguity, or long output length rather than a harmful decision. A high-entropy gate can add unnecessary cost, while confident failures may never trigger it.

### Hypothesis

A calibrated combination of action uncertainty and interaction-state signals predicts local failure/recoverability better than either raw entropy or fixed thresholding. Selective intervention at those points improves success per unit cost over unconditional intervention.

### Experiment design

- Define prediction targets before modelling: invalid action, deadlock within the next k steps, eventual failure, and recoverable local error.
- Compare entropy, action probability margin, repeated-observation features, step index, and a calibrated combination.
- Report AUROC/AUPRC, Brier score, expected calibration error, reliability diagrams, and decision utility at fixed intervention budgets.
- Compare no gate, random matched-rate gate, raw-entropy threshold, oracle gate, and calibrated gate.
- Report confident-failure cases separately; a gate that only catches uncertainty is incomplete.

**Falsifier:** the calibrated gate does not beat random selection at the same intervention rate or does not improve success/cost over unconditional intervention.

## RQ6 — Does memory generalize, or merely memorize task wording?

### Motivation

Task-keyed lesson memory can improve repeated tasks while failing under paraphrase, object substitutions, or unseen task families. The thesis needs to separate retrieval benefit from leakage.

### Current limitation

The legacy memory bank uses mutable exact task strings and a generic fallback. It contains no provenance, retrieval score, conflict management, or held-out generalization protocol.

### Hypothesis

Failure lessons generalized at the task-family or precondition level transfer more reliably than exact-task lessons, but only when retrieval is gated by a compatible failure state.

### Experiment design

- Store exact-task, task-family, and generic lessons separately with source trajectory IDs.
- Evaluate exact-match retrieval, semantic/family retrieval, random memory, and no memory.
- Split by task template/family and evaluate paraphrase/object substitution where possible.
- Measure retrieval precision, intervention rate, recovery benefit, and harmful-memory rate.

**Falsifier:** exact task matching explains all gains, family-level retrieval is harmful, or retrieved memory is no better than random memory.

## Evidence standards across questions

- Use fixed validation task lists, held-out task families where possible, and multiple seeds.
- Report matched compute/intervention budgets; an extra retry is an intervention, not a free improvement.
- Preserve task, trajectory, model, prompt, decoding, and data-lineage identifiers.
- Include negative cases and uncertainty intervals rather than only aggregate success.
- Treat ALFWorld as the initial testbed; do not claim cross-environment generality without additional evaluation.

