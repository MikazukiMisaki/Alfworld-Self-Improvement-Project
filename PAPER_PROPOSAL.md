# Paper Proposal

## Working title

**When Should Small Language Agents Spend Extra Reasoning? Recovery-Aware Intervention Selection in ALFWorld**

## Final research question

Given a fixed small language-model policy and a fixed-cost reflection-and-regeneration intervention, **can an agent predict which interaction states will gain enough reward from intervention to justify its cost?**

This is deliberately not the broader question “does reflection work?” It asks whether an agent can allocate a limited recovery budget better than always reflecting, never reflecting, or using raw uncertainty thresholds.

## Hypothesis

A selector trained on observable interaction features and model uncertainty can rank states by the expected marginal value of one recovery intervention. At a fixed intervention budget, this recovery-aware selector will improve success per added token/latency over no intervention, unconditional intervention, random selection, and raw token-entropy thresholding.

A secondary hypothesis is that failure probability alone is insufficient: some likely failures are not recoverable by one reflection, while some low-confidence states are harmless.

## Motivation

Small models are attractive for embodied and interactive tasks, but cannot afford to reflect, resample, or call a stronger model at every step. Current agent systems commonly trigger reflection by fixed schedules, terminal failure, or heuristic uncertainty. The legacy implementation has a token-entropy gate, but it does not test whether the triggered intervention changes the eventual outcome.

The useful operational decision is therefore not “is this action uncertain?” but “is one extra recovery attempt likely to improve this episode enough to warrant its cost?”

ALFWorld provides a controlled, sparse-reward, sequential setting where the same episode prefix can be replayed or reconstructed to compare continuation and intervention outcomes.

## Related-work positioning

- **Reflection and verbal memory:** Reflexion shows that language agents can use verbal feedback and episodic memory to improve future behavior. This paper fixes the reflection mechanism and studies selective allocation of it rather than proposing reflection itself. [Reflexion](https://arxiv.org/abs/2303.11366)
- **Iterative self-feedback:** Self-Refine establishes that a model can improve outputs through self-generated feedback. Our setting is sequential and budgeted: the central object is the decision to pay for one additional feedback cycle. [Self-Refine](https://arxiv.org/abs/2303.17651)
- **Self-improving agents:** Recent work studies learned reflections that transfer to unseen tasks. Our contribution is orthogonal: intervention selection is evaluated at the step level through paired continuation outcomes. [Training Language Agents to Learn from Experience](https://arxiv.org/abs/2605.20477)
- **Uncertainty estimation:** Raw token uncertainty need not be a useful intervention signal. We evaluate it as a baseline and require utility-based held-out evaluation. [Revisiting Uncertainty Estimation and Calibration of LLMs](https://arxiv.org/abs/2505.23854)
- **Preference optimisation:** DPO is not part of the main method. It remains future work or a small follow-up, avoiding a confound between online recovery and weight updates. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

## Method overview

The method has two fixed components and one learned/calibrated decision component.

1. **Base policy:** Qwen3-8B produces a grounded ALFWorld action.
2. **Recovery operator:** when called, it receives the current prefix, action/state diagnostics, and valid actions; it produces a concise error analysis and one replacement action. Its prompt and generation budget are fixed for all methods.
3. **Recovery-aware selector:** before acting, it scores the expected value of calling the recovery operator. The agent intervenes only at the highest-scoring states within a fixed episode or global budget.

The selector may use token/action uncertainty, action-parser status, valid-action information, repeated-observation/deadlock signals, step index, and compact trajectory features. It must not use future reward or terminal information at decision time.

## Algorithm description

For each candidate state s_t in a collection trajectory:

1. Record the policy action, generation trace, environment features, and prefix.
2. Construct two paired rollouts from the same prefix and with the same remaining horizon:
   - **continue:** execute the base policy normally;
   - **recover:** invoke the fixed recovery operator once, then execute the same base policy.
3. Define intervention value as the difference in discounted terminal return, with a fixed penalty for intervention tokens/latency.
4. Label the state as beneficial when recovery value is positive; retain the continuous value where available.
5. Fit/calibrate a selector on training prefixes only.
6. At evaluation time, rank eligible states by selector score and intervene under a predeclared budget.

The method is evaluated both as a binary beneficial-intervention predictor and as a budgeted decision policy. The latter is the paper's primary result.

## Expected contribution

- A precise formulation of **recovery value** for a bounded self-reflection intervention in a sequential language-agent environment.
- A paired-branching label protocol that separates failure likelihood from intervention usefulness.
- A recovery-aware selector for small agents, evaluated at fixed compute/intervention budgets.
- Controlled evidence showing where selective reflection improves, wastes compute, or fails on unseen ALFWorld tasks.

## Explicit non-contributions

The paper does not claim a new general reflection method, a universal uncertainty estimator, continuous lifelong learning, or cross-environment generalization. It does not require DPO, semantic memory, PPO/GRPO, hidden states, or a stronger model.

