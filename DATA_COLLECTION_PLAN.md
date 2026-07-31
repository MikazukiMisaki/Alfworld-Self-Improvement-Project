# Data Collection Plan

## Objective

Collect provenance-rich labels for whether one fixed reflection-and-regeneration intervention is beneficial at a particular ALFWorld state.

The target is **intervention value**, not merely episode failure. A label compares two futures from the same state prefix under the same remaining step budget.

## Unit of data

One intervention candidate contains:

- environment version, task ID, task family, split, seed, and episode prefix ID;
- step index, observation, action history, valid-action snapshot, and remaining horizon;
- base-policy action/generation trace and pre-action features;
- continue-branch trajectory and return;
- recover-branch trajectory and return;
- recovery prompt/version, output, parsed replacement action, token count, and latency;
- branch random seeds and replay/reconstruction status.

The record is immutable. It links to the source collection run and stores hashes for the policy, configuration, and prompt.

## Candidate-state sampling

Do not branch every step by default. Build a balanced collection pool using:

- all states with invalid-action/parser failure;
- all states following repeated observations or no state change;
- a stratified sample of ordinary states by task family and step index;
- a stratified sample of high and low raw token-entropy states;
- terminal-near and early-horizon states.

This prevents a selector from learning only obvious deadlocks and provides neutral/harmful examples needed for utility prediction.

## Trajectory branching protocol

For a base rollout, stop at candidate state s_t **before committing the next environment action**. Create paired branches from the identical prefix.

### Continue branch

1. Use the fixed base policy and normal action grounding.
2. Execute without reflection.
3. Continue until terminal state or fixed remaining horizon.
4. Record return, terminal success, episode length, tokens, and diagnostics.

### Recovery branch

1. Give the fixed recovery operator the same prefix, observation, valid actions, and allowed diagnostics.
2. The operator produces a short error analysis and one replacement action.
3. Validate/execute that action using the same environment action rules.
4. Resume the identical base policy, with no further recovery call in this branch.
5. Continue until terminal state or the same remaining horizon.
6. Record the same outputs plus recovery cost and explanation.

The base model, decoding policy, remaining horizon, and action-grounding policy are held constant. The recovery branch may differ only through the one recovery call and resulting action.

## Handling stochasticity

ALFWorld environment randomness and model decoding can make one branch comparison noisy.

- Use deterministic decoding for the core label corpus where possible.
- When stochastic decoding is necessary, run paired continuations with a predeclared number of matched seeds.
- Store both the individual branch outcomes and the mean/variance estimate.
- Label an example as uncertain when the estimated branch difference is not stable; keep it for robustness analysis but exclude it from the initial hard-label selector training.

## Intervention-value definition

For a candidate state i:

V_i = return(recover_i) − return(continue_i) − λ × recovery_cost_i

where return is terminal reward or a predeclared discounted return; recovery_cost is generated tokens and/or latency normalised to the evaluation budget; and λ is selected on development data then fixed.

Store both the continuous V_i and derived labels:

- **beneficial:** V_i > positive margin;
- **neutral:** within margin;
- **harmful:** V_i < negative margin.

The primary selector can rank continuous value or the probability of beneficial value. The paper must report all three categories.

## Oracle intervention

The oracle is an analysis upper bound, not a deployable method.

For an episode with a budget of k calls, it observes paired branch values and selects the k eligible states with the highest positive realised intervention value. It estimates:

- whether a recovery mechanism has enough opportunity to matter;
- regret of raw entropy, failure prediction, and the proposed selector;
- the gap between a perfect selector and a perfect recovery operator.

The oracle cannot be used to tune the recovery prompt or test protocol. It is computed separately on train, development, and test evaluation records after outcomes are frozen.

## Train, development, and test split

| Partition | Allowed use |
|---|---|
| Collection/train tasks | Generate branch labels, fit selector, inspect errors |
| Development tasks/templates | Choose features, calibration method, λ, eligibility rule, intervention budget, and prompt once |
| Test: valid_seen | Final within-distribution evaluation |
| Test: valid_unseen | Final generalization evaluation |

No task ID may appear in multiple partitions. Prefer splitting at task template/family level for a stricter transfer result; if that leaves insufficient data, report both ID-disjoint and template-disjoint settings.

The selector does not train on test branches. Test branching is allowed only to calculate the post-hoc oracle and analyse regret; it cannot alter the frozen model, prompt, budget, or decision rule.

## Leakage prevention

- Freeze task lists, prompts, policy revision, decoding parameters, and eligibility rules before final test runs.
- Do not use legacy memory lessons, success files, or DPO examples in the main corpus.
- Do not include task text, exact task ID, future observation, future reward, terminal state, or branch outcome as selector features.
- Keep trajectory prefixes from the same task instance in one partition only.
- Deduplicate equivalent prefixes and record source IDs.
- Store reflections separately from selector features unless their availability is part of the explicitly evaluated policy.
- Use retrieval-free recovery in the main experiment; this prevents task-memory leakage.
- Conduct feature audits to verify that filenames, batch IDs, split labels, and action-result strings do not leak outcomes.

## Quality-control checks

- Replay/reconstruction test: a subset of branch prefixes must reproduce the recorded prefix state exactly.
- Pair-consistency test: continue and recover branches have identical prefix, seed policy, horizon, and base policy.
- Prompt-identity test: all recovery arms use the same frozen prompt/version.
- Cost-accounting test: all recovery calls report tokens and latency.
- Manual audit: inspect a stratified sample of beneficial, neutral, harmful, and high-entropy examples.
- Class-balance report: publish candidate selection and label frequencies by task family and step index.

## Dataset scale guardrail

Begin with a pilot that establishes the recovery mechanism and branch reliability. Expand only after observing measurable positive, neutral, and harmful outcomes. A smaller clean paired corpus is preferable to a large corpus of noisy synthetic preferences.

