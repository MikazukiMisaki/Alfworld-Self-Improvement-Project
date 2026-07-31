# Experiment Plan

## Goal and timeline

The paper evaluates whether recovery-aware selection improves ALFWorld performance per unit of extra reasoning. The six-month schedule reserves months 1–2 for a stable baseline and labelled branching corpus, months 3–4 for selector/recovery experiments, month 5 for final evaluation, and month 6 for writing.

The primary experimental unit is a fixed task instance and seed. The primary comparison is matched by intervention count and recovery-generation budget.

## Baseline agents

| ID | Agent | Purpose |
|---|---|---|
| B0 | Base Qwen3-8B, grounded action policy | Required reference point |
| B1 | Base policy plus admissible-action control | Separates action grounding from recovery |
| B2 | Fixed recovery at every eligible state | Upper-cost reflection baseline |
| B3 | Random recovery at matched intervention rate | Tests whether selection, not occasional recovery, helps |
| B4 | Raw token-entropy threshold | Legacy-style heuristic uncertainty baseline |
| B5 | Failure-probability selector | Separates predicted failure from predicted recovery value |
| B6 | Oracle recovery-value selector | Upper bound on selector opportunity |
| M1 | Recovery-aware selector | Proposed method |

The recovery operator, prompt, decoding settings, maximum output length, and valid-action handling are identical for B2–M1. B0/B1 have no recovery tokens.

## Datasets and task sets

- **Collection/train:** ALFWorld training tasks only. Use a fixed, versioned list of task IDs and family labels.
- **Development:** disjoint training-task IDs or held-out task templates, used once for feature choice, calibration, threshold/budget selection, and prompt finalization.
- **Test:** ALFWorld valid_seen and valid_unseen, never used for branching labels, prompt tuning, selector fitting, or memory creation.
- **Analysis subset:** a stratified sample by task family, episode horizon, and failure signal for manual reflection-quality review.

Report task-family counts and all exclusions. No legacy memory or legacy DPO data is used in the main result; it is audit data only.

## Evaluation protocol

1. Freeze base policy, prompt, decoding parameters, maximum horizon, task lists, and random-seed schedule.
2. Run every method on the same task-instance/seed pairs.
3. Define an eligible state before evaluation, such as a nonterminal step where a recovery can still be applied.
4. Give B2–M1 the same maximum recovery calls per episode and the same recovery token limit. For global-budget results, fix the total recovery calls across the evaluation set.
5. Use a selector trained only on the collection set and locked after development selection.
6. Evaluate on seen and unseen validation sets, with multiple stochastic decoding seeds where practical.
7. Use paired bootstrap confidence intervals across task instances and a paired significance test declared in advance.
8. Log all branch/recovery decisions, raw model output, valid-action status, token counts, latency, rewards, and terminal results.

The main endpoint is test-set success under a fixed extra-reasoning budget. Do not tune on test performance.

## Metrics

### Primary

- Success rate at fixed intervention count/token/latency budget.
- Incremental success per recovery call.
- Incremental success per added generated token.

### Selector quality

- AUROC and AUPRC for beneficial-intervention labels.
- Brier score and expected calibration error where scores are probabilistic.
- Top-k recovery value: average realised benefit among the k selected states.
- Regret relative to the oracle selector.

### Agent behaviour

- Mean reward and episode length.
- Invalid-action rate.
- Deadlock/repeated-observation rate.
- Conditional recovery rate: success of an intervention after a locally observable error.
- Harmful-intervention rate: recovery reduces return relative to continue.
- Fraction of confident failures missed by the selector.

### Cost

- Recovery calls per episode.
- Generated tokens, model forward passes, and wall-clock latency.
- Collection branching cost, reported separately from online evaluation cost.

## Ablation plan

| Ablation | Question answered |
|---|---|
| No selector / no recovery | Is any improvement present? |
| Unconditional vs random matched-rate recovery | Does selective allocation matter? |
| Raw entropy vs failure selector vs recovery-value selector | Is recovery value distinct from uncertainty/failure likelihood? |
| Token entropy only vs trajectory features only vs combined features | Which signals carry useful information? |
| One reflection only vs reflection plus valid-action regeneration | Does language feedback itself help beyond constrained re-decoding? |
| Different budgets: 0, low, medium, high | Does the method remain useful under realistic scarcity? |
| Seen versus unseen | Does selection generalize beyond development tasks? |
| Failure category slices | Which failures are recoverable by this intervention? |

Do not add DPO, semantic retrieval, or multiple reflectors until the core table is complete. If time remains, one appendix experiment can test whether selective preference construction inherits the same benefit.

## Expected tables

1. **Main result:** success, reward, cost, and improvement-per-cost for B0–M1 on seen and unseen validation.
2. **Selector utility:** AUROC/AUPRC, calibration, top-k value, harmful-intervention rate, and oracle regret.
3. **Ablations:** feature sets, recovery variants, and budgets.
4. **Failure categories:** frequency, base success, recovery opportunity, and realised method benefit.
5. **Robustness:** decoding seeds and task-family slices.

## Expected figures

1. Budget–performance curve: success versus intervention/token budget.
2. Selector reliability diagram and precision/recall curve.
3. Cumulative recovered reward versus number of interventions selected.
4. Failure-category transition chart before and after recovery.
5. Three branch-paired case studies: beneficial recovery, harmful recovery, and confident unrecoverable failure.

## Decision checkpoints

- **End of month 2:** terminate or simplify if recovery itself does not outperform a matched extra-generation baseline on any development subset.
- **End of month 3:** continue only if paired labels show nontrivial heterogeneity—some states benefit and some are harmed or neutral—and the selector beats random ranking on development.
- **End of month 4:** freeze method and begin final test evaluation. Do not add a second research method after this point.

