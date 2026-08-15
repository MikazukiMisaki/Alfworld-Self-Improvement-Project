# Sprint 1.5 Experiment Readiness Audit

Audit date: 2026-08-15

## Executive conclusion

The repository is safer for baseline regression work, but it is not ready for
DPO or Sprint 2. Future runs now carry explicit pipeline provenance, runtime
outputs are isolated from historical results, and trajectory consumers can
fail closed on pipeline/mode/split mismatches. The indexed implementation has
no observed ID mapping, ordering, parsing, stale-list, or execution bug in the
available tests and 200 recorded B1 decisions.

A behavioral regression relative to the historical legacy system cannot yet
be established or rejected. The legacy runner and its reported 210 episodes,
including 28 successes, are absent from the checkout and audited server
workspace. The 15 available run directories contain no successful trajectory.

## A. Current pipeline architecture

### Historical legacy_v1 (documented, implementation unavailable)

```text
ALFWorld train state
  -> Qwen3-VL-8B-Instruct processor prompt (goal rules, example, Thought + Action)
  -> free-text model output (up to 128 tokens)
  -> first Action: line, otherwise final-line fallback
  -> unvalidated text sent to ALFWorld
  -> historical text/log trajectory (15-step runner horizon)
```

This flow is reconstructed from `docs/SPRINT_1_5_REGRESSION_REPORT.md`; its
source and artifacts are not available for direct verification.

### Current free_form_v1 / B0

```text
AlfWorldTextEnvironment.reset
  -> observation + current environment-owned admissible tuple + task
  -> QwenPolicy._prompt (task, full history, observation, commands)
  -> Qwen3-8B greedy output, thinking disabled, 32-token cap
  -> parse_action (one explicit or bare command)
  -> normalized admissible membership check
  -> exact environment command or fail-closed parser termination
  -> Step -> Trajectory -> JSONL + metrics + RunManifest
```

### Current indexed_v1 / B1

```text
AlfWorldTextEnvironment.reset
  -> observation + current environment-owned admissible tuple + task
  -> action_id_mapping preserves tuple order as A000, A001, ...
  -> QwenPolicy._indexed_prompt shows every ID-command pair
  -> Qwen3-8B greedy output, thinking disabled, 32-token cap
  -> parse_action_id accepts exactly one Action-ID: Axxx line
  -> selected ID maps against the same tuple snapshot shown in the prompt
  -> exact environment-owned command or fail-closed selection termination
  -> Step with complete mapping -> Trajectory -> JSONL + metrics + RunManifest
```

The environment adapter updates the admissible tuple after each transition.
The next prompt and mapping use that updated tuple; no independent cache or
sorting layer exists.

## B. Historical data provenance

`reports/result_provenance.json` is the machine-readable inventory and
`reports/result_provenance.md` is its compact human view.

| Classification | Runs | Successful trajectories | Confidence |
|---|---:|---:|---|
| `free_form_v1` | 13 | 0 | Medium; inferred from canonical manifests/configs/outputs |
| `indexed_v1` | 2 | 0 | High; resolved mode plus ID protocol and step mappings |
| `legacy_v1` | 0 | 0 | No source artifacts available |

Artifact roles are 9 smoke tests, 1 indexed diagnostic, and 5 other historical
runs. Five smoke directories contain only a manifest. Historical manifests do
not record a complete package, dataset, and hardware fingerprint, so their
runtime environment remains only partially attributable.

The classifier does not use directory names as primary pipeline evidence. It
records SHA-256 hashes and marks missing files and implicit historical fields.

## C. Data isolation status

- Runtime artifact files under `results/` were intentionally removed from Git
  tracking. They were not moved or rewritten and remain available locally on
  the audited server, where `results/` is ignored; only its README is tracked.
- Provenance for the locally retained runtime artifacts is preserved in
  `reports/result_provenance.json` and `reports/result_provenance.md`, including
  the audit inventory and recorded SHA-256 hashes.
- Portable collection configs now write to ignored `artifacts/runtime/`.
- `artifacts/README.md` distinguishes real runtime evidence from fixtures.
- `tests/fixtures/README.md` reserves fixtures for small synthetic data only.
- `legacy/` remains an immutable import boundary and records the missing source
  corpus explicitly.
- No machine-local path was added to portable configuration or reports.

Historical manifests already contain old absolute server paths. They were not
rewritten because doing so would alter preserved evidence.

## D. Data-consumption and contamination audit

The current working tree contains no DPO trainer, preference builder,
reflection-data loader, replay buffer, success-case loader, or generic result
glob. Current readers are:

- `scripts/validate_baseline_artifacts.py`, which reads one explicitly named
  run directory;
- the new provenance audit, which classifies but does not train on artifacts;
- the new regression harness, which requires explicit provenance and exact
  task/seed matching.

Removed Git-history modules included an in-memory preference pair builder and
an append-only trajectory store. They are not importable in the current tree.
The planning documents report that the unavailable legacy DPO set had 192
entries, 97 exact duplicates, malformed prompts, and no source lineage. It is
not training-ready evidence.

`trajectory.provenance.load_run_trajectories` now requires an explicit allowed
pipeline version, action mode, and split. Historical manifests without those
fields fail closed. This prevents a future consumer from silently combining
legacy, B0, and B1 trajectories through a broad directory scan.

## E. Legacy versus indexed regression suite

`configs/regression/legacy_success_cases.yaml` contains zero cases. This is a
deliberate evidence result: no successful legacy source trajectory is
available, so no case was fabricated from aggregate planning-document counts.

`scripts/run_action_interface_regression.py` supports a bounded current B0/B1
comparison. It verifies that environment, checkpoint, generation settings,
horizon, and seed schedule match; runs each interface; then refuses comparison
unless ordered task IDs and seeds match exactly. It writes JSON and Markdown
step tables with earliest divergence, actions, target observation/pickup,
transformation, destination, placement, repeat, timeout, reward, and success
signals.

This harness is not labelled a historical legacy reproduction. Current B0
differs materially from legacy_v1 in model modality, prompt, parser, split,
horizon, and fail-closed behavior.

## F. Confirmed issues and fixes

1. Future manifests did not explicitly identify the pipeline at top level.
   New manifests now record and cross-validate `pipeline_version`,
   `action_selection_mode`, and `split` against resolved configuration.
2. Portable configs wrote new runtime output into the source-controlled
   historical `results/` tree. They now target ignored `artifacts/runtime/`.
3. No trajectory-loading contract prevented mixed-pipeline consumption. The
   strict provenance loader and discovery filter now fail closed.
4. No matched comparison guard checked that separately initialized ALFWorld
   runs selected the same task. The harness now requires exact task/seed pairs.
5. Historical results lacked a machine-readable content-based inventory. The
   generated report classifies all 15 runs and records evidence and hashes.

No confirmed indexed implementation defect was found:

- IDs are zero-based and deterministic.
- Environment order is preserved; actions are not sorted or truncated.
- Duplicate commands retain distinct IDs and recover the exact command.
- The selected ID maps against the exact list stored with the prompt step.
- Malformed, ambiguous, and out-of-range IDs fail closed.
- The collector refreshes the action list after every environment step.
- Current B0 and B1 portable model configs differ only in declared pipeline and
  action-selection interface.

## G. Unresolved hypotheses

These are not confirmed bugs:

- **Interface-induced policy degradation.** On the same pan task in unmatched
  historical B0/B1 runs, the first action differed immediately. The runs have
  different revisions and are not a causal comparison.
- **Option-position bias.** B1 repeatedly selected IDs associated with locally
  available take/move or navigation actions. Dynamic list contents confound a
  simple index-frequency interpretation.
- **Prompt/token pressure.** Recorded B1 lists contain 30--59 actions and stored
  prompts grow from about 2,000 to 8,200 characters as full history grows.
  Output generation remained eight tokens, but input-length effects have not
  been experimentally isolated.
- **Semantic planning limitation.** B1 selected a pot for a pan goal, an egg
  for a mug goal, and looped between stoveburners. These actions were correctly
  mapped and admissible, so they may reflect model planning/state-tracking
  limitations rather than interface code.
- **Legacy prompt advantage.** The unavailable legacy system used a different
  model modality, train split, worked example, Thought + Action output, token
  budget, parser, and horizon. Any historical success gap could come from any
  of those confounds.

## H. Recommended next experiment

First recover the immutable legacy runner revision and source trajectories for
the reported 28 successes. Hash them, record original paths, and populate the
regression manifest only with cases whose task ID, split, seed, terminal reward,
and source lineage are verifiable.

In parallel, the smallest useful server test is a three-episode current B0/B1
run through the regression harness on `valid_seen`. It should be treated only
as an action-interface diagnostic, not a legacy comparison. Inspect earliest
divergence, target-object choices, selected-index distribution, input token
length, and loops. Do not start DPO, recovery, reflection, memory, or Sprint 2
until a matched baseline comparison has at least some reproducible task-solving
signal or a documented negative baseline decision.
