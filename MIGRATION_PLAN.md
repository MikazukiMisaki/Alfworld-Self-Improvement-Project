# Migration Plan

## Purpose and scope

This is a no-code migration plan. The goal is to make the new repository capable of testing the research roadmap, not to copy legacy scripts or recreate every historical experiment.

**Research contribution:** uncertainty-guided selective self-improvement and its controlled evaluation.

**Engineering infrastructure:** canonical records, ALFWorld adapter, baseline policy boundary, artifact lineage, and evaluation. Build only the infrastructure needed to make the contribution credible.

## Principles

- Preserve legacy artifacts before transforming them.
- Implement schemas and validation before importing or generating data.
- Reproduce one baseline before introducing reflection, memory, DPO, or uncertainty.
- Keep environment/model/training dependencies optional and isolated.
- Every migration output has provenance and a validation gate.
- Do not copy legacy scripts; extract only validated behavior into new modules.

## Step 1 — Legacy data preservation

### Files to create

- legacy/README.md — provenance, known limitations, and preservation policy.
- legacy/manifest.json — hashes, original relative paths, sizes, format, and import status.
- docs/legacy_data_audit.md — counts, schema variants, duplicates, and exclusions.
- configs/legacy_import.yaml — paths and importer policy, not experiment settings.

### Dependencies

Read-only filesystem access, JSON/YAML support, and checksum tooling. No ALFWorld, model, or GPU dependency is required.

### Expected output

A frozen archive/reference to raw batch logs, success-case copies, the legacy memory bank, the legacy DPO dataset, the base ALFWorld configuration, and historical runner revisions required for provenance.

The archive records that the legacy worktree is dirty and that older runner scripts are absent from its working tree but available in Git history.

### Validation criteria

- Every preserved artifact has a hash and source path.
- Success-case copies are linked to originals rather than treated as independent evidence.
- No importer or training job mutates preserved data.
- Audit records the known DPO issues: 192 entries, 97 exact duplicates, and 28 malformed observation prompts.

## Step 2 — Canonical schema implementation

### Files to create

- src/alfworld_research/trajectory/schema.py — Step, Trajectory, RunManifest, and validation contracts.
- src/alfworld_research/trajectory/store.py — append-only artifact/store interface.
- src/alfworld_research/data/provenance.py — artifact/source identity contracts.
- src/alfworld_research/evaluation/protocol.py — fixed task-set and seed protocol contracts.
- tests/test_trajectory_schema.py — schema/round-trip/validation tests.
- docs/data_schema.md — field semantics and versioning policy.

### Dependencies

Python standard library initially. Serialization libraries may be added only if they improve the chosen canonical format. No model or environment dependency is required.

### Expected output

A versioned canonical record definition with mandatory task ID, observation, action, reward, terminal state, model version, seed, timestamp, source IDs, and resolved run configuration.

### Validation criteria

- Valid records round-trip without losing required fields.
- Invalid/missing/ambiguous terminal state, provenance, or task metadata fails validation.
- A RunManifest fully identifies an evaluation or collection run.
- Schema version is stored on every artifact.

## Step 3 — Legacy trajectory importer

### Files to create

- src/alfworld_research/trajectory/legacy_importer.py — importer interface and log-format parsers.
- src/alfworld_research/trajectory/legacy_validation.py — quarantine/report rules.
- scripts/import_legacy_data.py — thin invocation entry point.
- tests/fixtures/legacy_logs/ — minimal representative and malformed text fixtures.
- tests/test_legacy_importer.py.
- reports/legacy_import_report.json — generated import counts, exclusions, and warnings.

### Dependencies

Canonical schema from Step 2. Standard parsing/regex tooling. No ALFWorld, Qwen, or training stack dependency.

### Expected output

Immutable canonical trajectories imported from legacy logs, plus an exclusion/quarantine set for records that cannot be interpreted without guessing. Legacy memory and DPO data are imported as explicitly labelled legacy artifacts, not training-ready datasets.

### Validation criteria

- Import count reconciles against the manifest.
- A sampled imported trajectory exactly matches source actions, rewards, and terminal state.
- Duplicate success copies resolve to the same source trajectory.
- The importer never silently invents an observation, action, or terminal result.
- All warnings are included in the report, not printed only to stdout.

## Step 4 — Baseline reproduction

### Files to create

- src/alfworld_research/env/alfworld.py — ALFWorld adapter implementing the architecture contract.
- src/alfworld_research/models/qwen.py — Qwen policy adapter and generation-metadata boundary.
- src/alfworld_research/models/action_parser.py — parser contract and tested implementation.
- src/alfworld_research/trajectory/collector.py — shared bounded episode collector.
- src/alfworld_research/evaluation/evaluator.py and metrics.py.
- configs/environment/alfworld_text.yaml.
- configs/model/qwen3_8b.yaml.
- configs/evaluation/baseline.yaml.
- scripts/run_baseline.py.
- tests/test_alfworld_adapter.py and tests/test_action_parser.py.

### Dependencies

ALFWorld and its data; Transformers/Qwen runtime; CUDA only when selected by configuration; canonical schema and evaluation protocol. These dependencies remain outside the pure schema/import path.

### Expected output

One reproducible direct-policy baseline evaluated on fixed seen and unseen ALFWorld validation task lists. The output includes trajectories, metrics, resolved config, environment/model versions, seeds, prompt/decode settings, and checkpoint reference.

### Validation criteria

- The adapter records valid actions and terminal states faithfully.
- The same seed/config produces an auditable equivalent run within documented nondeterminism limits.
- Baseline metrics are reported on validation splits, not only train.
- The policy does not load a DPO adapter, retrieve memory, reflect, or gate intervention in this phase.
- A baseline comparison to the legacy behavior states protocol differences instead of treating scores as directly equivalent.

## Step 5 — Research modules

Implement these one at a time after the Step 4 baseline is frozen. Each module has a no-op baseline, a matched-budget control, and a dedicated evaluation configuration.

### 5A. Failure analysis and taxonomy

**Files:** analysis/failure_taxonomy.py, analysis/trajectory_features.py, docs/annotation_guidelines.md, and analysis tests/fixtures.

**Dependencies:** canonical trajectories and baseline runs.

**Expected output:** labelled error sample, taxonomy counts, and candidate targets for intervention.

**Validation:** taxonomy is mutually exclusive where claimed; annotation protocol and ambiguous cases are documented.

### 5B. Reflection and memory

**Files:** reflection/base.py, reflection/structured.py, memory/base.py, memory/task_keyed.py, configs/reflection/, and configs/memory/.

**Dependencies:** baseline trajectories, policy generation interface, and canonical provenance.

**Expected output:** structured reflections and append-only memory records with source step/trajectory IDs.

**Validation:** schema/grounding checks pass; no-memory and random-memory controls exist; retrieval decisions are logged.

### 5C. Uncertainty and selective intervention — primary research module

**Files:** uncertainty/base.py, uncertainty/token_entropy.py, intervention/gate.py, intervention/selective_recover.py, analysis/calibration.py, and configs/intervention/.

**Dependencies:** policy token-score capability; collector; reflection/memory module only after their standalone baseline is measured.

**Expected output:** calibrated risk scores, gate decisions, intervention events, and matched-budget comparison reports.

**Validation:** prediction target is defined before fitting; calibration is measured on held-out development data; random, unconditional, raw-threshold, and oracle-gate controls are available.

### 5D. Preference construction and DPO

**Files:** preference/schema.py, preference/builder.py, preference/validators.py, trainer/base.py, trainer/dpo.py, configs/training/dpo.yaml, and dataset/trainer tests.

**Dependencies:** canonical trajectories, ranking policy, model/training runtime, and fixed train/evaluation split rules.

**Expected output:** deduplicated, provenance-backed preference dataset and DPO checkpoint evaluated by the same evaluator as the base policy.

**Validation:** every example has source IDs and a ranking reason; chosen/rejected contexts are comparable; no validation-task leakage; uniform and selective data construction use equal data budgets.

## Deferred work

Do not implement PPO, GRPO, hidden-state probes, vector retrieval, full counterfactual replay, or adapters for WebShop/ScienceWorld/MiniGrid/Minecraft until the Step 4 baseline and primary gate ablations are complete. Their interfaces may be documented, but they are not blockers for the paper.

## Completion definition

Migration is ready for research when:

1. Legacy evidence is preserved and auditable.
2. Canonical trajectories and run manifests are validated.
3. A fixed, reproducible Qwen3-8B ALFWorld baseline exists.
4. The failure taxonomy identifies an intervention target.
5. Selective intervention can be evaluated against matched controls without changing shared infrastructure.

