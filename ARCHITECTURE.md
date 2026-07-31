# Future Research Framework Architecture

## Design principles

This document defines interfaces, responsibilities, and data contracts only. It deliberately does not prescribe an implementation or dependency-injection framework.

The framework makes one ALFWorld paper feasible now while keeping environment-specific code at the boundary. A new environment is supported by writing an adapter, not by changing trajectory, training, reflection, or evaluation semantics.

**Research components are replaceable hypotheses.** Environment execution, data schemas, evaluation, provenance, and artifact persistence are infrastructure. Reflection, uncertainty, memory retrieval, credit assignment, preference ranking, and training algorithms are experimental modules.

## System boundary

Environment adapter → Policy → Collector → Trajectory store → Analysis

The collector owns interaction. A policy never writes files or calls an environment directly. A trainer never parses episode logs. The evaluator never decides which model-improvement method to use.

## Environment layer

### Environment interface

An adapter normalizes ALFWorld now and WebShop, ScienceWorld, MiniGrid, and Minecraft later.

| Method | Contract |
|---|---|
| reset(seed, task_selector=None) → ResetResult | Starts an episode and returns initial observation, task, environment metadata, and valid actions if known. |
| step(action) → Transition | Applies one canonical action and returns next observation, reward, terminal/truncated flags, diagnostics, and valid actions if known. |
| get_task() → Task | Returns stable task ID, natural-language goal, split, task family, and environment-specific metadata. |
| get_valid_actions() → ActionSpace | Returns admissible textual actions when available, otherwise explicitly marks the space as open-ended. |

ResetResult and Transition preserve raw environment data in namespaced metadata, but expose stable cross-environment fields. An adapter validates action serialization and records invalid-action outcomes rather than hiding them.

Environment-specific adapters:
- **ALFWorld:** textual observation, textual action, task family, admissible commands.
- **WebShop:** browser/page state and constrained action representation.
- **ScienceWorld:** textual/scientific state and action grammar.
- **MiniGrid:** symbolic/visual observation plus discrete-action mapping.
- **Minecraft:** multimodal state, tool/action schema, and potentially asynchronous transitions.

## Model layer

### Policy interface

A policy is model-agnostic and supports Qwen3-8B now and other LLMs later.

| Method | Contract |
|---|---|
| act(request) → ActionDecision | Produces one canonical action, optional reasoning, and generation trace. |
| generate(request, options) → GenerationResult | Lower-level generation used by planners, reflectors, and recovery policies. |
| load_adapter(adapter_ref) → Policy | Returns/configures a policy with a checkpoint or adapter without altering base provenance. |
| capabilities() → ModelCapabilities | Declares token-probability, hidden-state, multimodal, batching, and constrained-decoding support. |

ActionDecision contains parsed action, raw text, reasoning when requested, parser status, chosen-token log probabilities when available, and model/decode metadata. Token probabilities and hidden states are optional capabilities. Action parsing and action validation are separate: parsing creates a candidate; the environment determines admissibility.

## Trajectory layer

### Canonical structures

| Structure | Required fields | Purpose |
|---|---|---|
| Step | step index, observation, action, reasoning, reward, done, timestamp | One policy/environment transition. |
| Trajectory | trajectory ID, task ID, initial observation, ordered steps, terminal outcome, model version, seed | Immutable record of one episode. |
| RunManifest | run ID, Git revision, config, environment/model versions, seed policy, hardware, timestamps, artifact references | Reproducibility boundary for a collection/training/evaluation run. |

Each Step additionally supports action validity and valid-action snapshot/reference; raw model output and parser status; token probabilities/entropy; intervention events; reflection and memory IDs; and environment diagnostics.

Trajectory records task text/family, split, truncation reason, cumulative reward, parent/retry trajectory ID, and immutable source provenance. RunManifest stores resolved configuration values rather than only config file names.

JSONL is suitable for append-only trajectory records; an analytical columnar export may be added later. Text logs are a view, never the source of truth.

## Reflection layer

### Reflection interface

| Method | Contract |
|---|---|
| reflect(trajectory, context) → Reflection | Produces structured post-episode analysis. |
| propose_recovery(prefix, context) → RecoveryProposal | Produces one bounded next-step correction when explicitly invoked. |

A Reflection includes error analysis and predicted failure category; evidence step IDs; an improved solution or correction constraint; confidence; model/version metadata; and schema/grounding validation status.

Reflections must distinguish observed facts from model hypotheses. The evaluator measures usefulness by downstream recovery or ranking quality, not eloquence.

## Memory layer

### MemoryRecord and MemoryStore

A memory record stores a failure or success lesson without losing provenance:

- memory ID, task/task-family keys, textual lesson, optional embedding;
- reflection ID and source trajectory/step IDs;
- evidence summary, quality/confidence, generator/model version;
- creation time, retrieval count, expiry/supersession relationship.

| Method | Contract |
|---|---|
| store(record) → memory_id | Persists an append-only record. |
| retrieve(query, budget, filters) → RetrievalResult | Returns ranked records and retrieval scores. |
| supersede(old_id, new_id, reason) | Preserves conflicting or revised lessons rather than overwriting them. |

Retrieval filters support environment, task split, task family, model version, and source provenance. Evaluation includes no-memory, random-memory, and retrieved-memory controls.

## Preference layer

### PreferenceExample

A preference example represents a justified ranking, not just three strings:

- example ID and prompt/context;
- chosen and rejected trajectory fragments or action continuations;
- chosen/rejected source trajectory and step IDs;
- ranking reason, evidence, and ranking method;
- task/split metadata, generator/judge provenance, and deduplication key.

| Method | Contract |
|---|---|
| build(candidates, policy) → PreferenceExample[] | Creates validated examples from comparable candidates. |
| validate(example) → ValidationResult | Checks task alignment, source presence, distinction, and leakage constraints. |

This supports DPO today and future preference optimization without changing data meaning. Whole-trajectory and step-level examples are both allowed, but granularity is explicit.

## Training layer

### Trainer abstraction

| Method | Contract |
|---|---|
| train(dataset, base_model, config) → TrainingResult | Runs one method and returns checkpoint/artifact references. |
| evaluate_loss(dataset) → TrainingDiagnostics | Optional training-side diagnostics, not environment evaluation. |
| resume(run_ref) → TrainingResult | Continues only with an explicit prior manifest. |

TrainingConfig names the algorithm and method-specific settings. Implementations include SFTTrainer, DPOTrainer, PPOTrainer, and GRPOTrainer. All consume canonical data or explicit adapters, emit a checkpoint reference, and preserve base/reference-model provenance.

PPO and GRPO are deferred infrastructure: interfaces may exist before implementation, but they must not expand the first-paper scope.

## Evaluation layer

### Evaluator interface

| Method | Contract |
|---|---|
| evaluate(policy, task_set, protocol) → EvaluationReport | Runs fixed tasks/seeds and produces episode-level records plus aggregates. |
| compare(reports, comparison_plan) → ComparisonReport | Computes matched comparisons and uncertainty intervals. |

### Metrics

Current core metrics:
- success rate;
- episode reward;
- episode length;
- invalid-action and deadlock rate;
- token, latency, and intervention cost.

Future research metrics:
- **recovery ability:** success after a detected error, conditional on comparable failure opportunities;
- **reflection usefulness:** improvement over no-reflection and random-reflection controls at matched budget;
- **uncertainty calibration:** AUROC/AUPRC, Brier score, expected calibration error, and reliability curves for an operational target;
- **generalization:** seen-to-unseen gap, task-family transfer, and robustness to task/horizon variation;
- **data efficiency:** performance per collected trajectory or preference example.

Every report states split, fixed task identifiers, seeds, policy/checkpoint, decoding configuration, and whether tools/memory were available.

## Configuration and experiment boundary

Configurations are layered by environment, policy, collection, uncertainty/intervention, reflection/memory, preference, training, and evaluation. A resolved configuration is recorded in RunManifest.

A single experiment orchestrator may compose modules, but algorithm-specific logic remains outside shared infrastructure. This distinction prevents a thesis iteration from becoming a framework rewrite.

## Non-goals for the first paper

- A universal abstraction that erases meaningful environment differences.
- Full support for all four future environments.
- Mandatory hidden-state access.
- Online RL infrastructure before the selective-intervention study is complete.
- A vector-database requirement; task-keyed retrieval is a valid initial baseline.

