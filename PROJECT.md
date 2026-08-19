# Alfworld-Self-Improvement

This repository supports a master's thesis on recovery-aware intervention
selection for small language agents.

Sprint 1.5 completed and froze the reproducible ALFWorld baseline:

ALFWorld → Qwen3-8B → trajectory collection → evaluation → JSONL artifacts.

The frozen baseline is `indexed_bounded_context_v1`: indexed admissible-action
selection with bounded goal-preserving recent-state context (`k=4`). Its exact
configuration, revision, schedule hash, and formal 30-episode result are in
`reports/SPRINT_1_5_FINAL.md`.

The current milestone is Sprint 2A, which is limited to validating deterministic
trajectory-prefix replay/reconstruction. Do not tune the baseline unless replay
validation exposes a genuine implementation defect. Recovery, reflection,
selector logic, DPO, memory, PPO, and GRPO are deferred and are not part of
Sprint 2A.
