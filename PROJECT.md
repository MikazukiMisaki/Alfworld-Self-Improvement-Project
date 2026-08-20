# Alfworld-Self-Improvement

This repository supports a master's thesis on recovery-aware intervention
selection for small language agents.

Sprint 1.5 completed and froze the reproducible ALFWorld baseline:

ALFWorld → Qwen3-8B → trajectory collection → evaluation → JSONL artifacts.

The frozen baseline is `indexed_bounded_context_v1`: indexed admissible-action
selection with bounded goal-preserving recent-state context (`k=4`). Its exact
configuration, revision, schedule hash, and formal 30-episode result are in
`reports/SPRINT_1_5_FINAL.md`.

Sprint 2A passed deterministic trajectory-prefix replay validation. Sprint 2B
joint recovery demonstrated one non-zero positive causal intervention, but
diagnosis-to-action agreement remained 1/5 after the single allowed operator
revision. Formatting is reliable; semantic action selection is not.

The next architecture is two-stage bounded recovery with the same Qwen3-8B:
one diagnosis/subgoal call followed by one Action-ID call. No selector is
implemented. Seeds 1005, 1009, 1010, 1022, and 1027 are development-only and
are ineligible as held-out evaluation evidence.
