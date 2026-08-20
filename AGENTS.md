# AGENTS.md

This repository is a master's research project.

Before changing code, read:

- PROJECT.md
- PAPER_PROPOSAL.md
- EXPERIMENT_PLAN.md
- IMPLEMENTATION_PHASES.md
- reports/SPRINT_1_5_FINAL.md

Current milestone:
Repository consolidation before two-stage bounded recovery development.

Sprint 1.5 is complete. The frozen baseline is
`indexed_bounded_context_v1` at the revision and configuration recorded in
`reports/SPRINT_1_5_FINAL.md`.

Sprint 2A deterministic replay passed. Sprint 2B demonstrated one positive
causal recovery case, but the joint diagnosis/action operator remained
semantically unreliable after its one allowed revision. Do not tune or modify
the frozen baseline or joint operator further. The next architecture is a
two-stage bounded recovery intervention using the same Qwen3-8B.

Seeds 1005, 1009, 1010, 1022, and 1027 are development-only and must never be
reported as held-out evaluation evidence.

Do not add:
- recovery selector logic
- multi-round recovery or reflection
- DPO
- memory
- PPO/GRPO
- additional environments

Machine-specific paths and credentials must never be committed.

All server-specific configuration belongs under configs/local/,
which must remain gitignored.

Research correctness and reproducibility take priority over feature count.
