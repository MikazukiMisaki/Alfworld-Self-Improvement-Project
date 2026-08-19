# AGENTS.md

This repository is a master's research project.

Before changing code, read:

- PROJECT.md
- PAPER_PROPOSAL.md
- EXPERIMENT_PLAN.md
- IMPLEMENTATION_PHASES.md
- reports/SPRINT_1_5_FINAL.md

Current milestone:
Sprint 2B — fixed one-shot recovery pilot.

Sprint 1.5 is complete. The frozen baseline is
`indexed_bounded_context_v1` at the revision and configuration recorded in
`reports/SPRINT_1_5_FINAL.md`.

Sprint 2A deterministic replay passed. Do not tune or modify the frozen
baseline. Sprint 2B permits exactly one fixed recovery call at each frozen
pilot prefix, followed by the unchanged H4 policy.

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
