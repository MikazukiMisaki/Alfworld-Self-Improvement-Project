# AGENTS.md

This repository is a master's research project.

Before changing code, read:

- PROJECT.md
- PAPER_PROPOSAL.md
- EXPERIMENT_PLAN.md
- IMPLEMENTATION_PHASES.md
- reports/SPRINT_1_5_FINAL.md

Current milestone:
Sprint 2A — deterministic prefix replay/reconstruction validation.

Sprint 1.5 is complete. The frozen baseline is
`indexed_bounded_context_v1` at the revision and configuration recorded in
`reports/SPRINT_1_5_FINAL.md`.

Do not tune or modify the frozen baseline unless Sprint 2A replay validation
reveals a genuine implementation defect. Sprint 2A is infrastructure-only:
validate reconstruction before implementing any recovery behavior.

Do not add:
- recovery or reflection behavior before replay passes
- recovery selector logic
- DPO
- memory
- PPO/GRPO
- additional environments

Machine-specific paths and credentials must never be committed.

All server-specific configuration belongs under configs/local/,
which must remain gitignored.

Research correctness and reproducibility take priority over feature count.
