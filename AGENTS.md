# AGENTS.md

This repository is a master's research project.

Before changing code, read:

- PROJECT.md
- docs/PAPER_PROPOSAL.md
- docs/EXPERIMENT_PLAN.md
- docs/IMPLEMENTATION_PHASES.md

Current milestone:
Sprint 1.5 — baseline debugging and validation.

Do not implement Sprint 2 until the baseline exit criteria are satisfied.

Do not add:
- DPO
- memory
- PPO/GRPO
- additional environments
- selector logic

Machine-specific paths and credentials must never be committed.

All server-specific configuration belongs under configs/local/,
which must remain gitignored.

Research correctness and reproducibility take priority over feature count.