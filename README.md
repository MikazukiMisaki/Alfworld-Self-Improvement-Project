# Alfworld-Self-Improvement

Minimal reproducible baseline infrastructure for the thesis project
“Recovery-Aware Intervention Selection for Small Language Agents.”

## Project status

Sprint 1.5 is complete. The frozen baseline is
`indexed_bounded_context_v1`, using indexed admissible actions and bounded
goal-preserving recent-state context with `k=4`.

Sprint 2A deterministic prefix replay passed. Sprint 2B joint recovery showed
one positive causal recovery case, but diagnosis-to-action agreement remained
insufficient after one revision. The next architecture is two-stage bounded
recovery using the same Qwen3-8B; no selector has been implemented.

Seeds 1005, 1009, 1010, 1022, and 1027 are development-only and are not
eligible as held-out evaluation evidence. Baseline tuning, memory, DPO, PPO,
and GRPO remain out of scope.

See `reports/SPRINT_1_5_FINAL.md` for the frozen configuration and formal
30-episode result.

## Repository layout

    docs/        sprint, paper, TODO, and changelog documents
    src/env/     environment contracts and ALFWorld adapter
    src/models/  Qwen wrapper and action parser
    src/trajectory/ canonical trajectory records, collector, and JSONL store
    src/evaluation/ metrics and evaluation runner
    src/utils/   small shared utilities
    tests/       unit tests without external model/environment dependencies
    configs/     baseline configuration
    scripts/     runnable collection entry points
    legacy/      reserved immutable legacy artifacts
    results/     immutable historical result archive
    artifacts/   ignored runtime experiment output
    reports/     provenance and readiness audits

## Run a baseline

Install the ALFWorld, PyTorch, Transformers, and PyYAML runtimes. Then set:

    export ALFWORLD_CONFIG_PATH=/absolute/path/to/base_config.yaml
    export ALFWORLD_DATA=/absolute/path/to/alfworld_data
    python scripts/collect_baseline.py

The script creates a unique ignored `artifacts/runtime/baseline` directory containing:

- trajectory.jsonl
- metrics.json
- run_manifest.json

Every new manifest records explicit `pipeline_version`,
`action_selection_mode`, and `split` fields. Historical `results/` are not
implicit experiment or training inputs.

## Action-interface regression

Check that the portable B0/B1 configs differ only by action interface:

    python scripts/run_action_interface_regression.py check-configs

After activating the university runtime, a bounded matched comparison can be
run explicitly with:

    python scripts/run_action_interface_regression.py run --episodes 1 --run-name manual-check

The harness rejects task/seed mismatches and writes structured and Markdown
step comparisons under ignored `artifacts/runtime/regression/`.

The frozen H4 portable configuration is
`configs/collection/baseline_indexed_h4.yaml`. Its runtime outputs belong under
ignored `artifacts/runtime/`; full trajectories are not committed.

## Tests

    PYTHONPATH=src python -m unittest discover -s tests -v
