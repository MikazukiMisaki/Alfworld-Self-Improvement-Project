# ALFWorld Self-Improvement Research Framework

A small, modular Python foundation for experiments on self-improving language
models in interactive environments. The core package deliberately has no runtime
dependencies: integrate ALFWorld, model serving, and training libraries behind
the provided interfaces.

## Quick start

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest
python3 -m alfworld_research.experiments.run_demo
```

## Layout

- `src/alfworld_research/env`: environment protocols and adapters.
- `src/alfworld_research/trajectory`: typed trajectories and JSONL persistence.
- `src/alfworld_research/reflection`: interchangeable reflection generators.
- `src/alfworld_research/preference`: DPO-ready preference construction.
- `src/alfworld_research/trainer`: trainer interfaces.
- `src/alfworld_research/evaluation`: reusable evaluation loop and metrics.
- `configs`: versioned experiment configuration examples.

The demo uses a deterministic toy environment. Replace it with an ALFWorld
adapter and a model policy without changing the collector or evaluator APIs.
