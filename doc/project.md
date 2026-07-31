# PROJECT.md

# ALFWorld Self-Improvement Research Framework

---

# Project Overview

This repository contains the implementation of my master's thesis research.

The goal is NOT simply to improve benchmark performance.

The goal is to study how a relatively small Multimodal / Language Model can continuously improve itself through interaction with an embodied environment.

The project should be implemented as a clean, modular, and research-oriented framework that supports rapid experimentation.

This repository is expected to evolve throughout the entire master's research.

Code quality, modularity, extensibility, and reproducibility are more important than quick implementations.

---

# Research Topic

Self-Improvement of Small Language Models in Interactive Environments

Current Environment

- ALFWorld

Current Base Model

- Qwen3-8B

Current Training Method

- DPO

Potential Future Training

- SFT
- PPO
- GRPO
- ORPO
- Online DPO

The framework should support replacing the training algorithm with minimal code changes.

---

# Research Goal

Current self-improvement methods mostly optimize trajectories as a whole.

This project aims to investigate deeper research questions including:

- Why does self-improvement work?
- Which errors should be corrected?
- Which reflection is actually useful?
- Can small models continuously improve?
- How should failures be attributed to previous actions?
- Can uncertainty guide self-improvement?

The repository should therefore be designed for research rather than production.

---

# Long-Term Research Directions

The framework should support future modules including:

## Reflection

Generate reasoning after each trajectory.

Possible outputs:

- error analysis
- better reasoning
- improved action
- confidence
- uncertainty

---

## Credit Assignment

Future research direction.

Assign responsibility scores to every action.

Possible algorithms:

- Leave-one-out
- Counterfactual replay
- Temporal Difference
- Learned attribution
- Influence estimation

Each algorithm should be implemented as an interchangeable module.

---

## Uncertainty Estimation

Future direction.

Possible uncertainty sources:

- token entropy
- action entropy
- trajectory uncertainty
- disagreement
- Monte Carlo decoding

The framework should support multiple uncertainty estimators.

---

## Memory

Future module.

Store previous failures.

Support:

- retrieval
- summarization
- trajectory memory
- reflection memory

---

## Planning

Future module.

Support planning separately from execution.

Possible planner:

- Tree Search
- ReAct
- DFS/BFS
- LLM Planner

Planner should be replaceable.

---

# Repository Structure

Desired structure:

src/

    env/
        wrappers
        collectors

    models/
        qwen
        inference

    trajectory/
        collector.py
        trajectory.py
        replay_buffer.py

    reflection/
        generators.py
        prompts.py
        parser.py

    credit_assignment/
        base.py
        leave_one_out.py
        counterfactual.py

    uncertainty/
        entropy.py
        ensemble.py

    preference/
        builder.py

    trainer/
        dpo.py
        sft.py
        ppo.py

    evaluation/
        evaluator.py
        metrics.py

    visualization/
        plots.py

    analysis/
        statistics.py

    utils/

configs/

scripts/

experiments/

logs/

results/

papers/

docs/

---

# Software Engineering Principles

Always prefer

- modularity
- readability
- reproducibility

Avoid

- duplicated code
- hard-coded parameters
- hidden global variables

Every major module should expose clean interfaces.

Prefer dependency injection over tightly coupled implementations.

---

# Configuration

Every experiment should be configurable.

Avoid modifying source code to run experiments.

Use configuration files whenever possible.

Possible config categories:

- model
- environment
- reflection
- trainer
- evaluation
- logging

---

# Experiment Philosophy

Every experiment should be reproducible.

Each experiment should automatically save

- config
- git commit hash
- model checkpoint
- evaluation metrics
- trajectory logs
- random seed

Experiments should never overwrite previous results.

---

# Logging

Every trajectory should optionally save

- observation
- action
- reasoning
- reflection
- reward
- done
- confidence
- entropy
- timestamps

Support exporting to JSONL.

---

# Metrics

Current metrics

- Success Rate
- Episode Reward
- Episode Length

Future metrics

- Reflection Quality
- Planning Accuracy
- Error Recovery
- Credit Assignment Accuracy
- Improvement Rate
- Sample Efficiency
- Action Diversity
- Calibration
- Uncertainty Quality

Metrics should be implemented independently from training.

---

# Coding Style

Prefer

- Python type hints
- dataclasses
- pathlib
- logging

Avoid

- print()
- magic numbers
- large monolithic functions

Functions should generally remain below 100 lines.

Complex logic should be decomposed.

---

# Documentation

Every public class should have

- description
- input
- output

Every algorithm should include

- paper reference
- mathematical intuition
- implementation notes

---

# Testing

Critical modules should include tests.

Especially

- trajectory
- parser
- dataset generation
- evaluation

---

# Research Notes

This repository is intended to become a general research platform for self-improving language agents.

Future environments may include

- WebShop
- ScienceWorld
- BabyAI
- MiniGrid
- Minecraft
- OpenHands
- Browser environments

The architecture should therefore remain environment-agnostic whenever possible.

---

# Important Rule for Codex

When modifying the repository:

DO NOT simply make the requested code work.

Instead,

1. Understand the research goal.

2. Preserve modularity.

3. Suggest cleaner abstractions whenever appropriate.

4. Point out potential design problems.

5. If a requested implementation conflicts with long-term extensibility, explain why and propose a better design.

6. Prefer reusable research infrastructure over task-specific implementations.

When unsure, optimize for future research rather than short-term convenience.