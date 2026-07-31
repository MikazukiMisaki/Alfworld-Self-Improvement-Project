"""Evaluation loops and aggregate metrics."""

from .evaluator import evaluate
from .metrics import EvaluationMetrics

__all__ = ["EvaluationMetrics", "evaluate"]
