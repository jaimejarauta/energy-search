"""Energy-Search: Lyapunov Residual Stream Pruning for LLM Reasoning."""

from .telemetry import compute_relative_velocity, compute_token_entropy, RollingVelocityStats
from .pruner import DynamicalInstabilityPruner
from .sampler import sample_with_energy_pruning, sample_greedy
from .benchmarks import (
    GSM8K_STANDALONE_SUITE,
    extract_numerical_answer,
    evaluate_answer,
    TokenSavingsAccumulator,
    load_gsm8k_dataset,
)

__all__ = [
    "compute_relative_velocity",
    "compute_token_entropy",
    "RollingVelocityStats",
    "DynamicalInstabilityPruner",
    "sample_with_energy_pruning",
    "sample_greedy",
    "GSM8K_STANDALONE_SUITE",
    "extract_numerical_answer",
    "evaluate_answer",
    "TokenSavingsAccumulator",
    "load_gsm8k_dataset",
]
