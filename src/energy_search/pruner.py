"""Dynamical Instability Index computation and persistence pruning logic.
Protects valid 'aha moment' spikes while terminating chaotic runaway divergence.
"""

from typing import Tuple
from .telemetry import RollingVelocityStats

class DynamicalInstabilityPruner:
    """Evaluates multi-step persistence criteria on residual stream instability index."""
    def __init__(self, tau: float = 2.5, k_persist: int = 4, window_size: int = 16):
        self.tau = tau
        self.k_persist = k_persist
        self.stats = RollingVelocityStats(window_size=window_size)
        self.consecutive_spikes = 0
        self.prompt_entropy = 1.0

    def set_prompt_entropy(self, entropy: float):
        """Calibrates baseline prompt entropy."""
        self.prompt_entropy = max(0.1, entropy)

    def evaluate_step(self, v_rel: float, token_entropy: float) -> Tuple[bool, float]:
        """Evaluates step instability index and updates persistence counter.
        Returns: (should_prune, current_instability_index).
        """
        z_score = self.stats.update_and_normalize(v_rel)
        # Scaled by relative entropy
        entropy_ratio = token_entropy / self.prompt_entropy
        instability_index = z_score * entropy_ratio

        # Persistence rule: only increment counter if sustained above tau
        if instability_index > self.tau:
            self.consecutive_spikes += 1
        else:
            self.consecutive_spikes = 0

        # Terminate branch only after K consecutive high-instability tokens
        should_prune = self.consecutive_spikes >= self.k_persist
        return should_prune, instability_index
