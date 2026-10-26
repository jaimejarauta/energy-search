"""Dynamical Instability Index computation and persistence pruning logic.
Protects valid 'aha moment' spikes while terminating chaotic runaway divergence.
"""

from typing import Tuple
from .telemetry import RollingVelocityStats

class DynamicalInstabilityPruner:
    """Evaluates multi-step persistence criteria on residual stream instability index.
    Detects both hyper-dynamic chaotic divergence and hypo-dynamic cognitive freeze (loops).
    """
    def __init__(self, tau: float = 2.5, k_persist: int = 4, window_size: int = 16,
                 freeze_vel_tau: float = 0.04, freeze_ent_tau: float = 0.15, k_freeze: int = 6,
                 colinear_tau: float = 0.85):
        self.tau = tau
        self.k_persist = k_persist
        self.stats = RollingVelocityStats(window_size=window_size)
        self.consecutive_spikes = 0
        self.consecutive_freezes = 0
        self.prompt_entropy = 1.0
        self.freeze_vel_tau = freeze_vel_tau
        self.freeze_ent_tau = freeze_ent_tau
        self.k_freeze = k_freeze
        self.colinear_tau = colinear_tau

    def set_prompt_entropy(self, entropy: float):
        """Calibrates baseline prompt entropy."""
        self.prompt_entropy = max(0.1, entropy)

    def evaluate_step(self, v_rel: float, token_entropy: float, cos_sim: float = 0.0) -> Tuple[bool, float, str]:
        """Evaluates step instability index and updates persistence counters.
        Returns: (should_prune, current_instability_index, prune_reason).
        """
        z_score = self.stats.update_and_normalize(v_rel)
        # Scaled by relative entropy
        entropy_ratio = token_entropy / self.prompt_entropy
        instability_index = z_score * entropy_ratio

        # 1. Hyper-dynamic instability check (chaotic runaway)
        if instability_index > self.tau:
            self.consecutive_spikes += 1
        else:
            self.consecutive_spikes = 0

        # 2. Hypo-dynamic cognitive freeze check (repetitive loop collapse or co-linear trajectory lock-in)
        is_freeze = (v_rel < self.freeze_vel_tau and token_entropy < self.freeze_ent_tau) or (cos_sim > self.colinear_tau and token_entropy < 1.2)
        if is_freeze:
            self.consecutive_freezes += 1
        else:
            self.consecutive_freezes = 0

        # Prune decision
        if self.consecutive_spikes >= self.k_persist:
            return True, instability_index, "chaotic_divergence"
        if self.consecutive_freezes >= self.k_freeze:
            return True, instability_index, "cognitive_freeze_loop"

        return False, instability_index, "active"

