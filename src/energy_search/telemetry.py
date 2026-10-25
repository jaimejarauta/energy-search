"""Telemetry extraction for residual stream hidden-state velocity and token entropy.
See notes/01_lyapunov_residual_dynamics.md for theoretical grounding.
"""

import math
from typing import List
import torch

def compute_relative_velocity(h_curr: torch.Tensor, h_prev: torch.Tensor, eps: float = 1e-6) -> float:
    """Computes scale-invariant relative velocity: ||h_t - h_{t-1}||_2 / (||h_t||_2 + eps)."""
    with torch.no_grad():
        diff = torch.norm(h_curr.float() - h_prev.float(), p=2)
        norm_curr = torch.norm(h_curr.float(), p=2)
        return (diff / (norm_curr + eps)).item()

def compute_token_entropy(logits: torch.Tensor, eps: float = 1e-9) -> float:
    """Computes Shannon entropy of next-token distribution."""
    with torch.no_grad():
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + eps))
        return entropy.item()

class RollingVelocityStats:
    """Tracks running mean and standard deviation of relative velocity over a causal sliding window."""
    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        self.history: List[float] = []

    def update_and_normalize(self, v_rel: float, eps: float = 1e-6) -> float:
        """Appends v_rel and returns normalized z-score against historical window."""
        self.history.append(v_rel)
        if len(self.history) < 2:
            return 0.0
        
        window = self.history[-self.window_size:]
        mean_v = sum(window) / len(window)
        variance = sum((x - mean_v) ** 2 for x in window) / max(1, len(window) - 1)
        std_v = math.sqrt(variance) + eps
        
        z = (v_rel - mean_v) / std_v
        return max(0.0, z)
