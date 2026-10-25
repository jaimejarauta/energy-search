"""Unit tests for velocity and entropy telemetry."""

import torch
from energy_search.telemetry import compute_relative_velocity, compute_token_entropy, RollingVelocityStats

def test_relative_velocity_scale_invariance():
    torch.manual_seed(42)
    h1 = torch.randn(64)
    h2 = h1 + 0.1 * torch.randn(64)
    
    v_base = compute_relative_velocity(h2, h1)
    # Scaling both vectors by 10x should yield approximately identical relative velocity
    v_scaled = compute_relative_velocity(h2 * 10.0, h1 * 10.0)
    
    assert abs(v_base - v_scaled) < 1e-4, f"Relative velocity not scale-invariant: {v_base} vs {v_scaled}"

def test_token_entropy_deterministic():
    # One-hot logits -> 0 entropy
    logits_det = torch.tensor([100.0, -100.0, -100.0])
    ent_det = compute_token_entropy(logits_det)
    assert ent_det < 1e-4, f"Expected 0 entropy, got {ent_det}"
    
    # Uniform logits -> ln(3) entropy
    logits_uni = torch.tensor([0.0, 0.0, 0.0])
    ent_uni = compute_token_entropy(logits_uni)
    assert abs(ent_uni - 1.0986) < 1e-3, f"Expected ~1.0986 entropy, got {ent_uni}"
