"""Unit tests for Dynamical Instability Pruner and persistence logic."""

from energy_search.pruner import DynamicalInstabilityPruner

def test_persistence_logic_protects_single_spike():
    # A single high-velocity spike (e.g. an 'aha' transition) must NOT trigger pruning
    pruner = DynamicalInstabilityPruner(tau=2.0, k_persist=4, window_size=8)
    pruner.set_prompt_entropy(1.0)
    
    # Baseline steady tokens
    for _ in range(5):
        should_prune, _, _ = pruner.evaluate_step(v_rel=0.05, token_entropy=0.5)
        assert not should_prune, "Unexpected prune during baseline"
        
    # Single high-velocity spike (aha moment)
    should_prune, inst_val, _ = pruner.evaluate_step(v_rel=0.90, token_entropy=0.1)
    assert not should_prune, "Pruner incorrectly killed a single-step spike!"
    assert pruner.consecutive_spikes <= 1, "Counter should be at most 1"

def test_persistence_logic_triggers_on_sustained_chaos():
    pruner = DynamicalInstabilityPruner(tau=1.5, k_persist=4, window_size=8)
    pruner.set_prompt_entropy(1.0)
    
    # Baseline tokens to establish variance
    for _ in range(8):
        pruner.evaluate_step(v_rel=0.05, token_entropy=0.5)
        
    # Sustained chaotic tokens with high entropy
    pruned = False
    for step in range(6):
        should_prune, _, _ = pruner.evaluate_step(v_rel=0.80, token_entropy=2.0)
        if should_prune:
            pruned = True
            break
            
    assert pruned, "Pruner failed to terminate sustained chaotic divergence!"

def test_cognitive_freeze_loop_detection():
    # Repetitive looping ("the the the...") has zero velocity and zero entropy
    pruner = DynamicalInstabilityPruner(tau=2.0, k_persist=4, k_freeze=5, freeze_vel_tau=0.04, freeze_ent_tau=0.10)
    pruner.set_prompt_entropy(1.0)
    
    pruned = False
    prune_reason = ""
    for _ in range(6):
        should_prune, _, reason = pruner.evaluate_step(v_rel=0.01, token_entropy=0.02)
        if should_prune:
            pruned = True
            prune_reason = reason
            break
            
    assert pruned, "Pruner failed to detect hypo-dynamic cognitive freeze loop!"
    assert prune_reason == "cognitive_freeze_loop"

