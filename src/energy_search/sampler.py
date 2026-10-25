"""Search rollout engine with dynamic branch pruning for autoregressive generation."""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from .telemetry import compute_relative_velocity, compute_token_entropy
from .pruner import DynamicalInstabilityPruner

def sample_with_energy_pruning(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    monitor_layer_idx: int = -1,
    tau: float = 2.5,
    k_persist: int = 4,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Generates autoregressively while monitoring residual stream velocity."""
    if device is None:
        device = input_ids.device

    pruner = DynamicalInstabilityPruner(tau=tau, k_persist=k_persist)
    generated = input_ids.clone()
    h_history: List[torch.Tensor] = []
    instability_history: List[float] = []
    pruned_step: Optional[int] = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass: supports models returning hidden states
            outputs = model(generated, output_hidden_states=True) if hasattr(model, "config") else model(generated)
            
            if hasattr(outputs, "hidden_states"):
                h_layer = outputs.hidden_states[monitor_layer_idx][:, -1, :]
                logits = outputs.logits[:, -1, :]
            elif isinstance(outputs, tuple):
                logits, h_all = outputs
                h_layer = h_all[:, -1, :] if h_all.ndim == 3 else h_all
            else:
                logits = outputs[:, -1, :]
                h_layer = logits

            entropy = compute_token_entropy(logits)

            if step == 0:
                pruner.set_prompt_entropy(entropy)
                v_rel = 0.0
            else:
                v_rel = compute_relative_velocity(h_layer, h_history[-1])

            should_prune, inst_val = pruner.evaluate_step(v_rel, entropy)
            h_history.append(h_layer)
            instability_history.append(inst_val)

            if should_prune and pruned_step is None:
                pruned_step = step
                break

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

    return {
        "generated_ids": generated,
        "pruned": pruned_step is not None,
        "pruned_at_step": pruned_step,
        "tokens_generated": len(h_history),
        "mean_instability": sum(instability_history) / max(1, len(instability_history)),
    }
