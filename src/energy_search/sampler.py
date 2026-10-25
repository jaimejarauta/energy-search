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
    freeze_vel_tau: float = 0.05,
    freeze_ent_tau: float = 0.15,
    k_freeze: int = 6,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    eos_token_id: Optional[int] = None
) -> Dict[str, Any]:
    """Generates autoregressively while monitoring residual stream velocity with KV-caching.
    Prevents O(N^2) quadratic recomputation and detects both chaotic spikes and repetitive freeze loops.
    """
    if device is None:
        device = input_ids.device

    pruner = DynamicalInstabilityPruner(
        tau=tau,
        k_persist=k_persist,
        freeze_vel_tau=freeze_vel_tau,
        freeze_ent_tau=freeze_ent_tau,
        k_freeze=k_freeze
    )
    generated = input_ids.clone()
    h_history: List[torch.Tensor] = []
    instability_history: List[float] = []
    pruned_step: Optional[int] = None
    prune_reason: str = "completed"
    past_key_values = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            # Causal forward pass: use single-token forward with KV cache when available
            if step == 0:
                current_input = generated
                if hasattr(model, "config"):
                    outputs = model(current_input, output_hidden_states=True, use_cache=use_cache)
                else:
                    outputs = model(current_input)
            else:
                if past_key_values is not None:
                    current_input = generated[:, -1:]
                    outputs = model(current_input, past_key_values=past_key_values, output_hidden_states=True, use_cache=True)
                else:
                    current_input = generated
                    outputs = model(current_input, output_hidden_states=True) if hasattr(model, "config") else model(current_input)

            # Update KV cache
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                past_key_values = outputs.past_key_values

            # Extract monitored hidden state layer
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                h_all_layer = outputs.hidden_states[monitor_layer_idx]
                h_layer = h_all_layer[:, -1, :]
                logits = outputs.logits[:, -1, :]

                # Pre-populate velocity baseline from prompt tokens at step 0 to avoid cold-start dead-band
                if step == 0 and h_all_layer.size(1) > 2:
                    for t_idx in range(1, h_all_layer.size(1) - 1):
                        v_prompt = compute_relative_velocity(h_all_layer[:, t_idx, :], h_all_layer[:, t_idx - 1, :])
                        pruner.stats.history.append(v_prompt)
            elif isinstance(outputs, tuple):
                logits, h_all = outputs
                h_layer = h_all[:, -1, :] if (isinstance(h_all, torch.Tensor) and h_all.ndim == 3) else h_all
            else:
                logits = outputs[:, -1, :] if outputs.ndim == 3 else outputs
                h_layer = logits

            entropy = compute_token_entropy(logits)

            if step == 0:
                pruner.set_prompt_entropy(entropy)
                v_rel = 0.0
            else:
                v_rel = compute_relative_velocity(h_layer, h_history[-1])

            should_prune, inst_val, reason = pruner.evaluate_step(v_rel, entropy)
            h_history.append(h_layer)
            instability_history.append(inst_val)

            if should_prune and pruned_step is None:
                pruned_step = step
                prune_reason = reason
                break

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                prune_reason = "eos_token"
                break

    return {
        "generated_ids": generated,
        "pruned": pruned_step is not None,
        "pruned_at_step": pruned_step,
        "prune_reason": prune_reason,
        "tokens_generated": len(h_history),
        "mean_instability": sum(instability_history) / max(1, len(instability_history)),
    }

def sample_greedy(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    eos_token_id: Optional[int] = None
) -> Dict[str, Any]:
    """Standard unpruned greedy decoding baseline with KV caching."""
    if device is None:
        device = input_ids.device

    generated = input_ids.clone()
    past_key_values = None
    tokens_count = 0

    with torch.inference_mode():
        for step in range(max_new_tokens):
            if step == 0:
                current_input = generated
                if hasattr(model, "config"):
                    outputs = model(current_input, use_cache=use_cache)
                else:
                    outputs = model(current_input)
            else:
                if past_key_values is not None:
                    current_input = generated[:, -1:]
                    outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
                else:
                    current_input = generated
                    outputs = model(current_input)

            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                past_key_values = outputs.past_key_values

            if hasattr(outputs, "logits"):
                logits = outputs.logits[:, -1, :]
            elif isinstance(outputs, tuple):
                logits = outputs[0][:, -1, :] if outputs[0].ndim == 3 else outputs[0]
            else:
                logits = outputs[:, -1, :] if outputs.ndim == 3 else outputs

            tokens_count += 1
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return {
        "generated_ids": generated,
        "tokens_generated": tokens_count,
    }
