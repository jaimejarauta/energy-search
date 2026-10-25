#!/usr/bin/env python3
"""Execution script for Energy-Search smoke test."""

import argparse
import json
import os
import sys
import time
import yaml
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from energy_search.sampler import sample_with_energy_pruning

class ToyTransformerLM(nn.Module):
    """Minimal autoregressive Transformer returning hidden states for smoke testing."""
    def __init__(self, vocab_size=100, d_model=64, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        h = self.embed(token_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.head(h)
        return logits, h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/smoke_test.yaml")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens override")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Device auto-detection: CUDA -> Apple Silicon MPS -> CPU
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"Running Energy-Search Smoke Test on: {device} ({dev_name})")

    torch.manual_seed(cfg["inference"]["seed"])

    model = ToyTransformerLM(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"]
    ).to(device)
    model.eval()

    prompt = torch.randint(0, cfg["model"]["vocab_size"], (1, 10), device=device)

    max_new_tokens = args.max_tokens or cfg["inference"]["max_new_tokens"]

    t0 = time.time()
    result = sample_with_energy_pruning(
        model=model,
        input_ids=prompt,
        max_new_tokens=max_new_tokens,
        monitor_layer_idx=cfg["model"]["monitor_layer"],
        tau=cfg["inference"]["tau_instability"],
        k_persist=cfg["inference"]["k_persist"],
        freeze_vel_tau=cfg["inference"].get("freeze_vel_tau", 0.05),
        freeze_ent_tau=cfg["inference"].get("freeze_ent_tau", 0.15),
        k_freeze=cfg["inference"].get("k_freeze", 6),
        device=device
    )
    elapsed = time.time() - t0
    tokens = result["tokens_generated"]
    latency_ms_per_token = (elapsed / max(1, tokens)) * 1000.0

    print(f"Tokens Generated: {tokens} | Total Time: {elapsed:.3f}s")
    print(f"Latency per Token: {latency_ms_per_token:.3f} ms")
    print(f"Pruned: {result['pruned']} (Step: {result['pruned_at_step']}, Reason: {result.get('prune_reason', 'none')})")
    print(f"Mean Instability Index: {result['mean_instability']:.4f}")

    passed = latency_ms_per_token < 5.0
    print(f"Rule evaluation (Latency < 5ms & telemetry functional): {'PASS' if passed else 'FAIL'}")

    os.makedirs("results", exist_ok=True)
    manifest = {
        "config": args.config,
        "date": cfg["date"],
        "tokens_generated": tokens,
        "latency_ms_per_token": latency_ms_per_token,
        "pruned": result["pruned"],
        "pruned_at_step": result["pruned_at_step"],
        "prune_reason": result.get("prune_reason", "none"),
        "mean_instability": result["mean_instability"],
        "passed": passed,
        "elapsed_sec": elapsed,
    }
    with open("results/smoke_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()

