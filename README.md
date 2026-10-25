# Energy-Search

Lyapunov phase-space telemetry and early-exit reasoning path pruning for large language models.

## Overview
Energy-Search monitors internal hidden-state trajectory dynamics during autoregressive reasoning. By tracking layer-wise relative velocity:

$$v_t = \frac{\|h_t - h_{t-1}\|}{\|h_{t-1}\|}$$

and token entropy across decoding steps, it detects chaotic phase divergence and hypo-dynamic cognitive freeze loops with a persistence counter ($K=4$). This enables early termination of degenerated reasoning paths before token budgets are depleted.

## Pre-Registered Gate Criteria
- **Gate 1 (Smoke):** Telemetry overhead $< 5.0\text{ ms / token}$ without degrading decoding stability.
- **Gate 2 (Stage 2 GSM8k):** Token reduction $\ge 2.0\times$ speedup while preserving reasoning accuracy within $10\%$ of unpruned greedy decoding on frozen Qwen-0.5B / 1.5B.

## Quickstart

### Local (Apple Silicon MPS / NVIDIA CUDA):
```bash
# Run smoke test (40 tokens with KV-caching)
python scripts/run_smoke.py --config configs/smoke_test.yaml

# Run Stage 2 Benchmark (GSM8k reasoning evaluation)
python scripts/eval_gsm8k.py --model-name Qwen/Qwen2.5-Coder-0.5B --num-samples 50 --wandb
```

### Google Colab:
Open `notebooks/colab_runner_energy_search.ipynb` in Colab on a standard T4 GPU. Includes automatic machine teardown (`runtime.unassign()`) to protect compute credits upon completion.

## Telemetry
Native Weights & Biases support via `--wandb` (project: `moonshot-energy-search`).

## License
MIT

