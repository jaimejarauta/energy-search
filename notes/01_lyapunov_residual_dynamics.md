# Note 01: Lyapunov Residual Stream Dynamics and the 3-Way Discriminator

**Author:** Jaime Jarauta  
**Status:** Pre-registered active theory note  
**Target:** `src/energy_search/telemetry.py`, `src/energy_search/pruner.py`

---

## 1. The Naive Velocity Fallacy

A naive pruning heuristic that terminates generation whenever residual stream velocity spikes ($\|h_t - h_{t-1}\|_2 > \tau$) catastrophically prunes valid, high-quality reasoning.
When a model encounters a productive "aha moment" (e.g. backtracking: *"Wait, let me re-read the constraints"*), the semantic state undergoes a large discrete translation in latent space. The instantaneous velocity $\|h_t - h_{t-1}\|_2$ is as large as—or larger than—an ungrounded hallucination hallucinating an invalid formula.

---

## 2. The 3-Way Discriminator

To differentiate productive reasoning leaps from degenerative hallucinations, we analyze three coupled geometric properties:

| Property | Productive Leap / Backtracking | Hallucinatory Derailment / Chaos |
| :--- | :--- | :--- |
| **Entropy Dynamics** | Instantaneous entropy collapse ($\mathcal{H}(p_{t+1}) \to 0$). The model gains deterministic confidence in the new deduction. | Sustained elevated entropy ($\mathcal{H}(p_{t+k}) \ge \bar{\mathcal{H}}$). The model remains uncertain and confabulates. |
| **Velocity Half-Life** | Single-step Dirac impulse ($v_t \gg \mu$), followed by rapid exponential decay ($v_{t+1} \approx \frac{1}{2} v_t$) into a stable attractor. | Persistent high velocity across multiple tokens ($v_t, v_{t+1}, v_{t+2} > \mu + 2\sigma$). Continuous wandering outside capacity manifold. |
| **Directional Coherence** | Smooth trajectory curvature: $\cos(d_t, d_{t+1}) > 0.7$. | Erratic orthogonal scattering: $\cos(d_t, d_{t+1}) \approx 0$ or negative. |

---

## 3. Layer Selection & Formulation

### Layer 15 Rationale (Qwen-2.5-1.5B)
As shown by Akarlar (arXiv:2604.15400), low-level syntactic tokenization resolves by Layer 14, while semantic commitment and attractor basin alignment peak at **Layer 15** (53.6% depth), predicting hallucination susceptibility with Pearson $r = 0.776$. Final layers ($26\text{--}27$) suffer from unembedding projection noise.

### Metric Equations
Relative velocity:
$$v_t^{\text{rel}} = \frac{\|h_t^{(15)} - h_{t-1}^{(15)}\|_2}{\|h_t^{(15)}\|_2 + \epsilon}, \quad z_t = \frac{v_t^{\text{rel}} - \mu_{t, 16}}{\sigma_{t, 16} + \epsilon}$$

Dynamical Instability Index:
$$\mathcal{E}_t = \max(0, z_t) \times \left( \frac{\mathcal{H}_t}{\bar{\mathcal{H}}_{\text{prompt}} + \epsilon} \right)$$

### Persistence Pruning Rule
Prune branch $b$ if and only if:
$$\mathcal{E}_t > \tau_{\text{instability}} \quad \text{for } K = 4 \text{ consecutive tokens}$$
This protects isolated $K=1, 2$ spikes while terminating sustained chaotic orbits.


## Run log 2026-10-26: first real run (why greedy gives 1.00x)

Qwen2.5-0.5B-Instruct, GSM8K test first 50, greedy, chat template, max 768 tokens, strict scoring
(`####`, "the answer is", `\boxed{}`), MPS fp16. Default thresholds (tau 1.5, k 3, freeze_vel 0.25,
freeze_ent 0.8, k_freeze 3, colinear 0.85). Manifest: results/gsm8k_real_qwen05_manifest.json.

| Arm | Tokens | Accuracy |
|---|---|---|
| Greedy | 16,365 | 38.0% |
| Energy pruner | 16,365 (never fired) | 38.0% |
| Stop at first explicit answer | 16,520 | 38.0% |

Per-token telemetry on the last layer (p05 / p50 / p95): relative velocity 0.65 / 1.04 / 1.41; directional cosine
-0.71 / -0.41 / -0.11; token entropy 1.5e-5 / 0.009 / 1.19 nats.
Why nothing fires: the freeze rule needs v_rel < 0.25 (below p05 by 2.6x); the colinear rule needs cos > 0.85
(consecutive displacements of near-orthogonal states have cos ~ -0.5, so p95 is -0.11); the spike index multiplies
z by H/H_prompt, and the median entropy is 0.009 nats.
Decision: the thresholds were never calibrated to real telemetry. The next step is not re-tuning by hand but the
traced study (Colab run 3): does velocity predict wrong answers beyond entropy at all.
