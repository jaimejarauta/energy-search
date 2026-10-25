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

