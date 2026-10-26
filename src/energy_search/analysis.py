"""Offline analysis of greedy traces: signal AUROC and post-hoc replay of pruning rules.
For greedy decoding a pruning rule only truncates the trajectory, so one traced pass evaluates every rule.
"""

from typing import Dict, List, Optional, Sequence

from .telemetry import RollingVelocityStats


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """P(score of a random positive > score of a random negative), ties count 1/2 (Mann-Whitney)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        raise ValueError("auroc needs both classes")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):  # average ranks over ties
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_pos = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def entropy_rule_step(entropy: List[float], tau: float, k: int) -> Optional[int]:
    """First step at which entropy has exceeded tau for k consecutive tokens, else None."""
    run = 0
    for t, h in enumerate(entropy):
        run = run + 1 if h > tau else 0
        if run >= k:
            return t
    return None


def velocity_rule_step(v_rel: List[float], tau: float, k: int, window: int = 16) -> Optional[int]:
    """First step at which the past-window z-score of velocity has exceeded tau for k consecutive tokens."""
    stats = RollingVelocityStats(window_size=window)
    run = 0
    for t, v in enumerate(v_rel):
        run = run + 1 if stats.update_and_normalize(v) > tau else 0
        if run >= k:
            return t
    return None


def early_features(trace: Dict[str, List[float]], k: int) -> Dict[str, float]:
    """Summary statistics of the first k generated tokens (a pruner only sees the prefix)."""
    v, h, c = trace["v_rel"][:k], trace["entropy"][:k], trace["cos"][:k]
    n = max(1, len(v))
    return {
        "v_mean": sum(v) / n,
        "v_max": max(v, default=0.0),
        "h_mean": sum(h) / n,
        "h_max": max(h, default=0.0),
        "h_frac_gt1": sum(x > 1.0 for x in h) / n,
        "cos_mean": sum(c) / n,
    }
