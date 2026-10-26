#!/usr/bin/env python3
"""Offline analysis of traced greedy runs (Colab run 3).
1. Signal test: AUROC of early-prefix features for "final answer wrong", per layer, vs a shuffled-label control.
2. Pruning frontier: thresholds chosen on the calibration traces (GSM8K train), applied to the test traces.
Gate (2026-10-26): KILL if velocity AUROC <= entropy AUROC + 0.02 for every layer and k; PASS if velocity adds
>= 0.05 and its pruning rule beats entropy-only and stop-sequence at equal accuracy on test.
"""

import argparse
import glob
import json
import os
import random

import torch

from energy_search.analysis import auroc, early_features, entropy_rule_step, velocity_rule_step
from energy_search.benchmarks import evaluate_answer, extract_boxed, math_answers_equal, truncate_at_answer


def load(dir_):
    rows = []
    for path in sorted(glob.glob(os.path.join(dir_, "shard_*.pt"))):
        rows += torch.load(path, weights_only=False)
    cfg = json.load(open(os.path.join(dir_, "manifest.json")))["config"] if os.path.exists(os.path.join(dir_, "manifest.json")) else {}
    return rows, cfg


def layer_trace(row, layer):
    t = row["trace"]
    return {"v_rel": t[f"v{layer}"], "entropy": t["entropy"][1:], "cos": t[f"c{layer}"]}


def signal_table(rows, layers, ks, seed=0):
    """AUROC for predicting a wrong final answer from the first k tokens."""
    wrong = [int(not r["correct"]) for r in rows]
    shuffled = wrong[:]
    random.Random(seed).shuffle(shuffled)
    table = {}
    for layer in layers:
        for k in ks:
            feats = [early_features(layer_trace(r, layer), k) for r in rows]
            for name in feats[0]:
                vals = [f[name] for f in feats]
                table[f"L{layer}/k{k}/{name}"] = {"auroc": auroc(vals, wrong), "shuffled": auroc(vals, shuffled)}
    return table


def prune_outcome(row, step, tok, dataset):
    """(tokens used, correct) if generation stops after token `step` (None = no prune)."""
    if step is None:
        return len(row["ids"]), row["correct"]
    text = tok.decode(row["ids"][:step + 1], skip_special_tokens=True)
    ok = evaluate_answer(text, row["answer"]) if dataset == "gsm8k" else \
        (extract_boxed(text) is not None and math_answers_equal(extract_boxed(text), row["answer"]))
    return step + 1, ok


def evaluate_rule(rows, rule, tok, dataset):
    used, correct = 0, 0
    for r in rows:
        n, ok = prune_outcome(r, rule(r), tok, dataset)
        used += n
        correct += ok
    return used, correct / len(rows)


def stopseq_rule(tok):
    def rule(r):
        cut = truncate_at_answer(r["text"])
        return None if cut is None else max(0, len(tok(cut, add_special_tokens=False).input_ids) - 1)
    return rule


def pick_threshold(rows, make_rule, grid, tok, dataset, max_drop=0.01):
    """Most token-saving threshold whose accuracy drop on these rows is <= max_drop."""
    base_tokens, base_acc = evaluate_rule(rows, lambda r: None, tok, dataset)
    best = (None, base_tokens, base_acc)
    for g in grid:
        used, acc = evaluate_rule(rows, make_rule(g), tok, dataset)
        if base_acc - acc <= max_drop and used < best[1]:
            best = (g, used, acc)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib-dir", required=True, help="traces on GSM8K train (threshold selection)")
    p.add_argument("--test-dir", required=True, help="traces on GSM8K test or MATH-500")
    p.add_argument("--dataset", choices=["gsm8k", "math500"], default="gsm8k")
    p.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer
    calib, cfg = load(args.calib_dir)
    test, _ = load(args.test_dir)
    tok = AutoTokenizer.from_pretrained(cfg.get("model", "Qwen/Qwen2.5-1.5B-Instruct"))
    layers = sorted(int(k[1:]) for k in test[0]["trace"] if k.startswith("v"))

    signals = signal_table(test, layers, args.ks)
    best = {}
    for key, val in signals.items():
        layer_k, name = key.rsplit("/", 1)
        best.setdefault(layer_k, {})[name] = val["auroc"]
    deltas = []
    for layer_k, d in best.items():
        v_auc = max(abs(d["v_mean"] - 0.5), abs(d["v_max"] - 0.5)) + 0.5
        h_auc = max(abs(d["h_mean"] - 0.5), abs(d["h_max"] - 0.5), abs(d["h_frac_gt1"] - 0.5)) + 0.5
        deltas.append((layer_k, v_auc, h_auc, v_auc - h_auc))
    print(f"{'layer/k':<14} {'velocity AUROC':>15} {'entropy AUROC':>14} {'delta':>7}")
    for layer_k, v, h, dlt in deltas:
        print(f"{layer_k:<14} {v:>15.3f} {h:>14.3f} {dlt:>+7.3f}")
    max_delta = max(d[3] for d in deltas)

    # Pruning frontier: thresholds from calibration traces, evaluated once on test
    rules = {"stop_sequence": (lambda _: stopseq_rule(tok), [None])}
    rules["entropy"] = (lambda g: (lambda r: entropy_rule_step(r["trace"]["entropy"], g[0], g[1])),
                        [(t, k) for t in (0.5, 1.0, 1.5, 2.0, 3.0) for k in (3, 5, 8, 12)])
    for layer in layers:
        rules[f"velocity_L{layer}"] = (lambda g, L=layer: (lambda r: velocity_rule_step(r["trace"][f"v{L}"], g[0], g[1])),
                                       [(t, k) for t in (2.0, 2.5, 3.0, 4.0) for k in (2, 3, 5, 8)])
    base_tokens, base_acc = evaluate_rule(test, lambda r: None, tok, args.dataset)
    frontier = {"greedy": {"tokens": base_tokens, "accuracy": base_acc}}
    for name, (make, grid) in rules.items():
        g, _, _ = pick_threshold(calib, make, grid, tok, args.dataset)
        used, acc = evaluate_rule(test, make(g), tok, args.dataset) if g is not None or name == "stop_sequence" \
            else (base_tokens, base_acc)
        frontier[name] = {"threshold": g, "tokens": used, "accuracy": acc, "token_ratio": base_tokens / max(1, used)}
        print(f"{name:<16} thr={g} tokens={used} ({base_tokens / max(1, used):.2f}x) acc={acc:.3f}")

    best_vel = max((v for k, v in frontier.items() if k.startswith("velocity")), key=lambda v: v["token_ratio"])
    kill = max_delta <= 0.02
    passed = (max_delta >= 0.05 and best_vel["accuracy"] >= frontier["entropy"]["accuracy"] - 0.01
              and best_vel["tokens"] < min(frontier["entropy"]["tokens"], frontier["stop_sequence"]["tokens"]))
    verdict = "KILL" if kill else "PASS" if passed else "INCONCLUSIVE"
    print(f"max velocity-minus-entropy AUROC = {max_delta:+.3f} -> {verdict}")
    with open(args.out, "w") as f:
        json.dump({"signals": signals, "auroc_deltas": deltas, "frontier": frontier, "verdict": verdict,
                   "n_calib": len(calib), "n_test": len(test)}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
