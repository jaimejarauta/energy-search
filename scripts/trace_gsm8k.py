#!/usr/bin/env python3
"""Batched greedy generation that records per-token residual-stream telemetry (Colab run 3).
For each problem: token ids, final text, strict correctness, and per-layer relative velocity / directional cosine
plus token entropy, as scalars. Shards are written as they finish, so a disconnect loses at most one batch.
Analysis: scripts/analyze_traces.py. Plan: research/COLAB_LONG_RUNS_2026-10-26.md section 3.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from energy_search.benchmarks import evaluate_answer, extract_boxed, extract_numerical_answer, math_answers_equal
from energy_search.runtime import Run

INSTRUCTIONS = {
    "gsm8k": "Solve step by step, then give the final answer as '#### <number>'.",
    "math500": "Solve step by step, then put the final answer in \\boxed{}.",
}


def load_problems(dataset: str, split: str, n: int, offset: int):
    from datasets import load_dataset
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)  # MIT
        items = [{"question": x["question"], "answer": extract_numerical_answer(x["answer"])} for x in ds]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")  # MIT
        items = [{"question": x["problem"], "answer": x["answer"]} for x in ds]
    return items[offset:offset + n]


def is_correct(dataset: str, text: str, answer) -> bool:
    if dataset == "gsm8k":
        return evaluate_answer(text, answer)
    pred = extract_boxed(text)
    return pred is not None and math_answers_equal(pred, answer)


def decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot find decoder layers")


@torch.no_grad()
def trace_batch(model, tok, prompts, layer_ids, max_new_tokens, device):
    """Greedy-decode a left-padded batch; return per-row token ids and telemetry scalars."""
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    B = enc.input_ids.size(0)
    captured = {}

    def hook(i):
        def fn(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h[:, -1].float()  # residual stream (pre final norm) at the newest position
        return fn

    handles = [decoder_layers(model)[i].register_forward_hook(hook(i)) for i in layer_ids]
    try:
        attn = enc.attention_mask
        pos = (attn.cumsum(-1) - 1).clamp_min(0)
        out = model(input_ids=enc.input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
        past = out.past_key_values
        prev = {i: captured[i] for i in layer_ids}
        prev_d = {i: None for i in layer_ids}
        logits = out.logits[:, -1].float()
        next_pos = pos[:, -1:] + 1

        ids = [[] for _ in range(B)]
        tr = [{"entropy": [], **{f"v{i}": [] for i in layer_ids}, **{f"c{i}": [] for i in layer_ids}} for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=device)
        eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}

        for _ in range(max_new_tokens):
            logp = F.log_softmax(logits, -1)
            ent = -(logp.exp() * logp).sum(-1)
            nxt = logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    ids[b].append(int(nxt[b]))
                    tr[b]["entropy"].append(float(ent[b]))
            done |= torch.tensor([int(n) in eos for n in nxt], device=device)
            if done.all():
                break
            attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=device)], -1)
            out = model(input_ids=nxt[:, None], attention_mask=attn, position_ids=next_pos, past_key_values=past, use_cache=True)
            past, logits, next_pos = out.past_key_values, out.logits[:, -1].float(), next_pos + 1
            for i in layer_ids:
                h, d = captured[i], captured[i] - prev[i]
                v = d.norm(dim=-1) / (h.norm(dim=-1) + 1e-6)
                c = F.cosine_similarity(d, prev_d[i], dim=-1) if prev_d[i] is not None else torch.zeros(B, device=device)
                for b in range(B):
                    if not done[b]:
                        tr[b][f"v{i}"].append(float(v[b]))
                        tr[b][f"c{i}"].append(float(c[b]))
                prev[i], prev_d[i] = h, d
    finally:
        for hd in handles:
            hd.remove()
    return ids, tr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", choices=["gsm8k", "math500"], default="gsm8k")
    p.add_argument("--split", default="test")
    p.add_argument("--num", type=int, default=1319)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--layer-fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--out-dir", required=True, help="e.g. /content/drive/MyDrive/moonshot/energy-search/<run>")
    p.add_argument("--device", default="cuda")
    p.add_argument("--budget-hours", type=float, default=2.0)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if args.device == "cpu":
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    n_layers = len(decoder_layers(model))
    layer_ids = sorted({min(n_layers - 1, max(0, round(f * n_layers) - 1)) for f in args.layer_fracs})

    problems = load_problems(args.dataset, args.split, args.num, args.offset)
    run = Run(args.out_dir, vars(args) | {"layer_ids": layer_ids, "n_layers": n_layers}, budget_hours=args.budget_hours)
    shards = range(0, len(problems), args.batch_size)
    t0 = time.time()
    for s_idx, start in enumerate(shards):
        path = os.path.join(args.out_dir, f"shard_{s_idx:05d}.pt")
        if os.path.exists(path):
            continue  # resume: shard already done
        batch = problems[start:start + args.batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": f"{x['question']}\n{INSTRUCTIONS[args.dataset]}"}],
                                           tokenize=False, add_generation_prompt=True) for x in batch]
        ids, tr = trace_batch(model, tok, prompts, layer_ids, args.max_new_tokens, args.device)
        rows = []
        for x, i_, t_ in zip(batch, ids, tr):
            text = tok.decode(i_, skip_special_tokens=True)
            rows.append({"question": x["question"], "answer": x["answer"], "ids": i_, "text": text,
                         "correct": is_correct(args.dataset, text, x["answer"]), "trace": t_})
        torch.save(rows, path + ".tmp")
        os.replace(path + ".tmp", path)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"shard {s_idx + 1}/{len(shards)} | acc {acc:.2f} | mean tokens {sum(len(r['ids']) for r in rows) / len(rows):.0f} "
              f"| {time.time() - t0:.0f}s")
        run.check_budget(s_idx + 1, len(shards))
    run.write_manifest({"shards": len(shards), "layer_ids": layer_ids})


if __name__ == "__main__":
    main()
