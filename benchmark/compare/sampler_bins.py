#!/usr/bin/env python3
"""Real-logit statistics for the fused sampler's linear-bin candidate design (dev tool, M0).

  dump     run a small HF causal LM (default Qwen2.5-0.5B-Instruct) on a fixed prompt set and save
           the raw decode-step logits, [rows, V] float32, as .npy (needs torch + transformers)
  analyze  for each row and temperature, bin the raw logits into 4096 linear bins of width 1/32 on
           [-64, 64) and report how many tokens a filter has to collect as exact-refine candidates
           (numpy only)

The analysis mirrors the planned kernel: pass 1 counts per bin and keeps the exact row max M and
partition Z of y = x / T; afterwards the mass of bin j is bounded by cnt_j * exp(lo_j / T - M) and
cnt_j * exp(min(hi_j / T, M) - M). Bins that are kept (or dropped) under both bounds stream through
pass 2; only the boundary bins become candidates.
  top-k (vLLM ties)  candidates = tokens of the bin holding the k-th largest value
  top-p              candidates = bins j with  sum_{i>=j} U_i >= p Z  and  sum_{i>j} L_i < p Z
  top-k + top-p      sequential: the top-p range is computed inside the top-k set, so everything at
                     or above the top-k bin is collected (bounded by k + its tie bin)
Usage (WSL, torch venv):
  python benchmark/compare/sampler_bins.py dump --out ~/kops-logits/qwen05.npy
  python benchmark/compare/sampler_bins.py analyze ~/kops-logits/qwen05.npy
"""
import argparse
import json
import sys

PROMPTS = [
    "Write a Python function that merges two sorted lists and explain its complexity.",
    "Tell me a short story about a lighthouse keeper who finds a strange message in a bottle.",
    "Hãy giải thích ngắn gọn vì sao bầu trời có màu xanh.",
    "List ten creative names for a coffee shop and give one sentence about each.",
    "What is the capital of Australia? Answer, then give three facts about the city.",
    "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed?",
    "Translate into French: The weather is lovely today, let's go for a walk in the park.",
    "Write a haiku sequence about autumn rain, then explain the imagery you used.",
]

BINS, LO, WIDTH = 4096, -64.0, 1.0 / 32.0


def dump(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    rows = []
    for prompt in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=args.steps, output_logits=True,
                                 return_dict_in_generate=True)
        rows += [l[0].float().numpy() for l in out.logits]
        print(f"{len(out.logits):4d} steps  {prompt[:48]}", file=sys.stderr)
    arr = np.stack(rows).astype(np.float32)
    np.save(args.out, arr)
    print(f"saved {arr.shape} to {args.out}", file=sys.stderr)


def pct(v, qs=(50, 90, 99, 100)):
    import numpy as np
    return {f"p{q}": int(np.percentile(v, q)) for q in qs}


def analyze(args):
    import numpy as np

    raw = np.load(args.logits).astype(np.float64)
    views = {"fp32": raw}
    if args.bf16:
        b = raw.astype(np.float32).view(np.uint32)
        b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)  # RNE to bf16
        views["bf16"] = b.view(np.float32).astype(np.float64)
    report = {}
    edges_lo = LO + WIDTH * np.arange(BINS)
    for vname, X in views.items():
        rmax = X.max(axis=1)
        out = {"rows": int(X.shape[0]), "vocab": int(X.shape[1]),
               "row_max": {k: float(v) for k, v in zip(("min", "p50", "max"),
                                                       np.percentile(rmax, [0, 50, 100]))},
               "rows_max_ge_64": int((rmax >= 64).sum()), "rows_min_lt_-64": int((X.min(1) < -64).sum())}
        for T in args.temps:
            stats = {}
            cand = {"k50": [], "k1000": [], "p0.9": [], "p0.95": [], "k50p0.9": [],
                    "tie_k50": [], "nucleus0.95": [], "minp0.05_kept": []}
            for x in X:
                b = np.clip(np.floor((x - LO) / WIDTH), 0, BINS - 1).astype(np.int64)
                cnt = np.bincount(b, minlength=BINS)
                y = x / T
                M = y.max()
                Z = np.exp(y - M).sum()
                lo_y = edges_lo / T
                hi_y = np.minimum((edges_lo + WIDTH) / T, M)
                L = cnt * np.exp(lo_y - M)
                U = cnt * np.exp(hi_y - M)
                sufU = np.cumsum(U[::-1])[::-1]                 # sum_{i>=j} U_i
                sufL_above = np.concatenate([np.cumsum(L[::-1])[::-1][1:], [0.0]])  # sum_{i>j} L_i
                sufC = np.cumsum(cnt[::-1])[::-1]               # tokens at or above bin j
                xs = np.sort(x)[::-1]
                for k in (50, 1000):
                    bk = b[np.argsort(-x, kind="stable")[k - 1]]
                    cand[f"k{k}"].append(int(cnt[bk]))
                    if k == 50:
                        cand["tie_k50"].append(int((x >= xs[k - 1]).sum()))
                        cand["k50p0.9"].append(int(sufC[bk]))
                for p in (0.9, 0.95):
                    sel = (sufU >= p * Z) & (sufL_above < p * Z)
                    cand[f"p{p}"].append(int(cnt[sel].sum()))
                ps = np.exp(np.sort(y)[::-1] - M) / Z
                cand["nucleus0.95"].append(int(np.searchsorted(np.cumsum(ps), 0.95) + 1))
                cand["minp0.05_kept"].append(int((y >= M + np.log(0.05)).sum()))
            for name, v in cand.items():
                v = np.asarray(v)
                stats[name] = dict(pct(v), **{f"gt{c}": round(float((v > c).mean()), 4)
                                               for c in (1024, 2048, 4096)})
            out[f"T{T}"] = stats
        report[vname] = out
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    d.add_argument("--steps", type=int, default=128)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("logits")
    a.add_argument("--temps", type=float, nargs="+", default=[0.7, 1.0])
    a.add_argument("--bf16", action="store_true", help="also analyze the bf16-rounded logits")
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else analyze(args)


if __name__ == "__main__":
    main()
