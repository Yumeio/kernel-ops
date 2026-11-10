#!/usr/bin/env python3
"""PyTorch eager baselines for the B14 CSA/HCA cases in `bench_sparse`.

Dev-only: this script is not built, linked, tested, or shipped. It reads the
case table from `bench_sparse --dump-cases` and writes results in the same JSON
schema as the C++ benchmark, so `compare_results.py` can diff them.

    build/bench/benchmark/bench_sparse --dump-cases > out/cases-sparse.json
    build/bench/benchmark/bench_sparse --rounds 3 --json out/kops-sparse.json
    python benchmark/compare/csa_torch.py --cases out/cases-sparse.json \\
        --json out/torch-csa.json --rounds 3
    python benchmark/compare/compare_results.py out/kops-sparse.json out/torch-csa.json \\
        --only kv_compress compressed_attention

Why a separate script instead of a `make_op` branch in `torch_baseline.py`:
`kv_compress` is not an attention op and its reference math is a per-entry
weighted softmax over 2m overlapping candidate rows, which is easier to read (and
to check against the kernel) on its own. The `sparse_topk` / `lightning_indexer`
/ `sparse_mla` cases in the same case file are skipped and listed under
`skipped`, so pointing this script at the whole `bench_sparse` table works.

## Reference math and its mapping to the kernels

### `kv_compress` (csrc/attention/sparse.cu, `kv_compressor_kernel`)

One compressed entry per ratio window. For entry `e` of a sequence starting at
token 0 (`cu_seqlens = [0, T]`), with `m = ratio_m`, `c = entry_dim`:

    candidate rows, CSA (overlap = true, 2m candidates):
      j in [0, m):  z = zb[e*m + j], v = cb[e*m + j], bias = bias_b[j]   ("previous window")
      j in [0, m):  z = za[(e+1)*m + j], v = ca[(e+1)*m + j], bias = bias_a[j]  ("current window")
    candidate rows, HCA (overlap = false, m candidates):
      j in [0, m):  z = za[e*m + j], v = ca[e*m + j], bias = bias_a[j]

    w    = exp(z + bias - max(z + bias))          per channel
    entry = sum(w * v) / sum(w)                   per channel
    entry = entry * rsqrt(mean(entry^2) + eps) * norm_scale * norm_weight   (fused post-norm)

The kernel skips candidates past the end of the sequence. With `T = E * m` that
only affects the *current-window* half of the last entry (rows `T .. T+m-1`), so
the baseline pre-pads `za` with `-inf` logits and `ca` with zeros once; those
rows then contribute `exp(-inf) = 0` exactly like the kernel's `continue`. The
padding is input preparation (done outside the timed region), not part of the
math.

### `compressed_attention` (csrc/attention/sparse.cu, `compressed_attention_kernel`)

Shared-KV MQA (one latent head, `Hq` query heads) over three key sets under one
softmax:

    keys(CSA) = selected[e, :topk]            (physical compressed-entry slots)
    keys(HCA) = compressed[0 : num_compressed[req]]   (dense range, paged)
    keys     += sliding window of the last `window_size` raw tokens at `positions[e]`
    logits   = q . k * sm_scale               (shared K = V, value_dim = entry_dim)
    m        = max(max(logits), sink[head])
    o        = sum(exp(logits - m) * k) / (sum(exp(logits - m)) + exp(sink - m))
    lse      = m + log(sum(exp(logits - m)) + exp(sink - m))

The baseline builds the same key set as a `[T, K, D]` tensor (the selected
entries are gathered with `comp[selected]`, the window branch is a pre-gathered
`[T, window, D]` view of the raw cache - the kops kernel reaches it through the
page table, the baseline through a ring-buffer index) and runs the math with
`torch.matmul` + `torch.softmax`-style eager ops. Logits/softmax are fp32, the
QK/PV matmuls are bf16 (what an eager decode step does). The kops T1 body
accumulates both matmuls in fp32 from bf16 operands, so this script is a timing
reference, not a numerics reference; `kops_test_compressed_attention` is the
numerics check (it diffs against `ref::attention_f64`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
HEAD_DIM = 512       # c: DeepSeek-V4 compressed-entry dim (= value dim, shared K/V)
HEAD_DIM_MIN = 128   # the small-c compressor cases
EPS = 1e-6


def _kv_compress_shape(cid: str):
    m = re.search(r"\.t(\d+)\.c(\d+)\.m(\d+)\.(csa|hca)", cid)
    if not m:
        raise SystemExit(f"cannot parse kv_compress case id: {cid}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) == "csa"


def _compressed_attention_shape(cid: str):
    m = re.search(r"\.(csa|hca)\.t(\d+)\.h(\d+)\.", cid)
    if not m:
        raise SystemExit(f"cannot parse compressed_attention case id: {cid}")
    branch, rows, heads = m.group(1), int(m.group(2)), int(m.group(3))
    pool_m = re.search(r"pool(\d+)(k?)", cid)
    dense_m = re.search(r"dense(\d+)", cid)
    topk_m = re.search(r"topk(\d+)", cid)
    win_m = re.search(r"\.w(\d+)", cid)
    if win_m is None:
        raise SystemExit(f"compressed_attention case id has no window: {cid}")
    if dense_m is not None:
        pool = dense = int(dense_m.group(1))       # 64k tokens at ratio 128 -> 512 entries
        topk = dense
    elif pool_m is not None:
        pool = int(pool_m.group(1)) * (1024 if pool_m.group(2) == "k" else 1)
        dense = 0
        topk = int(topk_m.group(1)) if topk_m is not None else pool
    else:
        raise SystemExit(f"compressed_attention case id has no key set: {cid}")
    return branch == "csa", rows, heads, pool, topk, dense, int(win_m.group(1))


def make_kv_compress(case: dict, device: torch.device):
    """DeepSeek-V4 CSA (overlap, 2m candidates) / HCA (m candidates) compressor."""
    tokens, channels, ratio, overlap = _kv_compress_shape(case["id"])
    dt = DTYPES[case["dtype"]]
    entries = tokens // ratio

    ca = torch.randn(tokens, channels, device=device, dtype=dt)
    cb = torch.randn(tokens, channels, device=device, dtype=dt)
    za = torch.randn(tokens, channels, device=device, dtype=dt)
    zb = torch.randn(tokens, channels, device=device, dtype=dt)
    bias_a = torch.randn(ratio, channels, device=device, dtype=torch.float32) * 0.1
    bias_b = torch.randn(ratio, channels, device=device, dtype=torch.float32) * 0.1
    nw = torch.ones(channels, device=device, dtype=dt)

    # Input preparation: the kernel skips candidates past the sequence end, which
    # for T = E*m only hits the current-window half of the last entry.
    za_pad = torch.cat([za, torch.full((ratio, channels), float("-inf"), device=device, dtype=dt)], 0)
    ca_pad = torch.cat([ca, torch.zeros(ratio, channels, device=device, dtype=dt)], 0)

    def run():
        if overlap:
            z_prev = zb.view(entries, ratio, channels).float() + bias_b
            v_prev = cb.view(entries, ratio, channels).float()
            z_curr = za_pad[ratio:].view(entries, ratio, channels).float() + bias_a
            v_curr = ca_pad[ratio:].view(entries, ratio, channels).float()
            z = torch.cat([z_prev, z_curr], 1)
            v = torch.cat([v_prev, v_curr], 1)
        else:
            z = za.view(entries, ratio, channels).float() + bias_a
            v = ca.view(entries, ratio, channels).float()
        mx = z.amax(1, keepdim=True)
        w = torch.exp(z - mx)
        out = (w * v).sum(1) / w.sum(1)
        rstd = torch.rsqrt(out.square().mean(-1, keepdim=True) + EPS)
        return (out * rstd * nw.float()).to(dt)

    return run


def make_compressed_attention(case: dict, device: torch.device):
    """CSA (selected entries) / HCA (dense compressed range) + SWA + sink, one softmax."""
    csa, rows, heads, pool, topk, dense, window = _compressed_attention_shape(case["id"])
    dt = DTYPES[case["dtype"]]
    dim = HEAD_DIM
    npos = 65536                       # 64k decode context: window branch reads the tail
    scale = dim ** -0.5

    q = torch.randn(rows, heads, dim, device=device, dtype=dt)
    comp = torch.randn(pool, dim, device=device, dtype=dt)
    raw = torch.randn(rows, window, dim, device=device, dtype=dt)   # pre-gathered tail of the raw cache
    sink = torch.randn(heads, device=device, dtype=torch.float32)

    if csa:
        # Same selected set the C++ checkbox builds: distinct slots (stride coprime
        # with the power-of-two pool), scattered by a per-row offset.
        i = torch.arange(topk, device=device)
        stride, offset = 4099, 7919
        sel = (i[None, :] * stride + torch.arange(rows, device=device)[:, None] * offset) % pool
    else:
        sel = None
    dense_view = comp[:dense].unsqueeze(0).expand(rows, dense, dim)

    def run():
        # The compressed branch is gathered out of the paged cache per step, as an
        # eager decode does; the SWA branch is a contiguous slice of the raw ring
        # buffer, which is why it is pre-gathered here.
        keep = comp[sel] if csa else dense_view
        keys = torch.cat([keep, raw], 1) if window > 0 else keep   # [rows, K, dim]
        logits = torch.matmul(q, keys.transpose(1, 2)).float() * scale
        mx = torch.maximum(logits.amax(-1, keepdim=True), sink[None, :, None])
        p = torch.exp(logits - mx)
        den = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
        o = torch.matmul(p.to(dt), keys) / den.to(dt)
        lse = (mx + den.log()).squeeze(-1)
        return o, lse

    return run


def make_op(case: dict, device: torch.device):
    op = case["op"]
    if op == "kv_compress":
        return make_kv_compress(case, device)
    if op == "compressed_attention":
        return make_compressed_attention(case, device)
    raise SystemExit(f"unsupported op: {op}")


def bench(fn, warmup: int, iters: int, flush_mb: int = 0, rounds: int = 1) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    flush = None
    if flush_mb > 0:
        flush = (torch.empty(flush_mb * 1024 * 1024, dtype=torch.uint8, device="cuda"),
                 torch.empty(flush_mb * 1024 * 1024, dtype=torch.uint8, device="cuda"))
        flush[1].fill_(0x5A)
    rounds = max(1, rounds)
    for _ in range(rounds):
        for _ in range(iters):
            if flush is not None:
                flush[0].copy_(flush[1])
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
    samples.sort()

    n = len(samples)
    return {
        "median_ms": samples[n // 2],
        "p10_ms": samples[n // 10],
        "p90_ms": samples[min(n - 1, (n * 9) // 10)],
        "min_ms": samples[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="JSON from bench_sparse --dump-cases")
    ap.add_argument("--json", required=True, help="output results JSON")
    ap.add_argument("--only", default=None, help="comma-separated case-id substrings")
    ap.add_argument("--skip", default=None, help="comma-separated case-id substrings")
    ap.add_argument("--flush-l2", type=int, default=0, metavar="MB")
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("torch: CUDA is not available", file=sys.stderr)
        return 1
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    with open(args.cases, "r", encoding="utf-8") as f:
        spec = json.load(f)

    def matches(case_id: str, csv: str | None) -> bool:
        if not csv:
            return False
        return any(tok.strip() and tok.strip() in case_id for tok in csv.split(","))

    cases = spec["cases"]
    if args.only:
        cases = [c for c in cases if matches(c["id"], args.only)]
    if args.skip:
        cases = [c for c in cases if not matches(c["id"], args.skip)]
    if not cases:
        print("torch: no case matches the filters", file=sys.stderr)
        return 1

    results, skipped = [], []
    for case in cases:
        try:
            fn = make_op(case, device)
        except SystemExit:
            skipped.append(case["id"])
            continue
        with torch.inference_mode():
            st = bench(fn, int(case["warmup"]), int(case["iters"]),
                       flush_mb=args.flush_l2, rounds=args.rounds)
        ms = st["median_ms"]
        results.append(
            {
                "id": case["id"],
                "op": case["op"],
                "dtype": case["dtype"],
                "rows": case["rows"],
                "cols": case["cols"],
                "bytes": case["bytes"],
                "flops": case["flops"],
                **st,
                "gbs": case["bytes"] / (ms * 1e6) if ms > 0 else 0.0,
                "tflops": case["flops"] / (ms * 1e9) if ms > 0 else 0.0,
            }
        )
        print(f"{case['id']:<52} {ms:10.4f} ms  {st['min_ms']:10.4f} ms  "
              f"{results[-1]['gbs']:9.1f} GB/s")

    props = torch.cuda.get_device_properties(0)
    out = {
        "schema": 1,
        "backend": f"torch {torch.__version__} (eager)",
        "provenance": {
            "commit": "torch-baseline",
            "dirty": 1,
            "flags_hash": "",
            "archs": "",
            "toolkit": torch.version.cuda or "",
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "cc": f"{props.major}.{props.minor}",
            "sms": props.multi_processor_count,
            "driver": 0,
            "runtime": 0,
            "peak_gbs": 0.0,
        },
        "skipped": skipped,
        "results": results,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if skipped:
        print(f"skipped {len(skipped)} case(s): {', '.join(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
