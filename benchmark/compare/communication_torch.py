#!/usr/bin/env python3
"""PyTorch/NCCL baseline for the B12 communication benchmark (dev-only).

The C++ benchmark emits the same case table (`bench_communication --dump-cases`);
this script runs the NCCL/cuBLAS path PyTorch uses on the same two GPUs and
writes a results JSON in the schema `compare_results.py` understands.

    bench_communication --dump-cases > out/b12/cases.json
    python benchmark/compare/communication_torch.py \
        --cases out/b12/cases.json --json out/b12/torch-communication.json
    python benchmark/compare/compare_results.py \
        out/b12/official-quant.json out/b12/torch-communication.json

Supported: allreduce (one-shot ids), allreduce_two_shot, allgather,
reduce_scatter, allreduce_fusion (plain and dynamic per-token FP8), and the
W=2 comm-GEMM ids. `peer_barrier`, `ep_layout` and `ep_dispatch_combine` are
skipped: DeepEP has no PyTorch package here, and a barrier is not a useful
event-timed case. W=4/W=8 comm-GEMM cases are skipped because this node has two
GPUs; the kops entry points are virtual-rank cases.

Timing: `calls` back-to-back eager launches per sample, CUDA events on every
rank, max rank time, median over `groups` samples. This is the PyTorch eager
baseline; the C++ benchmark defaults to CUDA graphs (the serving path). Use the
same `--calls/--groups` values when comparing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F


def parse_id(case_id: str) -> dict:
    out = {}
    for key in ("w", "m", "n", "k", "c"):
        m = re.search(rf"_{key}(\d+)", case_id)
        if m:
            out[key] = int(m.group(1))
    return out


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def rank_time(fn, device, calls: int, groups: int, warmup: int) -> list[float]:
    for _ in range(max(2, warmup // max(calls, 1))):
        fn()
    dist.barrier()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(groups):
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        local = start.elapsed_time(end) / calls
        # Reduce outside the timed interval: the slowest rank defines the sample.
        t = torch.tensor([local], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(float(t.item()))
    samples.sort()
    return samples


def prepare_case(case: dict, rank: int, device: torch.device, world: int):
    """Return (fn, note). Tensors are allocated once per case, outside the events."""
    cid = case["id"]
    op = case["op"]
    dt = torch.bfloat16
    rows = int(case["rows"])
    cols = int(case["cols"])
    n = rows * cols

    if op in ("allreduce", "allreduce_two_shot"):
        x = torch.randn(n, device=device, dtype=dt)
        y = torch.empty_like(x)

        def fn():
            y.copy_(x)
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return fn, None

    if op == "allgather":
        x = torch.randn(n, device=device, dtype=dt)
        y = torch.empty(n * world, device=device, dtype=dt)
        return (lambda: dist.all_gather_into_tensor(y, x)), None

    if op == "reduce_scatter":
        x = torch.randn(n * world, device=device, dtype=dt)
        y = torch.empty(n, device=device, dtype=dt)
        return (lambda: dist.reduce_scatter_tensor(y, x)), None

    if op == "allreduce_fusion":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        residual = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.randn(cols, device=device, dtype=dt)
        quant = "fp8" in cid
        q = torch.empty(rows, cols, device=device, dtype=torch.float8_e4m3fn)
        scales = torch.empty(rows, device=device, dtype=torch.float32)

        def fn():
            y = x.clone()
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            residual.add_(y)
            norm = F.rms_norm(residual, (cols,), w, eps=1e-5)
            if quant:
                amax = norm.float().abs().amax(dim=1)
                s = torch.clamp(amax / 448.0, min=1.0 / (448.0 * 512.0))
                q.copy_((norm.float() / s[:, None]).to(torch.float8_e4m3fn))
                scales.copy_(s)
        return fn, ("dynamic per-token FP8" if quant else None)

    if op == "comm_gemm":
        p = parse_id(cid)
        w = p.get("w", 1)
        if w != world:
            return None, f"needs world {w}; this node has {world}"
        m, nn, k = p.get("m"), p.get("n"), p.get("k")
        if m is None or nn is None or k is None:
            return None, "cannot parse shape"
        if "allgather_gemm" in cid:
            a = torch.randn(m, k, device=device, dtype=dt)
            b = torch.randn(nn, k, device=device, dtype=dt)
            full = torch.empty(m * w, k, device=device, dtype=dt)
            out = torch.empty(m * w, nn, device=device, dtype=dt)

            def fn():
                dist.all_gather_into_tensor(full, a)
                torch.mm(full, b.t(), out=out)
            return fn, None
        if "gemm_reduce_scatter" in cid:
            a = torch.randn(m, k, device=device, dtype=dt)
            b = torch.randn(nn, k, device=device, dtype=dt)
            local = torch.empty(m, nn, device=device, dtype=dt)
            out = torch.empty(m // w, nn, device=device, dtype=dt)

            def fn():
                torch.mm(a, b.t(), out=local)
                dist.reduce_scatter_tensor(out, local)
            return fn, None
        if "gemm_allreduce" in cid:
            a = torch.randn(m, k, device=device, dtype=dt)
            b = torch.randn(nn, k, device=device, dtype=dt)
            local = torch.empty(m, nn, device=device, dtype=dt)
            residual = torch.randn(m, nn, device=device, dtype=dt)
            wgt = torch.randn(nn, device=device, dtype=dt)
            fuse = "norm" in cid

            def fn():
                torch.mm(a, b.t(), out=local)
                dist.all_reduce(local, op=dist.ReduceOp.SUM)
                if fuse:
                    residual.add_(local)
                    F.rms_norm(residual, (nn,), wgt, eps=1e-5)
            return fn, ("residual + RMSNorm" if fuse else None)
        return None, "unsupported comm-GEMM id"

    if op == "peer_barrier":
        return None, "barrier is not event-timed; use the C++ case"
    if op in ("ep_layout", "ep_dispatch_combine"):
        return None, "no PyTorch/DeepEP package on this node"
    return None, f"unsupported op {op}"


def worker(rank: int, world: int, port: int, cases: list[dict], out_path: str,
           calls: int, groups: int, warmup: int, device_info: dict) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    results = []
    skipped = []
    try:
        for case in cases:
            fn, note = prepare_case(case, rank, device, world)
            if fn is None:
                if rank == 0:
                    skipped.append({"id": case["id"], "note": note})
                continue
            samples = rank_time(fn, device, calls, groups, warmup)
            if rank == 0:
                med = samples[len(samples) // 2]
                p10 = samples[max(0, len(samples) // 10)]
                p90 = samples[min(len(samples) - 1, len(samples) * 9 // 10)]
                p99 = samples[min(len(samples) - 1, len(samples) * 99 // 100)]
                by = int(case.get("bytes", 0))
                fl = int(case.get("flops", 0))
                results.append({
                    "id": case["id"], "op": case["op"], "dtype": case["dtype"],
                    "rows": case["rows"], "cols": case["cols"],
                    "bytes": by, "flops": fl,
                    "median_ms": med, "p10_ms": p10, "p90_ms": p90, "p99_ms": p99,
                    "min_ms": samples[0],
                    "gbs": (by / (med * 1e6)) if med > 0 else 0.0,
                    "tflops": (fl / (med * 1e9)) if med > 0 else 0.0,
                    "note": note or "PyTorch eager / NCCL",
                })
    finally:
        dist.destroy_process_group()
    if rank == 0:
        spec = {
            "schema": 1,
            "backend": "pytorch-nccl",
            "torch": torch.__version__,
            "device": device_info,
            "protocol": {"calls": calls, "groups": groups, "mode": "eager"},
            "results": results,
            "skipped": skipped,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=1)
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="JSON from bench_communication --dump-cases")
    ap.add_argument("--json", required=True, help="output PyTorch results JSON")
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma-separated case-id substrings")
    ap.add_argument("--skip", default=None, help="comma-separated case-id substrings")
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("needs 2 CUDA devices", file=sys.stderr)
        return 77
    with open(args.cases, "r", encoding="utf-8") as f:
        spec = json.load(f)
    cases = spec.get("cases", [])
    if args.only:
        needles = [s for s in args.only.split(",") if s]
        cases = [c for c in cases if any(s in c["id"] for s in needles)]
    if args.skip:
        needles = [s for s in args.skip.split(",") if s]
        cases = [c for c in cases if not any(s in c["id"] for s in needles)]
    if not cases:
        print("no cases", file=sys.stderr)
        return 2
    device_info = spec.get("device", {})
    port = free_port()
    torch.multiprocessing.spawn(
        worker,
        args=(2, port, cases, args.json, args.calls, args.groups, args.warmup, device_info),
        nprocs=2,
        join=True,
    )
    with open(args.json, "r", encoding="utf-8") as f:
        out = json.load(f)
    print(f"wrote {len(out['results'])} results, {len(out['skipped'])} skipped -> {args.json}")
    for s in out["skipped"]:
        print(f"  skip {s['id']}: {s['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
