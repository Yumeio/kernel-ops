#!/usr/bin/env python3
"""Measure the local roofline used by the benchmark report.

Dev-only. Reports D2D copy bandwidth and cuBLAS bf16/fp16 GEMM throughput on
the same device as the kernel benchmark, plus the current clock/power state.
"""
from __future__ import annotations

import argparse
import json
import subprocess

import torch


def timed(fn, warmup=5, reps=20, rounds=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(reps):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / reps)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--mb", type=int, default=256)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    n = args.mb * 1024 * 1024
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    ms_copy = timed(lambda: dst.copy_(src))
    copy_gbs = 2.0 * n / (ms_copy * 1e6)

    size = 4096
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    ms_bf16 = timed(lambda: torch.matmul(a, b), warmup=3, reps=10, rounds=3)
    bf16_tflops = 2.0 * size**3 / (ms_bf16 * 1e9)

    a16 = a.to(torch.float16)
    b16 = b.to(torch.float16)
    ms_fp16 = timed(lambda: torch.matmul(a16, b16), warmup=3, reps=10, rounds=3)
    fp16_tflops = 2.0 * size**3 / (ms_fp16 * 1e9)

    smi = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,"
             "temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip()
        keys = ["clock_sm_mhz", "clock_mem_mhz", "power_w", "power_limit_w",
                "temperature_c", "memory_used_mb", "memory_total_mb"]
        smi = dict(zip(keys, [x.strip() for x in out.split(",")]))
    except Exception as exc:
        smi = {"error": str(exc)}

    out = {
        "device": torch.cuda.get_device_name(0),
        "cc": f"{props.major}.{props.minor}",
        "sms": props.multi_processor_count,
        "copy_mb": args.mb,
        "copy_ms": ms_copy,
        "copy_gbs": copy_gbs,
        "gemm_bf16_4096_ms": ms_bf16,
        "gemm_bf16_4096_tflops": bf16_tflops,
        "gemm_fp16_4096_ms": ms_fp16,
        "gemm_fp16_4096_tflops": fp16_tflops,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia_smi": smi,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
