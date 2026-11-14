#!/usr/bin/env python3
"""TensorRT-LLM (trtllm-gen) adapter for the kernel-ops benchmark cases.

Dev-only, like `thirdparty.py`. Adapter contract:

    adapter(case: dict, device: torch.device) -> (fn, note) | (None, reason)

Two providers, tried in this order because the review in
`research_notes/review-B0-B8-va-chien-luoc-bench-gpu-thue.md` (section 6.5)
notes that most trtllm-gen kernels are reachable through FlashInfer:

  1. FlashInfer `trtllm_*` kernels (`flashinfer.fused_moe.trtllm_fp4_block_scale_moe`,
     `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`, `flashinfer.gemm.mm_bf16_fp4`).
     FlashInfer 0.7.2 is what the vendored TRT-LLM pins.
  2. A full TensorRT-LLM install, which registers the `torch.ops.trtllm.*` kernels
     when `tensorrt_llm` is imported. Only the low-level ops that are a single
     kernel call are used; the Python layer of TRT-LLM is a model runtime, not a
     kernel API.

What maps, and what does not:

  * `fp8_mma_tensorwise` / `fp8_mma_rowwise` -> `trtllm::cublas_scaled_mm`
    (per-tensor or per-row x per-column dequant, exactly the kops epilogue).
    The note says it is the cuBLASLt backend, not trtllm-gen: trtllm-gen's
    block-scaled FP8 GEMM uses 128x128 blocks on *both* operands, while the kops
    blockwise recipe keeps A per-token.
  * `int8_mma` -> `trtllm::cublas_scaled_mm` (W8A8, per-row x per-column).
  * `fp4_gemv_nv` (NVFP4 W4A16) -> `flashinfer.gemm.mm_bf16_fp4` when the build
    exposes it.
  * MoE cases in `bench_moe` are routing/metadata/permute micro-ops, or a fused
    expert GEMM that already carries pre-computed routing. The trtllm-gen MoE
    kernels route inside the kernel, so benchmarking them here would time a
    different workload: those cases skip.

    python benchmark/compare/thirdparty_trtllm.py
    python benchmark/compare/thirdparty_trtllm.py --cases out/thirdparty/cases/gemm.json
    python benchmark/compare/thirdparty_trtllm.py --run --cases ... --json out/trtllm-gemm.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys

try:
    import torch
except Exception:  # noqa: BLE001 - a missing torch must not break --cases
    torch = None

IMPLS = {"trtllm": None}

HINT = ("install one of: `pip install flashinfer-python==0.7.2` (FlashInfer trtllm_* kernels), or a "
        "TensorRT-LLM build/container from 3rd/TensorRT-LLM (registers torch.ops.trtllm.*)")

RULES = (
    ("fp8_mma_tensorwise (e4m3 x e4m3, per-tensor dequant)",
     "trtllm::cublas_scaled_mm (cuBLASLt, scalar scales)"),
    ("fp8_mma_rowwise (e4m3 x e4m3, per-row x per-col dequant)",
     "trtllm::cublas_scaled_mm (cuBLASLt, 1-D scales)"),
    ("int8_mma (W8A8, per-row x per-col dequant)", "trtllm::cublas_scaled_mm (cuBLASLt int8)"),
    ("fp4_gemv_nv (NVFP4 W4A16, M=1)", "flashinfer.gemm.mm_bf16_fp4 (trtllm-gen on SM100)"),
    ("fp8_mma_blockwise", "none: kops keeps A per-token, trtllm-gen blocks both operands 128x128"),
    ("fp4_gemv_mx (MXFP4, 32-element blocks)", "none: the bf16 x fp4 GEMM path is NVFP4 (16-element blocks)"),
    ("moe_* / chain_moe_route_permute", "none: trtllm-gen MoE routes inside the kernel"),
)


def _gemm_shape(case: dict):
    n, k = int(case["rows"]), int(case["cols"])
    flops = int(case.get("flops") or 0)
    if n <= 0 or k <= 0 or flops <= 0:
        return None
    return flops // (2 * n * k), n, k


def _device_cc(device) -> tuple:
    if torch is None or not torch.cuda.is_available():
        return (0, 0)
    index = getattr(device, "index", None)
    if index is None:
        index = torch.cuda.current_device()
    return tuple(torch.cuda.get_device_capability(index))


def decide(case: dict) -> dict:
    """Pure mapping decision: no allocation, no package import."""
    op = case.get("op", "")

    if op in ("fp8_mma_tensorwise", "fp8_mma_rowwise"):
        shape = _gemm_shape(case)
        if shape is None:
            return {"api": None, "note": "unparsed FP8 GEMM shape"}
        m, n, k = shape
        per_channel = op == "fp8_mma_rowwise"
        return {"api": "scaled_mm", "dtype": "float8_e4m3fn", "m": m, "n": n, "k": k,
                "per_channel": per_channel,
                "note": ("trtllm::cublas_scaled_mm, per-row x per-col FP8 dequant (cuBLASLt backend, "
                         "not trtllm-gen)"
                         if per_channel else
                         "trtllm::cublas_scaled_mm, per-tensor FP8 dequant (cuBLASLt backend)")}

    if op == "int8_mma":
        shape = _gemm_shape(case)
        if shape is None:
            return {"api": None, "note": "unparsed int8 GEMM shape"}
        m, n, k = shape
        return {"api": "scaled_mm", "dtype": "int8", "m": m, "n": n, "k": k, "per_channel": True,
                "note": "trtllm::cublas_scaled_mm, int8 W8A8 with per-row x per-col dequant"}

    if op == "fp8_mma_blockwise":
        return {"api": None, "note":
                "trtllm-gen block-scaled FP8 uses 128x128 blocks on both operands; the kops case keeps A "
                "per-token (1x128), so the scale layout differs"}

    if op == "fp4_gemv_nv":
        shape = _gemm_shape(case)
        if shape is None:
            return {"api": None, "note": "unparsed FP4 GEMM shape"}
        m, n, k = shape
        if k % 32:
            return {"api": None, "note": f"NVFP4 needs K % 32 == 0 (K={k})"}
        return {"api": "bf16_fp4", "m": m, "n": n, "k": k,
                "note": "flashinfer.gemm.mm_bf16_fp4 (bf16 activations x NVFP4 weights)"}

    if op == "fp4_gemv_mx":
        return {"api": None, "note":
                "the bf16 x fp4 GEMM path is NVFP4 (e4m3 scales per 16 elements); this case is MXFP4 "
                "(ue8m0 scales per 32), so the scale format differs"}
    if op in ("fp4_marlin_mx", "fp4_marlin_nv"):
        return {"api": None, "note":
                "marlin W4A16 needs packed int4 tiles in the marlin layout; trtllm-gen only exposes "
                "NVFP4 (fp4 activations) and nvfp4 marlin takes FP4 activations, not bf16"}

    if op in ("grouped_fp8", "moe_gemv_fp8", "moe_w2_gemm"):
        return {"api": None, "note":
                "the trtllm-gen MoE kernels build their own routing/permutation from routing logits; these "
                "cases hand over pre-computed sorted slots, so the timed workload would differ"}
    if op.startswith("moe_") or op.startswith("chain_moe"):
        return {"api": None, "note":
                f"'{op}' is a routing/sort/permute/reduce micro-op with no standalone trtllm-gen kernel"}
    if op.startswith("w4a16_") or op.startswith("gguf_") or op in ("lora_rank16", "lora_rank64"):
        return {"api": None, "note": f"no trtllm-gen dense GEMM for '{op}' (Marlin/GGUF/LoRA layouts)"}
    if op in ("mma_bf16", "mma_bf16_tuned", "mma_fp16", "mma_fp16_tuned", "dense_gemv"):
        return {"api": None, "note":
                "bf16/fp16 dense GEMM is not a trtllm-gen target here; FlashInfer mm_bf16 or cuBLAS is "
                "the reference for these cases"}

    return {"api": None, "note": f"no trtllm-gen expression for op '{op}'"}


def _trtllm_op(name: str):
    """Return `torch.ops.trtllm.<name>`, importing the TRT-LLM runtime that registers it."""
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm; needs libnvinfer + the build)
    try:
        return getattr(torch.ops.trtllm, name)
    except AttributeError:
        return None


def _flashinfer_gemm():
    import flashinfer  # noqa: F401  (JIT/CUDA kernels; heavy)
    return getattr(flashinfer, "gemm", None)


def _call_filtered(fn, **kwargs):
    """Pass only the kwargs the installed build declares, so version drift degrades to a skip."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


def _build_scaled_mm(dec: dict, device):
    op = _trtllm_op("cublas_scaled_mm")
    if op is None:
        return None, ("the installed tensorrt_llm does not register trtllm::cublas_scaled_mm "
                      "(needs a build with the cuBLASLt scaled-mm binding)")
    m, n, k = dec["m"], dec["n"], dec["k"]
    dtype = torch.float8_e4m3fn if dec["dtype"] == "float8_e4m3fn" else torch.int8
    a = torch.ones(m, k, device=device, dtype=dtype)
    b = torch.ones(n, k, device=device, dtype=dtype)
    if dec["per_channel"]:
        sa = torch.ones(m, device=device, dtype=torch.float32)
        sb = torch.ones(n, device=device, dtype=torch.float32)
    else:
        sa = torch.ones(1, device=device, dtype=torch.float32)
        sb = torch.ones(1, device=device, dtype=torch.float32)
    # mat_b is [K, N]: TRT-LLM takes it K-major, kops stores B as [N, K] (D = A . B^T).
    return (lambda: op(a, b.t(), sa, sb, None, torch.bfloat16)), dec["note"]


def _build_bf16_fp4(dec: dict, device):
    gemm = _flashinfer_gemm()
    fn = getattr(gemm, "mm_bf16_fp4", None) if gemm is not None else None
    if fn is None:
        return None, ("the installed FlashInfer has no gemm.mm_bf16_fp4; TRT-LLM itself only exposes "
                      "FP4 x FP4 GEMMs, so a bf16-activation W4A16 kernel is unavailable")
    m, n, k = dec["m"], dec["n"], dec["k"]
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    # NVFP4: two e2m1 per byte, one e4m3 scale per 16 elements along K.
    b = torch.zeros(n, k // 2, device=device, dtype=torch.uint8)
    b_scale = torch.ones(n, k // 16, device=device, dtype=torch.float8_e4m3fn)
    global_scale = torch.tensor(1.0, device=device, dtype=torch.float32)

    def run():
        return _call_filtered(fn, a=a, b=b, b_descale=b_scale, b_scale=b_scale,
                              global_scale=global_scale, out_dtype=torch.bfloat16)
    return run, dec["note"]


def _build(dec: dict, device, cc: tuple):
    api = dec["api"]
    if torch is None:
        return None, "torch is required to build trtllm operands"
    if api == "scaled_mm":
        need = (8, 9) if dec["dtype"] == "float8_e4m3fn" else (7, 5)
        if cc < need:
            return None, (f"{dec['dtype']} scaled GEMM needs sm{need[0]}{need[1]}+; "
                          f"device is sm{cc[0]}{cc[1]}")
        return _build_scaled_mm(dec, device)
    if api == "bf16_fp4":
        if cc[0] < 10:
            return None, f"the trtllm-gen bf16 x NVFP4 GEMM is SM100+; device is sm{cc[0]}{cc[1]}"
        return _build_bf16_fp4(dec, device)
    return None, f"internal: no builder for '{api}'"


def adapter(case: dict, device):
    """thirdparty.py adapter entry point."""
    dec = decide(case)
    if dec["api"] is None:
        return None, dec["note"]

    cc = _device_cc(device)
    if cc[0] < 8:
        return None, f"no trtllm-gen kernel for sm{cc[0]}{cc[1]} (SM80+ only)"

    try:
        return _build(dec, device, cc)
    except ImportError as exc:
        return None, f"neither FlashInfer nor TensorRT-LLM is importable: {exc}; {HINT}"
    except Exception as exc:  # noqa: BLE001 - capability probe: unsupported shape/dtype
        return None, f"trtllm build: {type(exc).__name__}: {exc}"


IMPLS["trtllm"] = adapter


# ---------------------------------------------------------------------------
# CLI: the case-id -> API mapping, with or without the packages installed
# ---------------------------------------------------------------------------
def print_rules() -> None:
    print("TensorRT-LLM adapter rules (exact match only; everything else is skipped)")
    print(f"  providers: FlashInfer trtllm_* first, then torch.ops.trtllm (*). {HINT}")
    print("")
    for pattern, api in RULES:
        print(f"  {pattern:<56} -> {api}")
    print("")


def describe(case: dict) -> None:
    dec = decide(case)
    tag = "SKIP " if dec["api"] is None else "MAP  "
    api = "-" if dec["api"] is None else dec["api"]
    print(f"  {tag}{case.get('id', '?'):<48} {api:<12} {dec['note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=None, help="case JSON from `bench_<family> --dump-cases`")
    ap.add_argument("--run", action="store_true", help="benchmark the mapped cases")
    ap.add_argument("--json", default=None, help="result JSON for --run")
    ap.add_argument("--flush-l2", type=int, default=0, metavar="MB")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--iters", type=int, default=0)
    args = ap.parse_args()

    print_rules()
    cases = []
    if args.cases:
        with open(args.cases, "r", encoding="utf-8") as f:
            cases = json.load(f).get("cases", [])
        print(f"case mapping ({args.cases})")
        for case in cases:
            describe(case)
        mapped = sum(1 for c in cases if decide(c)["api"] is not None)
        print(f"\n{mapped}/{len(cases)} cases have a trtllm mapping")

    if not args.run:
        return 0
    if not cases:
        print("--run needs --cases", file=sys.stderr)
        return 2
    if not args.json:
        print("--run needs --json", file=sys.stderr)
        return 2
    if torch is None or not torch.cuda.is_available():
        print("--run needs torch with CUDA", file=sys.stderr)
        return 1

    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from thirdparty import bench  # noqa: E402 - dev harness, imported only for --run

    device = torch.device("cuda", 0)
    results, skipped = [], []
    for case in cases:
        try:
            fn, note = adapter(case, device)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": case["id"], "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if fn is None:
            skipped.append({"id": case["id"], "reason": note or "unsupported"})
            continue
        try:
            st = bench(fn, int(case["warmup"]), args.iters or int(case["iters"]),
                       args.flush_l2, args.rounds)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": case["id"], "reason": f"run: {type(exc).__name__}: {exc}"})
            continue
        ms = st["median_ms"]
        results.append({
            "id": case["id"], "op": case["op"], "dtype": case["dtype"],
            "rows": case["rows"], "cols": case["cols"], "bytes": case["bytes"],
            "flops": case["flops"], **st,
            "gbs": case["bytes"] / (ms * 1e6) if ms > 0 else 0.0,
            "tflops": case["flops"] / (ms * 1e9) if ms > 0 else 0.0,
            "impl": "trtllm", "note": note,
        })
        print(f"{case['id']:<48} {ms:10.4f} ms  {results[-1]['gbs']:9.1f} GB/s  {note}")

    props = torch.cuda.get_device_properties(0)
    out = {
        "schema": 1,
        "backend": "trtllm (thirdparty_trtllm.py)",
        "provenance": {"commit": "thirdparty_trtllm", "dirty": 1, "flags_hash": "",
                       "archs": f"sm{props.major}{props.minor}", "toolkit": torch.version.cuda or ""},
        "device": {"name": torch.cuda.get_device_name(0), "cc": f"{props.major}.{props.minor}",
                   "sms": props.multi_processor_count, "driver": 0, "runtime": 0, "peak_gbs": 0.0},
        "skipped": skipped,
        "results": results,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {args.json}: {len(results)} results, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
