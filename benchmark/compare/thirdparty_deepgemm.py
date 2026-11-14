#!/usr/bin/env python3
"""DeepGEMM adapter for the kernel-ops benchmark cases.

Dev-only, like `thirdparty.py`: not built, linked, tested or shipped. The module
follows the `thirdparty.py` adapter contract

    adapter(case: dict, device: torch.device) -> (fn, note) | (None, reason)

and maps only the cases DeepGEMM expresses *exactly*: same operand dtypes, same
scaling recipe, same layouts. Anything else returns `(None, reason)`; the
harness records the reason instead of aborting the run.

DeepGEMM (3rd/DeepGEMM, v2.8.2) compiles each kernel at runtime through DeepJIT,
so the first call of a shape pays a compile; the harness warm-up iterations
absorb it before the timed region.

    python benchmark/compare/thirdparty_deepgemm.py
    python benchmark/compare/thirdparty_deepgemm.py --cases out/thirdparty/cases/gemm.json
    python benchmark/compare/thirdparty_deepgemm.py --run --cases ... --json out/deepgemm-gemm.json

`--run` reuses the timing protocol of `thirdparty.py` (`bench`), so the JSON has
the same schema with `impl = "deepgemm"`.
"""
from __future__ import annotations

import argparse
import json
import sys

# torch is the benchmark runtime, not the adapter target: importing it is fine,
# but the mapping table must also print on a Python without torch installed.
try:
    import torch
except Exception:  # noqa: BLE001 - a missing torch must not break --cases
    torch = None

IMPLS = {"deepgemm": None}  # filled in after `adapter` is defined

# DeepGEMM compiles for the device it runs on and only has SM90 (wgmma) and
# SM100/SM103 (tcgen05) bodies; there is no mma.sync fallback.
SM_REASON = "deep_gemm requires SM90/SM100 (DeepJIT has no mma.sync body)"
HINT = ("install DeepGEMM from 3rd/DeepGEMM (`./install.sh`; CUDA >= 12.9, "
        "torch >= 2.3, CUTLASS submodule cloned)")

# Case ops the adapter can express. Everything else reports the generic reason.
#   mma_bf16 / mma_bf16_tuned  -> bf16_gemm_nt
#   dense_gemv (bf16, M=1)     -> bf16_gemm_nt (DeepGEMM has no GEMV kernel)
#   fp8_mma_blockwise          -> fp8_gemm_nt, 1D2D (A per-token x B 128x128)
#   fp8_scale_to_tma           -> get_mn_major_tma_aligned_tensor
#   fp8_scale_to_tma_ue8m0     -> get_mn_major_tma_aligned_packed_ue8m0_tensor
RULES = (
    ("mma_bf16, mma_bf16_tuned (bf16)", "deep_gemm.bf16_gemm_nt"),
    ("dense_gemv (bf16, M=1)", "deep_gemm.bf16_gemm_nt at M=1"),
    ("fp8_mma_blockwise (e4m3 x e4m3)", "deep_gemm.fp8_gemm_nt, recipe_a=(1,128), recipe_b=(128,128)"),
    ("fp8_scale_to_tma (fp32)", "deep_gemm.get_mn_major_tma_aligned_tensor"),
    ("fp8_scale_to_tma_ue8m0 (fp32 -> packed int32)",
     "deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor"),
)


def _id_int(case_id: str, key: str, default: int) -> int:
    """Read the first integer after `.<key>` in a case id (mirrors thirdparty.py)."""
    marker = "." + key
    if marker not in case_id:
        return default
    digits = ""
    for ch in case_id.split(marker, 1)[1]:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else default


def _gemm_shape(case: dict):
    """(M, N, K) from the shared case table: rows = N, cols = K, M = flops / (2 N K)."""
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


def _arch_reason(cc: tuple):
    if cc[0] in (9, 10):
        return None
    return f"{SM_REASON}; device is sm{cc[0]}{cc[1]}"


def decide(case: dict) -> dict:
    """Pure mapping decision: no torch allocation, no package import.

    Returns {"api": str | None, "note": str, ...shape fields}. `api is None`
    means the case is skipped and `note` is the reason.
    """
    op = case.get("op", "")
    cid = case.get("id", "")
    dtype = case.get("dtype", "")

    if op in ("mma_bf16", "mma_bf16_tuned"):
        if dtype != "bfloat16":
            return {"api": None, "note": f"DeepGEMM has no {dtype} GEMM (bf16 and fp8/fp4 only)"}
        shape = _gemm_shape(case)
        if shape is None:
            return {"api": None, "note": "unparsed GEMM shape"}
        m, n, k = shape
        note = "deep_gemm.bf16_gemm_nt (D = A @ B^T)"
        if op.endswith("_tuned"):
            note += "; the kops autotuned variant maps to the same call (DeepGEMM picks its own tactic)"
        return {"api": "bf16_gemm_nt", "note": note, "m": m, "n": n, "k": k}

    if op in ("mma_fp16", "mma_fp16_tuned"):
        return {"api": None, "note":
                f"DeepGEMM's dense GEMM is bf16-only (bf16_gemm_nt); this case is {dtype or 'fp16'}"}

    if op == "dense_gemv":
        if dtype != "bfloat16":
            return {"api": None, "note": f"dense_gemv adapter covers bf16 only, case is {dtype}"}
        shape = _gemm_shape(case)
        if shape is None:
            return {"api": None, "note": "unparsed dense_gemv shape"}
        m, n, k = shape
        if m != 1:
            return {"api": None, "note": f"dense_gemv adapter covers M=1, case is M={m}"}
        return {"api": "bf16_gemm_nt", "note":
                "deep_gemm.bf16_gemm_nt at M=1 (DeepGEMM has no GEMV kernel; tensor-core path)",
                "m": m, "n": n, "k": k}

    if op == "fp8_mma_blockwise":
        shape = _gemm_shape(case)
        if shape is None:
            return {"api": None, "note": "unparsed FP8 shape"}
        m, n, k = shape
        if k % 128 or n % 128:
            return {"api": None, "note": f"blockwise recipe needs N,K multiples of 128 (N={n}, K={k})"}
        return {"api": "fp8_gemm_blockwise", "m": m, "n": n, "k": k,
                "note": "deep_gemm.fp8_gemm_nt, 1D2D: A per-token x B 128x128 block scales"}

    if op == "fp8_mma_rowwise":
        return {"api": None, "note":
                "kops 'rowwise' is a per-output-row x per-output-column dequant epilogue; DeepGEMM 1D1D "
                "places the second scale per token along K, so the layout is not the same op"}

    if op == "fp8_mma_tensorwise":
        return {"api": None, "note":
                "DeepGEMM has no per-tensor scaling recipe (1D1D/1D2D group along K); "
                "running it as 1D1D with unit scales would benchmark a different layout"}

    if op in ("fp8_gemv_rowwise", "fp8_gemv_blockwise"):
        return {"api": None, "note": "DeepGEMM has no GEMV kernel for quantized weights"}

    if op.startswith("fp4_") or op in ("w4a16_awq", "w4a16_gptq", "w4a16_awq_marlin"):
        return {"api": None, "note":
                "DeepGEMM's FP4 GEMM needs both operands quantized (fp4 x fp4 or fp8 x fp4); "
                "these cases keep A in bf16 (W4A16)"}

    if op in ("grouped_fp8", "moe_gemv_fp8"):
        return {"api": None, "note":
                "grouped/fused MoE case uses per-channel (rowwise) FP8 weight scales; DeepGEMM grouped "
                "GEMMs are per-token 1D1D/1D2D, so the recipe is not the same op"}

    if op == "fp8_block_quant" or op == "fp8_block_quant_ue8m0":
        return {"api": None, "note":
                "deep_gemm.utils.per_block_cast_to_fp8 is a PyTorch helper, not a kernel; timing it "
                "would report torch, not DeepGEMM"}

    if op == "block_scale_swizzle":
        return {"api": None, "note":
                "kops swizzles a single scale plane; DeepGEMM only exposes the full SF transform "
                "(transform_sf_into_required_layout), which is not the same op"}

    if op == "fp8_scale_to_tma":
        rows, cols = int(case["rows"]), int(case["cols"])
        if cols % 4:
            return {"api": None, "note": f"TMA-aligned MN-major layout needs K % 4 == 0 (K={cols})"}
        return {"api": "sf_to_tma", "mn": rows, "k": cols,
                "note": "deep_gemm.get_mn_major_tma_aligned_tensor (fp32, no packing); "
                        "DeepGEMM returns a fresh tensor, so the call allocates from the torch cache"}

    if op == "fp8_scale_to_tma_ue8m0":
        rows, cols = int(case["rows"]), int(case["cols"])
        if cols % 4:
            return {"api": None, "note": f"UE8M0 packing needs K % 4 == 0 (K={cols})"}
        return {"api": "sf_to_tma_ue8m0", "mn": rows, "k": cols,
                "note": "deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor; the input is rounded "
                        "to UE8M0 once outside the timed region (the kernel requires mantissa-zero scales)"}

    if cid.startswith("gemm.gguf_") or op.startswith("gguf_"):
        return {"api": None, "note": "DeepGEMM has no GGUF (k-quant block) weight format"}
    if op.startswith("lora_"):
        return {"api": None, "note": "DeepGEMM has no multi-LoRA SGMV kernel"}
    if op == "int8_mma":
        return {"api": None, "note": "DeepGEMM has no INT8 (W8A8) GEMM: FP8, FP4 and BF16 only"}
    if op in ("w4a8_g128", "w4a8_chn"):
        return {"api": None, "note": "DeepGEMM has no QServe-style W4A8 kernel"}
    if op.startswith("bmm_"):
        return {"api": None, "note": "no batched GEMM in DeepGEMM (MLA absorption bmm has no API)"}
    if op in ("dual_bf16", "scaled_gemm"):
        return {"api": None, "note": f"DeepGEMM has no fused-GLU or generic scaled_gemm entry for '{op}'"}
    if op.startswith("machete") or op in ("grouped_prepare", "weight_transpose", "glu_interleave",
                                          "router_split", "router_gemm", "gemv_argmax",
                                          "int8_colsum"):
        return {"api": None, "note":
                f"'{op}' is a load-time or model-specific prep kernel with no DeepGEMM API"}
    return {"api": None, "note": f"no DeepGEMM expression for op '{op}'"}


def _load():
    import deep_gemm  # noqa: F401  (DeepJIT + C++ extension; heavy)
    return deep_gemm


def _build(dec: dict, deep_gemm, device, cc: tuple):
    api = dec["api"]
    if torch is None:
        return None, "torch is required to build DeepGEMM operands"

    if api == "bf16_gemm_nt":
        m, n, k = dec["m"], dec["n"], dec["k"]
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        d = torch.empty(m, n, device=device, dtype=torch.bfloat16)
        return (lambda: deep_gemm.bf16_gemm_nt(a, b, d)), dec["note"]

    if api == "fp8_gemm_blockwise":
        m, n, k = dec["m"], dec["n"], dec["k"]
        # SM100 recipes carry packed UE8M0 scale factors; SM90 carries fp32.
        ue8m0 = cc[0] == 10
        from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
        a_fp8, sfa = per_token_cast_to_fp8(
            torch.randn(m, k, device=device, dtype=torch.bfloat16), use_ue8m0=ue8m0, gran_k=128)
        b_fp8, sfb = per_block_cast_to_fp8(
            torch.randn(n, k, device=device, dtype=torch.bfloat16), use_ue8m0=ue8m0, gran_k=128)
        d = torch.empty(m, n, device=device, dtype=torch.bfloat16)

        def run():
            deep_gemm.fp8_gemm_nt((a_fp8, sfa), (b_fp8, sfb), d,
                                  recipe_a=(1, 128), recipe_b=(128, 128))
        return run, dec["note"]

    if api in ("sf_to_tma", "sf_to_tma_ue8m0"):
        mn, k = dec["mn"], dec["k"]
        x = torch.randn(mn, k, device=device, dtype=torch.float32)
        if api == "sf_to_tma":
            return (lambda: deep_gemm.get_mn_major_tma_aligned_tensor(x)), dec["note"]
        from deep_gemm.utils import ceil_to_ue8m0
        x_ue8m0 = ceil_to_ue8m0(x)  # untimed: the pack kernel asserts mantissa-zero scales
        return (lambda: deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(x_ue8m0)), dec["note"]

    return None, f"internal: no builder for '{api}'"


def adapter(case: dict, device):
    """thirdparty.py adapter entry point."""
    dec = decide(case)
    if dec["api"] is None:
        return None, dec["note"]

    cc = _device_cc(device)
    arch = _arch_reason(cc)
    if arch is not None:
        return None, arch

    try:
        deep_gemm = _load()
    except Exception as exc:  # ImportError, missing .so, no CUDA driver, ...
        return None, f"deep_gemm not importable: {type(exc).__name__}: {exc}; {HINT}"

    try:
        return _build(dec, deep_gemm, device, cc)
    except Exception as exc:  # noqa: BLE001 - capability probe: unsupported shape/dtype
        return None, f"deep_gemm build: {type(exc).__name__}: {exc}"


IMPLS["deepgemm"] = adapter


# ---------------------------------------------------------------------------
# CLI: the case-id -> API mapping, with or without DeepGEMM installed
# ---------------------------------------------------------------------------
def print_rules() -> None:
    print("DeepGEMM adapter rules (exact match only; everything else is skipped)")
    print(f"  hardware: SM90/SM100/SM103 required ({HINT})")
    print("")
    for pattern, api in RULES:
        print(f"  {pattern:<52} -> {api}")
    print("")


def describe(case: dict) -> None:
    dec = decide(case)
    tag = "SKIP " if dec["api"] is None else "MAP  "
    api = "-" if dec["api"] is None else dec["api"]
    print(f"  {tag}{case.get('id', '?'):<48} {api:<24} {dec['note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=None, help="case JSON from `bench_<family> --dump-cases`")
    ap.add_argument("--run", action="store_true", help="benchmark the mapped cases (needs DeepGEMM)")
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
        print(f"\n{mapped}/{len(cases)} cases have a DeepGEMM mapping")

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

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
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
            "impl": "deepgemm", "note": note,
        })
        print(f"{case['id']:<48} {ms:10.4f} ms  {results[-1]['gbs']:9.1f} GB/s  {note}")

    props = torch.cuda.get_device_properties(0)
    out = {
        "schema": 1,
        "backend": "deepgemm (thirdparty_deepgemm.py)",
        "provenance": {"commit": "thirdparty_deepgemm", "dirty": 1, "flags_hash": "",
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
