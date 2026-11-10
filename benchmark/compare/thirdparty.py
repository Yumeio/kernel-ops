#!/usr/bin/env python3
"""Third-party adapters for the kernel-ops benchmark cases.

Dev-only: not built, linked, tested or shipped. The script reads the same
`--dump-cases` JSON as `torch_baseline.py`, runs one library implementation per
case with the same timing protocol (warmup, optional L2 flush, grouped or
per-launch CUDA events), and writes the shared result schema plus an `impl`
field. Cases a library cannot express are skipped with a reason instead of
aborting the run.

    python benchmark/compare/thirdparty.py --cases out/cases.json \
        --impl vllm --json out/vllm.json --flush-l2 256 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import torch
import torch.nn.functional as F

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "int32": torch.int32,
    "uint32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "float8_e4m3fn": torch.float8_e4m3fn,
}
EPS = 1e-5


def _id_int(case_id: str, key: str, default: int) -> int:
    marker = "." + key
    if marker not in case_id:
        return default
    tail = case_id.split(marker, 1)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else default


def dtype_of(case: dict) -> torch.dtype:
    return DTYPES.get(case["dtype"], torch.bfloat16)


def _pair(case_id: str, key: str, default=(1, 1)):
    m = re.search(r"\.h(\d+)x(\d+)", case_id) if key == "h" else None
    if m:
        return int(m.group(1)), int(m.group(2))
    return default


# ---------------------------------------------------------------------------
# vLLM adapters
# ---------------------------------------------------------------------------
def vllm_adapter(case: dict, device: torch.device):
    import vllm._custom_ops as ops

    op = case["op"]
    cid = case["id"]
    rows, cols = int(case["rows"]), int(case["cols"])
    dt = dtype_of(case)

    if op == "act_and_mul" and hasattr(torch.ops._C, "silu_and_mul"):
        x = torch.randn(rows, 2 * cols, device=device, dtype=dt)
        out = torch.empty(rows, cols, device=device, dtype=dt)
        return lambda: torch.ops._C.silu_and_mul(out, x), "vllm silu_and_mul"

    if op == "quantize_per_group" and hasattr(torch.ops._C, "per_token_group_fp8_quant"):
        x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
        q = torch.empty(rows, cols, device=device, dtype=torch.float8_e4m3fn)
        s = torch.empty(rows, cols // 128, device=device, dtype=torch.float32)
        return (lambda: torch.ops._C.per_token_group_fp8_quant(
            x, q, s, 128, 1e-10, -448.0, 448.0, False, False, False),
            "vllm per_token_group_fp8_quant")

    if op == "rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        out = torch.empty_like(x)
        return lambda: ops.rms_norm(out, x, w, EPS), "vllm rms_norm"

    if op == "fused_add_rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt) * 0.1
        res = torch.randn(rows, cols, device=device, dtype=dt) * 0.1
        w = torch.ones(cols, device=device, dtype=dt)
        return lambda: ops.fused_add_rms_norm(x, res, w, EPS), "vllm fused_add_rms_norm"

    if op == "apply_rotary":
        hq, hkv = _pair(cid, "h", (32, 8))
        tokens, d = rows, cols
        q = torch.randn(tokens, hq * d, device=device, dtype=dt)
        k = torch.randn(tokens, hkv * d, device=device, dtype=dt)
        pos = (torch.arange(tokens, device=device) % 4096).to(torch.int64)
        cache = torch.randn(4096, d, device=device, dtype=torch.float32)
        return lambda: ops.rotary_embedding(pos, q, k, d, cache, True), "vllm rotary_embedding"

    if op == "quantize_static":
        x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
        scale = torch.tensor([1.0], device=device)
        return lambda: ops.scaled_fp8_quant(x, scale), "vllm static_scaled_fp8_quant"

    if op == "quantize_per_token":
        x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
        return (lambda: ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True),
                "vllm dynamic_per_token_scaled_fp8_quant")

    if op == "reshape_and_cache":
        hq, hkv = _pair(cid, "h", (4, 4))
        tokens, d = rows, cols
        page = 16
        blocks = (tokens + page - 1) // page
        k = torch.randn(tokens, hkv, d, device=device, dtype=dt)
        v = torch.randn(tokens, hkv, d, device=device, dtype=dt)
        kc = torch.empty(blocks, page, hkv, d, device=device, dtype=dt)
        vc = torch.empty_like(kc)
        slots = torch.arange(tokens, device=device, dtype=torch.int64)
        ks = torch.tensor(1.0, device=device)
        vs = torch.tensor(1.0, device=device)
        return (lambda: ops.reshape_and_cache_flash(k, v, kc, vc, slots, "auto", ks, vs),
                "vllm reshape_and_cache_flash")

    if op == "gptq_to_marlin":
        n, k = rows, cols
        q = torch.randint(-2**31, 2**31 - 1, (k // 8, n), device=device, dtype=torch.int32)
        return (lambda: ops.gptq_marlin_repack(q, k, n, 4),
                "vllm gptq_marlin_repack")

    if op == "awq_to_marlin":
        n, k = rows, cols
        q = torch.randint(-2**31, 2**31 - 1, (k, n // 8), device=device, dtype=torch.int32)
        return (lambda: ops.awq_marlin_repack(q, k, n, 4),
                "vllm awq_marlin_repack")

    if op == "moe_route":
        experts, k = cols, 8
        gating = torch.randn(rows, experts, device=device, dtype=torch.float32)
        w = torch.empty(rows, k, device=device, dtype=torch.float32)
        ids = torch.empty(rows, k, device=device, dtype=torch.int32)
        idx = torch.empty(rows, k, device=device, dtype=torch.int32)
        return (lambda: ops.topk_softmax(w, ids, idx, gating, renormalize=True),
                "vllm topk_softmax (softmax, not kops sigmoid+bias)")

    if op == "moe_align":
        block = 128
        experts = 256
        tokens = max(1, rows)
        ids = torch.randint(0, experts, (tokens, 1), device=device, dtype=torch.int32)
        pad = tokens + experts * block + block
        sorted_ids = torch.empty(pad, device=device, dtype=torch.int32)
        expert_ids = torch.empty(pad // block + experts + 1, device=device, dtype=torch.int32)
        num_post = torch.empty(1, device=device, dtype=torch.int32)
        return (lambda: ops.moe_align_block_size(ids, experts, block, sorted_ids, expert_ids, num_post),
                "vllm moe_align_block_size")

    if op == "moe_sum":
        k = 8
        x = torch.randn(rows, k, cols, device=device, dtype=torch.bfloat16)
        out = torch.empty(rows, cols, device=device, dtype=torch.bfloat16)
        return lambda: ops.moe_sum(x, out), "vllm moe_sum"

    return None, None


# ---------------------------------------------------------------------------
# FlashInfer adapters
# ---------------------------------------------------------------------------
def flashinfer_adapter(case: dict, device: torch.device):
    import flashinfer

    op = case["op"]
    cid = case["id"]
    rows, cols = int(case["rows"]), int(case["cols"])
    dt = dtype_of(case)

    if op == "rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        out = torch.empty_like(x)
        return lambda: flashinfer.norm.rmsnorm(x, w, EPS, out=out), "flashinfer rmsnorm"

    if op == "fused_add_rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt) * 0.1
        res = torch.randn(rows, cols, device=device, dtype=dt) * 0.1
        w = torch.ones(cols, device=device, dtype=dt)
        return (lambda: flashinfer.norm.fused_add_rmsnorm(x, res, w, EPS),
                "flashinfer fused_add_rmsnorm")

    if op == "act_and_mul":
        x = torch.randn(rows, 2 * cols, device=device, dtype=dt)
        out = torch.empty(rows, cols, device=device, dtype=dt)
        return lambda: flashinfer.activation.silu_and_mul(x, out=out), "flashinfer silu_and_mul"

    if op == "softmax":
        x = torch.randn(rows, cols, device=device, dtype=torch.float32)
        return lambda: flashinfer.sampling.softmax(x, temperature=1.0), "flashinfer softmax"

    if op == "topk_mask":
        x = torch.randn(rows, cols, device=device, dtype=torch.float32)
        return lambda: flashinfer.sampling.top_k_mask_logits(x, 50), "flashinfer top_k_mask_logits"

    if op == "append_paged_kv":
        hq, hkv = _pair(cid, "h", (4, 4))
        tokens, d = rows, cols
        page = 16
        pages = (tokens + page - 1) // page
        key = torch.randn(tokens, hkv, d, device=device, dtype=dt)
        val = torch.randn(tokens, hkv, d, device=device, dtype=dt)
        kc = torch.empty(pages, page, hkv, d, device=device, dtype=dt)
        vc = torch.empty_like(kc)
        batch = torch.zeros(tokens, device=device, dtype=torch.int32)
        pos = torch.arange(tokens, device=device, dtype=torch.int32)
        kv_indices = torch.arange(pages, device=device, dtype=torch.int32)
        kv_indptr = torch.tensor([0, pages], device=device, dtype=torch.int32)
        last = torch.tensor([page], device=device, dtype=torch.int32)
        return (lambda: flashinfer.page.append_paged_kv_cache(
            key, val, batch, pos, (kc, vc), kv_indices, kv_indptr, last, "NHD"),
            "flashinfer append_paged_kv_cache")

    if op == "attention":
        parsed = parse_attention(cid, rows, cols)
        if parsed is None:
            return None, "attention shape not supported by the flashinfer adapter"
        b, q, kv, hq, hkv, d, causal, softcap, window, fp8_kv, skew = parsed
        if fp8_kv:
            return None, "FP8 KV path not enabled in this adapter"
        if cid.startswith("decode") and q == 1:
            lens = skew if skew is not None else [kv] * b
            page = 16
            pages = [(n + page - 1) // page for n in lens]
            indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(pages), 0).tolist()),
                                  device=device, dtype=torch.int32)
            total_pages = int(indptr[-1])
            indices = torch.arange(total_pages, device=device, dtype=torch.int32)
            last = torch.tensor([((n - 1) % page) + 1 for n in lens],
                                device=device, dtype=torch.int32)
            paged = torch.randn(total_pages, 2, page, hkv, d, device=device,
                                dtype=torch.bfloat16)
            qq = torch.randn(b, hq, d, device=device, dtype=torch.bfloat16)
            out = torch.empty(b, hq, d, device=device, dtype=torch.bfloat16)
            workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD", use_tensor_cores=("tc" in cid))
            wrapper.plan(indptr, indices, last, hq, hkv, d, page,
                         q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                         pos_encoding_mode="NONE")
            return (lambda: wrapper.run(qq, paged, out=out),
                    "flashinfer batch_decode paged KV")
        if b != 1 or (skew is not None and len(set(skew)) > 1):
            return None, "flashinfer prefill adapter covers single-request attention only"
        qq = torch.randn(q, hq, d, device=device, dtype=torch.bfloat16)
        kk = torch.randn(kv, hkv, d, device=device, dtype=torch.bfloat16)
        vv = torch.randn(kv, hkv, d, device=device, dtype=torch.bfloat16)
        if q == 1:
            return (lambda: flashinfer.decode.single_decode_with_kv_cache(
                qq[0], kk, vv, window_left=window, logits_soft_cap=softcap),
                "flashinfer single_decode (dense KV)")
        return (lambda: flashinfer.prefill.single_prefill_with_kv_cache(
            qq, kk, vv, causal=causal, window_left=window, logits_soft_cap=softcap),
            "flashinfer single_prefill (dense KV)")

    return None, None


# ---------------------------------------------------------------------------
# PyTorch SDPA adapter (B8)
# ---------------------------------------------------------------------------
def parse_attention(cid: str, rows: int, cols: int):
    """Return (batch, q_len, kv_len, Hq, Hkv, D, causal, softcap, window, fp8_kv, kv_lens)."""
    d = cols
    m = re.search(r"\.h(\d+)x(\d+)", cid)
    if not m:
        return None
    hq, hkv = int(m.group(1)), int(m.group(2))
    causal = "causal" in cid or "decode" in cid
    softcap = 50.0 if "softcap" in cid else None
    window = 4096 if "swa" in cid else -1
    fp8_kv = "e4m3" in cid
    skew = None
    if "skew" in cid:
        skew = [32768] + [1024] * 31
    if cid.startswith("prefill"):
        m = re.search(r"\.b(\d+)(?=\.|$)", cid)
        b = int(m.group(1)) if m else 1
        if "paged" in cid:
            q = _id_int(cid, "chunk", 512)
            kv = _id_int(cid, "kv", 4096)
        else:
            q = _id_int(cid, "s", rows)
            kv = q
        return b, q, kv, hq, hkv, d, causal, softcap, window, fp8_kv, skew
    m = re.search(r"\.b(\d+)(?=\.|$)", cid)
    b = int(m.group(1)) if m else 1
    if skew is not None:
        b = len(skew)
    q = 4 if cid.endswith(".q4") else 1
    kv = _id_int(cid, "kv", 2048)
    return b, q, kv, hq, hkv, d, causal, softcap, window, fp8_kv, skew


def sdpa_adapter(case: dict, device: torch.device):
    if case["op"] != "attention":
        return None, None
    if case["id"].startswith("decode"):
        return None, "SDPA decode is a math fallback; use the FlashInfer batch-decode adapter"
    parsed = parse_attention(case["id"], int(case["rows"]), int(case["cols"]))
    if parsed is None:
        return None, "unparsed attention case"
    b, q, kv, hq, hkv, d, causal, softcap, window, fp8_kv, skew = parsed
    if softcap is not None or window > 0:
        return None, "SDPA adapter does not model softcap/window"
    if fp8_kv:
        return None, "SDPA adapter does not model FP8 KV"
    if skew is not None:
        qq = torch.randn(1, q, hq, d, device=device, dtype=torch.bfloat16)
        qh = qq.permute(0, 2, 1, 3)
        parts = []
        for n in skew:
            kk = torch.randn(1, n, hkv, d, device=device, dtype=torch.bfloat16)
            vv = torch.randn(1, n, hkv, d, device=device, dtype=torch.bfloat16)
            parts.append((kk.permute(0, 2, 1, 3), vv.permute(0, 2, 1, 3)))

        def run_skew():
            out = 0
            for kk, vv in parts:
                out = out + F.scaled_dot_product_attention(
                    qh, kk, vv, is_causal=False, enable_gqa=(hq != hkv))[:, -1:]
            return out

        return run_skew, "torch SDPA, per-request skew loop (dense KV)"

    qq = torch.randn(b, q, hq, d, device=device, dtype=torch.bfloat16)
    kk = torch.randn(b, kv, hkv, d, device=device, dtype=torch.bfloat16)
    vv = torch.randn(b, kv, hkv, d, device=device, dtype=torch.bfloat16)
    # SDPA takes [batch, heads, sequence, head_dim]; the case table is
    # [batch, sequence, heads, head_dim].
    qh = qq.permute(0, 2, 1, 3)
    kh = kk.permute(0, 2, 1, 3)
    vh = vv.permute(0, 2, 1, 3)
    # Decode reads every KV position; prefill uses the requested mask.
    is_causal = bool(causal and q > 1)
    return (lambda: F.scaled_dot_product_attention(
        qh, kh, vh, is_causal=is_causal, enable_gqa=(hq != hkv)).permute(0, 2, 1, 3),
        "torch SDPA (dense KV, paged case is unpaged)")


def bnb_adapter(case: dict, device: torch.device):
    import bitsandbytes as bnb

    if case["op"] != "nf4_lut":
        return None, None
    n = int(case["rows"])
    k = int(case["cols"])
    m = int(case["flops"]) // (2 * n * k)
    if m != 1:
        return None, "bitsandbytes gemv_4bit adapter covers M=1 only"
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    packed, state = bnb.functional.quantize_nf4(w, blocksize=64, compress_statistics=False)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    return (lambda: bnb.functional.gemv_4bit(x, packed, state=state),
            "bitsandbytes gemv_4bit NF4")


def fa2_adapter(case: dict, device: torch.device):
    from flash_attn import flash_attn_func

    if case["op"] != "attention":
        return None, None
    cid = case["id"]
    if cid.startswith("decode"):
        return None, "FA2 adapter covers dense prefill; decode uses FlashInfer"
    parsed = parse_attention(cid, int(case["rows"]), int(case["cols"]))
    if parsed is None:
        return None, "unparsed attention case"
    b, q, kv, hq, hkv, d, causal, softcap, window, fp8_kv, skew = parsed
    if fp8_kv:
        return None, "FA2 adapter does not model FP8 KV"
    if "paged" in cid:
        return None, "FA2 adapter uses dense KV; paged case not modeled"
    qq = torch.randn(b, q, hq, d, device=device, dtype=torch.bfloat16)
    kk = torch.randn(b, kv, hkv, d, device=device, dtype=torch.bfloat16)
    vv = torch.randn(b, kv, hkv, d, device=device, dtype=torch.bfloat16)
    import inspect
    params = inspect.signature(flash_attn_func).parameters
    kwargs = {"causal": causal}
    if "window_size" in params and window and window > 0:
        kwargs["window_size"] = (window, 0)
    if "softcap" in params and softcap:
        kwargs["softcap"] = softcap
    return (lambda: flash_attn_func(qq, kk, vv, **kwargs),
            "flash-attn FA2 (dense prefill)")


def sglang_adapter(case: dict, device: torch.device):
    import sgl_kernel

    op = case["op"]
    cid = case["id"]
    rows, cols = int(case["rows"]), int(case["cols"])
    dt = dtype_of(case)

    if op == "rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        out = torch.empty_like(x)
        return lambda: sgl_kernel.rmsnorm(x, w, EPS, out=out), "sgl_kernel rmsnorm"

    if op == "fused_add_rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt) * 0.1
        res = torch.randn(rows, cols, device=device, dtype=dt) * 0.1
        w = torch.ones(cols, device=device, dtype=dt)
        return (lambda: sgl_kernel.fused_add_rmsnorm(x, res, w, EPS),
                "sgl_kernel fused_add_rmsnorm")

    if op == "act_and_mul":
        x = torch.randn(rows, 2 * cols, device=device, dtype=dt)
        out = torch.empty(rows, cols, device=device, dtype=dt)
        return lambda: sgl_kernel.silu_and_mul(x, out=out), "sgl_kernel silu_and_mul"

    if op == "apply_rotary":
        hq, hkv = _pair(cid, "h", (32, 8))
        tokens, d = rows, cols
        q = torch.randn(tokens, hq * d, device=device, dtype=dt)
        k = torch.randn(tokens, hkv * d, device=device, dtype=dt)
        pos = (torch.arange(tokens, device=device) % 4096).to(torch.int64)
        cache = torch.randn(4096, d, device=device, dtype=torch.float32)
        return (lambda: sgl_kernel.rotary_embedding(pos, q, k, d, cache, True),
                "sgl_kernel rotary_embedding")

    if op == "moe_route":
        experts, k = cols, 8
        gating = torch.randn(rows, experts, device=device, dtype=torch.float32)
        w = torch.empty(rows, k, device=device, dtype=torch.float32)
        ids = torch.empty(rows, k, device=device, dtype=torch.int32)
        return (lambda: sgl_kernel.topk_softmax(w, ids, gating, renormalize=True),
                "sgl_kernel topk_softmax")

    if op == "moe_align":
        block = 128
        experts = 256
        tokens = max(1, rows)
        ids = torch.randint(0, experts, (tokens, 1), device=device, dtype=torch.int32)
        pad = tokens + experts * block + block
        sorted_ids = torch.empty(pad, device=device, dtype=torch.int32)
        expert_ids = torch.empty(pad // block + experts + 1, device=device,
                                 dtype=torch.int32)
        num_post = torch.empty(1, device=device, dtype=torch.int32)
        cumsum = torch.empty(experts + 1, device=device, dtype=torch.int32)
        return (lambda: sgl_kernel.moe_align_block_size(
            ids, experts, block, sorted_ids, expert_ids, num_post, cumsum),
            "sgl_kernel moe_align_block_size")

    return None, None


IMPLS = {
    "vllm": vllm_adapter,
    "flashinfer": flashinfer_adapter,
    "sdpa": sdpa_adapter,
    "bnb": bnb_adapter,
    "fa2": fa2_adapter,
    "sglang": sglang_adapter,
}


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def bench(fn, warmup: int, iters: int, flush_mb: int, rounds: int) -> dict:
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
    if flush is not None:
        for _ in range(rounds):
            for _ in range(iters):
                flush[0].copy_(flush[1])
                start.record()
                fn()
                stop.record()
                stop.synchronize()
                samples.append(start.elapsed_time(stop))
    else:
        for _ in range(rounds):
            for _ in range(iters):
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


def matches(case_id: str, csv: str | None) -> bool:
    if not csv:
        return False
    return any(tok.strip() and tok.strip() in case_id for tok in csv.split(","))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--impl", required=True, choices=sorted(IMPLS))
    ap.add_argument("--json", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--flush-l2", type=int, default=0, metavar="MB")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--iters", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("thirdparty: CUDA is not available", file=sys.stderr)
        return 1
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    with open(args.cases, "r", encoding="utf-8") as f:
        spec = json.load(f)
    cases = spec["cases"]
    if args.only:
        cases = [c for c in cases if matches(c["id"], args.only)]
    if args.skip:
        cases = [c for c in cases if not matches(c["id"], args.skip)]

    adapter = IMPLS[args.impl]
    results, skipped = [], []
    for case in cases:
        try:
            fn, note = adapter(case, device)
        except Exception as exc:  # capability probe: unsupported shape / dtype
            skipped.append({"id": case["id"], "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if fn is None:
            skipped.append({"id": case["id"], "reason": note or "unsupported"})
            continue
        try:
            st = bench(fn, int(case["warmup"]), args.iters or int(case["iters"]),
                       args.flush_l2, args.rounds)
        except Exception as exc:
            skipped.append({"id": case["id"], "reason": f"run: {type(exc).__name__}: {exc}"})
            continue
        ms = st["median_ms"]
        results.append({
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
            "impl": args.impl,
            "note": note,
        })
        print(f"{case['id']:<48} {ms:10.4f} ms  {results[-1]['gbs']:9.1f} GB/s  {note}")

    props = torch.cuda.get_device_properties(0)
    out = {
        "schema": 1,
        "backend": f"{args.impl} (thirdparty.py)",
        "provenance": {
            "commit": "thirdparty",
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
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {args.json}: {len(results)} results, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
