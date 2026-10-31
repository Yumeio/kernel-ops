#!/usr/bin/env python3
"""PyTorch baselines for the kernel-ops benchmark cases.

Dev-only: this script is not built, linked, tested, or shipped. It reads the
case table from `bench_<family> --dump-cases` and writes results in the same
JSON schema as the C++ benchmark, so `compare_results.py` can diff them.

    python benchmark/compare/torch_baseline.py \
        --cases out/cases.json --json out/torch.json
"""
from __future__ import annotations

import argparse
import json
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
}
EPS = 1e-5
STYLE = "builtin"


def rms_norm_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if STYLE == "builtin" and hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), w, EPS)
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (y * w.float()).to(x.dtype)


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


def _hf_rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps)
    return (y * w.float()).to(x.dtype)


def _hf_layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                   eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + eps)
    return (y * w.float() + b.float()).to(x.dtype)


def _rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, rot_dim: int):
    half = rot_dim // 2
    inv = 1.0 / (1.0e6 ** (torch.arange(0, half, device=q.device, dtype=torch.float32) / half))
    ang = positions.float()[:, None] * inv[None, :]
    cos = torch.cos(ang).to(q.dtype)
    sin = torch.sin(ang).to(q.dtype)
    cos = torch.cat([cos, cos], dim=-1)[:, None, :]
    sin = torch.cat([sin, sin], dim=-1)[:, None, :]

    def rot(x):
        x1 = x[..., :half]
        x2 = x[..., half:rot_dim]
        return torch.cat([-x2, x1], dim=-1)

    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def make_chain(case: dict, op: str, device: torch.device):
    dt = DTYPES[case["dtype"]]
    rows = int(case["rows"])
    cols = int(case["cols"])
    cid = case["id"]

    if op == "chain_pre_norm_residual":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        res = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)

        def run():
            x.add_(res)
            return _hf_rms(x, w)

        return run

    if op == "chain_pre_norm_glu":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        res = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        gate_up = torch.randn(rows, 2 * cols, device=device, dtype=dt)

        def run():
            x.add_(res)
            y = _hf_rms(x, w)
            return F.silu(gate_up[:, :cols]) * gate_up[:, cols:]

        return run

    if op == "chain_pre_norm_quant":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        res = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        def run_nq():
            x.add_(res)
            y = _hf_rms(x, w)
            s = torch.clamp(y.float().abs().amax(-1, keepdim=True) / 448.0, min=1.0 / (448.0 * 512.0))
            return (y.float() / s).to(torch.float8_e4m3fn)
        return run_nq

    if op == "chain_glu_quant":
        g = torch.randn(rows, 2 * cols, device=device, dtype=dt)
        def run_gq():
            y = F.silu(g[:, :cols]) * g[:, cols:]
            s = torch.clamp(y.float().abs().amax(-1, keepdim=True) / 448.0, min=1.0 / (448.0 * 512.0))
            return (y.float() / s).to(torch.float8_e4m3fn)
        return run_gq

    if op == "chain_qk_norm_rope":
        hq = _id_int(cid, "hq", 32)
        hkv = _id_int(cid, "hkv", 8)
        d = cols
        q = torch.randn(rows, hq, d, device=device, dtype=dt)
        k = torch.randn(rows, hkv, d, device=device, dtype=dt)
        qw = torch.ones(d, device=device, dtype=dt)
        kw = torch.ones(d, device=device, dtype=dt)
        pos = torch.arange(rows, device=device, dtype=torch.int64)

        def run():
            qn = _hf_rms(q, qw, 1e-6)
            kn = _hf_rms(k, kw, 1e-6)
            return _rope(qn, kn, pos, d)

        return run

    if op == "chain_embedding_norm":
        vocab = 32000
        tok_w = torch.randn(vocab, cols, device=device, dtype=dt)
        pos_w = torch.randn(1024, cols, device=device, dtype=dt)
        ids = torch.randint(0, vocab, (rows,), device=device)
        pos = torch.arange(rows, device=device) % 1024
        nw = torch.ones(cols, device=device, dtype=dt)

        def run():
            x = F.embedding(ids, tok_w) + F.embedding(pos, pos_w)
            return _hf_rms(x, nw, 1e-6)

        return run

    if op == "chain_logits_pipeline":
        b, v = rows, cols
        raw = torch.randn(b, v, device=device, dtype=dt)
        temp = torch.ones(b, device=device, dtype=torch.float32)
        counts = torch.zeros(b, v, device=device, dtype=torch.int32)
        prompt = torch.zeros(b, v, device=device, dtype=torch.bool)
        allowed = torch.ones(b, v, device=device, dtype=torch.bool)
        rep = torch.full((b,), 1.1, device=device, dtype=torch.float32)
        freq = torch.full((b,), 0.1, device=device, dtype=torch.float32)
        pres = torch.full((b,), 0.1, device=device, dtype=torch.float32)

        def run():
            x = raw.float() * 1.0
            x = 30.0 * torch.tanh(x / 30.0)
            seen = (counts > 0) | prompt
            x = torch.where(seen, torch.where(x > 0, x / rep[:, None], x * rep[:, None]), x)
            x = x - freq[:, None] * counts.float() - pres[:, None] * seen.float()
            x = x.masked_fill(~allowed, float("-inf"))
            probs = torch.softmax(x / temp[:, None], dim=-1)
            return torch.argmax(probs, dim=-1)

        return run

    if op == "chain_splitkv_merge":
        heads = 32
        splits = 8
        batches = rows
        d = cols
        o_partial = torch.randn(batches * splits, heads, d, device=device, dtype=torch.float32)
        lse_partial = torch.randn(batches * splits, heads, device=device, dtype=torch.float32)
        sink = torch.randn(heads, device=device, dtype=torch.float32)

        def run():
            op_ = o_partial.view(batches, splits, heads, d)
            lp = lse_partial.view(batches, splits, heads)
            m = torch.maximum(lp.max(dim=1, keepdim=True).values, sink[None, None, :])
            w = torch.exp(lp - m)
            den = w.sum(dim=1) + torch.exp(sink[None, :] - m[:, 0, :])
            out = (w[..., None] * op_).sum(dim=1) / den[..., None]
            return out, (m[:, 0, :] + torch.log(den))

        return run

    raise SystemExit(f"unsupported chain op: {op}")


def make_op(case: dict, device: torch.device):
    op = case["op"]
    dt = DTYPES[case["dtype"]]
    rows = int(case["rows"])
    cols = int(case["cols"])

    if op == "norm_quant":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        res = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        def run_nq():
            x.add_(res)
            y = _hf_rms(x, w)
            s = torch.clamp(y.float().abs().amax(-1, keepdim=True) / 448.0,
                            min=1.0 / (448.0 * 512.0))
            return (y.float() / s).to(torch.float8_e4m3fn)
        return run_nq

    if op == "act_and_mul_quant":
        x = torch.randn(rows, 2 * cols, device=device, dtype=dt)
        def run_aq():
            y = F.silu(x[:, :cols]) * x[:, cols:]
            s = torch.clamp(y.float().abs().amax(-1, keepdim=True) / 448.0,
                            min=1.0 / (448.0 * 512.0))
            return (y.float() / s).to(torch.float8_e4m3fn)
        return run_aq

    if op == "rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        return lambda: rms_norm_ref(x, w)

    if op == "fused_add_rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        res = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)

        def run_fused():
            x.add_(res)
            return rms_norm_ref(x, w)

        return run_fused

    if op == "activation":
        x = torch.randn(1, cols, device=device, dtype=dt)
        if "gelu" in case["id"]:
            return lambda: F.gelu(x)
        return lambda: F.silu(x)

    if op == "act_and_mul":
        x = torch.randn(rows, 2 * cols, device=device, dtype=dt)

        def run_mul():
            gate = x[:, :cols]
            up = x[:, cols:]
            return F.silu(gate) * up

        return run_mul

    if op == "residual_add":
        a = torch.randn(rows, cols, device=device, dtype=dt)
        b = torch.randn(rows, cols, device=device, dtype=dt)
        three = ".3." in case["id"]
        c = torch.randn(rows, cols, device=device, dtype=dt) if three else None
        if three:
            return lambda: a + b + c
        return lambda: a + b

    if op == "cast":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        out_dt = torch.float32
        if "fp32_to_bf16" in case["id"]:
            out_dt = torch.bfloat16
        elif "fp16_to_fp32" in case["id"]:
            out_dt = torch.float32
        return lambda: x.to(out_dt) * 1.0

    if op == "gate_mul":
        width = 1
        marker = ".w"
        if marker in case["id"]:
            tail = case["id"].split(marker, 1)[1]
            width = int(tail.split(".", 1)[0])
        x = torch.randn(rows, cols, device=device, dtype=dt)
        g = torch.randn(rows, width, device=device, dtype=dt)
        acc = ".acc" in case["id"]
        a = torch.randn(rows, cols, device=device, dtype=dt) if acc else None

        def run_gate():
            gg = g
            if width != cols:
                gg = g.unsqueeze(-1).expand(rows, width, cols // width).reshape(rows, cols)
            y = x * torch.sigmoid(gg)
            return y + a if acc else y

        return run_gate

    if op.startswith("chain_"):
        return make_chain(case, op, device)

    if op == "merge_states":
        b, h, d, n = rows, 32, cols, 8
        os = torch.randn(b, n, h, d, device=device, dtype=torch.float32)
        ls = torch.randn(b, n, h, device=device, dtype=torch.float32)

        def run_states():
            m = ls.max(dim=1, keepdim=True).values
            w = torch.exp(ls - m)
            den = w.sum(dim=1)
            out = (w[..., None] * os).sum(dim=1) / den[..., None]
            return out, m[:, 0, :] + torch.log(den)

        return run_states

    if op == "merge_split_kv":
        b, h, d, s = rows, 32, cols, 8
        op_ = torch.randn(b, s, h, d, device=device, dtype=torch.float32)
        lp = torch.randn(b, s, h, device=device, dtype=torch.float32)
        sink = torch.randn(h, device=device, dtype=torch.float32)

        def run_split():
            m = torch.maximum(lp.max(dim=1, keepdim=True).values, sink[None, None, :])
            w = torch.exp(lp - m)
            den = w.sum(dim=1) + torch.exp(sink[None, :] - m[:, 0, :])
            out = (w[..., None] * op_).sum(dim=1) / den[..., None]
            return out, m[:, 0, :] + torch.log(den)

        return run_split

    if op == "merge_state":
        oa = torch.randn(rows, cols, device=device, dtype=dt)
        ob = torch.randn(rows, cols, device=device, dtype=dt)
        la = torch.randn(rows, device=device, dtype=dt)
        lb = torch.randn(rows, device=device, dtype=dt)

        def run_merge():
            m = torch.maximum(la, lb)
            wa = torch.exp(la - m)
            wb = torch.exp(lb - m)
            den = wa + wb
            return (oa * wa[:, None] + ob * wb[:, None]) / den[:, None]

        return run_merge

    # ---- B1 norm ---------------------------------------------------------
    if op == "layer_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        b = torch.zeros(cols, device=device, dtype=dt)
        if STYLE == "builtin":
            return lambda: F.layer_norm(x, (cols,), w, b, EPS)
        return lambda: _hf_layer_norm(x, w, b)

    if op == "fused_add_layer_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        res = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        b = torch.zeros(cols, device=device, dtype=dt)

        def run_fused_ln():
            x.add_(res)
            if STYLE == "builtin":
                return F.layer_norm(x, (cols,), w, b, EPS)
            return _hf_layer_norm(x, w, b)

        return run_fused_ln

    if op == "fused_add_rms_norm_out":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        res = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)

        def run_out():
            r = res + x
            return r, _hf_rms(r, w)

        return run_out

    if op == "l2_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        return lambda: x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + 1e-6)

    if op == "gated_rms_norm":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        z = torch.randn(rows, cols, device=device, dtype=dt)
        w = torch.ones(cols, device=device, dtype=dt)
        group = 128

        def run_gated():
            xf = x.float().view(rows, -1, group)
            rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
            y = (xf * rstd).reshape(rows, cols) * w.float() * F.silu(z.float())
            return y.to(dt)

        return run_gated

    if op == "qk_norm":
        hq, hkv, d = 32, 8, 128
        qkv = torch.randn(rows, (hq + 2 * hkv) * d, device=device, dtype=dt)
        w = torch.ones(d, device=device, dtype=dt)

        def run_qk():
            v = qkv.view(rows, hq + 2 * hkv, d)
            q = v[:, :hq, :]
            k = v[:, hq:hq + hkv, :]
            qn = _hf_rms(q, w, 1e-6)
            kn = _hf_rms(k, w, 1e-6)
            return torch.cat([qn, kn, v[:, hq + hkv:, :]], dim=1)

        return run_qk

    # ---- B1 position -----------------------------------------------------
    if op == "attn_temperature_tuning":
        q = torch.randn(rows, 32, cols, device=device, dtype=dt)
        pos = torch.arange(rows, device=device, dtype=torch.float32)
        scale = torch.log(torch.floor((pos + 1.0) / 8192.0) + 1.0) * 0.1 + 1.0
        return lambda: q * scale[:, None, None]

    if op in ("apply_rotary", "apply_rotary_on_the_fly", "apply_mrope"):
        hq, hkv, d = 32, 8, cols
        q = torch.randn(rows, hq, d, device=device, dtype=dt)
        k = torch.randn(rows, hkv, d, device=device, dtype=dt)
        pos = torch.arange(rows, device=device, dtype=torch.int64)
        return lambda: _rope(q, k, pos, d)

    if op == "fused_qk_norm_rope":
        hq, hkv, d = 32, 8, cols
        q = torch.randn(rows, hq, d, device=device, dtype=dt)
        k = torch.randn(rows, hkv, d, device=device, dtype=dt)
        qw = torch.ones(d, device=device, dtype=dt)
        kw = torch.ones(d, device=device, dtype=dt)
        pos = torch.arange(rows, device=device, dtype=torch.int64)

        def run_fused():
            qn = _hf_rms(q, qw, 1e-6)
            kn = _hf_rms(k, kw, 1e-6)
            return _rope(qn, kn, pos, d)

        return run_fused

    if op == "build_cos_sin_cache":
        max_pos, rot = rows, cols
        half = rot // 2
        # Match the kernel's fp64 angle + fp64 sincos, then round to the cache
        # dtype; the kernel writes both cos and sin.
        inv = 1.0 / (1.0e4 ** (torch.arange(0, half, device=device, dtype=torch.float64) / half))
        pos = torch.arange(max_pos, device=device, dtype=torch.float64)
        ang = pos[:, None] * inv[None, :]

        def run_cache():
            c = torch.cos(ang).to(dt)
            s = torch.sin(ang).to(dt)
            return torch.cat([c, s], dim=-1)

        return run_cache

    # ---- B1 embedding ----------------------------------------------------
    if op == "embedding_lookup":
        vocab, hidden = 16384, cols
        w = torch.randn(vocab, hidden, device=device, dtype=dt)
        ids = torch.randint(0, vocab, (rows,), device=device)
        return lambda: F.embedding(ids, w)

    if op == "embedding_with_position":
        vocab, hidden, pos_rows = 16384, cols, 2048
        tw = torch.randn(vocab, hidden, device=device, dtype=dt)
        pw = torch.randn(pos_rows, hidden, device=device, dtype=dt)
        ids = torch.randint(0, vocab, (rows,), device=device)
        pos = torch.arange(rows, device=device) % pos_rows

        def run_embed_pos():
            return F.embedding(ids, tw) + F.embedding(pos, pw)

        return run_embed_pos

    if op == "gather_rows":
        vocab, hidden = 16384, cols
        src = torch.randn(vocab, hidden, device=device, dtype=dt)
        rows_idx = torch.randint(0, vocab, (rows,), device=device)
        return lambda: torch.index_select(src, 0, rows_idx)

    if op == "scatter_rows":
        hidden = cols
        src = torch.randn(rows, hidden, device=device, dtype=dt)
        out = torch.empty(rows, hidden, device=device, dtype=dt)
        rows_idx = torch.randperm(rows, device=device)
        return lambda: out.index_copy_(0, rows_idx, src)

    if op == "logits_postprocess":
        vocab = cols
        raw = torch.randn(rows, vocab, device=device, dtype=dt)
        return lambda: 30.0 * torch.tanh(raw.float() / 30.0)

    # ---- B1 sampling -----------------------------------------------------
    if op == "softmax":
        logits = torch.randn(rows, cols, device=device, dtype=dt)
        t = torch.full((rows,), 0.9, device=device, dtype=torch.float32)
        return lambda: torch.softmax(logits.float() / t[:, None], dim=-1)

    if op == "argmax":
        logits = torch.randn(rows, cols, device=device, dtype=dt)
        return lambda: torch.argmax(logits.float(), dim=-1)

    if op == "compute_logprobs":
        logits = torch.randn(rows, cols, device=device, dtype=dt)
        chosen = torch.zeros(rows, device=device, dtype=torch.int64)
        top_n = 20 if "top20" in case["id"] else 5

        def run_logprobs():
            lp = torch.log_softmax(logits.float(), dim=-1)
            c = lp.gather(1, chosen[:, None]).squeeze(1)
            return c, torch.topk(lp, top_n, dim=-1)

        return run_logprobs

    if op == "token_bincount":
        vocab = cols
        per_row = 64
        ids = torch.arange(rows * per_row, device=device, dtype=torch.int64) % vocab
        ids = ids.view(rows, per_row)
        return lambda: torch.stack([torch.bincount(r, minlength=vocab) for r in ids])

    if op == "apply_penalties":
        vocab = cols
        logits = torch.randn(rows, vocab, device=device, dtype=dt)
        counts = torch.full((rows, vocab), 16843009, device=device, dtype=torch.int32)
        seen = counts > 0
        rep = torch.full((rows,), 1.5, device=device)
        freq = torch.full((rows,), 0.1, device=device)
        pres = torch.full((rows,), 0.1, device=device)

        def run_pen():
            x = logits.float()
            x = torch.where(seen, torch.where(x > 0, x / rep[:, None], x * rep[:, None]), x)
            return x - freq[:, None] * counts.float() - pres[:, None] * seen.float()

        return run_pen

    if op == "apply_token_bitmask":
        logits = torch.randn(rows, cols, device=device, dtype=dt)
        allowed = torch.zeros(rows, cols, device=device, dtype=torch.bool)
        return lambda: logits.float().masked_fill(~allowed, float("-inf"))

    if op == "apply_logit_bias":
        vocab = cols
        logits = torch.randn(rows, vocab, device=device, dtype=dt)
        bias_ids = torch.arange(8, device=device, dtype=torch.int64)
        bias_val = torch.ones(8, device=device)

        def run_bias():
            return logits.float().index_add(1, bias_ids, bias_val.expand(rows, 8))

        return run_bias

    # ---- B3 quantization baselines -----------------------------------------
    if op == "absmax":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        return lambda: x.abs().amax()

    if op == "quantize_static":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        return lambda: (x.float() / 1.0).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    if op in ("quantize_per_token", "quantize_per_token_int8", "dequantize_per_token"):
        x = torch.randn(rows, cols, device=device, dtype=dt)
        if op == "quantize_per_token_int8":
            def run_i8():
                xf = x.float()
                mx = xf.amax(-1, keepdim=True)
                mn = xf.amin(-1, keepdim=True)
                s = torch.clamp((mx - mn) / 255.0, min=1e-10)
                z = torch.round(-128.0 - mn / s)
                return torch.round(xf / s + z).clamp(-128, 127).to(torch.int8)
            return run_i8
        q = (x.float() / (x.float().abs().amax(-1, keepdim=True) / 448.0)).to(torch.float8_e4m3fn)
        scale = x.float().abs().amax(-1, keepdim=True) / 448.0
        if op == "dequantize_per_token":
            return lambda: q.float() * scale
        def run_quant():
            s = x.float().abs().amax(-1, keepdim=True) / 448.0
            return (x.float() / s).to(torch.float8_e4m3fn)
        return run_quant

    if op == "quantize_per_group":
        group = int(_id_int(case["id"], "group", 128))
        x = torch.randn(rows, cols, device=device, dtype=dt)
        def run_group():
            xf = x.float().view(rows, cols // group, group)
            s = xf.abs().amax(-1, keepdim=True) / 448.0
            return (xf / s).to(torch.float8_e4m3fn)
        return run_group

    if op in ("quantize_mxfp8", "quantize_mxfp4"):
        x = torch.randn(rows, cols, device=device, dtype=dt)
        def run_mx():
            xf = x.float().view(rows, cols // 32, 32)
            amax = xf.abs().amax(-1, keepdim=True)
            e = torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=1e-10)))
            s = torch.pow(2.0, e)
            return (xf / s).to(torch.float8_e4m3fn)
        return run_mx

    if op == "quantize_nvfp4":
        x = torch.randn(rows, cols, device=device, dtype=dt)
        def run_nv():
            xf = x.float().view(rows, cols // 16, 16)
            amax = xf.abs().amax(-1, keepdim=True)
            s = (amax * (448.0 * 6.0 / x.float().abs().amax()) / 6.0).to(torch.float8_e4m3fn)
            return (xf * (448.0 * 6.0 / x.float().abs().amax()) / s.float()).to(torch.float8_e4m3fn)
        return run_nv

    if op in ("pack_int4", "unpack_int4"):
        if op == "pack_int4":
            x = torch.randint(-8, 8, (rows, cols), device=device, dtype=torch.int8)
            def run_pack():
                u = x.to(torch.int16) & 0xF
                return (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)
            return run_pack
        x = torch.randint(0, 256, (rows, cols // 2), device=device, dtype=torch.uint8)
        def run_unpack():
            lo = (x & 0xF).to(torch.int8)
            hi = ((x >> 4) & 0xF).to(torch.int8)
            return torch.stack([lo, hi], dim=-1).reshape(rows, cols)
        return run_unpack

    if op == "hadamard":
        n = cols
        x = torch.randn(rows, n, device=device, dtype=torch.float32)
        h = torch.ones(1, 1, device=device)
        while h.shape[-1] < n:
            h = torch.cat([torch.cat([h, h], -1), torch.cat([h, -h], -1)], -2)
        h = h / (n ** 0.5)
        return lambda: x @ h.t()

    raise SystemExit(f"unsupported op: {op}")


def bench(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="JSON from bench_<family> --dump-cases")
    ap.add_argument("--json", required=True, help="output results JSON")
    ap.add_argument("--style", choices=("builtin", "hf"), default="builtin",
                    help="builtin: strongest fused PyTorch primitive; hf: eager multi-op path")
    args = ap.parse_args()
    global STYLE
    STYLE = args.style

    if not torch.cuda.is_available():
        print("torch: CUDA is not available", file=sys.stderr)
        return 1
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    with open(args.cases, "r", encoding="utf-8") as f:
        spec = json.load(f)

    results = []
    for case in spec["cases"]:
        fn = make_op(case, device)
        st = bench(fn, int(case["warmup"]), int(case["iters"]))
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
        print(
            f"{case['id']:<42} {ms:10.4f} ms  {st['min_ms']:10.4f} ms  "
            f"{results[-1]['gbs']:9.1f} GB/s"
        )

    props = torch.cuda.get_device_properties(0)
    out = {
        "schema": 1,
        "backend": f"torch {torch.__version__} ({STYLE})",
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
        "results": results,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
