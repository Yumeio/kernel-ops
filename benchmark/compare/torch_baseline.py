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
    "int8": torch.int8,
    "uint8": torch.uint8,
    "float8_e4m3fn": torch.float8_e4m3fn,
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

    if op == "chain_moe_route_permute":
        T, E, H, k = rows, 256, cols, 8
        x = torch.randn(T, E, device=device)
        b = torch.randn(E, device=device) * 0.1
        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)

        def run():
            s = torch.sigmoid(x)
            sel = s + b
            w, ids = torch.topk(sel, k, dim=-1)
            ww = torch.gather(s, 1, ids)
            ww = ww / ww.sum(-1, keepdim=True)
            flat = ids.reshape(-1)
            order = torch.sort(flat).values
            cnt = torch.bincount(order, minlength=E)
            pad = ((cnt + k - 1) // k) * k
            torch.cumsum(pad, 0)
            return hidden[flat // k]

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

    b4 = make_b4_op(case, device)
    if b4 is not None:
        return b4

    b5 = make_b5_op(case, device)
    if b5 is not None:
        return b5

    b6 = make_b6_op(case, device)
    if b6 is not None:
        return b6

    raise SystemExit(f"unsupported op: {op}")


# ---- B4 weight preparation baselines ---------------------------------------------------
# What a Python checkpoint loader does today: the vLLM `marlin_utils_test` formulation of the
# Marlin repack (tile permutation + column packing), `awq_dequantize_pytorch`, the DeepGEMM
# `per_block_cast_to_fp8` / `ceil_to_ue8m0` helpers, torchao `to_blocked`, and plain
# `.t().contiguous()` / `.sum()` / in-place broadcast ops.  Where PyTorch has a single fused
# primitive the baseline uses it, so these ratios are not inflated by a weak reference.
_MARLIN_CACHE = {}
AWQ_REV = [0, 4, 1, 5, 2, 6, 3, 7]  # column c of an 8-column AWQ word sits at nibble AWQ_REV[c]


def _marlin_perm(num_bits: int, device):
    key = ("perm", num_bits, str(device))
    if key not in _MARLIN_CACHE:
        base = []
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in (0, 1):
                for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                base.extend(p + 256 * j for p in perm1)
        interleave = [0, 2, 4, 6, 1, 3, 5, 7] if num_bits == 4 else [0, 2, 1, 3]
        perm = torch.tensor(base, dtype=torch.int64).reshape(-1, len(interleave))[:, interleave].ravel()
        _MARLIN_CACHE[key] = perm.to(device)
    return _MARLIN_CACHE[key]


def _scale_perm(device, single: bool = False):
    key = ("sperm", single, str(device))
    if key not in _MARLIN_CACHE:
        if single:
            perm = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
        else:
            perm = [i + 8 * j for i in range(8) for j in range(8)]
        _MARLIN_CACHE[key] = torch.tensor(perm, dtype=torch.int64, device=device)
    return _MARLIN_CACHE[key]


def _marlin_pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """q: int32 [K, N] with values in [0, 2^bits) -> Marlin words [K/16, N*16/pack]."""
    k, n = q.shape
    pack = 32 // bits
    perm = _marlin_perm(bits, q.device)
    q = q.reshape(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(k // 16, n * 16)
    q = q.reshape(-1, perm.numel())[:, perm].reshape(k // 16, n * 16)
    out = torch.zeros(k // 16, n * 16 // pack, dtype=torch.int32, device=q.device)
    for i in range(pack):
        out |= q[:, i::pack] << (bits * i)
    return out


def _unpack_nibbles(words: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, device=words.device, dtype=torch.int32)
    return (words.unsqueeze(-1) >> shifts) & 0xF


def _unpack_awq(words: torch.Tensor) -> torch.Tensor:
    return _unpack_nibbles(words)[..., AWQ_REV].reshape(words.shape[0], -1)


def _make_scale_perm_op(scales: torch.Tensor):
    n = scales.shape[-1]
    sp = _scale_perm(scales.device)
    return lambda: scales.reshape(-1, 64)[:, sp].reshape(-1, n)


def _q4k_scale_min(s12: torch.Tensor):
    """get_scale_min_k4 for all eight sub-blocks: s12 is [nb, 12] uint8 -> ([nb, 8], [nb, 8]) int32."""
    s12 = s12.to(torch.int32)
    sc = torch.empty(s12.shape[0], 8, dtype=torch.int32, device=s12.device)
    mn = torch.empty_like(sc)
    sc[:, :4] = s12[:, 0:4] & 63
    mn[:, :4] = s12[:, 4:8] & 63
    sc[:, 4:] = (s12[:, 8:12] & 0xF) | ((s12[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s12[:, 8:12] >> 4) | ((s12[:, 4:8] >> 6) << 4)
    return sc, mn


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool):
    """DeepGEMM `per_block_cast_to_fp8` (128x128 blocks); skips the padding copy when aligned."""
    m, n = x.shape
    if m % 128 == 0 and n % 128 == 0:
        x_padded = x
    else:
        x_padded = torch.zeros((m + 127) // 128 * 128, (n + 127) // 128 * 128, dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(x_view.size(0), x_view.size(2))


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    """torchao / FlashInfer 128x4 scale swizzle."""
    rows, cols = m.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded = m
    if (rows, cols) != (n_row_blocks * 128, n_col_blocks * 4):
        padded = torch.zeros((n_row_blocks * 128, n_col_blocks * 4), device=m.device, dtype=m.dtype)
        padded[:rows, :cols] = m
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def make_b4_op(case: dict, device: torch.device):
    op = case["op"]
    rows = int(case["rows"])
    cols = int(case["cols"])
    k, n = rows, cols
    g = 128

    if op in ("gptq_to_marlin", "gptq_to_marlin_actorder"):
        qweight = torch.randint(-2**31, 2**31 - 1, (k // 8, n), device=device, dtype=torch.int32)
        scales = torch.randn(k // g, n, device=device).half()
        g_idx = (torch.arange(k, device=device, dtype=torch.int64) * 2654435761 % (k // g)).to(torch.int32)
        sperm = _make_scale_perm_op(scales)

        def run_gptq():
            shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32).view(1, 8, 1)
            q = ((qweight.unsqueeze(1) >> shifts) & 0xF).reshape(k, n)
            if op.endswith("actorder"):
                perm = torch.argsort(g_idx, stable=True)
                q = q[perm]
            return _marlin_pack(q, 4), sperm()

        return run_gptq

    if op == "awq_to_marlin":
        qweight = torch.randint(-2**31, 2**31 - 1, (k, n // 8), device=device, dtype=torch.int32)
        qzeros = torch.randint(-2**31, 2**31 - 1, (k // g, n // 8), device=device, dtype=torch.int32)
        scales = torch.randn(k // g, n, device=device).half()
        sperm = _make_scale_perm_op(scales)
        zperm = _scale_perm(device)
        interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device)

        def run_awq():
            qw = _marlin_pack(_unpack_awq(qweight), 4)
            z = _unpack_awq(qzeros)                                   # [G, N] natural column order
            z = z.reshape(-1, 64)[:, zperm].reshape(-1, 8)[:, interleave].reshape(k // g, n)
            zp = torch.zeros(k // g, n // 8, dtype=torch.int32, device=device)
            for i in range(8):
                zp |= z[:, i::8] << (4 * i)
            return qw, sperm(), zp

        return run_awq

    if op == "awq_dequantize":
        qweight = torch.randint(-2**31, 2**31 - 1, (k, n // 8), device=device, dtype=torch.int32)
        qzeros = torch.randint(-2**31, 2**31 - 1, (k // g, n // 8), device=device, dtype=torch.int32)
        scales = (torch.rand(k // g, n, device=device) * 0.2 + 0.001).half()

        def run_awq_dq():
            iw = _unpack_awq(qweight)
            iz = _unpack_awq(qzeros).repeat_interleave(g, dim=0)
            return (iw - iz) * scales.repeat_interleave(g, dim=0)

        return run_awq_dq

    if op == "gguf_dequantize_q4_k":
        nb = k * n // 256
        raw = torch.randint(0, 256, (nb * 144,), device=device, dtype=torch.uint8).view(nb, 144)
        raw[:, 0:4] = torch.tensor([0x00, 0x3C, 0x00, 0x34], device=device, dtype=torch.uint8)  # d = 1.0, dmin = 0.25

        def run_q4k():
            d = raw[:, 0:2].contiguous().view(torch.float16).float()
            dmin = raw[:, 2:4].contiguous().view(torch.float16).float()
            sc, mn = _q4k_scale_min(raw[:, 4:16])
            qs = raw[:, 16:144].view(nb, 4, 32)
            q = torch.stack([qs & 0xF, qs >> 4], dim=2)               # [nb, 4, 2, 32]
            d1 = (d * sc.float()).view(nb, 4, 2, 1)
            m1 = (dmin * mn.float()).view(nb, 4, 2, 1)
            return (d1 * q.float() - m1).reshape(k, n).half()

        return run_q4k

    if op == "gguf_dequantize_q8_0":
        nb = k * n // 32
        raw = torch.randint(0, 256, (nb * 34,), device=device, dtype=torch.uint8).view(nb, 34)

        def run_q80():
            d = raw[:, 0:2].contiguous().view(torch.float16).float()
            q = raw[:, 2:].contiguous().view(torch.int8).float()
            return (q * d).reshape(k, n).half()

        return run_q80

    if op == "fp4_to_marlin":
        weight = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8)
        scales = torch.randint(0, 256, (n, k // 32), device=device, dtype=torch.uint8)
        sperm = _scale_perm(device)

        def run_fp4():
            nib = torch.stack([weight & 0xF, weight >> 4], dim=-1).reshape(n, k)
            qw = _marlin_pack(nib.t().to(torch.int32), 4)
            s = scales.t().contiguous().reshape(-1, 64)[:, sperm].reshape(-1, n)
            return qw, s

        return run_fp4

    if op == "fp8_block_scale_to_ue8m0":
        sn, sk = k // 128, n // 128
        w_q = torch.randint(0, 0x7F, (k, n), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn)
        s_old = torch.full((sn, sk), 0.0137, device=device)

        def run_ue():
            bits = s_old.view(torch.int32) & 0x7FFFFFFF
            e = ((bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(1, 254)
            s_new = (e << 23).view(torch.float32)
            ratio = s_old / s_new
            q = (w_q.view(sn, 128, sk, 128).float() * ratio.view(sn, 1, sk, 1)).to(torch.float8_e4m3fn)
            return q.view(k, n), s_new, e.to(torch.uint8)

        return run_ue

    if op == "smoothquant_fold":
        w = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        nw = torch.ones(n, device=device, dtype=torch.bfloat16)
        sm = torch.ones(n, device=device)

        def run_fold():
            w.mul_(sm)
            nw.div_(sm)

        return run_fold

    # ---- gemm preparation ---------------------------------------------------------
    if op == "weight_transpose":
        dt = DTYPES[case["dtype"]]
        x = torch.randn(rows, cols, device=device).to(dt)
        return lambda: x.t().contiguous()

    if op == "weight_transpose_int4":
        x = torch.randint(0, 256, (rows, cols // 2), device=device, dtype=torch.uint8)

        def run_t4():
            u = torch.stack([x & 0xF, x >> 4], dim=-1).reshape(rows, cols)
            t = u.t().contiguous()
            return t[:, 0::2] | (t[:, 1::2] << 4)

        return run_t4

    if op == "glu_interleave":
        x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
        inter = rows // 2
        d = 64

        def run_glu():
            return torch.stack([x[:inter].view(inter // d, d, cols), x[inter:].view(inter // d, d, cols)],
                               dim=1).reshape(rows, cols)

        return run_glu

    if op == "router_split":
        w = torch.randn(rows, cols, device=device)

        def run_split():
            hi = w.to(torch.bfloat16)
            return hi, ((w - hi.float()) * 256.0).to(torch.bfloat16)

        return run_split

    if op == "int8_colsum":
        w = torch.randint(-128, 128, (rows, cols), device=device, dtype=torch.int8)
        return lambda: w.sum(dim=1, dtype=torch.int32)

    if op == "block_scale_swizzle":
        x = torch.randint(0, 256, (rows, cols), device=device, dtype=torch.uint8)
        return lambda: _to_blocked(x)

    if op == "fp8_scale_to_tma":
        x = torch.rand(rows, cols, device=device) + 0.5
        return lambda: x.t().contiguous()

    if op == "fp8_scale_to_tma_ue8m0":
        x = torch.exp2(torch.randint(-10, 10, (rows, cols), device=device).float())

        def run_pack():
            packed = ((x.view(torch.int32) >> 23) & 0xFF).to(torch.uint8).view(torch.int32)
            return packed.t().contiguous()

        return run_pack

    if op in ("fp8_block_quant", "fp8_block_quant_ue8m0"):
        x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
        ue = op.endswith("ue8m0")
        return lambda: _per_block_cast_to_fp8(x, ue)

    return None


# ---- B5 decode GEMV baselines -----------------------------------------------------------
# rows = N, cols = K, M = flops / (2 N K).  Every quantized case replaces the eager PyTorch
# "dequantize the weight, then F.linear" sequence an engine without a fused kernel runs:
# PyTorch has no fused weight-only GEMV for these formats on SM86 (torch._scaled_mm needs SM89,
# torch._int_mm needs M > 16, the int4 pack op has its own tile layout), so the dequantized
# weight is materialised in bf16 on every call, exactly as AutoAWQ / GPTQ / bitsandbytes /
# HF fp8 fallbacks do.  Scales that apply per output channel are applied on the output.


def make_b5_op(case: dict, device: torch.device):
    op = case["op"]
    if op not in ("w4a16_awq", "w4a16_gptq", "nf4_lut", "fp8_gemv_rowwise", "fp8_gemv_blockwise",
                  "int8_gemv_w8a8", "int8_gemv_w8a16", "dense_gemv", "fp4_gemv_mx", "fp4_gemv_nv",
                  "gguf_mmvq_q4_0", "gguf_mmvq_q8_0", "gguf_mmvq_q4_k", "router_gemm", "gemv_argmax",
                  "grouped_fp8", "moe_gemv_fp8"):
        return None
    n = int(case["rows"])
    k = int(case["cols"])
    m = int(case["flops"]) // (2 * n * k)
    bf = torch.bfloat16
    g = 128
    x = torch.randn(m, k, device=device, dtype=bf)
    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)

    if op == "w4a16_awq":
        qweight = torch.randint(-2**31, 2**31 - 1, (k, n // 8), device=device, dtype=torch.int32)
        qzeros = torch.randint(-2**31, 2**31 - 1, (k // g, n // 8), device=device, dtype=torch.int32)
        scales = (torch.rand(k // g, n, device=device) * 0.03 + 0.002).to(bf)

        def run_awq():
            iw = _unpack_awq(qweight)
            iz = _unpack_awq(qzeros).repeat_interleave(g, dim=0)
            w = ((iw - iz) * scales.repeat_interleave(g, dim=0)).to(bf)
            return x @ w

        return run_awq

    if op == "w4a16_gptq":
        qweight = torch.randint(-2**31, 2**31 - 1, (k // 8, n), device=device, dtype=torch.int32)
        qzeros = torch.randint(-2**31, 2**31 - 1, (k // g, n // 8), device=device, dtype=torch.int32)
        scales = (torch.rand(k // g, n, device=device) * 0.03 + 0.002).to(bf)

        def run_gptq():
            q = ((qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(k, n)
            z = ((qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF).reshape(k // g, n)
            w = ((q - z.repeat_interleave(g, dim=0)) * scales.repeat_interleave(g, dim=0)).to(bf)
            return x @ w

        return run_gptq

    if op == "nf4_lut":
        qw = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8)
        cb = torch.linspace(-1, 1, 16, device=device).to(bf)
        absmax = torch.rand(n * k // 64, device=device) + 0.1

        def run_nf4():
            codes = torch.stack([qw & 0xF, qw >> 4], dim=-1).reshape(n, k)
            w = cb[codes.long()] * absmax.view(n, k // 64).repeat_interleave(64, dim=1).to(bf)
            return x @ w.t()

        return run_nf4

    if op in ("fp8_gemv_rowwise", "fp8_gemv_blockwise"):
        w8 = torch.randn(n, k, device=device).to(torch.float8_e4m3fn)
        if op == "fp8_gemv_rowwise":
            col_s = torch.rand(n, device=device) + 0.5
            return lambda: (x @ w8.to(bf).t()) * col_s.to(bf)
        sb = torch.rand(n // 128, k // 128, device=device) + 0.5

        def run_fp8_block():
            w = (w8.to(bf).view(n // 128, 128, k // 128, 128) * sb.to(bf).view(n // 128, 1, k // 128, 1)).reshape(n, k)
            return x @ w.t()

        return run_fp8_block

    if op == "dense_gemv":
        w = torch.randn(n, k, device=device, dtype=bf)
        return lambda: F.linear(x, w)

    if op == "router_gemm":
        # fp32 logits from bf16 hidden/weights with fp32 accumulation: upcast, then an fp32 GEMM.
        w = torch.randn(n, k, device=device, dtype=bf) * 0.02
        return lambda: F.linear(x.float(), w.float())

    if op == "gemv_argmax":
        w = torch.randn(n, k, device=device, dtype=bf) * 0.02
        return lambda: (x @ w.t()).argmax(dim=-1)

    if op in ("fp4_gemv_mx", "fp4_gemv_nv"):
        nv = op == "fp4_gemv_nv"
        blk = 16 if nv else 32
        packed = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8)
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=device, dtype=bf)
        if nv:
            sc = (torch.rand(n, k // blk, device=device) + 0.1).to(torch.float8_e4m3fn)
        else:
            sc = torch.randint(110, 130, (n, k // blk), device=device, dtype=torch.uint8)

        def run_fp4_gemv():
            codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(n, k)
            s = sc.to(bf) if nv else torch.exp2(sc.float() - 127).to(bf)
            w = lut[codes.long()] * s.repeat_interleave(blk, dim=1)
            return x @ w.t()

        return run_fp4_gemv

    if op.startswith("gguf_mmvq_"):
        t = op[len("gguf_mmvq_"):]
        qk, nbytes = {"q4_0": (32, 18), "q8_0": (32, 34), "q4_k": (256, 144)}[t]
        nb = n * k // qk
        raw = torch.randint(0, 256, (nb * nbytes,), device=device, dtype=torch.uint8).view(nb, nbytes)
        raw[:, 0] = 0x3C  # keep the fp16 scales finite
        raw[:, 1] = 0x20
        if t == "q4_k":
            raw[:, 2] = 0x3C
            raw[:, 3] = 0x20

        def run_gguf():
            d = raw[:, 0:2].contiguous().view(torch.float16).float()
            if t == "q4_0":
                qs = raw[:, 2:]
                q = torch.cat([qs & 0xF, qs >> 4], dim=1).float() - 8.0
                w = (q * d).reshape(n, k)
            elif t == "q8_0":
                q = raw[:, 2:].contiguous().view(torch.int8).float()
                w = (q * d).reshape(n, k)
            else:
                dmin = raw[:, 2:4].contiguous().view(torch.float16).float()
                sc_, mn_ = _q4k_scale_min(raw[:, 4:16])
                qs = raw[:, 16:144].view(nb, 4, 32)
                q = torch.stack([qs & 0xF, qs >> 4], dim=2)
                w = ((d * sc_.float()).view(nb, 4, 2, 1) * q.float() - (dmin * mn_.float()).view(nb, 4, 2, 1)).reshape(n, k)
            return x @ w.to(bf).t()

        return run_gguf

    if op == "grouped_fp8":
        e = 32
        xg = torch.randn(e, 2, k, device=device, dtype=bf)
        w8 = torch.randn(e, n, k, device=device).to(torch.float8_e4m3fn)
        col_s = torch.rand(e, 1, n, device=device, dtype=bf) + 0.5
        return lambda: torch.bmm(xg, w8.to(bf).transpose(1, 2)) * col_s

    if op == "moe_gemv_fp8":
        e = 32
        xs = torch.randn(4, k, device=device, dtype=bf)
        w8 = torch.randn(e, n, k, device=device).to(torch.float8_e4m3fn)
        col_s = torch.rand(e, 1, n, device=device, dtype=bf) + 0.5
        eids = torch.arange(e, device=device)
        tok = torch.arange(e, device=device) // 8

        def run_moe():
            wsel = w8[eids].to(bf)                      # [slots, N, K] gather + cast
            return torch.bmm(xs[tok].unsqueeze(1), wsel.transpose(1, 2)) * col_s[eids]

        return run_moe

    w8 = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
    col_s = torch.rand(n, device=device) * 0.01
    if op == "int8_gemv_w8a16":
        return lambda: (x @ w8.to(bf).t()) * col_s.to(bf)
    a8 = torch.randint(-128, 128, (m, k), device=device, dtype=torch.int8)
    row_s = torch.rand(m, 1, device=device) * 0.01
    return lambda: ((a8.to(bf) @ w8.to(bf).t()).float() * row_s * col_s).to(bf)


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



# ---- B6 routing / metadata / sampling baselines ---------------------------------------
def make_b6_op(case: dict, device: torch.device):
    op = case["op"]
    rows = int(case["rows"])
    cols = int(case["cols"])
    T, E, k = rows, 256, 8

    if op == "moe_route":
        x = torch.randn(T, E, device=device)
        b = torch.randn(E, device=device) * 0.1

        def run():
            s = torch.sigmoid(x)
            sel = s + b
            w, ids = torch.topk(sel, k, dim=-1)
            ww = torch.gather(s, 1, ids)
            ww = ww / ww.sum(-1, keepdim=True)
            return ww * 2.5, ids

        return run

    if op == "moe_grouped":
        G, TG = 8, 4
        x = torch.randn(T, E, device=device)
        b = torch.randn(E, device=device) * 0.1

        def run():
            s = torch.sigmoid(x)
            sel = s + b
            gs = sel.view(T, G, E // G)
            g2 = gs.topk(2, dim=-1).values.sum(-1)
            gi = torch.topk(g2, TG, dim=-1).indices
            mask = torch.zeros(T, G, device=device, dtype=torch.bool).scatter_(1, gi, True)
            sel = sel.view(T, G, E // G).masked_fill(~mask[:, :, None], float("-inf")).view(T, E)
            w, ids = torch.topk(sel, k, dim=-1)
            ww = torch.gather(s, 1, ids)
            ww = ww / ww.sum(-1, keepdim=True)
            return ww * 2.5, ids

        return run

    if op == "moe_fused_router":
        H = cols
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        w = torch.randn(E, H, device=device, dtype=torch.bfloat16)
        b = torch.randn(E, device=device, dtype=torch.bfloat16)

        def run():
            logits = torch.nn.functional.linear(x, w, b).float()
            s = torch.sigmoid(logits)
            ww, ids = torch.topk(s, 8, dim=-1)
            ww = ww / ww.sum(-1, keepdim=True)
            return ww * 2.5, ids

        return run

    if op == "moe_hash":
        tok = torch.randint(0, 32000, (T,), device=device, dtype=torch.int64)
        tab = torch.randint(0, E, (32000, k), device=device, dtype=torch.int32)
        return lambda: (torch.nn.functional.embedding(tok, tab), torch.full((T, k), 2.0 / k, device=device))

    if op == "moe_count_offsets":
        ids = torch.randint(0, E, (rows,), device=device)

        def run():
            c = torch.bincount(ids, minlength=E)
            return torch.cumsum(c, 0)

        return run

    if op == "moe_align":
        block = 128
        slots = rows
        ids = torch.randint(0, E, (slots,), device=device)

        def run():
            s = torch.sort(ids).values
            cnt = torch.bincount(s, minlength=E)
            pad = ((cnt + block - 1) // block) * block
            off = torch.cumsum(pad, 0) - pad
            eids = torch.repeat_interleave(torch.arange(E, device=device), pad // block)
            return s, eids, int(pad.sum())

        return run

    if op in ("moe_permute", "moe_permute_fp8"):
        H = cols
        x = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        idx = torch.randint(0, T, (T * k,), device=device)
        if op == "moe_permute":
            return lambda: x[idx]

        def run_quant():
            y = x[idx].float().view(T * k, H // 128, 128)
            amax = y.abs().amax(-1, keepdim=True)
            scale = amax.clamp_min(1e-4) / 448.0
            q = (y / scale).to(torch.float8_e4m3fn)
            return q, scale

        return run_quant

    if op == "moe_unpermute":
        H = cols
        eo = torch.randn(T * k, H, device=device, dtype=torch.bfloat16)
        slot = torch.arange(T * k, device=device).view(T, k)
        w = torch.randn(T, k, device=device)
        sh = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        res = torch.randn(T, H, device=device, dtype=torch.bfloat16)

        def run():
            g = eo[slot]
            return (g * w[..., None]).sum(1) + sh + res

        return run

    if op == "moe_sum":
        H = cols
        x = torch.randn(T, k, H, device=device, dtype=torch.bfloat16)
        return lambda: x.sum(1)

    if op == "moe_latent":
        L = cols
        eo = torch.randn(T * k, L, device=device, dtype=torch.bfloat16)
        slot = torch.arange(T * k, device=device).view(T, k)
        w = torch.randn(T, k, device=device)
        nw = torch.ones(L, device=device, dtype=torch.bfloat16)

        def run():
            z = (eo[slot] * w[..., None]).sum(1)
            return torch.nn.functional.rms_norm(z, (L,), nw, 1e-5)

        return run

    if op == "chain_moe_route_permute":
        H = cols
        x = torch.randn(T, E, device=device)
        b = torch.randn(E, device=device) * 0.1
        hidden = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        block = 8

        def run():
            s = torch.sigmoid(x)
            sel = s + b
            w, ids = torch.topk(sel, k, dim=-1)
            ww = torch.gather(s, 1, ids)
            ww = ww / ww.sum(-1, keepdim=True)
            flat = ids.reshape(-1)
            order = torch.sort(flat).values
            cnt = torch.bincount(order, minlength=E)
            pad = ((cnt + block - 1) // block) * block
            torch.cumsum(pad, 0)
            return hidden[flat // k]

        return run

    if op == "sample":
        kk = 20 if "k20" in case["id"] else 50
        p = 0.95 if "p0.95" in case["id"] else 0.9
        x = torch.randn(T, cols, device=device)

        def run():
            probs = torch.softmax(x, dim=-1)
            w, ids = torch.topk(probs, kk, dim=-1)
            m = torch.zeros_like(probs).scatter_(1, ids, w)
            m = m / m.sum(-1, keepdim=True)
            sp, si = torch.sort(m, dim=-1, descending=True)
            cs = torch.cumsum(sp, dim=-1)
            keep = (cs <= p) | (cs - sp < p)
            m2 = torch.zeros_like(m).scatter_(1, si, sp * keep)
            m2 = m2 / m2.sum(-1, keepdim=True)
            return torch.multinomial(m2, 1)

        return run

    if op == "gumbel":
        x = torch.randn(T, cols, device=device)

        def run():
            probs = torch.softmax(x, dim=-1)
            return torch.multinomial(probs, 1)

        return run

    if op == "topk_renorm":
        probs = torch.rand(T, cols, device=device)
        probs = probs / probs.sum(-1, keepdim=True)

        def run():
            w, ids = torch.topk(probs, 50, dim=-1)
            out = torch.zeros_like(probs).scatter_(1, ids, w)
            return out / out.sum(-1, keepdim=True)

        return run

    if op == "topp_renorm":
        probs = torch.rand(T, cols, device=device)
        probs = probs / probs.sum(-1, keepdim=True)

        def run():
            sp, si = torch.sort(probs, dim=-1, descending=True)
            cs = torch.cumsum(sp, dim=-1)
            keep = (cs <= 0.9) | (cs - sp < 0.9)
            out = torch.zeros_like(probs).scatter_(1, si, sp * keep)
            return out / out.sum(-1, keepdim=True)

        return run

    if op == "topk_mask":
        x = torch.randn(T, cols, device=device)

        def run():
            w, ids = torch.topk(x, 50, dim=-1)
            return torch.full_like(x, float("-inf")).scatter_(1, ids, w)

        return run

    return None


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
