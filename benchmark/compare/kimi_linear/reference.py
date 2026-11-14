"""Reconstructed eager reference for the Kimi-Linear W4A8 hybrid decode step.

NOT the upstream reference.py (we do not have it): the semantics below are read
back out of `kimi_mega_triton.py` -- same parameter/buffer names, same rounding
points (bf16 where the megakernel rounds), same W4A8 arithmetic (int4 weights
with per-128-group scale/zero, activations quantised to int8 per 128-group by
abs-max).  The one deliberate difference: kv_b is applied un-absorbed and with
exact (unquantised) activations, i.e. the O(L*512*8192) form, so the absorbed
MLA paths of both megakernels are checked against the plain math.

Everything runs in fp64 on the GPU; this file is a correctness oracle and an
"eager PyTorch" timing baseline, nothing more.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Cfg:
    hidden: int = 2304
    kda_heads: int = 32
    kda_head_dim: int = 128
    mla_heads: int = 32
    kv_lora: int = 512
    qk_nope: int = 128
    qk_rope: int = 64
    v_head: int = 128
    moe_inter: int = 1024
    n_experts: int = 64
    n_shared: int = 1
    top_k: int = 8
    group: int = 128
    short_conv: int = 4
    pattern: str = "KKKM"
    rope_theta: float = 10000.0
    routed_scale: float = 2.446
    dtype: torch.dtype = torch.bfloat16


F64 = torch.float64


def bf(x):
    """round to bf16, keep fp64 container."""
    return x.to(torch.bfloat16).to(F64)


def unpack(wq):
    """(K//2, N) uint8 -> (K, N) nibbles; low nibble is the even contraction index."""
    k2, n = wq.shape
    v = torch.empty(k2 * 2, n, dtype=F64, device=wq.device)
    v[0::2] = (wq & 0xF).to(F64)
    v[1::2] = ((wq >> 4) & 0xF).to(F64)
    return v


def dequant(wq, scales, zeros, group=128):
    v = unpack(wq)
    s = scales.to(F64).repeat_interleave(group, 0)
    z = zeros.to(F64).repeat_interleave(group, 0)
    return (v - z) * s


def qlinear(x, wq, scales, zeros, group=128):
    """W4A8 GEMV: x (K,) -> (N,).  int8 activations per 128-group (abs-max)."""
    k = x.numel()
    xg = x.to(F64).view(k // group, group)
    mx = xg.abs().amax(1, keepdim=True)
    xq = torch.round(xg.to(torch.float32) * (127.0 / (mx.to(torch.float32) + 1.0e-30))).to(F64)
    xd = (xq * (mx.to(torch.float32) * (1.0 / 127.0)).to(F64)).reshape(-1)
    return xd @ dequant(wq, scales, zeros, group)


def rmsnorm(h, w):
    h = h.to(F64)
    return bf(h * torch.rsqrt((h * h).mean() + 1.0e-6) * w.to(F64))


def ql(mod, x, i=None):
    if i is None:
        return qlinear(x, mod.w_q, mod.scales, mod.zeros, mod.group)
    return qlinear(x, mod.w_q[i], mod.scales[i], mod.zeros[i], mod.group)


def rope(x, pos, theta):
    n = x.numel()
    inv = 1.0 / (theta ** (torch.arange(0, n, 2, device=x.device, dtype=torch.float32) / n))
    ang = (torch.tensor(float(pos), dtype=torch.float32, device=x.device) * inv)
    cs, sn = torch.cos(ang).to(F64), torch.sin(ang).to(F64)
    e, o = x[0::2], x[1::2]
    out = torch.empty_like(x)
    out[0::2] = e * cs - o * sn
    out[1::2] = o * cs + e * sn
    return bf(out)


def kda_step(cfg, blk, h, st):
    a = blk.attn
    H, DK = cfg.kda_heads, cfg.kda_head_dim
    xn = rmsnorm(h, blk.attn_norm)
    outs = []
    new = {}
    for mi, (name, proj) in enumerate((("cq", a.q_proj), ("ck", a.k_proj), ("cv", a.v_proj))):
        val = bf(ql(proj, xn))
        prev = st[name].to(F64)                       # (3, C4)
        w = a.conv_w[mi].to(F64)                      # (C4, 4)
        o = prev[0] * w[:, 0] + prev[1] * w[:, 1] + prev[2] * w[:, 2] + val * w[:, 3]
        o = bf(o * torch.sigmoid(o))
        outs.append(o)
        new[name] = torch.stack([prev[1], prev[2], val]).to(torch.bfloat16)
    q = (outs[0] * (float(DK) ** -0.5)).view(H, DK)
    k = outs[1].view(H, DK)
    v = outs[2].view(H, DK)
    decay = torch.sigmoid(-bf(ql(a.g_proj, xn))).view(H, DK)
    beta = torch.sigmoid(bf(a.beta_proj.weight.to(F64) @ xn))
    S = st["S"].to(F64) * decay[:, :, None]
    pred = torch.einsum("hkv,hk->hv", S, k)
    dv = beta[:, None] * (v - pred)
    S = S + k[:, :, None] * dv[:, None, :]
    o = bf(torch.einsum("hkv,hk->hv", S, q)).reshape(-1)
    new["S"] = S.to(torch.float32)
    return bf(h.to(F64)) + ql(a.o_proj, o), new


def mla_step(cfg, blk, h, st):
    a = blk.attn
    H = cfg.mla_heads
    xn = rmsnorm(h, blk.attn_norm)
    q = bf(ql(a.q_proj, xn)).view(H, cfg.qk_nope + cfg.qk_rope)
    kva = bf(ql(a.kv_a, xn))
    L = st["c_kv"].shape[0]
    ck = kva[:cfg.kv_lora]
    kr = rope(kva[cfg.kv_lora:], L, cfg.rope_theta)
    c_kv = torch.cat([st["c_kv"], ck.to(torch.bfloat16)[None]], 0)
    k_rope = torch.cat([st["k_rope"], kr.to(torch.bfloat16)[None]], 0)
    qn = q[:, :cfg.qk_nope]
    qr = torch.stack([rope(q[i, cfg.qk_nope:], L, cfg.rope_theta) for i in range(H)])
    Wb = dequant(a.kv_b.w_q, a.kv_b.scales, a.kv_b.zeros, cfg.group)
    Wb = Wb.view(cfg.kv_lora, H, cfg.qk_nope + cfg.v_head)
    scale = float(cfg.qk_nope + cfg.qk_rope) ** -0.5
    sc = torch.zeros(H, L + 1, dtype=F64, device=h.device)
    o = torch.zeros(H, cfg.v_head, dtype=F64, device=h.device)
    ckf = c_kv.to(F64)
    for hh in range(H):                               # per head keeps memory flat
        kn = ckf @ Wb[:, hh, :cfg.qk_nope]            # (L+1, 128)
        sc[hh] = (kn @ qn[hh] + k_rope.to(F64) @ qr[hh]) * scale
        p = torch.softmax(sc[hh], 0)
        o[hh] = (p @ ckf) @ Wb[:, hh, cfg.qk_nope:]
    new = {"c_kv": c_kv, "k_rope": k_rope}
    return bf(h.to(F64)) + ql(a.o_proj, o.reshape(-1)), new


def moe_step(cfg, blk, ha):
    m = blk.moe
    ha = bf(ha)
    xn = rmsnorm(ha, blk.moe_norm)
    lg = bf(m.router.weight.to(F64) @ xn).to(torch.float32)
    p = torch.softmax(lg, 0)
    order = sorted(range(cfg.n_experts), key=lambda i: (-float(lg[i]), i))[:cfg.top_k]
    ws = sum(float(p[i]) for i in order)
    out = ha.clone()
    for i in order:
        hh = ql(m.gate, xn, i)
        hh = hh * torch.sigmoid(hh) * ql(m.up, xn, i)
        out += (float(p[i]) * cfg.routed_scale / (ws + 1.0e-9)) * ql(m.down, hh, i)
    hh = ql(m.s_gate, xn, 0)
    hh = hh * torch.sigmoid(hh) * ql(m.s_up, xn, 0)
    out += ql(m.s_down, hh, 0)
    return bf(out), order


@torch.no_grad()
def reference_step(model, hidden, state):
    """model: any module tree with the reference parameter names. Returns (hidden, state)."""
    cfg = model.cfg
    h = hidden.to(F64)
    new_state = []
    for blk, kind, st in zip(model.blocks, cfg.pattern, state):
        if kind == "K":
            ha, ns = kda_step(cfg, blk, h, st)
        else:
            ha, ns = mla_step(cfg, blk, h, st)
        h, _ = moe_step(cfg, blk, ha)
        new_state.append(ns)
    return h.to(torch.bfloat16), new_state


# --------------------------------------------------------------------------- #
# shared fixtures: random weights and a random decode state at context length L
# --------------------------------------------------------------------------- #
@torch.no_grad()
def init_random(model, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name, buf in model.named_buffers():
        if name.endswith("w_q"):
            buf.copy_(torch.randint(0, 256, buf.shape, generator=g, dtype=torch.uint8))
        elif name.endswith("zeros"):
            buf.copy_(torch.randint(6, 10, buf.shape, generator=g).to(buf.dtype))
        elif name.endswith("scales"):
            k = buf.shape[-2] * 128
            base = 1.0 / (4.6 * k ** 0.5)
            buf.copy_((base * (0.5 + torch.rand(buf.shape, generator=g))).to(buf.dtype))
    for name, p in model.named_parameters():
        if name.endswith("norm"):
            p.copy_((1.0 + 0.1 * torch.randn(p.shape, generator=g)).to(p.dtype))
        elif name.endswith("conv_w"):
            p.copy_((0.5 * torch.randn(p.shape, generator=g)).to(p.dtype))
        else:                                   # router / beta_proj
            p.copy_((0.05 * torch.randn(p.shape, generator=g)).to(p.dtype))


def make_state(cfg, L, device, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = cfg.kda_heads * cfg.kda_head_dim
    st = []
    for kind in cfg.pattern:
        if kind == "K":
            st.append({
                "S": (0.05 * torch.randn(cfg.kda_heads, cfg.kda_head_dim, cfg.kda_head_dim,
                                         generator=g)).to(device),
                "cq": torch.randn(3, c, generator=g).to(device, torch.bfloat16),
                "ck": torch.randn(3, c, generator=g).to(device, torch.bfloat16),
                "cv": torch.randn(3, c, generator=g).to(device, torch.bfloat16),
            })
        else:
            st.append({
                "c_kv": torch.randn(L, cfg.kv_lora, generator=g).to(device, torch.bfloat16),
                "k_rope": torch.randn(L, cfg.qk_rope, generator=g).to(device, torch.bfloat16),
            })
    hidden = torch.randn(cfg.hidden, generator=g).to(device, torch.bfloat16)
    return hidden, st


def clone_state(state):
    return [{k: v.clone() for k, v in s.items()} for s in state]
