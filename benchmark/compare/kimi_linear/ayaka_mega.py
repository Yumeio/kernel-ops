"""Host side of ayaka_mega.cu: weight repacking + one ctypes launch per decode step.

`Model` has the same parameter/buffer names and the same `step(hidden, state)`
contract as kimi_mega_triton.Model, so bench.py can swap them.
"""
from __future__ import annotations

import ctypes
import os
import subprocess

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ayaka_mega.cu")
BUILD = os.path.abspath(os.path.join(HERE, "..", "..", "..", "build", "kimi_linear"))
LIB = os.path.join(BUILD, "libayaka_mega.so")

D, H, C4, KVL, QKN, QKR, VH, QH, KVBW, MI, NEXP, NBLK = 2304, 32, 4096, 512, 128, 64, 128, 192, 256, 1024, 64, 4
NIB = (0, 2, 4, 6, 1, 3, 5, 7)   # nibble slot of contraction element j in a packed word

_LAYOUT_NAMES = (
    "U_TOT UW WB_TOT FC_TOT SC_TOT CB_TOT PMAX NW "
    "A_KDA U_KDA U_QKVG A_MLA U_QKVA U_KVBN U_KVBV A_MOE U_MOE U_EXP U_GU "
    "WB_CW WB_BW WB_RW WB_NRM FC_INVF FC_SN8 FC_SNZ "
    "SC_ACCQ SC_HA SC_HM SC_O1 SC_BETA SC_RLOG SC_ACCGU SC_ACCQA SC_O1M SC_AM SC_AL "
    "CB_QAR CB_APC QROW NBAR").split()


class Params(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in
                ("WA", "WB", "FC", "SC", "CB", "BAR", "HIN", "HOUT", "SB", "CVI", "CVO", "CACHE")] + \
               [(n, ctypes.c_longlong) for n in ("L", "bar0", "sstop", "cw", "nch")]


_lib = None
LAY = None


def _load():
    global _lib, LAY
    if _lib is not None:
        return _lib
    os.makedirs(BUILD, exist_ok=True)
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        subprocess.check_call(["nvcc", "-O3", "-arch=sm_90a", "-shared", "-Xcompiler", "-fPIC",
                               "-o", LIB, SRC])
    _lib = ctypes.CDLL(LIB)
    _lib.ayk_step.argtypes = [ctypes.POINTER(Params), ctypes.c_int]
    _lib.ayk_step.restype = ctypes.c_int
    _lib.ayk_init.restype = ctypes.c_int
    buf = (ctypes.c_longlong * len(_LAYOUT_NAMES))()
    _lib.ayk_layout(buf)
    LAY = dict(zip(_LAYOUT_NAMES, list(buf)))
    return _lib


# --------------------------------------------------------------------------- #
# module skeletons: identical parameter/buffer names to the reference
# --------------------------------------------------------------------------- #
class QuantLinear(nn.Module):
    def __init__(self, in_f, out_f, group=128):
        super().__init__()
        self.in_f, self.out_f, self.group = in_f, out_f, group
        self.register_buffer("w_q", torch.zeros(in_f // 2, out_f, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(in_f // group, out_f, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(in_f // group, out_f, dtype=torch.bfloat16))


class QuantExperts(nn.Module):
    def __init__(self, n, in_f, out_f, group=128):
        super().__init__()
        self.n, self.in_f, self.out_f, self.group = n, in_f, out_f, group
        self.register_buffer("w_q", torch.zeros(n, in_f // 2, out_f, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(n, in_f // group, out_f, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(n, in_f // group, out_f, dtype=torch.bfloat16))


class KDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, c = cfg.hidden, cfg.kda_heads * cfg.kda_head_dim
        self.q_proj = QuantLinear(d, c, cfg.group)
        self.k_proj = QuantLinear(d, c, cfg.group)
        self.v_proj = QuantLinear(d, c, cfg.group)
        self.g_proj = QuantLinear(d, c, cfg.group)
        self.beta_proj = nn.Linear(d, cfg.kda_heads, bias=False, dtype=cfg.dtype)
        self.conv_w = nn.Parameter(torch.empty(3, c, cfg.short_conv, dtype=cfg.dtype))
        self.o_proj = QuantLinear(c, d, cfg.group)


class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden
        self.q_proj = QuantLinear(d, cfg.mla_heads * (cfg.qk_nope + cfg.qk_rope), cfg.group)
        self.kv_a = QuantLinear(d, cfg.kv_lora + cfg.qk_rope, cfg.group)
        self.kv_b = QuantLinear(cfg.kv_lora, cfg.mla_heads * (cfg.qk_nope + cfg.v_head), cfg.group)
        self.o_proj = QuantLinear(cfg.mla_heads * cfg.v_head, d, cfg.group)


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, m, E = cfg.hidden, cfg.moe_inter, cfg.n_experts
        self.router = nn.Linear(d, E, bias=False, dtype=cfg.dtype)
        self.gate = QuantExperts(E, d, m, cfg.group)
        self.up = QuantExperts(E, d, m, cfg.group)
        self.down = QuantExperts(E, m, d, cfg.group)
        self.s_gate = QuantExperts(cfg.n_shared, d, m, cfg.group)
        self.s_up = QuantExperts(cfg.n_shared, d, m, cfg.group)
        self.s_down = QuantExperts(cfg.n_shared, m, d, cfg.group)


class Block(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind
        self.attn_norm = nn.Parameter(torch.ones(cfg.hidden, dtype=cfg.dtype))
        self.moe_norm = nn.Parameter(torch.ones(cfg.hidden, dtype=cfg.dtype))
        self.attn = KDA(cfg) if kind == "K" else MLA(cfg)
        self.moe = MoE(cfg)


# --------------------------------------------------------------------------- #
# weight repacking: unit = [17 rows][128 cols] int32, arena = [tile][kg] units
# --------------------------------------------------------------------------- #
def _nib(wq, K):
    v = torch.empty((K, wq.shape[1]), dtype=torch.int32, device=wq.device)
    v[0::2] = (wq & 0xF).to(torch.int32)
    v[1::2] = ((wq >> 4) & 0xF).to(torch.int32)
    return v


def _szrow(scales, zeros):
    """{bf16 scale bits | int8 zero << 16} per (group, column)."""
    s = scales.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    z = zeros.float().round().to(torch.int32) & 0xFF
    return s | (z << 16)


def _words(v):
    """(..., 8) nibbles of 8 consecutive contraction elements -> packed word."""
    w = torch.zeros(v.shape[:-1], dtype=torch.int32, device=v.device)
    for j in range(8):
        w |= v[..., j] << (4 * NIB[j])
    return w


def pack_units(wq, scales, zeros, K, N):
    """(K//2, N) uint8 + (K//128, N) scale/zero -> flat int32 [ntile][nkg][17][128]."""
    nkg, ntile = K // 128, (N + 127) // 128
    npad = ntile * 128
    v = torch.zeros((K, npad), dtype=torch.int32, device=wq.device)
    v[:, :N] = _nib(wq, K)
    w = _words(v.view(nkg, 16, 8, npad).permute(0, 1, 3, 2))          # [nkg][16][npad]
    w = w.view(nkg, 16, ntile, 128).permute(2, 0, 1, 3)               # [ntile][nkg][16][128]
    sz = torch.zeros((nkg, npad), dtype=torch.int32, device=wq.device)
    sz[:, :N] = _szrow(scales, zeros)
    sz = sz.view(nkg, ntile, 128).permute(1, 0, 2)[:, :, None, :]
    return torch.cat([sz, w], 2).reshape(-1)


def pack_kvb(wq, scales, zeros):
    """kv_b -> (nope units [h][g] transposed, value units [h][g], sn8, snz)."""
    dev = wq.device
    v = _nib(wq, KVL).view(KVL, H, KVBW)
    # nope half, contraction over d: unit (h, g) rows [m][c], word = d 8m..8m+7
    vn = v[:, :, :QKN].permute(1, 0, 2).reshape(H, 4, 128, 16, 8)     # (h, g, c, m, j)
    wn = _words(vn).permute(0, 1, 3, 2)                               # (h, g, m, c)
    un = torch.cat([torch.zeros((H, 4, 1, 128), dtype=torch.int32, device=dev), wn], 2)
    # value half, contraction over c: unit (h, g) rows [m][dv]
    vv = v[:, :, QKN:].reshape(4, 16, 8, H, VH).permute(3, 0, 1, 4, 2)  # (h, g, m, dv, j)
    wv = _words(vv)
    sz = _szrow(scales, zeros).view(4, H, KVBW)[:, :, QKN:].permute(1, 0, 2)[:, :, None, :]
    uv = torch.cat([sz, wv], 2)
    s = scales.float().view(4, H, KVBW)[:, :, :QKN].permute(1, 0, 2).contiguous()
    z = zeros.float().view(4, H, KVBW)[:, :, :QKN].permute(1, 0, 2).contiguous()
    return un.reshape(-1), uv.reshape(-1), s.reshape(-1), (s * z).reshape(-1)


# --------------------------------------------------------------------------- #
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg, k) for k in cfg.pattern)
        self._ready = False
        self._sstop = 99
        self.min_chunk_rows = 32

    @torch.no_grad()
    def _build(self):
        lib = _load()
        dev = self.blocks[0].attn_norm.device
        cfg = self.cfg
        Y = LAY
        UW = Y["UW"]
        wa = torch.zeros(Y["U_TOT"] * UW, dtype=torch.int32, device=dev)
        wb = torch.zeros(Y["WB_TOT"], dtype=torch.bfloat16, device=dev)
        fc = torch.zeros(Y["FC_TOT"], dtype=torch.float32, device=dev)

        def put(unit, t):
            wa[unit * UW: unit * UW + t.numel()] = t

        for b in range(3):
            k = self.blocks[b].attn
            base = Y["A_KDA"] + b * Y["U_KDA"]
            put(base, pack_units(
                torch.cat([k.q_proj.w_q, k.k_proj.w_q, k.v_proj.w_q, k.g_proj.w_q], 1),
                torch.cat([k.q_proj.scales, k.k_proj.scales, k.v_proj.scales, k.g_proj.scales], 1),
                torch.cat([k.q_proj.zeros, k.k_proj.zeros, k.v_proj.zeros, k.g_proj.zeros], 1),
                D, 4 * C4))
            put(base + Y["U_QKVG"], pack_units(k.o_proj.w_q, k.o_proj.scales, k.o_proj.zeros, C4, D))
            o = Y["WB_CW"] + b * 12 * C4
            wb[o:o + 12 * C4] = k.conv_w.detach().permute(0, 2, 1).reshape(-1)
            o = Y["WB_BW"] + b * H * D
            wb[o:o + H * D] = k.beta_proj.weight.detach().reshape(-1)

        m = self.blocks[3].attn
        base = Y["A_MLA"]
        put(base, pack_units(torch.cat([m.q_proj.w_q, m.kv_a.w_q], 1),
                             torch.cat([m.q_proj.scales, m.kv_a.scales], 1),
                             torch.cat([m.q_proj.zeros, m.kv_a.zeros], 1), D, H * QH + KVL + QKR))
        un, uv, sn8, snz = pack_kvb(m.kv_b.w_q, m.kv_b.scales, m.kv_b.zeros)
        put(base + Y["U_QKVA"], un)
        put(base + Y["U_QKVA"] + Y["U_KVBN"], uv)
        put(base + Y["U_QKVA"] + Y["U_KVBN"] + Y["U_KVBV"],
            pack_units(m.o_proj.w_q, m.o_proj.scales, m.o_proj.zeros, C4, D))
        fc[Y["FC_SN8"]:Y["FC_SN8"] + sn8.numel()] = sn8
        fc[Y["FC_SNZ"]:Y["FC_SNZ"] + snz.numel()] = snz
        fc[Y["FC_INVF"]:Y["FC_INVF"] + QKR // 2] = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, QKR, 2, device=dev, dtype=torch.float32) / QKR))

        for b in range(NBLK):
            mo = self.blocks[b].moe
            for e in range(NEXP + 1):
                g, u, dn, i = (mo.gate, mo.up, mo.down, e) if e < NEXP else (mo.s_gate, mo.s_up, mo.s_down, 0)
                base = Y["A_MOE"] + b * Y["U_MOE"] + e * Y["U_EXP"]
                put(base, pack_units(torch.cat([g.w_q[i], u.w_q[i]], 1),
                                     torch.cat([g.scales[i], u.scales[i]], 1),
                                     torch.cat([g.zeros[i], u.zeros[i]], 1), D, 2 * MI))
                put(base + Y["U_GU"], pack_units(dn.w_q[i], dn.scales[i], dn.zeros[i], MI, D))
            o = Y["WB_RW"] + b * NEXP * D
            wb[o:o + NEXP * D] = mo.router.weight.detach().reshape(-1)
            o = Y["WB_NRM"] + b * 2 * D
            wb[o:o + D] = self.blocks[b].attn_norm.detach()
            wb[o + D:o + 2 * D] = self.blocks[b].moe_norm.detach()

        self._wa, self._wb, self._fc = wa, wb, fc
        self._sc = torch.zeros(Y["SC_TOT"], dtype=torch.float32, device=dev)
        self._cb = torch.zeros(Y["CB_TOT"], dtype=torch.bfloat16, device=dev)
        self._bar = torch.zeros(64, dtype=torch.int32, device=dev)
        self._barbase = 0
        self._hout = torch.zeros(D, dtype=torch.bfloat16, device=dev)
        self._sbuf = torch.zeros(3, H, 128, 128, dtype=torch.float32, device=dev)
        self._cv = torch.zeros(2, 9, 3, C4, dtype=torch.bfloat16, device=dev)
        self._par = 0
        self._cache = None
        self._cap = 0
        self._len = -1
        torch.cuda.synchronize()
        self._nsm = lib.ayk_init()
        assert self._nsm > 0, f"ayk_init failed: {self._nsm}"
        assert self._nsm <= Y["PMAX"]
        self._p = Params()
        P = self._p
        P.WA, P.WB, P.FC = wa.data_ptr(), wb.data_ptr(), fc.data_ptr()
        P.SC, P.CB, P.BAR = self._sc.data_ptr(), self._cb.data_ptr(), self._bar.data_ptr()
        P.HOUT, P.SB = self._hout.data_ptr(), self._sbuf.data_ptr()
        self._pref = ctypes.byref(P)
        self._step = lib.ayk_step
        self._cvp = (self._cv[0].data_ptr(), self._cv[1].data_ptr())
        self._ready = True

    @torch.no_grad()
    def _adopt(self, state, ki, mi, L):
        """take a state we did not produce into our own buffers (first step / cache growth)."""
        ck, kr = state[mi]["c_kv"], state[mi]["k_rope"]
        if self._cache is None or L + 1 > self._cap:
            cap = L + 4096
            cache = torch.empty(cap, KVL + QKR, dtype=torch.bfloat16, device=ck.device)
            self._cap = cap
        else:
            cache = self._cache
        if L:
            ckc, krc = ck.clone(), kr.clone()        # may alias the old cache
            cache[:L, :KVL] = ckc
            cache[:L, KVL:] = krc
        self._cache = cache
        self._p.CACHE = cache.data_ptr()
        if state[ki[0]]["S"].data_ptr() != self._sbuf.data_ptr():
            for j, i in enumerate(ki):
                self._sbuf[j].copy_(state[i]["S"])
                for n, name in enumerate(("cq", "ck", "cv")):
                    self._cv[self._par, j * 3 + n].copy_(state[i][name])

    @torch.no_grad()
    def step(self, hidden, state):
        if not self._ready:
            self._build()
        pat = self.cfg.pattern
        ki = [i for i, k in enumerate(pat) if k == "K"]
        mi = pat.index("M")
        ck = state[mi]["c_kv"]
        L = int(ck.shape[0])
        if (self._cache is None or L + 1 > self._cap or L != self._len
                or ck.data_ptr() != self._cache.data_ptr()
                or state[ki[0]]["S"].data_ptr() != self._sbuf.data_ptr()):
            self._adopt(state, ki, mi, L)
        if self._barbase > (1 << 30):
            torch.cuda.synchronize()
            self._bar.zero_()
            self._barbase = 0
        nsm = self._nsm
        cw = max(self.min_chunk_rows, (((L + nsm) // nsm) + 15) // 16 * 16)
        P = self._p
        P.HIN = hidden.data_ptr()
        P.CVI, P.CVO = self._cvp[self._par], self._cvp[1 - self._par]
        P.L, P.bar0, P.sstop, P.cw, P.nch = L, self._barbase, self._sstop, cw, (L + cw) // cw
        err = self._step(self._pref, nsm)
        if err:
            raise RuntimeError(f"ayk_step: CUDA error {err}")
        self._barbase += LAY["NBAR"] * nsm
        self._par = 1 - self._par
        out = self._cv[self._par]
        for j, i in enumerate(ki):
            st = state[i]
            st["S"] = self._sbuf[j]
            st["cq"], st["ck"], st["cv"] = out[j * 3], out[j * 3 + 1], out[j * 3 + 2]
        self._len = L + 1
        state[mi]["c_kv"] = self._cache[:L + 1, :KVL]
        state[mi]["k_rope"] = self._cache[:L + 1, KVL:]
        return self._hout, state
