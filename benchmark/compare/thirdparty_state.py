#!/usr/bin/env python3
"""Third-party adapters for the `state` family (B2 state cache, B9 recurrent).

Dev-only: not built, linked, tested or shipped. The module follows the adapter
contract of `benchmark/compare/thirdparty.py`:

    adapter(case, device) -> (fn, note) | (None, reason)

`case` is one entry of the JSON emitted by `benchmark/bench_state --dump-cases`
(`id`, `op`, `dtype`, `rows`, `cols`, `bytes`, `flops`, `warmup`, `iters`) and
`device` is the CUDA device to allocate on. `fn` is a zero-argument callable
that runs exactly one invocation of the library operator; every tensor is built
before it is returned, so the timed region holds the library call only (plus the
output allocation the library performs internally).

Libraries, pinned by `tools/dev/3rd_manifest.tsv`:

    fla             flash-linear-attention v0.6.0 (needs fla-core==0.6.0)
    mamba_ssm       mamba v2.3.2
    causal_conv1d   causal-conv1d v1.7.0

A case a library cannot express returns `(None, reason)`; it never runs a
different operator under the case id. Packages are imported inside the adapter
and a missing package returns `(None, ...)` instead of raising.

The head geometry that the case id does not carry is fixed by
`benchmark/bench_state.cu`; the defaults below are the emitted values:

    conv1d             dim 8192, width 4, SiLU; 2 x 2048 tokens or 64 rows
    ssm_update         Mamba-2 H 32, P 64, N 128, G 8, 64 rows, dt softplus
    selective_scan     Mamba-1 D 8192, N 16, ngroups 1, 2 x 2048 tokens
    ssd_chunked_scan   Mamba-2 H 32, P 64, N 128, G 8, chunk 256, 2 x 2048
    lightning          H 64, Dk = Dv = 128, explicit slope, 2 x 2048 / 64 rows
    delta rule         Hk 16 (GDN) or 32 (KDA), Hv 32, D 128, QK L2 norm;
                       64 rows (decode) or 2 x 2048 tokens (prefill)

`--synthetic` runs one tiny case per supported operator (needs CUDA and the
package); the mapping print does not touch CUDA.

    python benchmark/compare/thirdparty_state.py
    python benchmark/compare/thirdparty_state.py --cases out/state.raw.json
    python benchmark/compare/thirdparty_state.py --synthetic --impl fla
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys

# ---------------------------------------------------------------------------
# Case geometry (mirrors benchmark/bench_state.cu)
# ---------------------------------------------------------------------------
PREFILL_BATCH = 2          # conv1d_fwd, selective_scan, ssd, lightning, delta_prefill
CONV_WIDTH = 4             # Qwen3-Next short conv
MAMBA2_NGROUPS = 8         # case_ssm / case_ssd: G = 8
SSD_CHUNK = 256            # case_ssd: CH = 256
LIGHTNING_LAYER_IDX = 0    # FLA derives the slope from (layer_idx, num_layers)
LIGHTNING_NUM_LAYERS = 1

DELTA_DEFAULT_HEAD_DIM = 128
LIGHTNING_DEFAULT_HEADS = 64
SSM_DEFAULT = (32, 64, 128)   # H, P, N
SSD_DEFAULT = (32, 64, 128)   # H, P, N
SCAN_DEFAULT_DSTATE = 16

STATE_CACHE_OPS = ("state_copy", "state_gather", "state_scatter", "state_zero",
                   "state_commit")
CONV_OPS = ("conv1d_fwd", "conv1d_update")
MAMBA_OPS = ("ssm_update", "selective_scan", "ssd_chunked_scan")
FLA_OPS = ("delta_decode", "delta_prefill", "delta_prefill_tc",
           "lightning_prefill", "linear_decode")

HANDLED = {
    "fla": FLA_OPS,
    "mamba_ssm": MAMBA_OPS,
    "causal_conv1d": CONV_OPS,
}

_DTYPES = {"bfloat16", "float16", "float32"}


# ---------------------------------------------------------------------------
# small helpers (no torch import at module scope)
# ---------------------------------------------------------------------------
def _int(text: str, pattern: str, default: int) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default


def _torch_dtype(torch, name: str):
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return table.get(name, torch.bfloat16)


def _prefill_rows(case: dict) -> tuple[int, int]:
    """(batch, seqlen) of a flat-token prefill case."""
    total = int(case["rows"])
    batch = PREFILL_BATCH if total % PREFILL_BATCH == 0 else 1
    return batch, max(1, total // batch)


def _reason(impl: str, op: str) -> str:
    """Why `impl` does not run `op`, phrased for the skip table."""
    if op in STATE_CACHE_OPS:
        return ("engine-owned paged state pool (byte copy/gather/scatter/zero/"
                "commit); no third-party kernel")
    if impl == "fla":
        if op in CONV_OPS:
            return "causal_conv1d owns conv1d_fwd/conv1d_update; fla has no standalone conv1d"
        if op in MAMBA_OPS:
            return "mamba_ssm owns the Mamba-1/2 scan; fla has no selective scan/SSD kernel"
        if op == "delta_gating":
            return ("fla fuses decay gating into its delta-rule kernels and has no "
                    "one-call g+beta gate kernel")
        if op == "fused_delta_decode":
            return ("short conv + gate + delta recurrence + gated RMSNorm is a "
                    "kops-only chain; fla needs 4+ separate kernels")
        return f"fla has no {op} operator"
    if impl == "mamba_ssm":
        if op in CONV_OPS:
            return ("mamba_ssm reuses causal_conv1d; the conv1d case belongs to the "
                    "causal_conv1d adapter")
        return "mamba_ssm has no gated-delta-rule/lightning kernel; fla owns those"
    if impl == "causal_conv1d":
        return "causal_conv1d only implements conv1d_fwd/conv1d_update"
    return f"{impl} has no adapter for {op}"


def _floating(case: dict) -> bool:
    return case.get("dtype", "bfloat16") in _DTYPES


# ---------------------------------------------------------------------------
# causal_conv1d v1.7.0: conv1d_fwd / conv1d_update
# ---------------------------------------------------------------------------
def causal_conv1d_adapter(case: dict, device):
    op = case["op"]
    if op not in CONV_OPS:
        return None, _reason("causal_conv1d", op)
    if not _floating(case):
        return None, f"dtype {case['dtype']} is not supported by causal_conv1d"
    try:
        import torch
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError as exc:
        return None, f"causal_conv1d not installed ({exc})"

    dt = _torch_dtype(torch, case["dtype"])
    dim = int(case["cols"])
    width = CONV_WIDTH
    weight = torch.randn(dim, width, device=device, dtype=dt) * 0.1
    bias = torch.randn(dim, device=device, dtype=dt) * 0.1

    if op == "conv1d_fwd":
        batch, seqlen = _prefill_rows(case)
        # causal_conv1d wants channel-last storage: the dim axis must be the
        # contiguous one, both for x and for the (batch, dim, width-1) state.
        x = torch.randn(batch, seqlen, dim, device=device, dtype=dt).transpose(1, 2)
        state = torch.randn(batch, width - 1, dim, device=device, dtype=dt).transpose(1, 2)
        final = torch.empty_like(state)

        def run_fwd():
            return causal_conv1d_fn(x, weight, bias=bias, initial_states=state,
                                    return_final_states=True, final_states_out=final,
                                    activation="silu")

        return run_fwd, ("causal_conv1d_fn (silu, initial+final state, "
                         "channel-last buffers)")

    batch = int(case["rows"])
    x = torch.randn(batch, dim, device=device, dtype=dt)
    state = torch.randn(batch, dim, width - 1, device=device, dtype=dt)
    slots = torch.arange(batch, device=device, dtype=torch.int32)

    def run_update():
        return causal_conv1d_update(x, state, weight, bias, activation="silu",
                                    conv_state_indices=slots)

    return run_update, "causal_conv1d_update (silu, slot-indexed conv state)"


# ---------------------------------------------------------------------------
# mamba_ssm v2.3.2: ssm_update / selective_scan / ssd_chunked_scan
# ---------------------------------------------------------------------------
def mamba_adapter(case: dict, device):
    op = case["op"]
    if op not in MAMBA_OPS:
        return None, _reason("mamba_ssm", op)
    if not _floating(case):
        return None, f"dtype {case['dtype']} is not supported by mamba_ssm"
    try:
        import torch
    except ImportError as exc:
        return None, f"torch not installed ({exc})"

    dt = _torch_dtype(torch, case["dtype"])
    cid = case["id"]

    if op == "ssm_update":
        try:
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        except ImportError as exc:
            return None, f"mamba_ssm selective_state_update unavailable ({exc})"
        batch = int(case["rows"])
        heads = _int(cid, r"\.H(\d+)P\d+N\d+", SSM_DEFAULT[0])
        headdim = _int(cid, r"\.H\d+P(\d+)N\d+", SSM_DEFAULT[1])
        dstate = _int(cid, r"\.H\d+P\d+N(\d+)", SSM_DEFAULT[2])
        ngroups = MAMBA2_NGROUPS
        state = torch.zeros(batch, heads, headdim, dstate, device=device,
                            dtype=torch.float32)
        x = torch.randn(batch, heads, headdim, device=device, dtype=dt) * 0.1
        dt_in = torch.randn(batch, heads, device=device, dtype=dt) * 0.1
        # Zero-stride views select the kernel's TIE_HDIM path: A is one decay
        # per head and dt one step per (batch, head), exactly like the kops case.
        decay = -torch.rand(heads, device=device, dtype=torch.float32)
        a_view = decay.view(heads, 1, 1).expand(heads, headdim, dstate)
        dt_view = dt_in.view(batch, heads, 1).expand(batch, heads, headdim)
        dt_bias = torch.zeros(heads, 1, device=device,
                              dtype=torch.float32).expand(heads, headdim)
        d_view = torch.randn(heads, device=device,
                             dtype=torch.float32).view(heads, 1).expand(heads, headdim)
        b_in = torch.randn(batch, ngroups, dstate, device=device, dtype=dt) * 0.1
        c_in = torch.randn(batch, ngroups, dstate, device=device, dtype=dt) * 0.1
        z = torch.randn(batch, heads, headdim, device=device, dtype=dt) * 0.1
        slots = torch.arange(batch, device=device, dtype=torch.int32)

        def run_state():
            return selective_state_update(state, x, dt_view, a_view, b_in, c_in, d_view,
                                          z=z, dt_bias=dt_bias, dt_softplus=True,
                                          state_batch_indices=slots)

        return run_state, ("mamba_ssm selective_state_update (TIE_HDIM, dt softplus, "
                           "slot indices, z gate)")

    if op == "selective_scan":
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError as exc:
            return None, f"mamba_ssm selective_scan_cuda extension missing ({exc})"
        batch, seqlen = _prefill_rows(case)
        dim = int(case["cols"])
        dstate = _int(cid, r"\.D\d+N(\d+)", SCAN_DEFAULT_DSTATE)
        u = torch.randn(batch, dim, seqlen, device=device, dtype=dt) * 0.1
        delta = torch.randn(batch, dim, seqlen, device=device, dtype=dt) * 0.1
        a_in = -torch.rand(dim, dstate, device=device, dtype=torch.float32)
        b_in = torch.randn(batch, 1, dstate, seqlen, device=device, dtype=dt) * 0.1
        c_in = torch.randn(batch, 1, dstate, seqlen, device=device, dtype=dt) * 0.1
        d_in = torch.randn(dim, device=device, dtype=torch.float32)
        z = torch.randn(batch, dim, seqlen, device=device, dtype=dt) * 0.1

        def run_scan():
            return selective_scan_fn(u, delta, a_in, b_in, c_in, d_in, z=z,
                                     delta_bias=None, delta_softplus=True,
                                     return_last_state=True)

        return run_scan, ("mamba_ssm selective_scan_fn (CUDA ext, dt softplus, z gate; "
                          "v2.3.2 has no initial-state argument)")

    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError as exc:
        return None, f"mamba_ssm mamba_chunk_scan_combined unavailable ({exc})"
    batch, seqlen = _prefill_rows(case)
    heads = _int(cid, r"\.H(\d+)P\d+N\d+", SSD_DEFAULT[0])
    headdim = _int(cid, r"\.H\d+P(\d+)N\d+", SSD_DEFAULT[1])
    dstate = _int(cid, r"\.H\d+P\d+N(\d+)", SSD_DEFAULT[2])
    ngroups = MAMBA2_NGROUPS
    x = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dt) * 0.1
    dt_in = torch.randn(batch, seqlen, heads, device=device, dtype=dt) * 0.1
    a_in = -torch.rand(heads, device=device, dtype=torch.float32)
    b_in = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dt) * 0.1
    c_in = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dt) * 0.1
    d_in = torch.randn(heads, device=device, dtype=torch.float32)
    z = torch.randn(batch, seqlen, heads, headdim, device=device, dtype=dt) * 0.1
    state = torch.zeros(batch, heads, headdim, dstate, device=device,
                        dtype=torch.float32)

    def run_ssd():
        return mamba_chunk_scan_combined(x, dt_in, a_in, b_in, c_in, SSD_CHUNK,
                                         D=d_in, z=z, initial_states=state,
                                         dt_softplus=True, return_final_states=True)

    return run_ssd, ("mamba_ssm mamba_chunk_scan_combined (chunk 256, dt softplus, "
                     "initial+final state)")


# ---------------------------------------------------------------------------
# FLA v0.6.0: GDN / KDA delta rule, lightning attention
# ---------------------------------------------------------------------------
def fla_adapter(case: dict, device):
    op = case["op"]
    if op not in FLA_OPS:
        return None, _reason("fla", op)
    if not _floating(case):
        return None, f"dtype {case['dtype']} is not supported by fla"
    try:
        import torch
        from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                              fused_recurrent_gated_delta_rule)
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
        from fla.ops.lightning_attn import (chunk_lightning_attn,
                                            fused_recurrent_lightning_attn)
    except ImportError as exc:
        return None, f"fla not installed ({exc})"

    dt = _torch_dtype(torch, case["dtype"])
    cid = case["id"]
    kda = ".kda." in cid
    head_dim = _int(cid, r"\.H\d+D(\d+)",
                    _int(cid, r"\.D(\d+)(?=[.BN]|$)", DELTA_DEFAULT_HEAD_DIM))
    scale = head_dim ** -0.5     # kops uses 0.088 = 1/sqrt(128)

    if op in ("delta_decode", "delta_prefill", "delta_prefill_tc"):
        v_heads = max(1, int(case["cols"]) // head_dim)
        k_heads = v_heads if kda else max(1, v_heads // 2)
        decode = op == "delta_decode"
        batch, seqlen = (int(case["rows"]), 1) if decode else _prefill_rows(case)
        q = torch.randn(batch, seqlen, k_heads, head_dim, device=device, dtype=dt) * 0.1
        k = torch.randn(batch, seqlen, k_heads, head_dim, device=device, dtype=dt) * 0.1
        v = torch.randn(batch, seqlen, v_heads, head_dim, device=device, dtype=dt) * 0.1
        beta = torch.rand(batch, seqlen, v_heads, device=device, dtype=torch.float32)
        if kda:
            g = -torch.rand(batch, seqlen, v_heads, head_dim, device=device,
                            dtype=torch.float32) * 0.1
        else:
            g = -torch.rand(batch, seqlen, v_heads, device=device,
                            dtype=torch.float32) * 0.1
        state = torch.zeros(batch, v_heads, head_dim, head_dim, device=device,
                            dtype=torch.float32)
        if decode and kda:
            def run_kda_decode():
                return fused_recurrent_kda(q, k, v, g, beta, scale=scale,
                                           initial_state=state, output_final_state=True,
                                           use_qk_l2norm_in_kernel=True)

            return run_kda_decode, ("fla fused_recurrent_kda (per-channel decay, QK L2 "
                                    "norm, fp32 state)")
        if decode:
            def run_gdn_decode():
                return fused_recurrent_gated_delta_rule(
                    q, k, v, g, beta, scale=scale, initial_state=state,
                    output_final_state=True, use_qk_l2norm_in_kernel=True)

            return run_gdn_decode, ("fla fused_recurrent_gated_delta_rule (GDN, QK L2 "
                                    "norm, fp32 state)")
        if kda:
            def run_kda_prefill():
                return chunk_kda(q, k, v, g, beta, scale=scale, initial_state=state,
                                 output_final_state=True, use_qk_l2norm_in_kernel=True)

            note = "fla chunk_kda (per-channel decay, QK L2 norm, fp32 state)"
            if op == "delta_prefill_tc":
                note += ("; FLA has one chunked algorithm, so both kops math modes "
                         "share it")
            return run_kda_prefill, note
        note = "fla chunk_gated_delta_rule (GDN, QK L2 norm, fp32 state)"
        if op == "delta_prefill_tc":
            note += ("; FLA has one chunked algorithm, so both kops math modes share "
                     "it")

        def run_gdn_prefill():
            return chunk_gated_delta_rule(q, k, v, g, beta, scale=scale,
                                          initial_state=state, output_final_state=True,
                                          use_qk_l2norm_in_kernel=True)

        return run_gdn_prefill, note

    heads = _int(cid, r"\.H(\d+)D\d+", LIGHTNING_DEFAULT_HEADS)
    if op == "lightning_prefill":
        batch, seqlen = _prefill_rows(case)
        q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dt) * 0.1
        k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dt) * 0.1
        v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dt) * 0.1
        state = torch.zeros(batch, heads, head_dim, head_dim, device=device,
                            dtype=torch.float32)

        def run_lightning_prefill():
            return chunk_lightning_attn(q, k, v, LIGHTNING_LAYER_IDX,
                                        LIGHTNING_NUM_LAYERS, scale=scale,
                                        initial_state=state, output_final_state=True)

        return run_lightning_prefill, (
            "fla chunk_lightning_attn (slope derived from layer_idx/num_layers; the "
            "kops bench fills its explicit slope with 0)")

    batch = int(case["rows"])
    q = torch.randn(batch, 1, heads, head_dim, device=device, dtype=dt) * 0.1
    k = torch.randn(batch, 1, heads, head_dim, device=device, dtype=dt) * 0.1
    v = torch.randn(batch, 1, heads, head_dim, device=device, dtype=dt) * 0.1
    state = torch.zeros(batch, heads, head_dim, head_dim, device=device,
                        dtype=torch.float32)

    def run_linear_decode():
        return fused_recurrent_lightning_attn(q, k, v, LIGHTNING_LAYER_IDX,
                                              LIGHTNING_NUM_LAYERS, scale=scale,
                                              initial_state=state,
                                              output_final_state=True)

    return run_linear_decode, (
        "fla fused_recurrent_lightning_attn (slope derived from layer_idx/num_layers)")


IMPLS = {
    "fla": fla_adapter,
    "mamba_ssm": mamba_adapter,
    "causal_conv1d": causal_conv1d_adapter,
}


# ---------------------------------------------------------------------------
# mapping / self-check
# ---------------------------------------------------------------------------
# Emitted by `bench_state --dump-cases`; used when --cases is not given so the
# mapping print never needs the binary, a GPU or the packages.
DEFAULT_CASES = [
    {"id": "state_copy.2pools.64items", "op": "state_copy", "dtype": "uint8",
     "rows": 64, "cols": 6144},
    {"id": "state_gather.2pools.64items", "op": "state_gather", "dtype": "uint8",
     "rows": 64, "cols": 6144},
    {"id": "state_scatter.2pools.64items", "op": "state_scatter", "dtype": "uint8",
     "rows": 64, "cols": 6144},
    {"id": "state_zero.2pools.64items", "op": "state_zero", "dtype": "uint8",
     "rows": 64, "cols": 6144},
    {"id": "state_commit.2pools.b64.d8", "op": "state_commit", "dtype": "uint8",
     "rows": 64, "cols": 8},
    {"id": "conv1d_fwd.bf16.qwen3next.T4096", "op": "conv1d_fwd", "dtype": "bfloat16",
     "rows": 4096, "cols": 8192},
    {"id": "conv1d_update.bf16.qwen3next.B64", "op": "conv1d_update",
     "dtype": "bfloat16", "rows": 64, "cols": 8192},
    {"id": "ssm_update.bf16.mamba2.H32P64N128.B64", "op": "ssm_update",
     "dtype": "bfloat16", "rows": 64, "cols": 2048},
    {"id": "selective_scan.bf16.falcon_mamba.D8192N16.T4096", "op": "selective_scan",
     "dtype": "bfloat16", "rows": 4096, "cols": 8192},
    {"id": "ssd_chunked_scan.bf16.mamba2.H32P64N128.T4096", "op": "ssd_chunked_scan",
     "dtype": "bfloat16", "rows": 4096, "cols": 2048},
    {"id": "lightning_prefill.bf16.minimax.H64D128.T4096", "op": "lightning_prefill",
     "dtype": "bfloat16", "rows": 4096, "cols": 8192},
    {"id": "linear_decode.bf16.minimax.H64D128.B64", "op": "linear_decode",
     "dtype": "bfloat16", "rows": 64, "cols": 8192},
    {"id": "delta_decode.gdn.bf16.B64", "op": "delta_decode", "dtype": "bfloat16",
     "rows": 64, "cols": 4096},
    {"id": "fused_delta_decode.gdn.bf16.B64", "op": "fused_delta_decode",
     "dtype": "bfloat16", "rows": 64, "cols": 4096},
    {"id": "delta_prefill.gdn.fp32math.bf16.T4096", "op": "delta_prefill",
     "dtype": "bfloat16", "rows": 4096, "cols": 4096},
    {"id": "delta_prefill.gdn.tc.bf16.T4096", "op": "delta_prefill_tc",
     "dtype": "bfloat16", "rows": 4096, "cols": 4096},
    {"id": "delta_gating.gdn.bf16.T4096", "op": "delta_gating", "dtype": "bfloat16",
     "rows": 4096, "cols": 32},
    {"id": "delta_decode.kda.bf16.B64", "op": "delta_decode", "dtype": "bfloat16",
     "rows": 64, "cols": 4096},
    {"id": "fused_delta_decode.kda.bf16.B64", "op": "fused_delta_decode",
     "dtype": "bfloat16", "rows": 64, "cols": 4096},
    {"id": "delta_prefill.kda.fp32math.bf16.T4096", "op": "delta_prefill",
     "dtype": "bfloat16", "rows": 4096, "cols": 4096},
    {"id": "delta_prefill.kda.tc.bf16.T4096", "op": "delta_prefill_tc",
     "dtype": "bfloat16", "rows": 4096, "cols": 4096},
    {"id": "delta_gating.kda.bf16.T4096", "op": "delta_gating", "dtype": "bfloat16",
     "rows": 4096, "cols": 32},
]

# Small shapes for --synthetic; the ids carry the geometry the adapter parses.
SYNTHETIC_CASES = [
    {"id": "conv1d_fwd.bf16.qwen3next.D64.T8", "op": "conv1d_fwd",
     "dtype": "bfloat16", "rows": 8, "cols": 64},
    {"id": "conv1d_update.bf16.qwen3next.D64.B4", "op": "conv1d_update",
     "dtype": "bfloat16", "rows": 4, "cols": 64},
    {"id": "ssm_update.bf16.mamba2.H2P4N8.B4", "op": "ssm_update",
     "dtype": "bfloat16", "rows": 4, "cols": 8},
    {"id": "selective_scan.bf16.falcon_mamba.D64N8.T8", "op": "selective_scan",
     "dtype": "bfloat16", "rows": 8, "cols": 64},
    {"id": "ssd_chunked_scan.bf16.mamba2.H2P4N8.T8", "op": "ssd_chunked_scan",
     "dtype": "bfloat16", "rows": 8, "cols": 8},
    {"id": "lightning_prefill.bf16.minimax.H4D16.T8", "op": "lightning_prefill",
     "dtype": "bfloat16", "rows": 8, "cols": 64},
    {"id": "linear_decode.bf16.minimax.H4D16.B4", "op": "linear_decode",
     "dtype": "bfloat16", "rows": 4, "cols": 64},
    {"id": "delta_decode.gdn.bf16.D16.B4", "op": "delta_decode",
     "dtype": "bfloat16", "rows": 4, "cols": 32},
    {"id": "delta_prefill.gdn.fp32math.bf16.D16.T8", "op": "delta_prefill",
     "dtype": "bfloat16", "rows": 8, "cols": 32},
    {"id": "delta_prefill.kda.bf16.D16.T8", "op": "delta_prefill",
     "dtype": "bfloat16", "rows": 8, "cols": 32},
]


def load_cases(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_CASES
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    return doc["cases"] if isinstance(doc, dict) else doc


def package_state() -> dict[str, str]:
    """Importability of each backend, without importing it."""
    modules = {"fla": "fla", "mamba_ssm": "mamba_ssm",
               "causal_conv1d": "causal_conv1d"}
    out = {}
    for impl, module in modules.items():
        try:
            out[impl] = "installed" if importlib.util.find_spec(module) else "missing"
        except Exception as exc:  # noqa: BLE001 - namespace package edge cases
            out[impl] = f"missing ({type(exc).__name__})"
    return out


def print_mapping(cases: list[dict], impls: list[str]) -> None:
    id_width = max(len(case["id"]) for case in cases) + 2
    header = f"{'case id':<{id_width}}{'op':<20}" + "".join(
        f"{impl:<14}" for impl in impls)
    print(header)
    print("-" * len(header))
    for case in cases:
        row = f"{case['id']:<{id_width}}{case['op']:<20}"
        for impl in impls:
            row += f"{'yes' if case['op'] in HANDLED[impl] else '-':<14}"
        print(row)
    print()
    state = package_state()
    print("packages: " + "  ".join(f"{impl}={state[impl]}" for impl in impls))
    print()
    print("unsupported cases, grouped by reason:")
    for impl in impls:
        groups: dict[str, list[str]] = {}
        for case in cases:
            if case["op"] in HANDLED[impl]:
                continue
            reason = _reason(impl, case["op"])
            ops = groups.setdefault(reason, [])
            if case["op"] not in ops:
                ops.append(case["op"])
        if not groups:
            continue
        print(f"  {impl}:")
        for reason, ops in groups.items():
            print(f"    {', '.join(ops)}")
            print(f"      {reason}")


def run_synthetic(impls: list[str]) -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"synthetic: torch is not installed ({exc})", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("synthetic: CUDA is not available", file=sys.stderr)
        return 1
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    failures = 0
    for impl in impls:
        adapter = IMPLS[impl]
        for case in SYNTHETIC_CASES:
            try:
                fn, note = adapter(case, device)
            except Exception as exc:  # noqa: BLE001 - capability probe
                print(f"{impl:<14}{case['id']:<44}error  "
                      f"{type(exc).__name__}: {exc}")
                failures += 1
                continue
            if fn is None:
                print(f"{impl:<14}{case['id']:<44}skip   {note}")
                continue
            try:
                fn()
                torch.cuda.synchronize()
                print(f"{impl:<14}{case['id']:<44}ok     {note}")
            except Exception as exc:  # noqa: BLE001 - run probe
                print(f"{impl:<14}{case['id']:<44}FAIL   "
                      f"{type(exc).__name__}: {exc}")
                failures += 1
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=None,
                    help="bench_state --dump-cases JSON (default: built-in table)")
    ap.add_argument("--impl", default=None,
                    help="comma list of " + ",".join(IMPLS))
    ap.add_argument("--synthetic", action="store_true",
                    help="run one tiny case per supported operator (needs CUDA)")
    args = ap.parse_args()

    impls = list(IMPLS)
    if args.impl:
        impls = [name.strip() for name in args.impl.split(",") if name.strip()]
        unknown = [name for name in impls if name not in IMPLS]
        if unknown:
            print(f"unknown impls: {unknown}", file=sys.stderr)
            return 2

    if args.synthetic:
        return run_synthetic(impls)
    print_mapping(load_cases(args.cases), impls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
