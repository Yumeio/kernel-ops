#!/usr/bin/env python3
"""Quality loss of the B17 experimental codecs (dev-only, never shipped).

Models the exact record semantics of `csrc/attention/lowbit_kernels.cuh` and the W2
format of `csrc/moe/fused_moe.cu`, then reports:

  * KV reconstruction: normalized MSE, cosine, SQNR for every codec/bit width.
  * Attention: output/logit error and a KL-based perplexity proxy (full vs decoded KV).
  * W2 experts: weight NMSE and expert-output error for the kops pack vs attention.rs
    Moet (MXFP4 LUT) on the same random weights.

The C++ tests validate the kernels against ref::lowbit_* and ref::attention_f64; this
script measures the *algorithmic* loss of the same codecs, including the 3rd-party
formats used as references. Run:

  python3 benchmark/compare/lowbit_quality.py --json out/b17/lowbit-quality.json \
      --markdown out/b17/quality.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowbit_codebook import lloyd_max  # noqa: E402

D = 128
GROUP = 32


# ---- signed Hadamard / sign stream (same convention as lowbit_kernels.cuh) --------------------
def sign_word(seed: int, stream: int, word: int) -> int:
    x = (seed + 0x9E3779B9 * (stream + 1)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x ^ (0x85EBCA6B * (word + 1) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def negated(seed: int, stream: int, i: int) -> bool:
    return (sign_word(seed, stream, i // 32) >> (i % 32)) & 1 != 0


def hadamard(v: list[float]) -> list[float]:
    out = list(v)
    n = len(out)
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                a, b = out[j], out[j + h]
                out[j], out[j + h] = a + b, a - b
        h *= 2
    return out


def rotate(x: list[float], seed: int, stream: int, inverse: bool = False) -> list[float]:
    n = len(x)
    v = [(-x[i] if (not inverse and negated(seed, stream, i)) else x[i]) for i in range(n)]
    v = hadamard(v)
    s = 1.0 / math.sqrt(n)
    return [(-v[i] * s if (inverse and negated(seed, stream, i)) else v[i] * s) for i in range(n)]


# ---- codecs (exact level tables and scale rules) ---------------------------------------------
class Codec:
    def __init__(self, name: str, bits: int, mode: str, rotate_side: bool = True):
        self.name = name
        self.bits = bits                    # stored bits per element (kTurboProd: codes + sign)
        self.mode = mode                    # uniform | mse | prod
        self.rotated = rotate_side
        # kTurboProd<b> is TurboQuant_prod at b bits: a (b - 1)-bit Lloyd-Max code + 1 QJL sign.
        code_bits = bits - 1 if mode == "prod" else bits
        n = 1 << code_bits
        if mode == "uniform":
            self.levels = [2 * i - (n - 1) for i in range(n)]
            self.levels = [x / (n - 1) for x in self.levels]  # unit absmax
            self.unit = "absmax"
        else:
            cb, _ = lloyd_max(D, code_bits)
            pos = list(cb)
            self.levels = [-c for c in reversed(pos)] + pos
            self.unit = "norm"
        self.thresholds = [(self.levels[i] + self.levels[i + 1]) / 2 for i in range(n - 1)]

    def encode_decode(self, x: list[float], seed: int = 7) -> list[float]:
        if self.rotated:
            xr = rotate(x, seed, 0)
        else:
            xr = list(x)
        if self.unit == "absmax":
            amax = max(abs(v) for v in xr)
            scale = amax / self.levels[-1] if amax > 0 else 0.0
        else:
            scale = math.sqrt(sum(v * v for v in xr)) / math.sqrt(D)
        codes = []
        residual = []
        for v in xr:
            z = v / scale if scale > 0 else 0.0
            code = sum(1 for t in self.thresholds if t < z)
            codes.append(code)
            residual.append(z - self.levels[code])
        out = [scale * self.levels[c] for c in codes]
        if self.mode == "prod":
            # QJL: sign plane of Hn(s' .* residual), gain = a * ||e|| * sqrt(pi/2) / sqrt(D).
            e = [(-residual[i] if negated(seed, 1, i) else residual[i]) for i in range(D)]
            e = hadamard(e)
            sgn = [1.0 if e[i] >= 0.0 else -1.0 for i in range(D)]
            sgn = hadamard(sgn)
            s = 1.0 / math.sqrt(D)
            gain = scale * math.sqrt(sum(r * r for r in residual)) * math.sqrt(math.pi / 2.0) * s
            for i in range(D):
                t = sgn[i] * s
                out[i] += gain * (-t if negated(seed, 1, i) else t)
        if self.rotated:
            return rotate(out, seed, 0, inverse=True)
        return out


# attention.rs turbo3/turbo4 and TurboQuant_mse/prod reference families.
def codecs() -> dict[str, Codec]:
    out = {
        "turbo4 (attention.rs)": Codec("turbo4", 4, "uniform"),
        "turbo3 (attention.rs)": Codec("turbo3", 3, "uniform"),
    }
    for b in (2, 3, 4):
        out[f"turbo_mse{b}"] = Codec(f"turbo_mse{b}", b, "mse")
    for b in (2, 3, 4):
        out[f"turbo_prod{b}"] = Codec(f"turbo_prod{b}", b, "prod")
    return out


def random_vec(rng: random.Random, scale: float = 1.0) -> list[float]:
    v = [rng.gauss(0.0, scale) for _ in range(D)]
    # one outlier channel: the case the rotation is meant to spread
    v[0] *= 8.0
    return v


# ---- metrics ---------------------------------------------------------------------------------
def nmse(x: list[float], y: list[float]) -> float:
    num = sum((a - b) ** 2 for a, b in zip(x, y))
    den = sum(a * a for a in x)
    return num / den if den > 0 else 0.0


def cosine(x: list[float], y: list[float]) -> float:
    nx = math.sqrt(sum(a * a for a in x))
    ny = math.sqrt(sum(b * b for b in y))
    return sum(a * b for a, b in zip(x, y)) / (nx * ny) if nx > 0 and ny > 0 else 1.0


def softmax(s: list[float]) -> list[float]:
    m = max(s)
    e = [math.exp(v - m) for v in s]
    z = sum(e)
    return [v / z for v in e]


def attention_quality(c: Codec, rng: random.Random, heads: int = 2, kv: int = 256) -> dict:
    # One query per head, one KV head (GQA), shared K/V across heads.
    q = [[rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(heads)]
    k = [random_vec(rng, 0.5) for _ in range(kv)]
    v = [random_vec(rng, 0.5) for _ in range(kv)]
    qc = q  # B17 quantizes K/V only; Q is rotated but kept at input precision.
    kc = [c.encode_decode(x) for x in k]
    vc = [c.encode_decode(x) for x in v]
    out_full, out_q, lse_full, lse_q = [], [], [], []
    kl = 0.0
    inv = 1.0 / math.sqrt(D)
    for h in range(heads):
        sf = [inv * sum(a * b for a, b in zip(q[h], k[j])) for j in range(kv)]
        sq = [inv * sum(a * b for a, b in zip(qc[h], kc[j])) for j in range(kv)]
        pf, pq = softmax(sf), softmax(sq)
        kl += sum(p * math.log(p / max(qq, 1e-300)) for p, qq in zip(pf, pq))
        of = [sum(pf[j] * v[j][d] for j in range(kv)) for d in range(D)]
        oq = [sum(pq[j] * vc[j][d] for j in range(kv)) for d in range(D)]
        out_full.append(of)
        out_q.append(oq)
        mf, mq = max(sf), max(sq)
        lse_full.append(mf + math.log(sum(math.exp(x - mf) for x in sf)))
        lse_q.append(mq + math.log(sum(math.exp(x - mq) for x in sq)))
    n = 0.0
    for a, b in zip(sum(out_full, []), sum(out_q, [])):
        n += (a - b) ** 2
    den = sum(a * a for a in sum(out_full, []))
    logit_mse = sum((a - b) ** 2 for a, b in zip(sum([[inv * sum(x * y for x, y in zip(q[h], k[j])) for j in range(kv)] for h in range(heads)], []),
                                                  sum([[inv * sum(x * y for x, y in zip(qc[h], kc[j])) for j in range(kv)] for h in range(heads)], []))) / (heads * kv)
    lse_err = max(abs(a - b) for a, b in zip(lse_full, lse_q))
    return {
        "output_rel_fro": math.sqrt(n / den) if den > 0 else 0.0,
        "output_cosine": cosine(sum(out_full, []), sum(out_q, [])),
        "logit_mse": logit_mse,
        "lse_max_abs": lse_err,
        "kl_per_query": kl / heads,
        "ppl_proxy": math.exp(kl / heads),
    }


# ---- W2 experts -------------------------------------------------------------------------------
def bf16(x: float) -> float:
    # round-to-nearest-even bf16, sufficient for a quality model
    bits = 0
    raw = int.from_bytes(__import__("struct").pack("<f", x), "little")
    # use struct/math instead of bit twiddling: round mantissa to 8 bits
    if x == 0.0:
        return 0.0
    m, e = math.frexp(x)
    scale = 2.0 ** (e - 8)
    r = round(x / scale) * scale
    return r


def w2_kops(w: list[float], group: int = GROUP) -> list[float]:
    out = []
    for g in range(0, len(w), group):
        seg = w[g:g + group]
        amax = max(abs(x) for x in seg)
        scale = amax / 4.0 if amax > 0 else 0.0
        inv = 1.0 / scale if scale > 0 else 0.0
        levels = [-4.0, -1.0, 1.0, 4.0]
        for x in seg:
            z = x * inv
            code = 0 if z < -2.5 else (1 if z < 0 else (2 if z < 2.5 else 3))
            out.append(levels[code] * scale)
    return out


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
# attention.rs kNibbleToCode: nibble -> {2,3} positive, {1,0} negative.
MOET_CODE = [2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0]
MOET_LEVEL = [-4.0, -1.0, 1.0, 4.0]


def e2m1_quant(x: float, scale: float) -> tuple[int, float]:
    z = x / scale if scale > 0 else 0.0
    # nearest even-mantissa e2m1; ties do not matter for a random quality study
    best, bd = 0, 1e30
    for n, v in enumerate(E2M1):
        if abs(z - v) < bd:
            best, bd = n, abs(z - v)
    return best, E2M1[best]


def w2_attention_rs(w: list[float], group: int = GROUP) -> list[float]:
    out = []
    for g in range(0, len(w), group):
        seg = w[g:g + group]
        amax = max(abs(x) for x in seg)
        if amax == 0:
            out.extend([0.0] * group)
            continue
        # OCP MX: block scale is a power of two with e2m1 max 6.
        e = math.ceil(math.log2(amax / 6.0)) if amax > 0 else 0
        scale = 2.0 ** e
        for x in seg:
            nib, val = e2m1_quant(x, scale)
            del val
            out.append(MOET_LEVEL[MOET_CODE[nib]] * scale)
    return out


def matvec(a: list[float], w: list[float], n: int, k: int) -> list[float]:
    out = []
    for i in range(n):
        row = w[i * k:(i + 1) * k]
        out.append(sum(x * y for x, y in zip(a, row)))
    return out


def w2_quality(rng: random.Random, rows: int = 64, n: int = 128, k: int = 256) -> dict:
    w = [[rng.gauss(0.0, 0.1) for _ in range(k)] for _ in range(n)]
    a = [rng.gauss(0.0, 1.0) for _ in range(rows * k)]
    full_out, kops_out, rs_out = [], [], []
    wf, wk, wr = [], [], []
    for i in range(n):
        wf.extend(w[i])
        wk.extend(w2_kops(w[i]))
        wr.extend(w2_attention_rs(w[i]))
    for r in range(rows):
        av = a[r * k:(r + 1) * k]
        full_out.extend(matvec(av, wf, n, k))
        kops_out.extend(matvec(av, wk, n, k))
        rs_out.extend(matvec(av, wr, n, k))
    def rel(got):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(got, full_out)) /
                         max(1e-30, sum(y * y for y in full_out)))
    return {
        "w2_kops_weight_nmse": nmse(wf, wk),
        "w2_attention_rs_weight_nmse": nmse(wf, wr),
        "w2_kops_output_rel_fro": rel(kops_out),
        "w2_attention_rs_output_rel_fro": rel(rs_out),
        "w2_kops_output_cosine": cosine(full_out, kops_out),
        "w2_attention_rs_output_cosine": cosine(full_out, rs_out),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--vectors", type=int, default=128)
    args = ap.parse_args()
    rng = random.Random(20261110)
    xs = [random_vec(rng) for _ in range(args.vectors)]

    rows = []
    for name, c in codecs().items():
        rec = [c.encode_decode(x) for x in xs]
        q = attention_quality(c, random.Random(17))
        rows.append({
            "codec": name,
            "bits": c.bits,
            "kv_nmse": sum(nmse(x, y) for x, y in zip(xs, rec)) / len(xs),
            "kv_cosine": sum(cosine(x, y) for x, y in zip(xs, rec)) / len(xs),
            "sqnr_db": -10.0 * math.log10(max(1e-30, sum(nmse(x, y) for x, y in zip(xs, rec)) / len(xs))),
            **q,
        })
    w2 = w2_quality(random.Random(23))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"d": D, "vectors": args.vectors, "kv": rows, "w2": w2}, f, indent=2)

    if args.markdown:
        lines = ["# B17 low-bit quality loss (synthetic, fp64 model)", "",
                 f"D = {D}, {args.vectors} random unit-ish vectors with one 8x outlier channel, signed Hadamard seed 7.",
                 "The C++ tests validate the kernels against `ref::lowbit_*` and `ref::attention_f64`;",
                 "this table measures the algorithmic loss of the same codecs.", "",
                 "## KV codec", "",
                 "| codec | bits | KV NMSE | cosine | SQNR dB | attn out rel | logit MSE | LSE max | KL/query | exp(KL) |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(
                f"| {r['codec']} | {r['bits']} | {r['kv_nmse']:.4f} | {r['kv_cosine']:.6f} | "
                f"{r['sqnr_db']:.2f} | {r['output_rel_fro']:.4f} | {r['logit_mse']:.4f} | "
                f"{r['lse_max_abs']:.4f} | {r['kl_per_query']:.4f} | {r['ppl_proxy']:.4f} |")
        lines += ["", "## Expert W2", "",
                  "| format | weight NMSE | output rel Frobenius | output cosine |",
                  "|---|---:|---:|---:|",
                  f"| kops W2 nearest (-4,-1,+1,+4, scale amax/4) | {w2['w2_kops_weight_nmse']:.4f} | "
                  f"{w2['w2_kops_output_rel_fro']:.4f} | {w2['w2_kops_output_cosine']:.6f} |",
                  f"| attention.rs Moet (MXFP4 LUT, UE8M0 block scale) | {w2['w2_attention_rs_weight_nmse']:.4f} | "
                  f"{w2['w2_attention_rs_output_rel_fro']:.4f} | {w2['w2_attention_rs_output_cosine']:.6f} |",
                  "", "`exp(KL)` is an attention-distribution perplexity proxy, not a model PPL. "
                  "Model-level PPL needs an engine + checkpoint and remains outside this repo."]
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(f"{'codec':<22} {'bits':>4} {'KV NMSE':>9} {'cos':>8} {'SQNR':>7} "
          f"{'out rel':>8} {'KL/q':>8} {'exp(KL)':>8}")
    for r in rows:
        print(f"{r['codec']:<22} {r['bits']:4d} {r['kv_nmse']:9.4f} {r['kv_cosine']:8.5f} "
              f"{r['sqnr_db']:7.2f} {r['output_rel_fro']:8.4f} {r['kl_per_query']:8.4f} {r['ppl_proxy']:8.4f}")
    print("\nW2:")
    for key, value in w2.items():
        print(f"  {key:<38} {value:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
