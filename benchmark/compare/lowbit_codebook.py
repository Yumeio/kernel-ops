#!/usr/bin/env python3
"""Lloyd-Max codebooks of the B17 low-bit KV codec (kops::experimental::lowbit, TurboQuant).

Dev-only, never built or shipped (like the other scripts in benchmark/compare/). It prints the
`kLloydMax*` tables of csrc/attention/lowbit_kernels.cuh; `tests/unit/test_merge.cc` re-derives the
fixed-point condition in fp64, so the tables are checked without this script.

Model. A unit vector rotated by a random orthogonal matrix has coordinates distributed as one
coordinate of a uniform point on the sphere S^(d-1). In units of 1/sqrt(d) (variance 1):

    f_d(z)  ~  (1 - z^2 / d)^((d - 3) / 2),   |z| <= sqrt(d)

which tends to N(0, 1) as d grows. The b-bit Lloyd-Max quantizer of f_d is symmetric, so a table
holds the 2^(b-1) positive centroids; decision thresholds are the midpoints.

  python benchmark/compare/lowbit_codebook.py                 # C++ tables for d = 64, 128, 256
  python benchmark/compare/lowbit_codebook.py --check DIR     # compare with reference JSON files
                                                              # (codebook_d<d>_b<b>.json, centroids
                                                              # in 1/sqrt(d)-free units)
No numpy / scipy: plain Python floats, composite Simpson on a fixed grid.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

GRID = 1 << 16  # Simpson intervals over [0, sqrt(d)] (even)


def tables(d: int):
    """Cumulative integrals F(z) = int_0^z f, G(z) = int_0^z t f(t) dt on a uniform grid."""
    hi = math.sqrt(d)
    h = hi / GRID
    e = (d - 3) / 2.0

    def pdf(z: float) -> float:
        t = 1.0 - z * z / d
        return t ** e if t > 0.0 else 0.0

    f = [pdf(i * h) for i in range(GRID + 1)]
    # Simpson on pairs of intervals; odd nodes get the midpoint rule of their pair (quadratic).
    big_f = [0.0] * (GRID + 1)
    big_g = [0.0] * (GRID + 1)
    for i in range(0, GRID, 2):
        z0, z1, z2 = i * h, (i + 1) * h, (i + 2) * h
        f0, f1, f2 = f[i], f[i + 1], f[i + 2]
        g0, g1, g2 = z0 * f0, z1 * f1, z2 * f2
        big_f[i + 2] = big_f[i] + h / 3.0 * (f0 + 4.0 * f1 + f2)
        big_g[i + 2] = big_g[i] + h / 3.0 * (g0 + 4.0 * g1 + g2)
        big_f[i + 1] = big_f[i] + h / 12.0 * (5.0 * f0 + 8.0 * f1 - f2)
        big_g[i + 1] = big_g[i] + h / 12.0 * (5.0 * g0 + 8.0 * g1 - g2)
    return h, f, big_f, big_g


def cumulative(h, f, big, weight_z, z):
    """big(z) by cubic Hermite interpolation between grid nodes (derivative = integrand)."""
    x = z / h
    i = min(int(x), GRID - 1)
    t = x - i
    y0, y1 = big[i], big[i + 1]
    d0 = (i * h * f[i] if weight_z else f[i]) * h
    d1 = ((i + 1) * h * f[i + 1] if weight_z else f[i + 1]) * h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * d0 + (-2 * t3 + 3 * t2) * y1 +
            (t3 - t2) * d1)


def lloyd_max(d: int, bits: int, iters: int = 20000, tol: float = 1e-15):
    """Positive centroids (ascending) of the symmetric 2^bits-level quantizer and its MSE."""
    h, f, big_f, big_g = tables(d)
    hi = math.sqrt(d)
    n = 1 << (bits - 1)
    total = big_f[GRID]
    # start from equal-mass cells of the half distribution
    cent = []
    for i in range(n):
        target = (i + 0.5) / n * total
        lo_z, hi_z = 0.0, hi
        for _ in range(60):
            mid = 0.5 * (lo_z + hi_z)
            if cumulative(h, f, big_f, False, mid) < target:
                lo_z = mid
            else:
                hi_z = mid
        cent.append(0.5 * (lo_z + hi_z))
    for _ in range(iters):
        edges = [0.0] + [0.5 * (cent[i] + cent[i + 1]) for i in range(n - 1)] + [hi]
        new = []
        for i in range(n):
            mass = cumulative(h, f, big_f, False, edges[i + 1]) - cumulative(h, f, big_f, False, edges[i])
            mom = cumulative(h, f, big_g, True, edges[i + 1]) - cumulative(h, f, big_g, True, edges[i])
            new.append(mom / mass if mass > 0.0 else 0.5 * (edges[i] + edges[i + 1]))
        delta = max(abs(a - b) for a, b in zip(cent, new))
        cent = new
        if delta < tol:
            break
    # distortion: E[z^2] - sum mass_i c_i^2 (centroids are conditional means); E[z^2] = 1
    edges = [0.0] + [0.5 * (cent[i] + cent[i + 1]) for i in range(n - 1)] + [hi]
    captured = 0.0
    for i in range(n):
        mass = cumulative(h, f, big_f, False, edges[i + 1]) - cumulative(h, f, big_f, False, edges[i])
        captured += mass / total * cent[i] * cent[i]
    return cent, 1.0 - captured


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="64,128,256")
    ap.add_argument("--check", help="directory with codebook_d<d>_b<b>.json reference files")
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    worst = 0.0
    for d in dims:
        for bits in (1, 2, 3, 4):
            cent, mse = lloyd_max(d, bits)
            if args.check:
                path = pathlib.Path(args.check) / f"codebook_d{d}_b{bits}.json"
                if not path.exists():
                    continue
                ref = json.loads(path.read_text())["centroids"]
                ref_pos = [c * math.sqrt(d) for c in ref[len(ref) // 2:]]
                err = max(abs(a - b) for a, b in zip(cent, ref_pos))
                worst = max(worst, err)
                print(f"d={d} bits={bits}: max |centroid - reference| = {err:.3e}  mse = {mse:.6f}")
                continue
            body = ", ".join(f"{c:.9f}f" for c in cent)
            print(f"// d = {d}, {bits} bit: normalized MSE {mse:.6f}")
            print(f"inline constexpr float kLloydMax{d}B{bits}[{len(cent)}] = {{{body}}};")
    if args.check:
        print(f"worst = {worst:.3e}")
        return 0 if worst < 1e-6 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
