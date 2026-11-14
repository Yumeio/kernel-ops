"""Head-to-head decode benchmark: reference (eager) vs Triton megakernel vs ours.

    python benchmark/compare/kimi_linear/bench.py --impl triton ours --ctx 2048 4096 8192 16384

Timing: `steps` chained decode steps (output hidden fed back, state carried),
one `torch.cuda.synchronize()` at the end, median over `rounds`.  Correctness:
the first step of every impl is compared against `reference.reference_step`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402


def load_impl(name):
    if name == "triton":
        import kimi_mega_triton as m
    elif name == "ours":
        import ayaka_mega as m
    else:
        raise SystemExit(f"unknown impl {name}")
    return m


def relerr(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def check(model, cfg, L, dev):
    hidden, st = R.make_state(cfg, L, dev)
    ref_h, ref_s = R.reference_step(model, hidden, R.clone_state(st))
    out_h, out_s = model.step(hidden.clone(), R.clone_state(st))
    torch.cuda.synchronize()
    res = {"hidden": relerr(out_h, ref_h)}
    for i, kind in enumerate(cfg.pattern):
        keys = ("S", "cq", "ck", "cv") if kind == "K" else ("c_kv", "k_rope")
        for k in keys:
            res[f"b{i}.{k}"] = relerr(out_s[i][k], ref_s[i][k])
    return res


def chain_check(model, cfg, L, dev, n=4):
    """n chained steps vs the reference chained the same way (state carried)."""
    hidden, st = R.make_state(cfg, L, dev)
    rh, rs = hidden, R.clone_state(st)
    oh, os_ = hidden.clone(), R.clone_state(st)
    worst = 0.0
    for _ in range(n):
        rh, rs = R.reference_step(model, rh, rs)
        oh, os_ = model.step(oh, os_)
        oh = oh.clone()
        torch.cuda.synchronize()
        worst = max(worst, relerr(oh, rh))
    return worst


def timeit(model, cfg, L, dev, steps, rounds, warm=30):
    hidden, st = R.make_state(cfg, L, dev)
    h = hidden.clone()
    for _ in range(warm):
        h, st = model.step(h, st)
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(steps):
            h, st = model.step(h, st)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / steps * 1e6)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def time_reference(model, cfg, L, dev, steps=3):
    hidden, st = R.make_state(cfg, L, dev)
    h = hidden
    R.reference_step(model, h, R.clone_state(st))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        h, st = R.reference_step(model, h, st)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", nargs="+", default=["triton"])
    ap.add_argument("--ctx", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--ref-time", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda")
    cfg = R.Cfg()
    out = {"gpu": torch.cuda.get_device_name(0),
           "sms": torch.cuda.get_device_properties(0).multi_processor_count, "impl": {}}
    sd = None
    for name in a.impl:
        mod = load_impl(name)
        model = mod.Model(cfg)
        if sd is None:
            R.init_random(model)
            sd = model.state_dict()
        else:
            model.load_state_dict(sd)
        model = model.to(dev)
        rec = {"us": {}, "min_us": {}, "err": {}}
        if not a.no_check:
            for L in (a.ctx[0], 777):
                e = check(model, cfg, L, dev)
                rec["err"][L] = e
                worst = max(e, key=e.get)
                print(f"[{name}] check L={L}: hidden {e['hidden']:.2e}  worst {worst} {e[worst]:.2e}")
            ce = chain_check(model, cfg, a.ctx[0], dev)
            rec["err"]["chain4"] = ce
            print(f"[{name}] chained x4 hidden rel-err {ce:.2e}")
        for L in a.ctx:
            med, mn = timeit(model, cfg, L, dev, a.steps, a.rounds)
            rec["us"][L], rec["min_us"][L] = med, mn
            print(f"[{name}] ctx {L:6d}: {med:8.1f} us/step (min {mn:.1f})")
        rec["geomean_us"] = math.exp(sum(math.log(v) for v in rec["us"].values()) / len(rec["us"]))
        print(f"[{name}] geomean {rec['geomean_us']:.1f} us/step")
        out["impl"][name] = rec
        if a.ref_time and "reference" not in out:
            out["reference"] = {L: time_reference(model, cfg, L, dev) for L in a.ctx}
            for L, v in out["reference"].items():
                print(f"[reference] ctx {L:6d}: {v:10.1f} us/step")
        del model
        torch.cuda.empty_cache()
    if len(out["impl"]) > 1:
        names = list(out["impl"])
        base = out["impl"][names[0]]
        for n in names[1:]:
            r = out["impl"][n]
            print(f"{n} vs {names[0]}: " + "  ".join(
                f"ctx{L} {base['us'][L] / r['us'][L]:.3f}x" for L in a.ctx)
                + f"  geomean {base['geomean_us'] / r['geomean_us']:.3f}x")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
