#!/usr/bin/env python3
"""DeepEP baseline for the B12 expert-parallel cases (dev-only).

DeepEP (`3rd/DeepEP`, tag v1.2.1) is the reference implementation of MoE
dispatch/combine. This script runs the EP slice of
`bench_communication --dump-cases` through DeepEP's normal (intranode NVLink)
and low-latency (NVSHMEM/IBGDA) kernels and writes the shared baseline schema
`compare_results.py` reads:

    build/bench/benchmark/bench_communication --dump-cases > out/b12/cases.json
    python benchmark/compare/deepep_baseline.py \
        --cases out/b12/cases.json --json out/b12/deepep-comm.json --world-size 2
    python benchmark/compare/compare_results.py \
        out/b12/kops-comm.json out/b12/deepep-comm.json

Covered ops: `ep_layout` and `ep_dispatch_combine` (normal and low-latency).
Every other op in the file is reported under `skipped`, as is an EP case whose
`_wN` does not match `--world-size` (a 2-GPU host runs the w2 rows and records
the w4/w8 rows as skipped).

    op                    DeepEP call
    ep_layout             Buffer.get_dispatch_layout(topk_idx, num_experts)
    ep_dispatch_combine   Buffer.dispatch(...); Buffer.combine(...)
      normal              intranode NVLink path, BF16, num_worst_tokens=tokens*world
      low-latency         Buffer.low_latency_dispatch/combine, BF16 (use_fp8=False)

The normal path asks DeepEP for the fixed `[tokens*world, hidden]` receive
layout (the size the C++ case declares in `bytes`); `--no-worst-tokens` selects
DeepEP's implicit host-synced receive count instead. The low-latency receive
layout is DeepEP's masked worst case, `[experts/world, world*max_tokens, hidden]`,
which is larger than the C++ case's exact-capacity buffer - the `note` field
records that, and `bytes` stays the kops value so the ratio is only a time ratio.

Requirements: >= `--world-size` NVLink-connected GPUs on one node, and DeepEP
v1.2.1 built against NVSHMEM 3.3.9+ for the low-latency rows (SM90 only; the
normal rows also work on 2x A100). Without `deep_ep` importable, or with fewer
GPUs than the requested world, the script prints why and exits 77 - the
repository's CTest skip code. Build steps and the current status (a 1x RTX
3050 cannot run it) are in `reports/deepep-baseline.md`.

Timing matches `communication_torch.py`: `--calls` dispatch+combine round trips
between one pair of CUDA events per sample, `--groups` samples, the slowest
rank per sample, then median/p10/p90/p99. The token layout is computed once per
case outside the timed region (`ep_layout` times it explicitly). DeepEP
allocates its receive tensors on every call, so the allocator cost is part of
the measurement, exactly as in DeepEP's own `tests/`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import traceback

import torch
import torch.distributed as dist

SKIP_EXIT = 77
EP_OPS = ("ep_layout", "ep_dispatch_combine")
# DeepEP supports these world sizes out of the box (get_dispatch_config); this
# script only drives one node, so the ids above 8 are out of scope.
MIN_WORLD, MAX_WORLD = 2, 8


def _int_field(case_id: str, key: str):
    """`_t128` -> 128. The ids are `<op>_<dtype>_[normal|ll]_w<W>_t<T>_h<H>_e<E>_k<K>`."""
    match = re.search(r"(?:^|_)" + re.escape(key) + r"(\d+)", case_id)
    return int(match.group(1)) if match else None


def parse_ep_case(case: dict):
    """Return the EP shape encoded in the case id, or None if it cannot be read."""
    cid = str(case.get("id", ""))
    op = str(case.get("op", ""))
    if op == "ep_layout":
        tokens = _int_field(cid, "t")
        topk = _int_field(cid, "k")
        experts = _int_field(cid, "e")
        if tokens is None or topk is None or experts is None:
            return None
        return {"mode": "layout", "world": None, "tokens": tokens, "hidden": None,
                "topk": int(topk), "experts": int(experts)}
    if op == "ep_dispatch_combine":
        world = _int_field(cid, "w")
        tokens = _int_field(cid, "t")
        hidden = _int_field(cid, "h")
        experts = _int_field(cid, "e")
        topk = _int_field(cid, "k")
        if None in (world, tokens, hidden, experts, topk):
            return None
        if "normal" in cid:
            mode = "normal"
        elif re.search(r"_(?:ll|low_latency)_", cid):
            mode = "ll"
        else:
            return None
        return {"mode": mode, "world": int(world), "tokens": int(tokens),
                "hidden": int(hidden), "topk": int(topk), "experts": int(experts)}
    return None


def select_cases(cases: list, world_size: int):
    """Split the dumped case table into runnable rows and skip records."""
    runnable, skipped = [], []
    for case in cases:
        cid = str(case.get("id", ""))
        op = str(case.get("op", ""))
        if op not in EP_OPS:
            skipped.append({"id": cid,
                            "note": f"not an EP case (op {op}); DeepEP only covers EP dispatch/combine"})
            continue
        ep = parse_ep_case(case)
        if ep is None:
            skipped.append({"id": cid, "note": "cannot parse the EP shape/world from the case id"})
            continue
        if ep["world"] is not None and ep["world"] != world_size:
            skipped.append({"id": cid,
                            "note": f"needs {ep['world']} ranks; this run uses {world_size} (--world-size)"})
            continue
        world = ep["world"] if ep["world"] is not None else world_size
        if ep["experts"] % world != 0:
            skipped.append({"id": cid,
                            "note": f"{ep['experts']} experts is not divisible by world {world}"})
            continue
        if ep["topk"] > ep["experts"]:
            skipped.append({"id": cid, "note": f"top-k {ep['topk']} exceeds {ep['experts']} experts"})
            continue
        if ep["mode"] in ("normal", "ll") and ep["hidden"] % 8 != 0:
            skipped.append({"id": cid, "note": f"hidden {ep['hidden']} is not a multiple of 8"})
            continue
        if ep["mode"] == "ll" and ep["hidden"] % 128 != 0:
            skipped.append({"id": cid, "note": f"low-latency needs hidden % 128 == 0, got {ep['hidden']}"})
            continue
        runnable.append((case, ep))
    return runnable, skipped


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def rank_time(fn, device: torch.device, calls: int, groups: int, warmup: int) -> list:
    """Time `calls` round trips per sample; the slowest rank defines the sample."""
    dist.barrier()
    torch.cuda.synchronize(device)
    for _ in range(max(2, warmup // max(calls, 1))):
        fn()
    dist.barrier()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(groups):
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        local = start.elapsed_time(end) / calls
        # Reduce outside the timed interval: the slowest rank defines the sample.
        t = torch.tensor([local], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(float(t.item()))
    samples.sort()
    return samples


def percentiles(samples: list) -> dict:
    n = len(samples)
    return {
        "median_ms": samples[n // 2],
        "p10_ms": samples[max(0, n // 10)],
        "p90_ms": samples[min(n - 1, n * 9 // 10)],
        "p99_ms": samples[min(n - 1, n * 99 // 100)],
        "min_ms": samples[0],
    }


def make_routing(tokens: int, experts: int, topk: int, rank: int, device: torch.device):
    """Top-k routing with the same ~8% `-1` mask as the C++ EP case table."""
    gen = torch.Generator(device=device)
    gen.manual_seed(1234 + 97 * rank)
    scores = torch.rand((tokens, experts), generator=gen, device=device, dtype=torch.float32)
    idx = torch.topk(scores, topk, dim=-1, largest=True, sorted=False).indices.to(torch.int64)
    rows = torch.arange(tokens, device=device).view(-1, 1)
    cols = torch.arange(topk, device=device).view(1, -1)
    idx = idx.masked_fill(((rows + cols + rank) % 13) == 0, -1)
    weights = torch.rand((tokens, topk), generator=gen, device=device, dtype=torch.float32)
    return idx, weights


def num_max_tokens(ep: dict, requested: int) -> int:
    """Per-rank low-latency capacity; DeepEP requires (world * max) % 4 == 0."""
    value = max(ep["tokens"], requested or 0)
    return (value + 3) // 4 * 4


def note_of(ep: dict, world: int, worst_tokens: bool) -> str:
    if ep["mode"] == "layout":
        return f"DeepEP get_dispatch_layout (world {world})"
    if ep["mode"] == "normal":
        recv = ("fixed [tokens*world, hidden] worst-case layout, no host sync"
                if worst_tokens else "host-synced recv count")
        return f"DeepEP dispatch+combine (normal, intranode NVLink, BF16, {recv})"
    return ("DeepEP low_latency dispatch+combine (NVSHMEM/IBGDA, BF16, "
            "recv [E/world, world*max_tokens, hidden])")


def make_normal_fn(deep_ep, buffer, case, ep, rank, device, world, opts):
    tokens, hidden = ep["tokens"], ep["hidden"]
    experts, topk = ep["experts"], ep["topk"]
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = make_routing(tokens, experts, topk, rank, device)
    # Layout is the separate `ep_layout` case; keep it out of the round trip.
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, experts)
    # The kops normal case sizes recv_x as [tokens*world, hidden]. `num_worst_tokens`
    # gives DeepEP the same fixed layout and removes its implicit CPU wait.
    num_worst_tokens = tokens * world if opts["worst_tokens"] else 0

    def fn():
        recv_x, _recv_idx, recv_weights, _per_expert, handle, _event = buffer.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            num_worst_tokens=num_worst_tokens)
        # The MoE expert GEMM is out of scope: combine the dispatch output as
        # the expert output (the C++ case feeds a pre-filled expert_out).
        buffer.combine(recv_x, handle, topk_weights=recv_weights)

    return fn


def make_low_latency_fn(deep_ep, buffer, case, ep, rank, device, world, opts):
    tokens, hidden = ep["tokens"], ep["hidden"]
    experts, topk = ep["experts"], ep["topk"]
    max_tokens = num_max_tokens(ep, opts["max_dispatch_tokens"])
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
    topk_idx, topk_weights = make_routing(tokens, experts, topk, rank, device)
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)

    def fn():
        recv_x, _recv_count, handle, _event, _hook = buffer.low_latency_dispatch(
            x, topk_idx, max_tokens, experts, use_fp8=False)
        # recv_x is [E/world, world*max_tokens, hidden] bf16; combine consumes it
        # directly and reduces with the per-token weights (the kops semantics).
        buffer.low_latency_combine(recv_x, topk_idx, topk_weights, handle, out=out)

    return fn


def make_layout_fn(deep_ep, buffer, case, ep, rank, device, world, opts):
    topk_idx, _weights = make_routing(ep["tokens"], ep["experts"], ep["topk"], rank, device)
    experts = ep["experts"]
    return lambda: buffer.get_dispatch_layout(topk_idx, experts)


def make_buffer(deep_ep, group, world, cases, opts, low_latency: bool):
    if low_latency:
        qps = cases[0][1]["experts"] // world
        rdma_bytes = 0
        for _case, ep in cases:
            max_tokens = num_max_tokens(ep, opts["max_dispatch_tokens"])
            rdma_bytes = max(rdma_bytes, int(deep_ep.Buffer.get_low_latency_rdma_size_hint(
                max_tokens, ep["hidden"], world, ep["experts"])))
        rdma_bytes = (rdma_bytes + 127) // 128 * 128
        return deep_ep.Buffer(group, 0, int(rdma_bytes), low_latency_mode=True,
                              num_qps_per_rank=int(qps),
                              allow_nvlink_for_low_latency_mode=bool(opts["nvlink_ll"]),
                              explicitly_destroy=True)
    dispatch_cfg = deep_ep.Buffer.get_dispatch_config(world)
    combine_cfg = deep_ep.Buffer.get_combine_config(world)
    nvl_bytes = 1 << 20
    for _case, ep in cases:
        if ep["mode"] != "normal":
            continue
        hidden_bytes = ep["hidden"] * 2
        nvl_bytes = max(nvl_bytes,
                        int(dispatch_cfg.get_nvl_buffer_size_hint(hidden_bytes, world)),
                        int(combine_cfg.get_nvl_buffer_size_hint(hidden_bytes, world)))
    nvl_bytes = (nvl_bytes + 127) // 128 * 128
    return deep_ep.Buffer(group, int(nvl_bytes), 0, low_latency_mode=False,
                          explicitly_destroy=True)


def run_group(deep_ep, group, world, cases, rank, device, opts, results, skipped):
    """Run one buffer's worth of cases (layout/normal or one low-latency QP set)."""
    low_latency = cases[0][1]["mode"] == "ll"
    label = "low-latency" if low_latency else "normal"
    buffer = None
    reason = ""
    try:
        buffer = make_buffer(deep_ep, group, world, cases, opts, low_latency)
    except Exception as exc:  # noqa: BLE001 - environment failures become skips
        reason = f"DeepEP {label} buffer unavailable: {type(exc).__name__}: {exc}"
    # Agree on the failure before any rank starts a collective on this buffer.
    ready = torch.tensor([0 if reason else 1], dtype=torch.int32, device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if int(ready.item()) == 0:
        # A one-sided failure cannot be cleaned up collectively (destroy() is a
        # barrier/finalize); the process exits after the run, so leak it.
        if rank == 0:
            if not reason:
                reason = "DeepEP buffer unavailable on a peer rank"
            skipped.extend({"id": case["id"], "note": reason} for case, _ep in cases)
        return
    try:
        for case, ep in cases:
            cid = case["id"]
            err = ""
            samples = None
            try:
                if ep["mode"] == "layout":
                    fn = make_layout_fn(deep_ep, buffer, case, ep, rank, device, world, opts)
                elif ep["mode"] == "normal":
                    fn = make_normal_fn(deep_ep, buffer, case, ep, rank, device, world, opts)
                else:
                    fn = make_low_latency_fn(deep_ep, buffer, case, ep, rank, device, world, opts)
                samples = rank_time(fn, device, opts["calls"], opts["groups"], opts["warmup"])
            except Exception as exc:  # noqa: BLE001 - a failing case is skipped on every rank
                err = f"{type(exc).__name__}: {exc}"
            flag = torch.tensor([1 if err else 0], dtype=torch.int32, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if int(flag.item()):
                if rank == 0:
                    skipped.append({"id": cid,
                                    "note": f"DeepEP run failed: {err or 'a peer rank failed'}"})
                try:
                    dist.barrier()
                except Exception:  # noqa: BLE001 - the group may already be unusable
                    pass
                continue
            if rank == 0:
                stats = percentiles(samples)
                byte_count = int(case.get("bytes", 0) or 0)
                flop_count = int(case.get("flops", 0) or 0)
                med = stats["median_ms"]
                results.append({
                    "id": cid,
                    "op": case["op"],
                    "dtype": case.get("dtype", "bfloat16"),
                    "rows": int(case.get("rows", 0) or 0),
                    "cols": int(case.get("cols", 0) or 0),
                    "bytes": byte_count,
                    "flops": flop_count,
                    "median_ms": med,
                    "p10_ms": stats["p10_ms"],
                    "p90_ms": stats["p90_ms"],
                    "p99_ms": stats["p99_ms"],
                    "min_ms": stats["min_ms"],
                    "gbs": (byte_count / (med * 1e6)) if med > 0 else 0.0,
                    "tflops": (flop_count / (med * 1e9)) if med > 0 else 0.0,
                    "world": ep["world"] if ep["world"] is not None else world,
                    "note": note_of(ep, world, opts["worst_tokens"]),
                })
    finally:
        try:
            torch.cuda.synchronize(device)
        except Exception:  # noqa: BLE001
            pass
        try:
            if buffer is not None:
                buffer.destroy()
        except Exception:  # noqa: BLE001 - report the failure, do not mask it
            if rank == 0:
                skipped.append({"id": "deepep-buffer",
                                "note": f"DeepEP {label} buffer.destroy() failed"})
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass


def run_cases(cases, world, rank, device, opts, results: list, skipped: list):
    """Run every selected case into `results`/`skipped`; returns the environment."""
    import deep_ep  # noqa: PLC0415 - imported after the GPU/import preflight

    environment = {"ok": True, "reason": "", "deepep": getattr(deep_ep, "__file__", "")}
    try:
        from importlib.metadata import version as _pkg_version  # noqa: PLC0415
        environment["version"] = _pkg_version("deep_ep")
    except Exception:  # noqa: BLE001 - source builds may have no dist-info
        environment["version"] = None
    try:
        environment["sm90_compiled"] = bool(deep_ep.Buffer.is_sm90_compiled())
    except Exception:  # noqa: BLE001
        environment["sm90_compiled"] = None
    if opts["num_sms"] > 0:
        deep_ep.Buffer.set_num_sms(opts["num_sms"])
        environment["num_sms"] = opts["num_sms"]

    group = dist.group.WORLD
    layout_normal = [(case, ep) for case, ep in cases if ep["mode"] in ("layout", "normal")]
    low_latency = [(case, ep) for case, ep in cases if ep["mode"] == "ll"]

    if layout_normal:
        run_group(deep_ep, group, world, layout_normal, rank, device, opts, results, skipped)
    # Low-latency `num_qps_per_rank` must equal the local expert count, so a
    # buffer is built per distinct E/world value.
    by_qps: dict = {}
    for case, ep in low_latency:
        by_qps.setdefault(ep["experts"] // world, []).append((case, ep))
    for qps in sorted(by_qps):
        run_group(deep_ep, group, world, by_qps[qps], rank, device, opts, results, skipped)

    if not results:
        buffer_notes = [entry["note"] for entry in skipped if "buffer unavailable" in entry["note"]]
        if buffer_notes:
            environment["ok"] = False
            environment["reason"] = buffer_notes[0]
    return environment


def worker(rank: int, world: int, port: int, cases: list, out_path: str, opts: dict) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    device = torch.device("cuda", rank)
    results: list = []
    skipped: list = []
    environment = {"ok": True, "reason": "", "deepep": ""}
    try:
        environment = run_cases(cases, world, rank, device, opts, results, skipped)
    except Exception as exc:  # noqa: BLE001 - surfaced by `main` as a skip or an error
        environment = {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                       "deepep": environment.get("deepep", "")}
        if rank == 0:
            print(f"[deepep] rank 0 failed: {environment['reason']}", file=sys.stderr)
            traceback.print_exc()
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass

    if rank == 0:
        device_info = {
            "name": torch.cuda.get_device_name(0),
            "cc": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
            "count": torch.cuda.device_count(),
        }
        spec = {
            "schema": 1,
            "backend": "deepep",
            "torch": torch.__version__,
            "device": device_info,
            "protocol": {"calls": opts["calls"], "groups": opts["groups"],
                         "warmup": opts["warmup"], "world_size": world,
                         "mode": "eager", "worst_tokens": opts["worst_tokens"]},
            "environment": environment,
            "notes": [
                "bytes/flops are the kops case-table values, not DeepEP allocations",
                "normal dispatch recv_x is [tokens*world, hidden] with num_worst_tokens, "
                "unless --no-worst-tokens",
                "low-latency recv_x is [experts/world, world*max_dispatch_tokens, hidden] "
                "(masked worst case)",
                "layout is precomputed outside the timed region; ep_layout times it explicitly",
            ],
            "results": results,
            "skipped": skipped,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=1)
            f.write("\n")


def preflight(world: int) -> str:
    """Return "" when the host can run, or the skip message."""
    try:
        if not torch.cuda.is_available():
            return ("needs CUDA devices with NVLink; torch.cuda.is_available() is False. "
                    "See reports/deepep-baseline.md")
        count = torch.cuda.device_count()
        if count < world:
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(count)) or "none"
            return (f"needs >= {world} CUDA devices with NVLink, found {count} ({names}). "
                    "DeepEP's intranode EP runs over NVLink/IPC; see reports/deepep-baseline.md")
        import deep_ep  # noqa: F401,PLC0415
    except Exception as exc:  # noqa: BLE001
        return (f"'deep_ep' is not importable ({type(exc).__name__}: {exc}). "
                "Build 3rd/DeepEP v1.2.1 against NVSHMEM and install the wheel; "
                "see reports/deepep-baseline.md")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="JSON from bench_communication --dump-cases")
    ap.add_argument("--json", required=True, help="output DeepEP results JSON")
    ap.add_argument("--world-size", type=int, default=2, help="EP group size (default 2)")
    ap.add_argument("--calls", type=int, default=8, help="dispatch+combine round trips per sample")
    ap.add_argument("--groups", type=int, default=8, help="timed samples per case")
    ap.add_argument("--warmup", type=int, default=16, help="warmup round trips per case")
    ap.add_argument("--only", default=None, help="comma-separated case-id substrings")
    ap.add_argument("--skip", default=None, help="comma-separated case-id substrings")
    ap.add_argument("--num-sms", type=int, default=0,
                    help="SMs for DeepEP's normal kernels (0 keeps DeepEP's default 20)")
    ap.add_argument("--max-dispatch-tokens", type=int, default=0,
                    help="low-latency per-rank capacity (0 = the case's token count)")
    ap.add_argument("--no-worst-tokens", dest="worst_tokens", action="store_false",
                    help="let normal dispatch host-sync the recv count instead of allocating "
                         "the case's fixed [tokens*world, hidden] layout")
    ap.add_argument("--no-nvlink-ll", dest="nvlink_ll", action="store_false",
                    help="route low-latency traffic over RDMA only (DeepEP default allows NVLink)")
    args = ap.parse_args()

    if not MIN_WORLD <= args.world_size <= MAX_WORLD:
        print(f"--world-size must be in [{MIN_WORLD}, {MAX_WORLD}] for the intranode EP cases; "
              f"got {args.world_size}", file=sys.stderr)
        return 2
    if args.calls < 1 or args.groups < 1:
        print("--calls and --groups must be >= 1", file=sys.stderr)
        return 2

    reason = preflight(args.world_size)
    if reason:
        print(f"deepep baseline skipped: {reason}", file=sys.stderr)
        return SKIP_EXIT

    with open(args.cases, "r", encoding="utf-8") as f:
        spec = json.load(f)
    cases = spec.get("cases", [])
    if args.only:
        needles = [s for s in args.only.split(",") if s]
        cases = [c for c in cases if any(s in c.get("id", "") for s in needles)]
    if args.skip:
        needles = [s for s in args.skip.split(",") if s]
        cases = [c for c in cases if not any(s in c.get("id", "") for s in needles)]

    runnable, skipped = select_cases(cases, args.world_size)
    if not runnable:
        print("no EP cases to run", file=sys.stderr)
        for entry in skipped[:20]:
            print(f"  skip {entry['id']}: {entry['note']}", file=sys.stderr)
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more", file=sys.stderr)
        return 2

    opts = {
        "calls": args.calls, "groups": args.groups, "warmup": args.warmup,
        "num_sms": args.num_sms, "max_dispatch_tokens": args.max_dispatch_tokens,
        "nvlink_ll": args.nvlink_ll, "worst_tokens": args.worst_tokens,
    }
    port = free_port()
    try:
        torch.multiprocessing.spawn(
            worker,
            args=(args.world_size, port, runnable, args.json, opts),
            nprocs=args.world_size,
            join=True,
        )
    except Exception as exc:  # noqa: BLE001 - a dead rank leaves no JSON
        print(f"deepep: a worker failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1

    try:
        with open(args.json, "r", encoding="utf-8") as f:
            out = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"deepep: no results JSON ({exc})", file=sys.stderr)
        return 1

    env = out.get("environment", {})
    if not env.get("ok", True) and not out.get("results"):
        print(f"deepep baseline skipped: {env.get('reason', 'environment unavailable')}",
              file=sys.stderr)
        return SKIP_EXIT

    # Case-level skips for the ops that were filtered before the spawn.
    if skipped:
        out["skipped"] = skipped + out.get("skipped", [])
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")

    print(f"wrote {len(out['results'])} results, {len(out['skipped'])} skipped -> {args.json}")
    for row in out["results"]:
        print(f"  {row['id']:<52} {row['median_ms']:9.4f} ms  {row['gbs']:8.2f} GB/s")
    runnable_ids = {case["id"] for case, _ep in runnable}
    shown = [entry for entry in out["skipped"] if entry["id"] in runnable_ids]
    for entry in shown:
        print(f"  skip {entry['id']}: {entry['note']}")
    other = len(out["skipped"]) - len(shown)
    if other:
        print(f"  ... {other} rows outside this run skipped (other ops or worlds)")
    if not out["results"]:
        print("deepep: no case produced a result; see the skip notes above", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
