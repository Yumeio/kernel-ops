#!/usr/bin/env python3
"""Export target/draft traces from vLLM speculative decoding (dev-only).

The trace is the exact input of vLLM's rejection sampler:
  * flattened draft token ids and cu_num_draft_tokens,
  * target logits rows (target + bonus),
  * optional draft probabilities when the proposer runs probabilistically,
  * the tokens vLLM emitted (used to compare acceptance length).

The binary format is consumed by
`benchmark/compare/external/spec_trace_runner.cu`; nothing here is linked into
the shipped library.  vLLM 0.32 is pinned for the benchmark.

    python tools/dev/export_vllm_spec_trace.py \
        --tag eagle3_qwen3_8b \
        --model Qwen/Qwen3-8B \
        --speculative-config '{"method":"eagle3","model":"AngelSlim/Qwen3-8B_eagle3","num_speculative_tokens":4}' \
        --steps 24 --max-tokens 64 --output out/b10/traces/eagle3_qwen3_8b
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct
import sys
import time
from typing import Any

import numpy as np
import torch

MAGIC = 0x42313054  # "B10T"
VERSION = 1
HEADER = struct.Struct("<IIIIIII")  # magic, version, B, num_tokens, V, max_spec, has_q


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--speculative-config", default="")
    ap.add_argument("--steps", type=int, default=16, help="decode steps to dump")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="no speculative config (no-spec baseline)")
    return ap.parse_args()


PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number and explain its complexity.",
    "A train travels 240 km in 3 hours. What is its average speed? Show your work.",
    "Summarize the following idea in three bullets: speculative decoding uses a small draft model to propose tokens and a target model to verify them.",
    "Write a SQL query that returns the top 5 customers by total order value.",
    "Explain why a hash table lookup is usually O(1) and when it degrades.",
    "Translate 'The quick brown fox jumps over the lazy dog' into Vietnamese.",
    "List three differences between TCP and UDP.",
    "Write a short unit test for a function that reverses a string.",
]


def save_trace(path: pathlib.Path, b: int, num_tokens: int, vocab: int, max_spec: int,
               cu: np.ndarray, draft: np.ndarray, target_idx: np.ndarray, bonus_idx: np.ndarray,
               target_probs: np.ndarray, draft_probs: np.ndarray | None,
               sampled: np.ndarray, accepted: np.ndarray) -> None:
    has_q = 0 if draft_probs is None else 1
    with path.open("wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, b, num_tokens, vocab, max_spec, has_q))
        f.write(cu.astype(np.int32).tobytes())
        f.write(draft.astype(np.int32).tobytes())
        f.write(target_idx.astype(np.int32).tobytes())
        f.write(bonus_idx.astype(np.int32).tobytes())
        f.write(target_probs.astype(np.float32).tobytes())
        if draft_probs is not None:
            f.write(draft_probs.astype(np.float32).tobytes())
        f.write(sampled.astype(np.int32).tobytes())
        f.write(accepted.astype(np.int32).tobytes())


def main() -> int:
    args = parse_args()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    spec = None if args.baseline else (json.loads(args.speculative_config) if args.speculative_config else None)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler as gpu_sampler

    state: dict[str, Any] = {"count": 0, "traces": [], "path": None}
    original = RejectionSampler.forward
    original_gpu = gpu_sampler.rejection_sample

    def capture_gpu(target_logits, draft_logits, draft_sampled, cu, sampled, num_sampled):
        if state["count"] >= args.steps:
            return
        b = int(cu.numel()) - 1
        cu_h = cu.detach().to("cpu").numpy().astype(np.int32)
        draft_h = draft_sampled.detach().to("cpu").numpy().astype(np.int32)
        target_rows = []
        draft_rows = []
        q_rows = [] if draft_logits is not None else None
        draft_ids = []
        for r in range(b):
            start = int(cu_h[r])
            end = int(cu_h[r + 1])
            # The per-request block is [draft rows..., bonus row].
            if end - start < 1:
                continue
            k = end - start - 1
            target_rows.append(target_logits[start:start + k])
            draft_ids.append(draft_h[start + 1:end])
            if q_rows is not None:
                q_rows.append(draft_logits[r, :k])
            target_rows.append(target_logits[end - 1:end])
        tid = np.concatenate(draft_ids) if draft_ids else np.empty(0, np.int32)
        # Reorder target rows: all drafts, then all bonus rows (our layout).
        draft_targets = []
        bonus_targets = []
        for r in range(b):
            start = int(cu_h[r]); end = int(cu_h[r + 1]); k = max(0, end - start - 1)
            if k:
                draft_targets.append(target_logits[start:start + k])
            bonus_targets.append(target_logits[end - 1:end])
        rows = torch.cat(draft_targets + bonus_targets, dim=0) if (draft_targets or bonus_targets) else target_logits[:0]
        probs = torch.softmax(rows.float(), dim=-1).contiguous()
        q = None
        if q_rows is not None and q_rows:
            q = torch.softmax(torch.cat(q_rows, dim=0).float(), dim=-1).contiguous()
        cu_out = np.zeros((b + 1,), dtype=np.int32)
        for r in range(b):
            start = int(cu_h[r]); end = int(cu_h[r + 1]); cu_out[r + 1] = cu_out[r] + max(0, end - start - 1)
        T = int(cu_out[-1])
        V = int(target_logits.shape[1])
        max_spec = int(max((int(cu_h[r + 1] - cu_h[r] - 1) for r in range(b)), default=0))
        sampled_h = sampled.detach().to("cpu").numpy().astype(np.int32)
        num_sampled_h = num_sampled.detach().to("cpu").numpy().astype(np.int32)
        # vLLM leaves entries after num_sampled uninitialised; they are not
        # emitted tokens.  The first num_sampled entries are the accepted drafts
        # followed by the recovered/bonus token.
        for r in range(b):
            n = int(num_sampled_h[r])
            if 0 <= n < sampled_h.shape[1]:
                sampled_h[r, n:] = -1
        accepted_h = np.maximum(num_sampled_h - 1, 0)
        path = out / f"step_{state['count']:04d}.bin"
        save_trace(path, b, T, V, max_spec, cu_out, tid,
                   np.arange(T, dtype=np.int32),
                   np.arange(T, T + b, dtype=np.int32),
                   probs.detach().to("cpu").numpy().astype(np.float32),
                   q.detach().to("cpu").numpy().astype(np.float32) if q is not None else None,
                   sampled_h, accepted_h)
        state["path"] = "gpu"
        state["traces"].append({"file": path.name, "step": state["count"], "batch": b,
                                "num_draft_tokens": [int(cu_h[r + 1] - cu_h[r] - 1) for r in range(b)],
                                "accepted": accepted_h.tolist(),
                                "has_draft_probs": q is not None})

    def gpu_hook(
        target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
        expanded_idx_mapping, expanded_local_pos, temperature, seed, num_speculative_steps,
        *extra, **kwargs,
    ):
        save = state["count"] < args.steps
        out = original_gpu(
            target_logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
            expanded_idx_mapping, expanded_local_pos, temperature, seed,
            num_speculative_steps, *extra, **kwargs)
        if save:
            capture_gpu(target_logits, draft_logits, draft_sampled, cu_num_logits,
                        out[0], out[1])
            state["count"] += 1
        return out

    def hook(self, metadata, draft_probs, logits, sampling_metadata):
        nonlocal state
        step = state["count"]
        state["count"] += 1
        save = step < args.steps
        # Snapshot before the original mutates logits in place.
        if save:
            lg = logits.detach().float()
            dp = draft_probs.detach().float() if draft_probs is not None else None
            tid = metadata.draft_token_ids.detach().to("cpu").numpy().astype(np.int32)
            cu = metadata.cu_num_draft_tokens.detach().to("cpu").numpy().astype(np.int32)
            ti = metadata.target_logits_indices.detach().to("cpu").numpy().astype(np.int32)
            bi = metadata.bonus_logits_indices.detach().to("cpu").numpy().astype(np.int32)
            vocab = int(logits.shape[1])
            b = int(len(metadata.num_draft_tokens))
            num_tokens = int(tid.size)
            max_spec = int(metadata.max_spec_len)
        out_obj = original(self, metadata, draft_probs, logits, sampling_metadata)
        if save:
            sampled = out_obj.sampled_token_ids.detach().to("cpu").numpy().astype(np.int32)
            # Rows are [accepted drafts..., recovered/bonus, -1 padding].
            accepted = np.zeros((b,), dtype=np.int32)
            for i in range(b):
                row = sampled[i]
                valid = int(np.sum(row >= 0))
                accepted[i] = max(0, min(valid - 1, max_spec))
            # Target rows in the same layout as SpecVerifyParams: drafts first,
            # then B bonus rows.
            rows = torch.cat([lg.index_select(0, torch.as_tensor(ti, device=lg.device)),
                              lg.index_select(0, torch.as_tensor(bi, device=lg.device))], dim=0)
            probs = torch.softmax(rows, dim=-1).contiguous()
            target = probs.detach().to("cpu").numpy().astype(np.float32)
            q = None
            if dp is not None:
                q = dp.detach().to("cpu").numpy().astype(np.float32)
            path = out / f"step_{step:04d}.bin"
            save_trace(path, b, num_tokens, vocab, max_spec, cu, tid, ti, bi, target, q,
                       sampled, accepted)
            state["traces"].append({
                "file": path.name,
                "step": step,
                "batch": b,
                "num_draft_tokens": [int(x) for x in metadata.num_draft_tokens],
                "accepted": accepted.tolist(),
                "has_draft_probs": q is not None,
            })
        return out_obj

    gpu_sampler.rejection_sample = gpu_hook
    RejectionSampler.forward = hook
    try:
        llm = LLM(
            model=args.model,
            speculative_config=spec,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            disable_log_stats=True,
        )
        prompts = (PROMPTS * ((args.batch_size + len(PROMPTS) - 1) // len(PROMPTS)))[: args.batch_size]
        params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
        t0 = time.time()
        outputs = llm.generate(prompts, params)
        elapsed = time.time() - t0
    finally:
        RejectionSampler.forward = original
        gpu_sampler.rejection_sample = original_gpu

    manifest = {
        "tag": args.tag,
        "model": args.model,
        "speculative_config": spec,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "elapsed_s": elapsed,
        "traces": state["traces"],
        "outputs": [{"prompt": p, "text": o.outputs[0].text,
                     "token_ids": list(o.outputs[0].token_ids)} for p, o in zip(prompts, outputs)],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[trace] {len(state['traces'])} steps -> {out} ({elapsed:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
