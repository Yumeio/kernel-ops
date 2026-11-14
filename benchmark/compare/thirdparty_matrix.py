#!/usr/bin/env python3
"""Family -> third-party adapter matrix for benchmark/compare/run_thirdparty.py.

Dev-only. This file is data, not a build dependency. Keep the `impls` list to
adapters that exist in thirdparty.py; `external` entries are binaries built by
tools/dev/build_* scripts. Families with no generic third-party op are marked
`reference_only`/`no_standalone_kernel` so the coverage report is honest.
"""
from __future__ import annotations

# target: benchmark/bench_<target>.cu
# impls: thirdparty.py --impl names
# external: external harness names known to run_thirdparty.py
# special: csa (csa_torch.py), comm (communication_torch.py), mega (internal)
FAMILIES: dict[str, dict] = {
    "model_ops": {
        "target": "bench_model_ops",
        "impls": [],
        "external": [],
        "special": None,
        "status": "reference_only",
        "references": "DeepGEMM mhc_reference/sm100_mega_mhc, vLLM attn_res_kernel, hpc-ops iHC",
        "note": "mHC/AttnRes are model-specific reference kernels, not generic ops",
    },
    "norm": {
        "target": "bench_norm",
        "impls": ["vllm", "flashinfer", "sglang"],
        "external": [],
        "special": None,
        "status": "wired",
        "references": "vLLM, FlashInfer, SGLang sgl-kernel",
        "note": "",
    },
    "activation": {
        "target": "bench_activation",
        "impls": ["vllm", "flashinfer", "sglang"],
        "external": [],
        "special": None,
        "status": "wired",
        "references": "vLLM, FlashInfer, SGLang sgl-kernel",
        "note": "",
    },
    "position": {
        "target": "bench_position",
        "impls": ["vllm", "sglang"],
        "external": [],
        "special": None,
        "status": "wired",
        "references": "vLLM rotary_embedding, SGLang rotary_embedding",
        "note": "",
    },
    "embedding": {
        "target": "bench_embedding",
        "impls": [],
        "external": [],
        "special": None,
        "status": "no_standalone_kernel",
        "references": "PyTorch F.embedding; vLLM/SGLang embedding are model layers (TP allreduce included)",
        "note": "gather/memory-bound; compare to PyTorch only",
    },
    "sampling": {
        "target": "bench_sampling",
        "impls": ["flashinfer"],
        "external": ["hpc_sampler", "sglang_tree"],
        "special": None,
        "status": "wired",
        "references": "FlashInfer sampling, hpc-ops sampler, SGLang tree target-only",
        "note": "hpc-ops/sglang_tree cover spec cases only",
    },
    "chains": {
        "target": "bench_chains",
        "impls": [],
        "external": [],
        "special": None,
        "status": "torch_only",
        "references": "PyTorch eager/builtin",
        "note": "fused chains; no one-call third-party equivalent",
    },
    "kvcache": {
        "target": "bench_kvcache",
        "impls": ["vllm", "flashinfer"],
        "external": [],
        "special": None,
        "status": "wired",
        "references": "vLLM reshape_and_cache_flash, FlashInfer append_paged_kv_cache",
        "note": "",
    },
    "state": {
        "target": "bench_state",
        "impls": ["fla", "mamba_ssm", "causal_conv1d"],
        "external": [],
        "special": None,
        "status": "wired",
        "references": "FLA, mamba_ssm, causal-conv1d, vLLM selective_scan, SGLang causal_conv1d",
        "note": "adapters landed; packages required; fused_delta_decode/delta_gating unsupported",
    },
    "quantization": {
        "target": "bench_quantization",
        "impls": ["vllm", "flashinfer"],
        "external": [],
        "special": None,
        "status": "wired",
        "references": "vLLM FP8/INT8 quant, FlashInfer fp4_quantize",
        "note": "",
    },
    "gemm": {
        "target": "bench_gemm",
        "impls": ["vllm", "sglang", "bnb", "deepgemm", "trtllm"],
        "external": ["llamacpp"],
        "special": None,
        "status": "wired",
        "references": "vLLM Marlin/Machete, SGLang fp8, bitsandbytes NF4, llama.cpp mmvq, DeepGEMM, TRT-LLM",
        "note": "llama.cpp harness is hot-L2; compare with --flush-l2 0",
    },
    "moe": {
        "target": "bench_moe",
        "impls": ["vllm", "sglang", "trtllm"],
        "external": ["attention_rs_w2"],
        "special": None,
        "status": "wired",
        "references": "vLLM topk/align, SGLang moe_fused_gate, attention.rs W2, TRT-LLM",
        "note": "attention.rs W2 is B17 experimental",
    },
    "merge": {
        "target": "bench_merge",
        "impls": ["sdpa"],
        "external": ["attention_rs_lowbit"],
        "special": None,
        "status": "wired",
        "references": "torch SDPA for attention cases, attention.rs lowbit KV",
        "note": "sdpa skips non-attention merge cases",
    },
    "mla": {
        "target": "bench_mla",
        "impls": ["flashmla"],
        "external": [],
        "special": None,
        "status": "wired",
        "references": "FlashMLA (dense/sparse), attn-rs mla_paged_attention",
        "note": "flashmla adapter landed; package + SM90/SM100 required",
    },
    "sparse": {
        "target": "bench_sparse",
        "impls": ["flashmla"],
        "external": [],
        "special": "csa",
        "status": "wired",
        "references": "PyTorch CSA/HCA (csa_torch.py); FlashMLA sparse, DeepGEMM indexer",
        "note": "csa_torch covers kv_compress/compressed_attention; flashmla adapter needs SM90/SM100",
    },
    "hyper_connection": {
        "target": "bench_hyper_connection",
        "impls": [],
        "external": [],
        "special": None,
        "status": "reference_only",
        "references": "DeepGEMM mhc_reference/sm100_mega_mhc, hpc-ops iHC, vLLM attn_res",
        "note": "model-specific reference kernels, not generic ops",
    },
    "mega": {
        "target": "bench_mega",
        "impls": [],
        "external": [],
        "special": "mega",
        "status": "reference_only",
        "references": "DeepGEMM sm100_bf16_mega_moe, sm100_fp8_fp4_mega_moe, sm100_mega_mhc",
        "note": "B16 gate is internal modular+graph; DeepGEMM Mega MoE needs SM100 harness",
    },
    "communication": {
        "target": "bench_communication",
        "impls": [],
        "external": ["nccl", "deepep"],
        "special": "comm",
        "status": "wired",
        "references": "NCCL collectives; DeepEP EP dispatch/combine; vLLM/SGLang custom allreduce; hpc-ops fused allreduce",
        "note": "DeepEP is the B12 EP gate comparison; needs multi-GPU NVLink",
    },
}

# Known references that are not runnable through the current adapters.
# (family, name, status, hardware, note)
KNOWN_REFERENCES = [
    ("embedding", "PyTorch F.embedding", "torch_only", "any", "no standalone third-party kernel"),
    ("state", "FLA (GDN/KDA)", "wired", "Linux + Triton, SM80/SM90", "ran on SM86 (3 state cases)"),
    ("state", "mamba_ssm", "wired", "Linux + CUDA extension", "adapter landed; package not installed"),
    ("state", "causal-conv1d", "wired", "Linux + CUDA extension", "adapter landed; package not installed"),
    ("state", "vLLM selective_scan", "reference_only", "SM80", "source reference"),
    ("model_ops", "DeepGEMM mhc_reference", "reference_only", "Python, any", "official mHC reference"),
    ("model_ops", "DeepGEMM sm100_mega_mhc", "needs_hardware", "SM100/B200", ""),
    ("model_ops", "vLLM attn_res_kernel", "needs_hardware", "SM100, H=7168", "Kimi K3"),
    ("hyper_connection", "hpc-ops iHC", "reference_only", "SM90", "similar, not identical to mHC"),
    ("mega", "DeepGEMM Mega MoE", "needs_hardware", "SM100/B200", "no generic callable op"),
    ("communication", "DeepEP", "needs_hardware", "multi-GPU NVLink + NVSHMEM", "harness landed; needs >=2 GPUs"),
    ("communication", "NCCL", "wired", "multi-GPU", "collectives"),
    ("communication", "vLLM custom allreduce", "reference_only", "multi-GPU", ""),
    ("communication", "SGLang custom allreduce", "reference_only", "multi-GPU", ""),
    ("communication", "hpc-ops fused allreduce+RMSNorm", "reference_only", "SM90, multi-GPU", ""),
    ("communication", "TensorRT-LLM Lamport allreduce", "reference_only", "multi-GPU", ""),
    ("mla", "FlashMLA", "needs_hardware", "SM90/SM100", "adapter landed; package/arch required"),
    ("sparse", "FlashMLA sparse / DeepGEMM indexer", "needs_hardware", "SM90/SM100", "adapter landed; package/arch required"),
    ("gemm", "DeepGEMM", "needs_hardware", "SM90/SM100", "adapter landed; package/arch required"),
    ("gemm", "TensorRT-LLM / trtllm-gen", "needs_hardware", "SM100 via FlashInfer", "adapter landed; package/container required"),
]

PY_IMPLS = [
    "vllm", "flashinfer", "sglang", "sdpa", "fa2", "bnb",
    "fla", "mamba_ssm", "causal_conv1d",
    "deepgemm", "flashmla", "trtllm",
]
EXTERNALS = {
    "llamacpp": {
        "family": "gemm",
        "paths": ["build/compare/llamacpp_mmvq", "out/rtx5060/bin/llamacpp_mmvq", "/tmp/llamacpp_mmvq"],
        "build": "README.md#llamacpp-gguf-gemv",
    },
    "attention_rs_lowbit": {
        "family": "merge",
        "paths": ["build/compare/attention_rs_lowbit"],
        "build": "bash tools/dev/build_lowbit_baselines.sh",
    },
    "attention_rs_w2": {
        "family": "moe",
        "paths": ["build/compare/attention_rs_moe_w2"],
        "build": "bash tools/dev/build_lowbit_baselines.sh",
    },
    "hpc_sampler": {
        "family": "sampling",
        "paths": ["out/b10/hpc_sampler"],
        "build": "bash tools/dev/build_external_baselines.sh hpc",
    },
    "sglang_tree": {
        "family": "sampling",
        "paths": ["out/b10/sglang_tree"],
        "build": "bash tools/dev/build_external_baselines.sh sglang",
    },
    "nccl": {
        "family": "communication",
        "paths": ["out/b12/nccl_baseline"],
        "build": "bash tools/dev/build_comm_baselines.sh",
    },
    "deepep": {
        "family": "communication",
        "paths": ["benchmark/compare/deepep_baseline.py"],
        "build": "see reports/deepep-baseline.md (DeepEP v1.2.1 + NVSHMEM + multi-GPU)",
    },
}
