// =============================================================================
// kernel-ops | tools/header_check/check_all_headers.cc
// Includes every library header in the canonical dependency order (docs/TREE.md
// "Include order"). Used by CI as a syntax/ABI smoke test on hosts without nvcc:
//   g++     -std=c++17 -fsyntax-only -Wall -Wextra -I. tools/header_check/check_all_headers.cc
//   clang++ -std=c++17 -fsyntax-only -I. tools/header_check/check_all_headers.cc
// With nvcc the same order is used by every .cu translation unit (minus the shim).
// =============================================================================
#include "tools/header_check/cuda_host_shim.h"

// 0. common
#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/rng.cuh"
#include "csrc/common/sync.cuh"
// 1. elementwise families
#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"
#include "csrc/activation/activation_kernels.cuh"
#include "csrc/activation/ops.h"
#include "csrc/position/rope_kernels.cuh"
#include "csrc/position/ops.h"
#include "csrc/embedding/embedding_kernels.cuh"
#include "csrc/embedding/ops.h"
// 2. quantization
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/quantization/ops.h"
// 3. gemm
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/dense_gemm_kernels.cuh"
#include "csrc/gemm/fp8_gemm_kernels.cuh"
#include "csrc/gemm/int8_gemm_kernels.cuh"
#include "csrc/gemm/fp4_gemm_kernels.cuh"
#include "csrc/gemm/mixed_input_gemm_kernels.cuh"
#include "csrc/gemm/gguf_gemm_kernels.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/gemm/gemm_prep_kernels.cuh"
#include "csrc/gemm/ops.h"
// 4. moe
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/moe/permute_kernels.cuh"
#include "csrc/moe/fused_moe_kernels.cuh"
#include "csrc/moe/ops.h"
// 5. kv cache + attention
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/kvcache/kvcache_kernels.cuh"
#include "csrc/kvcache/ops.h"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/merge_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/lowbit_kernels.cuh"
#include "csrc/attention/ops.h"
// 6. hybrid state
#include "csrc/state/conv1d_kernels.cuh"
#include "csrc/state/ssm_kernels.cuh"
#include "csrc/state/delta_rule_kernels.cuh"
#include "csrc/state/linear_attention_kernels.cuh"
#include "csrc/state/state_cache_kernels.cuh"
#include "csrc/state/ops.h"
// 7. sampling
#include "csrc/sampling/sampling_kernels.cuh"
#include "csrc/sampling/logits_processor_kernels.cuh"
#include "csrc/sampling/speculative_kernels.cuh"
#include "csrc/sampling/ops.h"
// 8. communication
#include "csrc/communication/allreduce_kernels.cuh"
#include "csrc/communication/ep_kernels.cuh"
#include "csrc/communication/comm_gemm_kernels.cuh"
#include "csrc/communication/ops.h"
// 9. model ops
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/residual_kernels.cuh"
#include "csrc/model_ops/ops.h"
// 10. execution, megakernels, runtime
#include "csrc/execution/persistent.cuh"
#include "csrc/execution/launch.h"
#include "csrc/mega/mega_plan.cuh"
#include "csrc/mega/mega_kernels.cuh"
#include "csrc/mega/ops.h"
#include "csrc/runtime/registry.h"
#include "csrc/runtime/trace.h"

int main() { return 0; }
