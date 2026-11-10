// =============================================================================
// kernel-ops | csrc/gemm/marlin_moe.cu
// B7 row 6: MoE mode of marlin_gemm_kernel (sorted_token_ids / expert_ids blocks).
//   marlin_moe_gemm_kernel  the dense Marlin body (dev::marlin_body, MOE = true) over the rows of
//                           moe::align_block_size: block b of moe_block_size slots uses expert expert_ids[b]
// Host entry: gemm::marlin_moe_gemm (also reached through marlin_gemm when sorted_token_ids is set).
//
// Semantics (vLLM marlin_moe_wna16 / fused_moe "sorted_token_ids" contract):
//   rows      : slot i < *num_tokens_post_pad holds id = sorted_token_ids[i]; valid iff 0 <= id < prob_m * top_k.
//               A row = id / top_k (top_k = 1 for an already expanded [M * topk, K] input), C row = id.
//   expert    : e = expert_ids[i / moe_block_size]; e < 0 skips the whole block (rows untouched).
//   weights   : expert e at b + e * K * N * bits / 32 words, scales + e * num_groups * N, zeros + e * num_groups
//               * N / 8 words, bias + e * N; no bound check on e (the caller's expert count).
//   output    : C[id, n] = T((acc * channel_scale / global_scale + bias[e, n]) * (mul_topk_weights ?
//               topk_weights[id] : 1)), rounded once; padded slots and skipped blocks are never written.
// The grid is persistent (SMs x occupancy) because the row count lives on the device. One CTA tile of
// 16 / 32 / 64 rows is one moe_block_size block (16 / 32 / 64), so a tile never mixes experts.
// Act-order (g_idx / perm) is not supported in MoE mode (kNotImplemented).
// =============================================================================
#include <algorithm>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
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

namespace kops::gemm {

/// Defined in marlin_dense.cu (validation shared by the dense and MoE modes).
Status marlin_check_common(const MarlinParams& p, quant::MarlinWeightType w, DType act, const char* who);

namespace {

/// Same warp grid as marlin_gemm_kernel (marlin_dense.cu): THREAD_M_BLOCKS x 16 rows = one sorted block.
template <typename T, quant::MarlinWeightType W, int MB, int NB>
__global__ void __launch_bounds__(128) marlin_moe_gemm_kernel(const __grid_constant__ MarlinParams params) {
  constexpr int kTok = MB * 16;
  constexpr int WN = NB * 16 / 64;
  constexpr int TT = kTok / 8 < 4 ? kTok / 8 : 4;
  constexpr int WM = kTok / (8 * TT);
  constexpr int WK = 4 / (WN * WM);
  dev::marlin_body<T, static_cast<int>(W), TT, WM, WN, WK, true>(params);
}

Status moe_launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what, cudaGetErrorString(err));
}

template <typename T, quant::MarlinWeightType W, int MB, int NB>
Status launch_moe(const MarlinParams& p, const LaunchContext& ctx) {
  constexpr int kTok = MB * 16;
  constexpr int WN = NB * 16 / 64;
  constexpr int TT = kTok / 8 < 4 ? kTok / 8 : 4;
  constexpr int WM = kTok / (8 * TT);
  constexpr int WK = 4 / (WN * WM);
  using Cfg = dev::MarlinCfg<T, static_cast<int>(W), TT, WM, WN, WK>;
  auto kern = marlin_moe_gemm_kernel<T, W, MB, NB>;
  static const cudaError_t attr = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmem);
  if (attr != cudaSuccess) return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "marlin_moe_gemm_kernel: %s", cudaGetErrorString(attr));
  static const int occ = [&] {
    int v = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, 128, Cfg::kSmem);
    return std::max(v, 1);
  }();
  int sms = ctx.compute_units;
  if (sms <= 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx.device);
  const int grid = std::max(1, sms * occ);
  kern<<<static_cast<unsigned>(grid), 128, Cfg::kSmem, reinterpret_cast<cudaStream_t>(ctx.stream)>>>(p);
  return moe_launch_error("marlin_moe_gemm_kernel");
}

template <typename T, quant::MarlinWeightType W>
Status launch_moe_m(const MarlinParams& p, const LaunchContext& ctx) {
  if (p.moe_block_size == 16) return launch_moe<T, W, 1, 4>(p, ctx);
  if (p.moe_block_size == 32) return launch_moe<T, W, 2, 8>(p, ctx);
  return launch_moe<T, W, 4, 8>(p, ctx);
}

template <typename T>
Status launch_moe_w(const MarlinParams& p, quant::MarlinWeightType w, const LaunchContext& ctx) {
  switch (w) {
    case quant::MarlinWeightType::kU4: return launch_moe_m<T, quant::MarlinWeightType::kU4>(p, ctx);
    case quant::MarlinWeightType::kU4B8: return launch_moe_m<T, quant::MarlinWeightType::kU4B8>(p, ctx);
    case quant::MarlinWeightType::kU8B128: return launch_moe_m<T, quant::MarlinWeightType::kU8B128>(p, ctx);
    case quant::MarlinWeightType::kFE4M3: return launch_moe_m<T, quant::MarlinWeightType::kFE4M3>(p, ctx);
    default: return launch_moe_m<T, quant::MarlinWeightType::kFE2M1>(p, ctx);
  }
}

}  // namespace

Status marlin_moe_gemm(const MarlinParams& params, quant::MarlinWeightType wtype, DType act_dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "marlin_moe_gemm: CUDA backend only");
  Status st = marlin_check_common(params, wtype, act_dtype, "marlin_moe_gemm");
  if (st != Status::kSuccess) return st;
  const MarlinParams& p = params;
  if (p.sorted_token_ids == nullptr || p.expert_ids == nullptr || p.num_tokens_post_pad == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "marlin_moe_gemm: sorted_token_ids, expert_ids and num_tokens_post_pad are required");
  }
  if (p.moe_block_size != 16 && p.moe_block_size != 32 && p.moe_block_size != 64) {
    return detail::set_error(Status::kUnsupportedShape, "marlin_moe_gemm: moe_block_size %d not in {16, 32, 64}", p.moe_block_size);
  }
  if (p.top_k < 1 || static_cast<int64_t>(p.prob_m) * p.top_k > INT32_MAX) {
    return detail::set_error(Status::kInvalidArgument, "marlin_moe_gemm: top_k must be >= 1 with prob_m * top_k < 2^31");
  }
  if (p.mul_topk_weights && p.topk_weights == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "marlin_moe_gemm: mul_topk_weights needs topk_weights");
  }
  if (p.g_idx != nullptr || p.perm != nullptr) {
    return detail::set_error(Status::kNotImplemented, "marlin_moe_gemm: act-order (g_idx / perm) is not supported in MoE mode");
  }
  if (p.use_atomic_add && ctx.deterministic) {
    return detail::set_error(Status::kInvalidArgument, "marlin_moe_gemm: use_atomic_add is disabled by ctx.deterministic");
  }
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "marlin_moe_gemm: mma.sync bf16 / fp16 needs SM80 or newer (arch %d)", ctx.arch_id);
  }
  return act_dtype == DType::kBFloat16 ? launch_moe_w<__nv_bfloat16>(p, wtype, ctx) : launch_moe_w<__half>(p, wtype, ctx);
}

}  // namespace kops::gemm
