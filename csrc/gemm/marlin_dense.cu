// =============================================================================
// kernel-ops | csrc/gemm/marlin_dense.cu
// B7 row 6: Marlin mixed-input GEMM on mma.sync (SM80/86/89/120), dense rows.
//   marlin_gemm_kernel          weights in the Marlin tile layout of quantization::{gptq,awq,fp4}_to_marlin
//                               (U4 + zeros, U4B8, U8B128, FE4M3, FE2M1) x bf16 / fp16 activations; body
//                               dev::marlin_body (common/device_impl.cuh), shared with the MoE mode
//   marlin_permute_cols_kernel  act-order: a_tmp[m, k] = a[m, perm[k]] before the GEMM (variant of the row)
// Host entry: gemm::marlin_gemm (forwards the sorted_token_ids mode to marlin_moe_gemm, gemm/marlin_moe.cu).
//
// Contract: csrc/gemm/mixed_input_gemm_kernels.cuh (MarlinParams, marlin_gemm_kernel), csrc/gemm/ops.h,
// CONTRACTS.md §1d.
//
// Provenance.  The weight / scale / zero layouts are the vLLM Marlin layouts (Apache-2.0,
// vllm:libtorch_stable/quantization/marlin/marlin_template.h, dequant.h), reproduced by this repo's repack
// kernels; the mainloop is written for this repo (cp.async ring + ldmatrix + mma.sync with the weights as the
// MMA A operand), not copied. Numerics differ from vLLM Marlin on purpose: the dequantised weight (q - z, or the
// fp8 / e2m1 code) is exact in the activation dtype, every product is exact in fp32, and a group's scale
// multiplies the group's fp32 partial sum instead of the weight (no fp16 range limit for UE8M0 scales, one
// rounding fewer). The K split of the decode tile sums the warps in a fixed order: no atomics, no global
// reduce, so use_atomic_add / use_fp32_reduce / locks / c_tmp are accepted and unused.
//
// Dev-only tuning knobs (not part of the API/ABI): MARLIN_SPLITS caps the dense split-K factor, and
// MARLIN_DEBUG=1 prints the chosen tile/grid to stderr. Both are read at launch; unset means the
// heuristic above and no output.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

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

/// THREAD_M_BLOCKS x 16 tokens and THREAD_N_BLOCKS x 16 columns per CTA of 128 threads; the warp grid is
/// derived: WN = columns / 64 Marlin n-tiles, TT = min(4, tokens / 8) token tiles per warp, WM = tokens /
/// (8 TT) warp rows, WK = 4 / (WN WM) warps splitting K. THREAD_K_BLOCKS (4: 64 K per stage), STAGES (4),
/// GROUP_BLOCKS (-1: from params) and IS_ZP_FLOAT (false) are fixed in this body.
template <typename scalar_t, quant::MarlinWeightType kWType, int THREADS, int THREAD_M_BLOCKS, int THREAD_N_BLOCKS,
          int THREAD_K_BLOCKS, bool M_BLOCK_SIZE_8, int STAGES, int GROUP_BLOCKS, bool IS_ZP_FLOAT>
__global__ void __launch_bounds__(THREADS) marlin_gemm_kernel(const __grid_constant__ MarlinParams params) {
  static_assert(THREADS == 128 && THREAD_K_BLOCKS == 4 && STAGES == dev::kMarlinStages && GROUP_BLOCKS == -1 && !IS_ZP_FLOAT &&
                    !M_BLOCK_SIZE_8,
                "the kops Marlin body has one pipeline shape");
  constexpr int kTok = THREAD_M_BLOCKS * 16;
  constexpr int WN = THREAD_N_BLOCKS * 16 / 64;
  constexpr int TT = kTok / 8 < 4 ? kTok / 8 : 4;
  constexpr int WM = kTok / (8 * TT);
  constexpr int WK = 4 / (WN * WM);
  dev::marlin_body<scalar_t, static_cast<int>(kWType), TT, WM, WN, WK, false>(params);
}

namespace {

/// a_tmp[m, k] = a[m, perm[k]]: one thread per 8 output elements of a row.
template <typename T>
__global__ void marlin_permute_cols_kernel(T* __restrict__ out, const T* __restrict__ a, const int64_t lda, const int32_t* __restrict__ perm,
                                           const int m, const int k) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_row = k / 8;
  if (i >= static_cast<int64_t>(m) * per_row) return;
  const int64_t row = i / per_row;
  const int c0 = static_cast<int>(i - row * per_row) * 8;
  const T* src = a + row * lda;
  dev::Pack<T, 8> v;
  #pragma unroll
  for (int j = 0; j < 8; ++j) v.x[j] = src[perm[c0 + j]];
  dev::store_pack<8>(out + row * k + c0, v);
}

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what, cudaGetErrorString(err));
}

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

template <typename T, quant::MarlinWeightType W, int MB, int NB>
Status launch_marlin(const MarlinParams& p, const LaunchContext& ctx) {
  constexpr int kTok = MB * 16;
  constexpr int WN = NB * 16 / 64;
  constexpr int TT = kTok / 8 < 4 ? kTok / 8 : 4;
  constexpr int WM = kTok / (8 * TT);
  constexpr int WK = 4 / (WN * WM);
  using Cfg = dev::MarlinCfg<T, static_cast<int>(W), TT, WM, WN, WK>;
  auto kern = marlin_gemm_kernel<T, W, 128, MB, NB, 4, false, dev::kMarlinStages, -1, false>;
  static const cudaError_t attr = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmem);
  if (attr != cudaSuccess) return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(attr), "marlin_gemm_kernel: %s", cudaGetErrorString(attr));
  static const int occ = [&] {
    int v = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, 128, Cfg::kSmem);
    return std::max(v, 1);
  }();
  int sms = ctx.compute_units;
  if (sms <= 0) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx.device);
  const int64_t tiles = ((p.prob_m + kTok - 1) / kTok) * ((p.prob_n + NB * 16 - 1) / (NB * 16));
  const int64_t slots = static_cast<int64_t>(sms) * occ;   // co-resident CTAs
  int64_t grid = std::max<int64_t>(1, std::min<int64_t>(tiles, slots));
  // Global K split when the tiles leave SMs idle and the caller gave c_tmp ([M, N] fp32) + locks
  // ([sms * 16] int32, zero): every one of the tiles x splits CTAs is co-resident (the serial
  // slice hand-off spins), each slice keeps >= kMinSliceStages pipeline stages.
  constexpr int kMinSliceStages = 8;
  const int64_t nst = (p.prob_k + dev::kMarlinStageK - 1) / dev::kMarlinStageK;
  if (p.c_tmp != nullptr && p.locks != nullptr && tiles <= static_cast<int64_t>(sms) * 16 && 2 * tiles <= slots) {
    int64_t splits = std::min<int64_t>({slots / tiles, nst / kMinSliceStages, 16});
    if (getenv("MARLIN_SPLITS")) splits = std::min<int64_t>(slots / tiles, atoi(getenv("MARLIN_SPLITS")));
    if (splits >= 2) grid = tiles * splits;
  }
  if (getenv("MARLIN_DEBUG")) fprintf(stderr, "[marlin] MB %d NB %d smem %d occ %d tiles %lld grid %lld\n", MB, NB, Cfg::kSmem, occ, (long long)tiles, (long long)grid);
  kern<<<static_cast<unsigned>(grid), 128, Cfg::kSmem, reinterpret_cast<cudaStream_t>(ctx.stream)>>>(p);
  return launch_error("marlin_gemm_kernel");
}

/// Explicit instantiation list: (activation, weight type) x tile (decode 16 x 64 K-split, mid 32 x 128, 64 x 128).
template <typename T, quant::MarlinWeightType W>
Status launch_marlin_m(const MarlinParams& p, const LaunchContext& ctx) {
  if (p.prob_m <= 16) return launch_marlin<T, W, 1, 4>(p, ctx);
  if (p.prob_m <= 32) return launch_marlin<T, W, 2, 8>(p, ctx);
  return launch_marlin<T, W, 4, 8>(p, ctx);
}

template <typename T>
Status launch_marlin_w(const MarlinParams& p, quant::MarlinWeightType w, const LaunchContext& ctx) {
  switch (w) {
    case quant::MarlinWeightType::kU4: return launch_marlin_m<T, quant::MarlinWeightType::kU4>(p, ctx);
    case quant::MarlinWeightType::kU4B8: return launch_marlin_m<T, quant::MarlinWeightType::kU4B8>(p, ctx);
    case quant::MarlinWeightType::kU8B128: return launch_marlin_m<T, quant::MarlinWeightType::kU8B128>(p, ctx);
    case quant::MarlinWeightType::kFE4M3: return launch_marlin_m<T, quant::MarlinWeightType::kFE4M3>(p, ctx);
    default: return launch_marlin_m<T, quant::MarlinWeightType::kFE2M1>(p, ctx);
  }
}

}  // namespace

/// Validation shared by the dense and MoE modes (declared in ops.h through marlin_gemm's contract).
Status marlin_check_common(const MarlinParams& p, quant::MarlinWeightType w, DType act, const char* who);

Status marlin_check_common(const MarlinParams& p, quant::MarlinWeightType w, DType act, const char* who) {
  using MW = quant::MarlinWeightType;
  if (act != DType::kBFloat16 && act != DType::kFloat16) return detail::set_error(Status::kUnsupportedDtype, "%s: activations must be bf16 or fp16", who);
  if (static_cast<int>(w) > static_cast<int>(MW::kFE2M1)) return detail::set_error(Status::kInvalidArgument, "%s: unknown weight type", who);
  if (p.prob_m < 1 || p.prob_n < 1 || p.prob_k < 1) return detail::set_error(Status::kInvalidArgument, "%s: empty problem", who);
  if (p.a == nullptr || p.b == nullptr || p.c == nullptr || p.scales == nullptr) return detail::set_error(Status::kInvalidArgument, "%s: null operand", who);
  if (p.prob_n % 64 != 0 || p.prob_k % 16 != 0) return detail::set_error(Status::kUnsupportedShape, "%s: N must be a multiple of 64 and K of 16 (Marlin tiles)", who);
  if (p.lda < p.prob_k || p.ldc < p.prob_n) return detail::set_error(Status::kInvalidArgument, "%s: leading dimensions smaller than the extents", who);
  const bool fp4 = w == MW::kFE2M1;
  if (fp4) {
    if (p.group_size != 16 && p.group_size != 32) return detail::set_error(Status::kUnsupportedShape, "%s: FE2M1 needs group 16 (NVFP4, E4M3 scales) or 32 (MXFP4, UE8M0)", who);
    if (p.zeros != nullptr || p.g_idx != nullptr) return detail::set_error(Status::kInvalidArgument, "%s: FE2M1 takes no zeros or act-order", who);
  } else if (p.global_scale != nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: global_scale belongs to NVFP4 (FE2M1)", who);
  }
  const bool channel = !fp4 && (p.group_size < 0 || p.num_groups <= 1);
  if (!channel) {
    if (p.group_size != 16 && p.group_size != 32 && p.group_size != 64 && p.group_size != 128) {
      return detail::set_error(Status::kUnsupportedShape, "%s: group_size must be -1 (per channel), 32, 64 or 128 (16 / 32 for FE2M1)", who);
    }
    if (p.g_idx == nullptr && (p.prob_k % p.group_size != 0 || p.num_groups != p.prob_k / p.group_size)) {
      return detail::set_error(Status::kInvalidArgument, "%s: num_groups must be K / group_size", who);
    }
  }
  if (p.zeros != nullptr && w != MW::kU4) return detail::set_error(Status::kInvalidArgument, "%s: zero points belong to kU4 (AWQ)", who);
  if (w == MW::kU4 && p.zeros == nullptr) return detail::set_error(Status::kInvalidArgument, "%s: kU4 needs Marlin zero points (use kU4B8 for symmetric GPTQ)", who);
  if (p.g_idx != nullptr && (p.perm == nullptr || channel)) return detail::set_error(Status::kInvalidArgument, "%s: act-order needs perm and grouped scales", who);
  if (!aligned(p.a, 16) || (p.lda * 2) % 16 != 0 || !aligned(p.b, 16) || !aligned(p.scales, 16) || !aligned(p.c, 4) || p.ldc % 2 != 0) {
    return detail::set_error(Status::kMisaligned, "%s: a, b, scales 16 B aligned, lda a multiple of 8, c 4 B aligned with even ldc", who);
  }
  return Status::kSuccess;
}

Status marlin_gemm(const MarlinParams& params, quant::MarlinWeightType wtype, DType act_dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "marlin_gemm: CUDA backend only");
  if (params.sorted_token_ids != nullptr) return marlin_moe_gemm(params, wtype, act_dtype, ctx);
  Status st = marlin_check_common(params, wtype, act_dtype, "marlin_gemm");
  if (st != Status::kSuccess) return st;
  if (params.use_atomic_add && ctx.deterministic) {
    return detail::set_error(Status::kInvalidArgument, "marlin_gemm: use_atomic_add is disabled by ctx.deterministic");
  }
  MarlinParams p = params;
  if (p.g_idx != nullptr) {
    if (p.a_tmp == nullptr || !aligned(p.a_tmp, 16)) return detail::set_error(Status::kInvalidArgument, "marlin_gemm: act-order needs a 16 B aligned a_tmp [M, K]");
  }
  if (ctx.arch_id >= KOPS_TARGET_SM70 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "marlin_gemm: mma.sync bf16 / fp16 needs SM80 or newer (arch %d)", ctx.arch_id);
  }
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  const bool bf = act_dtype == DType::kBFloat16;
  if (p.g_idx != nullptr) {
    const int64_t work = static_cast<int64_t>(p.prob_m) * (p.prob_k / 8);
    const unsigned blocks = static_cast<unsigned>((work + 255) / 256);
    if (bf) {
      marlin_permute_cols_kernel<__nv_bfloat16><<<blocks, 256, 0, s>>>(static_cast<__nv_bfloat16*>(p.a_tmp), static_cast<const __nv_bfloat16*>(p.a), p.lda, p.perm, p.prob_m, p.prob_k);
    } else {
      marlin_permute_cols_kernel<__half><<<blocks, 256, 0, s>>>(static_cast<__half*>(p.a_tmp), static_cast<const __half*>(p.a), p.lda, p.perm, p.prob_m, p.prob_k);
    }
    st = launch_error("marlin_permute_cols_kernel");
    if (st != Status::kSuccess) return st;
    p.a = p.a_tmp;
    p.lda = p.prob_k;
  }
  return bf ? launch_marlin_w<__nv_bfloat16>(p, wtype, ctx) : launch_marlin_w<__half>(p, wtype, ctx);
}

}  // namespace kops::gemm
