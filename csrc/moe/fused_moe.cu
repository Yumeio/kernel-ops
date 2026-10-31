// =============================================================================
// kernel-ops | csrc/moe/fused_moe.cu
// B5 row 9: decode MoE GEMV on CUDA cores (opens the `moe` family's first .cu).
//   moe_gemv_kernel   one CTA per (block of sorted slots, N tile): the block's expert weight rows are
//                     streamed once for every slot of the block
// Host entry: moe::moe_gemv.
//
// Contract: csrc/moe/fused_moe_kernels.cuh + csrc/moe/ops.h.
//
// Layout.  `sorted_token_ids` / `expert_ids` / `num_tokens_post_pad` come from
// moe::align_block_size called with block_size == SLOTS_PER_CTA, so one block of
// SLOTS_PER_CTA sorted positions belongs to a single expert.  A slot id s >= num_valid_slots is
// padding.  The activation row is s (a_rows_are_slots, stage W2) or s / top_k (stage W13); the
// output row is s.  An expert id < 0 (not on this rank) leaves the block's rows untouched.
//
// Weights [E, N, K]: bf16 / fp16 (w_quant.granularity kNone) or OCP e4m3 with fp32 scales per
// channel ([E, N]), per expert ([E]) or 128x128 block ([E, N/128, K/128]).  Activations and output
// are bf16 / fp16.  Epilogue: bias [E, N] in the output dtype and mul_routed_weight; a fused
// activation (epi.ops != 0) and the int4 / FP4 weight formats return kNotImplemented.
// Determinism: same lane-strided FMA chain + xor-butterfly for every row, no atomics.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

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
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/moe/permute_kernels.cuh"
#include "csrc/moe/fused_moe_kernels.cuh"
#include "csrc/moe/ops.h"

namespace kops::moe {
namespace {

__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

}  // namespace

template <typename a_t, typename w_t, typename c_t, int SLOTS_PER_CTA, int ROWS_PER_WARP, int WARPS>
__global__ void moe_gemv_kernel(const __grid_constant__ FusedMoEGemmParams p) {
  constexpr bool kFp8 = std::is_same<w_t, uint8_t>::value;
  constexpr int R = ROWS_PER_WARP;
  constexpr int S = SLOTS_PER_CTA;
  constexpr int VEC = kFp8 ? 16 : 8;
  const int block = blockIdx.y;
  if (static_cast<int64_t>(block) * S >= *p.num_tokens_post_pad) return;
  const int expert = p.expert_ids[block];
  if (expert < 0) return;
  const int lane = threadIdx.x & 31;
  const int64_t n0 = (static_cast<int64_t>(blockIdx.x) * WARPS + (threadIdx.x >> 5)) * R;
  if (n0 >= p.n) return;
  const int k = p.k;

  int64_t slot[S];
  bool any = false;
  const a_t* arow[S];
  #pragma unroll
  for (int i = 0; i < S; ++i) {
    slot[i] = p.sorted_token_ids[static_cast<int64_t>(block) * S + i];
    const bool valid = slot[i] < p.num_valid_slots;
    any = any || valid;
    const int64_t src = valid ? (p.a_rows_are_slots ? slot[i] : slot[i] / p.top_k) : 0;
    arow[i] = static_cast<const a_t*>(p.a) + src * p.lda;
  }
  if (!any) return;

  const w_t* brow[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    brow[r] = static_cast<const w_t*>(p.w) + expert * p.w_expert_stride + imin(n0 + r, p.n - 1) * static_cast<int64_t>(k);
  }
  [[maybe_unused]] const float* sc = static_cast<const float*>(p.w_scale);
  const ScaleGranularity gran = p.w_quant.granularity;

  float acc[S][R];
  #pragma unroll
  for (int i = 0; i < S; ++i) {
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[i][r] = 0.0f;
  }
  #pragma unroll 2
  for (int kc = lane * VEC; kc < k; kc += 32 * VEC) {
    float wf[R][VEC];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint4 raw = dev::ld_stream16(brow[r] + kc);
      if constexpr (kFp8) {
        const uint32_t w4[4] = {raw.x, raw.y, raw.z, raw.w};
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          float t[4];
          dev::e4m3x4_decode(w4[q], t);
          wf[r][4 * q] = t[0]; wf[r][4 * q + 1] = t[1]; wf[r][4 * q + 2] = t[2]; wf[r][4 * q + 3] = t[3];
        }
      } else {
        dev::unpack8<a_t>(raw, wf[r]);
      }
    }
    float sb[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) sb[r] = 1.0f;
    if constexpr (kFp8) {
      if (gran == ScaleGranularity::kPerBlock2D) {
        const int64_t kblocks = k >> 7;
        const int64_t stride = p.w_scale_expert_stride != 0 ? p.w_scale_expert_stride : ((p.n + 127) >> 7) * kblocks;
        #pragma unroll
        for (int r = 0; r < R; ++r) sb[r] = sc[expert * stride + (imin(n0 + r, p.n - 1) >> 7) * kblocks + (kc >> 7)];
      }
    }
    #pragma unroll
    for (int i = 0; i < S; ++i) {
      float af[VEC];
      if constexpr (kFp8) {
        float lo[8], hi[8];
        dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(arow[i] + kc), lo);
        dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(arow[i] + kc + 8), hi);
        #pragma unroll
        for (int j = 0; j < 8; ++j) { af[j] = lo[j]; af[8 + j] = hi[j]; }
      } else {
        dev::unpack8<a_t>(*reinterpret_cast<const uint4*>(arow[i] + kc), af);
      }
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        if (kFp8 && gran == ScaleGranularity::kPerBlock2D) {
          float part = 0.0f;
          #pragma unroll
          for (int j = 0; j < VEC; ++j) part = fmaf(af[j], wf[r][j], part);
          acc[i][r] = fmaf(sb[r], part, acc[i][r]);
        } else {
          #pragma unroll
          for (int j = 0; j < VEC; ++j) acc[i][r] = fmaf(af[j], wf[r][j], acc[i][r]);
        }
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < S; ++i) {
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float v = acc[i][r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
      const int64_t n = n0 + r;
      if (lane == i * R + r && slot[i] < p.num_valid_slots && n < p.n) {
        if constexpr (kFp8) {
          if (gran == ScaleGranularity::kPerChannel) v *= sc[expert * p.w_scale_expert_stride + n];
          else if (gran == ScaleGranularity::kPerTensor) v *= sc[expert];
        }
        if (p.bias != nullptr) v += dev::to_f(static_cast<const c_t*>(p.bias)[expert * p.n + n]);
        if (p.mul_routed_weight) v *= p.topk_weights[slot[i]];
        dev::from_f(static_cast<c_t*>(p.c)[slot[i] * p.ldc + n], v);
      }
    }
  }
}

namespace {

Status launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) return Status::kSuccess;
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

template <typename a_t, typename w_t, int S>
Status launch_moe_tile(const FusedMoEGemmParams& p, int32_t max_blocks, cudaStream_t s) {
  constexpr int R = 2;
  constexpr int kWarps = 8;
  const dim3 grid(static_cast<unsigned>((p.n + kWarps * R - 1) / (kWarps * R)), static_cast<unsigned>(max_blocks));
  moe_gemv_kernel<a_t, w_t, a_t, S, R, kWarps><<<grid, kWarps * 32, 0, s>>>(p);
  return launch_error("moe_gemv_kernel");
}

template <typename a_t, typename w_t>
Status launch_moe(const FusedMoEGemmParams& p, int32_t max_blocks, int32_t slots, cudaStream_t s) {
  switch (slots) {
    case 1: return launch_moe_tile<a_t, w_t, 1>(p, max_blocks, s);
    case 2: return launch_moe_tile<a_t, w_t, 2>(p, max_blocks, s);
    case 4: return launch_moe_tile<a_t, w_t, 4>(p, max_blocks, s);
    default: return launch_moe_tile<a_t, w_t, 8>(p, max_blocks, s);
  }
}

}  // namespace

Status moe_gemv(const FusedMoEGemmParams& p, int32_t num_local_experts, int32_t slots_per_cta, DType a_dtype,
                DType w_dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "moe_gemv: CUDA backend only");
  if (a_dtype != DType::kBFloat16 && a_dtype != DType::kFloat16) {
    return detail::set_error(Status::kUnsupportedDtype, "moe_gemv: activations and output must be bf16 or fp16");
  }
  const bool fp8 = w_dtype == DType::kFloat8E4M3;
  if (!fp8 && w_dtype != a_dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "moe_gemv: weights must be e4m3 or share the activation dtype (int4 / FP4 weights are not implemented)");
  }
  if (slots_per_cta != 1 && slots_per_cta != 2 && slots_per_cta != 4 && slots_per_cta != 8) {
    return detail::set_error(Status::kUnsupportedShape, "moe_gemv: slots_per_cta must be 1, 2, 4 or 8 (the align_block_size block size)");
  }
  if (p.epi.ops != 0u || p.a_scale != nullptr || p.w_zeros != nullptr) {
    return detail::set_error(Status::kNotImplemented, "moe_gemv: fused epilogue, activation scales and zero points are not implemented");
  }
  if (num_local_experts < 1 || p.n < 1 || p.k < 1 || p.top_k < 1 || p.num_valid_slots < 0) {
    return detail::set_error(Status::kInvalidArgument, "moe_gemv: empty or inconsistent problem");
  }
  if (p.a == nullptr || p.w == nullptr || p.c == nullptr || p.sorted_token_ids == nullptr || p.expert_ids == nullptr ||
      p.num_tokens_post_pad == nullptr || (p.mul_routed_weight && p.topk_weights == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "moe_gemv: null operand");
  }
  const ScaleGranularity g = p.w_quant.granularity;
  if (fp8) {
    const bool ok = g == ScaleGranularity::kPerChannel || g == ScaleGranularity::kPerTensor || g == ScaleGranularity::kPerBlock2D;
    if (!ok || p.w_scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "moe_gemv: e4m3 weights need fp32 per-channel, per-expert or 128x128 block scales");
    }
    if (g == ScaleGranularity::kPerBlock2D && (p.w_quant.group_m != 128 || p.w_quant.group_k != 128 || p.k % 128 != 0)) {
      return detail::set_error(Status::kUnsupportedShape, "moe_gemv: block scales must be 128x128 and K a multiple of 128");
    }
    if (g == ScaleGranularity::kPerChannel && p.w_scale_expert_stride < p.n) {
      return detail::set_error(Status::kInvalidArgument, "moe_gemv: per-channel scales need w_scale_expert_stride >= N");
    }
  } else if (g != ScaleGranularity::kNone) {
    return detail::set_error(Status::kInvalidArgument, "moe_gemv: 16-bit weights take no scales");
  }
  const int vec = fp8 ? 16 : 8;
  const int wb = fp8 ? 1 : 2;
  if (p.k % vec != 0) return detail::set_error(Status::kUnsupportedShape, "moe_gemv: K must be a multiple of %d", vec);
  if (p.lda < p.k || p.ldc < p.n) return detail::set_error(Status::kInvalidArgument, "moe_gemv: leading dimensions smaller than the extents");
  if (!aligned(p.a, 16) || (p.lda * 2) % 16 != 0 || !aligned(p.w, 16) || (static_cast<int64_t>(p.k) * wb) % 16 != 0 ||
      (p.w_expert_stride * wb) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "moe_gemv: activations, weights and the expert stride must be 16 B aligned");
  }
  // Upper bound on the number of blocks: every expert wastes at most slots_per_cta - 1 padding slots.
  const int64_t max_blocks = (p.num_valid_slots + slots_per_cta - 1) / slots_per_cta + num_local_experts;
  if (max_blocks > 65535) return detail::set_error(Status::kCapacityExceeded, "moe_gemv: more than 65535 slot blocks");
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int32_t mb = static_cast<int32_t>(max_blocks);
  if (fp8) {
    return a_dtype == DType::kBFloat16 ? launch_moe<__nv_bfloat16, uint8_t>(p, mb, slots_per_cta, s)
                                       : launch_moe<__half, uint8_t>(p, mb, slots_per_cta, s);
  }
  return a_dtype == DType::kBFloat16 ? launch_moe<__nv_bfloat16, __nv_bfloat16>(p, mb, slots_per_cta, s)
                                     : launch_moe<__half, __half>(p, mb, slots_per_cta, s);
}

}  // namespace kops::moe
