// =============================================================================
// kernel-ops | csrc/moe/fused_moe.cu
// B5 row 9: decode MoE GEMV on CUDA cores (opens the `moe` family's first .cu).
//   moe_gemv_kernel   one CTA per (block of sorted slots, N tile): the block's expert weight rows are
//                     streamed once for every slot of the block
// Host entry: moe::moe_gemv.
//
// B7 row 8: fused_moe_gemm_kernel, the expert GEMM on tensor cores = gemm::gemm_mma_sync_kernel with the
// kSortedBlocks row map (Triton fused_moe semantics over the B6 sorted blocks): a tile of a block takes that
// block's expert weights, gathers A rows by slot id, writes C rows by slot id. W13 + GLU (interleaved
// weights) and W2 + routed weight are epilogue options. Host entry: moe::moe_gemm.
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

template <typename scalar_t, ActQuantMode kQuant, int VEC_SIZE>
__global__ void latent_moe_combine_norm_kernel(
    void* __restrict__ out, const ActQuantOut quant, const scalar_t* __restrict__ expert_out,
    const int32_t* __restrict__ slot_to_permuted_row, const float* __restrict__ topk_weights,
    const scalar_t* __restrict__ norm_weight, const float routed_scale, const float epsilon,
    const int num_tokens, const int topk, const int latent_dim) {
  static_assert(kQuant == ActQuantMode::kNone, "latent_moe_combine_norm supports the unquantized tail");
  (void)quant;
  extern __shared__ float smem[];
  float* z = smem;                    // [latent_dim]
  float* red = smem + latent_dim;     // [8]
  const int t = blockIdx.x;
  if (t >= num_tokens) return;
  const int64_t base = static_cast<int64_t>(t) * topk;
  for (int col = threadIdx.x; col < latent_dim; col += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < topk; ++k) {
      const int32_t row = slot_to_permuted_row[base + k];
      if (row < 0) continue;
      const float w = topk_weights != nullptr ? topk_weights[base + k] : 1.0f;
      acc = fmaf(w, dev::to_f(expert_out[static_cast<int64_t>(row) * latent_dim + col]), acc);
    }
    z[col] = acc * routed_scale;
  }
  __syncthreads();
  float ss = 0.0f;
  for (int col = threadIdx.x; col < latent_dim; col += blockDim.x) ss += z[col] * z[col];
  const float mean = block_reduce_sum<float, 8>(ss, red) / static_cast<float>(latent_dim);
  const float inv = rsqrtf(mean + epsilon);
  auto* dst = static_cast<scalar_t*>(out);
  for (int col = threadIdx.x; col < latent_dim; col += blockDim.x) {
    const float v = z[col] * inv * dev::to_f(norm_weight[col]);
    dev::from_f(dst[static_cast<int64_t>(t) * latent_dim + col], v);
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

Status moe_gemm(const FusedMoEGemmParams& p, int32_t num_local_experts, int32_t block_size, DType a_dtype,
                DType w_dtype, DType c_dtype, const LaunchContext& ctx) {
  using namespace kops::gemm;
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "moe_gemm: CUDA backend only");
  if (block_size != 16 && block_size != 32 && block_size != 64 && block_size != 128 && block_size != 256) {
    return detail::set_error(Status::kUnsupportedShape, "moe_gemm: block_size (align_block_size) must be 16, 32, 64, 128 or 256");
  }
  if (num_local_experts < 1 || p.n < 1 || p.k < 1 || p.top_k < 1 || p.num_valid_slots < 0) {
    return detail::set_error(Status::kInvalidArgument, "moe_gemm: empty or inconsistent problem");
  }
  if (p.a == nullptr || p.w == nullptr || p.c == nullptr || p.sorted_token_ids == nullptr || p.expert_ids == nullptr ||
      p.num_tokens_post_pad == nullptr || (p.mul_routed_weight && p.topk_weights == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "moe_gemm: null operand");
  }
  if (p.w_zeros != nullptr) return detail::set_error(Status::kNotImplemented, "moe_gemm: zero points (int4 experts run through marlin_gemm in MoE mode)");
  if (c_dtype != DType::kBFloat16 && c_dtype != DType::kFloat16 && c_dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "moe_gemm: output must be bf16, fp16 or fp32");
  }
  MmaGemmParams mp{};
  ScaledGemmArgs& a = mp.args;
  const ScaleGranularity wg = p.w_quant.granularity;
  if ((a_dtype == DType::kBFloat16 || a_dtype == DType::kFloat16) && w_dtype == a_dtype) {
    if (wg != ScaleGranularity::kNone || p.a_scale != nullptr) return detail::set_error(Status::kInvalidArgument, "moe_gemm: 16-bit experts take no scales");
    a.recipe = a_dtype == DType::kBFloat16 ? GemmRecipe::kBF16 : GemmRecipe::kFP16;
  } else if (a_dtype == DType::kFloat8E4M3 && w_dtype == DType::kFloat8E4M3) {
    if (p.w_scale == nullptr) return detail::set_error(Status::kInvalidArgument, "moe_gemm: e4m3 experts need w_scale");
    if (wg == ScaleGranularity::kPerTensor) {
      a.recipe = GemmRecipe::kFP8Tensorwise;
      a.b.granularity = ScaleGranularity::kPerTensor;
      a.b.scale_batch_stride = p.w_scale_expert_stride > 0 ? p.w_scale_expert_stride : 1;
    } else if (wg == ScaleGranularity::kPerChannel) {
      a.recipe = GemmRecipe::kFP8Rowwise;
      a.b.granularity = ScaleGranularity::kPerChannel;
      if (p.w_scale_expert_stride < p.n) return detail::set_error(Status::kInvalidArgument, "moe_gemm: per-channel scales need w_scale_expert_stride >= N");
      a.b.scale_batch_stride = p.w_scale_expert_stride;
    } else if (wg == ScaleGranularity::kPerBlock2D) {
      if (p.w_quant.group_m != 128 || p.w_quant.group_k != 128) return detail::set_error(Status::kUnsupportedShape, "moe_gemm: block scales must be 128x128");
      if (p.a_scale == nullptr) return detail::set_error(Status::kInvalidArgument, "moe_gemm: blockwise experts need 1x128 activation scales");
      a.recipe = p.w_quant.scale_dtype == DType::kFloat8E8M0 ? GemmRecipe::kFP8BlockwiseUE8M0 : GemmRecipe::kFP8Blockwise;
      if (p.k % 128 != 0) return detail::set_error(Status::kUnsupportedShape, "moe_gemm: blockwise experts need K %% 128 == 0");
      a.b.granularity = ScaleGranularity::kPerBlock2D;
      // [E, ceil(N / 128), K / 128] unless the caller gives the expert stride (same default as moe_gemv)
      a.b.scale_batch_stride = p.w_scale_expert_stride != 0 ? p.w_scale_expert_stride : static_cast<int64_t>((p.n + 127) / 128) * (p.k / 128);
      a.a.granularity = ScaleGranularity::kPerGroup;
      a.a.scale_layout = p.a_quant.scale_layout;
      a.a.scale_dtype = p.a_quant.scale_dtype == DType::kFloat8E8M0 ? DType::kFloat8E8M0 : DType::kFloat32;
      if (a.a.scale_layout == ScaleLayout::kColMajorTma) {
        // [K / 128, round_up(rows of A, 4)]: rows of A are the tokens (W13) or the slots (W2), not the sorted row space
        const int64_t a_rows = p.a_rows_are_slots ? p.num_valid_slots : (p.num_valid_slots + p.top_k - 1) / p.top_k;
        a.a.scale_ld = (std::max<int64_t>(a_rows, 1) + 3) / 4 * 4;
      }
    } else {
      return detail::set_error(Status::kInvalidArgument, "moe_gemm: e4m3 experts need per-expert, per-channel or 128x128 scales");
    }
    a.b.scale = p.w_scale;
    a.b.scale_dtype = wg == ScaleGranularity::kPerBlock2D && p.w_quant.scale_dtype == DType::kFloat8E8M0 ? DType::kFloat8E8M0 : DType::kFloat32;
  } else if (a_dtype == DType::kInt8 && w_dtype == DType::kInt8) {
    if (wg != ScaleGranularity::kPerChannel || p.w_scale == nullptr || p.w_scale_expert_stride < p.n) {
      return detail::set_error(Status::kInvalidArgument, "moe_gemm: int8 experts need per-channel w_scale [E, N]");
    }
    a.recipe = GemmRecipe::kINT8Rowwise;
    a.b.scale = p.w_scale;
    a.b.granularity = ScaleGranularity::kPerChannel;
    a.b.scale_batch_stride = p.w_scale_expert_stride;
  } else {
    return detail::set_error(Status::kUnsupportedDtype, "moe_gemm: weights must match bf16 / fp16 activations, or be e4m3 x e4m3 / int8 x int8 (int4 / FP4: marlin_gemm)");
  }
  if (p.a_scale != nullptr && a.a.granularity != ScaleGranularity::kPerGroup) {
    a.a.scale = p.a_scale;
    a.a.granularity = p.a_quant.granularity == ScaleGranularity::kPerTensor ? ScaleGranularity::kPerTensor : ScaleGranularity::kPerToken;
    a.a.scale_dtype = DType::kFloat32;
  } else if (p.a_scale != nullptr) {
    a.a.scale = p.a_scale;
  }
  a.a.data = p.a;
  a.a.ld = p.lda;
  a.a.dtype = a_dtype;
  a.b.data = p.w;
  a.b.ld = p.k;
  a.b.batch_stride = p.w_expert_stride;
  a.b.dtype = w_dtype;
  a.d = p.c;
  a.ldd = p.ldc;
  a.d_dtype = c_dtype;
  a.epi = p.epi;
  if (c_dtype == DType::kFloat32) a.epi.ops |= kEpiOutFp32;
  if (p.mul_routed_weight) {
    a.epi.ops |= kEpiRoutingWeight;
    a.epi.routing_weight = p.topk_weights;
  }
  if (p.bias != nullptr) {
    a.epi.ops |= kEpiBias;
    a.epi.bias = p.bias;
    a.epi.bias_dtype = c_dtype;
    a.epi.bias_group_stride = p.n;
  }
  // row space: the capacity of sorted_token_ids (every expert pads at most block_size - 1 slots)
  const int64_t cap = ((p.num_valid_slots + static_cast<int64_t>(num_local_experts) * (block_size - 1)) + block_size - 1) / block_size * block_size;
  a.m = cap > 0 ? cap : block_size;
  a.n = p.n;
  a.k = p.k;
  MmaRowMap& rm = mp.rows;
  rm.schedule = MmaSchedule::kSortedBlocks;
  rm.block_group = p.expert_ids;
  rm.num_rows = p.num_tokens_post_pad;
  rm.block_rows = block_size;
  rm.row_ids = p.sorted_token_ids;
  rm.id_limit = p.num_valid_slots;
  rm.a_rows_by_id = true;
  rm.a_row_div = p.a_rows_are_slots ? 1 : p.top_k;
  rm.out_rows_by_id = true;
  return launch_gemm_mma_sync(mp, nullptr, Workspace{}, ctx);
}

Status latent_combine_norm(const Tensor& out, const Tensor& expert_out,
                           const Tensor& slot_to_permuted_row, const Tensor& topk_weights,
                           const Tensor& norm_weight, float routed_scale, float epsilon,
                           int32_t topk, const ActQuantOut& quant, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "latent_combine_norm: CUDA backend only");
  }
  if (quant.mode != ActQuantMode::kNone) {
    return detail::set_error(Status::kNotImplemented,
                             "latent_combine_norm: quantized latent output is not implemented");
  }
  Status s = check_tensor(out, out.dtype, 2, 2, "latent_combine_norm.out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(expert_out, out.dtype, 2, 2, "latent_combine_norm.expert_out");
  if (s != Status::kSuccess) return s;
  s = check_tensor(slot_to_permuted_row, DType::kInt32, 1, 4,
                   "latent_combine_norm.slot_to_permuted_row");
  if (s != Status::kSuccess) return s;
  s = check_tensor(topk_weights, DType::kFloat32, 1, 4, "latent_combine_norm.topk_weights");
  if (s != Status::kSuccess) return s;
  s = check_tensor(norm_weight, out.dtype, 1, 2, "latent_combine_norm.norm_weight");
  if (s != Status::kSuccess) return s;
  const int num_tokens = static_cast<int>(out.shape[0]);
  const int latent = static_cast<int>(out.shape[1]);
  if (latent < 1 || latent > 8192) {
    return detail::set_error(Status::kUnsupportedShape,
                             "latent_combine_norm: latent_dim %d not in [1, 8192]", latent);
  }
  if (expert_out.shape[1] != latent || norm_weight.shape[0] != latent) {
    return detail::set_error(Status::kInvalidArgument, "latent_combine_norm: latent dimension mismatch");
  }
  if (num_tokens <= 0) return Status::kSuccess;
  if (topk < 1 || slot_to_permuted_row.shape[0] != static_cast<int64_t>(num_tokens) * topk ||
      topk_weights.shape[0] != static_cast<int64_t>(num_tokens) * topk) {
    return detail::set_error(Status::kInvalidArgument,
                             "latent_combine_norm: slot/weight arrays must be [T, topk]");
  }
  if (epsilon < 0.0f) {
    return detail::set_error(Status::kInvalidArgument, "latent_combine_norm: negative epsilon");
  }
  const int smem = (latent + 8) * static_cast<int>(sizeof(float));
  const cudaStream_t s2 = reinterpret_cast<cudaStream_t>(ctx.stream);
  if (out.dtype == DType::kBFloat16) {
    latent_moe_combine_norm_kernel<__nv_bfloat16, ActQuantMode::kNone, 8>
        <<<num_tokens, 256, smem, s2>>>(out.data, quant,
                                        static_cast<const __nv_bfloat16*>(expert_out.data),
                                        static_cast<const int32_t*>(slot_to_permuted_row.data),
                                        static_cast<const float*>(topk_weights.data),
                                        static_cast<const __nv_bfloat16*>(norm_weight.data),
                                        routed_scale, epsilon, num_tokens, topk, latent);
  } else if (out.dtype == DType::kFloat16) {
    latent_moe_combine_norm_kernel<__half, ActQuantMode::kNone, 8>
        <<<num_tokens, 256, smem, s2>>>(out.data, quant,
                                        static_cast<const __half*>(expert_out.data),
                                        static_cast<const int32_t*>(slot_to_permuted_row.data),
                                        static_cast<const float*>(topk_weights.data),
                                        static_cast<const __half*>(norm_weight.data), routed_scale,
                                        epsilon, num_tokens, topk, latent);
  } else if (out.dtype == DType::kFloat32) {
    latent_moe_combine_norm_kernel<float, ActQuantMode::kNone, 4>
        <<<num_tokens, 256, smem, s2>>>(out.data, quant,
                                        static_cast<const float*>(expert_out.data),
                                        static_cast<const int32_t*>(slot_to_permuted_row.data),
                                        static_cast<const float*>(topk_weights.data),
                                        static_cast<const float*>(norm_weight.data), routed_scale,
                                        epsilon, num_tokens, topk, latent);
  } else {
    return detail::set_error(Status::kUnsupportedDtype, "latent_combine_norm: %s",
                             dtype_name(out.dtype));
  }
  return launch_error("latent_moe_combine_norm_kernel");
}

}  // namespace kops::moe
