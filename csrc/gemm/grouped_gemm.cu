// =============================================================================
// kernel-ops | csrc/gemm/grouped_gemm.cu
// B5 row 9: grouped GEMV for decode MoE on CUDA cores.
//   grouped_gemv_kernel   one CTA column per (expert, N tile): the expert's weight rows are
//                         streamed once for every row routed to it; an empty expert reads nothing
// Host entries: grouped_gemm (kernel-ops layouts kContiguous / kMasked / kIndexed) and
// grouped_gemm_workspace_requirements (no scratch on this path).
//
// Contract: csrc/gemm/grouped_gemm_kernels.cuh + csrc/gemm/ops.h.
//
// Scope of this body: bf16 / fp16 weights (recipes kBF16 / kFP16) and OCP e4m3 weights
// (kFP8Tensorwise: per-expert alpha b_scale[G]; kFP8Rowwise: per-channel b_scale[G, N]) against
// bf16 / fp16 activations; the output has the activation dtype.  Epilogue: Bias ([G, N], stride
// bias_group_stride), Act and RoutingWeight (applied in that order, once).  Everything else
// (blockwise, MX/NV FP4, int4, e4m3 activations, scatter/atomic finalize, GLU) returns
// kNotImplemented rather than approximating.
//
// Determinism.  A row's result is the same lane-strided FMA chain + xor-butterfly whatever the
// number of rows the expert received or the grid shape: no atomics, no cross-CTA reduction.
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
namespace {

constexpr int kMChunk = 4;   // rows of one expert multiplied per weight pass

template <GemmRecipe kRecipe>
KOPS_HOST_DEVICE constexpr bool is_fp8_recipe() {
  return kRecipe == GemmRecipe::kFP8Tensorwise || kRecipe == GemmRecipe::kFP8Rowwise;
}

/// Weight element for the recipe: e4m3 bytes for FP8, the activation type otherwise.
template <typename a_t, GemmRecipe kRecipe>
using weight_t = typename std::conditional<is_fp8_recipe<kRecipe>(), uint8_t, a_t>::type;

__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

}  // namespace

template <typename a_t, GemmRecipe kRecipe, GroupedLayout kLayout, int ROWS_PER_WARP, int WARPS>
__global__ void grouped_gemv_kernel(const __grid_constant__ GroupedGemmParams p) {
  constexpr bool kFp8 = is_fp8_recipe<kRecipe>();
  constexpr int R = ROWS_PER_WARP;
  constexpr int VEC = kFp8 ? 16 : 8;
  using w_t = weight_t<a_t, kRecipe>;
  const int g = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int64_t n0 = (static_cast<int64_t>(blockIdx.x) * WARPS + (threadIdx.x >> 5)) * R;

  // Rows routed to this expert.
  int64_t row0 = 0;     // first row in the packed row space (D rows / routing weights)
  int m_g = 0;
  if constexpr (kLayout == GroupedLayout::kMasked) {
    row0 = static_cast<int64_t>(g) * p.max_m;
    m_g = min(p.masked_m[g], p.max_m);
  } else {
    row0 = p.expert_offsets[g];
    m_g = static_cast<int>(p.expert_offsets[g + 1] - row0);
  }
  if (m_g <= 0 || n0 >= p.n) return;          // an empty expert never touches its weights

  const int k = p.k;
  const w_t* brow[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    brow[r] = static_cast<const w_t*>(p.b) + g * p.b_group_stride + imin(n0 + r, p.n - 1) * p.ldb;
  }
  float alpha = p.epi.alpha_ptr != nullptr ? p.epi.alpha_ptr[g] : p.epi.alpha;
  if constexpr (kRecipe == GemmRecipe::kFP8Tensorwise) {
    if (p.b_scale != nullptr) alpha *= static_cast<const float*>(p.b_scale)[g];
  }

  for (int mc = 0; mc < m_g; mc += kMChunk) {
    const a_t* arow[kMChunk];
    #pragma unroll
    for (int i = 0; i < kMChunk; ++i) {
      const int64_t ri = row0 + mc + (mc + i < m_g ? i : m_g - 1 - mc);
      int64_t src = ri;
      if constexpr (kLayout == GroupedLayout::kIndexed) src = p.gather_idx[ri];
      arow[i] = static_cast<const a_t*>(p.a) + src * p.lda;
    }
    float acc[kMChunk][R];
    #pragma unroll
    for (int i = 0; i < kMChunk; ++i) {
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
      #pragma unroll
      for (int i = 0; i < kMChunk; ++i) {
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
          #pragma unroll
          for (int j = 0; j < VEC; ++j) acc[i][r] = fmaf(af[j], wf[r][j], acc[i][r]);
        }
      }
    }
    #pragma unroll
    for (int i = 0; i < kMChunk; ++i) {
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        float v = acc[i][r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        const int64_t n = n0 + r;
        if (lane == i * R + r && mc + i < m_g && n < p.n) {
          const int64_t row = row0 + mc + i;
          v *= alpha;
          if constexpr (kRecipe == GemmRecipe::kFP8Rowwise) {
            v *= static_cast<const float*>(p.b_scale)[g * p.b_scale_group_stride + n];
          }
          if (p.epi.ops & kEpiBias) v += dev::load_as_float(p.epi.bias, p.epi.bias_dtype, g * p.epi.bias_group_stride + n);
          if (p.epi.ops & kEpiAct) v = dev::act_runtime(p.epi.act, v, p.epi.glu);
          if (p.epi.ops & kEpiRoutingWeight) v *= p.epi.routing_weight[row];
          dev::from_f(static_cast<a_t*>(p.d)[row * p.ldd + n], v);
        }
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

template <typename a_t, GemmRecipe kR, GroupedLayout kL>
Status launch_grouped(const GroupedGemmParams& p, cudaStream_t s) {
  constexpr int R = 2;
  constexpr int kWarps = 8;
  const dim3 grid(static_cast<unsigned>((p.n + kWarps * R - 1) / (kWarps * R)), static_cast<unsigned>(p.num_groups));
  grouped_gemv_kernel<a_t, kR, kL, R, kWarps><<<grid, kWarps * 32, 0, s>>>(p);
  return launch_error("grouped_gemv_kernel");
}

template <typename a_t, GemmRecipe kR>
Status launch_layout(const GroupedGemmParams& p, cudaStream_t s) {
  switch (p.layout) {
    case GroupedLayout::kMasked: return launch_grouped<a_t, kR, GroupedLayout::kMasked>(p, s);
    case GroupedLayout::kIndexed: return launch_grouped<a_t, kR, GroupedLayout::kIndexed>(p, s);
    default: return launch_grouped<a_t, kR, GroupedLayout::kContiguous>(p, s);
  }
}

template <typename a_t>
Status launch_recipe(const GroupedGemmParams& p, cudaStream_t s) {
  switch (p.recipe) {
    case GemmRecipe::kFP8Tensorwise: return launch_layout<a_t, GemmRecipe::kFP8Tensorwise>(p, s);
    case GemmRecipe::kFP8Rowwise: return launch_layout<a_t, GemmRecipe::kFP8Rowwise>(p, s);
    case GemmRecipe::kFP16: return launch_layout<a_t, GemmRecipe::kFP16>(p, s);
    default: return launch_layout<a_t, GemmRecipe::kBF16>(p, s);
  }
}

constexpr uint32_t kGroupedEpiAllowed = kEpiBias | kEpiAct | kEpiRoutingWeight;

}  // namespace

Status grouped_gemm_workspace_requirements(const GroupedGemmParams& params, const GemmTactic* tactic,
                                           const LaunchContext& ctx, WorkspaceRequirements* out) {
  (void)params;
  (void)tactic;
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "grouped_gemm: CUDA backend only");
  if (out == nullptr) return detail::set_error(Status::kInvalidArgument, "grouped_gemm_workspace_requirements: null output");
  *out = WorkspaceRequirements{};   // the CUDA-core path keeps no scratch, task map or tensor map
  return Status::kSuccess;
}

Status grouped_gemm(const GroupedGemmParams& p, const GemmTactic* tactic, GemmBackend backend,
                    const Workspace& ws, const LaunchContext& ctx) {
  (void)tactic;
  (void)ws;
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "grouped_gemm: CUDA backend only");
  if (backend != GemmBackend::kNative && backend != GemmBackend::kAuto) {
    return detail::set_error(Status::kUnsupportedBackend, "grouped_gemm: only the native provider exists");
  }
  const bool fp8 = p.recipe == GemmRecipe::kFP8Tensorwise || p.recipe == GemmRecipe::kFP8Rowwise;
  if (!fp8 && p.recipe != GemmRecipe::kBF16 && p.recipe != GemmRecipe::kFP16) {
    return detail::set_error(Status::kNotImplemented, "grouped_gemm: recipe %d has no native body yet (B7 / B11 / B13)", static_cast<int>(p.recipe));
  }
  if (p.layout != GroupedLayout::kContiguous && p.layout != GroupedLayout::kMasked && p.layout != GroupedLayout::kIndexed) {
    return detail::set_error(Status::kNotImplemented, "grouped_gemm: layout %d has no native body yet", static_cast<int>(p.layout));
  }
  if (p.num_groups < 1 || p.n < 1 || p.k < 1) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: empty problem");
  if (p.a == nullptr || p.b == nullptr || p.d == nullptr) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: null operand");
  if (p.layout == GroupedLayout::kMasked ? (p.masked_m == nullptr || p.max_m < 1)
                                         : (p.expert_offsets == nullptr || (p.layout == GroupedLayout::kIndexed && p.gather_idx == nullptr))) {
    return detail::set_error(Status::kInvalidArgument, "grouped_gemm: row metadata missing for the selected layout");
  }
  if (p.b_packing != WeightPacking::kNone) return detail::set_error(Status::kUnsupportedLayout, "grouped_gemm: plain weights only");
  if (p.a_scale != nullptr || p.a_global_scale != nullptr || p.b_zeros != nullptr || p.b_global_scale != nullptr || p.swap_ab) {
    return detail::set_error(Status::kNotImplemented, "grouped_gemm: scaled activations, zeros, global scales and swap_ab are not implemented");
  }
  if (p.recipe == GemmRecipe::kFP8Rowwise && p.b_scale == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "grouped_gemm: kFP8Rowwise needs b_scale [G, N]");
  }
  if ((p.epi.ops & ~kGroupedEpiAllowed) != 0u) {
    return detail::set_error(Status::kNotImplemented, "grouped_gemm: epilogue ops 0x%x are not implemented", p.epi.ops & ~kGroupedEpiAllowed);
  }
  if (((p.epi.ops & kEpiBias) != 0u && p.epi.bias == nullptr) || ((p.epi.ops & kEpiRoutingWeight) != 0u && p.epi.routing_weight == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "grouped_gemm: epilogue operand missing");
  }
  const int vec = fp8 ? 16 : 8;
  if (p.k % vec != 0) return detail::set_error(Status::kUnsupportedShape, "grouped_gemm: K must be a multiple of %d", vec);
  if (p.ldb < p.k || p.lda < p.k || p.ldd < p.n) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: leading dimensions smaller than the extents");
  const int wb = fp8 ? 1 : 2;
  if (!aligned(p.a, 16) || (p.lda * 2) % 16 != 0 || !aligned(p.b, 16) || (p.ldb * wb) % 16 != 0 || (p.b_group_stride * wb) % 16 != 0) {
    return detail::set_error(Status::kMisaligned, "grouped_gemm: activations, weights, ldb and the expert stride must be 16 B aligned");
  }
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  return p.recipe == GemmRecipe::kFP16 ? launch_recipe<__half>(p, s) : launch_recipe<__nv_bfloat16>(p, s);
}

}  // namespace kops::gemm
