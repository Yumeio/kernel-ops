// =============================================================================
// kernel-ops | csrc/gemm/grouped_gemm.cu
// B5 row 9: grouped GEMV for decode MoE on CUDA cores.
//   grouped_gemv_kernel   one CTA column per (expert, N tile): the expert's weight rows are
//                         streamed once for every row routed to it; an empty expert reads nothing
// Host entries: grouped_gemm (kernel-ops layouts kContiguous / kMasked / kIndexed) and
// grouped_gemm_workspace_requirements (no scratch on this path).
//
// B7 rows 7 and 9: tensor-core grouped GEMM and multi-LoRA on the shared mma.sync body (gemm_mma.cu).
//   grouped_gemm_task_map_kernel   one CTA scans the groups and writes (group, m tile, n tile) for every
//                                  group with rows (M tiles fastest); an empty group writes no task
//   grouped_gemm_mma_sync_kernel   = gemm_mma_sync_kernel with a grouped row map: kContiguous, kIndexed
//                                  (A rows gathered by gather_idx), kMasked, kContiguousRowIndex (m_indices);
//                                  recipes kBF16 / kFP16, kFP8Tensorwise / Rowwise / Blockwise (Sm89+), kINT8Rowwise
//   lora_shrink_kernel / lora_expand_kernel = gemm_mma_sync_kernel over seg_indptr segments with the adapter
//                                  pointer array (Punica SGMV semantics; expand reads the fp32 shrink buffer)
// Host entries: grouped_gemm (tensor-core path when the tactic names mma.sync, a task map is prepared, the
// activations are 8-bit, or expected_m > 16), grouped_gemm_prepare (GroupedProblem[] + task map for the
// TMA-class mainloops, task map only for mma.sync), lora_sgmv.
//
// B11 operand prep: grouped_gemm_prepare_kernel builds GroupedProblem[G] (rows, per-group pointers, alpha,
// first tile id) for the TMA / tcgen05 / SM120 warp-specialized mainloops; the tile list is the shared
// grouped_gemm_task_map_kernel.  The generic form is architecture-independent: no TMA descriptor patching
// (group_tma stays null) and no host sync or allocation.
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

// =============================================================================
// B7 row 7: task map
// =============================================================================

/// One CTA (any block size up to 1024): groups are scanned blockDim at a time; each chunk's tasks are written
/// by the whole block (binary search of the task index in the chunk's exclusive scan). Writes stop at
/// `capacity`; *num_tasks = min(total, capacity) and *overflow_flag = total > capacity.
__global__ void grouped_gemm_task_map_kernel(int32_t* __restrict__ task_map, int32_t* __restrict__ num_tasks,
                                             int32_t* __restrict__ overflow_flag, const int64_t* __restrict__ expert_offsets,
                                             const int32_t* __restrict__ masked_m, const int num_groups, const int tile_m,
                                             const int n_tiles, const int capacity) {
  __shared__ int64_t s_excl[1024];
  __shared__ int s_mt[1024];
  __shared__ int64_t s_total;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  int64_t base = 0;
  for (int g0 = 0; g0 < num_groups; g0 += nt) {
    const int g = g0 + tid;
    int64_t rows = 0;
    if (g < num_groups) rows = masked_m != nullptr ? max(masked_m[g], 0) : expert_offsets[g + 1] - expert_offsets[g];
    const int mt = static_cast<int>((rows + tile_m - 1) / tile_m);
    const int64_t cnt = static_cast<int64_t>(mt) * n_tiles;
    // block-wide inclusive scan (Hillis-Steele in shared memory; groups are few)
    s_excl[tid] = cnt;
    s_mt[tid] = mt;
    __syncthreads();
    for (int o = 1; o < nt; o <<= 1) {
      const int64_t v = tid >= o ? s_excl[tid - o] : 0;
      __syncthreads();
      s_excl[tid] += v;
      __syncthreads();
    }
    if (tid == nt - 1) s_total = s_excl[tid];
    __syncthreads();
    s_excl[tid] -= cnt;   // exclusive
    __syncthreads();
    const int64_t total = s_total;
    const int last = min(nt, num_groups - g0) - 1;
    for (int64_t i = tid; i < total; i += nt) {
      const int64_t idx = base + i;
      if (idx >= capacity) break;
      int lo = 0, hi = last;   // largest lane with s_excl <= i
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (s_excl[mid] <= i) lo = mid;
        else hi = mid - 1;
      }
      const int64_t local = i - s_excl[lo];
      const int mtl = s_mt[lo];
      task_map[3 * idx] = g0 + lo;
      task_map[3 * idx + 1] = static_cast<int32_t>(local % mtl);
      task_map[3 * idx + 2] = static_cast<int32_t>(local / mtl);
    }
    base += total;
    __syncthreads();
  }
  if (tid == 0) {
    *num_tasks = static_cast<int32_t>(base < capacity ? base : capacity);
    *overflow_flag = base > capacity ? 1 : 0;
  }
}

namespace {

// =============================================================================
// B11: generic device-side group preparation (GroupedProblem per group)
// =============================================================================
// GroupedGemmParams carries no dtype fields, so the recipe is the contract for the element sizes (the
// same mapping grouped_to_mma applies when it fills MmaGemmParams); the host admits only the recipes whose
// activation and weight element sizes are implied by the recipe (bf16 / fp16 / FP8 / INT8).  Layouts: kContiguous / kIndexed
// take the group extent from expert_offsets, kMasked from masked_m clamped to max_m, kPtrArray from
// problem_m with the per-group pointer arrays.  Groups with m == 0 get no tiles.

KOPS_HOST_DEVICE_INLINE bool recipe_is_8bit(GemmRecipe r) {
  return r == GemmRecipe::kFP8Tensorwise || r == GemmRecipe::kFP8Rowwise ||
         r == GemmRecipe::kFP8Blockwise || r == GemmRecipe::kFP8BlockwiseUE8M0 ||
         r == GemmRecipe::kMXFP8 || r == GemmRecipe::kINT8Tensorwise || r == GemmRecipe::kINT8Rowwise;
}

/// Bits of one activation, weight and output element.
KOPS_HOST_DEVICE_INLINE void recipe_element_bits(GemmRecipe r, uint32_t epi_ops, int* a_bits, int* b_bits, int* d_bits) {
  const int w = recipe_is_8bit(r) ? 8 : 16;
  *a_bits = w;
  *b_bits = w;
  *d_bits = (epi_ops & kEpiOutFp32) != 0u ? 32 : (w == 8 ? 16 : w);
}

/// Bytes of one B-scale element: UE8M0 for the block-scaled FP8 / MX recipes, fp32 otherwise.
KOPS_HOST_DEVICE_INLINE int scale_element_bytes(GemmRecipe r) {
  return (r == GemmRecipe::kFP8BlockwiseUE8M0 || r == GemmRecipe::kMXFP8) ? 1 : 4;
}

/// Expert stride of the B-scale tensor: a zero stride is read as one scale per expert (the
/// grouped_to_mma convention for the tensorwise recipes), so alpha and sfb always agree.
__device__ __forceinline__ int64_t b_scale_stride(const GroupedGemmParams& p) {
  return p.b_scale_group_stride > 0 ? p.b_scale_group_stride : 1;
}

__device__ __forceinline__ float group_alpha(const GroupedGemmParams& p, int g) {
  float alpha = p.epi.alpha_ptr != nullptr ? p.epi.alpha_ptr[g] : p.epi.alpha;
  if (p.recipe == GemmRecipe::kFP8Tensorwise || p.recipe == GemmRecipe::kINT8Tensorwise) {
    if (p.b_scale != nullptr) alpha *= static_cast<const float*>(p.b_scale)[g * b_scale_stride(p)];
  }
  return alpha;
}

}  // namespace

/// One CTA, blockDim up to 1024, groups scanned blockDim at a time (same shape as the task map scan):
/// problems[g] gets the group's shapes, pointers, leading dimensions, alpha and first tile id, and the
/// block-wide exclusive scan of ceil(m / tile_m) * n_tiles gives both tile_offset and the total that
/// *num_tasks / *overflow_flag report.  The tile list itself is written by
/// grouped_gemm_task_map_kernel (the host launches it right after this kernel, same stream, so the two
/// counts agree); tensor-map patching (tensormap.replace) is the Sm90a/Sm100a TMA path and is left to
/// the caller: group_tma / tma_templates stay untouched, and *nothing* here requires TMA.
__global__ void grouped_gemm_prepare_kernel(GroupedProblem* __restrict__ problems,
                                            KOPS_TMA_DESC* __restrict__ group_tma,
                                            const KOPS_TMA_DESC* __restrict__ tma_templates,
                                            int32_t* __restrict__ tile_task_map,
                                            int32_t* __restrict__ num_tasks,
                                            int32_t* __restrict__ overflow_flag,
                                            const GroupedGemmParams params, const int tile_m,
                                            const int tile_n, const int task_capacity) {
  (void)group_tma;
  (void)tma_templates;
  (void)tile_task_map;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  const int n_tiles = (params.n + tile_n - 1) / tile_n;
  int a_bits = 16, b_bits = 16, d_bits = 16;
  recipe_element_bits(params.recipe, params.epi.ops, &a_bits, &b_bits, &d_bits);
  const int sb_bytes = scale_element_bytes(params.recipe);
  const int64_t a_elem = a_bits / 8, b_elem = b_bits / 8, d_elem = d_bits / 8;   // 4-bit: row * ld / 2

  __shared__ int64_t s_excl[1024];
  __shared__ int64_t s_total;
  int64_t base = 0;
  for (int g0 = 0; g0 < params.num_groups; g0 += nt) {
    const int g = g0 + tid;
    int32_t m = 0;
    int64_t row0 = 0;
    if (g < params.num_groups) {
      switch (params.layout) {
        case GroupedLayout::kMasked: {
          row0 = static_cast<int64_t>(g) * params.max_m;
          const int32_t v = params.masked_m[g];
          m = v < 0 ? 0 : (v > params.max_m ? params.max_m : v);
          break;
        }
        case GroupedLayout::kPtrArray:
          m = params.problem_m[g] < 0 ? 0 : params.problem_m[g];
          break;
        default: {
          const int64_t b0 = params.expert_offsets[g];
          const int64_t b1 = params.expert_offsets[g + 1];
          row0 = b0;
          m = b1 > b0 ? static_cast<int32_t>(b1 - b0) : 0;
        }
      }
    }
    const int64_t cnt = g < params.num_groups ? static_cast<int64_t>((m + tile_m - 1) / tile_m) * n_tiles : 0;
    s_excl[tid] = cnt;
    __syncthreads();
    for (int o = 1; o < nt; o <<= 1) {
      const int64_t v = tid >= o ? s_excl[tid - o] : 0;
      __syncthreads();
      s_excl[tid] += v;
      __syncthreads();
    }
    if (tid == nt - 1) s_total = s_excl[tid];
    __syncthreads();
    s_excl[tid] -= cnt;
    const int64_t first_tile = base + s_excl[tid];
    __syncthreads();

    if (g < params.num_groups && problems != nullptr) {
      GroupedProblem pr;
      pr.m = m;
      pr.n = params.n;
      pr.k = params.k;
      pr.tile_offset = static_cast<int32_t>(first_tile);
      pr.lda = params.lda;
      pr.ldb = params.ldb;
      pr.ldd = params.ldd;
      pr.alpha = group_alpha(params, g);
      const auto* abase = static_cast<const uint8_t*>(params.a);
      const auto* bbase = static_cast<const uint8_t*>(params.b);
      auto* dbase = static_cast<uint8_t*>(params.d);
      switch (params.layout) {
        case GroupedLayout::kPtrArray:
          pr.a = params.a_ptrs[g];
          pr.b = params.b_ptrs[g];
          pr.d = params.d_ptrs[g];
          break;
        case GroupedLayout::kIndexed:
          pr.a = params.a;   // rows are gathered by gather_idx, the table is shared by every group
          pr.b = bbase + static_cast<int64_t>(g) * params.b_group_stride * b_elem;
          pr.d = dbase + static_cast<int64_t>(row0) * params.ldd * d_elem;
          break;
        default:   // kContiguous / kMasked: A and D rows are [row0, row0 + m) of the packed row space
          pr.a = abase + static_cast<int64_t>(row0) * params.lda * a_elem;
          pr.b = bbase + static_cast<int64_t>(g) * params.b_group_stride * b_elem;
          pr.d = dbase + static_cast<int64_t>(row0) * params.ldd * d_elem;
          break;
      }
      // Block-scale layouts (col-major TMA / 128x4 swizzled / packed UE8M0) already carry the group's
      // rows, so their base passes through; row-major per-token scales move with the group's rows.
      const bool row_major_token_scales = params.a_scale_layout == ScaleLayout::kRowMajor &&
                                          (params.recipe == GemmRecipe::kFP8Rowwise ||
                                           params.recipe == GemmRecipe::kINT8Rowwise);
      pr.sfa = params.a_scale != nullptr
                   ? (row_major_token_scales
                          ? static_cast<const void*>(static_cast<const uint8_t*>(params.a_scale) +
                                                     static_cast<int64_t>(row0) * scale_element_bytes(params.recipe))
                          : params.a_scale)
                   : nullptr;
      pr.sfb = params.b_scale != nullptr
                   ? static_cast<const void*>(static_cast<const uint8_t*>(params.b_scale) +
                                              static_cast<int64_t>(g) * b_scale_stride(params) * sb_bytes)
                   : nullptr;
      problems[g] = pr;
    }
    base += s_total;
    __syncthreads();
  }
  if (tid == 0) {
    if (num_tasks != nullptr) {
      *num_tasks = static_cast<int32_t>(base < task_capacity ? base : task_capacity);
    }
    if (overflow_flag != nullptr) *overflow_flag = base > task_capacity ? 1 : 0;
  }
}

namespace {

// ---- tensor-core grouped GEMM: GroupedGemmParams -> MmaGemmParams -------------------------------------

/// Header of the task map grouped_gemm_prepare writes: tile_task_map points at the entries and the four
/// int32 before them are (num_tasks, overflow, reserved, reserved) — only the first two are written, the
/// (m tile, n tile) coordinates are in units of the tactic the map was prepared for.
constexpr int kMapHeader = 4;

int64_t ceil_div64(int64_t a, int64_t b) { return (a + b - 1) / b; }

bool fp8_recipe(GemmRecipe r) {
  return r == GemmRecipe::kFP8Tensorwise || r == GemmRecipe::kFP8Rowwise || r == GemmRecipe::kFP8Blockwise ||
         r == GemmRecipe::kFP8BlockwiseUE8M0;
}

Status grouped_to_mma(const GroupedGemmParams& p, MmaGemmParams* out) {
  MmaGemmParams q{};
  ScaledGemmArgs& a = q.args;
  MmaRowMap& rm = q.rows;
  const bool fp8 = fp8_recipe(p.recipe);
  const bool i8 = p.recipe == GemmRecipe::kINT8Rowwise || p.recipe == GemmRecipe::kINT8Tensorwise;
  if (!fp8 && !i8 && p.recipe != GemmRecipe::kBF16 && p.recipe != GemmRecipe::kFP16) {
    return detail::set_error(Status::kNotImplemented, "grouped_gemm: recipe %d has no tensor-core body on this arch class", static_cast<int>(p.recipe));
  }
  if (p.b_packing != WeightPacking::kNone || p.b_zeros != nullptr || p.b_global_scale != nullptr || p.a_global_scale != nullptr) {
    return detail::set_error(Status::kNotImplemented, "grouped_gemm: packed weights, zeros and global scales have no mma.sync body (Marlin MoE / B11 / B13)");
  }
  if (p.swap_ab) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: swap_ab is a tactic choice on the mma.sync path (pass a swap tile)");
  const DType act = fp8 ? DType::kFloat8E4M3 : (i8 ? DType::kInt8 : (p.recipe == GemmRecipe::kFP16 ? DType::kFloat16 : DType::kBFloat16));
  a.recipe = p.recipe;
  a.a.data = p.a;
  a.a.ld = p.lda;
  a.a.dtype = act;
  a.b.data = p.b;
  a.b.ld = p.ldb;
  a.b.batch_stride = p.b_group_stride;
  a.b.dtype = act;
  a.d = p.d;
  a.ldd = p.ldd;
  a.epi = p.epi;
  a.d_dtype = (p.epi.ops & kEpiOutFp32) != 0u ? DType::kFloat32 : ((fp8 || i8) ? DType::kBFloat16 : act);
  a.n = p.n;
  a.k = p.k;
  const bool block = p.recipe == GemmRecipe::kFP8Blockwise || p.recipe == GemmRecipe::kFP8BlockwiseUE8M0;
  const bool tensorwise = p.recipe == GemmRecipe::kFP8Tensorwise || p.recipe == GemmRecipe::kINT8Tensorwise;
  if (p.a_scale != nullptr) {
    a.a.scale = p.a_scale;
    a.a.scale_layout = p.a_scale_layout;
    a.a.scale_dtype = p.recipe == GemmRecipe::kFP8BlockwiseUE8M0 ? DType::kFloat8E8M0 : DType::kFloat32;
    // tensorwise recipes: one activation scale [1]; rowwise: per A row (the gather source row for kIndexed)
    a.a.granularity = block ? ScaleGranularity::kPerGroup : (tensorwise ? ScaleGranularity::kPerTensor : ScaleGranularity::kPerToken);
    if (block && p.a_scale_layout == ScaleLayout::kColMajorTma) {
      // [K / 128, round_up(rows of A, 4)]: the launcher cannot derive the rows of A from the grouped row space
      int64_t a_rows = 0;
      if (p.layout == GroupedLayout::kMasked) a_rows = static_cast<int64_t>(p.max_m) * p.num_groups;
      else if (p.layout == GroupedLayout::kContiguous || p.layout == GroupedLayout::kContiguousRowIndex) a_rows = p.max_m;
      if (a_rows < 1) {
        return detail::set_error(Status::kInvalidArgument,
                                 "grouped_gemm: column-major (TMA) 1x128 activation scales need max_m = rows of A (kContiguous / kContiguousRowIndex) "
                                 "or kMasked; kIndexed takes row-major scales");
      }
      a.a.scale_ld = (a_rows + 3) / 4 * 4;
    }
  }
  if (p.b_scale != nullptr) {
    a.b.scale = p.b_scale;
    a.b.scale_batch_stride = p.b_scale_group_stride;
    a.b.scale_dtype = p.recipe == GemmRecipe::kFP8BlockwiseUE8M0 ? DType::kFloat8E8M0 : DType::kFloat32;
    if (block) {
      a.b.granularity = ScaleGranularity::kPerBlock2D;
      // [G, ceil(N / 128), K / 128] unless the caller gives the expert stride
      if (a.b.scale_batch_stride == 0) a.b.scale_batch_stride = ceil_div64(p.n, 128) * (p.k / 128);
    } else if (p.recipe == GemmRecipe::kFP8Tensorwise || p.recipe == GemmRecipe::kINT8Tensorwise) {
      a.b.granularity = ScaleGranularity::kPerTensor;   // [G]: one scale per expert
      if (a.b.scale_batch_stride == 0) a.b.scale_batch_stride = 1;
    } else {
      a.b.granularity = ScaleGranularity::kPerChannel;  // [G, N]
      if (a.b.scale_batch_stride == 0) a.b.scale_batch_stride = p.n;
    }
  }
  if ((p.epi.ops & kEpiColScale) != 0u && a.epi.col_scale_group_stride == 0) a.epi.col_scale_group_stride = p.n;
  if ((p.epi.ops & kEpiBias) != 0u && a.epi.bias_group_stride == 0) a.epi.bias_group_stride = p.n;
  rm.num_groups = p.num_groups;
  const int64_t m_hint = p.expected_m > 0 ? p.expected_m : (p.max_m > 0 ? p.max_m : 64);
  switch (p.layout) {
    case GroupedLayout::kContiguous:
    case GroupedLayout::kIndexed:
      if (p.expert_offsets == nullptr) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: expert_offsets missing");
      rm.schedule = MmaSchedule::kRowGroups;
      rm.group_rows = p.expert_offsets;
      if (p.layout == GroupedLayout::kIndexed) {
        if (p.gather_idx == nullptr) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: kIndexed needs gather_idx");
        rm.row_ids = p.gather_idx;
        rm.a_rows_by_id = true;
        rm.a_row_div = 1;
      }
      a.m = m_hint * p.num_groups;   // row space bound used only for the tile heuristic (rows / group)
      break;
    case GroupedLayout::kMasked:
      if (p.masked_m == nullptr || p.max_m < 1) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: kMasked needs masked_m and max_m");
      rm.schedule = MmaSchedule::kMaskedGroups;
      rm.masked_rows = p.masked_m;
      rm.max_rows = p.max_m;
      a.m = static_cast<int64_t>(p.max_m) * p.num_groups;
      break;
    case GroupedLayout::kContiguousRowIndex:
      if (p.m_indices == nullptr || p.max_m < 1) {
        return detail::set_error(Status::kInvalidArgument, "grouped_gemm: kContiguousRowIndex needs m_indices and max_m (rows of A)");
      }
      rm.schedule = MmaSchedule::kSortedBlocks;
      rm.block_group = p.m_indices;
      rm.block_rows = 1;   // per-row expert ids; segments padded to the tile M (64)
      a.m = p.max_m;
      break;
    default:
      return detail::set_error(Status::kNotImplemented, "grouped_gemm: layout %d has no tensor-core body (kPtrArray: lora_sgmv / B11)", static_cast<int>(p.layout));
  }
  if (p.tile_task_map != nullptr) {
    if (rm.schedule != MmaSchedule::kRowGroups && rm.schedule != MmaSchedule::kMaskedGroups) {
      return detail::set_error(Status::kInvalidArgument, "grouped_gemm: a task map serves kContiguous / kIndexed / kMasked");
    }
    rm.schedule = MmaSchedule::kTaskMap;
    rm.task_map = p.tile_task_map;
    rm.num_tasks = p.tile_task_map - kMapHeader;
  }
  *out = q;
  return Status::kSuccess;
}

/// Tasks a task map may need: masked / ptr-array G * ceil(max_m / tm) * n_tiles; contiguous (max_m = rows
/// of A) (ceil(max_m / tm) + G) * n_tiles.
int64_t task_capacity(const GroupedGemmParams& p, int tile_m, int tile_n) {
  const int64_t n_tiles = ceil_div64(p.n, tile_n);
  if (p.layout == GroupedLayout::kMasked || p.layout == GroupedLayout::kPtrArray) {
    return static_cast<int64_t>(p.num_groups) * ceil_div64(p.max_m, tile_m) * n_tiles;
  }
  return (ceil_div64(p.max_m, tile_m) + p.num_groups) * n_tiles;
}

/// Mainloops whose grouped body consumes GroupedProblem[] (TMA / tcgen05 / SM120 warp-specialized) and
/// therefore needs the generic device-side preparation. kMultistageMmaSync derives the rows in the kernel
/// from the row map and only needs the task list.
bool prepare_builds_problems(MainloopSchedule m) {
  switch (m) {
    case MainloopSchedule::kTmaWarpSpecialized:
    case MainloopSchedule::kTmaWarpSpecializedCooperative:
    case MainloopSchedule::kTmaWarpSpecializedPingpong:
    case MainloopSchedule::kTcgen05OneSm:
    case MainloopSchedule::kTcgen05TwoSm:
    case MainloopSchedule::kSm120BlockScaledWS:
    case MainloopSchedule::kSm120WarpSpecialized:
      return true;
    default:
      return false;
  }
}

/// GroupedProblem[] region of the scratch (16 B aligned), then the task-map header + entries.
int64_t problems_bytes(const GroupedGemmParams& p) {
  const int64_t raw = ceil_div64(static_cast<int64_t>(p.num_groups) * static_cast<int64_t>(sizeof(GroupedProblem)), 16) * 16;
  return raw;
}

/// Path selection. kFP8Tensorwise / kFP8Rowwise mean different activations on the two bodies: e4m3
/// activations (W8A8) on mma.sync, bf16 / fp16 activations (W8A16) on the CUDA-core GEMV. GroupedGemmParams
/// carries no activation dtype, so these two recipes take the tensor-core body only when the caller says so
/// (an mma tactic, a prepared task map, or activation scales); expected_m never changes their meaning.
bool grouped_wants_mma(const GroupedGemmParams& p, const GemmTactic* tactic) {
  if (tactic != nullptr) return tactic->mainloop == MainloopSchedule::kMultistageMmaSync;
  if (p.tile_task_map != nullptr) return true;
  const bool tc_only = p.recipe == GemmRecipe::kFP8Blockwise || p.recipe == GemmRecipe::kFP8BlockwiseUE8M0 ||
                       p.recipe == GemmRecipe::kINT8Rowwise || p.recipe == GemmRecipe::kINT8Tensorwise ||
                       p.layout == GroupedLayout::kContiguousRowIndex || p.a_scale != nullptr;
  if (tc_only) return true;
  if (p.recipe == GemmRecipe::kFP8Tensorwise || p.recipe == GemmRecipe::kFP8Rowwise) return false;
  return p.expected_m > 16 || (p.epi.ops & ~kGroupedEpiAllowed) != 0u;
}

}  // namespace

Status grouped_gemm_workspace_requirements(const GroupedGemmParams& params, const GemmTactic* tactic,
                                           const LaunchContext& ctx, WorkspaceRequirements* out) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "grouped_gemm: CUDA backend only");
  if (out == nullptr) return detail::set_error(Status::kInvalidArgument, "grouped_gemm_workspace_requirements: null output");
  *out = WorkspaceRequirements{};   // the CUDA-core path and the in-kernel task scan keep no scratch
  if (tactic == nullptr || params.max_m < 1 || params.layout == GroupedLayout::kContiguousRowIndex) return Status::kSuccess;
  const bool task_map_ok = params.layout == GroupedLayout::kContiguous || params.layout == GroupedLayout::kIndexed ||
                           params.layout == GroupedLayout::kMasked || params.layout == GroupedLayout::kPtrArray;
  if (!task_map_ok) return Status::kSuccess;
  const int64_t cap = task_capacity(params, tactic->tile_m, tactic->tile_n);
  size_t need = static_cast<size_t>(kMapHeader + 3 * cap) * sizeof(int32_t);
  if (prepare_builds_problems(tactic->mainloop)) need += static_cast<size_t>(problems_bytes(params));
  out->device_scratch_bytes = need;
  return Status::kSuccess;
}

/// Device-side preparation, graph-safe (no host sync, no allocation, one or two launches):
///   kMultistageMmaSync  the task map only (the mma.sync body derives the rows from its row map)
///   TMA / tcgen05 / SM120 warp-specialized  GroupedProblem[g] for every group plus the task map
/// Scratch layout: [GroupedProblem[G], 16 B aligned][(num_tasks, overflow, reserved, reserved)][tasks].
/// params->problems and params->tile_task_map point into it; params->group_tma stays null (per-group
/// tensor-map patching is the Sm90a/Sm100a TMA path and is not built here).
Status grouped_gemm_prepare(GroupedGemmParams* params, const GemmTactic& tactic, const Workspace& ws, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "grouped_gemm_prepare: CUDA backend only");
  if (params == nullptr) return detail::set_error(Status::kInvalidArgument, "grouped_gemm_prepare: null params");
  const bool problems_path = prepare_builds_problems(tactic.mainloop);
  if (!problems_path && tactic.mainloop != MainloopSchedule::kMultistageMmaSync) {
    return detail::set_error(Status::kNotImplemented,
                             "grouped_gemm_prepare: mainloop %d has no grouped body", static_cast<int>(tactic.mainloop));
  }
  const GroupedGemmParams& p = *params;
  const bool ptr_array = p.layout == GroupedLayout::kPtrArray;
  if (p.layout != GroupedLayout::kContiguous && p.layout != GroupedLayout::kIndexed && p.layout != GroupedLayout::kMasked &&
      !(problems_path && ptr_array)) {
    return detail::set_error(Status::kNotImplemented,
                             "grouped_gemm_prepare: task maps serve kContiguous / kIndexed / kMasked, and kPtrArray only "
                             "needs the TMA-class preparation (mainloop %d)", static_cast<int>(tactic.mainloop));
  }
  if (p.num_groups < 1 || p.n < 1 || p.max_m < 1 || tactic.tile_m < 1 || tactic.tile_n < 1) {
    return detail::set_error(Status::kInvalidArgument, "grouped_gemm_prepare: num_groups, n, max_m (rows of A or per group) and the tactic tile are required");
  }
  if (problems_path && p.max_m < 1) {
    return detail::set_error(Status::kInvalidArgument, "grouped_gemm_prepare: max_m is the row bound of one group");
  }
  if (p.layout == GroupedLayout::kMasked ? p.masked_m == nullptr
                                         : (ptr_array ? (p.problem_m == nullptr || p.a_ptrs == nullptr || p.b_ptrs == nullptr ||
                                                         p.d_ptrs == nullptr)
                                                      : p.expert_offsets == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "grouped_gemm_prepare: row metadata missing for the selected layout");
  }
  if (problems_path) {
    int a_bits = 0, b_bits = 0, d_bits = 0;
    recipe_element_bits(p.recipe, p.epi.ops, &a_bits, &b_bits, &d_bits);
    if (!recipe_is_8bit(p.recipe) && p.recipe != GemmRecipe::kBF16 && p.recipe != GemmRecipe::kFP16) {
      return detail::set_error(Status::kNotImplemented,
                               "grouped_gemm_prepare: recipe %d has no grouped problem description "
                               "(packed weights / block-scaled operand dtypes are not in GroupedGemmParams)",
                               static_cast<int>(p.recipe));
    }
    if (p.layout != GroupedLayout::kPtrArray && (p.a == nullptr || p.b == nullptr || p.d == nullptr)) {
      return detail::set_error(Status::kInvalidArgument, "grouped_gemm_prepare: null operand");
    }
    if (p.recipe == GemmRecipe::kFP8Tensorwise && p.b_scale == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "grouped_gemm_prepare: kFP8Tensorwise needs the per-expert b_scale");
    }
  }
  const int64_t cap = task_capacity(p, tactic.tile_m, tactic.tile_n);
  if (cap > 0x7FFFFFFF / 3) return detail::set_error(Status::kUnsupportedShape, "grouped_gemm_prepare: task map too large");
  const int64_t probs = problems_path ? problems_bytes(p) : 0;
  const size_t need = static_cast<size_t>(probs) + static_cast<size_t>(kMapHeader + 3 * cap) * sizeof(int32_t);
  if (ws.device_scratch == nullptr || ws.device_scratch_bytes < need) {
    return detail::set_error(Status::kWorkspaceTooSmall, "grouped_gemm_prepare: needs %zu B of device scratch", need);
  }
  if (!aligned(ws.device_scratch, 16)) return detail::set_error(Status::kMisaligned, "grouped_gemm_prepare: device scratch must be 16 B aligned");
  int32_t* hdr = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(ws.device_scratch) + probs);
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int threads = p.num_groups >= 1024 ? 1024 : ((p.num_groups + 31) / 32) * 32;
  const int n_tiles = static_cast<int>(ceil_div64(p.n, tactic.tile_n));
  const int32_t* row_counts = p.layout == GroupedLayout::kMasked ? p.masked_m
                              : (ptr_array ? p.problem_m : nullptr);
  if (problems_path) {
    grouped_gemm_prepare_kernel<<<1, threads, 0, s>>>(
        static_cast<GroupedProblem*>(ws.device_scratch), nullptr, nullptr, hdr + kMapHeader, hdr, hdr + 1, p,
        tactic.tile_m, tactic.tile_n, static_cast<int>(cap));
    Status st = launch_error("grouped_gemm_prepare_kernel");
    if (st != Status::kSuccess) return st;
    params->problems = static_cast<const GroupedProblem*>(ws.device_scratch);
    params->group_tma = nullptr;
  }
  grouped_gemm_task_map_kernel<<<1, threads, 0, s>>>(hdr + kMapHeader, hdr, hdr + 1,
                                                     p.expert_offsets, row_counts, p.num_groups, tactic.tile_m, n_tiles,
                                                     static_cast<int>(cap));
  Status st = launch_error("grouped_gemm_task_map_kernel");
  if (st != Status::kSuccess) return st;
  params->tile_task_map = hdr + kMapHeader;
  return Status::kSuccess;
}

Status grouped_gemm(const GroupedGemmParams& p, const GemmTactic* tactic, GemmBackend backend,
                    const Workspace& ws, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "grouped_gemm: CUDA backend only");
  if (backend != GemmBackend::kNative && backend != GemmBackend::kAuto) {
    return detail::set_error(Status::kUnsupportedBackend, "grouped_gemm: only the native provider exists");
  }
  if (grouped_wants_mma(p, tactic)) {
    if (p.num_groups < 1 || p.n < 1 || p.k < 1) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: empty problem");
    if (p.a == nullptr || p.b == nullptr) return detail::set_error(Status::kInvalidArgument, "grouped_gemm: null operand");
    if (p.tile_task_map != nullptr && tactic == nullptr) {
      // the map's (m tile, n tile) coordinates are in units of the tile it was prepared for
      return detail::set_error(Status::kInvalidArgument, "grouped_gemm: a prepared task map needs the tactic grouped_gemm_prepare used");
    }
    if (p.layout == GroupedLayout::kContiguousRowIndex && tactic != nullptr && (tactic->tile_m < 1 || 64 % tactic->tile_m != 0)) {
      // a tile takes the expert of its first row: segments padded to 64 rows need a tile M dividing 64
      return detail::set_error(Status::kUnsupportedShape, "grouped_gemm: kContiguousRowIndex needs a tile M dividing 64 (segments padded to 64 rows)");
    }
    MmaGemmParams mp{};
    Status st = grouped_to_mma(p, &mp);
    if (st != Status::kSuccess) return st;
    GemmTactic t;
    const GemmTactic* tp = tactic;
    if (tp != nullptr && tp->split_k_slices > 1) {
      t = *tp;
      t.split_k_mode = SplitKMode::kNone;   // grouped schedules do not split K
      t.split_k_slices = 1;
      tp = &t;
    }
    (void)ws;
    return launch_gemm_mma_sync(mp, tp, Workspace{}, ctx);
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

// =============================================================================
// B7 row 9: multi-LoRA (Punica SGMV) on the shared mma.sync body
// =============================================================================
// shrink: buffer[t, :rank] = x[t] . A[id]^T            (fp32, written)
// expand: y[t, off : off + out_dim] += scaling * buffer[t] . B[id]^T   (in the output dtype: read, add, round)
// Tokens of segment s use adapter slot seg_lora_ids[s]; -1 leaves the segment untouched (buffer and y).
// lora_a of each stage is a DEVICE array of per-slot weight pointers. The shrink output and the expand
// input are shrink_buffer (fp32 [T, rank], T = shrink_buffer.shape[0]); shrink.y / expand.x, when set,
// must point at it. Either stage may be skipped by passing lora_a == nullptr.
Status lora_sgmv(const LoraSgmvParams& shrink, const LoraSgmvParams& expand, DType dtype, const Tensor& shrink_buffer,
                 const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) return detail::set_error(Status::kUnsupportedBackend, "lora_sgmv: CUDA backend only");
  if (dtype != DType::kBFloat16 && dtype != DType::kFloat16) return detail::set_error(Status::kUnsupportedDtype, "lora_sgmv: bf16 or fp16");
  if (shrink.lora_a == nullptr && expand.lora_a == nullptr) return detail::set_error(Status::kInvalidArgument, "lora_sgmv: nothing to do");
  if (shrink_buffer.data == nullptr || shrink_buffer.dtype != DType::kFloat32 || shrink_buffer.ndim != 2) {
    return detail::set_error(Status::kInvalidArgument, "lora_sgmv: shrink_buffer is an fp32 [T, rank] tensor");
  }
  float* const buf = reinterpret_cast<float*>(static_cast<char*>(shrink_buffer.data) + shrink_buffer.byte_offset);
  const int64_t T = shrink_buffer.shape[0];
  const int64_t ldbuf = shrink_buffer.stride[0];
  const GemmRecipe recipe = dtype == DType::kBFloat16 ? GemmRecipe::kBF16 : GemmRecipe::kFP16;
  auto segments = [](const LoraSgmvParams& q, MmaRowMap* rm) -> Status {
    if (q.seg_indptr == nullptr || q.seg_lora_ids == nullptr || q.num_segments < 1) {
      return detail::set_error(Status::kInvalidArgument, "lora_sgmv: seg_indptr, seg_lora_ids and num_segments are required");
    }
    rm->schedule = MmaSchedule::kRowGroups;
    rm->num_groups = q.num_segments;
    rm->group_rows32 = q.seg_indptr;
    rm->group_slot = q.seg_lora_ids;
    rm->b_ptrs = q.lora_a;
    return Status::kSuccess;
  };
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  (void)s;
  if (shrink.lora_a != nullptr) {
    if (shrink.y != nullptr && shrink.y != buf) return detail::set_error(Status::kInvalidArgument, "lora_sgmv: shrink.y must be shrink_buffer");
    if (shrink.x == nullptr || shrink.rank < 1 || shrink.in_dim < 1 || shrink.ldx < shrink.in_dim) {
      return detail::set_error(Status::kInvalidArgument, "lora_sgmv: shrink needs x, ldx >= in_dim and rank");
    }
    if (shrink_buffer.shape[1] < shrink.rank || ldbuf < shrink.rank) return detail::set_error(Status::kInvalidArgument, "lora_sgmv: shrink_buffer narrower than rank");
    MmaGemmParams mp{};
    Status st = segments(shrink, &mp.rows);
    if (st != Status::kSuccess) return st;
    ScaledGemmArgs& a = mp.args;
    a.recipe = recipe;
    a.a.data = shrink.x; a.a.ld = shrink.ldx; a.a.dtype = dtype;
    a.b.data = nullptr; a.b.ld = shrink.in_dim; a.b.dtype = dtype;
    a.d = buf; a.ldd = ldbuf; a.d_dtype = DType::kFloat32;
    a.epi.ops = kEpiOutFp32;
    a.m = T; a.n = shrink.rank; a.k = shrink.in_dim;
    st = launch_gemm_mma_sync(mp, nullptr, Workspace{}, ctx);
    if (st != Status::kSuccess) return st;
  }
  if (expand.lora_a != nullptr) {
    if (expand.x != nullptr && expand.x != buf) return detail::set_error(Status::kInvalidArgument, "lora_sgmv: expand.x must be shrink_buffer");
    if (expand.y == nullptr || expand.rank < 8 || expand.out_dim < 1 || expand.out_slice_offset < 0 || expand.ldy < expand.out_slice_offset + expand.out_dim) {
      return detail::set_error(Status::kInvalidArgument, "lora_sgmv: expand needs y, rank >= 8, out_dim and ldy covering the slice");
    }
    if (shrink_buffer.shape[1] < expand.rank) return detail::set_error(Status::kInvalidArgument, "lora_sgmv: shrink_buffer narrower than rank");
    MmaGemmParams mp{};
    Status st = segments(expand, &mp.rows);
    if (st != Status::kSuccess) return st;
    ScaledGemmArgs& a = mp.args;
    const int64_t es = 2;
    void* out = static_cast<char*>(expand.y) + expand.out_slice_offset * es;
    a.recipe = recipe;
    a.a.data = buf; a.a.ld = ldbuf; a.a.dtype = DType::kFloat32;   // fp32 activations, rounded to dtype in the loader
    a.b.data = nullptr; a.b.ld = expand.rank; a.b.dtype = dtype;
    a.d = out; a.ldd = expand.ldy; a.d_dtype = dtype;
    a.epi.ops = kEpiResidualAdd;
    a.epi.alpha = expand.scaling;
    a.epi.beta = 1.0f;
    a.epi.residual = out;           // y += ...: each element is read and written by one thread
    a.epi.residual_stride = expand.ldy;
    a.m = T; a.n = expand.out_dim; a.k = expand.rank;
    st = launch_gemm_mma_sync(mp, nullptr, Workspace{}, ctx);
    if (st != Status::kSuccess) return st;
  }
  return Status::kSuccess;
}

}  // namespace kops::gemm
