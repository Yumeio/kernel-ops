// =============================================================================
// kernel-ops | csrc/norm/norm.cu
// N01: rms_norm, fused_add_rms_norm, fused_add_rms_norm_out, layer_norm,
//      fused_add_layer_norm, qk_norm, l2_norm, gated_rms_norm.
// N02 (norm_quant) is still a declared contract.
//
// Shape: every kernel sees one row of `hidden` contiguous elements addressed by
// a single row stride (leading dims collapse; qk_norm indexes heads by
// token_stride + head * HEAD_DIM). The RMSNorm bodies read 32 B per thread (two
// 16 B packs) and the other vector paths 16 B (8 halves / 4 floats); a scalar
// tail handles hidden % VEC and a stride that is not a whole number of packs
// (the public contract already requires 16 B aligned row strides).
//
// Numerics: fp32 accumulation, one rounding at the store. RMSNorm accumulates
// the sum of squares; LayerNorm does mean then centred (biased) variance, like
// torch/F.layer_norm; L2 divides by the sum, not the mean. The residual-fused
// kernels round the sum once to the residual dtype and normalize the stored
// value; fused_add_rms_norm_out repeats fused_add_rms_norm's exact expression
// so the two entries agree bit for bit.
//
// Launch: one CTA per row (grid == rows) for the row kernels, 4 warps per CTA
// with one warp per head for per-head qk_norm, and one CTA (or one warp when
// group_size <= 256) per group for gated_rms_norm. All row kernels are
// graph-capturable: no allocation, no device query, no host sync.
//
// rms_norm and the fused add+RMSNorm variants stage the row in dynamic shared
// memory (46 KB cap, 8 KB for 4096 bf16), so the row crosses HBM once; rows
// that do not fit re-read it on the second pass from L1/L2, as LayerNorm still
// does. Staging is worth ~1% of DRAM bandwidth at 4096 wide and keeps the
// second pass out of L1.
//
// Target support: this body is arch-generic (no cp.async, no TMA) and is meant
// for Tier A/B/C alike; exact-target selection happens in the registry.
// =============================================================================
#include <cmath>
#include <cstdio>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"

namespace kops::norm {
namespace {

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T x[N];
};

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ void from_f(T& d, float s);
template <>
__device__ __forceinline__ void from_f<float>(float& d, float s) { d = s; }
template <>
__device__ __forceinline__ void from_f<__half>(__half& d, float s) { d = __float2half_rn(s); }
template <>
__device__ __forceinline__ void from_f<__nv_bfloat16>(__nv_bfloat16& d, float s) {
  d = __float2bfloat16_rn(s);
}

/// Sum across the whole block; every thread receives the result.
__device__ __forceinline__ float block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float t = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

/// Row offset for a view whose only non-contiguous dims are -2, -3, -4.
template <int NUM_DIMS>
__device__ __forceinline__ int64_t row_offset(int64_t row, int64_t s2, int64_t s3, int64_t s4,
                                              int64_t n2, int64_t n3) {
  if (NUM_DIMS <= 2) return row * s2;
  const int64_t i2 = row % n2;
  const int64_t t = row / n2;
  if (NUM_DIMS == 3) return t * s3 + i2 * s2;
  const int64_t i3 = t % n3;
  const int64_t i4 = t / n3;
  return i4 * s4 + i3 * s3 + i2 * s2;
}

// ---- row staging -------------------------------------------------------------
// Every row kernel below reads its row once into dynamic shared memory, reduces
// it, and normalizes from shared memory, so the row crosses HBM once (and the
// fused variants keep the just-written residual in shared memory instead of
// re-reading it from L1/L2). 46 KB keeps the static reduction buffer plus the
// dynamic stage below the 48 KB no-opt-in limit of every supported target;
// rows that do not fit take the direct path, which reads the row twice.
constexpr int64_t kNormStageBytes = 46 * 1024;

__host__ __device__ inline bool norm_stage_fits(int hidden, int64_t elem_bytes) {
  return static_cast<int64_t>(hidden) * elem_bytes <= kNormStageBytes;
}

/// One 16 B pack through the read-only path (ld.global.nc).
template <typename T, int VEC>
__device__ __forceinline__ Pack<T, VEC> ldg16(const T* __restrict__ p) {
  static_assert(sizeof(Pack<T, VEC>) == 16, "a pack is one 16-byte access");
  const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
  Pack<T, VEC> r;
  __builtin_memcpy(&r, &u, sizeof(r));
  return r;
}

/// One 16 B pack store; used for shared-memory staging and for global stores.
template <typename T, int VEC>
__device__ __forceinline__ void st16(T* __restrict__ p, const Pack<T, VEC>& v) {
  static_assert(sizeof(Pack<T, VEC>) == 16, "a pack is one 16-byte access");
  *reinterpret_cast<Pack<T, VEC>*>(p) = v;
}

/// One 16 B pack from a generic address (shared or global): the second pass of a
/// staged row reads shared memory, the direct path reads global memory again.
template <typename T, int VEC>
__device__ __forceinline__ Pack<T, VEC> ld16(const T* __restrict__ p) {
  static_assert(sizeof(Pack<T, VEC>) == 16, "a pack is one 16-byte access");
  return *reinterpret_cast<const Pack<T, VEC>*>(p);
}

/// 16 B stage buffer shared by the row kernels of this file.
extern __shared__ __align__(16) unsigned char kops_norm_stage[];

/// RMSNorm of one row. Pass 1 reads the row once from global memory (32 B per
/// thread), stages it in shared memory when it fits, and accumulates the sum of
/// squares in fp32; pass 2 normalizes from the stage (or re-reads the row on the
/// direct path). `red` is the float[32] reduction scratch.
template <typename T, int VEC, bool HasWeight>
__device__ __forceinline__ void rms_row(const T* __restrict__ x, T* __restrict__ y,
                                        const T* __restrict__ w, int num_weight_groups,
                                        float weight_offset, float eps, int hidden,
                                        int64_t weight_stride, int64_t row, float* red) {
  T* const srow = reinterpret_cast<T*>(kops_norm_stage);
  const bool staged = norm_stage_fits(hidden, static_cast<int64_t>(sizeof(T)));
  constexpr int CH = (VEC == 1) ? 1 : 2 * VEC;   // 32 B per thread
  float ss = 0.0f;
  if constexpr (VEC == 1) {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float v = to_f<T>(x[i]);
      if (staged) srow[i] = x[i];
      ss += v * v;
    }
  } else {
    const int nchunk = hidden / CH;
    for (int i = threadIdx.x; i < nchunk; i += blockDim.x) {
      const int k = i * CH;
      const Pack<T, VEC> p0 = ldg16<T, VEC>(x + k);
      const Pack<T, VEC> p1 = ldg16<T, VEC>(x + k + VEC);
      if (staged) {
        st16<T, VEC>(srow + k, p0);
        st16<T, VEC>(srow + k + VEC, p1);
      }
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float v = to_f<T>(p0.x[j]);
        ss += v * v;
      }
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float v = to_f<T>(p1.x[j]);
        ss += v * v;
      }
    }
    const int nvec = hidden / VEC;
    for (int i = nchunk * (CH / VEC) + threadIdx.x; i < nvec; i += blockDim.x) {
      const Pack<T, VEC> p = ldg16<T, VEC>(x + i * VEC);
      if (staged) st16<T, VEC>(srow + i * VEC, p);
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float v = to_f<T>(p.x[j]);
        ss += v * v;
      }
    }
    for (int i = nvec * VEC + threadIdx.x; i < hidden; i += blockDim.x) {
      const float v = to_f<T>(x[i]);
      if (staged) srow[i] = x[i];
      ss += v * v;
    }
  }
  ss = block_sum(ss, red);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  const T* const src = staged ? srow : x;
  const T* wp = w;
  if (HasWeight && num_weight_groups > 1) {
    wp = w + static_cast<int64_t>(row % num_weight_groups) * weight_stride;
  }
  if constexpr (VEC == 1) {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float s = to_f<T>(src[i]) * rstd;
      const float wv = HasWeight ? (to_f<T>(wp[i]) + weight_offset) : 1.0f;
      from_f<T>(y[i], s * wv);
    }
  } else {
    const int nvec = hidden / VEC;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      const Pack<T, VEC> p = ld16<T, VEC>(src + i * VEC);
      Pack<T, VEC> q;
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float s = to_f<T>(p.x[j]) * rstd;
        const float wv = HasWeight ? (to_f<T>(wp[i * VEC + j]) + weight_offset) : 1.0f;
        from_f<T>(q.x[j], s * wv);
      }
      st16<T, VEC>(y + i * VEC, q);
    }
    for (int i = nvec * VEC + threadIdx.x; i < hidden; i += blockDim.x) {
      const float s = to_f<T>(src[i]) * rstd;
      const float wv = HasWeight ? (to_f<T>(wp[i]) + weight_offset) : 1.0f;
      from_f<T>(y[i], s * wv);
    }
  }
}

// ---- helpers shared by the row kernels below ---------------------------------

/// Butterfly sum across one 32-lane warp; every lane receives the total. Used by
/// the warp-per-row launches, where block_sum's __syncthreads would couple rows
/// that merely share a CTA.
__device__ __forceinline__ float warp_sum_all(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

/// Row reduction: warp-local butterfly (kWarp) or the CTA tree of block_sum.
template <bool kWarp>
__device__ __forceinline__ float row_sum(float v, float* smem) {
  if constexpr (kWarp) {
    return warp_sum_all(v);
  } else {
    return block_sum(v, smem);
  }
}

/// One gate value in fp32. Only identity / sigmoid / SiLU are instantiated; the
/// host rejects every other ActFn before launching.
template <ActFn kAct>
__device__ __forceinline__ float gate_value(float z) {
  if constexpr (kAct == ActFn::kIdentity) return z;
  const float e = __expf(-fabsf(z));
  const float s = z >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
  if constexpr (kAct == ActFn::kSigmoid) return s;
  return z * s;  // kSiLU
}

/// 16-byte vector width of one head row, narrowed until a full warp has work (a
/// 64-wide bf16 head reads 2 elements per lane, not 8 elements on 8 lanes).
template <typename T, int HEAD_DIM>
__host__ __device__ constexpr int head_vector_width() {
  int n = 16 / static_cast<int>(sizeof(T));
  while (n > 1 && HEAD_DIM < 32 * n) n >>= 1;
  return n;
}

/// Norm of one contiguous row of `width` elements: fp32 reductions, one rounding
/// at the store. src and dst may be the same row (in-place norms). `bias` and
/// `weight` are optional. `global_mean_square` (RMSNorm only) is the caller's
/// all-reduced MEAN square, i.e. the sum of x^2 over every TP rank divided by the
/// global width; it replaces the local sum of squares so the full-width TP QK
/// norm never reads the row twice. The public qk_norm signature carries no TP
/// world size, so the mean (not the raw sum) is the only self-contained form of
/// that cross-rank partial. kL2Norm divides by the sum (not the mean) and keeps
/// the affine if present.
template <typename T, int VEC, NormType kN, bool kWarp>
__device__ __forceinline__ void norm_row(const T* src, T* dst, const T* __restrict__ w,
                                         const T* __restrict__ bias, int width, float weight_offset,
                                         float epsilon, const float* global_mean_square,
                                         float* smem) {
  const int nvec = width / VEC;
  const bool has_w = w != nullptr;
  const bool has_bias = bias != nullptr;
  [[maybe_unused]] float center = 0.0f;  // LayerNorm mean; unused by RMS/L2
  float inv = 0.0f;
  if constexpr (kN == NormType::kLayerNorm) {
    // torch/F.layer_norm order: mean, then the biased variance of (x - mean).
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      const Pack<T, VEC> p = *reinterpret_cast<const Pack<T, VEC>*>(src + i * VEC);
      for (int k = 0; k < VEC; ++k) sum += to_f<T>(p.x[k]);
    }
    for (int i = nvec * VEC + threadIdx.x; i < width; i += blockDim.x) sum += to_f<T>(src[i]);
    sum = row_sum<kWarp>(sum, smem);
    center = sum / static_cast<float>(width);
    float var = 0.0f;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      const Pack<T, VEC> p = *reinterpret_cast<const Pack<T, VEC>*>(src + i * VEC);
      for (int k = 0; k < VEC; ++k) {
        const float d = to_f<T>(p.x[k]) - center;
        var += d * d;
      }
    }
    for (int i = nvec * VEC + threadIdx.x; i < width; i += blockDim.x) {
      const float d = to_f<T>(src[i]) - center;
      var += d * d;
    }
    var = row_sum<kWarp>(var, smem);
    inv = rsqrtf(var / static_cast<float>(width) + epsilon);
  } else if (kN == NormType::kL2Norm) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      const Pack<T, VEC> p = *reinterpret_cast<const Pack<T, VEC>*>(src + i * VEC);
      for (int k = 0; k < VEC; ++k) {
        const float v = to_f<T>(p.x[k]);
        ss += v * v;
      }
    }
    for (int i = nvec * VEC + threadIdx.x; i < width; i += blockDim.x) {
      const float v = to_f<T>(src[i]);
      ss += v * v;
    }
    ss = row_sum<kWarp>(ss, smem);
    inv = rsqrtf(ss + epsilon);  // true L2: the sum, not the mean
  } else if (global_mean_square == nullptr) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      const Pack<T, VEC> p = *reinterpret_cast<const Pack<T, VEC>*>(src + i * VEC);
      for (int k = 0; k < VEC; ++k) {
        const float v = to_f<T>(p.x[k]);
        ss += v * v;
      }
    }
    for (int i = nvec * VEC + threadIdx.x; i < width; i += blockDim.x) {
      const float v = to_f<T>(src[i]);
      ss += v * v;
    }
    ss = row_sum<kWarp>(ss, smem);
    inv = rsqrtf(ss / static_cast<float>(width) + epsilon);
  } else {
    inv = rsqrtf(*global_mean_square + epsilon);
  }
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const Pack<T, VEC> p = *reinterpret_cast<const Pack<T, VEC>*>(src + i * VEC);
    Pack<T, VEC> q;
    for (int k = 0; k < VEC; ++k) {
      float v = to_f<T>(p.x[k]);
      if constexpr (kN == NormType::kLayerNorm) v -= center;
      v *= inv;
      if (has_w) v *= to_f<T>(w[i * VEC + k]) + weight_offset;
      if (has_bias) v += to_f<T>(bias[i * VEC + k]);
      from_f<T>(q.x[k], v);
    }
    *reinterpret_cast<Pack<T, VEC>*>(dst + i * VEC) = q;
  }
  for (int i = nvec * VEC + threadIdx.x; i < width; i += blockDim.x) {
    float v = to_f<T>(src[i]);
    if constexpr (kN == NormType::kLayerNorm) v -= center;
    v *= inv;
    if (has_w) v *= to_f<T>(w[i]) + weight_offset;
    if (has_bias) v += to_f<T>(bias[i]);
    from_f<T>(dst[i], v);
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// N01 kernels
// -----------------------------------------------------------------------------

template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HasWeight>
__global__ void rms_norm_kernel(scalar_t* __restrict__ out, const int64_t out_stride,
                                const scalar_t* __restrict__ input,
                                const int64_t input_stride_d2, const int64_t input_stride_d3,
                                const int64_t input_stride_d4, const int64_t input_shape_d2,
                                const int64_t input_shape_d3, const scalar_t* __restrict__ weight,
                                const int64_t weight_stride, const int num_weight_groups,
                                const float weight_offset, const float epsilon,
                                const int num_tokens, const int hidden_size) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  const int64_t off = row_offset<NUM_DIMS>(row, input_stride_d2, input_stride_d3, input_stride_d4,
                                           input_shape_d2, input_shape_d3);
  rms_row<scalar_t, VEC_SIZE, HasWeight>(
      input + off, out + row * out_stride, weight, num_weight_groups, weight_offset, epsilon,
      hidden_size, weight_stride, row, smem);
}

/// Fused residual-add + RMSNorm body, shared by the in-place and the
/// out-of-place entry point so the two agree bit for bit.
///   * pass 1 writes residual_out = residual_in + input_scale * input (rounded
///     once to residual_t) and accumulates ss from the unrounded fp32 sum, the
///     expression both entries used before staging existed;
///   * pass 2 normalizes the stored residual, read from shared memory when the
///     row fits (kNormStageBytes) and from residual_out otherwise.
template <typename scalar_t, typename residual_t, int VEC_SIZE, bool HasWeight>
__device__ __forceinline__ void fused_add_rms_norm_row(
    const scalar_t* __restrict__ x, const residual_t* __restrict__ rin,
    residual_t* __restrict__ rout, scalar_t* __restrict__ dst, const scalar_t* __restrict__ weight,
    float weight_offset, float input_scale, float epsilon, int hidden_size, float* red) {
  constexpr int VECR = (16 / static_cast<int>(sizeof(residual_t))) < VEC_SIZE
                           ? 16 / static_cast<int>(sizeof(residual_t))
                           : VEC_SIZE;
  constexpr int CH = (VEC_SIZE == 1) ? 1 : 2 * VEC_SIZE;
  constexpr int NRS = VEC_SIZE / VECR;  // residual packs per VEC-element group
  residual_t* const srow = reinterpret_cast<residual_t*>(kops_norm_stage);
  const bool staged = norm_stage_fits(hidden_size, static_cast<int64_t>(sizeof(residual_t)));
  const int nvec = hidden_size / VEC_SIZE;
  float ss = 0.0f;
  if constexpr (VEC_SIZE == 1) {
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      float rv = to_f<residual_t>(rin[i]) + input_scale * to_f<scalar_t>(x[i]);
      from_f<residual_t>(rout[i], rv);
      if (staged) srow[i] = rout[i];
      ss += rv * rv;
    }
  } else {
    // 32 B per thread per iteration, then whole 16 B groups, then the tail.
    const int nchunk = hidden_size / CH;
    for (int i = threadIdx.x; i < nchunk; i += blockDim.x) {
      const int k = i * CH;
      Pack<scalar_t, VEC_SIZE> p[CH / VEC_SIZE];
      Pack<residual_t, VECR> q[CH / VECR];
#pragma unroll
      for (int v = 0; v < CH / VEC_SIZE; ++v) p[v] = ldg16<scalar_t, VEC_SIZE>(x + k + v * VEC_SIZE);
#pragma unroll
      for (int v = 0; v < CH / VECR; ++v) q[v] = ld16<residual_t, VECR>(rin + k + v * VECR);
#pragma unroll
      for (int e = 0; e < CH; ++e) {
        float rv = to_f<residual_t>(q[e / VECR].x[e % VECR]) +
                   input_scale * to_f<scalar_t>(p[e / VEC_SIZE].x[e % VEC_SIZE]);
        from_f<residual_t>(q[e / VECR].x[e % VECR], rv);
        ss += rv * rv;
      }
#pragma unroll
      for (int v = 0; v < CH / VECR; ++v) {
        st16<residual_t, VECR>(rout + k + v * VECR, q[v]);
        if (staged) st16<residual_t, VECR>(srow + k + v * VECR, q[v]);
      }
    }
    for (int i = nchunk * (CH / VEC_SIZE) + threadIdx.x; i < nvec; i += blockDim.x) {
      const Pack<scalar_t, VEC_SIZE> p = ldg16<scalar_t, VEC_SIZE>(x + i * VEC_SIZE);
      Pack<residual_t, VECR> q[NRS];
#pragma unroll
      for (int v = 0; v < NRS; ++v) q[v] = ld16<residual_t, VECR>(rin + i * VEC_SIZE + v * VECR);
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        float rv = to_f<residual_t>(q[e / VECR].x[e % VECR]) +
                   input_scale * to_f<scalar_t>(p.x[e]);
        from_f<residual_t>(q[e / VECR].x[e % VECR], rv);
        ss += rv * rv;
      }
#pragma unroll
      for (int v = 0; v < NRS; ++v) {
        st16<residual_t, VECR>(rout + i * VEC_SIZE + v * VECR, q[v]);
        if (staged) st16<residual_t, VECR>(srow + i * VEC_SIZE + v * VECR, q[v]);
      }
    }
    for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden_size; i += blockDim.x) {
      float rv = to_f<residual_t>(rin[i]) + input_scale * to_f<scalar_t>(x[i]);
      from_f<residual_t>(rout[i], rv);
      if (staged) srow[i] = rout[i];
      ss += rv * rv;
    }
  }
  ss = block_sum(ss, red);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden_size) + epsilon);
  // Pass 2: normalize the stored residual. residual_out may alias residual_in,
  // and dst may alias x; everything is read before anything is overwritten.
  const residual_t* const rsrc = staged ? srow : rout;
  if constexpr (VEC_SIZE == 1) {
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      const float wv = HasWeight ? (to_f<scalar_t>(weight[i]) + weight_offset) : 1.0f;
      from_f<scalar_t>(dst[i], to_f<residual_t>(rsrc[i]) * rstd * wv);
    }
  } else {
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      Pack<residual_t, VECR> q[NRS];
#pragma unroll
      for (int v = 0; v < NRS; ++v) q[v] = ld16<residual_t, VECR>(rsrc + i * VEC_SIZE + v * VECR);
      Pack<scalar_t, VEC_SIZE> p;
#pragma unroll
      for (int e = 0; e < VEC_SIZE; ++e) {
        const float wv =
            HasWeight ? (to_f<scalar_t>(weight[i * VEC_SIZE + e]) + weight_offset) : 1.0f;
        from_f<scalar_t>(p.x[e], to_f<residual_t>(q[e / VECR].x[e % VECR]) * rstd * wv);
      }
      st16<scalar_t, VEC_SIZE>(dst + i * VEC_SIZE, p);
    }
    for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden_size; i += blockDim.x) {
      const float wv = HasWeight ? (to_f<scalar_t>(weight[i]) + weight_offset) : 1.0f;
      from_f<scalar_t>(dst[i], to_f<residual_t>(rsrc[i]) * rstd * wv);
    }
  }
}

template <typename scalar_t, typename residual_t, int VEC_SIZE, bool HasWeight>
__global__ void fused_add_rms_norm_kernel(scalar_t* __restrict__ input, const int64_t input_stride,
                                          residual_t* __restrict__ residual,
                                          const scalar_t* __restrict__ weight,
                                          const float weight_offset, const float input_scale,
                                          const float epsilon, const int num_tokens,
                                          const int hidden_size) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  fused_add_rms_norm_row<scalar_t, residual_t, VEC_SIZE, HasWeight>(
      input + row * input_stride, residual + row * input_stride, residual + row * input_stride,
      input + row * input_stride, weight, weight_offset, input_scale, epsilon, hidden_size, smem);
}

template <typename scalar_t, typename residual_t, int VEC_SIZE, bool HasWeight>
__global__ void fused_add_rms_norm_out_kernel(
    scalar_t* __restrict__ out, residual_t* __restrict__ residual_out,
    const scalar_t* __restrict__ input, const int64_t input_stride,
    const residual_t* __restrict__ residual_in, const scalar_t* __restrict__ weight,
    const float weight_offset, const float input_scale, const float epsilon, const int num_tokens,
    const int hidden_size) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  fused_add_rms_norm_row<scalar_t, residual_t, VEC_SIZE, HasWeight>(
      input + row * input_stride, residual_in + row * input_stride,
      residual_out + row * input_stride, out + row * static_cast<int64_t>(hidden_size), weight,
      weight_offset, input_scale, epsilon, hidden_size, smem);
}

template <typename scalar_t, int VEC_SIZE, bool HasWeight, bool HasBias>
__global__ void layer_norm_kernel(scalar_t* __restrict__ out, const int64_t out_stride,
                                  const scalar_t* __restrict__ input, const int64_t input_stride,
                                  const scalar_t* __restrict__ weight,
                                  const scalar_t* __restrict__ bias, const float weight_offset,
                                  const float epsilon, const int num_tokens,
                                  const int hidden_size) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  norm_row<scalar_t, VEC_SIZE, NormType::kLayerNorm, false>(
      input + row * input_stride, out + row * out_stride, HasWeight ? weight : nullptr,
      HasBias ? bias : nullptr, hidden_size, weight_offset, epsilon, nullptr, smem);
}

template <typename scalar_t, typename residual_t, int VEC_SIZE, bool HasWeight, bool HasBias>
__global__ void fused_add_layer_norm_kernel(
    scalar_t* __restrict__ input, const int64_t input_stride, residual_t* __restrict__ residual,
    const scalar_t* __restrict__ weight, const scalar_t* __restrict__ bias,
    const float weight_offset, const float input_scale, const float epsilon, const int num_tokens,
    const int hidden_size) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= num_tokens) return;
  scalar_t* x = input + row * input_stride;
  residual_t* r = residual + row * input_stride;
  const int nvec = hidden_size / VEC_SIZE;
  // Pass 1: residual += input_scale * input; accumulate the sum of the value
  // that is stored (rounded once to residual_t, as in fused_add_rms_norm).
  float sum = 0.0f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const Pack<scalar_t, VEC_SIZE> p =
        *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(x + i * VEC_SIZE);
    Pack<residual_t, VEC_SIZE> q =
        *reinterpret_cast<const Pack<residual_t, VEC_SIZE>*>(r + i * VEC_SIZE);
    for (int k = 0; k < VEC_SIZE; ++k) {
      const float rv = to_f<residual_t>(q.x[k]) + input_scale * to_f<scalar_t>(p.x[k]);
      from_f<residual_t>(q.x[k], rv);
      sum += to_f<residual_t>(q.x[k]);
    }
    *reinterpret_cast<Pack<residual_t, VEC_SIZE>*>(r + i * VEC_SIZE) = q;
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float rv = to_f<residual_t>(r[i]) + input_scale * to_f<scalar_t>(x[i]);
    from_f<residual_t>(r[i], rv);
    sum += to_f<residual_t>(r[i]);
  }
  sum = block_sum(sum, smem);
  const float mean = sum / static_cast<float>(hidden_size);
  // Pass 2: biased variance of the stored residual.
  float var = 0.0f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const Pack<residual_t, VEC_SIZE> q =
        *reinterpret_cast<const Pack<residual_t, VEC_SIZE>*>(r + i * VEC_SIZE);
    for (int k = 0; k < VEC_SIZE; ++k) {
      const float d = to_f<residual_t>(q.x[k]) - mean;
      var += d * d;
    }
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float d = to_f<residual_t>(r[i]) - mean;
    var += d * d;
  }
  var = block_sum(var, smem);
  const float rstd = rsqrtf(var / static_cast<float>(hidden_size) + epsilon);
  // Pass 3: write the normalized value over the sublayer output.
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const Pack<residual_t, VEC_SIZE> q =
        *reinterpret_cast<const Pack<residual_t, VEC_SIZE>*>(r + i * VEC_SIZE);
    Pack<scalar_t, VEC_SIZE> p;
    for (int k = 0; k < VEC_SIZE; ++k) {
      float v = (to_f<residual_t>(q.x[k]) - mean) * rstd;
      if (HasWeight) v *= to_f<scalar_t>(weight[i * VEC_SIZE + k]) + weight_offset;
      if (HasBias) v += to_f<scalar_t>(bias[i * VEC_SIZE + k]);
      from_f<scalar_t>(p.x[k], v);
    }
    *reinterpret_cast<Pack<scalar_t, VEC_SIZE>*>(x + i * VEC_SIZE) = p;
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < hidden_size; i += blockDim.x) {
    float v = (to_f<residual_t>(r[i]) - mean) * rstd;
    if (HasWeight) v *= to_f<scalar_t>(weight[i]) + weight_offset;
    if (HasBias) v += to_f<scalar_t>(bias[i]);
    from_f<scalar_t>(x[i], v);
  }
}

template <typename scalar_t, int HEAD_DIM, NormType kNorm, bool kFullWidth>
__global__ void qk_norm_kernel(scalar_t* __restrict__ qkv, const int64_t token_stride,
                               const int64_t q_offset, const int64_t k_offset,
                               const scalar_t* __restrict__ q_weight,
                               const scalar_t* __restrict__ k_weight,
                               const int64_t q_weight_head_stride,
                               const int64_t k_weight_head_stride,
                               const float* __restrict__ full_width_sq_partials,
                               const float weight_offset, const float epsilon, const int num_tokens,
                               const int num_q_heads, const int num_kv_heads) {
  if constexpr (kFullWidth) {
    // One CTA per (token, kind); blockIdx.y selects q (0) or k (1) when both
    // exist. A q-only or k-only launch (separate q/k tensors) uses grid.y == 1.
    __shared__ float smem[32];
    const int64_t token = blockIdx.x;
    if (token >= num_tokens) return;
    const int kind = ((num_q_heads > 0) ? 0 : 1) + static_cast<int>(blockIdx.y);
    const int heads = (kind == 0) ? num_q_heads : num_kv_heads;
    const int width = heads * HEAD_DIM;
    scalar_t* row = qkv + token * token_stride + (kind == 0 ? q_offset : k_offset);
    const scalar_t* w = (kind == 0) ? q_weight : k_weight;
    const float* mean_square =
        (full_width_sq_partials != nullptr) ? full_width_sq_partials + token * 2 + kind : nullptr;
    norm_row<scalar_t, 16 / static_cast<int>(sizeof(scalar_t)), kNorm, false>(
        row, row, w, nullptr, width, weight_offset, epsilon, mean_square, smem);
  } else {
    // One warp per (token, head); kWarps independent rows per CTA.
    constexpr int kWarps = 4;
    constexpr int VEC = head_vector_width<scalar_t, HEAD_DIM>();
    const int64_t heads = static_cast<int64_t>(num_q_heads) + num_kv_heads;
    const int64_t task = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.y;
    if (task >= heads * num_tokens) return;
    const int64_t token = task / heads;
    const int64_t h = task - token * heads;
    const bool is_q = h < num_q_heads;
    const int64_t hh = is_q ? h : h - num_q_heads;
    scalar_t* row = qkv + token * token_stride + (is_q ? q_offset : k_offset) + hh * HEAD_DIM;
    const scalar_t* w = is_q ? q_weight : k_weight;
    if (w != nullptr) w += hh * (is_q ? q_weight_head_stride : k_weight_head_stride);
    norm_row<scalar_t, VEC, kNorm, true>(row, row, w, nullptr, HEAD_DIM, weight_offset, epsilon,
                                         nullptr, nullptr);
  }
}

template <typename scalar_t, int VEC_SIZE>
__global__ void l2_norm_kernel(scalar_t* __restrict__ out, const int64_t out_stride,
                               const scalar_t* __restrict__ input, const int64_t input_stride,
                               const float scale, const float epsilon, const int num_rows,
                               const int dim) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= num_rows) return;
  const scalar_t* src = input + row * input_stride;
  scalar_t* dst = out + row * out_stride;
  const int nvec = dim / VEC_SIZE;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const Pack<scalar_t, VEC_SIZE> p =
        *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(src + i * VEC_SIZE);
    for (int k = 0; k < VEC_SIZE; ++k) {
      const float v = to_f<scalar_t>(p.x[k]);
      ss += v * v;
    }
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < dim; i += blockDim.x) {
    const float v = to_f<scalar_t>(src[i]);
    ss += v * v;
  }
  ss = block_sum(ss, smem);
  const float inv = rsqrtf(ss + epsilon);  // true L2: sum, not mean
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const Pack<scalar_t, VEC_SIZE> p =
        *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(src + i * VEC_SIZE);
    Pack<scalar_t, VEC_SIZE> q;
    for (int k = 0; k < VEC_SIZE; ++k) {
      from_f<scalar_t>(q.x[k], to_f<scalar_t>(p.x[k]) * inv * scale);
    }
    *reinterpret_cast<Pack<scalar_t, VEC_SIZE>*>(dst + i * VEC_SIZE) = q;
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < dim; i += blockDim.x) {
    from_f<scalar_t>(dst[i], to_f<scalar_t>(src[i]) * inv * scale);
  }
}

template <typename scalar_t, int VEC_SIZE, ActFn kGateAct, bool kNormBeforeGate>
__global__ void gated_rms_norm_kernel(
    scalar_t* __restrict__ out, const int64_t out_stride, const scalar_t* __restrict__ x,
    const int64_t x_stride, const scalar_t* __restrict__ z, const int64_t z_stride,
    const scalar_t* __restrict__ weight, const int64_t weight_group_stride,
    const float weight_offset, const float epsilon, const int num_rows, const int hidden_size,
    const int group_size) {
  __shared__ float smem[32];
  const int groups = hidden_size / group_size;
  // blockDim.x == 32 means one warp per group; otherwise one CTA per group.
  const bool warp_mode = blockDim.x <= 32;
  const int64_t task = warp_mode ? static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y
                                 : static_cast<int64_t>(blockIdx.x);
  if (task >= static_cast<int64_t>(num_rows) * groups) return;
  const int64_t row = task / groups;
  const int group = static_cast<int>(task - row * groups);
  const scalar_t* xp = x + row * x_stride + static_cast<int64_t>(group) * group_size;
  const scalar_t* zp = z + row * z_stride + static_cast<int64_t>(group) * group_size;
  scalar_t* op = out + row * out_stride + static_cast<int64_t>(group) * group_size;
  const scalar_t* wp =
      (weight != nullptr) ? weight + static_cast<int64_t>(group) * weight_group_stride : nullptr;
  const int nvec = group_size / VEC_SIZE;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const Pack<scalar_t, VEC_SIZE> px =
        *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(xp + i * VEC_SIZE);
    if constexpr (kNormBeforeGate) {
      for (int k = 0; k < VEC_SIZE; ++k) {
        const float v = to_f<scalar_t>(px.x[k]);
        ss += v * v;
      }
    } else {
      const Pack<scalar_t, VEC_SIZE> pz =
          *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(zp + i * VEC_SIZE);
      for (int k = 0; k < VEC_SIZE; ++k) {
        const float v = to_f<scalar_t>(px.x[k]) * gate_value<kGateAct>(to_f<scalar_t>(pz.x[k]));
        ss += v * v;
      }
    }
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < group_size; i += blockDim.x) {
    const float v = to_f<scalar_t>(xp[i]);
    if constexpr (kNormBeforeGate) {
      ss += v * v;
    } else {
      const float g = v * gate_value<kGateAct>(to_f<scalar_t>(zp[i]));
      ss += g * g;
    }
  }
  float total;
  if (warp_mode) {
    total = warp_sum_all(ss);
  } else {
    total = block_sum(ss, smem);
  }
  const float inv = rsqrtf(total / static_cast<float>(group_size) + epsilon);
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const Pack<scalar_t, VEC_SIZE> px =
        *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(xp + i * VEC_SIZE);
    const Pack<scalar_t, VEC_SIZE> pz =
        *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(zp + i * VEC_SIZE);
    Pack<scalar_t, VEC_SIZE> py;
    for (int k = 0; k < VEC_SIZE; ++k) {
      const float g = gate_value<kGateAct>(to_f<scalar_t>(pz.x[k]));
      float v = to_f<scalar_t>(px.x[k]);
      if constexpr (kNormBeforeGate) {
        v *= inv;
        if (wp != nullptr) v *= to_f<scalar_t>(wp[i * VEC_SIZE + k]) + weight_offset;
        v *= g;
      } else {
        v = (v * g) * inv;
        if (wp != nullptr) v *= to_f<scalar_t>(wp[i * VEC_SIZE + k]) + weight_offset;
      }
      from_f<scalar_t>(py.x[k], v);
    }
    *reinterpret_cast<Pack<scalar_t, VEC_SIZE>*>(op + i * VEC_SIZE) = py;
  }
  for (int i = nvec * VEC_SIZE + threadIdx.x; i < group_size; i += blockDim.x) {
    const float g = gate_value<kGateAct>(to_f<scalar_t>(zp[i]));
    float v = to_f<scalar_t>(xp[i]);
    if constexpr (kNormBeforeGate) {
      v *= inv;
      if (wp != nullptr) v *= to_f<scalar_t>(wp[i]) + weight_offset;
      v *= g;
    } else {
      v = (v * g) * inv;
      if (wp != nullptr) v *= to_f<scalar_t>(wp[i]) + weight_offset;
    }
    from_f<scalar_t>(op[i], v);
  }
}

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------
namespace {

template <typename T>
constexpr int vec_width() {
  return 8;  // 16 B for 2-byte types
}
template <>
constexpr int vec_width<float>() {
  return 4;
}

inline int block_for(int hidden, int vec) {
  int n = (hidden + vec - 1) / vec;
  n = ((n + 31) / 32) * 32;
  if (n < 64) n = 64;
  if (n > 512) n = 512;
  return n;
}

/// Elements per thread of one row pass: a 32 B chunk where the row stride keeps
/// every row start 16 B aligned, otherwise one element (the scalar path).
template <typename T>
int pick_vec(int64_t rows, int64_t s2, int64_t s3, int64_t s4) {
  const int v = vec_width<T>();
  if (rows <= 1) return v;
  return ((s2 % v) == 0 && (s3 % v) == 0 && (s4 % v) == 0) ? v : 1;
}

/// Dynamic shared memory for the row stage: 0 when the row does not fit, which
/// is exactly the condition the kernel re-evaluates to pick the direct path.
inline size_t stage_bytes(int hidden, int64_t elem_bytes) {
  return norm_stage_fits(hidden, elem_bytes)
             ? static_cast<size_t>(hidden) * static_cast<size_t>(elem_bytes)
             : 0u;
}

struct RowArgs {
  int num_tokens;
  int hidden;
  int64_t s2, s3, s4, n2, n3;
};

template <typename T, int VEC, bool HasWeight>
Status launch_rms(const Tensor& out, const Tensor& input, const Tensor& weight, int num_groups,
                  int64_t weight_stride, float eps, float w_off, const RowArgs& a,
                  const LaunchContext& ctx) {
  const T* x = reinterpret_cast<const T*>(static_cast<const char*>(input.data) + input.byte_offset);
  T* y = reinterpret_cast<T*>(static_cast<char*>(out.data) + out.byte_offset);
  const T* w = HasWeight ? static_cast<const T*>(weight.data) : nullptr;
  const int64_t out_stride = out.stride[out.ndim - 2];
  // One 16 B pack per thread of the row: measured best at both the 17 MB and
  // the 134 MB case (a 32 B-per-thread block halves the CTA and loses ~4% on
  // small rows). The 32 B chunk loop below still issues two packs per thread.
  const int block = block_for(a.hidden, VEC);
  const size_t smem = stage_bytes(a.hidden, static_cast<int64_t>(sizeof(T)));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
#define KOPS_LAUNCH_RMS(ND)                                                                       \
  rms_norm_kernel<T, VEC, ND, HasWeight><<<a.num_tokens, block, smem, stream>>>(                  \
      y, out_stride, x, a.s2, a.s3, a.s4, a.n2, a.n3, w, weight_stride, num_groups, w_off, eps,   \
      a.num_tokens, a.hidden)
  switch (input.ndim) {
    case 2: KOPS_LAUNCH_RMS(2); break;
    case 3: KOPS_LAUNCH_RMS(3); break;
    case 4: KOPS_LAUNCH_RMS(4); break;
    default: return detail::set_error(Status::kUnsupportedShape, "rms_norm: ndim %d", input.ndim);
  }
#undef KOPS_LAUNCH_RMS
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err),
                                     "rms_norm launch: %s", cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename T>
Status dispatch_rms(const Tensor& out, const Tensor& input, const Tensor& weight, int num_groups,
                    int64_t weight_stride, float eps, float w_off, const RowArgs& a,
                    const LaunchContext& ctx) {
  const int64_t out_stride = out.stride[out.ndim - 2];
  const bool wide = pick_vec<T>(a.num_tokens, a.s2, a.s3, a.s4) == vec_width<T>() &&
                    (a.num_tokens <= 1 || (out_stride % vec_width<T>()) == 0);
  if (wide) {
    if (tensor_present(weight))
      return launch_rms<T, vec_width<T>(), true>(out, input, weight, num_groups, weight_stride, eps,
                                                 w_off, a, ctx);
    return launch_rms<T, vec_width<T>(), false>(out, input, weight, num_groups, weight_stride, eps,
                                                w_off, a, ctx);
  }
  if (tensor_present(weight))
    return launch_rms<T, 1, true>(out, input, weight, num_groups, weight_stride, eps, w_off, a, ctx);
  return launch_rms<T, 1, false>(out, input, weight, num_groups, weight_stride, eps, w_off, a, ctx);
}

// ---- operand views and alias rules ------------------------------------------

bool is_float_dtype(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

int64_t element_bytes_of(DType t) { return (dtype_bits(t) + 7) / 8; }

Status check_epsilon(float eps) {
  if (!std::isfinite(eps) || eps < 0.0f) {
    return detail::set_error(Status::kInvalidArgument, "epsilon must be finite and non-negative");
  }
  return Status::kSuccess;
}

Status check_finite_param(float v, const char* name) {
  if (!std::isfinite(v)) {
    return detail::set_error(Status::kInvalidArgument, "%s must be finite", name);
  }
  return Status::kSuccess;
}

Status check_launch(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s launch: %s",
                                     what, cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

template <typename T>
const T* const_ptr(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

/// [rows, cols] with one row stride: the shape every kernel of this group takes.
/// Leading dims may only be a contiguous stack above a (possibly padded) row.
struct RowView {
  void* data = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t row_stride = 0;  // elements
  int64_t elem_bytes = 0;
  uintptr_t begin = 0;     // first byte of the view (data + byte_offset)
  uintptr_t end = 0;       // one past the last byte the view may touch
};

struct ByteRange {
  uintptr_t begin = 0;
  uintptr_t end = 0;
};

ByteRange range_of(const RowView& v) { return ByteRange{v.begin, v.end}; }

bool ranges_overlap(const ByteRange& a, const ByteRange& b) {
  return !(a.end <= b.begin || b.end <= a.begin);
}

Status require_disjoint(const ByteRange& a, const char* an, const ByteRange& b, const char* bn,
                        const char* what) {
  if (!ranges_overlap(a, b)) return Status::kSuccess;
  return detail::set_error(Status::kUnsupportedLayout, "%s: %s must not overlap %s", what, an, bn);
}

bool same_view(const RowView& a, const RowView& b) {
  return a.begin == b.begin && a.rows == b.rows && a.cols == b.cols &&
         a.row_stride == b.row_stride && a.elem_bytes == b.elem_bytes;
}

/// Overlap is allowed only for the exact in-place view; anything else would let
/// one thread read a value another thread has already overwritten.
Status require_inplace_only(const RowView& a, const char* an, const RowView& b, const char* bn,
                            const char* what) {
  if (!ranges_overlap(range_of(a), range_of(b))) return Status::kSuccess;
  if (same_view(a, b)) return Status::kSuccess;
  return detail::set_error(Status::kUnsupportedLayout,
                           "%s: %s overlaps %s; only an exact in-place view is allowed", what, an,
                           bn);
}

Status make_row_view(const Tensor& t, const char* name, RowView* out) {
  Status s = check_tensor(t, t.dtype, -1, 16, name);
  if (s != Status::kSuccess) return s;
  if (!is_float_dtype(t.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s", name, dtype_name(t.dtype));
  }
  const int32_t nd = t.ndim;
  const int64_t cols = t.shape[nd - 1];
  if (cols <= 0) {
    return detail::set_error(Status::kUnsupportedShape, "%s: last dim %lld", name,
                             static_cast<long long>(cols));
  }
  if (t.stride[nd - 1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: last dim must be contiguous", name);
  }
  for (int32_t d = 0; d + 2 < nd; ++d) {
    if (t.shape[d] > 1 && t.stride[d] != t.stride[d + 1] * t.shape[d + 1]) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "%s: leading dims do not collapse to one row stride", name);
    }
  }
  int64_t rows = 1;
  for (int32_t d = 0; d + 1 < nd; ++d) {
    if (t.shape[d] > 0 && rows > INT64_MAX / t.shape[d]) {
      return detail::set_error(Status::kInvalidArgument, "%s: row count overflow", name);
    }
    rows *= t.shape[d];
  }
  const int64_t stride = (nd > 1) ? t.stride[nd - 2] : 0;
  if (rows > INT32_MAX || cols > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "%s: rows %lld, cols %lld exceed int32",
                             name, static_cast<long long>(rows), static_cast<long long>(cols));
  }
  if (rows > 0 && stride > 0 && rows - 1 > (INT64_MAX - cols) / stride) {
    return detail::set_error(Status::kInvalidArgument, "%s: size overflow", name);
  }
  const int64_t span = (rows > 0) ? (rows - 1) * stride + cols : 0;
  const int64_t elem = element_bytes_of(t.dtype);
  if (span > 0 && span > INT64_MAX / elem) {
    return detail::set_error(Status::kInvalidArgument, "%s: size overflow", name);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(t.data) + static_cast<uintptr_t>(t.byte_offset);
  out->data = reinterpret_cast<void*>(begin);
  out->rows = rows;
  out->cols = cols;
  out->row_stride = stride;
  out->elem_bytes = elem;
  out->begin = begin;
  out->end = begin + static_cast<uintptr_t>(span * elem);
  return Status::kSuccess;
}

/// A 1-D unit-stride operand (weight / bias) of exactly `len` elements.
Status make_unit_vector(const Tensor& t, DType dtype, const char* name, int64_t len,
                        ByteRange* out) {
  Status s = check_tensor(t, dtype, 1, 16, name);
  if (s != Status::kSuccess) return s;
  if (t.shape[0] != len || t.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedShape, "%s: expected [%lld] unit-stride, got [%lld]",
                             name, static_cast<long long>(len),
                             static_cast<long long>(t.shape[0]));
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(t.data) + static_cast<uintptr_t>(t.byte_offset);
  out->begin = begin;
  out->end = begin + static_cast<uintptr_t>(len * element_bytes_of(dtype));
  return Status::kSuccess;
}

bool same_extent(const RowView& a, const RowView& b) {
  return a.rows == b.rows && a.cols == b.cols;
}

// ---- launchers ---------------------------------------------------------------

template <typename T, typename R, int VEC, bool HasWeight>
Status launch_fused_norm_out(const RowView& out, const RowView& rout, const RowView& in,
                             const RowView& rin, const T* weight, float eps, float w_off,
                             float input_scale, const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int block = block_for(static_cast<int>(in.cols), VEC);
  const size_t smem = stage_bytes(static_cast<int>(in.cols), static_cast<int64_t>(sizeof(R)));
  fused_add_rms_norm_out_kernel<T, R, VEC, HasWeight>
      <<<static_cast<unsigned>(in.rows), block, smem, stream>>>(
          static_cast<T*>(out.data), static_cast<R*>(rout.data), static_cast<const T*>(in.data),
          in.row_stride, static_cast<const R*>(rin.data), weight, w_off, input_scale, eps,
          static_cast<int>(in.rows), static_cast<int>(in.cols));
  return check_launch("fused_add_rms_norm_out");
}

template <typename T, typename R, int VEC, bool HasWeight>
Status launch_fused_norm(T* x, const int64_t stride, R* residual, const T* weight, float eps,
                         float w_off, float input_scale, int tokens, int hidden,
                         const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int block = block_for(hidden, VEC);
  const size_t smem = stage_bytes(hidden, static_cast<int64_t>(sizeof(R)));
  fused_add_rms_norm_kernel<T, R, VEC, HasWeight><<<tokens, block, smem, stream>>>(
      x, stride, residual, weight, w_off, input_scale, eps, tokens, hidden);
  return check_launch("fused_add_rms_norm");
}

template <typename T, typename R, bool HasWeight>
Status dispatch_fused_norm(T* x, const int64_t stride, R* residual, const T* weight, float eps,
                           float w_off, float input_scale, int tokens, int hidden,
                           const LaunchContext& ctx) {
  if (pick_vec<T>(tokens, stride, 0, 0) == vec_width<T>()) {
    return launch_fused_norm<T, R, vec_width<T>(), HasWeight>(x, stride, residual, weight, eps, w_off,
                                                              input_scale, tokens, hidden, ctx);
  }
  return launch_fused_norm<T, R, 1, HasWeight>(x, stride, residual, weight, eps, w_off, input_scale,
                                               tokens, hidden, ctx);
}

template <typename T, typename R, bool HasWeight>
Status dispatch_fused_norm_out(const RowView& out, const RowView& rout, const RowView& in,
                               const RowView& rin, const T* weight, float eps, float w_off,
                               float input_scale, const LaunchContext& ctx) {
  if (pick_vec<T>(in.rows, in.row_stride, 0, 0) == vec_width<T>()) {
    return launch_fused_norm_out<T, R, vec_width<T>(), HasWeight>(out, rout, in, rin, weight, eps,
                                                                  w_off, input_scale, ctx);
  }
  return launch_fused_norm_out<T, R, 1, HasWeight>(out, rout, in, rin, weight, eps, w_off,
                                                   input_scale, ctx);
}

template <typename T, int VEC, bool HasWeight, bool HasBias>
Status launch_layer_norm(const RowView& out, const RowView& in, const T* w, const T* b, float eps,
                         float w_off, const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int block = block_for(static_cast<int>(in.cols), VEC);
  layer_norm_kernel<T, VEC, HasWeight, HasBias>
      <<<static_cast<unsigned>(in.rows), block, 0, stream>>>(
          static_cast<T*>(out.data), out.row_stride, static_cast<const T*>(in.data), in.row_stride,
          w, b, w_off, eps, static_cast<int>(in.rows), static_cast<int>(in.cols));
  return check_launch("layer_norm");
}

template <typename T, typename R, int VEC, bool HasWeight, bool HasBias>
Status launch_fused_add_layer_norm(const RowView& inout, const RowView& residual, const T* w,
                                   const T* b, float eps, float w_off, float input_scale,
                                   const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int block = block_for(static_cast<int>(inout.cols), VEC);
  fused_add_layer_norm_kernel<T, R, VEC, HasWeight, HasBias>
      <<<static_cast<unsigned>(inout.rows), block, 0, stream>>>(
          static_cast<T*>(inout.data), inout.row_stride, static_cast<R*>(residual.data), w, b, w_off,
          input_scale, eps, static_cast<int>(inout.rows), static_cast<int>(inout.cols));
  return check_launch("fused_add_layer_norm");
}

template <typename T, int VEC>
Status launch_l2(const RowView& out, const RowView& in, float scale, float eps,
                 const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int block = block_for(static_cast<int>(in.cols), VEC);
  l2_norm_kernel<T, VEC><<<static_cast<unsigned>(in.rows), block, 0, stream>>>(
      static_cast<T*>(out.data), out.row_stride, static_cast<const T*>(in.data), in.row_stride,
      scale, eps, static_cast<int>(in.rows), static_cast<int>(in.cols));
  return check_launch("l2_norm");
}

struct QkWeight {
  const void* ptr = nullptr;
  int64_t head_stride = 0;
};

template <typename T, int HEAD_DIM, NormType kN, bool kFW>
Status launch_qk(const QkWeight& qw, const QkWeight& kw, void* qkv, int64_t token_stride,
                 int64_t q_offset, int64_t k_offset, const float* partials, int tokens, int hq,
                 int hkv, float eps, float w_off, const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  T* base = static_cast<T*>(qkv);
  const T* q_weight = static_cast<const T*>(qw.ptr);
  const T* k_weight = static_cast<const T*>(kw.ptr);
  if constexpr (kFW) {
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    const int64_t wq = static_cast<int64_t>(hq) * HEAD_DIM;
    const int64_t wk = static_cast<int64_t>(hkv) * HEAD_DIM;
    const int width = static_cast<int>(wq > wk ? wq : wk);
    if (width <= 0) return Status::kSuccess;
    const int block = block_for(width, VEC);
    const int kinds = (hq > 0 ? 1 : 0) + (hkv > 0 ? 1 : 0);
    qk_norm_kernel<T, HEAD_DIM, kN, true><<<dim3(static_cast<unsigned>(tokens),
                                                 static_cast<unsigned>(kinds)),
                                            block, 0, stream>>>(
        base, token_stride, q_offset, k_offset, q_weight, k_weight, qw.head_stride, kw.head_stride,
        partials, w_off, eps, tokens, hq, hkv);
  } else {
    const int64_t heads = static_cast<int64_t>(hq) + hkv;
    const int64_t tasks = static_cast<int64_t>(tokens) * heads;
    if (tasks == 0) return Status::kSuccess;
    if (tasks > (static_cast<int64_t>(1) << 33)) {
      return detail::set_error(Status::kUnsupportedShape, "qk_norm: too many head rows");
    }
    const unsigned grid = static_cast<unsigned>((tasks + 3) / 4);
    qk_norm_kernel<T, HEAD_DIM, kN, false><<<grid, dim3(32, 4), 0, stream>>>(
        base, token_stride, q_offset, k_offset, q_weight, k_weight, qw.head_stride, kw.head_stride,
        partials, w_off, eps, tokens, hq, hkv);
  }
  return check_launch("qk_norm");
}

template <typename T, int VEC, ActFn kAct, bool kBefore>
Status launch_gated(const RowView& out, const RowView& xv, const RowView& zv, const T* w,
                    int64_t w_stride, int group, float eps, float w_off,
                    const LaunchContext& ctx) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int64_t groups = xv.cols / group;
  const int64_t tasks = xv.rows * groups;
  if (tasks == 0) return Status::kSuccess;
  dim3 grid;
  dim3 block;
  if (group <= 256) {
    grid = dim3(static_cast<unsigned>((tasks + 3) / 4));
    block = dim3(32, 4);
  } else {
    grid = dim3(static_cast<unsigned>(tasks));
    block = dim3(static_cast<unsigned>(block_for(group, VEC)));
  }
  gated_rms_norm_kernel<T, VEC, kAct, kBefore><<<grid, block, 0, stream>>>(
      static_cast<T*>(out.data), out.row_stride, static_cast<const T*>(xv.data), xv.row_stride,
      static_cast<const T*>(zv.data), zv.row_stride, w, w_stride, w_off, eps,
      static_cast<int>(xv.rows), static_cast<int>(xv.cols), group);
  return check_launch("gated_rms_norm");
}

template <typename T>
Status dispatch_gated(const RowView& out, const RowView& xv, const RowView& zv, const T* w,
                      int64_t w_stride, int group, ActFn act, bool before, float eps, float w_off,
                      const LaunchContext& ctx) {
  constexpr int VEC = vec_width<T>();
  const bool aligned_group = (group % VEC) == 0;
#define KOPS_LAUNCH_GATED(ACT, BEFORE)                                                             \
  do {                                                                                             \
    if (aligned_group) {                                                                           \
      return launch_gated<T, VEC, ACT, BEFORE>(out, xv, zv, w, w_stride, group, eps, w_off, ctx);   \
    }                                                                                              \
    return launch_gated<T, 1, ACT, BEFORE>(out, xv, zv, w, w_stride, group, eps, w_off, ctx);       \
  } while (0)
  switch (act) {
    case ActFn::kIdentity:
      if (before) KOPS_LAUNCH_GATED(ActFn::kIdentity, true);
      KOPS_LAUNCH_GATED(ActFn::kIdentity, false);
    case ActFn::kSigmoid:
      if (before) KOPS_LAUNCH_GATED(ActFn::kSigmoid, true);
      KOPS_LAUNCH_GATED(ActFn::kSigmoid, false);
    case ActFn::kSiLU:
      if (before) KOPS_LAUNCH_GATED(ActFn::kSiLU, true);
      KOPS_LAUNCH_GATED(ActFn::kSiLU, false);
    default:
      return detail::set_error(Status::kUnsupportedShape,
                               "gated_rms_norm: gate_act %d not in {identity, sigmoid, silu}",
                               static_cast<int>(act));
  }
#undef KOPS_LAUNCH_GATED
}

// ---- QK plan and norm/head-dim dispatch --------------------------------------

struct QkPlan {
  void* q = nullptr;
  void* k = nullptr;  // separate k tensor; null for a packed qkv
  QkWeight qw{}, kw{};
  const float* partials = nullptr;
  int64_t q_stride = 0, k_stride = 0, k_offset = 0;
  int32_t tokens = 0, hq = 0, hkv = 0, head_dim = 0;
  NormType norm = NormType::kRMSNorm;
  bool full_width = false, split = false;
  float eps = 0.0f, w_off = 0.0f;
};

template <typename T, int HEAD_DIM, NormType kN, bool kFW>
Status run_qk(const QkPlan& p, const LaunchContext& ctx) {
  if (p.split) {
    // One launch per kind; the kernel's k branch is never taken when hkv == 0.
    Status s = launch_qk<T, HEAD_DIM, kN, kFW>(p.qw, QkWeight{}, p.q, p.q_stride, 0, 0, p.partials,
                                               p.tokens, p.hq, 0, p.eps, p.w_off, ctx);
    if (s != Status::kSuccess) return s;
    return launch_qk<T, HEAD_DIM, kN, kFW>(QkWeight{}, p.kw, p.k, p.k_stride, 0, 0, p.partials,
                                           p.tokens, 0, p.hkv, p.eps, p.w_off, ctx);
  }
  return launch_qk<T, HEAD_DIM, kN, kFW>(p.qw, p.kw, p.q, p.q_stride, 0, p.k_offset, p.partials,
                                         p.tokens, p.hq, p.hkv, p.eps, p.w_off, ctx);
}

template <typename T, int HEAD_DIM>
Status dispatch_qk_dim(const QkPlan& p, const LaunchContext& ctx) {
  if (p.full_width) {
    switch (p.norm) {
      case NormType::kRMSNorm: return run_qk<T, HEAD_DIM, NormType::kRMSNorm, true>(p, ctx);
      case NormType::kLayerNorm: return run_qk<T, HEAD_DIM, NormType::kLayerNorm, true>(p, ctx);
      default: return run_qk<T, HEAD_DIM, NormType::kL2Norm, true>(p, ctx);
    }
  }
  switch (p.norm) {
    case NormType::kRMSNorm: return run_qk<T, HEAD_DIM, NormType::kRMSNorm, false>(p, ctx);
    case NormType::kLayerNorm: return run_qk<T, HEAD_DIM, NormType::kLayerNorm, false>(p, ctx);
    default: return run_qk<T, HEAD_DIM, NormType::kL2Norm, false>(p, ctx);
  }
}

template <typename T>
Status dispatch_qk(const QkPlan& p, const LaunchContext& ctx) {
  switch (p.head_dim) {
    case 64: return dispatch_qk_dim<T, 64>(p, ctx);
    case 128: return dispatch_qk_dim<T, 128>(p, ctx);
    case 256: return dispatch_qk_dim<T, 256>(p, ctx);
    default:
      return detail::set_error(Status::kUnsupportedShape, "qk_norm: head_dim %d", p.head_dim);
  }
}

}  // namespace

Status rms_norm(const Tensor& out, const Tensor& input, const Tensor& weight, float epsilon,
                float weight_offset, const LaunchContext& ctx) {
  Status s = check_tensor(input, input.dtype, -1, 16, "rms_norm.input");
  if (s != Status::kSuccess) return s;
  if (input.ndim < 2 || input.ndim > 4) {
    return detail::set_error(Status::kUnsupportedShape, "rms_norm: ndim %d not in {2,3,4}", input.ndim);
  }
  s = check_tensor(out, input.dtype, input.ndim, 16, "rms_norm.out");
  if (s != Status::kSuccess) return s;
  for (int d = 0; d + 1 < input.ndim; ++d) {
    if (out.shape[d] != input.shape[d]) {
      return detail::set_error(Status::kInvalidArgument, "rms_norm: shape mismatch at dim %d", d);
    }
  }
  if (out.shape[input.ndim - 1] != input.shape[input.ndim - 1]) {
    return detail::set_error(Status::kInvalidArgument, "rms_norm: hidden size mismatch");
  }
  const int hidden = static_cast<int>(input.shape[input.ndim - 1]);
  if (hidden <= 0) return detail::set_error(Status::kUnsupportedShape, "rms_norm: hidden <= 0");
  const int64_t numel = tensor_numel(input);
  const int num_tokens = static_cast<int>(numel / hidden);
  int num_groups = 1;
  int64_t weight_stride = 0;
  if (tensor_present(weight)) {
    s = check_tensor(weight, input.dtype, -1, 16, "rms_norm.weight");
    if (s != Status::kSuccess) return s;
    if (weight.ndim == 2) {
      num_groups = static_cast<int>(weight.shape[0]);
      weight_stride = weight.stride[0];
    } else if (weight.ndim != 1) {
      return detail::set_error(Status::kUnsupportedShape, "rms_norm: weight ndim %d", weight.ndim);
    }
    if (weight.shape[weight.ndim - 1] != hidden) {
      return detail::set_error(Status::kInvalidArgument, "rms_norm: weight width mismatch");
    }
  }
  RowArgs a{num_tokens, hidden, input.stride[input.ndim - 2], 0, 0, 0, 0};
  if (input.ndim >= 3) a.s3 = input.stride[input.ndim - 3];
  if (input.ndim >= 4) a.s4 = input.stride[input.ndim - 4];
  if (input.ndim >= 3) a.n2 = input.shape[input.ndim - 2];
  if (input.ndim >= 4) a.n3 = input.shape[input.ndim - 3];

  switch (input.dtype) {
    case DType::kBFloat16:
      return dispatch_rms<__nv_bfloat16>(out, input, weight, num_groups, weight_stride, epsilon,
                                         weight_offset, a, ctx);
    case DType::kFloat16:
      return dispatch_rms<__half>(out, input, weight, num_groups, weight_stride, epsilon,
                                  weight_offset, a, ctx);
    case DType::kFloat32:
      return dispatch_rms<float>(out, input, weight, num_groups, weight_stride, epsilon,
                                 weight_offset, a, ctx);
    default:
      return detail::set_error(Status::kUnsupportedDtype, "rms_norm: dtype %s",
                               dtype_name(input.dtype));
  }
}

Status fused_add_rms_norm(const Tensor& input_inout, const Tensor& residual_inout,
                          const Tensor& weight, float epsilon, float weight_offset,
                          float input_scale, const LaunchContext& ctx) {
  Status s = check_tensor(input_inout, input_inout.dtype, 2, 16, "fused_add_rms_norm.input");
  if (s != Status::kSuccess) return s;
  s = check_tensor(residual_inout, residual_inout.dtype, 2, 16, "fused_add_rms_norm.residual");
  if (s != Status::kSuccess) return s;
  if (input_inout.shape[0] != residual_inout.shape[0] ||
      input_inout.shape[1] != residual_inout.shape[1]) {
    return detail::set_error(Status::kInvalidArgument, "fused_add_rms_norm: shape mismatch");
  }
  if (residual_inout.stride[0] != input_inout.stride[0]) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "fused_add_rms_norm: residual row stride %lld != input stride %lld",
                             static_cast<long long>(residual_inout.stride[0]),
                             static_cast<long long>(input_inout.stride[0]));
  }
  const int hidden = static_cast<int>(input_inout.shape[1]);
  const int tokens = static_cast<int>(input_inout.shape[0]);
  const bool has_w = tensor_present(weight);
  if (has_w) {
    s = check_tensor(weight, input_inout.dtype, 1, 16, "fused_add_rms_norm.weight");
    if (s != Status::kSuccess) return s;
  }
#define KOPS_LAUNCH_ADD(T, R, HW)                                                                 \
  return dispatch_fused_norm<T, R, HW>(static_cast<T*>(input_inout.data), input_inout.stride[0],  \
                                       static_cast<R*>(residual_inout.data),                      \
                                       has_w ? static_cast<const T*>(weight.data) : nullptr,      \
                                       epsilon, weight_offset, input_scale, tokens, hidden, ctx)

  if (input_inout.dtype == DType::kBFloat16) {
    if (residual_inout.dtype == DType::kFloat32) {
      if (has_w) KOPS_LAUNCH_ADD(__nv_bfloat16, float, true);
      else KOPS_LAUNCH_ADD(__nv_bfloat16, float, false);
    } else if (residual_inout.dtype == DType::kBFloat16) {
      if (has_w) KOPS_LAUNCH_ADD(__nv_bfloat16, __nv_bfloat16, true);
      else KOPS_LAUNCH_ADD(__nv_bfloat16, __nv_bfloat16, false);
    } else {
      return detail::set_error(Status::kUnsupportedDtype, "fused_add_rms_norm: residual %s",
                               dtype_name(residual_inout.dtype));
    }
  } else if (input_inout.dtype == DType::kFloat16) {
    if (residual_inout.dtype == DType::kFloat32) {
      if (has_w) KOPS_LAUNCH_ADD(__half, float, true);
      else KOPS_LAUNCH_ADD(__half, float, false);
    } else if (residual_inout.dtype == DType::kFloat16) {
      if (has_w) KOPS_LAUNCH_ADD(__half, __half, true);
      else KOPS_LAUNCH_ADD(__half, __half, false);
    } else {
      return detail::set_error(Status::kUnsupportedDtype, "fused_add_rms_norm: residual %s",
                               dtype_name(residual_inout.dtype));
    }
  } else if (input_inout.dtype == DType::kFloat32) {
    if (has_w) KOPS_LAUNCH_ADD(float, float, true);
    else KOPS_LAUNCH_ADD(float, float, false);
  } else {
    return detail::set_error(Status::kUnsupportedDtype, "fused_add_rms_norm: input %s",
                             dtype_name(input_inout.dtype));
  }
#undef KOPS_LAUNCH_ADD
}

Status fused_add_rms_norm_out(const Tensor& out, const Tensor& residual_out, const Tensor& input,
                              const Tensor& residual_in, const Tensor& weight, float epsilon,
                              float weight_offset, float input_scale, const LaunchContext& ctx) {
  Status s = check_epsilon(epsilon);
  if (s != Status::kSuccess) return s;
  s = check_finite_param(weight_offset, "fused_add_rms_norm_out: weight_offset");
  if (s != Status::kSuccess) return s;
  s = check_finite_param(input_scale, "fused_add_rms_norm_out: input_scale");
  if (s != Status::kSuccess) return s;
  if (input.ndim != 2 || out.ndim != 2 || residual_in.ndim != 2 || residual_out.ndim != 2) {
    return detail::set_error(Status::kUnsupportedShape,
                             "fused_add_rms_norm_out: operands must be 2-D [tokens, hidden]");
  }
  RowView vo{}, vro{}, vi{}, vri{};
  s = make_row_view(input, "fused_add_rms_norm_out.input", &vi);
  if (s != Status::kSuccess) return s;
  s = make_row_view(out, "fused_add_rms_norm_out.out", &vo);
  if (s != Status::kSuccess) return s;
  s = make_row_view(residual_in, "fused_add_rms_norm_out.residual_in", &vri);
  if (s != Status::kSuccess) return s;
  s = make_row_view(residual_out, "fused_add_rms_norm_out.residual_out", &vro);
  if (s != Status::kSuccess) return s;
  if (out.dtype != input.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "fused_add_rms_norm_out: out %s, input %s",
                             dtype_name(out.dtype), dtype_name(input.dtype));
  }
  if (residual_in.dtype != residual_out.dtype ||
      (residual_in.dtype != input.dtype && residual_in.dtype != DType::kFloat32)) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "fused_add_rms_norm_out: residual %s/%s, input %s",
                             dtype_name(residual_in.dtype), dtype_name(residual_out.dtype),
                             dtype_name(input.dtype));
  }
  if (!same_extent(vo, vi) || !same_extent(vri, vi) || !same_extent(vro, vi)) {
    return detail::set_error(Status::kInvalidArgument, "fused_add_rms_norm_out: shape mismatch");
  }
  if (vri.row_stride != vi.row_stride || vro.row_stride != vi.row_stride) {
    return detail::set_error(
        Status::kUnsupportedLayout,
        "fused_add_rms_norm_out: residual row stride %lld/%lld != input stride %lld",
        static_cast<long long>(vri.row_stride), static_cast<long long>(vro.row_stride),
        static_cast<long long>(vi.row_stride));
  }
  if (vo.row_stride != vo.cols) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "fused_add_rms_norm_out: out rows must be contiguous (stride %lld, "
                             "hidden %lld)",
                             static_cast<long long>(vo.row_stride),
                             static_cast<long long>(vo.cols));
  }
  // out may alias input and residual_out may alias residual_in (exact in-place
  // views); every other overlap would make the second pass read stale memory.
  s = require_inplace_only(vo, "out", vi, "input", "fused_add_rms_norm_out");
  if (s != Status::kSuccess) return s;
  s = require_inplace_only(vro, "residual_out", vri, "residual_in", "fused_add_rms_norm_out");
  if (s != Status::kSuccess) return s;
  s = require_disjoint(range_of(vo), "out", range_of(vro), "residual_out",
                       "fused_add_rms_norm_out");
  if (s != Status::kSuccess) return s;
  s = require_disjoint(range_of(vo), "out", range_of(vri), "residual_in",
                       "fused_add_rms_norm_out");
  if (s != Status::kSuccess) return s;
  s = require_disjoint(range_of(vro), "residual_out", range_of(vi), "input",
                       "fused_add_rms_norm_out");
  if (s != Status::kSuccess) return s;
  s = require_disjoint(range_of(vri), "residual_in", range_of(vi), "input",
                       "fused_add_rms_norm_out");
  if (s != Status::kSuccess) return s;
  const bool has_w = tensor_present(weight);
  if (has_w) {
    ByteRange wr{};
    s = make_unit_vector(weight, input.dtype, "fused_add_rms_norm_out.weight", vi.cols, &wr);
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vo), "out", wr, "weight", "fused_add_rms_norm_out");
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vro), "residual_out", wr, "weight",
                         "fused_add_rms_norm_out");
    if (s != Status::kSuccess) return s;
  }
  if (vi.rows == 0) return Status::kSuccess;

#define KOPS_LAUNCH_FUSED_OUT(T, R, HW)                                                            \
  return dispatch_fused_norm_out<T, R, HW>(                                                        \
      vo, vro, vi, vri, has_w ? const_ptr<T>(weight) : nullptr, epsilon, weight_offset,            \
      input_scale, ctx)
  if (input.dtype == DType::kBFloat16) {
    if (residual_in.dtype == DType::kFloat32) {
      if (has_w) { KOPS_LAUNCH_FUSED_OUT(__nv_bfloat16, float, true); }
      KOPS_LAUNCH_FUSED_OUT(__nv_bfloat16, float, false);
    }
    if (has_w) { KOPS_LAUNCH_FUSED_OUT(__nv_bfloat16, __nv_bfloat16, true); }
    KOPS_LAUNCH_FUSED_OUT(__nv_bfloat16, __nv_bfloat16, false);
  }
  if (input.dtype == DType::kFloat16) {
    if (residual_in.dtype == DType::kFloat32) {
      if (has_w) { KOPS_LAUNCH_FUSED_OUT(__half, float, true); }
      KOPS_LAUNCH_FUSED_OUT(__half, float, false);
    }
    if (has_w) { KOPS_LAUNCH_FUSED_OUT(__half, __half, true); }
    KOPS_LAUNCH_FUSED_OUT(__half, __half, false);
  }
  if (has_w) { KOPS_LAUNCH_FUSED_OUT(float, float, true); }
  KOPS_LAUNCH_FUSED_OUT(float, float, false);
#undef KOPS_LAUNCH_FUSED_OUT
}

Status layer_norm(const Tensor& out, const Tensor& input, const Tensor& weight, const Tensor& bias,
                  float epsilon, float weight_offset, const LaunchContext& ctx) {
  Status s = check_epsilon(epsilon);
  if (s != Status::kSuccess) return s;
  s = check_finite_param(weight_offset, "layer_norm: weight_offset");
  if (s != Status::kSuccess) return s;
  RowView vo{}, vi{};
  s = make_row_view(input, "layer_norm.input", &vi);
  if (s != Status::kSuccess) return s;
  if (out.dtype != input.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "layer_norm: out %s, input %s",
                             dtype_name(out.dtype), dtype_name(input.dtype));
  }
  s = make_row_view(out, "layer_norm.out", &vo);
  if (s != Status::kSuccess) return s;
  if (!same_extent(vo, vi)) {
    return detail::set_error(Status::kInvalidArgument, "layer_norm: shape mismatch");
  }
  s = require_inplace_only(vo, "out", vi, "input", "layer_norm");
  if (s != Status::kSuccess) return s;
  const bool has_w = tensor_present(weight);
  const bool has_b = tensor_present(bias);
  if (has_w) {
    ByteRange wr{};
    s = make_unit_vector(weight, input.dtype, "layer_norm.weight", vi.cols, &wr);
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vo), "out", wr, "weight", "layer_norm");
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vi), "input", wr, "weight", "layer_norm");
    if (s != Status::kSuccess) return s;
  }
  if (has_b) {
    ByteRange br{};
    s = make_unit_vector(bias, input.dtype, "layer_norm.bias", vi.cols, &br);
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vo), "out", br, "bias", "layer_norm");
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vi), "input", br, "bias", "layer_norm");
    if (s != Status::kSuccess) return s;
  }
  if (vi.rows == 0) return Status::kSuccess;

#define KOPS_LAUNCH_LN(T, HW, HB)                                                                  \
  return launch_layer_norm<T, vec_width<T>(), HW, HB>(                                             \
      vo, vi, has_w ? const_ptr<T>(weight) : nullptr, has_b ? const_ptr<T>(bias) : nullptr, epsilon, \
      weight_offset, ctx)
  if (input.dtype == DType::kBFloat16) {
    if (has_w) {
      if (has_b) { KOPS_LAUNCH_LN(__nv_bfloat16, true, true); }
      KOPS_LAUNCH_LN(__nv_bfloat16, true, false);
    }
    if (has_b) { KOPS_LAUNCH_LN(__nv_bfloat16, false, true); }
    KOPS_LAUNCH_LN(__nv_bfloat16, false, false);
  }
  if (input.dtype == DType::kFloat16) {
    if (has_w) {
      if (has_b) { KOPS_LAUNCH_LN(__half, true, true); }
      KOPS_LAUNCH_LN(__half, true, false);
    }
    if (has_b) { KOPS_LAUNCH_LN(__half, false, true); }
    KOPS_LAUNCH_LN(__half, false, false);
  }
  if (has_w) {
    if (has_b) { KOPS_LAUNCH_LN(float, true, true); }
    KOPS_LAUNCH_LN(float, true, false);
  }
  if (has_b) { KOPS_LAUNCH_LN(float, false, true); }
  KOPS_LAUNCH_LN(float, false, false);
#undef KOPS_LAUNCH_LN
}

Status fused_add_layer_norm(const Tensor& input_inout, const Tensor& residual_inout,
                            const Tensor& weight, const Tensor& bias, float epsilon,
                            float weight_offset, float input_scale, const LaunchContext& ctx) {
  Status s = check_epsilon(epsilon);
  if (s != Status::kSuccess) return s;
  s = check_finite_param(weight_offset, "fused_add_layer_norm: weight_offset");
  if (s != Status::kSuccess) return s;
  s = check_finite_param(input_scale, "fused_add_layer_norm: input_scale");
  if (s != Status::kSuccess) return s;
  if (input_inout.ndim != 2 || residual_inout.ndim != 2) {
    return detail::set_error(Status::kUnsupportedShape,
                             "fused_add_layer_norm: operands must be 2-D [tokens, hidden]");
  }
  RowView vi{}, vri{};
  s = make_row_view(input_inout, "fused_add_layer_norm.input", &vi);
  if (s != Status::kSuccess) return s;
  s = make_row_view(residual_inout, "fused_add_layer_norm.residual", &vri);
  if (s != Status::kSuccess) return s;
  if (residual_inout.dtype != input_inout.dtype &&
      residual_inout.dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "fused_add_layer_norm: residual %s, input %s",
                             dtype_name(residual_inout.dtype), dtype_name(input_inout.dtype));
  }
  if (!same_extent(vi, vri)) {
    return detail::set_error(Status::kInvalidArgument, "fused_add_layer_norm: shape mismatch");
  }
  if (vri.row_stride != vi.row_stride) {
    return detail::set_error(
        Status::kUnsupportedLayout,
        "fused_add_layer_norm: residual row stride %lld != input stride %lld",
        static_cast<long long>(vri.row_stride), static_cast<long long>(vi.row_stride));
  }
  s = require_disjoint(range_of(vi), "input", range_of(vri), "residual",
                       "fused_add_layer_norm");
  if (s != Status::kSuccess) return s;
  const bool has_w = tensor_present(weight);
  const bool has_b = tensor_present(bias);
  if (has_w) {
    ByteRange wr{};
    s = make_unit_vector(weight, input_inout.dtype, "fused_add_layer_norm.weight", vi.cols, &wr);
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vi), "input", wr, "weight", "fused_add_layer_norm");
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vri), "residual", wr, "weight", "fused_add_layer_norm");
    if (s != Status::kSuccess) return s;
  }
  if (has_b) {
    ByteRange br{};
    s = make_unit_vector(bias, input_inout.dtype, "fused_add_layer_norm.bias", vi.cols, &br);
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vi), "input", br, "bias", "fused_add_layer_norm");
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vri), "residual", br, "bias", "fused_add_layer_norm");
    if (s != Status::kSuccess) return s;
  }
  if (vi.rows == 0) return Status::kSuccess;

#define KOPS_LAUNCH_FUSED_LN(T, R, HW, HB)                                                         \
  return launch_fused_add_layer_norm<T, R, vec_width<T>(), HW, HB>(                                \
      vi, vri, has_w ? const_ptr<T>(weight) : nullptr, has_b ? const_ptr<T>(bias) : nullptr,        \
      epsilon, weight_offset, input_scale, ctx)
  if (input_inout.dtype == DType::kBFloat16) {
    if (residual_inout.dtype == DType::kFloat32) {
      if (has_w) {
        if (has_b) { KOPS_LAUNCH_FUSED_LN(__nv_bfloat16, float, true, true); }
        KOPS_LAUNCH_FUSED_LN(__nv_bfloat16, float, true, false);
      }
      if (has_b) { KOPS_LAUNCH_FUSED_LN(__nv_bfloat16, float, false, true); }
      KOPS_LAUNCH_FUSED_LN(__nv_bfloat16, float, false, false);
    }
    if (has_w) {
      if (has_b) { KOPS_LAUNCH_FUSED_LN(__nv_bfloat16, __nv_bfloat16, true, true); }
      KOPS_LAUNCH_FUSED_LN(__nv_bfloat16, __nv_bfloat16, true, false);
    }
    if (has_b) { KOPS_LAUNCH_FUSED_LN(__nv_bfloat16, __nv_bfloat16, false, true); }
    KOPS_LAUNCH_FUSED_LN(__nv_bfloat16, __nv_bfloat16, false, false);
  }
  if (input_inout.dtype == DType::kFloat16) {
    if (residual_inout.dtype == DType::kFloat32) {
      if (has_w) {
        if (has_b) { KOPS_LAUNCH_FUSED_LN(__half, float, true, true); }
        KOPS_LAUNCH_FUSED_LN(__half, float, true, false);
      }
      if (has_b) { KOPS_LAUNCH_FUSED_LN(__half, float, false, true); }
      KOPS_LAUNCH_FUSED_LN(__half, float, false, false);
    }
    if (has_w) {
      if (has_b) { KOPS_LAUNCH_FUSED_LN(__half, __half, true, true); }
      KOPS_LAUNCH_FUSED_LN(__half, __half, true, false);
    }
    if (has_b) { KOPS_LAUNCH_FUSED_LN(__half, __half, false, true); }
    KOPS_LAUNCH_FUSED_LN(__half, __half, false, false);
  }
  if (has_w) {
    if (has_b) { KOPS_LAUNCH_FUSED_LN(float, float, true, true); }
    KOPS_LAUNCH_FUSED_LN(float, float, true, false);
  }
  if (has_b) { KOPS_LAUNCH_FUSED_LN(float, float, false, true); }
  KOPS_LAUNCH_FUSED_LN(float, float, false, false);
#undef KOPS_LAUNCH_FUSED_LN
}

Status qk_norm(const Tensor& qkv_or_q, const Tensor& k, const Tensor& q_weight,
               const Tensor& k_weight, int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
               NormType norm, bool full_width, const Tensor& full_width_sq_partials, float epsilon,
               float weight_offset, const LaunchContext& ctx) {
  if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
    return detail::set_error(Status::kUnsupportedShape,
                             "qk_norm: head_dim %d not in {64,128,256}", head_dim);
  }
  if (norm != NormType::kRMSNorm && norm != NormType::kLayerNorm && norm != NormType::kL2Norm) {
    return detail::set_error(Status::kUnsupportedShape,
                             "qk_norm: norm %d not in {rms, layer, l2}", static_cast<int>(norm));
  }
  if (num_q_heads <= 0 || num_kv_heads <= 0) {
    return detail::set_error(Status::kUnsupportedShape, "qk_norm: q heads %d, kv heads %d",
                             num_q_heads, num_kv_heads);
  }
  Status s = check_epsilon(epsilon);
  if (s != Status::kSuccess) return s;
  s = check_finite_param(weight_offset, "qk_norm: weight_offset");
  if (s != Status::kSuccess) return s;
  const Tensor& q = qkv_or_q;
  if (!is_float_dtype(q.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "qk_norm: dtype %s", dtype_name(q.dtype));
  }
  if (q.ndim != 2 && q.ndim != 3) {
    return detail::set_error(Status::kUnsupportedShape, "qk_norm: qkv_or_q ndim %d", q.ndim);
  }
  const int64_t q_width = static_cast<int64_t>(num_q_heads) * head_dim;
  const int64_t k_width = static_cast<int64_t>(num_kv_heads) * head_dim;
  const int64_t packed_heads =
      static_cast<int64_t>(num_q_heads) + 2 * static_cast<int64_t>(num_kv_heads);
  if (packed_heads * head_dim > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "qk_norm: packed qkv width too large");
  }
  const bool split = tensor_present(k);
  const auto check_layout = [&](const Tensor& t, int64_t width, int32_t heads,
                                const char* name) -> Status {
    Status st = check_tensor(t, q.dtype, -1, 16, name);
    if (st != Status::kSuccess) return st;
    if (t.ndim != q.ndim) {
      return detail::set_error(Status::kInvalidArgument, "%s: ndim %d, expected %d", name, t.ndim,
                               q.ndim);
    }
    if (t.shape[0] != q.shape[0]) {
      return detail::set_error(Status::kInvalidArgument, "%s: %lld rows, expected %lld", name,
                               static_cast<long long>(t.shape[0]),
                               static_cast<long long>(q.shape[0]));
    }
    if (t.ndim == 2) {
      if (t.shape[1] != width || t.stride[1] != 1) {
        return detail::set_error(Status::kInvalidArgument,
                                 "%s: expected [tokens, %lld] with unit column stride", name,
                                 static_cast<long long>(width));
      }
    } else if (t.shape[1] != heads || t.shape[2] != head_dim || t.stride[1] != head_dim ||
               t.stride[2] != 1) {
      return detail::set_error(Status::kInvalidArgument, "%s: expected [tokens, %d, %d]", name,
                               heads, head_dim);
    }
    if (t.stride[0] < 0) {
      return detail::set_error(Status::kUnsupportedLayout, "%s: negative token stride", name);
    }
    return Status::kSuccess;
  };
  s = check_layout(q, split ? q_width : packed_heads * head_dim,
                   split ? num_q_heads : static_cast<int32_t>(packed_heads), "qk_norm.qkv_or_q");
  if (s != Status::kSuccess) return s;
  const int64_t q_stride = q.stride[0];
  int64_t k_stride = q_stride;
  if (split) {
    s = check_layout(k, k_width, num_kv_heads, "qk_norm.k");
    if (s != Status::kSuccess) return s;
    k_stride = k.stride[0];
  }
  if (q.shape[0] > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "qk_norm: %lld tokens exceed int32",
                             static_cast<long long>(q.shape[0]));
  }
  const auto span = [](const Tensor& t, int64_t cols, int64_t stride) -> ByteRange {
    const uintptr_t b = reinterpret_cast<uintptr_t>(t.data) + static_cast<uintptr_t>(t.byte_offset);
    const int64_t e =
        t.shape[0] > 0 ? (t.shape[0] - 1) * stride + cols : 0;
    return ByteRange{b, b + static_cast<uintptr_t>(e * element_bytes_of(t.dtype))};
  };
  const ByteRange q_range = span(q, split ? q_width : packed_heads * head_dim, q_stride);
  const ByteRange k_range = split ? span(k, k_width, k_stride) : q_range;
  const auto weight_span = [](const Tensor& w) -> ByteRange {
    const uintptr_t b = reinterpret_cast<uintptr_t>(w.data) + static_cast<uintptr_t>(w.byte_offset);
    const int64_t elems =
        (w.ndim == 1) ? w.shape[0]
                      : (w.shape[0] > 0 ? (w.shape[0] - 1) * w.stride[0] + w.shape[1] : 0);
    return ByteRange{b, b + static_cast<uintptr_t>(elems * element_bytes_of(w.dtype))};
  };
  const auto weight_info = [&](const Tensor& w, int64_t heads, const char* name,
                               QkWeight* out) -> Status {
    *out = QkWeight{};
    if (!tensor_present(w)) return Status::kSuccess;
    Status st = check_tensor(w, q.dtype, -1, 16, name);
    if (st != Status::kSuccess) return st;
    const int64_t width = heads * head_dim;
    if (full_width) {
      // OLMo-2 full-width norm: one weight over the whole row.
      if (w.ndim != 1 || w.shape[0] != width || w.stride[0] != 1) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "%s: full-width norm needs a [%lld] weight", name,
                                 static_cast<long long>(width));
      }
      out->head_stride = 0;
    } else if (w.ndim == 1) {
      if (w.stride[0] != 1 || (w.shape[0] != head_dim && w.shape[0] != width)) {
        return detail::set_error(Status::kUnsupportedShape, "%s: expected [%d] or [%lld]", name,
                                 head_dim, static_cast<long long>(width));
      }
      out->head_stride = (w.shape[0] == head_dim) ? 0 : head_dim;
    } else if (w.ndim == 2) {
      if (w.shape[0] != heads || w.shape[1] != head_dim || w.stride[1] != 1) {
        return detail::set_error(Status::kUnsupportedShape, "%s: expected [%lld, %d]", name,
                                 static_cast<long long>(heads), head_dim);
      }
      out->head_stride = w.stride[0];
    } else {
      return detail::set_error(Status::kUnsupportedShape, "%s: ndim %d", name, w.ndim);
    }
    out->ptr = static_cast<const char*>(w.data) + w.byte_offset;
    return Status::kSuccess;
  };
  QkWeight qw{}, kw{};
  s = weight_info(q_weight, num_q_heads, "qk_norm.q_weight", &qw);
  if (s != Status::kSuccess) return s;
  s = weight_info(k_weight, num_kv_heads, "qk_norm.k_weight", &kw);
  if (s != Status::kSuccess) return s;
  if (qw.ptr != nullptr) {
    s = require_disjoint(q_range, "qkv_or_q", weight_span(q_weight), "q_weight", "qk_norm");
    if (s != Status::kSuccess) return s;
  }
  if (kw.ptr != nullptr) {
    s = require_disjoint(q_range, "qkv_or_q", weight_span(k_weight), "k_weight", "qk_norm");
    if (s != Status::kSuccess) return s;
  }
  const float* partials = nullptr;
  if (tensor_present(full_width_sq_partials)) {
    const Tensor& p = full_width_sq_partials;
    if (!full_width || norm != NormType::kRMSNorm) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "qk_norm: full_width_sq_partials is only valid for full-width RMS");
    }
    s = check_tensor(p, DType::kFloat32, 2, 4, "qk_norm.full_width_sq_partials");
    if (s != Status::kSuccess) return s;
    if (p.shape[0] != q.shape[0] || p.shape[1] != 2 || p.stride[1] != 1 || p.stride[0] != 2) {
      return detail::set_error(Status::kUnsupportedLayout,
                               "qk_norm: full_width_sq_partials must be a contiguous [tokens, 2] "
                               "float32 tensor");
    }
    // Contract: entry (t, 0) is the all-reduced sum of q[t]^2 over every TP rank
    // divided by the global q width, (t, 1) the same for k[t]. The public
    // signature has no TP world size, so a raw sum could not be turned into a
    // mean here; with tp == 1 the local computation is identical, so the tensor
    // may simply be omitted.
    partials = const_ptr<float>(p);
    const ByteRange pr{reinterpret_cast<uintptr_t>(partials),
                       reinterpret_cast<uintptr_t>(partials) +
                           static_cast<uintptr_t>(p.shape[0] * 2 * sizeof(float))};
    s = require_disjoint(q_range, "qkv_or_q", pr, "full_width_sq_partials", "qk_norm");
    if (s != Status::kSuccess) return s;
    if (split) {
      s = require_disjoint(k_range, "k", pr, "full_width_sq_partials", "qk_norm");
      if (s != Status::kSuccess) return s;
    }
  }
  if (q.shape[0] == 0) return Status::kSuccess;

  QkPlan plan;
  plan.q = static_cast<char*>(q.data) + q.byte_offset;
  plan.k = split ? static_cast<char*>(k.data) + k.byte_offset : nullptr;
  plan.qw = qw;
  plan.kw = kw;
  plan.partials = partials;
  plan.q_stride = q_stride;
  plan.k_stride = k_stride;
  plan.k_offset = split ? 0 : q_width;
  plan.tokens = static_cast<int32_t>(q.shape[0]);
  plan.hq = num_q_heads;
  plan.hkv = num_kv_heads;
  plan.head_dim = head_dim;
  plan.norm = norm;
  plan.full_width = full_width;
  plan.split = split;
  plan.eps = epsilon;
  plan.w_off = weight_offset;
  switch (q.dtype) {
    case DType::kBFloat16: return dispatch_qk<__nv_bfloat16>(plan, ctx);
    case DType::kFloat16: return dispatch_qk<__half>(plan, ctx);
    default: return dispatch_qk<float>(plan, ctx);
  }
}

Status l2_norm(const Tensor& out, const Tensor& input, float scale, float epsilon,
               const LaunchContext& ctx) {
  Status s = check_epsilon(epsilon);
  if (s != Status::kSuccess) return s;
  s = check_finite_param(scale, "l2_norm: scale");
  if (s != Status::kSuccess) return s;
  RowView vo{}, vi{};
  s = make_row_view(input, "l2_norm.input", &vi);
  if (s != Status::kSuccess) return s;
  if (out.dtype != input.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "l2_norm: out %s, input %s",
                             dtype_name(out.dtype), dtype_name(input.dtype));
  }
  s = make_row_view(out, "l2_norm.out", &vo);
  if (s != Status::kSuccess) return s;
  if (!same_extent(vo, vi)) {
    return detail::set_error(Status::kInvalidArgument, "l2_norm: shape mismatch");
  }
  s = require_inplace_only(vo, "out", vi, "input", "l2_norm");
  if (s != Status::kSuccess) return s;
  if (vi.rows == 0) return Status::kSuccess;
  switch (input.dtype) {
    case DType::kBFloat16:
      return launch_l2<__nv_bfloat16, vec_width<__nv_bfloat16>()>(vo, vi, scale, epsilon, ctx);
    case DType::kFloat16:
      return launch_l2<__half, vec_width<__half>()>(vo, vi, scale, epsilon, ctx);
    default:
      return launch_l2<float, vec_width<float>()>(vo, vi, scale, epsilon, ctx);
  }
}

Status gated_rms_norm(const Tensor& out, const Tensor& x, const Tensor& z, const Tensor& weight,
                      int32_t group_size, ActFn gate_act, bool norm_before_gate, float epsilon,
                      float weight_offset, const LaunchContext& ctx) {
  if (gate_act != ActFn::kIdentity && gate_act != ActFn::kSigmoid && gate_act != ActFn::kSiLU) {
    return detail::set_error(Status::kUnsupportedShape,
                             "gated_rms_norm: gate_act %d not in {identity, sigmoid, silu}",
                             static_cast<int>(gate_act));
  }
  Status s = check_epsilon(epsilon);
  if (s != Status::kSuccess) return s;
  s = check_finite_param(weight_offset, "gated_rms_norm: weight_offset");
  if (s != Status::kSuccess) return s;
  RowView vo{}, vx{}, vz{};
  s = make_row_view(x, "gated_rms_norm.x", &vx);
  if (s != Status::kSuccess) return s;
  if (z.dtype != x.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "gated_rms_norm: z %s, x %s",
                             dtype_name(z.dtype), dtype_name(x.dtype));
  }
  s = make_row_view(z, "gated_rms_norm.z", &vz);
  if (s != Status::kSuccess) return s;
  if (out.dtype != x.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "gated_rms_norm: out %s, x %s",
                             dtype_name(out.dtype), dtype_name(x.dtype));
  }
  s = make_row_view(out, "gated_rms_norm.out", &vo);
  if (s != Status::kSuccess) return s;
  if (!same_extent(vo, vx) || !same_extent(vz, vx)) {
    return detail::set_error(Status::kInvalidArgument,
                             "gated_rms_norm: x, z and out must share one shape");
  }
  // group_size <= 0 means "one group over the whole row" (the reference's default).
  const int group = (group_size <= 0) ? static_cast<int>(vx.cols) : group_size;
  if (group <= 0 || vx.cols % group != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "gated_rms_norm: group_size %d does not divide hidden %lld", group,
                             static_cast<long long>(vx.cols));
  }
  const int64_t groups = vx.cols / group;
  if (vx.rows > 0 && vx.rows > INT32_MAX / groups) {
    return detail::set_error(Status::kUnsupportedShape, "gated_rms_norm: too many row groups");
  }
  s = require_inplace_only(vo, "out", vx, "x", "gated_rms_norm");
  if (s != Status::kSuccess) return s;
  s = require_disjoint(range_of(vo), "out", range_of(vz), "z", "gated_rms_norm");
  if (s != Status::kSuccess) return s;
  const bool has_w = tensor_present(weight);
  int64_t w_stride = 0;
  if (has_w) {
    s = check_tensor(weight, x.dtype, 1, 16, "gated_rms_norm.weight");
    if (s != Status::kSuccess) return s;
    if (weight.stride[0] != 1 ||
        (weight.shape[0] != group && weight.shape[0] != vx.cols)) {
      return detail::set_error(Status::kUnsupportedShape,
                               "gated_rms_norm: weight must be [group_size=%d] or [hidden=%lld]",
                               group, static_cast<long long>(vx.cols));
    }
    w_stride = (weight.shape[0] == group) ? 0 : group;
    ByteRange wr{};
    s = make_unit_vector(weight, x.dtype, "gated_rms_norm.weight", weight.shape[0], &wr);
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vo), "out", wr, "weight", "gated_rms_norm");
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vx), "x", wr, "weight", "gated_rms_norm");
    if (s != Status::kSuccess) return s;
    s = require_disjoint(range_of(vz), "z", wr, "weight", "gated_rms_norm");
    if (s != Status::kSuccess) return s;
  }
  if (vx.rows == 0) return Status::kSuccess;
  switch (x.dtype) {
    case DType::kBFloat16:
      return dispatch_gated<__nv_bfloat16>(
          vo, vx, vz, has_w ? const_ptr<__nv_bfloat16>(weight) : nullptr, w_stride, group,
          gate_act, norm_before_gate, epsilon, weight_offset, ctx);
    case DType::kFloat16:
      return dispatch_gated<__half>(vo, vx, vz, has_w ? const_ptr<__half>(weight) : nullptr,
                                    w_stride, group, gate_act, norm_before_gate, epsilon,
                                    weight_offset, ctx);
    default:
      return dispatch_gated<float>(vo, vx, vz, has_w ? const_ptr<float>(weight) : nullptr,
                                   w_stride, group, gate_act, norm_before_gate, epsilon,
                                   weight_offset, ctx);
  }
}

// -----------------------------------------------------------------------------
// N02: norm + residual + quantization (B3 row 5)
// -----------------------------------------------------------------------------

namespace {

__device__ __forceinline__ uint8_t nq_e4m3(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint8_t nq_e2m1_pair(float a, float b) {
  return static_cast<uint8_t>(
      __nv_cvt_float2_to_fp4x2(make_float2(a, b), __NV_E2M1, cudaRoundNearest));
}

__device__ __forceinline__ int64_t nq_scale_offset(ScaleLayout layout, int64_t row, int64_t group,
                                                   int64_t rows, int64_t groups) {
  switch (layout) {
    case ScaleLayout::kRowMajor:
    case ScaleLayout::kPackedUE8M0x4:
      return row * groups + group;
    case ScaleLayout::kColMajorTma: {
      const int64_t padded = (rows + 3) & ~int64_t(3);
      return group * padded + row;
    }
    case ScaleLayout::kSwizzled128x4: {
      const int64_t m_tile = row >> 7;
      const int64_t outer_m = row & 31;
      const int64_t inner_m = (row >> 5) & 3;
      const int64_t k_tile = group >> 2;
      const int64_t inner_k = group & 3;
      const int64_t k_tiles = (groups + 3) >> 2;
      return ((m_tile * k_tiles + k_tile) << 9) + (outer_m << 4) + (inner_m << 2) + inner_k;
    }
    default:
      return row * groups + group;
  }
}

__device__ __forceinline__ uint8_t nq_ue8m0(float s) {
  int e = 0;
  if (s > 0.0f && isfinite(s)) {
    frexpf(s, &e);
    e -= 1;
    if (ldexpf(1.0f, e) < s) ++e;
  } else {
    e = -127;
  }
  e = e < -127 ? -127 : (e > 127 ? 127 : e);
  return static_cast<uint8_t>(e + 127);
}

__device__ __forceinline__ float nq_ue8m0_to_float(uint8_t e) {
  return ldexpf(1.0f, static_cast<int>(e) - 127);
}

__device__ __forceinline__ float nq_warp_max(float v) {
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ float nq_block_max(float v, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = nq_warp_max(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    float t = (threadIdx.x < warps) ? smem[threadIdx.x] : -__int_as_float(0x7f800000);
    t = nq_warp_max(t);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

}  // namespace

template <typename scalar_t, typename residual_t, NormType kNorm, ActQuantMode kMode, int VEC_SIZE>
__global__ void norm_quant_kernel(const __grid_constant__ NormQuantParams params) {
  __shared__ float smem[32];
  const int64_t row = blockIdx.x;
  if (row >= params.num_tokens) return;
  const int64_t hidden = params.hidden_size;
  const scalar_t* in = static_cast<const scalar_t*>(params.input) + row * params.input_stride;
  residual_t* res = params.residual != nullptr
                        ? static_cast<residual_t*>(params.residual) + row * params.input_stride
                        : nullptr;
  const residual_t* res_in = params.residual_in != nullptr
                                 ? static_cast<const residual_t*>(params.residual_in) +
                                       row * params.input_stride
                                 : res;
  const scalar_t* weight = static_cast<const scalar_t*>(params.weight);
  const scalar_t* bias = static_cast<const scalar_t*>(params.bias);
  (void)bias;
  const int64_t vecs = hidden / VEC_SIZE;

  // Pass 1: optional residual update, then the row reduction.
  if (res != nullptr) {
    for (int64_t v = threadIdx.x; v < vecs; v += blockDim.x) {
      const int64_t c = v * VEC_SIZE;
      const Pack<scalar_t, VEC_SIZE> x = *reinterpret_cast<const Pack<scalar_t, VEC_SIZE>*>(in + c);
      Pack<residual_t, VEC_SIZE> r = *reinterpret_cast<const Pack<residual_t, VEC_SIZE>*>(res_in + c);
      #pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float rv = __fmaf_rn(params.input_scale, to_f<scalar_t>(x.x[j]),
                                   to_f<residual_t>(r.x[j]));
        from_f<residual_t>(r.x[j], rv);
      }
      *reinterpret_cast<Pack<residual_t, VEC_SIZE>*>(res + c) = r;
    }
    for (int64_t c = vecs * VEC_SIZE + threadIdx.x; c < hidden; c += blockDim.x) {
      const float rv = __fmaf_rn(params.input_scale, to_f<scalar_t>(in[c]),
                                 to_f<residual_t>(res_in[c]));
      from_f<residual_t>(res[c], rv);
    }
    __syncthreads();
  }
  const void* src_base = (res != nullptr) ? static_cast<const void*>(res) : static_cast<const void*>(in);
  const bool src_is_res = res != nullptr;

  float sum = 0.0f, sumsq = 0.0f;
  for (int64_t v = threadIdx.x; v < vecs; v += blockDim.x) {
    const int64_t c = v * VEC_SIZE;
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      float x;
      if (src_is_res) {
        x = to_f<residual_t>(static_cast<const residual_t*>(src_base)[c + j]);
      } else {
        x = to_f<scalar_t>(static_cast<const scalar_t*>(src_base)[c + j]);
      }
      sum += x;
      sumsq += x * x;
    }
  }
  for (int64_t c = vecs * VEC_SIZE + threadIdx.x; c < hidden; c += blockDim.x) {
    float x;
    if (src_is_res) {
      x = to_f<residual_t>(static_cast<const residual_t*>(src_base)[c]);
    } else {
      x = to_f<scalar_t>(static_cast<const scalar_t*>(src_base)[c]);
    }
    sum += x;
    sumsq += x * x;
  }
  if constexpr (kNorm == NormType::kLayerNorm) {
    sum = block_sum(sum, smem);
    sumsq = block_sum(sumsq, smem);
  } else {
    sumsq = block_sum(sumsq, smem);
    sum = 0.0f;
  }
  const float inv_n = 1.0f / static_cast<float>(hidden);
  const float mean = sum * inv_n;
  const float var = fmaxf(sumsq * inv_n - mean * mean, 0.0f);
  const float rstd = rsqrtf(var + params.epsilon);

  auto norm_at = [&](int64_t c) -> float {
    float x;
    if (src_is_res) {
      x = to_f<residual_t>(static_cast<const residual_t*>(src_base)[c]);
    } else {
      x = to_f<scalar_t>(static_cast<const scalar_t*>(src_base)[c]);
    }
    float y = (x - mean) * rstd;
    if (weight != nullptr) y *= to_f<scalar_t>(weight[c]) + params.weight_offset;
    if constexpr (kNorm == NormType::kLayerNorm) {
      if (bias != nullptr) y += to_f<scalar_t>(bias[c]);
    }
    return y;
  };
  auto write_norm_out = [&](int64_t c, float y) {
    if (params.norm_out != nullptr) {
      from_f<scalar_t>(static_cast<scalar_t*>(params.norm_out)[row * params.norm_out_stride + c],
                            y);
    }
  };

  ActQuantOut q = params.quant;
  const int64_t q_stride = q.q_row_stride > 0 ? q.q_row_stride : hidden;
  (void)q.scale_row_stride;
  const float qmax = q.q_dtype == DType::kInt8 ? kInt8Max : kFp8E4M3Max;
  uint8_t* qptr = static_cast<uint8_t*>(q.q);

  if constexpr (kMode == ActQuantMode::kNone) {
    for (int64_t c = threadIdx.x; c < hidden; c += blockDim.x) write_norm_out(c, norm_at(c));
    return;
  }

  if constexpr (kMode == ActQuantMode::kStaticPerTensor) {
    const float s = *q.static_scale;
    for (int64_t v = threadIdx.x; v < vecs; v += blockDim.x) {
      const int64_t c = v * VEC_SIZE;
      uint8_t codes[VEC_SIZE];
      #pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float y = norm_at(c + j);
        write_norm_out(c + j, y);
        if (q.q_dtype == DType::kInt8) {
          int qv = __float2int_rn(y / s);
          qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);
          codes[j] = static_cast<uint8_t>(static_cast<int8_t>(qv));
        } else {
          codes[j] = nq_e4m3(__fdiv_rn(y, s));
        }
      }
      #pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) qptr[row * q_stride + c + j] = codes[j];
    }
    for (int64_t c = vecs * VEC_SIZE + threadIdx.x; c < hidden; c += blockDim.x) {
      const float y = norm_at(c);
      write_norm_out(c, y);
      if (q.q_dtype == DType::kInt8) {
        int qv = __float2int_rn(y / s);
        qv = qv > 127 ? 127 : (qv < -127 ? -127 : qv);
        qptr[row * q_stride + c] = static_cast<uint8_t>(static_cast<int8_t>(qv));
      } else {
        qptr[row * q_stride + c] = nq_e4m3(__fdiv_rn(y, s));
      }
    }
    return;
  }

  if constexpr (kMode == ActQuantMode::kDynamicPerToken) {
    __shared__ __align__(16) float row_cache[4096];
    const bool cache_ok = hidden <= 4096;
    const bool asym = !q.symmetric && q.q_dtype == DType::kInt8;
    float amax = 0.0f, amin = __int_as_float(0x7f800000);
    const int64_t nv = hidden / 4;
    if (cache_ok && hidden % 4 == 0 && (q_stride % 4 == 0) && !asym) {
      for (int64_t v = threadIdx.x; v < nv; v += blockDim.x) {
        float y4[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          y4[j] = norm_at(v * 4 + j);
          write_norm_out(v * 4 + j, y4[j]);
          amax = fmaxf(amax, fabsf(y4[j]));
          amin = fminf(amin, y4[j]);
        }
        *reinterpret_cast<float4*>(&row_cache[v * 4]) =
            make_float4(y4[0], y4[1], y4[2], y4[3]);
      }
      amax = nq_block_max(amax, smem);
      __syncthreads();
      amin = -nq_block_max(-amin, smem);
      float s;
      if (threadIdx.x == 0) {
        s = fmaxf(amax, 1e-10f) / qmax;
        if (q.scale_upper_bound != nullptr) s = fminf(s, q.scale_upper_bound[0]);
        static_cast<float*>(q.scales)[row] = s;
      }
      __syncthreads();
      s = static_cast<float*>(q.scales)[row];
      for (int64_t v = threadIdx.x; v < nv; v += blockDim.x) {
        const float4 y4 = *reinterpret_cast<const float4*>(&row_cache[v * 4]);
        const float yy[4] = {y4.x, y4.y, y4.z, y4.w};
        uint8_t codes[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          codes[j] = q.q_dtype == DType::kInt8
                         ? static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(yy[j] / s)))
                         : nq_e4m3(__fdiv_rn(yy[j], s));
        }
        *reinterpret_cast<uint32_t*>(qptr + row * q_stride + v * 4) =
            static_cast<uint32_t>(codes[0]) | (static_cast<uint32_t>(codes[1]) << 8) |
            (static_cast<uint32_t>(codes[2]) << 16) | (static_cast<uint32_t>(codes[3]) << 24);
      }
      return;
    }
    for (int64_t c = threadIdx.x; c < hidden; c += blockDim.x) {
      const float y = norm_at(c);
      if (cache_ok) row_cache[c] = y;
      amax = fmaxf(amax, fabsf(y));
      amin = fminf(amin, y);
    }
    amax = nq_block_max(amax, smem);
    __syncthreads();
    amin = -nq_block_max(-amin, smem);
    float s;
    float zero = -128.0f;
    if (asym) {
      s = __fdiv_rn(fmaxf(amax, 1e-10f) - amin, 255.0f);
      if (threadIdx.x == 0) {
        const int z = __float2int_rn(-128.0f - __fdiv_rn(amin, s));
        q.zero_points[row] = z;
        zero = static_cast<float>(z);
        static_cast<float*>(q.scales)[row] = s;
      }
      __syncthreads();
      zero = static_cast<float>(q.zero_points[row]);
      s = static_cast<float*>(q.scales)[row];
    } else {
      s = fmaxf(amax, 1e-10f) / qmax;
      if (q.scale_upper_bound != nullptr) s = fminf(s, q.scale_upper_bound[0]);
      if (threadIdx.x == 0) static_cast<float*>(q.scales)[row] = s;
      __syncthreads();
      s = static_cast<float*>(q.scales)[row];
    }
    for (int64_t c = threadIdx.x; c < hidden; c += blockDim.x) {
      const float y = norm_at(c);
      write_norm_out(c, y);
      if (asym) {
        int v = __float2int_rn(y / s + zero);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        qptr[row * q_stride + c] = static_cast<uint8_t>(static_cast<int8_t>(v));
      } else {
        qptr[row * q_stride + c] = q.q_dtype == DType::kInt8
                                       ? static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(y / s)))
                                       : nq_e4m3(__fdiv_rn(y, s));
      }
    }
    return;
  }

  // Group modes: one warp owns a whole group (32/64/128 or 32 MX / 16 NVFP4).
  const int group = kMode == ActQuantMode::kNVFP4
                        ? kNvFp4Block
                        : (kMode == ActQuantMode::kMXFP8 || kMode == ActQuantMode::kMXFP4
                               ? kMxBlock
                               : (q.group_size > 0 ? q.group_size : 128));
  const int64_t groups = hidden / group;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int kWarps = 8;
  const int per_lane = group / 32;
  for (int64_t g = warp; g < groups; g += kWarps) {
    float y[8];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j = lane + 32 * i;
      y[i] = (i < per_lane) ? norm_at(g * group + j) : 0.0f;
      if (i < per_lane) amax = fmaxf(amax, fabsf(y[i]));
    }
    amax = nq_warp_max(amax);
    float s = fmaxf(amax, 1e-10f) / qmax;
    if (kMode == ActQuantMode::kPerGroup && q.scale_ue8m0) {
      const uint8_t e = nq_ue8m0(s);
      if (lane == 0) static_cast<float*>(q.scales)[nq_scale_offset(q.scale_layout, row, g,
                                                                   params.num_tokens, groups)] =
          nq_ue8m0_to_float(e);
      s = __shfl_sync(0xffffffffu, s, 0);
    } else if constexpr (kMode == ActQuantMode::kMXFP8 || kMode == ActQuantMode::kMXFP4) {
      const uint8_t e = nq_ue8m0(s);
      if (lane == 0) {
        static_cast<uint8_t*>(q.scales)[nq_scale_offset(q.scale_layout, row, g,
                                                        params.num_tokens, groups)] = e;
      }
      s = __shfl_sync(0xffffffffu, nq_ue8m0_to_float(e), 0);
    } else if constexpr (kMode == ActQuantMode::kNVFP4) {
      const float gscale = q.global_scale != nullptr ? q.global_scale[0] : 1.0f;
      const uint8_t e = nq_e4m3(__fdiv_rn(__fmul_rn(amax, gscale), kFp4E2M1Max));
      if (lane == 0) {
        static_cast<uint8_t*>(q.scales)[nq_scale_offset(q.scale_layout, row, g,
                                                        params.num_tokens, groups)] = e;
      }
      s = __shfl_sync(0xffffffffu, __half2float(__nv_cvt_fp8_to_halfraw(e, __NV_E4M3)), 0) / gscale;
    } else {
      if (lane == 0) {
        static_cast<float*>(q.scales)[nq_scale_offset(q.scale_layout, row, g,
                                                      params.num_tokens, groups)] = s;
      }
      s = __shfl_sync(0xffffffffu, s, 0);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i >= per_lane) break;
      const int j = lane + 32 * i;
      const int64_t c = g * group + j;
      const float yv = y[i];
      write_norm_out(c, yv);
      if constexpr (kMode == ActQuantMode::kMXFP4) {
        if ((lane & 1) == 0 && i < per_lane) {
          const float y2 = __shfl_down_sync(0xffffffffu, yv, 1);
          qptr[row * q_stride + (c / 2)] = nq_e2m1_pair(__fdiv_rn(yv, s), __fdiv_rn(y2, s));
        }
      } else if constexpr (kMode == ActQuantMode::kNVFP4) {
        if ((lane & 1) == 0 && i < per_lane) {
          const float y2 = __shfl_down_sync(0xffffffffu, yv, 1);
          const float gscale = q.global_scale != nullptr ? q.global_scale[0] : 1.0f;
          qptr[row * q_stride + (c / 2)] =
              nq_e2m1_pair(__fdiv_rn(__fmul_rn(yv, gscale), s),
                           __fdiv_rn(__fmul_rn(y2, gscale), s));
        }
      } else {
        const uint8_t code = q.q_dtype == DType::kInt8
                                 ? static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(yv / s)))
                                 : nq_e4m3(__fdiv_rn(yv, s));
        qptr[row * q_stride + c] = code;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Host dispatch
// -----------------------------------------------------------------------------

namespace {

template <typename scalar_t, typename residual_t, NormType kNorm, ActQuantMode kMode>
Status launch_norm_quant(const NormQuantParams& p, cudaStream_t stream) {
  const int block = 256;
  if (p.hidden_size % 8 == 0) {
    norm_quant_kernel<scalar_t, residual_t, kNorm, kMode, 8>
        <<<p.num_tokens, block, 0, stream>>>(p);
  } else if (p.hidden_size % 4 == 0) {
    norm_quant_kernel<scalar_t, residual_t, kNorm, kMode, 4>
        <<<p.num_tokens, block, 0, stream>>>(p);
  } else {
    norm_quant_kernel<scalar_t, residual_t, kNorm, kMode, 1>
        <<<p.num_tokens, block, 0, stream>>>(p);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return check_launch("norm_quant");
  (void)err;
  return Status::kSuccess;
}

template <typename scalar_t, typename residual_t>
Status dispatch_norm_mode(const NormQuantParams& p, NormType norm, ActQuantMode mode,
                          cudaStream_t stream) {
  if (mode == ActQuantMode::kNone) {
    if (norm == NormType::kLayerNorm) {
      return launch_norm_quant<scalar_t, residual_t, NormType::kLayerNorm, ActQuantMode::kNone>(
          p, stream);
    }
    if (norm == NormType::kRMSNorm) {
      return launch_norm_quant<scalar_t, residual_t, NormType::kRMSNorm, ActQuantMode::kNone>(
          p, stream);
    }
    return detail::set_error(Status::kUnsupportedShape, "norm_quant: norm %d",
                             static_cast<int>(norm));
  }
  if (norm == NormType::kLayerNorm) {
    #define KOPS_NQL(M) return launch_norm_quant<scalar_t, residual_t, NormType::kLayerNorm, M>(p, stream)
    switch (mode) {
      case ActQuantMode::kStaticPerTensor: KOPS_NQL(ActQuantMode::kStaticPerTensor);
      case ActQuantMode::kDynamicPerToken: KOPS_NQL(ActQuantMode::kDynamicPerToken);
      case ActQuantMode::kPerGroup: KOPS_NQL(ActQuantMode::kPerGroup);
      case ActQuantMode::kMXFP8: KOPS_NQL(ActQuantMode::kMXFP8);
      case ActQuantMode::kMXFP4: KOPS_NQL(ActQuantMode::kMXFP4);
      case ActQuantMode::kNVFP4: KOPS_NQL(ActQuantMode::kNVFP4);
      default: break;
    }
    #undef KOPS_NQL
    return detail::set_error(Status::kUnsupportedShape, "norm_quant: mode %d",
                             static_cast<int>(mode));
  }
  if (norm != NormType::kRMSNorm) {
    return detail::set_error(Status::kUnsupportedShape, "norm_quant: norm %d",
                             static_cast<int>(norm));
  }
  switch (mode) {
    case ActQuantMode::kStaticPerTensor:
      return launch_norm_quant<scalar_t, residual_t, NormType::kRMSNorm,
                               ActQuantMode::kStaticPerTensor>(p, stream);
    case ActQuantMode::kDynamicPerToken:
      return launch_norm_quant<scalar_t, residual_t, NormType::kRMSNorm,
                               ActQuantMode::kDynamicPerToken>(p, stream);
    case ActQuantMode::kPerGroup:
      return launch_norm_quant<scalar_t, residual_t, NormType::kRMSNorm, ActQuantMode::kPerGroup>(
          p, stream);
    case ActQuantMode::kMXFP8:
      return launch_norm_quant<scalar_t, residual_t, NormType::kRMSNorm, ActQuantMode::kMXFP8>(
          p, stream);
    case ActQuantMode::kMXFP4:
      return launch_norm_quant<scalar_t, residual_t, NormType::kRMSNorm, ActQuantMode::kMXFP4>(
          p, stream);
    case ActQuantMode::kNVFP4:
      return launch_norm_quant<scalar_t, residual_t, NormType::kRMSNorm, ActQuantMode::kNVFP4>(
          p, stream);
    default:
      return detail::set_error(Status::kUnsupportedShape, "norm_quant: mode %d",
                               static_cast<int>(mode));
  }
#undef KOPS_NQ
}

}  // namespace

Status norm_quant(const NormQuantParams& params, NormType norm, DType input_dtype,
                  DType residual_dtype, const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "norm_quant: CUDA backend only");
  }
  if (params.input == nullptr || params.hidden_size <= 0 || params.num_tokens < 0) {
    return detail::set_error(Status::kInvalidArgument, "norm_quant: input/extents");
  }
  if (params.num_tokens == 0) return Status::kSuccess;
  if (params.quant.mode != ActQuantMode::kNone && params.quant.q == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "norm_quant: quant.q is null");
  }
  if (params.quant.mode != ActQuantMode::kNone &&
      params.quant.mode != ActQuantMode::kStaticPerTensor && params.quant.scales == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "norm_quant: quant.scales is null");
  }
  if (params.quant.mode == ActQuantMode::kStaticPerTensor &&
      params.quant.static_scale == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "norm_quant: static_scale is null");
  }
  if (params.quant.mode == ActQuantMode::kDynamicPerToken &&
      params.quant.q_dtype == DType::kInt8 && !params.quant.symmetric &&
      params.quant.zero_points == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "norm_quant: asymmetric INT8 needs zero_points");
  }
  if ((params.quant.mode == ActQuantMode::kMXFP8 ||
       params.quant.mode == ActQuantMode::kMXFP4) &&
      params.hidden_size % kMxBlock != 0) {
    return detail::set_error(Status::kUnsupportedShape, "norm_quant: MX needs hidden %% 32");
  }
  if (params.quant.mode == ActQuantMode::kNVFP4) {
    if (params.hidden_size % kNvFp4Block != 0 || params.quant.global_scale == nullptr) {
      return detail::set_error(Status::kUnsupportedShape,
                               "norm_quant: NVFP4 needs hidden %% 16 and global_scale");
    }
  }
  if (params.quant.mode == ActQuantMode::kPerGroup) {
    const int group = params.quant.group_size > 0 ? params.quant.group_size : 128;
    if ((group != 32 && group != 64 && group != 128) || params.hidden_size % group != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "norm_quant: group %d does not divide hidden", group);
    }
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const bool res_f32 = residual_dtype == DType::kFloat32;
  const bool res_same = residual_dtype == input_dtype;
  if (!res_f32 && !res_same) {
    return detail::set_error(Status::kUnsupportedDtype, "norm_quant: residual dtype mismatch");
  }
#define KOPS_NORM_QUANT(T)                                                                          \
  do {                                                                                              \
    if (params.residual == nullptr) {                                                               \
      return dispatch_norm_mode<T, T>(params, norm, params.quant.mode, stream);                     \
    }                                                                                               \
    if (res_f32) {                                                                                  \
      return dispatch_norm_mode<T, float>(params, norm, params.quant.mode, stream);                 \
    }                                                                                               \
    return dispatch_norm_mode<T, T>(params, norm, params.quant.mode, stream);                       \
  } while (0)
  if (input_dtype == DType::kBFloat16) KOPS_NORM_QUANT(__nv_bfloat16);
  if (input_dtype == DType::kFloat16) KOPS_NORM_QUANT(__half);
  if (input_dtype == DType::kFloat32) KOPS_NORM_QUANT(float);
#undef KOPS_NORM_QUANT
  return detail::set_error(Status::kUnsupportedDtype, "norm_quant: input dtype %s",
                           dtype_name(input_dtype));
}

}  // namespace kops::norm
