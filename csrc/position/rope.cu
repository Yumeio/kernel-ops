// =============================================================================
// kernel-ops | csrc/position/rope.cu                                  Catalog: R01
// B1 position family: host frequency tables, the cos/sin cache builder and the
// RoPE application kernels (table, on-the-fly, M-RoPE, fused QK-norm+RoPE,
// Llama-4 temperature tuning).
//
// What this file implements
//   compute_inv_freq       every RopeScaling in fp64 host math -> float inv_freq[R/2]
//                          plus the cos/sin magnitude multiplier (YaRN mscale,
//                          LongRoPE sqrt(1 + ln f / ln L0), attention_factor).
//   compute_alibi_slopes   Press et al. slopes; a head count that is not a power
//                          of two uses the BLOOM/MPT interleaved rule.
//   build_cos_sin_cache    cache[p, i] = mscale * cos(p f_i), cache[p, R/2+i] = sin;
//                          fp64 angle, fp64 sincos, one rounding to the cache dtype.
//   apply_rotary           table path, NeoX and GPT-J, partial rot_dim, rot_offset,
//                          optional per-token cache row offsets, strided Q/K, in place.
//   apply_rotary_on_the_fly same rotation with sincos from inv_freq (no table).
//   apply_mrope            [3, T] positions; contiguous and interleaved sections.
//   fused_qk_norm_rope     per-head RMSNorm(Q) and RMSNorm(K), then RoPE, on packed
//                          QKV; V untouched; head_dim 64/128/256.
//   attn_temperature_tuning Llama-4 NoPE/iRoPE query scaling, in place.
//
// Numerics: the angle is formed as `(double)position * (double)inv_freq[i]` and
// taken with fp64 `sincos`; the only rounding is the store (one rounding per
// value). The pair rotation is fp32:
//     ox = fma(-y, s, x * c),  oy = fma(y, c, x * s)
// i.e. two rounded products plus one fused add per output, which is within a
// couple of fp32 ulp of the fp64 value. (The vendored reference carries the
// product error terms as well; that extra compensation is not part of this ABI.)
//
// ABI gaps that shape the host validation below (the declared kernels carry no
// such argument, so the host constrains the layout instead):
//   * no cache row stride / row count: cache rows must be exactly rot_dim wide,
//     and a position (or position + offset) outside the cache is not guarded
//     in the kernel - the engine owns position validity, as it owns the cache;
//   * no runtime head_dim in fused_qk_norm_rope_kernel: head_dim is a template
//     parameter, so only the instantiated 64 / 128 / 256 are accepted;
//   * no positions dtype flag: positions are int64;
//   * no positions row stride in mrope_kernel: positions_thw is contiguous [3, T];
//   * no head stride in attn_temperature_tuning_kernel: heads are contiguous
//     inside a token (query.stride(1) == head_size);
//   * build_cos_sin_cache carries the frequency table by value in the kernel
//     parameter; there is no module-level device buffer.
//
// Target support: arch-generic body (no cp.async, no TMA, no WGMMA, no fp64-only
// intrinsic), so it belongs to Tier A/B/C alike; exact-target selection stays in
// the runtime registry.
// =============================================================================
#include <algorithm>
#include <cmath>
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
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/position/rope_kernels.cuh"
#include "csrc/position/ops.h"

namespace kops::rope {

namespace {

// ---- limits and geometry ----------------------------------------------------

/// Fused QK-norm+RoPE stages one head in shared memory, so head_dim is bounded.
constexpr int kMaxFusedHeadDim = 256;
/// Positions written per CTA by build_cos_sin_cache_kernel.
constexpr int kCachePositionsPerCta = 4;
/// Warps (one (token, head) task each) per CTA of fused_qk_norm_rope_kernel.
constexpr int kFusedWarpsPerCta = 4;
constexpr int kFusedThreads = 32 * kFusedWarpsPerCta;
/// Largest block dimension used by the apply kernels (SM occupancy, no smem use).
constexpr int kMaxApplyBlock = 512;

constexpr double kTwoPi = 6.283185307179586476925286766559;

/// Per-type tag for the host dtype dispatch (avoids a macro per entry point).
template <typename T>
struct ScalarTag {
  using type = T;
};

// ---- device helpers --------------------------------------------------------

/// ox = x c - y s, oy = y c + x s (fp32; one fma + one rounded product each).
__device__ __forceinline__ void rotate_pair(float x, float y, float c, float s, float& ox,
                                            float& oy) {
  ox = __fmaf_rn(-y, s, __fmul_rn(x, c));
  oy = __fmaf_rn(y, c, __fmul_rn(x, s));
}

/// Element offsets of rotary pair i inside the rotated window.
template <RopeStyle kStyle>
__device__ __forceinline__ void pair_offsets(int i, int half, int& a, int& b) {
  if (kStyle == RopeStyle::kNeoX) {
    a = i;          // (i, i + rot_dim/2): two runs of the window
    b = half + i;
  } else {
    a = 2 * i;      // (2i, 2i + 1): interleaved
    b = 2 * i + 1;
  }
}

/// Axis (0 = t, 1 = h, 2 = w) that supplies the position of rotary pair i.
/// Contiguous: three runs [t | h | w] (Qwen2-VL / Qwen2.5-VL).
/// Interleaved: t occupies every third slot, h and w claim theirs while they fit
/// in 3 * section (Qwen3-VL, GLM-4.5V).
template <bool kInterleaved>
__device__ __forceinline__ int mrope_axis(int i, int section_t, int section_h, int section_w) {
  if (kInterleaved) {
    if (i % 3 == 1 && i < 3 * section_h) return 1;
    if (i % 3 == 2 && i < 3 * section_w) return 2;
    return 0;
  }
  return i < section_t ? 0 : (i < section_t + section_h ? 1 : 2);
}

// ---- host helpers ----------------------------------------------------------

/// std::isfinite through one name so the config checks read uniformly.
bool finite(float v) { return std::isfinite(v); }

/// 0.1 m ln(s) + 1 for s > 1, else 1 (YaRN magnitude correction).
double yarn_get_mscale(double scale, double m) {
  return scale <= 1.0 ? 1.0 : 0.1 * m * std::log(scale) + 1.0;
}

/// Rotary dimension that completes `num_rotations` turns within `max_position`.
double yarn_find_correction_dim(double num_rotations, int dim, double base, double max_position) {
  return (double(dim) * std::log(max_position / (num_rotations * kTwoPi))) /
         (2.0 * std::log(base));
}

int32_t element_bytes(DType dt) {
  switch (dt) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
    default:
      return 0;
  }
}

const void* const_data(const Tensor& t) {
  return static_cast<const char*>(t.data) + t.byte_offset;
}

void* mutable_data(const Tensor& t) { return static_cast<char*>(t.data) + t.byte_offset; }

Status check_launch_context(const LaunchContext& ctx, const char* op) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "%s: backend is not CUDA", op);
  }
  if (target_is_amd(ctx.arch_id)) {
    return detail::set_error(Status::kUnsupportedArch, "%s: arch %d has no built kernel", op,
                             ctx.arch_id);
  }
  if (ctx.device < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: launch context has no device", op);
  }
  return Status::kSuccess;
}

Status check_style(RopeStyle style, const char* op) {
  if (style != RopeStyle::kNeoX && style != RopeStyle::kGPTJ) {
    return detail::set_error(Status::kInvalidArgument, "%s: unknown RopeStyle %d", op,
                             static_cast<int>(style));
  }
  return Status::kSuccess;
}

Status check_rope_window(int32_t rot_dim, int32_t rot_offset, const char* op) {
  if (rot_dim < 2 || rot_dim > kMaxRotDim || (rot_dim % 2) != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: rot_dim %d must be even and in [2, %d]", op, rot_dim, kMaxRotDim);
  }
  if (rot_offset < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: rot_offset %d < 0", op, rot_offset);
  }
  return Status::kSuccess;
}

Status check_same_device(const Tensor& a, const Tensor& b, const char* label) {
  if (a.device != b.device) {
    return detail::set_error(Status::kInvalidArgument, "%s: device %d != %d", label, a.device,
                             b.device);
  }
  return Status::kSuccess;
}

Status check_device(const Tensor& t, const LaunchContext& ctx, const char* label) {
  if (t.device != ctx.device) {
    return detail::set_error(Status::kInvalidArgument, "%s: device %d != launch device %d", label,
                             t.device, ctx.device);
  }
  return Status::kSuccess;
}

/// [T, H, D] query/key view: fp32/fp16/bf16, unit innermost stride, D >= rot_end.
Status check_heads(const Tensor& t, const char* label, int32_t rot_end) {
  const DType dt = t.dtype;
  if (element_bytes(dt) == 0) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s (fp32/fp16/bf16 expected)",
                             label, dtype_name(dt));
  }
  Status s = check_tensor(t, dt, 3, static_cast<size_t>(element_bytes(dt)), label);
  if (s != Status::kSuccess) return s;
  if (t.shape[2] < rot_end) {
    return detail::set_error(Status::kUnsupportedShape, "%s: head size %lld < rot_end %d", label,
                             static_cast<long long>(t.shape[2]), rot_end);
  }
  if (t.shape[2] > 1 && t.stride[2] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: last-dim stride %lld != 1", label,
                             static_cast<long long>(t.stride[2]));
  }
  if (t.shape[0] > INT32_MAX || t.shape[1] > INT32_MAX || t.shape[2] > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "%s: a dimension exceeds INT32_MAX", label);
  }
  return Status::kSuccess;
}

/// [T] int64 positions with unit stride.
Status check_positions(const Tensor& t, const char* label, int64_t tokens) {
  if (t.dtype != DType::kInt64 && t.dtype != DType::kInt32) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: dtype %s (int64 expected)", label,
                             dtype_name(t.dtype));
  }
  if (t.dtype != DType::kInt64) {
    // The declared kernels take `const int64_t* positions`; no dtype flag exists.
    return detail::set_error(Status::kUnsupportedDtype,
                             "%s: dtype %s; this ABI takes int64 positions", label,
                             dtype_name(t.dtype));
  }
  Status s = check_tensor(t, DType::kInt64, 1, 8, label);
  if (s != Status::kSuccess) return s;
  if (t.shape[0] != tokens) {
    return detail::set_error(Status::kUnsupportedShape, "%s: %lld entries, expected %lld", label,
                             static_cast<long long>(t.shape[0]), static_cast<long long>(tokens));
  }
  if (tokens > 1 && t.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: stride %lld != 1", label,
                             static_cast<long long>(t.stride[0]));
  }
  return Status::kSuccess;
}

/// [rows, rot_dim] cos/sin table: unit column stride, rows exactly rot_dim wide.
Status check_cos_sin_cache(const Tensor& cache, DType query_dtype, int32_t rot_dim,
                           const char* label) {
  if (cache.dtype != DType::kFloat32 && cache.dtype != query_dtype) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "%s: dtype %s must be float32 or the query dtype", label,
                             dtype_name(cache.dtype));
  }
  Status s = check_tensor(cache, cache.dtype, 2, static_cast<size_t>(element_bytes(cache.dtype)),
                          label);
  if (s != Status::kSuccess) return s;
  if (cache.shape[1] != rot_dim) {
    return detail::set_error(Status::kUnsupportedShape, "%s: width %lld != rot_dim %d", label,
                             static_cast<long long>(cache.shape[1]), rot_dim);
  }
  if (cache.shape[0] < 1) {
    return detail::set_error(Status::kUnsupportedShape, "%s: needs at least one row", label);
  }
  if (cache.shape[1] > 1 && cache.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: column stride %lld != 1", label,
                             static_cast<long long>(cache.stride[1]));
  }
  if (cache.shape[0] > 1 && cache.stride[0] != rot_dim) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "%s: row stride %lld must equal rot_dim %d (the kernel has no "
                             "row-stride argument)",
                             label, static_cast<long long>(cache.stride[0]), rot_dim);
  }
  return Status::kSuccess;
}

Status check_launch(const char* op) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s launch: %s",
                                     op, cudaGetErrorString(err));
  }
  return Status::kSuccess;
}

}  // namespace

// =============================================================================
// Host: frequency tables
// =============================================================================

Status compute_inv_freq(const RopeConfig& cfg, float* inv_freq_host, float* cos_sin_mscale) {
  if (inv_freq_host == nullptr || cos_sin_mscale == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "compute_inv_freq: null output pointer");
  }
  const int dim = cfg.rotary_dim;
  if (dim < 2 || dim > kMaxRotDim || (dim % 2) != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "compute_inv_freq: rotary_dim %d must be even and in [2, %d]", dim,
                             kMaxRotDim);
  }
  if (!finite(cfg.theta) || !(cfg.theta > 0.0f)) {
    return detail::set_error(Status::kInvalidArgument, "compute_inv_freq: theta %g must be > 0",
                             static_cast<double>(cfg.theta));
  }
  if (!finite(cfg.factor) || !(cfg.factor > 0.0f)) {
    return detail::set_error(Status::kInvalidArgument, "compute_inv_freq: factor %g must be > 0",
                             static_cast<double>(cfg.factor));
  }
  if (!finite(cfg.attention_factor) || cfg.attention_factor < 0.0f) {
    return detail::set_error(Status::kInvalidArgument,
                             "compute_inv_freq: attention_factor %g must be >= 0",
                             static_cast<double>(cfg.attention_factor));
  }
  if (!finite(cfg.mscale) || !finite(cfg.mscale_all_dim) || !finite(cfg.beta_fast) ||
      !finite(cfg.beta_slow) || !finite(cfg.low_freq_factor) || !finite(cfg.high_freq_factor)) {
    return detail::set_error(Status::kInvalidArgument,
                             "compute_inv_freq: a scaling constant is not finite");
  }

  const int half = dim / 2;
  const double theta = static_cast<double>(cfg.theta);
  const double factor = static_cast<double>(cfg.factor);
  const double L0 = static_cast<double>(cfg.original_max_position);
  double mscale = 1.0;
  double base = theta;

  switch (cfg.scaling) {
    case RopeScaling::kNone:
    case RopeScaling::kLinear:
      break;
    case RopeScaling::kDynamicNTK: {
      // RopeConfig has no seq_len, so the extended length is the configured
      // target L = original_max_position * factor: the documented fallback of
      // the reference implementation (seq_len = 0) and of HF `_compute_dynamic_
      // ntk_parameters` evaluated at the maximum extended length. base = theta
      // at L = L0, so a factor of 1 reproduces kNone.
      if (!(L0 > 0.0)) {
        return detail::set_error(Status::kInvalidArgument,
                                 "compute_inv_freq: kDynamicNTK needs original_max_position > 0");
      }
      if (dim <= 2) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "compute_inv_freq: kDynamicNTK needs rotary_dim > 2");
      }
      const double length = L0 * factor;
      base = theta * std::pow(factor * length / L0 - (factor - 1.0), double(dim) / (dim - 2));
      break;
    }
    case RopeScaling::kNTKAlpha:
      if (dim <= 2) {
        return detail::set_error(Status::kUnsupportedShape,
                                 "compute_inv_freq: kNTKAlpha needs rotary_dim > 2");
      }
      base = theta * std::pow(factor, double(dim) / (dim - 2));
      break;
    case RopeScaling::kYaRN:
      if (!(L0 > 0.0)) {
        return detail::set_error(Status::kInvalidArgument,
                                 "compute_inv_freq: kYaRN needs original_max_position > 0");
      }
      if (!(cfg.beta_fast > 0.0f) || !(cfg.beta_slow > 0.0f)) {
        return detail::set_error(Status::kInvalidArgument,
                                 "compute_inv_freq: beta_fast and beta_slow must be > 0");
      }
      break;
    case RopeScaling::kLlama3:
      if (!(L0 > 0.0)) {
        return detail::set_error(Status::kInvalidArgument,
                                 "compute_inv_freq: kLlama3 needs original_max_position > 0");
      }
      if (!(cfg.low_freq_factor > 0.0f) || !(cfg.high_freq_factor > 0.0f)) {
        return detail::set_error(Status::kInvalidArgument,
                                 "compute_inv_freq: low_freq_factor and high_freq_factor must be > 0");
      }
      break;
    case RopeScaling::kLongRoPE: {
      if (!(L0 > 1.0)) {
        return detail::set_error(Status::kInvalidArgument,
                                 "compute_inv_freq: kLongRoPE needs original_max_position > 1");
      }
      // RopeConfig has no use_long_factor / seq_len switch: when both tables are
      // present the long (extended-context) table wins, otherwise the one that is
      // present is used. Pass only short_factor for the base phase.
      if (cfg.long_factor == nullptr && cfg.short_factor == nullptr) {
        return detail::set_error(Status::kInvalidArgument,
                                 "compute_inv_freq: kLongRoPE needs short_factor or long_factor");
      }
      break;
    }
    default:
      return detail::set_error(Status::kInvalidArgument, "compute_inv_freq: unknown RopeScaling %d",
                               static_cast<int>(cfg.scaling));
  }

  // pos_freq[i] = theta^(2i/dim) (the unscaled wavelength ratio).
  double pos_freq[kMaxRotDim / 2];
  for (int i = 0; i < half; ++i) pos_freq[i] = std::pow(base, double(2 * i) / dim);

  switch (cfg.scaling) {
    case RopeScaling::kNone:
    case RopeScaling::kDynamicNTK:
    case RopeScaling::kNTKAlpha:
      for (int i = 0; i < half; ++i) pos_freq[i] = 1.0 / pos_freq[i];
      break;
    case RopeScaling::kLinear:
      for (int i = 0; i < half; ++i) pos_freq[i] = 1.0 / pos_freq[i] / factor;
      break;
    case RopeScaling::kYaRN: {
      double low = yarn_find_correction_dim(double(cfg.beta_fast), dim, theta, L0);
      double high = yarn_find_correction_dim(double(cfg.beta_slow), dim, theta, L0);
      if (cfg.truncate) {
        low = std::floor(low);
        high = std::ceil(high);
      }
      low = std::max(low, 0.0);
      high = std::min(high, double(dim - 1));
      if (low == high) high += 0.001;  // keeps the ramp finite
      for (int i = 0; i < half; ++i) {
        const double ramp = std::min(std::max((double(i) - low) / (high - low), 0.0), 1.0);
        const double extrapolation_mask = 1.0 - ramp;
        const double extrapolation = 1.0 / pos_freq[i];
        const double interpolation = 1.0 / (factor * pos_freq[i]);
        pos_freq[i] = interpolation * (1.0 - extrapolation_mask) + extrapolation * extrapolation_mask;
      }
      mscale = yarn_get_mscale(factor, double(cfg.mscale)) /
               yarn_get_mscale(factor, double(cfg.mscale_all_dim));
      break;
    }
    case RopeScaling::kLlama3: {
      const double low_wavelen = L0 / double(cfg.low_freq_factor);
      const double high_wavelen = L0 / double(cfg.high_freq_factor);
      for (int i = 0; i < half; ++i) {
        const double inv = 1.0 / pos_freq[i];
        const double wavelen = kTwoPi / inv;
        const double smooth = cfg.low_freq_factor != cfg.high_freq_factor
                                  ? (L0 / wavelen - double(cfg.low_freq_factor)) /
                                        (double(cfg.high_freq_factor) - double(cfg.low_freq_factor))
                                  : 0.0;
        pos_freq[i] = wavelen < high_wavelen  ? inv
                      : wavelen > low_wavelen ? inv / factor
                                              : (1.0 - smooth) * inv / factor + smooth * inv;
      }
      break;
    }
    case RopeScaling::kLongRoPE: {
      const float* table = cfg.long_factor != nullptr ? cfg.long_factor : cfg.short_factor;
      for (int i = 0; i < half; ++i) pos_freq[i] = 1.0 / (double(table[i]) * pos_freq[i]);
      mscale = factor > 1.0 ? std::sqrt(1.0 + std::log(factor) / std::log(L0)) : 1.0;
      break;
    }
    default:
      return detail::set_error(Status::kInvalidArgument, "compute_inv_freq: unknown RopeScaling %d",
                               static_cast<int>(cfg.scaling));
  }
  if (cfg.attention_factor != 0.0f) mscale = double(cfg.attention_factor);

  for (int i = 0; i < half; ++i) {
    if (!std::isfinite(pos_freq[i]) || pos_freq[i] == 0.0) {
      return detail::set_error(Status::kInvalidArgument,
                               "compute_inv_freq: configuration yields a nonfinite frequency at %d",
                               i);
    }
  }
  if (!std::isfinite(mscale)) {
    return detail::set_error(Status::kInvalidArgument,
                             "compute_inv_freq: configuration yields a nonfinite mscale");
  }
  for (int i = 0; i < half; ++i) inv_freq_host[i] = static_cast<float>(pos_freq[i]);
  *cos_sin_mscale = static_cast<float>(mscale);
  return Status::kSuccess;
}

Status compute_alibi_slopes(int32_t num_heads, float max_bias, float* out_host) {
  if (out_host == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "compute_alibi_slopes: null output pointer");
  }
  if (num_heads <= 0) {
    return detail::set_error(Status::kInvalidArgument, "compute_alibi_slopes: num_heads %d <= 0",
                             num_heads);
  }
  if (!std::isfinite(max_bias) || !(max_bias > 0.0f)) {
    return detail::set_error(Status::kInvalidArgument,
                             "compute_alibi_slopes: max_bias %g must be > 0 and finite",
                             static_cast<double>(max_bias));
  }
  int n = 1;
  while (n <= num_heads / 2) n *= 2;  // largest power of two <= num_heads
  const double m0 = std::pow(2.0, -double(max_bias) / n);
  const double m1 = std::pow(2.0, -double(max_bias) / 2.0 / n);
  for (int h = 0; h < num_heads; ++h) {
    out_host[h] = static_cast<float>(h < n ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n) + 1));
  }
  return Status::kSuccess;
}

// =============================================================================
// Kernels (declared in position/rope_kernels.cuh)
// =============================================================================

template <typename cache_t>
__global__ void build_cos_sin_cache_kernel(
    cache_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const InvFreqTable inv_freq,          // [rot_dim / 2], by value
    const float mscale, const int max_position, const int rot_dim) {
  const int half = rot_dim / 2;
  const double scale = static_cast<double>(mscale);
  for (int k = 0; k < kCachePositionsPerCta; ++k) {
    const int position = blockIdx.x * kCachePositionsPerCta + k;
    if (position >= max_position) return;
    cache_t* row = cos_sin_cache + static_cast<int64_t>(position) * rot_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      double s = 0.0, c = 0.0;
      // fp64 angle: an fp32 angle loses whole radians at 1e6 positions.
      sincos(static_cast<double>(position) * static_cast<double>(inv_freq.v[i]), &s, &c);
      dev::from_f<cache_t>(row[i], static_cast<float>(scale * c));
      dev::from_f<cache_t>(row[half + i], static_cast<float>(scale * s));
    }
  }
}

template <typename scalar_t, typename cache_t, RopeStyle kStyle, bool kOnTheFly>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,     // [num_tokens]
    scalar_t* __restrict__ query,              // [num_tokens, num_heads, head_size] (strided)
    scalar_t* __restrict__ key,                // [num_tokens, num_kv_heads, head_size] or null
    const cache_t* __restrict__ cos_sin_cache, // [max_position, rot_dim]; null if kOnTheFly
    const float* __restrict__ inv_freq,        // [rot_dim / 2]; used if kOnTheFly
    const int64_t* __restrict__ cos_sin_cache_offsets,  // [num_tokens] or null
    const float on_the_fly_mscale, const int rot_dim, const int rot_offset,
    const int64_t query_token_stride, const int64_t query_head_stride,
    const int64_t key_token_stride, const int64_t key_head_stride, const int num_heads,
    const int num_kv_heads, const int head_size, const int num_tokens) {
  const int token = blockIdx.x;
  if (token >= num_tokens) return;
  const int half = rot_dim / 2;
  // On-the-fly: the token's cos/sin is built once per CTA and shared by its heads.
  __shared__ float shared_cs[kOnTheFly ? kMaxRotDim : 1];
  const cache_t* row = nullptr;
  if (kOnTheFly) {
    const int64_t position = positions[token];
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
      double s = 0.0, c = 0.0;
      sincos(static_cast<double>(position) * static_cast<double>(inv_freq[i]), &s, &c);
      const double m = static_cast<double>(on_the_fly_mscale);
      shared_cs[i] = static_cast<float>(m * c);
      shared_cs[half + i] = static_cast<float>(m * s);
    }
    __syncthreads();
  } else {
    const int64_t position = positions[token];
    const int64_t offset = cos_sin_cache_offsets != nullptr ? cos_sin_cache_offsets[token] : 0;
    row = cos_sin_cache + (position + offset) * static_cast<int64_t>(rot_dim);
    (void)inv_freq;
    (void)on_the_fly_mscale;
  }
  (void)head_size;
  const int64_t q_base = static_cast<int64_t>(token) * query_token_stride + rot_offset;
  const int64_t k_base = static_cast<int64_t>(token) * key_token_stride + rot_offset;
  const int head_count = num_heads + num_kv_heads;
  // One (head, pair) item per thread step: the whole block stays busy even when
  // the rotary window is narrower than the block (decode with partial rotary).
  const int items = head_count * half;
  for (int w = threadIdx.x; w < items; w += blockDim.x) {
    const int h = w / half;
    const int i = w - h * half;
    scalar_t* base = (h < num_heads)
                         ? query + q_base + static_cast<int64_t>(h) * query_head_stride
                         : key + k_base + static_cast<int64_t>(h - num_heads) * key_head_stride;
    const float c = kOnTheFly ? shared_cs[i] : dev::to_f<cache_t>(row[i]);
    const float s = kOnTheFly ? shared_cs[half + i] : dev::to_f<cache_t>(row[half + i]);
    int a = 0, b = 0;
    pair_offsets<kStyle>(i, half, a, b);
    float oa = 0.0f, ob = 0.0f;
    rotate_pair(dev::to_f<scalar_t>(base[a]), dev::to_f<scalar_t>(base[b]), c, s, oa, ob);
    dev::from_f<scalar_t>(base[a], oa);
    dev::from_f<scalar_t>(base[b], ob);
  }
}

template <typename scalar_t, typename cache_t, RopeStyle kStyle, bool kInterleaved>
__global__ void mrope_kernel(
    const int64_t* __restrict__ positions,     // [3, num_tokens]
    scalar_t* __restrict__ query,              // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                // [num_tokens, num_kv_heads, head_size] or null
    const cache_t* __restrict__ cos_sin_cache, // [max_position, rot_dim]
    const int mrope_section_t, const int mrope_section_h, const int mrope_section_w,
    const int rot_dim, const int64_t query_token_stride, const int64_t query_head_stride,
    const int64_t key_token_stride, const int64_t key_head_stride, const int num_heads,
    const int num_kv_heads, const int head_size, const int num_tokens) {
  const int token = blockIdx.x;
  if (token >= num_tokens) return;
  const int half = rot_dim / 2;
  (void)head_size;
  const int64_t q_base = static_cast<int64_t>(token) * query_token_stride;
  const int64_t k_base = static_cast<int64_t>(token) * key_token_stride;
  const int head_count = num_heads + num_kv_heads;
  const int items = head_count * half;
  for (int w = threadIdx.x; w < items; w += blockDim.x) {
    const int h = w / half;
    const int i = w - h * half;
    scalar_t* base = (h < num_heads)
                         ? query + q_base + static_cast<int64_t>(h) * query_head_stride
                         : key + k_base + static_cast<int64_t>(h - num_heads) * key_head_stride;
    const int axis =
        mrope_axis<kInterleaved>(i, mrope_section_t, mrope_section_h, mrope_section_w);
    // positions are [3, num_tokens] row-major: row `axis` starts at axis * num_tokens.
    const int64_t r = positions[static_cast<int64_t>(axis) * num_tokens + token];
    const cache_t* row = cos_sin_cache + r * static_cast<int64_t>(rot_dim);
    int a = 0, b = 0;
    pair_offsets<kStyle>(i, half, a, b);
    float oa = 0.0f, ob = 0.0f;
    rotate_pair(dev::to_f<scalar_t>(base[a]), dev::to_f<scalar_t>(base[b]),
                dev::to_f<cache_t>(row[i]), dev::to_f<cache_t>(row[half + i]), oa, ob);
    dev::from_f<scalar_t>(base[a], oa);
    dev::from_f<scalar_t>(base[b], ob);
  }
}

template <typename scalar_t, typename cache_t, int HEAD_DIM, RopeStyle kStyle>
__global__ void fused_qk_norm_rope_kernel(
    scalar_t* __restrict__ qkv,                // [num_tokens, (Hq + 2*Hkv) * HEAD_DIM]
    const int64_t token_stride,                // qkv.stride(0), >= (Hq + 2*Hkv) * HEAD_DIM
    const scalar_t* __restrict__ q_weight,     // [HEAD_DIM] or null
    const scalar_t* __restrict__ k_weight,     // [HEAD_DIM] or null
    const int64_t* __restrict__ positions,     // [num_tokens]
    const cache_t* __restrict__ cos_sin_cache, // [max_position, rot_dim]
    const float weight_offset, const float epsilon, const int rot_dim, const int rot_offset,
    const int num_tokens, const int num_q_heads, const int num_kv_heads) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256,
                "fused_qk_norm_rope_kernel is instantiated for head_dim 64 / 128 / 256");
  constexpr int kPerLane = HEAD_DIM / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int heads = num_q_heads + num_kv_heads;
  const int64_t total = static_cast<int64_t>(num_tokens) * heads;
  const int64_t task = static_cast<int64_t>(blockIdx.x) * kFusedWarpsPerCta + warp;
  // One warp stages one head's normalized rotary window; V heads never touch it.
  __shared__ float stage[kFusedWarpsPerCta][kMaxFusedHeadDim];
  if (task >= total || heads <= 0) return;  // uniform per warp
  const int token = static_cast<int>(task / heads);
  const int head = static_cast<int>(task - static_cast<int64_t>(token) * heads);
  const int half = rot_dim / 2;
  const int lo = rot_offset;
  const int hi = rot_offset + rot_dim;
  scalar_t* ptr = qkv + static_cast<int64_t>(token) * token_stride +
                  static_cast<int64_t>(head) * HEAD_DIM;
  const scalar_t* weight = head < num_q_heads ? q_weight : k_weight;
  const cache_t* row = cos_sin_cache + positions[token] * static_cast<int64_t>(rot_dim);
  float* tile = stage[warp];

  float x[kPerLane];
  float ss = 0.0f;
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) {
    x[c] = dev::to_f<scalar_t>(ptr[lane + 32 * c]);
    ss = __fmaf_rn(x[c], x[c], ss);
  }
  ss = warp_reduce_sum(ss);
  const float rstd = rsqrtf(__fmaf_rn(ss, 1.0f / static_cast<float>(HEAD_DIM), epsilon));
  // Normalize the whole head; dims inside the rotary window wait in shared memory
  // for their partner, the rest is final.
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) {
    const int j = lane + 32 * c;
    float v = x[c] * rstd;
    if (weight != nullptr) v *= dev::to_f<scalar_t>(weight[j]) + weight_offset;
    if (j >= lo && j < hi) {
      tile[j - lo] = v;
    } else {
      dev::from_f<scalar_t>(ptr[j], v);
    }
  }
  __syncwarp();
  for (int i = lane; i < half; i += 32) {
    int a = 0, b = 0;
    pair_offsets<kStyle>(i, half, a, b);
    float oa = 0.0f, ob = 0.0f;
    rotate_pair(tile[a], tile[b], dev::to_f<cache_t>(row[i]), dev::to_f<cache_t>(row[half + i]), oa,
                ob);
    dev::from_f<scalar_t>(ptr[lo + a], oa);
    dev::from_f<scalar_t>(ptr[lo + b], ob);
  }
}

template <typename scalar_t>
__global__ void attn_temperature_tuning_kernel(
    scalar_t* __restrict__ query,             // [num_tokens, num_heads, head_size] (heads contiguous)
    const int64_t query_token_stride,         // query.stride(0)
    const int64_t* __restrict__ positions,    // [num_tokens]
    const float floor_scale, const float attn_scale, const int num_tokens, const int num_heads,
    const int head_size) {
  const int token = blockIdx.x;
  if (token >= num_tokens) return;
  __shared__ float factor;
  __shared__ int active;
  if (threadIdx.x == 0) {
    const int64_t position = positions[token];
    active = position >= 0 ? 1 : 0;
    // s(p) = log(floor((p + 1) / floor_scale) + 1) * attn_scale + 1, fp64 once per token.
    const double s = ::log(::floor((static_cast<double>(position) + 1.0) /
                                   static_cast<double>(floor_scale)) +
                           1.0) *
                         static_cast<double>(attn_scale) +
                     1.0;
    factor = static_cast<float>(s);
  }
  __syncthreads();
  if (active == 0) return;  // negative position (padding): leave the token alone
  scalar_t* base = query + static_cast<int64_t>(token) * query_token_stride;
  const int64_t total = static_cast<int64_t>(num_heads) * head_size;
  for (int64_t w = threadIdx.x; w < total; w += blockDim.x) {
    dev::from_f<scalar_t>(base[w], dev::to_f<scalar_t>(base[w]) * factor);
  }
}

// =============================================================================
// Launchers
// =============================================================================

namespace {

/// Geometry of one apply_rotary / apply_rotary_on_the_fly / apply_mrope call.
struct RotaryCall {
  const int64_t* positions = nullptr;
  void* query = nullptr;
  void* key = nullptr;
  const void* cache = nullptr;
  const float* inv_freq = nullptr;
  const int64_t* cache_offsets = nullptr;
  float mscale = 1.0f;
  int32_t rot_dim = 0;
  int32_t rot_offset = 0;
  int64_t q_token_stride = 0;
  int64_t q_head_stride = 0;
  int64_t k_token_stride = 0;
  int64_t k_head_stride = 0;
  int32_t num_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_size = 0;
  int32_t num_tokens = 0;
};

/// grid = num_tokens, block = min(512, max(num_heads, num_kv_heads) * rot_dim / 2).
int rotary_block(const RotaryCall& c) {
  const int32_t heads = std::max(c.num_heads, c.num_kv_heads);
  int64_t items = static_cast<int64_t>(heads) * (c.rot_dim / 2);
  if (items < 32) items = 32;
  if (items > kMaxApplyBlock) items = kMaxApplyBlock;
  return static_cast<int>(((items + 31) / 32) * 32);
}

template <typename scalar_t, typename cache_t, RopeStyle kStyle, bool kOnTheFly>
Status launch_rotary(const RotaryCall& c, const LaunchContext& ctx) {
  if (c.num_tokens <= 0 || c.num_heads + c.num_kv_heads <= 0) return Status::kSuccess;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  rotary_embedding_kernel<scalar_t, cache_t, kStyle, kOnTheFly>
      <<<static_cast<unsigned>(c.num_tokens), rotary_block(c), 0, stream>>>(
          c.positions, static_cast<scalar_t*>(c.query), static_cast<scalar_t*>(c.key),
          kOnTheFly ? nullptr : static_cast<const cache_t*>(c.cache),
          kOnTheFly ? c.inv_freq : nullptr, kOnTheFly ? nullptr : c.cache_offsets, c.mscale,
          c.rot_dim, c.rot_offset, c.q_token_stride, c.q_head_stride, c.k_token_stride,
          c.k_head_stride, c.num_heads, c.num_kv_heads, c.head_size, c.num_tokens);
  return check_launch("apply_rotary");
}

template <typename scalar_t, typename cache_t, RopeStyle kStyle, bool kInterleaved>
Status launch_mrope(const RotaryCall& c, const int32_t section[3], const LaunchContext& ctx) {
  if (c.num_tokens <= 0 || c.num_heads + c.num_kv_heads <= 0) return Status::kSuccess;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  mrope_kernel<scalar_t, cache_t, kStyle, kInterleaved>
      <<<static_cast<unsigned>(c.num_tokens), rotary_block(c), 0, stream>>>(
          c.positions, static_cast<scalar_t*>(c.query), static_cast<scalar_t*>(c.key),
          static_cast<const cache_t*>(c.cache), section[0], section[1], section[2], c.rot_dim,
          c.q_token_stride, c.q_head_stride, c.k_token_stride, c.k_head_stride, c.num_heads,
          c.num_kv_heads, c.head_size, c.num_tokens);
  return check_launch("apply_mrope");
}

template <typename scalar_t, typename cache_t>
Status dispatch_rotary(const RotaryCall& c, RopeStyle style, bool on_the_fly,
                       const LaunchContext& ctx) {
  if (style == RopeStyle::kNeoX) {
    return on_the_fly ? launch_rotary<scalar_t, cache_t, RopeStyle::kNeoX, true>(c, ctx)
                      : launch_rotary<scalar_t, cache_t, RopeStyle::kNeoX, false>(c, ctx);
  }
  return on_the_fly ? launch_rotary<scalar_t, cache_t, RopeStyle::kGPTJ, true>(c, ctx)
                    : launch_rotary<scalar_t, cache_t, RopeStyle::kGPTJ, false>(c, ctx);
}

template <typename scalar_t>
Status dispatch_rotary_cache(const RotaryCall& c, RopeStyle style, bool on_the_fly,
                             DType cache_dtype, const LaunchContext& ctx) {
  if (on_the_fly || cache_dtype == DType::kFloat32) {
    return dispatch_rotary<scalar_t, float>(c, style, on_the_fly, ctx);
  }
  return dispatch_rotary<scalar_t, scalar_t>(c, style, on_the_fly, ctx);
}

template <typename scalar_t, typename cache_t>
Status dispatch_mrope(const RotaryCall& c, const int32_t section[3], bool interleaved,
                      RopeStyle style, const LaunchContext& ctx) {
  if (style == RopeStyle::kNeoX) {
    return interleaved ? launch_mrope<scalar_t, cache_t, RopeStyle::kNeoX, true>(c, section, ctx)
                       : launch_mrope<scalar_t, cache_t, RopeStyle::kNeoX, false>(c, section, ctx);
  }
  return interleaved ? launch_mrope<scalar_t, cache_t, RopeStyle::kGPTJ, true>(c, section, ctx)
                     : launch_mrope<scalar_t, cache_t, RopeStyle::kGPTJ, false>(c, section, ctx);
}

template <typename scalar_t>
Status dispatch_mrope_cache(const RotaryCall& c, const int32_t section[3], bool interleaved,
                            RopeStyle style, DType cache_dtype, const LaunchContext& ctx) {
  if (cache_dtype == DType::kFloat32) {
    return dispatch_mrope<scalar_t, float>(c, section, interleaved, style, ctx);
  }
  return dispatch_mrope<scalar_t, scalar_t>(c, section, interleaved, style, ctx);
}

template <typename Fn>
Status with_scalar_dtype(DType dt, const char* op, Fn&& fn) {
  switch (dt) {
    case DType::kFloat32:
      return fn(ScalarTag<float>{});
    case DType::kFloat16:
      return fn(ScalarTag<__half>{});
    case DType::kBFloat16:
      return fn(ScalarTag<__nv_bfloat16>{});
    default:
      return detail::set_error(Status::kUnsupportedDtype,
                               "%s: dtype %s (fp32/fp16/bf16 expected)", op, dtype_name(dt));
  }
}

/// The kernel enumerates (head, pair) items in a 32-bit index.
Status check_item_count(int32_t num_heads, int32_t num_kv_heads, int32_t rot_dim, const char* op) {
  const int64_t items = static_cast<int64_t>(num_heads + num_kv_heads) * (rot_dim / 2);
  if (items > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: (num_heads + num_kv_heads) * rot_dim/2 = %lld exceeds INT32_MAX",
                             op, static_cast<long long>(items));
  }
  return Status::kSuccess;
}

/// Shared validation of the [T, Hq, D] / optional [T, Hkv, D] pair.
Status check_head_pair(const Tensor& query, const Tensor& key, int32_t rot_end, const char* op,
                       int32_t* num_heads, int32_t* num_kv_heads) {
  Status s = check_heads(query, "query", rot_end);
  if (s != Status::kSuccess) return s;
  *num_heads = static_cast<int32_t>(query.shape[1]);
  *num_kv_heads = 0;
  if (!tensor_present(key)) return Status::kSuccess;
  s = check_heads(key, "key", rot_end);
  if (s != Status::kSuccess) return s;
  if (key.dtype != query.dtype) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: key dtype %s != query dtype %s", op,
                             dtype_name(key.dtype), dtype_name(query.dtype));
  }
  if (key.shape[0] != query.shape[0]) {
    return detail::set_error(Status::kInvalidArgument, "%s: key has %lld tokens, query has %lld", op,
                             static_cast<long long>(key.shape[0]),
                             static_cast<long long>(query.shape[0]));
  }
  if (key.shape[2] != query.shape[2]) {
    return detail::set_error(Status::kUnsupportedShape, "%s: key head size %lld != query %lld", op,
                             static_cast<long long>(key.shape[2]),
                             static_cast<long long>(query.shape[2]));
  }
  s = check_same_device(key, query, "key/query");
  if (s != Status::kSuccess) return s;
  *num_kv_heads = static_cast<int32_t>(key.shape[1]);
  return Status::kSuccess;
}

}  // namespace

// ---- apply_rotary ----------------------------------------------------------

Status apply_rotary(const Tensor& positions, const Tensor& query, const Tensor& key,
                    const Tensor& cos_sin_cache, const Tensor& cache_offsets, int32_t rot_dim,
                    int32_t rot_offset, RopeStyle style, const LaunchContext& ctx) {
  const char* op = "apply_rotary";
  Status s = check_launch_context(ctx, op);
  if (s != Status::kSuccess) return s;
  s = check_style(style, op);
  if (s != Status::kSuccess) return s;
  s = check_rope_window(rot_dim, rot_offset, op);
  if (s != Status::kSuccess) return s;
  if (!tensor_present(query)) {
    return detail::set_error(Status::kInvalidArgument, "%s: query is null", op);
  }
  s = check_device(query, ctx, "query");
  if (s != Status::kSuccess) return s;
  int32_t num_heads = 0, num_kv_heads = 0;
  s = check_head_pair(query, key, rot_offset + rot_dim, op, &num_heads, &num_kv_heads);
  if (s != Status::kSuccess) return s;
  s = check_item_count(num_heads, num_kv_heads, rot_dim, op);
  if (s != Status::kSuccess) return s;
  const int64_t tokens = query.shape[0];
  s = check_positions(positions, "positions", tokens);
  if (s != Status::kSuccess) return s;
  s = check_same_device(positions, query, "positions/query");
  if (s != Status::kSuccess) return s;
  s = check_cos_sin_cache(cos_sin_cache, query.dtype, rot_dim, "cos_sin_cache");
  if (s != Status::kSuccess) return s;
  s = check_same_device(cos_sin_cache, query, "cos_sin_cache/query");
  if (s != Status::kSuccess) return s;

  const int64_t* offsets = nullptr;
  if (tensor_present(cache_offsets)) {
    if (cache_offsets.dtype != DType::kInt64) {
      return detail::set_error(Status::kUnsupportedDtype, "%s: cache_offsets dtype %s (int64)", op,
                               dtype_name(cache_offsets.dtype));
    }
    s = check_tensor(cache_offsets, DType::kInt64, 1, 8, "cache_offsets");
    if (s != Status::kSuccess) return s;
    if (cache_offsets.shape[0] != tokens) {
      return detail::set_error(Status::kUnsupportedShape,
                               "%s: cache_offsets has %lld entries, expected %lld", op,
                               static_cast<long long>(cache_offsets.shape[0]),
                               static_cast<long long>(tokens));
    }
    if (tokens > 1 && cache_offsets.stride[0] != 1) {
      return detail::set_error(Status::kUnsupportedLayout, "%s: cache_offsets stride %lld != 1", op,
                               static_cast<long long>(cache_offsets.stride[0]));
    }
    s = check_same_device(cache_offsets, query, "cache_offsets/query");
    if (s != Status::kSuccess) return s;
    offsets = static_cast<const int64_t*>(const_data(cache_offsets));
  }

  RotaryCall call{};
  call.positions = static_cast<const int64_t*>(const_data(positions));
  call.query = mutable_data(query);
  call.key = tensor_present(key) ? mutable_data(key) : nullptr;
  call.cache = const_data(cos_sin_cache);
  call.cache_offsets = offsets;
  call.rot_dim = rot_dim;
  call.rot_offset = rot_offset;
  call.q_token_stride = query.stride[0];
  call.q_head_stride = query.stride[1];
  call.num_heads = num_heads;
  call.num_kv_heads = num_kv_heads;
  call.head_size = static_cast<int32_t>(query.shape[2]);
  call.num_tokens = static_cast<int32_t>(tokens);
  if (tensor_present(key)) {
    call.k_token_stride = key.stride[0];
    call.k_head_stride = key.stride[1];
  }
  const DType cache_dtype = cos_sin_cache.dtype;
  return with_scalar_dtype(query.dtype, op, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return dispatch_rotary_cache<T>(call, style, false, cache_dtype, ctx);
  });
}

// ---- apply_rotary_on_the_fly ----------------------------------------------

Status apply_rotary_on_the_fly(const Tensor& positions, const Tensor& query, const Tensor& key,
                               const Tensor& inv_freq, float mscale, int32_t rot_dim,
                               int32_t rot_offset, RopeStyle style, const LaunchContext& ctx) {
  const char* op = "apply_rotary_on_the_fly";
  Status s = check_launch_context(ctx, op);
  if (s != Status::kSuccess) return s;
  s = check_style(style, op);
  if (s != Status::kSuccess) return s;
  s = check_rope_window(rot_dim, rot_offset, op);
  if (s != Status::kSuccess) return s;
  if (!finite(mscale)) {
    return detail::set_error(Status::kInvalidArgument, "%s: mscale must be finite", op);
  }
  if (!tensor_present(query)) {
    return detail::set_error(Status::kInvalidArgument, "%s: query is null", op);
  }
  s = check_device(query, ctx, "query");
  if (s != Status::kSuccess) return s;
  int32_t num_heads = 0, num_kv_heads = 0;
  s = check_head_pair(query, key, rot_offset + rot_dim, op, &num_heads, &num_kv_heads);
  if (s != Status::kSuccess) return s;
  s = check_item_count(num_heads, num_kv_heads, rot_dim, op);
  if (s != Status::kSuccess) return s;
  const int64_t tokens = query.shape[0];
  s = check_positions(positions, "positions", tokens);
  if (s != Status::kSuccess) return s;
  s = check_same_device(positions, query, "positions/query");
  if (s != Status::kSuccess) return s;
  if (inv_freq.dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: inv_freq dtype %s (float32)", op,
                             dtype_name(inv_freq.dtype));
  }
  s = check_tensor(inv_freq, DType::kFloat32, 1, 4, "inv_freq");
  if (s != Status::kSuccess) return s;
  if (inv_freq.shape[0] != rot_dim / 2) {
    return detail::set_error(Status::kUnsupportedShape, "%s: inv_freq has %lld entries, expected %d",
                             op, static_cast<long long>(inv_freq.shape[0]), rot_dim / 2);
  }
  if (inv_freq.shape[0] > 1 && inv_freq.stride[0] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: inv_freq stride %lld != 1", op,
                             static_cast<long long>(inv_freq.stride[0]));
  }
  s = check_same_device(inv_freq, query, "inv_freq/query");
  if (s != Status::kSuccess) return s;

  RotaryCall call{};
  call.positions = static_cast<const int64_t*>(const_data(positions));
  call.query = mutable_data(query);
  call.key = tensor_present(key) ? mutable_data(key) : nullptr;
  call.inv_freq = static_cast<const float*>(const_data(inv_freq));
  call.mscale = mscale;
  call.rot_dim = rot_dim;
  call.rot_offset = rot_offset;
  call.q_token_stride = query.stride[0];
  call.q_head_stride = query.stride[1];
  call.num_heads = num_heads;
  call.num_kv_heads = num_kv_heads;
  call.head_size = static_cast<int32_t>(query.shape[2]);
  call.num_tokens = static_cast<int32_t>(tokens);
  if (tensor_present(key)) {
    call.k_token_stride = key.stride[0];
    call.k_head_stride = key.stride[1];
  }
  return with_scalar_dtype(query.dtype, op, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return dispatch_rotary_cache<T>(call, style, true, DType::kFloat32, ctx);
  });
}

// ---- apply_mrope ----------------------------------------------------------

Status apply_mrope(const Tensor& positions_thw, const Tensor& query, const Tensor& key,
                   const Tensor& cos_sin_cache, const int32_t mrope_section[3], bool interleaved,
                   RopeStyle style, const LaunchContext& ctx) {
  const char* op = "apply_mrope";
  Status s = check_launch_context(ctx, op);
  if (s != Status::kSuccess) return s;
  s = check_style(style, op);
  if (s != Status::kSuccess) return s;
  if (!tensor_present(query)) {
    return detail::set_error(Status::kInvalidArgument, "%s: query is null", op);
  }
  s = check_device(query, ctx, "query");
  if (s != Status::kSuccess) return s;
  if (mrope_section == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "%s: mrope_section is null", op);
  }
  // rot_dim is the cache width; M-RoPE always rotates from dimension 0.
  if (cos_sin_cache.ndim != 2) {
    return detail::set_error(Status::kInvalidArgument, "%s: cos_sin_cache must be 2-D", op);
  }
  if (cos_sin_cache.shape[1] < 2 || cos_sin_cache.shape[1] > kMaxRotDim ||
      (cos_sin_cache.shape[1] % 2) != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: cos_sin_cache width %lld must be even and in [2, %d]", op,
                             static_cast<long long>(cos_sin_cache.shape[1]), kMaxRotDim);
  }
  const int32_t rot_dim = static_cast<int32_t>(cos_sin_cache.shape[1]);
  if (mrope_section[0] < 0 || mrope_section[1] < 0 || mrope_section[2] < 0 ||
      static_cast<int64_t>(mrope_section[0]) + mrope_section[1] + mrope_section[2] !=
          rot_dim / 2) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: mrope_section must be nonnegative and sum to rot_dim / 2 = %d",
                             op, rot_dim / 2);
  }
  int32_t num_heads = 0, num_kv_heads = 0;
  s = check_head_pair(query, key, rot_dim, op, &num_heads, &num_kv_heads);
  if (s != Status::kSuccess) return s;
  s = check_item_count(num_heads, num_kv_heads, rot_dim, op);
  if (s != Status::kSuccess) return s;
  const int64_t tokens = query.shape[0];
  if (positions_thw.dtype != DType::kInt64) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: positions_thw dtype %s (int64)", op,
                             dtype_name(positions_thw.dtype));
  }
  s = check_tensor(positions_thw, DType::kInt64, 2, 8, "positions_thw");
  if (s != Status::kSuccess) return s;
  if (positions_thw.shape[0] != 3 || positions_thw.shape[1] != tokens) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: positions_thw must be [3, %lld], got [%lld, %lld]", op,
                             static_cast<long long>(tokens),
                             static_cast<long long>(positions_thw.shape[0]),
                             static_cast<long long>(positions_thw.shape[1]));
  }
  // The kernel indexes positions[axis * num_tokens + token]: rows must be contiguous.
  if ((tokens > 1 && positions_thw.stride[1] != 1) || positions_thw.stride[0] != tokens) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "%s: positions_thw must be a contiguous [3, T] tensor (rows stride T)",
                             op);
  }
  s = check_same_device(positions_thw, query, "positions_thw/query");
  if (s != Status::kSuccess) return s;
  s = check_cos_sin_cache(cos_sin_cache, query.dtype, rot_dim, "cos_sin_cache");
  if (s != Status::kSuccess) return s;
  s = check_same_device(cos_sin_cache, query, "cos_sin_cache/query");
  if (s != Status::kSuccess) return s;

  RotaryCall call{};
  call.positions = static_cast<const int64_t*>(const_data(positions_thw));
  call.query = mutable_data(query);
  call.key = tensor_present(key) ? mutable_data(key) : nullptr;
  call.cache = const_data(cos_sin_cache);
  call.rot_dim = rot_dim;
  call.rot_offset = 0;
  call.q_token_stride = query.stride[0];
  call.q_head_stride = query.stride[1];
  call.num_heads = num_heads;
  call.num_kv_heads = num_kv_heads;
  call.head_size = static_cast<int32_t>(query.shape[2]);
  call.num_tokens = static_cast<int32_t>(tokens);
  if (tensor_present(key)) {
    call.k_token_stride = key.stride[0];
    call.k_head_stride = key.stride[1];
  }
  const DType cache_dtype = cos_sin_cache.dtype;
  return with_scalar_dtype(query.dtype, op, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return dispatch_mrope_cache<T>(call, mrope_section, interleaved, style, cache_dtype, ctx);
  });
}

// ---- fused_qk_norm_rope ---------------------------------------------------

namespace {

struct FusedCall {
  void* qkv = nullptr;
  const void* q_weight = nullptr;
  const void* k_weight = nullptr;
  const int64_t* positions = nullptr;
  const void* cache = nullptr;
  int64_t token_stride = 0;
  float weight_offset = 0.0f;
  float epsilon = 0.0f;
  int32_t rot_dim = 0;
  int32_t rot_offset = 0;
  int32_t num_tokens = 0;
  int32_t num_q_heads = 0;
  int32_t num_kv_heads = 0;
};

template <typename scalar_t, typename cache_t, RopeStyle kStyle, int kHeadDim>
Status launch_fused(const FusedCall& c, const LaunchContext& ctx) {
  const int64_t tasks =
      static_cast<int64_t>(c.num_tokens) * (c.num_q_heads + c.num_kv_heads);
  if (tasks <= 0) return Status::kSuccess;
  const int64_t grid_exact = (tasks + kFusedWarpsPerCta - 1) / kFusedWarpsPerCta;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  fused_qk_norm_rope_kernel<scalar_t, cache_t, kHeadDim, kStyle>
      <<<static_cast<unsigned>(grid_exact), kFusedThreads, 0, stream>>>(
          static_cast<scalar_t*>(c.qkv), c.token_stride,
          static_cast<const scalar_t*>(c.q_weight), static_cast<const scalar_t*>(c.k_weight),
          c.positions, static_cast<const cache_t*>(c.cache), c.weight_offset, c.epsilon,
          c.rot_dim, c.rot_offset, c.num_tokens, c.num_q_heads, c.num_kv_heads);
  return check_launch("fused_qk_norm_rope");
}

template <typename scalar_t, typename cache_t, RopeStyle kStyle>
Status launch_fused_heads(int32_t head_dim, const FusedCall& c, const LaunchContext& ctx) {
  switch (head_dim) {
    case 64:
      return launch_fused<scalar_t, cache_t, kStyle, 64>(c, ctx);
    case 128:
      return launch_fused<scalar_t, cache_t, kStyle, 128>(c, ctx);
    case 256:
      return launch_fused<scalar_t, cache_t, kStyle, 256>(c, ctx);
    default:
      return detail::set_error(Status::kUnsupportedShape,
                               "fused_qk_norm_rope: head_dim %d has no instantiation "
                               "(64 / 128 / 256)",
                               head_dim);
  }
}

template <typename scalar_t, typename cache_t>
Status dispatch_fused(const FusedCall& c, int32_t head_dim, RopeStyle style,
                      const LaunchContext& ctx) {
  if (style == RopeStyle::kNeoX) {
    return launch_fused_heads<scalar_t, cache_t, RopeStyle::kNeoX>(head_dim, c, ctx);
  }
  return launch_fused_heads<scalar_t, cache_t, RopeStyle::kGPTJ>(head_dim, c, ctx);
}

template <typename scalar_t>
Status dispatch_fused_cache(const FusedCall& c, int32_t head_dim, RopeStyle style,
                            DType cache_dtype, const LaunchContext& ctx) {
  if (cache_dtype == DType::kFloat32) {
    return dispatch_fused<scalar_t, float>(c, head_dim, style, ctx);
  }
  return dispatch_fused<scalar_t, scalar_t>(c, head_dim, style, ctx);
}

}  // namespace

Status fused_qk_norm_rope(const Tensor& qkv, const Tensor& q_weight, const Tensor& k_weight,
                          const Tensor& positions, const Tensor& cos_sin_cache,
                          int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
                          int32_t rot_dim, int32_t rot_offset, RopeStyle style,
                          float weight_offset, float epsilon, const LaunchContext& ctx) {
  const char* op = "fused_qk_norm_rope";
  Status s = check_launch_context(ctx, op);
  if (s != Status::kSuccess) return s;
  s = check_style(style, op);
  if (s != Status::kSuccess) return s;
  s = check_rope_window(rot_dim, rot_offset, op);
  if (s != Status::kSuccess) return s;
  if (head_dim < 2 || head_dim > kMaxFusedHeadDim || (head_dim % 2) != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: head_dim %d must be even and in [2, %d]", op, head_dim,
                             kMaxFusedHeadDim);
  }
  if (rot_offset + rot_dim > head_dim) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: rot_offset %d + rot_dim %d exceeds head_dim %d", op, rot_offset,
                             rot_dim, head_dim);
  }
  if (num_q_heads < 0 || num_kv_heads < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: negative head count", op);
  }
  if (!finite(weight_offset) || !finite(epsilon) || epsilon < 0.0f) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: weight_offset must be finite and epsilon finite and >= 0", op);
  }
  if (element_bytes(qkv.dtype) == 0) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: qkv dtype %s (fp32/fp16/bf16)", op,
                             dtype_name(qkv.dtype));
  }
  s = check_tensor(qkv, qkv.dtype, 2, static_cast<size_t>(element_bytes(qkv.dtype)), "qkv");
  if (s != Status::kSuccess) return s;
  s = check_device(qkv, ctx, "qkv");
  if (s != Status::kSuccess) return s;
  const int64_t width = static_cast<int64_t>(num_q_heads + 2 * num_kv_heads) * head_dim;
  if (qkv.shape[1] != width) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: qkv width %lld != (Hq + 2*Hkv) * head_dim = %lld", op,
                             static_cast<long long>(qkv.shape[1]),
                             static_cast<long long>(width));
  }
  if (qkv.shape[1] > 1 && qkv.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: qkv last-dim stride %lld != 1", op,
                             static_cast<long long>(qkv.stride[1]));
  }
  if (qkv.shape[0] > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "%s: too many tokens", op);
  }
  const int64_t tokens = qkv.shape[0];
  s = check_positions(positions, "positions", tokens);
  if (s != Status::kSuccess) return s;
  s = check_same_device(positions, qkv, "positions/qkv");
  if (s != Status::kSuccess) return s;
  s = check_cos_sin_cache(cos_sin_cache, qkv.dtype, rot_dim, "cos_sin_cache");
  if (s != Status::kSuccess) return s;
  s = check_same_device(cos_sin_cache, qkv, "cos_sin_cache/qkv");
  if (s != Status::kSuccess) return s;

  const void* qw = nullptr;
  const void* kw = nullptr;
  const Tensor* weights[2] = {&q_weight, &k_weight};
  for (int i = 0; i < 2; ++i) {
    const Tensor& w = *weights[i];
    if (!tensor_present(w)) continue;
    if (w.dtype != qkv.dtype) {
      return detail::set_error(Status::kUnsupportedDtype, "%s: norm weight dtype %s != qkv %s", op,
                               dtype_name(w.dtype), dtype_name(qkv.dtype));
    }
    s = check_tensor(w, w.dtype, 1, static_cast<size_t>(element_bytes(w.dtype)),
                     i == 0 ? "q_weight" : "k_weight");
    if (s != Status::kSuccess) return s;
    if (w.shape[0] != head_dim) {
      return detail::set_error(Status::kUnsupportedShape, "%s: norm weight has %lld entries, "
                                                          "expected head_dim %d",
                               op, static_cast<long long>(w.shape[0]), head_dim);
    }
    if (head_dim > 1 && w.stride[0] != 1) {
      return detail::set_error(Status::kUnsupportedLayout, "%s: norm weight stride %lld != 1", op,
                               static_cast<long long>(w.stride[0]));
    }
    s = check_same_device(w, qkv, "norm weight/qkv");
    if (s != Status::kSuccess) return s;
    if (i == 0) {
      qw = const_data(w);
    } else {
      kw = const_data(w);
    }
  }

  FusedCall call{};
  call.qkv = mutable_data(qkv);
  call.q_weight = qw;
  call.k_weight = kw;
  call.positions = static_cast<const int64_t*>(const_data(positions));
  call.cache = const_data(cos_sin_cache);
  call.token_stride = qkv.stride[0];
  call.weight_offset = weight_offset;
  call.epsilon = epsilon;
  call.rot_dim = rot_dim;
  call.rot_offset = rot_offset;
  call.num_tokens = static_cast<int32_t>(tokens);
  call.num_q_heads = num_q_heads;
  call.num_kv_heads = num_kv_heads;
  const DType cache_dtype = cos_sin_cache.dtype;
  return with_scalar_dtype(qkv.dtype, op, [&](auto tag) {
    using T = typename decltype(tag)::type;
    return dispatch_fused_cache<T>(call, head_dim, style, cache_dtype, ctx);
  });
}

// ---- attn_temperature_tuning ----------------------------------------------

Status attn_temperature_tuning(const Tensor& query, const Tensor& positions, float floor_scale,
                               float attn_scale, const LaunchContext& ctx) {
  const char* op = "attn_temperature_tuning";
  Status s = check_launch_context(ctx, op);
  if (s != Status::kSuccess) return s;
  if (!finite(floor_scale) || !(floor_scale > 0.0f) || !finite(attn_scale)) {
    return detail::set_error(Status::kInvalidArgument,
                             "%s: floor_scale must be > 0 and finite, attn_scale finite", op);
  }
  if (!tensor_present(query)) {
    return detail::set_error(Status::kInvalidArgument, "%s: query is null", op);
  }
  s = check_heads(query, "query", 0);
  if (s != Status::kSuccess) return s;
  s = check_device(query, ctx, "query");
  if (s != Status::kSuccess) return s;
  const int64_t tokens = query.shape[0];
  const int64_t heads = query.shape[1];
  const int64_t head_size = query.shape[2];
  // The kernel has no head-stride argument: heads must be contiguous in a token.
  if (heads > 1 && query.stride[1] != head_size) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "%s: query head stride %lld must equal head_size %lld", op,
                             static_cast<long long>(query.stride[1]),
                             static_cast<long long>(head_size));
  }
  s = check_positions(positions, "positions", tokens);
  if (s != Status::kSuccess) return s;
  s = check_same_device(positions, query, "positions/query");
  if (s != Status::kSuccess) return s;
  if (tokens == 0 || heads == 0 || head_size == 0) return Status::kSuccess;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int block = 256;
  return with_scalar_dtype(query.dtype, op, [&](auto tag) {
    using T = typename decltype(tag)::type;
    attn_temperature_tuning_kernel<T><<<static_cast<unsigned>(tokens), block, 0, stream>>>(
        static_cast<T*>(mutable_data(query)), query.stride[0],
        static_cast<const int64_t*>(const_data(positions)), floor_scale, attn_scale,
        static_cast<int>(tokens), static_cast<int>(heads), static_cast<int>(head_size));
    return check_launch(op);
  });
}

// ---- build_cos_sin_cache ---------------------------------------------------

Status build_cos_sin_cache(const Tensor& cache, const RopeConfig& cfg, int32_t max_position,
                           const LaunchContext& ctx) {
  const char* op = "build_cos_sin_cache";
  Status s = check_launch_context(ctx, op);
  if (s != Status::kSuccess) return s;
  if (element_bytes(cache.dtype) == 0) {
    return detail::set_error(Status::kUnsupportedDtype, "%s: cache dtype %s (fp32/fp16/bf16)", op,
                             dtype_name(cache.dtype));
  }
  s = check_tensor(cache, cache.dtype, 2, static_cast<size_t>(element_bytes(cache.dtype)), "cache");
  if (s != Status::kSuccess) return s;
  s = check_device(cache, ctx, "cache");
  if (s != Status::kSuccess) return s;
  if (max_position < 0) {
    return detail::set_error(Status::kInvalidArgument, "%s: max_position %d < 0", op, max_position);
  }
  if (cache.shape[0] != max_position) {
    return detail::set_error(Status::kUnsupportedShape, "%s: cache has %lld rows, expected %d", op,
                             static_cast<long long>(cache.shape[0]), max_position);
  }
  const int32_t rot_dim =
      cfg.rotary_dim > 0 ? cfg.rotary_dim : static_cast<int32_t>(cache.shape[1]);
  s = check_rope_window(rot_dim, 0, op);
  if (s != Status::kSuccess) return s;
  if (cfg.rotary_dim > 0 && cache.shape[1] != cfg.rotary_dim) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: cache width %lld != cfg.rotary_dim %d", op,
                             static_cast<long long>(cache.shape[1]), cfg.rotary_dim);
  }
  s = check_cos_sin_cache(cache, cache.dtype, rot_dim, "cache");
  if (s != Status::kSuccess) return s;
  if (max_position == 0) return Status::kSuccess;

  InvFreqTable inv_freq{};
  float host_mscale = 1.0f;
  RopeConfig effective = cfg;
  effective.rotary_dim = rot_dim;  // 0 in cfg means "take the cache width"
  s = compute_inv_freq(effective, inv_freq.v, &host_mscale);
  if (s != Status::kSuccess) return s;

  const int64_t grid = (max_position + kCachePositionsPerCta - 1) / kCachePositionsPerCta;
  const int block = std::min(kMaxApplyBlock, std::max(32, rot_dim / 2));
  const DType dtype = cache.dtype;
  void* dst = mutable_data(cache);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.stream);
  switch (dtype) {
    case DType::kFloat32:
      build_cos_sin_cache_kernel<float><<<static_cast<unsigned>(grid), block, 0, stream>>>(
          static_cast<float*>(dst), inv_freq, host_mscale, max_position, rot_dim);
      break;
    case DType::kFloat16:
      build_cos_sin_cache_kernel<__half><<<static_cast<unsigned>(grid), block, 0, stream>>>(
          static_cast<__half*>(dst), inv_freq, host_mscale, max_position, rot_dim);
      break;
    case DType::kBFloat16:
      build_cos_sin_cache_kernel<__nv_bfloat16><<<static_cast<unsigned>(grid), block, 0, stream>>>(
          static_cast<__nv_bfloat16*>(dst), inv_freq, host_mscale, max_position, rot_dim);
      break;
    default:
      return detail::set_error(Status::kUnsupportedDtype, "%s: cache dtype %s", op,
                               dtype_name(dtype));
  }
  return check_launch(op);
}

}  // namespace kops::rope
