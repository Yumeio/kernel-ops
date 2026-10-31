// =============================================================================
// kernel-ops | tests/unit/test_position.cc                           Catalog: R01
// B1 position family: frequency tables, cos/sin cache, apply_rotary (table,
// on-the-fly, per-token cache offsets, partial/offset windows, strided views),
// apply_mrope (contiguous + interleaved), fused_qk_norm_rope (against the
// separate per-head RMSNorm -> RoPE chain) and attn_temperature_tuning.
//
// Oracle: fp64 loops in this file. csrc/reference/ does not define rope_f64 /
// mrope_f64 yet, so the oracle lives here. Two deliberate rules:
//   * the rotation oracle consumes the SAME fp32 frequencies the ABI passes to
//     the kernel (compute_inv_freq rounds to float), so a comparison isolates the
//     kernel instead of re-measuring the fp32 table rounding;
//   * the cache oracle keeps cos/sin in fp64, so "one rounding at the store" is
//     what the tolerances measure.
// Contract cases assert the Status of bad dtype / shape / stride / enum input.
// =============================================================================

#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"

#include "csrc/position/rope_kernels.cuh"
#include "csrc/position/ops.h"

namespace {

using kops::RopeStyle;
using kops::rope::RopeConfig;
using kops::rope::RopeScaling;

constexpr double kTwoPi = 6.283185307179586476925286766559;

// ---- buffers / tensors -----------------------------------------------------

void fill(std::vector<float>& values, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : values) v = dist(rng);
}

template <typename T>
std::vector<T> cast_from_float(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(out[i], src[i]);
  return out;
}

template <typename T>
std::vector<double> cast_to_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<double>(kops_test::to_float(src[i]));
  return out;
}

template <typename T>
kops::Tensor make1d(T* data, kops::DType dtype, int64_t n) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

// ---- fp64 oracle -----------------------------------------------------------

std::vector<double> ref_inv_freq(const RopeConfig& c) {
  const int dim = c.rotary_dim;
  const int half = dim / 2;
  const double theta = static_cast<double>(c.theta);
  const double factor = static_cast<double>(c.factor);
  const double L0 = static_cast<double>(c.original_max_position);
  double base = theta;
  if (c.scaling == RopeScaling::kDynamicNTK) {
    // No seq_len in RopeConfig: the length is original_max_position * factor,
    // i.e. factor * L / L0 == factor^2 (documented in csrc/position/rope.cu).
    base = theta * std::pow(factor * factor - (factor - 1.0), double(dim) / (dim - 2));
  } else if (c.scaling == RopeScaling::kNTKAlpha) {
    base = theta * std::pow(factor, double(dim) / (dim - 2));
  }
  std::vector<double> out(half);
  if (c.scaling == RopeScaling::kYaRN) {
    auto corr = [&](double rotations) {
      return (double(dim) * std::log(L0 / (rotations * kTwoPi))) / (2.0 * std::log(theta));
    };
    double low = corr(double(c.beta_fast));
    double high = corr(double(c.beta_slow));
    if (c.truncate) {
      low = std::floor(low);
      high = std::ceil(high);
    }
    low = std::max(low, 0.0);
    high = std::min(high, double(dim - 1));
    if (low == high) high += 0.001;
    for (int i = 0; i < half; ++i) {
      const double extrap = 1.0 / std::pow(theta, double(2 * i) / dim);
      const double interp = extrap / factor;
      const double ramp = std::min(std::max((double(i) - low) / (high - low), 0.0), 1.0);
      out[i] = interp * ramp + extrap * (1.0 - ramp);
    }
  } else if (c.scaling == RopeScaling::kLlama3) {
    const double low_wavelen = L0 / double(c.low_freq_factor);
    const double high_wavelen = L0 / double(c.high_freq_factor);
    for (int i = 0; i < half; ++i) {
      const double inv = 1.0 / std::pow(theta, double(2 * i) / dim);
      const double wavelen = kTwoPi / inv;
      if (wavelen < high_wavelen) {
        out[i] = inv;
      } else if (wavelen > low_wavelen) {
        out[i] = inv / factor;
      } else {
        const double smooth = (L0 / wavelen - double(c.low_freq_factor)) /
                              (double(c.high_freq_factor) - double(c.low_freq_factor));
        out[i] = (1.0 - smooth) * inv / factor + smooth * inv;
      }
    }
  } else if (c.scaling == RopeScaling::kLongRoPE) {
    const float* ext = c.long_factor != nullptr ? c.long_factor : c.short_factor;
    for (int i = 0; i < half; ++i) {
      out[i] = 1.0 / (double(ext[i]) * std::pow(theta, double(2 * i) / dim));
    }
  } else {
    for (int i = 0; i < half; ++i) {
      out[i] = 1.0 / std::pow(base, double(2 * i) / dim);
      if (c.scaling == RopeScaling::kLinear) out[i] /= factor;
    }
  }
  return out;
}

double ref_mscale(const RopeConfig& c) {
  auto yarn = [](double scale, double m) {
    return scale <= 1.0 ? 1.0 : 0.1 * m * std::log(scale) + 1.0;
  };
  const double factor = double(c.factor);
  const double L0 = double(c.original_max_position);
  double m = 1.0;
  if (c.scaling == RopeScaling::kYaRN) {
    m = yarn(factor, double(c.mscale)) / yarn(factor, double(c.mscale_all_dim));
  } else if (c.scaling == RopeScaling::kLongRoPE) {
    m = factor > 1.0 ? std::sqrt(1.0 + std::log(factor) / std::log(L0)) : 1.0;
  }
  if (c.attention_factor != 0.0f) m = double(c.attention_factor);
  return m;
}

/// Row-major table [rows, rot]: cos in [0, rot/2), sin in [rot/2, rot), fp64.
struct RefTable {
  int64_t rows = 0;
  int32_t rot = 0;
  std::vector<double> data;

  const double* row(int64_t r) const { return data.data() + r * rot; }
};

RefTable ref_table(const std::vector<float>& inv_freq_f32, double mscale, int64_t rows,
                   int32_t rot) {
  RefTable t;
  t.rows = rows;
  t.rot = rot;
  t.data.assign(static_cast<size_t>(rows) * rot, 0.0);
  const int half = rot / 2;
  for (int64_t p = 0; p < rows; ++p) {
    for (int i = 0; i < half; ++i) {
      const double angle = static_cast<double>(p) * static_cast<double>(inv_freq_f32[i]);
      t.data[p * rot + i] = mscale * std::cos(angle);
      t.data[p * rot + half + i] = mscale * std::sin(angle);
    }
  }
  return t;
}

void ref_rotate_pair(double& xa, double& xb, double c, double s) {
  const double a = xa, b = xb;
  xa = a * c - b * s;
  xb = b * c + a * s;
}

/// In-place fp64 RoPE over x[t, h, head_dim] using table rows `row_index[t]`.
void ref_rope(std::vector<double>& x, int64_t rows, int64_t heads, int64_t head_dim,
              const std::vector<int64_t>& row_index, const RefTable& table, int32_t rot_dim,
              int32_t rot_offset, RopeStyle style) {
  const int half = rot_dim / 2;
  for (int64_t t = 0; t < rows; ++t) {
    const double* cs = table.row(row_index[t]);
    for (int64_t h = 0; h < heads; ++h) {
      double* base = x.data() + (t * heads + h) * head_dim + rot_offset;
      for (int i = 0; i < half; ++i) {
        const int a = style == RopeStyle::kNeoX ? i : 2 * i;
        const int b = style == RopeStyle::kNeoX ? half + i : 2 * i + 1;
        ref_rotate_pair(base[a], base[b], cs[i], cs[half + i]);
      }
    }
  }
}

int mrope_axis(int i, int section_t, int section_h, int section_w, bool interleaved) {
  if (interleaved) {
    if (i % 3 == 1 && i < 3 * section_h) return 1;
    if (i % 3 == 2 && i < 3 * section_w) return 2;
    return 0;
  }
  return i < section_t ? 0 : (i < section_t + section_h ? 1 : 2);
}

void ref_mrope(std::vector<double>& x, int64_t rows, int64_t heads, int64_t head_dim,
               const std::vector<std::vector<int64_t>>& positions_thw, const RefTable& table,
               int32_t rot_dim, const int32_t section[3], bool interleaved, RopeStyle style) {
  const int half = rot_dim / 2;
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      double* base = x.data() + (t * heads + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        const int axis = mrope_axis(i, section[0], section[1], section[2], interleaved);
        const double* cs = table.row(positions_thw[axis][t]);
        const int a = style == RopeStyle::kNeoX ? i : 2 * i;
        const int b = style == RopeStyle::kNeoX ? half + i : 2 * i + 1;
        ref_rotate_pair(base[a], base[b], cs[i], cs[half + i]);
      }
    }
  }
}

/// Separate chain: per-head RMSNorm over head_dim, then RoPE on the window.
void ref_fused_qk_norm_rope(std::vector<double>& qkv, int64_t rows, int64_t num_q_heads,
                            int64_t num_kv_heads, int64_t head_dim, const std::vector<double>& qw,
                            const std::vector<double>& kw, double weight_offset, double eps,
                            const std::vector<int64_t>& positions, const RefTable& table,
                            int32_t rot_dim, int32_t rot_offset, RopeStyle style) {
  const int64_t heads = num_q_heads + num_kv_heads;
  const int64_t width = (num_q_heads + 2 * num_kv_heads) * head_dim;
  const int half = rot_dim / 2;
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      double* base = qkv.data() + t * width + h * head_dim;
      const std::vector<double>& w = h < num_q_heads ? qw : kw;
      double ss = 0.0;
      for (int64_t j = 0; j < head_dim; ++j) ss += base[j] * base[j];
      const double rstd = 1.0 / std::sqrt(ss / double(head_dim) + eps);
      for (int64_t j = 0; j < head_dim; ++j) base[j] = base[j] * rstd * (w[j] + weight_offset);
      const double* cs = table.row(positions[t]);
      double* win = base + rot_offset;
      for (int i = 0; i < half; ++i) {
        const int a = style == RopeStyle::kNeoX ? i : 2 * i;
        const int b = style == RopeStyle::kNeoX ? half + i : 2 * i + 1;
        ref_rotate_pair(win[a], win[b], cs[i], cs[half + i]);
      }
    }
  }
}

double ref_temperature(int64_t position, double floor_scale, double attn_scale) {
  if (position < 0) return 1.0;
  return std::log(std::floor((double(position) + 1.0) / floor_scale) + 1.0) * attn_scale + 1.0;
}

// ---- default configurations ------------------------------------------------

RopeConfig base_config(int32_t rot, float theta) {
  RopeConfig c;
  c.scaling = RopeScaling::kNone;
  c.theta = theta;
  c.rotary_dim = rot;
  return c;
}

/// Frequency table + cache as the ABI carries them (fp32 frequencies).
struct HostFreq {
  std::vector<float> inv_freq;
  float mscale = 1.0f;
};

HostFreq host_freq(const RopeConfig& cfg) {
  HostFreq out;
  out.inv_freq.resize(static_cast<size_t>(cfg.rotary_dim / 2));
  KOPS_EXPECT_OK(kops::rope::compute_inv_freq(cfg, out.inv_freq.data(), &out.mscale),
                 "compute_inv_freq");
  return out;
}

// =============================================================================
// compute_inv_freq / compute_alibi_slopes
// =============================================================================

void test_inv_freq() {
  struct Case {
    const char* name;
    RopeConfig cfg;
  };
  std::vector<float> short_factor = {1.0f, 1.1f, 0.9f, 1.05f, 1.2f, 0.95f, 1.0f, 1.3f};
  std::vector<float> long_factor = {1.5f, 1.4f, 1.3f, 1.25f, 1.2f, 1.15f, 1.1f, 1.05f};
  std::vector<Case> cases;

  cases.push_back({"none", base_config(64, 1.0e4f)});
  cases.push_back({"none-theta5e5", base_config(128, 5.0e5f)});

  RopeConfig linear = base_config(64, 1.0e4f);
  linear.scaling = RopeScaling::kLinear;
  linear.factor = 4.0f;
  cases.push_back({"linear", linear});

  RopeConfig dntk = base_config(128, 1.0e4f);
  dntk.scaling = RopeScaling::kDynamicNTK;
  dntk.factor = 4.0f;
  dntk.original_max_position = 4096;
  cases.push_back({"dynamic-ntk", dntk});

  RopeConfig alpha = base_config(128, 1.0e4f);
  alpha.scaling = RopeScaling::kNTKAlpha;
  alpha.factor = 4.0f;
  cases.push_back({"ntk-alpha", alpha});

  RopeConfig yarn = base_config(64, 1.0e4f);
  yarn.scaling = RopeScaling::kYaRN;
  yarn.factor = 4.0f;
  yarn.original_max_position = 4096;
  yarn.beta_fast = 32.0f;
  yarn.beta_slow = 1.0f;
  cases.push_back({"yarn", yarn});

  RopeConfig yarn_attn = yarn;
  yarn_attn.attention_factor = 1.5f;
  cases.push_back({"yarn-attention-factor", yarn_attn});

  RopeConfig yarn_deepseek = yarn;
  yarn_deepseek.rotary_dim = 128;
  yarn_deepseek.theta = 1.0e6f;
  yarn_deepseek.mscale = 1.0f;
  yarn_deepseek.mscale_all_dim = 1.0f;
  cases.push_back({"yarn-mscale-all-dim", yarn_deepseek});

  RopeConfig yarn_notrunc = yarn;
  yarn_notrunc.truncate = false;
  cases.push_back({"yarn-no-truncate", yarn_notrunc});

  RopeConfig llama3 = base_config(64, 5.0e5f);
  llama3.scaling = RopeScaling::kLlama3;
  llama3.factor = 8.0f;
  llama3.original_max_position = 8192;
  llama3.low_freq_factor = 1.0f;
  llama3.high_freq_factor = 4.0f;
  cases.push_back({"llama3", llama3});

  RopeConfig longrope = base_config(16, 1.0e4f);
  longrope.scaling = RopeScaling::kLongRoPE;
  longrope.factor = 4.0f;
  longrope.original_max_position = 4096;
  longrope.short_factor = short_factor.data();
  cases.push_back({"longrope-short", longrope});

  RopeConfig longrope_long = longrope;
  longrope_long.long_factor = long_factor.data();
  cases.push_back({"longrope-long", longrope_long});

  for (const Case& c : cases) {
    const std::vector<double> want = ref_inv_freq(c.cfg);
    std::vector<float> got(want.size());
    float mscale = 0.0f;
    const kops::Status s = kops::rope::compute_inv_freq(c.cfg, got.data(), &mscale);
    if (s != kops::Status::kSuccess) {
      kops_test::record_failure(c.name, kops::last_error_message(), __FILE__, __LINE__);
      continue;
    }
    kops_test::compare(got, want, 0.0, 2e-6, c.name);

    // mscale is fp32 in the ABI: 1e-6 relative is enough for every family.
    const std::vector<double> mscale_want = {ref_mscale(c.cfg)};
    const std::vector<float> mscale_got = {mscale};
    kops_test::compare(mscale_got, mscale_want, 0.0, 2e-6, "inv_freq.mscale");
  }

  // attention_factor replaces the derived magnitude.
  {
    const std::vector<double> want = {1.5};
    const std::vector<float> got = {host_freq(yarn_attn).mscale};
    kops_test::compare(got, want, 0.0, 1e-6, "inv_freq.attention_factor");
  }
  // ... and a factor of 1 for kDynamicNTK reproduces the unscaled table.
  {
    RopeConfig unit = dntk;
    unit.factor = 1.0f;
    RopeConfig plain = base_config(128, 1.0e4f);
    const std::vector<float> a = host_freq(unit).inv_freq;
    const std::vector<float> b = host_freq(plain).inv_freq;
    std::vector<double> want(b.begin(), b.end());
    kops_test::compare(a, want, 0.0, 1e-7, "inv_freq.dynamic_ntk_factor1");
  }
}

void test_alibi_slopes() {
  for (int32_t heads : {1, 2, 5, 8, 12, 32}) {
    for (float max_bias : {8.0f, 1.0f}) {
      const int32_t num_heads = heads;
      std::vector<float> got(static_cast<size_t>(num_heads));
      KOPS_EXPECT_OK(kops::rope::compute_alibi_slopes(num_heads, max_bias, got.data()),
                     "compute_alibi_slopes");
      int n = 1;
      while (n <= num_heads / 2) n *= 2;
      const double m0 = std::pow(2.0, -double(max_bias) / n);
      const double m1 = std::pow(2.0, -double(max_bias) / 2.0 / n);
      std::vector<double> want(static_cast<size_t>(num_heads));
      for (int h = 0; h < num_heads; ++h) {
        want[h] = h < n ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n) + 1);
      }
      kops_test::compare(got, want, 0.0, 1e-6, "alibi_slopes");
    }
  }
}

// =============================================================================
// build_cos_sin_cache
// =============================================================================

template <typename T>
void test_build_cache(const char* name, kops::DType dtype, double atol, double rtol) {
  const int32_t rot = 32;
  const int64_t max_position = 96;
  RopeConfig cfg = base_config(rot, 1.0e4f);
  cfg.scaling = RopeScaling::kYaRN;  // non-unit mscale must be baked into the table
  cfg.factor = 4.0f;
  cfg.original_max_position = 512;

  const HostFreq freq = host_freq(cfg);
  const RefTable want_table = ref_table(freq.inv_freq, double(freq.mscale), max_position, rot);

  T* d_cache = kops_test::device_alloc<T>(static_cast<size_t>(max_position) * rot);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  cudaMemset(d_cache, 0, sizeof(T) * static_cast<size_t>(max_position) * rot);
  const kops::Tensor cache =
      kops_test::tensor2d(d_cache, dtype, max_position, rot);
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(cache, cfg, static_cast<int32_t>(max_position), ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(d_cache, static_cast<size_t>(max_position) * rot);
  kops_test::compare(got, want_table.data, atol, rtol, name);

  // cfg.rotary_dim == 0 means "take the cache width".
  RopeConfig derived = cfg;
  derived.rotary_dim = 0;
  cudaMemset(d_cache, 0, sizeof(T) * static_cast<size_t>(max_position) * rot);
  KOPS_EXPECT_OK(
      kops::rope::build_cos_sin_cache(cache, derived, static_cast<int32_t>(max_position), ctx),
      "build_cos_sin_cache.derived_rotary_dim");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got_derived = kops_test::copy_out(d_cache, static_cast<size_t>(max_position) * rot);
  kops_test::compare(got_derived, want_table.data, atol, rtol,
                     "build_cos_sin_cache.derived_rotary_dim");
  kops_test::device_free(d_cache);
}

// Two independent cache builds on different streams, a graph replay, and the
// maximum rotary width. This is the regression test for the by-value frequency
// table: no module-level device buffer may be shared between launches.
void test_build_cache_isolated() {
  const int32_t rot = 64;
  const int64_t rows = 128;
  RopeConfig cfg1 = base_config(rot, 1.0e4f);
  RopeConfig cfg2 = base_config(rot, 5.0e5f);
  const HostFreq f1 = host_freq(cfg1);
  const HostFreq f2 = host_freq(cfg2);
  const RefTable w1 = ref_table(f1.inv_freq, f1.mscale, rows, rot);
  const RefTable w2 = ref_table(f2.inv_freq, f2.mscale, rows, rot);

  float* d1 = kops_test::device_alloc<float>(static_cast<size_t>(rows) * rot);
  float* d2 = kops_test::device_alloc<float>(static_cast<size_t>(rows) * rot);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  cudaStream_t s1 = nullptr, s2 = nullptr;
  KOPS_TEST_CHECK(cudaStreamCreate(&s1) == cudaSuccess);
  KOPS_TEST_CHECK(cudaStreamCreate(&s2) == cudaSuccess);
  kops::LaunchContext c1 = ctx, c2 = ctx;
  c1.stream = reinterpret_cast<kops::kops_stream_t>(s1);
  c2.stream = reinterpret_cast<kops::kops_stream_t>(s2);
  const kops::Tensor t1 = kops_test::tensor2d(d1, kops::DType::kFloat32, rows, rot);
  const kops::Tensor t2 = kops_test::tensor2d(d2, kops::DType::kFloat32, rows, rot);

  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(t1, cfg1, static_cast<int32_t>(rows), c1),
                 "cache stream 1");
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(t2, cfg2, static_cast<int32_t>(rows), c2),
                 "cache stream 2");
  KOPS_TEST_CHECK(cudaStreamSynchronize(s1) == cudaSuccess);
  KOPS_TEST_CHECK(cudaStreamSynchronize(s2) == cudaSuccess);
  kops_test::compare(kops_test::copy_out(d1, static_cast<size_t>(rows) * rot), w1.data, 1e-6, 1e-6,
                     "cache stream 1 values");
  kops_test::compare(kops_test::copy_out(d2, static_cast<size_t>(rows) * rot), w2.data, 1e-6, 1e-6,
                     "cache stream 2 values");

  // Capture/replay the builder; the frequency table is a kernel parameter.
  KOPS_TEST_CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(t1, cfg1, static_cast<int32_t>(rows), c1),
                 "cache graph");
  cudaGraph_t graph = nullptr;
  KOPS_TEST_CHECK(cudaStreamEndCapture(s1, &graph) == cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  KOPS_TEST_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
  KOPS_TEST_CHECK(cudaGraphLaunch(exec, s1) == cudaSuccess);
  KOPS_TEST_CHECK(cudaStreamSynchronize(s1) == cudaSuccess);
  kops_test::compare(kops_test::copy_out(d1, static_cast<size_t>(rows) * rot), w1.data, 1e-6, 1e-6,
                     "cache graph values");
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);

  // Maximum rotary width accepted by the ABI.
  const int32_t wide_rot = kops::rope::kMaxRotDim;
  RopeConfig wide_cfg = base_config(wide_rot, 1.0e4f);
  const HostFreq wf = host_freq(wide_cfg);
  const RefTable ww = ref_table(wf.inv_freq, wf.mscale, 16, wide_rot);
  float* dw = kops_test::device_alloc<float>(16u * wide_rot);
  const kops::Tensor tw = kops_test::tensor2d(dw, kops::DType::kFloat32, 16, wide_rot);
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(tw, wide_cfg, 16, ctx), "cache max rot");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  kops_test::compare(kops_test::copy_out(dw, 16u * wide_rot), ww.data, 1e-6, 1e-6,
                     "cache max rot values");

  kops_test::device_free(d1);
  kops_test::device_free(d2);
  kops_test::device_free(dw);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
}

// =============================================================================
// apply_rotary: table path
// =============================================================================

struct RotaryFixture {
  int64_t rows = 5;
  int64_t heads = 3;
  int64_t kv_heads = 2;
  int64_t head_dim = 32;
  int64_t max_position = 64;
  int32_t rot_dim = 32;
  int32_t rot_offset = 0;

  int64_t q_count() const { return rows * heads * head_dim; }
  int64_t k_count() const { return rows * kv_heads * head_dim; }
};

/// Builds q/k, runs apply_rotary (table) for one dtype and compares both outputs.
template <typename T>
void run_apply_rotary(const char* name, RopeStyle style, kops::DType dtype, RopeStyle ref_style,
                      RotaryFixture f, kops::DType cache_dtype, double atol, double rtol,
                      bool with_key) {
  RopeConfig cfg = base_config(f.rot_dim, 1.0e4f);
  cfg.scaling = RopeScaling::kYaRN;
  cfg.factor = 2.0f;
  cfg.original_max_position = 256;
  const HostFreq freq = host_freq(cfg);
  const RefTable table =
      ref_table(freq.inv_freq, double(freq.mscale), f.max_position, f.rot_dim);

  std::vector<float> q_f(static_cast<size_t>(f.q_count()));
  std::vector<float> k_f(static_cast<size_t>(f.k_count()));
  fill(q_f, 21u);
  fill(k_f, 22u);
  const auto q = cast_from_float<T>(q_f);
  const auto k = cast_from_float<T>(k_f);
  const std::vector<int64_t> positions = {0, 1, 7, 13, 63};

  T* d_q = kops_test::device_alloc<T>(q.size());
  T* d_k = kops_test::device_alloc<T>(k.size());
  int64_t* d_pos = kops_test::device_alloc<int64_t>(positions.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_q, q);
  kops_test::copy_in(d_k, k);
  kops_test::copy_in(d_pos, positions);
  void* cache_buf = nullptr;
  kops::Tensor t_cache{};
  if (cache_dtype == kops::DType::kFloat32) {
    cache_buf = kops_test::device_alloc<float>(static_cast<size_t>(f.max_position) * f.rot_dim);
    t_cache = kops_test::tensor2d(cache_buf, kops::DType::kFloat32, f.max_position, f.rot_dim);
  } else {
    cache_buf = kops_test::device_alloc<T>(static_cast<size_t>(f.max_position) * f.rot_dim);
    t_cache = kops_test::tensor2d(cache_buf, cache_dtype, f.max_position, f.rot_dim);
  }
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(t_cache, cfg, static_cast<int32_t>(f.max_position), ctx),
                 "build cache");

  kops::Tensor t_q = kops_test::tensor3d(d_q, dtype, f.rows, f.heads, f.head_dim);
  kops::Tensor t_k = kops_test::tensor3d(d_k, dtype, f.rows, f.kv_heads, f.head_dim);
  kops::Tensor t_pos = make1d(d_pos, kops::DType::kInt64, static_cast<int64_t>(positions.size()));
  KOPS_EXPECT_OK(kops::rope::apply_rotary(t_pos, t_q, with_key ? t_k : kops::Tensor{}, t_cache,
                                          kops::Tensor{}, f.rot_dim, f.rot_offset, style, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_q = kops_test::copy_out(d_q, q.size());
  std::vector<double> want_q = cast_to_double(q);
  ref_rope(want_q, f.rows, f.heads, f.head_dim, positions, table, f.rot_dim, f.rot_offset, ref_style);
  kops_test::compare(got_q, want_q, atol, rtol, name);

  if (with_key) {
    const auto got_k = kops_test::copy_out(d_k, k.size());
    std::vector<double> want_k = cast_to_double(k);
    ref_rope(want_k, f.rows, f.kv_heads, f.head_dim, positions, table, f.rot_dim, f.rot_offset,
             ref_style);
    kops_test::compare(got_k, want_k, atol, rtol, "apply_rotary.key");
  }

  kops_test::device_free(d_q);
  kops_test::device_free(d_k);
  kops_test::device_free(d_pos);
  cudaFree(cache_buf);
}

void test_apply_rotary() {
  RotaryFixture f;
  for (RopeStyle style : {RopeStyle::kNeoX, RopeStyle::kGPTJ}) {
    const bool neox = style == RopeStyle::kNeoX;
    run_apply_rotary<float>(neox ? "apply_rotary.fp32.neox" : "apply_rotary.fp32.gptj", style,
                            kops::DType::kFloat32, style, f, kops::DType::kFloat32, 1e-6, 1e-5, true);
    run_apply_rotary<__nv_bfloat16>(
        neox ? "apply_rotary.bf16.neox" : "apply_rotary.bf16.gptj", style,
        kops::DType::kBFloat16, style, f, kops::DType::kFloat32, 2e-2, 1e-2, true);
    // bf16 cache: the same table rounded to the tensor dtype.
    run_apply_rotary<__nv_bfloat16>(
        neox ? "apply_rotary.bf16.neox.bf16cache" : "apply_rotary.bf16.gptj.bf16cache", style,
        kops::DType::kBFloat16, style, f, kops::DType::kBFloat16, 3e-2, 2e-2, false);
  }
}

// ---- partial rotary window + cache offsets + strided views -----------------

template <typename T>
void test_partial_rot(const char* name, RopeStyle style, kops::DType dtype, double atol,
                      double rtol) {
  RotaryFixture f;
  f.head_dim = 32;
  f.rot_dim = 8;
  f.rot_offset = 8;
  run_apply_rotary<T>(name, style, dtype, style, f, kops::DType::kFloat32, atol, rtol, true);
}

template <typename T>
void run_strided_offsets(const char* name, RopeStyle style, kops::DType dtype, double atol,
                         double rtol) {
  // Layout: [T, H, 32] with a 40-element head stride and a 3*40 token stride.
  const int64_t rows = 4, heads = 3, kv_heads = 2, head_dim = 40, used = 32;
  const int32_t rot = 32;
  const int64_t max_position = 48;
  const int64_t concat_rows = 2 * max_position;  // two concatenated tables
  RopeConfig cfg = base_config(rot, 1.0e4f);
  const HostFreq freq = host_freq(cfg);
  const RefTable table = ref_table(freq.inv_freq, double(freq.mscale), concat_rows, rot);

  std::vector<float> q_f(static_cast<size_t>(rows * heads * head_dim));
  std::vector<float> k_f(static_cast<size_t>(rows * kv_heads * head_dim));
  fill(q_f, 31u);
  fill(k_f, 32u);
  const auto q = cast_from_float<T>(q_f);
  const auto k = cast_from_float<T>(k_f);
  const std::vector<int64_t> positions = {0, 5, 11, 21};
  const std::vector<int64_t> offsets = {max_position, max_position, max_position, max_position};

  T* d_q = kops_test::device_alloc<T>(q.size());
  T* d_k = kops_test::device_alloc<T>(k.size());
  float* d_cache = kops_test::device_alloc<float>(static_cast<size_t>(concat_rows) * rot);
  int64_t* d_pos = kops_test::device_alloc<int64_t>(positions.size());
  int64_t* d_off = kops_test::device_alloc<int64_t>(offsets.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_q, q);
  kops_test::copy_in(d_k, k);
  kops_test::copy_in(d_pos, positions);
  kops_test::copy_in(d_off, offsets);

  const kops::Tensor t_cache =
      kops_test::tensor2d(d_cache, kops::DType::kFloat32, concat_rows, rot);
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(t_cache, cfg, static_cast<int32_t>(concat_rows), ctx),
                 "build cache (concat)");

  kops::Tensor t_q = kops_test::tensor3d(d_q, dtype, rows, heads, head_dim);
  kops::Tensor t_k = kops_test::tensor3d(d_k, dtype, rows, kv_heads, head_dim);
  // Shrink the last dim to the rotated part, leaving the 8-element tail as padding.
  t_q.shape[2] = used;
  t_k.shape[2] = used;
  t_q.stride[1] = head_dim;
  t_k.stride[1] = head_dim;
  t_q.stride[0] = heads * head_dim;
  t_k.stride[0] = kv_heads * head_dim;

  const kops::Tensor t_pos = make1d(d_pos, kops::DType::kInt64, rows);
  const kops::Tensor t_off = make1d(d_off, kops::DType::kInt64, rows);
  KOPS_EXPECT_OK(kops::rope::apply_rotary(t_pos, t_q, t_k, t_cache, t_off, rot, 0, style, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_q = kops_test::copy_out(d_q, q.size());
  std::vector<double> want_q = cast_to_double(q);
  std::vector<int64_t> row_index(rows);
  for (int64_t t = 0; t < rows; ++t) row_index[t] = positions[t] + offsets[t];
  ref_rope(want_q, rows, heads, head_dim, row_index, table, rot, 0, style);
  kops_test::compare(got_q, want_q, atol, rtol, name);

  const auto got_k = kops_test::copy_out(d_k, k.size());
  std::vector<double> want_k = cast_to_double(k);
  ref_rope(want_k, rows, kv_heads, head_dim, row_index, table, rot, 0, style);
  kops_test::compare(got_k, want_k, atol, rtol, "apply_rotary.strided.offsets.key");

  kops_test::device_free(d_q);
  kops_test::device_free(d_k);
  kops_test::device_free(d_cache);
  kops_test::device_free(d_pos);
  kops_test::device_free(d_off);
}

// ---- on-the-fly ------------------------------------------------------------

template <typename T>
void test_on_the_fly(const char* name, RopeStyle style, kops::DType dtype, double atol,
                     double rtol) {
  RotaryFixture f;
  RopeConfig cfg = base_config(f.rot_dim, 1.0e4f);
  cfg.scaling = RopeScaling::kYaRN;
  cfg.factor = 4.0f;
  cfg.original_max_position = 512;
  const HostFreq freq = host_freq(cfg);
  const RefTable table = ref_table(freq.inv_freq, double(freq.mscale), f.max_position, f.rot_dim);

  std::vector<float> q_f(static_cast<size_t>(f.q_count()));
  fill(q_f, 41u);
  const auto q = cast_from_float<T>(q_f);
  const std::vector<int64_t> positions = {0, 2, 9, 30, 63};

  T* d_q_table = kops_test::device_alloc<T>(q.size());
  T* d_q_fly = kops_test::device_alloc<T>(q.size());
  T* d_cache = kops_test::device_alloc<T>(static_cast<size_t>(f.max_position) * f.rot_dim);
  float* d_inv = kops_test::device_alloc<float>(freq.inv_freq.size());
  int64_t* d_pos = kops_test::device_alloc<int64_t>(positions.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_q_table, q);
  kops_test::copy_in(d_q_fly, q);
  kops_test::copy_in(d_inv, freq.inv_freq);
  kops_test::copy_in(d_pos, positions);

  const kops::Tensor t_cache = kops_test::tensor2d(d_cache, dtype, f.max_position, f.rot_dim);
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(t_cache, cfg, static_cast<int32_t>(f.max_position), ctx),
                 "build cache (on the fly)");

  kops::Tensor t_table = kops_test::tensor3d(d_q_table, dtype, f.rows, f.heads, f.head_dim);
  kops::Tensor t_fly = kops_test::tensor3d(d_q_fly, dtype, f.rows, f.heads, f.head_dim);
  const kops::Tensor t_pos = make1d(d_pos, kops::DType::kInt64, f.rows);
  const kops::Tensor t_inv = make1d(d_inv, kops::DType::kFloat32, static_cast<int64_t>(freq.inv_freq.size()));

  KOPS_EXPECT_OK(kops::rope::apply_rotary(t_pos, t_table, kops::Tensor{}, t_cache, kops::Tensor{},
                                          f.rot_dim, 0, style, ctx),
                 "apply_rotary (table, reference for on the fly)");
  KOPS_EXPECT_OK(kops::rope::apply_rotary_on_the_fly(t_pos, t_fly, kops::Tensor{}, t_inv,
                                                     freq.mscale, f.rot_dim, 0, style, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_fly = kops_test::copy_out(d_q_fly, q.size());
  std::vector<double> want = cast_to_double(q);
  ref_rope(want, f.rows, f.heads, f.head_dim, positions, table, f.rot_dim, 0, style);
  kops_test::compare(got_fly, want, atol, rtol, name);

  // The table path and the on-the-fly path share the fp32 frequencies and the
  // fp64 sincos, so they must agree to the last bit of the store dtype.
  const auto got_table = kops_test::copy_out(d_q_table, q.size());
  std::vector<double> table_d = cast_to_double(got_table);
  kops_test::compare(got_fly, table_d, atol, rtol, "apply_rotary_on_the_fly == table");

  kops_test::device_free(d_q_table);
  kops_test::device_free(d_q_fly);
  kops_test::device_free(d_cache);
  kops_test::device_free(d_inv);
  kops_test::device_free(d_pos);
}

// =============================================================================
// apply_mrope
// =============================================================================

template <typename T>
void test_mrope(const char* name, RopeStyle style, kops::DType dtype, bool interleaved,
                double atol, double rtol) {
  const int64_t rows = 4, heads = 2, kv_heads = 1, head_dim = 16;
  const int32_t rot = 16;
  const int32_t section[3] = {4, 2, 2};  // sums to rot / 2
  const int64_t max_position = 32;
  RopeConfig cfg = base_config(rot, 1.0e4f);
  const HostFreq freq = host_freq(cfg);
  const RefTable table = ref_table(freq.inv_freq, double(freq.mscale), max_position, rot);

  std::vector<float> q_f(static_cast<size_t>(rows * heads * head_dim));
  std::vector<float> k_f(static_cast<size_t>(rows * kv_heads * head_dim));
  fill(q_f, 51u);
  fill(k_f, 52u);
  const auto q = cast_from_float<T>(q_f);
  const auto k = cast_from_float<T>(k_f);
  // [3, T]: text tokens carry t == h == w; vision tokens differ per axis.
  const std::vector<std::vector<int64_t>> thw = {{0, 3, 7, 11}, {0, 1, 2, 5}, {0, 2, 4, 6}};

  T* d_q = kops_test::device_alloc<T>(q.size());
  T* d_k = kops_test::device_alloc<T>(k.size());
  T* d_cache = kops_test::device_alloc<T>(static_cast<size_t>(max_position) * rot);
  int64_t* d_pos = kops_test::device_alloc<int64_t>(static_cast<size_t>(3 * rows));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_q, q);
  kops_test::copy_in(d_k, k);
  std::vector<int64_t> flat;
  flat.insert(flat.end(), thw[0].begin(), thw[0].end());
  flat.insert(flat.end(), thw[1].begin(), thw[1].end());
  flat.insert(flat.end(), thw[2].begin(), thw[2].end());
  kops_test::copy_in(d_pos, flat);

  const kops::Tensor t_cache = kops_test::tensor2d(d_cache, dtype, max_position, rot);
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(t_cache, cfg, static_cast<int32_t>(max_position), ctx),
                 "build cache (mrope)");

  kops::Tensor t_q = kops_test::tensor3d(d_q, dtype, rows, heads, head_dim);
  kops::Tensor t_k = kops_test::tensor3d(d_k, dtype, rows, kv_heads, head_dim);
  kops::Tensor t_pos = kops_test::tensor2d(d_pos, kops::DType::kInt64, 3, rows);
  KOPS_EXPECT_OK(
      kops::rope::apply_mrope(t_pos, t_q, t_k, t_cache, section, interleaved, style, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_q = kops_test::copy_out(d_q, q.size());
  std::vector<double> want_q = cast_to_double(q);
  ref_mrope(want_q, rows, heads, head_dim, thw, table, rot, section, interleaved, style);
  kops_test::compare(got_q, want_q, atol, rtol, name);

  const auto got_k = kops_test::copy_out(d_k, k.size());
  std::vector<double> want_k = cast_to_double(k);
  ref_mrope(want_k, rows, kv_heads, head_dim, thw, table, rot, section, interleaved, style);
  kops_test::compare(got_k, want_k, atol, rtol, "apply_mrope.key");

  kops_test::device_free(d_q);
  kops_test::device_free(d_k);
  kops_test::device_free(d_cache);
  kops_test::device_free(d_pos);
}

// =============================================================================
// fused_qk_norm_rope
// =============================================================================

template <typename T>
void test_fused(const char* name, RopeStyle style, kops::DType dtype, int32_t head_dim,
                int32_t rot_dim, int32_t rot_offset, float weight_offset, double atol,
                double rtol) {
  const int64_t rows = 4, num_q_heads = 4, num_kv_heads = 2;
  const int64_t width = (num_q_heads + 2 * num_kv_heads) * head_dim;
  const int64_t max_position = 32;
  const float eps = 1e-6f;

  RopeConfig cfg = base_config(rot_dim, 1.0e4f);
  const HostFreq freq = host_freq(cfg);
  const RefTable table = ref_table(freq.inv_freq, double(freq.mscale), max_position, rot_dim);

  std::vector<float> qkv_f(static_cast<size_t>(rows * width));
  std::vector<float> qw_f(static_cast<size_t>(head_dim));
  std::vector<float> kw_f(static_cast<size_t>(head_dim));
  fill(qkv_f, 61u);
  fill(qw_f, 62u);
  fill(kw_f, 63u);
  for (float& v : qw_f) v += 1.5f;  // keep the weights away from 0
  for (float& v : kw_f) v += 1.5f;
  const auto qkv = cast_from_float<T>(qkv_f);
  const auto qw = cast_from_float<T>(qw_f);
  const auto kw = cast_from_float<T>(kw_f);
  const std::vector<int64_t> positions = {0, 3, 9, 20};

  T* d_qkv = kops_test::device_alloc<T>(qkv.size());
  T* d_qw = kops_test::device_alloc<T>(qw.size());
  T* d_kw = kops_test::device_alloc<T>(kw.size());
  T* d_cache = kops_test::device_alloc<T>(static_cast<size_t>(max_position) * rot_dim);
  int64_t* d_pos = kops_test::device_alloc<int64_t>(positions.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_qkv, qkv);
  kops_test::copy_in(d_qw, qw);
  kops_test::copy_in(d_kw, kw);
  kops_test::copy_in(d_pos, positions);

  const kops::Tensor t_cache = kops_test::tensor2d(d_cache, dtype, max_position, rot_dim);
  KOPS_EXPECT_OK(kops::rope::build_cos_sin_cache(t_cache, cfg, static_cast<int32_t>(max_position), ctx),
                 "build cache (fused)");
  const kops::Tensor t_qkv = kops_test::tensor2d(d_qkv, dtype, rows, width);
  const kops::Tensor t_qw = make1d(d_qw, dtype, head_dim);
  const kops::Tensor t_kw = make1d(d_kw, dtype, head_dim);
  const kops::Tensor t_pos = make1d(d_pos, kops::DType::kInt64, rows);
  KOPS_EXPECT_OK(kops::rope::fused_qk_norm_rope(t_qkv, t_qw, t_kw, t_pos, t_cache,
                                                static_cast<int32_t>(num_q_heads),
                                                static_cast<int32_t>(num_kv_heads), head_dim, rot_dim,
                                                rot_offset, style, weight_offset, eps, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(d_qkv, qkv.size());
  std::vector<double> want = cast_to_double(qkv);
  ref_fused_qk_norm_rope(want, rows, num_q_heads, num_kv_heads, head_dim, cast_to_double(qw),
                         cast_to_double(kw), double(weight_offset), double(eps), positions, table,
                         rot_dim, rot_offset, style);
  kops_test::compare(got, want, atol, rtol, name);

  // V is copied through untouched: the V slice must still hold the input bytes.
  const int64_t v_begin = (num_q_heads + num_kv_heads) * head_dim;
  const int64_t v_end = (num_q_heads + 2 * num_kv_heads) * head_dim;
  std::vector<T> got_v;
  std::vector<double> want_v;
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t j = v_begin; j < v_end; ++j) {
      got_v.push_back(got[t * width + j]);
      want_v.push_back(static_cast<double>(kops_test::to_float(qkv[t * width + j])));
    }
  }
  kops_test::compare(got_v, want_v, 0.0, 0.0, "fused_qk_norm_rope.v_untouched");

  kops_test::device_free(d_qkv);
  kops_test::device_free(d_qw);
  kops_test::device_free(d_kw);
  kops_test::device_free(d_cache);
  kops_test::device_free(d_pos);
}

// =============================================================================
// attn_temperature_tuning
// =============================================================================

template <typename T>
void test_temperature(const char* name, kops::DType dtype, double atol, double rtol) {
  const int64_t rows = 5, heads = 2, head_dim = 16;
  const float floor_scale = 8192.0f;
  const float attn_scale = 0.1f;
  std::vector<float> q_f(static_cast<size_t>(rows * heads * head_dim));
  fill(q_f, 71u);
  const auto q = cast_from_float<T>(q_f);
  const std::vector<int64_t> positions = {0, 100, 8191, 16384, -1};

  T* d_q = kops_test::device_alloc<T>(q.size());
  int64_t* d_pos = kops_test::device_alloc<int64_t>(positions.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_q, q);
  kops_test::copy_in(d_pos, positions);

  const kops::Tensor t_q = kops_test::tensor3d(d_q, dtype, rows, heads, head_dim);
  const kops::Tensor t_pos = make1d(d_pos, kops::DType::kInt64, rows);
  KOPS_EXPECT_OK(kops::rope::attn_temperature_tuning(t_q, t_pos, floor_scale, attn_scale, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(d_q, q.size());
  std::vector<double> want = cast_to_double(q);
  for (int64_t t = 0; t < rows; ++t) {
    const double factor = ref_temperature(positions[t], double(floor_scale), double(attn_scale));
    for (int64_t i = 0; i < heads * head_dim; ++i) want[t * heads * head_dim + i] *= factor;
  }
  kops_test::compare(got, want, atol, rtol, name);

  // p = 16384 gives floor(16385 / 8192) = 2 -> log(3) * 0.1 + 1, i.e. the scale
  // has to grow with the position (this is the property tuned for Llama-4 iRoPE).
  kops_test::device_free(d_q);
  kops_test::device_free(d_pos);
}

// =============================================================================
// Contract cases
// =============================================================================

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  const int64_t tokens = 2, heads = 2, head_dim = 16;
  const int32_t rot = 16;
  std::vector<float> pool(4096, 1.0f);
  float* d = kops_test::device_alloc<float>(pool.size());
  int64_t* d_pos = kops_test::device_alloc<int64_t>(tokens);
  kops_test::copy_in(d, pool);
  const std::vector<int64_t> pos = {0, 1};
  kops_test::copy_in(d_pos, pos);

  const kops::Tensor t_q = kops_test::tensor3d(d, kops::DType::kFloat32, tokens, heads, head_dim);
  const kops::Tensor t_q16 = kops_test::tensor3d(d, kops::DType::kFloat16, tokens, heads, head_dim);
  const kops::Tensor t_pos = make1d(d_pos, kops::DType::kInt64, tokens);
  const kops::Tensor t_cache = kops_test::tensor2d(d, kops::DType::kFloat32, 32, rot);
  kops::Tensor t_cache_padded = t_cache;
  t_cache_padded.stride[0] = rot + 8;
  t_cache_padded.shape[0] = 2;  // keep it inside the pool
  const kops::Tensor t_cache_i8 = kops_test::tensor2d(d, kops::DType::kInt8, 32, rot);
  const kops::Tensor t_pos_f32 = kops_test::tensor2d(d, kops::DType::kFloat32, tokens, 1);
  RopeConfig cfg = base_config(rot, 1.0e4f);

  // odd rot_dim / window beyond the head
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary(t_pos, t_q, kops::Tensor{}, t_cache, kops::Tensor{},
                                              15, 0, RopeStyle::kNeoX, ctx),
                     kops::Status::kUnsupportedShape, "apply_rotary odd rot_dim");
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary(t_pos, t_q, kops::Tensor{}, t_cache, kops::Tensor{},
                                              16, 8, RopeStyle::kNeoX, ctx),
                     kops::Status::kUnsupportedShape, "apply_rotary window > head_dim");
  // unknown style
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary(t_pos, t_q, kops::Tensor{}, t_cache, kops::Tensor{},
                                              rot, 0, static_cast<RopeStyle>(9), ctx),
                     kops::Status::kInvalidArgument, "apply_rotary style");
  // positions dtype
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary(t_pos_f32, t_q, kops::Tensor{}, t_cache,
                                              kops::Tensor{}, rot, 0, RopeStyle::kNeoX, ctx),
                     kops::Status::kUnsupportedDtype, "apply_rotary positions dtype");
  // key dtype must match query
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary(t_pos, t_q, t_q16, t_cache, kops::Tensor{}, rot, 0,
                                              RopeStyle::kNeoX, ctx),
                     kops::Status::kUnsupportedDtype, "apply_rotary key dtype");
  // padded cache rows (no row-stride argument exists)
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary(t_pos, t_q, kops::Tensor{}, t_cache_padded,
                                              kops::Tensor{}, rot, 0, RopeStyle::kNeoX, ctx),
                     kops::Status::kUnsupportedLayout, "apply_rotary cache row stride");
  // cache dtype
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary(t_pos, t_q, kops::Tensor{}, t_cache_i8,
                                              kops::Tensor{}, rot, 0, RopeStyle::kNeoX, ctx),
                     kops::Status::kUnsupportedDtype, "apply_rotary cache dtype");
  // null query
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary(t_pos, kops::Tensor{}, kops::Tensor{}, t_cache,
                                              kops::Tensor{}, rot, 0, RopeStyle::kNeoX, ctx),
                     kops::Status::kInvalidArgument, "apply_rotary null query");

  // on-the-fly: inv_freq must be fp32 [rot_dim/2]
  const kops::Tensor t_inv_bad = make1d(d, kops::DType::kFloat32, rot / 2 - 2);
  KOPS_EXPECT_STATUS(kops::rope::apply_rotary_on_the_fly(t_pos, t_q, kops::Tensor{}, t_inv_bad, 1.0f,
                                                         rot, 0, RopeStyle::kNeoX, ctx),
                     kops::Status::kUnsupportedShape, "on_the_fly inv_freq width");

  // M-RoPE: sections must sum to rot_dim / 2
  const int32_t bad_section[3] = {4, 4, 4};
  const kops::Tensor t_thw = kops_test::tensor2d(d_pos, kops::DType::kInt64, 1, tokens);
  kops::Tensor t_thw3 = kops_test::tensor2d(d_pos, kops::DType::kInt64, 3, tokens);
  const int32_t good_section[3] = {4, 2, 2};
  KOPS_EXPECT_STATUS(kops::rope::apply_mrope(t_thw3, t_q, kops::Tensor{}, t_cache, bad_section,
                                             false, RopeStyle::kNeoX, ctx),
                     kops::Status::kInvalidArgument, "apply_mrope section sum");
  KOPS_EXPECT_STATUS(kops::rope::apply_mrope(t_thw, t_q, kops::Tensor{}, t_cache, good_section,
                                             false, RopeStyle::kNeoX, ctx),
                     kops::Status::kUnsupportedShape, "apply_mrope positions rows");

  // fused: head_dim without an instantiation, and a width mismatch
  const kops::Tensor t_qkv = kops_test::tensor2d(d, kops::DType::kFloat32, tokens, 8 * 16);
  const kops::Tensor t_qkv_96 = kops_test::tensor2d(d, kops::DType::kFloat32, tokens, 8 * 96);
  KOPS_EXPECT_STATUS(kops::rope::fused_qk_norm_rope(t_qkv_96, kops::Tensor{}, kops::Tensor{}, t_pos,
                                                    t_cache, 4, 2, 96, 96, 0, RopeStyle::kNeoX, 0.0f,
                                                    1e-6f, ctx),
                     kops::Status::kUnsupportedShape, "fused head_dim 96");
  KOPS_EXPECT_STATUS(kops::rope::fused_qk_norm_rope(t_qkv, kops::Tensor{}, kops::Tensor{}, t_pos,
                                                    t_cache, 4, 2, 16, 16, 0, RopeStyle::kNeoX, 0.0f,
                                                    1e-6f, ctx),
                     kops::Status::kUnsupportedShape, "fused head_dim 16 (no instantiation)");
  KOPS_EXPECT_STATUS(kops::rope::fused_qk_norm_rope(t_qkv, kops::Tensor{}, kops::Tensor{}, t_pos,
                                                    t_cache, 4, 3, 16, 16, 0, RopeStyle::kNeoX, 0.0f,
                                                    1e-6f, ctx),
                     kops::Status::kUnsupportedShape, "fused qkv width");
  // negative epsilon is rejected, and so is epsilon = NaN
  KOPS_EXPECT_STATUS(kops::rope::fused_qk_norm_rope(t_qkv, kops::Tensor{}, kops::Tensor{}, t_pos,
                                                    t_cache, 4, 2, 64, 16, 0, RopeStyle::kNeoX, 0.0f,
                                                    -1.0f, ctx),
                     kops::Status::kInvalidArgument, "fused negative epsilon");

  // temperature: floor_scale > 0 and contiguous heads
  kops::Tensor t_q_gapped = t_q;
  t_q_gapped.stride[1] = head_dim + 8;
  KOPS_EXPECT_STATUS(kops::rope::attn_temperature_tuning(t_q, t_pos, 0.0f, 0.1f, ctx),
                     kops::Status::kInvalidArgument, "temperature floor_scale");
  KOPS_EXPECT_STATUS(kops::rope::attn_temperature_tuning(t_q_gapped, t_pos, 8192.0f, 0.1f, ctx),
                     kops::Status::kUnsupportedLayout, "temperature head stride");

  // build_cos_sin_cache: rows / width must agree with the config and the tensor
  KOPS_EXPECT_STATUS(kops::rope::build_cos_sin_cache(t_cache, cfg, 16, ctx),
                     kops::Status::kUnsupportedShape, "build_cache max_position");
  RopeConfig cfg64 = base_config(64, 1.0e4f);
  KOPS_EXPECT_STATUS(kops::rope::build_cos_sin_cache(t_cache, cfg64, 32, ctx),
                     kops::Status::kUnsupportedShape, "build_cache cfg width");

  // host math: config and pointer contracts
  float out_inv[8] = {};
  float out_mscale = 0.0f;
  KOPS_EXPECT_STATUS(kops::rope::compute_inv_freq(cfg, nullptr, &out_mscale),
                     kops::Status::kInvalidArgument, "compute_inv_freq null inv_freq");
  KOPS_EXPECT_STATUS(kops::rope::compute_inv_freq(cfg, out_inv, nullptr),
                     kops::Status::kInvalidArgument, "compute_inv_freq null mscale");
  RopeConfig cfg_odd = base_config(15, 1.0e4f);
  KOPS_EXPECT_STATUS(kops::rope::compute_inv_freq(cfg_odd, out_inv, &out_mscale),
                     kops::Status::kUnsupportedShape, "compute_inv_freq odd dim");
  RopeConfig cfg_theta = base_config(8, 0.0f);
  KOPS_EXPECT_STATUS(kops::rope::compute_inv_freq(cfg_theta, out_inv, &out_mscale),
                     kops::Status::kInvalidArgument, "compute_inv_freq theta 0");
  RopeConfig cfg_dntk = base_config(8, 1.0e4f);
  cfg_dntk.scaling = RopeScaling::kDynamicNTK;
  KOPS_EXPECT_STATUS(kops::rope::compute_inv_freq(cfg_dntk, out_inv, &out_mscale),
                     kops::Status::kInvalidArgument, "compute_inv_freq dynamic ntk needs L0");
  RopeConfig cfg_yarn = base_config(8, 1.0e4f);
  cfg_yarn.scaling = RopeScaling::kYaRN;
  KOPS_EXPECT_STATUS(kops::rope::compute_inv_freq(cfg_yarn, out_inv, &out_mscale),
                     kops::Status::kInvalidArgument, "compute_inv_freq yarn needs L0");
  RopeConfig cfg_long = base_config(8, 1.0e4f);
  cfg_long.scaling = RopeScaling::kLongRoPE;
  cfg_long.original_max_position = 4096;
  KOPS_EXPECT_STATUS(kops::rope::compute_inv_freq(cfg_long, out_inv, &out_mscale),
                     kops::Status::kInvalidArgument, "compute_inv_freq longrope needs a table");

  KOPS_EXPECT_STATUS(kops::rope::compute_alibi_slopes(0, 8.0f, out_inv),
                     kops::Status::kInvalidArgument, "alibi num_heads 0");
  KOPS_EXPECT_STATUS(kops::rope::compute_alibi_slopes(8, 0.0f, out_inv),
                     kops::Status::kInvalidArgument, "alibi max_bias 0");
  KOPS_EXPECT_STATUS(kops::rope::compute_alibi_slopes(8, 8.0f, nullptr),
                     kops::Status::kInvalidArgument, "alibi null out");

  kops_test::device_free(d);
  kops_test::device_free(d_pos);
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;

  test_inv_freq();
  test_alibi_slopes();
  test_build_cache<float>("build_cos_sin_cache.fp32", kops::DType::kFloat32, 1e-6, 1e-6);
  test_build_cache<__nv_bfloat16>("build_cos_sin_cache.bf16", kops::DType::kBFloat16, 1e-2, 1e-2);
  test_build_cache_isolated();
  test_apply_rotary();
  test_partial_rot<float>("apply_rotary.partial.fp32.neox", RopeStyle::kNeoX,
                          kops::DType::kFloat32, 1e-6, 1e-5);
  test_partial_rot<float>("apply_rotary.partial.fp32.gptj", RopeStyle::kGPTJ,
                          kops::DType::kFloat32, 1e-6, 1e-5);
  test_partial_rot<__nv_bfloat16>("apply_rotary.partial.bf16.neox", RopeStyle::kNeoX,
                                  kops::DType::kBFloat16, 2e-2, 1e-2);
  run_strided_offsets<float>("apply_rotary.strided.offsets.fp32", RopeStyle::kNeoX,
                             kops::DType::kFloat32, 1e-6, 1e-5);
  run_strided_offsets<__nv_bfloat16>("apply_rotary.strided.offsets.bf16", RopeStyle::kGPTJ,
                                     kops::DType::kBFloat16, 2e-2, 1e-2);
  test_on_the_fly<float>("apply_rotary_on_the_fly.fp32.neox", RopeStyle::kNeoX,
                         kops::DType::kFloat32, 1e-6, 1e-5);
  test_on_the_fly<__nv_bfloat16>("apply_rotary_on_the_fly.bf16.gptj", RopeStyle::kGPTJ,
                                 kops::DType::kBFloat16, 2e-2, 1e-2);
  test_mrope<float>("apply_mrope.contiguous.fp32.neox", RopeStyle::kNeoX, kops::DType::kFloat32,
                    false, 1e-6, 1e-5);
  test_mrope<float>("apply_mrope.interleaved.fp32.gptj", RopeStyle::kGPTJ, kops::DType::kFloat32,
                    true, 1e-6, 1e-5);
  test_mrope<__nv_bfloat16>("apply_mrope.contiguous.bf16.neox", RopeStyle::kNeoX,
                            kops::DType::kBFloat16, false, 2e-2, 1e-2);
  test_mrope<__nv_bfloat16>("apply_mrope.interleaved.bf16.gptj", RopeStyle::kGPTJ,
                            kops::DType::kBFloat16, true, 2e-2, 1e-2);
  test_fused<float>("fused_qk_norm_rope.fp32.d128.neox", RopeStyle::kNeoX, kops::DType::kFloat32,
                    128, 128, 0, 0.0f, 1e-5, 1e-4);
  test_fused<__nv_bfloat16>("fused_qk_norm_rope.bf16.d128.gptj", RopeStyle::kGPTJ,
                            kops::DType::kBFloat16, 128, 128, 0, 0.0f, 2e-2, 2e-2);
  test_fused<float>("fused_qk_norm_rope.fp32.d64.partial", RopeStyle::kNeoX,
                    kops::DType::kFloat32, 64, 32, 32, 1.0f, 1e-5, 1e-4);
  test_fused<__nv_bfloat16>("fused_qk_norm_rope.bf16.d256.gptj", RopeStyle::kGPTJ,
                            kops::DType::kBFloat16, 256, 256, 0, 0.0f, 2e-2, 2e-2);
  test_fused<float>("fused_qk_norm_rope.fp32.d256.offset64", RopeStyle::kGPTJ,
                    kops::DType::kFloat32, 256, 64, 192, 1.0f, 1e-5, 1e-4);
  test_temperature<float>("attn_temperature_tuning.fp32", kops::DType::kFloat32, 1e-6, 1e-6);
  test_temperature<__nv_bfloat16>("attn_temperature_tuning.bf16", kops::DType::kBFloat16, 1e-2,
                                  1e-2);
  test_contracts();
  return kops_test::finish("test_position");
}
