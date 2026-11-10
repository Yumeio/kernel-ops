// =============================================================================
// kernel-ops | tests/unit/test_norm.cc
// B1 norm family:
//   rms_norm, fused_add_rms_norm            (pre-existing entries, re-checked)
//   fused_add_rms_norm_out, layer_norm, fused_add_layer_norm, qk_norm,
//   l2_norm, gated_rms_norm                 (this step)
//
// Oracle: csrc/reference (fp64) for rms_norm; the other norms compare against
// local fp64 row oracles below, because csrc/reference only implements
// rms_norm_f64. Every case uses a padded row stride where noted, so the code
// paths for hidden % VEC != 0 and for non-contiguous views are exercised.
// =============================================================================

#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"

#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"
#include "csrc/reference/ops.h"

namespace {

constexpr float kEps = 1e-5f;

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

void fill(std::vector<float>& values, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& v : values) v = dist(rng);
}

template <typename T>
std::vector<T> cast_from_float(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(out[i], src[i]);
  return out;
}

template <typename T>
std::vector<double> as_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<double>(kops_test::to_float(src[i]));
  return out;
}

template <typename T>
std::vector<T> compact(const std::vector<T>& buf, int64_t rows, int64_t cols, int64_t stride) {
  std::vector<T> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(r * cols + c)] = buf[static_cast<size_t>(r * stride + c)];
    }
  }
  return out;
}

kops::Tensor tensor2d_strided(void* data, kops::DType dtype, int64_t rows, int64_t cols,
                              int64_t row_stride) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = row_stride;
  t.stride[1] = 1;
  return t;
}

kops::Tensor tensor1d(void* data, kops::DType dtype, int64_t n) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

template <typename T>
struct DevBuf {
  T* p = nullptr;
  explicit DevBuf(size_t n) : p(kops_test::device_alloc<T>(n)) {}
  ~DevBuf() { kops_test::device_free(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  bool ok() const { return p != nullptr; }
};

// -----------------------------------------------------------------------------
// fp64 row oracles (compact [rows, cols] output, strided input)
// -----------------------------------------------------------------------------

void ref_rms(const std::vector<double>& x, int64_t rows, int64_t cols, int64_t stride,
             const double* w, int64_t groups, int64_t w_stride, double w_off, double eps,
             std::vector<double>& out) {
  out.assign(static_cast<size_t>(rows * cols), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    const double* xr = x.data() + r * stride;
    double ss = 0.0;
    for (int64_t c = 0; c < cols; ++c) ss += xr[c] * xr[c];
    const double rstd = 1.0 / std::sqrt(ss / static_cast<double>(cols) + eps);
    const double* wr = (w != nullptr) ? w + (r % groups) * w_stride : nullptr;
    for (int64_t c = 0; c < cols; ++c) {
      double v = xr[c] * rstd;
      if (wr != nullptr) v *= wr[c] + w_off;
      out[static_cast<size_t>(r * cols + c)] = v;
    }
  }
}

void ref_layer_norm(const std::vector<double>& x, int64_t rows, int64_t cols, int64_t stride,
                    const double* w, const double* b, double w_off, double eps,
                    std::vector<double>& out) {
  out.assign(static_cast<size_t>(rows * cols), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    const double* xr = x.data() + r * stride;
    double sum = 0.0;
    for (int64_t c = 0; c < cols; ++c) sum += xr[c];
    const double mean = sum / static_cast<double>(cols);
    double var = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const double d = xr[c] - mean;
      var += d * d;
    }
    const double rstd = 1.0 / std::sqrt(var / static_cast<double>(cols) + eps);
    for (int64_t c = 0; c < cols; ++c) {
      double v = (xr[c] - mean) * rstd;
      if (w != nullptr) v *= w[c] + w_off;
      if (b != nullptr) v += b[c];
      out[static_cast<size_t>(r * cols + c)] = v;
    }
  }
}

void ref_l2(const std::vector<double>& x, int64_t rows, int64_t cols, int64_t stride, double scale,
            double eps, std::vector<double>& out) {
  out.assign(static_cast<size_t>(rows * cols), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    const double* xr = x.data() + r * stride;
    double ss = 0.0;
    for (int64_t c = 0; c < cols; ++c) ss += xr[c] * xr[c];
    const double inv = 1.0 / std::sqrt(ss + eps);
    for (int64_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(r * cols + c)] = xr[c] * inv * scale;
    }
  }
}

/// Residual add (rounded to the residual dtype by the kernel) + norm.
void ref_fused(const std::vector<double>& x, const std::vector<double>& res, int64_t rows,
               int64_t cols, int64_t stride, double input_scale, bool layer_norm,
               const double* w, const double* b, double w_off, double eps,
               std::vector<double>& res_out, std::vector<double>& norm_out) {
  std::vector<double> sum(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      sum[static_cast<size_t>(r * cols + c)] =
          res[static_cast<size_t>(r * stride + c)] +
          input_scale * x[static_cast<size_t>(r * stride + c)];
    }
  }
  res_out = sum;
  if (layer_norm) {
    ref_layer_norm(sum, rows, cols, cols, w, b, w_off, eps, norm_out);
  } else {
    ref_rms(sum, rows, cols, cols, w, 1, 0, w_off, eps, norm_out);
  }
}

double gate_value(double z, int act) {
  if (act == 0) return z;  // identity
  const double e = std::exp(-std::fabs(z));
  const double s = (z >= 0.0) ? 1.0 / (1.0 + e) : e / (1.0 + e);
  if (act == 1) return s;  // sigmoid
  return z * s;            // silu
}

void ref_gated(const std::vector<double>& x, const std::vector<double>& z, int64_t rows,
               int64_t cols, int64_t x_stride, int64_t z_stride, const double* w, int64_t w_stride,
               int64_t group, int act, bool before, double w_off, double eps,
               std::vector<double>& out) {
  out.assign(static_cast<size_t>(rows * cols), 0.0);
  const int64_t groups = cols / group;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t g = 0; g < groups; ++g) {
      const int64_t base = r * cols + g * group;
      const double* xr = x.data() + r * x_stride + g * group;
      const double* zr = z.data() + r * z_stride + g * group;
      const double* wr = (w != nullptr) ? w + g * w_stride : nullptr;
      double ss = 0.0;
      for (int64_t c = 0; c < group; ++c) {
        double v = xr[c];
        if (!before) v *= gate_value(zr[c], act);
        ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / static_cast<double>(group) + eps);
      for (int64_t c = 0; c < group; ++c) {
        double v = xr[c];
        const double gv = gate_value(zr[c], act);
        if (before) {
          v = v * inv;
          if (wr != nullptr) v *= wr[c] + w_off;
          v *= gv;
        } else {
          v = (v * gv) * inv;
          if (wr != nullptr) v *= wr[c] + w_off;
        }
        out[static_cast<size_t>(base + c)] = v;
      }
    }
  }
}

/// In-place norm of one head row (kind: 0 rms, 1 layer, 2 l2).
void ref_norm_vec(std::vector<double>& v, const double* w, double w_off, int kind, double eps,
                  double mean_square) {
  const int64_t n = static_cast<int64_t>(v.size());
  std::vector<double> out(v.size());
  if (kind == 1) {
    double sum = 0.0;
    for (double x : v) sum += x;
    const double mean = sum / static_cast<double>(n);
    double var = 0.0;
    for (double x : v) {
      const double d = x - mean;
      var += d * d;
    }
    const double rstd = 1.0 / std::sqrt(var / static_cast<double>(n) + eps);
    for (int64_t c = 0; c < n; ++c) {
      double y = (v[c] - mean) * rstd;
      if (w != nullptr) y *= w[c] + w_off;
      out[c] = y;
    }
  } else {
    double ss = 0.0;
    for (double x : v) ss += x * x;
    const double denom = (kind == 2) ? ss : (mean_square >= 0.0 ? mean_square * n : ss);
    const double inv = 1.0 / std::sqrt(denom / ((kind == 2) ? 1.0 : static_cast<double>(n)) + eps);
    for (int64_t c = 0; c < n; ++c) {
      double y = v[c] * inv;
      if (w != nullptr) y *= w[c] + w_off;
      out[c] = y;
    }
  }
  v = out;
}

/// Per-row weight groups for rms_norm: [groups, hidden] with a possibly padded row.
struct Weight2d {
  std::vector<float> host;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t stride = 0;
};

Weight2d make_weight2d(int64_t groups, int64_t cols, int64_t pad, unsigned seed) {
  Weight2d w;
  w.rows = groups;
  w.cols = cols;
  w.stride = cols + pad;
  w.host.assign(static_cast<size_t>(groups * w.stride), 7.0f);  // pad = poison
  std::vector<float> body(static_cast<size_t>(groups * cols));
  fill(body, seed);
  for (int64_t g = 0; g < groups; ++g) {
    for (int64_t c = 0; c < cols; ++c) {
      w.host[static_cast<size_t>(g * w.stride + c)] = body[static_cast<size_t>(g * cols + c)];
    }
  }
  return w;
}

// -----------------------------------------------------------------------------
// rms_norm / fused_add_rms_norm (pre-existing entries)
// -----------------------------------------------------------------------------

template <typename T>
void test_rms_norm(const char* name, kops::DType dtype, int64_t rows, int64_t cols, int64_t pad,
                   int64_t groups, bool with_w, float w_off, double atol, double rtol) {
  const int64_t stride = cols + pad;
  std::vector<float> body(static_cast<size_t>(rows * stride), 7.0f);  // pad = poison
  std::vector<float> src(static_cast<size_t>(rows * cols));
  fill(src, 11u);
  for (int64_t r = 0; r < rows; ++r) {
    std::copy(src.begin() + r * cols, src.begin() + (r + 1) * cols, body.begin() + r * stride);
  }
  const std::vector<T> in = cast_from_float<T>(body);
  const Weight2d w2 = make_weight2d(with_w ? groups : 0, cols, pad, 12u);
  const std::vector<T> w = cast_from_float<T>(with_w ? w2.host : std::vector<float>{});

  DevBuf<T> d_in(in.size());
  DevBuf<T> d_w(std::max<size_t>(w.size(), 1));
  DevBuf<T> d_out(in.size());
  if (!d_in.ok() || !d_w.ok() || !d_out.ok()) return;
  kops_test::copy_in(d_in.p, in);
  if (!w.empty()) kops_test::copy_in(d_w.p, w);

  kops::Tensor t_in = tensor2d_strided(d_in.p, dtype, rows, cols, stride);
  kops::Tensor t_out = tensor2d_strided(d_out.p, dtype, rows, cols, stride);
  kops::Tensor t_w{};
  if (with_w) {
    t_w = tensor2d_strided(d_w.p, dtype, groups, cols, w2.stride);
  }
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::rms_norm(t_out, t_in, t_w, kEps, w_off, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = compact(kops_test::copy_out(d_out.p, in.size()), rows, cols, stride);
  const auto in_d = as_double(in);
  const auto w_d = as_double(w);
  std::vector<double> want;
  ref_rms(in_d, rows, cols, stride, with_w ? w_d.data() : nullptr, with_w ? groups : 1,
          with_w ? w2.stride : 0, w_off, kEps, want);
  kops_test::compare(got, want, atol, rtol, name);

  // Cross-check the fp64 reference implementation for the contiguous-group case.
  if (with_w && groups == 1 && pad == 0) {
    std::vector<double> want_ref(static_cast<size_t>(rows * cols));
    KOPS_EXPECT_OK(kops::ref::rms_norm_f64(in_d.data(), stride, w_d.data(), w_off, kEps, rows, cols,
                                          want_ref.data(), cols),
                   "rms_norm_f64");
    kops_test::compare(got, want_ref, atol, rtol, "rms_norm vs ref::rms_norm_f64");
  }
}

template <typename T, typename R>
void test_fused_add_rms_norm(const char* name, kops::DType dtype, kops::DType rdtype, int64_t rows,
                             int64_t cols, float input_scale, bool with_w, float w_off,
                             double atol, double rtol) {
  std::vector<float> x_f(static_cast<size_t>(rows * cols)), res_f(x_f.size()), w_f(cols);
  fill(x_f, 13u);
  fill(res_f, 14u);
  fill(w_f, 15u);
  const std::vector<T> x = cast_from_float<T>(x_f);
  const std::vector<R> res = cast_from_float<R>(res_f);
  const std::vector<T> w = cast_from_float<T>(with_w ? w_f : std::vector<float>{});

  DevBuf<T> d_x(x.size());
  DevBuf<R> d_res(res.size());
  DevBuf<T> d_w(std::max<size_t>(w.size(), 1));
  if (!d_x.ok() || !d_res.ok() || !d_w.ok()) return;
  kops_test::copy_in(d_x.p, x);
  kops_test::copy_in(d_res.p, res);
  if (!w.empty()) kops_test::copy_in(d_w.p, w);

  kops::Tensor t_x = tensor2d_strided(d_x.p, dtype, rows, cols, cols);
  kops::Tensor t_res = tensor2d_strided(d_res.p, rdtype, rows, cols, cols);
  kops::Tensor t_w = with_w ? tensor1d(d_w.p, dtype, cols) : kops::Tensor{};
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::fused_add_rms_norm(t_x, t_res, t_w, kEps, w_off, input_scale, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_x = kops_test::copy_out(d_x.p, x.size());
  const auto got_res = kops_test::copy_out(d_res.p, res.size());
  const auto x_d = as_double(x);
  const auto res_d = as_double(res);
  const auto w_d = as_double(w);
  std::vector<double> res_want, norm_want;
  ref_fused(x_d, res_d, rows, cols, cols, input_scale, false, with_w ? w_d.data() : nullptr,
            nullptr, w_off, kEps, res_want, norm_want);
  kops_test::compare(got_res, res_want, atol, rtol, name);
  kops_test::compare(got_x, norm_want, atol, rtol, name);
}

// -----------------------------------------------------------------------------
// fused_add_rms_norm_out
// -----------------------------------------------------------------------------

template <typename T, typename R>
void test_fused_add_rms_norm_out(const char* name, kops::DType dtype, kops::DType rdtype,
                                 int64_t rows, int64_t cols, float input_scale, bool with_w,
                                 float w_off, bool alias_residual, bool alias_normed, double atol,
                                 double rtol) {
  std::vector<float> x_f(static_cast<size_t>(rows * cols)), res_f(x_f.size()), w_f(cols);
  fill(x_f, 21u);
  fill(res_f, 22u);
  fill(w_f, 23u);
  const std::vector<T> x = cast_from_float<T>(x_f);
  const std::vector<R> res = cast_from_float<R>(res_f);
  const std::vector<T> w = cast_from_float<T>(with_w ? w_f : std::vector<float>{});

  DevBuf<T> d_x(x.size());
  DevBuf<R> d_res(res.size());
  DevBuf<T> d_out(x.size());
  DevBuf<R> d_rout(res.size());
  DevBuf<T> d_w(std::max<size_t>(w.size(), 1));
  if (!d_x.ok() || !d_res.ok() || !d_out.ok() || !d_rout.ok() || !d_w.ok()) return;
  kops_test::copy_in(d_x.p, x);
  kops_test::copy_in(d_res.p, res);
  if (!w.empty()) kops_test::copy_in(d_w.p, w);

  void* out_ptr = alias_normed ? static_cast<void*>(d_x.p) : static_cast<void*>(d_out.p);
  void* rout_ptr = alias_residual ? static_cast<void*>(d_res.p) : static_cast<void*>(d_rout.p);
  kops::Tensor t_in = tensor2d_strided(d_x.p, dtype, rows, cols, cols);
  kops::Tensor t_out = tensor2d_strided(out_ptr, dtype, rows, cols, cols);
  kops::Tensor t_rin = tensor2d_strided(d_res.p, rdtype, rows, cols, cols);
  kops::Tensor t_rout = tensor2d_strided(rout_ptr, rdtype, rows, cols, cols);
  kops::Tensor t_w = with_w ? tensor1d(d_w.p, dtype, cols) : kops::Tensor{};
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::fused_add_rms_norm_out(t_out, t_rout, t_in, t_rin, t_w, kEps, w_off,
                                                    input_scale, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_out = kops_test::copy_out(static_cast<T*>(out_ptr), x.size());
  const auto got_rout = kops_test::copy_out(static_cast<R*>(rout_ptr), res.size());
  const auto x_d = as_double(x);
  const auto res_d = as_double(res);
  const auto w_d = as_double(w);
  std::vector<double> res_want, norm_want;
  ref_fused(x_d, res_d, rows, cols, cols, input_scale, false, with_w ? w_d.data() : nullptr,
            nullptr, w_off, kEps, res_want, norm_want);
  kops_test::compare(got_rout, res_want, atol, rtol, name);
  kops_test::compare(got_out, norm_want, atol, rtol, name);

  // The out-of-place variant must agree with the in-place one on fresh inputs.
  DevBuf<T> d_x2(x.size());
  DevBuf<R> d_res2(res.size());
  if (!d_x2.ok() || !d_res2.ok()) return;
  kops_test::copy_in(d_x2.p, x);
  kops_test::copy_in(d_res2.p, res);
  kops::Tensor t_x2 = tensor2d_strided(d_x2.p, dtype, rows, cols, cols);
  kops::Tensor t_res2 = tensor2d_strided(d_res2.p, rdtype, rows, cols, cols);
  KOPS_EXPECT_OK(
      kops::norm::fused_add_rms_norm(t_x2, t_res2, t_w, kEps, w_off, input_scale, ctx),
      "fused_add_rms_norm (equivalence)");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto inplace_out = kops_test::copy_out(d_x2.p, x.size());
  const auto inplace_res = kops_test::copy_out(d_res2.p, res.size());
  kops_test::compare(got_out, as_double(inplace_out), 1e-6, 1e-6, name);
  kops_test::compare(got_rout, as_double(inplace_res), 1e-6, 1e-6, name);
}

// -----------------------------------------------------------------------------
// layer_norm
// -----------------------------------------------------------------------------

template <typename T>
void test_layer_norm(const char* name, kops::DType dtype, int64_t rows, int64_t cols, int64_t pad,
                     bool with_w, bool with_b, float w_off, bool inplace, double atol, double rtol) {
  const int64_t stride = cols + pad;
  std::vector<float> body(static_cast<size_t>(rows * stride), 7.0f);
  std::vector<float> src(static_cast<size_t>(rows * cols));
  fill(src, 31u);
  for (int64_t r = 0; r < rows; ++r) {
    std::copy(src.begin() + r * cols, src.begin() + (r + 1) * cols, body.begin() + r * stride);
  }
  std::vector<float> w_f(cols), b_f(cols);
  fill(w_f, 32u);
  fill(b_f, 33u);
  const std::vector<T> in = cast_from_float<T>(body);
  const std::vector<T> w = cast_from_float<T>(with_w ? w_f : std::vector<float>{});
  const std::vector<T> b = cast_from_float<T>(with_b ? b_f : std::vector<float>{});

  DevBuf<T> d_in(in.size());
  DevBuf<T> d_out(inplace ? 1 : in.size());
  DevBuf<T> d_w(std::max<size_t>(w.size(), 1));
  DevBuf<T> d_b(std::max<size_t>(b.size(), 1));
  if (!d_in.ok() || !d_out.ok() || !d_w.ok() || !d_b.ok()) return;
  kops_test::copy_in(d_in.p, in);
  if (!w.empty()) kops_test::copy_in(d_w.p, w);
  if (!b.empty()) kops_test::copy_in(d_b.p, b);

  kops::Tensor t_in = tensor2d_strided(d_in.p, dtype, rows, cols, stride);
  kops::Tensor t_out = inplace ? t_in : tensor2d_strided(d_out.p, dtype, rows, cols, stride);
  kops::Tensor t_w = with_w ? tensor1d(d_w.p, dtype, cols) : kops::Tensor{};
  kops::Tensor t_b = with_b ? tensor1d(d_b.p, dtype, cols) : kops::Tensor{};
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::layer_norm(t_out, t_in, t_w, t_b, kEps, w_off, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const T* src_ptr = inplace ? d_in.p : d_out.p;
  const auto got = compact(kops_test::copy_out(src_ptr, in.size()), rows, cols, stride);
  const auto in_d = as_double(in);
  const auto w_d = as_double(w);
  const auto b_d = as_double(b);
  std::vector<double> want;
  ref_layer_norm(in_d, rows, cols, stride, with_w ? w_d.data() : nullptr,
                 with_b ? b_d.data() : nullptr, w_off, kEps, want);
  kops_test::compare(got, want, atol, rtol, name);
}

// -----------------------------------------------------------------------------
// fused_add_layer_norm
// -----------------------------------------------------------------------------

template <typename T, typename R>
void test_fused_add_layer_norm(const char* name, kops::DType dtype, kops::DType rdtype,
                               int64_t rows, int64_t cols, float input_scale, bool with_w,
                               bool with_b, float w_off, double atol, double rtol) {
  std::vector<float> x_f(static_cast<size_t>(rows * cols)), res_f(x_f.size()), w_f(cols), b_f(cols);
  fill(x_f, 41u);
  fill(res_f, 42u);
  fill(w_f, 43u);
  fill(b_f, 44u);
  const std::vector<T> x = cast_from_float<T>(x_f);
  const std::vector<R> res = cast_from_float<R>(res_f);
  const std::vector<T> w = cast_from_float<T>(with_w ? w_f : std::vector<float>{});
  const std::vector<T> b = cast_from_float<T>(with_b ? b_f : std::vector<float>{});

  DevBuf<T> d_x(x.size());
  DevBuf<R> d_res(res.size());
  DevBuf<T> d_w(std::max<size_t>(w.size(), 1));
  DevBuf<T> d_b(std::max<size_t>(b.size(), 1));
  if (!d_x.ok() || !d_res.ok() || !d_w.ok() || !d_b.ok()) return;
  kops_test::copy_in(d_x.p, x);
  kops_test::copy_in(d_res.p, res);
  if (!w.empty()) kops_test::copy_in(d_w.p, w);
  if (!b.empty()) kops_test::copy_in(d_b.p, b);

  kops::Tensor t_x = tensor2d_strided(d_x.p, dtype, rows, cols, cols);
  kops::Tensor t_res = tensor2d_strided(d_res.p, rdtype, rows, cols, cols);
  kops::Tensor t_w = with_w ? tensor1d(d_w.p, dtype, cols) : kops::Tensor{};
  kops::Tensor t_b = with_b ? tensor1d(d_b.p, dtype, cols) : kops::Tensor{};
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::fused_add_layer_norm(t_x, t_res, t_w, t_b, kEps, w_off, input_scale, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_x = kops_test::copy_out(d_x.p, x.size());
  const auto got_res = kops_test::copy_out(d_res.p, res.size());
  const auto x_d = as_double(x);
  const auto res_d = as_double(res);
  const auto w_d = as_double(w);
  const auto b_d = as_double(b);
  std::vector<double> res_want, norm_want;
  ref_fused(x_d, res_d, rows, cols, cols, input_scale, true, with_w ? w_d.data() : nullptr,
            with_b ? b_d.data() : nullptr, w_off, kEps, res_want, norm_want);
  kops_test::compare(got_res, res_want, atol, rtol, name);
  kops_test::compare(got_x, norm_want, atol, rtol, name);
}

// -----------------------------------------------------------------------------
// qk_norm
// -----------------------------------------------------------------------------

/// Packed QKV, one norm per head: q [hq, d], k [hkv, d], v untouched.
template <typename T>
void test_qk_packed(const char* name, kops::DType dtype, int32_t hq, int32_t hkv, int32_t d,
                    int64_t tokens, int64_t pad, bool with_weight, bool per_head_weight,
                    kops::NormType norm, float w_off, double atol, double rtol) {
  const int64_t width = static_cast<int64_t>(hq + 2 * hkv) * d;
  const int64_t stride = width + pad;
  std::vector<float> body(static_cast<size_t>(tokens * stride), 7.0f);
  std::vector<float> src(static_cast<size_t>(tokens * width));
  fill(src, 51u);
  for (int64_t t = 0; t < tokens; ++t) {
    std::copy(src.begin() + t * width, src.begin() + (t + 1) * width, body.begin() + t * stride);
  }
  const std::vector<T> qkv = cast_from_float<T>(body);
  std::vector<float> qw_f(with_weight ? (per_head_weight ? static_cast<size_t>(hq) * d : d) : 0);
  std::vector<float> kw_f(with_weight ? (per_head_weight ? static_cast<size_t>(hkv) * d : d) : 0);
  fill(qw_f, 52u);
  fill(kw_f, 53u);
  const std::vector<T> qw = cast_from_float<T>(qw_f);
  const std::vector<T> kw = cast_from_float<T>(kw_f);

  DevBuf<T> d_qkv(qkv.size());
  DevBuf<T> d_qw(std::max<size_t>(qw.size(), 1));
  DevBuf<T> d_kw(std::max<size_t>(kw.size(), 1));
  if (!d_qkv.ok() || !d_qw.ok() || !d_kw.ok()) return;
  kops_test::copy_in(d_qkv.p, qkv);
  if (with_weight) {
    kops_test::copy_in(d_qw.p, qw);
    kops_test::copy_in(d_kw.p, kw);
  }

  const kops::Tensor t_qkv = tensor2d_strided(d_qkv.p, dtype, tokens, width, stride);
  const kops::Tensor t_qw =
      !with_weight ? kops::Tensor{}
                   : (per_head_weight ? tensor2d_strided(d_qw.p, dtype, hq, d, d)
                                      : tensor1d(d_qw.p, dtype, d));
  const kops::Tensor t_kw =
      !with_weight ? kops::Tensor{}
                   : (per_head_weight ? tensor2d_strided(d_kw.p, dtype, hkv, d, d)
                                      : tensor1d(d_kw.p, dtype, d));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw, t_kw, hq, hkv, d, norm, false,
                                     kops::Tensor{}, kEps, w_off, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_buf = kops_test::copy_out(d_qkv.p, qkv.size());
  const auto got = compact(got_buf, tokens, width, stride);

  // Expected: q and k normalized per head, v copied verbatim.
  const auto qkv_d = as_double(qkv);
  const auto qw_d = as_double(qw);
  const auto kw_d = as_double(kw);
  const int kind = (norm == kops::NormType::kLayerNorm) ? 1
                   : (norm == kops::NormType::kL2Norm)     ? 2
                                                          : 0;
  std::vector<double> want(static_cast<size_t>(tokens * width), 0.0);
  for (int64_t t = 0; t < tokens; ++t) {
    for (int32_t h = 0; h < hq + 2 * hkv; ++h) {
      std::vector<double> head(d);
      const int64_t head_off = static_cast<int64_t>(h) * d;
      for (int32_t c = 0; c < d; ++c) {
        head[c] = qkv_d[static_cast<size_t>(t * stride + head_off + c)];
      }
      const bool is_qk = h < hq + hkv;
      const double* wh = nullptr;
      if (with_weight && h < hq) {
        wh = qw_d.data() + (per_head_weight ? static_cast<int64_t>(h) * d : 0);
      } else if (with_weight && h < hq + hkv) {
        wh = kw_d.data() + (per_head_weight ? static_cast<int64_t>(h - hq) * d : 0);
      }
      if (is_qk) ref_norm_vec(head, wh, w_off, kind, kEps, -1.0);
      for (int32_t c = 0; c < d; ++c) {
        want[static_cast<size_t>(t * width + head_off + c)] = head[c];
      }
    }
  }
  kops_test::compare(got, want, atol, rtol, name);

  // V must be bit-identical to the input.
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t c = static_cast<int64_t>(hq + hkv) * d; c < width; ++c) {
      KOPS_TEST_CHECK(got[static_cast<size_t>(t * width + c)] ==
                      qkv[static_cast<size_t>(t * stride + c)]);
    }
  }
}

/// Separate q and k tensors with independent token strides.
template <typename T>
void test_qk_split(const char* name, kops::DType dtype, int32_t hq, int32_t hkv, int32_t d,
                   int64_t tokens, int64_t q_pad, int64_t k_pad, kops::NormType norm, float w_off,
                   double atol, double rtol) {
  const int64_t q_width = static_cast<int64_t>(hq) * d;
  const int64_t k_width = static_cast<int64_t>(hkv) * d;
  const int64_t q_stride = q_width + q_pad;
  const int64_t k_stride = k_width + k_pad;
  std::vector<float> q_f(static_cast<size_t>(tokens * q_stride), 7.0f);
  std::vector<float> k_f(static_cast<size_t>(tokens * k_stride), 7.0f);
  std::vector<float> q_body(static_cast<size_t>(tokens * q_width));
  std::vector<float> k_body(static_cast<size_t>(tokens * k_width));
  fill(q_body, 61u);
  fill(k_body, 62u);
  for (int64_t t = 0; t < tokens; ++t) {
    std::copy(q_body.begin() + t * q_width, q_body.begin() + (t + 1) * q_width,
              q_f.begin() + t * q_stride);
    std::copy(k_body.begin() + t * k_width, k_body.begin() + (t + 1) * k_width,
              k_f.begin() + t * k_stride);
  }
  std::vector<float> w_f(d);
  fill(w_f, 63u);
  const std::vector<T> q = cast_from_float<T>(q_f);
  const std::vector<T> k = cast_from_float<T>(k_f);
  const std::vector<T> w = cast_from_float<T>(w_f);

  DevBuf<T> d_q(q.size());
  DevBuf<T> d_k(k.size());
  DevBuf<T> d_w(w.size());
  if (!d_q.ok() || !d_k.ok() || !d_w.ok()) return;
  kops_test::copy_in(d_q.p, q);
  kops_test::copy_in(d_k.p, k);
  kops_test::copy_in(d_w.p, w);

  const kops::Tensor t_q = tensor2d_strided(d_q.p, dtype, tokens, q_width, q_stride);
  const kops::Tensor t_k = tensor2d_strided(d_k.p, dtype, tokens, k_width, k_stride);
  const kops::Tensor t_w = tensor1d(d_w.p, dtype, d);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::qk_norm(t_q, t_k, t_w, t_w, hq, hkv, d, norm, false, kops::Tensor{},
                                     kEps, w_off, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_q = compact(kops_test::copy_out(d_q.p, q.size()), tokens, q_width, q_stride);
  const auto got_k = compact(kops_test::copy_out(d_k.p, k.size()), tokens, k_width, k_stride);
  const auto q_d = as_double(q);
  const auto k_d = as_double(k);
  const auto w_d = as_double(w);
  const int kind = (norm == kops::NormType::kLayerNorm) ? 1
                   : (norm == kops::NormType::kL2Norm)     ? 2
                                                          : 0;
  std::vector<double> want_q(static_cast<size_t>(tokens * q_width));
  std::vector<double> want_k(static_cast<size_t>(tokens * k_width));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int32_t h = 0; h < hq; ++h) {
      std::vector<double> head(d);
      for (int32_t c = 0; c < d; ++c) {
        head[c] = q_d[static_cast<size_t>(t * q_stride + h * d + c)];
      }
      ref_norm_vec(head, w_d.data(), w_off, kind, kEps, -1.0);
      for (int32_t c = 0; c < d; ++c) {
        want_q[static_cast<size_t>(t * q_width + h * d + c)] = head[c];
      }
    }
    for (int32_t h = 0; h < hkv; ++h) {
      std::vector<double> head(d);
      for (int32_t c = 0; c < d; ++c) {
        head[c] = k_d[static_cast<size_t>(t * k_stride + h * d + c)];
      }
      ref_norm_vec(head, w_d.data(), w_off, kind, kEps, -1.0);
      for (int32_t c = 0; c < d; ++c) {
        want_k[static_cast<size_t>(t * k_width + h * d + c)] = head[c];
      }
    }
  }
  kops_test::compare(got_q, want_q, atol, rtol, name);
  kops_test::compare(got_k, want_k, atol, rtol, name);
}

/// Full-width (OLMo-2 style) RMS norm over the whole q/k row, with optional
/// cross-rank mean-square partials.
template <typename T>
void test_qk_full_width(const char* name, kops::DType dtype, int32_t hq, int32_t hkv, int32_t d,
                        int64_t tokens, bool with_partials, bool double_partials, double atol,
                        double rtol) {
  const int64_t width = static_cast<int64_t>(hq + 2 * hkv) * d;
  const int64_t q_width = static_cast<int64_t>(hq) * d;
  const int64_t k_width = static_cast<int64_t>(hkv) * d;
  std::vector<float> body(static_cast<size_t>(tokens * width));
  fill(body, 71u);
  std::vector<float> qw_f(q_width), kw_f(k_width);
  fill(qw_f, 72u);
  fill(kw_f, 73u);
  const std::vector<T> qkv = cast_from_float<T>(body);
  const std::vector<T> qw = cast_from_float<T>(qw_f);
  const std::vector<T> kw = cast_from_float<T>(kw_f);

  std::vector<float> partials_f(static_cast<size_t>(tokens * 2), 0.0f);
  const auto qkv_d = as_double(qkv);
  for (int64_t t = 0; t < tokens; ++t) {
    double q_ss = 0.0, k_ss = 0.0;
    for (int64_t c = 0; c < q_width; ++c) q_ss += qkv_d[static_cast<size_t>(t * width + c)] *
                                                  qkv_d[static_cast<size_t>(t * width + c)];
    for (int64_t c = 0; c < k_width; ++c) {
      const int64_t i = t * width + q_width + c;
      k_ss += qkv_d[static_cast<size_t>(i)] * qkv_d[static_cast<size_t>(i)];
    }
    partials_f[static_cast<size_t>(t * 2 + 0)] =
        static_cast<float>(q_ss / static_cast<double>(q_width) * (double_partials ? 2.0 : 1.0));
    partials_f[static_cast<size_t>(t * 2 + 1)] =
        static_cast<float>(k_ss / static_cast<double>(k_width) * (double_partials ? 2.0 : 1.0));
  }

  DevBuf<T> d_qkv(qkv.size());
  DevBuf<T> d_qw(qw.size());
  DevBuf<T> d_kw(kw.size());
  DevBuf<float> d_part(std::max<size_t>(partials_f.size(), 1));
  if (!d_qkv.ok() || !d_qw.ok() || !d_kw.ok() || !d_part.ok()) return;
  kops_test::copy_in(d_qkv.p, qkv);
  kops_test::copy_in(d_qw.p, qw);
  kops_test::copy_in(d_kw.p, kw);
  if (with_partials) kops_test::copy_in(d_part.p, partials_f);

  const kops::Tensor t_qkv = tensor2d_strided(d_qkv.p, dtype, tokens, width, width);
  const kops::Tensor t_qw = tensor1d(d_qw.p, dtype, q_width);
  const kops::Tensor t_kw = tensor1d(d_kw.p, dtype, k_width);
  const kops::Tensor t_part =
      with_partials ? tensor2d_strided(d_part.p, kops::DType::kFloat32, tokens, 2, 2)
                    : kops::Tensor{};
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw, t_kw, hq, hkv, d,
                                     kops::NormType::kRMSNorm, true, t_part, kEps, 0.0f, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(d_qkv.p, qkv.size());
  const auto qw_d = as_double(qw);
  const auto kw_d = as_double(kw);
  std::vector<double> want = qkv_d;  // v (and every un-normalized element) stays as-is
  for (int64_t t = 0; t < tokens; ++t) {
    for (int kind = 0; kind < 2; ++kind) {
      const int64_t w = (kind == 0) ? q_width : k_width;
      const int64_t off = (kind == 0) ? 0 : q_width;
      std::vector<double> row(w);
      for (int64_t c = 0; c < w; ++c) {
        row[c] = qkv_d[static_cast<size_t>(t * width + off + c)];
      }
      const double mean_sq = with_partials
                                 ? static_cast<double>(partials_f[static_cast<size_t>(t * 2 + kind)])
                                 : -1.0;
      ref_norm_vec(row, (kind == 0) ? qw_d.data() : kw_d.data(), 0.0, 0, kEps, mean_sq);
      for (int64_t c = 0; c < w; ++c) {
        want[static_cast<size_t>(t * width + off + c)] = row[c];
      }
    }
  }
  kops_test::compare(got, want, atol, rtol, name);

  // V is never touched.
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t c = q_width + k_width; c < width; ++c) {
      KOPS_TEST_CHECK(got[static_cast<size_t>(t * width + c)] ==
                      qkv[static_cast<size_t>(t * width + c)]);
    }
  }
}

/// 3-D packed qkv view [tokens, heads, head_dim].
template <typename T>
void test_qk_3d(const char* name, kops::DType dtype, int32_t hq, int32_t hkv, int32_t d,
                int64_t tokens, double atol, double rtol) {
  const int32_t heads = hq + 2 * hkv;
  const int64_t width = static_cast<int64_t>(heads) * d;
  std::vector<float> body(static_cast<size_t>(tokens * width));
  fill(body, 81u);
  std::vector<float> qw_f(d), kw_f(d);
  fill(qw_f, 82u);
  fill(kw_f, 83u);
  const std::vector<T> qkv = cast_from_float<T>(body);
  const std::vector<T> qw = cast_from_float<T>(qw_f);
  const std::vector<T> kw = cast_from_float<T>(kw_f);

  DevBuf<T> d_qkv(qkv.size());
  DevBuf<T> d_qw(qw.size());
  DevBuf<T> d_kw(kw.size());
  if (!d_qkv.ok() || !d_qw.ok() || !d_kw.ok()) return;
  kops_test::copy_in(d_qkv.p, qkv);
  kops_test::copy_in(d_qw.p, qw);
  kops_test::copy_in(d_kw.p, kw);

  kops::Tensor t_qkv{};
  t_qkv.data = d_qkv.p;
  t_qkv.dtype = dtype;
  t_qkv.ndim = 3;
  t_qkv.device = 0;
  t_qkv.shape[0] = tokens;
  t_qkv.shape[1] = heads;
  t_qkv.shape[2] = d;
  t_qkv.stride[0] = width;
  t_qkv.stride[1] = d;
  t_qkv.stride[2] = 1;
  const kops::Tensor t_qw = tensor1d(d_qw.p, dtype, d);
  const kops::Tensor t_kw = tensor1d(d_kw.p, dtype, d);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw, t_kw, hq, hkv, d,
                                     kops::NormType::kRMSNorm, false, kops::Tensor{}, kEps, 0.0f,
                                     ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(d_qkv.p, qkv.size());
  const auto qkv_d = as_double(qkv);
  const auto qw_d = as_double(qw);
  const auto kw_d = as_double(kw);
  std::vector<double> want(qkv_d.size());
  for (int64_t t = 0; t < tokens; ++t) {
    for (int32_t h = 0; h < heads; ++h) {
      std::vector<double> head(d);
      for (int32_t c = 0; c < d; ++c) {
        head[c] = qkv_d[static_cast<size_t>((t * heads + h) * d + c)];
      }
      if (h < hq) {
        ref_norm_vec(head, qw_d.data(), 0.0, 0, kEps, -1.0);
      } else if (h < hq + hkv) {
        ref_norm_vec(head, kw_d.data(), 0.0, 0, kEps, -1.0);
      }
      for (int32_t c = 0; c < d; ++c) {
        want[static_cast<size_t>((t * heads + h) * d + c)] = head[c];
      }
    }
  }
  kops_test::compare(got, want, atol, rtol, name);
}

// -----------------------------------------------------------------------------
// l2_norm
// -----------------------------------------------------------------------------

template <typename T>
void test_l2_norm(const char* name, kops::DType dtype, int64_t rows, int64_t cols, int64_t pad,
                  float scale, bool inplace, double atol, double rtol) {
  const int64_t stride = cols + pad;
  std::vector<float> body(static_cast<size_t>(rows * stride), 7.0f);
  std::vector<float> src(static_cast<size_t>(rows * cols));
  fill(src, 91u);
  for (int64_t r = 0; r < rows; ++r) {
    std::copy(src.begin() + r * cols, src.begin() + (r + 1) * cols, body.begin() + r * stride);
  }
  const std::vector<T> in = cast_from_float<T>(body);

  DevBuf<T> d_in(in.size());
  DevBuf<T> d_out(inplace ? 1 : in.size());
  if (!d_in.ok() || !d_out.ok()) return;
  kops_test::copy_in(d_in.p, in);

  kops::Tensor t_in = tensor2d_strided(d_in.p, dtype, rows, cols, stride);
  kops::Tensor t_out = inplace ? t_in : tensor2d_strided(d_out.p, dtype, rows, cols, stride);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::l2_norm(t_out, t_in, scale, kEps, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const T* src_ptr = inplace ? d_in.p : d_out.p;
  const auto got = compact(kops_test::copy_out(src_ptr, in.size()), rows, cols, stride);
  const auto in_d = as_double(in);
  std::vector<double> want;
  ref_l2(in_d, rows, cols, stride, scale, kEps, want);
  kops_test::compare(got, want, atol, rtol, name);
}

// -----------------------------------------------------------------------------
// gated_rms_norm
// -----------------------------------------------------------------------------

template <typename T>
void test_gated(const char* name, kops::DType dtype, int64_t rows, int64_t hidden, int64_t group,
                kops::ActFn act, bool before, bool with_weight, bool per_group_weight,
                float w_off, int64_t pad, bool inplace, double atol, double rtol) {
  const int64_t stride = hidden + pad;
  std::vector<float> x_f(static_cast<size_t>(rows * stride), 7.0f);
  std::vector<float> z_f(static_cast<size_t>(rows * stride), 7.0f);
  std::vector<float> x_body(static_cast<size_t>(rows * hidden));
  std::vector<float> z_body(static_cast<size_t>(rows * hidden));
  fill(x_body, 101u);
  fill(z_body, 102u);
  for (int64_t r = 0; r < rows; ++r) {
    std::copy(x_body.begin() + r * hidden, x_body.begin() + (r + 1) * hidden,
              x_f.begin() + r * stride);
    std::copy(z_body.begin() + r * hidden, z_body.begin() + (r + 1) * hidden,
              z_f.begin() + r * stride);
  }
  const int64_t w_len = with_weight ? (per_group_weight ? hidden : group) : 0;
  std::vector<float> w_f(static_cast<size_t>(std::max<int64_t>(w_len, 1)));
  fill(w_f, 103u);
  const std::vector<T> x = cast_from_float<T>(x_f);
  const std::vector<T> z = cast_from_float<T>(z_f);
  const std::vector<T> w = cast_from_float<T>(with_weight ? w_f : std::vector<float>{});

  DevBuf<T> d_x(x.size());
  DevBuf<T> d_z(z.size());
  DevBuf<T> d_out(inplace ? 1 : x.size());
  DevBuf<T> d_w(std::max<size_t>(w.size(), 1));
  if (!d_x.ok() || !d_z.ok() || !d_out.ok() || !d_w.ok()) return;
  kops_test::copy_in(d_x.p, x);
  kops_test::copy_in(d_z.p, z);
  if (!w.empty()) kops_test::copy_in(d_w.p, w);

  kops::Tensor t_x = tensor2d_strided(d_x.p, dtype, rows, hidden, stride);
  kops::Tensor t_z = tensor2d_strided(d_z.p, dtype, rows, hidden, stride);
  kops::Tensor t_out = inplace ? t_x : tensor2d_strided(d_out.p, dtype, rows, hidden, stride);
  kops::Tensor t_w = with_weight ? tensor1d(d_w.p, dtype, w_len) : kops::Tensor{};
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::gated_rms_norm(t_out, t_x, t_z, t_w, static_cast<int32_t>(group), act,
                                            before, kEps, w_off, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const T* src_ptr = inplace ? d_x.p : d_out.p;
  const auto got = compact(kops_test::copy_out(src_ptr, x.size()), rows, hidden, stride);
  const auto x_d = as_double(x);
  const auto z_d = as_double(z);
  const auto w_d = as_double(w);
  const int act_i = (act == kops::ActFn::kIdentity) ? 0 : (act == kops::ActFn::kSigmoid) ? 1 : 2;
  std::vector<double> want;
  ref_gated(x_d, z_d, rows, hidden, stride, stride, with_weight ? w_d.data() : nullptr,
            per_group_weight ? group : 0, group, act_i, before, w_off, kEps, want);
  kops_test::compare(got, want, atol, rtol, name);
}

// -----------------------------------------------------------------------------
// Contract errors
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Layout / staging coverage: padded row strides, the direct (non-staged) path
// for rows larger than the shared-memory stage, and the bit-identity of the
// in-place and out-of-place fused entries that share one device body.
// -----------------------------------------------------------------------------

template <typename T, typename R>
void test_fused_layouts(const char* name, kops::DType dtype, kops::DType rdtype, int64_t rows,
                        int64_t cols, int64_t pad, float input_scale, bool with_w, double atol,
                        double rtol) {
  const int64_t stride = cols + pad;
  std::vector<float> x_body(static_cast<size_t>(rows * stride), 7.0f);
  std::vector<float> r_body(static_cast<size_t>(rows * stride), 7.0f);
  std::vector<float> x_f(static_cast<size_t>(rows * cols)), r_f(x_f.size()), w_f(cols);
  fill(x_f, 55u);
  fill(r_f, 56u);
  fill(w_f, 57u);
  for (int64_t r = 0; r < rows; ++r) {
    std::copy(x_f.begin() + r * cols, x_f.begin() + (r + 1) * cols, x_body.begin() + r * stride);
    std::copy(r_f.begin() + r * cols, r_f.begin() + (r + 1) * cols, r_body.begin() + r * stride);
  }
  const std::vector<T> x = cast_from_float<T>(x_body);
  const std::vector<R> res = cast_from_float<R>(r_body);
  const std::vector<T> w = cast_from_float<T>(with_w ? w_f : std::vector<float>{});
  const int64_t n = static_cast<int64_t>(x.size());

  DevBuf<T> d_x(n), d_x2(n), d_out(rows * cols);
  DevBuf<R> d_res(n), d_res2(n), d_rout(n);
  DevBuf<T> d_w(std::max<size_t>(w.size(), 1));
  if (!d_x.ok() || !d_x2.ok() || !d_out.ok() || !d_res.ok() || !d_res2.ok() || !d_rout.ok() ||
      !d_w.ok()) {
    return;
  }
  kops_test::copy_in(d_x.p, x);
  kops_test::copy_in(d_x2.p, x);
  kops_test::copy_in(d_res.p, res);
  kops_test::copy_in(d_res2.p, res);
  if (!w.empty()) kops_test::copy_in(d_w.p, w);

  const kops::Tensor t_x = tensor2d_strided(d_x.p, dtype, rows, cols, stride);
  const kops::Tensor t_res = tensor2d_strided(d_res.p, rdtype, rows, cols, stride);
  const kops::Tensor t_x2 = tensor2d_strided(d_x2.p, dtype, rows, cols, stride);
  const kops::Tensor t_res2 = tensor2d_strided(d_res2.p, rdtype, rows, cols, stride);
  // out must be contiguous; the residual sides may be padded.
  const kops::Tensor t_out = tensor2d_strided(d_out.p, dtype, rows, cols, cols);
  const kops::Tensor t_rout = tensor2d_strided(d_rout.p, rdtype, rows, cols, stride);
  const kops::Tensor t_w = with_w ? tensor1d(d_w.p, dtype, cols) : kops::Tensor{};
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  KOPS_EXPECT_OK(kops::norm::fused_add_rms_norm(t_x, t_res, t_w, kEps, 0.0f, input_scale, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  KOPS_EXPECT_OK(kops::norm::fused_add_rms_norm_out(t_out, t_rout, t_x2, t_res2, t_w, kEps, 0.0f,
                                                    input_scale, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto in_x = compact(kops_test::copy_out(d_x.p, static_cast<size_t>(n)), rows, cols, stride);
  const auto in_res = compact(kops_test::copy_out(d_res.p, static_cast<size_t>(n)), rows, cols,
                              stride);
  const auto out_x = kops_test::copy_out(d_out.p, static_cast<size_t>(rows * cols));
  const auto out_res =
      compact(kops_test::copy_out(d_rout.p, static_cast<size_t>(n)), rows, cols, stride);
  // The two entries share one device body: they must agree bit for bit.
  kops_test::compare(out_x, as_double(in_x), 0.0, 0.0, name);
  kops_test::compare(out_res, as_double(in_res), 0.0, 0.0, name);

  // And both must match the fp64 oracle.
  const auto x_d = as_double(x);
  const auto res_d = as_double(res);
  const auto w_d = as_double(w);
  std::vector<double> res_want, norm_want;
  ref_fused(x_d, res_d, rows, cols, stride, input_scale, false, with_w ? w_d.data() : nullptr,
            nullptr, 0.0, kEps, res_want, norm_want);
  kops_test::compare(in_res, res_want, atol, rtol, name);
  kops_test::compare(in_x, norm_want, atol, rtol, name);
}

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  const int64_t rows = 4, cols = 256;
  std::vector<float> data(static_cast<size_t>(rows * cols), 1.0f);
  float* d = kops_test::device_alloc<float>(data.size());
  float* d_z = kops_test::device_alloc<float>(data.size());
  if (d == nullptr || d_z == nullptr) {
    kops_test::device_free(d);
    kops_test::device_free(d_z);
    return;
  }
  kops_test::copy_in(d, data);
  kops_test::copy_in(d_z, data);
  const kops::Tensor t = tensor2d_strided(d, kops::DType::kFloat32, rows, cols, cols);
  const kops::Tensor t_z = tensor2d_strided(d_z, kops::DType::kFloat32, rows, cols, cols);
  const kops::Tensor t_out = tensor2d_strided(d, kops::DType::kFloat32, rows, cols, cols);
  const kops::Tensor t_mis = tensor2d_strided(reinterpret_cast<char*>(d) + 1,
                                              kops::DType::kFloat32, rows, cols, cols);
  const kops::Tensor t_short = tensor2d_strided(d, kops::DType::kFloat32, rows, 128, 128);
  const kops::Tensor t16 = tensor2d_strided(d, kops::DType::kFloat16, rows, cols, cols);

  // layer_norm: dtype, shape, weight width, negative epsilon, alignment.
  KOPS_EXPECT_STATUS(kops::norm::layer_norm(t16, t, kops::Tensor{}, kops::Tensor{}, kEps, 0.0f, ctx),
                     kops::Status::kUnsupportedDtype, "layer_norm dtype");
  KOPS_EXPECT_STATUS(kops::norm::layer_norm(t_out, t16, kops::Tensor{}, kops::Tensor{}, kEps, 0.0f,
                                            ctx),
                     kops::Status::kUnsupportedDtype, "layer_norm out dtype");
  KOPS_EXPECT_STATUS(kops::norm::layer_norm(t_out, t_short, kops::Tensor{}, kops::Tensor{}, kEps,
                                            0.0f, ctx),
                     kops::Status::kInvalidArgument, "layer_norm shape");
  KOPS_EXPECT_STATUS(kops::norm::layer_norm(t_out, t, tensor1d(d, kops::DType::kFloat32, 128),
                                            kops::Tensor{}, kEps, 0.0f, ctx),
                     kops::Status::kUnsupportedShape, "layer_norm weight width");
  KOPS_EXPECT_STATUS(kops::norm::layer_norm(t_out, t, kops::Tensor{}, kops::Tensor{}, -1.0f, 0.0f,
                                            ctx),
                     kops::Status::kInvalidArgument, "layer_norm epsilon");
  KOPS_EXPECT_STATUS(kops::norm::layer_norm(t_out, t_mis, kops::Tensor{}, kops::Tensor{}, kEps,
                                            0.0f, ctx),
                     kops::Status::kMisaligned, "layer_norm alignment");

  // l2_norm: shape mismatch, unsupported dtype.
  KOPS_EXPECT_STATUS(kops::norm::l2_norm(t_out, t_short, 1.0f, kEps, ctx),
                     kops::Status::kInvalidArgument, "l2_norm shape");
  KOPS_EXPECT_STATUS(kops::norm::l2_norm(t16, t, 1.0f, kEps, ctx),
                     kops::Status::kUnsupportedDtype, "l2_norm dtype");

  // gated_rms_norm: group does not divide, unsupported activation, weight length.
  KOPS_EXPECT_STATUS(kops::norm::gated_rms_norm(t_out, t, t_z, kops::Tensor{}, 100,
                                                kops::ActFn::kSiLU, true, kEps, 0.0f, ctx),
                     kops::Status::kUnsupportedShape, "gated group");
  KOPS_EXPECT_STATUS(kops::norm::gated_rms_norm(t_out, t, t_z, kops::Tensor{}, 128,
                                                kops::ActFn::kTanh, true, kEps, 0.0f, ctx),
                     kops::Status::kUnsupportedShape, "gated act");
  KOPS_EXPECT_STATUS(kops::norm::gated_rms_norm(t_out, t, t_z,
                                                tensor1d(d, kops::DType::kFloat32, 64), 128,
                                                kops::ActFn::kSiLU, true, kEps, 0.0f, ctx),
                     kops::Status::kUnsupportedShape, "gated weight length");
  KOPS_EXPECT_STATUS(kops::norm::gated_rms_norm(t_out, t, t16, kops::Tensor{}, 128,
                                                kops::ActFn::kSiLU, true, kEps, 0.0f, ctx),
                     kops::Status::kUnsupportedDtype, "gated z dtype");

  // fused_add_rms_norm_out: stride rules, out row contiguity, overlap, dtype.
  kops::Tensor t_res_bad = tensor2d_strided(d, kops::DType::kFloat32, rows, cols, cols + 8);
  kops::Tensor t_out_off =
      tensor2d_strided(reinterpret_cast<char*>(d) + 8 * sizeof(float), kops::DType::kFloat32, rows,
                       cols, cols);
  KOPS_EXPECT_STATUS(kops::norm::fused_add_rms_norm_out(t_out, t_out, t, t_res_bad, kops::Tensor{},
                                                        kEps, 0.0f, 1.0f, ctx),
                     kops::Status::kUnsupportedLayout, "fused_out residual stride");
  KOPS_EXPECT_STATUS(kops::norm::fused_add_rms_norm_out(t_res_bad, t_res_bad, t, t, kops::Tensor{},
                                                        kEps, 0.0f, 1.0f, ctx),
                     kops::Status::kUnsupportedLayout, "fused_out out stride");
  KOPS_EXPECT_STATUS(kops::norm::fused_add_rms_norm_out(t_out_off, t, t, t, kops::Tensor{}, kEps,
                                                        0.0f, 1.0f, ctx),
                     kops::Status::kUnsupportedLayout, "fused_out overlap");
  KOPS_EXPECT_STATUS(kops::norm::fused_add_rms_norm_out(t16, t, t, t, kops::Tensor{}, kEps, 0.0f,
                                                        1.0f, ctx),
                     kops::Status::kUnsupportedDtype, "fused_out out dtype");

  // fused_add_layer_norm: residual stride must match the input stride, and the
  // two in-place operands must be different buffers.
  KOPS_EXPECT_STATUS(
      kops::norm::fused_add_layer_norm(t, t_res_bad, kops::Tensor{}, kops::Tensor{}, kEps, 0.0f,
                                       1.0f, ctx),
      kops::Status::kUnsupportedLayout, "fused_add_layer_norm stride");
  KOPS_EXPECT_STATUS(kops::norm::fused_add_layer_norm(t, t, kops::Tensor{}, kops::Tensor{}, kEps,
                                                      0.0f, 1.0f, ctx),
                     kops::Status::kUnsupportedLayout, "fused_add_layer_norm overlap");

  // qk_norm: head_dim / norm / width / partials rules and weight shapes.
  const int32_t hq = 4, hkv = 2, hd = 64;
  const int64_t qkv_width = static_cast<int64_t>(hq + 2 * hkv) * hd;
  const int64_t q_width = static_cast<int64_t>(hq) * hd;
  const int64_t k_width = static_cast<int64_t>(hkv) * hd;
  std::vector<float> qkv_host(static_cast<size_t>(rows * qkv_width), 1.0f);
  float* d_qkv = kops_test::device_alloc<float>(qkv_host.size());
  float* d_w = kops_test::device_alloc<float>(static_cast<size_t>(qkv_width));
  float* d_part = kops_test::device_alloc<float>(static_cast<size_t>(rows * 2));
  if (d_qkv == nullptr || d_w == nullptr || d_part == nullptr) {
    kops_test::device_free(d_qkv);
    kops_test::device_free(d_w);
    kops_test::device_free(d_part);
    kops_test::device_free(d);
    kops_test::device_free(d_z);
    return;
  }
  kops_test::copy_in(d_qkv, qkv_host);
  kops_test::copy_in(d_w, std::vector<float>(static_cast<size_t>(qkv_width), 1.0f));
  kops_test::copy_in(d_part, std::vector<float>(static_cast<size_t>(rows * 2), 1.0f));

  const kops::Tensor t_qkv =
      tensor2d_strided(d_qkv, kops::DType::kFloat32, rows, qkv_width, qkv_width);
  const kops::Tensor t_qw = tensor1d(d_w, kops::DType::kFloat32, hd);
  const kops::Tensor t_qw_wide = tensor1d(d_w, kops::DType::kFloat32, qkv_width);
  const kops::Tensor t_qw_full = tensor1d(d_w, kops::DType::kFloat32, q_width);
  const kops::Tensor t_kw_full = tensor1d(d_w, kops::DType::kFloat32, k_width);
  const kops::Tensor t_part = tensor2d_strided(d_part, kops::DType::kFloat32, rows, 2, 2);
  const kops::Tensor t_part_bad = tensor2d_strided(d_part, kops::DType::kFloat32, rows, 1, 1);
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw, t_qw, hq, hkv, 96,
                                         kops::NormType::kRMSNorm, false, kops::Tensor{}, kEps,
                                         0.0f, ctx),
                     kops::Status::kUnsupportedShape, "qk head_dim");
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw, t_qw, hq, hkv, hd,
                                         kops::NormType::kNone, false, kops::Tensor{}, kEps, 0.0f,
                                         ctx),
                     kops::Status::kUnsupportedShape, "qk norm type");
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw_wide, t_qw_wide, hq, hkv, hd,
                                         kops::NormType::kRMSNorm, false, kops::Tensor{}, kEps,
                                         0.0f, ctx),
                     kops::Status::kUnsupportedShape, "qk per-head weight width");
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw, t_qw, hq, hkv, hd,
                                         kops::NormType::kRMSNorm, false, t_part, kEps, 0.0f, ctx),
                     kops::Status::kUnsupportedLayout, "qk partials without full width");
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw_full, t_kw_full, hq, hkv, hd,
                                         kops::NormType::kLayerNorm, true, t_part, kEps, 0.0f, ctx),
                     kops::Status::kUnsupportedLayout, "qk partials with LayerNorm");
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw, t_kw_full, hq, hkv, hd,
                                         kops::NormType::kRMSNorm, true, kops::Tensor{}, kEps, 0.0f,
                                         ctx),
                     kops::Status::kUnsupportedShape, "qk full-width weight width");
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_qkv, kops::Tensor{}, t_qw_full, t_kw_full, hq, hkv, hd,
                                         kops::NormType::kRMSNorm, true, t_part_bad, kEps, 0.0f,
                                         ctx),
                     kops::Status::kUnsupportedLayout, "qk partials shape");
  // Width mismatch: the packed width must be (hq + 2*hkv) * head_dim.
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_out, kops::Tensor{}, t_qw, t_qw, hq, hkv, hd,
                                         kops::NormType::kRMSNorm, false, kops::Tensor{}, kEps,
                                         0.0f, ctx),
                     kops::Status::kInvalidArgument, "qk qkv width");
  // A separate k tensor with a different row count is rejected.
  const kops::Tensor t_k_bad = tensor2d_strided(d_qkv, kops::DType::kFloat32, rows + 1, k_width,
                                                k_width);
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(t_qkv, t_k_bad, t_qw, t_qw, hq, hkv, hd,
                                         kops::NormType::kRMSNorm, false, kops::Tensor{}, kEps,
                                         0.0f, ctx),
                     kops::Status::kInvalidArgument, "qk k rows");

  // Zero tokens is a no-op success on every entry (never a zero-sized launch).
  const kops::Tensor t0 = tensor2d_strided(d, kops::DType::kFloat32, 0, cols, cols);
  const kops::Tensor tz0 = tensor2d_strided(d_z, kops::DType::kFloat32, 0, cols, cols);
  const kops::Tensor tq0 = tensor2d_strided(d_qkv, kops::DType::kFloat32, 0, qkv_width, qkv_width);
  KOPS_EXPECT_STATUS(kops::norm::layer_norm(t0, t0, kops::Tensor{}, kops::Tensor{}, kEps, 0.0f,
                                            ctx),
                     kops::Status::kSuccess, "layer_norm empty");
  KOPS_EXPECT_STATUS(kops::norm::l2_norm(t0, t0, 1.0f, kEps, ctx), kops::Status::kSuccess,
                     "l2_norm empty");
  KOPS_EXPECT_STATUS(kops::norm::gated_rms_norm(t0, t0, tz0, kops::Tensor{}, 128,
                                                kops::ActFn::kSiLU, true, kEps, 0.0f, ctx),
                     kops::Status::kSuccess, "gated empty");
  KOPS_EXPECT_STATUS(kops::norm::fused_add_rms_norm_out(t0, t0, t0, t0, kops::Tensor{}, kEps, 0.0f,
                                                        1.0f, ctx),
                     kops::Status::kSuccess, "fused_out empty");
  KOPS_EXPECT_STATUS(kops::norm::fused_add_layer_norm(t0, tz0, kops::Tensor{}, kops::Tensor{}, kEps,
                                                      0.0f, 1.0f, ctx),
                     kops::Status::kSuccess, "fused_add_layer_norm empty");
  KOPS_EXPECT_STATUS(kops::norm::qk_norm(tq0, kops::Tensor{}, t_qw, t_kw_full, hq, hkv, hd,
                                         kops::NormType::kRMSNorm, false, kops::Tensor{}, kEps,
                                         0.0f, ctx),
                     kops::Status::kSuccess, "qk empty");

  kops_test::device_free(d_qkv);
  kops_test::device_free(d_w);
  kops_test::device_free(d_part);
  kops_test::device_free(d);
  kops_test::device_free(d_z);
}

// -----------------------------------------------------------------------------
// N02: norm_quant (B3 row 5)
// -----------------------------------------------------------------------------

uint8_t nq_e4m3_host(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

float e4m3_value(uint8_t b) {
  return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

void test_norm_quant(kops::LaunchContext& ctx) {
  const int rows = 7, hidden = 512, group = 128;
  const int groups = hidden / group;
  DevBuf<__nv_bfloat16> x(rows * hidden);
  DevBuf<__nv_bfloat16> res(rows * hidden);
  DevBuf<__nv_bfloat16> w(static_cast<size_t>(hidden));
  DevBuf<uint8_t> q(rows * hidden);
  DevBuf<float> scales(rows * groups);
  if (!x.ok() || !res.ok() || !w.ok() || !q.ok() || !scales.ok()) return;
  std::vector<float> xf(rows * hidden), rf(rows * hidden), wf(static_cast<size_t>(hidden));
  fill(xf, 301, -2.0f, 2.0f);
  fill(rf, 302, -1.0f, 1.0f);
  fill(wf, 303, 0.5f, 1.5f);
  auto xh = cast_from_float<__nv_bfloat16>(xf);
  auto rh = cast_from_float<__nv_bfloat16>(rf);
  auto wh = cast_from_float<__nv_bfloat16>(wf);
  kops_test::copy_in(x.p, xh);
  kops_test::copy_in(res.p, rh);
  kops_test::copy_in(w.p, wh);
  const float eps = 1e-6f;

  // RMSNorm + dynamic per-token FP8.
  kops::norm::NormQuantParams p{};
  p.input = x.p;
  p.input_stride = hidden;
  p.weight = w.p;
  p.epsilon = eps;
  p.num_tokens = rows;
  p.hidden_size = hidden;
  p.quant.q = q.p;
  p.quant.q_row_stride = hidden;
  p.quant.scales = scales.p;
  p.quant.scale_row_stride = 1;
  p.quant.q_dtype = kops::DType::kFloat8E4M3;
  p.quant.mode = kops::ActQuantMode::kDynamicPerToken;
  KOPS_EXPECT_OK(kops::norm::norm_quant(p, kops::NormType::kRMSNorm, kops::DType::kBFloat16,
                                        kops::DType::kBFloat16, ctx),
                 "norm_quant per-token");
  const auto got = kops_test::copy_out(q.p, rows * hidden);
  const auto gs = kops_test::copy_out(scales.p, rows);
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int c = 0; c < hidden; ++c) {
      const double v = kops_test::to_float(xh[static_cast<size_t>(r * hidden + c)]);
      ss += v * v;
    }
    const float rstd = static_cast<float>(1.0 / std::sqrt(ss / hidden + eps));
    float amax = 0.0f;
    for (int c = 0; c < hidden; ++c) {
      const float v = kops_test::to_float(xh[static_cast<size_t>(r * hidden + c)]);
      const float y = v * rstd * kops_test::to_float(wh[static_cast<size_t>(c)]);
      amax = std::max(amax, std::fabs(y));
    }
    const float s = std::max(amax / 448.0f, 1e-10f);
    KOPS_TEST_CHECK(std::fabs(gs[static_cast<size_t>(r)] - s) <= 1e-6f * s + 1e-8f);
    for (int c = 0; c < hidden; ++c) {
      const float v = kops_test::to_float(xh[static_cast<size_t>(r * hidden + c)]);
      const float y = v * rstd * kops_test::to_float(wh[static_cast<size_t>(c)]);
      const float qv = e4m3_value(got[static_cast<size_t>(r * hidden + c)]) * s;
      KOPS_TEST_CHECK(std::fabs(qv - y) <= 0.07f * std::fabs(y) + 1e-4f);
    }
  }

  // Residual add + static per-group FP8, col-major TMA scales.
  DevBuf<float> static_val(1);
  kops_test::copy_in(static_val.p, std::vector<float>{0.5f});
  DevBuf<float> tma(groups * ((rows + 3) / 4 * 4));
  if (!tma.ok()) return;
  p.quant.mode = kops::ActQuantMode::kPerGroup;
  p.quant.group_size = group;
  p.quant.scale_layout = kops::ScaleLayout::kColMajorTma;
  p.quant.scales = tma.p;
  p.quant.scale_row_stride = (rows + 3) / 4 * 4;
  p.quant.static_scale = nullptr;
  p.residual = res.p;
  p.residual_in = nullptr;
  p.input_scale = 1.0f;
  KOPS_EXPECT_OK(kops::norm::norm_quant(p, kops::NormType::kRMSNorm, kops::DType::kBFloat16,
                                        kops::DType::kBFloat16, ctx),
                 "norm_quant per-group residual");
  const auto qg = kops_test::copy_out(q.p, rows * hidden);
  const auto tg = kops_test::copy_out(tma.p, groups * ((rows + 3) / 4 * 4));
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int c = 0; c < hidden; ++c) {
      __nv_bfloat16 rb;
      kops_test::from_float(rb, kops_test::to_float(rh[static_cast<size_t>(r * hidden + c)]) +
                                    kops_test::to_float(xh[static_cast<size_t>(r * hidden + c)]));
      const double v = kops_test::to_float(rb);
      ss += v * v;
    }
    const float rstd = static_cast<float>(1.0 / std::sqrt(ss / hidden + eps));
    for (int g = 0; g < groups; ++g) {
      float amax = 0.0f;
      for (int j = 0; j < group; ++j) {
        const int c = g * group + j;
        __nv_bfloat16 rb;
        kops_test::from_float(rb, kops_test::to_float(rh[static_cast<size_t>(r * hidden + c)]) +
                                      kops_test::to_float(xh[static_cast<size_t>(r * hidden + c)]));
        const float y = kops_test::to_float(rb) * rstd *
                        kops_test::to_float(wh[static_cast<size_t>(c)]);
        amax = std::max(amax, std::fabs(y));
      }
      const float s = std::max(amax / 448.0f, 1e-10f);
      KOPS_TEST_CHECK(std::fabs(tg[static_cast<size_t>(g * ((rows + 3) / 4 * 4) + r)] - s) <=
                      1e-6f * s + 1e-8f);
      for (int j = 0; j < group; ++j) {
        const int c = g * group + j;
        __nv_bfloat16 rb;
        kops_test::from_float(rb, kops_test::to_float(rh[static_cast<size_t>(r * hidden + c)]) +
                                      kops_test::to_float(xh[static_cast<size_t>(r * hidden + c)]));
        const float y = kops_test::to_float(rb) * rstd *
                        kops_test::to_float(wh[static_cast<size_t>(c)]);
        const float qv = e4m3_value(qg[static_cast<size_t>(r * hidden + c)]) * s;
        KOPS_TEST_CHECK(std::fabs(qv - y) <= 0.07f * std::fabs(y) + 1e-4f);
      }
    }
  }
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;

  // ---- existing entries, re-checked at the new sizes -----------------------
  test_rms_norm<__nv_bfloat16>("rms_norm bf16 h64", kops::DType::kBFloat16, 5, 64, 8, 1, true, 0.0f,
                               2e-2, 2e-2);
  test_rms_norm<__nv_bfloat16>("rms_norm bf16 h256 w+1", kops::DType::kBFloat16, 5, 256, 8, 1, true,
                               1.0f, 2e-2, 2e-2);
  test_rms_norm<float>("rms_norm fp32 h1024", kops::DType::kFloat32, 7, 1024, 0, 1, true, 0.0f,
                       1e-5, 1e-5);
  test_rms_norm<float>("rms_norm fp32 h4096 tail", kops::DType::kFloat32, 3, 4096, 4, 1, true, 0.0f,
                       1e-5, 1e-5);
  test_rms_norm<__nv_bfloat16>("rms_norm bf16 h16384", kops::DType::kBFloat16, 2, 16384, 8, 1, true,
                               0.0f, 2e-2, 2e-2);
  test_rms_norm<__nv_bfloat16>("rms_norm bf16 h250 tail", kops::DType::kBFloat16, 3, 250, 6, 1, true,
                               0.0f, 2e-2, 2e-2);
  test_rms_norm<float>("rms_norm fp32 h250 tail", kops::DType::kFloat32, 3, 250, 2, 1, true, 0.0f,
                       1e-5, 1e-5);
  test_rms_norm<__nv_bfloat16>("rms_norm bf16 no weight", kops::DType::kBFloat16, 4, 512, 0, 1,
                               false, 0.0f, 2e-2, 2e-2);
  test_rms_norm<__nv_bfloat16>("rms_norm bf16 weight groups", kops::DType::kBFloat16, 6, 256, 8, 3,
                               true, 0.0f, 2e-2, 2e-2);
  test_rms_norm<float>("rms_norm fp32 weight groups", kops::DType::kFloat32, 6, 128, 4, 2, true,
                       1.0f, 1e-5, 1e-5);
  test_rms_norm<__half>("rms_norm fp16 h128", kops::DType::kFloat16, 5, 128, 8, 1, true, 0.0f,
                        1e-3, 1e-3);

  test_fused_add_rms_norm<float, float>("fused_add_rms_norm fp32 h4096", kops::DType::kFloat32,
                                        kops::DType::kFloat32, 4, 4096, 1.0f, true, 0.0f, 1e-5,
                                        1e-5);
  test_fused_add_rms_norm<__nv_bfloat16, __nv_bfloat16>(
      "fused_add_rms_norm bf16 h64", kops::DType::kBFloat16, kops::DType::kBFloat16, 5, 64, 0.5f,
      true, 0.0f, 2e-2, 2e-2);
  test_fused_add_rms_norm<__nv_bfloat16, float>(
      "fused_add_rms_norm bf16/fp32 residual h1024", kops::DType::kBFloat16,
      kops::DType::kFloat32, 5, 1024, 1.0f, true, 1.0f, 2e-2, 2e-2);
  test_fused_add_rms_norm<__half, float>("fused_add_rms_norm fp16/fp32 h256", kops::DType::kFloat16,
                                         kops::DType::kFloat32, 4, 256, 0.25f, false, 0.0f, 1e-3,
                                         1e-3);

  // ---- staging / layout coverage ------------------------------------------
  // Rows larger than the shared-memory stage take the direct path (two reads).
  test_rms_norm<__nv_bfloat16>("rms_norm bf16 h24576 direct", kops::DType::kBFloat16, 2, 24576, 0,
                               1, true, 0.0f, 2e-2, 2e-2);
  test_rms_norm<float>("rms_norm fp32 h12288 direct", kops::DType::kFloat32, 2, 12288, 0, 1, true,
                       1.0f, 1e-5, 1e-5);
  test_fused_layouts<__nv_bfloat16, __nv_bfloat16>(
      "fused_add_rms_norm bf16 padded stride", kops::DType::kBFloat16, kops::DType::kBFloat16, 5,
      1024, 8, 1.0f, true, 2e-2, 2e-2);
  test_fused_layouts<__nv_bfloat16, float>(
      "fused_add_rms_norm bf16/fp32 padded stride", kops::DType::kBFloat16, kops::DType::kFloat32,
      4, 512, 8, 0.5f, true, 2e-2, 2e-2);
  test_fused_layouts<__nv_bfloat16, __nv_bfloat16>(
      "fused_add_rms_norm bf16 h24576 direct", kops::DType::kBFloat16, kops::DType::kBFloat16, 2,
      24576, 0, 1.0f, false, 2e-2, 2e-2);

  // ---- fused_add_rms_norm_out ---------------------------------------------
  test_fused_add_rms_norm_out<float, float>("fused_add_rms_norm_out fp32 h4096",
                                            kops::DType::kFloat32, kops::DType::kFloat32, 4, 4096,
                                            1.0f, true, 0.0f, false, false, 1e-5, 1e-5);
  test_fused_add_rms_norm_out<float, float>("fused_add_rms_norm_out fp32 alias residual",
                                            kops::DType::kFloat32, kops::DType::kFloat32, 4, 1024,
                                            0.75f, true, 1.0f, true, false, 1e-5, 1e-5);
  test_fused_add_rms_norm_out<float, float>("fused_add_rms_norm_out fp32 alias normed",
                                            kops::DType::kFloat32, kops::DType::kFloat32, 4, 1024,
                                            1.0f, false, 0.0f, false, true, 1e-5, 1e-5);
  test_fused_add_rms_norm_out<__nv_bfloat16, __nv_bfloat16>(
      "fused_add_rms_norm_out bf16 h256", kops::DType::kBFloat16, kops::DType::kBFloat16, 6, 256,
      0.5f, true, 0.0f, false, false, 2e-2, 2e-2);
  test_fused_add_rms_norm_out<__nv_bfloat16, float>(
      "fused_add_rms_norm_out bf16/fp32 residual alias", kops::DType::kBFloat16,
      kops::DType::kFloat32, 5, 1024, 1.0f, true, 0.0f, true, true, 2e-2, 2e-2);

  // ---- layer_norm ----------------------------------------------------------
  test_layer_norm<float>("layer_norm fp32 h64 w+b", kops::DType::kFloat32, 5, 64, 0, true, true,
                         0.0f, false, 1e-5, 1e-5);
  test_layer_norm<float>("layer_norm fp32 h1024 no bias", kops::DType::kFloat32, 5, 1024, 0, true,
                         false, 0.0f, false, 1e-5, 1e-5);
  test_layer_norm<float>("layer_norm fp32 h4096 no weight", kops::DType::kFloat32, 3, 4096, 4,
                         false, false, 0.0f, false, 1e-5, 1e-5);
  test_layer_norm<float>("layer_norm fp32 h16384", kops::DType::kFloat32, 2, 16384, 0, true, true,
                         0.0f, false, 1e-5, 1e-5);
  test_layer_norm<float>("layer_norm fp32 h256 w+1 inplace", kops::DType::kFloat32, 4, 256, 0,
                         true, false, 1.0f, true, 1e-5, 1e-5);
  test_layer_norm<__nv_bfloat16>("layer_norm bf16 h128 w+b", kops::DType::kBFloat16, 5, 128, 8,
                                 true, true, 0.0f, false, 2e-2, 2e-2);
  test_layer_norm<__nv_bfloat16>("layer_norm bf16 h250 tail", kops::DType::kBFloat16, 3, 250, 6,
                                 true, true, 0.0f, false, 2e-2, 2e-2);
  test_layer_norm<__nv_bfloat16>("layer_norm bf16 h4096 w+1", kops::DType::kBFloat16, 2, 4096, 8,
                                 true, false, 1.0f, false, 2e-2, 2e-2);
  test_layer_norm<__nv_bfloat16>("layer_norm bf16 h16384", kops::DType::kBFloat16, 2, 16384, 0,
                                 true, true, 0.0f, false, 2e-2, 2e-2);
  test_layer_norm<float>("layer_norm fp32 h250 tail", kops::DType::kFloat32, 3, 250, 2, true, true,
                         0.0f, false, 1e-5, 1e-5);
  test_layer_norm<__half>("layer_norm fp16 h256", kops::DType::kFloat16, 4, 256, 8, true, true,
                          0.0f, false, 1e-3, 1e-3);

  // ---- fused_add_layer_norm ------------------------------------------------
  test_fused_add_layer_norm<float, float>("fused_add_layer_norm fp32 h4096",
                                          kops::DType::kFloat32, kops::DType::kFloat32, 4, 4096,
                                          1.0f, true, true, 0.0f, 1e-5, 1e-5);
  test_fused_add_layer_norm<float, float>("fused_add_layer_norm fp32 h256 scale",
                                          kops::DType::kFloat32, kops::DType::kFloat32, 5, 256,
                                          0.5f, true, false, 1.0f, 1e-5, 1e-5);
  test_fused_add_layer_norm<__nv_bfloat16, __nv_bfloat16>(
      "fused_add_layer_norm bf16 h128", kops::DType::kBFloat16, kops::DType::kBFloat16, 5, 128,
      1.0f, true, true, 0.0f, 2e-2, 2e-2);
  test_fused_add_layer_norm<__nv_bfloat16, float>(
      "fused_add_layer_norm bf16/fp32 residual h1024", kops::DType::kBFloat16,
      kops::DType::kFloat32, 4, 1024, 0.75f, false, true, 0.0f, 2e-2, 2e-2);
  test_fused_add_layer_norm<__nv_bfloat16, __nv_bfloat16>(
      "fused_add_layer_norm bf16 h16384", kops::DType::kBFloat16, kops::DType::kBFloat16, 2, 16384,
      1.0f, true, false, 0.0f, 2e-2, 2e-2);
  test_fused_add_layer_norm<__half, __half>("fused_add_layer_norm fp16 h64",
                                            kops::DType::kFloat16, kops::DType::kFloat16, 6, 64,
                                            1.0f, true, true, 0.0f, 1e-3, 1e-3);

  // ---- qk_norm -------------------------------------------------------------
  test_qk_packed<__nv_bfloat16>("qk per-head rms bf16 d64", kops::DType::kBFloat16, 8, 2, 64, 5, 8,
                                true, false, kops::NormType::kRMSNorm, 0.0f, 2e-2, 2e-2);
  test_qk_packed<__nv_bfloat16>("qk per-head rms bf16 d128 per-head w",
                                kops::DType::kBFloat16, 8, 2, 128, 5, 0, true, true,
                                kops::NormType::kRMSNorm, 0.0f, 2e-2, 2e-2);
  test_qk_packed<__nv_bfloat16>("qk per-head rms bf16 d256 w+1", kops::DType::kBFloat16, 4, 1, 256,
                                3, 0, true, false, kops::NormType::kRMSNorm, 1.0f, 2e-2, 2e-2);
  test_qk_packed<float>("qk per-head rms fp32 d64", kops::DType::kFloat32, 4, 2, 64, 5, 0, true, true,
                        kops::NormType::kRMSNorm, 0.0f, 1e-5, 1e-5);
  test_qk_packed<float>("qk per-head layer fp32 d128", kops::DType::kFloat32, 4, 2, 128, 5, 0, true, true,
                        kops::NormType::kLayerNorm, 0.0f, 1e-5, 1e-5);
  test_qk_packed<__nv_bfloat16>("qk per-head l2 bf16 d64", kops::DType::kBFloat16, 8, 2, 64, 5, 0,
                                true, false, kops::NormType::kL2Norm, 0.0f, 2e-2, 2e-2);
  test_qk_packed<__half>("qk per-head rms fp16 d128", kops::DType::kFloat16, 4, 2, 128, 4, 0, true, false,
                         kops::NormType::kRMSNorm, 0.0f, 1e-3, 1e-3);

  test_qk_packed<__nv_bfloat16>("qk per-head l2 bf16 d128 no weight", kops::DType::kBFloat16, 4, 2,
                                128, 5, 0, false, false, kops::NormType::kL2Norm, 0.0f, 2e-2,
                                2e-2);
  test_qk_packed<float>("qk per-head rms fp32 d256 no weight", kops::DType::kFloat32, 4, 2, 256, 4,
                        0, false, false, kops::NormType::kRMSNorm, 0.0f, 1e-5, 1e-5);

  test_qk_split<__nv_bfloat16>("qk split rms bf16 d128", kops::DType::kBFloat16, 8, 2, 128, 5, 8,
                               16, kops::NormType::kRMSNorm, 0.0f, 2e-2, 2e-2);
  test_qk_split<float>("qk split rms fp32 d64 padded", kops::DType::kFloat32, 4, 2, 64, 5, 4, 8,
                       kops::NormType::kRMSNorm, 1.0f, 1e-5, 1e-5);
  test_qk_split<__nv_bfloat16>("qk split layer bf16 d256", kops::DType::kBFloat16, 4, 1, 256, 3, 0,
                               0, kops::NormType::kLayerNorm, 0.0f, 2e-2, 2e-2);

  test_qk_full_width<__nv_bfloat16>("qk full-width rms bf16 d64", kops::DType::kBFloat16, 8, 2, 64,
                                    5, false, false, 2e-2, 2e-2);
  test_qk_full_width<float>("qk full-width rms fp32 d128", kops::DType::kFloat32, 4, 2, 128, 4,
                            false, false, 1e-5, 1e-5);
  test_qk_full_width<__nv_bfloat16>("qk full-width rms bf16 partials", kops::DType::kBFloat16, 8, 2,
                                    64, 5, true, false, 2e-2, 2e-2);
  test_qk_full_width<__nv_bfloat16>("qk full-width rms bf16 partials x2", kops::DType::kBFloat16, 4,
                                    1, 128, 4, true, true, 2e-2, 2e-2);

  test_qk_3d<__nv_bfloat16>("qk 3-D packed rms bf16 d64", kops::DType::kBFloat16, 4, 2, 64, 5,
                            2e-2, 2e-2);
  test_qk_3d<float>("qk 3-D packed rms fp32 d128", kops::DType::kFloat32, 4, 2, 128, 4, 1e-5,
                    1e-5);

  // ---- l2_norm -------------------------------------------------------------
  test_l2_norm<float>("l2_norm fp32 h64", kops::DType::kFloat32, 5, 64, 0, 1.0f, false, 1e-5,
                      1e-5);
  test_l2_norm<float>("l2_norm fp32 h1024 scale", kops::DType::kFloat32, 5, 1024, 4, 0.125f, false,
                      1e-5, 1e-5);
  test_l2_norm<float>("l2_norm fp32 h4096 inplace", kops::DType::kFloat32, 3, 4096, 0, 1.0f, true,
                      1e-5, 1e-5);
  test_l2_norm<__nv_bfloat16>("l2_norm bf16 h128", kops::DType::kBFloat16, 5, 128, 8, 1.0f, false,
                              2e-2, 2e-2);
  test_l2_norm<__nv_bfloat16>("l2_norm bf16 h16384", kops::DType::kBFloat16, 2, 16384, 8, 1.0f,
                              false, 2e-2, 2e-2);
  test_l2_norm<__nv_bfloat16>("l2_norm bf16 h250 tail", kops::DType::kBFloat16, 3, 250, 6, 1.0f,
                              false, 2e-2, 2e-2);
  test_l2_norm<float>("l2_norm fp32 h250 tail", kops::DType::kFloat32, 3, 250, 2, 0.5f, false, 1e-5,
                      1e-5);

  // ---- gated_rms_norm ------------------------------------------------------
  test_gated<float>("gated fp32 g128 silu before w", kops::DType::kFloat32, 4, 512, 128,
                    kops::ActFn::kSiLU, true, true, true, 0.0f, 0, false, 1e-5, 1e-5);
  test_gated<float>("gated fp32 g128 silu after", kops::DType::kFloat32, 4, 512, 128,
                    kops::ActFn::kSiLU, false, false, false, 0.0f, 0, false, 1e-5, 1e-5);
  test_gated<float>("gated fp32 g256 sigmoid before shared w", kops::DType::kFloat32, 3, 1024, 256,
                    kops::ActFn::kSigmoid, true, true, false, 0.0f, 4, false, 1e-5, 1e-5);
  test_gated<__nv_bfloat16>("gated bf16 g128 sigmoid before", kops::DType::kBFloat16, 4, 4096, 128,
                            kops::ActFn::kSigmoid, true, true, true, 0.0f, 8, false, 2e-2, 2e-2);
  test_gated<__nv_bfloat16>("gated bf16 g256 silu after", kops::DType::kBFloat16, 3, 2048, 256,
                            kops::ActFn::kSiLU, false, true, false, 0.0f, 8, false, 2e-2, 2e-2);
  test_gated<__nv_bfloat16>("gated bf16 g512 identity before w+1", kops::DType::kBFloat16, 2, 2048,
                            512, kops::ActFn::kIdentity, true, true, true, 1.0f, 0, false, 2e-2,
                            2e-2);
  test_gated<__nv_bfloat16>("gated bf16 g1024 silu before inplace", kops::DType::kBFloat16, 2,
                            4096, 1024, kops::ActFn::kSiLU, true, false, false, 0.0f, 0, true,
                            2e-2, 2e-2);
  test_gated<__half>("gated fp16 g128 silu after w", kops::DType::kFloat16, 4, 512, 128,
                     kops::ActFn::kSiLU, false, true, true, 0.0f, 8, false, 1e-3, 1e-3);

  test_contracts();
  test_norm_quant(probe);
  return kops_test::finish("test_norm");
}
