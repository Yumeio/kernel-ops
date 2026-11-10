// =============================================================================
// kernel-ops | tests/unit/test_model_ops.cc
// B0 numerics and contract tests: residual_add and cast.
// =============================================================================
#include "tests/unit/harness.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/ops.h"
#include "csrc/runtime/registry.h"

namespace {

using kops::DType;

void fill(std::vector<float>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (float& x : v) x = d(rng);
}

template <typename T>
std::vector<T> to_type(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(out[i], src[i]);
  return out;
}

template <typename T>
std::vector<double> to_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = kops_test::to_float(src[i]);
  return out;
}

template <typename T>
void test_residual(kops::DType dt, const char* name, bool three) {
  const int64_t rows = 17;
  const int64_t d = 257;  // deliberately not a vector multiple
  std::vector<float> af(rows * d), bf(rows * d), cf(rows * d);
  fill(af, 1u);
  fill(bf, 2u);
  fill(cf, 3u);
  const auto a = to_type<T>(af), b = to_type<T>(bf), c = to_type<T>(cf);
  T* da = kops_test::device_alloc<T>(a.size());
  T* db = kops_test::device_alloc<T>(b.size());
  T* dc = three ? kops_test::device_alloc<T>(c.size()) : nullptr;
  T* dout = kops_test::device_alloc<T>(a.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(da, a);
  kops_test::copy_in(db, b);
  if (three) kops_test::copy_in(dc, c);

  kops::Tensor ta = kops_test::tensor2d(da, dt, rows, d);
  kops::Tensor tb = kops_test::tensor2d(db, dt, rows, d);
  kops::Tensor tc = three ? kops_test::tensor2d(dc, dt, rows, d) : kops::Tensor{};
  kops::Tensor tout = kops_test::tensor2d(dout, dt, rows, d);
  const float as = 0.5f, bs = -1.25f, cs = 2.0f;
  KOPS_EXPECT_OK(kops::model_ops::residual_add(tout, ta, as, tb, bs, tc, cs, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(dout, a.size());
  const auto ad = to_double(a), bd = to_double(b), cd = to_double(c);
  std::vector<double> ref(a.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    ref[i] = as * ad[i] + bs * bd[i] + (three ? cs * cd[i] : 0.0);
  }
  const double atol = dt == DType::kFloat32 ? 1e-5 : 3e-2;
  const double rtol = dt == DType::kFloat32 ? 1e-5 : 3e-2;
  kops_test::compare(got, ref, atol, rtol, name);

  // exact alias: out == a
  if (!three) {
    KOPS_EXPECT_OK(kops::model_ops::residual_add(ta, ta, as, tb, bs, kops::Tensor{}, cs, ctx),
                   "residual_add alias");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto alias = kops_test::copy_out(da, a.size());
    std::vector<double> alias_ref(a.size());
    for (size_t i = 0; i < alias_ref.size(); ++i) {
      alias_ref[i] = as * ad[i] + bs * bd[i];
    }
    kops_test::compare(alias, alias_ref, atol, rtol, "residual_add alias");
  }

  kops_test::device_free(da);
  kops_test::device_free(db);
  kops_test::device_free(dc);
  kops_test::device_free(dout);
}

template <typename Out, typename In>
void test_cast(DType out_dt, DType in_dt, const char* name) {
  const int64_t rows = 11;
  const int64_t d = 129;
  std::vector<float> xf(rows * d);
  fill(xf, 4u);
  const auto x = to_type<In>(xf);
  In* dx = kops_test::device_alloc<In>(x.size());
  Out* dy = kops_test::device_alloc<Out>(x.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(dx, x);
  const kops::Tensor tx = kops_test::tensor2d(dx, in_dt, rows, d);
  const kops::Tensor ty = kops_test::tensor2d(dy, out_dt, rows, d);
  KOPS_EXPECT_OK(kops::model_ops::cast(ty, tx, 1.75f, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(dy, x.size());
  const auto xd = to_double(x);
  std::vector<double> ref(xd.size());
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = 1.75 * xd[i];
  const double tol = out_dt == DType::kFloat32 ? 1e-5 : 3e-2;
  kops_test::compare(got, ref, tol, tol, name);
  kops_test::device_free(dx);
  kops_test::device_free(dy);
}

// -----------------------------------------------------------------------------
// Layout and dtype coverage for the persistent chunk kernel: the same logical
// row data through the 32 B chunk path (contiguous), the 16 B / padded-stride
// path and the scalar path (odd stride), plus exact in-place reuse. Every path
// must agree bit for bit with the contiguous one, because each output element
// is produced by one thread with one rounding.
// -----------------------------------------------------------------------------

kops::Tensor view2d(void* data, kops::DType dt, int64_t rows, int64_t cols, int64_t stride) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dt;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = stride;
  t.stride[1] = 1;
  return t;
}

template <typename T>
std::vector<T> gather_rows(const std::vector<T>& body, int64_t rows, int64_t cols,
                           int64_t stride) {
  std::vector<T> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) out[static_cast<size_t>(r * cols + c)] = body[static_cast<size_t>(r * stride + c)];
  }
  return out;
}

template <typename T>
std::vector<T> spread_rows(const std::vector<T>& flat, int64_t rows, int64_t cols,
                           int64_t stride) {
  std::vector<T> body(static_cast<size_t>(rows * stride));
  for (auto& v : body) kops_test::from_float(v, 7.0f);  // poison the padding
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) body[static_cast<size_t>(r * stride + c)] = flat[static_cast<size_t>(r * cols + c)];
  }
  return body;
}

/// One residual_add configuration through every layout path.
template <typename T>
void test_residual_layouts(kops::DType dt, const char* name, int64_t rows, int64_t cols,
                           bool three) {
  const float as = 0.5f, bs = -1.25f, cs = 2.0f;
  std::vector<float> af(static_cast<size_t>(rows * cols)), bf(af.size()), cf(af.size());
  fill(af, 7u);
  fill(bf, 8u);
  fill(cf, 9u);
  const std::vector<T> a = to_type<T>(af), b = to_type<T>(bf), c = to_type<T>(cf);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  // Reference: contiguous views (the chunk path).
  T* da = kops_test::device_alloc<T>(a.size());
  T* db = kops_test::device_alloc<T>(b.size());
  T* dc = three ? kops_test::device_alloc<T>(c.size()) : nullptr;
  T* dout = kops_test::device_alloc<T>(a.size());
  if (da == nullptr || db == nullptr || dout == nullptr || (three && dc == nullptr)) return;
  kops_test::copy_in(da, a);
  kops_test::copy_in(db, b);
  if (three) kops_test::copy_in(dc, c);
  KOPS_EXPECT_OK(kops::model_ops::residual_add(view2d(dout, dt, rows, cols, cols),
                                               view2d(da, dt, rows, cols, cols), as,
                                               view2d(db, dt, rows, cols, cols), bs,
                                               three ? view2d(dc, dt, rows, cols, cols) : kops::Tensor{},
                                               cs, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto ref = kops_test::copy_out(dout, a.size());
  const auto ref_d = to_double(ref);

  // Same data with a padded row stride (keeps the vector path) and with an odd
  // stride (forces the scalar path). Both must match the reference exactly.
  for (const int64_t pad : {int64_t{8}, int64_t{1}}) {
    const int64_t stride = cols + pad;
    const auto ab = spread_rows(a, rows, cols, stride);
    const auto bb = spread_rows(b, rows, cols, stride);
    const auto cb = spread_rows(c, rows, cols, stride);
    T* pa = kops_test::device_alloc<T>(ab.size());
    T* pb = kops_test::device_alloc<T>(bb.size());
    T* pc = three ? kops_test::device_alloc<T>(cb.size()) : nullptr;
    T* po = kops_test::device_alloc<T>(ab.size());
    if (pa == nullptr || pb == nullptr || po == nullptr || (three && pc == nullptr)) return;
    kops_test::copy_in(pa, ab);
    kops_test::copy_in(pb, bb);
    if (three) kops_test::copy_in(pc, cb);
    KOPS_EXPECT_OK(kops::model_ops::residual_add(view2d(po, dt, rows, cols, stride),
                                                 view2d(pa, dt, rows, cols, stride), as,
                                                 view2d(pb, dt, rows, cols, stride), bs,
                                                 three ? view2d(pc, dt, rows, cols, stride)
                                                       : kops::Tensor{},
                                                 cs, ctx),
                   name);
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = gather_rows(kops_test::copy_out(po, ab.size()), rows, cols, stride);
    kops_test::compare(got, ref_d, 0.0, 0.0, name);
    kops_test::device_free(pa);
    kops_test::device_free(pb);
    kops_test::device_free(pc);
    kops_test::device_free(po);
  }

  // Exact in-place output (out == a) must reproduce the out-of-place result.
  kops_test::copy_in(da, a);
  KOPS_EXPECT_OK(kops::model_ops::residual_add(view2d(da, dt, rows, cols, cols),
                                              view2d(da, dt, rows, cols, cols), as,
                                              view2d(db, dt, rows, cols, cols), bs,
                                              three ? view2d(dc, dt, rows, cols, cols) : kops::Tensor{},
                                              cs, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  kops_test::compare(kops_test::copy_out(da, a.size()), ref_d, 0.0, 0.0, name);

  kops_test::device_free(da);
  kops_test::device_free(db);
  kops_test::device_free(dc);
  kops_test::device_free(dout);
}

/// Odd hidden sizes exercise the tail loops of the scalar path.
template <typename T>
void test_residual_odd(kops::DType dt, const char* name, int64_t cols) {
  const int64_t rows = 5;
  const float as = 0.75f, bs = -2.0f;
  std::vector<float> af(static_cast<size_t>(rows * cols)), bf(af.size());
  fill(af, 21u);
  fill(bf, 22u);
  const std::vector<T> a = to_type<T>(af), b = to_type<T>(bf);
  T* da = kops_test::device_alloc<T>(a.size());
  T* db = kops_test::device_alloc<T>(b.size());
  T* dout = kops_test::device_alloc<T>(a.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(da, a);
  kops_test::copy_in(db, b);
  KOPS_EXPECT_OK(kops::model_ops::residual_add(view2d(dout, dt, rows, cols, cols),
                                               view2d(da, dt, rows, cols, cols), as,
                                               view2d(db, dt, rows, cols, cols), bs,
                                               kops::Tensor{}, 0.0f, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(dout, a.size());
  const auto ad = to_double(a), bd = to_double(b);
  std::vector<double> ref(a.size());
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = as * ad[i] + bs * bd[i];
  const double tol = dt == DType::kFloat32 ? 1e-5 : 3e-2;
  kops_test::compare(got, ref, tol, tol, name);
  kops_test::device_free(da);
  kops_test::device_free(db);
  kops_test::device_free(dout);
}

/// One cast configuration through the contiguous, padded and odd-stride paths.
template <typename Out, typename In>
void test_cast_layouts(DType out_dt, DType in_dt, const char* name, int64_t rows, int64_t cols,
                       float scale) {
  std::vector<float> xf(static_cast<size_t>(rows * cols));
  fill(xf, 31u);
  const std::vector<In> x = to_type<In>(xf);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  In* dx = kops_test::device_alloc<In>(x.size());
  Out* dy = kops_test::device_alloc<Out>(x.size());
  if (dx == nullptr || dy == nullptr) return;
  kops_test::copy_in(dx, x);
  KOPS_EXPECT_OK(kops::model_ops::cast(view2d(dy, out_dt, rows, cols, cols),
                                       view2d(dx, in_dt, rows, cols, cols), scale, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto ref = kops_test::copy_out(dy, x.size());
  const auto ref_d = to_double(ref);

  for (const int64_t pad : {int64_t{8}, int64_t{1}}) {
    const int64_t stride = cols + pad;
    const auto xb = spread_rows(x, rows, cols, stride);
    In* px = kops_test::device_alloc<In>(xb.size());
    Out* py = kops_test::device_alloc<Out>(xb.size());
    if (px == nullptr || py == nullptr) return;
    kops_test::copy_in(px, xb);
    KOPS_EXPECT_OK(kops::model_ops::cast(view2d(py, out_dt, rows, cols, stride),
                                         view2d(px, in_dt, rows, cols, stride), scale, ctx), name);
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = gather_rows(kops_test::copy_out(py, xb.size()), rows, cols, stride);
    kops_test::compare(got, ref_d, 0.0, 0.0, name);
    kops_test::device_free(px);
    kops_test::device_free(py);
  }
  kops_test::device_free(dx);
  kops_test::device_free(dy);
}

// -----------------------------------------------------------------------------
// B15: attention residuals (AttnRes). One CTA per token row, fp32 reductions,
// softmax over depth, optional fused output RMSNorm + MXFP8, commit boundary.
// The fp64 reference below is a direct transcription of the documented math and
// every check goes through the public host entries.
// -----------------------------------------------------------------------------

double ref_e4m3_value(uint8_t code) {
  return static_cast<double>(__half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
}

/// UE8M0 exponent byte of a positive scale, rounded up (OCP MX recipe).
uint8_t ref_ue8m0(float s) {
  int e = -127;
  if (s > 0.0f && std::isfinite(s)) {
    int ex = 0;
    std::frexp(s, &ex);
    e = ex - 1;
    if (std::ldexp(1.0f, e) < s) ++e;
    if (e > 127) e = 127;
  }
  return static_cast<uint8_t>(e + 127);
}

struct RefArgs {
  int num_sources = 0;
  int tokens = 0;
  int d = 0;
  bool include_partial = false;
  double key_eps = 0.0;
  double out_norm_eps = 0.0;
  const std::vector<double>* sources = nullptr;          // [S, T, d]
  const std::vector<double>* partial = nullptr;          // [T, d], before delta
  const std::vector<double>* delta = nullptr;            // [T, d] or null
  const std::vector<double>* query = nullptr;            // [d]
  const std::vector<double>* knw = nullptr;              // [d] or null
  const std::vector<double>* rinv = nullptr;             // [S, T] or null
  const std::vector<double>* out_norm_weight = nullptr;  // [d] or null
};

struct RefResult {
  std::vector<double> h;       // [T, d]
  std::vector<double> normed;  // [T, d]
  std::vector<double> alpha;   // [T, S + 1]
};

RefResult attn_ref(const RefArgs& a) {
  const int depth = a.num_sources + (a.include_partial ? 1 : 0);
  const size_t n = static_cast<size_t>(a.tokens) * a.d;
  RefResult r;
  r.h.assign(n, 0.0);
  r.normed.assign(n, 0.0);
  r.alpha.assign(static_cast<size_t>(a.tokens) * (a.num_sources + 1), 0.0);
  std::vector<double> prow(static_cast<size_t>(a.d));
  std::vector<double> logit(static_cast<size_t>(depth));
  std::vector<double> w(static_cast<size_t>(depth));

  for (int t = 0; t < a.tokens; ++t) {
    const double* partial_row = nullptr;
    if (a.partial != nullptr) {
      for (int j = 0; j < a.d; ++j) {
        prow[j] = (*a.partial)[static_cast<size_t>(t) * a.d + j];
        if (a.delta != nullptr) prow[j] += (*a.delta)[static_cast<size_t>(t) * a.d + j];
      }
      partial_row = prow.data();
    }
    auto row_of = [&](int i) -> const double* {
      return i < a.num_sources ? a.sources->data() + (static_cast<size_t>(i) * a.tokens + t) * a.d
                               : partial_row;
    };
    double m = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < depth; ++i) {
      const double* v = row_of(i);
      double sq = 0.0, dot = 0.0;
      for (int j = 0; j < a.d; ++j) {
        sq += v[j] * v[j];
        dot += v[j] * (*a.query)[j] * (a.knw != nullptr ? (*a.knw)[j] : 1.0);
      }
      const double rinv = (i < a.num_sources && a.rinv != nullptr)
                              ? (*a.rinv)[static_cast<size_t>(i) * a.tokens + t]
                              : 1.0 / std::sqrt(sq / a.d + a.key_eps);
      logit[i] = rinv * dot;
      m = std::max(m, logit[i]);
    }
    double denom = 0.0;
    for (int i = 0; i < depth; ++i) {
      w[i] = std::exp(logit[i] - m);
      denom += w[i];
    }
    for (int i = 0; i < depth; ++i) w[i] /= denom;
    double* h = r.h.data() + static_cast<size_t>(t) * a.d;
    for (int i = 0; i < depth; ++i) {
      const double* v = row_of(i);
      for (int j = 0; j < a.d; ++j) h[j] += w[i] * v[j];
    }
    for (int i = 0; i < depth; ++i) {
      r.alpha[static_cast<size_t>(t) * (a.num_sources + 1) + i] = w[i];
    }
    if (a.out_norm_weight != nullptr) {
      double sq = 0.0;
      for (int j = 0; j < a.d; ++j) sq += h[j] * h[j];
      const double rstd = 1.0 / std::sqrt(sq / a.d + a.out_norm_eps);
      double* nr = r.normed.data() + static_cast<size_t>(t) * a.d;
      for (int j = 0; j < a.d; ++j) nr[j] = h[j] * rstd * (*a.out_norm_weight)[j];
    }
  }
  return r;
}

template <typename T>
T* upload(const std::vector<float>& src) {
  const std::vector<T> v = to_type<T>(src);
  T* p = kops_test::device_alloc<T>(v.size());
  if (p != nullptr) kops_test::copy_in(p, v);
  return p;
}

template <typename T>
std::vector<double> download_d(const T* src, size_t count) {
  return to_double(kops_test::copy_out(src, count));
}

/// (a) two completed sources +/- the running partial: raw output and the debug
/// coefficients, through a padded source/token/row stride (still vectorized).
template <typename T>
void test_attn_res_basic(kops::DType dt, const char* name, int64_t d, int64_t pad) {
  const int64_t rows = 3, ns = 2;
  std::vector<float> sf(static_cast<size_t>(ns * rows * d)), pf(static_cast<size_t>(rows * d));
  std::vector<float> qf(static_cast<size_t>(d)), kf(static_cast<size_t>(d));
  fill(sf, 41u);
  fill(pf, 42u);
  fill(qf, 43u);
  fill(kf, 44u);
  const std::vector<T> src_t = to_type<T>(sf), part_t = to_type<T>(pf);
  const std::vector<T> q_t = to_type<T>(qf), k_t = to_type<T>(kf);

  const int64_t sstride = rows * (d + pad) + pad;   // elements between sources
  const int64_t tstride = d + pad;                  // elements between token rows
  std::vector<T> src_st(static_cast<size_t>(ns * sstride));
  std::vector<T> part_st(static_cast<size_t>(rows * tstride));
  for (auto& x : src_st) kops_test::from_float(x, -7.0f);   // poison the padding
  for (auto& x : part_st) kops_test::from_float(x, -7.0f);
  for (int64_t i = 0; i < ns; ++i) {
    for (int64_t t = 0; t < rows; ++t) {
      for (int64_t j = 0; j < d; ++j) {
        src_st[static_cast<size_t>(i * sstride + t * tstride + j)] =
            src_t[static_cast<size_t>((i * rows + t) * d + j)];
      }
    }
  }
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t j = 0; j < d; ++j) {
      part_st[static_cast<size_t>(t * tstride + j)] = part_t[static_cast<size_t>(t * d + j)];
    }
  }

  T* dev_src = kops_test::device_alloc<T>(src_st.size());
  T* dev_part = kops_test::device_alloc<T>(part_st.size());
  T* dev_q = upload<T>(qf);
  T* dev_k = upload<T>(kf);
  T* dev_out = kops_test::device_alloc<T>(static_cast<size_t>(rows * tstride));
  float* dev_alpha = kops_test::device_alloc<float>(static_cast<size_t>(rows * (ns + 1)));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || dev_src == nullptr || dev_part == nullptr ||
      dev_q == nullptr || dev_k == nullptr || dev_out == nullptr || dev_alpha == nullptr) {
    return;
  }
  kops_test::copy_in(dev_src, src_st);
  kops_test::copy_in(dev_part, part_st);

  const auto src_d = to_double(src_t), part_d = to_double(part_t), q_d = to_double(q_t),
             k_d = to_double(k_t);
  const double tol = dt == DType::kFloat32 ? 2e-5 : 2e-2;
  for (const bool with_partial : {true, false}) {
    kops::model_ops::AttnResParams p{};
    p.sources = dev_src;
    p.source_stride = sstride;
    p.source_token_stride = tstride;
    p.num_sources = static_cast<int32_t>(ns);
    p.partial = dev_part;
    p.partial_stride = tstride;
    p.include_partial = with_partial;
    p.query = dev_q;
    p.key_norm_weight = dev_k;
    p.key_eps = 1e-5f;
    p.out = dev_out;
    p.out_stride = tstride;
    p.alpha_out = dev_alpha;
    p.num_tokens = static_cast<int32_t>(rows);
    p.d = static_cast<int32_t>(d);
    KOPS_EXPECT_OK(kops::model_ops::attn_res(p, dt, ctx), name);
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

    const auto got = kops_test::copy_out(dev_out, static_cast<size_t>(rows * tstride));
    std::vector<T> gathered(static_cast<size_t>(rows * d));
    for (int64_t t = 0; t < rows; ++t) {
      for (int64_t j = 0; j < d; ++j) {
        gathered[static_cast<size_t>(t * d + j)] = got[static_cast<size_t>(t * tstride + j)];
      }
    }
    RefArgs ra{};
    ra.num_sources = static_cast<int>(ns);
    ra.tokens = static_cast<int>(rows);
    ra.d = static_cast<int>(d);
    ra.include_partial = with_partial;
    ra.key_eps = 1e-5;
    ra.sources = &src_d;
    ra.partial = &part_d;
    ra.query = &q_d;
    ra.knw = &k_d;
    const RefResult rr = attn_ref(ra);
    kops_test::compare(gathered, rr.h, tol, tol, name);
    const auto alpha = kops_test::copy_out(dev_alpha, static_cast<size_t>(rows * (ns + 1)));
    std::vector<double> alpha_d(alpha.begin(), alpha.end());
    kops_test::compare(alpha_d, rr.alpha, 1e-6, 1e-6, name);
  }

  // Exact in-place output: `out` is the partial row itself. It is read as the
  // last source and then overwritten column by column, so the mix must reproduce
  // the out-of-place result.
  {
    kops::model_ops::AttnResParams p{};
    p.sources = dev_src;
    p.source_stride = sstride;
    p.source_token_stride = tstride;
    p.num_sources = static_cast<int32_t>(ns);
    p.partial = dev_part;
    p.partial_stride = tstride;
    p.include_partial = true;
    p.query = dev_q;
    p.key_norm_weight = dev_k;
    p.key_eps = 1e-5f;
    p.out = dev_part;
    p.out_stride = tstride;
    p.num_tokens = static_cast<int32_t>(rows);
    p.d = static_cast<int32_t>(d);
    KOPS_EXPECT_OK(kops::model_ops::attn_res(p, dt, ctx), name);
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = kops_test::copy_out(dev_part, static_cast<size_t>(rows * tstride));
    std::vector<T> gathered(static_cast<size_t>(rows * d));
    for (int64_t t = 0; t < rows; ++t) {
      for (int64_t j = 0; j < d; ++j) {
        gathered[static_cast<size_t>(t * d + j)] = got[static_cast<size_t>(t * tstride + j)];
      }
    }
    RefArgs ra{};
    ra.num_sources = static_cast<int>(ns);
    ra.tokens = static_cast<int>(rows);
    ra.d = static_cast<int>(d);
    ra.include_partial = true;
    ra.key_eps = 1e-5;
    ra.sources = &src_d;
    ra.partial = &part_d;
    ra.query = &q_d;
    ra.knw = &k_d;
    const RefResult rr = attn_ref(ra);
    kops_test::compare(gathered, rr.h, tol, tol, name);
  }

  kops_test::device_free(dev_src);
  kops_test::device_free(dev_part);
  kops_test::device_free(dev_q);
  kops_test::device_free(dev_k);
  kops_test::device_free(dev_out);
  kops_test::device_free(dev_alpha);
}

/// (b) delta accumulation, then the block-boundary commit: the partial becomes a
/// source slot with a cached fp32 inverse RMS, and restarts from `restart_with`.
void test_attn_res_commit_boundary() {
  const int64_t rows = 2, d = 120;
  const int32_t slots = 3;
  std::vector<float> sf(static_cast<size_t>(slots * rows * d)), pf(static_cast<size_t>(rows * d));
  std::vector<float> df(static_cast<size_t>(rows * d)), rf(static_cast<size_t>(rows * d));
  std::vector<float> qf(static_cast<size_t>(d));
  fill(sf, 51u);
  fill(pf, 52u);
  fill(df, 53u);
  fill(rf, 54u);
  fill(qf, 55u);
  // Slots 0 and 1 hold real block residuals; slot 2 is written by the commit.
  for (int64_t i = 2 * rows * d; i < slots * rows * d; ++i) sf[static_cast<size_t>(i)] = 0.0f;

  float* dev_src = upload<float>(sf);
  float* dev_part = upload<float>(pf);
  float* dev_delta = upload<float>(df);
  float* dev_restart = upload<float>(rf);
  float* dev_q = upload<float>(qf);
  float* dev_rinv = kops_test::device_alloc<float>(static_cast<size_t>(slots * rows));
  float* dev_out = kops_test::device_alloc<float>(static_cast<size_t>(rows * d));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || dev_src == nullptr || dev_part == nullptr ||
      dev_delta == nullptr || dev_restart == nullptr || dev_q == nullptr ||
      dev_rinv == nullptr || dev_out == nullptr) {
    return;
  }
  const float eps = 1e-5f;
  const kops::Tensor ts = kops_test::tensor3d(dev_src, DType::kFloat32, slots, rows, d);
  const kops::Tensor trinv = kops_test::tensor2d(dev_rinv, DType::kFloat32, slots, rows);
  const kops::Tensor tp = kops_test::tensor2d(dev_part, DType::kFloat32, rows, d);
  const kops::Tensor trw = kops_test::tensor2d(dev_restart, DType::kFloat32, rows, d);

  // 1) fused residual update: partial += delta, and the partial mixes as the
  //    last source (so `out` must use P + D).
  kops::model_ops::AttnResParams p{};
  p.sources = dev_src;
  p.source_stride = rows * d;
  p.source_token_stride = d;
  p.num_sources = 2;
  p.partial = dev_part;
  p.partial_stride = d;
  p.delta = dev_delta;
  p.delta_stride = d;
  p.include_partial = true;
  p.query = dev_q;
  p.key_eps = eps;
  p.out = dev_out;
  p.out_stride = d;
  p.num_tokens = static_cast<int32_t>(rows);
  p.d = static_cast<int32_t>(d);
  KOPS_EXPECT_OK(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                 "attn_res delta + partial");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto pf_d = to_double(to_type<float>(pf));
  const auto df_d = to_double(to_type<float>(df));
  std::vector<double> updated(pf_d.size());
  for (size_t i = 0; i < updated.size(); ++i) updated[i] = pf_d[i] + df_d[i];
  const auto part_after = download_d(dev_part, static_cast<size_t>(rows * d));
  kops_test::compare(part_after, updated, 2e-6, 2e-6, "attn_res partial += delta");

  {
    RefArgs ra{};
    ra.num_sources = 2;
    ra.tokens = static_cast<int>(rows);
    ra.d = static_cast<int>(d);
    ra.include_partial = true;
    ra.key_eps = eps;
    const auto src_d = to_double(to_type<float>(sf));
    std::vector<double> part_before = pf_d;
    ra.sources = &src_d;
    ra.partial = &part_before;
    ra.delta = &df_d;
    const auto q_d = to_double(to_type<float>(qf));
    ra.query = &q_d;
    const RefResult rr = attn_ref(ra);
    const auto got = download_d(dev_out, static_cast<size_t>(rows * d));
    kops_test::compare(got, rr.h, 2e-5, 2e-5, "attn_res delta + partial");
  }

  // 2) commit the partial into slot 2 and restart the partial from R.
  KOPS_EXPECT_OK(kops::model_ops::attn_res_commit(ts, trinv, tp, trw, 2, eps, ctx),
                 "attn_res_commit");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto slot2 = download_d(dev_src + 2 * rows * d, static_cast<size_t>(rows * d));
  // The copy is exact; the reference sum P + D is fp64 while the kernel rounds
  // the fp32 add once, so allow one fp32 add of slack.
  kops_test::compare(slot2, updated, 1e-9, 2e-7, "attn_res_commit copy");
  const auto rinv = kops_test::copy_out(dev_rinv, static_cast<size_t>(slots * rows));
  for (int64_t t = 0; t < rows; ++t) {
    double sq = 0.0;
    for (int64_t j = 0; j < d; ++j) {
      const double v = updated[static_cast<size_t>(t * d + j)];
      sq += v * v;
    }
    const double want = 1.0 / std::sqrt(sq / d + eps);
    const double got = static_cast<double>(rinv[static_cast<size_t>(2 * rows + t)]);
    if (!(std::fabs(got - want) <= 1e-6 * std::fabs(want))) {
      kops_test::record_failure("attn_res_commit rinv", "cached inverse RMS differs", __FILE__,
                                __LINE__);
    }
  }
  const auto restart_after = download_d(dev_part, static_cast<size_t>(rows * d));
  const auto rf_d = to_double(to_type<float>(rf));
  kops_test::compare(restart_after, rf_d, 0.0, 0.0, "attn_res_commit restart");

  // 3) the committed slot participates as a completed source; feed the fp32
  //    cache from the host so the cached path is exercised too.
  std::vector<float> rinv_host(static_cast<size_t>(slots * rows));
  std::vector<double> src_d = to_double(to_type<float>(sf));
  for (int64_t t = 0; t < rows; ++t) {   // the committed slot holds P + D now
    for (int64_t j = 0; j < d; ++j) {
      src_d[static_cast<size_t>((2 * rows + t) * d + j)] =
          slot2[static_cast<size_t>(t * d + j)];
    }
  }
  for (int64_t i = 0; i < slots; ++i) {
    for (int64_t t = 0; t < rows; ++t) {
      double sq = 0.0;
      for (int64_t j = 0; j < d; ++j) {
        const double v = src_d[static_cast<size_t>((i * rows + t) * d + j)];
        sq += v * v;
      }
      rinv_host[static_cast<size_t>(i * rows + t)] =
          static_cast<float>(1.0 / std::sqrt(sq / d + eps));
    }
  }
  rinv_host[static_cast<size_t>(2 * rows + 0)] = rinv[static_cast<size_t>(2 * rows + 0)];
  rinv_host[static_cast<size_t>(2 * rows + 1)] = rinv[static_cast<size_t>(2 * rows + 1)];
  kops_test::copy_in(dev_rinv, rinv_host);
  std::vector<double> rinv_d(rinv_host.begin(), rinv_host.end());
  {
    kops::model_ops::AttnResParams q = p;
    q.num_sources = 3;
    q.source_rinv = dev_rinv;
    q.delta = nullptr;
    q.include_partial = false;   // right after a commit the partial is not a source
    KOPS_EXPECT_OK(kops::model_ops::attn_res(q, DType::kFloat32, ctx),
                   "attn_res cached rinv");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    RefArgs ra{};
    ra.num_sources = 3;
    ra.tokens = static_cast<int>(rows);
    ra.d = static_cast<int>(d);
    ra.include_partial = false;
    ra.key_eps = eps;
    ra.sources = &src_d;
    ra.rinv = &rinv_d;
    const auto q_d = to_double(to_type<float>(qf));
    ra.query = &q_d;
    const RefResult rr = attn_ref(ra);
    const auto got = download_d(dev_out, static_cast<size_t>(rows * d));
    kops_test::compare(got, rr.h, 2e-5, 2e-5, "attn_res cached rinv");
  }

  // 4) commit with no restart source zeroes the partial (after the copy).
  KOPS_EXPECT_OK(kops::model_ops::attn_res_commit(ts, trinv, tp, kops::Tensor{}, 2, eps, ctx),
                 "attn_res_commit zero restart");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto zeroed = download_d(dev_part, static_cast<size_t>(rows * d));
  kops_test::compare(zeroed, std::vector<double>(static_cast<size_t>(rows * d), 0.0), 0.0, 0.0,
                     "attn_res_commit zero restart");
  const auto slot2b = download_d(dev_src + 2 * rows * d, static_cast<size_t>(rows * d));
  kops_test::compare(slot2b, rf_d, 0.0, 0.0, "attn_res_commit copy of restart");

  kops_test::device_free(dev_src);
  kops_test::device_free(dev_part);
  kops_test::device_free(dev_delta);
  kops_test::device_free(dev_restart);
  kops_test::device_free(dev_q);
  kops_test::device_free(dev_rinv);
  kops_test::device_free(dev_out);
}

/// (c) fused output RMSNorm and the MXFP8 epilogue: the payload must reproduce
/// the normed row with the UE8M0 scale the kernel wrote (amax / 448 rounded up).
void test_attn_res_norm_quant() {
  const int64_t rows = 2, d = 64, ns = 3, groups = d / 32;
  std::vector<float> sf(static_cast<size_t>(ns * rows * d)), pf(static_cast<size_t>(rows * d));
  std::vector<float> qf(static_cast<size_t>(d)), kf(static_cast<size_t>(d)),
      wf(static_cast<size_t>(d));
  fill(sf, 61u);
  fill(pf, 62u);
  fill(qf, 63u);
  fill(kf, 64u);
  fill(wf, 65u);

  float* dev_src = upload<float>(sf);
  float* dev_part = upload<float>(pf);
  float* dev_q = upload<float>(qf);
  float* dev_k = upload<float>(kf);
  float* dev_w = upload<float>(wf);
  float* dev_out = kops_test::device_alloc<float>(static_cast<size_t>(rows * d));
  float* dev_normed = kops_test::device_alloc<float>(static_cast<size_t>(rows * d));
  uint8_t* dev_qout = kops_test::device_alloc<uint8_t>(static_cast<size_t>(rows * d));
  float* dev_scales = kops_test::device_alloc<float>(static_cast<size_t>(rows * groups));
  float* dev_alpha = kops_test::device_alloc<float>(static_cast<size_t>(rows * (ns + 1)));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || dev_src == nullptr || dev_part == nullptr ||
      dev_q == nullptr || dev_k == nullptr || dev_w == nullptr || dev_out == nullptr ||
      dev_normed == nullptr || dev_qout == nullptr || dev_scales == nullptr ||
      dev_alpha == nullptr) {
    return;
  }
  kops::model_ops::AttnResParams p{};
  p.sources = dev_src;
  p.source_stride = rows * d;
  p.source_token_stride = d;
  p.num_sources = static_cast<int32_t>(ns);
  p.partial = dev_part;
  p.partial_stride = d;
  p.include_partial = true;
  p.query = dev_q;
  p.key_norm_weight = dev_k;
  p.key_eps = 1e-5f;
  p.out = dev_out;
  p.out_stride = d;
  p.normed_out = dev_normed;
  p.out_norm_weight = dev_w;
  p.out_norm_eps = 1e-6f;
  p.normed_quant.q = dev_qout;
  p.normed_quant.q_row_stride = d;
  p.normed_quant.scales = dev_scales;
  p.normed_quant.scale_row_stride = groups;
  p.normed_quant.q_dtype = DType::kFloat8E4M3;
  p.normed_quant.mode = kops::ActQuantMode::kMXFP8;
  p.normed_quant.scale_layout = kops::ScaleLayout::kRowMajor;
  p.normed_quant.group_size = 32;
  p.normed_quant.scale_ue8m0 = true;
  p.normed_quant.fp8 = kops::Fp8Encoding::kOCP;
  p.alpha_out = dev_alpha;
  p.num_tokens = static_cast<int32_t>(rows);
  p.d = static_cast<int32_t>(d);
  KOPS_EXPECT_OK(kops::model_ops::attn_res(p, DType::kFloat32, ctx), "attn_res norm + MXFP8");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto src_d = to_double(to_type<float>(sf));
  const auto part_d = to_double(to_type<float>(pf));
  const auto q_d = to_double(to_type<float>(qf));
  const auto k_d = to_double(to_type<float>(kf));
  const auto w_d = to_double(to_type<float>(wf));
  RefArgs ra{};
  ra.num_sources = static_cast<int>(ns);
  ra.tokens = static_cast<int>(rows);
  ra.d = static_cast<int>(d);
  ra.include_partial = true;
  ra.key_eps = 1e-5;
  ra.sources = &src_d;
  ra.partial = &part_d;
  ra.query = &q_d;
  ra.knw = &k_d;
  ra.out_norm_weight = &w_d;
  ra.out_norm_eps = 1e-6;
  const RefResult rr = attn_ref(ra);
  const auto h = download_d(dev_out, static_cast<size_t>(rows * d));
  const auto normed = download_d(dev_normed, static_cast<size_t>(rows * d));
  kops_test::compare(h, rr.h, 2e-5, 2e-5, "attn_res norm h");
  kops_test::compare(normed, rr.normed, 2e-5, 2e-5, "attn_res normed_out");

  const auto codes = kops_test::copy_out(dev_qout, static_cast<size_t>(rows * d));
  const auto scales = kops_test::copy_out(dev_scales, static_cast<size_t>(rows * groups));
  std::vector<double> dequant(static_cast<size_t>(rows * d));
  double amax_all = 0.0;
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t g = 0; g < groups; ++g) {
      float amax = 0.0f;
      for (int64_t j = 0; j < 32; ++j) {
        const float y = static_cast<float>(normed[static_cast<size_t>(t * d + g * 32 + j)]);
        amax = std::fmax(amax, std::fabs(y));
      }
      amax_all = std::max(amax_all, static_cast<double>(amax));
      const uint8_t want = ref_ue8m0(std::fmax(amax, 1e-10f) / 448.0f);
      const float got = scales[static_cast<size_t>(t * groups + g)];
      // The scale is a power of two written through an e4m3->float decode path
      // of the exponent byte; compare the exponent, not the float bits.
      const uint8_t got_e = ref_ue8m0(got);
      if (got_e != want || !(got > 0.0f)) {
        kops_test::record_failure("attn_res MXFP8 scale", "group scale mismatch", __FILE__,
                                  __LINE__);
      }
      for (int64_t j = 0; j < 32; ++j) {
        dequant[static_cast<size_t>(t * d + g * 32 + j)] =
            ref_e4m3_value(codes[static_cast<size_t>(t * d + g * 32 + j)]) *
            static_cast<double>(got);
      }
    }
  }
  kops_test::compare(dequant, normed, std::fmax(1e-5 * amax_all, 1e-9), 0.09,
                     "attn_res MXFP8 payload");
}

/// Every element of this file's new tests must be bit-identical for a row alone
/// and the same row inside a batch (no atomics, fixed reduction order).
void test_attn_res_batch_invariance() {
  const int64_t rows = 4, d = 256, ns = 3, groups = d / 32, row_single = 2;
  std::vector<float> sf(static_cast<size_t>(ns * rows * d)), pf(static_cast<size_t>(rows * d));
  std::vector<float> df(static_cast<size_t>(rows * d)), qf(static_cast<size_t>(d)),
      kf(static_cast<size_t>(d)), wf(static_cast<size_t>(d));
  fill(sf, 71u);
  fill(pf, 72u);
  fill(df, 73u);
  fill(qf, 74u);
  fill(kf, 75u);
  fill(wf, 76u);

  using bf16 = __nv_bfloat16;
  bf16* dev_src = upload<bf16>(sf);
  bf16* dev_part = upload<bf16>(pf);
  bf16* dev_delta = upload<bf16>(df);
  bf16* dev_q = upload<bf16>(qf);
  bf16* dev_k = upload<bf16>(kf);
  bf16* dev_w = upload<bf16>(wf);
  bf16* dev_out = kops_test::device_alloc<bf16>(static_cast<size_t>(rows * d));
  bf16* dev_normed = kops_test::device_alloc<bf16>(static_cast<size_t>(rows * d));
  uint8_t* dev_qout = kops_test::device_alloc<uint8_t>(static_cast<size_t>(rows * d));
  float* dev_scales = kops_test::device_alloc<float>(static_cast<size_t>(rows * groups));
  float* dev_alpha = kops_test::device_alloc<float>(static_cast<size_t>(rows * (ns + 1)));
  float* dev_rinv = kops_test::device_alloc<float>(static_cast<size_t>(ns * rows));
  // Single-row buffers: fresh partial (so the delta update is applied to the
  // same starting value) and fresh outputs.
  bf16* one_part = kops_test::device_alloc<bf16>(static_cast<size_t>(d));
  bf16* one_out = kops_test::device_alloc<bf16>(static_cast<size_t>(d));
  bf16* one_normed = kops_test::device_alloc<bf16>(static_cast<size_t>(d));
  uint8_t* one_qout = kops_test::device_alloc<uint8_t>(static_cast<size_t>(d));
  float* one_scales = kops_test::device_alloc<float>(static_cast<size_t>(groups));
  float* one_alpha = kops_test::device_alloc<float>(static_cast<size_t>(ns + 1));
  float* one_rinv = kops_test::device_alloc<float>(static_cast<size_t>(ns));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || dev_src == nullptr || dev_part == nullptr ||
      dev_delta == nullptr || dev_q == nullptr || dev_k == nullptr || dev_w == nullptr ||
      dev_out == nullptr || dev_normed == nullptr || dev_qout == nullptr ||
      dev_scales == nullptr || dev_alpha == nullptr || dev_rinv == nullptr ||
      one_part == nullptr || one_out == nullptr || one_normed == nullptr ||
      one_qout == nullptr || one_scales == nullptr || one_alpha == nullptr ||
      one_rinv == nullptr) {
    return;
  }
  // The cached inverse RMS of every source row, computed in fp64 and rounded once.
  std::vector<float> rinv_host(static_cast<size_t>(ns * rows));
  const auto src_d = to_double(to_type<bf16>(sf));
  for (int64_t i = 0; i < ns; ++i) {
    for (int64_t t = 0; t < rows; ++t) {
      double sq = 0.0;
      for (int64_t j = 0; j < d; ++j) {
        const double v = src_d[static_cast<size_t>((i * rows + t) * d + j)];
        sq += v * v;
      }
      rinv_host[static_cast<size_t>(i * rows + t)] =
          static_cast<float>(1.0 / std::sqrt(sq / d + 1e-5));
    }
  }
  kops_test::copy_in(dev_rinv, rinv_host);
  std::vector<float> one_rinv_host(static_cast<size_t>(ns));
  for (int64_t i = 0; i < ns; ++i) {
    one_rinv_host[static_cast<size_t>(i)] = rinv_host[static_cast<size_t>(i * rows + row_single)];
  }
  kops_test::copy_in(one_rinv, one_rinv_host);
  {
    const std::vector<bf16> part_all = to_type<bf16>(pf);
    const std::vector<bf16> part_one(part_all.begin() + row_single * d,
                                     part_all.begin() + (row_single + 1) * d);
    kops_test::copy_in(one_part, part_one);
  }

  auto fill_common = [&](kops::model_ops::AttnResParams& p, int64_t tokens) {
    p.sources = dev_src;
    p.source_stride = rows * d;
    p.source_token_stride = d;
    p.num_sources = static_cast<int32_t>(ns);
    p.source_rinv = dev_rinv;
    p.partial_stride = d;
    p.include_partial = true;
    p.query = dev_q;
    p.key_norm_weight = dev_k;
    p.key_eps = 1e-5f;
    p.out_stride = d;
    p.out_norm_weight = dev_w;
    p.out_norm_eps = 1e-5f;
    p.num_tokens = static_cast<int32_t>(tokens);
    p.d = static_cast<int32_t>(d);
    p.normed_quant.q_row_stride = d;
    p.normed_quant.scale_row_stride = groups;
    p.normed_quant.q_dtype = DType::kFloat8E4M3;
    p.normed_quant.mode = kops::ActQuantMode::kMXFP8;
    p.normed_quant.scale_ue8m0 = true;
    p.normed_quant.group_size = 32;
    p.normed_quant.fp8 = kops::Fp8Encoding::kOCP;
  };

  kops::model_ops::AttnResParams full{};
  fill_common(full, rows);
  full.partial = dev_part;
  full.delta = dev_delta;
  full.delta_stride = d;
  full.out = dev_out;
  full.normed_out = dev_normed;
  full.normed_quant.q = dev_qout;
  full.normed_quant.scales = dev_scales;
  full.alpha_out = dev_alpha;
  KOPS_EXPECT_OK(kops::model_ops::attn_res(full, DType::kBFloat16, ctx), "attn_res batch");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  kops::model_ops::AttnResParams one{};
  fill_common(one, 1);
  one.sources = static_cast<const bf16*>(dev_src) + row_single * d;
  one.source_rinv = one_rinv;
  one.partial = one_part;
  one.delta = static_cast<const bf16*>(dev_delta) + row_single * d;
  one.delta_stride = d;
  one.out = one_out;
  one.normed_out = one_normed;
  one.normed_quant.q = one_qout;
  one.normed_quant.scales = one_scales;
  one.alpha_out = one_alpha;
  KOPS_EXPECT_OK(kops::model_ops::attn_res(one, DType::kBFloat16, ctx), "attn_res single row");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  // Bitwise equality between row `row_single` of the batch and the same row run
  // alone, for every output the kernel writes.
  const auto got_out = kops_test::copy_out(dev_out, static_cast<size_t>(rows * d));
  const auto got_normed = kops_test::copy_out(dev_normed, static_cast<size_t>(rows * d));
  const auto got_codes = kops_test::copy_out(dev_qout, static_cast<size_t>(rows * d));
  const auto got_scales = kops_test::copy_out(dev_scales, static_cast<size_t>(rows * groups));
  const auto got_alpha = kops_test::copy_out(dev_alpha, static_cast<size_t>(rows * (ns + 1)));
  std::vector<double> h_full(got_out.begin() + row_single * d,
                             got_out.begin() + (row_single + 1) * d);
  std::vector<double> n_full(got_normed.begin() + row_single * d,
                             got_normed.begin() + (row_single + 1) * d);
  std::vector<double> h_one = download_d(one_out, static_cast<size_t>(d));
  std::vector<double> n_one = download_d(one_normed, static_cast<size_t>(d));
  kops_test::compare(h_one, h_full, 0.0, 0.0, "attn_res batch invariance (out)");
  kops_test::compare(n_one, n_full, 0.0, 0.0, "attn_res batch invariance (normed)");
  const auto codes_one = kops_test::copy_out(one_qout, static_cast<size_t>(d));
  for (int64_t j = 0; j < d; ++j) {
    if (codes_one[static_cast<size_t>(j)] != got_codes[static_cast<size_t>(row_single * d + j)]) {
      kops_test::record_failure("attn_res batch invariance (payload)", "code mismatch", __FILE__,
                                __LINE__);
      break;
    }
  }
  const auto scales_one = kops_test::copy_out(one_scales, static_cast<size_t>(groups));
  for (int64_t g = 0; g < groups; ++g) {
    if (scales_one[static_cast<size_t>(g)] != got_scales[static_cast<size_t>(row_single * groups + g)]) {
      kops_test::record_failure("attn_res batch invariance (scale)", "scale mismatch", __FILE__,
                                __LINE__);
      break;
    }
  }
  const auto alpha_one = kops_test::copy_out(one_alpha, static_cast<size_t>(ns + 1));
  std::vector<double> a_full(got_alpha.begin() + row_single * (ns + 1),
                             got_alpha.begin() + (row_single + 1) * (ns + 1));
  std::vector<double> a_one(alpha_one.begin(), alpha_one.end());
  kops_test::compare(a_one, a_full, 0.0, 0.0, "attn_res batch invariance (alpha)");

  kops_test::device_free(dev_src);
  kops_test::device_free(dev_part);
  kops_test::device_free(dev_delta);
  kops_test::device_free(dev_q);
  kops_test::device_free(dev_k);
  kops_test::device_free(dev_w);
  kops_test::device_free(dev_out);
  kops_test::device_free(dev_normed);
  kops_test::device_free(dev_qout);
  kops_test::device_free(dev_scales);
  kops_test::device_free(dev_alpha);
  kops_test::device_free(dev_rinv);
  kops_test::device_free(one_part);
  kops_test::device_free(one_out);
  kops_test::device_free(one_normed);
  kops_test::device_free(one_qout);
  kops_test::device_free(one_scales);
  kops_test::device_free(one_alpha);
  kops_test::device_free(one_rinv);
}

/// K3-shaped smoke: H=7168, 8 block residuals + partial, delta, fusion of the
/// output norm and the MXFP8 payload.
void test_attn_res_k3() {
  const int64_t rows = 2, d = 7168, ns = 8, groups = d / 32;
  std::vector<float> sf(static_cast<size_t>(ns * rows * d)), pf(static_cast<size_t>(rows * d));
  std::vector<float> df(static_cast<size_t>(rows * d)), qf(static_cast<size_t>(d)),
      kf(static_cast<size_t>(d)), wf(static_cast<size_t>(d));
  fill(sf, 81u);
  fill(pf, 82u);
  fill(df, 83u);
  fill(qf, 84u);
  fill(kf, 85u);
  fill(wf, 86u);
  using bf16 = __nv_bfloat16;
  bf16* dev_src = upload<bf16>(sf);
  bf16* dev_part = upload<bf16>(pf);
  bf16* dev_delta = upload<bf16>(df);
  bf16* dev_q = upload<bf16>(qf);
  bf16* dev_k = upload<bf16>(kf);
  bf16* dev_w = upload<bf16>(wf);
  bf16* dev_out = kops_test::device_alloc<bf16>(static_cast<size_t>(rows * d));
  bf16* dev_normed = kops_test::device_alloc<bf16>(static_cast<size_t>(rows * d));
  uint8_t* dev_qout = kops_test::device_alloc<uint8_t>(static_cast<size_t>(rows * d));
  float* dev_scales = kops_test::device_alloc<float>(static_cast<size_t>(rows * groups));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || dev_src == nullptr || dev_part == nullptr ||
      dev_delta == nullptr || dev_q == nullptr || dev_k == nullptr || dev_w == nullptr ||
      dev_out == nullptr || dev_normed == nullptr || dev_qout == nullptr ||
      dev_scales == nullptr) {
    return;
  }
  kops::model_ops::AttnResParams p{};
  p.sources = dev_src;
  p.source_stride = rows * d;
  p.source_token_stride = d;
  p.num_sources = static_cast<int32_t>(ns);
  p.partial = dev_part;
  p.partial_stride = d;
  p.delta = dev_delta;
  p.delta_stride = d;
  p.include_partial = true;
  p.query = dev_q;
  p.key_norm_weight = dev_k;
  p.key_eps = 1e-5f;
  p.out = dev_out;
  p.out_stride = d;
  p.normed_out = dev_normed;
  p.out_norm_weight = dev_w;
  p.out_norm_eps = 1e-5f;
  p.normed_quant.q = dev_qout;
  p.normed_quant.q_row_stride = d;
  p.normed_quant.scales = dev_scales;
  p.normed_quant.scale_row_stride = groups;
  p.normed_quant.q_dtype = DType::kFloat8E4M3;
  p.normed_quant.mode = kops::ActQuantMode::kMXFP8;
  p.normed_quant.scale_ue8m0 = true;
  p.normed_quant.group_size = 32;
  p.normed_quant.fp8 = kops::Fp8Encoding::kOCP;
  p.num_tokens = static_cast<int32_t>(rows);
  p.d = static_cast<int32_t>(d);
  KOPS_EXPECT_OK(kops::model_ops::attn_res(p, DType::kBFloat16, ctx), "attn_res K3 H7168");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto src_d = to_double(to_type<bf16>(sf));
  const auto part_d = to_double(to_type<bf16>(pf));
  const auto delta_d = to_double(to_type<bf16>(df));
  const auto q_d = to_double(to_type<bf16>(qf));
  const auto k_d = to_double(to_type<bf16>(kf));
  const auto w_d = to_double(to_type<bf16>(wf));
  RefArgs ra{};
  ra.num_sources = static_cast<int>(ns);
  ra.tokens = static_cast<int>(rows);
  ra.d = static_cast<int>(d);
  ra.include_partial = true;
  ra.key_eps = 1e-5;
  ra.sources = &src_d;
  ra.partial = &part_d;
  ra.delta = &delta_d;
  ra.query = &q_d;
  ra.knw = &k_d;
  ra.out_norm_weight = &w_d;
  ra.out_norm_eps = 1e-5;
  const RefResult rr = attn_ref(ra);
  const auto h = download_d(dev_out, static_cast<size_t>(rows * d));
  const auto normed = download_d(dev_normed, static_cast<size_t>(rows * d));
  kops_test::compare(h, rr.h, 2e-2, 2e-2, "attn_res K3 h");
  kops_test::compare(normed, rr.normed, 3e-2, 3e-2, "attn_res K3 normed");
  // The payload must decode back to the normed row within one e4m3 step.
  const auto codes = kops_test::copy_out(dev_qout, static_cast<size_t>(rows * d));
  const auto scales = kops_test::copy_out(dev_scales, static_cast<size_t>(rows * groups));
  std::vector<double> dequant(static_cast<size_t>(rows * d));
  double amax_all = 0.0;
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t g = 0; g < groups; ++g) {
      for (int64_t j = 0; j < 32; ++j) {
        amax_all = std::max(amax_all,
                            std::fabs(normed[static_cast<size_t>(t * d + g * 32 + j)]));
      }
      const double s = static_cast<double>(scales[static_cast<size_t>(t * groups + g)]);
      for (int64_t j = 0; j < 32; ++j) {
        dequant[static_cast<size_t>(t * d + g * 32 + j)] =
            ref_e4m3_value(codes[static_cast<size_t>(t * d + g * 32 + j)]) * s;
      }
    }
  }
  kops_test::compare(dequant, normed, std::fmax(1e-5 * amax_all, 1e-9), 0.09,
                     "attn_res K3 payload");

  kops_test::device_free(dev_src);
  kops_test::device_free(dev_part);
  kops_test::device_free(dev_delta);
  kops_test::device_free(dev_q);
  kops_test::device_free(dev_k);
  kops_test::device_free(dev_w);
  kops_test::device_free(dev_out);
  kops_test::device_free(dev_normed);
  kops_test::device_free(dev_qout);
  kops_test::device_free(dev_scales);
}

/// Degenerate source sets must not produce NaN: an empty depth, and a depth
/// whose logits are all non-finite (a NaN cached inverse RMS), both collapse to
/// zero output and zero payload codes.
void test_attn_res_invalid_sources() {
  const int64_t rows = 2, d = 64, groups = d / 32;
  std::vector<float> sf(static_cast<size_t>(2 * rows * d), std::numeric_limits<float>::quiet_NaN());
  std::vector<float> pf(static_cast<size_t>(rows * d), std::numeric_limits<float>::infinity());
  std::vector<float> qf(static_cast<size_t>(d), 1.0f), kf(static_cast<size_t>(d), 1.0f),
      wf(static_cast<size_t>(d), 1.0f);
  float* dev_src = upload<float>(sf);
  float* dev_part = upload<float>(pf);
  float* dev_q = upload<float>(qf);
  float* dev_k = upload<float>(kf);
  float* dev_w = upload<float>(wf);
  float* dev_rinv = upload<float>(
      std::vector<float>(static_cast<size_t>(2 * rows), std::numeric_limits<float>::quiet_NaN()));
  float* dev_out = kops_test::device_alloc<float>(static_cast<size_t>(rows * d));
  float* dev_normed = kops_test::device_alloc<float>(static_cast<size_t>(rows * d));
  uint8_t* dev_qout = kops_test::device_alloc<uint8_t>(static_cast<size_t>(rows * d));
  float* dev_scales = kops_test::device_alloc<float>(static_cast<size_t>(rows * groups));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || dev_src == nullptr || dev_part == nullptr ||
      dev_q == nullptr || dev_k == nullptr || dev_w == nullptr || dev_rinv == nullptr ||
      dev_out == nullptr || dev_normed == nullptr || dev_qout == nullptr ||
      dev_scales == nullptr) {
    return;
  }
  kops::model_ops::AttnResParams p{};
  p.sources = dev_src;
  p.source_stride = rows * d;
  p.source_token_stride = d;
  p.source_rinv = dev_rinv;
  p.query = dev_q;
  p.key_norm_weight = dev_k;
  p.key_eps = 1e-5f;
  p.out = dev_out;
  p.out_stride = d;
  p.normed_out = dev_normed;
  p.out_norm_weight = dev_w;
  p.out_norm_eps = 1e-6f;
  p.normed_quant.q = dev_qout;
  p.normed_quant.q_row_stride = d;
  p.normed_quant.scales = dev_scales;
  p.normed_quant.scale_row_stride = groups;
  p.normed_quant.q_dtype = DType::kFloat8E4M3;
  p.normed_quant.mode = kops::ActQuantMode::kMXFP8;
  p.num_tokens = static_cast<int32_t>(rows);
  p.d = static_cast<int32_t>(d);

  // All logits non-finite through the cached inverse RMS.
  p.num_sources = 2;
  p.include_partial = true;
  p.partial = dev_part;
  p.partial_stride = d;
  KOPS_EXPECT_OK(kops::model_ops::attn_res(p, DType::kFloat32, ctx), "attn_res non-finite logits");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto nan_out = download_d(dev_out, static_cast<size_t>(rows * d));
  const auto nan_normed = download_d(dev_normed, static_cast<size_t>(rows * d));
  const auto nan_codes = kops_test::copy_out(dev_qout, static_cast<size_t>(rows * d));
  for (const auto& v : nan_out) {
    if (!(v == 0.0)) kops_test::record_failure("attn_res no-NaN", "out is not zero", __FILE__, __LINE__);
  }
  for (const auto& v : nan_normed) {
    if (!(v == 0.0)) {
      kops_test::record_failure("attn_res no-NaN", "normed is not zero", __FILE__, __LINE__);
    }
  }
  for (const auto c : nan_codes) {
    if (c != 0) kops_test::record_failure("attn_res no-NaN", "payload is not zero", __FILE__, __LINE__);
  }

  // Empty source set: nothing to mix, so the outputs are zero as well. Poison
  // the buffer first so the check cannot pass on an untouched allocation.
  KOPS_TEST_CHECK(cudaMemset(dev_out, 0xFF, static_cast<size_t>(rows * d) * sizeof(float)) ==
                  cudaSuccess);
  p.num_sources = 0;
  p.include_partial = false;
  p.partial = nullptr;
  p.source_rinv = nullptr;
  KOPS_EXPECT_OK(kops::model_ops::attn_res(p, DType::kFloat32, ctx), "attn_res empty depth");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto empty_out = download_d(dev_out, static_cast<size_t>(rows * d));
  for (const auto& v : empty_out) {
    if (!(v == 0.0)) {
      kops_test::record_failure("attn_res empty depth", "out is not zero", __FILE__, __LINE__);
    }
  }

  kops_test::device_free(dev_src);
  kops_test::device_free(dev_part);
  kops_test::device_free(dev_q);
  kops_test::device_free(dev_k);
  kops_test::device_free(dev_w);
  kops_test::device_free(dev_rinv);
  kops_test::device_free(dev_out);
  kops_test::device_free(dev_normed);
  kops_test::device_free(dev_qout);
  kops_test::device_free(dev_scales);
}

/// (e) the host contract: every host-visible invariant maps to a Status.
void test_attn_res_contracts() {
  const int64_t rows = 2, d = 64;
  std::vector<float> f(static_cast<size_t>(rows * d), 1.0f);
  float* dev = upload<float>(f);
  float* dev_rinv = kops_test::device_alloc<float>(static_cast<size_t>(3 * rows));
  uint8_t* dev_q = kops_test::device_alloc<uint8_t>(static_cast<size_t>(rows * d));
  float* dev_scales = kops_test::device_alloc<float>(static_cast<size_t>(rows * (d / 32)));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || dev == nullptr || dev_rinv == nullptr ||
      dev_q == nullptr || dev_scales == nullptr) {
    return;
  }
  kops::model_ops::AttnResParams ok{};
  ok.sources = dev;
  ok.source_stride = rows * d;
  ok.source_token_stride = d;
  ok.num_sources = 1;
  ok.partial = dev;
  ok.partial_stride = d;
  ok.include_partial = true;
  ok.delta = dev;
  ok.delta_stride = d;
  ok.query = dev;
  ok.key_eps = 1e-5f;
  ok.out = dev;
  ok.out_stride = d;
  ok.num_tokens = static_cast<int32_t>(rows);
  ok.d = static_cast<int32_t>(d);
  KOPS_EXPECT_OK(kops::model_ops::attn_res(ok, DType::kFloat32, ctx), "attn_res contract ok");

  {
    kops::model_ops::AttnResParams p = ok;
    p.num_sources = 9;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kUnsupportedShape, "attn_res num_sources");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.sources = nullptr;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kInvalidArgument, "attn_res null sources");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.partial = nullptr;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kInvalidArgument, "attn_res partial required by delta");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.delta = nullptr;
    p.partial = nullptr;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kInvalidArgument, "attn_res partial required as source");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.query = nullptr;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kInvalidArgument, "attn_res null query");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.out = nullptr;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kInvalidArgument, "attn_res no output");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.out = nullptr;
    p.normed_out = dev;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kInvalidArgument, "attn_res normed without weight");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.normed_quant.mode = kops::ActQuantMode::kMXFP8;
    p.normed_quant.q = dev_q;
    p.normed_quant.scales = dev_scales;
    p.normed_quant.q_dtype = DType::kFloat8E4M3;
    p.normed_quant.q_row_stride = d;
    p.normed_quant.scale_row_stride = d / 32;
    p.out_norm_weight = dev;
    p.d = 48;   // 48 % 32 != 0
    p.out_stride = 48;
    p.partial_stride = 48;
    p.delta_stride = 48;
    p.source_stride = rows * 48;
    p.source_token_stride = 48;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kUnsupportedShape, "attn_res MX needs d % 32");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.normed_quant.mode = kops::ActQuantMode::kDynamicPerToken;
    p.out_norm_weight = dev;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kUnsupportedShape, "attn_res quant mode");
  }
  {
    kops::model_ops::AttnResParams p = ok;
    p.out_stride = d - 1;
    p.normed_out = dev;
    KOPS_EXPECT_STATUS(kops::model_ops::attn_res(p, DType::kFloat32, ctx),
                       kops::Status::kUnsupportedLayout, "attn_res out_stride");
  }
  KOPS_EXPECT_STATUS(kops::model_ops::attn_res(ok, DType::kInt8, ctx),
                     kops::Status::kUnsupportedDtype, "attn_res dtype");

  // commit: slot range, dense layout, eps, rinv shape.
  const int32_t slots = 3;
  float* dev_stack = kops_test::device_alloc<float>(static_cast<size_t>(slots * rows * d));
  if (dev_stack == nullptr) return;
  kops_test::copy_in(dev_stack, std::vector<float>(static_cast<size_t>(slots * rows * d), 1.0f));
  const kops::Tensor ts = kops_test::tensor3d(dev_stack, DType::kFloat32, slots, rows, d);
  const kops::Tensor trinv = kops_test::tensor2d(dev_rinv, DType::kFloat32, slots, rows);
  const kops::Tensor tp = kops_test::tensor2d(dev, DType::kFloat32, rows, d);
  KOPS_EXPECT_OK(kops::model_ops::attn_res_commit(ts, trinv, tp, kops::Tensor{}, 0, 1e-5f, ctx),
                 "commit contract ok");
  KOPS_EXPECT_STATUS(
      kops::model_ops::attn_res_commit(ts, trinv, tp, kops::Tensor{}, slots, 1e-5f, ctx),
      kops::Status::kInvalidArgument, "commit slot range");
  KOPS_EXPECT_STATUS(
      kops::model_ops::attn_res_commit(ts, trinv, tp, kops::Tensor{}, -1, 1e-5f, ctx),
      kops::Status::kInvalidArgument, "commit slot negative");
  KOPS_EXPECT_STATUS(
      kops::model_ops::attn_res_commit(ts, trinv, tp, kops::Tensor{}, 0, -1.0f, ctx),
      kops::Status::kInvalidArgument, "commit eps");
  {
    const kops::Tensor bad = view2d(dev, DType::kFloat32, rows, d, d + 4);
    KOPS_EXPECT_STATUS(
        kops::model_ops::attn_res_commit(ts, trinv, bad, kops::Tensor{}, 0, 1e-5f, ctx),
        kops::Status::kUnsupportedLayout, "commit dense partial");
  }
  {
    const kops::Tensor bad = kops_test::tensor2d(dev_rinv, DType::kFloat32, slots, rows - 1);
    KOPS_EXPECT_STATUS(
        kops::model_ops::attn_res_commit(ts, bad, tp, kops::Tensor{}, 0, 1e-5f, ctx),
        kops::Status::kUnsupportedLayout, "commit rinv shape");
  }

  kops_test::device_free(dev);
  kops_test::device_free(dev_rinv);
  kops_test::device_free(dev_q);
  kops_test::device_free(dev_scales);
  kops_test::device_free(dev_stack);
}

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  std::vector<float> xf(64 * 16, 1.0f);
  float* d = kops_test::device_alloc<float>(xf.size());
  kops_test::copy_in(d, xf);
  kops::Tensor a = kops_test::tensor2d(d, DType::kFloat32, 64, 16);
  kops::Tensor b = kops_test::tensor2d(d, DType::kFloat32, 64, 15);
  kops::Tensor out = kops_test::tensor2d(d, DType::kFloat32, 64, 16);

  KOPS_EXPECT_STATUS(kops::model_ops::residual_add(out, a, 1.0f, b, 1.0f, kops::Tensor{}, 0.0f, ctx),
                     kops::Status::kInvalidArgument, "residual_add shape");
  kops::Tensor h = kops_test::tensor2d(d, DType::kFloat16, 64, 16);
  KOPS_EXPECT_STATUS(kops::model_ops::residual_add(out, a, 1.0f, h, 1.0f, kops::Tensor{}, 0.0f, ctx),
                     kops::Status::kUnsupportedDtype, "residual_add dtype");
  kops::Tensor mis = kops_test::tensor2d(reinterpret_cast<char*>(d) + 1, DType::kFloat32, 8, 16);
  KOPS_EXPECT_STATUS(kops::model_ops::cast(out, mis, 1.0f, ctx), kops::Status::kMisaligned,
                     "cast alignment");
  kops_test::device_free(d);
}

void test_graph() {
  const int64_t rows = 8, d = 64;
  const size_t n = static_cast<size_t>(rows * d);
  std::vector<float> xf(n, 1.0f), yf(n, 2.0f);
  float* x = kops_test::device_alloc<float>(n);
  float* y = kops_test::device_alloc<float>(n);
  float* z = kops_test::device_alloc<float>(n);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || x == nullptr || y == nullptr || z == nullptr) return;
  kops_test::copy_in(x, xf);
  kops_test::copy_in(y, yf);
  kops::Tensor tx = kops_test::tensor2d(x, DType::kFloat32, rows, d);
  kops::Tensor ty = kops_test::tensor2d(y, DType::kFloat32, rows, d);
  kops::Tensor tz = kops_test::tensor2d(z, DType::kFloat32, rows, d);
  cudaStream_t stream = nullptr;
  KOPS_TEST_CHECK(cudaStreamCreate(&stream) == cudaSuccess);
  kops::LaunchContext gctx = ctx;
  gctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  KOPS_TEST_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  KOPS_EXPECT_OK(kops::model_ops::residual_add(tz, tx, 1.0f, ty, 1.0f, kops::Tensor{}, 0.0f, gctx),
                 "graph residual");
  KOPS_EXPECT_OK(kops::model_ops::cast(tz, tz, 0.5f, gctx), "graph cast");
  cudaGraph_t graph = nullptr;
  KOPS_TEST_CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  KOPS_TEST_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
  std::vector<float> xf2(n, 3.0f), yf2(n, 4.0f);
  kops_test::copy_in(x, xf2);
  kops_test::copy_in(y, yf2);
  KOPS_TEST_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
  KOPS_TEST_CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  const auto got = kops_test::copy_out(z, n);
  std::vector<double> ref(n, 3.5);  // (3 + 4) * 0.5
  kops_test::compare(got, ref, 1e-5, 1e-5, "graph replay");
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  kops_test::device_free(x);
  kops_test::device_free(y);
  kops_test::device_free(z);
}

void test_registry() {
  const kops::registry::Capability* rows[4] = {};
  int32_t n = 0;
  KOPS_EXPECT_OK(kops::registry::find_capabilities("X02.residual_add", 860, rows, 4, &n),
                 "registry find");
  KOPS_TEST_CHECK(n >= 1);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  // The capability table is per exact SM. select() can only answer where a row
  // for this arch_id exists, so only assert it on a machine the table covers
  // (an exact-SM row outside the table is a registry gap, not a kernel bug).
  int32_t here = 0;
  KOPS_EXPECT_OK(
      kops::registry::find_capabilities("X02.residual_add", ctx.arch_id, rows, 4, &here),
      "registry find (this arch)");
  if (here == 0) return;
  const kops::registry::Capability* chosen = nullptr;
  KOPS_EXPECT_OK(kops::registry::select("X02.residual_add", kops::registry::SpecKind::kNone,
                                        nullptr, ctx, kops::registry::Provider::kNative, true,
                                        &chosen),
                 "registry select");
  KOPS_TEST_CHECK(chosen != nullptr);
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;
  test_residual<float>(DType::kFloat32, "residual_add fp32 3-input", true);
  test_residual<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 2-input", false);
  test_cast<float, __nv_bfloat16>(DType::kFloat32, DType::kBFloat16, "cast bf16->fp32");
  test_cast<__nv_bfloat16, float>(DType::kBFloat16, DType::kFloat32, "cast fp32->bf16");
  test_cast<float, __half>(DType::kFloat32, DType::kFloat16, "cast fp16->fp32");

  // ---- layout paths, mixed dtypes and odd tails ----------------------------
  test_residual_layouts<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 layouts", 9, 4096,
                                       false);
  test_residual_layouts<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 3-in layouts", 7, 512,
                                       true);
  test_residual_layouts<__half>(DType::kFloat16, "residual_add fp16 layouts", 6, 1024, false);
  test_residual_layouts<float>(DType::kFloat32, "residual_add fp32 layouts", 5, 1000, true);
  // bf16 input with an fp32 output (mixed chunk granule).
  test_cast_layouts<float, __nv_bfloat16>(DType::kFloat32, DType::kBFloat16,
                                          "cast bf16->fp32 layouts", 6, 1000, 1.5f);
  test_cast_layouts<__nv_bfloat16, float>(DType::kBFloat16, DType::kFloat32,
                                          "cast fp32->bf16 layouts", 6, 1000, 0.5f);
  test_cast_layouts<__nv_bfloat16, __half>(DType::kBFloat16, DType::kFloat16,
                                           "cast fp16->bf16 layouts", 5, 4096, 1.0f);
  test_cast_layouts<__half, __nv_bfloat16>(DType::kFloat16, DType::kBFloat16,
                                           "cast bf16->fp16 layouts", 5, 4096, 2.0f);
  test_residual_odd<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 h1", 1);
  test_residual_odd<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 h251", 251);
  test_residual_odd<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 h4095", 4095);
  test_residual_odd<float>(DType::kFloat32, "residual_add fp32 h250", 250);
  test_residual_odd<__half>(DType::kFloat16, "residual_add fp16 h4097", 4097);

  // ---- B15: attention residuals (AttnRes) ---------------------------------
  test_attn_res_basic<float>(DType::kFloat32, "attn_res fp32 sources+partial", 256, 8);
  test_attn_res_basic<__nv_bfloat16>(DType::kBFloat16, "attn_res bf16 sources+partial", 256, 8);
  test_attn_res_basic<__nv_bfloat16>(DType::kBFloat16, "attn_res bf16 scalar path", 100, 0);
  test_attn_res_basic<__half>(DType::kFloat16, "attn_res fp16 sources+partial", 256, 0);
  test_attn_res_commit_boundary();
  test_attn_res_norm_quant();
  test_attn_res_batch_invariance();
  test_attn_res_k3();
  test_attn_res_invalid_sources();
  test_attn_res_contracts();

  test_contracts();
  test_graph();
  test_registry();
  return kops_test::finish("test_model_ops");
}
