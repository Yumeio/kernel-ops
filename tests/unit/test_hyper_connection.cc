// =============================================================================
// kernel-ops | tests/unit/test_hyper_connection.cc
// B15 DeepSeek-V4 mHC (X01): hc_expand, hc_head, hc_post, hc_coefficients,
// hc_pre and the fused mega_mhc generic body.
//
// Oracle: a local fp64 transcription of
//   3rd/DeepGEMM/tests/test_mega_mhc.py::{mhc_reference, _sinkhorn_reference}
// with the repo's dtype rounding (bf16/fp16/fp32 stream) and the DeepGEMM
// `per_token_cast_to_fp8(gran_k=32, packed UE8M0)` recipe for the FP8 output.
// Covers n_hc = 2/4, H = 64/128/256, T = 1/3/17, shifted mode, Sinkhorn
// iterations 1/3/20, HCConstraint::kNone, raw_logits (kFromLogits), FP8 payload +
// both scale layouts, the 48 KB row-stage fallback, a H=7168 smoke, chain vs
// mega_mhc bitwise equality, determinism, and the Status contract.
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/device.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/ops.h"

namespace {

using kops::ActQuantMode;
using kops::ActQuantOut;
using kops::DType;
using kops::model_ops::HCConstraint;
using kops::model_ops::HCParams;
using kops::model_ops::HCStreamParams;
using kops::LaunchContext;
using kops::ScaleLayout;
using kops::Status;
using kops::Tensor;

// -----------------------------------------------------------------------------
// Host reference helpers (fp64; dtype rounding where the reference rounds)
// -----------------------------------------------------------------------------

float round_dtype(DType dt, float v) {
  if (dt == DType::kBFloat16) {
    __nv_bfloat16 b;
    kops_test::from_float(b, v);
    return kops_test::to_float(b);
  }
  if (dt == DType::kFloat16) {
    __half h;
    kops_test::from_float(h, v);
    return kops_test::to_float(h);
  }
  return v;
}

uint8_t e4m3_bits(float v) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

float e4m3_value(uint8_t b) { return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)); }

/// DeepGEMM `ceil_to_ue8m0`: biased exponent, +1 when the mantissa is non-zero.
uint8_t ue8m0_ceil(double s) {
  const float f = static_cast<float>(s);
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t e = ((bits >> 23) & 0xFFu) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
  e = e < 1u ? 1u : (e > 254u ? 254u : e);
  return static_cast<uint8_t>(e);
}

double ue8m0_value(uint8_t e) { return std::ldexp(1.0, static_cast<int>(e) - 127); }

struct Problem {
  int64_t T = 0, H = 0;
  int n = 0;
  bool shifted = false;
  HCConstraint constraint = HCConstraint::kManifold;
  int32_t iters = 20;
  float hc_norm_eps = 2e-5f;
  float hc_pre_eps = 3e-4f;
  float hc_post_scale = 1.25f;
  float sinkhorn_eps = 2e-6f;
  float rmsnorm_eps = 7e-6f;
  float rmsnorm_scale = 1.25f;
  DType stream_dtype = DType::kBFloat16;
  DType out_dtype = DType::kBFloat16;
  std::vector<float> x;         // [T, H]
  std::vector<float> residual;  // [T, n, H]
  std::vector<float> post;      // [T, n]
  std::vector<float> comb;      // [T, n, n]
  std::vector<float> shift;     // [T, n] or empty
  std::vector<float> fn;        // [n(n+2), n*H]
  std::vector<float> scales;    // [3]
  std::vector<float> bases;     // [n(n+2)]
  std::vector<float> w;         // [H] (already rounded to the norm dtype)
};

Problem make_problem(int64_t T, int64_t H, int n, DType stream_dtype, DType out_dtype,
                     bool shifted, HCConstraint constraint, int iters, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
  const int N = n * (n + 2);

  Problem p;
  p.T = T;
  p.H = H;
  p.n = n;
  p.shifted = shifted;
  p.constraint = constraint;
  p.iters = iters;
  p.stream_dtype = stream_dtype;
  p.out_dtype = out_dtype;

  p.x.resize(static_cast<size_t>(T * H));
  p.residual.resize(static_cast<size_t>(T * n * H));
  p.post.resize(static_cast<size_t>(T * n));
  p.comb.resize(static_cast<size_t>(T * n * n));
  p.shift.assign(shifted ? static_cast<size_t>(T * n) : 0u, 0.0f);
  p.fn.resize(static_cast<size_t>(N) * n * H);
  p.scales.resize(3);
  p.bases.resize(N);
  p.w.resize(static_cast<size_t>(H));
  // x and residual are stream-dtype tensors in the model; round them here so the
  // fp64 reference and the kernel see exactly the same inputs.
  for (float& v : p.x) v = round_dtype(stream_dtype, nd(rng));
  for (float& v : p.residual) v = round_dtype(stream_dtype, nd(rng));
  for (float& v : p.post) v = sig(nd(rng));
  for (float& v : p.comb) v = ud(rng);
  for (float& v : p.shift) v = sig(nd(rng));
  for (float& v : p.fn) v = nd(rng) * 1e-2f;
  for (float& v : p.scales) v = nd(rng) * 0.1f;
  for (float& v : p.bases) v = nd(rng) * 0.1f;
  for (float& v : p.w) v = round_dtype(DType::kBFloat16, 1.0f + 0.1f * nd(rng));

  // Three row/column normalizations, like the official generator.
  for (int round = 0; round < 3; ++round) {
    for (int64_t t = 0; t < T; ++t) {
      for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += p.comb[(t * n + i) * n + j];
        for (int j = 0; j < n; ++j) p.comb[(t * n + i) * n + j] /= static_cast<float>(s);
      }
      for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += p.comb[(t * n + i) * n + j];
        for (int i = 0; i < n; ++i) p.comb[(t * n + i) * n + j] /= static_cast<float>(s);
      }
    }
  }
  return p;
}

struct RefOutputs {
  std::vector<float> new_residual;  // rounded to stream_dtype
  std::vector<float> y;             // rounded to out_dtype
  std::vector<double> coeff_a;      // [T, n]
  std::vector<double> coeff_b;      // [T, n, n]
  std::vector<double> coeff_c;      // [T, n]
  std::vector<double> sumsq;        // [T]
  std::vector<double> dot;          // [T, n(n+2)] projection before rms scaling
};

void sinkhorn_ref(std::vector<double>& v, int n, int iters, double eps) {
  // v is row-major [n, n]; softmax over dim -1, +eps, then alternate.
  for (int i = 0; i < n; ++i) {
    double m = v[i * n];
    for (int j = 1; j < n; ++j) m = std::max(m, v[i * n + j]);
    double s = 0.0;
    for (int j = 0; j < n; ++j) {
      v[i * n + j] = std::exp(v[i * n + j] - m);
      s += v[i * n + j];
    }
    for (int j = 0; j < n; ++j) v[i * n + j] = v[i * n + j] / s + eps;
  }
  for (int j = 0; j < n; ++j) {
    double c = 0.0;
    for (int i = 0; i < n; ++i) c += v[i * n + j];
    for (int i = 0; i < n; ++i) v[i * n + j] /= (c + eps);
  }
  for (int it = 1; it < iters; ++it) {
    for (int i = 0; i < n; ++i) {
      double r = 0.0;
      for (int j = 0; j < n; ++j) r += v[i * n + j];
      for (int j = 0; j < n; ++j) v[i * n + j] /= (r + eps);
    }
    for (int j = 0; j < n; ++j) {
      double c = 0.0;
      for (int i = 0; i < n; ++i) c += v[i * n + j];
      for (int i = 0; i < n; ++i) v[i * n + j] /= (c + eps);
    }
  }
}

RefOutputs reference(const Problem& p) {
  const int64_t T = p.T, H = p.H;
  const int n = p.n;
  const int N = n * (n + 2);
  const int64_t K = static_cast<int64_t>(n) * H;
  RefOutputs r;
  r.new_residual.assign(static_cast<size_t>(T * n * H), 0.0f);
  r.y.assign(static_cast<size_t>(T * H), 0.0f);
  r.coeff_a.assign(static_cast<size_t>(T * n), 0.0);
  r.coeff_b.assign(static_cast<size_t>(T * n * n), 0.0);
  r.coeff_c.assign(static_cast<size_t>(T * n), 0.0);
  r.sumsq.assign(static_cast<size_t>(T), 0.0);
  r.dot.assign(static_cast<size_t>(T * N), 0.0);

  std::vector<double> pre(n), post(n), comb(static_cast<size_t>(n * n));
  std::vector<double> mix(N), nr(static_cast<size_t>(n * H)), ni(static_cast<size_t>(H));

  for (int64_t t = 0; t < T; ++t) {
    double sq = 0.0;
    for (int j = 0; j < n; ++j) {
      for (int64_t h = 0; h < H; ++h) {
        double acc = static_cast<double>(p.x[t * H + h]) * static_cast<double>(p.post[t * n + j]);
        for (int i = 0; i < n; ++i) {
          acc += static_cast<double>(p.comb[(t * n + i) * n + j]) *
                 static_cast<double>(p.residual[(t * n + i) * H + h]);
        }
        const float v = round_dtype(p.stream_dtype, static_cast<float>(acc));
        nr[static_cast<size_t>(j) * H + h] = v;
        r.new_residual[static_cast<size_t>((t * n + j) * H + h)] = v;
        sq += static_cast<double>(v) * static_cast<double>(v);
      }
    }
    r.sumsq[t] = sq;
    const double rms = 1.0 / std::sqrt(sq / static_cast<double>(K) + p.hc_norm_eps);

    for (int o = 0; o < N; ++o) {
      double d = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        d += static_cast<double>(nr[static_cast<size_t>(k)]) *
             static_cast<double>(p.fn[static_cast<size_t>(o) * K + k]);
      }
      r.dot[t * N + o] = d;
      const double sc = (o < n) ? p.scales[0] : (o < 2 * n ? p.scales[1] : p.scales[2]);
      mix[o] = d * rms * sc + p.bases[o];
    }

    if (p.constraint == HCConstraint::kManifold) {
      for (int i = 0; i < n; ++i) {
        pre[i] = 1.0 / (1.0 + std::exp(-mix[i])) + p.hc_pre_eps;
        post[i] = (1.0 / (1.0 + std::exp(-mix[n + i]))) * p.hc_post_scale;
        for (int j = 0; j < n; ++j) comb[i * n + j] = mix[2 * n + i * n + j];
      }
      sinkhorn_ref(comb, n, p.iters, p.sinkhorn_eps);
    } else {
      for (int i = 0; i < n; ++i) {
        pre[i] = mix[i];
        post[i] = mix[n + i];
        for (int j = 0; j < n; ++j) comb[i * n + j] = mix[2 * n + i * n + j];
      }
    }
    for (int i = 0; i < n; ++i) {
      r.coeff_a[t * n + i] = pre[i];
      r.coeff_c[t * n + i] = post[i];
      for (int j = 0; j < n; ++j) r.coeff_b[(t * n + i) * n + j] = comb[i * n + j];
    }

    for (int64_t h = 0; h < H; ++h) {
      double acc = 0.0;
      if (p.shifted) {
        for (int i = 0; i < n; ++i) {
          acc += static_cast<double>(p.shift[t * n + i]) *
                 static_cast<double>(nr[static_cast<size_t>(i) * H + h]);
        }
      } else {
        for (int i = 0; i < n; ++i) {
          acc += pre[i] * static_cast<double>(nr[static_cast<size_t>(i) * H + h]);
        }
      }
      ni[h] = static_cast<double>(round_dtype(p.stream_dtype, static_cast<float>(acc)));
    }
    double ysq = 0.0;
    for (int64_t h = 0; h < H; ++h) ysq += ni[h] * ni[h];
    const double ys = 1.0 / std::sqrt(ysq / static_cast<double>(H) + p.rmsnorm_eps) *
                      static_cast<double>(p.rmsnorm_scale);
    for (int64_t h = 0; h < H; ++h) {
      r.y[t * H + h] =
          round_dtype(p.out_dtype, static_cast<float>(ni[h] * ys * static_cast<double>(p.w[h])));
    }
  }
  return r;
}

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

struct Arena {
  std::vector<void*> owned;

  Arena() = default;
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  ~Arena() {
    for (void* p : owned) {
      if (p != nullptr) cudaFree(p);
    }
  }

  template <typename T>
  T* alloc(size_t count) {
    T* p = nullptr;
    const cudaError_t err =
        cudaMalloc(reinterpret_cast<void**>(&p), (count == 0 ? 1 : count) * sizeof(T));
    if (err != cudaSuccess) {
      kops_test::record_failure("cudaMalloc", cudaGetErrorString(err), __FILE__, __LINE__);
      return nullptr;
    }
    owned.push_back(p);
    return p;
  }

  template <typename T>
  T* upload(const std::vector<T>& src) {
    T* p = alloc<T>(src.size());
    if (p != nullptr && !src.empty()) kops_test::copy_in(p, src);
    return p;
  }
};

template <typename T>
T* upload_from_float(Arena* a, const std::vector<float>& src) {
  std::vector<T> conv(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(conv[i], src[i]);
  return a->upload(conv);
}

template <typename T>
DType dtype_of_stream();

template <>
DType dtype_of_stream<float>() {
  return DType::kFloat32;
}
template <>
DType dtype_of_stream<__half>() {
  return DType::kFloat16;
}
template <>
DType dtype_of_stream<__nv_bfloat16>() {
  return DType::kBFloat16;
}

Tensor tensor1d(void* data, DType dtype, int64_t n) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

template <typename T>
void expect_bits_equal(const T* got, const T* want, size_t count, const char* what) {
  const std::vector<T> hg = kops_test::copy_out(got, count);
  const std::vector<T> hw = kops_test::copy_out(want, count);
  size_t bad = 0;
  size_t first = count;
  for (size_t i = 0; i < count; ++i) {
    if (std::memcmp(&hg[i], &hw[i], sizeof(T)) != 0) {
      ++bad;
      if (first == count) first = i;
    }
  }
  if (bad != 0) {
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu/%zu values differ, first index %zu", bad, count,
                  first);
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
}

template <typename T>
void expect_bytes_equal(const T* got, const T* want, size_t count, const char* what) {
  expect_bits_equal(got, want, count, what);
}

template <typename T>
void compare_output(const T* got, size_t count, const std::vector<double>& want, double atol,
                    double rtol, const char* what) {
  kops_test::compare(kops_test::copy_out(got, count), want, atol, rtol, what);
}

/// Rounding-aware comparison: `ulps` steps of the stored dtype. The device body
/// and the fp64 oracle agree to a rounding step on a handful of elements (fp32
/// accumulation vs fp64 before the one store rounding); everything larger is a
/// bug, so this is much tighter than a blanket relative tolerance.
template <typename T>
void compare_rounded(const T* got, size_t count, const std::vector<float>& want, DType dt,
                     double ulps, const char* what) {
  const std::vector<T> hg = kops_test::copy_out(got, count);
  const double eps = dt == DType::kBFloat16 ? 1.0 / 256.0
                     : dt == DType::kFloat16 ? 1.0 / 2048.0
                                             : 1.0 / 8388608.0;
  // A fp32 store keeps the full accumulation error instead of hiding it behind a
  // bf16/fp16 rounding: the oracle is fp64 and the body fp32/FMA, so a few ULPs
  // are expected. bf16/fp16 rows only need the one store-rounding step.
  if (dt == DType::kFloat32) ulps = 32.0;
  size_t bad = 0;
  size_t first = count;
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double g = static_cast<double>(kops_test::to_float(hg[i]));
    const double w = static_cast<double>(want[i]);
    if (!std::isfinite(g) && !std::isfinite(w)) continue;
    // The bound scales with the operand magnitude (1 + |value|), not the result:
    // a residual mix can cancel to near zero while its fp32 terms are O(1), and
    // a pure relative bound would call that cancellation a bug.
    const double tol = ulps * eps * (1.0 + std::max(std::fabs(g), std::fabs(w)));
    const double err = std::fabs(g - w);
    if (err > tol) {
      ++bad;
      if (err > worst) {
        worst = err;
        first = i;
      }
    }
  }
  if (bad != 0) {
    char detail[192];
    std::snprintf(detail, sizeof(detail),
                  "%zu/%zu beyond %.1f %s ULP, worst index %zu: got %.7g want %.7g", bad, count,
                  ulps, dt == DType::kBFloat16 ? "bf16" : (dt == DType::kFloat16 ? "fp16" : "fp32"),
                  first, static_cast<double>(kops_test::to_float(hg[first])), want[first]);
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
}

// -----------------------------------------------------------------------------
// One full chain + mega case
// -----------------------------------------------------------------------------

struct Fp8Ref {
  std::vector<uint8_t> payload;  // [T*H]
  std::vector<uint8_t> exp;      // [T * H/32] UE8M0 exponent
};

/// per_token_cast_to_fp8(y, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
Fp8Ref quantize_fp8_ref(const std::vector<float>& y_rounded, int64_t T, int64_t H) {
  Fp8Ref r;
  r.payload.assign(static_cast<size_t>(T * H), 0);
  r.exp.assign(static_cast<size_t>(T * H / 32), 0);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t g = 0; g < H / 32; ++g) {
      double amax = 0.0;
      for (int64_t j = 0; j < 32; ++j) {
        amax = std::max(amax, std::fabs(static_cast<double>(y_rounded[t * H + g * 32 + j])));
      }
      const double s = std::max(amax, 1e-4) / 448.0;
      const uint8_t e = ue8m0_ceil(s);
      const double sf = ue8m0_value(e);
      r.exp[static_cast<size_t>(t * H / 32 + g)] = e;
      for (int64_t j = 0; j < 32; ++j) {
        r.payload[static_cast<size_t>(t * H + g * 32 + j)] =
            e4m3_bits(static_cast<float>(static_cast<double>(y_rounded[t * H + g * 32 + j]) / sf));
      }
    }
  }
  return r;
}

/// Fill an HCParams for one case. Null pointers select the optional outputs.
void fill_params(HCParams* p, const Problem& pb, const Arena& arena, const void* x, const void* res,
                 void* nres, float* post, float* comb, float* shift, float* fn, float* scales,
                 float* bases, void* w, float* ca, float* cb, float* cc, float* raw, float* sumsq,
                 void* npre, void* npost, void* ncomb) {
  const int64_t H = pb.H;
  const int n = pb.n;
  *p = HCParams{};
  p->x = x;
  p->x_stride = H;
  p->x_dtype = pb.stream_dtype;
  p->residual = const_cast<void*>(res);
  p->residual_token_stride = static_cast<int64_t>(n) * H;
  p->residual_route_stride = H;
  p->residual_dtype = pb.stream_dtype;
  p->new_residual = nres;
  p->new_residual_token_stride = static_cast<int64_t>(n) * H;
  p->new_residual_route_stride = H;
  p->post_mix = post;
  p->comb_res_mix = comb;
  p->shifted_prev_mix = shift;
  p->fn = fn;
  p->mix_scales = scales;
  p->mix_bases = bases;
  p->hc_norm_eps = pb.hc_norm_eps;
  p->hc_pre_eps = pb.hc_pre_eps;
  p->hc_post_scale = pb.hc_post_scale;
  p->sinkhorn_eps = pb.sinkhorn_eps;
  p->num_sinkhorn_iters = pb.iters;
  p->constraint = pb.constraint;
  p->rmsnorm_weight = w;
  p->rmsnorm_dtype = DType::kBFloat16;
  p->rmsnorm_eps = pb.rmsnorm_eps;
  p->rmsnorm_scale = pb.rmsnorm_scale;
  p->coeff_a = ca;
  p->coeff_b = cb;
  p->coeff_c = cc;
  p->raw_logits = raw;
  p->row_sumsq = sumsq;
  p->new_prev_mix = npre;
  p->new_post_mix = npost;
  p->new_comb_res_mix = ncomb;
  p->num_tokens = static_cast<int32_t>(pb.T);
  p->n_hc = n;
  p->d = static_cast<int32_t>(H);
  (void)arena;
}

/// Run the unfused chain and compare every output against the fp64 reference.
/// `with_scratch` hands hc_coefficients the raw_logits + row_sumsq pair, which selects the
/// m16 tensor-core projection body for T >= 16 (the generic per-token body otherwise); the
/// raw dots and row sums are then checked against the oracle as well.
template <typename stream_t>
void run_chain(const Problem& pb, const char* label, LaunchContext* ctx, bool with_shift_outputs,
               bool run_raw_logits, bool with_scratch = false) {
  const int64_t T = pb.T, H = pb.H;
  const int n = pb.n;
  const int N = n * (n + 2);
  const DType sdt = dtype_of_stream<stream_t>();

  Arena arena;
  stream_t* d_x = upload_from_float<stream_t>(&arena, pb.x);
  stream_t* d_res = upload_from_float<stream_t>(&arena, pb.residual);
  stream_t* d_nres = arena.alloc<stream_t>(static_cast<size_t>(T * n * H));
  float* d_post = arena.upload(pb.post);
  float* d_comb = arena.upload(pb.comb);
  float* d_shift = pb.shifted ? arena.upload(pb.shift) : nullptr;
  float* d_fn = arena.upload(pb.fn);
  float* d_scales = arena.upload(pb.scales);
  float* d_bases = arena.upload(pb.bases);
  __nv_bfloat16* d_w = upload_from_float<__nv_bfloat16>(&arena, pb.w);
  float* d_ca = arena.alloc<float>(static_cast<size_t>(T * n));
  float* d_cb = arena.alloc<float>(static_cast<size_t>(T * n * n));
  float* d_cc = arena.alloc<float>(static_cast<size_t>(T * n));
  float* d_sumsq = arena.alloc<float>(static_cast<size_t>(T));
  float* d_raw = with_scratch ? arena.alloc<float>(static_cast<size_t>(T * N)) : nullptr;
  float* d_npre = with_shift_outputs ? arena.alloc<float>(static_cast<size_t>(T * n)) : nullptr;
  float* d_npost = arena.alloc<float>(static_cast<size_t>(T * n));
  float* d_ncomb = arena.alloc<float>(static_cast<size_t>(T * n * n));
  stream_t* d_y = arena.alloc<stream_t>(static_cast<size_t>(T * H));

  HCParams p{};
  fill_params(&p, pb, arena, d_x, d_res, d_nres, d_post, d_comb, d_shift, d_fn, d_scales, d_bases,
              d_w, d_ca, d_cb, d_cc, d_raw, d_sumsq, d_npre, d_npost, d_ncomb);

  const Tensor ty = kops_test::tensor2d(d_x, sdt, T, H);
  KOPS_EXPECT_OK(kops::model_ops::hc_post(p, ty, *ctx), label);
  KOPS_EXPECT_OK(kops::model_ops::hc_coefficients(p, *ctx), label);
  const Tensor tl = kops_test::tensor2d(d_y, pb.out_dtype, T, H);
  KOPS_EXPECT_OK(
      kops::model_ops::hc_pre(p, tl, ActQuantOut{}, Tensor{}, 0.0f, *ctx), label);

  const RefOutputs ref = reference(pb);
  // The m16 tensor-core body (raw_logits + row_sumsq scratch pair, T >= kTileMinTokens)
  // splits `fn` into hi/lo tf32 halves, so every element is fp32-accurate and one mma's k8 dot
  // is a single fp32 rounding folded into an RNE fp32 sum. Measured against the fp64 oracle at
  // K = n*H = 28672 the raw dots stay within ~4e-4 (typically ~2e-6) and the coefficients
  // within ~4e-5, i.e. the TF32-row tolerance of docs/MODEL_COVERAGE.md and ~2x tighter than a
  // plain TF32 GEMM (~8e-4 dots). Every other body is compared at the fp32 tolerance.
  const bool tile = with_scratch && T >= 256 && H % 8 == 0 && (n * (n + 2)) % 8 == 0;
  const double kCoeffTol = tile ? 1e-4 : 1e-6;
  const double kRawTol = tile ? 6e-4 : 1e-5;
  compare_rounded(d_nres, static_cast<size_t>(T * n * H), ref.new_residual, pb.stream_dtype, 2.0,
                  (std::string(label) + ".new_residual").c_str());
  compare_output(d_ca, static_cast<size_t>(T * n), ref.coeff_a, kCoeffTol, kCoeffTol,
                 (std::string(label) + ".coeff_a").c_str());
  compare_output(d_cb, static_cast<size_t>(T * n * n), ref.coeff_b, kCoeffTol, kCoeffTol,
                 (std::string(label) + ".coeff_b").c_str());
  compare_output(d_cc, static_cast<size_t>(T * n), ref.coeff_c, kCoeffTol, kCoeffTol,
                 (std::string(label) + ".coeff_c").c_str());
  compare_output(d_sumsq, static_cast<size_t>(T), ref.sumsq, 1e-3, 2e-5,
                 (std::string(label) + ".row_sumsq").c_str());
  compare_rounded(d_y, static_cast<size_t>(T * H), ref.y, pb.out_dtype, 2.0,
                  (std::string(label) + ".layer_in").c_str());
  if (with_scratch) {
    compare_output(d_raw, static_cast<size_t>(T * N), ref.dot, kRawTol, kRawTol,
                   (std::string(label) + ".raw_logits").c_str());
  }
  if (with_shift_outputs) {
    compare_output(d_npre, static_cast<size_t>(T * n), ref.coeff_a, kCoeffTol, kCoeffTol,
                   (std::string(label) + ".new_prev_mix").c_str());
  }
  compare_output(d_npost, static_cast<size_t>(T * n), ref.coeff_c, kCoeffTol, kCoeffTol,
                 (std::string(label) + ".new_post_mix").c_str());
  compare_output(d_ncomb, static_cast<size_t>(T * n * n), ref.coeff_b, kCoeffTol, kCoeffTol,
                 (std::string(label) + ".new_comb_res_mix").c_str());

  // raw_logits without row_sumsq is the DeepGEMM contract: the caller ran the projection, so
  // the buffer is read and left untouched and only the constraint runs.
  if (run_raw_logits) {
    std::vector<float> raw(static_cast<size_t>(T * N));
    std::vector<float> sq(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      sq[t] = static_cast<float>(ref.sumsq[t]);
      for (int o = 0; o < N; ++o) raw[t * N + o] = static_cast<float>(ref.dot[t * N + o]);
    }
    float* d_raw2 = arena.upload(raw);
    float* d_raw2_copy = arena.upload(raw);
    float* d_ca2 = arena.alloc<float>(static_cast<size_t>(T * n));
    float* d_cb2 = arena.alloc<float>(static_cast<size_t>(T * n * n));
    float* d_cc2 = arena.alloc<float>(static_cast<size_t>(T * n));
    HCParams q = p;
    q.coeff_a = d_ca2;
    q.coeff_b = d_cb2;
    q.coeff_c = d_cc2;
    q.raw_logits = d_raw2;
    q.row_sumsq = nullptr;
    q.new_prev_mix = nullptr;
    q.new_post_mix = nullptr;
    q.new_comb_res_mix = nullptr;
    KOPS_EXPECT_OK(kops::model_ops::hc_coefficients(q, *ctx), "hc_coefficients/from_logits");
    expect_bits_equal(d_raw2, d_raw2_copy, static_cast<size_t>(T * N),
                      (std::string(label) + ".from_logits.raw_kept").c_str());
    compare_output(d_ca2, static_cast<size_t>(T * n), ref.coeff_a, 1e-6, 1e-6,
                   (std::string(label) + ".raw.coeff_a").c_str());
    compare_output(d_cb2, static_cast<size_t>(T * n * n), ref.coeff_b, 1e-6, 1e-6,
                   (std::string(label) + ".raw.coeff_b").c_str());
    compare_output(d_cc2, static_cast<size_t>(T * n), ref.coeff_c, 1e-6, 1e-6,
                   (std::string(label) + ".raw.coeff_c").c_str());
  }
}

/// Run mega_mhc on a fresh copy of the inputs and compare bitwise with the chain.
template <typename stream_t>
void run_mega(const Problem& pb, const char* label, LaunchContext* ctx) {
  const int64_t T = pb.T, H = pb.H;
  const int n = pb.n;
  const DType sdt = dtype_of_stream<stream_t>();

  Arena arena;
  stream_t* d_x = upload_from_float<stream_t>(&arena, pb.x);
  stream_t* d_res = upload_from_float<stream_t>(&arena, pb.residual);
  float* d_post = arena.upload(pb.post);
  float* d_comb = arena.upload(pb.comb);
  float* d_shift = pb.shifted ? arena.upload(pb.shift) : nullptr;
  float* d_fn = arena.upload(pb.fn);
  float* d_scales = arena.upload(pb.scales);
  float* d_bases = arena.upload(pb.bases);
  __nv_bfloat16* d_w = upload_from_float<__nv_bfloat16>(&arena, pb.w);

  auto alloc_outputs = [&](Arena* a, stream_t** nres, float** ca, float** cb, float** cc,
                           float** sumsq, stream_t** y) {
    *nres = a->alloc<stream_t>(static_cast<size_t>(T * n * H));
    *ca = a->alloc<float>(static_cast<size_t>(T * n));
    *cb = a->alloc<float>(static_cast<size_t>(T * n * n));
    *cc = a->alloc<float>(static_cast<size_t>(T * n));
    *sumsq = a->alloc<float>(static_cast<size_t>(T));
    *y = a->alloc<stream_t>(static_cast<size_t>(T * H));
  };

  // Reference side: the unfused chain. It gets the same raw_logits / row_sumsq scratch pair
  // mega_mhc uses internally, so both sides select the same projection body (m16 tile for
  // T >= 16, per-token otherwise) and the bitwise comparison covers the whole chain.
  stream_t* c_nres = nullptr;
  float *c_ca = nullptr, *c_cb = nullptr, *c_cc = nullptr, *c_sq = nullptr;
  stream_t* c_y = nullptr;
  alloc_outputs(&arena, &c_nres, &c_ca, &c_cb, &c_cc, &c_sq, &c_y);
  float* c_raw = arena.alloc<float>(static_cast<size_t>(T * n * (n + 2)));
  HCParams cp{};
  fill_params(&cp, pb, arena, d_x, d_res, c_nres, d_post, d_comb, d_shift, d_fn, d_scales, d_bases,
              d_w, c_ca, c_cb, c_cc, c_raw, c_sq, nullptr, nullptr, nullptr);
  const Tensor ty = kops_test::tensor2d(d_x, sdt, T, H);
  KOPS_EXPECT_OK(kops::model_ops::hc_post(cp, ty, *ctx), label);
  KOPS_EXPECT_OK(kops::model_ops::hc_coefficients(cp, *ctx), label);
  KOPS_EXPECT_OK(kops::model_ops::hc_pre(cp, kops_test::tensor2d(c_y, pb.out_dtype, T, H),
                                         ActQuantOut{}, Tensor{}, 0.0f, *ctx),
                 label);

  // mega_mhc side (fresh residual copy; prev coefficients are the input mixes).
  stream_t* d_res2 = upload_from_float<stream_t>(&arena, pb.residual);
  stream_t* m_nres = nullptr;
  float *m_ca = nullptr, *m_cb = nullptr, *m_cc = nullptr, *m_sq = nullptr;
  stream_t* m_y = nullptr;
  alloc_outputs(&arena, &m_nres, &m_ca, &m_cb, &m_cc, &m_sq, &m_y);
  HCParams mp{};
  fill_params(&mp, pb, arena, nullptr, d_res2, m_nres, d_post, d_comb, d_shift, d_fn, d_scales,
              d_bases, d_w, m_ca, m_cb, m_cc, nullptr, m_sq, nullptr, nullptr, nullptr);
  const Tensor tb = kops_test::tensor3d(d_comb, DType::kFloat32, T, n, n);
  const Tensor tcc = kops_test::tensor3d(d_post, DType::kFloat32, T, n, 1);
  const Tensor py = kops_test::tensor2d(d_x, sdt, T, H);
  const Tensor ml = kops_test::tensor2d(m_y, pb.out_dtype, T, H);
  KOPS_EXPECT_OK(kops::model_ops::mega_mhc(mp, tb, tcc, py, ml, ActQuantOut{}, Tensor{}, 0.0f, *ctx),
                 label);

  expect_bits_equal(m_nres, c_nres, static_cast<size_t>(T * n * H),
                    (std::string(label) + ".mega.new_residual").c_str());
  expect_bits_equal(m_ca, c_ca, static_cast<size_t>(T * n),
                    (std::string(label) + ".mega.coeff_a").c_str());
  expect_bits_equal(m_cb, c_cb, static_cast<size_t>(T * n * n),
                    (std::string(label) + ".mega.coeff_b").c_str());
  expect_bits_equal(m_cc, c_cc, static_cast<size_t>(T * n),
                    (std::string(label) + ".mega.coeff_c").c_str());
  expect_bits_equal(m_sq, c_sq, static_cast<size_t>(T),
                    (std::string(label) + ".mega.row_sumsq").c_str());
  expect_bits_equal(m_y, c_y, static_cast<size_t>(T * H),
                    (std::string(label) + ".mega.layer_in").c_str());

  // Determinism: a second launch into fresh buffers is bitwise identical.
  stream_t* d_res3 = upload_from_float<stream_t>(&arena, pb.residual);
  stream_t* m2_nres = nullptr;
  float *m2_ca = nullptr, *m2_cb = nullptr, *m2_cc = nullptr, *m2_sq = nullptr;
  stream_t* m2_y = nullptr;
  alloc_outputs(&arena, &m2_nres, &m2_ca, &m2_cb, &m2_cc, &m2_sq, &m2_y);
  HCParams mp2 = mp;
  mp2.residual = d_res3;
  mp2.new_residual = m2_nres;
  mp2.coeff_a = m2_ca;
  mp2.coeff_b = m2_cb;
  mp2.coeff_c = m2_cc;
  mp2.row_sumsq = m2_sq;
  KOPS_EXPECT_OK(kops::model_ops::mega_mhc(mp2, tb, tcc, py,
                                           kops_test::tensor2d(m2_y, pb.out_dtype, T, H),
                                           ActQuantOut{}, Tensor{}, 0.0f, *ctx),
                 label);
  expect_bits_equal(m2_nres, m_nres, static_cast<size_t>(T * n * H),
                    (std::string(label) + ".twice.new_residual").c_str());
  expect_bits_equal(m2_y, m_y, static_cast<size_t>(T * H),
                    (std::string(label) + ".twice.layer_in").c_str());
  expect_bits_equal(m2_ca, m_ca, static_cast<size_t>(T * n),
                    (std::string(label) + ".twice.coeff_a").c_str());
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

void test_expand_head(LaunchContext* ctx) {
  for (int n : {2, 4}) {
    for (int64_t H : {64, 128}) {
      for (int64_t T : {1, 3, 17}) {
        for (DType dt : {DType::kBFloat16, DType::kFloat32}) {
          Problem pb = make_problem(T, H, n, dt, dt, false, HCConstraint::kManifold, 20, 11u);
          Arena arena;
          const bool bf16 = dt == DType::kBFloat16;
          __nv_bfloat16* d_embed_bf = bf16 ? upload_from_float<__nv_bfloat16>(&arena, pb.x)
                                           : nullptr;
          float* d_embed_f = bf16 ? nullptr : arena.upload(pb.x);
          void* d_embed = bf16 ? static_cast<void*>(d_embed_bf) : static_cast<void*>(d_embed_f);
          const size_t stream_elems = static_cast<size_t>(T * n * H);
          __nv_bfloat16* d_streams_bf = bf16 ? arena.alloc<__nv_bfloat16>(stream_elems) : nullptr;
          float* d_streams_f = bf16 ? nullptr : arena.alloc<float>(stream_elems);
          void* d_streams = bf16 ? static_cast<void*>(d_streams_bf) : static_cast<void*>(d_streams_f);

          HCStreamParams sp{};
          sp.streams = d_streams;
          sp.stream_token_stride = static_cast<int64_t>(n) * H;
          sp.stream_dtype = dt;
          sp.num_tokens = static_cast<int32_t>(T);
          sp.n_hc = n;
          sp.d = static_cast<int32_t>(H);
          KOPS_EXPECT_OK(kops::model_ops::hc_expand(sp, kops_test::tensor2d(d_embed, dt, T, H), *ctx),
                         "hc_expand");
          cudaDeviceSynchronize();

          // Every stream holds the embedding row (exact; same dtype copy).
          std::vector<float> embed_host(static_cast<size_t>(T * H));
          if (bf16) {
            const auto e = kops_test::copy_out(d_embed_bf, static_cast<size_t>(T * H));
            for (size_t i = 0; i < e.size(); ++i) embed_host[i] = kops_test::to_float(e[i]);
            const auto got = kops_test::copy_out(d_streams_bf, stream_elems);
            for (size_t i = 0; i < got.size(); ++i) {
              const float g = kops_test::to_float(got[i]);
              const float w = embed_host[static_cast<size_t>((i / (static_cast<size_t>(H))) % n) *
                                                             0 +
                                         (i % static_cast<size_t>(H))];
              // route i is (token, stream, h): compare against embed[token, h]
              const size_t h = i % static_cast<size_t>(H);
              const size_t token = i / static_cast<size_t>(n * H);
              const float want_embed = embed_host[token * static_cast<size_t>(H) + h];
              if (g != want_embed) {
                kops_test::record_failure("hc_expand stream value", "mismatch", __FILE__, __LINE__);
                break;
              }
              (void)w;
            }
          } else {
            const auto e = kops_test::copy_out(d_embed_f, static_cast<size_t>(T * H));
            embed_host.assign(e.begin(), e.end());
            const auto got = kops_test::copy_out(d_streams_f, stream_elems);
            for (size_t i = 0; i < got.size(); ++i) {
              const size_t h = i % static_cast<size_t>(H);
              const size_t token = i / static_cast<size_t>(n * H);
              if (got[i] != embed_host[token * static_cast<size_t>(H) + h]) {
                kops_test::record_failure("hc_expand stream value", "mismatch", __FILE__, __LINE__);
                break;
              }
            }
          }

          // Head: uniform, static and per-token weights.
          __nv_bfloat16* d_out_bf = bf16 ? arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * H))
                                         : nullptr;
          float* d_out_f = bf16 ? nullptr : arena.alloc<float>(static_cast<size_t>(T * H));
          void* d_out = bf16 ? static_cast<void*>(d_out_bf) : static_cast<void*>(d_out_f);
          auto head_check = [&](const Tensor& hw, bool per_token,
                                const std::vector<float>& host_w) {
            KOPS_EXPECT_OK(
                kops::model_ops::hc_head(kops_test::tensor2d(d_out, dt, T, H), sp, hw, *ctx),
                "hc_head");
            cudaDeviceSynchronize();
            std::vector<float> w(static_cast<size_t>(per_token ? T * n : n));
            if (host_w.empty()) {
              for (int i = 0; i < n; ++i) w[i] = 1.0f / static_cast<float>(n);
            } else {
              w = host_w;
            }
            std::vector<double> want_out(static_cast<size_t>(T * H), 0.0);
            std::vector<double> streams_host(stream_elems, 0.0);
            if (bf16) {
              const auto hs = kops_test::copy_out(d_streams_bf, stream_elems);
              for (size_t i = 0; i < hs.size(); ++i) streams_host[i] = kops_test::to_float(hs[i]);
            } else {
              const auto hs = kops_test::copy_out(d_streams_f, stream_elems);
              for (size_t i = 0; i < hs.size(); ++i) streams_host[i] = hs[i];
            }
            for (int64_t t = 0; t < T; ++t)
              for (int64_t h = 0; h < H; ++h) {
                double acc = 0.0;
                for (int i = 0; i < n; ++i) {
                  const size_t idx = static_cast<size_t>((t * n + i) * H + h);
                  acc += static_cast<double>(w[per_token ? t * n + i : i]) * streams_host[idx];
                }
                want_out[static_cast<size_t>(t * H + h)] = acc;
              }
            std::vector<float> want_f(want_out.begin(), want_out.end());
            if (bf16) {
              compare_rounded(d_out_bf, static_cast<size_t>(T * H), want_f, dt, 2.0, "hc_head");
            } else {
              compare_rounded(d_out_f, static_cast<size_t>(T * H), want_f, dt, 2.0, "hc_head");
            }
          };
          head_check(Tensor{}, false, {});
          std::vector<float> wstatic(n, 1.0f / static_cast<float>(n + 1));
          float* d_ws = arena.upload(wstatic);
          head_check(tensor1d(d_ws, DType::kFloat32, n), false, wstatic);
          std::vector<float> wtok(static_cast<size_t>(T * n));
          for (size_t i = 0; i < wtok.size(); ++i) wtok[i] = 0.25f + 0.5f * static_cast<float>(i % 3);
          float* d_wt = arena.upload(wtok);
          head_check(kops_test::tensor2d(d_wt, DType::kFloat32, T, n), true, wtok);
        }
      }
    }
  }
}

void test_post(LaunchContext* ctx) {
  for (int n : {2, 4}) {
    for (int64_t H : {64, 128, 256}) {
      for (int64_t T : {1, 3, 17}) {
        for (DType dt : {DType::kBFloat16, DType::kFloat16, DType::kFloat32}) {
          Problem pb = make_problem(T, H, n, dt, dt, false, HCConstraint::kManifold, 20, 21u);
          const RefOutputs ref = reference(pb);
          Arena arena;
          const size_t elems = static_cast<size_t>(T * n * H);
          float* d_post = arena.upload(pb.post);
          float* d_comb = arena.upload(pb.comb);
          float* d_scales = arena.upload(pb.scales);
          float* d_bases = arena.upload(pb.bases);
          float* d_ca = arena.alloc<float>(static_cast<size_t>(T * n));
          float* d_cb = arena.alloc<float>(static_cast<size_t>(T * n * n));
          float* d_cc = arena.alloc<float>(static_cast<size_t>(T * n));
          HCParams p{};

          auto run = [&](auto tag) {
            using stream_t = decltype(tag);
            stream_t* d_x = upload_from_float<stream_t>(&arena, pb.x);
            stream_t* d_res = upload_from_float<stream_t>(&arena, pb.residual);
            stream_t* d_nres = arena.alloc<stream_t>(elems);
            stream_t* d_inplace = upload_from_float<stream_t>(&arena, pb.residual);
            const DType sdt = dtype_of_stream<stream_t>();
            fill_params(&p, pb, arena, d_x, d_res, d_nres, d_post, d_comb, nullptr, nullptr,
                        d_scales, d_bases, nullptr, d_ca, d_cb, d_cc, nullptr, nullptr, nullptr,
                        nullptr, nullptr);
            const Tensor ty = kops_test::tensor2d(d_x, sdt, T, H);
            KOPS_EXPECT_OK(kops::model_ops::hc_post(p, ty, *ctx), "hc_post");
            compare_rounded(d_nres, elems, ref.new_residual, sdt, 2.0, "hc_post");
            // In place: new_residual == null writes into the residual buffer.
            HCParams q = p;
            q.residual = d_inplace;
            q.new_residual = nullptr;
            KOPS_EXPECT_OK(kops::model_ops::hc_post(q, ty, *ctx), "hc_post/in-place");
            expect_bits_equal(d_inplace, d_nres, elems, "hc_post/in-place equals out-of-place");

            // Padded token/route layout (still 16 B aligned).
            const int64_t pad = 16;
            const int64_t route_stride = H + pad;
            stream_t* d_resp = arena.alloc<stream_t>(static_cast<size_t>(T) * n * route_stride);
            stream_t* d_outp = arena.alloc<stream_t>(static_cast<size_t>(T) * n * route_stride);
            std::vector<stream_t> padded(static_cast<size_t>(T) * n * route_stride);
            for (size_t i = 0; i < padded.size(); ++i) kops_test::from_float(padded[i], 0.0f);
            for (int64_t t = 0; t < T; ++t)
              for (int i = 0; i < n; ++i)
                for (int64_t h = 0; h < H; ++h)
                  kops_test::from_float(
                      padded[static_cast<size_t>((t * n + i) * route_stride + h)],
                      pb.residual[static_cast<size_t>((t * n + i) * H + h)]);
            kops_test::copy_in(d_resp, padded);
            HCParams r = p;
            r.residual = d_resp;
            r.residual_token_stride = static_cast<int64_t>(n) * route_stride;
            r.residual_route_stride = route_stride;
            r.new_residual = d_outp;
            r.new_residual_token_stride = static_cast<int64_t>(n) * route_stride;
            r.new_residual_route_stride = route_stride;
            KOPS_EXPECT_OK(kops::model_ops::hc_post(r, ty, *ctx), "hc_post/padded");
            std::vector<stream_t> got_padded = kops_test::copy_out(
                d_outp, static_cast<size_t>(T) * n * route_stride);
            std::vector<stream_t> packed(static_cast<size_t>(T * n * H));
            for (int64_t t = 0; t < T; ++t)
              for (int i = 0; i < n; ++i)
                for (int64_t h = 0; h < H; ++h)
                  packed[static_cast<size_t>((t * n + i) * H + h)] =
                      got_padded[static_cast<size_t>((t * n + i) * route_stride + h)];
            compare_rounded(packed.data(), elems, ref.new_residual, sdt, 2.0, "hc_post/padded");
          };
          if (dt == DType::kFloat32) run(float{});
          else if (dt == DType::kFloat16) run(__half{});
          else run(__nv_bfloat16{});
        }
      }
    }
  }
}

void test_coeff_pre(LaunchContext* ctx) {
  for (int n : {2, 4}) {
    for (int64_t H : {64, 128, 256}) {
      for (int64_t T : {1, 3, 17}) {
        for (int iters : {1, 3, 20}) {
          for (bool shifted : {false, true}) {
            Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, shifted,
                                      HCConstraint::kManifold, iters, 31u + iters + (shifted ? 7 : 0));
            char label[128];
            std::snprintf(label, sizeof(label), "coeff_pre.n%d.h%lld.t%lld.it%d.%s", n,
                          static_cast<long long>(H), static_cast<long long>(T), iters,
                          shifted ? "shifted" : "normal");
            run_chain<__nv_bfloat16>(pb, label, ctx, shifted, true);
            run_mega<__nv_bfloat16>(pb, label, ctx);
          }
        }
        // kNone (generic HC / iHC) and fp32 stream.
        {
          Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, false,
                                    HCConstraint::kNone, 1, 41u);
          run_chain<__nv_bfloat16>(pb, "coeff_pre.kNone", ctx, false, false);
          run_mega<__nv_bfloat16>(pb, "coeff_pre.kNone", ctx);
        }
        {
          Problem pb = make_problem(T, H, n, DType::kFloat32, DType::kFloat32, true,
                                    HCConstraint::kManifold, 8, 42u);
          run_chain<float>(pb, "coeff_pre.fp32", ctx, true, false);
          run_mega<float>(pb, "coeff_pre.fp32", ctx);
        }
      }
    }
  }

  // Sub-layer pre-norm path (rmsnorm_weight absent, sub_norm_weight present).
  {
    const int64_t T = 5, H = 128;
    const int n = 4;
    Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, false,
                              HCConstraint::kManifold, 20, 43u);
    Arena arena;
    __nv_bfloat16* d_x = upload_from_float<__nv_bfloat16>(&arena, pb.x);
    __nv_bfloat16* d_res = upload_from_float<__nv_bfloat16>(&arena, pb.residual);
    __nv_bfloat16* d_nres = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * n * H));
    float* d_post = arena.upload(pb.post);
    float* d_comb = arena.upload(pb.comb);
    float* d_fn = arena.upload(pb.fn);
    float* d_scales = arena.upload(pb.scales);
    float* d_bases = arena.upload(pb.bases);
    float* d_ca = arena.alloc<float>(static_cast<size_t>(T * n));
    float* d_cb = arena.alloc<float>(static_cast<size_t>(T * n * n));
    float* d_cc = arena.alloc<float>(static_cast<size_t>(T * n));
    __nv_bfloat16* d_w = upload_from_float<__nv_bfloat16>(&arena, pb.w);
    __nv_bfloat16* d_y = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * H));
    HCParams p{};
    fill_params(&p, pb, arena, d_x, d_res, d_nres, d_post, d_comb, nullptr, d_fn, d_scales, d_bases,
                nullptr, d_ca, d_cb, d_cc, nullptr, nullptr, nullptr, nullptr, nullptr);
    const Tensor ty = kops_test::tensor2d(d_x, DType::kBFloat16, T, H);
    KOPS_EXPECT_OK(kops::model_ops::hc_post(p, ty, *ctx), "sub_norm");
    KOPS_EXPECT_OK(kops::model_ops::hc_coefficients(p, *ctx), "sub_norm");
    const float sub_eps = 1e-5f;
    const Tensor tw = tensor1d(d_w, DType::kBFloat16, H);
    KOPS_EXPECT_OK(kops::model_ops::hc_pre(p, kops_test::tensor2d(d_y, DType::kBFloat16, T, H),
                                           ActQuantOut{}, tw, sub_eps, *ctx),
                   "sub_norm");
    Problem q = pb;
    q.rmsnorm_eps = sub_eps;
    q.rmsnorm_scale = 1.0f;
    const RefOutputs ref = reference(q);
    compare_rounded(d_y, static_cast<size_t>(T * H), ref.y, DType::kBFloat16, 2.0, "sub_norm.y");
  }
}

void test_fp8(LaunchContext* ctx) {
  for (int n : {2, 4}) {
    for (int64_t H : {128, 256}) {
      for (int64_t T : {1, 3, 17}) {
        Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, false,
                                  HCConstraint::kManifold, 10, 51u);
        Arena arena;
        __nv_bfloat16* d_x = upload_from_float<__nv_bfloat16>(&arena, pb.x);
        __nv_bfloat16* d_res = upload_from_float<__nv_bfloat16>(&arena, pb.residual);
        __nv_bfloat16* d_nres = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * n * H));
        float* d_post = arena.upload(pb.post);
        float* d_comb = arena.upload(pb.comb);
        float* d_fn = arena.upload(pb.fn);
        float* d_scales = arena.upload(pb.scales);
        float* d_bases = arena.upload(pb.bases);
        float* d_ca = arena.alloc<float>(static_cast<size_t>(T * n));
        float* d_cb = arena.alloc<float>(static_cast<size_t>(T * n * n));
        float* d_cc = arena.alloc<float>(static_cast<size_t>(T * n));
        __nv_bfloat16* d_w = upload_from_float<__nv_bfloat16>(&arena, pb.w);
        __nv_bfloat16* d_y = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * H));
        uint8_t* d_q = arena.alloc<uint8_t>(static_cast<size_t>(T * H));
        int32_t* d_sf = arena.alloc<int32_t>(static_cast<size_t>(T * H / 128));
        const int32_t block_m = 64;
        const int64_t num_rows = (T + block_m - 1) / block_m * 128;
        int32_t* d_shared = arena.alloc<int32_t>(static_cast<size_t>(num_rows * H / 128));

        HCParams p{};
        fill_params(&p, pb, arena, d_x, d_res, d_nres, d_post, d_comb, nullptr, d_fn, d_scales,
                    d_bases, d_w, d_ca, d_cb, d_cc, nullptr, nullptr, nullptr, nullptr, nullptr);
        p.y_shared_sf = d_shared;
        p.shared_sf_block_m = block_m;

        ActQuantOut q{};
        q.q = d_q;
        q.q_row_stride = H;
        q.q_dtype = DType::kFloat8E4M3;
        q.scales = d_sf;
        q.scale_row_stride = H / 128;
        q.scale_layout = ScaleLayout::kPackedUE8M0x4;
        q.mode = ActQuantMode::kMXFP8;
        q.group_size = 32;
        q.scale_ue8m0 = true;

        const Tensor ty = kops_test::tensor2d(d_x, DType::kBFloat16, T, H);
        KOPS_EXPECT_OK(kops::model_ops::hc_post(p, ty, *ctx), "fp8");
        KOPS_EXPECT_OK(kops::model_ops::hc_coefficients(p, *ctx), "fp8");
        KOPS_EXPECT_OK(kops::model_ops::hc_pre(p, kops_test::tensor2d(d_y, DType::kBFloat16, T, H),
                                               q, Tensor{}, 0.0f, *ctx),
                       "fp8");

        const RefOutputs ref = reference(pb);
        compare_rounded(d_y, static_cast<size_t>(T * H), ref.y, DType::kBFloat16, 2.0,
                        "fp8.y_bf16");
        const Fp8Ref want = quantize_fp8_ref(ref.y, T, H);
        const auto got_payload = kops_test::copy_out(d_q, static_cast<size_t>(T * H));
        const auto got_sf = kops_test::copy_out(d_sf, static_cast<size_t>(T * H / 128));
        const auto got_shared =
            kops_test::copy_out(d_shared, static_cast<size_t>(num_rows * H / 128));

        size_t payload_bad = 0;
        for (size_t i = 0; i < want.payload.size(); ++i) {
          if (got_payload[i] != want.payload[i]) ++payload_bad;
        }
        if (payload_bad * 100 > want.payload.size()) {
          char detail[128];
          std::snprintf(detail, sizeof(detail), "%zu/%zu payload bytes differ", payload_bad,
                        want.payload.size());
          kops_test::record_failure("fp8.payload", detail, __FILE__, __LINE__);
        }
        // Packed UE8M0 words: four consecutive 32-element groups, low byte first.
        size_t sf_bad = 0;
        for (int64_t t = 0; t < T; ++t) {
          for (int64_t w = 0; w < H / 128; ++w) {
            uint32_t packed = 0;
            for (int g = 0; g < 4; ++g) {
              const uint8_t e = want.exp[static_cast<size_t>((t * H / 32) + w * 4 + g)];
              packed |= static_cast<uint32_t>(e) << (8 * g);
            }
            if (static_cast<uint32_t>(got_sf[t * (H / 128) + w]) != packed) ++sf_bad;
            const int64_t idx = t % block_m;
            const int64_t row = t / block_m * 128 + (idx & ~int64_t(127)) + (idx & 31) * 4 +
                                ((idx >> 5) & 3);
            if (static_cast<uint32_t>(got_shared[row + w * num_rows]) != packed) ++sf_bad;
          }
        }
        if (sf_bad != 0) {
          char detail[128];
          std::snprintf(detail, sizeof(detail), "%zu scale words differ", sf_bad);
          kops_test::record_failure("fp8.scales", detail, __FILE__, __LINE__);
        }
        // Dequantized payload equals the reference quantization of the bf16 y.
        size_t deq_bad = 0;
        for (int64_t t = 0; t < T; ++t) {
          for (int64_t h = 0; h < H; ++h) {
            const double sf =
                ue8m0_value(want.exp[static_cast<size_t>((t * H / 32) + h / 32)]);
            const double got = e4m3_value(got_payload[static_cast<size_t>(t * H + h)]) * sf;
            const double refv =
                e4m3_value(want.payload[static_cast<size_t>(t * H + h)]) * sf;
            if (std::fabs(got - refv) > 2e-3 * std::fabs(refv) + 1e-6) ++deq_bad;
          }
        }
        if (deq_bad * 100 > static_cast<size_t>(T * H)) {
          char detail[128];
          std::snprintf(detail, sizeof(detail), "%zu/%lld dequantized values off",
                        deq_bad, static_cast<long long>(T * H));
          kops_test::record_failure("fp8.dequant", detail, __FILE__, __LINE__);
        }
        // Same payload, byte-per-group row-major UE8M0 scales.
        {
          uint8_t* d_sf_b = arena.alloc<uint8_t>(static_cast<size_t>(T * H / 32));
          ActQuantOut qb = q;
          qb.scales = d_sf_b;
          qb.scale_row_stride = H / 32;
          qb.scale_layout = ScaleLayout::kRowMajor;
          KOPS_EXPECT_OK(kops::model_ops::hc_pre(
                             p, kops_test::tensor2d(d_y, DType::kBFloat16, T, H), qb, Tensor{},
                             0.0f, *ctx),
                         "fp8/byte-sf");
          const auto got = kops_test::copy_out(d_sf_b, static_cast<size_t>(T * H / 32));
          size_t bad = 0;
          for (size_t i = 0; i < want.exp.size(); ++i) bad += got[i] != want.exp[i];
          if (bad != 0) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "%zu byte scales differ", bad);
            kops_test::record_failure("fp8.byte_scales", detail, __FILE__, __LINE__);
          }
        }

        // Group-128 fp32 scales: a different granularity and dtype branch.
        if (H % 128 == 0) {
          float* d_sf_f = arena.alloc<float>(static_cast<size_t>(T * H / 128));
          uint8_t* d_q128 = arena.alloc<uint8_t>(static_cast<size_t>(T * H));
          ActQuantOut qf = q;
          qf.q = d_q128;
          qf.scales = d_sf_f;
          qf.scale_row_stride = H / 128;
          qf.scale_layout = ScaleLayout::kRowMajor;
          qf.mode = ActQuantMode::kPerGroup;
          qf.scale_ue8m0 = false;
          qf.group_size = 128;
          p.y_shared_sf = nullptr;
          KOPS_EXPECT_OK(kops::model_ops::hc_pre(
                             p, kops_test::tensor2d(d_y, DType::kBFloat16, T, H), qf, Tensor{},
                             0.0f, *ctx),
                         "fp8/group128");
          const auto got_scales = kops_test::copy_out(d_sf_f, static_cast<size_t>(T * H / 128));
          const auto got_q = kops_test::copy_out(d_q128, static_cast<size_t>(T * H));
          // Quantize the exact bf16 values the kernel produced (the y rounding
          // itself is checked above against the fp64 oracle); this isolates the
          // group-128 scale and payload arithmetic.
          const auto y_dev = kops_test::copy_out(d_y, static_cast<size_t>(T * H));
          std::vector<float> yv(static_cast<size_t>(T * H));
          for (size_t i = 0; i < yv.size(); ++i) yv[i] = kops_test::to_float(y_dev[i]);
          size_t bad = 0;
          for (int64_t t = 0; t < T; ++t) {
            for (int64_t g = 0; g < H / 128; ++g) {
              double amax = 0.0;
              for (int64_t j = 0; j < 128; ++j) {
                amax = std::max(
                    amax, std::fabs(static_cast<double>(yv[static_cast<size_t>(t * H + g * 128 + j)])));
              }
              // DeepGEMM builds the scale and payload in fp32 and applies the
              // reciprocal (`x * (1/sf)`), so mirror that rounding exactly.
              const float want_s = fmaxf(static_cast<float>(amax), 1e-4f) / 448.0f;
              const float inv = 1.0f / want_s;
              const float got_s = got_scales[t * (H / 128) + g];
              if (std::fabs(got_s - want_s) > 1e-6f * want_s) ++bad;
              for (int64_t j = 0; j < 128; ++j) {
                const uint8_t want_b =
                    e4m3_bits(yv[static_cast<size_t>(t * H + g * 128 + j)] * inv);
                if (got_q[static_cast<size_t>(t * H + g * 128 + j)] != want_b) ++bad;
              }
            }
          }
          if (bad != 0) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "%zu group-128 values differ", bad);
            kops_test::record_failure("fp8.group128", detail, __FILE__, __LINE__);
          }
        }
      }
    }
  }
}

void test_extra_paths(LaunchContext* ctx) {
  // n_hc = 1 (degenerate HC): the projection is 3 wide and Sinkhorn is 1x1.
  {
    Problem pb = make_problem(2, 64, 1, DType::kBFloat16, DType::kBFloat16, true,
                              HCConstraint::kManifold, 5, 61u);
    run_chain<__nv_bfloat16>(pb, "paths.n1", ctx, true, true);
    run_mega<__nv_bfloat16>(pb, "paths.n1", ctx);
  }
  // Scalar (non-16 B) pack path of hc_post / hc_expand / hc_head.
  {
    const int64_t T = 2, H = 63;
    const int n = 2;
    Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, false,
                              HCConstraint::kManifold, 20, 62u);
    const RefOutputs ref = reference(pb);
    Arena arena;
    __nv_bfloat16* d_x = upload_from_float<__nv_bfloat16>(&arena, pb.x);
    __nv_bfloat16* d_res = upload_from_float<__nv_bfloat16>(&arena, pb.residual);
    __nv_bfloat16* d_nres = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * n * H));
    float* d_post = arena.upload(pb.post);
    float* d_comb = arena.upload(pb.comb);
    HCParams p{};
    fill_params(&p, pb, arena, d_x, d_res, d_nres, d_post, d_comb, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr);
    KOPS_EXPECT_OK(kops::model_ops::hc_post(p, kops_test::tensor2d(d_x, DType::kBFloat16, T, H),
                                            *ctx),
                   "hc_post/scalar");
    compare_rounded(d_nres, static_cast<size_t>(T * n * H), ref.new_residual, DType::kBFloat16,
                    2.0, "hc_post/scalar");

    // expand + head with a hidden size that is not a 16 B multiple.
    __nv_bfloat16* d_streams = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * n * H));
    __nv_bfloat16* d_out = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * H));
    HCStreamParams sp{};
    sp.streams = d_streams;
    sp.stream_token_stride = static_cast<int64_t>(n) * H;
    sp.stream_dtype = DType::kBFloat16;
    sp.num_tokens = static_cast<int32_t>(T);
    sp.n_hc = n;
    sp.d = static_cast<int32_t>(H);
    KOPS_EXPECT_OK(kops::model_ops::hc_expand(sp, kops_test::tensor2d(d_x, DType::kBFloat16, T, H),
                                              *ctx),
                   "hc_expand/scalar");
    KOPS_EXPECT_OK(kops::model_ops::hc_head(kops_test::tensor2d(d_out, DType::kBFloat16, T, H), sp,
                                            Tensor{}, *ctx),
                   "hc_head/scalar");
    std::vector<float> want(static_cast<size_t>(T * H));
    for (int64_t t = 0; t < T; ++t)
      for (int64_t h = 0; h < H; ++h) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += (1.0 / n) * static_cast<double>(pb.x[t * H + h]);
        want[static_cast<size_t>(t * H + h)] = static_cast<float>(acc);
      }
    compare_rounded(d_out, static_cast<size_t>(T * H), want, DType::kBFloat16, 2.0,
                    "hc_head/scalar");
  }
  // Wide row beyond the 48 KB shared stage: hc_pre recomputes instead of staging.
  {
    Problem pb = make_problem(1, 16384, 2, DType::kBFloat16, DType::kBFloat16, false,
                              HCConstraint::kManifold, 20, 63u);
    run_chain<__nv_bfloat16>(pb, "paths.wide_row", ctx, false, false);
    run_mega<__nv_bfloat16>(pb, "paths.wide_row", ctx);
  }
  // raw_logits without a GEMM-provided row_sumsq: the kernel derives it from X.
  {
    const int64_t T = 3, H = 128;
    const int n = 4, N = n * (n + 2);
    Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, false,
                              HCConstraint::kManifold, 20, 64u);
    const RefOutputs ref = reference(pb);
    Arena arena;
    __nv_bfloat16* d_x = upload_from_float<__nv_bfloat16>(&arena, pb.x);
    __nv_bfloat16* d_res = upload_from_float<__nv_bfloat16>(&arena, pb.residual);
    __nv_bfloat16* d_nres = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * n * H));
    float* d_post = arena.upload(pb.post);
    float* d_comb = arena.upload(pb.comb);
    float* d_fn = arena.upload(pb.fn);
    float* d_scales = arena.upload(pb.scales);
    float* d_bases = arena.upload(pb.bases);
    float* d_ca = arena.alloc<float>(static_cast<size_t>(T * n));
    float* d_cb = arena.alloc<float>(static_cast<size_t>(T * n * n));
    float* d_cc = arena.alloc<float>(static_cast<size_t>(T * n));
    std::vector<float> raw(static_cast<size_t>(T * N));
    for (int64_t t = 0; t < T; ++t)
      for (int o = 0; o < N; ++o) raw[t * N + o] = static_cast<float>(ref.dot[t * N + o]);
    float* d_raw = arena.upload(raw);
    HCParams p{};
    fill_params(&p, pb, arena, d_x, d_res, d_nres, d_post, d_comb, nullptr, d_fn, d_scales, d_bases,
                nullptr, d_ca, d_cb, d_cc, d_raw, nullptr, nullptr, nullptr, nullptr);
    KOPS_EXPECT_OK(kops::model_ops::hc_post(p, kops_test::tensor2d(d_x, DType::kBFloat16, T, H),
                                            *ctx),
                   "raw_no_sumsq");
    KOPS_EXPECT_OK(kops::model_ops::hc_coefficients(p, *ctx), "raw_no_sumsq");
    compare_output(d_ca, static_cast<size_t>(T * n), ref.coeff_a, 1e-6, 1e-6, "raw_no_sumsq.a");
    compare_output(d_cb, static_cast<size_t>(T * n * n), ref.coeff_b, 1e-6, 1e-6,
                   "raw_no_sumsq.b");
    compare_output(d_cc, static_cast<size_t>(T * n), ref.coeff_c, 1e-6, 1e-6, "raw_no_sumsq.c");
  }
}

/// The m16 tensor-core projection body: raw_logits + row_sumsq are the scratch pair, so
/// hc_coefficients (T >= 16) projects 16 tokens per CTA on tf32 mma and writes both. Every
/// output is checked against the fp64 oracle and chain vs mega stays bitwise (run_mega).
void test_tile_projection(LaunchContext* ctx) {
  for (int n : {2, 4}) {
    for (int64_t H : {64, 256, 1024}) {
      for (int64_t T : {256, 257, 272}) {
        Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, false,
                                  HCConstraint::kManifold, 20, 71u + static_cast<unsigned>(T + H));
        char label[128];
        std::snprintf(label, sizeof(label), "tile.n%d.h%lld.t%lld", n,
                      static_cast<long long>(H), static_cast<long long>(T));
        run_chain<__nv_bfloat16>(pb, label, ctx, false, false, true);
        run_mega<__nv_bfloat16>(pb, label, ctx);
      }
    }
  }
  // fp32 stream: the A operand is split into tf32 halves like fn (kSplitA).
  {
    Problem pb = make_problem(272, 256, 4, DType::kFloat32, DType::kFloat32, true,
                              HCConstraint::kManifold, 20, 74u);
    run_chain<float>(pb, "tile.fp32", ctx, true, false, true);
    run_mega<float>(pb, "tile.fp32", ctx);
  }
  // n_hc = 1: three outputs cannot fill an n8 tile, so both bodies stay on the generic path.
  {
    Problem pb = make_problem(272, 128, 1, DType::kBFloat16, DType::kBFloat16, false,
                              HCConstraint::kManifold, 20, 75u);
    run_chain<__nv_bfloat16>(pb, "tile.n1", ctx, false, true, true);
    run_mega<__nv_bfloat16>(pb, "tile.n1", ctx);
  }
  // kNone (raw coefficients, no Sinkhorn) through the tile body, and a partial last tile.
  {
    Problem pb = make_problem(259, 512, 4, DType::kBFloat16, DType::kBFloat16, false,
                              HCConstraint::kNone, 1, 76u);
    run_chain<__nv_bfloat16>(pb, "tile.kNone", ctx, false, false, true);
    run_mega<__nv_bfloat16>(pb, "tile.kNone", ctx);
  }
  // FP8 quantized layer input through the tile body (kPerGroup 32 UE8M0 packed).
  {
    const int64_t T = 258, H = 256;
    const int n = 4;
    Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, false,
                              HCConstraint::kManifold, 10, 77u);
    Arena arena;
    __nv_bfloat16* d_x = upload_from_float<__nv_bfloat16>(&arena, pb.x);
    __nv_bfloat16* d_res = upload_from_float<__nv_bfloat16>(&arena, pb.residual);
    __nv_bfloat16* d_nres = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * n * H));
    float* d_post = arena.upload(pb.post);
    float* d_comb = arena.upload(pb.comb);
    float* d_fn = arena.upload(pb.fn);
    float* d_scales = arena.upload(pb.scales);
    float* d_bases = arena.upload(pb.bases);
    float* d_ca = arena.alloc<float>(static_cast<size_t>(T * n));
    float* d_cb = arena.alloc<float>(static_cast<size_t>(T * n * n));
    float* d_cc = arena.alloc<float>(static_cast<size_t>(T * n));
    float* d_sq = arena.alloc<float>(static_cast<size_t>(T));
    float* d_raw = arena.alloc<float>(static_cast<size_t>(T * n * (n + 2)));
    __nv_bfloat16* d_w = upload_from_float<__nv_bfloat16>(&arena, pb.w);
    __nv_bfloat16* d_y = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * H));
    uint8_t* d_q = arena.alloc<uint8_t>(static_cast<size_t>(T * H));
    int32_t* d_sf = arena.alloc<int32_t>(static_cast<size_t>(T * H / 128));
    HCParams p{};
    fill_params(&p, pb, arena, d_x, d_res, d_nres, d_post, d_comb, nullptr, d_fn, d_scales, d_bases,
                d_w, d_ca, d_cb, d_cc, d_raw, d_sq, nullptr, nullptr, nullptr);
    ActQuantOut q{};
    q.q = d_q;
    q.q_row_stride = H;
    q.q_dtype = DType::kFloat8E4M3;
    q.scales = d_sf;
    q.scale_row_stride = H / 128;
    q.scale_layout = ScaleLayout::kPackedUE8M0x4;
    q.mode = ActQuantMode::kMXFP8;
    q.group_size = 32;
    q.scale_ue8m0 = true;
    const Tensor ty = kops_test::tensor2d(d_x, DType::kBFloat16, T, H);
    KOPS_EXPECT_OK(kops::model_ops::hc_post(p, ty, *ctx), "tile.fp8");
    KOPS_EXPECT_OK(kops::model_ops::hc_coefficients(p, *ctx), "tile.fp8");
    KOPS_EXPECT_OK(kops::model_ops::hc_pre(p, kops_test::tensor2d(d_y, DType::kBFloat16, T, H), q,
                                           Tensor{}, 0.0f, *ctx),
                   "tile.fp8");
    const RefOutputs ref = reference(pb);
    compare_output(d_ca, static_cast<size_t>(T * n), ref.coeff_a, 3e-5, 3e-5, "tile.fp8.coeff_a");
    compare_rounded(d_y, static_cast<size_t>(T * H), ref.y, DType::kBFloat16, 2.0, "tile.fp8.y");
    // mega_mhc must reproduce the same quantized payload.
    __nv_bfloat16* m_y = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * H));
    uint8_t* m_q = arena.alloc<uint8_t>(static_cast<size_t>(T * H));
    int32_t* m_sf = arena.alloc<int32_t>(static_cast<size_t>(T * H / 128));
    __nv_bfloat16* d_res2 = upload_from_float<__nv_bfloat16>(&arena, pb.residual);
    __nv_bfloat16* m_nres = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * n * H));
    float* m_ca = arena.alloc<float>(static_cast<size_t>(T * n));
    float* m_cb = arena.alloc<float>(static_cast<size_t>(T * n * n));
    float* m_cc = arena.alloc<float>(static_cast<size_t>(T * n));
    float* m_sq = arena.alloc<float>(static_cast<size_t>(T));
    HCParams mp = p;
    mp.residual = d_res2;
    mp.new_residual = m_nres;
    mp.coeff_a = m_ca;
    mp.coeff_b = m_cb;
    mp.coeff_c = m_cc;
    mp.row_sumsq = m_sq;
    mp.raw_logits = nullptr;
    mp.y_shared_sf = nullptr;
    ActQuantOut mq = q;
    mq.q = m_q;
    mq.scales = m_sf;
    const Tensor tb = kops_test::tensor3d(d_comb, DType::kFloat32, T, n, n);
    const Tensor tcc = kops_test::tensor3d(d_post, DType::kFloat32, T, n, 1);
    KOPS_EXPECT_OK(kops::model_ops::mega_mhc(mp, tb, tcc, ty,
                                             kops_test::tensor2d(m_y, DType::kBFloat16, T, H), mq,
                                             Tensor{}, 0.0f, *ctx),
                   "tile.fp8.mega");
    expect_bits_equal(m_y, d_y, static_cast<size_t>(T * H), "tile.fp8.mega.y");
    expect_bits_equal(m_q, d_q, static_cast<size_t>(T * H), "tile.fp8.mega.payload");
    expect_bits_equal(m_sf, d_sf, static_cast<size_t>(T * H / 128), "tile.fp8.mega.scales");
  }
}

void test_large_smoke(LaunchContext* ctx) {
  // H = 7168, n = 4, T = 4: the DeepSeek-V4 shape, exercised on the generic body.
  Problem pb = make_problem(4, 7168, 4, DType::kBFloat16, DType::kBFloat16, true,
                            HCConstraint::kManifold, 20, 77u);
  run_chain<__nv_bfloat16>(pb, "large.h7168", ctx, true, true);
  run_mega<__nv_bfloat16>(pb, "large.h7168", ctx);

  // Same shape with a full tile and a partial one: the tensor-core projection at
  // K = n*H = 28672 (the benchmarked case), checked against the fp64 oracle.
  for (int64_t T : {256, 272}) {
    Problem tile = make_problem(T, 7168, 4, DType::kBFloat16, DType::kBFloat16, false,
                                HCConstraint::kManifold, 20, 78u + static_cast<unsigned>(T));
    char label[64];
    std::snprintf(label, sizeof(label), "large.tile.t%lld", static_cast<long long>(T));
    run_chain<__nv_bfloat16>(tile, label, ctx, false, false, true);
    run_mega<__nv_bfloat16>(tile, label, ctx);
  }
}

void test_errors(LaunchContext* ctx) {
  const int64_t T = 3, H = 64;
  const int n = 4;
  Problem pb = make_problem(T, H, n, DType::kBFloat16, DType::kBFloat16, true,
                            HCConstraint::kManifold, 20, 99u);
  Arena arena;
  __nv_bfloat16* d_x = upload_from_float<__nv_bfloat16>(&arena, pb.x);
  __nv_bfloat16* d_res = upload_from_float<__nv_bfloat16>(&arena, pb.residual);
  __nv_bfloat16* d_nres = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * n * H));
  float* d_post = arena.upload(pb.post);
  float* d_comb = arena.upload(pb.comb);
  float* d_shift = arena.upload(pb.shift);
  float* d_fn = arena.upload(pb.fn);
  float* d_scales = arena.upload(pb.scales);
  float* d_bases = arena.upload(pb.bases);
  __nv_bfloat16* d_w = upload_from_float<__nv_bfloat16>(&arena, pb.w);
  float* d_ca = arena.alloc<float>(static_cast<size_t>(T * n));
  float* d_cb = arena.alloc<float>(static_cast<size_t>(T * n * n));
  float* d_cc = arena.alloc<float>(static_cast<size_t>(T * n));
  __nv_bfloat16* d_y = arena.alloc<__nv_bfloat16>(static_cast<size_t>(T * H));
  HCParams p{};
  fill_params(&p, pb, arena, d_x, d_res, d_nres, d_post, d_comb, d_shift, d_fn, d_scales, d_bases,
              d_w, d_ca, d_cb, d_cc, nullptr, nullptr, nullptr, nullptr, nullptr);
  const Tensor ty = kops_test::tensor2d(d_x, DType::kBFloat16, T, H);

  {
    HCParams q = p;
    q.residual = nullptr;
    KOPS_EXPECT_STATUS(kops::model_ops::hc_post(q, ty, *ctx), Status::kInvalidArgument,
                       "hc_post null residual");
  }
  {
    HCParams q = p;
    q.n_hc = 3;
    KOPS_EXPECT_STATUS(kops::model_ops::hc_coefficients(q, *ctx), Status::kUnsupportedShape,
                       "hc_coefficients n_hc=3");
  }
  {
    HCParams q = p;
    q.fn = nullptr;
    KOPS_EXPECT_STATUS(kops::model_ops::hc_coefficients(q, *ctx), Status::kInvalidArgument,
                       "hc_coefficients missing fn");
  }
  {
    HCParams q = p;
    q.num_sinkhorn_iters = 0;
    KOPS_EXPECT_STATUS(kops::model_ops::hc_coefficients(q, *ctx), Status::kInvalidArgument,
                       "hc_coefficients sinkhorn iters 0");
  }
  {
    HCParams q = p;
    q.residual_route_stride = H + 2;  // 2-byte stream: not a 16 B multiple
    q.residual_token_stride = static_cast<int64_t>(n) * (H + 2);
    q.new_residual_token_stride = static_cast<int64_t>(n) * (H + 2);
    q.new_residual_route_stride = H + 2;
    KOPS_EXPECT_STATUS(kops::model_ops::hc_pre(q, kops_test::tensor2d(d_y, DType::kBFloat16, T, H),
                                               ActQuantOut{}, Tensor{}, 0.0f, *ctx),
                       Status::kMisaligned, "hc_pre misaligned pack");
  }
  {
    HCParams q = p;
    q.coeff_a = nullptr;
    q.shifted_prev_mix = nullptr;
    KOPS_EXPECT_STATUS(kops::model_ops::hc_pre(q, kops_test::tensor2d(d_y, DType::kBFloat16, T, H),
                                               ActQuantOut{}, Tensor{}, 0.0f, *ctx),
                       Status::kInvalidArgument, "hc_pre missing coeff_a");
  }
  {
    ActQuantOut q{};
    q.mode = ActQuantMode::kStaticPerTensor;
    KOPS_EXPECT_STATUS(kops::model_ops::hc_pre(p, kops_test::tensor2d(d_y, DType::kBFloat16, T, H),
                                               q, Tensor{}, 0.0f, *ctx),
                       Status::kUnsupportedDtype, "hc_pre unsupported quant mode");
  }
  {
    Tensor bad_out = kops_test::tensor2d(d_y, DType::kBFloat16, H, T);
    KOPS_EXPECT_STATUS(kops::model_ops::hc_post(p, bad_out, *ctx), Status::kInvalidArgument,
                       "hc_post y shape");
  }
  {
    float* d_xf = upload_from_float<float>(&arena, pb.x);
    Tensor yf = kops_test::tensor2d(d_xf, DType::kFloat32, T, H);
    KOPS_EXPECT_STATUS(kops::model_ops::hc_post(p, yf, *ctx), Status::kUnsupportedDtype,
                       "hc_post y dtype");
  }
  {
    const Tensor tb = kops_test::tensor3d(d_comb, DType::kFloat32, T, n, n);
    const Tensor tc = kops_test::tensor3d(d_post, DType::kFloat32, T, n, 1);
    HCParams q = p;
    q.raw_logits = d_scales;
    KOPS_EXPECT_STATUS(kops::model_ops::mega_mhc(q, tb, tc, ty,
                                                 kops_test::tensor2d(d_y, DType::kBFloat16, T, H),
                                                 ActQuantOut{}, Tensor{}, 0.0f, *ctx),
                       Status::kInvalidArgument, "mega raw_logits");
  }
  {
    const Tensor tb = kops_test::tensor3d(d_comb, DType::kFloat32, T, n, n);
    const Tensor tc = kops_test::tensor3d(d_post, DType::kFloat32, T, n, 1);
    float* d_yf = upload_from_float<float>(&arena, pb.x);
    Tensor yy = kops_test::tensor2d(d_yf, DType::kFloat32, T, H);
    KOPS_EXPECT_STATUS(kops::model_ops::mega_mhc(p, tb, tc, yy,
                                                 kops_test::tensor2d(d_y, DType::kBFloat16, T, H),
                                                 ActQuantOut{}, Tensor{}, 0.0f, *ctx),
                       Status::kUnsupportedDtype, "mega prev_y dtype");
  }
  {
    HCStreamParams sp{};
    sp.streams = d_res;
    sp.stream_token_stride = 7;
    sp.stream_dtype = DType::kBFloat16;
    sp.num_tokens = static_cast<int32_t>(T);
    sp.n_hc = n;
    sp.d = static_cast<int32_t>(H);
    KOPS_EXPECT_STATUS(kops::model_ops::hc_expand(sp, ty, *ctx), Status::kUnsupportedLayout,
                       "hc_expand unpacked streams");
  }
  {
    HCStreamParams sp{};
    sp.streams = d_res;
    sp.stream_token_stride = static_cast<int64_t>(n) * H;
    sp.stream_dtype = DType::kBFloat16;
    sp.num_tokens = static_cast<int32_t>(T);
    sp.n_hc = n;
    sp.d = static_cast<int32_t>(H);
    Tensor wbad = tensor1d(d_scales, DType::kFloat32, n);
    wbad.shape[0] = n + 1;
    KOPS_EXPECT_STATUS(kops::model_ops::hc_head(kops_test::tensor2d(d_y, DType::kBFloat16, T, H),
                                                sp, wbad, *ctx),
                       Status::kInvalidArgument, "hc_head static weight length");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const kops_test::Options opt = kops_test::parse_options(argc, argv);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 1;
  if (!kops_test::tier_enabled(opt, "smoke")) return kops_test::finish("kops_test_hyper_connection");

  test_expand_head(&ctx);
  test_post(&ctx);
  test_coeff_pre(&ctx);
  test_fp8(&ctx);
  test_errors(&ctx);
  test_extra_paths(&ctx);
  test_tile_projection(&ctx);
  if (kops_test::tier_enabled(opt, "standard")) test_large_smoke(&ctx);
  return kops_test::finish("kops_test_hyper_connection");
}
