// =============================================================================
// kernel-ops | tests/unit/test_sm120.cc
// B13 SM120a arch cell (csrc/gemm/sm120a_impl.cuh) through the public surface:
//   gemm_sm120_kernel            bf16 / fp16 dense
//   fp8_gemm_sm120_kernel        e4m3 tensorwise / rowwise
// Both are reached with gemm_with_tactic() + a tactic whose mainloop is
// kSm120WarpSpecialized. The oracle is
//   (a) the fp64 host product of the *stored* operand values (bf16/fp16 rounded, e4m3
//       decoded), with the epilogue applied in the documented order, and
//   (b) the B7 mma.sync body (gemm_mma_sync_kernel via a kMultistageMmaSync tactic),
//       which must agree bitwise: same atom, same increasing k-step chain.
// Contract checks assert the Status of refused calls and that a refused call writes
// nothing (sentinel output).
// Tier: standard. Skips (77) unless the machine is exactly KOPS_TARGET_SM120A.
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
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

namespace {

using kops::DType;
using kops::LaunchContext;
using kops::Status;
namespace kg = kops::gemm;

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  float uni(float lo, float hi) {
    const float u = static_cast<float>((next() >> 40) * (1.0 / 16777216.0));
    return lo + (hi - lo) * u;
  }
};

/// One operand as the kernel will read it: the stored bytes and the exact value of every
/// stored element (bf16/fp16 rounded, e4m3 decoded).
struct Operand {
  std::vector<uint8_t> raw;
  std::vector<float> value;
};

uint8_t e4m3_from_float(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
float e4m3_to_float(uint8_t b) { return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)); }

template <typename T>
float as_float(T v) {
  return kops_test::to_float(v);
}

/// 16-bit operand of `count` elements in [-1, 1] (bf16 or fp16 according to `dt`).
Operand make_16(int64_t count, unsigned seed, DType dt, float lo = -1.0f, float hi = 1.0f) {
  Rng r(seed);
  Operand op;
  op.raw.resize(static_cast<size_t>(count) * 2);
  op.value.resize(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const float x = r.uni(lo, hi);
    if (dt == DType::kFloat16) {
      const __half h = __float2half_rn(x);
      std::memcpy(&op.raw[static_cast<size_t>(i) * 2], &h, 2);
      op.value[static_cast<size_t>(i)] = __half2float(h);
    } else {
      const __nv_bfloat16 h = __float2bfloat16_rn(x);
      std::memcpy(&op.raw[static_cast<size_t>(i) * 2], &h, 2);
      op.value[static_cast<size_t>(i)] = __bfloat162float(h);
    }
  }
  return op;
}

/// e4m3 operand of `count` elements in [-1, 1].
Operand make_e4m3(int64_t count, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
  Rng r(seed);
  Operand op;
  op.raw.resize(static_cast<size_t>(count));
  op.value.resize(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint8_t b = e4m3_from_float(r.uni(lo, hi));
    op.raw[static_cast<size_t>(i)] = b;
    op.value[static_cast<size_t>(i)] = e4m3_to_float(b);
  }
  return op;
}

template <typename T>
struct Dev {
  T* p = nullptr;
  explicit Dev(size_t n) : p(kops_test::device_alloc<T>(n)) {}
  ~Dev() { kops_test::device_free(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  Dev(Dev&& other) noexcept : p(other.p) { other.p = nullptr; }
};

template <typename T>
Dev<T> upload(const std::vector<uint8_t>& raw) {
  Dev<T> d(raw.size() / sizeof(T));
  KOPS_TEST_CHECK(raw.size() % sizeof(T) == 0);
  const cudaError_t err = cudaMemcpy(d.p, raw.data(), raw.size(), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) kops_test::record_failure("cudaMemcpy H2D", cudaGetErrorString(err), __FILE__, __LINE__);
  return d;
}

std::vector<uint8_t> download_bytes(const void* p, size_t bytes) {
  std::vector<uint8_t> out(bytes);
  const cudaError_t err = cudaMemcpy(out.data(), p, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) kops_test::record_failure("cudaMemcpy D2H", cudaGetErrorString(err), __FILE__, __LINE__);
  return out;
}

/// Stored bytes -> float values (for the tolerance comparison).
std::vector<float> bytes_to_values(const std::vector<uint8_t>& raw, size_t count, DType dt) {
  std::vector<float> v(count, 0.0f);
  for (size_t i = 0; i < count; ++i) {
    if (dt == DType::kFloat32) {
      float f;
      std::memcpy(&f, &raw[i * 4], 4);
      v[i] = f;
    } else if (dt == DType::kFloat8E4M3) {
      v[i] = e4m3_to_float(raw[i]);
    } else if (dt == DType::kFloat16) {
      __half h;
      std::memcpy(&h, &raw[i * 2], 2);
      v[i] = __half2float(h);
    } else {
      __nv_bfloat16 h;
      std::memcpy(&h, &raw[i * 2], 2);
      v[i] = __bfloat162float(h);
    }
  }
  return v;
}

kg::GemmTactic sm120_tactic(int tm = 64, int tn = 128, int tk = 0, int stages = 0) {
  kg::GemmTactic t{};
  t.tile_m = static_cast<int16_t>(tm);
  t.tile_n = static_cast<int16_t>(tn);
  t.tile_k = static_cast<int16_t>(tk);
  t.stages = static_cast<int8_t>(stages);
  t.mainloop = kg::MainloopSchedule::kSm120WarpSpecialized;
  t.epilogue = kg::EpilogueSchedule::kDirectStore;
  return t;
}

/// B7 reference tactic: the shared mma.sync body (tile 64x128, four stages).
kg::GemmTactic b7_tactic() {
  kg::GemmTactic t{};
  t.tile_m = 64;
  t.tile_n = 128;
  t.mainloop = kg::MainloopSchedule::kMultistageMmaSync;
  t.epilogue = kg::EpilogueSchedule::kDirectStore;
  return t;
}

kg::ScaledGemmArgs make_args(kg::GemmRecipe recipe, DType dt, DType out_dt, int64_t m, int64_t n, int64_t k,
                             const void* a, const void* b, void* d, int64_t ldd) {
  kg::ScaledGemmArgs x{};
  x.recipe = recipe;
  x.a.data = a;
  x.a.ld = k;
  x.a.dtype = dt;
  x.b.data = b;
  x.b.ld = k;
  x.b.dtype = dt;
  x.d = d;
  x.ldd = ldd;
  x.d_dtype = out_dt;
  x.m = m;
  x.n = n;
  x.k = k;
  x.batch = 1;
  return x;
}

/// Host copies of the epilogue operands (the EpilogueParams struct carries device pointers, so the
/// reference cannot read them).
struct EpiHost {
  const std::vector<float>* row_scale = nullptr;
  const std::vector<float>* col_scale = nullptr;
  const std::vector<float>* bias = nullptr;       // [n] fp32
  const std::vector<float>* residual = nullptr;   // [m * residual_stride]
  int64_t residual_stride = 0;
};

/// Product of the *stored* operand values in fp64 plus the epilogue in the documented order.
std::vector<double> reference(const std::vector<float>& av, const std::vector<float>& bv, int64_t m, int64_t n,
                             int64_t k, const kg::EpilogueParams& e, const EpiHost& eh,
                             const std::vector<float>* a_scale, const std::vector<float>* b_scale) {
  std::vector<double> out(static_cast<size_t>(m * n), 0.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t kk = 0; kk < k; ++kk) acc += static_cast<double>(av[i * k + kk]) * bv[j * k + kk];
      double v = acc * (e.alpha_ptr != nullptr ? *e.alpha_ptr : e.alpha);
      if (a_scale != nullptr) v *= (*a_scale)[i];
      if (b_scale != nullptr) v *= (*b_scale)[j];
      if ((e.ops & kg::kEpiRowScale) != 0u) v *= (*eh.row_scale)[i];
      if ((e.ops & kg::kEpiColScale) != 0u) v *= (*eh.col_scale)[j];
      if ((e.ops & kg::kEpiBias) != 0u) v += (*eh.bias)[j];
      if ((e.ops & kg::kEpiClamp) != 0u) v = std::fmin(std::fmax(v, -e.clamp_limit), e.clamp_limit);
      if ((e.ops & kg::kEpiSoftcap) != 0u) v = static_cast<double>(e.softcap) * std::tanh(v / e.softcap);
      if ((e.ops & kg::kEpiAct) != 0u && e.act == kops::ActFn::kReLU) v = std::fmax(v, 0.0);
      if ((e.ops & kg::kEpiResidualAdd) != 0u) {
        v = static_cast<double>(e.beta) * (*eh.residual)[i * eh.residual_stride + j] + v;
      }
      out[static_cast<size_t>(i * n + j)] = v;
    }
  }
  return out;
}

/// Rounds a host vector to the storage type and back (what D holds after one rounding).
std::vector<float> round_to(const std::vector<double>& v, DType dt) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    const float f = static_cast<float>(v[i]);
    if (dt == DType::kFloat32) {
      out[i] = f;
    } else if (dt == DType::kFloat8E4M3) {
      out[i] = e4m3_to_float(e4m3_from_float(f));
    } else if (dt == DType::kFloat16) {
      out[i] = __half2float(__float2half_rn(f));
    } else {
      out[i] = __bfloat162float(__float2bfloat16_rn(f));
    }
  }
  return out;
}

void compare_close(const std::vector<float>& got, const std::vector<double>& want, DType out_dt, double atol,
                   double rtol, const char* what) {
  if (got.size() != want.size()) {
    kops_test::record_failure(what, "size mismatch", __FILE__, __LINE__);
    return;
  }
  // The one rounding to the output dtype is expected; compare the rounded reference.
  const std::vector<float> rounded = round_to(want, out_dt);
  double worst = 0.0;
  size_t worst_i = 0;
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - static_cast<double>(rounded[i]));
    if (err > atol + rtol * std::fabs(static_cast<double>(rounded[i]))) {
      ++bad;
      if (err > worst) {
        worst = err;
        worst_i = i;
      }
    }
  }
  if (bad != 0) {
    char detail[256];
    std::snprintf(detail, sizeof(detail), "%zu/%zu mismatched, worst %zu: got %.8g want %.8g (err %.3g)",
                  bad, got.size(), worst_i, static_cast<double>(got[worst_i]),
                  static_cast<double>(rounded[worst_i]), worst);
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
}

bool same_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) { return a == b; }

// -----------------------------------------------------------------------------
// BF16 / FP16 dense
// -----------------------------------------------------------------------------

/// Runs one 16-bit case on both the SM120a cell and the B7 body and checks them against the
/// fp64 product of the stored operands.
void dense_case(const LaunchContext& ctx, DType dt, int64_t m, int64_t n, int64_t k, int tm, int tn,
                const kg::EpilogueParams& epi_in, const EpiHost& eh, double atol, double rtol, const char* what,
                bool expect_bitwise_b7 = true) {
  kg::EpilogueParams epi = epi_in;
  const kg::GemmRecipe recipe = dt == DType::kFloat16 ? kg::GemmRecipe::kFP16 : kg::GemmRecipe::kBF16;
  const Operand a = make_16(m * k, static_cast<unsigned>(m * 131 + n * 7 + k), dt);
  const Operand b = make_16(n * k, static_cast<unsigned>(m * 17 + n * 3 + k * 5 + 1), dt);
  auto da = upload<uint16_t>(a.raw);
  auto db = upload<uint16_t>(b.raw);
  Dev<uint16_t> d_sm120(static_cast<size_t>(m * n));
  Dev<uint16_t> d_b7(static_cast<size_t>(m * n));
  const size_t out_bytes = static_cast<size_t>(m * n) * 2;
  const cudaError_t memset_err = cudaMemset(d_sm120.p, 0, out_bytes);
  if (memset_err != cudaSuccess) kops_test::record_failure("cudaMemset", cudaGetErrorString(memset_err), __FILE__, __LINE__);
  cudaMemset(d_b7.p, 0, out_bytes);

  kg::ScaledGemmArgs args = make_args(recipe, dt, dt, m, n, k, da.p, db.p, d_sm120.p, n);
  args.epi = epi;
  KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm120_tactic(tm, tn), kops::Workspace{}, ctx), what);
  if (expect_bitwise_b7) {
    auto args_b7 = args;
    args_b7.d = d_b7.p;
    KOPS_EXPECT_OK(kg::gemm_with_tactic(args_b7, b7_tactic(), kops::Workspace{}, ctx), what);
  }
  const cudaError_t sync = cudaDeviceSynchronize();
  if (sync != cudaSuccess) {
    kops_test::record_failure(what, cudaGetErrorString(sync), __FILE__, __LINE__);
    return;
  }
  const auto raw = download_bytes(d_sm120.p, out_bytes);
  compare_close(bytes_to_values(raw, static_cast<size_t>(m * n), dt),
                reference(a.value, b.value, m, n, k, args.epi, eh, nullptr, nullptr), dt, atol, rtol, what);
  if (expect_bitwise_b7) {
    const auto raw_b7 = download_bytes(d_b7.p, out_bytes);
    if (!same_bytes(raw, raw_b7)) {
      size_t diff = 0;
      for (size_t i = 0; i < raw.size(); ++i) diff += raw[i] != raw_b7[i];
      char detail[128];
      std::snprintf(detail, sizeof(detail), "%zu/%zu bytes differ from the B7 mma.sync result", diff, raw.size());
      kops_test::record_failure(what, detail, __FILE__, __LINE__);
    }
  }
}

void test_dense_sweep(const LaunchContext& ctx) {
  kg::EpilogueParams plain{};
  const EpiHost none{};
  // M sweep: every M is exercised, including rows past the tile and the M < tile tail.
  for (DType dt : {DType::kBFloat16, DType::kFloat16}) {
    for (int64_t m : {1, 7, 16, 64, 128}) {
      dense_case(ctx, dt, m, 200, 256, 64, 128, plain, none, 0.05, 0.01, "sm120 dense M sweep 64x128", true);
      dense_case(ctx, dt, m, 200, 200, 128, 128, plain, none, 0.05, 0.01, "sm120 dense M sweep 128x128", true);
    }
  }
}

void test_dense_tails(const LaunchContext& ctx) {
  kg::EpilogueParams plain{};
  const EpiHost none{};
  // N and K not divisible by the tile: N = 1 / 3 / 7 / 129 / 200 (tile N 128), K = 8 / 40 / 200 / 256.
  for (const auto& shape : {std::pair<int64_t, int64_t>{1, 8}, {3, 40}, {7, 200}, {129, 200}, {200, 256}, {65, 264}}) {
    dense_case(ctx, DType::kBFloat16, 40, shape.first, shape.second, 64, 128, plain, none, 0.05, 0.01, "sm120 dense N/K tails", true);
    dense_case(ctx, DType::kFloat16, 40, shape.first, shape.second, 128, 128, plain, none, 0.05, 0.01, "sm120 dense N/K tails", true);
  }
}

/// Zero operands with a sentinel output: the padding rows / columns and the K tail must
/// contribute exact zeros, never a NaN from uninitialised shared memory.
void test_dense_zero_and_guard(const LaunchContext& ctx) {
  const int64_t m = 5, n = 200, k = 200;
  const int64_t ldd = n + 7;                       // guard columns
  const int64_t rows = m + 3;                      // guard rows
  const float sentinel = 7.0f;
  const Operand b = make_16(n * k, 99u, DType::kBFloat16);
  auto db = upload<uint16_t>(b.raw);
  const std::vector<uint8_t> zeros(static_cast<size_t>(m * k) * 2, 0);
  auto da = upload<uint16_t>(zeros);
  std::vector<__nv_bfloat16> host_out(static_cast<size_t>(rows * ldd));
  for (auto& v : host_out) v = __float2bfloat16_rn(sentinel);
  Dev<__nv_bfloat16> dout(host_out.size());
  kops_test::copy_in(dout.p, host_out);

  kg::ScaledGemmArgs args = make_args(kg::GemmRecipe::kBF16, DType::kBFloat16, DType::kBFloat16, m, n, k, da.p, db.p,
                                      dout.p, ldd);
  KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm120_tactic(), kops::Workspace{}, ctx), "sm120 zero operands");
  if (cudaDeviceSynchronize() != cudaSuccess) {
    kops_test::record_failure("sm120 zero operands", "device error", __FILE__, __LINE__);
    return;
  }
  const auto got = kops_test::copy_out(dout.p, host_out.size());
  size_t nonzero = 0, touched = 0;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < ldd; ++c) {
      const float v = __bfloat162float(got[static_cast<size_t>(r * ldd + c)]);
      if (r < m && c < n) {
        if (v != 0.0f) ++nonzero;                  // zero x random weights = exact zero
      } else if (v != sentinel) {
        ++touched;                                 // nothing outside [0, M) x [0, N) may be written
      }
    }
  }
  if (nonzero != 0) kops_test::record_failure("sm120 zero operands", "a zero product wrote a non-zero output", __FILE__, __LINE__);
  if (touched != 0) kops_test::record_failure("sm120 zero operands", "the tail rows / columns were written", __FILE__, __LINE__);
}

void test_dense_epilogue(const LaunchContext& ctx) {
  const int64_t m = 33, n = 200, k = 256;
  Rng r(4242);
  std::vector<float> row_scale(static_cast<size_t>(m)), col_scale(static_cast<size_t>(n)), bias(static_cast<size_t>(n));
  std::vector<float> residual(static_cast<size_t>(m * n));
  for (auto& v : row_scale) v = r.uni(0.5f, 1.5f);
  for (auto& v : col_scale) v = r.uni(0.5f, 1.5f);
  for (auto& v : bias) v = r.uni(-1.0f, 1.0f);
  for (auto& v : residual) v = r.uni(-1.0f, 1.0f);
  auto d_row = kops_test::device_alloc<float>(row_scale.size());
  auto d_col = kops_test::device_alloc<float>(col_scale.size());
  auto d_bias = kops_test::device_alloc<float>(bias.size());
  auto d_res = kops_test::device_alloc<float>(residual.size());
  kops_test::copy_in(d_row, row_scale);
  kops_test::copy_in(d_col, col_scale);
  kops_test::copy_in(d_bias, bias);
  kops_test::copy_in(d_res, residual);
  // The residual is read in the output dtype (EpilogueParams contract), so the bf16 case needs a bf16
  // residual buffer and the reference has to use the rounded values.
  std::vector<float> residual_bf16_value(static_cast<size_t>(m * n));
  std::vector<uint8_t> residual_bf16(static_cast<size_t>(m * n) * 2);
  for (size_t i = 0; i < residual.size(); ++i) {
    const __nv_bfloat16 h = __float2bfloat16_rn(residual[i]);
    std::memcpy(&residual_bf16[i * 2], &h, 2);
    residual_bf16_value[i] = __bfloat162float(h);
  }
  auto d_res_bf16 = upload<uint16_t>(residual_bf16);

  kg::EpilogueParams epi{};
  epi.ops = kg::kEpiRowScale | kg::kEpiColScale | kg::kEpiBias | kg::kEpiClamp | kg::kEpiAct | kg::kEpiResidualAdd;
  epi.alpha = 1.5f;
  epi.beta = 0.25f;
  epi.clamp_limit = 2.0f;
  epi.act = kops::ActFn::kReLU;
  epi.row_scale = d_row;
  epi.col_scale = d_col;
  epi.bias = d_bias;
  epi.bias_dtype = DType::kFloat32;
  epi.residual = d_res_bf16.p;
  epi.residual_stride = n;
  EpiHost eh;
  eh.row_scale = &row_scale;
  eh.col_scale = &col_scale;
  eh.bias = &bias;
  eh.residual = &residual_bf16_value;
  eh.residual_stride = n;
  dense_case(ctx, DType::kBFloat16, m, n, k, 64, 128, epi, eh, 0.05, 0.01, "sm120 dense epilogue", false);
  // fp32 output: kEpiOutFp32 keeps the accumulator in fp32 (tighter tolerance, B7 comparison off
  // because the B7 tactic resolves a different tile for an fp32 D).
  kg::EpilogueParams epi32 = epi;
  epi32.ops |= kg::kEpiOutFp32;
  epi32.residual = d_res;   // fp32 output reads an fp32 residual
  EpiHost eh32 = eh;
  eh32.residual = &residual;
  {
    const Operand a = make_16(m * k, 11u, DType::kBFloat16);
    const Operand b = make_16(n * k, 12u, DType::kBFloat16);
    auto da = upload<uint16_t>(a.raw);
    auto db = upload<uint16_t>(b.raw);
    Dev<float> dout(static_cast<size_t>(m * n));
    kg::ScaledGemmArgs args = make_args(kg::GemmRecipe::kBF16, DType::kBFloat16, DType::kFloat32, m, n, k, da.p, db.p,
                                        dout.p, n);
    args.epi = epi32;
    KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm120_tactic(), kops::Workspace{}, ctx), "sm120 dense fp32 output");
    cudaDeviceSynchronize();
    const auto raw = download_bytes(dout.p, static_cast<size_t>(m * n) * 4);
    compare_close(bytes_to_values(raw, static_cast<size_t>(m * n), DType::kFloat32),
                  reference(a.value, b.value, m, n, k, args.epi, eh32, nullptr, nullptr), DType::kFloat32, 0.05, 0.01,
                  "sm120 dense fp32 output");
  }
  kops_test::device_free(d_row);
  kops_test::device_free(d_col);
  kops_test::device_free(d_bias);
  kops_test::device_free(d_res);
}

// -----------------------------------------------------------------------------
// FP8 (e4m3 tensorwise / rowwise)
// -----------------------------------------------------------------------------

void fp8_case(const LaunchContext& ctx, bool rowwise, int64_t m, int64_t n, int64_t k, int tm, int tn, double atol,
              double rtol, const char* what) {
  const Operand a = make_e4m3(m * k, static_cast<unsigned>(m * 31 + n * 13 + k * 3 + (rowwise ? 1 : 0)));
  const Operand b = make_e4m3(n * k, static_cast<unsigned>(m * 7 + n * 5 + k * 11 + 2));
  auto da = upload<uint8_t>(a.raw);
  auto db = upload<uint8_t>(b.raw);
  Rng r(static_cast<uint64_t>(m * 3 + n * 7 + k));
  std::vector<float> sa(static_cast<size_t>(rowwise ? m : 1)), sb(static_cast<size_t>(rowwise ? n : 1));
  for (auto& v : sa) v = rowwise ? r.uni(0.02f, 0.09f) : 0.0625f;
  for (auto& v : sb) v = rowwise ? r.uni(0.5f, 2.0f) : 1.0f;
  auto dsa = kops_test::device_alloc<float>(sa.size());
  auto dsb = kops_test::device_alloc<float>(sb.size());
  kops_test::copy_in(dsa, sa);
  kops_test::copy_in(dsb, sb);
  Dev<uint16_t> d_sm120(static_cast<size_t>(m * n));
  Dev<uint16_t> d_b7(static_cast<size_t>(m * n));
  const size_t out_bytes = static_cast<size_t>(m * n) * 2;
  cudaMemset(d_sm120.p, 0, out_bytes);
  cudaMemset(d_b7.p, 0, out_bytes);

  kg::ScaledGemmArgs args = make_args(rowwise ? kg::GemmRecipe::kFP8Rowwise : kg::GemmRecipe::kFP8Tensorwise,
                                      DType::kFloat8E4M3, DType::kBFloat16, m, n, k, da.p, db.p, d_sm120.p, n);
  args.a.scale = dsa;
  args.a.scale_dtype = DType::kFloat32;
  args.a.granularity = rowwise ? kops::ScaleGranularity::kPerToken : kops::ScaleGranularity::kPerTensor;
  args.b.scale = dsb;
  args.b.scale_dtype = DType::kFloat32;
  args.b.granularity = rowwise ? kops::ScaleGranularity::kPerChannel : kops::ScaleGranularity::kPerTensor;
  KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm120_tactic(tm, tn), kops::Workspace{}, ctx), what);
  auto args_b7 = args;
  args_b7.d = d_b7.p;
  KOPS_EXPECT_OK(kg::gemm_with_tactic(args_b7, b7_tactic(), kops::Workspace{}, ctx), what);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    kops_test::record_failure(what, "device error", __FILE__, __LINE__);
    return;
  }
  const auto raw = download_bytes(d_sm120.p, out_bytes);
  const std::vector<float> a_s = rowwise ? sa : std::vector<float>{sa[0]};
  const std::vector<float> b_s = rowwise ? sb : std::vector<float>{sb[0]};
  std::vector<float> a_row(static_cast<size_t>(m)), b_col(static_cast<size_t>(n));
  for (int64_t i = 0; i < m; ++i) a_row[static_cast<size_t>(i)] = a_s[rowwise ? static_cast<size_t>(i) : 0];
  for (int64_t j = 0; j < n; ++j) b_col[static_cast<size_t>(j)] = b_s[rowwise ? static_cast<size_t>(j) : 0];
  compare_close(bytes_to_values(raw, static_cast<size_t>(m * n), DType::kBFloat16),
                reference(a.value, b.value, m, n, k, args.epi, EpiHost{}, &a_row, &b_col), DType::kBFloat16, atol, rtol,
                what);
  const auto raw_b7 = download_bytes(d_b7.p, out_bytes);
  if (!same_bytes(raw, raw_b7)) {
    size_t diff = 0;
    for (size_t i = 0; i < raw.size(); ++i) diff += raw[i] != raw_b7[i];
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%zu/%zu bytes differ from the B7 mma.sync result", diff, raw.size());
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
  kops_test::device_free(dsa);
  kops_test::device_free(dsb);
}

void test_fp8(const LaunchContext& ctx) {
  for (int64_t m : {1, 7, 16, 64, 128}) {
    fp8_case(ctx, false, m, 200, 256, 64, 128, 0.03, 0.01, "sm120 fp8 tensorwise");
    fp8_case(ctx, true, m, 200, 256, 128, 128, 0.03, 0.01, "sm120 fp8 rowwise");
  }
  fp8_case(ctx, false, 40, 3, 64, 64, 128, 0.03, 0.01, "sm120 fp8 N tail");
  fp8_case(ctx, true, 40, 129, 208, 128, 128, 0.03, 0.01, "sm120 fp8 K tail");
}

// -----------------------------------------------------------------------------
// Contracts
// -----------------------------------------------------------------------------

void test_contracts(const LaunchContext& ctx) {
  const int64_t m = 40, n = 200, k = 256;
  const Operand a = make_16(m * k, 5u, DType::kBFloat16);
  const Operand b = make_16(n * k, 6u, DType::kBFloat16);
  auto da = upload<uint16_t>(a.raw);
  auto db = upload<uint16_t>(b.raw);
  std::vector<uint16_t> sentinel(static_cast<size_t>(m * n));
  const __nv_bfloat16 s7 = __float2bfloat16_rn(7.0f);
  for (auto& v : sentinel) std::memcpy(&v, &s7, 2);
  Dev<uint16_t> dout(sentinel.size());
  kops_test::copy_in(dout.p, sentinel);

  auto base = [&]() {
    return make_args(kg::GemmRecipe::kBF16, DType::kBFloat16, DType::kBFloat16, m, n, k, da.p, db.p, dout.p, n);
  };
  auto run = [&](const kg::ScaledGemmArgs& x, const kg::GemmTactic& t, const LaunchContext& c) {
    return kg::gemm_with_tactic(x, t, kops::Workspace{}, c);
  };

  // Argument errors come first, so they are reported on any exact SM.
  LaunchContext sm86 = ctx;
  sm86.arch_id = KOPS_TARGET_SM86;
  KOPS_EXPECT_STATUS(run(base(), sm120_tactic(), sm86), Status::kUnsupportedArch, "sm120: arch gate");
  KOPS_EXPECT_STATUS(run(base(), sm120_tactic(), ctx), Status::kSuccess, "sm120: baseline");
  { auto x = base(); x.m = 0; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kInvalidArgument, "sm120: M = 0"); }
  { auto x = base(); x.a.data = nullptr; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kInvalidArgument, "sm120: null A"); }
  { auto x = base(); x.b.data = nullptr; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kInvalidArgument, "sm120: null B"); }
  { auto x = base(); x.d = nullptr; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kInvalidArgument, "sm120: null D"); }
  { auto x = base(); x.a.ld = k - 8; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kInvalidArgument, "sm120: lda < K"); }
  { auto x = base(); x.ldd = n - 1; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kInvalidArgument, "sm120: ldd < N"); }
  { auto x = base(); x.a.ld = k + 4; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kMisaligned, "sm120: lda not 16 B"); }
  { auto x = base(); x.k = 260; x.a.ld = 264; x.b.ld = 264;
    KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kMisaligned, "sm120: K % 8 != 0"); }
  { auto x = base(); x.a.major = kg::MajorOrder::kMNMajor; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kNotImplemented, "sm120: MN-major A"); }
  { auto x = base(); x.workspace = dout.p; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kInvalidArgument, "sm120: args.workspace"); }
  { auto x = base(); x.epi.ops = kg::kEpiGLU; x.epi.glu_layout = kops::GateLayout::kInterleaved; x.epi.glu_interleave = 1;
    KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kNotImplemented, "sm120: GLU epilogue"); }
  { auto x = base(); x.epi.ops = kg::kEpiQuantOut; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kNotImplemented, "sm120: quantized output"); }
  { auto x = base(); x.a.scale = da.p; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kInvalidArgument, "sm120: 16-bit operand scales"); }
  { auto x = base(); x.d_dtype = DType::kFloat16; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kUnsupportedDtype, "sm120: fp16 output from bf16 operands"); }
  { auto x = base(); x.recipe = kg::GemmRecipe::kFP16; KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kUnsupportedDtype, "sm120: kFP16 with bf16 operands"); }

  // Tactic shape / scheduling contracts.
  { KOPS_EXPECT_STATUS(run(base(), sm120_tactic(64, 64), ctx), Status::kUnsupportedShape, "sm120: uncompiled tile"); }
  { KOPS_EXPECT_STATUS(run(base(), sm120_tactic(64, 128, 0, 7), ctx), Status::kUnsupportedShape, "sm120: wrong stage count"); }
  { auto t = sm120_tactic(); t.split_k_slices = 2; t.split_k_mode = kg::SplitKMode::kSerial;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kNotImplemented, "sm120: split-K"); }
  { auto t = sm120_tactic(); t.swap_ab = true; KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kNotImplemented, "sm120: swap_ab"); }
  { auto t = sm120_tactic(); t.persistent = true; KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kNotImplemented, "sm120: persistent"); }
  { auto t = sm120_tactic(); t.mainloop = kg::MainloopSchedule::kSm120BlockScaledWS;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kSuccess, "sm120: block-scaled tactic on bf16"); }

  // FP8 contracts.
  {
    const Operand a8 = make_e4m3(m * k, 7u);
    const Operand b8 = make_e4m3(n * k, 8u);
    auto da8 = upload<uint8_t>(a8.raw);
    auto db8 = upload<uint8_t>(b8.raw);
    std::vector<float> sa(static_cast<size_t>(m)), sb(static_cast<size_t>(n), 1.0f);
    auto dsa = kops_test::device_alloc<float>(sa.size());
    auto dsb = kops_test::device_alloc<float>(sb.size());
    auto fp8_base = [&]() {
      auto x = make_args(kg::GemmRecipe::kFP8Tensorwise, DType::kFloat8E4M3, DType::kBFloat16, m, n, k, da8.p, db8.p,
                         dout.p, n);
      x.a.scale = dsa;
      x.a.scale_dtype = DType::kFloat32;
      x.a.granularity = kops::ScaleGranularity::kPerTensor;
      x.b.scale = dsb;
      x.b.scale_dtype = DType::kFloat32;
      x.b.granularity = kops::ScaleGranularity::kPerTensor;
      return x;
    };
    KOPS_EXPECT_STATUS(run(fp8_base(), sm120_tactic(), ctx), Status::kSuccess, "sm120 fp8: baseline");
    { auto x = fp8_base(); x.a.dtype = DType::kBFloat16;
      KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kUnsupportedDtype, "sm120 fp8: bf16 A"); }
    { auto x = fp8_base(); x.a.fp8 = kops::Fp8Encoding::kFNUZ;
      KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kUnsupportedDtype, "sm120 fp8: FNUZ A"); }
    { auto x = fp8_base(); x.b.scale_dtype = DType::kFloat8E8M0;
      KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kUnsupportedDtype, "sm120 fp8: UE8M0 B scale"); }
    { auto x = fp8_base(); x.a.granularity = kops::ScaleGranularity::kPerBlock2D;
      KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kNotImplemented, "sm120 fp8: 1x128 block scales"); }
    { auto x = fp8_base(); x.a.scale_layout = kops::ScaleLayout::kColMajorTma;
      KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kUnsupportedLayout, "sm120 fp8: TMA scale layout"); }
    { auto x = fp8_base(); x.recipe = kg::GemmRecipe::kFP8Blockwise;
      KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kNotImplemented, "sm120 fp8: blockwise recipe"); }
    { auto x = fp8_base(); x.recipe = kg::GemmRecipe::kMXFP8;
      KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kNotImplemented, "sm120 fp8: MXFP8 recipe"); }
    kops_test::device_free(dsa);
    kops_test::device_free(dsb);
  }

  // Recipes with no SM120a body report exactly that (NVFP4 / MXFP4 are covered by test_fp4*).
  for (kg::GemmRecipe rc : {kg::GemmRecipe::kMXFP4xMXFP8, kg::GemmRecipe::kMXFP6xMXFP8,
                            kg::GemmRecipe::kINT8Rowwise, kg::GemmRecipe::kMXFP8}) {
    auto x = base();
    x.recipe = rc;
    auto t = sm120_tactic();
    if (rc == kg::GemmRecipe::kMXFP4xMXFP8 || rc == kg::GemmRecipe::kMXFP6xMXFP8 || rc == kg::GemmRecipe::kMXFP8) {
      t.mainloop = kg::MainloopSchedule::kSm120BlockScaledWS;
    }
    KOPS_EXPECT_STATUS(run(x, t, ctx), Status::kNotImplemented, "sm120: unimplemented recipe");
  }

  // A refused call must not write the output.
  if (cudaDeviceSynchronize() != cudaSuccess) kops_test::record_failure("sm120 contracts", "device error", __FILE__, __LINE__);
  const auto out = kops_test::copy_out(dout.p, sentinel.size());
  size_t touched = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    __nv_bfloat16 h;
    std::memcpy(&h, &out[i], 2);
    touched += __bfloat162float(h) != 7.0f;
  }
  // The three successful baseline calls above overwrite the buffer; anything else must not. Re-check
  // with a fresh sentinel and only refused calls.
  kops_test::copy_in(dout.p, sentinel);
  {
    auto x = base();
    x.epi.ops = kg::kEpiGLU;
    x.epi.glu_layout = kops::GateLayout::kInterleaved;
    x.epi.glu_interleave = 1;
    KOPS_EXPECT_STATUS(run(x, sm120_tactic(), ctx), Status::kNotImplemented, "sm120: refused call");
    auto y = base();
    y.k = 260;
    y.a.ld = 264;
    y.b.ld = 264;
    KOPS_EXPECT_STATUS(run(y, sm120_tactic(), ctx), Status::kMisaligned, "sm120: refused call (misaligned)");
  }
  cudaDeviceSynchronize();
  const auto after = kops_test::copy_out(dout.p, sentinel.size());
  size_t written = 0;
  for (size_t i = 0; i < after.size(); ++i) {
    __nv_bfloat16 h;
    std::memcpy(&h, &after[i], 2);
    written += __bfloat162float(h) != 7.0f;
  }
  if (written != 0) kops_test::record_failure("sm120 contracts", "a refused call wrote the output", __FILE__, __LINE__);
  (void)touched;
}

/// Strided batch: D[b] = A[b] . B[b]^T, batch in blockIdx.z. Compared with the B7 body (bitwise)
/// and with the fp64 product of the stored operands.
void test_dense_batch(const LaunchContext& ctx) {
  const int64_t m = 20, n = 200, k = 128, batch = 3;
  const Operand a = make_16(batch * m * k, 21u, DType::kBFloat16);
  const Operand b = make_16(batch * n * k, 22u, DType::kBFloat16);
  auto da = upload<uint16_t>(a.raw);
  auto db = upload<uint16_t>(b.raw);
  Dev<uint16_t> d_sm120(static_cast<size_t>(batch * m * n));
  Dev<uint16_t> d_b7(static_cast<size_t>(batch * m * n));
  const size_t out_bytes = static_cast<size_t>(batch * m * n) * 2;
  cudaMemset(d_sm120.p, 0, out_bytes);
  cudaMemset(d_b7.p, 0, out_bytes);
  kg::ScaledGemmArgs args = make_args(kg::GemmRecipe::kBF16, DType::kBFloat16, DType::kBFloat16, m, n, k, da.p, db.p,
                                      d_sm120.p, n);
  args.batch = static_cast<int32_t>(batch);
  args.a.batch_stride = m * k;
  args.b.batch_stride = n * k;
  args.batch_stride_d = m * n;
  KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm120_tactic(), kops::Workspace{}, ctx), "sm120 dense batch");
  auto args_b7 = args;
  args_b7.d = d_b7.p;
  KOPS_EXPECT_OK(kg::gemm_with_tactic(args_b7, b7_tactic(), kops::Workspace{}, ctx), "sm120 dense batch");
  if (cudaDeviceSynchronize() != cudaSuccess) {
    kops_test::record_failure("sm120 dense batch", "device error", __FILE__, __LINE__);
    return;
  }
  const auto raw = download_bytes(d_sm120.p, out_bytes);
  for (int64_t z = 0; z < batch; ++z) {
    std::vector<float> av(a.value.begin() + static_cast<std::ptrdiff_t>(z * m * k),
                          a.value.begin() + static_cast<std::ptrdiff_t>((z + 1) * m * k));
    std::vector<float> bv(b.value.begin() + static_cast<std::ptrdiff_t>(z * n * k),
                          b.value.begin() + static_cast<std::ptrdiff_t>((z + 1) * n * k));
    std::vector<uint8_t> sub(raw.begin() + static_cast<std::ptrdiff_t>(z * m * n * 2),
                             raw.begin() + static_cast<std::ptrdiff_t>((z + 1) * m * n * 2));
    compare_close(bytes_to_values(sub, static_cast<size_t>(m * n), DType::kBFloat16),
                  reference(av, bv, m, n, k, args.epi, EpiHost{}, nullptr, nullptr), DType::kBFloat16, 0.05, 0.01,
                  "sm120 dense batch");
  }
  const auto raw_b7 = download_bytes(d_b7.p, out_bytes);
  if (!same_bytes(raw, raw_b7)) {
    kops_test::record_failure("sm120 dense batch", "differs from the B7 mma.sync result", __FILE__, __LINE__);
  }
}

// -----------------------------------------------------------------------------
// FP4: NVFP4 (block 16, E4M3 scales) and MXFP4 (block 32, UE8M0 scales)
// -----------------------------------------------------------------------------

/// Packed e2m1 operand: `count` elements, two per byte (element 2i in the low nibble), plus the exact
/// value of every stored code.
struct Fp4Op {
  std::vector<uint8_t> raw;
  std::vector<float> value;
};

constexpr float kE2M1Mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

Fp4Op make_e2m1(int64_t count, unsigned seed, bool all_zero = false) {
  Rng r(seed);
  Fp4Op op;
  op.raw.assign(static_cast<size_t>(count / 2), 0);
  op.value.assign(static_cast<size_t>(count), 0.0f);
  for (int64_t i = 0; i < count; ++i) {
    const uint32_t code = all_zero ? 0u : static_cast<uint32_t>(r.next() & 0xF);
    op.value[static_cast<size_t>(i)] = (code & 8u ? -1.0f : 1.0f) * kE2M1Mag[code & 7u];
    uint8_t& byte = op.raw[static_cast<size_t>(i / 2)];
    byte = static_cast<uint8_t>((i & 1) ? (byte | (code << 4)) : (byte | code));
  }
  return op;
}

/// Row-major block scales: E4M3 bytes in [0.4, 2.0] for NVFP4, UE8M0 bytes (2^-7 .. 2^3) for MXFP4.
std::vector<uint8_t> make_block_scales(int64_t rows, int64_t blocks, unsigned seed, bool nvfp4) {
  Rng r(seed);
  std::vector<uint8_t> s(static_cast<size_t>(rows * blocks), 0);
  for (auto& v : s) {
    v = nvfp4 ? e4m3_from_float(r.uni(0.4f, 2.0f)) : static_cast<uint8_t>(120u + (r.next() % 11u));
  }
  return s;
}

float block_scale_value(uint8_t b, bool nvfp4) {
  return nvfp4 ? e4m3_to_float(b) : std::ldexp(1.0f, static_cast<int>(b) - 127);
}

/// Row-major block scales -> 128x4 swizzled tiles via the repo kernel (kFloat8E8M0 is the byte view).
kops::Status koks_swizzle(void* out, int64_t rows_pad, int64_t cols_pad, const void* in, int64_t rows, int64_t blocks,
                         const LaunchContext& ctx) {
  return kops::gemm::block_scale_swizzle(kops_test::tensor2d(out, DType::kFloat8E8M0, rows_pad, cols_pad),
                                         kops_test::tensor2d(const_cast<void*>(in), DType::kFloat8E8M0, rows, blocks),
                                         false, ctx);
}

kg::GemmTactic fp4_tactic(int tm = 64, int tn = 128, int stages = 0) {
  kg::GemmTactic t = sm120_tactic(tm, tn, 0, stages);
  t.mainloop = kg::MainloopSchedule::kSm120BlockScaledWS;
  return t;
}

/// fp64 reference of one FP4 case: per-block products scaled by sA * sB, then the global-scale divide.
std::vector<double> fp4_reference(const Fp4Op& a, const Fp4Op& b, const std::vector<float>& sa,
                                  const std::vector<float>& sb, int64_t m, int64_t n, int64_t k, int64_t block,
                                  double inv_g) {
  const int64_t blocks = k / block;
  std::vector<double> out(static_cast<size_t>(m * n), 0.0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t kb = 0; kb < blocks; ++kb) {
        double part = 0.0;
        for (int64_t kk = 0; kk < block; ++kk) {
          const int64_t kx = kb * block + kk;
          part += static_cast<double>(a.value[i * k + kx]) * b.value[j * k + kx];
        }
        acc += part * static_cast<double>(sa[static_cast<size_t>(i * blocks + kb)]) *
               static_cast<double>(sb[static_cast<size_t>(j * blocks + kb)]);
      }
      out[static_cast<size_t>(i * n + j)] = acc * inv_g;
    }
  }
  return out;
}

void fp4_case(const LaunchContext& ctx, bool nvfp4, bool swizzled, int64_t m, int64_t n, int64_t k, int tm, int tn,
              double atol, double rtol, const char* what, bool zero_operands = false) {
  const int block = nvfp4 ? 16 : 32;
  const int64_t blocks = k / block;
  const Fp4Op a = make_e2m1(m * k, static_cast<unsigned>(m * 37 + n * 11 + k * 3 + (nvfp4 ? 1 : 0)), zero_operands);
  const Fp4Op b = make_e2m1(n * k, static_cast<unsigned>(m * 5 + n * 17 + k * 7 + 2), zero_operands);
  auto raw_sa = make_block_scales(m, blocks, static_cast<unsigned>(m + 3 * n + k), nvfp4);
  auto raw_sb = make_block_scales(n, blocks, static_cast<unsigned>(m + 7 * n + 2 * k), nvfp4);
  auto da = upload<uint8_t>(a.raw);
  auto db = upload<uint8_t>(b.raw);
  // Swizzled scales are produced by the repo's own block_scale_swizzle_kernel, so this also checks that
  // sf_swizzled_offset() reads exactly what that kernel writes.
  const int64_t cols_pad = (blocks + 3) / 4 * 4;
  const int64_t rows_pad_a = (m + 127) / 128 * 128, rows_pad_b = (n + 127) / 128 * 128;
  Dev<uint8_t> dsa(static_cast<size_t>(swizzled ? rows_pad_a * cols_pad : m * blocks));
  Dev<uint8_t> dsb(static_cast<size_t>(swizzled ? rows_pad_b * cols_pad : n * blocks));
  if (swizzled) {
    auto ta = upload<uint8_t>(raw_sa);
    auto tb = upload<uint8_t>(raw_sb);
    cudaMemset(dsa.p, 0, static_cast<size_t>(rows_pad_a * cols_pad));
    cudaMemset(dsb.p, 0, static_cast<size_t>(rows_pad_b * cols_pad));
    KOPS_EXPECT_OK(koks_swizzle(dsa.p, rows_pad_a, cols_pad, ta.p, m, blocks, ctx), what);
    KOPS_EXPECT_OK(koks_swizzle(dsb.p, rows_pad_b, cols_pad, tb.p, n, blocks, ctx), what);
  } else {
    kops_test::copy_in(dsa.p, raw_sa);
    kops_test::copy_in(dsb.p, raw_sb);
  }
  // NVFP4 per-tensor global scales (dequant: divide by gA * gB).
  const float gA = 4.0f, gB = 0.5f;
  auto dga = kops_test::device_alloc<float>(1);
  auto dgb = kops_test::device_alloc<float>(1);
  kops_test::copy_in(dga, std::vector<float>{gA});
  kops_test::copy_in(dgb, std::vector<float>{gB});
  Dev<uint16_t> dout(static_cast<size_t>(m * n));
  const size_t out_bytes = static_cast<size_t>(m * n) * 2;
  cudaMemset(dout.p, 0, out_bytes);

  kg::ScaledGemmArgs args{};
  args.recipe = nvfp4 ? kg::GemmRecipe::kNVFP4 : kg::GemmRecipe::kMXFP4;
  args.a.data = da.p;
  args.a.ld = k / 2;                    // packed: byte pitch
  args.a.dtype = DType::kFloat4E2M1x2;
  args.a.packing = kops::WeightPacking::kFP4E2M1;
  args.a.scale = dsa.p;
  args.a.scale_dtype = nvfp4 ? DType::kFloat8E4M3 : DType::kFloat8E8M0;
  args.a.granularity = nvfp4 ? kops::ScaleGranularity::kNVFP4 : kops::ScaleGranularity::kMX;
  args.a.scale_layout = swizzled ? kops::ScaleLayout::kSwizzled128x4 : kops::ScaleLayout::kRowMajor;
  args.a.global_scale = nvfp4 ? dga : nullptr;
  args.b.data = db.p;
  args.b.ld = k / 2;
  args.b.dtype = DType::kFloat4E2M1x2;
  args.b.packing = kops::WeightPacking::kFP4E2M1;
  args.b.scale = dsb.p;
  args.b.scale_dtype = args.a.scale_dtype;
  args.b.granularity = args.a.granularity;
  args.b.scale_layout = args.a.scale_layout;
  args.b.global_scale = nvfp4 ? dgb : nullptr;
  args.d = dout.p;
  args.ldd = n;
  args.d_dtype = DType::kBFloat16;
  args.m = m;
  args.n = n;
  args.k = k;
  args.epi.alpha = 1.0f;

  KOPS_EXPECT_OK(kg::gemm_with_tactic(args, fp4_tactic(tm, tn), kops::Workspace{}, ctx), what);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    kops_test::record_failure(what, "device error", __FILE__, __LINE__);
    return;
  }
  std::vector<float> sa_v(static_cast<size_t>(m * blocks)), sb_v(static_cast<size_t>(n * blocks));
  for (int64_t i = 0; i < m * blocks; ++i) sa_v[static_cast<size_t>(i)] = block_scale_value(raw_sa[static_cast<size_t>(i)], nvfp4);
  for (int64_t j = 0; j < n * blocks; ++j) sb_v[static_cast<size_t>(j)] = block_scale_value(raw_sb[static_cast<size_t>(j)], nvfp4);
  const double inv_g = nvfp4 ? 1.0 / (static_cast<double>(gA) * gB) : 1.0;
  const auto raw = download_bytes(dout.p, out_bytes);
  compare_close(bytes_to_values(raw, static_cast<size_t>(m * n), DType::kBFloat16),
                fp4_reference(a, b, sa_v, sb_v, m, n, k, block, inv_g), DType::kBFloat16, atol, rtol, what);
  kops_test::device_free(dga);
  kops_test::device_free(dgb);
}

void test_fp4(const LaunchContext& ctx) {
  // Tolerance: the reference is fp64 and the kernel accumulates in fp32, so the only expected gap is the
  // final bf16 rounding of a sum whose magnitude reaches a few hundred (one bf16 ulp there is ~0.5), plus
  // the fp32 rounding of the block sums (< 1e-3). 0.3 / 1% is inside one bf16 ulp.
  // M sweep on both tiles, row-major scales, NVFP4 and MXFP4.
  for (int64_t m : {1, 7, 16, 64, 128}) {
    fp4_case(ctx, true, false, m, 200, 256, 64, 128, 0.3, 0.01, "sm120 nvfp4 rowmajor 64x128");
    fp4_case(ctx, false, false, m, 200, 256, 64, 128, 0.3, 0.01, "sm120 mxfp4 rowmajor 64x128");
    fp4_case(ctx, true, false, m, 200, 256, 128, 128, 0.3, 0.01, "sm120 nvfp4 rowmajor 128x128");
    fp4_case(ctx, false, false, m, 200, 256, 128, 128, 0.3, 0.01, "sm120 mxfp4 rowmajor 128x128");
  }
  // 128x4 swizzled scales (what block_scale_swizzle_kernel produces).
  for (int64_t m : {1, 33, 129}) {
    fp4_case(ctx, true, true, m, 200, 256, 64, 128, 0.3, 0.01, "sm120 nvfp4 swizzled");
    fp4_case(ctx, false, true, m, 200, 256, 64, 128, 0.3, 0.01, "sm120 mxfp4 swizzled");
  }
  // N / K tails: N = 1 / 3 / 129, K = one block, three blocks, and a block count that is not a multiple
  // of the swizzle width (13 blocks NVFP4, 7 MXFP4).
  for (const auto& shape : {std::pair<int64_t, int64_t>{1, 16}, {3, 48}, {129, 208}, {200, 256}}) {
    fp4_case(ctx, true, false, 40, shape.first, shape.second, 64, 128, 0.3, 0.01, "sm120 nvfp4 N/K tails");
    fp4_case(ctx, true, true, 40, shape.first, shape.second, 64, 128, 0.3, 0.01, "sm120 nvfp4 swizzled tails");
  }
  for (const auto& shape : {std::pair<int64_t, int64_t>{1, 32}, {3, 96}, {129, 224}, {200, 256}}) {
    fp4_case(ctx, false, false, 40, shape.first, shape.second, 128, 128, 0.3, 0.01, "sm120 mxfp4 N/K tails");
    fp4_case(ctx, false, true, 40, shape.first, shape.second, 128, 128, 0.3, 0.01, "sm120 mxfp4 swizzled tails");
  }
  // Zero operands with a sentinel output: the K tail and the padded rows must contribute exact zeros.
  {
    const int64_t m = 5, n = 200, k = 208;
    const int64_t ldd = n + 5, rows = m + 2;
    const float sentinel = 7.0f;
    const Fp4Op b = make_e2m1(n * k, 77u);
    auto db = upload<uint8_t>(b.raw);
    const Fp4Op a0 = make_e2m1(m * k, 78u, true);   // all codes 0
    auto da = upload<uint8_t>(a0.raw);
    const int64_t blocks = k / 16;
    auto raw_sa = make_block_scales(m, blocks, 5u, true);
    auto raw_sb = make_block_scales(n, blocks, 6u, true);
    auto dsa = upload<uint8_t>(raw_sa);
    auto dsb = upload<uint8_t>(raw_sb);
    std::vector<__nv_bfloat16> host_out(static_cast<size_t>(rows * ldd), __float2bfloat16_rn(sentinel));
    Dev<__nv_bfloat16> dout(host_out.size());
    kops_test::copy_in(dout.p, host_out);
    kg::ScaledGemmArgs args{};
    args.recipe = kg::GemmRecipe::kNVFP4;
    args.a.data = da.p;
    args.a.ld = k / 2;
    args.a.dtype = DType::kFloat4E2M1x2;
    args.a.scale = dsa.p;
    args.a.scale_dtype = DType::kFloat8E4M3;
    args.a.granularity = kops::ScaleGranularity::kNVFP4;
    args.b.data = db.p;
    args.b.ld = k / 2;
    args.b.dtype = DType::kFloat4E2M1x2;
    args.b.scale = dsb.p;
    args.b.scale_dtype = DType::kFloat8E4M3;
    args.b.granularity = kops::ScaleGranularity::kNVFP4;
    args.d = dout.p;
    args.ldd = ldd;
    args.d_dtype = DType::kBFloat16;
    args.m = m;
    args.n = n;
    args.k = k;
    KOPS_EXPECT_OK(kg::gemm_with_tactic(args, fp4_tactic(), kops::Workspace{}, ctx), "sm120 fp4 zero operands");
    if (cudaDeviceSynchronize() != cudaSuccess) {
      kops_test::record_failure("sm120 fp4 zero operands", "device error", __FILE__, __LINE__);
    } else {
      const auto got = kops_test::copy_out(dout.p, host_out.size());
      size_t bad = 0, touched = 0;
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < ldd; ++c) {
          const float v = __bfloat162float(got[static_cast<size_t>(r * ldd + c)]);
          if (r < m && c < n) {
            if (v != 0.0f) ++bad;
          } else if (v != sentinel) {
            ++touched;
          }
        }
      }
      if (bad != 0) kops_test::record_failure("sm120 fp4 zero operands", "a zero product wrote a non-zero output", __FILE__, __LINE__);
      if (touched != 0) kops_test::record_failure("sm120 fp4 zero operands", "the tail rows / columns were written", __FILE__, __LINE__);
    }
  }
}

void test_fp4_contracts(const LaunchContext& ctx) {
  const int64_t m = 40, n = 200, k = 256;
  const Fp4Op a = make_e2m1(m * k, 81u);
  const Fp4Op b = make_e2m1(n * k, 82u);
  auto da = upload<uint8_t>(a.raw);
  auto db = upload<uint8_t>(b.raw);
  auto raw_sa = make_block_scales(m, k / 16, 83u, true);
  auto raw_sb = make_block_scales(n, k / 16, 84u, true);
  auto dsa = upload<uint8_t>(raw_sa);
  auto dsb = upload<uint8_t>(raw_sb);
  Dev<uint16_t> dout(static_cast<size_t>(m * n));
  std::vector<uint16_t> sentinel(static_cast<size_t>(m * n));
  const __nv_bfloat16 s7 = __float2bfloat16_rn(7.0f);
  for (auto& v : sentinel) std::memcpy(&v, &s7, 2);
  kops_test::copy_in(dout.p, sentinel);

  auto base = [&]() {
    kg::ScaledGemmArgs x{};
    x.recipe = kg::GemmRecipe::kNVFP4;
    x.a.data = da.p;
    x.a.ld = k / 2;
    x.a.dtype = DType::kFloat4E2M1x2;
    x.a.scale = dsa.p;
    x.a.scale_dtype = DType::kFloat8E4M3;
    x.a.granularity = kops::ScaleGranularity::kNVFP4;
    x.b.data = db.p;
    x.b.ld = k / 2;
    x.b.dtype = DType::kFloat4E2M1x2;
    x.b.scale = dsb.p;
    x.b.scale_dtype = DType::kFloat8E4M3;
    x.b.granularity = kops::ScaleGranularity::kNVFP4;
    x.d = dout.p;
    x.ldd = n;
    x.d_dtype = DType::kBFloat16;
    x.m = m;
    x.n = n;
    x.k = k;
    return x;
  };
  auto run = [&](const kg::ScaledGemmArgs& x, const kg::GemmTactic& t, const LaunchContext& c) {
    return kg::gemm_with_tactic(x, t, kops::Workspace{}, c);
  };
  KOPS_EXPECT_STATUS(run(base(), fp4_tactic(), ctx), Status::kSuccess, "sm120 fp4: baseline");
  { auto x = base(); x.recipe = kg::GemmRecipe::kMXFP4;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedDtype, "fp4: MXFP4 with E4M3 scales"); }
  { auto x = base(); x.a.scale_dtype = DType::kFloat8E8M0;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedDtype, "fp4: NVFP4 with UE8M0 scales"); }
  { auto x = base(); x.a.scale = nullptr;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kInvalidArgument, "fp4: missing A scales"); }
  { auto x = base(); x.b.scale_layout = kops::ScaleLayout::kColMajorTma;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedLayout, "fp4: TMA column-major scales"); }
  { auto x = base(); x.a.scale_layout = kops::ScaleLayout::kPackedUE8M0x4;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedLayout, "fp4: packed UE8M0 scales"); }
  { auto x = base(); x.a.granularity = kops::ScaleGranularity::kMX;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kInvalidArgument, "fp4: NVFP4 with MX granularity"); }
  { auto x = base(); x.a.group_k = 32;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kInvalidArgument, "fp4: NVFP4 with group_k 32"); }
  { auto x = base(); x.k = 264;   // 264 % 16 == 8
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedShape, "fp4: K % block != 0"); }
  { auto x = base(); x.a.ld = k / 2 - 8;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kInvalidArgument, "fp4: lda < K/2"); }
  { auto x = base(); x.a.ld = k / 2 + 2;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kMisaligned, "fp4: lda not 4 B"); }
  { auto x = base(); x.a.packing = kops::WeightPacking::kMarlin;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedLayout, "fp4: Marlin packing"); }
  { auto x = base(); x.d_dtype = DType::kFloat8E4M3;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedDtype, "fp4: 8-bit output"); }
  { auto x = base(); x.a.dtype = DType::kFloat8E4M3;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedDtype, "fp4: e4m3 operand"); }
  { auto x = base(); x.b.global_scale = static_cast<const float*>(nullptr);
    x.recipe = kg::GemmRecipe::kMXFP4; x.a.scale_dtype = DType::kFloat8E8M0; x.b.scale_dtype = DType::kFloat8E8M0;
    x.a.granularity = kops::ScaleGranularity::kMX; x.b.granularity = kops::ScaleGranularity::kMX;
    auto t = fp4_tactic();
    KOPS_EXPECT_STATUS(run(x, t, ctx), Status::kSuccess, "fp4: MXFP4 baseline"); }
  { auto x = base(); x.recipe = kg::GemmRecipe::kMXFP4;
    x.a.scale_dtype = DType::kFloat8E8M0; x.b.scale_dtype = DType::kFloat8E8M0;
    x.a.granularity = kops::ScaleGranularity::kMX; x.b.granularity = kops::ScaleGranularity::kMX;
    const float one = 1.0f;
    auto dg = kops_test::device_alloc<float>(1);
    kops_test::copy_in(dg, std::vector<float>{one});
    x.a.global_scale = dg;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kInvalidArgument, "fp4: MXFP4 with a global scale");
    kops_test::device_free(dg); }
  { auto x = base(); x.recipe = kg::GemmRecipe::kMXFP4xMXFP8;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kNotImplemented, "fp4: mixed MX recipe"); }
  { auto x = base(); x.recipe = kg::GemmRecipe::kMXFP6xMXFP8;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kNotImplemented, "fp4: MXFP6 recipe"); }
  LaunchContext sm86 = ctx;
  sm86.arch_id = KOPS_TARGET_SM86;
  KOPS_EXPECT_STATUS(run(base(), fp4_tactic(), sm86), Status::kUnsupportedArch, "fp4: arch gate");
  { auto t = fp4_tactic(); t.split_k_slices = 2; t.split_k_mode = kg::SplitKMode::kSerial;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kNotImplemented, "fp4: split-K"); }
  { KOPS_EXPECT_STATUS(run(base(), fp4_tactic(64, 64), ctx), Status::kUnsupportedShape, "fp4: uncompiled tile"); }

  // A refused call must not write the output.
  cudaDeviceSynchronize();
  kops_test::copy_in(dout.p, sentinel);
  { auto x = base(); x.a.scale_layout = kops::ScaleLayout::kColMajorTma;
    KOPS_EXPECT_STATUS(run(x, fp4_tactic(), ctx), Status::kUnsupportedLayout, "fp4: refused call"); }
  cudaDeviceSynchronize();
  const auto after = kops_test::copy_out(dout.p, sentinel.size());
  size_t written = 0;
  for (size_t i = 0; i < after.size(); ++i) {
    __nv_bfloat16 h;
    std::memcpy(&h, &after[i], 2);
    written += __bfloat162float(h) != 7.0f;
  }
  if (written != 0) kops_test::record_failure("fp4 contracts", "a refused call wrote the output", __FILE__, __LINE__);
}

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
  if (ctx.arch_id != KOPS_TARGET_SM120A) {
    std::printf("SKIP test_sm120: the SM120a cells need KOPS_TARGET_SM120A (found arch %d)\n", ctx.arch_id);
    return 77;
  }
  test_dense_sweep(ctx);
  test_dense_tails(ctx);
  test_dense_zero_and_guard(ctx);
  test_dense_epilogue(ctx);
  test_dense_batch(ctx);
  test_fp8(ctx);
  test_fp4(ctx);
  test_contracts(ctx);
  test_fp4_contracts(ctx);
  return kops_test::finish("test_sm120");
}
