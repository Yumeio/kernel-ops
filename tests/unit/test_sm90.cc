// =============================================================================
// kernel-ops | tests/unit/test_sm90.cc
// B11 SM90a arch cell (csrc/gemm/sm90a_impl.cuh + csrc/gemm/gemm_wgmma.cu) through the public
// surface:
//   gemm_wgmma_kernel            bf16 / fp16 dense on TMA + wgmma
//   fp8_gemm_wgmma_kernel        e4m3 x e4m3: tensorwise / rowwise (fast and promoted accumulation) and
//                                the DeepSeek 1x128 x 128x128 blockwise recipe (fp32 and UE8M0 scales)
//   grouped_gemm_wgmma_kernel    kContiguous / kMasked, bf16 and FP8 blockwise, graph replay with changing
//                                tokens per expert
// reached with gemm_with_tactic() + a tactic whose mainloop is kTmaWarpSpecialized (64-row tiles)
// or kTmaWarpSpecializedCooperative (128 / 256-row tiles).
// Oracles:
//   (a) the fp64 host product of the *stored* operand values (bf16 / fp16 rounded) with the
//       epilogue applied in the documented order;
//   (b) tile invariance: every compiled tile must return the same bytes (same k16 atom chain);
//   (c) the B7 mma.sync body (kMultistageMmaSync tactic) within tolerance (a different atom, so
//       not bitwise).
// Classes (ROADMAP 2.4): (b) contract checks run on every device: argument errors first, then
// kUnsupportedArch on any exact SM but 90a, and a refused call writes nothing. (c) numerics and
// (d) system (CUDA graph replay, determinism) need KOPS_TARGET_SM90A and return 77 elsewhere.
// Tier: standard.
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

/// One operand as the kernel reads it: the stored bytes and the exact value of every element.
struct Operand {
  std::vector<uint8_t> raw;
  std::vector<float> value;
};

/// 16-bit operand [rows, ld] holding `cols` valid elements per row (the pitch tail is poisoned with
/// NaN so that a kernel reading past K would show up).
Operand make_16(int64_t rows, int64_t cols, int64_t ld, unsigned seed, DType dt, float lo = -1.0f, float hi = 1.0f) {
  Rng r(seed);
  Operand op;
  op.raw.resize(static_cast<size_t>(rows * ld) * 2);
  op.value.resize(static_cast<size_t>(rows * cols));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < ld; ++j) {
      const float x = j < cols ? r.uni(lo, hi) : std::nanf("");
      uint16_t bits;
      float stored;
      if (dt == DType::kFloat16) {
        const __half h = __float2half_rn(x);
        std::memcpy(&bits, &h, 2);
        stored = __half2float(h);
      } else {
        const __nv_bfloat16 h = __float2bfloat16_rn(x);
        std::memcpy(&bits, &h, 2);
        stored = __bfloat162float(h);
      }
      std::memcpy(&op.raw[static_cast<size_t>(i * ld + j) * 2], &bits, 2);
      if (j < cols) op.value[static_cast<size_t>(i * cols + j)] = stored;
    }
  }
  return op;
}

struct DevBytes {
  uint8_t* p = nullptr;
  size_t n = 0;
  explicit DevBytes(size_t bytes) : p(kops_test::device_alloc<uint8_t>(bytes == 0 ? 1 : bytes)), n(bytes) {}
  ~DevBytes() { kops_test::device_free(p); }
  DevBytes(const DevBytes&) = delete;
  DevBytes& operator=(const DevBytes&) = delete;
  DevBytes(DevBytes&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
};

DevBytes upload(const std::vector<uint8_t>& raw) {
  DevBytes d(raw.size());
  const cudaError_t err = cudaMemcpy(d.p, raw.data(), raw.size(), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) kops_test::record_failure("cudaMemcpy H2D", cudaGetErrorString(err), __FILE__, __LINE__);
  return d;
}

std::vector<uint8_t> download(const void* p, size_t bytes) {
  std::vector<uint8_t> out(bytes);
  const cudaError_t err = cudaMemcpy(out.data(), p, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) kops_test::record_failure("cudaMemcpy D2H", cudaGetErrorString(err), __FILE__, __LINE__);
  return out;
}

int dt_bytes(DType dt) { return dt == DType::kFloat32 ? 4 : 2; }

float load_value(const uint8_t* p, DType dt) {
  if (dt == DType::kFloat32) {
    float f;
    std::memcpy(&f, p, 4);
    return f;
  }
  if (dt == DType::kFloat16) {
    __half h;
    std::memcpy(&h, p, 2);
    return __half2float(h);
  }
  __nv_bfloat16 h;
  std::memcpy(&h, p, 2);
  return __bfloat162float(h);
}

float round_value(double v, DType dt) {
  const float f = static_cast<float>(v);
  if (dt == DType::kFloat32) return f;
  if (dt == DType::kFloat16) return __half2float(__float2half_rn(f));
  return __bfloat162float(__float2bfloat16_rn(f));
}

/// Compiled tiles of the cell and the mainloop that goes with the tile height.
/// (cm, cn) is the cluster: 1x1, 1x2 (A tile multicast) or 2x1 (B tile multicast); the two-CTA clusters
/// exist for the 128 / 256-row tiles with a 16-bit output.
struct Tile {
  int tm, tn, cm, cn;
};
const Tile kTiles[] = {{64, 128, 1, 1},  {64, 256, 1, 1},  {128, 128, 1, 1}, {128, 256, 1, 1}, {256, 128, 1, 1},
                       {128, 128, 1, 2}, {128, 128, 2, 1}, {128, 256, 1, 2}, {128, 256, 2, 1}, {256, 128, 1, 2},
                       {256, 128, 2, 1}};

kg::GemmTactic sm90_tactic(int tm = 128, int tn = 128, int stages = 0, int cm = 1, int cn = 1) {
  kg::GemmTactic t{};
  t.tile_m = static_cast<int16_t>(tm);
  t.tile_n = static_cast<int16_t>(tn);
  t.tile_k = 64;
  t.cluster_m = static_cast<int8_t>(cm);
  t.cluster_n = static_cast<int8_t>(cn);
  t.tma_multicast = cm * cn > 1;
  t.stages = static_cast<int8_t>(stages);
  t.mainloop = tm <= 64 ? kg::MainloopSchedule::kTmaWarpSpecialized
                        : kg::MainloopSchedule::kTmaWarpSpecializedCooperative;
  t.epilogue = kg::EpilogueSchedule::kTmaStore;
  t.persistent = true;
  return t;
}

/// B7 reference tactic: the shared mma.sync body.
kg::GemmTactic b7_tactic() {
  kg::GemmTactic t{};
  t.tile_m = 64;
  t.tile_n = 128;
  t.mainloop = kg::MainloopSchedule::kMultistageMmaSync;
  t.epilogue = kg::EpilogueSchedule::kDirectStore;
  return t;
}

kg::ScaledGemmArgs make_args(DType dt, DType out_dt, int64_t m, int64_t n, int64_t k, const void* a, int64_t lda,
                             const void* b, int64_t ldb, void* d, int64_t ldd) {
  kg::ScaledGemmArgs x{};
  x.recipe = dt == DType::kFloat16 ? kg::GemmRecipe::kFP16 : kg::GemmRecipe::kBF16;
  x.a.data = a;
  x.a.ld = lda;
  x.a.dtype = dt;
  x.b.data = b;
  x.b.ld = ldb;
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

/// Host copies of the epilogue operands (EpilogueParams carries device pointers).
struct EpiHost {
  const std::vector<float>* row_scale = nullptr;
  const std::vector<float>* col_scale = nullptr;
  const std::vector<float>* bias = nullptr;       // [n]
  const std::vector<float>* residual = nullptr;   // [m, residual_stride], values as stored in the output dtype
  int64_t residual_stride = 0;
};

/// --mutate: the oracle is deliberately wrong (2 % off), so every numeric check must go red. It proves
/// that the comparisons can fail before a kernel regression has to.
bool g_mutate = false;

/// fp64 product of the stored operand values plus the epilogue in the documented order.
std::vector<double> reference(const std::vector<float>& av, const std::vector<float>& bv, int64_t m, int64_t n,
                             int64_t k, const kg::EpilogueParams& e, const EpiHost& eh) {
  std::vector<double> out(static_cast<size_t>(m * n), 0.0);
  for (int64_t i = 0; i < m; ++i) {
    const float* ar = &av[static_cast<size_t>(i * k)];
    for (int64_t j = 0; j < n; ++j) {
      const float* br = &bv[static_cast<size_t>(j * k)];
      double acc = 0.0;
      for (int64_t kk = 0; kk < k; ++kk) acc += static_cast<double>(ar[kk]) * br[kk];
      double v = acc * e.alpha;
      if ((e.ops & kg::kEpiRowScale) != 0u) v *= (*eh.row_scale)[static_cast<size_t>(i)];
      if ((e.ops & kg::kEpiColScale) != 0u) v *= (*eh.col_scale)[static_cast<size_t>(j)];
      if ((e.ops & kg::kEpiBias) != 0u) v += (*eh.bias)[static_cast<size_t>(j)];
      if ((e.ops & kg::kEpiClamp) != 0u) v = std::fmin(std::fmax(v, -e.clamp_limit), e.clamp_limit);
      if ((e.ops & kg::kEpiSoftcap) != 0u) v = static_cast<double>(e.softcap) * std::tanh(v / e.softcap);
      if ((e.ops & kg::kEpiAct) != 0u && e.act == kops::ActFn::kReLU) v = std::fmax(v, 0.0);
      if ((e.ops & kg::kEpiResidualAdd) != 0u) {
        v = static_cast<double>(e.beta) * (*eh.residual)[static_cast<size_t>(i * eh.residual_stride + j)] + v;
      }
      out[static_cast<size_t>(i * n + j)] = g_mutate ? v * 1.02 + 0.05 : v;
    }
  }
  return out;
}

/// Compares the [m, n] block of a [*, ldd] output against the reference after the one rounding to the
/// output dtype. Returns the number of mismatches.
size_t compare_block(const std::vector<uint8_t>& raw, int64_t m, int64_t n, int64_t ldd, DType out_dt,
                     const std::vector<double>& want, double atol, double rtol, const char* what) {
  const int es = dt_bytes(out_dt);
  size_t bad = 0, worst_i = 0;
  double worst = 0.0, worst_got = 0.0, worst_want = 0.0;
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      const double got = load_value(&raw[static_cast<size_t>(i * ldd + j) * es], out_dt);
      const double w = round_value(want[static_cast<size_t>(i * n + j)], out_dt);
      const double err = std::fabs(got - w);
      if (!(err <= atol + rtol * std::fabs(w))) {   // also catches NaN
        ++bad;
        if (!(err <= worst)) {
          worst = err;
          worst_i = static_cast<size_t>(i * n + j);
          worst_got = got;
          worst_want = w;
        }
      }
    }
  }
  if (bad != 0) {
    char detail[256];
    std::snprintf(detail, sizeof(detail), "%zu/%lld mismatched, worst (%zu, %zu): got %.8g want %.8g (err %.3g)", bad,
                  static_cast<long long>(m * n), worst_i / static_cast<size_t>(n), worst_i % static_cast<size_t>(n),
                  worst_got, worst_want, worst);
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
  return bad;
}

double rms_of(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x * x;
  return v.empty() ? 0.0 : std::sqrt(s / static_cast<double>(v.size()));
}

// FP8 on wgmma accumulates with reduced precision inside the instruction (measured on H100, K = 7168 /
// 18432, error relative to the rms of the result; see test_fp8_accumulation_precision):
//   kFastAccum : rms 1.7e-3 / 3.1e-3, max 1.5e-2 / 3.1e-2  (grows with K)
//   kPromoted  : rms 1.0e-4 / 1.1e-4, max 4.7e-4 / 3.9e-4  (the 128-K block sum is still a wgmma sum)
// The FP8 tolerances are therefore absolute bounds proportional to the rms of the reference, with a
// margin over the measured maxima, plus the rounding of a 16-bit output.
constexpr double kFp8FastRel = 6e-2;
constexpr double kFp8PromotedRel = 3e-3;

bool sync_ok(const char* what) {
  // KOPS_TEST_TRACE=1 names every launch before its synchronization (a hang then names its case).
  static const bool trace = std::getenv("KOPS_TEST_TRACE") != nullptr;
  if (trace) std::fprintf(stderr, "[trace] %s\n", what);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess) return true;
  kops_test::record_failure(what, cudaGetErrorString(e), __FILE__, __LINE__);
  return false;
}

// Tolerance of a 16-bit dense product of operands in [-1, 1]: the fp32 accumulation error is far
// below the half ulp of the 16-bit output, which the rounded reference already accounts for; one
// output ulp of slack covers a rounding tie decided differently in fp32 and fp64.
constexpr double kAtol = 2e-3;
constexpr double kRtol = 8e-3;

// -----------------------------------------------------------------------------
// BF16 / FP16 dense
// -----------------------------------------------------------------------------

/// One problem on every compiled tile: fp64 oracle, tile invariance, B7 cross-check.
void dense_case(const LaunchContext& ctx, DType dt, DType out_dt, int64_t m, int64_t n, int64_t k, const char* what,
                bool with_b7 = true) {
  const int64_t ldk = (k + 7) / 8 * 8;             // TMA row pitch: 16 bytes
  const Operand a = make_16(m, k, ldk, static_cast<unsigned>(m * 131 + n * 7 + k), dt);
  const Operand b = make_16(n, k, ldk, static_cast<unsigned>(m * 17 + n * 3 + k * 5 + 1), dt);
  const DevBytes da = upload(a.raw);
  const DevBytes db = upload(b.raw);
  const size_t out_bytes = static_cast<size_t>(m * n) * dt_bytes(out_dt);
  const std::vector<double> want = reference(a.value, b.value, m, n, k, kg::EpilogueParams{}, EpiHost{});

  std::vector<uint8_t> first;
  char label[160];
  for (const Tile& tile : kTiles) {
    DevBytes dout(out_bytes);
    cudaMemset(dout.p, 0xFF, out_bytes);
    kg::ScaledGemmArgs args = make_args(dt, out_dt, m, n, k, da.p, ldk, db.p, ldk, dout.p, n);
    if (tile.cm * tile.cn > 1 && out_dt == DType::kFloat32) continue;   // clusters: 16-bit output only
    std::snprintf(label, sizeof(label), "%s [%lldx%lldx%lld tile %dx%d cluster %dx%d]", what, static_cast<long long>(m),
                  static_cast<long long>(n), static_cast<long long>(k), tile.tm, tile.tn, tile.cm, tile.cn);
    KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm90_tactic(tile.tm, tile.tn, 0, tile.cm, tile.cn), kops::Workspace{}, ctx), label);
    if (!sync_ok(label)) return;
    const std::vector<uint8_t> raw = download(dout.p, out_bytes);
    compare_block(raw, m, n, n, out_dt, want, kAtol, kRtol, label);
    if (first.empty()) {
      first = raw;
    } else if (raw != first) {
      size_t diff = 0;
      for (size_t i = 0; i < raw.size(); ++i) diff += raw[i] != first[i];
      char detail[128];
      std::snprintf(detail, sizeof(detail), "%zu/%zu bytes differ from tile 64x128 (tile invariance)", diff, raw.size());
      kops_test::record_failure(label, detail, __FILE__, __LINE__);
    }
  }
  if (with_b7 && out_dt == dt) {
    DevBytes dref(out_bytes);
    kg::ScaledGemmArgs args = make_args(dt, out_dt, m, n, k, da.p, ldk, db.p, ldk, dref.p, n);
    DevBytes scratch(64u << 20);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = scratch.n;
    std::snprintf(label, sizeof(label), "%s [B7 cross-check %lldx%lldx%lld]", what, static_cast<long long>(m),
                  static_cast<long long>(n), static_cast<long long>(k));
    KOPS_EXPECT_OK(kg::gemm_with_tactic(args, b7_tactic(), ws, ctx), label);
    if (!sync_ok(label)) return;
    const std::vector<uint8_t> ref = download(dref.p, out_bytes);
    size_t bad = 0;
    for (int64_t i = 0; i < m * n; ++i) {
      const double g = load_value(&first[static_cast<size_t>(i) * 2], dt);
      const double w = load_value(&ref[static_cast<size_t>(i) * 2], dt);
      if (!(std::fabs(g - w) <= kAtol + kRtol * std::fabs(w))) ++bad;
    }
    if (bad != 0) {
      char detail[128];
      std::snprintf(detail, sizeof(detail), "%zu elements differ from the B7 mma.sync result beyond tolerance", bad);
      kops_test::record_failure(label, detail, __FILE__, __LINE__);
    }
  }
}

void test_dense_sweep(const LaunchContext& ctx) {
  // M sweep: rows below one wgmma (64), the two-warpgroup split (65..128) and the second wave (129..256).
  for (DType dt : {DType::kBFloat16, DType::kFloat16}) {
    for (int64_t m : {1, 7, 63, 64, 65, 128, 129, 200, 300}) {
      dense_case(ctx, dt, dt, m, 200, 256, "sm90 dense M sweep");
    }
  }
  // N and K tails: N around the 64-column store box and the 128 / 256 tile, K around the 64-element
  // stage and the 16-element atom.
  for (const auto& s : {std::pair<int64_t, int64_t>{1, 8}, {3, 40}, {63, 64}, {64, 72}, {65, 200}, {129, 264},
                        {257, 16}, {600, 1000}}) {
    dense_case(ctx, DType::kBFloat16, DType::kBFloat16, 40, s.first, s.second, "sm90 dense N/K tails");
    dense_case(ctx, DType::kFloat16, DType::kFloat16, 150, s.first, s.second, "sm90 dense N/K tails");
  }
  // fp32 output (direct-store epilogue).
  dense_case(ctx, DType::kBFloat16, DType::kFloat32, 77, 300, 320, "sm90 dense fp32 out", false);
  dense_case(ctx, DType::kFloat16, DType::kFloat32, 200, 129, 72, "sm90 dense fp32 out", false);
}

/// More tiles than SMs: every CTA runs several persistent iterations and the raster covers each tile
/// exactly once.
void test_dense_persistent(const LaunchContext& ctx) {
  dense_case(ctx, DType::kBFloat16, DType::kBFloat16, 1100, 1500, 320, "sm90 dense persistent");
  dense_case(ctx, DType::kBFloat16, DType::kBFloat16, 4096, 96, 128, "sm90 dense tall");
  dense_case(ctx, DType::kBFloat16, DType::kBFloat16, 96, 4096, 128, "sm90 dense wide");
}

/// Zero A with a sentinel output larger than [M, N]: padding rows / columns and the K tail contribute
/// exact zeros and nothing outside [0, M) x [0, N) is written, on both store paths.
void test_dense_zero_and_guard(const LaunchContext& ctx) {
  const int64_t m = 70, n = 200, k = 200, ldk = 200;
  const float sentinel = 7.0f;
  const Operand b = make_16(n, k, ldk, 99u, DType::kBFloat16);
  const DevBytes db = upload(b.raw);
  const DevBytes da = upload(std::vector<uint8_t>(static_cast<size_t>(m * ldk) * 2, 0));
  for (const int64_t ldd : {n + 8, n + 7}) {         // 16-byte pitch (TMA store) and odd pitch (direct store)
    for (const Tile& tile : kTiles) {
      const int64_t rows = m + 3;
      std::vector<__nv_bfloat16> host(static_cast<size_t>(rows * ldd), __float2bfloat16_rn(sentinel));
      DevBytes dout(host.size() * 2);
      cudaMemcpy(dout.p, host.data(), host.size() * 2, cudaMemcpyHostToDevice);
      kg::ScaledGemmArgs args = make_args(DType::kBFloat16, DType::kBFloat16, m, n, k, da.p, ldk, db.p, ldk, dout.p, ldd);
      KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm90_tactic(tile.tm, tile.tn, 0, tile.cm, tile.cn), kops::Workspace{}, ctx), "sm90 zero operands");
      if (!sync_ok("sm90 zero operands")) return;
      const std::vector<uint8_t> raw = download(dout.p, host.size() * 2);
      size_t nonzero = 0, touched = 0;
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < ldd; ++c) {
          const float v = load_value(&raw[static_cast<size_t>(r * ldd + c) * 2], DType::kBFloat16);
          if (r < m && c < n) {
            nonzero += v != 0.0f;                    // zero x random weights = exact zero (and never NaN)
          } else {
            touched += v != sentinel;
          }
        }
      }
      char detail[128];
      std::snprintf(detail, sizeof(detail), "tile %dx%d cluster %dx%d ldd %lld: %zu non-zero outputs, %zu guard cells written",
                    tile.tm, tile.tn, tile.cm, tile.cn, static_cast<long long>(ldd), nonzero, touched);
      if (nonzero != 0 || touched != 0) kops_test::record_failure("sm90 zero operands / guard", detail, __FILE__, __LINE__);
    }
  }
}

/// An output whose base is not 16-byte aligned takes the direct-store epilogue and must agree bitwise
/// with the TMA-store result.
void test_dense_unaligned_out(const LaunchContext& ctx) {
  const int64_t m = 130, n = 264, k = 136, ldk = 136;
  const Operand a = make_16(m, k, ldk, 11u, DType::kBFloat16);
  const Operand b = make_16(n, k, ldk, 12u, DType::kBFloat16);
  const DevBytes da = upload(a.raw);
  const DevBytes db = upload(b.raw);
  const size_t bytes = static_cast<size_t>(m * n) * 2;
  DevBytes d0(bytes), d1(bytes + 16);
  for (const Tile& tile : kTiles) {
    kg::ScaledGemmArgs x = make_args(DType::kBFloat16, DType::kBFloat16, m, n, k, da.p, ldk, db.p, ldk, d0.p, n);
    KOPS_EXPECT_OK(kg::gemm_with_tactic(x, sm90_tactic(tile.tm, tile.tn, 0, tile.cm, tile.cn), kops::Workspace{}, ctx), "sm90 aligned out");
    x.d = d1.p + 2;                                  // 2-byte aligned only
    KOPS_EXPECT_OK(kg::gemm_with_tactic(x, sm90_tactic(tile.tm, tile.tn, 0, tile.cm, tile.cn), kops::Workspace{}, ctx), "sm90 unaligned out");
    if (!sync_ok("sm90 unaligned out")) return;
    if (download(d0.p, bytes) != download(d1.p + 2, bytes)) {
      kops_test::record_failure("sm90 unaligned out", "direct-store and TMA-store results differ", __FILE__, __LINE__);
    }
  }
}

void test_dense_epilogue(const LaunchContext& ctx) {
  const int64_t m = 133, n = 200, k = 256;
  Rng r(4242);
  std::vector<float> row_scale(static_cast<size_t>(m)), col_scale(static_cast<size_t>(n)), bias(static_cast<size_t>(n));
  for (auto& v : row_scale) v = r.uni(0.5f, 1.5f);
  for (auto& v : col_scale) v = r.uni(0.5f, 1.5f);
  for (auto& v : bias) v = r.uni(-1.0f, 1.0f);
  float* d_row = kops_test::device_alloc<float>(row_scale.size());
  float* d_col = kops_test::device_alloc<float>(col_scale.size());
  float* d_bias = kops_test::device_alloc<float>(bias.size());
  kops_test::copy_in(d_row, row_scale);
  kops_test::copy_in(d_col, col_scale);
  kops_test::copy_in(d_bias, bias);

  const Operand a = make_16(m, k, k, 21u, DType::kBFloat16);
  const Operand b = make_16(n, k, k, 22u, DType::kBFloat16);
  const DevBytes da = upload(a.raw);
  const DevBytes db = upload(b.raw);

  struct Case {
    const char* name;
    uint32_t ops;
    DType out;
    float alpha;
  };
  const Case cases[] = {
      {"alpha", kg::kEpiNone, DType::kBFloat16, 0.125f},
      {"alpha fp32", kg::kEpiNone, DType::kFloat32, -0.5f},
      {"bias", kg::kEpiBias, DType::kBFloat16, 1.0f},
      {"row+col scale", kg::kEpiRowScale | kg::kEpiColScale, DType::kBFloat16, 1.0f},
      {"bias+clamp", kg::kEpiBias | kg::kEpiClamp, DType::kBFloat16, 1.0f},
      {"softcap", kg::kEpiSoftcap, DType::kBFloat16, 1.0f},
      {"bias+relu", kg::kEpiBias | kg::kEpiAct, DType::kBFloat16, 1.0f},
      {"residual", kg::kEpiResidualAdd, DType::kBFloat16, 1.0f},
      {"all fp32", kg::kEpiRowScale | kg::kEpiColScale | kg::kEpiBias | kg::kEpiClamp | kg::kEpiAct |
                       kg::kEpiResidualAdd | kg::kEpiOutFp32, DType::kFloat32, 0.25f},
  };
  for (const Case& c : cases) {
    const int es = dt_bytes(c.out);
    // Residual in the output dtype, values as stored.
    std::vector<uint8_t> res_raw(static_cast<size_t>(m * n) * es);
    std::vector<float> res_val(static_cast<size_t>(m * n));
    for (size_t i = 0; i < res_val.size(); ++i) {
      const float x = r.uni(-2.0f, 2.0f);
      if (c.out == DType::kFloat32) {
        std::memcpy(&res_raw[i * 4], &x, 4);
        res_val[i] = x;
      } else {
        const __nv_bfloat16 h = __float2bfloat16_rn(x);
        std::memcpy(&res_raw[i * 2], &h, 2);
        res_val[i] = __bfloat162float(h);
      }
    }
    const DevBytes dres = upload(res_raw);
    kg::EpilogueParams e{};
    e.ops = c.ops;
    e.alpha = c.alpha;
    e.row_scale = d_row;
    e.col_scale = d_col;
    e.bias = d_bias;
    e.bias_dtype = DType::kFloat32;
    e.clamp_limit = 3.0f;
    e.softcap = 4.0f;
    e.act = kops::ActFn::kReLU;
    e.beta = 0.5f;
    e.residual = dres.p;
    e.residual_stride = n;
    EpiHost eh;
    eh.row_scale = &row_scale;
    eh.col_scale = &col_scale;
    eh.bias = &bias;
    eh.residual = &res_val;
    eh.residual_stride = n;
    const std::vector<double> want = reference(a.value, b.value, m, n, k, e, eh);
    for (const Tile& tile : kTiles) {
      if (tile.cm * tile.cn > 1 && c.out == DType::kFloat32) continue;
      const size_t out_bytes = static_cast<size_t>(m * n) * es;
      DevBytes dout(out_bytes);
      cudaMemset(dout.p, 0xFF, out_bytes);
      kg::ScaledGemmArgs args = make_args(DType::kBFloat16, c.out, m, n, k, da.p, k, db.p, k, dout.p, n);
      args.epi = e;
      char label[128];
      std::snprintf(label, sizeof(label), "sm90 epilogue %s [tile %dx%d cluster %dx%d]", c.name, tile.tm, tile.tn, tile.cm,
                    tile.cn);
      KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm90_tactic(tile.tm, tile.tn, 0, tile.cm, tile.cn), kops::Workspace{}, ctx), label);
      if (!sync_ok(label)) return;
      compare_block(download(dout.p, out_bytes), m, n, n, c.out, want, 4e-3, 8e-3, label);
    }
  }
  kops_test::device_free(d_row);
  kops_test::device_free(d_col);
  kops_test::device_free(d_bias);
}

/// (d) system: the launch is graph safe (tensor maps are kernel parameters) and deterministic. The
/// graph is replayed after the inputs change in place.
void test_graph_and_determinism(const LaunchContext& ctx) {
  const int64_t m = 300, n = 520, k = 192;
  Operand a = make_16(m, k, k, 31u, DType::kBFloat16);
  const Operand b = make_16(n, k, k, 32u, DType::kBFloat16);
  DevBytes da = upload(a.raw);
  const DevBytes db = upload(b.raw);
  const size_t bytes = static_cast<size_t>(m * n) * 2;
  DevBytes dout(bytes);

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    kops_test::record_failure("sm90 graph", "cudaStreamCreate", __FILE__, __LINE__);
    return;
  }
  LaunchContext gctx = ctx;
  gctx.stream = stream;
  kg::ScaledGemmArgs args = make_args(DType::kBFloat16, DType::kBFloat16, m, n, k, da.p, k, db.p, k, dout.p, n);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess;
  KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm90_tactic(128, 256, 0, 1, 2), kops::Workspace{}, gctx), "sm90 graph capture");
  ok = ok && cudaStreamEndCapture(stream, &graph) == cudaSuccess;
  ok = ok && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
  if (!ok) {
    kops_test::record_failure("sm90 graph", "capture / instantiate failed", __FILE__, __LINE__);
  } else {
    for (int round = 0; round < 3; ++round) {
      // New A values in the same buffer: the replay must follow them.
      a = make_16(m, k, k, 100u + static_cast<unsigned>(round), DType::kBFloat16);
      cudaMemcpy(da.p, a.raw.data(), a.raw.size(), cudaMemcpyHostToDevice);
      cudaMemset(dout.p, 0xFF, bytes);
      if (cudaGraphLaunch(exec, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        kops_test::record_failure("sm90 graph", "replay failed", __FILE__, __LINE__);
        break;
      }
      const std::vector<uint8_t> replay = download(dout.p, bytes);
      compare_block(replay, m, n, n, DType::kBFloat16,
                    reference(a.value, b.value, m, n, k, kg::EpilogueParams{}, EpiHost{}), kAtol, kRtol, "sm90 graph replay");
      // Determinism: an eager launch of the same problem returns the same bytes.
      cudaMemset(dout.p, 0xFF, bytes);
      KOPS_EXPECT_OK(kg::gemm_with_tactic(args, sm90_tactic(128, 256), kops::Workspace{}, ctx), "sm90 eager rerun");
      if (!sync_ok("sm90 eager rerun")) break;
      if (download(dout.p, bytes) != replay) {
        kops_test::record_failure("sm90 determinism", "graph replay and eager launch differ", __FILE__, __LINE__);
      }
    }
  }
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

// -----------------------------------------------------------------------------
// FP8 (fp8_gemm_wgmma_kernel)
// -----------------------------------------------------------------------------

uint8_t e4m3_from_float(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
float e4m3_to_float(uint8_t b) { return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)); }

/// e4m3 operand [rows, ld] with `cols` valid elements per row; the pitch tail holds the largest code so
/// that a read past K would show up.
Operand make_e4m3(int64_t rows, int64_t cols, int64_t ld, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
  Rng r(seed);
  Operand op;
  op.raw.assign(static_cast<size_t>(rows * ld), 0x7E);
  op.value.resize(static_cast<size_t>(rows * cols));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      const uint8_t b = e4m3_from_float(r.uni(lo, hi));
      op.raw[static_cast<size_t>(i * ld + j)] = b;
      op.value[static_cast<size_t>(i * cols + j)] = e4m3_to_float(b);
    }
  }
  return op;
}

kg::ScaledGemmArgs make_fp8_args(kg::GemmRecipe recipe, DType out_dt, int64_t m, int64_t n, int64_t k, const void* a,
                                 int64_t lda, const void* b, int64_t ldb, void* d, int64_t ldd, kg::Fp8Accum accum) {
  kg::ScaledGemmArgs x = make_args(DType::kBFloat16, out_dt, m, n, k, a, lda, b, ldb, d, ldd);
  x.recipe = recipe;
  x.a.dtype = DType::kFloat8E4M3;
  x.b.dtype = DType::kFloat8E4M3;
  x.fp8_accum = accum;
  return x;
}

Status run_fp8(const kg::ScaledGemmArgs& x, const kg::GemmTactic& t, const LaunchContext& ctx) {
  return kg::fp8_gemm(x, &t, kg::GemmBackend::kNative, kops::Workspace{}, ctx);
}

/// Tiles of the promoted body (one 128-column B scale block per tile).
const Tile kPromotedTiles[] = {{64, 128, 1, 1}, {128, 128, 1, 1}, {256, 128, 1, 1}};
const Tile kFastTiles[] = {{64, 128, 1, 1}, {64, 256, 1, 1}, {128, 128, 1, 1}, {128, 256, 1, 1}, {256, 128, 1, 1}};

/// Tensorwise (per-tensor scales) and rowwise (per-token x per-channel) on both accumulation modes.
void fp8_plain_case(const LaunchContext& ctx, bool rowwise, DType out_dt, int64_t m, int64_t n, int64_t k) {
  const int64_t ldk = (k + 15) / 16 * 16;
  const Operand a = make_e4m3(m, k, ldk, static_cast<unsigned>(m * 7 + n * 13 + k + 3));
  const Operand b = make_e4m3(n, k, ldk, static_cast<unsigned>(m * 3 + n * 5 + k * 11 + 9));
  const DevBytes da = upload(a.raw);
  const DevBytes db = upload(b.raw);
  Rng r(static_cast<uint64_t>(m * 1000 + n));
  std::vector<float> sa(static_cast<size_t>(rowwise ? m : 1)), sb(static_cast<size_t>(rowwise ? n : 1));
  for (auto& v : sa) v = r.uni(0.25f, 2.0f);
  for (auto& v : sb) v = r.uni(0.25f, 2.0f);
  float* dsa = kops_test::device_alloc<float>(sa.size());
  float* dsb = kops_test::device_alloc<float>(sb.size());
  kops_test::copy_in(dsa, sa);
  kops_test::copy_in(dsb, sb);
  std::vector<double> want(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t kk = 0; kk < k; ++kk) acc += static_cast<double>(a.value[static_cast<size_t>(i * k + kk)]) * b.value[static_cast<size_t>(j * k + kk)];
      acc *= static_cast<double>(sa[rowwise ? static_cast<size_t>(i) : 0]) * sb[rowwise ? static_cast<size_t>(j) : 0];
      want[static_cast<size_t>(i * n + j)] = g_mutate ? acc * 1.02 + 0.05 : acc;
    }
  }
  const size_t out_bytes = static_cast<size_t>(m * n) * dt_bytes(out_dt);
  for (const bool promoted : {false, true}) {
    for (const Tile& tile : kFastTiles) {
      if (promoted && tile.tn > 128) continue;
      DevBytes dout(out_bytes);
      cudaMemset(dout.p, 0xFF, out_bytes);
      kg::ScaledGemmArgs x = make_fp8_args(rowwise ? kg::GemmRecipe::kFP8Rowwise : kg::GemmRecipe::kFP8Tensorwise, out_dt, m,
                                           n, k, da.p, ldk, db.p, ldk, dout.p, n,
                                           promoted ? kg::Fp8Accum::kPromoted : kg::Fp8Accum::kFastAccum);
      x.a.scale = dsa;
      x.b.scale = dsb;
      x.a.granularity = rowwise ? kops::ScaleGranularity::kPerToken : kops::ScaleGranularity::kPerTensor;
      x.b.granularity = rowwise ? kops::ScaleGranularity::kPerChannel : kops::ScaleGranularity::kPerTensor;
      char label[160];
      std::snprintf(label, sizeof(label), "sm90 fp8 %s %s [%lldx%lldx%lld tile %dx%d]", rowwise ? "rowwise" : "tensorwise",
                    promoted ? "promoted" : "fast", static_cast<long long>(m), static_cast<long long>(n),
                    static_cast<long long>(k), tile.tm, tile.tn);
      KOPS_EXPECT_OK(run_fp8(x, sm90_tactic(tile.tm, tile.tn), ctx), label);
      if (!sync_ok(label)) return;
      compare_block(download(dout.p, out_bytes), m, n, n, out_dt, want,
                    2e-3 + (promoted ? kFp8PromotedRel : kFp8FastRel) * rms_of(want), 8e-3, label);
    }
  }
  kops_test::device_free(dsa);
  kops_test::device_free(dsb);
}

/// DeepSeek recipe: A scales 1x128 ([M, K/128] row-major or [K/128, round_up(M, 4)] column-major), B
/// scales 128x128 ([ceil(N/128), K/128]), fp32 or UE8M0 bytes. Oracle: fp64 sum of the scaled block
/// sums; cross-check: the B7 mma.sync blockwise body on the same arguments.
void fp8_block_case(const LaunchContext& ctx, bool col_major, bool ue8m0, DType out_dt, int64_t m, int64_t n, int64_t k) {
  const int64_t kb = k / 128, nbk = (n + 127) / 128;
  const Operand a = make_e4m3(m, k, k, static_cast<unsigned>(m * 5 + n * 17 + k + 1));
  const Operand b = make_e4m3(n, k, k, static_cast<unsigned>(m * 9 + n * 2 + k * 3 + 4));
  const DevBytes da = upload(a.raw);
  const DevBytes db = upload(b.raw);
  Rng r(static_cast<uint64_t>(m * 77 + n * 3 + k));
  const int64_t sa_ld = col_major ? (m + 3) / 4 * 4 : kb;
  const size_t sa_count = static_cast<size_t>(col_major ? kb * sa_ld : m * kb);
  const int es = ue8m0 ? 1 : 4;
  std::vector<uint8_t> sa_raw(sa_count * es, 0), sb_raw(static_cast<size_t>(nbk * kb) * es, 0);
  std::vector<float> sa_val(static_cast<size_t>(m * kb)), sb_val(static_cast<size_t>(nbk * kb));
  auto put = [&](std::vector<uint8_t>& raw, size_t idx) {
    float v;
    if (ue8m0) {
      const int e = static_cast<int>(r.next() % 7) - 3;      // 2^-3 .. 2^3
      raw[idx] = static_cast<uint8_t>(127 + e);
      v = std::ldexp(1.0f, e);
    } else {
      v = r.uni(0.25f, 2.0f);
      std::memcpy(&raw[idx * 4], &v, 4);
    }
    return v;
  };
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t q = 0; q < kb; ++q) {
      sa_val[static_cast<size_t>(i * kb + q)] = put(sa_raw, static_cast<size_t>(col_major ? q * sa_ld + i : i * kb + q));
    }
  }
  for (size_t i = 0; i < sb_val.size(); ++i) sb_val[i] = put(sb_raw, i);
  const DevBytes dsa = upload(sa_raw);
  const DevBytes dsb = upload(sb_raw);

  std::vector<double> want(static_cast<size_t>(m * n));
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t q = 0; q < kb; ++q) {
        double blk = 0.0;
        for (int64_t kk = q * 128; kk < (q + 1) * 128; ++kk) {
          blk += static_cast<double>(a.value[static_cast<size_t>(i * k + kk)]) * b.value[static_cast<size_t>(j * k + kk)];
        }
        acc += blk * sa_val[static_cast<size_t>(i * kb + q)] * sb_val[static_cast<size_t>((j / 128) * kb + q)];
      }
      want[static_cast<size_t>(i * n + j)] = g_mutate ? acc * 1.02 + 0.05 : acc;
    }
  }
  const size_t out_bytes = static_cast<size_t>(m * n) * dt_bytes(out_dt);
  auto make = [&](void* d) {
    kg::ScaledGemmArgs x = make_fp8_args(ue8m0 ? kg::GemmRecipe::kFP8BlockwiseUE8M0 : kg::GemmRecipe::kFP8Blockwise, out_dt, m, n,
                                         k, da.p, k, db.p, k, d, n, kg::Fp8Accum::kPromoted);
    x.a.scale = dsa.p;
    x.b.scale = dsb.p;
    x.a.scale_dtype = ue8m0 ? DType::kFloat8E8M0 : DType::kFloat32;
    x.b.scale_dtype = x.a.scale_dtype;
    x.a.granularity = kops::ScaleGranularity::kPerGroup;
    x.b.granularity = kops::ScaleGranularity::kPerBlock2D;
    x.a.scale_layout = col_major ? kops::ScaleLayout::kColMajorTma : kops::ScaleLayout::kRowMajor;
    return x;
  };
  char label[176];
  std::vector<uint8_t> first;
  for (const Tile& tile : kPromotedTiles) {
    DevBytes dout(out_bytes);
    cudaMemset(dout.p, 0xFF, out_bytes);
    std::snprintf(label, sizeof(label), "sm90 fp8 blockwise %s %s [%lldx%lldx%lld tile %dx%d]", col_major ? "col-major" : "row-major",
                  ue8m0 ? "ue8m0" : "fp32", static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k),
                  tile.tm, tile.tn);
    KOPS_EXPECT_OK(run_fp8(make(dout.p), sm90_tactic(tile.tm, tile.tn), ctx), label);
    if (!sync_ok(label)) return;
    const std::vector<uint8_t> raw = download(dout.p, out_bytes);
    compare_block(raw, m, n, n, out_dt, want, 2e-3 + kFp8PromotedRel * rms_of(want), 8e-3, label);
    if (first.empty()) {
      first = raw;
    } else if (raw != first) {
      kops_test::record_failure(label, "bytes differ from tile 64x128 (tile invariance)", __FILE__, __LINE__);
    }
  }
  // B7 mma.sync blockwise body, same arguments (the tensor-core path of fp8_gemm without an SM90a tactic).
  if (m > 16 && out_dt != DType::kFloat32) {
    DevBytes dref(out_bytes), scratch(64u << 20);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = scratch.n;
    std::snprintf(label, sizeof(label), "sm90 fp8 blockwise B7 cross-check [%lldx%lldx%lld]", static_cast<long long>(m),
                  static_cast<long long>(n), static_cast<long long>(k));
    KOPS_EXPECT_OK(kg::fp8_gemm(make(dref.p), nullptr, kg::GemmBackend::kNative, ws, ctx), label);
    if (!sync_ok(label)) return;
    const std::vector<uint8_t> ref = download(dref.p, out_bytes);
    size_t bad = 0;
    for (int64_t i = 0; i < m * n; ++i) {
      const double g = load_value(&first[static_cast<size_t>(i) * 2], out_dt);
      const double w = load_value(&ref[static_cast<size_t>(i) * 2], out_dt);
      if (!(std::fabs(g - w) <= 2e-3 + kFp8PromotedRel * rms_of(want) + 8e-3 * std::fabs(w))) ++bad;
    }
    if (bad != 0) {
      char detail[128];
      std::snprintf(detail, sizeof(detail), "%zu elements differ from the B7 mma.sync blockwise result", bad);
      kops_test::record_failure(label, detail, __FILE__, __LINE__);
    }
  }
}

void test_fp8(const LaunchContext& ctx) {
  for (const bool rowwise : {false, true}) {
    fp8_plain_case(ctx, rowwise, DType::kBFloat16, 1, 200, 256);
    fp8_plain_case(ctx, rowwise, DType::kBFloat16, 70, 129, 144);     // K tail inside a stage
    fp8_plain_case(ctx, rowwise, DType::kFloat16, 130, 264, 512);
    fp8_plain_case(ctx, rowwise, DType::kFloat32, 300, 600, 1024);
  }
  for (const bool col : {false, true}) {
    for (const bool ue : {false, true}) {
      fp8_block_case(ctx, col, ue, DType::kBFloat16, 1, 128, 128);
      fp8_block_case(ctx, col, ue, DType::kBFloat16, 77, 300, 512);
      fp8_block_case(ctx, col, ue, DType::kFloat16, 200, 129, 256);
      fp8_block_case(ctx, col, ue, DType::kFloat32, 330, 520, 1024);
    }
  }
}

/// B11 gate: with a large K, Fp8Accum::kPromoted is at least as accurate as kFastAccum against the fp64
/// product. Reported as the RMS and max error relative to the RMS of the result (fp32 output, so the
/// output rounding does not hide the accumulation error).
void test_fp8_accumulation_precision(const LaunchContext& ctx) {
  for (const int64_t k : {int64_t{7168}, int64_t{18432}}) {
    const int64_t m = 64, n = 128;
    const Operand a = make_e4m3(m, k, k, 1000u + static_cast<unsigned>(k), -1.0f, 1.0f);
    const Operand b = make_e4m3(n, k, k, 2000u + static_cast<unsigned>(k), -1.0f, 1.0f);
    const DevBytes da = upload(a.raw);
    const DevBytes db = upload(b.raw);
    const std::vector<double> want = reference(a.value, b.value, m, n, k, kg::EpilogueParams{}, EpiHost{});
    double rms = 0.0;
    for (double v : want) rms += v * v;
    rms = std::sqrt(rms / static_cast<double>(want.size()));
    double err_rms[2] = {0.0, 0.0}, err_max[2] = {0.0, 0.0};
    for (int mode = 0; mode < 2; ++mode) {
      const size_t bytes = static_cast<size_t>(m * n) * 4;
      DevBytes dout(bytes);
      kg::ScaledGemmArgs x = make_fp8_args(kg::GemmRecipe::kFP8Tensorwise, DType::kFloat32, m, n, k, da.p, k, db.p, k, dout.p, n,
                                           mode == 0 ? kg::Fp8Accum::kFastAccum : kg::Fp8Accum::kPromoted);
      KOPS_EXPECT_OK(run_fp8(x, sm90_tactic(64, 128), ctx), "sm90 fp8 accumulation precision");
      if (!sync_ok("sm90 fp8 accumulation precision")) return;
      const std::vector<uint8_t> raw = download(dout.p, bytes);
      for (size_t i = 0; i < want.size(); ++i) {
        const double e = std::fabs(static_cast<double>(load_value(&raw[i * 4], DType::kFloat32)) - (g_mutate ? want[i] * 1.5 : want[i]));
        err_rms[mode] += e * e;
        err_max[mode] = std::max(err_max[mode], e);
      }
      err_rms[mode] = std::sqrt(err_rms[mode] / static_cast<double>(want.size())) / rms;
      err_max[mode] /= rms;
    }
    std::printf("[sm90 fp8 accumulation] K=%lld  fast: rms %.3e max %.3e   promoted: rms %.3e max %.3e  (relative to result rms)\n",
                static_cast<long long>(k), err_rms[0], err_max[0], err_rms[1], err_max[1]);
    if (!(err_rms[1] <= err_rms[0])) {
      kops_test::record_failure("sm90 fp8 accumulation precision", "kPromoted is less accurate than kFastAccum", __FILE__, __LINE__);
    }
    if (!(err_max[1] < kFp8PromotedRel) || !(err_max[0] < kFp8FastRel)) {
      kops_test::record_failure("sm90 fp8 accumulation precision", "error above the documented bound", __FILE__, __LINE__);
    }
  }
}

void test_fp8_contracts(const LaunchContext& ctx) {
  const bool on_sm90 = ctx.arch_id == KOPS_TARGET_SM90A;
  const int64_t m = 40, n = 256, k = 256;
  const Operand a = make_e4m3(m, k, k, 5u);
  const Operand b = make_e4m3(n, k, k, 6u);
  const DevBytes da = upload(a.raw);
  const DevBytes db = upload(b.raw);
  std::vector<float> ones(static_cast<size_t>(std::max<int64_t>(m, n) * (k / 128)), 1.0f);
  float* dsc = kops_test::device_alloc<float>(ones.size());
  kops_test::copy_in(dsc, ones);
  std::vector<__nv_bfloat16> sentinel(static_cast<size_t>(m * n), __float2bfloat16_rn(7.0f));
  DevBytes dout(sentinel.size() * 2);
  cudaMemcpy(dout.p, sentinel.data(), sentinel.size() * 2, cudaMemcpyHostToDevice);

  auto block = [&]() {
    kg::ScaledGemmArgs x = make_fp8_args(kg::GemmRecipe::kFP8Blockwise, DType::kBFloat16, m, n, k, da.p, k, db.p, k, dout.p, n,
                                         kg::Fp8Accum::kPromoted);
    x.a.scale = dsc;
    x.b.scale = dsc;
    return x;
  };
  auto plain = [&]() {
    return make_fp8_args(kg::GemmRecipe::kFP8Tensorwise, DType::kBFloat16, m, n, k, da.p, k, db.p, k, dout.p, n,
                         kg::Fp8Accum::kFastAccum);
  };
  const kg::GemmTactic t0 = sm90_tactic(128, 128);
  { auto x = block(); x.fp8_accum = kg::Fp8Accum::kFastAccum;
    KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kInvalidArgument, "sm90 fp8: blockwise with fast accumulation"); }
  { auto x = block(); x.fp8_promotion_k = 64;
    KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kUnsupportedShape, "sm90 fp8: promotion every 64 K"); }
  { auto x = block(); x.k = 200; KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kUnsupportedShape, "sm90 fp8: blockwise K % 128"); }
  { auto x = block(); x.a.scale = nullptr; KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kInvalidArgument, "sm90 fp8: blockwise without A scales"); }
  { auto x = block(); x.b.scale_layout = kops::ScaleLayout::kColMajorTma;
    KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kUnsupportedLayout, "sm90 fp8: column-major B scales"); }
  { auto x = block(); x.recipe = kg::GemmRecipe::kFP8BlockwiseUE8M0;
    KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kUnsupportedDtype, "sm90 fp8: UE8M0 recipe with fp32 scales"); }
  { KOPS_EXPECT_STATUS(run_fp8(block(), sm90_tactic(128, 256), ctx), Status::kUnsupportedShape, "sm90 fp8: promoted on a 256-wide tile"); }
  { KOPS_EXPECT_STATUS(run_fp8(plain(), sm90_tactic(128, 128, 0, 1, 2), ctx), Status::kUnsupportedShape, "sm90 fp8: cluster"); }
  { auto x = plain(); x.a.dtype = DType::kBFloat16;
    KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kUnsupportedDtype, "sm90 fp8: bf16 activations"); }
  { auto x = plain(); x.a.scale = dsc; x.a.granularity = kops::ScaleGranularity::kPerGroup;
    KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kUnsupportedLayout, "sm90 fp8: block granularity on a tensorwise recipe"); }
  { auto x = plain(); x.a.ld = k + 8; KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kMisaligned, "sm90 fp8: lda pitch not 16 B"); }
  { auto x = plain(); x.d_dtype = DType::kFloat8E4M3;
    KOPS_EXPECT_STATUS(run_fp8(x, t0, ctx), Status::kUnsupportedDtype, "sm90 fp8: e4m3 output"); }
  LaunchContext other = ctx;
  other.arch_id = on_sm90 ? KOPS_TARGET_SM89 : ctx.arch_id;
  KOPS_EXPECT_STATUS(run_fp8(plain(), t0, other), Status::kUnsupportedArch, "sm90 fp8: arch gate");
  KOPS_EXPECT_STATUS(run_fp8(block(), t0, other), Status::kUnsupportedArch, "sm90 fp8: arch gate (blockwise)");
  if (!sync_ok("sm90 fp8 contracts")) return;
  const std::vector<uint8_t> after = download(dout.p, sentinel.size() * 2);
  size_t written = 0;
  for (size_t i = 0; i < sentinel.size(); ++i) written += load_value(&after[i * 2], DType::kBFloat16) != 7.0f;
  if (written != 0) kops_test::record_failure("sm90 fp8 contracts", "a refused call wrote the output", __FILE__, __LINE__);
  kops_test::device_free(dsc);
}

// -----------------------------------------------------------------------------
// Grouped (grouped_gemm_wgmma_kernel)
// -----------------------------------------------------------------------------

/// Host image of one grouped problem: operands, the device row metadata and the fp64 reference.
struct Grouped {
  bool masked = false, fp8 = false;
  int64_t g = 0, n = 0, k = 0, cap = 0, max_m = 0;   // cap = rows of A / D
  std::vector<int64_t> begin, rows;                   // per group
  Operand a, b;
  std::vector<float> sa, sb;                          // FP8: [cap, K/128], [G, ceil(N/128), K/128]
};

Grouped make_grouped(bool masked, bool fp8, int64_t n, int64_t k, const std::vector<int64_t>& rows, int64_t max_m,
                     unsigned seed) {
  Grouped q;
  q.masked = masked;
  q.fp8 = fp8;
  q.g = static_cast<int64_t>(rows.size());
  q.n = n;
  q.k = k;
  q.rows = rows;
  q.max_m = max_m;
  int64_t total = 0;
  for (int64_t i = 0; i < q.g; ++i) {
    q.begin.push_back(masked ? i * max_m : total);
    total += rows[static_cast<size_t>(i)];
  }
  q.cap = masked ? q.g * max_m : total + 5;            // contiguous: a few spare rows after the last group
  // Rows that belong to no group hold NaN (bf16) / the largest e4m3 code: they must never reach an output.
  q.a = fp8 ? make_e4m3(q.cap, k, k, seed) : make_16(q.cap, k, k, seed, DType::kBFloat16);
  for (int64_t r = 0; r < q.cap; ++r) {
    bool used = false;
    for (int64_t i = 0; i < q.g; ++i) used = used || (r >= q.begin[static_cast<size_t>(i)] && r < q.begin[static_cast<size_t>(i)] + rows[static_cast<size_t>(i)]);
    if (used) continue;
    for (int64_t c = 0; c < k; ++c) {
      if (fp8) {
        q.a.raw[static_cast<size_t>(r * k + c)] = 0x7E;
      } else {
        const __nv_bfloat16 nan = __float2bfloat16_rn(std::nanf(""));
        std::memcpy(&q.a.raw[static_cast<size_t>(r * k + c) * 2], &nan, 2);
      }
    }
  }
  q.b = fp8 ? make_e4m3(q.g * n, k, k, seed + 1) : make_16(q.g * n, k, k, seed + 1, DType::kBFloat16);
  if (fp8) {
    Rng r(seed + 7);
    q.sa.resize(static_cast<size_t>(q.cap * (k / 128)));
    q.sb.resize(static_cast<size_t>(q.g * ((n + 127) / 128) * (k / 128)));
    for (auto& v : q.sa) v = r.uni(0.25f, 2.0f);
    for (auto& v : q.sb) v = r.uni(0.25f, 2.0f);
  }
  return q;
}

/// fp64 reference of row r (which belongs to group gi).
void grouped_reference_row(const Grouped& q, int64_t gi, int64_t r, std::vector<double>* out) {
  const int64_t kb = q.k / 128, nbk = (q.n + 127) / 128;
  out->assign(static_cast<size_t>(q.n), 0.0);
  for (int64_t j = 0; j < q.n; ++j) {
    double acc = 0.0;
    const float* ar = &q.a.value[static_cast<size_t>(r * q.k)];
    const float* br = &q.b.value[static_cast<size_t>((gi * q.n + j) * q.k)];
    if (q.fp8) {
      for (int64_t b = 0; b < kb; ++b) {
        double blk = 0.0;
        for (int64_t kk = b * 128; kk < (b + 1) * 128; ++kk) blk += static_cast<double>(ar[kk]) * br[kk];
        acc += blk * q.sa[static_cast<size_t>(r * kb + b)] * q.sb[static_cast<size_t>((gi * nbk + j / 128) * kb + b)];
      }
    } else {
      for (int64_t kk = 0; kk < q.k; ++kk) acc += static_cast<double>(ar[kk]) * br[kk];
    }
    (*out)[static_cast<size_t>(j)] = g_mutate ? acc * 1.02 + 0.05 : acc;
  }
}

/// Device side of a Grouped: buffers plus the GroupedGemmParams that describe them.
struct GroupedDev {
  DevBytes a, b, d, sa, sb, meta;
  kg::GroupedGemmParams p{};
  GroupedDev(const Grouped& q)
      : a(upload(q.a.raw)), b(upload(q.b.raw)), d(static_cast<size_t>(q.cap * q.n) * 2),
        sa(q.sa.size() * 4), sb(q.sb.size() * 4), meta(static_cast<size_t>(q.g + 1) * 8) {
    if (q.fp8) {
      cudaMemcpy(sa.p, q.sa.data(), q.sa.size() * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(sb.p, q.sb.data(), q.sb.size() * 4, cudaMemcpyHostToDevice);
    }
    set_rows(q);
    p.layout = q.masked ? kg::GroupedLayout::kMasked : kg::GroupedLayout::kContiguous;
    p.recipe = q.fp8 ? kg::GemmRecipe::kFP8Blockwise : kg::GemmRecipe::kBF16;
    p.num_groups = static_cast<int32_t>(q.g);
    p.a = a.p;
    p.lda = q.k;
    p.b = b.p;
    p.ldb = q.k;
    p.b_group_stride = q.n * q.k;
    p.d = d.p;
    p.ldd = q.n;
    p.n = static_cast<int32_t>(q.n);
    p.k = static_cast<int32_t>(q.k);
    p.max_m = static_cast<int32_t>(q.masked ? q.max_m : q.cap);
    if (q.masked) p.masked_m = reinterpret_cast<const int32_t*>(meta.p);
    else p.expert_offsets = reinterpret_cast<const int64_t*>(meta.p);
    if (q.fp8) {
      p.a_scale = sa.p;
      p.a_scale_layout = kops::ScaleLayout::kRowMajor;
      p.b_scale = sb.p;
    }
  }
  /// Writes the row counts (the only thing a graph replay changes besides the data).
  void set_rows(const Grouped& q) {
    if (q.masked) {
      std::vector<int32_t> m(static_cast<size_t>(q.g));
      for (int64_t i = 0; i < q.g; ++i) m[static_cast<size_t>(i)] = static_cast<int32_t>(q.rows[static_cast<size_t>(i)]);
      cudaMemcpy(meta.p, m.data(), m.size() * 4, cudaMemcpyHostToDevice);
    } else {
      std::vector<int64_t> off(static_cast<size_t>(q.g + 1));
      for (int64_t i = 0; i < q.g; ++i) off[static_cast<size_t>(i)] = q.begin[static_cast<size_t>(i)];
      off[static_cast<size_t>(q.g)] = q.g == 0 ? 0 : q.begin.back() + q.rows.back();
      cudaMemcpy(meta.p, off.data(), off.size() * 8, cudaMemcpyHostToDevice);
    }
  }
  void fill_sentinel() {
    std::vector<__nv_bfloat16> sent(d.n / 2, __float2bfloat16_rn(7.0f));
    cudaMemcpy(d.p, sent.data(), d.n, cudaMemcpyHostToDevice);
  }
};

/// Every group row against the oracle, every other row untouched.
void check_grouped(const Grouped& q, const GroupedDev& dev, const char* label) {
  const std::vector<uint8_t> raw = download(dev.d.p, dev.d.n);
  std::vector<double> want;
  std::vector<int8_t> owner(static_cast<size_t>(q.cap), -1);
  size_t bad = 0, touched = 0;
  double worst = 0.0;
  for (int64_t gi = 0; gi < q.g; ++gi) {
    for (int64_t r = q.begin[static_cast<size_t>(gi)]; r < q.begin[static_cast<size_t>(gi)] + q.rows[static_cast<size_t>(gi)]; ++r) {
      owner[static_cast<size_t>(r)] = 1;
      grouped_reference_row(q, gi, r, &want);
      const double atol = 2e-3 + (q.fp8 ? kFp8PromotedRel * rms_of(want) : 0.0);
      for (int64_t j = 0; j < q.n; ++j) {
        const double got = load_value(&raw[static_cast<size_t>(r * q.n + j) * 2], DType::kBFloat16);
        const double w = round_value(want[static_cast<size_t>(j)], DType::kBFloat16);
        const double err = std::fabs(got - w);
        if (!(err <= atol + 8e-3 * std::fabs(w))) {
          ++bad;
          if (!(err <= worst)) worst = err;
        }
      }
    }
  }
  for (int64_t r = 0; r < q.cap; ++r) {
    if (owner[static_cast<size_t>(r)] == 1) continue;
    for (int64_t j = 0; j < q.n; ++j) touched += load_value(&raw[static_cast<size_t>(r * q.n + j) * 2], DType::kBFloat16) != 7.0f;
  }
  if (bad != 0 || touched != 0) {
    char detail[160];
    std::snprintf(detail, sizeof(detail), "%zu elements off (worst err %.3g), %zu cells outside every group written", bad, worst, touched);
    kops_test::record_failure(label, detail, __FILE__, __LINE__);
  }
}

void grouped_case(const LaunchContext& ctx, bool masked, bool fp8, int64_t n, int64_t k, const std::vector<int64_t>& rows,
                  int64_t max_m, unsigned seed) {
  const Grouped q = make_grouped(masked, fp8, n, k, rows, max_m, seed);
  GroupedDev dev(q);
  for (const Tile& tile : kTiles) {
    if (tile.cm * tile.cn > 1 || tile.tm == 256 || (tile.tm == 64 && tile.tn == 256) || (fp8 && tile.tn > 128)) continue;
    dev.fill_sentinel();
    const kg::GemmTactic t = sm90_tactic(tile.tm, tile.tn);
    char label[160];
    std::snprintf(label, sizeof(label), "sm90 grouped %s %s [G=%lld N=%lld K=%lld tile %dx%d]", masked ? "masked" : "contiguous",
                  fp8 ? "fp8 blockwise" : "bf16", static_cast<long long>(q.g), static_cast<long long>(n),
                  static_cast<long long>(k), tile.tm, tile.tn);
    KOPS_EXPECT_OK(kg::grouped_gemm(dev.p, &t, kg::GemmBackend::kNative, kops::Workspace{}, ctx), label);
    if (!sync_ok(label)) return;
    check_grouped(q, dev, label);
  }
}

void test_grouped(const LaunchContext& ctx) {
  // empty groups, groups below / above one tile, group ends inside a tile
  const std::vector<int64_t> r0 = {5, 0, 64, 130, 1, 0, 257, 33};
  const std::vector<int64_t> r1 = {128, 128, 0, 128};
  for (const bool fp8 : {false, true}) {
    grouped_case(ctx, false, fp8, 264, 256, r0, 0, 71u);
    grouped_case(ctx, false, fp8, 128, 128, r1, 0, 72u);
    grouped_case(ctx, true, fp8, 264, 256, r0, 320, 73u);
    grouped_case(ctx, true, fp8, 130, 384, {64, 64, 7}, 64, 74u);
  }
  // more tiles than SMs
  std::vector<int64_t> many(40);
  for (size_t i = 0; i < many.size(); ++i) many[i] = static_cast<int64_t>((i * 37) % 300);
  grouped_case(ctx, false, false, 1024, 192, many, 0, 75u);
  grouped_case(ctx, true, true, 512, 256, many, 300, 76u);
}

/// B11 gate: capture once, replay while the tokens per expert change between replays.
void test_grouped_graph(const LaunchContext& ctx) {
  const int64_t n = 264, k = 256;
  Grouped q = make_grouped(false, false, n, k, {40, 100, 0, 60, 200}, 0, 81u);
  GroupedDev dev(q);
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return;
  LaunchContext gctx = ctx;
  gctx.stream = stream;
  const kg::GemmTactic t = sm90_tactic(128, 128);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess;
  KOPS_EXPECT_OK(kg::grouped_gemm(dev.p, &t, kg::GemmBackend::kNative, kops::Workspace{}, gctx), "sm90 grouped graph capture");
  ok = ok && cudaStreamEndCapture(stream, &graph) == cudaSuccess && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
  if (!ok) {
    kops_test::record_failure("sm90 grouped graph", "capture / instantiate failed", __FILE__, __LINE__);
  } else {
    const std::vector<std::vector<int64_t>> rounds = {{40, 100, 0, 60, 200}, {0, 0, 400, 0, 0}, {129, 1, 127, 64, 79}, {1, 1, 1, 1, 1}};
    for (const auto& rows : rounds) {
      // same capacity (400 routed rows + spare), different split between the experts
      int64_t total = 0;
      for (size_t i = 0; i < rows.size(); ++i) {
        q.begin[i] = total;
        q.rows[i] = rows[i];
        total += rows[i];
      }
      dev.set_rows(q);
      dev.fill_sentinel();
      if (cudaGraphLaunch(exec, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        kops_test::record_failure("sm90 grouped graph", "replay failed", __FILE__, __LINE__);
        break;
      }
      // NaN rows of the original image now belong to groups in some rounds: give every row real data once
      check_grouped(q, dev, "sm90 grouped graph replay");
    }
  }
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

void test_grouped_contracts(const LaunchContext& ctx) {
  const bool on_sm90 = ctx.arch_id == KOPS_TARGET_SM90A;
  const Grouped q = make_grouped(false, false, 128, 128, {10, 20}, 0, 91u);
  GroupedDev dev(q);
  dev.fill_sentinel();
  const kg::GemmTactic t0 = sm90_tactic(128, 128);
  auto run = [&](const kg::GroupedGemmParams& x, const kg::GemmTactic& t, const LaunchContext& c) {
    return kg::grouped_gemm(x, &t, kg::GemmBackend::kNative, kops::Workspace{}, c);
  };
  { auto x = dev.p; x.max_m = 0; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90 grouped: contiguous without max_m"); }
  { auto x = dev.p; x.expert_offsets = nullptr; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90 grouped: no offsets"); }
  { auto x = dev.p; x.b_group_stride += 8; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kUnsupportedLayout, "sm90 grouped: expert stride"); }
  { auto x = dev.p; x.epi.ops = kg::kEpiRoutingWeight; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kNotImplemented, "sm90 grouped: epilogue op"); }
  { auto x = dev.p; x.recipe = kg::GemmRecipe::kFP8Rowwise; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kNotImplemented, "sm90 grouped: rowwise"); }
  { auto x = dev.p; x.layout = kg::GroupedLayout::kIndexed; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kNotImplemented, "sm90 grouped: indexed"); }
  { auto x = dev.p; x.recipe = kg::GemmRecipe::kFP8Blockwise; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90 grouped: blockwise without scales"); }
  { auto x = dev.p; x.lda = q.k + 4; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kMisaligned, "sm90 grouped: lda pitch"); }
  { KOPS_EXPECT_STATUS(run(dev.p, sm90_tactic(256, 128), ctx), Status::kUnsupportedShape, "sm90 grouped: uncompiled tile"); }
  { KOPS_EXPECT_STATUS(run(dev.p, sm90_tactic(128, 128, 0, 1, 2), ctx), Status::kUnsupportedShape, "sm90 grouped: cluster"); }
  LaunchContext other = ctx;
  other.arch_id = on_sm90 ? KOPS_TARGET_SM120A : ctx.arch_id;
  KOPS_EXPECT_STATUS(run(dev.p, t0, other), Status::kUnsupportedArch, "sm90 grouped: arch gate");
  if (!sync_ok("sm90 grouped contracts")) return;
  const std::vector<uint8_t> after = download(dev.d.p, dev.d.n);
  size_t written = 0;
  for (size_t i = 0; i < after.size() / 2; ++i) written += load_value(&after[i * 2], DType::kBFloat16) != 7.0f;
  if (written != 0) kops_test::record_failure("sm90 grouped contracts", "a refused call wrote the output", __FILE__, __LINE__);
}

// -----------------------------------------------------------------------------
// (b) contracts: every device
// -----------------------------------------------------------------------------

void test_contracts(const LaunchContext& ctx) {
  const bool on_sm90 = ctx.arch_id == KOPS_TARGET_SM90A;
  const int64_t m = 40, n = 200, k = 256;
  const Operand a = make_16(m, k, k, 5u, DType::kBFloat16);
  const Operand b = make_16(n, k, k, 6u, DType::kBFloat16);
  const DevBytes da = upload(a.raw);
  const DevBytes db = upload(b.raw);
  std::vector<__nv_bfloat16> sentinel(static_cast<size_t>(m * n), __float2bfloat16_rn(7.0f));
  DevBytes dout(sentinel.size() * 2);
  cudaMemcpy(dout.p, sentinel.data(), sentinel.size() * 2, cudaMemcpyHostToDevice);

  auto base = [&]() { return make_args(DType::kBFloat16, DType::kBFloat16, m, n, k, da.p, k, db.p, k, dout.p, n); };
  auto run = [&](const kg::ScaledGemmArgs& x, const kg::GemmTactic& t, const LaunchContext& c) {
    return kg::gemm_with_tactic(x, t, kops::Workspace{}, c);
  };
  const kg::GemmTactic t0 = sm90_tactic();

  // Argument errors come first, so they are reported on any exact SM.
  { auto x = base(); x.m = 0; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: M = 0"); }
  { auto x = base(); x.a.data = nullptr; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: null A"); }
  { auto x = base(); x.b.data = nullptr; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: null B"); }
  { auto x = base(); x.d = nullptr; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: null D"); }
  { auto x = base(); x.a.ld = k - 8; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: lda < K"); }
  { auto x = base(); x.ldd = n - 1; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: ldd < N"); }
  { auto x = base(); x.a.ld = k + 4; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kMisaligned, "sm90: lda pitch not 16 B"); }
  { auto x = base(); x.b.data = db.p + 2; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kMisaligned, "sm90: B pointer not 16 B"); }
  { auto x = base(); x.a.major = kg::MajorOrder::kMNMajor;
    KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kNotImplemented, "sm90: MN-major A"); }
  { auto x = base(); x.workspace = dout.p; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: args.workspace"); }
  { auto x = base(); x.batch = 2; x.a.batch_stride = m * k; x.b.batch_stride = n * k; x.batch_stride_d = m * n;
    KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kNotImplemented, "sm90: batch > 1"); }
  { auto x = base(); x.epi.ops = kg::kEpiGLU; x.epi.glu_layout = kops::GateLayout::kInterleaved; x.epi.glu_interleave = 1;
    KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kNotImplemented, "sm90: GLU epilogue"); }
  { auto x = base(); x.epi.ops = kg::kEpiQuantOut; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kNotImplemented, "sm90: quantized output"); }
  { auto x = base(); x.epi.ops = kg::kEpiBias; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: bias without operand"); }
  { auto x = base(); x.a.scale = da.p; KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kInvalidArgument, "sm90: 16-bit operand scales"); }
  { auto x = base(); x.d_dtype = DType::kFloat16;
    KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kUnsupportedDtype, "sm90: fp16 output from bf16 operands"); }
  { auto x = base(); x.recipe = kg::GemmRecipe::kFP16;
    KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kUnsupportedDtype, "sm90: kFP16 with bf16 operands"); }

  // Tactic shape / scheduling contracts.
  { KOPS_EXPECT_STATUS(run(base(), sm90_tactic(64, 64), ctx), Status::kUnsupportedShape, "sm90: uncompiled tile"); }
  { KOPS_EXPECT_STATUS(run(base(), sm90_tactic(128, 128, 5), ctx), Status::kUnsupportedShape, "sm90: wrong stage count"); }
  { auto t = sm90_tactic(128, 128); t.mainloop = kg::MainloopSchedule::kTmaWarpSpecialized;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kUnsupportedShape, "sm90: one-warpgroup mainloop on a 128-row tile"); }
  { auto t = sm90_tactic(64, 128); t.mainloop = kg::MainloopSchedule::kTmaWarpSpecializedCooperative;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kUnsupportedShape, "sm90: cooperative mainloop on a 64-row tile"); }
  { auto t = sm90_tactic(); t.mainloop = kg::MainloopSchedule::kTmaWarpSpecializedPingpong;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kNotImplemented, "sm90: ping-pong"); }
  { auto t = sm90_tactic(); t.split_k_slices = 2; t.split_k_mode = kg::SplitKMode::kSerial;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kNotImplemented, "sm90: split-K"); }
  { auto t = sm90_tactic(); t.swap_ab = true; KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kNotImplemented, "sm90: swap_ab"); }
  { KOPS_EXPECT_STATUS(run(base(), sm90_tactic(128, 128, 0, 2, 2), ctx), Status::kUnsupportedShape, "sm90: cluster 2x2"); }
  { auto t = sm90_tactic(128, 128, 0, 1, 2); t.tma_multicast = false;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kInvalidArgument, "sm90: cluster without tma_multicast"); }
  { auto t = sm90_tactic(); t.tma_multicast = true;
    KOPS_EXPECT_STATUS(run(base(), t, ctx), Status::kInvalidArgument, "sm90: tma_multicast without a cluster"); }
  { KOPS_EXPECT_STATUS(run(base(), sm90_tactic(64, 128, 0, 1, 2), ctx), Status::kUnsupportedShape, "sm90: multicast on a 64-row tile"); }
  { auto x = base(); x.d_dtype = DType::kFloat32;
    KOPS_EXPECT_STATUS(run(x, sm90_tactic(128, 128, 0, 2, 1), ctx), Status::kUnsupportedShape, "sm90: multicast with an fp32 output"); }
  { auto x = base(); x.recipe = kg::GemmRecipe::kINT8Rowwise; x.a.dtype = DType::kInt8; x.b.dtype = DType::kInt8;
    KOPS_EXPECT_STATUS(run(x, t0, ctx), Status::kNotImplemented, "sm90: recipe without an SM90a body"); }

  // Exact-SM gate: a valid call on any target but sm_90a is refused, with the arch named.
  LaunchContext other = ctx;
  other.arch_id = on_sm90 ? KOPS_TARGET_SM86 : ctx.arch_id;
  KOPS_EXPECT_STATUS(run(base(), t0, other), Status::kUnsupportedArch, "sm90: arch gate");
  LaunchContext sm100 = ctx;
  sm100.arch_id = KOPS_TARGET_SM100A;
  KOPS_EXPECT_STATUS(run(base(), t0, sm100), Status::kUnsupportedArch, "sm90: a larger SM is not sm_90a");

  // No refused call wrote the output.
  if (!sync_ok("sm90 contracts")) return;
  const std::vector<uint8_t> after = download(dout.p, sentinel.size() * 2);
  size_t written = 0;
  for (size_t i = 0; i < sentinel.size(); ++i) written += load_value(&after[i * 2], DType::kBFloat16) != 7.0f;
  if (written != 0) kops_test::record_failure("sm90 contracts", "a refused call wrote the output", __FILE__, __LINE__);

  // Workspace: the cell needs none.
  kg::GemmProblem pr{};
  pr.m = m; pr.n = n; pr.k = k;
  kops::WorkspaceRequirements req{};
  req.device_scratch_bytes = 123;
  KOPS_EXPECT_OK(kg::gemm_workspace_requirements(pr, &t0, kg::GemmBackend::kNative, ctx, &req), "sm90: workspace query");
  KOPS_TEST_CHECK(req.device_scratch_bytes == 0 && req.device_persistent_bytes == 0);

  if (on_sm90) KOPS_EXPECT_STATUS(run(base(), t0, ctx), Status::kSuccess, "sm90: baseline");
  sync_ok("sm90 contracts baseline");
}

}  // namespace

int main(int argc, char** argv) {
  const kops_test::Options opt = kops_test::parse_options(argc, argv);
  g_mutate = opt.mutate;
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
  test_contracts(ctx);
  test_fp8_contracts(ctx);
  test_grouped_contracts(ctx);
  if (ctx.arch_id != KOPS_TARGET_SM90A) {
    if (kops_test::failure_count() != 0) return kops_test::finish("test_sm90 (contracts)");
    std::printf("SKIP test_sm90: contracts pass; the SM90a numerics need KOPS_TARGET_SM90A (found arch %d)\n",
                ctx.arch_id);
    return 77;
  }
  test_dense_sweep(ctx);
  test_dense_zero_and_guard(ctx);
  test_dense_unaligned_out(ctx);
  test_dense_epilogue(ctx);
  test_fp8(ctx);
  test_fp8_accumulation_precision(ctx);
  test_grouped(ctx);
  test_grouped_graph(ctx);
  if (kops_test::tier_enabled(opt, "standard")) {
    test_dense_persistent(ctx);
    test_graph_and_determinism(ctx);
  }
  return kops_test::finish("test_sm90");
}
