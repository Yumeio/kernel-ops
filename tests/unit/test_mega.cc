// =============================================================================
// kernel-ops | tests/unit/test_mega.cc
// B16 persistent execution / megakernel tests.
//
// Covered:
//   mega_mlp              vs modular norm + mma_gemm + act_and_mul + mma_gemm,
//                         and vs a local fp64 reference; graph replay (1500x);
//                         grids of 1 and 2 CTAs; resource report; error contract.
//   mega_attention_block  vs a local fp64 reference (norm/QKV/QK-norm/RoPE/KV
//                         append/paged decode/O-proj); appended K/V page check.
//   mega_moe_local        vs a local fp64 reference (precomputed routes).
//   mega_moe_ep           vs the same reference through the single-rank EP buffer.
//   mega_decoder_layer    vs the standalone mega attention block + mega MLP chain.
//
// Every wait is guarded by a host watchdog (fail fast instead of hanging CTest);
// CTest itself carries TIMEOUT 300.  The test returns 77 when no CUDA device or
// an SM < 80 is present.
// =============================================================================

#include "tests/unit/harness.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/communication/allreduce_kernels.cuh"
#include "csrc/communication/ep_kernels.cuh"
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
#include "csrc/execution/persistent.cuh"
#include "csrc/mega/mega_plan.cuh"
#include "csrc/mega/mega_kernels.cuh"
#include "csrc/mega/ops.h"
#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"
#include "csrc/activation/activation_kernels.cuh"
#include "csrc/activation/ops.h"
#include "csrc/gemm/ops.h"

namespace {

using kops::LaunchContext;
using kops::Status;
using kops::Workspace;
using kops::WorkspaceRequirements;
using kops::mega::MegaKind;
using kops::mega::MegaPlan;
using bf16 = __nv_bfloat16;

constexpr float kEps = 1e-5f;
constexpr int kWatchdogSeconds = 20;

// ---- watchdog ---------------------------------------------------------------
bool wait_stream(kops::kops_stream_t stream, const char* what) {
  cudaEvent_t ev = nullptr;
  if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return false;
  if (cudaEventRecord(ev, reinterpret_cast<cudaStream_t>(stream)) != cudaSuccess) {
    cudaEventDestroy(ev);
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    const cudaError_t q = cudaEventQuery(ev);
    if (q == cudaSuccess) break;
    if (q != cudaErrorNotReady) {
      std::fprintf(stderr, "FAIL %s: cudaEventQuery: %s\n", what, cudaGetErrorString(q));
      cudaEventDestroy(ev);
      return false;
    }
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (sec > kWatchdogSeconds) {
      std::fprintf(stderr, "FAIL %s: watchdog: stream did not finish in %d s (deadlock?)\n",
                   what, kWatchdogSeconds);
      std::fflush(stderr);
      std::_Exit(2);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  cudaEventDestroy(ev);
  return true;
}

// ---- host data ---------------------------------------------------------------
void fill(std::vector<float>& v, unsigned seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& x : v) x = dist(rng);
}

std::vector<bf16> to_bf16(const std::vector<float>& src) {
  std::vector<bf16> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = __float2bfloat16_rn(src[i]);
  return out;
}

std::vector<double> to_double(const std::vector<bf16>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<double>(__bfloat162float(src[i]));
  return out;
}

template <typename T>
T* upload(const std::vector<T>& v) {
  T* p = kops_test::device_alloc<T>(v.size());
  if (p != nullptr) kops_test::copy_in(p, v);
  return p;
}

template <typename T>
std::vector<T> download(const T* p, size_t n) {
  return kops_test::copy_out(p, n);
}

// ---- workspaces / plans -------------------------------------------------------
struct PlanBuf {
  MegaPlan plan{};
  void* mem = nullptr;
  size_t bytes = 0;
};

bool make_plan(MegaKind kind, const void* params, const LaunchContext& ctx, PlanBuf* pb,
               const char* what) {
  WorkspaceRequirements req{};
  Status st = kops::mega::mega_workspace_requirements(kind, params, ctx, &req);
  if (st != Status::kSuccess) {
    kops_test::record_failure(what, kops::last_error_message(), __FILE__, __LINE__);
    return false;
  }
  pb->bytes = req.device_persistent_bytes;
  if (cudaMalloc(&pb->mem, pb->bytes) != cudaSuccess) {
    kops_test::record_failure(what, "cudaMalloc(workspace)", __FILE__, __LINE__);
    return false;
  }
  Workspace ws{};
  ws.device_persistent = pb->mem;
  ws.device_persistent_bytes = pb->bytes;
  st = kops::mega::mega_prepare(kind, params, ws, ctx, &pb->plan);
  if (st != Status::kSuccess) {
    kops_test::record_failure(what, kops::last_error_message(), __FILE__, __LINE__);
    return false;
  }
  return true;
}

void free_plan(PlanBuf* pb) {
  if (pb->mem != nullptr) cudaFree(pb->mem);
  pb->mem = nullptr;
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

// ---- modular GEMM helper ------------------------------------------------------
bool gemm_bf16(const void* a, int64_t lda, const void* b, int64_t ldb, void* d, int64_t ldd,
               int64_t m, int64_t n, int64_t k, const kops::gemm::EpilogueParams& epi,
               const Workspace& ws, const LaunchContext& ctx, const char* what) {
  kops::gemm::ScaledGemmArgs args{};
  args.a.data = a;
  args.a.ld = lda;
  args.a.dtype = kops::DType::kBFloat16;
  args.b.data = b;
  args.b.ld = ldb;
  args.b.dtype = kops::DType::kBFloat16;
  args.d = d;
  args.ldd = ldd;
  args.d_dtype = kops::DType::kBFloat16;
  args.epi = epi;
  args.m = m;
  args.n = n;
  args.k = k;
  args.recipe = kops::gemm::GemmRecipe::kBF16;
  const Status st = kops::gemm::mma_gemm(args, nullptr, ws, ctx);
  if (st != Status::kSuccess) {
    kops_test::record_failure(what, kops::last_error_message(), __FILE__, __LINE__);
    return false;
  }
  return true;
}

// ---- fp64 MLP reference -------------------------------------------------------
std::vector<double> mlp_ref(const std::vector<bf16>& x, const std::vector<bf16>& w13,
                            const std::vector<bf16>& w2, const std::vector<bf16>& nw,
                            const std::vector<bf16>& residual, int M, int H, int I,
                            kops::ActFn act, kops::GateLayout layout) {
  const std::vector<double> xd = to_double(x);
  const std::vector<double> wd = to_double(w13);
  const std::vector<double> w2d = to_double(w2);
  const std::vector<double> nwd = to_double(nw);
  const std::vector<double> rd = to_double(residual);
  std::vector<double> out(static_cast<size_t>(M) * H, 0.0);
  std::vector<double> normed(static_cast<size_t>(H));
  std::vector<double> inter(static_cast<size_t>(I));
  for (int m = 0; m < M; ++m) {
    double ss = 0.0;
    for (int k = 0; k < H; ++k) ss += xd[static_cast<size_t>(m) * H + k] * xd[static_cast<size_t>(m) * H + k];
    const double inv = 1.0 / std::sqrt(ss / H + kEps);
    for (int k = 0; k < H; ++k) normed[k] = xd[static_cast<size_t>(m) * H + k] * inv * nwd[k];
    for (int i = 0; i < I; ++i) {
      const int gi = layout == kops::GateLayout::kUpGate ? i + I
                     : layout == kops::GateLayout::kInterleaved ? 2 * i : i;
      const int ui = layout == kops::GateLayout::kUpGate ? i
                     : layout == kops::GateLayout::kInterleaved ? 2 * i + 1 : i + I;
      double g = 0.0, u = 0.0;
      for (int k = 0; k < H; ++k) {
        g += normed[k] * wd[static_cast<size_t>(gi) * H + k];
        u += normed[k] * wd[static_cast<size_t>(ui) * H + k];
      }
      double v = 0.0;
      switch (act) {
        case kops::ActFn::kSiLU: v = g / (1.0 + std::exp(-g)) * u; break;
        case kops::ActFn::kGeLUTanh:
          v = 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u;
          break;
        case kops::ActFn::kReLU: v = (g > 0.0 ? g : 0.0) * u; break;
        default: v = g * u; break;
      }
      inter[i] = v;
    }
    for (int h = 0; h < H; ++h) {
      double v = 0.0;
      for (int i = 0; i < I; ++i) v += inter[i] * w2d[static_cast<size_t>(h) * I + i];
      out[static_cast<size_t>(m) * H + h] = v + rd[static_cast<size_t>(m) * H + h];
    }
  }
  return out;
}

// ---- MLP test ----------------------------------------------------------------
void test_mlp(const LaunchContext& base_ctx, const Workspace& gws) {
  constexpr int M = 4, H = 128, I = 256;
  std::vector<float> xf(static_cast<size_t>(M) * H), w13f(static_cast<size_t>(2 * I) * H),
      w2f(static_cast<size_t>(H) * I), nwf(H), resf(static_cast<size_t>(M) * H);
  fill(xf, 11, -1.0f, 1.0f);
  fill(w13f, 12, -0.08f, 0.08f);
  fill(w2f, 13, -0.08f, 0.08f);
  fill(nwf, 14, 0.8f, 1.2f);
  fill(resf, 15, -0.5f, 0.5f);
  const std::vector<bf16> x = to_bf16(xf), w13 = to_bf16(w13f), w2 = to_bf16(w2f),
                          nw = to_bf16(nwf), residual = to_bf16(resf);
  const std::vector<double> ref = mlp_ref(x, w13, w2, nw, residual, M, H, I,
                                          kops::ActFn::kSiLU, kops::GateLayout::kGateUp);

  bf16* dx = upload(x);
  bf16* dw13 = upload(w13);
  bf16* dw2 = upload(w2);
  bf16* dnw = upload(nw);
  bf16* dres = upload(residual);
  bf16* dnormed = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);
  bf16* dgateup = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * 2 * I);
  bf16* dinter = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * I);
  bf16* dinter_mod = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * I);
  bf16* dmod = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);
  bf16* dmega = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);

  // ---- modular chain ----
  const kops::Tensor xt = kops_test::tensor2d(dx, kops::DType::kBFloat16, M, H);
  const kops::Tensor nwt = tensor1d(dnw, kops::DType::kBFloat16, H);
  const kops::Tensor normedt = kops_test::tensor2d(dnormed, kops::DType::kBFloat16, M, H);
  KOPS_EXPECT_OK(kops::norm::rms_norm(normedt, xt, nwt, kEps, 0.0f, base_ctx), "modular rms_norm");
  if (!gemm_bf16(dnormed, H, dw13, H, dgateup, 2 * I, M, 2 * I, H, {}, gws, base_ctx,
                 "modular gate_up gemm")) {
    return;
  }
  const kops::Tensor gut = kops_test::tensor2d(dgateup, kops::DType::kBFloat16, M, 2 * I);
  const kops::Tensor it = kops_test::tensor2d(dinter, kops::DType::kBFloat16, M, I);
  kops::GluParams glu{};
  KOPS_EXPECT_OK(kops::act::act_and_mul(it, gut, kops::Tensor{}, kops::ActFn::kSiLU,
                                        kops::GateLayout::kGateUp, glu, base_ctx),
                 "modular act_and_mul");
  KOPS_TEST_CHECK(cudaMemcpy(dinter_mod, dinter, static_cast<size_t>(M) * I * sizeof(bf16),
                             cudaMemcpyDeviceToDevice) == cudaSuccess);
  kops::gemm::EpilogueParams epi{};
  epi.ops = kops::gemm::kEpiResidualAdd;
  epi.beta = 1.0f;
  epi.residual = dres;
  epi.residual_stride = H;
  if (!gemm_bf16(dinter, I, dw2, I, dmod, H, M, H, I, epi, gws, base_ctx, "modular down gemm")) {
    return;
  }
  if (!wait_stream(base_ctx.stream, "mlp modular")) return;
  const std::vector<double> mod_d = to_double(download(dmod, static_cast<size_t>(M) * H));

  // ---- mega MLP ----
  kops::mega::MegaMLPParams p{};
  p.x = dx;
  p.ldx = H;
  p.pre_norm_weight = dnw;
  p.pre_norm_eps = kEps;
  p.w13 = dw13;
  p.w2_scale = nullptr;
  p.w13_scale = nullptr;
  p.w2 = dw2;
  p.w_quant = kops::QuantScheme{};
  p.w_quant.storage_dtype = kops::DType::kBFloat16;
  p.act = kops::ActFn::kSiLU;
  p.gate_layout = kops::GateLayout::kGateUp;
  p.glu = glu;
  p.intermediate = dinter;
  p.out = dmega;
  p.ldo = H;
  p.residual = dres;
  p.m = M;
  p.hidden_size = H;
  p.intermediate_size = I;

  PlanBuf pb;
  if (!make_plan(MegaKind::kMLP, &p, base_ctx, &pb, "mega_prepare(mlp)")) return;
  std::printf("[mega/mlp] ctas=%d tasks=%d regs=%d smem=%d launches=%d\n", pb.plan.info.ctas,
              pb.plan.info.num_tasks, pb.plan.info.regs_per_thread, pb.plan.info.smem_per_cta,
              pb.plan.info.num_device_launches);
  KOPS_TEST_CHECK(pb.plan.info.num_device_launches == 1);
  KOPS_TEST_CHECK(pb.plan.info.ctas >= 1);
  KOPS_TEST_CHECK(pb.plan.info.regs_per_thread > 0);
  KOPS_TEST_CHECK(pb.plan.info.smem_per_cta >= H * 2);
  KOPS_EXPECT_OK(kops::mega::mega_run_mlp(pb.plan, p, base_ctx), "mega_run_mlp");
  if (!wait_stream(base_ctx.stream, "mega mlp")) return;
  const std::vector<double> mega_d = to_double(download(dmega, static_cast<size_t>(M) * H));
  kops_test::compare(download(dmega, static_cast<size_t>(M) * H), ref, 5e-2, 5e-2, "mega_mlp ref");
  kops_test::compare(download(dmega, static_cast<size_t>(M) * H), mod_d, 3e-2, 3e-2,
                     "mega_mlp vs modular");

  // Same plan, tiny grids: 1 CTA and 2 CTAs (no deadlock, same result).
  for (int units = 1; units <= 2; ++units) {
    LaunchContext small = base_ctx;
    small.compute_units = units;
    small.max_ctas_per_cu = 1;
    PlanBuf ps;
    if (!make_plan(MegaKind::kMLP, &p, small, &ps, "mega_prepare(mlp small grid)")) return;
    std::printf("[mega/mlp] units=%d -> ctas=%d\n", units, ps.plan.info.ctas);
    KOPS_TEST_CHECK(ps.plan.info.ctas <= units);
    KOPS_EXPECT_OK(kops::mega::mega_run_mlp(ps.plan, p, small), "mega_run_mlp small grid");
    if (!wait_stream(small.stream, "mega mlp small grid")) return;
    {
      const std::vector<bf16> im = download(dinter, static_cast<size_t>(M) * I);
      const std::vector<bf16> imr = download(dinter_mod, static_cast<size_t>(M) * I);
      size_t bad = 0, first = 0;
      double worst = 0.0;
      for (size_t i = 0; i < im.size(); ++i) {
        const double a = kops_test::to_float(im[i]);
        const double b = kops_test::to_float(imr[i]);
        if (std::fabs(a - b) > 1e-2 + 1e-2 * std::fabs(b)) {
          ++bad;
          if (std::fabs(a - b) > worst) { worst = std::fabs(a - b); first = i; }
        }
      }
      KOPS_TEST_CHECK(bad == 0);
      if (bad != 0) {
        std::printf("[mega/mlp] intermediate mismatch at %zu: got %.5f want %.5f\n", first,
                    kops_test::to_float(im[first]), kops_test::to_float(imr[first]));
      }
    }
    kops_test::compare(download(dmega, static_cast<size_t>(M) * H), mod_d, 3e-2, 3e-2,
                       "mega_mlp small grid vs modular");
    free_plan(&ps);
  }

  // ---- graph replay (1500x) ----
  cudaStream_t gs = nullptr;
  KOPS_TEST_CHECK(cudaStreamCreate(&gs) == cudaSuccess);
  LaunchContext gctx = base_ctx;
  gctx.stream = reinterpret_cast<kops::kops_stream_t>(gs);
  PlanBuf pg;
  if (!make_plan(MegaKind::kMLP, &p, gctx, &pg, "mega_prepare(mlp graph)")) return;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  KOPS_TEST_CHECK(cudaStreamBeginCapture(gs, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  KOPS_EXPECT_OK(kops::mega::mega_run_mlp(pg.plan, p, gctx), "mega_run_mlp capture");
  KOPS_TEST_CHECK(cudaStreamEndCapture(gs, &graph) == cudaSuccess);
  KOPS_TEST_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
  for (int i = 0; i < 1500; ++i) {
    if (cudaGraphLaunch(exec, gs) != cudaSuccess) {
      kops_test::record_failure("cudaGraphLaunch", cudaGetErrorString(cudaGetLastError()),
                                __FILE__, __LINE__);
      break;
    }
  }
  if (!wait_stream(gctx.stream, "mega mlp graph replay")) return;
  kops_test::compare(download(dmega, static_cast<size_t>(M) * H), mod_d, 3e-2, 3e-2,
                     "mega_mlp graph replay vs modular");
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(gs);
  free_plan(&pg);

  // ---- error contract ----
  {
    WorkspaceRequirements req{};
    KOPS_EXPECT_STATUS(kops::mega::mega_workspace_requirements(MegaKind::kMLP, nullptr, base_ctx, &req),
                       Status::kInvalidArgument, "workspace null params");
    KOPS_EXPECT_STATUS(kops::mega::mega_workspace_requirements(MegaKind::kMLP, &p, base_ctx, nullptr),
                       Status::kInvalidArgument, "workspace null out");
    Workspace tiny{};
    tiny.device_persistent = pb.mem;
    tiny.device_persistent_bytes = 16;
    MegaPlan bad_plan{};
    KOPS_EXPECT_STATUS(kops::mega::mega_prepare(MegaKind::kMLP, &p, tiny, base_ctx, &bad_plan),
                       Status::kWorkspaceTooSmall, "prepare tiny workspace");
    kops::mega::MegaMLPParams too_big = p;
    too_big.m = M + 1;
    KOPS_EXPECT_STATUS(kops::mega::mega_run_mlp(pb.plan, too_big, base_ctx),
                       Status::kCapacityExceeded, "run beyond capacity");
    kops::mega::MegaMLPParams bad_q = p;
    bad_q.w_quant.storage_dtype = kops::DType::kFloat8E4M3;
    KOPS_EXPECT_STATUS(kops::mega::mega_run_mlp(pb.plan, bad_q, base_ctx),
                       Status::kUnsupportedDtype, "run FP8 weights");
    kops::mega::MegaAttentionBlockParams attn_zero{};
    KOPS_EXPECT_STATUS(kops::mega::mega_run_attention_block(pb.plan, attn_zero, base_ctx),
                       Status::kInvalidArgument, "run wrong kind");
    cudaStream_t other = nullptr;
    KOPS_TEST_CHECK(cudaStreamCreate(&other) == cudaSuccess);
    LaunchContext ctx2 = base_ctx;
    ctx2.stream = reinterpret_cast<kops::kops_stream_t>(other);
    KOPS_EXPECT_STATUS(kops::mega::mega_run_mlp(pb.plan, p, ctx2), Status::kInvalidArgument,
                       "one plan one stream");
    cudaStreamDestroy(other);
  }

  free_plan(&pb);
  kops_test::device_free(dx);
  kops_test::device_free(dw13);
  kops_test::device_free(dw2);
  kops_test::device_free(dnw);
  kops_test::device_free(dres);
  kops_test::device_free(dnormed);
  kops_test::device_free(dgateup);
  kops_test::device_free(dinter);
  kops_test::device_free(dinter_mod);
  kops_test::device_free(dmod);
  kops_test::device_free(dmega);
}

// ---- fp64 attention reference --------------------------------------------------
struct KVBuf {
  int page_size = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  std::vector<bf16> k;   // [page_size * Hkv * D]
  std::vector<bf16> v;
  int stride_page() const { return page_size * num_kv_heads * head_dim; }
  int stride_token() const { return num_kv_heads * head_dim; }
  size_t idx(int pos, int hk, int d) const {
    return static_cast<size_t>(pos) * stride_token() + hk * head_dim + d;
  }
};

struct AttnRef {
  std::vector<double> out;
  std::vector<double> k_app;   // [M * Hkv * D]
  std::vector<double> v_app;
};

AttnRef attention_block_ref(const std::vector<bf16>& hidden, const std::vector<bf16>& resid,
                            const std::vector<bf16>& nw, float eps,
                            const std::vector<bf16>& wqkv, const std::vector<bf16>& bias,
                            const std::vector<bf16>& qnw, const std::vector<bf16>& knw,
                            const std::vector<bf16>& w_o, const std::vector<float>& cache,
                            const KVBuf& old_kv, const std::vector<int64_t>& positions,
                            int M, int H, int Hq, int Hkv, int D, float sm_scale) {
  const int total = (Hq + 2 * Hkv) * D;
  const std::vector<double> xd = to_double(hidden);
  const std::vector<double> rd = to_double(resid);
  const std::vector<double> nwd = to_double(nw);
  const std::vector<double> wd = to_double(wqkv);
  const std::vector<double> bd = to_double(bias);
  const std::vector<double> qwd = to_double(qnw);
  const std::vector<double> kwd = to_double(knw);
  const std::vector<double> wod = to_double(w_o);
  AttnRef r;
  r.out.assign(static_cast<size_t>(M) * H, 0.0);
  r.k_app.assign(static_cast<size_t>(M) * Hkv * D, 0.0);
  r.v_app.assign(static_cast<size_t>(M) * Hkv * D, 0.0);
  std::vector<double> qkv(static_cast<size_t>(M) * total, 0.0);
  for (int m = 0; m < M; ++m) {
    std::vector<double> normed(H);
    double ss = 0.0;
    for (int k = 0; k < H; ++k) ss += xd[static_cast<size_t>(m) * H + k] * xd[static_cast<size_t>(m) * H + k];
    const double inv = 1.0 / std::sqrt(ss / H + eps);
    for (int k = 0; k < H; ++k) normed[k] = xd[static_cast<size_t>(m) * H + k] * inv * nwd[k];
    for (int j = 0; j < total; ++j) {
      double v = bd[j];
      for (int k = 0; k < H; ++k) v += normed[k] * wd[static_cast<size_t>(j) * H + k];
      qkv[static_cast<size_t>(m) * total + j] = v;
    }
    for (int h = 0; h < Hq + Hkv; ++h) {
      const std::vector<double>& wgt = h < Hq ? qwd : kwd;
      double* head = &qkv[static_cast<size_t>(m) * total + static_cast<size_t>(h) * D];
      double ss2 = 0.0;
      for (int d = 0; d < D; ++d) ss2 += head[d] * head[d];
      const double inv2 = 1.0 / std::sqrt(ss2 / D + eps);
      for (int d = 0; d < D; ++d) head[d] = head[d] * inv2 * wgt[d];
      const float* cs = &cache[static_cast<size_t>(positions[m]) * D];
      const int half = D / 2;
      for (int e = 0; e < half; ++e) {
        const double a = head[e], b = head[e + half];
        head[e] = a * cs[e] - b * cs[half + e];
        head[e + half] = b * cs[e] + a * cs[half + e];
      }
    }
    for (int hk = 0; hk < Hkv; ++hk) {
      for (int d = 0; d < D; ++d) {
        r.k_app[(static_cast<size_t>(m) * Hkv + hk) * D + d] =
            qkv[static_cast<size_t>(m) * total + static_cast<size_t>(Hq + hk) * D + d];
        r.v_app[(static_cast<size_t>(m) * Hkv + hk) * D + d] =
            qkv[static_cast<size_t>(m) * total + static_cast<size_t>(Hq + Hkv + hk) * D + d];
      }
    }
  }
  std::vector<std::vector<double>> K(old_kv.stride_page(), std::vector<double>(Hkv * D, 0.0));
  std::vector<std::vector<double>> V(K);
  for (int pos = 0; pos < old_kv.page_size; ++pos) {
    for (int hk = 0; hk < Hkv; ++hk) {
      for (int d = 0; d < D; ++d) {
        K[pos][hk * D + d] = static_cast<double>(__bfloat162float(old_kv.k[old_kv.idx(pos, hk, d)]));
        V[pos][hk * D + d] = static_cast<double>(__bfloat162float(old_kv.v[old_kv.idx(pos, hk, d)]));
      }
    }
  }
  for (int m = 0; m < M; ++m) {
    const int pos = static_cast<int>(positions[m]);
    for (int hk = 0; hk < Hkv; ++hk) {
      for (int d = 0; d < D; ++d) {
        K[pos][hk * D + d] = r.k_app[(static_cast<size_t>(m) * Hkv + hk) * D + d];
        V[pos][hk * D + d] = r.v_app[(static_cast<size_t>(m) * Hkv + hk) * D + d];
      }
    }
  }
  const int G = Hq / Hkv;
  std::vector<double> attn(static_cast<size_t>(M) * Hq * D, 0.0);
  for (int m = 0; m < M; ++m) {
    const int pos = static_cast<int>(positions[m]);
    for (int hq = 0; hq < Hq; ++hq) {
      const int hk = hq / G;
      std::vector<double> score(pos + 1);
      double mx = -1e300;
      for (int j = 0; j <= pos; ++j) {
        double s = 0.0;
        for (int d = 0; d < D; ++d) {
          s += qkv[static_cast<size_t>(m) * total + static_cast<size_t>(hq) * D + d] * K[j][hk * D + d];
        }
        score[j] = s * sm_scale;
        mx = std::max(mx, score[j]);
      }
      double den = 0.0;
      std::vector<double> acc(D, 0.0);
      for (int j = 0; j <= pos; ++j) {
        const double p = std::exp(score[j] - mx);
        den += p;
        for (int d = 0; d < D; ++d) acc[d] += p * V[j][hk * D + d];
      }
      for (int d = 0; d < D; ++d) {
        attn[(static_cast<size_t>(m) * Hq + hq) * D + d] = den > 0.0 ? acc[d] / den : 0.0;
      }
    }
  }
  for (int m = 0; m < M; ++m) {
    for (int h = 0; h < H; ++h) {
      double v = 0.0;
      for (int j = 0; j < Hq * D; ++j) {
        v += attn[static_cast<size_t>(m) * Hq * D + j] * wod[static_cast<size_t>(h) * Hq * D + j];
      }
      r.out[static_cast<size_t>(m) * H + h] = v + rd[static_cast<size_t>(m) * H + h];
    }
  }
  return r;
}

// ---- attention test ------------------------------------------------------------
void test_attention(const LaunchContext& ctx) {
  constexpr int M = 2, H = 64, Hq = 4, Hkv = 2, D = 32, PS = 16, POS0 = 4;
  const int total = (Hq + 2 * Hkv) * D;
  std::vector<float> hf(static_cast<size_t>(M) * H), rf(static_cast<size_t>(M) * H),
      nwf(H), wqf(static_cast<size_t>(total) * H), bf(static_cast<size_t>(total)),
      qnwf(D), knwf(D), wof(static_cast<size_t>(H) * Hq * D);
  fill(hf, 21, -1.0f, 1.0f);
  fill(rf, 22, -0.5f, 0.5f);
  fill(nwf, 23, 0.8f, 1.2f);
  fill(wqf, 24, -0.08f, 0.08f);
  fill(bf, 25, -0.1f, 0.1f);
  fill(qnwf, 26, 0.8f, 1.2f);
  fill(knwf, 27, 0.8f, 1.2f);
  fill(wof, 28, -0.08f, 0.08f);
  const std::vector<bf16> hidden = to_bf16(hf), residual = to_bf16(rf), nw = to_bf16(nwf),
                          wqkv = to_bf16(wqf), bias = to_bf16(bf), qnw = to_bf16(qnwf),
                          knw = to_bf16(knwf), w_o = to_bf16(wof);
  std::vector<int64_t> positions(M);
  std::vector<int64_t> slots(M);
  for (int m = 0; m < M; ++m) {
    positions[m] = POS0 + m;
    slots[m] = POS0 + m;
  }
  std::vector<float> cache(static_cast<size_t>(PS) * D, 0.0f);
  const float theta = 10000.0f;
  for (int p = 0; p < PS; ++p) {
    for (int e = 0; e < D / 2; ++e) {
      const double inv = std::pow(theta, -2.0 * e / D);
      cache[static_cast<size_t>(p) * D + e] = static_cast<float>(std::cos(p * inv));
      cache[static_cast<size_t>(p) * D + D / 2 + e] = static_cast<float>(std::sin(p * inv));
    }
  }
  KVBuf old_kv;
  old_kv.page_size = PS;
  old_kv.num_kv_heads = Hkv;
  old_kv.head_dim = D;
  old_kv.k.assign(static_cast<size_t>(PS) * Hkv * D, __float2bfloat16(0.0f));
  old_kv.v.assign(static_cast<size_t>(PS) * Hkv * D, __float2bfloat16(0.0f));
  std::vector<float> okf(static_cast<size_t>(PS) * Hkv * D), ovf(static_cast<size_t>(PS) * Hkv * D);
  fill(okf, 29, -0.5f, 0.5f);
  fill(ovf, 30, -0.5f, 0.5f);
  for (int pos = 0; pos < POS0; ++pos) {
    for (int hk = 0; hk < Hkv; ++hk) {
      for (int d = 0; d < D; ++d) {
        const size_t at = old_kv.idx(pos, hk, d);
        old_kv.k[at] = __float2bfloat16_rn(okf[at]);
        old_kv.v[at] = __float2bfloat16_rn(ovf[at]);
      }
    }
  }
  const float sm = 1.0f / std::sqrt(static_cast<float>(D));
  const AttnRef ref = attention_block_ref(hidden, residual, nw, kEps, wqkv, bias, qnw, knw, w_o,
                                          cache, old_kv, positions, M, H, Hq, Hkv, D, sm);

  bf16* dh = upload(hidden);
  bf16* dres = upload(residual);
  bf16* dnw = upload(nw);
  bf16* dwqkv = upload(wqkv);
  bf16* dbias = upload(bias);
  bf16* dqnw = upload(qnw);
  bf16* dknw = upload(knw);
  bf16* dwo = upload(w_o);
  bf16* dk = upload(old_kv.k);
  bf16* dv = upload(old_kv.v);
  float* dcs = upload(cache);
  int64_t* dpos = upload(positions);
  int64_t* dslot = upload(slots);
  int32_t* btab = upload(std::vector<int32_t>{0});
  int32_t* seqlens = upload(std::vector<int32_t>{POS0 + M});
  bf16* dout = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);

  kops::kv::PagedKVDesc kvd{};
  kvd.k_data = dk;
  kvd.v_data = dv;
  kvd.k_dtype = kops::DType::kBFloat16;
  kvd.v_dtype = kops::DType::kBFloat16;
  kvd.layout = kops::kv::KVLayout::kNHD;
  kvd.page_size = PS;
  kvd.num_kv_heads = Hkv;
  kvd.head_dim_k = D;
  kvd.head_dim_v = D;
  kvd.k_stride_page = PS * Hkv * D;
  kvd.k_stride_token = Hkv * D;
  kvd.k_stride_head = D;
  kvd.v_stride_page = PS * Hkv * D;
  kvd.v_stride_token = Hkv * D;
  kvd.v_stride_head = D;
  kvd.table_format = kops::kv::PageTableFormat::kBlockTable;
  kvd.block_table = btab;
  kvd.block_table_stride = 1;
  kvd.seq_lens = seqlens;

  kops::mega::MegaAttentionBlockParams p{};
  p.hidden = dh;
  p.residual = dres;
  p.in_norm_weight = dnw;
  p.norm_eps = kEps;
  p.w_qkv = dwqkv;
  p.qkv_bias = dbias;
  p.q_norm_weight = dqnw;
  p.k_norm_weight = dknw;
  p.positions = dpos;
  p.cos_sin_cache = dcs;
  p.rot_dim = D;
  p.rot_offset = 0;
  p.rope_style = kops::RopeStyle::kNeoX;
  p.kv = kvd;
  p.slot_mapping = dslot;
  p.mod = kops::attn::AttentionModifiers{};
  p.mod.sm_scale = sm;
  p.w_o = dwo;
  p.out = dout;
  p.m = M;
  p.hidden_size = H;
  p.num_q_heads = Hq;
  p.num_kv_heads = Hkv;
  p.head_dim = D;

  PlanBuf pb;
  if (!make_plan(MegaKind::kAttentionBlock, &p, ctx, &pb, "mega_prepare(attention)")) return;
  std::printf("[mega/attn] ctas=%d tasks=%d regs=%d smem=%d launches=%d\n", pb.plan.info.ctas,
              pb.plan.info.num_tasks, pb.plan.info.regs_per_thread, pb.plan.info.smem_per_cta,
              pb.plan.info.num_device_launches);
  KOPS_EXPECT_OK(kops::mega::mega_run_attention_block(pb.plan, p, ctx), "mega_run_attention_block");
  if (!wait_stream(ctx.stream, "mega attention")) return;
  kops_test::compare(download(dout, static_cast<size_t>(M) * H), ref.out, 6e-2, 6e-2,
                     "mega_attention vs fp64 ref");
  kops_test::compare(download(dres, static_cast<size_t>(M) * H), ref.out, 6e-2, 6e-2,
                     "attention residual stream");
  // Appended K/V at positions 4,5: compare the whole page against the fp64 reference.
  const std::vector<bf16> kback = download(dk, static_cast<size_t>(PS) * Hkv * D);
  const std::vector<bf16> vback = download(dv, static_cast<size_t>(PS) * Hkv * D);
  std::vector<double> kfull(static_cast<size_t>(PS) * Hkv * D);
  std::vector<double> vfull(static_cast<size_t>(PS) * Hkv * D);
  for (size_t i = 0; i < kfull.size(); ++i) {
    kfull[i] = static_cast<double>(kops_test::to_float(old_kv.k[i]));
    vfull[i] = static_cast<double>(kops_test::to_float(old_kv.v[i]));
  }
  for (int m = 0; m < M; ++m) {
    for (int hk = 0; hk < Hkv; ++hk) {
      for (int d = 0; d < D; ++d) {
        const size_t dst = static_cast<size_t>(POS0 + m) * Hkv * D + hk * D + d;
        const size_t src = (static_cast<size_t>(m) * Hkv + hk) * D + d;
        kfull[dst] = ref.k_app[src];
        vfull[dst] = ref.v_app[src];
      }
    }
  }
  kops_test::compare(kback, kfull, 2e-2, 2e-2, "attention K append");
  kops_test::compare(vback, vfull, 2e-2, 2e-2, "attention V append");

  free_plan(&pb);
  for (void* p2 : {static_cast<void*>(dh), static_cast<void*>(dres), static_cast<void*>(dnw),
                   static_cast<void*>(dwqkv), static_cast<void*>(dbias), static_cast<void*>(dqnw),
                   static_cast<void*>(dknw), static_cast<void*>(dwo), static_cast<void*>(dk),
                   static_cast<void*>(dv), static_cast<void*>(dcs), static_cast<void*>(dpos),
                   static_cast<void*>(dslot), static_cast<void*>(btab), static_cast<void*>(seqlens),
                   static_cast<void*>(dout)}) {
    cudaFree(p2);
  }
}

// ---- MoE reference -------------------------------------------------------------
std::vector<double> moe_ref(const std::vector<bf16>& hidden, const std::vector<bf16>& w13,
                            const std::vector<bf16>& w2, const std::vector<int32_t>& ids,
                            const std::vector<float>& weights, const std::vector<bf16>& shared,
                            int M, int H, int I, int E, int K, float scale,
                            kops::ActFn act) {
  const std::vector<double> xd = to_double(hidden);
  const std::vector<double> w13d = to_double(w13);
  const std::vector<double> w2d = to_double(w2);
  const std::vector<double> shd = to_double(shared);
  std::vector<double> out(static_cast<size_t>(M) * H, 0.0);
  std::vector<double> acc(H), inter(I);
  for (int m = 0; m < M; ++m) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int kk = 0; kk < K; ++kk) {
      const int e = ids[static_cast<size_t>(m) * K + kk];
      if (e < 0 || e >= E) continue;
      const double w = weights[static_cast<size_t>(m) * K + kk];
      for (int i = 0; i < I; ++i) {
        double g = 0.0, u = 0.0;
        for (int k = 0; k < H; ++k) {
          g += xd[static_cast<size_t>(m) * H + k] * w13d[(static_cast<size_t>(e) * 2 * I + i) * H + k];
          u += xd[static_cast<size_t>(m) * H + k] * w13d[(static_cast<size_t>(e) * 2 * I + I + i) * H + k];
        }
        const double a = act == kops::ActFn::kSiLU ? g / (1.0 + std::exp(-g)) : g;
        inter[i] = a * u;
      }
      for (int h = 0; h < H; ++h) {
        double v = 0.0;
        for (int i = 0; i < I; ++i) v += inter[i] * w2d[(static_cast<size_t>(e) * H + h) * I + i];
        acc[h] += w * v;
      }
    }
    for (int h = 0; h < H; ++h) {
      out[static_cast<size_t>(m) * H + h] = scale * acc[h] + shd[static_cast<size_t>(m) * H + h];
    }
  }
  return out;
}

void test_moe(const LaunchContext& ctx) {
  constexpr int M = 3, H = 64, I = 64, E = 4, K = 2;
  std::vector<float> hf(static_cast<size_t>(M) * H), w13f(static_cast<size_t>(E) * 2 * I * H),
      w2f(static_cast<size_t>(E) * H * I), shf(static_cast<size_t>(M) * H);
  fill(hf, 31, -1.0f, 1.0f);
  fill(w13f, 32, -0.1f, 0.1f);
  fill(w2f, 33, -0.1f, 0.1f);
  fill(shf, 34, -0.2f, 0.2f);
  const std::vector<bf16> hidden = to_bf16(hf), w13 = to_bf16(w13f), w2 = to_bf16(w2f),
                          shared = to_bf16(shf);
  (void)0;
  const std::vector<int32_t> ids = {0, 1, 2, 3, 1, 0};
  const std::vector<float> weights = {0.6f, 0.4f, 0.5f, 0.5f, 0.7f, 0.3f};
  const float scale = 1.0f;
  const std::vector<double> ref = moe_ref(hidden, w13, w2, ids, weights, shared, M, H, I, E, K,
                                          scale, kops::ActFn::kSiLU);

  bf16* dh = upload(hidden);
  bf16* dw13 = upload(w13);
  bf16* dw2 = upload(w2);
  bf16* dsh = upload(shared);
  int32_t* dids = upload(ids);
  float* dwts = upload(weights);
  bf16* dout = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);
  bf16* dep_stage = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);

  kops::mega::MegaMoELocalParams p{};
  p.hidden = dh;
  p.topk_ids = dids;
  p.topk_weights = dwts;
  p.w13 = dw13;
  p.w2 = dw2;
  p.w_quant.storage_dtype = kops::DType::kBFloat16;
  p.act = kops::ActFn::kSiLU;
  p.gate_layout = kops::GateLayout::kGateUp;
  p.out = dout;
  p.shared_out = dsh;
  p.routed_scaling_factor = scale;
  p.m = M;
  p.hidden_size = H;
  p.intermediate_size = I;
  p.num_local_experts = E;
  p.topk = K;

  PlanBuf pb;
  if (!make_plan(MegaKind::kMoELocal, &p, ctx, &pb, "mega_prepare(moe)")) return;
  std::printf("[mega/moe] ctas=%d tasks=%d regs=%d smem=%d\n", pb.plan.info.ctas,
              pb.plan.info.num_tasks, pb.plan.info.regs_per_thread, pb.plan.info.smem_per_cta);
  KOPS_EXPECT_OK(kops::mega::mega_run_moe_local(pb.plan, p, ctx), "mega_run_moe_local");
  if (!wait_stream(ctx.stream, "mega moe")) return;
  kops_test::compare(download(dout, static_cast<size_t>(M) * H), ref, 6e-2, 6e-2,
                     "mega_moe_local vs fp64 ref");

  // Fused router is a separate capability: with topk_ids == null, prepare rejects.
  kops::mega::MegaMoELocalParams p_fused = p;
  p_fused.topk_ids = nullptr;
  WorkspaceRequirements req{};
  KOPS_EXPECT_STATUS(
      kops::mega::mega_workspace_requirements(MegaKind::kMoELocal, &p_fused, ctx, &req),
      Status::kUnsupportedBackend, "moe fused router rejected");

  // ---- EP single virtual rank ----
  kops::mega::MegaMoEEPParams ep{};
  ep.local = p;
  ep.local.out = dep_stage;   // EP body writes through the staged buffer
  ep.ep.rank = 0;
  ep.ep.num_ranks = 1;
  ep.ep.num_local_ranks = 1;
  ep.ep.transport = kops::comm::EPTransport::kNvlink;
  ep.ep.nvl_buffers[0] = dep_stage;
  ep.ep.nvl_bytes = static_cast<size_t>(M) * H * sizeof(bf16);
  ep.payload_quant = kops::ActQuantMode::kNone;
  ep.max_tokens_per_rank = M;
  ep.epoch = 1;
  PlanBuf pe;
  if (!make_plan(MegaKind::kMoEEP, &ep, ctx, &pe, "mega_prepare(moe_ep)")) return;
  KOPS_EXPECT_OK(kops::mega::mega_run_moe_ep(pe.plan, ep, ctx), "mega_run_moe_ep");
  if (!wait_stream(ctx.stream, "mega moe ep")) return;
  kops_test::compare(download(dep_stage, static_cast<size_t>(M) * H), ref, 6e-2, 6e-2,
                     "mega_moe_ep vs fp64 ref");
  kops::mega::MegaMoEEPParams ep2 = ep;
  ep2.ep.num_ranks = 2;
  KOPS_EXPECT_STATUS(kops::mega::mega_workspace_requirements(MegaKind::kMoEEP, &ep2, ctx, &req),
                     Status::kUnsupportedBackend, "moe_ep multi-rank rejected");

  free_plan(&pe);
  free_plan(&pb);
  for (void* q : {static_cast<void*>(dh), static_cast<void*>(dw13), static_cast<void*>(dw2),
                  static_cast<void*>(dsh), static_cast<void*>(dids), static_cast<void*>(dwts),
                  static_cast<void*>(dout), static_cast<void*>(dep_stage)}) {
    cudaFree(q);
  }
}

// ---- decoder test --------------------------------------------------------------
void test_decoder(const LaunchContext& ctx) {
  constexpr int M = 2, H = 64, Hq = 4, Hkv = 2, D = 32, PS = 16, POS0 = 4, I = 128;
  const int total = (Hq + 2 * Hkv) * D;
  std::vector<float> hf(static_cast<size_t>(M) * H), rf(static_cast<size_t>(M) * H),
      nwf(H), wqf(static_cast<size_t>(total) * H), bf(static_cast<size_t>(total)),
      qnwf(D), knwf(D), wof(static_cast<size_t>(H) * Hq * D),
      w13f(static_cast<size_t>(2 * I) * H), w2f(static_cast<size_t>(H) * I),
      mlpnwf(H);
  fill(hf, 41, -1.0f, 1.0f);
  fill(rf, 42, -0.5f, 0.5f);
  fill(nwf, 43, 0.8f, 1.2f);
  fill(wqf, 44, -0.08f, 0.08f);
  fill(bf, 45, -0.1f, 0.1f);
  fill(qnwf, 46, 0.8f, 1.2f);
  fill(knwf, 47, 0.8f, 1.2f);
  fill(wof, 48, -0.08f, 0.08f);
  fill(w13f, 49, -0.08f, 0.08f);
  fill(w2f, 50, -0.08f, 0.08f);
  fill(mlpnwf, 51, 0.8f, 1.2f);
  const std::vector<bf16> hidden = to_bf16(hf), residual = to_bf16(rf), nw = to_bf16(nwf),
                          wqkv = to_bf16(wqf), bias = to_bf16(bf), qnw = to_bf16(qnwf),
                          knw = to_bf16(knwf), w_o = to_bf16(wof), w13 = to_bf16(w13f),
                          w2 = to_bf16(w2f), mlpnw = to_bf16(mlpnwf);
  std::vector<int64_t> positions(M);
  std::vector<int64_t> slots(M);
  for (int m = 0; m < M; ++m) {
    positions[m] = POS0 + m;
    slots[m] = POS0 + m;
  }
  std::vector<float> cache(static_cast<size_t>(PS) * D, 0.0f);
  const float theta = 10000.0f;
  for (int p = 0; p < PS; ++p) {
    for (int e = 0; e < D / 2; ++e) {
      const double inv = std::pow(theta, -2.0 * e / D);
      cache[static_cast<size_t>(p) * D + e] = static_cast<float>(std::cos(p * inv));
      cache[static_cast<size_t>(p) * D + D / 2 + e] = static_cast<float>(std::sin(p * inv));
    }
  }
  KVBuf old_kv;
  old_kv.page_size = PS;
  old_kv.num_kv_heads = Hkv;
  old_kv.head_dim = D;
  old_kv.k.assign(static_cast<size_t>(PS) * Hkv * D, __float2bfloat16(0.0f));
  old_kv.v.assign(static_cast<size_t>(PS) * Hkv * D, __float2bfloat16(0.0f));
  std::vector<float> okf(static_cast<size_t>(PS) * Hkv * D), ovf(static_cast<size_t>(PS) * Hkv * D);
  fill(okf, 52, -0.5f, 0.5f);
  fill(ovf, 53, -0.5f, 0.5f);
  for (int pos = 0; pos < POS0; ++pos) {
    for (int hk = 0; hk < Hkv; ++hk) {
      for (int d = 0; d < D; ++d) {
        const size_t at = old_kv.idx(pos, hk, d);
        old_kv.k[at] = __float2bfloat16_rn(okf[at]);
        old_kv.v[at] = __float2bfloat16_rn(ovf[at]);
      }
    }
  }

  auto make_attn = [&](bf16* dh, bf16* dres, bf16* dout, bf16* dnw, bf16* dwqkv, bf16* dbias,
                       bf16* dqnw, bf16* dknw, bf16* dwo, bf16* dk, bf16* dv, float* dcs,
                       int64_t* dpos, int64_t* dslot, int32_t* btab, int32_t* seqlens,
                       kops::kv::PagedKVDesc* kvd) {
    kvd->k_data = dk;
    kvd->v_data = dv;
    kvd->k_dtype = kops::DType::kBFloat16;
    kvd->v_dtype = kops::DType::kBFloat16;
    kvd->layout = kops::kv::KVLayout::kNHD;
    kvd->page_size = PS;
    kvd->num_kv_heads = Hkv;
    kvd->head_dim_k = D;
    kvd->head_dim_v = D;
    kvd->k_stride_page = PS * Hkv * D;
    kvd->k_stride_token = Hkv * D;
    kvd->k_stride_head = D;
    kvd->v_stride_page = PS * Hkv * D;
    kvd->v_stride_token = Hkv * D;
    kvd->v_stride_head = D;
    kvd->table_format = kops::kv::PageTableFormat::kBlockTable;
    kvd->block_table = btab;
    kvd->block_table_stride = 1;
    kvd->seq_lens = seqlens;
    kops::mega::MegaAttentionBlockParams a{};
    a.hidden = dh;
    a.residual = dres;
    a.in_norm_weight = dnw;
    a.norm_eps = kEps;
    a.w_qkv = dwqkv;
    a.qkv_bias = dbias;
    a.q_norm_weight = dqnw;
    a.k_norm_weight = dknw;
    a.positions = dpos;
    a.cos_sin_cache = dcs;
    a.rot_dim = D;
    a.rot_offset = 0;
    a.rope_style = kops::RopeStyle::kNeoX;
    a.kv = *kvd;
    a.slot_mapping = dslot;
    a.mod = kops::attn::AttentionModifiers{};
    a.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    a.w_o = dwo;
    a.out = dout;
    a.m = M;
    a.hidden_size = H;
    a.num_q_heads = Hq;
    a.num_kv_heads = Hkv;
    a.head_dim = D;
    return a;
  };

  bf16* dh = upload(hidden);
  bf16* dnw = upload(nw);
  bf16* dwqkv = upload(wqkv);
  bf16* dbias = upload(bias);
  bf16* dqnw = upload(qnw);
  bf16* dknw = upload(knw);
  bf16* dwo = upload(w_o);
  bf16* dmlpnw = upload(mlpnw);
  bf16* dw13 = upload(w13);
  bf16* dw2 = upload(w2);
  float* dcs = upload(cache);
  int64_t* dpos = upload(positions);
  int64_t* dslot = upload(slots);
  int32_t* btab = upload(std::vector<int32_t>{0});
  int32_t* seqlens = upload(std::vector<int32_t>{POS0 + M});
  bf16* dk1 = upload(old_kv.k);
  bf16* dv1 = upload(old_kv.v);
  bf16* dk2 = upload(old_kv.k);
  bf16* dv2 = upload(old_kv.v);
  bf16* stream_a = upload(residual);
  bf16* stream_dec = upload(residual);
  bf16* layer_out = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);
  bf16* chain_out = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);
  bf16* chain_inter = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * I);
  bf16* dec_inter = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * I);

  kops::kv::PagedKVDesc kvd1{}, kvd2{};
  kops::mega::MegaAttentionBlockParams attn1 =
      make_attn(dh, stream_a, stream_a, dnw, dwqkv, dbias, dqnw, dknw, dwo, dk1, dv1, dcs, dpos,
                dslot, btab, seqlens, &kvd1);
  kops::mega::MegaAttentionBlockParams attn2 =
      make_attn(dh, stream_dec, stream_dec, dnw, dwqkv, dbias, dqnw, dknw, dwo, dk2, dv2, dcs, dpos,
                dslot, btab, seqlens, &kvd2);

  // Standalone attention block (validated separately above).
  PlanBuf pa;
  if (!make_plan(MegaKind::kAttentionBlock, &attn1, ctx, &pa, "mega_prepare(decoder/attn)")) return;
  KOPS_EXPECT_OK(kops::mega::mega_run_attention_block(pa.plan, attn1, ctx), "decoder/attn run");
  if (!wait_stream(ctx.stream, "decoder attn")) return;

  // Standalone MLP on the updated stream.
  kops::mega::MegaMLPParams mlp{};
  mlp.x = stream_a;
  mlp.ldx = H;
  mlp.pre_norm_weight = dmlpnw;
  mlp.pre_norm_eps = kEps;
  mlp.w13 = dw13;
  mlp.w13_scale = nullptr;
  mlp.w2 = dw2;
  mlp.w2_scale = nullptr;
  mlp.w_quant.storage_dtype = kops::DType::kBFloat16;
  mlp.act = kops::ActFn::kSiLU;
  mlp.gate_layout = kops::GateLayout::kGateUp;
  mlp.intermediate = chain_inter;
  mlp.out = chain_out;
  mlp.ldo = H;
  mlp.residual = stream_a;
  mlp.m = M;
  mlp.hidden_size = H;
  mlp.intermediate_size = I;
  PlanBuf pm;
  if (!make_plan(MegaKind::kMLP, &mlp, ctx, &pm, "mega_prepare(decoder/mlp)")) return;
  KOPS_EXPECT_OK(kops::mega::mega_run_mlp(pm.plan, mlp, ctx), "decoder/mlp run");
  if (!wait_stream(ctx.stream, "decoder mlp")) return;

  // ---- decoder DAG ----
  kops::mega::MegaMLPParams mlp_dec = mlp;
  mlp_dec.x = stream_dec;
  mlp_dec.residual = stream_dec;
  mlp_dec.out = layer_out;
  mlp_dec.intermediate = dec_inter;
  kops::mega::MegaDecoderParams dp{};
  dp.attn = attn2;
  dp.mlp = mlp_dec;
  dp.moe = kops::mega::MegaMoELocalParams{};
  PlanBuf pd;
  if (!make_plan(MegaKind::kDecoder, &dp, ctx, &pd, "mega_prepare(decoder)")) return;
  std::printf("[mega/dec] ctas=%d tasks=%d regs=%d smem=%d launches=%d\n", pd.plan.info.ctas,
              pd.plan.info.num_tasks, pd.plan.info.regs_per_thread, pd.plan.info.smem_per_cta,
              pd.plan.info.num_device_launches);
  KOPS_EXPECT_OK(kops::mega::mega_run_decoder(pd.plan, attn2, mlp_dec, dp.moe, ctx),
                 "mega_run_decoder");
  if (!wait_stream(ctx.stream, "decoder")) return;
  const std::vector<bf16> chain_v = download(chain_out, static_cast<size_t>(M) * H);
  const std::vector<double> chain_d = to_double(chain_v);
  kops_test::compare(download(layer_out, static_cast<size_t>(M) * H), chain_d, 2e-2, 2e-2,
                     "decoder out vs standalone chain");
  kops_test::compare(download(stream_dec, static_cast<size_t>(M) * H), chain_d, 2e-2, 2e-2,
                     "decoder residual stream vs standalone chain");

  free_plan(&pd);
  free_plan(&pm);
  free_plan(&pa);
  for (void* q : {static_cast<void*>(dh), static_cast<void*>(dnw), static_cast<void*>(dwqkv),
                  static_cast<void*>(dbias), static_cast<void*>(dqnw), static_cast<void*>(dknw),
                  static_cast<void*>(dwo), static_cast<void*>(dmlpnw), static_cast<void*>(dw13),
                  static_cast<void*>(dw2), static_cast<void*>(dcs), static_cast<void*>(dpos),
                  static_cast<void*>(dslot), static_cast<void*>(btab), static_cast<void*>(seqlens),
                  static_cast<void*>(dk1), static_cast<void*>(dv1), static_cast<void*>(dk2),
                  static_cast<void*>(dv2), static_cast<void*>(stream_a), static_cast<void*>(stream_dec),
                  static_cast<void*>(layer_out), static_cast<void*>(chain_out),
                  static_cast<void*>(chain_inter), static_cast<void*>(dec_inter)}) {
    cudaFree(q);
  }
}


// ---- decoder with a local MoE FFN ----------------------------------------------
void test_decoder_moe(const LaunchContext& ctx) {
  constexpr int M = 2, H = 32, Hq = 2, Hkv = 1, D = 16, PS = 16, POS0 = 4, I = 64, E = 2, K = 1;
  const int total = (Hq + 2 * Hkv) * D;
  std::vector<float> hf(static_cast<size_t>(M) * H), rf(static_cast<size_t>(M) * H),
      nwf(H), wqf(static_cast<size_t>(total) * H), bf(static_cast<size_t>(total)),
      qnwf(D), knwf(D), wof(static_cast<size_t>(H) * Hq * D),
      w13f(static_cast<size_t>(E) * 2 * I * H), w2f(static_cast<size_t>(E) * H * I);
  fill(hf, 61, -1.0f, 1.0f);
  fill(rf, 62, -0.5f, 0.5f);
  fill(nwf, 63, 0.8f, 1.2f);
  fill(wqf, 64, -0.08f, 0.08f);
  fill(bf, 65, -0.1f, 0.1f);
  fill(qnwf, 66, 0.8f, 1.2f);
  fill(knwf, 67, 0.8f, 1.2f);
  fill(wof, 68, -0.08f, 0.08f);
  fill(w13f, 69, -0.1f, 0.1f);
  fill(w2f, 70, -0.1f, 0.1f);
  const std::vector<bf16> hidden = to_bf16(hf), residual = to_bf16(rf), nw = to_bf16(nwf),
                          wqkv = to_bf16(wqf), bias = to_bf16(bf), qnw = to_bf16(qnwf),
                          knw = to_bf16(knwf), w_o = to_bf16(wof), w13 = to_bf16(w13f),
                          w2 = to_bf16(w2f);
  std::vector<int64_t> positions = {POS0, POS0 + 1};
  std::vector<int64_t> slots = {POS0, POS0 + 1};
  std::vector<int32_t> ids = {0, 1};
  std::vector<float> weights = {1.0f, 1.0f};
  std::vector<float> cache(static_cast<size_t>(PS) * D, 0.0f);
  for (int p = 0; p < PS; ++p) {
    for (int e = 0; e < D / 2; ++e) {
      const double inv = std::pow(10000.0f, -2.0 * e / D);
      cache[static_cast<size_t>(p) * D + e] = static_cast<float>(std::cos(p * inv));
      cache[static_cast<size_t>(p) * D + D / 2 + e] = static_cast<float>(std::sin(p * inv));
    }
  }
  KVBuf kv0;
  kv0.page_size = PS;
  kv0.num_kv_heads = Hkv;
  kv0.head_dim = D;
  kv0.k.assign(static_cast<size_t>(PS) * Hkv * D, __float2bfloat16(0.0f));
  kv0.v.assign(static_cast<size_t>(PS) * Hkv * D, __float2bfloat16(0.0f));

  bf16* dh = upload(hidden);
  bf16* dnw = upload(nw);
  bf16* dwqkv = upload(wqkv);
  bf16* dbias = upload(bias);
  bf16* dqnw = upload(qnw);
  bf16* dknw = upload(knw);
  bf16* dwo = upload(w_o);
  bf16* dw13 = upload(w13);
  bf16* dw2 = upload(w2);
  float* dcs = upload(cache);
  int64_t* dpos = upload(positions);
  int64_t* dslot = upload(slots);
  int32_t* dids = upload(ids);
  float* dwts = upload(weights);
  int32_t* btab = upload(std::vector<int32_t>{0});
  int32_t* seqlens = upload(std::vector<int32_t>{POS0 + M});
  bf16* dk1 = upload(kv0.k);
  bf16* dv1 = upload(kv0.v);
  bf16* dk2 = upload(kv0.k);
  bf16* dv2 = upload(kv0.v);
  bf16* stream_a = upload(residual);
  bf16* stream_dec = upload(residual);
  bf16* normed = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);
  bf16* chain_out = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);
  bf16* layer_out = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * H);
  bf16* dec_moe_inter = kops_test::device_alloc<bf16>(static_cast<size_t>(M) * I);

  auto make_attn = [&](bf16* dres, bf16* dk, bf16* dv) {
    kops::kv::PagedKVDesc kvd{};
    kvd.k_data = dk;
    kvd.v_data = dv;
    kvd.k_dtype = kops::DType::kBFloat16;
    kvd.v_dtype = kops::DType::kBFloat16;
    kvd.layout = kops::kv::KVLayout::kNHD;
    kvd.page_size = PS;
    kvd.num_kv_heads = Hkv;
    kvd.head_dim_k = D;
    kvd.head_dim_v = D;
    kvd.k_stride_page = PS * Hkv * D;
    kvd.k_stride_token = Hkv * D;
    kvd.k_stride_head = D;
    kvd.v_stride_page = PS * Hkv * D;
    kvd.v_stride_token = Hkv * D;
    kvd.v_stride_head = D;
    kvd.table_format = kops::kv::PageTableFormat::kBlockTable;
    kvd.block_table = btab;
    kvd.block_table_stride = 1;
    kvd.seq_lens = seqlens;
    kops::mega::MegaAttentionBlockParams a{};
    a.hidden = dh;
    a.residual = dres;
    a.out = dres;
    a.in_norm_weight = dnw;
    a.norm_eps = kEps;
    a.w_qkv = dwqkv;
    a.qkv_bias = dbias;
    a.q_norm_weight = dqnw;
    a.k_norm_weight = dknw;
    a.positions = dpos;
    a.cos_sin_cache = dcs;
    a.rot_dim = D;
    a.rope_style = kops::RopeStyle::kNeoX;
    a.kv = kvd;
    a.slot_mapping = dslot;
    a.mod = kops::attn::AttentionModifiers{};
    a.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    a.w_o = dwo;
    a.m = M;
    a.hidden_size = H;
    a.num_q_heads = Hq;
    a.num_kv_heads = Hkv;
    a.head_dim = D;
    return a;
  };
  kops::mega::MegaAttentionBlockParams attn1 = make_attn(stream_a, dk1, dv1);
  kops::mega::MegaAttentionBlockParams attn2 = make_attn(stream_dec, dk2, dv2);

  PlanBuf pa;
  if (!make_plan(MegaKind::kAttentionBlock, &attn1, ctx, &pa, "moe-dec/attn plan")) return;
  KOPS_EXPECT_OK(kops::mega::mega_run_attention_block(pa.plan, attn1, ctx), "moe-dec/attn");
  if (!wait_stream(ctx.stream, "moe decoder attn")) return;

  const kops::Tensor stream_t = kops_test::tensor2d(stream_a, kops::DType::kBFloat16, M, H);
  const kops::Tensor norm_t = kops_test::tensor2d(normed, kops::DType::kBFloat16, M, H);
  const kops::Tensor nw_t = tensor1d(dnw, kops::DType::kBFloat16, H);
  KOPS_EXPECT_OK(kops::norm::rms_norm(norm_t, stream_t, nw_t, kEps, 0.0f, ctx), "moe-dec norm");

  kops::mega::MegaMoELocalParams moe{};
  moe.hidden = normed;
  moe.topk_ids = dids;
  moe.topk_weights = dwts;
  moe.w13 = dw13;
  moe.w2 = dw2;
  moe.w_quant.storage_dtype = kops::DType::kBFloat16;
  moe.act = kops::ActFn::kSiLU;
  moe.gate_layout = kops::GateLayout::kGateUp;
  moe.out = chain_out;
  moe.routed_scaling_factor = 1.0f;
  moe.m = M;
  moe.hidden_size = H;
  moe.intermediate_size = I;
  moe.num_local_experts = E;
  moe.topk = K;
  PlanBuf pm;
  if (!make_plan(MegaKind::kMoELocal, &moe, ctx, &pm, "moe-dec/moe plan")) return;
  KOPS_EXPECT_OK(kops::mega::mega_run_moe_local(pm.plan, moe, ctx), "moe-dec/moe run");
  if (!wait_stream(ctx.stream, "moe decoder moe")) return;

  kops::mega::MegaMLPParams mlp{};
  mlp.x = stream_dec;
  mlp.pre_norm_weight = dnw;
  mlp.pre_norm_eps = kEps;
  mlp.w13 = dw13;
  mlp.w2 = dw2;
  mlp.out = layer_out;
  mlp.intermediate = dec_moe_inter;
  mlp.m = M;
  mlp.hidden_size = H;
  mlp.intermediate_size = I;
  kops::mega::MegaMoELocalParams moe_dec = moe;
  moe_dec.hidden = stream_dec;
  moe_dec.out = layer_out;
  kops::mega::MegaDecoderParams dp{};
  dp.attn = attn2;
  dp.mlp = mlp;
  dp.moe = moe_dec;
  PlanBuf pd;
  if (!make_plan(MegaKind::kDecoder, &dp, ctx, &pd, "moe-dec plan")) return;
  KOPS_EXPECT_OK(kops::mega::mega_run_decoder(pd.plan, attn2, mlp, moe_dec, ctx), "moe-dec run");
  if (!wait_stream(ctx.stream, "moe decoder")) return;
  const std::vector<double> chain_d = to_double(download(chain_out, static_cast<size_t>(M) * H));
  kops_test::compare(download(layer_out, static_cast<size_t>(M) * H), chain_d, 4e-2, 4e-2,
                     "moe decoder out vs standalone chain");
  kops_test::compare(download(stream_dec, static_cast<size_t>(M) * H), chain_d, 4e-2, 4e-2,
                     "moe decoder stream vs standalone chain");

  free_plan(&pd);
  free_plan(&pm);
  free_plan(&pa);
  for (void* q : {static_cast<void*>(dh), static_cast<void*>(dnw), static_cast<void*>(dwqkv),
                  static_cast<void*>(dbias), static_cast<void*>(dqnw), static_cast<void*>(dknw),
                  static_cast<void*>(dwo), static_cast<void*>(dw13), static_cast<void*>(dw2),
                  static_cast<void*>(dcs), static_cast<void*>(dpos), static_cast<void*>(dslot),
                  static_cast<void*>(dids), static_cast<void*>(dwts), static_cast<void*>(btab),
                  static_cast<void*>(seqlens), static_cast<void*>(dk1), static_cast<void*>(dv1),
                  static_cast<void*>(dk2), static_cast<void*>(dv2), static_cast<void*>(stream_a),
                  static_cast<void*>(stream_dec), static_cast<void*>(normed),
                  static_cast<void*>(chain_out), static_cast<void*>(layer_out),
                  static_cast<void*>(dec_moe_inter)}) {
    cudaFree(q);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const kops_test::Options opts = kops_test::parse_options(argc, argv);
  (void)opts;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::printf("SKIP test_mega: no CUDA device\n");
    return 77;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || prop.major * 10 + prop.minor < 80) {
    std::printf("SKIP test_mega: SM80+ required (got %d.%d)\n", prop.major, prop.minor);
    return 77;
  }
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 1;
  const kops::kops_stream_t stream = ctx.stream;
  (void)stream;

  // Shared scratch for the modular GEMM oracle.
  Workspace gws{};
  gws.device_scratch_bytes = size_t(64) << 20;
  if (cudaMalloc(&gws.device_scratch, gws.device_scratch_bytes) != cudaSuccess) {
    std::printf("SKIP test_mega: cannot allocate modular GEMM scratch\n");
    return 77;
  }

  test_mlp(ctx, gws);
  test_attention(ctx);
  test_moe(ctx);
  test_decoder(ctx);
  test_decoder_moe(ctx);

  cudaFree(gws.device_scratch);
  return kops_test::finish("test_mega");
}
