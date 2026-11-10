// =============================================================================
// kernel-ops | benchmark/bench_mega.cu
// B16 megakernel latency versus the modular chain and a captured CUDA graph.
//
// Cases (all BF16 weights/activations; one dedicated stream):
//   mega.mlp      m=8  H=1024 I=2048   norm + W13 + GLU + W2 + residual
//   mega.attn     m=1  H=1024 Hq=8 Hkv=2 D=128 kv=64   QKV -> qk-norm/rope ->
//                 KV append -> paged decode -> O-proj -> residual
//   mega.moe      m=8  H=512 I=512 E=8 topk=2          routed W13/W2/combine
//   mega.decoder  m=1  H=512 I=1024 attention block + dense MLP
//
// Each case reports four rows: mega, mega captured in a CUDA graph, modular,
// modular captured in a CUDA graph.  PDL is deliberately not reported: the
// megakernel is one launch and the reset launch does not use griddepcontrol, so
// there is nothing to compare on SM120a.  `--dump-cases` exports the table.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

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
#include "csrc/gemm/ops.h"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/merge_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"
#include "csrc/position/rope_kernels.cuh"
#include "csrc/position/ops.h"
#include "csrc/kvcache/kvcache_kernels.cuh"
#include "csrc/kvcache/ops.h"
#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"
#include "csrc/activation/activation_kernels.cuh"
#include "csrc/activation/ops.h"
#include "csrc/moe/fused_moe_kernels.cuh"
#include "csrc/moe/permute_kernels.cuh"
#include "csrc/moe/ops.h"
#include "csrc/execution/persistent.cuh"
#include "csrc/mega/mega_plan.cuh"
#include "csrc/mega/mega_kernels.cuh"
#include "csrc/mega/ops.h"

namespace {

using namespace kops_bench;
using bf16 = __nv_bfloat16;

struct DevBuf {
  void* p = nullptr;
  size_t n = 0;
  void alloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
      std::fprintf(stderr, "[bench] cudaMalloc(%zu) failed\n", bytes);
      p = nullptr;
    }
    n = bytes;
  }
  ~DevBuf() {
    if (p != nullptr) cudaFree(p);
  }
};

void fill_bf16(void* dst, size_t n, unsigned seed, float lo, float hi) {
  std::vector<bf16> h(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  for (auto& x : h) x = __float2bfloat16_rn(d(rng));
  cudaMemcpy(dst, h.data(), n * sizeof(bf16), cudaMemcpyHostToDevice);
}

template <typename T>
T* up(T* dst, const std::vector<T>& v) {
  cudaMemcpy(dst, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return dst;
}

kops::Tensor t2(void* data, kops::DType dt, int64_t r, int64_t c) {
  return kops_bench::tensor2d(data, dt, r, c);
}
kops::Tensor t3s(void* data, kops::DType dt, int64_t d0, int64_t d1, int64_t d2, int64_t s0,
                 int64_t s1) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dt;
  t.ndim = 3;
  t.device = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.stride[0] = s0;
  t.stride[1] = s1;
  t.stride[2] = 1;
  return t;
}

bool gemm_bf16(const void* a, int64_t lda, const void* b, int64_t ldb, void* d, int64_t ldd,
               int64_t m, int64_t n, int64_t k, const kops::gemm::EpilogueParams& epi,
               const kops::Workspace& ws, const kops::LaunchContext& ctx) {
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
  return kops::gemm::mma_gemm(args, nullptr, ws, ctx) == kops::Status::kSuccess;
}

double max_abs_diff(const void* a, const void* b, size_t n, const char* what) {
  std::vector<bf16> ha(n), hb(n);
  cudaMemcpy(ha.data(), a, n * sizeof(bf16), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), b, n * sizeof(bf16), cudaMemcpyDeviceToHost);
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = std::fabs(double(__bfloat162float(ha[i])) - double(__bfloat162float(hb[i])));
    if (!(d <= worst)) worst = d;
  }
  std::fprintf(stderr, "[bench] %s: max|mega-modular| = %.5g\n", what, worst);
  return worst;
}

bool capture(cudaStream_t stream, const std::function<bool()>& fn, cudaGraphExec_t* exec,
             const char* what) {
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    std::fprintf(stderr, "[bench] %s: begin capture failed\n", what);
    return false;
  }
  if (!fn()) {
    std::fprintf(stderr, "[bench] %s: capture body failed: %s\n", what, kops::last_error_message());
    cudaStreamEndCapture(stream, nullptr);
    return false;
  }
  cudaGraph_t graph = nullptr;
  if (cudaStreamEndCapture(stream, &graph) != cudaSuccess) {
    std::fprintf(stderr, "[bench] %s: end capture failed\n", what);
    return false;
  }
  if (cudaGraphInstantiate(exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
    std::fprintf(stderr, "[bench] %s: instantiate failed\n", what);
    cudaGraphDestroy(graph);
    return false;
  }
  cudaGraphDestroy(graph);
  return true;
}

// -----------------------------------------------------------------------------
// Modular chains
// -----------------------------------------------------------------------------
struct ModularMLP {
  int M = 0, H = 0, I = 0;
  DevBuf normed, gateup, inter, out;
  void* x = nullptr;
  const void* nw = nullptr;
  const void* w13 = nullptr;
  const void* w2 = nullptr;
  const void* residual = nullptr;
  kops::Workspace ws;
  kops::GluParams glu{};

  bool run(const kops::LaunchContext& ctx) {
    if (kops::norm::rms_norm(t2(normed.p, kops::DType::kBFloat16, M, H),
                             t2(x, kops::DType::kBFloat16, M, H),
                             kops_bench::tensor1d(const_cast<void*>(nw), kops::DType::kBFloat16, H),
                             1e-5f, 0.0f, ctx) != kops::Status::kSuccess) {
      return false;
    }
    if (!gemm_bf16(normed.p, H, w13, H, gateup.p, 2 * I, M, 2 * I, H, {}, ws, ctx)) return false;
    if (kops::act::act_and_mul(t2(inter.p, kops::DType::kBFloat16, M, I),
                               t2(gateup.p, kops::DType::kBFloat16, M, 2 * I), kops::Tensor{},
                               kops::ActFn::kSiLU, kops::GateLayout::kGateUp, glu, ctx) !=
        kops::Status::kSuccess) {
      return false;
    }
    kops::gemm::EpilogueParams epi{};
    epi.ops = kops::gemm::kEpiResidualAdd;
    epi.beta = 1.0f;
    epi.residual = residual;
    epi.residual_stride = H;
    return gemm_bf16(inter.p, I, w2, I, out.p, H, M, H, I, epi, ws, ctx);
  }
};

struct ModularAttention {
  int M = 0, H = 0, Hq = 0, Hkv = 0, D = 0, PS = 0, kv_len = 0;
  DevBuf normed, qkv, attn, out, kcache, vcache, cache_cs;
  const void* hidden = nullptr;
  const void* nw = nullptr;
  const void* wqkv = nullptr;
  const void* qnw = nullptr;
  const void* knw = nullptr;
  const void* wo = nullptr;
  const void* residual = nullptr;
  int64_t* positions = nullptr;
  int64_t* slots = nullptr;
  int32_t* block_table = nullptr;
  int32_t* seq_lens = nullptr;
  kops::kv::PagedKVDesc kv{};
  kops::attn::AttentionPlan plan{};
  kops::Workspace ws{};
  kops::Workspace gws{};
  bool ready = false;

  bool prepare(const kops::LaunchContext& ctx) {
    kops::attn::AttentionSpec spec{};
    spec.num_q_heads = Hq;
    spec.num_kv_heads = Hkv;
    spec.head_dim_qk = D;
    spec.head_dim_vo = D;
    spec.q_dtype = kops::DType::kBFloat16;
    spec.kv_dtype = kops::DType::kBFloat16;
    spec.o_dtype = kops::DType::kBFloat16;
    spec.kv_layout = kops::kv::KVLayout::kNHD;
    spec.page_size = PS;
    spec.mask = kops::attn::MaskMode::kCausal;
    kops::LaunchContext pctx = ctx;
    pctx.batch_invariant = true;   // split_kv = false: no merge launch in the baseline
    kops::Status st = kops::attn::decode_plan(spec, 1, kv_len, M, false, pctx, ws, &plan);
    if (st == kops::Status::kWorkspaceTooSmall) {
      ws.device_persistent_bytes = plan.workspace.device_persistent_bytes;
      ws.device_scratch_bytes = plan.workspace.device_scratch_bytes;
      if (ws.device_persistent_bytes > 0) cudaMalloc(&ws.device_persistent, ws.device_persistent_bytes);
      if (ws.device_scratch_bytes > 0) cudaMalloc(&ws.device_scratch, ws.device_scratch_bytes);
      st = kops::attn::decode_plan(spec, 1, kv_len, M, false, pctx, ws, &plan);
    }
    if (st != kops::Status::kSuccess) {
      std::fprintf(stderr, "[bench] decode_plan: %s\n", kops::last_error_message());
      return false;
    }
    ready = true;
    return true;
  }

  bool run(const kops::LaunchContext& ctx) {
    const int total = (Hq + 2 * Hkv) * D;
    if (kops::norm::rms_norm(t2(normed.p, kops::DType::kBFloat16, M, H),
                             t2(const_cast<void*>(hidden), kops::DType::kBFloat16, M, H),
                             kops_bench::tensor1d(const_cast<void*>(nw), kops::DType::kBFloat16, H),
                             1e-5f, 0.0f, ctx) != kops::Status::kSuccess) {
      return false;
    }
    if (!gemm_bf16(normed.p, H, wqkv, H, qkv.p, total, M, total, H, {}, gws, ctx)) return false;
    if (kops::rope::fused_qk_norm_rope(
            t2(qkv.p, kops::DType::kBFloat16, M, total),
            kops_bench::tensor1d(const_cast<void*>(qnw), kops::DType::kBFloat16, D),
            kops_bench::tensor1d(const_cast<void*>(knw), kops::DType::kBFloat16, D),
            kops_bench::tensor1d(positions, kops::DType::kInt64, M),
            kops_bench::tensor2d(cache_cs.p, kops::DType::kFloat32, kv_len, D), Hq, Hkv, D, D, 0,
            kops::RopeStyle::kNeoX, 0.0f, 1e-5f, ctx) != kops::Status::kSuccess) {
      return false;
    }
    kops::Tensor kview = t3s(static_cast<char*>(qkv.p) + static_cast<int64_t>(Hq) * D * 2,
                             kops::DType::kBFloat16, M, Hkv, D, total, D);
    kops::Tensor vview = t3s(static_cast<char*>(qkv.p) + static_cast<int64_t>(Hq + Hkv) * D * 2,
                             kops::DType::kBFloat16, M, Hkv, D, total, D);
    if (kops::kv::reshape_and_cache(kview, vview, kv,
                                    kops_bench::tensor1d(slots, kops::DType::kInt64, M),
                                    ctx) != kops::Status::kSuccess) {
      return false;
    }
    kops::attn::DecodeParams dp{};
    dp.q = qkv.p;
    dp.q_stride_n = total;
    dp.q_stride_h = D;
    dp.q_len = M;
    dp.kv = kv;
    dp.mod = kops::attn::AttentionModifiers{};
    dp.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    dp.o = attn.p;
    dp.o_stride_n = Hq * D;
    dp.o_stride_h = D;
    dp.batch_size = 1;
    dp.num_qo_heads = Hq;
    dp.num_kv_heads = Hkv;
    if (kops::attn::decode_run(plan, dp, ws, ctx) != kops::Status::kSuccess) return false;
    kops::gemm::EpilogueParams epi{};
    epi.ops = kops::gemm::kEpiResidualAdd;
    epi.beta = 1.0f;
    epi.residual = residual;
    epi.residual_stride = H;
    return gemm_bf16(attn.p, Hq * D, wo, Hq * D, out.p, H, M, H, Hq * D, epi, gws, ctx);
  }
};

// -----------------------------------------------------------------------------
// Case data
// -----------------------------------------------------------------------------
struct MlpCase {
  int M = 8, H = 1024, I = 2048;
  DevBuf x, nw, w13, w2, residual, inter, out_mega, out_mod;
  kops::mega::MegaMLPParams p{};
  kops::mega::MegaPlan plan{};
  kops::Workspace ws{};
  ModularMLP mod;
};

struct AttnCase {
  int M = 1, H = 1024, Hq = 8, Hkv = 2, D = 128, PS = 64, kv_len = 64;
  DevBuf hidden, nw, wqkv, qnw, knw, wo, residual, qkv, attn, out_mega, out_mod;
  DevBuf k, v, cs, positions, slots, btab, seq;
  kops::mega::MegaAttentionBlockParams p{};
  kops::mega::MegaPlan plan{};
  kops::Workspace ws{};
  ModularAttention mod;
};

struct MoeCase {
  int M = 8, H = 512, I = 512, E = 8, K = 2;
  DevBuf hidden, w13, w2, ids, weights, out_mega, out_mod;
  kops::mega::MegaMoELocalParams p{};
  kops::mega::MegaPlan plan{};
  kops::Workspace ws{};
};

struct DecCase {
  int M = 1, H = 512, I = 1024, Hq = 8, Hkv = 2, D = 64, PS = 32, kv_len = 32;
  DevBuf hidden, anw, wqkv, qnw, knw, wo, mnw, w13, w2, residual, qkv, attn, inter, out_mega,
      out_mod;
  DevBuf k, v, cs, positions, slots, btab, seq;
  kops::mega::MegaAttentionBlockParams attn_p{};
  kops::mega::MegaMLPParams mlp_p{};
  kops::mega::MegaPlan plan{};
  kops::Workspace ws{};
  ModularAttention mod_attn;
  ModularMLP mod_mlp;
};

void add_case(std::vector<Case>* cases, const char* id, const char* op, int64_t rows, int64_t cols,
              int64_t bytes, int64_t flops) {
  cases->push_back({id, op, kops::DType::kBFloat16, rows, cols, bytes, flops, 10, 100});
}

// -----------------------------------------------------------------------------
// MLP
// -----------------------------------------------------------------------------
bool setup_mlp(MlpCase* c, const kops::LaunchContext& ctx) {
  const size_t x_n = size_t(c->M) * c->H, w13_n = size_t(2 * c->I) * c->H, w2_n = size_t(c->H) * c->I;
  c->x.alloc(x_n * 2);
  c->nw.alloc(size_t(c->H) * 2);
  c->w13.alloc(w13_n * 2);
  c->w2.alloc(w2_n * 2);
  c->residual.alloc(x_n * 2);
  c->inter.alloc(size_t(c->M) * c->I * 2);
  c->out_mega.alloc(x_n * 2);
  c->out_mod.alloc(x_n * 2);
  fill_bf16(c->x.p, x_n, 1, -1, 1);
  fill_bf16(c->nw.p, c->H, 2, 0.8f, 1.2f);
  fill_bf16(c->w13.p, w13_n, 3, -0.08f, 0.08f);
  fill_bf16(c->w2.p, w2_n, 4, -0.08f, 0.08f);
  fill_bf16(c->residual.p, x_n, 5, -0.5f, 0.5f);

  c->p = kops::mega::MegaMLPParams{};
  c->p.x = c->x.p;
  c->p.ldx = c->H;
  c->p.pre_norm_weight = c->nw.p;
  c->p.pre_norm_eps = 1e-5f;
  c->p.w13 = c->w13.p;
  c->p.w2 = c->w2.p;
  c->p.w_quant.storage_dtype = kops::DType::kBFloat16;
  c->p.act = kops::ActFn::kSiLU;
  c->p.gate_layout = kops::GateLayout::kGateUp;
  c->p.intermediate = c->inter.p;
  c->p.out = c->out_mega.p;
  c->p.ldo = c->H;
  c->p.residual = c->residual.p;
  c->p.m = c->M;
  c->p.hidden_size = c->H;
  c->p.intermediate_size = c->I;
  kops::WorkspaceRequirements req{};
  if (kops::mega::mega_workspace_requirements(kops::mega::MegaKind::kMLP, &c->p, ctx, &req) !=
      kops::Status::kSuccess) {
    return false;
  }
  c->ws.device_persistent_bytes = req.device_persistent_bytes;
  cudaMalloc(&c->ws.device_persistent, c->ws.device_persistent_bytes);
  if (kops::mega::mega_prepare(kops::mega::MegaKind::kMLP, &c->p, c->ws, ctx, &c->plan) !=
      kops::Status::kSuccess) {
    return false;
  }
  c->mod.M = c->M;
  c->mod.H = c->H;
  c->mod.I = c->I;
  c->mod.normed.alloc(x_n * 2);
  c->mod.gateup.alloc(size_t(c->M) * 2 * c->I * 2);
  c->mod.inter.alloc(size_t(c->M) * c->I * 2);
  c->mod.out.alloc(x_n * 2);
  c->mod.x = c->x.p;
  c->mod.nw = c->nw.p;
  c->mod.w13 = c->w13.p;
  c->mod.w2 = c->w2.p;
  c->mod.residual = c->residual.p;
  c->mod.ws.device_scratch_bytes = size_t(64) << 20;
  cudaMalloc(&c->mod.ws.device_scratch, c->mod.ws.device_scratch_bytes);
  return true;
}

void bench_mlp(const kops::LaunchContext& ctx, cudaStream_t stream, std::vector<Result>* results,
               std::vector<Case>* cases) {
  MlpCase c;
  if (!setup_mlp(&c, ctx)) return;
  if (c.mod.run(ctx) &&
      kops::mega::mega_run_mlp(c.plan, c.p, ctx) == kops::Status::kSuccess) {
    cudaStreamSynchronize(stream);
    max_abs_diff(c.out_mega.p, c.mod.out.p, size_t(c.M) * c.H, "mlp");
  }
  const int64_t bytes = int64_t(c.M) * (c.H * 3 + c.I * 2) * 2;
  const int64_t flops = int64_t(c.M) * (2 * int64_t(c.H) * c.I * 2 + int64_t(c.I) * c.H * 2);
  add_case(cases, "mega.mlp.m8.h1024.i2048", "mega_mlp", c.M, c.H, bytes, flops);
  const Case& cs = cases->back();

  cudaGraphExec_t mega_graph = nullptr, mod_graph = nullptr;
  auto mega_run = [&]() {
    return kops::mega::mega_run_mlp(c.plan, c.p, ctx) == kops::Status::kSuccess;
  };
  auto mod_run = [&]() { return c.mod.run(ctx); };
  capture(stream, mega_run, &mega_graph, "mega.mlp mega graph");
  capture(stream, mod_run, &mod_graph, "mega.mlp modular graph");
  auto mega_direct = [&]() { mega_run(); };
  auto mod_direct = [&]() { mod_run(); };
  auto mega_graph_fn = [&]() {
    if (mega_graph != nullptr) cudaGraphLaunch(mega_graph, stream);
  };
  auto mod_graph_fn = [&]() {
    if (mod_graph != nullptr) cudaGraphLaunch(mod_graph, stream);
  };

  struct Run { const char* id; std::function<void()> fn; };
  const Run runs[] = {
                      {"mega.mlp.mega", mega_direct},
                      {"mega.mlp.mega_graph", mega_graph_fn},
                      {"mega.mlp.modular", mod_direct},
                      {"mega.mlp.modular_graph", mod_graph_fn}};
  for (const Run& item : runs) {
    Stats s;
    if (measure(cs, item.fn, &s)) {
      results->push_back({&cs, s});
      std::fprintf(stderr, "%-28s %.4f ms\n", item.id, s.median_ms);
    }
  }
  if (mod_graph != nullptr) cudaGraphExecDestroy(mod_graph);
  if (mega_graph != nullptr) cudaGraphExecDestroy(mega_graph);
  if (c.ws.device_persistent != nullptr) cudaFree(c.ws.device_persistent);
  if (c.mod.ws.device_scratch != nullptr) cudaFree(c.mod.ws.device_scratch);
}

// -----------------------------------------------------------------------------
// Attention
// -----------------------------------------------------------------------------
bool setup_attn(AttnCase* c, const kops::LaunchContext& ctx) {
  const int total = (c->Hq + 2 * c->Hkv) * c->D;
  c->hidden.alloc(size_t(c->M) * c->H * 2);
  c->nw.alloc(size_t(c->H) * 2);
  c->wqkv.alloc(size_t(total) * c->H * 2);
  c->qnw.alloc(size_t(c->D) * 2);
  c->knw.alloc(size_t(c->D) * 2);
  c->wo.alloc(size_t(c->H) * c->Hq * c->D * 2);
  c->residual.alloc(size_t(c->M) * c->H * 2);
  c->qkv.alloc(size_t(c->M) * total * 2);
  c->attn.alloc(size_t(c->M) * c->Hq * c->D * 2);
  c->out_mega.alloc(size_t(c->M) * c->H * 2);
  c->out_mod.alloc(size_t(c->M) * c->H * 2);
  c->k.alloc(size_t(c->PS) * c->Hkv * c->D * 2);
  c->v.alloc(size_t(c->PS) * c->Hkv * c->D * 2);
  c->cs.alloc(size_t(c->kv_len) * c->D * 4);
  c->positions.alloc(size_t(c->M) * 8);
  c->slots.alloc(size_t(c->M) * 8);
  c->btab.alloc(4);
  c->seq.alloc(4);
  fill_bf16(c->hidden.p, size_t(c->M) * c->H, 11, -1, 1);
  fill_bf16(c->nw.p, c->H, 12, 0.8f, 1.2f);
  fill_bf16(c->wqkv.p, size_t(total) * c->H, 13, -0.08f, 0.08f);
  fill_bf16(c->qnw.p, c->D, 14, 0.8f, 1.2f);
  fill_bf16(c->knw.p, c->D, 15, 0.8f, 1.2f);
  fill_bf16(c->wo.p, size_t(c->H) * c->Hq * c->D, 16, -0.08f, 0.08f);
  fill_bf16(c->residual.p, size_t(c->M) * c->H, 17, -0.5f, 0.5f);
  fill_bf16(c->k.p, size_t(c->PS) * c->Hkv * c->D, 18, -0.5f, 0.5f);
  fill_bf16(c->v.p, size_t(c->PS) * c->Hkv * c->D, 19, -0.5f, 0.5f);
  std::vector<float> cs(size_t(c->kv_len) * c->D);
  for (int p = 0; p < c->kv_len; ++p) {
    for (int e = 0; e < c->D / 2; ++e) {
      const double inv = std::pow(10000.0, -2.0 * e / c->D);
      cs[size_t(p) * c->D + e] = float(std::cos(p * inv));
      cs[size_t(p) * c->D + c->D / 2 + e] = float(std::sin(p * inv));
    }
  }
  up(static_cast<float*>(c->cs.p), cs);
  std::vector<int64_t> pos(c->M), sl(c->M);
  const int pos0 = c->kv_len - c->M;
  for (int m = 0; m < c->M; ++m) {
    pos[m] = pos0 + m;
    sl[m] = pos0 + m;
  }
  up(static_cast<int64_t*>(c->positions.p), pos);
  up(static_cast<int64_t*>(c->slots.p), sl);
  up(static_cast<int32_t*>(c->btab.p), std::vector<int32_t>{0});
  up(static_cast<int32_t*>(c->seq.p), std::vector<int32_t>{c->kv_len});
  c->p = kops::mega::MegaAttentionBlockParams{};
  c->p.hidden = c->hidden.p;
  c->p.residual = c->residual.p;
  c->p.in_norm_weight = c->nw.p;
  c->p.norm_eps = 1e-5f;
  c->p.w_qkv = c->wqkv.p;
  c->p.q_norm_weight = c->qnw.p;
  c->p.k_norm_weight = c->knw.p;
  c->p.positions = static_cast<const int64_t*>(c->positions.p);
  c->p.cos_sin_cache = c->cs.p;
  c->p.rot_dim = c->D;
  c->p.rope_style = kops::RopeStyle::kNeoX;
  c->p.kv.k_data = c->k.p;
  c->p.kv.v_data = c->v.p;
  c->p.kv.k_dtype = c->p.kv.v_dtype = kops::DType::kBFloat16;
  c->p.kv.layout = kops::kv::KVLayout::kNHD;
  c->p.kv.page_size = c->PS;
  c->p.kv.num_kv_heads = c->Hkv;
  c->p.kv.head_dim_k = c->p.kv.head_dim_v = c->D;
  c->p.kv.k_stride_page = c->p.kv.v_stride_page = c->PS * c->Hkv * c->D;
  c->p.kv.k_stride_token = c->p.kv.v_stride_token = c->Hkv * c->D;
  c->p.kv.k_stride_head = c->p.kv.v_stride_head = c->D;
  c->p.kv.table_format = kops::kv::PageTableFormat::kBlockTable;
  c->p.kv.block_table = static_cast<const int32_t*>(c->btab.p);
  c->p.kv.block_table_stride = 1;
  c->p.kv.seq_lens = static_cast<const int32_t*>(c->seq.p);
  c->p.slot_mapping = static_cast<const int64_t*>(c->slots.p);
  c->p.mod.sm_scale = 1.0f / std::sqrt(float(c->D));
  c->p.w_o = c->wo.p;
  c->p.out = c->out_mega.p;
  c->p.m = c->M;
  c->p.hidden_size = c->H;
  c->p.num_q_heads = c->Hq;
  c->p.num_kv_heads = c->Hkv;
  c->p.head_dim = c->D;
  kops::WorkspaceRequirements req{};
  if (kops::mega::mega_workspace_requirements(kops::mega::MegaKind::kAttentionBlock, &c->p, ctx,
                                              &req) != kops::Status::kSuccess) {
    return false;
  }
  c->ws.device_persistent_bytes = req.device_persistent_bytes;
  cudaMalloc(&c->ws.device_persistent, c->ws.device_persistent_bytes);
  if (kops::mega::mega_prepare(kops::mega::MegaKind::kAttentionBlock, &c->p, c->ws, ctx, &c->plan) !=
      kops::Status::kSuccess) {
    return false;
  }
  c->mod.M = c->M;
  c->mod.H = c->H;
  c->mod.Hq = c->Hq;
  c->mod.Hkv = c->Hkv;
  c->mod.D = c->D;
  c->mod.PS = c->PS;
  c->mod.kv_len = c->kv_len;
  c->mod.normed.alloc(size_t(c->M) * c->H * 2);
  c->mod.qkv.alloc(size_t(c->M) * total * 2);
  c->mod.attn.alloc(size_t(c->M) * c->Hq * c->D * 2);
  c->mod.out.alloc(size_t(c->M) * c->H * 2);
  c->mod.kcache.alloc(size_t(c->PS) * c->Hkv * c->D * 2);
  c->mod.vcache.alloc(size_t(c->PS) * c->Hkv * c->D * 2);
  c->mod.cache_cs.alloc(size_t(c->kv_len) * c->D * 4);
  cudaMemcpy(c->mod.kcache.p, c->k.p, size_t(c->PS) * c->Hkv * c->D * 2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(c->mod.vcache.p, c->v.p, size_t(c->PS) * c->Hkv * c->D * 2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(c->mod.cache_cs.p, c->cs.p, size_t(c->kv_len) * c->D * 4, cudaMemcpyDeviceToDevice);
  c->mod.hidden = c->hidden.p;
  c->mod.nw = c->nw.p;
  c->mod.wqkv = c->wqkv.p;
  c->mod.qnw = c->qnw.p;
  c->mod.knw = c->knw.p;
  c->mod.wo = c->wo.p;
  c->mod.residual = c->residual.p;
  c->mod.positions = static_cast<int64_t*>(c->positions.p);
  c->mod.slots = static_cast<int64_t*>(c->slots.p);
  c->mod.block_table = static_cast<int32_t*>(c->btab.p);
  c->mod.seq_lens = static_cast<int32_t*>(c->seq.p);
  c->mod.kv = c->p.kv;
  c->mod.kv.k_data = c->mod.kcache.p;
  c->mod.kv.v_data = c->mod.vcache.p;
  c->mod.gws.device_scratch_bytes = size_t(64) << 20;
  cudaMalloc(&c->mod.gws.device_scratch, c->mod.gws.device_scratch_bytes);
  return c->mod.prepare(ctx);
}

void bench_attn(const kops::LaunchContext& ctx, cudaStream_t stream, std::vector<Result>* results,
                std::vector<Case>* cases) {
  AttnCase c;
  if (!setup_attn(&c, ctx)) return;
  if (c.mod.run(ctx) &&
      kops::mega::mega_run_attention_block(c.plan, c.p, ctx) == kops::Status::kSuccess) {
    cudaStreamSynchronize(stream);
    max_abs_diff(c.out_mega.p, c.mod.out.p, size_t(c.M) * c.H, "attention");
  }
  const int64_t bytes = int64_t(c.M) * (int64_t(c.H) * 6 + int64_t(c.Hq + 2 * c.Hkv) * c.D * 2 +
                                        int64_t(c.Hq) * c.D * 2);
  const int64_t flops = int64_t(c.M) * int64_t(c.H) * (c.Hq + 2 * c.Hkv) * c.D * 4;
  add_case(cases, "mega.attn.m1.h1024.hq8.hkv2.d128.kv64", "mega_attention", c.M, c.H, bytes, flops);
  const Case& cs = cases->back();
  cudaGraphExec_t mega_graph = nullptr, mod_graph = nullptr;
  auto mega_run = [&]() {
    return kops::mega::mega_run_attention_block(c.plan, c.p, ctx) == kops::Status::kSuccess;
  };
  auto mod_run = [&]() { return c.mod.run(ctx); };
  capture(stream, mega_run, &mega_graph, "mega.attn mega graph");
  capture(stream, mod_run, &mod_graph, "mega.attn modular graph");
  auto mega_direct = [&]() { mega_run(); };
  auto mod_direct = [&]() { mod_run(); };
  auto mega_graph_fn = [&]() { if (mega_graph) cudaGraphLaunch(mega_graph, stream); };
  auto mod_graph_fn = [&]() { if (mod_graph) cudaGraphLaunch(mod_graph, stream); };
  struct Run { const char* id; std::function<void()> fn; };
  const Run runs[] = {
                      {"mega.attn.mega", mega_direct},
                      {"mega.attn.mega_graph", mega_graph_fn},
                      {"mega.attn.modular", mod_direct},
                      {"mega.attn.modular_graph", mod_graph_fn}};
  for (const Run& item : runs) {
    Stats s;
    if (measure(cs, item.fn, &s)) {
      results->push_back({&cs, s});
      std::fprintf(stderr, "%-28s %.4f ms\n", item.id, s.median_ms);
    }
  }
  if (mod_graph) cudaGraphExecDestroy(mod_graph);
  if (mega_graph) cudaGraphExecDestroy(mega_graph);
  if (c.ws.device_persistent != nullptr) cudaFree(c.ws.device_persistent);
  if (c.mod.ws.device_persistent != nullptr) cudaFree(c.mod.ws.device_persistent);
  if (c.mod.ws.device_scratch != nullptr) cudaFree(c.mod.ws.device_scratch);
  if (c.mod.gws.device_scratch != nullptr) cudaFree(c.mod.gws.device_scratch);
}

// -----------------------------------------------------------------------------
// MoE
// -----------------------------------------------------------------------------
bool setup_moe(MoeCase* c, const kops::LaunchContext& ctx) {
  const size_t h_n = size_t(c->M) * c->H;
  const size_t w13_n = size_t(c->E) * 2 * c->I * c->H;
  const size_t w2_n = size_t(c->E) * c->H * c->I;
  c->hidden.alloc(h_n * 2);
  c->w13.alloc(w13_n * 2);
  c->w2.alloc(w2_n * 2);
  c->ids.alloc(size_t(c->M) * c->K * 4);
  c->weights.alloc(size_t(c->M) * c->K * 4);
  c->out_mega.alloc(h_n * 2);
  c->out_mod.alloc(h_n * 2);
  fill_bf16(c->hidden.p, h_n, 21, -1, 1);
  fill_bf16(c->w13.p, w13_n, 22, -0.1f, 0.1f);
  fill_bf16(c->w2.p, w2_n, 23, -0.1f, 0.1f);
  std::vector<int32_t> ids(size_t(c->M) * c->K);
  std::vector<float> wts(size_t(c->M) * c->K);
  for (int m = 0; m < c->M; ++m) {
    for (int k = 0; k < c->K; ++k) {
      ids[size_t(m) * c->K + k] = (m + 2 * k) % c->E;
      wts[size_t(m) * c->K + k] = m % 2 == 0 ? 0.6f : 0.4f;
    }
  }
  up(static_cast<int32_t*>(c->ids.p), ids);
  up(static_cast<float*>(c->weights.p), wts);
  c->p = kops::mega::MegaMoELocalParams{};
  c->p.hidden = c->hidden.p;
  c->p.topk_ids = static_cast<const int32_t*>(c->ids.p);
  c->p.topk_weights = static_cast<const float*>(c->weights.p);
  c->p.w13 = c->w13.p;
  c->p.w2 = c->w2.p;
  c->p.w_quant.storage_dtype = kops::DType::kBFloat16;
  c->p.act = kops::ActFn::kSiLU;
  c->p.gate_layout = kops::GateLayout::kGateUp;
  c->p.out = c->out_mega.p;
  c->p.routed_scaling_factor = 1.0f;
  c->p.m = c->M;
  c->p.hidden_size = c->H;
  c->p.intermediate_size = c->I;
  c->p.num_local_experts = c->E;
  c->p.topk = c->K;
  kops::WorkspaceRequirements req{};
  if (kops::mega::mega_workspace_requirements(kops::mega::MegaKind::kMoELocal, &c->p, ctx, &req) !=
      kops::Status::kSuccess) {
    return false;
  }
  c->ws.device_persistent_bytes = req.device_persistent_bytes;
  cudaMalloc(&c->ws.device_persistent, c->ws.device_persistent_bytes);
  if (kops::mega::mega_prepare(kops::mega::MegaKind::kMoELocal, &c->p, c->ws, ctx, &c->plan) !=
      kops::Status::kSuccess) {
    return false;
  }
  return true;
}

void bench_moe(const kops::LaunchContext& ctx, cudaStream_t stream, std::vector<Result>* results,
               std::vector<Case>* cases) {
  MoeCase c;
  if (!setup_moe(&c, ctx)) return;
  const int64_t bytes = int64_t(c.M) * c.H * 2 + int64_t(c.E) * (int64_t(c.H) * c.I * 3) * 2 +
                        int64_t(c.M) * c.K * 8;
  const int64_t flops = int64_t(c.M) * c.K * 3 * int64_t(c.H) * c.I * 2;
  add_case(cases, "mega.moe.m8.h512.i512.e8.k2", "mega_moe_local", c.M, c.H, bytes, flops);
  const Case& cs = cases->back();
  cudaGraphExec_t mega_graph = nullptr;
  auto mega_run = [&]() {
    return kops::mega::mega_run_moe_local(c.plan, c.p, ctx) == kops::Status::kSuccess;
  };
  capture(stream, mega_run, &mega_graph, "mega.moe mega graph");
  auto mega_direct = [&]() { mega_run(); };
  auto mega_graph_fn = [&]() { if (mega_graph) cudaGraphLaunch(mega_graph, stream); };
  struct Run { const char* id; std::function<void()> fn; };
  const Run runs[] = {{"mega.moe.mega", mega_direct}, {"mega.moe.mega_graph", mega_graph_fn}};
  for (const Run& item : runs) {
    Stats s;
    if (measure(cs, item.fn, &s)) {
      results->push_back({&cs, s});
      std::fprintf(stderr, "%-28s %.4f ms\n", item.id, s.median_ms);
    }
  }
  if (mega_graph) cudaGraphExecDestroy(mega_graph);
  if (c.ws.device_persistent != nullptr) cudaFree(c.ws.device_persistent);
  // The modular MoE orchestrator (moe_forward) has no body in this snapshot;
  // only the megakernel rows are reported for this case.
}

// -----------------------------------------------------------------------------
// Decoder (attention + dense MLP)
// -----------------------------------------------------------------------------
bool setup_dec(DecCase* c, const kops::LaunchContext& ctx) {
  const int total = (c->Hq + 2 * c->Hkv) * c->D;
  const size_t h_n = size_t(c->M) * c->H;
  c->hidden.alloc(h_n * 2);
  c->anw.alloc(size_t(c->H) * 2);
  c->wqkv.alloc(size_t(total) * c->H * 2);
  c->qnw.alloc(size_t(c->D) * 2);
  c->knw.alloc(size_t(c->D) * 2);
  c->wo.alloc(size_t(c->H) * c->Hq * c->D * 2);
  c->mnw.alloc(size_t(c->H) * 2);
  c->w13.alloc(size_t(2 * c->I) * c->H * 2);
  c->w2.alloc(size_t(c->H) * c->I * 2);
  c->residual.alloc(h_n * 2);
  c->qkv.alloc(size_t(c->M) * total * 2);
  c->attn.alloc(size_t(c->M) * c->Hq * c->D * 2);
  c->inter.alloc(size_t(c->M) * c->I * 2);
  c->out_mega.alloc(h_n * 2);
  c->out_mod.alloc(h_n * 2);
  c->k.alloc(size_t(c->PS) * c->Hkv * c->D * 2);
  c->v.alloc(size_t(c->PS) * c->Hkv * c->D * 2);
  c->cs.alloc(size_t(c->kv_len) * c->D * 4);
  c->positions.alloc(size_t(c->M) * 8);
  c->slots.alloc(size_t(c->M) * 8);
  c->btab.alloc(4);
  c->seq.alloc(4);
  fill_bf16(c->hidden.p, h_n, 31, -1, 1);
  fill_bf16(c->anw.p, c->H, 32, 0.8f, 1.2f);
  fill_bf16(c->wqkv.p, size_t(total) * c->H, 33, -0.08f, 0.08f);
  fill_bf16(c->qnw.p, c->D, 34, 0.8f, 1.2f);
  fill_bf16(c->knw.p, c->D, 35, 0.8f, 1.2f);
  fill_bf16(c->wo.p, size_t(c->H) * c->Hq * c->D, 36, -0.08f, 0.08f);
  fill_bf16(c->mnw.p, c->H, 37, 0.8f, 1.2f);
  fill_bf16(c->w13.p, size_t(2 * c->I) * c->H, 38, -0.08f, 0.08f);
  fill_bf16(c->w2.p, size_t(c->H) * c->I, 39, -0.08f, 0.08f);
  fill_bf16(c->residual.p, h_n, 40, -0.5f, 0.5f);
  fill_bf16(c->k.p, size_t(c->PS) * c->Hkv * c->D, 41, -0.5f, 0.5f);
  fill_bf16(c->v.p, size_t(c->PS) * c->Hkv * c->D, 42, -0.5f, 0.5f);
  std::vector<float> cs(size_t(c->kv_len) * c->D);
  for (int p = 0; p < c->kv_len; ++p) {
    for (int e = 0; e < c->D / 2; ++e) {
      const double inv = std::pow(10000.0, -2.0 * e / c->D);
      cs[size_t(p) * c->D + e] = float(std::cos(p * inv));
      cs[size_t(p) * c->D + c->D / 2 + e] = float(std::sin(p * inv));
    }
  }
  up(static_cast<float*>(c->cs.p), cs);
  std::vector<int64_t> pos(c->M), sl(c->M);
  const int pos0 = c->kv_len - c->M;
  for (int m = 0; m < c->M; ++m) pos[m] = sl[m] = pos0 + m;
  up(static_cast<int64_t*>(c->positions.p), pos);
  up(static_cast<int64_t*>(c->slots.p), sl);
  up(static_cast<int32_t*>(c->btab.p), std::vector<int32_t>{0});
  up(static_cast<int32_t*>(c->seq.p), std::vector<int32_t>{c->kv_len});
  c->attn_p = kops::mega::MegaAttentionBlockParams{};
  c->attn_p.hidden = c->hidden.p;
  c->attn_p.residual = c->residual.p;
  c->attn_p.in_norm_weight = c->anw.p;
  c->attn_p.norm_eps = 1e-5f;
  c->attn_p.w_qkv = c->wqkv.p;
  c->attn_p.q_norm_weight = c->qnw.p;
  c->attn_p.k_norm_weight = c->knw.p;
  c->attn_p.positions = static_cast<const int64_t*>(c->positions.p);
  c->attn_p.cos_sin_cache = c->cs.p;
  c->attn_p.rot_dim = c->D;
  c->attn_p.rope_style = kops::RopeStyle::kNeoX;
  c->attn_p.kv.k_data = c->k.p;
  c->attn_p.kv.v_data = c->v.p;
  c->attn_p.kv.k_dtype = c->attn_p.kv.v_dtype = kops::DType::kBFloat16;
  c->attn_p.kv.layout = kops::kv::KVLayout::kNHD;
  c->attn_p.kv.page_size = c->PS;
  c->attn_p.kv.num_kv_heads = c->Hkv;
  c->attn_p.kv.head_dim_k = c->attn_p.kv.head_dim_v = c->D;
  c->attn_p.kv.k_stride_page = c->attn_p.kv.v_stride_page = c->PS * c->Hkv * c->D;
  c->attn_p.kv.k_stride_token = c->attn_p.kv.v_stride_token = c->Hkv * c->D;
  c->attn_p.kv.k_stride_head = c->attn_p.kv.v_stride_head = c->D;
  c->attn_p.kv.table_format = kops::kv::PageTableFormat::kBlockTable;
  c->attn_p.kv.block_table = static_cast<const int32_t*>(c->btab.p);
  c->attn_p.kv.block_table_stride = 1;
  c->attn_p.kv.seq_lens = static_cast<const int32_t*>(c->seq.p);
  c->attn_p.slot_mapping = static_cast<const int64_t*>(c->slots.p);
  c->attn_p.mod.sm_scale = 1.0f / std::sqrt(float(c->D));
  c->attn_p.w_o = c->wo.p;
  c->attn_p.out = c->residual.p;
  c->attn_p.m = c->M;
  c->attn_p.hidden_size = c->H;
  c->attn_p.num_q_heads = c->Hq;
  c->attn_p.num_kv_heads = c->Hkv;
  c->attn_p.head_dim = c->D;

  c->mlp_p = kops::mega::MegaMLPParams{};
  c->mlp_p.x = c->residual.p;
  c->mlp_p.pre_norm_weight = c->mnw.p;
  c->mlp_p.pre_norm_eps = 1e-5f;
  c->mlp_p.w13 = c->w13.p;
  c->mlp_p.w2 = c->w2.p;
  c->mlp_p.w_quant.storage_dtype = kops::DType::kBFloat16;
  c->mlp_p.act = kops::ActFn::kSiLU;
  c->mlp_p.gate_layout = kops::GateLayout::kGateUp;
  c->mlp_p.intermediate = c->inter.p;
  c->mlp_p.out = c->out_mega.p;
  c->mlp_p.ldo = c->H;
  c->mlp_p.residual = c->residual.p;
  c->mlp_p.m = c->M;
  c->mlp_p.hidden_size = c->H;
  c->mlp_p.intermediate_size = c->I;
  kops::mega::MegaDecoderParams dp{};
  dp.attn = c->attn_p;
  dp.mlp = c->mlp_p;
  kops::WorkspaceRequirements req{};
  if (kops::mega::mega_workspace_requirements(kops::mega::MegaKind::kDecoder, &dp, ctx, &req) !=
      kops::Status::kSuccess) {
    return false;
  }
  c->ws.device_persistent_bytes = req.device_persistent_bytes;
  cudaMalloc(&c->ws.device_persistent, c->ws.device_persistent_bytes);
  if (kops::mega::mega_prepare(kops::mega::MegaKind::kDecoder, &dp, c->ws, ctx, &c->plan) !=
      kops::Status::kSuccess) {
    return false;
  }
  // Modular chain shares the same attention/MLP shapes.
  c->mod_attn.M = c->M;
  c->mod_attn.H = c->H;
  c->mod_attn.Hq = c->Hq;
  c->mod_attn.Hkv = c->Hkv;
  c->mod_attn.D = c->D;
  c->mod_attn.PS = c->PS;
  c->mod_attn.kv_len = c->kv_len;
  c->mod_attn.normed.alloc(h_n * 2);
  c->mod_attn.qkv.alloc(size_t(c->M) * total * 2);
  c->mod_attn.attn.alloc(size_t(c->M) * c->Hq * c->D * 2);
  c->mod_attn.out.alloc(h_n * 2);
  c->mod_attn.kcache.alloc(size_t(c->PS) * c->Hkv * c->D * 2);
  c->mod_attn.vcache.alloc(size_t(c->PS) * c->Hkv * c->D * 2);
  c->mod_attn.cache_cs.alloc(size_t(c->kv_len) * c->D * 4);
  cudaMemcpy(c->mod_attn.kcache.p, c->k.p, size_t(c->PS) * c->Hkv * c->D * 2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(c->mod_attn.vcache.p, c->v.p, size_t(c->PS) * c->Hkv * c->D * 2, cudaMemcpyDeviceToDevice);
  cudaMemcpy(c->mod_attn.cache_cs.p, c->cs.p, size_t(c->kv_len) * c->D * 4, cudaMemcpyDeviceToDevice);
  c->mod_attn.hidden = c->hidden.p;
  c->mod_attn.nw = c->anw.p;
  c->mod_attn.wqkv = c->wqkv.p;
  c->mod_attn.qnw = c->qnw.p;
  c->mod_attn.knw = c->knw.p;
  c->mod_attn.wo = c->wo.p;
  c->mod_attn.residual = c->residual.p;
  c->mod_attn.positions = static_cast<int64_t*>(c->positions.p);
  c->mod_attn.slots = static_cast<int64_t*>(c->slots.p);
  c->mod_attn.block_table = static_cast<int32_t*>(c->btab.p);
  c->mod_attn.seq_lens = static_cast<int32_t*>(c->seq.p);
  c->mod_attn.kv = c->attn_p.kv;
  c->mod_attn.kv.k_data = c->mod_attn.kcache.p;
  c->mod_attn.kv.v_data = c->mod_attn.vcache.p;
  c->mod_attn.gws.device_scratch_bytes = size_t(64) << 20;
  cudaMalloc(&c->mod_attn.gws.device_scratch, c->mod_attn.gws.device_scratch_bytes);
  if (!c->mod_attn.prepare(ctx)) return false;
  // The modular attention writes the post-attention residual into mod_attn.out.
  c->mod_mlp.M = c->M;
  c->mod_mlp.H = c->H;
  c->mod_mlp.I = c->I;
  c->mod_mlp.normed.alloc(h_n * 2);
  c->mod_mlp.gateup.alloc(size_t(c->M) * 2 * c->I * 2);
  c->mod_mlp.inter.alloc(size_t(c->M) * c->I * 2);
  c->mod_mlp.out.alloc(h_n * 2);
  c->mod_mlp.x = c->mod_attn.out.p;
  c->mod_mlp.nw = c->mnw.p;
  c->mod_mlp.w13 = c->w13.p;
  c->mod_mlp.w2 = c->w2.p;
  c->mod_mlp.residual = c->mod_attn.out.p;
  c->mod_mlp.ws.device_scratch_bytes = size_t(64) << 20;
  cudaMalloc(&c->mod_mlp.ws.device_scratch, c->mod_mlp.ws.device_scratch_bytes);
  return true;
}

void bench_decoder(const kops::LaunchContext& ctx, cudaStream_t stream,
                   std::vector<Result>* results, std::vector<Case>* cases) {
  DecCase c;
  if (!setup_dec(&c, ctx)) return;
  if (c.mod_attn.run(ctx) && c.mod_mlp.run(ctx) &&
      kops::mega::mega_run_decoder(c.plan, c.attn_p, c.mlp_p,
                                   kops::mega::MegaMoELocalParams{}, ctx) ==
          kops::Status::kSuccess) {
    cudaStreamSynchronize(stream);
    max_abs_diff(c.out_mega.p, c.mod_mlp.out.p, size_t(c.M) * c.H, "decoder");
  }
  const int64_t bytes = int64_t(c.M) * (int64_t(c.H) * 8 + int64_t(c.Hq) * c.D * 2) +
                        int64_t(c.H) * c.I * 2;
  const int64_t flops = int64_t(c.M) * int64_t(c.H) * (c.Hq + 2 * c.Hkv) * c.D * 4 +
                        int64_t(c.M) * 3 * int64_t(c.H) * c.I * 2;
  add_case(cases, "mega.decoder.m1.h512.i1024", "mega_decoder", c.M, c.H, bytes, flops);
  const Case& cs = cases->back();
  cudaGraphExec_t mega_graph = nullptr, mod_graph = nullptr;
  auto mega_run = [&]() {
    return kops::mega::mega_run_decoder(c.plan, c.attn_p, c.mlp_p,
                                        kops::mega::MegaMoELocalParams{}, ctx) ==
           kops::Status::kSuccess;
  };
  auto mod_run = [&]() { return c.mod_attn.run(ctx) && c.mod_mlp.run(ctx); };
  capture(stream, mega_run, &mega_graph, "mega.decoder mega graph");
  capture(stream, mod_run, &mod_graph, "mega.decoder modular graph");
  auto mega_direct = [&]() { mega_run(); };
  auto mod_direct = [&]() { mod_run(); };
  auto mega_graph_fn = [&]() { if (mega_graph) cudaGraphLaunch(mega_graph, stream); };
  auto mod_graph_fn = [&]() { if (mod_graph) cudaGraphLaunch(mod_graph, stream); };
  struct Run { const char* id; std::function<void()> fn; };
  const Run runs[] = {
                      {"mega.decoder.mega", mega_direct},
                      {"mega.decoder.mega_graph", mega_graph_fn},
                      {"mega.decoder.modular", mod_direct},
                      {"mega.decoder.modular_graph", mod_graph_fn}};
  for (const Run& item : runs) {
    Stats s;
    if (measure(cs, item.fn, &s)) {
      results->push_back({&cs, s});
      std::fprintf(stderr, "%-28s %.4f ms\n", item.id, s.median_ms);
    }
  }
  if (mod_graph) cudaGraphExecDestroy(mod_graph);
  if (mega_graph) cudaGraphExecDestroy(mega_graph);
  if (c.ws.device_persistent != nullptr) cudaFree(c.ws.device_persistent);
  if (c.mod_attn.ws.device_persistent != nullptr) cudaFree(c.mod_attn.ws.device_persistent);
  if (c.mod_attn.ws.device_scratch != nullptr) cudaFree(c.mod_attn.ws.device_scratch);
  if (c.mod_attn.gws.device_scratch != nullptr) cudaFree(c.mod_attn.gws.device_scratch);
  if (c.mod_mlp.ws.device_scratch != nullptr) cudaFree(c.mod_mlp.ws.device_scratch);
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = parse_options(argc, argv);
  DeviceInfo dev{};
  if (!query_device(&dev)) {
    std::fprintf(stderr, "[bench] CUDA device query failed\n");
    return 2;
  }
  if (dev.major * 10 + dev.minor < 80) {
    std::fprintf(stderr, "[bench] SM80+ required (got %d.%d)\n", dev.major, dev.minor);
    return 77;
  }
  cudaStream_t stream = nullptr;
  if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) return 2;
  kops::LaunchContext ctx{};
  if (kops::make_launch_context(0, reinterpret_cast<kops::kops_stream_t>(stream), &ctx) !=
      kops::Status::kSuccess) {
    std::fprintf(stderr, "[bench] make_launch_context failed: %s\n", kops::last_error_message());
    return 2;
  }
  std::vector<Case> cases;
  cases.reserve(64);   // Result::c points into this vector: never reallocate
  std::vector<Result> results;
  results.reserve(64);
  bench_mlp(ctx, stream, &results, &cases);
  bench_attn(ctx, stream, &results, &cases);
  bench_moe(ctx, stream, &results, &cases);
  bench_decoder(ctx, stream, &results, &cases);
  if (opts.dump_cases) {
    dump_cases(cases, dev);
  } else {
    print_table(results, dev);
    if (opts.json_path != nullptr) write_results(opts.json_path, results, dev);
  }
  cudaStreamDestroy(stream);
  return 0;
}
