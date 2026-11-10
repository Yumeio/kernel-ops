// =============================================================================
// kernel-ops | benchmark/bench_hyper_connection.cu
// B15 mHC (X01): the unfused chain (hc_post + hc_coefficients + hc_pre) versus
// the single-pass mega_mhc fused body, at the DeepSeek-V4 shape
// n_hc = 4, H = 7168 and T = 1 / 64 / 1024.
//
// Every case is one "episode": form X_l from the previous sublayer output,
// predict (pre, comb, post) from X_l, and emit the next layer input. The chain
// reads and writes the widened residual stream between kernels; mega_mhc keeps
// the sequence inside one launch, so the GB/s columns include the same logical
// traffic and the intermediate I/O is visible as the chain's extra traffic.
//
// Both paths are measured with the projection scratch pair (raw_logits +
// row_sumsq), i.e. the m16 tf32 tensor-core projection body for T >= 16: one CTA
// per 16 tokens reuses the [24, nH] predictor across 16 rows instead of one CTA
// per token re-reading it. T = 1 keeps the generic per-token body (no tile).
// The case table is the shared spec read by benchmark/compare/*.py.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <vector>

#include "csrc/common/quant.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/ops.h"

namespace {

using namespace kops_bench;
using kops::ActQuantOut;
using kops::model_ops::HCParams;
using kops::Tensor;

constexpr int kRoutes = 4;
constexpr int kOuts = kRoutes * (kRoutes + 2);

int64_t mhc_bytes(int64_t tokens, int64_t hidden, bool fused, int64_t* flops) {
  const int64_t nres = tokens * kRoutes * hidden * 2;  // one widened stream
  const int64_t stream = tokens * hidden * 2;          // x / layer input
  const int64_t fn = static_cast<int64_t>(kOuts) * kRoutes * hidden * 4;
  const int64_t coeff = tokens * (kRoutes + kRoutes * kRoutes + kRoutes + 1) * 4;
  const int64_t logits = tokens * kOuts * 4;  // projection output / constraint input
  *flops = 2 * tokens * static_cast<int64_t>(kOuts) * kRoutes * hidden;
  // Logical traffic is identical on both paths: post reads/writes the widened
  // stream, the projection reads it again plus the [N, n*H] predictor, and pre
  // reads it once more with the norm weight. `fused` is kept in the signature so
  // a future split-K body with a different intermediate footprint can diverge.
  (void)fused;
  return nres + stream + nres + nres + fn + coeff + logits + nres + hidden * 2 + stream;
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  const int64_t hidden = 7168;
  for (int64_t tokens : {1, 64, 1024}) {
    const int iters = tokens >= 1024 ? 20 : 50;
    for (bool fused : {false, true}) {
      char id[96];
      std::snprintf(id, sizeof(id), "model_ops.%s.bf16.t%lld.h7168.n4",
                    fused ? "mega_mhc" : "hc_chain", static_cast<long long>(tokens));
      int64_t flops = 0;
      const int64_t bytes = mhc_bytes(tokens, hidden, fused, &flops);
      cases.push_back({id, fused ? "mega_mhc" : "hc_chain", kops::DType::kBFloat16, tokens, hidden,
                       bytes, flops, 10, iters});
    }
  }
  return cases;
}

struct Buffers {
  __nv_bfloat16* residual = nullptr;
  __nv_bfloat16* new_residual = nullptr;
  __nv_bfloat16* x = nullptr;
  __nv_bfloat16* layer_in = nullptr;
  float* post = nullptr;
  float* comb = nullptr;
  float* fn = nullptr;
  float* mix_scales = nullptr;
  float* mix_bases = nullptr;
  __nv_bfloat16* rmsnorm_weight = nullptr;
  float* coeff_a = nullptr;
  float* coeff_b = nullptr;
  float* coeff_c = nullptr;
  float* raw_logits = nullptr;
  float* row_sumsq = nullptr;

  ~Buffers() {
    cudaFree(residual);
    cudaFree(new_residual);
    cudaFree(x);
    cudaFree(layer_in);
    cudaFree(post);
    cudaFree(comb);
    cudaFree(fn);
    cudaFree(mix_scales);
    cudaFree(mix_bases);
    cudaFree(rmsnorm_weight);
    cudaFree(coeff_a);
    cudaFree(coeff_b);
    cudaFree(coeff_c);
    cudaFree(raw_logits);
    cudaFree(row_sumsq);
  }
};

bool alloc_bf16(__nv_bfloat16** p, int64_t count) {
  return cuda_ok(cudaMalloc(reinterpret_cast<void**>(p), static_cast<size_t>(count) * 2), "bf16");
}
bool alloc_f32(float** p, int64_t count) {
  return cuda_ok(cudaMalloc(reinterpret_cast<void**>(p), static_cast<size_t>(count) * 4), "f32");
}

bool fill(Buffers* b, int64_t tokens, int64_t hidden) {
  const int64_t nres = tokens * kRoutes * hidden;
  if (!alloc_bf16(&b->residual, nres) || !alloc_bf16(&b->new_residual, nres) ||
      !alloc_bf16(&b->x, tokens * hidden) || !alloc_bf16(&b->layer_in, tokens * hidden) ||
      !alloc_f32(&b->post, tokens * kRoutes) || !alloc_f32(&b->comb, tokens * kRoutes * kRoutes) ||
      !alloc_f32(&b->fn, static_cast<int64_t>(kOuts) * kRoutes * hidden) ||
      !alloc_f32(&b->mix_scales, 3) || !alloc_f32(&b->mix_bases, kOuts) ||
      !alloc_bf16(&b->rmsnorm_weight, hidden) || !alloc_f32(&b->coeff_a, tokens * kRoutes) ||
      !alloc_f32(&b->coeff_b, tokens * kRoutes * kRoutes) ||
      !alloc_f32(&b->coeff_c, tokens * kRoutes) || !alloc_f32(&b->row_sumsq, tokens) ||
      !alloc_f32(&b->raw_logits, tokens * kOuts)) {
    return false;
  }
  // Mild non-zero patterns: zeros would let DRAM compression flatter the GB/s.
  cudaMemset(b->residual, 0x3c, static_cast<size_t>(nres) * 2);
  cudaMemset(b->new_residual, 0x3c, static_cast<size_t>(nres) * 2);
  cudaMemset(b->x, 0x3c, static_cast<size_t>(tokens * hidden) * 2);
  cudaMemset(b->layer_in, 0x3c, static_cast<size_t>(tokens * hidden) * 2);
  cudaMemset(b->post, 0x3c, static_cast<size_t>(tokens * kRoutes) * 4);
  cudaMemset(b->comb, 0x3c, static_cast<size_t>(tokens * kRoutes * kRoutes) * 4);
  cudaMemset(b->fn, 0x3c, static_cast<size_t>(kOuts) * kRoutes * hidden * 4);
  cudaMemset(b->mix_scales, 0x3c, 3 * sizeof(float));
  cudaMemset(b->mix_bases, 0x3c, static_cast<size_t>(kOuts) * 4);
  cudaMemset(b->rmsnorm_weight, 0x3c, static_cast<size_t>(hidden) * 2);
  return cudaGetLastError() == cudaSuccess;
}

HCParams make_params(const Buffers& b, int64_t tokens, int64_t hidden) {
  HCParams p{};
  p.x = b.x;
  p.x_stride = hidden;
  p.x_dtype = kops::DType::kBFloat16;
  p.residual = b.residual;
  p.residual_token_stride = static_cast<int64_t>(kRoutes) * hidden;
  p.residual_route_stride = hidden;
  p.residual_dtype = kops::DType::kBFloat16;
  p.new_residual = b.new_residual;
  p.new_residual_token_stride = static_cast<int64_t>(kRoutes) * hidden;
  p.new_residual_route_stride = hidden;
  p.post_mix = b.post;
  p.comb_res_mix = b.comb;
  p.fn = b.fn;
  p.mix_scales = b.mix_scales;
  p.mix_bases = b.mix_bases;
  p.hc_norm_eps = 2e-5f;
  p.hc_pre_eps = 3e-4f;
  p.hc_post_scale = 1.25f;
  p.sinkhorn_eps = 2e-6f;
  p.num_sinkhorn_iters = 20;
  p.constraint = kops::model_ops::HCConstraint::kManifold;
  p.rmsnorm_weight = b.rmsnorm_weight;
  p.rmsnorm_dtype = kops::DType::kBFloat16;
  p.rmsnorm_eps = 7e-6f;
  p.rmsnorm_scale = 1.25f;
  p.coeff_a = b.coeff_a;
  p.coeff_b = b.coeff_b;
  p.coeff_c = b.coeff_c;
  // raw_logits + row_sumsq are the projection scratch pair: hc_coefficients then projects
  // 16 tokens per CTA on tf32 tensor cores (T >= 16) and stores the raw dots here.
  p.raw_logits = b.raw_logits;
  p.row_sumsq = b.row_sumsq;
  p.num_tokens = static_cast<int32_t>(tokens);
  p.n_hc = kRoutes;
  p.d = static_cast<int32_t>(hidden);
  return p;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t tokens = c.rows;
  const int64_t hidden = c.cols;
  Buffers b;
  if (!fill(&b, tokens, hidden)) return false;
  const HCParams p = make_params(b, tokens, hidden);
  const Tensor residual = tensor3d(b.residual, kops::DType::kBFloat16, tokens, kRoutes, hidden);
  const Tensor x = tensor2d(b.x, kops::DType::kBFloat16, tokens, hidden);
  const Tensor layer_in = tensor2d(b.layer_in, kops::DType::kBFloat16, tokens, hidden);

  if (c.op == "hc_chain") {
    const bool ok =
        status_ok(kops::model_ops::hc_post(p, x, ctx), "hc_post") &&
        status_ok(kops::model_ops::hc_coefficients(p, ctx), "hc_coefficients") &&
        status_ok(kops::model_ops::hc_pre(p, layer_in, ActQuantOut{}, Tensor{}, 0.0f, ctx),
                  "hc_pre");
    if (!ok) return false;
    return measure(c, [&] {
      (void)kops::model_ops::hc_post(p, x, ctx);
      (void)kops::model_ops::hc_coefficients(p, ctx);
      (void)kops::model_ops::hc_pre(p, layer_in, ActQuantOut{}, Tensor{}, 0.0f, ctx);
    }, st);
  }

  if (c.op == "mega_mhc") {
    const Tensor prev_b = tensor3d(b.comb, kops::DType::kFloat32, tokens, kRoutes, kRoutes);
    const Tensor prev_c = tensor3d(b.post, kops::DType::kFloat32, tokens, kRoutes, 1);
    // mega_mhc runs the projection itself and rejects a raw_logits buffer.
    HCParams mp = p;
    mp.raw_logits = nullptr;
    const bool ok = status_ok(
        kops::model_ops::mega_mhc(mp, prev_b, prev_c, x, layer_in, ActQuantOut{}, Tensor{}, 0.0f,
                                  ctx),
        "mega_mhc");
    if (!ok) return false;
    return measure(c, [&] {
      (void)kops::model_ops::mega_mhc(mp, prev_b, prev_c, x, layer_in, ActQuantOut{}, Tensor{},
                                      0.0f, ctx);
    }, st);
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace kops_bench;
  const Options opt = parse_options(argc, argv);
  if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice")) return 1;
  DeviceInfo dev{};
  query_device(&dev);
  std::vector<Case> cases = make_cases();
  filter_cases(&cases, opt);
  if (opt.dump_cases) {
    dump_cases(cases, dev);
    return 0;
  }
  kops::LaunchContext ctx{};
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) return 1;
  std::vector<Result> results;
  for (const Case& c : cases) {
    Stats st{};
    if (!run_case(c, ctx, &st)) {
      std::fprintf(stderr, "[bench] %s failed\n", c.id.c_str());
      return 1;
    }
    results.push_back({&c, st});
  }
  print_table(results, dev);
  if (opt.json_path != nullptr) write_results(opt.json_path, results, dev);
  return 0;
}
