// =============================================================================
// kernel-ops | benchmark/bench_model_ops.cu
// B0 benchmark cases: residual_add and cast.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <vector>

#include "csrc/common/quant.cuh"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/ops.h"

namespace {

using namespace kops_bench;

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto add_res = [&](const char* id, kops::DType dt, int64_t rows, int64_t d, bool three) {
    const int es = dtype_size(dt);
    const int64_t reads = (three ? 3 : 2) * rows * d;
    const int64_t bytes = (reads + rows * d) * es;
    cases.push_back({id, "residual_add", dt, rows, d, bytes, (three ? 3 : 2) * rows * d, 20, 200});
  };
  add_res("model_ops.residual_add.fp32.2.t1024.h4096", kops::DType::kFloat32, 1024, 4096, false);
  add_res("model_ops.residual_add.fp32.3.t1024.h4096", kops::DType::kFloat32, 1024, 4096, true);
  add_res("model_ops.residual_add.bf16.2.t1024.h4096", kops::DType::kBFloat16, 1024, 4096, false);
  add_res("model_ops.residual_add.bf16.3.t1024.h4096", kops::DType::kBFloat16, 1024, 4096, true);
  // Above L2 (64 MB per operand): the DRAM-roofline data point, where the
  // benchmark's 256 MB L2 flush no longer dominates the sample.
  add_res("model_ops.residual_add.bf16.2.t4096.h8192", kops::DType::kBFloat16, 4096, 8192, false);
  add_res("model_ops.residual_add.fp32.2.t4096.h8192", kops::DType::kFloat32, 4096, 8192, false);

  auto add_cast = [&](const char* id, kops::DType in, kops::DType out, int64_t rows, int64_t d) {
    const int es = dtype_size(in) + dtype_size(out);
    cases.push_back({id, "cast", in, rows, d, rows * d * es, 0, 20, 200});
  };
  add_cast("model_ops.cast.bf16_to_fp32.t1024.h4096", kops::DType::kBFloat16,
           kops::DType::kFloat32, 1024, 4096);
  add_cast("model_ops.cast.fp32_to_bf16.t1024.h4096", kops::DType::kFloat32,
           kops::DType::kBFloat16, 1024, 4096);
  add_cast("model_ops.cast.fp16_to_fp32.t1024.h4096", kops::DType::kFloat16,
           kops::DType::kFloat32, 1024, 4096);
  add_cast("model_ops.cast.bf16_to_fp32.t4096.h8192", kops::DType::kBFloat16,
           kops::DType::kFloat32, 4096, 8192);

  // AttnRes (Kimi K3 shape). Roofline bytes: every source byte is read from
  // DRAM once (the stats pass and the mix pass touch one row's sources
  // back-to-back, so the second read hits L2), plus the fused partial += delta
  // update, the raw output, and (with the fused output norm) the normed row and
  // the MXFP8 payload with its UE8M0 scales.
  auto add_attn_res = [&](const char* id, int64_t tokens, int64_t d, int sources, bool norm) {
    constexpr int es = 2;   // bf16
    const int depth = sources + 1;  // the partial is always a source here
    int64_t bytes = depth * tokens * d * es;       // sources
    bytes += 2 * tokens * d * es;                  // delta read + partial write
    bytes += tokens * d * es;                      // raw output
    if (norm) bytes += tokens * d * es + tokens * d + tokens * d / 8;
    const int64_t flops = 4 * depth * tokens * d;  // sq, dot, mix, one rounding
    cases.push_back({id, "attn_res", kops::DType::kBFloat16, tokens, d, bytes, flops, 20, 200});
  };
  for (const int64_t tokens : {int64_t{1}, int64_t{16}, int64_t{256}}) {
    for (const int sources : {1, 4, 8}) {
      char id[128];
      std::snprintf(id, sizeof(id), "model_ops.attn_res.bf16.t%lld.h7168.s%d",
                    static_cast<long long>(tokens), sources);
      add_attn_res(id, tokens, 7168, sources, false);
      std::snprintf(id, sizeof(id), "model_ops.attn_res.bf16.t%lld.h7168.s%d.norm",
                    static_cast<long long>(tokens), sources);
      add_attn_res(id, tokens, 7168, sources, true);
    }
  }
  return cases;
}

template <typename T>
bool run_residual(const Case& c, bool three, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T *a = nullptr, *b = nullptr, *cc = nullptr, *out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&a), n * sizeof(T)), "cudaMalloc a") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b), n * sizeof(T)), "cudaMalloc b") ||
      (three && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cc), n * sizeof(T)), "cudaMalloc c")) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(T)), "cudaMalloc out")) {
    return false;
  }
  cudaMemset(a, 0, n * sizeof(T));
  cudaMemset(b, 0, n * sizeof(T));
  if (three) cudaMemset(cc, 0, n * sizeof(T));
  cudaMemset(out, 0, n * sizeof(T));
  const kops::Tensor ta = tensor2d(a, c.dtype, c.rows, c.cols);
  const kops::Tensor tb = tensor2d(b, c.dtype, c.rows, c.cols);
  const kops::Tensor tc = three ? tensor2d(cc, c.dtype, c.rows, c.cols) : kops::Tensor{};
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  bool ok = status_ok(kops::model_ops::residual_add(to, ta, 1.0f, tb, 1.0f, tc, 1.0f, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(c, [&] {
      (void)kops::model_ops::residual_add(to, ta, 1.0f, tb, 1.0f, tc, 1.0f, ctx);
    }, st);
  }
  cudaFree(a); cudaFree(b); if (three) cudaFree(cc); cudaFree(out);
  return ok;
}

template <typename In, typename Out>
bool run_cast(const Case& c, kops::DType in_dt, kops::DType out_dt, const kops::LaunchContext& ctx,
              Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  In* in = nullptr;
  Out* out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), n * sizeof(In)), "cudaMalloc in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(Out)), "cudaMalloc out")) {
    return false;
  }
  cudaMemset(in, 0, n * sizeof(In));
  cudaMemset(out, 0, n * sizeof(Out));
  const kops::Tensor ti = tensor2d(in, in_dt, c.rows, c.cols);
  const kops::Tensor to = tensor2d(out, out_dt, c.rows, c.cols);
  bool ok = status_ok(kops::model_ops::cast(to, ti, 1.0f, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::model_ops::cast(to, ti, 1.0f, ctx); }, st);
  cudaFree(in); cudaFree(out);
  return ok;
}

/// Digits following `.<key>` in a case id, or `fallback`.
int id_int(const Case& c, const char* key, int fallback) {
  const std::string marker = std::string(".") + key;
  size_t i = c.id.find(marker);
  if (i == std::string::npos) return fallback;
  i += marker.size();
  int value = 0;
  bool any = false;
  while (i < c.id.size() && c.id[i] >= '0' && c.id[i] <= '9') {
    value = value * 10 + (c.id[i] - '0');
    ++i;
    any = true;
  }
  return any ? value : fallback;
}

/// AttnRes benchmark: `sources` completed block residuals plus the running
/// partial (always a source, as in K3), fused delta update, cached fp32 inverse
/// RMS, and the optional fused output norm / MXFP8 epilogue.
template <typename T>
bool run_attn_res(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int sources = id_int(c, "s", 1);
  const bool norm = c.id.find(".norm") != std::string::npos;
  const int64_t tokens = c.rows, d = c.cols;
  if (sources < 1 || tokens < 1 || d < 1) return false;
  T *src = nullptr, *partial = nullptr, *delta = nullptr, *out = nullptr, *normed = nullptr;
  T *query = nullptr, *knw = nullptr, *onw = nullptr;
  float* rinv = nullptr;
  const size_t row = static_cast<size_t>(d);
  const size_t src_n = static_cast<size_t>(sources) * tokens * d;
  auto alloc = [](void** p, size_t bytes) {
    return cuda_ok(cudaMalloc(p, bytes), "cudaMalloc attn_res");
  };
  bool alloc_ok = alloc(reinterpret_cast<void**>(&src), src_n * sizeof(T)) &&
                  alloc(reinterpret_cast<void**>(&partial), tokens * row * sizeof(T)) &&
                  alloc(reinterpret_cast<void**>(&delta), tokens * row * sizeof(T)) &&
                  alloc(reinterpret_cast<void**>(&out), tokens * row * sizeof(T)) &&
                  alloc(reinterpret_cast<void**>(&query), row * sizeof(T)) &&
                  alloc(reinterpret_cast<void**>(&knw), row * sizeof(T)) &&
                  alloc(reinterpret_cast<void**>(&rinv), sources * tokens * sizeof(float));
  if (alloc_ok && norm) {
    alloc_ok = alloc(reinterpret_cast<void**>(&normed), tokens * row * sizeof(T)) &&
               alloc(reinterpret_cast<void**>(&onw), row * sizeof(T));
  }
  if (!alloc_ok) {
    std::fprintf(stderr, "[bench] %s: allocation failed\n", c.id.c_str());
    return false;
  }
  cudaMemset(src, 0, src_n * sizeof(T));
  cudaMemset(partial, 0, tokens * row * sizeof(T));
  cudaMemset(delta, 0, tokens * row * sizeof(T));
  cudaMemset(out, 0, tokens * row * sizeof(T));
  cudaMemset(query, 0, row * sizeof(T));
  cudaMemset(knw, 0, row * sizeof(T));
  cudaMemset(rinv, 0, sources * tokens * sizeof(float));
  if (norm) {
    cudaMemset(normed, 0, tokens * row * sizeof(T));
    cudaMemset(onw, 0, row * sizeof(T));
  }
  kops::model_ops::AttnResParams p{};
  p.sources = src;
  p.source_stride = tokens * d;
  p.source_token_stride = d;
  p.num_sources = sources;
  p.source_rinv = rinv;
  p.partial = partial;
  p.partial_stride = d;
  p.delta = delta;
  p.delta_stride = d;
  p.include_partial = true;
  p.query = query;
  p.key_norm_weight = knw;
  p.key_eps = 1e-5f;
  p.out = out;
  p.out_stride = d;
  if (norm) {
    p.normed_out = normed;
    p.out_norm_weight = onw;
    p.out_norm_eps = 1e-5f;
  }
  p.num_tokens = static_cast<int32_t>(tokens);
  p.d = static_cast<int32_t>(d);
  bool ok = status_ok(kops::model_ops::attn_res(p, c.dtype, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::model_ops::attn_res(p, c.dtype, ctx); }, st);
  cudaFree(src);
  cudaFree(partial);
  cudaFree(delta);
  cudaFree(out);
  if (norm) {
    cudaFree(normed);
    cudaFree(onw);
  }
  cudaFree(query);
  cudaFree(knw);
  cudaFree(rinv);
  return ok;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const bool three = c.id.find(".3.") != std::string::npos;
  if (c.op == "residual_add") {
    if (c.dtype == kops::DType::kFloat32) return run_residual<float>(c, three, ctx, st);
    if (c.dtype == kops::DType::kBFloat16) return run_residual<__nv_bfloat16>(c, three, ctx, st);
    return false;
  }
  if (c.op == "cast") {
    if (c.id.find("bf16_to_fp32") != std::string::npos)
      return run_cast<__nv_bfloat16, float>(c, c.dtype, kops::DType::kFloat32, ctx, st);
    if (c.id.find("fp32_to_bf16") != std::string::npos)
      return run_cast<float, __nv_bfloat16>(c, c.dtype, kops::DType::kBFloat16, ctx, st);
    if (c.id.find("fp16_to_fp32") != std::string::npos)
      return run_cast<__half, float>(c, c.dtype, kops::DType::kFloat32, ctx, st);
  }
  if (c.op == "attn_res") {
    if (c.dtype == kops::DType::kBFloat16) return run_attn_res<__nv_bfloat16>(c, ctx, st);
    if (c.dtype == kops::DType::kFloat32) return run_attn_res<float>(c, ctx, st);
    if (c.dtype == kops::DType::kFloat16) return run_attn_res<__half>(c, ctx, st);
    return false;
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
