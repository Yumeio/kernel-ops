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
