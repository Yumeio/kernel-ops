// =============================================================================
// kernel-ops | benchmark/bench_activation.cu
// Activation and gated-MLP cases (F01). The case table is the shared spec read
// by benchmark/compare/torch_baseline.py through --dump-cases.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>

#include <cstdlib>
#include <vector>

#include "csrc/common/op_types.cuh"
#include "csrc/activation/activation_kernels.cuh"
#include "csrc/activation/ops.h"

namespace {

using namespace kops_bench;

std::vector<Case> make_cases() {
  std::vector<Case> cases;

  auto add_act = [&](const char* id, kops::DType dt, int64_t n, int64_t flops_per_elem) {
    const int es = dtype_size(dt);
    cases.push_back({id, "activation", dt, 1, n, 2 * n * es, flops_per_elem * n, 20, 200});
  };
  add_act("activation.silu.fp32.n4m", kops::DType::kFloat32, 4000000, 10);
  add_act("activation.silu.bf16.n4m", kops::DType::kBFloat16, 4000000, 10);

  auto add_mul = [&](const char* id, kops::DType dt, int64_t tokens, int64_t inter) {
    const int es = dtype_size(dt);
    const int64_t n = tokens * inter;
    cases.push_back(
        {id, "act_and_mul", dt, tokens, inter, 3 * n * es, 14 * n, 20, 200});
  };
  add_mul("act_and_mul.silu.fp32.t2048.i4096", kops::DType::kFloat32, 2048, 4096);
  add_mul("act_and_mul.silu.bf16.t2048.i4096", kops::DType::kBFloat16, 2048, 4096);
  cases.push_back({"act_and_mul_quant.silu.bf16.t2048.i4096", "act_and_mul_quant",
                   kops::DType::kBFloat16, 2048, 4096, 4LL * 2048 * 4096,
                   15LL * 2048 * 4096, 20, 200});

  auto add_gate = [&](const char* id, kops::DType dt, int64_t tokens, int64_t d, int64_t gate_width,
                       bool accumulate) {
    const int es = dtype_size(dt);
    const int64_t elems = tokens * d;
    const int64_t bytes = (2 * elems + (accumulate ? elems : 0) + tokens * gate_width) * es;
    cases.push_back({id, "gate_mul", dt, tokens, d, bytes, 3 * elems, 20, 200});
    (void)gate_width;
  };
  add_gate("gate_mul.sigmoid.bf16.t2048.d4096.w4096", kops::DType::kBFloat16, 2048, 4096, 4096, false);
  add_gate("gate_mul.sigmoid.bf16.t2048.d4096.w1", kops::DType::kBFloat16, 2048, 4096, 1, false);
  add_gate("gate_mul.sigmoid.bf16.t2048.d4096.w32.acc", kops::DType::kBFloat16, 2048, 4096, 32, true);
  add_gate("gate_mul.sigmoid.fp32.t2048.d4096.w4096", kops::DType::kFloat32, 2048, 4096, 4096, false);

  return cases;
}

template <typename T>
bool run_activation(const Case& c, kops::ActFn fn, const kops::LaunchContext& ctx,
                    Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  T* out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), n * sizeof(T)), "cudaMalloc in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(T)), "cudaMalloc out")) {
    return false;
  }
  cudaMemset(in, 0, n * sizeof(T));
  cudaMemset(out, 0, n * sizeof(T));

  const kops::Tensor ti = tensor2d(in, c.dtype, c.rows, c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  const kops::GluParams params{};

  bool ok = status_ok(kops::act::activation(to, ti, fn, params, ctx), c.id.c_str());
  if (ok) {
    ok = measure(c, [&] { (void)kops::act::activation(to, ti, fn, params, ctx); }, st);
  }

  cudaFree(in);
  cudaFree(out);
  return ok;
}

template <typename T>
bool run_act_and_mul(const Case& c, kops::ActFn fn, const kops::LaunchContext& ctx,
                     Stats* st) {
  const size_t in_n = static_cast<size_t>(c.rows * 2 * c.cols);
  const size_t out_n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  T* out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), in_n * sizeof(T)), "cudaMalloc in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), out_n * sizeof(T)),
               "cudaMalloc out")) {
    return false;
  }
  cudaMemset(in, 0, in_n * sizeof(T));
  cudaMemset(out, 0, out_n * sizeof(T));

  const kops::Tensor ti = tensor2d(in, c.dtype, c.rows, 2 * c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  const kops::GluParams params{};

  bool ok = status_ok(kops::act::act_and_mul(to, ti, kops::Tensor{}, fn,
                                             kops::GateLayout::kGateUp, params, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] {
          (void)kops::act::act_and_mul(to, ti, kops::Tensor{}, fn,
                                       kops::GateLayout::kGateUp, params, ctx);
        },
        st);
  }

  cudaFree(in);
  cudaFree(out);
  return ok;
}

template <typename T>
bool run_gate_mul(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const bool accumulate = c.id.find(".acc") != std::string::npos;
  const size_t p = c.id.find(".w");
  int64_t gate_width = 0;
  if (p != std::string::npos) {
    const char* s = c.id.c_str() + p + 2;
    gate_width = std::strtoll(s, nullptr, 10);
  }
  if (gate_width <= 0) return false;
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  const size_t gn = static_cast<size_t>(c.rows * gate_width);
  T* x = nullptr;
  T* g = nullptr;
  T* a = nullptr;
  T* out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&x), n * sizeof(T)), "cudaMalloc x") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&g), gn * sizeof(T)), "cudaMalloc gate") ||
      (accumulate && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&a), n * sizeof(T)),
                              "cudaMalloc acc")) ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(T)), "cudaMalloc out")) {
    return false;
  }
  cudaMemset(x, 0, n * sizeof(T));
  cudaMemset(g, 0, gn * sizeof(T));
  if (accumulate) cudaMemset(a, 0, n * sizeof(T));
  cudaMemset(out, 0, n * sizeof(T));
  const kops::Tensor tx = tensor2d(x, c.dtype, c.rows, c.cols);
  const kops::Tensor tg = tensor2d(g, c.dtype, c.rows, gate_width);
  const kops::Tensor ta = accumulate ? tensor2d(a, c.dtype, c.rows, c.cols) : kops::Tensor{};
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  bool ok = status_ok(kops::act::gate_mul(to, tx, tg, ta, kops::ActFn::kSigmoid, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(c, [&] {
      (void)kops::act::gate_mul(to, tx, tg, ta, kops::ActFn::kSigmoid, ctx);
    }, st);
  }
  cudaFree(x); cudaFree(g); if (accumulate) cudaFree(a); cudaFree(out);
  return ok;
}

template <typename T>
bool run_act_and_mul_quant(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  uint8_t* q = nullptr;
  float* scales = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), 2 * n * sizeof(T)), "in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), n), "q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), c.rows * sizeof(float)), "scales")) {
    return false;
  }
  cudaMemset(in, 0, 2 * n * sizeof(T));
  kops::act::ActMulQuantParams p{};
  p.input = in;
  p.input_stride = 2 * c.cols;
  p.num_rows = static_cast<int32_t>(c.rows);
  p.d = static_cast<int32_t>(c.cols);
  p.quant.q = q;
  p.quant.q_row_stride = c.cols;
  p.quant.scales = scales;
  p.quant.scale_row_stride = 1;
  p.quant.q_dtype = kops::DType::kFloat8E4M3;
  p.quant.mode = kops::ActQuantMode::kDynamicPerToken;
  bool ok = status_ok(kops::act::act_and_mul_quant(p, kops::ActFn::kSiLU,
                                                   kops::GateLayout::kGateUp,
                                                   kops::DType::kBFloat16, ctx),
                      c.id.c_str());
  if (ok) ok = measure(c, [&] {
    (void)kops::act::act_and_mul_quant(p, kops::ActFn::kSiLU, kops::GateLayout::kGateUp,
                                       kops::DType::kBFloat16, ctx);
  }, st);
  cudaFree(in); cudaFree(q); cudaFree(scales);
  return ok;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "act_and_mul_quant") return run_act_and_mul_quant<__nv_bfloat16>(c, ctx, st);
  const bool is_mul = c.op == "act_and_mul";
  const kops::ActFn fn = kops::ActFn::kSiLU;

  if (c.dtype == kops::DType::kFloat32) {
    if (c.op == "gate_mul") return run_gate_mul<float>(c, ctx, st);
    return is_mul ? run_act_and_mul<float>(c, fn, ctx, st)
                  : run_activation<float>(c, fn, ctx, st);
  }
  if (c.dtype == kops::DType::kBFloat16) {
    if (c.op == "gate_mul") return run_gate_mul<__nv_bfloat16>(c, ctx, st);
    return is_mul ? run_act_and_mul<__nv_bfloat16>(c, fn, ctx, st)
                  : run_activation<__nv_bfloat16>(c, fn, ctx, st);
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

  const std::vector<Case> cases = make_cases();
  if (opt.dump_cases) {
    dump_cases(cases, dev);
    return 0;
  }

  kops::LaunchContext ctx{};
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) {
    return 1;
  }

  std::vector<Result> results;
  results.reserve(cases.size());
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
