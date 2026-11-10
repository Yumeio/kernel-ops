// =============================================================================
// kernel-ops | benchmark/bench_norm.cu
// RMSNorm / fused-add RMSNorm cases. The case table is the shared spec read by
// benchmark/compare/torch_baseline.py through --dump-cases.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>

#include <vector>

#include "csrc/common/op_types.cuh"
#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"

namespace {

using namespace kops_bench;

std::vector<Case> make_cases() {
  std::vector<Case> cases;

  auto add_rms = [&](const char* id, kops::DType dt, int64_t rows, int64_t hidden) {
    const int es = dtype_size(dt);
    cases.push_back({id, "rms_norm", dt, rows, hidden,
                     (2 * rows * hidden + hidden) * es, 5 * rows * hidden, 20, 200});
  };
  add_rms("norm.rms_norm.fp32.h1024.b1024", kops::DType::kFloat32, 1024, 1024);
  add_rms("norm.rms_norm.fp32.h4096.b1024", kops::DType::kFloat32, 1024, 4096);
  add_rms("norm.rms_norm.bf16.h1024.b1024", kops::DType::kBFloat16, 1024, 1024);
  add_rms("norm.rms_norm.bf16.h4096.b1024", kops::DType::kBFloat16, 1024, 4096);
  // Above L2 (64 MB in + 64 MB out): the DRAM-roofline data point.
  add_rms("norm.rms_norm.bf16.h8192.b4096", kops::DType::kBFloat16, 4096, 8192);
  add_rms("norm.rms_norm.fp32.h8192.b4096", kops::DType::kFloat32, 4096, 8192);

  auto add_fused = [&](const char* id, kops::DType dt, int64_t rows, int64_t hidden) {
    const int es = dtype_size(dt);
    cases.push_back({id, "fused_add_rms_norm", dt, rows, hidden,
                     (4 * rows * hidden + hidden) * es, 7 * rows * hidden, 20, 200});
  };
  add_fused("norm.fused_add_rms_norm.fp32.h4096.b1024", kops::DType::kFloat32, 1024, 4096);
  add_fused("norm.fused_add_rms_norm.bf16.h4096.b1024", kops::DType::kBFloat16, 1024, 4096);

  auto add_plain = [&](const char* id, const char* op, kops::DType dt, int64_t rows,
                       int64_t cols, int64_t reads, int64_t writes, int64_t flops,
                       int64_t weight_elems = 0) {
    const int es = dtype_size(dt);
    cases.push_back({id, op, dt, rows, cols,
                     ((reads + writes) * rows * cols + weight_elems) * es, flops, 20, 200});
  };
  add_plain("norm.layer_norm.bf16.h4096.b1024", "layer_norm", kops::DType::kBFloat16, 1024, 4096,
            1, 1, 6 * 1024 * 4096, 4096);
  add_plain("norm.layer_norm.fp32.h4096.b1024", "layer_norm", kops::DType::kFloat32, 1024, 4096,
            1, 1, 6 * 1024 * 4096, 4096);
  add_plain("norm.fused_add_layer_norm.bf16.h4096.b1024", "fused_add_layer_norm",
            kops::DType::kBFloat16, 1024, 4096, 2, 2, 8 * 1024 * 4096, 4096);
  add_plain("norm.fused_add_rms_norm_out.bf16.h4096.b1024", "fused_add_rms_norm_out",
            kops::DType::kBFloat16, 1024, 4096, 2, 2, 7 * 1024 * 4096, 4096);
  add_plain("norm.l2_norm.bf16.h4096.b1024", "l2_norm", kops::DType::kBFloat16, 1024, 4096, 1, 1,
            3 * 1024 * 4096);
  add_plain("norm.gated_rms_norm.bf16.g128.h4096.b1024", "gated_rms_norm",
            kops::DType::kBFloat16, 1024, 4096, 2, 1, 6 * 1024 * 4096, 4096);
  add_plain("norm.qk_norm.bf16.h32kv8.d128.b1024", "qk_norm", kops::DType::kBFloat16, 1024,
            6144, 2, 1, 4 * 1024 * 40 * 128);
  add_plain("norm.norm_quant.bf16.h4096.b1024", "norm_quant", kops::DType::kBFloat16, 1024, 4096,
            2, 1, 6 * 1024 * 4096, 4096);

  return cases;
}

template <typename T>
bool run_rms(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  T* out = nullptr;
  T* weight = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), n * sizeof(T)), "cudaMalloc in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(T)), "cudaMalloc out") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&weight), c.cols * sizeof(T)),
               "cudaMalloc weight")) {
    return false;
  }
  cudaMemset(in, 0, n * sizeof(T));
  cudaMemset(out, 0, n * sizeof(T));
  cudaMemset(weight, 0, static_cast<size_t>(c.cols) * sizeof(T));

  const kops::Tensor ti = tensor2d(in, c.dtype, c.rows, c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  const kops::Tensor tw = tensor1d(weight, c.dtype, c.cols);

  bool ok = status_ok(kops::norm::rms_norm(to, ti, tw, 1e-5f, 0.0f, ctx), c.id.c_str());
  if (ok) {
    ok = measure(c, [&] { (void)kops::norm::rms_norm(to, ti, tw, 1e-5f, 0.0f, ctx); }, st);
  }

  cudaFree(in);
  cudaFree(out);
  cudaFree(weight);
  return ok;
}

template <typename T>
bool run_fused(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* x = nullptr;
  T* res = nullptr;
  T* weight = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&x), n * sizeof(T)), "cudaMalloc x") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&res), n * sizeof(T)), "cudaMalloc res") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&weight), c.cols * sizeof(T)),
               "cudaMalloc weight")) {
    return false;
  }
  cudaMemset(x, 0, n * sizeof(T));
  cudaMemset(res, 0, n * sizeof(T));
  cudaMemset(weight, 0, static_cast<size_t>(c.cols) * sizeof(T));

  const kops::Tensor tx = tensor2d(x, c.dtype, c.rows, c.cols);
  const kops::Tensor tres = tensor2d(res, c.dtype, c.rows, c.cols);
  const kops::Tensor tw = tensor1d(weight, c.dtype, c.cols);

  bool ok = status_ok(
      kops::norm::fused_add_rms_norm(tx, tres, tw, 1e-5f, 0.0f, 1.0f, ctx), c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] { (void)kops::norm::fused_add_rms_norm(tx, tres, tw, 1e-5f, 0.0f, 1.0f, ctx); },
        st);
  }

  cudaFree(x);
  cudaFree(res);
  cudaFree(weight);
  return ok;
}

template <typename T>
T* alloc(size_t n) {
  T* p = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)) != cudaSuccess) return nullptr;
  cudaMemset(p, 0, n * sizeof(T));
  return p;
}

template <typename T>
bool run_layer(const Case& c, bool fused, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* x = alloc<T>(n);
  T* res = alloc<T>(n);
  T* w = alloc<T>(static_cast<size_t>(c.cols));
  T* b = alloc<T>(static_cast<size_t>(c.cols));
  T* out = alloc<T>(n);
  if (!x || !res || !w || !b || !out) return false;
  const kops::Tensor tx = tensor2d(x, c.dtype, c.rows, c.cols);
  const kops::Tensor tres = tensor2d(res, c.dtype, c.rows, c.cols);
  const kops::Tensor tw = tensor1d(w, c.dtype, c.cols);
  const kops::Tensor tb = tensor1d(b, c.dtype, c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  bool ok = false;
  if (fused) {
    ok = status_ok(kops::norm::fused_add_layer_norm(tx, tres, tw, tb, 1e-5f, 0.0f, 1.0f, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::norm::fused_add_layer_norm(tx, tres, tw, tb, 1e-5f, 0.0f, 1.0f, ctx);
    }, st);
  } else {
    ok = status_ok(kops::norm::layer_norm(to, tx, tw, tb, 1e-5f, 0.0f, ctx), c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::norm::layer_norm(to, tx, tw, tb, 1e-5f, 0.0f, ctx);
    }, st);
  }
  cudaFree(x); cudaFree(res); cudaFree(w); cudaFree(b); cudaFree(out);
  return ok;
}

template <typename T>
bool run_fused_out(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* x = alloc<T>(n);
  T* rin = alloc<T>(n);
  T* rout = alloc<T>(n);
  T* w = alloc<T>(static_cast<size_t>(c.cols));
  T* out = alloc<T>(n);
  if (!x || !rin || !rout || !w || !out) return false;
  const kops::Tensor tx = tensor2d(x, c.dtype, c.rows, c.cols);
  const kops::Tensor tri = tensor2d(rin, c.dtype, c.rows, c.cols);
  const kops::Tensor tro = tensor2d(rout, c.dtype, c.rows, c.cols);
  const kops::Tensor tw = tensor1d(w, c.dtype, c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  bool ok = status_ok(kops::norm::fused_add_rms_norm_out(to, tro, tx, tri, tw, 1e-5f, 0.0f, 1.0f, ctx),
                      c.id.c_str());
  if (ok) ok = measure(c, [&] {
    (void)kops::norm::fused_add_rms_norm_out(to, tro, tx, tri, tw, 1e-5f, 0.0f, 1.0f, ctx);
  }, st);
  cudaFree(x); cudaFree(rin); cudaFree(rout); cudaFree(w); cudaFree(out);
  return ok;
}

template <typename T>
bool run_l2(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* x = alloc<T>(n);
  T* out = alloc<T>(n);
  if (!x || !out) return false;
  const kops::Tensor tx = tensor2d(x, c.dtype, c.rows, c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  bool ok = status_ok(kops::norm::l2_norm(to, tx, 1.0f, 1e-6f, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::norm::l2_norm(to, tx, 1.0f, 1e-6f, ctx); }, st);
  cudaFree(x); cudaFree(out);
  return ok;
}

template <typename T>
bool run_gated(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* x = alloc<T>(n);
  T* z = alloc<T>(n);
  T* w = alloc<T>(static_cast<size_t>(c.cols));
  T* out = alloc<T>(n);
  if (!x || !z || !w || !out) return false;
  const kops::Tensor tx = tensor2d(x, c.dtype, c.rows, c.cols);
  const kops::Tensor tz = tensor2d(z, c.dtype, c.rows, c.cols);
  const kops::Tensor tw = tensor1d(w, c.dtype, c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  bool ok = status_ok(kops::norm::gated_rms_norm(to, tx, tz, tw, 128, kops::ActFn::kSiLU, true,
                                                1e-6f, 0.0f, ctx),
                      c.id.c_str());
  if (ok) ok = measure(c, [&] {
    (void)kops::norm::gated_rms_norm(to, tx, tz, tw, 128, kops::ActFn::kSiLU, true, 1e-6f, 0.0f, ctx);
  }, st);
  cudaFree(x); cudaFree(z); cudaFree(w); cudaFree(out);
  return ok;
}

template <typename T>
bool run_qk(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* qkv = alloc<T>(n);
  if (!qkv) return false;
  const kops::Tensor tq = tensor2d(qkv, c.dtype, c.rows, c.cols);
  bool ok = status_ok(kops::norm::qk_norm(tq, kops::Tensor{}, kops::Tensor{}, kops::Tensor{}, 32, 8,
                                          128, kops::NormType::kRMSNorm, false, kops::Tensor{},
                                          1e-6f, 0.0f, ctx),
                      c.id.c_str());
  if (ok) ok = measure(c, [&] {
    (void)kops::norm::qk_norm(tq, kops::Tensor{}, kops::Tensor{}, kops::Tensor{}, 32, 8, 128,
                              kops::NormType::kRMSNorm, false, kops::Tensor{}, 1e-6f, 0.0f, ctx);
  }, st);
  cudaFree(qkv);
  return ok;
}

template <typename T>
bool run_norm_quant(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* x = nullptr;
  T* res = nullptr;
  T* w = nullptr;
  uint8_t* q = nullptr;
  float* scales = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&x), n * sizeof(T)), "x") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&res), n * sizeof(T)), "res") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&w), c.cols * sizeof(T)), "w") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), n), "q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), c.rows * sizeof(float)), "scales")) {
    return false;
  }
  cudaMemset(x, 0, n * sizeof(T));
  cudaMemset(res, 0, n * sizeof(T));
  cudaMemset(w, 0, c.cols * sizeof(T));
  kops::norm::NormQuantParams p{};
  p.input = x;
  p.input_stride = c.cols;
  p.residual = res;
  p.input_scale = 1.0f;
  p.weight = w;
  p.epsilon = 1e-5f;
  p.num_tokens = static_cast<int32_t>(c.rows);
  p.hidden_size = static_cast<int32_t>(c.cols);
  p.quant.q = q;
  p.quant.q_row_stride = c.cols;
  p.quant.scales = scales;
  p.quant.scale_row_stride = 1;
  p.quant.q_dtype = kops::DType::kFloat8E4M3;
  p.quant.mode = kops::ActQuantMode::kDynamicPerToken;
  bool ok = status_ok(kops::norm::norm_quant(p, kops::NormType::kRMSNorm, kops::DType::kBFloat16,
                                             kops::DType::kBFloat16, ctx),
                      c.id.c_str());
  if (ok) ok = measure(c, [&] {
    (void)kops::norm::norm_quant(p, kops::NormType::kRMSNorm, kops::DType::kBFloat16,
                                 kops::DType::kBFloat16, ctx);
  }, st);
  cudaFree(x); cudaFree(res); cudaFree(w); cudaFree(q); cudaFree(scales);
  return ok;
}

template <typename T>
bool run_typed(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "norm_quant") return run_norm_quant<T>(c, ctx, st);
  if (c.op == "rms_norm") return run_rms<T>(c, ctx, st);
  if (c.op == "fused_add_rms_norm") return run_fused<T>(c, ctx, st);
  if (c.op == "layer_norm") return run_layer<T>(c, false, ctx, st);
  if (c.op == "fused_add_layer_norm") return run_layer<T>(c, true, ctx, st);
  if (c.op == "fused_add_rms_norm_out") return run_fused_out<T>(c, ctx, st);
  if (c.op == "l2_norm") return run_l2<T>(c, ctx, st);
  if (c.op == "gated_rms_norm") return run_gated<T>(c, ctx, st);
  if (c.op == "qk_norm") return run_qk<T>(c, ctx, st);
  return false;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.dtype == kops::DType::kFloat32) return run_typed<float>(c, ctx, st);
  if (c.dtype == kops::DType::kBFloat16) return run_typed<__nv_bfloat16>(c, ctx, st);
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
