// =============================================================================
// kernel-ops | benchmark/bench_quantization.cu
// B3 quantization-family cases. The case table is the shared spec read by
// benchmark/compare/torch_baseline.py through --dump-cases.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <vector>

#include "csrc/common/quant.cuh"
#include "csrc/quantization/ops.h"

namespace {

using namespace kops_bench;

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto add = [&](const char* id, const char* op, kops::DType dt, int64_t rows, int64_t cols,
                 int64_t bytes, int64_t flops) {
    cases.push_back({id, op, dt, rows, cols, bytes, flops, 20, 200});
  };
  const int64_t r = 4096;
  const int64_t c = 4096;
  add("quant.absmax.bf16.4096x4096", "absmax", kops::DType::kBFloat16, r, c, 2 * r * c, r * c);
  add("quant.absmax.fp32.4096x4096", "absmax", kops::DType::kFloat32, r, c, 4 * r * c, r * c);
  add("quant.static.bf16.fp8.4096x4096", "quantize_static", kops::DType::kBFloat16, r, c,
      3 * r * c, r * c);
  add("quant.per_token.bf16.fp8.4096x4096", "quantize_per_token", kops::DType::kBFloat16, r, c,
      3 * r * c, 3 * r * c);
  add("quant.per_token.fp32.int8.4096x4096", "quantize_per_token_int8", kops::DType::kFloat32, r,
      c, 5 * r * c, 3 * r * c);
  add("quant.per_group128.bf16.fp8.4096x4096", "quantize_per_group", kops::DType::kBFloat16, r, c,
      3 * r * c, 3 * r * c);
  add("quant.mxfp8.bf16.4096x4096", "quantize_mxfp8", kops::DType::kBFloat16, r, c, 3 * r * c,
      3 * r * c);
  add("quant.mxfp4.bf16.4096x4096", "quantize_mxfp4", kops::DType::kBFloat16, r, c,
      3 * r * c / 2, 3 * r * c);
  add("quant.nvfp4.bf16.4096x4096", "quantize_nvfp4", kops::DType::kBFloat16, r, c,
      3 * r * c / 2, 3 * r * c);
  add("quant.dequant_per_token.fp8.fp32.4096x4096", "dequantize_per_token",
      kops::DType::kFloat32, r, c, 5 * r * c, r * c);
  add("quant.pack_int4.4096x4096", "pack_int4", kops::DType::kFloat32, r, c, r * c * 3 / 2, r * c);
  add("quant.unpack_int4.4096x4096", "unpack_int4", kops::DType::kFloat32, r, c, r * c * 3 / 2,
      r * c);
  add("quant.hadamard.fp32.4096x4096", "hadamard", kops::DType::kFloat32, r, 4096, 2 * r * 4096,
      r * 4096 * 12);
  return cases;
}

template <typename T>
bool run_typed(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  uint8_t* q = nullptr;
  float* scales = nullptr;
  int32_t* zero = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), n * sizeof(T)), "in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), n), "q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), (c.rows + 1) * sizeof(float)),
               "scales") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&zero), c.rows * sizeof(int32_t)), "zero")) {
    return false;
  }
  cudaMemset(in, 0, n * sizeof(T));
  cudaMemset(q, 0, n);
  cudaMemset(scales, 0, (c.rows + 1) * sizeof(float));
  const kops::Tensor tin = tensor2d(in, c.dtype, c.rows, c.cols);
  bool ok = false;
  if (c.op == "absmax") {
    kops::Tensor amax = tensor1d(scales, kops::DType::kFloat32, 1);
    cudaMemset(scales, 0, sizeof(float));
    ok = status_ok(kops::quant::absmax(amax, tin, ctx), c.id.c_str());
    if (ok) ok = measure(c, [&] { (void)kops::quant::absmax(amax, tin, ctx); }, st);
  } else if (c.op == "quantize_static") {
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols;
    o.q_dtype = kops::DType::kFloat8E4M3;
    o.mode = kops::ActQuantMode::kStaticPerTensor;
    o.static_scale = scales;
    const float one = 1.0f;
    cudaMemcpy(scales, &one, sizeof(float), cudaMemcpyHostToDevice);
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
  } else if (c.op == "quantize_per_token" || c.op == "quantize_per_token_int8") {
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols;
    o.scales = scales;
    o.scale_row_stride = 1;
    o.q_dtype = c.op == "quantize_per_token" ? kops::DType::kFloat8E4M3 : kops::DType::kInt8;
    o.mode = kops::ActQuantMode::kDynamicPerToken;
    o.symmetric = o.q_dtype != kops::DType::kInt8;
    o.zero_points = o.symmetric ? nullptr : zero;
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
  } else if (c.op == "quantize_per_group") {
    const int64_t groups = c.cols / 128;
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols;
    o.scales = scales;
    o.scale_row_stride = groups;
    o.q_dtype = kops::DType::kFloat8E4M3;
    o.mode = kops::ActQuantMode::kPerGroup;
    o.group_size = 128;
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
  } else if (c.op == "quantize_mxfp8" || c.op == "quantize_mxfp4") {
    const int64_t groups = c.cols / 32;
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.op == "quantize_mxfp4" ? c.cols / 2 : c.cols;
    o.scales = scales;
    o.scale_row_stride = groups;
    o.q_dtype = c.op == "quantize_mxfp4" ? kops::DType::kFloat4E2M1x2 : kops::DType::kFloat8E4M3;
    o.mode = c.op == "quantize_mxfp4" ? kops::ActQuantMode::kMXFP4 : kops::ActQuantMode::kMXFP8;
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
  } else if (c.op == "quantize_nvfp4") {
    const int64_t groups = c.cols / 16;
    const int64_t sf_bytes = ((c.rows + 127) / 128) * ((groups + 3) / 4) * 512;
    const int64_t sf_padded = (sf_bytes + 3) & ~int64_t(3);
    uint8_t* sf = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sf), sf_padded + sizeof(float)), "sf")) return false;
    float* gscale = reinterpret_cast<float*>(sf + sf_padded);
    float g = 1.0f;
    cudaMemcpy(gscale, &g, sizeof(float), cudaMemcpyHostToDevice);
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols / 2;
    o.scales = sf;
    o.scale_row_stride = groups;
    o.q_dtype = kops::DType::kFloat4E2M1x2;
    o.mode = kops::ActQuantMode::kNVFP4;
    o.global_scale = gscale;
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
    cudaFree(sf);
  } else if (c.op == "dequantize_per_token") {
    float* out = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(float)), "out")) return false;
    kops::QuantScheme scheme{};
    scheme.storage_dtype = kops::DType::kFloat8E4M3;
    scheme.scale_dtype = kops::DType::kFloat32;
    scheme.granularity = kops::ScaleGranularity::kPerToken;
    const kops::Tensor tq = tensor2d(q, kops::DType::kFloat8E4M3, c.rows, c.cols);
    const kops::Tensor tsc = tensor1d(scales, kops::DType::kFloat32, c.rows);
    const kops::Tensor tout = tensor2d(out, kops::DType::kFloat32, c.rows, c.cols);
    ok = status_ok(kops::quant::dequantize(tout, tq, tsc, kops::Tensor{}, scheme, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::dequantize(tout, tq, tsc, kops::Tensor{}, scheme, ctx);
    }, st);
    cudaFree(out);
  } else if (c.op == "pack_int4" || c.op == "unpack_int4") {
    int8_t* i8 = nullptr;
    uint8_t* u8 = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&i8), n), "i8") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&u8), n / 2), "u8")) {
      return false;
    }
    if (c.op == "pack_int4") {
      const kops::Tensor ti = tensor2d(i8, kops::DType::kInt8, c.rows, c.cols);
      const kops::Tensor to = tensor2d(u8, kops::DType::kUInt8, c.rows, c.cols / 2);
      ok = status_ok(kops::quant::pack_int4(to, ti, false, ctx), c.id.c_str());
      if (ok) ok = measure(c, [&] { (void)kops::quant::pack_int4(to, ti, false, ctx); }, st);
    } else {
      const kops::Tensor ti = tensor2d(u8, kops::DType::kUInt8, c.rows, c.cols / 2);
      const kops::Tensor to = tensor2d(i8, kops::DType::kInt8, c.rows, c.cols);
      ok = status_ok(kops::quant::unpack_int4(to, ti, false, ctx), c.id.c_str());
      if (ok) ok = measure(c, [&] { (void)kops::quant::unpack_int4(to, ti, false, ctx); }, st);
    }
    cudaFree(i8);
    cudaFree(u8);
  } else if (c.op == "hadamard") {
    float* out = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(float)), "out")) return false;
    const kops::Tensor ti = tensor2d(in, kops::DType::kFloat32, c.rows, c.cols);
    const kops::Tensor to = tensor2d(out, kops::DType::kFloat32, c.rows, c.cols);
    ok = status_ok(kops::quant::hadamard_transform(to, ti, 1.0f / std::sqrt(float(c.cols)), false,
                                                   0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::hadamard_transform(to, ti, 1.0f / std::sqrt(float(c.cols)), false, 0, ctx);
    }, st);
    cudaFree(out);
  }
  cudaFree(in);
  cudaFree(q);
  cudaFree(scales);
  cudaFree(zero);
  return ok;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.dtype == kops::DType::kFloat32 || c.op == "dequantize_per_token" ||
      c.op == "hadamard") {
    return run_typed<float>(c, ctx, st);
  }
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
  const std::vector<Case> cases = make_cases();
  if (opt.dump_cases) {
    dump_cases(cases, dev);
    return 0;
  }
  kops::LaunchContext ctx{};
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) return 1;
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
