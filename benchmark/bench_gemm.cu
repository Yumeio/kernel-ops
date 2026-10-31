// =============================================================================
// kernel-ops | benchmark/bench_gemm.cu
// B4 gemm-family cases (operand preparation).  The case table is the shared spec read by
// benchmark/compare/torch_baseline.py through --dump-cases.
//
// These are load-time operations: the figure of merit is the time to prepare one weight
// ("prep time"), reported per recipe.  bytes counts the compulsory input + output traffic.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <vector>

#include "csrc/common/quant.cuh"
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

using namespace kops_bench;

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto add = [&](const char* id, const char* op, kops::DType dt, int64_t rows, int64_t cols,
                 int64_t bytes) {
    cases.push_back({id, op, dt, rows, cols, bytes, 0, 20, 200});
  };
  const int64_t r = 4096;
  const int64_t c = 4096;
  add("gemm.transpose.bf16.4096x4096", "weight_transpose", kops::DType::kBFloat16, r, c, 4 * r * c);
  add("gemm.transpose.fp8.4096x4096", "weight_transpose", kops::DType::kFloat8E4M3, r, c, 2 * r * c);
  add("gemm.transpose.int4.4096x4096", "weight_transpose_int4", kops::DType::kUInt8, r, c, r * c);
  add("gemm.glu_interleave.bf16.8192x4096", "glu_interleave", kops::DType::kBFloat16, 8192, c,
      4 * 8192 * c);
  add("gemm.router_split.fp32.4096x4096", "router_split", kops::DType::kFloat32, r, c, 8 * r * c);
  add("gemm.int8_colsum.4096x4096", "int8_colsum", kops::DType::kInt8, r, c, r * c);
  add("gemm.scale_swizzle.14336x256", "block_scale_swizzle", kops::DType::kUInt8, 14336, 256,
      2 * 14336 * 256);
  add("gemm.scale_tma.f32.32768x56", "fp8_scale_to_tma", kops::DType::kFloat32, 32768, 56,
      8 * 32768 * 56);
  add("gemm.scale_tma.ue8m0.32768x56", "fp8_scale_to_tma_ue8m0", kops::DType::kFloat32, 32768, 56,
      5 * 32768 * 56);
  add("gemm.fp8_block_quant.bf16.4096x4096", "fp8_block_quant", kops::DType::kBFloat16, r, c,
      3 * r * c);
  add("gemm.fp8_block_quant.ue8m0.bf16.4096x4096", "fp8_block_quant_ue8m0", kops::DType::kBFloat16,
      r, c, 3 * r * c);
  return cases;
}

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) {
    if (!cuda_ok(cudaMalloc(&p, bytes), "cudaMalloc")) p = nullptr;
    else cudaMemset(p, 0x3C, bytes);
  }
  ~Buf() { if (p != nullptr) cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t r = c.rows;
  const int64_t k = c.cols;
  const size_t n = static_cast<size_t>(r * k);
  const char* id = c.id.c_str();
  auto go = [&](auto&& op) {
    if (!status_ok(op(), id)) return false;
    return measure(c, [&] { (void)op(); }, st);
  };
  if (c.op == "weight_transpose") {
    const size_t eb = c.dtype == kops::DType::kFloat8E4M3 ? 1 : 2;
    Buf in(n * eb), out(n * eb);
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, c.dtype, r, k);
    const kops::Tensor to = tensor2d(out.p, c.dtype, k, r);
    return go([&] { return kops::gemm::weight_transpose(to, ti, ctx); });
  }
  if (c.op == "weight_transpose_int4") {
    Buf in(n / 2), out(n / 2);
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, kops::DType::kInt4x2, r, k);
    const kops::Tensor to = tensor2d(out.p, kops::DType::kInt4x2, k, r);
    return go([&] { return kops::gemm::weight_transpose(to, ti, ctx); });
  }
  if (c.op == "glu_interleave") {
    Buf in(n * 2), out(n * 2);
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, c.dtype, r, k);
    const kops::Tensor to = tensor2d(out.p, c.dtype, r, k);
    // [gate | up] halves -> blocks of 64 rows (a 128-row N tile holds matching gate/up rows).
    return go([&] {
      return kops::gemm::glu_weight_interleave(to, kops::Tensor{}, kops::Tensor{}, ti, kops::Tensor{},
                                               kops::Tensor{}, 0, 64, ctx);
    });
  }
  if (c.op == "router_split") {
    Buf w(n * 4), hi(n * 2), lo(n * 2);
    if (w.p == nullptr || hi.p == nullptr || lo.p == nullptr) return false;
    const kops::Tensor tw = tensor2d(w.p, kops::DType::kFloat32, r, k);
    const kops::Tensor thi = tensor2d(hi.p, kops::DType::kBFloat16, r, k);
    const kops::Tensor tlo = tensor2d(lo.p, kops::DType::kBFloat16, r, k);
    return go([&] { return kops::gemm::router_weight_split_bf16x2(thi, tlo, tw, ctx); });
  }
  if (c.op == "int8_colsum") {
    Buf w(n), sum(static_cast<size_t>(r) * 4);
    if (w.p == nullptr || sum.p == nullptr) return false;
    const kops::Tensor tw = tensor2d(w.p, kops::DType::kInt8, r, k);
    const kops::Tensor ts = tensor1d(sum.p, kops::DType::kInt32, r);
    return go([&] { return kops::gemm::int8_weight_colsum(ts, tw, ctx); });
  }
  if (c.op == "block_scale_swizzle") {
    const int64_t rp = (r + 127) / 128 * 128;
    const int64_t cp = (k + 3) / 4 * 4;
    Buf in(n), out(static_cast<size_t>(rp * cp));
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, kops::DType::kFloat8E4M3, r, k);
    const kops::Tensor to = tensor2d(out.p, kops::DType::kFloat8E4M3, rp, cp);
    return go([&] { return kops::gemm::block_scale_swizzle(to, ti, false, ctx); });
  }
  if (c.op == "fp8_scale_to_tma" || c.op == "fp8_scale_to_tma_ue8m0") {
    const bool pack = c.op == "fp8_scale_to_tma_ue8m0";
    Buf in(n * 4), out(n * 4);
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, kops::DType::kFloat32, r, k);
    const kops::Tensor to = pack ? tensor2d(out.p, kops::DType::kInt32, k / 4, r)
                                 : tensor2d(out.p, kops::DType::kFloat32, k, r);
    return go([&] { return kops::gemm::fp8_scale_to_tma_layout(to, ti, kops::Tensor{}, pack, ctx); });
  }
  if (c.op == "fp8_block_quant" || c.op == "fp8_block_quant_ue8m0") {
    const bool ue = c.op == "fp8_block_quant_ue8m0";
    Buf w(n * 2), q(n), s(static_cast<size_t>((r / 128) * (k / 128)) * 4);
    if (w.p == nullptr || q.p == nullptr || s.p == nullptr) return false;
    const kops::Tensor tw = tensor2d(w.p, kops::DType::kBFloat16, r, k);
    const kops::Tensor tq = tensor2d(q.p, kops::DType::kFloat8E4M3, r, k);
    const kops::Tensor ts = tensor2d(s.p, kops::DType::kFloat32, r / 128, k / 128);
    return go([&] { return kops::gemm::fp8_block2d_quant_weight(tq, ts, tw, 128, 128, ue, ctx); });
  }
  std::fprintf(stderr, "[bench] unknown op %s\n", c.op.c_str());
  return false;
}

}  // namespace

int main(int argc, char** argv) {
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
