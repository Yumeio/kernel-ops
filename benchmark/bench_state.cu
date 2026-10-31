// =============================================================================
// kernel-ops | benchmark/bench_state.cu
// B2 L03 state cache: state_copy, state_gather, state_scatter, state_zero.
// Two pools with different slot sizes exercise the per-pool byte geometry.
// =============================================================================
#include "benchmark/harness.h"

#include <cstdint>
#include <vector>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/state/state_cache_kernels.cuh"
#include "csrc/state/conv1d_kernels.cuh"
#include "csrc/state/ssm_kernels.cuh"
#include "csrc/state/linear_attention_kernels.cuh"
#include "csrc/state/delta_rule_kernels.cuh"
#include "csrc/state/ops.h"

namespace {

using namespace kops_bench;
using kops::DType;
using kops::Tensor;

Tensor t1(void* p, int64_t n) { return tensor1d(p, DType::kInt64, n); }

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kPools = 2, kSlots = 256, kB0 = 4096, kB1 = 2048, kItems = 64;
  const int64_t total = static_cast<int64_t>(kItems) * (kB0 + kB1);
  void* pool0 = nullptr;
  void* pool1 = nullptr;
  int64_t* ptrs = nullptr;
  int64_t* bytes = nullptr;
  int32_t* pairs = nullptr;
  int32_t* slots = nullptr;
  void* staging = nullptr;
  if (!cuda_ok(cudaMalloc(&pool0, static_cast<size_t>(kSlots) * kB0), "pool0") ||
      !cuda_ok(cudaMalloc(&pool1, static_cast<size_t>(kSlots) * kB1), "pool1") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ptrs), kPools * 8), "ptrs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&bytes), kPools * 8), "bytes") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pairs), static_cast<size_t>(kItems) * 8), "pairs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&slots), static_cast<size_t>(kItems) * 4), "slots") ||
      !cuda_ok(cudaMalloc(&staging, static_cast<size_t>(total)), "staging")) {
    return false;
  }
  const int64_t pv[kPools] = {reinterpret_cast<int64_t>(pool0), reinterpret_cast<int64_t>(pool1)};
  const int64_t bv[kPools] = {kB0, kB1};
  std::vector<int32_t> qv(static_cast<size_t>(kItems) * 2);
  std::vector<int32_t> sv(static_cast<size_t>(kItems));
  for (int i = 0; i < kItems; ++i) {
    qv[static_cast<size_t>(i * 2)] = i;
    qv[static_cast<size_t>(i * 2 + 1)] = (i + 1) % kSlots;
    sv[static_cast<size_t>(i)] = i;
  }
  cudaMemcpy(ptrs, pv, sizeof(pv), cudaMemcpyHostToDevice);
  cudaMemcpy(bytes, bv, sizeof(bv), cudaMemcpyHostToDevice);
  cudaMemcpy(pairs, qv.data(), qv.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(slots, sv.data(), sv.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(pool0, 1, static_cast<size_t>(kSlots) * kB0);
  cudaMemset(pool1, 1, static_cast<size_t>(kSlots) * kB1);
  cudaMemset(staging, 1, static_cast<size_t>(total));

  const Tensor tp = t1(ptrs, kPools);
  const Tensor tb = t1(bytes, kPools);
  const Tensor tq = tensor2d(pairs, DType::kInt32, kItems, 2);
  const Tensor ts = tensor1d(slots, DType::kInt32, kItems);
  const Tensor tg = tensor2d(staging, DType::kUInt8, kItems, kB0 + kB1);

  bool ok = true;
  auto fn = [&] {
    if (c.op == "state_copy") {
      (void)kops::state::state_copy(tp, tb, tq, ctx);
    } else if (c.op == "state_gather") {
      (void)kops::state::state_gather(tg, tp, tb, ts, ctx);
    } else if (c.op == "state_scatter") {
      (void)kops::state::state_scatter(tp, tg, tb, ts, ctx);
    } else {
      (void)kops::state::state_zero(tp, tb, ts, ctx);
    }
  };
  if (ok) {
    if (c.op == "state_copy") {
      ok = status_ok(kops::state::state_copy(tp, tb, tq, ctx), c.id.c_str());
    } else if (c.op == "state_gather") {
      ok = status_ok(kops::state::state_gather(tg, tp, tb, ts, ctx), c.id.c_str());
    } else if (c.op == "state_scatter") {
      ok = status_ok(kops::state::state_scatter(tp, tg, tb, ts, ctx), c.id.c_str());
    } else {
      ok = status_ok(kops::state::state_zero(tp, tb, ts, ctx), c.id.c_str());
    }
  }
  if (ok) ok = measure(c, fn, st);
  cudaFree(pool0);
  cudaFree(pool1);
  cudaFree(ptrs);
  cudaFree(bytes);
  cudaFree(pairs);
  cudaFree(slots);
  cudaFree(staging);
  return ok;
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  const int64_t bytes = 64LL * (4096 + 2048) * 2;
  cases.push_back({"state_copy.2pools.64items", "state_copy", DType::kUInt8, 64, 6144, bytes, 0,
                   10, 100});
  cases.push_back({"state_gather.2pools.64items", "state_gather", DType::kUInt8, 64, 6144, bytes,
                   0, 10, 100});
  cases.push_back({"state_scatter.2pools.64items", "state_scatter", DType::kUInt8, 64, 6144, bytes,
                   0, 10, 100});
  cases.push_back({"state_zero.2pools.64items", "state_zero", DType::kUInt8, 64, 6144, bytes / 2,
                   0, 10, 100});
  return cases;
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
