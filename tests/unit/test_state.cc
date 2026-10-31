// =============================================================================
// kernel-ops | tests/unit/test_state.cc
// B2 state_cache (L03): state_copy, state_gather, state_scatter, state_zero.
//
// The test uses two pools with different slot sizes (64 B conv state, 32 B
// SSM/delta state) to exercise the per-pool byte geometry, plus graph replay of
// the copy-on-write path. state_commit_accepted belongs to B10 and must still
// report kNotImplemented.
// =============================================================================
#include "tests/unit/harness.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"
#include "csrc/state/state_cache_kernels.cuh"
#include "csrc/state/conv1d_kernels.cuh"
#include "csrc/state/ssm_kernels.cuh"
#include "csrc/state/linear_attention_kernels.cuh"
#include "csrc/state/delta_rule_kernels.cuh"
#include "csrc/state/ops.h"

namespace {

using kops::DType;
using kops::LaunchContext;
using kops::Status;
using kops::Tensor;

template <typename T>
struct DevBuf {
  T* p = nullptr;
  explicit DevBuf(size_t n) : p(kops_test::device_alloc<T>(n)) {}
  ~DevBuf() { kops_test::device_free(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  bool ok() const { return p != nullptr; }
};

Tensor tensor1d(void* data, DType dtype, int64_t n) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

Tensor tensor2d(void* data, DType dtype, int64_t rows, int64_t cols) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = cols;
  t.stride[1] = 1;
  return t;
}

struct Fixture {
  static constexpr int kSlots = 2;
  static constexpr int kBytes0 = 64;
  static constexpr int kBytes1 = 32;
  static constexpr int kTotal = kBytes0 + kBytes1;

  DevBuf<uint8_t> pool0;
  DevBuf<uint8_t> pool1;
  DevBuf<int64_t> ptrs;
  DevBuf<int64_t> bytes;

  explicit Fixture()
      : pool0(static_cast<size_t>(kSlots) * kBytes0),
        pool1(static_cast<size_t>(kSlots) * kBytes1),
        ptrs(2),
        bytes(2) {}

  bool ok() const { return pool0.ok() && pool1.ok() && ptrs.ok() && bytes.ok(); }

  void init() {
    std::vector<uint8_t> a(static_cast<size_t>(kSlots) * kBytes0);
    std::vector<uint8_t> b(static_cast<size_t>(kSlots) * kBytes1);
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(0x10 + (i & 0x3F));
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint8_t>(0x80 + (i & 0x1F));
    kops_test::copy_in(pool0.p, a);
    kops_test::copy_in(pool1.p, b);
    const int64_t pv[2] = {reinterpret_cast<int64_t>(pool0.p), reinterpret_cast<int64_t>(pool1.p)};
    const int64_t bv[2] = {kBytes0, kBytes1};
    kops_test::copy_in(ptrs.p, std::vector<int64_t>(pv, pv + 2));
    kops_test::copy_in(bytes.p, std::vector<int64_t>(bv, bv + 2));
  }

  Tensor tp() { return tensor1d(ptrs.p, DType::kInt64, 2); }
  Tensor tb() { return tensor1d(bytes.p, DType::kInt64, 2); }
};

void test_copy(LaunchContext& ctx) {
  Fixture f;
  if (!f.ok()) return;
  f.init();
  DevBuf<int32_t> pairs(2);
  if (!pairs.ok()) return;
  kops_test::copy_in(pairs.p, std::vector<int32_t>{0, 1});
  const Tensor tp = tensor2d(pairs.p, DType::kInt32, 1, 2);
  KOPS_EXPECT_OK(kops::state::state_copy(f.tp(), f.tb(), tp, ctx), "state_copy");
  cudaDeviceSynchronize();
  const auto p0 = kops_test::copy_out(f.pool0.p, Fixture::kSlots * Fixture::kBytes0);
  const auto p1 = kops_test::copy_out(f.pool1.p, Fixture::kSlots * Fixture::kBytes1);
  for (int i = 0; i < Fixture::kBytes0; ++i) {
    KOPS_TEST_CHECK(p0[static_cast<size_t>(Fixture::kBytes0 + i)] == p0[static_cast<size_t>(i)]);
  }
  for (int i = 0; i < Fixture::kBytes1; ++i) {
    KOPS_TEST_CHECK(p1[static_cast<size_t>(Fixture::kBytes1 + i)] == p1[static_cast<size_t>(i)]);
  }
}

void test_gather_scatter(LaunchContext& ctx) {
  Fixture f;
  if (!f.ok()) return;
  f.init();
  DevBuf<int32_t> slots(2);
  DevBuf<uint8_t> staging(2 * Fixture::kTotal);
  if (!slots.ok() || !staging.ok()) return;
  kops_test::copy_in(slots.p, std::vector<int32_t>{0, 1});
  const Tensor ts = tensor1d(slots.p, DType::kInt32, 2);
  const Tensor to = tensor2d(staging.p, DType::kUInt8, 2, Fixture::kTotal);
  KOPS_EXPECT_OK(kops::state::state_gather(to, f.tp(), f.tb(), ts, ctx), "state_gather");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(staging.p, 2 * Fixture::kTotal);
  const auto p0 = kops_test::copy_out(f.pool0.p, Fixture::kSlots * Fixture::kBytes0);
  const auto p1 = kops_test::copy_out(f.pool1.p, Fixture::kSlots * Fixture::kBytes1);
  for (int item = 0; item < 2; ++item) {
    for (int i = 0; i < Fixture::kBytes0; ++i) {
      KOPS_TEST_CHECK(got[static_cast<size_t>(item * Fixture::kTotal + i)] ==
                      p0[static_cast<size_t>(item * Fixture::kBytes0 + i)]);
    }
    for (int i = 0; i < Fixture::kBytes1; ++i) {
      KOPS_TEST_CHECK(got[static_cast<size_t>(item * Fixture::kTotal + Fixture::kBytes0 + i)] ==
                      p1[static_cast<size_t>(item * Fixture::kBytes1 + i)]);
    }
  }

  // Modify the staging rows, scatter to the other slot, and read them back.
  std::vector<uint8_t> modified = got;
  for (uint8_t& v : modified) v = static_cast<uint8_t>(~v);
  kops_test::copy_in(staging.p, modified);
  const Tensor ts2 = tensor1d(slots.p, DType::kInt32, 2);
  KOPS_EXPECT_OK(kops::state::state_scatter(f.tp(), to, f.tb(), ts2, ctx), "state_scatter");
  cudaDeviceSynchronize();
  DevBuf<uint8_t> check(2 * Fixture::kTotal);
  if (!check.ok()) return;
  const Tensor tc = tensor2d(check.p, DType::kUInt8, 2, Fixture::kTotal);
  KOPS_EXPECT_OK(kops::state::state_gather(tc, f.tp(), f.tb(), ts2, ctx), "state_gather check");
  cudaDeviceSynchronize();
  const auto round = kops_test::copy_out(check.p, 2 * Fixture::kTotal);
  KOPS_TEST_CHECK(std::memcmp(round.data(), modified.data(), modified.size()) == 0);
}

void test_zero(LaunchContext& ctx) {
  Fixture f;
  if (!f.ok()) return;
  f.init();
  DevBuf<int32_t> slots(1);
  if (!slots.ok()) return;
  kops_test::copy_in(slots.p, std::vector<int32_t>{1});
  const Tensor ts = tensor1d(slots.p, DType::kInt32, 1);
  KOPS_EXPECT_OK(kops::state::state_zero(f.tp(), f.tb(), ts, ctx), "state_zero");
  cudaDeviceSynchronize();
  const auto p0 = kops_test::copy_out(f.pool0.p, Fixture::kSlots * Fixture::kBytes0);
  const auto p1 = kops_test::copy_out(f.pool1.p, Fixture::kSlots * Fixture::kBytes1);
  for (int i = 0; i < Fixture::kBytes0; ++i) {
    KOPS_TEST_CHECK(p0[static_cast<size_t>(Fixture::kBytes0 + i)] == 0);
  }
  for (int i = 0; i < Fixture::kBytes1; ++i) {
    KOPS_TEST_CHECK(p1[static_cast<size_t>(Fixture::kBytes1 + i)] == 0);
  }
  // Slot 0 is untouched.
  KOPS_TEST_CHECK(p0[0] != 0);
  KOPS_TEST_CHECK(p1[0] != 0);
}

void test_graph(LaunchContext& ctx) {
  Fixture f;
  if (!f.ok()) return;
  f.init();
  DevBuf<int32_t> pairs(2);
  if (!pairs.ok()) return;
  kops_test::copy_in(pairs.p, std::vector<int32_t>{0, 1});
  const Tensor tp = tensor2d(pairs.p, DType::kInt32, 1, 2);
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return;
  LaunchContext gc = ctx;
  gc.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  KOPS_EXPECT_OK(kops::state::state_copy(f.tp(), f.tb(), tp, gc), "graph state_copy");
  cudaStreamEndCapture(stream, &graph);
  if (graph != nullptr && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    const auto p0 = kops_test::copy_out(f.pool0.p, Fixture::kSlots * Fixture::kBytes0);
    KOPS_TEST_CHECK(p0[static_cast<size_t>(Fixture::kBytes0)] == p0[0]);
    cudaGraphExecDestroy(exec);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

void test_contracts(LaunchContext& ctx) {
  Fixture f;
  if (!f.ok()) return;
  f.init();
  DevBuf<int32_t> slots(1);
  DevBuf<uint8_t> out(Fixture::kTotal);
  if (!slots.ok() || !out.ok()) return;
  kops_test::copy_in(slots.p, std::vector<int32_t>{0});
  const Tensor ts = tensor1d(slots.p, DType::kInt32, 1);
  const Tensor to = tensor2d(out.p, DType::kUInt8, 1, Fixture::kTotal);
  // Pool count disagreeing with slot_bytes.
  const Tensor bad_pools = tensor1d(f.ptrs.p, DType::kInt64, 1);
  KOPS_EXPECT_STATUS(kops::state::state_copy(bad_pools, f.tb(), tensor2d(slots.p, DType::kInt32, 1, 2), ctx),
                     Status::kInvalidArgument, "state pool count");
  // Wrong slots dtype.
  KOPS_EXPECT_STATUS(kops::state::state_gather(to, f.tp(), f.tb(),
                                               tensor1d(slots.p, DType::kInt64, 1), ctx),
                     Status::kUnsupportedDtype, "state slot dtype");
  // Empty staging width.
  KOPS_EXPECT_STATUS(kops::state::state_gather(tensor2d(out.p, DType::kUInt8, 1, 0), f.tp(), f.tb(),
                                               ts, ctx),
                     Status::kInvalidArgument, "state empty out");
  KOPS_EXPECT_STATUS(kops::state::state_commit_accepted(f.tp(), f.tb(), ts, ts, ts, ctx),
                     Status::kNotImplemented, "state_commit_accepted B10");
}

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
  test_copy(ctx);
  test_gather_scatter(ctx);
  test_zero(ctx);
  test_graph(ctx);
  test_contracts(ctx);
  return kops_test::finish("test_state");
}
