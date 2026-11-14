// =============================================================================
// kernel-ops | tests/unit/test_state.cc
// B2 state_cache (L03): state_copy, state_gather, state_scatter, state_zero.
//
// The test uses two pools with different slot sizes (64 B conv state, 32 B
// SSM/delta state) to exercise the per-pool byte geometry, plus graph replay of
// the copy-on-write path. state_commit_accepted belongs to B10 and must still
// report kNotImplemented.
//
// B9 (L01-L02): conv1d, selective state update / scan, SSD, lightning attention and the
// gated delta rule (recurrent, fused decode, gating, chunked fp32 / tensor core) against the
// fp64 oracles of csrc/reference, plus chunked <-> recurrent parity and bitwise resume.
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/state/state_cache_kernels.cuh"
#include "csrc/state/conv1d_kernels.cuh"
#include "csrc/state/ssm_kernels.cuh"
#include "csrc/state/linear_attention_kernels.cuh"
#include "csrc/state/delta_rule_kernels.cuh"
#include "csrc/state/ops.h"
#include "csrc/reference/ops.h"

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

void test_commit(LaunchContext& ctx) {
  constexpr int kSlots = 8, kB0 = 64, kB1 = 32, B = 4, M = 4;
  DevBuf<uint8_t> pool0(static_cast<size_t>(kSlots) * kB0);
  DevBuf<uint8_t> pool1(static_cast<size_t>(kSlots) * kB1);
  DevBuf<int64_t> ptrs(2);
  DevBuf<int64_t> bytes(2);
  DevBuf<int32_t> live(B);
  DevBuf<int32_t> inter(B * M);
  DevBuf<int32_t> num(B);
  if (!pool0.ok() || !pool1.ok() || !ptrs.ok() || !bytes.ok() || !live.ok() || !inter.ok() ||
      !num.ok()) return;
  std::vector<uint8_t> a(static_cast<size_t>(kSlots) * kB0);
  std::vector<uint8_t> b(static_cast<size_t>(kSlots) * kB1);
  for (int s = 0; s < kSlots; ++s) {
    for (int i = 0; i < kB0; ++i) a[static_cast<size_t>(s * kB0 + i)] = static_cast<uint8_t>(s * 17 + i * 3);
    for (int i = 0; i < kB1; ++i) b[static_cast<size_t>(s * kB1 + i)] = static_cast<uint8_t>(s * 29 + i * 5);
  }
  kops_test::copy_in(pool0.p, a);
  kops_test::copy_in(pool1.p, b);
  const int64_t pv[2] = {reinterpret_cast<int64_t>(pool0.p), reinterpret_cast<int64_t>(pool1.p)};
  const int64_t bv[2] = {kB0, kB1};
  kops_test::copy_in(ptrs.p, std::vector<int64_t>(pv, pv + 2));
  kops_test::copy_in(bytes.p, std::vector<int64_t>(bv, bv + 2));
  kops_test::copy_in(live.p, std::vector<int32_t>{0, 1, 2, -1});
  kops_test::copy_in(inter.p, std::vector<int32_t>{4, 5, 6, 7, 5, 6, 7, 4, 6, 7, 4, 5, 4, 5, 6, 7});
  kops_test::copy_in(num.p, std::vector<int32_t>{0, 2, 3, 1});

  const Tensor tp = tensor1d(ptrs.p, DType::kInt64, 2);
  const Tensor tb = tensor1d(bytes.p, DType::kInt64, 2);
  const Tensor tl = tensor1d(live.p, DType::kInt32, B);
  const Tensor ti = tensor2d(inter.p, DType::kInt32, B, M);
  const Tensor tn = tensor1d(num.p, DType::kInt32, B);
  KOPS_EXPECT_OK(kops::state::state_commit_accepted(tp, tb, tl, ti, tn, ctx), "state_commit");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  auto p0 = kops_test::copy_out(pool0.p, static_cast<size_t>(kSlots) * kB0);
  auto p1 = kops_test::copy_out(pool1.p, static_cast<size_t>(kSlots) * kB1);
  // row 0: accepted 0 -> slot 4; row 1: accepted 2 -> slot 7; row 2 all
  // accepted -> live untouched; row 3 negative live -> untouched.
  for (int i = 0; i < kB0; ++i) {
    KOPS_TEST_CHECK(p0[static_cast<size_t>(0 * kB0 + i)] == a[static_cast<size_t>(4 * kB0 + i)]);
    KOPS_TEST_CHECK(p0[static_cast<size_t>(1 * kB0 + i)] == a[static_cast<size_t>(7 * kB0 + i)]);
    KOPS_TEST_CHECK(p0[static_cast<size_t>(2 * kB0 + i)] == a[static_cast<size_t>(2 * kB0 + i)]);
  }
  for (int i = 0; i < kB1; ++i) {
    KOPS_TEST_CHECK(p1[static_cast<size_t>(0 * kB1 + i)] == b[static_cast<size_t>(4 * kB1 + i)]);
    KOPS_TEST_CHECK(p1[static_cast<size_t>(1 * kB1 + i)] == b[static_cast<size_t>(7 * kB1 + i)]);
    KOPS_TEST_CHECK(p1[static_cast<size_t>(2 * kB1 + i)] == b[static_cast<size_t>(2 * kB1 + i)]);
  }

  // Graph replay: the recorded pointers stay valid, only num_accepted changes.
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) == cudaSuccess) {
    LaunchContext gc = ctx;
    gc.stream = reinterpret_cast<kops::kops_stream_t>(stream);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    KOPS_EXPECT_OK(kops::state::state_commit_accepted(tp, tb, tl, ti, tn, gc), "graph commit");
    cudaStreamEndCapture(stream, &graph);
    if (graph != nullptr && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
      kops_test::copy_in(num.p, std::vector<int32_t>{1, 1, 1, 1});
      cudaGraphLaunch(exec, stream);
      cudaStreamSynchronize(stream);
      p0 = kops_test::copy_out(pool0.p, static_cast<size_t>(kSlots) * kB0);
      for (int i = 0; i < kB0; ++i) {
        KOPS_TEST_CHECK(p0[static_cast<size_t>(0 * kB0 + i)] == a[static_cast<size_t>(5 * kB0 + i)]);
        KOPS_TEST_CHECK(p0[static_cast<size_t>(1 * kB0 + i)] == a[static_cast<size_t>(6 * kB0 + i)]);
        KOPS_TEST_CHECK(p0[static_cast<size_t>(2 * kB0 + i)] == a[static_cast<size_t>(7 * kB0 + i)]);
      }
      cudaGraphExecDestroy(exec);
    }
    if (graph != nullptr) cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }

  // Contract checks.
  const Tensor bad_live = tensor1d(live.p, DType::kInt32, B - 1);
  KOPS_EXPECT_STATUS(kops::state::state_commit_accepted(tp, tb, bad_live, ti, tn, ctx),
                     Status::kInvalidArgument, "commit live length");
  const Tensor bad_inter = tensor2d(inter.p, DType::kInt32, B + 1, M);
  KOPS_EXPECT_STATUS(kops::state::state_commit_accepted(tp, tb, tl, bad_inter, tn, ctx),
                     Status::kInvalidArgument, "commit intermediate rows");
  const Tensor bad_num = tensor1d(num.p, DType::kInt64, B);
  KOPS_EXPECT_STATUS(kops::state::state_commit_accepted(tp, tb, tl, ti, bad_num, ctx),
                     Status::kUnsupportedDtype, "commit num dtype");
  Tensor padded = tensor2d(inter.p, DType::kInt32, B, M);
  padded.stride[0] = M + 1;
  KOPS_EXPECT_STATUS(kops::state::state_commit_accepted(tp, tb, tl, padded, tn, ctx),
                     Status::kInvalidArgument, "commit padded rows");
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
                     Status::kInvalidArgument, "state_commit_accepted shape");
}

// =============================================================================
// B9 recurrent kernels against the fp64 oracles of csrc/reference (one sequence per oracle
// call; varlen batches, slots, initial states and snapshots are composed here).
// =============================================================================

using kops::ActFn;

struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  double next() {  // uniform in [-1, 1)
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<double>(s >> 11) * (2.0 / 9007199254740992.0) - 1.0;
  }
};

template <typename T>
T to_t(double v) {
  T r;
  kops_test::from_float(r, static_cast<float>(v));
  return r;
}
template <typename T>
double to_d(T v) {
  return static_cast<double>(kops_test::to_float(v));
}

/// n values lo + (hi - lo) * u stored as T; `exact` receives the stored values in fp64.
template <typename T>
std::vector<T> rand_t(size_t n, Lcg& r, double lo, double hi, std::vector<double>* exact) {
  std::vector<T> v(n);
  if (exact) exact->resize(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = to_t<T>(lo + (hi - lo) * 0.5 * (r.next() + 1.0));
    if (exact) (*exact)[i] = to_d(v[i]);
  }
  return v;
}

template <typename T>
DType dtype_of() {
  if (std::is_same<T, float>::value) return DType::kFloat32;
  if (std::is_same<T, __half>::value) return DType::kFloat16;
  return DType::kBFloat16;
}

template <typename T>
T* upload(const std::vector<T>& h) {
  T* p = kops_test::device_alloc<T>(h.empty() ? 1 : h.size());
  if (p && !h.empty()) kops_test::copy_in(p, h);
  return p;
}

/// Owns every device buffer of one test case.
struct Arena {
  std::vector<void*> ptrs;
  template <typename T>
  T* put(const std::vector<T>& h) {
    T* p = upload(h);
    ptrs.push_back(p);
    return p;
  }
  ~Arena() {
    for (void* p : ptrs) kops_test::device_free(p);
  }
};

template <typename T>
bool bits_equal(T a, T b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// Output tolerances relative to the case's output scale (max |want| over the rows a kernel writes):
/// measured floors are ~1 ULP of the max for 16-bit outputs and 1e-7..1e-6 of the max in fp32.
template <typename T>
double rel_atol() { return std::is_same<T, float>::value ? 5e-6 : 4.5e-3; }
template <typename T>
double rel_rtol() { return std::is_same<T, float>::value ? 5e-5 : 1e-2; }

/// max |want| ignoring the sentinel 7.0 of rows a kernel must leave untouched.
double out_scale(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v)
    if (x != 7.0) m = std::max(m, std::fabs(x));
  return m;
}

template <typename T>
void compare_out(const std::vector<T>& got, const std::vector<double>& want, const char* what, double widen = 1.0) {
  kops_test::compare(got, want, widen * rel_atol<T>() * out_scale(want), widen * rel_rtol<T>(), what);
}

// ---- B9 row 1: causal_conv1d_fwd / causal_conv1d_update ------------------------------------

template <typename T, typename W>
void test_conv1d_fwd(LaunchContext& ctx, int dim, int width, ActFn act, bool with_bias) {
  Lcg rng(static_cast<uint64_t>(dim * 31 + width * 7 + static_cast<int>(act)));
  const std::vector<int32_t> lens = {0, 1, 70, 133, 9};
  const std::vector<int32_t> slots = {5, 1, 2, 0, 7};  // slot 7 == pad_slot_id: skipped
  const std::vector<uint8_t> has_init = {1, 0, 1, 1, 1};
  const int B = static_cast<int>(lens.size()), kSlots = 8, H = width - 1, kSnap = 2;
  std::vector<int32_t> qsl(B + 1, 0);
  for (int b = 0; b < B; ++b) qsl[b + 1] = qsl[b] + lens[b];
  const int ntok = qsl[B];
  std::vector<int32_t> snap(static_cast<size_t>(B) * kSnap, -1);
  snap[3 * kSnap + 0] = 3;  // sequence 3: states after 64 and 128 tokens
  snap[3 * kSnap + 1] = 4;
  snap[2 * kSnap + 0] = 6;  // sequence 2: after 64 tokens

  std::vector<double> xd, wd, bd, sd;
  const auto x = rand_t<T>(static_cast<size_t>(ntok) * dim, rng, -1.0, 1.0, &xd);
  const auto w = rand_t<W>(static_cast<size_t>(dim) * width, rng, -0.6, 0.6, &wd);
  const auto bias = rand_t<W>(static_cast<size_t>(dim), rng, -0.3, 0.3, &bd);
  const auto states0 = rand_t<T>(static_cast<size_t>(kSlots) * dim * H, rng, -1.0, 1.0, &sd);
  const std::vector<T> out0(static_cast<size_t>(ntok) * dim, to_t<T>(7.0));

  Arena ar;
  kops::state::Conv1dFwdParams p{};
  p.x = ar.put(x);
  p.x_token_stride = dim;
  p.x_dim_stride = 1;
  p.weight = ar.put(w);
  p.weight_dim_stride = width;
  p.weight_width_stride = 1;
  p.bias = with_bias ? ar.put(bias) : nullptr;
  T* out = ar.put(out0);
  p.out = out;
  p.out_token_stride = dim;
  p.out_dim_stride = 1;
  p.query_start_loc = ar.put(qsl);
  p.cache_indices = ar.put(slots);
  p.has_initial_state = ar.put(has_init);
  T* states = ar.put(states0);
  p.conv_states = states;
  p.conv_state_slot_stride = static_cast<int64_t>(dim) * H;
  p.conv_state_dim_stride = H;
  p.conv_state_len_stride = 1;
  p.num_slots = kSlots;
  p.snapshot_slots = ar.put(snap);
  p.snapshot_interval = 64;
  p.max_snapshots = kSnap;
  p.batch_size = B;
  p.dim = dim;
  p.width = width;
  p.total_tokens = ntok;
  p.max_seqlen = 0;
  p.pad_slot_id = 7;
  p.act = act;
  p.weight_dtype = dtype_of<W>();
  KOPS_EXPECT_OK(kops::state::causal_conv1d_fwd(p, dtype_of<T>(), ctx), "causal_conv1d_fwd");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(out, out0.size());
  const auto got_states = kops_test::copy_out(states, states0.size());

  std::vector<double> want(out0.size());
  for (size_t i = 0; i < want.size(); ++i) want[i] = 7.0;
  std::vector<T> want_states = states0;
  for (int b = 0; b < B; ++b) {
    if (slots[b] == 7) continue;
    const size_t so = static_cast<size_t>(slots[b]) * dim * H;
    std::vector<double> hist(static_cast<size_t>(dim) * H, 0.0);
    std::vector<T> hist_t(hist.size(), to_t<T>(0.0));
    if (has_init[b]) {
      for (size_t i = 0; i < hist.size(); ++i) {
        hist[i] = sd[so + i];
        hist_t[i] = states0[so + i];
      }
    }
    std::vector<double> y(static_cast<size_t>(lens[b]) * dim);
    KOPS_EXPECT_OK(kops::ref::causal_conv1d_f64(xd.data() + static_cast<size_t>(qsl[b]) * dim, lens[b], dim,
                                                wd.data(), width, with_bias ? bd.data() : nullptr, act,
                                                hist.data(), y.data()),
                   "causal_conv1d_f64");
    std::copy(y.begin(), y.end(), want.begin() + static_cast<ptrdiff_t>(qsl[b]) * dim);
    // expected raw state after `count` tokens: entries count .. count + H - 1 of [history, x]
    const auto state_after = [&](int count, size_t dst) {
      for (int c = 0; c < dim; ++c) {
        for (int l = 0; l < H; ++l) {
          const int idx = count + l;
          want_states[dst + static_cast<size_t>(c) * H + l] =
              idx < H ? hist_t[static_cast<size_t>(c) * H + idx]
                      : x[static_cast<size_t>(qsl[b] + idx - H) * dim + c];
        }
      }
    };
    for (int j = 0; j < kSnap; ++j) {
      const int dst = snap[static_cast<size_t>(b) * kSnap + j];
      if (dst >= 0 && (j + 1) * 64 <= lens[b]) state_after((j + 1) * 64, static_cast<size_t>(dst) * dim * H);
    }
    state_after(lens[b], so);
  }
  compare_out(got, want, "conv1d_fwd out");
  size_t bad = 0;
  for (size_t i = 0; i < want_states.size(); ++i) bad += bits_equal(got_states[i], want_states[i]) ? 0 : 1;
  KOPS_TEST_CHECK(bad == 0);
}

template <typename T, typename W>
void test_conv1d_update(LaunchContext& ctx, int dim, int width, int steps, bool in_place) {
  Lcg rng(static_cast<uint64_t>(dim + width * 13 + steps * 101 + (in_place ? 5 : 0)));
  const int B = 3, kSlots = 4 + B * steps, H = width - 1;
  const std::vector<int32_t> slots = {2, 0, 3};  // slot 3 == pad_slot_id
  std::vector<int32_t> inter(static_cast<size_t>(B) * steps);
  for (size_t i = 0; i < inter.size(); ++i) inter[i] = 4 + static_cast<int32_t>(i);
  std::vector<double> xd, wd, sd;
  const auto x = rand_t<T>(static_cast<size_t>(B) * steps * dim, rng, -1.0, 1.0, &xd);
  const auto w = rand_t<W>(static_cast<size_t>(dim) * width, rng, -0.6, 0.6, &wd);
  const auto states0 = rand_t<T>(static_cast<size_t>(kSlots) * dim * H, rng, -1.0, 1.0, &sd);
  Arena ar;
  kops::state::Conv1dUpdateParams p{};
  T* xp = ar.put(x);
  T* out = in_place ? xp : ar.put(std::vector<T>(x.size(), to_t<T>(7.0)));
  T* states = ar.put(states0);
  p.x = xp;
  p.x_batch_stride = static_cast<int64_t>(steps) * dim;
  p.x_token_stride = dim;
  p.weight = ar.put(w);
  p.weight_dim_stride = width;
  p.weight_width_stride = 1;
  p.bias = nullptr;
  p.out = out;
  p.out_batch_stride = p.x_batch_stride;
  p.out_token_stride = dim;
  p.conv_states = states;
  p.conv_state_slot_stride = static_cast<int64_t>(dim) * H;
  p.conv_state_dim_stride = H;
  p.conv_state_len_stride = 1;
  p.num_slots = kSlots;
  p.cache_indices = ar.put(slots);
  p.intermediate_state_slots = steps > 1 ? ar.put(inter) : nullptr;
  p.batch_size = B;
  p.seqlen = steps;
  p.dim = dim;
  p.width = width;
  p.pad_slot_id = 3;
  p.act = ActFn::kSiLU;
  p.weight_dtype = dtype_of<W>();
  KOPS_EXPECT_OK(kops::state::causal_conv1d_update(p, dtype_of<T>(), ctx), "causal_conv1d_update");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(out, x.size());
  const auto got_states = kops_test::copy_out(states, states0.size());
  std::vector<double> want(x.size());
  for (size_t i = 0; i < want.size(); ++i) want[i] = in_place ? xd[i] : 7.0;
  std::vector<T> want_states = states0;
  for (int b = 0; b < B; ++b) {
    if (slots[b] == 3) continue;
    const size_t so = static_cast<size_t>(slots[b]) * dim * H;
    std::vector<double> hist(sd.begin() + static_cast<ptrdiff_t>(so), sd.begin() + static_cast<ptrdiff_t>(so + static_cast<size_t>(dim) * H));
    std::vector<double> y(static_cast<size_t>(steps) * dim);
    kops::ref::causal_conv1d_f64(xd.data() + static_cast<size_t>(b) * steps * dim, steps, dim, wd.data(), width,
                                 nullptr, ActFn::kSiLU, hist.data(), y.data());
    std::copy(y.begin(), y.end(), want.begin() + static_cast<ptrdiff_t>(b) * steps * dim);
    const auto state_after = [&](int count, size_t dst) {
      for (int c = 0; c < dim; ++c) {
        for (int l = 0; l < H; ++l) {
          const int idx = count + l;
          want_states[dst + static_cast<size_t>(c) * H + l] =
              idx < H ? states0[so + static_cast<size_t>(c) * H + idx]
                      : x[(static_cast<size_t>(b) * steps + idx - H) * dim + c];
        }
      }
    };
    if (steps > 1) {
      for (int i = 0; i < steps; ++i) state_after(i + 1, static_cast<size_t>(inter[static_cast<size_t>(b) * steps + i]) * dim * H);
    }
    state_after(steps, so);
  }
  compare_out(got, want, "conv1d_update out");
  size_t bad = 0;
  for (size_t i = 0; i < want_states.size(); ++i) bad += bits_equal(got_states[i], want_states[i]) ? 0 : 1;
  KOPS_TEST_CHECK(bad == 0);
}

void test_conv1d(LaunchContext& ctx) {
  for (int width = 2; width <= 4; ++width) {
    test_conv1d_fwd<__nv_bfloat16, __nv_bfloat16>(ctx, 1024, width, ActFn::kSiLU, true);  // 16-byte path
    test_conv1d_fwd<__nv_bfloat16, float>(ctx, 40, width, ActFn::kIdentity, false);       // scalar path
    test_conv1d_fwd<float, float>(ctx, 72, width, ActFn::kSiLU, true);
    test_conv1d_update<__nv_bfloat16, __nv_bfloat16>(ctx, 1024, width, 1, true);
    test_conv1d_update<__half, float>(ctx, 48, width, 3, false);
  }
  test_conv1d_update<float, float>(ctx, 96, 4, 4, true);
  // contracts
  kops::state::Conv1dFwdParams p{};
  p.batch_size = 1;
  p.dim = 8;
  p.width = 5;
  p.act = ActFn::kSiLU;
  KOPS_EXPECT_STATUS(kops::state::causal_conv1d_fwd(p, DType::kBFloat16, ctx), Status::kUnsupportedShape,
                     "conv1d width 5");
  p.width = 4;
  p.act = ActFn::kGeLUErf;
  KOPS_EXPECT_STATUS(kops::state::causal_conv1d_fwd(p, DType::kBFloat16, ctx), Status::kInvalidArgument,
                     "conv1d gelu");
  p.act = ActFn::kSiLU;
  KOPS_EXPECT_STATUS(kops::state::causal_conv1d_fwd(p, DType::kInt8, ctx), Status::kUnsupportedDtype,
                     "conv1d int8");
  KOPS_EXPECT_STATUS(kops::state::causal_conv1d_fwd(p, DType::kBFloat16, ctx), Status::kInvalidArgument,
                     "conv1d null x");
}

// ---- B10 hybrid consistency: speculative conv state == sequential decode ---------------------

void test_commit_hybrid(LaunchContext& ctx) {
  constexpr int B = 1, K = 4, dim = 32, width = 4, H = width - 1;
  constexpr int kSlots = 8;
  constexpr int kPrefix = 0, kLive = 1, kRef = 2, kSnap0 = 3, kSnap1 = 4, kSnap2 = 5,
                kSnap3 = 6, kSnap4 = 7;
  Lcg rng(4242u);
  std::vector<double> xd, wd, sd;
  const auto x = rand_t<float>(static_cast<size_t>(B) * K * dim, rng, -1.0, 1.0, &xd);
  const auto w = rand_t<float>(static_cast<size_t>(dim) * width, rng, -0.6, 0.6, &wd);
  const auto states0 = rand_t<float>(static_cast<size_t>(kSlots) * dim * H, rng, -1.0, 1.0, &sd);

  Arena ar;
  float* states = ar.put(states0);
  float* xp = ar.put(x);
  float* out = ar.put(std::vector<float>(x.size(), 0.0f));
  int64_t* ptrs = ar.put(std::vector<int64_t>{reinterpret_cast<int64_t>(states)});
  int64_t* slot_bytes = ar.put(std::vector<int64_t>{static_cast<int64_t>(dim) * H * sizeof(float)});
  int32_t* copy_pair = ar.put(std::vector<int32_t>{kPrefix, kLive});
  int32_t* copy_pair0 = ar.put(std::vector<int32_t>{kPrefix, kSnap0});
  int32_t* live = ar.put(std::vector<int32_t>{kLive});
  int32_t* ref = ar.put(std::vector<int32_t>{kRef});
  int32_t* inter = ar.put(std::vector<int32_t>{kSnap1, kSnap2, kSnap3, kSnap4});
  int32_t* table = ar.put(std::vector<int32_t>{kSnap0, kSnap1, kSnap2, kSnap3, kSnap4});
  int32_t* accepted = ar.put(std::vector<int32_t>{0});

  const Tensor tp = tensor1d(ptrs, DType::kInt64, 1);
  const Tensor tb = tensor1d(slot_bytes, DType::kInt64, 1);
  const Tensor tc = tensor2d(copy_pair, DType::kInt32, 1, 2);
  const Tensor tc0 = tensor2d(copy_pair0, DType::kInt32, 1, 2);
  const Tensor tl = tensor1d(live, DType::kInt32, 1);
  const Tensor ti = tensor2d(table, DType::kInt32, 1, 5);
  const Tensor tn = tensor1d(accepted, DType::kInt32, 1);

  kops::state::Conv1dUpdateParams p{};
  p.x = xp;
  p.x_batch_stride = static_cast<int64_t>(K) * dim;
  p.x_token_stride = dim;
  p.weight = ar.put(w);
  p.weight_dim_stride = width;
  p.weight_width_stride = 1;
  p.bias = nullptr;
  p.out = out;
  p.out_batch_stride = static_cast<int64_t>(K) * dim;
  p.out_token_stride = dim;
  p.conv_states = states;
  p.conv_state_slot_stride = static_cast<int64_t>(dim) * H;
  p.conv_state_dim_stride = H;
  p.conv_state_len_stride = 1;
  p.num_slots = kSlots;
  p.cache_indices = live;
  p.intermediate_state_slots = inter;
  p.batch_size = B;
  p.seqlen = K;
  p.dim = dim;
  p.width = width;
  p.pad_slot_id = 100;
  p.act = ActFn::kSiLU;
  p.weight_dtype = DType::kFloat32;

  // Snapshot slot 0 is the prefix state (state after zero accepted drafts).
  KOPS_EXPECT_OK(kops::state::state_copy(tp, tb, tc0, ctx), "hybrid copy prefix");
  for (int r : {0, 2, 4}) {
    KOPS_EXPECT_OK(kops::state::state_copy(tp, tb, tc, ctx), "hybrid reset live");
    KOPS_EXPECT_OK(kops::state::causal_conv1d_update(p, DType::kFloat32, ctx),
                   "hybrid draft update");
    kops_test::copy_in(accepted, std::vector<int32_t>{r});
    KOPS_EXPECT_OK(kops::state::state_commit_accepted(tp, tb, tl, ti, tn, ctx), "hybrid commit");

    // Reference: decode just the accepted prefix + drafts from the same state.
    int32_t reset_pair[2] = {kPrefix, kRef};
    kops_test::copy_in(copy_pair, std::vector<int32_t>(reset_pair, reset_pair + 2));
    KOPS_EXPECT_OK(kops::state::state_copy(tp, tb, tensor2d(copy_pair, DType::kInt32, 1, 2), ctx),
                   "hybrid reset ref");
    if (r > 0) {
      auto pr = p;
      pr.cache_indices = ref;
      pr.intermediate_state_slots = nullptr;
      pr.seqlen = r;
      KOPS_EXPECT_OK(kops::state::causal_conv1d_update(pr, DType::kFloat32, ctx),
                     "hybrid reference update");
    }
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = kops_test::copy_out(states + kLive * dim * H, dim * H);
    const auto want = kops_test::copy_out(states + kRef * dim * H, dim * H);
    KOPS_TEST_CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
  }
}

// ---- shared helpers of the recurrent tests ---------------------------------------------------

std::vector<float> rand_f(size_t n, Lcg& r, double lo, double hi, std::vector<double>* exact) {
  return rand_t<float>(n, r, lo, hi, exact);
}

std::vector<double> to_double(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

/// Capture `launch(ctx on a private stream)` into a CUDA graph, replay it once, wait. True when the
/// capture, the launcher Status and the replay all succeed.
template <typename F>
bool capture_replay(const LaunchContext& ctx, F&& launch) {
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return false;
  LaunchContext gc = ctx;
  gc.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  gc.strategy = kops::ExecutionStrategy::kGraph;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
  const Status st = launch(gc);
  ok = cudaStreamEndCapture(stream, &graph) == cudaSuccess && ok && st == Status::kSuccess;
  ok = ok && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess;
  ok = ok && cudaGraphLaunch(exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
  if (exec) cudaGraphExecDestroy(exec);
  if (graph) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  return ok;
}

template <typename T>
size_t count_bit_diffs(const std::vector<T>& a, const std::vector<T>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) n += bits_equal(a[i], b[i]) ? 0 : 1;
  return n + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

// ---- B9 row 2: selective_state_update (Mamba-1 / Mamba-2 decode) -----------------------------

template <typename T>
void test_ssm_update(LaunchContext& ctx, bool mamba2, int dstate, int steps) {
  Lcg rng(static_cast<uint64_t>(dstate * 3 + steps * 17 + (mamba2 ? 1 : 0)));
  const int B = 3, H = mamba2 ? 8 : 48, P = mamba2 ? 16 : 1, N = dstate, G = mamba2 ? 2 : 1;
  const int kSlots = 4 + B * steps;
  const std::vector<int32_t> slots = {1, 3, 0};  // slot 3 == pad_slot_id
  std::vector<int32_t> inter(static_cast<size_t>(B) * steps);
  for (size_t i = 0; i < inter.size(); ++i) inter[i] = 4 + static_cast<int32_t>(i);
  std::vector<double> xd, dtd, Bd, Cd, zd, Ad, Dd, bd, sd;
  const auto x = rand_t<T>(static_cast<size_t>(B) * steps * H * P, rng, -1.0, 1.0, &xd);
  const auto dt = rand_t<T>(static_cast<size_t>(B) * steps * H, rng, -3.0, 1.0, &dtd);
  const auto Bm = rand_t<T>(static_cast<size_t>(B) * steps * G * N, rng, -1.0, 1.0, &Bd);
  const auto Cm = rand_t<T>(static_cast<size_t>(B) * steps * G * N, rng, -1.0, 1.0, &Cd);
  const auto z = rand_t<T>(x.size(), rng, -2.0, 2.0, &zd);
  const auto A = rand_f(static_cast<size_t>(H) * (mamba2 ? 1 : N), rng, -2.0, -0.05, &Ad);
  const auto D = rand_f(static_cast<size_t>(H) * (mamba2 ? 1 : P), rng, -1.0, 1.0, &Dd);
  const auto bias = rand_f(static_cast<size_t>(H), rng, -0.5, 0.5, &bd);
  const auto st0 = rand_f(static_cast<size_t>(kSlots) * H * P * N, rng, -1.0, 1.0, &sd);
  Arena ar;
  kops::state::SSMStateUpdateParams p{};
  float* st = ar.put(st0);
  T* out = ar.put(std::vector<T>(x.size(), to_t<T>(7.0)));
  p.state = st;
  p.state_slot_stride = static_cast<int64_t>(H) * P * N;
  p.state_head_stride = static_cast<int64_t>(P) * N;
  p.state_p_stride = N;
  p.x = ar.put(x);
  p.x_batch_stride = static_cast<int64_t>(steps) * H * P;
  p.x_token_stride = static_cast<int64_t>(H) * P;
  p.x_head_stride = P;
  p.dt = ar.put(dt);
  p.dt_batch_stride = static_cast<int64_t>(steps) * H;
  p.dt_token_stride = H;
  p.dt_head_stride = 1;
  p.dt_bias = ar.put(bias);
  p.A = ar.put(A);
  p.A_head_stride = mamba2 ? 1 : N;
  p.a_per_state = !mamba2;
  p.B = ar.put(Bm);
  p.B_batch_stride = static_cast<int64_t>(steps) * G * N;
  p.B_token_stride = static_cast<int64_t>(G) * N;
  p.B_group_stride = N;
  p.C = ar.put(Cm);
  p.C_batch_stride = p.B_batch_stride;
  p.C_token_stride = p.B_token_stride;
  p.C_group_stride = N;
  p.D = ar.put(D);
  p.D_head_stride = mamba2 ? 1 : P;
  p.D_p_stride = 0;
  p.z = ar.put(z);
  p.z_batch_stride = p.x_batch_stride;
  p.z_token_stride = p.x_token_stride;
  p.z_head_stride = P;
  p.out = out;
  p.out_batch_stride = p.x_batch_stride;
  p.out_token_stride = p.x_token_stride;
  p.out_head_stride = P;
  p.state_batch_indices = ar.put(slots);
  p.intermediate_state_slots = steps > 1 ? ar.put(inter) : nullptr;
  p.dt_softplus = true;
  p.has_dt_limit = mamba2;
  p.dt_min = 0.01f;
  p.dt_max = 0.9f;
  p.num_slots = kSlots;
  p.num_steps = steps;
  p.batch_size = B;
  p.nheads = H;
  p.headdim = P;
  p.dstate = N;
  p.ngroups = G;
  p.pad_slot_id = 3;
  KOPS_EXPECT_OK(kops::state::selective_state_update(p, dtype_of<T>(), ctx), "selective_state_update");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(out, x.size());
  const auto got_st = kops_test::copy_out(st, st0.size());
  std::vector<double> want(x.size(), 7.0), want_st = sd;
  const size_t slot_elems = static_cast<size_t>(H) * P * N;
  for (int b = 0; b < B; ++b) {
    if (slots[b] == 3) continue;
    std::vector<double> s(sd.begin() + static_cast<ptrdiff_t>(slot_elems * slots[b]),
                          sd.begin() + static_cast<ptrdiff_t>(slot_elems * (slots[b] + 1)));
    for (int t = 0; t < steps; ++t) {  // one oracle step per token: intermediate states
      kops::ref::SsmRefArgs r;
      const size_t tok = static_cast<size_t>(b) * steps + t;
      r.x = xd.data() + tok * H * P;
      r.dt = dtd.data() + tok * H;
      const std::vector<double> bias_d = to_double(bias);
      r.dt_bias = bias_d.data();
      r.A = Ad.data();
      r.B = Bd.data() + tok * G * N;
      r.C = Cd.data() + tok * G * N;
      r.D = Dd.data();
      r.z = zd.data() + tok * H * P;
      r.len = 1;
      r.nheads = H;
      r.headdim = P;
      r.dstate = N;
      r.ngroups = G;
      r.a_per_state = !mamba2;
      r.d_per_p = false;
      r.dt_softplus = true;
      r.has_dt_limit = mamba2;
      r.dt_min = 0.01f;
      r.dt_max = 0.9f;
      r.state = s.data();
      r.y = want.data() + tok * H * P;
      KOPS_EXPECT_OK(kops::ref::ssm_recurrence_f64(r), "ssm_recurrence_f64");
      if (steps > 1) {
        std::copy(s.begin(), s.end(), want_st.begin() + static_cast<ptrdiff_t>(slot_elems * inter[tok]));
      }
    }
    std::copy(s.begin(), s.end(), want_st.begin() + static_cast<ptrdiff_t>(slot_elems * slots[b]));
  }
  compare_out(got, want, "ssm_update out");
  kops_test::compare(got_st, want_st, 2e-6, 5e-5, "ssm_update state");
}

// ---- B9 row 4: selective_scan_fwd (Mamba-1 varlen prefill) ------------------------------------

template <typename T>
void test_selective_scan(LaunchContext& ctx, int dstate, int ngroups, bool token_major) {
  Lcg rng(static_cast<uint64_t>(dstate * 5 + ngroups * 11 + (token_major ? 3 : 0)));
  const int dim = 64, N = dstate, G = ngroups, kSlots = 6;
  const std::vector<int32_t> lens = {1, 300, 0, 130, 77};
  const std::vector<int32_t> slots = {0, 2, 3, 5, 4};  // slot 5 == pad
  const std::vector<uint8_t> has_init = {1, 1, 0, 0, 1};
  const int B = static_cast<int>(lens.size());
  std::vector<int32_t> qsl(B + 1, 0);
  for (int b = 0; b < B; ++b) qsl[b + 1] = qsl[b] + lens[b];
  const int ntok = qsl[B];
  // logical [dim, T] / [G, N, T]; storage token-major ([T, dim], [T, G, N]) or dim-major
  std::vector<double> ud, dd, zd, Bd, Cd, Ad, Dd, bd, sd;
  const auto u = rand_t<T>(static_cast<size_t>(dim) * ntok, rng, -1.0, 1.0, &ud);
  const auto delta = rand_t<T>(static_cast<size_t>(dim) * ntok, rng, -4.0, 0.5, &dd);
  const auto z = rand_t<T>(static_cast<size_t>(dim) * ntok, rng, -2.0, 2.0, &zd);
  const auto Bm = rand_t<T>(static_cast<size_t>(G) * N * ntok, rng, -1.0, 1.0, &Bd);
  const auto Cm = rand_t<T>(static_cast<size_t>(G) * N * ntok, rng, -1.0, 1.0, &Cd);
  const auto A = rand_f(static_cast<size_t>(dim) * N, rng, -3.0, -0.1, &Ad);
  const auto D = rand_f(static_cast<size_t>(dim), rng, -1.0, 1.0, &Dd);
  const auto bias = rand_f(static_cast<size_t>(dim), rng, -0.5, 0.5, &bd);
  const auto st0 = rand_f(static_cast<size_t>(kSlots) * dim * N, rng, -1.0, 1.0, &sd);
  // element (c, t) of u/delta/z is at c * cs + t * ts; (g, n, t) of B/C at g * gs + n * ns + t * ts2
  const int64_t cs = token_major ? 1 : ntok, ts = token_major ? dim : 1;
  const int64_t gs = token_major ? N : static_cast<int64_t>(N) * ntok, ns = token_major ? 1 : ntok,
                ts2 = token_major ? static_cast<int64_t>(G) * N : 1;
  Arena ar;
  T* out = ar.put(std::vector<T>(u.size(), to_t<T>(7.0)));
  float* st = ar.put(st0);
  kops::state::SelectiveScanParams p{};
  p.u = ar.put(u);
  p.u_dim_stride = cs;
  p.u_token_stride = ts;
  p.delta = ar.put(delta);
  p.delta_dim_stride = cs;
  p.delta_token_stride = ts;
  p.A = ar.put(A);
  p.A_dim_stride = N;
  p.B = ar.put(Bm);
  p.B_group_stride = gs;
  p.B_state_stride = ns;
  p.B_token_stride = ts2;
  p.C = ar.put(Cm);
  p.C_group_stride = gs;
  p.C_state_stride = ns;
  p.C_token_stride = ts2;
  p.D = ar.put(D);
  p.z = ar.put(z);
  p.z_dim_stride = cs;
  p.z_token_stride = ts;
  p.delta_bias = ar.put(bias);
  p.delta_softplus = true;
  p.out = out;
  p.out_dim_stride = cs;
  p.out_token_stride = ts;
  p.ssm_states = st;
  p.ssm_state_slot_stride = static_cast<int64_t>(dim) * N;
  p.ssm_state_dim_stride = N;
  p.query_start_loc = ar.put(qsl);
  p.cache_indices = ar.put(slots);
  p.has_initial_state = ar.put(has_init);
  p.num_slots = kSlots;
  p.batch_size = B;
  p.dim = dim;
  p.dstate = N;
  p.ngroups = G;
  p.total_tokens = ntok;
  p.pad_slot_id = 5;
  KOPS_EXPECT_OK(kops::state::selective_scan_fwd(p, dtype_of<T>(), ctx), "selective_scan_fwd");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(out, u.size());
  const auto got_st = kops_test::copy_out(st, st0.size());
  std::vector<double> want(u.size(), 7.0), want_st = sd;
  const std::vector<double> bias_d = to_double(bias);
  for (int b = 0; b < B; ++b) {
    if (slots[b] == 5) continue;
    const int L = lens[b], t0 = qsl[b];
    std::vector<double> s(static_cast<size_t>(dim) * N, 0.0);
    if (has_init[b]) std::copy(sd.begin() + static_cast<ptrdiff_t>(dim) * N * slots[b],
                               sd.begin() + static_cast<ptrdiff_t>(dim) * N * (slots[b] + 1), s.begin());
    // oracle layout: x [L, H=dim, P=1], dt [L, dim], B/C [L, G, N]
    std::vector<double> x(static_cast<size_t>(L) * dim), dt(x.size()), zz(x.size()), Bo(static_cast<size_t>(L) * G * N),
        Co(Bo.size()), y(x.size());
    for (int t = 0; t < L; ++t) {
      for (int c = 0; c < dim; ++c) {
        const size_t at = static_cast<size_t>(c * cs + (t0 + t) * ts);
        x[static_cast<size_t>(t) * dim + c] = ud[at];
        dt[static_cast<size_t>(t) * dim + c] = dd[at];
        zz[static_cast<size_t>(t) * dim + c] = zd[at];
      }
      for (int g = 0; g < G; ++g) {
        for (int n = 0; n < N; ++n) {
          const size_t at = static_cast<size_t>(g * gs + n * ns + (t0 + t) * ts2);
          Bo[(static_cast<size_t>(t) * G + g) * N + n] = Bd[at];
          Co[(static_cast<size_t>(t) * G + g) * N + n] = Cd[at];
        }
      }
    }
    kops::ref::SsmRefArgs r;
    r.x = x.data();
    r.dt = dt.data();
    r.A = Ad.data();
    r.B = Bo.data();
    r.C = Co.data();
    r.D = Dd.data();
    r.z = zz.data();
    r.len = L;
    r.nheads = dim;
    r.headdim = 1;
    r.dstate = N;
    r.ngroups = G;
    r.a_per_state = true;
    r.dt_softplus = true;
    r.state = s.data();
    r.y = y.data();
    // per-channel delta bias: fold it into dt (the oracle's dt_bias is per head = per channel here)
    r.dt_bias = bias_d.data();
    KOPS_EXPECT_OK(kops::ref::ssm_recurrence_f64(r), "ssm_recurrence_f64 (scan)");
    for (int t = 0; t < L; ++t) {
      for (int c = 0; c < dim; ++c) want[static_cast<size_t>(c * cs + (t0 + t) * ts)] = y[static_cast<size_t>(t) * dim + c];
    }
    std::copy(s.begin(), s.end(), want_st.begin() + static_cast<ptrdiff_t>(dim) * N * slots[b]);
  }
  compare_out(got, want, "selective_scan out");
  kops_test::compare(got_st, want_st, 5e-6, 1e-4, "selective_scan state");

  // resume at the 128-token tile boundary reproduces the one-call result bitwise
  if (!token_major) return;
  const std::vector<int32_t> one = {2};
  const std::vector<uint8_t> init1 = {1};
  float* st2 = ar.put(st0);
  T* out2 = ar.put(std::vector<T>(u.size(), to_t<T>(7.0)));
  kops::state::SelectiveScanParams q = p;
  q.batch_size = 1;
  q.cache_indices = ar.put(one);
  q.has_initial_state = ar.put(init1);
  q.ssm_states = st2;
  q.out = out2;
  q.query_start_loc = ar.put(std::vector<int32_t>{qsl[1], qsl[1] + 128});
  KOPS_EXPECT_OK(kops::state::selective_scan_fwd(q, dtype_of<T>(), ctx), "scan part 1");
  q.query_start_loc = ar.put(std::vector<int32_t>{qsl[1] + 128, qsl[2]});
  KOPS_EXPECT_OK(kops::state::selective_scan_fwd(q, dtype_of<T>(), ctx), "scan part 2");
  cudaDeviceSynchronize();
  const auto got2 = kops_test::copy_out(out2, u.size());
  const auto got_st2 = kops_test::copy_out(st2, st0.size());
  size_t diff = 0;
  for (int t = qsl[1]; t < qsl[2]; ++t) {
    for (int c = 0; c < dim; ++c) diff += bits_equal(got2[static_cast<size_t>(t) * dim + c], got[static_cast<size_t>(t) * dim + c]) ? 0 : 1;
  }
  for (int i = 0; i < dim * N; ++i) diff += bits_equal(got_st2[static_cast<size_t>(2 * dim * N + i)], got_st[static_cast<size_t>(2 * dim * N + i)]) ? 0 : 1;
  KOPS_TEST_CHECK(diff == 0);
}

// ---- B9 row 6: Mamba-2 SSD chunked scan ---------------------------------------------------------

template <typename T>
void test_ssd(LaunchContext& ctx, int chunk, int headdim, int dstate, int ngroups) {
  Lcg rng(static_cast<uint64_t>(chunk + headdim * 3 + dstate * 7 + ngroups));
  const int H = 4, P = headdim, N = dstate, G = ngroups, kSlots = 8;
  const std::vector<int32_t> lens = {1, chunk, 2 * chunk + 37, 0, 45};
  const std::vector<int32_t> slots = {0, 1, 2, 3, 7};  // slot 7 == pad
  const std::vector<uint8_t> has_init = {1, 0, 1, 1, 1};
  const int B = static_cast<int>(lens.size());
  std::vector<int32_t> cu(B + 1, 0), cuc(B + 1, 0), ccs(1, 0);
  for (int b = 0; b < B; ++b) {
    cu[b + 1] = cu[b] + lens[b];
    for (int s = 0; s < lens[b]; s += chunk) ccs.push_back(cu[b] + std::min(lens[b], s + chunk));
    cuc[b + 1] = static_cast<int32_t>(ccs.size()) - 1;
  }
  const int ntok = cu[B], NC = static_cast<int>(ccs.size()) - 1;
  std::vector<int32_t> snap(static_cast<size_t>(B) * 2, -1);
  snap[2 * 2 + 0] = 5;  // sequence 2: state after `chunk` and 2 * chunk tokens
  snap[2 * 2 + 1] = 6;
  std::vector<double> xd, dtd, Bd, Cd, zd, Ad, Dd, bd, sd;
  const auto x = rand_t<T>(static_cast<size_t>(ntok) * H * P, rng, -1.0, 1.0, &xd);
  const auto dt = rand_t<T>(static_cast<size_t>(ntok) * H, rng, -4.0, -1.0, &dtd);
  const auto Bm = rand_t<T>(static_cast<size_t>(ntok) * G * N, rng, -1.0, 1.0, &Bd);
  const auto Cm = rand_t<T>(static_cast<size_t>(ntok) * G * N, rng, -1.0, 1.0, &Cd);
  const auto z = rand_t<T>(x.size(), rng, -2.0, 2.0, &zd);
  const auto A = rand_f(H, rng, -2.0, -0.2, &Ad);
  const auto D = rand_f(H, rng, -1.0, 1.0, &Dd);
  const auto bias = rand_f(H, rng, -0.5, 0.5, &bd);
  const auto st0 = rand_f(static_cast<size_t>(kSlots) * H * P * N, rng, -0.5, 0.5, &sd);
  Arena ar;
  T* out = ar.put(std::vector<T>(x.size(), to_t<T>(7.0)));
  float* st = ar.put(st0);
  kops::state::SSDParams p{};
  p.x = ar.put(x);
  p.x_token_stride = static_cast<int64_t>(H) * P;
  p.x_head_stride = P;
  p.dt = ar.put(dt);
  p.dt_token_stride = H;
  p.dt_head_stride = 1;
  p.dt_bias = ar.put(bias);
  p.A = ar.put(A);
  p.B = ar.put(Bm);
  p.B_token_stride = static_cast<int64_t>(G) * N;
  p.B_group_stride = N;
  p.C = ar.put(Cm);
  p.C_token_stride = p.B_token_stride;
  p.C_group_stride = N;
  p.D = ar.put(D);
  p.D_head_stride = 1;
  p.D_p_stride = 0;
  p.z = ar.put(z);
  p.z_token_stride = p.x_token_stride;
  p.z_head_stride = P;
  p.out = out;
  p.out_token_stride = p.x_token_stride;
  p.out_head_stride = P;
  p.dt_softplus = true;
  p.has_dt_limit = true;
  p.dt_min = 0.0f;
  p.dt_max = 0.5f;
  p.cu_seqlens = ar.put(cu);
  p.cu_chunks = ar.put(cuc);
  p.cu_chunk_seqlens = ar.put(ccs);
  p.ssm_states = st;
  p.ssm_state_slot_stride = static_cast<int64_t>(H) * P * N;
  p.state_indices = ar.put(slots);
  p.has_initial_state = ar.put(has_init);
  p.snapshot_slots = ar.put(snap);
  p.snapshot_interval = chunk;
  p.max_snapshots = 2;
  p.num_slots = kSlots;
  p.batch_size = B;
  p.num_chunks = NC;
  p.chunk_size = chunk;
  p.nheads = H;
  p.headdim = P;
  p.dstate = N;
  p.ngroups = G;
  p.pad_slot_id = 7;
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::state::ssd_workspace_requirements(ntok, B, H, P, N, G, chunk, &req), "ssd_workspace");
  DevBuf<uint8_t> wsb(req.device_scratch_bytes);
  kops::Workspace ws{};
  ws.device_scratch = wsb.p;
  ws.device_scratch_bytes = req.device_scratch_bytes;
  KOPS_EXPECT_STATUS(kops::state::ssd_chunked_scan(p, dtype_of<T>(), kops::Workspace{}, ctx), Status::kWorkspaceTooSmall,
                     "ssd empty workspace");
  KOPS_EXPECT_OK(kops::state::ssd_chunked_scan(p, dtype_of<T>(), ws, ctx), "ssd_chunked_scan");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(out, x.size());
  const auto got_st = kops_test::copy_out(st, st0.size());
  std::vector<double> want(x.size(), 7.0), want_st = sd;
  const size_t se = static_cast<size_t>(H) * P * N;
  const std::vector<double> bias_d = to_double(bias);
  for (int b = 0; b < B; ++b) {
    const bool live = slots[b] != 7;  // an invalid slot: output from a zero state, no state written
    std::vector<double> s(se, 0.0);
    if (live && has_init[b]) std::copy(sd.begin() + static_cast<ptrdiff_t>(se * slots[b]), sd.begin() + static_cast<ptrdiff_t>(se * (slots[b] + 1)), s.begin());
    for (int t = 0; t < lens[b]; ++t) {
      const size_t tok = static_cast<size_t>(cu[b] + t);
      kops::ref::SsmRefArgs r;
      r.x = xd.data() + tok * H * P;
      r.dt = dtd.data() + tok * H;
      r.dt_bias = bias_d.data();
      r.A = Ad.data();
      r.B = Bd.data() + tok * G * N;
      r.C = Cd.data() + tok * G * N;
      r.D = Dd.data();
      r.z = zd.data() + tok * H * P;
      r.len = 1;
      r.nheads = H;
      r.headdim = P;
      r.dstate = N;
      r.ngroups = G;
      r.dt_softplus = true;
      r.has_dt_limit = true;
      r.dt_min = 0.0;
      r.dt_max = 0.5;
      r.state = s.data();
      r.y = want.data() + tok * H * P;
      kops::ref::ssm_recurrence_f64(r);
      for (int j = 0; j < 2; ++j) {
        const int dst = snap[static_cast<size_t>(b) * 2 + j];
        if (live && dst >= 0 && t + 1 == (j + 1) * chunk) std::copy(s.begin(), s.end(), want_st.begin() + static_cast<ptrdiff_t>(se * dst));
      }
    }
    if (live) std::copy(s.begin(), s.end(), want_st.begin() + static_cast<ptrdiff_t>(se * slots[b]));
  }
  compare_out(got, want, "ssd out");
  kops_test::compare(got_st, want_st, 5e-6, 2e-4, "ssd state");
  if (chunk == 128) {  // graph replay from the same initial pool is bitwise the eager call
    kops_test::copy_in(st, st0);
    kops_test::copy_in(out, std::vector<T>(x.size(), to_t<T>(7.0)));
    KOPS_TEST_CHECK(capture_replay(ctx, [&](const LaunchContext& g) { return kops::state::ssd_chunked_scan(p, dtype_of<T>(), ws, g); }));
    KOPS_TEST_CHECK(count_bit_diffs(kops_test::copy_out(out, x.size()), got) == 0);
    KOPS_TEST_CHECK(count_bit_diffs(kops_test::copy_out(st, st0.size()), got_st) == 0);
  }
}

void test_ssm(LaunchContext& ctx) {
  for (int n : {16, 64, 128}) {
    test_ssm_update<__nv_bfloat16>(ctx, true, n, 1);
    test_ssm_update<float>(ctx, true, n, 3);
  }
  test_ssm_update<__half>(ctx, false, 16, 2);
  test_ssm_update<float>(ctx, false, 8, 1);
  test_ssm_update<__nv_bfloat16>(ctx, true, 256, 2);
  for (int n : {8, 16, 32}) {
    test_selective_scan<float>(ctx, n, 1, true);
    test_selective_scan<__nv_bfloat16>(ctx, n, 2, false);
  }
  test_selective_scan<__half>(ctx, 16, 4, true);
  test_ssd<float>(ctx, 64, 64, 64, 1);
  test_ssd<__nv_bfloat16>(ctx, 128, 64, 128, 2);
  test_ssd<float>(ctx, 256, 32, 16, 2);
  test_ssd<__half>(ctx, 64, 80, 96, 1);
  kops::state::SSMStateUpdateParams bad{};
  bad.batch_size = 1;
  bad.nheads = 4;
  bad.headdim = 4;
  bad.ngroups = 1;
  bad.dstate = 48;
  KOPS_EXPECT_STATUS(kops::state::selective_state_update(bad, DType::kBFloat16, ctx), Status::kUnsupportedShape,
                     "ssm dstate 48");
  kops::state::SelectiveScanParams bs{};
  bs.batch_size = 1;
  bs.dim = 24;
  bs.ngroups = 2;
  bs.dstate = 16;
  KOPS_EXPECT_STATUS(kops::state::selective_scan_fwd(bs, DType::kBFloat16, ctx), Status::kUnsupportedShape,
                     "scan dim/ngroups tile");
}

// ---- B9 rows 2 and 5: decayed linear attention (decode, lightning prefill) --------------------

template <typename T>
void test_linear_attention(LaunchContext& ctx, int dk, int dv, bool prefill) {
  Lcg rng(static_cast<uint64_t>(dk * 7 + dv + (prefill ? 1000 : 0)));
  const int H = 3, kSlots = 8;
  const std::vector<int32_t> lens = prefill ? std::vector<int32_t>{1, 150, 64, 0, 20} : std::vector<int32_t>{1, 1, 1, 1, 1};
  const std::vector<int32_t> slots = {0, 1, 2, 3, 7};  // slot 7 == pad
  const std::vector<uint8_t> has_init = {1, 1, 0, 1, 1};
  const int B = static_cast<int>(lens.size());
  std::vector<int32_t> cu(B + 1, 0);
  for (int b = 0; b < B; ++b) cu[b + 1] = cu[b] + lens[b];
  const int ntok = cu[B];
  std::vector<int32_t> snap(static_cast<size_t>(B) * 2, -1);
  snap[1 * 2 + 0] = 4;  // sequence 1: after 64 and 128 tokens
  snap[1 * 2 + 1] = 5;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dk));
  std::vector<double> qd, kd, vd, sl, sd;
  const auto q = rand_t<T>(static_cast<size_t>(ntok) * H * dk, rng, -1.0, 1.0, &qd);
  const auto k = rand_t<T>(q.size(), rng, -1.0, 1.0, &kd);
  const auto v = rand_t<T>(static_cast<size_t>(ntok) * H * dv, rng, -1.0, 1.0, &vd);
  const auto slope = rand_f(H, rng, 0.0, 0.3, &sl);
  const auto st0 = rand_f(static_cast<size_t>(kSlots) * H * dk * dv, rng, -0.5, 0.5, &sd);
  Arena ar;
  T* o = ar.put(std::vector<T>(v.size(), to_t<T>(7.0)));
  float* st = ar.put(st0);
  kops::state::LinearAttentionParams p{};
  p.q = ar.put(q);
  p.k = ar.put(k);
  p.v = ar.put(v);
  p.q_token_stride = static_cast<int64_t>(H) * dk;
  p.q_head_stride = dk;
  p.k_token_stride = p.q_token_stride;
  p.k_head_stride = dk;
  p.v_token_stride = static_cast<int64_t>(H) * dv;
  p.v_head_stride = dv;
  p.slope = ar.put(slope);
  p.scale = static_cast<float>(scale);
  p.state_pool = st;
  p.state_slot_stride = static_cast<int64_t>(H) * dk * dv;
  p.state_indices = ar.put(slots);
  p.o = o;
  p.o_token_stride = p.v_token_stride;
  p.o_head_stride = dv;
  p.cu_seqlens = ar.put(cu);
  p.has_initial_state = prefill ? ar.put(has_init) : nullptr;
  p.snapshot_slots = prefill ? ar.put(snap) : nullptr;
  p.snapshot_interval = prefill ? 64 : 0;
  p.max_snapshots = prefill ? 2 : 0;
  p.num_slots = kSlots;
  p.total_tokens = ntok;
  p.batch_size = B;
  p.num_heads = H;
  p.head_dim_k = dk;
  p.head_dim_v = dv;
  p.pad_slot_id = 7;
  if (prefill) KOPS_EXPECT_OK(kops::state::lightning_attention_prefill(p, dtype_of<T>(), ctx), "lightning_attention_prefill");
  else KOPS_EXPECT_OK(kops::state::linear_attention_decode(p, dtype_of<T>(), ctx), "linear_attention_decode");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(o, v.size());
  const auto got_st = kops_test::copy_out(st, st0.size());
  std::vector<double> want(v.size(), 7.0), want_st = sd;
  const size_t se = static_cast<size_t>(H) * dk * dv;
  for (int b = 0; b < B; ++b) {
    if (slots[b] == 7) continue;
    std::vector<double> s(se, 0.0);
    if (!prefill || has_init[b]) std::copy(sd.begin() + static_cast<ptrdiff_t>(se * slots[b]), sd.begin() + static_cast<ptrdiff_t>(se * (slots[b] + 1)), s.begin());
    for (int t = 0; t < lens[b]; ++t) {
      const size_t tok = static_cast<size_t>(cu[b] + t);
      kops::ref::linear_attention_f64(qd.data() + tok * H * dk, kd.data() + tok * H * dk, vd.data() + tok * H * dv, sl.data(),
                                      scale, 1, H, dk, dv, s.data(), want.data() + tok * H * dv);
      for (int j = 0; j < 2 && prefill; ++j) {
        const int dst = snap[static_cast<size_t>(b) * 2 + j];
        if (dst >= 0 && t + 1 == (j + 1) * 64) std::copy(s.begin(), s.end(), want_st.begin() + static_cast<ptrdiff_t>(se * dst));
      }
    }
    std::copy(s.begin(), s.end(), want_st.begin() + static_cast<ptrdiff_t>(se * slots[b]));
  }
  compare_out(got, want, prefill ? "lightning out" : "linear decode out");
  kops_test::compare(got_st, want_st, 1e-5, 2e-4, prefill ? "lightning state" : "linear decode state");
  if (prefill && std::is_same<T, __nv_bfloat16>::value) {  // graph replay of the persistent body is bitwise
    kops_test::copy_in(st, st0);
    kops_test::copy_in(o, std::vector<T>(v.size(), to_t<T>(7.0)));
    KOPS_TEST_CHECK(capture_replay(ctx, [&](const LaunchContext& g) {
      return kops::state::lightning_attention_prefill(p, dtype_of<T>(), g);
    }));
    KOPS_TEST_CHECK(count_bit_diffs(kops_test::copy_out(o, v.size()), got) == 0);
    KOPS_TEST_CHECK(count_bit_diffs(kops_test::copy_out(st, st0.size()), got_st) == 0);
  }
}

// ---- B9 rows 3 and 7: gated delta rule ----------------------------------------------------------

/// One delta-rule case: inputs, slot tables and the fp64 oracle results.
template <typename T>
struct DeltaCase {
  int B = 0, Hk = 0, Hv = 0, Dk = 0, Dv = 0, ntok = 0, kSlots = 0;
  bool kda = false, transposed = false;
  std::vector<int32_t> lens, cu, slots, snap, inter;
  std::vector<uint8_t> has_init;
  std::vector<T> q, k, v, o0;
  std::vector<float> g, beta, st0;
  std::vector<double> qd, kd, vd, gd, bd, sd;
  std::vector<double> want, want_st;
  int snap_every = 0, max_snap = 2, max_inter = 0;

  DeltaCase(uint64_t seed, std::vector<int32_t> l, int hk, int hv, int dk, int dv, bool per_channel, bool trans,
            int snapshot_interval, int intermediate)
      : B(static_cast<int>(l.size())), Hk(hk), Hv(hv), Dk(dk), Dv(dv), kda(per_channel), transposed(trans), lens(std::move(l)),
        snap_every(snapshot_interval), max_inter(intermediate) {
    Lcg rng(seed);
    kSlots = 2 * B + 2 + B * max_inter;
    cu.assign(static_cast<size_t>(B) + 1, 0);
    for (int b = 0; b < B; ++b) cu[b + 1] = cu[b] + lens[b];
    ntok = cu[B];
    for (int b = 0; b < B; ++b) slots.push_back(b == B - 1 ? kSlots - 1 : b);  // last sequence: pad slot
    for (int b = 0; b < B; ++b) has_init.push_back(b % 3 == 1 ? 0 : 1);
    snap.assign(static_cast<size_t>(B) * max_snap, -1);
    for (int b = 0; b < B; ++b)
      for (int j = 0; j < max_snap; ++j) snap[static_cast<size_t>(b) * max_snap + j] = B + b * max_snap + j < 2 * B + 2 ? B + b * max_snap + j : -1;
    for (int b = 0; b < B; ++b)
      for (int i = 0; i < max_inter; ++i) inter.push_back(2 * B + 2 + b * max_inter + i);
    q = rand_t<T>(static_cast<size_t>(ntok) * Hk * Dk, rng, -1.0, 1.0, &qd);
    k = rand_t<T>(q.size(), rng, -1.0, 1.0, &kd);
    v = rand_t<T>(static_cast<size_t>(ntok) * Hv * Dv, rng, -1.0, 1.0, &vd);
    g = rand_f(static_cast<size_t>(ntok) * Hv * (kda ? Dk : 1), rng, kda ? -0.4 : -0.6, -0.002, &gd);
    beta = rand_f(static_cast<size_t>(ntok) * Hv, rng, 0.05, 0.95, &bd);
    st0 = rand_f(static_cast<size_t>(kSlots) * Hv * Dk * Dv, rng, -0.3, 0.3, &sd);
    o0.assign(v.size(), to_t<T>(7.0));
  }
  size_t se() const { return static_cast<size_t>(Hv) * Dk * Dv; }
  /// slot element (h, i, j) of the [Hv, Dk, Dv] oracle state in the pool layout.
  size_t pool_at(int slot, size_t e) const {
    const size_t h = e / (static_cast<size_t>(Dk) * Dv), i = (e / Dv) % Dk, j = e % Dv;
    return static_cast<size_t>(slot) * se() + h * Dk * Dv + (transposed ? j * Dk + i : i * Dv + j);
  }
  void oracle(double scale, bool snapshots, bool intermediates) {
    want.assign(v.size(), 7.0);
    want_st = sd;
    for (int b = 0; b < B; ++b) {
      if (slots[b] == kSlots - 1) continue;
      std::vector<double> s(se(), 0.0);
      if (has_init[b])
        for (size_t e = 0; e < se(); ++e) s[e] = sd[pool_at(slots[b], e)];
      const auto put = [&](int slot) {
        for (size_t e = 0; e < se(); ++e) want_st[pool_at(slot, e)] = s[e];
      };
      for (int t = 0; t < lens[b]; ++t) {
        const size_t tok = static_cast<size_t>(cu[b] + t);
        kops::ref::DeltaRuleRefArgs r;
        r.q = qd.data() + tok * Hk * Dk;
        r.k = kd.data() + tok * Hk * Dk;
        r.v = vd.data() + tok * Hv * Dv;
        r.g = gd.data() + tok * Hv * (kda ? Dk : 1);
        r.beta = bd.data() + tok * Hv;
        r.len = 1;
        r.num_k_heads = Hk;
        r.num_v_heads = Hv;
        r.head_dim_k = Dk;
        r.head_dim_v = Dv;
        r.per_channel = kda;
        r.use_qk_l2norm = true;
        r.scale = scale;
        r.state = s.data();
        r.o = want.data() + tok * Hv * Dv;
        kops::ref::delta_rule_f64(r);
        if (snapshots && snap_every > 0 && (t + 1) % snap_every == 0) {
          const int j = (t + 1) / snap_every - 1;
          if (j < max_snap && snap[static_cast<size_t>(b) * max_snap + j] >= 0) put(snap[static_cast<size_t>(b) * max_snap + j]);
        }
        if (intermediates && t < max_inter) put(inter[static_cast<size_t>(b) * max_inter + t]);
      }
      put(slots[b]);
    }
  }
  void fill(kops::state::DeltaRuleParams& p, Arena& ar, T** o, float** st, double scale, bool snapshots, bool intermediates) {
    p = kops::state::DeltaRuleParams{};
    p.q = ar.put(q);
    p.k = ar.put(k);
    p.v = ar.put(v);
    p.q_token_stride = static_cast<int64_t>(Hk) * Dk;
    p.q_head_stride = Dk;
    p.k_token_stride = p.q_token_stride;
    p.k_head_stride = Dk;
    p.v_token_stride = static_cast<int64_t>(Hv) * Dv;
    p.v_head_stride = Dv;
    p.g = ar.put(g);
    p.beta = ar.put(beta);
    p.scale = static_cast<float>(scale);
    p.use_qk_l2norm = true;
    p.cu_seqlens = ar.put(cu);
    p.state_indices = ar.put(slots);
    p.has_initial_state = ar.put(has_init);
    *st = ar.put(st0);
    p.state_pool = *st;
    p.state_slot_stride = static_cast<int64_t>(se());
    p.state_transposed = transposed;
    if (snapshots) {
      p.snapshot_slots = ar.put(snap);
      p.snapshot_interval = snap_every;
      p.max_snapshots = max_snap;
    }
    if (intermediates) {
      p.intermediate_state_slots = ar.put(inter);
      p.max_intermediate = max_inter;
    }
    *o = ar.put(o0);
    p.o = *o;
    p.o_token_stride = p.v_token_stride;
    p.o_head_stride = Dv;
    p.num_slots = kSlots;
    p.total_tokens = ntok;
    p.batch_size = B;
    p.num_k_heads = Hk;
    p.num_v_heads = Hv;
    p.head_dim_k = Dk;
    p.head_dim_v = Dv;
    p.decay = kda ? kops::state::DecayGranularity::kPerChannel : kops::state::DecayGranularity::kPerHead;
    p.pad_slot_id = kSlots - 1;
  }
};

template <typename T>
void test_delta_recurrent(LaunchContext& ctx, int dk, int dv, bool kda, bool transposed) {
  DeltaCase<T> c(static_cast<uint64_t>(dk * 13 + dv + (kda ? 7 : 0) + (transposed ? 3 : 0)), {1, 3, 37, 9, 5}, 2, 4, dk,
                 dv, kda, transposed, 8, 3);
  const double scale = 1.0 / std::sqrt(static_cast<double>(dk));
  c.oracle(scale, true, true);
  Arena ar;
  kops::state::DeltaRuleParams p;
  T* o = nullptr;
  float* st = nullptr;
  c.fill(p, ar, &o, &st, scale, true, true);
  KOPS_EXPECT_OK(kops::state::delta_rule_decode(p, dtype_of<T>(), ctx), "delta_rule_decode");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(o, c.o0.size());
  const auto got_st = kops_test::copy_out(st, c.st0.size());
  compare_out(got, c.want, kda ? "kda recurrent out" : "gdn recurrent out");
  kops_test::compare(got_st, c.want_st, 1e-6, 2e-5, kda ? "kda recurrent state" : "gdn recurrent state");
}

/// Chunked prefill vs the oracle, snapshots at chunk multiples, and chunked <-> recurrent parity.
template <typename T>
void test_delta_chunked(LaunchContext& ctx, int dk, int dv, bool kda, kops::state::ChunkMath math, bool transposed) {
  DeltaCase<T> c(static_cast<uint64_t>(dk * 3 + dv * 5 + (kda ? 11 : 0) + static_cast<int>(math) * 101), {1, 64, 65, 129, 200, 7},
                 2, 4, dk, dv, kda, transposed, 64, 0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(dk));
  c.oracle(scale, true, false);
  Arena ar;
  kops::state::DeltaRuleParams p;
  T* o = nullptr;
  float* st = nullptr;
  c.fill(p, ar, &o, &st, scale, true, false);
  p.math = math;
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::state::delta_rule_workspace_requirements(c.ntok, c.B, c.Hv, dk, dv, 64, p.decay, &req), "delta ws");
  DevBuf<uint8_t> wsb(req.device_scratch_bytes);
  kops::Workspace ws{};
  ws.device_scratch = wsb.p;
  ws.device_scratch_bytes = req.device_scratch_bytes;
  KOPS_EXPECT_OK(kops::state::delta_rule_prefill(p, dtype_of<T>(), ws, ctx), "delta_rule_prefill");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(o, c.o0.size());
  const auto got_st = kops_test::copy_out(st, c.st0.size());
  const bool tc = math == kops::state::ChunkMath::kTensorCore;
  // tensor cores: every product's operands rounded to bf16 (measured ~5e-3 of the output scale)
  const double mo = out_scale(c.want), ms = out_scale(c.want_st);
  const char* name = kda ? (tc ? "kda chunk tc out" : "kda chunk out") : (tc ? "gdn chunk tc out" : "gdn chunk out");
  if (tc) kops_test::compare(got, c.want, 7e-3 * mo, 3e-2, name);
  else compare_out(got, c.want, name);
  kops_test::compare(got_st, c.want_st, tc ? 4e-3 * ms : 3e-6, tc ? 1e-2 : 1e-4, kda ? "kda chunk state" : "gdn chunk state");
  if (dk == 128) {  // graph replay from the same initial pool is bitwise the eager call
    kops_test::copy_in(st, c.st0);
    kops_test::copy_in(o, c.o0);
    KOPS_TEST_CHECK(capture_replay(ctx, [&](const LaunchContext& g) { return kops::state::delta_rule_prefill(p, dtype_of<T>(), ws, g); }));
    KOPS_TEST_CHECK(count_bit_diffs(kops_test::copy_out(o, c.o0.size()), got) == 0);
    KOPS_TEST_CHECK(count_bit_diffs(kops_test::copy_out(st, c.st0.size()), got_st) == 0);
  }
  // the recurrent kernel over the same sequences (no snapshots: its slots would be the same anyway)
  T* o2 = nullptr;
  float* st2 = nullptr;
  kops::state::DeltaRuleParams r;
  c.fill(r, ar, &o2, &st2, scale, true, false);
  KOPS_EXPECT_OK(kops::state::delta_rule_decode(r, dtype_of<T>(), ctx), "delta_rule_decode (parity)");
  cudaDeviceSynchronize();
  const auto rec = kops_test::copy_out(o2, c.o0.size());
  std::vector<double> rec_d(rec.size());
  for (size_t i = 0; i < rec.size(); ++i) rec_d[i] = to_d(rec[i]);
  if (tc) kops_test::compare(got, rec_d, 7e-3 * mo, 3e-2, "chunked vs recurrent out");
  else compare_out(got, rec_d, "chunked vs recurrent out", 2.0);
}

template <typename T>
void test_delta_gating(LaunchContext& ctx, bool kda, float lower_bound) {
  Lcg rng(kda ? 91u : 92u);
  const int ntok = 37, Hv = 6, Dk = kda ? 32 : 1;
  std::vector<double> ad, bd, Ad, dbd;
  const auto a = rand_t<T>(static_cast<size_t>(ntok) * Hv * Dk, rng, -30.0, 30.0, &ad);
  const auto b = rand_t<T>(static_cast<size_t>(ntok) * Hv, rng, -6.0, 6.0, &bd);
  const auto A_log = rand_f(Hv, rng, -1.0, 1.5, &Ad);
  const auto dtb = rand_f(static_cast<size_t>(Hv) * Dk, rng, -1.0, 1.0, &dbd);
  Arena ar;
  float* g = ar.put(std::vector<float>(a.size(), 0.0f));
  float* beta = ar.put(std::vector<float>(b.size(), 0.0f));
  kops::state::DeltaGatingParams p{};
  p.g = g;
  p.beta = beta;
  p.a = ar.put(a);
  p.a_token_stride = static_cast<int64_t>(Hv) * Dk;
  p.a_head_stride = Dk;
  p.b = ar.put(b);
  p.b_token_stride = Hv;
  p.A_log = ar.put(A_log);
  p.dt_bias = ar.put(dtb);
  p.lower_bound = lower_bound;
  p.decay = kda ? kops::state::DecayGranularity::kPerChannel : kops::state::DecayGranularity::kPerHead;
  p.total_tokens = ntok;
  p.num_v_heads = Hv;
  p.head_dim_k = Dk;
  KOPS_EXPECT_OK(kops::state::delta_rule_gating(p, dtype_of<T>(), ctx), "delta_rule_gating");
  cudaDeviceSynchronize();
  std::vector<double> wg(a.size()), wb(b.size());
  kops::ref::delta_rule_gating_f64(ad.data(), bd.data(), Ad.data(), dbd.data(), lower_bound, ntok, Hv, Dk, wg.data(), wb.data());
  kops_test::compare(kops_test::copy_out(g, a.size()), wg, 1e-5, 2e-5, "gating g");
  kops_test::compare(kops_test::copy_out(beta, b.size()), wb, 1e-6, 1e-5, "gating beta");
}

/// fused_short_conv_delta_rule_decode vs the chain conv -> gating -> rule -> gated RMSNorm (fp64).
template <typename T>
void test_fused_delta(LaunchContext& ctx, int Hk, int Hv, int dk, int dv, int width, bool kda, ActFn gate_act, float eps) {
  Lcg rng(static_cast<uint64_t>(Hk * 3 + Hv * 5 + dk + dv * 7 + width + (kda ? 1 : 0)));
  const int B = 3, kSlots = 4, conv_dim = 2 * Hk * dk + Hv * dv, per = kda ? dk : 1;
  const std::vector<int32_t> slots = {2, 0, 3};  // slot 3 == pad
  const std::vector<uint8_t> has_init = {1, 0, 1};
  const double scale = 1.0 / std::sqrt(static_cast<double>(dk));
  std::vector<double> xd, wd, cbd, csd, gad, gbd, Ad, dbd, ogd, nwd, sd;
  const auto qkv = rand_t<T>(static_cast<size_t>(B) * conv_dim, rng, -1.0, 1.0, &xd);
  const auto cw = rand_t<T>(static_cast<size_t>(conv_dim) * width, rng, -0.6, 0.6, &wd);
  const auto cb = rand_t<T>(static_cast<size_t>(conv_dim), rng, -0.2, 0.2, &cbd);
  const auto cs0 = rand_t<T>(static_cast<size_t>(kSlots) * conv_dim * (width - 1), rng, -1.0, 1.0, &csd);
  const auto ga = rand_t<T>(static_cast<size_t>(B) * Hv * per, rng, -3.0, 3.0, &gad);
  const auto gb = rand_t<T>(static_cast<size_t>(B) * Hv, rng, -3.0, 3.0, &gbd);
  const auto A_log = rand_f(Hv, rng, -1.0, 1.0, &Ad);
  const auto dtb = rand_f(static_cast<size_t>(Hv) * per, rng, -1.0, 1.0, &dbd);
  const auto og = rand_t<T>(static_cast<size_t>(B) * Hv * dv, rng, -3.0, 3.0, &ogd);
  const auto nw = rand_f(dv, rng, 0.5, 1.5, &nwd);
  const auto st0 = rand_f(static_cast<size_t>(kSlots) * Hv * dk * dv, rng, -0.3, 0.3, &sd);
  Arena ar;
  T* out = ar.put(std::vector<T>(static_cast<size_t>(B) * Hv * dv, to_t<T>(7.0)));
  T* cs = ar.put(cs0);
  float* st = ar.put(st0);
  kops::state::FusedDeltaDecodeParams fp{};
  kops::state::DeltaRuleParams& p = fp.rule;
  p.scale = static_cast<float>(scale);
  p.use_qk_l2norm = true;
  p.state_indices = ar.put(slots);
  p.has_initial_state = ar.put(has_init);
  p.state_pool = st;
  p.state_slot_stride = static_cast<int64_t>(Hv) * dk * dv;
  p.num_slots = kSlots;
  p.total_tokens = B;
  p.batch_size = B;
  p.num_k_heads = Hk;
  p.num_v_heads = Hv;
  p.head_dim_k = dk;
  p.head_dim_v = dv;
  p.decay = kda ? kops::state::DecayGranularity::kPerChannel : kops::state::DecayGranularity::kPerHead;
  p.pad_slot_id = 3;
  fp.qkv = ar.put(qkv);
  fp.qkv_batch_stride = conv_dim;
  fp.conv_weight = ar.put(cw);
  fp.conv_weight_dim_stride = width;
  fp.conv_weight_width_stride = 1;
  fp.conv_bias = ar.put(cb);
  fp.conv_states = cs;
  fp.conv_state_slot_stride = static_cast<int64_t>(conv_dim) * (width - 1);
  fp.conv_state_dim_stride = width - 1;
  fp.conv_state_len_stride = 1;
  fp.conv_width = width;
  fp.gate_a = ar.put(ga);
  fp.gate_a_batch_stride = static_cast<int64_t>(Hv) * per;
  fp.gate_b = ar.put(gb);
  fp.gate_b_batch_stride = Hv;
  fp.A_log = ar.put(A_log);
  fp.dt_bias = ar.put(dtb);
  fp.lower_bound = 0.0f;
  fp.out_gate = ar.put(og);
  fp.out_gate_batch_stride = static_cast<int64_t>(Hv) * dv;
  fp.out_gate_act = gate_act;
  fp.out_norm_weight = ar.put(nw);
  fp.out_norm_eps = eps;
  fp.out = out;
  fp.out_batch_stride = static_cast<int64_t>(Hv) * dv;
  KOPS_EXPECT_OK(kops::state::fused_delta_decode(fp, dtype_of<T>(), ctx), "fused_delta_decode");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(out, static_cast<size_t>(B) * Hv * dv);
  const auto got_st = kops_test::copy_out(st, st0.size());
  std::vector<double> want(got.size(), 7.0), want_st = sd;
  const size_t se = static_cast<size_t>(Hv) * dk * dv;
  for (int b = 0; b < B; ++b) {
    if (slots[b] == 3) continue;
    std::vector<double> hist(csd.begin() + static_cast<ptrdiff_t>(slots[b]) * conv_dim * (width - 1),
                             csd.begin() + static_cast<ptrdiff_t>(slots[b] + 1) * conv_dim * (width - 1));
    std::vector<double> y(static_cast<size_t>(conv_dim));
    kops::ref::causal_conv1d_f64(xd.data() + static_cast<size_t>(b) * conv_dim, 1, conv_dim, wd.data(), width, cbd.data(),
                                 ActFn::kSiLU, hist.data(), y.data());
    std::vector<double> g(static_cast<size_t>(Hv) * per), beta(static_cast<size_t>(Hv));
    kops::ref::delta_rule_gating_f64(gad.data() + static_cast<size_t>(b) * Hv * per, gbd.data() + static_cast<size_t>(b) * Hv,
                                     Ad.data(), dbd.data(), 0.0, 1, Hv, per, g.data(), beta.data());
    std::vector<double> s(se, 0.0), o(static_cast<size_t>(Hv) * dv);
    if (has_init[b]) std::copy(sd.begin() + static_cast<ptrdiff_t>(se * slots[b]), sd.begin() + static_cast<ptrdiff_t>(se * (slots[b] + 1)), s.begin());
    kops::ref::DeltaRuleRefArgs r;
    r.q = y.data();
    r.k = y.data() + static_cast<size_t>(Hk) * dk;
    r.v = y.data() + static_cast<size_t>(2) * Hk * dk;
    r.g = g.data();
    r.beta = beta.data();
    r.len = 1;
    r.num_k_heads = Hk;
    r.num_v_heads = Hv;
    r.head_dim_k = dk;
    r.head_dim_v = dv;
    r.per_channel = kda;
    r.scale = scale;
    r.state = s.data();
    r.o = o.data();
    kops::ref::delta_rule_f64(r);
    std::copy(s.begin(), s.end(), want_st.begin() + static_cast<ptrdiff_t>(se * slots[b]));
    for (int h = 0; h < Hv; ++h) {
      double ss = 0.0;
      for (int j = 0; j < dv; ++j) ss += o[static_cast<size_t>(h) * dv + j] * o[static_cast<size_t>(h) * dv + j];
      const double inv = 1.0 / std::sqrt(ss / dv + eps);
      for (int j = 0; j < dv; ++j) {
        const double gv = ogd[(static_cast<size_t>(b) * Hv + h) * dv + j];
        const double act = gate_act == ActFn::kSiLU ? gv / (1.0 + std::exp(-gv)) : 1.0 / (1.0 + std::exp(-gv));
        want[(static_cast<size_t>(b) * Hv + h) * dv + j] = o[static_cast<size_t>(h) * dv + j] * inv * nwd[j] * act;
      }
    }
  }
  // 16-bit: the conv output and the rule output are rounded at the unfused chain's boundaries
  compare_out(got, want, kda ? "fused kda out" : "fused gdn out", 1.4);
  const bool f32 = std::is_same<T, float>::value;
  kops_test::compare(got_st, want_st, f32 ? 1e-6 : 5e-3 * out_scale(want_st), f32 ? 2e-5 : 1e-2, "fused state");
}

void test_linear_and_delta(LaunchContext& ctx) {
  for (int dk : {64, 128, 96}) {
    test_linear_attention<__nv_bfloat16>(ctx, dk, 64, true);
    test_linear_attention<float>(ctx, dk, dk == 96 ? 40 : 128, true);
  }
  test_linear_attention<float>(ctx, 256, 32, true);
  test_linear_attention<__nv_bfloat16>(ctx, 128, 128, false);
  test_linear_attention<__half>(ctx, 48, 200, false);
  for (int dk : {32, 64, 128}) {
    test_delta_recurrent<float>(ctx, dk, 64, false, false);
    test_delta_recurrent<__nv_bfloat16>(ctx, dk, 128, true, dk == 64);
  }
  test_delta_recurrent<__half>(ctx, 16, 40, true, false);
  test_delta_gating<__nv_bfloat16>(ctx, false, 0.0f);
  test_delta_gating<float>(ctx, true, 0.0f);
  test_delta_gating<__half>(ctx, false, -5.0f);
  // Kimi Linear (KDA, Hk = Hv, Dk = Dv = 128, width 4, sigmoid gate, eps 1e-5) and Qwen3-Next (GDN, GVA 2, silu gate)
  test_fused_delta<__nv_bfloat16>(ctx, 2, 2, 128, 128, 4, true, ActFn::kSigmoid, 1e-5f);
  test_fused_delta<float>(ctx, 2, 2, 128, 128, 4, true, ActFn::kSigmoid, 1e-5f);
  test_fused_delta<__nv_bfloat16>(ctx, 2, 4, 128, 128, 4, false, ActFn::kSiLU, 1e-6f);
  test_fused_delta<float>(ctx, 1, 8, 64, 64, 3, false, ActFn::kSiLU, 1e-6f);
  test_fused_delta<__half>(ctx, 3, 3, 64, 32, 2, true, ActFn::kSigmoid, 1e-5f);
  using kops::state::ChunkMath;
  for (int dk : {64, 128}) {
    test_delta_chunked<float>(ctx, dk, 64, false, ChunkMath::kFp32, false);
    test_delta_chunked<float>(ctx, dk, 128, true, ChunkMath::kFp32, dk == 64);
    test_delta_chunked<__nv_bfloat16>(ctx, dk, 128, false, ChunkMath::kTensorCore, false);
    test_delta_chunked<__nv_bfloat16>(ctx, dk, 64, true, ChunkMath::kTensorCore, dk == 128);
  }
  test_delta_chunked<__nv_bfloat16>(ctx, 32, 40, true, ChunkMath::kFp32, false);
  test_delta_chunked<__half>(ctx, 16, 24, false, ChunkMath::kFp32, false);
  test_delta_chunked<__nv_bfloat16>(ctx, 16, 32, false, ChunkMath::kTensorCore, false);
  kops::state::DeltaRuleParams bad{};
  bad.batch_size = 1;
  bad.num_k_heads = 1;
  bad.num_v_heads = 1;
  bad.head_dim_k = 48;
  bad.head_dim_v = 64;
  bad.state_slot_stride = 48 * 64;
  KOPS_EXPECT_STATUS(kops::state::delta_rule_decode(bad, DType::kBFloat16, ctx), Status::kUnsupportedShape, "delta Dk 48");
  bad.head_dim_k = 64;
  bad.state_slot_stride = 64 * 64;
  bad.math = ChunkMath::kTensorCore;
  KOPS_EXPECT_STATUS(kops::state::delta_rule_prefill(bad, DType::kFloat32, kops::Workspace{}, ctx), Status::kUnsupportedDtype,
                     "delta tc fp32");
}

// ---- B9 boundary gate: 64k-token sequences (no overflow, no NaN) --------------------------------

template <typename T>
bool all_finite(const std::vector<T>& v) {
  for (const T& x : v)
    if (!std::isfinite(to_d(x))) return false;
  return true;
}

/// Chunked (fp32 or tensor-core) vs recurrent over one 65 536-token sequence: every chunk's decay
/// factors stay <= 1 and the cumulative products underflow to zero instead of overflowing.
template <typename T>
void test_delta_long(LaunchContext& ctx, bool kda, kops::state::ChunkMath math) {
  DeltaCase<T> c(kda ? 501u : 502u, {65536, 1}, 1, 2, 64, 64, kda, false, 0, 0);
  const double scale = 0.125;
  Arena ar;
  kops::state::DeltaRuleParams p, r;
  T* o = nullptr;
  T* o2 = nullptr;
  float* st = nullptr;
  float* st2 = nullptr;
  c.fill(p, ar, &o, &st, scale, false, false);
  c.fill(r, ar, &o2, &st2, scale, false, false);
  p.math = math;
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::state::delta_rule_workspace_requirements(c.ntok, c.B, c.Hv, 64, 64, 64, p.decay, &req), "long ws");
  DevBuf<uint8_t> wsb(req.device_scratch_bytes);
  kops::Workspace ws{};
  ws.device_scratch = wsb.p;
  ws.device_scratch_bytes = req.device_scratch_bytes;
  KOPS_EXPECT_OK(kops::state::delta_rule_prefill(p, dtype_of<T>(), ws, ctx), "long delta_rule_prefill");
  KOPS_EXPECT_OK(kops::state::delta_rule_decode(r, dtype_of<T>(), ctx), "long delta_rule_decode");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(o, c.o0.size());
  const auto rec = kops_test::copy_out(o2, c.o0.size());
  const auto gst = kops_test::copy_out(st, c.st0.size());
  const auto rst = kops_test::copy_out(st2, c.st0.size());
  KOPS_TEST_CHECK(all_finite(got) && all_finite(gst));
  std::vector<double> rec_d(rec.size()), rst_d(rst.size());
  for (size_t i = 0; i < rec.size(); ++i) rec_d[i] = to_d(rec[i]);
  for (size_t i = 0; i < rst.size(); ++i) rst_d[i] = rst[i];
  const bool tc = math == kops::state::ChunkMath::kTensorCore;
  if (tc) kops_test::compare(got, rec_d, 7e-3 * out_scale(rec_d), 3e-2, "long chunk tc vs recurrent");
  else compare_out(got, rec_d, "long chunk vs recurrent", 2.0);
  kops_test::compare(gst, rst_d, tc ? 4e-3 * out_scale(rst_d) : 1e-5, tc ? 1e-2 : 1e-4, "long chunk state vs recurrent");
}

/// Lightning attention and the SSD scan over 65 536 tokens against the fp64 oracles.
void test_linear_ssd_long(LaunchContext& ctx) {
  using T = __nv_bfloat16;
  const int L = 65536;
  {
    Lcg rng(611);
    const int H = 1, D = 32;
    std::vector<double> qd, kd, vd, sl;
    const auto q = rand_t<T>(static_cast<size_t>(L) * D, rng, -1.0, 1.0, &qd);
    const auto k = rand_t<T>(q.size(), rng, -1.0, 1.0, &kd);
    const auto v = rand_t<T>(q.size(), rng, -1.0, 1.0, &vd);
    const std::vector<float> slope = {0.002f};
    sl = to_double(slope);
    Arena ar;
    T* o = ar.put(std::vector<T>(q.size(), to_t<T>(7.0)));
    float* st = ar.put(std::vector<float>(static_cast<size_t>(D) * D, 0.0f));
    kops::state::LinearAttentionParams p{};
    p.q = ar.put(q);
    p.k = ar.put(k);
    p.v = ar.put(v);
    p.q_token_stride = p.k_token_stride = p.v_token_stride = p.o_token_stride = D;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = D;
    p.slope = ar.put(slope);
    p.scale = 0.1f;
    p.state_pool = st;
    p.state_slot_stride = static_cast<int64_t>(D) * D;
    p.o = o;
    p.cu_seqlens = ar.put(std::vector<int32_t>{0, L});
    p.has_initial_state = ar.put(std::vector<uint8_t>{0});
    p.num_slots = 1;
    p.total_tokens = L;
    p.batch_size = 1;
    p.num_heads = H;
    p.head_dim_k = D;
    p.head_dim_v = D;
    p.pad_slot_id = -1;
    KOPS_EXPECT_OK(kops::state::lightning_attention_prefill(p, DType::kBFloat16, ctx), "long lightning");
    cudaDeviceSynchronize();
    std::vector<double> s(static_cast<size_t>(D) * D, 0.0), want(q.size());
    kops::ref::linear_attention_f64(qd.data(), kd.data(), vd.data(), sl.data(), 0.1, L, H, D, D, s.data(), want.data());
    const auto got = kops_test::copy_out(o, q.size());
    KOPS_TEST_CHECK(all_finite(got));
    compare_out(got, want, "long lightning out");
    kops_test::compare(kops_test::copy_out(st, s.size()), s, 1e-5 * out_scale(s), 2e-4, "long lightning state");
  }
  {
    Lcg rng(612);
    const int H = 2, P = 32, N = 32, CH = 256, NC = L / CH;
    std::vector<double> xd, dtd, Bd, Cd, Ad;
    const auto x = rand_t<T>(static_cast<size_t>(L) * H * P, rng, -1.0, 1.0, &xd);
    const auto dt = rand_t<T>(static_cast<size_t>(L) * H, rng, -4.0, -1.0, &dtd);
    const auto Bm = rand_t<T>(static_cast<size_t>(L) * N, rng, -1.0, 1.0, &Bd);
    const auto Cm = rand_t<T>(static_cast<size_t>(L) * N, rng, -1.0, 1.0, &Cd);
    const auto A = rand_f(H, rng, -0.5, -0.1, &Ad);
    Arena ar;
    T* out = ar.put(std::vector<T>(x.size(), to_t<T>(7.0)));
    float* st = ar.put(std::vector<float>(static_cast<size_t>(H) * P * N, 0.0f));
    std::vector<int32_t> ccs(static_cast<size_t>(NC) + 1);
    for (int c = 0; c <= NC; ++c) ccs[static_cast<size_t>(c)] = c * CH;
    kops::state::SSDParams p{};
    p.x = ar.put(x);
    p.x_token_stride = static_cast<int64_t>(H) * P;
    p.x_head_stride = P;
    p.dt = ar.put(dt);
    p.dt_token_stride = H;
    p.dt_head_stride = 1;
    p.A = ar.put(A);
    p.B = ar.put(Bm);
    p.B_token_stride = N;
    p.B_group_stride = N;
    p.C = ar.put(Cm);
    p.C_token_stride = N;
    p.C_group_stride = N;
    p.out = out;
    p.out_token_stride = p.x_token_stride;
    p.out_head_stride = P;
    p.dt_softplus = true;
    p.cu_seqlens = ar.put(std::vector<int32_t>{0, L});
    p.cu_chunks = ar.put(std::vector<int32_t>{0, NC});
    p.cu_chunk_seqlens = ar.put(ccs);
    p.ssm_states = st;
    p.ssm_state_slot_stride = static_cast<int64_t>(H) * P * N;
    p.has_initial_state = ar.put(std::vector<uint8_t>{0});
    p.num_slots = 1;
    p.batch_size = 1;
    p.num_chunks = NC;
    p.chunk_size = CH;
    p.nheads = H;
    p.headdim = P;
    p.dstate = N;
    p.ngroups = 1;
    p.pad_slot_id = -1;
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kops::state::ssd_workspace_requirements(L, 1, H, P, N, 1, CH, &req), "long ssd ws");
    DevBuf<uint8_t> wsb(req.device_scratch_bytes);
    kops::Workspace ws{};
    ws.device_scratch = wsb.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    KOPS_EXPECT_OK(kops::state::ssd_chunked_scan(p, DType::kBFloat16, ws, ctx), "long ssd");
    cudaDeviceSynchronize();
    std::vector<double> s(static_cast<size_t>(H) * P * N, 0.0), want(x.size());
    kops::ref::SsmRefArgs r;
    r.x = xd.data();
    r.dt = dtd.data();
    r.A = Ad.data();
    r.B = Bd.data();
    r.C = Cd.data();
    r.len = L;
    r.nheads = H;
    r.headdim = P;
    r.dstate = N;
    r.ngroups = 1;
    r.dt_softplus = true;
    r.state = s.data();
    r.y = want.data();
    kops::ref::ssm_recurrence_f64(r);
    const auto got = kops_test::copy_out(out, x.size());
    KOPS_TEST_CHECK(all_finite(got));
    compare_out(got, want, "long ssd out");
    kops_test::compare(kops_test::copy_out(st, s.size()), s, 5e-6 * out_scale(s), 2e-4, "long ssd state");
  }
}

void test_long(LaunchContext& ctx) {
  using kops::state::ChunkMath;
  test_delta_long<float>(ctx, false, ChunkMath::kFp32);
  test_delta_long<float>(ctx, true, ChunkMath::kFp32);
  test_delta_long<__nv_bfloat16>(ctx, true, ChunkMath::kTensorCore);
  test_linear_ssd_long(ctx);
}

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
  test_copy(ctx);
  test_gather_scatter(ctx);
  test_zero(ctx);
  test_graph(ctx);
  test_commit(ctx);
  test_contracts(ctx);
  test_conv1d(ctx);
  test_commit_hybrid(ctx);
  test_ssm(ctx);
  test_linear_and_delta(ctx);
  test_long(ctx);
  return kops_test::finish("test_state");
}
