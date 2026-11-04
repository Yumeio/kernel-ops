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

template <typename T>
double out_atol() { return std::is_same<T, float>::value ? 2e-5 : 4e-3; }
template <typename T>
double out_rtol() { return std::is_same<T, float>::value ? 2e-5 : 1.2e-2; }

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
  kops_test::compare(got, want, out_atol<T>(), out_rtol<T>(), "conv1d_fwd out");
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
  kops_test::compare(got, want, out_atol<T>(), out_rtol<T>(), "conv1d_update out");
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

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
  test_copy(ctx);
  test_gather_scatter(ctx);
  test_zero(ctx);
  test_graph(ctx);
  test_contracts(ctx);
  test_conv1d(ctx);
  return kops_test::finish("test_state");
}
