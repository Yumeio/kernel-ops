// =============================================================================
// kernel-ops | tests/unit/test_communication.cc
// B12 tests: peer collectives, fused allreduce + RMSNorm, expert-parallel dispatch / combine and the
// overlapped TP GEMMs.
//
// Rank models (ROADMAP 3.3 K5):
//   virtual  N ranks = N regions of one allocation on device 0, N streams. The kernels take peer
//            pointers, so they run unchanged; ranks need to be co-resident (small grids).
//   host     two real GPUs without P2P: staging in pinned write-combined host memory, signals in
//            pinned host memory. This is the transport the benchmark measures against NCCL.
// Real P2P / NVLink, multimem and RDMA are not exercised here; those cells stay `authored`.
//
// Oracles: collectives sum in fp32 in ascending rank order and round once (bit-exact), plus an fp64
// sum as an independent bound; fusion, EP and comm-GEMM oracles are in the sections below.
// Options: --tier smoke|standard|full, --dryrun (host-side oracle self-checks only, no kernel),
// --mutate (flip one output bit before comparing: the run must FAIL).
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/quantization/ops.h"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/communication/allreduce_kernels.cuh"
#include "csrc/communication/ep_kernels.cuh"
#include "csrc/communication/comm_gemm_kernels.cuh"
#include "csrc/communication/ops.h"

namespace {

using kops::ActQuantMode;
using kops::ActQuantOut;
using kops::DType;
using kops::Fp8Encoding;
using kops::LaunchContext;
using kops::Status;
using kops::Tensor;
using namespace kops::comm;

kops_test::Options g_opt;

Tensor flat(void* data, DType dtype, int64_t n, int device = 0) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = device;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

#define CUDA_OK(expr)                                                              \
  do {                                                                             \
    const cudaError_t e_ = (expr);                                                 \
    if (e_ != cudaSuccess) {                                                       \
      kops_test::record_failure(#expr, cudaGetErrorString(e_), __FILE__, __LINE__); \
    }                                                                              \
  } while (0)

// ---- element type helpers ---------------------------------------------------------------------------
template <typename T> DType dtype_of();
template <> DType dtype_of<__nv_bfloat16>() { return DType::kBFloat16; }
template <> DType dtype_of<__half>() { return DType::kFloat16; }
template <> DType dtype_of<float>() { return DType::kFloat32; }

template <typename T>
uint32_t bits_of(T v) {
  uint32_t b = 0;
  std::memcpy(&b, &v, sizeof(T));
  return b;
}

template <typename T>
T round_to(float f) {
  T t;
  kops_test::from_float(t, f);
  return t;
}

template <typename T>
std::vector<T> random_vec(size_t n, unsigned seed, float amp = 4.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-amp, amp);
  std::vector<T> v(n);
  for (T& x : v) x = round_to<T>(d(rng));
  return v;
}

/// Ascending-rank fp32 accumulation, one rounding: what every collective here must produce bit for bit.
template <typename T>
std::vector<T> sum_ranks(const std::vector<std::vector<T>>& in, size_t off, size_t n) {
  std::vector<T> out(n);
  for (size_t i = 0; i < n; ++i) {
    float acc = kops_test::to_float(in[0][off + i]);
    for (size_t p = 1; p < in.size(); ++p) acc += kops_test::to_float(in[p][off + i]);
    out[i] = round_to<T>(acc);
  }
  return out;
}

template <typename T>
void expect_bits(const std::vector<T>& got, const std::vector<T>& want, const char* what) {
  if (got.size() != want.size()) {
    kops_test::record_failure(what, "size mismatch", __FILE__, __LINE__);
    return;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (bits_of(got[i]) != bits_of(want[i])) {
      char d[200];
      std::snprintf(d, sizeof(d), "index %zu of %zu: got %.9g (0x%x) want %.9g (0x%x)", i, got.size(),
                    kops_test::to_float(got[i]), bits_of(got[i]), kops_test::to_float(want[i]),
                    bits_of(want[i]));
      kops_test::record_failure(what, d, __FILE__, __LINE__);
      return;
    }
  }
}

/// --mutate: flip one low bit of the first element so a comparison against the oracle must fail.
template <typename T>
void maybe_mutate(std::vector<T>* v) {
  if (!g_opt.mutate || v->empty()) return;
  uint8_t* p = reinterpret_cast<uint8_t*>(v->data());
  p[0] ^= 1u;
}

// ---- multi-rank fixture ------------------------------------------------------------------------------------
struct RankCtx {
  int device = 0;
  cudaStream_t stream = nullptr;
  PeerBuffers peers{};
  LaunchContext ctx{};
};

struct World {
  int size = 0;
  bool host_staged = false;
  size_t buffer_bytes = 0;
  void* staging = nullptr;
  uint32_t* signals = nullptr;
  std::vector<RankCtx> r;

  /// Returns false (test skipped, caller returns) when the machine cannot host the world.
  bool init(int world, bool host, size_t bytes, uint32_t timeout_us) {
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (host && ndev < world) return false;
    size = world;
    host_staged = host;
    buffer_bytes = (bytes + 127) & ~size_t(127);
    const size_t sig_bytes = size_t(kSignalWords) * sizeof(uint32_t);
    if (host) {
      CUDA_OK(cudaHostAlloc(&staging, buffer_bytes * world,
                            cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined));
      CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&signals), sig_bytes * world,
                            cudaHostAllocPortable | cudaHostAllocMapped));
    } else {
      CUDA_OK(cudaSetDevice(0));
      CUDA_OK(cudaMalloc(&staging, buffer_bytes * world));
      CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&signals), sig_bytes * world));
    }
    r.resize(world);
    for (int k = 0; k < world; ++k) {
      RankCtx& rk = r[k];
      rk.device = host ? k : 0;
      CUDA_OK(cudaSetDevice(rk.device));
      CUDA_OK(cudaStreamCreateWithFlags(&rk.stream, cudaStreamNonBlocking));
      std::memset(&rk.peers, 0, sizeof(rk.peers));
      for (int j = 0; j < world; ++j) {
        rk.peers.data[j] = static_cast<char*>(staging) + buffer_bytes * j;
        rk.peers.flags[j] = signals + size_t(kSignalWords) * j;
      }
      rk.peers.rank = k;
      rk.peers.world_size = world;
      rk.peers.num_nodes = 1;
      rk.peers.node_rank = 0;
      rk.peers.buffer_bytes = buffer_bytes;
      CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&rk.peers.scratch), kScratchWords * sizeof(uint32_t)));
      KOPS_EXPECT_OK(kops::make_launch_context(rk.device, rk.stream, &rk.ctx), "make_launch_context");
    }
    reconfigure(timeout_us);
    return true;
  }

  void reconfigure(uint32_t timeout_us) {
    sync_all();
    for (int k = 0; k < size; ++k) {
      CUDA_OK(cudaSetDevice(r[k].device));
      KOPS_EXPECT_OK(comm_configure(r[k].peers, timeout_us), "comm_configure");
    }
  }

  void sync_all() {
    for (RankCtx& rk : r) {
      CUDA_OK(cudaSetDevice(rk.device));
      CUDA_OK(cudaStreamSynchronize(rk.stream));
    }
  }

  void destroy() {
    sync_all();
    for (RankCtx& rk : r) {
      cudaSetDevice(rk.device);
      cudaStreamDestroy(rk.stream);
      cudaFree(rk.peers.scratch);
    }
    if (host_staged) {
      cudaFreeHost(staging);
      cudaFreeHost(signals);
    } else {
      cudaSetDevice(0);
      cudaFree(staging);
      cudaFree(signals);
    }
    r.clear();
  }
};

/// Per-rank device buffers of one element type.
template <typename T>
struct RankBuf {
  T* p = nullptr;
  size_t n = 0;
  int device = 0;
  void alloc(int dev, size_t count) {
    device = dev;
    n = count;
    cudaSetDevice(dev);
    CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(count, 1) * sizeof(T)));
  }
  // The rank streams are non-blocking, so legacy-stream copies and memsets are not ordered against
  // them: finish the device before a kernel can observe the buffer.
  void put(const std::vector<T>& h) {
    cudaSetDevice(device);
    CUDA_OK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_OK(cudaDeviceSynchronize());
  }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaSetDevice(device);
    CUDA_OK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  void fill_bytes(int v) {
    cudaSetDevice(device);
    CUDA_OK(cudaMemset(p, v, n * sizeof(T)));
    CUDA_OK(cudaDeviceSynchronize());
  }
  ~RankBuf() {
    if (p != nullptr) {
      cudaSetDevice(device);
      cudaFree(p);
    }
  }
};

/// Launch fn(rank) for every rank in `order`, with an optional host-side stagger between launches so
/// ranks reach the barrier at different times.
template <typename Fn>
void launch_ranks(World& w, const std::vector<int>& order, int stagger_us, Fn&& fn) {
  for (int k : order) {
    CUDA_OK(cudaSetDevice(w.r[k].device));
    fn(k);
    if (stagger_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(stagger_us));
  }
}
std::vector<int> iota_order(int n) {
  std::vector<int> v(n);
  std::iota(v.begin(), v.end(), 0);
  return v;
}

const char* algo_name(AllReduceAlgo a) {
  switch (a) {
    case AllReduceAlgo::kOneShot: return "one_shot";
    case AllReduceAlgo::kTwoShot: return "two_shot";
    case AllReduceAlgo::kDeterministic: return "deterministic";
    default: return "other";
  }
}

// ======================================================================================================
// (b) contract
// ======================================================================================================
void test_contract_collectives() {
  World w;
  if (!w.init(2, false, 1 << 16, 0)) return;
  RankBuf<__nv_bfloat16> in, out;
  in.alloc(0, 1024);
  out.alloc(0, 1024);
  in.put(random_vec<__nv_bfloat16>(1024, 1));
  const auto sentinel = [&]() { out.fill_bytes(0x5A); };
  auto untouched = [&](const char* what) {
    CUDA_OK(cudaStreamSynchronize(w.r[0].stream));
    const std::vector<__nv_bfloat16> got = out.get();
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(got.data());
    for (size_t i = 0; i < got.size() * sizeof(__nv_bfloat16); ++i) {
      if (raw[i] != 0x5A) {
        kops_test::record_failure(what, "output was written despite a rejected call", __FILE__, __LINE__);
        return;
      }
    }
  };
  const PeerBuffers& peers = w.r[0].peers;
  const LaunchContext& ctx = w.r[0].ctx;
  const AllReduceAlgo a = AllReduceAlgo::kOneShot;
  Tensor tin = flat(in.p, DType::kBFloat16, 1024), tout = flat(out.p, DType::kBFloat16, 1024);

  sentinel();
  Tensor bad = flat(in.p, DType::kFloat8E4M3, 1024);
  KOPS_EXPECT_STATUS(allreduce(tout, bad, peers, a, 0, ctx), Status::kUnsupportedDtype, "allreduce fp8 dtype");
  Tensor odd_in = flat(in.p, DType::kBFloat16, 1001), odd_out = flat(out.p, DType::kBFloat16, 1001);
  KOPS_EXPECT_STATUS(allreduce(odd_out, odd_in, peers, a, 0, ctx), Status::kUnsupportedShape,
                     "allreduce numel not a multiple of 8");
  Tensor small_out = flat(out.p, DType::kBFloat16, 512);
  KOPS_EXPECT_STATUS(allreduce(small_out, tin, peers, a, 0, ctx), Status::kInvalidArgument,
                     "allreduce out/in size mismatch");
  Tensor big_in = flat(in.p, DType::kBFloat16, 1 << 20), big_out = flat(out.p, DType::kBFloat16, 1 << 20);
  KOPS_EXPECT_STATUS(allreduce(big_out, big_in, peers, a, 0, ctx), Status::kWorkspaceTooSmall,
                     "allreduce staging too small");
  KOPS_EXPECT_STATUS(allreduce(tin, tin, peers, a, 0, ctx), Status::kUnsupportedLayout, "allreduce in place");
  KOPS_EXPECT_STATUS(allreduce(tout, tin, peers, AllReduceAlgo::kMultimem, 0, ctx),
                     Status::kUnsupportedBackend, "allreduce multimem without a multicast object");
  KOPS_EXPECT_STATUS(allreduce(tout, tin, peers, AllReduceAlgo::kQuickReduce, 0, ctx),
                     Status::kNotImplemented, "allreduce quick reduce");
  Tensor mis = tin;
  mis.byte_offset = 2;  // 2 B: breaks the 16 B alignment contract
  KOPS_EXPECT_STATUS(allreduce(tout, mis, peers, a, 0, ctx), Status::kMisaligned, "allreduce misaligned input");

  PeerBuffers p3 = peers;
  p3.world_size = 3;
  KOPS_EXPECT_STATUS(allreduce(tout, tin, p3, a, 0, ctx), Status::kUnsupportedShape, "allreduce world 3");
  PeerBuffers prank = peers;
  prank.rank = 5;
  KOPS_EXPECT_STATUS(allreduce(tout, tin, prank, a, 0, ctx), Status::kInvalidArgument, "allreduce rank >= world");
  PeerBuffers pnull = peers;
  pnull.data[1] = nullptr;
  KOPS_EXPECT_STATUS(allreduce(tout, tin, pnull, a, 0, ctx), Status::kInvalidArgument, "allreduce null staging");
  PeerBuffers pnode = peers;
  pnode.num_nodes = 2;
  KOPS_EXPECT_STATUS(allreduce(tout, tin, pnode, a, 0, ctx), Status::kUnsupportedBackend, "allreduce multi-node");
  PeerBuffers pscr = peers;
  pscr.scratch = nullptr;
  KOPS_EXPECT_STATUS(allreduce(tout, tin, pscr, a, 0, ctx), Status::kInvalidArgument, "allreduce null scratch");
  PeerBuffers pmis = peers;
  pmis.data[1] = static_cast<char*>(pmis.data[1]) + 16;
  KOPS_EXPECT_STATUS(allreduce(tout, tin, pmis, a, 0, ctx), Status::kMisaligned, "allreduce staging alignment");

  Tensor g_out = flat(out.p, DType::kBFloat16, 1024);
  KOPS_EXPECT_STATUS(allgather(g_out, tin, peers, 0, ctx), Status::kInvalidArgument, "allgather out size");
  Tensor r_out = flat(out.p, DType::kBFloat16, 513);
  KOPS_EXPECT_STATUS(reduce_scatter(r_out, tin, peers, 0, ctx), Status::kInvalidArgument, "reduce_scatter size");
  Tensor r_ok = flat(out.p, DType::kBFloat16, 512);
  Tensor r_in3 = flat(in.p, DType::kBFloat16, 1020);  // 1020 % 2 == 0 but 510 elements: not a multiple of 8
  KOPS_EXPECT_STATUS(reduce_scatter(flat(out.p, DType::kBFloat16, 510), r_in3, peers, 0, ctx),
                     Status::kUnsupportedShape, "reduce_scatter per-rank numel");
  (void)r_ok;
  untouched("rejected calls");

  int32_t code = -1;
  KOPS_EXPECT_STATUS(comm_error(peers, nullptr), Status::kInvalidArgument, "comm_error null out");
  KOPS_EXPECT_OK(comm_error(peers, &code), "comm_error");
  KOPS_TEST_CHECK(code == 0);
  KOPS_EXPECT_OK(validate_peer_buffers(peers, 2 * 1024 * 2), "validate_peer_buffers");
  KOPS_EXPECT_STATUS(validate_peer_buffers(peers, size_t(1) << 40), Status::kWorkspaceTooSmall,
                     "validate_peer_buffers capacity");
  PeerBuffers pmc = peers;
  int dummy = 0;
  pmc.multicast_ptr = &dummy;
  KOPS_EXPECT_STATUS(validate_peer_buffers(pmc, 1024), Status::kUnsupportedBackend, "validate multicast");
  KOPS_TEST_CHECK(allreduce_staging_bytes(a, DType::kBFloat16, 1024, 2) == 2 * 2048);
  KOPS_TEST_CHECK(allreduce_staging_bytes(a, DType::kFloat8E4M3, 1024, 2) == 0);
  KOPS_TEST_CHECK(allreduce_staging_bytes(a, DType::kBFloat16, 1024, 3) == 0);
  KOPS_TEST_CHECK(allgather_staging_bytes(DType::kFloat32, 256, 4) == 2 * 1024);
  KOPS_TEST_CHECK(reduce_scatter_staging_bytes(DType::kFloat16, 4096, 8) == 2 * 8192);
  w.destroy();
}

void test_validate_no_p2p() {
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (ndev < 2) return;
  int can = 0;
  cudaDeviceCanAccessPeer(&can, 0, 1);
  if (can != 0) return;  // a machine with P2P cannot show the rejection
  World w;
  if (!w.init(2, false, 1 << 12, 0)) return;
  void* far = nullptr;
  CUDA_OK(cudaSetDevice(1));
  CUDA_OK(cudaMalloc(&far, 1 << 12));
  CUDA_OK(cudaSetDevice(0));
  PeerBuffers p = w.r[0].peers;
  p.data[1] = far;
  KOPS_EXPECT_STATUS(validate_peer_buffers(p, 1 << 10), Status::kCommError, "device memory of a non-peer GPU");
  CUDA_OK(cudaSetDevice(1));
  cudaFree(far);
  CUDA_OK(cudaSetDevice(0));
  w.destroy();
}

// ======================================================================================================
// (c)/(d) numerics on virtual and host-staged ranks
// ======================================================================================================
template <typename T>
void run_allreduce_case(World& w, AllReduceAlgo algo, size_t n, unsigned seed, const char* tag,
                        const std::vector<int>* order = nullptr, int stagger_us = 0) {
  const int W = w.size;
  std::vector<std::vector<T>> host_in(W);
  std::vector<RankBuf<T>> in(W), out(W);
  for (int k = 0; k < W; ++k) {
    host_in[k] = random_vec<T>(n, seed * 31 + k * 7 + 1);
    in[k].alloc(w.r[k].device, n);
    out[k].alloc(w.r[k].device, n);
    in[k].put(host_in[k]);
    out[k].fill_bytes(0);
  }
  const std::vector<int> ord = order ? *order : iota_order(W);
  launch_ranks(w, ord, stagger_us, [&](int k) {
    KOPS_EXPECT_OK(allreduce(flat(out[k].p, dtype_of<T>(), n, w.r[k].device),
                             flat(in[k].p, dtype_of<T>(), n, w.r[k].device), w.r[k].peers, algo, 0,
                             w.r[k].ctx),
                   tag);
  });
  w.sync_all();
  const std::vector<T> want = sum_ranks<T>(host_in, 0, n);
  // independent fp64 bound: the fp32 rank-order sum is within 2 ulp of the exact sum
  for (int k = 0; k < W; ++k) {
    std::vector<T> got = out[k].get();
    if (k == 0) maybe_mutate(&got);
    char what[160];
    std::snprintf(what, sizeof(what), "%s rank %d", tag, k);
    expect_bits(got, want, what);
  }
  // independent fp64 bound: the result is within one rounding of the exact sum, measured against
  // sum |x_p| so cancellation cannot make the relative error meaningless
  // Auxiliary fp64 bound. The bitwise check above is the strict one; W fp32 additions can each
  // round, so scale the one-add tolerance by the number of rank contributions.
  const double eps =
      (std::is_same<T, float>::value ? 0x1p-22 : (std::is_same<T, __half>::value ? 0x1p-10 : 0x1p-8)) *
      double(W);
  const std::vector<T> got0 = out[0].get();
  for (size_t i = 0; i < n; ++i) {
    double s = 0, a = 0;
    for (int p = 0; p < W; ++p) {
      const double x = double(kops_test::to_float(host_in[p][i]));
      s += x;
      a += std::fabs(x);
    }
    if (std::fabs(double(kops_test::to_float(got0[i])) - s) > eps * a + 1e-30) {
      kops_test::record_failure(tag, "differs from the fp64 sum by more than one rounding", __FILE__, __LINE__);
      break;
    }
  }
}

template <typename T>
void test_allreduce_family(bool host) {
  const std::vector<int> worlds = host ? std::vector<int>{2} : std::vector<int>{2, 4, 8};
  const std::vector<size_t> sizes = [&] {
    std::vector<size_t> v = {8, 64, 1032, 4096 + 8 * 13, 65536};
    if (kops_test::tier_enabled(g_opt, "standard")) v.push_back(size_t(1) << 20);
    if (host && kops_test::tier_enabled(g_opt, "full")) v.push_back(size_t(8) << 20);
    return v;
  }();
  for (int W : worlds) {
    World w;
    const size_t cap = 2 * (sizes.back() * sizeof(T) + 128);
    if (!w.init(W, host, cap, 0)) return;
    for (AllReduceAlgo algo : {AllReduceAlgo::kOneShot, AllReduceAlgo::kTwoShot, AllReduceAlgo::kDeterministic}) {
      for (size_t n : sizes) {
        const size_t epv = 16 / sizeof(T);
        const size_t nn = (n / epv) * epv;
        char tag[128];
        std::snprintf(tag, sizeof(tag), "allreduce %s %s W=%d n=%zu %s", host ? "host" : "virtual",
                      algo_name(algo), W, nn, kops::dtype_name(dtype_of<T>()));
        run_allreduce_case<T>(w, algo, nn, 11, tag);
      }
    }
    w.destroy();
  }
}

template <typename T>
void run_allgather_case(World& w, size_t n, unsigned seed, const char* tag) {
  const int W = w.size;
  std::vector<std::vector<T>> host_in(W);
  std::vector<RankBuf<T>> in(W), out(W);
  for (int k = 0; k < W; ++k) {
    host_in[k] = random_vec<T>(n, seed * 17 + k * 3 + 5);
    in[k].alloc(w.r[k].device, n);
    out[k].alloc(w.r[k].device, n * W);
    in[k].put(host_in[k]);
    out[k].fill_bytes(0);
  }
  launch_ranks(w, iota_order(W), 0, [&](int k) {
    KOPS_EXPECT_OK(allgather(flat(out[k].p, dtype_of<T>(), n * W, w.r[k].device),
                             flat(in[k].p, dtype_of<T>(), n, w.r[k].device), w.r[k].peers, 0, w.r[k].ctx),
                   tag);
  });
  w.sync_all();
  std::vector<T> want;
  for (int p = 0; p < W; ++p) want.insert(want.end(), host_in[p].begin(), host_in[p].end());
  for (int k = 0; k < W; ++k) {
    std::vector<T> got = out[k].get();
    if (k == 0) maybe_mutate(&got);
    char what[160];
    std::snprintf(what, sizeof(what), "%s rank %d", tag, k);
    expect_bits(got, want, what);
  }
}

template <typename T>
void run_reduce_scatter_case(World& w, size_t m, unsigned seed, const char* tag) {
  const int W = w.size;
  const size_t n = m * W;
  std::vector<std::vector<T>> host_in(W);
  std::vector<RankBuf<T>> in(W), out(W);
  for (int k = 0; k < W; ++k) {
    host_in[k] = random_vec<T>(n, seed * 13 + k * 5 + 2);
    in[k].alloc(w.r[k].device, n);
    out[k].alloc(w.r[k].device, m);
    in[k].put(host_in[k]);
    out[k].fill_bytes(0);
  }
  launch_ranks(w, iota_order(W), 0, [&](int k) {
    KOPS_EXPECT_OK(reduce_scatter(flat(out[k].p, dtype_of<T>(), m, w.r[k].device),
                                  flat(in[k].p, dtype_of<T>(), n, w.r[k].device), w.r[k].peers, 0, w.r[k].ctx),
                   tag);
  });
  w.sync_all();
  for (int k = 0; k < W; ++k) {
    const std::vector<T> want = sum_ranks<T>(host_in, size_t(k) * m, m);
    std::vector<T> got = out[k].get();
    if (k == 0) maybe_mutate(&got);
    char what[160];
    std::snprintf(what, sizeof(what), "%s rank %d", tag, k);
    expect_bits(got, want, what);
  }
}

template <typename T>
void test_gather_scatter_family(bool host) {
  const std::vector<int> worlds = host ? std::vector<int>{2} : std::vector<int>{2, 4, 8};
  const std::vector<size_t> per_rank = {8, 72, 4096, 16392, size_t(1) << 16};
  for (int W : worlds) {
    World w;
    const size_t cap = 2 * (per_rank.back() * W * sizeof(T) + 128);
    if (!w.init(W, host, cap, 0)) return;
    for (size_t n : per_rank) {
      const size_t epv = 16 / sizeof(T);
      const size_t nn = (n / epv) * epv;
      char tag[128];
      std::snprintf(tag, sizeof(tag), "allgather %s W=%d n=%zu %s", host ? "host" : "virtual", W, nn,
                    kops::dtype_name(dtype_of<T>()));
      run_allgather_case<T>(w, nn, 3, tag);
      std::snprintf(tag, sizeof(tag), "reduce_scatter %s W=%d m=%zu %s", host ? "host" : "virtual", W, nn,
                    kops::dtype_name(dtype_of<T>()));
      run_reduce_scatter_case<T>(w, nn, 4, tag);
    }
    w.destroy();
  }
}

// ======================================================================================================
// (d) system: arrival order, slot reuse across mixed calls, graph replay, timeout
// ======================================================================================================
/// The chunked producer/consumer pipeline (run_pipeline): a message above the single-phase threshold
/// runs one producer CTA plus kPipeConsumers consumer CTAs, and the producer posts one system-scope
/// release per chunk. Cover the switch-over size, an odd chunk count, a last chunk shorter than the
/// others, both staging-slot parities, the collectives that share the skeleton, and graph replay.
template <typename T>
void test_pipeline_chunked(bool host) {
  const int W = host ? 2 : 4;
  const size_t epv = 16 / sizeof(T);
  // 128 KiB + 16 B is the first size above the single-phase threshold (2 chunks, the last one a
  // single vector), 384 KiB gives an odd number of 128 KiB chunks, 1 MiB + 8 KiB a short last chunk,
  // 4 MiB the 16-chunk cap.
  std::vector<size_t> sizes = {(size_t(128) << 10) + 16, size_t(384) << 10, (size_t(1) << 20) + 8192};
  if (kops_test::tier_enabled(g_opt, "standard")) sizes.push_back(size_t(4) << 20);
  size_t max_elem = 0;
  for (size_t b : sizes) max_elem = std::max(max_elem, b / sizeof(T));
  const size_t cap = 2 * (max_elem * sizeof(T) + 128);
  const int reps = kops_test::tier_enabled(g_opt, "standard") ? 3 : 2;
  World w;
  if (!w.init(W, host, cap, 0)) return;
  for (size_t b : sizes) {
    const size_t n = ((b / sizeof(T)) / epv) * epv;
    for (int rep = 0; rep < reps; ++rep) {
      // alternating algorithms and repeats cover both staging-slot parities (the slot is epoch & 1)
      const AllReduceAlgo algo = rep % 3 == 0 ? AllReduceAlgo::kOneShot
                                : rep % 3 == 1 ? AllReduceAlgo::kTwoShot
                                               : AllReduceAlgo::kDeterministic;
      char tag[160];
      std::snprintf(tag, sizeof(tag), "chunked allreduce %s %s W=%d n=%zu %s", host ? "host" : "virtual",
                    algo_name(algo), W, n, kops::dtype_name(dtype_of<T>()));
      run_allreduce_case<T>(w, algo, n, 31 + rep, tag);
    }
  }
  // allgather and reduce_scatter share run_pipeline, so they also switch to the producer/consumer split
  const size_t per_rank = ((size_t(1) << 20) / sizeof(T) / epv) * epv;
  for (int rep = 0; rep < reps; ++rep) {
    char tag[160];
    std::snprintf(tag, sizeof(tag), "chunked allgather %s W=%d n=%zu", host ? "host" : "virtual", W, per_rank);
    run_allgather_case<T>(w, per_rank, 41 + rep, tag);
    std::snprintf(tag, sizeof(tag), "chunked reduce_scatter %s W=%d m=%zu", host ? "host" : "virtual", W,
                  per_rank);
    run_reduce_scatter_case<T>(w, per_rank, 43 + rep, tag);
  }
  {
    // Graph replay of a chunked call: the per-chunk flags are epoch-tagged, so a captured graph must
    // replay without any host update and every replay must reduce the inputs of that replay.
    const size_t n = ((size_t(1) << 20) / sizeof(T) / epv) * epv;
    const int calls = 3;
    std::vector<RankBuf<T>> in(W), out(W);
    std::vector<cudaGraph_t> graph(W);
    std::vector<cudaGraphExec_t> exec(W);
    for (int k = 0; k < W; ++k) {
      const int dev = w.r[k].device;
      in[k].alloc(dev, n);
      out[k].alloc(dev, n);
      out[k].fill_bytes(0);
      CUDA_OK(cudaSetDevice(dev));
      CUDA_OK(cudaStreamBeginCapture(w.r[k].stream, cudaStreamCaptureModeThreadLocal));
      for (int c = 0; c < calls; ++c) {
        KOPS_EXPECT_OK(allreduce(flat(out[k].p, dtype_of<T>(), n, dev), flat(in[k].p, dtype_of<T>(), n, dev),
                                 w.r[k].peers, AllReduceAlgo::kOneShot, 0, w.r[k].ctx),
                       "capture chunked allreduce");
      }
      CUDA_OK(cudaStreamEndCapture(w.r[k].stream, &graph[k]));
      CUDA_OK(cudaGraphInstantiate(&exec[k], graph[k], 0));
    }
    const int replays = kops_test::tier_enabled(g_opt, "standard") ? 4 : 2;
    for (int it = 0; it < replays; ++it) {
      std::vector<std::vector<T>> hin(W);
      for (int k = 0; k < W; ++k) {
        hin[k] = random_vec<T>(n, 700 + it * 11 + k);
        in[k].put(hin[k]);
      }
      for (int k = 0; k < W; ++k) {
        CUDA_OK(cudaSetDevice(w.r[k].device));
        CUDA_OK(cudaGraphLaunch(exec[k], w.r[k].stream));
      }
      w.sync_all();
      const std::vector<T> want = sum_ranks<T>(hin, 0, n);
      for (int k = 0; k < W; ++k) {
        char what[128];
        std::snprintf(what, sizeof(what), "chunked graph replay %d rank %d", it, k);
        expect_bits(out[k].get(), want, what);
      }
    }
    for (int k = 0; k < W; ++k) {
      CUDA_OK(cudaSetDevice(w.r[k].device));
      cudaGraphExecDestroy(exec[k]);
      cudaGraphDestroy(graph[k]);
    }
  }
  w.destroy();
}

void test_arrival_order_determinism(bool host) {
  const int W = host ? 2 : 4;
  World w;
  const size_t n = 4096;
  if (!w.init(W, host, 2 * (n * 2 + 128), 0)) return;
  std::mt19937 rng(7);
  std::vector<std::vector<__nv_bfloat16>> first;
  for (int iter = 0; iter < (kops_test::tier_enabled(g_opt, "standard") ? 24 : 6); ++iter) {
    std::vector<int> ord = iota_order(W);
    std::shuffle(ord.begin(), ord.end(), rng);
    const int stagger = int(rng() % 300);
    run_allreduce_case<__nv_bfloat16>(w, iter % 2 ? AllReduceAlgo::kTwoShot : AllReduceAlgo::kOneShot, n, 99,
                                      host ? "arrival order (host)" : "arrival order (virtual)", &ord, stagger);
  }
  w.destroy();
}

/// Mixed collectives back to back on one stream per rank with no synchronisation in between:
/// the staging slots are reused every second call, so this is the slot-reuse hazard test.
void test_mixed_sequence(bool host) {
  const int W = host ? 2 : 4;
  World w;
  const size_t max_n = 1 << 15;
  if (!w.init(W, host, 2 * (max_n * W * 2 + 128), 0)) return;
  using T = __nv_bfloat16;
  const int calls = kops_test::tier_enabled(g_opt, "standard") ? 120 : 30;
  std::mt19937 rng(2024);
  struct Call { int op; size_t n; };
  std::vector<Call> plan;
  for (int c = 0; c < calls; ++c) {
    const int op = int(rng() % 4);
    size_t n = (size_t(rng() % 2000) + 1) * 8;
    if (op == 2) n = std::min<size_t>(n, max_n / W) / 8 * 8 + 8;
    plan.push_back({op, n});
  }
  std::vector<std::vector<std::vector<T>>> host_in(calls, std::vector<std::vector<T>>(W));
  std::vector<std::vector<RankBuf<T>>> in(calls), out(calls);
  for (int c = 0; c < calls; ++c) {
    const Call& cl = plan[c];
    in[c] = std::vector<RankBuf<T>>(W);
    out[c] = std::vector<RankBuf<T>>(W);
    for (int k = 0; k < W; ++k) {
      size_t n_in = cl.n, n_out = cl.n;
      if (cl.op == 2) { n_in = cl.n * W; n_out = cl.n; }       // reduce_scatter: in = W * out
      if (cl.op == 1) { n_in = cl.n; n_out = cl.n * W; }       // allgather
      host_in[c][k] = random_vec<T>(n_in, c * 101 + k + 1);
      in[c][k].alloc(w.r[k].device, n_in);
      out[c][k].alloc(w.r[k].device, n_out);
      in[c][k].put(host_in[c][k]);
    }
  }
  for (int c = 0; c < calls; ++c) {
    const Call& cl = plan[c];
    for (int k = 0; k < W; ++k) {
      CUDA_OK(cudaSetDevice(w.r[k].device));
      const int dev = w.r[k].device;
      Status s = Status::kSuccess;
      if (cl.op == 0 || cl.op == 3) {
        s = allreduce(flat(out[c][k].p, DType::kBFloat16, cl.n, dev), flat(in[c][k].p, DType::kBFloat16, cl.n, dev),
                      w.r[k].peers, cl.op == 0 ? AllReduceAlgo::kOneShot : AllReduceAlgo::kTwoShot, 0, w.r[k].ctx);
      } else if (cl.op == 1) {
        s = allgather(flat(out[c][k].p, DType::kBFloat16, cl.n * W, dev), flat(in[c][k].p, DType::kBFloat16, cl.n, dev),
                      w.r[k].peers, 0, w.r[k].ctx);
      } else {
        s = reduce_scatter(flat(out[c][k].p, DType::kBFloat16, cl.n, dev),
                           flat(in[c][k].p, DType::kBFloat16, cl.n * W, dev), w.r[k].peers, 0, w.r[k].ctx);
      }
      KOPS_EXPECT_OK(s, "mixed sequence call");
    }
  }
  w.sync_all();
  for (int c = 0; c < calls; ++c) {
    const Call& cl = plan[c];
    for (int k = 0; k < W; ++k) {
      std::vector<T> want;
      if (cl.op == 0 || cl.op == 3) want = sum_ranks<T>(host_in[c], 0, cl.n);
      else if (cl.op == 1) for (int p = 0; p < W; ++p) want.insert(want.end(), host_in[c][p].begin(), host_in[c][p].end());
      else want = sum_ranks<T>(host_in[c], size_t(k) * cl.n, cl.n);
      char what[96];
      std::snprintf(what, sizeof(what), "mixed sequence call %d op %d n %zu rank %d", c, cl.op, cl.n, k);
      expect_bits(out[c][k].get(), want, what);
    }
  }
  int32_t err = 0;
  for (int k = 0; k < W; ++k) {
    CUDA_OK(cudaSetDevice(w.r[k].device));
    KOPS_EXPECT_OK(comm_error(w.r[k].peers, &err), "comm_error");
    KOPS_TEST_CHECK(err == 0);
  }
  w.destroy();
}

/// The epoch is advanced on the device: one captured graph replays many times with new data.
void test_graph_replay(bool host) {
  const int W = host ? 2 : 4;
  World w;
  const size_t n = 8192;
  using T = __nv_bfloat16;
  if (!w.init(W, host, 2 * (n * 2 * W + 128), 0)) return;
  std::vector<RankBuf<T>> in(W), out(W), gout(W), rout(W);
  std::vector<cudaGraph_t> graph(W);
  std::vector<cudaGraphExec_t> exec(W);
  for (int k = 0; k < W; ++k) {
    const int dev = w.r[k].device;
    in[k].alloc(dev, n * W);  // allreduce / allgather use the first n elements, reduce_scatter all of them
    out[k].alloc(dev, n);
    gout[k].alloc(dev, n * W);
    rout[k].alloc(dev, n);
    CUDA_OK(cudaSetDevice(dev));
    CUDA_OK(cudaStreamBeginCapture(w.r[k].stream, cudaStreamCaptureModeThreadLocal));
    // one graph = allreduce + allgather + reduce_scatter + barrier: four epochs per replay
    KOPS_EXPECT_OK(allreduce(flat(out[k].p, DType::kBFloat16, n, dev), flat(in[k].p, DType::kBFloat16, n, dev),
                             w.r[k].peers, AllReduceAlgo::kOneShot, 0, w.r[k].ctx), "capture allreduce");
    KOPS_EXPECT_OK(allgather(flat(gout[k].p, DType::kBFloat16, n * W, dev), flat(in[k].p, DType::kBFloat16, n, dev),
                             w.r[k].peers, 0, w.r[k].ctx), "capture allgather");
    KOPS_EXPECT_OK(reduce_scatter(flat(rout[k].p, DType::kBFloat16, n, dev),
                                  flat(in[k].p, DType::kBFloat16, n * W, dev), w.r[k].peers, 0, w.r[k].ctx),
                   "capture reduce_scatter");
    KOPS_EXPECT_OK(peer_barrier(w.r[k].peers, 0, w.r[k].ctx), "capture barrier");
    CUDA_OK(cudaStreamEndCapture(w.r[k].stream, &graph[k]));
    CUDA_OK(cudaGraphInstantiate(&exec[k], graph[k], 0));
  }
  const int replays = kops_test::tier_enabled(g_opt, "standard") ? 40 : 8;
  for (int it = 0; it < replays; ++it) {
    std::vector<std::vector<T>> hin(W);
    for (int k = 0; k < W; ++k) {
      hin[k] = random_vec<T>(n * W, 500 + it * 13 + k);
      in[k].put(hin[k]);
    }
    for (int k = 0; k < W; ++k) {
      CUDA_OK(cudaSetDevice(w.r[k].device));
      CUDA_OK(cudaGraphLaunch(exec[k], w.r[k].stream));
    }
    w.sync_all();
    const std::vector<T> want_ar = sum_ranks<T>(hin, 0, n);
    for (int k = 0; k < W; ++k) {
      char what[96];
      std::snprintf(what, sizeof(what), "graph replay %d allreduce rank %d", it, k);
      expect_bits(out[k].get(), want_ar, what);
      std::vector<T> want_ag;
      for (int p = 0; p < W; ++p) want_ag.insert(want_ag.end(), hin[p].begin(), hin[p].begin() + n);
      std::snprintf(what, sizeof(what), "graph replay %d allgather rank %d", it, k);
      std::vector<T> got_ag = gout[k].get();
      got_ag.resize(n * W);
      expect_bits(got_ag, want_ag, what);
      std::snprintf(what, sizeof(what), "graph replay %d reduce_scatter rank %d", it, k);
      expect_bits(rout[k].get(), sum_ranks<T>(hin, size_t(k) * n, n), what);
    }
  }
  for (int k = 0; k < W; ++k) {
    CUDA_OK(cudaSetDevice(w.r[k].device));
    cudaGraphExecDestroy(exec[k]);
    cudaGraphDestroy(graph[k]);
  }
  // eager calls after the graph keep working: the counters are shared
  run_allreduce_case<T>(w, AllReduceAlgo::kOneShot, n, 5, host ? "eager after graph (host)" : "eager after graph (virtual)");
  w.destroy();
}

/// A rank that never arrives must produce an error, not a hang; the communicator recovers after
/// comm_configure on every rank.
void test_late_rank_timeout(bool host) {
  const int W = host ? 2 : 4;
  World w;
  const size_t n = 2048;
  using T = __nv_bfloat16;
  if (!w.init(W, host, 2 * (n * 2 + 128), 50 * 1000)) return;  // 50 ms per wait
  std::vector<RankBuf<T>> in(W), out(W);
  for (int k = 0; k < W; ++k) {
    in[k].alloc(w.r[k].device, n);
    out[k].alloc(w.r[k].device, n);
    in[k].put(random_vec<T>(n, 77 + k));
    out[k].fill_bytes(0x5A);
  }
  // only rank 0 launches: it must give up
  CUDA_OK(cudaSetDevice(w.r[0].device));
  KOPS_EXPECT_OK(allreduce(flat(out[0].p, DType::kBFloat16, n, w.r[0].device),
                           flat(in[0].p, DType::kBFloat16, n, w.r[0].device), w.r[0].peers,
                           AllReduceAlgo::kOneShot, 0, w.r[0].ctx), "late rank launch");
  const auto t0 = std::chrono::steady_clock::now();
  CUDA_OK(cudaStreamSynchronize(w.r[0].stream));
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  KOPS_TEST_CHECK(ms < 5000.0);
  int32_t err = 0;
  KOPS_EXPECT_OK(comm_error(w.r[0].peers, &err), "comm_error after timeout");
  KOPS_TEST_CHECK(err == static_cast<int32_t>(CommError::kPeerTimeout));
  // the data phase was skipped
  const std::vector<T> got = out[0].get();
  const std::vector<uint8_t> raw(reinterpret_cast<const uint8_t*>(got.data()),
                                 reinterpret_cast<const uint8_t*>(got.data()) + 2 * n);
  KOPS_TEST_CHECK(std::all_of(raw.begin(), raw.end(), [](uint8_t b) { return b == 0x5A; }));
  // recover: every rank re-initialises its signal area, then a normal call succeeds
  w.reconfigure(0);
  run_allreduce_case<T>(w, AllReduceAlgo::kTwoShot, n, 8, host ? "after recovery (host)" : "after recovery (virtual)");
  for (int k = 0; k < W; ++k) {
    CUDA_OK(cudaSetDevice(w.r[k].device));
    KOPS_EXPECT_OK(comm_error(w.r[k].peers, &err), "comm_error after recovery");
    KOPS_TEST_CHECK(err == 0);
  }
  w.destroy();
}


// ======================================================================================================
// B12 allreduce + residual + RMSNorm (+ dynamic per-token FP8)
// ======================================================================================================
template <typename T>
void test_allreduce_fusion(bool host) {
  const int W = host ? 2 : 4;
  const int NT = 4, H = 64;
  const size_t cap = 2 * (size_t(NT) * H * 2 + 128);
  World w;
  if (!w.init(W, host, cap, 0)) return;
  std::vector<RankBuf<T>> input(W), residual(W), weight(W), norm_out(W);
  std::vector<RankBuf<uint8_t>> q(W);
  std::vector<RankBuf<float>> scales(W);
  std::vector<std::vector<T>> hinput(W), hres(W), hw(W);
  for (int k = 0; k < W; ++k) {
    hinput[k] = random_vec<T>(size_t(NT) * H, 600 + k);
    hres[k] = random_vec<T>(size_t(NT) * H, 700 + k);
    hw[k] = random_vec<T>(H, 800 + k, 1.5f);
    input[k].alloc(w.r[k].device, size_t(NT) * H);
    residual[k].alloc(w.r[k].device, size_t(NT) * H);
    weight[k].alloc(w.r[k].device, H);
    norm_out[k].alloc(w.r[k].device, size_t(NT) * H);
    input[k].put(hinput[k]);
    residual[k].put(hres[k]);
    weight[k].put(hw[k]);
    norm_out[k].fill_bytes(0x5A);
  }
  auto run = [&](bool quant, uint32_t epoch) {
    // Prepare every rank before launching any kernel: RankBuf::put() synchronises the device, and
    // a synchronise between two rank launches would deadlock the first rank's collective.
    for (int k = 0; k < W; ++k) {
      residual[k].put(hres[k]);  // each call starts from the same residual stream
      if (quant) {
        q[k].alloc(w.r[k].device, size_t(NT) * H);
        scales[k].alloc(w.r[k].device, NT);
        q[k].fill_bytes(0x5A);
      }
    }
    for (int k = 0; k < W; ++k) {
      AllReduceFusionParams p{};
      p.input = input[k].p;
      p.residual_inout = residual[k].p;
      p.norm_weight = weight[k].p;
      p.norm_eps = 1e-5f;
      p.weight_offset = 0.0f;
      p.norm_out = norm_out[k].p;
      p.quant = ActQuantOut{};
      if (quant) {
        p.quant.mode = ActQuantMode::kDynamicPerToken;
        p.quant.q_dtype = DType::kFloat8E4M3;
        p.quant.fp8 = Fp8Encoding::kOCP;
        p.quant.symmetric = true;
        p.quant.q = q[k].p;
        p.quant.q_row_stride = H;
        p.quant.scales = scales[k].p;
      }
      p.peers = w.r[k].peers;
      p.epoch = epoch;
      p.num_tokens = NT;
      p.hidden = H;
      KOPS_EXPECT_OK(allreduce_fusion(p, AllReduceAlgo::kOneShot, dtype_of<T>(), w.r[k].ctx),
                     "allreduce_fusion");
    }
    w.sync_all();
  };
  auto verify = [&](bool quant, const char* what) {
    for (int k = 0; k < W; ++k) {
      std::vector<T> got = norm_out[k].get();
      const std::vector<T> res = residual[k].get();
      if (k == 0) maybe_mutate(&got);
      const std::vector<uint8_t> hq = quant ? q[k].get() : std::vector<uint8_t>();
      const std::vector<float> hs = quant ? scales[k].get() : std::vector<float>();
      for (int r = 0; r < NT; ++r) {
        float ss = 0.0f;
        for (int c = 0; c < H; ++c) {
          float x = kops_test::to_float(hinput[0][size_t(r) * H + c]);
          for (int p = 1; p < W; ++p) x += kops_test::to_float(hinput[p][size_t(r) * H + c]);
          // allreduce_fusion reduces to T first (the collective's one rounding), then adds the
          // residual in fp32; the sum of squares uses that unrounded residual value.
          const float xr = kops_test::to_float(round_to<T>(x));
          const float rv = kops_test::to_float(hres[k][size_t(r) * H + c]) + xr;
          const T rounded = round_to<T>(rv);
          KOPS_TEST_CHECK(bits_of(res[size_t(r) * H + c]) == bits_of(rounded));
          ss += rv * rv;
        }
        const float rstd = rsqrtf(ss / float(H) + 1e-5f);
        if (quant) {
          // The kernel writes the scale of the row and quantizes the rounded T value with that
          // scale. Recompute the same amax/scale and check the dequantized value against the row.
          std::vector<float> nv(H);
          float amax = 0.0f;
          for (int c = 0; c < H; ++c) {
            const float rv = kops_test::to_float(res[size_t(r) * H + c]);
            nv[c] = rv * rstd * kops_test::to_float(hw[k][c]);
            amax = std::max(amax, std::fabs(nv[c]));
          }
          const float scale_want = std::max(amax / 448.0f, 1.0f / (448.0f * 512.0f));
          if (std::fabs(hs[r] - scale_want) > 1e-4f * scale_want + 1e-30f) {
            kops_test::record_failure(what, "quant per-token scale", __FILE__, __LINE__);
            return;
          }
          for (int c = 0; c < H; ++c) {
            const uint8_t q_want = static_cast<uint8_t>(
                __nv_cvt_float_to_fp8(nv[c] / hs[r], __NV_SATFINITE, __NV_E4M3));
            if (hq[size_t(r) * H + c] != q_want) {
              kops_test::record_failure(what, "quantized normalized output", __FILE__, __LINE__);
              return;
            }
          }
        } else {
          for (int c = 0; c < H; ++c) {
            const float rv = kops_test::to_float(res[size_t(r) * H + c]);
            const float n = rv * rstd * kops_test::to_float(hw[k][c]);
            if (std::fabs(kops_test::to_float(got[size_t(r) * H + c]) - n) >
                4e-2 + 4e-2 * std::fabs(n)) {
              kops_test::record_failure(what, "normalized output", __FILE__, __LINE__);
              return;
            }
          }
        }
      }
    }
  };
  run(false, 0);
  verify(false, "allreduce_fusion bf16");
  // second call exercises a different epoch/slot parity
  run(false, 1);
  verify(false, "allreduce_fusion bf16 replay");
  run(true, 2);
  verify(true, "allreduce_fusion dynamic per-token fp8");
  // contract: hidden not a multiple of the two-vector step, null output, unsupported quant
  {
    AllReduceFusionParams p{};
    p.input = input[0].p;
    p.residual_inout = residual[0].p;
    p.norm_weight = weight[0].p;
    p.norm_out = norm_out[0].p;
    p.peers = w.r[0].peers;
    p.num_tokens = 1;
    p.hidden = 7;
    p.norm_eps = 1e-5f;
    KOPS_EXPECT_STATUS(allreduce_fusion(p, AllReduceAlgo::kOneShot, dtype_of<T>(), w.r[0].ctx),
                       Status::kUnsupportedShape, "fusion hidden step");
    p.hidden = H;
    p.norm_out = nullptr;
    KOPS_EXPECT_STATUS(allreduce_fusion(p, AllReduceAlgo::kOneShot, dtype_of<T>(), w.r[0].ctx),
                       Status::kInvalidArgument, "fusion missing output");
    p.norm_out = norm_out[0].p;
    p.quant.mode = ActQuantMode::kMXFP8;
    KOPS_EXPECT_STATUS(allreduce_fusion(p, AllReduceAlgo::kOneShot, dtype_of<T>(), w.r[0].ctx),
                       Status::kNotImplemented, "fusion unsupported quant");
  }
  w.destroy();
}

// ======================================================================================================
// B12 EP: layout, normal dispatch/combine, low-latency masked dispatch/combine, hook mode
// ======================================================================================================
EPBuffers ep_buffers(World& w, int rank) {
  EPBuffers b{};
  for (int j = 0; j < w.size; ++j) b.nvl_buffers[j] = w.r[rank].peers.data[j];
  b.signals = w.signals;
  b.rank = rank;
  b.num_ranks = w.size;
  b.num_local_ranks = w.size;
  b.transport = EPTransport::kNvlink;
  b.nvl_bytes = w.buffer_bytes;
  return b;
}

size_t ep_dispatch_rec(int hidden, int topk) {
  return ((16 + size_t(topk) * 8 + size_t(hidden) * 2 + 15) / 16) * 16;
}
size_t ep_combine_rec(int hidden) { return ((16 + size_t(hidden) * 2 + 15) / 16) * 16; }

template <typename T>
struct EpInputs {
  std::vector<std::vector<T>> x;
  std::vector<std::vector<int32_t>> topk;
  std::vector<std::vector<float>> w;
  std::vector<std::vector<uint8_t>> member;  // [rank][token * W + dest]
};

template <typename T>
EpInputs<T> make_ep_inputs(int W, int tokens, int hidden, int topk, int experts, unsigned seed) {
  EpInputs<T> in;
  in.x.resize(W);
  in.topk.resize(W);
  in.w.resize(W);
  in.member.resize(W);
  const int e_local = experts / W;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-3.0f, 3.0f);
  for (int k = 0; k < W; ++k) {
    in.x[k] = random_vec<T>(size_t(tokens) * hidden, seed * 17 + k);
    in.topk[k].resize(size_t(tokens) * topk);
    in.w[k].resize(size_t(tokens) * topk);
    in.member[k].assign(size_t(tokens) * W, 0);
    for (int t = 0; t < tokens; ++t) {
      for (int j = 0; j < topk; ++j) {
        int e = (t * 3 + j * 2 + k) % experts;
        if (((t + j + k) % 13) == 0) e = -1;  // dropped assignment
        in.topk[k][size_t(t) * topk + j] = e;
        in.w[k][size_t(t) * topk + j] = d(rng);
        if (e >= 0) in.member[k][size_t(t) * W + e / e_local] = 1;
      }
    }
  }
  return in;
}

void test_ep_layout() {
  const int W = 2, E = 4, T = 16, K = 2;
  World w;
  if (!w.init(W, false, 1 << 14, 0)) return;
  EpInputs<__nv_bfloat16> in = make_ep_inputs<__nv_bfloat16>(W, T, 8, K, E, 9);
  RankBuf<int32_t> topk, per_rank, per_expert;
  RankBuf<uint8_t> member;
  topk.alloc(0, T * K);
  per_rank.alloc(0, W);
  per_expert.alloc(0, E);
  member.alloc(0, T * W);
  topk.put(in.topk[0]);
  std::vector<uint8_t> host_member(T * W);
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < W; ++r) host_member[t * W + r] = in.member[0][t * W + r];
  member.put(host_member);
  Tensor tt = kops_test::tensor2d(topk.p, DType::kInt32, T, K);
  Tensor tr = flat(per_rank.p, DType::kInt32, W);
  Tensor te = flat(per_expert.p, DType::kInt32, E);
  Tensor tm = flat(member.p, DType::kUInt8, T * W);
  KOPS_EXPECT_OK(ep_dispatch_layout(tr, te, tm, tt, E, W, w.r[0].ctx), "ep_dispatch_layout");
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<int32_t> want_rank(W, 0), want_expert(E, 0);
  for (int t = 0; t < T; ++t) {
    uint32_t mask = 0;
    for (int j = 0; j < K; ++j) {
      const int e = in.topk[0][t * K + j];
      if (e < 0) continue;
      ++want_expert[e];
      mask |= 1u << (e / (E / W));
    }
    for (int r = 0; r < W; ++r)
      if (mask & (1u << r)) ++want_rank[r];
  }
  for (int r = 0; r < W; ++r) KOPS_TEST_CHECK(per_rank.get()[r] == want_rank[r]);
  for (int e = 0; e < E; ++e) KOPS_TEST_CHECK(per_expert.get()[e] == want_expert[e]);
  const std::vector<uint8_t> got_member = member.get();
  for (size_t i = 0; i < got_member.size(); ++i) KOPS_TEST_CHECK(got_member[i] == host_member[i]);
  w.destroy();
}

template <typename T>
void test_ep_normal(bool hook) {
  const int W = 2, NT = 12, H = 32, K = 2, E = 4;
  const int e_local = E / W;
  const int cap_rows = NT * W;
  const size_t rec = ep_dispatch_rec(H, K);
  const size_t crec = ep_combine_rec(H);
  const size_t need_dispatch = size_t(W) * ((size_t(NT) * rec + 127) & ~size_t(127));
  const size_t need_combine = size_t(W) * ((size_t(cap_rows) * crec + 127) & ~size_t(127));
  const size_t nvl = std::max(need_dispatch, need_combine);
  World w;
  if (!w.init(W, false, nvl, 0)) return;
  ep_buffers(w, 0);  // validates layout
  CUDA_OK(cudaMemset(w.signals, 0, sizeof(uint32_t) * 2 * W));
  CUDA_OK(cudaDeviceSynchronize());
  EpInputs<T> in = make_ep_inputs<T>(W, NT, H, K, E, 21);
  std::vector<RankBuf<T>> x(W), recv_x(W);
  std::vector<RankBuf<int64_t>> recv_src(W);
  std::vector<RankBuf<int32_t>> recv_idx(W);
  std::vector<RankBuf<float>> recv_w(W);
  std::vector<RankBuf<T>> expert_out(W), out(W), residual(W);
  std::vector<RankBuf<int32_t>> recv_cnt(W);
  std::vector<RankBuf<int32_t>> topk_dev(W);
  std::vector<RankBuf<float>> w_dev(W);
  std::vector<RankBuf<uint8_t>> member_dev(W);
  std::vector<std::vector<uint8_t>> member(W);
  for (int k = 0; k < W; ++k) {
    x[k].alloc(0, size_t(NT) * H);
    x[k].put(in.x[k]);
    topk_dev[k].alloc(0, in.topk[k].size());
    topk_dev[k].put(in.topk[k]);
    w_dev[k].alloc(0, in.w[k].size());
    w_dev[k].put(in.w[k]);
    member[k].assign(size_t(NT) * W, 0);
    for (int t = 0; t < NT; ++t)
      for (int r = 0; r < W; ++r) member[k][size_t(t) * W + r] = in.member[k][size_t(t) * W + r];
    member_dev[k].alloc(0, member[k].size());
    member_dev[k].put(member[k]);
    recv_x[k].alloc(0, size_t(cap_rows) * H);
    recv_x[k].fill_bytes(0x5A);
    recv_src[k].alloc(0, cap_rows);
    recv_src[k].fill_bytes(0xFF);  // -1
    recv_idx[k].alloc(0, size_t(cap_rows) * K);
    recv_w[k].alloc(0, size_t(cap_rows) * K);
    recv_idx[k].fill_bytes(0x5A);
    recv_w[k].fill_bytes(0x5A);
    recv_cnt[k].alloc(0, e_local);
  }
  const uint32_t epoch = 3;
  auto dispatch_rank = [&](int k, int phase) {
    EPDispatchParams p{};
    p.x = x[k].p;
    p.topk_idx = topk_dev[k].p;
    p.topk_weights = w_dev[k].p;
    p.is_token_in_rank = member_dev[k].p;
    p.recv_x = recv_x[k].p;
    p.recv_topk_idx = recv_idx[k].p;
    p.recv_topk_weights = recv_w[k].p;
    p.recv_count_per_expert = recv_cnt[k].p;
    p.recv_src_info = recv_src[k].p;
    p.buffers = ep_buffers(w, k);
    p.num_tokens = NT;
    p.hidden = H;
    p.topk = K;
    p.num_experts = E;
    p.max_tokens_per_rank = cap_rows;
    p.epoch = epoch;
    p.low_latency = false;
    p.hook_mode = hook;
    p.phase = phase;
    KOPS_EXPECT_OK(ep_dispatch(p, DType::kBFloat16, w.r[k].ctx), "ep_dispatch normal");
  };
  for (int k = 0; k < W; ++k) dispatch_rank(k, 0);
  if (hook) {
    w.sync_all();
    CUDA_OK(cudaSetDevice(0));
    for (int k = 0; k < W; ++k) dispatch_rank(k, 1);
  }
  w.sync_all();

  struct RowRef { int src, token, row; };
  std::vector<std::vector<RowRef>> map(W, std::vector<RowRef>(size_t(NT) * W, {-1, -1, -1}));
  for (int d = 0; d < W; ++d) {
    int row = 0;
    for (int s = 0; s < W; ++s)
      for (int t = 0; t < NT; ++t) {
        if (in.member[s][size_t(t) * W + d] == 0) continue;
        map[s][size_t(d) * NT + t] = {s, t, row};
        ++row;
      }
    const std::vector<T> gotx = recv_x[d].get();
    const std::vector<int64_t> gotsrc = recv_src[d].get();
    const std::vector<int32_t> gotidx = recv_idx[d].get();
    const std::vector<float> gotw = recv_w[d].get();
    int rr = 0;
    for (int s = 0; s < W; ++s)
      for (int t = 0; t < NT; ++t) {
        if (in.member[s][size_t(t) * W + d] == 0) continue;
        const int64_t want_src = (int64_t(s) << 32) | uint32_t(t);
        KOPS_TEST_CHECK(gotsrc[rr] == want_src);
        for (int j = 0; j < K; ++j) {
          const int e = in.topk[s][size_t(t) * K + j];
          int l = -1;
          float wv = 0.0f;
          if (e >= 0 && e / e_local == d) { l = e - d * e_local; wv = in.w[s][size_t(t) * K + j]; }
          KOPS_TEST_CHECK(gotidx[size_t(rr) * K + j] == l);
          KOPS_TEST_CHECK(gotw[size_t(rr) * K + j] == wv);
        }
        for (int c = 0; c < H; ++c) {
          if (bits_of(gotx[size_t(rr) * H + c]) != bits_of(in.x[s][size_t(t) * H + c])) {
            kops_test::record_failure("ep dispatch normal payload", "payload mismatch", __FILE__, __LINE__);
            c = H;
            t = NT;
            s = W;
          }
        }
        ++rr;
      }
  }

  // Combine: each expert rank returns a deterministic row; the source rank sums the contributions
  // in source-rank order and applies the local weight sum.
  CUDA_OK(cudaSetDevice(0));
  for (int k = 0; k < W; ++k) {
    expert_out[k].alloc(0, size_t(cap_rows) * H);
    out[k].alloc(0, size_t(cap_rows) * H);
    residual[k].alloc(0, size_t(cap_rows) * H);
    std::vector<T> eo(size_t(cap_rows) * H), res(size_t(cap_rows) * H);
    for (size_t i = 0; i < eo.size(); ++i) {
      eo[i] = round_to<T>(0.01f * float(i % 97) + 0.25f * float(k + 1));
      res[i] = round_to<T>(0.02f * float(i % 31) - 0.3f);
    }
    expert_out[k].put(eo);
    residual[k].put(res);
    out[k].fill_bytes(0x5A);
  }
  (void)crec;
  CUDA_OK(cudaMemset(w.signals, 0, sizeof(uint32_t) * 2 * W));
  CUDA_OK(cudaDeviceSynchronize());
  const uint32_t cepoch = 4;
  auto combine_rank = [&](int k, int phase) {
    EPCombineParams p{};
    p.expert_out = expert_out[k].p;
    p.topk_weights = recv_w[k].p;
    p.weights_already_applied = false;
    p.src_info = recv_src[k].p;
    p.buffers = ep_buffers(w, k);
    p.out = out[k].p;
    p.residual = residual[k].p;
    p.num_tokens = cap_rows;
    p.hidden = H;
    p.topk = K;
    p.epoch = cepoch;
    p.low_latency = false;
    p.hook_mode = hook;
    p.phase = phase;
    KOPS_EXPECT_OK(ep_combine(p, DType::kBFloat16, w.r[k].ctx), "ep_combine normal");
  };
  for (int k = 0; k < W; ++k) combine_rank(k, 0);
  if (hook) {
    w.sync_all();
    CUDA_OK(cudaSetDevice(0));
    for (int k = 0; k < W; ++k) combine_rank(k, 1);
  }
  w.sync_all();

  std::vector<std::vector<T>> ho(W), hres(W);
  std::vector<std::vector<float>> hw(W);
  for (int s = 0; s < W; ++s) {
    ho[s] = expert_out[s].get();
    hw[s] = recv_w[s].get();
    hres[s] = residual[s].get();
  }

  for (int d = 0; d < W; ++d) {
    std::vector<T> got = out[d].get();
    if (d == 0) maybe_mutate(&got);
    for (int t = 0; t < NT; ++t) {
      for (int c = 0; c < H; ++c) {
        double want = kops_test::to_float(hres[d][size_t(t) * H + c]);
        for (int s = 0; s < W; ++s) {
          const RowRef& ref = map[d][size_t(s) * NT + t];
          if (ref.row < 0) continue;
          double wsum = 0.0;
          for (int j = 0; j < K; ++j) wsum += hw[s][size_t(ref.row) * K + j];
          want += double(kops_test::to_float(ho[s][size_t(ref.row) * H + c])) * wsum;
        }
        const double g = kops_test::to_float(got[size_t(t) * H + c]);
        if (std::fabs(g - want) > 3e-2 + 3e-2 * std::fabs(want)) {
          char msg[160];
          std::snprintf(msg, sizeof(msg), "ep combine d=%d t=%d c=%d got %.6g want %.6g", d, t, c, g,
                        want);
          kops_test::record_failure("ep combine normal", msg, __FILE__, __LINE__);
          c = H;
          t = NT;
        }
      }
    }
  }
  w.destroy();
}

template <typename T>
void test_ep_low_latency() {
  const int W = 1, NT = 8, H = 16, K = 2, E = 4;
  const int cap = 8;  // masked rows per local expert (capacity)
  const size_t rec = ep_dispatch_rec(H, K);
  const size_t nvl = (size_t(NT) * rec + 127) & ~size_t(127);
  World w;
  if (!w.init(W, false, nvl, 0)) return;
  CUDA_OK(cudaMemset(w.signals, 0, sizeof(uint32_t) * 2 * W));
  CUDA_OK(cudaDeviceSynchronize());
  EpInputs<T> in = make_ep_inputs<T>(W, NT, H, K, E, 31);
  RankBuf<T> x, recv_x;
  RankBuf<int64_t> recv_src;
  RankBuf<int32_t> recv_cnt, recv_idx;
  RankBuf<float> recv_w;
  RankBuf<int32_t> topk_dev;
  RankBuf<float> w_dev;
  RankBuf<uint8_t> member_dev;
  x.alloc(0, size_t(NT) * H);
  x.put(in.x[0]);
  topk_dev.alloc(0, in.topk[0].size());
  topk_dev.put(in.topk[0]);
  w_dev.alloc(0, in.w[0].size());
  w_dev.put(in.w[0]);
  member_dev.alloc(0, in.member[0].size());
  member_dev.put(in.member[0]);
  recv_x.alloc(0, size_t(E) * cap * H);
  recv_x.fill_bytes(0x5A);
  recv_src.alloc(0, size_t(E) * cap);
  recv_src.fill_bytes(0xFF);
  recv_w.alloc(0, size_t(E) * cap);
  recv_w.fill_bytes(0x5A);
  recv_cnt.alloc(0, E);
  recv_cnt.fill_bytes(0x5A);
  recv_idx.alloc(0, size_t(E) * cap * K);
  EPDispatchParams p{};
  p.x = x.p;
  p.topk_idx = topk_dev.p;
  p.topk_weights = w_dev.p;
  p.is_token_in_rank = member_dev.p;
  p.recv_x = recv_x.p;
  p.recv_topk_idx = recv_idx.p;
  p.recv_topk_weights = recv_w.p;
  p.recv_count_per_expert = recv_cnt.p;
  p.recv_src_info = recv_src.p;
  p.buffers = ep_buffers(w, 0);
  p.num_tokens = NT;
  p.hidden = H;
  p.topk = K;
  p.num_experts = E;
  p.max_tokens_per_rank = cap;
  p.epoch = 5;
  p.low_latency = true;
  p.hook_mode = false;
  KOPS_EXPECT_OK(ep_dispatch(p, DType::kBFloat16, w.r[0].ctx), "ep_dispatch ll");
  w.sync_all();
  const std::vector<int32_t> cnt = recv_cnt.get();
  const std::vector<int64_t> src = recv_src.get();
  const std::vector<float> wv = recv_w.get();
  const std::vector<T> rx = recv_x.get();
  std::vector<int> expect_cnt(E, 0);
  std::vector<int> slot(E, 0);
  for (int t = 0; t < NT; ++t)
    for (int j = 0; j < K; ++j) {
      const int e = in.topk[0][size_t(t) * K + j];
      if (e < 0) continue;
      const int s = slot[e]++;
      const int64_t want_src = (int64_t(0) << 32) | uint32_t(t);
      if (s < cap) {
        KOPS_TEST_CHECK(src[size_t(e) * cap + s] == want_src);
        KOPS_TEST_CHECK(wv[size_t(e) * cap + s] == in.w[0][size_t(t) * K + j]);
        for (int c = 0; c < H; ++c)
          KOPS_TEST_CHECK(bits_of(rx[(size_t(e) * cap + s) * H + c]) ==
                          bits_of(in.x[0][size_t(t) * H + c]));
      }
      ++expect_cnt[e];
    }
  for (int e = 0; e < E; ++e) KOPS_TEST_CHECK(cnt[e] == std::min(expect_cnt[e], cap));

  // combine in the same masked layout
  RankBuf<T> expert_out, out;
  expert_out.alloc(0, size_t(E) * cap * H);
  out.alloc(0, size_t(E) * cap * H);
  std::vector<T> eo(size_t(E) * cap * H);
  for (size_t i = 0; i < eo.size(); ++i) eo[i] = round_to<T>(0.01f * float(i % 53) + 0.1f);
  expert_out.put(eo);
  CUDA_OK(cudaMemset(w.signals, 0, sizeof(uint32_t) * 2 * W));
  CUDA_OK(cudaDeviceSynchronize());
  EPCombineParams cp{};
  cp.expert_out = expert_out.p;
  cp.topk_weights = recv_w.p;
  cp.weights_already_applied = false;
  cp.src_info = recv_src.p;
  cp.buffers = ep_buffers(w, 0);
  cp.out = out.p;
  cp.residual = nullptr;
  cp.num_tokens = E * cap;
  cp.hidden = H;
  cp.topk = K;
  cp.epoch = 6;
  cp.low_latency = true;
  cp.hook_mode = false;
  KOPS_EXPECT_OK(ep_combine(cp, DType::kBFloat16, w.r[0].ctx), "ep_combine ll");
  w.sync_all();
  const std::vector<T> got = out.get();
  for (int t = 0; t < NT; ++t) {
    for (int c = 0; c < H; ++c) {
      double want = 0.0;
      for (int e = 0; e < E; ++e)
        for (int s = 0; s < cap; ++s) {
          const int64_t si = src[size_t(e) * cap + s];
          if (si < 0 || (si & 0xFFFFFFFFu) != uint32_t(t)) continue;
          want += double(kops_test::to_float(eo[(size_t(e) * cap + s) * H + c])) * wv[size_t(e) * cap + s];
        }
      const double g = kops_test::to_float(got[size_t(t) * H + c]);
      if (std::fabs(g - want) > 3e-2 + 3e-2 * std::fabs(want)) {
        char msg[160];
        std::snprintf(msg, sizeof(msg), "ep combine ll t=%d c=%d got %.6g want %.6g", t, c, g, want);
        kops_test::record_failure("ep combine ll", msg, __FILE__, __LINE__);
        c = H;
        t = NT;
      }
    }
  }
  w.destroy();
}

// ======================================================================================================
// B12 comm-GEMM: overlapped allgather / reduce-scatter / allreduce
// ======================================================================================================
PeerBuffers mg_peers(World& w, int rank) { return w.r[rank].peers; }

template <typename T>
std::vector<double> mm_ref(const std::vector<T>& A, int M, int K, const std::vector<T>& B, int N) {
  std::vector<double> D(size_t(M) * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += double(kops_test::to_float(A[size_t(m) * K + k])) * double(kops_test::to_float(B[size_t(n) * K + k]));
      D[size_t(m) * N + n] = acc;
    }
  return D;
}

template <typename T>
void test_comm_gemm_impl() {
  const int W = 2, N = 256, K = 32, M_local = 32, M = 64;
  const int chunks = 2;  // chunk_rows 32, 8 N tiles -> group 8 (the benchmark's CTA group)
  World wa;
  if (!wa.init(W, false, 2 * (size_t(M_local) * K * 2 + 256), 0)) return;
  CUDA_OK(cudaMemset(wa.signals, 0, sizeof(uint32_t) * 2 * W));
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<RankBuf<T>> a(W), b(W), d(W);
  std::vector<std::vector<T>> ha(W), hb(W);
  for (int k = 0; k < W; ++k) {
    ha[k] = random_vec<T>(size_t(M_local) * K, 100 + k);
    hb[k] = random_vec<T>(size_t(N) * K, 200 + k);
    a[k].alloc(0, ha[k].size());
    b[k].alloc(0, hb[k].size());
    d[k].alloc(0, size_t(M) * N);
    a[k].put(ha[k]);
    b[k].put(hb[k]);
    d[k].fill_bytes(0x5A);
  }
  for (int k = 0; k < W; ++k) {
    CommGemmParams p{};
    p.gemm.args.a.data = a[k].p;
    p.gemm.args.a.ld = K;
    p.gemm.args.a.dtype = DType::kBFloat16;
    p.gemm.args.b.data = b[k].p;
    p.gemm.args.b.ld = K;
    p.gemm.args.b.dtype = DType::kBFloat16;
    p.gemm.args.d = d[k].p;
    p.gemm.args.ldd = N;
    p.gemm.args.d_dtype = DType::kBFloat16;
    p.gemm.args.m = M_local;
    p.gemm.args.n = N;
    p.gemm.args.k = K;
    p.gemm.args.recipe = kops::gemm::GemmRecipe::kBF16;
    p.peers = mg_peers(wa, k);
    p.num_chunks = chunks;
    p.chunk_ready = wa.signals;
    p.epoch = 0;
    KOPS_EXPECT_OK(allgather_gemm(p, kops::gemm::GemmRecipe::kBF16, wa.r[k].ctx), "allgather_gemm");
  }
  wa.sync_all();
  std::vector<T> a_full;
  for (int k = 0; k < W; ++k) a_full.insert(a_full.end(), ha[k].begin(), ha[k].end());
  for (int k = 0; k < W; ++k) {
    std::vector<T> got = d[k].get();
    if (k == 0) maybe_mutate(&got);
    const std::vector<double> want = mm_ref<T>(a_full, M, K, hb[k], N);
    kops_test::compare(got, want, 2e-2, 2e-2, "allgather_gemm");
  }
  wa.destroy();

  // reduce_scatter and allreduce need fp32 partial staging: M * N * 4 bytes per slot.
  World wr;
  const size_t stage = 2 * (size_t(M) * N * 4 + 256);
  if (!wr.init(W, false, stage, 0)) return;
  CUDA_OK(cudaMemset(wr.signals, 0, sizeof(uint32_t) * 2 * W));
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<RankBuf<T>> ra(W), rb(W), rout(W), rar(W), rplain(W), rres(W), rw(W);
  std::vector<std::vector<T>> hra(W), hrb(W);
  for (int k = 0; k < W; ++k) {
    hra[k] = random_vec<T>(size_t(M) * K, 300 + k);
    hrb[k] = random_vec<T>(size_t(N) * K, 400 + k);
    ra[k].alloc(0, hra[k].size());
    rb[k].alloc(0, hrb[k].size());
    rout[k].alloc(0, size_t(M / W) * N);
    rar[k].alloc(0, size_t(M) * N);
    rplain[k].alloc(0, size_t(M) * N);
    rres[k].alloc(0, size_t(M) * N);
    rw[k].alloc(0, N);
    ra[k].put(hra[k]);
    rb[k].put(hrb[k]);
    rout[k].fill_bytes(0x5A);
    rar[k].fill_bytes(0x5A);
    rplain[k].fill_bytes(0x5A);
    std::vector<T> res(size_t(M) * N), wt(N);
    for (size_t i = 0; i < res.size(); ++i) res[i] = round_to<T>(0.01f * float(i % 41));
    for (int i = 0; i < N; ++i) wt[i] = round_to<T>(0.5f + 0.01f * float(i));
    rres[k].put(res);
    rw[k].put(wt);
  }
  for (int k = 0; k < W; ++k) {
    CommGemmParams p{};
    p.gemm.args.a.data = ra[k].p;
    p.gemm.args.a.ld = K;
    p.gemm.args.a.dtype = DType::kBFloat16;
    p.gemm.args.b.data = rb[k].p;
    p.gemm.args.b.ld = K;
    p.gemm.args.b.dtype = DType::kBFloat16;
    p.gemm.args.d = rout[k].p;
    p.gemm.args.ldd = N;
    p.gemm.args.d_dtype = DType::kBFloat16;
    p.gemm.args.m = M;
    p.gemm.args.n = N;
    p.gemm.args.k = K;
    p.gemm.args.recipe = kops::gemm::GemmRecipe::kBF16;
    p.peers = mg_peers(wr, k);
    p.num_chunks = chunks;
    p.chunk_ready = wr.signals;
    p.epoch = 1;
    KOPS_EXPECT_OK(gemm_reduce_scatter(p, kops::gemm::GemmRecipe::kBF16, wr.r[k].ctx), "gemm_reduce_scatter");
  }
  wr.sync_all();
  std::vector<double> sum(M * N, 0.0);
  for (int k = 0; k < W; ++k) {
    const std::vector<double> part = mm_ref<T>(hra[k], M, K, hrb[k], N);
    for (size_t i = 0; i < sum.size(); ++i) sum[i] += part[i];
  }
  for (int k = 0; k < W; ++k) {
    std::vector<double> want(size_t(M / W) * N);
    for (int r = 0; r < M / W; ++r)
      for (int c = 0; c < N; ++c) want[size_t(r) * N + c] = sum[size_t(k * (M / W) + r) * N + c];
    kops_test::compare(rout[k].get(), want, 3e-2, 3e-2, "gemm_reduce_scatter");
  }

  // allreduce without the fused epilogue: D must equal the chunk-summed fp32 partials.
  for (int k = 0; k < W; ++k) {
    CommGemmParams p{};
    p.gemm.args.a.data = ra[k].p;
    p.gemm.args.a.ld = K;
    p.gemm.args.a.dtype = DType::kBFloat16;
    p.gemm.args.b.data = rb[k].p;
    p.gemm.args.b.ld = K;
    p.gemm.args.b.dtype = DType::kBFloat16;
    p.gemm.args.d = rplain[k].p;
    p.gemm.args.ldd = N;
    p.gemm.args.d_dtype = DType::kBFloat16;
    p.gemm.args.m = M;
    p.gemm.args.n = N;
    p.gemm.args.k = K;
    p.gemm.args.recipe = kops::gemm::GemmRecipe::kBF16;
    p.peers = mg_peers(wr, k);
    p.num_chunks = chunks;
    p.chunk_ready = wr.signals;
    p.epoch = 2;
    KOPS_EXPECT_OK(gemm_allreduce(p, kops::gemm::GemmRecipe::kBF16, AllReduceAlgo::kOneShot, wr.r[k].ctx),
                   "gemm_allreduce plain");
  }
  wr.sync_all();
  for (int k = 0; k < W; ++k) {
    std::vector<double> want = sum;
    kops_test::compare(rplain[k].get(), want, 3e-2, 3e-2, "gemm_allreduce plain");
  }

  // allreduce with epoch 3 and no fusion into rar, as a control for the staging parity.
  for (int k = 0; k < W; ++k) {
    CommGemmParams p{};
    p.gemm.args.a.data = ra[k].p;
    p.gemm.args.a.ld = K;
    p.gemm.args.a.dtype = DType::kBFloat16;
    p.gemm.args.b.data = rb[k].p;
    p.gemm.args.b.ld = K;
    p.gemm.args.b.dtype = DType::kBFloat16;
    p.gemm.args.d = rar[k].p;
    p.gemm.args.ldd = N;
    p.gemm.args.d_dtype = DType::kBFloat16;
    p.gemm.args.m = M;
    p.gemm.args.n = N;
    p.gemm.args.k = K;
    p.gemm.args.recipe = kops::gemm::GemmRecipe::kBF16;
    p.peers = mg_peers(wr, k);
    p.num_chunks = chunks;
    p.chunk_ready = wr.signals;
    p.epoch = 3;
    KOPS_EXPECT_OK(gemm_allreduce(p, kops::gemm::GemmRecipe::kBF16, AllReduceAlgo::kOneShot, wr.r[k].ctx),
                   "gemm_allreduce control");
  }
  wr.sync_all();
  for (int k = 0; k < W; ++k) kops_test::compare(rar[k].get(), sum, 3e-2, 3e-2, "gemm_allreduce control");

  // allreduce (+ residual + RMSNorm)
  for (int k = 0; k < W; ++k) {
    CommGemmParams p{};
    p.gemm.args.a.data = ra[k].p;
    p.gemm.args.a.ld = K;
    p.gemm.args.a.dtype = DType::kBFloat16;
    p.gemm.args.b.data = rb[k].p;
    p.gemm.args.b.ld = K;
    p.gemm.args.b.dtype = DType::kBFloat16;
    p.gemm.args.d = rar[k].p;
    p.gemm.args.ldd = N;
    p.gemm.args.d_dtype = DType::kBFloat16;
    p.gemm.args.m = M;
    p.gemm.args.n = N;
    p.gemm.args.k = K;
    p.gemm.args.recipe = kops::gemm::GemmRecipe::kBF16;
    p.peers = mg_peers(wr, k);
    p.num_chunks = chunks;
    p.chunk_ready = wr.signals;
    p.epoch = 4;
    p.fuse_residual_rmsnorm = true;
    p.residual = rres[k].p;
    p.norm_weight = rw[k].p;
    p.norm_eps = 1e-5f;
    KOPS_EXPECT_OK(gemm_allreduce(p, kops::gemm::GemmRecipe::kBF16, AllReduceAlgo::kOneShot, wr.r[k].ctx),
                   "gemm_allreduce");
  }
  wr.sync_all();
  for (int k = 0; k < W; ++k) {
    const std::vector<T> res = rres[k].get();
    const std::vector<T> wt = rw[k].get();
    std::vector<double> want(size_t(M) * N);
    for (int r = 0; r < M; ++r) {
      double ss = 0.0;
      for (int c = 0; c < N; ++c) {
        const double v = sum[size_t(r) * N + c] + double(kops_test::to_float(res[size_t(r) * N + c]));
        want[size_t(r) * N + c] = v;
        ss += v * v;
      }
      const double rstd = std::sqrt(ss / N + 1e-5);
      for (int c = 0; c < N; ++c)
        want[size_t(r) * N + c] = want[size_t(r) * N + c] / rstd * double(kops_test::to_float(wt[c]));
    }
    const std::vector<T> got = rar[k].get();
    kops_test::compare(got, want, 5e-2, 5e-2, "gemm_allreduce fused norm");
  }
  wr.destroy();
}

void test_comm_gemm_contract() {
  World w;
  if (!w.init(2, false, 4096, 0)) return;
  CommGemmParams p{};
  int a = 0, b = 0, d = 0;
  p.gemm.args.a.data = &a;
  p.gemm.args.a.ld = 16;
  p.gemm.args.a.dtype = DType::kBFloat16;
  p.gemm.args.b.data = &b;
  p.gemm.args.b.ld = 16;
  p.gemm.args.b.dtype = DType::kBFloat16;
  p.gemm.args.d = &d;
  p.gemm.args.ldd = 16;
  p.gemm.args.d_dtype = DType::kBFloat16;
  p.gemm.args.m = 16;
  p.gemm.args.n = 16;
  p.gemm.args.k = 16;
  p.gemm.args.recipe = kops::gemm::GemmRecipe::kBF16;
  p.peers = mg_peers(w, 0);
  p.num_chunks = 2;
  p.chunk_ready = w.signals;
  const auto& ctx = w.r[0].ctx;
  KOPS_EXPECT_STATUS(allgather_gemm(p, kops::gemm::GemmRecipe::kFP8Tensorwise, ctx), Status::kNotImplemented,
                     "comm_gemm fp8 recipe");
  CommGemmParams badk = p;
  badk.gemm.args.k = 24;
  KOPS_EXPECT_STATUS(allgather_gemm(badk, kops::gemm::GemmRecipe::kBF16, ctx), Status::kUnsupportedShape,
                     "comm_gemm K % 16");
  CommGemmParams badc = p;
  badc.num_chunks = 3;
  KOPS_EXPECT_STATUS(allgather_gemm(badc, kops::gemm::GemmRecipe::kBF16, ctx), Status::kInvalidArgument,
                     "comm_gemm chunk multiple");
  CommGemmParams badf = p;
  badf.chunk_ready = nullptr;
  KOPS_EXPECT_STATUS(allgather_gemm(badf, kops::gemm::GemmRecipe::kBF16, ctx), Status::kInvalidArgument,
                     "comm_gemm null chunk_ready");
  CommGemmParams bade = p;
  bade.gemm.args.epi.ops = kops::gemm::kEpiAct;
  KOPS_EXPECT_STATUS(allgather_gemm(bade, kops::gemm::GemmRecipe::kBF16, ctx), Status::kNotImplemented,
                     "comm_gemm unsupported epilogue");
  CommGemmParams badfuse = p;
  badfuse.fuse_residual_rmsnorm = true;
  badfuse.residual = &a;
  badfuse.norm_weight = &b;
  KOPS_EXPECT_STATUS(gemm_reduce_scatter(badfuse, kops::gemm::GemmRecipe::kBF16, ctx),
                     Status::kInvalidArgument, "comm_gemm RS fuse norm");
  KOPS_EXPECT_STATUS(gemm_allreduce(p, kops::gemm::GemmRecipe::kBF16, AllReduceAlgo::kMultimem, ctx),
                     Status::kUnsupportedBackend, "comm_gemm multimem");
  CommGemmParams small = p;
  small.peers.buffer_bytes = 256;
  KOPS_EXPECT_STATUS(allgather_gemm(small, kops::gemm::GemmRecipe::kBF16, ctx), Status::kWorkspaceTooSmall,
                     "comm_gemm staging too small");
  w.destroy();
}


// ======================================================================================================
// allreduce_fusion: bit-exact against the separate chain
// ======================================================================================================
template <typename T>
void run_fusion_case(World& w, int tokens, int hidden, bool quant, unsigned seed, const char* tag) {
  const int W = w.size;
  const size_t n = size_t(tokens) * hidden;
  const DType dt = dtype_of<T>();
  std::vector<std::vector<T>> hin(W);
  std::vector<T> hres = random_vec<T>(n, seed + 99), hw = random_vec<T>(hidden, seed + 7, 1.5f);
  std::vector<RankBuf<T>> in(W), res_f(W), res_c(W), nrm_f(W), nrm_c(W), weight(W), ar(W);
  std::vector<RankBuf<uint8_t>> q_f(W), q_c(W);
  std::vector<RankBuf<float>> s_f(W), s_c(W);
  for (int k = 0; k < W; ++k) {
    const int d = w.r[k].device;
    hin[k] = random_vec<T>(n, seed * 5 + k);
    in[k].alloc(d, n); in[k].put(hin[k]);
    res_f[k].alloc(d, n); res_f[k].put(hres);
    res_c[k].alloc(d, n); res_c[k].put(hres);
    nrm_f[k].alloc(d, n); nrm_c[k].alloc(d, n); ar[k].alloc(d, n);
    weight[k].alloc(d, hidden); weight[k].put(hw);
    q_f[k].alloc(d, n); q_c[k].alloc(d, n); s_f[k].alloc(d, tokens); s_c[k].alloc(d, tokens);
  }
  // fused
  launch_ranks(w, iota_order(W), 0, [&](int k) {
    AllReduceFusionParams p{};
    p.input = in[k].p; p.residual_inout = res_f[k].p; p.norm_weight = weight[k].p; p.norm_eps = 1e-6f;
    p.weight_offset = 0.0f; p.norm_out = nrm_f[k].p; p.peers = w.r[k].peers; p.epoch = 0;
    p.num_tokens = tokens; p.hidden = hidden;
    if (quant) {
      p.quant.q = q_f[k].p; p.quant.q_row_stride = hidden; p.quant.scales = s_f[k].p;
      p.quant.q_dtype = DType::kFloat8E4M3; p.quant.mode = kops::ActQuantMode::kDynamicPerToken;
    }
    KOPS_EXPECT_OK(allreduce_fusion(p, AllReduceAlgo::kOneShot, dt, w.r[k].ctx), tag);
  });
  w.sync_all();
  // chain
  launch_ranks(w, iota_order(W), 0, [&](int k) {
    const int d = w.r[k].device;
    KOPS_EXPECT_OK(allreduce(flat(ar[k].p, dt, n, d), flat(in[k].p, dt, n, d), w.r[k].peers, AllReduceAlgo::kOneShot, 0, w.r[k].ctx), "chain ar");
    Tensor x = kops_test::tensor2d(ar[k].p, dt, tokens, hidden); x.device = d;
    Tensor r = kops_test::tensor2d(res_c[k].p, dt, tokens, hidden); r.device = d;
    Tensor o = kops_test::tensor2d(nrm_c[k].p, dt, tokens, hidden); o.device = d;
    Tensor wt = flat(weight[k].p, dt, hidden, d);
    KOPS_EXPECT_OK(kops::norm::fused_add_rms_norm_out(o, r, x, r, wt, 1e-6f, 0.0f, 1.0f, w.r[k].ctx), "chain norm");
    if (quant) {
      kops::ActQuantOut qo{};
      qo.q = q_c[k].p; qo.q_row_stride = hidden; qo.scales = s_c[k].p; qo.q_dtype = DType::kFloat8E4M3;
      qo.mode = kops::ActQuantMode::kDynamicPerToken;
      KOPS_EXPECT_OK(kops::quant::quantize_activation(qo, o, Tensor{}, 0, w.r[k].ctx), "chain quant");
    }
  });
  w.sync_all();
  for (int k = 0; k < W; ++k) {
    char what[160];
    std::snprintf(what, sizeof(what), "%s residual rank %d", tag, k);
    std::vector<T> got = res_f[k].get();
    if (k == 0) maybe_mutate(&got);
    expect_bits(got, res_c[k].get(), what);
    std::snprintf(what, sizeof(what), "%s norm rank %d", tag, k);
    expect_bits(nrm_f[k].get(), nrm_c[k].get(), what);
    if (quant) {
      std::snprintf(what, sizeof(what), "%s q rank %d", tag, k);
      const std::vector<uint8_t> a = q_f[k].get(), b = q_c[k].get();
      KOPS_TEST_CHECK(a == b);
      std::snprintf(what, sizeof(what), "%s scales rank %d", tag, k);
      expect_bits(s_f[k].get(), s_c[k].get(), what);
    }
  }
}

template <typename T>
void test_fusion_family(bool host) {
  const std::vector<int> worlds = host ? std::vector<int>{2} : std::vector<int>{1, 2, 4, 8};
  struct S { int t, h; };
  const S shapes[] = {{1, 4096}, {3, 1024}, {17, 2048}, {64, 4096}, {200, 8192}};
  for (int W : worlds) {
    World w;
    if (!w.init(W, host, 2 * (size_t(200) * 8192 * sizeof(T) + 128), 0)) return;
    for (const S& s : shapes) {
      for (int q = 0; q < 2; ++q) {
        if (q && std::is_same<T, float>::value) continue;
        char tag[128];
        std::snprintf(tag, sizeof(tag), "fusion %s W=%d t=%d h=%d %s %s", host ? "host" : "virtual", W, s.t, s.h,
                      q ? "fp8" : "bf", kops::dtype_name(dtype_of<T>()));
        run_fusion_case<T>(w, s.t, s.h, q != 0, 31, tag);
      }
    }
    w.destroy();
  }
}

void test_fusion_contract() {
  World w;
  if (!w.init(2, false, 1 << 20, 0)) return;
  RankBuf<__nv_bfloat16> a, b, c, wt;
  a.alloc(0, 4096); b.alloc(0, 4096); c.alloc(0, 4096); wt.alloc(0, 4096);
  AllReduceFusionParams p{};
  p.input = a.p; p.residual_inout = b.p; p.norm_weight = wt.p; p.norm_out = c.p; p.peers = w.r[0].peers;
  p.num_tokens = 1; p.hidden = 4096; p.norm_eps = 1e-6f;
  const LaunchContext& ctx = w.r[0].ctx;
  KOPS_EXPECT_STATUS(allreduce_fusion(p, AllReduceAlgo::kOneShot, DType::kInt8, ctx), Status::kUnsupportedDtype, "fusion dtype");
  AllReduceFusionParams p2 = p; p2.hidden = 4100;
  KOPS_EXPECT_STATUS(allreduce_fusion(p2, AllReduceAlgo::kOneShot, DType::kBFloat16, ctx), Status::kUnsupportedShape, "fusion hidden");
  AllReduceFusionParams p3 = p; p3.norm_out = nullptr;
  KOPS_EXPECT_STATUS(allreduce_fusion(p3, AllReduceAlgo::kOneShot, DType::kBFloat16, ctx), Status::kInvalidArgument, "fusion no output");
  AllReduceFusionParams p4 = p; p4.quant.mode = kops::ActQuantMode::kPerGroup;
  KOPS_EXPECT_STATUS(allreduce_fusion(p4, AllReduceAlgo::kOneShot, DType::kBFloat16, ctx), Status::kNotImplemented, "fusion quant mode");
  AllReduceFusionParams p5 = p; p5.hidden = 32768;
  KOPS_EXPECT_STATUS(allreduce_fusion(p5, AllReduceAlgo::kOneShot, DType::kBFloat16, ctx), Status::kUnsupportedShape, "fusion smem");
  KOPS_EXPECT_STATUS(allreduce_fusion(p, AllReduceAlgo::kMultimem, DType::kBFloat16, ctx), Status::kUnsupportedBackend, "fusion multimem");
  KOPS_TEST_CHECK(allreduce_fusion_staging_bytes(DType::kBFloat16, 4, 4096, 2) == 2 * 4 * 4096 * 2);
  w.destroy();
}

}  // namespace

int main(int argc, char** argv) {
  g_opt = kops_test::parse_options(argc, argv);
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
    std::printf("SKIP test_communication: no CUDA device\n");
    return 77;
  }
  if (g_opt.dryrun) {
    // host-side oracle self-check: the rank-order sum is order independent only for two operands
    std::vector<std::vector<float>> in = {{1e8f, 1.f}, {-1e8f, 2.f}, {3.f, 4.f}};
    const auto s = sum_ranks<float>(in, 0, 2);
    KOPS_TEST_CHECK(s[0] == 3.f && s[1] == 7.f);
    return kops_test::finish("test_communication --dryrun");
  }
  test_contract_collectives();
  test_validate_no_p2p();
  test_allreduce_family<__nv_bfloat16>(false);
  test_allreduce_family<__half>(false);
  test_allreduce_family<float>(false);
  test_allreduce_family<__nv_bfloat16>(true);
  test_allreduce_family<float>(true);
  test_pipeline_chunked<__nv_bfloat16>(false);
  test_pipeline_chunked<__nv_bfloat16>(true);
  test_gather_scatter_family<__nv_bfloat16>(false);
  test_gather_scatter_family<float>(false);
  test_gather_scatter_family<__nv_bfloat16>(true);
  test_arrival_order_determinism(false);
  test_arrival_order_determinism(true);
  test_mixed_sequence(false);
  test_mixed_sequence(true);
  test_graph_replay(false);
  test_graph_replay(true);
  test_late_rank_timeout(false);
  test_late_rank_timeout(true);
  test_fusion_contract();
  test_fusion_family<__nv_bfloat16>(false);
  test_fusion_family<float>(false);
  test_fusion_family<__nv_bfloat16>(true);
  test_allreduce_fusion<__nv_bfloat16>(false);
  test_allreduce_fusion<__nv_bfloat16>(true);
  test_ep_layout();
  test_ep_normal<__nv_bfloat16>(false);
  test_ep_normal<__nv_bfloat16>(true);
  test_ep_low_latency<__nv_bfloat16>();
  test_comm_gemm_impl<__nv_bfloat16>();
  test_comm_gemm_contract();
  return kops_test::finish("test_communication");
}
