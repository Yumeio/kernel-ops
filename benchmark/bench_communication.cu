// =============================================================================
// kernel-ops | benchmark/bench_communication.cu
// B12 benchmarks. The TP collectives and the allreduce fusion use the two real GPUs; without P2P
// (this dev box) their transport is pinned write-combined host memory. EP and the overlapped
// comm-GEMMs use virtual ranks (N regions of one device allocation, one stream per rank) so the
// same peer-pointer kernels run without a multi-GPU allocation; that is the rank model of ROADMAP
// 3.3 K5. A NCCL baseline with the same case ids lives in benchmark/compare/external/nccl_baseline.cu.
// Cases:
//   allreduce / allgather / reduce_scatter   TP shapes: decode (tokens x hidden, 8 KiB - 2 MiB) and
//                                            prefill (4 - 64 MiB)
//   allreduce_fusion                         AR + residual + RMSNorm vs NCCL AR + fused_add_rms_norm
//   ep_dispatch_combine                      DeepEP-style normal / low-latency round trip
//   allgather_gemm / gemm_reduce_scatter / gemm_allreduce
//                                            overlapped TP GEMMs vs GEMM + NCCL (cuBLAS), sequential
//
// Timing: `--mode graph` (default) captures K calls into one CUDA graph per rank and times graph
// launches (how a serving engine runs decode); `--mode eager` times K back-to-back host launches.
// EP is forced eager because its epoch is a host parameter. Every sample is the time of K calls,
// reduced to the longest per-rank stream, divided by K.
// =============================================================================
#include "benchmark/harness.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "csrc/gemm/gemm_params.cuh"
#include "csrc/communication/allreduce_kernels.cuh"
#include "csrc/communication/ep_kernels.cuh"
#include "csrc/communication/comm_gemm_kernels.cuh"
#include "csrc/communication/ops.h"

namespace {

using kops::ActQuantMode;
using kops::DType;
using kops::Fp8Encoding;
using kops::Status;
using namespace kops::comm;
using kops_bench::Case;
using kops_bench::Result;
using kops_bench::Stats;

bool g_graph_mode = true;
int g_group_calls = 16;  // K calls per sample

struct Rank {
  int device = 0;
  cudaStream_t stream = nullptr;
  PeerBuffers peers{};
  kops::LaunchContext ctx{};
};

struct World {
  int size = 2;
  size_t buffer_bytes = 0;
  void* staging = nullptr;
  uint32_t* signals = nullptr;
  std::vector<Rank> r;
  bool init(size_t bytes) {
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev < size) {
      std::fprintf(stderr, "[bench] needs %d GPUs\n", size);
      return false;
    }
    buffer_bytes = (bytes + 127) & ~size_t(127);
    const size_t sig = size_t(kSignalWords) * sizeof(uint32_t);
    if (!kops_bench::cuda_ok(cudaHostAlloc(&staging, buffer_bytes * size,
                                           cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined),
                             "staging") ||
        !kops_bench::cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&signals), sig * size,
                                           cudaHostAllocPortable | cudaHostAllocMapped),
                             "signals")) {
      return false;
    }
    r.resize(size);
    for (int k = 0; k < size; ++k) {
      Rank& rk = r[k];
      rk.device = k;
      cudaSetDevice(k);
      cudaStreamCreateWithFlags(&rk.stream, cudaStreamNonBlocking);
      std::memset(&rk.peers, 0, sizeof(rk.peers));
      for (int j = 0; j < size; ++j) {
        rk.peers.data[j] = static_cast<char*>(staging) + buffer_bytes * j;
        rk.peers.flags[j] = signals + size_t(kSignalWords) * j;
      }
      rk.peers.rank = k;
      rk.peers.world_size = size;
      rk.peers.num_nodes = 1;
      rk.peers.buffer_bytes = buffer_bytes;
      if (!kops_bench::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&rk.peers.scratch), kScratchWords * sizeof(uint32_t)),
                               "scratch")) {
        return false;
      }
      if (!kops_bench::status_ok(kops::make_launch_context(k, rk.stream, &rk.ctx), "context")) return false;
    }
    for (int k = 0; k < size; ++k) {
      cudaSetDevice(k);
      if (!kops_bench::status_ok(comm_configure(r[k].peers, 0), "configure")) return false;
    }
    return true;
  }
  void sync() {
    for (Rank& rk : r) {
      cudaSetDevice(rk.device);
      cudaStreamSynchronize(rk.stream);
    }
  }
};

World g_world;

/// Times K calls of issue(rank) per sample: eager launches or one captured graph per rank.
/// `eager` forces host launches (the EP epoch is advanced by the host), otherwise the graph mode of
/// the command line is used. Per-rank events are reduced to the longest stream, so a lagging peer is
/// not hidden by rank 0 finishing first.
bool measure_ranks(const std::vector<Rank>& ranks, bool eager, const Case& c,
                   const std::function<void(int)>& issue, Stats* out) {
  const int nranks = static_cast<int>(ranks.size());
  const int K = g_group_calls;
  const bool use_graph = g_graph_mode && !eager;
  std::vector<cudaGraphExec_t> exec(nranks, nullptr);
  std::vector<cudaGraph_t> graph(nranks, nullptr);
  if (use_graph) {
    for (int k = 0; k < nranks; ++k) {
      cudaSetDevice(ranks[k].device);
      if (!kops_bench::cuda_ok(cudaStreamBeginCapture(ranks[k].stream, cudaStreamCaptureModeThreadLocal),
                               "capture")) {
        return false;
      }
      for (int j = 0; j < K; ++j) issue(k);
      if (!kops_bench::cuda_ok(cudaStreamEndCapture(ranks[k].stream, &graph[k]), "end capture") ||
          !kops_bench::cuda_ok(cudaGraphInstantiate(&exec[k], graph[k], 0), "instantiate")) {
        return false;
      }
    }
  }
  auto run_group = [&]() {
    for (int k = 0; k < nranks; ++k) {
      cudaSetDevice(ranks[k].device);
      if (use_graph) {
        cudaGraphLaunch(exec[k], ranks[k].stream);
      } else {
        for (int j = 0; j < K; ++j) issue(k);
      }
    }
  };
  for (int i = 0; i < std::max(2, c.warmup / K); ++i) run_group();
  for (int k = 0; k < nranks; ++k) {
    cudaSetDevice(ranks[k].device);
    cudaStreamSynchronize(ranks[k].stream);
  }
  std::vector<cudaEvent_t> e0(nranks), e1(nranks);
  for (int k = 0; k < nranks; ++k) {
    cudaSetDevice(ranks[k].device);
    cudaEventCreate(&e0[k]);
    cudaEventCreate(&e1[k]);
  }
  const int groups = std::max(8, c.iters / K);
  std::vector<float> samples;
  for (int g = 0; g < groups; ++g) {
    for (int k = 0; k < nranks; ++k) {
      cudaSetDevice(ranks[k].device);
      cudaEventRecord(e0[k], ranks[k].stream);
    }
    run_group();
    for (int k = 0; k < nranks; ++k) {
      cudaSetDevice(ranks[k].device);
      cudaEventRecord(e1[k], ranks[k].stream);
      cudaStreamSynchronize(ranks[k].stream);
    }
    float ms = 0;
    for (int k = 0; k < nranks; ++k) {
      cudaSetDevice(ranks[k].device);
      float one = 0;
      cudaEventElapsedTime(&one, e0[k], e1[k]);
      ms = std::max(ms, one);
    }
    samples.push_back(ms / K);
  }
  for (int k = 0; k < nranks; ++k) {
    cudaSetDevice(ranks[k].device);
    cudaEventDestroy(e0[k]);
    cudaEventDestroy(e1[k]);
    if (exec[k]) cudaGraphExecDestroy(exec[k]);
    if (graph[k]) cudaGraphDestroy(graph[k]);
  }
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  out->min_ms = samples.front();
  out->median_ms = samples[n / 2];
  out->p10_ms = samples[n / 10];
  out->p90_ms = samples[std::min(n - 1, (n * 9) / 10)];
  out->p99_ms = samples[std::min(n - 1, (n * 99) / 100)];
  return true;
}

/// Per-case metadata that does not fit the shared Case layout (shape + family + rank model).
struct Extra {
  int world = 0;       // 0 = the 2-GPU host-staged World; >0 = a virtual-rank VWorld on device 0
  int n = 0, k = 0;
  int chunks = 1;
  int experts = 0, topk = 0, cap = 0;
  bool low_latency = false;
  bool eager = false;
  int variant = 0;     // 0 normal, 1 layout, 2 AR+norm, ...
};
std::map<std::string, Extra> g_extra;

/// Virtual ranks: N regions of one device allocation, one stream per rank (ROADMAP 3.3 K5).
struct VWorld {
  int size = 0;
  size_t buffer_bytes = 0;
  void* staging = nullptr;
  uint32_t* flags = nullptr;
  uint32_t* signals = nullptr;
  std::vector<Rank> r;
  bool init(int n, size_t bytes) {
    size = n;
    buffer_bytes = (bytes + 127) & ~size_t(127);
    cudaSetDevice(0);
    if (!kops_bench::cuda_ok(cudaMalloc(&staging, std::max<size_t>(1, buffer_bytes * n)), "vstaging")) return false;
    if (!kops_bench::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&flags),
                                        sizeof(uint32_t) * size_t(kSignalWords) * n), "vflags")) return false;
    if (!kops_bench::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&signals),
                                        sizeof(uint32_t) * size_t(kSignalWords) * n), "vsignals")) return false;
    cudaMemset(flags, 0, sizeof(uint32_t) * size_t(kSignalWords) * n);
    cudaMemset(signals, 0, sizeof(uint32_t) * size_t(kSignalWords) * n);
    r.resize(n);
    for (int k = 0; k < n; ++k) {
      Rank& rk = r[k];
      rk.device = 0;
      cudaSetDevice(0);
      cudaStreamCreateWithFlags(&rk.stream, cudaStreamNonBlocking);
      std::memset(&rk.peers, 0, sizeof(rk.peers));
      for (int j = 0; j < n; ++j) {
        rk.peers.data[j] = static_cast<char*>(staging) + buffer_bytes * j;
        rk.peers.flags[j] = flags + size_t(kSignalWords) * j;
      }
      rk.peers.rank = k;
      rk.peers.world_size = n;
      rk.peers.num_nodes = 1;
      rk.peers.buffer_bytes = buffer_bytes;
      if (!kops_bench::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&rk.peers.scratch),
                                          kScratchWords * sizeof(uint32_t)), "vscratch")) return false;
      if (!kops_bench::status_ok(kops::make_launch_context(0, rk.stream, &rk.ctx), "vcontext")) return false;
      if (!kops_bench::status_ok(comm_configure(rk.peers, 0), "vconfigure")) return false;
    }
    cudaMemset(signals, 0, sizeof(uint32_t) * size_t(kSignalWords) * n);
    return true;
  }
  void sync() {
    for (Rank& rk : r) {
      cudaSetDevice(rk.device);
      cudaStreamSynchronize(rk.stream);
    }
  }
  ~VWorld() {
    cudaSetDevice(0);
    for (Rank& rk : r) {
      if (rk.stream) cudaStreamDestroy(rk.stream);
      if (rk.peers.scratch) cudaFree(rk.peers.scratch);
    }
    if (staging) cudaFree(staging);
    if (flags) cudaFree(flags);
    if (signals) cudaFree(signals);
  }
};

template <typename T>
struct DevBuf {
  T* p = nullptr;
  size_t n = 0;
  void alloc(size_t count) {
    n = count;
    if (!kops_bench::cuda_ok(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(count, 1) * sizeof(T)),
                             "devbuf")) p = nullptr;
    if (p) cudaMemset(p, 0x3c, std::max<size_t>(count, 1) * sizeof(T));
  }
  void put(const std::vector<T>& h) {
    if (p) cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  ~DevBuf() {
    cudaSetDevice(0);
    if (p) cudaFree(p);
  }
};

struct Buf {
  void* p[2] = {nullptr, nullptr};
  size_t bytes = 0;
  void alloc(size_t b) {
    bytes = b;
    for (int k = 0; k < 2; ++k) {
      cudaSetDevice(k);
      cudaMalloc(&p[k], std::max<size_t>(b, 16));
      cudaMemset(p[k], 0x3c, std::max<size_t>(b, 16));
    }
    cudaSetDevice(0);
  }
  ~Buf() {
    for (int k = 0; k < 2; ++k)
      if (p[k]) {
        cudaSetDevice(k);
        cudaFree(p[k]);
      }
  }
};

// ---- case table ---------------------------------------------------------------------------------------------
struct Shape {
  int tokens, hidden;
};
// TP all-reduce shapes of a 4096 / 8192 hidden model: decode batches and prefill chunks
const Shape kShapes[] = {{1, 4096},   {8, 4096},   {32, 4096},   {128, 4096}, {1, 8192},    {8, 8192},
                         {32, 8192},  {128, 8192}, {512, 4096},  {2048, 4096}, {8192, 4096}, {2048, 8192}};

std::string id_of(const char* op, const Shape& s) {
  char b[96];
  std::snprintf(b, sizeof(b), "%s_bf16_t%d_h%d", op, s.tokens, s.hidden);
  return b;
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  for (const Shape& s : kShapes) {
    const int64_t bytes = int64_t(s.tokens) * s.hidden * 2;
    const int iters = bytes <= (1 << 20) ? 800 : (bytes <= (8 << 20) ? 160 : 48);
    for (const char* op : {"allreduce", "allgather", "reduce_scatter"}) {
      // allgather / reduce_scatter operate on the same per-rank payload (sequence-parallel residuals)
      if (std::strcmp(op, "allreduce") != 0 && bytes > (16 << 20)) continue;
      cases.push_back({id_of(op, s), op, DType::kBFloat16, s.tokens, s.hidden, bytes, 0, 32, iters});
      if (std::strcmp(op, "allreduce") == 0 && bytes >= (128 << 10)) {
        char id[96];
        std::snprintf(id, sizeof(id), "allreduce_two_shot_bf16_t%d_h%d", s.tokens, s.hidden);
        cases.push_back({id, "allreduce_two_shot", DType::kBFloat16, s.tokens, s.hidden, bytes, 0, 32, iters});
      }
    }
  }

  auto add = [&](Case cc, Extra ex) {
    g_extra[cc.id] = ex;
    cases.push_back(cc);
  };
  // ---- B12 collectives that are not part of the TP allreduce sweep ----
  add({"peer_barrier_bf16_w2", "peer_barrier", DType::kBFloat16, 0, 0, 0, 0, 64, 2000}, Extra{});
  for (const Shape& s : {Shape{1, 4096}, Shape{8, 4096}, Shape{32, 4096}, Shape{128, 4096},
                         Shape{512, 4096}, Shape{2048, 4096}}) {
    char id[96];
    std::snprintf(id, sizeof(id), "allreduce_fusion_bf16_t%d_h%d", s.tokens, s.hidden);
    add({id, "allreduce_fusion", DType::kBFloat16, s.tokens, s.hidden,
         int64_t(s.tokens) * s.hidden * 2, 0, 16, 160}, Extra{});
    char qid[96];
    std::snprintf(qid, sizeof(qid), "allreduce_fusion_fp8_bf16_t%d_h%d", s.tokens, s.hidden);
    Extra qx{};
    qx.variant = 1;
    add({qid, "allreduce_fusion", DType::kBFloat16, s.tokens, s.hidden,
         int64_t(s.tokens) * s.hidden, 0, 16, 160}, qx);
  }

  // ---- B12 EP: layout, normal dispatch+combine, low-latency masked round trip ----
  {
    Extra e{}; e.world = 1; e.experts = 64; e.topk = 8;
    add({"ep_layout_bf16_e64_t1024_k8", "ep_layout", DType::kBFloat16, 1024, 8, 0, 0, 64, 2000}, e);
  }
  struct EpCase { int world, tokens, hidden, experts, topk, iters; bool ll; };
  for (const EpCase& e : {EpCase{2, 128, 2048, 32, 4, 200, false},
                          EpCase{4, 256, 4096, 64, 8, 100, false},
                          EpCase{8, 64, 1024, 16, 4, 300, false},
                          EpCase{4, 256, 4096, 64, 8, 100, true}}) {
    char id[128];
    const int el = e.experts / e.world;
    const int cap = (e.tokens * e.topk + el - 1) / el;
    const int cap_rows = e.ll ? el * cap : e.tokens * e.world;
    std::snprintf(id, sizeof(id), "ep_dispatch_combine_bf16_%s_w%d_t%d_h%d_e%d_k%d", e.ll ? "ll" : "normal",
                  e.world, e.tokens, e.hidden, e.experts, e.topk);
    Extra x{}; x.world = e.world; x.experts = e.experts; x.topk = e.topk;
    x.low_latency = e.ll; x.cap = cap; x.eager = true;
    const int64_t bytes = int64_t(cap_rows) * e.hidden * 2;
    add({id, "ep_dispatch_combine", DType::kBFloat16, e.tokens, e.hidden, bytes,
         0, 16, e.iters}, x);
  }

  // ---- B12 comm-GEMM (virtual ranks on device 0) ----
  struct CgCase { int variant, world, m, n, k, chunks, iters; const char* name; };
  for (const CgCase& e : {CgCase{0, 2, 512, 1024, 1024, 8, 64, "allgather_gemm"},
                          CgCase{0, 4, 256, 1024, 1024, 8, 64, "allgather_gemm"},
                          CgCase{1, 2, 1024, 1024, 1024, 8, 64, "gemm_reduce_scatter"},
                          CgCase{2, 2, 1024, 1024, 1024, 8, 64, "gemm_allreduce"},
                          CgCase{3, 2, 512, 4096, 4096, 8, 24, "gemm_allreduce_norm"}}) {
    char id[160];
    std::snprintf(id, sizeof(id), "%s_bf16_w%d_m%d_n%d_k%d_c%d", e.name, e.world, e.m, e.n, e.k, e.chunks);
    const int64_t flops = e.variant == 0 ? int64_t(2) * e.m * e.world * e.n * e.k
                                         : int64_t(2) * e.m * e.n * e.k;
    const int64_t bytes = e.variant == 0 ? int64_t(e.m) * e.k * 2 + int64_t(e.m) * e.world * e.n * 2
                                         : int64_t(e.m) * e.n * 2;
    Extra x{}; x.world = e.world; x.n = e.n; x.k = e.k; x.chunks = e.chunks; x.variant = e.variant;
    add({id, "comm_gemm", DType::kBFloat16, e.m, e.n, bytes, flops, 8, e.iters}, x);
  }
  return cases;
}

bool run_collective(const Case& c, Stats* st) {
  World& w = g_world;
  const size_t n = size_t(c.rows) * c.cols;
  const bool ag = c.op == "allgather", rs = c.op == "reduce_scatter";
  const bool ar2 = c.op == "allreduce_two_shot";
  Buf in, out;
  in.alloc(n * 2 * (rs ? 2 : 1));
  out.alloc(n * 2 * (ag ? 2 : 1));
  bool failed = false;
  auto issue = [&](int k) {
    const int dev = w.r[k].device;
    kops::Tensor ti = kops_bench::tensor1d(in.p[k], DType::kBFloat16, int64_t(n) * (rs ? 2 : 1));
    kops::Tensor to = kops_bench::tensor1d(out.p[k], DType::kBFloat16, int64_t(n) * (ag ? 2 : 1));
    ti.device = to.device = dev;
    Status s = Status::kSuccess;
    if (ag) {
      ti = kops_bench::tensor1d(in.p[k], DType::kBFloat16, int64_t(n));
      ti.device = dev;
      s = allgather(to, ti, w.r[k].peers, 0, w.r[k].ctx);
    } else if (rs) {
      to = kops_bench::tensor1d(out.p[k], DType::kBFloat16, int64_t(n));
      to.device = dev;
      s = reduce_scatter(to, ti, w.r[k].peers, 0, w.r[k].ctx);
    } else {
      s = allreduce(to, ti, w.r[k].peers, ar2 ? AllReduceAlgo::kTwoShot : AllReduceAlgo::kOneShot, 0,
                    w.r[k].ctx);
    }
    if (s != Status::kSuccess) {
      failed = true;
      std::fprintf(stderr, "[bench] %s: %s\n", c.id.c_str(), kops::last_error_message());
    }
  };
  return !failed && measure_ranks(g_world.r, false, c, issue, st);
}


// ---- B12 EP -----------------------------------------------------------------------------------
size_t ep_dispatch_rec_b(int hidden, int topk) {
  return ((16 + size_t(topk) * 8 + size_t(hidden) * 2 + 15) / 16) * 16;
}
size_t ep_combine_rec_b(int hidden) { return ((16 + size_t(hidden) * 2 + 15) / 16) * 16; }

EPBuffers ep_buffers_of(VWorld& vw, int rank) {
  EPBuffers b{};
  for (int j = 0; j < vw.size; ++j) b.nvl_buffers[j] = vw.r[rank].peers.data[j];
  b.signals = vw.signals;
  b.rank = rank;
  b.num_ranks = vw.size;
  b.num_local_ranks = vw.size;
  b.transport = EPTransport::kNvlink;
  b.nvl_bytes = vw.buffer_bytes;
  return b;
}

bool run_ep_layout(const Case& c, const Extra& x, Stats* st) {
  const int T = int(c.rows), K = int(c.cols), E = x.experts;
  VWorld vw;
  if (!vw.init(1, 1 << 12)) return false;
  std::vector<int32_t> topk(size_t(T) * K);
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < K; ++j) topk[size_t(t) * K + j] = (t * 3 + j * 2) % E;
  DevBuf<int32_t> td, pr, pe;
  DevBuf<uint8_t> mem;
  td.alloc(topk.size());
  td.put(topk);
  pr.alloc(1);
  pe.alloc(E);
  mem.alloc(size_t(T) * 1);
  bool failed = false;
  auto issue = [&](int) {
    kops::Tensor tt = kops_bench::tensor2d(td.p, DType::kInt32, T, K);
    kops::Tensor tr = kops_bench::tensor1d(pr.p, DType::kInt32, 1);
    kops::Tensor te = kops_bench::tensor1d(pe.p, DType::kInt32, E);
    kops::Tensor tm = kops_bench::tensor1d(mem.p, DType::kUInt8, size_t(T));
    const Status s = ep_dispatch_layout(tr, te, tm, tt, E, 1, vw.r[0].ctx);
    if (s != Status::kSuccess) {
      failed = true;
      std::fprintf(stderr, "[bench] %s: %s\n", c.id.c_str(), kops::last_error_message());
    }
  };
  return !failed && measure_ranks(vw.r, false, c, issue, st);
}

bool run_ep(const Case& c, const Extra& x, Stats* st) {
  const int W = x.world > 0 ? x.world : 4;
  const int T = int(c.rows), H = int(c.cols), K = x.topk, E = x.experts;
  const int e_local = E / W;
  const int cap = x.cap > 0 ? x.cap : std::max(1, (T * K + e_local - 1) / e_local);
  const int cap_rows = x.low_latency ? e_local * cap : T * W;
  const size_t rd = ep_dispatch_rec_b(H, K), rc = ep_combine_rec_b(H);
  const size_t need_d = size_t(W) * ((size_t(T) * rd + 127) & ~size_t(127));
  const size_t need_c = size_t(W) * ((size_t(cap_rows) * rc + 127) & ~size_t(127));
  VWorld vw;
  if (!vw.init(W, std::max(need_d, need_c))) return false;

  std::vector<std::vector<__nv_bfloat16>> hx(W), heo(W);
  std::vector<std::vector<int32_t>> htopk(W);
  std::vector<std::vector<float>> hw(W);
  std::vector<std::vector<uint8_t>> hmem(W);
  std::vector<DevBuf<__nv_bfloat16>> dx(W), recv_x(W), expert_out(W), out(W);
  std::vector<DevBuf<int64_t>> recv_src(W);
  std::vector<DevBuf<int32_t>> topk_d(W), recv_idx(W), recv_cnt(W);
  std::vector<DevBuf<float>> w_d(W), recv_w(W);
  std::vector<DevBuf<uint8_t>> member_d(W);
  for (int k = 0; k < W; ++k) {
    hx[k].resize(size_t(T) * H);
    for (size_t i = 0; i < hx[k].size(); ++i)
      hx[k][i] = __float2bfloat16_rn(0.01f * float((i + 7 * k) % 131) - 0.6f);
    htopk[k].resize(size_t(T) * K);
    hw[k].resize(size_t(T) * K);
    hmem[k].assign(size_t(T) * W, 0);
    for (int t = 0; t < T; ++t)
      for (int j = 0; j < K; ++j) {
        int e = (t * 3 + j * 2 + k) % E;
        if (((t + j + k) % 13) == 0) e = -1;
        htopk[k][size_t(t) * K + j] = e;
        hw[k][size_t(t) * K + j] = 0.5f * float((t + 3 * j + k) % 11) - 2.5f;
        if (e >= 0) hmem[k][size_t(t) * W + e / e_local] = 1;
      }
    heo[k].resize(size_t(cap_rows) * H);
    for (size_t i = 0; i < heo[k].size(); ++i)
      heo[k][i] = __float2bfloat16_rn(0.01f * float(i % 97) + 0.25f * float(k + 1));
    dx[k].alloc(size_t(T) * H); dx[k].put(hx[k]);
    topk_d[k].alloc(htopk[k].size()); topk_d[k].put(htopk[k]);
    w_d[k].alloc(hw[k].size()); w_d[k].put(hw[k]);
    member_d[k].alloc(hmem[k].size()); member_d[k].put(hmem[k]);
    recv_x[k].alloc(size_t(cap_rows) * H);
    recv_src[k].alloc(cap_rows);
    cudaMemset(recv_src[k].p, 0xFF, cap_rows * sizeof(int64_t));
    recv_idx[k].alloc(size_t(cap_rows) * K);
    recv_w[k].alloc(size_t(cap_rows) * K);
    recv_cnt[k].alloc(e_local);
    expert_out[k].alloc(heo[k].size()); expert_out[k].put(heo[k]);
    out[k].alloc(size_t(cap_rows) * H);
  }
  std::vector<uint32_t> epoch(W, 0);
  bool failed = false;
  auto issue = [&](int k) {
    EPDispatchParams p{};
    p.x = dx[k].p;
    p.topk_idx = topk_d[k].p;
    p.topk_weights = w_d[k].p;
    p.is_token_in_rank = member_d[k].p;
    p.recv_x = recv_x[k].p;
    p.recv_topk_idx = recv_idx[k].p;
    p.recv_topk_weights = recv_w[k].p;
    p.recv_count_per_expert = recv_cnt[k].p;
    p.recv_src_info = recv_src[k].p;
    p.buffers = ep_buffers_of(vw, k);
    p.num_tokens = T;
    p.hidden = H;
    p.topk = K;
    p.num_experts = E;
    p.max_tokens_per_rank = x.low_latency ? cap : cap_rows;
    p.epoch = epoch[k];
    p.low_latency = x.low_latency;
    p.hook_mode = false;
    Status s = ep_dispatch(p, DType::kBFloat16, vw.r[k].ctx);
    if (s != Status::kSuccess) {
      failed = true;
      std::fprintf(stderr, "[bench] %s dispatch: %s\n", c.id.c_str(), kops::last_error_message());
    }
    EPCombineParams cp{};
    cp.expert_out = expert_out[k].p;
    cp.topk_weights = recv_w[k].p;
    cp.weights_already_applied = false;
    cp.src_info = recv_src[k].p;
    cp.buffers = ep_buffers_of(vw, k);
    cp.out = out[k].p;
    cp.residual = nullptr;
    cp.num_tokens = cap_rows;
    cp.hidden = H;
    cp.topk = K;
    cp.epoch = epoch[k] + 1;
    cp.low_latency = x.low_latency;
    cp.hook_mode = false;
    s = ep_combine(cp, DType::kBFloat16, vw.r[k].ctx);
    if (s != Status::kSuccess) {
      failed = true;
      std::fprintf(stderr, "[bench] %s combine: %s\n", c.id.c_str(), kops::last_error_message());
    }
    epoch[k] += 2;
    s = peer_barrier(vw.r[k].peers, 0, vw.r[k].ctx);
    if (s != Status::kSuccess) {
      failed = true;
      std::fprintf(stderr, "[bench] %s barrier: %s\n", c.id.c_str(), kops::last_error_message());
    }
  };
  return !failed && measure_ranks(vw.r, true, c, issue, st);
}

// ---- B12 comm-GEMM ----------------------------------------------------------------------------
bool run_comm_gemm(const Case& c, const Extra& x, Stats* st) {
  const int W = x.world > 0 ? x.world : 2;
  const int variant = x.variant;  // 0 AG, 1 RS, 2 AR, 3 AR + norm
  const int m = int(c.rows), n = int(x.n), k = int(x.k), chunks = x.chunks > 0 ? x.chunks : 8;
  const int mt = variant == 0 ? m : m;  // AG: m is the local shard; RS/AR: m is the total M
  const size_t stage = variant == 0 ? 2 * (size_t(m) * k * 2 + 512)
                                    : 2 * (size_t(mt) * n * 4 + 512);
  VWorld vw;
  if (!vw.init(W, stage)) return false;
  std::vector<DevBuf<__nv_bfloat16>> a(W), b(W), d(W), res(W), wt(W);
  for (int r = 0; r < W; ++r) {
    const size_t rows_a = size_t(m) * k;
    a[r].alloc(rows_a);
    b[r].alloc(size_t(n) * k);
    std::vector<__nv_bfloat16> ha(rows_a), hb(size_t(n) * k);
    for (size_t i = 0; i < ha.size(); ++i) ha[i] = __float2bfloat16_rn(0.02f * float((i + 5 * r) % 67) - 0.7f);
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = __float2bfloat16_rn(0.03f * float((i + 3 * r) % 71) - 1.0f);
    a[r].put(ha);
    b[r].put(hb);
    const size_t rows_d = variant == 0 ? size_t(m) * W * n : (variant == 1 ? size_t(m / W) * n : size_t(m) * n);
    d[r].alloc(rows_d);
    if (variant == 3) {
      res[r].alloc(size_t(m) * n);
      wt[r].alloc(n);
      std::vector<__nv_bfloat16> hr(size_t(m) * n), hwt(n);
      for (size_t i = 0; i < hr.size(); ++i) hr[i] = __float2bfloat16_rn(0.01f * float(i % 37));
      for (int i = 0; i < n; ++i) hwt[i] = __float2bfloat16_rn(0.5f + 0.01f * float(i));
      res[r].put(hr);
      wt[r].put(hwt);
    }
  }
  bool failed = false;
  auto issue = [&](int r) {
    CommGemmParams p{};
    p.gemm.args.a.data = a[r].p;
    p.gemm.args.a.ld = k;
    p.gemm.args.a.dtype = DType::kBFloat16;
    p.gemm.args.b.data = b[r].p;
    p.gemm.args.b.ld = k;
    p.gemm.args.b.dtype = DType::kBFloat16;
    p.gemm.args.d = d[r].p;
    p.gemm.args.ldd = n;
    p.gemm.args.d_dtype = DType::kBFloat16;
    p.gemm.args.m = m;
    p.gemm.args.n = n;
    p.gemm.args.k = k;
    p.gemm.args.recipe = kops::gemm::GemmRecipe::kBF16;
    p.peers = vw.r[r].peers;
    p.num_chunks = chunks;
    p.chunk_ready = vw.signals;
    p.epoch = 0;
    if (variant == 3) {
      p.fuse_residual_rmsnorm = true;
      p.residual = res[r].p;
      p.norm_weight = wt[r].p;
      p.norm_eps = 1e-5f;
    }
    Status s = Status::kSuccess;
    if (variant == 0) s = allgather_gemm(p, kops::gemm::GemmRecipe::kBF16, vw.r[r].ctx);
    else if (variant == 1) s = gemm_reduce_scatter(p, kops::gemm::GemmRecipe::kBF16, vw.r[r].ctx);
    else s = gemm_allreduce(p, kops::gemm::GemmRecipe::kBF16, AllReduceAlgo::kOneShot, vw.r[r].ctx);
    if (s != Status::kSuccess) {
      failed = true;
      std::fprintf(stderr, "[bench] %s: %s\n", c.id.c_str(), kops::last_error_message());
    }
  };
  return !failed && measure_ranks(vw.r, false, c, issue, st);
}

bool run_barrier(const Case& c, Stats* st) {
  bool failed = false;
  auto issue = [&](int k) {
    const Status s = peer_barrier(g_world.r[k].peers, 0, g_world.r[k].ctx);
    if (s != Status::kSuccess) {
      failed = true;
      std::fprintf(stderr, "[bench] %s: %s\n", c.id.c_str(), kops::last_error_message());
    }
  };
  return !failed && measure_ranks(g_world.r, false, c, issue, st);
}

bool run_fusion(const Case& c, const Extra& x, Stats* st) {
  const int T = int(c.rows), H = int(c.cols);
  const size_t cap = 2 * (size_t(T) * H * 2 + 128);
  VWorld vw;
  if (!vw.init(2, cap)) return false;
  std::vector<DevBuf<__nv_bfloat16>> in(2), residual(2), norm_out(2), norm_w(2);
  std::vector<DevBuf<uint8_t>> q(2);
  std::vector<DevBuf<float>> scales(2);
  for (int k = 0; k < 2; ++k) {
    in[k].alloc(size_t(T) * H);
    residual[k].alloc(size_t(T) * H);
    norm_out[k].alloc(size_t(T) * H);
    norm_w[k].alloc(H);
    if (x.variant == 1) {
      q[k].alloc(size_t(T) * H);
      scales[k].alloc(T);
    }
    std::vector<__nv_bfloat16> h(size_t(T) * H), hw(H);
    for (size_t i = 0; i < h.size(); ++i) h[i] = __float2bfloat16_rn(0.01f * float(i % 97) - 0.3f);
    for (int i = 0; i < H; ++i) hw[i] = __float2bfloat16_rn(0.5f + 0.01f * float(i));
    in[k].put(h);
    residual[k].put(h);
    norm_w[k].put(hw);
  }
  bool failed = false;
  auto issue = [&](int k) {
    AllReduceFusionParams p{};
    p.input = in[k].p;
    p.residual_inout = residual[k].p;
    p.norm_weight = norm_w[k].p;
    p.norm_eps = 1e-5f;
    p.weight_offset = 0.0f;
    p.norm_out = norm_out[k].p;
    if (x.variant == 1) {
      p.quant.mode = ActQuantMode::kDynamicPerToken;
      p.quant.q_dtype = DType::kFloat8E4M3;
      p.quant.fp8 = Fp8Encoding::kOCP;
      p.quant.symmetric = true;
      p.quant.q = q[k].p;
      p.quant.q_row_stride = H;
      p.quant.scales = scales[k].p;
    }
    p.peers = vw.r[k].peers;
    p.epoch = 0;
    p.num_tokens = T;
    p.hidden = H;
    const Status s = allreduce_fusion(p, AllReduceAlgo::kOneShot, DType::kBFloat16, vw.r[k].ctx);
    if (s != Status::kSuccess) {
      failed = true;
      std::fprintf(stderr, "[bench] %s: %s\n", c.id.c_str(), kops::last_error_message());
    }
  };
  return !failed && measure_ranks(vw.r, false, c, issue, st);
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) g_graph_mode = std::strcmp(argv[i + 1], "eager") != 0;
    if (std::strcmp(argv[i], "--calls") == 0 && i + 1 < argc) g_group_calls = std::max(1, std::atoi(argv[i + 1]));
  }
  const kops_bench::Options opt = kops_bench::parse_options(argc, argv);
  std::vector<Case> cases = make_cases();
  kops_bench::filter_cases(&cases, opt);
  kops_bench::DeviceInfo dev;
  cudaSetDevice(0);
  kops_bench::query_device(&dev);
  if (opt.dump_cases) {
    kops_bench::dump_cases(cases, dev);
    return 0;
  }
  size_t max_bytes = 0;
  for (const Case& c : cases) {
    if (c.op == "allreduce" || c.op == "allreduce_two_shot" || c.op == "allgather" ||
        c.op == "reduce_scatter" || c.op == "allreduce_fusion" || c.op == "peer_barrier") {
      max_bytes = std::max<size_t>(max_bytes, size_t(c.bytes) * 2);
    }
  }
  {
    // The host-staged collectives need two physical GPUs. A single-GPU kit (sm90a / sm100a) skips the
    // family with the CTest skip code instead of reporting a failure; the `multi` kit runs it.
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev < g_world.size) {
      std::fprintf(stderr, "[bench] SKIP bench_communication: needs %d GPUs, found %d\n", g_world.size, ndev);
      return 77;
    }
  }
  if (!g_world.init(2 * (max_bytes + 4096) * 2)) return 1;
  std::vector<Result> results;
  for (const Case& c : cases) {
    Stats st;
    bool ok = false;
    Extra ex{};
    auto it = g_extra.find(c.id);
    if (it != g_extra.end()) ex = it->second;
    if (c.op == "allreduce" || c.op == "allreduce_two_shot" || c.op == "allgather" ||
        c.op == "reduce_scatter") ok = run_collective(c, &st);
    else if (c.op == "peer_barrier") ok = run_barrier(c, &st);
    else if (c.op == "allreduce_fusion") ok = run_fusion(c, ex, &st);
    else if (c.op == "ep_layout") ok = run_ep_layout(c, ex, &st);
    else if (c.op == "ep_dispatch_combine") ok = run_ep(c, ex, &st);
    else if (c.op == "comm_gemm") ok = run_comm_gemm(c, ex, &st);
    if (!ok) {
      std::fprintf(stderr, "[bench] %s failed\n", c.id.c_str());
      continue;
    }
    results.push_back({&c, st});
  }
  kops_bench::print_table(results, dev);
  if (opt.json_path != nullptr) kops_bench::write_results(opt.json_path, results, dev);
  return 0;
}
