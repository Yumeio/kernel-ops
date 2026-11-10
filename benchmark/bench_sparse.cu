// =============================================================================
// kernel-ops | benchmark/bench_sparse.cu
// B14 A11 sparse_topk cases: exact top-K over an fp32 score row for
// rows = 1 / 16, max_kv = 64K / 1M and topk = 512 / 2048. The case table is the
// shared spec read by benchmark/compare/torch_baseline.py through --dump-cases.
//
// bytes is the logical volume of the operation (one read of the scores plus the
// two output rows), which is what the shared case table reports to every
// implementation: indices int32 and values fp32 are rows * topk each, scores are
// rows * max_kv fp32. The radix select reads the row four times (three histogram
// passes + one collect pass), so its effective DRAM traffic is up to 4x this
// figure when the rows are not L2-resident.
// =============================================================================
#include "benchmark/harness.h"

#include <cstdio>
#include <cmath>
#include <vector>

#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/kvcache/kvcache_kernels.cuh"
#include "csrc/kvcache/ops.h"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"

namespace {

using namespace kops_bench;

struct TopkBench {
  const char* id;
  int rows;
  int max_kv;
  int topk;
  int warmup;
  int iters;
};

struct IndexerBench {
  const char* id;
  int rows;
  int max_kv;
  int warmup;
  int iters;
};

struct MlaBench {
  const char* id;
  int heads;
  int topk;
  int warmup;
  int iters;
};

const std::vector<IndexerBench>& indexer_benches() {
  static const std::vector<IndexerBench> b = {
      {"lightning_indexer.bf16.r1.kv64k", 1, 64 * 1024, 5, 20},
      {"lightning_indexer.bf16.r1.kv1m", 1, 1024 * 1024, 3, 10},
      {"lightning_indexer.bf16.r16.kv64k", 16, 64 * 1024, 5, 10},
  };
  return b;
}

const std::vector<MlaBench>& mla_benches() {
  static const std::vector<MlaBench> b = {
      {"sparse_mla.bf16.h128.topk512", 128, 512, 5, 20},
      {"sparse_mla.bf16.h128.topk2048", 128, 2048, 3, 10},
  };
  return b;
}

const std::vector<TopkBench>& topk_benches() {
  static const std::vector<TopkBench> b = {
      {"sparse_topk.fp32.r1.kv64k.topk512", 1, 64 * 1024, 512, 5, 40},
      {"sparse_topk.fp32.r16.kv64k.topk512", 16, 64 * 1024, 512, 5, 40},
      {"sparse_topk.fp32.r1.kv1m.topk512", 1, 1024 * 1024, 512, 3, 20},
      {"sparse_topk.fp32.r16.kv1m.topk512", 16, 1024 * 1024, 512, 3, 20},
      {"sparse_topk.fp32.r1.kv64k.topk2048", 1, 64 * 1024, 2048, 5, 40},
      {"sparse_topk.fp32.r16.kv64k.topk2048", 16, 64 * 1024, 2048, 5, 40},
      {"sparse_topk.fp32.r1.kv1m.topk2048", 1, 1024 * 1024, 2048, 3, 20},
      {"sparse_topk.fp32.r16.kv1m.topk2048", 16, 1024 * 1024, 2048, 3, 20},
      // Boundary of the multi-CTA fast path (max_kv == 256 Ki) and its tie-heavy
      // companion: a constant row overflows the candidate bucket and takes the
      // exact single-CTA fallback inside the fast launch.
      {"sparse_topk.fp32.r1.kv256k.topk2048", 1, 256 * 1024, 2048, 5, 40},
      {"sparse_topk.fp32.r4.kv1m.topk2048", 4, 1024 * 1024, 2048, 3, 20},
  };
  return b;
}

const TopkBench* find_bench(const std::string& id) {
  for (const TopkBench& b : topk_benches()) {
    if (id == b.id) return &b;
  }
  return nullptr;
}

const IndexerBench* find_indexer(const std::string& id) {
  for (const IndexerBench& b : indexer_benches()) {
    if (id == b.id) return &b;
  }
  return nullptr;
}

const MlaBench* find_mla(const std::string& id) {
  for (const MlaBench& b : mla_benches()) {
    if (id == b.id) return &b;
  }
  return nullptr;
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  for (const TopkBench& b : topk_benches()) {
    const int64_t bytes = static_cast<int64_t>(b.rows) *
                          (static_cast<int64_t>(b.max_kv) * 4 + static_cast<int64_t>(b.topk) * 8);
    cases.push_back({b.id, "sparse_topk", kops::DType::kFloat32, b.rows, b.max_kv, bytes, 0,
                     b.warmup, b.iters});
  }
  for (const IndexerBench& b : indexer_benches()) {
    const int64_t qb = static_cast<int64_t>(b.rows) * 64 * 128 * 2;
    const int64_t kb = static_cast<int64_t>(b.max_kv) * 128 * 2;
    const int64_t lb = static_cast<int64_t>(b.rows) * b.max_kv * 4;
    cases.push_back({b.id, "lightning_indexer", kops::DType::kBFloat16, b.rows, b.max_kv,
                     qb + kb + lb, 0, b.warmup, b.iters});
  }
  for (const MlaBench& b : mla_benches()) {
    const int64_t qb = static_cast<int64_t>(b.heads) * 576 * 2;
    const int64_t kb = static_cast<int64_t>(b.topk) * 576 * 2;
    const int64_t ob = static_cast<int64_t>(b.heads) * 512 * 2;
    cases.push_back({b.id, "sparse_mla", kops::DType::kBFloat16, 1, b.topk, qb + kb + ob, 0,
                     b.warmup, b.iters});
  }
  return cases;
}

/// Deterministic uniform scores in [-1, 1): a counter-hash, so the histogram is
/// flat and the 4 passes are pure bandwidth (no tie-heavy path).
__global__ void bench_fill_scores(float* p, int64_t n, uint32_t seed) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n; i += step) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;
    p[i] = static_cast<float>(static_cast<double>(h & 0xFFFFFFu) / 8388608.0 - 1.0);
  }
}

bool run_sparse_topk(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const TopkBench* b = find_bench(c.id);
  if (b == nullptr) return false;
  const int64_t rows = b->rows, max_kv = b->max_kv, topk = b->topk;
  float* scores = nullptr;
  float* values = nullptr;
  int32_t* indices = nullptr;
  int32_t* ends = nullptr;
  void* ws = nullptr;
  const size_t ws_bytes = kops::attn::sparse_topk_workspace_bytes();
  bool ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scores),
                               static_cast<size_t>(rows * max_kv) * sizeof(float)),
                    "scores") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&values),
                               static_cast<size_t>(rows * topk) * sizeof(float)),
                    "values") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&indices),
                               static_cast<size_t>(rows * topk) * sizeof(int32_t)),
                    "indices") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ends),
                               static_cast<size_t>(rows) * sizeof(int32_t)),
                    "ends") &&
            cuda_ok(cudaMalloc(&ws, ws_bytes), "topk workspace");
  if (!ok) return false;
  bench_fill_scores<<<256, 256>>>(scores, rows * max_kv, 7u);
  std::vector<int32_t> host_ends(static_cast<size_t>(rows), b->max_kv);
  cudaMemcpy(ends, host_ends.data(), host_ends.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

  const kops::Tensor t_indices = tensor2d(indices, kops::DType::kInt32, rows, topk);
  const kops::Tensor t_values = tensor2d(values, kops::DType::kFloat32, rows, topk);
  const kops::Tensor t_scores = tensor2d(scores, kops::DType::kFloat32, rows, max_kv);
  const kops::Tensor t_ends = tensor1d(ends, kops::DType::kInt32, rows);
  const kops::Workspace lease{ws, ws_bytes};
  auto call = [&]() {
    return kops::attn::sparse_topk_ws(t_indices, t_values, t_scores, kops::Tensor{}, t_ends, b->topk,
                                      kops::Tensor{}, kops::Tensor{}, 0, true, false, lease, ctx);
  };
  ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);

  cudaFree(scores);
  cudaFree(values);
  cudaFree(indices);
  cudaFree(ends);
  cudaFree(ws);
  return ok;
}

bool run_indexer(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const IndexerBench* b = find_indexer(c.id);
  if (b == nullptr) return false;
  const int64_t rows = b->rows, max_kv = b->max_kv;
  auto* q = static_cast<__nv_bfloat16*>(nullptr);
  auto* k = static_cast<__nv_bfloat16*>(nullptr);
  auto* w = static_cast<float*>(nullptr);
  auto* logits = static_cast<float*>(nullptr);
  int32_t* starts = nullptr;
  int32_t* ends = nullptr;
  bool ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q),
                               static_cast<size_t>(rows * 64 * 128) * sizeof(__nv_bfloat16)),
                    "q") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&k),
                               static_cast<size_t>(max_kv * 128) * sizeof(__nv_bfloat16)),
                    "k") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&w), static_cast<size_t>(rows * 64) * 4),
                    "w") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits),
                               static_cast<size_t>(rows * max_kv) * 4),
                    "logits") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&starts), rows * 4), "starts") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ends), rows * 4), "ends");
  if (!ok) return false;
  cudaMemset(q, 0, static_cast<size_t>(rows * 64 * 128) * sizeof(__nv_bfloat16));
  cudaMemset(k, 0, static_cast<size_t>(max_kv * 128) * sizeof(__nv_bfloat16));
  cudaMemset(w, 0, static_cast<size_t>(rows * 64) * 4);
  std::vector<int32_t> hs(static_cast<size_t>(rows), 0), he(static_cast<size_t>(rows), max_kv);
  cudaMemcpy(starts, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(ends, he.data(), he.size() * 4, cudaMemcpyHostToDevice);

  kops::attn::IndexerParams p{};
  p.q = q;
  p.q_stride_n = 64 * 128;
  p.q_stride_h = 128;
  p.head_weights = w;
  p.k_cache.k_data = k;
  p.k_cache.k_dtype = kops::DType::kBFloat16;
  p.k_cache.k_codec = kops::kv::KVCodec::kNative;
  p.k_cache.layout = kops::kv::KVLayout::kLatent;
  p.k_cache.page_size = 0;
  p.k_cache.table_format = kops::kv::PageTableFormat::kSlotMapping;
  p.k_stride_n = 128;
  p.k_stride_h = 128;
  p.row_starts = starts;
  p.row_ends = ends;
  p.logits = logits;
  p.logits_stride = max_kv;
  p.num_tokens = static_cast<int>(rows);
  p.num_index_heads = 64;
  p.index_dim = 128;
  p.max_kv = static_cast<int>(max_kv);
  p.compress_ratio = 1;
  auto call = [&]() { return kops::attn::lightning_indexer(p, kops::DType::kBFloat16, ctx); };
  ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(q);
  cudaFree(k);
  cudaFree(w);
  cudaFree(logits);
  cudaFree(starts);
  cudaFree(ends);
  return ok;
}

bool run_sparse_mla(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const MlaBench* b = find_mla(c.id);
  if (b == nullptr) return false;
  const int heads = b->heads, topk = b->topk, slots = 4096, page = 64, pages = slots / page;
  void* kv = nullptr;
  void* q = nullptr;
  void* o = nullptr;
  int32_t* idx = nullptr;
  int32_t* tab = nullptr;
  float* lse = nullptr;
  bool ok = cuda_ok(cudaMalloc(&kv, static_cast<size_t>(slots) * 576 * sizeof(__nv_bfloat16)), "kv") &&
            cuda_ok(cudaMalloc(&q, static_cast<size_t>(heads) * 576 * sizeof(__nv_bfloat16)), "q") &&
            cuda_ok(cudaMalloc(&o, static_cast<size_t>(heads) * 512 * sizeof(__nv_bfloat16)), "o") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&idx), topk * 4), "idx") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&tab), pages * 4), "tab") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&lse), heads * 4), "lse");
  if (!ok) return false;
  cudaMemset(kv, 0, static_cast<size_t>(slots) * 576 * sizeof(__nv_bfloat16));
  cudaMemset(q, 0, static_cast<size_t>(heads) * 576 * sizeof(__nv_bfloat16));
  std::vector<int32_t> hi(static_cast<size_t>(topk)), ht(static_cast<size_t>(pages));
  for (int i = 0; i < topk; ++i) hi[i] = i % slots;
  for (int i = 0; i < pages; ++i) ht[i] = i;
  cudaMemcpy(idx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(tab, ht.data(), ht.size() * 4, cudaMemcpyHostToDevice);

  kops::attn::SparseMLAParams p{};
  p.q = q;
  // One query row that holds all `heads` heads: the row stride is the full head
  // block (the host API rejects overlapping head rows).
  p.q_stride_n = static_cast<int64_t>(heads) * 576;
  p.q_stride_h = 576;
  p.q_dtype = kops::DType::kBFloat16;
  p.primary.kv.k_data = kv;
  p.primary.kv.k_dtype = kops::DType::kBFloat16;
  p.primary.kv.k_codec = kops::kv::KVCodec::kNative;
  p.primary.kv.layout = kops::kv::KVLayout::kLatent;
  p.primary.kv.page_size = page;
  p.primary.kv.num_kv_heads = 1;
  p.primary.kv.head_dim_k = 576;
  p.primary.kv.head_dim_v = 512;
  p.primary.kv.stride_unit = kops::StrideUnit::kElement;
  p.primary.kv.k_stride_page = page * 576;
  p.primary.kv.k_stride_token = 576;
  p.primary.kv.table_format = kops::kv::PageTableFormat::kBlockTable;
  p.primary.kv.block_table = tab;
  p.primary.kv.block_table_stride = pages;
  p.primary.indices = idx;
  p.primary.topk = topk;
  p.mod.sm_scale = 1.0f / std::sqrt(576.0f);
  p.o = o;
  p.o_stride_n = static_cast<int64_t>(heads) * 512;
  p.o_stride_h = 512;
  p.o_dtype = kops::DType::kBFloat16;
  p.lse = lse;
  p.num_tokens = 1;
  p.num_heads = heads;
  p.qk_dim = 576;
  p.v_dim = 512;
  auto call = [&]() { return kops::attn::sparse_mla_run(p, ctx); };
  ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(kv);
  cudaFree(q);
  cudaFree(o);
  cudaFree(idx);
  cudaFree(tab);
  cudaFree(lse);
  return ok;
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
    bool ok = false;
    if (c.op == "sparse_topk") ok = run_sparse_topk(c, ctx, &st);
    else if (c.op == "lightning_indexer") ok = run_indexer(c, ctx, &st);
    else if (c.op == "sparse_mla") ok = run_sparse_mla(c, ctx, &st);
    if (!ok) {
      std::fprintf(stderr, "[bench] %s failed\n", c.id.c_str());
      return 1;
    }
    results.push_back({&c, st});
  }

  print_table(results, dev);
  if (opt.json_path != nullptr) write_results(opt.json_path, results, dev);
  return 0;
}
