// =============================================================================
// kernel-ops | benchmark/bench_sparse.cu
// B14 A11 sparse cases: exact top-K over an fp32 score row for rows = 1 / 16,
// max_kv = 64K / 1M and topk = 512 / 2048; the DSA/CSA indexer and sparse MLA;
// and DeepSeek-V4 compressed attention (CSA selected / HCA dense range + SWA +
// sink) at T = 1 / 16, 4K / 64K compressed entries and topk = 512 / 2048. The
// case table is the shared spec read by benchmark/compare/torch_baseline.py and
// benchmark/compare/csa_torch.py through --dump-cases.
//
// Split heuristics: sparse_topk reserves one scratch slice per row
// (sparse_topk_rows_workspace_bytes), so the r16/r4 1M cases take the multi-CTA
// fast path too; sparse_mla and compressed_attention call their *_ws entry when
// the natural grid (T * ceil(H/16)) is below the partition's compute units and
// the printed "split S=..." line reports the resolved count. The two
// ".plain" compressed-attention cases force the pre-split launch for A/B.
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

/// DeepSeek-V4 compressed-attention case: T decode rows x Hq shared-KV heads,
/// attending to `topk` compressed entries (CSA selected, or HCA dense range)
/// plus a `window`-token SWA branch and a sink.
struct CAttnBench {
  const char* id;
  int rows;       // T
  int heads;      // Hq
  int pool;       // compressed entries in the cache
  int topk;       // CSA selected entries per row; HCA dense-range cap
  int window;     // SWA raw tokens
  bool hca;       // dense compressed range (num_compressed) instead of `selected`
  int warmup;
  int iters;
  // 0 = auto: the split-KV ws entry when T*ceil(Hq/16) < compute_units (the shape leaves the
  // partition idle), else the plain one-CTA-per-16-heads launch. 1 = always plain, so a case pair
  // can A/B the split without changing any other ID.
  int plain = 0;
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

const std::vector<CAttnBench>& cattn_benches() {
  static const std::vector<CAttnBench> b = {
      // CSA: indexer-selected compressed entries + SWA + sink (V4 decode).
      {"compressed_attention.bf16.csa.t1.h128.pool4k.topk512.w512.sink", 1, 128, 4096, 512, 512,
       false, 3, 20},
      {"compressed_attention.bf16.csa.t1.h128.pool4k.topk2048.w512.sink", 1, 128, 4096, 2048, 512,
       false, 3, 20},
      {"compressed_attention.bf16.csa.t1.h128.pool64k.topk2048.w512.sink", 1, 128, 65536, 2048, 512,
       false, 3, 20},
      {"compressed_attention.bf16.csa.t16.h128.pool4k.topk2048.w512.sink", 16, 128, 4096, 2048,
       512, false, 3, 10},
      {"compressed_attention.bf16.csa.t16.h128.pool64k.topk2048.w512.sink", 16, 128, 65536, 2048,
       512, false, 3, 10},
      // Same two shapes forced onto the plain launch (no split): the A/B pair for the t1 split.
      {"compressed_attention.bf16.csa.t1.h128.pool4k.topk512.w512.sink.plain", 1, 128, 4096, 512,
       512, false, 3, 20, 1},
      {"compressed_attention.bf16.csa.t1.h128.pool4k.topk2048.w512.sink.plain", 1, 128, 4096, 2048,
       512, false, 3, 20, 1},
      // HCA: the dense compressed range (64k tokens / m'=128 = 512 entries) + SWA + sink.
      {"compressed_attention.bf16.hca.t1.h128.dense512.w512.sink", 1, 128, 512, 512, 512, true, 3,
       20},
      {"compressed_attention.bf16.hca.t16.h128.dense512.w512.sink", 16, 128, 512, 512, 512, true, 3,
       10},
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

const CAttnBench* find_cattn(const std::string& id) {
  for (const CAttnBench& b : cattn_benches()) {
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
  for (const CAttnBench& b : cattn_benches()) {
    const int64_t keys = static_cast<int64_t>(b.topk) + b.window;
    const int64_t kvb = static_cast<int64_t>(b.rows) * keys * 512 * 2;
    const int64_t qob = static_cast<int64_t>(b.rows) * b.heads * 512 * 2 * 2 +
                        static_cast<int64_t>(b.rows) * b.heads * 4;
    // QK dot + shared-KV PV, 2 flops per element each, per (row, head, key).
    const int64_t flops = static_cast<int64_t>(b.rows) * b.heads * keys * 2 * 512 * 2;
    cases.push_back({b.id, "compressed_attention", kops::DType::kBFloat16, b.rows, 512,
                     kvb + qob, flops, b.warmup, b.iters});
  }
  return cases;
}

/// Deterministic uniform value in [-1, 1) from a counter: a counter-hash, so the
/// histogram is flat and the 4 passes are pure bandwidth (no tie-heavy path).
__device__ __forceinline__ float bench_unit(int64_t i, uint32_t seed) {
  uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  h *= 3266489917u;
  h ^= h >> 16;
  return static_cast<float>(static_cast<double>(h & 0xFFFFFFu) / 8388608.0 - 1.0);
}

__global__ void bench_fill_scores(float* p, int64_t n, uint32_t seed) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n; i += step) {
    p[i] = bench_unit(i, seed);
  }
}

__global__ void bench_fill_bf16(__nv_bfloat16* p, int64_t n, uint32_t seed) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n; i += step) {
    p[i] = __float2bfloat16(bench_unit(i, seed));
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
  // One scratch slice per wide row: the cooperative fast path splits every row across CTAs, so a
  // 16-row case needs 16 slices instead of the single-row lease.
  const size_t ws_bytes = kops::attn::sparse_topk_rows_workspace_bytes(static_cast<int32_t>(rows));
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
  // Auto split-KV: one query row of 128 heads is only 8 CTAs, so sparse_mla_run_ws multiplies the
  // grid by a host-chosen S (sparse_mla_split_count) and merges the fp32 partials. The plain entry
  // remains the fallback when the shape already fills the partition (or is not a TC shape).
  const int32_t mla_splits = kops::attn::sparse_mla_split_count(p, ctx);
  const size_t mla_ws_bytes = kops::attn::sparse_mla_workspace_bytes(p, ctx);
  void* mla_ws = nullptr;
  if (mla_ws_bytes > 0) {
    ok = cuda_ok(cudaMalloc(&mla_ws, mla_ws_bytes), "sparse_mla ws");
    if (!ok) return false;
  }
  const bool mla_split = mla_ws_bytes > 0;
  const kops::Workspace mla_lease{mla_ws, mla_ws_bytes};
  if (ok) {
    std::fprintf(stderr, "[bench] %s: split S=%d, %zu B scratch\n", c.id.c_str(), mla_splits,
                 mla_ws_bytes);
  }
  auto call = [&]() {
    return mla_split ? kops::attn::sparse_mla_run_ws(p, mla_lease, ctx)
                     : kops::attn::sparse_mla_run(p, ctx);
  };
  ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(mla_ws);
  cudaFree(kv);
  cudaFree(q);
  cudaFree(o);
  cudaFree(idx);
  cudaFree(tab);
  cudaFree(lse);
  return ok;
}

// -----------------------------------------------------------------------------
// B14 DeepSeek-V4 CSA/HCA compressed attention
// -----------------------------------------------------------------------------

/// Shared-KV MQA over the selected compressed entries (CSA) or the dense
/// compressed range (HCA) + a sliding-window branch over the raw cache tail +
/// a per-head sink, one softmax. Compressed and window caches are paged with an
/// identity block table (a page per 64 entries/tokens), so `selected` can carry
/// physical slots and the dense/window branches go through the page table.
bool run_compressed_attention(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const CAttnBench* b = find_cattn(c.id);
  if (b == nullptr) return false;
  constexpr int kDim = 512;
  constexpr int64_t kPage = 64;
  constexpr int64_t kContext = 65536;   // decode at a 64k context: the window is its tail
  const int64_t rows = b->rows, heads = b->heads, pool = b->pool, topk = b->topk;
  const int64_t window = b->window;
  const int64_t pool_pages = pool / kPage;
  const int64_t win_page_stride = kContext / kPage;             // pages covering the context
  const int64_t win_pages = (window + kPage - 1) / kPage;       // pages the SWA branch touches
  const int64_t win_rows = rows * win_pages * kPage;

  void* q = nullptr;
  void* comp = nullptr;
  void* win = nullptr;
  void* o = nullptr;
  float* lse = nullptr;
  float* sink = nullptr;
  int32_t* selected = nullptr;
  int32_t* tab_comp = nullptr;
  int32_t* tab_win = nullptr;
  int32_t* ncomp = nullptr;
  int32_t* t2r = nullptr;
  int64_t* positions = nullptr;
  const size_t qo_bytes = static_cast<size_t>(rows * heads * kDim) * sizeof(__nv_bfloat16);
  bool ok = cuda_ok(cudaMalloc(&q, qo_bytes), "q") &&
            cuda_ok(cudaMalloc(&comp, static_cast<size_t>(pool * kDim) * sizeof(__nv_bfloat16)),
                    "compressed") &&
            cuda_ok(cudaMalloc(&win, static_cast<size_t>(win_rows * kDim) * sizeof(__nv_bfloat16)),
                    "window") &&
            cuda_ok(cudaMalloc(&o, qo_bytes), "o") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&lse),
                               static_cast<size_t>(rows * heads) * 4),
                    "lse") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sink), static_cast<size_t>(heads) * 4),
                    "sink") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&selected),
                               static_cast<size_t>(rows * topk) * 4),
                    "selected") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&tab_comp),
                               static_cast<size_t>(rows * pool_pages) * 4),
                    "tab_compressed") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&tab_win),
                               static_cast<size_t>(rows * win_page_stride) * 4),
                    "tab_window") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ncomp), static_cast<size_t>(rows) * 4),
                    "num_compressed") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&t2r), static_cast<size_t>(rows) * 4),
                    "token_to_request") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&positions), static_cast<size_t>(rows) * 8),
                    "positions");
  if (!ok) return false;
  bench_fill_bf16<<<256, 256>>>(static_cast<__nv_bfloat16*>(q), rows * heads * kDim, 37u);
  bench_fill_bf16<<<256, 256>>>(static_cast<__nv_bfloat16*>(comp), pool * kDim, 41u);
  bench_fill_bf16<<<256, 256>>>(static_cast<__nv_bfloat16*>(win), win_rows * kDim, 43u);
  bench_fill_scores<<<256, 256>>>(sink, heads, 47u);

  std::vector<int64_t> host_pos(static_cast<size_t>(rows), kContext - 1);
  std::vector<int32_t> host_t2r(static_cast<size_t>(rows)), host_ncomp(static_cast<size_t>(rows),
                                                                      static_cast<int32_t>(topk));
  std::vector<int32_t> host_tab_comp(static_cast<size_t>(rows * pool_pages));
  std::vector<int32_t> host_tab_win(static_cast<size_t>(rows * win_page_stride), 0);
  std::vector<int32_t> host_sel(static_cast<size_t>(rows * topk));
  for (int64_t r = 0; r < rows; ++r) {
    host_t2r[static_cast<size_t>(r)] = static_cast<int32_t>(r);
    for (int64_t pg = 0; pg < pool_pages; ++pg) {
      host_tab_comp[static_cast<size_t>(r * pool_pages + pg)] = static_cast<int32_t>(pg);
    }
    // Each request owns the `win_pages` pages its window covers; the slot
    // pattern is distinct (4099 is coprime with the power-of-two pool).
    const int64_t first_page = win_page_stride - win_pages;
    for (int64_t pg = first_page; pg < win_page_stride; ++pg) {
      host_tab_win[static_cast<size_t>(r * win_page_stride + pg)] =
          static_cast<int32_t>(r * win_pages + (pg - first_page));
    }
    for (int64_t i = 0; i < topk; ++i) {
      host_sel[static_cast<size_t>(r * topk + i)] =
          static_cast<int32_t>((i * 4099 + r * 7919) % pool);
    }
  }
  cudaMemcpy(selected, host_sel.data(), host_sel.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(tab_comp, host_tab_comp.data(), host_tab_comp.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(tab_win, host_tab_win.data(), host_tab_win.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(ncomp, host_ncomp.data(), host_ncomp.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(t2r, host_t2r.data(), host_t2r.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(positions, host_pos.data(), host_pos.size() * 8, cudaMemcpyHostToDevice);

  kops::attn::CompressedAttentionParams p{};
  p.q = q;
  p.q_stride_n = heads * kDim;
  p.q_stride_h = kDim;
  p.q_dtype = kops::DType::kBFloat16;
  p.compressed.k_data = comp;
  p.compressed.k_dtype = kops::DType::kBFloat16;
  p.compressed.k_codec = kops::kv::KVCodec::kNative;
  p.compressed.layout = kops::kv::KVLayout::kLatent;
  p.compressed.page_size = static_cast<int32_t>(kPage);
  p.compressed.num_kv_heads = 1;
  p.compressed.head_dim_k = kDim;
  p.compressed.head_dim_v = kDim;
  p.compressed.stride_unit = kops::StrideUnit::kElement;
  p.compressed.k_stride_page = kPage * kDim;
  p.compressed.k_stride_token = kDim;
  p.compressed.table_format = kops::kv::PageTableFormat::kBlockTable;
  p.compressed.block_table = tab_comp;
  p.compressed.block_table_stride = static_cast<int32_t>(pool_pages);
  p.selected = b->hca ? nullptr : selected;
  p.selected_stride = static_cast<int32_t>(topk);
  p.topk = static_cast<int32_t>(topk);
  p.num_compressed = b->hca ? ncomp : nullptr;
  p.window.k_data = win;
  p.window.k_dtype = kops::DType::kBFloat16;
  p.window.k_codec = kops::kv::KVCodec::kNative;
  p.window.layout = kops::kv::KVLayout::kLatent;
  p.window.page_size = static_cast<int32_t>(kPage);
  p.window.num_kv_heads = 1;
  p.window.head_dim_k = kDim;
  p.window.head_dim_v = kDim;
  p.window.stride_unit = kops::StrideUnit::kElement;
  p.window.k_stride_page = kPage * kDim;
  p.window.k_stride_token = kDim;
  p.window.table_format = kops::kv::PageTableFormat::kBlockTable;
  p.window.block_table = tab_win;
  p.window.block_table_stride = static_cast<int32_t>(win_page_stride);
  p.window_size = static_cast<int32_t>(window);
  p.token_to_request = t2r;
  p.positions = positions;
  p.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(kDim));
  p.mod.sink_logits = sink;
  p.o = o;
  p.o_stride_n = heads * kDim;
  p.o_stride_h = kDim;
  p.o_dtype = kops::DType::kBFloat16;
  p.lse = lse;
  p.num_tokens = static_cast<int32_t>(rows);
  p.num_q_heads = static_cast<int32_t>(heads);
  p.entry_dim = kDim;
  p.value_dim = kDim;

  // Auto split-KV: a t1 x Hq128 launch is 8 CTAs, so a 132-SM partition runs at 6% occupancy. The
  // ws entry splits the visible key list (compressed + window, still one softmax) across grid.z
  // and merges the fp32 partials. The entry point is taken only when the plain grid leaves the
  // partition more than 4x idle (t1: 8 CTAs); t16 (128 CTAs, ~one per SM) keeps the plain launch
  // so the existing t16 cases are measured on the body they were tuned on.
  const int64_t ca_groups = (static_cast<int64_t>(p.num_q_heads) + 15) / 16;
  const int32_t ca_units = ctx.compute_units > 0 ? ctx.compute_units : 32;
  const int32_t ca_splits = kops::attn::compressed_attention_split_count(p, ctx);
  const bool ca_split =
      b->plain == 0 && 4 * static_cast<int64_t>(p.num_tokens) * ca_groups < ca_units &&
      ca_splits > 1;
  const size_t ca_ws_bytes =
      ca_split ? kops::attn::compressed_attention_workspace_bytes(p, ctx) : 0;
  void* ca_ws = nullptr;
  if (ca_ws_bytes > 0) {
    ok = cuda_ok(cudaMalloc(&ca_ws, ca_ws_bytes), "compressed_attention ws");
    if (!ok) return false;
  }
  const kops::Workspace ca_lease{ca_ws, ca_ws_bytes};
  if (ok) {
    std::fprintf(stderr, "[bench] %s: split S=%d, %zu B scratch\n", c.id.c_str(),
                 ca_split ? ca_splits : 1, ca_ws_bytes);
  }
  auto call = [&]() {
    return ca_split ? kops::attn::compressed_attention_ws(p, ca_lease, ctx)
                    : kops::attn::compressed_attention(p, ctx);
  };
  ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(ca_ws);

  cudaFree(q);
  cudaFree(comp);
  cudaFree(win);
  cudaFree(o);
  cudaFree(lse);
  cudaFree(sink);
  cudaFree(selected);
  cudaFree(tab_comp);
  cudaFree(tab_win);
  cudaFree(ncomp);
  cudaFree(t2r);
  cudaFree(positions);
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
    else if (c.op == "compressed_attention") ok = run_compressed_attention(c, ctx, &st);
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
