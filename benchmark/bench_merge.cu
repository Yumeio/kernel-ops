// =============================================================================
// kernel-ops | benchmark/bench_merge.cu
// merge_state (split-KV combine), B8 native attention (prefill FA2, CUDA-core /
// tensor-core / dynamic decode) and B17 experimental low-bit rotated KV store /
// decode / prefill. The case table is the shared spec read by
// benchmark/compare/torch_baseline.py through --dump-cases (the lowbit ops have
// no PyTorch equivalent and are skipped there).
// =============================================================================
#include "benchmark/harness.h"

#include <cmath>
#include <functional>
#include <vector>

#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/kvcache/kvcache_kernels.cuh"
#include "csrc/kvcache/ops.h"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/lowbit_kernels.cuh"
#include "csrc/attention/ops.h"

void add_attn_cases(std::vector<kops_bench::Case>& cases);  // B8 attention cases (below)
void add_lowbit_cases(std::vector<kops_bench::Case>& cases);

namespace {

using namespace kops_bench;

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto add = [&](int64_t batch, int64_t heads, int64_t head_dim) {
    const int64_t rows = batch * heads;
    const int64_t elems = rows * head_dim;
    // o_a + o_b read, o written; lse_a + lse_b read, lse written (all fp32).
    const int64_t bytes = 3 * elems * 4 + 3 * rows * 4;
    char id[64];
    std::snprintf(id, sizeof(id), "merge_state.fp32.b%lld.h%lld.d%lld",
                  static_cast<long long>(batch), static_cast<long long>(heads),
                  static_cast<long long>(head_dim));
    cases.push_back({id, "merge_state", kops::DType::kFloat32, rows, head_dim, bytes,
                     6 * elems, 20, 200});
  };
  add(8, 32, 128);
  add(64, 32, 128);
  {
    const int64_t elems = 64 * 32 * 128;
    const int64_t lse = 64 * 32;
    cases.push_back({"merge_states.fp32.b64.h32.d128.n8", "merge_states",
                     kops::DType::kFloat32, 64, 128, 9 * elems * 4 + 9 * lse * 4, 0, 10, 100});
    cases.push_back({"merge_split_kv.fp32.b64.h32.d128.s8", "merge_split_kv",
                     kops::DType::kFloat32, 64, 128, 9 * elems * 4 + 9 * lse * 4, 0, 10, 100});
  }
  {
    const int64_t t = 1024, h = 8, nope = 512, pe = 64;
    cases.push_back({"concat_mla_q.bf16.t1024.h8.n512.p64", "concat_mla_q",
                     kops::DType::kBFloat16, t, nope + pe,
                     t * h * (nope + pe) * 2 * 2, 0, 10, 100});
    const int64_t tokens = 2048;
    cases.push_back({"mla_context_gather.bf16.t2048.r512.p64", "mla_context_gather",
                     kops::DType::kBFloat16, tokens, 576, tokens * 576 * 2 * 2, 0, 10, 100});
  }
  add_attn_cases(cases);
  add_lowbit_cases(cases);
  return cases;
}

bool run_concat_mla_q(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int64_t kT = 1024, kH = 8, kNope = 512, kPe = 64;
  __nv_bfloat16* ql = nullptr;
  __nv_bfloat16* qp = nullptr;
  __nv_bfloat16* qo = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ql), static_cast<size_t>(kT * kH * kNope) * 2),
               "ql") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&qp), static_cast<size_t>(kT * kH * kPe) * 2),
               "qp") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&qo),
                          static_cast<size_t>(kT * kH * (kNope + kPe)) * 2),
               "qo")) {
    return false;
  }
  cudaMemset(ql, 1, static_cast<size_t>(kT * kH * kNope) * 2);
  cudaMemset(qp, 1, static_cast<size_t>(kT * kH * kPe) * 2);
  const kops::Tensor tql = tensor3d(ql, kops::DType::kBFloat16, kT, kH, kNope);
  const kops::Tensor tqp = tensor3d(qp, kops::DType::kBFloat16, kT, kH, kPe);
  const kops::Tensor tqo = tensor3d(qo, kops::DType::kBFloat16, kT, kH, kNope + kPe);
  bool ok = status_ok(kops::attn::concat_mla_q(tqo, tql, tqp, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::attn::concat_mla_q(tqo, tql, tqp, ctx); }, st);
  cudaFree(ql);
  cudaFree(qp);
  cudaFree(qo);
  return ok;
}

bool run_mla_context_gather(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int64_t kTokens = 2048, kRank = 512, kPe = 64, kPage = 16;
  const int pages = static_cast<int>(kTokens / kPage);
  __nv_bfloat16* cache = nullptr;
  __nv_bfloat16* ckv = nullptr;
  __nv_bfloat16* kpe = nullptr;
  int32_t* table = nullptr;
  int32_t* cu = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cache),
                          static_cast<size_t>(pages * kPage * (kRank + kPe)) * 2),
               "cache") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ckv), static_cast<size_t>(kTokens * kRank) * 2),
               "ckv") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&kpe), static_cast<size_t>(kTokens * kPe) * 2),
               "kpe") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&table), static_cast<size_t>(pages) * 2 * 4),
               "table") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cu), 3 * 4), "cu")) {
    return false;
  }
  std::vector<int32_t> tab(static_cast<size_t>(pages) * 2);
  for (int i = 0; i < pages; ++i) {
    tab[static_cast<size_t>(i)] = i;
    tab[static_cast<size_t>(pages + i)] = i;
  }
  cudaMemcpy(table, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(cache, 1, static_cast<size_t>(pages * kPage * (kRank + kPe)) * 2);
  kops::kv::PagedKVDesc d{};
  d.k_data = cache;
  d.k_dtype = kops::DType::kBFloat16;
  d.v_dtype = kops::DType::kBFloat16;
  d.k_codec = kops::kv::KVCodec::kNative;
  d.v_codec = kops::kv::KVCodec::kNative;
  d.layout = kops::kv::KVLayout::kLatent;
  d.page_size = static_cast<int32_t>(kPage);
  d.num_kv_heads = 1;
  d.head_dim_k = static_cast<int32_t>(kRank + kPe);
  d.head_dim_v = static_cast<int32_t>(kRank);
  d.stride_unit = kops::StrideUnit::kElement;
  d.k_stride_page = kPage * (kRank + kPe);
  d.k_stride_token = kRank + kPe;
  d.k_stride_head = kRank + kPe;
  d.table_format = kops::kv::PageTableFormat::kBlockTable;
  d.block_table = table;
  d.block_table_stride = pages;
  // Two requests, 1024 context tokens each; ctx_start = 0 for both.
  std::vector<int32_t> cv2 = {0, static_cast<int32_t>(kTokens / 2), static_cast<int32_t>(kTokens)};
  cudaMemcpy(cu, cv2.data(), cv2.size() * 4, cudaMemcpyHostToDevice);
  int32_t* dev_start = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dev_start), 2 * 4), "start")) return false;
  std::vector<int32_t> sv = {0, 0};
  cudaMemcpy(dev_start, sv.data(), sv.size() * 4, cudaMemcpyHostToDevice);
  const kops::Tensor tc = tensor2d(ckv, kops::DType::kBFloat16, kTokens, kRank);
  const kops::Tensor tp = tensor2d(kpe, kops::DType::kBFloat16, kTokens, kPe);
  const kops::Tensor ti2 = tensor1d(cu, kops::DType::kInt32, 3);
  const kops::Tensor tst2 = tensor1d(dev_start, kops::DType::kInt32, 2);
  bool ok = status_ok(kops::attn::mla_context_gather(tc, tp, d, ti2, tst2, ctx), c.id.c_str());
  if (ok) {
    ok = measure(c, [&] { (void)kops::attn::mla_context_gather(tc, tp, d, ti2, tst2, ctx); }, st);
  }
  cudaFree(cache);
  cudaFree(ckv);
  cudaFree(kpe);
  cudaFree(table);
  cudaFree(cu);
  cudaFree(dev_start);
  return ok;
}

bool run_merge(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  long long batch = 0, heads = 0, head_dim = 0;
  if (std::sscanf(c.id.c_str(), "merge_state.fp32.b%lld.h%lld.d%lld", &batch, &heads,
                  &head_dim) != 3 ||
      batch <= 0 || heads <= 0 || head_dim <= 0) {
    std::fprintf(stderr, "[bench] cannot parse %s\n", c.id.c_str());
    return false;
  }
  const size_t elems = static_cast<size_t>(c.rows * c.cols);
  const size_t lse_n = static_cast<size_t>(batch * heads);
  float* o = nullptr;
  float* oa = nullptr;
  float* ob = nullptr;
  float* lse = nullptr;
  float* la = nullptr;
  float* lb = nullptr;
  auto alloc = [](float** p, size_t n, const char* what) {
    return cuda_ok(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(float)), what);
  };
  if (!alloc(&o, elems, "cudaMalloc o") || !alloc(&oa, elems, "cudaMalloc o_a") ||
      !alloc(&ob, elems, "cudaMalloc o_b") || !alloc(&lse, lse_n, "cudaMalloc lse") ||
      !alloc(&la, lse_n, "cudaMalloc lse_a") || !alloc(&lb, lse_n, "cudaMalloc lse_b")) {
    return false;
  }
  cudaMemset(o, 0, elems * sizeof(float));
  cudaMemset(oa, 0, elems * sizeof(float));
  cudaMemset(ob, 0, elems * sizeof(float));
  cudaMemset(lse, 0, lse_n * sizeof(float));
  cudaMemset(la, 0, lse_n * sizeof(float));
  cudaMemset(lb, 0, lse_n * sizeof(float));

  const kops::Tensor to = tensor3d(o, kops::DType::kFloat32, batch, heads, head_dim);
  const kops::Tensor toa = tensor3d(oa, kops::DType::kFloat32, batch, heads, head_dim);
  const kops::Tensor tob = tensor3d(ob, kops::DType::kFloat32, batch, heads, head_dim);
  const kops::Tensor tls = tensor2d(lse, kops::DType::kFloat32, batch, heads);
  const kops::Tensor tla = tensor2d(la, kops::DType::kFloat32, batch, heads);
  const kops::Tensor tlb = tensor2d(lb, kops::DType::kFloat32, batch, heads);

  bool ok = status_ok(kops::attn::merge_state(to, tls, toa, tla, tob, tlb, ctx), c.id.c_str());
  if (ok) {
    ok = measure(
        c, [&] { (void)kops::attn::merge_state(to, tls, toa, tla, tob, tlb, ctx); }, st);
  }

  cudaFree(o);
  cudaFree(oa);
  cudaFree(ob);
  cudaFree(lse);
  cudaFree(la);
  cudaFree(lb);
  return ok;
}

}  // namespace

bool run_states(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int batch = static_cast<int>(c.rows), heads = 32, d = static_cast<int>(c.cols);
  const int n = 8;
  const size_t elems = static_cast<size_t>(batch) * heads * d;
  const size_t lse_n = static_cast<size_t>(batch) * heads;
  float* o = nullptr; float* lse = nullptr; float* os = nullptr; float* ls = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&o), elems * sizeof(float)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&lse), lse_n * sizeof(float)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&os), elems * n * sizeof(float)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&ls), lse_n * n * sizeof(float)) != cudaSuccess) {
    return false;
  }
  const kops::Tensor to = tensor3d(o, kops::DType::kFloat32, batch, heads, d);
  const kops::Tensor tl = tensor2d(lse, kops::DType::kFloat32, batch, heads);
  const kops::Tensor tos = tensor4d(os, kops::DType::kFloat32, batch, n, heads, d);
  const kops::Tensor tls = tensor4d(ls, kops::DType::kFloat32, batch, n, heads, 1);
  // lse_states is [rows, N, heads], not [rows,N,heads,1]; build a 3-D view.
  kops::Tensor tls3 = tls;
  tls3.ndim = 3;
  tls3.shape[3] = 0;
  tls3.stride[3] = 0;
  bool ok = status_ok(kops::attn::merge_states(to, tl, tos, tls3, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::attn::merge_states(to, tl, tos, tls3, ctx); }, st);
  cudaFree(o); cudaFree(lse); cudaFree(os); cudaFree(ls);
  return ok;
}

bool run_split(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int batch = static_cast<int>(c.rows), heads = 32, d = static_cast<int>(c.cols);
  const int splits = 8;
  const int64_t pr = static_cast<int64_t>(batch) * splits;
  float* o = nullptr; float* lse = nullptr; float* op = nullptr; float* lp = nullptr;
  int32_t* indptr = nullptr; float* sink = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&o), static_cast<size_t>(batch * heads * d) * 4) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&lse), static_cast<size_t>(batch * heads) * 4) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&op), static_cast<size_t>(pr * heads * d) * 4) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&lp), static_cast<size_t>(pr * heads) * 4) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&indptr), static_cast<size_t>(batch + 1) * 4) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&sink), static_cast<size_t>(heads) * 4) != cudaSuccess) {
    return false;
  }
  std::vector<int32_t> hind(static_cast<size_t>(batch) + 1);
  for (int i = 0; i <= batch; ++i) hind[static_cast<size_t>(i)] = i * splits;
  cudaMemcpy(indptr, hind.data(), hind.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  const kops::Tensor to = tensor3d(o, kops::DType::kFloat32, batch, heads, d);
  const kops::Tensor tl = tensor2d(lse, kops::DType::kFloat32, batch, heads);
  const kops::Tensor tsink = tensor1d(sink, kops::DType::kFloat32, heads);
  kops::attn::SplitKVBuffers split{op, lp, indptr, splits};
  bool ok = status_ok(kops::attn::merge_split_kv(to, tl, split, tsink, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::attn::merge_split_kv(to, tl, split, tsink, ctx); }, st);
  cudaFree(o); cudaFree(lse); cudaFree(op); cudaFree(lp); cudaFree(indptr); cudaFree(sink);
  return ok;
}

// ---- B8 native attention ---------------------------------------------------------------------------
// Prefill cases report FLOPs (4 * D * visible (q, k) pairs * Hq); decode cases report the K/V bytes
// read (the roofline of a memory-bound decode).  Data: uniform [-1, 1) bf16, identity page table.

struct AttnBench {
  const char* id;
  bool decode, dynamic, paged;
  int batch, q, kv, Hq, Hkv, D;
  kops::DType kv_dt;
  kops::attn::MaskMode mask;
  bool gemma_mods;            // softcap 50 + sliding window 4096 (cost of the modifier path)
  std::vector<int> kv_lens;   // per-request kv (empty: all = kv)
};

const std::vector<AttnBench>& attn_benches() {
  using kops::DType;
  using kops::attn::MaskMode;
  static const std::vector<AttnBench> b = {
      {"prefill.ragged.bf16.d128.causal.s2048.h32x8", false, false, false, 1, 2048, 2048, 32, 8, 128,
       DType::kBFloat16, MaskMode::kCausal, false, {}},
      {"prefill.ragged.bf16.d128.full.b4.s1024.h32x8", false, false, false, 4, 1024, 1024, 32, 8, 128,
       DType::kBFloat16, MaskMode::kNone, false, {}},
      {"prefill.ragged.bf16.d128.causal.s2048.h32x8.softcap_swa", false, false, false, 1, 2048, 2048, 32, 8,
       128, DType::kBFloat16, MaskMode::kCausal, true, {}},
      {"prefill.ragged.bf16.d64.causal.s2048.h64x8", false, false, false, 1, 2048, 2048, 64, 8, 64,
       DType::kBFloat16, MaskMode::kCausal, false, {}},
      {"prefill.ragged.bf16.d256.causal.s1024.h8x4", false, false, false, 1, 1024, 1024, 8, 4, 256,
       DType::kBFloat16, MaskMode::kCausal, false, {}},
      {"prefill.paged.bf16.d128.chunk512.kv4096.ps16.h32x8", false, false, true, 1, 512, 4096, 32, 8, 128,
       DType::kBFloat16, MaskMode::kCausal, false, {}},
      {"prefill.paged.e4m3.d128.chunk512.kv4096.ps16.h32x8", false, false, true, 1, 512, 4096, 32, 8, 128,
       DType::kFloat8E4M3, MaskMode::kCausal, false, {}},
      {"decode.simt.bf16.d128.b8.kv4096.h32x32", true, false, true, 8, 1, 4096, 32, 32, 128,
       DType::kBFloat16, MaskMode::kCausal, false, {}},
      {"decode.simt.e4m3.d128.b8.kv4096.h32x32", true, false, true, 8, 1, 4096, 32, 32, 128,
       DType::kFloat8E4M3, MaskMode::kCausal, false, {}},
      {"decode.tc.bf16.d128.b32.kv2048.h32x8", true, false, true, 32, 1, 2048, 32, 8, 128,
       DType::kBFloat16, MaskMode::kCausal, false, {}},
      {"decode.tc.bf16.d128.b32.kv2048.h32x8.q4", true, false, true, 32, 4, 2048, 32, 8, 128,
       DType::kBFloat16, MaskMode::kCausal, false, {}},
      {"decode.tc.e4m3.d128.b32.kv2048.h32x8", true, false, true, 32, 1, 2048, 32, 8, 128,
       DType::kFloat8E4M3, MaskMode::kCausal, false, {}},
      {"decode.tc.bf16.d128.b1.kv32768.h32x8", true, false, true, 1, 1, 32768, 32, 8, 128,
       DType::kBFloat16, MaskMode::kCausal, false, {}},
      {"decode.tc.bf16.d128.skew1x32768+31x1024.h32x8", true, false, true, 32, 1, 32768, 32, 8, 128,
       DType::kBFloat16, MaskMode::kCausal, false, {32768}},
      {"decode.dynamic.bf16.d128.skew1x32768+31x1024.h32x8", true, true, true, 32, 1, 32768, 32, 8, 128,
       DType::kBFloat16, MaskMode::kCausal, false, {32768}},
  };
  return b;
}

const AttnBench* find_attn(const std::string& id) {
  for (const AttnBench& a : attn_benches()) {
    if (id == a.id) return &a;
  }
  return nullptr;
}

std::vector<int> attn_kv_lens(const AttnBench& a) {
  std::vector<int> kv(static_cast<size_t>(a.batch), a.kv_lens.empty() ? a.kv : 1024);
  for (size_t i = 0; i < a.kv_lens.size() && i < kv.size(); ++i) kv[i] = a.kv_lens[i];
  return kv;
}

void add_attn_cases(std::vector<Case>& cases) {
  for (const AttnBench& a : attn_benches()) {
    const std::vector<int> kv = attn_kv_lens(a);
    int64_t pairs = 0, kv_tokens = 0;
    for (int len : kv) {
      kv_tokens += len;
      for (int i = 0; i < a.q; ++i) {
        const int64_t pos = len - a.q + i;
        pairs += a.mask == kops::attn::MaskMode::kCausal ? pos + 1 : len;
      }
    }
    const int kvb = a.kv_dt == kops::DType::kFloat8E4M3 ? 1 : 2;
    const int64_t bytes = kv_tokens * a.Hkv * a.D * 2 * kvb;
    const int64_t flops = 4 * pairs * a.D * a.Hq;
    cases.push_back({a.id, "attention", kops::DType::kBFloat16, static_cast<int64_t>(a.batch) * a.q, a.D,
                     a.decode ? bytes : 0, a.decode ? 0 : flops, 3, a.decode ? 100 : 40});
  }
}

__global__ void bench_fill_bf16(__nv_bfloat16* p, int64_t n, uint32_t seed) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    p[i] = __float2bfloat16(static_cast<float>(h & 0xFFFF) / 32768.0f - 1.0f);
  }
}

bool run_attention(const Case& c, const kops::LaunchContext& ctx_in, Stats* st) {
  const AttnBench* a = find_attn(c.id);
  if (a == nullptr) return false;
  kops::LaunchContext ctx = ctx_in;
  const int B = a->batch, D = a->D, ps = 16;
  const std::vector<int> kv = attn_kv_lens(*a);
  std::vector<int32_t> qo(B + 1, 0), kv_tok(B + 1, 0), page_indptr(B + 1, 0), last(B), pages;
  for (int b = 0; b < B; ++b) {
    qo[b + 1] = qo[b] + a->q;
    kv_tok[b + 1] = kv_tok[b] + kv[b];
    const int np = (kv[b] + ps - 1) / ps;
    page_indptr[b + 1] = page_indptr[b] + np;
    for (int i = 0; i < np; ++i) pages.push_back(page_indptr[b] + i);
    last[b] = (kv[b] - 1) % ps + 1;
  }
  const int64_t total_q = qo[B], pool = a->paged ? static_cast<int64_t>(page_indptr[B]) * ps : kv_tok[B];
  const int kvb = a->kv_dt == kops::DType::kFloat8E4M3 ? 1 : 2;
  void *q = nullptr, *o = nullptr, *k = nullptr, *v = nullptr, *pers = nullptr, *scr = nullptr;
  int32_t *d_qo = nullptr, *d_kv = nullptr, *d_pi = nullptr, *d_pg = nullptr, *d_last = nullptr;
  float* d_scale = nullptr;
  bool ok = cuda_ok(cudaMalloc(&q, static_cast<size_t>(total_q) * a->Hq * D * 2), "q") &&
            cuda_ok(cudaMalloc(&o, static_cast<size_t>(total_q) * a->Hq * D * 2), "o") &&
            cuda_ok(cudaMalloc(&k, static_cast<size_t>(pool) * a->Hkv * D * kvb), "k") &&
            cuda_ok(cudaMalloc(&v, static_cast<size_t>(pool) * a->Hkv * D * kvb), "v") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_qo), qo.size() * 4), "qo") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_kv), kv_tok.size() * 4), "kv") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_pi), page_indptr.size() * 4), "pi") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_pg), pages.size() * 4 + 4), "pg") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_last), last.size() * 4), "last") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_scale), 4), "scale");
  if (!ok) return false;
  bench_fill_bf16<<<256, 256>>>(static_cast<__nv_bfloat16*>(q), total_q * a->Hq * D, 1u);
  // K/V: bf16 values, or (FP8) the same bytes reinterpreted as e4m3 codes of moderate magnitude
  bench_fill_bf16<<<256, 256>>>(static_cast<__nv_bfloat16*>(k), pool * a->Hkv * D * kvb / 2, 2u);
  bench_fill_bf16<<<256, 256>>>(static_cast<__nv_bfloat16*>(v), pool * a->Hkv * D * kvb / 2, 3u);
  if (kvb == 1) {
    cudaMemset(k, 0x38, static_cast<size_t>(pool) * a->Hkv * D);
    cudaMemset(v, 0x30, static_cast<size_t>(pool) * a->Hkv * D);
  }
  const float one = 1.0f;
  cudaMemcpy(d_scale, &one, 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_qo, qo.data(), qo.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kv, kv_tok.data(), kv_tok.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pi, page_indptr.data(), page_indptr.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pg, pages.data(), pages.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_last, last.data(), last.size() * 4, cudaMemcpyHostToDevice);

  kops::attn::AttentionSpec spec;
  spec.num_q_heads = a->Hq;
  spec.num_kv_heads = a->Hkv;
  spec.head_dim_qk = spec.head_dim_vo = D;
  spec.kv_dtype = a->kv_dt;
  if (kvb == 1) spec.k_codec = spec.v_codec = kops::kv::KVCodec::kFP8PerTensor;
  spec.page_size = a->paged ? ps : 0;
  spec.mask = a->mask;
  spec.sliding_window = spec.softcap = a->gemma_mods;
  kops::attn::AttentionModifiers mod;
  mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
  if (a->gemma_mods) {
    mod.logits_softcap = 50.0f;
    mod.window_left = 4096;
  }
  kops::kv::PagedKVDesc desc;
  desc.k_data = k;
  desc.v_data = v;
  desc.k_dtype = desc.v_dtype = a->kv_dt;
  desc.k_codec = desc.v_codec = spec.k_codec;
  desc.page_size = ps;
  desc.num_kv_heads = a->Hkv;
  desc.head_dim_k = desc.head_dim_v = D;
  desc.k_stride_page = desc.v_stride_page = static_cast<int64_t>(ps) * a->Hkv * D;
  desc.k_stride_token = desc.v_stride_token = static_cast<int64_t>(a->Hkv) * D;
  desc.k_stride_head = desc.v_stride_head = D;
  desc.kv_indptr = d_pi;
  desc.kv_page_indices = d_pg;
  desc.kv_last_page_len = d_last;
  if (kvb == 1) desc.k_scale = desc.v_scale = d_scale;

  kops::attn::AttentionPlan plan;
  kops::Workspace ws{};
  std::vector<int32_t> kvl(kv.begin(), kv.end());
  kops::Tensor qo_t = tensor1d(qo.data(), kops::DType::kInt32, B + 1);
  qo_t.device = -1;
  kops::Tensor kv_t = tensor1d(kvl.data(), kops::DType::kInt32, B);
  kv_t.device = -1;
  auto do_plan = [&]() {
    return a->decode ? kops::attn::decode_plan(spec, B, a->kv, a->q, a->dynamic, ctx, ws, &plan)
                     : kops::attn::prefill_plan(spec, qo_t, kv_t, B, ctx, ws, &plan);
  };
  (void)do_plan();
  ok = cuda_ok(cudaMalloc(&pers, plan.workspace.device_persistent_bytes), "persistent") &&
       cuda_ok(cudaMalloc(&scr, std::max<size_t>(plan.workspace.device_scratch_bytes, 256)), "scratch");
  ws.device_persistent = pers;
  ws.device_persistent_bytes = plan.workspace.device_persistent_bytes;
  ws.device_scratch = scr;
  ws.device_scratch_bytes = std::max<size_t>(plan.workspace.device_scratch_bytes, 256);
  ok = ok && status_ok(do_plan(), c.id.c_str());
  std::function<kops::Status()> run;
  kops::attn::RaggedPrefillParams rp{};
  kops::attn::PagedPrefillParams pp{};
  kops::attn::DecodeParams dp{};
  if (a->decode) {
    dp.q = q;
    dp.q_stride_n = static_cast<int64_t>(a->Hq) * D;
    dp.q_stride_h = D;
    dp.q_len = a->q;
    dp.kv = desc;
    dp.mod = mod;
    dp.o = o;
    dp.o_stride_n = static_cast<int64_t>(a->Hq) * D;
    dp.o_stride_h = D;
    dp.batch_size = B;
    dp.num_qo_heads = a->Hq;
    dp.num_kv_heads = a->Hkv;
    run = [&]() { return kops::attn::decode_run(plan, dp, ws, ctx); };
  } else if (a->paged) {
    pp.q = q;
    pp.q_stride_n = static_cast<int64_t>(a->Hq) * D;
    pp.q_stride_h = D;
    pp.qo_indptr = d_qo;
    pp.kv = desc;
    pp.mod = mod;
    pp.o = o;
    pp.o_stride_n = static_cast<int64_t>(a->Hq) * D;
    pp.o_stride_h = D;
    pp.batch_size = B;
    pp.num_qo_heads = a->Hq;
    pp.num_kv_heads = a->Hkv;
    pp.total_q = static_cast<int32_t>(total_q);
    run = [&]() { return kops::attn::paged_prefill_run(plan, pp, ws, ctx); };
  } else {
    rp.q = q;
    rp.q_stride_n = static_cast<int64_t>(a->Hq) * D;
    rp.q_stride_h = D;
    rp.qo_indptr = d_qo;
    rp.kv.k = k;
    rp.kv.v = v;
    rp.kv.k_stride_n = rp.kv.v_stride_n = static_cast<int64_t>(a->Hkv) * D;
    rp.kv.k_stride_h = rp.kv.v_stride_h = D;
    rp.kv.kv_indptr = d_kv;
    rp.mod = mod;
    rp.o = o;
    rp.o_stride_n = static_cast<int64_t>(a->Hq) * D;
    rp.o_stride_h = D;
    rp.batch_size = B;
    rp.num_qo_heads = a->Hq;
    rp.num_kv_heads = a->Hkv;
    rp.total_q = static_cast<int32_t>(total_q);
    run = [&]() { return kops::attn::ragged_prefill_run(plan, rp, ws, ctx); };
  }
  ok = ok && status_ok(run(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)run(); }, st);
  for (void* p : {q, o, k, v, pers, scr, static_cast<void*>(d_qo), static_cast<void*>(d_kv),
                  static_cast<void*>(d_pi), static_cast<void*>(d_pg), static_cast<void*>(d_last),
                  static_cast<void*>(d_scale)}) {
    cudaFree(p);
  }
  return ok;
}

// ---- B17 low-bit rotated KV (experimental) ------------------------------------------------------
namespace lbt = kops::experimental::lowbit;

struct LbBench {
  const char* id;
  bool prefill;             // false: decode
  bool store;               // only lowbit_kv_store, no attention
  int batch, q, kv, Hq, Hkv, D, page;
  lbt::LowBitFormat kf, vf;
};

const std::vector<LbBench>& lb_benches() {
  using F = lbt::LowBitFormat;
  static const std::vector<LbBench> b = {
      {"lowbit.store.bf16.t2048.h8.d128.turbo4", false, true, 1, 1, 2048, 8, 8, 128, 16,
       F::kUniform4, F::kUniform4},
      {"lowbit.store.bf16.t2048.h8.d128.turbo3", false, true, 1, 1, 2048, 8, 8, 128, 16,
       F::kUniform3, F::kUniform4},
      {"lowbit.store.bf16.t4096.h8.d128.turbo4", false, true, 1, 1, 4096, 8, 8, 128, 16,
       F::kUniform4, F::kUniform4},
      {"lowbit.store.bf16.t4096.h8.d128.turbo3", false, true, 1, 1, 4096, 8, 8, 128, 16,
       F::kUniform3, F::kUniform4},
      {"lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4", false, false, 8, 1, 4096, 32, 8, 128, 16,
       F::kUniform4, F::kUniform4},
      {"lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo3", false, false, 8, 1, 4096, 32, 8, 128, 16,
       F::kUniform3, F::kUniform4},
      {"lowbit.decode.bf16.d128.b8.kv4096.h32x8.prod3", false, false, 8, 1, 4096, 32, 8, 128, 16,
       F::kTurboProd3, F::kUniform4},
      {"lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo4", true, false, 1, 512, 4096, 32, 8, 128, 16,
       F::kUniform4, F::kUniform4},
      {"lowbit.prefill.bf16.d128.t512.kv4096.h32x8.prod3", true, false, 1, 512, 4096, 32, 8, 128, 16,
       F::kTurboProd3, F::kUniform4},
  };
  return b;
}

const LbBench* find_lb(const std::string& id) {
  for (const LbBench& b : lb_benches())
    if (id == b.id) return &b;
  return nullptr;
}

void add_lowbit_cases(std::vector<Case>& cases) {
  for (const LbBench& b : lb_benches()) {
    lbt::LowBitCodec codec;
    lbt::LowBitTables tb;
    if (lbt::lowbit_codec(b.kf, b.vf, b.D, 7, &codec) != kops::Status::kSuccess ||
        lbt::lowbit_resolve(codec, &tb) != kops::Status::kSuccess) {
      continue;
    }
    const int64_t tokens = static_cast<int64_t>(b.batch) * b.kv;
    const int64_t rec = tokens * b.Hkv * (tb.k.entry_bytes + tb.v.entry_bytes);
    const int64_t qo = static_cast<int64_t>(b.batch) * b.q * b.Hq * b.D * 2 * 2;
    int64_t bytes = rec + qo, flops = 0;
    if (b.store) {
      bytes = tokens * b.Hkv * b.D * 2 * 2 + rec;
      cases.push_back({b.id, "lowbit_store", kops::DType::kBFloat16, tokens, b.D, bytes, 0, 3, 20});
      continue;
    }
    if (b.prefill) {
      int64_t pairs = static_cast<int64_t>(b.batch) * b.q * (2 * b.kv - b.q + 1) / 2;
      flops = 4 * pairs * b.D * b.Hq;
      bytes = 0;
    }
    cases.push_back({b.id, b.prefill ? "lowbit_prefill" : "lowbit_decode", kops::DType::kBFloat16,
                     b.batch, b.D, bytes, flops, 3, b.prefill ? 40 : 100});
  }
}

bool run_lowbit(const Case& c, const kops::LaunchContext& ctx_in, Stats* st) {
  const LbBench* b = find_lb(c.id);
  if (b == nullptr) return false;
  const int B = b->batch, D = b->D, ps = b->page;
  lbt::LowBitCodec codec;
  lbt::LowBitTables tb;
  if (lbt::lowbit_codec(b->kf, b->vf, D, 7, &codec) != kops::Status::kSuccess ||
      lbt::lowbit_resolve(codec, &tb) != kops::Status::kSuccess) {
    return false;
  }
  const int64_t tokens = static_cast<int64_t>(B) * b->kv;
  int total_pages = 0;
  std::vector<int32_t> block_table(static_cast<size_t>(B), 0), seq_lens(B), page_of(B, 0);
  int max_pages = 1;
  for (int r = 0; r < B; ++r) {
    page_of[r] = total_pages;
    const int np = (b->kv + ps - 1) / ps;
    total_pages += np;
    max_pages = std::max(max_pages, np);
    seq_lens[r] = b->kv;
  }
  std::vector<int64_t> slots(tokens);
  for (int r = 0; r < B; ++r) {
    for (int t = 0; t < b->kv; ++t) {
      slots[static_cast<size_t>(r) * b->kv + t] = static_cast<int64_t>(page_of[r] + t / ps) * ps + t % ps;
    }
  }
  std::vector<int32_t> bt(static_cast<size_t>(B) * max_pages, 0);
  for (int r = 0; r < B; ++r) {
    const int np = (b->kv + ps - 1) / ps;
    for (int i = 0; i < np; ++i) bt[static_cast<size_t>(r) * max_pages + i] = page_of[r] + i;
  }

  const int64_t kbytes = static_cast<int64_t>(total_pages) * ps * b->Hkv * tb.k.entry_bytes;
  const int64_t vbytes = static_cast<int64_t>(total_pages) * ps * b->Hkv * tb.v.entry_bytes;
  __nv_bfloat16* q = nullptr;
  __nv_bfloat16* o = nullptr;
  __nv_bfloat16* key = nullptr;
  __nv_bfloat16* val = nullptr;
  uint8_t* kc = nullptr;
  uint8_t* vc = nullptr;
  int64_t* smap = nullptr;
  int32_t* dbt = nullptr;
  int32_t* dseq = nullptr;
  int32_t* dqo = nullptr;
  float* lse = nullptr;
  void* pers = nullptr;
  void* scr = nullptr;
  void* alibi = nullptr;
  bool ok = true;
  auto alloc = [&](auto** p, size_t n) {
    ok = ok && cuda_ok(cudaMalloc(reinterpret_cast<void**>(p), std::max<size_t>(n, 1)), c.id.c_str());
  };
  alloc(&q, static_cast<size_t>(B) * b->q * b->Hq * D * 2);
  alloc(&o, static_cast<size_t>(B) * b->q * b->Hq * D * 2);
  alloc(&key, static_cast<size_t>(tokens) * b->Hkv * D * 2);
  alloc(&val, static_cast<size_t>(tokens) * b->Hkv * D * 2);
  alloc(&kc, static_cast<size_t>(kbytes));
  alloc(&vc, static_cast<size_t>(vbytes));
  alloc(&smap, static_cast<size_t>(tokens) * 8);
  alloc(&dbt, bt.size() * 4);
  alloc(&dseq, seq_lens.size() * 4);
  alloc(&dqo, static_cast<size_t>(B + 1) * 4);
  alloc(&lse, static_cast<size_t>(B) * b->q * b->Hq * 4);
  if (!ok) return false;
  std::vector<int32_t> qo(B + 1, 0), kvl(B, b->kv);
  for (int r = 0; r < B; ++r) qo[r + 1] = qo[r] + b->q;
  cudaMemcpy(smap, slots.data(), slots.size() * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dseq, seq_lens.data(), seq_lens.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dqo, qo.data(), qo.size() * 4, cudaMemcpyHostToDevice);
  bench_fill_bf16<<<32, 256>>>(q, static_cast<int64_t>(B) * b->q * b->Hq * D, 7);
  bench_fill_bf16<<<64, 256>>>(key, static_cast<int64_t>(tokens) * b->Hkv * D, 11);
  bench_fill_bf16<<<64, 256>>>(val, static_cast<int64_t>(tokens) * b->Hkv * D, 13);

  kops::kv::PagedKVDesc kd{};
  kd.k_data = kc;
  kd.v_data = vc;
  kd.k_dtype = kd.v_dtype = kops::DType::kUInt8;
  kd.k_codec = kd.v_codec = kops::kv::KVCodec::kLowBitWHT;
  kd.layout = kops::kv::KVLayout::kNHD;
  kd.page_size = ps;
  kd.num_kv_heads = b->Hkv;
  kd.head_dim_k = kd.head_dim_v = D;
  kd.entry_bytes = tb.k.entry_bytes;
  kd.stride_unit = kops::StrideUnit::kByte;
  kd.k_stride_page = static_cast<int64_t>(ps) * b->Hkv * tb.k.entry_bytes;
  kd.k_stride_token = static_cast<int64_t>(b->Hkv) * tb.k.entry_bytes;
  kd.k_stride_head = tb.k.entry_bytes;
  kd.v_stride_page = static_cast<int64_t>(ps) * b->Hkv * tb.v.entry_bytes;
  kd.v_stride_token = static_cast<int64_t>(b->Hkv) * tb.v.entry_bytes;
  kd.v_stride_head = tb.v.entry_bytes;
  kd.table_format = kops::kv::PageTableFormat::kBlockTable;
  kd.block_table = dbt;
  kd.block_table_stride = max_pages;
  kd.seq_lens = dseq;

  const kops::Tensor tq = tensor3d(q, kops::DType::kBFloat16, static_cast<int64_t>(B) * b->q, b->Hq, D);
  const kops::Tensor to = tensor3d(o, kops::DType::kBFloat16, static_cast<int64_t>(B) * b->q, b->Hq, D);
  const kops::Tensor tk = tensor3d(key, kops::DType::kBFloat16, tokens, b->Hkv, D);
  const kops::Tensor tv = tensor3d(val, kops::DType::kBFloat16, tokens, b->Hkv, D);
  kops::LaunchContext ctx = ctx_in;
  if (b->store) {
    const kops::Tensor ts = tensor1d(smap, kops::DType::kInt64, tokens);
    ok = status_ok(lbt::lowbit_kv_store(tk, tv, kd, ts, codec, ctx), c.id.c_str());
    if (ok) ok = measure(c, [&] { (void)lbt::lowbit_kv_store(tk, tv, kd, ts, codec, ctx); }, st);
    for (void* p : {static_cast<void*>(q), static_cast<void*>(o), static_cast<void*>(key),
                    static_cast<void*>(val), static_cast<void*>(kc), static_cast<void*>(vc),
                    static_cast<void*>(smap), static_cast<void*>(dbt), static_cast<void*>(dseq),
                    static_cast<void*>(dqo), static_cast<void*>(lse)}) {
      cudaFree(p);
    }
    return ok;
  }
  // Fill the cache through the public store entry once (setup, not timed).
  {
    const kops::Tensor ts = tensor1d(smap, kops::DType::kInt64, tokens);
    ok = status_ok(lbt::lowbit_kv_store(tk, tv, kd, ts, codec, ctx), c.id.c_str());
    if (!ok) return false;
  }
  kops::attn::AttentionSpec spec;
  spec.num_q_heads = b->Hq;
  spec.num_kv_heads = b->Hkv;
  spec.head_dim_qk = spec.head_dim_vo = D;
  spec.q_dtype = spec.o_dtype = kops::DType::kBFloat16;
  spec.kv_dtype = kops::DType::kUInt8;
  spec.k_codec = spec.v_codec = kops::kv::KVCodec::kLowBitWHT;
  spec.kv_layout = kd.layout;
  spec.page_size = ps;
  spec.mask = kops::attn::MaskMode::kCausal;
  spec.return_lse = true;
  kops::attn::AttentionPlan plan;
  kops::Workspace ws{};
  auto do_plan = [&]() {
    if (b->prefill) {
      std::vector<int32_t> qoh = qo, kvh = kvl;
      kops::Tensor qot = tensor1d(qoh.data(), kops::DType::kInt32, B + 1);
      qot.device = -1;
      kops::Tensor kvt = tensor1d(kvh.data(), kops::DType::kInt32, B);
      kvt.device = -1;
      return lbt::lowbit_prefill_plan(spec, codec, qot, kvt, ctx, ws, &plan);
    }
    return lbt::lowbit_decode_plan(spec, codec, B, b->kv, b->q, ctx, ws, &plan);
  };
  if (do_plan() != kops::Status::kWorkspaceTooSmall) {
    std::fprintf(stderr, "[bench] %s: workspace probe failed\n", c.id.c_str());
    return false;
  }
  alloc(&pers, plan.workspace.device_persistent_bytes);
  alloc(&scr, std::max<size_t>(plan.workspace.device_scratch_bytes, 1));
  ws.device_persistent = pers;
  ws.device_persistent_bytes = plan.workspace.device_persistent_bytes;
  ws.device_scratch = scr;
  ws.device_scratch_bytes = plan.workspace.device_scratch_bytes;
  ok = ok && status_ok(do_plan(), c.id.c_str());
  if (!ok) return false;
  kops::attn::AttentionModifiers mod;
  mod.sm_scale = 0.0f;
  kops::attn::PagedPrefillParams pp{};
  kops::attn::DecodeParams dp{};
  auto run = [&]() {
    if (b->prefill) {
      pp.q = q;
      pp.q_stride_n = static_cast<int64_t>(b->Hq) * D;
      pp.q_stride_h = D;
      pp.qo_indptr = dqo;
      pp.kv = kd;
      pp.mod = mod;
      pp.o = o;
      pp.o_stride_n = static_cast<int64_t>(b->Hq) * D;
      pp.o_stride_h = D;
      pp.lse = lse;
      pp.batch_size = B;
      pp.num_qo_heads = b->Hq;
      pp.num_kv_heads = b->Hkv;
      pp.total_q = qo[B];
      return lbt::lowbit_prefill_run(plan, pp, codec, ws, ctx);
    }
    dp.q = q;
    dp.q_stride_n = static_cast<int64_t>(b->Hq) * D;
    dp.q_stride_h = D;
    dp.q_len = b->q;
    dp.kv = kd;
    dp.mod = mod;
    dp.o = o;
    dp.o_stride_n = static_cast<int64_t>(b->Hq) * D;
    dp.o_stride_h = D;
    dp.lse = lse;
    dp.batch_size = B;
    dp.num_qo_heads = b->Hq;
    dp.num_kv_heads = b->Hkv;
    return lbt::lowbit_decode_run(plan, dp, codec, ws, ctx);
  };
  ok = status_ok(run(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)run(); }, st);
  for (void* p : {static_cast<void*>(q), static_cast<void*>(o), static_cast<void*>(key),
                  static_cast<void*>(val), static_cast<void*>(kc), static_cast<void*>(vc),
                  static_cast<void*>(smap), static_cast<void*>(dbt), static_cast<void*>(dseq),
                  static_cast<void*>(dqo), static_cast<void*>(lse), pers, scr}) {
    cudaFree(p);
  }
  return ok;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "attention") return run_attention(c, ctx, st);
  if (c.op == "merge_state") return run_merge(c, ctx, st);
  if (c.op == "merge_states") return run_states(c, ctx, st);
  if (c.op == "merge_split_kv") return run_split(c, ctx, st);
  if (c.op == "concat_mla_q") return run_concat_mla_q(c, ctx, st);
  if (c.op == "mla_context_gather") return run_mla_context_gather(c, ctx, st);
  if (c.op == "lowbit_store" || c.op == "lowbit_decode" || c.op == "lowbit_prefill")
    return run_lowbit(c, ctx, st);
  return false;
}

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
