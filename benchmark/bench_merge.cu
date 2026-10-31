// =============================================================================
// kernel-ops | benchmark/bench_merge.cu
// merge_state (split-KV combine) cases. The case table is the shared spec read
// by benchmark/compare/torch_baseline.py through --dump-cases.
// =============================================================================
#include "benchmark/harness.h"

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

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "merge_state") return run_merge(c, ctx, st);
  if (c.op == "merge_states") return run_states(c, ctx, st);
  if (c.op == "merge_split_kv") return run_split(c, ctx, st);
  if (c.op == "concat_mla_q") return run_concat_mla_q(c, ctx, st);
  if (c.op == "mla_context_gather") return run_mla_context_gather(c, ctx, st);
  return false;
}

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
