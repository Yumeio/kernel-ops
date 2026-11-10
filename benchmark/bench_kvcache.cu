// =============================================================================
// kernel-ops | benchmark/bench_kvcache.cu
// B2 kvcache cases: reshape_and_cache, append_paged_kv, ring_kv_append,
// concat_and_cache_mla, gather/scatter pages, cp_gather_cache,
// gather_and_maybe_dequant_cache, copy/zero pages, convert_cache,
// kv_relayout_pages. Case ids are the shared spec read by --dump-cases.
// =============================================================================
#include "benchmark/harness.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/kvcache/kvcache_kernels.cuh"
#include "csrc/kvcache/ops.h"

namespace {

using namespace kops_bench;
using kops::DType;
using kops::kv::KVCodec;
using kops::kv::KVLayout;
using kops::kv::PageTableFormat;
using kops::Tensor;
using kops::kv::PagedKVDesc;

kops::kv::PagedKVDesc desc_nhd(void* k, void* v, int heads, int dk, int dv, int page) {
  kops::kv::PagedKVDesc d{};
  d.k_data = k;
  d.v_data = v;
  d.k_dtype = DType::kBFloat16;
  d.v_dtype = DType::kBFloat16;
  d.k_codec = KVCodec::kNative;
  d.v_codec = KVCodec::kNative;
  d.layout = KVLayout::kNHD;
  d.page_size = page;
  d.num_kv_heads = heads;
  d.head_dim_k = dk;
  d.head_dim_v = dv;
  d.stride_unit = kops::StrideUnit::kElement;
  d.k_stride_page = static_cast<int64_t>(page) * heads * dk;
  d.k_stride_token = static_cast<int64_t>(heads) * dk;
  d.k_stride_head = dk;
  d.v_stride_page = static_cast<int64_t>(page) * heads * dv;
  d.v_stride_token = static_cast<int64_t>(heads) * dv;
  d.v_stride_head = dv;
  d.table_format = PageTableFormat::kSlotMapping;
  return d;
}

Tensor t1(void* p, int64_t n) { return tensor1d(p, DType::kInt64, n); }
Tensor t3b(void* p, int64_t a, int64_t b, int64_t c) {
  return tensor3d(p, DType::kBFloat16, a, b, c);
}
Tensor t2b(void* p, int64_t r, int64_t c) { return tensor2d(p, DType::kBFloat16, r, c); }

bool run_reshape(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kT = 2048, kH = 4, kD = 128, kPage = 16;
  const int pages = kT / kPage + 1;
  void* key = nullptr;
  void* val = nullptr;
  void* kc = nullptr;
  void* vc = nullptr;
  int64_t* slots = nullptr;
  if (!cuda_ok(cudaMalloc(&key, static_cast<size_t>(kT * kH * kD) * 2), "key") ||
      !cuda_ok(cudaMalloc(&val, static_cast<size_t>(kT * kH * kD) * 2), "val") ||
      !cuda_ok(cudaMalloc(&kc, static_cast<size_t>(pages * kPage * kH * kD) * 2), "kc") ||
      !cuda_ok(cudaMalloc(&vc, static_cast<size_t>(pages * kPage * kH * kD) * 2), "vc") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&slots), static_cast<size_t>(kT) * 8), "slots")) {
    return false;
  }
  std::vector<int64_t> sh(static_cast<size_t>(kT));
  for (int i = 0; i < kT; ++i) sh[static_cast<size_t>(i)] = i;
  cudaMemcpy(slots, sh.data(), sh.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(key, 1, static_cast<size_t>(kT * kH * kD) * 2);
  cudaMemset(val, 1, static_cast<size_t>(kT * kH * kD) * 2);
  PagedKVDesc d = desc_nhd(kc, vc, kH, kD, kD, kPage);
  const Tensor tk = t3b(key, kT, kH, kD);
  const Tensor tv = t3b(val, kT, kH, kD);
  const Tensor ts = t1(slots, kT);
  bool ok = status_ok(kops::kv::reshape_and_cache(tk, tv, d, ts, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::reshape_and_cache(tk, tv, d, ts, ctx); }, st);
  cudaFree(key);
  cudaFree(val);
  cudaFree(kc);
  cudaFree(vc);
  cudaFree(slots);
  return ok;
}

bool run_reshape_legacy(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kT = 2048, kH = 4, kD = 128, kPage = 16;
  const int pages = kT / kPage + 1;
  void* key = nullptr;
  void* val = nullptr;
  void* kc = nullptr;
  void* vc = nullptr;
  int64_t* slots = nullptr;
  if (!cuda_ok(cudaMalloc(&key, static_cast<size_t>(kT * kH * kD) * 2), "key") ||
      !cuda_ok(cudaMalloc(&val, static_cast<size_t>(kT * kH * kD) * 2), "val") ||
      !cuda_ok(cudaMalloc(&kc, static_cast<size_t>(pages * kH * kD * kPage) * 2), "kc") ||
      !cuda_ok(cudaMalloc(&vc, static_cast<size_t>(pages * kH * kD * kPage) * 2), "vc") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&slots), static_cast<size_t>(kT) * 8), "slots")) {
    return false;
  }
  std::vector<int64_t> sh(static_cast<size_t>(kT));
  for (int i = 0; i < kT; ++i) sh[static_cast<size_t>(i)] = i;
  cudaMemcpy(slots, sh.data(), sh.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(key, 1, static_cast<size_t>(kT * kH * kD) * 2);
  cudaMemset(val, 1, static_cast<size_t>(kT * kH * kD) * 2);
  PagedKVDesc d = desc_nhd(kc, vc, kH, kD, kD, kPage);
  d.layout = KVLayout::kVllmLegacy;
  d.legacy_x = 8;
  const Tensor tk = t3b(key, kT, kH, kD);
  const Tensor tv = t3b(val, kT, kH, kD);
  const Tensor ts = t1(slots, kT);
  bool ok = status_ok(kops::kv::reshape_and_cache(tk, tv, d, ts, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::reshape_and_cache(tk, tv, d, ts, ctx); }, st);
  cudaFree(key);
  cudaFree(val);
  cudaFree(kc);
  cudaFree(vc);
  cudaFree(slots);
  return ok;
}

bool run_mla_grouped(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kLayers = 2, kT = 1024, kRank = 512, kPe = 64, kPage = 16;
  const int pages = kT / kPage + 1;
  void* ckv = nullptr;
  void* kpe = nullptr;
  void* caches = nullptr;
  int64_t* ptrs = nullptr;
  int64_t* slots = nullptr;
  if (!cuda_ok(cudaMalloc(&ckv, static_cast<size_t>(kLayers * kT * kRank) * 2), "ckv") ||
      !cuda_ok(cudaMalloc(&kpe, static_cast<size_t>(kLayers * kT * kPe) * 2), "kpe") ||
      !cuda_ok(cudaMalloc(&caches, static_cast<size_t>(kLayers * pages * kPage * (kRank + kPe)) * 2),
               "caches") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ptrs), kLayers * 8), "ptrs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&slots), static_cast<size_t>(kLayers * kT) * 8),
               "slots")) {
    return false;
  }
  std::vector<int64_t> sh(static_cast<size_t>(kLayers * kT));
  for (int i = 0; i < kLayers * kT; ++i) sh[static_cast<size_t>(i)] = i;
  std::vector<int64_t> pv(static_cast<size_t>(kLayers));
  for (int l = 0; l < kLayers; ++l) {
    pv[static_cast<size_t>(l)] = reinterpret_cast<int64_t>(
        static_cast<uint8_t*>(caches) + static_cast<int64_t>(l) * pages * kPage * (kRank + kPe) * 2);
  }
  cudaMemcpy(slots, sh.data(), sh.size() * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(ptrs, pv.data(), pv.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(ckv, 1, static_cast<size_t>(kLayers * kT * kRank) * 2);
  cudaMemset(kpe, 1, static_cast<size_t>(kLayers * kT * kPe) * 2);
  PagedKVDesc d = desc_nhd(caches, nullptr, 1, kRank + kPe, kRank, kPage);
  d.layout = KVLayout::kLatent;
  d.k_stride_page = static_cast<int64_t>(kPage) * (kRank + kPe);
  d.k_stride_token = kRank + kPe;
  d.k_stride_head = kRank + kPe;
  const Tensor tc = tensor3d(ckv, DType::kBFloat16, kLayers, kT, kRank);
  const Tensor tp = tensor3d(kpe, DType::kBFloat16, kLayers, kT, kPe);
  const Tensor tptr = t1(ptrs, kLayers);
  const Tensor ts = tensor2d(slots, DType::kInt64, kLayers, kT);
  bool ok = status_ok(kops::kv::concat_and_cache_mla_grouped(tc, tp, tptr, Tensor{}, ts, d, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] { (void)kops::kv::concat_and_cache_mla_grouped(tc, tp, tptr, Tensor{}, ts, d, ctx); },
        st);
  }
  cudaFree(ckv);
  cudaFree(kpe);
  cudaFree(caches);
  cudaFree(ptrs);
  cudaFree(slots);
  return ok;
}

bool run_append(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kT = 2048, kH = 4, kD = 128, kPage = 16;
  const int pages = kT / kPage + 1;
  void* key = nullptr;
  void* val = nullptr;
  void* kc = nullptr;
  void* vc = nullptr;
  int32_t* indptr = nullptr;
  int32_t* page_ids = nullptr;
  int32_t* batch = nullptr;
  int32_t* pos = nullptr;
  if (!cuda_ok(cudaMalloc(&key, static_cast<size_t>(kT * kH * kD) * 2), "key") ||
      !cuda_ok(cudaMalloc(&val, static_cast<size_t>(kT * kH * kD) * 2), "val") ||
      !cuda_ok(cudaMalloc(&kc, static_cast<size_t>(pages * kPage * kH * kD) * 2), "kc") ||
      !cuda_ok(cudaMalloc(&vc, static_cast<size_t>(pages * kPage * kH * kD) * 2), "vc") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&indptr), 2 * 4), "indptr") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&page_ids), static_cast<size_t>(pages) * 4), "pages") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&batch), static_cast<size_t>(kT) * 4), "batch") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pos), static_cast<size_t>(kT) * 4), "pos")) {
    return false;
  }
  std::vector<int32_t> ip = {0, pages};
  std::vector<int32_t> pi(static_cast<size_t>(pages));
  std::vector<int32_t> bi(static_cast<size_t>(kT), 0);
  std::vector<int32_t> po(static_cast<size_t>(kT));
  for (int i = 0; i < pages; ++i) pi[static_cast<size_t>(i)] = i;
  for (int i = 0; i < kT; ++i) po[static_cast<size_t>(i)] = i;
  cudaMemcpy(indptr, ip.data(), ip.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(page_ids, pi.data(), pi.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(batch, bi.data(), bi.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(pos, po.data(), po.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(key, 1, static_cast<size_t>(kT * kH * kD) * 2);
  cudaMemset(val, 1, static_cast<size_t>(kT * kH * kD) * 2);
  PagedKVDesc d = desc_nhd(kc, vc, kH, kD, kD, kPage);
  d.table_format = PageTableFormat::kCSR;
  d.kv_indptr = indptr;
  d.kv_page_indices = page_ids;
  const Tensor tk = t3b(key, kT, kH, kD);
  const Tensor tv = t3b(val, kT, kH, kD);
  const Tensor tb = tensor1d(batch, DType::kInt32, kT);
  const Tensor tp = tensor1d(pos, DType::kInt32, kT);
  bool ok = status_ok(kops::kv::append_paged_kv(tk, tv, d, tb, tp, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::append_paged_kv(tk, tv, d, tb, tp, ctx); }, st);
  cudaFree(key);
  cudaFree(val);
  cudaFree(kc);
  cudaFree(vc);
  cudaFree(indptr);
  cudaFree(page_ids);
  cudaFree(batch);
  cudaFree(pos);
  return ok;
}

bool run_ring(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kT = 1024, kH = 4, kD = 128, kWindow = 256, kSlots = 2;
  void* key = nullptr;
  void* val = nullptr;
  void* rk = nullptr;
  void* rv = nullptr;
  int32_t* slot = nullptr;
  int64_t* pos = nullptr;
  if (!cuda_ok(cudaMalloc(&key, static_cast<size_t>(kT * kH * kD) * 2), "key") ||
      !cuda_ok(cudaMalloc(&val, static_cast<size_t>(kT * kH * kD) * 2), "val") ||
      !cuda_ok(cudaMalloc(&rk, static_cast<size_t>(kSlots * kWindow * kH * kD) * 2), "rk") ||
      !cuda_ok(cudaMalloc(&rv, static_cast<size_t>(kSlots * kWindow * kH * kD) * 2), "rv") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&slot), static_cast<size_t>(kT) * 4), "slot") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pos), static_cast<size_t>(kT) * 8), "pos")) {
    return false;
  }
  std::vector<int32_t> si(static_cast<size_t>(kT));
  std::vector<int64_t> pi(static_cast<size_t>(kT));
  for (int i = 0; i < kT; ++i) {
    si[static_cast<size_t>(i)] = 0;
    pi[static_cast<size_t>(i)] = i;
  }
  cudaMemcpy(slot, si.data(), si.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(pos, pi.data(), pi.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(key, 1, static_cast<size_t>(kT * kH * kD) * 2);
  cudaMemset(val, 1, static_cast<size_t>(kT * kH * kD) * 2);
  const Tensor tk = t3b(key, kT, kH, kD);
  const Tensor tv = t3b(val, kT, kH, kD);
  const Tensor trk = tensor4d(rk, DType::kBFloat16, kSlots, kWindow, kH, kD);
  const Tensor trv = tensor4d(rv, DType::kBFloat16, kSlots, kWindow, kH, kD);
  const Tensor ts = tensor1d(slot, DType::kInt32, kT);
  const Tensor tp = t1(pos, kT);
  bool ok = status_ok(kops::kv::ring_kv_append(tk, tv, trk, trv, ts, tp, Tensor{}, Tensor{}, kWindow, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(
        c, [&] {
          (void)kops::kv::ring_kv_append(tk, tv, trk, trv, ts, tp, Tensor{}, Tensor{}, kWindow, ctx);
        },
        st);
  }
  cudaFree(key);
  cudaFree(val);
  cudaFree(rk);
  cudaFree(rv);
  cudaFree(slot);
  cudaFree(pos);
  return ok;
}

bool run_concat_mla(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kT = 2048, kRank = 512, kPe = 64, kPage = 16;
  const int pages = kT / kPage + 1;
  void* ckv = nullptr;
  void* kpe = nullptr;
  void* cache = nullptr;
  int64_t* slots = nullptr;
  if (!cuda_ok(cudaMalloc(&ckv, static_cast<size_t>(kT * kRank) * 2), "ckv") ||
      !cuda_ok(cudaMalloc(&kpe, static_cast<size_t>(kT * kPe) * 2), "kpe") ||
      !cuda_ok(cudaMalloc(&cache, static_cast<size_t>(pages * kPage * (kRank + kPe)) * 2), "cache") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&slots), static_cast<size_t>(kT) * 8), "slots")) {
    return false;
  }
  std::vector<int64_t> sh(static_cast<size_t>(kT));
  for (int i = 0; i < kT; ++i) sh[static_cast<size_t>(i)] = i;
  cudaMemcpy(slots, sh.data(), sh.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(ckv, 1, static_cast<size_t>(kT * kRank) * 2);
  cudaMemset(kpe, 1, static_cast<size_t>(kT * kPe) * 2);
  PagedKVDesc d = desc_nhd(cache, nullptr, 1, kRank + kPe, kRank, kPage);
  d.layout = KVLayout::kLatent;
  d.k_stride_page = static_cast<int64_t>(kPage) * (kRank + kPe);
  d.k_stride_token = kRank + kPe;
  d.k_stride_head = kRank + kPe;
  const Tensor tc = t2b(ckv, kT, kRank);
  const Tensor tp = t2b(kpe, kT, kPe);
  const Tensor ts = t1(slots, kT);
  bool ok = status_ok(kops::kv::concat_and_cache_mla(tc, tp, d, ts, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::concat_and_cache_mla(tc, tp, d, ts, ctx); }, st);
  cudaFree(ckv);
  cudaFree(kpe);
  cudaFree(cache);
  cudaFree(slots);
  return ok;
}

bool run_gather(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kT = 4096, kH = 4, kD = 128, kPage = 16;
  const int pages = (kT + kPage - 1) / kPage;
  void* cache = nullptr;
  void* out = nullptr;
  int64_t* idx = nullptr;
  if (!cuda_ok(cudaMalloc(&cache, static_cast<size_t>(pages * kPage * kH * kD) * 2), "cache") ||
      !cuda_ok(cudaMalloc(&out, static_cast<size_t>(kT * kH * kD) * 2), "out") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&idx), static_cast<size_t>(kT) * 8), "idx")) {
    return false;
  }
  std::vector<int64_t> iv(static_cast<size_t>(kT));
  for (int i = 0; i < kT; ++i) iv[static_cast<size_t>(i)] = i;
  cudaMemcpy(idx, iv.data(), iv.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(cache, 1, static_cast<size_t>(pages * kPage * kH * kD) * 2);
  PagedKVDesc d = desc_nhd(cache, nullptr, kH, kD, kD, kPage);
  const Tensor to = t2b(out, kT, kH * kD);
  const Tensor ti = t1(idx, kT);
  bool ok = status_ok(kops::kv::gather_pages(to, d, ti, true, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::gather_pages(to, d, ti, true, ctx); }, st);
  cudaFree(cache);
  cudaFree(out);
  cudaFree(idx);
  return ok;
}

bool run_scatter(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kT = 4096, kH = 4, kD = 128, kPage = 16;
  const int pages = (kT + kPage - 1) / kPage;
  void* cache = nullptr;
  void* src = nullptr;
  int64_t* idx = nullptr;
  if (!cuda_ok(cudaMalloc(&cache, static_cast<size_t>(pages * kPage * kH * kD) * 2), "cache") ||
      !cuda_ok(cudaMalloc(&src, static_cast<size_t>(kT * kH * kD) * 2), "src") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&idx), static_cast<size_t>(kT) * 8), "idx")) {
    return false;
  }
  std::vector<int64_t> iv(static_cast<size_t>(kT));
  for (int i = 0; i < kT; ++i) iv[static_cast<size_t>(i)] = i;
  cudaMemcpy(idx, iv.data(), iv.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(src, 1, static_cast<size_t>(kT * kH * kD) * 2);
  PagedKVDesc d = desc_nhd(cache, nullptr, kH, kD, kD, kPage);
  const Tensor ts = t2b(src, kT, kH * kD);
  const Tensor ti = t1(idx, kT);
  bool ok = status_ok(kops::kv::scatter_pages(d, ts, ti, true, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::scatter_pages(d, ts, ti, true, ctx); }, st);
  cudaFree(cache);
  cudaFree(src);
  cudaFree(idx);
  return ok;
}

bool run_cp_gather(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kT = 4096, kBlock = 16, kBytes = 576;
  const int pages = (kT + kBlock - 1) / kBlock;
  const int batch = 8;
  void* cache = nullptr;
  void* dst = nullptr;
  int32_t* table = nullptr;
  int32_t* cu = nullptr;
  if (!cuda_ok(cudaMalloc(&cache, static_cast<size_t>(pages * kBlock * kBytes)), "cache") ||
      !cuda_ok(cudaMalloc(&dst, static_cast<size_t>(kT * kBytes)), "dst") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&table), static_cast<size_t>(batch) * pages * 4),
               "table") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cu), static_cast<size_t>(batch + 1) * 4), "cu")) {
    return false;
  }
  std::vector<int32_t> tab(static_cast<size_t>(batch) * pages);
  for (size_t i = 0; i < tab.size(); ++i) {
    tab[i] = static_cast<int32_t>(i % static_cast<size_t>(pages));
  }
  std::vector<int32_t> cv(static_cast<size_t>(batch + 1));
  const int per = kT / batch;
  for (int i = 0; i <= batch; ++i) cv[static_cast<size_t>(i)] = i * per;
  cudaMemcpy(table, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(cu, cv.data(), cv.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(cache, 1, static_cast<size_t>(pages * kBlock * kBytes));
  const Tensor td = tensor2d(dst, DType::kUInt8, kT, kBytes);
  const Tensor tc = tensor3d(cache, DType::kUInt8, pages, kBlock, kBytes);
  kops::kv::GatherSpec spec{};
  spec.block_table = table;
  spec.block_table_stride = pages;
  spec.cu_seq_lens = cu;
  spec.num_reqs = batch;
  spec.num_tokens = kT;
  bool ok = status_ok(kops::kv::cp_gather_cache(td, tc, spec, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::cp_gather_cache(td, tc, spec, ctx); }, st);
  cudaFree(cache);
  cudaFree(dst);
  cudaFree(table);
  cudaFree(cu);
  return ok;
}

bool run_copy_zero(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kLayers = 2, kCaches = 2, kPages = 512, kPageBytes = 4096;
  void* caches = nullptr;
  int64_t* ptrs = nullptr;
  int64_t* pairs = nullptr;
  int64_t* pages = nullptr;
  if (!cuda_ok(cudaMalloc(&caches, static_cast<size_t>(kLayers * kCaches * kPages * kPageBytes)),
               "caches") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ptrs), kLayers * kCaches * 8), "ptrs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pairs), static_cast<size_t>(kPages) * 16), "pairs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pages), static_cast<size_t>(kPages) * 8), "pages")) {
    return false;
  }
  std::vector<int64_t> pv(static_cast<size_t>(kLayers * kCaches));
  for (int i = 0; i < kLayers * kCaches; ++i) {
    pv[static_cast<size_t>(i)] =
        reinterpret_cast<int64_t>(static_cast<uint8_t*>(caches) + static_cast<int64_t>(i) * kPages * kPageBytes);
  }
  std::vector<int64_t> qv(static_cast<size_t>(kPages) * 2);
  std::vector<int64_t> zv(static_cast<size_t>(kPages));
  for (int i = 0; i < kPages; ++i) {
    qv[static_cast<size_t>(i * 2)] = i;
    qv[static_cast<size_t>(i * 2 + 1)] = (i + 1) % kPages;
    zv[static_cast<size_t>(i)] = i;
  }
  cudaMemcpy(ptrs, pv.data(), pv.size() * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(pairs, qv.data(), qv.size() * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(pages, zv.data(), zv.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(caches, 1, static_cast<size_t>(kLayers * kCaches * kPages * kPageBytes));
  const Tensor tp = t1(ptrs, kLayers * kCaches);
  const Tensor tq = tensor2d(pairs, DType::kInt64, kPages, 2);
  const Tensor tz = t1(pages, kPages);
  bool copy = c.op == "copy_pages";
  bool ok = copy ? status_ok(kops::kv::copy_pages(tp, tq, kPageBytes, kCaches, ctx), c.id.c_str())
                 : status_ok(kops::kv::zero_pages(tp, tz, kPageBytes, kCaches, ctx), c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] {
          if (copy) {
            (void)kops::kv::copy_pages(tp, tq, kPageBytes, kCaches, ctx);
          } else {
            (void)kops::kv::zero_pages(tp, tz, kPageBytes, kCaches, ctx);
          }
        },
        st);
  }
  cudaFree(caches);
  cudaFree(ptrs);
  cudaFree(pairs);
  cudaFree(pages);
  return ok;
}

bool run_convert(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kBlocks = 256, kStride = 4096;
  void* src = nullptr;
  void* dst = nullptr;
  if (!cuda_ok(cudaMalloc(&src, static_cast<size_t>(kBlocks * kStride) * 2), "src") ||
      !cuda_ok(cudaMalloc(&dst, static_cast<size_t>(kBlocks * kStride) * 4), "dst")) {
    return false;
  }
  cudaMemset(src, 1, static_cast<size_t>(kBlocks * kStride) * 2);
  const Tensor ts = tensor2d(src, DType::kBFloat16, kBlocks, kStride);
  const Tensor td = tensor2d(dst, DType::kFloat32, kBlocks, kStride);
  bool ok = status_ok(kops::kv::convert_cache(td, ts, 2.0f, KVCodec::kNative, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::convert_cache(td, ts, 2.0f, KVCodec::kNative, ctx); }, st);
  cudaFree(src);
  cudaFree(dst);
  return ok;
}

bool run_gather_dequant(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kTokens = 4096, kBlock = 16, kEntry = 576;
  const int pages = (kTokens + kBlock - 1) / kBlock;
  void* cache = nullptr;
  void* dst = nullptr;
  int32_t* table = nullptr;
  int32_t* cu = nullptr;
  if (!cuda_ok(cudaMalloc(&cache, static_cast<size_t>(pages * kBlock * kEntry) * 2), "cache") ||
      !cuda_ok(cudaMalloc(&dst, static_cast<size_t>(kTokens * kEntry) * 2), "dst") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&table), static_cast<size_t>(pages) * 4), "table") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cu), 2 * 4), "cu")) {
    return false;
  }
  std::vector<int32_t> tab(static_cast<size_t>(pages));
  for (int i = 0; i < pages; ++i) tab[static_cast<size_t>(i)] = i;
  std::vector<int32_t> cv = {0, kTokens};
  cudaMemcpy(table, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(cu, cv.data(), cv.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(cache, 1, static_cast<size_t>(pages * kBlock * kEntry) * 2);
  PagedKVDesc d = desc_nhd(cache, nullptr, 1, kEntry, kEntry - 64, kBlock);
  d.layout = KVLayout::kLatent;
  d.k_stride_page = static_cast<int64_t>(kBlock) * kEntry;
  d.k_stride_token = kEntry;
  d.k_stride_head = kEntry;
  const Tensor td = t2b(dst, kTokens, kEntry);
  kops::kv::GatherSpec spec{};
  spec.block_table = table;
  spec.block_table_stride = pages;
  spec.cu_seq_lens = cu;
  spec.num_reqs = 1;
  spec.num_tokens = kTokens;
  bool ok = status_ok(kops::kv::gather_and_maybe_dequant_cache(td, d, spec, ctx), c.id.c_str());
  if (ok) {
    ok = measure(c, [&] { (void)kops::kv::gather_and_maybe_dequant_cache(td, d, spec, ctx); }, st);
  }
  cudaFree(cache);
  cudaFree(dst);
  cudaFree(table);
  cudaFree(cu);
  return ok;
}

bool run_relayout(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kSeq = 1, kPages = 512, kPage = 16, kH = 4, kD = 128;
  void* src = nullptr;
  void* dst = nullptr;
  int32_t* ind = nullptr;
  if (!cuda_ok(cudaMalloc(&src, static_cast<size_t>(kPages * kPage * kH * kD) * 2), "src") ||
      !cuda_ok(cudaMalloc(&dst, static_cast<size_t>(kPages * kPage * kH * kD) * 2), "dst") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ind), 2 * 4), "ind")) {
    return false;
  }
  std::vector<int32_t> iv = {0, kPages};
  cudaMemcpy(ind, iv.data(), iv.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(src, 1, static_cast<size_t>(kPages * kPage * kH * kD) * 2);
  PagedKVDesc s = desc_nhd(src, nullptr, kH, kD, kD, kPage);
  PagedKVDesc d = desc_nhd(dst, nullptr, kH, kD, kD, kPage);
  d.layout = KVLayout::kHND;
  d.k_stride_page = static_cast<int64_t>(kH) * kPage * kD;
  d.k_stride_token = kD;
  d.k_stride_head = static_cast<int64_t>(kPage) * kD;
  const Tensor ti = tensor1d(ind, DType::kInt32, 2);
  bool ok = status_ok(kops::kv::relayout_pages(s, d, ti, ti, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::kv::relayout_pages(s, d, ti, ti, ctx); }, st);
  (void)kSeq;
  cudaFree(src);
  cudaFree(dst);
  cudaFree(ind);
  return ok;
}

bool run_b3(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t tokens = c.rows;
  const int H = 8, D = 128, kPage = 16;
  int64_t* slots = nullptr;
  std::vector<int64_t> sh(static_cast<size_t>(tokens));
  for (int64_t i = 0; i < tokens; ++i) sh[static_cast<size_t>(i)] = i;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&slots), static_cast<size_t>(tokens) * 8),
               "slots")) {
    return false;
  }
  cudaMemcpy(slots, sh.data(), static_cast<size_t>(tokens) * 8, cudaMemcpyHostToDevice);
  const Tensor ts = t1(slots, tokens);
  bool ok = false;
  if (c.op == "reshape_per_token_head") {
    const int64_t elems = tokens * H * D;
    const int64_t pages = (tokens + kPage - 1) / kPage;
    void* key = nullptr; void* val = nullptr; void* kc = nullptr; void* vc = nullptr;
    float* ks = nullptr; float* vs = nullptr;
    if (cuda_ok(cudaMalloc(&key, elems * 2), "key") && cuda_ok(cudaMalloc(&val, elems * 2), "val") &&
        cuda_ok(cudaMalloc(&kc, elems), "kc") && cuda_ok(cudaMalloc(&vc, elems), "vc") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ks), pages * kPage * H * 4), "ks") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&vs), pages * kPage * H * 4), "vs")) {
      PagedKVDesc d = desc_nhd(kc, vc, H, D, D, kPage);
      d.k_codec = KVCodec::kFP8PerTokenHead; d.v_codec = d.k_codec;
      d.k_dtype = DType::kFloat8E4M3;
      d.v_dtype = DType::kFloat8E4M3;
      d.k_token_scales = ks; d.v_token_scales = vs;
      const Tensor tk = t3b(key, tokens, H, D);
      const Tensor tv = t3b(val, tokens, H, D);
      auto fn = [&]() { return kops::kv::reshape_and_cache(tk, tv, d, ts, ctx); };
      ok = status_ok(fn(), c.id.c_str());
      if (ok) ok = measure(c, fn, st);
      cudaFree(key); cudaFree(val); cudaFree(kc); cudaFree(vc); cudaFree(ks); cudaFree(vs);
    }
  } else if (c.op == "ds_mla") {
    const int64_t pages = (tokens + kPage - 1) / kPage;
    void* kc = nullptr; void* pe = nullptr; uint8_t* cache = nullptr;
    if (cuda_ok(cudaMalloc(&kc, tokens * 512 * 2), "kc") &&
        cuda_ok(cudaMalloc(&pe, tokens * 64 * 2), "pe") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cache), pages * kPage * 656), "cache")) {
      PagedKVDesc d{};
      d.k_data = cache; d.k_dtype = DType::kFloat8E4M3; d.k_codec = KVCodec::kMLAFp8DS;
      d.layout = KVLayout::kLatent; d.page_size = kPage; d.num_kv_heads = 1;
      d.head_dim_k = 576; d.head_dim_v = 512; d.entry_bytes = 656;
      d.stride_unit = kops::StrideUnit::kByte; d.k_stride_page = kPage * 656;
      d.k_stride_token = 656; d.k_stride_head = 656;
      d.table_format = PageTableFormat::kSlotMapping;
      const Tensor tc = t2b(kc, tokens, 512);
      const Tensor tp = t2b(pe, tokens, 64);
      auto fn = [&]() { return kops::kv::concat_and_cache_mla(tc, tp, d, ts, ctx); };
      ok = status_ok(fn(), c.id.c_str());
      if (ok) ok = measure(c, fn, st);
      cudaFree(kc); cudaFree(pe); cudaFree(cache);
    }
  } else if (c.op == "indexer_fp8") {
    const int64_t record = D + 4;
    const int64_t pages = (tokens + kPage - 1) / kPage;
    void* key = nullptr; uint8_t* cache = nullptr;
    if (cuda_ok(cudaMalloc(&key, tokens * D * 2), "key") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cache), pages * kPage * record), "cache")) {
      PagedKVDesc d{};
      d.k_data = cache; d.k_dtype = DType::kUInt8; d.k_codec = KVCodec::kIndexerFP8;
      d.layout = KVLayout::kLatent; d.page_size = kPage; d.num_kv_heads = 1;
      d.head_dim_k = D; d.entry_bytes = record; d.stride_unit = kops::StrideUnit::kByte;
      d.k_stride_page = kPage * record; d.k_stride_token = record; d.k_stride_head = record;
      d.table_format = PageTableFormat::kSlotMapping;
      const Tensor tk = t2b(key, tokens, D);
      auto fn = [&]() { return kops::kv::indexer_k_quant_and_cache(tk, d, ts, 128, false, ctx); };
      ok = status_ok(fn(), c.id.c_str());
      if (ok) ok = measure(c, fn, st);
      cudaFree(key); cudaFree(cache);
    }
  } else if (c.op == "fused_qkv_append" || c.op == "fused_kv_append") {
    // fused_qkv_append: full QKV projection tile (Q head count included).
    // fused_kv_append : the append case's own shape (Hq = 0, Hkv = 4, D = 128), so the only work
    //                   is norm + RoPE on K plus the K/V cache write — the same bytes append moves,
    //                   which is what "fusion instead of a separate norm+RoPE pass" has to beat.
    const bool kv_only = c.op == "fused_kv_append";
    const int Hq = kv_only ? 0 : 32, Hkv = kv_only ? 4 : 8;
    const int64_t width = static_cast<int64_t>(Hq + 2 * Hkv) * D;
    const int64_t pages = (tokens + kPage - 1) / kPage;
    void* qkv = nullptr; void* qw = nullptr; void* kw = nullptr; void* pos = nullptr;
    void* cs = nullptr; void* kc = nullptr; void* vc = nullptr;
    if (cuda_ok(cudaMalloc(&qkv, tokens * width * 2), "qkv") &&
        cuda_ok(cudaMalloc(&kw, D * 2), "kw") &&
        cuda_ok(cudaMalloc(&pos, tokens * 8), "pos") &&
        cuda_ok(cudaMalloc(&cs, 4096 * D * 4), "cs") &&
        cuda_ok(cudaMalloc(&kc, pages * kPage * Hkv * D * 2), "kc") &&
        cuda_ok(cudaMalloc(&vc, pages * kPage * Hkv * D * 2), "vc") &&
        (!kv_only || cuda_ok(cudaMalloc(&qw, D * 2), "qw"))) {
      cudaMemcpy(pos, sh.data(), static_cast<size_t>(tokens) * 8, cudaMemcpyHostToDevice);
      cudaMemset(cs, 0x3c, 4096 * D * 4);   // normal floats, not zero pages
      PagedKVDesc cache = desc_nhd(kc, vc, Hkv, D, D, kPage);
      kops::kv::FusedQKVAppendParams p{};
      p.qkv = qkv; p.qkv_token_stride = width; p.q_norm_weight = qw; p.k_norm_weight = kw;
      p.norm_eps = 1e-5f; p.positions = static_cast<const int64_t*>(pos);
      p.cos_sin_cache = cs; p.rot_dim = D;
      p.rot_offset = 0; p.rope_style = kops::RopeStyle::kNeoX; p.cache = cache;
      p.slot_mapping = slots; p.num_tokens = static_cast<int32_t>(tokens); p.num_q_heads = Hq;
      p.num_kv_heads = Hkv; p.head_dim = D;
      auto fn = [&]() { return kops::kv::fused_qkv_norm_rope_append(p, DType::kBFloat16, ctx); };
      ok = status_ok(fn(), c.id.c_str());
      if (ok) ok = measure(c, fn, st);
      cudaFree(qkv); cudaFree(qw); cudaFree(kw); cudaFree(pos); cudaFree(cs); cudaFree(kc); cudaFree(vc);
    }
  } else if (c.op == "requantize_pages") {
    const int64_t elems = tokens * H * D;
    const int64_t pages = (tokens + kPage - 1) / kPage;
    void* sk = nullptr; void* sv = nullptr; uint8_t* dk = nullptr; uint8_t* dv = nullptr;
    float* dks = nullptr; float* dvs = nullptr; int64_t* pairs = nullptr;
    std::vector<int64_t> pv(static_cast<size_t>(pages) * 2);
    for (int64_t i = 0; i < pages; ++i) { pv[2*i] = i; pv[2*i+1] = i; }
    if (cuda_ok(cudaMalloc(&sk, elems * 2), "sk") && cuda_ok(cudaMalloc(&sv, elems * 2), "sv") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dk), elems), "dk") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dv), elems), "dv") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dks), pages * kPage * H * 4), "dks") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dvs), pages * kPage * H * 4), "dvs") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pairs), pages * 2 * 8), "pairs")) {
      cudaMemcpy(pairs, pv.data(), pv.size() * 8, cudaMemcpyHostToDevice);
      PagedKVDesc s = desc_nhd(sk, sv, H, D, D, kPage);
      PagedKVDesc t = desc_nhd(dk, dv, H, D, D, kPage);
      t.k_codec = KVCodec::kInt8PerTokenHead; t.v_codec = t.k_codec;
      t.k_dtype = t.v_dtype = DType::kUInt8; t.k_token_scales = dks; t.v_token_scales = dvs;
      const Tensor tp = tensor2d(pairs, DType::kInt64, pages, 2);
      auto fn = [&]() { return kops::kv::requantize_pages(s, t, tp, ctx); };
      ok = status_ok(fn(), c.id.c_str());
      if (ok) ok = measure(c, fn, st);
      cudaFree(sk); cudaFree(sv); cudaFree(dk); cudaFree(dv); cudaFree(dks); cudaFree(dvs); cudaFree(pairs);
    }
  } else if (c.op == "kv_scale_update") {
    const int64_t elems = tokens * H * D;
    void* key = nullptr; void* val = nullptr; float* ka = nullptr; float* va = nullptr;
    if (cuda_ok(cudaMalloc(&key, elems * 2), "key") && cuda_ok(cudaMalloc(&val, elems * 2), "val") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ka), H * 4), "ka") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void**>(&va), H * 4), "va")) {
      const Tensor tk = t3b(key, tokens, H, D);
      const Tensor tv = t3b(val, tokens, H, D);
      const Tensor tka = tensor1d(ka, DType::kFloat32, H);
      const Tensor tva = tensor1d(va, DType::kFloat32, H);
      auto fn = [&]() { return kops::kv::kv_scale_update(tka, tva, tk, tv, ctx); };
      ok = status_ok(fn(), c.id.c_str());
      if (ok) ok = measure(c, fn, st);
      cudaFree(key); cudaFree(val); cudaFree(ka); cudaFree(va);
    }
  } else if (c.op == "mla_rope_fused") {
    const int Hq = 16;
    const int64_t rank = 512, rot = 64, pages = (tokens + kPage - 1) / kPage;
    void* q = nullptr; void* kp = nullptr; void* kc = nullptr; void* cache = nullptr;
    void* pos = nullptr; void* cs = nullptr;
    if (cuda_ok(cudaMalloc(&q, tokens * Hq * rot * 2), "q") &&
        cuda_ok(cudaMalloc(&kp, tokens * rot * 2), "kp") &&
        cuda_ok(cudaMalloc(&kc, tokens * rank * 2), "kc") &&
        cuda_ok(cudaMalloc(&cache, pages * kPage * (rank + rot) * 2), "cache") &&
        cuda_ok(cudaMalloc(&pos, tokens * 8), "pos") &&
        cuda_ok(cudaMalloc(&cs, 4096 * rot * 4), "cs")) {
      cudaMemcpy(pos, sh.data(), static_cast<size_t>(tokens) * 8, cudaMemcpyHostToDevice);
      PagedKVDesc d = desc_nhd(cache, nullptr, 1, rank + rot, rank, kPage);
      d.layout = KVLayout::kLatent;
      const Tensor tq = tensor3d(q, DType::kBFloat16, tokens, Hq, rot);
      const Tensor tk = t2b(kp, tokens, rot);
      const Tensor tc = t2b(kc, tokens, rank);
      const Tensor tcs = tensor2d(cs, DType::kFloat32, 4096, rot);
      const Tensor tp = t1(pos, tokens);
      auto fn = [&]() {
        return kops::kv::concat_and_cache_mla_rope_fused(tp, tq, tk, tc, tcs,
                                                         kops::RopeStyle::kNeoX, ts, d, ctx);
      };
      ok = status_ok(fn(), c.id.c_str());
      if (ok) ok = measure(c, fn, st);
      cudaFree(q); cudaFree(kp); cudaFree(kc); cudaFree(cache); cudaFree(pos); cudaFree(cs);
    }
  }
  cudaFree(slots);
  return ok;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "reshape_per_token_head" || c.op == "ds_mla" || c.op == "indexer_fp8" ||
      c.op == "fused_qkv_append" || c.op == "fused_kv_append" ||
      c.op == "requantize_pages" || c.op == "kv_scale_update" || c.op == "mla_rope_fused") {
    return run_b3(c, ctx, st);
  }
  if (c.op == "reshape_and_cache") return run_reshape(c, ctx, st);
  if (c.op == "reshape_and_cache_legacy") return run_reshape_legacy(c, ctx, st);
  if (c.op == "append_paged_kv") return run_append(c, ctx, st);
  if (c.op == "concat_and_cache_mla_grouped") return run_mla_grouped(c, ctx, st);
  if (c.op == "ring_kv_append") return run_ring(c, ctx, st);
  if (c.op == "concat_and_cache_mla") return run_concat_mla(c, ctx, st);
  if (c.op == "gather_pages") return run_gather(c, ctx, st);
  if (c.op == "scatter_pages") return run_scatter(c, ctx, st);
  if (c.op == "cp_gather_cache") return run_cp_gather(c, ctx, st);
  if (c.op == "copy_pages" || c.op == "zero_pages") return run_copy_zero(c, ctx, st);
  if (c.op == "convert_cache") return run_convert(c, ctx, st);
  if (c.op == "gather_and_maybe_dequant_cache") return run_gather_dequant(c, ctx, st);
  if (c.op == "relayout_pages") return run_relayout(c, ctx, st);
  return false;
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto add = [&](const char* id, const char* op, int64_t rows, int64_t cols, int64_t bytes,
                 int64_t flops = 0) {
    cases.push_back({id, op, DType::kBFloat16, rows, cols, bytes, flops, 10, 100});
  };
  add("reshape.nhd.bf16.t2048.h4.d128", "reshape_and_cache", 2048, 128,
      static_cast<int64_t>(2048) * 4 * 128 * 2 * 4);
  add("reshape.legacy.bf16.t2048.h4.d128", "reshape_and_cache_legacy", 2048, 128,
      static_cast<int64_t>(2048) * 4 * 128 * 2 * 4);
  add("append.nhd.bf16.t2048.h4.d128", "append_paged_kv", 2048, 128,
      static_cast<int64_t>(2048) * 4 * 128 * 2 * 4);
  add("ring.bf16.t1024.h4.d128", "ring_kv_append", 1024, 128,
      static_cast<int64_t>(1024) * 4 * 128 * 2 * 2);
  add("concat_mla.t2048.r512.p64", "concat_and_cache_mla", 2048, 576,
      static_cast<int64_t>(2048) * (512 + 64) * 2 * 2);
  add("concat_mla_grouped.l2.t1024.r512.p64", "concat_and_cache_mla_grouped", 1024, 576,
      static_cast<int64_t>(2 * 1024) * (512 + 64) * 2 * 2);
  add("gather_pages.nhd.bf16.t4096.h4.d128", "gather_pages", 4096, 512,
      static_cast<int64_t>(4096) * 4 * 128 * 2 * 2);
  add("scatter_pages.nhd.bf16.t4096.h4.d128", "scatter_pages", 4096, 512,
      static_cast<int64_t>(4096) * 4 * 128 * 2 * 2);
  add("cp_gather.bytes.t4096.e576", "cp_gather_cache", 4096, 576,
      static_cast<int64_t>(4096) * 576 * 2);
  add("copy_pages.bytes.p512.b4096", "copy_pages", 512, 4096,
      static_cast<int64_t>(2 * 2 * 512) * 4096 * 2);
  add("zero_pages.bytes.p512.b4096", "zero_pages", 512, 4096,
      static_cast<int64_t>(2 * 2 * 512) * 4096);
  add("convert.bf16_to_f32.b256.s4096", "convert_cache", 256, 4096,
      static_cast<int64_t>(256) * 4096 * 6);
  add("gather_dequant.t4096.e576", "gather_and_maybe_dequant_cache", 4096, 576,
      static_cast<int64_t>(4096) * 576 * 2 * 2);
  add("relayout.nhd_to_hnd.p512", "relayout_pages", 512, 512,
      static_cast<int64_t>(512) * 16 * 4 * 128 * 2 * 2);
  add("reshape_per_token_head.fp8.t2048.h8.d128", "reshape_per_token_head", 2048, 128,
      static_cast<int64_t>(2048) * 8 * 128 * 4);
  add("ds_mla.fp8.t2048.r512.p64", "ds_mla", 2048, 576,
      static_cast<int64_t>(2048) * (656 + 1024));
  add("indexer_fp8.t2048.d128", "indexer_fp8", 2048, 128,
      static_cast<int64_t>(2048) * (128 + 4) * 3);
  add("fused_qkv_append.bf16.t2048.hq32.hkv8.d128", "fused_qkv_append", 2048, 128,
      static_cast<int64_t>(2048) * (32 + 16) * 128 * 2 * 3);
  add("fused_kv_append.bf16.t2048.hkv4.d128", "fused_kv_append", 2048, 128,
      static_cast<int64_t>(2048) * (0 + 8) * 128 * 2 * 2);
  add("requantize.int8.t2048.h8.d128", "requantize_pages", 2048, 128,
      static_cast<int64_t>(2048) * 8 * 128 * 4);
  add("kv_scale_update.bf16.t2048.h8.d128", "kv_scale_update", 2048, 128,
      static_cast<int64_t>(2048) * 8 * 128 * 2 * 2);
  add("mla_rope_fused.bf16.t2048.hq16.r512.p64", "mla_rope_fused", 2048, 576,
      static_cast<int64_t>(2048) * (576 + 16 * 64) * 2 * 2);
  return cases;
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
