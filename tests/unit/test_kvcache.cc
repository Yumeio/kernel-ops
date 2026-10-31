// =============================================================================
// kernel-ops | tests/unit/test_kvcache.cc
// B2 kvcache family:
//   K01 reshape_and_cache (NHD/HND/legacy, native + FP8), append_paged_kv,
//       ring_kv_append, concat_and_cache_mla (+ grouped)
//   K02 gather_pages / scatter_pages, copy_pages / zero_pages,
//       cp_gather_cache, gather_and_maybe_dequant_cache
//   K03 convert_cache, kv_relayout_pages
//   host copies swap_pages / copy_batch_async
//
// Oracle: local fp64/bit-exact host loops over the documented layouts plus
// golden byte records for the gather kernels. FP8 writes are checked by a
// gather round trip against the dequantized source (e4m3 step), not against a
// third-party library. Every test asserts the Status contract where a negative
// case is meaningful; graph capture/replay covers the two hottest paths.
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

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

using kops::DType;
using kops::LaunchContext;
using kops::kv::KVCodec;
using kops::kv::KVLayout;
using kops::kv::PageTableFormat;
using kops::Status;
using kops::Tensor;
using kops::kv::PagedKVDesc;

constexpr DType kBf16 = DType::kBFloat16;

template <typename T>
struct DevBuf {
  T* p = nullptr;
  explicit DevBuf(size_t n) : p(kops_test::device_alloc<T>(n)) {}
  ~DevBuf() { kops_test::device_free(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  bool ok() const { return p != nullptr; }
};

void fill_uniform(std::vector<float>& v, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& x : v) x = dist(rng);
}

template <typename T>
std::vector<T> cast_from_float(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(out[i], src[i]);
  return out;
}

template <typename T>
std::vector<double> as_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<double>(kops_test::to_float(src[i]));
  return out;
}

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

Tensor tensor2d(void* data, DType dtype, int64_t rows, int64_t cols, int64_t stride = 0) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = stride == 0 ? cols : stride;
  t.stride[1] = 1;
  return t;
}

Tensor tensor3d(void* data, DType dtype, int64_t d0, int64_t d1, int64_t d2) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 3;
  t.device = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.stride[0] = d1 * d2;
  t.stride[1] = d2;
  t.stride[2] = 1;
  return t;
}

Tensor tensor4d(void* data, DType dtype, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 4;
  t.device = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.shape[3] = d3;
  t.stride[0] = d1 * d2 * d3;
  t.stride[1] = d2 * d3;
  t.stride[2] = d3;
  t.stride[3] = 1;
  return t;
}

int64_t nhd_off(int64_t page, int64_t off, int64_t h, int64_t d, int64_t heads, int64_t dim,
                int64_t ps) {
  return page * ps * heads * dim + off * heads * dim + h * dim + d;
}

int64_t hnd_off(int64_t page, int64_t off, int64_t h, int64_t d, int64_t heads, int64_t dim,
                int64_t ps) {
  return page * heads * ps * dim + h * ps * dim + off * dim + d;
}

PagedKVDesc make_desc(KVLayout layout, void* k, void* v, int heads, int dk, int dv, int page,
                      int pages, DType dtype) {
  PagedKVDesc d{};
  d.k_data = k;
  d.v_data = v;
  d.k_dtype = dtype;
  d.v_dtype = dtype;
  d.k_codec = KVCodec::kNative;
  d.v_codec = KVCodec::kNative;
  d.layout = layout;
  d.page_size = page;
  d.num_kv_heads = heads;
  d.head_dim_k = dk;
  d.head_dim_v = dv;
  d.stride_unit = kops::StrideUnit::kElement;
  if (layout == KVLayout::kHND) {
    d.k_stride_page = static_cast<int64_t>(heads) * page * dk;
    d.k_stride_token = dk;
    d.k_stride_head = static_cast<int64_t>(page) * dk;
    d.v_stride_page = static_cast<int64_t>(heads) * page * dv;
    d.v_stride_token = dv;
    d.v_stride_head = static_cast<int64_t>(page) * dv;
  } else {
    d.k_stride_page = static_cast<int64_t>(page) * heads * dk;
    d.k_stride_token = static_cast<int64_t>(heads) * dk;
    d.k_stride_head = dk;
    d.v_stride_page = static_cast<int64_t>(page) * heads * dv;
    d.v_stride_token = static_cast<int64_t>(heads) * dv;
    d.v_stride_head = dv;
  }
  d.table_format = PageTableFormat::kSlotMapping;
  (void)pages;
  return d;
}

// -----------------------------------------------------------------------------
// K01 reshape_and_cache
// -----------------------------------------------------------------------------

void test_reshape_nhd_bf16(LaunchContext& ctx) {
  constexpr int kPages = 3, kPage = 4, kHeads = 2, kDk = 8, kDv = 8;
  DevBuf<__nv_bfloat16> key(5 * kHeads * kDk);
  DevBuf<__nv_bfloat16> val(5 * kHeads * kDv);
  DevBuf<__nv_bfloat16> kc(kPages * kPage * kHeads * kDk);
  DevBuf<__nv_bfloat16> vc(kPages * kPage * kHeads * kDv);
  DevBuf<int64_t> slots(3);
  if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !slots.ok()) return;

  std::vector<float> kf(5 * kHeads * kDk);
  std::vector<float> vf(5 * kHeads * kDv);
  fill_uniform(kf, 11);
  fill_uniform(vf, 12);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(key.p, kh);
  kops_test::copy_in(val.p, vh);
  const std::vector<int64_t> sh = {0, 5, -1};
  kops_test::copy_in(slots.p, sh);
  cudaMemset(kc.p, 0, kPages * kPage * kHeads * kDk * sizeof(__nv_bfloat16));
  cudaMemset(vc.p, 0, kPages * kPage * kHeads * kDv * sizeof(__nv_bfloat16));

  PagedKVDesc d = make_desc(KVLayout::kNHD, kc.p, vc.p, kHeads, kDk, kDv, kPage, kPages, kBf16);
  const Tensor tk = tensor3d(key.p, kBf16, 5, kHeads, kDk);
  const Tensor tv = tensor3d(val.p, kBf16, 5, kHeads, kDv);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, 3);
  KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, d, ts, ctx), "reshape nhd bf16");

  const auto got_k = kops_test::copy_out(kc.p, kPages * kPage * kHeads * kDk);
  const auto got_v = kops_test::copy_out(vc.p, kPages * kPage * kHeads * kDv);
  std::vector<__nv_bfloat16> want_k(kPages * kPage * kHeads * kDk);
  std::vector<__nv_bfloat16> want_v(kPages * kPage * kHeads * kDv);
  for (int t = 0; t < 3; ++t) {
    if (sh[static_cast<size_t>(t)] < 0) continue;
    const int64_t slot = sh[static_cast<size_t>(t)];
    const int64_t page = slot / kPage, off = slot % kPage;
    for (int h = 0; h < kHeads; ++h) {
      for (int dd = 0; dd < kDk; ++dd) {
        want_k[static_cast<size_t>(nhd_off(page, off, h, dd, kHeads, kDk, kPage))] =
            kh[static_cast<size_t>(t * kHeads * kDk + h * kDk + dd)];
        want_v[static_cast<size_t>(nhd_off(page, off, h, dd, kHeads, kDv, kPage))] =
            vh[static_cast<size_t>(t * kHeads * kDv + h * kDv + dd)];
      }
    }
  }
  KOPS_TEST_CHECK(std::memcmp(got_k.data(), want_k.data(), got_k.size() * sizeof(__nv_bfloat16)) == 0);
  KOPS_TEST_CHECK(std::memcmp(got_v.data(), want_v.data(), got_v.size() * sizeof(__nv_bfloat16)) == 0);
}

void test_reshape_hnd_bf16(LaunchContext& ctx) {
  constexpr int kPages = 2, kPage = 4, kHeads = 2, kD = 8;
  DevBuf<__nv_bfloat16> key(2 * kHeads * kD);
  DevBuf<__nv_bfloat16> val(2 * kHeads * kD);
  DevBuf<__nv_bfloat16> kc(kPages * kPage * kHeads * kD);
  DevBuf<__nv_bfloat16> vc(kPages * kPage * kHeads * kD);
  DevBuf<int64_t> slots(2);
  if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !slots.ok()) return;
  std::vector<float> kf(2 * kHeads * kD), vf(2 * kHeads * kD);
  fill_uniform(kf, 21);
  fill_uniform(vf, 22);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(key.p, kh);
  kops_test::copy_in(val.p, vh);
  const std::vector<int64_t> sh = {1, 6};
  kops_test::copy_in(slots.p, sh);
  cudaMemset(kc.p, 0, kPage * kPages * kHeads * kD * sizeof(__nv_bfloat16));
  cudaMemset(vc.p, 0, kPage * kPages * kHeads * kD * sizeof(__nv_bfloat16));
  PagedKVDesc d = make_desc(KVLayout::kHND, kc.p, vc.p, kHeads, kD, kD, kPage, kPages, kBf16);
  const Tensor tk = tensor3d(key.p, kBf16, 2, kHeads, kD);
  const Tensor tv = tensor3d(val.p, kBf16, 2, kHeads, kD);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, 2);
  KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, d, ts, ctx), "reshape hnd bf16");

  const auto gotk = kops_test::copy_out(kc.p, kPages * kPage * kHeads * kD);
  const auto gotv = kops_test::copy_out(vc.p, kPages * kPage * kHeads * kD);
  std::vector<__nv_bfloat16> wantk(kPages * kPage * kHeads * kD);
  std::vector<__nv_bfloat16> wantv(kPages * kPage * kHeads * kD);
  for (int t = 0; t < 2; ++t) {
    const int64_t slot = sh[static_cast<size_t>(t)];
    const int64_t page = slot / kPage, off = slot % kPage;
    for (int h = 0; h < kHeads; ++h) {
      for (int dd = 0; dd < kD; ++dd) {
        wantk[static_cast<size_t>(hnd_off(page, off, h, dd, kHeads, kD, kPage))] =
            kh[static_cast<size_t>(t * kHeads * kD + h * kD + dd)];
        wantv[static_cast<size_t>(hnd_off(page, off, h, dd, kHeads, kD, kPage))] =
            vh[static_cast<size_t>(t * kHeads * kD + h * kD + dd)];
      }
    }
  }
  KOPS_TEST_CHECK(std::memcmp(gotk.data(), wantk.data(), gotk.size() * sizeof(__nv_bfloat16)) == 0);
  KOPS_TEST_CHECK(std::memcmp(gotv.data(), wantv.data(), gotv.size() * sizeof(__nv_bfloat16)) == 0);
}

void test_reshape_legacy_bf16(LaunchContext& ctx) {
  constexpr int kPages = 2, kPage = 4, kHeads = 2, kD = 8;
  constexpr int kX = 8;  // 16 B / sizeof(bf16)
  DevBuf<__nv_bfloat16> key(2 * kHeads * kD);
  DevBuf<__nv_bfloat16> val(2 * kHeads * kD);
  DevBuf<__nv_bfloat16> kc(kPages * kHeads * (kD / kX) * kPage * kX);
  DevBuf<__nv_bfloat16> vc(kPages * kHeads * kD * kPage);
  DevBuf<int64_t> slots(2);
  if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !slots.ok()) return;
  std::vector<float> kf(2 * kHeads * kD), vf(2 * kHeads * kD);
  fill_uniform(kf, 31);
  fill_uniform(vf, 32);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(key.p, kh);
  kops_test::copy_in(val.p, vh);
  const std::vector<int64_t> sh = {2, 5};
  kops_test::copy_in(slots.p, sh);
  cudaMemset(kc.p, 0, kc.p ? kPages * kHeads * (kD / kX) * kPage * kX * sizeof(__nv_bfloat16) : 0);
  cudaMemset(vc.p, 0, kPages * kHeads * kD * kPage * sizeof(__nv_bfloat16));
  PagedKVDesc d = make_desc(KVLayout::kVllmLegacy, kc.p, vc.p, kHeads, kD, kD, kPage, kPages, kBf16);
  d.legacy_x = kX;
  const Tensor tk = tensor3d(key.p, kBf16, 2, kHeads, kD);
  const Tensor tv = tensor3d(val.p, kBf16, 2, kHeads, kD);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, 2);
  KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, d, ts, ctx), "reshape legacy bf16");

  const auto gotk = kops_test::copy_out(kc.p, kPages * kHeads * (kD / kX) * kPage * kX);
  const auto gotv = kops_test::copy_out(vc.p, kPages * kHeads * kD * kPage);
  std::vector<__nv_bfloat16> wantk(gotk.size());
  std::vector<__nv_bfloat16> wantv(gotv.size());
  for (int t = 0; t < 2; ++t) {
    const int64_t slot = sh[static_cast<size_t>(t)];
    const int64_t page = slot / kPage, off = slot % kPage;
    for (int h = 0; h < kHeads; ++h) {
      for (int dd = 0; dd < kD; ++dd) {
        const int64_t ki = page * kHeads * kD * kPage +
                           h * kD * kPage + (dd / kX) * kPage * kX + off * kX + (dd % kX);
        const int64_t vi = page * kHeads * kD * kPage + h * kD * kPage + dd * kPage + off;
        wantk[static_cast<size_t>(ki)] = kh[static_cast<size_t>(t * kHeads * kD + h * kD + dd)];
        wantv[static_cast<size_t>(vi)] = vh[static_cast<size_t>(t * kHeads * kD + h * kD + dd)];
      }
    }
  }
  KOPS_TEST_CHECK(std::memcmp(gotk.data(), wantk.data(), gotk.size() * sizeof(__nv_bfloat16)) == 0);
  KOPS_TEST_CHECK(std::memcmp(gotv.data(), wantv.data(), gotv.size() * sizeof(__nv_bfloat16)) == 0);
}

void test_reshape_fp8_roundtrip(LaunchContext& ctx) {
  constexpr int kPages = 2, kPage = 4, kHeads = 2, kD = 8;
  DevBuf<__nv_bfloat16> key(3 * kHeads * kD);
  DevBuf<__nv_bfloat16> val(3 * kHeads * kD);
  DevBuf<uint8_t> kc(kPages * kPage * kHeads * kD);
  DevBuf<uint8_t> vc(kPages * kPage * kHeads * kD);
  DevBuf<int64_t> slots(3);
  DevBuf<float> kscale(1), vscale(1);
  DevBuf<float> head_scales(kHeads);
  if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !slots.ok() || !kscale.ok() ||
      !vscale.ok() || !head_scales.ok()) {
    return;
  }
  std::vector<float> kf(3 * kHeads * kD), vf(3 * kHeads * kD);
  fill_uniform(kf, 41, -2.0f, 2.0f);
  fill_uniform(vf, 42, -2.0f, 2.0f);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(key.p, kh);
  kops_test::copy_in(val.p, vh);
  const std::vector<int64_t> sh = {0, 5, 7};
  kops_test::copy_in(slots.p, sh);
  const float ks = 0.01f, vs = 0.01f;
  kops_test::copy_in(kscale.p, std::vector<float>{ks});
  kops_test::copy_in(vscale.p, std::vector<float>{vs});
  kops_test::copy_in(head_scales.p, std::vector<float>{0.5f, 0.25f});
  cudaMemset(kc.p, 0, kc.p ? 1 : 0);

  PagedKVDesc d = make_desc(KVLayout::kNHD, kc.p, vc.p, kHeads, kD, kD, kPage, kPages, kBf16);
  d.k_codec = KVCodec::kFP8PerTensor;
  d.v_codec = KVCodec::kFP8PerTensor;
  d.k_dtype = DType::kFloat8E4M3;
  d.v_dtype = DType::kFloat8E4M3;
  d.k_scale = kscale.p;
  d.v_scale = vscale.p;
  d.scale_head_stride = 0;
  const Tensor tk = tensor3d(key.p, kBf16, 3, kHeads, kD);
  const Tensor tv = tensor3d(val.p, kBf16, 3, kHeads, kD);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, 3);
  KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, d, ts, ctx), "reshape fp8");

  // Gather the two written slots back as float and compare to the bf16 source.
  const int64_t items = 2;
  DevBuf<int64_t> idx(items);
  DevBuf<float> out(items * kHeads * kD);
  DevBuf<int64_t> good_slots(2);
  std::vector<int64_t> gs = {0, 5};
  kops_test::copy_in(good_slots.p, gs);
  if (!idx.ok() || !out.ok() || !good_slots.ok()) return;
  kops_test::copy_in(idx.p, gs);
  const Tensor tout = tensor2d(out.p, DType::kFloat32, items, kHeads * kD);
  const Tensor tidx = tensor1d(good_slots.p, DType::kInt64, 2);
  KOPS_EXPECT_OK(kops::kv::gather_pages(tout, d, tidx, true, ctx), "fp8 gather");
  const auto got = kops_test::copy_out(out.p, items * kHeads * kD);
  const auto got_k = kops_test::copy_out(kc.p, kPages * kPage * kHeads * kD);
  KOPS_TEST_CHECK(got_k[0] != 0 || got_k[1] != 0 || got_k[2] != 0);
  // Values are e4m3-rounded with a 0.01 dequant scale.
  for (size_t i = 0; i < got.size(); ++i) {
    const double want = static_cast<double>(kf[i]);
    KOPS_TEST_CHECK(std::fabs(static_cast<double>(got[i]) - want) < 0.20);
  }

  // Per-head scale variant (same code path with scale_head_stride = 1).
  PagedKVDesc hd = d;
  hd.k_codec = KVCodec::kFP8PerHead;
  hd.v_codec = KVCodec::kFP8PerHead;
  hd.k_scale = head_scales.p;
  hd.v_scale = head_scales.p;
  hd.scale_head_stride = 1;
  cudaMemset(kc.p, 0, static_cast<size_t>(kPages * kPage * kHeads * kD));
  cudaMemset(vc.p, 0, static_cast<size_t>(kPages * kPage * kHeads * kD));
  KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, hd, ts, ctx), "reshape fp8 per-head");
  KOPS_EXPECT_OK(kops::kv::gather_pages(tout, hd, tidx, true, ctx), "fp8 per-head gather");
  const auto got_h = kops_test::copy_out(out.p, items * kHeads * kD);
  for (size_t i = 0; i < got_h.size(); ++i) {
    const double want = static_cast<double>(kf[i]);
    KOPS_TEST_CHECK(std::fabs(static_cast<double>(got_h[i]) - want) < 0.20);
  }
  (void)idx;
}

void test_reshape_sweep(LaunchContext& ctx) {
  const int dims[] = {64, 80, 96, 128, 192, 256};
  const int pagesizes[] = {16, 32, 64, 128};
  int seed = 700;
  for (int d : dims) {
    for (int page : pagesizes) {
      constexpr int kHeads = 1, kPages = 2, kT = 2;
      const int64_t elems = static_cast<int64_t>(kHeads) * d;
      DevBuf<__nv_bfloat16> key(kT * elems);
      DevBuf<__nv_bfloat16> val(kT * elems);
      DevBuf<__nv_bfloat16> kc(static_cast<int64_t>(kPages) * page * elems);
      DevBuf<__nv_bfloat16> vc(static_cast<int64_t>(kPages) * page * elems);
      DevBuf<int64_t> slots(kT);
      if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !slots.ok()) return;
      std::vector<float> kf(kT * elems), vf(kT * elems);
      fill_uniform(kf, static_cast<unsigned>(seed++));
      fill_uniform(vf, static_cast<unsigned>(seed++));
      auto kh = cast_from_float<__nv_bfloat16>(kf);
      auto vh = cast_from_float<__nv_bfloat16>(vf);
      kops_test::copy_in(key.p, kh);
      kops_test::copy_in(val.p, vh);
      std::vector<int64_t> sh = {0, page + 1};
      kops_test::copy_in(slots.p, sh);
      cudaMemset(kc.p, 0, static_cast<size_t>(kPages) * page * elems * 2);
      cudaMemset(vc.p, 0, static_cast<size_t>(kPages) * page * elems * 2);
      PagedKVDesc desc = make_desc(KVLayout::kNHD, kc.p, vc.p, kHeads, d, d, page, kPages, kBf16);
      const Tensor tk = tensor3d(key.p, kBf16, kT, kHeads, d);
      const Tensor tv = tensor3d(val.p, kBf16, kT, kHeads, d);
      const Tensor ts = tensor1d(slots.p, DType::kInt64, kT);
      KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, desc, ts, ctx), "reshape sweep");
      const auto gotk = kops_test::copy_out(kc.p, static_cast<int64_t>(kPages) * page * elems);
      const auto gotv = kops_test::copy_out(vc.p, static_cast<int64_t>(kPages) * page * elems);
      for (int t = 0; t < kT; ++t) {
        const int64_t page_id = sh[static_cast<size_t>(t)] / page;
        const int64_t off = sh[static_cast<size_t>(t)] % page;
        const int64_t base = (page_id * page + off) * elems;
        for (int64_t i = 0; i < elems; ++i) {
          KOPS_TEST_CHECK(gotk[static_cast<size_t>(base + i)] == kh[static_cast<size_t>(t * elems + i)]);
          KOPS_TEST_CHECK(gotv[static_cast<size_t>(base + i)] == vh[static_cast<size_t>(t * elems + i)]);
        }
      }
    }
  }
}

void test_reshape_dk_ne_dv(LaunchContext& ctx) {
  constexpr int kHeads = 2, kPage = 16, kPages = 2, kT = 2, kDk = 128, kDv = 64;
  for (int layout = 0; layout < 2; ++layout) {
    DevBuf<__nv_bfloat16> key(kT * kHeads * kDk);
    DevBuf<__nv_bfloat16> val(kT * kHeads * kDv);
    DevBuf<__nv_bfloat16> kc(static_cast<int64_t>(kPages) * kPage * kHeads * kDk);
    DevBuf<__nv_bfloat16> vc(static_cast<int64_t>(kPages) * kPage * kHeads * kDv);
    DevBuf<int64_t> slots(kT);
    if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !slots.ok()) return;
    std::vector<float> kf(kT * kHeads * kDk), vf(kT * kHeads * kDv);
    fill_uniform(kf, 801 + layout);
    fill_uniform(vf, 811 + layout);
    auto kh = cast_from_float<__nv_bfloat16>(kf);
    auto vh = cast_from_float<__nv_bfloat16>(vf);
    kops_test::copy_in(key.p, kh);
    kops_test::copy_in(val.p, vh);
    std::vector<int64_t> sh = {3, 20};
    kops_test::copy_in(slots.p, sh);
    cudaMemset(kc.p, 0, static_cast<size_t>(kPages) * kPage * kHeads * kDk * 2);
    cudaMemset(vc.p, 0, static_cast<size_t>(kPages) * kPage * kHeads * kDv * 2);
    PagedKVDesc d = layout == 0
                        ? make_desc(KVLayout::kNHD, kc.p, vc.p, kHeads, kDk, kDv, kPage, kPages, kBf16)
                        : make_desc(KVLayout::kHND, kc.p, vc.p, kHeads, kDk, kDv, kPage, kPages, kBf16);
    const Tensor tk = tensor3d(key.p, kBf16, kT, kHeads, kDk);
    const Tensor tv = tensor3d(val.p, kBf16, kT, kHeads, kDv);
    const Tensor ts = tensor1d(slots.p, DType::kInt64, kT);
    KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, d, ts, ctx), "reshape Dk != Dv");
    const auto gotk = kops_test::copy_out(kc.p, static_cast<int64_t>(kPages) * kPage * kHeads * kDk);
    const auto gotv = kops_test::copy_out(vc.p, static_cast<int64_t>(kPages) * kPage * kHeads * kDv);
    for (int t = 0; t < kT; ++t) {
      const int64_t page = sh[static_cast<size_t>(t)] / kPage;
      const int64_t off = sh[static_cast<size_t>(t)] % kPage;
      for (int h = 0; h < kHeads; ++h) {
        for (int dd = 0; dd < kDk; ++dd) {
          const int64_t idx = layout == 0 ? nhd_off(page, off, h, dd, kHeads, kDk, kPage)
                                          : hnd_off(page, off, h, dd, kHeads, kDk, kPage);
          KOPS_TEST_CHECK(gotk[static_cast<size_t>(idx)] ==
                          kh[static_cast<size_t>(t * kHeads * kDk + h * kDk + dd)]);
        }
        for (int dd = 0; dd < kDv; ++dd) {
          const int64_t idx = layout == 0 ? nhd_off(page, off, h, dd, kHeads, kDv, kPage)
                                          : hnd_off(page, off, h, dd, kHeads, kDv, kPage);
          KOPS_TEST_CHECK(gotv[static_cast<size_t>(idx)] ==
                          vh[static_cast<size_t>(t * kHeads * kDv + h * kDv + dd)]);
        }
      }
    }
  }
}

void test_append_paged_kv(LaunchContext& ctx) {
  constexpr int kPages = 8, kPage = 4, kHeads = 2, kD = 8;
  DevBuf<__nv_bfloat16> key(3 * kHeads * kD);
  DevBuf<__nv_bfloat16> val(3 * kHeads * kD);
  DevBuf<__nv_bfloat16> kc(kPages * kPage * kHeads * kD);
  DevBuf<__nv_bfloat16> vc(kPages * kPage * kHeads * kD);
  DevBuf<int32_t> indptr(3), pages(3), batch(3), pos(3);
  if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !indptr.ok() || !pages.ok() ||
      !batch.ok() || !pos.ok()) {
    return;
  }
  std::vector<float> kf(3 * kHeads * kD), vf(3 * kHeads * kD);
  fill_uniform(kf, 51);
  fill_uniform(vf, 52);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(key.p, kh);
  kops_test::copy_in(val.p, vh);
  kops_test::copy_in(indptr.p, std::vector<int32_t>{0, 2, 3});
  kops_test::copy_in(pages.p, std::vector<int32_t>{7, 2, 5});
  kops_test::copy_in(batch.p, std::vector<int32_t>{0, 0, 1});
  kops_test::copy_in(pos.p, std::vector<int32_t>{0, 3, 2});
  cudaMemset(kc.p, 0, kPages * kPage * kHeads * kD * sizeof(__nv_bfloat16));
  cudaMemset(vc.p, 0, kPages * kPage * kHeads * kD * sizeof(__nv_bfloat16));
  PagedKVDesc d = make_desc(KVLayout::kNHD, kc.p, vc.p, kHeads, kD, kD, kPage, kPages, kBf16);
  d.table_format = PageTableFormat::kCSR;
  d.kv_indptr = indptr.p;
  d.kv_page_indices = pages.p;
  const Tensor tk = tensor3d(key.p, kBf16, 3, kHeads, kD);
  const Tensor tv = tensor3d(val.p, kBf16, 3, kHeads, kD);
  const Tensor tb = tensor1d(batch.p, DType::kInt32, 3);
  const Tensor tp = tensor1d(pos.p, DType::kInt32, 3);
  KOPS_EXPECT_OK(kops::kv::append_paged_kv(tk, tv, d, tb, tp, ctx), "append_paged_kv");

  const auto gotk = kops_test::copy_out(kc.p, kPages * kPage * kHeads * kD);
  std::vector<__nv_bfloat16> wantk(gotk.size());
  const int written_pages[3] = {7, 7, 5};
  const int written_offs[3] = {0, 3, 2};
  for (int t = 0; t < 3; ++t) {
    for (int h = 0; h < kHeads; ++h) {
      for (int dd = 0; dd < kD; ++dd) {
        wantk[static_cast<size_t>(
            nhd_off(written_pages[t], written_offs[t], h, dd, kHeads, kD, kPage))] =
            kh[static_cast<size_t>(t * kHeads * kD + h * kD + dd)];
      }
    }
  }
  KOPS_TEST_CHECK(std::memcmp(gotk.data(), wantk.data(), gotk.size() * sizeof(__nv_bfloat16)) == 0);
}

void test_ring_kv(LaunchContext& ctx) {
  constexpr int kSlots = 2, kWindow = 4, kHeads = 2, kD = 8, kT = 5;
  DevBuf<__nv_bfloat16> key(kT * kHeads * kD);
  DevBuf<__nv_bfloat16> val(kT * kHeads * kD);
  DevBuf<__nv_bfloat16> rk(kSlots * kWindow * kHeads * kD);
  DevBuf<__nv_bfloat16> rv(kSlots * kWindow * kHeads * kD);
  DevBuf<int32_t> slot(kT);
  DevBuf<int64_t> positions(kT);
  if (!key.ok() || !val.ok() || !rk.ok() || !rv.ok() || !slot.ok() || !positions.ok()) return;
  std::vector<float> kf(kT * kHeads * kD), vf(kT * kHeads * kD);
  fill_uniform(kf, 61);
  fill_uniform(vf, 62);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(key.p, kh);
  kops_test::copy_in(val.p, vh);
  kops_test::copy_in(slot.p, std::vector<int32_t>{0, 0, 1, 0, -1});
  kops_test::copy_in(positions.p, std::vector<int64_t>{0, 1, 5, 4, 2});
  cudaMemset(rk.p, 0, kSlots * kWindow * kHeads * kD * sizeof(__nv_bfloat16));
  cudaMemset(rv.p, 0, kSlots * kWindow * kHeads * kD * sizeof(__nv_bfloat16));
  const Tensor tk = tensor3d(key.p, kBf16, kT, kHeads, kD);
  const Tensor tv = tensor3d(val.p, kBf16, kT, kHeads, kD);
  const Tensor trk = tensor4d(rk.p, kBf16, kSlots, kWindow, kHeads, kD);
  const Tensor trv = tensor4d(rv.p, kBf16, kSlots, kWindow, kHeads, kD);
  const Tensor ts = tensor1d(slot.p, DType::kInt32, kT);
  const Tensor tp = tensor1d(positions.p, DType::kInt64, kT);
  KOPS_EXPECT_OK(kops::kv::ring_kv_append(tk, tv, trk, trv, ts, tp, Tensor{}, Tensor{}, kWindow, ctx),
                 "ring_kv_append");

  const auto got = kops_test::copy_out(rk.p, kSlots * kWindow * kHeads * kD);
  std::vector<__nv_bfloat16> want(got.size());
  const int32_t slots[kT] = {0, 0, 1, 0, -1};
  const int64_t pos[kT] = {0, 1, 5, 4, 2};
  for (int t = 0; t < kT; ++t) {
    if (slots[t] < 0) continue;
    const int64_t rp = pos[t] % kWindow;
    for (int h = 0; h < kHeads; ++h) {
      for (int dd = 0; dd < kD; ++dd) {
        const int64_t idx = ((static_cast<int64_t>(slots[t]) * kWindow + rp) * kHeads + h) * kD + dd;
        want[static_cast<size_t>(idx)] = kh[static_cast<size_t>(t * kHeads * kD + h * kD + dd)];
      }
    }
  }
  KOPS_TEST_CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(__nv_bfloat16)) == 0);
}

void test_concat_mla(LaunchContext& ctx) {
  constexpr int kPages = 3, kPage = 2, kRank = 8, kPe = 8, kT = 2;
  DevBuf<__nv_bfloat16> kv_c(kT * kRank);
  DevBuf<__nv_bfloat16> k_pe(kT * kPe);
  DevBuf<__nv_bfloat16> cache(kPages * kPage * (kRank + kPe));
  DevBuf<int64_t> slots(kT);
  if (!kv_c.ok() || !k_pe.ok() || !cache.ok() || !slots.ok()) return;
  std::vector<float> cf(kT * kRank), pf(kT * kPe);
  fill_uniform(cf, 71);
  fill_uniform(pf, 72);
  auto ch = cast_from_float<__nv_bfloat16>(cf);
  auto ph = cast_from_float<__nv_bfloat16>(pf);
  kops_test::copy_in(kv_c.p, ch);
  kops_test::copy_in(k_pe.p, ph);
  kops_test::copy_in(slots.p, std::vector<int64_t>{1, 4});
  cudaMemset(cache.p, 0, kPages * kPage * (kRank + kPe) * sizeof(__nv_bfloat16));
  PagedKVDesc d = make_desc(KVLayout::kLatent, cache.p, nullptr, 1, kRank + kPe, kRank, kPage, kPages,
                            kBf16);
  const Tensor tk = tensor2d(kv_c.p, kBf16, kT, kRank);
  const Tensor tp = tensor2d(k_pe.p, kBf16, kT, kPe);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, kT);
  KOPS_EXPECT_OK(kops::kv::concat_and_cache_mla(tk, tp, d, ts, ctx), "concat_and_cache_mla");

  const auto got = kops_test::copy_out(cache.p, kPages * kPage * (kRank + kPe));
  std::vector<__nv_bfloat16> want(got.size());
  const int64_t sh[kT] = {1, 4};
  for (int t = 0; t < kT; ++t) {
    const int64_t page = sh[t] / kPage, off = sh[t] % kPage;
    const int64_t base = (page * kPage + off) * (kRank + kPe);
    for (int i = 0; i < kRank; ++i) want[static_cast<size_t>(base + i)] = ch[t * kRank + i];
    for (int i = 0; i < kPe; ++i) want[static_cast<size_t>(base + kRank + i)] = ph[t * kPe + i];
  }
  KOPS_TEST_CHECK(std::memcmp(got.data(), want.data(), got.size() * sizeof(__nv_bfloat16)) == 0);

  // Grouped form: two layers, one cache each.
  constexpr int kLayers = 2;
  DevBuf<__nv_bfloat16> gc(kLayers * kT * kRank);
  DevBuf<__nv_bfloat16> gp(kLayers * kT * kPe);
  DevBuf<__nv_bfloat16> cache2(kLayers * kPages * kPage * (kRank + kPe));
  DevBuf<int64_t> ptrs(kLayers);
  DevBuf<int64_t> gslots(kLayers * kT);
  if (!gc.ok() || !gp.ok() || !cache2.ok() || !ptrs.ok() || !gslots.ok()) return;
  std::vector<float> gcf(kLayers * kT * kRank), gpf(kLayers * kT * kPe);
  fill_uniform(gcf, 73);
  fill_uniform(gpf, 74);
  auto gch = cast_from_float<__nv_bfloat16>(gcf);
  auto gph = cast_from_float<__nv_bfloat16>(gpf);
  kops_test::copy_in(gc.p, gch);
  kops_test::copy_in(gp.p, gph);
  std::vector<int64_t> gpv(static_cast<size_t>(kLayers));
  for (int l = 0; l < kLayers; ++l) {
    gpv[static_cast<size_t>(l)] = reinterpret_cast<int64_t>(
        cache2.p + static_cast<int64_t>(l) * kPages * kPage * (kRank + kPe));
  }
  kops_test::copy_in(ptrs.p, gpv);
  std::vector<int64_t> gsv = {0, 2, kLayers * 0 + 0, 3};
  kops_test::copy_in(gslots.p, gsv);
  const Tensor tgc = tensor3d(gc.p, kBf16, kLayers, kT, kRank);
  const Tensor tgp = tensor3d(gp.p, kBf16, kLayers, kT, kPe);
  const Tensor tptr = tensor1d(ptrs.p, DType::kInt64, kLayers);
  const Tensor tgs = tensor2d(gslots.p, DType::kInt64, kLayers, kT);
  KOPS_EXPECT_OK(kops::kv::concat_and_cache_mla_grouped(tgc, tgp, tptr, Tensor{}, tgs, d, ctx),
                 "concat_and_cache_mla_grouped");
}

// -----------------------------------------------------------------------------
// K02 movement / gathers
// -----------------------------------------------------------------------------

void test_copy_zero_pages(LaunchContext& ctx) {
  constexpr int kLayers = 2, kCaches = 2, kPages = 4, kPageBytes = 64;
  DevBuf<uint8_t> caches(kLayers * kCaches * kPages * kPageBytes);
  DevBuf<int64_t> ptrs(kLayers * kCaches);
  DevBuf<int64_t> pairs(2 * 2);
  DevBuf<int64_t> pages(2);
  if (!caches.ok() || !ptrs.ok() || !pairs.ok() || !pages.ok()) return;
  std::vector<uint8_t> init(static_cast<size_t>(kLayers * kCaches * kPages * kPageBytes));
  for (size_t i = 0; i < init.size(); ++i) init[i] = static_cast<uint8_t>(i & 0xFFu);
  kops_test::copy_in(caches.p, init);
  std::vector<int64_t> pv(static_cast<size_t>(kLayers * kCaches));
  for (int i = 0; i < kLayers * kCaches; ++i) {
    pv[static_cast<size_t>(i)] = reinterpret_cast<int64_t>(caches.p + i * kPages * kPageBytes);
  }
  kops_test::copy_in(ptrs.p, pv);
  kops_test::copy_in(pairs.p, std::vector<int64_t>{0, 2, 1, 3});
  const Tensor tp = tensor1d(ptrs.p, DType::kInt64, kLayers * kCaches);
  const Tensor tq = tensor2d(pairs.p, DType::kInt64, 2, 2);
  KOPS_EXPECT_OK(kops::kv::copy_pages(tp, tq, kPageBytes, kCaches, ctx), "copy_pages");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(caches.p, init.size());
  std::vector<uint8_t> want = init;
  for (int c = 0; c < kLayers * kCaches; ++c) {
    const size_t base = static_cast<size_t>(c) * kPages * kPageBytes;
    std::memcpy(&want[base + 2 * kPageBytes], &init[base + 0 * kPageBytes], kPageBytes);
    std::memcpy(&want[base + 3 * kPageBytes], &init[base + 1 * kPageBytes], kPageBytes);
  }
  KOPS_TEST_CHECK(std::memcmp(got.data(), want.data(), got.size()) == 0);

  kops_test::copy_in(pages.p, std::vector<int64_t>{1, 3});
  const Tensor tz = tensor1d(pages.p, DType::kInt64, 2);
  KOPS_EXPECT_OK(kops::kv::zero_pages(tp, tz, kPageBytes, kCaches, ctx), "zero_pages");
  cudaDeviceSynchronize();
  const auto zeroed = kops_test::copy_out(caches.p, init.size());
  for (int c = 0; c < kLayers * kCaches; ++c) {
    const size_t base = static_cast<size_t>(c) * kPages * kPageBytes;
    for (size_t i = 0; i < static_cast<size_t>(kPageBytes); ++i) {
      KOPS_TEST_CHECK(zeroed[base + kPageBytes + i] == 0);
      KOPS_TEST_CHECK(zeroed[base + 3 * kPageBytes + i] == 0);
    }
  }
}

void test_gather_scatter_pages(LaunchContext& ctx) {
  constexpr int kPages = 4, kPage = 4, kHeads = 2, kD = 8;
  const int64_t elems = static_cast<int64_t>(kHeads) * kD;
  DevBuf<__nv_bfloat16> cache(kPages * kPage * elems);
  DevBuf<int64_t> idx(2);
  DevBuf<__nv_bfloat16> out(2 * elems);
  if (!cache.ok() || !idx.ok() || !out.ok()) return;
  std::vector<float> cf(kPages * kPage * elems);
  fill_uniform(cf, 81);
  auto ch = cast_from_float<__nv_bfloat16>(cf);
  kops_test::copy_in(cache.p, ch);
  kops_test::copy_in(idx.p, std::vector<int64_t>{0, 5});
  PagedKVDesc d = make_desc(KVLayout::kNHD, cache.p, nullptr, kHeads, kD, kD, kPage, kPages, kBf16);
  const Tensor to = tensor2d(out.p, kBf16, 2, elems);
  const Tensor ti = tensor1d(idx.p, DType::kInt64, 2);
  KOPS_EXPECT_OK(kops::kv::gather_pages(to, d, ti, true, ctx), "gather_pages token");
  const auto got = kops_test::copy_out(out.p, 2 * elems);
  for (int it = 0; it < 2; ++it) {
    const int64_t slot = it == 0 ? 0 : 5;
    const int64_t off = slot % kPage;
    const int64_t page = slot / kPage;
    for (int i = 0; i < elems; ++i) {
      const int64_t src = (page * kPage + off) * elems + i;
      KOPS_TEST_CHECK(got[it * elems + i] == ch[static_cast<size_t>(src)]);
    }
  }

  // by-page gather
  DevBuf<__nv_bfloat16> page_out(kPage * elems);
  DevBuf<int64_t> pidx(1);
  if (!page_out.ok() || !pidx.ok()) return;
  kops_test::copy_in(pidx.p, std::vector<int64_t>{2});
  const Tensor tpo = tensor2d(page_out.p, kBf16, 1, kPage * elems);
  const Tensor tpi = tensor1d(pidx.p, DType::kInt64, 1);
  KOPS_EXPECT_OK(kops::kv::gather_pages(tpo, d, tpi, false, ctx), "gather_pages page");
  const auto gotp = kops_test::copy_out(page_out.p, kPage * elems);
  for (int i = 0; i < kPage * elems; ++i) {
    KOPS_TEST_CHECK(gotp[static_cast<size_t>(i)] == ch[static_cast<size_t>(2 * kPage * elems + i)]);
  }

  // scatter two modified rows to slots 6 and 7
  DevBuf<__nv_bfloat16> src(2 * elems);
  DevBuf<int64_t> sidx(2);
  if (!src.ok() || !sidx.ok()) return;
  std::vector<float> sf(2 * elems);
  fill_uniform(sf, 82);
  auto sh = cast_from_float<__nv_bfloat16>(sf);
  kops_test::copy_in(src.p, sh);
  kops_test::copy_in(sidx.p, std::vector<int64_t>{6, 7});
  const Tensor tsrc = tensor2d(src.p, kBf16, 2, elems);
  const Tensor tsi = tensor1d(sidx.p, DType::kInt64, 2);
  KOPS_EXPECT_OK(kops::kv::scatter_pages(d, tsrc, tsi, true, ctx), "scatter_pages");
  const auto after = kops_test::copy_out(cache.p, kPages * kPage * elems);
  for (int it = 0; it < 2; ++it) {
    const int64_t slot = it == 0 ? 6 : 7;
    const int64_t page = slot / kPage, off = slot % kPage;
    for (int i = 0; i < elems; ++i) {
      KOPS_TEST_CHECK(after[static_cast<size_t>((page * kPage + off) * elems + i)] ==
                      sh[static_cast<size_t>(it * elems + i)]);
    }
  }

  // HND gather uses the descriptor strides, not the NHD flattening.
  DevBuf<__nv_bfloat16> hcache(kPages * kPage * elems);
  if (!hcache.ok()) return;
  kops_test::copy_in(hcache.p, ch);
  PagedKVDesc hd = make_desc(KVLayout::kHND, hcache.p, nullptr, kHeads, kD, kD, kPage, kPages, kBf16);
  DevBuf<int64_t> hidx(1);
  DevBuf<__nv_bfloat16> hout(elems);
  if (!hidx.ok() || !hout.ok()) return;
  kops_test::copy_in(hidx.p, std::vector<int64_t>{5});
  const Tensor tho = tensor2d(hout.p, kBf16, 1, elems);
  const Tensor thi = tensor1d(hidx.p, DType::kInt64, 1);
  KOPS_EXPECT_OK(kops::kv::gather_pages(tho, hd, thi, true, ctx), "gather_pages hnd");
  const auto hgot = kops_test::copy_out(hout.p, elems);
  const int64_t hpage = 1, hoff = 1;
  for (int h = 0; h < kHeads; ++h) {
    for (int dd = 0; dd < kD; ++dd) {
      const int64_t src = hnd_off(hpage, hoff, h, dd, kHeads, kD, kPage);
      KOPS_TEST_CHECK(hgot[static_cast<size_t>(h * kD + dd)] == ch[static_cast<size_t>(src)]);
    }
  }
}

void test_cp_gather(LaunchContext& ctx) {
  constexpr int kPages = 8, kBlock = 4, kEntry = 32;
  DevBuf<uint8_t> cache(kPages * kBlock * kEntry);
  DevBuf<uint8_t> dst(10 * kEntry);
  DevBuf<int32_t> table(4), cu(3), starts(2);
  if (!cache.ok() || !dst.ok() || !table.ok() || !cu.ok() || !starts.ok()) return;
  std::vector<uint8_t> ch(static_cast<size_t>(kPages * kBlock * kEntry));
  for (size_t i = 0; i < ch.size(); ++i) ch[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFFu);
  kops_test::copy_in(cache.p, ch);
  kops_test::copy_in(table.p, std::vector<int32_t>{0, 1, 3, 2});
  kops_test::copy_in(cu.p, std::vector<int32_t>{0, 5, 8});
  kops_test::copy_in(starts.p, std::vector<int32_t>{0, 2});
  cudaMemset(dst.p, 0xEE, 10 * kEntry);
  const Tensor td = tensor2d(dst.p, DType::kUInt8, 10, kEntry);
  const Tensor tc = tensor3d(cache.p, DType::kUInt8, kPages, kBlock, kEntry);
  kops::kv::GatherSpec spec{};
  spec.block_table = table.p;
  spec.block_table_stride = 2;
  spec.cu_seq_lens = cu.p;
  spec.seq_starts = starts.p;
  spec.num_reqs = 2;
  spec.num_tokens = 10;
  KOPS_EXPECT_OK(kops::kv::cp_gather_cache(td, tc, spec, ctx), "cp_gather_cache");
  const auto got = kops_test::copy_out(dst.p, 10 * kEntry);
  const int32_t tab[4] = {0, 1, 3, 2};
  const int32_t cus[3] = {0, 5, 8};
  const int32_t st[2] = {0, 2};
  for (int token = 0; token < 8; ++token) {
    int r = 0;
    while (cus[r + 1] <= token) ++r;
    const int64_t pos = st[r] + (token - cus[r]);
    const int64_t logical = pos / kBlock;
    const int64_t page = tab[r * 2 + logical];
    const int64_t off = pos % kBlock;
    for (int i = 0; i < kEntry; ++i) {
      const size_t src = static_cast<size_t>((page * kBlock + off) * kEntry + i);
      KOPS_TEST_CHECK(got[static_cast<size_t>(token * kEntry + i)] == ch[src]);
    }
  }
  for (int token = 8; token < 10; ++token) {
    for (int i = 0; i < kEntry; ++i) {
      KOPS_TEST_CHECK(got[static_cast<size_t>(token * kEntry + i)] == 0xEE);
    }
  }
}

void test_cp_gather_page_sizes(LaunchContext& ctx) {
  const int sizes[] = {16, 32, 64, 128};
  constexpr int kEntry = 64;
  for (int block : sizes) {
    const int pages = 4;
    const int tokens = 2 * block;
    DevBuf<uint8_t> cache(static_cast<int64_t>(pages) * block * kEntry);
    DevBuf<uint8_t> dst(static_cast<int64_t>(tokens) * kEntry);
    DevBuf<int32_t> table(4), cu(3), starts(2);
    if (!cache.ok() || !dst.ok() || !table.ok() || !cu.ok() || !starts.ok()) return;
    std::vector<uint8_t> ch(static_cast<size_t>(pages) * block * kEntry);
    for (size_t i = 0; i < ch.size(); ++i) ch[i] = static_cast<uint8_t>((i * 13 + 7) & 0xFFu);
    kops_test::copy_in(cache.p, ch);
    kops_test::copy_in(table.p, std::vector<int32_t>{0, 1, 2, 3});
    kops_test::copy_in(cu.p, std::vector<int32_t>{0, block, tokens});
    kops_test::copy_in(starts.p, std::vector<int32_t>{0, 0});
    const Tensor td = tensor2d(dst.p, DType::kUInt8, tokens, kEntry);
    const Tensor tc = tensor3d(cache.p, DType::kUInt8, pages, block, kEntry);
    kops::kv::GatherSpec spec{};
    spec.block_table = table.p;
    spec.block_table_stride = 2;
    spec.cu_seq_lens = cu.p;
    spec.seq_starts = starts.p;
    spec.num_reqs = 2;
    spec.num_tokens = tokens;
    KOPS_EXPECT_OK(kops::kv::cp_gather_cache(td, tc, spec, ctx), "cp_gather page sweep");
    const auto got = kops_test::copy_out(dst.p, static_cast<int64_t>(tokens) * kEntry);
    const int32_t tab[4] = {0, 1, 2, 3};
    for (int token = 0; token < tokens; ++token) {
      const int r = token < block ? 0 : 1;
      const int pos = token - (r == 0 ? 0 : block);
      const int64_t logical = static_cast<int64_t>(pos) / block;
      const int64_t page = tab[r * 2 + logical];
      const int64_t off = pos % block;
      for (int i = 0; i < kEntry; ++i) {
        const size_t src = static_cast<size_t>((page * block + off) * kEntry + i);
        KOPS_TEST_CHECK(got[static_cast<size_t>(token * kEntry + i)] == ch[src]);
      }
    }
  }
}

void test_gather_dequant(LaunchContext& ctx) {
  constexpr int kPages = 4, kBlock = 3, kEntry = 576, kTokens = 5;
  DevBuf<__nv_bfloat16> cache(kPages * kBlock * kEntry);
  DevBuf<__nv_bfloat16> dst(kTokens * kEntry);
  DevBuf<int32_t> table(2), cu(2), starts(1);
  if (!cache.ok() || !dst.ok() || !table.ok() || !cu.ok() || !starts.ok()) return;
  std::vector<float> cf(kPages * kBlock * kEntry);
  fill_uniform(cf, 91, -3.0f, 3.0f);
  auto ch = cast_from_float<__nv_bfloat16>(cf);
  kops_test::copy_in(cache.p, ch);
  kops_test::copy_in(table.p, std::vector<int32_t>{0, 1});
  kops_test::copy_in(cu.p, std::vector<int32_t>{0, 5});
  kops_test::copy_in(starts.p, std::vector<int32_t>{0});
  PagedKVDesc d = make_desc(KVLayout::kLatent, cache.p, nullptr, 1, kEntry, kEntry - 64, kBlock,
                            kPages, kBf16);
  const Tensor td = tensor2d(dst.p, kBf16, kTokens, kEntry);
  kops::kv::GatherSpec spec{};
  spec.block_table = table.p;
  spec.block_table_stride = 2;
  spec.cu_seq_lens = cu.p;
  spec.seq_starts = starts.p;
  spec.num_reqs = 1;
  spec.num_tokens = kTokens;
  KOPS_EXPECT_OK(kops::kv::gather_and_maybe_dequant_cache(td, d, spec, ctx), "gather_dequant");
  const auto got = kops_test::copy_out(dst.p, kTokens * kEntry);
  const int32_t tab[2] = {0, 1};
  for (int t = 0; t < kTokens; ++t) {
    const int64_t page = tab[t / kBlock == 0 ? 0 : 1];
    const int64_t off = t % kBlock;
    for (int i = 0; i < kEntry; ++i) {
      const size_t src = static_cast<size_t>((page * kBlock + off) * kEntry + i);
      KOPS_TEST_CHECK(got[static_cast<size_t>(t * kEntry + i)] == ch[src]);
    }
  }
}

void test_convert_cache(LaunchContext& ctx) {
  constexpr int kBlocks = 2, kStride = 32;
  DevBuf<__nv_bfloat16> src(kBlocks * kStride);
  DevBuf<float> dst(kBlocks * kStride);
  DevBuf<uint8_t> fp8(kBlocks * kStride);
  DevBuf<float> back(kBlocks * kStride);
  if (!src.ok() || !dst.ok() || !fp8.ok() || !back.ok()) return;
  std::vector<float> vf(kBlocks * kStride);
  fill_uniform(vf, 101, -4.0f, 4.0f);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(src.p, vh);
  const Tensor ts = tensor2d(src.p, kBf16, kBlocks, kStride);
  const Tensor td = tensor2d(dst.p, DType::kFloat32, kBlocks, kStride);
  KOPS_EXPECT_OK(kops::kv::convert_cache(td, ts, 2.0f, KVCodec::kNative, ctx),
                 "convert_cache bf16->f32");
  const auto got = kops_test::copy_out(dst.p, kBlocks * kStride);
  for (size_t i = 0; i < got.size(); ++i) {
    KOPS_TEST_CHECK(std::fabs(got[i] - 2.0f * static_cast<double>(kops_test::to_float(vh[i]))) < 1e-5);
  }

  const Tensor tf = tensor2d(fp8.p, DType::kFloat8E4M3, kBlocks, kStride);
  KOPS_EXPECT_OK(kops::kv::convert_cache(tf, ts, 0.5f, KVCodec::kFP8PerTensor, ctx),
                 "convert_cache bf16->fp8");
  const Tensor tb = tensor2d(back.p, DType::kFloat32, kBlocks, kStride);
  KOPS_EXPECT_OK(kops::kv::convert_cache(tb, tf, 0.5f, KVCodec::kFP8PerTensor, ctx),
                 "convert_cache fp8->f32");
  const auto gotb = kops_test::copy_out(back.p, kBlocks * kStride);
  for (size_t i = 0; i < gotb.size(); ++i) {
    KOPS_TEST_CHECK(std::fabs(gotb[i] - static_cast<double>(kops_test::to_float(vh[i]))) < 0.30);
  }
}

void test_relayout(LaunchContext& ctx) {
  constexpr int kHeads = 2, kD = 8, kPage = 4;
  constexpr int64_t kElems = static_cast<int64_t>(kHeads) * kD;
  DevBuf<__nv_bfloat16> src(kPage * kElems);
  DevBuf<__nv_bfloat16> dst(kPage * kElems);
  DevBuf<int32_t> ind(2), dstind(2);
  if (!src.ok() || !dst.ok() || !ind.ok() || !dstind.ok()) return;
  std::vector<float> vf(kPage * kElems);
  fill_uniform(vf, 111);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(src.p, vh);
  kops_test::copy_in(ind.p, std::vector<int32_t>{0, 1});
  kops_test::copy_in(dstind.p, std::vector<int32_t>{0, 1});
  cudaMemset(dst.p, 0, kPage * kElems * sizeof(__nv_bfloat16));
  PagedKVDesc s = make_desc(KVLayout::kNHD, src.p, nullptr, kHeads, kD, kD, kPage, 1, kBf16);
  PagedKVDesc t = make_desc(KVLayout::kHND, dst.p, nullptr, kHeads, kD, kD, kPage, 1, kBf16);
  const Tensor ti = tensor1d(ind.p, DType::kInt32, 2);
  const Tensor tdi = tensor1d(dstind.p, DType::kInt32, 2);
  KOPS_EXPECT_OK(kops::kv::relayout_pages(s, t, ti, tdi, ctx), "relayout nhd->hnd");
  const auto got = kops_test::copy_out(dst.p, kPage * kElems);
  for (int h = 0; h < kHeads; ++h) {
    for (int off = 0; off < kPage; ++off) {
      for (int dd = 0; dd < kD; ++dd) {
        const int64_t si = nhd_off(0, off, h, dd, kHeads, kD, kPage);
        const int64_t di = hnd_off(0, off, h, dd, kHeads, kD, kPage);
        KOPS_TEST_CHECK(got[static_cast<size_t>(di)] == vh[static_cast<size_t>(si)]);
      }
    }
  }

  // Page-size change: src page=4 -> dst page=2.
  DevBuf<__nv_bfloat16> src2(kPage * kElems);
  DevBuf<__nv_bfloat16> dst2(kPage * kElems);
  DevBuf<int32_t> si2(2), di2(2);
  if (!src2.ok() || !dst2.ok() || !si2.ok() || !di2.ok()) return;
  kops_test::copy_in(src2.p, vh);
  kops_test::copy_in(si2.p, std::vector<int32_t>{0, 1});
  kops_test::copy_in(di2.p, std::vector<int32_t>{0, 2});
  cudaMemset(dst2.p, 0, kPage * kElems * sizeof(__nv_bfloat16));
  PagedKVDesc s2 = make_desc(KVLayout::kNHD, src2.p, nullptr, kHeads, kD, kD, 4, 1, kBf16);
  PagedKVDesc d2 = make_desc(KVLayout::kNHD, dst2.p, nullptr, kHeads, kD, kD, 2, 2, kBf16);
  const Tensor tsi = tensor1d(si2.p, DType::kInt32, 2);
  const Tensor tdi2 = tensor1d(di2.p, DType::kInt32, 2);
  KOPS_EXPECT_OK(kops::kv::relayout_pages(s2, d2, tsi, tdi2, ctx), "relayout page size");
  const auto got2 = kops_test::copy_out(dst2.p, kPage * kElems);
  for (int t2 = 0; t2 < kPage; ++t2) {
    for (int i = 0; i < kElems; ++i) {
      const int64_t di = (t2 / 2) * 2 * kElems + (t2 % 2) * kElems + i;
      KOPS_TEST_CHECK(got2[static_cast<size_t>(di)] == vh[static_cast<size_t>(t2 * kElems + i)]);
    }
  }
}

void test_host_copies(LaunchContext& ctx) {
  constexpr int kPages = 4, kBytes = 16;
  DevBuf<uint8_t> src(kPages * kBytes);
  DevBuf<uint8_t> dst(kPages * kBytes);
  if (!src.ok() || !dst.ok()) return;
  std::vector<uint8_t> sv(static_cast<size_t>(kPages * kBytes));
  for (size_t i = 0; i < sv.size(); ++i) sv[i] = static_cast<uint8_t>(i);
  kops_test::copy_in(src.p, sv);
  cudaMemset(dst.p, 0, kPages * kBytes);
  std::vector<int64_t> map = {0, 2, 1, 3};
  Tensor tm{};
  tm.data = map.data();
  tm.dtype = DType::kInt64;
  tm.ndim = 2;
  tm.device = -1;
  tm.shape[0] = 2;
  tm.shape[1] = 2;
  tm.stride[0] = 2;
  tm.stride[1] = 1;
  const Tensor ts = tensor1d(src.p, DType::kUInt8, kPages * kBytes);
  const Tensor td = tensor1d(dst.p, DType::kUInt8, kPages * kBytes);
  KOPS_EXPECT_OK(kops::kv::swap_pages(ts, td, kBytes, tm, ctx), "swap_pages");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dst.p, kPages * kBytes);
  KOPS_TEST_CHECK(std::memcmp(&got[2 * kBytes], &sv[0], kBytes) == 0);
  KOPS_TEST_CHECK(std::memcmp(&got[3 * kBytes], &sv[kBytes], kBytes) == 0);

  // copy_batch_async: fallback (legacy stream) and batch (explicit stream).
  cudaMemset(dst.p, 0, kPages * kBytes);
  int64_t sp[2] = {reinterpret_cast<int64_t>(src.p),
                   reinterpret_cast<int64_t>(src.p + kBytes)};
  int64_t dp[2] = {reinterpret_cast<int64_t>(dst.p),
                   reinterpret_cast<int64_t>(dst.p + 2 * kBytes)};
  int64_t sz[2] = {kBytes, kBytes};
  kops::kv::BatchCopyDesc desc{};
  desc.src_ptrs_host = sp;
  desc.dst_ptrs_host = dp;
  desc.sizes_host = sz;
  desc.count = 2;
  LaunchContext no_stream = ctx;
  no_stream.stream = nullptr;
  KOPS_EXPECT_OK(kops::kv::copy_batch_async(desc, no_stream), "copy_batch_async fallback");
  cudaDeviceSynchronize();
  auto gotb = kops_test::copy_out(dst.p, kPages * kBytes);
  KOPS_TEST_CHECK(std::memcmp(&gotb[0], &sv[0], kBytes) == 0);
  KOPS_TEST_CHECK(std::memcmp(&gotb[2 * kBytes], &sv[kBytes], kBytes) == 0);

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) == cudaSuccess) {
    cudaMemset(dst.p, 0, kPages * kBytes);
    LaunchContext with_stream = ctx;
    with_stream.stream = reinterpret_cast<kops::kops_stream_t>(stream);
    KOPS_EXPECT_OK(kops::kv::copy_batch_async(desc, with_stream), "copy_batch_async batch");
    cudaStreamSynchronize(stream);
    gotb = kops_test::copy_out(dst.p, kPages * kBytes);
    KOPS_TEST_CHECK(std::memcmp(&gotb[0], &sv[0], kBytes) == 0);
    KOPS_TEST_CHECK(std::memcmp(&gotb[2 * kBytes], &sv[kBytes], kBytes) == 0);

    // Chunked submission and the "source may be read out of stream order" flag.
    cudaMemset(dst.p, 0, kPages * kBytes);
    kops::kv::BatchCopyDesc chunked = desc;
    chunked.max_descriptors_per_call = 1;
    chunked.src_access_order_any = true;
    KOPS_EXPECT_OK(kops::kv::copy_batch_async(chunked, with_stream), "copy_batch_async chunked");
    cudaStreamSynchronize(stream);
    gotb = kops_test::copy_out(dst.p, kPages * kBytes);
    KOPS_TEST_CHECK(std::memcmp(&gotb[0], &sv[0], kBytes) == 0);
    KOPS_TEST_CHECK(std::memcmp(&gotb[2 * kBytes], &sv[kBytes], kBytes) == 0);
    cudaStreamDestroy(stream);
  }
}

void test_graph_replay(LaunchContext& ctx) {
  constexpr int kPages = 2, kPage = 4, kHeads = 2, kD = 8;
  DevBuf<__nv_bfloat16> key(1 * kHeads * kD);
  DevBuf<__nv_bfloat16> val(1 * kHeads * kD);
  DevBuf<__nv_bfloat16> kc(kPages * kPage * kHeads * kD);
  DevBuf<__nv_bfloat16> vc(kPages * kPage * kHeads * kD);
  DevBuf<int64_t> slots(1);
  if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !slots.ok()) return;
  std::vector<float> kf(kHeads * kD, 1.5f), vf(kHeads * kD, -2.5f);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  auto vh = cast_from_float<__nv_bfloat16>(vf);
  kops_test::copy_in(key.p, kh);
  kops_test::copy_in(val.p, vh);
  kops_test::copy_in(slots.p, std::vector<int64_t>{3});
  cudaMemset(kc.p, 0, kPages * kPage * kHeads * kD * sizeof(__nv_bfloat16));
  cudaMemset(vc.p, 0, kPages * kPage * kHeads * kD * sizeof(__nv_bfloat16));
  PagedKVDesc d = make_desc(KVLayout::kNHD, kc.p, vc.p, kHeads, kD, kD, kPage, kPages, kBf16);
  const Tensor tk = tensor3d(key.p, kBf16, 1, kHeads, kD);
  const Tensor tv = tensor3d(val.p, kBf16, 1, kHeads, kD);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, 1);

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return;
  LaunchContext gc = ctx;
  gc.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, d, ts, gc), "graph reshape");
  cudaStreamEndCapture(stream, &graph);
  if (graph != nullptr && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    auto got = kops_test::copy_out(kc.p, kPages * kPage * kHeads * kD);
    KOPS_TEST_CHECK(got[static_cast<size_t>(3 * kHeads * kD)] == kh[0]);
    // Change the source between replays.
    std::vector<float> kf2(kHeads * kD, 4.0f);
    auto kh2 = cast_from_float<__nv_bfloat16>(kf2);
    cudaMemcpy(key.p, kh2.data(), kh2.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    got = kops_test::copy_out(kc.p, kPages * kPage * kHeads * kD);
    KOPS_TEST_CHECK(got[static_cast<size_t>(3 * kHeads * kD)] == kh2[0]);
    cudaGraphExecDestroy(exec);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

// -----------------------------------------------------------------------------
// Contracts
// -----------------------------------------------------------------------------

void test_contracts(LaunchContext& ctx) {
  constexpr int kPage = 2, kHeads = 1, kD = 8;
  DevBuf<__nv_bfloat16> k(kPage * kHeads * kD), v(kPage * kHeads * kD);
  DevBuf<int64_t> slots(1);
  if (!k.ok() || !v.ok() || !slots.ok()) return;
  kops_test::copy_in(slots.p, std::vector<int64_t>{0});
  PagedKVDesc d = make_desc(KVLayout::kNHD, k.p, v.p, kHeads, kD, kD, kPage, 1, kBf16);
  const Tensor tk = tensor3d(k.p, kBf16, 1, kHeads, kD);
  const Tensor tv = tensor3d(v.p, kBf16, 1, kHeads, kD);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, 1);

  PagedKVDesc bad = d;
  bad.k_data = nullptr;
  KOPS_EXPECT_STATUS(kops::kv::kv_desc_validate(bad), Status::kInvalidArgument, "desc null data");
  bad = d;
  bad.page_size = 0;
  KOPS_EXPECT_STATUS(kops::kv::kv_desc_validate(bad), Status::kInvalidArgument, "desc page size");
  bad = d;
  bad.k_codec = KVCodec::kFP8PerTensor;
  bad.k_dtype = kBf16;
  KOPS_EXPECT_STATUS(kops::kv::kv_desc_validate(bad), Status::kUnsupportedDtype, "desc fp8 dtype");
  bad = d;
  bad.entry_bytes = 656;
  bad.stride_unit = kops::StrideUnit::kElement;
  KOPS_EXPECT_STATUS(kops::kv::kv_desc_validate(bad), Status::kInvalidArgument, "desc stride unit");
  bad = d;
  bad.table_format = PageTableFormat::kCSR;
  bad.kv_indptr = nullptr;
  KOPS_EXPECT_STATUS(kops::kv::kv_desc_validate(bad), Status::kInvalidArgument, "desc csr");
  bad = d;
  bad.layout = KVLayout::kLatent;
  bad.v_data = v.p;
  KOPS_EXPECT_STATUS(kops::kv::kv_desc_validate(bad), Status::kInvalidArgument, "desc latent v");
  bad = d;
  bad.k_codec = KVCodec::kMLAFp8DS;
  bad.entry_bytes = 512;
  bad.stride_unit = kops::StrideUnit::kByte;
  bad.head_dim_k = 576;
  KOPS_EXPECT_STATUS(kops::kv::kv_desc_validate(bad), Status::kUnsupportedShape, "desc ds_mla size");

  PagedKVDesc mixed = d;
  mixed.v_codec = KVCodec::kFP8PerTensor;
  KOPS_EXPECT_STATUS(kops::kv::reshape_and_cache(tk, tv, mixed, ts, ctx), Status::kUnsupportedShape,
                     "reshape mixed K/V codec");

  KOPS_EXPECT_STATUS(kops::kv::reshape_and_cache(Tensor{}, tv, d, ts, ctx),
                     Status::kInvalidArgument, "reshape null key");
  KOPS_EXPECT_STATUS(kops::kv::reshape_and_cache(tk, tv, d, tensor1d(slots.p, DType::kInt32, 1), ctx),
                     Status::kUnsupportedDtype, "reshape slot dtype");
  PagedKVDesc fp8 = d;
  fp8.k_codec = KVCodec::kFP8PerTensor;
  fp8.k_dtype = DType::kFloat8E4M3;
  fp8.v_codec = KVCodec::kFP8PerTensor;
  fp8.v_dtype = DType::kFloat8E4M3;
  KOPS_EXPECT_STATUS(kops::kv::reshape_and_cache(tk, tv, fp8, ts, ctx), Status::kInvalidArgument,
                     "reshape fp8 no scale");
  fp8.k_scale = reinterpret_cast<const float*>(k.p);
  fp8.v_scale = reinterpret_cast<const float*>(v.p);
  fp8.k_codec = KVCodec::kInt4PerTokenHead;
  fp8.v_data = nullptr;
  fp8.k_token_scales = k.p;
  KOPS_EXPECT_STATUS(kops::kv::reshape_and_cache(tk, tv, fp8, ts, ctx), Status::kUnsupportedShape,
                     "reshape mixed per-token K/V");

  kops::kv::GatherSpec spec{};
  const Tensor td = tensor2d(k.p, kBf16, 1, kD);
  KOPS_EXPECT_STATUS(kops::kv::cp_gather_cache(td, tensor3d(k.p, kBf16, 1, kPage, kD), spec, ctx),
                     Status::kInvalidArgument, "cp_gather empty spec");
  KOPS_EXPECT_STATUS(kops::kv::convert_cache(td, td, 0.0f, KVCodec::kNative, ctx),
                     Status::kInvalidArgument, "convert zero scale");
  PagedKVDesc hnd = d;
  hnd.layout = KVLayout::kHND;
  KOPS_EXPECT_STATUS(kops::kv::relayout_pages(d, hnd, tensor1d(slots.p, DType::kInt32, 1),
                                              tensor1d(slots.p, DType::kInt32, 2), ctx),
                     Status::kInvalidArgument, "relayout batch mismatch");
}

// -----------------------------------------------------------------------------
// B3: per-token-head codecs, ds_mla, indexer, requantize, scale update, fusions
// -----------------------------------------------------------------------------

uint8_t q8_byte(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

float q8_value(uint8_t b) {
  return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

void test_per_token_head_codecs(LaunchContext& ctx) {
  constexpr int kPages = 3, kPage = 4, kHeads = 2, kD = 64, kT = 5;
  const int64_t elems = static_cast<int64_t>(kHeads) * kD;
  for (int codec = 0; codec < 3; ++codec) {    const kops::kv::KVCodec c = codec == 0 ? KVCodec::kFP8PerTokenHead
                                           : (codec == 1 ? KVCodec::kInt8PerTokenHead
                                                         : KVCodec::kInt4PerTokenHead);
    const int64_t row_bytes = codec == 2 ? elems / 2 : elems;
    DevBuf<__nv_bfloat16> key(static_cast<size_t>(kT) * elems);
    DevBuf<__nv_bfloat16> val(static_cast<size_t>(kT) * elems);
    DevBuf<uint8_t> kc(static_cast<size_t>(kPages) * kPage * row_bytes);
    DevBuf<uint8_t> vc(static_cast<size_t>(kPages) * kPage * row_bytes);
    DevBuf<float> ks(static_cast<size_t>(kPages) * kPage * kHeads * (codec == 2 ? 2 : 1));
    DevBuf<float> vs(static_cast<size_t>(kPages) * kPage * kHeads * (codec == 2 ? 2 : 1));
    DevBuf<int64_t> slots(kT);
    if (!key.ok() || !val.ok() || !kc.ok() || !vc.ok() || !ks.ok() || !vs.ok() || !slots.ok()) {
      return;
    }
    std::vector<float> kf(static_cast<size_t>(kT) * elems), vf(static_cast<size_t>(kT) * elems);
    fill_uniform(kf, 500 + codec, -3.0f, 3.0f);
    fill_uniform(vf, 510 + codec, -2.0f, 2.0f);
    auto kh = cast_from_float<__nv_bfloat16>(kf);
    auto vh = cast_from_float<__nv_bfloat16>(vf);
    kops_test::copy_in(key.p, kh);
    kops_test::copy_in(val.p, vh);
    kops_test::copy_in(slots.p, std::vector<int64_t>{0, 5, 9, -1, 7});
    cudaMemset(kc.p, 0, static_cast<size_t>(kPages) * kPage * row_bytes);
    cudaMemset(vc.p, 0, static_cast<size_t>(kPages) * kPage * row_bytes);
    PagedKVDesc d = make_desc(KVLayout::kNHD, kc.p, vc.p, kHeads, 0, 0, kPage, kPages, kBf16);
    d.k_codec = c;
    d.v_codec = c;
    d.k_dtype = codec == 0 ? DType::kFloat8E4M3 : DType::kUInt8;
    d.v_dtype = d.k_dtype;
    d.head_dim_k = kD;
    d.head_dim_v = kD;
    d.k_stride_page = static_cast<int64_t>(kPage) * row_bytes;
    d.v_stride_page = d.k_stride_page;
    d.k_stride_token = row_bytes;
    d.v_stride_token = row_bytes;
    d.k_stride_head = codec == 2 ? kD / 2 : kD;
    d.v_stride_head = d.k_stride_head;
    d.k_token_scales = ks.p;
    d.v_token_scales = vs.p;
    const Tensor tk = tensor3d(key.p, kBf16, kT, kHeads, kD);
    const Tensor tv = tensor3d(val.p, kBf16, kT, kHeads, kD);
    const Tensor ts = tensor1d(slots.p, DType::kInt64, kT);
    KOPS_EXPECT_OK(kops::kv::reshape_and_cache(tk, tv, d, ts, ctx), "per-token-head write");
    const auto got_k = kops_test::copy_out(kc.p, static_cast<size_t>(kPages) * kPage * row_bytes);
    const auto got_v = kops_test::copy_out(vc.p, static_cast<size_t>(kPages) * kPage * row_bytes);
    const auto got_ks = kops_test::copy_out(
        ks.p, static_cast<size_t>(kPages) * kPage * kHeads * (codec == 2 ? 2 : 1));
    const auto got_vs = kops_test::copy_out(
        vs.p, static_cast<size_t>(kPages) * kPage * kHeads * (codec == 2 ? 2 : 1));
    const int64_t sh[kT] = {0, 5, 9, -1, 7};
    for (int t = 0; t < kT; ++t) {
      if (sh[t] < 0) continue;
      const int64_t page = sh[t] / kPage, off = sh[t] % kPage;
      const int64_t base = (page * kPage + off) * row_bytes;
      for (int h = 0; h < kHeads; ++h) {
        const int64_t meta_row = sh[t] * kHeads + h;
        std::vector<float> krow(static_cast<size_t>(kD)), vrow(static_cast<size_t>(kD));
        for (int j = 0; j < kD; ++j) {
          krow[static_cast<size_t>(j)] =
              kops_test::to_float(kh[static_cast<size_t>(t) * elems + h * kD + j]);
          vrow[static_cast<size_t>(j)] =
              kops_test::to_float(vh[static_cast<size_t>(t) * elems + h * kD + j]);
        }
        const float* src_k = krow.data();
        const float* src_v = vrow.data();
        if (codec == 0) {
          const float sk = got_ks[static_cast<size_t>(meta_row)];
          const float sv = got_vs[static_cast<size_t>(meta_row)];
          for (int j = 0; j < kD; ++j) {
            KOPS_TEST_CHECK(got_k[static_cast<size_t>(base + h * kD + j)] == q8_byte(src_k[j] / sk));
            KOPS_TEST_CHECK(got_v[static_cast<size_t>(base + h * kD + j)] == q8_byte(src_v[j] / sv));
          }
        } else if (codec == 1) {
          const float sk = got_ks[static_cast<size_t>(meta_row)];
          const float sv = got_vs[static_cast<size_t>(meta_row)];
          for (int j = 0; j < kD; ++j) {
            int q = static_cast<int>(std::nearbyint(src_k[j] / sk));
            q = std::max(-127, std::min(127, q));
            KOPS_TEST_CHECK(got_k[static_cast<size_t>(base + h * kD + j)] ==
                            static_cast<uint8_t>(static_cast<int8_t>(q)));
            q = static_cast<int>(std::nearbyint(src_v[j] / sv));
            q = std::max(-127, std::min(127, q));
            KOPS_TEST_CHECK(got_v[static_cast<size_t>(base + h * kD + j)] ==
                            static_cast<uint8_t>(static_cast<int8_t>(q)));
          }
        } else {
          const float sk = got_ks[static_cast<size_t>(2 * meta_row)];
          const float zk = got_ks[static_cast<size_t>(2 * meta_row + 1)];
          const float sv = got_vs[static_cast<size_t>(2 * meta_row)];
          const float zv = got_vs[static_cast<size_t>(2 * meta_row + 1)];
          for (int j = 0; j < kD; j += 2) {
            const int q0 = std::max(0, std::min(15, static_cast<int>(std::nearbyint(src_k[j] / sk)) + static_cast<int>(zk)));
            const int q1 = std::max(0, std::min(15, static_cast<int>(std::nearbyint(src_k[j + 1] / sk)) + static_cast<int>(zk)));
            const uint8_t byte = got_k[static_cast<size_t>(base + h * (kD / 2) + j / 2)];
            KOPS_TEST_CHECK(byte == static_cast<uint8_t>(q0 | (q1 << 4)));
            const int w0 = std::max(0, std::min(15, static_cast<int>(std::nearbyint(src_v[j] / sv)) + static_cast<int>(zv)));
            const int w1 = std::max(0, std::min(15, static_cast<int>(std::nearbyint(src_v[j + 1] / sv)) + static_cast<int>(zv)));
            const uint8_t vbyte = got_v[static_cast<size_t>(base + h * (kD / 2) + j / 2)];
            KOPS_TEST_CHECK(vbyte == static_cast<uint8_t>(w0 | (w1 << 4)));
          }
        }
      }
    }
    (void)got_ks;
    (void)got_vs;
  }
}

void test_ds_mla(LaunchContext& ctx) {
  constexpr int kPages = 2, kPage = 4, kRank = 512, kPe = 64, kT = 3;
  DevBuf<__nv_bfloat16> kv_c(static_cast<size_t>(kT) * kRank);
  DevBuf<__nv_bfloat16> k_pe(static_cast<size_t>(kT) * kPe);
  DevBuf<uint8_t> cache(static_cast<size_t>(kPages) * kPage * 656);
  DevBuf<int64_t> slots(kT);
  if (!kv_c.ok() || !k_pe.ok() || !cache.ok() || !slots.ok()) return;
  std::vector<float> cf(static_cast<size_t>(kT) * kRank), pf(static_cast<size_t>(kT) * kPe);
  fill_uniform(cf, 601, -2.0f, 2.0f);
  fill_uniform(pf, 602, -1.0f, 1.0f);
  auto ch = cast_from_float<__nv_bfloat16>(cf);
  auto ph = cast_from_float<__nv_bfloat16>(pf);
  kops_test::copy_in(kv_c.p, ch);
  kops_test::copy_in(k_pe.p, ph);
  kops_test::copy_in(slots.p, std::vector<int64_t>{0, 1, 2});
  cudaMemset(cache.p, 0, static_cast<size_t>(kPages) * kPage * 656);
  PagedKVDesc d{};
  d.k_data = cache.p;
  d.k_dtype = DType::kFloat8E4M3;
  d.k_codec = KVCodec::kMLAFp8DS;
  d.layout = KVLayout::kLatent;
  d.page_size = kPage;
  d.num_kv_heads = 1;
  d.head_dim_k = 576;
  d.head_dim_v = kRank;
  d.entry_bytes = 656;
  d.stride_unit = kops::StrideUnit::kByte;
  d.k_stride_page = static_cast<int64_t>(kPage) * 656;
  d.k_stride_token = 656;
  d.k_stride_head = 656;
  d.table_format = PageTableFormat::kSlotMapping;
  const Tensor tc = tensor2d(kv_c.p, kBf16, kT, kRank);
  const Tensor tp = tensor2d(k_pe.p, kBf16, kT, kPe);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, kT);
  KOPS_EXPECT_OK(kops::kv::concat_and_cache_mla(tc, tp, d, ts, ctx), "ds_mla write");
  const auto rec = kops_test::copy_out(cache.p, static_cast<size_t>(kPages) * kPage * 656);
  const int64_t sh[kT] = {0, 1, 2};
  for (int t = 0; t < kT; ++t) {
    const int64_t page = sh[t] / kPage, off = sh[t] % kPage;
    const uint8_t* r = &rec[static_cast<size_t>((page * kPage + off) * 656)];
    for (int tile = 0; tile < 4; ++tile) {
      const float s = *reinterpret_cast<const float*>(r + 512 + 4 * tile);
      float amax = 0.0f;
      for (int j = 0; j < 128; ++j) {
        amax = std::max(amax, std::fabs(kops_test::to_float(ch[static_cast<size_t>(t * kRank + tile * 128 + j)])));
      }
      const float want_s = std::max(amax / 448.0f, 1e-4f);
      const float p2 = std::exp2(std::ceil(std::log2(want_s)));
      KOPS_TEST_CHECK(s == p2);
      for (int j = 0; j < 128; ++j) {
        const float x = kops_test::to_float(ch[static_cast<size_t>(t * kRank + tile * 128 + j)]);
        KOPS_TEST_CHECK(r[tile * 128 + j] == q8_byte(x / s));
      }
    }
    for (int j = 0; j < kPe; ++j) {
      const __nv_bfloat16 got = *reinterpret_cast<const __nv_bfloat16*>(r + 528 + 2 * j);
      const __nv_bfloat16 want = ph[static_cast<size_t>(t * kPe + j)];
      KOPS_TEST_CHECK(*reinterpret_cast<const uint16_t*>(&got) ==
                      *reinterpret_cast<const uint16_t*>(&want));
    }
  }

  // Gather + upconvert to bf16 [T, 576].
  DevBuf<__nv_bfloat16> dst(static_cast<size_t>(kT) * 576);
  DevBuf<int32_t> table(2), cu(2);
  if (!dst.ok() || !table.ok() || !cu.ok()) return;
  kops_test::copy_in(table.p, std::vector<int32_t>{0, 0});
  kops_test::copy_in(cu.p, std::vector<int32_t>{0, kT});
  kops::kv::GatherSpec spec{};
  spec.block_table = table.p;
  spec.block_table_stride = 1;
  spec.cu_seq_lens = cu.p;
  spec.num_reqs = 1;
  spec.num_tokens = kT;
  const Tensor td = tensor2d(dst.p, kBf16, kT, 576);
  KOPS_EXPECT_OK(kops::kv::cp_gather_and_upconvert_ds_mla(td, d, spec, Tensor{}, Tensor{},
                                                          Tensor{}, ctx),
                 "ds_mla gather");
  const auto got = kops_test::copy_out(dst.p, static_cast<size_t>(kT) * 576);
  for (int t = 0; t < kT; ++t) {
    for (int tile = 0; tile < 4; ++tile) {
      const uint8_t* r = &rec[static_cast<size_t>(sh[t]) * 656];
      const float s = *reinterpret_cast<const float*>(r + 512 + 4 * tile);
      for (int j = 0; j < 128; ++j) {
        const float want = q8_value(r[tile * 128 + j]) * s;
        KOPS_TEST_CHECK(std::fabs(kops_test::to_float(got[static_cast<size_t>(t * 576 + tile * 128 + j)]) - want) <=
                        2e-2f + 2e-2f * std::fabs(want));
      }
    }
    for (int j = 0; j < kPe; ++j) {
      KOPS_TEST_CHECK(*reinterpret_cast<const uint16_t*>(&got[static_cast<size_t>(t * 576 + 512 + j)]) ==
                      *reinterpret_cast<const uint16_t*>(&rec[static_cast<size_t>(sh[t]) * 656 + 528 + 2 * j]));
    }
  }
}

void test_indexer_cache(LaunchContext& ctx) {
  constexpr int kPages = 4, kPage = 2, kT = 3, kD = 128, kQb = 128;
  const int groups = kD / kQb;
  const int record = kD + groups * 4;
  DevBuf<__nv_bfloat16> key(static_cast<size_t>(kT) * kD);
  DevBuf<uint8_t> cache(static_cast<size_t>(kPages) * kPage * record);
  DevBuf<int64_t> slots(kT);
  if (!key.ok() || !cache.ok() || !slots.ok()) return;
  std::vector<float> kf(static_cast<size_t>(kT) * kD);
  fill_uniform(kf, 701, -2.0f, 2.0f);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  kops_test::copy_in(key.p, kh);
  kops_test::copy_in(slots.p, std::vector<int64_t>{0, 1, 2});
  cudaMemset(cache.p, 0, static_cast<size_t>(kPages) * kPage * record);
  PagedKVDesc d{};
  d.k_data = cache.p;
  d.k_dtype = DType::kUInt8;
  d.k_codec = KVCodec::kIndexerFP8;
  d.layout = KVLayout::kLatent;
  d.page_size = kPage;
  d.num_kv_heads = 1;
  d.head_dim_k = kD;
  d.entry_bytes = record;
  d.stride_unit = kops::StrideUnit::kByte;
  d.k_stride_page = static_cast<int64_t>(kPage) * record;
  d.k_stride_token = record;
  d.k_stride_head = record;
  d.table_format = PageTableFormat::kSlotMapping;
  const Tensor tk = tensor2d(key.p, kBf16, kT, kD);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, kT);
  KOPS_EXPECT_OK(kops::kv::indexer_k_quant_and_cache(tk, d, ts, kQb, false, ctx), "indexer write");
  const auto rec = kops_test::copy_out(cache.p, static_cast<size_t>(kPages) * kPage * record);
  const int64_t sh[kT] = {0, 1, 2};
  for (int t = 0; t < kT; ++t) {
    const uint8_t* r = &rec[static_cast<size_t>(sh[t]) * record];
    const uint8_t* page_base = &rec[static_cast<size_t>(sh[t] / kPage) * kPage * record];
    const float s = *reinterpret_cast<const float*>(page_base + kPage * kD +
                                                   (sh[t] % kPage) * groups * 4);
    float amax = 0.0f;
    for (int j = 0; j < kD; ++j) {
      amax = std::max(amax, std::fabs(kops_test::to_float(kh[static_cast<size_t>(t * kD + j)])));
    }
    const float want_s = std::max(amax, 1e-4f) / 448.0f;
    KOPS_TEST_CHECK(std::fabs(s - want_s) <= 1e-6f * want_s);
    const uint8_t* keyrow = page_base + (sh[t] % kPage) * kD;
    for (int j = 0; j < kD; ++j) {
      const float x = kops_test::to_float(kh[static_cast<size_t>(t * kD + j)]);
      KOPS_TEST_CHECK(keyrow[j] == q8_byte(x / s));
    }
  }

  DevBuf<uint8_t> dst_k(static_cast<size_t>(kT) * kD);
  DevBuf<float> dst_s(static_cast<size_t>(kT) * groups);
  DevBuf<int32_t> table(2), cu(2);
  if (!dst_k.ok() || !dst_s.ok() || !table.ok() || !cu.ok()) return;
  kops_test::copy_in(table.p, std::vector<int32_t>{0, 1});
  kops_test::copy_in(cu.p, std::vector<int32_t>{0, kT});
  kops::kv::GatherSpec spec{};
  spec.block_table = table.p;
  spec.block_table_stride = 2;
  spec.cu_seq_lens = cu.p;
  spec.num_reqs = 1;
  spec.num_tokens = kT;
  const Tensor tdk = tensor2d(dst_k.p, DType::kUInt8, kT, kD);
  const Tensor tds = tensor2d(dst_s.p, DType::kFloat32, kT, groups);
  KOPS_EXPECT_OK(kops::kv::cp_gather_indexer_k_quant_cache(tdk, tds, d, spec, ctx), "indexer gather");
  const auto gk = kops_test::copy_out(dst_k.p, static_cast<size_t>(kT) * kD);
  const auto gs = kops_test::copy_out(dst_s.p, static_cast<size_t>(kT) * groups);
  for (int t = 0; t < kT; ++t) {
    const uint8_t* r = &rec[static_cast<size_t>(sh[t]) * record];
    const uint8_t* page_base = &rec[static_cast<size_t>(sh[t] / kPage) * kPage * record];
    for (int j = 0; j < kD; ++j) {
      KOPS_TEST_CHECK(gk[static_cast<size_t>(t * kD + j)] == r[j]);
    }
    for (int g = 0; g < groups; ++g) {
      const float want = *reinterpret_cast<const float*>(page_base + kPage * kD +
                                                         (sh[t] % kPage) * groups * 4 + 4 * g);
      KOPS_TEST_CHECK(gs[static_cast<size_t>(t * groups + g)] == want);
    }
  }
}

void test_requantize_and_scale(LaunchContext& ctx) {
  constexpr int kPages = 2, kPage = 4, kHeads = 2, kD = 64, kT = 3;
  const int64_t elems = static_cast<int64_t>(kHeads) * kD;
  DevBuf<__nv_bfloat16> src_k(static_cast<size_t>(kPages) * kPage * elems);
  DevBuf<__nv_bfloat16> src_v(static_cast<size_t>(kPages) * kPage * elems);
  DevBuf<uint8_t> dst_k(static_cast<size_t>(kPages) * kPage * elems);
  DevBuf<uint8_t> dst_v(static_cast<size_t>(kPages) * kPage * elems);
  DevBuf<float> dst_ks(static_cast<size_t>(kPages) * kPage * kHeads);
  DevBuf<float> dst_vs(static_cast<size_t>(kPages) * kPage * kHeads);
  DevBuf<int64_t> pairs(2);
  if (!src_k.ok() || !src_v.ok() || !dst_k.ok() || !dst_v.ok() || !dst_ks.ok() ||
      !dst_vs.ok() || !pairs.ok()) {
    return;
  }
  std::vector<float> kf(static_cast<size_t>(kPages) * kPage * elems);
  fill_uniform(kf, 801, -3.0f, 3.0f);
  auto kh = cast_from_float<__nv_bfloat16>(kf);
  kops_test::copy_in(src_k.p, kh);
  kops_test::copy_in(src_v.p, kh);
  cudaMemset(dst_k.p, 0, static_cast<size_t>(kPages) * kPage * elems);
  cudaMemset(dst_v.p, 0, static_cast<size_t>(kPages) * kPage * elems);
  cudaMemset(dst_ks.p, 0, static_cast<size_t>(kPages) * kPage * kHeads * sizeof(float));
  cudaMemset(dst_vs.p, 0, static_cast<size_t>(kPages) * kPage * kHeads * sizeof(float));
  kops_test::copy_in(pairs.p, std::vector<int64_t>{0, 1});
  PagedKVDesc s = make_desc(KVLayout::kNHD, src_k.p, src_v.p, kHeads, kD, kD, kPage, kPages, kBf16);
  PagedKVDesc t = make_desc(KVLayout::kNHD, dst_k.p, dst_v.p, kHeads, kD, kD, kPage, kPages, kBf16);
  t.k_codec = KVCodec::kInt8PerTokenHead;
  t.v_codec = KVCodec::kInt8PerTokenHead;
  t.k_dtype = DType::kUInt8;
  t.v_dtype = DType::kUInt8;
  t.k_token_scales = dst_ks.p;
  t.v_token_scales = dst_vs.p;
  KOPS_EXPECT_OK(kops::kv::requantize_pages(s, t, tensor2d(pairs.p, DType::kInt64, 1, 2), ctx),
                 "requantize native -> int8");
  const auto gotk = kops_test::copy_out(dst_k.p, static_cast<size_t>(kPages) * kPage * elems);
  const auto gsk = kops_test::copy_out(dst_ks.p, static_cast<size_t>(kPages) * kPage * kHeads);
  for (int h = 0; h < kHeads; ++h) {
    for (int off = 0; off < kPage; ++off) {
      const float sc = gsk[static_cast<size_t>((kPage + off) * kHeads + h)];
      for (int j = 0; j < kD; ++j) {
        const size_t si = static_cast<size_t>(nhd_off(0, off, h, j, kHeads, kD, kPage));
        int want = static_cast<int>(std::nearbyint(kops_test::to_float(kh[si]) / sc));
        want = std::max(-127, std::min(127, want));
        const size_t di = static_cast<size_t>(nhd_off(1, off, h, j, kHeads, kD, kPage));
        KOPS_TEST_CHECK(gotk[di] == static_cast<uint8_t>(static_cast<int8_t>(want)));
      }
    }
  }

  // Running per-head absmax.
  DevBuf<float> ka(kHeads), va(kHeads);
  if (!ka.ok() || !va.ok()) return;
  cudaMemset(ka.p, 0, kHeads * sizeof(float));
  cudaMemset(va.p, 0, kHeads * sizeof(float));
  const Tensor tk = tensor3d(src_k.p, kBf16, kT, kHeads, kD);
  const Tensor tv = tensor3d(src_v.p, kBf16, kT, kHeads, kD);
  KOPS_EXPECT_OK(kops::kv::kv_scale_update(tensor1d(ka.p, DType::kFloat32, kHeads),
                                           tensor1d(va.p, DType::kFloat32, kHeads), tk, tv, ctx),
                 "kv_scale_update");
  const auto ga = kops_test::copy_out(ka.p, kHeads);
  for (int h = 0; h < kHeads; ++h) {
    float want = 0.0f;
    for (int t = 0; t < kT; ++t) {
      for (int j = 0; j < kD; ++j) {
        want = std::max(want, std::fabs(kops_test::to_float(
                                    kh[static_cast<size_t>(nhd_off(t / kPage, t % kPage, h, j,
                                                                   kHeads, kD, kPage))])));
      }
    }
    KOPS_TEST_CHECK(std::fabs(ga[h] - want) <= 1e-6f * want);
  }
}

void test_fused_qkv_append(LaunchContext& ctx) {
  constexpr int kT = 3, kHq = 4, kHkv = 2, kD = 64, kRot = 64, kMaxPos = 8;
  const int width = (kHq + 2 * kHkv) * kD;
  DevBuf<__nv_bfloat16> qkv(static_cast<size_t>(kT) * width);
  DevBuf<__nv_bfloat16> qw(kD), kw(kD);
  DevBuf<float> cos_sin(kMaxPos * kRot);
  constexpr int kPage = 4;
  DevBuf<__nv_bfloat16> kc(static_cast<size_t>(kPage) * kHkv * kD);
  DevBuf<__nv_bfloat16> vc(static_cast<size_t>(kPage) * kHkv * kD);
  DevBuf<int64_t> slots(kT), pos(kT);
  if (!qkv.ok() || !qw.ok() || !kw.ok() || !cos_sin.ok() || !kc.ok() || !vc.ok() ||
      !slots.ok() || !pos.ok()) {
    return;
  }
  std::vector<float> host(static_cast<size_t>(kT) * width);
  fill_uniform(host, 901, -2.0f, 2.0f);
  auto h = cast_from_float<__nv_bfloat16>(host);
  kops_test::copy_in(qkv.p, h);
  kops_test::copy_in(qw.p, std::vector<__nv_bfloat16>(kD, __float2bfloat16_rn(1.0f)));
  kops_test::copy_in(kw.p, std::vector<__nv_bfloat16>(kD, __float2bfloat16_rn(1.0f)));
  kops_test::copy_in(slots.p, std::vector<int64_t>{0, 1, 2});
  kops_test::copy_in(pos.p, std::vector<int64_t>{0, 1, 2});
  std::vector<float> cs(static_cast<size_t>(kMaxPos) * kRot);
  const int half = kRot / 2;
  for (int p = 0; p < kMaxPos; ++p) {
    for (int i = 0; i < half; ++i) {
      const double inv = std::pow(10000.0, -2.0 * i / kRot);
      cs[static_cast<size_t>(p * kRot + i)] = static_cast<float>(std::cos(p * inv));
      cs[static_cast<size_t>(p * kRot + half + i)] = static_cast<float>(std::sin(p * inv));
    }
  }
  kops_test::copy_in(cos_sin.p, cs);
  cudaMemset(kc.p, 0, static_cast<size_t>(kPage) * kHkv * kD * 2);
  cudaMemset(vc.p, 0, static_cast<size_t>(kPage) * kHkv * kD * 2);
  PagedKVDesc cache = make_desc(KVLayout::kNHD, kc.p, vc.p, kHkv, kD, kD, kPage, 1, kBf16);
  kops::kv::FusedQKVAppendParams p{};
  p.qkv = qkv.p;
  p.qkv_token_stride = width;
  p.q_norm_weight = qw.p;
  p.k_norm_weight = kw.p;
  p.norm_eps = 1e-5f;
  p.positions = pos.p;
  p.cos_sin_cache = cos_sin.p;
  p.rot_dim = kRot;
  p.rot_offset = 0;
  p.rope_style = kops::RopeStyle::kNeoX;
  p.cache = cache;
  p.slot_mapping = slots.p;
  p.num_tokens = kT;
  p.num_q_heads = kHq;
  p.num_kv_heads = kHkv;
  p.head_dim = kD;
  KOPS_EXPECT_OK(kops::kv::fused_qkv_norm_rope_append(p, kBf16, ctx), "fused qkv append");
  const auto gq = kops_test::copy_out(qkv.p, static_cast<size_t>(kT) * width);
  const auto gk = kops_test::copy_out(kc.p, static_cast<size_t>(kT) * kHkv * kD);
  const auto gv = kops_test::copy_out(vc.p, static_cast<size_t>(kT) * kHkv * kD);
  for (int t = 0; t < kT; ++t) {
    for (int head = 0; head < kHq + kHkv; ++head) {
      const size_t src_base = static_cast<size_t>(t) * width + head * kD;
      float x[kD];
      double ss = 0.0;
      for (int j = 0; j < kD; ++j) {
        x[j] = kops_test::to_float(h[src_base + j]);
        ss += x[j] * x[j];
      }
      const float rstd = static_cast<float>(1.0 / std::sqrt(ss / kD + 1e-5));
      float ref[kD];
      for (int j = 0; j < kD; ++j) {
        __nv_bfloat16 rounded = __float2bfloat16_rn(x[j] * rstd);
        ref[j] = kops_test::to_float(rounded);
      }
      for (int i = 0; i < half; ++i) {
        const float c = cs[static_cast<size_t>(t * kRot + i)];
        const float s = cs[static_cast<size_t>(t * kRot + half + i)];
        const float a = ref[i], b = ref[i + half];
        ref[i] = a * c - b * s;
        ref[i + half] = b * c + a * s;
      }
      if (head < kHq) {
        for (int j = 0; j < kD; ++j) {
          const float got = kops_test::to_float(gq[src_base + j]);
          KOPS_TEST_CHECK(std::fabs(got - ref[j]) <= 2e-2f + 2e-2f * std::fabs(ref[j]));
        }
      } else {
        const int hh = head - kHq;
        const size_t cache_base = static_cast<size_t>(t) * kHkv * kD + hh * kD;
        for (int j = 0; j < kD; ++j) {
          const float got = kops_test::to_float(gk[cache_base + j]);
          KOPS_TEST_CHECK(std::fabs(got - ref[j]) <= 2e-2f + 2e-2f * std::fabs(ref[j]));
        }
      }
    }
    for (int hh = 0; hh < kHkv; ++hh) {
      const size_t vbase = static_cast<size_t>(t) * kHkv * kD + hh * kD;
      const size_t src_base =
          static_cast<size_t>(t) * width + static_cast<size_t>(kHq + kHkv + hh) * kD;
      for (int j = 0; j < kD; ++j) {
        KOPS_TEST_CHECK(kops_test::to_float(gv[vbase + j]) == kops_test::to_float(h[src_base + j]));
      }
    }
  }
}

void test_mla_rope_fused(LaunchContext& ctx) {
  constexpr int kT = 2, kHq = 2, kRank = 64, kRot = 32, kMaxPos = 4, kPage = 4;
  DevBuf<__nv_bfloat16> q_pe(static_cast<size_t>(kT) * kHq * kRot);
  DevBuf<__nv_bfloat16> k_pe(static_cast<size_t>(kT) * kRot);
  DevBuf<__nv_bfloat16> kv_c(static_cast<size_t>(kT) * kRank);
  DevBuf<__nv_bfloat16> cache(static_cast<size_t>(kPage) * (kRank + kRot));
  DevBuf<float> cos_sin(kMaxPos * kRot);
  DevBuf<int64_t> slots(kT), pos(kT);
  if (!q_pe.ok() || !k_pe.ok() || !kv_c.ok() || !cache.ok() || !cos_sin.ok() || !slots.ok() ||
      !pos.ok()) {
    return;
  }
  std::vector<float> qf(static_cast<size_t>(kT) * kHq * kRot), pf(static_cast<size_t>(kT) * kRot),
      cf(static_cast<size_t>(kT) * kRank);
  fill_uniform(qf, 911);
  fill_uniform(pf, 912);
  fill_uniform(cf, 913);
  auto qh = cast_from_float<__nv_bfloat16>(qf);
  auto ph = cast_from_float<__nv_bfloat16>(pf);
  auto ch = cast_from_float<__nv_bfloat16>(cf);
  kops_test::copy_in(q_pe.p, qh);
  kops_test::copy_in(k_pe.p, ph);
  kops_test::copy_in(kv_c.p, ch);
  kops_test::copy_in(slots.p, std::vector<int64_t>{0, 1});
  kops_test::copy_in(pos.p, std::vector<int64_t>{0, 1});
  std::vector<float> cs(static_cast<size_t>(kMaxPos) * kRot);
  const int half = kRot / 2;
  for (int p = 0; p < kMaxPos; ++p) {
    for (int i = 0; i < half; ++i) {
      const double inv = std::pow(10000.0, -2.0 * i / kRot);
      cs[static_cast<size_t>(p * kRot + i)] = static_cast<float>(std::cos(p * inv));
      cs[static_cast<size_t>(p * kRot + half + i)] = static_cast<float>(std::sin(p * inv));
    }
  }
  kops_test::copy_in(cos_sin.p, cs);
  cudaMemset(cache.p, 0, static_cast<size_t>(kPage) * (kRank + kRot) * 2);
  PagedKVDesc d = make_desc(KVLayout::kNHD, cache.p, nullptr, 1, kRank + kRot, kRank, kPage, 1,
                            kBf16);
  d.layout = KVLayout::kLatent;
  const Tensor tq = tensor3d(q_pe.p, kBf16, kT, kHq, kRot);
  const Tensor tk = tensor2d(k_pe.p, kBf16, kT, kRot);
  const Tensor tc = tensor2d(kv_c.p, kBf16, kT, kRank);
  const Tensor tcs = tensor2d(cos_sin.p, DType::kFloat32, kMaxPos, kRot);
  const Tensor ts = tensor1d(slots.p, DType::kInt64, kT);
  KOPS_EXPECT_OK(kops::kv::concat_and_cache_mla_rope_fused(
                     tensor1d(pos.p, DType::kInt64, kT), tq, tk, tc, tcs,
                     kops::RopeStyle::kNeoX, ts, d, ctx),
                 "mla rope fused");
  const auto gq = kops_test::copy_out(q_pe.p, static_cast<size_t>(kT) * kHq * kRot);
  const auto gk = kops_test::copy_out(k_pe.p, static_cast<size_t>(kT) * kRot);
  const auto gc = kops_test::copy_out(cache.p, static_cast<size_t>(kPage) * (kRank + kRot));
  for (int t = 0; t < kT; ++t) {
    std::vector<float> kq(static_cast<size_t>(kHq) * kRot);
    for (int h = 0; h < kHq; ++h) {
      for (int i = 0; i < half; ++i) {
        const float a = kops_test::to_float(qh[static_cast<size_t>((t * kHq + h) * kRot + i)]);
        const float b =
            kops_test::to_float(qh[static_cast<size_t>((t * kHq + h) * kRot + i + half)]);
        const float c = cs[static_cast<size_t>(t * kRot + i)];
        const float s = cs[static_cast<size_t>(t * kRot + half + i)];
        kq[static_cast<size_t>(h * kRot + i)] = a * c - b * s;
        kq[static_cast<size_t>(h * kRot + i + half)] = b * c + a * s;
      }
    }
    for (int h = 0; h < kHq; ++h) {
      for (int j = 0; j < kRot; ++j) {
        const float got = kops_test::to_float(gq[static_cast<size_t>((t * kHq + h) * kRot + j)]);
        KOPS_TEST_CHECK(std::fabs(got - kq[static_cast<size_t>(h * kRot + j)]) <= 2e-2f);
      }
    }
    for (int i = 0; i < half; ++i) {
      const float a = kops_test::to_float(ph[static_cast<size_t>(t * kRot + i)]);
      const float b = kops_test::to_float(ph[static_cast<size_t>(t * kRot + i + half)]);
      const float c = cs[static_cast<size_t>(t * kRot + i)];
      const float s = cs[static_cast<size_t>(t * kRot + half + i)];
      KOPS_TEST_CHECK(std::fabs(kops_test::to_float(gk[static_cast<size_t>(t * kRot + i)]) -
                                (a * c - b * s)) <= 2e-2f);
      KOPS_TEST_CHECK(std::fabs(kops_test::to_float(gk[static_cast<size_t>(t * kRot + i + half)]) -
                                (b * c + a * s)) <= 2e-2f);
    }
    const size_t base = static_cast<size_t>(t) * (kRank + kRot);
    for (int j = 0; j < kRank; ++j) {
      KOPS_TEST_CHECK(gc[base + j] == ch[static_cast<size_t>(t * kRank + j)]);
    }
    for (int j = 0; j < kRot; ++j) {
      KOPS_TEST_CHECK(kops_test::to_float(gc[base + kRank + j]) ==
                      kops_test::to_float(gk[static_cast<size_t>(t * kRot + j)]));
    }
  }
}

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;

  test_reshape_nhd_bf16(ctx);
  test_reshape_hnd_bf16(ctx);
  test_reshape_legacy_bf16(ctx);
  test_reshape_fp8_roundtrip(ctx);
  test_reshape_sweep(ctx);
  test_reshape_dk_ne_dv(ctx);
  test_append_paged_kv(ctx);
  test_ring_kv(ctx);
  test_concat_mla(ctx);
  test_copy_zero_pages(ctx);
  test_gather_scatter_pages(ctx);
  test_cp_gather(ctx);
  test_cp_gather_page_sizes(ctx);
  test_gather_dequant(ctx);
  test_convert_cache(ctx);
  test_relayout(ctx);
  test_host_copies(ctx);
  test_graph_replay(ctx);
  test_per_token_head_codecs(ctx);
  test_ds_mla(ctx);
  test_indexer_cache(ctx);
  test_requantize_and_scale(ctx);
  test_fused_qkv_append(ctx);
  test_mla_rope_fused(ctx);
  test_contracts(ctx);
  return kops_test::finish("test_kvcache");
}
