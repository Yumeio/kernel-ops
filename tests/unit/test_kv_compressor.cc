// =============================================================================
// kernel-ops | tests/unit/test_kv_compressor.cc
// B14 DeepSeek-V4 KV compressor: one entry per ratio window, softmax over
// (Z + positional bias), weighted C^a/C^b, optional overlap.
// =============================================================================
#include "tests/unit/harness.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"

namespace {

void run_case(bool overlap) {
  constexpr int T = 8, R = 2, D = 4, PAGE = 16, ENTRIES = 4;
  std::vector<float> caf(T * D), cbf(T * D), zaf(T * D), zbf(T * D), ba(R * D), bb(R * D);
  std::mt19937 rng(31);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : caf) v = dist(rng);
  for (auto& v : cbf) v = dist(rng);
  for (auto& v : zaf) v = dist(rng);
  for (auto& v : zbf) v = dist(rng);
  for (auto& v : ba) v = dist(rng) * 0.1f;
  for (auto& v : bb) v = dist(rng) * 0.1f;
  std::vector<__nv_bfloat16> ca(caf.size()), cb(cbf.size()), za(zaf.size()), zb(zbf.size());
  for (size_t i = 0; i < ca.size(); ++i) {
    ca[i] = __float2bfloat16(caf[i]);
    cb[i] = __float2bfloat16(cbf[i]);
    za[i] = __float2bfloat16(zaf[i]);
    zb[i] = __float2bfloat16(zbf[i]);
  }
  std::vector<int64_t> slots = {0, 1, 2, 3};
  std::vector<int32_t> cu = {0, T};
  std::vector<int32_t> start_pos = {0};
  std::vector<__nv_bfloat16> out(static_cast<size_t>(PAGE) * D, __float2bfloat16(0.0f));

  auto* d_ca = kops_test::device_alloc<__nv_bfloat16>(ca.size());
  auto* d_cb = kops_test::device_alloc<__nv_bfloat16>(cb.size());
  auto* d_za = kops_test::device_alloc<__nv_bfloat16>(za.size());
  auto* d_zb = kops_test::device_alloc<__nv_bfloat16>(zb.size());
  auto* d_ba = kops_test::device_alloc<float>(ba.size());
  auto* d_bb = kops_test::device_alloc<float>(bb.size());
  auto* d_slots = kops_test::device_alloc<int64_t>(slots.size());
  auto* d_cu = kops_test::device_alloc<int32_t>(cu.size());
  auto* d_sp = kops_test::device_alloc<int32_t>(start_pos.size());
  auto* d_out = kops_test::device_alloc<__nv_bfloat16>(out.size());
  auto* d_nout = kops_test::device_alloc<int32_t>(1);
  kops_test::copy_in(d_ca, ca);
  kops_test::copy_in(d_cb, cb);
  kops_test::copy_in(d_za, za);
  kops_test::copy_in(d_zb, zb);
  kops_test::copy_in(d_ba, ba);
  kops_test::copy_in(d_bb, bb);
  kops_test::copy_in(d_slots, slots);
  kops_test::copy_in(d_cu, cu);
  kops_test::copy_in(d_sp, start_pos);
  kops_test::copy_in(d_out, out);

  kops::attn::KVCompressorParams p{};
  p.ca = d_ca;
  p.cb = d_cb;
  p.za = d_za;
  p.zb = d_zb;
  p.token_stride = D;
  p.bias_a = d_ba;
  p.bias_b = d_bb;
  p.ratio_m = R;
  p.entry_dim = D;
  p.num_tokens = T;
  p.max_entries = ENTRIES;
  p.overlap = overlap;
  p.cu_seqlens = d_cu;
  p.seq_start_pos = d_sp;
  p.out_cache.k_data = d_out;
  p.out_cache.k_dtype = kops::DType::kBFloat16;
  p.out_cache.k_codec = kops::kv::KVCodec::kNative;
  p.out_cache.layout = kops::kv::KVLayout::kLatent;
  p.out_cache.page_size = PAGE;
  p.out_cache.num_kv_heads = 1;
  p.out_cache.head_dim_k = D;
  p.out_cache.head_dim_v = D;
  p.out_cache.stride_unit = kops::StrideUnit::kElement;
  p.out_cache.k_stride_page = PAGE * D;
  p.out_cache.k_stride_token = D;
  p.out_slot_mapping = d_slots;
  p.num_entries_out = d_nout;
  p.batch_size = 1;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const auto st = kops::attn::kv_compress(p, kops::DType::kBFloat16, ctx);
  KOPS_TEST_CHECK(st == kops::Status::kSuccess);
  const auto got = kops_test::copy_out(d_out, out.size());
  const auto nout = kops_test::copy_out(d_nout, 1);
  KOPS_TEST_CHECK(nout.size() == 1 && nout[0] == ENTRIES);

  std::vector<double> want(static_cast<size_t>(PAGE) * D, 0.0);
  for (int e = 0; e < ENTRIES; ++e) {
    const int base = e * R;
    for (int c = 0; c < D; ++c) {
      const int cand = overlap ? 2 * R : R;
      double mx = -1e300;
      for (int k = 0; k < cand; ++k) {
        const int t = base + k;
        if (t >= T) continue;
        const bool prev = overlap && k < R;
        mx = std::max(mx, static_cast<double>(prev ? zbf[t * D + c] : zaf[t * D + c]) +
                              static_cast<double>(prev ? bb[(k % R) * D + c]
                                                       : ba[(k % R) * D + c]));
      }
      double den = 0.0, acc = 0.0;
      for (int k = 0; k < cand; ++k) {
        const int t = base + k;
        if (t >= T) continue;
        const bool prev = overlap && k < R;
        const double z = prev ? zbf[t * D + c] : zaf[t * D + c];
        const double b = prev ? bb[(k % R) * D + c] : ba[(k % R) * D + c];
        const double w = std::exp(z + b - mx);
        den += w;
        acc += w * (prev ? cbf[t * D + c] : caf[t * D + c]);
      }
      want[slots[e] * D + c] = den > 0.0 ? acc / den : 0.0;
    }
  }
  kops_test::compare(got, want, 2e-2, 2e-2, "kv_compressor");

  kops_test::device_free(d_ca);
  kops_test::device_free(d_cb);
  kops_test::device_free(d_za);
  kops_test::device_free(d_zb);
  kops_test::device_free(d_ba);
  kops_test::device_free(d_bb);
  kops_test::device_free(d_slots);
  kops_test::device_free(d_cu);
  kops_test::device_free(d_sp);
  kops_test::device_free(d_out);
  kops_test::device_free(d_nout);
}

}  // namespace

int main() {
  run_case(false);
  run_case(true);
  return kops_test::finish("kv_compressor");
}
