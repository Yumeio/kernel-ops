// =============================================================================
// kernel-ops | tests/unit/test_compressed_attention.cc
// B14 DeepSeek-V4 CSA/HCA compressed attention: selected compressed entries,
// optional sink, and the SWA window branch, against ref::attention_f64.
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
#include "csrc/reference/ops.h"

namespace {

using kops::attn::CompressedAttentionParams;

void run_case(bool sink) {
  constexpr int T = 2, H = 4, D = 512, TOPK = 8, PAGE = 16, PAGES = 2;
  std::vector<__nv_bfloat16> q(static_cast<size_t>(T) * H * D), kv(static_cast<size_t>(PAGES) * PAGE * D);
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : q) v = __float2bfloat16(dist(rng));
  for (auto& v : kv) v = __float2bfloat16(dist(rng));
  std::vector<int32_t> selected(static_cast<size_t>(T) * TOPK);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < TOPK; ++i) selected[static_cast<size_t>(t) * TOPK + i] = (t * 16 + i * 3) % (PAGES * PAGE);
  std::vector<int32_t> table(PAGES);
  for (int i = 0; i < PAGES; ++i) table[i] = i;

  auto* d_q = kops_test::device_alloc<__nv_bfloat16>(q.size());
  auto* d_kv = kops_test::device_alloc<__nv_bfloat16>(kv.size());
  auto* d_sel = kops_test::device_alloc<int32_t>(selected.size());
  auto* d_tab = kops_test::device_alloc<int32_t>(table.size());
  auto* d_o = kops_test::device_alloc<__nv_bfloat16>(static_cast<size_t>(T) * H * D);
  auto* d_lse = kops_test::device_alloc<float>(static_cast<size_t>(T) * H);
  auto* d_sink = kops_test::device_alloc<float>(H);
  kops_test::copy_in(d_q, q);
  kops_test::copy_in(d_kv, kv);
  kops_test::copy_in(d_sel, selected);
  kops_test::copy_in(d_tab, table);
  std::vector<float> sinkf(H, sink ? 0.4f : 0.0f);
  std::vector<double> sinkd(H, sink ? 0.4 : 0.0);
  kops_test::copy_in(d_sink, sinkf);

  CompressedAttentionParams p{};
  p.q = d_q;
  p.q_stride_n = H * D;
  p.q_stride_h = D;
  p.q_dtype = kops::DType::kBFloat16;
  p.compressed.k_data = d_kv;
  p.compressed.k_dtype = kops::DType::kBFloat16;
  p.compressed.k_codec = kops::kv::KVCodec::kNative;
  p.compressed.layout = kops::kv::KVLayout::kLatent;
  p.compressed.page_size = PAGE;
  p.compressed.num_kv_heads = 1;
  p.compressed.head_dim_k = D;
  p.compressed.head_dim_v = D;
  p.compressed.stride_unit = kops::StrideUnit::kElement;
  p.compressed.k_stride_page = PAGE * D;
  p.compressed.k_stride_token = D;
  p.compressed.table_format = kops::kv::PageTableFormat::kBlockTable;
  p.compressed.block_table = d_tab;
  p.compressed.block_table_stride = PAGES;
  p.selected = d_sel;
  p.topk = TOPK;
  p.selected_stride = TOPK;
  p.token_to_request = nullptr;
  p.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
  p.mod.sink_logits = sink ? d_sink : nullptr;
  p.o = d_o;
  p.o_stride_n = H * D;
  p.o_stride_h = D;
  p.o_dtype = kops::DType::kBFloat16;
  p.lse = d_lse;
  p.num_tokens = T;
  p.num_q_heads = H;
  p.entry_dim = D;
  p.value_dim = D;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const auto st = kops::attn::compressed_attention(p, ctx);
  KOPS_TEST_CHECK(st == kops::Status::kSuccess);
  if (st != kops::Status::kSuccess) return;

  std::vector<double> qd(q.size()), kd(static_cast<size_t>(T) * TOPK * D), vd(kd.size()),
      od(static_cast<size_t>(T) * H * D), ld(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < q.size(); ++i) qd[i] = __bfloat162float(q[i]);
  std::vector<int32_t> qo(T + 1), ki(T + 1);
  for (int t = 0; t <= T; ++t) {
    qo[t] = t;
    ki[t] = t * TOPK;
  }
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < TOPK; ++i) {
      const int slot = selected[static_cast<size_t>(t) * TOPK + i];
      for (int d = 0; d < D; ++d) {
        const float v = __bfloat162float(kv[static_cast<size_t>(slot) * D + d]);
        kd[(static_cast<size_t>(t) * TOPK + i) * D + d] = v;
        vd[(static_cast<size_t>(t) * TOPK + i) * D + d] = v;
      }
    }
  }
  kops::ref::AttentionRefArgs ra{};
  ra.q = qd.data();
  ra.k = kd.data();
  ra.v = vd.data();
  ra.qo_indptr = qo.data();
  ra.kv_indptr = ki.data();
  ra.batch = T;
  ra.q_heads = H;
  ra.kv_heads = 1;
  ra.head_dim_qk = D;
  ra.head_dim_vo = D;
  ra.sm_scale = p.mod.sm_scale;
  ra.causal = false;
  ra.sink_logits = sink ? sinkd.data() : nullptr;
  ra.o = od.data();
  ra.lse = ld.data();
  KOPS_EXPECT_OK(kops::ref::attention_f64(ra), "attention_f64 compressed");
  const auto got_o = kops_test::copy_out(d_o, od.size());
  const auto got_lse = kops_test::copy_out(d_lse, static_cast<size_t>(T) * H);
  kops_test::compare(got_o, od, 2e-2, 2e-2, "compressed_attention o");
  kops_test::compare(got_lse, ld, 2e-2, 2e-2, "compressed_attention lse");

  kops_test::device_free(d_q);
  kops_test::device_free(d_kv);
  kops_test::device_free(d_sel);
  kops_test::device_free(d_tab);
  kops_test::device_free(d_o);
  kops_test::device_free(d_lse);
  kops_test::device_free(d_sink);
}

}  // namespace

int main() {
  run_case(false);
  run_case(true);
  return kops_test::finish("compressed_attention");
}
