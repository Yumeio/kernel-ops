// =============================================================================
// kernel-ops | tests/unit/test_sparse_mla.cc
// B14 sparse MLA (T1): one/two latent scopes, topk_length, sink, against
// ref::attention_f64 on the gathered visible entries. Native BF16 first; the
// record codecs are exercised by test_kvcache and the handoff goldens.
// =============================================================================
#include "tests/unit/harness.h"

#include <cstdint>
#include <cstdio>
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

using kops::attn::SparseMLAParams;
using kops::attn::SparseMLAScope;

struct ScopeData {
  int topk = 0;
  int pages = 0;
  std::vector<int32_t> indices;  // [T, topk]
  std::vector<int32_t> length;   // [T]
  std::vector<__nv_bfloat16> kv; // [pages, page, 576]
  std::vector<int32_t> table;    // [T, pages_per_row]
};

ScopeData make_scope(int T, int topk, int page, int pages_per_row, unsigned seed) {
  ScopeData s;
  s.topk = topk;
  s.pages = T * pages_per_row;
  s.indices.assign(static_cast<size_t>(T) * topk, -1);
  s.length.assign(T, topk);
  s.table.assign(static_cast<size_t>(T) * pages_per_row, 0);
  s.kv.assign(static_cast<size_t>(s.pages) * page * 576, __float2bfloat16(0.0f));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : s.kv) v = __float2bfloat16(dist(rng));
  for (int t = 0; t < T; ++t) {
    for (int p = 0; p < pages_per_row; ++p) s.table[t * pages_per_row + p] = t * pages_per_row + p;
    for (int i = 0; i < topk; ++i) {
      const int slot = (i * 7 + t * 3) % (pages_per_row * page);
      s.indices[static_cast<size_t>(t) * topk + i] = t * pages_per_row * page + slot;
    }
  }
  // One row has a short topk_length and one explicit -1 hole.
  if (T > 1) {
    s.length[1] = topk / 2;
    s.indices[static_cast<size_t>(1) * topk + 3] = -1;
  }
  return s;
}

kops::kv::PagedKVDesc desc_of(const ScopeData& s, void* kv, int32_t* table, int T, int page,
                              int pages_per_row) {
  kops::kv::PagedKVDesc d{};
  d.k_data = kv;
  d.v_data = nullptr;
  d.k_dtype = kops::DType::kBFloat16;
  d.k_codec = kops::kv::KVCodec::kNative;
  d.layout = kops::kv::KVLayout::kLatent;
  d.page_size = page;
  d.num_kv_heads = 1;
  d.head_dim_k = 576;
  d.head_dim_v = 512;
  d.entry_bytes = 0;
  d.stride_unit = kops::StrideUnit::kElement;
  d.k_stride_page = static_cast<int64_t>(page) * 576;
  d.k_stride_token = 576;
  d.table_format = kops::kv::PageTableFormat::kBlockTable;
  d.block_table = table;
  d.block_table_stride = pages_per_row;
  d.seq_lens = nullptr;
  (void)s;
  (void)T;
  return d;
}

void compare_case(int T, int H, bool two_scopes, bool sink, int topk = 16, bool use_ws = false) {
  constexpr int kHeads = 4;
  const int kTopk = topk;
  constexpr int kPage = 16;
  constexpr int kPagesPerRow = 4;
  H = kHeads;
  const auto a = make_scope(T, kTopk, kPage, kPagesPerRow, 11u);
  const auto b = make_scope(T, kTopk, kPage, kPagesPerRow, 29u);

  auto* d_kv_a = kops_test::device_alloc<__nv_bfloat16>(a.kv.size());
  auto* d_kv_b = kops_test::device_alloc<__nv_bfloat16>(b.kv.size());
  auto* d_tab_a = kops_test::device_alloc<int32_t>(a.table.size());
  auto* d_tab_b = kops_test::device_alloc<int32_t>(b.table.size());
  auto* d_idx_a = kops_test::device_alloc<int32_t>(a.indices.size());
  auto* d_idx_b = kops_test::device_alloc<int32_t>(b.indices.size());
  auto* d_len_a = kops_test::device_alloc<int32_t>(a.length.size());
  auto* d_len_b = kops_test::device_alloc<int32_t>(b.length.size());
  kops_test::copy_in(d_kv_a, a.kv);
  kops_test::copy_in(d_kv_b, b.kv);
  kops_test::copy_in(d_tab_a, a.table);
  kops_test::copy_in(d_tab_b, b.table);
  kops_test::copy_in(d_idx_a, a.indices);
  kops_test::copy_in(d_idx_b, b.indices);
  kops_test::copy_in(d_len_a, a.length);
  kops_test::copy_in(d_len_b, b.length);

  const int64_t qn = static_cast<int64_t>(T) * H * 576;
  const int64_t on = static_cast<int64_t>(T) * H * 512;
  std::vector<float> qf(qn);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : qf) v = dist(rng);
  std::vector<__nv_bfloat16> q(qn);
  for (size_t i = 0; i < q.size(); ++i) q[i] = __float2bfloat16(qf[i]);
  auto* d_q = kops_test::device_alloc<__nv_bfloat16>(qn);
  auto* d_o = kops_test::device_alloc<__nv_bfloat16>(on);
  auto* d_lse = kops_test::device_alloc<float>(static_cast<int64_t>(T) * H);
  kops_test::copy_in(d_q, q);

  std::vector<float> sinkv(H, sink ? 0.3f : 0.0f);
  std::vector<double> sinkd(H, sink ? 0.3 : 0.0);
  auto* d_sink = kops_test::device_alloc<float>(H);
  kops_test::copy_in(d_sink, sinkv);

  SparseMLAParams p{};
  p.q = d_q;
  p.q_stride_n = static_cast<int64_t>(H) * 576;
  p.q_stride_h = 576;
  p.q_dtype = kops::DType::kBFloat16;
  p.primary.kv = desc_of(a, d_kv_a, d_tab_a, T, kPage, kPagesPerRow);
  p.primary.indices = d_idx_a;
  p.primary.topk_length = d_len_a;
  p.primary.topk = kTopk;
  if (two_scopes) {
    p.extra.kv = desc_of(b, d_kv_b, d_tab_b, T, kPage, kPagesPerRow);
    p.extra.indices = d_idx_b;
    p.extra.topk_length = d_len_b;
    p.extra.topk = kTopk;
  }
  p.mod.sm_scale = 1.0f / std::sqrt(576.0f);
  p.mod.sink_logits = sink ? d_sink : nullptr;
  p.o = d_o;
  p.o_stride_n = static_cast<int64_t>(H) * 512;
  p.o_stride_h = 512;
  p.o_dtype = kops::DType::kBFloat16;
  p.lse = d_lse;
  p.num_tokens = T;
  p.num_heads = H;
  p.qk_dim = 576;
  p.v_dim = 512;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  // Split-KV entry: sparse_mla_run_ws must reproduce the plain rows exactly enough for the same
  // fp64 oracle. The split count is a pure host function, so a shape that already fills the
  // partition would silently stay on the plain body - require a real split when asked for the ws.
  const int32_t splits = kops::attn::sparse_mla_split_count(p, ctx);
  const size_t ws_bytes = use_ws ? kops::attn::sparse_mla_workspace_bytes(p, ctx) : 0;
  float* d_ws = nullptr;
  if (use_ws) {
    KOPS_TEST_CHECK(splits > 1);
    KOPS_TEST_CHECK(ws_bytes > 0);
    d_ws = kops_test::device_alloc<float>(ws_bytes / sizeof(float));
    if (d_ws == nullptr) return;
    kops::Workspace small{d_ws, ws_bytes >= 512 ? ws_bytes - 256 : 0};
    KOPS_EXPECT_STATUS(kops::attn::sparse_mla_run_ws(p, small, ctx),
                       kops::Status::kWorkspaceTooSmall, "sparse_mla_run_ws short lease");
  }
  const kops::Workspace lease{d_ws, ws_bytes};
  const auto st = use_ws ? kops::attn::sparse_mla_run_ws(p, lease, ctx)
                         : kops::attn::sparse_mla_run(p, ctx);
  if (st != kops::Status::kSuccess) {
    kops_test::record_failure(use_ws ? "sparse_mla_run_ws" : "sparse_mla_run",
                              kops::last_error_message(), __FILE__, __LINE__);
    return;
  }
  if (use_ws) {
    // Determinism: the split body writes one partial per CTA and merge_states has a fixed order,
    // so a second call must reproduce every byte of o and lse.
    const auto o1 = kops_test::copy_out(d_o, on);
    const auto l1 = kops_test::copy_out(d_lse, static_cast<int64_t>(T) * H);
    const auto st2 = kops::attn::sparse_mla_run_ws(p, lease, ctx);
    KOPS_TEST_CHECK(st2 == kops::Status::kSuccess);
    if (st2 != kops::Status::kSuccess) return;
    const auto o2 = kops_test::copy_out(d_o, on);
    const auto l2 = kops_test::copy_out(d_lse, static_cast<int64_t>(T) * H);
    bool same = o1.size() == o2.size() && l1.size() == l2.size();
    for (size_t i = 0; same && i < o1.size(); ++i) same = std::memcmp(&o1[i], &o2[i], sizeof(o1[i])) == 0;
    for (size_t i = 0; same && i < l1.size(); ++i) same = std::memcmp(&l1[i], &l2[i], sizeof(l1[i])) == 0;
    KOPS_TEST_CHECK(same);
  }

  // Reference: one request per query row, each with its own concatenated key set.
  std::vector<double> qd(qn), kd, vd, od(static_cast<size_t>(on)), ld(static_cast<size_t>(T) * H);
  for (int64_t i = 0; i < qn; ++i) qd[i] = static_cast<double>(__bfloat162float(q[i]));
  std::vector<int32_t> qo(T + 1), kv(T + 1);
  for (int t = 0; t <= T; ++t) qo[t] = t;
  int total = 0;
  for (int t = 0; t < T; ++t) {
    int valid = 0;
    for (int i = 0; i < kTopk; ++i) {
      if (i < a.length[t] && a.indices[static_cast<size_t>(t) * kTopk + i] >= 0) ++valid;
    }
    if (two_scopes) {
      for (int i = 0; i < kTopk; ++i) {
        if (i < b.length[t] && b.indices[static_cast<size_t>(t) * kTopk + i] >= 0) ++valid;
      }
    }
    kv[t + 1] = total + valid;
    total += valid;
  }
  kd.assign(static_cast<size_t>(total) * 576, 0.0);
  vd.assign(static_cast<size_t>(total) * 512, 0.0);
  for (int t = 0; t < T; ++t) {
    int out = kv[t];
    auto append = [&](const ScopeData& s, const int32_t* idx, const int32_t* len,
                      const std::vector<__nv_bfloat16>& kvdata) {
      (void)s;
      for (int i = 0; i < kTopk; ++i) {
        if (i >= len[t] || idx[static_cast<size_t>(t) * kTopk + i] < 0) continue;
        const int slot = idx[static_cast<size_t>(t) * kTopk + i];
        for (int d = 0; d < 576; ++d) kd[static_cast<size_t>(out) * 576 + d] =
            static_cast<double>(__bfloat162float(kvdata[static_cast<size_t>(slot) * 576 + d]));
        for (int d = 0; d < 512; ++d) vd[static_cast<size_t>(out) * 512 + d] = kd[static_cast<size_t>(out) * 576 + d];
        ++out;
      }
    };
    append(a, a.indices.data(), a.length.data(), a.kv);
    if (two_scopes) append(b, b.indices.data(), b.length.data(), b.kv);
  }
  kops::ref::AttentionRefArgs ra{};
  ra.q = qd.data();
  ra.k = kd.data();
  ra.v = vd.data();
  ra.qo_indptr = qo.data();
  ra.kv_indptr = kv.data();
  ra.batch = T;
  ra.q_heads = H;
  ra.kv_heads = 1;
  ra.head_dim_qk = 576;
  ra.head_dim_vo = 512;
  ra.sm_scale = p.mod.sm_scale;
  ra.causal = false;
  ra.sink_logits = sink ? sinkd.data() : nullptr;
  ra.o = od.data();
  ra.lse = ld.data();
  KOPS_EXPECT_OK(kops::ref::attention_f64(ra), "attention_f64 sparse");

  const auto got_o = kops_test::copy_out(d_o, on);
  const auto got_lse = kops_test::copy_out(d_lse, static_cast<int64_t>(T) * H);
  kops_test::compare(got_o, od, 2e-2, 2e-2, "sparse_mla o");
  kops_test::compare(got_lse, ld, 2e-2, 2e-2, "sparse_mla lse");

  kops_test::device_free(d_kv_a);
  kops_test::device_free(d_kv_b);
  kops_test::device_free(d_tab_a);
  kops_test::device_free(d_tab_b);
  kops_test::device_free(d_idx_a);
  kops_test::device_free(d_idx_b);
  kops_test::device_free(d_len_a);
  kops_test::device_free(d_len_b);
  kops_test::device_free(d_q);
  kops_test::device_free(d_o);
  kops_test::device_free(d_lse);
  kops_test::device_free(d_sink);
  kops_test::device_free(d_ws);
}

}  // namespace

int main() {
  compare_case(1, 4, false, false);
  compare_case(2, 4, false, true);
  compare_case(2, 4, true, true);
  // Split-KV entry (sparse_mla_run_ws): a 2048-key row with one or two scopes leaves 1-2 CTAs on
  // a 100+ SM partition, so the visible list is split across grid.z. Covers the sink in split 0
  // only, topk_length truncation (row 1 of a 2-row case) and a split boundary inside the extra
  // scope.
  compare_case(1, 4, false, false, 2048, true);
  compare_case(1, 4, false, true, 2048, true);
  compare_case(2, 4, true, true, 2048, true);
  return kops_test::finish("sparse_mla");
}
