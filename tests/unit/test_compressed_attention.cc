// =============================================================================
// kernel-ops | tests/unit/test_compressed_attention.cc
// B14 DeepSeek-V4 CSA/HCA compressed attention: selected compressed entries,
// HCA dense range, optional sink, and the SWA window branch, against
// ref::attention_f64. Covers the head-grouped tensor-core body (16 heads/CTA,
// entry/value dim 512, topk 64..2048, 4K/64K compressed pools, T = 1/16) and
// the fully masked row convention (o = 0, lse = -inf).
// =============================================================================
#include "tests/unit/harness.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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

constexpr int kPage = 64;
struct Cfg {
  const char* name;
  int T, H, pool, topk, window;
  bool sink, hca;
  int masked_row;  // fully masked CSA row, or -1
  int dim;         // entry_dim; != 512 takes the scalar fallback
  int vdim;        // value_dim <= dim (shared-KV output slice)
  int sel_len;     // selected_topk_length (0 = topk)
};

/// Deterministic uniform value in [-1, 1) from a counter (no host RNG cost on
/// the 64K pooled cases).
inline float unit_hash(int64_t i, uint32_t seed) {
  uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  h *= 3266489917u;
  h ^= h >> 16;
  return static_cast<float>(static_cast<double>(h & 0xFFFFFFu) / 8388608.0 - 1.0);
}

inline __nv_bfloat16 bf16_at(int64_t i, uint32_t seed) {
  return __float2bfloat16(unit_hash(i, seed));
}

void run_case(const Cfg& c) {
  const int T = c.T, H = c.H, pool = c.pool, topk = c.topk, window = c.window;
  const int dim = c.dim;
  const int vdim = c.vdim;
  const int sel_len = c.sel_len > 0 ? (c.sel_len < topk ? c.sel_len : topk) : topk;
  const int pool_pages = pool / kPage;
  const int win_pages = window > 0 ? (window + kPage - 1) / kPage : 0;
  const int win_rows = T * win_pages * kPage;
  const int kn = topk + window;

  std::vector<__nv_bfloat16> q(static_cast<size_t>(T) * H * dim);
  std::vector<__nv_bfloat16> comp(static_cast<size_t>(pool) * dim);
  std::vector<__nv_bfloat16> win(window > 0 ? static_cast<size_t>(win_rows) * dim : 0);
  for (size_t i = 0; i < q.size(); ++i) q[i] = bf16_at(static_cast<int64_t>(i), 37u);
  for (size_t i = 0; i < comp.size(); ++i) comp[i] = bf16_at(static_cast<int64_t>(i), 41u);
  for (size_t i = 0; i < win.size(); ++i) win[i] = bf16_at(static_cast<int64_t>(i), 43u);

  std::vector<int32_t> selected(c.hca ? 0 : static_cast<size_t>(T) * topk);
  if (!c.hca) {
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < topk; ++i) {
        const int32_t slot =
            (static_cast<int32_t>(i) * 4099 + t * 7919) % pool;
        selected[static_cast<size_t>(t) * topk + i] =
            (t == c.masked_row || i % 7 == 3) ? -1 : slot;
      }
    }
  }
  std::vector<int32_t> t2r(static_cast<size_t>(T));
  std::vector<int32_t> sel_len_host(static_cast<size_t>(T), sel_len);
  std::vector<int32_t> ncomp(static_cast<size_t>(T), c.hca ? topk : 0);
  std::vector<int64_t> positions(static_cast<size_t>(T), window > 0 ? window - 1 : 0);
  std::vector<int32_t> tab_comp(static_cast<size_t>(T) * pool_pages);
  std::vector<int32_t> tab_win(static_cast<size_t>(T) * (win_pages > 0 ? win_pages : 1), 0);
  for (int t = 0; t < T; ++t) {
    t2r[static_cast<size_t>(t)] = t;
    for (int pg = 0; pg < pool_pages; ++pg) {
      tab_comp[static_cast<size_t>(t) * pool_pages + pg] = pg;
    }
    for (int pg = 0; pg < win_pages; ++pg) {
      tab_win[static_cast<size_t>(t) * win_pages + pg] = t * win_pages + pg;
    }
  }

  auto* d_q = kops_test::device_alloc<__nv_bfloat16>(q.size());
  auto* d_comp = kops_test::device_alloc<__nv_bfloat16>(comp.size());
  auto* d_win = kops_test::device_alloc<__nv_bfloat16>(win.empty() ? 1 : win.size());
  auto* d_sel = kops_test::device_alloc<int32_t>(selected.empty() ? 1 : selected.size());
  auto* d_sel_len = kops_test::device_alloc<int32_t>(static_cast<size_t>(T));
  auto* d_tab_comp = kops_test::device_alloc<int32_t>(tab_comp.size());
  auto* d_tab_win = kops_test::device_alloc<int32_t>(tab_win.size());
  auto* d_t2r = kops_test::device_alloc<int32_t>(t2r.size());
  auto* d_ncomp = kops_test::device_alloc<int32_t>(ncomp.size());
  auto* d_pos = kops_test::device_alloc<int64_t>(positions.size());
  auto* d_o = kops_test::device_alloc<__nv_bfloat16>(static_cast<size_t>(T) * H * vdim);
  auto* d_lse = kops_test::device_alloc<float>(static_cast<size_t>(T) * H);
  auto* d_sink = kops_test::device_alloc<float>(static_cast<size_t>(H));
  kops_test::copy_in(d_q, q);
  kops_test::copy_in(d_comp, comp);
  if (!win.empty()) kops_test::copy_in(d_win, win);
  if (!selected.empty()) kops_test::copy_in(d_sel, selected);
  kops_test::copy_in(d_tab_comp, tab_comp);
  kops_test::copy_in(d_tab_win, tab_win);
  kops_test::copy_in(d_t2r, t2r);
  kops_test::copy_in(d_sel_len, sel_len_host);
  kops_test::copy_in(d_ncomp, ncomp);
  kops_test::copy_in(d_pos, positions);
  std::vector<float> sinkf(H, 0.0f);
  std::vector<double> sinkd(H, 0.0);
  if (c.sink) {
    for (int h = 0; h < H; ++h) {
      sinkf[static_cast<size_t>(h)] = unit_hash(h, 47u);
      sinkd[static_cast<size_t>(h)] = sinkf[static_cast<size_t>(h)];
    }
  }
  kops_test::copy_in(d_sink, sinkf);

  CompressedAttentionParams p{};
  p.q = d_q;
  p.q_stride_n = static_cast<int64_t>(H) * dim;
  p.q_stride_h = dim;
  p.q_dtype = kops::DType::kBFloat16;
  p.compressed.k_data = d_comp;
  p.compressed.k_dtype = kops::DType::kBFloat16;
  p.compressed.k_codec = kops::kv::KVCodec::kNative;
  p.compressed.layout = kops::kv::KVLayout::kLatent;
  p.compressed.page_size = kPage;
  p.compressed.num_kv_heads = 1;
  p.compressed.head_dim_k = dim;
  p.compressed.head_dim_v = dim;
  p.compressed.stride_unit = kops::StrideUnit::kElement;
  p.compressed.k_stride_page = kPage * dim;
  p.compressed.k_stride_token = dim;
  p.compressed.table_format = kops::kv::PageTableFormat::kBlockTable;
  p.compressed.block_table = d_tab_comp;
  p.compressed.block_table_stride = pool_pages;
  p.selected = c.hca ? nullptr : d_sel;
  p.selected_topk_length = (!c.hca && c.sel_len > 0) ? d_sel_len : nullptr;
  p.selected_stride = topk;
  p.topk = topk;
  p.num_compressed = c.hca ? d_ncomp : nullptr;
  if (window > 0) {
    p.window.k_data = d_win;
    p.window.k_dtype = kops::DType::kBFloat16;
    p.window.k_codec = kops::kv::KVCodec::kNative;
    p.window.layout = kops::kv::KVLayout::kLatent;
    p.window.page_size = kPage;
    p.window.num_kv_heads = 1;
    p.window.head_dim_k = dim;
    p.window.head_dim_v = dim;
    p.window.stride_unit = kops::StrideUnit::kElement;
    p.window.k_stride_page = kPage * dim;
    p.window.k_stride_token = dim;
    p.window.table_format = kops::kv::PageTableFormat::kBlockTable;
    p.window.block_table = d_tab_win;
    p.window.block_table_stride = win_pages;
  }
  p.window_size = window;
  p.token_to_request = d_t2r;
  p.positions = d_pos;
  p.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(dim));
  p.mod.sink_logits = c.sink ? d_sink : nullptr;
  p.o = d_o;
  p.o_stride_n = static_cast<int64_t>(H) * vdim;
  p.o_stride_h = vdim;
  p.o_dtype = kops::DType::kBFloat16;
  p.lse = d_lse;
  p.num_tokens = T;
  p.num_q_heads = H;
  p.entry_dim = dim;
  p.value_dim = vdim;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const auto st = kops::attn::compressed_attention(p, ctx);
  KOPS_TEST_CHECK(st == kops::Status::kSuccess);
  if (st != kops::Status::kSuccess) return;

  // Reference: gather this row's visible keys (compressed branch first, then the
  // SWA tail) and clear the custom-mask bits of `-1` holes / the masked row.
  std::vector<double> qd(q.size()), kd(static_cast<size_t>(T) * kn * dim),
      od(static_cast<size_t>(T) * H * vdim), ld(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < q.size(); ++i) qd[i] = __bfloat162float(q[i]);
  std::vector<uint8_t> mask((static_cast<size_t>(T) * kn + 7) / 8, 0xFF);
  auto clear_bit = [&](int64_t bit) {
    mask[static_cast<size_t>(bit >> 3)] &= static_cast<uint8_t>(~(1u << (bit & 7)));
  };
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < topk; ++i) {
      const int32_t slot = c.hca ? static_cast<int32_t>(i) : selected[static_cast<size_t>(t) * topk + i];
      if (slot < 0 || i >= sel_len) {
        clear_bit(static_cast<int64_t>(t) * kn + i);
        continue;
      }
      for (int d = 0; d < dim; ++d) {
        kd[(static_cast<size_t>(t) * kn + i) * dim + d] =
            __bfloat162float(comp[static_cast<size_t>(slot) * dim + d]);
      }
    }
    for (int j = 0; j < window; ++j) {
      const size_t dst = (static_cast<size_t>(t) * kn + topk + j) * dim;
      const size_t src = (static_cast<size_t>(t) * win_pages * kPage + j) * dim;
      for (int d = 0; d < dim; ++d) kd[dst + d] = __bfloat162float(win[src + d]);
    }
    if (t == c.masked_row) {
      for (int j = 0; j < kn; ++j) clear_bit(static_cast<int64_t>(t) * kn + j);
    }
  }
  std::vector<int32_t> qo(T + 1), ki(T + 1);
  std::vector<int64_t> mind(T + 1);
  for (int t = 0; t <= T; ++t) {
    qo[t] = t;
    ki[t] = t * kn;
    mind[t] = t * kn;
  }
  kops::ref::AttentionRefArgs ra{};
  ra.q = qd.data();
  ra.k = kd.data();
  std::vector<double> vd;
  if (vdim != dim) {
    vd.resize(static_cast<size_t>(T) * kn * vdim);
    for (int t = 0; t < T; ++t) {
      for (int j = 0; j < kn; ++j) {
        for (int d = 0; d < vdim; ++d) {
          vd[(static_cast<size_t>(t) * kn + j) * vdim + d] =
              kd[(static_cast<size_t>(t) * kn + j) * dim + d];
        }
      }
    }
  }
  ra.v = vd.empty() ? kd.data() : vd.data();
  ra.qo_indptr = qo.data();
  ra.kv_indptr = ki.data();
  ra.custom_mask = mask.data();
  ra.mask_indptr = mind.data();
  ra.batch = T;
  ra.q_heads = H;
  ra.kv_heads = 1;
  ra.head_dim_qk = dim;
  ra.head_dim_vo = vdim;
  ra.sm_scale = p.mod.sm_scale;
  ra.causal = false;
  ra.sink_logits = c.sink ? sinkd.data() : nullptr;
  ra.o = od.data();
  ra.lse = ld.data();
  KOPS_EXPECT_OK(kops::ref::attention_f64(ra), "attention_f64 compressed");
  const auto got_o = kops_test::copy_out(d_o, od.size());
  const auto got_lse = kops_test::copy_out(d_lse, static_cast<size_t>(T) * H);
  char what[128];
  std::snprintf(what, sizeof(what), "compressed_attention o [%s]", c.name);
  kops_test::compare(got_o, od, 2e-2, 2e-2, what);
  std::snprintf(what, sizeof(what), "compressed_attention lse [%s]", c.name);
  kops_test::compare(got_lse, ld, 2e-2, 2e-2, what);

  kops_test::device_free(d_q);
  kops_test::device_free(d_comp);
  kops_test::device_free(d_win);
  kops_test::device_free(d_sel);
  kops_test::device_free(d_sel_len);
  kops_test::device_free(d_tab_comp);
  kops_test::device_free(d_tab_win);
  kops_test::device_free(d_t2r);
  kops_test::device_free(d_ncomp);
  kops_test::device_free(d_pos);
  kops_test::device_free(d_o);
  kops_test::device_free(d_lse);
  kops_test::device_free(d_sink);
}

}  // namespace

int main() {
  // Small CSA shapes (Hq < 16 heads in one CTA) with and without the sink.
  run_case({"csa.t2.h4.topk8", 2, 4, 32, 8, 0, false, false, -1, 512, 512, 0});
  run_case({"csa.t2.h4.topk8.sink", 2, 4, 32, 8, 0, true, false, -1, 512, 512, 0});
  // The reported cases: T = 1 / 16, 4K / 64K compressed pool, topk 2048,
  // 16 query heads per CTA, SWA window + sink.
  run_case({"csa.t1.h16.pool4k.topk2048.w96.sink", 1, 16, 4096, 2048, 96, true, false, -1, 512, 512, 0});
  run_case({"csa.t16.h16.pool4k.topk2048.w96.sink", 16, 16, 4096, 2048, 96, true, false, -1, 512, 512, 0});
  run_case({"csa.t1.h16.pool64k.topk2048.w96.sink", 1, 16, 65536, 2048, 96, true, false, -1, 512, 512, 0});
  run_case({"csa.t16.h16.pool64k.topk2048.w96.sink", 16, 16, 65536, 2048, 96, true, false, -1, 512, 512, 0});
  // topk 1024 / 512 + two CTAs per row (Hq = 32 -> grid.y = 2).
  run_case({"csa.t1.h16.pool4k.topk1024.w96.sink", 1, 16, 4096, 1024, 96, true, false, -1, 512, 512, 0});
  run_case({"csa.t2.h32.pool4k.topk2048.w96.sink", 2, 32, 4096, 2048, 96, true, false, -1, 512, 512, 0});
  // HCA dense compressed range.
  run_case({"hca.t1.h16.dense512.w96.sink", 1, 16, 512, 512, 96, true, true, -1, 512, 512, 0});
  run_case({"hca.t16.h16.dense512.w96.sink", 16, 16, 512, 512, 96, true, true, -1, 512, 512, 0});
  // Fully masked row: every selected entry is -1 -> o = 0, lse = -inf.
  run_case({"csa.t4.h16.topk64.maskedrow", 4, 16, 1024, 64, 0, false, false, 1, 512, 512, 0});
  // Entry/value slices outside the m16n8k16 tiles keep the scalar fallback:
  // entry_dim 256, and a 256-wide value slice out of a 512-wide record.
  run_case({"csa.t2.h16.dim256.topk64.w32.sink", 2, 16, 1024, 64, 32, true, false, -1, 256, 256, 0});
  run_case({"csa.t2.h16.value256.topk64.w32.sink", 2, 16, 1024, 64, 32, true, false, -1, 512, 256, 0});
  // selected_topk_length truncates the CSA branch mid-row; the rest stays masked.
  run_case({"csa.t1.h16.pool4k.topk2048.sellen512.w96.sink", 1, 16, 4096, 2048, 96, true, false, -1, 512, 512, 512});
  return kops_test::finish("compressed_attention");
}
