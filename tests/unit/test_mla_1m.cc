// =============================================================================
// kernel-ops | tests/unit/test_mla_1m.cc
// B11 long-context smoke: dense MLA decode over a 1M-token latent cache.
// All-zero Q/K/V makes the expected result exact: uniform softmax, o = 0 and
// LSE = log(kv_len). The test catches page-table/coordinate bugs that only show
// up beyond the L2 / page range.
// =============================================================================
#include "tests/unit/harness.h"

#include <cmath>
#include <cstdint>
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

void run() {
  constexpr int H = 128, NOPE = 512, PE = 64, PAGE = 64;
  constexpr int64_t KV = 1 << 20;
  constexpr int64_t PAGES = KV / PAGE;
  auto* d_cache = kops_test::device_alloc<__nv_bfloat16>(KV * (NOPE + PE));
  auto* d_qn = kops_test::device_alloc<__nv_bfloat16>(H * NOPE);
  auto* d_qp = kops_test::device_alloc<__nv_bfloat16>(H * PE);
  auto* d_o = kops_test::device_alloc<__nv_bfloat16>(H * NOPE);
  auto* d_lse = kops_test::device_alloc<float>(H);
  auto* d_table = kops_test::device_alloc<int32_t>(PAGES);
  auto* d_seq = kops_test::device_alloc<int32_t>(1);
  if (!d_cache || !d_qn || !d_qp || !d_o || !d_lse || !d_table || !d_seq) return;
  cudaMemset(d_cache, 0, KV * (NOPE + PE) * sizeof(__nv_bfloat16));
  cudaMemset(d_qn, 0, H * NOPE * sizeof(__nv_bfloat16));
  cudaMemset(d_qp, 0, H * PE * sizeof(__nv_bfloat16));
  std::vector<int32_t> table(PAGES);
  for (int64_t i = 0; i < PAGES; ++i) table[i] = static_cast<int32_t>(i);
  std::vector<int32_t> seq = {static_cast<int32_t>(KV)};
  kops_test::copy_in(d_table, table);
  kops_test::copy_in(d_seq, seq);

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops::attn::AttentionSpec spec;
  spec.num_q_heads = H;
  spec.num_kv_heads = 1;
  spec.head_dim_qk = NOPE + PE;
  spec.head_dim_vo = NOPE;
  spec.q_dtype = spec.o_dtype = kops::DType::kBFloat16;
  spec.kv_dtype = kops::DType::kBFloat16;
  spec.kv_layout = kops::kv::KVLayout::kLatent;
  spec.page_size = PAGE;
  spec.return_lse = true;
  kops::attn::AttentionPlan plan;
  kops::Workspace ws{};
  KOPS_EXPECT_OK(kops::attn::mla_decode_plan(spec, 1, 1, ctx, ws, &plan), "mla_decode_plan 1m");

  kops::kv::PagedKVDesc kd{};
  kd.k_data = d_cache;
  kd.k_dtype = kd.v_dtype = kops::DType::kBFloat16;
  kd.layout = kops::kv::KVLayout::kLatent;
  kd.page_size = PAGE;
  kd.num_kv_heads = 1;
  kd.head_dim_k = NOPE + PE;
  kd.head_dim_v = NOPE;
  kd.stride_unit = kops::StrideUnit::kElement;
  kd.k_stride_page = PAGE * (NOPE + PE);
  kd.k_stride_token = NOPE + PE;
  kd.k_stride_head = NOPE + PE;
  kd.table_format = kops::kv::PageTableFormat::kBlockTable;
  kd.block_table = d_table;
  kd.block_table_stride = static_cast<int32_t>(PAGES);
  kd.seq_lens = d_seq;

  kops::attn::MLADecodeParams p{};
  p.q_nope = d_qn;
  p.q_nope_stride_n = H * NOPE;
  p.q_nope_stride_h = NOPE;
  p.q_pe = d_qp;
  p.q_pe_stride_n = H * PE;
  p.q_pe_stride_h = PE;
  p.q_len = 1;
  p.kv = kd;
  p.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(NOPE + PE));
  p.o = d_o;
  p.o_stride_n = H * NOPE;
  p.o_stride_h = NOPE;
  p.lse = d_lse;
  p.batch_size = 1;
  p.num_heads = H;
  p.kv_lora_rank = NOPE;
  p.rope_dim = PE;
  KOPS_EXPECT_OK(kops::attn::mla_decode_run(plan, p, ws, ctx), "mla_decode_run 1m");

  const auto lse = kops_test::copy_out(d_lse, H);
  const auto o = kops_test::copy_out(d_o, H * NOPE);
  std::vector<double> want(H, std::log(static_cast<double>(KV)));
  kops_test::compare(lse, want, 2e-2, 2e-2, "1M LSE");
  for (int i = 0; i < 64; ++i) {
    const float v = kops_test::to_float(o[static_cast<size_t>(i) * 97]);
    KOPS_TEST_CHECK(std::fabs(v) < 1e-5f);
  }

  kops_test::device_free(d_cache);
  kops_test::device_free(d_qn);
  kops_test::device_free(d_qp);
  kops_test::device_free(d_o);
  kops_test::device_free(d_lse);
  kops_test::device_free(d_table);
  kops_test::device_free(d_seq);
}

}  // namespace

int main() {
  run();
  return kops_test::finish("mla_1m");
}
