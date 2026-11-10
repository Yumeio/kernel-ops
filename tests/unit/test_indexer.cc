// =============================================================================
// kernel-ops | tests/unit/test_indexer.cc
// B14 lightning indexer (T1): weighted ReLU MQA logits over a ragged key range.
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

void run_case() {
  constexpr int T = 2, H = 64, D = 128, S = 96;
  std::vector<float> qf(static_cast<size_t>(T) * H * D), kf(static_cast<size_t>(S) * D),
      wf(static_cast<size_t>(T) * H);
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : qf) v = dist(rng);
  for (float& v : kf) v = dist(rng);
  for (float& v : wf) v = 0.5f + 0.5f * std::fabs(dist(rng));
  std::vector<__nv_bfloat16> q(qf.size()), k(kf.size());
  for (size_t i = 0; i < q.size(); ++i) q[i] = __float2bfloat16(qf[i]);
  for (size_t i = 0; i < k.size(); ++i) k[i] = __float2bfloat16(kf[i]);
  std::vector<int32_t> starts = {0, 10}, ends = {80, 96};

  auto* d_q = kops_test::device_alloc<__nv_bfloat16>(q.size());
  auto* d_k = kops_test::device_alloc<__nv_bfloat16>(k.size());
  auto* d_w = kops_test::device_alloc<float>(wf.size());
  auto* d_s = kops_test::device_alloc<int32_t>(T);
  auto* d_e = kops_test::device_alloc<int32_t>(T);
  auto* d_logits = kops_test::device_alloc<float>(static_cast<size_t>(T) * S);
  kops_test::copy_in(d_q, q);
  kops_test::copy_in(d_k, k);
  kops_test::copy_in(d_w, wf);
  kops_test::copy_in(d_s, starts);
  kops_test::copy_in(d_e, ends);

  kops::attn::IndexerParams p{};
  p.q = d_q;
  p.q_stride_n = H * D;
  p.q_stride_h = D;
  p.head_weights = d_w;
  p.k_cache.k_data = d_k;
  p.k_cache.k_dtype = kops::DType::kBFloat16;
  p.k_cache.k_codec = kops::kv::KVCodec::kNative;
  p.k_cache.layout = kops::kv::KVLayout::kLatent;
  p.k_cache.page_size = 0;
  p.k_cache.head_dim_k = 0;
  p.k_cache.table_format = kops::kv::PageTableFormat::kSlotMapping;
  p.row_starts = d_s;
  p.row_ends = d_e;
  p.logits = d_logits;
  p.logits_stride = S;
  p.num_tokens = T;
  p.num_index_heads = H;
  p.index_dim = D;
  p.max_kv = S;
  p.compress_ratio = 1;
  p.k_stride_n = D;
  p.k_stride_h = D;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const auto st = kops::attn::lightning_indexer(p, kops::DType::kBFloat16, ctx);
  KOPS_TEST_CHECK(st == kops::Status::kSuccess);
  if (st != kops::Status::kSuccess) return;
  const auto got = kops_test::copy_out(d_logits, static_cast<size_t>(T) * S);
  std::vector<double> want(static_cast<size_t>(T) * S, -INFINITY);
  for (int t = 0; t < T; ++t) {
    for (int s = starts[t]; s < ends[t]; ++s) {
      double acc = 0.0;
      for (int h = 0; h < H; ++h) {
        double dot = 0.0;
        for (int d = 0; d < D; ++d) {
          dot += static_cast<double>(__bfloat162float(q[(static_cast<size_t>(t) * H + h) * D + d])) *
                 static_cast<double>(__bfloat162float(k[static_cast<size_t>(s) * D + d]));
        }
        acc += wf[static_cast<size_t>(t) * H + h] * std::max(dot, 0.0);
      }
      want[static_cast<size_t>(t) * S + s] = acc;
    }
  }
  kops_test::compare(got, want, 2e-2, 2e-2, "lightning_indexer");

  kops_test::device_free(d_q);
  kops_test::device_free(d_k);
  kops_test::device_free(d_w);
  kops_test::device_free(d_s);
  kops_test::device_free(d_e);
  kops_test::device_free(d_logits);
}

}  // namespace

int main() {
  run_case();
  return kops_test::finish("indexer");
}
