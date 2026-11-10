// =============================================================================
// kernel-ops | tests/unit/test_mla_metadata.cc
// B11 mla_get_metadata: bit-exact against the host port of FlashMLA
// get_decoding_sched_meta (dense decode). This is a generic kernel, so the test
// runs on every CUDA target.
// =============================================================================
#include "tests/unit/harness.h"

#include <cstdint>
#include <cstdio>
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

struct Ref {
  std::vector<int32_t> meta;    // [num_sm, 8]
  std::vector<int32_t> splits;  // [B + 1]
};

Ref reference(const std::vector<int32_t>& seq_lens, int num_sm_parts, int block_size_n,
               int fixed_overhead) {
  const int b = static_cast<int>(seq_lens.size());
  Ref r;
  r.meta.assign(static_cast<size_t>(num_sm_parts) * 8, 0);
  r.splits.assign(static_cast<size_t>(b) + 1, 0);
  std::vector<int32_t> blocks(b), first(b), last(b), sk(b);
  int total = 0;
  for (int i = 0; i < b; ++i) {
    sk[i] = seq_lens[i] < 0 ? 0 : seq_lens[i];
    const int lt = sk[i] > 0 ? sk[i] - 1 : 0;
    first[i] = 0;
    last[i] = lt / block_size_n;
    blocks[i] = last[i] - first[i] + 1;
    total += blocks[i] + fixed_overhead;
  }
  int now_req = 0, now_block = 0, now_split = 0, cum = 0;
  const int payload = (total + num_sm_parts - 1) / num_sm_parts + fixed_overhead;
  for (int i = 0; i < num_sm_parts; ++i) {
    if (now_req >= b) {
      int32_t* m = r.meta.data() + static_cast<size_t>(i) * 8;
      const int empty_block = b > 0 ? last[b - 1] + 1 : 0;
      m[0] = b;
      m[1] = b - 1;
      m[2] = empty_block;
      m[3] = empty_block;
      m[4] = now_split;
      m[5] = 0;
      m[6] = 0;
      m[7] = 0;
      continue;
    }
    const int begin_req = now_req;
    const int begin_block = now_block + first[now_req];
    const int begin_split = now_split;
    bool first_split = now_block != 0;
    int remain = payload;
    while (now_req < b) {
      const int remain_blocks = blocks[now_req] - now_block;
      if (remain >= remain_blocks + fixed_overhead) {
        cum += now_split + 1;
        r.splits[now_req + 1] = cum;
        remain -= remain_blocks + fixed_overhead;
        ++now_req;
        now_block = 0;
        now_split = 0;
      } else {
        if (remain - fixed_overhead > 0) {
          now_block += remain - fixed_overhead;
          ++now_split;
        }
        break;
      }
    }
    const int end_req = now_block > 0 ? now_req : now_req - 1;
    const int end_block = now_block > 0
                              ? now_block + first[now_req]
                              : (sk[now_req - 1] == 0 ? 0 : last[now_req - 1] + 1);
    bool last_split = end_block != last[end_req] + 1 && sk[end_req] != 0;
    if (begin_req == end_req) first_split = last_split = first_split || last_split;
    int32_t* m = r.meta.data() + static_cast<size_t>(i) * 8;
    m[0] = begin_req;
    m[1] = end_req;
    m[2] = begin_block;
    m[3] = end_block;
    m[4] = begin_split;
    m[5] = first_split ? 1 : 0;
    m[6] = last_split ? 1 : 0;
    m[7] = 0;
  }
  return r;
}

void run_case(const std::vector<int32_t>& seq, int num_sm, int block, int overhead) {
  const int b = static_cast<int>(seq.size());
  auto* d_seq = kops_test::device_alloc<int32_t>(b);
  auto* d_meta = kops_test::device_alloc<int32_t>(static_cast<size_t>(num_sm) * 8);
  auto* d_splits = kops_test::device_alloc<int32_t>(b + 1);
  kops_test::copy_in(d_seq, seq);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) {
    kops_test::device_free(d_seq);
    kops_test::device_free(d_meta);
    kops_test::device_free(d_splits);
    return;
  }
  auto meta = kops_test::tensor2d(d_meta, kops::DType::kInt32, num_sm, 8);
  auto splits = kops_test::tensor2d(d_splits, kops::DType::kInt32, b + 1, 1);
  auto lens = kops_test::tensor2d(d_seq, kops::DType::kInt32, b, 1);
  // tensor2d gives contiguous rows; num_splits/seq_lens are 1-D views.
  splits.ndim = 1;
  splits.shape[0] = b + 1;
  splits.stride[0] = 1;
  lens.ndim = 1;
  lens.shape[0] = b;
  lens.stride[0] = 1;
  const auto st = kops::attn::mla_get_metadata(meta, splits, lens, 0, num_sm, block, overhead, ctx);
  KOPS_TEST_CHECK(st == kops::Status::kSuccess);
  if (st == kops::Status::kSuccess) {
    const Ref want = reference(seq, num_sm, block, overhead);
    const auto got_meta = kops_test::copy_out(d_meta, static_cast<size_t>(num_sm) * 8);
    const auto got_splits = kops_test::copy_out(d_splits, b + 1);
    if (got_meta != want.meta) {
      std::fprintf(stderr, "meta mismatch B=%d sm=%d block=%d overhead=%d\n", b, num_sm, block,
                   overhead);
      for (int i = 0; i < num_sm; ++i) {
        bool diff = false;
        for (int j = 0; j < 8; ++j) diff |= got_meta[i * 8 + j] != want.meta[i * 8 + j];
        if (diff) {
          std::fprintf(stderr, " i=%d got", i);
          for (int j = 0; j < 8; ++j) std::fprintf(stderr, " %d", got_meta[i * 8 + j]);
          std::fprintf(stderr, " want");
          for (int j = 0; j < 8; ++j) std::fprintf(stderr, " %d", want.meta[i * 8 + j]);
          std::fprintf(stderr, "\n");
        }
      }
    }
    KOPS_TEST_CHECK(got_meta == want.meta);
    KOPS_TEST_CHECK(got_splits == want.splits);
  }
  kops_test::device_free(d_seq);
  kops_test::device_free(d_meta);
  kops_test::device_free(d_splits);
}

}  // namespace

int main() {
  run_case({0}, 1, 64, 0);
  run_case({1}, 4, 64, 1);
  run_case({64, 65, 4096}, 4, 64, 0);
  run_case({0, 1, 63, 64, 65, 4096, 1 << 20}, 32, 64, 1);
  run_case({1 << 20}, 132, 64, 2);
  run_case({0, 0, 1, 1, 4096, 1 << 20}, 192, 64, 0);
  return kops_test::finish("mla_metadata");
}
