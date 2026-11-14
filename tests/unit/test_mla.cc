// =============================================================================
// kernel-ops | tests/unit/test_mla.cc                                 Catalog: A12
// Dense absorbed MLA decode (T1) and the FlashMLA split scheduler metadata.
//
//   mla_get_metadata        vs a host port of FlashMLA get_decoding_sched_meta
//                           (dense decode): B 1/3/32, seq 0/1/64/65/4096/1M,
//                           num_sm_parts 1/4/132/192, block 16/64, overhead 0/2.
//   mla_decode_kernel       vs ref::attention_f64 on the dequantized latent cache:
//                           bf16 and kMLAFp8DS 656 B, fp16, sinks, output gate,
//                           page sizes 16/32/64, q_len 1..8, H 32..128, split
//                           S=1/2/4/16/32 through merge_split_kv, empty requests.
//   contract                every rejected dtype/shape/layout/alignment returns
//                           the documented Status.
//
// Tests return 77 (CTest SKIP_RETURN_CODE) when the machine lacks sm_80+ or a GPU.
// =============================================================================

#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/occupancy.h"
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

using kops::DType;
using kops::Status;

constexpr int kNope = 512;
constexpr int kPe = 64;
constexpr int kRow = kNope + kPe;

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

void fill_uniform(std::vector<float>* v, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& x : *v) x = dist(rng);
}

kops::Tensor tensor1d(void* data, DType dtype, int64_t n) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

template <typename T>
std::vector<T> cast_from_float(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(out[i], src[i]);
  return out;
}

/// Power-of-two tile scale of the kMLAFp8DS record (floor of the kvcache writer).
float fp8_ds_scale(float amax) {
  const float unrounded = std::fmax(amax / 448.0f, 1e-4f);
  uint32_t bits = 0;
  std::memcpy(&bits, &unrounded, sizeof(bits));
  bits = (bits + 0x7fffffu) & 0x7f800000u;
  float scale = 0.0f;
  std::memcpy(&scale, &bits, sizeof(scale));
  return scale;
}

// =============================================================================
// 1. mla_get_metadata: host port of FlashMLA get_decoding_sched_meta (dense).
// =============================================================================

struct HostMeta {
  std::vector<int32_t> meta;    // [num_sm_parts][8]
  std::vector<int32_t> splits;  // [B + 1]
};

HostMeta flashmla_metadata_host(const std::vector<int32_t>& seq_lens, int num_sm_parts,
                                int block_size_n, int overhead) {
  HostMeta out;
  const int B = static_cast<int>(seq_lens.size());
  out.meta.assign(static_cast<size_t>(num_sm_parts) * 8, 0);
  out.splits.assign(static_cast<size_t>(B) + 1, 0);
  if (B == 0 || num_sm_parts <= 0) return out;
  std::vector<int> nb(B), lb(B), sk(B);
  int total = 0;
  for (int i = 0; i < B; ++i) {
    sk[i] = std::max(seq_lens[i], 0);
    const int last = sk[i] > 0 ? sk[i] - 1 : 0;  // dense: first token is always 0
    lb[i] = last / block_size_n;
    nb[i] = lb[i] + 1;                              // a zero-length row still owns one block
    total += nb[i] + overhead;
  }
  const int payload = (total + num_sm_parts - 1) / num_sm_parts + overhead;
  int now_req = 0, now_block = 0, now_split = 0, cum = 0;
  for (int i = 0; i < num_sm_parts; ++i) {
    int32_t* m = out.meta.data() + static_cast<size_t>(i) * 8;
    // Trailing partitions: nothing left to schedule. mla_get_metadata_kernel
    // reports an explicit empty range (begin > end, zero splits, empty block
    // range ending at the last block of the batch), so the partition is skipped
    // deterministically and no field is left uninitialized.
    if (now_req >= B) {
      const int empty_block = lb[B - 1] + 1;
      m[0] = B;
      m[1] = B - 1;
      m[2] = empty_block;
      m[3] = empty_block;
      m[4] = now_split;
      m[5] = 0;
      m[6] = 0;
      m[7] = 0;
      continue;
    }
    const int begin_req = now_req;
    const int begin_block = now_block;  // first_block[] is 0 for dense decode
    const int begin_split = now_split;
    const bool first_splitted = now_block != 0;
    int remain = payload;
    while (now_req < B) {
      const int blocks = nb[now_req];
      const int remain_blocks = blocks - now_block;
      if (remain >= remain_blocks + overhead) {
        cum += now_split + 1;
        out.splits[static_cast<size_t>(now_req) + 1] = cum;
        remain -= remain_blocks + overhead;
        ++now_req;
        now_block = 0;
        now_split = 0;
      } else {
        if (remain - overhead > 0) {
          now_block += remain - overhead;
          ++now_split;
        }
        break;
      }
    }
    const int end_req = now_block > 0 ? now_req : now_req - 1;
    const int end_block =
        now_block > 0 ? now_block : (sk[end_req] == 0 ? 0 : lb[end_req] + 1);
    bool last_splitted = end_block != lb[end_req] + 1 && sk[end_req] != 0;
    bool first = first_splitted;
    if (begin_req == end_req) first = last_splitted = first || last_splitted;
    m[0] = begin_req;
    m[1] = end_req;
    m[2] = begin_block;
    m[3] = end_block;
    m[4] = begin_split;
    m[5] = first ? 1 : 0;
    m[6] = last_splitted ? 1 : 0;
    m[7] = 0;
  }
  return out;
}

void test_metadata_case(const std::vector<int32_t>& seq, int num_sm_parts, int block_size_n,
                        int overhead, const char* tag) {
  const int B = static_cast<int>(seq.size());
  const HostMeta want = flashmla_metadata_host(seq, num_sm_parts, block_size_n, overhead);
  int32_t* d_meta = kops_test::device_alloc<int32_t>(static_cast<size_t>(num_sm_parts) * 8);
  int32_t* d_splits = kops_test::device_alloc<int32_t>(static_cast<size_t>(B) + 1);
  int32_t* d_seq = kops_test::device_alloc<int32_t>(static_cast<size_t>(std::max(B, 1)));
  if (d_meta == nullptr || d_splits == nullptr || d_seq == nullptr) return;
  kops_test::copy_in(d_seq, seq);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const kops::Tensor t_meta =
      kops_test::tensor2d(d_meta, DType::kInt32, num_sm_parts, 8);
  const kops::Tensor t_splits = tensor1d(d_splits, DType::kInt32, B + 1);
  const kops::Tensor t_seq = tensor1d(d_seq, DType::kInt32, std::max(B, 1));
  const Status st = kops::attn::mla_get_metadata(t_meta, t_splits, t_seq, 1, num_sm_parts,
                                                 block_size_n, overhead, ctx);
  if (st != Status::kSuccess) {
    kops_test::record_failure(tag, kops::last_error_message(), __FILE__, __LINE__);
    return;
  }
  const std::vector<int32_t> got_meta = kops_test::copy_out(d_meta, static_cast<size_t>(num_sm_parts) * 8);
  const std::vector<int32_t> got_splits = kops_test::copy_out(d_splits, static_cast<size_t>(B) + 1);
  if (got_splits != want.splits) {
    kops_test::record_failure(tag, "num_splits differs from the FlashMLA port", __FILE__, __LINE__);
  }
  for (int i = 0; i < num_sm_parts; ++i) {
    const int32_t* g = got_meta.data() + static_cast<size_t>(i) * 8;
    const int32_t* w = want.meta.data() + static_cast<size_t>(i) * 8;
    const bool same = g[0] == w[0] && g[1] == w[1] && g[2] == w[2] && g[3] == w[3] &&
                      g[4] == w[4] && g[5] == w[5] && g[6] == w[6] && g[7] == w[7];
    if (!same) {
      char detail[256];
      std::snprintf(detail, sizeof(detail),
                    "part %d: got [%d %d %d %d %d %d %d %d] want [%d %d %d %d %d %d %d %d]", i,
                    g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], w[0], w[1], w[2], w[3], w[4],
                    w[5], w[6], w[7]);
      kops_test::record_failure(tag, detail, __FILE__, __LINE__);
      break;
    }
  }
  kops_test::device_free(d_meta);
  kops_test::device_free(d_splits);
  kops_test::device_free(d_seq);
}

void test_metadata() {
  const std::vector<std::vector<int32_t>> lens = {
      {0}, {1}, {64}, {65}, {4096}, {1 << 20}, {0, 1, 64}, {64, 0, 4096},
      {4096, 1, 65, 0, 128},
  };
  const int parts[] = {1, 4, 132, 192};
  for (const std::vector<int32_t>& seq : lens) {
    for (int p : parts) {
      for (int block : {16, 64}) {
        for (int overhead : {0, 2}) {
          char tag[160];
          std::snprintf(tag, sizeof(tag), "mla_get_metadata B=%zu parts=%d block=%d oh=%d",
                        seq.size(), p, block, overhead);
          test_metadata_case(seq, p, block, overhead, tag);
        }
      }
    }
  }
  // A full 32-request batch makes the greedy partition span several parts.
  std::vector<int32_t> wide(32);
  for (int i = 0; i < 32; ++i) wide[static_cast<size_t>(i)] = (i % 4 == 0) ? 0 : (1 << (i % 12));
  test_metadata_case(wide, 4, 64, 0, "mla_get_metadata B=32 parts=4");
  test_metadata_case(wide, 132, 64, 2, "mla_get_metadata B=32 parts=132");
  test_metadata_case(wide, 192, 16, 0, "mla_get_metadata B=32 parts=192");

  // Contract: shapes and counts.
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  int32_t* dev = kops_test::device_alloc<int32_t>(64);
  if (dev == nullptr) return;
  const kops::Tensor meta_small = kops_test::tensor2d(dev, DType::kInt32, 1, 8);
  const kops::Tensor meta_ok = kops_test::tensor2d(dev, DType::kInt32, 4, 8);
  const kops::Tensor splits4 = tensor1d(dev, DType::kInt32, 4);
  const kops::Tensor seq3 = tensor1d(dev, DType::kInt32, 3);
  KOPS_EXPECT_STATUS(kops::attn::mla_get_metadata(meta_ok, splits4, seq3, 1, 0, 16, 0, ctx),
                     Status::kInvalidArgument, "metadata rejects num_sm_parts=0");
  KOPS_EXPECT_STATUS(kops::attn::mla_get_metadata(meta_ok, splits4, seq3, 1, 4, 0, 0, ctx),
                     Status::kInvalidArgument, "metadata rejects block_size_n=0");
  KOPS_EXPECT_STATUS(kops::attn::mla_get_metadata(meta_small, splits4, seq3, 1, 4, 16, 0, ctx),
                     Status::kInvalidArgument, "metadata rejects a short metadata buffer");
  KOPS_EXPECT_STATUS(kops::attn::mla_get_metadata(meta_ok, splits4, seq3, 1, 4, 16, -1, ctx),
                     Status::kInvalidArgument, "metadata rejects a negative overhead");
  kops_test::device_free(dev);
}

// =============================================================================
// 2. Dense absorbed MLA decode
// =============================================================================

struct MlaCase {
  const char* name;
  int batch = 1;
  std::vector<int> kv = {64};
  int heads = 64;
  int q_len = 1;
  int page = 16;
  bool fp8 = false;    // kMLAFp8DS 656 B record
  bool fp16 = false;   // q/o/fp16 instead of bf16
  bool sink = false;
  bool gate = false;
  int splits = 1;                  // per-request split count when split_counts is empty
  bool full_tier = false;          // only with --tier full
  std::vector<int> split_counts;   // [B] variable per-request splits (FlashMLA pattern)
};

struct LatentPool {
  std::vector<uint8_t> cache;      // pages * page * record_bytes
  std::vector<double> nope, rope;  // dequantized [total_kv][512] / [64]
  std::vector<int32_t> block_table, seq_lens;
  int max_pages = 0;
  int64_t total_pages = 0;
  int total_kv = 0;
};

/// Lay out the paged latent cache with a shuffled physical page order and keep the
/// dequantized value of every (request, position) for the fp64 reference.
LatentPool build_latent(const MlaCase& c, unsigned seed) {
  LatentPool p;
  const int B = c.batch, ps = c.page;
  const int64_t rec = c.fp8 ? 656 : kRow * 2;
  std::vector<int> npages(static_cast<size_t>(B), 0);
  int64_t pages = 0;
  for (int b = 0; b < B; ++b) {
    npages[static_cast<size_t>(b)] = (c.kv[static_cast<size_t>(b)] + ps - 1) / ps;
    pages += npages[static_cast<size_t>(b)];
  }
  p.total_pages = pages;
  p.max_pages = 1;
  for (int n : npages) p.max_pages = std::max(p.max_pages, n);
  p.block_table.assign(static_cast<size_t>(B) * p.max_pages, 0);
  p.seq_lens.assign(static_cast<size_t>(B), 0);
  // Physical page ids in reverse order: the kernel must follow the table.
  int64_t next = pages - 1;
  for (int b = 0; b < B; ++b) {
    p.seq_lens[static_cast<size_t>(b)] = c.kv[static_cast<size_t>(b)];
    for (int i = 0; i < npages[static_cast<size_t>(b)]; ++i) {
      p.block_table[static_cast<size_t>(b) * p.max_pages + i] = static_cast<int32_t>(next--);
    }
  }
  p.cache.assign(static_cast<size_t>(pages * ps) * rec, 0);
  for (int b = 0; b < B; ++b) p.total_kv += c.kv[static_cast<size_t>(b)];
  p.nope.assign(static_cast<size_t>(p.total_kv) * kNope, 0.0);
  p.rope.assign(static_cast<size_t>(p.total_kv) * kPe, 0.0);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> nope(kNope), rope(kPe);
  int64_t row = 0;
  for (int b = 0; b < B; ++b) {
    for (int j = 0; j < c.kv[static_cast<size_t>(b)]; ++j, ++row) {
      for (float& x : nope) x = dist(rng);
      for (float& x : rope) x = dist(rng);
      const int page = p.block_table[static_cast<size_t>(b) * p.max_pages + j / ps];
      const int off = j % ps;
      uint8_t* dst = p.cache.data() + (static_cast<size_t>(page) * ps + off) * rec;
      if (c.fp8) {
        for (int tile = 0; tile < 4; ++tile) {
          float amax = 0.0f;
          for (int i = 0; i < 128; ++i) amax = std::fmax(amax, std::fabs(nope[tile * 128 + i]));
          const float scale = fp8_ds_scale(amax);
          reinterpret_cast<float*>(dst + kNope)[tile] = scale;
          for (int i = 0; i < 128; ++i) {
            const float q = std::fmax(-448.0f, std::fmin(448.0f, nope[tile * 128 + i] / scale));
            dst[tile * 128 + i] = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
          }
        }
        __nv_bfloat16* pe = reinterpret_cast<__nv_bfloat16*>(dst + kNope + 16);
        for (int i = 0; i < kPe; ++i) pe[i] = __float2bfloat16_rn(rope[i]);
        for (int i = 0; i < kNope; ++i) {
          p.nope[static_cast<size_t>(row) * kNope + i] =
              static_cast<double>(__half2float(__nv_cvt_fp8_to_halfraw(dst[i], __NV_E4M3)) *
                                  reinterpret_cast<const float*>(dst + kNope)[i / 128]);
        }
        for (int i = 0; i < kPe; ++i) {
          p.rope[static_cast<size_t>(row) * kPe + i] = static_cast<double>(__bfloat162float(pe[i]));
        }
      } else if (c.fp16) {
        __half* h = reinterpret_cast<__half*>(dst);
        for (int i = 0; i < kNope; ++i) h[i] = __float2half_rn(nope[i]);
        for (int i = 0; i < kPe; ++i) h[kNope + i] = __float2half_rn(rope[i]);
        for (int i = 0; i < kNope; ++i) {
          p.nope[static_cast<size_t>(row) * kNope + i] = static_cast<double>(__half2float(h[i]));
        }
        for (int i = 0; i < kPe; ++i) {
          p.rope[static_cast<size_t>(row) * kPe + i] =
              static_cast<double>(__half2float(h[kNope + i]));
        }
      } else {
        __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(dst);
        for (int i = 0; i < kNope; ++i) h[i] = __float2bfloat16_rn(nope[i]);
        for (int i = 0; i < kPe; ++i) h[kNope + i] = __float2bfloat16_rn(rope[i]);
        for (int i = 0; i < kNope; ++i) {
          p.nope[static_cast<size_t>(row) * kNope + i] =
              static_cast<double>(__bfloat162float(h[i]));
        }
        for (int i = 0; i < kPe; ++i) {
          p.rope[static_cast<size_t>(row) * kPe + i] =
              static_cast<double>(__bfloat162float(h[kNope + i]));
        }
      }
    }
  }
  return p;
}


template <typename T>
__device__ __forceinline__ T from_float_dev(float x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(x);
  } else {
    return __float2bfloat16_rn(x);
  }
}

// merge_split_kv does not compile head_dim 512 yet (merge.cu dispatches
// 64/80/96/128/192/256), so the split-KV tests perform the identical state
// algebra here: m = logsumexp(partial lse, sink), o = sum_s w_s o_s / denom,
// lse = log(denom) + m. This is the same math as merge_split_kv_kernel.
template <typename T>
__global__ void test_merge_split_kv(T* __restrict__ o, float* __restrict__ lse,
                                    const float* __restrict__ o_partial,
                                    const float* __restrict__ lse_partial,
                                    const int32_t* __restrict__ indptr,
                                    const float* __restrict__ sink, const int heads,
                                    const int head_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int first = indptr[row], last = indptr[row + 1];
  const float neg = -INFINITY;
  const float sk = sink != nullptr ? sink[h] : neg;
  float m = sk;
  for (int s = first; s < last; ++s) m = fmaxf(m, lse_partial[s * heads + h]);
  float den = 0.0f;
  if (m != neg) {
    if (sink != nullptr) den = expf(sk - m);
    for (int s = first; s < last; ++s) {
      const float l = lse_partial[s * heads + h];
      if (l != neg) den += expf(l - m);
    }
  }
  const float inv = den > 0.0f ? 1.0f / den : 0.0f;
  for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x)) {
    float acc = 0.0f;
    for (int s = first; s < last; ++s) {
      const float l = lse_partial[s * heads + h];
      const float w = (l == neg || m == neg) ? 0.0f : expf(l - m);
      if (w != 0.0f) {
        acc = fmaf(w, o_partial[(static_cast<int64_t>(s) * heads + h) * head_dim + d], acc);
      }
    }
    o[(static_cast<int64_t>(row) * heads + h) * head_dim + d] = from_float_dev<T>(acc * inv);
  }
  if (threadIdx.x == 0) {
    lse[static_cast<int64_t>(row) * heads + h] = den > 0.0f ? m + logf(den) : neg;
  }
}

template <typename T>
void merge_split_ref(T* o, float* lse, const float* o_partial, const float* lse_partial,
                     const int32_t* indptr, const float* sink, int rows, int heads,
                     int head_dim, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(heads));
  test_merge_split_kv<T><<<grid, 128, 0, stream>>>(o, lse, o_partial, lse_partial, indptr, sink,
                                                   heads, head_dim);
}

template <typename T>
void run_mla_case_t(const MlaCase& c, double atol, double rtol) {
  const int B = c.batch, H = c.heads, ql = c.q_len, ps = c.page;
  const int64_t total_q = static_cast<int64_t>(B) * ql;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  const LatentPool pool = build_latent(c, 0x51a7u + static_cast<unsigned>(ql * 17 + ps));
  const DType dt = c.fp16 ? DType::kFloat16 : DType::kBFloat16;
  const DType kv_dt = c.fp8 ? DType::kFloat8E4M3 : dt;

  // q tensors [B*q_len, H, 512] / [B*q_len, H, 64] and gate [B*q_len, H, 512].
  std::vector<float> qn_f(static_cast<size_t>(total_q) * H * kNope);
  std::vector<float> qp_f(static_cast<size_t>(total_q) * H * kPe);
  fill_uniform(&qn_f, 1u + static_cast<unsigned>(H));
  fill_uniform(&qp_f, 2u + static_cast<unsigned>(H), -0.5f, 0.5f);
  const auto qn_h = cast_from_float<T>(qn_f);
  const auto qp_h = cast_from_float<T>(qp_f);
  std::vector<float> gate_f;
  std::vector<T> gate_h;
  if (c.gate) {
    gate_f.resize(qn_f.size());
    fill_uniform(&gate_f, 3u, -2.0f, 2.0f);
    gate_h = cast_from_float<T>(gate_f);
  }
  std::vector<float> sink_f(static_cast<size_t>(H));
  if (c.sink) fill_uniform(&sink_f, 4u, -1.5f, 1.5f);

  T* d_qn = kops_test::device_alloc<T>(qn_h.size());
  T* d_qp = kops_test::device_alloc<T>(qp_h.size());
  T* d_o = kops_test::device_alloc<T>(static_cast<size_t>(total_q) * H * kNope);
  float* d_lse = kops_test::device_alloc<float>(static_cast<size_t>(total_q) * H);
  T* d_gate = c.gate ? kops_test::device_alloc<T>(gate_h.size()) : nullptr;
  float* d_sink = c.sink ? kops_test::device_alloc<float>(static_cast<size_t>(H)) : nullptr;
  void* d_cache = nullptr;
  int32_t* d_table = kops_test::device_alloc<int32_t>(pool.block_table.size());
  int32_t* d_seq = kops_test::device_alloc<int32_t>(static_cast<size_t>(B));
  float* d_opart = nullptr;
  float* d_lpart = nullptr;
  int32_t* d_indptr = nullptr;
  // Per-request split counts: FlashMLA lets a request with a long KV own more splits.
  std::vector<int> ns_req(static_cast<size_t>(B), c.splits);
  // splits == 0 is the auto mode of the benchmark: the host helper picks a uniform count from
  // the context's compute units (occupancy.h). Never reads a device sequence length.
  if (c.splits == 0) {
    int32_t max_kv = 0;
    for (int kv : c.kv) max_kv = std::max(max_kv, kv);
    kops::attn::MLASplitHint hint{};
    const Status hs = kops::attn::mla_decode_split_hint(B, ql, H, max_kv, 64, 2, ctx, &hint);
    if (hs != Status::kSuccess) {
      kops_test::record_failure(c.name, kops::last_error_message(), __FILE__, __LINE__);
      return;
    }
    if (hint.splits < 1 || hint.splits > 64 || (hint.splits > 1 && hint.total_bytes == 0)) {
      kops_test::record_failure(c.name, "auto split hint out of range", __FILE__, __LINE__);
      return;
    }
    for (int& n : ns_req) n = hint.splits;
  }
  if (!c.split_counts.empty()) ns_req = c.split_counts;
  int max_splits = 1;
  std::vector<int> row_splits(static_cast<size_t>(total_q), 1);
  for (int b = 0; b < B; ++b) {
    max_splits = std::max(max_splits, ns_req[static_cast<size_t>(b)]);
    for (int i = 0; i < ql; ++i) {
      row_splits[static_cast<size_t>(b) * ql + i] = ns_req[static_cast<size_t>(b)];
    }
  }
  int64_t prow = 0;
  for (int n : row_splits) prow += n;
  const bool split_mode = max_splits > 1;
  if (split_mode) {
    d_opart = kops_test::device_alloc<float>(static_cast<size_t>(prow) * H * kNope);
    d_lpart = kops_test::device_alloc<float>(static_cast<size_t>(prow) * H);
    d_indptr = kops_test::device_alloc<int32_t>(static_cast<size_t>(total_q) + 1);
  }
  if (d_qn == nullptr || d_qp == nullptr || d_o == nullptr || d_lse == nullptr ||
      d_table == nullptr || d_seq == nullptr ||
      (c.gate && d_gate == nullptr) || (c.sink && d_sink == nullptr) ||
      (split_mode && (d_opart == nullptr || d_lpart == nullptr || d_indptr == nullptr))) {
    return;
  }
  if (cudaMalloc(&d_cache, std::max<size_t>(pool.cache.size(), 16)) != cudaSuccess) {
    kops_test::record_failure("cudaMalloc cache", cudaGetErrorString(cudaGetLastError()),
                              __FILE__, __LINE__);
    return;
  }
  kops_test::copy_in(d_qn, qn_h);
  kops_test::copy_in(d_qp, qp_h);
  kops_test::copy_in(d_table, pool.block_table);
  kops_test::copy_in(d_seq, pool.seq_lens);
  cudaMemcpy(d_cache, pool.cache.data(), pool.cache.size(), cudaMemcpyHostToDevice);
  cudaMemset(d_o, 1, static_cast<size_t>(total_q) * H * kNope * sizeof(T));
  if (c.gate) kops_test::copy_in(d_gate, gate_h);
  if (c.sink) kops_test::copy_in(d_sink, sink_f);
  std::vector<int32_t> indptr(static_cast<size_t>(total_q) + 1, 0);
  if (split_mode) {
    for (int64_t r = 0; r < total_q; ++r) {
      indptr[static_cast<size_t>(r) + 1] =
          indptr[static_cast<size_t>(r)] + row_splits[static_cast<size_t>(r)];
    }
    kops_test::copy_in(d_indptr, indptr);
  }

  kops::attn::AttentionSpec spec;
  spec.num_q_heads = H;
  spec.num_kv_heads = 1;
  spec.head_dim_qk = kRow;
  spec.head_dim_vo = kNope;
  spec.q_dtype = dt;
  spec.o_dtype = dt;
  spec.kv_dtype = kv_dt;
  spec.k_codec = c.fp8 ? kops::kv::KVCodec::kMLAFp8DS : kops::kv::KVCodec::kNative;
  spec.v_codec = kops::kv::KVCodec::kNative;
  spec.kv_layout = kops::kv::KVLayout::kLatent;
  spec.page_size = ps;
  spec.mask = kops::attn::MaskMode::kCausal;
  spec.sinks = c.sink;
  spec.return_lse = true;

  kops::attn::AttentionPlan plan;
  kops::Workspace ws{};
  Status st = kops::attn::mla_decode_plan(spec, B, ql, ctx, ws, &plan);
  if (st != Status::kSuccess) {
    kops_test::record_failure(c.name, kops::last_error_message(), __FILE__, __LINE__);
    return;
  }

  kops::kv::PagedKVDesc kd{};
  kd.k_data = d_cache;
  kd.k_dtype = kv_dt;
  kd.v_dtype = dt;
  kd.k_codec = c.fp8 ? kops::kv::KVCodec::kMLAFp8DS : kops::kv::KVCodec::kNative;
  kd.v_codec = kops::kv::KVCodec::kNative;
  kd.layout = kops::kv::KVLayout::kLatent;
  kd.page_size = ps;
  kd.num_kv_heads = 1;
  kd.head_dim_k = kRow;
  kd.head_dim_v = kNope;
  kd.entry_bytes = c.fp8 ? 656 : 0;
  kd.stride_unit = c.fp8 ? kops::StrideUnit::kByte : kops::StrideUnit::kElement;
  const int64_t token_unit = c.fp8 ? 656 : kRow;  // bytes / elements per record
  kd.k_stride_page = static_cast<int64_t>(ps) * token_unit;
  kd.k_stride_token = token_unit;
  kd.k_stride_head = token_unit;
  kd.table_format = kops::kv::PageTableFormat::kBlockTable;
  kd.block_table = d_table;
  kd.block_table_stride = pool.max_pages;
  kd.seq_lens = d_seq;

  kops::attn::AttentionModifiers mod;
  mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(kRow));
  mod.sink_logits = c.sink ? d_sink : nullptr;

  std::vector<int32_t> splits_dev;
  int32_t* d_splits = nullptr;
  if (split_mode) {
    splits_dev.resize(static_cast<size_t>(B) + 1, 0);
    for (int b = 0; b < B; ++b) {
      splits_dev[static_cast<size_t>(b) + 1] =
          splits_dev[static_cast<size_t>(b)] + ns_req[static_cast<size_t>(b)];
    }
    d_splits = kops_test::device_alloc<int32_t>(static_cast<size_t>(B) + 1);
    if (d_splits == nullptr) return;
    kops_test::copy_in(d_splits, splits_dev);
  }

  kops::attn::MLADecodeParams p{};
  p.q_nope = d_qn;
  p.q_nope_stride_n = static_cast<int64_t>(H) * kNope;
  p.q_nope_stride_h = kNope;
  p.q_pe = d_qp;
  p.q_pe_stride_n = static_cast<int64_t>(H) * kPe;
  p.q_pe_stride_h = kPe;
  p.q_len = ql;
  p.kv = kd;
  p.mod = mod;
  p.o = d_o;
  p.o_stride_n = static_cast<int64_t>(H) * kNope;
  p.o_stride_h = kNope;
  p.lse = d_lse;
  p.output_gate = c.gate ? d_gate : nullptr;
  p.gate_stride_n = c.gate ? static_cast<int64_t>(H) * kNope : 0;
  p.gate_stride_h = c.gate ? kNope : 0;
  p.gate_dtype = dt;
  p.split.o_partial = d_opart;
  p.split.lse_partial = d_lpart;
  p.split.merge_indptr = d_indptr;
  p.split.max_splits = max_splits;
  p.tile_scheduler_metadata = nullptr;
  p.num_splits = d_splits;
  p.batch_size = B;
  p.num_heads = H;
  p.kv_lora_rank = kNope;
  p.rope_dim = kPe;

  st = kops::attn::mla_decode_run(plan, p, ws, ctx);
  if (st != Status::kSuccess) {
    kops_test::record_failure(c.name, kops::last_error_message(), __FILE__, __LINE__);
    return;
  }
  if (split_mode) {
    merge_split_ref<T>(d_o, d_lse, d_opart, d_lpart, d_indptr, c.sink ? d_sink : nullptr,
                       static_cast<int>(total_q), H, kNope,
                       reinterpret_cast<cudaStream_t>(ctx.stream));
  }

  // ---- fp64 reference over the dequantized latent cache ----------------------
  std::vector<double> q_ref(static_cast<size_t>(total_q) * H * kRow);
  std::vector<double> k_ref(static_cast<size_t>(pool.total_kv) * kRow);
  std::vector<double> v_ref(static_cast<size_t>(pool.total_kv) * kNope);
  for (int64_t r = 0; r < total_q * H; ++r) {
    for (int d = 0; d < kNope; ++d) {
      q_ref[static_cast<size_t>(r) * kRow + d] = static_cast<double>(kops_test::to_float(qn_h[r * kNope + d]));
    }
    for (int d = 0; d < kPe; ++d) {
      q_ref[static_cast<size_t>(r) * kRow + kNope + d] =
          static_cast<double>(kops_test::to_float(qp_h[r * kPe + d]));
    }
  }
  for (int64_t t = 0; t < pool.total_kv; ++t) {
    for (int d = 0; d < kNope; ++d) {
      k_ref[static_cast<size_t>(t) * kRow + d] = pool.nope[static_cast<size_t>(t) * kNope + d];
      v_ref[static_cast<size_t>(t) * kNope + d] = pool.nope[static_cast<size_t>(t) * kNope + d];
    }
    for (int d = 0; d < kPe; ++d) {
      k_ref[static_cast<size_t>(t) * kRow + kNope + d] = pool.rope[static_cast<size_t>(t) * kPe + d];
    }
  }
  if (pool.total_kv == 0) {
    // attention_f64 rejects null k/v; a zero-length batch has nothing to read.
    k_ref.assign(static_cast<size_t>(kRow), 0.0);
    v_ref.assign(static_cast<size_t>(kNope), 0.0);
  }
  std::vector<int32_t> qo_indptr(static_cast<size_t>(B) + 1, 0), kv_indptr(static_cast<size_t>(B) + 1, 0);
  for (int b = 0; b < B; ++b) {
    qo_indptr[static_cast<size_t>(b) + 1] = qo_indptr[static_cast<size_t>(b)] + ql;
    kv_indptr[static_cast<size_t>(b) + 1] = kv_indptr[static_cast<size_t>(b)] + c.kv[static_cast<size_t>(b)];
  }
  std::vector<double> sink_ref(static_cast<size_t>(H), 0.0);
  for (int h = 0; h < H && c.sink; ++h) sink_ref[static_cast<size_t>(h)] = static_cast<double>(sink_f[static_cast<size_t>(h)]);
  std::vector<double> o_ref(static_cast<size_t>(total_q) * H * kNope, 0.0);
  std::vector<double> lse_ref(static_cast<size_t>(total_q) * H, 0.0);
  kops::ref::AttentionRefArgs ra{};
  ra.q = q_ref.data();
  ra.k = k_ref.data();
  ra.v = v_ref.data();
  ra.qo_indptr = qo_indptr.data();
  ra.kv_indptr = kv_indptr.data();
  ra.batch = B;
  ra.q_heads = H;
  ra.kv_heads = 1;
  ra.head_dim_qk = kRow;
  ra.head_dim_vo = kNope;
  ra.sm_scale = static_cast<double>(mod.sm_scale);
  ra.sink_logits = c.sink ? sink_ref.data() : nullptr;
  ra.o = o_ref.data();
  ra.lse = lse_ref.data();
  st = kops::ref::attention_f64(ra);
  if (st != Status::kSuccess) {
    kops_test::record_failure(c.name, kops::last_error_message(), __FILE__, __LINE__);
    return;
  }
  if (c.gate) {
    for (size_t i = 0; i < o_ref.size(); ++i) {
      o_ref[i] *= 1.0 / (1.0 + std::exp(-static_cast<double>(gate_f[i])));
    }
  }

  const std::vector<T> o_got = kops_test::copy_out(d_o, static_cast<size_t>(total_q) * H * kNope);
  const std::vector<float> lse_got = kops_test::copy_out(d_lse, static_cast<size_t>(total_q) * H);
  std::vector<float> lse_got_f(lse_got.begin(), lse_got.end());
  kops_test::compare(o_got, o_ref, atol, rtol, c.name);
  kops_test::compare(lse_got_f, lse_ref, 4e-3, 4e-3, c.name);

  kops_test::device_free(d_qn);
  kops_test::device_free(d_qp);
  kops_test::device_free(d_o);
  kops_test::device_free(d_lse);
  kops_test::device_free(d_gate);
  kops_test::device_free(d_sink);
  kops_test::device_free(d_table);
  kops_test::device_free(d_seq);
  kops_test::device_free(d_opart);
  kops_test::device_free(d_lpart);
  kops_test::device_free(d_indptr);
  kops_test::device_free(d_splits);
  cudaFree(d_cache);
}

void run_mla_case(const MlaCase& c, const kops_test::Options& opt) {
  if (c.full_tier && !kops_test::tier_enabled(opt, "full")) return;
  const double atol = c.fp8 ? 3e-2 : 2e-2;
  const double rtol = c.fp8 ? 3e-2 : 2e-2;
  if (c.fp16) {
    run_mla_case_t<__half>(c, atol, rtol);
  } else {
    run_mla_case_t<__nv_bfloat16>(c, atol, rtol);
  }
}

void test_dense_mla(const kops_test::Options& opt) {
  const std::vector<MlaCase> cases = {
      {"dense bf16 kv128 h64 pg16", 1, {128}, 64, 1, 16, false, false, false, false, 1},
      {"dense bf16 kv512 h128 pg16 sink", 1, {512}, 128, 1, 16, false, false, true, false, 1},
      {"dense bf16 kv512 h64 q4 pg32", 1, {512}, 64, 4, 32},
      {"dense bf16 ragged b3 h64 q2 pg64", 3, {64, 128, 257}, 64, 2, 64},
      {"dense bf16 empty+short h64", 2, {0, 96}, 64, 2, 16},
      {"dense bf16 empty+sink h64", 1, {0}, 64, 1, 16, false, false, true},
      {"dense bf16 head tail h40 kv64", 1, {64}, 40, 1, 16},
      {"dense bf16 q8 h32 kv64", 1, {64}, 32, 8, 16},
      {"dense bf16 drafts longer than kv", 1, {2}, 64, 4, 16},
      {"dense fp8 kv512 h128 q2 sink", 1, {512}, 128, 2, 16, true, false, true, false, 1},
      {"dense fp8 kv128 h64 pg64", 1, {128}, 64, 1, 64, true},
      {"dense fp16 kv128 h64", 1, {128}, 64, 1, 16, false, true},
      {"dense fp16 fp8 kv256 h128 q2", 1, {256}, 128, 2, 32, true, true},
      {"dense bf16 gate h64", 1, {128}, 64, 1, 16, false, false, false, true, 1},
      {"dense bf16 gate+sink h128", 1, {256}, 128, 1, 32, false, false, true, true, 1},
      {"dense split2 bf16 h64 kv512", 1, {512}, 64, 1, 16, false, false, false, false, 2},
      {"dense split4 bf16 h128 kv4096", 1, {4096}, 128, 1, 16, false, false, false, false, 4},
      {"dense split16 bf16 h128 kv4096", 1, {4096}, 128, 1, 16, false, false, false, false, 16},
      {"dense split16 bf16 h64 kv4103 q2", 1, {4103}, 64, 2, 16, false, false, false, false, 16},
      {"dense split32 bf16 h128 kv4096", 1, {4096}, 128, 1, 16, false, false, false, false, 32},
      {"dense split32 fp8 h128 kv4096", 1, {4096}, 128, 1, 16, true, false, false, false, 32},
      {"dense auto split bf16 h128 kv8192", 1, {8192}, 128, 1, 16, false, false, false, false, 0},
      {"dense auto split bf16 h128 kv8192 q4 sink", 1, {8192}, 128, 4, 16, false, false, true,
       false, 0},
      {"dense auto split fp8 h64 kv4096", 1, {4096}, 64, 1, 16, true, false, false, false, 0},
      {"dense auto split ragged b2", 2, {1024, 4096}, 64, 2, 32, false, false, false, false, 0},
      {"dense split4 fp8 h128 kv2048 sink", 1, {2048}, 128, 4, 64, true, false, true, false, 4},
      {"dense split2 fp16 ragged b2", 2, {256, 384}, 64, 2, 32, false, true, false, false, 2},
      {"dense split ragged counts b2", 2, {256, 384}, 64, 1, 16, false, false, false, false, 1,
       false, std::vector<int>{2, 3}},
      {"dense bf16 kv4096 h64", 1, {4096}, 64, 1, 16},
      {"dense bf16 kv4096 h128 q2", 1, {4096}, 128, 2, 16, false, false, true, false, 1, true},
  };
  for (const MlaCase& c : cases) run_mla_case(c, opt);
}

// =============================================================================
// 3. Contract checks
// =============================================================================

/// mla_decode_split_hint is a pure host function: its arithmetic must be stable across calls and
/// bounded by the requested cap and by the per-split key floor, and it must never require device
/// data. The auto-split cases above already prove the produced partials merge to the plain result.
void test_split_hint() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const int32_t units = kops::occupancy_compute_units(ctx);
  KOPS_TEST_CHECK(units >= 1);

  kops::attn::MLASplitHint h{};
  KOPS_EXPECT_OK(kops::attn::mla_decode_split_hint(1, 1, 128, 64 * 1024, 64, 2, ctx, &h),
                 "split hint 64k");
  KOPS_TEST_CHECK(h.splits >= 1 && h.splits <= 64);
  KOPS_TEST_CHECK(h.total_q == 1);
  KOPS_TEST_CHECK(h.splits == 1 || h.splits <= (64 * 1024) / 512);
  if (h.splits > 1) {
    const size_t want_o = 256u;  // aligned: 1 row * S * 128 heads * 512 floats * 4 B
    (void)want_o;
    KOPS_TEST_CHECK(h.o_partial_bytes ==
                    ((static_cast<size_t>(h.splits) * 128 * 512 * 4 + 255u) & ~size_t{255u}));
    KOPS_TEST_CHECK(h.lse_partial_bytes ==
                    ((static_cast<size_t>(h.splits) * 128 * 4 + 255u) & ~size_t{255u}));
    KOPS_TEST_CHECK(h.merge_indptr_bytes == 256u);
    KOPS_TEST_CHECK(h.total_bytes ==
                    h.o_partial_bytes + h.lse_partial_bytes + h.merge_indptr_bytes);
  }

  // A context that was never probed must still plan more than one CTA for a wide row.
  kops::LaunchContext unprobed = ctx;
  unprobed.compute_units = 0;
  kops::attn::MLASplitHint fallback{};
  KOPS_EXPECT_OK(
      kops::attn::mla_decode_split_hint(1, 1, 128, 1 << 20, 64, 2, unprobed, &fallback),
      "split hint unprobed");
  KOPS_TEST_CHECK(fallback.splits > 1);
  KOPS_TEST_CHECK(fallback.splits <= (1 << 20) / 512);

  // Monotone in the visible length and inverse in the head-tile count.
  kops::attn::MLASplitHint short_kv{};
  KOPS_EXPECT_OK(kops::attn::mla_decode_split_hint(1, 1, 128, 256, 64, 2, ctx, &short_kv),
                 "split hint short kv");
  KOPS_TEST_CHECK(short_kv.splits == 1);  // below the 512-key floor
  kops::attn::MLASplitHint q8{};
  KOPS_EXPECT_OK(kops::attn::mla_decode_split_hint(1, 8, 128, 64 * 1024, 64, 2, ctx, &q8),
                 "split hint q8");
  KOPS_TEST_CHECK(q8.splits <= h.splits);
  KOPS_TEST_CHECK(q8.total_q == 8);

  // Contract: the cap, the shape and the required arguments.
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_split_hint(1, 1, 128, 4096, 1, 2, ctx, &h),
                     Status::kSuccess, "split hint cap 1");
  KOPS_TEST_CHECK(h.splits == 1);
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_split_hint(1, 1, 128, 4096, 64, 2, ctx, nullptr),
                     Status::kInvalidArgument, "split hint null out");
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_split_hint(-1, 1, 128, 4096, 64, 2, ctx, &h),
                     Status::kInvalidArgument, "split hint bad batch");
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_split_hint(1, 9, 128, 4096, 64, 2, ctx, &h),
                     Status::kInvalidArgument, "split hint bad q_len");
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_split_hint(1, 1, 0, 4096, 64, 2, ctx, &h),
                     Status::kInvalidArgument, "split hint bad heads");
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_split_hint(1, 1, 128, -1, 64, 2, ctx, &h),
                     Status::kInvalidArgument, "split hint bad kv_len");
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_split_hint(1, 1, 128, 4096, 0, 2, ctx, &h),
                     Status::kInvalidArgument, "split hint bad max_splits");
  kops::LaunchContext hip = ctx;
  hip.backend = kops::BackendKind::kHip;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_split_hint(1, 1, 128, 4096, 64, 2, hip, &h),
                     Status::kUnsupportedBackend, "split hint HIP rejected");
}

void test_contract() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops::attn::AttentionSpec spec;
  spec.num_q_heads = 64;
  spec.num_kv_heads = 1;
  spec.head_dim_qk = kRow;
  spec.head_dim_vo = kNope;
  spec.q_dtype = spec.o_dtype = DType::kBFloat16;
  spec.kv_dtype = DType::kBFloat16;
  spec.kv_layout = kops::kv::KVLayout::kLatent;
  spec.page_size = 16;
  kops::attn::AttentionPlan plan;
  kops::Workspace ws{};
  KOPS_EXPECT_OK(kops::attn::mla_decode_plan(spec, 4, 2, ctx, ws, &plan), "mla_decode_plan ok");

  auto plan_status = [&](const kops::attn::AttentionSpec& s, int32_t batch, int32_t ql) {
    kops::attn::AttentionPlan p;
    return kops::attn::mla_decode_plan(s, batch, ql, ctx, ws, &p);
  };
  kops::attn::AttentionSpec bad = spec;
  bad.q_dtype = DType::kFloat32;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedDtype, "plan rejects fp32 q");
  bad = spec;
  bad.o_dtype = DType::kFloat16;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedDtype, "plan rejects q/o dtype mix");
  bad = spec;
  bad.head_dim_qk = 128;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedShape, "plan rejects dim 128");
  bad = spec;
  bad.head_dim_vo = 64;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedShape, "plan rejects vo 64");
  bad = spec;
  bad.k_codec = kops::kv::KVCodec::kFP8PerTensor;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kNotImplemented, "plan rejects per-tensor fp8");
  bad = spec;
  bad.v_codec = kops::kv::KVCodec::kMLAFp8DS;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedDtype, "plan rejects a V codec");
  bad = spec;
  bad.kv_dtype = DType::kFloat16;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedDtype, "plan rejects kv dtype mix");
  bad = spec;
  bad.kv_layout = kops::kv::KVLayout::kNHD;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedLayout, "plan rejects non-latent");
  bad = spec;
  bad.page_size = 8;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedShape, "plan rejects page 8");
  bad = spec;
  bad.mask = kops::attn::MaskMode::kNone;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedShape, "plan rejects full mask");
  bad = spec;
  bad.softcap = true;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedShape, "plan rejects softcap");
  bad = spec;
  bad.num_kv_heads = 2;
  KOPS_EXPECT_STATUS(plan_status(bad, 4, 1), Status::kUnsupportedShape, "plan rejects 2 kv heads");
  KOPS_EXPECT_STATUS(plan_status(spec, 0, 1), Status::kInvalidArgument, "plan rejects batch 0 cap");
  KOPS_EXPECT_STATUS(plan_status(spec, 4, 0), Status::kInvalidArgument, "plan rejects q_len 0");
  KOPS_EXPECT_STATUS(plan_status(spec, 4, 9), Status::kInvalidArgument, "plan rejects q_len 9");

  // run-side contract on the planned spec.
  const int H = 64, ql = 2, B = 1, ps = 16, kv = 32;
  const size_t qn_elems = static_cast<size_t>(B) * ql * H * kNope;
  __nv_bfloat16* d_qn = kops_test::device_alloc<__nv_bfloat16>(qn_elems);
  __nv_bfloat16* d_qp = kops_test::device_alloc<__nv_bfloat16>(static_cast<size_t>(B) * ql * H * kPe);
  __nv_bfloat16* d_o = kops_test::device_alloc<__nv_bfloat16>(qn_elems);
  void* d_cache = nullptr;
  int32_t* d_table = kops_test::device_alloc<int32_t>(static_cast<size_t>(B) * 2);
  int32_t* d_seq = kops_test::device_alloc<int32_t>(static_cast<size_t>(B));
  int32_t* d_indptr = kops_test::device_alloc<int32_t>(static_cast<size_t>(B * ql) + 1);
  float* d_part = kops_test::device_alloc<float>(1);
  float* d_lpart = kops_test::device_alloc<float>(1);
  if (d_qn == nullptr || d_qp == nullptr || d_o == nullptr || d_table == nullptr ||
      d_seq == nullptr || d_indptr == nullptr || d_part == nullptr || d_lpart == nullptr) {
    return;
  }
  if (cudaMalloc(&d_cache, static_cast<size_t>(2 * ps) * kRow * 2) != cudaSuccess) return;
  cudaMemset(d_cache, 0, static_cast<size_t>(2 * ps) * kRow * 2);
  std::vector<int32_t> tab = {0, 1}, seq = {kv}, ip = {0, 2, 4};
  kops_test::copy_in(d_table, tab);
  kops_test::copy_in(d_seq, seq);
  kops_test::copy_in(d_indptr, ip);

  kops::kv::PagedKVDesc kd{};
  kd.k_data = d_cache;
  kd.k_dtype = kd.v_dtype = DType::kBFloat16;
  kd.layout = kops::kv::KVLayout::kLatent;
  kd.page_size = ps;
  kd.num_kv_heads = 1;
  kd.head_dim_k = kRow;
  kd.head_dim_v = kNope;
  kd.stride_unit = kops::StrideUnit::kElement;
  kd.k_stride_page = static_cast<int64_t>(ps) * kRow;
  kd.k_stride_token = kRow;
  kd.k_stride_head = kRow;
  kd.table_format = kops::kv::PageTableFormat::kBlockTable;
  kd.block_table = d_table;
  kd.block_table_stride = 2;
  kd.seq_lens = d_seq;

  kops::attn::MLADecodeParams p{};
  p.q_nope = d_qn;
  p.q_nope_stride_n = static_cast<int64_t>(H) * kNope;
  p.q_nope_stride_h = kNope;
  p.q_pe = d_qp;
  p.q_pe_stride_n = static_cast<int64_t>(H) * kPe;
  p.q_pe_stride_h = kPe;
  p.q_len = ql;
  p.kv = kd;
  p.mod.sm_scale = 0.0f;
  p.o = d_o;
  p.o_stride_n = static_cast<int64_t>(H) * kNope;
  p.o_stride_h = kNope;
  p.lse = nullptr;
  p.batch_size = B;
  p.num_heads = H;
  p.kv_lora_rank = kNope;
  p.rope_dim = kPe;

  kops::attn::AttentionPlan bad_plan;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(bad_plan, p, ws, ctx), Status::kInvalidArgument,
                     "run rejects a foreign plan");
  KOPS_EXPECT_OK(kops::attn::mla_decode_run(plan, p, ws, ctx), "run accepts a valid call");

  kops::attn::MLADecodeParams q = p;
  q.batch_size = 5;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kCapacityExceeded,
                     "run rejects batch > plan");
  q = p;
  q.q_len = 1;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kInvalidArgument,
                     "run rejects a different q_len");
  q = p;
  q.num_heads = 128;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kInvalidArgument,
                     "run rejects a different head count");
  q = p;
  q.q_nope = nullptr;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kInvalidArgument,
                     "run rejects a null q");
  q = p;
  q.q_nope = reinterpret_cast<char*>(d_qn) + 2;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kMisaligned,
                     "run rejects a misaligned q");
  q = p;
  q.q_nope_stride_n = kNope;  // overlapping rows
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kUnsupportedLayout,
                     "run rejects overlapping q rows");
  q = p;
  q.kv.seq_lens = nullptr;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kInvalidArgument,
                     "run rejects a missing seq_lens");
  q = p;
  q.kv.table_format = kops::kv::PageTableFormat::kCSR;
  q.kv.kv_indptr = d_indptr;
  q.kv.kv_page_indices = d_table;
  q.kv.kv_last_page_len = d_seq;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kNotImplemented,
                     "run rejects the CSR table");
  q = p;
  q.kv.page_size = 32;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kUnsupportedShape,
                     "run rejects a page size the plan did not compile");
  q = p;
  q.split.max_splits = 2;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kInvalidArgument,
                     "run rejects split-KV without num_splits");
  q = p;
  q.split.max_splits = 2;
  q.num_splits = d_indptr;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kInvalidArgument,
                     "run rejects split-KV without partial buffers");
  q = p;
  q.mod.window_left = 32;
  KOPS_EXPECT_STATUS(kops::attn::mla_decode_run(plan, q, ws, ctx), Status::kUnsupportedShape,
                     "run rejects a sliding window");
  q = p;
  q.batch_size = 0;
  KOPS_EXPECT_OK(kops::attn::mla_decode_run(plan, q, ws, ctx), "run accepts an empty batch");

  kops_test::device_free(d_qn);
  kops_test::device_free(d_qp);
  kops_test::device_free(d_o);
  kops_test::device_free(d_table);
  kops_test::device_free(d_seq);
  kops_test::device_free(d_indptr);
  kops_test::device_free(d_part);
  kops_test::device_free(d_lpart);
  cudaFree(d_cache);
}

// =============================================================================
// 4. 64k-context consistency smoke: S=1 vs S=8 (no fp64 reference at 64k).
// =============================================================================
void test_long_split_consistency() {
  const int B = 1, H = 128, ql = 1, ps = 16, kv = 65536, S = 8;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const size_t qn = static_cast<size_t>(B) * ql * H * kNope;
  std::vector<float> qnf(qn), qpf(static_cast<size_t>(B) * ql * H * kPe);
  fill_uniform(&qnf, 91u);
  fill_uniform(&qpf, 92u, -0.5f, 0.5f);
  const auto qnh = cast_from_float<__nv_bfloat16>(qnf);
  const auto qph = cast_from_float<__nv_bfloat16>(qpf);
  __nv_bfloat16* d_qn = kops_test::device_alloc<__nv_bfloat16>(qnh.size());
  __nv_bfloat16* d_qp = kops_test::device_alloc<__nv_bfloat16>(qph.size());
  __nv_bfloat16* d_o1 = kops_test::device_alloc<__nv_bfloat16>(qn);
  __nv_bfloat16* d_o8 = kops_test::device_alloc<__nv_bfloat16>(qn);
  float* d_l1 = kops_test::device_alloc<float>(static_cast<size_t>(B) * ql * H);
  float* d_l8 = kops_test::device_alloc<float>(static_cast<size_t>(B) * ql * H);
  const int npages = kv / ps;
  void* d_cache = nullptr;
  int32_t* d_table = kops_test::device_alloc<int32_t>(npages);
  int32_t* d_seq = kops_test::device_alloc<int32_t>(1);
  float* d_part = kops_test::device_alloc<float>(static_cast<size_t>(B) * ql * S * H * kNope);
  float* d_lpart = kops_test::device_alloc<float>(static_cast<size_t>(B) * ql * S * H);
  int32_t* d_ind = kops_test::device_alloc<int32_t>(static_cast<size_t>(B) * ql + 1);
  int32_t* d_splits = kops_test::device_alloc<int32_t>(2);
  if (d_qn == nullptr || d_qp == nullptr || d_o1 == nullptr || d_o8 == nullptr || d_l1 == nullptr ||
      d_l8 == nullptr || d_table == nullptr || d_seq == nullptr || d_part == nullptr ||
      d_lpart == nullptr || d_ind == nullptr || d_splits == nullptr) {
    return;
  }
  if (cudaMalloc(&d_cache, static_cast<size_t>(npages) * ps * kRow * 2) != cudaSuccess) return;
  {
    std::vector<__nv_bfloat16> cache(static_cast<size_t>(npages) * ps * kRow);
    std::vector<float> f(cache.size());
    fill_uniform(&f, 93u);
    for (size_t i = 0; i < cache.size(); ++i) cache[i] = __float2bfloat16_rn(f[i]);
    cudaMemcpy(d_cache, cache.data(), cache.size() * 2, cudaMemcpyHostToDevice);
  }
  std::vector<int32_t> tab(static_cast<size_t>(npages));
  for (int i = 0; i < npages; ++i) tab[static_cast<size_t>(i)] = npages - 1 - i;
  std::vector<int32_t> seq = {kv}, ind = {0, S}, splits = {0, S};
  kops_test::copy_in(d_table, tab);
  kops_test::copy_in(d_seq, seq);
  kops_test::copy_in(d_ind, ind);
  kops_test::copy_in(d_splits, splits);

  kops::attn::AttentionSpec spec;
  spec.num_q_heads = H;
  spec.num_kv_heads = 1;
  spec.head_dim_qk = kRow;
  spec.head_dim_vo = kNope;
  spec.q_dtype = spec.o_dtype = DType::kBFloat16;
  spec.kv_dtype = DType::kBFloat16;
  spec.kv_layout = kops::kv::KVLayout::kLatent;
  spec.page_size = ps;
  spec.return_lse = true;
  kops::attn::AttentionPlan plan;
  kops::Workspace ws{};
  if (kops::attn::mla_decode_plan(spec, B, ql, ctx, ws, &plan) != Status::kSuccess) {
    kops_test::record_failure("64k plan", kops::last_error_message(), __FILE__, __LINE__);
    return;
  }
  kops::kv::PagedKVDesc kd{};
  kd.k_data = d_cache;
  kd.k_dtype = kd.v_dtype = DType::kBFloat16;
  kd.layout = kops::kv::KVLayout::kLatent;
  kd.page_size = ps;
  kd.num_kv_heads = 1;
  kd.head_dim_k = kRow;
  kd.head_dim_v = kNope;
  kd.stride_unit = kops::StrideUnit::kElement;
  kd.k_stride_page = static_cast<int64_t>(ps) * kRow;
  kd.k_stride_token = kRow;
  kd.k_stride_head = kRow;
  kd.table_format = kops::kv::PageTableFormat::kBlockTable;
  kd.block_table = d_table;
  kd.block_table_stride = npages;
  kd.seq_lens = d_seq;
  kops::attn::MLADecodeParams p{};
  p.q_nope = d_qn;
  p.q_nope_stride_n = static_cast<int64_t>(H) * kNope;
  p.q_nope_stride_h = kNope;
  p.q_pe = d_qp;
  p.q_pe_stride_n = static_cast<int64_t>(H) * kPe;
  p.q_pe_stride_h = kPe;
  p.q_len = ql;
  p.kv = kd;
  p.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(kRow));
  p.o = d_o1;
  p.o_stride_n = static_cast<int64_t>(H) * kNope;
  p.o_stride_h = kNope;
  p.lse = d_l1;
  p.batch_size = B;
  p.num_heads = H;
  p.kv_lora_rank = kNope;
  p.rope_dim = kPe;
  kops_test::copy_in(d_qn, qnh);
  kops_test::copy_in(d_qp, qph);
  if (kops::attn::mla_decode_run(plan, p, ws, ctx) != Status::kSuccess) {
    kops_test::record_failure("64k S=1", kops::last_error_message(), __FILE__, __LINE__);
    return;
  }
  kops::attn::MLADecodeParams ps8 = p;
  ps8.o = d_o8;
  ps8.lse = d_l8;
  ps8.split.o_partial = d_part;
  ps8.split.lse_partial = d_lpart;
  ps8.split.merge_indptr = d_ind;
  ps8.split.max_splits = S;
  ps8.num_splits = d_splits;
  if (kops::attn::mla_decode_run(plan, ps8, ws, ctx) != Status::kSuccess) {
    kops_test::record_failure("64k S=8", kops::last_error_message(), __FILE__, __LINE__);
    return;
  }
  merge_split_ref<__nv_bfloat16>(d_o8, d_l8, d_part, d_lpart, d_ind, nullptr, 1, H, kNope,
                                 reinterpret_cast<cudaStream_t>(ctx.stream));
  const auto o1 = kops_test::copy_out(d_o1, qn);
  const auto o8 = kops_test::copy_out(d_o8, qn);
  const auto l1 = kops_test::copy_out(d_l1, static_cast<size_t>(H));
  const auto l8 = kops_test::copy_out(d_l8, static_cast<size_t>(H));
  std::vector<double> ref(o1.size());
  for (size_t i = 0; i < o1.size(); ++i) ref[i] = static_cast<double>(kops_test::to_float(o1[i]));
  kops_test::compare(o8, ref, 2e-2, 2e-2, "64k split S=8 vs S=1");
  std::vector<double> lref(l1.size());
  for (size_t i = 0; i < l1.size(); ++i) lref[i] = static_cast<double>(l1[i]);
  kops_test::compare(l8, lref, 4e-3, 4e-3, "64k lse S=8 vs S=1");

  kops_test::device_free(d_qn);
  kops_test::device_free(d_qp);
  kops_test::device_free(d_o1);
  kops_test::device_free(d_o8);
  kops_test::device_free(d_l1);
  kops_test::device_free(d_l8);
  kops_test::device_free(d_table);
  kops_test::device_free(d_seq);
  kops_test::device_free(d_part);
  kops_test::device_free(d_lpart);
  kops_test::device_free(d_ind);
  kops_test::device_free(d_splits);
  cudaFree(d_cache);
}

}  // namespace

int main(int argc, char** argv) {
  const kops_test::Options opt = kops_test::parse_options(argc, argv);
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    std::printf("SKIP test_mla: no CUDA device\n");
    return 77;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || prop.major < 8) {
    std::printf("SKIP test_mla: needs sm_80+\n");
    return 77;
  }
  test_metadata();
  test_dense_mla(opt);
  test_split_hint();
  test_contract();
  if (kops_test::tier_enabled(opt, "full")) test_long_split_consistency();
  return kops_test::finish("test_mla");
}
