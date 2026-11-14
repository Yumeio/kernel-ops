// =============================================================================
// kernel-ops | benchmark/bench_mla.cu                                Catalog: A12
// Dense absorbed MLA decode (T1) and the FlashMLA split scheduler metadata.
//   mla_decode   B=1, H=128, q_len 1/4, latent cache 64k / 256k / 1M tokens,
//                native bf16 and kMLAFp8DS 656 B.
//                splits = 0 is AUTO: the case calls mla_decode_split_hint (two waves of
//                ctx.compute_units, >= 512 keys per split) and allocates exactly the partials
//                it reports, so the case ID is independent of the device it runs on.
//                splits > 0 forces that count: s8 / s16 / s32 / s64 (s64 only for 64k, the
//                deep end of the sweep) keep the pre-auto variants for A/B comparison.
//                Every split case measures the whole op: the split launch plus merge_split_kv
//                (the fp32 partial merge, and the sink when the case has one).
//                bytes = latent records read + q/o traffic (the decode roofline);
//                flops = the nominal 2*(512+576) MACs per (head, visible key),
//                i.e. the useful math, not the 4x QK redundancy of this variant.
//   mla_metadata small scheduler cases (FlashMLA get_decoding_sched_meta shape).
// The case table is the shared spec read by benchmark/compare/*.py via --dump-cases.
// =============================================================================
#include "benchmark/harness.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"

namespace {

using namespace kops_bench;
using kops::DType;
using kops::Status;

constexpr int64_t kNope = 512;
constexpr int64_t kPe = 64;
constexpr int64_t kRow = kNope + kPe;

struct MlaBench {
  const char* id;
  int batch, heads, kv, q, page, splits;  // splits 0 = auto (mla_decode_split_hint)
  bool fp8;
};

const std::vector<MlaBench>& mla_benches() {
  static const std::vector<MlaBench> b = {
      {"mla_decode.bf16.kv64k.q1", 1, 128, 1 << 16, 1, 16, 0, false},
      {"mla_decode.bf16.kv64k.q4", 1, 128, 1 << 16, 4, 16, 0, false},
      {"mla_decode.fp8.kv64k.q1", 1, 128, 1 << 16, 1, 16, 0, true},
      {"mla_decode.fp8.kv64k.q4", 1, 128, 1 << 16, 4, 16, 0, true},
      {"mla_decode.bf16.kv256k.q1", 1, 128, 1 << 18, 1, 16, 0, false},
      {"mla_decode.bf16.kv256k.q4", 1, 128, 1 << 18, 4, 16, 0, false},
      {"mla_decode.fp8.kv256k.q1", 1, 128, 1 << 18, 1, 16, 0, true},
      {"mla_decode.fp8.kv256k.q4", 1, 128, 1 << 18, 4, 16, 0, true},
      {"mla_decode.bf16.kv1m.q1", 1, 128, 1 << 20, 1, 16, 0, false},
      {"mla_decode.bf16.kv1m.q4", 1, 128, 1 << 20, 4, 16, 0, false},
      {"mla_decode.fp8.kv1m.q1", 1, 128, 1 << 20, 1, 16, 0, true},
      {"mla_decode.fp8.kv1m.q4", 1, 128, 1 << 20, 4, 16, 0, true},
      {"mla_decode.bf16.kv64k.q1.s8", 1, 128, 1 << 16, 1, 16, 8, false},
      {"mla_decode.bf16.kv256k.q1.s8", 1, 128, 1 << 18, 1, 16, 8, false},
      {"mla_decode.bf16.kv1m.q1.s8", 1, 128, 1 << 20, 1, 16, 8, false},
      {"mla_decode.fp8.kv64k.q1.s8", 1, 128, 1 << 16, 1, 16, 8, true},
      {"mla_decode.fp8.kv256k.q1.s8", 1, 128, 1 << 18, 1, 16, 8, true},
      {"mla_decode.fp8.kv1m.q1.s8", 1, 128, 1 << 20, 1, 16, 8, true},
      {"mla_decode.bf16.kv1m.q4.s8", 1, 128, 1 << 20, 4, 16, 8, false},
      {"mla_decode.fp8.kv1m.q4.s8", 1, 128, 1 << 20, 4, 16, 8, true},
      // Deeper split-KV: 8 head tiles x S CTAs; S=16/32 fill the 36-SM device with
      // one 16-head tile per SM and shorten the per-CTA key range.
      {"mla_decode.bf16.kv64k.q1.s16", 1, 128, 1 << 16, 1, 16, 16, false},
      {"mla_decode.bf16.kv64k.q1.s32", 1, 128, 1 << 16, 1, 16, 32, false},
      {"mla_decode.bf16.kv64k.q1.s64", 1, 128, 1 << 16, 1, 16, 64, false},
      {"mla_decode.bf16.kv256k.q1.s16", 1, 128, 1 << 18, 1, 16, 16, false},
      {"mla_decode.bf16.kv256k.q1.s32", 1, 128, 1 << 18, 1, 16, 32, false},
      {"mla_decode.bf16.kv1m.q1.s16", 1, 128, 1 << 20, 1, 16, 16, false},
      {"mla_decode.bf16.kv1m.q1.s32", 1, 128, 1 << 20, 1, 16, 32, false},
      {"mla_decode.fp8.kv64k.q1.s16", 1, 128, 1 << 16, 1, 16, 16, true},
      {"mla_decode.fp8.kv64k.q1.s32", 1, 128, 1 << 16, 1, 16, 32, true},
      {"mla_decode.fp8.kv64k.q1.s64", 1, 128, 1 << 16, 1, 16, 64, true},
      {"mla_decode.fp8.kv1m.q1.s16", 1, 128, 1 << 20, 1, 16, 16, true},
      {"mla_decode.fp8.kv1m.q1.s32", 1, 128, 1 << 20, 1, 16, 32, true},
  };
  return b;
}

const MlaBench* find_mla(const std::string& id) {
  for (const MlaBench& b : mla_benches()) {
    if (id == b.id) return &b;
  }
  return nullptr;
}

struct MetaBench {
  const char* id;
  int batch, parts, block, q_heads;
};

const std::vector<MetaBench>& meta_benches() {
  static const std::vector<MetaBench> b = {
      {"mla_metadata.b8.parts132.block64", 8, 132, 64, 128},
      {"mla_metadata.b32.parts132.block64", 32, 132, 64, 128},
      {"mla_metadata.b128.parts192.block64", 128, 192, 64, 128},
      {"mla_metadata.b32.parts132.block16", 32, 132, 16, 64},
  };
  return b;
}

const MetaBench* find_meta(const std::string& id) {
  for (const MetaBench& b : meta_benches()) {
    if (id == b.id) return &b;
  }
  return nullptr;
}

__global__ void bench_fill_bf16(__nv_bfloat16* p, int64_t n, uint32_t seed) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    p[i] = __float2bfloat16(static_cast<float>(h & 0xFFFF) / 32768.0f - 1.0f);
  }
}

}  // namespace

void add_mla_cases(std::vector<kops_bench::Case>& cases) {
  using namespace kops_bench;
  for (const MlaBench& b : mla_benches()) {
    const int64_t records = static_cast<int64_t>(b.batch) * b.kv;
    const int64_t rec = b.fp8 ? 656 : kRow * 2;
    const int64_t bytes = records * rec + static_cast<int64_t>(b.batch) * b.q * b.heads * kRow * 2 +
                          static_cast<int64_t>(b.batch) * b.q * b.heads * kNope * 2 +
                          static_cast<int64_t>(b.batch) * b.q * b.heads * 4;
    const int64_t pairs = static_cast<int64_t>(b.batch) * b.heads *
                          (static_cast<int64_t>(b.q) * b.kv - static_cast<int64_t>(b.q - 1) * b.q / 2);
    const int64_t flops = 2 * pairs * (kNope + kRow);
    cases.push_back({b.id, "mla_decode", b.fp8 ? DType::kFloat8E4M3 : DType::kBFloat16,
                     static_cast<int64_t>(b.batch) * b.q, kRow, bytes, flops, 3, 30});
  }
  for (const MetaBench& m : meta_benches()) {
    cases.push_back({m.id, "mla_metadata", DType::kInt32, m.batch, 8,
                     static_cast<int64_t>(m.parts) * 8 * 4 + static_cast<int64_t>(m.batch + 1) * 4,
                     0, 10, 200});
  }
}

namespace {

bool run_mla_decode(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const MlaBench* b = find_mla(c.id);
  if (b == nullptr) return false;
  const int B = b->batch, H = b->heads, ql = b->q, ps = b->page;
  const int64_t total_q = static_cast<int64_t>(B) * ql;
  const int64_t rec = b->fp8 ? 656 : kRow * 2;
  const int64_t npages = (static_cast<int64_t>(b->kv) + ps - 1) / ps;
  const int64_t cache_bytes = npages * ps * rec;

  // split count: 0 = auto, resolved on the host from the case shape and ctx (no device read).
  kops::attn::MLASplitHint hint{};
  bool ok = true;
  if (b->splits == 0) {
    ok = status_ok(kops::attn::mla_decode_split_hint(B, ql, H, b->kv, /*max_splits=*/64,
                                                    /*waves=*/2, ctx, &hint),
                   c.id.c_str());
    if (!ok) return false;
  }
  const int S = b->splits == 0 ? hint.splits : b->splits;
  const size_t o_partial_bytes =
      b->splits == 0 ? hint.o_partial_bytes
                     : static_cast<size_t>(total_q) * S * H * kNope * sizeof(float);
  const size_t lse_partial_bytes =
      b->splits == 0 ? hint.lse_partial_bytes
                     : static_cast<size_t>(total_q) * S * H * sizeof(float);
  const size_t indptr_bytes = b->splits == 0
                                  ? hint.merge_indptr_bytes
                                  : (static_cast<size_t>(total_q) + 1) * sizeof(int32_t);

  __nv_bfloat16* qn = nullptr;
  __nv_bfloat16* qp = nullptr;
  __nv_bfloat16* o = nullptr;
  float* lse = nullptr;
  float* opart = nullptr;
  float* lpart = nullptr;
  int32_t* indptr = nullptr;
  int32_t* splits = nullptr;
  void* cache = nullptr;
  int32_t* table = nullptr;
  int32_t* seq = nullptr;
  ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&qn),
                          static_cast<size_t>(total_q) * H * kNope * 2),
               "q_nope") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void**>(&qp),
                          static_cast<size_t>(total_q) * H * kPe * 2),
               "q_pe") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void**>(&o),
                          static_cast<size_t>(total_q) * H * kNope * 2),
               "o") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void**>(&lse),
                          static_cast<size_t>(total_q) * H * 4),
               "lse") &&
       cuda_ok(cudaMalloc(&cache, static_cast<size_t>(cache_bytes)), "cache") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void**>(&table), npages * 4), "block_table") &&
       cuda_ok(cudaMalloc(reinterpret_cast<void**>(&seq), B * 4), "seq_lens");
  if (!ok) return false;
  if (S > 1) {
    ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&opart), o_partial_bytes), "o_partial") &&
         cuda_ok(cudaMalloc(reinterpret_cast<void**>(&lpart), lse_partial_bytes), "lse_partial") &&
         cuda_ok(cudaMalloc(reinterpret_cast<void**>(&indptr), indptr_bytes), "merge_indptr") &&
         cuda_ok(cudaMalloc(reinterpret_cast<void**>(&splits), (B + 1) * 4), "num_splits");
    if (!ok) return false;
  }
  bench_fill_bf16<<<256, 256>>>(qn, total_q * H * kNope, 1u);
  bench_fill_bf16<<<256, 256>>>(qp, total_q * H * kPe, 2u);
  cudaMemset(cache, b->fp8 ? 0x38 : 0x3c, static_cast<size_t>(cache_bytes));
  std::vector<int32_t> tab(static_cast<size_t>(npages)), seqv(static_cast<size_t>(B), b->kv);
  for (int64_t i = 0; i < npages; ++i) tab[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  cudaMemcpy(table, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(seq, seqv.data(), seqv.size() * 4, cudaMemcpyHostToDevice);
  std::vector<int32_t> ind(static_cast<size_t>(total_q) + 1, 0), spl(static_cast<size_t>(B) + 1, 0);
  if (S > 1) {
    for (int64_t r = 0; r <= total_q; ++r) ind[static_cast<size_t>(r)] = static_cast<int32_t>(r * S);
    for (int i = 0; i <= B; ++i) spl[static_cast<size_t>(i)] = i * S;
    cudaMemcpy(indptr, ind.data(), ind.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(splits, spl.data(), spl.size() * 4, cudaMemcpyHostToDevice);
  }

  kops::attn::AttentionSpec spec;
  spec.num_q_heads = H;
  spec.num_kv_heads = 1;
  spec.head_dim_qk = static_cast<int32_t>(kRow);
  spec.head_dim_vo = static_cast<int32_t>(kNope);
  spec.q_dtype = spec.o_dtype = DType::kBFloat16;
  spec.kv_dtype = b->fp8 ? DType::kFloat8E4M3 : DType::kBFloat16;
  spec.k_codec = b->fp8 ? kops::kv::KVCodec::kMLAFp8DS : kops::kv::KVCodec::kNative;
  spec.v_codec = kops::kv::KVCodec::kNative;
  spec.kv_layout = kops::kv::KVLayout::kLatent;
  spec.page_size = ps;
  spec.return_lse = true;

  kops::attn::AttentionPlan plan;
  kops::Workspace ws{};
  ok = status_ok(kops::attn::mla_decode_plan(spec, B, ql, ctx, ws, &plan), c.id.c_str());
  if (!ok) return false;

  kops::kv::PagedKVDesc kd{};
  kd.k_data = cache;
  kd.k_dtype = b->fp8 ? DType::kFloat8E4M3 : DType::kBFloat16;
  kd.v_dtype = DType::kBFloat16;
  kd.k_codec = b->fp8 ? kops::kv::KVCodec::kMLAFp8DS : kops::kv::KVCodec::kNative;
  kd.v_codec = kops::kv::KVCodec::kNative;
  kd.layout = kops::kv::KVLayout::kLatent;
  kd.page_size = ps;
  kd.num_kv_heads = 1;
  kd.head_dim_k = static_cast<int32_t>(kRow);
  kd.head_dim_v = static_cast<int32_t>(kNope);
  kd.entry_bytes = b->fp8 ? 656 : 0;
  kd.stride_unit = b->fp8 ? kops::StrideUnit::kByte : kops::StrideUnit::kElement;
  const int64_t tok = b->fp8 ? 656 : kRow;
  kd.k_stride_page = tok * ps;
  kd.k_stride_token = tok;
  kd.k_stride_head = tok;
  kd.table_format = kops::kv::PageTableFormat::kBlockTable;
  kd.block_table = table;
  kd.block_table_stride = static_cast<int32_t>(npages);
  kd.seq_lens = seq;

  kops::attn::MLADecodeParams p{};
  p.q_nope = qn;
  p.q_nope_stride_n = static_cast<int64_t>(H) * kNope;
  p.q_nope_stride_h = kNope;
  p.q_pe = qp;
  p.q_pe_stride_n = static_cast<int64_t>(H) * kPe;
  p.q_pe_stride_h = kPe;
  p.q_len = ql;
  p.kv = kd;
  p.mod.sm_scale = 1.0f / std::sqrt(static_cast<float>(kRow));
  p.o = o;
  p.o_stride_n = static_cast<int64_t>(H) * kNope;
  p.o_stride_h = kNope;
  p.lse = lse;
  p.split.o_partial = opart;
  p.split.lse_partial = lpart;
  p.split.merge_indptr = indptr;
  p.split.max_splits = S;
  p.num_splits = splits;
  p.batch_size = B;
  p.num_heads = H;
  p.kv_lora_rank = static_cast<int32_t>(kNope);
  p.rope_dim = static_cast<int32_t>(kPe);

  // Split cases measure the whole op: the split launch plus merge_split_kv, which folds the fp32
  // partials (and the sink, absent here) back into o/lse. The merge is a 512-wide fp32 read of
  // total_q * S * H rows, i.e. negligible next to the latent scan it combines.
  const kops::Tensor t_o = [&] {
    kops::Tensor t{};
    t.data = o;
    t.dtype = DType::kBFloat16;
    t.ndim = 3;
    t.device = ctx.device;
    t.shape[0] = total_q;
    t.shape[1] = H;
    t.shape[2] = kNope;
    t.stride[0] = static_cast<int64_t>(H) * kNope;
    t.stride[1] = kNope;
    t.stride[2] = 1;
    return t;
  }();
  const kops::Tensor t_lse = [&] {
    kops::Tensor t{};
    t.data = lse;
    t.dtype = DType::kFloat32;
    t.ndim = 2;
    t.device = ctx.device;
    t.shape[0] = total_q;
    t.shape[1] = H;
    t.stride[0] = H;
    t.stride[1] = 1;
    return t;
  }();
  const kops::attn::SplitKVBuffers merge_split{p.split.o_partial, p.split.lse_partial,
                                               p.split.merge_indptr, S};
  auto call = [&]() {
    kops::Status st = kops::attn::mla_decode_run(plan, p, ws, ctx);
    if (st == kops::Status::kSuccess && S > 1) {
      st = kops::attn::merge_split_kv(t_o, t_lse, merge_split, kops::Tensor{}, ctx);
    }
    return st;
  };
  ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  for (void* ptr : {static_cast<void*>(qn), static_cast<void*>(qp), static_cast<void*>(o),
                    static_cast<void*>(lse), static_cast<void*>(opart),
                    static_cast<void*>(lpart), static_cast<void*>(indptr),
                    static_cast<void*>(splits), cache, static_cast<void*>(table),
                    static_cast<void*>(seq)}) {
    cudaFree(ptr);
  }
  return ok;
}

bool run_mla_metadata(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const MetaBench* m = find_meta(c.id);
  if (m == nullptr) return false;
  int32_t* meta = nullptr;
  int32_t* splits = nullptr;
  int32_t* seq = nullptr;
  bool ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&meta), static_cast<size_t>(m->parts) * 8 * 4),
                    "metadata") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&splits), static_cast<size_t>(m->batch + 1) * 4),
                    "num_splits") &&
            cuda_ok(cudaMalloc(reinterpret_cast<void**>(&seq), static_cast<size_t>(m->batch) * 4),
                    "seq_lens");
  if (!ok) return false;
  std::vector<int32_t> lens(static_cast<size_t>(m->batch));
  for (int i = 0; i < m->batch; ++i) lens[static_cast<size_t>(i)] = 1024 + (i % 7) * 4096;
  cudaMemcpy(seq, lens.data(), lens.size() * 4, cudaMemcpyHostToDevice);
  const kops::Tensor t_meta = tensor2d(meta, DType::kInt32, m->parts, 8);
  kops::Tensor t_splits{};
  t_splits.data = splits;
  t_splits.dtype = DType::kInt32;
  t_splits.ndim = 1;
  t_splits.device = 0;
  t_splits.shape[0] = m->batch + 1;
  t_splits.stride[0] = 1;
  kops::Tensor t_seq{};
  t_seq.data = seq;
  t_seq.dtype = DType::kInt32;
  t_seq.ndim = 1;
  t_seq.device = 0;
  t_seq.shape[0] = m->batch;
  t_seq.stride[0] = 1;
  ok = status_ok(kops::attn::mla_get_metadata(t_meta, t_splits, t_seq, m->q_heads, m->parts,
                                              m->block, 2, ctx),
                 c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] {
          (void)kops::attn::mla_get_metadata(t_meta, t_splits, t_seq, m->q_heads, m->parts, m->block,
                                             2, ctx);
        },
        st);
  }
  cudaFree(meta);
  cudaFree(splits);
  cudaFree(seq);
  return ok;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "mla_decode") return run_mla_decode(c, ctx, st);
  if (c.op == "mla_metadata") return run_mla_metadata(c, ctx, st);
  return false;
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  add_mla_cases(cases);
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
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) {
    return 1;
  }
  std::vector<Result> results;
  results.reserve(cases.size());
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
