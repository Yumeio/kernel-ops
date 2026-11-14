// Dev-only baseline: attention.rs TurboQuant low-bit KV store/decode/prefill (3rd/attention.rs,
// MIT). Never built by CMake, never linked into kops. It exists only to time the upstream
// flash_tq4/flash_tq3 kernels on the same GPU and shape as `bench_merge --only lowbit`.
//
//   tools/dev/build_lowbit_baselines.sh
//   build/compare/attention_rs_lowbit --flush-mb 256 --rounds 3 --json out/b17/attention-rs-lowbit.json
//
// Workloads (identical to the kops rows in benchmark/bench_merge.cu lb_benches()):
//   lowbit.store.bf16.t{2048,4096}.h8.d128.*      : 1 sequence, T tokens x Hkv=8 heads, page 16
//   lowbit.decode.bf16.d128.b8.kv4096.h32x8.*      : B=8 sequences, q=1, kv=4096, Hq=32, Hkv=8
//   lowbit.prefill.bf16.d128.t512.kv4096.h32x8.*   : B=1 sequence, q=512 (bottom-right causal),
//                                                    kv=4096
// Launch grids follow the upstream launchers in flash_instantiate.cu: store is
// dim3(num_tokens, num_kv_heads) (head = blockIdx.y), decode dim3(Hq, num_seqs), prefill
// dim3(Hq, ceil(q/32), num_seqs). Timing follows benchmark/harness.h (see ext_timing.h).
// The upstream format stores K absmax and V absmax as separate float arrays and packs 4-bit
// K/V (or 3-bit K + 4-bit V).
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ext_timing.h"

#define FLASH_HDIM 128
#include "flash_turboquant.cuh"
#include "flash_turboquant_lowbit.cuh"
#include "flash_prefill_tq4.cuh"
#include "flash_prefill_tq3.cuh"

namespace {

constexpr int kB = 8, kKV = 4096, kHq = 32, kHkv = 8, kD = FLASH_HDIM, kPage = 16;
constexpr int kPagesPerSeq = kKV / kPage;
constexpr int kPages = kB * kPagesPerSeq;
constexpr int kAllTokens = kB * kKV;    // cache fill for the decode rows (setup, untimed)
constexpr int kPrefillB = 1, kQ = 512;  // kops prefill case: batch 1, q 512, kv 4096

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;
  Buf(size_t n, size_t elem = 1) : bytes(n * elem) {
    if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess) {
      std::fprintf(stderr, "cudaMalloc(%zu) failed\n", bytes);
      std::exit(1);
    }
    cudaMemset(p, 0, bytes ? bytes : 1);
  }
  ~Buf() { cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

void fill_bf16(__nv_bfloat16* p, size_t n, unsigned seed) {
  std::vector<__nv_bfloat16> h(n);
  unsigned x = seed;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    h[i] = __float2bfloat16(static_cast<float>(x >> 8) / 8388608.0f - 1.0f);
  }
  cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice);
}

/// Every (token, head) absmax slot of the first `tokens` tokens must have been written.
/// The buffers are zeroed before the store, and a random bf16 K/V row always has a nonzero
/// absmax, so a zero slot proves the grid never reached that (token, head).
bool check_absmax(const Buf& am, int tokens, const char* what) {
  std::vector<float> h(static_cast<size_t>(tokens) * kHkv);
  if (cudaMemcpy(h.data(), am.p, h.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "%s: absmax copy failed\n", what);
    return false;
  }
  size_t zero = 0;
  for (float v : h) zero += (v == 0.0f);
  if (zero != 0) {
    std::fprintf(stderr, "%s: %zu of %zu (token, head) absmax slots not written\n", what, zero,
                 h.size());
    return false;
  }
  return true;
}

/// The packed payload of every (token, head) row must contain at least one nonzero code.
/// This catches a grid that writes one slot per row but skips the per-lane packing.
bool check_payload(const Buf& q, int tokens, int row_bytes, const char* what) {
  const size_t rows = static_cast<size_t>(tokens) * kHkv;
  std::vector<unsigned char> h(rows * static_cast<size_t>(row_bytes));
  if (cudaMemcpy(h.data(), q.p, h.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "%s: payload copy failed\n", what);
    return false;
  }
  size_t empty = 0;
  for (size_t r = 0; r < rows; ++r) {
    size_t nonzero = 0;
    for (int j = 0; j < row_bytes; ++j) nonzero += (h[r * row_bytes + j] != 0);
    empty += (nonzero == 0);
  }
  if (empty != 0) {
    std::fprintf(stderr, "%s: %zu of %zu (token, head) payload rows are all zero\n", what, empty,
                 rows);
    return false;
  }
  return true;
}

/// A decode over a populated cache must write a finite, nonzero O for every row.
bool check_decode_out(const Buf& o, size_t n, const char* what) {
  std::vector<unsigned short> h(n);
  if (cudaMemcpy(h.data(), o.p, n * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "%s: output copy failed\n", what);
    return false;
  }
  size_t nonzero = 0, nonfinite = 0;
  for (unsigned short bits : h) {
    if ((bits & 0x7f80u) == 0x7f80u) ++nonfinite;  // NaN or Inf
    if ((bits & 0x7fffu) != 0) ++nonzero;
  }
  if (nonfinite != 0 || nonzero == 0) {
    std::fprintf(stderr, "%s: %zu/%zu nonzero, %zu nonfinite\n", what, nonzero, n, nonfinite);
    return false;
  }
  std::printf("%s: %zu/%zu outputs nonzero, all finite\n", what, nonzero, n);
  return true;
}

bool launch_ok(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const ext_bench::Options opt = ext_bench::parse_options(argc, argv);
  ext_bench::Flusher flush(opt.flush_mb);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kD));

  Buf q(static_cast<size_t>(kB) * kHq * kD, 2), o(static_cast<size_t>(kB) * kHq * kD, 2);
  Buf key(static_cast<size_t>(kAllTokens) * kHkv * kD, 2);
  Buf val(static_cast<size_t>(kAllTokens) * kHkv * kD, 2);
  fill_bf16(static_cast<__nv_bfloat16*>(q.p), static_cast<size_t>(kB) * kHq * kD, 7);
  fill_bf16(static_cast<__nv_bfloat16*>(key.p), static_cast<size_t>(kAllTokens) * kHkv * kD, 11);
  fill_bf16(static_cast<__nv_bfloat16*>(val.p), static_cast<size_t>(kAllTokens) * kHkv * kD, 13);

  // Prefill: one sequence (kops case batch=1); cu_seqlens = {0, q}, context = kv. The kernel's
  // grid z is the sequence index, so z = kPrefillB.
  Buf qf(static_cast<size_t>(kPrefillB) * kQ * kHq * kD, 2);
  Buf of(static_cast<size_t>(kPrefillB) * kQ * kHq * kD, 2);
  Buf cu(kPrefillB + 1, sizeof(unsigned)), ctxl(kPrefillB, sizeof(unsigned));
  {
    std::vector<unsigned> hc(kPrefillB + 1), hl(kPrefillB, kKV);
    for (int b = 0; b <= kPrefillB; ++b) hc[b] = static_cast<unsigned>(b) * kQ;
    cudaMemcpy(cu.p, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ctxl.p, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice);
  }
  fill_bf16(static_cast<__nv_bfloat16*>(qf.p), static_cast<size_t>(kPrefillB) * kQ * kHq * kD, 17);
  // Sequence b owns pages [b*kPagesPerSeq, (b+1)*kPagesPerSeq). The store rows use the first T
  // tokens of sequence 0, which is exactly the kops store case (batch 1, pages 0..T/16-1).
  Buf slot(kAllTokens, sizeof(long long));
  {
    std::vector<long long> h(kAllTokens);
    for (int b = 0; b < kB; ++b)
      for (int t = 0; t < kKV; ++t)
        h[static_cast<size_t>(b) * kKV + t] =
            static_cast<long long>(b * kPagesPerSeq + t / kPage) * kPage + t % kPage;
    cudaMemcpy(slot.p, h.data(), h.size() * sizeof(long long), cudaMemcpyHostToDevice);
  }
  Buf block_table(kB * kPagesPerSeq, sizeof(int));
  Buf seq_lens(kB, sizeof(int));
  {
    std::vector<int> bt(kB * kPagesPerSeq), sl(kB, kKV);
    for (int b = 0; b < kB; ++b)
      for (int i = 0; i < kPagesPerSeq; ++i)
        bt[static_cast<size_t>(b) * kPagesPerSeq + i] = b * kPagesPerSeq + i;
    cudaMemcpy(block_table.p, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(seq_lens.p, sl.data(), sl.size() * 4, cudaMemcpyHostToDevice);
  }

  // Upstream layout: [pages, page, Hkv] absmax + [pages, page, Hkv, D/2] payload.
  const size_t am_n = static_cast<size_t>(kPages) * kPage * kHkv;
  const size_t q4_n = am_n * (kD / 2);
  const size_t q3_n = am_n * ((kD * 3 + 7) / 8);
  Buf k4_am(am_n, 4), v4_am(am_n, 4), k4(q4_n), v4(q4_n);
  Buf k3_am(am_n, 4), v3_am(am_n, 4), k3(q3_n), v3(q4_n);

  auto store4 = [&](int tokens) {
    flash_tq4_store<__nv_bfloat16><<<dim3(tokens, kHkv), 32>>>(
        static_cast<const __nv_bfloat16*>(key.p), static_cast<const __nv_bfloat16*>(val.p),
        static_cast<float*>(k4_am.p), static_cast<unsigned char*>(k4.p),
        static_cast<float*>(v4_am.p), static_cast<unsigned char*>(v4.p),
        static_cast<const long long*>(slot.p), tokens, kHkv, kD, kPage);
  };
  auto store3 = [&](int tokens) {
    flash_tq3_store<__nv_bfloat16><<<dim3(tokens, kHkv), 32>>>(
        static_cast<const __nv_bfloat16*>(key.p), static_cast<const __nv_bfloat16*>(val.p),
        static_cast<float*>(k3_am.p), static_cast<unsigned char*>(k3.p),
        static_cast<float*>(v3_am.p), static_cast<unsigned char*>(v3.p),
        static_cast<const long long*>(slot.p), tokens, kHkv, kD, kPage);
  };
  auto decode4 = [&] {
    flash_tq4_decode<__nv_bfloat16><<<dim3(kHq, kB), 256>>>(
        static_cast<const __nv_bfloat16*>(q.p), static_cast<const float*>(k4_am.p),
        static_cast<const unsigned char*>(k4.p), static_cast<const float*>(v4_am.p),
        static_cast<const unsigned char*>(v4.p), static_cast<__nv_bfloat16*>(o.p),
        static_cast<const int*>(block_table.p), static_cast<const int*>(seq_lens.p),
        kPagesPerSeq, kHq, kHkv, kD, kPage, scale, kB, kHq * kD, 0.0f, 0u);
  };
  auto decode3 = [&] {
    flash_tq3_decode<__nv_bfloat16><<<dim3(kHq, kB), 256>>>(
        static_cast<const __nv_bfloat16*>(q.p), static_cast<const float*>(k3_am.p),
        static_cast<const unsigned char*>(k3.p), static_cast<const float*>(v3_am.p),
        static_cast<const unsigned char*>(v3.p), static_cast<__nv_bfloat16*>(o.p),
        static_cast<const int*>(block_table.p), static_cast<const int*>(seq_lens.p),
        kPagesPerSeq, kHq, kHkv, kD, kPage, scale, kB, kHq * kD, 0.0f, 0u);
  };
  // Upstream launch configurations from flash_instantiate.cu::call_flash_tq{4,3}_*:
  //   store   dim3(num_tokens, num_kv_heads), 32 threads
  //   decode  dim3(num_q_heads, num_seqs), 256 threads
  //   prefill dim3(num_q_heads, ceil(max_q_len/32), num_seqs), 128 threads (head_dim 128)
  const dim3 prefill_grid(kHq, (kQ + 31) / 32, kPrefillB);
  std::printf("grids: store (%d,%d)/32 timed, (%d,%d)/32 setup; decode (%d,%d)/256; "
              "prefill (%d,%d,%d)/128\n",
              2048, kHkv, kAllTokens, kHkv, kHq, kB, kHq, (kQ + 31) / 32, kPrefillB);
  auto prefill4 = [&] {
    flash_tq4_prefill<__nv_bfloat16><<<prefill_grid, 128>>>(
        static_cast<const __nv_bfloat16*>(qf.p), static_cast<const float*>(k4_am.p),
        static_cast<const unsigned char*>(k4.p), static_cast<const float*>(v4_am.p),
        static_cast<const unsigned char*>(v4.p), static_cast<__nv_bfloat16*>(of.p),
        static_cast<const int*>(block_table.p), kPagesPerSeq,
        static_cast<const unsigned*>(cu.p), static_cast<const unsigned*>(ctxl.p), kHq, kHkv, kD,
        kPage, 0u, 1u, scale, 0.0f);
  };
  auto prefill3 = [&] {
    flash_tq3_prefill<__nv_bfloat16><<<prefill_grid, 128>>>(
        static_cast<const __nv_bfloat16*>(qf.p), static_cast<const float*>(k3_am.p),
        static_cast<const unsigned char*>(k3.p), static_cast<const float*>(v3_am.p),
        static_cast<const unsigned char*>(v3.p), static_cast<__nv_bfloat16*>(of.p),
        static_cast<const int*>(block_table.p), kPagesPerSeq,
        static_cast<const unsigned*>(cu.p), static_cast<const unsigned*>(ctxl.p), kHq, kHkv, kD,
        kPage, 0u, 1u, scale, 0.0f);
  };

  // Setup (untimed): fill both caches for all B x kv tokens and every KV head, then verify
  // that no (token, head) absmax or payload row was skipped. This is the semantic proof
  // that the (num_tokens, num_kv_heads) grid covers the whole cache, not just the first
  // CTA row.
  store4(kAllTokens);
  store3(kAllTokens);
  if (cudaDeviceSynchronize() != cudaSuccess || !launch_ok("store setup")) {
    std::fprintf(stderr, "store launch failed\n");
    return 1;
  }
  const bool covered =
      check_absmax(k4_am, kAllTokens, "turbo4 K absmax") &&
      check_absmax(v4_am, kAllTokens, "turbo4 V absmax") &&
      check_absmax(k3_am, kAllTokens, "turbo3 K absmax") &&
      check_absmax(v3_am, kAllTokens, "turbo3 V absmax") &&
      check_payload(k4, kAllTokens, kD / 2, "turbo4 K payload") &&
      check_payload(v4, kAllTokens, kD / 2, "turbo4 V payload") &&
      check_payload(k3, kAllTokens, (kD * 3 + 7) / 8, "turbo3 K payload") &&
      check_payload(v3, kAllTokens, kD / 2, "turbo3 V payload");
  if (!covered) return 1;
  std::printf("store coverage: all %d tokens x %d KV heads written (turbo4, turbo3 absmax+payload)\n",
              kAllTokens, kHkv);

  // Decode semantics: one decode over the populated cache must produce finite, nonzero O.
  decode4();
  if (cudaDeviceSynchronize() != cudaSuccess || !launch_ok("turbo4 decode check")) return 1;
  if (!check_decode_out(o, static_cast<size_t>(kB) * kHq * kD, "turbo4 decode")) return 1;
  decode3();
  if (cudaDeviceSynchronize() != cudaSuccess || !launch_ok("turbo3 decode check")) return 1;
  if (!check_decode_out(o, static_cast<size_t>(kB) * kHq * kD, "turbo3 decode")) return 1;

  // Bytes: bf16 K+V in + codes/absmax out (store); codes/absmax + q/o (decode).
  auto rec = [](double tokens, double kcode) { return tokens * kHkv * (kcode + 4.0); };
  const double c4 = kD / 2.0, c3 = (kD * 3 + 7) / 8;
  const double bf16_kv = static_cast<double>(kHkv) * kD * 2 * 2;  // per token
  std::vector<ext_bench::Row> rows;
  auto add = [&](const std::string& id, const char* op, double bytes, double flops, auto&& fn,
                 int warmup, int iters) {
    if (!ext_bench::wanted(opt, id)) return;
    ext_bench::Row r;
    r.id = id;
    r.op = op;
    r.bytes = bytes;
    r.flops = flops;
    r.elems = 0.0;
    r.note = "attention.rs flash_tq (upstream grid)";
    if (!ext_bench::measure(fn, warmup, iters, opt, flush, &r.st)) {
      std::fprintf(stderr, "%s: timing failed\n", id.c_str());
      std::exit(1);
    }
    rows.push_back(r);
  };
  // warmup / iters follow the kops case table (store 3/20, decode 3/100, prefill 3/40).
  for (int t : {2048, 4096}) {
    const std::string base = "lowbit.store.bf16.t" + std::to_string(t) + ".h8.d128.";
    add(base + "turbo4", "lowbit_store", t * bf16_kv + rec(t, c4) + rec(t, c4), 0.0,
        [&] { store4(t); }, 3, 20);
    add(base + "turbo3", "lowbit_store", t * bf16_kv + rec(t, c3) + rec(t, c4), 0.0,
        [&] { store3(t); }, 3, 20);
  }
  const double qo = static_cast<double>(kB) * kHq * kD * 2 * 2;
  add("lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4", "lowbit_decode",
      rec(kAllTokens, c4) * 2 + qo, 0.0, decode4, 3, 100);
  add("lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo3", "lowbit_decode",
      rec(kAllTokens, c3) + rec(kAllTokens, c4) + qo, 0.0, decode3, 3, 100);
  {
    const double pairs = static_cast<double>(kPrefillB) * kQ * (2.0 * kKV - kQ + 1.0) / 2.0;
    const double flops = 4.0 * pairs * kD * kHq;
    add("lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo4", "lowbit_prefill", 0.0, flops,
        prefill4, 3, 40);
    // kops has no turbo3 prefill row (it has prod3); kept for reference, the runner filters it.
    add("lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo3", "lowbit_prefill", 0.0, flops,
        prefill3, 3, 40);
  }

  ext_bench::print_rows("attention.rs TurboQuant baseline (dev-only)", opt, rows);
  const double peak = ext_bench::peak_gbs();
  for (const auto& r : rows) {
    const double gbs = r.st.median_ms > 0.0 ? r.bytes / (r.st.median_ms * 1e6) : 0.0;
    if (peak > 0.0 && gbs > peak * 1.05) {
      std::fprintf(stderr, "warning: %s reports %.0f GB/s > device peak %.0f GB/s (byte count?)\n",
                   r.id.c_str(), gbs, peak);
    }
  }
  ext_bench::write_json(opt.json, "attention.rs TurboQuant (dev baseline)", opt, rows);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
