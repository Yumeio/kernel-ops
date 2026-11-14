// =============================================================================
// kernel-ops | benchmark/bench_gemm.cu
// B4 gemm-family cases (operand preparation).  The case table is the shared spec read by
// benchmark/compare/torch_baseline.py through --dump-cases.
//
// These are load-time operations: the figure of merit is the time to prepare one weight
// ("prep time"), reported per recipe.  bytes counts the compulsory input + output traffic.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "csrc/common/quant.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/dense_gemm_kernels.cuh"
#include "csrc/gemm/fp8_gemm_kernels.cuh"
#include "csrc/gemm/int8_gemm_kernels.cuh"
#include "csrc/gemm/fp4_gemm_kernels.cuh"
#include "csrc/gemm/mixed_input_gemm_kernels.cuh"
#include "csrc/gemm/gguf_gemm_kernels.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/gemm/gemm_prep_kernels.cuh"
#include "csrc/gemm/ops.h"
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/moe/permute_kernels.cuh"
#include "csrc/moe/fused_moe_kernels.cuh"
#include "csrc/moe/ops.h"

namespace {

using namespace kops_bench;

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) {
    if (!cuda_ok(cudaMalloc(&p, bytes), "cudaMalloc")) p = nullptr;
    else cudaMemset(p, 0x3C, bytes);
  }
  ~Buf() { if (p != nullptr) cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

// ---- B5: decode GEMV ---------------------------------------------------------------------
// rows = N, cols = K, M = flops / (2 N K).  bytes = weight + scale/zero + activation + output traffic.
// The baseline for every quantized case is the eager PyTorch dequantize-then-F.linear sequence
// (benchmark/compare/torch_baseline.py, make_b5_op); the figure of merit is GB/s of weight bytes.

void add_b5_gemv(std::vector<Case>* cases, const char* id, const char* op, kops::DType dt, int64_t n,
                 int64_t k, int64_t m, int64_t bytes) {
  cases->push_back({id, op, dt, n, k, bytes, 2 * m * n * k, 20, 200});
}

void add_b5_cases(std::vector<Case>* cases) {
  const kops::DType bf = kops::DType::kBFloat16;
  struct Shape { int64_t n, k; const char* tag; };
  const Shape shapes[] = {{4096, 4096, "4096x4096"}, {14336, 4096, "14336x4096"}, {4096, 14336, "4096x14336"}};
  char id[128];
  for (const Shape& s : shapes) {
    for (int64_t m : {1, 4}) {
      if (m == 4 && s.n != 4096) continue;
      const int64_t w4 = s.n * s.k / 2 + (s.k / 128) * s.n * 2 + (s.k / 128) * s.n / 2 + m * s.k * 2 + m * s.n * 2;
      std::snprintf(id, sizeof(id), "gemm.w4a16_awq.g128.m%lld.%s", static_cast<long long>(m), s.tag);
      add_b5_gemv(cases, id, "w4a16_awq", bf, s.n, s.k, m, w4);
      // Same weight bytes as the AWQ case; the tiles are awq_to_marlin's, read on mma.sync (SM80+).
      std::snprintf(id, sizeof(id), "gemm.w4a16_awq_marlin.g128.m%lld.%s", static_cast<long long>(m), s.tag);
      add_b5_gemv(cases, id, "w4a16_awq_marlin", bf, s.n, s.k, m, w4);
      if (m != 1) continue;
      const int64_t w4g = s.n * s.k / 2 + (s.k / 128) * s.n * 2 + (s.k / 128) * s.n / 2 + m * s.k * 2 + m * s.n * 2;
      std::snprintf(id, sizeof(id), "gemm.w4a16_gptq.g128.m%lld.%s", static_cast<long long>(m), s.tag);
      add_b5_gemv(cases, id, "w4a16_gptq", bf, s.n, s.k, m, w4g);
      const int64_t nf = s.n * s.k / 2 + (s.n * s.k / 64) * 4 + m * s.k * 2 + m * s.n * 2;
      std::snprintf(id, sizeof(id), "gemm.nf4_lut.b64.m%lld.%s", static_cast<long long>(m), s.tag);
      add_b5_gemv(cases, id, "nf4_lut", bf, s.n, s.k, m, nf);
    }
    for (int64_t m : {1, 4}) {
      if (m == 4 && s.n != 4096) continue;
      const int64_t f8 = s.n * s.k + m * s.k * 2 + m * s.n * 2 + (s.n + m) * 4;
      std::snprintf(id, sizeof(id), "gemm.fp8_gemv.rowwise.m%lld.%s", static_cast<long long>(m), s.tag);
      add_b5_gemv(cases, id, "fp8_gemv_rowwise", bf, s.n, s.k, m, f8);
      const int64_t i8 = s.n * s.k + m * s.k + m * s.n * 2 + (s.n + m) * 4;
      std::snprintf(id, sizeof(id), "gemm.int8_gemv.w8a8.m%lld.%s", static_cast<long long>(m), s.tag);
      add_b5_gemv(cases, id, "int8_gemv_w8a8", bf, s.n, s.k, m, i8);
      if (m != 1) continue;
      const int64_t f8b = s.n * s.k + m * s.k * 2 + m * s.n * 2 + (s.n / 128) * (s.k / 128) * 4;
      std::snprintf(id, sizeof(id), "gemm.fp8_gemv.blockwise.m%lld.%s", static_cast<long long>(m), s.tag);
      add_b5_gemv(cases, id, "fp8_gemv_blockwise", bf, s.n, s.k, m, f8b);
      std::snprintf(id, sizeof(id), "gemm.int8_gemv.w8a16.m%lld.%s", static_cast<long long>(m), s.tag);
      add_b5_gemv(cases, id, "int8_gemv_w8a16", bf, s.n, s.k, m, s.n * s.k + m * s.k * 2 + m * s.n * 2 + s.n * 4);
    }
    if (s.n == 4096 && s.k == 14336) continue;
    // Dense 16-bit GEMV: PyTorch's cuBLAS GEMV already streams the weight once, so parity is expected.
    std::snprintf(id, sizeof(id), "gemm.dense_gemv.bf16.m1.%s", s.tag);
    add_b5_gemv(cases, id, "dense_gemv", bf, s.n, s.k, 1, s.n * s.k * 2 + s.k * 2 + s.n * 2);
    for (const char* kind : {"mx", "nv"}) {
      const int64_t blk = kind[0] == 'm' ? 32 : 16;
      std::snprintf(id, sizeof(id), "gemm.fp4_gemv.%s.m1.%s", kind, s.tag);
      add_b5_gemv(cases, id, kind[0] == 'm' ? "fp4_gemv_mx" : "fp4_gemv_nv", bf, s.n, s.k, 1,
                  s.n * s.k / 2 + s.n * s.k / blk + s.k * 2 + s.n * 2);
      // Same bytes; fp4_to_marlin tiles on mma.sync (SM80+).
      std::snprintf(id, sizeof(id), "gemm.fp4_marlin.%s.m1.%s", kind, s.tag);
      add_b5_gemv(cases, id, kind[0] == 'm' ? "fp4_marlin_mx" : "fp4_marlin_nv", bf, s.n, s.k, 1,
                  s.n * s.k / 2 + s.n * s.k / blk + s.k * 2 + s.n * 2);
      if (s.n != 4096 || s.k != 4096) continue;
      for (int64_t mm : {4, 16}) {
        std::snprintf(id, sizeof(id), "gemm.fp4_marlin.%s.m%lld.%s", kind, static_cast<long long>(mm), s.tag);
        add_b5_gemv(cases, id, kind[0] == 'm' ? "fp4_marlin_mx" : "fp4_marlin_nv", bf, s.n, s.k, mm,
                    s.n * s.k / 2 + s.n * s.k / blk + mm * s.k * 2 + mm * s.n * 2);
      }
    }
    struct G { const char* name; const char* tag; int64_t qk, bytes; };
    for (const G& g : {G{"q4_0", "gguf_mmvq_q4_0", 32, 18}, G{"q8_0", "gguf_mmvq_q8_0", 32, 34}, G{"q4_k", "gguf_mmvq_q4_k", 256, 144}}) {
      std::snprintf(id, sizeof(id), "gemm.gguf_mmvq.%s.m1.%s", g.name, s.tag);
      add_b5_gemv(cases, id, g.tag, bf, s.n, s.k, 1, s.n * s.k / g.qk * g.bytes + s.k * 2 + s.n * 2);
    }
  }
  for (int64_t m : {2, 4, 8}) {
    // NF4 with several tokens in one launch (kLutRows changes with M).
    const int64_t n = 4096, k = 4096;
    std::snprintf(id, sizeof(id), "gemm.nf4_lut.b64.m%lld.4096x4096", static_cast<long long>(m));
    add_b5_gemv(cases, id, "nf4_lut", bf, n, k, m, n * k / 2 + (n * k / 64) * 4 + m * k * 2 + m * n * 2);
  }
  {
    // Two token tiles of 8 inside one CTA.
    const int64_t n = 4096, k = 4096, m = 16;
    add_b5_gemv(cases, "gemm.w4a16_awq_marlin.g128.m16.4096x4096", "w4a16_awq_marlin", bf, n, k, m,
                n * k / 2 + (k / 128) * n * 2 + (k / 128) * n / 2 + m * k * 2 + m * n * 2);
  }
  // Router logits (E x H), greedy LM head (V x H), grouped / MoE decode GEMV (E experts, N x K each).
  add_b5_gemv(cases, "gemm.router.fp32out.m8.256x7168", "router_gemm", bf, 256, 7168, 8, 256 * 7168 * 2 + 8 * 7168 * 2 + 8 * 256 * 4);
  add_b5_gemv(cases, "gemm.argmax.bf16.m1.32000x4096", "gemv_argmax", bf, 32000, 4096, 1, 32000LL * 4096 * 2);
  add_b5_gemv(cases, "gemm.grouped_gemv.fp8.e32x2rows.2048x2048", "grouped_fp8", bf, 2048, 2048, 64, 32LL * 2048 * 2048 + 64 * 2048 * 4);
  add_b5_gemv(cases, "gemm.moe_gemv.fp8.s32.2048x2048", "moe_gemv_fp8", bf, 2048, 2048, 32, 32LL * 2048 * 2048 + 32 * 2048 * 4);
}


// ---- B7: tensor-core GEMM (gemm_mma_sync_kernel) -------------------------------------------------
// rows = N, cols = K, M = flops / (2 N K).  Figure of merit: TFLOP/s; bytes is the compulsory A + B + D
// traffic.  "mma_*" uses the static heuristic, "mma_*_tuned" autotunes the M bucket first (outside the timing).
// M 32..256 is the bandwidth-bound band: it runs the swap tiles with the 128-byte row fetch (wide_ld).
void add_b7_cases(std::vector<Case>* cases) {
  char id[128];
  struct S { int64_t m, n, k; const char* tag; };
  const S shapes[] = {{32, 4096, 4096, "m32.4096x4096"},     {64, 4096, 4096, "m64.4096x4096"},
                      {128, 4096, 4096, "m128.4096x4096"},   {256, 4096, 4096, "m256.4096x4096"},
                      {1024, 4096, 4096, "m1024.4096x4096"}, {4096, 4096, 4096, "m4096.4096x4096"},
                      {2048, 14336, 4096, "m2048.14336x4096"}, {2048, 4096, 14336, "m2048.4096x14336"}};
  for (const S& s : shapes) {
    const int64_t bytes = (s.m * s.k + s.n * s.k + s.m * s.n) * 2;
    for (const char* kind : {"bf16", "fp16", "bf16_tuned"}) {
      if (kind[0] == 'f' && s.m != 1024 && s.m != 256) continue;
      if (kind[4] == '_' && s.m != 64 && s.m != 256 && s.m != 1024 && s.m != 4096) continue;
      std::snprintf(id, sizeof(id), "gemm.mma_%s.%s", kind, s.tag);
      std::string op = std::string("mma_") + kind;
      cases->push_back({id, op, kind[0] == 'f' ? kops::DType::kFloat16 : kops::DType::kBFloat16, s.n, s.k, bytes, 2 * s.m * s.n * s.k, 5, 50});
    }
  }
}

// ---- B13: SM120a tile cell (gemm_sm120_kernel, csrc/gemm/sm120a_impl.cuh) -------------------------
// Same problem as the B7 mma cases and the same figure of merit (TFLOP/s), so the two rows are directly
// comparable; the difference is only the tactic (kSm120WarpSpecialized, tile 64x128), exactly like
// "mma_bf16_tuned" is a tactic variant of "mma_bf16". The op stays mma_bf16: the math (D = A . B^T in
// bf16 with an fp32 accumulator) and therefore the PyTorch baseline of the case are the same.
// Only a KOPS_TARGET_SM120A device has this cell; elsewhere gemm_with_tactic refuses the tactic and the
// case reports the Status instead of a time.
void add_b13_cases(std::vector<Case>* cases) {
  char id[128];
  // The tile is part of the id: both compiled tiles are measured, at the shape where each one wins
  // (64x128 on the 2048 square, 128x128 once the operands no longer fit in the 32 MB L2).
  struct S { int64_t m, n, k; int tm, tn; const char* tag; };
  const S shapes[] = {{2048, 2048, 2048, 64, 128, "m2048.2048x2048.t64x128"},
                      {4096, 4096, 4096, 128, 128, "m4096.4096x4096.t128x128"}};
  for (const S& s : shapes) {
    const int64_t bytes = (s.m * s.k + s.n * s.k + s.m * s.n) * 2;
    std::snprintf(id, sizeof(id), "gemm.mma_sm120.bf16.%s", s.tag);
    cases->push_back({id, "mma_bf16", kops::DType::kBFloat16, s.n, s.k, bytes, 2 * s.m * s.n * s.k, 5, 50});
  }
}

// ---- B11: SM90a cell (gemm_wgmma_kernel, csrc/gemm/sm90a_impl.cuh) --------------------------------
// Same problem and figure of merit as the B7 mma cases (op mma_bf16, TFLOP/s); the tactic names the
// Hopper mainloop and the tile. Only a KOPS_TARGET_SM90A device has this cell; elsewhere the cases
// are dropped before the run (see main).
void add_b11_wgmma_cases(std::vector<Case>* cases) {
  char id[128];
  // The id carries the tactic: tile `.t<M>x<N>` and cluster `.c<M>x<N>` (1x2 = A tile multicast, 2x1 = B).
  struct S { int64_t m, n, k; };
  struct T { int tm, tn, cm, cn; };
  const S shapes[] = {{256, 4096, 4096},  {1024, 4096, 4096}, {2048, 2048, 2048}, {4096, 4096, 4096},
                      {8192, 8192, 8192}, {16384, 7168, 2048} /* DeepSeek-V3 shaped prefill projection */};
  const T tactics[] = {{64, 128, 1, 1},  {64, 256, 1, 1},  {128, 128, 1, 1}, {128, 256, 1, 1}, {256, 128, 1, 1},
                       {128, 128, 1, 2}, {128, 128, 2, 1}, {128, 256, 1, 2}, {128, 256, 2, 1}, {256, 128, 1, 2},
                       {256, 128, 2, 1}};
  for (const S& s : shapes) {
    for (const T& t : tactics) {
      if (s.m >= 8192 && t.tm == 64) continue;       // small tiles on the largest problems add nothing
      const int64_t bytes = (s.m * s.k + s.n * s.k + s.m * s.n) * 2;
      std::snprintf(id, sizeof(id), "gemm.wgmma.bf16.m%lld.%lldx%lld.t%dx%d.c%dx%d", static_cast<long long>(s.m),
                    static_cast<long long>(s.n), static_cast<long long>(s.k), t.tm, t.tn, t.cm, t.cn);
      cases->push_back({id, "mma_bf16", kops::DType::kBFloat16, s.n, s.k, bytes, 2 * s.m * s.n * s.k, 5, 50});
    }
  }
  // The same cell serves kFP16; a smaller sweep keeps the family honest without duplicating the matrix.
  const S f16_shapes[] = {{4096, 4096, 4096}, {16384, 7168, 2048}};
  const T f16_tactics[] = {{128, 128, 1, 1}, {128, 256, 1, 1}, {256, 128, 1, 1}, {128, 128, 1, 2}};
  for (const S& s : f16_shapes) {
    for (const T& t : f16_tactics) {
      const int64_t bytes = (s.m * s.k + s.n * s.k + s.m * s.n) * 2;
      std::snprintf(id, sizeof(id), "gemm.wgmma.fp16.m%lld.%lldx%lld.t%dx%d.c%dx%d", static_cast<long long>(s.m),
                    static_cast<long long>(s.n), static_cast<long long>(s.k), t.tm, t.tn, t.cm, t.cn);
      cases->push_back({id, "mma_fp16", kops::DType::kFloat16, s.n, s.k, bytes, 2 * s.m * s.n * s.k, 5, 50});
    }
  }
}

// ---- B11: SM90a FP8 (fp8_gemm_wgmma_kernel) --------------------------------------------------------
// Same ops and operands as the B7 FP8 cases below (op fp8_mma_<kind>), on the Hopper mainloop. The id
// names the accumulation (`fast` = Fp8Accum::kFastAccum, `promoted` = kPromoted; blockwise is always
// promoted) and the tile. The promoted body exists for the 128-wide tiles.
void add_b11_fp8_cases(std::vector<Case>* cases) {
  char id[160];
  const int64_t n = 4096, k = 4096;
  struct T { int tm, tn; };
  for (int64_t m : {256, 1024, 4096}) {
    for (const char* kind : {"tensorwise.fast", "tensorwise.promoted", "rowwise.fast", "rowwise.promoted",
                             "blockwise.promoted", "blockwise.ue8m0"}) {
      const bool promoted = std::strstr(kind, "promoted") != nullptr;
      const bool ue8m0 = std::strstr(kind, "ue8m0") != nullptr;
      const bool row = kind[0] == 'r', block = kind[0] == 'b';
      const int64_t scales = row ? (m + n) * 4
                                 : block ? ((m + (n + 127) / 128) * (k / 128)) * (ue8m0 ? 1 : 4)
                                         : 0;
      for (const T& t : {T{64, 128}, T{128, 128}, T{128, 256}, T{256, 128}}) {
        if (promoted && t.tn > 128) continue;
        if (block && (t.tm > 128 || t.tn > 128)) continue;   // the FP8 blockwise cell has 64x128 and 128x128 only
        std::snprintf(id, sizeof(id), "gemm.wgmma.fp8.%s.m%lld.4096x4096.t%dx%d", kind, static_cast<long long>(m), t.tm, t.tn);
        cases->push_back({id, std::string("fp8_mma_") + (row ? "rowwise" : block ? "blockwise" : "tensorwise"),
                          kops::DType::kFloat8E4M3, n, k, m * k + n * k + m * n * 2 + scales, 2 * m * n * k, 5, 50});
      }
    }
  }
}

// ---- B11: SM90a grouped cell (grouped_gemm_wgmma_kernel) ---------------------------------------------
// Contiguous (expert_offsets) and masked (masked_m) row maps, bf16 / fp16 / FP8 blockwise / FP8 UE8M0.
// The routed rows are split evenly: rows(g) = R/G + (g < R % G). The id carries G, the routed row
// count R and the tile so torch_baseline.py can rebuild the same per-expert loop.
void add_b11_grouped_cases(std::vector<Case>* cases) {
  char id[192];
  struct S { int64_t g, r, n, k; const char* tag; };
  const S shapes[] = {{8, 1024, 4096, 4096, "8x1024.4096x4096"},
                      {32, 4096, 2048, 2048, "32x4096.2048x2048"},
                      {128, 100, 1024, 1024, "128x100.1024x1024"}};
  struct T { int tm, tn; };
  struct D { const char* tag; const char* op; kops::DType dt; int64_t eb; bool block; bool ue8; };
  // One tile per entry where the cell restricts the shape (FP8: 64x128 / 128x128; 16-bit also 128x256).
  const D kinds[] = {
      {"bf16", "wgmma_grouped_bf16", kops::DType::kBFloat16, 2, false, false},
      {"fp16", "wgmma_grouped_fp16", kops::DType::kFloat16, 2, false, false},
      {"fp8", "wgmma_grouped_fp8", kops::DType::kFloat8E4M3, 1, true, false},
      {"fp8ue8m0", "wgmma_grouped_fp8ue8m0", kops::DType::kFloat8E4M3, 1, true, true},
  };
  for (const S& s : shapes) {
    for (const char* lay : {"contig", "masked"}) {
      const bool masked = lay[0] == 'm';
      const int64_t per = (s.r + s.g - 1) / s.g;
      const int64_t cap = masked ? per * s.g : s.r;
      for (const D& d : kinds) {
        const T tiles[] = {{64, 128}, {128, 128}, {128, 256}};
        for (const T& t : tiles) {
          if (d.eb == 1 && t.tn > 128) continue;            // FP8 blockwise: 64x128 / 128x128
          if (std::strcmp(d.tag, "fp16") == 0 && !(t.tm == 128 && t.tn == 128)) continue;  // fp16: 128x128 only
          const int64_t kb = s.k / 128, nb = (s.n + 127) / 128;
          const int64_t scale_bytes = d.block ? (cap * kb + s.g * nb * kb) * (d.ue8 ? 1 : 4) : 0;
          const int64_t bytes = cap * s.k * d.eb + s.g * s.n * s.k * d.eb + cap * s.n * 2 + scale_bytes;
          std::snprintf(id, sizeof(id), "gemm.wgmma.grouped.%s.%s.g%lld.r%lld.%lldx%lld.t%dx%d", d.tag, lay,
                        static_cast<long long>(s.g), static_cast<long long>(s.r), static_cast<long long>(s.n),
                        static_cast<long long>(s.k), t.tm, t.tn);
          cases->push_back({id, d.op, d.dt, s.n, s.k, bytes, 2 * s.r * s.n * s.k, 5, 50});
        }
      }
    }
  }
}

// ---- B7 row 3: e4m3 x e4m3 on mma.sync (fp8_gemm_mma_sync_kernel, SM89+) -------------------------------
// rows = N, cols = K, M = flops / (2 N K). Tensorwise / rowwise / 128x128 blockwise scales; the scale
// buffers are zero-filled (the product is exact in the epilogue, so the timing is the same).
void add_b7_fp8_cases(std::vector<Case>* cases) {
  char id[128];
  const int64_t n = 4096, k = 4096;
  for (int64_t m : {256, 1024, 4096}) {
    for (const char* kind : {"tensorwise", "rowwise", "blockwise"}) {
      const int64_t scales = std::strcmp(kind, "rowwise") == 0 ? (m + n) * 4
                            : std::strcmp(kind, "blockwise") == 0 ? ((m + (n + 127) / 128) * (k / 128)) * 4
                                                                  : 0;
      std::snprintf(id, sizeof(id), "gemm.fp8_mma.%s.m%lld.4096x4096", kind, static_cast<long long>(m));
      cases->push_back({id, std::string("fp8_mma_") + kind, kops::DType::kFloat8E4M3, n, k,
                        m * k + n * k + m * n * 2 + scales, 2 * m * n * k, 5, 50});
    }
  }
}

// ---- B7 row 9: multi-LoRA SGMV (lora_shrink_kernel + lora_expand_kernel) -----------------------------
// T tokens, 4 adapters, K = N = 4096, rank 16 / 64. Figure of merit: the two GEMM passes
// (2 T rank K + 2 T rank N); the shrink buffer and the adapter pointers are part of the traffic.
void add_b7_lora_cases(std::vector<Case>* cases) {
  char id[128];
  const int64_t k = 4096, n = 4096, t = 512;
  for (int64_t rank : {16, 64}) {
    const int64_t bytes = t * k * 2 + t * n * 2 + 4 * rank * k * 2 + 4 * n * rank * 2 + t * rank * 4;
    std::snprintf(id, sizeof(id), "gemm.lora.rank%lld.t%lld.4096x4096", static_cast<long long>(rank),
                  static_cast<long long>(t));
    cases->push_back({id, std::string("lora_rank") + std::to_string(rank), kops::DType::kBFloat16, n, k, bytes,
                      2 * t * rank * (k + n), 5, 50});
  }
}

// ---- B7 row 2: dual_gemm (D = GLU(A.B0^T, A.B1^T)) and bmm (MLA absorption, 128 heads) ----------------
// dual: rows = N, cols = K, M = flops / (4 N K).  bmm_mla_*: batch 128, A and D are [M, 128, *] head views.
void add_b7_row2_cases(std::vector<Case>* cases) {
  char id[128];
  struct S { int64_t m, n, k; };
  for (const S& s : {S{64, 4096, 4096}, S{512, 4096, 4096}, S{2048, 4096, 4096}, S{1024, 14336, 4096}}) {
    std::snprintf(id, sizeof(id), "gemm.dual_bf16.m%lld.%lldx%lld", static_cast<long long>(s.m), static_cast<long long>(s.n), static_cast<long long>(s.k));
    cases->push_back({id, "dual_bf16", kops::DType::kBFloat16, s.n, s.k, (s.m * s.k + 2 * s.n * s.k + s.m * s.n) * 2, 4 * s.m * s.n * s.k, 5, 50});
  }
  for (int64_t m : {1, 8, 64, 512}) {
    for (int uv = 0; uv < 2; ++uv) {
      const int64_t k = uv ? 512 : 128, n = uv ? 128 : 512, batch = 128;
      std::snprintf(id, sizeof(id), "gemm.bmm_mla_%s.m%lld", uv ? "uv" : "uk", static_cast<long long>(m));
      cases->push_back({id, uv ? "bmm_mla_uv" : "bmm_mla_uk", kops::DType::kBFloat16, n, k, batch * (m * k + k * n + m * n) * 2, 2 * batch * m * n * k, 5, 50});
    }
  }
}

// ---- B7 row 5: INT8 W8A8 on mma.sync (int8_gemm, M > 16) and QServe W4A8 (w4a8_int_gemm_kernel) ----------
// rows = N, cols = K, M = flops / (2 N K).  Figure of merit: TOPS (the TFLOP/s column counts int8 ops).
// bytes: A (s8) + B (s8, or packed u4 + uint8 level-2 scale / zero) + fp32 row / column scales + bf16 D.
void add_b7_int8_cases(std::vector<Case>* cases) {
  char id[128];
  const int64_t n = 4096, k = 4096;
  for (int64_t m : {256, 1024, 4096}) {
    const int64_t io = m * k + m * 4 + n * 4 + m * n * 2;
    std::snprintf(id, sizeof(id), "gemm.int8_mma.m%lld.4096x4096", static_cast<long long>(m));
    cases->push_back({id, "int8_mma", kops::DType::kInt8, n, k, io + n * k, 2 * m * n * k, 5, 50});
    std::snprintf(id, sizeof(id), "gemm.w4a8_g128.m%lld.4096x4096", static_cast<long long>(m));
    cases->push_back({id, "w4a8_g128", kops::DType::kInt8, n, k, io + n * k / 2 + 2 * (k / 128) * n, 2 * m * n * k, 5, 50});
    std::snprintf(id, sizeof(id), "gemm.w4a8_chn.m%lld.4096x4096", static_cast<long long>(m));
    cases->push_back({id, "w4a8_chn", kops::DType::kInt8, n, k, io + n * k / 2 + n, 2 * m * n * k, 5, 50});
  }
}

// ---- B7 row 10: GGUF mmq (prefill, int8 mma.sync) -------------------------------------------------------
// rows = N, cols = K, M = flops / (2 N K); bytes = ggml weight blocks + bf16 A + bf16 D (+ q8_1 workspace).
void add_b7_gguf_cases(std::vector<Case>* cases) {
  char id[128];
  struct G { const char* name; int64_t qk, bytes; };
  for (const G& g : {G{"q4_k", 256, 144}, G{"q8_0", 32, 34}}) {
    for (int64_t m : {64, 512, 2048}) {
      const int64_t n = 4096, k = 4096;
      std::snprintf(id, sizeof(id), "gemm.gguf_mmq.%s.m%lld.4096x4096", g.name, static_cast<long long>(m));
      const int64_t bytes = n * k / g.qk * g.bytes + m * k * 2 + m * k / 32 * 36 * 2 + m * n * 2;
      cases->push_back({id, std::string("gguf_mmq_") + g.name, kops::DType::kBFloat16, n, k, bytes, 2 * m * n * k, 5, 50});
    }
  }
}

// ---- B11: Machete prepack and grouped preparation -------------------------------------------------
// Load-time weight repack and per-step MoE group preparation. rows = N, cols = K (prepack) or the
// grouped problem's N x K with `rows` experts; bytes is the compulsory traffic of the operation.
void add_b11_cases(std::vector<Case>* cases) {
  char id[128];
  for (const char* kind : {"int4", "int8"}) {
    const int64_t n = 4096, k = 4096;
    const int64_t w = std::strcmp(kind, "int4") == 0 ? n * k / 2 : n * k;
    std::snprintf(id, sizeof(id), "gemm.machete_prepack.%s.4096x4096", kind);
    // The dumped case spec names the packed storage type (kUInt8) like weight_transpose_int4; the
    // launch below uses kInt4x2 / kInt8 so the kernel sees the logical element type.
    cases->push_back({id, std::string("machete_prepack_") + kind,
                      std::strcmp(kind, "int4") == 0 ? kops::DType::kUInt8 : kops::DType::kInt8, n, k, 2 * w, 0,
                      20, 200});
  }
  {
    // 32 experts, 128 rows each, one task per 128x128 tile: the preparation only touches the row
    // metadata, the problems and the task map, so the case keeps the problem small on purpose.
    const int64_t g = 32, n = 1024, k = 1024, rows_per_group = 128;
    const int64_t tasks = g * ((rows_per_group + 127) / 128) * ((n + 127) / 128);
    const int64_t bytes = (g + 1) * 8 + g * static_cast<int64_t>(sizeof(kops::gemm::GroupedProblem)) +
                          (4 + 3 * tasks) * 4;
    cases->push_back({"gemm.grouped_prepare.e32.1024x1024", "grouped_prepare", kops::DType::kBFloat16, n, k,
                      bytes, 0, 20, 200});
  }
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto add = [&](const char* id, const char* op, kops::DType dt, int64_t rows, int64_t cols,
                 int64_t bytes) {
    cases.push_back({id, op, dt, rows, cols, bytes, 0, 20, 200});
  };
  const int64_t r = 4096;
  const int64_t c = 4096;
  add("gemm.transpose.bf16.4096x4096", "weight_transpose", kops::DType::kBFloat16, r, c, 4 * r * c);
  add("gemm.transpose.fp8.4096x4096", "weight_transpose", kops::DType::kFloat8E4M3, r, c, 2 * r * c);
  add("gemm.transpose.int4.4096x4096", "weight_transpose_int4", kops::DType::kUInt8, r, c, r * c);
  add("gemm.glu_interleave.bf16.8192x4096", "glu_interleave", kops::DType::kBFloat16, 8192, c,
      4 * 8192 * c);
  add("gemm.router_split.fp32.4096x4096", "router_split", kops::DType::kFloat32, r, c, 8 * r * c);
  add("gemm.int8_colsum.4096x4096", "int8_colsum", kops::DType::kInt8, r, c, r * c);
  add("gemm.scale_swizzle.14336x256", "block_scale_swizzle", kops::DType::kUInt8, 14336, 256,
      2 * 14336 * 256);
  add("gemm.scale_tma.f32.32768x56", "fp8_scale_to_tma", kops::DType::kFloat32, 32768, 56,
      8 * 32768 * 56);
  add("gemm.scale_tma.ue8m0.32768x56", "fp8_scale_to_tma_ue8m0", kops::DType::kFloat32, 32768, 56,
      5 * 32768 * 56);
  add("gemm.fp8_block_quant.bf16.4096x4096", "fp8_block_quant", kops::DType::kBFloat16, r, c,
      3 * r * c);
  add("gemm.fp8_block_quant.ue8m0.bf16.4096x4096", "fp8_block_quant_ue8m0", kops::DType::kBFloat16,
      r, c, 3 * r * c);
  add_b5_cases(&cases);
  add_b7_cases(&cases);
  add_b13_cases(&cases);
  add_b11_wgmma_cases(&cases);
  add_b11_fp8_cases(&cases);
  add_b11_grouped_cases(&cases);
  add_b7_fp8_cases(&cases);
  add_b7_lora_cases(&cases);
  add_b7_row2_cases(&cases);
  add_b7_gguf_cases(&cases);
  add_b7_int8_cases(&cases);
  add_b11_cases(&cases);
  return cases;
}

/// Runs a B5 GEMV case; *handled is false when c.op is not a B5 op.
bool run_b5_case(const Case& c, const kops::LaunchContext& ctx, Stats* st, bool* handled) {
  *handled = true;
  const int64_t n = c.rows;
  const int64_t k = c.cols;
  const int64_t m = c.flops / (2 * n * k);
  const char* id = c.id.c_str();
  using kops::DType;
  using namespace kops::gemm;
  auto go = [&](auto&& op) {
    if (!status_ok(op(), id)) return false;
    return measure(c, [&] { (void)op(); }, st);
  };
  if (c.op == "w4a16_awq" || c.op == "w4a16_gptq" || c.op == "w4a16_awq_marlin") {
    const bool marlin = c.op == "w4a16_awq_marlin";
    const bool awq = c.op == "w4a16_awq" || marlin;
    Buf a(m * k * 2), qw(n * k / 2), sc((k / 128) * n * 2), qz((k / 128) * n / 2), out(m * n * 2);
    if (!a.p || !qw.p || !sc.p || !qz.p || !out.p) return false;
    ScaledGemmArgs args{};
    args.recipe = GemmRecipe::kW4A16Int;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = DType::kBFloat16;
    args.b.data = qw.p; args.b.dtype = DType::kUInt32;
    args.b.packing = marlin ? kops::WeightPacking::kMarlin : (awq ? kops::WeightPacking::kAWQ : kops::WeightPacking::kGPTQ);
    if (marlin) args.b.scale_layout = kops::ScaleLayout::kMarlinPermuted;
    args.b.scale = sc.p; args.b.scale_dtype = DType::kBFloat16; args.b.group_k = 128;
    args.b.zero_points = qz.p; args.b.zero_dtype = DType::kUInt32;
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    return go([&] { return mixed_input_gemm(args, nullptr, GemmBackend::kNative, kops::Workspace{}, ctx); });
  }
  if (c.op == "nf4_lut") {
    Buf a(m * k * 2), qw(n * k / 2), cb(16 * 4), am((n * k / 64) * 4), out(m * n * 2);
    if (!a.p || !qw.p || !cb.p || !am.p || !out.p) return false;
    const kops::Tensor to = tensor2d(out.p, DType::kBFloat16, m, n);
    const kops::Tensor ta = tensor2d(a.p, DType::kBFloat16, m, k);
    const kops::Tensor tq = tensor2d(qw.p, DType::kUInt8, n, k / 2);
    const kops::Tensor tc = tensor1d(cb.p, DType::kFloat32, 16);
    const kops::Tensor tm = tensor1d(am.p, DType::kFloat32, n * k / 64);
    return go([&] { return lut_gemv(to, ta, tq, tc, tm, kops::Tensor{}, kops::Tensor{}, 64, ctx); });
  }
  if (c.op == "fp8_gemv_rowwise" || c.op == "fp8_gemv_blockwise") {
    const bool block = c.op == "fp8_gemv_blockwise";
    Buf a(m * k * 2), w(n * k), rs(m * 4), cs(n * 4), sb((n / 128) * (k / 128) * 4), out(m * n * 2);
    if (!a.p || !w.p || !rs.p || !cs.p || !sb.p || !out.p) return false;
    ScaledGemmArgs args{};
    args.recipe = block ? GemmRecipe::kFP8Blockwise : GemmRecipe::kFP8Rowwise;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = DType::kBFloat16;
    args.b.data = w.p; args.b.ld = k; args.b.dtype = DType::kFloat8E4M3;
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    if (block) {
      args.b.scale = sb.p; args.b.scale_dtype = DType::kFloat32;
    } else {
      args.epi.ops = kEpiRowScale | kEpiColScale;
      args.epi.row_scale = static_cast<const float*>(rs.p); args.epi.col_scale = static_cast<const float*>(cs.p);
    }
    return go([&] { return fp8_gemm(args, nullptr, GemmBackend::kNative, kops::Workspace{}, ctx); });
  }
  if (c.op == "int8_gemv_w8a8" || c.op == "int8_gemv_w8a16") {
    const bool f16a = c.op == "int8_gemv_w8a16";
    Buf a(m * k * 2), w(n * k), rs(m * 4), cs(n * 4), out(m * n * 2);
    if (!a.p || !w.p || !rs.p || !cs.p || !out.p) return false;
    ScaledGemmArgs args{};
    args.recipe = GemmRecipe::kINT8Rowwise;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = f16a ? DType::kBFloat16 : DType::kInt8;
    args.b.data = w.p; args.b.ld = k; args.b.dtype = DType::kInt8;
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    args.epi.ops = kEpiColScale | (f16a ? 0u : kEpiRowScale);
    args.epi.row_scale = static_cast<const float*>(rs.p); args.epi.col_scale = static_cast<const float*>(cs.p);
    return go([&] { return int8_gemm(args, nullptr, GemmBackend::kNative, kops::Workspace{}, ctx); });
  }
  if (c.op == "dense_gemv") {
    Buf a(m * k * 2), w(n * k * 2), out(m * n * 2);
    if (!a.p || !w.p || !out.p) return false;
    ScaledGemmArgs args{};
    args.recipe = GemmRecipe::kBF16;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = DType::kBFloat16;
    args.b.data = w.p; args.b.ld = k; args.b.dtype = DType::kBFloat16;
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    return go([&] { return scaled_gemm(args, nullptr, GemmBackend::kNative, kops::Workspace{}, ctx); });
  }
  if (c.op == "fp4_gemv_mx" || c.op == "fp4_gemv_nv") {
    const bool nv = c.op == "fp4_gemv_nv";
    Buf a(m * k * 2), w(n * k / 2), sc(n * k / (nv ? 16 : 32)), out(m * n * 2);
    if (!a.p || !w.p || !sc.p || !out.p) return false;
    ScaledGemmArgs args{};
    args.recipe = nv ? GemmRecipe::kW4A16NVFP4 : GemmRecipe::kW4A16MXFP4;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = DType::kBFloat16;
    args.b.data = w.p; args.b.ld = k / 2; args.b.dtype = DType::kFloat4E2M1x2;
    args.b.scale = sc.p; args.b.scale_dtype = nv ? DType::kFloat8E4M3 : DType::kFloat8E8M0;
    args.b.group_k = nv ? 16 : 32;
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    return go([&] { return scaled_gemm(args, nullptr, GemmBackend::kNative, kops::Workspace{}, ctx); });
  }
  if (c.op == "fp4_marlin_mx" || c.op == "fp4_marlin_nv") {
    const bool nv = c.op == "fp4_marlin_nv";
    Buf a(m * k * 2), w(n * k / 2), sc(n * k / (nv ? 16 : 32)), out(m * n * 2);
    if (!a.p || !w.p || !sc.p || !out.p) return false;
    ScaledGemmArgs args{};
    args.recipe = nv ? GemmRecipe::kW4A16NVFP4 : GemmRecipe::kW4A16MXFP4;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = DType::kBFloat16;
    args.b.data = w.p; args.b.dtype = DType::kUInt32; args.b.packing = kops::WeightPacking::kMarlin;
    args.b.scale = sc.p; args.b.scale_dtype = nv ? DType::kFloat8E4M3 : DType::kFloat8E8M0;
    args.b.scale_layout = kops::ScaleLayout::kMarlinPermuted; args.b.group_k = nv ? 16 : 32;
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    return go([&] { return scaled_gemm(args, nullptr, GemmBackend::kNative, kops::Workspace{}, ctx); });
  }
  if (c.op.rfind("gguf_mmvq_", 0) == 0) {
    const std::string t = c.op.substr(10);
    const kops::GGUFType type = t == "q4_0" ? kops::GGUFType::kQ4_0 : (t == "q8_0" ? kops::GGUFType::kQ8_0 : kops::GGUFType::kQ4_K);
    const int64_t qk = t == "q4_k" ? 256 : 32;
    const int64_t bytes = t == "q4_0" ? 18 : (t == "q8_0" ? 34 : 144);
    Buf a(m * k * 2), w(n * k / qk * bytes), ws(m * k / 32 * 36), out(m * n * 2);
    if (!a.p || !w.p || !ws.p || !out.p) return false;
    const kops::Tensor to = tensor2d(out.p, DType::kBFloat16, m, n);
    const kops::Tensor ta = tensor2d(a.p, DType::kBFloat16, m, k);
    const kops::Tensor tw = tensor1d(w.p, DType::kUInt8, n * k / qk * bytes);
    const kops::Tensor tws = tensor1d(ws.p, DType::kUInt8, m * k / 32 * 36);
    return go([&] { return gguf_gemm(to, ta, tw, type, tws, ctx); });
  }
  if (c.op.rfind("gguf_mmq_", 0) == 0) {
    const bool q4k = c.op == "gguf_mmq_q4_k";
    const kops::GGUFType type = q4k ? kops::GGUFType::kQ4_K : kops::GGUFType::kQ8_0;
    const int64_t qk = q4k ? 256 : 32, bytes = q4k ? 144 : 34;
    Buf a(m * k * 2), w(n * k / qk * bytes), ws(m * k / 32 * 36), out(m * n * 2);
    if (!a.p || !w.p || !ws.p || !out.p) return false;
    const kops::Tensor to = tensor2d(out.p, DType::kBFloat16, m, n);
    const kops::Tensor ta = tensor2d(a.p, DType::kBFloat16, m, k);
    const kops::Tensor tw = tensor1d(w.p, DType::kUInt8, n * k / qk * bytes);
    const kops::Tensor tws = tensor1d(ws.p, DType::kUInt8, m * k / 32 * 36);
    return go([&] { return gguf_gemm(to, ta, tw, type, tws, ctx); });
  }
  if (c.op == "router_gemm") {
    Buf x(m * k * 2), w(n * k * 2), logits(m * n * 4);
    if (!x.p || !w.p || !logits.p) return false;
    RouterGemmParams p{};
    p.x = x.p; p.ldx = k; p.w = w.p; p.logits = static_cast<float*>(logits.p); p.ld_logits = n;
    p.m = static_cast<int32_t>(m); p.num_experts = static_cast<int32_t>(n); p.hidden = static_cast<int32_t>(k);
    return go([&] { return router_gemm(p, false, DType::kBFloat16, DType::kBFloat16, ctx); });
  }
  if (c.op == "gemv_argmax") {
    const int32_t nctas = gemv_argmax_num_ctas(static_cast<int32_t>(n));
    Buf h(m * k * 2), w(n * k * 2), pm(m * nctas * 4), pi(m * nctas * 4), ps(m * nctas * 4), ids(m * 4), cnt(4);
    if (!h.p || !w.p || !pm.p || !pi.p || !ps.p || !ids.p || !cnt.p) return false;
    cudaMemset(cnt.p, 0, 4);
    GemvArgmaxParams p{};
    p.hidden = h.p; p.ld_hidden = k; p.lm_head = w.p; p.ld_lm_head = k;
    p.logit_scale = 1.0f; p.vocab_start = 0; p.vocab_size = n;
    p.partial_max = static_cast<float*>(pm.p); p.partial_idx = static_cast<int32_t*>(pi.p);
    p.partial_sumexp = static_cast<float*>(ps.p); p.out_ids = static_cast<int32_t*>(ids.p);
    p.done_counter = static_cast<uint32_t*>(cnt.p);
    p.m = static_cast<int32_t>(m); p.hidden_size = static_cast<int32_t>(k); p.v_shard = static_cast<int32_t>(n);
    return go([&] { return gemv_argmax(p, DType::kBFloat16, DType::kBFloat16, ctx); });
  }
  if (c.op == "grouped_fp8") {
    const int E = 32;
    std::vector<int64_t> offs(E + 1);
    for (int e = 0; e <= E; ++e) offs[e] = 2 * e;
    int64_t* d_offs = nullptr;
    if (!cuda_ok(cudaMalloc(&d_offs, offs.size() * 8), "cudaMalloc")) return false;
    cudaMemcpy(d_offs, offs.data(), offs.size() * 8, cudaMemcpyHostToDevice);
    Buf a(m * k * 2), w(static_cast<size_t>(E) * n * k), sc(static_cast<size_t>(E) * n * 4), out(m * n * 2);
    if (!a.p || !w.p || !sc.p || !out.p) return false;
    GroupedGemmParams p{};
    p.layout = GroupedLayout::kContiguous; p.recipe = GemmRecipe::kFP8Rowwise; p.num_groups = E;
    p.a = a.p; p.lda = k; p.expert_offsets = d_offs;
    p.b = w.p; p.ldb = k; p.b_group_stride = n * k; p.b_scale = sc.p; p.b_scale_group_stride = n;
    p.d = out.p; p.ldd = n; p.n = static_cast<int32_t>(n); p.k = static_cast<int32_t>(k);
    const bool ok = go([&] { return grouped_gemm(p, nullptr, GemmBackend::kNative, kops::Workspace{}, ctx); });
    cudaFree(d_offs);
    return ok;
  }
  if (c.op == "moe_gemv_fp8") {
    const int E = 32, top_k = 8;
    std::vector<int32_t> ids(E), blk(E);
    for (int i = 0; i < E; ++i) { ids[i] = i; blk[i] = i; }
    int32_t *d_sorted = nullptr, *d_blk = nullptr, *d_pp = nullptr;
    const int32_t pp = E;
    cudaMalloc(&d_sorted, E * 4); cudaMalloc(&d_blk, E * 4); cudaMalloc(&d_pp, 4);
    cudaMemcpy(d_sorted, ids.data(), E * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blk, blk.data(), E * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pp, &pp, 4, cudaMemcpyHostToDevice);
    Buf a(static_cast<size_t>(E / top_k) * k * 2), w(static_cast<size_t>(E) * n * k), sc(static_cast<size_t>(E) * n * 4), out(static_cast<size_t>(E) * n * 2);
    if (!a.p || !w.p || !sc.p || !out.p) return false;
    kops::moe::FusedMoEGemmParams p{};
    p.a = a.p; p.lda = k; p.a_rows_are_slots = false;
    p.w = w.p; p.w_expert_stride = n * k; p.w_scale = sc.p; p.w_scale_expert_stride = n;
    p.c = out.p; p.ldc = n;
    p.sorted_token_ids = d_sorted; p.expert_ids = d_blk; p.num_tokens_post_pad = d_pp;
    p.top_k = top_k; p.n = static_cast<int32_t>(n); p.k = static_cast<int32_t>(k); p.num_valid_slots = E;
    p.w_quant.storage_dtype = DType::kFloat8E4M3; p.w_quant.granularity = kops::ScaleGranularity::kPerChannel;
    const bool ok = go([&] { return kops::moe::moe_gemv(p, E, 1, DType::kBFloat16, DType::kFloat8E4M3, ctx); });
    cudaFree(d_sorted); cudaFree(d_blk); cudaFree(d_pp);
    return ok;
  }
  if (c.op.rfind("lora_rank", 0) == 0) {
    const int rank = std::atoi(c.op.c_str() + 9);
    const int slots = 4, nseg = 8;
    const int64_t t = c.flops / (2 * static_cast<int64_t>(rank) * (k + n));
    Buf x(t * k * 2), ya(static_cast<size_t>(slots) * rank * k * 2), wb(static_cast<size_t>(slots) * n * rank * 2),
        buf(t * rank * 4), y(t * n * 2);
    if (!x.p || !ya.p || !wb.p || !buf.p || !y.p) return false;
    std::vector<int32_t> indptr(nseg + 1), ids(nseg);
    for (int i = 0; i <= nseg; ++i) indptr[i] = static_cast<int32_t>(i * t / nseg);
    for (int i = 0; i < nseg; ++i) ids[i] = i % slots;
    std::vector<const void*> pa(slots), pb(slots);
    for (int s = 0; s < slots; ++s) {
      pa[s] = static_cast<char*>(ya.p) + static_cast<int64_t>(s) * rank * k * 2;
      pb[s] = static_cast<char*>(wb.p) + static_cast<int64_t>(s) * n * rank * 2;
    }
    void *dpa = nullptr, *dpb = nullptr;
    if (!cuda_ok(cudaMalloc(&dpa, slots * sizeof(void*)), "cudaMalloc lora_a") ||
        !cuda_ok(cudaMalloc(&dpb, slots * sizeof(void*)), "cudaMalloc lora_b")) return false;
    cudaMemcpy(dpa, pa.data(), slots * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(dpb, pb.data(), slots * sizeof(void*), cudaMemcpyHostToDevice);
    Buf dind(indptr.size() * sizeof(int32_t)), dids(ids.size() * sizeof(int32_t));
    if (!dind.p || !dids.p) return false;
    cudaMemcpy(dind.p, indptr.data(), indptr.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dids.p, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    kops::Tensor tbuf{};
    tbuf.data = buf.p; tbuf.dtype = DType::kFloat32; tbuf.ndim = 2;
    tbuf.shape[0] = t; tbuf.shape[1] = rank; tbuf.stride[0] = rank; tbuf.stride[1] = 1;
    LoraSgmvParams sh{}, ex{};
    const int32_t* const sind = static_cast<const int32_t*>(dind.p);
    const int32_t* const sids = static_cast<const int32_t*>(dids.p);
    sh.x = x.p; sh.ldx = k; sh.lora_a = static_cast<const void* const*>(dpa); sh.seg_indptr = sind;
    sh.seg_lora_ids = sids; sh.num_segments = nseg; sh.in_dim = static_cast<int32_t>(k);
    sh.rank = rank;
    ex.lora_a = static_cast<const void* const*>(dpb); ex.y = y.p; ex.ldy = n; ex.seg_indptr = sind;
    ex.seg_lora_ids = sids; ex.num_segments = nseg; ex.out_dim = static_cast<int32_t>(n);
    ex.rank = rank; ex.scaling = 0.5f;
    const bool ok = go([&] { return lora_sgmv(sh, ex, DType::kBFloat16, tbuf, ctx); });
    cudaFree(dpa);
    cudaFree(dpb);
    return ok;
  }
  if (c.op.rfind("fp8_mma_", 0) == 0) {
    const bool row = c.op == "fp8_mma_rowwise";
    const bool block = c.op == "fp8_mma_blockwise";
    const bool ue8 = std::strstr(id, ".ue8m0.") != nullptr;
    const int64_t kb = k / 128, nb = (n + 127) / 128;
    Buf a(m * k), w(n * k), out(m * n * 2), rs(m * 4), cs(n * 4), sa(m * kb * 4), sb(nb * kb * 4), scratch(64u << 20);
    if (!a.p || !w.p || !out.p || !rs.p || !cs.p || !sa.p || !sb.p || !scratch.p) return false;
    cudaMemset(a.p, 0x38, m * k);   // e4m3 1.0
    cudaMemset(w.p, 0x38, n * k);
    ScaledGemmArgs args{};
    args.recipe = ue8 ? GemmRecipe::kFP8BlockwiseUE8M0
                      : block ? GemmRecipe::kFP8Blockwise
                              : (row ? GemmRecipe::kFP8Rowwise : GemmRecipe::kFP8Tensorwise);
    args.a.data = a.p; args.a.ld = k; args.a.dtype = DType::kFloat8E4M3;
    args.b.data = w.p; args.b.ld = k; args.b.dtype = DType::kFloat8E4M3;
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    if (row) {
      args.epi.ops = kEpiRowScale | kEpiColScale;
      args.epi.row_scale = static_cast<const float*>(rs.p);
      args.epi.col_scale = static_cast<const float*>(cs.p);
    }
    if (block) {
      args.a.scale = sa.p; args.b.scale = sb.p;
      args.a.scale_dtype = ue8 ? DType::kFloat8E8M0 : DType::kFloat32;
      args.b.scale_dtype = ue8 ? DType::kFloat8E8M0 : DType::kFloat32;
      args.a.granularity = kops::ScaleGranularity::kPerGroup;
      args.b.granularity = kops::ScaleGranularity::kPerBlock2D;
    }
    kops::Workspace ws{};
    ws.device_scratch = scratch.p; ws.device_scratch_bytes = 64u << 20;
    if (std::strstr(id, ".wgmma.") != nullptr) {
      // B11 SM90a cell: tactic (tile) and accumulation mode from the case id.
      int tm = 128, tn = 128;
      if (const char* tt = std::strstr(id, ".t"); tt != nullptr) std::sscanf(tt + 2, "%dx%d", &tm, &tn);
      GemmTactic t{};
      t.tile_m = static_cast<int16_t>(tm); t.tile_n = static_cast<int16_t>(tn); t.tile_k = 128;
      t.mainloop = tm <= 64 ? MainloopSchedule::kTmaWarpSpecialized : MainloopSchedule::kTmaWarpSpecializedCooperative;
      t.epilogue = EpilogueSchedule::kTmaStore;
      t.persistent = true;
      args.fp8_accum = std::strstr(id, ".fast.") != nullptr ? Fp8Accum::kFastAccum : Fp8Accum::kPromoted;
      if (row) {   // the cell takes the rowwise scales as operand scales (same math as the epilogue form)
        args.epi.ops = kEpiNone;
        args.a.scale = rs.p; args.a.granularity = kops::ScaleGranularity::kPerToken;
        args.b.scale = cs.p; args.b.granularity = kops::ScaleGranularity::kPerChannel;
      }
      return go([&] { return fp8_gemm(args, &t, GemmBackend::kNative, ws, ctx); });
    }
    return go([&] { return fp8_gemm(args, nullptr, GemmBackend::kNative, ws, ctx); });
  }
  // B11 SM90a grouped cell: id `gemm.wgmma.grouped.<dt>.<contig|masked>.g<G>.r<R>.<n>x<k>.t<tm>x<tn>`.
  if (c.op.rfind("wgmma_grouped_", 0) == 0) {
    const bool masked = std::strstr(id, ".masked.") != nullptr;
    const bool ue8 = c.op == "wgmma_grouped_fp8ue8m0";
    const bool fp8 = c.op == "wgmma_grouped_fp8" || ue8;
    const bool f16 = c.op == "wgmma_grouped_fp16";
    int G = 8;
    // `.g<G>` after the `.grouped.` marker (a bare strstr(id, ".g") would match ".grouped").
    if (const char* gp = std::strstr(id, ".grouped."); gp != nullptr) {
      if (const char* gg = std::strstr(gp + 9, ".g"); gg != nullptr) std::sscanf(gg + 2, "%d", &G);
    }
    int tm = 128, tn = 128;
    if (const char* tt = std::strstr(id, ".t"); tt != nullptr) std::sscanf(tt + 2, "%dx%d", &tm, &tn);
    const int64_t R = m;                       // routed rows = flops / (2 N K)
    if (G < 1 || R < 1) return false;
    const int64_t per = (R + G - 1) / G;       // masked capacity per group
    const int64_t cap = masked ? per * G : R;
    std::vector<int64_t> rows_g(static_cast<size_t>(G));
    for (int g = 0; g < G; ++g) rows_g[static_cast<size_t>(g)] = R / G + (g < R % G ? 1 : 0);
    const int64_t eb = fp8 ? 1 : 2;
    const int64_t kb = k / 128, nb = (n + 127) / 128;
    const int64_t sb_bytes = ue8 ? 1 : 4;
    Buf a(cap * k * eb), b(static_cast<size_t>(G) * n * k * eb), out(cap * n * 2);
    Buf sa(fp8 ? cap * kb * sb_bytes : 1), sb(fp8 ? static_cast<size_t>(G) * nb * kb * sb_bytes : 1);
    Buf meta(static_cast<size_t>(G + 1) * 8);
    if (!a.p || !b.p || !out.p || !meta.p || (fp8 && (!sa.p || !sb.p))) return false;
    if (masked) {
      std::vector<int32_t> mm(static_cast<size_t>(G));
      for (int g = 0; g < G; ++g) mm[static_cast<size_t>(g)] = static_cast<int32_t>(rows_g[static_cast<size_t>(g)]);
      cudaMemcpy(meta.p, mm.data(), mm.size() * 4, cudaMemcpyHostToDevice);
    } else {
      std::vector<int64_t> off(static_cast<size_t>(G + 1));
      int64_t acc = 0;
      for (int g = 0; g < G; ++g) { off[static_cast<size_t>(g)] = acc; acc += rows_g[static_cast<size_t>(g)]; }
      off[static_cast<size_t>(G)] = acc;
      cudaMemcpy(meta.p, off.data(), off.size() * 8, cudaMemcpyHostToDevice);
    }
    GroupedGemmParams p{};
    p.layout = masked ? GroupedLayout::kMasked : GroupedLayout::kContiguous;
    p.recipe = ue8 ? GemmRecipe::kFP8BlockwiseUE8M0
                   : fp8 ? GemmRecipe::kFP8Blockwise : (f16 ? GemmRecipe::kFP16 : GemmRecipe::kBF16);
    p.num_groups = G;
    p.a = a.p; p.lda = k;
    p.b = b.p; p.ldb = k; p.b_group_stride = n * k;
    p.d = out.p; p.ldd = n;
    p.n = static_cast<int32_t>(n); p.k = static_cast<int32_t>(k);
    p.max_m = static_cast<int32_t>(masked ? per : cap);
    if (masked) p.masked_m = static_cast<const int32_t*>(meta.p);
    else p.expert_offsets = static_cast<const int64_t*>(meta.p);
    if (fp8) {
      p.a_scale = sa.p;
      p.a_scale_layout = kops::ScaleLayout::kRowMajor;
      p.b_scale = sb.p;
      p.b_scale_group_stride = nb * kb;
    }
    GemmTactic t{};
    t.tile_m = static_cast<int16_t>(tm); t.tile_n = static_cast<int16_t>(tn);
    t.tile_k = static_cast<int16_t>(fp8 ? 128 : 64);
    t.mainloop = tm <= 64 ? MainloopSchedule::kTmaWarpSpecialized
                          : MainloopSchedule::kTmaWarpSpecializedCooperative;
    t.epilogue = EpilogueSchedule::kTmaStore;
    t.persistent = true;
    kops::Workspace ws{};
    return go([&] { return grouped_gemm(p, &t, GemmBackend::kNative, ws, ctx); });
  }
  // B11 SM90a cell: same op and operands as the B7 mma cases, selected by the case id.
  if (c.op.rfind("mma_", 0) == 0 && std::strstr(id, ".wgmma.") != nullptr) {
    const bool f16 = c.dtype == DType::kFloat16;
    Buf a(m * k * 2), w(n * k * 2), out(m * n * 2);
    if (!a.p || !w.p || !out.p) return false;
    ScaledGemmArgs args{};
    args.recipe = f16 ? GemmRecipe::kFP16 : GemmRecipe::kBF16;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = c.dtype;
    args.b.data = w.p; args.b.ld = k; args.b.dtype = c.dtype;
    args.d = out.p; args.ldd = n; args.d_dtype = c.dtype;
    args.m = m; args.n = n; args.k = k;
    int tm = 128, tn = 128, cm = 1, cn = 1;
    if (const char* tt = std::strstr(id, ".t"); tt != nullptr) std::sscanf(tt + 2, "%dx%d.c%dx%d", &tm, &tn, &cm, &cn);
    GemmTactic t{};
    t.tile_m = static_cast<int16_t>(tm); t.tile_n = static_cast<int16_t>(tn); t.tile_k = 64;
    t.cluster_m = static_cast<int8_t>(cm); t.cluster_n = static_cast<int8_t>(cn); t.tma_multicast = cm * cn > 1;
    t.mainloop = tm <= 64 ? MainloopSchedule::kTmaWarpSpecialized : MainloopSchedule::kTmaWarpSpecializedCooperative;
    t.epilogue = EpilogueSchedule::kTmaStore;
    t.persistent = true;
    kops::Workspace ws{};
    return go([&] { return gemm_with_tactic(args, t, ws, ctx); });
  }
  // B13 tile cell: same op and operands as the B7 mma cases, selected by the case id.
  if (c.op.rfind("mma_", 0) == 0 && std::strstr(id, "mma_sm120") != nullptr) {
    Buf a(m * k * 2), w(n * k * 2), out(m * n * 2);
    if (!a.p || !w.p || !out.p) return false;
    ScaledGemmArgs args{};
    args.recipe = GemmRecipe::kBF16;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = DType::kBFloat16;
    args.b.data = w.p; args.b.ld = k; args.b.dtype = DType::kBFloat16;
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    int tm = 64, tn = 128;
    if (const char* tt = std::strstr(id, ".t"); tt != nullptr) std::sscanf(tt + 2, "%dx%d", &tm, &tn);
    GemmTactic t{};
    t.tile_m = static_cast<int16_t>(tm); t.tile_n = static_cast<int16_t>(tn); t.tile_k = 32;
    t.mainloop = MainloopSchedule::kSm120WarpSpecialized;
    t.epilogue = EpilogueSchedule::kDirectStore;
    kops::Workspace ws{};
    return go([&] { return gemm_with_tactic(args, t, ws, ctx); });
  }
  if (c.op.rfind("mma_", 0) == 0) {
    const bool f16 = c.dtype == DType::kFloat16;
    const bool tuned = c.op.size() > 5 && c.op.compare(c.op.size() - 6, 6, "_tuned") == 0;
    Buf a(m * k * 2), w(n * k * 2), out(m * n * 2), scratch(64u << 20);
    if (!a.p || !w.p || !out.p || !scratch.p) return false;
    ScaledGemmArgs args{};
    args.recipe = f16 ? GemmRecipe::kFP16 : GemmRecipe::kBF16;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = c.dtype;
    args.b.data = w.p; args.b.ld = k; args.b.dtype = c.dtype;
    args.d = out.p; args.ldd = n; args.d_dtype = c.dtype;
    args.m = m; args.n = n; args.k = k;
    kops::Workspace ws{};
    ws.device_scratch = scratch.p; ws.device_scratch_bytes = 64u << 20;
    if (tuned) {
      GemmProblem pr{};
      pr.m = m; pr.n = n; pr.k = k; pr.recipe = args.recipe;
      const int64_t bucket = m;
      if (!status_ok(autotune(pr, &bucket, 1, GemmBackend::kNative, ws, ctx), id)) return false;
    }
    return go([&] { return scaled_gemm(args, nullptr, GemmBackend::kNative, ws, ctx); });
  }
  if (c.op == "dual_bf16") {
    const int64_t dn = c.rows, dk = c.cols, dm = c.flops / (4 * dn * dk);
    Buf a(dm * dk * 2), w0(dn * dk * 2), w1(dn * dk * 2), out(dm * dn * 2);
    if (!a.p || !w0.p || !w1.p || !out.p) return false;
    DualGemmParams p{};
    p.a = a.p; p.lda = dk; p.b0 = w0.p; p.ldb0 = dk; p.b1 = w1.p; p.ldb1 = dk; p.d = out.p; p.ldd = dn;
    p.act = kops::ActFn::kSiLU; p.m = static_cast<int32_t>(dm); p.n = static_cast<int32_t>(dn); p.k = static_cast<int32_t>(dk);
    return go([&] { return dual_gemm(p, DType::kBFloat16, ctx); });
  }
  if (c.op == "bmm_mla_uk" || c.op == "bmm_mla_uv") {
    const int64_t batch = 128, bk = c.cols, bn = c.rows, bm = c.flops / (2 * batch * bn * bk);
    Buf a(bm * batch * bk * 2), w(batch * bk * bn * 2), out(bm * batch * bn * 2);
    if (!a.p || !w.p || !out.p) return false;
    BmmParams p{};
    p.a = a.p; p.lda = batch * bk; p.stride_a_batch = bk;
    p.b = w.p; p.ldb = bn; p.stride_b_batch = bk * bn;
    p.d = out.p; p.ldd = batch * bn; p.stride_d_batch = bn;
    p.b_k_major = false; p.batch = static_cast<int32_t>(batch); p.m = static_cast<int32_t>(bm); p.n = static_cast<int32_t>(bn); p.k = static_cast<int32_t>(bk);
    return go([&] { return bmm(p, DType::kBFloat16, DType::kBFloat16, DType::kBFloat16, ctx); });
  }
  if (c.op == "int8_mma" || c.op == "w4a8_g128" || c.op == "w4a8_chn") {
    const bool w4 = c.op != "int8_mma";
    const int64_t groups = c.op == "w4a8_g128" ? k / 128 : 1;
    Buf a(m * k), w(w4 ? n * k / 2 : n * k), s2(groups * n), zp(groups * n), rs(m * 4), cs(n * 4), out(m * n * 2), scratch(64u << 20);
    if (!a.p || !w.p || !s2.p || !zp.p || !rs.p || !cs.p || !out.p || !scratch.p) return false;
    cudaMemset(s2.p, 1, groups * n);   // q * 1 - 0: in the s8 range
    cudaMemset(zp.p, 0, groups * n);
    ScaledGemmArgs args{};
    args.recipe = w4 ? GemmRecipe::kW4A8IntInt8 : GemmRecipe::kINT8Rowwise;
    args.a.data = a.p; args.a.ld = k; args.a.dtype = DType::kInt8;
    args.b.data = w.p; args.b.ld = w4 ? k / 2 : k; args.b.dtype = w4 ? DType::kUInt4x2 : DType::kInt8;
    if (w4) {
      args.b.group_k = static_cast<int16_t>(c.op == "w4a8_g128" ? 128 : 0);
      args.b.second_level = c.op == "w4a8_g128" ? s2.p : nullptr;
      args.b.zero_points = zp.p; args.b.zero_dtype = DType::kUInt8;
    }
    args.d = out.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = m; args.n = n; args.k = k;
    args.epi.ops = kEpiRowScale | kEpiColScale;
    args.epi.row_scale = static_cast<const float*>(rs.p); args.epi.col_scale = static_cast<const float*>(cs.p);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p; ws.device_scratch_bytes = 64u << 20;
    return go([&] { return int8_gemm(args, nullptr, GemmBackend::kNative, ws, ctx); });
  }
  *handled = false;
  return true;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  bool handled = false;
  const bool ok = run_b5_case(c, ctx, st, &handled);
  if (handled) return ok;
  const int64_t r = c.rows;
  const int64_t k = c.cols;
  const size_t n = static_cast<size_t>(r * k);
  const char* id = c.id.c_str();
  auto go = [&](auto&& op) {
    if (!status_ok(op(), id)) return false;
    return measure(c, [&] { (void)op(); }, st);
  };
  if (c.op == "weight_transpose") {
    const size_t eb = c.dtype == kops::DType::kFloat8E4M3 ? 1 : 2;
    Buf in(n * eb), out(n * eb);
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, c.dtype, r, k);
    const kops::Tensor to = tensor2d(out.p, c.dtype, k, r);
    return go([&] { return kops::gemm::weight_transpose(to, ti, ctx); });
  }
  if (c.op == "weight_transpose_int4") {
    Buf in(n / 2), out(n / 2);
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, kops::DType::kInt4x2, r, k);
    const kops::Tensor to = tensor2d(out.p, kops::DType::kInt4x2, k, r);
    return go([&] { return kops::gemm::weight_transpose(to, ti, ctx); });
  }
  if (c.op == "glu_interleave") {
    Buf in(n * 2), out(n * 2);
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, c.dtype, r, k);
    const kops::Tensor to = tensor2d(out.p, c.dtype, r, k);
    // [gate | up] halves -> blocks of 64 rows (a 128-row N tile holds matching gate/up rows).
    return go([&] {
      return kops::gemm::glu_weight_interleave(to, kops::Tensor{}, kops::Tensor{}, ti, kops::Tensor{},
                                               kops::Tensor{}, 0, 64, ctx);
    });
  }
  if (c.op == "router_split") {
    Buf w(n * 4), hi(n * 2), lo(n * 2);
    if (w.p == nullptr || hi.p == nullptr || lo.p == nullptr) return false;
    const kops::Tensor tw = tensor2d(w.p, kops::DType::kFloat32, r, k);
    const kops::Tensor thi = tensor2d(hi.p, kops::DType::kBFloat16, r, k);
    const kops::Tensor tlo = tensor2d(lo.p, kops::DType::kBFloat16, r, k);
    return go([&] { return kops::gemm::router_weight_split_bf16x2(thi, tlo, tw, ctx); });
  }
  if (c.op == "int8_colsum") {
    Buf w(n), sum(static_cast<size_t>(r) * 4);
    if (w.p == nullptr || sum.p == nullptr) return false;
    const kops::Tensor tw = tensor2d(w.p, kops::DType::kInt8, r, k);
    const kops::Tensor ts = tensor1d(sum.p, kops::DType::kInt32, r);
    return go([&] { return kops::gemm::int8_weight_colsum(ts, tw, ctx); });
  }
  if (c.op == "block_scale_swizzle") {
    const int64_t rp = (r + 127) / 128 * 128;
    const int64_t cp = (k + 3) / 4 * 4;
    Buf in(n), out(static_cast<size_t>(rp * cp));
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, kops::DType::kFloat8E4M3, r, k);
    const kops::Tensor to = tensor2d(out.p, kops::DType::kFloat8E4M3, rp, cp);
    return go([&] { return kops::gemm::block_scale_swizzle(to, ti, false, ctx); });
  }
  if (c.op == "fp8_scale_to_tma" || c.op == "fp8_scale_to_tma_ue8m0") {
    const bool pack = c.op == "fp8_scale_to_tma_ue8m0";
    Buf in(n * 4), out(n * 4);
    if (in.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor ti = tensor2d(in.p, kops::DType::kFloat32, r, k);
    const kops::Tensor to = pack ? tensor2d(out.p, kops::DType::kInt32, k / 4, r)
                                 : tensor2d(out.p, kops::DType::kFloat32, k, r);
    return go([&] { return kops::gemm::fp8_scale_to_tma_layout(to, ti, kops::Tensor{}, pack, ctx); });
  }
  if (c.op == "fp8_block_quant" || c.op == "fp8_block_quant_ue8m0") {
    const bool ue = c.op == "fp8_block_quant_ue8m0";
    Buf w(n * 2), q(n), s(static_cast<size_t>((r / 128) * (k / 128)) * 4);
    if (w.p == nullptr || q.p == nullptr || s.p == nullptr) return false;
    const kops::Tensor tw = tensor2d(w.p, kops::DType::kBFloat16, r, k);
    const kops::Tensor tq = tensor2d(q.p, kops::DType::kFloat8E4M3, r, k);
    const kops::Tensor ts = tensor2d(s.p, kops::DType::kFloat32, r / 128, k / 128);
    return go([&] { return kops::gemm::fp8_block2d_quant_weight(tq, ts, tw, 128, 128, ue, ctx); });
  }
  if (c.op == "machete_prepack_int4" || c.op == "machete_prepack_int8") {
    const bool nib = c.op == "machete_prepack_int4";
    const kops::DType dt = nib ? kops::DType::kInt4x2 : kops::DType::kInt8;
    Buf w(n * (nib ? 1 : 2) / 2), out(n * (nib ? 1 : 2) / 2);
    if (w.p == nullptr || out.p == nullptr) return false;
    const kops::Tensor tw = tensor2d(w.p, dt, r, k);
    const kops::Tensor to = tensor2d(out.p, dt, r, k);
    const int bits = nib ? 4 : 8;
    const int atom = nib ? 0 : 1;
    return go([&] { return kops::gemm::machete_prepack(to, tw, bits, atom, ctx); });
  }
  if (c.op == "grouped_prepare") {
    const int experts = 32, rows_per_group = 128;
    const int64_t total_rows = static_cast<int64_t>(experts) * rows_per_group;
    std::vector<int64_t> offs(experts + 1);
    for (int e = 0; e <= experts; ++e) offs[e] = static_cast<int64_t>(e) * rows_per_group;
    Buf a(static_cast<size_t>(total_rows) * k * 2), w(static_cast<size_t>(experts) * r * k * 2),
        d(static_cast<size_t>(total_rows) * r * 2);
    if (a.p == nullptr || w.p == nullptr || d.p == nullptr) return false;
    int64_t* d_offs = nullptr;
    if (!cuda_ok(cudaMalloc(&d_offs, offs.size() * 8), "cudaMalloc expert_offsets")) return false;
    cudaMemcpy(d_offs, offs.data(), offs.size() * 8, cudaMemcpyHostToDevice);
    kops::gemm::GroupedGemmParams p{};
    p.layout = kops::gemm::GroupedLayout::kContiguous;
    p.recipe = kops::gemm::GemmRecipe::kBF16;
    p.num_groups = experts;
    p.a = a.p; p.lda = k;
    p.expert_offsets = d_offs;
    p.b = w.p; p.ldb = k; p.b_group_stride = r * k;
    p.d = d.p; p.ldd = r;
    p.n = static_cast<int32_t>(r); p.k = static_cast<int32_t>(k);
    p.max_m = static_cast<int32_t>(total_rows);
    p.expected_m = rows_per_group;
    kops::gemm::GemmTactic t{};
    t.tile_m = 128; t.tile_n = 128;
    t.mainloop = kops::gemm::MainloopSchedule::kTmaWarpSpecializedCooperative;
    kops::WorkspaceRequirements req{};
    const bool req_ok = status_ok(kops::gemm::grouped_gemm_workspace_requirements(p, &t, ctx, &req), id);
    Buf scratch(req.device_scratch_bytes + 64);
    if (!req_ok || scratch.p == nullptr) {
      cudaFree(d_offs);
      return false;
    }
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    const bool ok = go([&] {
      kops::gemm::GroupedGemmParams q = p;   // prepare writes the prepared pointers into the params
      return kops::gemm::grouped_gemm_prepare(&q, t, ws, ctx);
    });
    cudaFree(d_offs);
    return ok;
  }
  std::fprintf(stderr, "[bench] unknown op %s\n", c.op.c_str());
  return false;
}

}  // namespace

int main(int argc, char** argv) {
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
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) return 1;
  // Arch-cell cases only exist on their exact SM: on any other device the launcher refuses the tactic
  // (kUnsupportedArch), which is a skip of that case, not a failure of the family.
  cases.erase(std::remove_if(cases.begin(), cases.end(),
                             [&](const Case& c) {
                               const bool sm120 = c.id.find("mma_sm120") != std::string::npos;
                               const bool sm90 = c.id.find(".wgmma.") != std::string::npos;
                               const bool skip = (sm120 && ctx.arch_id != KOPS_TARGET_SM120A) ||
                                                 (sm90 && ctx.arch_id != KOPS_TARGET_SM90A);
                               if (skip) {
                                 std::fprintf(stderr, "[bench] skip %s: needs %s, found arch %d\n", c.id.c_str(),
                                              sm120 ? "KOPS_TARGET_SM120A" : "KOPS_TARGET_SM90A", ctx.arch_id);
                               }
                               return skip;
                             }),
              cases.end());
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
