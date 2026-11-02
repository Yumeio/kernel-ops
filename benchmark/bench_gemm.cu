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
  std::fprintf(stderr, "[bench] unknown op %s\n", c.op.c_str());
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice")) return 1;
  DeviceInfo dev{};
  query_device(&dev);
  const std::vector<Case> cases = make_cases();
  if (opt.dump_cases) {
    dump_cases(cases, dev);
    return 0;
  }
  kops::LaunchContext ctx{};
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) return 1;
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
