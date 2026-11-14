// =============================================================================
// kernel-ops | benchmark/bench_quantization.cu
// B3 quantization-family cases. The case table is the shared spec read by
// benchmark/compare/torch_baseline.py through --dump-cases.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <vector>

#include "csrc/common/quant.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/quantization/ops.h"

namespace {

using namespace kops_bench;

/// Group size of a per-group case id ("quant.per_group64.bf16.fp8..." -> 64).
int group_from_id(const std::string& id) {
  const std::string key = "per_group";
  const size_t pos = id.find(key);
  if (pos == std::string::npos) return 128;
  size_t i = pos + key.size();
  int group = 0;
  while (i < id.size() && id[i] >= '0' && id[i] <= '9') {
    group = 10 * group + (id[i] - '0');
    ++i;
  }
  return group > 0 ? group : 128;
}

/// `.fast` suffixes select ActQuantOut::math = kReciprocal: the approximate quotient (one RN
/// reciprocal and a raw multiply, at most one payload code away), which skips the Markstein
/// correction that kExact uses for 16-bit inputs.
bool fast_from_id(const std::string& id) { return id.find(".fast") != std::string::npos; }

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto add = [&](const char* id, const char* op, kops::DType dt, int64_t rows, int64_t cols,
                 int64_t bytes, int64_t flops) {
    cases.push_back({id, op, dt, rows, cols, bytes, flops, 20, 200});
  };
  const int64_t r = 4096;
  const int64_t c = 4096;
  add("quant.absmax.bf16.4096x4096", "absmax", kops::DType::kBFloat16, r, c, 2 * r * c, r * c);
  add("quant.absmax.fp32.4096x4096", "absmax", kops::DType::kFloat32, r, c, 4 * r * c, r * c);
  add("quant.static.bf16.fp8.4096x4096", "quantize_static", kops::DType::kBFloat16, r, c,
      3 * r * c, r * c);
  add("quant.per_token.bf16.fp8.4096x4096", "quantize_per_token", kops::DType::kBFloat16, r, c,
      3 * r * c, 3 * r * c);
  add("quant.per_token.fp32.int8.4096x4096", "quantize_per_token_int8", kops::DType::kFloat32, r,
      c, 5 * r * c, 3 * r * c);
  // 1 x G activation quantization: 2 B read + 1 B written per element + one fp32 scale per group.
  auto add_per_group = [&](int64_t group, bool fast) {
    char id[64];
    std::snprintf(id, sizeof(id), "quant.per_group%lld.bf16.fp8.4096x4096%s",
                  static_cast<long long>(group), fast ? ".fast" : "");
    const int64_t scale_bytes = 4 * (r * c / group);
    add(id, "quantize_per_group", kops::DType::kBFloat16, r, c, 3 * r * c + scale_bytes,
        3 * r * c);
  };
  for (int64_t group : {128, 64, 32}) {
    add_per_group(group, false);
    add_per_group(group, true);
  }
  add("quant.mxfp8.bf16.4096x4096", "quantize_mxfp8", kops::DType::kBFloat16, r, c, 3 * r * c,
      3 * r * c);
  add("quant.mxfp4.bf16.4096x4096", "quantize_mxfp4", kops::DType::kBFloat16, r, c,
      3 * r * c / 2, 3 * r * c);
  add("quant.nvfp4.bf16.4096x4096", "quantize_nvfp4", kops::DType::kBFloat16, r, c,
      3 * r * c / 2, 3 * r * c);
  add("quant.dequant_per_token.fp8.fp32.4096x4096", "dequantize_per_token",
      kops::DType::kFloat32, r, c, 5 * r * c, r * c);
  add("quant.pack_int4.4096x4096", "pack_int4", kops::DType::kFloat32, r, c, r * c * 3 / 2, r * c);
  add("quant.unpack_int4.4096x4096", "unpack_int4", kops::DType::kFloat32, r, c, r * c * 3 / 2,
      r * c);
  add("quant.hadamard.fp32.4096x4096", "hadamard", kops::DType::kFloat32, r, 4096, 2 * r * 4096,
      r * 4096 * 12);

  // ---- B4 weight repack: bytes = compulsory input + output traffic ----------------------
  const int64_t pk = r * c / 2;  // 4-bit payload bytes
  add("quant.gptq_marlin.4bit.g128.4096x4096", "gptq_to_marlin", kops::DType::kFloat16, r, c,
      2 * pk + 4 * (r / 128) * c, 0);
  add("quant.gptq_marlin.4bit.g128.actorder.4096x4096", "gptq_to_marlin_actorder",
      kops::DType::kFloat16, r, c, 2 * pk + 4 * (r / 128) * c + 12 * r, 0);
  add("quant.awq_marlin.4bit.g128.4096x4096", "awq_to_marlin", kops::DType::kFloat16, r, c,
      2 * pk + 8 * (r / 128) * c, 0);
  add("quant.awq_dequant.fp16.g128.4096x4096", "awq_dequantize", kops::DType::kFloat16, r, c,
      pk + 2 * r * c + 3 * (r / 128) * c, r * c);
  add("quant.gguf_q4_k.fp16.4096x4096", "gguf_dequantize_q4_k", kops::DType::kFloat16, r, c,
      r * c / 256 * 144 + 2 * r * c, r * c);
  add("quant.gguf_q8_0.fp16.4096x4096", "gguf_dequantize_q8_0", kops::DType::kFloat16, r, c,
      r * c / 32 * 34 + 2 * r * c, r * c);
  add("quant.fp4_marlin.mxfp4.4096x4096", "fp4_to_marlin", kops::DType::kUInt8, r, c,
      2 * pk + 2 * r * (c / 32), 0);
  add("quant.fp8_to_ue8m0.4096x4096", "fp8_block_scale_to_ue8m0", kops::DType::kFloat8E4M3, r, c,
      2 * r * c, r * c);
  add("quant.smoothquant_fold.bf16.4096x4096", "smoothquant_fold", kops::DType::kBFloat16, r, c,
      4 * r * c, r * c);
  return cases;
}

template <typename T>
bool run_typed(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  uint8_t* q = nullptr;
  float* scales = nullptr;
  int32_t* zero = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), n * sizeof(T)), "in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), n), "q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), (c.rows + 1) * sizeof(float)),
               "scales") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&zero), c.rows * sizeof(int32_t)), "zero")) {
    return false;
  }
  cudaMemset(in, 0, n * sizeof(T));
  cudaMemset(q, 0, n);
  cudaMemset(scales, 0, (c.rows + 1) * sizeof(float));
  const kops::Tensor tin = tensor2d(in, c.dtype, c.rows, c.cols);
  bool ok = false;
  if (c.op == "absmax") {
    kops::Tensor amax = tensor1d(scales, kops::DType::kFloat32, 1);
    cudaMemset(scales, 0, sizeof(float));
    ok = status_ok(kops::quant::absmax(amax, tin, ctx), c.id.c_str());
    if (ok) ok = measure(c, [&] { (void)kops::quant::absmax(amax, tin, ctx); }, st);
  } else if (c.op == "quantize_static") {
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols;
    o.q_dtype = kops::DType::kFloat8E4M3;
    o.mode = kops::ActQuantMode::kStaticPerTensor;
    o.static_scale = scales;
    const float one = 1.0f;
    cudaMemcpy(scales, &one, sizeof(float), cudaMemcpyHostToDevice);
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
  } else if (c.op == "quantize_per_token" || c.op == "quantize_per_token_int8") {
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols;
    o.scales = scales;
    o.scale_row_stride = 1;
    o.q_dtype = c.op == "quantize_per_token" ? kops::DType::kFloat8E4M3 : kops::DType::kInt8;
    o.mode = kops::ActQuantMode::kDynamicPerToken;
    o.symmetric = o.q_dtype != kops::DType::kInt8;
    o.zero_points = o.symmetric ? nullptr : zero;
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
  } else if (c.op == "quantize_per_group") {
    // The scale buffer is rows * (cols / group) fp32 words, so it has to follow the case's group.
    const int group = group_from_id(c.id);
    const int64_t groups = c.cols / group;
    float* pg = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pg),
                            static_cast<size_t>(c.rows * groups) * sizeof(float)),
                 "per-group scales")) {
      return false;
    }
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols;
    o.scales = pg;
    o.scale_row_stride = groups;
    o.q_dtype = kops::DType::kFloat8E4M3;
    o.mode = kops::ActQuantMode::kPerGroup;
    o.group_size = group;
    o.math = fast_from_id(c.id) ? kops::QuantMath::kReciprocal : kops::QuantMath::kExact;
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
    cudaFree(pg);
  } else if (c.op == "quantize_mxfp8" || c.op == "quantize_mxfp4") {
    const int64_t groups = c.cols / 32;
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.op == "quantize_mxfp4" ? c.cols / 2 : c.cols;
    o.scales = scales;
    o.scale_row_stride = groups;
    o.q_dtype = c.op == "quantize_mxfp4" ? kops::DType::kFloat4E2M1x2 : kops::DType::kFloat8E4M3;
    o.mode = c.op == "quantize_mxfp4" ? kops::ActQuantMode::kMXFP4 : kops::ActQuantMode::kMXFP8;
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
  } else if (c.op == "quantize_nvfp4") {
    const int64_t groups = c.cols / 16;
    const int64_t sf_bytes = ((c.rows + 127) / 128) * ((groups + 3) / 4) * 512;
    const int64_t sf_padded = (sf_bytes + 3) & ~int64_t(3);
    uint8_t* sf = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sf), sf_padded + sizeof(float)), "sf")) return false;
    float* gscale = reinterpret_cast<float*>(sf + sf_padded);
    float g = 1.0f;
    cudaMemcpy(gscale, &g, sizeof(float), cudaMemcpyHostToDevice);
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols / 2;
    o.scales = sf;
    o.scale_row_stride = groups;
    o.q_dtype = kops::DType::kFloat4E2M1x2;
    o.mode = kops::ActQuantMode::kNVFP4;
    o.global_scale = gscale;
    ok = status_ok(kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::quantize_activation(o, tin, kops::Tensor{}, 0, ctx);
    }, st);
    cudaFree(sf);
  } else if (c.op == "dequantize_per_token") {
    float* out = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(float)), "out")) return false;
    kops::QuantScheme scheme{};
    scheme.storage_dtype = kops::DType::kFloat8E4M3;
    scheme.scale_dtype = kops::DType::kFloat32;
    scheme.granularity = kops::ScaleGranularity::kPerToken;
    const kops::Tensor tq = tensor2d(q, kops::DType::kFloat8E4M3, c.rows, c.cols);
    const kops::Tensor tsc = tensor1d(scales, kops::DType::kFloat32, c.rows);
    const kops::Tensor tout = tensor2d(out, kops::DType::kFloat32, c.rows, c.cols);
    ok = status_ok(kops::quant::dequantize(tout, tq, tsc, kops::Tensor{}, scheme, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::dequantize(tout, tq, tsc, kops::Tensor{}, scheme, ctx);
    }, st);
    cudaFree(out);
  } else if (c.op == "pack_int4" || c.op == "unpack_int4") {
    int8_t* i8 = nullptr;
    uint8_t* u8 = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&i8), n), "i8") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&u8), n / 2), "u8")) {
      return false;
    }
    if (c.op == "pack_int4") {
      const kops::Tensor ti = tensor2d(i8, kops::DType::kInt8, c.rows, c.cols);
      const kops::Tensor to = tensor2d(u8, kops::DType::kUInt8, c.rows, c.cols / 2);
      ok = status_ok(kops::quant::pack_int4(to, ti, false, ctx), c.id.c_str());
      if (ok) ok = measure(c, [&] { (void)kops::quant::pack_int4(to, ti, false, ctx); }, st);
    } else {
      const kops::Tensor ti = tensor2d(u8, kops::DType::kUInt8, c.rows, c.cols / 2);
      const kops::Tensor to = tensor2d(i8, kops::DType::kInt8, c.rows, c.cols);
      ok = status_ok(kops::quant::unpack_int4(to, ti, false, ctx), c.id.c_str());
      if (ok) ok = measure(c, [&] { (void)kops::quant::unpack_int4(to, ti, false, ctx); }, st);
    }
    cudaFree(i8);
    cudaFree(u8);
  } else if (c.op == "hadamard") {
    float* out = nullptr;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(float)), "out")) return false;
    const kops::Tensor ti = tensor2d(in, kops::DType::kFloat32, c.rows, c.cols);
    const kops::Tensor to = tensor2d(out, kops::DType::kFloat32, c.rows, c.cols);
    ok = status_ok(kops::quant::hadamard_transform(to, ti, 1.0f / std::sqrt(float(c.cols)), false,
                                                   0, ctx),
                   c.id.c_str());
    if (ok) ok = measure(c, [&] {
      (void)kops::quant::hadamard_transform(to, ti, 1.0f / std::sqrt(float(c.cols)), false, 0, ctx);
    }, st);
    cudaFree(out);
  }
  cudaFree(in);
  cudaFree(q);
  cudaFree(scales);
  cudaFree(zero);
  return ok;
}

struct B4Buf {
  void* p = nullptr;
  explicit B4Buf(size_t bytes) {
    if (!cuda_ok(cudaMalloc(&p, bytes), "cudaMalloc")) p = nullptr;
    else cudaMemset(p, 0x3C, bytes);
  }
  ~B4Buf() { if (p != nullptr) cudaFree(p); }
  B4Buf(const B4Buf&) = delete;
  B4Buf& operator=(const B4Buf&) = delete;
};

bool is_b4_op(const std::string& op) {
  return op == "gptq_to_marlin" || op == "gptq_to_marlin_actorder" || op == "awq_to_marlin" ||
         op == "awq_dequantize" || op == "gguf_dequantize_q4_k" || op == "gguf_dequantize_q8_0" ||
         op == "fp4_to_marlin" || op == "fp8_block_scale_to_ue8m0" || op == "smoothquant_fold";
}

/// Inputs only need to be well formed: the content never changes the amount of work.
bool run_b4(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t k = c.rows;
  const int64_t n = c.cols;
  const char* id = c.id.c_str();
  auto go = [&](auto&& op) {
    if (!status_ok(op(), id)) return false;
    return measure(c, [&] { (void)op(); }, st);
  };
  const int64_t groups = k / 128;
  if (c.op == "gptq_to_marlin" || c.op == "gptq_to_marlin_actorder") {
    const bool act = c.op == "gptq_to_marlin_actorder";
    B4Buf qw(static_cast<size_t>(k * n / 2)), out(static_cast<size_t>(k * n / 2)),
        sc(static_cast<size_t>(groups * n * 2)), osc(static_cast<size_t>(groups * n * 2)),
        gi(static_cast<size_t>(k * 4)), perm(static_cast<size_t>(k * 4));
    if (qw.p == nullptr || out.p == nullptr || sc.p == nullptr || osc.p == nullptr ||
        gi.p == nullptr || perm.p == nullptr) {
      return false;
    }
    if (act) {
      std::vector<int32_t> g(static_cast<size_t>(k));
      for (int64_t i = 0; i < k; ++i) {
        g[static_cast<size_t>(i)] = static_cast<int32_t>((static_cast<uint64_t>(i) * 2654435761u) % groups);
      }
      cudaMemcpy(gi.p, g.data(), g.size() * 4, cudaMemcpyHostToDevice);
    }
    const kops::Tensor tg = act ? tensor1d(gi.p, kops::DType::kInt32, k) : kops::Tensor{};
    return go([&] {
      return kops::quant::gptq_to_marlin(tensor2d(out.p, kops::DType::kInt32, k / 16, n * 16 / 8),
                                         tensor2d(osc.p, kops::DType::kFloat16, groups, n),
                                         tensor1d(perm.p, kops::DType::kInt32, k),
                                         tensor2d(qw.p, kops::DType::kInt32, k / 8, n),
                                         tensor2d(sc.p, kops::DType::kFloat16, groups, n), tg, 4, 128,
                                         true, ctx);
    });
  }
  if (c.op == "awq_to_marlin") {
    B4Buf qw(static_cast<size_t>(k * n / 2)), out(static_cast<size_t>(k * n / 2)),
        sc(static_cast<size_t>(groups * n * 2)), osc(static_cast<size_t>(groups * n * 2)),
        qz(static_cast<size_t>(groups * n / 2)), oz(static_cast<size_t>(groups * n / 2));
    if (qw.p == nullptr || out.p == nullptr || sc.p == nullptr || osc.p == nullptr ||
        qz.p == nullptr || oz.p == nullptr) {
      return false;
    }
    return go([&] {
      return kops::quant::awq_to_marlin(tensor2d(out.p, kops::DType::kInt32, k / 16, n * 16 / 8),
                                        tensor2d(osc.p, kops::DType::kFloat16, groups, n),
                                        tensor2d(oz.p, kops::DType::kInt32, groups, n / 8),
                                        tensor2d(qw.p, kops::DType::kInt32, k, n / 8),
                                        tensor2d(sc.p, kops::DType::kFloat16, groups, n),
                                        tensor2d(qz.p, kops::DType::kInt32, groups, n / 8), 4, 128, ctx);
    });
  }
  if (c.op == "awq_dequantize") {
    B4Buf qw(static_cast<size_t>(k * n / 2)), out(static_cast<size_t>(k * n * 2)),
        sc(static_cast<size_t>(groups * n * 2)), qz(static_cast<size_t>(groups * n / 2));
    if (qw.p == nullptr || out.p == nullptr || sc.p == nullptr || qz.p == nullptr) return false;
    return go([&] {
      return kops::quant::awq_dequantize(tensor2d(out.p, kops::DType::kFloat16, k, n),
                                         tensor2d(qw.p, kops::DType::kInt32, k, n / 8),
                                         tensor2d(sc.p, kops::DType::kFloat16, groups, n),
                                         tensor2d(qz.p, kops::DType::kInt32, groups, n / 8), 128, ctx);
    });
  }
  if (c.op == "gguf_dequantize_q4_k" || c.op == "gguf_dequantize_q8_0") {
    const bool q4k = c.op == "gguf_dequantize_q4_k";
    const size_t bytes = q4k ? static_cast<size_t>(k * n / 256 * 144) : static_cast<size_t>(k * n / 32 * 34);
    B4Buf raw(bytes), out(static_cast<size_t>(k * n * 2));
    if (raw.p == nullptr || out.p == nullptr) return false;
    return go([&] {
      return kops::quant::gguf_dequantize(tensor2d(out.p, kops::DType::kFloat16, k, n),
                                          tensor1d(raw.p, kops::DType::kUInt8, static_cast<int64_t>(bytes)),
                                          q4k ? kops::GGUFType::kQ4_K : kops::GGUFType::kQ8_0, ctx);
    });
  }
  if (c.op == "fp4_to_marlin") {
    // n output channels x k reduction; weight [n, k/2] bytes, MXFP4 scales [n, k/32].
    B4Buf w(static_cast<size_t>(k * n / 2)), out(static_cast<size_t>(k * n / 2)),
        sc(static_cast<size_t>(n * (k / 32))), osc(static_cast<size_t>(n * (k / 32)));
    if (w.p == nullptr || out.p == nullptr || sc.p == nullptr || osc.p == nullptr) return false;
    return go([&] {
      return kops::quant::fp4_to_marlin(tensor2d(out.p, kops::DType::kInt32, k / 16, n * 2),
                                        tensor2d(osc.p, kops::DType::kFloat8E8M0, k / 32, n),
                                        tensor2d(w.p, kops::DType::kUInt8, n, k / 2),
                                        tensor2d(sc.p, kops::DType::kFloat8E8M0, n, k / 32), 32, true, ctx);
    });
  }
  if (c.op == "fp8_block_scale_to_ue8m0") {
    // In place: restore the (non power-of-two) scales before every call, otherwise from the
    // second call on every block is already UE8M0 and the kernel skips the payload.
    const int64_t sn = k / 128;
    const int64_t sk = n / 128;
    B4Buf w(static_cast<size_t>(k * n)), s(static_cast<size_t>(sn * sk * 4)),
        s0(static_cast<size_t>(sn * sk * 4)), e(static_cast<size_t>(sn * sk));
    if (w.p == nullptr || s.p == nullptr || s0.p == nullptr || e.p == nullptr) return false;
    std::vector<float> host(static_cast<size_t>(sn * sk), 0.0137f);
    cudaMemcpy(s0.p, host.data(), host.size() * 4, cudaMemcpyHostToDevice);
    const size_t sbytes = host.size() * 4;
    return go([&] {
      cudaMemcpyAsync(s.p, s0.p, sbytes, cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(ctx.stream));
      return kops::quant::fp8_block_scale_to_ue8m0(tensor2d(w.p, kops::DType::kFloat8E4M3, k, n),
                                                   tensor2d(s.p, kops::DType::kFloat32, sn, sk),
                                                   tensor2d(e.p, kops::DType::kFloat8E8M0, sn, sk), ctx);
    });
  }
  if (c.op == "smoothquant_fold") {
    // Smoothing factors of 1 keep the repeated in-place fold finite; the work is the same.
    B4Buf w(static_cast<size_t>(k * n * 2)), nw(static_cast<size_t>(n * 2)), s(static_cast<size_t>(n * 4));
    if (w.p == nullptr || nw.p == nullptr || s.p == nullptr) return false;
    std::vector<float> ones(static_cast<size_t>(n), 1.0f);
    cudaMemcpy(s.p, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice);
    return go([&] {
      return kops::quant::smoothquant_fold(tensor1d(nw.p, kops::DType::kBFloat16, n), kops::Tensor{},
                                           tensor2d(w.p, kops::DType::kBFloat16, k, n),
                                           tensor1d(s.p, kops::DType::kFloat32, n), ctx);
    });
  }
  return false;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (is_b4_op(c.op)) return run_b4(c, ctx, st);
  if (c.dtype == kops::DType::kFloat32 || c.op == "dequantize_per_token" ||
      c.op == "hadamard") {
    return run_typed<float>(c, ctx, st);
  }
  if (c.dtype == kops::DType::kBFloat16) return run_typed<__nv_bfloat16>(c, ctx, st);
  return false;
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
