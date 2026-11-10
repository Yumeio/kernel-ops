// =============================================================================
// kernel-ops | tests/unit/test_moe.cc
// B6 numerics and contract tests for M01 (routing), M02 (metadata, permute,
// unpermute, moe_sum) and the M03 latent tail. The oracle is an fp64 evaluation
// of the documented formulas; route_fused is additionally compared bit-for-bit
// with gemm::router_gemm + moe::route.
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/quantization/ops.h"
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

using kops::DType;
using kops::Status;
using kops::Tensor;

Tensor tensor1d(void* data, DType dtype, int64_t n) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

float e4m3_host(uint8_t b) {
  const uint32_t m = b & 0x7Fu;
  float v;
  if (m < 8u) {
    v = static_cast<float>(m) / 512.0f;
  } else {
    v = std::ldexp(1.0f + static_cast<float>(m & 7u) / 8.0f, static_cast<int>((m >> 3) & 0xFu) - 7);
  }
  return (b & 0x80u) ? -v : v;
}

void fill(std::vector<float>& v, unsigned seed, float lo = -3.0f, float hi = 3.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& x : v) x = dist(rng);
}

template <typename T>
std::vector<T> cast_from_float(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(out[i], src[i]);
  return out;
}

template <typename T>
std::vector<double> cast_to_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<double>(kops_test::to_float(src[i]));
  return out;
}

void expect_i32(const std::vector<int32_t>& got, const std::vector<int32_t>& want, const char* what) {
  if (got.size() != want.size()) {
    kops_test::record_failure(what, "size mismatch", __FILE__, __LINE__);
    return;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      char detail[160];
      std::snprintf(detail, sizeof(detail), "index %zu: got %d want %d", i, got[i], want[i]);
      kops_test::record_failure(what, detail, __FILE__, __LINE__);
      return;
    }
  }
}

struct RouteSpec {
  int experts = 8;
  int topk = 2;
  int groups = 1;
  int topk_groups = 1;
  int score = static_cast<int>(kops::moe::RouterScore::kSoftmax);
  int group_score = static_cast<int>(kops::moe::GroupScore::kMax);
  bool renormalize = true;
  float scale = 1.0f;
  float softcap = 0.0f;
  int stride_pad = 3;   // logits row padding in elements (0 = rows are contiguous)
  bool bias_nz = false; // fill correction_bias for the unbiased recipes too
};

void route_ref(const std::vector<double>& logits, int tokens, const RouteSpec& s,
               const std::vector<double>& bias, std::vector<int32_t>* ids,
               std::vector<double>* weights) {
  ids->assign(static_cast<size_t>(tokens * s.topk), 0);
  weights->assign(static_cast<size_t>(tokens * s.topk), 0.0);
  for (int t = 0; t < tokens; ++t) {
    std::vector<double> x(static_cast<size_t>(s.experts));
    for (int e = 0; e < s.experts; ++e) {
      double v = logits[static_cast<size_t>(t * s.experts + e)];
      if (s.softcap > 0.0) v = s.softcap * std::tanh(v / s.softcap);
      x[static_cast<size_t>(e)] = v;
    }
    std::vector<double> wv(x.size());
    if (s.score == static_cast<int>(kops::moe::RouterScore::kSoftmax)) {
      double m = -1e300;
      for (double v : x) m = std::max(m, v);
      double z = 0.0;
      for (int e = 0; e < s.experts; ++e) {
        wv[static_cast<size_t>(e)] = std::exp(x[static_cast<size_t>(e)] - m);
        z += wv[static_cast<size_t>(e)];
      }
      for (double& v : wv) v /= z;
    } else if (s.score == static_cast<int>(kops::moe::RouterScore::kSigmoid)) {
      for (int e = 0; e < s.experts; ++e) {
        wv[static_cast<size_t>(e)] = 1.0 / (1.0 + std::exp(-x[static_cast<size_t>(e)]));
      }
    } else if (s.score == static_cast<int>(kops::moe::RouterScore::kSqrtSoftplus)) {
      for (int e = 0; e < s.experts; ++e) {
        const double v = x[static_cast<size_t>(e)];
        const double sp = v > 20.0 ? v : std::log1p(std::exp(v));
        wv[static_cast<size_t>(e)] = std::sqrt(sp);
      }
    } else {
      wv = x;
    }
    std::vector<double> sel(wv.size());
    for (int e = 0; e < s.experts; ++e) {
      sel[static_cast<size_t>(e)] =
          wv[static_cast<size_t>(e)] + (bias.empty() ? 0.0 : bias[static_cast<size_t>(e)]);
    }
    uint32_t keep_groups = 0xFFFFFFFFu;
    if (s.groups > 1 && s.topk_groups < s.groups) {
      std::vector<double> gs(static_cast<size_t>(s.groups), -1e300);
      std::vector<double> g2(static_cast<size_t>(s.groups), -1e300);
      for (int e = 0; e < s.experts; ++e) {
        const int g = (e * s.groups) / s.experts;
        const double v = sel[static_cast<size_t>(e)];
        if (v > gs[static_cast<size_t>(g)]) {
          g2[static_cast<size_t>(g)] = gs[static_cast<size_t>(g)];
          gs[static_cast<size_t>(g)] = v;
        } else if (v > g2[static_cast<size_t>(g)]) {
          g2[static_cast<size_t>(g)] = v;
        }
      }
      std::vector<double> gscore(static_cast<size_t>(s.groups));
      for (int g = 0; g < s.groups; ++g) {
        gscore[static_cast<size_t>(g)] =
            s.group_score == static_cast<int>(kops::moe::GroupScore::kMax)
                ? gs[static_cast<size_t>(g)]
                : gs[static_cast<size_t>(g)] + g2[static_cast<size_t>(g)];
      }
      keep_groups = 0;
      for (int r = 0; r < s.topk_groups; ++r) {
        double best = -1e300;
        int bg = -1;
        for (int g = 0; g < s.groups; ++g) {
          if (keep_groups & (1u << g)) continue;
          if (gscore[static_cast<size_t>(g)] > best) {
            best = gscore[static_cast<size_t>(g)];
            bg = g;
          }
        }
        keep_groups |= (1u << bg);
      }
    }
    std::vector<double> chosen(static_cast<size_t>(s.topk), 0.0);
    for (int r = 0; r < s.topk; ++r) {
      double best = -1e300;
      int bi = -1;
      for (int e = 0; e < s.experts; ++e) {
        if (s.groups > 1 && (keep_groups & (1u << ((e * s.groups) / s.experts))) == 0u) {
          continue;
        }
        if (sel[static_cast<size_t>(e)] > best) {
          best = sel[static_cast<size_t>(e)];
          bi = e;
        }
      }
      chosen[static_cast<size_t>(r)] = wv[static_cast<size_t>(bi)];
      (*ids)[static_cast<size_t>(t * s.topk + r)] = bi;
      sel[static_cast<size_t>(bi)] = -1e300;
      wv[static_cast<size_t>(bi)] = -1e300;
    }
    if (s.score == static_cast<int>(kops::moe::RouterScore::kSoftmaxAfterTopK)) {
      double m = -1e300;
      for (double v : chosen) m = std::max(m, v);
      double z = 0.0;
      for (int r = 0; r < s.topk; ++r) {
        chosen[static_cast<size_t>(r)] = std::exp(chosen[static_cast<size_t>(r)] - m);
        z += chosen[static_cast<size_t>(r)];
      }
      for (double& v : chosen) v /= z;
    }
    if (s.renormalize) {
      double z = 0.0;
      for (double v : chosen) z += v;
      if (z > 0.0) {
        for (double& v : chosen) v /= z;
      }
    }
    for (int r = 0; r < s.topk; ++r) {
      (*weights)[static_cast<size_t>(t * s.topk + r)] =
          chosen[static_cast<size_t>(r)] * s.scale;
    }
  }
}

struct RouterBuffers {
  std::vector<float> logits;
  std::vector<int32_t> ids;
  std::vector<float> weights;
  kops::moe::RouterParams params{};
  float* d_logits = nullptr;
  int32_t* d_ids = nullptr;
  float* d_weights = nullptr;
  float* d_bias = nullptr;
  float* d_load = nullptr;
  int32_t* d_map = nullptr;
  int32_t* d_replicas = nullptr;
  std::vector<float> bias;
  std::vector<int32_t> map;
  std::vector<int32_t> replicas;
};

void run_route_case(const char* name, int tokens, const RouteSpec& s, unsigned seed) {
  RouterBuffers b;
  const int64_t stride = s.experts + s.stride_pad;
  b.logits.assign(static_cast<size_t>(tokens * stride), 0.0f);
  std::vector<float> dense(static_cast<size_t>(tokens * s.experts));
  fill(dense, seed);
  for (int t = 0; t < tokens; ++t) {
    for (int e = 0; e < s.experts; ++e) {
      b.logits[static_cast<size_t>(t * stride + e)] = dense[static_cast<size_t>(t * s.experts + e)];
    }
  }
  b.bias.assign(static_cast<size_t>(s.experts), 0.0f);
  if (s.bias_nz || (s.score != static_cast<int>(kops::moe::RouterScore::kSigmoid) &&
                    s.score != static_cast<int>(kops::moe::RouterScore::kSqrtSoftplus))) {
    fill(b.bias, seed + 1u, -0.5f, 0.5f);
  }
  b.ids.assign(static_cast<size_t>(tokens * s.topk), 0);
  b.weights.assign(static_cast<size_t>(tokens * s.topk), 0.0f);
  b.d_logits = kops_test::device_alloc<float>(b.logits.size());
  b.d_ids = kops_test::device_alloc<int32_t>(b.ids.size());
  b.d_weights = kops_test::device_alloc<float>(b.weights.size());
  b.d_bias = kops_test::device_alloc<float>(b.bias.size());
  kops_test::copy_in(b.d_logits, b.logits);
  kops_test::copy_in(b.d_bias, b.bias);

  b.params.logits = b.d_logits;
  b.params.logits_stride = stride;
  b.params.correction_bias = b.d_bias;
  b.params.num_tokens = tokens;
  b.params.num_experts = s.experts;
  b.params.topk = s.topk;
  b.params.num_expert_groups = s.groups;
  b.params.topk_groups = s.topk_groups;
  b.params.group_score = static_cast<kops::moe::GroupScore>(s.group_score);
  b.params.renormalize = s.renormalize;
  b.params.routed_scaling_factor = s.scale;
  b.params.logit_softcap = s.softcap;
  b.params.topk_weights = b.d_weights;
  b.params.topk_ids = b.d_ids;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::moe::route(b.params, static_cast<kops::moe::RouterScore>(s.score),
                                  DType::kFloat32, ctx),
                 name);
  b.ids = kops_test::copy_out(b.d_ids, b.ids.size());
  b.weights = kops_test::copy_out(b.d_weights, b.weights.size());

  std::vector<double> logits_d(static_cast<size_t>(tokens * s.experts));
  for (int t = 0; t < tokens; ++t) {
    for (int e = 0; e < s.experts; ++e) {
      logits_d[static_cast<size_t>(t * s.experts + e)] = b.logits[static_cast<size_t>(t * stride + e)];
    }
  }
  std::vector<double> bias_d(b.bias.begin(), b.bias.end());
  std::vector<int32_t> want_ids;
  std::vector<double> want_w;
  route_ref(logits_d, tokens, s, bias_d, &want_ids, &want_w);
  expect_i32(b.ids, want_ids, name);
  kops_test::compare(b.weights, want_w, 1e-5, 1e-5, name);

  kops_test::device_free(b.d_logits);
  kops_test::device_free(b.d_ids);
  kops_test::device_free(b.d_weights);
  kops_test::device_free(b.d_bias);
}

void test_route() {
  RouteSpec a;
  a.experts = 8;
  a.topk = 2;
  run_route_case("route softmax mixtral", 5, a, 3u);

  RouteSpec b;
  b.experts = 64;
  b.topk = 4;
  b.score = static_cast<int>(kops::moe::RouterScore::kSigmoid);
  b.scale = 2.5f;
  run_route_case("route sigmoid bias ds", 7, b, 5u);

  RouteSpec c;
  c.experts = 256;
  c.topk = 8;
  c.score = static_cast<int>(kops::moe::RouterScore::kSqrtSoftplus);
  c.softcap = 30.0f;
  run_route_case("route sqrtsoftplus", 3, c, 7u);

  RouteSpec d;
  d.experts = 16;
  d.topk = 4;
  d.score = static_cast<int>(kops::moe::RouterScore::kSoftmaxAfterTopK);
  d.renormalize = false;
  run_route_case("route gpt-oss after-topk", 4, d, 9u);

  RouteSpec e;
  e.experts = 32;
  e.topk = 2;
  e.score = static_cast<int>(kops::moe::RouterScore::kNone);
  e.renormalize = false;
  run_route_case("route raw logits", 2, e, 11u);
}

void test_route_grouped() {
  RouteSpec a;
  a.experts = 64;
  a.topk = 4;
  a.groups = 8;
  a.topk_groups = 2;
  run_route_case("route grouped softmax max", 6, a, 13u);

  RouteSpec b;
  b.experts = 256;
  b.topk = 8;
  b.groups = 8;
  b.topk_groups = 4;
  b.score = static_cast<int>(kops::moe::RouterScore::kSigmoid);
  b.group_score = static_cast<int>(kops::moe::GroupScore::kTop2Sum);
  b.scale = 2.5f;
  run_route_case("route grouped sigmoid top2", 5, b, 19u);
}

void test_route_ties() {
  const int tokens = 3;
  const int experts = 16;
  const int topk = 4;
  std::vector<float> logits(static_cast<size_t>(tokens * experts), 0.0f);
  float* d_logits = kops_test::device_alloc<float>(logits.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(tokens * topk));
  float* d_weights = kops_test::device_alloc<float>(static_cast<size_t>(tokens * topk));
  kops_test::copy_in(d_logits, logits);
  kops::moe::RouterParams p{};
  p.logits = d_logits;
  p.logits_stride = experts;
  p.num_tokens = tokens;
  p.num_experts = experts;
  p.topk = topk;
  p.num_expert_groups = 1;
  p.topk_groups = 1;
  p.renormalize = true;
  p.routed_scaling_factor = 1.0f;
  p.topk_ids = d_ids;
  p.topk_weights = d_weights;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::moe::route(p, kops::moe::RouterScore::kSoftmax, DType::kFloat32, ctx),
                 "route ties");
  const auto ids = kops_test::copy_out(d_ids, static_cast<size_t>(tokens * topk));
  const auto w = kops_test::copy_out(d_weights, static_cast<size_t>(tokens * topk));
  for (int t = 0; t < tokens; ++t) {
    for (int r = 0; r < topk; ++r) {
      KOPS_TEST_CHECK(ids[static_cast<size_t>(t * topk + r)] == r);
      KOPS_TEST_CHECK(std::fabs(w[static_cast<size_t>(t * topk + r)] - 0.25f) < 1e-6f);
    }
  }
  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_weights);
}

void test_route_eplb_shared() {
  const int tokens = 4;
  const int experts = 8;
  const int topk = 2;
  std::vector<float> logits(static_cast<size_t>(tokens * experts));
  fill(logits, 21u);
  float* d_logits = kops_test::device_alloc<float>(logits.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(tokens * (topk + 1)));
  float* d_weights = kops_test::device_alloc<float>(static_cast<size_t>(tokens * (topk + 1)));
  int32_t* d_load = kops_test::device_alloc<int32_t>(experts);
  int32_t* d_map = kops_test::device_alloc<int32_t>(experts * 2);
  int32_t* d_rep = kops_test::device_alloc<int32_t>(experts);
  std::vector<int32_t> map(static_cast<size_t>(experts * 2));
  std::vector<int32_t> rep(static_cast<size_t>(experts), 2);
  for (int e = 0; e < experts; ++e) {
    map[static_cast<size_t>(e * 2 + 0)] = 100 + e;
    map[static_cast<size_t>(e * 2 + 1)] = 200 + e;
  }
  kops_test::copy_in(d_logits, logits);
  kops_test::copy_in(d_map, map);
  kops_test::copy_in(d_rep, rep);
  cudaMemset(d_load, 0, static_cast<size_t>(experts) * sizeof(int32_t));
  kops::moe::RouterParams p{};
  p.logits = d_logits;
  p.logits_stride = experts;
  p.num_tokens = tokens;
  p.num_experts = experts;
  p.topk = topk;
  p.num_expert_groups = 1;
  p.topk_groups = 1;
  p.renormalize = true;
  p.routed_scaling_factor = 1.0f;
  p.topk_ids = d_ids;
  p.topk_weights = d_weights;
  p.logical_to_physical = d_map;
  p.replica_count = d_rep;
  p.max_replicas = 2;
  p.num_fused_shared = 1;
  p.shared_weight = 0.5f;
  p.expert_load = d_load;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::moe::route(p, kops::moe::RouterScore::kSoftmax, DType::kFloat32, ctx),
                 "route eplb");
  const auto ids = kops_test::copy_out(d_ids, static_cast<size_t>(tokens * (topk + 1)));
  const auto w = kops_test::copy_out(d_weights, static_cast<size_t>(tokens * (topk + 1)));
  const auto load = kops_test::copy_out(d_load, static_cast<size_t>(experts));
  int32_t total = 0;
  for (int e = 0; e < experts; ++e) total += load[static_cast<size_t>(e)];
  KOPS_TEST_CHECK(total == tokens * topk);
  for (int t = 0; t < tokens; ++t) {
    KOPS_TEST_CHECK(ids[static_cast<size_t>(t * (topk + 1) + topk)] == experts);
    KOPS_TEST_CHECK(std::fabs(w[static_cast<size_t>(t * (topk + 1) + topk)] - 0.5f) < 1e-6f);
    for (int r = 0; r < topk; ++r) {
      const int id = ids[static_cast<size_t>(t * (topk + 1) + r)];
      KOPS_TEST_CHECK(id >= 100 && id < 300);
    }
  }
  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_weights);
  kops_test::device_free(d_load);
  kops_test::device_free(d_map);
  kops_test::device_free(d_rep);
}

// =============================================================================
// E == 256 fp32 route fast path (two float4 loads per lane, k <= 8). The body is
// chosen from the tensor layout: rows that are 16 B aligned (stride % 4 == 0) take
// the vector body, every other shape keeps the scalar body, and both must match the
// fp64 oracle and each other bit for bit.
// =============================================================================

namespace e256 {

struct Run {
  std::vector<int32_t> ids;
  std::vector<float> weights;
  std::vector<int32_t> load;
};

/// route() on E = 256 with an explicit storage stride, optional EPLB mapping, fused
/// shared expert and expert-load histogram.
Run route(const std::vector<float>& dense, int tokens, int topk, int stride,
          kops::moe::RouterScore score, bool eplb, int n_shared) {
  const int experts = 256;
  const int out_k = topk + n_shared;
  Run run;
  run.ids.assign(static_cast<size_t>(tokens * out_k), -1);
  run.weights.assign(static_cast<size_t>(tokens * out_k), 0.0f);
  run.load.assign(static_cast<size_t>(experts), 0);
  std::vector<float> logits(static_cast<size_t>(tokens) * stride, 0.0f);
  for (int t = 0; t < tokens; ++t) {
    for (int e = 0; e < experts; ++e) {
      logits[static_cast<size_t>(t * stride + e)] = dense[static_cast<size_t>(t * experts + e)];
    }
  }
  std::vector<int32_t> map(static_cast<size_t>(experts) * 2);
  std::vector<int32_t> rep(static_cast<size_t>(experts), 2);
  for (int e = 0; e < experts; ++e) {
    map[static_cast<size_t>(e * 2)] = 100 + e;
    map[static_cast<size_t>(e * 2 + 1)] = 200 + e;
  }
  float* d_logits = kops_test::device_alloc<float>(logits.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(run.ids.size());
  float* d_w = kops_test::device_alloc<float>(run.weights.size());
  int32_t* d_load = kops_test::device_alloc<int32_t>(run.load.size());
  int32_t* d_map = kops_test::device_alloc<int32_t>(map.size());
  int32_t* d_rep = kops_test::device_alloc<int32_t>(rep.size());
  kops_test::copy_in(d_logits, logits);
  kops_test::copy_in(d_map, map);
  kops_test::copy_in(d_rep, rep);
  cudaMemset(d_load, 0, run.load.size() * sizeof(int32_t));
  kops::moe::RouterParams p{};
  p.logits = d_logits;
  p.logits_stride = stride;
  p.num_tokens = tokens;
  p.num_experts = experts;
  p.topk = topk;
  p.num_expert_groups = 1;
  p.topk_groups = 1;
  p.renormalize = true;
  p.routed_scaling_factor = 2.5f;
  p.topk_ids = d_ids;
  p.topk_weights = d_w;
  p.expert_load = d_load;
  if (eplb) {
    p.logical_to_physical = d_map;
    p.replica_count = d_rep;
    p.max_replicas = 2;
  }
  p.num_fused_shared = n_shared;
  p.shared_weight = 0.5f;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return run;
  KOPS_EXPECT_OK(kops::moe::route(p, score, DType::kFloat32, ctx), "e256 route");
  run.ids = kops_test::copy_out(d_ids, run.ids.size());
  run.weights = kops_test::copy_out(d_w, run.weights.size());
  run.load = kops_test::copy_out(d_load, run.load.size());
  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_w);
  kops_test::device_free(d_load);
  kops_test::device_free(d_map);
  kops_test::device_free(d_rep);
  return run;
}

}  // namespace e256

// Vector body against the fp64 oracle: contiguous rows (the benchmark layout),
// padded-but-16 B-aligned rows, grouped routing, softcap, bias, after-top-k softmax
// and short k values. Also the shape tails that stay on the scalar body (E = 200/192
// are not multiples of the 32-expert lane stride, E = 8 is far below the vector lane
// count, and an odd stride removes the alignment the vector loads require).
void test_route_e256_fast() {
  using kops::moe::GroupScore;
  using kops::moe::RouterScore;

  RouteSpec a;  // benchmark shape
  a.experts = 256;
  a.topk = 8;
  a.stride_pad = 0;
  a.score = static_cast<int>(RouterScore::kSigmoid);
  a.bias_nz = true;
  a.scale = 2.5f;
  run_route_case("route e256 sigmoid k8", 33, a, 71u);

  RouteSpec b;  // padded rows that stay 16 B aligned
  b.experts = 256;
  b.topk = 8;
  b.stride_pad = 4;
  b.score = static_cast<int>(RouterScore::kSqrtSoftplus);
  b.softcap = 30.0f;
  run_route_case("route e256 sqrtsoftplus softcap", 17, b, 73u);

  RouteSpec c;  // odd stride: the scalar body
  c.experts = 256;
  c.topk = 8;
  c.stride_pad = 1;
  c.score = static_cast<int>(RouterScore::kSigmoid);
  c.bias_nz = true;
  run_route_case("route e256 sigmoid unaligned", 11, c, 75u);

  RouteSpec d;  // short rank loops with the vector body
  d.experts = 256;
  d.topk = 1;
  d.stride_pad = 0;
  d.score = static_cast<int>(RouterScore::kSigmoid);
  run_route_case("route e256 sigmoid k1", 9, d, 77u);
  d.topk = 2;
  run_route_case("route e256 sigmoid k2", 9, d, 79u);
  d.topk = 8;
  d.stride_pad = 8;
  d.renormalize = false;
  d.score = static_cast<int>(RouterScore::kNone);
  run_route_case("route e256 raw logits k8", 9, d, 80u);

  RouteSpec e;  // after-top-k softmax keeps its exact accumulation order
  e.experts = 256;
  e.topk = 6;
  e.stride_pad = 8;
  e.score = static_cast<int>(RouterScore::kSoftmaxAfterTopK);
  e.renormalize = false;
  run_route_case("route e256 after-topk k6", 13, e, 81u);

  RouteSpec f;  // grouped routing on the vector body
  f.experts = 256;
  f.topk = 8;
  f.groups = 8;
  f.topk_groups = 4;
  f.stride_pad = 0;
  f.score = static_cast<int>(RouterScore::kSigmoid);
  f.group_score = static_cast<int>(GroupScore::kTop2Sum);
  f.scale = 2.5f;
  run_route_case("route e256 grouped sigmoid", 15, f, 83u);
  f.topk_groups = 2;
  f.group_score = static_cast<int>(GroupScore::kMax);
  f.topk = 4;
  f.stride_pad = 4;
  run_route_case("route e256 grouped top2 max", 15, f, 84u);

  RouteSpec g;  // expert-count tails: neither shape is covered by the vector body
  g.experts = 200;
  g.topk = 5;
  g.stride_pad = 0;
  g.score = static_cast<int>(RouterScore::kSoftmax);
  run_route_case("route e200 softmax tail", 7, g, 85u);
  RouteSpec h;
  h.experts = 192;
  h.topk = 4;
  h.stride_pad = 0;
  h.score = static_cast<int>(RouterScore::kSigmoid);
  h.scale = 2.5f;
  run_route_case("route e192 sigmoid k4", 7, h, 87u);
  RouteSpec i;
  i.experts = 8;
  i.topk = 2;
  i.stride_pad = 0;
  run_route_case("route e8 softmax contiguous", 5, i, 89u);

  RouteSpec j;  // softmax over all E keeps its warp reduction order (scalar body)
  j.experts = 256;
  j.topk = 8;
  j.stride_pad = 0;
  j.score = static_cast<int>(RouterScore::kSoftmax);
  j.scale = 2.5f;
  run_route_case("route e256 softmax k8", 19, j, 91u);
  j.stride_pad = 4;
  run_route_case("route e256 softmax padded", 19, j, 92u);

  RouteSpec k;  // k > 8 stays on the scalar body
  k.experts = 256;
  k.topk = 16;
  k.stride_pad = 0;
  k.score = static_cast<int>(RouterScore::kSigmoid);
  k.bias_nz = true;
  k.scale = 2.5f;
  run_route_case("route e256 sigmoid k16", 13, k, 93u);
}

// Stable tie-break inside the vector body: equal scores must order by lower expert
// id, both between candidates of one lane (0..3 / 128..131) and across lanes.
void test_route_e256_ties() {
  const int tokens = 3;
  const int experts = 256;
  const int topk = 8;
  std::vector<float> logits(static_cast<size_t>(tokens * experts), 0.0f);
  logits[1 * experts + 3] = 5.0f;    // same lane, same score
  logits[1 * experts + 200] = 5.0f;  // other lane
  logits[2 * experts + 0] = 5.0f;    // first and last expert of the row
  logits[2 * experts + 255] = 5.0f;
  float* d_logits = kops_test::device_alloc<float>(logits.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(tokens * topk));
  float* d_w = kops_test::device_alloc<float>(static_cast<size_t>(tokens * topk));
  kops_test::copy_in(d_logits, logits);
  kops::moe::RouterParams p{};
  p.logits = d_logits;
  p.logits_stride = experts;  // contiguous rows -> vector body
  p.num_tokens = tokens;
  p.num_experts = experts;
  p.topk = topk;
  p.num_expert_groups = 1;
  p.topk_groups = 1;
  p.renormalize = false;
  p.routed_scaling_factor = 1.0f;
  p.topk_ids = d_ids;
  p.topk_weights = d_w;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::moe::route(p, kops::moe::RouterScore::kSigmoid, DType::kFloat32, ctx),
                 "route e256 ties");
  const auto ids = kops_test::copy_out(d_ids, static_cast<size_t>(tokens * topk));
  const auto w = kops_test::copy_out(d_w, static_cast<size_t>(tokens * topk));
  const int32_t want[3][topk] = {{0, 1, 2, 3, 4, 5, 6, 7},
                                 {3, 200, 0, 1, 2, 4, 5, 6},
                                 {0, 255, 1, 2, 3, 4, 5, 6}};
  int bad_id = 0, bad_w = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int r = 0; r < topk; ++r) {
      const size_t i = static_cast<size_t>(t * topk + r);
      bad_id += ids[i] != want[t][r];
      // weights are the unbiased score of the selected expert
      const double x = static_cast<double>(logits[static_cast<size_t>(t * experts + ids[i])]);
      bad_w += std::fabs(static_cast<double>(w[i]) - 1.0 / (1.0 + std::exp(-x))) > 1e-6;
    }
  }
  KOPS_TEST_CHECK(bad_id == 0);
  KOPS_TEST_CHECK(bad_w == 0);
  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_w);
}

// The vector body and the scalar body must be interchangeable: same ids, bit-equal
// weights and same expert-load histogram, with and without EPLB mapping and the
// fused shared expert.
void test_route_e256_paths() {
  const int tokens = 21;
  const int topk = 8;
  const int experts = 256;
  std::vector<float> dense(static_cast<size_t>(tokens) * experts);
  fill(dense, 95u);
  for (int eplb = 0; eplb < 2; ++eplb) {
    for (int shared = 0; shared < 2; ++shared) {
      const int out_k = topk + shared;
      const auto fast = e256::route(dense, tokens, topk, experts, kops::moe::RouterScore::kSigmoid,
                                    eplb != 0, shared);
      // 259 elements per row: the vector loads would be misaligned, so this run takes
      // the scalar body on the very same logical data.
      const auto scalar = e256::route(dense, tokens, topk, experts + 3,
                                      kops::moe::RouterScore::kSigmoid, eplb != 0, shared);
      expect_i32(fast.ids, scalar.ids, "e256 vector ids == scalar ids");
      int bad = 0;
      for (size_t i = 0; i < fast.weights.size(); ++i) bad += fast.weights[i] != scalar.weights[i];
      for (size_t i = 0; i < fast.load.size(); ++i) bad += fast.load[i] != scalar.load[i];
      KOPS_TEST_CHECK(bad == 0);
      int bad_range = 0, bad_shared = 0, total = 0;
      for (int e = 0; e < experts; ++e) total += fast.load[static_cast<size_t>(e)];
      for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < topk; ++r) {
          const int id = fast.ids[static_cast<size_t>(t * out_k + r)];
          // EPLB maps to 100 + e or 200 + e, so every routed id is in [100, 456).
          bad_range += eplb != 0 && (id < 100 || id >= 200 + experts);
        }
        if (shared != 0) {
          bad_shared += fast.ids[static_cast<size_t>(t * out_k + topk)] != experts;
          bad_shared += std::fabs(fast.weights[static_cast<size_t>(t * out_k + topk)] - 0.5f) > 1e-6f;
        }
      }
      KOPS_TEST_CHECK(total == tokens * topk);
      KOPS_TEST_CHECK(bad_range == 0);
      KOPS_TEST_CHECK(bad_shared == 0);
    }
  }

  // Signed zeros compare equal in the fp32 score, so the raw-logit recipe must still
  // order the tied experts by id (a key that separated -0.0 from +0.0 would not).
  std::vector<float> zeros(static_cast<size_t>(tokens) * experts, 0.0f);
  for (int t = 0; t < tokens; ++t) {
    zeros[static_cast<size_t>(t * experts + 5)] = -0.0f;
    zeros[static_cast<size_t>(t * experts + 7)] = 0.0f;
    zeros[static_cast<size_t>(t * experts + 200)] = -0.0f;
  }
  const auto zf = e256::route(zeros, tokens, topk, experts, kops::moe::RouterScore::kNone, false, 0);
  const auto zs =
      e256::route(zeros, tokens, topk, experts + 3, kops::moe::RouterScore::kNone, false, 0);
  expect_i32(zf.ids, zs.ids, "e256 signed-zero ids");
  int zbad = 0;
  for (size_t i = 0; i < zf.weights.size(); ++i) zbad += zf.weights[i] != zs.weights[i];
  KOPS_TEST_CHECK(zbad == 0);
  int zorder = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int r = 0; r < topk; ++r) zorder += zf.ids[static_cast<size_t>(t * topk + r)] != r;
  }
  KOPS_TEST_CHECK(zorder == 0);
}

void test_route_fused() {
  for (int variant = 0; variant < 2; ++variant) {
    const int tokens = variant == 0 ? 5 : 3;
    const int experts = variant == 0 ? 8 : 64;
    const int hidden = variant == 0 ? 64 : 128;
    const int topk = variant == 0 ? 2 : 4;
    std::vector<float> x_f(static_cast<size_t>(tokens * hidden));
    std::vector<float> w_f(static_cast<size_t>(experts * hidden));
    std::vector<float> bias_f(static_cast<size_t>(experts));
    fill(x_f, 31u, -1.0f, 1.0f);
    fill(w_f, 33u, -0.5f, 0.5f);
    fill(bias_f, 35u, -0.2f, 0.2f);
    const auto x_bf = cast_from_float<__nv_bfloat16>(x_f);
    const auto w_bf = cast_from_float<__nv_bfloat16>(w_f);
    __nv_bfloat16* d_x = kops_test::device_alloc<__nv_bfloat16>(x_bf.size());
    __nv_bfloat16* d_w = kops_test::device_alloc<__nv_bfloat16>(w_bf.size());
    float* d_bias = kops_test::device_alloc<float>(bias_f.size());
    float* d_logits = kops_test::device_alloc<float>(static_cast<size_t>(tokens * experts));
    int32_t* d_ids_a = kops_test::device_alloc<int32_t>(static_cast<size_t>(tokens * topk));
    float* d_w_a = kops_test::device_alloc<float>(static_cast<size_t>(tokens * topk));
    int32_t* d_ids_b = kops_test::device_alloc<int32_t>(static_cast<size_t>(tokens * topk));
    float* d_w_b = kops_test::device_alloc<float>(static_cast<size_t>(tokens * topk));
    kops_test::copy_in(d_x, x_bf);
    kops_test::copy_in(d_w, w_bf);
    kops_test::copy_in(d_bias, bias_f);

    const Tensor t_x = kops_test::tensor2d(d_x, DType::kBFloat16, tokens, hidden);
    const Tensor t_w = kops_test::tensor2d(d_w, DType::kBFloat16, experts, hidden);
    const Tensor t_bias = tensor1d(d_bias, DType::kFloat32, experts);
    kops::moe::RouterParams p{};
    p.logits = d_logits;
    p.logits_stride = experts;
    p.num_tokens = tokens;
    p.num_experts = experts;
    p.topk = topk;
    p.num_expert_groups = 1;
    p.topk_groups = 1;
    p.renormalize = true;
    p.routed_scaling_factor = 1.5f;
    p.correction_bias = d_bias;
    p.topk_ids = d_ids_a;
    p.topk_weights = d_w_a;
    kops::LaunchContext ctx{};
    if (!kops_test::make_context(&ctx)) return;
    KOPS_EXPECT_OK(kops::moe::route_fused(p, t_x, t_w, t_bias, kops::moe::RouterScore::kSigmoid,
                                          DType::kBFloat16, ctx),
                   "route_fused");

    kops::gemm::RouterGemmParams gp{};
    gp.x = d_x;
    gp.ldx = hidden;
    gp.w = d_w;
    gp.bias = d_bias;
    gp.logits = d_logits;
    gp.ld_logits = experts;
    gp.m = tokens;
    gp.num_experts = experts;
    gp.hidden = hidden;
    gp.lo_scale = 1.0f / 256.0f;
    KOPS_EXPECT_OK(kops::gemm::router_gemm(gp, false, DType::kBFloat16, DType::kBFloat16, ctx),
                   "router_gemm");
    p.logits = d_logits;
    p.logits_stride = experts;
    p.topk_ids = d_ids_b;
    p.topk_weights = d_w_b;
    KOPS_EXPECT_OK(kops::moe::route(p, kops::moe::RouterScore::kSigmoid, DType::kFloat32, ctx),
                   "route reference");
    const auto ids_a = kops_test::copy_out(d_ids_a, static_cast<size_t>(tokens * topk));
    const auto ids_b = kops_test::copy_out(d_ids_b, static_cast<size_t>(tokens * topk));
    const auto w_a = kops_test::copy_out(d_w_a, static_cast<size_t>(tokens * topk));
    const auto w_b = kops_test::copy_out(d_w_b, static_cast<size_t>(tokens * topk));
    expect_i32(ids_a, ids_b, "route_fused ids bit-equal");
    for (size_t i = 0; i < w_a.size(); ++i) KOPS_TEST_CHECK(w_a[i] == w_b[i]);

    // The no-scratch fallback (single-CTA fused body) must agree bit-for-bit too.
    p.logits = nullptr;
    p.logits_stride = 0;
    p.topk_ids = d_ids_b;
    p.topk_weights = d_w_b;
    KOPS_EXPECT_OK(kops::moe::route_fused(p, t_x, t_w, t_bias, kops::moe::RouterScore::kSigmoid,
                                          DType::kBFloat16, ctx),
                   "route_fused fallback");
    const auto ids_fb = kops_test::copy_out(d_ids_b, static_cast<size_t>(tokens * topk));
    const auto w_fb = kops_test::copy_out(d_w_b, static_cast<size_t>(tokens * topk));
    expect_i32(ids_fb, ids_a, "route_fused fallback ids");
    for (size_t i = 0; i < w_fb.size(); ++i) KOPS_TEST_CHECK(w_fb[i] == w_a[i]);

    kops_test::device_free(d_x);
    kops_test::device_free(d_w);
    kops_test::device_free(d_bias);
    kops_test::device_free(d_logits);
    kops_test::device_free(d_ids_a);
    kops_test::device_free(d_w_a);
    kops_test::device_free(d_ids_b);
    kops_test::device_free(d_w_b);
  }
}

void test_hash_route() {
  const int tokens = 5;
  const int experts = 16;
  const int topk = 2;
  const int64_t vocab = 32;
  std::vector<int64_t> tok = {0, 3, 7, 31, 12};
  std::vector<int32_t> table(static_cast<size_t>(vocab * topk));
  for (int64_t v = 0; v < vocab; ++v) {
    table[static_cast<size_t>(v * topk + 0)] = static_cast<int32_t>(v % experts);
    table[static_cast<size_t>(v * topk + 1)] = static_cast<int32_t>((v + 5) % experts);
  }
  int64_t* d_tok = kops_test::device_alloc<int64_t>(tok.size());
  int32_t* d_table = kops_test::device_alloc<int32_t>(table.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(tokens * topk));
  float* d_w = kops_test::device_alloc<float>(static_cast<size_t>(tokens * topk));
  kops_test::copy_in(d_tok, tok);
  kops_test::copy_in(d_table, table);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const Tensor t_w = kops_test::tensor2d(d_w, DType::kFloat32, tokens, topk);
  const Tensor t_ids = kops_test::tensor2d(d_ids, DType::kInt32, tokens, topk);
  const Tensor t_tok = tensor1d(d_tok, DType::kInt64, tokens);
  const Tensor t_table = kops_test::tensor2d(d_table, DType::kInt32, vocab, topk);
  KOPS_EXPECT_OK(kops::moe::hash_route(t_w, t_ids, t_tok, t_table, Tensor{}, 2.0f, experts, ctx),
                 "hash_route");
  const auto ids = kops_test::copy_out(d_ids, static_cast<size_t>(tokens * topk));
  const auto weights = kops_test::copy_out(d_w, static_cast<size_t>(tokens * topk));
  for (int t = 0; t < tokens; ++t) {
    for (int r = 0; r < topk; ++r) {
      KOPS_TEST_CHECK(ids[static_cast<size_t>(t * topk + r)] ==
                      table[static_cast<size_t>(tok[static_cast<size_t>(t)] * topk + r)]);
      KOPS_TEST_CHECK(std::fabs(weights[static_cast<size_t>(t * topk + r)] - 1.0f) < 1e-6f);
    }
  }
  kops_test::device_free(d_tok);
  kops_test::device_free(d_table);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_w);
}

void test_align_and_offsets() {
  const int tokens = 9;
  const int topk = 2;
  const int experts = 6;
  const int block = 4;
  std::vector<int32_t> slots = {0, 1, 0, 2, 3, 0, 1, 5, 2, 4, 0, 1, 3, 3, 4, 5, 4, 5};
  std::vector<int32_t> map = {0, 1, 2, -1, 3, 4};
  const int64_t numel = tokens * topk;
  const int32_t sentinel = static_cast<int32_t>(numel);
  int32_t* d_ids = kops_test::device_alloc<int32_t>(numel);
  int32_t* d_map = kops_test::device_alloc<int32_t>(experts);
  int32_t* d_sorted = kops_test::device_alloc<int32_t>(numel + experts * block);
  int32_t* d_expert_ids = kops_test::device_alloc<int32_t>(numel + experts * block);
  int32_t* d_post = kops_test::device_alloc<int32_t>(1);
  int32_t* d_counts = kops_test::device_alloc<int32_t>(5);
  int64_t* d_offsets = kops_test::device_alloc<int64_t>(experts + 1);
  kops_test::copy_in(d_ids, slots);
  kops_test::copy_in(d_map, map);
  const size_t capacity = static_cast<size_t>(numel + experts * block);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, numel);
  const Tensor t_map = tensor1d(d_map, DType::kInt32, experts);
  const Tensor t_sorted = tensor1d(d_sorted, DType::kInt32, capacity);
  const Tensor t_expert_ids = tensor1d(d_expert_ids, DType::kInt32, capacity);
  const Tensor t_post = tensor1d(d_post, DType::kInt32, 1);
  const Tensor t_counts = tensor1d(d_counts, DType::kInt32, 5);
  const Tensor t_offsets = tensor1d(d_offsets, DType::kInt64, 6);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  int32_t* scratch = kops_test::device_alloc<int32_t>(experts + 1);
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = static_cast<size_t>(experts + 1) * sizeof(int32_t);
  KOPS_EXPECT_OK(kops::moe::align_block_size(t_ids, t_sorted, t_expert_ids, t_post, t_map, experts,
                                             block, ws, ctx),
                 "align_block_size");
  KOPS_EXPECT_OK(kops::moe::count_and_offsets(t_ids, t_counts, t_offsets, t_map, 5, 1, ctx),
                 "count_and_offsets");
  const auto got_sorted = kops_test::copy_out(d_sorted, capacity);
  const auto got_expert_ids = kops_test::copy_out(d_expert_ids, capacity);
  const auto got_post = kops_test::copy_out(d_post, static_cast<size_t>(1));
  const auto got_counts = kops_test::copy_out(d_counts, static_cast<size_t>(5));
  const auto got_offsets = kops_test::copy_out(d_offsets, static_cast<size_t>(6));

  std::vector<int32_t> want_sorted;
  std::vector<int32_t> want_blocks;
  std::vector<int32_t> counts(static_cast<size_t>(experts), 0);
  std::vector<int32_t> local_counts(5, 0);
  std::vector<std::vector<int32_t>> per_expert(static_cast<size_t>(experts));
  for (int64_t i = 0; i < numel; ++i) {
    const int32_t e = slots[static_cast<size_t>(i)];
    const int32_t l = map[static_cast<size_t>(e)];
    if (l < 0) continue;
    counts[static_cast<size_t>(e)] += 1;
    local_counts[static_cast<size_t>(l)] += 1;
    per_expert[static_cast<size_t>(e)].push_back(static_cast<int32_t>(i));
  }
  for (int e = 0; e < experts; ++e) {
    if (map[static_cast<size_t>(e)] < 0) continue;
    const int blocks = (counts[static_cast<size_t>(e)] + block - 1) / block;
    for (int b = 0; b < blocks; ++b) want_blocks.push_back(map[static_cast<size_t>(e)]);
    for (int i = 0; i < blocks * block; ++i) {
      const size_t idx = static_cast<size_t>(i);
      want_sorted.push_back(idx < per_expert[static_cast<size_t>(e)].size()
                                ? per_expert[static_cast<size_t>(e)][idx]
                                : sentinel);
    }
  }
  KOPS_TEST_CHECK(got_post[0] == static_cast<int32_t>(want_sorted.size()));
  for (size_t i = 0; i < want_sorted.size(); ++i) {
    KOPS_TEST_CHECK(got_sorted[i] == want_sorted[i]);
  }
  for (size_t i = 0; i < want_blocks.size(); ++i) {
    KOPS_TEST_CHECK(got_expert_ids[i] == want_blocks[i]);
  }
  for (int l = 0; l < 5; ++l) {
    KOPS_TEST_CHECK(got_counts[static_cast<size_t>(l)] == local_counts[static_cast<size_t>(l)]);
  }
  int64_t run = 0;
  for (int l = 0; l < 5; ++l) {
    KOPS_TEST_CHECK(got_offsets[static_cast<size_t>(l)] == run);
    run += local_counts[static_cast<size_t>(l)];
  }
  KOPS_TEST_CHECK(got_offsets[5] == run);

  kops_test::device_free(d_ids);
  kops_test::device_free(d_map);
  kops_test::device_free(d_sorted);
  kops_test::device_free(d_expert_ids);
  kops_test::device_free(d_post);
  kops_test::device_free(d_counts);
  kops_test::device_free(d_offsets);
  kops_test::device_free(scratch);
}

// Large randomized sorts: many slots per warp segment, several segments, experts not on this rank.
// Reference is the plain stable counting sort the kernels must reproduce bit for bit.
void check_large_sort(int experts, int block, bool with_map, uint32_t seed) {
  const int tokens = 5000, topk = 8, hidden = 8;
  const int64_t numel = static_cast<int64_t>(tokens) * topk;
  std::mt19937 rng(seed);
  std::vector<int32_t> slots(static_cast<size_t>(numel));
  for (auto& v : slots) v = static_cast<int32_t>(rng() % static_cast<uint32_t>(experts));
  std::vector<int32_t> map(static_cast<size_t>(experts));
  int local = 0;
  for (int e = 0; e < experts; ++e) map[static_cast<size_t>(e)] = (with_map && e % 4 == 3) ? -1 : local++;
  const int sentinel = static_cast<int>(numel);
  const size_t capacity = static_cast<size_t>(numel + static_cast<int64_t>(experts) * block);
  int32_t* d_ids = kops_test::device_alloc<int32_t>(numel);
  int32_t* d_map = kops_test::device_alloc<int32_t>(experts);
  int32_t* d_sorted = kops_test::device_alloc<int32_t>(capacity);
  int32_t* d_expert_ids = kops_test::device_alloc<int32_t>(capacity);
  int32_t* d_post = kops_test::device_alloc<int32_t>(1);
  int32_t* d_counts = kops_test::device_alloc<int32_t>(local);
  int64_t* d_offsets = kops_test::device_alloc<int64_t>(local + 1);
  int32_t* scratch = kops_test::device_alloc<int32_t>(experts + 1);
  kops_test::copy_in(d_ids, slots);
  kops_test::copy_in(d_map, map);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = static_cast<size_t>(experts + 1) * sizeof(int32_t);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, numel);
  const Tensor t_map = tensor1d(d_map, DType::kInt32, experts);
  KOPS_EXPECT_OK(kops::moe::align_block_size(t_ids, tensor1d(d_sorted, DType::kInt32, capacity),
                                             tensor1d(d_expert_ids, DType::kInt32, capacity),
                                             tensor1d(d_post, DType::kInt32, 1), t_map, experts, block,
                                             ws, ctx),
                 "large align_block_size");
  std::vector<std::vector<int32_t>> per(static_cast<size_t>(experts));
  for (int64_t i = 0; i < numel; ++i) {
    const int e = slots[static_cast<size_t>(i)];
    if (map[static_cast<size_t>(e)] >= 0) per[static_cast<size_t>(e)].push_back(static_cast<int32_t>(i));
  }
  std::vector<int32_t> want_sorted, want_blocks;
  for (int e = 0; e < experts; ++e) {
    if (map[static_cast<size_t>(e)] < 0) continue;
    const int n = static_cast<int>(per[static_cast<size_t>(e)].size());
    const int blocks = (n + block - 1) / block;
    for (int b = 0; b < blocks; ++b) want_blocks.push_back(map[static_cast<size_t>(e)]);
    for (int i = 0; i < blocks * block; ++i) {
      want_sorted.push_back(i < n ? per[static_cast<size_t>(e)][static_cast<size_t>(i)] : sentinel);
    }
  }
  const auto got_sorted = kops_test::copy_out(d_sorted, capacity);
  const auto got_blocks = kops_test::copy_out(d_expert_ids, capacity);
  const auto got_post = kops_test::copy_out(d_post, static_cast<size_t>(1));
  KOPS_TEST_CHECK(got_post[0] == static_cast<int32_t>(want_sorted.size()));
  int bad = 0;
  for (size_t i = 0; i < want_sorted.size(); ++i) bad += got_sorted[i] != want_sorted[i];
  for (size_t i = 0; i < want_blocks.size(); ++i) bad += got_blocks[i] != want_blocks[i];
  KOPS_TEST_CHECK(bad == 0);

  // Permute map on the same slots: row = expert_offsets[l] + rank of the slot inside l.
  KOPS_EXPECT_OK(kops::moe::count_and_offsets(t_ids, tensor1d(d_counts, DType::kInt32, local),
                                              tensor1d(d_offsets, DType::kInt64, local + 1), t_map,
                                              local, 1, ctx),
                 "large count_and_offsets");
  const auto offsets = kops_test::copy_out(d_offsets, static_cast<size_t>(local + 1));
  const int rows = static_cast<int>(offsets[static_cast<size_t>(local)]);
  float* d_hidden = kops_test::device_alloc<float>(static_cast<size_t>(tokens) * hidden);
  float* d_perm = kops_test::device_alloc<float>(static_cast<size_t>(rows) * hidden);
  int32_t* d_slot_row = kops_test::device_alloc<int32_t>(numel);
  int32_t* d_row_slot = kops_test::device_alloc<int32_t>(rows);
  kops::moe::PermuteParams pp{};
  pp.hidden = d_hidden;
  pp.hidden_stride = hidden;
  pp.topk_ids = d_ids;
  pp.expert_offsets = d_offsets;
  pp.expert_map = d_map;
  pp.num_local_experts = local;
  pp.permuted = d_perm;
  pp.permuted_stride = hidden;
  pp.permuted_row_to_slot = d_row_slot;
  pp.slot_to_permuted_row = d_slot_row;
  pp.num_tokens = tokens;
  pp.topk = topk;
  pp.hidden_size = hidden;
  KOPS_EXPECT_OK(kops::moe::permute(pp, DType::kFloat32, ctx), "large permute");
  const auto slot_row = kops_test::copy_out(d_slot_row, static_cast<size_t>(numel));
  const auto row_slot = kops_test::copy_out(d_row_slot, static_cast<size_t>(rows));
  std::vector<int64_t> next(offsets.begin(), offsets.end() - 1);
  int bad_map = 0;
  for (int64_t i = 0; i < numel; ++i) {
    const int l = map[static_cast<size_t>(slots[static_cast<size_t>(i)])];
    if (l < 0) {
      bad_map += slot_row[static_cast<size_t>(i)] != -1;
      continue;
    }
    const int64_t row = next[static_cast<size_t>(l)]++;
    bad_map += slot_row[static_cast<size_t>(i)] != row;
    bad_map += row_slot[static_cast<size_t>(row)] != static_cast<int32_t>(i);
  }
  KOPS_TEST_CHECK(bad_map == 0);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_map);
  kops_test::device_free(d_sorted);
  kops_test::device_free(d_expert_ids);
  kops_test::device_free(d_post);
  kops_test::device_free(d_counts);
  kops_test::device_free(d_offsets);
  kops_test::device_free(scratch);
  kops_test::device_free(d_hidden);
  kops_test::device_free(d_perm);
  kops_test::device_free(d_slot_row);
  kops_test::device_free(d_row_slot);
}

void test_large_sorts() {
  check_large_sort(256, 64, false, 11u);
  check_large_sort(256, 128, true, 12u);
  check_large_sort(1000, 16, true, 13u);  // 1000 bins leave only 11 warps of shared memory
  check_large_sort(7, 4, false, 14u);
}

void test_moe_sum_bf16() {
  const int tokens = 5, topk = 8, hidden = 1032;  // 129 vectors: the last block is partly idle
  std::vector<__nv_bfloat16> in(static_cast<size_t>(tokens) * topk * hidden);
  std::mt19937 rng(77u);
  for (auto& v : in) kops_test::from_float(v, static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 256.0f);
  __nv_bfloat16* d_in = kops_test::device_alloc<__nv_bfloat16>(in.size());
  __nv_bfloat16* d_out = kops_test::device_alloc<__nv_bfloat16>(static_cast<size_t>(tokens) * hidden);
  kops_test::copy_in(d_in, in);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::moe::moe_sum(kops_test::tensor2d(d_out, DType::kBFloat16, tokens, hidden),
                                    kops_test::tensor3d(d_in, DType::kBFloat16, tokens, topk, hidden),
                                    topk, ctx),
                 "moe_sum bf16");
  const auto out = kops_test::copy_out(d_out, static_cast<size_t>(tokens) * hidden);
  int bad = 0;
  for (int t = 0; t < tokens; ++t) {
    for (int c = 0; c < hidden; ++c) {
      float acc = 0.0f;  // same order and precision as the kernel: fp32 in ascending k
      for (int k = 0; k < topk; ++k) acc += kops_test::to_float(in[static_cast<size_t>((t * topk + k) * hidden + c)]);
      __nv_bfloat16 want;
      kops_test::from_float(want, acc);
      bad += out[static_cast<size_t>(t * hidden + c)] != want;
    }
  }
  KOPS_TEST_CHECK(bad == 0);
  kops_test::device_free(d_in);
  kops_test::device_free(d_out);
}

void test_permute_unpermute() {
  const int tokens = 7;
  const int topk = 2;
  const int experts = 8;
  const int hidden = 32;
  std::vector<int32_t> slots = {0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 3};
  std::vector<float> hidden_f(static_cast<size_t>(tokens * hidden));
  fill(hidden_f, 41u, -1.0f, 1.0f);
  std::vector<float> weights(static_cast<size_t>(tokens * topk), 1.0f);
  int32_t* d_ids = kops_test::device_alloc<int32_t>(slots.size());
  int32_t* d_map = kops_test::device_alloc<int32_t>(experts);
  int64_t* d_offsets = kops_test::device_alloc<int64_t>(experts + 1);
  int32_t* d_counts = kops_test::device_alloc<int32_t>(experts);
  int32_t* d_cursor = kops_test::device_alloc<int32_t>(experts);
  int32_t* d_slot_row = kops_test::device_alloc<int32_t>(slots.size());
  int32_t* d_row_slot = kops_test::device_alloc<int32_t>(slots.size());
  float* d_hidden = kops_test::device_alloc<float>(hidden_f.size());
  float* d_perm = nullptr;
  float* d_w = kops_test::device_alloc<float>(weights.size());
  float* d_out = kops_test::device_alloc<float>(hidden_f.size());
  std::vector<int32_t> map(static_cast<size_t>(experts));
  for (int e = 0; e < experts; ++e) map[static_cast<size_t>(e)] = e;
  kops_test::copy_in(d_ids, slots);
  kops_test::copy_in(d_map, map);
  kops_test::copy_in(d_hidden, hidden_f);
  kops_test::copy_in(d_w, weights);
  cudaMemset(d_cursor, 0, static_cast<size_t>(experts) * sizeof(int32_t));
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, slots.size());
  const Tensor t_map = tensor1d(d_map, DType::kInt32, experts);
  const Tensor t_counts = tensor1d(d_counts, DType::kInt32, experts);
  const Tensor t_offsets = tensor1d(d_offsets, DType::kInt64, experts + 1);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::moe::count_and_offsets(t_ids, t_counts, t_offsets, t_map, experts, 1, ctx),
                 "permute offsets");
  const auto offsets = kops_test::copy_out(d_offsets, static_cast<size_t>(experts + 1));
  const int rows = static_cast<int>(offsets[experts]);
  d_perm = kops_test::device_alloc<float>(static_cast<size_t>(rows) * hidden);
  kops::moe::PermuteParams pp{};
  pp.hidden = d_hidden;
  pp.hidden_stride = hidden;
  pp.topk_ids = d_ids;
  pp.topk_weights = d_w;
  pp.apply_weight_on_input = true;
  pp.expert_offsets = d_offsets;
  pp.expert_map = d_map;
  pp.write_cursor = d_cursor;
  pp.num_local_experts = experts;
  pp.permuted = d_perm;
  pp.permuted_stride = hidden;
  pp.permuted_row_to_slot = d_row_slot;
  pp.slot_to_permuted_row = d_slot_row;
  pp.num_tokens = tokens;
  pp.topk = topk;
  pp.hidden_size = hidden;
  KOPS_EXPECT_OK(kops::moe::permute(pp, DType::kFloat32, ctx), "permute");
  const auto slot_row = kops_test::copy_out(d_slot_row, slots.size());
  const auto perm = kops_test::copy_out(d_perm, static_cast<size_t>(rows) * hidden);
  // Each slot maps to a unique row ordered by slot id inside its expert.
  for (int e = 0; e < experts; ++e) {
    int64_t next = offsets[static_cast<size_t>(e)];
    for (size_t i = 0; i < slots.size(); ++i) {
      if (slots[i] != e) continue;
      KOPS_TEST_CHECK(slot_row[i] == next);
      for (int c = 0; c < hidden; ++c) {
        KOPS_TEST_CHECK(std::fabs(perm[static_cast<size_t>(next * hidden + c)] -
                                  hidden_f[static_cast<size_t>((i / topk) * hidden + c)] * 1.0f) < 1e-6f);
      }
      ++next;
    }
  }
  // Unpermute: weighted sum with shared + residual.
  std::vector<float> shared(static_cast<size_t>(tokens * hidden), 0.25f);
  std::vector<float> residual(static_cast<size_t>(tokens * hidden), -0.5f);
  float* d_shared = kops_test::device_alloc<float>(shared.size());
  float* d_res = kops_test::device_alloc<float>(residual.size());
  kops_test::copy_in(d_shared, shared);
  kops_test::copy_in(d_res, residual);
  const Tensor t_out = kops_test::tensor2d(d_out, DType::kFloat32, tokens, hidden);
  const Tensor t_perm = kops_test::tensor2d(d_perm, DType::kFloat32, rows, hidden);
  const Tensor t_slot_row = tensor1d(d_slot_row, DType::kInt32, slots.size());
  const Tensor t_w = tensor1d(d_w, DType::kFloat32, weights.size());
  const Tensor t_shared = kops_test::tensor2d(d_shared, DType::kFloat32, tokens, hidden);
  const Tensor t_res = kops_test::tensor2d(d_res, DType::kFloat32, tokens, hidden);
  KOPS_EXPECT_OK(kops::moe::unpermute_reduce(t_out, t_perm, t_slot_row, t_w, t_shared, t_res, 1.0f,
                                             false, ctx),
                 "unpermute");
  const auto out = kops_test::copy_out(d_out, hidden_f.size());
  for (int t = 0; t < tokens; ++t) {
    for (int c = 0; c < hidden; ++c) {
      double want = 0.0;
      for (int r = 0; r < topk; ++r) {
        const int32_t row = slot_row[static_cast<size_t>(t * topk + r)];
        want += static_cast<double>(perm[static_cast<size_t>(row * hidden + c)]);
      }
      want += shared[static_cast<size_t>(t * hidden + c)] + residual[static_cast<size_t>(t * hidden + c)];
      KOPS_TEST_CHECK(std::fabs(out[static_cast<size_t>(t * hidden + c)] - want) < 1e-5);
    }
  }
  kops_test::device_free(d_ids);
  kops_test::device_free(d_map);
  kops_test::device_free(d_offsets);
  kops_test::device_free(d_counts);
  kops_test::device_free(d_cursor);
  kops_test::device_free(d_slot_row);
  kops_test::device_free(d_row_slot);
  kops_test::device_free(d_hidden);
  kops_test::device_free(d_perm);
  kops_test::device_free(d_w);
  kops_test::device_free(d_out);
  kops_test::device_free(d_shared);
  kops_test::device_free(d_res);
}

void test_permute_quant() {
  const int tokens = 3;
  const int topk = 1;
  const int experts = 4;
  const int hidden = 256;
  std::vector<int32_t> slots = {0, 1, 2};
  std::vector<float> hidden_f(static_cast<size_t>(tokens * hidden));
  fill(hidden_f, 51u, -2.0f, 2.0f);
  int32_t* d_ids = kops_test::device_alloc<int32_t>(slots.size());
  int64_t* d_offsets = kops_test::device_alloc<int64_t>(experts + 1);
  int32_t* d_counts = kops_test::device_alloc<int32_t>(experts);
  int32_t* d_cursor = kops_test::device_alloc<int32_t>(experts);
  int32_t* d_slot_row = kops_test::device_alloc<int32_t>(slots.size());
  int32_t* d_row_slot = kops_test::device_alloc<int32_t>(slots.size());
  float* d_hidden = kops_test::device_alloc<float>(hidden_f.size());
  uint8_t* d_q = kops_test::device_alloc<uint8_t>(hidden_f.size());
  float* d_scales = kops_test::device_alloc<float>(static_cast<size_t>(tokens * (hidden / 128)));
  kops_test::copy_in(d_ids, slots);
  kops_test::copy_in(d_hidden, hidden_f);
  cudaMemset(d_cursor, 0, static_cast<size_t>(experts) * sizeof(int32_t));
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, slots.size());
  const Tensor t_counts = tensor1d(d_counts, DType::kInt32, experts);
  const Tensor t_offsets = tensor1d(d_offsets, DType::kInt64, experts + 1);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::moe::count_and_offsets(t_ids, t_counts, t_offsets, Tensor{}, experts, 1, ctx),
                 "quant offsets");
  kops::moe::PermuteParams pp{};
  pp.hidden = d_hidden;
  pp.hidden_stride = hidden;
  pp.topk_ids = d_ids;
  pp.expert_offsets = d_offsets;
  pp.write_cursor = d_cursor;
  pp.num_local_experts = experts;
  pp.permuted = d_q;
  pp.permuted_stride = hidden;
  pp.permuted_row_to_slot = d_row_slot;
  pp.slot_to_permuted_row = d_slot_row;
  pp.quant.q = d_q;
  pp.quant.q_row_stride = hidden;
  pp.quant.scales = d_scales;
  pp.quant.scale_row_stride = hidden / 128;
  pp.quant.q_dtype = DType::kFloat8E4M3;
  pp.quant.mode = kops::ActQuantMode::kPerGroup;
  pp.quant.group_size = 128;
  pp.num_tokens = tokens;
  pp.topk = topk;
  pp.hidden_size = hidden;
  KOPS_EXPECT_OK(kops::moe::permute(pp, DType::kFloat32, ctx), "permute quant");
  const auto q = kops_test::copy_out(d_q, hidden_f.size());
  const auto scales = kops_test::copy_out(d_scales, static_cast<size_t>(tokens * (hidden / 128)));
  const auto rows = kops_test::copy_out(d_slot_row, slots.size());
  for (int t = 0; t < tokens; ++t) {
    for (int g = 0; g < hidden / 128; ++g) {
      float amax = 0.0f;
      for (int i = 0; i < 128; ++i) {
        amax = std::max(amax, std::fabs(hidden_f[static_cast<size_t>(t * hidden + g * 128 + i)]));
      }
      const float want_scale = std::max(amax, 1e-4f) / 448.0f;
      KOPS_TEST_CHECK(std::fabs(scales[static_cast<size_t>(t * (hidden / 128) + g)] - want_scale) < 1e-6f);
      const int row = rows[static_cast<size_t>(t)];
      for (int i = 0; i < 128; ++i) {
        const float deq =
            e4m3_host(q[static_cast<size_t>(row * hidden + g * 128 + i)]) * want_scale;
        KOPS_TEST_CHECK(std::fabs(deq - hidden_f[static_cast<size_t>(t * hidden + g * 128 + i)]) <=
                        0.08f * std::fabs(hidden_f[static_cast<size_t>(t * hidden + g * 128 + i)]) + 1e-3f);
      }
    }
  }
  kops_test::device_free(d_ids);
  kops_test::device_free(d_offsets);
  kops_test::device_free(d_counts);
  kops_test::device_free(d_cursor);
  kops_test::device_free(d_slot_row);
  kops_test::device_free(d_row_slot);
  kops_test::device_free(d_hidden);
  kops_test::device_free(d_q);
  kops_test::device_free(d_scales);
}

void test_moe_sum_and_latent() {
  const int tokens = 4;
  const int topk = 4;
  const int hidden = 256;
  std::vector<float> in_f(static_cast<size_t>(tokens * topk * hidden));
  fill(in_f, 61u);
  float* d_in = kops_test::device_alloc<float>(in_f.size());
  float* d_out = kops_test::device_alloc<float>(static_cast<size_t>(tokens * hidden));
  kops_test::copy_in(d_in, in_f);
  const Tensor t_out = kops_test::tensor2d(d_out, DType::kFloat32, tokens, hidden);
  const Tensor t_in = kops_test::tensor3d(d_in, DType::kFloat32, tokens, topk, hidden);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::moe::moe_sum(t_out, t_in, topk, ctx), "moe_sum");
  const auto sum = kops_test::copy_out(d_out, static_cast<size_t>(tokens * hidden));
  for (int t = 0; t < tokens; ++t) {
    for (int c = 0; c < hidden; ++c) {
      double want = 0.0;
      for (int k = 0; k < topk; ++k) want += in_f[static_cast<size_t>((t * topk + k) * hidden + c)];
      KOPS_TEST_CHECK(std::fabs(sum[static_cast<size_t>(t * hidden + c)] - want) < 1e-5);
    }
  }
  // Latent combine + RMSNorm.
  const int rows = tokens * topk;
  std::vector<float> e_f(static_cast<size_t>(rows * hidden));
  std::vector<float> w_f(static_cast<size_t>(rows));
  std::vector<float> nw(static_cast<size_t>(hidden));
  std::vector<int32_t> slots(static_cast<size_t>(rows));
  fill(e_f, 63u);
  fill(nw, 65u, 0.5f, 1.5f);
  for (int i = 0; i < rows; ++i) {
    slots[static_cast<size_t>(i)] = i;
    w_f[static_cast<size_t>(i)] = 0.25f + 0.1f * static_cast<float>(i % topk);
  }
  float* d_e = kops_test::device_alloc<float>(e_f.size());
  float* d_w = kops_test::device_alloc<float>(w_f.size());
  float* d_nw = kops_test::device_alloc<float>(nw.size());
  int32_t* d_slots = kops_test::device_alloc<int32_t>(slots.size());
  float* d_lat = kops_test::device_alloc<float>(static_cast<size_t>(tokens * hidden));
  kops_test::copy_in(d_e, e_f);
  kops_test::copy_in(d_w, w_f);
  kops_test::copy_in(d_nw, nw);
  kops_test::copy_in(d_slots, slots);
  const Tensor t_lat = kops_test::tensor2d(d_lat, DType::kFloat32, tokens, hidden);
  const Tensor t_e = kops_test::tensor2d(d_e, DType::kFloat32, rows, hidden);
  const Tensor t_w = tensor1d(d_w, DType::kFloat32, w_f.size());
  const Tensor t_nw = tensor1d(d_nw, DType::kFloat32, nw.size());
  const Tensor t_slots = tensor1d(d_slots, DType::kInt32, slots.size());
  KOPS_EXPECT_OK(kops::moe::latent_combine_norm(t_lat, t_e, t_slots, t_w, t_nw, 1.0f, 1e-5f, topk,
                                                kops::ActQuantOut{}, ctx),
                 "latent_combine_norm");
  const auto lat = kops_test::copy_out(d_lat, static_cast<size_t>(tokens * hidden));
  for (int t = 0; t < tokens; ++t) {
    std::vector<double> z(static_cast<size_t>(hidden), 0.0);
    for (int c = 0; c < hidden; ++c) {
      for (int k = 0; k < topk; ++k) {
        z[static_cast<size_t>(c)] +=
            static_cast<double>(w_f[static_cast<size_t>(t * topk + k)]) *
            e_f[static_cast<size_t>((t * topk + k) * hidden + c)];
      }
    }
    double ss = 0.0;
    for (double v : z) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / hidden + 1e-5);
    for (int c = 0; c < hidden; ++c) {
      const double want = z[static_cast<size_t>(c)] * inv * nw[static_cast<size_t>(c)];
      KOPS_TEST_CHECK(std::fabs(lat[static_cast<size_t>(t * hidden + c)] - want) < 1e-4);
    }
  }
  kops_test::device_free(d_in);
  kops_test::device_free(d_out);
  kops_test::device_free(d_e);
  kops_test::device_free(d_w);
  kops_test::device_free(d_nw);
  kops_test::device_free(d_slots);
  kops_test::device_free(d_lat);
}

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops::moe::RouterParams p{};
  int32_t ids[8] = {};
  float w[8] = {};
  p.num_tokens = 1;
  p.num_experts = 8;
  p.topk = 2;
  p.num_expert_groups = 1;
  p.topk_groups = 1;
  p.topk_ids = ids;
  p.topk_weights = w;
  p.logits_stride = 8;
  KOPS_EXPECT_STATUS(kops::moe::route(p, kops::moe::RouterScore::kSoftmax, DType::kFloat32, ctx),
                     kops::Status::kInvalidArgument, "route null logits");
  p.topk = 17;
  KOPS_EXPECT_STATUS(kops::moe::route(p, kops::moe::RouterScore::kSoftmax, DType::kFloat32, ctx),
                     kops::Status::kUnsupportedShape, "route topk too large");
  p.topk = 2;
  p.num_experts = 2048;
  KOPS_EXPECT_STATUS(kops::moe::route(p, kops::moe::RouterScore::kSoftmax, DType::kFloat32, ctx),
                     kops::Status::kUnsupportedShape, "route experts too large");

  Tensor t{};
  KOPS_EXPECT_STATUS(kops::moe::moe_sum(t, t, 4, ctx), kops::Status::kInvalidArgument,
                     "moe_sum null tensors");
  KOPS_EXPECT_STATUS(kops::moe::align_block_size(t, t, t, t, t, 8, 1, kops::Workspace{}, ctx),
                     kops::Status::kInvalidArgument, "align null tensors");
}

// =============================================================================
// B7 row 8: moe::moe_gemm, the expert GEMM on tensor cores (gemm_mma_sync_kernel over the block-sorted slots
// of align_block_size). Oracle: fp64 host loops over the decoded operands with the documented row semantics
// (A row = slot / top_k for W13, slot for W2; C row = slot; padding slots and experts < 0 untouched) and the
// epilogue order bias -> GLU (interleaved) -> routed weight. bf16 cases are also compared with moe_gemv.
// =============================================================================
namespace b7moe {

template <typename T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  explicit Buf(size_t count) : p(kops_test::device_alloc<T>(count ? count : 1)), n(count) {}
  explicit Buf(const std::vector<T>& h) : Buf(h.size()) { if (!h.empty()) kops_test::copy_in(p, h); }
  ~Buf() { kops_test::device_free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  std::vector<T> get() const { return kops_test::copy_out(p, n); }
};

struct Rng {
  std::mt19937 g;
  explicit Rng(unsigned seed) : g(seed) {}
  float uni(float lo, float hi) { return lo + (hi - lo) * (static_cast<float>(g() >> 8) / 16777216.0f); }
  int range(int n) { return static_cast<int>(g() % static_cast<uint32_t>(n)); }
};

uint8_t e4m3_enc(float x) { return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)); }
float e4m3_dec(uint8_t b) { return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)); }

enum { W_BF16 = 0, W_FP8_EXPERT, W_FP8_CHANNEL, W_FP8_BLOCK, W_FP8_BLOCK_CM, W_FP8_BLOCK_UE, W_INT8 };
const char* const kWName[] = {"16-bit", "e4m3 per expert", "e4m3 per channel", "e4m3 128x128", "e4m3 128x128 col-major sA",
                              "e4m3 128x128 ue8m0", "int8 per channel"};

struct Case {
  int wmode = W_BF16;
  int block = 16;
  bool w2 = false;        // stage W2 (A rows are slots, routed weight) else W13 (GLU on interleaved weights)
  bool glu = true;        // W13 only
  int glu_db = 1;
  bool fp16 = false;      // 16-bit activations / output in fp16
  bool fp32_out = false;
  bool drop_expert = true;
  unsigned seed = 1;
};

constexpr int kE = 8, kTokens = 40, kTopK = 2, kSlots = kTokens * kTopK;
constexpr int kEmptyExpert = 5, kEpExpert = 6;

struct Routing {
  std::vector<int32_t> expert_of;                  // [slots]
  std::vector<int32_t> sorted, expert_ids;         // device results of align_block_size (expert_ids EP-edited)
  int32_t post_pad = 0;
};

/// align_block_size on random routing (expert kEmptyExpert gets nothing); blocks of kEpExpert are set to -1
/// afterwards, as an EP rank's local id map does.
Routing route(const kops::LaunchContext& ctx, int block, bool drop, unsigned seed) {
  Routing R;
  Rng r(seed);
  R.expert_of.resize(kSlots);
  for (auto& e : R.expert_of) {
    do { e = r.range(kE); } while (e == kEmptyExpert);
  }
  R.expert_of[0] = kEpExpert;
  const int64_t cap = kSlots + static_cast<int64_t>(kE) * (block - 1);
  const int64_t blocks = (cap + block - 1) / block + 1;
  Buf<int32_t> ids(R.expert_of), sorted(static_cast<size_t>(cap)), eids(static_cast<size_t>(blocks)), post(1), scratch(kE + 1);
  kops::Workspace ws{};
  ws.device_scratch = scratch.p;
  ws.device_scratch_bytes = (kE + 1) * sizeof(int32_t);
  KOPS_EXPECT_OK(kops::moe::align_block_size(tensor1d(ids.p, DType::kInt32, kSlots), tensor1d(sorted.p, DType::kInt32, cap),
                                             tensor1d(eids.p, DType::kInt32, blocks), tensor1d(post.p, DType::kInt32, 1), Tensor{}, kE, block, ws, ctx),
                 "b7 moe_gemm align_block_size");
  cudaDeviceSynchronize();
  R.sorted = sorted.get();
  R.expert_ids = eids.get();
  R.post_pad = post.get()[0];
  if (drop) {
    for (int b = 0; b < R.post_pad / block; ++b) {
      if (R.expert_ids[b] == kEpExpert) R.expert_ids[b] = -1;
    }
  }
  return R;
}

struct Result {
  std::vector<uint8_t> c;   // raw bytes of C
  bool ok = false;
};

/// Element-wise one output ulp + 2e-5 rms, Frobenius 0.6 ulp (the gemm tests' rule). `mag` is the optional
/// FP8 bound: 2^-11 * sum|a*w*scale|, the e4m3 mma's effective accumulation mantissa (expect_close_fp8).
void close(const std::vector<double>& got, const std::vector<double>& want, double eps, const char* what,
           const std::vector<double>* mag = nullptr) {
  double ss = 0.0, num = 0.0, mag_ss = 0.0;
  for (double w : want) ss += w * w;
  if (mag != nullptr) for (double m : *mag) mag_ss += m * m;
  const double rms = std::sqrt(ss / static_cast<double>(want.empty() ? 1 : want.size()));
  size_t bad = 0, worst = 0;
  double worst_err = -1.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double err = std::fabs(got[i] - want[i]);
    num += err * err;
    const double lim = eps * std::fabs(want[i]) + 2e-5 * rms + (mag != nullptr ? std::ldexp((*mag)[i], -11) : 0.0);
    if (!(err <= lim)) {
      ++bad;
      if (err > worst_err) { worst_err = err; worst = i; }
    }
  }
  const double frob = ss > 0.0 ? std::sqrt(num / ss) : std::sqrt(num);
  // FP8: the aggregate bound follows the same 2^-11 sum|p| term (mag_rms / rms is the relative size of
  // the accumulated magnitudes; 1 for 16-bit / INT8 because mag is then never passed).
  const double frob_lim = 0.6 * (eps + (mag != nullptr && ss > 0.0 ? std::ldexp(std::sqrt(mag_ss / ss), -11) : 0.0));
  if (bad != 0 || !(frob <= frob_lim)) {
    char det[256];
    std::snprintf(det, sizeof(det), "%zu/%zu off, frobenius %.3g, worst #%zu got %.7g want %.7g", bad, want.size(), frob, worst,
                  bad ? got[worst] : 0.0, bad ? want[worst] : 0.0);
    kops_test::record_failure(what, det, __FILE__, __LINE__);
  }
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

/// One moe_gemm stage checked against the oracle. Returns C's bytes for bit comparisons.
Result run(const kops::LaunchContext& ctx, const Case& cs, const Routing& R, const char* what_in, bool via_gemv = false) {
  char what[300];
  std::snprintf(what, sizeof(what), "%s [%s block %d %s glu %d db %d%s%s]", what_in, kWName[cs.wmode], cs.block, cs.w2 ? "W2" : "W13",
                cs.glu && !cs.w2, cs.glu_db, cs.fp16 ? " fp16" : "", cs.fp32_out ? " fp32 out" : "");
  Result res;
  Rng r(777u + cs.seed * 13u + static_cast<unsigned>(cs.wmode) * 7u + (cs.w2 ? 1u : 0u));
  const bool glu = cs.glu && !cs.w2;
  const int K = 256, N = cs.w2 ? 192 : 256, n_out = glu ? N / 2 : N, ldc = n_out + 4;
  const int a_rows = cs.w2 ? kSlots : kTokens;
  const bool fp8 = cs.wmode >= W_FP8_EXPERT && cs.wmode <= W_FP8_BLOCK_UE;
  const bool i8 = cs.wmode == W_INT8;
  const bool block = cs.wmode == W_FP8_BLOCK || cs.wmode == W_FP8_BLOCK_CM || cs.wmode == W_FP8_BLOCK_UE;
  const bool ue = cs.wmode == W_FP8_BLOCK_UE;
  const bool cm = cs.wmode == W_FP8_BLOCK_CM;
  const int kb = K / 128, nb = (N + 127) / 128;
  // operands
  std::vector<uint8_t> a_b, w_b;
  std::vector<float> a_v, w_v;
  auto gen = [&](size_t n, float lo, float hi, std::vector<uint8_t>* b, std::vector<float>* v) {
    v->resize(n);
    b->resize(n * ((fp8 || i8) ? 1 : 2));
    for (size_t i = 0; i < n; ++i) {
      if (fp8) {
        (*b)[i] = e4m3_enc(r.uni(2 * lo, 2 * hi));
        (*v)[i] = e4m3_dec((*b)[i]);
      } else if (i8) {
        const int q = r.range(129) - 64;
        (*b)[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
        (*v)[i] = static_cast<float>(q);
      } else if (cs.fp16) {
        __half h;
        kops_test::from_float(h, r.uni(lo, hi));
        std::memcpy(&(*b)[2 * i], &h, 2);
        (*v)[i] = kops_test::to_float(h);
      } else {
        __nv_bfloat16 h;
        kops_test::from_float(h, r.uni(lo, hi));
        std::memcpy(&(*b)[2 * i], &h, 2);
        (*v)[i] = kops_test::to_float(h);
      }
    }
  };
  gen(static_cast<size_t>(a_rows) * K, -1.0f, 1.0f, &a_b, &a_v);
  gen(static_cast<size_t>(kE) * N * K, -0.5f, 0.5f, &w_b, &w_v);
  // scales (oracle values + device bytes)
  std::vector<float> sa, sw;
  std::vector<uint8_t> sa_dev, sw_dev;
  auto f32_bytes = [](const std::vector<float>& v) {
    std::vector<uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
  };
  if (cs.wmode == W_FP8_EXPERT) {
    sa = {0.8f};
    sw.resize(kE);
    for (auto& v : sw) v = r.uni(0.5f, 1.5f);
  } else if (cs.wmode == W_FP8_CHANNEL || i8) {
    sa.resize(a_rows);
    sw.resize(static_cast<size_t>(kE) * N);
    for (auto& v : sa) v = i8 ? r.uni(0.01f, 0.03f) : r.uni(0.5f, 1.5f);
    for (auto& v : sw) v = i8 ? r.uni(0.01f, 0.03f) : r.uni(0.5f, 1.5f);
  } else if (block) {
    auto pick = [&](uint8_t* e8) {
      if (ue) { const int p = r.range(5) - 2; *e8 = static_cast<uint8_t>(127 + p); return std::ldexp(1.0f, p); }
      return r.uni(0.5f, 2.0f);
    };
    const int ldm = (a_rows + 3) / 4 * 4;
    sa.resize(static_cast<size_t>(a_rows) * kb);
    std::vector<float> sa_f(cm ? static_cast<size_t>(kb) * ldm : sa.size(), 0.0f);
    std::vector<uint8_t> sa_e(sa_f.size(), 0);
    for (int row = 0; row < a_rows; ++row) {
      for (int j = 0; j < kb; ++j) {
        uint8_t e8 = 0;
        const float v = pick(&e8);
        sa[static_cast<size_t>(row) * kb + j] = v;
        const size_t pos = cm ? static_cast<size_t>(j) * ldm + row : static_cast<size_t>(row) * kb + j;
        sa_f[pos] = v;
        sa_e[pos] = e8;
      }
    }
    sw.resize(static_cast<size_t>(kE) * nb * kb);
    std::vector<uint8_t> sw_e(sw.size());
    for (size_t i = 0; i < sw.size(); ++i) sw[i] = pick(&sw_e[i]);
    sa_dev = ue ? sa_e : f32_bytes(sa_f);
    sw_dev = ue ? sw_e : f32_bytes(sw);
  }
  if (!block) {
    sa_dev = f32_bytes(sa);
    sw_dev = f32_bytes(sw);
  }
  // bias [E, N] in c_dtype, routed weights
  std::vector<float> bias_v(static_cast<size_t>(kE) * N);
  std::vector<uint8_t> bias_b(bias_v.size() * (cs.fp32_out ? 4 : 2));
  for (size_t i = 0; i < bias_v.size(); ++i) {
    const float x = r.uni(-1.0f, 1.0f);
    if (cs.fp32_out) { bias_v[i] = x; std::memcpy(&bias_b[4 * i], &x, 4); }
    else if (cs.fp16) { __half h; kops_test::from_float(h, x); bias_v[i] = kops_test::to_float(h); std::memcpy(&bias_b[2 * i], &h, 2); }
    else { __nv_bfloat16 h; kops_test::from_float(h, x); bias_v[i] = kops_test::to_float(h); std::memcpy(&bias_b[2 * i], &h, 2); }
  }
  std::vector<float> tw(kSlots);
  for (auto& v : tw) v = r.uni(0.3f, 1.2f);
  // oracle
  const int c_rows = kSlots + 8;   // rows past the valid slots catch writes of padding ids
  std::vector<double> ref(static_cast<size_t>(c_rows) * ldc, 0.0), ref_mag(ref.size(), 0.0);
  std::vector<char> written(ref.size(), 0);
  std::vector<double> lin(N), lin_mag(N);
  for (int s = 0; s < kSlots; ++s) {
    const int e = R.expert_of[s];
    if (cs.drop_expert && e == kEpExpert) continue;
    const int src = cs.w2 ? s : s / kTopK;
    const float* av = &a_v[static_cast<size_t>(src) * K];
    for (int c = 0; c < N; ++c) {
      const float* wv = &w_v[(static_cast<size_t>(e) * N + c) * K];
      double acc = 0.0, mag = 0.0;
      if (block) {
        for (int j = 0; j < kb; ++j) {
          double part = 0.0, pmag = 0.0;
          for (int kk = 128 * j; kk < 128 * j + 128; ++kk) {
            const double p = static_cast<double>(av[kk]) * wv[kk];
            part += p;
            pmag += std::fabs(p);
          }
          const double sc = static_cast<double>(sa[static_cast<size_t>(src) * kb + j]) *
                            sw[(static_cast<size_t>(e) * nb + c / 128) * kb + j];
          acc += part * sc;
          mag += pmag * std::fabs(sc);
        }
      } else {
        for (int kk = 0; kk < K; ++kk) {
          const double p = static_cast<double>(av[kk]) * wv[kk];
          acc += p;
          mag += std::fabs(p);
        }
        if (cs.wmode == W_FP8_EXPERT) {
          const double sc = static_cast<double>(sa[0]) * sw[e];
          acc *= sc;
          mag *= std::fabs(sc);
        }
        if (cs.wmode == W_FP8_CHANNEL || i8) {
          const double sc = static_cast<double>(sa[src]) * sw[static_cast<size_t>(e) * N + c];
          acc *= sc;
          mag *= std::fabs(sc);
        }
      }
      acc += bias_v[static_cast<size_t>(e) * N + c];
      lin[c] = acc;
      lin_mag[c] = mag;
    }
    for (int oc = 0; oc < n_out; ++oc) {
      double v = lin[oc], mag = lin_mag[oc];
      if (glu) {
        const int gc = (oc / cs.glu_db) * 2 * cs.glu_db + oc % cs.glu_db;
        const double gate = lin[gc], up = lin[gc + cs.glu_db];
        const double sig = 1.0 / (1.0 + std::exp(-gate));
        v = silu(gate) * up;
        mag = std::fabs(silu(gate)) * lin_mag[gc + cs.glu_db] +
              std::fabs(up) * std::fabs(sig * (1.0 + gate * (1.0 - sig))) * lin_mag[gc];
      }
      if (cs.w2) {
        v *= tw[s];
        mag *= std::fabs(tw[s]);
      }
      ref[static_cast<size_t>(s) * ldc + oc] = v;
      ref_mag[static_cast<size_t>(s) * ldc + oc] = mag;
      written[static_cast<size_t>(s) * ldc + oc] = 1;
    }
  }
  // device
  const int ce = cs.fp32_out ? 4 : 2;
  Buf<uint8_t> da(a_b), dw(w_b), dsa(sa_dev.empty() ? std::vector<uint8_t>(16) : sa_dev), dsw(sw_dev.empty() ? std::vector<uint8_t>(16) : sw_dev);
  Buf<uint8_t> dbias(bias_b);
  Buf<float> dtw(tw);
  Buf<int32_t> dsorted(R.sorted), deids(R.expert_ids), dpost(std::vector<int32_t>(1, R.post_pad));
  std::vector<uint8_t> c0(static_cast<size_t>(c_rows) * ldc * ce, 0);
  {
    const float sen = 7.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (cs.fp32_out) std::memcpy(&c0[4 * i], &sen, 4);
      else if (cs.fp16) { __half h; kops_test::from_float(h, sen); std::memcpy(&c0[2 * i], &h, 2); }
      else { __nv_bfloat16 h; kops_test::from_float(h, sen); std::memcpy(&c0[2 * i], &h, 2); }
    }
  }
  Buf<uint8_t> dc(c0);
  kops::moe::FusedMoEGemmParams p{};
  p.a = da.p; p.lda = K; p.a_rows_are_slots = cs.w2;
  p.w = dw.p; p.w_expert_stride = static_cast<int64_t>(N) * K;
  p.bias = dbias.p; p.c = dc.p; p.ldc = ldc;
  p.sorted_token_ids = dsorted.p; p.expert_ids = deids.p; p.num_tokens_post_pad = dpost.p;
  p.topk_weights = dtw.p; p.mul_routed_weight = cs.w2; p.top_k = kTopK; p.n = N; p.k = K;
  p.num_valid_slots = kSlots;
  if (glu) {
    p.epi.ops = kops::gemm::kEpiGLU;
    p.epi.act = kops::ActFn::kSiLU;
    p.epi.glu_layout = kops::GateLayout::kInterleaved;
    p.epi.glu_interleave = cs.glu_db;
  }
  if (fp8 || i8) {
    p.a_scale = dsa.p;
    p.w_scale = dsw.p;
    p.w_quant.storage_dtype = fp8 ? DType::kFloat8E4M3 : DType::kInt8;
    if (cs.wmode == W_FP8_EXPERT) {
      p.w_quant.granularity = kops::ScaleGranularity::kPerTensor;
      p.a_quant.granularity = kops::ScaleGranularity::kPerTensor;
    } else if (block) {
      p.w_quant.granularity = kops::ScaleGranularity::kPerBlock2D;
      p.w_quant.group_m = 128; p.w_quant.group_k = 128;
      p.w_quant.scale_dtype = ue ? DType::kFloat8E8M0 : DType::kFloat32;
      p.a_quant.granularity = kops::ScaleGranularity::kPerGroup;
      p.a_quant.group_k = 128;
      p.a_quant.scale_dtype = ue ? DType::kFloat8E8M0 : DType::kFloat32;
      p.a_quant.scale_layout = cm ? kops::ScaleLayout::kColMajorTma : kops::ScaleLayout::kRowMajor;
      p.w_scale_expert_stride = 0;   // default [E, ceil(N / 128), K / 128]
    } else {
      p.w_quant.granularity = kops::ScaleGranularity::kPerChannel;
      p.a_quant.granularity = kops::ScaleGranularity::kPerToken;
      p.w_scale_expert_stride = N;
    }
  }
  const DType adt = fp8 ? DType::kFloat8E4M3 : (i8 ? DType::kInt8 : (cs.fp16 ? DType::kFloat16 : DType::kBFloat16));
  const DType cdt = cs.fp32_out ? DType::kFloat32 : (cs.fp16 ? DType::kFloat16 : DType::kBFloat16);
  const Status st = via_gemv ? kops::moe::moe_gemv(p, kE, cs.block, adt, adt, ctx) : kops::moe::moe_gemm(p, kE, cs.block, adt, adt, cdt, ctx);
  if (st != Status::kSuccess) {
    KOPS_EXPECT_OK(st, what);
    return res;
  }
  cudaDeviceSynchronize();
  res.c = dc.get();
  res.ok = true;
  std::vector<double> got, want, want_mag;
  size_t untouched = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double g;
    if (cs.fp32_out) { float f; std::memcpy(&f, &res.c[4 * i], 4); g = f; }
    else if (cs.fp16) { __half h; std::memcpy(&h, &res.c[2 * i], 2); g = kops_test::to_float(h); }
    else { __nv_bfloat16 h; std::memcpy(&h, &res.c[2 * i], 2); g = kops_test::to_float(h); }
    if (written[i]) { got.push_back(g); want.push_back(ref[i]); want_mag.push_back(ref_mag[i]); }
    else untouched += g != 7.0;
  }
  close(got, want, cs.fp32_out ? 2e-6 : (cs.fp16 ? 0.0009765625 : 0.0078125), what, fp8 ? &want_mag : nullptr);
  if (untouched != 0) {
    char det[160];
    std::snprintf(det, sizeof(det), "%zu elements of padding slots / EP experts / ld padding were written", untouched);
    kops_test::record_failure(what, det, __FILE__, __LINE__);
  }
  return res;
}


// ---- B17 experimental W2 experts -------------------------------------------------------------
float w2_level_host(int code) {
  const float l[4] = {-4.0f, -1.0f, 1.0f, 4.0f};
  return l[code & 3];
}
int w2_code_host(float z) { return z < -2.5f ? 0 : (z < 0.0f ? 1 : (z < 2.5f ? 2 : 3)); }

struct W2Pack {
  std::vector<uint8_t> codes;
  std::vector<float> scales;
};

W2Pack w2_pack_host(const std::vector<__nv_bfloat16>& w, int E, int N, int K, int group) {
  W2Pack out;
  out.codes.assign(static_cast<size_t>(E) * N * (K / 4), 0);
  out.scales.assign(static_cast<size_t>(E) * N * (K / group), 0.0f);
  const int ng = K / group;
  for (int e = 0; e < E; ++e) {
    for (int n = 0; n < N; ++n) {
      for (int g = 0; g < ng; ++g) {
        float amax = 0.0f;
        float x[128];
        for (int j = 0; j < group; ++j) {
          x[j] = __bfloat162float(w[(static_cast<size_t>(e) * N + n) * K + g * group + j]);
          amax = std::max(amax, std::fabs(x[j]));
        }
        const float scale = amax > 0.0f ? amax / 4.0f : 0.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int j = 0; j < group; j += 4) {
          uint8_t b = 0;
          for (int t = 0; t < 4; ++t) {
            const int c = scale > 0.0f ? w2_code_host(x[j + t] * inv) : 2;
            b = static_cast<uint8_t>(b | (c << (2 * t)));
          }
          out.codes[(static_cast<size_t>(e) * N + n) * (K / 4) + g * (group / 4) + j / 4] = b;
        }
        out.scales[(static_cast<size_t>(e) * N + n) * ng + g] = scale;
      }
    }
  }
  return out;
}

void test_moe_w2_pack(const kops::LaunchContext& ctx) {
  const int E = 2, N = 4, K = 64, G = 32;
  Rng r(4242);
  std::vector<__nv_bfloat16> w(static_cast<size_t>(E) * N * K);
  for (auto& x : w) x = __float2bfloat16(r.uni(-1.5f, 1.5f));
  const W2Pack ref = w2_pack_host(w, E, N, K, G);
  Buf<__nv_bfloat16> dw(w);
  Buf<uint8_t> dc(ref.codes.size());
  Buf<float> ds(ref.scales.size());
  cudaMemset(dc.p, 0x5A, dc.n);
  cudaMemset(ds.p, 0, ds.n * 4);
  Tensor tw = kops_test::tensor3d(dw.p, DType::kBFloat16, E, N, K);
  Tensor tc = kops_test::tensor3d(dc.p, DType::kUInt8, E, N, K / 4);
  Tensor ts = kops_test::tensor3d(ds.p, DType::kFloat32, E, N, K / G);
  KOPS_EXPECT_OK(kops::experimental::moe_w2::pack(tc, ts, tw, kops::WeightPacking::kNone, G, ctx),
                 "w2 pack bf16");
  cudaDeviceSynchronize();
  const auto got_c = dc.get();
  const auto got_s = ds.get();
  int bad = 0;
  for (size_t i = 0; i < ref.codes.size(); ++i) bad += got_c[i] != ref.codes[i];
  for (size_t i = 0; i < ref.scales.size(); ++i) bad += std::memcmp(&got_s[i], &ref.scales[i], 4) != 0;
  if (bad != 0) kops_test::record_failure("w2 pack bytes", "codes or scales differ", __FILE__, __LINE__);

  // Packed e2m1 source: decode nibbles exactly, then quantize to the four W2 levels.
  static const float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                  0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  std::vector<uint8_t> q4(static_cast<size_t>(E) * N * (K / 2));
  for (auto& b : q4) b = static_cast<uint8_t>(r.range(256));
  std::vector<__nv_bfloat16> w4(static_cast<size_t>(E) * N * K);
  for (size_t i = 0; i < w4.size(); ++i) {
    const uint8_t b = q4[i / 2];
    const float x = kE2M1[(b >> (4 * (i & 1))) & 0x0F];
    w4[i] = __float2bfloat16(x);
  }
  const W2Pack ref4 = w2_pack_host(w4, E, N, K, G);
  Buf<uint8_t> dq4(q4.size());
  kops_test::copy_in(dq4.p, q4);
  Buf<uint8_t> dc4(ref4.codes.size());
  Buf<float> ds4(ref4.scales.size());
  Tensor tq4 = kops_test::tensor3d(dq4.p, DType::kUInt8, E, N, K / 2);
  Tensor tc4 = kops_test::tensor3d(dc4.p, DType::kUInt8, E, N, K / 4);
  Tensor ts4 = kops_test::tensor3d(ds4.p, DType::kFloat32, E, N, K / G);
  KOPS_EXPECT_OK(kops::experimental::moe_w2::pack(tc4, ts4, tq4, kops::WeightPacking::kFP4E2M1, G, ctx),
                 "w2 pack fp4");
  cudaDeviceSynchronize();
  const auto got_c4 = dc4.get();
  const auto got_s4 = ds4.get();
  bad = 0;
  for (size_t i = 0; i < ref4.codes.size(); ++i) bad += got_c4[i] != ref4.codes[i];
  for (size_t i = 0; i < ref4.scales.size(); ++i) bad += std::memcmp(&got_s4[i], &ref4.scales[i], 4) != 0;
  if (bad != 0) kops_test::record_failure("w2 pack fp4 bytes", "codes or scales differ", __FILE__, __LINE__);

  // Contract errors.
  Tensor badshape = tc;
  badshape.shape[2] = K / 8;
  KOPS_EXPECT_STATUS(kops::experimental::moe_w2::pack(badshape, ts, tw, kops::WeightPacking::kNone, G, ctx),
                     Status::kUnsupportedShape, "w2 pack shape");
  KOPS_EXPECT_STATUS(kops::experimental::moe_w2::pack(tc, ts, tw, kops::WeightPacking::kNone, 24, ctx),
                     Status::kUnsupportedShape, "w2 pack group");
  KOPS_EXPECT_STATUS(kops::experimental::moe_w2::pack(tc, ts, tw, kops::WeightPacking::kInt4LowNibbleFirst, G, ctx),
                     Status::kUnsupportedDtype, "w2 pack source");
  Tensor f32w = tw;
  f32w.dtype = DType::kFloat32;
  KOPS_EXPECT_STATUS(kops::experimental::moe_w2::pack(tc, ts, f32w, kops::WeightPacking::kNone, G, ctx),
                     Status::kUnsupportedDtype, "w2 pack weight dtype");
  kops::LaunchContext old{};
  old.arch_id = KOPS_TARGET_SM70;
  KOPS_EXPECT_STATUS(kops::experimental::moe_w2::pack(tc, ts, tw, kops::WeightPacking::kNone, G, old),
                     Status::kUnsupportedArch, "w2 pack arch");
}

void test_moe_w2_gemm(const kops::LaunchContext& ctx) {
  const int N = 192, K = 256, G = 32;
  const int kBlock = 16;
  Rng r(9182);
  Routing R = route(ctx, kBlock, true, 11);
  const int cap = static_cast<int>(R.sorted.size());
  std::vector<__nv_bfloat16> w(static_cast<size_t>(kE) * N * K);
  for (auto& x : w) x = __float2bfloat16(r.uni(-0.5f, 0.5f));
  const W2Pack ref = w2_pack_host(w, kE, N, K, G);
  Buf<__nv_bfloat16> dw(w);
  Buf<uint8_t> dc(ref.codes);
  Buf<float> ds(ref.scales);
  Tensor tw = kops_test::tensor3d(dw.p, DType::kBFloat16, kE, N, K);
  Tensor tc = kops_test::tensor3d(dc.p, DType::kUInt8, kE, N, K / 4);
  Tensor ts = kops_test::tensor3d(ds.p, DType::kFloat32, kE, N, K / G);
  KOPS_EXPECT_OK(kops::experimental::moe_w2::pack(tc, ts, tw, kops::WeightPacking::kNone, G, ctx),
                 "w2 gemm pack");

  // A rows are slots (W2 stage); routed weights are indexed by slot.
  const int slots = kSlots;
  std::vector<__nv_bfloat16> a(static_cast<size_t>(slots) * K);
  for (auto& x : a) x = __float2bfloat16(r.uni(-1.0f, 1.0f));
  std::vector<float> tw8(slots);
  for (auto& x : tw8) x = r.uni(0.3f, 1.2f);
  std::vector<__nv_bfloat16> bias(static_cast<size_t>(kE) * N);
  for (auto& x : bias) x = __float2bfloat16(r.uni(-0.2f, 0.2f));
  Buf<__nv_bfloat16> da(a);
  Buf<float> dtw(tw8);
  Buf<__nv_bfloat16> db(bias);
  const int ldc = N + 4;
  Buf<__nv_bfloat16> dcb(static_cast<size_t>(cap) * ldc);
  cudaMemset(dcb.p, 0xCD, dcb.n * 2);
  Buf<int32_t> dsorted(R.sorted), deids(R.expert_ids), dpost(1);
  std::vector<int32_t> post = {R.post_pad};
  kops_test::copy_in(dpost.p, post);

  kops::moe::FusedMoEGemmParams p{};
  p.a = da.p;
  p.lda = K;
  p.a_rows_are_slots = true;
  p.w = dc.p;
  p.w_expert_stride = static_cast<int64_t>(N) * (K / 4);
  p.w_scale = ds.p;
  p.w_scale_expert_stride = static_cast<int64_t>(N) * (K / G);
  p.bias = db.p;
  p.c = dcb.p;
  p.ldc = ldc;
  p.sorted_token_ids = dsorted.p;
  p.expert_ids = deids.p;
  p.num_tokens_post_pad = dpost.p;
  p.topk_weights = dtw.p;
  p.mul_routed_weight = true;
  p.top_k = kTopK;
  p.n = N;
  p.k = K;
  p.num_valid_slots = slots;
  p.w_quant.packing = kops::WeightPacking::kW2;
  p.w_quant.granularity = kops::ScaleGranularity::kPerGroup;
  p.w_quant.group_k = G;
  KOPS_EXPECT_OK(kops::experimental::moe_w2::gemm(p, kE, kBlock, DType::kBFloat16,
                                                  DType::kBFloat16, ctx),
                 "w2 gemm bf16");
  cudaDeviceSynchronize();
  const auto got = dcb.get();

  std::vector<double> want(static_cast<size_t>(cap) * ldc, 0.0);
  for (int b = 0; b < R.post_pad / kBlock; ++b) {
    const int e = R.expert_ids[static_cast<size_t>(b)];
    if (e < 0) continue;
    for (int m = 0; m < kBlock; ++m) {
      const int slot = b * kBlock + m;
      if (slot >= R.post_pad) continue;
      const int tok = R.sorted[static_cast<size_t>(slot)];
      if (tok >= slots) continue;
      for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          const uint8_t code = ref.codes[(static_cast<size_t>(e) * N + n) * (K / 4) + k / 4] >> (2 * (k & 3)) & 3;
          const float s = ref.scales[(static_cast<size_t>(e) * N + n) * (K / G) + k / G];
          const double av = static_cast<double>(a[static_cast<size_t>(tok) * K + k]);
          acc += av * static_cast<double>(w2_level_host(code) * s);
        }
        acc += static_cast<double>(__bfloat162float(bias[static_cast<size_t>(e) * N + n]));
        acc *= static_cast<double>(tw8[static_cast<size_t>(tok)]);
        want[static_cast<size_t>(slot) * ldc + n] = acc;
      }
    }
  }
  std::vector<double> flat, wantv;
  for (int slot = 0; slot < cap; ++slot) {
    const bool valid = slot < R.post_pad && R.sorted[static_cast<size_t>(slot)] < slots &&
                       R.expert_ids[static_cast<size_t>(slot / kBlock)] >= 0;
    if (!valid) continue;
    for (int n = 0; n < N; ++n) {
      flat.push_back(static_cast<double>(__bfloat162float(got[static_cast<size_t>(slot) * ldc + n])));
      wantv.push_back(want[static_cast<size_t>(slot) * ldc + n]);
    }
  }
  close(flat, wantv, 1e-2, "w2 gemm bf16 oracle");

  // Padding slots, padding columns and rejected expert blocks keep their sentinel bytes.
  int touched = 0;
  for (int slot = 0; slot < cap; ++slot) {
    const bool valid = slot < R.post_pad && R.sorted[static_cast<size_t>(slot)] < slots &&
                       R.expert_ids[static_cast<size_t>(slot / kBlock)] >= 0;
    for (int n = 0; n < ldc; ++n) {
      if (n < N && valid) continue;
      touched += __bfloat16_as_ushort(got[static_cast<size_t>(slot) * ldc + n]) != 0xCDCDu ? 1 : 0;
    }
  }
  if (touched != 0) kops_test::record_failure("w2 gemm padding", "wrote a padding row", __FILE__, __LINE__);

  // fp16 A / fp32 C and W13-style A rows (slot / top_k).
  {
    std::vector<__half> ah(static_cast<size_t>(kTokens) * K);
    for (auto& x : ah) x = __float2half_rn(r.uni(-1.0f, 1.0f));
    Buf<__half> dah(ah);
    Buf<float> dcf(static_cast<size_t>(cap) * ldc);
    cudaMemset(dcf.p, 0, dcf.n * 4);
    kops::moe::FusedMoEGemmParams q = p;
    q.a = dah.p;
    q.a_rows_are_slots = false;
    q.c = dcf.p;
    q.bias = nullptr;
    q.mul_routed_weight = false;
    KOPS_EXPECT_OK(kops::experimental::moe_w2::gemm(q, kE, kBlock, DType::kFloat16, DType::kFloat32, ctx),
                   "w2 gemm fp16/fp32");
    cudaDeviceSynchronize();
    const auto gotf = dcf.get();
    std::vector<double> wantf(static_cast<size_t>(cap) * ldc, 0.0);
    for (int b = 0; b < R.post_pad / kBlock; ++b) {
      const int e = R.expert_ids[static_cast<size_t>(b)];
      if (e < 0) continue;
      for (int m = 0; m < kBlock; ++m) {
        const int slot = b * kBlock + m;
        if (slot >= R.post_pad) continue;
        const int tok = R.sorted[static_cast<size_t>(slot)];
        if (tok >= slots) continue;
        const int row = tok / kTopK;
        for (int n = 0; n < N; ++n) {
          double acc = 0.0;
          for (int k = 0; k < K; ++k) {
            const uint8_t code = ref.codes[(static_cast<size_t>(e) * N + n) * (K / 4) + k / 4] >> (2 * (k & 3)) & 3;
            const float s = ref.scales[(static_cast<size_t>(e) * N + n) * (K / G) + k / G];
            acc += static_cast<double>(__half2float(ah[static_cast<size_t>(row) * K + k])) * static_cast<double>(w2_level_host(code) * s);
          }
          wantf[static_cast<size_t>(slot) * ldc + n] = acc;
        }
      }
    }
    std::vector<double> gotfd, wantfv;
    for (int slot = 0; slot < cap; ++slot) {
      const bool valid = slot < R.post_pad && R.sorted[static_cast<size_t>(slot)] < slots &&
                         R.expert_ids[static_cast<size_t>(slot / kBlock)] >= 0;
      if (!valid) continue;
      for (int n = 0; n < N; ++n) {
        gotfd.push_back(static_cast<double>(gotf[static_cast<size_t>(slot) * ldc + n]));
        wantfv.push_back(wantf[static_cast<size_t>(slot) * ldc + n]);
      }
    }
    close(gotfd, wantfv, 2e-3, "w2 gemm fp16/fp32 oracle");
  }

  // Group 16 / 64 take the scalar path; check the first valid slots against the same oracle.
  for (int G2 : {16, 64}) {
    const W2Pack r2 = w2_pack_host(w, kE, N, K, G2);
    Buf<uint8_t> c2(r2.codes);
    Buf<float> s2(r2.scales);
    Tensor tc2 = kops_test::tensor3d(c2.p, DType::kUInt8, kE, N, K / 4);
    Tensor ts2 = kops_test::tensor3d(s2.p, DType::kFloat32, kE, N, K / G2);
    KOPS_EXPECT_OK(kops::experimental::moe_w2::pack(tc2, ts2, tw, kops::WeightPacking::kNone, G2, ctx),
                   "w2 gemm pack group");
    Buf<__nv_bfloat16> c2out(static_cast<size_t>(cap) * ldc);
    cudaMemset(c2out.p, 0xCD, c2out.n * 2);
    kops::moe::FusedMoEGemmParams q2 = p;
    q2.w = c2.p;
    q2.w_scale = s2.p;
    q2.w_scale_expert_stride = static_cast<int64_t>(N) * (K / G2);
    q2.w_quant.group_k = G2;
    q2.bias = nullptr;
    q2.mul_routed_weight = false;
    q2.c = c2out.p;
    KOPS_EXPECT_OK(kops::experimental::moe_w2::gemm(q2, kE, kBlock, DType::kBFloat16,
                                                    DType::kBFloat16, ctx),
                   "w2 gemm group");
    cudaDeviceSynchronize();
    const auto got2 = c2out.get();
    int checked = 0;
    for (int b = 0; b < R.post_pad / kBlock && checked < 8; ++b) {
      const int e = R.expert_ids[static_cast<size_t>(b)];
      if (e < 0) continue;
      for (int m = 0; m < kBlock && checked < 8; ++m) {
        const int slot = b * kBlock + m;
        const int tok = R.sorted[static_cast<size_t>(slot)];
        if (tok >= slots) continue;
        ++checked;
        for (int n = 0; n < N; ++n) {
          double acc = 0.0;
          for (int k = 0; k < K; ++k) {
            const uint8_t code = r2.codes[(static_cast<size_t>(e) * N + n) * (K / 4) + k / 4] >> (2 * (k & 3)) & 3;
            const float sc = r2.scales[(static_cast<size_t>(e) * N + n) * (K / G2) + k / G2];
            acc += static_cast<double>(__bfloat162float(a[static_cast<size_t>(tok) * K + k])) *
                   static_cast<double>(w2_level_host(code) * sc);
          }
          const double g = static_cast<double>(__bfloat162float(got2[static_cast<size_t>(slot) * ldc + n]));
          if (std::fabs(g - acc) > 1e-2 * std::fabs(acc) + 2e-5) {
            char det[96];
            std::snprintf(det, sizeof(det), "G=%d slot %d n %d got %.5f want %.5f", G2, slot, n, g, acc);
            kops_test::record_failure("w2 gemm group oracle", det, __FILE__, __LINE__);
            break;
          }
        }
      }
    }
  }

  // Contract errors.
  p.lda = K;
  auto expect = [&](const kops::moe::FusedMoEGemmParams& q, int block, DType ad, DType cd, Status want,
                    const char* what) {
    KOPS_EXPECT_STATUS(kops::experimental::moe_w2::gemm(q, kE, block, ad, cd, ctx), want, what);
    cudaDeviceSynchronize();
  };
  expect(p, 8, DType::kBFloat16, DType::kBFloat16, Status::kUnsupportedShape, "w2 gemm block 8");
  auto q = p;
  q.w_quant.packing = kops::WeightPacking::kNone;
  expect(q, kBlock, DType::kBFloat16, DType::kBFloat16, Status::kInvalidArgument, "w2 gemm packing");
  q = p;
  q.w_expert_stride = 10;
  expect(q, kBlock, DType::kBFloat16, DType::kBFloat16, Status::kInvalidArgument, "w2 gemm w stride");
  q = p;
  q.a_scale = ds.p;
  expect(q, kBlock, DType::kBFloat16, DType::kBFloat16, Status::kNotImplemented, "w2 gemm act scales");
  q = p;
  q.w_zeros = ds.p;
  expect(q, kBlock, DType::kBFloat16, DType::kBFloat16, Status::kNotImplemented, "w2 gemm zeros");
  q = p;
  q.w_quant.group_k = 24;
  expect(q, kBlock, DType::kBFloat16, DType::kBFloat16, Status::kUnsupportedShape, "w2 gemm group 24");
  expect(p, kBlock, DType::kBFloat16, DType::kInt8, Status::kUnsupportedDtype, "w2 gemm c dtype");
  q = p;
  q.sorted_token_ids = nullptr;
  expect(q, kBlock, DType::kBFloat16, DType::kBFloat16, Status::kInvalidArgument, "w2 gemm null ids");
}


}  // namespace b7moe

void test_b7_moe_gemm() {
  using namespace b7moe;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  for (int block : {16, 32, 64, 128}) {
    const Routing R = route(ctx, block, true, 100u + block);
    if (R.post_pad <= 0) continue;
    for (int wmode = W_BF16; wmode <= W_INT8; ++wmode) {
      for (bool w2 : {false, true}) {
        Case cs;
        cs.wmode = wmode; cs.block = block; cs.w2 = w2; cs.seed = static_cast<unsigned>(block);
        const Result r1 = run(ctx, cs, R, "b7 moe_gemm");
        const Result r2 = run(ctx, cs, R, "b7 moe_gemm (repeat)");
        if (r1.ok && r2.ok && r1.c != r2.c) kops_test::record_failure("b7 moe_gemm determinism", "repeat run differs", __FILE__, __LINE__);
      }
    }
    // GLU interleave 16 / 64, fp16, fp32 output, no EP drop
    Case cs;
    cs.block = block; cs.seed = 3;
    for (int db : {16, 64}) {
      cs.glu_db = db;
      run(ctx, cs, R, "b7 moe_gemm GLU interleave");
      cs.wmode = W_FP8_BLOCK;
      run(ctx, cs, R, "b7 moe_gemm GLU interleave");
      cs.wmode = W_BF16;
    }
    cs.glu_db = 1;
    cs.fp16 = true;
    run(ctx, cs, R, "b7 moe_gemm fp16");
    cs.w2 = true;
    run(ctx, cs, R, "b7 moe_gemm fp16");
    cs.fp16 = false;
    cs.fp32_out = true;
    run(ctx, cs, R, "b7 moe_gemm fp32 out");
    cs.wmode = W_FP8_CHANNEL;
    run(ctx, cs, R, "b7 moe_gemm fp32 out");
  }
  // bf16 against moe_gemv (same slots, rows independent of the block layout): W13 without GLU and W2
  for (bool w2 : {false, true}) {
    const Routing R16 = route(ctx, 16, false, 555u);
    const Routing R8 = route(ctx, 8, false, 555u);
    Case cs;
    cs.block = 16; cs.w2 = w2; cs.glu = false; cs.drop_expert = false; cs.seed = 77;
    const Result g = run(ctx, cs, R16, "b7 moe_gemm vs moe_gemv: moe_gemm");
    cs.block = 8;
    const Result v = run(ctx, cs, R8, "b7 moe_gemm vs moe_gemv: moe_gemv", true);
    if (g.ok && v.ok) {
      size_t far = 0;
      for (size_t i = 0; i + 1 < g.c.size(); i += 2) {
        __nv_bfloat16 a, b;
        std::memcpy(&a, &g.c[i], 2);
        std::memcpy(&b, &v.c[i], 2);
        const double x = kops_test::to_float(a), y = kops_test::to_float(b);
        if (!(std::fabs(x - y) <= 2.0 * 0.0078125 * std::max(std::fabs(x), std::fabs(y)) + 1e-3)) ++far;
      }
      if (far != 0) kops_test::record_failure("b7 moe_gemm vs moe_gemv", "outputs differ by more than two bf16 ulps", __FILE__, __LINE__);
    }
  }
  // contracts: Status, C untouched
  {
    const Routing R = route(ctx, 16, true, 9u);
    Buf<__nv_bfloat16> a(static_cast<size_t>(kTokens) * 256), w(static_cast<size_t>(kE) * 256 * 256);
    Buf<float> sc(static_cast<size_t>(kE) * 256), tw(kSlots);
    Buf<int32_t> dsorted(R.sorted), deids(R.expert_ids), dpost(std::vector<int32_t>(1, R.post_pad));
    std::vector<uint8_t> c0(static_cast<size_t>(kSlots) * 128 * 2, 0x3C);
    Buf<uint8_t> c(c0);
    kops::moe::FusedMoEGemmParams base{};
    base.a = a.p; base.lda = 256; base.w = w.p; base.w_expert_stride = 256 * 256; base.c = c.p; base.ldc = 128;
    base.sorted_token_ids = dsorted.p; base.expert_ids = deids.p; base.num_tokens_post_pad = dpost.p;
    base.topk_weights = tw.p; base.top_k = kTopK; base.n = 256; base.k = 256; base.num_valid_slots = kSlots;
    base.epi.ops = kops::gemm::kEpiGLU; base.epi.glu_layout = kops::GateLayout::kInterleaved; base.epi.act = kops::ActFn::kSiLU;
    const DType bf = DType::kBFloat16, e4 = DType::kFloat8E4M3;
    auto expect = [&](const kops::moe::FusedMoEGemmParams& p, int block, DType ad, DType wd, DType cd, Status want, const char* what) {
      KOPS_EXPECT_STATUS(kops::moe::moe_gemm(p, kE, block, ad, wd, cd, ctx), want, what);
      cudaDeviceSynchronize();
      if (c.get() != c0) kops_test::record_failure(what, "C written by a rejected call", __FILE__, __LINE__);
    };
    expect(base, 24, bf, bf, bf, Status::kUnsupportedShape, "b7 moe_gemm: block 24");
    auto p = base; p.sorted_token_ids = nullptr;
    expect(p, 16, bf, bf, bf, Status::kInvalidArgument, "b7 moe_gemm: null sorted ids");
    p = base; p.mul_routed_weight = true; p.topk_weights = nullptr;
    expect(p, 16, bf, bf, bf, Status::kInvalidArgument, "b7 moe_gemm: routed weight without weights");
    p = base; p.w_zeros = w.p;
    expect(p, 16, bf, bf, bf, Status::kNotImplemented, "b7 moe_gemm: zero points");
    expect(base, 16, bf, DType::kFloat16, bf, Status::kUnsupportedDtype, "b7 moe_gemm: mixed 16-bit dtypes");
    expect(base, 16, bf, e4, bf, Status::kUnsupportedDtype, "b7 moe_gemm: bf16 x e4m3");
    expect(base, 16, bf, bf, DType::kInt8, Status::kUnsupportedDtype, "b7 moe_gemm: int8 output");
    p = base; p.w_quant.granularity = kops::ScaleGranularity::kPerChannel;
    expect(p, 16, bf, bf, bf, Status::kInvalidArgument, "b7 moe_gemm: 16-bit experts with scales");
    expect(base, 16, e4, e4, bf, Status::kInvalidArgument, "b7 moe_gemm: e4m3 without scales");
    p = base; p.w_scale = sc.p; p.w_quant.granularity = kops::ScaleGranularity::kPerChannel; p.w_scale_expert_stride = 100;
    expect(p, 16, e4, e4, bf, Status::kInvalidArgument, "b7 moe_gemm: per-channel stride < N");
    p.w_quant.granularity = kops::ScaleGranularity::kPerBlock2D; p.w_quant.group_m = 128; p.w_quant.group_k = 128;
    expect(p, 16, e4, e4, bf, Status::kInvalidArgument, "b7 moe_gemm: blockwise without activation scales");
    p.w_quant.group_k = 64; p.a_scale = sc.p;
    expect(p, 16, e4, e4, bf, Status::kUnsupportedShape, "b7 moe_gemm: 128x64 blocks");
    p = base; p.w_scale = sc.p;
    expect(p, 16, DType::kInt8, DType::kInt8, bf, Status::kInvalidArgument, "b7 moe_gemm: int8 without per-channel scales");
    p = base; p.ldc = 100;
    expect(p, 16, bf, bf, bf, Status::kInvalidArgument, "b7 moe_gemm: ldc < N / 2");
    p = base; p.epi.glu_layout = kops::GateLayout::kGateUp;
    expect(p, 16, bf, bf, bf, Status::kNotImplemented, "b7 moe_gemm: GLU on split gate / up");
    p = base; p.lda = 260;
    expect(p, 16, bf, bf, bf, Status::kMisaligned, "b7 moe_gemm: lda not 16 B");
  }
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;
  if (std::getenv("KOPS_TEST_ONLY_W2") != nullptr) {
    b7moe::test_moe_w2_pack(probe);
    b7moe::test_moe_w2_gemm(probe);
    return kops_test::finish("test_moe --w2");
  }
  test_route();
  test_route_grouped();
  test_route_ties();
  test_route_eplb_shared();
  test_route_e256_fast();
  test_route_e256_ties();
  test_route_e256_paths();
  test_route_fused();
  test_hash_route();
  test_align_and_offsets();
  test_large_sorts();
  test_moe_sum_bf16();
  test_permute_unpermute();
  test_permute_quant();
  test_moe_sum_and_latent();
  test_contracts();
  test_b7_moe_gemm();
  b7moe::test_moe_w2_pack(probe);
  b7moe::test_moe_w2_gemm(probe);
  return kops_test::finish("test_moe");
}
