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
  const int64_t stride = s.experts + 3;
  b.logits.assign(static_cast<size_t>(tokens * stride), 0.0f);
  std::vector<float> dense(static_cast<size_t>(tokens * s.experts));
  fill(dense, seed);
  for (int t = 0; t < tokens; ++t) {
    for (int e = 0; e < s.experts; ++e) {
      b.logits[static_cast<size_t>(t * stride + e)] = dense[static_cast<size_t>(t * s.experts + e)];
    }
  }
  b.bias.assign(static_cast<size_t>(s.experts), 0.0f);
  if (s.score != static_cast<int>(kops::moe::RouterScore::kSigmoid) &&
      s.score != static_cast<int>(kops::moe::RouterScore::kSqrtSoftplus)) {
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

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;
  test_route();
  test_route_grouped();
  test_route_ties();
  test_route_eplb_shared();
  test_route_fused();
  test_hash_route();
  test_align_and_offsets();
  test_permute_unpermute();
  test_permute_quant();
  test_moe_sum_and_latent();
  test_contracts();
  return kops_test::finish("test_moe");
}
