// =============================================================================
// kernel-ops | tests/unit/test_activation.cc
// B1 activation tests: act/act_and_mul/gate_mul, including broadcast widths.
// =============================================================================
#include "tests/unit/harness.h"

#include <cmath>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/common/tensor.h"
#include "csrc/activation/activation_kernels.cuh"
#include "csrc/activation/ops.h"

namespace {

void fill(std::vector<float>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-2.0f, 2.0f);
  for (float& x : v) x = d(rng);
}

template <typename T>
std::vector<T> to_type(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(out[i], src[i]);
  return out;
}

template <typename T>
std::vector<double> to_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = kops_test::to_float(src[i]);
  return out;
}

template <typename T>
void test_gate_mul(kops::DType dt, int gate_width, bool accumulate, const char* name) {
  const int64_t rows = 7;
  const int64_t d = 96;
  std::vector<float> xf(rows * d), gf(rows * gate_width), af(rows * d);
  fill(xf, 1u);
  fill(gf, 2u);
  fill(af, 3u);
  const auto x = to_type<T>(xf), g = to_type<T>(gf), a = to_type<T>(af);
  T* dx = kops_test::device_alloc<T>(x.size());
  T* dg = kops_test::device_alloc<T>(g.size());
  T* da = accumulate ? kops_test::device_alloc<T>(a.size()) : nullptr;
  T* dy = kops_test::device_alloc<T>(x.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(dx, x);
  kops_test::copy_in(dg, g);
  if (accumulate) kops_test::copy_in(da, a);
  const kops::Tensor tx = kops_test::tensor2d(dx, dt, rows, d);
  const kops::Tensor tg = kops_test::tensor2d(dg, dt, rows, gate_width);
  const kops::Tensor ta = accumulate ? kops_test::tensor2d(da, dt, rows, d) : kops::Tensor{};
  const kops::Tensor ty = kops_test::tensor2d(dy, dt, rows, d);
  KOPS_EXPECT_OK(kops::act::gate_mul(ty, tx, tg, ta, kops::ActFn::kSigmoid, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(dy, x.size());
  const auto xd = to_double(x), gd = to_double(g), ad = to_double(a);
  std::vector<double> ref(x.size());
  const int group = static_cast<int>(d / gate_width);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t i = 0; i < d; ++i) {
      const double gv = gd[static_cast<size_t>(r * gate_width + (gate_width == 1 ? 0 : i / group))];
      const double s = 1.0 / (1.0 + std::exp(-gv));
      ref[static_cast<size_t>(r * d + i)] = xd[static_cast<size_t>(r * d + i)] * s +
                                            (accumulate ? ad[static_cast<size_t>(r * d + i)] : 0.0);
    }
  }
  const double tol = dt == kops::DType::kFloat32 ? 1e-5 : 2e-2;
  kops_test::compare(got, ref, tol, tol, name);
  kops_test::device_free(dx);
  kops_test::device_free(dg);
  kops_test::device_free(da);
  kops_test::device_free(dy);
}

void test_all_act_dispatch() {
  const int64_t rows = 2, cols = 64;
  std::vector<float> xf(rows * cols), gf(rows * 2 * cols);
  fill(xf, 9u);
  fill(gf, 10u);
  const auto x = to_type<__nv_bfloat16>(xf);
  const auto g = to_type<__nv_bfloat16>(gf);
  __nv_bfloat16* dx = kops_test::device_alloc<__nv_bfloat16>(x.size());
  __nv_bfloat16* dg = kops_test::device_alloc<__nv_bfloat16>(g.size());
  __nv_bfloat16* dy = kops_test::device_alloc<__nv_bfloat16>(x.size());
  __nv_bfloat16* dz = kops_test::device_alloc<__nv_bfloat16>(x.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(dx, x);
  kops_test::copy_in(dg, g);
  const kops::Tensor tx = kops_test::tensor2d(dx, kops::DType::kBFloat16, rows, cols);
  const kops::Tensor tg = kops_test::tensor2d(dg, kops::DType::kBFloat16, rows, 2 * cols);
  const kops::Tensor ty = kops_test::tensor2d(dy, kops::DType::kBFloat16, rows, cols);
  const kops::Tensor tz = kops_test::tensor2d(dz, kops::DType::kBFloat16, rows, cols);
  const kops::GluParams p{};
  const kops::ActFn fns[] = {kops::ActFn::kIdentity, kops::ActFn::kSiLU, kops::ActFn::kGeLUErf,
                             kops::ActFn::kGeLUTanh, kops::ActFn::kQuickGeLU, kops::ActFn::kReLU,
                             kops::ActFn::kReLU2, kops::ActFn::kSigmoid, kops::ActFn::kTanh,
                             kops::ActFn::kSoftplus, kops::ActFn::kSiTU, kops::ActFn::kFatReLU};
  for (kops::ActFn fn : fns) {
    KOPS_EXPECT_OK(kops::act::activation(ty, tx, fn, p, ctx), "activation all ActFn");
    KOPS_EXPECT_OK(kops::act::act_and_mul(tz, tg, kops::Tensor{}, fn, kops::GateLayout::kGateUp,
                                          p, ctx),
                   "act_and_mul all ActFn");
  }
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  kops_test::device_free(dx);
  kops_test::device_free(dg);
  kops_test::device_free(dy);
  kops_test::device_free(dz);
}

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  std::vector<float> v(4 * 16, 1.0f);
  float* d = kops_test::device_alloc<float>(v.size());
  kops_test::copy_in(d, v);
  kops::Tensor x = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 16);
  kops::Tensor bad_gate = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 5);
  KOPS_EXPECT_STATUS(kops::act::gate_mul(x, x, bad_gate, kops::Tensor{}, kops::ActFn::kSigmoid, ctx),
                     kops::Status::kUnsupportedShape, "gate_mul width");
  kops::Tensor h = kops_test::tensor2d(d, kops::DType::kFloat16, 4, 16);
  KOPS_EXPECT_STATUS(kops::act::gate_mul(x, x, h, kops::Tensor{}, kops::ActFn::kSigmoid, ctx),
                     kops::Status::kUnsupportedDtype, "gate_mul dtype");
  kops_test::device_free(d);
}

// -----------------------------------------------------------------------------
// B3 row 5: act_and_mul_quant
// -----------------------------------------------------------------------------

float aq_e4m3_value(uint8_t b) {
  return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

void test_act_and_mul_quant() {
  const int rows = 9, d = 512, group = 128, groups = d / group;
  __nv_bfloat16* in = kops_test::device_alloc<__nv_bfloat16>(static_cast<size_t>(rows) * 2 * d);
  uint8_t* q = kops_test::device_alloc<uint8_t>(static_cast<size_t>(rows) * d);
  float* scales = kops_test::device_alloc<float>(static_cast<size_t>(rows) * groups);
  if (in == nullptr || q == nullptr || scales == nullptr) return;
  std::vector<float> host(static_cast<size_t>(rows) * 2 * d);
  fill(host, 401);
  const auto h = to_type<__nv_bfloat16>(host);
  kops_test::copy_in(in, h);
  cudaMemset(scales, 0, static_cast<size_t>(rows) * groups * sizeof(float));

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  for (int layout_i = 0; layout_i < 3; ++layout_i) {
    const kops::GateLayout layout =
        layout_i == 0 ? kops::GateLayout::kGateUp
                      : (layout_i == 1 ? kops::GateLayout::kUpGate : kops::GateLayout::kInterleaved);
    for (int scale_mode = 0; scale_mode < 2; ++scale_mode) {
      cudaMemset(scales, 0, static_cast<size_t>(rows) * groups * sizeof(float));
      kops::act::ActMulQuantParams p{};
      p.input = in;
      p.input_stride = 2 * d;
      p.glu = kops::GluParams{};
      p.num_rows = rows;
      p.d = d;
      p.quant.q = q;
      p.quant.q_row_stride = d;
      p.quant.scales = scales;
      p.quant.scale_row_stride = scale_mode == 0 ? 1 : groups;
      p.quant.q_dtype = kops::DType::kFloat8E4M3;
      p.quant.group_size = group;
      p.quant.mode = scale_mode == 0 ? kops::ActQuantMode::kDynamicPerToken
                                     : kops::ActQuantMode::kPerGroup;
      KOPS_EXPECT_OK(kops::act::act_and_mul_quant(p, kops::ActFn::kSiLU, layout,
                                                  kops::DType::kBFloat16, ctx),
                     "act_and_mul_quant");
      const auto got = kops_test::copy_out(q, static_cast<size_t>(rows) * d);
      const auto gs = kops_test::copy_out(scales, static_cast<size_t>(rows) * groups);
      for (int r = 0; r < rows; ++r) {
        std::vector<float> y(static_cast<size_t>(d));
        for (int c = 0; c < d; ++c) {
          float gate, up;
          const size_t base = static_cast<size_t>(r) * 2 * d;
          if (layout == kops::GateLayout::kGateUp) {
            gate = kops_test::to_float(h[base + c]);
            up = kops_test::to_float(h[base + d + c]);
          } else if (layout == kops::GateLayout::kUpGate) {
            gate = kops_test::to_float(h[base + d + c]);
            up = kops_test::to_float(h[base + c]);
          } else {
            gate = kops_test::to_float(h[base + 2 * c]);
            up = kops_test::to_float(h[base + 2 * c + 1]);
          }
          y[static_cast<size_t>(c)] = (gate / (1.0f + std::exp(-gate))) * up;
        }
        if (scale_mode == 0) {
          float amax = 0.0f;
          for (float v : y) amax = std::max(amax, std::fabs(v));
          const float s = std::max(amax / 448.0f, 1e-10f);
          KOPS_TEST_CHECK(std::fabs(gs[static_cast<size_t>(r)] - s) <= 1e-6f * s + 1e-8f);
          for (int c = 0; c < d; ++c) {
            const float qv = aq_e4m3_value(got[static_cast<size_t>(r * d + c)]) * s;
            KOPS_TEST_CHECK(std::fabs(qv - y[static_cast<size_t>(c)]) <=
                            0.07f * std::fabs(y[static_cast<size_t>(c)]) + 1e-4f);
          }
        } else {
          for (int g = 0; g < groups; ++g) {
            float amax = 0.0f;
            for (int j = 0; j < group; ++j) {
              amax = std::max(amax, std::fabs(y[static_cast<size_t>(g * group + j)]));
            }
            const float s = std::max(amax / 448.0f, 1e-10f);
            KOPS_TEST_CHECK(std::fabs(gs[static_cast<size_t>(r * groups + g)] - s) <=
                            1e-6f * s + 1e-8f);
            for (int j = 0; j < group; ++j) {
              const int c = g * group + j;
              const float qv = aq_e4m3_value(got[static_cast<size_t>(r * d + c)]) * s;
              KOPS_TEST_CHECK(std::fabs(qv - y[static_cast<size_t>(c)]) <=
                              0.07f * std::fabs(y[static_cast<size_t>(c)]) + 1e-4f);
            }
          }
        }
      }
    }
  }

  // Masked (DeepEP) layout: dead rows are skipped and their scales stay zero.
  const int experts = 3, max_m = 4;
  __nv_bfloat16* m_in =
      kops_test::device_alloc<__nv_bfloat16>(static_cast<size_t>(experts) * max_m * 2 * d);
  uint8_t* m_q = kops_test::device_alloc<uint8_t>(static_cast<size_t>(experts) * max_m * d);
  float* m_s = kops_test::device_alloc<float>(static_cast<size_t>(experts) * max_m * groups);
  int32_t* m_m = kops_test::device_alloc<int32_t>(experts);
  if (m_in == nullptr || m_q == nullptr || m_s == nullptr || m_m == nullptr) return;
  std::vector<float> mh(static_cast<size_t>(experts) * max_m * 2 * d);
  fill(mh, 402);
  const auto mhh = to_type<__nv_bfloat16>(mh);
  kops_test::copy_in(m_in, mhh);
  kops_test::copy_in(m_m, std::vector<int32_t>{4, 2, 3});
  cudaMemset(m_q, 0, static_cast<size_t>(experts) * max_m * d);
  cudaMemset(m_s, 0, static_cast<size_t>(experts) * max_m * groups * sizeof(float));
  kops::act::ActMulQuantParams mp{};
  mp.input = m_in;
  mp.input_stride = 2 * d;
  mp.glu = kops::GluParams{};
  mp.masked_m = m_m;
  mp.max_m = max_m;
  mp.num_experts = experts;
  mp.num_rows = experts * max_m;
  mp.d = d;
  mp.quant.q = m_q;
  mp.quant.q_row_stride = d;
  mp.quant.scales = m_s;
  mp.quant.scale_row_stride = groups;
  mp.quant.group_size = group;
  mp.quant.q_dtype = kops::DType::kFloat8E4M3;
  mp.quant.mode = kops::ActQuantMode::kPerGroup;
  KOPS_EXPECT_OK(kops::act::act_and_mul_quant(mp, kops::ActFn::kSiLU, kops::GateLayout::kGateUp,
                                              kops::DType::kBFloat16, ctx),
                 "act_and_mul_quant masked");
  const auto ms = kops_test::copy_out(m_s, static_cast<size_t>(experts) * max_m * groups);
  const auto mq = kops_test::copy_out(m_q, static_cast<size_t>(experts) * max_m * d);
  const int dead_row = 1 * max_m + 2;
  for (int g = 0; g < groups; ++g) {
    KOPS_TEST_CHECK(ms[static_cast<size_t>(dead_row * groups + g)] == 0.0f);
  }
  for (int c = 0; c < d; ++c) {
    KOPS_TEST_CHECK(mq[static_cast<size_t>(dead_row * d + c)] == 0u);
  }
  cudaFree(in);
  cudaFree(q);
  cudaFree(scales);
  cudaFree(m_in);
  cudaFree(m_q);
  cudaFree(m_s);
  cudaFree(m_m);
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;
  test_gate_mul<float>(kops::DType::kFloat32, 96, false, "gate_mul fp32 width=d");
  test_gate_mul<__nv_bfloat16>(kops::DType::kBFloat16, 96, false, "gate_mul bf16 width=d");
  test_gate_mul<__nv_bfloat16>(kops::DType::kBFloat16, 8, false, "gate_mul bf16 width=heads");
  test_gate_mul<float>(kops::DType::kFloat32, 8, true, "gate_mul fp32 heads+acc");
  test_gate_mul<__nv_bfloat16>(kops::DType::kBFloat16, 1, true, "gate_mul bf16 width=1+acc");
  test_all_act_dispatch();
  test_contracts();
  test_act_and_mul_quant();
  return kops_test::finish("test_activation");
}
