// =============================================================================
// kernel-ops | tests/unit/test_m1.cc
// Walking-skeleton test for the five kernels that have bodies in M1:
//   rms_norm, fused_add_rms_norm, act, act_and_mul, merge_state.
// Oracle: csrc/reference/ (fp64). Contract cases: dtype / shape / stride /
// alignment / NotImplemented. No Python, no framework.
// =============================================================================

#include "tests/unit/harness.h"

#include <random>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"

#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"
#include "csrc/activation/activation_kernels.cuh"
#include "csrc/activation/ops.h"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/merge_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"
#include "csrc/reference/ops.h"

namespace {

constexpr double kEps = 1e-5;

void fill(std::vector<float>& values, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : values) v = dist(rng);
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

template <typename T>
void test_rms_norm(const char* name, kops::DType dtype, double atol, double rtol) {
  const int64_t rows = 8;
  const int64_t hidden = 256;
  const float weight_offset = 1.0f;  // Gemma-style (w + 1)

  std::vector<float> in_f(rows * hidden), w_f(hidden);
  fill(in_f, 1u);
  fill(w_f, 2u);
  const auto in = cast_from_float<T>(in_f);
  const auto w = cast_from_float<T>(w_f);

  T* d_in = kops_test::device_alloc<T>(in.size());
  T* d_w = kops_test::device_alloc<T>(w.size());
  T* d_out = kops_test::device_alloc<T>(in.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_in, in);
  kops_test::copy_in(d_w, w);
  kops::Tensor t_in = kops_test::tensor2d(d_in, dtype, rows, hidden);
  kops::Tensor t_w = kops_test::tensor2d(d_w, dtype, 1, hidden);
  t_w.ndim = 1;
  t_w.shape[0] = hidden;
  t_w.shape[1] = 0;
  t_w.stride[0] = 1;
  kops::Tensor t_out = kops_test::tensor2d(d_out, dtype, rows, hidden);

  KOPS_EXPECT_OK(kops::norm::rms_norm(t_out, t_in, t_w, static_cast<float>(kEps), weight_offset, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto out = kops_test::copy_out(d_out, static_cast<size_t>(rows * hidden));
  const auto in_d = cast_to_double(in);
  const auto w_d = cast_to_double(w);
  std::vector<double> ref(static_cast<size_t>(rows * hidden));
  KOPS_EXPECT_OK(kops::ref::rms_norm_f64(in_d.data(), hidden, w_d.data(), weight_offset, kEps, rows,
                                         hidden, ref.data(), hidden),
                 "rms_norm_f64");
  kops_test::compare(out, ref, atol, rtol, name);

  kops_test::device_free(d_in);
  kops_test::device_free(d_w);
  kops_test::device_free(d_out);
}

void test_fused_add_rms_norm() {
  const int64_t rows = 8;
  const int64_t hidden = 256;
  const float input_scale = 0.5f;
  std::vector<float> in_f(rows * hidden), res_f(rows * hidden), w_f(hidden);
  fill(in_f, 3u);
  fill(res_f, 4u);
  fill(w_f, 5u);

  float* d_in = kops_test::device_alloc<float>(in_f.size());
  float* d_res = kops_test::device_alloc<float>(res_f.size());
  float* d_w = kops_test::device_alloc<float>(w_f.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_in, in_f);
  kops_test::copy_in(d_res, res_f);
  kops_test::copy_in(d_w, w_f);
  kops::Tensor t_in = kops_test::tensor2d(d_in, kops::DType::kFloat32, rows, hidden);
  kops::Tensor t_res = kops_test::tensor2d(d_res, kops::DType::kFloat32, rows, hidden);
  kops::Tensor t_w = kops_test::tensor2d(d_w, kops::DType::kFloat32, 1, hidden);
  t_w.ndim = 1;
  t_w.shape[0] = hidden;
  t_w.shape[1] = 0;
  t_w.stride[0] = 1;

  KOPS_EXPECT_OK(
      kops::norm::fused_add_rms_norm(t_in, t_res, t_w, static_cast<float>(kEps), 0.0f, input_scale, ctx),
      "fused_add_rms_norm");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto out = kops_test::copy_out(d_in, in_f.size());
  const auto res = kops_test::copy_out(d_res, res_f.size());
  std::vector<double> res_ref(res_f.size());
  for (size_t i = 0; i < res_f.size(); ++i) {
    res_ref[i] = static_cast<double>(res_f[i]) + input_scale * static_cast<double>(in_f[i]);
  }
  std::vector<double> out_ref(in_f.size());
  const auto w_d = cast_to_double(w_f);
  KOPS_EXPECT_OK(kops::ref::rms_norm_f64(res_ref.data(), hidden, w_d.data(), 0.0, kEps, rows, hidden,
                                         out_ref.data(), hidden),
                 "fused_add_rms_norm reference");
  kops_test::compare(out, out_ref, 1e-4, 1e-4, "fused_add_rms_norm.norm");
  kops_test::compare(res, res_ref, 1e-4, 1e-4, "fused_add_rms_norm.residual");

  kops_test::device_free(d_in);
  kops_test::device_free(d_res);
  kops_test::device_free(d_w);
}

void test_activation() {
  const int64_t rows = 8;
  const int64_t cols = 256;
  std::vector<float> in_f(rows * cols);
  fill(in_f, 6u);
  const auto in = cast_from_float<__nv_bfloat16>(in_f);

  __nv_bfloat16* d_in = kops_test::device_alloc<__nv_bfloat16>(in.size());
  __nv_bfloat16* d_out = kops_test::device_alloc<__nv_bfloat16>(in.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_in, in);
  kops::Tensor t_in = kops_test::tensor2d(d_in, kops::DType::kBFloat16, rows, cols);
  kops::Tensor t_out = kops_test::tensor2d(d_out, kops::DType::kBFloat16, rows, cols);
  const kops::GluParams params{};
  KOPS_EXPECT_OK(kops::act::activation(t_out, t_in, kops::ActFn::kSiLU, params, ctx), "activation");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto out = kops_test::copy_out(d_out, in.size());
  const auto in_d = cast_to_double(in);
  std::vector<double> ref(in.size());
  KOPS_EXPECT_OK(kops::ref::act_f64(in_d.data(), static_cast<int64_t>(in_d.size()),
                                    kops::ActFn::kSiLU, params, ref.data()),
                 "act_f64");
  kops_test::compare(out, ref, 2e-2, 2e-2, "activation bf16");

  kops_test::device_free(d_in);
  kops_test::device_free(d_out);
}

void test_act_and_mul() {
  const int64_t rows = 4;
  const int64_t d = 128;
  std::vector<float> in_f(rows * 2 * d);
  fill(in_f, 7u);
  const auto in = cast_from_float<__nv_bfloat16>(in_f);

  __nv_bfloat16* d_in = kops_test::device_alloc<__nv_bfloat16>(in.size());
  __nv_bfloat16* d_out = kops_test::device_alloc<__nv_bfloat16>(rows * d);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_in, in);
  kops::Tensor t_in = kops_test::tensor2d(d_in, kops::DType::kBFloat16, rows, 2 * d);
  kops::Tensor t_out = kops_test::tensor2d(d_out, kops::DType::kBFloat16, rows, d);
  const kops::GluParams params{};
  KOPS_EXPECT_OK(kops::act::act_and_mul(t_out, t_in, kops::Tensor{}, kops::ActFn::kSiLU,
                                        kops::GateLayout::kGateUp, params, ctx),
                 "act_and_mul");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto out = kops_test::copy_out(d_out, static_cast<size_t>(rows * d));
  const auto in_d = cast_to_double(in);
  std::vector<double> ref(static_cast<size_t>(rows * d));
  KOPS_EXPECT_OK(kops::ref::act_and_mul_f64(in_d.data(), 2 * d, nullptr, params,
                                            kops::ActFn::kSiLU, kops::GateLayout::kGateUp, rows, d,
                                            ref.data()),
                 "act_and_mul_f64");
  kops_test::compare(out, ref, 2e-2, 2e-2, "act_and_mul bf16");

  kops_test::device_free(d_in);
  kops_test::device_free(d_out);
}

void test_merge_state() {
  const int64_t heads = 4;
  const int64_t head_dim = 64;
  std::vector<float> oa_f(heads * head_dim), ob_f(heads * head_dim), la_f(heads), lb_f(heads);
  fill(oa_f, 8u);
  fill(ob_f, 9u);
  fill(la_f, 10u);
  fill(lb_f, 11u);
  for (float& v : la_f) v = -2.0f + 0.1f * v;
  for (float& v : lb_f) v = -2.0f + 0.1f * v;

  float* d_oa = kops_test::device_alloc<float>(oa_f.size());
  float* d_ob = kops_test::device_alloc<float>(ob_f.size());
  float* d_la = kops_test::device_alloc<float>(la_f.size());
  float* d_lb = kops_test::device_alloc<float>(lb_f.size());
  float* d_o = kops_test::device_alloc<float>(oa_f.size());
  float* d_l = kops_test::device_alloc<float>(la_f.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_oa, oa_f);
  kops_test::copy_in(d_ob, ob_f);
  kops_test::copy_in(d_la, la_f);
  kops_test::copy_in(d_lb, lb_f);
  kops::Tensor t_o = kops_test::tensor3d(d_o, kops::DType::kFloat32, 1, heads, head_dim);
  kops::Tensor t_l = kops_test::tensor2d(d_l, kops::DType::kFloat32, 1, heads);
  kops::Tensor t_oa = kops_test::tensor3d(d_oa, kops::DType::kFloat32, 1, heads, head_dim);
  kops::Tensor t_ob = kops_test::tensor3d(d_ob, kops::DType::kFloat32, 1, heads, head_dim);
  kops::Tensor t_la = kops_test::tensor2d(d_la, kops::DType::kFloat32, 1, heads);
  kops::Tensor t_lb = kops_test::tensor2d(d_lb, kops::DType::kFloat32, 1, heads);

  KOPS_EXPECT_OK(kops::attn::merge_state(t_o, t_l, t_oa, t_la, t_ob, t_lb, ctx), "merge_state");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto out = kops_test::copy_out(d_o, oa_f.size());
  const auto lse = kops_test::copy_out(d_l, la_f.size());
  std::vector<double> oa_d(oa_f.begin(), oa_f.end());
  std::vector<double> ob_d(ob_f.begin(), ob_f.end());
  std::vector<double> la_d(la_f.begin(), la_f.end());
  std::vector<double> lb_d(lb_f.begin(), lb_f.end());
  std::vector<double> states(2 * heads * head_dim);
  std::vector<double> lses(2 * heads);
  for (int64_t h = 0; h < heads; ++h) {
    lses[h] = la_d[h];
    lses[heads + h] = lb_d[h];
    for (int64_t x = 0; x < head_dim; ++x) {
      states[h * head_dim + x] = oa_d[h * head_dim + x];
      states[(heads + h) * head_dim + x] = ob_d[h * head_dim + x];
    }
  }
  std::vector<double> ref_o(oa_f.size()), ref_l(la_f.size());
  KOPS_EXPECT_OK(kops::ref::merge_states_f64(states.data(), lses.data(), 1, 2, heads, head_dim,
                                             ref_o.data(), ref_l.data()),
                 "merge_states_f64");
  kops_test::compare(out, ref_o, 1e-4, 1e-4, "merge_state.output");
  kops_test::compare(lse, ref_l, 1e-5, 1e-5, "merge_state.lse");

  kops_test::device_free(d_oa);
  kops_test::device_free(d_ob);
  kops_test::device_free(d_la);
  kops_test::device_free(d_lb);
  kops_test::device_free(d_o);
  kops_test::device_free(d_l);
}

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  std::vector<float> data(4096, 1.0f);
  float* d = kops_test::device_alloc<float>(data.size());
  kops_test::copy_in(d, data);

  // ndim out of range -> unsupported shape
  kops::Tensor t1 = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  t1.ndim = 1;
  t1.shape[0] = 1024;
  t1.shape[1] = 0;
  t1.stride[0] = 1;
  kops::Tensor t1_out = kops_test::tensor2d(d, kops::DType::kFloat32, 1, 1024);
  KOPS_EXPECT_STATUS(kops::norm::rms_norm(t1_out, t1, kops::Tensor{}, 1e-5f, 0.0f, ctx),
                     kops::Status::kUnsupportedShape, "rms_norm ndim");

  // misaligned pointer -> misaligned
  kops::Tensor t2 = kops_test::tensor2d(reinterpret_cast<char*>(d) + 1, kops::DType::kFloat32, 4, 256);
  kops::Tensor t2_out = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  KOPS_EXPECT_STATUS(kops::norm::rms_norm(t2_out, t2, kops::Tensor{}, 1e-5f, 0.0f, ctx),
                     kops::Status::kMisaligned, "rms_norm alignment");

  // dtype mismatch -> unsupported dtype
  kops::Tensor t3_in = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  kops::Tensor t3_out = kops_test::tensor2d(d, kops::DType::kFloat16, 4, 256);
  KOPS_EXPECT_STATUS(kops::norm::rms_norm(t3_out, t3_in, kops::Tensor{}, 1e-5f, 0.0f, ctx),
                     kops::Status::kUnsupportedDtype, "rms_norm dtype");

  // residual stride mismatch -> unsupported layout
  kops::Tensor t4_in = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  kops::Tensor t4_res = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  t4_res.stride[0] = 264;
  KOPS_EXPECT_STATUS(kops::norm::fused_add_rms_norm(t4_in, t4_res, kops::Tensor{}, 1e-5f, 0.0f,
                                                    1.0f, ctx),
                     kops::Status::kUnsupportedLayout, "fused_add_rms_norm stride");

  // merge_state partial dtype must be fp32
  kops::Tensor t5_o = kops_test::tensor3d(d, kops::DType::kFloat32, 1, 4, 64);
  kops::Tensor t5_l = kops_test::tensor2d(d, kops::DType::kFloat32, 1, 4);
  kops::Tensor t5_a = kops_test::tensor3d(d, kops::DType::kFloat16, 1, 4, 64);
  KOPS_EXPECT_STATUS(kops::attn::merge_state(t5_o, t5_l, t5_a, t5_l, t5_a, t5_l, ctx),
                     kops::Status::kUnsupportedDtype, "merge_state partial dtype");

  // declared but not implemented yet
  KOPS_EXPECT_STATUS(kops::norm::layer_norm(t5_o, t5_o, kops::Tensor{}, kops::Tensor{}, 1e-5f, 0.0f,
                                            ctx),
                     kops::Status::kNotImplemented, "layer_norm planned");
  KOPS_EXPECT_STATUS(kops::attn::merge_states(t5_o, t5_l, t5_o, t5_l, ctx),
                     kops::Status::kNotImplemented, "merge_states planned");

  kops_test::device_free(d);
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;

  test_rms_norm<__nv_bfloat16>("rms_norm bf16", kops::DType::kBFloat16, 2e-2, 2e-2);
  test_rms_norm<float>("rms_norm fp32", kops::DType::kFloat32, 1e-5, 1e-5);
  test_fused_add_rms_norm();
  test_activation();
  test_act_and_mul();
  test_merge_state();
  test_contracts();
  return kops_test::finish("test_m1");
}
