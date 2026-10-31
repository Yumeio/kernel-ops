// =============================================================================
// kernel-ops | tests/unit/test_model_ops.cc
// B0 numerics and contract tests: residual_add and cast.
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
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/ops.h"
#include "csrc/runtime/registry.h"

namespace {

using kops::DType;

void fill(std::vector<float>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
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
void test_residual(kops::DType dt, const char* name, bool three) {
  const int64_t rows = 17;
  const int64_t d = 257;  // deliberately not a vector multiple
  std::vector<float> af(rows * d), bf(rows * d), cf(rows * d);
  fill(af, 1u);
  fill(bf, 2u);
  fill(cf, 3u);
  const auto a = to_type<T>(af), b = to_type<T>(bf), c = to_type<T>(cf);
  T* da = kops_test::device_alloc<T>(a.size());
  T* db = kops_test::device_alloc<T>(b.size());
  T* dc = three ? kops_test::device_alloc<T>(c.size()) : nullptr;
  T* dout = kops_test::device_alloc<T>(a.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(da, a);
  kops_test::copy_in(db, b);
  if (three) kops_test::copy_in(dc, c);

  kops::Tensor ta = kops_test::tensor2d(da, dt, rows, d);
  kops::Tensor tb = kops_test::tensor2d(db, dt, rows, d);
  kops::Tensor tc = three ? kops_test::tensor2d(dc, dt, rows, d) : kops::Tensor{};
  kops::Tensor tout = kops_test::tensor2d(dout, dt, rows, d);
  const float as = 0.5f, bs = -1.25f, cs = 2.0f;
  KOPS_EXPECT_OK(kops::model_ops::residual_add(tout, ta, as, tb, bs, tc, cs, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(dout, a.size());
  const auto ad = to_double(a), bd = to_double(b), cd = to_double(c);
  std::vector<double> ref(a.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    ref[i] = as * ad[i] + bs * bd[i] + (three ? cs * cd[i] : 0.0);
  }
  const double atol = dt == DType::kFloat32 ? 1e-5 : 3e-2;
  const double rtol = dt == DType::kFloat32 ? 1e-5 : 3e-2;
  kops_test::compare(got, ref, atol, rtol, name);

  // exact alias: out == a
  if (!three) {
    KOPS_EXPECT_OK(kops::model_ops::residual_add(ta, ta, as, tb, bs, kops::Tensor{}, cs, ctx),
                   "residual_add alias");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto alias = kops_test::copy_out(da, a.size());
    std::vector<double> alias_ref(a.size());
    for (size_t i = 0; i < alias_ref.size(); ++i) {
      alias_ref[i] = as * ad[i] + bs * bd[i];
    }
    kops_test::compare(alias, alias_ref, atol, rtol, "residual_add alias");
  }

  kops_test::device_free(da);
  kops_test::device_free(db);
  kops_test::device_free(dc);
  kops_test::device_free(dout);
}

template <typename Out, typename In>
void test_cast(DType out_dt, DType in_dt, const char* name) {
  const int64_t rows = 11;
  const int64_t d = 129;
  std::vector<float> xf(rows * d);
  fill(xf, 4u);
  const auto x = to_type<In>(xf);
  In* dx = kops_test::device_alloc<In>(x.size());
  Out* dy = kops_test::device_alloc<Out>(x.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(dx, x);
  const kops::Tensor tx = kops_test::tensor2d(dx, in_dt, rows, d);
  const kops::Tensor ty = kops_test::tensor2d(dy, out_dt, rows, d);
  KOPS_EXPECT_OK(kops::model_ops::cast(ty, tx, 1.75f, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(dy, x.size());
  const auto xd = to_double(x);
  std::vector<double> ref(xd.size());
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = 1.75 * xd[i];
  const double tol = out_dt == DType::kFloat32 ? 1e-5 : 3e-2;
  kops_test::compare(got, ref, tol, tol, name);
  kops_test::device_free(dx);
  kops_test::device_free(dy);
}

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  std::vector<float> xf(64 * 16, 1.0f);
  float* d = kops_test::device_alloc<float>(xf.size());
  kops_test::copy_in(d, xf);
  kops::Tensor a = kops_test::tensor2d(d, DType::kFloat32, 64, 16);
  kops::Tensor b = kops_test::tensor2d(d, DType::kFloat32, 64, 15);
  kops::Tensor out = kops_test::tensor2d(d, DType::kFloat32, 64, 16);

  KOPS_EXPECT_STATUS(kops::model_ops::residual_add(out, a, 1.0f, b, 1.0f, kops::Tensor{}, 0.0f, ctx),
                     kops::Status::kInvalidArgument, "residual_add shape");
  kops::Tensor h = kops_test::tensor2d(d, DType::kFloat16, 64, 16);
  KOPS_EXPECT_STATUS(kops::model_ops::residual_add(out, a, 1.0f, h, 1.0f, kops::Tensor{}, 0.0f, ctx),
                     kops::Status::kUnsupportedDtype, "residual_add dtype");
  kops::Tensor mis = kops_test::tensor2d(reinterpret_cast<char*>(d) + 1, DType::kFloat32, 8, 16);
  KOPS_EXPECT_STATUS(kops::model_ops::cast(out, mis, 1.0f, ctx), kops::Status::kMisaligned,
                     "cast alignment");
  kops_test::device_free(d);
}

void test_graph() {
  const int64_t rows = 8, d = 64;
  const size_t n = static_cast<size_t>(rows * d);
  std::vector<float> xf(n, 1.0f), yf(n, 2.0f);
  float* x = kops_test::device_alloc<float>(n);
  float* y = kops_test::device_alloc<float>(n);
  float* z = kops_test::device_alloc<float>(n);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx) || x == nullptr || y == nullptr || z == nullptr) return;
  kops_test::copy_in(x, xf);
  kops_test::copy_in(y, yf);
  kops::Tensor tx = kops_test::tensor2d(x, DType::kFloat32, rows, d);
  kops::Tensor ty = kops_test::tensor2d(y, DType::kFloat32, rows, d);
  kops::Tensor tz = kops_test::tensor2d(z, DType::kFloat32, rows, d);
  cudaStream_t stream = nullptr;
  KOPS_TEST_CHECK(cudaStreamCreate(&stream) == cudaSuccess);
  kops::LaunchContext gctx = ctx;
  gctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  KOPS_TEST_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  KOPS_EXPECT_OK(kops::model_ops::residual_add(tz, tx, 1.0f, ty, 1.0f, kops::Tensor{}, 0.0f, gctx),
                 "graph residual");
  KOPS_EXPECT_OK(kops::model_ops::cast(tz, tz, 0.5f, gctx), "graph cast");
  cudaGraph_t graph = nullptr;
  KOPS_TEST_CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  KOPS_TEST_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
  std::vector<float> xf2(n, 3.0f), yf2(n, 4.0f);
  kops_test::copy_in(x, xf2);
  kops_test::copy_in(y, yf2);
  KOPS_TEST_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
  KOPS_TEST_CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  const auto got = kops_test::copy_out(z, n);
  std::vector<double> ref(n, 3.5);  // (3 + 4) * 0.5
  kops_test::compare(got, ref, 1e-5, 1e-5, "graph replay");
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  kops_test::device_free(x);
  kops_test::device_free(y);
  kops_test::device_free(z);
}

void test_registry() {
  const kops::registry::Capability* rows[4] = {};
  int32_t n = 0;
  KOPS_EXPECT_OK(kops::registry::find_capabilities("X02.residual_add", 860, rows, 4, &n),
                 "registry find");
  KOPS_TEST_CHECK(n >= 1);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const kops::registry::Capability* chosen = nullptr;
  KOPS_EXPECT_OK(kops::registry::select("X02.residual_add", kops::registry::SpecKind::kNone,
                                        nullptr, ctx, kops::registry::Provider::kNative, true,
                                        &chosen),
                 "registry select");
  KOPS_TEST_CHECK(chosen != nullptr);
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;
  test_residual<float>(DType::kFloat32, "residual_add fp32 3-input", true);
  test_residual<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 2-input", false);
  test_cast<float, __nv_bfloat16>(DType::kFloat32, DType::kBFloat16, "cast bf16->fp32");
  test_cast<__nv_bfloat16, float>(DType::kBFloat16, DType::kFloat32, "cast fp32->bf16");
  test_cast<float, __half>(DType::kFloat32, DType::kFloat16, "cast fp16->fp32");
  test_contracts();
  test_graph();
  test_registry();
  return kops_test::finish("test_model_ops");
}
