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

// -----------------------------------------------------------------------------
// Layout and dtype coverage for the persistent chunk kernel: the same logical
// row data through the 32 B chunk path (contiguous), the 16 B / padded-stride
// path and the scalar path (odd stride), plus exact in-place reuse. Every path
// must agree bit for bit with the contiguous one, because each output element
// is produced by one thread with one rounding.
// -----------------------------------------------------------------------------

kops::Tensor view2d(void* data, kops::DType dt, int64_t rows, int64_t cols, int64_t stride) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dt;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = stride;
  t.stride[1] = 1;
  return t;
}

template <typename T>
std::vector<T> gather_rows(const std::vector<T>& body, int64_t rows, int64_t cols,
                           int64_t stride) {
  std::vector<T> out(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) out[static_cast<size_t>(r * cols + c)] = body[static_cast<size_t>(r * stride + c)];
  }
  return out;
}

template <typename T>
std::vector<T> spread_rows(const std::vector<T>& flat, int64_t rows, int64_t cols,
                           int64_t stride) {
  std::vector<T> body(static_cast<size_t>(rows * stride));
  for (auto& v : body) kops_test::from_float(v, 7.0f);  // poison the padding
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) body[static_cast<size_t>(r * stride + c)] = flat[static_cast<size_t>(r * cols + c)];
  }
  return body;
}

/// One residual_add configuration through every layout path.
template <typename T>
void test_residual_layouts(kops::DType dt, const char* name, int64_t rows, int64_t cols,
                           bool three) {
  const float as = 0.5f, bs = -1.25f, cs = 2.0f;
  std::vector<float> af(static_cast<size_t>(rows * cols)), bf(af.size()), cf(af.size());
  fill(af, 7u);
  fill(bf, 8u);
  fill(cf, 9u);
  const std::vector<T> a = to_type<T>(af), b = to_type<T>(bf), c = to_type<T>(cf);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  // Reference: contiguous views (the chunk path).
  T* da = kops_test::device_alloc<T>(a.size());
  T* db = kops_test::device_alloc<T>(b.size());
  T* dc = three ? kops_test::device_alloc<T>(c.size()) : nullptr;
  T* dout = kops_test::device_alloc<T>(a.size());
  if (da == nullptr || db == nullptr || dout == nullptr || (three && dc == nullptr)) return;
  kops_test::copy_in(da, a);
  kops_test::copy_in(db, b);
  if (three) kops_test::copy_in(dc, c);
  KOPS_EXPECT_OK(kops::model_ops::residual_add(view2d(dout, dt, rows, cols, cols),
                                               view2d(da, dt, rows, cols, cols), as,
                                               view2d(db, dt, rows, cols, cols), bs,
                                               three ? view2d(dc, dt, rows, cols, cols) : kops::Tensor{},
                                               cs, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto ref = kops_test::copy_out(dout, a.size());
  const auto ref_d = to_double(ref);

  // Same data with a padded row stride (keeps the vector path) and with an odd
  // stride (forces the scalar path). Both must match the reference exactly.
  for (const int64_t pad : {int64_t{8}, int64_t{1}}) {
    const int64_t stride = cols + pad;
    const auto ab = spread_rows(a, rows, cols, stride);
    const auto bb = spread_rows(b, rows, cols, stride);
    const auto cb = spread_rows(c, rows, cols, stride);
    T* pa = kops_test::device_alloc<T>(ab.size());
    T* pb = kops_test::device_alloc<T>(bb.size());
    T* pc = three ? kops_test::device_alloc<T>(cb.size()) : nullptr;
    T* po = kops_test::device_alloc<T>(ab.size());
    if (pa == nullptr || pb == nullptr || po == nullptr || (three && pc == nullptr)) return;
    kops_test::copy_in(pa, ab);
    kops_test::copy_in(pb, bb);
    if (three) kops_test::copy_in(pc, cb);
    KOPS_EXPECT_OK(kops::model_ops::residual_add(view2d(po, dt, rows, cols, stride),
                                                 view2d(pa, dt, rows, cols, stride), as,
                                                 view2d(pb, dt, rows, cols, stride), bs,
                                                 three ? view2d(pc, dt, rows, cols, stride)
                                                       : kops::Tensor{},
                                                 cs, ctx),
                   name);
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = gather_rows(kops_test::copy_out(po, ab.size()), rows, cols, stride);
    kops_test::compare(got, ref_d, 0.0, 0.0, name);
    kops_test::device_free(pa);
    kops_test::device_free(pb);
    kops_test::device_free(pc);
    kops_test::device_free(po);
  }

  // Exact in-place output (out == a) must reproduce the out-of-place result.
  kops_test::copy_in(da, a);
  KOPS_EXPECT_OK(kops::model_ops::residual_add(view2d(da, dt, rows, cols, cols),
                                              view2d(da, dt, rows, cols, cols), as,
                                              view2d(db, dt, rows, cols, cols), bs,
                                              three ? view2d(dc, dt, rows, cols, cols) : kops::Tensor{},
                                              cs, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  kops_test::compare(kops_test::copy_out(da, a.size()), ref_d, 0.0, 0.0, name);

  kops_test::device_free(da);
  kops_test::device_free(db);
  kops_test::device_free(dc);
  kops_test::device_free(dout);
}

/// Odd hidden sizes exercise the tail loops of the scalar path.
template <typename T>
void test_residual_odd(kops::DType dt, const char* name, int64_t cols) {
  const int64_t rows = 5;
  const float as = 0.75f, bs = -2.0f;
  std::vector<float> af(static_cast<size_t>(rows * cols)), bf(af.size());
  fill(af, 21u);
  fill(bf, 22u);
  const std::vector<T> a = to_type<T>(af), b = to_type<T>(bf);
  T* da = kops_test::device_alloc<T>(a.size());
  T* db = kops_test::device_alloc<T>(b.size());
  T* dout = kops_test::device_alloc<T>(a.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(da, a);
  kops_test::copy_in(db, b);
  KOPS_EXPECT_OK(kops::model_ops::residual_add(view2d(dout, dt, rows, cols, cols),
                                               view2d(da, dt, rows, cols, cols), as,
                                               view2d(db, dt, rows, cols, cols), bs,
                                               kops::Tensor{}, 0.0f, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(dout, a.size());
  const auto ad = to_double(a), bd = to_double(b);
  std::vector<double> ref(a.size());
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = as * ad[i] + bs * bd[i];
  const double tol = dt == DType::kFloat32 ? 1e-5 : 3e-2;
  kops_test::compare(got, ref, tol, tol, name);
  kops_test::device_free(da);
  kops_test::device_free(db);
  kops_test::device_free(dout);
}

/// One cast configuration through the contiguous, padded and odd-stride paths.
template <typename Out, typename In>
void test_cast_layouts(DType out_dt, DType in_dt, const char* name, int64_t rows, int64_t cols,
                       float scale) {
  std::vector<float> xf(static_cast<size_t>(rows * cols));
  fill(xf, 31u);
  const std::vector<In> x = to_type<In>(xf);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  In* dx = kops_test::device_alloc<In>(x.size());
  Out* dy = kops_test::device_alloc<Out>(x.size());
  if (dx == nullptr || dy == nullptr) return;
  kops_test::copy_in(dx, x);
  KOPS_EXPECT_OK(kops::model_ops::cast(view2d(dy, out_dt, rows, cols, cols),
                                       view2d(dx, in_dt, rows, cols, cols), scale, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto ref = kops_test::copy_out(dy, x.size());
  const auto ref_d = to_double(ref);

  for (const int64_t pad : {int64_t{8}, int64_t{1}}) {
    const int64_t stride = cols + pad;
    const auto xb = spread_rows(x, rows, cols, stride);
    In* px = kops_test::device_alloc<In>(xb.size());
    Out* py = kops_test::device_alloc<Out>(xb.size());
    if (px == nullptr || py == nullptr) return;
    kops_test::copy_in(px, xb);
    KOPS_EXPECT_OK(kops::model_ops::cast(view2d(py, out_dt, rows, cols, stride),
                                         view2d(px, in_dt, rows, cols, stride), scale, ctx), name);
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = gather_rows(kops_test::copy_out(py, xb.size()), rows, cols, stride);
    kops_test::compare(got, ref_d, 0.0, 0.0, name);
    kops_test::device_free(px);
    kops_test::device_free(py);
  }
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
  // The capability table is per exact SM. select() can only answer where a row
  // for this arch_id exists, so only assert it on a machine the table covers
  // (an exact-SM row outside the table is a registry gap, not a kernel bug).
  int32_t here = 0;
  KOPS_EXPECT_OK(
      kops::registry::find_capabilities("X02.residual_add", ctx.arch_id, rows, 4, &here),
      "registry find (this arch)");
  if (here == 0) return;
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

  // ---- layout paths, mixed dtypes and odd tails ----------------------------
  test_residual_layouts<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 layouts", 9, 4096,
                                       false);
  test_residual_layouts<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 3-in layouts", 7, 512,
                                       true);
  test_residual_layouts<__half>(DType::kFloat16, "residual_add fp16 layouts", 6, 1024, false);
  test_residual_layouts<float>(DType::kFloat32, "residual_add fp32 layouts", 5, 1000, true);
  // bf16 input with an fp32 output (mixed chunk granule).
  test_cast_layouts<float, __nv_bfloat16>(DType::kFloat32, DType::kBFloat16,
                                          "cast bf16->fp32 layouts", 6, 1000, 1.5f);
  test_cast_layouts<__nv_bfloat16, float>(DType::kBFloat16, DType::kFloat32,
                                          "cast fp32->bf16 layouts", 6, 1000, 0.5f);
  test_cast_layouts<__nv_bfloat16, __half>(DType::kBFloat16, DType::kFloat16,
                                           "cast fp16->bf16 layouts", 5, 4096, 1.0f);
  test_cast_layouts<__half, __nv_bfloat16>(DType::kFloat16, DType::kBFloat16,
                                           "cast bf16->fp16 layouts", 5, 4096, 2.0f);
  test_residual_odd<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 h1", 1);
  test_residual_odd<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 h251", 251);
  test_residual_odd<__nv_bfloat16>(DType::kBFloat16, "residual_add bf16 h4095", 4095);
  test_residual_odd<float>(DType::kFloat32, "residual_add fp32 h250", 250);
  test_residual_odd<__half>(DType::kFloat16, "residual_add fp16 h4097", 4097);
  test_contracts();
  test_graph();
  test_registry();
  return kops_test::finish("test_model_ops");
}
