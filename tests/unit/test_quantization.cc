// =============================================================================
// kernel-ops | tests/unit/test_quantization.cc
// B3 quantization family (rows 1-4):
//   absmax, per-tensor / per-token / per-group FP8 + INT8, MXFP8 / MXFP4,
//   NVFP4, dequantize, int4 pack/unpack, Hadamard.
//
// Oracle: host fp64 arithmetic plus the toolkit's host-side e4m3 / e2m1
// conversions. No third-party library runs, no PyTorch is linked.
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
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/quantization/ops.h"
#include "csrc/common/op_types.cuh"
#include "csrc/common/device.cuh"
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

namespace {

using kops::DType;
using kops::LaunchContext;
using kops::Status;
using kops::Tensor;
using kops::ActQuantMode;
using kops::ActQuantOut;
using kops::QuantScheme;
using kops::ScaleGranularity;
using kops::ScaleLayout;

template <typename T>
struct DevBuf {
  T* p = nullptr;
  explicit DevBuf(size_t n) : p(kops_test::device_alloc<T>(n)) {}
  ~DevBuf() { kops_test::device_free(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  bool ok() const { return p != nullptr; }
};

void fill_uniform(std::vector<float>& v, unsigned seed, float lo = -2.0f, float hi = 2.0f) {
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

float e4m3_round(float x) {
  const uint8_t b = static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
  return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
}

uint8_t e4m3_byte(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

float e2m1_round(uint8_t code) {
  constexpr float kTable[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float v = kTable[code & 7u];
  return (code & 8u) ? -v : v;
}

uint8_t e2m1_pair(float a, float b) {
  return static_cast<uint8_t>(
      __nv_cvt_float2_to_fp4x2(make_float2(a, b), __NV_E2M1, cudaRoundNearest));
}

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

Tensor tensor2d(void* data, DType dtype, int64_t rows, int64_t cols, int64_t stride = 0) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = stride == 0 ? cols : stride;
  t.stride[1] = 1;
  return t;
}

ActQuantOut make_out(void* q, int64_t q_stride, void* scales, int64_t scale_stride, DType qdtype,
                     ActQuantMode mode, ScaleLayout layout = ScaleLayout::kRowMajor,
                     int group = 0) {
  ActQuantOut o{};
  o.q = q;
  o.q_row_stride = q_stride;
  o.scales = scales;
  o.scale_row_stride = scale_stride;
  o.q_dtype = qdtype;
  o.mode = mode;
  o.scale_layout = layout;
  o.group_size = group;
  return o;
}

// -----------------------------------------------------------------------------
// absmax
// -----------------------------------------------------------------------------

void test_absmax(LaunchContext& ctx) {
  for (int dt = 0; dt < 3; ++dt) {
    const int rows = 7, cols = 1024;
    const DType dtype = dt == 0 ? DType::kFloat32
                                : (dt == 1 ? DType::kFloat16 : DType::kBFloat16);
    DevBuf<float> amax(1);
    std::vector<float> host(rows * cols);
    fill_uniform(host, 11 + dt, -5.0f, 5.0f);
    float want = 0.0f;
    std::vector<float> rounded(host.size());
    if (dtype == DType::kFloat32) {
      for (size_t i = 0; i < host.size(); ++i) rounded[i] = host[i];
    } else if (dtype == DType::kFloat16) {
      for (size_t i = 0; i < host.size(); ++i) {
        rounded[i] = __half2float(__float2half_rn(host[i]));
      }
    } else {
      for (size_t i = 0; i < host.size(); ++i) {
        rounded[i] = __bfloat162float(__float2bfloat16_rn(host[i]));
      }
    }
    for (float v : rounded) want = std::max(want, std::fabs(v));
    if (!amax.ok()) return;
    const std::vector<float> zero = {0.0f};
    kops_test::copy_in(amax.p, zero);
    void* in = nullptr;
    Tensor tin{};
    if (dtype == DType::kFloat32) {
      in = kops_test::device_alloc<float>(host.size());
      kops_test::copy_in(static_cast<float*>(in), rounded);
      tin = tensor2d(in, dtype, rows, cols);
    } else if (dtype == DType::kFloat16) {
      auto h = cast_from_float<__half>(rounded);
      in = kops_test::device_alloc<__half>(h.size());
      kops_test::copy_in(static_cast<__half*>(in), h);
      tin = tensor2d(in, dtype, rows, cols);
    } else {
      auto h = cast_from_float<__nv_bfloat16>(rounded);
      in = kops_test::device_alloc<__nv_bfloat16>(h.size());
      kops_test::copy_in(static_cast<__nv_bfloat16*>(in), h);
      tin = tensor2d(in, dtype, rows, cols);
    }
    if (in == nullptr) return;
    KOPS_EXPECT_OK(kops::quant::absmax(tensor1d(amax.p, DType::kFloat32, 1), tin, ctx), "absmax");
    const auto got = kops_test::copy_out(amax.p, 1);
    KOPS_TEST_CHECK(std::fabs(got[0] - want) <= 1e-6f + 1e-6f * want);
    cudaFree(in);
  }
}

// -----------------------------------------------------------------------------
// per-tensor / per-token / per-group
// -----------------------------------------------------------------------------

void test_per_tensor_fp8(LaunchContext& ctx) {
  const int rows = 5, cols = 1024;
  DevBuf<__nv_bfloat16> in(rows * cols);
  DevBuf<uint8_t> q(rows * cols);
  DevBuf<float> scale(1);
  if (!in.ok() || !q.ok() || !scale.ok()) return;
  std::vector<float> host(rows * cols);
  fill_uniform(host, 21);
  auto half = cast_from_float<__nv_bfloat16>(host);
  kops_test::copy_in(in.p, half);
  const float s = 0.0125f;
  kops_test::copy_in(scale.p, std::vector<float>{s});
  ActQuantOut out = make_out(q.p, cols, nullptr, 0, DType::kFloat8E4M3,
                             ActQuantMode::kStaticPerTensor);
  out.static_scale = scale.p;
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     out, tensor2d(in.p, DType::kBFloat16, rows, cols), Tensor{}, 0, ctx),
                 "per-tensor fp8");
  const auto got = kops_test::copy_out(q.p, rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    const float x = kops_test::to_float(half[static_cast<size_t>(i)]);
    KOPS_TEST_CHECK(got[static_cast<size_t>(i)] == e4m3_byte(x / s));
  }
}

void test_per_tensor_int8(LaunchContext& ctx) {
  const int rows = 3, cols = 640;
  DevBuf<float> in(rows * cols);
  DevBuf<int8_t> q(rows * cols);
  DevBuf<float> scale(1);
  if (!in.ok() || !q.ok() || !scale.ok()) return;
  std::vector<float> host(rows * cols);
  fill_uniform(host, 22);
  kops_test::copy_in(in.p, host);
  kops_test::copy_in(scale.p, std::vector<float>{0.02f});
  ActQuantOut out = make_out(q.p, cols, nullptr, 0, DType::kInt8,
                             ActQuantMode::kStaticPerTensor);
  out.static_scale = scale.p;
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     out, tensor2d(in.p, DType::kFloat32, rows, cols), Tensor{}, 0, ctx),
                 "per-tensor int8");
  const auto got = kops_test::copy_out(q.p, rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    int want = static_cast<int>(std::nearbyint(host[static_cast<size_t>(i)] / 0.02f));
    want = std::max(-127, std::min(127, want));
    KOPS_TEST_CHECK(got[static_cast<size_t>(i)] == static_cast<int8_t>(want));
  }
}

void test_per_token_fp8(LaunchContext& ctx) {
  const int rows = 17, cols = 4096;
  DevBuf<__nv_bfloat16> in(rows * cols);
  DevBuf<uint8_t> q(rows * cols);
  DevBuf<float> scales(rows);
  if (!in.ok() || !q.ok() || !scales.ok()) return;
  std::vector<float> host(rows * cols);
  fill_uniform(host, 31, -6.0f, 6.0f);
  auto half = cast_from_float<__nv_bfloat16>(host);
  kops_test::copy_in(in.p, half);
  ActQuantOut out = make_out(q.p, cols, scales.p, 1, DType::kFloat8E4M3,
                             ActQuantMode::kDynamicPerToken);
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     out, tensor2d(in.p, DType::kBFloat16, rows, cols), Tensor{}, 0, ctx),
                 "per-token fp8");
  const auto got = kops_test::copy_out(q.p, rows * cols);
  const auto gs = kops_test::copy_out(scales.p, rows);
  for (int r = 0; r < rows; ++r) {
    float amax = 0.0f;
    for (int c = 0; c < cols; ++c) {
      amax = std::max(amax, std::fabs(kops_test::to_float(half[static_cast<size_t>(r * cols + c)])));
    }
    const float s = std::max(amax / 448.0f, 1.0f / (448.0f * 512.0f));
    KOPS_TEST_CHECK(std::fabs(gs[static_cast<size_t>(r)] - s) <= 1e-7f + 1e-6f * s);
    for (int c = 0; c < cols; ++c) {
      const float x = kops_test::to_float(half[static_cast<size_t>(r * cols + c)]);
      KOPS_TEST_CHECK(got[static_cast<size_t>(r * cols + c)] == e4m3_byte(x / s));
    }
  }
}

void test_per_token_int8_asym(LaunchContext& ctx) {
  const int rows = 9, cols = 512;
  DevBuf<float> in(rows * cols);
  DevBuf<int8_t> q(rows * cols);
  DevBuf<float> scales(rows), azp(rows);
  if (!in.ok() || !q.ok() || !scales.ok() || !azp.ok()) return;
  std::vector<float> host(rows * cols);
  fill_uniform(host, 32, -3.0f, 5.0f);
  kops_test::copy_in(in.p, host);
  ActQuantOut out = make_out(q.p, cols, scales.p, 1, DType::kInt8,
                             ActQuantMode::kDynamicPerToken);
  out.symmetric = false;
  out.zero_points = static_cast<int32_t*>(nullptr);
  DevBuf<int32_t> zero(rows);
  out.zero_points = zero.p;
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     out, tensor2d(in.p, DType::kFloat32, rows, cols), Tensor{}, 0, ctx),
                 "per-token int8 asym");
  const auto got = kops_test::copy_out(q.p, rows * cols);
  const auto gs = kops_test::copy_out(scales.p, rows);
  const auto gz = kops_test::copy_out(zero.p, rows);
  for (int r = 0; r < rows; ++r) {
    float mx = -1e30f, mn = 1e30f;
    for (int c = 0; c < cols; ++c) {
      mx = std::max(mx, host[static_cast<size_t>(r * cols + c)]);
      mn = std::min(mn, host[static_cast<size_t>(r * cols + c)]);
    }
    const float s = (mx - mn) / 255.0f;
    const int z = static_cast<int>(std::nearbyint(-128.0f - mn / s));
    KOPS_TEST_CHECK(std::fabs(gs[static_cast<size_t>(r)] - s) <= 1e-6f * s);
    KOPS_TEST_CHECK(gz[static_cast<size_t>(r)] == z);
    for (int c = 0; c < cols; ++c) {
      int want = static_cast<int>(std::nearbyint(host[static_cast<size_t>(r * cols + c)] / s + z));
      want = std::max(-128, std::min(127, want));
      KOPS_TEST_CHECK(got[static_cast<size_t>(r * cols + c)] == static_cast<int8_t>(want));
    }
  }
}

void test_per_group(LaunchContext& ctx) {
  const int rows = 6, cols = 1024, group = 128;
  const int groups = cols / group;
  DevBuf<__nv_bfloat16> in(rows * cols);
  DevBuf<uint8_t> q(rows * cols);
  DevBuf<float> scales(rows * groups);
  if (!in.ok() || !q.ok() || !scales.ok()) return;
  std::vector<float> host(rows * cols);
  fill_uniform(host, 41, -4.0f, 4.0f);
  auto half = cast_from_float<__nv_bfloat16>(host);
  kops_test::copy_in(in.p, half);
  ActQuantOut out = make_out(q.p, cols, scales.p, groups, DType::kFloat8E4M3,
                             ActQuantMode::kPerGroup, ScaleLayout::kRowMajor, group);
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     out, tensor2d(in.p, DType::kBFloat16, rows, cols), Tensor{}, 0, ctx),
                 "per-group fp8");
  const auto got = kops_test::copy_out(q.p, rows * cols);
  const auto gs = kops_test::copy_out(scales.p, rows * groups);
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      float amax = 0.0f;
      for (int j = 0; j < group; ++j) {
        amax = std::max(amax,
                        std::fabs(kops_test::to_float(half[static_cast<size_t>(r * cols + g * group + j)])));
      }
      const float s = std::max(amax / 448.0f, 1e-10f);
      KOPS_TEST_CHECK(std::fabs(gs[static_cast<size_t>(r * groups + g)] - s) <= 1e-7f + 1e-6f * s);
      for (int j = 0; j < group; ++j) {
        const float x = kops_test::to_float(half[static_cast<size_t>(r * cols + g * group + j)]);
        KOPS_TEST_CHECK(got[static_cast<size_t>(r * cols + g * group + j)] == e4m3_byte(x / s));
      }
    }
  }

  // Column-major TMA scales: [groups, round_up(rows, 4)].
  DevBuf<float> tma(groups * ((rows + 3) / 4 * 4));
  if (!tma.ok()) return;
  ActQuantOut tma_out = make_out(q.p, cols, tma.p, (rows + 3) / 4 * 4, DType::kFloat8E4M3,
                                 ActQuantMode::kPerGroup, ScaleLayout::kColMajorTma, group);
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     tma_out, tensor2d(in.p, DType::kBFloat16, rows, cols), Tensor{}, 0, ctx),
                 "per-group col-major");
  const auto ts = kops_test::copy_out(tma.p, groups * ((rows + 3) / 4 * 4));
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      KOPS_TEST_CHECK(std::fabs(ts[static_cast<size_t>(g * ((rows + 3) / 4 * 4) + r)] -
                                gs[static_cast<size_t>(r * groups + g)]) < 1e-12f);
    }
  }

  // Masked (DeepEP) layout: expert 1 has only two live rows.
  const int experts = 3, max_m = 4;
  DevBuf<__nv_bfloat16> min(experts * max_m * cols);
  DevBuf<uint8_t> mq(experts * max_m * cols);
  DevBuf<float> msc(experts * max_m * groups);
  DevBuf<int32_t> mm(experts);
  if (!min.ok() || !mq.ok() || !msc.ok() || !mm.ok()) return;
  std::vector<float> mh(static_cast<size_t>(experts) * max_m * cols);
  fill_uniform(mh, 42, -1.0f, 1.0f);
  auto mhh = cast_from_float<__nv_bfloat16>(mh);
  kops_test::copy_in(min.p, mhh);
  kops_test::copy_in(mm.p, std::vector<int32_t>{4, 2, 3});
  ActQuantOut mo = make_out(mq.p, cols, msc.p, groups, DType::kFloat8E4M3,
                            ActQuantMode::kPerGroup, ScaleLayout::kRowMajor, group);
  Tensor min3{};
  min3.data = min.p;
  min3.dtype = DType::kBFloat16;
  min3.ndim = 3;
  min3.device = 0;
  min3.shape[0] = experts;
  min3.shape[1] = max_m;
  min3.shape[2] = cols;
  min3.stride[0] = max_m * cols;
  min3.stride[1] = cols;
  min3.stride[2] = 1;
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     mo, min3, tensor1d(mm.p, DType::kInt32, experts), max_m, ctx),
                 "masked per-group");
  const auto ms = kops_test::copy_out(msc.p, experts * max_m * groups);
  const auto mqo = kops_test::copy_out(mq.p, experts * max_m * cols);
  // Dead row 1*max_m + 2 stays untouched (zeros from cudaMalloc == 0 here).
  const int dead = (1 * max_m + 2) * groups;
  for (int g = 0; g < groups; ++g) KOPS_TEST_CHECK(ms[static_cast<size_t>(dead + g)] == 0.0f);
  KOPS_TEST_CHECK(mqo[static_cast<size_t>((1 * max_m + 2) * cols)] == 0u);
}

template <typename T>
std::vector<float> rounded_values(const std::vector<T>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = kops_test::to_float(v[i]);
  return out;
}

// -----------------------------------------------------------------------------
// Per-group vector path: shapes, layouts, masked expert strides, arithmetic modes
// -----------------------------------------------------------------------------

float pg_ue8m0_ceil(float s) {
  int e = 0;
  if (s > 0.0f && std::isfinite(s)) {
    std::frexp(s, &e);
    e -= 1;
    if (std::ldexp(1.0f, e) < s) ++e;
  } else {
    e = -127;
  }
  e = std::max(-127, std::min(127, e));
  return std::ldexp(1.0f, e);
}

/// Host oracle of the documented per-group contract: s = max(amax, eps) / 448 (rounded up to a
/// power of two with `ue8m0`) and one saturating e4m3 byte per element.
void pg_oracle(const std::vector<float>& vals, int rows, int cols, int group, bool ue8m0,
               std::vector<float>* scales, std::vector<uint8_t>* codes) {
  const int groups = cols / group;
  scales->assign(static_cast<size_t>(rows) * groups, 0.0f);
  codes->assign(static_cast<size_t>(rows) * cols, 0u);
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      float amax = 0.0f;
      for (int j = 0; j < group; ++j) {
        amax = std::fmax(amax, std::fabs(vals[static_cast<size_t>(r) * cols + g * group + j]));
      }
      float s = std::fmax(amax, 1e-10f) / 448.0f;
      if (ue8m0) s = pg_ue8m0_ceil(s);
      (*scales)[static_cast<size_t>(r) * groups + g] = s;
      for (int j = 0; j < group; ++j) {
        const size_t i = static_cast<size_t>(r) * cols + g * group + j;
        (*codes)[i] = e4m3_byte(vals[i] / s);
      }
    }
  }
}

template <typename T>
DType pg_dtype();
template <>
DType pg_dtype<__nv_bfloat16>() { return DType::kBFloat16; }
template <>
DType pg_dtype<__half>() { return DType::kFloat16; }
template <>
DType pg_dtype<float>() { return DType::kFloat32; }

float pg_u32_as_float(uint32_t bits) {
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

template <typename T>
T pg_bitcast(uint32_t bits);
template <>
__nv_bfloat16 pg_bitcast<__nv_bfloat16>(uint32_t bits) {
  return __ushort_as_bfloat16(static_cast<unsigned short>(bits));
}
template <>
__half pg_bitcast<__half>(uint32_t bits) {
  return __ushort_as_half(static_cast<unsigned short>(bits));
}

struct PgBuffers {
  std::vector<uint8_t> q;
  std::vector<float> s;
};

/// One dense per-group launch. `in_offset` shifts the input pointer by that many elements and
/// `q_stride` / `s_stride` are the declared row strides, so the caller can force the scalar
/// fallback (misaligned input or a q row stride that is not 16 B) and padded scale layouts.
template <typename T>
bool pg_dense(LaunchContext& ctx, int rows, int cols, int group, ScaleLayout layout, bool ue8m0,
              bool fast, const std::vector<float>& vals, int in_offset, int64_t q_stride,
              int64_t s_stride, int64_t s_count, PgBuffers* out) {
  DevBuf<T> in(static_cast<size_t>(rows) * cols + static_cast<size_t>(in_offset));
  DevBuf<uint8_t> q(static_cast<size_t>(rows) * static_cast<size_t>(q_stride));
  DevBuf<float> s(static_cast<size_t>(s_count));
  if (!in.ok() || !q.ok() || !s.ok()) return false;
  std::vector<T> staged(static_cast<size_t>(in_offset), T{});
  const std::vector<T> rounded = cast_from_float<T>(vals);
  staged.insert(staged.end(), rounded.begin(), rounded.end());
  kops_test::copy_in(in.p, staged);
  kops_test::copy_in(q.p, std::vector<uint8_t>(static_cast<size_t>(rows) * q_stride, 0xCDu));
  kops_test::copy_in(s.p, std::vector<float>(static_cast<size_t>(s_count), -1.0f));
  ActQuantOut o = make_out(q.p, q_stride, s.p, s_stride, DType::kFloat8E4M3, ActQuantMode::kPerGroup,
                           layout, group);
  o.scale_ue8m0 = ue8m0;
  o.math = fast ? kops::QuantMath::kReciprocal : kops::QuantMath::kExact;
  const Status st = kops::quant::quantize_activation(
      o, tensor2d(in.p + in_offset, pg_dtype<T>(), rows, cols), Tensor{}, 0, ctx);
  if (st != Status::kSuccess) {
    kops_test::record_failure("per-group dense", kops::last_error_message(), __FILE__, __LINE__);
    return false;
  }
  out->q = kops_test::copy_out(q.p, static_cast<size_t>(rows) * q_stride);
  out->s = kops_test::copy_out(s.p, static_cast<size_t>(s_count));
  return true;
}

/// Adversarial finite values: zeros, the e4m3 saturation and subnormal boundaries, the fp32
/// extremes and a random magnitude spanning 2^-30 .. 2^30.
std::vector<float> pg_adversarial(size_t n, unsigned seed) {
  static const float kFixed[] = {0.0f,  -0.0f, 1.0f,    -1.0f,   448.0f, -448.0f, 464.0f,
                                 -464.0f, 896.0f, 224.0f, 0x1p-9f, -0x1p-9f, 0x1p-10f,
                                 0x1p-10f * 1.5f, 0x1p-11f, 3.0e38f, -3.0e38f, 1e-10f,
                                 1e-30f, 1e30f, 0.1f,   -0.1f,  447.5f, 448.5f, 6.0f, 7.0f};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::uniform_int_distribution<int> expo(-30, 30);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    if (i % 5 == 0) {
      v[i] = kFixed[i % (sizeof(kFixed) / sizeof(kFixed[0]))];
    } else {
      v[i] = std::ldexp(unit(rng), expo(rng));
    }
  }
  return v;
}

/// Shapes 1 x group, one row and a dense grid, every dtype x group x layout. Checks the oracle,
/// the vectorized body against the scalar fallback (byte for byte) and the two arithmetic modes
/// against each other.
template <typename T>
void pg_shape_cases(LaunchContext& ctx, int group, ScaleLayout layout, int rows, int cols,
                    bool ue8m0, unsigned seed) {
  const int groups = cols / group;
  const int64_t padded_rows = (rows + 3) / 4 * 4;
  const bool col_major = layout == ScaleLayout::kColMajorTma;
  const int64_t s_stride = col_major ? padded_rows : groups;
  const int64_t s_count = col_major ? groups * padded_rows : int64_t(rows) * groups;
  std::vector<float> vals(static_cast<size_t>(rows) * cols);
  fill_uniform(vals, seed, -4.0f, 4.0f);
  const int64_t q_stride = cols;
  PgBuffers exact, fallback, fast;
  if (!pg_dense<T>(ctx, rows, cols, group, layout, ue8m0, false, vals, 0, q_stride, s_stride,
                   s_count, &exact)) {
    return;
  }
  if (!pg_dense<T>(ctx, rows, cols, group, layout, ue8m0, false, vals, 1, q_stride + 3, s_stride,
                   s_count, &fallback)) {
    return;
  }
  if (!pg_dense<T>(ctx, rows, cols, group, layout, ue8m0, true, vals, 0, q_stride, s_stride,
                   s_count, &fast)) {
    return;
  }
  std::vector<float> want_s;
  std::vector<uint8_t> want_q;
  pg_oracle(rounded_values(cast_from_float<T>(vals)), rows, cols, group, ue8m0, &want_s, &want_q);
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      const size_t off = col_major ? static_cast<size_t>(g) * padded_rows + r
                                   : static_cast<size_t>(r) * groups + g;
      KOPS_TEST_CHECK(exact.s[off] == want_s[static_cast<size_t>(r) * groups + g]);
      KOPS_TEST_CHECK(fallback.s[off] == want_s[static_cast<size_t>(r) * groups + g]);
      KOPS_TEST_CHECK(fast.s[off] == want_s[static_cast<size_t>(r) * groups + g]);
    }
    for (int c = 0; c < cols; ++c) {
      const uint8_t want = want_q[static_cast<size_t>(r) * cols + c];
      KOPS_TEST_CHECK(exact.q[static_cast<size_t>(r) * q_stride + c] == want);
      KOPS_TEST_CHECK(fallback.q[static_cast<size_t>(r) * (q_stride + 3) + c] == want);
      KOPS_TEST_CHECK(fast.q[static_cast<size_t>(r) * q_stride + c] == want);
    }
    for (int c = cols; c < cols + 3; ++c) {
      KOPS_TEST_CHECK(fallback.q[static_cast<size_t>(r) * (q_stride + 3) + c] == 0xCDu);
    }
  }
}

void test_per_group_vector(LaunchContext& ctx) {
  struct Shape {
    int rows;
    int cols;
  };
  const Shape shapes[] = {{1, 0}, {3, 0}, {6, 1024}};  // cols 0 => cols = group
  unsigned seed = 700;
  for (int dt = 0; dt < 3; ++dt) {
    for (int group : {32, 64, 128}) {
      for (const Shape& sh : shapes) {
        for (int ui = 0; ui < 2; ++ui) {
          for (int li = 0; li < 2; ++li) {
            const int cols = sh.cols == 0 ? group : sh.cols;
            const ScaleLayout layout =
                li == 0 ? ScaleLayout::kRowMajor : ScaleLayout::kColMajorTma;
            if (dt == 0) {
              pg_shape_cases<__nv_bfloat16>(ctx, group, layout, sh.rows, cols, ui != 0, seed++);
            } else if (dt == 1) {
              pg_shape_cases<__half>(ctx, group, layout, sh.rows, cols, ui != 0, seed++);
            } else {
              pg_shape_cases<float>(ctx, group, layout, sh.rows, cols, ui != 0, seed++);
            }
          }
        }
      }
    }
  }
}

/// Masked (DeepEP) layout with a padded expert stride: input.stride[0] > max_m * stride[1] and
/// stride[1] > cols. The reference is a dense per-group launch over the compacted live rows.
template <typename T>
void pg_masked_cases(LaunchContext& ctx, DType dtype, int group, int row_pad, int expert_pad,
                     bool ue8m0, bool fast, unsigned seed) {
  const int experts = 3, max_m = 4, cols = 256;
  const int groups = cols / group;
  const int64_t row_stride = cols + row_pad;
  const int64_t expert_stride = int64_t(max_m) * row_stride + expert_pad;
  const int64_t in_n = expert_stride * (experts - 1) + int64_t(max_m) * row_stride;
  std::vector<float> host(static_cast<size_t>(in_n), 0.0f);
  fill_uniform(host, seed, -3.0f, 5.0f);
  const std::vector<int32_t> live = {max_m, 2, max_m - 1};
  std::vector<float> compact;
  for (int e = 0; e < experts; ++e) {
    for (int m = 0; m < live[static_cast<size_t>(e)]; ++m) {
      for (int c = 0; c < cols; ++c) {
        compact.push_back(host[static_cast<size_t>(e) * expert_stride + m * row_stride + c]);
      }
    }
  }
  const int compact_rows = static_cast<int>(compact.size()) / cols;
  PgBuffers want;
  if (!pg_dense<T>(ctx, compact_rows, cols, group, ScaleLayout::kRowMajor, ue8m0, fast, compact, 0,
                   cols, groups, int64_t(compact_rows) * groups, &want)) {
    return;
  }

  const int64_t q_count = static_cast<int64_t>(experts) * max_m * cols;
  const int64_t s_count = static_cast<int64_t>(experts) * max_m * groups;
  DevBuf<T> in(static_cast<size_t>(in_n));
  DevBuf<uint8_t> q(static_cast<size_t>(q_count));
  DevBuf<float> s(static_cast<size_t>(s_count));
  DevBuf<int32_t> mm(experts);
  if (!in.ok() || !q.ok() || !s.ok() || !mm.ok()) return;
  kops_test::copy_in(in.p, cast_from_float<T>(host));
  kops_test::copy_in(q.p, std::vector<uint8_t>(static_cast<size_t>(q_count), 0xCDu));
  kops_test::copy_in(s.p, std::vector<float>(static_cast<size_t>(s_count), -1.0f));
  kops_test::copy_in(mm.p, live);
  Tensor t3{};
  t3.data = in.p;
  t3.dtype = dtype;
  t3.ndim = 3;
  t3.device = 0;
  t3.shape[0] = experts;
  t3.shape[1] = max_m;
  t3.shape[2] = cols;
  t3.stride[0] = expert_stride;
  t3.stride[1] = row_stride;
  t3.stride[2] = 1;
  ActQuantOut o = make_out(q.p, cols, s.p, groups, DType::kFloat8E4M3, ActQuantMode::kPerGroup,
                           ScaleLayout::kRowMajor, group);
  o.scale_row_stride = groups;
  o.scale_ue8m0 = ue8m0;
  o.math = fast ? kops::QuantMath::kReciprocal : kops::QuantMath::kExact;
  KOPS_EXPECT_OK(kops::quant::quantize_activation(o, t3, tensor1d(mm.p, DType::kInt32, experts),
                                                   max_m, ctx),
                 "masked per-group padded expert stride");
  const std::vector<uint8_t> gq = kops_test::copy_out(q.p, static_cast<size_t>(q_count));
  const std::vector<float> gs = kops_test::copy_out(s.p, static_cast<size_t>(s_count));
  int ref_row = 0;
  for (int e = 0; e < experts; ++e) {
    for (int m = 0; m < max_m; ++m) {
      const int64_t out_row = static_cast<int64_t>(e) * max_m + m;
      if (m < live[static_cast<size_t>(e)]) {
        for (int g = 0; g < groups; ++g) {
          KOPS_TEST_CHECK(gs[static_cast<size_t>(out_row) * groups + g] ==
                          want.s[static_cast<size_t>(ref_row) * groups + g]);
        }
        for (int c = 0; c < cols; ++c) {
          KOPS_TEST_CHECK(gq[static_cast<size_t>(out_row) * cols + c] ==
                          want.q[static_cast<size_t>(ref_row) * cols + c]);
        }
        ++ref_row;
      } else {
        // A skipped row is never touched, so its payload and scales keep the sentinel.
        for (int c = 0; c < cols; ++c) {
          KOPS_TEST_CHECK(gq[static_cast<size_t>(out_row) * cols + c] == 0xCDu);
        }
        for (int g = 0; g < groups; ++g) {
          KOPS_TEST_CHECK(gs[static_cast<size_t>(out_row) * groups + g] == -1.0f);
        }
      }
    }
  }
}

void test_per_group_masked(LaunchContext& ctx) {
  unsigned seed = 800;
  for (int dt = 0; dt < 3; ++dt) {
    for (int group : {32, 64, 128}) {
      // 8/4-element pads keep every row start and the expert stride 16 B aligned (vector path);
      // 1/3-element pads force the scalar fallback.
      for (int pad = 0; pad < 2; ++pad) {
        const int row_pad = pad == 0 ? 16 / (dt == 2 ? 4 : 2) : 1;
        const int expert_pad = pad == 0 ? row_pad : 3;
        for (int fast = 0; fast < 2; ++fast) {
          if (dt == 0) {
            pg_masked_cases<__nv_bfloat16>(ctx, DType::kBFloat16, group, row_pad, expert_pad, false,
                                           fast != 0, seed++);
          } else if (dt == 1) {
            pg_masked_cases<__half>(ctx, DType::kFloat16, group, row_pad, expert_pad, false,
                                    fast != 0, seed++);
          } else {
            pg_masked_cases<float>(ctx, DType::kFloat32, group, row_pad, expert_pad, false,
                                   fast != 0, seed++);
          }
        }
      }
    }
  }
}

/// Exact vs reciprocal over adversarial data: inf/NaN groups (the reciprocal falls back to the
/// exact body), the saturation and subnormal boundaries, and every 16-bit bit pattern.
template <typename T>
void pg_math_compare(LaunchContext& ctx, DType dtype, int group, const std::vector<float>& vals) {
  const int rows = static_cast<int>(vals.size()) / group;
  const int groups = 1;
  PgBuffers exact, fast;
  if (!pg_dense<T>(ctx, rows, group, group, ScaleLayout::kRowMajor, false, false, vals, 0, group,
                   groups, rows, &exact)) {
    return;
  }
  if (!pg_dense<T>(ctx, rows, group, group, ScaleLayout::kRowMajor, false, true, vals, 0, group,
                   groups, rows, &fast)) {
    return;
  }
  size_t mismatch = 0;
  for (size_t i = 0; i < exact.q.size(); ++i) {
    if (exact.q[i] != fast.q[i]) ++mismatch;
  }
  KOPS_TEST_CHECK(mismatch == 0);
  for (size_t i = 0; i < exact.s.size(); ++i) KOPS_TEST_CHECK(exact.s[i] == fast.s[i]);
  (void)dtype;
}

std::vector<float> pg_sweep_16(DType dtype) {
  const float anchors[] = {1.0f, 1e-3f, 1e5f, 0.02f};
  std::vector<float> vals;
  for (float a : anchors) {
    vals.push_back(a);
    for (uint32_t bits = 0; bits < 65536u; ++bits) {
      vals.push_back(dtype == DType::kBFloat16
                         ? kops_test::to_float(pg_bitcast<__nv_bfloat16>(bits))
                         : kops_test::to_float(pg_bitcast<__half>(bits)));
    }
  }
  while (vals.size() % 128 != 0) vals.push_back(0.0f);
  return vals;
}

std::vector<float> pg_sweep_fp32(unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<float> vals;
  for (int e : {-40, -20, -6, -1, 0, 1, 6, 20, 40}) {
    vals.push_back(std::ldexp(1.0f, e));
    for (int i = 0; i < 65536; ++i) {
      const uint32_t mantissa = rng() & 0x807FFFFFu;
      const int exponent = -60 + static_cast<int>(rng() % 121u);
      vals.push_back(pg_u32_as_float(mantissa | (static_cast<uint32_t>(exponent + 127) << 23)));
    }
  }
  while (vals.size() % 128 != 0) vals.push_back(0.0f);
  return vals;
}

void test_per_group_math(LaunchContext& ctx) {
  unsigned seed = 900;
  for (int group : {32, 128}) {
    // Adversarial finite data plus a dedicated inf/NaN group per case.
    for (int dt = 0; dt < 3; ++dt) {
      std::vector<float> vals = pg_adversarial(size_t(64) * group, seed++);
      const float specials[] = {std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()};
      for (float sp : specials) {
        for (int j = 0; j < group; ++j) vals.push_back(j == 0 ? sp : 1.0f);
      }
      if (dt == 0) {
        pg_math_compare<__nv_bfloat16>(ctx, DType::kBFloat16, group, vals);
      } else if (dt == 1) {
        pg_math_compare<__half>(ctx, DType::kFloat16, group, vals);
      } else {
        pg_math_compare<float>(ctx, DType::kFloat32, group, vals);
      }
    }
    pg_math_compare<__nv_bfloat16>(ctx, DType::kBFloat16, group,
                                   pg_sweep_16(DType::kBFloat16));
    pg_math_compare<__half>(ctx, DType::kFloat16, group, pg_sweep_16(DType::kFloat16));
    pg_math_compare<float>(ctx, DType::kFloat32, group, pg_sweep_fp32(seed++));
  }
}


/// Graph capture/replay of both per-group arithmetic modes: replaying with different data must
/// reproduce the oracle (and therefore a fresh direct launch) exactly.
template <typename T>
void pg_graph_case(LaunchContext& ctx, bool fast, int rows, int cols, int group) {
  const int groups = cols / group;
  DevBuf<T> in(static_cast<size_t>(rows) * cols);
  DevBuf<uint8_t> q(static_cast<size_t>(rows) * cols);
  DevBuf<float> s(static_cast<size_t>(rows) * groups);
  if (!in.ok() || !q.ok() || !s.ok()) return;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return;
  LaunchContext gc = ctx;
  gc.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  ActQuantOut o = make_out(q.p, cols, s.p, groups, DType::kFloat8E4M3, ActQuantMode::kPerGroup,
                           ScaleLayout::kRowMajor, group);
  o.math = fast ? kops::QuantMath::kReciprocal : kops::QuantMath::kExact;
  const Tensor tin = tensor2d(in.p, pg_dtype<T>(), rows, cols);
  std::vector<float> v0(static_cast<size_t>(rows) * cols);
  fill_uniform(v0, 910, -2.0f, 2.0f);
  kops_test::copy_in(in.p, cast_from_float<T>(v0));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  KOPS_EXPECT_OK(kops::quant::quantize_activation(o, tin, Tensor{}, 0, gc),
                 "per-group graph capture");
  cudaStreamEndCapture(stream, &graph);
  if (graph != nullptr && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
    for (int round = 0; round < 2; ++round) {
      std::vector<float> v(static_cast<size_t>(rows) * cols);
      fill_uniform(v, 911 + round, -4.0f * (round + 1), 6.0f * (round + 1));
      kops_test::copy_in(in.p, cast_from_float<T>(v));
      cudaGraphLaunch(exec, stream);
      cudaStreamSynchronize(stream);
      const std::vector<uint8_t> gq = kops_test::copy_out(q.p, static_cast<size_t>(rows) * cols);
      const std::vector<float> gs = kops_test::copy_out(s.p, static_cast<size_t>(rows) * groups);
      std::vector<float> want_s;
      std::vector<uint8_t> want_q;
      pg_oracle(rounded_values(cast_from_float<T>(v)), rows, cols, group, false, &want_s, &want_q);
      for (size_t i = 0; i < gq.size(); ++i) KOPS_TEST_CHECK(gq[i] == want_q[i]);
      for (size_t i = 0; i < gs.size(); ++i) KOPS_TEST_CHECK(gs[i] == want_s[i]);
    }
    cudaGraphExecDestroy(exec);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

void test_per_group_graph(LaunchContext& ctx) {
  for (int fast = 0; fast < 2; ++fast) {
    for (int group : {32, 128}) {
      pg_graph_case<__nv_bfloat16>(ctx, fast != 0, 8, 512, group);
      pg_graph_case<float>(ctx, fast != 0, 1, 256, group);
    }
  }
}

void test_per_group_contracts(LaunchContext& ctx) {
  const int rows = 2, cols = 256, group = 128, groups = 2;
  DevBuf<__nv_bfloat16> in(static_cast<size_t>(rows) * cols);
  DevBuf<uint8_t> q(static_cast<size_t>(rows) * cols);
  DevBuf<int8_t> qi8(static_cast<size_t>(rows) * cols);
  DevBuf<float> s(static_cast<size_t>(rows) * (cols / 32));
  if (!in.ok() || !q.ok() || !qi8.ok() || !s.ok()) return;
  std::vector<float> host(static_cast<size_t>(rows) * cols, 1.0f);
  kops_test::copy_in(in.p, cast_from_float<__nv_bfloat16>(host));
  const Tensor tin = tensor2d(in.p, DType::kBFloat16, rows, cols);

  // Scale layouts the per-group bodies do not implement: an error, never a silent row-major write.
  for (ScaleLayout layout : {ScaleLayout::kSwizzled128x4, ScaleLayout::kPackedUE8M0x4,
                             ScaleLayout::kMarlinPermuted, ScaleLayout::kInlineWithData}) {
    ActQuantOut o = make_out(q.p, cols, s.p, groups, DType::kFloat8E4M3, ActQuantMode::kPerGroup,
                             layout, group);
    KOPS_EXPECT_STATUS(kops::quant::quantize_activation(o, tin, Tensor{}, 0, ctx),
                       Status::kUnsupportedLayout, "per-group unsupported scale layout");
  }
  // The opt-in arithmetic is per-group only.
  ActQuantOut tok = make_out(q.p, cols, s.p, 1, DType::kFloat8E4M3,
                             ActQuantMode::kDynamicPerToken);
  tok.scales = s.p;
  tok.math = kops::QuantMath::kReciprocal;
  KOPS_EXPECT_STATUS(kops::quant::quantize_activation(tok, tin, Tensor{}, 0, ctx),
                     Status::kUnsupportedLayout, "reciprocal is per-group only");
  // INT8 per-group payload is not compiled, group sizes are 32/64/128.
  ActQuantOut i8 = make_out(qi8.p, cols, s.p, groups, DType::kInt8, ActQuantMode::kPerGroup,
                            ScaleLayout::kRowMajor, group);
  KOPS_EXPECT_STATUS(kops::quant::quantize_activation(i8, tin, Tensor{}, 0, ctx),
                     Status::kUnsupportedDtype, "per-group INT8");
  ActQuantOut g16 = make_out(q.p, cols, s.p, groups, DType::kFloat8E4M3, ActQuantMode::kPerGroup,
                             ScaleLayout::kRowMajor, 16);
  KOPS_EXPECT_STATUS(kops::quant::quantize_activation(g16, tin, Tensor{}, 0, ctx),
                     Status::kUnsupportedShape, "per-group group size 16");
}

template <typename T>
std::vector<T> pad_rows(const std::vector<T>& src, int rows, int cols, int pad) {
  std::vector<T> out(static_cast<size_t>(rows) * (cols + pad), T{});
  for (int r = 0; r < rows; ++r) {
    std::copy_n(src.begin() + static_cast<size_t>(r) * cols, cols,
                out.begin() + static_cast<size_t>(r) * (cols + pad));
  }
  return out;
}

struct EdgeCase {
  int rows;
  int cols;
  int in_pad;
  int out_pad;
  float scale;
};

template <typename T>
void run_per_tensor_edge(LaunchContext& ctx, DType dtype, bool fp8, const EdgeCase& c,
                         unsigned seed) {
  const int64_t in_stride = c.cols + c.in_pad;
  const int64_t out_stride = c.cols + c.out_pad;
  std::vector<float> host(static_cast<size_t>(c.rows) * c.cols);
  fill_uniform(host, seed, -5.0f, 5.0f);
  const std::vector<T> rounded = cast_from_float<T>(host);
  const std::vector<float> vals = rounded_values(rounded);
  const size_t q_count = static_cast<size_t>(c.rows) * out_stride;
  DevBuf<T> in(static_cast<size_t>(c.rows) * in_stride);
  DevBuf<uint8_t> q(q_count);
  DevBuf<float> scale(1);
  if (!in.ok() || !q.ok() || !scale.ok()) return;
  kops_test::copy_in(in.p, pad_rows(rounded, c.rows, c.cols, c.in_pad));
  kops_test::copy_in(scale.p, std::vector<float>{c.scale});
  const std::vector<uint8_t> sentinel(q_count, 0xCDu);
  kops_test::copy_in(q.p, sentinel);
  ActQuantOut out = make_out(q.p, out_stride, nullptr, 0,
                             fp8 ? DType::kFloat8E4M3 : DType::kInt8,
                             ActQuantMode::kStaticPerTensor);
  out.static_scale = scale.p;
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     out, tensor2d(in.p, dtype, c.rows, c.cols, in_stride), Tensor{}, 0, ctx),
                 "per-tensor pack edge");
  const std::vector<uint8_t> got = kops_test::copy_out(q.p, q_count);
  for (int r = 0; r < c.rows; ++r) {
    for (int col = 0; col < c.cols; ++col) {
      const float x = vals[static_cast<size_t>(r) * c.cols + col];
      uint8_t want;
      if (fp8) {
        want = e4m3_byte(x / c.scale);
      } else {
        int v = static_cast<int>(std::nearbyint(x / c.scale));
        v = std::max(-127, std::min(127, v));
        want = static_cast<uint8_t>(static_cast<int8_t>(v));
      }
      KOPS_TEST_CHECK(got[static_cast<size_t>(r) * out_stride + col] == want);
    }
    for (int col = c.cols; col < out_stride; ++col) {
      KOPS_TEST_CHECK(got[static_cast<size_t>(r) * out_stride + col] == 0xCDu);  // row padding
    }
  }
}

template <typename T>
void run_per_token_edge(LaunchContext& ctx, DType dtype, bool fp8, const EdgeCase& c,
                        bool use_ub, unsigned seed) {
  const int64_t in_stride = c.cols + c.in_pad;
  const int64_t out_stride = c.cols + c.out_pad;
  std::vector<float> host(static_cast<size_t>(c.rows) * c.cols);
  fill_uniform(host, seed, -6.0f, 5.0f);
  const std::vector<T> rounded = cast_from_float<T>(host);
  const std::vector<float> vals = rounded_values(rounded);
  const size_t q_count = static_cast<size_t>(c.rows) * out_stride;
  DevBuf<T> in(static_cast<size_t>(c.rows) * in_stride);
  DevBuf<uint8_t> q(q_count);
  DevBuf<float> scales(c.rows), ub(1);
  DevBuf<int32_t> zero(c.rows);
  if (!in.ok() || !q.ok() || !scales.ok() || !ub.ok() || !zero.ok()) return;
  kops_test::copy_in(in.p, pad_rows(rounded, c.rows, c.cols, c.in_pad));
  const float ub_value = 0.005f;  // below amax/448 for the [-6, 5] data: always binds
  kops_test::copy_in(ub.p, std::vector<float>{ub_value});
  const std::vector<uint8_t> sentinel(q_count, 0xCDu);
  kops_test::copy_in(q.p, sentinel);
  ActQuantOut out = make_out(q.p, out_stride, scales.p, 1,
                             fp8 ? DType::kFloat8E4M3 : DType::kInt8,
                             ActQuantMode::kDynamicPerToken);
  out.symmetric = fp8;
  if (!fp8) out.zero_points = zero.p;
  out.scale_upper_bound = use_ub ? ub.p : nullptr;
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     out, tensor2d(in.p, dtype, c.rows, c.cols, in_stride), Tensor{}, 0, ctx),
                 "per-token pack edge");
  const std::vector<uint8_t> got = kops_test::copy_out(q.p, q_count);
  const std::vector<float> gs = kops_test::copy_out(scales.p, c.rows);
  const std::vector<int32_t> gz = kops_test::copy_out(zero.p, c.rows);
  for (int r = 0; r < c.rows; ++r) {
    float mx = -1e30f, mn = 1e30f;
    for (int col = 0; col < c.cols; ++col) {
      const float x = vals[static_cast<size_t>(r) * c.cols + col];
      mx = std::max(mx, x);
      mn = std::min(mn, x);
    }
    float s;
    int z = 0;
    if (fp8) {
      const float amax = std::max(std::fabs(mx), std::fabs(mn));
      s = std::fmax(amax / 448.0f, 1.0f / (448.0f * 512.0f));
      if (use_ub) s = std::fmin(s, ub_value);
    } else {
      s = (mx - mn) / 255.0f;
      s = std::fmax(s, 1.0f / (127.0f * 512.0f));
      z = static_cast<int>(std::nearbyint(-128.0f - mn / s));
      KOPS_TEST_CHECK(gz[static_cast<size_t>(r)] == z);
    }
    KOPS_TEST_CHECK(std::fabs(gs[static_cast<size_t>(r)] - s) <= 1e-7f + 1e-6f * s);
    for (int col = 0; col < c.cols; ++col) {
      const float x = vals[static_cast<size_t>(r) * c.cols + col];
      uint8_t want;
      if (fp8) {
        want = e4m3_byte(x / s);
      } else {
        int v = static_cast<int>(std::nearbyint(x / s + static_cast<float>(z)));
        v = std::max(-128, std::min(127, v));
        want = static_cast<uint8_t>(static_cast<int8_t>(v));
      }
      KOPS_TEST_CHECK(got[static_cast<size_t>(r) * out_stride + col] == want);
    }
    for (int col = c.cols; col < out_stride; ++col) {
      KOPS_TEST_CHECK(got[static_cast<size_t>(r) * out_stride + col] == 0xCDu);
    }
  }
}

void test_quant_pack_edges(LaunchContext& ctx) {
  const EdgeCase tensor_cases[] = {
      {1, 1, 0, 0, 1.0f},        // one element, power-of-two scale
      {3, 7, 0, 0, 0.0125f},     // shorter than one pack
      {2, 17, 4, 3, 1.0e-3f},    // padded input and output rows
      {5, 1031, 0, 1, 1.0f},     // not a multiple of the pack
      {4, 4095, 2, 0, 0.02f},    // just below a pack-aligned row
      {3, 4096, 0, 0, 0.0125f},  // vector fast path
  };
  unsigned seed = 301;
  for (const EdgeCase& c : tensor_cases) {
    run_per_tensor_edge<__nv_bfloat16>(ctx, DType::kBFloat16, true, c, seed++);
    run_per_tensor_edge<__nv_bfloat16>(ctx, DType::kBFloat16, false, c, seed++);
    run_per_tensor_edge<__half>(ctx, DType::kFloat16, true, c, seed++);
    run_per_tensor_edge<__half>(ctx, DType::kFloat16, false, c, seed++);
    run_per_tensor_edge<float>(ctx, DType::kFloat32, true, c, seed++);
    run_per_tensor_edge<float>(ctx, DType::kFloat32, false, c, seed++);
  }
  const EdgeCase token_cases[] = {
      {1, 1, 0, 0, 0.0f},       // single value
      {5, 17, 0, 1, 0.0f},      // warp path
      {4, 33, 2, 0, 0.0f},      // warp path, padded input
      {3, 1024, 0, 0, 0.0f},    // warp path boundary
      {2, 1025, 0, 3, 0.0f},    // smem fallback (not pack aligned)
      {4, 1031, 0, 1, 0.0f},    // smem fallback, odd width
      {2, 4095, 0, 0, 0.0f},    // smem fallback, just below aligned
      {3, 4096, 0, 0, 0.0f},    // vector fast path
      {2, 4097, 2, 0, 0.0f},    // smem fallback, padded input
  };
  for (const EdgeCase& c : token_cases) {
    run_per_token_edge<__nv_bfloat16>(ctx, DType::kBFloat16, true, c, false, seed++);
    run_per_token_edge<__nv_bfloat16>(ctx, DType::kBFloat16, true, c, true, seed++);
    run_per_token_edge<__nv_bfloat16>(ctx, DType::kBFloat16, false, c, false, seed++);
    run_per_token_edge<__half>(ctx, DType::kFloat16, true, c, false, seed++);
    run_per_token_edge<__half>(ctx, DType::kFloat16, false, c, false, seed++);
    run_per_token_edge<float>(ctx, DType::kFloat32, true, c, false, seed++);
    run_per_token_edge<float>(ctx, DType::kFloat32, false, c, false, seed++);
  }
}

// -----------------------------------------------------------------------------
// MX / NVFP4
// -----------------------------------------------------------------------------

void test_mx(LaunchContext& ctx) {
  const int rows = 5, cols = 256;
  const int groups = cols / 32;
  DevBuf<__nv_bfloat16> in(rows * cols);
  DevBuf<uint8_t> q(rows * cols);
  DevBuf<uint8_t> q4(rows * cols / 2);
  DevBuf<uint8_t> s8(rows * groups);
  DevBuf<uint8_t> s4(rows * groups);
  if (!in.ok() || !q.ok() || !q4.ok() || !s8.ok() || !s4.ok()) return;
  std::vector<float> host(rows * cols);
  fill_uniform(host, 51, -8.0f, 8.0f);
  auto half = cast_from_float<__nv_bfloat16>(host);
  kops_test::copy_in(in.p, half);
  ActQuantOut mx8 = make_out(q.p, cols, s8.p, groups, DType::kFloat8E4M3,
                             ActQuantMode::kMXFP8, ScaleLayout::kRowMajor);
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     mx8, tensor2d(in.p, DType::kBFloat16, rows, cols), Tensor{}, 0, ctx),
                 "mxfp8");
  ActQuantOut mx4 = make_out(q4.p, cols / 2, s4.p, groups, DType::kFloat4E2M1x2,
                             ActQuantMode::kMXFP4, ScaleLayout::kRowMajor);
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     mx4, tensor2d(in.p, DType::kBFloat16, rows, cols), Tensor{}, 0, ctx),
                 "mxfp4");
  const auto got8 = kops_test::copy_out(q.p, rows * cols);
  const auto got4 = kops_test::copy_out(q4.p, rows * cols / 2);
  const auto gs8 = kops_test::copy_out(s8.p, rows * groups);
  const auto gs4 = kops_test::copy_out(s4.p, rows * groups);
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      float amax = 0.0f;
      for (int j = 0; j < 32; ++j) {
        amax = std::max(amax,
                        std::fabs(kops_test::to_float(half[static_cast<size_t>(r * cols + g * 32 + j)])));
      }
      auto ue8 = [](float amax, float qmax) {
        float s = std::max(amax, 1e-10f) / qmax;
        int e = 0;
        std::frexp(s, &e);
        e -= 1;
        if (std::ldexp(1.0f, e) < s) ++e;
        return e;
      };
      const int e8 = ue8(amax, 448.0f);
      const int e4 = ue8(amax, 6.0f);
      KOPS_TEST_CHECK(gs8[static_cast<size_t>(r * groups + g)] == static_cast<uint8_t>(e8 + 127));
      KOPS_TEST_CHECK(gs4[static_cast<size_t>(r * groups + g)] == static_cast<uint8_t>(e4 + 127));
      const float s8v = std::ldexp(1.0f, e8);
      const float s4v = std::ldexp(1.0f, e4);
      for (int j = 0; j < 32; ++j) {
        const float x = kops_test::to_float(half[static_cast<size_t>(r * cols + g * 32 + j)]);
        KOPS_TEST_CHECK(got8[static_cast<size_t>(r * cols + g * 32 + j)] == e4m3_byte(x / s8v));
      }
      for (int j = 0; j < 16; ++j) {
        const float a = kops_test::to_float(half[static_cast<size_t>(r * cols + g * 32 + 2 * j)]);
        const float b = kops_test::to_float(half[static_cast<size_t>(r * cols + g * 32 + 2 * j + 1)]);
        KOPS_TEST_CHECK(got4[static_cast<size_t>(r * (cols / 2) + g * 16 + j)] ==
                        e2m1_pair(a / s4v, b / s4v));
      }
    }
  }
}

void test_nvfp4(LaunchContext& ctx) {
  const int rows = 4, cols = 128;
  const int groups = cols / 16;
  DevBuf<__nv_bfloat16> in(rows * cols);
  DevBuf<uint8_t> q(rows * cols / 2);
  const int64_t sf_bytes = ((rows + 127) / 128) * ((groups + 3) / 4) * 512;
  DevBuf<uint8_t> scales(sf_bytes);
  DevBuf<float> gscale(1);
  if (!in.ok() || !q.ok() || !scales.ok() || !gscale.ok()) return;
  std::vector<float> host(rows * cols);
  fill_uniform(host, 61, -5.0f, 5.0f);
  auto half = cast_from_float<__nv_bfloat16>(host);
  kops_test::copy_in(in.p, half);
  float amax = 0.0f;
  for (float v : host) amax = std::max(amax, std::fabs(v));
  const float gs = 448.0f * 6.0f / amax;
  kops_test::copy_in(gscale.p, std::vector<float>{gs});
  ActQuantOut out = make_out(q.p, cols / 2, scales.p, groups, DType::kFloat4E2M1x2,
                             ActQuantMode::kNVFP4, ScaleLayout::kSwizzled128x4);
  out.global_scale = gscale.p;
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     out, tensor2d(in.p, DType::kBFloat16, rows, cols), Tensor{}, 0, ctx),
                 "nvfp4");
  const auto got = kops_test::copy_out(q.p, rows * cols / 2);
  const auto gsc = kops_test::copy_out(scales.p, sf_bytes);
  auto swizzle = [](int row, int group, int groups_) {
    const int m_tile = row >> 7;
    const int outer_m = row & 31;
    const int inner_m = (row >> 5) & 3;
    const int k_tile = group >> 2;
    const int inner_k = group & 3;
    const int k_tiles = (groups_ + 3) >> 2;
    return ((m_tile * k_tiles + k_tile) << 9) + (outer_m << 4) + (inner_m << 2) + inner_k;
  };
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      float bmax = 0.0f;
      for (int j = 0; j < 16; ++j) {
        bmax = std::max(bmax,
                        std::fabs(kops_test::to_float(half[static_cast<size_t>(r * cols + g * 16 + j)])));
      }
      const float sf = bmax * gs / 6.0f;
      const uint8_t sf_byte = e4m3_byte(sf);
      KOPS_TEST_CHECK(gsc[static_cast<size_t>(swizzle(r, g, groups))] == sf_byte);
      const float sfv = e4m3_round(sf);
      for (int j = 0; j < 8; ++j) {
        const float a = kops_test::to_float(
            half[static_cast<size_t>(r * cols + g * 16 + 2 * j)]);
        const float b = kops_test::to_float(
            half[static_cast<size_t>(r * cols + g * 16 + 2 * j + 1)]);
        KOPS_TEST_CHECK(got[static_cast<size_t>(r * (cols / 2) + g * 8 + j)] ==
                        e2m1_pair(a * gs / sfv, b * gs / sfv));
      }
    }
  }
}

// -----------------------------------------------------------------------------
// dequantize / int4 / hadamard
// -----------------------------------------------------------------------------

void test_dequantize(LaunchContext& ctx) {
  const int rows = 4, cols = 256;
  DevBuf<__nv_bfloat16> in(rows * cols);
  DevBuf<uint8_t> q(rows * cols);
  DevBuf<float> scales(rows);
  DevBuf<float> out(rows * cols);
  if (!in.ok() || !q.ok() || !scales.ok() || !out.ok()) return;
  std::vector<float> host(rows * cols);
  fill_uniform(host, 71);
  auto half = cast_from_float<__nv_bfloat16>(host);
  kops_test::copy_in(in.p, half);
  ActQuantOut qo = make_out(q.p, cols, scales.p, 1, DType::kFloat8E4M3,
                            ActQuantMode::kDynamicPerToken);
  KOPS_EXPECT_OK(kops::quant::quantize_activation(
                     qo, tensor2d(in.p, DType::kBFloat16, rows, cols), Tensor{}, 0, ctx),
                 "dequant setup");
  QuantScheme scheme{};
  scheme.storage_dtype = DType::kFloat8E4M3;
  scheme.scale_dtype = DType::kFloat32;
  scheme.granularity = ScaleGranularity::kPerToken;
  KOPS_EXPECT_OK(kops::quant::dequantize(
                     tensor2d(out.p, DType::kFloat32, rows, cols),
                     tensor2d(q.p, DType::kFloat8E4M3, rows, cols),
                     tensor1d(scales.p, DType::kFloat32, rows), Tensor{}, scheme, ctx),
                 "dequant per-token");
  const auto got = kops_test::copy_out(out.p, rows * cols);
  const auto gq = kops_test::copy_out(q.p, rows * cols);
  const auto gs = kops_test::copy_out(scales.p, rows);
  for (int i = 0; i < rows * cols; ++i) {
    const int r = i / cols;
    const float want = __half2float(__nv_cvt_fp8_to_halfraw(gq[static_cast<size_t>(i)], __NV_E4M3)) *
                       gs[static_cast<size_t>(r)];
    KOPS_TEST_CHECK(std::fabs(got[static_cast<size_t>(i)] - want) <= 1e-7f + 1e-6f * std::fabs(want));
  }
}

void test_int4_pack(LaunchContext& ctx) {
  const int rows = 3, cols = 256;
  DevBuf<int8_t> in(rows * cols);
  DevBuf<uint8_t> packed(rows * cols / 2);
  DevBuf<int8_t> back(rows * cols);
  if (!in.ok() || !packed.ok() || !back.ok()) return;
  std::vector<int8_t> host(rows * cols);
  std::mt19937 rng(81);
  std::uniform_int_distribution<int> dist(-8, 7);
  for (int8_t& v : host) v = static_cast<int8_t>(dist(rng));
  kops_test::copy_in(in.p, host);
  KOPS_EXPECT_OK(kops::quant::pack_int4(tensor2d(packed.p, DType::kUInt8, rows, cols / 2),
                                        tensor2d(in.p, DType::kInt8, rows, cols), false, ctx),
                 "pack_int4");
  const auto got = kops_test::copy_out(packed.p, rows * cols / 2);
  for (int i = 0; i < rows * cols / 2; ++i) {
    const uint8_t want = static_cast<uint8_t>((host[2 * i] & 15) | ((host[2 * i + 1] & 15) << 4));
    KOPS_TEST_CHECK(got[static_cast<size_t>(i)] == want);
  }
  KOPS_EXPECT_OK(kops::quant::unpack_int4(tensor2d(back.p, DType::kInt8, rows, cols),
                                          tensor2d(packed.p, DType::kUInt8, rows, cols / 2),
                                          false, ctx),
                 "unpack_int4");
  const auto round = kops_test::copy_out(back.p, rows * cols);
  KOPS_TEST_CHECK(std::memcmp(round.data(), host.data(), host.size()) == 0);
}

void test_hadamard(LaunchContext& ctx) {
  const int rows = 3, n = 256;
  DevBuf<float> in(rows * n);
  DevBuf<float> out(rows * n);
  if (!in.ok() || !out.ok()) return;
  std::vector<float> host(rows * n);
  fill_uniform(host, 91);
  kops_test::copy_in(in.p, host);
  const float scale = 1.0f / std::sqrt(static_cast<float>(n));
  KOPS_EXPECT_OK(kops::quant::hadamard_transform(
                     tensor2d(out.p, DType::kFloat32, rows, n),
                     tensor2d(in.p, DType::kFloat32, rows, n), scale, false, 0, ctx),
                 "hadamard");
  const auto got = kops_test::copy_out(out.p, rows * n);
  // Reference Walsh-Hadamard in fp64 (natural order).
  for (int r = 0; r < rows; ++r) {
    std::vector<double> h(host.begin() + r * n, host.begin() + (r + 1) * n);
    for (int len = 1; len < n; len <<= 1) {
      for (int i = 0; i < n; i += 2 * len) {
        for (int j = 0; j < len; ++j) {
          const double a = h[static_cast<size_t>(i + j)];
          const double b = h[static_cast<size_t>(i + j + len)];
          h[static_cast<size_t>(i + j)] = a + b;
          h[static_cast<size_t>(i + j + len)] = a - b;
        }
      }
    }
    for (int j = 0; j < n; ++j) {
      KOPS_TEST_CHECK(std::fabs(got[static_cast<size_t>(r * n + j)] - h[static_cast<size_t>(j)] * scale) <=
                      2e-4f + 1e-4f * std::fabs(h[static_cast<size_t>(j)] * scale));
    }
  }
  // In-place allowed.
  KOPS_EXPECT_OK(kops::quant::hadamard_transform(
                     tensor2d(in.p, DType::kFloat32, rows, n),
                     tensor2d(in.p, DType::kFloat32, rows, n), scale, false, 0, ctx),
                 "hadamard in place");
}

// -----------------------------------------------------------------------------
// Contracts and graph replay
// -----------------------------------------------------------------------------

void test_contracts(LaunchContext& ctx) {
  DevBuf<float> in(64);
  DevBuf<uint8_t> q(64);
  DevBuf<float> scales(2);
  std::vector<float> host(64, 1.0f);
  kops_test::copy_in(in.p, host);
  ActQuantOut out = make_out(q.p, 8, nullptr, 0, DType::kFloat8E4M3,
                             ActQuantMode::kStaticPerTensor);
  KOPS_EXPECT_STATUS(kops::quant::quantize_activation(
                         out, tensor2d(in.p, DType::kFloat32, 8, 8), Tensor{}, 0, ctx),
                     Status::kInvalidArgument, "static without scale");
  ActQuantOut dyn = make_out(q.p, 8, scales.p, 1, DType::kFloat8E4M3,
                             ActQuantMode::kDynamicPerToken);
  DevBuf<int32_t> mask(2);
  std::vector<int32_t> mask_host = {0, 0};
  kops_test::copy_in(mask.p, mask_host);
  Tensor masked3 = tensor2d(in.p, DType::kFloat32, 2, 8);
  masked3.ndim = 3;
  masked3.shape[0] = 2;
  masked3.shape[1] = 4;
  masked3.shape[2] = 8;
  masked3.stride[0] = 32;
  masked3.stride[1] = 8;
  masked3.stride[2] = 1;
  KOPS_EXPECT_STATUS(kops::quant::quantize_activation(
                         dyn, masked3, tensor1d(mask.p, DType::kInt32, 2), 4, ctx),
                     Status::kUnsupportedLayout, "masked only for per-group");
  KOPS_EXPECT_STATUS(kops::quant::absmax(tensor1d(scales.p, DType::kInt32, 1),
                                         tensor2d(in.p, DType::kFloat32, 8, 8), ctx),
                     Status::kUnsupportedDtype, "absmax dtype");
  KOPS_EXPECT_STATUS(kops::quant::pack_int4(tensor2d(q.p, DType::kUInt8, 4, 4),
                                            tensor2d(in.p, DType::kFloat32, 8, 8), false, ctx),
                     Status::kUnsupportedDtype, "pack dtype");
}

void test_graph_replay(LaunchContext& ctx) {
  const int rows = 8, cols = 512;
  DevBuf<float> in(rows * cols);
  DevBuf<uint8_t> q(rows * cols);
  DevBuf<float> scales(rows);
  std::vector<float> host(rows * cols, 0.5f);
  kops_test::copy_in(in.p, host);
  ActQuantOut out = make_out(q.p, cols, scales.p, 1, DType::kFloat8E4M3,
                             ActQuantMode::kDynamicPerToken);
  const Tensor tin = tensor2d(in.p, DType::kFloat32, rows, cols);
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return;
  LaunchContext gc = ctx;
  gc.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  KOPS_EXPECT_OK(kops::quant::quantize_activation(out, tin, Tensor{}, 0, gc), "graph quant");
  cudaStreamEndCapture(stream, &graph);
  if (graph != nullptr && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    auto got = kops_test::copy_out(q.p, rows * cols);
    const uint8_t want = e4m3_byte(0.5f / (0.5f / 448.0f));
    KOPS_TEST_CHECK(got[0] == want);
    std::vector<float> host2(rows * cols, 2.0f);
    cudaMemcpy(in.p, host2.data(), host2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    got = kops_test::copy_out(q.p, rows * cols);
    KOPS_TEST_CHECK(got[0] >= 100u);
    cudaGraphExecDestroy(exec);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

// =============================================================================
// B4 weight repack (rows 1-4).  Oracles are host re-derivations from the published
// formulations, written in a different shape from the kernels:
//   * Marlin: vLLM marlin_utils_test `get_perms` + `marlin_weights` (numpy permutation of a
//     16x16 sub-tile row, then column packing), `marlin_permute_scales`,
//     `awq_to_marlin_zero_points`;
//   * GGUF: the ggml-quants.c `dequantize_row_*` loop structure;
//   * FP8 UE8M0 / SmoothQuant: the formulas in the header comments.
// =============================================================================

template <typename T>
void b4_put(DevBuf<T>& d, const std::vector<T>& v) {
  kops_test::copy_in(d.p, v);
}

int64_t b4_round_up(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

float b4_e4m3_to_float(uint8_t b) { return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)); }

std::vector<int> b4_marlin_perm(int num_bits) {
  std::vector<int> base;
  for (int i = 0; i < 32; ++i) {
    std::vector<int> perm1;
    const int col = i / 4;
    for (int block = 0; block < 2; ++block) {
      for (const int row : {2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1}) {
        perm1.push_back(16 * row + col + 8 * block);
      }
    }
    for (int j = 0; j < 4; ++j) {
      for (const int p : perm1) base.push_back(p + 256 * j);
    }
  }
  const std::vector<int> il = num_bits == 4 ? std::vector<int>{0, 2, 4, 6, 1, 3, 5, 7}
                                            : std::vector<int>{0, 2, 1, 3};
  std::vector<int> out(base.size());
  for (size_t b = 0; b < base.size() / il.size(); ++b) {
    for (size_t i = 0; i < il.size(); ++i) out[b * il.size() + i] = base[b * il.size() + il[i]];
  }
  return out;
}

/// q[k * n + c] in [0, 2^bits) -> Marlin words, shape [k/16, n*16/pack] row-major.
std::vector<uint32_t> b4_marlin_weights(const std::vector<uint32_t>& q, int k, int n, int bits) {
  const int pack = 32 / bits;
  const auto perm = b4_marlin_perm(bits);
  std::vector<uint32_t> permuted(q.size());
  for (int kt = 0; kt < k / 16; ++kt) {
    // row kt of the (k/16, n*16) matrix: sub-tiles of 16 columns, each 16x16 row-major.
    std::vector<uint32_t> row(static_cast<size_t>(n) * 16);
    for (int nt = 0; nt < n / 16; ++nt) {
      for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
          row[static_cast<size_t>((nt * 16 + r) * 16 + c)] = q[static_cast<size_t>(kt * 16 + r) * n + nt * 16 + c];
        }
      }
    }
    for (size_t chunk = 0; chunk < row.size() / perm.size(); ++chunk) {
      for (size_t i = 0; i < perm.size(); ++i) {
        permuted[static_cast<size_t>(kt) * n * 16 + chunk * perm.size() + i] = row[chunk * perm.size() + perm[i]];
      }
    }
  }
  std::vector<uint32_t> packed(permuted.size() / pack, 0u);
  for (size_t w = 0; w < packed.size(); ++w) {
    for (int i = 0; i < pack; ++i) packed[w] |= permuted[w * pack + i] << (bits * i);
  }
  return packed;
}

std::vector<int> b4_scale_perm(bool per_channel) {
  std::vector<int> perm;
  if (!per_channel) {
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) perm.push_back(i + 8 * j);
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      for (const int j : {0, 1, 8, 9, 16, 17, 24, 25}) perm.push_back(2 * i + j);
    }
  }
  return perm;
}

template <typename T>
std::vector<T> b4_permute_scales(const std::vector<T>& s, bool per_channel) {
  const auto perm = b4_scale_perm(per_channel);
  std::vector<T> out(s.size());
  for (size_t c = 0; c < s.size() / perm.size(); ++c) {
    for (size_t i = 0; i < perm.size(); ++i) out[c * perm.size() + i] = s[c * perm.size() + perm[i]];
  }
  return out;
}

const int kAwqOrder4[8] = {0, 2, 4, 6, 1, 3, 5, 7};
const int kAwqOrder8[4] = {0, 2, 1, 3};

/// AWQ packing: nibble i of word c holds column pack*c + order[i].
std::vector<uint32_t> b4_pack_awq(const std::vector<uint32_t>& q, int rows, int n, int bits) {
  const int pack = 32 / bits;
  std::vector<uint32_t> out(static_cast<size_t>(rows) * (n / pack), 0u);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < n / pack; ++c) {
      for (int i = 0; i < pack; ++i) {
        const int col = pack * c + (bits == 4 ? kAwqOrder4[i] : kAwqOrder8[i]);
        out[static_cast<size_t>(r) * (n / pack) + c] |= q[static_cast<size_t>(r) * n + col] << (bits * i);
      }
    }
  }
  return out;
}

/// GPTQ packing: row k sits in word k/pack at position k%pack.
std::vector<uint32_t> b4_pack_gptq(const std::vector<uint32_t>& q, int k, int n, int bits) {
  const int pack = 32 / bits;
  std::vector<uint32_t> out(static_cast<size_t>(k / pack) * n, 0u);
  for (int kk = 0; kk < k; ++kk) {
    for (int c = 0; c < n; ++c) {
      out[static_cast<size_t>(kk / pack) * n + c] |= q[static_cast<size_t>(kk) * n + c] << (bits * (kk % pack));
    }
  }
  return out;
}

/// vLLM awq_to_marlin_zero_points: unpack, undo the AWQ interleave, permute, interleave, pack.
std::vector<uint32_t> b4_awq_zero_points(const std::vector<uint32_t>& z, int groups, int n, int bits) {
  const int pack = 32 / bits;
  const uint32_t mask = (1u << bits) - 1u;
  const std::vector<int> interleave = bits == 4 ? std::vector<int>{0, 2, 4, 6, 1, 3, 5, 7} : std::vector<int>{0, 2, 1, 3};
  std::vector<int> undo(interleave.size());
  for (size_t i = 0; i < interleave.size(); ++i) undo[interleave[i]] = static_cast<int>(i);  // argsort
  std::vector<uint32_t> out(static_cast<size_t>(groups) * (n / pack), 0u);
  for (int g = 0; g < groups; ++g) {
    std::vector<uint32_t> unpacked(n);
    for (int w = 0; w < n / pack; ++w) {
      for (int i = 0; i < pack; ++i) {
        unpacked[w * pack + i] = (z[static_cast<size_t>(g) * (n / pack) + w] >> (bits * i)) & mask;
      }
    }
    std::vector<uint32_t> undone(n);
    for (int w = 0; w < n / pack; ++w) {
      for (int i = 0; i < pack; ++i) undone[w * pack + i] = unpacked[w * pack + undo[i]];
    }
    const auto perm = b4_scale_perm(false);
    std::vector<uint32_t> permuted(n);
    for (int c = 0; c < n / 64; ++c) {
      for (size_t i = 0; i < perm.size(); ++i) permuted[c * 64 + i] = undone[c * 64 + perm[i]];
    }
    std::vector<uint32_t> inter(n);
    for (int w = 0; w < n / pack; ++w) {
      for (int i = 0; i < pack; ++i) inter[w * pack + i] = permuted[w * pack + interleave[i]];
    }
    for (int w = 0; w < n / pack; ++w) {
      for (int i = 0; i < pack; ++i) out[static_cast<size_t>(g) * (n / pack) + w] |= inter[w * pack + i] << (bits * i);
    }
  }
  return out;
}

struct B4MarlinCase {
  int k;
  int n;
  int group;
};

void b4_fill_ints(std::vector<uint32_t>& q, int bits, unsigned seed) {
  std::mt19937 rng(seed);
  for (auto& x : q) x = rng() & ((1u << bits) - 1u);
}

void test_b4_marlin(LaunchContext& ctx) {
  const B4MarlinCase cases[] = {{64, 128, 32}, {128, 192, 64}, {48, 64, -1}, {256, 128, 128}, {32, 64, 32}};
  for (const int bits : {4, 8}) {
    const int pack = 32 / bits;
    for (const auto& c : cases) {
      const int groups = (c.group < 0 || c.group >= c.k) ? 1 : c.k / c.group;
      const bool per_channel = groups == 1;
      std::vector<uint32_t> q(static_cast<size_t>(c.k) * c.n);
      b4_fill_ints(q, bits, 100 + c.k + c.n);
      std::vector<uint16_t> scales(static_cast<size_t>(groups) * c.n);
      std::mt19937 rng(7);
      for (auto& s : scales) s = static_cast<uint16_t>(rng());
      const auto want_scales = b4_permute_scales(scales, per_channel);
      const auto want_weight = b4_marlin_weights(q, c.k, c.n, bits);
      const int64_t out_cols = static_cast<int64_t>(c.n) * 16 / pack;
      char what[96];

      // ---- AWQ -------------------------------------------------------------
      {
        std::vector<uint32_t> zeros(static_cast<size_t>(groups) * c.n);
        b4_fill_ints(zeros, bits, 9);
        const auto qweight = b4_pack_awq(q, c.k, c.n, bits);
        const auto qzeros = b4_pack_awq(zeros, groups, c.n, bits);
        const auto want_zeros = b4_awq_zero_points(qzeros, groups, c.n, bits);
        DevBuf<uint32_t> dq(qweight.size()), dz(qzeros.size()), ow(want_weight.size()), oz(want_zeros.size());
        DevBuf<uint16_t> ds(scales.size()), os(scales.size());
        b4_put(dq, qweight);
        b4_put(dz, qzeros);
        b4_put(ds, scales);
        cudaMemset(ow.p, 0xCD, want_weight.size() * 4);
        std::snprintf(what, sizeof(what), "awq_to_marlin bits=%d K=%d N=%d g=%d", bits, c.k, c.n, c.group);
        KOPS_EXPECT_OK(kops::quant::awq_to_marlin(
                           tensor2d(ow.p, DType::kInt32, c.k / 16, out_cols),
                           tensor2d(os.p, DType::kFloat16, groups, c.n),
                           tensor2d(oz.p, DType::kInt32, groups, c.n / pack),
                           tensor2d(dq.p, DType::kInt32, c.k, c.n / pack),
                           tensor2d(ds.p, DType::kFloat16, groups, c.n),
                           tensor2d(dz.p, DType::kInt32, groups, c.n / pack), bits, c.group, ctx),
                       what);
        cudaDeviceSynchronize();
        if (kops_test::copy_out(ow.p, want_weight.size()) != want_weight) kops_test::record_failure(what, "weights differ", __FILE__, __LINE__);
        if (kops_test::copy_out(os.p, scales.size()) != want_scales) kops_test::record_failure(what, "scales differ", __FILE__, __LINE__);
        if (kops_test::copy_out(oz.p, want_zeros.size()) != want_zeros) kops_test::record_failure(what, "zeros differ", __FILE__, __LINE__);
      }

      // ---- GPTQ, with and without act-order -------------------------------------
      for (const bool act_order : {false, true}) {
        std::vector<int32_t> g_idx(c.k);
        std::vector<int32_t> perm(c.k);
        for (int i = 0; i < c.k; ++i) perm[i] = i;
        if (act_order) {
          std::mt19937 grng(55);
          for (auto& g : g_idx) g = static_cast<int32_t>(grng() % std::max(groups, 2));  // ties on purpose
          std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) { return g_idx[a] < g_idx[b]; });
        }
        std::vector<uint32_t> q_sorted(q.size());
        for (int kk = 0; kk < c.k; ++kk) {
          std::copy(q.begin() + static_cast<size_t>(perm[kk]) * c.n, q.begin() + static_cast<size_t>(perm[kk] + 1) * c.n,
                    q_sorted.begin() + static_cast<size_t>(kk) * c.n);
        }
        const auto want_gptq = b4_marlin_weights(q_sorted, c.k, c.n, bits);
        const auto qweight = b4_pack_gptq(q, c.k, c.n, bits);  // checkpoint order, not permuted
        DevBuf<uint32_t> dq(qweight.size()), ow(want_gptq.size());
        DevBuf<uint16_t> ds(scales.size()), os(scales.size());
        DevBuf<int32_t> dg(c.k), op(c.k);
        b4_put(dq, qweight);
        b4_put(ds, scales);
        b4_put(dg, g_idx);
        cudaMemset(ow.p, 0xCD, want_gptq.size() * 4);
        cudaMemset(op.p, 0xFF, c.k * 4);
        std::snprintf(what, sizeof(what), "gptq_to_marlin bits=%d K=%d N=%d g=%d act_order=%d", bits, c.k, c.n, c.group, act_order);
        KOPS_EXPECT_OK(kops::quant::gptq_to_marlin(
                           tensor2d(ow.p, DType::kInt32, c.k / 16, out_cols),
                           tensor2d(os.p, DType::kBFloat16, groups, c.n), tensor1d(op.p, DType::kInt32, c.k),
                           tensor2d(dq.p, DType::kInt32, c.k / pack, c.n),
                           tensor2d(ds.p, DType::kBFloat16, groups, c.n),
                           act_order ? tensor1d(dg.p, DType::kInt32, c.k) : Tensor{}, bits, c.group, true, ctx),
                       what);
        cudaDeviceSynchronize();
        if (kops_test::copy_out(ow.p, want_gptq.size()) != want_gptq) kops_test::record_failure(what, "weights differ", __FILE__, __LINE__);
        if (kops_test::copy_out(os.p, scales.size()) != want_scales) kops_test::record_failure(what, "scales differ", __FILE__, __LINE__);
        if (kops_test::copy_out(op.p, c.k) != perm) kops_test::record_failure(what, "perm differs", __FILE__, __LINE__);
      }
    }
  }
}

// ---- FP4 -> Marlin, backend shuffle ------------------------------------------------

std::vector<uint8_t> b4_to_blocked(const std::vector<uint8_t>& in, int64_t rows, int64_t cols) {
  const int64_t rb_n = (rows + 127) / 128;
  const int64_t cb_n = (cols + 3) / 4;
  std::vector<uint8_t> out(static_cast<size_t>(rb_n * cb_n * 512), 0);
  for (int64_t rb = 0; rb < rb_n; ++rb) {
    for (int64_t cb = 0; cb < cb_n; ++cb) {
      for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 4; ++c) {
          const int64_t gr = rb * 128 + r;
          const int64_t gc = cb * 4 + c;
          out[static_cast<size_t>((rb * cb_n + cb) * 512 + (r % 32) * 16 + (r / 32) * 4 + c)] =
              (gr < rows && gc < cols) ? in[static_cast<size_t>(gr * cols + gc)] : 0;
        }
      }
    }
  }
  return out;
}

// ---- TRT-LLM-gen layout (oracle: TensorRT-LLM @ 09198b7) ----------------------------------------
// tensorrt_llm/quantization/utils/fp4_utils.py: srcToDstBlk32RowMap and get_shuffle_matrix_a_row_indices
// (epilogue_tile_m = 128), get_reorder_rows_for_gated_act_gemm_row_indices; cpp/tensorrt_llm/thop/fp8Op.h:
// computeSFIndex (SWIZZLED) which block_scale_interleave writes; tensorrt_llm/_torch/moe/fused_moe/quantization.py:
// trtllmgen_maybe_get_cached_w3_w1_permute_indices chains permute0[permute1].

const int kB4TrtllmRowMap[32] = {0, 8,  16, 24, 1, 9,  17, 25, 2, 10, 18, 26, 3, 11, 19, 27,
                                 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31};

/// row_indices[new_row] = old_row.
std::vector<int> b4_trtllm_shuffle_indices(int m) {
  std::vector<int> idx(static_cast<size_t>(m));
  for (int old_row = 0; old_row < m; ++old_row) idx[static_cast<size_t>((old_row / 32) * 32 + kB4TrtllmRowMap[old_row % 32])] = old_row;
  return idx;
}

/// [r0, rM/2, r1, rM/2+1, ...]
std::vector<int> b4_trtllm_gated_indices(int m) {
  std::vector<int> idx(static_cast<size_t>(m));
  for (int i = 0; i < m / 2; ++i) {
    idx[static_cast<size_t>(2 * i)] = i;
    idx[static_cast<size_t>(2 * i + 1)] = m / 2 + i;
  }
  return idx;
}

int b4_trtllm_sf_index(int row, int col, int total_col) {
  const int padded_col = (total_col + 3) / 4 * 4;
  return col % 4 + (col / 4) * (4 * 128) + (row % 32) * 16 + ((row % 128) / 32) * 4 + (row / 128) * (128 * padded_col);
}

struct B4TrtllmCase {
  int experts, n, k, block;
  bool gated;  // fused gate/up weight: gemm::glu_weight_interleave (halves -> alternating rows) first
};

void b4_trtllm_case(LaunchContext& ctx, const B4TrtllmCase& c, bool mx) {
  const int groups = c.k / c.block;
  const size_t row_bytes = static_cast<size_t>(c.k) / 2;
  std::vector<uint8_t> weight(static_cast<size_t>(c.experts) * c.n * row_bytes);
  std::vector<uint8_t> scales(static_cast<size_t>(c.experts) * c.n * groups);
  std::mt19937 rng(static_cast<unsigned>(c.n * 31 + c.k + c.block + (c.gated ? 7 : 0)));
  for (auto& b : weight) b = static_cast<uint8_t>(rng());
  for (auto& s : scales) s = static_cast<uint8_t>(rng());

  const auto shuffle = b4_trtllm_shuffle_indices(c.n);
  const auto reorder = b4_trtllm_gated_indices(c.n);
  std::vector<int> rows(static_cast<size_t>(c.n));  // output row -> source row of the original weight
  for (int i = 0; i < c.n; ++i) rows[static_cast<size_t>(i)] = c.gated ? reorder[static_cast<size_t>(shuffle[static_cast<size_t>(i)])] : shuffle[static_cast<size_t>(i)];
  std::vector<uint8_t> want_weight(weight.size()), want_scales(scales.size());
  for (int e = 0; e < c.experts; ++e) {
    for (int i = 0; i < c.n; ++i) {
      const size_t src = static_cast<size_t>(e) * c.n + static_cast<size_t>(rows[static_cast<size_t>(i)]);
      std::memcpy(&want_weight[(static_cast<size_t>(e) * c.n + i) * row_bytes], &weight[src * row_bytes], row_bytes);
      for (int g = 0; g < groups; ++g) {
        want_scales[static_cast<size_t>(e) * c.n * groups + static_cast<size_t>(b4_trtllm_sf_index(i, g, groups))] =
            scales[src * groups + g];
      }
    }
  }

  const DType sdt = mx ? DType::kFloat8E8M0 : DType::kFloat8E4M3;
  DevBuf<uint8_t> dw(weight.size()), ds(scales.size()), ow(weight.size()), os(scales.size());
  b4_put(dw, weight);
  b4_put(ds, scales);
  cudaMemset(ow.p, 0xCD, weight.size());
  cudaMemset(os.p, 0xCD, scales.size());
  Tensor t_w = kops_test::tensor3d(dw.p, DType::kUInt8, c.experts, c.n, row_bytes);
  Tensor t_s = kops_test::tensor3d(ds.p, sdt, c.experts, c.n, groups);
  const Tensor t_ow = kops_test::tensor3d(ow.p, DType::kUInt8, c.experts, c.n, row_bytes);
  const Tensor t_os = kops_test::tensor3d(os.p, sdt, c.experts, c.n, groups);
  DevBuf<uint8_t> gw(c.gated ? weight.size() : 1), gs(c.gated ? scales.size() : 1);
  if (c.gated) {
    const Tensor g_w = kops_test::tensor3d(gw.p, DType::kUInt8, c.experts, c.n, row_bytes);
    const Tensor g_s = kops_test::tensor3d(gs.p, sdt, c.experts, c.n, groups);
    KOPS_EXPECT_OK(kops::gemm::glu_weight_interleave(g_w, g_s, Tensor{}, t_w, t_s, Tensor{}, 0, 1, ctx), "glu_weight_interleave");
    t_w = g_w;
    t_s = g_s;
  }
  KOPS_EXPECT_OK(kops::quant::fp4_backend_shuffle(t_ow, t_os, t_w, t_s, c.block, kops::WeightPacking::kTrtllmShuffled, ctx),
                 "fp4_backend_shuffle trtllm");
  cudaDeviceSynchronize();
  KOPS_TEST_CHECK(kops_test::copy_out(ow.p, weight.size()) == want_weight);
  KOPS_TEST_CHECK(kops_test::copy_out(os.p, scales.size()) == want_scales);
}

void b4_trtllm_contracts(LaunchContext& ctx) {
  const auto target = kops::WeightPacking::kTrtllmShuffled;
  struct Buffers {
    DevBuf<uint8_t> w, s, ow, os;
    int n, k, block;
    Buffers(int n_, int k_, int block_)
        : w(static_cast<size_t>(n_) * k_ / 2), s(static_cast<size_t>(n_) * (k_ / block_)),
          ow(static_cast<size_t>(n_) * k_ / 2), os(static_cast<size_t>(n_) * (k_ / block_)), n(n_), k(k_), block(block_) {}
    Tensor weight(DevBuf<uint8_t>& b) const { return kops_test::tensor3d(b.p, DType::kUInt8, 1, n, k / 2); }
    Tensor scale(DevBuf<uint8_t>& b) const { return kops_test::tensor3d(b.p, DType::kFloat8E4M3, 1, n, k / block); }
  };
  Buffers ok(128, 128, 16);
  KOPS_EXPECT_OK(kops::quant::fp4_backend_shuffle(ok.weight(ok.ow), ok.scale(ok.os), ok.weight(ok.w), ok.scale(ok.s), ok.block, target, ctx),
                 "trtllm shuffle");
  KOPS_EXPECT_STATUS(kops::quant::fp4_backend_shuffle(ok.weight(ok.w), ok.scale(ok.os), ok.weight(ok.w), ok.scale(ok.s), ok.block, target, ctx),
                     Status::kInvalidArgument, "in-place weight");
  KOPS_EXPECT_STATUS(kops::quant::fp4_backend_shuffle(ok.weight(ok.ow), ok.scale(ok.s), ok.weight(ok.w), ok.scale(ok.s), ok.block, target, ctx),
                     Status::kInvalidArgument, "in-place scales");
  KOPS_EXPECT_STATUS(kops::quant::fp4_backend_shuffle(ok.weight(ok.ow), ok.scale(ok.os), ok.weight(ok.w), ok.scale(ok.s), ok.block,
                                                      kops::WeightPacking::kMarlin, ctx),
                     Status::kNotImplemented, "not a backend layout");
  // N must be a multiple of 128 and K / block a multiple of 4 (the asserts of get_shuffle_matrix_sf_a_row_indices).
  Buffers n130(130, 128, 16);
  KOPS_EXPECT_STATUS(kops::quant::fp4_backend_shuffle(n130.weight(n130.ow), n130.scale(n130.os), n130.weight(n130.w), n130.scale(n130.s),
                                                      n130.block, target, ctx),
                     Status::kUnsupportedShape, "N not a multiple of 128");
  Buffers n160(160, 128, 16);  // a multiple of the 32-row shuffle block, but not of the 128-row scale tile
  KOPS_EXPECT_STATUS(kops::quant::fp4_backend_shuffle(n160.weight(n160.ow), n160.scale(n160.os), n160.weight(n160.w), n160.scale(n160.s),
                                                      n160.block, target, ctx),
                     Status::kUnsupportedShape, "N not a multiple of the scale tile");
  Buffers g3(128, 96, 32);
  KOPS_EXPECT_STATUS(kops::quant::fp4_backend_shuffle(g3.weight(g3.ow), g3.scale(g3.os), g3.weight(g3.w), g3.scale(g3.s), g3.block, target, ctx),
                     Status::kUnsupportedShape, "K / block not a multiple of 4");
}

void test_b4_fp4(LaunchContext& ctx) {
  for (const bool mx : {true, false}) {
    const int block = mx ? 32 : 16;
    const int n = 128;
    const int k = 256;
    const int groups = k / block;
    std::vector<uint32_t> q(static_cast<size_t>(k) * n);  // q[k][n], 4-bit
    b4_fill_ints(q, 4, 17);
    std::vector<uint8_t> weight(static_cast<size_t>(n) * k / 2, 0);  // [n][k/2], low nibble = even k
    for (int r = 0; r < n; ++r) {
      for (int kk = 0; kk < k; ++kk) {
        weight[static_cast<size_t>(r) * (k / 2) + kk / 2] |= static_cast<uint8_t>(q[static_cast<size_t>(kk) * n + r] << (4 * (kk & 1)));
      }
    }
    std::vector<uint8_t> scales(static_cast<size_t>(n) * groups);
    std::mt19937 rng(23);
    for (auto& s : scales) s = static_cast<uint8_t>(rng());
    std::vector<uint8_t> scales_t(scales.size());  // [groups][n]
    for (int r = 0; r < n; ++r) {
      for (int g = 0; g < groups; ++g) scales_t[static_cast<size_t>(g) * n + r] = scales[static_cast<size_t>(r) * groups + g];
    }
    const auto want_scales = b4_permute_scales(scales_t, false);
    const auto want_weight = b4_marlin_weights(q, k, n, 4);
    DevBuf<uint8_t> dw(weight.size()), ds(scales.size()), os(scales.size());
    DevBuf<uint32_t> ow(want_weight.size());
    b4_put(dw, weight);
    b4_put(ds, scales);
    cudaMemset(ow.p, 0xCD, want_weight.size() * 4);
    KOPS_EXPECT_OK(kops::quant::fp4_to_marlin(tensor2d(ow.p, DType::kInt32, k / 16, n * 2),
                                              tensor2d(os.p, mx ? DType::kFloat8E8M0 : DType::kFloat8E4M3, groups, n),
                                              tensor2d(dw.p, DType::kUInt8, n, k / 2),
                                              tensor2d(ds.p, mx ? DType::kFloat8E8M0 : DType::kFloat8E4M3, n, groups),
                                              block, mx, ctx),
                   "fp4_to_marlin");
    cudaDeviceSynchronize();
    KOPS_TEST_CHECK(kops_test::copy_out(ow.p, want_weight.size()) == want_weight);
    KOPS_TEST_CHECK(kops_test::copy_out(os.p, scales.size()) == want_scales);
  }

  // fp4_backend_shuffle (CUTLASS / SM120 interleaved scales), two experts, N not a multiple of 128.
  for (const int block : {16, 32}) {
    const int e = 2, n = 130, k = 128, groups = k / block;
    const auto weight = [&] {
      std::vector<uint8_t> w(static_cast<size_t>(e) * n * k / 2);
      std::mt19937 rng(3);
      for (auto& b : w) b = static_cast<uint8_t>(rng());
      return w;
    }();
    std::vector<uint8_t> scales(static_cast<size_t>(e) * n * groups);
    std::mt19937 rng(4);
    for (auto& s : scales) s = static_cast<uint8_t>(rng());
    const int64_t n_pad = b4_round_up(n, 128);
    const int64_t g_pad = b4_round_up(groups, 4);
    std::vector<uint8_t> want_scales;
    for (int x = 0; x < e; ++x) {
      const std::vector<uint8_t> slice(scales.begin() + static_cast<size_t>(x) * n * groups,
                                       scales.begin() + static_cast<size_t>(x + 1) * n * groups);
      const auto blocked = b4_to_blocked(slice, n, groups);
      want_scales.insert(want_scales.end(), blocked.begin(), blocked.end());
    }
    DevBuf<uint8_t> dw(weight.size()), ds(scales.size()), ow(weight.size()), os(want_scales.size());
    b4_put(dw, weight);
    b4_put(ds, scales);
    Tensor t_w = kops_test::tensor3d(dw.p, DType::kUInt8, e, n, k / 2);
    Tensor t_ow = kops_test::tensor3d(ow.p, DType::kUInt8, e, n, k / 2);
    Tensor t_s = kops_test::tensor3d(ds.p, DType::kFloat8E4M3, e, n, groups);
    Tensor t_os = kops_test::tensor3d(os.p, DType::kFloat8E4M3, e, n_pad, g_pad);
    KOPS_EXPECT_OK(kops::quant::fp4_backend_shuffle(t_ow, t_os, t_w, t_s, block, kops::WeightPacking::kFP4E2M1, ctx), "fp4_backend_shuffle");
    cudaDeviceSynchronize();
    KOPS_TEST_CHECK(kops_test::copy_out(ow.p, weight.size()) == weight);
    KOPS_TEST_CHECK(kops_test::copy_out(os.p, want_scales.size()) == want_scales);
  }

  // fp4_backend_shuffle, TRT-LLM-gen layout: plain (down projection) and gated (fused gate/up) weights.
  for (const bool mx : {false, true}) {
    for (const bool gated : {false, true}) {
      b4_trtllm_case(ctx, {2, 256, 128, mx ? 32 : 16, gated}, mx);
    }
  }
  b4_trtllm_case(ctx, {2, 384, 256, 16, true}, false);  // three row tiles, four column tiles
  b4_trtllm_case(ctx, {1, 128, 512, 32, false}, true);
  b4_trtllm_contracts(ctx);
}

// ---- AWQ dequantize ---------------------------------------------------------------

template <typename T>
void b4_awq_dequant_case(LaunchContext& ctx, DType dtype, int k, int n, int group) {
  const int groups = group > 0 ? k / group : 1;
  std::vector<uint32_t> q(static_cast<size_t>(k) * n), z(static_cast<size_t>(groups) * n);
  b4_fill_ints(q, 4, 61);
  b4_fill_ints(z, 4, 62);
  std::vector<float> sf(static_cast<size_t>(groups) * n);
  fill_uniform(sf, 63, 0.001f, 0.2f);
  const auto s = cast_from_float<T>(sf);
  const auto qweight = b4_pack_awq(q, k, n, 4);
  const auto qzeros = b4_pack_awq(z, groups, n, 4);
  DevBuf<uint32_t> dq(qweight.size()), dz(qzeros.size());
  DevBuf<T> ds(s.size()), out(static_cast<size_t>(k) * n);
  b4_put(dq, qweight);
  b4_put(dz, qzeros);
  b4_put(ds, s);
  KOPS_EXPECT_OK(kops::quant::awq_dequantize(tensor2d(out.p, dtype, k, n), tensor2d(dq.p, DType::kInt32, k, n / 8),
                                             tensor2d(ds.p, dtype, groups, n), tensor2d(dz.p, DType::kInt32, groups, n / 8), group, ctx),
                 "awq_dequantize");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(out.p, static_cast<size_t>(k) * n);
  size_t bad = 0;
  for (int r = 0; r < k; ++r) {
    for (int c = 0; c < n; ++c) {
      const int g = group > 0 ? r / group : 0;
      const float want_f = static_cast<float>(static_cast<int>(q[static_cast<size_t>(r) * n + c]) -
                                              static_cast<int>(z[static_cast<size_t>(g) * n + c])) *
                           kops_test::to_float(s[static_cast<size_t>(g) * n + c]);
      T want;
      kops_test::from_float(want, want_f);
      bad += kops_test::to_float(got[static_cast<size_t>(r) * n + c]) != kops_test::to_float(want);
    }
  }
  KOPS_TEST_CHECK(bad == 0);
}

void test_b4_awq_dequant(LaunchContext& ctx) {
  b4_awq_dequant_case<__half>(ctx, DType::kFloat16, 64, 128, 32);
  b4_awq_dequant_case<__nv_bfloat16>(ctx, DType::kBFloat16, 128, 64, 128);
  b4_awq_dequant_case<float>(ctx, DType::kFloat32, 32, 256, -1);
  b4_awq_dequant_case<__half>(ctx, DType::kFloat16, 256, 1024, 64);
}

// ---- GGUF -------------------------------------------------------------------------

float b4_half(const uint8_t* p) {
  const unsigned short raw = static_cast<unsigned short>(p[0] | (p[1] << 8));
  return __half2float(__ushort_as_half(raw));
}

void b4_put_half(uint8_t* p, float v) {
  const unsigned short raw = __half_as_ushort(__float2half_rn(v));
  p[0] = static_cast<uint8_t>(raw);
  p[1] = static_cast<uint8_t>(raw >> 8);
}

void b4_put_half_bits(uint8_t* p, uint16_t bits) {
  p[0] = static_cast<uint8_t>(bits);
  p[1] = static_cast<uint8_t>(bits >> 8);
}

void b4_scale_min_k4(int j, const uint8_t* q, int* d, int* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

const int8_t kB4Iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

/// ggml-quants.c dequantize_row_* for one block, same operation order (fp32, no FMA).
void b4_ggml_block(kops::GGUFType type, const uint8_t* b, float* y) {
  using kops::GGUFType;
  switch (type) {
    case GGUFType::kQ4_0: {
      const float d = b4_half(b);
      for (int j = 0; j < 16; ++j) {
        y[j] = ((b[2 + j] & 0xF) - 8) * d;
        y[j + 16] = ((b[2 + j] >> 4) - 8) * d;
      }
      break;
    }
    case GGUFType::kQ4_1: {
      const float d = b4_half(b), m = b4_half(b + 2);
      for (int j = 0; j < 16; ++j) {
        y[j] = (b[4 + j] & 0xF) * d + m;
        y[j + 16] = (b[4 + j] >> 4) * d + m;
      }
      break;
    }
    case GGUFType::kQ5_0: {
      const float d = b4_half(b);
      uint32_t qh;
      std::memcpy(&qh, b + 2, 4);
      for (int j = 0; j < 16; ++j) {
        const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
        y[j] = (((b[6 + j] & 0x0F) | xh0) - 16) * d;
        y[j + 16] = (((b[6 + j] >> 4) | xh1) - 16) * d;
      }
      break;
    }
    case GGUFType::kQ5_1: {
      const float d = b4_half(b), m = b4_half(b + 2);
      uint32_t qh;
      std::memcpy(&qh, b + 4, 4);
      for (int j = 0; j < 16; ++j) {
        const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
        y[j] = ((b[8 + j] & 0xF) | xh0) * d + m;
        y[j + 16] = ((b[8 + j] >> 4) | xh1) * d + m;
      }
      break;
    }
    case GGUFType::kQ8_0: {
      const float d = b4_half(b);
      for (int j = 0; j < 32; ++j) y[j] = static_cast<int8_t>(b[2 + j]) * d;
      break;
    }
    case GGUFType::kQ2_K: {
      const float d = b4_half(b + 80), min = b4_half(b + 82);
      const uint8_t* q = b + 16;
      int is = 0;
      for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
          uint8_t sc = b[is++];
          float dl = d * (sc & 0xF), ml = min * (sc >> 4);
          for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
          sc = b[is++];
          dl = d * (sc & 0xF);
          ml = min * (sc >> 4);
          for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
          shift += 2;
        }
        q += 32;
      }
      break;
    }
    case GGUFType::kQ3_K: {
      const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
      uint32_t aux[4];
      const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
      const float d_all = b4_half(b + 108);
      const uint8_t* q = b + 32;
      const uint8_t* hm = b;
      uint8_t m = 1;
      std::memcpy(aux, b + 96, 12);
      const uint32_t tmp = aux[2];
      aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
      aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
      aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
      int is = 0;
      float dl;
      for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
          dl = d_all * (scales[is++] - 32);
          for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
          dl = d_all * (scales[is++] - 32);
          for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
          shift += 2;
          m <<= 1;
        }
        q += 32;
      }
      break;
    }
    case GGUFType::kQ4_K:
    case GGUFType::kQ5_K: {
      const bool five = type == GGUFType::kQ5_K;
      const float d = b4_half(b), min = b4_half(b + 2);
      const uint8_t* scales = b + 4;
      const uint8_t* qh = b + 16;
      const uint8_t* q = b + (five ? 48 : 16);
      int is = 0;
      uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < 256; j += 64) {
        int sc, m;
        b4_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        b4_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = min * m;
        if (five) {
          for (int l = 0; l < 32; ++l) *y++ = d1 * ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
          for (int l = 0; l < 32; ++l) *y++ = d2 * ((q[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
          u1 = static_cast<uint8_t>(u1 << 2);
          u2 = static_cast<uint8_t>(u2 << 2);
        } else {
          for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
          for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
      }
      break;
    }
    case GGUFType::kQ6_K: {
      const float d = b4_half(b + 208);
      const uint8_t* ql = b;
      const uint8_t* qh = b + 128;
      const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
      for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
          const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
          const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
          const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
          y[l + 0] = d * sc[is + 0] * q1;
          y[l + 32] = d * sc[is + 2] * q2;
          y[l + 64] = d * sc[is + 4] * q3;
          y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
      }
      break;
    }
    case GGUFType::kQ8_K: {
      float d;
      std::memcpy(&d, b, 4);
      for (int j = 0; j < 256; ++j) y[j] = d * static_cast<int8_t>(b[4 + j]);
      break;
    }
    case GGUFType::kIQ4_NL: {
      const float d = b4_half(b);
      for (int j = 0; j < 16; ++j) {
        y[j + 0] = d * kB4Iq4nl[b[2 + j] & 0xf];
        y[j + 16] = d * kB4Iq4nl[b[2 + j] >> 4];
      }
      break;
    }
    default: {  // kIQ4_XS
      const float d = b4_half(b);
      const uint16_t scales_h = static_cast<uint16_t>(b[2] | (b[3] << 8));
      const uint8_t* scales_l = b + 4;
      const uint8_t* qs = b + 8;
      for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) {
          y[j + 0] = dl * kB4Iq4nl[qs[j] & 0xf];
          y[j + 16] = dl * kB4Iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
      }
      break;
    }
  }
}

template <typename T>
void b4_gguf_case(LaunchContext& ctx, DType dtype, kops::GGUFType type, int elems, int bytes,
                  const std::vector<int>& half_offsets, int n, int k, const char* what) {
  const int blocks = n * k / elems;
  auto raw = std::vector<uint8_t>(static_cast<size_t>(blocks) * bytes);
  std::mt19937 rng(static_cast<unsigned>(type) * 131 + 7);
  for (auto& x : raw) x = static_cast<uint8_t>(rng());
  std::uniform_real_distribution<float> dist(0.01f, 2.0f);
  for (int b = 0; b < blocks; ++b) {
    uint8_t* blk = &raw[static_cast<size_t>(b) * bytes];
    if (type == kops::GGUFType::kQ8_K) {
      const float d = dist(rng);
      std::memcpy(blk, &d, 4);
    }
    for (const int off : half_offsets) b4_put_half(blk + off, (rng() & 1 ? 1.0f : -1.0f) * dist(rng));
  }
  std::vector<float> want(static_cast<size_t>(n) * k);
  for (int b = 0; b < blocks; ++b) b4_ggml_block(type, &raw[static_cast<size_t>(b) * bytes], &want[static_cast<size_t>(b) * elems]);

  DevBuf<uint8_t> dq(raw.size());
  b4_put(dq, raw);
  DevBuf<T> out(static_cast<size_t>(n) * k);
  KOPS_EXPECT_OK(kops::quant::gguf_dequantize(tensor2d(out.p, dtype, n, k), tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(raw.size())), type, ctx), what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(out.p, want.size());
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    T w;
    kops_test::from_float(w, want[i]);
    bad += kops_test::to_float(got[i]) != kops_test::to_float(w);
  }
  if (bad != 0) {
    char detail[96];
    std::snprintf(detail, sizeof(detail), "%zu/%zu elements differ", bad, want.size());
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
}

// Golden check against llama.cpp itself.  The blocks come from the splitmix64 stream below; the expected hash is
// the FNV-1a-64 of the fp32 output of llama.cpp @ 0ec4027 dequantize_row_* (ggml/src/ggml-quants.c, gcc -O2
// -ffp-contract=off) on the very same blocks.  1024 blocks per type reference every codebook entry of the IQ
// formats, so a wrong table entry changes the hash.
struct B4Splitmix {
  uint64_t state;
  uint64_t next() {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  uint16_t half() {  // finite fp16 in +-[0.0625, 1.0)
    const uint64_t r = next();
    return static_cast<uint16_t>(((r & 1) << 15) | (0x2C00 + static_cast<unsigned>((r >> 1) % 0x1000)));
  }
  uint32_t float_bits() { return 0x3C000000u + static_cast<uint32_t>(next() % 0x03000000u); }
};

uint64_t b4_fnv1a64(const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

void b4_gguf_golden(LaunchContext& ctx) {
  using kops::GGUFType;
  struct Golden {
    GGUFType type;
    const char* name;
    int elems;
    int bytes;
    int halves[3];  // offsets of fp16 scale fields, -1 terminated
    int special;    // 1: float32 scale at offset 0 (Q8_K), 2: fp16 scale spread over the IQ1_M scale nibbles
    uint64_t hash;
  };
  const Golden formats[] = {
      {GGUFType::kQ4_0, "Q4_0", 32, 18, {0, -1, -1}, 0, 0x4adb48757b194871ULL},
      {GGUFType::kQ4_1, "Q4_1", 32, 20, {0, 2, -1}, 0, 0x9816652ec390d657ULL},
      {GGUFType::kQ5_0, "Q5_0", 32, 22, {0, -1, -1}, 0, 0x2d8263cc53e4d9e4ULL},
      {GGUFType::kQ5_1, "Q5_1", 32, 24, {0, 2, -1}, 0, 0xd48c06148a145245ULL},
      {GGUFType::kQ8_0, "Q8_0", 32, 34, {0, -1, -1}, 0, 0xae2a5c43b8a5c657ULL},
      {GGUFType::kQ2_K, "Q2_K", 256, 84, {80, 82, -1}, 0, 0x457ea82288500929ULL},
      {GGUFType::kQ3_K, "Q3_K", 256, 110, {108, -1, -1}, 0, 0xb7ea81fd6ffd5309ULL},
      {GGUFType::kQ4_K, "Q4_K", 256, 144, {0, 2, -1}, 0, 0xc23331b52fd0e076ULL},
      {GGUFType::kQ5_K, "Q5_K", 256, 176, {0, 2, -1}, 0, 0xb3c96e3dfb4633d3ULL},
      {GGUFType::kQ6_K, "Q6_K", 256, 210, {208, -1, -1}, 0, 0x57284c5cd19896c1ULL},
      {GGUFType::kQ8_K, "Q8_K", 256, 292, {-1, -1, -1}, 1, 0x0cd759ecc83d6becULL},
      {GGUFType::kIQ2_XXS, "IQ2_XXS", 256, 66, {0, -1, -1}, 0, 0xeaa64901846d6e69ULL},
      {GGUFType::kIQ2_XS, "IQ2_XS", 256, 74, {0, -1, -1}, 0, 0xda4db9cb4646de8eULL},
      {GGUFType::kIQ3_XXS, "IQ3_XXS", 256, 98, {0, -1, -1}, 0, 0xbc186cca17356beeULL},
      {GGUFType::kIQ1_S, "IQ1_S", 256, 50, {0, -1, -1}, 0, 0x878f7625e826fdf9ULL},
      {GGUFType::kIQ4_NL, "IQ4_NL", 32, 18, {0, -1, -1}, 0, 0x1bf0edaa21fdee0fULL},
      {GGUFType::kIQ3_S, "IQ3_S", 256, 110, {0, -1, -1}, 0, 0x1481eeb9d505057cULL},
      {GGUFType::kIQ2_S, "IQ2_S", 256, 82, {0, -1, -1}, 0, 0xb281defacaac0520ULL},
      {GGUFType::kIQ4_XS, "IQ4_XS", 256, 136, {0, -1, -1}, 0, 0xf4f955cf05fa63fbULL},
      {GGUFType::kIQ1_M, "IQ1_M", 256, 56, {-1, -1, -1}, 2, 0xe0b63ec6c7f4ee70ULL},
  };
  const int blocks = 1024;
  for (const Golden& f : formats) {
    std::vector<uint8_t> raw(static_cast<size_t>(blocks) * f.bytes);
    B4Splitmix rng{1000u + static_cast<uint64_t>(f.type)};
    for (auto& x : raw) x = static_cast<uint8_t>(rng.next() & 0xFF);
    for (int b = 0; b < blocks; ++b) {
      uint8_t* blk = &raw[static_cast<size_t>(b) * f.bytes];
      for (int h = 0; h < 3 && f.halves[h] >= 0; ++h) b4_put_half_bits(blk + f.halves[h], rng.half());
      if (f.special == 1) {
        const uint32_t v = rng.float_bits();
        std::memcpy(blk, &v, 4);
      } else if (f.special == 2) {
        const uint16_t v = rng.half();
        for (int i = 0; i < 4; ++i) {
          const uint16_t sc = static_cast<uint16_t>(blk[48 + 2 * i] | (blk[49 + 2 * i] << 8));
          b4_put_half_bits(blk + 48 + 2 * i, static_cast<uint16_t>((sc & 0x0FFF) | (((v >> (4 * i)) & 0xF) << 12)));
        }
      }
    }
    const size_t total = static_cast<size_t>(blocks) * f.elems;
    DevBuf<uint8_t> dq(raw.size());
    b4_put(dq, raw);
    DevBuf<float> out(total);
    DevBuf<__half> out_h(total);
    DevBuf<__nv_bfloat16> out_b(total);
    const Tensor qw = tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(raw.size()));
    KOPS_EXPECT_OK(kops::quant::gguf_dequantize(tensor2d(out.p, DType::kFloat32, blocks, f.elems), qw, f.type, ctx), f.name);
    KOPS_EXPECT_OK(kops::quant::gguf_dequantize(tensor2d(out_h.p, DType::kFloat16, blocks, f.elems), qw, f.type, ctx), f.name);
    KOPS_EXPECT_OK(kops::quant::gguf_dequantize(tensor2d(out_b.p, DType::kBFloat16, blocks, f.elems), qw, f.type, ctx), f.name);
    cudaDeviceSynchronize();
    const auto got = kops_test::copy_out(out.p, total);
    if (b4_fnv1a64(got.data(), total * sizeof(float)) != f.hash) {
      kops_test::record_failure(f.name, "fp32 output differs from llama.cpp dequantize_row_*", __FILE__, __LINE__);
    }
    // fp16 / bf16 outputs are the fp32 values rounded once.
    const auto got_h = kops_test::copy_out(out_h.p, total);
    const auto got_b = kops_test::copy_out(out_b.p, total);
    size_t bad = 0;
    for (size_t i = 0; i < total; ++i) {
      __half h;
      __nv_bfloat16 b;
      kops_test::from_float(h, got[i]);
      kops_test::from_float(b, got[i]);
      bad += std::memcmp(&h, &got_h[i], sizeof(h)) != 0;
      bad += std::memcmp(&b, &got_b[i], sizeof(b)) != 0;
    }
    if (bad != 0) kops_test::record_failure(f.name, "fp16/bf16 output is not the rounded fp32 output", __FILE__, __LINE__);
  }
}

void test_b4_gguf(LaunchContext& ctx) {
  using kops::GGUFType;
  struct Fmt { GGUFType t; int elems; int bytes; std::vector<int> halves; const char* name; };
  const Fmt formats[] = {
      {GGUFType::kQ4_0, 32, 18, {0}, "Q4_0"},     {GGUFType::kQ4_1, 32, 20, {0, 2}, "Q4_1"},
      {GGUFType::kQ5_0, 32, 22, {0}, "Q5_0"},     {GGUFType::kQ5_1, 32, 24, {0, 2}, "Q5_1"},
      {GGUFType::kQ8_0, 32, 34, {0}, "Q8_0"},     {GGUFType::kIQ4_NL, 32, 18, {0}, "IQ4_NL"},
      {GGUFType::kQ2_K, 256, 84, {80, 82}, "Q2_K"}, {GGUFType::kQ3_K, 256, 110, {108}, "Q3_K"},
      {GGUFType::kQ4_K, 256, 144, {0, 2}, "Q4_K"}, {GGUFType::kQ5_K, 256, 176, {0, 2}, "Q5_K"},
      {GGUFType::kQ6_K, 256, 210, {208}, "Q6_K"}, {GGUFType::kQ8_K, 256, 292, {}, "Q8_K"},
      {GGUFType::kIQ4_XS, 256, 136, {0}, "IQ4_XS"}};
  for (const auto& f : formats) {
    b4_gguf_case<float>(ctx, DType::kFloat32, f.t, f.elems, f.bytes, f.halves, 5, 512, f.name);
    b4_gguf_case<__half>(ctx, DType::kFloat16, f.t, f.elems, f.bytes, f.halves, 3, 1024, f.name);
    b4_gguf_case<__nv_bfloat16>(ctx, DType::kBFloat16, f.t, f.elems, f.bytes, f.halves, 2, 256, f.name);
  }
  b4_gguf_golden(ctx);

  // Types that are not quantized block formats are refused, not silently wrong.
  DevBuf<uint8_t> raw(4096);
  DevBuf<float> out(4096);
  for (const GGUFType t : {GGUFType::kF32, GGUFType::kF16, GGUFType::kBF16, GGUFType::kQ8_1}) {
    KOPS_EXPECT_STATUS(kops::quant::gguf_dequantize(tensor2d(out.p, DType::kFloat32, 1, 256), tensor1d(raw.p, DType::kUInt8, 66), t, ctx),
                       Status::kNotImplemented, "gguf non-block type");
  }
}

// ---- FP8 block scales -> UE8M0 ----------------------------------------------------

void test_b4_ue8m0(LaunchContext& ctx) {
  for (const auto& nk : {std::pair<int, int>{256, 384}, {200, 300}, {128, 128}, {130, 256}}) {
    const int n = nk.first, k = nk.second;
    const int sn = (n + 127) / 128, sk = (k + 127) / 128;
    std::vector<uint8_t> w(static_cast<size_t>(n) * k);
    std::mt19937 rng(91);
    for (auto& b : w) {
      do { b = static_cast<uint8_t>(rng()); } while ((b & 0x7F) == 0x7F);  // no NaN codes
    }
    std::vector<float> s(static_cast<size_t>(sn) * sk);
    std::uniform_real_distribution<float> dist(1.0e-4f, 3.0f);
    for (size_t i = 0; i < s.size(); ++i) s[i] = (i % 3 == 0) ? std::ldexp(1.0f, static_cast<int>(i % 5) - 3) : dist(rng);
    std::vector<uint8_t> want_w = w;
    std::vector<float> want_s = s;
    std::vector<uint8_t> want_e(s.size());
    double worst_rel = 0.0;
    for (int bi = 0; bi < sn; ++bi) {
      for (int bj = 0; bj < sk; ++bj) {
        const float s_old = s[static_cast<size_t>(bi) * sk + bj];
        uint32_t bits;
        std::memcpy(&bits, &s_old, 4);
        bits &= 0x7FFFFFFFu;
        uint32_t e = (bits >> 23) + ((bits & 0x7FFFFFu) != 0 ? 1u : 0u);
        e = std::min(254u, std::max(1u, e));
        const uint32_t nb = e << 23;
        float s_new;
        std::memcpy(&s_new, &nb, 4);
        want_s[static_cast<size_t>(bi) * sk + bj] = s_new;
        want_e[static_cast<size_t>(bi) * sk + bj] = static_cast<uint8_t>(e);
        const float ratio = s_old / s_new;
        if (ratio == 1.0f) continue;
        for (int r = bi * 128; r < std::min(n, (bi + 1) * 128); ++r) {
          for (int c = bj * 128; c < std::min(k, (bj + 1) * 128); ++c) {
            const size_t idx = static_cast<size_t>(r) * k + c;
            const float scaled = b4_e4m3_to_float(w[idx]) * ratio;
            want_w[idx] = e4m3_byte(scaled);
            if (std::fabs(scaled) >= 0.015625f && std::fabs(scaled) <= 448.0f) {
              worst_rel = std::max(worst_rel, std::fabs(static_cast<double>(b4_e4m3_to_float(want_w[idx])) - scaled) / std::fabs(scaled));
            }
          }
        }
      }
    }
    DevBuf<uint8_t> dw(w.size()), de(s.size());
    DevBuf<float> ds(s.size());
    b4_put(dw, w);
    b4_put(ds, s);
    KOPS_EXPECT_OK(kops::quant::fp8_block_scale_to_ue8m0(tensor2d(dw.p, DType::kFloat8E4M3, n, k), tensor2d(ds.p, DType::kFloat32, sn, sk),
                                                         tensor2d(de.p, DType::kFloat8E8M0, sn, sk), ctx),
                   "fp8_block_scale_to_ue8m0");
    cudaDeviceSynchronize();
    KOPS_TEST_CHECK(kops_test::copy_out(dw.p, w.size()) == want_w);
    KOPS_TEST_CHECK(kops_test::copy_out(ds.p, s.size()) == want_s);
    KOPS_TEST_CHECK(kops_test::copy_out(de.p, s.size()) == want_e);
    KOPS_TEST_CHECK(worst_rel <= 0.0626);  // within one e4m3 ulp (3 mantissa bits)
    for (const float v : want_s) {
      int ex = 0;
      KOPS_TEST_CHECK(std::frexp(v, &ex) == 0.5f);
    }
  }
}

// ---- SmoothQuant ------------------------------------------------------------------

template <typename T>
void b4_smooth_fold_case(LaunchContext& ctx, DType dtype, int n, int k, bool bias, int row_pad) {
  const int ld = k + row_pad;
  std::vector<float> wf(static_cast<size_t>(n) * ld), nwf(k), nbf(k), sm(k);
  fill_uniform(wf, 71, -1.0f, 1.0f);
  fill_uniform(nwf, 72, 0.5f, 1.5f);
  fill_uniform(nbf, 73, -0.5f, 0.5f);
  fill_uniform(sm, 74, 0.05f, 8.0f);
  const auto w = cast_from_float<T>(wf);
  const auto nw = cast_from_float<T>(nwf);
  const auto nb = cast_from_float<T>(nbf);
  DevBuf<T> dw(w.size()), dnw(nw.size()), dnb(nb.size());
  DevBuf<float> ds(sm.size());
  b4_put(dw, w);
  b4_put(dnw, nw);
  b4_put(dnb, nb);
  b4_put(ds, sm);
  KOPS_EXPECT_OK(kops::quant::smoothquant_fold(tensor1d(dnw.p, dtype, k), bias ? tensor1d(dnb.p, dtype, k) : Tensor{},
                                               tensor2d(dw.p, dtype, n, k, ld), tensor1d(ds.p, DType::kFloat32, k), ctx),
                 "smoothquant_fold");
  cudaDeviceSynchronize();
  const auto got_w = kops_test::copy_out(dw.p, w.size());
  const auto got_nw = kops_test::copy_out(dnw.p, nw.size());
  const auto got_nb = kops_test::copy_out(dnb.p, nb.size());
  size_t bad = 0;
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < ld; ++c) {
      const size_t i = static_cast<size_t>(r) * ld + c;
      T want = w[i];
      if (c < k) kops_test::from_float(want, kops_test::to_float(w[i]) * sm[c]);
      bad += kops_test::to_float(got_w[i]) != kops_test::to_float(want);  // padding untouched
    }
  }
  for (int c = 0; c < k; ++c) {
    T want_w, want_b = nb[c];
    kops_test::from_float(want_w, kops_test::to_float(nw[c]) / sm[c]);
    if (bias) kops_test::from_float(want_b, kops_test::to_float(nb[c]) / sm[c]);
    bad += kops_test::to_float(got_nw[c]) != kops_test::to_float(want_w);
    bad += kops_test::to_float(got_nb[c]) != kops_test::to_float(want_b);
  }
  KOPS_TEST_CHECK(bad == 0);
}

void test_b4_smoothquant(LaunchContext& ctx) {
  for (const float alpha : {0.5f, 0.75f, 0.0f, 1.0f}) {
    const int k = 1000;
    std::vector<float> act(k), wmax(k);
    fill_uniform(act, 81, 0.0f, 50.0f);
    fill_uniform(wmax, 82, 0.0f, 20.0f);
    act[0] = 0.0f;
    wmax[1] = 1.0e-9f;  // below the 1e-5 floor
    DevBuf<float> da(k), dw(k), out(k);
    b4_put(da, act);
    b4_put(dw, wmax);
    KOPS_EXPECT_OK(kops::quant::smoothquant_compute_scales(tensor1d(out.p, DType::kFloat32, k), tensor1d(da.p, DType::kFloat32, k),
                                                           tensor1d(dw.p, DType::kFloat32, k), alpha, ctx),
                   "smoothquant_compute_scales");
    cudaDeviceSynchronize();
    const auto got = kops_test::copy_out(out.p, k);
    double worst = 0.0;
    for (int i = 0; i < k; ++i) {
      const double w = std::max<double>(wmax[i], 1.0e-5);
      const double want = std::max(std::pow(static_cast<double>(act[i]), alpha) / std::pow(w, 1.0 - alpha), 1.0e-5);
      worst = std::max(worst, std::fabs(got[i] - want) / want);
    }
    KOPS_TEST_CHECK(worst < 5.0e-6);
  }
  b4_smooth_fold_case<__half>(ctx, DType::kFloat16, 70, 64, true, 0);
  b4_smooth_fold_case<__nv_bfloat16>(ctx, DType::kBFloat16, 33, 128, false, 8);
  b4_smooth_fold_case<float>(ctx, DType::kFloat32, 40, 96, true, 0);
  b4_smooth_fold_case<__half>(ctx, DType::kFloat16, 17, 33, true, 3);  // scalar path
}

// ---- Contracts --------------------------------------------------------------------

void test_b4_contracts(LaunchContext& ctx) {
  DevBuf<uint32_t> a(1 << 14), b(1 << 14), c(1 << 14), d(1 << 14);
  DevBuf<uint16_t> s(1 << 14), os(1 << 14);
  // K not a multiple of 16; N not a multiple of 64; bad group; fp32 scales; wrong output shape.
  KOPS_EXPECT_STATUS(kops::quant::awq_to_marlin(tensor2d(b.p, DType::kInt32, 4, 128), tensor2d(os.p, DType::kFloat16, 1, 64),
                                                tensor2d(c.p, DType::kInt32, 1, 8), tensor2d(a.p, DType::kInt32, 56, 8),
                                                tensor2d(s.p, DType::kFloat16, 1, 64), tensor2d(d.p, DType::kInt32, 1, 8), 4, -1, ctx),
                     Status::kUnsupportedShape, "awq K % 16");
  KOPS_EXPECT_STATUS(kops::quant::awq_to_marlin(tensor2d(b.p, DType::kInt32, 4, 128), tensor2d(os.p, DType::kFloat16, 1, 64),
                                                tensor2d(c.p, DType::kInt32, 1, 8), tensor2d(a.p, DType::kInt32, 64, 8),
                                                tensor2d(s.p, DType::kFloat16, 1, 64), tensor2d(d.p, DType::kInt32, 1, 8), 4, 17, ctx),
                     Status::kUnsupportedShape, "awq group 17");
  KOPS_EXPECT_STATUS(kops::quant::awq_to_marlin(tensor2d(b.p, DType::kInt32, 4, 128), tensor2d(os.p, DType::kFloat32, 1, 64),
                                                tensor2d(c.p, DType::kInt32, 1, 8), tensor2d(a.p, DType::kInt32, 64, 8),
                                                tensor2d(s.p, DType::kFloat32, 1, 64), tensor2d(d.p, DType::kInt32, 1, 8), 4, -1, ctx),
                     Status::kUnsupportedDtype, "awq fp32 scales");
  KOPS_EXPECT_STATUS(kops::quant::awq_to_marlin(tensor2d(b.p, DType::kInt32, 4, 64), tensor2d(os.p, DType::kFloat16, 1, 64),
                                                tensor2d(c.p, DType::kInt32, 1, 8), tensor2d(a.p, DType::kInt32, 64, 8),
                                                tensor2d(s.p, DType::kFloat16, 1, 64), tensor2d(d.p, DType::kInt32, 1, 8), 4, -1, ctx),
                     Status::kInvalidArgument, "awq out shape");
  KOPS_EXPECT_STATUS(kops::quant::gptq_to_marlin(tensor2d(b.p, DType::kInt32, 4, 128), tensor2d(os.p, DType::kFloat16, 1, 64),
                                                 tensor1d(c.p, DType::kInt32, 64), tensor2d(a.p, DType::kInt32, 8, 64),
                                                 tensor2d(s.p, DType::kFloat16, 1, 64), Tensor{}, 4, -1, false, ctx),
                     Status::kUnsupportedLayout, "gptq asymmetric");
  KOPS_EXPECT_STATUS(kops::quant::gptq_to_marlin(tensor2d(b.p, DType::kInt32, 4, 128), tensor2d(os.p, DType::kFloat16, 1, 64),
                                                 Tensor{}, tensor2d(a.p, DType::kInt32, 8, 64),
                                                 tensor2d(s.p, DType::kFloat16, 1, 64), tensor1d(d.p, DType::kInt32, 64), 4, -1, true, ctx),
                     Status::kInvalidArgument, "gptq act-order needs perm_out");
  KOPS_EXPECT_STATUS(kops::quant::gptq_to_marlin(tensor2d(b.p, DType::kInt32, 4, 128), tensor2d(os.p, DType::kFloat16, 1, 64),
                                                 tensor1d(c.p, DType::kInt32, 64), tensor2d(a.p, DType::kInt32, 8, 64),
                                                 tensor2d(s.p, DType::kFloat16, 1, 64), Tensor{}, 3, -1, true, ctx),
                     Status::kUnsupportedShape, "gptq num_bits");
  KOPS_EXPECT_STATUS(kops::quant::fp4_to_marlin(tensor2d(b.p, DType::kInt32, 8, 128), tensor2d(os.p, DType::kFloat8E8M0, 4, 64),
                                                tensor2d(a.p, DType::kUInt8, 64, 64), tensor2d(s.p, DType::kFloat8E8M0, 64, 4), 16, true, ctx),
                     Status::kUnsupportedShape, "fp4 block vs ue8m0");
  KOPS_EXPECT_STATUS(kops::quant::awq_dequantize(tensor2d(b.p, DType::kFloat16, 64, 128), tensor2d(a.p, DType::kInt32, 64, 16),
                                                 tensor2d(s.p, DType::kFloat32, 2, 128), tensor2d(c.p, DType::kInt32, 2, 16), 32, ctx),
                     Status::kUnsupportedDtype, "awq_dequantize out dtype != scales dtype");
  KOPS_EXPECT_STATUS(kops::quant::awq_dequantize(tensor2d(b.p, DType::kFloat16, 64, 128), tensor2d(a.p, DType::kInt32, 64, 16),
                                                 tensor2d(s.p, DType::kFloat16, 2, 128), tensor2d(c.p, DType::kInt32, 2, 16), 48, ctx),
                     Status::kUnsupportedShape, "awq_dequantize group does not divide K");
  // GGUF: K must be a whole number of blocks and the byte count must match.
  DevBuf<uint8_t> raw(8192);
  KOPS_EXPECT_STATUS(kops::quant::gguf_dequantize(tensor2d(b.p, DType::kFloat32, 1, 48), tensor1d(raw.p, DType::kUInt8, 90),
                                                  kops::GGUFType::kQ4_0, ctx),
                     Status::kUnsupportedShape, "gguf K % 32");
  KOPS_EXPECT_STATUS(kops::quant::gguf_dequantize(tensor2d(b.p, DType::kFloat32, 1, 64), tensor1d(raw.p, DType::kUInt8, 35),
                                                  kops::GGUFType::kQ4_0, ctx),
                     Status::kInvalidArgument, "gguf byte count");
  KOPS_EXPECT_STATUS(kops::quant::gguf_dequantize(tensor2d(b.p, DType::kInt32, 1, 64), tensor1d(raw.p, DType::kUInt8, 36),
                                                  kops::GGUFType::kQ4_0, ctx),
                     Status::kUnsupportedDtype, "gguf out dtype");
  // FP8 -> UE8M0 and SmoothQuant.
  KOPS_EXPECT_STATUS(kops::quant::fp8_block_scale_to_ue8m0(tensor2d(raw.p, DType::kInt8, 128, 128), tensor2d(a.p, DType::kFloat32, 1, 1), Tensor{}, ctx),
                     Status::kUnsupportedDtype, "ue8m0 weight dtype");
  KOPS_EXPECT_STATUS(kops::quant::fp8_block_scale_to_ue8m0(tensor2d(raw.p, DType::kFloat8E4M3, 128, 128), tensor2d(a.p, DType::kFloat32, 2, 1), Tensor{}, ctx),
                     Status::kInvalidArgument, "ue8m0 scales shape");
  KOPS_EXPECT_STATUS(kops::quant::smoothquant_compute_scales(tensor1d(a.p, DType::kFloat32, 8), tensor1d(b.p, DType::kFloat32, 8),
                                                             tensor1d(c.p, DType::kFloat32, 8), 1.5f, ctx),
                     Status::kInvalidArgument, "smoothquant alpha");
  KOPS_EXPECT_STATUS(kops::quant::smoothquant_fold(tensor1d(a.p, DType::kFloat16, 8), Tensor{}, tensor2d(b.p, DType::kBFloat16, 4, 8),
                                                   tensor1d(c.p, DType::kFloat32, 8), ctx),
                     Status::kUnsupportedDtype, "smoothquant norm dtype != linear dtype");
}

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
  test_absmax(ctx);
  test_per_tensor_fp8(ctx);
  test_per_tensor_int8(ctx);
  test_per_token_fp8(ctx);
  test_per_token_int8_asym(ctx);
  test_quant_pack_edges(ctx);
  test_per_group(ctx);
  test_per_group_vector(ctx);
  test_per_group_masked(ctx);
  test_per_group_math(ctx);
  test_per_group_graph(ctx);
  test_per_group_contracts(ctx);
  test_mx(ctx);
  test_nvfp4(ctx);
  test_dequantize(ctx);
  test_int4_pack(ctx);
  test_hadamard(ctx);
  test_contracts(ctx);
  test_graph_replay(ctx);
  test_b4_marlin(ctx);
  test_b4_fp4(ctx);
  test_b4_awq_dequant(ctx);
  test_b4_gguf(ctx);
  test_b4_ue8m0(ctx);
  test_b4_smoothquant(ctx);
  test_b4_contracts(ctx);
  return kops_test::finish("test_quantization");
}
