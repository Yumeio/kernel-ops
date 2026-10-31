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
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/ops.h"

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

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
  test_absmax(ctx);
  test_per_tensor_fp8(ctx);
  test_per_tensor_int8(ctx);
  test_per_token_fp8(ctx);
  test_per_token_int8_asym(ctx);
  test_per_group(ctx);
  test_mx(ctx);
  test_nvfp4(ctx);
  test_dequantize(ctx);
  test_int4_pack(ctx);
  test_hadamard(ctx);
  test_contracts(ctx);
  test_graph_replay(ctx);
  return kops_test::finish("test_quantization");
}
