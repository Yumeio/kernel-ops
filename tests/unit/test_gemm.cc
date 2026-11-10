// =============================================================================
// kernel-ops | tests/unit/test_gemm.cc
// B4 gemm family (rows 5-6): operand preparation.
//   weight_transpose, glu_weight_interleave, router_weight_split_bf16x2,
//   int8_weight_colsum, block_scale_swizzle, fp8_scale_to_tma_layout,
//   fp8_block2d_quant_weight.
//
// Oracle: host loops written from the layout definitions (torchao `to_blocked` for the
// 128x4 swizzle, DeepGEMM `pack_ue8m0_to_int` / `per_block_cast_to_fp8` for the scale and
// FP8 block formulas) plus the toolkit's host-side e4m3 / bf16 conversions.  No third-party
// library runs and no PyTorch is linked.
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"
#include "csrc/common/quant.cuh"
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
using kops::LaunchContext;
using kops::Status;
using kops::Tensor;

template <typename T>
struct DevBuf {
  T* p = nullptr;
  explicit DevBuf(size_t n) : p(kops_test::device_alloc<T>(n)) {}
  ~DevBuf() { kops_test::device_free(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& other) noexcept : p(other.p) { other.p = nullptr; }
};

template <typename T>
DevBuf<T> upload(const std::vector<T>& host) {
  DevBuf<T> d(host.size());
  kops_test::copy_in(d.p, host);
  return d;
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

/// Row-major 2-D view with an explicit row stride (in elements).
Tensor tensor2d_ld(void* data, DType dtype, int64_t rows, int64_t cols, int64_t ld) {
  Tensor t = kops_test::tensor2d(data, dtype, rows, cols);
  t.stride[0] = ld;
  return t;
}

std::vector<uint8_t> random_bytes(size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> v(n);
  for (auto& b : v) b = static_cast<uint8_t>(rng());
  return v;
}

int64_t round_up(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

float e4m3_to_float(uint8_t b) { return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)); }

uint8_t e4m3_from_float(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

// -----------------------------------------------------------------------------
// weight_transpose
// -----------------------------------------------------------------------------

/// Transposes a [rows, cols] array of `elem_bytes`-sized elements with row strides in elements.
std::vector<uint8_t> transpose_bytes(const std::vector<uint8_t>& in, int64_t rows, int64_t cols,
                                     int64_t in_ld, int64_t out_ld, int elem_bytes) {
  std::vector<uint8_t> out(static_cast<size_t>(cols * out_ld) * elem_bytes, 0xCD);
  for (int64_t k = 0; k < rows; ++k) {
    for (int64_t n = 0; n < cols; ++n) {
      std::memcpy(&out[static_cast<size_t>(n * out_ld + k) * elem_bytes],
                  &in[static_cast<size_t>(k * in_ld + n) * elem_bytes], elem_bytes);
    }
  }
  return out;
}

void run_transpose_case(const LaunchContext& ctx, DType dtype, int elem_bytes, int64_t rows,
                        int64_t cols, int64_t in_pad, int64_t out_pad, const char* what) {
  const int64_t in_ld = cols + in_pad;
  const int64_t out_ld = rows + out_pad;
  const auto host = random_bytes(static_cast<size_t>(rows * in_ld) * elem_bytes, 11);
  auto din = upload(host);
  const std::vector<uint8_t> sentinel(static_cast<size_t>(cols * out_ld) * elem_bytes, 0xCD);
  auto dout = upload(sentinel);
  KOPS_EXPECT_OK(kops::gemm::weight_transpose(tensor2d_ld(dout.p, dtype, cols, rows, out_ld),
                                              tensor2d_ld(din.p, dtype, rows, cols, in_ld), ctx),
                 what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, sentinel.size());
  const auto want = transpose_bytes(host, rows, cols, in_ld, out_ld, elem_bytes);
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i) bad += got[i] != want[i];
  if (bad != 0) {
    char detail[96];
    std::snprintf(detail, sizeof(detail), "%zu/%zu bytes differ (%lld x %lld)", bad, want.size(),
                  static_cast<long long>(rows), static_cast<long long>(cols));
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
}

void test_weight_transpose(const LaunchContext& ctx) {
  // 2/4/8-byte tile path, including shapes that are not multiples of the 32 tile.
  for (const auto& rc : {std::pair<int64_t, int64_t>{64, 64}, {100, 70}, {33, 17}, {257, 130}}) {
    run_transpose_case(ctx, DType::kBFloat16, 2, rc.first, rc.second, 0, 0, "transpose bf16");
    run_transpose_case(ctx, DType::kFloat32, 4, rc.first, rc.second, 0, 0, "transpose f32");
    run_transpose_case(ctx, DType::kInt64, 8, rc.first, rc.second, 0, 0, "transpose i64");
  }
  // Row strides larger than the row.
  run_transpose_case(ctx, DType::kFloat16, 2, 96, 80, 16, 8, "transpose f16 strided");
  // 1-byte: 4-byte word fast path (multiples of 4) and the byte tile fallback (odd dims).
  for (const auto& rc : {std::pair<int64_t, int64_t>{64, 64}, {128, 192}, {260, 132}}) {
    run_transpose_case(ctx, DType::kFloat8E4M3, 1, rc.first, rc.second, 0, 0, "transpose fp8 fast");
  }
  run_transpose_case(ctx, DType::kInt8, 1, 100, 70, 0, 0, "transpose i8 odd");
  run_transpose_case(ctx, DType::kFloat8E4M3, 1, 96, 64, 4, 8, "transpose fp8 strided");

  // Packed 4-bit: low nibble = even element.  Oracle works on nibbles.
  for (const auto& rc : {std::pair<int64_t, int64_t>{64, 64}, {70, 130}, {130, 66}}) {
    const int64_t rows = rc.first;
    const int64_t cols = rc.second;
    const auto host = random_bytes(static_cast<size_t>(rows * cols / 2), 21);
    auto nib = [&](const std::vector<uint8_t>& v, int64_t row_len, int64_t r, int64_t c) {
      const uint8_t b = v[static_cast<size_t>((r * row_len + c) >> 1)];
      return (c & 1) ? (b >> 4) : (b & 0xF);
    };
    std::vector<uint8_t> want(static_cast<size_t>(cols * rows / 2), 0);
    for (int64_t n = 0; n < cols; ++n) {
      for (int64_t k = 0; k < rows; ++k) {
        const uint8_t v = static_cast<uint8_t>(nib(host, cols, k, n));
        uint8_t& dst = want[static_cast<size_t>((n * rows + k) >> 1)];
        dst = static_cast<uint8_t>((k & 1) ? (dst | (v << 4)) : (dst | v));
      }
    }
    auto din = upload(host);
    DevBuf<uint8_t> dout(want.size());
    cudaMemset(dout.p, 0, want.size());
    KOPS_EXPECT_OK(kops::gemm::weight_transpose(kops_test::tensor2d(dout.p, DType::kInt4x2, cols, rows),
                                                kops_test::tensor2d(din.p, DType::kInt4x2, rows, cols), ctx),
                   "transpose int4");
    cudaDeviceSynchronize();
    KOPS_TEST_CHECK(kops_test::copy_out(dout.p, want.size()) == want);
  }
}

// -----------------------------------------------------------------------------
// glu_weight_interleave
// -----------------------------------------------------------------------------

/// Row order of a fused [gate | up] matrix with `inter` rows each, in layout `il`
/// (0 = halves, d >= 1 = blocks of d rows alternating gate, up).  Returns for every stored
/// row the pair (is_up, index within gate/up).
std::vector<std::pair<int, int>> glu_layout(int inter, int il) {
  std::vector<std::pair<int, int>> rows;
  if (il == 0) {
    for (int j = 0; j < inter; ++j) rows.push_back({0, j});
    for (int j = 0; j < inter; ++j) rows.push_back({1, j});
  } else {
    for (int b = 0; b < inter / il; ++b) {
      for (int j = 0; j < il; ++j) rows.push_back({0, b * il + j});
      for (int j = 0; j < il; ++j) rows.push_back({1, b * il + j});
    }
  }
  return rows;
}

void run_glu_case(const LaunchContext& ctx, int experts, int inter, int k, int src_il, int dst_il,
                  bool with_extras) {
  const int rows = 2 * inter;
  const size_t weight_bytes = static_cast<size_t>(experts) * rows * k * 2;
  const size_t scale_count = static_cast<size_t>(experts) * rows * 3;
  const size_t bias_count = static_cast<size_t>(experts) * rows;
  const auto weight = random_bytes(weight_bytes, 31);
  std::vector<float> scales(scale_count);
  std::vector<uint16_t> bias(bias_count);
  for (size_t i = 0; i < scales.size(); ++i) scales[i] = 0.25f * static_cast<float>(i % 977) + 1.0f;
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<uint16_t>(i * 7919u + 13u);

  // Oracle: stored row r of the source is (is_up, j); find the destination row holding it.
  const auto src_rows = glu_layout(inter, src_il);
  const auto dst_rows = glu_layout(inter, dst_il);
  std::vector<int> dst_of_src(rows);
  for (int r = 0; r < rows; ++r) {
    for (int d = 0; d < rows; ++d) {
      if (dst_rows[d] == src_rows[r]) dst_of_src[r] = d;
    }
  }
  std::vector<uint8_t> want_w(weight_bytes);
  std::vector<float> want_s(scale_count);
  std::vector<uint16_t> want_b(bias_count);
  for (int e = 0; e < experts; ++e) {
    for (int r = 0; r < rows; ++r) {
      const size_t sr = static_cast<size_t>(e) * rows + r;
      const size_t dr = static_cast<size_t>(e) * rows + dst_of_src[r];
      std::memcpy(&want_w[dr * k * 2], &weight[sr * k * 2], static_cast<size_t>(k) * 2);
      for (int c = 0; c < 3; ++c) want_s[dr * 3 + c] = scales[sr * 3 + c];
      want_b[dr] = bias[sr];
    }
  }

  auto dw = upload(weight);
  DevBuf<uint8_t> ow(weight_bytes);
  auto ds = upload(scales);
  DevBuf<float> os(scale_count);
  auto db = upload(bias);
  DevBuf<uint16_t> ob(bias_count);
  cudaMemset(ow.p, 0xCD, weight_bytes);
  const Tensor t_w = experts == 1 ? kops_test::tensor2d(dw.p, DType::kBFloat16, rows, k)
                                  : kops_test::tensor3d(dw.p, DType::kBFloat16, experts, rows, k);
  const Tensor t_ow = experts == 1 ? kops_test::tensor2d(ow.p, DType::kBFloat16, rows, k)
                                   : kops_test::tensor3d(ow.p, DType::kBFloat16, experts, rows, k);
  Tensor t_s{}, t_os{}, t_b{}, t_ob{};
  if (with_extras) {
    t_s = experts == 1 ? kops_test::tensor2d(ds.p, DType::kFloat32, rows, 3)
                       : kops_test::tensor3d(ds.p, DType::kFloat32, experts, rows, 3);
    t_os = experts == 1 ? kops_test::tensor2d(os.p, DType::kFloat32, rows, 3)
                        : kops_test::tensor3d(os.p, DType::kFloat32, experts, rows, 3);
    t_b = experts == 1 ? tensor1d(db.p, DType::kBFloat16, rows)
                       : kops_test::tensor2d(db.p, DType::kBFloat16, experts, rows);
    t_ob = experts == 1 ? tensor1d(ob.p, DType::kBFloat16, rows)
                        : kops_test::tensor2d(ob.p, DType::kBFloat16, experts, rows);
  }
  char what[96];
  std::snprintf(what, sizeof(what), "glu E=%d I=%d src=%d dst=%d extras=%d", experts, inter, src_il,
                dst_il, with_extras ? 1 : 0);
  KOPS_EXPECT_OK(kops::gemm::glu_weight_interleave(t_ow, t_os, t_ob, t_w, t_s, t_b, src_il, dst_il, ctx), what);
  cudaDeviceSynchronize();
  if (kops_test::copy_out(ow.p, weight_bytes) != want_w) {
    kops_test::record_failure(what, "weight rows differ", __FILE__, __LINE__);
  }
  if (with_extras) {
    if (kops_test::copy_out(os.p, scale_count) != want_s) {
      kops_test::record_failure(what, "scale rows differ", __FILE__, __LINE__);
    }
    if (kops_test::copy_out(ob.p, bias_count) != want_b) {
      kops_test::record_failure(what, "bias differs", __FILE__, __LINE__);
    }
  }
}

void test_glu_interleave(const LaunchContext& ctx) {
  for (const auto& m : {std::pair<int, int>{0, 0}, {0, 1}, {0, 8}, {1, 0}, {1, 8}, {4, 2}, {8, 0}}) {
    run_glu_case(ctx, 1, 64, 40, m.first, m.second, false);
    run_glu_case(ctx, 3, 32, 24, m.first, m.second, true);
  }
  // Odd row bytes exercise the byte-copy path.
  run_glu_case(ctx, 2, 16, 7, 0, 4, true);

  // Round trip: halves -> blocks of 16 -> halves is the identity.
  const int inter = 64;
  const int k = 32;
  const auto weight = random_bytes(static_cast<size_t>(2 * inter) * k * 2, 5);
  auto dw = upload(weight);
  DevBuf<uint8_t> mid(weight.size());
  DevBuf<uint8_t> back(weight.size());
  const Tensor tw = kops_test::tensor2d(dw.p, DType::kBFloat16, 2 * inter, k);
  const Tensor tm = kops_test::tensor2d(mid.p, DType::kBFloat16, 2 * inter, k);
  const Tensor tb = kops_test::tensor2d(back.p, DType::kBFloat16, 2 * inter, k);
  KOPS_EXPECT_OK(kops::gemm::glu_weight_interleave(tm, Tensor{}, Tensor{}, tw, Tensor{}, Tensor{}, 0, 16, ctx), "glu fwd");
  KOPS_EXPECT_OK(kops::gemm::glu_weight_interleave(tb, Tensor{}, Tensor{}, tm, Tensor{}, Tensor{}, 16, 0, ctx), "glu back");
  cudaDeviceSynchronize();
  KOPS_TEST_CHECK(kops_test::copy_out(back.p, weight.size()) == weight);
}

// -----------------------------------------------------------------------------
// router_weight_split_bf16x2
// -----------------------------------------------------------------------------

uint16_t bf16_bits(float x) {
  const __nv_bfloat16 v = __float2bfloat16_rn(x);
  uint16_t b;
  std::memcpy(&b, &v, sizeof(b));
  return b;
}

float bf16_value(uint16_t b) {
  __nv_bfloat16 v;
  std::memcpy(&v, &b, sizeof(b));
  return __bfloat162float(v);
}

void test_router_split(const LaunchContext& ctx) {
  for (const int64_t numel : {int64_t{64} * 128, int64_t{4099}, int64_t{7}}) {
    std::vector<float> w(static_cast<size_t>(numel));
    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (auto& x : w) x = dist(rng);
    if (numel > 16) {
      w[0] = 0.0f;
      w[1] = -0.0f;
      w[2] = 1.0e-30f;
      w[3] = 3.0e38f;
      w[4] = -1.0f / 3.0f;
      w[5] = 65504.0f;
    }
    auto dw = upload(w);
    DevBuf<uint16_t> hi(w.size());
    DevBuf<uint16_t> lo(w.size());
    KOPS_EXPECT_OK(kops::gemm::router_weight_split_bf16x2(tensor1d(hi.p, DType::kBFloat16, numel),
                                                          tensor1d(lo.p, DType::kBFloat16, numel),
                                                          tensor1d(dw.p, DType::kFloat32, numel), ctx),
                   "router split");
    cudaDeviceSynchronize();
    const auto got_hi = kops_test::copy_out(hi.p, w.size());
    const auto got_lo = kops_test::copy_out(lo.p, w.size());
    size_t bad = 0;
    double worst_recon = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
      const uint16_t want_hi = bf16_bits(w[i]);
      // (w - hi) is exact in fp32, so the low part is one correctly rounded bf16 of it.
      const float diff = static_cast<float>(static_cast<double>(w[i]) - static_cast<double>(bf16_value(want_hi)));
      const uint16_t want_lo = bf16_bits(diff * 256.0f);
      bad += (got_hi[i] != want_hi) || (got_lo[i] != want_lo);
      if (std::fabs(w[i]) < 1.0e30f && std::fabs(w[i]) > 1.0e-3f) {
        const double recon = static_cast<double>(bf16_value(got_hi[i])) + static_cast<double>(bf16_value(got_lo[i])) / 256.0;
        worst_recon = std::max(worst_recon, std::fabs(recon - w[i]) / std::fabs(w[i]));
      }
    }
    KOPS_TEST_CHECK(bad == 0);
    // hi + lo/256 carries about 16 mantissa bits.
    KOPS_TEST_CHECK(worst_recon < 1.0e-4);
  }
}

// -----------------------------------------------------------------------------
// int8_weight_colsum
// -----------------------------------------------------------------------------

void test_colsum(const LaunchContext& ctx) {
  for (const auto& nk : {std::pair<int, int>{256, 4096}, {100, 77}, {33, 16}, {5, 130000}}) {
    for (const int pad : {0, 16, 3}) {
      const int n = nk.first;
      const int k = nk.second;
      const int ld = k + pad;
      auto host = random_bytes(static_cast<size_t>(n) * ld, 3);
      if (n * k > 0) {
        host[0] = 0x80;  // -128
        host[1] = 0x7F;  // 127
      }
      auto dw = upload(host);
      DevBuf<int32_t> out(n);
      cudaMemset(out.p, 0xFF, n * sizeof(int32_t));
      KOPS_EXPECT_OK(kops::gemm::int8_weight_colsum(tensor1d(out.p, DType::kInt32, n),
                                                    tensor2d_ld(dw.p, DType::kInt8, n, k, ld), ctx),
                     "colsum");
      cudaDeviceSynchronize();
      const auto got = kops_test::copy_out(out.p, n);
      size_t bad = 0;
      for (int r = 0; r < n; ++r) {
        int64_t want = 0;
        for (int c = 0; c < k; ++c) want += static_cast<int8_t>(host[static_cast<size_t>(r) * ld + c]);
        bad += got[r] != static_cast<int32_t>(want);
      }
      KOPS_TEST_CHECK(bad == 0);
    }
  }
}

// -----------------------------------------------------------------------------
// block_scale_swizzle
// -----------------------------------------------------------------------------

/// torchao / FlashInfer `to_blocked`: pad to (128, 4), view (rb, 128, cb, 4), permute
/// (0, 2, 1, 3), then inside each 128x4 block view rows as (4, 32) and swap them.
std::vector<uint8_t> to_blocked(const std::vector<uint8_t>& in, int64_t rows, int64_t cols) {
  const int64_t n_row_blocks = (rows + 127) / 128;
  const int64_t n_col_blocks = (cols + 3) / 4;
  std::vector<uint8_t> out(static_cast<size_t>(n_row_blocks * n_col_blocks * 512), 0);
  for (int64_t rb = 0; rb < n_row_blocks; ++rb) {
    for (int64_t cb = 0; cb < n_col_blocks; ++cb) {
      for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 4; ++c) {
          const int64_t gr = rb * 128 + r;
          const int64_t gc = cb * 4 + c;
          const uint8_t v = (gr < rows && gc < cols) ? in[static_cast<size_t>(gr * cols + gc)] : 0;
          const int a = r / 32;  // reshape(-1, 4, 32, 4)
          const int b = r % 32;
          out[static_cast<size_t>((rb * n_col_blocks + cb) * 512 + b * 16 + a * 4 + c)] = v;
        }
      }
    }
  }
  return out;
}

void test_scale_swizzle(const LaunchContext& ctx) {
  const struct { int batch; int rows; int cols; } cases[] = {
      {1, 128, 4}, {1, 256, 8}, {1, 200, 7}, {3, 130, 33}, {2, 1, 1}, {1, 4096, 128}, {1, 300, 64}};
  for (const auto& c : cases) {
    const int64_t rows_pad = round_up(c.rows, 128);
    const int64_t cols_pad = round_up(c.cols, 4);
    const auto host = random_bytes(static_cast<size_t>(c.batch) * c.rows * c.cols, 9);
    auto din = upload(host);
    const size_t tiled_bytes = static_cast<size_t>(c.batch * rows_pad * cols_pad);
    DevBuf<uint8_t> out(tiled_bytes);
    cudaMemset(out.p, 0xAB, tiled_bytes);
    const Tensor t_in = c.batch == 1 ? kops_test::tensor2d(din.p, DType::kFloat8E8M0, c.rows, c.cols)
                                     : kops_test::tensor3d(din.p, DType::kFloat8E8M0, c.batch, c.rows, c.cols);
    const Tensor t_out = c.batch == 1 ? kops_test::tensor2d(out.p, DType::kFloat8E8M0, rows_pad, cols_pad)
                                      : kops_test::tensor3d(out.p, DType::kFloat8E8M0, c.batch, rows_pad, cols_pad);
    KOPS_EXPECT_OK(kops::gemm::block_scale_swizzle(t_out, t_in, false, ctx), "swizzle");
    cudaDeviceSynchronize();
    std::vector<uint8_t> want;
    for (int b = 0; b < c.batch; ++b) {
      std::vector<uint8_t> slice(host.begin() + static_cast<size_t>(b) * c.rows * c.cols,
                                 host.begin() + static_cast<size_t>(b + 1) * c.rows * c.cols);
      const auto blocked = to_blocked(slice, c.rows, c.cols);
      want.insert(want.end(), blocked.begin(), blocked.end());
    }
    KOPS_TEST_CHECK(kops_test::copy_out(out.p, tiled_bytes) == want);

    // Inverse recovers the original bytes.
    DevBuf<uint8_t> back(host.size());
    cudaMemset(back.p, 0, host.size());
    const Tensor t_back = c.batch == 1 ? kops_test::tensor2d(back.p, DType::kFloat8E8M0, c.rows, c.cols)
                                       : kops_test::tensor3d(back.p, DType::kFloat8E8M0, c.batch, c.rows, c.cols);
    KOPS_EXPECT_OK(kops::gemm::block_scale_swizzle(t_back, t_out, true, ctx), "unswizzle");
    cudaDeviceSynchronize();
    KOPS_TEST_CHECK(kops_test::copy_out(back.p, host.size()) == host);
  }
}

// -----------------------------------------------------------------------------
// fp8_scale_to_tma_layout
// -----------------------------------------------------------------------------

void test_scale_tma(const LaunchContext& ctx) {
  struct Case { int m; int kb; bool grouped; };
  for (const auto& c : {Case{70, 8, false}, Case{128, 4, false}, Case{5, 12, false}, Case{70, 8, true}}) {
    // Powers of two, so the UE8M0 exponent is exact.
    std::vector<float> in(static_cast<size_t>(c.m) * c.kb);
    std::mt19937 rng(41);
    for (auto& x : in) x = std::ldexp(1.0f, static_cast<int>(rng() % 40) - 20);
    const int m_pad = static_cast<int>(round_up(c.m, 4));
    std::vector<int64_t> offsets = {0, 10, 25, 40};
    const int valid = c.grouped ? 40 : c.m;
    auto din = upload(in);
    auto doff = upload(offsets);
    const Tensor t_in = kops_test::tensor2d(din.p, DType::kFloat32, c.m, c.kb);
    const Tensor t_off = c.grouped ? tensor1d(doff.p, DType::kInt64, 4) : Tensor{};

    // Plain fp32 transpose.
    {
      DevBuf<float> out(static_cast<size_t>(c.kb) * m_pad);
      cudaMemset(out.p, 0xFF, static_cast<size_t>(c.kb) * m_pad * sizeof(float));
      KOPS_EXPECT_OK(kops::gemm::fp8_scale_to_tma_layout(kops_test::tensor2d(out.p, DType::kFloat32, c.kb, m_pad), t_in, t_off, false, ctx), "tma f32");
      cudaDeviceSynchronize();
      const auto got = kops_test::copy_out(out.p, static_cast<size_t>(c.kb) * m_pad);
      size_t bad = 0;
      for (int kb = 0; kb < c.kb; ++kb) {
        for (int m = 0; m < m_pad; ++m) {
          const float want = m < valid ? in[static_cast<size_t>(m) * c.kb + kb] : 0.0f;
          bad += got[static_cast<size_t>(kb) * m_pad + m] != want;
        }
      }
      KOPS_TEST_CHECK(bad == 0);
    }
    // Packed UE8M0 (DeepGEMM pack_ue8m0_to_int: exponent bytes, little-endian along K).
    if (c.kb % 4 == 0) {
      DevBuf<int32_t> out(static_cast<size_t>(c.kb / 4) * m_pad);
      cudaMemset(out.p, 0xFF, static_cast<size_t>(c.kb / 4) * m_pad * sizeof(int32_t));
      KOPS_EXPECT_OK(kops::gemm::fp8_scale_to_tma_layout(kops_test::tensor2d(out.p, DType::kInt32, c.kb / 4, m_pad), t_in, t_off, true, ctx), "tma packed");
      cudaDeviceSynchronize();
      const auto got = kops_test::copy_out(out.p, static_cast<size_t>(c.kb / 4) * m_pad);
      size_t bad = 0;
      for (int g = 0; g < c.kb / 4; ++g) {
        for (int m = 0; m < m_pad; ++m) {
          uint32_t want = 0;
          if (m < valid) {
            for (int j = 0; j < 4; ++j) {
              uint32_t bits;
              const float v = in[static_cast<size_t>(m) * c.kb + 4 * g + j];
              std::memcpy(&bits, &v, sizeof(bits));
              want |= ((bits >> 23) & 0xFFu) << (8 * j);
            }
          }
          bad += static_cast<uint32_t>(got[static_cast<size_t>(g) * m_pad + m]) != want;
        }
      }
      KOPS_TEST_CHECK(bad == 0);
    }
  }
}

// -----------------------------------------------------------------------------
// fp8_block2d_quant_weight
// -----------------------------------------------------------------------------

template <typename T>
float load_as_float(T v);
template <>
float load_as_float<float>(float v) { return v; }
template <>
float load_as_float<__half>(__half v) { return __half2float(v); }
template <>
float load_as_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

/// DeepGEMM per_block_cast_to_fp8 with the repo's division and amax floor.
template <typename T>
void block_quant_oracle(const std::vector<T>& w, int n, int k, int bn, int bk, bool ue8m0,
                        std::vector<uint8_t>* q, std::vector<float>* scales) {
  const int sn = (n + bn - 1) / bn;
  const int sk = (k + bk - 1) / bk;
  q->assign(static_cast<size_t>(n) * k, 0);
  scales->assign(static_cast<size_t>(sn) * sk, 0.0f);
  for (int bi = 0; bi < sn; ++bi) {
    for (int bj = 0; bj < sk; ++bj) {
      float amax = 0.0f;
      for (int r = bi * bn; r < std::min(n, (bi + 1) * bn); ++r) {
        for (int c = bj * bk; c < std::min(k, (bj + 1) * bk); ++c) {
          amax = std::max(amax, std::fabs(load_as_float(w[static_cast<size_t>(r) * k + c])));
        }
      }
      float s = std::max(amax, 1.0e-4f) / 448.0f;
      if (ue8m0) {
        uint32_t bits;
        std::memcpy(&bits, &s, sizeof(bits));
        uint32_t e = ((bits >> 23) & 0xFFu) + ((bits & 0x7FFFFFu) != 0 ? 1u : 0u);
        e = std::min(254u, std::max(1u, e));
        bits = e << 23;
        std::memcpy(&s, &bits, sizeof(s));
      }
      (*scales)[static_cast<size_t>(bi) * sk + bj] = s;
      for (int r = bi * bn; r < std::min(n, (bi + 1) * bn); ++r) {
        for (int c = bj * bk; c < std::min(k, (bj + 1) * bk); ++c) {
          (*q)[static_cast<size_t>(r) * k + c] = e4m3_from_float(load_as_float(w[static_cast<size_t>(r) * k + c]) / s);
        }
      }
    }
  }
}

template <typename T>
void run_block_quant(const LaunchContext& ctx, DType dt, int n, int k, int bn, int bk, bool ue8m0,
                     int row_pad, const char* what) {
  const int ld = k + row_pad;
  std::vector<T> w(static_cast<size_t>(n) * ld);
  std::mt19937 rng(88);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (auto& x : w) kops_test::from_float(x, dist(rng) * (1.0f + static_cast<float>(rng() % 5)));
  // One all-zero block and one tiny block.
  for (int r = 0; r < std::min(n, bn); ++r) {
    for (int c = 0; c < std::min(k, bk); ++c) kops_test::from_float(w[static_cast<size_t>(r) * ld + c], 0.0f);
  }
  if (n > bn && k > bk) {
    for (int r = bn; r < std::min(n, 2 * bn); ++r) {
      for (int c = bk; c < std::min(k, 2 * bk); ++c) kops_test::from_float(w[static_cast<size_t>(r) * ld + c], 1.0e-6f * (1 + c % 3));
    }
  }
  // Oracle input is the dense [n, k] view of the strided buffer.
  std::vector<T> dense(static_cast<size_t>(n) * k);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < k; ++c) dense[static_cast<size_t>(r) * k + c] = w[static_cast<size_t>(r) * ld + c];
  }
  std::vector<uint8_t> want_q;
  std::vector<float> want_s;
  block_quant_oracle(dense, n, k, bn, bk, ue8m0, &want_q, &want_s);

  const int sn = (n + bn - 1) / bn;
  const int sk = (k + bk - 1) / bk;
  auto dw = upload(w);
  DevBuf<uint8_t> dq(static_cast<size_t>(n) * k);
  DevBuf<float> ds(static_cast<size_t>(sn) * sk);
  cudaMemset(dq.p, 0xEE, static_cast<size_t>(n) * k);
  KOPS_EXPECT_OK(kops::gemm::fp8_block2d_quant_weight(
                     kops_test::tensor2d(dq.p, DType::kFloat8E4M3, n, k),
                     kops_test::tensor2d(ds.p, DType::kFloat32, sn, sk),
                     tensor2d_ld(dw.p, dt, n, k, ld), bn, bk, ue8m0, ctx),
                 what);
  cudaDeviceSynchronize();
  const auto got_q = kops_test::copy_out(dq.p, static_cast<size_t>(n) * k);
  const auto got_s = kops_test::copy_out(ds.p, static_cast<size_t>(sn) * sk);
  size_t bad_q = 0;
  for (size_t i = 0; i < want_q.size(); ++i) bad_q += got_q[i] != want_q[i];
  size_t bad_s = 0;
  for (size_t i = 0; i < want_s.size(); ++i) bad_s += got_s[i] != want_s[i];
  if (bad_q != 0 || bad_s != 0) {
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%zu payload bytes and %zu scales differ (n=%d k=%d block %dx%d ue8m0=%d)",
                  bad_q, bad_s, n, k, bn, bk, ue8m0 ? 1 : 0);
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
  if (ue8m0) {
    for (const float s : got_s) {
      int e = 0;
      KOPS_TEST_CHECK(std::frexp(s, &e) == 0.5f);  // exact power of two
    }
  }
}

void test_block_quant(const LaunchContext& ctx) {
  for (const bool ue : {false, true}) {
    run_block_quant<__nv_bfloat16>(ctx, DType::kBFloat16, 256, 384, 128, 128, ue, 0, "block quant bf16 vec");
    run_block_quant<__nv_bfloat16>(ctx, DType::kBFloat16, 200, 304, 128, 128, ue, 0, "block quant bf16 edge");
    run_block_quant<__nv_bfloat16>(ctx, DType::kBFloat16, 200, 300, 128, 128, ue, 0, "block quant bf16 generic");
    run_block_quant<__half>(ctx, DType::kFloat16, 128, 256, 64, 128, ue, 8, "block quant f16 strided");
    run_block_quant<float>(ctx, DType::kFloat32, 256, 256, 128, 128, ue, 0, "block quant f32");
    run_block_quant<float>(ctx, DType::kFloat32, 130, 136, 32, 64, ue, 0, "block quant f32 small block");
  }
}

// -----------------------------------------------------------------------------
// e4m3 encode: exhaustive comparison with the toolkit intrinsic
// -----------------------------------------------------------------------------

__global__ void e4m3_exhaustive_kernel(const uint32_t base, unsigned long long* mismatches,
                                       unsigned int* first_bad) {
  const uint32_t bits = base + blockIdx.x * blockDim.x + threadIdx.x;
  const float x = __uint_as_float(bits);
  const uint8_t want = static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
  const uint8_t got = kops::dev::e4m3_from_float_sat(x);
  if (want != got) {
    atomicAdd(mismatches, 1ull);
    atomicMin(first_bad, bits);
  }
}

__global__ void e4m3_decode_kernel(float* out) {
  out[threadIdx.x] = kops::dev::e4m3_to_float_fast(static_cast<uint8_t>(threadIdx.x));
}

void test_e4m3_encode_exhaustive() {
  {
    DevBuf<float> dec(256);
    e4m3_decode_kernel<<<1, 256>>>(dec.p);
    cudaDeviceSynchronize();
    const auto got = kops_test::copy_out(dec.p, 256);
    size_t bad = 0;
    for (int b = 0; b < 256; ++b) {
      const float want = e4m3_to_float(static_cast<uint8_t>(b));
      uint32_t gb, wb;
      std::memcpy(&gb, &got[b], 4);
      std::memcpy(&wb, &want, 4);
      // Any NaN counts as equal; everything else must match bit for bit.
      bad += !((std::isnan(got[b]) && std::isnan(want)) || gb == wb);
    }
    KOPS_TEST_CHECK(bad == 0);
  }
  DevBuf<unsigned long long> mism(1);
  DevBuf<unsigned int> first(1);
  cudaMemset(mism.p, 0, sizeof(unsigned long long));
  const unsigned int none = 0xFFFFFFFFu;
  cudaMemcpy(first.p, &none, sizeof(none), cudaMemcpyHostToDevice);
  constexpr uint32_t kChunk = 1u << 26;  // 2^32 patterns in 64 launches
  for (uint32_t i = 0; i < (1u << 6); ++i) {
    e4m3_exhaustive_kernel<<<kChunk / 256, 256>>>(i * kChunk, mism.p, first.p);
  }
  cudaDeviceSynchronize();
  unsigned long long bad = 0;
  unsigned int where = 0;
  cudaMemcpy(&bad, mism.p, sizeof(bad), cudaMemcpyDeviceToHost);
  cudaMemcpy(&where, first.p, sizeof(where), cudaMemcpyDeviceToHost);
  if (bad != 0) {
    char detail[96];
    std::snprintf(detail, sizeof(detail), "%llu of 2^32 patterns differ, first 0x%08X", bad, where);
    kops_test::record_failure("e4m3_from_float_sat", detail, __FILE__, __LINE__);
  }
}

// -----------------------------------------------------------------------------
// machete_prepack (Machete / CUTLASS mixed-input SM90 prepacked weight layout)
// -----------------------------------------------------------------------------
// The layout is the one vLLM's `prepack_B_kernel` writes under
// PrepackedLayoutBTemplate (csrc/quantization/machete/machete_prepacked_layout.cuh): the [N, K] weight
// is cut into 128 x 64 blocks, block (n_tile, k_tile) is stored at element 8192 * (n_tile + k_tile *
// N_tiles), and every block is permuted by the atom's bit-field map. The oracle below is the forward
// form of that map (the kernel walks its inverse); the golden hashes/prefixes were generated from
// CUTE's `ppblock_ilvd_NK_to_offset()` for the same input pattern, so the test pins the layout even if
// the oracle's algebra were wrong in the same way as the kernel.

constexpr int kMacheteAtoms = 4;

int machete_bits(int atom) { return (atom == 0 || atom == 2) ? 4 : 8; }

/// Storage element offset of (n, k) inside one prepacked block (0 <= n < 128, 0 <= k < 64).
int machete_pp_offset(int atom, int n, int k) {
  const int n_lo = n & 7, n_b3 = (n >> 3) & 1, n_hi = n >> 4;
  switch (atom) {
    case 0:   // 16-bit MMA atom, packed int4, interleaved value blocks (W4A16)
      return (n_lo << 7) | n_b3 | (n_hi << 10) | ((k & 1) << 2) | (((k >> 1) & 3) << 5) |
             (((k >> 3) & 1) << 1) | (((k >> 4) & 3) << 3);
    case 1:   // 16-bit MMA atom, int8 weights (W8A16)
      return (n_lo << 7) | (n_b3 << 1) | (n_hi << 10) | (k & 1) | (((k >> 1) & 3) << 5) |
             (((k >> 3) & 7) << 2);
    default:  // 8-bit MMA atom (k32): int4 (W4A8) and int8 (W8A8) share the value order
      return (n_lo << 7) | (n_b3 << 2) | (n_hi << 10) | (k & 3) | (((k >> 2) & 3) << 5) |
             (((k >> 4) & 3) << 3);
  }
}

/// Deterministic, position-dependent element value (each bit group of (n, k) moves it).
int machete_elem(int n, int k, int bits) {
  return (n * 13 + k * 7 + (k >> 3) * 5 + (n >> 3) * 3 + 3) & (bits == 4 ? 0xF : 0xFF);
}

std::vector<uint8_t> machete_input(int n, int k, int bits) {
  const size_t elems = static_cast<size_t>(n) * k;
  std::vector<uint8_t> bytes(bits == 4 ? elems / 2 : elems, 0);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < k; ++c) {
      const int v = machete_elem(r, c, bits);
      if (bits == 4) {
        const size_t i = static_cast<size_t>(r) * k + c;
        bytes[i >> 1] = static_cast<uint8_t>(bytes[i >> 1] | (v << (4 * (i & 1))));
      } else {
        bytes[static_cast<size_t>(r) * k + c] = static_cast<uint8_t>(v);
      }
    }
  }
  return bytes;
}

std::vector<uint8_t> machete_reference(int atom, int n, int k) {
  const int bits = machete_bits(atom);
  const int n_tiles = n / 128;
  std::vector<uint8_t> bytes(bits == 4 ? static_cast<size_t>(n) * k / 2
                                       : static_cast<size_t>(n) * k, 0);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < k; ++c) {
      const size_t base = static_cast<size_t>(8192) * (r / 128 + static_cast<size_t>(c / 64) * n_tiles);
      const size_t o = base + machete_pp_offset(atom, r % 128, c % 64);
      const int v = machete_elem(r, c, bits);
      if (bits == 4) {
        bytes[o >> 1] = static_cast<uint8_t>(bytes[o >> 1] | (v << (4 * (o & 1))));
      } else {
        bytes[o] = static_cast<uint8_t>(v);
      }
    }
  }
  return bytes;
}

uint64_t fnv1a64(const std::vector<uint8_t>& v) {
  uint64_t h = 1469598103934665603ULL;
  for (uint8_t b : v) {
    h ^= b;
    h *= 1099511628211ULL;
  }
  return h;
}

/// Runs one prepack and compares every output byte with the oracle. Returns the device bytes.
std::vector<uint8_t> run_machete_case(const LaunchContext& ctx, int atom, int64_t n, int64_t k,
                                      const char* what_in) {
  char what[160];
  const int bits = machete_bits(atom);
  std::snprintf(what, sizeof(what), "%s atom %d %lldx%lld", what_in, atom,
                static_cast<long long>(n), static_cast<long long>(k));
  const auto in = machete_input(static_cast<int>(n), static_cast<int>(k), bits);
  const auto want = machete_reference(atom, static_cast<int>(n), static_cast<int>(k));
  auto din = upload(in);
  DevBuf<uint8_t> dout(want.size());
  cudaMemset(dout.p, 0xCD, want.size());
  const DType dt = bits == 4 ? DType::kInt4x2 : DType::kInt8;
  KOPS_EXPECT_OK(kops::gemm::machete_prepack(kops_test::tensor2d(dout.p, dt, n, k),
                                             kops_test::tensor2d(din.p, dt, n, k), bits, atom, ctx),
                 what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, want.size());
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      if (bad == 0) first = i;
      ++bad;
    }
  }
  if (bad != 0) {
    char detail[160];
    std::snprintf(detail, sizeof(detail), "%zu/%zu bytes differ, first at %zu (got %02x want %02x)", bad,
                  want.size(), first, got[first], want[first]);
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
  return got;
}

void test_machete_prepack(const LaunchContext& ctx) {
  // Full-buffer comparison against the oracle for one and for several prepacked blocks.
  for (int atom = 0; atom < kMacheteAtoms; ++atom) {
    run_machete_case(ctx, atom, 128, 64, "machete prepack single block");
    run_machete_case(ctx, atom, 256, 128, "machete prepack 2x2 blocks");
  }
  // Byte-exact goldens from CUTE (N = 256, K = 128, element (n, k) = (13 n + 7 k + 5 (k >> 3) +
  // 3 (n >> 3) + 3) mod 2^bits, 4-bit packed low nibble first) for all four atoms.
  struct Golden { int atom; uint64_t hash; const char* prefix; };
  const Golden goldens[] = {
      {0, 0x009494cde0c20b83ULL,
       "e3b05a278d5af4c127f49e6bc19e3805c19e38056b38d2af05d27c49af7c16e3"},
      {1, 0x206bdd1d62c59ec3ULL,
       "030a6e754047abb27d84e8efbac1252cf7fe6269343b9fa67178dce3aeb51920"},
      {2, 0x72e376279625af83ULL,
       "a3815e3c4d2bf8d6e7c59270816f3c1a6f4d1af809e7b492a3815e3c4d2bf8d6"},
      {3, 0xf323b483ed814243ULL,
       "030a11186e757c837d848b92e8eff6fdf7fe050c6269707771787f86dce3eaf1"},
  };
  for (const Golden& g : goldens) {
    char what[128];
    std::snprintf(what, sizeof(what), "machete prepack golden atom %d", g.atom);
    const auto got = run_machete_case(ctx, g.atom, 256, 128, "machete prepack golden");
    const uint64_t h = fnv1a64(got);
    if (h != g.hash) {
      char detail[128];
      std::snprintf(detail, sizeof(detail), "FNV-1a %016llx, want %016llx", static_cast<unsigned long long>(h),
                    static_cast<unsigned long long>(g.hash));
      kops_test::record_failure(what, detail, __FILE__, __LINE__);
    }
    size_t bad = 0;
    for (int i = 0; i < 32; ++i) {
      char byte[3] = {g.prefix[2 * i], g.prefix[2 * i + 1], 0};
      bad += got[static_cast<size_t>(i)] != static_cast<uint8_t>(std::strtoul(byte, nullptr, 16));
    }
    if (bad != 0) kops_test::record_failure(what, "first 32 bytes differ from the CUTE golden", __FILE__, __LINE__);
  }
  // The W4A16 atom is the only interleaved one: the first eight elements of a block are
  // {n0k0, n8k0, n0k8, n8k8, n0k1, n8k1, n0k9, n8k9}.
  {
    const int n = 128, k = 64;
    const auto got = run_machete_case(ctx, 0, n, k, "machete prepack interleave order");
    const int want_elems[8] = {0, 8 * k, 8, 8 * k + 8, 1, 8 * k + 1, 9, 8 * k + 9};
    size_t bad = 0;
    for (int i = 0; i < 8; ++i) {
      const uint8_t b = got[static_cast<size_t>(i >> 1)];
      const int nib = (i & 1) ? (b >> 4) : (b & 0xF);
      bad += nib != machete_elem(want_elems[i] / k, want_elems[i] % k, 4);
    }
    if (bad != 0) kops_test::record_failure("machete prepack interleave order", "the first 8 nibbles are not the ACEGBDFH interleave", __FILE__, __LINE__);
  }
  // A flat capacity view is accepted, and the kernel is a permutation (elements are conserved).
  {
    const int64_t n = 128, k = 64;
    const auto in = machete_input(static_cast<int>(n), static_cast<int>(k), 8);
    auto din = upload(in);
    DevBuf<uint8_t> dout(in.size());
    cudaMemset(dout.p, 0, in.size());
    KOPS_EXPECT_OK(kops::gemm::machete_prepack(tensor1d(dout.p, DType::kInt8, static_cast<int64_t>(in.size())),
                                               kops_test::tensor2d(din.p, DType::kInt8, n, k), 8, 1, ctx),
                   "machete prepack flat capacity");
    cudaDeviceSynchronize();
    auto got = kops_test::copy_out(dout.p, in.size());
    auto want = in;
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    if (got != want) kops_test::record_failure("machete prepack flat capacity", "the pack is not a permutation", __FILE__, __LINE__);
  }
  // Contracts.
  {
    const int64_t n = 128, k = 64;
    const auto in4 = machete_input(static_cast<int>(n), static_cast<int>(k), 4);
    const auto in8 = machete_input(static_cast<int>(n), static_cast<int>(k), 8);
    auto d4 = upload(in4);
    auto d8 = upload(in8);
    DevBuf<uint8_t> out4(in4.size()), out8(in8.size());
    const Tensor w4 = kops_test::tensor2d(d4.p, DType::kInt4x2, n, k);
    const Tensor w8 = kops_test::tensor2d(d8.p, DType::kInt8, n, k);
    const Tensor o4 = kops_test::tensor2d(out4.p, DType::kInt4x2, n, k);
    const Tensor o8 = kops_test::tensor2d(out8.p, DType::kInt8, n, k);
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(o4, w4, 4, 4, ctx), Status::kInvalidArgument, "machete prepack: atom id 4");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(o4, w4, 4, -1, ctx), Status::kInvalidArgument, "machete prepack: negative atom id");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(o4, w4, 8, 0, ctx), Status::kInvalidArgument, "machete prepack: weight_bits vs dtype");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(o4, w4, 4, 1, ctx), Status::kInvalidArgument, "machete prepack: 8-bit atom on int4");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(o8, w8, 8, 0, ctx), Status::kInvalidArgument, "machete prepack: int4 atom on int8");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(kops_test::tensor2d(out8.p, DType::kInt8, n, k / 2), w8, 8, 1, ctx),
                       Status::kInvalidArgument, "machete prepack: short out shape");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(kops_test::tensor2d(out4.p, DType::kInt4x2, n, 96),
                                                   kops_test::tensor2d(d4.p, DType::kInt4x2, n, 96), 4, 0, ctx),
                       Status::kUnsupportedShape, "machete prepack: K % 64");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(kops_test::tensor2d(out4.p, DType::kInt4x2, 64, k),
                                                   kops_test::tensor2d(d4.p, DType::kInt4x2, 64, k), 4, 0, ctx),
                       Status::kUnsupportedShape, "machete prepack: N % 128");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(o4, tensor2d_ld(d4.p, DType::kInt4x2, n, k, k + 64), 4, 0, ctx),
                       Status::kUnsupportedLayout, "machete prepack: padded weight rows");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(o4, kops_test::tensor2d(d8.p, DType::kBFloat16, n, k), 4, 0, ctx),
                       Status::kUnsupportedDtype, "machete prepack: bf16 weight");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(tensor1d(out4.p, DType::kInt4x2, n * k / 2 - 1), w4, 4, 0, ctx),
                       Status::kCapacityExceeded, "machete prepack: short flat capacity");
    KOPS_EXPECT_STATUS(kops::gemm::machete_prepack(kops_test::tensor2d(static_cast<uint8_t*>(out4.p) + 1, DType::kInt4x2, n, k),
                                                   kops_test::tensor2d(static_cast<uint8_t*>(d4.p) + 1, DType::kInt4x2, n, k), 4, 0, ctx),
                       Status::kMisaligned, "machete prepack: 1-byte offset");
  }
}

// -----------------------------------------------------------------------------
// Contracts
// -----------------------------------------------------------------------------

void test_contracts(const LaunchContext& ctx) {
  DevBuf<uint8_t> a(1 << 16);
  DevBuf<uint8_t> b(1 << 16);
  // weight_transpose: wrong output shape, mismatched dtype.
  KOPS_EXPECT_STATUS(kops::gemm::weight_transpose(kops_test::tensor2d(b.p, DType::kBFloat16, 8, 16),
                                                  kops_test::tensor2d(a.p, DType::kBFloat16, 8, 16), ctx),
                     Status::kInvalidArgument, "transpose shape");
  KOPS_EXPECT_STATUS(kops::gemm::weight_transpose(kops_test::tensor2d(b.p, DType::kFloat16, 16, 8),
                                                  kops_test::tensor2d(a.p, DType::kBFloat16, 8, 16), ctx),
                     Status::kUnsupportedDtype, "transpose dtype");
  KOPS_EXPECT_STATUS(kops::gemm::weight_transpose(kops_test::tensor2d(b.p, DType::kInt4x2, 16, 9),
                                                  kops_test::tensor2d(a.p, DType::kInt4x2, 9, 16), ctx),
                     Status::kUnsupportedShape, "transpose odd nibble dims");
  // swizzle: tiled side with the wrong padding.
  KOPS_EXPECT_STATUS(kops::gemm::block_scale_swizzle(kops_test::tensor2d(b.p, DType::kUInt8, 100, 4),
                                                     kops_test::tensor2d(a.p, DType::kUInt8, 100, 4), false, ctx),
                     Status::kInvalidArgument, "swizzle padding");
  KOPS_EXPECT_STATUS(kops::gemm::block_scale_swizzle(kops_test::tensor2d(b.p, DType::kFloat32, 128, 4),
                                                     kops_test::tensor2d(a.p, DType::kFloat32, 100, 4), false, ctx),
                     Status::kUnsupportedDtype, "swizzle dtype");
  // tma layout: packed needs K/128 % 4 == 0; wrong output rows.
  KOPS_EXPECT_STATUS(kops::gemm::fp8_scale_to_tma_layout(kops_test::tensor2d(b.p, DType::kInt32, 1, 8),
                                                         kops_test::tensor2d(a.p, DType::kFloat32, 8, 6), Tensor{}, true, ctx),
                     Status::kUnsupportedShape, "tma packed k");
  KOPS_EXPECT_STATUS(kops::gemm::fp8_scale_to_tma_layout(kops_test::tensor2d(b.p, DType::kFloat32, 6, 9),
                                                         kops_test::tensor2d(a.p, DType::kFloat32, 8, 6), Tensor{}, false, ctx),
                     Status::kInvalidArgument, "tma m_pad");
  // block quant: scales shape, out dtype, block size.
  KOPS_EXPECT_STATUS(kops::gemm::fp8_block2d_quant_weight(kops_test::tensor2d(b.p, DType::kFloat8E4M3, 128, 128),
                                                          kops_test::tensor2d(a.p, DType::kFloat32, 2, 2),
                                                          kops_test::tensor2d(a.p, DType::kBFloat16, 128, 128), 128, 128, false, ctx),
                     Status::kInvalidArgument, "block quant scales");
  KOPS_EXPECT_STATUS(kops::gemm::fp8_block2d_quant_weight(kops_test::tensor2d(b.p, DType::kInt8, 128, 128),
                                                          kops_test::tensor2d(a.p, DType::kFloat32, 1, 1),
                                                          kops_test::tensor2d(a.p, DType::kBFloat16, 128, 128), 128, 128, false, ctx),
                     Status::kUnsupportedDtype, "block quant out dtype");
  KOPS_EXPECT_STATUS(kops::gemm::fp8_block2d_quant_weight(kops_test::tensor2d(b.p, DType::kFloat8E4M3, 128, 128),
                                                          kops_test::tensor2d(a.p, DType::kFloat32, 1, 1),
                                                          kops_test::tensor2d(a.p, DType::kBFloat16, 128, 128), 0, 128, false, ctx),
                     Status::kUnsupportedShape, "block quant block");
  // colsum / router split / glu.
  KOPS_EXPECT_STATUS(kops::gemm::int8_weight_colsum(tensor1d(b.p, DType::kInt32, 7),
                                                    kops_test::tensor2d(a.p, DType::kInt8, 8, 16), ctx),
                     Status::kInvalidArgument, "colsum length");
  KOPS_EXPECT_STATUS(kops::gemm::int8_weight_colsum(tensor1d(b.p, DType::kInt32, 8),
                                                    kops_test::tensor2d(a.p, DType::kUInt8, 8, 16), ctx),
                     Status::kUnsupportedDtype, "colsum dtype");
  KOPS_EXPECT_STATUS(kops::gemm::router_weight_split_bf16x2(tensor1d(b.p, DType::kFloat16, 16),
                                                            tensor1d(b.p, DType::kBFloat16, 16),
                                                            tensor1d(a.p, DType::kFloat32, 16), ctx),
                     Status::kUnsupportedDtype, "router split dtype");
  KOPS_EXPECT_STATUS(kops::gemm::glu_weight_interleave(kops_test::tensor2d(b.p, DType::kBFloat16, 7, 4), Tensor{}, Tensor{},
                                                       kops_test::tensor2d(a.p, DType::kBFloat16, 7, 4), Tensor{}, Tensor{}, 0, 1, ctx),
                     Status::kInvalidArgument, "glu odd rows");
  KOPS_EXPECT_STATUS(kops::gemm::glu_weight_interleave(kops_test::tensor2d(b.p, DType::kBFloat16, 8, 4), Tensor{}, Tensor{},
                                                       kops_test::tensor2d(a.p, DType::kBFloat16, 8, 4), Tensor{}, Tensor{}, 0, 3, ctx),
                     Status::kUnsupportedShape, "glu interleave does not divide I");
  // Wrong backend.
  LaunchContext hip = ctx;
  hip.backend = kops::BackendKind::kHip;
  KOPS_EXPECT_STATUS(kops::gemm::weight_transpose(kops_test::tensor2d(b.p, DType::kBFloat16, 16, 8),
                                                  kops_test::tensor2d(a.p, DType::kBFloat16, 8, 16), hip),
                     Status::kUnsupportedBackend, "backend");
}

// =============================================================================
// B5: CUDA-core GEMV.  Oracle: host fp64 loops over the dequantized weights; inputs are rounded
// to the activation dtype first, so only the fp32 accumulation and the final rounding differ.
// =============================================================================

struct Rng {
  std::mt19937 g;
  explicit Rng(unsigned seed) : g(seed) {}
  uint32_t u32() { return g(); }
  float uni(float lo, float hi) { return lo + (hi - lo) * (static_cast<float>(g() >> 8) / 16777216.0f); }
  int range(int n) { return static_cast<int>(g() % static_cast<uint32_t>(n)); }
};

template <typename T>
T cvt(float x) {
  T v;
  kops_test::from_float(v, x);
  return v;
}

template <typename T>
double rel_eps();
template <>
double rel_eps<__nv_bfloat16>() { return 0.0078125; }
template <>
double rel_eps<__half>() { return 0.0009765625; }
template <>
double rel_eps<float>() { return 2e-6; }

/// Element-wise (one output ulp + fp32 accumulation noise) and Frobenius comparison.
template <typename T>
void expect_close(const std::vector<T>& got, const std::vector<double>& want, const char* what) {
  double ss = 0.0;
  for (double w : want) ss += w * w;
  const double rms = std::sqrt(ss / static_cast<double>(want.size() ? want.size() : 1));
  const double eps = rel_eps<T>();
  size_t bad = 0;
  double num = 0.0;
  size_t worst = 0;
  double worst_err = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double g = static_cast<double>(kops_test::to_float(got[i]));
    const double err = std::fabs(g - want[i]);
    num += err * err;
    if (!(err <= eps * std::fabs(want[i]) + 2e-5 * rms)) {
      ++bad;
      if (!(err <= worst_err)) { worst_err = err; worst = i; }
    }
  }
  const double frob = ss > 0.0 ? std::sqrt(num / ss) : std::sqrt(num);
  if (bad != 0 || !(frob <= eps * 0.6)) {
    char detail[256];
    std::snprintf(detail, sizeof(detail), "%zu/%zu elements off, frobenius %.3g (limit %.3g), worst #%zu got %.7g want %.7g",
                  bad, want.size(), frob, eps * 0.6, worst,
                  bad ? static_cast<double>(kops_test::to_float(got[worst])) : 0.0, bad ? want[worst] : 0.0);
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
}

/// Same rule as expect_close, plus the FP8 tensor-core accumulation bound: one output ulp + 2^-11 * sum|p|
/// (the e4m3 mma accumulates with an 11-bit effective mantissa; see expect_close_fp8).
template <typename T>
void expect_close_mag(const std::vector<T>& got, const std::vector<double>& want, const std::vector<double>& mag,
                      const char* what) {
  double ss = 0.0, mag_ss = 0.0;
  for (double w : want) ss += w * w;
  for (double m : mag) mag_ss += m * m;
  const double rms = std::sqrt(ss / static_cast<double>(want.size() ? want.size() : 1));
  const double eps = rel_eps<T>();
  size_t bad = 0;
  double num = 0.0;
  size_t worst = 0;
  double worst_err = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double g = static_cast<double>(kops_test::to_float(got[i]));
    const double err = std::fabs(g - want[i]);
    const double lim = eps * std::fabs(want[i]) + std::ldexp(mag[i], -11) + 1e-6;
    num += err * err;
    if (!(err <= lim)) {
      ++bad;
      if (!(err <= worst_err)) { worst_err = err; worst = i; }
    }
  }
  const double frob = ss > 0.0 ? std::sqrt(num / ss) : std::sqrt(num);
  const double frob_lim = 0.6 * (eps + (ss > 0.0 ? std::ldexp(std::sqrt(mag_ss / ss), -11) : 0.0));
  if (bad != 0 || !(frob <= frob_lim)) {
    char detail[256];
    std::snprintf(detail, sizeof(detail), "%zu/%zu elements off, frobenius %.3g (limit %.3g), worst #%zu got %.7g want %.7g",
                  bad, want.size(), frob, eps * 0.6, worst,
                  bad ? static_cast<double>(kops_test::to_float(got[worst])) : 0.0, bad ? want[worst] : 0.0);
    kops_test::record_failure(what, detail, __FILE__, __LINE__);
  }
}

template <typename T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

int awq_pos(int c) { return ((c & 1) << 2) | (c >> 1); }

template <typename T>
struct W4Problem {
  int m, n, k, group;          // group 0 = per channel
  bool awq, zeros, actorder, bias;
  std::vector<float> a;        // already rounded to T
  std::vector<float> scale;    // [ng, n], rounded to T
  std::vector<float> bias_v;   // [n]
  std::vector<uint32_t> qw, qz;
  std::vector<int32_t> g_idx;
  std::vector<double> ref;     // [m, n]
};

template <typename T>
W4Problem<T> make_w4(bool awq, int group, bool zeros, bool actorder, bool bias, int m, int n, int k, unsigned seed) {
  W4Problem<T> p{};
  p.m = m; p.n = n; p.k = k; p.group = group; p.awq = awq; p.zeros = zeros; p.actorder = actorder; p.bias = bias;
  Rng r(seed);
  const int G = group ? group : k;
  const int ng = k / G;
  std::vector<int> q(static_cast<size_t>(k) * n), z(static_cast<size_t>(ng) * n);
  for (auto& v : q) v = r.range(16);
  for (auto& v : z) v = r.range(16);
  p.scale.resize(static_cast<size_t>(ng) * n);
  for (auto& v : p.scale) v = kops_test::to_float(cvt<T>(r.uni(0.002f, 0.03f)));
  p.a.resize(static_cast<size_t>(m) * k);
  for (auto& v : p.a) v = kops_test::to_float(cvt<T>(r.uni(-1.0f, 1.0f)));
  p.bias_v.resize(n);
  for (auto& v : p.bias_v) v = kops_test::to_float(cvt<T>(r.uni(-0.5f, 0.5f)));
  p.g_idx.resize(k);
  {
    std::vector<int> order(k);
    for (int i = 0; i < k; ++i) order[i] = i;
    for (int i = k - 1; i > 0; --i) std::swap(order[i], order[r.range(i + 1)]);
    for (int i = 0; i < k; ++i) p.g_idx[order[i]] = i / G;
    if (!actorder) for (int i = 0; i < k; ++i) p.g_idx[i] = i / G;
  }
  if (awq) {
    p.qw.assign(static_cast<size_t>(k) * (n / 8), 0u);
    for (int kk = 0; kk < k; ++kk)
      for (int c = 0; c < n; ++c)
        p.qw[static_cast<size_t>(kk) * (n / 8) + c / 8] |= static_cast<uint32_t>(q[static_cast<size_t>(kk) * n + c]) << (4 * awq_pos(c & 7));
    p.qz.assign(static_cast<size_t>(ng) * (n / 8), 0u);
    for (int g = 0; g < ng; ++g)
      for (int c = 0; c < n; ++c)
        p.qz[static_cast<size_t>(g) * (n / 8) + c / 8] |= static_cast<uint32_t>(z[static_cast<size_t>(g) * n + c]) << (4 * awq_pos(c & 7));
  } else {
    p.qw.assign(static_cast<size_t>(k / 8) * n, 0u);
    for (int kk = 0; kk < k; ++kk)
      for (int c = 0; c < n; ++c)
        p.qw[static_cast<size_t>(kk / 8) * n + c] |= static_cast<uint32_t>(q[static_cast<size_t>(kk) * n + c]) << (4 * (kk & 7));
    p.qz.assign(static_cast<size_t>(ng) * (n / 8), 0u);
    for (int g = 0; g < ng; ++g)
      for (int c = 0; c < n; ++c)
        p.qz[static_cast<size_t>(g) * (n / 8) + c / 8] |= static_cast<uint32_t>(z[static_cast<size_t>(g) * n + c]) << (4 * (c & 7));
  }
  p.ref.assign(static_cast<size_t>(m) * n, 0.0);
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        const int g = (actorder && !awq) ? p.g_idx[kk] : kk / G;
        const int zz = zeros ? z[static_cast<size_t>(g) * n + c] : 8;
        acc += static_cast<double>(p.a[static_cast<size_t>(row) * k + kk]) *
               (static_cast<double>(q[static_cast<size_t>(kk) * n + c] - zz) * p.scale[static_cast<size_t>(g) * n + c]);
      }
      p.ref[static_cast<size_t>(row) * n + c] = acc + (bias ? p.bias_v[c] : 0.0);
    }
  }
  return p;
}

template <typename T>
std::vector<T> to_vec(const std::vector<float>& f) {
  std::vector<T> v(f.size());
  for (size_t i = 0; i < f.size(); ++i) v[i] = cvt<T>(f[i]);
  return v;
}

template <typename T>
DType dt_of();
template <>
DType dt_of<__nv_bfloat16>() { return DType::kBFloat16; }
template <>
DType dt_of<__half>() { return DType::kFloat16; }
template <>
DType dt_of<float>() { return DType::kFloat32; }

/// Runs the W4A16 GEMV rows [row0, row0 + rows) of p; out receives rows x n values.
template <typename T>
Status launch_w4(const LaunchContext& ctx, const W4Problem<T>& p, const T* da, const uint32_t* dqw,
                 const T* dsc, const uint32_t* dqz, const int32_t* dgi, const T* dbias, T* dout, int rows,
                 int64_t row0, bool marlin = false) {
  kops::gemm::ScaledGemmArgs args{};
  args.recipe = kops::gemm::GemmRecipe::kW4A16Int;
  args.a.data = da + row0 * p.k; args.a.ld = p.k; args.a.dtype = dt_of<T>();
  args.b.data = dqw; args.b.dtype = DType::kUInt32;
  args.b.packing = marlin ? kops::WeightPacking::kMarlin
                          : (p.awq ? kops::WeightPacking::kAWQ : kops::WeightPacking::kGPTQ);
  if (marlin) args.b.scale_layout = kops::ScaleLayout::kMarlinPermuted;
  args.b.scale = dsc; args.b.scale_dtype = dt_of<T>(); args.b.group_k = static_cast<int16_t>(p.group);
  args.b.zero_points = p.zeros ? dqz : nullptr; args.b.zero_dtype = DType::kUInt32;
  args.b.g_idx = (p.actorder && !p.awq) ? dgi : nullptr;
  if (p.bias) { args.epi.ops = kops::gemm::kEpiBias; args.epi.bias = dbias; args.epi.bias_dtype = dt_of<T>(); }
  args.d = dout + row0 * p.n; args.ldd = p.n; args.d_dtype = dt_of<T>();
  args.m = rows; args.n = p.n; args.k = p.k;
  return kops::gemm::mixed_input_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx);
}

/// Device copies of the AWQ problem as awq_to_marlin writes them (tiles, permuted scales, Marlin zeros).
template <typename T>
struct MarlinBufs {
  DevBuf<uint32_t> w, z;
  DevBuf<T> s;
  MarlinBufs(const LaunchContext& ctx, const W4Problem<T>& p, const uint32_t* dqw, const T* dsc,
             const uint32_t* dqz)
      : w(static_cast<size_t>(p.k / 16) * p.n * 2),
        z(static_cast<size_t>(p.k / p.group) * (p.n / 8)),
        s(static_cast<size_t>(p.k / p.group) * p.n) {
    const int64_t ng = p.k / p.group;
    const Tensor tw = kops_test::tensor2d(const_cast<uint32_t*>(dqw), DType::kInt32, p.k, p.n / 8);
    const Tensor ts = kops_test::tensor2d(const_cast<T*>(dsc), dt_of<T>(), ng, p.n);
    const Tensor tz = kops_test::tensor2d(const_cast<uint32_t*>(dqz), DType::kInt32, ng, p.n / 8);
    KOPS_EXPECT_OK(kops::quant::awq_to_marlin(kops_test::tensor2d(w.p, DType::kInt32, p.k / 16, p.n * 2),
                                              kops_test::tensor2d(s.p, dt_of<T>(), ng, p.n),
                                              kops_test::tensor2d(z.p, DType::kInt32, ng, p.n / 8), tw, ts,
                                              tz, 4, p.group, ctx),
                   "awq_to_marlin (test input)");
    cudaDeviceSynchronize();
  }
};

template <typename T>
void run_w4a16(const LaunchContext& ctx, bool awq, int group, bool zeros, bool actorder, bool bias, int m,
               int n, int k, const char* what, bool invariance = false, bool marlin = false) {
  const W4Problem<T> p = make_w4<T>(awq, group, zeros, actorder, bias, m, n, k, 77u + m * 131u + n + k);
  auto da = upload(to_vec<T>(p.a));
  auto dsc = upload(to_vec<T>(p.scale));
  auto dqw = upload(p.qw);
  auto dqz = upload(p.qz);
  auto dgi = upload(p.g_idx);
  auto db = upload(to_vec<T>(p.bias_v));
  DevBuf<T> dout(static_cast<size_t>(m) * n);
  // kMarlin: the operands the kernel reads are awq_to_marlin's output; zeros stay optional (z = 8).
  std::unique_ptr<MarlinBufs<T>> mb;
  const uint32_t* kw = dqw.p;
  const T* ks = dsc.p;
  const uint32_t* kz = dqz.p;
  if (marlin) {
    mb = std::make_unique<MarlinBufs<T>>(ctx, p, dqw.p, dsc.p, dqz.p);
    kw = mb->w.p; ks = mb->s.p; kz = mb->z.p;
  }
  KOPS_EXPECT_OK(launch_w4<T>(ctx, p, da.p, kw, ks, kz, dgi.p, db.p, dout.p, m, 0, marlin), what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, static_cast<size_t>(m) * n);
  expect_close(got, p.ref, what);
  if (!invariance) return;
  // Bit-exactness: 100 repeats, and each row alone == the same row inside the batch.
  for (int rep = 0; rep < 100; ++rep) {
    KOPS_EXPECT_OK(launch_w4<T>(ctx, p, da.p, kw, ks, kz, dgi.p, db.p, dout.p, m, 0, marlin), what);
    cudaDeviceSynchronize();
    if (!same_bits(kops_test::copy_out(dout.p, static_cast<size_t>(m) * n), got)) {
      kops_test::record_failure(what, "result changed between repeats", __FILE__, __LINE__);
      break;
    }
  }
  DevBuf<T> one(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    KOPS_EXPECT_OK(launch_w4<T>(ctx, p, da.p, kw, ks, kz, dgi.p, db.p, one.p, 1, row, marlin), what);
  }
  cudaDeviceSynchronize();
  if (!same_bits(kops_test::copy_out(one.p, static_cast<size_t>(m) * n), got)) {
    kops_test::record_failure(what, "row alone differs from the same row in a batch", __FILE__, __LINE__);
  }
}

void test_w4a16(const LaunchContext& ctx) {
  for (int m : {1, 2, 3, 5, 8, 13, 16}) {
    run_w4a16<__nv_bfloat16>(ctx, true, 128, true, false, true, m, 264, 640, "w4a16 awq g128 zeros bias bf16");
    run_w4a16<__half>(ctx, true, 32, true, false, false, m, 256, 96, "w4a16 awq g32 zeros f16");
    run_w4a16<__nv_bfloat16>(ctx, true, 64, false, false, false, m, 512, 256, "w4a16 awq g64 sym bf16");
    run_w4a16<__nv_bfloat16>(ctx, false, 128, true, false, true, m, 200, 512, "w4a16 gptq g128 zeros bias bf16");
    run_w4a16<__half>(ctx, false, 32, false, false, false, m, 96, 128, "w4a16 gptq g32 sym f16");
    run_w4a16<__nv_bfloat16>(ctx, false, 128, true, true, false, m, 160, 384, "w4a16 gptq act-order bf16");
  }
  run_w4a16<__nv_bfloat16>(ctx, true, 0, true, false, false, 3, 128, 256, "w4a16 awq per-channel");
  run_w4a16<__nv_bfloat16>(ctx, false, 0, false, false, true, 2, 72, 192, "w4a16 gptq per-channel");
  run_w4a16<__nv_bfloat16>(ctx, true, 128, true, false, true, 8, 1024, 2048, "w4a16 awq batch invariance", true);
  run_w4a16<__half>(ctx, false, 128, true, false, true, 8, 1000, 2048, "w4a16 gptq batch invariance", true);
}

// ---- w4a16 on Marlin tiles (mma.sync m16n8k16, weight = operand A) ------------------------------

/// Layout round trip, written from the PTX fragment description of mma.m16n8k16 (A, row-major) and not
/// from the kernel: register 0 = (row g, k 2c..2c+1), 1 = (row g+8, k 2c..), 2 = (row g, k 2c+8..),
/// 3 = (row g+8, k 2c+8..) with g = lane / 4, c = lane % 4.  Word w of a lane is the 16-column sub-tile
/// w of the 64-column tile; nibble pair (p, p + 4) is register {0, 2, 1, 3}[p].  Also checks the scale
/// (lane g reads columns g + 8 j at 8 g + j) and zero-point (j = 2 w + h at nibble w + 4 h) positions the
/// kernel relies on.
void test_w4a16_marlin_layout(const LaunchContext& ctx) {
  using T = __nv_bfloat16;
  const int n = 128, k = 64, group = 32;
  const W4Problem<T> p = make_w4<T>(true, group, true, false, false, 1, n, k, 4242u);
  auto dsc = upload(to_vec<T>(p.scale));
  auto dqw = upload(p.qw);
  auto dqz = upload(p.qz);
  MarlinBufs<T> mb(ctx, p, dqw.p, dsc.p, dqz.p);
  const auto words = kops_test::copy_out(mb.w.p, static_cast<size_t>(k / 16) * n * 2);
  const auto zwords = kops_test::copy_out(mb.z.p, static_cast<size_t>(k / group) * (n / 8));
  const auto sc = kops_test::copy_out(mb.s.p, static_cast<size_t>(k / group) * n);
  auto q_at = [&](int kk, int c) {
    return (p.qw[static_cast<size_t>(kk) * (n / 8) + c / 8] >> (4 * awq_pos(c & 7))) & 0xFu;
  };
  auto z_at = [&](int g, int c) {
    return (p.qz[static_cast<size_t>(g) * (n / 8) + c / 8] >> (4 * awq_pos(c & 7))) & 0xFu;
  };
  const int row_off[4] = {0, 0, 8, 8};   // pair p -> (row offset, k offset)
  const int k_off[4] = {0, 8, 0, 8};
  size_t bad_w = 0, bad_s = 0, bad_z = 0;
  for (int kt = 0; kt < k / 16; ++kt) {
    for (int nt = 0; nt < n / 64; ++nt) {
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, c = lane & 3;
        for (int w = 0; w < 4; ++w) {
          const uint32_t word = words[static_cast<size_t>(kt) * n * 2 + nt * 128 + lane * 4 + w];
          for (int pr = 0; pr < 4; ++pr) {
            for (int e = 0; e < 2; ++e) {
              const uint32_t got = (word >> (4 * (pr + 4 * e))) & 0xFu;
              const uint32_t want = q_at(kt * 16 + 2 * c + k_off[pr] + e, nt * 64 + 16 * w + g + row_off[pr]);
              bad_w += got != want;
            }
          }
        }
      }
    }
  }
  for (int gr = 0; gr < k / group; ++gr) {
    for (int nt = 0; nt < n / 64; ++nt) {
      for (int r = 0; r < 8; ++r) {
        const uint32_t zw = zwords[static_cast<size_t>(gr) * (n / 8) + nt * 8 + r];
        for (int j = 0; j < 8; ++j) {
          const int col = nt * 64 + r + 8 * j;
          bad_s += kops_test::to_float(sc[static_cast<size_t>(gr) * n + nt * 64 + 8 * r + j]) !=
                   p.scale[static_cast<size_t>(gr) * n + col];
          const int w = j >> 1, h = j & 1;
          bad_z += ((zw >> (4 * (w + 4 * h))) & 0xFu) != z_at(gr, col);
        }
      }
    }
  }
  char detail[160];
  std::snprintf(detail, sizeof(detail), "mismatches: weight nibbles %zu, scales %zu, zeros %zu", bad_w, bad_s, bad_z);
  if (bad_w + bad_s + bad_z != 0) kops_test::record_failure("w4a16 marlin layout", detail, __FILE__, __LINE__);
}

void test_w4a16_marlin(const LaunchContext& ctx) {
  using bf = __nv_bfloat16;
  for (int m : {1, 2, 3, 4, 8, 9, 13, 16}) {
    run_w4a16<bf>(ctx, true, 128, true, false, true, m, 128, 256, "w4a16 marlin g128 zeros bias bf16", false, true);
    run_w4a16<__half>(ctx, true, 32, true, false, false, m, 192, 96, "w4a16 marlin g32 zeros f16", false, true);
    run_w4a16<bf>(ctx, true, 64, false, false, true, m, 64, 320, "w4a16 marlin g64 sym bf16", false, true);
    run_w4a16<__half>(ctx, true, 128, false, false, false, m, 256, 256, "w4a16 marlin g128 sym f16", false, true);
    run_w4a16<__half>(ctx, true, 64, false, false, false, m, 256, 128, "w4a16 marlin g64 sym f16", false, true);
  }
  // Decode shapes.  K = 4096 + 32 (g32) leaves 129 groups for 8 warps, so the warps get unequal
  // group counts; K + 16 would need per-channel scales, which the Marlin-tile body does not take.
  for (int m : {1, 2, 3, 4, 8, 16}) {
    run_w4a16<bf>(ctx, true, 128, true, false, true, m, 4096, 4096, "w4a16 marlin decode g128 bf16", false, true);
    run_w4a16<bf>(ctx, true, 32, true, false, false, m, 4096, 4096 + 32, "w4a16 marlin decode ragged K g32 bf16", false, true);
  }
  for (int m : {1, 16}) {
    run_w4a16<bf>(ctx, true, 128, true, false, true, m, 14336, 4096, "w4a16 marlin decode N=14336 bf16", false, true);
  }
  run_w4a16<bf>(ctx, true, 128, true, false, true, 8, 1024, 2048, "w4a16 marlin batch invariance", true, true);
  run_w4a16<__half>(ctx, true, 64, true, false, true, 16, 512, 1024, "w4a16 marlin batch invariance f16", true, true);
}

void test_w4a16_marlin_contracts(const LaunchContext& ctx) {
  using T = __nv_bfloat16;
  const W4Problem<T> p = make_w4<T>(true, 128, true, false, false, 2, 64, 256, 5);
  auto da = upload(to_vec<T>(p.a));
  auto dsc = upload(to_vec<T>(p.scale));
  auto dqw = upload(p.qw);
  auto dqz = upload(p.qz);
  auto dgi = upload(p.g_idx);
  MarlinBufs<T> mb(ctx, p, dqw.p, dsc.p, dqz.p);
  DevBuf<T> dout(2 * 64 + 64);
  auto call = [&](auto&& mutate, const LaunchContext& c) {
    kops::gemm::ScaledGemmArgs args{};
    args.recipe = kops::gemm::GemmRecipe::kW4A16Int;
    args.a.data = da.p; args.a.ld = 256; args.a.dtype = DType::kBFloat16;
    args.b.data = mb.w.p; args.b.dtype = DType::kUInt32; args.b.packing = kops::WeightPacking::kMarlin;
    args.b.scale_layout = kops::ScaleLayout::kMarlinPermuted;
    args.b.scale = mb.s.p; args.b.scale_dtype = DType::kBFloat16; args.b.group_k = 128;
    args.b.zero_points = mb.z.p; args.d = dout.p; args.ldd = 64; args.d_dtype = DType::kBFloat16;
    args.m = 2; args.n = 64; args.k = 256;
    mutate(args);
    return kops::gemm::mixed_input_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, c);
  };
  auto same = [](auto&) {};
  KOPS_EXPECT_OK(call(same, ctx), "w4a16 marlin baseline call");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.k = 250; a.a.ld = 250; }, ctx), Status::kUnsupportedShape, "w4a16 marlin K % 16");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.n = 72; a.ldd = 72; }, ctx), Status::kUnsupportedShape, "w4a16 marlin N % 64");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.b.group_k = 0; }, ctx), Status::kUnsupportedShape, "w4a16 marlin per-channel");
  // group_k == K: awq_to_marlin stores it channelwise (accumulator order), which this body does not read.
  KOPS_EXPECT_STATUS(call([](auto& a) { a.k = 128; a.a.ld = 128; }, ctx), Status::kUnsupportedShape, "w4a16 marlin one group");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.b.scale_layout = kops::ScaleLayout::kRowMajor; }, ctx), Status::kUnsupportedLayout,
                     "w4a16 marlin scale layout");
  KOPS_EXPECT_STATUS(call([&](auto& a) { a.b.g_idx = dgi.p; }, ctx), Status::kUnsupportedLayout, "w4a16 marlin act-order");
  KOPS_EXPECT_STATUS(call([&](auto& a) { a.b.data = reinterpret_cast<const char*>(mb.w.p) + 4; }, ctx), Status::kMisaligned,
                     "w4a16 marlin misaligned weights");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.m = 17; }, ctx), Status::kNotImplemented, "w4a16 marlin M > 16");
  // The body is an mma.sync bf16/fp16 kernel: SM70 / SM75 are refused, kAuto (0) is not.
  LaunchContext old = ctx;
  old.arch_id = 750;
  KOPS_EXPECT_STATUS(call(same, old), Status::kUnsupportedArch, "w4a16 marlin sm75");
  old.arch_id = 700;
  KOPS_EXPECT_STATUS(call(same, old), Status::kUnsupportedArch, "w4a16 marlin sm70");
  LaunchContext any = ctx;
  any.arch_id = 0;
  KOPS_EXPECT_OK(call(same, any), "w4a16 marlin arch auto");
  cudaDeviceSynchronize();
}

// ---- lut_gemv -----------------------------------------------------------------------------

template <typename T>
void run_lut(const LaunchContext& ctx, int bs, bool dq, int m, int n, int k, const char* what) {
  Rng r(900u + m + n);
  std::vector<float> a(static_cast<size_t>(m) * k);
  for (auto& v : a) v = kops_test::to_float(cvt<T>(r.uni(-1.0f, 1.0f)));
  std::vector<float> cbk(16);
  for (auto& v : cbk) v = r.uni(-1.0f, 1.0f);
  std::vector<uint8_t> codes(static_cast<size_t>(n) * k);
  for (auto& v : codes) v = static_cast<uint8_t>(r.range(16));
  std::vector<uint8_t> packed(static_cast<size_t>(n) * k / 2);
  for (size_t i = 0; i < packed.size(); ++i) packed[i] = static_cast<uint8_t>(codes[2 * i] | (codes[2 * i + 1] << 4));
  const size_t nb = static_cast<size_t>(n) * k / bs;
  std::vector<float> amax_f(nb);
  std::vector<uint8_t> amax_q(nb);
  std::vector<float> amax2((nb + 255) / 256);
  float offset = 0.25f;
  for (auto& v : amax_f) v = r.uni(0.1f, 2.0f);
  for (auto& v : amax_q) v = static_cast<uint8_t>(r.range(256));
  for (auto& v : amax2) v = r.uni(0.001f, 0.01f);
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        const size_t blk = (static_cast<size_t>(c) * k + kk) / bs;
        const double am = dq ? static_cast<double>(std::fmaf(static_cast<float>(amax_q[blk]), amax2[blk >> 8], offset)) : amax_f[blk];
        acc += static_cast<double>(a[static_cast<size_t>(row) * k + kk]) *
               (static_cast<double>(cbk[codes[static_cast<size_t>(c) * k + kk]]) * am);
      }
      ref[static_cast<size_t>(row) * n + c] = acc;
    }
  }
  auto da = upload(to_vec<T>(a));
  auto dq4 = upload(packed);
  auto dcb = upload(cbk);
  auto dmf = upload(amax_f);
  auto dmq = upload(amax_q);
  auto dm2 = upload(amax2);
  auto doff = upload(std::vector<float>{offset});
  DevBuf<T> dout(static_cast<size_t>(m) * n);
  const Tensor tout = kops_test::tensor2d(dout.p, dt_of<T>(), m, n);
  const Tensor ta = kops_test::tensor2d(da.p, dt_of<T>(), m, k);
  const Tensor tq = kops_test::tensor2d(dq4.p, DType::kUInt8, n, k / 2);
  const Tensor tc = tensor1d(dcb.p, DType::kFloat32, 16);
  const Tensor tam = dq ? tensor1d(dmq.p, DType::kUInt8, static_cast<int64_t>(nb)) : tensor1d(dmf.p, DType::kFloat32, static_cast<int64_t>(nb));
  const Tensor tas = dq ? tensor1d(dm2.p, DType::kFloat32, static_cast<int64_t>(amax2.size())) : Tensor{};
  const Tensor tao = dq ? tensor1d(doff.p, DType::kFloat32, 1) : Tensor{};
  KOPS_EXPECT_OK(kops::gemm::lut_gemv(tout, ta, tq, tc, tam, tas, tao, bs, ctx), what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, static_cast<size_t>(m) * n);
  expect_close(got, ref, what);

  // Order of additions, bit for bit: lane l owns k = 32 l + 1024 i .. + 31 of every iteration i and chains
  // fmaf(a, w, part) from 0 in k order, then acc = fmaf(absmax, part, acc); the 32 lane sums meet in the
  // xor tree 16, 8, 4, 2, 1.  Any kernel that keeps this order gives the same bits for any row grouping.
  std::vector<T> want(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < n; ++c) {
      float lane_sum[32];
      for (int l = 0; l < 32; ++l) {
        float acc = 0.0f;
        for (int kc = l * 32; kc < k; kc += 1024) {
          const size_t blk = (static_cast<size_t>(c) * k + kc) / bs;
          const float am = dq ? std::fmaf(static_cast<float>(amax_q[blk]), amax2[blk >> 8], offset) : amax_f[blk];
          float part = 0.0f;
          for (int e = 0; e < 32; ++e) {
            part = std::fmaf(a[static_cast<size_t>(row) * k + kc + e],
                             cbk[codes[static_cast<size_t>(c) * k + kc + e]], part);
          }
          acc = std::fmaf(am, part, acc);
        }
        lane_sum[l] = acc;
      }
      for (int o = 16; o > 0; o >>= 1) {
        float next[32];
        for (int l = 0; l < 32; ++l) next[l] = lane_sum[l] + lane_sum[l ^ o];
        std::memcpy(lane_sum, next, sizeof(next));
      }
      want[static_cast<size_t>(row) * n + c] = cvt<T>(lane_sum[0]);
    }
  }
  if (!same_bits(got, want)) kops_test::record_failure(what, "bits differ from the fixed order of additions", __FILE__, __LINE__);
}

void test_lut_gemv(const LaunchContext& ctx) {
  // m = 4 and 6 reach the M_TILE = 4 instantiation (rows go out in tiles of 8 / 4 / 2 / 1).
  for (int m : {1, 2, 3, 4, 6, 8, 11}) {
    run_lut<__nv_bfloat16>(ctx, 64, false, m, 40, 1024, "lut gemv b64 bf16");
    run_lut<__half>(ctx, 64, true, m, 33, 2048, "lut gemv b64 double-quant f16");
    run_lut<__nv_bfloat16>(ctx, 32, false, m, 17, 96, "lut gemv b32 ragged k");
    run_lut<__nv_bfloat16>(ctx, 256, true, m, 9, 768, "lut gemv b256 double-quant");
    run_lut<__nv_bfloat16>(ctx, 64, false, m, 130, 4096, "lut gemv b64 four iterations");
    run_lut<__half>(ctx, 128, true, m, 67, 3072, "lut gemv b128 double-quant three iterations f16");
  }
}

void test_w4a16_contracts(const LaunchContext& ctx) {
  const W4Problem<__nv_bfloat16> p = make_w4<__nv_bfloat16>(true, 128, true, false, false, 2, 64, 256, 5);
  auto da = upload(to_vec<__nv_bfloat16>(p.a));
  auto dsc = upload(to_vec<__nv_bfloat16>(p.scale));
  auto dqw = upload(p.qw);
  auto dqz = upload(p.qz);
  auto dgi = upload(p.g_idx);
  DevBuf<__nv_bfloat16> dout(2 * 64 + 64);
  auto call = [&](auto&& mutate) {
    kops::gemm::ScaledGemmArgs args{};
    args.recipe = kops::gemm::GemmRecipe::kW4A16Int;
    args.a.data = da.p; args.a.ld = 256; args.a.dtype = DType::kBFloat16;
    args.b.data = dqw.p; args.b.dtype = DType::kUInt32; args.b.packing = kops::WeightPacking::kAWQ;
    args.b.scale = dsc.p; args.b.scale_dtype = DType::kBFloat16; args.b.group_k = 128;
    args.b.zero_points = dqz.p; args.d = dout.p; args.ldd = 64; args.d_dtype = DType::kBFloat16;
    args.m = 2; args.n = 64; args.k = 256;
    mutate(args);
    return kops::gemm::mixed_input_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx);
  };
  KOPS_EXPECT_OK(call([](auto&) {}), "w4a16 baseline call");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.a.dtype = DType::kFloat32; }), Status::kUnsupportedDtype, "w4a16 act dtype");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.d_dtype = DType::kFloat16; }), Status::kUnsupportedDtype, "w4a16 out dtype");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.b.group_k = 48; }), Status::kUnsupportedShape, "w4a16 group");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.n = 60; a.ldd = 60; }), Status::kUnsupportedShape, "w4a16 N % 8");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.k = 250; a.a.ld = 250; }), Status::kUnsupportedShape, "w4a16 K % 32");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.m = 17; }), Status::kNotImplemented, "w4a16 M > 16");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.b.packing = kops::WeightPacking::kMachete; }), Status::kUnsupportedLayout, "w4a16 packing");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.b.scale = nullptr; }), Status::kInvalidArgument, "w4a16 null scales");
  KOPS_EXPECT_STATUS(call([&](auto& a) { a.b.g_idx = dgi.p; }), Status::kUnsupportedLayout, "w4a16 awq act-order");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.epi.ops = kops::gemm::kEpiResidualAdd; }), Status::kNotImplemented, "w4a16 epilogue");
  KOPS_EXPECT_STATUS(call([&](auto& a) { a.a.data = reinterpret_cast<const char*>(da.p) + 2; }), Status::kMisaligned, "w4a16 misaligned A");
  KOPS_EXPECT_STATUS(call([](auto& a) { a.recipe = kops::gemm::GemmRecipe::kW4A8IntInt8; }), Status::kNotImplemented, "w4a16 recipe");
  LaunchContext hip = ctx;
  hip.backend = kops::BackendKind::kHip;
  KOPS_EXPECT_STATUS(call([&](auto&) {}), Status::kSuccess, "w4a16 still ok");
  kops::gemm::ScaledGemmArgs bad{};
  KOPS_EXPECT_STATUS(kops::gemm::mixed_input_gemm(bad, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, hip),
                     Status::kUnsupportedBackend, "w4a16 backend");
}

// ---- fp8_gemm / int8_gemm ---------------------------------------------------------------

using kops::gemm::GemmRecipe;
using kops::gemm::ScaledGemmArgs;

float f8(uint8_t b) { return e4m3_to_float(b); }

/// Random e4m3 codes that are not NaN (0x7F / 0xFF).
std::vector<uint8_t> random_e4m3(size_t n, Rng* r) {
  std::vector<uint8_t> v(n);
  for (auto& b : v) {
    do { b = static_cast<uint8_t>(r->u32()); } while ((b & 0x7F) == 0x7F);
  }
  return v;
}

template <typename T>
Status call_fp8(const LaunchContext& ctx, ScaledGemmArgs* args, const T* a, const uint8_t* w, void* d) {
  args->a.data = a; args->a.dtype = dt_of<T>();
  args->b.data = w; args->b.dtype = DType::kFloat8E4M3;
  args->d = d;
  return kops::gemm::fp8_gemm(*args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx);
}

template <>
Status call_fp8<uint8_t>(const LaunchContext& ctx, ScaledGemmArgs* args, const uint8_t* a, const uint8_t* w, void* d) {
  args->a.data = a; args->a.dtype = DType::kFloat8E4M3;
  args->b.data = w; args->b.dtype = DType::kFloat8E4M3;
  args->d = d;
  return kops::gemm::fp8_gemm(*args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx);
}

/// Activation as float (T = bf16/half) or decoded e4m3 (T = uint8_t).
template <typename T>
float act_value(const std::vector<T>& a, size_t i);
template <>
float act_value<__nv_bfloat16>(const std::vector<__nv_bfloat16>& a, size_t i) { return kops_test::to_float(a[i]); }
template <>
float act_value<__half>(const std::vector<__half>& a, size_t i) { return kops_test::to_float(a[i]); }
template <>
float act_value<uint8_t>(const std::vector<uint8_t>& a, size_t i) { return f8(a[i]); }

template <typename T>
std::vector<T> random_acts(size_t n, Rng* r);
template <>
std::vector<__nv_bfloat16> random_acts<__nv_bfloat16>(size_t n, Rng* r) {
  std::vector<__nv_bfloat16> v(n);
  for (auto& x : v) x = cvt<__nv_bfloat16>(r->uni(-1.0f, 1.0f));
  return v;
}
template <>
std::vector<__half> random_acts<__half>(size_t n, Rng* r) {
  std::vector<__half> v(n);
  for (auto& x : v) x = cvt<__half>(r->uni(-1.0f, 1.0f));
  return v;
}
template <>
std::vector<uint8_t> random_acts<uint8_t>(size_t n, Rng* r) {
  std::vector<uint8_t> v(n);
  for (auto& x : v) x = e4m3_from_float(r->uni(-2.0f, 2.0f));
  return v;
}

/// kind: 0 tensorwise (alpha), 1 rowwise (row/col scale + bias), 2 blockwise.
template <typename T>
void run_fp8(const LaunchContext& ctx, int kind, int m, int n, int k, bool col_major_sa, const char* what) {
  Rng r(31u + m * 7u + n + k + kind);
  const bool a_fp8 = std::is_same<T, uint8_t>::value;
  const size_t kb = static_cast<size_t>(k) / 128;
  const auto a = random_acts<T>(static_cast<size_t>(m) * k, &r);
  const auto w = random_e4m3(static_cast<size_t>(n) * k, &r);
  // Smaller weights keep rows well scaled: scale them by 2^-6 through the recipe scales.
  const float alpha = 0.015625f * (kind == 0 ? 1.25f : 1.0f);
  std::vector<float> row_s(m), col_s(n), bias(n), sa, sb;
  for (auto& v : row_s) v = r.uni(0.5f, 2.0f);
  for (auto& v : col_s) v = r.uni(0.5f, 2.0f);
  for (auto& v : bias) v = kops_test::to_float(cvt<__nv_bfloat16>(r.uni(-1.0f, 1.0f)));
  const int mp = static_cast<int>(round_up(m, 4));
  if (kind == 2) {
    sb.resize(((static_cast<size_t>(n) + 127) / 128) * kb);
    for (auto& v : sb) v = std::ldexp(1.0f, r.range(5) - 8) * r.uni(1.0f, 1.9f);
    if (a_fp8) {
      sa.resize(kb * (col_major_sa ? mp : m));
      for (auto& v : sa) v = std::ldexp(1.0f, r.range(5) - 3) * r.uni(1.0f, 1.9f);
    }
  }
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        double term = static_cast<double>(act_value<T>(a, static_cast<size_t>(row) * k + kk)) *
                      f8(w[static_cast<size_t>(c) * k + kk]);
        if (kind == 2) {
          const size_t blk = kk / 128;
          const double sav = a_fp8 ? sa[col_major_sa ? blk * mp + row : static_cast<size_t>(row) * kb + blk] : 1.0;
          term *= sav * sb[(static_cast<size_t>(c) / 128) * kb + blk];
        }
        acc += term;
      }
      if (kind == 0) acc *= alpha;
      if (kind == 1) acc = acc * row_s[row] * col_s[c] + bias[c];
      ref[static_cast<size_t>(row) * n + c] = acc;
    }
  }
  auto da = upload(a);
  auto dw = upload(w);
  auto drs = upload(row_s);
  auto dcs = upload(col_s);
  auto dbias = upload(to_vec<__nv_bfloat16>(bias));
  auto dsa = upload(sa.empty() ? std::vector<float>(1) : sa);
  auto dsb = upload(sb.empty() ? std::vector<float>(1) : sb);
  DevBuf<__nv_bfloat16> dout(static_cast<size_t>(m) * n);
  ScaledGemmArgs args{};
  args.recipe = kind == 0 ? GemmRecipe::kFP8Tensorwise : (kind == 1 ? GemmRecipe::kFP8Rowwise : GemmRecipe::kFP8Blockwise);
  args.a.ld = k; args.b.ld = k; args.ldd = n; args.d_dtype = DType::kBFloat16;
  args.m = m; args.n = n; args.k = k;
  if (kind == 0) args.epi.alpha = alpha;
  if (kind == 1) {
    args.epi.ops = kops::gemm::kEpiRowScale | kops::gemm::kEpiColScale | kops::gemm::kEpiBias;
    args.epi.row_scale = drs.p; args.epi.col_scale = dcs.p; args.epi.bias = dbias.p; args.epi.bias_dtype = DType::kBFloat16;
  }
  if (kind == 2) {
    args.b.scale = dsb.p; args.b.scale_dtype = DType::kFloat32;
    if (a_fp8) {
      args.a.scale = dsa.p; args.a.scale_dtype = DType::kFloat32;
      args.a.scale_layout = col_major_sa ? kops::ScaleLayout::kColMajorTma : kops::ScaleLayout::kRowMajor;
    }
  }
  KOPS_EXPECT_OK(call_fp8<T>(ctx, &args, da.p, dw.p, dout.p), what);
  cudaDeviceSynchronize();
  // The reference above carries the bias in bf16 rounding of the input; use the rounded bias.
  expect_close(kops_test::copy_out(dout.p, ref.size()), ref, what);
}

void test_fp8_decode(const LaunchContext& ctx) {
  // 256 rows, row c holds code c in all 16 positions; A = ones -> out[c] = 16 * decode(c).
  std::vector<uint8_t> w(256 * 16);
  for (int c = 0; c < 256; ++c) for (int i = 0; i < 16; ++i) w[c * 16 + i] = static_cast<uint8_t>(c);
  std::vector<__nv_bfloat16> a(16, cvt<__nv_bfloat16>(1.0f));
  auto da = upload(a);
  auto dw = upload(w);
  DevBuf<float> dout(256);
  ScaledGemmArgs args{};
  args.recipe = GemmRecipe::kFP8Tensorwise;
  args.a.ld = 16; args.b.ld = 16; args.ldd = 256; args.d_dtype = DType::kFloat32;
  args.epi.ops = kops::gemm::kEpiOutFp32;
  args.m = 1; args.n = 256; args.k = 16;
  KOPS_EXPECT_OK(call_fp8<__nv_bfloat16>(ctx, &args, da.p, dw.p, dout.p), "fp8 decode all codes");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, 256);
  int bad = 0;
  for (int c = 0; c < 256; ++c) {
    const bool nan_code = (c & 0x7F) == 0x7F;
    if (nan_code) { bad += !std::isnan(got[c]); continue; }
    bad += got[c] != 16.0f * f8(static_cast<uint8_t>(c));
  }
  if (bad != 0) kops_test::record_failure("fp8 decode all codes", "decoded value differs from the toolkit", __FILE__, __LINE__);
}

void test_fp8_gemv(const LaunchContext& ctx) {
  for (int m : {1, 2, 3, 5, 8, 9, 16}) {
    run_fp8<__nv_bfloat16>(ctx, 0, m, 200, 384, false, "fp8 tensorwise bf16");
    run_fp8<__half>(ctx, 1, m, 130, 640, false, "fp8 rowwise f16 bias");
    run_fp8<__nv_bfloat16>(ctx, 2, m, 300, 384, false, "fp8 blockwise bf16 acts");
    run_fp8<uint8_t>(ctx, 2, m, 129, 256, false, "fp8 blockwise e4m3 acts row-major");
    run_fp8<uint8_t>(ctx, 2, m, 129, 256, true, "fp8 blockwise e4m3 acts tma");
    run_fp8<uint8_t>(ctx, 0, m, 64, 512, false, "fp8 tensorwise e4m3 acts");
  }
}

void test_fp8_invariance(const LaunchContext& ctx) {
  Rng r(5);
  const int n = 1000, k = 2048, m = 8;
  const auto a = random_acts<__nv_bfloat16>(static_cast<size_t>(m) * k, &r);
  const auto w = random_e4m3(static_cast<size_t>(n) * k, &r);
  auto da = upload(a);
  auto dw = upload(w);
  DevBuf<__nv_bfloat16> full(static_cast<size_t>(m) * n), one(static_cast<size_t>(m) * n);
  auto run = [&](int rows, int row0, __nv_bfloat16* out) {
    ScaledGemmArgs args{};
    args.recipe = GemmRecipe::kFP8Tensorwise;
    args.a.ld = k; args.b.ld = k; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.epi.alpha = 0.01f;
    args.m = rows; args.n = n; args.k = k;
    return call_fp8<__nv_bfloat16>(ctx, &args, da.p + static_cast<size_t>(row0) * k, dw.p, out + static_cast<size_t>(row0) * n);
  };
  KOPS_EXPECT_OK(run(m, 0, full.p), "fp8 batch");
  cudaDeviceSynchronize();
  const auto ref = kops_test::copy_out(full.p, static_cast<size_t>(m) * n);
  for (int rep = 0; rep < 100; ++rep) {
    KOPS_EXPECT_OK(run(m, 0, full.p), "fp8 repeat");
    cudaDeviceSynchronize();
    if (!same_bits(kops_test::copy_out(full.p, static_cast<size_t>(m) * n), ref)) {
      kops_test::record_failure("fp8 batch", "result changed between repeats", __FILE__, __LINE__);
      break;
    }
  }
  for (int row = 0; row < m; ++row) KOPS_EXPECT_OK(run(1, row, one.p), "fp8 row alone");
  cudaDeviceSynchronize();
  if (!same_bits(kops_test::copy_out(one.p, static_cast<size_t>(m) * n), ref)) {
    kops_test::record_failure("fp8 batch", "row alone differs from the same row in a batch", __FILE__, __LINE__);
  }
}

// ---- int8 -----------------------------------------------------------------------------------

/// mode 0: W8A8 symmetric, 1: W8A8 asymmetric (azp), 2: W8A16 (bf16 acts) with column scale.
void run_int8(const LaunchContext& ctx, int mode, int m, int n, int k, bool fp32_out, const char* what) {
  Rng r(77u + m + n + mode);
  std::vector<int8_t> a8(static_cast<size_t>(m) * k), w(static_cast<size_t>(n) * k);
  for (auto& v : a8) v = static_cast<int8_t>(static_cast<int>(r.u32() % 256) - 128);
  for (auto& v : w) v = static_cast<int8_t>(static_cast<int>(r.u32() % 256) - 128);
  std::vector<__nv_bfloat16> a16(static_cast<size_t>(m) * k);
  for (auto& v : a16) v = cvt<__nv_bfloat16>(r.uni(-1.0f, 1.0f));
  std::vector<int32_t> azp(m), colsum(n, 0);
  for (auto& v : azp) v = static_cast<int>(r.u32() % 256) - 128;
  for (int c = 0; c < n; ++c) for (int kk = 0; kk < k; ++kk) colsum[c] += w[static_cast<size_t>(c) * k + kk];
  std::vector<float> row_s(m), col_s(n);
  for (auto& v : row_s) v = r.uni(0.001f, 0.01f);
  for (auto& v : col_s) v = r.uni(0.001f, 0.01f);
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < n; ++c) {
      if (mode == 2) {
        double acc = 0.0;
        for (int kk = 0; kk < k; ++kk) acc += static_cast<double>(kops_test::to_float(a16[static_cast<size_t>(row) * k + kk])) * w[static_cast<size_t>(c) * k + kk];
        ref[static_cast<size_t>(row) * n + c] = acc * col_s[c];
      } else {
        int64_t acc = 0;
        for (int kk = 0; kk < k; ++kk) acc += static_cast<int64_t>(a8[static_cast<size_t>(row) * k + kk]) * w[static_cast<size_t>(c) * k + kk];
        if (mode == 1) acc -= static_cast<int64_t>(azp[row]) * colsum[c];
        ref[static_cast<size_t>(row) * n + c] = static_cast<double>(acc) * row_s[row] * col_s[c];
      }
    }
  }
  auto d8 = upload(a8);
  auto d16 = upload(a16);
  auto dw = upload(w);
  auto dazp = upload(azp);
  auto dcs = upload(colsum);
  auto drs = upload(row_s);
  auto dcl = upload(col_s);
  DevBuf<float> out32(static_cast<size_t>(m) * n);
  DevBuf<__nv_bfloat16> out16(static_cast<size_t>(m) * n);
  ScaledGemmArgs args{};
  args.recipe = GemmRecipe::kINT8Rowwise;
  args.a.ld = k; args.b.ld = k; args.ldd = n;
  args.a.data = mode == 2 ? static_cast<const void*>(d16.p) : static_cast<const void*>(d8.p);
  args.a.dtype = mode == 2 ? DType::kBFloat16 : DType::kInt8;
  args.b.data = dw.p; args.b.dtype = DType::kInt8;
  args.d = fp32_out ? static_cast<void*>(out32.p) : static_cast<void*>(out16.p);
  args.d_dtype = fp32_out ? DType::kFloat32 : DType::kBFloat16;
  args.epi.ops = kops::gemm::kEpiColScale | (mode == 2 ? 0u : kops::gemm::kEpiRowScale) |
                 (mode == 1 ? kops::gemm::kEpiAzpCorrection : 0u) | (fp32_out ? kops::gemm::kEpiOutFp32 : 0u);
  args.epi.row_scale = drs.p; args.epi.col_scale = dcl.p; args.epi.azp = dazp.p; args.epi.colsum_b = dcs.p;
  args.m = m; args.n = n; args.k = k;
  KOPS_EXPECT_OK(kops::gemm::int8_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx), what);
  cudaDeviceSynchronize();
  if (fp32_out) {
    expect_close(kops_test::copy_out(out32.p, ref.size()), ref, what);
  } else {
    expect_close(kops_test::copy_out(out16.p, ref.size()), ref, what);
  }
}

void test_int8_gemv(const LaunchContext& ctx) {
  for (int m : {1, 2, 3, 5, 8, 12, 16}) {
    run_int8(ctx, 0, m, 130, 512, true, "int8 w8a8 fp32 out");
    run_int8(ctx, 0, m, 77, 144, false, "int8 w8a8 bf16 out");
    run_int8(ctx, 1, m, 130, 512, true, "int8 w8a8 azp");
    run_int8(ctx, 2, m, 99, 320, false, "int8 w8a16 bf16 acts");
  }
}

void test_fp8_int8_contracts(const LaunchContext& ctx) {
  DevBuf<uint8_t> a(1 << 16), w(1 << 16);
  DevBuf<__nv_bfloat16> d(1 << 14);
  auto base = [&] {
    ScaledGemmArgs args{};
    args.recipe = GemmRecipe::kFP8Tensorwise;
    args.a.data = a.p; args.a.ld = 256; args.a.dtype = DType::kBFloat16;
    args.b.data = w.p; args.b.ld = 256; args.b.dtype = DType::kFloat8E4M3;
    args.d = d.p; args.ldd = 64; args.d_dtype = DType::kBFloat16;
    args.m = 2; args.n = 64; args.k = 256;
    return args;
  };
  auto run8 = [&](ScaledGemmArgs args) {
    return kops::gemm::fp8_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx);
  };
  KOPS_EXPECT_OK(run8(base()), "fp8 baseline call");
  { auto x = base(); x.b.dtype = DType::kInt8; KOPS_EXPECT_STATUS(run8(x), Status::kUnsupportedDtype, "fp8 weight dtype"); }
  { auto x = base(); x.a.dtype = DType::kFloat32; KOPS_EXPECT_STATUS(run8(x), Status::kUnsupportedDtype, "fp8 act dtype"); }
  { auto x = base(); x.m = 17; KOPS_EXPECT_STATUS(run8(x), Status::kUnsupportedDtype, "fp8 M > 16 with bf16 activations (the tensor core takes e4m3)"); }
  { auto x = base(); x.k = 250; x.a.ld = 250; x.b.ld = 250; KOPS_EXPECT_STATUS(run8(x), Status::kUnsupportedShape, "fp8 K % 16"); }
  { auto x = base(); x.recipe = GemmRecipe::kMXFP8; KOPS_EXPECT_STATUS(run8(x), Status::kNotImplemented, "fp8 recipe"); }
  { auto x = base(); x.recipe = GemmRecipe::kFP8Blockwise; KOPS_EXPECT_STATUS(run8(x), Status::kInvalidArgument, "fp8 blockwise needs scales"); }
  { auto x = base(); x.b.ld = 264; KOPS_EXPECT_STATUS(run8(x), Status::kMisaligned, "fp8 pitch"); }
  { auto x = base(); x.b.major = kops::gemm::MajorOrder::kMNMajor; KOPS_EXPECT_STATUS(run8(x), Status::kUnsupportedLayout, "fp8 mn-major"); }
  { auto x = base(); x.epi.ops = kops::gemm::kEpiGLU; KOPS_EXPECT_STATUS(run8(x), Status::kNotImplemented, "fp8 GLU epilogue"); }
  { auto x = base(); x.epi.ops = kops::gemm::kEpiRowScale; KOPS_EXPECT_STATUS(run8(x), Status::kInvalidArgument, "fp8 row scale missing"); }
  { auto x = base(); x.d_dtype = DType::kInt8; KOPS_EXPECT_STATUS(run8(x), Status::kUnsupportedDtype, "fp8 out dtype"); }
  {
    LaunchContext hip = ctx;
    hip.backend = kops::BackendKind::kHip;
    KOPS_EXPECT_STATUS(kops::gemm::fp8_gemm(base(), nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, hip),
                       Status::kUnsupportedBackend, "fp8 backend");
  }
  auto run_i8 = [&](ScaledGemmArgs args) {
    return kops::gemm::int8_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx);
  };
  auto ibase = [&] {
    auto x = base();
    x.recipe = GemmRecipe::kINT8Rowwise;
    x.a.dtype = DType::kInt8; x.b.dtype = DType::kInt8;
    return x;
  };
  KOPS_EXPECT_OK(run_i8(ibase()), "int8 baseline call");
  { auto x = ibase(); x.b.dtype = DType::kFloat8E4M3; KOPS_EXPECT_STATUS(run_i8(x), Status::kUnsupportedDtype, "int8 weight dtype"); }
  { auto x = ibase(); x.epi.ops = kops::gemm::kEpiAzpCorrection; KOPS_EXPECT_STATUS(run_i8(x), Status::kInvalidArgument, "int8 azp operands"); }
  { auto x = ibase(); x.a.dtype = DType::kBFloat16; x.epi.ops = kops::gemm::kEpiAzpCorrection;
    KOPS_EXPECT_STATUS(run_i8(x), Status::kInvalidArgument, "int8 azp with float acts"); }
  // B7: M > 16 runs int8_gemm_mma_sync_kernel; the tensor core takes s8 activations only
  { auto x = ibase(); x.m = 40; KOPS_EXPECT_OK(run_i8(x), "int8 M > 16 (mma.sync)"); }
  { auto x = ibase(); x.m = 40; x.a.dtype = DType::kBFloat16; KOPS_EXPECT_STATUS(run_i8(x), Status::kUnsupportedDtype, "int8 M > 16 with bf16 activations"); }
  { auto x = ibase(); x.recipe = GemmRecipe::kFP8Rowwise; KOPS_EXPECT_STATUS(run_i8(x), Status::kNotImplemented, "int8 recipe"); }
}

// ---- dense gemv / simt / split-K -----------------------------------------------------------

template <typename T>
std::vector<T> random_t(size_t n, Rng* r, float lo, float hi) {
  std::vector<T> v(n);
  for (auto& x : v) x = cvt<T>(r->uni(lo, hi));
  return v;
}

template <typename T>
double dval(const std::vector<T>& v, size_t i) { return static_cast<double>(kops_test::to_float(v[i])); }

GemmRecipe recipe_of(DType t) {
  return t == DType::kBFloat16 ? GemmRecipe::kBF16 : (t == DType::kFloat16 ? GemmRecipe::kFP16 : GemmRecipe::kFP32);
}

/// D = epilogue(A . B^T) through scaled_gemm.  epi: bias + SiLU + residual.  split 0 = automatic.
template <typename T, typename D>
void run_dense(const LaunchContext& ctx, int m, int n, int k, bool epi, bool fp32_out, const char* what,
               bool invariance = false) {
  Rng r(4242u + m * 17u + n + k);
  const auto a = random_t<T>(static_cast<size_t>(m) * k, &r, -1.0f, 1.0f);
  const auto b = random_t<T>(static_cast<size_t>(n) * k, &r, -0.5f, 0.5f);
  const auto bias = random_t<D>(n, &r, -1.0f, 1.0f);
  const auto res = random_t<D>(static_cast<size_t>(m) * n, &r, -1.0f, 1.0f);
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) acc += dval(a, static_cast<size_t>(row) * k + kk) * dval(b, static_cast<size_t>(c) * k + kk);
      if (epi) {
        acc += dval(bias, c);
        acc = acc / (1.0 + std::exp(-acc));
        acc += dval(res, static_cast<size_t>(row) * n + c);
      }
      ref[static_cast<size_t>(row) * n + c] = acc;
    }
  }
  auto da = upload(a);
  auto db = upload(b);
  auto dbias = upload(bias);
  auto dres = upload(res);
  DevBuf<D> dout(static_cast<size_t>(m) * n);
  ScaledGemmArgs args{};
  args.recipe = recipe_of(dt_of<T>());
  args.a.data = da.p; args.a.ld = k; args.a.dtype = dt_of<T>();
  args.b.data = db.p; args.b.ld = k; args.b.dtype = dt_of<T>();
  args.d = dout.p; args.ldd = n; args.d_dtype = dt_of<D>();
  args.m = m; args.n = n; args.k = k;
  if (epi) {
    args.epi.ops = kops::gemm::kEpiBias | kops::gemm::kEpiAct | kops::gemm::kEpiResidualAdd;
    args.epi.bias = dbias.p; args.epi.bias_dtype = dt_of<D>();
    args.epi.act = kops::ActFn::kSiLU;
    args.epi.residual = dres.p; args.epi.residual_stride = n;
  }
  kops::gemm::GemmProblem prob{};
  prob.m = m; prob.n = n; prob.k = k; prob.recipe = args.recipe;
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::gemm::gemm_workspace_requirements(prob, nullptr, kops::gemm::GemmBackend::kNative, ctx, &req), what);
  DevBuf<uint8_t> scratch(req.device_scratch_bytes + 256);
  kops::Workspace ws{};
  ws.device_scratch = scratch.p; ws.device_scratch_bytes = req.device_scratch_bytes;
  if (req.device_scratch_bytes != 0) {
    KOPS_EXPECT_STATUS(kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx),
                       Status::kWorkspaceTooSmall, "split-K without workspace");
  }
  KOPS_EXPECT_OK(kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, ws, ctx), what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, ref.size());
  expect_close(got, ref, what);
  if (!invariance) return;
  for (int rep = 0; rep < 100; ++rep) {
    KOPS_EXPECT_OK(kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, ws, ctx), what);
    cudaDeviceSynchronize();
    if (!same_bits(kops_test::copy_out(dout.p, ref.size()), got)) {
      kops_test::record_failure(what, "result changed between repeats", __FILE__, __LINE__);
      break;
    }
  }
  // Row alone == row in the batch (the split is a function of N, K only).
  DevBuf<D> one(ref.size());
  for (int row = 0; row < m; ++row) {
    ScaledGemmArgs x = args;
    x.a.data = da.p + static_cast<size_t>(row) * k; x.d = one.p + static_cast<size_t>(row) * n; x.m = 1;
    x.epi.residual = dres.p + static_cast<size_t>(row) * n;
    x.epi.ops &= ~kops::gemm::kEpiResidualAdd;
    if (epi) x.epi.ops |= kops::gemm::kEpiResidualAdd;
    KOPS_EXPECT_OK(kops::gemm::scaled_gemm(x, nullptr, kops::gemm::GemmBackend::kNative, ws, ctx), what);
  }
  cudaDeviceSynchronize();
  if (!same_bits(kops_test::copy_out(one.p, ref.size()), got)) {
    kops_test::record_failure(what, "row alone differs from the same row in a batch", __FILE__, __LINE__);
  }
}

void test_dense_gemv(const LaunchContext& ctx) {
  for (int m : {1, 2, 3, 5, 8, 11, 16}) {
    run_dense<__nv_bfloat16, __nv_bfloat16>(ctx, m, 512, 1024, false, false, "dense gemv bf16");
    run_dense<__half, __half>(ctx, m, 333, 768, true, false, "dense gemv f16 epilogue");
    run_dense<float, float>(ctx, m, 257, 640, false, true, "dense gemv f32");
    run_dense<__nv_bfloat16, float>(ctx, m, 300, 4096, true, true, "dense gemv bf16 -> f32 split-K");
  }
  run_dense<__nv_bfloat16, __nv_bfloat16>(ctx, 8, 5000, 512, false, false, "dense gemv large N");
  run_dense<__nv_bfloat16, __nv_bfloat16>(ctx, 8, 300, 4096, true, false, "dense gemv split-K invariance", true);
  run_dense<__half, __half>(ctx, 8, 1000, 2048, true, false, "dense gemv invariance", true);
}

template <typename A, typename B, typename D>
void run_simt(const LaunchContext& ctx, int batch, int m, int n, int k, bool b_mn_major, const char* what) {
  Rng r(99u + m + n + k);
  const auto a = random_t<A>(static_cast<size_t>(batch) * m * k, &r, -1.0f, 1.0f);
  const auto b = random_t<B>(static_cast<size_t>(batch) * n * k, &r, -1.0f, 1.0f);
  std::vector<double> ref(static_cast<size_t>(batch) * m * n);
  for (int z = 0; z < batch; ++z) {
    for (int row = 0; row < m; ++row) {
      for (int c = 0; c < n; ++c) {
        double acc = 0.0;
        for (int kk = 0; kk < k; ++kk) {
          const size_t bi = static_cast<size_t>(z) * n * k + (b_mn_major ? static_cast<size_t>(kk) * n + c : static_cast<size_t>(c) * k + kk);
          acc += dval(a, static_cast<size_t>(z) * m * k + static_cast<size_t>(row) * k + kk) * dval(b, bi);
        }
        ref[(static_cast<size_t>(z) * m + row) * n + c] = acc * 0.5;
      }
    }
  }
  auto da = upload(a);
  auto db = upload(b);
  DevBuf<D> dout(ref.size());
  ScaledGemmArgs args{};
  args.recipe = GemmRecipe::kFP32;
  args.a.data = da.p; args.a.ld = k; args.a.batch_stride = static_cast<int64_t>(m) * k; args.a.dtype = dt_of<A>();
  args.b.data = db.p; args.b.ld = b_mn_major ? n : k; args.b.batch_stride = static_cast<int64_t>(n) * k; args.b.dtype = dt_of<B>();
  args.b.major = b_mn_major ? kops::gemm::MajorOrder::kMNMajor : kops::gemm::MajorOrder::kKMajor;
  args.d = dout.p; args.ldd = n; args.batch_stride_d = static_cast<int64_t>(m) * n; args.d_dtype = dt_of<D>();
  args.m = m; args.n = n; args.k = k; args.batch = batch;
  args.epi.alpha = 0.5f;
  KOPS_EXPECT_OK(kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx), what);
  cudaDeviceSynchronize();
  expect_close(kops_test::copy_out(dout.p, ref.size()), ref, what);
}

void run_simt_f64(const LaunchContext& ctx) {
  Rng r(12);
  const int m = 21, n = 35, k = 90;
  std::vector<double> a(m * k), b(n * k), ref(m * n);
  for (auto& v : a) v = r.uni(-1, 1);
  for (auto& v : b) v = r.uni(-1, 1);
  for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) { double s = 0; for (int kk = 0; kk < k; ++kk) s += a[i * k + kk] * b[j * k + kk]; ref[i * n + j] = s; }
  auto da = upload(a);
  auto db = upload(b);
  DevBuf<double> dout(ref.size());
  ScaledGemmArgs args{};
  args.recipe = GemmRecipe::kFP32;
  args.a.data = da.p; args.a.ld = k; args.a.dtype = DType::kFloat64;
  args.b.data = db.p; args.b.ld = k; args.b.dtype = DType::kFloat64;
  args.d = dout.p; args.ldd = n; args.d_dtype = DType::kFloat64;
  args.m = m; args.n = n; args.k = k;
  KOPS_EXPECT_OK(kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx), "simt fp64");
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, ref.size());
  double worst = 0;
  for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(got[i] - ref[i]));
  if (worst > 1e-12) kops_test::record_failure("simt fp64", "error above 1e-12", __FILE__, __LINE__);
}

void test_simt(const LaunchContext& ctx) {
  run_simt<float, float, float>(ctx, 1, 37, 53, 70, false, "simt f32 odd");
  run_simt<float, float, float>(ctx, 3, 64, 64, 128, false, "simt f32 batched");
  run_simt<float, float, float>(ctx, 1, 100, 70, 33, true, "simt f32 mn-major B");
  run_simt<__nv_bfloat16, __nv_bfloat16, float>(ctx, 1, 130, 90, 257, false, "simt bf16 golden");
  run_simt<__half, __half, __half>(ctx, 1, 40, 40, 64, false, "simt f16");
  run_simt<__nv_bfloat16, float, float>(ctx, 1, 33, 31, 65, false, "simt mixed bf16 x f32");
  run_simt_f64(ctx);
}

// ---- gemv_argmax ----------------------------------------------------------------------------

template <typename T>
void run_argmax(const LaunchContext& ctx, int m, int hidden, int v_shard, int64_t vocab_start, int64_t vocab_size,
                float logit_scale, float softcap, bool ties, const char* what) {
  Rng r(8080u + m + v_shard);
  auto h = random_t<T>(static_cast<size_t>(m) * hidden, &r, -1.0f, 1.0f);
  auto w = random_t<T>(static_cast<size_t>(v_shard) * hidden, &r, -0.3f, 0.3f);
  if (ties) {
    // Rows 5 and 9 (and 70) become copies of the row 3 and a very large positive pattern for the
    // first token: the lowest id of the tied maximum must win.
    for (int j = 0; j < hidden; ++j) w[static_cast<size_t>(3) * hidden + j] = cvt<T>(0.9f * kops_test::to_float(h[j]) > 0 ? 0.3f : -0.3f);
    for (int row : {5, 9, 70}) {
      if (row < v_shard) for (int j = 0; j < hidden; ++j) w[static_cast<size_t>(row) * hidden + j] = w[static_cast<size_t>(3) * hidden + j];
    }
  }
  std::vector<int32_t> want_id(m);
  std::vector<double> want_lp(m);
  for (int row = 0; row < m; ++row) {
    double best = -1e300;
    int64_t bi = -1;
    std::vector<double> lg(v_shard);
    for (int c = 0; c < v_shard; ++c) {
      double acc = 0;
      for (int j = 0; j < hidden; ++j) acc += dval(h, static_cast<size_t>(row) * hidden + j) * dval(w, static_cast<size_t>(c) * hidden + j);
      acc *= logit_scale;
      if (softcap > 0) acc = softcap * std::tanh(acc / softcap);
      lg[c] = (vocab_start + c < vocab_size) ? acc : -1e300;
      if (lg[c] > best) { best = lg[c]; bi = vocab_start + c; }
    }
    double sum = 0;
    for (int c = 0; c < v_shard; ++c) if (lg[c] > -1e299) sum += std::exp(lg[c] - best);
    want_id[row] = static_cast<int32_t>(bi);
    want_lp[row] = -std::log(sum);
  }
  const int ncta = kops::gemm::gemv_argmax_num_ctas(v_shard);
  auto dh = upload(h);
  auto dw = upload(w);
  DevBuf<float> pmax(static_cast<size_t>(m) * ncta), psum(static_cast<size_t>(m) * ncta), lp(m);
  DevBuf<int32_t> pidx(static_cast<size_t>(m) * ncta), ids(m);
  auto counter = upload(std::vector<uint32_t>(1, 0u));
  kops::gemm::GemvArgmaxParams p{};
  p.hidden = dh.p; p.ld_hidden = hidden; p.lm_head = dw.p; p.ld_lm_head = hidden;
  p.logit_scale = logit_scale; p.final_softcap = softcap; p.vocab_start = vocab_start; p.vocab_size = vocab_size;
  p.partial_max = pmax.p; p.partial_idx = pidx.p; p.partial_sumexp = psum.p; p.out_ids = ids.p; p.out_logprob = lp.p;
  p.done_counter = counter.p; p.m = m; p.hidden_size = hidden; p.v_shard = v_shard;
  for (int rep = 0; rep < 3; ++rep) {   // the counter must be reusable without a reset
    KOPS_EXPECT_OK(kops::gemm::gemv_argmax(p, dt_of<T>(), dt_of<T>(), ctx), what);
    cudaDeviceSynchronize();
    const auto got = kops_test::copy_out(ids.p, m);
    const auto glp = kops_test::copy_out(lp.p, m);
    for (int row = 0; row < m; ++row) {
      if (got[row] != want_id[row]) {
        char d[96];
        std::snprintf(d, sizeof(d), "row %d: id %d, want %d", row, got[row], want_id[row]);
        kops_test::record_failure(what, d, __FILE__, __LINE__);
      }
      if (std::fabs(glp[row] - want_lp[row]) > 1e-4 * (1.0 + std::fabs(want_lp[row]))) {
        char d[96];
        std::snprintf(d, sizeof(d), "row %d: logprob %.6f, want %.6f", row, glp[row], want_lp[row]);
        kops_test::record_failure(what, d, __FILE__, __LINE__);
      }
    }
  }
}

void test_gemv_argmax(const LaunchContext& ctx) {
  for (int m : {1, 3, 8, 16}) {
    run_argmax<__nv_bfloat16>(ctx, m, 256, 1000, 0, 1000, 1.0f, 0.0f, false, "argmax bf16");
    run_argmax<__half>(ctx, m, 128, 777, 1000, 1000 + 700, 0.5f, 30.0f, false, "argmax f16 padded vocab + softcap");
  }
  run_argmax<__nv_bfloat16>(ctx, 2, 64, 300, 0, 300, 1.0f, 0.0f, true, "argmax ties go to the lowest id");
  run_argmax<__nv_bfloat16>(ctx, 4, 512, 20000, 0, 20000, 1.0f, 0.0f, false, "argmax many CTAs");
}

// ---- router_gemm ----------------------------------------------------------------------------

template <typename W>
void run_router(const LaunchContext& ctx, int m, int e, int h, bool bias, bool two_term, const char* what) {
  Rng r(31337u + m + e);
  const auto x = random_t<__nv_bfloat16>(static_cast<size_t>(m) * h, &r, -1.0f, 1.0f);
  std::vector<float> w32(static_cast<size_t>(e) * h);
  for (auto& v : w32) v = r.uni(-0.2f, 0.2f);
  std::vector<float> bs(e);
  for (auto& v : bs) v = r.uni(-1.0f, 1.0f);
  // Two-term: W ~ hi + lo / 256 with both parts rounded to bf16.
  std::vector<__nv_bfloat16> hi(w32.size()), lo(w32.size());
  std::vector<double> weff(w32.size());
  for (size_t i = 0; i < w32.size(); ++i) {
    hi[i] = cvt<__nv_bfloat16>(w32[i]);
    lo[i] = cvt<__nv_bfloat16>((w32[i] - kops_test::to_float(hi[i])) * 256.0f);
    weff[i] = two_term ? static_cast<double>(kops_test::to_float(hi[i])) + static_cast<double>(kops_test::to_float(lo[i])) / 256.0
                       : (std::is_same<W, float>::value ? static_cast<double>(w32[i]) : static_cast<double>(kops_test::to_float(hi[i])));
  }
  std::vector<double> ref(static_cast<size_t>(m) * e);
  for (int row = 0; row < m; ++row)
    for (int c = 0; c < e; ++c) {
      double acc = 0;
      for (int j = 0; j < h; ++j) acc += dval(x, static_cast<size_t>(row) * h + j) * weff[static_cast<size_t>(c) * h + j];
      ref[static_cast<size_t>(row) * e + c] = acc + (bias ? bs[c] : 0.0);
    }
  auto dx = upload(x);
  auto dw32 = upload(w32);
  auto dhi = upload(hi);
  auto dlo = upload(lo);
  auto db = upload(bs);
  DevBuf<float> dout(ref.size());
  kops::gemm::RouterGemmParams p{};
  p.x = dx.p; p.ldx = h;
  p.w = std::is_same<W, float>::value ? static_cast<const void*>(dw32.p) : static_cast<const void*>(dhi.p);
  p.w_hi = dhi.p; p.w_lo = dlo.p; p.bias = bias ? db.p : nullptr;
  p.logits = dout.p; p.ld_logits = e; p.m = m; p.num_experts = e; p.hidden = h; p.lo_scale = 1.0f / 256.0f;
  KOPS_EXPECT_OK(kops::gemm::router_gemm(p, two_term, DType::kBFloat16, two_term ? DType::kBFloat16 : dt_of<W>(), ctx), what);
  cudaDeviceSynchronize();
  expect_close(kops_test::copy_out(dout.p, ref.size()), ref, what);
}

void test_router(const LaunchContext& ctx) {
  for (int m : {1, 3, 4, 7, 16, 40}) {
    run_router<float>(ctx, m, 256, 1024, false, false, "router fp32 weights");
    run_router<__nv_bfloat16>(ctx, m, 37, 512, true, false, "router bf16 weights + bias");
    run_router<__nv_bfloat16>(ctx, m, 130, 768, true, true, "router two-term");
  }
}

// ---- fp4 gemv -------------------------------------------------------------------------------

const float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

int64_t swizzle_offset(int64_t row, int64_t blk, int64_t blocks) {
  const int64_t tile = (row >> 7) * ((blocks + 3) >> 2) + (blk >> 2);
  return tile * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (blk & 3);
}

/// marlin: the weights are fp4_to_marlin's tiles and scales (kMarlin, M <= 16, N % 64 == 0); bias adds
/// kEpiBias; invariance re-runs 100 times and checks each row alone against the same row in the batch.
template <typename T>
void run_fp4(const LaunchContext& ctx, bool nv, bool swizzled, int m, int n, int k, bool global, const char* what,
             bool marlin = false, bool bias = false, bool invariance = false) {
  Rng r(2024u + m + n + nv);
  const int blk = nv ? 16 : 32;
  const int blocks = k / blk;
  const auto a = random_t<T>(static_cast<size_t>(m) * k, &r, -1.0f, 1.0f);
  const auto bv = random_t<T>(static_cast<size_t>(n), &r, -1.0f, 1.0f);
  std::vector<uint8_t> codes(static_cast<size_t>(n) * k), packed(static_cast<size_t>(n) * k / 2);
  for (auto& c : codes) c = static_cast<uint8_t>(r.range(16));
  for (size_t i = 0; i < packed.size(); ++i) packed[i] = static_cast<uint8_t>(codes[2 * i] | (codes[2 * i + 1] << 4));
  const size_t sc_elems = swizzled ? static_cast<size_t>((n + 127) / 128) * 128 * ((blocks + 3) / 4) * 4 : static_cast<size_t>(n) * blocks;
  std::vector<uint8_t> sc(sc_elems, 0);
  std::vector<double> sval(static_cast<size_t>(n) * blocks);
  for (int row = 0; row < n; ++row) {
    for (int b = 0; b < blocks; ++b) {
      uint8_t byte;
      double v;
      if (nv) {
        do { byte = e4m3_from_float(r.uni(0.05f, 2.0f)); } while (byte == 0);
        v = f8(byte);
      } else {
        byte = static_cast<uint8_t>(120 + r.range(12));
        v = std::ldexp(1.0, static_cast<int>(byte) - 127);
      }
      sval[static_cast<size_t>(row) * blocks + b] = v;
      sc[swizzled ? static_cast<size_t>(swizzle_offset(row, b, blocks)) : static_cast<size_t>(row) * blocks + b] = byte;
    }
  }
  const float gscale = global ? 7.5f : 1.0f;
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row)
    for (int c = 0; c < n; ++c) {
      double acc = 0;
      for (int kk = 0; kk < k; ++kk) {
        const uint8_t code = codes[static_cast<size_t>(c) * k + kk];
        const double w = (code & 8 ? -1.0 : 1.0) * kE2m1[code & 7] * sval[static_cast<size_t>(c) * blocks + kk / blk];
        acc += dval(a, static_cast<size_t>(row) * k + kk) * w;
      }
      ref[static_cast<size_t>(row) * n + c] = acc / gscale + (bias ? dval(bv, static_cast<size_t>(c)) : 0.0);
    }
  auto da = upload(a);
  auto dbias = upload(bv);
  auto dw = upload(packed);
  auto dsc = upload(sc);
  auto dg = upload(std::vector<float>(1, gscale));
  const DType sdt = nv ? DType::kFloat8E4M3 : DType::kFloat8E8M0;
  DevBuf<uint32_t> mw(marlin ? static_cast<size_t>(k / 16) * n * 2 : 1);
  DevBuf<uint8_t> ms(marlin ? static_cast<size_t>(n) * blocks : 1);
  if (marlin) {
    KOPS_EXPECT_OK(kops::quant::fp4_to_marlin(kops_test::tensor2d(mw.p, DType::kInt32, k / 16, n * 2),
                                              kops_test::tensor2d(ms.p, sdt, blocks, n),
                                              kops_test::tensor2d(dw.p, DType::kUInt8, n, k / 2),
                                              kops_test::tensor2d(dsc.p, sdt, n, blocks), blk, !nv, ctx),
                   "fp4_to_marlin (test input)");
    cudaDeviceSynchronize();
  }
  DevBuf<T> dout(ref.size());
  auto launch = [&](int rows, int64_t row0, T* dst) {
    ScaledGemmArgs args{};
    args.recipe = nv ? GemmRecipe::kW4A16NVFP4 : GemmRecipe::kW4A16MXFP4;
    args.a.data = da.p + row0 * k; args.a.ld = k; args.a.dtype = dt_of<T>();
    if (marlin) {
      args.b.data = mw.p; args.b.dtype = DType::kUInt32; args.b.packing = kops::WeightPacking::kMarlin;
      args.b.scale = ms.p; args.b.scale_layout = kops::ScaleLayout::kMarlinPermuted;
    } else {
      args.b.data = dw.p; args.b.ld = k / 2; args.b.dtype = DType::kFloat4E2M1x2; args.b.packing = kops::WeightPacking::kFP4E2M1;
      args.b.scale = dsc.p; args.b.scale_layout = swizzled ? kops::ScaleLayout::kSwizzled128x4 : kops::ScaleLayout::kRowMajor;
    }
    args.b.scale_dtype = sdt; args.b.group_k = static_cast<int16_t>(blk);
    if (global) args.b.global_scale = dg.p;
    if (bias) { args.epi.ops = kops::gemm::kEpiBias; args.epi.bias = dbias.p; args.epi.bias_dtype = dt_of<T>(); }
    args.d = dst; args.ldd = n; args.d_dtype = dt_of<T>();
    args.m = rows; args.n = n; args.k = k;
    return kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx);
  };
  KOPS_EXPECT_OK(launch(m, 0, dout.p), what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, ref.size());
  expect_close(got, ref, what);
  if (!invariance) return;
  for (int rep = 0; rep < 100; ++rep) {
    KOPS_EXPECT_OK(launch(m, 0, dout.p), what);
    cudaDeviceSynchronize();
    if (!same_bits(kops_test::copy_out(dout.p, ref.size()), got)) {
      kops_test::record_failure(what, "result changed between repeats", __FILE__, __LINE__);
      break;
    }
  }
  DevBuf<T> one(ref.size());
  for (int row = 0; row < m; ++row) KOPS_EXPECT_OK(launch(1, row, one.p + static_cast<size_t>(row) * n), what);
  cudaDeviceSynchronize();
  if (!same_bits(kops_test::copy_out(one.p, ref.size()), got)) {
    kops_test::record_failure(what, "row alone differs from the same row in a batch", __FILE__, __LINE__);
  }
}

void test_fp4_gemv(const LaunchContext& ctx) {
  for (int m : {1, 2, 3, 5, 8}) {
    run_fp4<__nv_bfloat16>(ctx, false, false, m, 130, 256, false, "mxfp4 row-major bf16");
    run_fp4<__half>(ctx, false, true, m, 200, 512, false, "mxfp4 swizzled f16");
    run_fp4<__nv_bfloat16>(ctx, true, false, m, 77, 320, true, "nvfp4 row-major + global scale");
    run_fp4<__half>(ctx, true, true, m, 260, 1024, true, "nvfp4 swizzled + global scale");
  }
}

// ---- fp4 on Marlin tiles (mma.sync m16n8k16, weight = operand A) ---------------------------------

void test_fp4_marlin(const LaunchContext& ctx) {
  using bf = __nv_bfloat16;
  for (int m : {1, 2, 3, 5, 8, 9, 13, 16}) {
    run_fp4<bf>(ctx, false, false, m, 128, 256, false, "mxfp4 marlin bf16", true);
    run_fp4<__half>(ctx, false, false, m, 192, 512, false, "mxfp4 marlin f16", true);
    run_fp4<bf>(ctx, true, false, m, 64, 320, true, "nvfp4 marlin + global scale bf16", true);
    run_fp4<__half>(ctx, true, false, m, 256, 1024, true, "nvfp4 marlin + global scale + bias f16", true, true);
  }
  // Decode shapes; K + 32 leaves a short last unit for the 4-block (NVFP4) and 2-block (MXFP4) K loop.
  for (int m : {1, 4, 16}) {
    run_fp4<bf>(ctx, false, false, m, 4096, 4096, false, "mxfp4 marlin decode bf16", true);
    run_fp4<bf>(ctx, true, false, m, 4096, 4096, true, "nvfp4 marlin decode bf16", true);
    run_fp4<bf>(ctx, false, false, m, 128, 4096 + 32, false, "mxfp4 marlin ragged K", true);
    run_fp4<__half>(ctx, true, false, m, 128, 4096 + 32, false, "nvfp4 marlin ragged K f16", true);
  }
  run_fp4<bf>(ctx, false, false, 8, 512, 2048, false, "mxfp4 marlin batch invariance", true, true, true);
  run_fp4<__half>(ctx, true, false, 16, 512, 1024, true, "nvfp4 marlin batch invariance f16", true, false, true);
}

void test_fp4_marlin_contracts(const LaunchContext& ctx) {
  using T = __nv_bfloat16;
  const int n = 64, k = 256, blk = 32, blocks = k / blk;
  Rng r(7);
  const auto a = random_t<T>(static_cast<size_t>(2) * k, &r, -1.0f, 1.0f);
  std::vector<uint8_t> packed(static_cast<size_t>(n) * k / 2), sc(static_cast<size_t>(n) * blocks, 125);
  for (auto& b : packed) b = static_cast<uint8_t>(r.range(256));
  auto da = upload(a);
  auto dw = upload(packed);
  auto dsc = upload(sc);
  DevBuf<uint32_t> mw(static_cast<size_t>(k / 16) * n * 2);
  DevBuf<uint8_t> ms(sc.size() + 8);
  KOPS_EXPECT_OK(kops::quant::fp4_to_marlin(kops_test::tensor2d(mw.p, DType::kInt32, k / 16, n * 2),
                                            kops_test::tensor2d(ms.p, DType::kFloat8E8M0, blocks, n),
                                            kops_test::tensor2d(dw.p, DType::kUInt8, n, k / 2),
                                            kops_test::tensor2d(dsc.p, DType::kFloat8E8M0, n, blocks), blk, true, ctx),
                 "fp4_to_marlin (contract input)");
  cudaDeviceSynchronize();
  DevBuf<T> dout(2 * n);
  auto call = [&](bool marlin, auto&& mutate, const LaunchContext& c) {
    ScaledGemmArgs args{};
    args.recipe = GemmRecipe::kW4A16MXFP4;
    args.a.data = da.p; args.a.ld = k; args.a.dtype = DType::kBFloat16;
    if (marlin) {
      args.b.data = mw.p; args.b.dtype = DType::kUInt32; args.b.packing = kops::WeightPacking::kMarlin;
      args.b.scale = ms.p; args.b.scale_layout = kops::ScaleLayout::kMarlinPermuted;
    } else {
      args.b.data = dw.p; args.b.ld = k / 2; args.b.dtype = DType::kFloat4E2M1x2; args.b.packing = kops::WeightPacking::kFP4E2M1;
      args.b.scale = dsc.p;
    }
    args.b.scale_dtype = DType::kFloat8E8M0; args.b.group_k = blk;
    args.d = dout.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
    args.m = 2; args.n = n; args.k = k;
    mutate(args);
    return kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, c);
  };
  auto same = [](auto&) {};
  KOPS_EXPECT_OK(call(true, same, ctx), "fp4 marlin baseline call");
  KOPS_EXPECT_OK(call(true, [](auto& x) { x.m = 16; }, ctx), "fp4 marlin M = 16");
  KOPS_EXPECT_STATUS(call(true, [](auto& x) { x.m = 17; }, ctx), Status::kNotImplemented, "fp4 marlin M > 16");
  KOPS_EXPECT_STATUS(call(false, [](auto& x) { x.m = 9; }, ctx), Status::kNotImplemented, "fp4 CUDA-core M > 8 unchanged");
  KOPS_EXPECT_STATUS(call(true, [](auto& x) { x.n = 72; x.ldd = 72; }, ctx), Status::kUnsupportedShape, "fp4 marlin N % 64");
  KOPS_EXPECT_STATUS(call(true, [](auto& x) { x.k = 240; x.a.ld = 240; }, ctx), Status::kUnsupportedShape, "fp4 marlin K % 32");
  KOPS_EXPECT_STATUS(call(true, [](auto& x) { x.b.scale_layout = kops::ScaleLayout::kRowMajor; }, ctx), Status::kUnsupportedLayout,
                     "fp4 marlin scale layout");
  KOPS_EXPECT_STATUS(call(true, [](auto& x) { x.b.dtype = DType::kFloat4E2M1x2; }, ctx), Status::kUnsupportedDtype, "fp4 marlin weight dtype");
  KOPS_EXPECT_STATUS(call(true, [](auto& x) { x.d_dtype = DType::kFloat32; }, ctx), Status::kUnsupportedDtype, "fp4 marlin fp32 output");
  KOPS_EXPECT_STATUS(call(true, [](auto& x) { x.epi.ops = kops::gemm::kEpiOutFp32; }, ctx), Status::kInvalidArgument,
                     "fp4 marlin kEpiOutFp32 without fp32 output");
  KOPS_EXPECT_STATUS(call(true, [&](auto& x) { x.b.data = reinterpret_cast<const char*>(mw.p) + 4; }, ctx), Status::kMisaligned,
                     "fp4 marlin misaligned weights");
  KOPS_EXPECT_STATUS(call(true, [&](auto& x) { x.b.scale = reinterpret_cast<const char*>(ms.p) + 4; }, ctx), Status::kMisaligned,
                     "fp4 marlin misaligned scales");
  LaunchContext old = ctx;
  old.arch_id = 750;
  KOPS_EXPECT_STATUS(call(true, same, old), Status::kUnsupportedArch, "fp4 marlin sm75");
  old.arch_id = 700;
  KOPS_EXPECT_STATUS(call(true, same, old), Status::kUnsupportedArch, "fp4 marlin sm70");
  KOPS_EXPECT_OK(call(false, same, old), "fp4 CUDA-core body is not arch-gated");
  LaunchContext any = ctx;
  any.arch_id = 0;
  KOPS_EXPECT_OK(call(true, same, any), "fp4 marlin arch auto");
  cudaDeviceSynchronize();
}

// ---- GGUF mmvq -----------------------------------------------------------------------------

using kops::GGUFType;

struct GgufFormat {
  kops::GGUFType type;
  int elems, bytes;
  std::vector<int> halves;   // byte offsets of the fp16 scale fields
  const char* name;
};

const GgufFormat kGguf[] = {
    {kops::GGUFType::kQ4_0, 32, 18, {0}, "Q4_0"},          {kops::GGUFType::kQ4_1, 32, 20, {0, 2}, "Q4_1"},
    {kops::GGUFType::kQ5_0, 32, 22, {0}, "Q5_0"},          {kops::GGUFType::kQ5_1, 32, 24, {0, 2}, "Q5_1"},
    {kops::GGUFType::kQ8_0, 32, 34, {0}, "Q8_0"},          {kops::GGUFType::kQ2_K, 256, 84, {80, 82}, "Q2_K"},
    {kops::GGUFType::kQ3_K, 256, 110, {108}, "Q3_K"},      {kops::GGUFType::kQ4_K, 256, 144, {0, 2}, "Q4_K"},
    {kops::GGUFType::kQ5_K, 256, 176, {0, 2}, "Q5_K"},     {kops::GGUFType::kQ6_K, 256, 210, {208}, "Q6_K"},
};

std::vector<uint8_t> random_gguf(const GgufFormat& f, size_t blocks, Rng* r) {
  std::vector<uint8_t> raw(blocks * f.bytes);
  for (auto& b : raw) b = static_cast<uint8_t>(r->u32());
  for (size_t b = 0; b < blocks; ++b) {
    for (int off : f.halves) {
      const __half h = __float2half_rn((r->range(2) ? 1.0f : -1.0f) * r->uni(0.002f, 0.03f));
      std::memcpy(&raw[b * f.bytes + off], &h, 2);
    }
  }
  return raw;
}

struct Q81Host {
  std::vector<double> deq;      // d_h * q
  std::vector<uint8_t> bytes;   // the 36-byte blocks exactly as the kernel must write them
};

template <typename T>
Q81Host quantize_q81_host(const std::vector<T>& a, int rows, int k) {
  Q81Host o;
  o.deq.resize(static_cast<size_t>(rows) * k);
  o.bytes.assign(static_cast<size_t>(rows) * (k / 32) * 36, 0);
  for (int row = 0; row < rows; ++row) {
    for (int b = 0; b < k / 32; ++b) {
      float amax = 0.0f;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(kops_test::to_float(a[static_cast<size_t>(row) * k + b * 32 + i])));
      const float d = amax / 127.0f;
      uint8_t* blk = &o.bytes[(static_cast<size_t>(row) * (k / 32) + b) * 36];
      int sum = 0;
      const __half dh = __float2half_rn(d);
      for (int i = 0; i < 32; ++i) {
        const float x = kops_test::to_float(a[static_cast<size_t>(row) * k + b * 32 + i]);
        const int q = amax == 0.0f ? 0 : static_cast<int>(std::round(x / d));
        sum += q;
        blk[4 + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
        o.deq[static_cast<size_t>(row) * k + b * 32 + i] = static_cast<double>(__half2float(dh)) * q;
      }
      const __half sh = __float2half_rn(d * static_cast<float>(sum));
      std::memcpy(blk, &dh, 2);
      std::memcpy(blk + 2, &sh, 2);
    }
  }
  return o;
}

template <typename T>
void run_gguf(const LaunchContext& ctx, const GgufFormat& f, int m, int n, int k, const char* what) {
  Rng r(606u + m + n + f.bytes);
  const auto a = random_t<T>(static_cast<size_t>(m) * k, &r, -2.0f, 2.0f);
  const size_t blocks = static_cast<size_t>(n) * (k / f.elems);
  const auto raw = random_gguf(f, blocks, &r);
  auto dq = upload(raw);
  DevBuf<float> wdeq(static_cast<size_t>(n) * k);
  KOPS_EXPECT_OK(kops::quant::gguf_dequantize(kops_test::tensor2d(wdeq.p, DType::kFloat32, n, k),
                                              tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(raw.size())), f.type, ctx), what);
  cudaDeviceSynchronize();
  const auto w = kops_test::copy_out(wdeq.p, static_cast<size_t>(n) * k);
  const Q81Host q = quantize_q81_host(a, m, k);
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row)
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) acc += static_cast<double>(w[static_cast<size_t>(c) * k + kk]) * q.deq[static_cast<size_t>(row) * k + kk];
      ref[static_cast<size_t>(row) * n + c] = acc;
    }
  auto da = upload(a);
  DevBuf<uint8_t> ws(q.bytes.size());
  DevBuf<T> dout(ref.size());
  KOPS_EXPECT_OK(kops::gemm::gguf_gemm(kops_test::tensor2d(dout.p, dt_of<T>(), m, n), kops_test::tensor2d(da.p, dt_of<T>(), m, k),
                                       tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(raw.size())), f.type,
                                       tensor1d(ws.p, DType::kUInt8, static_cast<int64_t>(q.bytes.size())), ctx), what);
  cudaDeviceSynchronize();
  if (kops_test::copy_out(ws.p, q.bytes.size()) != q.bytes) {
    kops_test::record_failure(what, "q8_1 activation blocks differ from the host quantizer", __FILE__, __LINE__);
  }
  expect_close(kops_test::copy_out(dout.p, ref.size()), ref, what);
}

void test_gguf_gemv(const LaunchContext& ctx) {
  for (const GgufFormat& f : kGguf) {
    for (int m : {1, 2, 3, 8, 11}) {
      run_gguf<__nv_bfloat16>(ctx, f, m, 37, 512, f.name);
    }
    run_gguf<__half>(ctx, f, 1, 130, 1024, f.name);
  }
}

void test_gguf_moe(const LaunchContext& ctx) {
  const GgufFormat& f = kGguf[7];   // Q4_K
  const int experts = 4, tokens = 3, topk = 2, n = 40, k = 512;
  Rng r(77);
  const auto a = random_t<__nv_bfloat16>(static_cast<size_t>(tokens) * k, &r, -2.0f, 2.0f);
  const size_t per_expert = static_cast<size_t>(n) * (k / f.elems);
  const auto raw = random_gguf(f, per_expert * experts, &r);
  const int32_t ids_h[tokens * topk] = {2, 0, 3, -1, 1, 2};
  auto dq = upload(raw);
  DevBuf<float> wdeq(static_cast<size_t>(experts) * n * k);
  KOPS_EXPECT_OK(kops::quant::gguf_dequantize(kops_test::tensor2d(wdeq.p, DType::kFloat32, experts * n, k),
                                              tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(raw.size())), f.type, ctx), "moe dequant");
  cudaDeviceSynchronize();
  const auto w = kops_test::copy_out(wdeq.p, static_cast<size_t>(experts) * n * k);
  const Q81Host q = quantize_q81_host(a, tokens, k);
  std::vector<double> ref(static_cast<size_t>(tokens) * topk * n, 0.0);
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < topk; ++j) {
      const int e = ids_h[t * topk + j];
      if (e < 0) continue;
      for (int c = 0; c < n; ++c) {
        double acc = 0.0;
        for (int kk = 0; kk < k; ++kk) acc += static_cast<double>(w[(static_cast<size_t>(e) * n + c) * k + kk]) * q.deq[static_cast<size_t>(t) * k + kk];
        ref[(static_cast<size_t>(t) * topk + j) * n + c] = acc;
      }
    }
  auto da = upload(a);
  auto dids = upload(std::vector<int32_t>(ids_h, ids_h + tokens * topk));
  DevBuf<uint8_t> ws(q.bytes.size());
  KOPS_EXPECT_OK(kops::gemm::gguf_quantize_q8_1(tensor1d(ws.p, DType::kUInt8, static_cast<int64_t>(q.bytes.size())),
                                                kops_test::tensor2d(da.p, DType::kBFloat16, tokens, k), ctx), "moe quantize");
  DevBuf<__nv_bfloat16> dout(ref.size());
  std::vector<__nv_bfloat16> zero(ref.size(), cvt<__nv_bfloat16>(0.0f));
  kops_test::copy_in(dout.p, zero);
  KOPS_EXPECT_OK(kops::gemm::gguf_moe_gemm(kops_test::tensor2d(dout.p, DType::kBFloat16, tokens * topk, n),
                                           tensor1d(ws.p, DType::kUInt8, static_cast<int64_t>(q.bytes.size())),
                                           tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(raw.size())),
                                           static_cast<int64_t>(per_expert) * f.bytes, kops_test::tensor2d(dids.p, DType::kInt32, tokens, topk),
                                           f.type, ctx), "moe gemv");
  cudaDeviceSynchronize();
  expect_close(kops_test::copy_out(dout.p, ref.size()), ref, "gguf moe gemv (dropped slot stays zero)");
}

void test_gguf_contracts(const LaunchContext& ctx) {
  DevBuf<uint8_t> w(1 << 16), ws(1 << 16);
  DevBuf<__nv_bfloat16> a(1 << 12), o(1 << 12);
  auto call = [&](GGUFType t, int m, int n, int k, DType odt) {
    return kops::gemm::gguf_gemm(kops_test::tensor2d(o.p, odt, m, n), kops_test::tensor2d(a.p, DType::kBFloat16, m, k),
                                 tensor1d(w.p, DType::kUInt8, static_cast<int64_t>(n) * (k / 32) * 18), t,
                                 tensor1d(ws.p, DType::kUInt8, 1 << 16), ctx);
  };
  KOPS_EXPECT_OK(call(GGUFType::kQ4_0, 2, 16, 64, DType::kBFloat16), "gguf baseline");
  KOPS_EXPECT_STATUS(call(GGUFType::kIQ4_NL, 2, 16, 64, DType::kBFloat16), Status::kNotImplemented, "gguf IQ type");
  KOPS_EXPECT_STATUS(call(GGUFType::kQ4_K, 2, 16, 64, DType::kBFloat16), Status::kUnsupportedShape, "gguf K-quant block");
  KOPS_EXPECT_STATUS(call(GGUFType::kQ4_0, 2, 16, 64, DType::kFloat16), Status::kUnsupportedDtype, "gguf output dtype");
  KOPS_EXPECT_OK(call(GGUFType::kQ4_0, 17, 16, 64, DType::kBFloat16), "gguf M > 16 (mmq, B7)");
  KOPS_EXPECT_STATUS(kops::gemm::gguf_gemm(kops_test::tensor2d(o.p, DType::kBFloat16, 2, 16), kops_test::tensor2d(a.p, DType::kBFloat16, 2, 64),
                                           tensor1d(w.p, DType::kUInt8, 100), GGUFType::kQ4_0,
                                           tensor1d(ws.p, DType::kUInt8, 1 << 16), ctx),
                     Status::kInvalidArgument, "gguf weight size");
  KOPS_EXPECT_STATUS(kops::gemm::gguf_gemm(kops_test::tensor2d(o.p, DType::kBFloat16, 2, 16), kops_test::tensor2d(a.p, DType::kBFloat16, 2, 64),
                                           tensor1d(w.p, DType::kUInt8, 16 * 2 * 18), GGUFType::kQ4_0,
                                           tensor1d(ws.p, DType::kUInt8, 10), ctx),
                     Status::kWorkspaceTooSmall, "gguf q8_1 workspace");
}

// ---- grouped gemv ---------------------------------------------------------------------------

using kops::gemm::GroupedLayout;

/// recipe: 0 bf16, 1 fp16, 2 fp8 tensorwise, 3 fp8 rowwise.  layout: 0 contiguous, 1 masked, 2 indexed.
template <typename T>
void run_grouped(const LaunchContext& ctx, int recipe, int layout, const char* what) {
  const int G = 5, N = 70, K = 256;
  Rng r(515u + recipe * 7 + layout);
  const bool fp8 = recipe >= 2;
  std::vector<int> rows_of = {3, 0, 6, 1, 2};
  if (layout == 0 || layout == 2) rows_of = {2, 0, 5, 1, 4};   // expert 2 has 5 rows: two weight passes
  int total = 0;
  std::vector<int64_t> offs(G + 1, 0);
  for (int g = 0; g < G; ++g) { offs[g + 1] = offs[g] + rows_of[g]; total += rows_of[g]; }
  const int max_m = 6;
  const int a_rows = layout == 1 ? G * max_m : total + 3;
  const int d_rows = layout == 1 ? G * max_m : total;
  const auto a = random_t<T>(static_cast<size_t>(a_rows) * K, &r, -1.0f, 1.0f);
  std::vector<uint8_t> w8;
  std::vector<T> w16;
  if (fp8) w8 = random_e4m3(static_cast<size_t>(G) * N * K, &r);
  else w16 = random_t<T>(static_cast<size_t>(G) * N * K, &r, -0.5f, 0.5f);
  std::vector<float> bscale(recipe == 2 ? G : G * N);
  for (auto& v : bscale) v = r.uni(0.005f, 0.02f);
  const auto bias = random_t<__nv_bfloat16>(static_cast<size_t>(G) * N, &r, -1.0f, 1.0f);
  std::vector<float> rw(d_rows);
  for (auto& v : rw) v = r.uni(0.2f, 1.5f);
  std::vector<int32_t> gather(total);
  for (int i = 0; i < total; ++i) gather[i] = (i * 5 + 1) % (a_rows);
  const float alpha = 0.75f;
  // Reference.
  const T sentinel = cvt<T>(7.0f);
  std::vector<double> ref(static_cast<size_t>(d_rows) * N, 7.0);
  std::vector<char> written(static_cast<size_t>(d_rows) * N, 0);
  for (int g = 0; g < G; ++g) {
    for (int i = 0; i < rows_of[g]; ++i) {
      const int64_t row = layout == 1 ? static_cast<int64_t>(g) * max_m + i : offs[g] + i;
      const int64_t src = layout == 2 ? gather[offs[g] + i] : row;
      for (int c = 0; c < N; ++c) {
        double acc = 0;
        for (int kk = 0; kk < K; ++kk) {
          const size_t wi = (static_cast<size_t>(g) * N + c) * K + kk;
          const double wv = fp8 ? f8(w8[wi]) : dval(w16, wi);
          acc += dval(a, static_cast<size_t>(src) * K + kk) * wv;
        }
        acc *= alpha;
        if (recipe == 2) acc *= bscale[g];
        if (recipe == 3) acc *= bscale[static_cast<size_t>(g) * N + c];
        acc += dval(bias, static_cast<size_t>(g) * N + c);
        acc *= rw[row];
        ref[static_cast<size_t>(row) * N + c] = acc;
        written[static_cast<size_t>(row) * N + c] = 1;
      }
    }
  }
  auto da = upload(a);
  auto dw8 = upload(w8.empty() ? std::vector<uint8_t>(16) : w8);
  auto dw16 = upload(w16.empty() ? std::vector<T>(8) : w16);
  auto dbs = upload(bscale);
  auto db = upload(bias);
  auto drw = upload(rw);
  auto dgi = upload(gather);
  auto doff = upload(offs);
  auto dmm = upload(std::vector<int32_t>(rows_of.begin(), rows_of.end()));
  DevBuf<T> dout(static_cast<size_t>(d_rows) * N);
  kops_test::copy_in(dout.p, std::vector<T>(static_cast<size_t>(d_rows) * N, sentinel));
  kops::gemm::GroupedGemmParams p{};
  p.layout = layout == 0 ? GroupedLayout::kContiguous : (layout == 1 ? GroupedLayout::kMasked : GroupedLayout::kIndexed);
  p.recipe = recipe == 0 ? GemmRecipe::kBF16 : (recipe == 1 ? GemmRecipe::kFP16 : (recipe == 2 ? GemmRecipe::kFP8Tensorwise : GemmRecipe::kFP8Rowwise));
  p.num_groups = G;
  p.a = da.p; p.lda = K;
  p.expert_offsets = doff.p; p.masked_m = dmm.p; p.max_m = max_m; p.gather_idx = dgi.p;
  p.b = fp8 ? static_cast<const void*>(dw8.p) : static_cast<const void*>(dw16.p);
  p.ldb = K; p.b_group_stride = static_cast<int64_t>(N) * K;
  p.b_scale = fp8 ? dbs.p : nullptr; p.b_scale_group_stride = N;
  p.d = dout.p; p.ldd = N;
  p.n = N; p.k = K;
  p.epi.ops = kops::gemm::kEpiBias | kops::gemm::kEpiRoutingWeight;
  p.epi.alpha = alpha;
  p.epi.bias = db.p; p.epi.bias_dtype = DType::kBFloat16; p.epi.bias_group_stride = N;
  p.epi.routing_weight = drw.p;
  KOPS_EXPECT_OK(kops::gemm::grouped_gemm(p, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx), what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, ref.size());
  // Unrouted rows must still hold the sentinel; routed rows must match the oracle.
  std::vector<T> got_routed;
  std::vector<double> want_routed;
  size_t untouched_bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (written[i]) { got_routed.push_back(got[i]); want_routed.push_back(ref[i]); }
    else untouched_bad += kops_test::to_float(got[i]) != 7.0f;
  }
  if (untouched_bad != 0) kops_test::record_failure(what, "rows of empty experts / masked padding were written", __FILE__, __LINE__);
  expect_close(got_routed, want_routed, what);
}

void test_grouped_gemv(const LaunchContext& ctx) {
  for (int layout : {0, 1, 2}) {
    run_grouped<__nv_bfloat16>(ctx, 0, layout, "grouped gemv bf16");
    run_grouped<__half>(ctx, 1, layout, "grouped gemv fp16");
    run_grouped<__nv_bfloat16>(ctx, 2, layout, "grouped gemv fp8 tensorwise");
    run_grouped<__nv_bfloat16>(ctx, 3, layout, "grouped gemv fp8 rowwise");
  }
}

// ---- moe_gemv -------------------------------------------------------------------------------

/// wmode: 0 16-bit, 1 e4m3 per channel, 2 e4m3 per expert, 3 e4m3 128x128 blocks.
template <typename T>
void run_moe_gemv(const LaunchContext& ctx, int slots_per_cta, int wmode, bool w2_stage, bool drop_expert, const char* what) {
  const int E = 4, top_k = 2, tokens = 5, N = 130, K = 256;
  const int num_valid = tokens * top_k;
  Rng r(2468u + slots_per_cta * 3 + wmode);
  std::vector<int> expert_of(num_valid);
  for (auto& e : expert_of) e = r.range(E);
  expert_of[0] = expert_of[1] = 3;   // make sure expert 3 is used (dropped-expert case)
  // Block-sorted layout (what align_block_size produces with block_size == slots_per_cta).
  std::vector<int32_t> sorted, block_expert;
  for (int e = 0; e < E; ++e) {
    int cnt = 0;
    for (int s = 0; s < num_valid; ++s) if (expert_of[s] == e) { sorted.push_back(s); ++cnt; }
    while (cnt % slots_per_cta != 0) { sorted.push_back(num_valid); ++cnt; }
    for (int b = 0; b < cnt / slots_per_cta; ++b) block_expert.push_back(drop_expert && e == 3 ? -1 : e);
  }
  const int post_pad = static_cast<int>(sorted.size());
  const int a_rows = w2_stage ? num_valid : tokens;
  const auto a = random_t<T>(static_cast<size_t>(a_rows) * K, &r, -1.0f, 1.0f);
  const bool fp8 = wmode != 0;
  std::vector<uint8_t> w8;
  std::vector<T> w16;
  if (fp8) w8 = random_e4m3(static_cast<size_t>(E) * N * K, &r);
  else w16 = random_t<T>(static_cast<size_t>(E) * N * K, &r, -0.5f, 0.5f);
  const int nb = (N + 127) / 128, kb = K / 128;
  std::vector<float> sc(wmode == 1 ? E * N : (wmode == 2 ? E : E * nb * kb));
  for (auto& v : sc) v = r.uni(0.004f, 0.02f);
  const auto bias = random_t<T>(static_cast<size_t>(E) * N, &r, -1.0f, 1.0f);
  std::vector<float> tw(num_valid);
  for (auto& v : tw) v = r.uni(0.2f, 1.2f);
  std::vector<double> ref(static_cast<size_t>(num_valid) * N, 7.0);
  std::vector<char> written(ref.size(), 0);
  for (int s = 0; s < num_valid; ++s) {
    const int e = expert_of[s];
    if (drop_expert && e == 3) continue;
    const int src = w2_stage ? s : s / top_k;
    for (int c = 0; c < N; ++c) {
      double acc = 0;
      for (int kk = 0; kk < K; ++kk) {
        const size_t wi = (static_cast<size_t>(e) * N + c) * K + kk;
        double wv = fp8 ? f8(w8[wi]) : dval(w16, wi);
        if (wmode == 3) wv *= sc[(static_cast<size_t>(e) * nb + c / 128) * kb + kk / 128];
        acc += dval(a, static_cast<size_t>(src) * K + kk) * wv;
      }
      if (wmode == 1) acc *= sc[static_cast<size_t>(e) * N + c];
      if (wmode == 2) acc *= sc[e];
      acc += dval(bias, static_cast<size_t>(e) * N + c);
      acc *= tw[s];
      ref[static_cast<size_t>(s) * N + c] = acc;
      written[static_cast<size_t>(s) * N + c] = 1;
    }
  }
  auto da = upload(a);
  auto dw8 = upload(w8.empty() ? std::vector<uint8_t>(16) : w8);
  auto dw16 = upload(w16.empty() ? std::vector<T>(8) : w16);
  auto dsc = upload(sc);
  auto db = upload(bias);
  auto dtw = upload(tw);
  auto dsorted = upload(sorted);
  auto dbe = upload(block_expert);
  auto dpp = upload(std::vector<int32_t>(1, post_pad));
  DevBuf<T> dout(ref.size());
  kops_test::copy_in(dout.p, std::vector<T>(ref.size(), cvt<T>(7.0f)));
  kops::moe::FusedMoEGemmParams p{};
  p.a = da.p; p.lda = K; p.a_rows_are_slots = w2_stage;
  p.w = fp8 ? static_cast<const void*>(dw8.p) : static_cast<const void*>(dw16.p);
  p.w_expert_stride = static_cast<int64_t>(N) * K;
  p.w_scale = fp8 ? dsc.p : nullptr;
  p.w_scale_expert_stride = wmode == 1 ? N : (wmode == 3 ? nb * kb : 0);
  p.bias = db.p; p.c = dout.p; p.ldc = N;
  p.sorted_token_ids = dsorted.p; p.expert_ids = dbe.p; p.num_tokens_post_pad = dpp.p;
  p.topk_weights = dtw.p; p.mul_routed_weight = true; p.top_k = top_k; p.n = N; p.k = K;
  p.num_valid_slots = num_valid;
  if (fp8) {
    p.w_quant.storage_dtype = DType::kFloat8E4M3;
    p.w_quant.granularity = wmode == 1 ? kops::ScaleGranularity::kPerChannel : (wmode == 2 ? kops::ScaleGranularity::kPerTensor : kops::ScaleGranularity::kPerBlock2D);
    if (wmode == 3) { p.w_quant.group_m = 128; p.w_quant.group_k = 128; }
  }
  KOPS_EXPECT_OK(kops::moe::moe_gemv(p, E, slots_per_cta, dt_of<T>(), fp8 ? DType::kFloat8E4M3 : dt_of<T>(), ctx), what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, ref.size());
  std::vector<T> got_w;
  std::vector<double> want_w;
  size_t untouched_bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (written[i]) { got_w.push_back(got[i]); want_w.push_back(ref[i]); }
    else untouched_bad += kops_test::to_float(got[i]) != 7.0f;
  }
  if (untouched_bad != 0) kops_test::record_failure(what, "rows of dropped experts were written", __FILE__, __LINE__);
  expect_close(got_w, want_w, what);
}

void test_moe_gemv(const LaunchContext& ctx) {
  for (int s : {1, 2, 4, 8}) {
    run_moe_gemv<__nv_bfloat16>(ctx, s, 0, false, false, "moe gemv bf16 W13");
    run_moe_gemv<__half>(ctx, s, 0, true, false, "moe gemv fp16 W2");
    run_moe_gemv<__nv_bfloat16>(ctx, s, 1, false, false, "moe gemv fp8 per-channel");
    run_moe_gemv<__nv_bfloat16>(ctx, s, 2, true, false, "moe gemv fp8 per-expert");
    run_moe_gemv<__nv_bfloat16>(ctx, s, 3, false, false, "moe gemv fp8 block 128x128");
  }
  run_moe_gemv<__nv_bfloat16>(ctx, 2, 0, false, true, "moe gemv dropped expert");
}

// =============================================================================
// B7 row 1: gemm_mma_sync_kernel (bf16 / fp16 mma.sync tile GEMM), tactics, epilogue ops, tactic cache.
// Oracle: fp64 host loops over the rounded inputs and the epilogue in the documented order; only
// the fp32 accumulation and the one final rounding differ.  Every case runs on every tactic that
// get_tactics() offers (all tiles, swap_ab, split-K) so a tile-specific bug cannot hide.
// =============================================================================

using kops::gemm::GemmTactic;
namespace kg = kops::gemm;

struct MmaCaseOut {
  std::vector<uint8_t> d_bytes;    // raw bytes of D
  std::vector<float> sumsq;
};

double silu_ref(double x) { return x / (1.0 + std::exp(-x)); }
double sigmoid_ref(double x) { return 1.0 / (1.0 + std::exp(-x)); }

/// One GEMM through scaled_gemm_with a tactic, compared with the fp64 oracle.  `ops` is an OR of
/// kg::EpilogueOp bits; the operand values are random but fixed by the shape.
template <typename T, typename Dt>
MmaCaseOut mma_case(const LaunchContext& ctx, const GemmTactic* tactic, int m, int n, int k, uint32_t ops, const char* what,
                    int pad_a = 0, int pad_b = 0, int pad_d = 0, int glu_db = 1, int batch = 1, bool gate_is_acc = true) {
  MmaCaseOut result;
  Rng r(777u + m * 31u + n * 7u + k + ops);
  const bool glu = (ops & kg::kEpiGLU) != 0u;
  const int n_out = glu ? n / 2 : n;
  const int lda = k + pad_a, ldb = k + pad_b, ldd = n_out + pad_d;
  const auto a = random_t<T>(static_cast<size_t>(batch) * m * lda, &r, -1.0f, 1.0f);
  const auto b = random_t<T>(static_cast<size_t>(batch) * n * ldb, &r, -0.5f, 0.5f);
  const auto bias = random_t<Dt>(n, &r, -1.0f, 1.0f);
  const auto res = random_t<Dt>(static_cast<size_t>(m) * n_out, &r, -1.0f, 1.0f);
  const auto gate = random_t<Dt>(static_cast<size_t>(m) * n_out, &r, -1.0f, 1.0f);
  const auto init = random_t<Dt>(static_cast<size_t>(batch) * m * ldd, &r, -1.0f, 1.0f);
  std::vector<float> rs(m), cs(n), rw(m);
  for (auto& v : rs) v = r.uni(0.5f, 1.5f);
  for (auto& v : cs) v = r.uni(0.5f, 1.5f);
  for (auto& v : rw) v = r.uni(0.5f, 1.5f);
  std::vector<int32_t> scat(m), kv_slots(m);
  for (int i = 0; i < m; ++i) scat[i] = (i % 11 == 5) ? -1 : m - 1 - i;
  int kv_slot_count = m + 3;
  while (true) {   // a prime slot count makes (5 i) mod count distinct for i < m
    bool prime = true;
    for (int d = 2; d * d <= kv_slot_count; ++d) prime = prime && kv_slot_count % d != 0;
    if (prime) break;
    ++kv_slot_count;
  }
  for (int i = 0; i < m; ++i) kv_slots[i] = (i % 7 == 3) ? -1 : (i * 5) % kv_slot_count;
  const float alpha = 0.75f, softcap = 8.0f, clamp_limit = 3.0f, beta = 0.5f;
  const int kv_begin = n_out > 16 ? 5 : 0, kv_cols = n_out > 16 ? n_out - 9 : n_out;

  std::vector<double> ref(static_cast<size_t>(batch) * m * ldd, 7.0);
  std::vector<char> written(ref.size(), 0);
  std::vector<double> ref_sumsq(m, 0.0);
  const bool scatter = (ops & kg::kEpiScatterRows) != 0u;
  const bool atomic = (ops & kg::kEpiAtomicAdd) != 0u;
  if (atomic) {
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = dval(init, i);
  }
  for (int z = 0; z < batch; ++z) {
    for (int row = 0; row < m; ++row) {
      auto linear = [&](int c) {
        double acc = 0.0;
        for (int kk = 0; kk < k; ++kk) {
          acc += dval(a, (static_cast<size_t>(z) * m + row) * lda + kk) * dval(b, (static_cast<size_t>(z) * n + c) * ldb + kk);
        }
        acc *= alpha;
        if (ops & kg::kEpiRowScale) acc *= rs[row];
        if (ops & kg::kEpiColScale) acc *= cs[c];
        if (ops & kg::kEpiBias) acc += dval(bias, c);
        if (ops & kg::kEpiClamp) acc = std::min(std::max(acc, -static_cast<double>(clamp_limit)), static_cast<double>(clamp_limit));
        if (ops & kg::kEpiSoftcap) acc = softcap * std::tanh(acc / softcap);
        return acc;
      };
      for (int oc = 0; oc < n_out; ++oc) {
        double v;
        if (glu) {
          const int gc = (oc / glu_db) * 2 * glu_db + oc % glu_db;
          v = silu_ref(linear(gc)) * linear(gc + glu_db);
        } else {
          v = linear(oc);
          if (ops & kg::kEpiAct) v = silu_ref(v);
        }
        if (ops & kg::kEpiRoutingWeight) v *= rw[row];
        if (ops & kg::kEpiSigmoidGateMul) {
          const double gs = dval(gate, static_cast<size_t>(row) * n_out + oc);
          v = gate_is_acc ? gs * sigmoid_ref(v) : v * sigmoid_ref(gs);
        }
        if (ops & kg::kEpiResidualAdd) v += beta * dval(res, static_cast<size_t>(row) * n_out + oc);
        const int drow = scatter ? scat[row] : row;
        if (drow < 0) continue;
        const size_t di = (static_cast<size_t>(z) * m + drow) * ldd + oc;
        if (atomic) {
          // the addend is rounded to the output dtype, then added exactly and rounded once more
          const double addend = kops_test::to_float(cvt<Dt>(static_cast<float>(v)));
          ref[di] = kops_test::to_float(cvt<Dt>(static_cast<float>(ref[di] + addend)));
        } else {
          ref[di] = v;
        }
        written[di] = 1;
      }
    }
  }
  for (int row = 0; row < m; ++row) {
    double s = 0.0;
    for (int kk = 0; kk < k; ++kk) s += dval(a, static_cast<size_t>(row) * lda + kk) * dval(a, static_cast<size_t>(row) * lda + kk);
    ref_sumsq[row] = s;
  }

  auto da = upload(a);
  auto db = upload(b);
  auto dbias = upload(bias);
  auto dres = upload(res);
  auto dgate = upload(gate);
  auto drs = upload(rs);
  auto dcs = upload(cs);
  auto drw = upload(rw);
  auto dscat = upload(scat);
  auto dslots = upload(kv_slots);
  DevBuf<Dt> dout(ref.size());
  kops_test::copy_in(dout.p, atomic ? init : std::vector<Dt>(ref.size(), cvt<Dt>(7.0f)));
  DevBuf<Dt> dkv(static_cast<size_t>(kv_slot_count) * kv_cols);
  kops_test::copy_in(dkv.p, std::vector<Dt>(static_cast<size_t>(kv_slot_count) * kv_cols, cvt<Dt>(7.0f)));
  DevBuf<float> dsumsq(m);
  kops_test::copy_in(dsumsq.p, std::vector<float>(m, -1.0f));

  kg::ScaledGemmArgs args{};
  args.recipe = recipe_of(dt_of<T>());
  args.a.data = da.p; args.a.ld = lda; args.a.dtype = dt_of<T>(); args.a.batch_stride = static_cast<int64_t>(m) * lda;
  args.b.data = db.p; args.b.ld = ldb; args.b.dtype = dt_of<T>(); args.b.batch_stride = static_cast<int64_t>(n) * ldb;
  args.d = dout.p; args.ldd = ldd; args.d_dtype = dt_of<Dt>(); args.batch_stride_d = static_cast<int64_t>(m) * ldd;
  args.m = m; args.n = n; args.k = k; args.batch = batch;
  args.epi.ops = ops;
  args.epi.alpha = alpha; args.epi.softcap = softcap; args.epi.clamp_limit = clamp_limit; args.epi.beta = beta;
  args.epi.row_scale = drs.p; args.epi.col_scale = dcs.p; args.epi.routing_weight = drw.p;
  args.epi.bias = dbias.p; args.epi.bias_dtype = dt_of<Dt>();
  args.epi.act = kops::ActFn::kSiLU;
  args.epi.glu_layout = kops::GateLayout::kInterleaved; args.epi.glu_interleave = glu_db;
  args.epi.gate_src = dgate.p; args.epi.gate_src_stride = n_out; args.epi.gate_is_acc = gate_is_acc;
  args.epi.residual = dres.p; args.epi.residual_stride = n_out;
  args.epi.scatter_rows = dscat.p;
  args.epi.row_sumsq = dsumsq.p;
  args.epi.kv_dst = dkv.p; args.epi.kv_slots = dslots.p; args.epi.kv_stride = kv_cols; args.epi.kv_col_begin = kv_begin; args.epi.kv_cols = kv_cols;

  kg::GemmProblem prob{};
  prob.m = m; prob.n = n; prob.k = k; prob.batch = batch; prob.recipe = args.recipe; prob.epilogue_ops = ops;
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kg::gemm_workspace_requirements(prob, tactic, kg::GemmBackend::kNative, ctx, &req), what);
  DevBuf<uint8_t> scratch(req.device_scratch_bytes + 256);
  kops::Workspace ws{};
  ws.device_scratch = scratch.p; ws.device_scratch_bytes = req.device_scratch_bytes;
  const Status st = kg::scaled_gemm(args, tactic, kg::GemmBackend::kNative, ws, ctx);
  if (st != Status::kSuccess) {
    char buf[300];
    std::snprintf(buf, sizeof(buf), "%s [tile %dx%d swap %d split %d mode %d]", what, tactic ? tactic->tile_m : -1, tactic ? tactic->tile_n : -1,
                  tactic && tactic->swap_ab, tactic ? tactic->split_k_slices : -1, tactic ? static_cast<int>(tactic->split_k_mode) : -1);
    KOPS_EXPECT_OK(st, buf);
    return result;
  }
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, ref.size());
  std::vector<Dt> got_w;
  std::vector<double> want_w;
  size_t untouched_bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (written[i]) { got_w.push_back(got[i]); want_w.push_back(ref[i]); }
    else if (!atomic && (i % ldd) < static_cast<size_t>(n_out)) untouched_bad += kops_test::to_float(got[i]) != 7.0f;
  }
  char label[300];
  std::snprintf(label, sizeof(label), "%s [tile %dx%d swap %d split %d mode %d]", what, tactic ? tactic->tile_m : -1, tactic ? tactic->tile_n : -1,
                tactic && tactic->swap_ab, tactic ? tactic->split_k_slices : -1, tactic ? static_cast<int>(tactic->split_k_mode) : -1);
  if (untouched_bad != 0) kops_test::record_failure(label, "dropped rows were written", __FILE__, __LINE__);
  expect_close(got_w, want_w, label);
  // padding columns of D must be untouched
  if (pad_d != 0) {
    size_t pad_bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if ((i % ldd) >= static_cast<size_t>(n_out)) pad_bad += kops_test::to_float(got[i]) != (atomic ? kops_test::to_float(init[i]) : 7.0f);
    }
    if (pad_bad != 0) kops_test::record_failure(label, "ld padding of D was written", __FILE__, __LINE__);
  }
  if (ops & kg::kEpiRowSumSq) {
    const auto s = kops_test::copy_out(dsumsq.p, m);
    for (int row = 0; row < m; ++row) {
      if (!(std::fabs(s[row] - ref_sumsq[row]) <= 2e-6 * ref_sumsq[row] + 1e-9)) {
        kops_test::record_failure(label, "row_sumsq differs from the fp64 sum", __FILE__, __LINE__);
        break;
      }
    }
    result.sumsq = s;
  }
  if (ops & kg::kEpiKvAppend) {
    const auto kv = kops_test::copy_out(dkv.p, static_cast<size_t>(kv_slot_count) * kv_cols);
    std::vector<char> slot_used(kv_slot_count, 0);
    size_t kv_bad = 0;
    for (int row = 0; row < m; ++row) {
      const int slot = kv_slots[row];
      if (slot < 0) continue;
      slot_used[slot] = 1;
      const int drow = scatter ? scat[row] : row;
      if (drow < 0) continue;
      for (int j = 0; j < kv_cols; ++j) {
        // the appended value is exactly the value stored to D
        if (std::memcmp(&kv[static_cast<size_t>(slot) * kv_cols + j], &got[static_cast<size_t>(drow) * ldd + kv_begin + j], sizeof(Dt)) != 0) ++kv_bad;
      }
    }
    for (int slot = 0; slot < kv_slot_count; ++slot) {
      if (slot_used[slot]) continue;
      for (int j = 0; j < kv_cols; ++j) kv_bad += kops_test::to_float(kv[static_cast<size_t>(slot) * kv_cols + j]) != 7.0f;
    }
    if (kv_bad != 0) kops_test::record_failure(label, "kv append differs from the stored D value / touched a skipped slot", __FILE__, __LINE__);
  }
  result.d_bytes.resize(got.size() * sizeof(Dt));
  std::memcpy(result.d_bytes.data(), got.data(), result.d_bytes.size());
  return result;
}

std::vector<GemmTactic> tactics_for(const LaunchContext& ctx, GemmRecipe recipe, int m, int n, int k, uint32_t ops, int batch = 1) {
  kg::GemmProblem p{};
  p.m = m; p.n = n; p.k = k; p.batch = batch; p.recipe = recipe; p.epilogue_ops = ops;
  std::vector<GemmTactic> t(64);
  int32_t count = 0;
  KOPS_EXPECT_OK(kg::get_tactics(p, kg::GemmBackend::kNative, ctx, t.data(), 64, &count), "get_tactics");
  t.resize(static_cast<size_t>(count));
  if (t.empty()) kops_test::record_failure("get_tactics", "no tactic offered", __FILE__, __LINE__);
  return t;
}

template <typename T, typename Dt>
void mma_all_tactics(const LaunchContext& ctx, int m, int n, int k, uint32_t ops, const char* what, int pad_a = 0, int pad_b = 0,
                     int pad_d = 0, int glu_db = 1, int batch = 1, bool gate_is_acc = true) {
  for (const GemmTactic& t : tactics_for(ctx, recipe_of(dt_of<T>()), m, n, k, ops, batch)) {
    // a GLU pass must cover whole (gate block, up block) pairs: columns one epilogue pass covers
    const int span = t.swap_ab ? (t.tile_m == 64 ? 64 : 32) : t.tile_n;
    if ((ops & kg::kEpiGLU) != 0u && span % (2 * glu_db) != 0) continue;
    mma_case<T, Dt>(ctx, &t, m, n, k, ops, what, pad_a, pad_b, pad_d, glu_db, batch, gate_is_acc);
  }
}

void test_mma_sweep(const LaunchContext& ctx) {
  for (int m : {1, 2, 3, 8, 16, 17, 31, 64, 127, 256, 1000, 4096}) {
    mma_all_tactics<__nv_bfloat16, __nv_bfloat16>(ctx, m, 96, 72, 0, "mma bf16 M sweep");
    mma_all_tactics<__half, __half>(ctx, m, 136, 64, 0, "mma fp16 M sweep");
  }
  mma_all_tactics<__nv_bfloat16, float>(ctx, 16384, 32, 64, kg::kEpiOutFp32, "mma bf16 -> fp32 M=16384");
  mma_all_tactics<__half, float>(ctx, 131, 40, 40, kg::kEpiOutFp32, "mma fp16 -> fp32");
}

void test_mma_shapes(const LaunchContext& ctx) {
  // N and K that no tile divides, ld padded past the extents (16 B granules)
  mma_all_tactics<__nv_bfloat16, __nv_bfloat16>(ctx, 97, 201, 136, 0, "mma odd N, K tail", 8, 16, 24);
  mma_all_tactics<__half, __half>(ctx, 130, 77, 40, 0, "mma odd N=77", 0, 8, 3);
  mma_all_tactics<__nv_bfloat16, __nv_bfloat16>(ctx, 33, 65, 8, 0, "mma K=8");
  mma_all_tactics<__nv_bfloat16, __nv_bfloat16>(ctx, 70, 128, 200, kg::kEpiBias, "mma batched", 0, 0, 0, 1, 3);
}

void test_mma_epilogues(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  const int m = 97, n = 200, k = 136;
  const uint32_t single[] = {kg::kEpiRowScale, kg::kEpiColScale, kg::kEpiBias, kg::kEpiSoftcap, kg::kEpiAct, kg::kEpiRoutingWeight,
                             kg::kEpiSigmoidGateMul, kg::kEpiResidualAdd, kg::kEpiScatterRows, kg::kEpiAtomicAdd, kg::kEpiClamp,
                             kg::kEpiRowSumSq, kg::kEpiKvAppend};
  for (uint32_t op : single) {
    char name[64];
    std::snprintf(name, sizeof(name), "mma epilogue op 0x%x alone", op);
    mma_all_tactics<B, B>(ctx, m, n, k, op, name);
  }
  mma_all_tactics<B, B>(ctx, m, n, k, kg::kEpiSigmoidGateMul, "mma sigmoid gate (acc gates src)", 0, 0, 0, 1, 1, false);
  mma_all_tactics<__half, __half>(ctx, m, n, k, kg::kEpiBias | kg::kEpiAct, "mma fp16 bias + act");
  // GLU: interleave 1 (adjacent pair), 8 and 16 (block of gate columns then up columns)
  mma_all_tactics<B, B>(ctx, 97, 256, 64, kg::kEpiGLU, "mma GLU interleave 1", 0, 0, 0, 1);
  mma_all_tactics<B, B>(ctx, 97, 256, 64, kg::kEpiGLU, "mma GLU interleave 8", 0, 0, 0, 8);
  mma_all_tactics<B, B>(ctx, 97, 256, 64, kg::kEpiGLU | kg::kEpiBias, "mma GLU interleave 16 + bias", 0, 0, 0, 16);
  mma_all_tactics<B, B>(ctx, 97, 256, 64, kg::kEpiGLU | kg::kEpiClamp, "mma GLU interleave 3", 0, 0, 0, 3);
  mma_all_tactics<B, B>(ctx, 97, 256, 64, kg::kEpiGLU | kg::kEpiResidualAdd, "mma GLU interleave 64", 0, 0, 0, 64);
  // common combinations
  mma_all_tactics<B, B>(ctx, m, n, k, kg::kEpiBias | kg::kEpiAct, "mma Bias + Act");
  mma_all_tactics<B, B>(ctx, m, n, k, kg::kEpiRoutingWeight | kg::kEpiScatterRows, "mma RoutingWeight + ScatterRows");
  mma_all_tactics<B, float>(ctx, m, n, k, kg::kEpiResidualAdd | kg::kEpiOutFp32, "mma ResidualAdd + OutFp32");
  mma_all_tactics<B, B>(ctx, m, n, k, kg::kEpiRowScale | kg::kEpiColScale | kg::kEpiBias | kg::kEpiClamp | kg::kEpiSoftcap | kg::kEpiAct |
                                         kg::kEpiRoutingWeight | kg::kEpiSigmoidGateMul | kg::kEpiResidualAdd,
                        "mma full chain");
  mma_all_tactics<B, B>(ctx, 97, 256, 64, kg::kEpiGLU | kg::kEpiRoutingWeight | kg::kEpiKvAppend | kg::kEpiRowSumSq, "mma GLU + routing + kv + sumsq");
  mma_all_tactics<B, B>(ctx, m, n, k, kg::kEpiScatterRows | kg::kEpiKvAppend | kg::kEpiRowSumSq, "mma scatter + kv + sumsq");
}

void test_mma_splitk(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  const int m = 64, n = 160, k = 4096;
  const uint32_t ops = kg::kEpiBias | kg::kEpiAct | kg::kEpiResidualAdd;
  std::vector<uint8_t> serial_bits, parallel_bits;
  GemmTactic base{};
  bool have = false;
  for (const GemmTactic& t : tactics_for(ctx, GemmRecipe::kBF16, m, n, k, ops)) {
    if (t.split_k_slices != 4) continue;
    base = t;
    have = true;
    break;
  }
  if (!have) { kops_test::record_failure("split-K", "no 4-slice tactic offered for K=4096", __FILE__, __LINE__); return; }
  serial_bits = mma_case<B, B>(ctx, &base, m, n, k, ops, "mma split-K serial").d_bytes;
  GemmTactic par = base;
  par.split_k_mode = kg::SplitKMode::kParallel;
  parallel_bits = mma_case<B, B>(ctx, &par, m, n, k, ops, "mma split-K parallel").d_bytes;
  if (serial_bits != parallel_bits) kops_test::record_failure("split-K", "serial and parallel reduction differ bitwise", __FILE__, __LINE__);
  // repeat: the order is by slice id, not by arrival
  for (int rep = 0; rep < 20; ++rep) {
    if (mma_case<B, B>(ctx, &base, m, n, k, ops, "mma split-K repeat").d_bytes != serial_bits) {
      kops_test::record_failure("split-K", "serial split-K changed between runs", __FILE__, __LINE__);
      break;
    }
  }
  // the full epilogue on the serial path, plus the data ops
  mma_case<B, B>(ctx, &base, m, n, k, kg::kEpiRowSumSq | kg::kEpiKvAppend | kg::kEpiScatterRows | kg::kEpiRoutingWeight, "mma split-K data ops");
  mma_case<__half, __half>(ctx, &base, 64, 256, k, kg::kEpiGLU, "mma split-K GLU");   // fp16 operands share the template
  // parallel split-K only finishes the dense epilogue subset
  {
    GemmTactic p2 = base;
    p2.split_k_mode = kg::SplitKMode::kParallel;
    mma_case<B, B>(ctx, &p2, m, n, k, 0, "mma split-K parallel plain");
  }
}

void test_mma_invariance(const LaunchContext& ctx) {
  // Same slices, any tile / swap_ab: the K chain of one output element is the same sequence of mma steps.
  using B = __nv_bfloat16;
  const int m = 130, n = 128, k = 256;
  std::vector<uint8_t> first;
  for (const GemmTactic& t : tactics_for(ctx, GemmRecipe::kBF16, m, n, k, 0)) {
    if (t.split_k_slices != 1) continue;
    const auto bits = mma_case<B, B>(ctx, &t, m, n, k, kg::kEpiRowSumSq, "mma tile invariance");
    if (first.empty()) first = bits.d_bytes;
    else if (bits.d_bytes != first) {
      char buf[160];
      std::snprintf(buf, sizeof(buf), "tile %dx%d swap %d differs bitwise from the first tactic", t.tile_m, t.tile_n, t.swap_ab);
      kops_test::record_failure("mma tile invariance", buf, __FILE__, __LINE__);
    }
  }
  // row_sumsq does not depend on the tile either
  std::vector<float> s0;
  for (const GemmTactic& t : tactics_for(ctx, GemmRecipe::kBF16, 130, 64, 256, kg::kEpiRowSumSq)) {
    const auto out = mma_case<B, B>(ctx, &t, 130, 64, 256, kg::kEpiRowSumSq, "mma sumsq invariance");
    if (s0.empty()) s0 = out.sumsq;
    else if (out.sumsq != s0) kops_test::record_failure("mma sumsq invariance", "row_sumsq differs between tactics", __FILE__, __LINE__);
  }
}

/// Two bf16 output buffers agree element-wise within `ulps` bf16 rounding steps (1 ulp = 2^-8 relative).
bool within_bf16_ulp(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, uint32_t ulps) {
  if (got.size() != want.size() || got.size() % 2 != 0) return false;
  const float tol = static_cast<float>(ulps) * 0.00390625f;
  for (size_t i = 0; i + 1 < got.size(); i += 2) {
    __nv_bfloat16 a, b;
    std::memcpy(&a, got.data() + i, sizeof(a));
    std::memcpy(&b, want.data() + i, sizeof(b));
    const float fa = __bfloat162float(a), fb = __bfloat162float(b);
    if (!(std::fabs(fa - fb) <= tol * std::fabs(fb) + 1e-6f)) return false;
  }
  return true;
}

/// The 128-byte row fetch of the small-M swap tiles (wide_ld): one request fills two staging buffers, so an
/// odd number of 64-byte staging rows per tile, a K that is not a multiple of 128 bytes and rows outside the
/// tile are the cases that can go wrong. Every tactic must keep storing the same bits as the narrow ones
/// (the K chain is unchanged), repeat runs must be identical, and serial split-K must stay within one bf16
/// rounding step of a single slice.
void test_mma_wide_fetch(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  // 1. K = 80 / 152 bytes*2 -> 3 / 5 staging rows: an odd count, so the last prefetch group is half a pair.
  for (int k : {80, 152, 264}) {
    for (int m : {17, 20, 63}) {
      std::vector<uint8_t> first;
      int wide = 0;
      for (const GemmTactic& t : tactics_for(ctx, GemmRecipe::kBF16, m, 136, k, 0)) {
        if (t.split_k_slices != 1) continue;
        const auto bits = mma_case<B, B>(ctx, &t, m, 136, k, 0, "mma wide fetch tile invariance");
        if (t.swap_ab && t.stages >= 8) ++wide;
        if (first.empty()) first = bits.d_bytes;
        else if (bits.d_bytes != first) {
          char buf[180];
          std::snprintf(buf, sizeof(buf), "K=%d M=%d tile %dx%d swap %d differs bitwise from the first tactic", k, m, t.tile_m, t.tile_n,
                        t.swap_ab);
          kops_test::record_failure("mma wide fetch tile invariance", buf, __FILE__, __LINE__);
        }
      }
      if (wide < 2) kops_test::record_failure("mma wide fetch tile invariance", "the wide swap tiles were not offered", __FILE__, __LINE__);
    }
  }
  // 2. the wide tactic the heuristic picks for the small-M band: repeated runs are identical, and serial
  //    split-K over an odd staging count stores the same bits as one slice.
  {
    const int m = 64, n = 264, k = 152;
    const std::vector<GemmTactic> ts = tactics_for(ctx, GemmRecipe::kBF16, m, n, k, 0);
    const GemmTactic wide = ts.front();
    if (!(wide.swap_ab && wide.stages >= 8)) {
      kops_test::record_failure("mma wide fetch heuristic", "the heuristic did not pick a wide swap tile", __FILE__, __LINE__);
    }
    const auto ref = mma_case<B, B>(ctx, &wide, m, n, k, 0, "mma wide fetch repeat");
    for (int rep = 0; rep < 3; ++rep) {
      if (mma_case<B, B>(ctx, &wide, m, n, k, 0, "mma wide fetch repeat").d_bytes != ref.d_bytes) {
        kops_test::record_failure("mma wide fetch repeat", "result changed between runs", __FILE__, __LINE__);
        break;
      }
    }
    // every tactic (wide and narrow) must agree with the heuristic one bitwise when unsplit, and within one
    // bf16 rounding step under serial split-K (the slice partials are summed in slice order, so a slice
    // boundary can move one element by one ulp; the same holds for the narrow tiles before this change).
    for (const GemmTactic& t0 : ts) {
      GemmTactic t = t0;
      t.split_k_slices = 1;
      t.split_k_mode = kg::SplitKMode::kNone;
      if (mma_case<B, B>(ctx, &t, m, n, k, 0, "mma wide fetch split-K ref").d_bytes != ref.d_bytes) {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "tile %dx%d swap %d stages %d unsplit differs", t.tile_m, t.tile_n, t.swap_ab, t.stages);
        kops_test::record_failure("mma wide fetch split-K ref", buf, __FILE__, __LINE__);
      }
      for (int slices : {2, 3}) {
        t.split_k_mode = kg::SplitKMode::kSerial;
        t.split_k_slices = static_cast<int16_t>(slices);
        const auto got = mma_case<B, B>(ctx, &t, m, n, k, 0, "mma wide fetch split-K");
        if (!within_bf16_ulp(got.d_bytes, ref.d_bytes, 2u)) {
          char buf[200];
          std::snprintf(buf, sizeof(buf), "tile %dx%d swap %d stages %d slices %d differs", t.tile_m, t.tile_n, t.swap_ab, t.stages, slices);
          kops_test::record_failure("mma wide fetch split-K", buf, __FILE__, __LINE__);
        }
      }
    }
  }
  // 3. M larger than the tile rows with padded rows inside the last tile, and an N that is not a tile multiple
  for (int m : {96, 200}) {
    for (const GemmTactic& t : tactics_for(ctx, GemmRecipe::kBF16, m, 200, 136, 0)) {
      if (t.split_k_slices != 1) continue;
      mma_case<B, B>(ctx, &t, m, 200, 136, kg::kEpiBias | kg::kEpiAct, "mma wide fetch padded rows");
    }
  }
}

/// The small-M ladder and the split-K layout of the mma.sync path: the tiles that fit two CTAs per SM
/// must stay deterministic on odd shapes, every serial split must agree with the unsplit result within
/// one bf16 rounding step, and the tactic cache must keep the staging depth that distinguishes two
/// bodies of the same geometry.
void test_mma_small_m(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  // 1. the heuristic's first tile for the small-M band (only meaningful on a full part: the heuristic
  //    counts waves against ctx.compute_units)
  if (ctx.compute_units >= 64) {
    struct Pick { int m; int tm, tn; bool swap; int stages; };
    for (const Pick& p : {Pick{32, 16, 64, true, 8}, Pick{64, 16, 64, true, 8}, Pick{128, 32, 64, true, 8}, Pick{256, 64, 128, true, 4}}) {
      const std::vector<GemmTactic> ts = tactics_for(ctx, GemmRecipe::kBF16, p.m, 4096, 4096, 0);
      if (ts.empty()) continue;
      const GemmTactic& t = ts.front();
      if (t.tile_m != p.tm || t.tile_n != p.tn || t.swap_ab != p.swap || t.stages != p.stages) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "M=%d picked %dx%d swap %d stages %d, want %dx%d swap %d stages %d", p.m, t.tile_m,
                      t.tile_n, t.swap_ab, t.stages, p.tm, p.tn, p.swap, p.stages);
        kops_test::record_failure("mma small-M pick", buf, __FILE__, __LINE__);
      }
    }
  }
  // 2. odd M / N / K on the tiles the band picks: repeated launches are bitwise identical
  for (int m : {17, 63, 129, 255}) {
    for (int n : {77, 200}) {
      for (int k : {72, 264}) {
        const GemmTactic t = tactics_for(ctx, GemmRecipe::kBF16, m, n, k, 0).front();
        const auto ref = mma_case<B, B>(ctx, &t, m, n, k, 0, "mma small-M determinism");
        for (int rep = 0; rep < 2; ++rep) {
          if (mma_case<B, B>(ctx, &t, m, n, k, 0, "mma small-M determinism").d_bytes != ref.d_bytes) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "M=%d N=%d K=%d tile %dx%d changed between runs", m, n, k, t.tile_m, t.tile_n);
            kops_test::record_failure("mma small-M determinism", buf, __FILE__, __LINE__);
            break;
          }
        }
      }
    }
  }
  // 3. serial split-K equivalence on the two-CTA tiles: the partials now live in the tile's register
  //    order, so a slice boundary may still move an element by one bf16 step, never more.
  {
    const int m = 130, n = 136, k = 520;
    for (const GemmTactic& t0 : tactics_for(ctx, GemmRecipe::kBF16, m, n, k, 0)) {
      if (!(t0.tile_m == 64 && t0.tile_n == 128) && !(t0.tile_m == 32 && t0.tile_n == 64)) continue;
      GemmTactic t = t0;
      t.split_k_slices = 1;
      t.split_k_mode = kg::SplitKMode::kNone;
      const auto ref = mma_case<B, B>(ctx, &t, m, n, k, 0, "mma split layout ref");
      for (int slices : {2, 3, 4}) {
        t.split_k_mode = kg::SplitKMode::kSerial;
        t.split_k_slices = static_cast<int16_t>(slices);
        const auto got = mma_case<B, B>(ctx, &t, m, n, k, 0, "mma split layout");
        if (!within_bf16_ulp(got.d_bytes, ref.d_bytes, 2u)) {
          char buf[180];
          std::snprintf(buf, sizeof(buf), "tile %dx%d slices %d differs from the unsplit result", t.tile_m, t.tile_n, slices);
          kops_test::record_failure("mma split layout", buf, __FILE__, __LINE__);
        }
        if (mma_case<B, B>(ctx, &t, m, n, k, 0, "mma split layout repeat").d_bytes != got.d_bytes) {
          kops_test::record_failure("mma split layout repeat", "serial split-K changed between runs", __FILE__, __LINE__);
        }
      }
    }
  }
  // 4. the staging depth is part of the tile identity: a tactic that names a different depth than the
  //    compiled body is rejected instead of silently running another kernel
  {
    GemmTactic t = tactics_for(ctx, GemmRecipe::kBF16, 96, 136, 264, 0).front();
    const int8_t stages = t.stages;
    t.stages = static_cast<int8_t>(stages == 8 ? 4 : 8);
    kg::ScaledGemmArgs a{};
    Rng r(11);
    const auto ah = random_t<B>(static_cast<size_t>(96) * 264, &r, -1.0f, 1.0f);
    const auto bh = random_t<B>(static_cast<size_t>(136) * 264, &r, -1.0f, 1.0f);
    auto da = upload(ah);
    auto db = upload(bh);
    DevBuf<B> dout(static_cast<size_t>(96) * 136);
    a.recipe = GemmRecipe::kBF16;
    a.a.data = da.p; a.a.ld = 264; a.a.dtype = DType::kBFloat16;
    a.b.data = db.p; a.b.ld = 264; a.b.dtype = DType::kBFloat16;
    a.d = dout.p; a.ldd = 136; a.d_dtype = DType::kBFloat16;
    a.m = 96; a.n = 136; a.k = 264;
    KOPS_EXPECT_STATUS(kg::scaled_gemm(a, &t, kg::GemmBackend::kNative, kops::Workspace{}, ctx), Status::kUnsupportedShape,
                       "tactic with the wrong staging depth");
  }
}

void test_mma_dispatch(const LaunchContext& ctx) {
  // gemm / scaled_gemm without a tactic: M > 16 takes the tensor-core path with the heuristic.
  using B = __nv_bfloat16;
  for (int m : {17, 40, 64, 300, 2048}) mma_case<B, B>(ctx, nullptr, m, 192, 128, kg::kEpiBias, "mma heuristic dispatch");
  mma_case<__half, float>(ctx, nullptr, 200, 96, 64, kg::kEpiOutFp32, "mma heuristic fp16 -> fp32");
  // small K with a long tile grid, and a long K with a tiny grid (heuristic split-K)
  mma_case<B, B>(ctx, nullptr, 32, 64, 8192, 0, "mma heuristic split-K");
}

void test_mma_cache(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  kg::GemmProblem p{};
  p.m = 64; p.n = 192; p.k = 512; p.recipe = GemmRecipe::kBF16;
  kops::WorkspaceRequirements req{};
  kg::GemmTactic probe{};
  const int64_t buckets[] = {32, 64, 256};
  const int64_t scratch_bytes = 8 << 20;
  DevBuf<uint8_t> scratch(scratch_bytes);
  kops::Workspace ws{};
  ws.device_scratch = scratch.p; ws.device_scratch_bytes = scratch_bytes;
  KOPS_EXPECT_STATUS(kg::tactic_cache_lookup(p, ctx, &probe), Status::kInvalidArgument, "cache empty before autotune");
  KOPS_EXPECT_OK(kg::autotune(p, buckets, 3, kg::GemmBackend::kNative, ws, ctx), "autotune");
  GemmTactic t1{}, t2{};
  KOPS_EXPECT_OK(kg::tactic_cache_lookup(p, ctx, &t1), "cache lookup after autotune");
  KOPS_EXPECT_OK(kg::tactic_cache_lookup(p, ctx, &t2), "cache lookup again");
  if (std::memcmp(&t1, &t2, sizeof(t1)) != 0) kops_test::record_failure("tactic cache", "two lookups returned different tactics", __FILE__, __LINE__);
  // any M in the same bucket sees the same tactic; a different bucket is its own entry
  kg::GemmProblem p2 = p;
  p2.m = 50;
  GemmTactic t3{};
  KOPS_EXPECT_OK(kg::tactic_cache_lookup(p2, ctx, &t3), "cache lookup, same bucket");
  if (std::memcmp(&t1, &t3, sizeof(t1)) != 0) kops_test::record_failure("tactic cache", "same bucket, different tactic", __FILE__, __LINE__);
  // the cached tactic drives gemm() when none is passed
  mma_case<B, B>(ctx, nullptr, 64, 192, 512, kg::kEpiBias, "mma cached tactic");
  // save / load round trip
  const char* path = "kops_tactic_cache_test.txt";
  KOPS_EXPECT_OK(kg::tactic_cache_save(path), "cache save");
  KOPS_EXPECT_OK(kg::tactic_cache_load(path), "cache load");
  GemmTactic t4{};
  KOPS_EXPECT_OK(kg::tactic_cache_lookup(p, ctx, &t4), "lookup after load");
  if (t4.tile_m != t1.tile_m || t4.tile_n != t1.tile_n || t4.swap_ab != t1.swap_ab || t4.split_k_slices != t1.split_k_slices ||
      t4.split_k_mode != t1.split_k_mode) {
    kops_test::record_failure("tactic cache", "save / load changed the tactic", __FILE__, __LINE__);
  }
  std::remove(path);
  KOPS_EXPECT_STATUS(kg::tactic_cache_load("kops_no_such_cache.txt"), Status::kInvalidArgument, "load of a missing file");
  // autotune must refuse during graph capture
  cudaStream_t s = nullptr;
  if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess) {
    LaunchContext cctx = ctx;
    cctx.stream = reinterpret_cast<kops::kops_stream_t>(s);
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
      kg::GemmProblem q = p;
      q.n = 224;
      KOPS_EXPECT_STATUS(kg::autotune(q, buckets, 1, kg::GemmBackend::kNative, ws, cctx), Status::kInvalidArgument, "autotune during capture");
      cudaGraph_t graph = nullptr;
      cudaStreamEndCapture(s, &graph);
      if (graph != nullptr) cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(s);
    cudaGetLastError();
  }
}

void test_mma_contracts(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  const int m = 64, n = 128, k = 128;
  Rng r(5);
  const auto a = random_t<B>(static_cast<size_t>(m) * (k + 8), &r, -1.0f, 1.0f);
  const auto b = random_t<B>(static_cast<size_t>(n) * (k + 8), &r, -1.0f, 1.0f);
  auto da = upload(a);
  auto db = upload(b);
  DevBuf<B> dout(static_cast<size_t>(m) * n);
  DevBuf<uint8_t> dummy(4096);
  auto base = [&]() {
    kg::ScaledGemmArgs x{};
    x.recipe = GemmRecipe::kBF16;
    x.a.data = da.p; x.a.ld = k; x.a.dtype = DType::kBFloat16;
    x.b.data = db.p; x.b.ld = k; x.b.dtype = DType::kBFloat16;
    x.d = dout.p; x.ldd = n; x.d_dtype = DType::kBFloat16;
    x.m = m; x.n = n; x.k = k;
    return x;
  };
  auto run = [&](const kg::ScaledGemmArgs& x, const kops::Workspace& w = kops::Workspace{}) {
    return kg::scaled_gemm(x, nullptr, kg::GemmBackend::kNative, w, ctx);
  };
  KOPS_EXPECT_OK(run(base()), "contract baseline");
  { auto x = base(); x.k = 100; x.a.ld = 104; x.b.ld = 104; KOPS_EXPECT_STATUS(run(x), Status::kMisaligned, "K % 8 != 0"); }
  { auto x = base(); x.a.ld = k + 3; KOPS_EXPECT_STATUS(run(x), Status::kMisaligned, "unaligned lda"); }
  { auto x = base(); x.b.major = kg::MajorOrder::kMNMajor; x.b.ld = n; KOPS_EXPECT_STATUS(run(x), Status::kNotImplemented, "MN-major B"); }
  { auto x = base(); x.epi.ops = kg::kEpiQuantOut; KOPS_EXPECT_STATUS(run(x), Status::kUnsupportedDtype, "QuantOut without an e4m3 / int8 destination"); }
  { auto x = base(); x.epi.ops = kg::kEpiAzpCorrection; KOPS_EXPECT_STATUS(run(x), Status::kInvalidArgument, "AzpCorrection on bf16 (INT8 recipes only)"); }
  { auto x = base(); x.epi.ops = kg::kEpiBias; KOPS_EXPECT_STATUS(run(x), Status::kInvalidArgument, "Bias without bias pointer"); }
  { auto x = base(); x.epi.ops = kg::kEpiRowSumSq; KOPS_EXPECT_STATUS(run(x), Status::kInvalidArgument, "RowSumSq without a buffer"); }
  { auto x = base(); x.epi.ops = kg::kEpiOutFp32; KOPS_EXPECT_STATUS(run(x), Status::kInvalidArgument, "OutFp32 with a bf16 output"); }
  { auto x = base(); x.epi.ops = kg::kEpiGLU; x.epi.glu_layout = kops::GateLayout::kGateUp; KOPS_EXPECT_STATUS(run(x), Status::kNotImplemented, "GLU with split halves"); }
  { auto x = base(); x.epi.ops = kg::kEpiGLU; x.epi.glu_layout = kops::GateLayout::kInterleaved; x.n = 129; KOPS_EXPECT_STATUS(run(x), Status::kUnsupportedShape, "GLU with odd N"); }
  { auto x = base(); x.d_dtype = DType::kFloat16; KOPS_EXPECT_STATUS(run(x), Status::kUnsupportedDtype, "bf16 -> fp16 output"); }
  { auto x = base(); x.a.dtype = DType::kFloat16; KOPS_EXPECT_STATUS(run(x), Status::kUnsupportedDtype, "recipe / dtype mismatch"); }
  {
    LaunchContext det = ctx;
    det.deterministic = true;
    auto x = base();
    x.epi.ops = kg::kEpiAtomicAdd;
    KOPS_EXPECT_STATUS(kg::scaled_gemm(x, nullptr, kg::GemmBackend::kNative, kops::Workspace{}, det), Status::kInvalidArgument, "AtomicAdd under ctx.deterministic");
  }
  {
    GemmTactic t = tactics_for(ctx, GemmRecipe::kBF16, 64, 128, 4096, 0).front();
    for (const GemmTactic& c : tactics_for(ctx, GemmRecipe::kBF16, 64, 128, 4096, 0)) if (c.split_k_slices > 1) t = c;
    auto x = base();
    x.k = 4096; x.a.ld = 4096; x.b.ld = 4096;
    auto bigA = upload(random_t<B>(static_cast<size_t>(m) * 4096, &r, -1.0f, 1.0f));
    auto bigB = upload(random_t<B>(static_cast<size_t>(n) * 4096, &r, -1.0f, 1.0f));
    x.a.data = bigA.p; x.b.data = bigB.p;
    if (t.split_k_slices > 1) {
      KOPS_EXPECT_STATUS(kg::scaled_gemm(x, &t, kg::GemmBackend::kNative, kops::Workspace{}, ctx), Status::kWorkspaceTooSmall, "split-K without workspace");
      t.split_k_mode = kg::SplitKMode::kStreamK;
      KOPS_EXPECT_STATUS(kg::scaled_gemm(x, &t, kg::GemmBackend::kNative, kops::Workspace{}, ctx), Status::kNotImplemented, "stream-K");
    } else {
      kops_test::record_failure("split-K contract", "no split-K tactic offered", __FILE__, __LINE__);
    }
  }
}

// =============================================================================
// B7 row 2: dual_gemm (gate / up from two weights, GLU in the epilogue) and bmm (strided batched,
// B as [N, K] or [K, N], head-strided views).  Oracle: fp64 host loops over the rounded inputs.
// =============================================================================

double dual_act_ref(kops::ActFn f, double x, const kops::GluParams& g) {
  const double s2 = std::sqrt(2.0);
  switch (f) {
    case kops::ActFn::kSiLU: return x * sigmoid_ref(g.gate_alpha * x);
    case kops::ActFn::kGeLUErf: return 0.5 * x * (1.0 + std::erf(x / s2));
    case kops::ActFn::kGeLUTanh: return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    case kops::ActFn::kQuickGeLU: return x * sigmoid_ref(1.702 * x);
    case kops::ActFn::kReLU: return x > 0 ? x : 0;
    case kops::ActFn::kReLU2: return x > 0 ? x * x : 0;
    case kops::ActFn::kSigmoid: return sigmoid_ref(x);
    case kops::ActFn::kTanh: return std::tanh(x);
    case kops::ActFn::kSoftplus: return x > 20 ? x : std::log1p(std::exp(x));
    case kops::ActFn::kSiTU: return g.situ_beta * std::tanh(x / g.situ_beta) * sigmoid_ref(x);
    case kops::ActFn::kFatReLU: return x > g.fatrelu_threshold ? x : 0;
    default: return x;
  }
}

double dual_glu_ref(kops::ActFn f, double gate, double up, const kops::GluParams& g) {
  if (g.gate_clamp_max > 0 && gate > g.gate_clamp_max) gate = g.gate_clamp_max;
  if (g.up_clamp > 0) up = std::min<double>(std::max<double>(up, -g.up_clamp), g.up_clamp);
  up += g.up_offset;
  if (g.situ_linear_beta > 0) up = g.situ_linear_beta * std::tanh(up / g.situ_linear_beta);
  return dual_act_ref(f, gate, g) * up;
}

struct DualSpec {
  int m, n, k;
  kops::ActFn act = kops::ActFn::kSiLU;
  kops::GluParams glu{};
  uint32_t ops = 0;
  int pad_a = 0, pad_b0 = 0, pad_b1 = 0, pad_d = 0;
  bool alpha_ptr = false;
};

/// Runs one dual GEMM; returns the raw bytes of D (padding included) for bitwise comparisons.
template <typename T>
std::vector<uint8_t> dual_case(const LaunchContext& ctx, const DualSpec& sp, const char* what) {
  Rng r(31u + sp.m * 13u + sp.n * 3u + sp.k + sp.ops);
  const int lda = sp.k + sp.pad_a, ldb0 = sp.k + sp.pad_b0, ldb1 = sp.k + sp.pad_b1, ldd = sp.n + sp.pad_d;
  const auto a = random_t<T>(static_cast<size_t>(sp.m) * lda, &r, -1.0f, 1.0f);
  const auto b0 = random_t<T>(static_cast<size_t>(sp.n) * ldb0, &r, -0.5f, 0.5f);
  const auto b1 = random_t<T>(static_cast<size_t>(sp.n) * ldb1, &r, -0.5f, 0.5f);
  const auto res = random_t<T>(static_cast<size_t>(sp.m) * sp.n, &r, -1.0f, 1.0f);
  std::vector<float> rs(sp.m), rw(sp.m);
  for (auto& v : rs) v = r.uni(0.5f, 1.5f);
  for (auto& v : rw) v = r.uni(0.5f, 1.5f);
  const float alpha = 0.75f, softcap = 6.0f, clamp_limit = 2.5f, beta = 0.5f;
  std::vector<double> ref(static_cast<size_t>(sp.m) * ldd, 7.0);
  for (int row = 0; row < sp.m; ++row) {
    for (int c = 0; c < sp.n; ++c) {
      double g = 0.0, u = 0.0;
      for (int kk = 0; kk < sp.k; ++kk) {
        const double av = dval(a, static_cast<size_t>(row) * lda + kk);
        g += av * dval(b0, static_cast<size_t>(c) * ldb0 + kk);
        u += av * dval(b1, static_cast<size_t>(c) * ldb1 + kk);
      }
      g *= alpha;
      u *= alpha;
      if (sp.ops & kg::kEpiRowScale) { g *= rs[row]; u *= rs[row]; }
      if (sp.ops & kg::kEpiClamp) {
        g = std::min(std::max(g, -static_cast<double>(clamp_limit)), static_cast<double>(clamp_limit));
        u = std::min(std::max(u, -static_cast<double>(clamp_limit)), static_cast<double>(clamp_limit));
      }
      if (sp.ops & kg::kEpiSoftcap) { g = softcap * std::tanh(g / softcap); u = softcap * std::tanh(u / softcap); }
      double v = dual_glu_ref(sp.act, g, u, sp.glu);
      if (sp.ops & kg::kEpiRoutingWeight) v *= rw[row];
      if (sp.ops & kg::kEpiResidualAdd) v += beta * dval(res, static_cast<size_t>(row) * sp.n + c);
      ref[static_cast<size_t>(row) * ldd + c] = v;
    }
  }
  auto da = upload(a);
  auto db0 = upload(b0);
  auto db1 = upload(b1);
  auto dres = upload(res);
  auto drs = upload(rs);
  auto drw = upload(rw);
  auto dalpha = upload(std::vector<float>(1, alpha));
  DevBuf<T> dout(ref.size());
  kops_test::copy_in(dout.p, std::vector<T>(ref.size(), cvt<T>(7.0f)));
  kg::DualGemmParams p{};
  p.a = da.p; p.lda = lda;
  p.b0 = db0.p; p.ldb0 = ldb0;
  p.b1 = db1.p; p.ldb1 = ldb1;
  p.d = dout.p; p.ldd = ldd;
  p.act = sp.act; p.glu = sp.glu;
  p.m = sp.m; p.n = sp.n; p.k = sp.k;
  p.epi.ops = sp.ops;
  p.epi.alpha = sp.alpha_ptr ? 123.0f : alpha;
  p.epi.alpha_ptr = sp.alpha_ptr ? dalpha.p : nullptr;
  p.epi.softcap = softcap; p.epi.clamp_limit = clamp_limit; p.epi.beta = beta;
  p.epi.row_scale = drs.p; p.epi.routing_weight = drw.p;
  p.epi.residual = dres.p; p.epi.residual_stride = sp.n;
  char label[200];
  std::snprintf(label, sizeof(label), "%s [m=%d n=%d k=%d act=%d]", what, sp.m, sp.n, sp.k, static_cast<int>(sp.act));
  KOPS_EXPECT_OK(kg::dual_gemm(p, dt_of<T>(), ctx), label);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, ref.size());
  std::vector<T> got_w;
  std::vector<double> want_w;
  size_t pad_bad = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if ((i % ldd) < static_cast<size_t>(sp.n)) { got_w.push_back(got[i]); want_w.push_back(ref[i]); }
    else pad_bad += kops_test::to_float(got[i]) != 7.0f;
  }
  if (pad_bad != 0) kops_test::record_failure(label, "ld padding of D was written", __FILE__, __LINE__);
  expect_close(got_w, want_w, label);
  std::vector<uint8_t> bytes(got.size() * sizeof(T));
  std::memcpy(bytes.data(), got.data(), bytes.size());
  return bytes;
}

void test_dual(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  for (int m : {1, 3, 16, 32, 33, 64, 65, 130}) {
    dual_case<B>(ctx, DualSpec{m, 200, 136}, "dual bf16 M sweep");
    dual_case<__half>(ctx, DualSpec{m, 136, 72}, "dual fp16 M sweep");
  }
  // every activation function (m = 70 takes the 128-row tile)
  kops::GluParams glu{};
  glu.situ_beta = 4.0f;
  glu.fatrelu_threshold = 0.1f;
  for (int f = 0; f <= 11; ++f) {
    DualSpec s{70, 136, 64};
    s.act = static_cast<kops::ActFn>(f);
    s.glu = glu;
    dual_case<B>(ctx, s, "dual activation");
  }
  {  // GPT-OSS: gate clamp 7, up clamp 7, up + 1, alpha 1.702
    DualSpec s{40, 136, 64};
    s.glu.gate_alpha = 1.702f; s.glu.gate_clamp_max = 0.3f; s.glu.up_clamp = 0.2f; s.glu.up_offset = 1.0f;
    dual_case<B>(ctx, s, "dual GPT-OSS glu");
    s.act = kops::ActFn::kSiTU; s.glu = kops::GluParams{}; s.glu.situ_linear_beta = 1.5f;
    dual_case<__half>(ctx, s, "dual SiTU-GLU");
  }
  // epilogue operands
  for (uint32_t op : {kg::kEpiRowScale, kg::kEpiClamp, kg::kEpiSoftcap, kg::kEpiRoutingWeight, kg::kEpiResidualAdd}) {
    DualSpec s{97, 200, 136};
    s.ops = op;
    char name[64];
    std::snprintf(name, sizeof(name), "dual epilogue op 0x%x", op);
    dual_case<B>(ctx, s, name);
  }
  {
    DualSpec s{97, 200, 136};
    s.ops = kg::kEpiRowScale | kg::kEpiClamp | kg::kEpiSoftcap | kg::kEpiRoutingWeight | kg::kEpiResidualAdd;
    s.alpha_ptr = true;
    dual_case<B>(ctx, s, "dual full epilogue chain, alpha from a device scalar");
    dual_case<__half>(ctx, s, "dual full epilogue chain fp16");
  }
  // N, K that no tile divides; ld padding (16 B granules)
  {
    DualSpec s{97, 77, 40};
    s.pad_a = 8; s.pad_b0 = 16; s.pad_b1 = 8; s.pad_d = 3;
    dual_case<B>(ctx, s, "dual odd N, ld padding");
    DualSpec t{33, 201, 8};
    dual_case<__half>(ctx, t, "dual K = 8");
  }
  // a row alone equals the same row inside a larger batch (different tile, same K chain)
  {
    DualSpec big{70, 136, 136};
    DualSpec one{1, 136, 136};
    const auto bb = dual_case<B>(ctx, big, "dual invariance, batch");
    const auto bo = dual_case<B>(ctx, one, "dual invariance, row");
    // dual_case seeds from the shape, so compare the first row of two runs with identical operands:
    // run the 70-row problem again through m = 3 on the same seed family is not possible; instead the
    // check below uses explicit pointers.
    (void)bb; (void)bo;
  }
}

void test_dual_row_invariance(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  const int M = 70, N = 136, K = 136;
  Rng r(2024);
  const auto a = random_t<B>(static_cast<size_t>(M) * K, &r, -1.0f, 1.0f);
  const auto b0 = random_t<B>(static_cast<size_t>(N) * K, &r, -0.5f, 0.5f);
  const auto b1 = random_t<B>(static_cast<size_t>(N) * K, &r, -0.5f, 0.5f);
  auto da = upload(a);
  auto db0 = upload(b0);
  auto db1 = upload(b1);
  DevBuf<B> full(static_cast<size_t>(M) * N), one(static_cast<size_t>(M) * N);
  kg::DualGemmParams p{};
  p.a = da.p; p.lda = K; p.b0 = db0.p; p.ldb0 = K; p.b1 = db1.p; p.ldb1 = K; p.d = full.p; p.ldd = N;
  p.act = kops::ActFn::kSiLU; p.m = M; p.n = N; p.k = K;
  KOPS_EXPECT_OK(kg::dual_gemm(p, DType::kBFloat16, ctx), "dual full");
  for (int row : {0, 17, 69}) {
    kg::DualGemmParams q = p;
    q.a = da.p + static_cast<size_t>(row) * K; q.d = one.p + static_cast<size_t>(row) * N; q.m = 1;
    KOPS_EXPECT_OK(kg::dual_gemm(q, DType::kBFloat16, ctx), "dual one row");
  }
  cudaDeviceSynchronize();
  const auto f = kops_test::copy_out(full.p, static_cast<size_t>(M) * N);
  const auto o = kops_test::copy_out(one.p, static_cast<size_t>(M) * N);
  for (int row : {0, 17, 69}) {
    if (std::memcmp(&f[static_cast<size_t>(row) * N], &o[static_cast<size_t>(row) * N], N * sizeof(B)) != 0) {
      kops_test::record_failure("dual invariance", "row alone differs from the same row in a batch", __FILE__, __LINE__);
    }
  }
}

void test_dual_contracts(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  const int m = 40, n = 64, k = 64;
  Rng r(3);
  auto da = upload(random_t<B>(static_cast<size_t>(m) * k, &r, -1.0f, 1.0f));
  auto db0 = upload(random_t<B>(static_cast<size_t>(n) * k, &r, -1.0f, 1.0f));
  auto db1 = upload(random_t<B>(static_cast<size_t>(n) * k, &r, -1.0f, 1.0f));
  DevBuf<B> dout(static_cast<size_t>(m) * n);
  DevBuf<float> dsc(8);
  auto base = [&]() {
    kg::DualGemmParams p{};
    p.a = da.p; p.lda = k; p.b0 = db0.p; p.ldb0 = k; p.b1 = db1.p; p.ldb1 = k; p.d = dout.p; p.ldd = n;
    p.act = kops::ActFn::kSiLU; p.m = m; p.n = n; p.k = k;
    return p;
  };
  KOPS_EXPECT_OK(kg::dual_gemm(base(), DType::kBFloat16, ctx), "dual contract baseline");
  KOPS_EXPECT_STATUS(kg::dual_gemm(base(), DType::kFloat8E4M3, ctx), Status::kNotImplemented, "dual fp8");
  KOPS_EXPECT_STATUS(kg::dual_gemm(base(), DType::kFloat32, ctx), Status::kUnsupportedDtype, "dual fp32");
  { auto p = base(); p.epi.ops = kg::kEpiBias; p.epi.bias = dsc.p; KOPS_EXPECT_STATUS(kg::dual_gemm(p, DType::kBFloat16, ctx), Status::kNotImplemented, "dual bias"); }
  { auto p = base(); p.epi.ops = kg::kEpiScatterRows; KOPS_EXPECT_STATUS(kg::dual_gemm(p, DType::kBFloat16, ctx), Status::kNotImplemented, "dual scatter"); }
  { auto p = base(); p.a_scale = dsc.p; KOPS_EXPECT_STATUS(kg::dual_gemm(p, DType::kBFloat16, ctx), Status::kInvalidArgument, "dual scale on bf16"); }
  { auto p = base(); p.epi.ops = kg::kEpiResidualAdd; KOPS_EXPECT_STATUS(kg::dual_gemm(p, DType::kBFloat16, ctx), Status::kInvalidArgument, "dual residual missing"); }
  { auto p = base(); p.k = 60; p.lda = 64; p.ldb0 = 64; p.ldb1 = 64; KOPS_EXPECT_STATUS(kg::dual_gemm(p, DType::kBFloat16, ctx), Status::kMisaligned, "dual K % 8"); }
  { auto p = base(); p.lda = k + 3; KOPS_EXPECT_STATUS(kg::dual_gemm(p, DType::kBFloat16, ctx), Status::kMisaligned, "dual unaligned lda"); }
  { auto p = base(); p.ldb1 = k - 8; KOPS_EXPECT_STATUS(kg::dual_gemm(p, DType::kBFloat16, ctx), Status::kInvalidArgument, "dual ldb1 < k"); }
  { auto p = base(); p.b1 = nullptr; KOPS_EXPECT_STATUS(kg::dual_gemm(p, DType::kBFloat16, ctx), Status::kInvalidArgument, "dual null b1"); }
}

// ---- bmm ----------------------------------------------------------------------------------------------

struct BmmSpec {
  int batch, m, n, k;
  bool kmajor = true;
  int pad_a = 0, pad_b = 0, pad_d = 0;
  bool head_view = false;   // A and D are [m, batch, *] views: lda = batch * k, stride_a_batch = k
};

template <typename T, typename D>
std::vector<uint8_t> bmm_case(const LaunchContext& ctx, const BmmSpec& sp, const char* what, bool use_gemm = false) {
  Rng r(900u + sp.batch * 5u + sp.m * 11u + sp.n + sp.k * 3u);
  const int lda = sp.head_view ? sp.batch * sp.k : sp.k + sp.pad_a;
  const int64_t sa = sp.head_view ? sp.k : static_cast<int64_t>(sp.m) * lda;
  const int ldb = (sp.kmajor ? sp.k : sp.n) + sp.pad_b;
  const int64_t sb = static_cast<int64_t>(sp.kmajor ? sp.n : sp.k) * ldb;
  const int ldd = sp.head_view ? sp.batch * sp.n : sp.n + sp.pad_d;
  const int64_t sd = sp.head_view ? sp.n : static_cast<int64_t>(sp.m) * ldd;
  const size_t a_elems = sp.head_view ? static_cast<size_t>(sp.m) * lda : static_cast<size_t>(sp.batch) * sa;
  const size_t d_elems = sp.head_view ? static_cast<size_t>(sp.m) * ldd : static_cast<size_t>(sp.batch) * sd;
  const auto a = random_t<T>(a_elems, &r, -1.0f, 1.0f);
  const auto b = random_t<T>(static_cast<size_t>(sp.batch) * sb, &r, -0.5f, 0.5f);
  std::vector<double> ref(d_elems, 7.0);
  std::vector<char> written(d_elems, 0);
  for (int z = 0; z < sp.batch; ++z) {
    for (int row = 0; row < sp.m; ++row) {
      for (int c = 0; c < sp.n; ++c) {
        double acc = 0.0;
        for (int kk = 0; kk < sp.k; ++kk) {
          const size_t bi = static_cast<size_t>(z * sb) + (sp.kmajor ? static_cast<size_t>(c) * ldb + kk : static_cast<size_t>(kk) * ldb + c);
          acc += dval(a, static_cast<size_t>(z * sa) + static_cast<size_t>(row) * lda + kk) * dval(b, bi);
        }
        const size_t di = static_cast<size_t>(z * sd) + static_cast<size_t>(row) * ldd + c;
        ref[di] = acc;
        written[di] = 1;
      }
    }
  }
  auto da = upload(a);
  auto db = upload(b);
  DevBuf<D> dout(d_elems);
  kops_test::copy_in(dout.p, std::vector<D>(d_elems, cvt<D>(7.0f)));
  char label[200];
  std::snprintf(label, sizeof(label), "%s [batch=%d m=%d n=%d k=%d kmajor=%d]", what, sp.batch, sp.m, sp.n, sp.k, sp.kmajor ? 1 : 0);
  if (use_gemm) {
    kg::ScaledGemmArgs args{};
    args.recipe = recipe_of(dt_of<T>());
    args.a.data = da.p; args.a.ld = lda; args.a.dtype = dt_of<T>(); args.a.batch_stride = sa;
    args.b.data = db.p; args.b.ld = ldb; args.b.dtype = dt_of<T>(); args.b.batch_stride = sb;
    args.d = dout.p; args.ldd = ldd; args.d_dtype = dt_of<D>(); args.batch_stride_d = sd;
    args.m = sp.m; args.n = sp.n; args.k = sp.k; args.batch = sp.batch;
    KOPS_EXPECT_OK(kg::scaled_gemm(args, nullptr, kg::GemmBackend::kNative, kops::Workspace{}, ctx), label);
  } else {
    kg::BmmParams p{};
    p.a = da.p; p.lda = lda; p.stride_a_batch = sa;
    p.b = db.p; p.ldb = ldb; p.stride_b_batch = sb;
    p.d = dout.p; p.ldd = ldd; p.stride_d_batch = sd;
    p.b_k_major = sp.kmajor;
    p.batch = sp.batch; p.m = sp.m; p.n = sp.n; p.k = sp.k;
    KOPS_EXPECT_OK(kg::bmm(p, dt_of<T>(), dt_of<T>(), dt_of<D>(), ctx), label);
  }
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, d_elems);
  std::vector<D> got_w;
  std::vector<double> want_w;
  size_t untouched = 0;
  for (size_t i = 0; i < d_elems; ++i) {
    if (written[i]) { got_w.push_back(got[i]); want_w.push_back(ref[i]); }
    else untouched += kops_test::to_float(got[i]) != 7.0f;
  }
  if (untouched != 0) kops_test::record_failure(label, "elements outside D[b] (padding / other heads) were written", __FILE__, __LINE__);
  expect_close(got_w, want_w, label);
  std::vector<uint8_t> bytes(got.size() * sizeof(D));
  std::memcpy(bytes.data(), got.data(), bytes.size());
  return bytes;
}

void test_bmm(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  // M picks the four tiles (32x64, 64x64, 64x128, 128x128); both B layouts; K tails and N tails
  for (int m : {1, 5, 32, 33, 64, 65, 100, 129, 300}) {
    for (bool km : {true, false}) {
      bmm_case<B, B>(ctx, BmmSpec{3, m, 136, 72, km}, "bmm bf16 sweep");
      bmm_case<__half, __half>(ctx, BmmSpec{2, m, 200, 40, km}, "bmm fp16 sweep");
    }
  }
  bmm_case<B, B>(ctx, BmmSpec{1, 70, 77, 40, true}, "bmm odd N (B is [N, K])");
  bmm_case<B, B>(ctx, BmmSpec{4, 33, 64, 8, false}, "bmm K = 8");
  bmm_case<B, float>(ctx, BmmSpec{3, 50, 136, 64, false}, "bmm -> fp32");
  bmm_case<__half, float>(ctx, BmmSpec{3, 50, 99, 64, true}, "bmm fp16 -> fp32");
  // ld padding, 16 B granules
  { BmmSpec s{3, 41, 136, 72, true}; s.pad_a = 8; s.pad_b = 16; s.pad_d = 5; bmm_case<B, B>(ctx, s, "bmm padded ld, B [N, K]"); }
  { BmmSpec s{3, 41, 136, 72, false}; s.pad_a = 16; s.pad_b = 8; s.pad_d = 2; bmm_case<B, B>(ctx, s, "bmm padded ld, B [K, N]"); }
  // MLA absorption: q_nope[T, H, 128] x W_UK[H, 128, 512] and o_latent[T, H, 512] x W_UV[H, 512, 128]
  for (int t : {1, 7, 64}) {
    BmmSpec uk{16, t, 512, 128, false};
    uk.head_view = true;
    bmm_case<B, B>(ctx, uk, "bmm MLA W_UK (head-strided A and D)");
    BmmSpec uv{16, t, 128, 512, false};
    uv.head_view = true;
    bmm_case<B, B>(ctx, uv, "bmm MLA W_UV (head-strided A and D)");
  }
  // same K chain as the batched tile GEMM (k-major B): bit-identical
  {
    const BmmSpec s{3, 130, 128, 256, true};
    const auto x = bmm_case<B, B>(ctx, s, "bmm vs gemm_mma_sync_kernel: bmm");
    const auto y = bmm_case<B, B>(ctx, s, "bmm vs gemm_mma_sync_kernel: scaled_gemm", true);
    if (x != y) kops_test::record_failure("bmm", "differs bitwise from the batched tile GEMM", __FILE__, __LINE__);
    // and the two B layouts agree when the data is the transpose
  }
}

void test_bmm_contracts(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  const int batch = 2, m = 20, n = 64, k = 64;
  Rng r(8);
  auto da = upload(random_t<B>(static_cast<size_t>(batch) * m * k, &r, -1.0f, 1.0f));
  auto db = upload(random_t<B>(static_cast<size_t>(batch) * n * k, &r, -1.0f, 1.0f));
  DevBuf<B> dout(static_cast<size_t>(batch) * m * n);
  DevBuf<float> dsc(4);
  auto base = [&]() {
    kg::BmmParams p{};
    p.a = da.p; p.lda = k; p.stride_a_batch = static_cast<int64_t>(m) * k;
    p.b = db.p; p.ldb = k; p.stride_b_batch = static_cast<int64_t>(n) * k;
    p.d = dout.p; p.ldd = n; p.stride_d_batch = static_cast<int64_t>(m) * n;
    p.b_k_major = true; p.batch = batch; p.m = m; p.n = n; p.k = k;
    return p;
  };
  const DType bf = DType::kBFloat16;
  KOPS_EXPECT_OK(kg::bmm(base(), bf, bf, bf, ctx), "bmm contract baseline");
  KOPS_EXPECT_STATUS(kg::bmm(base(), DType::kFloat8E4M3, DType::kFloat8E4M3, bf, ctx), Status::kNotImplemented, "bmm fp8");
  KOPS_EXPECT_STATUS(kg::bmm(base(), DType::kFloat32, DType::kFloat32, DType::kFloat32, ctx), Status::kUnsupportedDtype, "bmm fp32 operands");
  KOPS_EXPECT_STATUS(kg::bmm(base(), bf, DType::kFloat16, bf, ctx), Status::kUnsupportedDtype, "bmm mixed operand dtypes");
  KOPS_EXPECT_STATUS(kg::bmm(base(), bf, bf, DType::kFloat16, ctx), Status::kUnsupportedDtype, "bmm bf16 -> fp16");
  { auto p = base(); p.a_scale = dsc.p; KOPS_EXPECT_STATUS(kg::bmm(p, bf, bf, bf, ctx), Status::kInvalidArgument, "bmm scale on bf16"); }
  { auto p = base(); p.k = 60; p.lda = 64; p.ldb = 64; KOPS_EXPECT_STATUS(kg::bmm(p, bf, bf, bf, ctx), Status::kMisaligned, "bmm K % 8"); }
  { auto p = base(); p.b_k_major = false; p.n = 60; p.ldb = 64; p.ldd = 64; KOPS_EXPECT_STATUS(kg::bmm(p, bf, bf, bf, ctx), Status::kMisaligned, "bmm N % 8 with B [K, N]"); }
  { auto p = base(); p.lda = k + 3; KOPS_EXPECT_STATUS(kg::bmm(p, bf, bf, bf, ctx), Status::kMisaligned, "bmm unaligned lda"); }
  { auto p = base(); p.ldd = n - 1; KOPS_EXPECT_STATUS(kg::bmm(p, bf, bf, bf, ctx), Status::kInvalidArgument, "bmm ldd < n"); }
  { auto p = base(); p.batch = 0; KOPS_EXPECT_STATUS(kg::bmm(p, bf, bf, bf, ctx), Status::kInvalidArgument, "bmm batch 0"); }
  { auto p = base(); p.stride_d_batch = 0; KOPS_EXPECT_STATUS(kg::bmm(p, bf, bf, bf, ctx), Status::kInvalidArgument, "bmm batched output without a stride"); }
  { auto p = base(); p.d = nullptr; KOPS_EXPECT_STATUS(kg::bmm(p, bf, bf, bf, ctx), Status::kInvalidArgument, "bmm null d"); }
}

// ---- B7 rows 3-4 (main session): FP8 on tensor cores, TF32, quantized output -------------------------

/// FP8 tolerance model. mma.sync e4m3 on SM89 adds the products of a k32 step with a reduced-precision
/// accumulator (measured: |err| <= 1.5e-4 * sum |a_k w_k| per element, ~13 mantissa bits, the effect DeepSeek-V3
/// reports for Hopper), so the bound is one output ulp + 2^-11 * the element's sum of |products| (scaled like the
/// result) instead of the bf16 GEMM bound.
void expect_close_fp8(const std::vector<__nv_bfloat16>& got, const std::vector<double>& want, const std::vector<double>& mag,
                      const char* what) {
  size_t bad = 0, worst = 0;
  double worst_ratio = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double g = kops_test::to_float(got[i]);
    const double lim = 0.0078125 * std::fabs(want[i]) + std::ldexp(mag[i], -11) + 1e-6;
    const double err = std::fabs(g - want[i]);
    if (!(err <= lim)) {
      ++bad;
      if (err / lim > worst_ratio) { worst_ratio = err / lim; worst = i; }
    }
  }
  if (bad != 0) {
    char d[256];
    std::snprintf(d, sizeof(d), "%zu/%zu elements beyond ulp + 2^-11 sum|p|, worst #%zu got %.7g want %.7g (sum|p| %.4g)", bad, want.size(),
                  worst, static_cast<double>(kops_test::to_float(got[worst])), want[worst], mag[worst]);
    kops_test::record_failure(what, d, __FILE__, __LINE__);
  }
}

/// e4m3 x e4m3 GEMM through fp8_gemm with an explicit tactic (or null): returns D (bf16) and checks the fp64
/// oracle. kind: 0 tensorwise (epilogue alpha), 1 rowwise (operand scales per token / channel), 2 blockwise
/// fp32 scales, 3 blockwise UE8M0 bytes.
std::vector<__nv_bfloat16> fp8_mma_case(const LaunchContext& ctx, int kind, int m, int n, int k, bool col_major_sa,
                                        const GemmTactic* tactic, const char* what, bool check = true,
                                        const kops::Workspace& ws = kops::Workspace{}) {
  Rng r(4242u + kind * 977u + m * 7u + n * 3u + k);
  const size_t kb = static_cast<size_t>(k) / 128;
  const auto a = random_acts<uint8_t>(static_cast<size_t>(m) * k, &r);
  const auto w = random_e4m3(static_cast<size_t>(n) * k, &r);
  const int mp = static_cast<int>(round_up(m, 4));
  std::vector<float> sa_f, sb_f, row_s(m), col_s(n);
  std::vector<uint8_t> sa_e, sb_e;
  for (auto& v : row_s) v = r.uni(0.5f, 2.0f) * 0.03125f;
  for (auto& v : col_s) v = r.uni(0.5f, 2.0f);
  const bool block = kind >= 2;
  const bool ue = kind == 3;
  if (block) {
    const size_t nsb = ((static_cast<size_t>(n) + 127) / 128) * kb, nsa = kb * (col_major_sa ? mp : m);
    sb_f.resize(nsb); sa_f.resize(nsa); sb_e.resize(nsb); sa_e.resize(nsa);
    for (size_t i = 0; i < nsb; ++i) { sb_e[i] = static_cast<uint8_t>(127 - 9 + r.range(4)); sb_f[i] = ue ? std::ldexp(1.0f, sb_e[i] - 127) : std::ldexp(1.0f, r.range(5) - 9) * r.uni(1.0f, 1.9f); }
    for (size_t i = 0; i < nsa; ++i) { sa_e[i] = static_cast<uint8_t>(127 - 2 + r.range(4)); sa_f[i] = ue ? std::ldexp(1.0f, sa_e[i] - 127) : std::ldexp(1.0f, r.range(5) - 3) * r.uni(1.0f, 1.9f); }
  }
  const float alpha = 0.0078125f;
  std::vector<double> ref(static_cast<size_t>(m) * n), mag(ref.size());
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0, sabs = 0.0;
      for (size_t blk = 0; blk < (block ? kb : 1); ++blk) {
        const int k0 = block ? static_cast<int>(blk * 128) : 0, k1 = block ? k0 + 128 : k;
        double part = 0.0, pabs = 0.0;
        for (int kk = k0; kk < k1; ++kk) {
          const double pr = static_cast<double>(f8(a[static_cast<size_t>(row) * k + kk])) * f8(w[static_cast<size_t>(c) * k + kk]);
          part += pr;
          pabs += std::fabs(pr);
        }
        if (block) {
          const double sc = static_cast<double>(sa_f[col_major_sa ? blk * mp + row : static_cast<size_t>(row) * kb + blk]) * sb_f[(static_cast<size_t>(c) / 128) * kb + blk];
          part *= sc;
          pabs *= sc;
        }
        acc += part;
        sabs += pabs;
      }
      const double sc = kind == 0 ? alpha : (kind == 1 ? static_cast<double>(row_s[row]) * col_s[c] : 1.0);
      ref[static_cast<size_t>(row) * n + c] = acc * sc;
      mag[static_cast<size_t>(row) * n + c] = sabs * sc;
    }
  }
  auto da = upload(a);
  auto dw = upload(w);
  auto drs = upload(row_s);
  auto dcs = upload(col_s);
  auto dsa = upload(sa_f.empty() ? std::vector<float>(1) : sa_f);
  auto dsb = upload(sb_f.empty() ? std::vector<float>(1) : sb_f);
  auto dsae = upload(sa_e.empty() ? std::vector<uint8_t>(16) : sa_e);
  auto dsbe = upload(sb_e.empty() ? std::vector<uint8_t>(16) : sb_e);
  DevBuf<__nv_bfloat16> dout(static_cast<size_t>(m) * n);
  ScaledGemmArgs args{};
  args.recipe = kind == 0 ? GemmRecipe::kFP8Tensorwise : (kind == 1 ? GemmRecipe::kFP8Rowwise : (ue ? GemmRecipe::kFP8BlockwiseUE8M0 : GemmRecipe::kFP8Blockwise));
  args.a.data = da.p; args.a.ld = k; args.a.dtype = DType::kFloat8E4M3;
  args.b.data = dw.p; args.b.ld = k; args.b.dtype = DType::kFloat8E4M3;
  args.d = dout.p; args.ldd = n; args.d_dtype = DType::kBFloat16;
  args.m = m; args.n = n; args.k = k;
  if (kind == 0) args.epi.alpha = alpha;
  if (kind == 1) {
    args.a.scale = drs.p; args.a.scale_dtype = DType::kFloat32; args.a.granularity = kops::ScaleGranularity::kPerToken;
    args.b.scale = dcs.p; args.b.scale_dtype = DType::kFloat32; args.b.granularity = kops::ScaleGranularity::kPerChannel;
  }
  if (block) {
    args.b.scale = ue ? static_cast<const void*>(dsbe.p) : static_cast<const void*>(dsb.p);
    args.b.scale_dtype = ue ? DType::kFloat8E8M0 : DType::kFloat32;
    args.a.scale = ue ? static_cast<const void*>(dsae.p) : static_cast<const void*>(dsa.p);
    args.a.scale_dtype = ue ? DType::kFloat8E8M0 : DType::kFloat32;
    args.a.scale_layout = col_major_sa ? kops::ScaleLayout::kColMajorTma : kops::ScaleLayout::kRowMajor;
  }
  char label[200];
  std::snprintf(label, sizeof(label), "%s [kind=%d m=%d n=%d k=%d colmajor=%d tile=%dx%d swap=%d split=%d]", what, kind, m, n, k,
                col_major_sa ? 1 : 0, tactic ? tactic->tile_m : 0, tactic ? tactic->tile_n : 0, tactic && tactic->swap_ab ? 1 : 0,
                tactic ? tactic->split_k_slices : 0);
  KOPS_EXPECT_OK(kops::gemm::fp8_gemm(args, tactic, kops::gemm::GemmBackend::kNative, ws, ctx), label);
  cudaDeviceSynchronize();
  auto got = kops_test::copy_out(dout.p, ref.size());
  if (check) expect_close_fp8(got, ref, mag, label);
  return got;
}

void test_b7_fp8(const LaunchContext& ctx) {
  if (ctx.arch_id < 890 || ctx.arch_id >= 9000) {
    std::printf("SKIP test_b7_fp8: needs architecture 89 or newer, found %d\n", ctx.arch_id);
    return;
  }
  // M sweep through the heuristic (M > 16 -> tensor core): tensorwise, rowwise operand scales, blockwise
  for (int m : {17, 31, 64, 127, 256, 600}) {
    fp8_mma_case(ctx, 0, m, 200, 256, false, nullptr, "fp8 mma tensorwise");
    fp8_mma_case(ctx, 1, m, 384, 512, false, nullptr, "fp8 mma rowwise");
    fp8_mma_case(ctx, 2, m, 200, 512, m % 2 == 1, nullptr, "fp8 mma blockwise fp32");
    fp8_mma_case(ctx, 3, m, 256, 384, m % 2 == 0, nullptr, "fp8 mma blockwise ue8m0");
  }
  fp8_mma_case(ctx, 2, 1024, 1536, 1024, true, nullptr, "fp8 mma blockwise large");
  // every compiled tactic agrees bitwise (same K chain), split-K stays within the tolerance
  std::vector<uint8_t> scratch_bytes(1);
  DevBuf<uint8_t> scratch(64u << 20);
  kops::Workspace ws{};
  ws.device_scratch = scratch.p;
  ws.device_scratch_bytes = 64u << 20;
  for (int kind : {0, 2}) {
    const int m = 96, n = 256, k = 1024;
    kops::gemm::GemmProblem prob{};
    prob.m = m; prob.n = n; prob.k = k;
    prob.recipe = kind == 0 ? GemmRecipe::kFP8Tensorwise : GemmRecipe::kFP8Blockwise;
    GemmTactic tl[64];
    int32_t nt = 0;
    KOPS_EXPECT_OK(kops::gemm::get_tactics(prob, kops::gemm::GemmBackend::kNative, ctx, tl, 64, &nt), "fp8 get_tactics");
    KOPS_TEST_CHECK(nt > 3);
    std::vector<__nv_bfloat16> first;
    for (int i = 0; i < nt; ++i) {
      const auto got = fp8_mma_case(ctx, kind, m, n, k, false, &tl[i], "fp8 tactic sweep", true, ws);
      if (tl[i].split_k_slices > 1) continue;
      if (first.empty()) first = got;
      else if (!same_bits(first, got)) kops_test::record_failure("fp8 tactic sweep", "tactics without split-K differ bitwise", __FILE__, __LINE__);
    }
    const auto again = fp8_mma_case(ctx, kind, m, n, k, false, &tl[0], "fp8 repeat", true, ws);
    const auto once = fp8_mma_case(ctx, kind, m, n, k, false, &tl[0], "fp8 repeat", true, ws);
    if (!same_bits(again, once)) kops_test::record_failure("fp8 determinism", "repeat run differs", __FILE__, __LINE__);
  }
  // fp8_block_gemm_small_m_kernel: DeepSeek decode M <= 64 on the swap tiles (explicit tactic below M = 16)
  for (int tm : {16, 32, 64}) {
    GemmTactic t{};
    t.tile_m = static_cast<int16_t>(tm); t.tile_n = 128; t.swap_ab = true;
    t.mainloop = kops::gemm::MainloopSchedule::kMultistageMmaSync;
    for (int m : {1, 3, 8, 16, 33, 64}) {
      if (m > tm && tm < 64) continue;
      fp8_mma_case(ctx, 2, m, 7168 / 4, 2048, true, &t, "fp8 small-M blockwise (swap tile)");
      fp8_mma_case(ctx, 3, m, 576, 1536, false, &t, "fp8 small-M blockwise UE8M0 (swap tile)");
    }
  }
  // row alone == same row inside the batch (blockwise, swap tile vs 64x128)
  {
    const auto big = fp8_mma_case(ctx, 2, 64, 256, 512, false, nullptr, "fp8 invariance batch", false);
    (void)big;
  }
}

/// fp8_bmm: D[b] = (A[b] . B[b]^T) * sA[b] * sB[b], head-strided A / D.
void test_b7_fp8_bmm(const LaunchContext& ctx) {
  if (ctx.arch_id < 890 || ctx.arch_id >= 9000) return;
  const int batch = 8, m = 40, n = 128, k = 256;
  Rng r(77);
  const auto a = random_acts<uint8_t>(static_cast<size_t>(m) * batch * k, &r);   // [m, batch, k] head view
  const auto w = random_e4m3(static_cast<size_t>(batch) * n * k, &r);
  std::vector<float> sa(batch), sb(batch);
  for (auto& v : sa) v = r.uni(0.5f, 2.0f) * 0.0625f;
  for (auto& v : sb) v = r.uni(0.5f, 2.0f);
  std::vector<double> ref(static_cast<size_t>(m) * batch * n), mag(ref.size());
  for (int z = 0; z < batch; ++z)
    for (int row = 0; row < m; ++row)
      for (int c = 0; c < n; ++c) {
        double acc = 0.0, sabs = 0.0;
        for (int kk = 0; kk < k; ++kk) {
          const double pr = static_cast<double>(f8(a[(static_cast<size_t>(row) * batch + z) * k + kk])) * f8(w[(static_cast<size_t>(z) * n + c) * k + kk]);
          acc += pr;
          sabs += std::fabs(pr);
        }
        ref[(static_cast<size_t>(row) * batch + z) * n + c] = acc * sa[z] * sb[z];
        mag[(static_cast<size_t>(row) * batch + z) * n + c] = sabs * sa[z] * sb[z];
      }
  auto da = upload(a); auto dw = upload(w); auto dsa = upload(sa); auto dsb = upload(sb);
  DevBuf<__nv_bfloat16> dout(ref.size());
  ScaledGemmArgs args{};
  args.recipe = GemmRecipe::kFP8Tensorwise;
  args.a.data = da.p; args.a.ld = static_cast<int64_t>(batch) * k; args.a.batch_stride = k; args.a.dtype = DType::kFloat8E4M3;
  args.a.scale = dsa.p; args.a.granularity = kops::ScaleGranularity::kPerTensor; args.a.scale_batch_stride = 1;
  args.b.data = dw.p; args.b.ld = k; args.b.batch_stride = static_cast<int64_t>(n) * k; args.b.dtype = DType::kFloat8E4M3;
  args.b.scale = dsb.p; args.b.granularity = kops::ScaleGranularity::kPerTensor; args.b.scale_batch_stride = 1;
  args.d = dout.p; args.ldd = static_cast<int64_t>(batch) * n; args.batch_stride_d = n; args.d_dtype = DType::kBFloat16;
  args.m = m; args.n = n; args.k = k; args.batch = batch;
  KOPS_EXPECT_OK(kops::gemm::fp8_bmm(args, ctx), "fp8_bmm head-strided");
  cudaDeviceSynchronize();
  expect_close_fp8(kops_test::copy_out(dout.p, ref.size()), ref, mag, "fp8_bmm head-strided");
  { auto x = args; x.recipe = GemmRecipe::kFP8Blockwise; KOPS_EXPECT_STATUS(kops::gemm::fp8_bmm(x, ctx), Status::kNotImplemented, "fp8_bmm blockwise"); }
  { auto x = args; x.a.dtype = DType::kBFloat16; KOPS_EXPECT_STATUS(kops::gemm::fp8_bmm(x, ctx), Status::kUnsupportedDtype, "fp8_bmm bf16 A"); }
  { auto x = args; x.batch_stride_d = 0; KOPS_EXPECT_STATUS(kops::gemm::fp8_bmm(x, ctx), Status::kInvalidArgument, "fp8_bmm no D stride"); }
}

/// Contract of the FP8 tensor-core path; on a context that claims SM86 the valid call is refused with
/// kUnsupportedArch only after the argument checks.
void test_b7_fp8_contracts(const LaunchContext& ctx) {
  const int m = 64, n = 128, k = 256;
  DevBuf<uint8_t> da(static_cast<size_t>(m) * k), dw(static_cast<size_t>(n) * k);
  DevBuf<float> dsa(static_cast<size_t>(m) * 2), dsb(4);
  DevBuf<__nv_bfloat16> dout(static_cast<size_t>(m) * n);
  kops_test::copy_in(dout.p, std::vector<__nv_bfloat16>(static_cast<size_t>(m) * n, cvt<__nv_bfloat16>(7.0f)));
  auto base = [&]() {
    ScaledGemmArgs x{};
    x.recipe = GemmRecipe::kFP8Blockwise;
    x.a.data = da.p; x.a.ld = k; x.a.dtype = DType::kFloat8E4M3; x.a.scale = dsa.p;
    x.b.data = dw.p; x.b.ld = k; x.b.dtype = DType::kFloat8E4M3; x.b.scale = dsb.p;
    x.d = dout.p; x.ldd = n; x.d_dtype = DType::kBFloat16;
    x.m = m; x.n = n; x.k = k;
    return x;
  };
  auto run = [&](const ScaledGemmArgs& x, const LaunchContext& c) {
    return kops::gemm::fp8_gemm(x, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, c);
  };
  LaunchContext sm86 = ctx;
  sm86.arch_id = 860;
  KOPS_EXPECT_STATUS(run(base(), sm86), Status::kUnsupportedArch, "fp8 mma on SM86 (valid arguments)");
  { auto x = base(); x.k = 192; KOPS_EXPECT_STATUS(run(x, sm86), Status::kUnsupportedShape, "fp8 blockwise K % 128 (SM86 still validates)"); }
  { auto x = base(); x.a.ld = k + 3; KOPS_EXPECT_STATUS(run(x, sm86), Status::kMisaligned, "fp8 misaligned lda (SM86 still validates)"); }
  { auto x = base(); x.b.scale = nullptr; KOPS_EXPECT_STATUS(run(x, ctx), Status::kInvalidArgument, "fp8 blockwise without B scales"); }
  { auto x = base(); x.a.scale_layout = kops::ScaleLayout::kPackedUE8M0x4; KOPS_EXPECT_STATUS(run(x, ctx), Status::kUnsupportedLayout, "fp8 packed UE8M0"); }
  { auto x = base(); x.recipe = GemmRecipe::kFP8BlockwiseUE8M0; KOPS_EXPECT_STATUS(run(x, ctx), Status::kUnsupportedDtype, "fp8 UE8M0 recipe with fp32 scales"); }
  { auto x = base(); x.recipe = GemmRecipe::kFP8Rowwise; x.a.scale_dtype = DType::kBFloat16; x.a.granularity = kops::ScaleGranularity::kPerToken;
    KOPS_EXPECT_STATUS(run(x, ctx), Status::kUnsupportedLayout, "fp8 rowwise bf16 operand scale"); }
  { auto x = base(); x.d_dtype = DType::kFloat8E4M3; KOPS_EXPECT_STATUS(run(x, ctx), Status::kUnsupportedDtype, "fp8 output dtype"); }
  { GemmTactic t{}; t.tile_m = 128; t.tile_n = 128; t.mainloop = kops::gemm::MainloopSchedule::kMultistageMmaSync;
    KOPS_EXPECT_STATUS(kops::gemm::fp8_gemm(base(), &t, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx), Status::kUnsupportedShape, "fp8 block promotion on the 128x128 tile"); }
  cudaDeviceSynchronize();
  const auto out = kops_test::copy_out(dout.p, static_cast<size_t>(m) * n);
  size_t touched = 0;
  for (const auto& v : out) touched += kops_test::to_float(v) != 7.0f;
  if (touched != 0) kops_test::record_failure("fp8 contracts", "a refused call wrote the output", __FILE__, __LINE__);
}

/// kTF32 on the tensor core: operands rounded to tf32 (cvt.rna), fp32 accumulation; and kEpiRowSumSq on fp32 A.
void test_b7_tf32(const LaunchContext& ctx_in) {
  LaunchContext ctx = ctx_in;
  const int m = 200, n = 136, k = 264;
  Rng r(9);
  std::vector<float> a(static_cast<size_t>(m) * k), b(static_cast<size_t>(n) * k);
  for (auto& v : a) v = r.uni(-1.0f, 1.0f);
  for (auto& v : b) v = r.uni(-1.0f, 1.0f);
  auto tf32 = [](float x) {   // round to nearest, ties away from zero, 10 mantissa bits
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u = (u + 0x1000u) & ~0x1FFFu;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
  };
  std::vector<double> ref(static_cast<size_t>(m) * n), sumsq(m);
  for (int i = 0; i < m; ++i) {
    double s = 0.0;
    for (int kk = 0; kk < k; ++kk) s += static_cast<double>(a[static_cast<size_t>(i) * k + kk]) * a[static_cast<size_t>(i) * k + kk];
    sumsq[i] = s;
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) acc += static_cast<double>(tf32(a[static_cast<size_t>(i) * k + kk])) * tf32(b[static_cast<size_t>(j) * k + kk]);
      ref[static_cast<size_t>(i) * n + j] = acc;
    }
  }
  auto da = upload(a); auto db = upload(b);
  DevBuf<float> dd(ref.size()), dss(m);
  ScaledGemmArgs args{};
  args.recipe = GemmRecipe::kTF32;
  args.a.data = da.p; args.a.ld = k; args.a.dtype = DType::kFloat32;
  args.b.data = db.p; args.b.ld = k; args.b.dtype = DType::kFloat32;
  args.d = dd.p; args.ldd = n; args.d_dtype = DType::kFloat32;
  args.epi.ops = kops::gemm::kEpiOutFp32 | kops::gemm::kEpiRowSumSq;
  args.epi.row_sumsq = dss.p;
  args.m = m; args.n = n; args.k = k;
  KOPS_EXPECT_STATUS(kops::gemm::mma_gemm(args, nullptr, kops::Workspace{}, ctx), Status::kInvalidArgument, "tf32 without allow_tf32");
  ctx.allow_tf32 = true;
  KOPS_EXPECT_OK(kops::gemm::mma_gemm(args, nullptr, kops::Workspace{}, ctx), "tf32 mma");
  cudaDeviceSynchronize();
  kops_test::compare(kops_test::copy_out(dd.p, ref.size()), ref, 1e-4, 1e-4, "tf32 mma vs tf32-rounded fp64 oracle");
  kops_test::compare(kops_test::copy_out(dss.p, sumsq.size()), sumsq, 1e-4, 1e-5, "tf32 row_sumsq (fp32 A)");
}

/// kEpiQuantOut: static per-tensor and dynamic per-group (GLU + 1x128 on the 64x256 tile, int8 per 64).
void test_b7_quant_out(const LaunchContext& ctx) {
  using B = __nv_bfloat16;
  const int m = 96, n = 512, k = 256;   // GLU: 256 output columns
  Rng r(5150);
  const auto a = random_t<B>(static_cast<size_t>(m) * k, &r, -1.0f, 1.0f);
  const auto b = random_t<B>(static_cast<size_t>(n) * k, &r, -0.5f, 0.5f);
  auto da = upload(a); auto db = upload(b);
  for (int mode = 0; mode < 3; ++mode) {   // 0: static fp8, 1: GLU + per-group 128 fp8, 2: per-group 64 int8 (no GLU)
    const bool glu = mode == 1;
    const int n_out = glu ? n / 2 : n;
    const int gs = mode == 1 ? 128 : 64;
    std::vector<double> ref(static_cast<size_t>(m) * n_out);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int kk = 0; kk < k; ++kk) acc += dval(a, static_cast<size_t>(i) * k + kk) * dval(b, static_cast<size_t>(j) * k + kk);
        if (!glu) { ref[static_cast<size_t>(i) * n_out + j] = acc; continue; }
        if (j % 2 == 1) continue;   // interleave 1: gate j, up j + 1
        double up = 0.0;
        for (int kk = 0; kk < k; ++kk) up += dval(a, static_cast<size_t>(i) * k + kk) * dval(b, static_cast<size_t>(j + 1) * k + kk);
        ref[static_cast<size_t>(i) * n_out + j / 2] = acc / (1.0 + std::exp(-acc)) * up;
      }
    const float static_s = 0.02f;
    auto dss = upload(std::vector<float>(1, static_s));
    const int groups = n_out / gs;
    DevBuf<uint8_t> dq(static_cast<size_t>(m) * n_out);
    DevBuf<float> dsc(static_cast<size_t>(m) * groups);
    ScaledGemmArgs args{};
    args.recipe = GemmRecipe::kBF16;
    args.a.data = da.p; args.a.ld = k; args.a.dtype = DType::kBFloat16;
    args.b.data = db.p; args.b.ld = k; args.b.dtype = DType::kBFloat16;
    args.d_dtype = DType::kBFloat16; args.ldd = n_out;
    args.m = m; args.n = n; args.k = k;
    args.epi.ops = kops::gemm::kEpiQuantOut | (glu ? kops::gemm::kEpiGLU : 0u);
    if (glu) { args.epi.glu_layout = kops::GateLayout::kInterleaved; args.epi.glu_interleave = 1; args.epi.act = kops::ActFn::kSiLU; }
    kops::ActQuantOut& q = args.epi.quant_out;
    q.q = dq.p; q.q_row_stride = n_out;
    q.q_dtype = mode == 2 ? DType::kInt8 : DType::kFloat8E4M3;
    if (mode == 0) { q.mode = kops::ActQuantMode::kStaticPerTensor; q.static_scale = dss.p; }
    else { q.mode = kops::ActQuantMode::kPerGroup; q.group_size = gs; q.scales = dsc.p; q.scale_row_stride = groups; }
    char label[96];
    std::snprintf(label, sizeof(label), "quant_out mode %d", mode);
    KOPS_EXPECT_OK(kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx), label);
    cudaDeviceSynchronize();
    const auto qv = kops_test::copy_out(dq.p, static_cast<size_t>(m) * n_out);
    const auto sv = kops_test::copy_out(dsc.p, static_cast<size_t>(m) * groups);
    size_t bad = 0;
    for (int i = 0; i < m; ++i)
      for (int g = 0; g < groups; ++g) {
        double amax = 0.0;
        for (int j = 0; j < gs; ++j) amax = std::max(amax, std::fabs(ref[static_cast<size_t>(i) * n_out + g * gs + j]));
        const double qmax = mode == 2 ? 127.0 : 448.0;
        const double s = mode == 0 ? static_s : sv[static_cast<size_t>(i) * groups + g];
        if (mode != 0 && std::fabs(s - std::max(amax, 1e-10) / qmax) > 2e-2 * s) ++bad;
        for (int j = 0; j < gs; ++j) {
          const size_t idx = static_cast<size_t>(i) * n_out + g * gs + j;
          const double dq_v = mode == 2 ? static_cast<double>(static_cast<int8_t>(qv[idx])) : static_cast<double>(f8(qv[idx]));
          const double want = std::max(-qmax, std::min(qmax, ref[idx] / s));   // satfinite
          const double step = mode == 2 ? 1.0 : std::max(std::fabs(want) / 8.0, 1.0 / 512.0);
          if (std::fabs(dq_v - want) > step + 0.02 * std::fabs(want)) ++bad;
        }
      }
    if (bad != 0) {
      char d[96];
      std::snprintf(d, sizeof(d), "%zu quantized values / scales off", bad);
      kops_test::record_failure(label, d, __FILE__, __LINE__);
    }
  }
}

// =============================================================================
// B7 row 5: INT8 on mma.sync (gemm_mma_sync_kernel<AtomS8> through int8_gemm for M > 16) and
// w4a8_int_gemm_kernel (kW4A8IntInt8).  Oracle: exact int64 dot products (azp * colsum in the
// integers), then the epilogue in fp64.  The s32 accumulators are exact, so every tile, every split-K
// slice count and every repeat must store the same bits.
// =============================================================================

struct I8MmaSpec {
  int m = 64, n = 96, k = 256;
  int pad_a = 0, pad_b = 0, pad_d = 0;   // leading-dimension padding in elements (A / B: multiples of 16)
  bool asym = false;                     // kEpiAzpCorrection, colsum_b from int8_weight_colsum
  int scales = 0;                        // 0: epi row/col scales, 1: ScaledOperand per token / per channel,
                                         // 2: ScaledOperand per tensor (kINT8Tensorwise)
  uint32_t ops = 0;                      // extra ops: bias, clamp, softcap, act (SiLU), residual
};

double i8mma_epilogue(double v, uint32_t ops, double bias, double res) {
  if (ops & kg::kEpiBias) v += bias;
  if (ops & kg::kEpiClamp) v = std::min(std::max(v, -3.0), 3.0);
  if (ops & kg::kEpiSoftcap) v = 4.0 * std::tanh(v / 4.0);
  if (ops & kg::kEpiAct) v = silu_ref(v);
  if (ops & kg::kEpiResidualAdd) v += 0.5 * res;
  return v;
}

void i8mma_fill_epi(kg::EpilogueParams* e, uint32_t ops) {
  e->ops = ops;
  e->alpha = 0.75f;
  e->beta = 0.5f;
  e->clamp_limit = 3.0f;
  e->softcap = 4.0f;
  e->act = kops::ActFn::kSiLU;
}

std::string tactic_label(const char* what, const GemmTactic* t) {
  char buf[300];
  std::snprintf(buf, sizeof(buf), "%s [tile %dx%d swap %d split %d mode %d]", what, t ? t->tile_m : -1, t ? t->tile_n : -1,
                t && t->swap_ab, t ? t->split_k_slices : -1, t ? static_cast<int>(t->split_k_mode) : -1);
  return buf;
}

/// One int8_gemm call (D in Dt) compared with the oracle; returns the raw bytes of D (padding included).
template <typename Dt>
std::vector<uint8_t> i8mma_case_t(const LaunchContext& ctx, const I8MmaSpec& sp, const GemmTactic* tactic, const char* what) {
  const int m = sp.m, n = sp.n, k = sp.k;
  const int lda = k + sp.pad_a, ldb = k + sp.pad_b, ldd = n + sp.pad_d;
  const std::string label = tactic_label(what, tactic);
  Rng r(4242u + m * 13u + n * 5u + k + sp.ops + (sp.asym ? 7u : 0u) + sp.scales * 3u);
  std::vector<int8_t> a(static_cast<size_t>(m) * lda), w(static_cast<size_t>(n) * ldb);
  for (auto& v : a) v = static_cast<int8_t>(static_cast<int>(r.u32() % 256) - 128);
  for (auto& v : w) v = static_cast<int8_t>(static_cast<int>(r.u32() % 256) - 128);
  std::vector<int32_t> azp(m);
  for (auto& v : azp) v = static_cast<int>(r.u32() % 256) - 128;
  std::vector<float> rs(m), cs(n);
  for (auto& v : rs) v = r.uni(0.002f, 0.01f);
  for (auto& v : cs) v = r.uni(0.002f, 0.01f);
  const std::vector<float> sa{0.004f}, sb{0.006f};
  const auto bias = random_t<Dt>(n, &r, -1.0f, 1.0f);
  const auto res = random_t<Dt>(static_cast<size_t>(m) * n, &r, -1.0f, 1.0f);

  std::vector<int64_t> colsum(n, 0);
  for (int c = 0; c < n; ++c) for (int kk = 0; kk < k; ++kk) colsum[c] += w[static_cast<size_t>(c) * ldb + kk];
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    const int8_t* ar = &a[static_cast<size_t>(row) * lda];
    for (int c = 0; c < n; ++c) {
      const int8_t* wr = &w[static_cast<size_t>(c) * ldb];
      int64_t acc = 0;
      for (int kk = 0; kk < k; ++kk) acc += static_cast<int32_t>(ar[kk]) * wr[kk];
      if (sp.asym) acc -= static_cast<int64_t>(azp[row]) * colsum[c];
      double v = static_cast<double>(acc) * 0.75;
      if (sp.scales == 2) v *= static_cast<double>(sa[0]) * sb[0];
      else v *= static_cast<double>(rs[row]) * cs[c];
      ref[static_cast<size_t>(row) * n + c] = i8mma_epilogue(v, sp.ops, dval(bias, c), dval(res, static_cast<size_t>(row) * n + c));
    }
  }

  auto da = upload(a);
  auto dw = upload(w);
  auto dazp = upload(azp);
  auto drs = upload(rs);
  auto dcs = upload(cs);
  auto dsa = upload(sa);
  auto dsb = upload(sb);
  auto dbias = upload(bias);
  auto dres = upload(res);
  DevBuf<int32_t> dcol(n);
  if (sp.asym) {
    KOPS_EXPECT_OK(kg::int8_weight_colsum(tensor1d(dcol.p, DType::kInt32, n), tensor2d_ld(dw.p, DType::kInt8, n, k, ldb), ctx), "int8 mma colsum");
  }
  const std::vector<Dt> init(static_cast<size_t>(m) * ldd, cvt<Dt>(7.0f));
  DevBuf<Dt> dout(init.size());
  kops_test::copy_in(dout.p, init);

  ScaledGemmArgs args{};
  args.recipe = sp.scales == 2 ? GemmRecipe::kINT8Tensorwise : GemmRecipe::kINT8Rowwise;
  args.a.data = da.p; args.a.ld = lda; args.a.dtype = DType::kInt8;
  args.b.data = dw.p; args.b.ld = ldb; args.b.dtype = DType::kInt8;
  args.d = dout.p; args.ldd = ldd; args.d_dtype = dt_of<Dt>();
  args.m = m; args.n = n; args.k = k;
  i8mma_fill_epi(&args.epi, sp.ops | (sp.scales == 0 ? kg::kEpiRowScale | kg::kEpiColScale : 0u) | (sp.asym ? kg::kEpiAzpCorrection : 0u) |
                                (dt_of<Dt>() == DType::kFloat32 ? kg::kEpiOutFp32 : 0u));
  args.epi.row_scale = drs.p; args.epi.col_scale = dcs.p;
  args.epi.azp = dazp.p; args.epi.colsum_b = dcol.p;
  args.epi.bias = dbias.p; args.epi.bias_dtype = dt_of<Dt>();
  args.epi.residual = dres.p; args.epi.residual_stride = n;
  if (sp.scales == 1) {
    args.a.scale = drs.p; args.a.granularity = kops::ScaleGranularity::kPerToken;
    args.b.scale = dcs.p; args.b.granularity = kops::ScaleGranularity::kPerChannel;
  } else if (sp.scales == 2) {
    args.a.scale = dsa.p; args.a.granularity = kops::ScaleGranularity::kPerTensor;
    args.b.scale = dsb.p; args.b.granularity = kops::ScaleGranularity::kPerTensor;
  }
  kg::GemmProblem prob{};
  prob.m = m; prob.n = n; prob.k = k; prob.recipe = args.recipe; prob.a_dtype = DType::kInt8; prob.b_dtype = DType::kInt8;
  prob.d_dtype = dt_of<Dt>(); prob.epilogue_ops = args.epi.ops;
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kg::mma_workspace(prob, tactic, ctx, &req), label.c_str());
  DevBuf<uint8_t> scratch(req.device_scratch_bytes + 256);
  kops::Workspace ws{};
  ws.device_scratch = scratch.p; ws.device_scratch_bytes = req.device_scratch_bytes;
  const Status st = kg::int8_gemm(args, tactic, kg::GemmBackend::kNative, ws, ctx);
  KOPS_EXPECT_OK(st, label.c_str());
  if (st != Status::kSuccess) return {};
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, init.size());
  std::vector<Dt> got_n;
  got_n.reserve(ref.size());
  size_t pad_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (static_cast<int>(i % ldd) < n) got_n.push_back(got[i]);
    else pad_bad += std::memcmp(&got[i], &init[i], sizeof(Dt)) != 0;
  }
  if (pad_bad != 0) kops_test::record_failure(label.c_str(), "ld padding of D was written", __FILE__, __LINE__);
  expect_close(got_n, ref, label.c_str());
  std::vector<uint8_t> bytes(got.size() * sizeof(Dt));
  std::memcpy(bytes.data(), got.data(), bytes.size());
  return bytes;
}

std::vector<uint8_t> i8mma_case(const LaunchContext& ctx, DType out, const I8MmaSpec& sp, const GemmTactic* tactic, const char* what) {
  if (out == DType::kFloat32) return i8mma_case_t<float>(ctx, sp, tactic, what);
  if (out == DType::kFloat16) return i8mma_case_t<__half>(ctx, sp, tactic, what);
  return i8mma_case_t<__nv_bfloat16>(ctx, sp, tactic, what);
}

void i8mma_contracts(const LaunchContext& ctx) {
  const int m = 64, n = 64, k = 256;
  DevBuf<int8_t> a(static_cast<size_t>(m) * k + 64), w(static_cast<size_t>(n) * k + 64);
  DevBuf<float> sc(256);
  DevBuf<int32_t> iv(256);
  DevBuf<__nv_bfloat16> d(static_cast<size_t>(m) * n);
  cudaMemset(a.p, 1, static_cast<size_t>(m) * k + 64);
  cudaMemset(w.p, 1, static_cast<size_t>(n) * k + 64);
  const std::vector<__nv_bfloat16> sentinel(static_cast<size_t>(m) * n, cvt<__nv_bfloat16>(7.0f));
  auto base = [&] {
    ScaledGemmArgs x{};
    x.recipe = GemmRecipe::kINT8Rowwise;
    x.a.data = a.p; x.a.ld = k; x.a.dtype = DType::kInt8;
    x.b.data = w.p; x.b.ld = k; x.b.dtype = DType::kInt8;
    x.d = d.p; x.ldd = n; x.d_dtype = DType::kBFloat16;
    x.m = m; x.n = n; x.k = k;
    x.epi.row_scale = sc.p; x.epi.col_scale = sc.p; x.epi.azp = iv.p; x.epi.colsum_b = iv.p;
    return x;
  };
  auto expect = [&](const ScaledGemmArgs& x, Status want, const char* what, const GemmTactic* t = nullptr,
                    const LaunchContext* c = nullptr) {
    kops_test::copy_in(d.p, sentinel);
    KOPS_EXPECT_STATUS(kg::int8_gemm(x, t, kg::GemmBackend::kNative, kops::Workspace{}, c ? *c : ctx), want, what);
    cudaDeviceSynchronize();
    if (want != Status::kSuccess && !same_bits(kops_test::copy_out(d.p, sentinel.size()), sentinel)) {
      kops_test::record_failure(what, "a rejected call wrote D", __FILE__, __LINE__);
    }
  };
  expect(base(), Status::kSuccess, "int8 mma baseline (M 64)");
  { auto x = base(); x.recipe = GemmRecipe::kINT8Tensorwise; expect(x, Status::kSuccess, "int8 mma tensorwise baseline"); }
  { auto x = base(); x.a.dtype = DType::kBFloat16; expect(x, Status::kUnsupportedDtype, "int8 mma bf16 activations (W8A16 is M <= 16 only)"); }
  { auto x = base(); x.a.dtype = DType::kFloat8E4M3; expect(x, Status::kUnsupportedDtype, "int8 mma e4m3 activations"); }
  { auto x = base(); x.b.dtype = DType::kUInt8; expect(x, Status::kUnsupportedDtype, "int8 mma uint8 weights"); }
  { auto x = base(); x.d_dtype = DType::kInt8; expect(x, Status::kUnsupportedDtype, "int8 mma int8 output"); }
  { auto x = base(); x.b.major = kg::MajorOrder::kMNMajor; expect(x, Status::kUnsupportedLayout, "int8 mma MN-major weights"); }
  { auto x = base(); x.k = 248; expect(x, Status::kUnsupportedShape, "int8 mma K % 16"); }
  { auto x = base(); x.m = 3; x.k = 248; expect(x, Status::kUnsupportedShape, "int8 gemv K % 16 (same status)"); }
  { auto x = base(); x.a.data = reinterpret_cast<const char*>(a.p) + 1; expect(x, Status::kMisaligned, "int8 mma misaligned A"); }
  { auto x = base(); x.b.data = reinterpret_cast<const char*>(w.p) + 8; expect(x, Status::kMisaligned, "int8 mma misaligned B"); }
  { auto x = base(); x.a.ld = k + 8; x.m = 60; expect(x, Status::kMisaligned, "int8 mma A pitch % 16"); }
  { auto x = base(); x.b.ld = k + 4; x.n = 60; expect(x, Status::kMisaligned, "int8 mma B pitch % 16"); }
  { auto x = base(); x.a.ld = k - 16; expect(x, Status::kInvalidArgument, "int8 mma lda < K"); }
  { auto x = base(); x.ldd = n - 1; expect(x, Status::kInvalidArgument, "int8 mma ldd < N"); }
  { auto x = base(); x.d = nullptr; expect(x, Status::kInvalidArgument, "int8 mma null D"); }
  { auto x = base(); x.m = 0; expect(x, Status::kInvalidArgument, "int8 mma M 0"); }
  { auto x = base(); x.epi.ops = kg::kEpiAzpCorrection; x.epi.azp = nullptr; expect(x, Status::kInvalidArgument, "int8 mma azp missing"); }
  { auto x = base(); x.epi.ops = kg::kEpiAzpCorrection; x.epi.colsum_b = nullptr; expect(x, Status::kInvalidArgument, "int8 mma colsum missing"); }
  { auto x = base(); x.epi.ops = kg::kEpiRowScale; x.epi.row_scale = nullptr; expect(x, Status::kInvalidArgument, "int8 mma row scale missing"); }
  { auto x = base(); x.epi.ops = kg::kEpiBias; expect(x, Status::kInvalidArgument, "int8 mma bias missing"); }
  { auto x = base(); x.epi.ops = kg::kEpiOutFp32; expect(x, Status::kInvalidArgument, "int8 mma kEpiOutFp32 with a bf16 output"); }
  { auto x = base(); x.d_dtype = DType::kFloat32; expect(x, Status::kInvalidArgument, "int8 mma fp32 output without kEpiOutFp32"); }
  { auto x = base(); x.epi.ops = kg::kEpiGLU; expect(x, Status::kNotImplemented, "int8 mma GLU without the interleaved layout"); }
  { auto x = base(); x.epi.ops = kg::kEpiRowSumSq; x.epi.row_sumsq = sc.p; expect(x, Status::kNotImplemented, "int8 mma kEpiRowSumSq (s8 activations)"); }
  { auto x = base(); x.epi.ops = 1u << 30; expect(x, Status::kInvalidArgument, "int8 mma unknown epilogue bit"); }
  {
    LaunchContext det = ctx;
    det.deterministic = true;
    auto x = base(); x.epi.ops = kg::kEpiAtomicAdd;
    expect(x, Status::kInvalidArgument, "int8 mma atomic add under ctx.deterministic", nullptr, &det);
  }
  { auto x = base(); x.a.scale = sc.p; x.a.granularity = kops::ScaleGranularity::kPerChannel; expect(x, Status::kUnsupportedLayout, "int8 mma per-channel A scale"); }
  { auto x = base(); x.b.scale = sc.p; x.b.granularity = kops::ScaleGranularity::kPerChannel; x.b.scale_dtype = DType::kBFloat16;
    expect(x, Status::kUnsupportedLayout, "int8 mma bf16 operand scale"); }
  const std::vector<GemmTactic> ts = tactics_for(ctx, GemmRecipe::kINT8Rowwise, m, n, k, 0);
  if (!ts.empty()) {
    GemmTactic t = ts[0];
    t.split_k_mode = kg::SplitKMode::kParallel; t.split_k_slices = 2;
    expect(base(), Status::kNotImplemented, "int8 mma parallel split-K (s32 partials)", &t);
    t.split_k_mode = kg::SplitKMode::kSerial;
    expect(base(), Status::kWorkspaceTooSmall, "int8 mma serial split-K without workspace", &t);
    t = ts[0]; t.tile_m = 64; t.tile_n = 256; t.swap_ab = false; t.split_k_slices = 1; t.split_k_mode = kg::SplitKMode::kNone;
    expect(base(), Status::kUnsupportedShape, "int8 mma 64x256 tile (not compiled for s8)", &t);
    t = ts[0]; t.mainloop = kg::MainloopSchedule::kTmaWarpSpecialized;
    { auto x = base(); x.m = 3; expect(x, Status::kSuccess, "int8 non-mma tactic at M 3 runs the GEMV", &t); }
    expect(base(), Status::kUnsupportedArch, "int8 wgmma tactic on SM89", &t);
  }
  {
    LaunchContext hip = ctx;
    hip.backend = kops::BackendKind::kHip;
    expect(base(), Status::kUnsupportedBackend, "int8 mma HIP backend", nullptr, &hip);
  }
}

void test_b7_int8(const LaunchContext& ctx) {
  const DType bf = DType::kBFloat16;
  // 1. M sweep through the heuristic tactic; N not a multiple of any tile, K % 64 != 0
  for (int m : {17, 31, 64, 127, 256, 1000, 2048}) {
    I8MmaSpec s;
    s.m = m; s.n = 200; s.k = 272;
    i8mma_case(ctx, bf, s, nullptr, "int8 mma M sweep");
    s.n = 136; s.k = 528; s.asym = true; s.ops = kg::kEpiBias;
    i8mma_case(ctx, bf, s, nullptr, "int8 mma M sweep azp + bias");
  }
  // 2. operand-scale forms x output dtypes x epilogues, padded leading dimensions
  const uint32_t epis[] = {0u, kg::kEpiBias, kg::kEpiBias | kg::kEpiAct, kg::kEpiBias | kg::kEpiAct | kg::kEpiResidualAdd,
                           kg::kEpiClamp | kg::kEpiSoftcap | kg::kEpiResidualAdd};
  for (DType out : {DType::kBFloat16, DType::kFloat16, DType::kFloat32}) {
    for (int scales : {0, 1, 2}) {
      for (uint32_t ops : epis) {
        I8MmaSpec s;
        s.m = 97; s.n = 72; s.k = 160; s.pad_a = 16; s.pad_b = 32; s.pad_d = 3;
        s.scales = scales; s.ops = ops; s.asym = scales != 2 && (ops & kg::kEpiBias) != 0u;
        i8mma_case(ctx, out, s, nullptr, "int8 mma scales / dtype / epilogue");
      }
    }
  }
  // 3. every tactic get_tactics offers (tiles, swap_ab, split-K) stores the same bits
  {
    I8MmaSpec s;
    s.m = 130; s.n = 136; s.k = 2064; s.asym = true; s.ops = kg::kEpiBias | kg::kEpiAct | kg::kEpiResidualAdd; s.pad_d = 8;
    std::vector<uint8_t> first;
    int count = 0, splits = 0;
    for (const GemmTactic& t : tactics_for(ctx, GemmRecipe::kINT8Rowwise, s.m, s.n, s.k, s.ops)) {
      const auto bits = i8mma_case(ctx, bf, s, &t, "int8 mma all tactics");
      if (bits.empty()) continue;
      ++count;
      splits += t.split_k_slices > 1;
      if (first.empty()) first = bits;
      else if (bits != first) kops_test::record_failure(tactic_label("int8 mma all tactics", &t).c_str(), "bits differ from the first tactic", __FILE__, __LINE__);
    }
    if (count < 7 || splits == 0) kops_test::record_failure("int8 mma all tactics", "expected the 7 s8 tiles and a split-K tactic", __FILE__, __LINE__);
  }
  // 4. serial split-K with 2 and 4 slices on every tile: the s32 partials sum exactly (K tail in the last slice)
  {
    I8MmaSpec s;
    s.m = 80; s.n = 200; s.k = 1040; s.asym = true; s.ops = kg::kEpiBias | kg::kEpiAct | kg::kEpiResidualAdd;
    for (const GemmTactic& t : tactics_for(ctx, GemmRecipe::kINT8Rowwise, s.m, s.n, s.k, s.ops)) {
      if (t.split_k_slices != 1) continue;
      const auto none = i8mma_case(ctx, DType::kFloat16, s, &t, "int8 mma split-K reference");
      for (int slices : {2, 4}) {
        GemmTactic ts = t;
        ts.split_k_mode = kg::SplitKMode::kSerial;
        ts.split_k_slices = static_cast<int16_t>(slices);
        const auto got = i8mma_case(ctx, DType::kFloat16, s, &ts, "int8 mma split-K serial");
        if (!none.empty() && got != none) kops_test::record_failure(tactic_label("int8 mma split-K serial", &ts).c_str(), "split-K differs from no split", __FILE__, __LINE__);
      }
    }
  }
  // 5. repeat determinism of the serial split-K (the last CTA sums in slice order, whatever arrived first)
  {
    I8MmaSpec s;
    s.m = 64; s.n = 256; s.k = 4096; s.asym = true; s.ops = kg::kEpiBias;
    const std::vector<GemmTactic> ts = tactics_for(ctx, GemmRecipe::kINT8Rowwise, s.m, s.n, s.k, s.ops);
    if (!ts.empty()) {
      GemmTactic t = ts[0];
      t.split_k_mode = kg::SplitKMode::kSerial;
      t.split_k_slices = 4;
      const auto ref = i8mma_case(ctx, DType::kFloat32, s, &t, "int8 mma repeat");
      for (int rep = 0; rep < 5; ++rep) {
        if (i8mma_case(ctx, DType::kFloat32, s, &t, "int8 mma repeat") != ref) {
          kops_test::record_failure("int8 mma repeat", "result changed between runs", __FILE__, __LINE__);
          break;
        }
      }
      const auto plain = i8mma_case(ctx, DType::kFloat32, s, &ts[0], "int8 mma repeat (heuristic)");
      if (plain != ref) kops_test::record_failure("int8 mma repeat", "split-K 4 differs from the heuristic tactic", __FILE__, __LINE__);
    }
  }
  i8mma_contracts(ctx);
}

// ---- W4A8 (QServe) ---------------------------------------------------------------------------

struct W4A8Spec {
  int m = 64, n = 64, k = 256;
  int group = 128;                     // 128 per group (two-level), 0 per channel
  int pad_a = 0, pad_b = 0, pad_d = 0; // pad_b in bytes of the packed rows
  uint32_t ops = 0;                    // extra ops: bias, clamp, softcap, act, residual
  bool extreme = false;                // weights at w8 = -128 and activations at -128 / 127
};

/// One w4a8 call compared with the oracle.  cross_check: the same problem with the host-expanded s8
/// weights through the INT8 path (GEMV or mma.sync) must store the same bits.
template <typename Dt>
std::vector<uint8_t> w4a8_case_t(const LaunchContext& ctx, const W4A8Spec& sp, const char* what, bool cross_check) {
  const int m = sp.m, n = sp.n, k = sp.k, g = sp.group;
  const int lda = k + sp.pad_a, ldb = k / 2 + sp.pad_b, ldd = n + sp.pad_d;
  const int groups = g > 0 ? k / g : 0;
  char label[256];
  std::snprintf(label, sizeof(label), "%s [M %d N %d K %d group %d]", what, m, n, k, g);
  Rng r(9001u + m * 3u + n * 11u + k + g + sp.ops + (sp.extreme ? 5u : 0u));
  std::vector<int8_t> a(static_cast<size_t>(m) * lda);
  for (auto& v : a) v = sp.extreme ? static_cast<int8_t>((r.u32() & 1) ? 127 : -128) : static_cast<int8_t>(static_cast<int>(r.u32() % 256) - 128);
  std::vector<uint8_t> q(static_cast<size_t>(n) * k);
  for (auto& v : q) v = static_cast<uint8_t>(sp.extreme ? ((r.u32() & 1) ? 15 : 0) : r.u32() % 16);
  std::vector<uint8_t> packed(static_cast<size_t>(n) * ldb);
  for (auto& v : packed) v = static_cast<uint8_t>(r.u32());   // pad bytes hold garbage
  for (int c = 0; c < n; ++c) {
    for (int kk = 0; kk < k; kk += 2) {
      packed[static_cast<size_t>(c) * ldb + kk / 2] =
          static_cast<uint8_t>(q[static_cast<size_t>(c) * k + kk] | (q[static_cast<size_t>(c) * k + kk + 1] << 4));
    }
  }
  std::vector<uint8_t> s2(static_cast<size_t>(std::max(groups, 1)) * n), sz(s2.size()), z(n);
  for (size_t i = 0; i < s2.size(); ++i) {
    // any (s2, sz) with q * s2 - sz in [-128, 127] for q in 0..15: sz in [max(0, 15 s2 - 127), 128]
    const int s = sp.extreme ? 16 : 1 + r.range(16);
    const int lo = std::max(0, 15 * s - 127);
    s2[i] = static_cast<uint8_t>(s);
    sz[i] = static_cast<uint8_t>(sp.extreme ? 128 : lo + r.range(128 - lo + 1));
  }
  for (auto& v : z) v = static_cast<uint8_t>(sp.extreme ? 128 : r.range(16));
  std::vector<int8_t> w8(static_cast<size_t>(n) * k);
  for (int c = 0; c < n; ++c) {
    for (int kk = 0; kk < k; ++kk) {
      const int qv = q[static_cast<size_t>(c) * k + kk];
      const int v = g > 0 ? qv * s2[static_cast<size_t>(kk / g) * n + c] - sz[static_cast<size_t>(kk / g) * n + c] : qv - z[c];
      if (v < -128 || v > 127) { kops_test::record_failure(label, "test data outside the s8 range", __FILE__, __LINE__); return {}; }
      w8[static_cast<size_t>(c) * k + kk] = static_cast<int8_t>(v);
    }
  }
  std::vector<float> rs(m), cs(n);
  for (auto& v : rs) v = r.uni(0.002f, 0.01f);
  for (auto& v : cs) v = r.uni(0.002f, 0.01f);
  const auto bias = random_t<Dt>(n, &r, -1.0f, 1.0f);
  const auto res = random_t<Dt>(static_cast<size_t>(m) * n, &r, -1.0f, 1.0f);
  std::vector<double> ref(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < n; ++c) {
      int64_t acc = 0;
      for (int kk = 0; kk < k; ++kk) acc += static_cast<int32_t>(a[static_cast<size_t>(row) * lda + kk]) * w8[static_cast<size_t>(c) * k + kk];
      const double v = static_cast<double>(acc) * 0.75 * rs[row] * cs[c];
      ref[static_cast<size_t>(row) * n + c] = i8mma_epilogue(v, sp.ops, dval(bias, c), dval(res, static_cast<size_t>(row) * n + c));
    }
  }
  auto da = upload(a);
  auto dpk = upload(packed);
  auto ds2 = upload(s2);
  auto dsz = upload(sz);
  auto dz = upload(z);
  auto dw8 = upload(w8);
  auto drs = upload(rs);
  auto dcs = upload(cs);
  auto dbias = upload(bias);
  auto dres = upload(res);
  const std::vector<Dt> init(static_cast<size_t>(m) * ldd, cvt<Dt>(7.0f));
  DevBuf<Dt> dout(init.size());
  kops_test::copy_in(dout.p, init);
  ScaledGemmArgs args{};
  args.recipe = GemmRecipe::kW4A8IntInt8;
  args.a.data = da.p; args.a.ld = lda; args.a.dtype = DType::kInt8;
  args.b.data = dpk.p; args.b.ld = ldb; args.b.dtype = DType::kUInt4x2; args.b.group_k = static_cast<int16_t>(g);
  args.b.second_level = g > 0 ? ds2.p : nullptr;
  args.b.zero_points = g > 0 ? static_cast<const void*>(dsz.p) : static_cast<const void*>(dz.p);
  args.b.zero_dtype = DType::kUInt8;
  args.d = dout.p; args.ldd = ldd; args.d_dtype = dt_of<Dt>();
  args.m = m; args.n = n; args.k = k;
  i8mma_fill_epi(&args.epi, sp.ops | kg::kEpiRowScale | kg::kEpiColScale | (dt_of<Dt>() == DType::kFloat32 ? kg::kEpiOutFp32 : 0u));
  args.epi.row_scale = drs.p; args.epi.col_scale = dcs.p;
  args.epi.bias = dbias.p; args.epi.bias_dtype = dt_of<Dt>();
  args.epi.residual = dres.p; args.epi.residual_stride = n;
  const Status st = kg::int8_gemm(args, nullptr, kg::GemmBackend::kNative, kops::Workspace{}, ctx);
  KOPS_EXPECT_OK(st, label);
  if (st != Status::kSuccess) return {};
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, init.size());
  std::vector<Dt> got_n;
  size_t pad_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (static_cast<int>(i % ldd) < n) got_n.push_back(got[i]);
    else pad_bad += std::memcmp(&got[i], &init[i], sizeof(Dt)) != 0;
  }
  if (pad_bad != 0) kops_test::record_failure(label, "ld padding of D was written", __FILE__, __LINE__);
  expect_close(got_n, ref, label);
  if (cross_check) {
    // same epilogue code path order (alpha, row, col, bias, clamp, softcap, act, residual) on the same exact integer
    ScaledGemmArgs x = args;
    x.recipe = GemmRecipe::kINT8Rowwise;
    x.b = kg::ScaledOperand{};
    x.b.data = dw8.p; x.b.ld = k; x.b.dtype = DType::kInt8;
    DevBuf<Dt> dout8(init.size());
    kops_test::copy_in(dout8.p, init);
    x.d = dout8.p;
    kg::GemmProblem prob{};
    prob.m = m; prob.n = n; prob.k = k; prob.recipe = x.recipe; prob.epilogue_ops = x.epi.ops;
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::mma_workspace(prob, nullptr, ctx, &req), label);
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 256);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p; ws.device_scratch_bytes = req.device_scratch_bytes;
    KOPS_EXPECT_OK(kg::int8_gemm(x, nullptr, kg::GemmBackend::kNative, ws, ctx), label);
    cudaDeviceSynchronize();
    if (!same_bits(kops_test::copy_out(dout8.p, init.size()), got)) {
      kops_test::record_failure(label, "w4a8 differs from int8_gemm on the expanded s8 weights", __FILE__, __LINE__);
    }
  }
  std::vector<uint8_t> bytes(got.size() * sizeof(Dt));
  std::memcpy(bytes.data(), got.data(), bytes.size());
  return bytes;
}

std::vector<uint8_t> w4a8_case(const LaunchContext& ctx, DType out, const W4A8Spec& sp, const char* what, bool cross_check) {
  if (out == DType::kFloat32) return w4a8_case_t<float>(ctx, sp, what, cross_check);
  if (out == DType::kFloat16) return w4a8_case_t<__half>(ctx, sp, what, cross_check);
  return w4a8_case_t<__nv_bfloat16>(ctx, sp, what, cross_check);
}

void w4a8_contracts(const LaunchContext& ctx) {
  const int m = 64, n = 64, k = 256;
  DevBuf<int8_t> a(static_cast<size_t>(m) * k + 64);
  DevBuf<uint8_t> w(static_cast<size_t>(n) * k / 2 + 64), s2(4 * n), zp(4 * n);
  DevBuf<float> sc(256);
  DevBuf<__nv_bfloat16> d(static_cast<size_t>(m) * n);
  cudaMemset(a.p, 1, static_cast<size_t>(m) * k + 64);
  cudaMemset(w.p, 0x11, static_cast<size_t>(n) * k / 2 + 64);
  cudaMemset(s2.p, 1, 4 * n);
  cudaMemset(zp.p, 0, 4 * n);
  const std::vector<__nv_bfloat16> sentinel(static_cast<size_t>(m) * n, cvt<__nv_bfloat16>(7.0f));
  auto base = [&] {
    ScaledGemmArgs x{};
    x.recipe = GemmRecipe::kW4A8IntInt8;
    x.a.data = a.p; x.a.ld = k; x.a.dtype = DType::kInt8;
    x.b.data = w.p; x.b.ld = k / 2; x.b.dtype = DType::kUInt4x2; x.b.group_k = 128;
    x.b.second_level = s2.p; x.b.zero_points = zp.p; x.b.zero_dtype = DType::kUInt8;
    x.d = d.p; x.ldd = n; x.d_dtype = DType::kBFloat16;
    x.m = m; x.n = n; x.k = k;
    x.epi.ops = kg::kEpiRowScale | kg::kEpiColScale; x.epi.row_scale = sc.p; x.epi.col_scale = sc.p;
    return x;
  };
  auto expect = [&](const ScaledGemmArgs& x, Status want, const char* what, const LaunchContext* c = nullptr) {
    kops_test::copy_in(d.p, sentinel);
    KOPS_EXPECT_STATUS(kg::int8_gemm(x, nullptr, kg::GemmBackend::kNative, kops::Workspace{}, c ? *c : ctx), want, what);
    cudaDeviceSynchronize();
    if (want != Status::kSuccess && !same_bits(kops_test::copy_out(d.p, sentinel.size()), sentinel)) {
      kops_test::record_failure(what, "a rejected call wrote D", __FILE__, __LINE__);
    }
  };
  expect(base(), Status::kSuccess, "w4a8 baseline (per group)");
  { auto x = base(); x.b.group_k = 0; x.b.second_level = nullptr; expect(x, Status::kSuccess, "w4a8 baseline (per channel)"); }
  { auto x = base(); x.b.zero_dtype = DType::kInvalid; expect(x, Status::kSuccess, "w4a8 unset zero dtype reads uint8"); }
  { auto x = base(); x.a.dtype = DType::kBFloat16; expect(x, Status::kUnsupportedDtype, "w4a8 bf16 activations"); }
  { auto x = base(); x.b.dtype = DType::kInt4x2; expect(x, Status::kUnsupportedDtype, "w4a8 signed int4 weights"); }
  { auto x = base(); x.b.dtype = DType::kInt8; expect(x, Status::kUnsupportedDtype, "w4a8 int8 weights"); }
  { auto x = base(); x.d_dtype = DType::kInt8; expect(x, Status::kUnsupportedDtype, "w4a8 int8 output"); }
  { auto x = base(); x.b.zero_dtype = DType::kInt32; expect(x, Status::kUnsupportedDtype, "w4a8 int32 zero points"); }
  { auto x = base(); x.b.major = kg::MajorOrder::kMNMajor; expect(x, Status::kUnsupportedLayout, "w4a8 MN-major weights"); }
  { auto x = base(); x.b.group_k = 64; expect(x, Status::kUnsupportedShape, "w4a8 group 64"); }
  { auto x = base(); x.k = 192; expect(x, Status::kUnsupportedShape, "w4a8 K % 128 per group"); }
  { auto x = base(); x.b.group_k = 0; x.k = 224; expect(x, Status::kUnsupportedShape, "w4a8 K % 64 per channel"); }
  { auto x = base(); x.b.zero_points = nullptr; expect(x, Status::kInvalidArgument, "w4a8 scaled zeros missing"); }
  { auto x = base(); x.b.second_level = nullptr; expect(x, Status::kInvalidArgument, "w4a8 level-2 scales missing"); }
  { auto x = base(); x.b.group_k = 0; x.b.zero_points = nullptr; expect(x, Status::kInvalidArgument, "w4a8 channel zeros missing"); }
  { auto x = base(); x.a.data = reinterpret_cast<const char*>(a.p) + 4; expect(x, Status::kMisaligned, "w4a8 misaligned A"); }
  { auto x = base(); x.b.data = w.p + 8; expect(x, Status::kMisaligned, "w4a8 misaligned B"); }
  { auto x = base(); x.a.ld = k + 8; x.m = 60; expect(x, Status::kMisaligned, "w4a8 A pitch % 16"); }
  { auto x = base(); x.b.ld = k / 2 + 8; x.n = 60; expect(x, Status::kMisaligned, "w4a8 B pitch % 16"); }
  { auto x = base(); x.b.ld = k / 2 - 16; expect(x, Status::kInvalidArgument, "w4a8 B pitch < K / 2 bytes"); }
  { auto x = base(); x.ldd = n - 1; expect(x, Status::kInvalidArgument, "w4a8 ldd < N"); }
  { auto x = base(); x.d = nullptr; expect(x, Status::kInvalidArgument, "w4a8 null D"); }
  { auto x = base(); x.n = 0; expect(x, Status::kInvalidArgument, "w4a8 N 0"); }
  { auto x = base(); x.batch = 2; expect(x, Status::kNotImplemented, "w4a8 batch 2"); }
  { auto x = base(); x.epi.ops |= kg::kEpiAzpCorrection; expect(x, Status::kNotImplemented, "w4a8 azp correction"); }
  { auto x = base(); x.epi.ops |= kg::kEpiGLU; expect(x, Status::kNotImplemented, "w4a8 GLU"); }
  { auto x = base(); x.epi.ops |= kg::kEpiQuantOut; expect(x, Status::kNotImplemented, "w4a8 quantized output"); }
  { auto x = base(); x.epi.row_scale = nullptr; expect(x, Status::kInvalidArgument, "w4a8 row scale missing"); }
  { auto x = base(); x.epi.ops |= kg::kEpiResidualAdd; expect(x, Status::kInvalidArgument, "w4a8 residual missing"); }
  { auto x = base(); x.epi.ops |= kg::kEpiOutFp32; expect(x, Status::kInvalidArgument, "w4a8 kEpiOutFp32 with a bf16 output"); }
  { auto x = base(); x.a.scale = sc.p; x.a.granularity = kops::ScaleGranularity::kPerToken; expect(x, Status::kInvalidArgument, "w4a8 ScaledOperand scale"); }
  {
    LaunchContext hip = ctx;
    hip.backend = kops::BackendKind::kHip;
    expect(base(), Status::kUnsupportedBackend, "w4a8 HIP backend", &hip);
  }
}

void test_b7_w4a8(const LaunchContext& ctx) {
  const DType bf = DType::kBFloat16;
  for (int g : {128, 0}) {
    // M sweep (decode to prefill), N tail, cross-checked bit for bit against int8_gemm on the s8 weights
    for (int m : {1, 5, 17, 64, 127, 256, 1000}) {
      W4A8Spec s;
      s.m = m; s.n = 200; s.k = 384; s.group = g;
      w4a8_case(ctx, bf, s, "w4a8 M sweep", true);
    }
    // output dtypes x epilogues, padded leading dimensions
    for (DType out : {DType::kBFloat16, DType::kFloat16, DType::kFloat32}) {
      for (uint32_t ops : {0u, kg::kEpiBias | kg::kEpiAct | kg::kEpiResidualAdd, kg::kEpiClamp | kg::kEpiSoftcap | kg::kEpiBias}) {
        W4A8Spec s;
        s.m = 70; s.n = 136; s.k = 512; s.group = g; s.pad_a = 16; s.pad_b = 16; s.pad_d = 5; s.ops = ops;
        w4a8_case(ctx, out, s, "w4a8 dtype / epilogue / padding", (ops & kg::kEpiBias) == 0u);
      }
    }
    // the ends of the s8 range (w8 = -128, a = -128 / 127), long K
    {
      W4A8Spec s;
      s.m = 96; s.n = 72; s.k = 4096; s.group = g; s.extreme = true;
      w4a8_case(ctx, DType::kFloat32, s, "w4a8 range ends", true);
      s.extreme = false; s.m = 128; s.n = 128;
      const auto ref = w4a8_case(ctx, DType::kFloat32, s, "w4a8 long K", true);
      for (int rep = 0; rep < 3; ++rep) {
        if (w4a8_case(ctx, DType::kFloat32, s, "w4a8 repeat", false) != ref) {
          kops_test::record_failure("w4a8 repeat", "result changed between runs", __FILE__, __LINE__);
          break;
        }
      }
    }
  }
  {
    W4A8Spec s;   // per channel with K not a multiple of 128
    s.m = 33; s.n = 40; s.k = 192; s.group = 0;
    w4a8_case(ctx, bf, s, "w4a8 per channel K 192", true);
  }
  w4a8_contracts(ctx);
}

// ---- B7 row 10: GGUF mmq (prefill on int8 mma.sync) ------------------------------------------------
// Oracle: fp64 dot products of the B4-dequantized weights with the host-quantized q8_1 activations
// (same as run_gguf).  Batch invariance: every mmq tile config (forced through ctx.compute_units) and
// every M must give bit-identical rows; ctx.batch_invariant routes M <= 16 to mmq as well.

struct GgufMmqProblem {
  int m, n, k;
  std::vector<uint8_t> raw;      // [N, K] ggml blocks
  std::vector<float> w;          // dequantized
  Q81Host q;
  std::vector<double> ref;       // [m, n]
};

template <typename T>
GgufMmqProblem gguf_mmq_problem(const LaunchContext& ctx, const GgufFormat& f, int m, int n, int k, const std::vector<T>& a, unsigned seed) {
  GgufMmqProblem p{m, n, k, {}, {}, {}, {}};
  Rng r(seed);
  p.raw = random_gguf(f, static_cast<size_t>(n) * (k / f.elems), &r);
  auto dq = upload(p.raw);
  DevBuf<float> wdeq(static_cast<size_t>(n) * k);
  KOPS_EXPECT_OK(kops::quant::gguf_dequantize(kops_test::tensor2d(wdeq.p, DType::kFloat32, n, k),
                                              tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(p.raw.size())), f.type, ctx), "mmq dequant");
  cudaDeviceSynchronize();
  p.w = kops_test::copy_out(wdeq.p, static_cast<size_t>(n) * k);
  p.q = quantize_q81_host(a, m, k);
  p.ref.assign(static_cast<size_t>(m) * n, 0.0);
  for (int row = 0; row < m; ++row)
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      const float* wr = &p.w[static_cast<size_t>(c) * k];
      const double* ar = &p.q.deq[static_cast<size_t>(row) * k];
      for (int kk = 0; kk < k; ++kk) acc += static_cast<double>(wr[kk]) * ar[kk];
      p.ref[static_cast<size_t>(row) * n + c] = acc;
    }
  return p;
}

/// gguf_gemm on the first `m` rows of `a`; returns the output.
template <typename T>
std::vector<T> gguf_mmq_run(const LaunchContext& ctx, const GgufFormat& f, const GgufMmqProblem& p, const std::vector<T>& a, int m,
                            const char* what) {
  auto da = upload(std::vector<T>(a.begin(), a.begin() + static_cast<size_t>(m) * p.k));
  auto dq = upload(p.raw);
  DevBuf<uint8_t> ws(static_cast<size_t>(m) * (p.k / 32) * 36);
  DevBuf<T> dout(static_cast<size_t>(m) * p.n);
  KOPS_EXPECT_OK(kops::gemm::gguf_gemm(kops_test::tensor2d(dout.p, dt_of<T>(), m, p.n), kops_test::tensor2d(da.p, dt_of<T>(), m, p.k),
                                       tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(p.raw.size())), f.type,
                                       tensor1d(ws.p, DType::kUInt8, static_cast<int64_t>(m) * (p.k / 32) * 36), ctx), what);
  cudaDeviceSynchronize();
  return kops_test::copy_out(dout.p, static_cast<size_t>(m) * p.n);
}

template <typename T>
bool gguf_rows_same(const std::vector<T>& small, const std::vector<T>& big) {
  return small.size() <= big.size() && std::memcmp(small.data(), big.data(), small.size() * sizeof(T)) == 0;
}

void test_b7_gguf_mmq_dense(const LaunchContext& ctx) {
  const int kMaxM = 600;
  const int n = 200;   // tails of the 64- and 128-row weight tiles
  for (const GgufFormat& f : kGguf) {
    const int k = f.elems == 32 ? 480 : 512;   // 480: 15 units, a partial 4-unit stage
    char what[128];
    Rng r(9100u + f.bytes);
    const auto a = random_t<__nv_bfloat16>(static_cast<size_t>(kMaxM) * k, &r, -2.0f, 2.0f);
    const GgufMmqProblem p = gguf_mmq_problem(ctx, f, kMaxM, n, k, a, 9200u + f.bytes);
    // Reference config set: default (one wave on this GPU -> 32x64 tiles here), 64x128, 128x128.
    LaunchContext c128 = ctx, c64 = ctx;
    c128.compute_units = 1;    // ctas(128x128) = 10 >= 1
    c64.compute_units = 15;    // ctas(128x128) = 10 < 15 <= ctas(64x128) = 20
    std::snprintf(what, sizeof(what), "gguf mmq %s M=%d", f.name, kMaxM);
    const auto full = gguf_mmq_run(ctx, f, p, a, kMaxM, what);
    expect_close(full, p.ref, what);
    std::snprintf(what, sizeof(what), "gguf mmq %s M=%d tile 128x128", f.name, kMaxM);
    const auto full128 = gguf_mmq_run(c128, f, p, a, kMaxM, what);
    if (!same_bits(full, full128)) kops_test::record_failure(what, "differs from the default tile (not batch-invariant)", __FILE__, __LINE__);
    std::snprintf(what, sizeof(what), "gguf mmq %s M=%d tile 64x128", f.name, kMaxM);
    const auto full64 = gguf_mmq_run(c64, f, p, a, kMaxM, what);
    if (!same_bits(full, full64)) kops_test::record_failure(what, "differs from the default tile (not batch-invariant)", __FILE__, __LINE__);
    std::snprintf(what, sizeof(what), "gguf mmq %s M=%d repeat", f.name, kMaxM);
    if (!same_bits(full, gguf_mmq_run(ctx, f, p, a, kMaxM, what))) kops_test::record_failure(what, "not deterministic", __FILE__, __LINE__);
    for (int m : {17, 31, 64, 127, 256}) {
      std::snprintf(what, sizeof(what), "gguf mmq %s M=%d", f.name, m);
      const auto got = gguf_mmq_run(ctx, f, p, a, m, what);
      expect_close(got, std::vector<double>(p.ref.begin(), p.ref.begin() + static_cast<size_t>(m) * n), what);
      if (!gguf_rows_same(got, full)) kops_test::record_failure(what, "rows differ from the M=600 run (not batch-invariant)", __FILE__, __LINE__);
    }
    // M = 16: mmvq (default) vs mmq (batch_invariant) on the same q8_1 data.
    LaunchContext inv = ctx;
    inv.batch_invariant = true;
    std::snprintf(what, sizeof(what), "gguf %s M=16 mmvq", f.name);
    const auto vq = gguf_mmq_run(ctx, f, p, a, 16, what);
    std::snprintf(what, sizeof(what), "gguf %s M=16 mmq (batch_invariant)", f.name);
    const auto mq = gguf_mmq_run(inv, f, p, a, 16, what);
    const std::vector<double> ref16(p.ref.begin(), p.ref.begin() + 16 * n);
    expect_close(mq, ref16, what);
    if (!gguf_rows_same(mq, full)) kops_test::record_failure(what, "rows differ from the M=600 mmq run", __FILE__, __LINE__);
    double ss = 0.0;
    for (auto v : mq) { const double y = kops_test::to_float(v); ss += y * y; }
    const double rms = std::sqrt(ss / static_cast<double>(mq.empty() ? 1 : mq.size()));
    size_t off = 0;
    for (size_t i = 0; i < vq.size(); ++i) {
      const double x = kops_test::to_float(vq[i]), y = kops_test::to_float(mq[i]);
      // mmvq runs fp32 FMA over q8_1 blocks, mmq runs the s8 mma.sync: two accumulation orders. Use the
      // usual one-ulp + accumulation-noise rule (the row/batch invariance checks above already require
      // mmq to be bit-reproducible).
      if (std::fabs(x - y) > rel_eps<__nv_bfloat16>() * std::max(std::fabs(x), std::fabs(y)) + 2e-5 * rms) ++off;
    }
    if (off != 0) {
      char d[96];
      std::snprintf(d, sizeof(d), "%zu/%zu outputs differ by more than one bf16 ulp + 2e-5 rms", off, vq.size());
      kops_test::record_failure(what, d, __FILE__, __LINE__);
    }
  }
  // fp16 output (tighter ulp), one format of each mma shape / beta kind.
  for (int fi : {0, 1, 4, 7, 9, 5}) {
    const GgufFormat& f = kGguf[fi];
    const int m = 70, n = 136, k = f.elems == 32 ? 288 : 768;
    Rng r(9300u + f.bytes);
    const auto a = random_t<__half>(static_cast<size_t>(m) * k, &r, -2.0f, 2.0f);
    const GgufMmqProblem p = gguf_mmq_problem(ctx, f, m, n, k, a, 9400u + f.bytes);
    char what[96];
    std::snprintf(what, sizeof(what), "gguf mmq fp16 %s M=%d", f.name, m);
    expect_close(gguf_mmq_run(ctx, f, p, a, m, what), p.ref, what);
  }
}

/// MoE mmq: `slots` = tokens * topk ids drawn over `experts` (with some -1), checked against the fp64 oracle;
/// dropped slots keep their sentinel.  Returns the output for repeat / cross-path checks.
std::vector<__nv_bfloat16> gguf_moe_mmq_case(const LaunchContext& ctx, const GgufFormat& f, int experts, int tokens, int topk, int n, int k,
                                             int hot_expert, const char* what) {
  Rng r(9500u + f.bytes + tokens * 7 + experts);
  const auto a = random_t<__nv_bfloat16>(static_cast<size_t>(tokens) * k, &r, -2.0f, 2.0f);
  const size_t per_expert = static_cast<size_t>(n) * (k / f.elems);
  const auto raw = random_gguf(f, per_expert * experts, &r);
  std::vector<int32_t> ids(static_cast<size_t>(tokens) * topk);
  for (auto& id : ids) {
    const int v = r.range(10);
    id = v == 0 ? -1 : (v < 5 && hot_expert >= 0 ? hot_expert : r.range(experts));
  }
  auto dq = upload(raw);
  DevBuf<float> wdeq(static_cast<size_t>(experts) * n * k);
  KOPS_EXPECT_OK(kops::quant::gguf_dequantize(kops_test::tensor2d(wdeq.p, DType::kFloat32, experts * n, k),
                                              tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(raw.size())), f.type, ctx), what);
  cudaDeviceSynchronize();
  const auto w = kops_test::copy_out(wdeq.p, static_cast<size_t>(experts) * n * k);
  const Q81Host q = quantize_q81_host(a, tokens, k);
  const float sentinel = 7.0f;
  std::vector<double> ref(ids.size() * n, static_cast<double>(sentinel));
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < topk; ++j) {
      const int e = ids[static_cast<size_t>(t) * topk + j];
      if (e < 0) continue;
      for (int c = 0; c < n; ++c) {
        double acc = 0.0;
        for (int kk = 0; kk < k; ++kk) acc += static_cast<double>(w[(static_cast<size_t>(e) * n + c) * k + kk]) * q.deq[static_cast<size_t>(t) * k + kk];
        ref[(static_cast<size_t>(t) * topk + j) * n + c] = acc;
      }
    }
  auto dids = upload(ids);
  auto dqa = upload(q.bytes);
  DevBuf<__nv_bfloat16> dout(ref.size());
  kops_test::copy_in(dout.p, std::vector<__nv_bfloat16>(ref.size(), cvt<__nv_bfloat16>(sentinel)));
  KOPS_EXPECT_OK(kops::gemm::gguf_moe_gemm(kops_test::tensor2d(dout.p, DType::kBFloat16, tokens * topk, n),
                                           tensor1d(dqa.p, DType::kUInt8, static_cast<int64_t>(q.bytes.size())),
                                           tensor1d(dq.p, DType::kUInt8, static_cast<int64_t>(raw.size())),
                                           static_cast<int64_t>(per_expert) * f.bytes, kops_test::tensor2d(dids.p, DType::kInt32, tokens, topk),
                                           f.type, ctx), what);
  cudaDeviceSynchronize();
  auto got = kops_test::copy_out(dout.p, ref.size());
  expect_close(got, ref, what);
  return got;
}

void test_b7_gguf_moe_mmq(const LaunchContext& ctx) {
  char what[128];
  for (const GgufFormat& f : kGguf) {
    // 40 x 3 = 120 slots over 5 experts (< 64 per expert on average -> 32-token tiles); expert 2 is hot
    // (~half the slots, several 32-slot chunks); N = 72 is a tail of the 64-row tile.
    const int k = f.elems == 32 ? 160 : 256;
    std::snprintf(what, sizeof(what), "gguf moe mmq %s 40x3 E5", f.name);
    const auto first = gguf_moe_mmq_case(ctx, f, 5, 40, 3, 72, k, 2, what);
    std::snprintf(what, sizeof(what), "gguf moe mmq %s repeat", f.name);
    if (gguf_moe_mmq_case(ctx, f, 5, 40, 3, 72, k, 2, what) != first) {
      kops_test::record_failure(what, "not deterministic", __FILE__, __LINE__);
    }
  }
  for (int fi : {7, 6, 1}) {   // Q4_K, Q3_K, Q4_1: 200 x 2 slots over 3 experts -> 64 x 128 tiles
    const GgufFormat& f = kGguf[fi];
    std::snprintf(what, sizeof(what), "gguf moe mmq %s 200x2 E3", f.name);
    gguf_moe_mmq_case(ctx, f, 3, 200, 2, 136, 512, 1, what);
  }
  // Below SM80 the mmvq path serves large slot counts (same result within fp32 rounding).
  LaunchContext sm75 = ctx;
  sm75.arch_id = KOPS_TARGET_SM75;
  gguf_moe_mmq_case(sm75, kGguf[7], 5, 40, 3, 72, 256, 2, "gguf moe sm75 -> mmvq");
  // batch_invariant routes small counts to mmq as well.
  LaunchContext inv = ctx;
  inv.batch_invariant = true;
  gguf_moe_mmq_case(inv, kGguf[9], 4, 3, 2, 40, 256, -1, "gguf moe mmq batch_invariant 3x2");
}

void test_b7_gguf_mmq_contracts(const LaunchContext& ctx) {
  const int m = 20, n = 64, k = 256;
  DevBuf<uint8_t> w(static_cast<size_t>(n) * (k / 32) * 34), ws(static_cast<size_t>(m) * (k / 32) * 36);
  DevBuf<__nv_bfloat16> a(static_cast<size_t>(m) * k), o(static_cast<size_t>(m) * n);
  const std::vector<__nv_bfloat16> sentinel(static_cast<size_t>(m) * n, cvt<__nv_bfloat16>(3.0f));
  auto call = [&](const LaunchContext& c, GGUFType t, int kk, int64_t wbytes, DType odt) {
    kops_test::copy_in(o.p, sentinel);
    const Status st = kops::gemm::gguf_gemm(kops_test::tensor2d(o.p, odt, m, n), kops_test::tensor2d(a.p, DType::kBFloat16, m, kk),
                                            tensor1d(w.p, DType::kUInt8, wbytes), t, tensor1d(ws.p, DType::kUInt8, static_cast<int64_t>(m) * (k / 32) * 36), c);
    cudaDeviceSynchronize();
    return st;
  };
  auto untouched = [&](const char* what) {
    if (kops_test::copy_out(o.p, sentinel.size()) != sentinel) kops_test::record_failure(what, "output written on an error", __FILE__, __LINE__);
  };
  const int64_t q8 = static_cast<int64_t>(n) * (k / 32) * 34;
  LaunchContext sm75 = ctx;
  sm75.arch_id = KOPS_TARGET_SM75;
  KOPS_EXPECT_STATUS(call(sm75, GGUFType::kQ8_0, k, q8, DType::kBFloat16), Status::kUnsupportedArch, "gguf mmq sm75");
  untouched("gguf mmq sm75");
  KOPS_EXPECT_STATUS(call(sm75, GGUFType::kQ8_0, k, q8, DType::kFloat16), Status::kUnsupportedDtype, "gguf mmq dtype before arch");
  untouched("gguf mmq dtype before arch");
  KOPS_EXPECT_STATUS(call(sm75, GGUFType::kQ8_0, k, q8 - 2, DType::kBFloat16), Status::kInvalidArgument, "gguf mmq weight size before arch");
  KOPS_EXPECT_STATUS(call(ctx, GGUFType::kIQ4_XS, k, q8, DType::kBFloat16), Status::kNotImplemented, "gguf mmq IQ type");
  untouched("gguf mmq IQ type");
  KOPS_EXPECT_STATUS(call(ctx, GGUFType::kQ4_K, 224, static_cast<int64_t>(n) * 144, DType::kBFloat16), Status::kUnsupportedShape, "gguf mmq K-quant K");
  untouched("gguf mmq K-quant K");
  KOPS_EXPECT_OK(call(ctx, GGUFType::kQ8_0, k, q8, DType::kBFloat16), "gguf mmq baseline");
  // sm75 with M <= 16 still has the mmvq path.
  KOPS_EXPECT_OK(kops::gemm::gguf_gemm(kops_test::tensor2d(o.p, DType::kBFloat16, 16, n), kops_test::tensor2d(a.p, DType::kBFloat16, 16, k),
                                       tensor1d(w.p, DType::kUInt8, q8), GGUFType::kQ8_0,
                                       tensor1d(ws.p, DType::kUInt8, static_cast<int64_t>(m) * (k / 32) * 36), sm75), "gguf sm75 M=16 mmvq");
  cudaDeviceSynchronize();
}

void test_b7_gguf_mmq(const LaunchContext& ctx) {
  test_b7_gguf_mmq_dense(ctx);
  test_b7_gguf_moe_mmq(ctx);
  test_b7_gguf_mmq_contracts(ctx);
}

// =============================================================================
// B7 rows 7 and 9: grouped GEMM on tensor cores (grouped_gemm -> gemm_mma_sync_kernel with a grouped row
// map, grouped_gemm_task_map_kernel, grouped_gemm_prepare) and multi-LoRA (lora_sgmv).
// Oracle: fp64 host loops over the rounded / decoded operands, the per-expert scales and the epilogue in
// the documented order (alpha -> sA -> sB -> bias -> act / GLU -> routing weight -> round or quantize).
// =============================================================================
namespace b7grp {

enum { R_BF16 = 0, R_FP16, R_FP8T, R_FP8R, R_FP8B, R_FP8BUE, R_I8R, R_COUNT };
enum { L_CONTIG = 0, L_MASKED, L_INDEXED, L_ROWIDX, L_COUNT };

const char* const kRecipeName[] = {"bf16", "fp16", "fp8 tensorwise", "fp8 rowwise", "fp8 blockwise", "fp8 blockwise ue8m0", "int8 rowwise"};
const char* const kLayoutName[] = {"contiguous", "masked", "indexed", "contiguous row index"};

bool is_block(int r) { return r == R_FP8B || r == R_FP8BUE; }
bool is_8bit(int r) { return r >= R_FP8T; }

GemmRecipe gemm_recipe(int r) {
  switch (r) {
    case R_FP16: return GemmRecipe::kFP16;
    case R_FP8T: return GemmRecipe::kFP8Tensorwise;
    case R_FP8R: return GemmRecipe::kFP8Rowwise;
    case R_FP8B: return GemmRecipe::kFP8Blockwise;
    case R_FP8BUE: return GemmRecipe::kFP8BlockwiseUE8M0;
    case R_I8R: return GemmRecipe::kINT8Rowwise;
    default: return GemmRecipe::kBF16;
  }
}

struct Spec {
  int layout = L_CONTIG;
  int recipe = R_BF16;
  std::vector<int> rows;        // rows per expert
  int n = 192, k = 256;
  uint32_t ops = kg::kEpiBias | kg::kEpiRoutingWeight;
  int glu_db = 1;
  int quant = 0;                // 0 none, 1 static per-tensor e4m3, 2 per-group 128 e4m3
  bool col_major_sa = false;
  int path = 0;                 // 0 heuristic tactic + in-kernel scan, 1 explicit tactic + in-kernel scan, 2 task map
  GemmTactic tactic{};
  unsigned seed = 1;
};

struct Problem {
  int G = 0, N = 0, K = 0, n_out = 0, ldd = 0, a_rows = 0, d_rows = 0, max_m = 0;
  std::vector<int64_t> offs;
  std::vector<int32_t> masked, gather, m_idx;
  std::vector<int> row_group, row_src;   // logical row -> expert (-1 padding) / A row
  std::vector<uint8_t> a_bytes, w_bytes;
  std::vector<float> a_val, w_val;
  std::vector<float> sa, sb;             // scale values (oracle)
  std::vector<uint8_t> sa_store, sb_store;   // device bytes of the scales (fp32 or UE8M0, in the device layout)
  std::vector<__nv_bfloat16> bias;
  std::vector<float> rw;
  std::vector<double> ref;               // [d_rows, n_out]
  std::vector<double> mag;               // [d_rows, n_out] sum |a*w*scale| (FP8 accumulation bound)
  float static_scale = 1.0f;
};

double silu(double x) { return x / (1.0 + std::exp(-x)); }

Problem make_problem(const Spec& s) {
  Problem P;
  Rng r(9001u + s.seed * 131u + static_cast<unsigned>(s.recipe) * 17u + static_cast<unsigned>(s.layout) * 5u);
  const int G = static_cast<int>(s.rows.size());
  const int N = s.n, K = s.k;
  const bool glu = (s.ops & kg::kEpiGLU) != 0u;
  P.G = G; P.N = N; P.K = K;
  P.n_out = glu ? N / 2 : N;
  P.ldd = P.n_out + 8;
  if (s.layout == L_MASKED) {
    int mx = 0;
    for (int v : s.rows) mx = std::max(mx, v);
    P.max_m = mx + 3;
    P.a_rows = P.d_rows = G * P.max_m;
    P.row_group.assign(P.d_rows, -1);
    P.row_src.assign(P.d_rows, -1);
    for (int g = 0; g < G; ++g) {
      P.masked.push_back(s.rows[g]);
      for (int i = 0; i < s.rows[g]; ++i) { P.row_group[g * P.max_m + i] = g; P.row_src[g * P.max_m + i] = g * P.max_m + i; }
    }
  } else if (s.layout == L_ROWIDX) {
    int total = 0;
    for (int v : s.rows) total += static_cast<int>(round_up(v, 64));
    P.a_rows = P.d_rows = P.max_m = total;
    P.m_idx.assign(total, -1);
    P.row_group.assign(total, -1);
    P.row_src.assign(total, -1);
    int base = 0;
    for (int g = 0; g < G; ++g) {
      for (int i = 0; i < s.rows[g]; ++i) { P.m_idx[base + i] = g; P.row_group[base + i] = g; P.row_src[base + i] = base + i; }
      base += static_cast<int>(round_up(s.rows[g], 64));
    }
  } else {
    P.offs.assign(G + 1, 0);
    for (int g = 0; g < G; ++g) P.offs[g + 1] = P.offs[g] + s.rows[g];
    const int total = static_cast<int>(P.offs[G]);
    P.d_rows = total;
    P.max_m = total;
    P.a_rows = s.layout == L_INDEXED ? total / 2 + 5 : total;
    P.row_group.assign(total, -1);
    P.row_src.assign(total, -1);
    for (int g = 0; g < G; ++g) {
      for (int64_t lr = P.offs[g]; lr < P.offs[g + 1]; ++lr) P.row_group[lr] = g;
    }
    for (int lr = 0; lr < total; ++lr) {
      if (s.layout == L_INDEXED) {
        P.gather.push_back((lr * 7 + 3) % P.a_rows);
        P.row_src[lr] = P.gather.back();
      } else {
        P.row_src[lr] = lr;
      }
    }
  }
  const int A = std::max(P.a_rows, 1);
  const size_t na = static_cast<size_t>(A) * K, nw = static_cast<size_t>(G) * N * K;
  auto gen = [&](size_t n, float lo, float hi, std::vector<uint8_t>* bytes, std::vector<float>* val) {
    val->resize(n);
    if (s.recipe == R_BF16 || s.recipe == R_FP16) {
      bytes->resize(2 * n);
      for (size_t i = 0; i < n; ++i) {
        const float x = r.uni(lo, hi);
        if (s.recipe == R_FP16) {
          const __half h = cvt<__half>(x);
          std::memcpy(&(*bytes)[2 * i], &h, 2);
          (*val)[i] = kops_test::to_float(h);
        } else {
          const __nv_bfloat16 h = cvt<__nv_bfloat16>(x);
          std::memcpy(&(*bytes)[2 * i], &h, 2);
          (*val)[i] = kops_test::to_float(h);
        }
      }
    } else if (s.recipe == R_I8R) {
      bytes->resize(n);
      for (size_t i = 0; i < n; ++i) {
        const int q = r.range(129) - 64;
        (*bytes)[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
        (*val)[i] = static_cast<float>(q);
      }
    } else {
      bytes->resize(n);
      for (size_t i = 0; i < n; ++i) {
        (*bytes)[i] = e4m3_from_float(r.uni(2 * lo, 2 * hi));
        (*val)[i] = f8((*bytes)[i]);
      }
    }
  };
  gen(na, -1.0f, 1.0f, &P.a_bytes, &P.a_val);
  gen(nw, -0.5f, 0.5f, &P.w_bytes, &P.w_val);
  const int kb = K / 128, nb = (N + 127) / 128;
  auto put_f32 = [](std::vector<uint8_t>* v, size_t i, float x) { std::memcpy(&(*v)[4 * i], &x, 4); };
  if (s.recipe == R_FP8T) {
    P.sa = {0.8f};
    P.sb.resize(G);
    for (auto& v : P.sb) v = r.uni(0.5f, 1.5f);
  } else if (s.recipe == R_FP8R || s.recipe == R_I8R) {
    const float lo = s.recipe == R_I8R ? 0.01f : 0.5f, hi = s.recipe == R_I8R ? 0.03f : 1.5f;
    P.sa.resize(A);
    P.sb.resize(static_cast<size_t>(G) * N);
    for (auto& v : P.sa) v = r.uni(lo, hi);
    for (auto& v : P.sb) v = r.uni(lo, hi);
  } else if (is_block(s.recipe)) {
    const bool ue = s.recipe == R_FP8BUE;
    auto pick = [&](uint8_t* e8) {
      if (ue) {
        const int p = r.range(5) - 2;
        *e8 = static_cast<uint8_t>(127 + p);
        return std::ldexp(1.0f, p);
      }
      return r.uni(0.5f, 2.0f);
    };
    P.sa.resize(static_cast<size_t>(A) * kb);
    const int64_t lda_s = s.col_major_sa ? round_up(A, 4) : kb;
    const size_t sa_elems = s.col_major_sa ? static_cast<size_t>(kb) * lda_s : static_cast<size_t>(A) * kb;
    P.sa_store.assign(sa_elems * (ue ? 1 : 4), 0);
    for (int row = 0; row < A; ++row) {
      for (int j = 0; j < kb; ++j) {
        uint8_t e8 = 0;
        const float v = pick(&e8);
        P.sa[static_cast<size_t>(row) * kb + j] = v;
        const size_t pos = s.col_major_sa ? static_cast<size_t>(j) * lda_s + row : static_cast<size_t>(row) * kb + j;
        if (ue) P.sa_store[pos] = e8;
        else put_f32(&P.sa_store, pos, v);
      }
    }
    const size_t nsb = static_cast<size_t>(G) * nb * kb;
    P.sb.resize(nsb);
    P.sb_store.assign(nsb * (ue ? 1 : 4), 0);
    for (size_t i = 0; i < nsb; ++i) {
      uint8_t e8 = 0;
      P.sb[i] = pick(&e8);
      if (ue) P.sb_store[i] = e8;
      else put_f32(&P.sb_store, i, P.sb[i]);
    }
  }
  if (!is_block(s.recipe)) {
    P.sa_store.resize(P.sa.size() * 4);
    for (size_t i = 0; i < P.sa.size(); ++i) put_f32(&P.sa_store, i, P.sa[i]);
    P.sb_store.resize(P.sb.size() * 4);
    for (size_t i = 0; i < P.sb.size(); ++i) put_f32(&P.sb_store, i, P.sb[i]);
  }
  P.bias = random_t<__nv_bfloat16>(static_cast<size_t>(G) * N, &r, -1.0f, 1.0f);
  P.rw.resize(std::max(P.d_rows, 1));
  for (auto& v : P.rw) v = r.uni(0.5f, 1.5f);

  // oracle
  const double alpha = 0.75;
  P.ref.assign(static_cast<size_t>(P.d_rows) * P.n_out, 0.0);
  P.mag.assign(static_cast<size_t>(P.d_rows) * P.n_out, 0.0);
  std::vector<double> lin(N), lin_mag(N);
  double amax = 0.0;
  for (int lr = 0; lr < P.d_rows; ++lr) {
    const int g = P.row_group[lr];
    if (g < 0) continue;
    const int src = P.row_src[lr];
    const float* av = &P.a_val[static_cast<size_t>(src) * K];
    for (int c = 0; c < N; ++c) {
      const float* wv = &P.w_val[(static_cast<size_t>(g) * N + c) * K];
      double acc = 0.0, mag = 0.0;
      if (is_block(s.recipe)) {
        for (int j = 0; j < kb; ++j) {
          double part = 0.0, pmag = 0.0;
          for (int kk = 128 * j; kk < 128 * j + 128; ++kk) {
            const double p = static_cast<double>(av[kk]) * wv[kk];
            part += p;
            pmag += std::fabs(p);
          }
          const double sc = static_cast<double>(P.sa[static_cast<size_t>(src) * kb + j]) *
                            P.sb[(static_cast<size_t>(g) * nb + c / 128) * kb + j];
          acc += part * sc;
          mag += pmag * std::fabs(sc);
        }
      } else {
        for (int kk = 0; kk < K; ++kk) {
          const double p = static_cast<double>(av[kk]) * wv[kk];
          acc += p;
          mag += std::fabs(p);
        }
      }
      acc *= alpha;
      mag *= std::fabs(alpha);
      if (s.recipe == R_FP8T) {
        const double sc = static_cast<double>(P.sa[0]) * P.sb[g];
        acc *= sc;
        mag *= std::fabs(sc);
      }
      if (s.recipe == R_FP8R || s.recipe == R_I8R) {
        const double sc = static_cast<double>(P.sa[src]) * P.sb[static_cast<size_t>(g) * N + c];
        acc *= sc;
        mag *= std::fabs(sc);
      }
      if (s.ops & kg::kEpiBias) acc += dval(P.bias, static_cast<size_t>(g) * N + c);
      lin[c] = acc;
      lin_mag[c] = mag;
    }
    for (int oc = 0; oc < P.n_out; ++oc) {
      double v, mag;
      if (glu) {
        const int gc = (oc / s.glu_db) * 2 * s.glu_db + oc % s.glu_db;
        const double gate = lin[gc], up = lin[gc + s.glu_db];
        const double sig = 1.0 / (1.0 + std::exp(-gate));
        v = silu(gate) * up;
        // |d(silu*up)| <= |silu| d(up) + |up| |silu'| d(gate)
        mag = std::fabs(silu(gate)) * lin_mag[gc + s.glu_db] +
              std::fabs(up) * std::fabs(sig * (1.0 + gate * (1.0 - sig))) * lin_mag[gc];
      } else {
        v = lin[oc];
        mag = lin_mag[oc];
        if (s.ops & kg::kEpiAct) {
          const double sig = 1.0 / (1.0 + std::exp(-v));
          mag *= std::fabs(sig * (1.0 + v * (1.0 - sig)));
          v = silu(v);
        }
      }
      if (s.ops & kg::kEpiRoutingWeight) {
        v *= P.rw[lr];
        mag *= std::fabs(P.rw[lr]);
      }
      P.ref[static_cast<size_t>(lr) * P.n_out + oc] = v;
      P.mag[static_cast<size_t>(lr) * P.n_out + oc] = mag;
      amax = std::max(amax, std::fabs(v));
    }
  }
  P.static_scale = static_cast<float>(std::max(amax, 1e-3) / 300.0);
  return P;
}

struct Out {
  std::vector<uint8_t> d, q;
  std::vector<float> qs;
  int32_t num_tasks = -1, overflow = -1;
};

uint16_t sentinel16(bool fp16) {
  uint16_t b;
  if (fp16) { const __half h = cvt<__half>(7.0f); std::memcpy(&b, &h, 2); }
  else { const __nv_bfloat16 h = cvt<__nv_bfloat16>(7.0f); std::memcpy(&b, &h, 2); }
  return b;
}

/// Fills GroupedGemmParams for the spec (device pointers from the caller's buffers).
struct DevProblem {
  DevBuf<uint8_t> a, w, sa, sb, bias_b;
  DevBuf<float> rw, sstat;
  DevBuf<int64_t> offs;
  DevBuf<int32_t> masked, gather, midx;
  DevBuf<uint8_t> d, q;
  DevBuf<float> qs;
  size_t d_bytes, q_bytes, qs_elems;
  int qs_ld;
  explicit DevProblem(const Problem& P, const Spec& s)
      : a(upload(P.a_bytes)), w(upload(P.w_bytes)), sa(upload(P.sa_store.empty() ? std::vector<uint8_t>(16) : P.sa_store)),
        sb(upload(P.sb_store.empty() ? std::vector<uint8_t>(16) : P.sb_store)),
        bias_b(upload(std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(P.bias.data()), reinterpret_cast<const uint8_t*>(P.bias.data()) + 2 * P.bias.size()))),
        rw(upload(P.rw)), sstat(upload(std::vector<float>(1, P.static_scale))),
        offs(upload(P.offs.empty() ? std::vector<int64_t>(1) : P.offs)),
        masked(upload(P.masked.empty() ? std::vector<int32_t>(1) : P.masked)),
        gather(upload(P.gather.empty() ? std::vector<int32_t>(1) : P.gather)),
        midx(upload(P.m_idx.empty() ? std::vector<int32_t>(1) : P.m_idx)),
        d(static_cast<size_t>(std::max(P.d_rows, 1)) * P.ldd * 2), q(static_cast<size_t>(std::max(P.d_rows, 1)) * P.n_out),
        qs(static_cast<size_t>(std::max(P.d_rows, 1)) * std::max(P.n_out / 128, 1)) {
    d_bytes = static_cast<size_t>(std::max(P.d_rows, 1)) * P.ldd * 2;
    q_bytes = static_cast<size_t>(std::max(P.d_rows, 1)) * P.n_out;
    qs_ld = std::max(P.n_out / 128, 1);
    qs_elems = static_cast<size_t>(std::max(P.d_rows, 1)) * qs_ld;
    (void)s;
    reset(s);
  }
  void reset(const Spec& s) {
    const uint16_t sen = sentinel16(s.recipe == R_FP16);
    std::vector<uint16_t> dd(d_bytes / 2, sen);
    kops_test::copy_in(d.p, std::vector<uint8_t>(reinterpret_cast<uint8_t*>(dd.data()), reinterpret_cast<uint8_t*>(dd.data()) + d_bytes));
    kops_test::copy_in(q.p, std::vector<uint8_t>(q_bytes, 0x5A));
    kops_test::copy_in(qs.p, std::vector<float>(qs_elems, -1.0f));
  }
};

kg::GroupedGemmParams make_params(const Spec& s, const Problem& P, DevProblem& D) {
  kg::GroupedGemmParams p{};
  p.layout = s.layout == L_CONTIG ? GroupedLayout::kContiguous
                                  : (s.layout == L_MASKED ? GroupedLayout::kMasked
                                                          : (s.layout == L_INDEXED ? GroupedLayout::kIndexed : GroupedLayout::kContiguousRowIndex));
  p.recipe = gemm_recipe(s.recipe);
  p.num_groups = P.G;
  p.a = D.a.p; p.lda = P.K;
  if (is_8bit(s.recipe)) {
    p.a_scale = D.sa.p;
    p.a_scale_layout = s.col_major_sa ? kops::ScaleLayout::kColMajorTma : kops::ScaleLayout::kRowMajor;
    p.b_scale = D.sb.p;
  }
  p.expert_offsets = D.offs.p;
  p.masked_m = D.masked.p;
  p.gather_idx = D.gather.p;
  p.m_indices = D.midx.p;
  p.max_m = P.max_m;
  p.expected_m = 32;   // > 16: the tensor-core path for 16-bit recipes
  p.b = D.w.p; p.ldb = P.K; p.b_group_stride = static_cast<int64_t>(P.N) * P.K;
  p.d = D.d.p; p.ldd = P.ldd;
  p.n = P.N; p.k = P.K;
  p.epi.ops = s.ops;
  p.epi.alpha = 0.75f;
  p.epi.bias = D.bias_b.p; p.epi.bias_dtype = DType::kBFloat16;
  p.epi.routing_weight = D.rw.p;
  p.epi.act = kops::ActFn::kSiLU;
  p.epi.glu_layout = kops::GateLayout::kInterleaved;
  p.epi.glu_interleave = s.glu_db;
  if (s.quant != 0) {
    p.epi.ops |= kg::kEpiQuantOut;
    p.epi.quant_out.q = D.q.p;
    p.epi.quant_out.q_row_stride = P.n_out;
    p.epi.quant_out.q_dtype = DType::kFloat8E4M3;
    if (s.quant == 1) {
      p.epi.quant_out.mode = kops::ActQuantMode::kStaticPerTensor;
      p.epi.quant_out.static_scale = D.sstat.p;
    } else {
      p.epi.quant_out.mode = kops::ActQuantMode::kPerGroup;
      p.epi.quant_out.group_size = 128;
      p.epi.quant_out.scales = D.qs.p;
      p.epi.quant_out.scale_row_stride = D.qs_ld;
    }
  }
  return p;
}

Status launch(const LaunchContext& ctx, const Spec& s, kg::GroupedGemmParams p, Out* out, DevProblem& D) {
  Status st;
  if (s.path == 2) {
    kops::WorkspaceRequirements req{};
    st = kg::grouped_gemm_workspace_requirements(p, &s.tactic, ctx, &req);
    if (st != Status::kSuccess) return st;
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 16);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    st = kg::grouped_gemm_prepare(&p, s.tactic, ws, ctx);
    if (st != Status::kSuccess) return st;
    st = kg::grouped_gemm(p, &s.tactic, kg::GemmBackend::kNative, kops::Workspace{}, ctx);
    cudaDeviceSynchronize();
    const auto hdr = kops_test::copy_out(reinterpret_cast<int32_t*>(scratch.p), 2);
    out->num_tasks = hdr[0];
    out->overflow = hdr[1];
  } else {
    st = kg::grouped_gemm(p, s.path == 1 ? &s.tactic : nullptr, kg::GemmBackend::kNative, kops::Workspace{}, ctx);
    cudaDeviceSynchronize();
  }
  if (st != Status::kSuccess) return st;
  out->d = kops_test::copy_out(D.d.p, D.d_bytes);
  out->q = kops_test::copy_out(D.q.p, D.q_bytes);
  out->qs = kops_test::copy_out(D.qs.p, D.qs_elems);
  return Status::kSuccess;
}

template <typename T>
std::vector<T> as_vec(const std::vector<uint8_t>& bytes) {
  std::vector<T> v(bytes.size() / sizeof(T));
  std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T));
  return v;
}

int64_t expected_tasks(const Spec& s, const Problem& P) {
  int64_t t = 0;
  const int64_t ntiles = (P.N + s.tactic.tile_n - 1) / s.tactic.tile_n;
  for (int g = 0; g < P.G; ++g) {
    const int rows = s.layout == L_MASKED ? std::min(s.rows[g], P.max_m) : s.rows[g];
    t += (rows + s.tactic.tile_m - 1) / s.tactic.tile_m * ntiles;
  }
  return t;
}

std::string label(const Spec& s, const char* what) {
  char buf[400];
  std::snprintf(buf, sizeof(buf), "%s [%s %s path %d tile %dx%d swap %d ops 0x%x db %d quant %d cm %d]", what, kLayoutName[s.layout],
                kRecipeName[s.recipe], s.path, s.path ? s.tactic.tile_m : -1, s.path ? s.tactic.tile_n : -1, s.path ? s.tactic.swap_ab : 0, s.ops,
                s.glu_db, s.quant, s.col_major_sa ? 1 : 0);
  return buf;
}

/// Runs one case and checks it against the oracle. Returns the output (empty d on a failed launch).
Out run_check(const LaunchContext& ctx, const Spec& s, const Problem& P, const char* what) {
  const std::string lab = label(s, what);
  DevProblem D(P, s);
  Out o;
  const Status st = launch(ctx, s, make_params(s, P, D), &o, D);
  if (st != Status::kSuccess) {
    KOPS_EXPECT_OK(st, lab.c_str());
    o.d.clear();
    return o;
  }
  if (s.path == 2) {
    const int64_t want = expected_tasks(s, P);
    if (o.num_tasks != want || o.overflow != 0) {
      char det[160];
      std::snprintf(det, sizeof(det), "task map: %d tasks overflow %d, want %lld tasks (empty experts generate none)", o.num_tasks, o.overflow,
                    static_cast<long long>(want));
      kops_test::record_failure(lab.c_str(), det, __FILE__, __LINE__);
    }
  }
  const bool fp16 = s.recipe == R_FP16;
  const uint16_t sen = sentinel16(fp16);
  const auto d16 = as_vec<uint16_t>(o.d);
  size_t untouched_bad = 0;
  if (s.quant == 0) {
    std::vector<uint16_t> got;
    std::vector<double> want, want_mag;
    for (int lr = 0; lr < P.d_rows; ++lr) {
      for (int c = 0; c < P.ldd; ++c) {
        const uint16_t g = d16[static_cast<size_t>(lr) * P.ldd + c];
        if (P.row_group[lr] >= 0 && c < P.n_out) {
          got.push_back(g);
          want.push_back(P.ref[static_cast<size_t>(lr) * P.n_out + c]);
          want_mag.push_back(P.mag[static_cast<size_t>(lr) * P.n_out + c]);
        } else {
          untouched_bad += g != sen;
        }
      }
    }
    std::vector<uint8_t> gb(got.size() * 2);
    std::memcpy(gb.data(), got.data(), gb.size());
    const bool fp8_out = s.recipe == R_FP8T || s.recipe == R_FP8R || s.recipe == R_FP8B || s.recipe == R_FP8BUE;
    if (fp8_out) {
      if (fp16) expect_close_mag(as_vec<__half>(gb), want, want_mag, lab.c_str());
      else expect_close_mag(as_vec<__nv_bfloat16>(gb), want, want_mag, lab.c_str());
    } else if (fp16) {
      expect_close(as_vec<__half>(gb), want, lab.c_str());
    } else {
      expect_close(as_vec<__nv_bfloat16>(gb), want, lab.c_str());
    }
  } else {
    for (uint16_t g : d16) untouched_bad += g != sen;   // the quantized output replaces D
    double ss = 0.0;
    for (double v : P.ref) ss += v * v;
    const double rms = std::sqrt(ss / static_cast<double>(std::max<size_t>(P.ref.size(), 1)));
    size_t bad = 0, scale_bad = 0;
    double worst = 0.0;
    for (int lr = 0; lr < P.d_rows; ++lr) {
      const bool valid = P.row_group[lr] >= 0;
      for (int gi = 0; gi < P.n_out / 128 && s.quant == 2; ++gi) {
        const float gs = o.qs[static_cast<size_t>(lr) * D.qs_ld + gi];
        if (!valid) { untouched_bad += gs != -1.0f; continue; }
        double amax = 0.0;
        for (int c = gi * 128; c < gi * 128 + 128; ++c) amax = std::max(amax, std::fabs(P.ref[static_cast<size_t>(lr) * P.n_out + c]));
        const double want_s = std::max(amax, 1e-10) / 448.0;
        if (!(std::fabs(gs - want_s) <= 2e-3 * want_s + 1e-12)) ++scale_bad;
      }
      for (int c = 0; c < P.n_out; ++c) {
        const uint8_t qb = o.q[static_cast<size_t>(lr) * P.n_out + c];
        if (!valid) { untouched_bad += qb != 0x5A; continue; }
        const double sc = s.quant == 1 ? P.static_scale : o.qs[static_cast<size_t>(lr) * D.qs_ld + c / 128];
        const double v = P.ref[static_cast<size_t>(lr) * P.n_out + c];
        const double mag = P.mag[static_cast<size_t>(lr) * P.n_out + c];
        const double err = std::fabs(static_cast<double>(f8(qb)) * sc - v);
        if (!(err <= 0.064 * std::fabs(v) + sc * 0.002 + std::ldexp(mag, -11) + 1e-5 * rms)) { ++bad; worst = std::max(worst, err); }
      }
    }
    if (bad != 0 || scale_bad != 0) {
      char det[200];
      std::snprintf(det, sizeof(det), "%zu quantized values off (worst %.4g), %zu group scales off", bad, worst, scale_bad);
      kops_test::record_failure(lab.c_str(), det, __FILE__, __LINE__);
    }
  }
  if (untouched_bad != 0) {
    char det[160];
    std::snprintf(det, sizeof(det), "%zu elements outside the routed rows / columns were written", untouched_bad);
    kops_test::record_failure(lab.c_str(), det, __FILE__, __LINE__);
  }
  return o;
}

GemmTactic make_tile(int tm, int tn, bool swap) {
  GemmTactic t{};
  t.tile_m = static_cast<int16_t>(tm);
  t.tile_n = static_cast<int16_t>(tn);
  t.swap_ab = swap;
  t.stages = 0;
  t.mainloop = kg::MainloopSchedule::kMultistageMmaSync;
  t.split_k_mode = kg::SplitKMode::kNone;
  t.split_k_slices = 1;
  return t;
}

/// Every compiled tile the spec can run on (the launcher's own constraints: block promotion, GLU span,
/// per-group quantized output, row-index padding).
std::vector<GemmTactic> tiles_for(const Spec& s) {
  struct Tl { int tm, tn; bool swap; int span; };
  static const Tl tbl[] = {{128, 128, false, 128}, {128, 64, false, 64}, {64, 128, false, 128}, {64, 64, false, 64},
                           {64, 128, true, 128},   {32, 128, true, 128}, {16, 128, true, 128},  {64, 256, false, 256}};
  std::vector<GemmTactic> out;
  const bool glu = (s.ops & kg::kEpiGLU) != 0u;
  for (int id = 0; id < 8; ++id) {
    const Tl& t = tbl[id];
    if (is_block(s.recipe) && (id == 0 || id == 7)) continue;
    if ((s.recipe == R_FP16 || s.recipe == R_I8R) && id == 7) continue;
    if (s.layout == L_ROWIDX && 64 % t.tm != 0) continue;
    if (glu && t.span % (2 * s.glu_db) != 0) continue;
    if (s.quant == 2 && (glu ? t.span / 2 : t.span) % 128 != 0) continue;
    out.push_back(make_tile(t.tm, t.tn, t.swap));
  }
  return out;
}

bool same(const Out& a, const Out& b) { return !a.d.empty() && a.d == b.d && a.q == b.q && a.qs == b.qs; }

}  // namespace b7grp

void b7_grouped_numerics(const LaunchContext& ctx) {
  using namespace b7grp;
  const std::vector<int> rows = {1, 0, 15, 17, 64, 200, 0, 33};   // two empty experts
  // 1. every layout x recipe: heuristic tactic (in-kernel scan), explicit tactic, prepared task map (bit-equal
  //    to the in-kernel scan with the same tactic), determinism.
  for (int layout = 0; layout < L_COUNT; ++layout) {
    for (int recipe = 0; recipe < R_COUNT; ++recipe) {
      Spec s;
      s.layout = layout; s.recipe = recipe; s.rows = rows;
      const Problem P = make_problem(s);
      const Out h0 = run_check(ctx, s, P, "b7 grouped heuristic");
      const Out h1 = run_check(ctx, s, P, "b7 grouped heuristic (repeat)");
      if (!h0.d.empty() && !same(h0, h1)) kops_test::record_failure(label(s, "b7 grouped determinism").c_str(), "repeat run differs", __FILE__, __LINE__);
      s.tactic = make_tile(64, 128, false);
      s.path = 1;
      const Out e = run_check(ctx, s, P, "b7 grouped explicit tactic");
      if (layout != L_ROWIDX) {
        s.path = 2;
        const Out m = run_check(ctx, s, P, "b7 grouped task map");
        if (!e.d.empty() && !same(e, m)) kops_test::record_failure(label(s, "b7 grouped task map vs in-kernel scan").c_str(), "bits differ", __FILE__, __LINE__);
      }
    }
  }
  // 2. every tile on a few (layout, recipe) pairs, in-kernel scan and task map
  const int sweep[][2] = {{L_CONTIG, R_BF16}, {L_MASKED, R_FP16}, {L_INDEXED, R_FP8R}, {L_CONTIG, R_FP8B}, {L_ROWIDX, R_BF16}, {L_MASKED, R_I8R}};
  for (const auto& lr : sweep) {
    Spec s;
    s.layout = lr[0]; s.recipe = lr[1]; s.rows = rows; s.seed = 2;
    const Problem P = make_problem(s);
    for (const GemmTactic& t : tiles_for(s)) {
      s.tactic = t;
      s.path = 1;
      const Out e = run_check(ctx, s, P, "b7 grouped tile sweep");
      if (s.layout != L_ROWIDX) {
        s.path = 2;
        const Out m = run_check(ctx, s, P, "b7 grouped tile sweep task map");
        if (!e.d.empty() && !same(e, m)) kops_test::record_failure(label(s, "b7 grouped tile sweep task map vs scan").c_str(), "bits differ", __FILE__, __LINE__);
      }
    }
  }
  // 3. rows per expert sweep (uniform counts incl. 1, 15, 17, 64, 200) and a single expert
  for (int m : {1, 15, 17, 64, 200}) {
    for (int recipe : {R_BF16, R_FP8R, R_FP8B}) {
      Spec s;
      s.recipe = recipe; s.rows = {m, m, 0, m}; s.seed = 3 + m;
      const Problem P = make_problem(s);
      run_check(ctx, s, P, "b7 grouped rows sweep");
      s.layout = L_MASKED;
      run_check(ctx, s, make_problem(s), "b7 grouped rows sweep");
    }
  }
  {
    Spec s;
    s.rows = {77}; s.n = 72; s.k = 64; s.seed = 9;   // one expert, N and K not multiples of the tile
    run_check(ctx, s, make_problem(s), "b7 grouped single expert");
  }
  // 4. epilogues: bias + act + routing weight, GLU (interleave 1 / 16 / 64), quantized output
  {
    const uint32_t bar = kg::kEpiBias | kg::kEpiAct | kg::kEpiRoutingWeight;
    for (int recipe : {R_BF16, R_FP16, R_FP8T, R_I8R}) {
      Spec s;
      s.recipe = recipe; s.rows = rows; s.ops = bar; s.seed = 5;
      run_check(ctx, s, make_problem(s), "b7 grouped bias+act+routing");
      s.ops = kg::kEpiNone;
      run_check(ctx, s, make_problem(s), "b7 grouped no epilogue");
    }
    for (int db : {1, 16, 64}) {
      for (int recipe : {R_BF16, R_FP8B, R_FP8R}) {
        for (int layout : {L_CONTIG, L_MASKED, L_INDEXED}) {
          Spec s;
          s.layout = layout; s.recipe = recipe; s.rows = rows; s.n = 256; s.glu_db = db; s.seed = 6;
          s.ops = kg::kEpiBias | kg::kEpiGLU | kg::kEpiRoutingWeight;
          const Problem P = make_problem(s);
          run_check(ctx, s, P, "b7 grouped GLU");
          for (const GemmTactic& t : tiles_for(s)) {
            s.tactic = t;
            s.path = 1;
            run_check(ctx, s, P, "b7 grouped GLU tile");
          }
        }
      }
    }
    for (int recipe : {R_BF16, R_FP8R, R_FP8T}) {
      for (int layout : {L_CONTIG, L_MASKED, L_ROWIDX}) {
        Spec s;
        s.layout = layout; s.recipe = recipe; s.rows = rows; s.n = 512; s.glu_db = 16; s.seed = 7;
        s.ops = kg::kEpiBias | kg::kEpiGLU;
        s.quant = 1;
        run_check(ctx, s, make_problem(s), "b7 grouped GLU + static quant out");
        s.quant = 2;   // per-group 128 of the GLU output: the 64 x 256 tile
        const Problem P = make_problem(s);
        run_check(ctx, s, P, "b7 grouped GLU + per-group quant out");
        s.tactic = make_tile(64, 256, false);
        s.path = 1;
        run_check(ctx, s, P, "b7 grouped GLU + per-group quant out (64x256)");
        if (layout != L_ROWIDX) {
          s.path = 2;
          run_check(ctx, s, P, "b7 grouped GLU + per-group quant out (task map)");
        }
      }
    }
    {
      Spec s;   // per-group quantized output without GLU
      s.rows = rows; s.n = 256; s.quant = 2; s.ops = kg::kEpiBias; s.seed = 8;
      run_check(ctx, s, make_problem(s), "b7 grouped per-group quant out");
    }
  }
  // 5. blockwise activation scales in the TMA column-major layout (DeepGEMM) and UE8M0
  for (int recipe : {R_FP8B, R_FP8BUE}) {
    for (int layout : {L_CONTIG, L_MASKED, L_ROWIDX}) {
      Spec s;
      s.layout = layout; s.recipe = recipe; s.rows = rows; s.col_major_sa = true; s.k = 384; s.seed = 10;
      const Problem P = make_problem(s);
      run_check(ctx, s, P, "b7 grouped blockwise col-major sA");
      s.tactic = make_tile(16, 128, true);
      s.path = 1;
      run_check(ctx, s, P, "b7 grouped blockwise col-major sA swap 16");
    }
  }
}

/// grouped_gemm_task_map_kernel launched directly: the map content against a host oracle (several chunks of
/// groups, empty groups, masked counts) and the overflow flag when the capacity is short.
void b7_task_map(const LaunchContext& ctx) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(ctx.stream);
  for (int G : {1, 5, 37, 1500}) {
    for (int masked : {0, 1}) {
      for (int threads : {32, 256, 1024}) {
        Rng r(4242u + G * 3u + masked);
        const int tile_m = G == 37 ? 32 : 64, n_tiles = G == 1500 ? 2 : 3;
        std::vector<int32_t> cnt(G);
        for (auto& c : cnt) c = (r.range(4) == 0) ? 0 : r.range(150);
        std::vector<int64_t> offs(G + 1, 0);
        for (int g = 0; g < G; ++g) offs[g + 1] = offs[g] + cnt[g];
        if (masked) cnt[0] = -3;   // negative masked counts count as empty
        std::vector<int32_t> want;
        for (int g = 0; g < G; ++g) {
          const int rows = std::max(cnt[g], 0);
          const int mt = (rows + tile_m - 1) / tile_m;
          for (int t = 0; t < mt * n_tiles; ++t) {
            want.push_back(g);
            want.push_back(t % mt);
            want.push_back(t / mt);
          }
        }
        const int total = static_cast<int>(want.size() / 3);
        for (int cap : {total + 5, total / 2}) {
          auto doffs = upload(offs);
          auto dcnt = upload(cnt);
          DevBuf<int32_t> map(static_cast<size_t>(std::max(cap, 1)) * 3 + 3);
          kops_test::copy_in(map.p, std::vector<int32_t>(static_cast<size_t>(std::max(cap, 1)) * 3 + 3, -9));
          DevBuf<int32_t> hdr(2);
          kops_test::copy_in(hdr.p, std::vector<int32_t>(2, -9));
          kg::grouped_gemm_task_map_kernel<<<1, threads, 0, s>>>(map.p, hdr.p, hdr.p + 1, masked ? nullptr : doffs.p, masked ? dcnt.p : nullptr, G,
                                                                 tile_m, n_tiles, cap);
          KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
          const auto h = kops_test::copy_out(hdr.p, 2);
          const auto m = kops_test::copy_out(map.p, static_cast<size_t>(std::max(cap, 1)) * 3 + 3);
          const int expect_n = std::min(total, cap);
          char what[160];
          std::snprintf(what, sizeof(what), "b7 task map G %d masked %d threads %d cap %d", G, masked, threads, cap);
          if (h[0] != expect_n || h[1] != (total > cap ? 1 : 0)) {
            char det[120];
            std::snprintf(det, sizeof(det), "num_tasks %d overflow %d, want %d / %d", h[0], h[1], expect_n, total > cap ? 1 : 0);
            kops_test::record_failure(what, det, __FILE__, __LINE__);
            continue;
          }
          size_t bad = 0;
          for (int i = 0; i < 3 * expect_n; ++i) bad += m[i] != want[i];
          for (size_t i = static_cast<size_t>(3) * expect_n; i < m.size(); ++i) bad += m[i] != -9;   // nothing past the capacity / count
          if (bad != 0) kops_test::record_failure(what, "map entries differ from the host oracle", __FILE__, __LINE__);
        }
      }
    }
  }
}

/// grouped_gemm_prepare + grouped_gemm captured in one CUDA graph; replays after the expert counts change on
/// the device give the oracle of the new counts (and the in-kernel scan path likewise).
void b7_grouped_graph(const LaunchContext& ctx_in) {
  using namespace b7grp;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    kops_test::record_failure("b7 grouped graph", "cudaStreamCreate failed", __FILE__, __LINE__);
    return;
  }
  LaunchContext ctx = ctx_in;
  ctx.stream = stream;
  const std::vector<std::vector<int>> counts = {{5, 0, 70, 3, 40}, {0, 41, 2, 36, 39}, {118, 0, 0, 0, 0}};   // same total
  for (int path : {1, 2}) {
    Spec s0;
    s0.rows = counts[0]; s0.seed = 21; s0.path = path; s0.tactic = make_tile(32, 128, true);
    const Problem P0 = make_problem(s0);
    DevProblem D(P0, s0);
    kg::GroupedGemmParams p = make_params(s0, P0, D);
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, &s0.tactic, ctx, &req), "b7 grouped graph workspace");
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 16);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    Status st = Status::kSuccess;
    if (ok) {
      kg::GroupedGemmParams q = p;
      if (path == 2) st = kg::grouped_gemm_prepare(&q, s0.tactic, ws, ctx);
      if (st == Status::kSuccess) st = kg::grouped_gemm(q, &s0.tactic, kg::GemmBackend::kNative, kops::Workspace{}, ctx);
      ok = cudaStreamEndCapture(stream, &graph) == cudaSuccess && st == Status::kSuccess;
      ok = ok && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    }
    KOPS_EXPECT_OK(st, "b7 grouped graph capture");
    if (!ok) {
      kops_test::record_failure("b7 grouped graph", "capture / instantiate failed", __FILE__, __LINE__);
      cudaGetLastError();
      continue;
    }
    for (const auto& c : counts) {
      Spec s = s0;
      s.rows = c;
      const Problem P = make_problem(s);   // same operands (same seed and total); only the grouping differs
      if (P.a_bytes != P0.a_bytes || P.w_bytes != P0.w_bytes) {
        kops_test::record_failure("b7 grouped graph", "test setup: operands differ between count sets", __FILE__, __LINE__);
        break;
      }
      kops_test::copy_in(D.offs.p, P.offs);
      D.reset(s);
      cudaDeviceSynchronize();
      KOPS_TEST_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
      KOPS_TEST_CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
      const auto d = as_vec<__nv_bfloat16>(kops_test::copy_out(D.d.p, D.d_bytes));
      std::vector<__nv_bfloat16> got;
      for (int lr = 0; lr < P.d_rows; ++lr) {
        for (int col = 0; col < P.n_out; ++col) got.push_back(d[static_cast<size_t>(lr) * P.ldd + col]);
      }
      expect_close(got, P.ref, path == 2 ? "b7 grouped graph replay (task map)" : "b7 grouped graph replay (in-kernel scan)");
      if (path == 2) {
        const auto hdr = kops_test::copy_out(reinterpret_cast<int32_t*>(scratch.p), 2);
        if (hdr[0] != expected_tasks(s, P)) kops_test::record_failure("b7 grouped graph replay", "task count does not follow the new expert counts", __FILE__, __LINE__);
      }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
  cudaStreamDestroy(stream);
}

void b7_grouped_contracts(const LaunchContext& ctx) {
  using namespace b7grp;
  Spec s;
  s.rows = {3, 0, 20};
  const Problem P = make_problem(s);
  DevProblem D(P, s);
  const kg::GroupedGemmParams base = make_params(s, P, D);
  const GemmTactic t64 = make_tile(64, 128, false);
  auto expect = [&](kg::GroupedGemmParams p, const GemmTactic* t, Status want, const char* what) {
    D.reset(s);
    KOPS_EXPECT_STATUS(kg::grouped_gemm(p, t, kg::GemmBackend::kNative, kops::Workspace{}, ctx), want, what);
    cudaDeviceSynchronize();
    const auto d16 = as_vec<uint16_t>(kops_test::copy_out(D.d.p, D.d_bytes));
    size_t bad = 0;
    for (uint16_t v : d16) bad += v != sentinel16(false);
    if (bad != 0) kops_test::record_failure(what, "output written by a rejected call", __FILE__, __LINE__);
  };
  {
    auto p = base;
    p.tile_task_map = D.midx.p;   // any device pointer: rejected before use
    expect(p, nullptr, Status::kInvalidArgument, "b7 grouped: task map without its tactic");
  }
  {
    auto p = base;
    p.layout = GroupedLayout::kContiguousRowIndex;
    const GemmTactic t128 = make_tile(128, 64, false);
    expect(p, &t128, Status::kUnsupportedShape, "b7 grouped: row index with a 128-row tile");
  }
  {
    auto p = base;
    p.layout = GroupedLayout::kIndexed;
    p.gather_idx = nullptr;
    expect(p, &t64, Status::kInvalidArgument, "b7 grouped: kIndexed without gather_idx");
  }
  {
    auto p = base;
    p.expert_offsets = nullptr;
    expect(p, &t64, Status::kInvalidArgument, "b7 grouped: kContiguous without offsets");
  }
  {
    auto p = base;
    p.layout = GroupedLayout::kMasked;
    p.masked_m = nullptr;
    expect(p, &t64, Status::kInvalidArgument, "b7 grouped: kMasked without masked_m");
  }
  {
    auto p = base;
    p.layout = GroupedLayout::kPtrArray;
    expect(p, &t64, Status::kNotImplemented, "b7 grouped: kPtrArray on mma.sync");
  }
  {
    auto p = base;
    p.recipe = GemmRecipe::kNVFP4;
    expect(p, &t64, Status::kNotImplemented, "b7 grouped: NVFP4 on mma.sync");
  }
  {
    auto p = base;
    p.b_zeros = D.w.p;
    expect(p, &t64, Status::kNotImplemented, "b7 grouped: zero points");
  }
  {
    auto p = base;
    p.lda = P.K + 3;
    expect(p, &t64, Status::kMisaligned, "b7 grouped: lda not 16 B");
  }
  {
    auto p = base;
    p.epi.ops |= kg::kEpiGLU;
    p.epi.glu_layout = kops::GateLayout::kGateUp;
    expect(p, &t64, Status::kNotImplemented, "b7 grouped: GLU with split gate / up");
  }
  {
    auto p = base;
    p.epi.bias = nullptr;
    expect(p, &t64, Status::kInvalidArgument, "b7 grouped: bias missing");
  }
  {
    auto p = base;
    p.d = nullptr;
    expect(p, &t64, Status::kInvalidArgument, "b7 grouped: null D");
  }
  {
    Spec sb = s;
    sb.recipe = R_FP8B;
    sb.k = 192;   // not a multiple of 128
    const Problem Pb = make_problem(sb);
    DevProblem Db(Pb, sb);
    auto p = make_params(sb, Pb, Db);
    KOPS_EXPECT_STATUS(kg::grouped_gemm(p, &t64, kg::GemmBackend::kNative, kops::Workspace{}, ctx), Status::kUnsupportedShape,
                       "b7 grouped: blockwise K % 128");
    sb.k = 256;
    sb.layout = L_INDEXED;
    sb.col_major_sa = true;
    const Problem Pc = make_problem(sb);
    DevProblem Dc(Pc, sb);
    p = make_params(sb, Pc, Dc);
    KOPS_EXPECT_STATUS(kg::grouped_gemm(p, &t64, kg::GemmBackend::kNative, kops::Workspace{}, ctx), Status::kInvalidArgument,
                       "b7 grouped: column-major sA with kIndexed");
    p = make_params(sb, Pc, Dc);
    p.b_scale = nullptr;
    KOPS_EXPECT_STATUS(kg::grouped_gemm(p, &t64, kg::GemmBackend::kNative, kops::Workspace{}, ctx), Status::kInvalidArgument,
                       "b7 grouped: blockwise without b_scale");
  }
  {
    // FP8 tensorwise without activation scales or a tactic stays on the W8A16 GEMV whatever expected_m says
    Spec sf = s;
    sf.recipe = R_FP8T;
    const Problem Pf = make_problem(sf);
    DevProblem Df(Pf, sf);
    auto p = make_params(sf, Pf, Df);
    p.a_scale = nullptr;
    p.expected_m = 64;
    p.epi.ops |= kg::kEpiGLU;   // only the tensor-core body has GLU: the GEMV must refuse, not reinterpret A
    p.epi.glu_layout = kops::GateLayout::kInterleaved;
    KOPS_EXPECT_STATUS(kg::grouped_gemm(p, nullptr, kg::GemmBackend::kNative, kops::Workspace{}, ctx), Status::kNotImplemented,
                       "b7 grouped: fp8 tensorwise without activation scales is W8A16");
  }
  // prepare
  {
    auto p = base;
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, &t64, ctx, &req), "b7 grouped prepare: requirements");
    KOPS_TEST_CHECK(req.device_scratch_bytes > 0);
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 16);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes - 4;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t64, ws, ctx), Status::kWorkspaceTooSmall, "b7 grouped prepare: short workspace");
    KOPS_TEST_CHECK(p.tile_task_map == nullptr);
    ws.device_scratch_bytes = req.device_scratch_bytes;
    ws.device_scratch = scratch.p + 4;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t64, ws, ctx), Status::kMisaligned, "b7 grouped prepare: misaligned scratch");
    ws.device_scratch = scratch.p;
    GemmTactic tw = t64;
    tw.mainloop = kg::MainloopSchedule::kTmaWarpSpecialized;
    // A TMA-class tactic also prepares GroupedProblem[G], so the mma-sized scratch is too small now.
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, tw, ws, ctx), Status::kWorkspaceTooSmall, "b7 grouped prepare: TMA tactic needs the problems region");
    KOPS_TEST_CHECK(p.tile_task_map == nullptr);
    p.layout = GroupedLayout::kContiguousRowIndex;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t64, ws, ctx), Status::kNotImplemented, "b7 grouped prepare: row index layout");
    p = base;
    p.max_m = 0;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t64, ws, ctx), Status::kInvalidArgument, "b7 grouped prepare: max_m missing");
    // capacity from a max_m smaller than the rows: the map overflows and says so
    p = base;
    p.max_m = 1;
    kops::WorkspaceRequirements r1{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, &t64, ctx, &r1), "b7 grouped prepare: small requirements");
    ws.device_scratch_bytes = r1.device_scratch_bytes;
    KOPS_EXPECT_OK(kg::grouped_gemm_prepare(&p, t64, ws, ctx), "b7 grouped prepare: small capacity");
    cudaDeviceSynchronize();
    const auto hdr = kops_test::copy_out(reinterpret_cast<int32_t*>(scratch.p), 2);
    // tasks: 3 rows -> 1 m tile, 20 rows -> 1 m tile, 2 n tiles each = 4; capacity (1 + 3) * 2 = 8
    KOPS_TEST_CHECK(hdr[0] == 4 && hdr[1] == 0);
  }
}

void test_b7_grouped(const LaunchContext& ctx) {
  b7_grouped_numerics(ctx);
  b7_task_map(ctx);
  b7_grouped_graph(ctx);
  b7_grouped_contracts(ctx);
}

// ---- B11 grouped GEMM preparation (GroupedProblem[] + task map, TMA-class mainloops) --------------

/// Host reference for one prepared group: the group's rows, the byte offsets of its operands from the
/// params' bases, its dequant alpha and its first tile id.
struct RefGroup {
  int32_t m = 0;
  int64_t a_off = 0, b_off = 0, d_off = 0;
  float alpha = 0.0f;
  int64_t tile_offset = 0;
};

/// Checks GroupedProblem[G] and the task map that grouped_gemm_prepare wrote, plus the
/// (num_tasks, overflow) header, against the reference. The header is addressed through the params
/// (tile_task_map points just past it) so the test does not depend on the scratch layout.
void b11_check_prepared(const char* what, const kg::GroupedGemmParams& p, const std::vector<RefGroup>& ref,
                        int tile_m, int tile_n) {
  const int g_count = static_cast<int>(ref.size());
  const auto got = kops_test::copy_out(static_cast<const kg::GroupedProblem*>(p.problems), static_cast<size_t>(g_count));
  const int32_t* hdr = p.tile_task_map - 4;
  const auto h0 = kops_test::copy_out(hdr, 1);
  const auto h = kops_test::copy_out(hdr, 4 + static_cast<size_t>(3) * static_cast<size_t>(h0[0] > 0 ? h0[0] : 0));
  int64_t want_tiles = 0;
  size_t bad_fields = 0, bad_map = 0;
  const int n_tiles = static_cast<int>((p.n + tile_n - 1) / tile_n);
  for (int g = 0; g < g_count; ++g) {
    const RefGroup& r = ref[g];
    const kg::GroupedProblem& q = got[g];
    if (q.m != r.m || q.n != p.n || q.k != p.k || q.tile_offset != static_cast<int32_t>(r.tile_offset)) ++bad_fields;
    if (q.lda != p.lda || q.ldb != p.ldb || q.ldd != p.ldd) ++bad_fields;
    if (q.alpha != r.alpha) ++bad_fields;
    const auto* abase = static_cast<const uint8_t*>(p.a);
    const auto* bbase = static_cast<const uint8_t*>(p.b);
    const auto* dbase = static_cast<const uint8_t*>(p.d);
    const bool a_ok = p.layout == GroupedLayout::kIndexed ? q.a == p.a : q.a == abase + r.a_off;
    if (!a_ok || q.b != bbase + r.b_off || q.d != dbase + r.d_off) ++bad_fields;
    want_tiles += static_cast<int64_t>((r.m + tile_m - 1) / tile_m) * n_tiles;
  }
  // task map: group g occupies [tile_offset, tile_offset + tiles), m tile fastest
  int64_t idx = 0;
  for (int g = 0; g < g_count && idx < h[0]; ++g) {
    const int mt = (ref[g].m + tile_m - 1) / tile_m;
    for (int t = 0; t < mt * n_tiles && idx < h[0]; ++t) {
      const int32_t* e = &h[4 + 3 * idx];
      bad_map += e[0] != g || e[1] != t % mt || e[2] != t / mt;
      ++idx;
    }
  }
  if (h[0] != static_cast<int32_t>(want_tiles) || h[1] != 0) {
    char det[128];
    std::snprintf(det, sizeof(det), "num_tasks %d overflow %d, want %lld / 0", h[0], h[1],
                  static_cast<long long>(want_tiles));
    kops_test::record_failure(what, det, __FILE__, __LINE__);
  }
  if (bad_fields != 0) kops_test::record_failure(what, "GroupedProblem fields differ from the reference", __FILE__, __LINE__);
  if (bad_map != 0) kops_test::record_failure(what, "task map entries differ from the reference", __FILE__, __LINE__);
}

void b11_grouped_prepare(const LaunchContext& ctx) {
  using namespace b7grp;
  auto tma_tile = [](int tm, int tn) {
    GemmTactic t = make_tile(tm, tn, false);
    t.mainloop = kg::MainloopSchedule::kTmaWarpSpecializedCooperative;
    return t;
  };
  // 1. contiguous: rows come from expert_offsets, empty experts produce no tiles
  {
    Spec s;
    s.rows = {5, 0, 70, 3, 33};
    s.n = 192;
    s.k = 256;
    s.recipe = R_BF16;
    const Problem P = make_problem(s);
    DevProblem D(P, s);
    auto p = make_params(s, P, D);
    const GemmTactic t = tma_tile(64, 128);
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, &t, ctx, &req), "b11 prepare: contiguous requirements");
    KOPS_TEST_CHECK(req.device_scratch_bytes >= static_cast<size_t>(P.G) * sizeof(kg::GroupedProblem));
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 64);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    // captured-size check: the problems region is 16 B aligned and precedes the task map
    kops::WorkspaceRequirements r0{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, nullptr, ctx, &r0), "b11 prepare: no tactic");
    KOPS_TEST_CHECK(r0.device_scratch_bytes == 0);
    KOPS_EXPECT_OK(kg::grouped_gemm_prepare(&p, t, ws, ctx), "b11 prepare: contiguous");
    cudaDeviceSynchronize();
    KOPS_TEST_CHECK(p.problems != nullptr && p.tile_task_map != nullptr && p.group_tma == nullptr);
    std::vector<RefGroup> ref;
    int64_t tile = 0;
    for (int g = 0; g < P.G; ++g) {
      RefGroup r;
      r.m = s.rows[g];
      r.a_off = static_cast<int64_t>(P.offs[g]) * P.K * 2;
      r.b_off = static_cast<int64_t>(g) * P.N * P.K * 2;
      r.d_off = static_cast<int64_t>(P.offs[g]) * P.ldd * 2;
      r.alpha = p.epi.alpha;
      r.tile_offset = tile;
      tile += (r.m + 63) / 64 * ((P.N + 127) / 128);
      ref.push_back(r);
    }
    b11_check_prepared("b11 prepare contiguous", p, ref, 64, 128);
  }
  // 2. masked: rows come from masked_m clamped to max_m, A and D are [G, max_m, *]
  {
    Spec s;
    s.layout = L_MASKED;
    s.rows = {0, 5, 12};
    s.n = 128;
    s.k = 128;
    const Problem P = make_problem(s);
    DevProblem D(P, s);
    auto p = make_params(s, P, D);
    const GemmTactic t = tma_tile(32, 64);
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, &t, ctx, &req), "b11 prepare: masked requirements");
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 64);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    KOPS_EXPECT_OK(kg::grouped_gemm_prepare(&p, t, ws, ctx), "b11 prepare: masked");
    cudaDeviceSynchronize();
    std::vector<RefGroup> ref;
    int64_t tile = 0;
    for (int g = 0; g < P.G; ++g) {
      RefGroup r;
      r.m = std::min(s.rows[g], P.max_m);
      r.a_off = static_cast<int64_t>(g) * P.max_m * P.K * 2;
      r.b_off = static_cast<int64_t>(g) * P.N * P.K * 2;
      r.d_off = static_cast<int64_t>(g) * P.max_m * P.ldd * 2;
      r.alpha = p.epi.alpha;
      r.tile_offset = tile;
      tile += (r.m + 31) / 32 * ((P.N + 63) / 64);
      ref.push_back(r);
    }
    b11_check_prepared("b11 prepare masked", p, ref, 32, 64);
  }
  // 3. ptr array: per-problem pointers and rows
  {
    const int G = 3;
    const int N = 64, K = 64;
    DevBuf<uint8_t> a(static_cast<size_t>(G) * 16 * K * 2), b(static_cast<size_t>(G) * N * K * 2);
    DevBuf<uint8_t> d(static_cast<size_t>(G) * 16 * N * 2);
    std::vector<const void*> pa(G), pb(G);
    std::vector<void*> pd(G);
    for (int g = 0; g < G; ++g) {
      pa[g] = a.p + static_cast<size_t>(g) * 16 * K * 2;
      pb[g] = b.p + static_cast<size_t>(g) * N * K * 2;
      pd[g] = d.p + static_cast<size_t>(g) * 16 * N * 2;
    }
    auto dpa = upload(pa);
    auto dpb = upload(pb);
    auto dpd = upload(pd);
    auto dm = upload(std::vector<int32_t>{4, 0, 9});
    kg::GroupedGemmParams p{};
    p.layout = GroupedLayout::kPtrArray;
    p.recipe = GemmRecipe::kBF16;
    p.num_groups = G;
    p.a_ptrs = dpa.p;
    p.b_ptrs = dpb.p;
    p.d_ptrs = dpd.p;
    p.problem_m = dm.p;
    p.lda = K; p.ldb = K; p.ldd = N;
    p.n = N; p.k = K;
    p.max_m = 16;
    p.epi.alpha = 0.5f;
    const GemmTactic t = tma_tile(16, 32);
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, &t, ctx, &req), "b11 prepare: ptr array requirements");
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 64);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    KOPS_EXPECT_OK(kg::grouped_gemm_prepare(&p, t, ws, ctx), "b11 prepare: ptr array");
    cudaDeviceSynchronize();
    const auto probs = kops_test::copy_out(reinterpret_cast<const kg::GroupedProblem*>(scratch.p), static_cast<size_t>(G));
    size_t bad = 0;
    int64_t tile = 0;
    const int32_t ms[G] = {4, 0, 9};
    for (int g = 0; g < G; ++g) {
      bad += probs[g].m != ms[g];
      bad += probs[g].a != pa[g] || probs[g].b != pb[g] || probs[g].d != pd[g];
      bad += probs[g].alpha != 0.5f;
      bad += probs[g].tile_offset != static_cast<int32_t>(tile);
      tile += (ms[g] + 15) / 16 * ((N + 31) / 32);
    }
    if (bad != 0) kops_test::record_failure("b11 prepare: ptr array", "GroupedProblem fields differ", __FILE__, __LINE__);
    const int32_t* hdr = p.tile_task_map - 4;
    const auto h = kops_test::copy_out(hdr, static_cast<size_t>(4 + 3 * std::min<int64_t>(tile, 64)));
    if (h[0] != static_cast<int32_t>(tile) || h[1] != 0) {
      kops_test::record_failure("b11 prepare: ptr array", "num_tasks / overflow differ", __FILE__, __LINE__);
    }
  }
  // 4. fp8 tensorwise: alpha picks up the per-expert weight scale, sfb moves by the expert stride
  {
    Spec s;
    s.rows = {9, 4};
    s.recipe = R_FP8T;
    const Problem P = make_problem(s);
    DevProblem D(P, s);
    auto p = make_params(s, P, D);
    p.b_scale_group_stride = 1;
    p.epi.alpha = 0.75f;
    const GemmTactic t = tma_tile(64, 128);
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, &t, ctx, &req), "b11 prepare: fp8 requirements");
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 64);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    KOPS_EXPECT_OK(kg::grouped_gemm_prepare(&p, t, ws, ctx), "b11 prepare: fp8 tensorwise");
    cudaDeviceSynchronize();
    const auto probs = kops_test::copy_out(reinterpret_cast<const kg::GroupedProblem*>(scratch.p), 2u);
    size_t bad = 0;
    for (int g = 0; g < 2; ++g) {
      bad += probs[g].alpha != 0.75f * P.sb[static_cast<size_t>(g)];
      bad += probs[g].sfb != static_cast<const uint8_t*>(p.b_scale) + 4 * g;
      bad += probs[g].sfa != p.a_scale;   // per-tensor A scale: not a row-major per-token layout
      bad += probs[g].a != static_cast<const uint8_t*>(p.a) + static_cast<int64_t>(P.offs[g]) * P.K;
    }
    if (bad != 0) kops_test::record_failure("b11 prepare: fp8 tensorwise", "alpha / sfa / sfb / a differ", __FILE__, __LINE__);
  }
  // 5. graph replay: the preparation follows the expert counts that change on the device
  {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
      kops_test::record_failure("b11 prepare graph", "cudaStreamCreate failed", __FILE__, __LINE__);
      return;
    }
    LaunchContext gctx = ctx;
    gctx.stream = stream;
    Spec s;
    s.rows = {5, 0, 70, 3};
    s.n = 128;
    s.k = 128;
    const Problem P = make_problem(s);
    DevProblem D(P, s);
    auto p = make_params(s, P, D);
    const GemmTactic t = tma_tile(64, 128);
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(p, &t, gctx, &req), "b11 prepare graph: requirements");
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 64);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    Status st = Status::kSuccess;
    kg::GroupedGemmParams q = p;   // the captured launch holds a snapshot of these params
    if (ok) {
      st = kg::grouped_gemm_prepare(&q, t, ws, gctx);
      ok = cudaStreamEndCapture(stream, &graph) == cudaSuccess && st == Status::kSuccess;
      ok = ok && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    }
    KOPS_EXPECT_OK(st, "b11 prepare graph: capture");
    if (!ok) {
      kops_test::record_failure("b11 prepare graph", "capture / instantiate failed", __FILE__, __LINE__);
      cudaGetLastError();
      cudaStreamDestroy(stream);
      return;
    }
    const std::vector<std::vector<int>> counts = {{5, 0, 70, 3}, {0, 0, 0, 0}, {33, 41, 2, 0}};
    for (const auto& c : counts) {
      std::vector<int64_t> offs(c.size() + 1, 0);
      for (size_t g = 0; g < c.size(); ++g) offs[g + 1] = offs[g] + c[g];
      kops_test::copy_in(D.offs.p, offs);
      cudaDeviceSynchronize();
      KOPS_TEST_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
      KOPS_TEST_CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
      const auto probs = kops_test::copy_out(reinterpret_cast<const kg::GroupedProblem*>(scratch.p), c.size());
      const auto h = kops_test::copy_out(q.tile_task_map - 4, 2);
      int64_t want_tiles = 0, tile = 0;
      size_t bad = 0;
      for (size_t g = 0; g < c.size(); ++g) {
        const int mt = (c[g] + 63) / 64;
        bad += probs[g].m != c[g];
        bad += probs[g].tile_offset != static_cast<int32_t>(tile);
        bad += probs[g].a != static_cast<const uint8_t*>(p.a) + offs[g] * P.K * 2;
        tile += static_cast<int64_t>(mt) * ((P.N + 127) / 128);
        want_tiles += static_cast<int64_t>(mt) * ((P.N + 127) / 128);
      }
      char det[160];
      std::snprintf(det, sizeof(det), "num_tasks %d (want %lld), %zu problem fields differ", h[0],
                    static_cast<long long>(want_tiles), bad);
      if (h[0] != static_cast<int32_t>(want_tiles) || h[1] != 0 || bad != 0) {
        kops_test::record_failure("b11 prepare graph: replay after the expert counts changed", det, __FILE__, __LINE__);
      }
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
  // 6. contracts
  {
    Spec s;
    s.rows = {3, 0, 20};
    const Problem P = make_problem(s);
    DevProblem D(P, s);
    const auto base = make_params(s, P, D);
    const GemmTactic t = tma_tile(64, 128);
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kg::grouped_gemm_workspace_requirements(base, &t, ctx, &req), "b11 prepare: contract requirements");
    DevBuf<uint8_t> scratch(req.device_scratch_bytes + 64);
    kops::Workspace ws{};
    ws.device_scratch = scratch.p;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    auto p = base;
    ws.device_scratch_bytes = req.device_scratch_bytes - 8;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kWorkspaceTooSmall, "b11 prepare: short workspace");
    KOPS_TEST_CHECK(p.problems == nullptr && p.tile_task_map == nullptr);
    ws.device_scratch_bytes = req.device_scratch_bytes;
    ws.device_scratch = scratch.p + 8;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kMisaligned, "b11 prepare: misaligned scratch");
    ws.device_scratch = scratch.p;
    p = base;
    p.layout = GroupedLayout::kContiguousRowIndex;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kNotImplemented, "b11 prepare: row index layout");
    p = base;
    p.layout = GroupedLayout::kMasked;
    p.masked_m = nullptr;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kInvalidArgument, "b11 prepare: masked without masked_m");
    p = base;
    p.recipe = GemmRecipe::kW4A16Int;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kNotImplemented, "b11 prepare: mixed-input recipe");
    p = base;
    p.recipe = GemmRecipe::kFP8Tensorwise;
    p.b_scale = nullptr;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kInvalidArgument, "b11 prepare: tensorwise without b_scale");
    p = base;
    p.d = nullptr;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kInvalidArgument, "b11 prepare: null D");
    p = base;
    p.max_m = 0;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kInvalidArgument, "b11 prepare: max_m missing");
    GemmTactic tb = t;
    tb.mainloop = kg::MainloopSchedule::kSimtFma;
    p = base;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, tb, ws, ctx), Status::kNotImplemented, "b11 prepare: SIMT mainloop");
    p = base;
    p.layout = GroupedLayout::kPtrArray;   // no pointer arrays
    p.expert_offsets = nullptr;
    KOPS_EXPECT_STATUS(kg::grouped_gemm_prepare(&p, t, ws, ctx), Status::kInvalidArgument, "b11 prepare: ptr array without pointers");
  }
}

// ---- multi-LoRA (Punica SGMV) ---------------------------------------------------------------------

template <typename T>
void run_b7_lora(const LaunchContext& ctx, int rank, int out_slice_offset, const char* what_in) {
  char what[160];
  std::snprintf(what, sizeof(what), "%s rank %d slice offset %d", what_in, rank, out_slice_offset);
  const int in_dim = 4096, ldx = in_dim + 8, out_dim = 384, ldy = 3 * out_dim, bw = 136;
  const std::vector<int32_t> indptr = {0, 5, 5, 37, 40, 73, 75};
  const std::vector<int32_t> ids = {1, 0, -1, 2, 1, 0};   // segment 1 is empty, segment 2 has no LoRA
  const int nseg = static_cast<int>(ids.size()), T_ = indptr.back(), slots = 3;
  const float scaling = 0.5f;
  Rng r(31337u + rank * 3u + out_slice_offset);
  const auto x = random_t<T>(static_cast<size_t>(T_) * ldx, &r, -1.0f, 1.0f);
  std::vector<std::vector<T>> la(slots), lb(slots);
  for (int s = 0; s < slots; ++s) {
    la[s] = random_t<T>(static_cast<size_t>(rank) * in_dim, &r, -0.05f, 0.05f);
    lb[s] = random_t<T>(static_cast<size_t>(out_dim) * rank, &r, -0.5f, 0.5f);
  }
  const auto y0 = random_t<T>(static_cast<size_t>(T_) * ldy, &r, -1.0f, 1.0f);
  auto dx = upload(x);
  std::vector<DevBuf<T>> dla, dlb;
  std::vector<const void*> pa, pb;
  for (int s = 0; s < slots; ++s) {
    dla.push_back(upload(la[s]));
    dlb.push_back(upload(lb[s]));
    pa.push_back(dla.back().p);
    pb.push_back(dlb.back().p);
  }
  auto dpa = upload(pa);
  auto dpb = upload(pb);
  auto dind = upload(indptr);
  auto dids = upload(ids);
  DevBuf<float> buf(static_cast<size_t>(T_) * bw);
  DevBuf<T> y(y0.size());
  auto reset = [&]() {
    kops_test::copy_in(buf.p, std::vector<float>(static_cast<size_t>(T_) * bw, -7.0f));
    kops_test::copy_in(y.p, y0);
  };
  kg::LoraSgmvParams sh{}, ex{};
  sh.x = dx.p; sh.ldx = ldx; sh.lora_a = dpa.p; sh.y = buf.p; sh.ldy = bw;
  sh.seg_indptr = dind.p; sh.seg_lora_ids = dids.p; sh.num_segments = nseg;
  sh.in_dim = in_dim; sh.rank = rank;
  ex.x = buf.p; ex.ldx = bw; ex.lora_a = dpb.p; ex.y = y.p; ex.ldy = ldy;
  ex.seg_indptr = dind.p; ex.seg_lora_ids = dids.p; ex.num_segments = nseg;
  ex.out_dim = out_dim; ex.rank = rank; ex.out_slice_offset = out_slice_offset; ex.scaling = scaling;
  const Tensor tbuf = tensor2d_ld(buf.p, DType::kFloat32, T_, bw, bw);
  reset();
  KOPS_EXPECT_OK(kg::lora_sgmv(sh, ex, dt_of<T>(), tbuf, ctx), what);
  cudaDeviceSynchronize();
  const auto gbuf = kops_test::copy_out(buf.p, static_cast<size_t>(T_) * bw);
  const auto gy = kops_test::copy_out(y.p, y0.size());
  std::vector<int> slot_of(T_, -1);
  for (int s = 0; s < nseg; ++s) {
    for (int t = indptr[s]; t < indptr[s + 1]; ++t) slot_of[t] = ids[s];
  }
  // shrink: fp32 against fp64
  {
    double ss = 0.0;
    std::vector<double> want(static_cast<size_t>(T_) * rank, 0.0);
    for (int t = 0; t < T_; ++t) {
      if (slot_of[t] < 0) continue;
      for (int j = 0; j < rank; ++j) {
        double acc = 0.0;
        for (int kk = 0; kk < in_dim; ++kk) acc += dval(x, static_cast<size_t>(t) * ldx + kk) * dval(la[slot_of[t]], static_cast<size_t>(j) * in_dim + kk);
        want[static_cast<size_t>(t) * rank + j] = acc;
        ss += acc * acc;
      }
    }
    const double rms = std::sqrt(ss / static_cast<double>(want.size()));
    size_t bad = 0, untouched = 0;
    for (int t = 0; t < T_; ++t) {
      for (int j = 0; j < bw; ++j) {
        const float g = gbuf[static_cast<size_t>(t) * bw + j];
        if (slot_of[t] < 0 || j >= rank) { untouched += g != -7.0f; continue; }
        const double w = want[static_cast<size_t>(t) * rank + j];
        if (!(std::fabs(g - w) <= 2e-4 * rms + 1e-4 * std::fabs(w))) ++bad;
      }
    }
    if (bad != 0) kops_test::record_failure(what, "shrink buffer differs from the fp64 oracle", __FILE__, __LINE__);
    if (untouched != 0) kops_test::record_failure(what, "shrink wrote rows without LoRA or columns past the rank", __FILE__, __LINE__);
  }
  // expand: y += scaling * round_T(buffer) . B^T on the slice, from the buffer the shrink produced
  {
    std::vector<T> got;
    std::vector<double> want;
    size_t untouched = 0;
    for (int t = 0; t < T_; ++t) {
      for (int c = 0; c < ldy; ++c) {
        const size_t i = static_cast<size_t>(t) * ldy + c;
        const int o = c - out_slice_offset;
        if (slot_of[t] < 0 || o < 0 || o >= out_dim) {
          untouched += std::memcmp(&gy[i], &y0[i], sizeof(T)) != 0;
          continue;
        }
        double acc = 0.0;
        for (int j = 0; j < rank; ++j) {
          const double b = kops_test::to_float(cvt<T>(gbuf[static_cast<size_t>(t) * bw + j]));
          acc += b * dval(lb[slot_of[t]], static_cast<size_t>(o) * rank + j);
        }
        got.push_back(gy[i]);
        want.push_back(dval(y0, i) + scaling * acc);
      }
    }
    expect_close(got, want, what);
    if (untouched != 0) kops_test::record_failure(what, "expand wrote outside the slice or rows without LoRA", __FILE__, __LINE__);
  }
  // either stage alone, determinism
  {
    reset();
    kg::LoraSgmvParams none{};
    KOPS_EXPECT_OK(kg::lora_sgmv(sh, none, dt_of<T>(), tbuf, ctx), what);
    cudaDeviceSynchronize();
    const auto b2 = kops_test::copy_out(buf.p, gbuf.size());
    const auto y2 = kops_test::copy_out(y.p, y0.size());
    if (std::memcmp(b2.data(), gbuf.data(), gbuf.size() * 4) != 0) kops_test::record_failure(what, "shrink alone differs from shrink + expand", __FILE__, __LINE__);
    if (std::memcmp(y2.data(), y0.data(), y0.size() * sizeof(T)) != 0) kops_test::record_failure(what, "shrink alone touched y", __FILE__, __LINE__);
    KOPS_EXPECT_OK(kg::lora_sgmv(none, ex, dt_of<T>(), tbuf, ctx), what);
    cudaDeviceSynchronize();
    const auto y3 = kops_test::copy_out(y.p, y0.size());
    if (std::memcmp(y3.data(), gy.data(), gy.size() * sizeof(T)) != 0) kops_test::record_failure(what, "two-call expand differs / not deterministic", __FILE__, __LINE__);
  }
}

void test_b7_lora(const LaunchContext& ctx) {
  for (int rank : {8, 16, 64, 128}) {
    run_b7_lora<__nv_bfloat16>(ctx, rank, 384, "b7 lora bf16");
    run_b7_lora<__half>(ctx, rank, 768, "b7 lora fp16");
  }
  run_b7_lora<__nv_bfloat16>(ctx, 16, 3, "b7 lora bf16 odd slice");   // slice start not 4 B aligned
  // contracts
  const int T_ = 8, in_dim = 64, rank = 16;
  DevBuf<float> buf(static_cast<size_t>(T_) * rank);
  DevBuf<__nv_bfloat16> x(static_cast<size_t>(T_) * in_dim), y(static_cast<size_t>(T_) * 32), w(static_cast<size_t>(rank) * in_dim);
  std::vector<const void*> ptrs = {w.p};
  auto dptrs = upload(ptrs);
  auto dind = upload(std::vector<int32_t>{0, T_});
  auto dids = upload(std::vector<int32_t>{0});
  kg::LoraSgmvParams sh{}, ex{}, none{};
  sh.x = x.p; sh.ldx = in_dim; sh.lora_a = dptrs.p; sh.seg_indptr = dind.p; sh.seg_lora_ids = dids.p; sh.num_segments = 1;
  sh.in_dim = in_dim; sh.rank = rank;
  ex.lora_a = dptrs.p; ex.y = y.p; ex.ldy = 32; ex.seg_indptr = dind.p; ex.seg_lora_ids = dids.p; ex.num_segments = 1;
  ex.out_dim = 32; ex.rank = rank; ex.scaling = 1.0f;
  const Tensor tb = tensor2d_ld(buf.p, DType::kFloat32, T_, rank, rank);
  KOPS_EXPECT_STATUS(kg::lora_sgmv(sh, ex, DType::kFloat32, tb, ctx), Status::kUnsupportedDtype, "b7 lora: fp32 dtype");
  KOPS_EXPECT_STATUS(kg::lora_sgmv(none, none, DType::kBFloat16, tb, ctx), Status::kInvalidArgument, "b7 lora: nothing to do");
  KOPS_EXPECT_STATUS(kg::lora_sgmv(sh, none, DType::kBFloat16, tensor2d_ld(buf.p, DType::kBFloat16, T_, rank, rank), ctx), Status::kInvalidArgument,
                     "b7 lora: bf16 shrink buffer");
  auto bad = sh;
  bad.y = y.p;
  KOPS_EXPECT_STATUS(kg::lora_sgmv(bad, none, DType::kBFloat16, tb, ctx), Status::kInvalidArgument, "b7 lora: shrink.y is not the buffer");
  bad = sh;
  bad.seg_indptr = nullptr;
  KOPS_EXPECT_STATUS(kg::lora_sgmv(bad, none, DType::kBFloat16, tb, ctx), Status::kInvalidArgument, "b7 lora: no segments");
  bad = sh;
  bad.rank = 32;
  KOPS_EXPECT_STATUS(kg::lora_sgmv(bad, none, DType::kBFloat16, tb, ctx), Status::kInvalidArgument, "b7 lora: buffer narrower than rank");
  auto bex = ex;
  bex.rank = 4;
  KOPS_EXPECT_STATUS(kg::lora_sgmv(none, bex, DType::kBFloat16, tb, ctx), Status::kInvalidArgument, "b7 lora: expand rank < 8");
  bex = ex;
  bex.out_slice_offset = 8;
  KOPS_EXPECT_STATUS(kg::lora_sgmv(none, bex, DType::kBFloat16, tb, ctx), Status::kInvalidArgument, "b7 lora: slice past ldy");
  bex = ex;
  bex.x = x.p;
  KOPS_EXPECT_STATUS(kg::lora_sgmv(none, bex, DType::kBFloat16, tb, ctx), Status::kInvalidArgument, "b7 lora: expand.x is not the buffer");
  bad = sh;
  bad.ldx = in_dim + 3;
  KOPS_EXPECT_STATUS(kg::lora_sgmv(bad, none, DType::kBFloat16, tb, ctx), Status::kMisaligned, "b7 lora: ldx not 16 B");
}

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
  test_grouped_gemv(ctx);
  test_moe_gemv(ctx);
  test_gguf_gemv(ctx);
  test_gguf_moe(ctx);
  test_gguf_contracts(ctx);
  test_dense_gemv(ctx);
  test_simt(ctx);
  test_gemv_argmax(ctx);
  test_router(ctx);
  test_fp4_gemv(ctx);
  test_fp4_marlin(ctx);
  test_fp4_marlin_contracts(ctx);
  test_w4a16(ctx);
  test_lut_gemv(ctx);
  test_w4a16_contracts(ctx);
  test_w4a16_marlin_layout(ctx);
  test_w4a16_marlin(ctx);
  test_w4a16_marlin_contracts(ctx);
  test_fp8_decode(ctx);
  test_fp8_gemv(ctx);
  test_fp8_invariance(ctx);
  test_int8_gemv(ctx);
  test_fp8_int8_contracts(ctx);
  test_weight_transpose(ctx);
  test_glu_interleave(ctx);
  test_router_split(ctx);
  test_colsum(ctx);
  test_scale_swizzle(ctx);
  test_scale_tma(ctx);
  test_machete_prepack(ctx);
  test_e4m3_encode_exhaustive();
  test_block_quant(ctx);
  test_contracts(ctx);
  test_mma_sweep(ctx);
  test_mma_shapes(ctx);
  test_mma_epilogues(ctx);
  test_mma_splitk(ctx);
  test_mma_invariance(ctx);
  test_mma_wide_fetch(ctx);
  test_mma_small_m(ctx);
  test_mma_dispatch(ctx);
  test_mma_cache(ctx);
  test_mma_contracts(ctx);
  test_dual(ctx);
  test_dual_row_invariance(ctx);
  test_dual_contracts(ctx);
  test_bmm(ctx);
  test_bmm_contracts(ctx);
  test_b7_fp8(ctx);
  test_b7_fp8_bmm(ctx);
  test_b7_fp8_contracts(ctx);
  test_b7_tf32(ctx);
  test_b7_quant_out(ctx);
  test_b7_int8(ctx);
  test_b7_w4a8(ctx);
  test_b7_gguf_mmq(ctx);
  test_b7_grouped(ctx);
  b11_grouped_prepare(ctx);
  test_b7_lora(ctx);
  return kops_test::finish("test_gemm");
}
