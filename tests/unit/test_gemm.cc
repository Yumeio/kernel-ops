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
                 int64_t row0) {
  kops::gemm::ScaledGemmArgs args{};
  args.recipe = kops::gemm::GemmRecipe::kW4A16Int;
  args.a.data = da + row0 * p.k; args.a.ld = p.k; args.a.dtype = dt_of<T>();
  args.b.data = dqw; args.b.dtype = DType::kUInt32;
  args.b.packing = p.awq ? kops::WeightPacking::kAWQ : kops::WeightPacking::kGPTQ;
  args.b.scale = dsc; args.b.scale_dtype = dt_of<T>(); args.b.group_k = static_cast<int16_t>(p.group);
  args.b.zero_points = p.zeros ? dqz : nullptr; args.b.zero_dtype = DType::kUInt32;
  args.b.g_idx = (p.actorder && !p.awq) ? dgi : nullptr;
  if (p.bias) { args.epi.ops = kops::gemm::kEpiBias; args.epi.bias = dbias; args.epi.bias_dtype = dt_of<T>(); }
  args.d = dout + row0 * p.n; args.ldd = p.n; args.d_dtype = dt_of<T>();
  args.m = rows; args.n = p.n; args.k = p.k;
  return kops::gemm::mixed_input_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx);
}

template <typename T>
void run_w4a16(const LaunchContext& ctx, bool awq, int group, bool zeros, bool actorder, bool bias, int m,
               int n, int k, const char* what, bool invariance = false) {
  const W4Problem<T> p = make_w4<T>(awq, group, zeros, actorder, bias, m, n, k, 77u + m * 131u + n + k);
  auto da = upload(to_vec<T>(p.a));
  auto dsc = upload(to_vec<T>(p.scale));
  auto dqw = upload(p.qw);
  auto dqz = upload(p.qz);
  auto dgi = upload(p.g_idx);
  auto db = upload(to_vec<T>(p.bias_v));
  DevBuf<T> dout(static_cast<size_t>(m) * n);
  KOPS_EXPECT_OK(launch_w4<T>(ctx, p, da.p, dqw.p, dsc.p, dqz.p, dgi.p, db.p, dout.p, m, 0), what);
  cudaDeviceSynchronize();
  const auto got = kops_test::copy_out(dout.p, static_cast<size_t>(m) * n);
  expect_close(got, p.ref, what);
  if (!invariance) return;
  // Bit-exactness: 100 repeats, and each row alone == the same row inside the batch.
  for (int rep = 0; rep < 100; ++rep) {
    KOPS_EXPECT_OK(launch_w4<T>(ctx, p, da.p, dqw.p, dsc.p, dqz.p, dgi.p, db.p, dout.p, m, 0), what);
    cudaDeviceSynchronize();
    if (!same_bits(kops_test::copy_out(dout.p, static_cast<size_t>(m) * n), got)) {
      kops_test::record_failure(what, "result changed between repeats", __FILE__, __LINE__);
      break;
    }
  }
  DevBuf<T> one(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    KOPS_EXPECT_OK(launch_w4<T>(ctx, p, da.p, dqw.p, dsc.p, dqz.p, dgi.p, db.p, one.p, 1, row), what);
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
  expect_close(kops_test::copy_out(dout.p, static_cast<size_t>(m) * n), ref, what);
}

void test_lut_gemv(const LaunchContext& ctx) {
  for (int m : {1, 2, 3, 8, 11}) {
    run_lut<__nv_bfloat16>(ctx, 64, false, m, 40, 1024, "lut gemv b64 bf16");
    run_lut<__half>(ctx, 64, true, m, 33, 2048, "lut gemv b64 double-quant f16");
    run_lut<__nv_bfloat16>(ctx, 32, false, m, 17, 96, "lut gemv b32 ragged k");
    run_lut<__nv_bfloat16>(ctx, 256, true, m, 9, 768, "lut gemv b256 double-quant");
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
  KOPS_EXPECT_STATUS(call([](auto& a) { a.b.packing = kops::WeightPacking::kMarlin; }), Status::kUnsupportedLayout, "w4a16 packing");
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
  { auto x = base(); x.m = 17; KOPS_EXPECT_STATUS(run8(x), Status::kNotImplemented, "fp8 M > 16"); }
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
  { auto x = ibase(); x.m = 40; KOPS_EXPECT_STATUS(run_i8(x), Status::kNotImplemented, "int8 M > 16"); }
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

template <typename T>
void run_fp4(const LaunchContext& ctx, bool nv, bool swizzled, int m, int n, int k, bool global, const char* what) {
  Rng r(2024u + m + n + nv);
  const int blk = nv ? 16 : 32;
  const int blocks = k / blk;
  const auto a = random_t<T>(static_cast<size_t>(m) * k, &r, -1.0f, 1.0f);
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
      ref[static_cast<size_t>(row) * n + c] = acc / gscale;
    }
  auto da = upload(a);
  auto dw = upload(packed);
  auto dsc = upload(sc);
  auto dg = upload(std::vector<float>(1, gscale));
  DevBuf<T> dout(ref.size());
  ScaledGemmArgs args{};
  args.recipe = nv ? GemmRecipe::kW4A16NVFP4 : GemmRecipe::kW4A16MXFP4;
  args.a.data = da.p; args.a.ld = k; args.a.dtype = dt_of<T>();
  args.b.data = dw.p; args.b.ld = k / 2; args.b.dtype = DType::kFloat4E2M1x2;
  args.b.packing = kops::WeightPacking::kFP4E2M1;
  args.b.scale = dsc.p; args.b.scale_dtype = nv ? DType::kFloat8E4M3 : DType::kFloat8E8M0; args.b.group_k = static_cast<int16_t>(blk);
  args.b.scale_layout = swizzled ? kops::ScaleLayout::kSwizzled128x4 : kops::ScaleLayout::kRowMajor;
  if (global) args.b.global_scale = dg.p;
  args.d = dout.p; args.ldd = n; args.d_dtype = dt_of<T>();
  args.m = m; args.n = n; args.k = k;
  KOPS_EXPECT_OK(kops::gemm::scaled_gemm(args, nullptr, kops::gemm::GemmBackend::kNative, kops::Workspace{}, ctx), what);
  cudaDeviceSynchronize();
  expect_close(kops_test::copy_out(dout.p, ref.size()), ref, what);
}

void test_fp4_gemv(const LaunchContext& ctx) {
  for (int m : {1, 2, 3, 5, 8}) {
    run_fp4<__nv_bfloat16>(ctx, false, false, m, 130, 256, false, "mxfp4 row-major bf16");
    run_fp4<__half>(ctx, false, true, m, 200, 512, false, "mxfp4 swizzled f16");
    run_fp4<__nv_bfloat16>(ctx, true, false, m, 77, 320, true, "nvfp4 row-major + global scale");
    run_fp4<__half>(ctx, true, true, m, 260, 1024, true, "nvfp4 swizzled + global scale");
  }
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
  KOPS_EXPECT_STATUS(call(GGUFType::kQ4_0, 17, 16, 64, DType::kBFloat16), Status::kNotImplemented, "gguf M > 16");
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
  test_w4a16(ctx);
  test_lut_gemv(ctx);
  test_w4a16_contracts(ctx);
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
  test_e4m3_encode_exhaustive();
  test_block_quant(ctx);
  test_contracts(ctx);
  return kops_test::finish("test_gemm");
}
