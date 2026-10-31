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

}  // namespace

int main() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return 77;
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
