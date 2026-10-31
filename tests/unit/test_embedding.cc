// =============================================================================
// kernel-ops | tests/unit/test_embedding.cc
// B1.2 embedding family: embedding_lookup, embedding_with_position, gather_rows,
// scatter_rows and logits_postprocess.
// Oracle: fp64 arithmetic on the quantized inputs (no framework, no reference
// TU). Contract cases: dtype / ndim / stride / alignment / vocab range / alias.
// Graph replay: lookup recorded into a caller-owned CUDA graph and replayed.
// =============================================================================

#include "tests/unit/harness.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

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
#include "csrc/embedding/embedding_kernels.cuh"
#include "csrc/embedding/ops.h"

namespace {

void fill(std::vector<float>& values, unsigned seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& v : values) v = dist(rng);
}

template <typename T>
std::vector<T> cast_from_float(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) kops_test::from_float(out[i], src[i]);
  return out;
}

template <typename T>
std::vector<double> to_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<double>(kops_test::to_float(src[i]));
  return out;
}

kops::Tensor tensor1d(void* data, kops::DType dtype, int64_t n) {
  kops::Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

/// Copy the first `hidden` columns of every padded row into a dense vector.
template <typename T>
std::vector<T> compact_rows(const std::vector<T>& padded, int64_t rows, int64_t hidden,
                            int64_t stride) {
  std::vector<T> dense(static_cast<size_t>(rows * hidden));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t j = 0; j < hidden; ++j) dense[r * hidden + j] = padded[r * stride + j];
  }
  return dense;
}

// -----------------------------------------------------------------------------
// embedding_lookup
// -----------------------------------------------------------------------------

template <typename T>
void test_lookup(const char* name, kops::DType dtype, double atol, double rtol) {
  const int64_t tokens = 6;
  const int64_t hidden = 256;
  const int64_t vocab_start = 128;
  const int64_t vocab_end = 384;
  const int64_t vocab_rows = vocab_end - vocab_start;
  const float scale = 16.0f;  // Gemma 1-3 scale = sqrt(hidden)
  // in shard, first/last shard row, below the shard, above the shard, inside.
  const std::vector<int32_t> ids = {128, 129, 383, 127, 384, 328};

  std::vector<float> w_f(static_cast<size_t>(vocab_rows * hidden));
  fill(w_f, 21u, -1.0f, 1.0f);
  const std::vector<T> w = cast_from_float<T>(w_f);

  T* d_w = kops_test::device_alloc<T>(w.size());
  T* d_out = kops_test::device_alloc<T>(static_cast<size_t>(tokens * hidden));
  int32_t* d_ids = kops_test::device_alloc<int32_t>(ids.size());
  int64_t* d_ids64 = kops_test::device_alloc<int64_t>(ids.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_w, w);
  kops_test::copy_in(d_ids, ids);
  kops_test::copy_in(d_ids64, std::vector<int64_t>(ids.begin(), ids.end()));

  kops::Tensor t_out = kops_test::tensor2d(d_out, dtype, tokens, hidden);
  kops::Tensor t_w = kops_test::tensor2d(d_w, dtype, vocab_rows, hidden);
  kops::Tensor t_ids = tensor1d(d_ids, kops::DType::kInt32, tokens);
  kops::Tensor t_ids64 = tensor1d(d_ids64, kops::DType::kInt64, tokens);

  // fp64 oracle: scale * W[id - vocab_start] inside the shard, zeros outside.
  const std::vector<double> w_d = to_double(w);
  std::vector<double> ref(static_cast<size_t>(tokens * hidden), 0.0);
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t id = ids[t];
    if (id < vocab_start || id >= vocab_end) continue;
    for (int64_t j = 0; j < hidden; ++j) {
      ref[t * hidden + j] = static_cast<double>(scale) * w_d[(id - vocab_start) * hidden + j];
    }
  }

  KOPS_EXPECT_OK(
      kops::embed::embedding_lookup(t_out, t_ids, t_w, vocab_start, vocab_end, scale, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  kops_test::compare(kops_test::copy_out(d_out, static_cast<size_t>(tokens * hidden)), ref, atol,
                     rtol, name);

  // int64 ids must take the identical path.
  KOPS_EXPECT_OK(
      kops::embed::embedding_lookup(t_out, t_ids64, t_w, vocab_start, vocab_end, scale, ctx),
      "embedding_lookup int64 ids");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  kops_test::compare(kops_test::copy_out(d_out, static_cast<size_t>(tokens * hidden)), ref, atol,
                     rtol, "embedding_lookup int64 ids");

  // scale = 1 with a padded weight row is a pure copy: it must be bit-exact in
  // the valid columns and leave the padding alone.
  std::vector<float> wpad_f(static_cast<size_t>(vocab_rows * (hidden + 8)));
  fill(wpad_f, 22u, -1.0f, 1.0f);
  const std::vector<float> wpad = wpad_f;
  float* d_wpad = kops_test::device_alloc<float>(wpad.size());
  float* d_oupad = kops_test::device_alloc<float>(static_cast<size_t>(tokens * (hidden + 8)));
  kops_test::copy_in(d_wpad, wpad);
  cudaMemset(d_oupad, 0, static_cast<size_t>(tokens * (hidden + 8)) * sizeof(float));
  kops::Tensor t_wpad = kops_test::tensor2d(d_wpad, kops::DType::kFloat32, vocab_rows, hidden);
  t_wpad.stride[0] = hidden + 8;
  kops::Tensor t_oupad = kops_test::tensor2d(d_oupad, kops::DType::kFloat32, tokens, hidden);
  t_oupad.stride[0] = hidden + 8;
  KOPS_EXPECT_OK(
      kops::embed::embedding_lookup(t_oupad, t_ids, t_wpad, vocab_start, vocab_end, 1.0f, ctx),
      "embedding_lookup padded stride");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto padded =
      kops_test::copy_out(d_oupad, static_cast<size_t>(tokens * (hidden + 8)));
  const auto dense = compact_rows(padded, tokens, hidden, hidden + 8);
  std::vector<double> ref_copy(static_cast<size_t>(tokens * hidden), 0.0);
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t id = ids[t];
    if (id < vocab_start || id >= vocab_end) continue;
    for (int64_t j = 0; j < hidden; ++j) {
      ref_copy[t * hidden + j] = static_cast<double>(wpad[(id - vocab_start) * (hidden + 8) + j]);
    }
  }
  kops_test::compare(dense, ref_copy, 0.0, 0.0, "embedding_lookup copy");

  kops_test::device_free(d_w);
  kops_test::device_free(d_out);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_ids64);
  kops_test::device_free(d_wpad);
  kops_test::device_free(d_oupad);
}

/// hidden = 70 with a 72-element row stride: 17 16-byte bundles plus a 2-element
/// scalar tail per row.
void test_lookup_tail() {
  const int64_t tokens = 3;
  const int64_t hidden = 70;
  const int64_t stride = 72;
  const int64_t vocab_rows = 16;
  const float scale = 2.0f;  // power of two, so the product is exact
  const std::vector<int32_t> ids = {0, 9, 15};

  std::vector<float> w_f(static_cast<size_t>(vocab_rows * stride), 0.0f);
  fill(w_f, 23u, -1.0f, 1.0f);
  float* d_w = kops_test::device_alloc<float>(w_f.size());
  float* d_out = kops_test::device_alloc<float>(static_cast<size_t>(tokens * stride));
  int32_t* d_ids = kops_test::device_alloc<int32_t>(ids.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_w, w_f);
  kops_test::copy_in(d_ids, ids);
  cudaMemset(d_out, 0, static_cast<size_t>(tokens * stride) * sizeof(float));

  kops::Tensor t_out = kops_test::tensor2d(d_out, kops::DType::kFloat32, tokens, hidden);
  t_out.stride[0] = stride;
  kops::Tensor t_w = kops_test::tensor2d(d_w, kops::DType::kFloat32, vocab_rows, hidden);
  t_w.stride[0] = stride;
  kops::Tensor t_ids = tensor1d(d_ids, kops::DType::kInt32, tokens);

  KOPS_EXPECT_OK(kops::embed::embedding_lookup(t_out, t_ids, t_w, 0, vocab_rows, scale, ctx),
                 "embedding_lookup tail");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto padded = kops_test::copy_out(d_out, static_cast<size_t>(tokens * stride));
  const auto dense = compact_rows(padded, tokens, hidden, stride);
  std::vector<double> ref(static_cast<size_t>(tokens * hidden));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t j = 0; j < hidden; ++j) {
      ref[t * hidden + j] = static_cast<double>(scale) * static_cast<double>(w_f[ids[t] * stride + j]);
    }
  }
  kops_test::compare(dense, ref, 0.0, 0.0, "embedding_lookup tail");

  kops_test::device_free(d_w);
  kops_test::device_free(d_out);
  kops_test::device_free(d_ids);
}

// -----------------------------------------------------------------------------
// embedding_with_position
// -----------------------------------------------------------------------------

template <typename T>
void test_with_position(const char* name, kops::DType dtype, double atol, double rtol) {
  const int64_t tokens = 5;
  const int64_t hidden = 128;
  const int64_t vocab_start = 0;
  const int64_t vocab_end = 64;  // TP shard of a 64-row table
  const int64_t vocab_rows = vocab_end - vocab_start;
  const int64_t pos_rows = 32;
  const int32_t position_offset = 2;  // OPT stores position 0 at row 2
  const std::vector<int32_t> ids = {0, 63, 64, 5, 900};          // 64 and 900 are outside
  const std::vector<int64_t> positions = {0, 10, 29, 1, 3};      // +2 -> {2,12,31,3,5}

  std::vector<float> tw_f(static_cast<size_t>(vocab_rows * hidden));
  std::vector<float> pw_f(static_cast<size_t>(pos_rows * hidden));
  fill(tw_f, 31u, -1.0f, 1.0f);
  fill(pw_f, 32u, -1.0f, 1.0f);
  const std::vector<T> tw = cast_from_float<T>(tw_f);
  const std::vector<T> pw = cast_from_float<T>(pw_f);

  T* d_tw = kops_test::device_alloc<T>(tw.size());
  T* d_pw = kops_test::device_alloc<T>(pw.size());
  T* d_out = kops_test::device_alloc<T>(static_cast<size_t>(tokens * hidden));
  int32_t* d_ids = kops_test::device_alloc<int32_t>(ids.size());
  int64_t* d_pos = kops_test::device_alloc<int64_t>(positions.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_tw, tw);
  kops_test::copy_in(d_pw, pw);
  kops_test::copy_in(d_ids, ids);
  kops_test::copy_in(d_pos, positions);

  kops::Tensor t_out = kops_test::tensor2d(d_out, dtype, tokens, hidden);
  kops::Tensor t_tw = kops_test::tensor2d(d_tw, dtype, vocab_rows, hidden);
  kops::Tensor t_pw = kops_test::tensor2d(d_pw, dtype, pos_rows, hidden);
  kops::Tensor t_ids = tensor1d(d_ids, kops::DType::kInt32, tokens);
  kops::Tensor t_pos = tensor1d(d_pos, kops::DType::kInt64, tokens);

  const std::vector<double> tw_d = to_double(tw);
  const std::vector<double> pw_d = to_double(pw);
  for (int first = 1; first >= 0; --first) {
    std::vector<double> ref(static_cast<size_t>(tokens * hidden), 0.0);
    for (int64_t t = 0; t < tokens; ++t) {
      const int64_t id = ids[t];
      const bool in_shard = id >= vocab_start && id < vocab_end;
      const int64_t pos = positions[t] + position_offset;
      const bool has_pos = first != 0 && pos >= 0 && pos < pos_rows;
      for (int64_t j = 0; j < hidden; ++j) {
        const double a = in_shard ? tw_d[(id - vocab_start) * hidden + j] : 0.0;
        const double b = has_pos ? pw_d[pos * hidden + j] : 0.0;
        ref[t * hidden + j] = a + b;
      }
    }
    const char* what = first != 0 ? name : "embedding_with_position no position";
    KOPS_EXPECT_OK(kops::embed::embedding_with_position(t_out, t_ids, t_pos, t_tw, t_pw, vocab_start,
                                                       vocab_end, position_offset, first != 0, ctx),
                   what);
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    kops_test::compare(kops_test::copy_out(d_out, static_cast<size_t>(tokens * hidden)), ref, atol,
                       rtol, what);
  }

  kops_test::device_free(d_tw);
  kops_test::device_free(d_pw);
  kops_test::device_free(d_out);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_pos);
}

// -----------------------------------------------------------------------------
// gather_rows / scatter_rows
// -----------------------------------------------------------------------------

/// hidden = 98 with a 100-element row stride exercises the bundle + scalar tail.
void test_gather_rows_tail() {
  const int64_t in_rows = 8;
  const int64_t out_rows = 5;
  const int64_t hidden = 98;
  const int64_t stride = 100;
  const std::vector<int32_t> rows = {3, 0, 7, 3, 3};  // duplicates are allowed

  std::vector<float> in_f(static_cast<size_t>(in_rows * stride), 0.0f);
  fill(in_f, 41u, -1.0f, 1.0f);
  float* d_in = kops_test::device_alloc<float>(in_f.size());
  float* d_out = kops_test::device_alloc<float>(static_cast<size_t>(out_rows * stride));
  int32_t* d_rows = kops_test::device_alloc<int32_t>(rows.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_in, in_f);
  kops_test::copy_in(d_rows, rows);
  cudaMemset(d_out, 0, static_cast<size_t>(out_rows * stride) * sizeof(float));

  kops::Tensor t_in = kops_test::tensor2d(d_in, kops::DType::kFloat32, in_rows, hidden);
  t_in.stride[0] = stride;
  kops::Tensor t_out = kops_test::tensor2d(d_out, kops::DType::kFloat32, out_rows, hidden);
  t_out.stride[0] = stride;
  kops::Tensor t_rows = tensor1d(d_rows, kops::DType::kInt32, out_rows);

  KOPS_EXPECT_OK(kops::embed::gather_rows(t_out, t_in, t_rows, ctx), "gather_rows tail");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto padded = kops_test::copy_out(d_out, static_cast<size_t>(out_rows * stride));
  const auto dense = compact_rows(padded, out_rows, hidden, stride);
  std::vector<double> ref(static_cast<size_t>(out_rows * hidden));
  for (int64_t r = 0; r < out_rows; ++r) {
    for (int64_t j = 0; j < hidden; ++j) {
      ref[r * hidden + j] = static_cast<double>(in_f[rows[r] * stride + j]);
    }
  }
  kops_test::compare(dense, ref, 0.0, 0.0, "gather_rows tail");

  kops_test::device_free(d_in);
  kops_test::device_free(d_out);
  kops_test::device_free(d_rows);
}

template <typename T>
void test_gather_rows(const char* name, kops::DType dtype) {
  const int64_t in_rows = 4;
  const int64_t out_rows = 4;
  const int64_t hidden = 128;
  const std::vector<int32_t> rows = {2, 1, 0, 2};

  std::vector<float> in_f(static_cast<size_t>(in_rows * hidden));
  fill(in_f, 42u, -1.0f, 1.0f);
  const std::vector<T> in = cast_from_float<T>(in_f);

  T* d_in = kops_test::device_alloc<T>(in.size());
  T* d_out = kops_test::device_alloc<T>(static_cast<size_t>(out_rows * hidden));
  int32_t* d_rows = kops_test::device_alloc<int32_t>(rows.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_in, in);
  kops_test::copy_in(d_rows, rows);
  kops::Tensor t_in = kops_test::tensor2d(d_in, dtype, in_rows, hidden);
  kops::Tensor t_out = kops_test::tensor2d(d_out, dtype, out_rows, hidden);
  kops::Tensor t_rows = tensor1d(d_rows, kops::DType::kInt32, out_rows);

  KOPS_EXPECT_OK(kops::embed::gather_rows(t_out, t_in, t_rows, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(d_out, static_cast<size_t>(out_rows * hidden));
  const std::vector<double> in_d = to_double(in);
  std::vector<double> ref(static_cast<size_t>(out_rows * hidden));
  for (int64_t r = 0; r < out_rows; ++r) {
    for (int64_t j = 0; j < hidden; ++j) ref[r * hidden + j] = in_d[rows[r] * hidden + j];
  }
  kops_test::compare(got, ref, 0.0, 0.0, name);

  kops_test::device_free(d_in);
  kops_test::device_free(d_out);
  kops_test::device_free(d_rows);
}

template <typename T>
void test_scatter_rows(const char* name, kops::DType dtype) {
  const int64_t out_rows = 8;
  const int64_t src_rows = 5;
  const int64_t hidden = 128;
  const std::vector<int32_t> dst = {6, 1, 0, 4, 2};  // unique destinations

  std::vector<float> src_f(static_cast<size_t>(src_rows * hidden));
  fill(src_f, 43u, -1.0f, 1.0f);
  const std::vector<T> src = cast_from_float<T>(src_f);
  std::vector<float> out_f(static_cast<size_t>(out_rows * hidden), -7.0f);
  const std::vector<T> out_init = cast_from_float<T>(out_f);

  T* d_out = kops_test::device_alloc<T>(out_init.size());
  T* d_src = kops_test::device_alloc<T>(src.size());
  int32_t* d_dst = kops_test::device_alloc<int32_t>(dst.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_out, out_init);
  kops_test::copy_in(d_src, src);
  kops_test::copy_in(d_dst, dst);
  kops::Tensor t_out = kops_test::tensor2d(d_out, dtype, out_rows, hidden);
  kops::Tensor t_src = kops_test::tensor2d(d_src, dtype, src_rows, hidden);
  kops::Tensor t_dst = tensor1d(d_dst, kops::DType::kInt32, src_rows);

  KOPS_EXPECT_OK(kops::embed::scatter_rows(t_out, t_src, t_dst, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(d_out, static_cast<size_t>(out_rows * hidden));
  const std::vector<double> src_d = to_double(src);
  const std::vector<double> init_d = to_double(out_init);
  std::vector<double> ref = init_d;
  for (int64_t r = 0; r < src_rows; ++r) {
    for (int64_t j = 0; j < hidden; ++j) ref[dst[r] * hidden + j] = src_d[r * hidden + j];
  }
  kops_test::compare(got, ref, 0.0, 0.0, name);

  kops_test::device_free(d_out);
  kops_test::device_free(d_src);
  kops_test::device_free(d_dst);
}

// -----------------------------------------------------------------------------
// logits_postprocess
// -----------------------------------------------------------------------------

template <typename T>
void test_logits(const char* name, kops::DType dtype) {
  const int64_t rows = 4;
  const int64_t v_shard = 128;
  const int64_t vocab_start = 1000;  // this shard covers [1000, 1128)
  const int64_t vocab_size = 1100;   // ... but only 100 of its columns are real
  const float scale = 0.5f;
  const float softcap = 30.0f;

  std::vector<float> x_f(static_cast<size_t>(rows * v_shard));
  fill(x_f, 51u, -8.0f, 8.0f);
  const std::vector<T> x = cast_from_float<T>(x_f);

  T* d_x = kops_test::device_alloc<T>(x.size());
  float* d_out = kops_test::device_alloc<float>(static_cast<size_t>(rows * v_shard));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_x, x);
  kops::Tensor t_in = kops_test::tensor2d(d_x, dtype, rows, v_shard);
  kops::Tensor t_out = kops_test::tensor2d(d_out, kops::DType::kFloat32, rows, v_shard);

  const std::vector<double> x_d = to_double(x);
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> ref(static_cast<size_t>(rows * v_shard));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < v_shard; ++c) {
      const size_t i = static_cast<size_t>(r * v_shard + c);
      if (vocab_start + c >= vocab_size) {
        ref[i] = -inf;  // TP vocabulary padding
      } else {
        const double y = static_cast<double>(scale) * x_d[i];
        ref[i] = softcap > 0.0f ? static_cast<double>(softcap) * std::tanh(y / softcap) : y;
      }
    }
  }

  KOPS_EXPECT_OK(
      kops::embed::logits_postprocess(t_out, t_in, scale, softcap, vocab_start, vocab_size, ctx),
      name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(d_out, static_cast<size_t>(rows * v_shard));
  kops_test::compare(got, ref, 1e-5, 1e-5, name);

  // softcap = 0 disables the cap (pure scaled logits).
  std::vector<double> ref_plain(ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    const int64_t c = static_cast<int64_t>(i % v_shard);
    ref_plain[i] = vocab_start + c >= vocab_size ? -inf : static_cast<double>(scale) * x_d[i];
  }
  KOPS_EXPECT_OK(
      kops::embed::logits_postprocess(t_out, t_in, scale, 0.0f, vocab_start, vocab_size, ctx),
      "logits_postprocess no softcap");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  kops_test::compare(kops_test::copy_out(d_out, static_cast<size_t>(rows * v_shard)), ref_plain,
                     1e-5, 1e-5, "logits_postprocess no softcap");

  kops_test::device_free(d_x);
  kops_test::device_free(d_out);
}

/// One row whose width is not a multiple of four: the kernel must fall back to
/// the scalar loop (no 16-byte column blocks are possible).
void test_logits_scalar_path() {
  const int64_t rows = 1;
  const int64_t v_shard = 98;
  const int64_t vocab_size = 90;  // the last 8 columns are padding
  const float scale = 2.0f;       // power of two, so the product is exact

  std::vector<float> x_f(static_cast<size_t>(rows * v_shard));
  fill(x_f, 53u, -6.0f, 6.0f);
  const std::vector<__nv_bfloat16> x = cast_from_float<__nv_bfloat16>(x_f);

  __nv_bfloat16* d_x = kops_test::device_alloc<__nv_bfloat16>(x.size());
  float* d_out = kops_test::device_alloc<float>(x.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_x, x);

  kops::Tensor t_in = kops_test::tensor2d(d_x, kops::DType::kBFloat16, rows, v_shard);
  kops::Tensor t_out = kops_test::tensor2d(d_out, kops::DType::kFloat32, rows, v_shard);

  KOPS_EXPECT_OK(kops::embed::logits_postprocess(t_out, t_in, scale, 0.0f, 0, vocab_size, ctx),
                 "logits_postprocess scalar path");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<double> x_d = to_double(x);
  std::vector<double> ref(x_d.size());
  for (int64_t c = 0; c < v_shard; ++c) {
    ref[c] = c >= vocab_size ? -std::numeric_limits<double>::infinity()
                             : static_cast<double>(scale) * x_d[c];
  }
  kops_test::compare(kops_test::copy_out(d_out, x.size()), ref, 0.0, 0.0,
                     "logits_postprocess scalar path");

  kops_test::device_free(d_x);
  kops_test::device_free(d_out);
}

/// fp32 in place: `out` and `logits` are the same buffer and must match the
/// out-of-place result bit for bit.
void test_logits_in_place() {
  const int64_t rows = 3;
  const int64_t v_shard = 96;
  const int64_t vocab_start = 0;
  const int64_t vocab_size = 96;
  const float scale = 1.5f;
  const float softcap = 30.0f;

  std::vector<float> x_f(static_cast<size_t>(rows * v_shard));
  fill(x_f, 52u, -4.0f, 4.0f);

  float* d_x = kops_test::device_alloc<float>(x_f.size());
  float* d_out = kops_test::device_alloc<float>(x_f.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  kops_test::copy_in(d_x, x_f);
  kops::Tensor t_x = kops_test::tensor2d(d_x, kops::DType::kFloat32, rows, v_shard);
  kops::Tensor t_out = kops_test::tensor2d(d_out, kops::DType::kFloat32, rows, v_shard);

  KOPS_EXPECT_OK(
      kops::embed::logits_postprocess(t_out, t_x, scale, softcap, vocab_start, vocab_size, ctx),
      "logits_postprocess separate");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto want = kops_test::copy_out(d_out, x_f.size());

  kops_test::copy_in(d_x, x_f);  // restore the input, then run on the same buffer
  KOPS_EXPECT_OK(
      kops::embed::logits_postprocess(t_x, t_x, scale, softcap, vocab_start, vocab_size, ctx),
      "logits_postprocess in place");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(d_x, x_f.size());
  kops_test::compare(got, std::vector<double>(want.begin(), want.end()), 0.0, 0.0,
                     "logits_postprocess in place");

  kops_test::device_free(d_x);
  kops_test::device_free(d_out);
}

// -----------------------------------------------------------------------------
// Contract errors
// -----------------------------------------------------------------------------

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  std::vector<float> data(4096, 1.0f);
  float* d = kops_test::device_alloc<float>(data.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(16);
  int64_t* d_ids64 = kops_test::device_alloc<int64_t>(16);
  kops_test::copy_in(d, data);
  cudaMemset(d_ids, 0, 16 * sizeof(int32_t));
  cudaMemset(d_ids64, 0, 16 * sizeof(int64_t));

  const kops::Tensor t_rows = tensor1d(d_ids, kops::DType::kInt32, 4);
  const kops::Tensor t_rows64 = tensor1d(d_ids64, kops::DType::kInt64, 4);
  const kops::Tensor t_f32 = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  const kops::Tensor t_f16 = kops_test::tensor2d(d, kops::DType::kFloat16, 4, 256);
  const kops::Tensor t_w = kops_test::tensor2d(d, kops::DType::kFloat32, 16, 256);

  // token ids must be int32/int64
  const kops::Tensor t_rows_f32 = tensor1d(d, kops::DType::kFloat32, 4);
  KOPS_EXPECT_STATUS(kops::embed::embedding_lookup(t_f32, t_rows_f32, t_w, 0, 16, 1.0f, ctx),
                     kops::Status::kUnsupportedDtype, "embedding_lookup ids dtype");
  // out and weight dtype must agree
  KOPS_EXPECT_STATUS(kops::embed::embedding_lookup(t_f16, t_rows, t_w, 0, 16, 1.0f, ctx),
                     kops::Status::kUnsupportedDtype, "embedding_lookup out dtype");
  // shard must fit the weight table
  KOPS_EXPECT_STATUS(kops::embed::embedding_lookup(t_f32, t_rows, t_w, 0, 17, 1.0f, ctx),
                     kops::Status::kInvalidArgument, "embedding_lookup vocab range");
  // scale must be finite
  KOPS_EXPECT_STATUS(
      kops::embed::embedding_lookup(t_f32, t_rows, t_w, 0, 16,
                                    std::numeric_limits<float>::quiet_NaN(), ctx),
      kops::Status::kInvalidArgument, "embedding_lookup scale");
  // ndim must be 2
  kops::Tensor t3 = kops_test::tensor3d(d, kops::DType::kFloat32, 1, 4, 256);
  KOPS_EXPECT_STATUS(kops::embed::embedding_lookup(t3, t_rows, t_w, 0, 16, 1.0f, ctx),
                     kops::Status::kInvalidArgument, "embedding_lookup ndim");
  // 16-byte alignment
  kops::Tensor t_mis = kops_test::tensor2d(reinterpret_cast<char*>(d) + 4, kops::DType::kFloat32, 4,
                                           256);
  KOPS_EXPECT_STATUS(kops::embed::embedding_lookup(t_mis, t_rows, t_w, 0, 16, 1.0f, ctx),
                     kops::Status::kMisaligned, "embedding_lookup alignment");
  // unit column stride
  kops::Tensor t_col = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  t_col.stride[1] = 2;
  KOPS_EXPECT_STATUS(kops::embed::embedding_lookup(t_col, t_rows, t_w, 0, 16, 1.0f, ctx),
                     kops::Status::kUnsupportedLayout, "embedding_lookup column stride");
  // index count must match the row count
  const kops::Tensor t_rows16 = tensor1d(d_ids, kops::DType::kInt32, 16);
  KOPS_EXPECT_STATUS(kops::embed::embedding_lookup(t_f32, t_rows16, t_w, 0, 16, 1.0f, ctx),
                     kops::Status::kInvalidArgument, "embedding_lookup index count");
  // output must not overlap the table
  kops::Tensor t_alias_in = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  kops::Tensor t_alias_tab = kops_test::tensor2d(reinterpret_cast<char*>(d) + 16,
                                                 kops::DType::kFloat32, 4, 256);
  KOPS_EXPECT_STATUS(kops::embed::embedding_lookup(t_alias_in, t_rows, t_alias_tab, 0, 4, 1.0f, ctx),
                     kops::Status::kUnsupportedLayout, "embedding_lookup overlap");

  // positions must be int64
  KOPS_EXPECT_STATUS(kops::embed::embedding_with_position(t_f32, t_rows, t_rows, t_f32, t_f32, 0, 4,
                                                         0, true, ctx),
                     kops::Status::kUnsupportedDtype, "embedding_with_position positions dtype");
  // position_offset must be nonnegative
  KOPS_EXPECT_STATUS(kops::embed::embedding_with_position(t_f32, t_rows, t_rows64, t_f32, t_f32, 0,
                                                         4, -1, true, ctx),
                     kops::Status::kInvalidArgument, "embedding_with_position offset");
  // dense rows only: the device contract has no stride argument
  kops::Tensor t_pad = kops_test::tensor2d(d, kops::DType::kFloat32, 4, 256);
  t_pad.stride[0] = 264;
  KOPS_EXPECT_STATUS(
      kops::embed::embedding_with_position(t_pad, t_rows, t_rows64, t_f32, t_f32, 0, 4, 0, true, ctx),
      kops::Status::kUnsupportedLayout, "embedding_with_position stride");

  // row indices must be int32
  KOPS_EXPECT_STATUS(kops::embed::gather_rows(t_f32, t_w, t_rows64, ctx),
                     kops::Status::kUnsupportedDtype, "gather_rows index dtype");
  // width mismatch
  const kops::Tensor t_w128 = kops_test::tensor2d(d, kops::DType::kFloat32, 16, 128);
  KOPS_EXPECT_STATUS(kops::embed::gather_rows(t_f32, t_w128, t_rows, ctx),
                     kops::Status::kInvalidArgument, "gather_rows width");
  // destination index count must match the source rows
  KOPS_EXPECT_STATUS(kops::embed::scatter_rows(t_w, t_f32, t_rows16, ctx),
                     kops::Status::kInvalidArgument, "scatter_rows index count");

  // out of logits_postprocess must be fp32
  KOPS_EXPECT_STATUS(kops::embed::logits_postprocess(t_f16, t_f32, 1.0f, 0.0f, 0, 256, ctx),
                     kops::Status::kUnsupportedDtype, "logits_postprocess out dtype");
  // final softcap must be finite and nonnegative
  KOPS_EXPECT_STATUS(kops::embed::logits_postprocess(t_f32, t_f32, 1.0f, -1.0f, 0, 256, ctx),
                     kops::Status::kInvalidArgument, "logits_postprocess softcap");
  // shapes must match
  KOPS_EXPECT_STATUS(kops::embed::logits_postprocess(t_f32, kops_test::tensor2d(
                                                                 d, kops::DType::kFloat32, 2, 256),
                                                     1.0f, 0.0f, 0, 256, ctx),
                     kops::Status::kInvalidArgument, "logits_postprocess shape");
  // a partial overlap is rejected (only the exact in-place view is allowed)
  kops::Tensor t_part = kops_test::tensor2d(reinterpret_cast<char*>(d) + 32, kops::DType::kFloat32,
                                            4, 256);
  KOPS_EXPECT_STATUS(kops::embed::logits_postprocess(t_f32, t_part, 1.0f, 0.0f, 0, 256, ctx),
                     kops::Status::kUnsupportedLayout, "logits_postprocess overlap");

  // zero tokens is a successful no-op
  kops::Tensor t_empty = kops_test::tensor2d(d, kops::DType::kFloat32, 0, 256);
  KOPS_EXPECT_OK(kops::embed::embedding_lookup(t_empty, tensor1d(d_ids, kops::DType::kInt32, 0),
                                               t_w, 0, 16, 1.0f, ctx),
                 "embedding_lookup empty");

  kops_test::device_free(d);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_ids64);
}

// -----------------------------------------------------------------------------
// CUDA graph replay
// -----------------------------------------------------------------------------

void test_graph_replay() {
  const int64_t tokens = 4;
  const int64_t hidden = 128;
  const int64_t vocab_rows = 32;
  const std::vector<int32_t> ids = {0, 31, 5, 9};

  std::vector<float> w_f(static_cast<size_t>(vocab_rows * hidden));
  fill(w_f, 61u, -1.0f, 1.0f);
  float* d_w = kops_test::device_alloc<float>(w_f.size());
  float* d_out = kops_test::device_alloc<float>(static_cast<size_t>(tokens * hidden));
  int32_t* d_ids = kops_test::device_alloc<int32_t>(ids.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_w, w_f);
  kops_test::copy_in(d_ids, ids);
  cudaMemset(d_out, 0, static_cast<size_t>(tokens * hidden) * sizeof(float));

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    kops_test::record_failure("cudaStreamCreate", cudaGetErrorString(cudaGetLastError()), __FILE__,
                              __LINE__);
    return;
  }
  ctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);

  const kops::Tensor t_out = kops_test::tensor2d(d_out, kops::DType::kFloat32, tokens, hidden);
  const kops::Tensor t_w = kops_test::tensor2d(d_w, kops::DType::kFloat32, vocab_rows, hidden);
  const kops::Tensor t_ids = tensor1d(d_ids, kops::DType::kInt32, tokens);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
  KOPS_TEST_CHECK(ok);
  if (ok) {
    KOPS_EXPECT_OK(kops::embed::embedding_lookup(t_out, t_ids, t_w, 0, vocab_rows, 3.0f, ctx),
                   "graph capture lookup");
    ok = cudaStreamEndCapture(stream, &graph) == cudaSuccess;
    KOPS_TEST_CHECK(ok);
  }
  if (ok) {
    ok = cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    KOPS_TEST_CHECK(ok);
  }
  if (ok) {
    cudaMemsetAsync(d_out, 0, static_cast<size_t>(tokens * hidden) * sizeof(float), stream);
    ok = cudaGraphLaunch(exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
    KOPS_TEST_CHECK(ok);
  }
  if (ok) {
    std::vector<double> ref(static_cast<size_t>(tokens * hidden));
    for (int64_t t = 0; t < tokens; ++t) {
      for (int64_t j = 0; j < hidden; ++j) {
        // The kernel multiplies in fp32, so round the oracle the same way.
        ref[t * hidden + j] = static_cast<double>(3.0f * w_f[ids[t] * hidden + j]);
      }
    }
    kops_test::compare(kops_test::copy_out(d_out, static_cast<size_t>(tokens * hidden)), ref, 0.0,
                       0.0, "graph replay lookup");
  }

  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  kops_test::device_free(d_w);
  kops_test::device_free(d_out);
  kops_test::device_free(d_ids);
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;

  test_lookup<float>("embedding_lookup fp32", kops::DType::kFloat32, 1e-5, 1e-5);
  test_lookup<__half>("embedding_lookup fp16", kops::DType::kFloat16, 2e-2, 2e-2);
  test_lookup<__nv_bfloat16>("embedding_lookup bf16", kops::DType::kBFloat16, 2e-2, 2e-2);
  test_lookup_tail();
  test_with_position<float>("embedding_with_position fp32", kops::DType::kFloat32, 1e-5, 1e-5);
  test_with_position<__nv_bfloat16>("embedding_with_position bf16", kops::DType::kBFloat16, 2e-2,
                                    2e-2);
  test_gather_rows_tail();
  test_gather_rows<__nv_bfloat16>("gather_rows bf16", kops::DType::kBFloat16);
  test_scatter_rows<float>("scatter_rows fp32", kops::DType::kFloat32);
  test_scatter_rows<__half>("scatter_rows fp16", kops::DType::kFloat16);
  test_logits<float>("logits_postprocess fp32", kops::DType::kFloat32);
  test_logits<__half>("logits_postprocess fp16", kops::DType::kFloat16);
  test_logits<__nv_bfloat16>("logits_postprocess bf16", kops::DType::kBFloat16);
  test_logits_scalar_path();
  test_logits_in_place();
  test_contracts();
  test_graph_replay();
  return kops_test::finish("test_embedding");
}
