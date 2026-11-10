// =============================================================================
// kernel-ops | tests/unit/test_sampling.cc
// B1 numerics and contract tests for S01 (softmax, argmax, compute_logprobs)
// and S02 (token_bincount, apply_penalties, apply_token_bitmask,
// apply_logit_bias). The oracle is the fp64 reference (csrc/reference) for the
// softmax family and an fp64 re-evaluation of the documented processor formulas
// for the in-place paths. No Python, no framework.
//
// The sampling headers declare device parameter types, so this TU is compiled
// by nvcc (CMake must mark it LANGUAGE CUDA, like test_m1.cc).
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
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
#include "csrc/common/rng.cuh"
#include "csrc/sampling/sampling_kernels.cuh"
#include "csrc/sampling/fused_sampler_kernels.cuh"
#include "csrc/sampling/logits_processor_kernels.cuh"
#include "csrc/sampling/speculative_kernels.cuh"
#include "csrc/sampling/ops.h"
#include "csrc/reference/ops.h"

namespace {

using kops::DType;
using kops::Tensor;

constexpr float kPosInfF = std::numeric_limits<float>::infinity();
constexpr float kNegInfF = -std::numeric_limits<float>::infinity();

void fill(std::vector<float>& v, unsigned seed, float lo = -3.0f, float hi = 3.0f) {
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

template <typename T>
std::vector<double> cast_to_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<double>(kops_test::to_float(src[i]));
  return out;
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

void expect_i32(const std::vector<int32_t>& got, const std::vector<int32_t>& want, const char* what) {
  if (got.size() != want.size()) {
    kops_test::record_failure(what, "size mismatch", __FILE__, __LINE__);
    return;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      char detail[160];
      std::snprintf(detail, sizeof(detail), "index %zu: got %d want %d", i, got[i], want[i]);
      kops_test::record_failure(what, detail, __FILE__, __LINE__);
      return;
    }
  }
}

double row_lse_f64(const double* x, int64_t n) {
  double m = -std::numeric_limits<double>::infinity();
  for (int64_t i = 0; i < n; ++i) m = std::max(m, x[i]);
  double s = 0.0;
  for (int64_t i = 0; i < n; ++i) s += std::exp(x[i] - m);
  return m + std::log(s);
}

int64_t argmax_lowest(const double* x, int64_t n) {
  int64_t best = 0;
  for (int64_t i = 1; i < n; ++i) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

// ---------------------------------------------------------------------------
// softmax (fp32, per-row temperature, greedy row, +inf / empty rows)
// ---------------------------------------------------------------------------
void test_softmax_fp32() {
  const int64_t batch = 5;
  const int64_t vocab = 1001;  // not a multiple of 4 or 8
  std::vector<float> logits_f(static_cast<size_t>(batch * vocab));
  fill(logits_f, 11u);
  // Row 3: two +inf logits. Row 4: no finite mass at all.
  for (int64_t v = 0; v < vocab; ++v) {
    logits_f[static_cast<size_t>(3 * vocab + v)] = 0.25f * static_cast<float>(v % 7) - 1.0f;
    logits_f[static_cast<size_t>(4 * vocab + v)] = kNegInfF;
  }
  logits_f[static_cast<size_t>(3 * vocab + 3)] = kPosInfF;
  logits_f[static_cast<size_t>(3 * vocab + 8)] = kPosInfF;

  const std::vector<float> temperature = {1.0f, 0.5f, -1.0f, 1.0f, 1.0f};

  float* d_logits = kops_test::device_alloc<float>(logits_f.size());
  float* d_probs = kops_test::device_alloc<float>(logits_f.size());
  float* d_temp = kops_test::device_alloc<float>(temperature.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_f);
  kops_test::copy_in(d_temp, temperature);

  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kFloat32, batch, vocab);
  const Tensor t_probs = kops_test::tensor2d(d_probs, DType::kFloat32, batch, vocab);
  const Tensor t_temp = tensor1d(d_temp, DType::kFloat32, batch);
  const kops::Workspace ws{};
  KOPS_EXPECT_OK(kops::sample::softmax(t_probs, t_logits, t_temp, ws, ctx), "softmax fp32");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto probs = kops_test::copy_out(d_probs, logits_f.size());
  const auto logits = cast_to_double(logits_f);

  // Rows 0 and 1: fp64 reference softmax with each row's temperature.
  for (int64_t r = 0; r < 2; ++r) {
    std::vector<double> ref(static_cast<size_t>(vocab));
    KOPS_EXPECT_OK(kops::ref::softmax_f64(logits.data() + r * vocab, 1, vocab,
                                         static_cast<double>(temperature[r]), ref.data()),
                   "softmax_f64");
    const std::vector<float> got(probs.begin() + r * vocab, probs.begin() + (r + 1) * vocab);
    kops_test::compare(got, ref, 1e-6, 3e-4, "softmax fp32 row");
  }

  // Row 2: temperature -1 is greedy -> one-hot at the lowest argmax id.
  {
    const int64_t winner = argmax_lowest(logits.data() + 2 * vocab, vocab);
    std::vector<double> ref(static_cast<size_t>(vocab), 0.0);
    ref[static_cast<size_t>(winner)] = 1.0;
    const std::vector<float> got(probs.begin() + 2 * vocab, probs.begin() + 3 * vocab);
    kops_test::compare(got, ref, 1e-6, 0.0, "softmax greedy row");
  }

  // Row 3: uniform over the two +inf entries.
  {
    std::vector<double> ref(static_cast<size_t>(vocab), 0.0);
    ref[3] = 0.5;
    ref[8] = 0.5;
    const std::vector<float> got(probs.begin() + 3 * vocab, probs.begin() + 4 * vocab);
    kops_test::compare(got, ref, 1e-6, 0.0, "softmax +inf row");
  }

  // Row 4: all -inf -> all zero probabilities.
  {
    const std::vector<double> ref(static_cast<size_t>(vocab), 0.0);
    const std::vector<float> got(probs.begin() + 4 * vocab, probs.begin() + 5 * vocab);
    kops_test::compare(got, ref, 0.0, 0.0, "softmax empty row");
  }

  kops_test::device_free(d_logits);
  kops_test::device_free(d_probs);
  kops_test::device_free(d_temp);
}

// fp16 / bf16 logits, plus the absent-temperature (T = 1) path.
template <typename T>
void test_softmax_half(DType dtype, const char* name) {
  const int64_t batch = 2;
  const int64_t vocab = 1001;
  std::vector<float> logits_f(static_cast<size_t>(batch * vocab));
  fill(logits_f, 21u);
  const auto logits_t = cast_from_float<T>(logits_f);
  std::vector<float> temperature = {0.8f, 1.3f};

  T* d_logits = kops_test::device_alloc<T>(logits_t.size());
  float* d_probs = kops_test::device_alloc<float>(logits_t.size());
  float* d_temp = kops_test::device_alloc<float>(temperature.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_t);
  kops_test::copy_in(d_temp, temperature);

  const Tensor t_logits = kops_test::tensor2d(d_logits, dtype, batch, vocab);
  const Tensor t_probs = kops_test::tensor2d(d_probs, DType::kFloat32, batch, vocab);
  const Tensor t_temp = tensor1d(d_temp, DType::kFloat32, batch);
  const kops::Workspace ws{};

  KOPS_EXPECT_OK(kops::sample::softmax(t_probs, t_logits, t_temp, ws, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto probs = kops_test::copy_out(d_probs, logits_t.size());
  const auto logits = cast_to_double(logits_t);
  for (int64_t r = 0; r < batch; ++r) {
    std::vector<double> ref(static_cast<size_t>(vocab));
    KOPS_EXPECT_OK(kops::ref::softmax_f64(logits.data() + r * vocab, 1, vocab,
                                         static_cast<double>(temperature[r]), ref.data()),
                   "softmax_f64 half");
    const std::vector<float> got(probs.begin() + r * vocab, probs.begin() + (r + 1) * vocab);
    kops_test::compare(got, ref, 2e-6, 3e-4, "softmax half row");
  }

  // Null temperature -> 1.0 for every row.
  KOPS_EXPECT_OK(kops::sample::softmax(t_probs, t_logits, Tensor{}, ws, ctx), "softmax no-temp");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto probs_unit = kops_test::copy_out(d_probs, logits_t.size());
  for (int64_t r = 0; r < batch; ++r) {
    std::vector<double> ref(static_cast<size_t>(vocab));
    KOPS_EXPECT_OK(kops::ref::softmax_f64(logits.data() + r * vocab, 1, vocab, 1.0, ref.data()),
                   "softmax_f64 no-temp");
    const std::vector<float> got(probs_unit.begin() + r * vocab,
                                 probs_unit.begin() + (r + 1) * vocab);
    kops_test::compare(got, ref, 2e-6, 3e-4, "softmax half no-temp row");
  }

  kops_test::device_free(d_logits);
  kops_test::device_free(d_probs);
  kops_test::device_free(d_temp);
}

// ---------------------------------------------------------------------------
// argmax (ties -> lowest id, optional logprob)
// ---------------------------------------------------------------------------
void test_argmax_fp32() {
  const int64_t batch = 2;
  const int64_t vocab = 37;
  std::vector<float> logits_f(static_cast<size_t>(batch * vocab));
  fill(logits_f, 31u, -2.0f, 2.0f);
  // Row 0: duplicated maximum. Row 1: duplicated negative maximum.
  logits_f[static_cast<size_t>(0 * vocab + 7)] = 5.0f;
  logits_f[static_cast<size_t>(0 * vocab + 21)] = 5.0f;
  for (int64_t v = 0; v < vocab; ++v) logits_f[static_cast<size_t>(1 * vocab + v)] = -3.0f;
  logits_f[static_cast<size_t>(1 * vocab + 0)] = -1.0f;
  logits_f[static_cast<size_t>(1 * vocab + 5)] = -1.0f;

  float* d_logits = kops_test::device_alloc<float>(logits_f.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
  float* d_logprob = kops_test::device_alloc<float>(static_cast<size_t>(batch));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_f);

  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kFloat32, batch, vocab);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, batch);
  const Tensor t_logprob = tensor1d(d_logprob, DType::kFloat32, batch);

  KOPS_EXPECT_OK(kops::sample::argmax(t_ids, t_logprob, t_logits, ctx), "argmax fp32");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  expect_i32(kops_test::copy_out(d_ids, static_cast<size_t>(batch)), {7, 0}, "argmax ids");

  const auto logprob = kops_test::copy_out(d_logprob, static_cast<size_t>(batch));
  const auto logits = cast_to_double(logits_f);
  for (int64_t r = 0; r < batch; ++r) {
    const double lse = row_lse_f64(logits.data() + r * vocab, vocab);
    const std::vector<float> got = {logprob[static_cast<size_t>(r)]};
    const std::vector<double> ref = {static_cast<double>(logits_f[static_cast<size_t>(r * vocab +
                                                                                     (r == 0 ? 7 : 0))]) -
                                     lse};
    kops_test::compare(got, ref, 1e-5, 1e-5, "argmax logprob");
  }

  // Absent logprob output: ids only.
  KOPS_EXPECT_OK(kops::sample::argmax(t_ids, Tensor{}, t_logits, ctx), "argmax no-logprob");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  expect_i32(kops_test::copy_out(d_ids, static_cast<size_t>(batch)), {7, 0}, "argmax ids no-logprob");

  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_logprob);
}

void test_argmax_bf16() {
  const int64_t batch = 1;
  const int64_t vocab = 129;
  std::vector<float> logits_f(static_cast<size_t>(vocab));
  fill(logits_f, 32u, -4.0f, 4.0f);
  logits_f[97] = 4.0f;
  logits_f[13] = 4.0f;
  const auto logits_t = cast_from_float<__nv_bfloat16>(logits_f);

  __nv_bfloat16* d_logits = kops_test::device_alloc<__nv_bfloat16>(logits_t.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(1);
  float* d_logprob = kops_test::device_alloc<float>(1);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_t);

  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kBFloat16, batch, vocab);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, batch);
  const Tensor t_logprob = tensor1d(d_logprob, DType::kFloat32, batch);
  KOPS_EXPECT_OK(kops::sample::argmax(t_ids, t_logprob, t_logits, ctx), "argmax bf16");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  expect_i32(kops_test::copy_out(d_ids, 1), {13}, "argmax bf16 tie");

  const auto logits = cast_to_double(logits_t);
  const double lse = row_lse_f64(logits.data(), vocab);
  const std::vector<float> got = {kops_test::copy_out(d_logprob, 1)[0]};
  const std::vector<double> ref = {logits[13] - lse};
  kops_test::compare(got, ref, 1e-4, 1e-4, "argmax bf16 logprob");

  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_logprob);
}

// ---------------------------------------------------------------------------
// compute_logprobs (chosen logprob, rank, top-n)
// ---------------------------------------------------------------------------
template <typename T>
void test_compute_logprobs(DType dtype, const char* name, int top_n, double atol, double rtol) {
  const int64_t batch = 2;
  const int64_t vocab = 257;
  std::vector<float> logits_f(static_cast<size_t>(batch * vocab));
  fill(logits_f, 41u, -5.0f, 5.0f);
  const auto logits_t = cast_from_float<T>(logits_f);
  const auto logits = cast_to_double(logits_t);

  // Chosen token: row 0 = the argmax (rank 1), row 1 = the 3rd largest.
  std::vector<std::vector<std::pair<double, int32_t>>> ranked(static_cast<size_t>(batch));
  for (int64_t r = 0; r < batch; ++r) {
    for (int64_t v = 0; v < vocab; ++v) {
      ranked[static_cast<size_t>(r)].emplace_back(logits[static_cast<size_t>(r * vocab + v)],
                                                  static_cast<int32_t>(v));
    }
    std::sort(ranked[static_cast<size_t>(r)].begin(), ranked[static_cast<size_t>(r)].end(),
              [](const std::pair<double, int32_t>& a, const std::pair<double, int32_t>& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
              });
  }
  std::vector<int32_t> chosen_ids = {ranked[0][0].second, ranked[1][2].second};

  T* d_logits = kops_test::device_alloc<T>(logits_t.size());
  int32_t* d_chosen = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
  float* d_logprob = kops_test::device_alloc<float>(static_cast<size_t>(batch));
  int32_t* d_rank = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
  const size_t top_elems = static_cast<size_t>(batch * top_n);
  int32_t* d_top_ids = kops_test::device_alloc<int32_t>(top_elems);
  float* d_top_logprobs = kops_test::device_alloc<float>(top_elems);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_t);
  kops_test::copy_in(d_chosen, chosen_ids);

  const Tensor t_logits = kops_test::tensor2d(d_logits, dtype, batch, vocab);
  const Tensor t_chosen = tensor1d(d_chosen, DType::kInt32, batch);
  const Tensor t_logprob = tensor1d(d_logprob, DType::kFloat32, batch);
  const Tensor t_rank = tensor1d(d_rank, DType::kInt32, batch);
  const Tensor t_top_ids = kops_test::tensor2d(d_top_ids, DType::kInt32, batch, top_n);
  const Tensor t_top_logprobs = kops_test::tensor2d(d_top_logprobs, DType::kFloat32, batch, top_n);

  KOPS_EXPECT_OK(kops::sample::compute_logprobs(t_logprob, t_rank, t_top_ids, t_top_logprobs,
                                                t_logits, t_chosen, top_n, ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got_logprob = kops_test::copy_out(d_logprob, static_cast<size_t>(batch));
  const auto got_rank = kops_test::copy_out(d_rank, static_cast<size_t>(batch));
  const auto got_top_ids = kops_test::copy_out(d_top_ids, top_elems);
  const auto got_top_logprobs = kops_test::copy_out(d_top_logprobs, top_elems);

  std::vector<double> ref_logprob(static_cast<size_t>(batch));
  std::vector<int32_t> ref_rank(static_cast<size_t>(batch));
  std::vector<int32_t> ref_top_ids;
  std::vector<double> ref_top_logprobs;
  for (int64_t r = 0; r < batch; ++r) {
    const double lse = row_lse_f64(logits.data() + r * vocab, vocab);
    const double chosen = logits[static_cast<size_t>(r * vocab + chosen_ids[static_cast<size_t>(r)])];
    ref_logprob[static_cast<size_t>(r)] = chosen - lse;
    int32_t greater = 0;
    for (int64_t v = 0; v < vocab; ++v) {
      if (logits[static_cast<size_t>(r * vocab + v)] > chosen) ++greater;
    }
    ref_rank[static_cast<size_t>(r)] = greater + 1;
    for (int t = 0; t < top_n; ++t) {
      ref_top_ids.push_back(ranked[static_cast<size_t>(r)][static_cast<size_t>(t)].second);
      ref_top_logprobs.push_back(ranked[static_cast<size_t>(r)][static_cast<size_t>(t)].first - lse);
    }
  }

  kops_test::compare(got_logprob, ref_logprob, atol, rtol, "compute_logprobs.chosen_logprob");
  expect_i32(got_rank, ref_rank, "compute_logprobs.chosen_rank");
  expect_i32(got_top_ids, ref_top_ids, "compute_logprobs.top_ids");
  kops_test::compare(got_top_logprobs, ref_top_logprobs, atol, rtol,
                     "compute_logprobs.top_logprobs");

  // top_n = 0: only the chosen statistics, no top tensors.
  KOPS_EXPECT_OK(kops::sample::compute_logprobs(t_logprob, t_rank, Tensor{}, Tensor{}, t_logits,
                                                t_chosen, 0, ctx),
                 "compute_logprobs top_n 0");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got_logprob0 = kops_test::copy_out(d_logprob, static_cast<size_t>(batch));
  const auto got_rank0 = kops_test::copy_out(d_rank, static_cast<size_t>(batch));
  kops_test::compare(got_logprob0, ref_logprob, atol, rtol, "compute_logprobs top_n0 logprob");
  expect_i32(got_rank0, ref_rank, "compute_logprobs top_n0 rank");

  kops_test::device_free(d_logits);
  kops_test::device_free(d_chosen);
  kops_test::device_free(d_logprob);
  kops_test::device_free(d_rank);
  kops_test::device_free(d_top_ids);
  kops_test::device_free(d_top_logprobs);
}

// ---------------------------------------------------------------------------
// token_bincount (ragged ids, prompt vs output, out-of-vocab ids)
// ---------------------------------------------------------------------------
void test_token_bincount() {
  const int64_t batch = 3;
  const int64_t vocab = 64;
  const int64_t words = (vocab + 31) / 32;
  const int64_t num_ids = 12;
  const std::vector<int32_t> indptr = {0, 5, 9, 12};
  const std::vector<int32_t> ids = {1, 2, 2, 3, 63, 0, 0, 63, 64, 5, 5, 5};
  const std::vector<uint8_t> is_prompt = {1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0};

  int32_t* d_counts = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch * vocab));
  uint32_t* d_mask = kops_test::device_alloc<uint32_t>(static_cast<size_t>(batch * words));
  int32_t* d_ids = kops_test::device_alloc<int32_t>(ids.size());
  int32_t* d_indptr = kops_test::device_alloc<int32_t>(indptr.size());
  uint8_t* d_is_prompt = kops_test::device_alloc<uint8_t>(is_prompt.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_TEST_CHECK(cudaMemset(d_counts, 0, static_cast<size_t>(batch * vocab) * sizeof(int32_t)) ==
                  cudaSuccess);
  KOPS_TEST_CHECK(cudaMemset(d_mask, 0, static_cast<size_t>(batch * words) * sizeof(uint32_t)) ==
                  cudaSuccess);
  kops_test::copy_in(d_ids, ids);
  kops_test::copy_in(d_indptr, indptr);
  kops_test::copy_in(d_is_prompt, is_prompt);

  const Tensor t_counts = kops_test::tensor2d(d_counts, DType::kInt32, batch, vocab);
  const Tensor t_mask = kops_test::tensor2d(d_mask, DType::kUInt32, batch, words);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, num_ids);
  const Tensor t_indptr = tensor1d(d_indptr, DType::kInt32, batch + 1);
  const Tensor t_is_prompt = tensor1d(d_is_prompt, DType::kUInt8, num_ids);

  KOPS_EXPECT_OK(kops::sample::token_bincount(t_counts, t_mask, t_ids, t_indptr, t_is_prompt, ctx),
                 "token_bincount prompt/output");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<int32_t> want_counts(static_cast<size_t>(batch * vocab), 0);
  std::vector<uint32_t> want_mask(static_cast<size_t>(batch * words), 0u);
  for (int64_t r = 0; r < batch; ++r) {
    for (int64_t i = indptr[static_cast<size_t>(r)]; i < indptr[static_cast<size_t>(r + 1)]; ++i) {
      const int32_t id = ids[static_cast<size_t>(i)];
      if (id < 0 || id >= vocab) continue;
      if (is_prompt[static_cast<size_t>(i)] != 0) {
        want_mask[static_cast<size_t>(r * words + (id >> 5))] |= (1u << (id & 31));
      } else {
        ++want_counts[static_cast<size_t>(r * vocab + id)];
      }
    }
  }
  expect_i32(kops_test::copy_out(d_counts, want_counts.size()), want_counts, "token_bincount counts");
  const auto got_mask = kops_test::copy_out(d_mask, want_mask.size());
  for (size_t i = 0; i < want_mask.size(); ++i) {
    if (got_mask[i] != want_mask[i]) {
      kops_test::record_failure("token_bincount mask", "bit mismatch", __FILE__, __LINE__);
      break;
    }
  }
  // Explicit spot checks: prompt-only tokens stay out of the output counts.
  const auto counts = kops_test::copy_out(d_counts, want_counts.size());
  KOPS_TEST_CHECK(counts[static_cast<size_t>(0 * vocab + 1)] == 0);
  KOPS_TEST_CHECK(counts[static_cast<size_t>(0 * vocab + 2)] == 1);
  KOPS_TEST_CHECK(counts[static_cast<size_t>(1 * vocab + 63)] == 0);
  KOPS_TEST_CHECK((got_mask[static_cast<size_t>(1 * words + (63 >> 5))] >> (63 & 31)) & 1u);

  // No is_prompt: every id is an output token (id 64 is still out of range).
  KOPS_TEST_CHECK(cudaMemset(d_counts, 0, static_cast<size_t>(batch * vocab) * sizeof(int32_t)) ==
                  cudaSuccess);
  KOPS_TEST_CHECK(cudaMemset(d_mask, 0, static_cast<size_t>(batch * words) * sizeof(uint32_t)) ==
                  cudaSuccess);
  KOPS_EXPECT_OK(kops::sample::token_bincount(t_counts, t_mask, t_ids, t_indptr, Tensor{}, ctx),
                 "token_bincount all-output");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto counts_all = kops_test::copy_out(d_counts, want_counts.size());
  const auto mask_all = kops_test::copy_out(d_mask, want_mask.size());
  for (size_t i = 0; i < mask_all.size(); ++i) KOPS_TEST_CHECK(mask_all[i] == 0u);
  KOPS_TEST_CHECK(counts_all[static_cast<size_t>(0 * vocab + 1)] == 1);
  KOPS_TEST_CHECK(counts_all[static_cast<size_t>(0 * vocab + 2)] == 2);
  KOPS_TEST_CHECK(counts_all[static_cast<size_t>(1 * vocab + 63)] == 1);
  KOPS_TEST_CHECK(counts_all[static_cast<size_t>(2 * vocab + 5)] == 3);

  kops_test::device_free(d_counts);
  kops_test::device_free(d_mask);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_indptr);
  kops_test::device_free(d_is_prompt);
}

// ---------------------------------------------------------------------------
// apply_penalties
// ---------------------------------------------------------------------------
template <typename T>
void test_penalties(DType dtype, const char* name, double atol, double rtol) {
  const int64_t batch = 3;
  const int64_t vocab = 16;
  const int64_t words = (vocab + 31) / 32;
  std::vector<float> logits_f(static_cast<size_t>(batch * vocab));
  fill(logits_f, 51u, -2.0f, 2.0f);
  const auto logits_t = cast_from_float<T>(logits_f);

  std::vector<int32_t> counts(static_cast<size_t>(batch * vocab), 0);
  counts[static_cast<size_t>(0 * vocab + 0)] = 1;
  counts[static_cast<size_t>(0 * vocab + 1)] = 2;
  counts[static_cast<size_t>(1 * vocab + 3)] = 1;
  std::vector<uint32_t> mask(static_cast<size_t>(batch * words), 0u);
  mask[static_cast<size_t>(0 * words + 0)] |= (1u << 2);   // prompt-only token 2
  mask[static_cast<size_t>(1 * words + 0)] |= (1u << 4);   // prompt-only token 4
  const std::vector<float> repetition = {2.0f, 1.0f, 1.0f};
  const std::vector<float> frequency = {0.5f, 0.0f, 0.0f};
  const std::vector<float> presence = {1.0f, 0.0f, 0.0f};

  T* d_logits = kops_test::device_alloc<T>(logits_t.size());
  int32_t* d_counts = kops_test::device_alloc<int32_t>(counts.size());
  uint32_t* d_mask = kops_test::device_alloc<uint32_t>(mask.size());
  float* d_rep = kops_test::device_alloc<float>(repetition.size());
  float* d_freq = kops_test::device_alloc<float>(frequency.size());
  float* d_pres = kops_test::device_alloc<float>(presence.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_t);
  kops_test::copy_in(d_counts, counts);
  kops_test::copy_in(d_mask, mask);
  kops_test::copy_in(d_rep, repetition);
  kops_test::copy_in(d_freq, frequency);
  kops_test::copy_in(d_pres, presence);

  const Tensor t_logits = kops_test::tensor2d(d_logits, dtype, batch, vocab);
  const Tensor t_counts = kops_test::tensor2d(d_counts, DType::kInt32, batch, vocab);
  const Tensor t_mask = kops_test::tensor2d(d_mask, DType::kUInt32, batch, words);
  const Tensor t_rep = tensor1d(d_rep, DType::kFloat32, batch);
  const Tensor t_freq = tensor1d(d_freq, DType::kFloat32, batch);
  const Tensor t_pres = tensor1d(d_pres, DType::kFloat32, batch);

  KOPS_EXPECT_OK(kops::sample::apply_penalties(t_logits, t_counts, t_mask, t_rep, t_freq, t_pres,
                                               ctx),
                 name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(d_logits, logits_t.size());
  const auto base = cast_to_double(logits_t);
  std::vector<double> ref(base.size());
  for (int64_t r = 0; r < batch; ++r) {
    for (int64_t v = 0; v < vocab; ++v) {
      const size_t i = static_cast<size_t>(r * vocab + v);
      const bool prompt = ((mask[static_cast<size_t>(r * words + (v >> 5))] >> (v & 31)) & 1u) != 0u;
      const int32_t count = counts[i];
      const bool seen = prompt || count > 0;
      double l = base[i];
      if (seen) l = l > 0.0 ? l / repetition[static_cast<size_t>(r)]
                            : l * repetition[static_cast<size_t>(r)];
      if (count > 0) {
        l -= static_cast<double>(frequency[static_cast<size_t>(r)]) * count;
        l -= static_cast<double>(presence[static_cast<size_t>(r)]);
      }
      ref[i] = l;
    }
  }
  kops_test::compare(got, ref, atol, rtol, name);

  kops_test::device_free(d_logits);
  kops_test::device_free(d_counts);
  kops_test::device_free(d_mask);
  kops_test::device_free(d_rep);
  kops_test::device_free(d_freq);
  kops_test::device_free(d_pres);
}

// ---------------------------------------------------------------------------
// apply_token_bitmask (partial last word, optional row map)
// ---------------------------------------------------------------------------
template <typename T>
void test_bitmask(DType dtype, const char* name) {
  const int64_t rows = 3;
  const int64_t mask_rows = 2;
  const int64_t vocab = 100;  // 4 words, the last one only carries bits 0..3
  const int64_t words = (vocab + 31) / 32;
  std::vector<float> logits_f(static_cast<size_t>(rows * vocab));
  fill(logits_f, 61u, -2.0f, 2.0f);
  const auto logits_t = cast_from_float<T>(logits_f);

  std::vector<uint32_t> mask(static_cast<size_t>(mask_rows * words), 0u);
  // Mask row 0 allows {0, 1, 2, 5, 31} | {32} | {96 + 3}.
  mask[0] |= (1u << 0) | (1u << 1) | (1u << 2) | (1u << 5) | (1u << 31);
  mask[1] |= (1u << 0);
  mask[3] |= (1u << 3);
  // Mask row 1 allows everything except id 7.
  for (int64_t w = 0; w < words; ++w) mask[static_cast<size_t>(words + w)] = 0xffffffffu;
  mask[static_cast<size_t>(words + 0)] &= ~(1u << 7);
  const std::vector<int32_t> row_map = {2, 0};

  T* d_logits = kops_test::device_alloc<T>(logits_t.size());
  uint32_t* d_mask = kops_test::device_alloc<uint32_t>(mask.size());
  int32_t* d_rows = kops_test::device_alloc<int32_t>(row_map.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_t);
  kops_test::copy_in(d_mask, mask);
  kops_test::copy_in(d_rows, row_map);

  const Tensor t_logits = kops_test::tensor2d(d_logits, dtype, rows, vocab);
  const Tensor t_mask = kops_test::tensor2d(d_mask, DType::kInt32, mask_rows, words);
  const Tensor t_rows = tensor1d(d_rows, DType::kInt32, mask_rows);

  KOPS_EXPECT_OK(kops::sample::apply_token_bitmask(t_logits, t_mask, t_rows, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(d_logits, logits_t.size());
  const auto base = cast_to_double(logits_t);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t m = 0; m < mask_rows; ++m) {
      if (row_map[static_cast<size_t>(m)] != r) continue;
      for (int64_t v = 0; v < vocab; ++v) {
        const size_t i = static_cast<size_t>(r * vocab + v);
        const bool allowed =
            ((mask[static_cast<size_t>(m * words + (v >> 5))] >> (v & 31)) & 1u) != 0u;
        if (allowed) {
          KOPS_TEST_CHECK(kops_test::to_float(got[i]) == base[i]);
        } else {
          KOPS_TEST_CHECK(!std::isfinite(kops_test::to_float(got[i])));
        }
      }
    }
  }
  // Row 1 carries no mask and must be untouched.
  for (int64_t v = 0; v < vocab; ++v) {
    KOPS_TEST_CHECK(kops_test::to_float(got[static_cast<size_t>(vocab + v)]) ==
                    base[static_cast<size_t>(vocab + v)]);
  }
  // Partial last word: id 99 is bit 3 of word 3 (allowed by mask row 0), ids
  // 96 and 98 are clear (disallowed), and bits past id 99 do not matter.
  KOPS_TEST_CHECK(std::isfinite(kops_test::to_float(got[static_cast<size_t>(2 * vocab + 99)])));
  KOPS_TEST_CHECK(!std::isfinite(kops_test::to_float(got[static_cast<size_t>(2 * vocab + 96)])));
  KOPS_TEST_CHECK(!std::isfinite(kops_test::to_float(got[static_cast<size_t>(2 * vocab + 98)])));

  kops_test::device_free(d_logits);
  kops_test::device_free(d_mask);
  kops_test::device_free(d_rows);
}

void test_bitmask_identity_rows() {
  const int64_t rows = 2;
  const int64_t vocab = 70;  // 3 words, last word bits 0..5
  const int64_t words = (vocab + 31) / 32;
  std::vector<float> logits_f(static_cast<size_t>(rows * vocab));
  fill(logits_f, 62u);
  std::vector<uint32_t> mask(static_cast<size_t>(rows * words), 0u);
  for (int64_t v = 0; v < vocab; v += 2) {
    mask[static_cast<size_t>(0 * words + (v >> 5))] |= (1u << (v & 31));  // row 0: even ids
  }
  for (int64_t w = 0; w < words; ++w) mask[static_cast<size_t>(words + w)] = 0xffffffffu;

  __nv_bfloat16* d_logits =
      kops_test::device_alloc<__nv_bfloat16>(static_cast<size_t>(rows * vocab));
  uint32_t* d_mask = kops_test::device_alloc<uint32_t>(mask.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const auto logits_t = cast_from_float<__nv_bfloat16>(logits_f);
  kops_test::copy_in(d_logits, logits_t);
  kops_test::copy_in(d_mask, mask);

  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kBFloat16, rows, vocab);
  const Tensor t_mask = kops_test::tensor2d(d_mask, DType::kInt32, rows, words);
  KOPS_EXPECT_OK(kops::sample::apply_token_bitmask(t_logits, t_mask, Tensor{}, ctx),
                 "bitmask identity rows");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(d_logits, static_cast<size_t>(rows * vocab));
  for (int64_t v = 0; v < vocab; ++v) {
    const bool allowed = (v % 2) == 0;
    KOPS_TEST_CHECK(std::isfinite(kops_test::to_float(got[static_cast<size_t>(v)])) == allowed);
    KOPS_TEST_CHECK(std::isfinite(kops_test::to_float(got[static_cast<size_t>(vocab + v)])));
  }
  kops_test::device_free(d_logits);
  kops_test::device_free(d_mask);
}

// ---------------------------------------------------------------------------
// apply_logit_bias (sparse, per-row sequential, duplicate ids)
// ---------------------------------------------------------------------------
template <typename T>
void test_logit_bias(DType dtype, const char* name, double atol, double rtol) {
  const int64_t batch = 2;
  const int64_t vocab = 32;
  const int64_t nnz = 7;
  std::vector<float> logits_f(static_cast<size_t>(batch * vocab));
  fill(logits_f, 71u, -1.0f, 1.0f);
  const auto logits_t = cast_from_float<T>(logits_f);
  const std::vector<int32_t> indptr = {0, 4, 7};
  const std::vector<int32_t> ids = {3, 3, 5, -1, 0, 31, 7};
  const std::vector<float> values = {0.5f, 0.25f, 1.0f, 2.0f, -1.0f, 2.0f, 0.125f};

  T* d_logits = kops_test::device_alloc<T>(logits_t.size());
  int32_t* d_indptr = kops_test::device_alloc<int32_t>(indptr.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(ids.size());
  float* d_values = kops_test::device_alloc<float>(values.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_t);
  kops_test::copy_in(d_indptr, indptr);
  kops_test::copy_in(d_ids, ids);
  kops_test::copy_in(d_values, values);

  const Tensor t_logits = kops_test::tensor2d(d_logits, dtype, batch, vocab);
  const Tensor t_indptr = tensor1d(d_indptr, DType::kInt32, batch + 1);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, nnz);
  const Tensor t_values = tensor1d(d_values, DType::kFloat32, nnz);
  KOPS_EXPECT_OK(kops::sample::apply_logit_bias(t_logits, t_indptr, t_ids, t_values, ctx), name);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  const auto got = kops_test::copy_out(d_logits, logits_t.size());
  const auto base = cast_to_double(logits_t);
  std::vector<double> ref = base;
  for (int64_t r = 0; r < batch; ++r) {
    for (int32_t i = indptr[static_cast<size_t>(r)]; i < indptr[static_cast<size_t>(r + 1)]; ++i) {
      const int32_t id = ids[static_cast<size_t>(i)];
      if (id < 0 || id >= vocab) continue;
      ref[static_cast<size_t>(r * vocab + id)] += static_cast<double>(values[static_cast<size_t>(i)]);
    }
  }
  kops_test::compare(got, ref, atol, rtol, name);
  // Duplicate id 3 in row 0 adds both entries (0.5 + 0.25) in order.
  KOPS_TEST_CHECK(std::fabs(kops_test::to_float(got[static_cast<size_t>(3)]) -
                            (static_cast<float>(base[3]) + 0.75f)) <= atol);

  kops_test::device_free(d_logits);
  kops_test::device_free(d_indptr);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_values);
}

// ---------------------------------------------------------------------------
// Vectorized (16-byte pack) row loads: vocab divisible by the pack width.
// ---------------------------------------------------------------------------
template <typename T>
void test_vectorized_paths(DType dtype, const char* tag) {
  const int64_t batch = 2;
  const int64_t vocab = 1024;  // divisible by 4 (fp32) and 8 (16-bit)
  std::vector<float> logits_f(static_cast<size_t>(batch * vocab));
  fill(logits_f, 81u, -4.0f, 4.0f);
  for (int64_t r = 0; r < batch; ++r) {
    logits_f[static_cast<size_t>(r * vocab + 3)] = 4.0f;
    logits_f[static_cast<size_t>(r * vocab + 900)] = 4.0f;
  }
  const auto logits_t = cast_from_float<T>(logits_f);
  const auto logits = cast_to_double(logits_t);
  const std::vector<float> temperature = {0.7f, 1.4f};

  T* d_logits = kops_test::device_alloc<T>(logits_t.size());
  float* d_probs = kops_test::device_alloc<float>(logits_t.size());
  float* d_temp = kops_test::device_alloc<float>(temperature.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
  float* d_logprob = kops_test::device_alloc<float>(static_cast<size_t>(batch));
  int32_t* d_chosen = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
  float* d_chosen_lp = kops_test::device_alloc<float>(static_cast<size_t>(batch));
  int32_t* d_rank = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
  const int64_t top_n = 4;
  int32_t* d_top_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch * top_n));
  float* d_top_lp = kops_test::device_alloc<float>(static_cast<size_t>(batch * top_n));
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_t);
  kops_test::copy_in(d_temp, temperature);

  const Tensor t_logits = kops_test::tensor2d(d_logits, dtype, batch, vocab);
  const Tensor t_probs = kops_test::tensor2d(d_probs, DType::kFloat32, batch, vocab);
  const Tensor t_temp = tensor1d(d_temp, DType::kFloat32, batch);
  const kops::Workspace ws{};

  KOPS_EXPECT_OK(kops::sample::softmax(t_probs, t_logits, t_temp, ws, ctx), tag);
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto probs = kops_test::copy_out(d_probs, logits_t.size());
  for (int64_t r = 0; r < batch; ++r) {
    std::vector<double> ref(static_cast<size_t>(vocab));
    KOPS_EXPECT_OK(kops::ref::softmax_f64(logits.data() + r * vocab, 1, vocab,
                                         static_cast<double>(temperature[r]), ref.data()),
                   "softmax_f64 vec");
    const std::vector<float> got(probs.begin() + r * vocab, probs.begin() + (r + 1) * vocab);
    kops_test::compare(got, ref, 2e-6, 3e-4, "softmax vec row");
  }

  // Argmax over the vectorized row: ids 3 and 900 tie, lowest wins.
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, batch);
  const Tensor t_logprob = tensor1d(d_logprob, DType::kFloat32, batch);
  KOPS_EXPECT_OK(kops::sample::argmax(t_ids, t_logprob, t_logits, ctx), "argmax vec");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  expect_i32(kops_test::copy_out(d_ids, static_cast<size_t>(batch)), {3, 3}, "argmax vec ids");
  const auto logprob = kops_test::copy_out(d_logprob, static_cast<size_t>(batch));
  for (int64_t r = 0; r < batch; ++r) {
    const double lse = row_lse_f64(logits.data() + r * vocab, vocab);
    const std::vector<float> got = {logprob[static_cast<size_t>(r)]};
    const std::vector<double> ref = {logits[static_cast<size_t>(r * vocab + 3)] - lse};
    kops_test::compare(got, ref, 1e-5, 1e-5, "argmax vec logprob");
  }

  // compute_logprobs over the vectorized row: chosen = 3rd largest of row 0.
  std::vector<std::vector<std::pair<double, int32_t>>> ranked(static_cast<size_t>(batch));
  for (int64_t r = 0; r < batch; ++r) {
    for (int64_t v = 0; v < vocab; ++v) {
      ranked[static_cast<size_t>(r)].emplace_back(logits[static_cast<size_t>(r * vocab + v)],
                                                  static_cast<int32_t>(v));
    }
    std::sort(ranked[static_cast<size_t>(r)].begin(), ranked[static_cast<size_t>(r)].end(),
              [](const std::pair<double, int32_t>& a, const std::pair<double, int32_t>& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
              });
  }
  std::vector<int32_t> chosen = {ranked[0][2].second, ranked[1][0].second};
  kops_test::copy_in(d_chosen, chosen);
  const Tensor t_chosen = tensor1d(d_chosen, DType::kInt32, batch);
  const Tensor t_chosen_lp = tensor1d(d_chosen_lp, DType::kFloat32, batch);
  const Tensor t_rank = tensor1d(d_rank, DType::kInt32, batch);
  const Tensor t_top_ids = kops_test::tensor2d(d_top_ids, DType::kInt32, batch, top_n);
  const Tensor t_top_lp = kops_test::tensor2d(d_top_lp, DType::kFloat32, batch, top_n);
  KOPS_EXPECT_OK(kops::sample::compute_logprobs(t_chosen_lp, t_rank, t_top_ids, t_top_lp,
                                                t_logits, t_chosen, static_cast<int32_t>(top_n),
                                                ctx),
                 "compute_logprobs vec");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got_chosen = kops_test::copy_out(d_chosen_lp, static_cast<size_t>(batch));
  const auto got_rank = kops_test::copy_out(d_rank, static_cast<size_t>(batch));
  const auto got_top_ids = kops_test::copy_out(d_top_ids, static_cast<size_t>(batch * top_n));
  const auto got_top_lp = kops_test::copy_out(d_top_lp, static_cast<size_t>(batch * top_n));
  std::vector<double> ref_chosen(static_cast<size_t>(batch));
  std::vector<int32_t> ref_rank(static_cast<size_t>(batch));
  std::vector<int32_t> ref_top_ids;
  std::vector<double> ref_top_lp;
  for (int64_t r = 0; r < batch; ++r) {
    const double lse = row_lse_f64(logits.data() + r * vocab, vocab);
    const double chosen_value =
        logits[static_cast<size_t>(r * vocab + chosen[static_cast<size_t>(r)])];
    ref_chosen[static_cast<size_t>(r)] = chosen_value - lse;
    int32_t greater = 0;
    for (int64_t v = 0; v < vocab; ++v) {
      if (logits[static_cast<size_t>(r * vocab + v)] > chosen_value) ++greater;
    }
    ref_rank[static_cast<size_t>(r)] = greater + 1;
    for (int64_t t = 0; t < top_n; ++t) {
      ref_top_ids.push_back(ranked[static_cast<size_t>(r)][static_cast<size_t>(t)].second);
      ref_top_lp.push_back(ranked[static_cast<size_t>(r)][static_cast<size_t>(t)].first - lse);
    }
  }
  kops_test::compare(got_chosen, ref_chosen, 1e-4, 1e-4, "compute_logprobs vec chosen");
  expect_i32(got_rank, ref_rank, "compute_logprobs vec rank");
  expect_i32(got_top_ids, ref_top_ids, "compute_logprobs vec top ids");
  kops_test::compare(got_top_lp, ref_top_lp, 1e-4, 1e-4, "compute_logprobs vec top logprobs");

  kops_test::device_free(d_logits);
  kops_test::device_free(d_probs);
  kops_test::device_free(d_temp);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_logprob);
  kops_test::device_free(d_chosen);
  kops_test::device_free(d_chosen_lp);
  kops_test::device_free(d_rank);
  kops_test::device_free(d_top_ids);
  kops_test::device_free(d_top_lp);
}

// ---------------------------------------------------------------------------
// contract errors
// ---------------------------------------------------------------------------
void test_softmax_split() {
  const int64_t batch = 4;
  const int64_t vocab = 131072;
  std::vector<float> logits_f(static_cast<size_t>(batch * vocab));
  fill(logits_f, 123u, -2.0f, 2.0f);
  for (int64_t v = 0; v < vocab; ++v) {
    logits_f[static_cast<size_t>(3 * vocab + v)] = kNegInfF;
  }
  logits_f[static_cast<size_t>(2 * vocab + 3)] = kPosInfF;
  logits_f[static_cast<size_t>(2 * vocab + 8)] = kPosInfF;
  const auto logits_t = cast_from_float<__nv_bfloat16>(logits_f);
  const auto logits_d = cast_to_double(logits_t);
  const std::vector<float> temperature = {0.7f, -1.0f, 1.0f, 1.0f};

  __nv_bfloat16* d_logits = kops_test::device_alloc<__nv_bfloat16>(logits_t.size());
  float* d_probs = kops_test::device_alloc<float>(logits_t.size());
  float* d_temp = kops_test::device_alloc<float>(temperature.size());
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_logits, logits_t);
  kops_test::copy_in(d_temp, temperature);

  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kBFloat16, batch, vocab);
  const Tensor t_probs = kops_test::tensor2d(d_probs, DType::kFloat32, batch, vocab);
  const Tensor t_temp = tensor1d(d_temp, DType::kFloat32, batch);

  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::sample::sampling_workspace_requirements(
                     static_cast<int32_t>(batch), static_cast<int32_t>(vocab), 0, &req),
                 "split workspace requirements");
  KOPS_TEST_CHECK(req.device_scratch_bytes ==
                  static_cast<size_t>(batch * vocab) * sizeof(float) +
                      static_cast<size_t>(batch) * sizeof(float) +
                      static_cast<size_t>(batch) * 4u * 2u * sizeof(float));
  void* scratch = nullptr;
  KOPS_TEST_CHECK(cudaMalloc(&scratch, req.device_scratch_bytes) == cudaSuccess);
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = req.device_scratch_bytes;

  KOPS_EXPECT_OK(kops::sample::softmax(t_probs, t_logits, t_temp, ws, ctx), "softmax split 4");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto probs = kops_test::copy_out(d_probs, logits_t.size());

  // Normal row vs fp64.
  std::vector<double> ref(static_cast<size_t>(vocab));
  KOPS_EXPECT_OK(kops::ref::softmax_f64(logits_d.data(), 1, vocab, 0.7, ref.data()),
                 "split softmax fp64");
  const std::vector<float> row0(probs.begin(), probs.begin() + vocab);
  kops_test::compare(row0, ref, 2e-5, 4e-4, "softmax split normal row");

  // Greedy row.
  const int64_t winner = argmax_lowest(logits_d.data() + vocab, vocab);
  std::vector<double> greedy(static_cast<size_t>(vocab), 0.0);
  greedy[static_cast<size_t>(winner)] = 1.0;
  const std::vector<float> row1(probs.begin() + vocab, probs.begin() + 2 * vocab);
  kops_test::compare(row1, greedy, 1e-6, 0.0, "softmax split greedy row");

  // +inf row: uniform over the two +inf entries.
  std::vector<double> inf_ref(static_cast<size_t>(vocab), 0.0);
  inf_ref[3] = 0.5;
  inf_ref[8] = 0.5;
  const std::vector<float> row2(probs.begin() + 2 * vocab, probs.begin() + 3 * vocab);
  kops_test::compare(row2, inf_ref, 1e-6, 0.0, "softmax split +inf row");

  // Fully masked row: all zero.
  const std::vector<double> empty_ref(static_cast<size_t>(vocab), 0.0);
  const std::vector<float> row3(probs.begin() + 3 * vocab, probs.begin() + 4 * vocab);
  kops_test::compare(row3, empty_ref, 0.0, 0.0, "softmax split empty row");

  // Determinism: a second run writes identical bits.
  KOPS_EXPECT_OK(kops::sample::softmax(t_probs, t_logits, t_temp, ws, ctx), "softmax split repeat");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto probs2 = kops_test::copy_out(d_probs, logits_t.size());
  KOPS_TEST_CHECK(probs.size() == probs2.size() &&
                  std::memcmp(probs.data(), probs2.data(), probs.size() * sizeof(float)) == 0);

  // Missing workspace must be rejected.
  const kops::Workspace empty_ws{};
  KOPS_EXPECT_STATUS(kops::sample::softmax(t_probs, t_logits, t_temp, empty_ws, ctx),
                     kops::Status::kWorkspaceTooSmall, "softmax split no workspace");

  // CUDA graph capture/replay with the workspace fixed.
  cudaStream_t stream = nullptr;
  KOPS_TEST_CHECK(cudaStreamCreate(&stream) == cudaSuccess);
  kops::LaunchContext gctx = ctx;
  gctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  KOPS_TEST_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  KOPS_EXPECT_OK(kops::sample::softmax(t_probs, t_logits, t_temp, ws, gctx), "softmax split graph");
  cudaGraph_t graph = nullptr;
  KOPS_TEST_CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  KOPS_TEST_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
  KOPS_TEST_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
  KOPS_TEST_CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);

  cudaFree(scratch);
  kops_test::device_free(d_logits);
  kops_test::device_free(d_probs);
  kops_test::device_free(d_temp);
}

void test_contracts() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  const int64_t batch = 2;
  const int64_t vocab = 16;
  const int64_t words = (vocab + 31) / 32;
  std::vector<float> data_f(static_cast<size_t>(batch * vocab), 1.0f);
  std::vector<int32_t> counts(static_cast<size_t>(batch * vocab), 0);
  std::vector<uint32_t> mask(static_cast<size_t>(batch * words), 0u);
  std::vector<int32_t> indptr = {0, 1, 1};
  std::vector<int32_t> ids = {0};
  std::vector<float> values = {1.0f};
  std::vector<float> temperature = {1.0f, 1.0f, 1.0f};
  std::vector<int32_t> chosen = {0, 1};

  float* d_logits = kops_test::device_alloc<float>(data_f.size());
  float* d_probs = kops_test::device_alloc<float>(data_f.size());
  __half* d_half = kops_test::device_alloc<__half>(data_f.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(4);
  int32_t* d_counts = kops_test::device_alloc<int32_t>(counts.size());
  uint32_t* d_mask = kops_test::device_alloc<uint32_t>(mask.size());
  float* d_temp = kops_test::device_alloc<float>(temperature.size());
  int32_t* d_indptr = kops_test::device_alloc<int32_t>(indptr.size());
  int32_t* d_bias_ids = kops_test::device_alloc<int32_t>(ids.size());
  float* d_values = kops_test::device_alloc<float>(static_cast<size_t>(batch));
  int32_t* d_token_ids = kops_test::device_alloc<int32_t>(4);
  int32_t* d_token_indptr = kops_test::device_alloc<int32_t>(batch + 1);
  uint8_t* d_is_prompt = kops_test::device_alloc<uint8_t>(4);
  kops_test::copy_in(d_logits, data_f);
  kops_test::copy_in(d_ids, chosen);
  kops_test::copy_in(d_counts, counts);
  kops_test::copy_in(d_mask, mask);
  kops_test::copy_in(d_temp, temperature);
  kops_test::copy_in(d_indptr, indptr);
  kops_test::copy_in(d_bias_ids, ids);
  kops_test::copy_in(d_values, std::vector<float>{1.0f, -1.0f});
  kops_test::copy_in(d_token_ids, std::vector<int32_t>{0, 1, 2, 3});
  kops_test::copy_in(d_token_indptr, std::vector<int32_t>{0, 2, 4});
  kops_test::copy_in(d_is_prompt, std::vector<uint8_t>{1, 0, 1, 0});

  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kFloat32, batch, vocab);
  const Tensor t_probs = kops_test::tensor2d(d_probs, DType::kFloat32, batch, vocab);
  const Tensor t_half = kops_test::tensor2d(d_half, DType::kFloat16, batch, vocab);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, batch);
  const Tensor t_counts = kops_test::tensor2d(d_counts, DType::kInt32, batch, vocab);
  const Tensor t_mask = kops_test::tensor2d(d_mask, DType::kUInt32, batch, words);
  const Tensor t_indptr = tensor1d(d_indptr, DType::kInt32, batch + 1);
  const Tensor t_bias_ids = tensor1d(d_bias_ids, DType::kInt32, 1);
  const Tensor t_values = tensor1d(d_values, DType::kFloat32, 1);
  const Tensor t_token_ids = tensor1d(d_token_ids, DType::kInt32, 4);
  const Tensor t_token_indptr = tensor1d(d_token_indptr, DType::kInt32, batch + 1);
  const Tensor t_is_prompt = tensor1d(d_is_prompt, DType::kUInt8, 4);
  const Tensor t_top = kops_test::tensor2d(d_ids, DType::kInt32, batch, 2);
  const Tensor t_top_lp = kops_test::tensor2d(d_probs, DType::kFloat32, batch, 2);
  const kops::Workspace ws{};

  // Workspace requirements.
  KOPS_EXPECT_STATUS(kops::sample::sampling_workspace_requirements(1, 1, 1, nullptr),
                     kops::Status::kInvalidArgument, "workspace null out");
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::sample::sampling_workspace_requirements(8, 262144, 1, &req),
                 "workspace requirements");
  // 262144 tokens selects 16 CTAs/row, even when the caller asks for one.
  KOPS_TEST_CHECK(req.device_scratch_bytes ==
                  8u * 262144u * sizeof(float) + 8u * sizeof(float) +
                      8u * 16u * 2u * sizeof(float));
  KOPS_TEST_CHECK(req.device_persistent_bytes == 0);
  KOPS_TEST_CHECK(req.host_pinned_bytes == 0);
  KOPS_TEST_CHECK(req.alignment == 256);

  // softmax: dtype / shape / layout / temperature.
  KOPS_EXPECT_STATUS(kops::sample::softmax(t_half, t_logits, Tensor{}, ws, ctx),
                     kops::Status::kUnsupportedDtype, "softmax probs dtype");
  const Tensor t_probs_small = kops_test::tensor2d(d_probs, DType::kFloat32, batch, 8);
  KOPS_EXPECT_STATUS(kops::sample::softmax(t_probs_small, t_logits, Tensor{}, ws, ctx),
                     kops::Status::kInvalidArgument, "softmax probs shape");
  Tensor t_probs_strided = t_probs;
  t_probs_strided.stride[0] = vocab + 4;
  KOPS_EXPECT_STATUS(kops::sample::softmax(t_probs_strided, t_logits, Tensor{}, ws, ctx),
                     kops::Status::kUnsupportedLayout, "softmax probs stride");
  const Tensor t_temp_bad = tensor1d(d_temp, DType::kFloat32, 3);
  KOPS_EXPECT_STATUS(kops::sample::softmax(t_probs, t_logits, t_temp_bad, ws, ctx),
                     kops::Status::kInvalidArgument, "softmax temperature length");
  const Tensor t_logits_i32 = kops_test::tensor2d(d_counts, DType::kInt32, batch, vocab);
  KOPS_EXPECT_STATUS(kops::sample::softmax(t_probs, t_logits_i32, Tensor{}, ws, ctx),
                     kops::Status::kUnsupportedDtype, "softmax logits dtype");

  // argmax: id dtype.
  KOPS_EXPECT_STATUS(kops::sample::argmax(t_values, Tensor{}, t_logits, ctx),
                     kops::Status::kUnsupportedDtype, "argmax ids dtype");

  // compute_logprobs: top-n bound and missing top tensors.
  const Tensor t_lp = tensor1d(d_values, DType::kFloat32, batch);
  KOPS_EXPECT_STATUS(kops::sample::compute_logprobs(t_lp, Tensor{}, t_top, t_top_lp, t_logits,
                                                    t_ids, 21, ctx),
                     kops::Status::kUnsupportedShape, "compute_logprobs top_n bound");
  KOPS_EXPECT_STATUS(kops::sample::compute_logprobs(t_lp, Tensor{}, Tensor{}, Tensor{}, t_logits,
                                                    t_ids, 3, ctx),
                     kops::Status::kInvalidArgument, "compute_logprobs missing top");

  // token_bincount: prompt without mask, counts dtype.
  KOPS_EXPECT_STATUS(kops::sample::token_bincount(t_counts, Tensor{}, t_token_ids, t_token_indptr,
                                                  t_is_prompt, ctx),
                     kops::Status::kInvalidArgument, "token_bincount mask required");
  KOPS_EXPECT_STATUS(kops::sample::token_bincount(t_logits, t_mask, t_token_ids, t_token_indptr,
                                                  Tensor{}, ctx),
                     kops::Status::kUnsupportedDtype, "token_bincount counts dtype");

  // apply_penalties: counts shape.
  const Tensor t_counts_small = kops_test::tensor2d(d_counts, DType::kInt32, 1, vocab);
  KOPS_EXPECT_STATUS(kops::sample::apply_penalties(t_logits, t_counts_small, Tensor{}, Tensor{},
                                                   Tensor{}, Tensor{}, ctx),
                     kops::Status::kInvalidArgument, "apply_penalties counts shape");

  // apply_token_bitmask: mask width, and a row map is required when counts differ.
  const Tensor t_mask_bad = kops_test::tensor2d(d_mask, DType::kInt32, batch, words + 1);
  KOPS_EXPECT_STATUS(kops::sample::apply_token_bitmask(t_logits, t_mask_bad, Tensor{}, ctx),
                     kops::Status::kInvalidArgument, "bitmask width");
  const Tensor t_mask_one = kops_test::tensor2d(d_mask, DType::kInt32, 1, words);
  KOPS_EXPECT_STATUS(kops::sample::apply_token_bitmask(t_logits, t_mask_one, Tensor{}, ctx),
                     kops::Status::kInvalidArgument, "bitmask rows without map");

  // apply_logit_bias: indptr length, token/value length mismatch.
  const Tensor t_indptr_bad = tensor1d(d_indptr, DType::kInt32, 2);
  KOPS_EXPECT_STATUS(kops::sample::apply_logit_bias(t_logits, t_indptr_bad, t_bias_ids, t_values,
                                                    ctx),
                     kops::Status::kInvalidArgument, "logit_bias indptr length");
  const Tensor t_values_bad = tensor1d(d_values, DType::kFloat32, 0);
  KOPS_EXPECT_STATUS(kops::sample::apply_logit_bias(t_logits, t_indptr, t_bias_ids, t_values_bad,
                                                    ctx),
                     kops::Status::kInvalidArgument, "logit_bias value length");

  kops_test::device_free(d_logits);
  kops_test::device_free(d_probs);
  kops_test::device_free(d_half);
  kops_test::device_free(d_ids);
  kops_test::device_free(d_counts);
  kops_test::device_free(d_mask);
  kops_test::device_free(d_temp);
  kops_test::device_free(d_indptr);
  kops_test::device_free(d_bias_ids);
  kops_test::device_free(d_values);
  kops_test::device_free(d_token_ids);
  kops_test::device_free(d_token_indptr);
  kops_test::device_free(d_is_prompt);
}

// ---------------------------------------------------------------------------
// B6: truncation and sampling
// ---------------------------------------------------------------------------
std::vector<int> sorted_desc(const std::vector<double>& p) {
  std::vector<int> order(p.size());
  for (size_t i = 0; i < p.size(); ++i) order[i] = static_cast<int>(i);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return p[static_cast<size_t>(a)] > p[static_cast<size_t>(b)]; });
  return order;
}

std::vector<double> truncate_kpp(const std::vector<double>& prob, int k, double top_p) {
  const std::vector<int> order = sorted_desc(prob);
  std::vector<double> kept(prob.size(), 0.0);
  int nk = k > 0 && k < static_cast<int>(prob.size()) ? k : static_cast<int>(prob.size());
  double mass = 0.0;
  for (int i = 0; i < nk; ++i) mass += prob[static_cast<size_t>(order[static_cast<size_t>(i)])];
  double acc = 0.0;
  int take = nk;
  if (top_p > 0.0 && top_p < 1.0) {
    for (int i = 0; i < nk; ++i) {
      acc += prob[static_cast<size_t>(order[static_cast<size_t>(i)])];
      if (acc >= top_p * mass - 1e-15) {
        take = i + 1;
        break;
      }
    }
  }
  double z = 0.0;
  for (int i = 0; i < take; ++i) z += prob[static_cast<size_t>(order[static_cast<size_t>(i)])];
  for (int i = 0; i < take; ++i) {
    kept[static_cast<size_t>(order[static_cast<size_t>(i)])] =
        prob[static_cast<size_t>(order[static_cast<size_t>(i)])] / z;
  }
  return kept;
}

void test_truncation_probs() {
  const int64_t batch = 4;
  const int64_t vocab = 17;
  std::vector<float> probs_f(static_cast<size_t>(batch * vocab));
  std::mt19937 rng(77u);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<double> probs_d(probs_f.size());
  for (int64_t r = 0; r < batch; ++r) {
    double sum = 0.0;
    for (int64_t v = 0; v < vocab; ++v) {
      float x = dist(rng);
      if (v == 2 || v == 9) x = 0.25f;  // duplicate values exercise id tie-breaks
      probs_f[static_cast<size_t>(r * vocab + v)] = x;
      sum += static_cast<double>(x);
    }
    for (int64_t v = 0; v < vocab; ++v) {
      probs_f[static_cast<size_t>(r * vocab + v)] =
          static_cast<float>(static_cast<double>(probs_f[static_cast<size_t>(r * vocab + v)]) / sum);
      probs_d[static_cast<size_t>(r * vocab + v)] = probs_f[static_cast<size_t>(r * vocab + v)];
    }
  }
  float* d_probs = kops_test::device_alloc<float>(probs_f.size());
  float* d_out = kops_test::device_alloc<float>(probs_f.size());
  kops_test::copy_in(d_probs, probs_f);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const Tensor t_probs = kops_test::tensor2d(d_probs, DType::kFloat32, batch, vocab);
  const Tensor t_out = kops_test::tensor2d(d_out, DType::kFloat32, batch, vocab);

  // top-k renorm (k = 5).
  KOPS_EXPECT_OK(kops::sample::top_k_renorm_probs(t_out, t_probs, Tensor{}, 5, ctx),
                 "top_k_renorm_probs");
  const auto got_k = kops_test::copy_out(d_out, probs_f.size());
  for (int64_t r = 0; r < batch; ++r) {
    const std::vector<double> row(probs_d.begin() + r * vocab, probs_d.begin() + (r + 1) * vocab);
    const auto want = truncate_kpp(row, 5, 1.0);
    for (int64_t v = 0; v < vocab; ++v) {
      const double g = got_k[static_cast<size_t>(r * vocab + v)];
      const double w = want[static_cast<size_t>(v)];
      KOPS_TEST_CHECK(std::fabs(g - w) <= 1e-5 + 1e-5 * std::fabs(w));
    }
  }

  // top-p renorm (p = 0.7): compare the kept set bitwise and the values vs fp64.
  KOPS_EXPECT_OK(kops::sample::top_p_renorm_probs(t_out, t_probs, Tensor{}, 0.7f, ctx),
                 "top_p_renorm_probs");
  const auto got_p = kops_test::copy_out(d_out, probs_f.size());
  for (int64_t r = 0; r < batch; ++r) {
    const std::vector<double> row(probs_d.begin() + r * vocab, probs_d.begin() + (r + 1) * vocab);
    const auto want = truncate_kpp(row, static_cast<int>(vocab), 0.7);
    for (int64_t v = 0; v < vocab; ++v) {
      const double g = got_p[static_cast<size_t>(r * vocab + v)];
      const double w = want[static_cast<size_t>(v)];
      KOPS_TEST_CHECK((g > 0.0) == (w > 0.0));
      KOPS_TEST_CHECK(std::fabs(g - w) <= 1e-5 + 1e-5 * std::fabs(w));
    }
  }

  // top-k mask logits: kept logits unchanged, the rest exactly -inf.
  std::vector<float> logits_f(probs_f.size());
  for (size_t i = 0; i < probs_f.size(); ++i) logits_f[i] = std::log(probs_f[i] + 1e-12f);
  float* d_logits = kops_test::device_alloc<float>(logits_f.size());
  kops_test::copy_in(d_logits, logits_f);
  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kFloat32, batch, vocab);
  KOPS_EXPECT_OK(kops::sample::top_k_mask_logits(t_out, t_logits, Tensor{}, 5, ctx),
                 "top_k_mask_logits");
  const auto got_m = kops_test::copy_out(d_out, logits_f.size());
  for (int64_t r = 0; r < batch; ++r) {
    const std::vector<double> row(probs_d.begin() + r * vocab, probs_d.begin() + (r + 1) * vocab);
    const auto order = sorted_desc(row);
    for (int64_t v = 0; v < vocab; ++v) {
      bool keep = false;
      for (int i = 0; i < 5; ++i) keep = keep || order[static_cast<size_t>(i)] == v;
      const float g = got_m[static_cast<size_t>(r * vocab + v)];
      if (keep) {
        KOPS_TEST_CHECK(g == logits_f[static_cast<size_t>(r * vocab + v)]);
      } else {
        KOPS_TEST_CHECK(isinf(g) && g < 0.0f);
      }
    }
  }
  kops_test::device_free(d_probs);
  kops_test::device_free(d_out);
  kops_test::device_free(d_logits);
}

// top_k_mask_logits on 16-bit logits: 2 (bf16) / 3 (fp16) radix passes, vector and scalar paths, and
// heavy ties so the "first `need` equal keys by index" scan decides which rows survive.
template <typename T>
void test_top_k_mask_16bit(DType dtype, const char* name) {
  struct Shape { int batch, vocab, levels; };
  const Shape shapes[] = {{6, 4096, 0}, {6, 4099, 0}, {5, 8192, 9}, {5, 8195, 9}, {3, 12288, 3}};
  const int ks[] = {1, 7, 50, 333};
  std::mt19937 rng(2026u);
  for (const Shape& sh : shapes) {
    std::vector<T> logits(static_cast<size_t>(sh.batch) * sh.vocab);
    for (auto& x : logits) {
      const float f = sh.levels > 0 ? static_cast<float>(static_cast<int>(rng() % sh.levels) - sh.levels / 2) * 0.5f
                                    : (static_cast<float>(rng() % 20001) - 10000.0f) / 700.0f;
      kops_test::from_float(x, f);
    }
    T* d_in = kops_test::device_alloc<T>(logits.size());
    T* d_out = kops_test::device_alloc<T>(logits.size());
    kops_test::copy_in(d_in, logits);
    const Tensor t_in = kops_test::tensor2d(d_in, dtype, sh.batch, sh.vocab);
    const Tensor t_out = kops_test::tensor2d(d_out, dtype, sh.batch, sh.vocab);
    kops::LaunchContext ctx{};
    if (!kops_test::make_context(&ctx)) return;
    for (int k : ks) {
      KOPS_EXPECT_OK(kops::sample::top_k_mask_logits(t_out, t_in, Tensor{}, k, ctx), name);
      const auto got = kops_test::copy_out(d_out, logits.size());
      int bad = 0;
      for (int r = 0; r < sh.batch; ++r) {
        std::vector<int> order(static_cast<size_t>(sh.vocab));
        for (int i = 0; i < sh.vocab; ++i) order[static_cast<size_t>(i)] = i;
        const size_t base = static_cast<size_t>(r) * sh.vocab;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
          return kops_test::to_float(logits[base + a]) > kops_test::to_float(logits[base + b]);
        });
        std::vector<char> keep(static_cast<size_t>(sh.vocab), 0);
        for (int i = 0; i < k; ++i) keep[static_cast<size_t>(order[static_cast<size_t>(i)])] = 1;
        for (int i = 0; i < sh.vocab; ++i) {
          const float g = kops_test::to_float(got[base + i]);
          if (keep[static_cast<size_t>(i)]) bad += !(g == kops_test::to_float(logits[base + i]));
          else bad += !(std::isinf(g) && g < 0.0f);
        }
      }
      KOPS_TEST_CHECK(bad == 0);
    }
    kops_test::device_free(d_in);
    kops_test::device_free(d_out);
  }
}

// top_k_renorm_probs with tied probabilities: the float4 and scalar row paths must keep the same
// k entries (largest value, lowest index among ties) and renormalise them.
void test_top_k_renorm_ties() {
  struct Shape { int batch, vocab, levels; };
  const Shape shapes[] = {{4, 4096, 5}, {4, 4099, 5}, {3, 8192, 40}, {3, 8195, 0}, {2, 16384, 3}};  // last: bins > 2048 -> radix fallback
  const int ks[] = {1, 9, 64, 500};
  std::mt19937 rng(515u);
  for (const Shape& sh : shapes) {
    std::vector<float> probs(static_cast<size_t>(sh.batch) * sh.vocab);
    for (int r = 0; r < sh.batch; ++r) {
      double sum = 0.0;
      const size_t base = static_cast<size_t>(r) * sh.vocab;
      for (int i = 0; i < sh.vocab; ++i) {
        const double v = sh.levels > 0 ? static_cast<double>(1 + rng() % sh.levels)
                                       : static_cast<double>(1 + rng() % 100000);
        probs[base + i] = static_cast<float>(v);
        sum += v;
      }
      for (int i = 0; i < sh.vocab; ++i) probs[base + i] = static_cast<float>(probs[base + i] / sum);
    }
    float* d_in = kops_test::device_alloc<float>(probs.size());
    float* d_out = kops_test::device_alloc<float>(probs.size());
    kops_test::copy_in(d_in, probs);
    const Tensor t_in = kops_test::tensor2d(d_in, DType::kFloat32, sh.batch, sh.vocab);
    const Tensor t_out = kops_test::tensor2d(d_out, DType::kFloat32, sh.batch, sh.vocab);
    kops::LaunchContext ctx{};
    if (!kops_test::make_context(&ctx)) return;
    for (int k : ks) {
      KOPS_EXPECT_OK(kops::sample::top_k_renorm_probs(t_out, t_in, Tensor{}, k, ctx), "top_k_renorm ties");
      const auto got = kops_test::copy_out(d_out, probs.size());
      int bad = 0;
      for (int r = 0; r < sh.batch; ++r) {
        const size_t base = static_cast<size_t>(r) * sh.vocab;
        std::vector<int> order(static_cast<size_t>(sh.vocab));
        for (int i = 0; i < sh.vocab; ++i) order[static_cast<size_t>(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return probs[base + a] > probs[base + b]; });
        std::vector<char> keep(static_cast<size_t>(sh.vocab), 0);
        double kept = 0.0;
        for (int i = 0; i < k; ++i) {
          keep[static_cast<size_t>(order[static_cast<size_t>(i)])] = 1;
          kept += probs[base + order[static_cast<size_t>(i)]];
        }
        for (int i = 0; i < sh.vocab; ++i) {
          const double g = got[base + i];
          const double w = keep[static_cast<size_t>(i)] ? probs[base + i] / kept : 0.0;
          bad += std::fabs(g - w) > 1e-6 * std::fabs(w) + 1e-12 || ((g > 0.0) != (w > 0.0));
        }
      }
      KOPS_TEST_CHECK(bad == 0);
    }
    kops_test::device_free(d_in);
    kops_test::device_free(d_out);
  }
}

double chi_square(const std::vector<double>& obs, const std::vector<double>& exp, int* df) {
  double chi = 0.0;
  int n = 0;
  for (size_t i = 0; i < obs.size(); ++i) {
    if (exp[i] <= 1e-12) {
      if (obs[i] > 0.0) chi += 1e6;
      continue;
    }
    ++n;
    chi += (obs[i] - exp[i]) * (obs[i] - exp[i]) / exp[i];
  }
  *df = n > 0 ? n - 1 : 0;
  return chi;
}

void test_fused_unfiltered() {
  const int64_t batch = 2048;
  const int64_t vocab = 16;
  const int launches = 512;
  std::vector<float> logits(static_cast<size_t>(vocab));
  std::mt19937 rng(123u);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (float& x : logits) x = dist(rng);
  std::vector<float> logits_all(static_cast<size_t>(batch * vocab));
  for (int64_t r = 0; r < batch; ++r) {
    for (int64_t v = 0; v < vocab; ++v) {
      logits_all[static_cast<size_t>(r * vocab + v)] = logits[static_cast<size_t>(v)];
    }
  }
  float* d_logits = kops_test::device_alloc<float>(logits_all.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
  kops_test::copy_in(d_logits, logits_all);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kFloat32, batch, vocab);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, batch);

  kops::sample::SamplingParams params{};
  params.default_temperature = 1.0f;
  params.default_top_k = 0;
  params.default_top_p = 1.0f;
  params.default_min_p = 0.0f;
  params.rng.global_seed = 9876u;
  kops::sample::FusedSamplingParams fp{};
  fp.sampling = params;
  fp.default_repetition_penalty = 1.0f;

  kops::sample::SamplingPlan plan{};
  KOPS_EXPECT_OK(kops::sample::sampling_plan(t_logits, params, ctx, &plan),
                 "fused sampling_plan");
  KOPS_TEST_CHECK(plan.splits == 1);
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::sample::sampling_fused_workspace_requirements(
                     static_cast<int32_t>(batch), static_cast<int32_t>(vocab), plan, &req),
                 "fused workspace requirements");
  kops::Workspace ws{};
  ws.device_scratch = nullptr;
  ws.device_scratch_bytes = 0;

  // Full-vocab categorical distribution: chi-square over 1,048,576 draws.
  std::vector<double> probs_d(static_cast<size_t>(vocab));
  double mx = -1e300;
  for (float x : logits) mx = std::max(mx, static_cast<double>(x));
  double z = 0.0;
  for (int64_t v = 0; v < vocab; ++v) {
    probs_d[static_cast<size_t>(v)] =
        std::exp(static_cast<double>(logits[static_cast<size_t>(v)]) - mx);
    z += probs_d[static_cast<size_t>(v)];
  }
  for (double& p : probs_d) p /= z;
  std::vector<double> hist(static_cast<size_t>(vocab), 0.0);
  for (int it = 0; it < launches; ++it) {
    fp.sampling.rng.global_seed = 200000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::sample_fused(t_ids, t_logits, fp, plan, ws, ctx),
                   "fused categorical");
    const auto ids = kops_test::copy_out(d_ids, static_cast<size_t>(batch));
    for (int32_t id : ids) {
      KOPS_TEST_CHECK(id >= 0 && id < vocab);
      if (id >= 0 && id < vocab) hist[static_cast<size_t>(id)] += 1.0;
    }
  }
  std::vector<double> exp_counts(static_cast<size_t>(vocab));
  for (int64_t v = 0; v < vocab; ++v) {
    exp_counts[static_cast<size_t>(v)] = probs_d[static_cast<size_t>(v)] * batch * launches;
  }
  int df = 0;
  const double chi = chi_square(hist, exp_counts, &df);
  KOPS_TEST_CHECK(df >= 1);
  KOPS_TEST_CHECK(chi < 80.0);

  // Same seed reproduces the same ids.
  fp.sampling.rng.global_seed = 4242u;
  KOPS_EXPECT_OK(kops::sample::sample_fused(t_ids, t_logits, fp, plan, ws, ctx),
                 "fused categorical repeat 1");
  const auto ids1 = kops_test::copy_out(d_ids, static_cast<size_t>(batch));
  KOPS_EXPECT_OK(kops::sample::sample_fused(t_ids, t_logits, fp, plan, ws, ctx),
                 "fused categorical repeat 2");
  const auto ids2 = kops_test::copy_out(d_ids, static_cast<size_t>(batch));
  KOPS_TEST_CHECK(ids1 == ids2);

  // Greedy path: temperature 0 selects the lowest-index argmax.
  int64_t best = 0;
  for (int64_t v = 1; v < vocab; ++v) {
    if (logits[static_cast<size_t>(v)] > logits[static_cast<size_t>(best)]) best = v;
  }
  fp.sampling.default_temperature = 0.0f;
  kops::sample::SamplingPlan greedy_plan{};
  KOPS_EXPECT_OK(kops::sample::sampling_plan(t_logits, fp.sampling, ctx, &greedy_plan),
                 "fused greedy plan");
  KOPS_EXPECT_OK(kops::sample::sample_fused(t_ids, t_logits, fp, greedy_plan, ws, ctx),
                 "fused greedy");
  const auto greedy_ids = kops_test::copy_out(d_ids, static_cast<size_t>(batch));
  for (int32_t id : greedy_ids) KOPS_TEST_CHECK(id == static_cast<int32_t>(best));

  // A NaN top_p is a caller error.
  kops::sample::SamplingParams filtered = params;
  filtered.default_top_p = std::numeric_limits<float>::quiet_NaN();
  kops::sample::SamplingPlan bad_plan{};
  KOPS_EXPECT_STATUS(kops::sample::sampling_plan(t_logits, filtered, ctx, &bad_plan),
                     kops::Status::kInvalidArgument, "fused NaN top_p rejected");

  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
}

// ---- sample_fused M1: greedy / categorical / processors ---------------------------------------

/// Device copy of a host vector (freed by the caller).
template <typename T>
T* dev_copy(const std::vector<T>& v) {
  T* d = kops_test::device_alloc<T>(v.size());
  kops_test::copy_in(d, v);
  return d;
}

template <typename T>
DType fused_dtype() {
  if constexpr (std::is_same<T, float>::value) return DType::kFloat32;
  if constexpr (std::is_same<T, __half>::value) return DType::kFloat16;
  return DType::kBFloat16;
}

/// Host model of the fused processors for one row (same float operations as the kernel).
struct FusedHostProc {
  const uint8_t* mask = nullptr;   // row of the packed bitmask
  const int32_t* counts = nullptr; // row of the token counts
  float rep = 1.0f, freq = 0.0f, pres = 0.0f;
  float apply(float x, int64_t token) const {
    if (std::isnan(x)) x = -std::numeric_limits<float>::infinity();
    if (mask != nullptr && rep > 0.0f && ((mask[token >> 3] >> (token & 7)) & 1u)) {
      x = x > 0.0f ? x / rep : x * rep;
    }
    if (counts != nullptr) {
      const int32_t c = counts[token];
      x -= freq * static_cast<float>(c) + (c > 0 ? pres : 0.0f);
    }
    return x;
  }
};

/// Lowest-index argmax of the processed row, -1 when every value is -inf.
template <typename T>
int32_t fused_host_greedy(const T* row, int64_t vocab, const FusedHostProc& proc) {
  float best = -std::numeric_limits<float>::infinity();
  int32_t id = -1;
  for (int64_t v = 0; v < vocab; ++v) {
    const float y = proc.apply(kops_test::to_float(row[v]), v);
    if (y > best) {
      best = y;
      id = static_cast<int32_t>(v);
    }
  }
  return id;
}

/// One sample_fused call on a host [rows, stride] buffer; returns the ids (status via *st).
template <typename T>
std::vector<int32_t> fused_run(const std::vector<T>& host, int64_t rows, int64_t vocab,
                               int64_t stride, const kops::sample::FusedSamplingParams& fp,
                               const kops::LaunchContext& ctx, kops::Status* st = nullptr,
                               int splits = 0) {
  T* d = kops_test::device_alloc<T>(host.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(rows));
  kops_test::copy_in(d, host);
  kops_test::copy_in(d_ids, std::vector<int32_t>(static_cast<size_t>(rows), -7));
  Tensor t = kops_test::tensor2d(d, fused_dtype<T>(), rows, vocab);
  t.stride[0] = stride;
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, rows);
  kops::sample::SamplingPlan plan{};
  KOPS_EXPECT_OK(kops::sample::sampling_plan(t, fp.sampling, ctx, &plan), "fused plan");
  if (splits > 0) plan.splits = splits;  // force Path A (1) or a split-row count
  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::sample::sampling_fused_workspace_requirements(static_cast<int32_t>(rows),
                                                                      static_cast<int32_t>(vocab), plan, &req),
                 "fused workspace");
  kops::Workspace ws{};
  ws.device_scratch = req.device_scratch_bytes ? kops_test::device_alloc<uint8_t>(req.device_scratch_bytes) : nullptr;
  ws.device_scratch_bytes = req.device_scratch_bytes;
  ws.device_persistent =
      req.device_persistent_bytes ? kops_test::device_alloc<uint8_t>(req.device_persistent_bytes) : nullptr;
  ws.device_persistent_bytes = req.device_persistent_bytes;
  if (ws.device_persistent != nullptr) cudaMemset(ws.device_persistent, 0, ws.device_persistent_bytes);
  const kops::Status s = kops::sample::sample_fused(t_ids, t, fp, plan, ws, ctx);
  if (st != nullptr) {
    *st = s;
  } else {
    KOPS_EXPECT_OK(s, "sample_fused");
  }
  auto ids = kops_test::copy_out(d_ids, static_cast<size_t>(rows));
  kops_test::device_free(static_cast<uint8_t*>(ws.device_scratch));
  kops_test::device_free(static_cast<uint8_t*>(ws.device_persistent));
  kops_test::device_free(d);
  kops_test::device_free(d_ids);
  return ids;
}

/// Split count that sampling_plan() resolves for a host row buffer (the plan is a pure function of
/// the metadata and the device, so one host copy is enough to inspect it).
template <typename T>
int32_t fused_auto_splits(const std::vector<T>& host, int64_t rows, int64_t vocab,
                          const kops::sample::FusedSamplingParams& fp,
                          const kops::LaunchContext& ctx) {
  T* d = kops_test::device_alloc<T>(host.size());
  kops_test::copy_in(d, host);
  const Tensor t = kops_test::tensor2d(d, fused_dtype<T>(), rows, vocab);
  kops::sample::SamplingPlan plan{};
  KOPS_EXPECT_OK(kops::sample::sampling_plan(t, fp.sampling, ctx, &plan), "auto plan");
  kops_test::device_free(d);
  return plan.splits;
}

kops::sample::FusedSamplingParams fused_params(float temperature, uint64_t seed) {
  kops::sample::FusedSamplingParams fp{};
  fp.sampling.default_temperature = temperature;
  fp.sampling.default_top_k = 0;
  fp.sampling.default_top_p = 1.0f;
  fp.sampling.default_min_p = 0.0f;
  fp.sampling.rng.global_seed = seed;
  fp.default_repetition_penalty = 1.0f;
  return fp;
}

/// Greedy: random rows, ties, NaN / +-inf, all -inf, all NaN, maximum in the last token; packed
/// and padded rows; the all-greedy kernel and the per-row T <= 0 branch of the categorical kernel.
template <typename T>
void test_fused_greedy_edges(const char* name) {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int64_t vocab : {1, 17, 1001, 4096, 151936}) {
    for (int64_t pad : {0, 8}) {
      const int64_t rows = 6;
      const int64_t stride = vocab + pad;
      std::mt19937 gen(static_cast<uint32_t>(vocab * 31 + pad));
      std::normal_distribution<float> dist(0.0f, 3.0f);
      std::vector<float> f(static_cast<size_t>(rows * stride), 0.0f);
      auto at = [&](int64_t r, int64_t v) -> float& { return f[static_cast<size_t>(r * stride + v)]; };
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t v = 0; v < vocab; ++v) at(r, v) = dist(gen);
      }
      // Ties at the max: the lowest index wins, also when the tied tokens share a thread
      // (p, p + 1 sit in one 16-byte chunk; p - 32, p in one lane of the scalar path at V = 1001).
      const int64_t tp = vocab / 2;
      at(1, vocab - 1) = 40.0f;
      at(1, tp) = 40.0f;
      if (tp + 1 < vocab) at(1, tp + 1) = 40.0f;
      if (tp >= 32) at(1, tp - 32) = 40.0f;
      for (int64_t v = 0; v < vocab; v += 3) at(2, v) = nan;
      at(2, vocab / 3) = inf;
      for (int64_t v = 0; v < vocab; ++v) at(3, v) = -inf;
      for (int64_t v = 0; v < vocab; ++v) at(4, v) = nan;
      at(5, vocab - 1) = 50.0f;
      const auto h = cast_from_float<T>(f);
      std::vector<int32_t> want(static_cast<size_t>(rows));
      for (int64_t r = 0; r < rows; ++r) {
        want[static_cast<size_t>(r)] = fused_host_greedy(h.data() + r * stride, vocab, FusedHostProc{});
      }
      KOPS_TEST_CHECK(want[3] == -1 && want[4] == -1);
      const auto got = fused_run(h, rows, vocab, stride, fused_params(0.0f, 1u), ctx);
      if (got != want) std::fprintf(stderr, "%s greedy V=%lld pad=%lld\n", name,
                                    static_cast<long long>(vocab), static_cast<long long>(pad));
      KOPS_TEST_CHECK(got == want);
      // Same rows through the categorical kernel with per-row temperature 0 / -1 / NaN.
      float* d_t = kops_test::device_alloc<float>(static_cast<size_t>(rows));
      kops_test::copy_in(d_t, std::vector<float>{0.0f, -1.0f, nan, 0.0f, 0.0f, 0.0f});
      auto fp = fused_params(1.0f, 1u);
      fp.sampling.temperature = d_t;
      KOPS_TEST_CHECK(fused_run(h, rows, vocab, stride, fp, ctx) == want);
      kops_test::device_free(d_t);
    }
  }
}

/// Repetition (bitmask via slot_id), frequency/presence (token counts), mask write-back.
void test_fused_processors() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  int changed = 0;
  for (int64_t vocab : {1001, 4096}) {
    const int64_t rows = 3, slots = 4;
    const int64_t mbytes = (vocab + 7) / 8;
    std::mt19937 gen(static_cast<uint32_t>(vocab));
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::uniform_int_distribution<int> bit(0, 1), cnt(0, 3);
    std::vector<float> logits(static_cast<size_t>(rows * vocab));
    for (float& x : logits) x = dist(gen);
    std::vector<uint8_t> mask(static_cast<size_t>(slots * mbytes), 0);
    std::vector<int32_t> counts(static_cast<size_t>(slots * vocab));
    for (int64_t s = 0; s < slots; ++s) {
      for (int64_t v = 0; v < vocab; ++v) {
        if (bit(gen)) mask[static_cast<size_t>(s * mbytes + (v >> 3))] |= static_cast<uint8_t>(1u << (v & 7));
        counts[static_cast<size_t>(s * vocab + v)] = cnt(gen);
      }
    }
    // Row 0 (freq 0.5, pres 0.25): tb wins (12.65 vs 12.25); with frequency and presence
    // interchanged ta would win (12.75), so the two must not be confused.
    const int64_t ta = 3, tb = vocab - 2;
    logits[static_cast<size_t>(ta)] = 14.0f;
    logits[static_cast<size_t>(tb)] = 13.4f;
    for (int64_t s : {int64_t{0}, int64_t{2}}) {
      counts[static_cast<size_t>(s * vocab + ta)] = 3;
      counts[static_cast<size_t>(s * vocab + tb)] = 1;
      mask[static_cast<size_t>(s * mbytes + (ta >> 3))] &= static_cast<uint8_t>(~(1u << (ta & 7)));
      mask[static_cast<size_t>(s * mbytes + (tb >> 3))] &= static_cast<uint8_t>(~(1u << (tb & 7)));
    }
    const std::vector<int32_t> slot = {2, 0, 3};
    const std::vector<float> rep = {2.0f, 1.5f, 1.0f}, freq = {0.5f, 0.0f, 0.25f}, pres = {0.25f, 0.5f, 0.0f};
    uint8_t* d_mask = dev_copy(mask);
    int32_t* d_counts = dev_copy(counts);
    int32_t* d_slot = dev_copy(slot);
    float* d_rep = dev_copy(rep);
    float* d_freq = dev_copy(freq);
    float* d_pres = dev_copy(pres);
    for (int use_slot = 0; use_slot < 2; ++use_slot) {
      auto fp = fused_params(0.0f, 3u);
      fp.penalty_mask = d_mask;
      fp.slot_id = use_slot ? d_slot : nullptr;
      fp.repetition_penalty = d_rep;
      fp.token_counts = d_counts;
      fp.frequency_penalty = d_freq;
      fp.presence_penalty = d_pres;
      std::vector<int32_t> want(3);
      for (int64_t r = 0; r < rows; ++r) {
        const int64_t s = use_slot ? slot[static_cast<size_t>(r)] : r;
        FusedHostProc p;
        p.mask = mask.data() + s * mbytes;
        p.counts = counts.data() + s * vocab;
        p.rep = rep[static_cast<size_t>(r)];
        p.freq = freq[static_cast<size_t>(r)];
        p.pres = pres[static_cast<size_t>(r)];
        want[static_cast<size_t>(r)] = fused_host_greedy(logits.data() + r * vocab, vocab, p);
        changed += want[static_cast<size_t>(r)] !=
                   fused_host_greedy(logits.data() + r * vocab, vocab, FusedHostProc{});
      }
      KOPS_TEST_CHECK(want[0] == static_cast<int32_t>(tb));
      KOPS_TEST_CHECK(fused_run(logits, rows, vocab, vocab, fp, ctx) == want);
      if (use_slot) {
        // Write-back: the drawn token's bit is set in its slot row, nothing else changes.
        fp.update_penalty_mask = 1;
        const auto ids = fused_run(logits, rows, vocab, vocab, fp, ctx);
        KOPS_TEST_CHECK(ids == want);
        auto expect = mask;
        for (int64_t r = 0; r < rows; ++r) {
          const int64_t s = slot[static_cast<size_t>(r)];
          const int32_t id = ids[static_cast<size_t>(r)];
          expect[static_cast<size_t>(s * mbytes + (id >> 3))] |= static_cast<uint8_t>(1u << (id & 7));
        }
        KOPS_TEST_CHECK(kops_test::copy_out(d_mask, mask.size()) == expect);
        kops_test::copy_in(d_mask, mask);
      }
    }
    // Write-back without a mask is a caller error.
    auto bad = fused_params(0.0f, 3u);
    bad.update_penalty_mask = 1;
    kops::Status st = kops::Status::kSuccess;
    (void)fused_run(logits, rows, vocab, vocab, bad, ctx, &st);
    KOPS_TEST_CHECK(st == kops::Status::kInvalidArgument);
    kops_test::device_free(d_mask);
    kops_test::device_free(d_counts);
    kops_test::device_free(d_slot);
    kops_test::device_free(d_rep);
    kops_test::device_free(d_freq);
    kops_test::device_free(d_pres);
  }
  KOPS_TEST_CHECK(changed > 0);  // the processors moved at least one argmax
}

/// Categorical distribution on identical rows, grouped into 64 contiguous token groups so every
/// warp segment (and, at V = 40001 fp32, the grouped records) carries probability mass. Tokens
/// set to -inf / NaN must never be drawn.
template <typename T>
void test_fused_categorical(const char* name, int64_t vocab, int64_t rows, int launches) {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const float temperature = 0.8f;
  std::mt19937 gen(static_cast<uint32_t>(vocab + 7));
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> row(static_cast<size_t>(vocab));
  for (float& x : row) x = dist(gen);
  for (int64_t v = 5; v < vocab; v += 97) row[static_cast<size_t>(v)] = -std::numeric_limits<float>::infinity();
  for (int64_t v = 11; v < vocab; v += 101) row[static_cast<size_t>(v)] = std::numeric_limits<float>::quiet_NaN();
  const auto one = cast_from_float<T>(row);
  std::vector<T> all(static_cast<size_t>(rows * vocab));
  for (int64_t r = 0; r < rows; ++r) std::copy(one.begin(), one.end(), all.begin() + r * vocab);
  constexpr int kGroups = 64;
  std::vector<double> p(static_cast<size_t>(vocab), 0.0);
  double mx = -1e300, z = 0.0;
  for (int64_t v = 0; v < vocab; ++v) {
    const float x = kops_test::to_float(one[static_cast<size_t>(v)]);
    if (!std::isnan(x)) mx = std::max(mx, static_cast<double>(x) / temperature);
  }
  for (int64_t v = 0; v < vocab; ++v) {
    const float x = kops_test::to_float(one[static_cast<size_t>(v)]);
    if (!std::isnan(x)) p[static_cast<size_t>(v)] = std::exp(static_cast<double>(x) / temperature - mx);
    z += p[static_cast<size_t>(v)];
  }
  std::vector<double> expect(kGroups, 0.0), hist(kGroups, 0.0);
  for (int64_t v = 0; v < vocab; ++v) {
    expect[static_cast<size_t>(v * kGroups / vocab)] += p[static_cast<size_t>(v)] / z * rows * launches;
  }
  int64_t banned = 0;
  std::vector<int32_t> first;
  for (int it = 0; it < launches; ++it) {
    const auto ids = fused_run(all, rows, vocab, vocab, fused_params(temperature, 5000u + it), ctx);
    if (it == 0) first = ids;
    for (int32_t id : ids) {
      KOPS_TEST_CHECK(id >= 0 && id < vocab);
      if (id < 0 || id >= vocab) continue;
      if (p[static_cast<size_t>(id)] == 0.0) ++banned;
      hist[static_cast<size_t>(id * kGroups / vocab)] += 1.0;
    }
  }
  int df = 0;
  const double chi = chi_square(hist, expect, &df);
  if (chi >= 130.0 || banned != 0) {
    std::fprintf(stderr, "%s V=%lld chi=%.1f df=%d banned=%lld\n", name, static_cast<long long>(vocab),
                 chi, df, static_cast<long long>(banned));
  }
  KOPS_TEST_CHECK(df == kGroups - 1);
  KOPS_TEST_CHECK(chi < 130.0);  // df 63: mean 63, sd 11.2
  KOPS_TEST_CHECK(banned == 0);
  // Determinism, and a row's token does not depend on the batch size (same row index).
  KOPS_TEST_CHECK(fused_run(all, rows, vocab, vocab, fused_params(temperature, 5000u), ctx) == first);
  const int64_t half = rows / 2;
  std::vector<T> head(all.begin(), all.begin() + half * vocab);
  const auto ids_half = fused_run(head, half, vocab, vocab, fused_params(temperature, 5000u), ctx);
  KOPS_TEST_CHECK(std::equal(ids_half.begin(), ids_half.end(), first.begin()));
}

// ---- sample_fused M2: top-k (vLLM tie group) / min-p --------------------------------------------

/// fp64 probabilities of the filtered draw for one processed row (NaN already counts as -inf):
/// +inf entries win uniformly; top-k keeps every finite x >= the k-th largest x (ties evaluated on
/// the logits before temperature); min-p keeps y >= max(y) + log(min_p) with y = x * (1 / T).
std::vector<double> fused_filtered_oracle(const std::vector<float>& x, float temperature, int k,
                                          float min_p, float top_p = 1.0f, bool joint = false,
                                          std::vector<char>* ambiguous = nullptr) {
  const size_t V = x.size();
  std::vector<double> p(V, 0.0);
  if (ambiguous != nullptr) ambiguous->assign(V, 0);
  const float inf = std::numeric_limits<float>::infinity();
  size_t infs = 0;
  for (float v : x) infs += v == inf;
  if (infs > 0) {
    for (size_t i = 0; i < V; ++i) p[i] = x[i] == inf ? 1.0 / static_cast<double>(infs) : 0.0;
    return p;
  }
  std::vector<float> fin;
  for (float v : x) {
    if (!std::isnan(v) && v != -inf) fin.push_back(v);
  }
  if (fin.empty()) return p;
  const float inv_t = 1.0f / temperature;
  float xk = -inf;
  if (k > 0 && static_cast<size_t>(k) < fin.size()) {
    std::vector<float> s = fin;
    std::nth_element(s.begin(), s.begin() + (k - 1), s.end(), std::greater<float>());
    xk = s[static_cast<size_t>(k - 1)];
  }
  float m = -inf;
  for (float v : fin) m = std::max(m, v * inv_t);
  const float thr = min_p > 0.0f ? m + std::log(min_p) : -inf;
  auto finite_at = [&](size_t i) { return !std::isnan(x[i]) && x[i] != -inf; };
  auto in_base = [&](size_t i) { return finite_at(i) && x[i] >= xk && x[i] * inv_t >= thr; };
  auto mass = [&](size_t i) { return std::exp(static_cast<double>(x[i] * inv_t) - static_cast<double>(m)); };
  std::vector<char> nucleus(V, 1);
  if (top_p < 1.0f) {
    // Domain in (value desc, token asc) order; a token is kept while the mass before it is below
    // p * Z (the first one always). Tokens within 1e-6 Z of the target may go either way.
    std::vector<size_t> dom;
    for (size_t i = 0; i < V; ++i) {
      if (joint ? finite_at(i) : in_base(i)) dom.push_back(i);
    }
    std::sort(dom.begin(), dom.end(), [&](size_t a, size_t b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    double zd = 0.0;
    for (size_t i : dom) zd += mass(i);
    const double target = (top_p > 0.0f ? top_p : 0.0f) * zd;
    std::fill(nucleus.begin(), nucleus.end(), 0);
    double cum = 0.0;
    for (size_t j = 0; j < dom.size(); ++j) {
      nucleus[dom[j]] = j == 0 || cum < target;
      if (ambiguous != nullptr && j > 0 && std::fabs(cum - target) <= 1e-6 * zd) (*ambiguous)[dom[j]] = 1;
      cum += mass(dom[j]);
    }
  }
  double z = 0.0;
  for (size_t i = 0; i < V; ++i) {
    if (!in_base(i) || !nucleus[i]) continue;
    p[i] = mass(i);
    z += p[i];
  }
  for (double& q : p) q = z > 0.0 ? q / z : 0.0;
  return p;
}

/// Draw `launches` x `rows` tokens from identical rows and compare with the oracle: no draw may
/// leave the support, and a chi-square over the support (or over 64 token groups) must pass.
template <typename T>
void check_fused_filtered(const char* name, const std::vector<float>& row_f, float temperature,
                          int k, float min_p, int64_t rows, int launches, bool grouped,
                          float top_p = 1.0f, bool joint = false) {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const int64_t vocab = static_cast<int64_t>(row_f.size());
  const auto one = cast_from_float<T>(row_f);
  std::vector<float> seen(one.size());
  for (size_t i = 0; i < one.size(); ++i) seen[i] = kops_test::to_float(one[i]);
  std::vector<char> amb;
  const auto p = fused_filtered_oracle(seen, temperature, k, min_p, top_p, joint, &amb);
  std::vector<T> all(static_cast<size_t>(rows * vocab));
  for (int64_t r = 0; r < rows; ++r) std::copy(one.begin(), one.end(), all.begin() + r * vocab);
  std::vector<double> hist(static_cast<size_t>(vocab), 0.0);
  int64_t outside = 0;
  std::vector<int32_t> first;
  for (int it = 0; it < launches; ++it) {
    auto fp = fused_params(temperature, 70000u + it);
    fp.sampling.default_top_k = k;
    fp.sampling.default_min_p = min_p;
    fp.sampling.default_top_p = top_p;
    fp.sampling.joint_topk_topp = joint;
    const auto ids = fused_run(all, rows, vocab, vocab, fp, ctx);
    if (it == 0) first = ids;
    for (int32_t id : ids) {
      if (id >= 0 && id < vocab && amb[static_cast<size_t>(id)]) continue;
      if (id < 0 || id >= vocab || p[static_cast<size_t>(id)] == 0.0) {
        ++outside;
        continue;
      }
      hist[static_cast<size_t>(id)] += 1.0;
    }
  }
  const double n = static_cast<double>(rows) * launches;
  std::vector<double> obs, expect;
  if (grouped) {
    obs.assign(64, 0.0);
    expect.assign(64, 0.0);
    for (int64_t v = 0; v < vocab; ++v) {
      obs[static_cast<size_t>(v * 64 / vocab)] += hist[static_cast<size_t>(v)];
      expect[static_cast<size_t>(v * 64 / vocab)] += p[static_cast<size_t>(v)] * n;
    }
  } else {
    for (int64_t v = 0; v < vocab; ++v) {
      if (p[static_cast<size_t>(v)] == 0.0 || amb[static_cast<size_t>(v)]) continue;
      obs.push_back(hist[static_cast<size_t>(v)]);
      expect.push_back(p[static_cast<size_t>(v)] * n);
    }
  }
  int df = 0;
  const double chi = chi_square(obs, expect, &df);
  const double limit = df + 6.0 * std::sqrt(2.0 * std::max(df, 1)) + 10.0;
  if (outside != 0 || chi >= limit) {
    std::fprintf(stderr, "%s: outside=%lld chi=%.1f df=%d limit=%.1f\n", name,
                 static_cast<long long>(outside), chi, df, limit);
  }
  KOPS_TEST_CHECK(outside == 0);
  KOPS_TEST_CHECK(chi < limit);
  // Determinism and batch-size invariance of the filtered path.
  auto fp = fused_params(temperature, 70000u);
  fp.sampling.default_top_k = k;
  fp.sampling.default_min_p = min_p;
  fp.sampling.default_top_p = top_p;
  fp.sampling.joint_topk_topp = joint;
  KOPS_TEST_CHECK(fused_run(all, rows, vocab, vocab, fp, ctx) == first);
  const int64_t half = rows / 2;
  std::vector<T> head(all.begin(), all.begin() + half * vocab);
  const auto ids_half = fused_run(head, half, vocab, vocab, fp, ctx);
  KOPS_TEST_CHECK(std::equal(ids_half.begin(), ids_half.end(), first.begin()));
}

std::vector<float> uniform_row(int64_t vocab, float lo, float hi, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> r(static_cast<size_t>(vocab));
  for (float& x : r) x = dist(gen);
  return r;
}

void test_fused_topk_minp() {
  // vLLM ties: 10 distinct values above a 20-way tie at the 15th value keep all 30 tokens
  // (an index tie-break would keep 15).
  {
    std::vector<float> r = uniform_row(4096, -3.0f, 3.0f, 1u);
    for (int j = 0; j < 10; ++j) r[static_cast<size_t>(100 + 37 * j)] = 5.5f + 0.1f * j;
    for (int j = 0; j < 20; ++j) r[static_cast<size_t>(7 + 191 * j)] = 5.0f;
    check_fused_filtered<__nv_bfloat16>("topk ties bf16", r, 1.0f, 15, 0.0f, 1024, 32, false);
    check_fused_filtered<float>("topk ties fp32", r, 1.0f, 15, 0.0f, 1024, 32, false);
    std::vector<float> r2(r.begin(), r.begin() + 4095);  // scalar (non-vector) path
    check_fused_filtered<float>("topk ties fp32 V=4095", r2, 1.0f, 15, 0.0f, 1024, 32, false);
  }
  // top_k == 1 with T > 0 samples among the tied maxima; without a tie it is the argmax.
  {
    std::vector<float> r = uniform_row(4096, -3.0f, 3.0f, 2u);
    r[11] = r[2000] = r[4095] = 9.0f;
    check_fused_filtered<float>("topk1 tied maxima", r, 0.7f, 1, 0.0f, 1024, 16, false);
    r[2000] = r[4095] = 8.0f;
    check_fused_filtered<float>("topk1 argmax", r, 0.7f, 1, 0.0f, 256, 4, false);
  }
  // A boundary bin that fills the candidate buffer exactly (4096 equal values), one that
  // overflows it (radix fallback), and logits far outside [-64, 64) (clamped end bins).
  check_fused_filtered<float>("topk all-equal V=2048", std::vector<float>(2048, 1.25f), 1.0f, 5, 0.0f,
                              512, 64, true);
  check_fused_filtered<float>("topk all-equal V=8192 (fallback)", std::vector<float>(8192, 1.25f), 1.0f,
                              5, 0.0f, 512, 64, true);
  {
    // 8192 distinct values inside one bin [1, 1 + 1/32): the radix fallback must find the exact
    // 100th value.
    std::vector<float> r(8192);
    for (size_t i = 0; i < r.size(); ++i) r[i] = 1.0f + 1e-6f * static_cast<float>((i * 2654435761u) % 8192);
    check_fused_filtered<float>("topk distinct fallback", r, 1.0f, 100, 0.0f, 512, 64, false);
  }
  {
    std::mt19937 gen(3u);
    std::normal_distribution<float> dist(0.0f, 100.0f);
    std::vector<float> r(4096);
    for (float& x : r) x = dist(gen);
    check_fused_filtered<float>("topk clamped bins", r, 60.0f, 50, 0.0f, 1024, 32, false);
    check_fused_filtered<float>("topk k=3000 wide", r, 60.0f, 3000, 0.0f, 512, 64, true);
  }
  // min-p alone and with top-k; min_p = 1 keeps only the maxima.
  {
    const std::vector<float> r = uniform_row(4096, -3.0f, 3.0f, 4u);
    check_fused_filtered<__nv_bfloat16>("minp 0.3", r, 0.7f, 0, 0.3f, 1024, 32, false);
    check_fused_filtered<__half>("minp 0.1 + topk 20", r, 1.0f, 20, 0.1f, 1024, 32, false);
    std::vector<float> r2 = r;
    r2[17] = r2[3000] = 6.0f;
    check_fused_filtered<float>("minp 1", r2, 1.0f, 0, 1.0f, 256, 8, false);
  }
  // A row with +inf entries samples uniformly among them (categorical and filtered kernels).
  {
    std::vector<float> r = uniform_row(4096, -3.0f, 3.0f, 6u);
    r[5] = r[900] = r[4095] = std::numeric_limits<float>::infinity();
    check_fused_filtered<float>("+inf row", r, 1.0f, 0, 0.0f, 2048, 24, false);
    check_fused_filtered<float>("+inf row topk", r, 1.0f, 50, 0.0f, 1024, 16, false);
  }
  // NaN / -inf entries never win; a large vocabulary through the vector path.
  {
    std::vector<float> r = uniform_row(151936, -6.0f, 6.0f, 5u);
    for (size_t v = 3; v < r.size(); v += 5) r[v] = std::numeric_limits<float>::quiet_NaN();
    for (size_t v = 4; v < r.size(); v += 7) r[v] = -std::numeric_limits<float>::infinity();
    check_fused_filtered<__nv_bfloat16>("topk 50 V=151936", r, 1.0f, 50, 0.0f, 256, 32, false);
  }

  // Per-row parameters: each row checked against its own oracle support.
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const int64_t rows = 6, vocab = 4096;
  std::vector<float> all;
  for (int64_t r = 0; r < rows; ++r) {
    const auto row = uniform_row(vocab, -3.0f, 3.0f, 10u + static_cast<uint32_t>(r));
    all.insert(all.end(), row.begin(), row.end());
  }
  const std::vector<int32_t> ks = {0, 1, 50, 5000, 3, 0};
  const std::vector<float> mps = {0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.5f};
  const std::vector<float> ts = {1.0f, 1.0f, 0.5f, 2.0f, 1.0f, 0.0f};
  int32_t* d_k = dev_copy(ks);
  float* d_mp = dev_copy(mps);
  float* d_t = dev_copy(ts);
  auto fp = fused_params(1.0f, 11u);
  fp.sampling.top_k = d_k;
  fp.sampling.min_p = d_mp;
  fp.sampling.temperature = d_t;
  int64_t bad = 0;
  for (int it = 0; it < 64; ++it) {
    fp.sampling.rng.global_seed = 11u + it;
    const auto ids = fused_run(all, rows, vocab, vocab, fp, ctx);
    for (int64_t r = 0; r < rows; ++r) {
      const std::vector<float> row(all.begin() + r * vocab, all.begin() + (r + 1) * vocab);
      const int32_t id = ids[static_cast<size_t>(r)];
      if (ts[static_cast<size_t>(r)] <= 0.0f) {
        bad += id != fused_host_greedy(row.data(), vocab, FusedHostProc{});
        continue;
      }
      const auto p = fused_filtered_oracle(row, ts[static_cast<size_t>(r)], ks[static_cast<size_t>(r)],
                                           mps[static_cast<size_t>(r)]);
      bad += id < 0 || id >= vocab || p[static_cast<size_t>(id)] == 0.0;
    }
  }
  KOPS_TEST_CHECK(bad == 0);
  kops_test::device_free(d_k);
  kops_test::device_free(d_mp);
  kops_test::device_free(d_t);

  // Plan contract: a plan resolved without filters is stale for a filtered call (top-p too);
  // NaN min_p is rejected.
  {
    std::vector<float> h = uniform_row(64, -1.0f, 1.0f, 9u);
    float* d = kops_test::device_alloc<float>(h.size());
    int32_t* d_ids = kops_test::device_alloc<int32_t>(1);
    kops_test::copy_in(d, h);
    const Tensor t = kops_test::tensor2d(d, DType::kFloat32, 1, 64);
    const Tensor t_ids = tensor1d(d_ids, DType::kInt32, 1);
    auto plain = fused_params(1.0f, 1u);
    kops::sample::SamplingPlan plan{};
    KOPS_EXPECT_OK(kops::sample::sampling_plan(t, plain.sampling, ctx, &plan), "plain plan");
    auto topk = plain;
    topk.sampling.default_top_k = 8;
    kops::Workspace ws{};
    KOPS_EXPECT_STATUS(kops::sample::sample_fused(t_ids, t, topk, plan, ws, ctx),
                       kops::Status::kPlanInvalidated, "stale plan for top-k");
    auto topp = plain;
    topp.sampling.default_top_p = 0.9f;
    KOPS_EXPECT_STATUS(kops::sample::sample_fused(t_ids, t, topp, plan, ws, ctx),
                       kops::Status::kPlanInvalidated, "stale plan for top-p");
    auto nanmp = plain;
    nanmp.sampling.default_min_p = std::numeric_limits<float>::quiet_NaN();
    KOPS_EXPECT_STATUS(kops::sample::sampling_plan(t, nanmp.sampling, ctx, &plan),
                       kops::Status::kInvalidArgument, "NaN min_p");
    kops_test::device_free(d);
    kops_test::device_free(d_ids);
  }
}

/// Top-p (sequential and joint), with top-k / min-p, ties at the boundary, the exact fallback.
void test_fused_topp() {
  const std::vector<float> r = uniform_row(4096, -3.0f, 3.0f, 21u);
  check_fused_filtered<float>("topp 0.9", r, 1.0f, 0, 0.0f, 1024, 32, true, 0.9f);
  check_fused_filtered<__nv_bfloat16>("topp 0.95 bf16", r, 1.0f, 0, 0.0f, 1024, 32, true, 0.95f);
  check_fused_filtered<__half>("topp 0.5 T=0.5 fp16", r, 0.5f, 0, 0.0f, 1024, 32, false, 0.5f);
  check_fused_filtered<float>("topp 0.99", r, 1.0f, 0, 0.0f, 1024, 32, true, 0.99f);
  check_fused_filtered<float>("topp 0.5", r, 1.0f, 0, 0.0f, 1024, 32, false, 0.5f);
  check_fused_filtered<float>("topp flat p0.99", std::vector<float>(8192, 2.0f), 1.0f, 0, 0.0f, 512,
                              64, true, 0.99f);
  check_fused_filtered<float>("topp 0.7 V=4095", std::vector<float>(r.begin(), r.begin() + 4095), 0.6f, 0, 0.0f,
                              1024, 32, false, 0.7f);
  // top_p = 0 keeps the largest token only (the lowest id among tied maxima).
  {
    std::vector<float> t = r;
    t[700] = t[30] = t[3000] = 8.0f;
    check_fused_filtered<float>("topp 0 ties", t, 1.0f, 0, 0.0f, 256, 4, false, 0.0f);
  }
  // Sequential (top-p of the renormalized top-k set) and joint (both on the full row) differ here:
  // the 8 largest values carry ~12% of the mass.
  {
    std::vector<float> t = r;
    for (int j = 0; j < 8; ++j) t[static_cast<size_t>(50 + 400 * j)] = 5.0f + 0.1f * j;
    std::vector<char> amb;
    const auto ps = fused_filtered_oracle(t, 1.0f, 8, 0.0f, 0.5f, false, &amb);
    const auto pj = fused_filtered_oracle(t, 1.0f, 8, 0.0f, 0.5f, true, &amb);
    int ns = 0, nj = 0;
    for (size_t i = 0; i < t.size(); ++i) {
      ns += ps[i] > 0.0;
      nj += pj[i] > 0.0;
    }
    KOPS_TEST_CHECK(ns < nj && nj == 8);
    check_fused_filtered<float>("topk8 topp0.5 sequential", t, 1.0f, 8, 0.0f, 1024, 32, false, 0.5f, false);
    check_fused_filtered<float>("topk8 topp0.5 joint", t, 1.0f, 8, 0.0f, 1024, 32, false, 0.5f, true);
    check_fused_filtered<__nv_bfloat16>("topk50 topp0.9 minp0.02", t, 1.0f, 50, 0.02f, 1024, 32, false, 0.9f);
    check_fused_filtered<float>("minp0.1 topp0.8 joint", t, 0.8f, 0, 0.1f, 1024, 32, false, 0.8f, true);
  }
  // Boundary bins over the buffer: distinct values in one bin, equal values (the boundary token is
  // the r-th of a tie in token order), and the top-k + top-p fallback.
  {
    std::vector<float> d(8192);
    for (size_t i = 0; i < d.size(); ++i) d[i] = 1.0f + 1e-6f * static_cast<float>((i * 2654435761u) % 8192);
    check_fused_filtered<float>("topp fallback distinct", d, 1.0f, 0, 0.0f, 512, 64, true, 0.3f);
    check_fused_filtered<float>("topp fallback equal", std::vector<float>(8192, 2.0f), 1.0f, 0, 0.0f, 512,
                                64, true, 0.3f);
    check_fused_filtered<float>("topk5000 topp0.6 fallback", d, 1.0f, 5000, 0.0f, 512, 64, true, 0.6f);
    // ~1024 tokens per bin: the bound range overflows the buffer but its boundary bin does not
    // (three-pass path), sequential and joint with min-p.
    const std::vector<float> flat = uniform_row(65536, -1.0f, 1.0f, 23u);
    check_fused_filtered<float>("topp three-pass", flat, 1.0f, 0, 0.0f, 512, 64, true, 0.9f);
    check_fused_filtered<__nv_bfloat16>("topp three-pass joint minp", flat, 1.0f, 0, 0.5f, 512, 64, true,
                                        0.7f, true);
    // Four levels 0.1 apart, 3000 tokens each: the mass radix must filter by the key prefix of
    // every pass (the levels differ in the high key digits) and end inside a tie.
    std::vector<float> lv(12000);
    for (size_t i = 0; i < lv.size(); ++i) lv[i] = 1.0f + 0.1f * static_cast<float>(i % 4);
    check_fused_filtered<float>("topp fallback levels", lv, 1.0f, 0, 0.0f, 512, 64, true, 0.4f);
  }
  // A Qwen-sized row through the vector path.
  {
    std::mt19937 gen(22u);
    std::normal_distribution<float> dist(0.0f, 2.5f);
    std::vector<float> big(151936);
    for (float& x : big) x = dist(gen);
    check_fused_filtered<__nv_bfloat16>("topp 0.95 V=151936", big, 1.0f, 0, 0.0f, 256, 32, true, 0.95f);
  }
  // Per-row top_p: off, 0.5, 0 (largest only), NaN (largest only, as in sample()).
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const int64_t rows = 4, vocab = 4096;
  std::vector<float> all;
  for (int64_t i = 0; i < rows; ++i) {
    const auto row = uniform_row(vocab, -3.0f, 3.0f, 30u + static_cast<uint32_t>(i));
    all.insert(all.end(), row.begin(), row.end());
  }
  const std::vector<float> tps = {1.0f, 0.5f, 0.0f, std::numeric_limits<float>::quiet_NaN()};
  float* d_tp = dev_copy(tps);
  auto fp = fused_params(1.0f, 5u);
  fp.sampling.top_p = d_tp;
  int64_t bad = 0;
  for (int it = 0; it < 32; ++it) {
    fp.sampling.rng.global_seed = 5u + it;
    const auto ids = fused_run(all, rows, vocab, vocab, fp, ctx);
    for (int64_t i = 0; i < rows; ++i) {
      const std::vector<float> row(all.begin() + i * vocab, all.begin() + (i + 1) * vocab);
      const float tp = std::isnan(tps[static_cast<size_t>(i)]) ? 0.0f : tps[static_cast<size_t>(i)];
      const auto q = fused_filtered_oracle(row, 1.0f, 0, 0.0f, tp);
      const int32_t id = ids[static_cast<size_t>(i)];
      bad += id < 0 || id >= vocab || q[static_cast<size_t>(id)] == 0.0;
    }
  }
  KOPS_TEST_CHECK(bad == 0);
  kops_test::device_free(d_tp);
}

/// Path B: every split count draws Path A's tokens bit for bit on rows that reach each branch
/// (greedy, categorical with +inf / all -inf rows, top-k ties, min-p, top-p, joint, three-pass,
/// radix fallbacks), and leaves the same penalty mask; a captured graph replays with new seeds;
/// two streams with their own workspaces run at once.
void test_fused_split() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const int64_t rows = 4;
  struct Mode {
    const char* name;
    float t, mp, tp;
    int k, kind;  // kind 0: N(0,3) rows, 1: flat, 2: distinct values in a few bins, 3: four levels
    bool joint;
  };
  const Mode modes[] = {{"greedy", 0.0f, 0.0f, 1.0f, 0, 0, false},  {"t1", 1.0f, 0.0f, 1.0f, 0, 0, false},
                        {"k50", 0.8f, 0.0f, 1.0f, 50, 0, false},    {"k1", 1.0f, 0.0f, 1.0f, 1, 0, false},
                        {"minp", 1.0f, 0.1f, 1.0f, 0, 0, false},    {"p0.9", 1.0f, 0.0f, 0.9f, 0, 0, false},
                        {"k20p0.8j", 1.0f, 0.0f, 0.8f, 20, 0, true}, {"three", 1.0f, 0.0f, 0.9f, 0, 1, false},
                        {"fb p", 1.0f, 0.0f, 0.3f, 0, 2, false},    {"fb k", 1.0f, 0.0f, 1.0f, 5000, 2, false},
                        {"levels", 1.0f, 0.0f, 0.4f, 0, 3, false}};
  for (const Mode& md : modes) {
    const int64_t vocab = md.kind == 0 ? 32768 : 65536;
    std::vector<float> f(static_cast<size_t>(rows * vocab));
    std::mt19937 gen(static_cast<uint32_t>(vocab + md.kind));
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    for (size_t i = 0; i < f.size(); ++i) {
      const size_t v = i % static_cast<size_t>(vocab);
      f[i] = md.kind == 0 ? nd(gen)
             : md.kind == 1 ? ud(gen)
             : md.kind == 2 ? 1.0f + 1e-6f * static_cast<float>((v * 2654435761u) % 65536)
                            : 1.0f + 0.1f * static_cast<float>(v % 4);
    }
    if (md.kind == 0) {
      f[static_cast<size_t>(vocab + 77)] = f[static_cast<size_t>(vocab + 30000)] = std::numeric_limits<float>::infinity();
      for (int64_t v = 0; v < vocab; ++v) f[static_cast<size_t>(2 * vocab + v)] = -std::numeric_limits<float>::infinity();
      for (int64_t v = 5; v < vocab; v += 11) f[static_cast<size_t>(3 * vocab + v)] = std::numeric_limits<float>::quiet_NaN();
    }
    const auto h = cast_from_float<__nv_bfloat16>(f);
    for (uint64_t seed = 1; seed <= 2; ++seed) {
      auto fp = fused_params(md.t, 900u + seed);
      fp.sampling.default_top_k = md.k;
      fp.sampling.default_min_p = md.mp;
      fp.sampling.default_top_p = md.tp;
      fp.sampling.joint_topk_topp = md.joint;
      const auto ref = fused_run(h, rows, vocab, vocab, fp, ctx, nullptr, 1);
      for (int s : {2, 4, 8, 16}) {
        kops::Status st = kops::Status::kSuccess;
        const auto got = fused_run(h, rows, vocab, vocab, fp, ctx, &st, s);
        if (st == kops::Status::kPlanInvalidated) continue;  // more CTAs than the device holds
        KOPS_EXPECT_OK(st, "forced split");
        if (got != ref) std::fprintf(stderr, "split %s S=%d differs\n", md.name, s);
        KOPS_TEST_CHECK(got == ref);
      }
      if (md.kind == 0 && md.k == 50) {  // rows that are not 16-byte vectors stay on Path A
        kops::Status st = kops::Status::kSuccess;
        (void)fused_run(std::vector<float>(f.begin(), f.begin() + rows * (vocab - 1)), rows, vocab - 1, vocab - 1,
                        fp, ctx, &st, 8);
        KOPS_TEST_CHECK(st == kops::Status::kPlanInvalidated);
      }
    }
  }

  // Processors and mask write-back: same ids and the same final mask.
  const int64_t vocab = 32768, mbytes = vocab / 8;
  std::vector<float> lg(static_cast<size_t>(rows * vocab));
  std::mt19937 gen(5u);
  std::normal_distribution<float> nd(0.0f, 2.0f);
  for (float& x : lg) x = nd(gen);
  std::vector<uint8_t> mask(static_cast<size_t>(rows * mbytes));
  for (auto& b : mask) b = static_cast<uint8_t>(gen());
  uint8_t* d_mask = dev_copy(mask);
  int32_t* d_slot = dev_copy(std::vector<int32_t>{3, 1, 0, 2});
  auto fp = fused_params(1.0f, 77u);
  fp.sampling.default_top_k = 40;
  fp.penalty_mask = d_mask;
  fp.slot_id = d_slot;
  fp.default_repetition_penalty = 1.7f;
  fp.update_penalty_mask = 1;
  const auto ids_a = fused_run(lg, rows, vocab, vocab, fp, ctx, nullptr, 1);
  const auto mask_a = kops_test::copy_out(d_mask, mask.size());
  kops_test::copy_in(d_mask, mask);
  KOPS_TEST_CHECK(fused_run(lg, rows, vocab, vocab, fp, ctx, nullptr, 8) == ids_a);
  KOPS_TEST_CHECK(kops_test::copy_out(d_mask, mask.size()) == mask_a);
  kops_test::device_free(d_mask);
  kops_test::device_free(d_slot);

  // Graph capture (Path B, S = 8) on two streams with two workspaces, both in flight; seeds change
  // between replays through the device seed arrays and must match eager Path A draws.
  struct Job {
    std::vector<__nv_bfloat16> h;
    __nv_bfloat16* logits;
    int32_t* ids;
    uint64_t* seeds;
    uint8_t *scratch, *persistent;
    cudaStream_t st;
    cudaGraphExec_t exec;
  } jobs[2];
  kops::sample::FusedSamplingParams gp = fused_params(1.0f, 0u);
  gp.sampling.default_top_p = 0.9f;
  auto seeds_of = [&](int replay, int j) {
    return std::vector<uint64_t>{11u + replay, 12u + replay, 13u + replay, 14u + replay + 9u * j};
  };
  for (int j = 0; j < 2; ++j) {
    Job& jb = jobs[j];
    jb.h = cast_from_float<__nv_bfloat16>(lg);
    if (j == 1) std::reverse(jb.h.begin(), jb.h.end());
    jb.logits = kops_test::device_alloc<__nv_bfloat16>(jb.h.size());
    jb.ids = kops_test::device_alloc<int32_t>(rows);
    jb.seeds = kops_test::device_alloc<uint64_t>(rows);
    kops_test::copy_in(jb.logits, jb.h);
    cudaStreamCreateWithFlags(&jb.st, cudaStreamNonBlocking);
    kops::LaunchContext cj = ctx;
    cj.stream = reinterpret_cast<decltype(cj.stream)>(jb.st);
    const Tensor t = kops_test::tensor2d(jb.logits, DType::kBFloat16, rows, vocab);
    const Tensor ti = tensor1d(jb.ids, DType::kInt32, rows);
    kops::sample::SamplingPlan plan{};
    KOPS_EXPECT_OK(kops::sample::sampling_plan(t, gp.sampling, cj, &plan), "graph plan");
    plan.splits = 8;
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kops::sample::sampling_fused_workspace_requirements(rows, vocab, plan, &req), "graph ws");
    jb.scratch = kops_test::device_alloc<uint8_t>(req.device_scratch_bytes);
    jb.persistent = kops_test::device_alloc<uint8_t>(req.device_persistent_bytes);
    cudaMemset(jb.persistent, 0, req.device_persistent_bytes);
    kops::Workspace ws{jb.scratch, req.device_scratch_bytes, jb.persistent, req.device_persistent_bytes, nullptr, 0};
    auto jp = gp;
    jp.sampling.rng.per_request_seed = jb.seeds;
    kops::Workspace small = ws;  // a too-small workspace is refused before anything is launched
    small.device_scratch_bytes = 0;
    KOPS_EXPECT_STATUS(kops::sample::sample_fused(ti, t, jp, plan, small, cj), kops::Status::kWorkspaceTooSmall,
                       "split workspace too small");
    cudaGraph_t g = nullptr;
    KOPS_TEST_CHECK(cudaStreamBeginCapture(jb.st, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
    KOPS_EXPECT_OK(kops::sample::sample_fused(ti, t, jp, plan, ws, cj), "captured sample_fused");
    KOPS_TEST_CHECK(cudaStreamEndCapture(jb.st, &g) == cudaSuccess);
    KOPS_TEST_CHECK(cudaGraphInstantiate(&jb.exec, g, 0) == cudaSuccess);
    cudaGraphDestroy(g);
  }
  for (int replay = 0; replay < 3; ++replay) {
    for (int j = 0; j < 2; ++j) {
      kops_test::copy_in(jobs[j].seeds, seeds_of(replay, j));
      cudaGraphLaunch(jobs[j].exec, jobs[j].st);
    }
    for (int j = 0; j < 2; ++j) {
      cudaStreamSynchronize(jobs[j].st);
      const auto got = kops_test::copy_out(jobs[j].ids, rows);
      auto ep = gp;
      ep.sampling.rng.per_request_seed = jobs[j].seeds;  // same seeds, eager Path A
      KOPS_TEST_CHECK(fused_run(jobs[j].h, rows, vocab, vocab, ep, ctx, nullptr, 1) == got);
    }
  }
  for (Job& jb : jobs) {
    cudaGraphExecDestroy(jb.exec);
    cudaStreamDestroy(jb.st);
    for (void* q : {static_cast<void*>(jb.logits), static_cast<void*>(jb.ids), static_cast<void*>(jb.seeds),
                    static_cast<void*>(jb.scratch), static_cast<void*>(jb.persistent)}) {
      cudaFree(q);
    }
  }

  // Plan contract: a stale capture generation and a split count outside the barrier protocol are
  // refused before anything is launched.
  {
    auto* d = kops_test::device_alloc<__nv_bfloat16>(lg.size());
    auto* ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(rows));
    kops_test::copy_in(d, cast_from_float<__nv_bfloat16>(lg));
    const Tensor t = kops_test::tensor2d(d, DType::kBFloat16, rows, vocab);
    const Tensor ti = tensor1d(ids, DType::kInt32, rows);
    auto cp = fused_params(1.0f, 31u);
    cp.sampling.default_top_k = 40;
    kops::sample::SamplingPlan plan{};
    KOPS_EXPECT_OK(kops::sample::sampling_plan(t, cp.sampling, ctx, &plan), "contract plan");
    kops::LaunchContext stale = ctx;
    stale.capture_generation = ctx.capture_generation + 1;
    KOPS_EXPECT_STATUS(kops::sample::sample_fused(ti, t, cp, plan, kops::Workspace{}, stale),
                       kops::Status::kPlanInvalidated, "stale capture generation");
    plan.splits = 32;
    KOPS_EXPECT_STATUS(kops::sample::sample_fused(ti, t, cp, plan, kops::Workspace{}, ctx),
                       kops::Status::kPlanInvalidated, "split count above the barrier protocol");
    kops_test::device_free(d);
    kops_test::device_free(ids);
  }
}

/// Target batches: the auto plan keeps the split grid resident, every split count the device can
/// hold draws Path A's tokens bit for bit, and the plan is a valid power-of-two split count.
/// The RTX 3050 (16 SMs, 64 co-resident filtered CTAs) skips the grids it cannot hold; an H100
/// (132 SMs) exercises B=32/64 with splits > 1.
void test_fused_split_batches() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);

  struct Mode {
    const char* name;
    float t, tp;
    int k;
  };
  const Mode modes[] = {{"greedy", 0.0f, 1.0f, 0}, {"t1", 1.0f, 1.0f, 0},
                        {"k50", 0.8f, 1.0f, 50}, {"p0.9", 1.0f, 0.9f, 0},
                        {"p0.95", 1.0f, 0.95f, 0}};
  for (int64_t rows : {int64_t{32}, int64_t{64}, int64_t{128}}) {
    const int64_t vocab = 32768;
    std::vector<float> f(static_cast<size_t>(rows * vocab));
    std::mt19937 gen(static_cast<uint32_t>(rows));
    std::normal_distribution<float> nd(0.0f, 2.0f);
    for (float& x : f) x = nd(gen);
    const auto h = cast_from_float<__nv_bfloat16>(f);
    for (const Mode& md : modes) {
      if (rows == 128 && md.k != 50 && md.tp >= 1.0f) continue;  // keep the 128-row sweep short
      auto fp = fused_params(md.t, 4242u + static_cast<uint64_t>(rows));
      fp.sampling.default_top_k = md.k;
      fp.sampling.default_top_p = md.tp;
      const int32_t plan_splits = fused_auto_splits(h, rows, vocab, fp, ctx);
      KOPS_TEST_CHECK(plan_splits >= 1 && plan_splits <= 16 &&
                      (plan_splits & (plan_splits - 1)) == 0);
      if (md.k == 0 && md.tp >= 1.0f) {
        // A greedy or unfiltered plan only depends on the row shape; assert the split policy at
        // the batches where the device has enough co-resident CTAs to make a split pay.
        if (sms >= 32 && rows == 32) KOPS_TEST_CHECK(plan_splits >= 2);
        if (sms >= 64 && rows == 64) KOPS_TEST_CHECK(plan_splits >= 2);
      }
      const auto ref = fused_run(h, rows, vocab, vocab, fp, ctx, nullptr, 1);
      KOPS_TEST_CHECK(fused_run(h, rows, vocab, vocab, fp, ctx) == ref);
      for (int s : {2, 4, 8, 16}) {
        if (rows * s > static_cast<int64_t>(sms) * 4) continue;  // more CTAs than the device holds
        kops::Status st = kops::Status::kSuccess;
        const auto got = fused_run(h, rows, vocab, vocab, fp, ctx, &st, s);
        if (st == kops::Status::kPlanInvalidated) continue;
        KOPS_EXPECT_OK(st, "forced split");
        if (got != ref) {
          std::fprintf(stderr, "batch %lld mode %s S=%d differs\n", static_cast<long long>(rows),
                       md.name, s);
        }
        KOPS_TEST_CHECK(got == ref);
      }
    }
  }
}

/// One persistent barrier buffer serves S = 8, 2, 16, 4, 2 without being zeroed again: the
/// counter accumulates 16/S per arrival, so every barrier target stays reachable across split
/// counts. Each call must still draw Path A's tokens.
void test_fused_split_reuse() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int64_t rows = 8, vocab = 32768;
  std::vector<float> f(static_cast<size_t>(rows * vocab));
  std::mt19937 gen(77u);
  std::normal_distribution<float> nd(0.0f, 2.0f);
  for (float& x : f) x = nd(gen);
  const auto h = cast_from_float<__nv_bfloat16>(f);
  auto* d = kops_test::device_alloc<__nv_bfloat16>(h.size());
  auto* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(rows));
  kops_test::copy_in(d, h);
  const Tensor t = kops_test::tensor2d(d, DType::kBFloat16, rows, vocab);
  const Tensor ti = tensor1d(d_ids, DType::kInt32, rows);

  struct Mode {
    const char* name;
    float t, tp;
    int k;
  };
  const Mode modes[] = {{"k50", 1.0f, 1.0f, 50}, {"p0.9", 1.0f, 0.9f, 0}};
  for (const Mode& md : modes) {
    auto fp = fused_params(md.t, 5150u);
    fp.sampling.default_top_k = md.k;
    fp.sampling.default_top_p = md.tp;
    kops::sample::SamplingPlan plan{};
    KOPS_EXPECT_OK(kops::sample::sampling_plan(t, fp.sampling, ctx, &plan), "reuse plan");
    kops::sample::SamplingPlan wide = plan;
    wide.splits = 16;
    kops::WorkspaceRequirements req{};
    KOPS_EXPECT_OK(kops::sample::sampling_fused_workspace_requirements(
                       static_cast<int32_t>(rows), static_cast<int32_t>(vocab), wide, &req),
                   "reuse workspace requirements");
    kops::Workspace ws{};
    ws.device_scratch = req.device_scratch_bytes ? kops_test::device_alloc<uint8_t>(req.device_scratch_bytes) : nullptr;
    ws.device_scratch_bytes = req.device_scratch_bytes;
    ws.device_persistent =
        req.device_persistent_bytes ? kops_test::device_alloc<uint8_t>(req.device_persistent_bytes) : nullptr;
    ws.device_persistent_bytes = req.device_persistent_bytes;
    if (ws.device_persistent != nullptr) cudaMemset(ws.device_persistent, 0, ws.device_persistent_bytes);
    auto run = [&](int splits, kops::Status* out = nullptr) {
      plan.splits = splits;
      const kops::Status s = kops::sample::sample_fused(ti, t, fp, plan, ws, ctx);
      if (out != nullptr) {
        *out = s;
      } else {
        KOPS_EXPECT_OK(s, "reuse sample_fused");
      }
      return kops_test::copy_out(d_ids, static_cast<size_t>(rows));
    };
    const auto ref = run(1);
    for (int s : {8, 2, 16, 4, 2}) {
      kops::Status st = kops::Status::kSuccess;
      const auto got = run(s, &st);
      if (st == kops::Status::kPlanInvalidated) continue;  // more CTAs than the device holds
      KOPS_EXPECT_OK(st, "reuse sample_fused");
      if (got != ref) {
        std::fprintf(stderr, "reuse mode %s S=%d differs:", md.name, s);
        for (size_t i = 0; i < ref.size(); ++i) std::fprintf(stderr, " %d/%d", ref[i], got[i]);
        std::fprintf(stderr, "\n");
      }
      KOPS_TEST_CHECK(got == ref);
    }
    kops_test::device_free(static_cast<uint8_t*>(ws.device_scratch));
    kops_test::device_free(static_cast<uint8_t*>(ws.device_persistent));
  }
  kops_test::device_free(d);
  kops_test::device_free(d_ids);
}

void test_sampling_distribution() {
  const int64_t batch = 2048;
  const int64_t vocab = 16;
  const int launches = 512;  // 1,048,576 draws per sampler
  std::vector<float> logits(static_cast<size_t>(vocab));
  std::mt19937 rng(99u);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (float& x : logits) x = dist(rng);
  std::vector<float> logits_all(static_cast<size_t>(batch * vocab));
  for (int64_t r = 0; r < batch; ++r) {
    for (int64_t v = 0; v < vocab; ++v) logits_all[static_cast<size_t>(r * vocab + v)] = logits[static_cast<size_t>(v)];
  }
  float* d_logits = kops_test::device_alloc<float>(logits_all.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
  kops_test::copy_in(d_logits, logits_all);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kFloat32, batch, vocab);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, batch);

  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::sample::sampling_workspace_requirements(
                     static_cast<int32_t>(batch), static_cast<int32_t>(vocab), 0, &req),
                 "sample workspace");
  void* scratch = nullptr;
  KOPS_TEST_CHECK(cudaMalloc(&scratch, req.device_scratch_bytes) == cudaSuccess);
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = req.device_scratch_bytes;

  // fp64 expected distribution: softmax, top-k (4), top-p (0.8) on the remainder.
  std::vector<double> probs_d(static_cast<size_t>(vocab));
  double mx = -1e300;
  for (float x : logits) mx = std::max(mx, static_cast<double>(x));
  double z = 0.0;
  for (int64_t v = 0; v < vocab; ++v) {
    probs_d[static_cast<size_t>(v)] = std::exp(static_cast<double>(logits[static_cast<size_t>(v)]) - mx);
    z += probs_d[static_cast<size_t>(v)];
  }
  for (double& p : probs_d) p /= z;
  const std::vector<double> expected = truncate_kpp(probs_d, 4, 0.8);

  kops::sample::SamplingParams params{};
  params.default_temperature = 1.0f;
  params.default_top_k = 4;
  params.default_top_p = 0.8f;
  params.default_min_p = 0.0f;
  params.rng.global_seed = 1234u;
  std::vector<double> hist(static_cast<size_t>(vocab), 0.0);
  for (int it = 0; it < launches; ++it) {
    params.rng.global_seed = 100000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::sample(t_ids, t_logits, params, ws, ctx), "sample");
    const auto ids = kops_test::copy_out(d_ids, static_cast<size_t>(batch));
    for (int32_t id : ids) {
      KOPS_TEST_CHECK(id >= 0 && id < vocab);
      if (id >= 0 && id < vocab) hist[static_cast<size_t>(id)] += 1.0;
    }
  }
  std::vector<double> exp_counts(static_cast<size_t>(vocab));
  for (int64_t v = 0; v < vocab; ++v) {
    exp_counts[static_cast<size_t>(v)] =
        expected[static_cast<size_t>(v)] * static_cast<double>(batch) * launches;
  }
  int df = 0;
  const double chi = chi_square(hist, exp_counts, &df);
  KOPS_TEST_CHECK(df >= 1);
  KOPS_TEST_CHECK(chi < 80.0);  // df <= 15; p < 1e-9 at 80

  // Same seed reproduces the same ids.
  params.rng.global_seed = 424242u;
  KOPS_EXPECT_OK(kops::sample::sample(t_ids, t_logits, params, ws, ctx), "sample repeat 1");
  const auto ids1 = kops_test::copy_out(d_ids, static_cast<size_t>(batch));
  KOPS_EXPECT_OK(kops::sample::sample(t_ids, t_logits, params, ws, ctx), "sample repeat 2");
  const auto ids2 = kops_test::copy_out(d_ids, static_cast<size_t>(batch));
  KOPS_TEST_CHECK(ids1 == ids2);

  // Gumbel-max sampler: full softmax distribution (top-p disabled).
  kops::sample::SamplingParams gp{};
  gp.default_temperature = 1.0f;
  gp.default_top_k = 0;
  gp.default_top_p = 1.0f;
  gp.default_min_p = 0.0f;
  std::vector<double> ghist(static_cast<size_t>(vocab), 0.0);
  for (int it = 0; it < launches; ++it) {
    gp.rng.global_seed = 700000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::gumbel_max_sample(t_ids, t_logits, gp, ctx), "gumbel");
    const auto ids = kops_test::copy_out(d_ids, static_cast<size_t>(batch));
    for (int32_t id : ids) {
      if (id >= 0 && id < vocab) ghist[static_cast<size_t>(id)] += 1.0;
    }
  }
  for (int64_t v = 0; v < vocab; ++v) {
    exp_counts[static_cast<size_t>(v)] = probs_d[static_cast<size_t>(v)] * batch * launches;
  }
  const double gchi = chi_square(ghist, exp_counts, &df);
  KOPS_TEST_CHECK(df >= 1);
  KOPS_TEST_CHECK(gchi < 80.0);

  // Contract: gumbel rejects top-p truncation, tiny workspace is reported.
  kops::sample::SamplingParams bad = gp;
  bad.default_top_p = 0.9f;
  KOPS_EXPECT_STATUS(kops::sample::gumbel_max_sample(t_ids, t_logits, bad, ctx),
                     kops::Status::kNotImplemented, "gumbel top_p");
  kops::Workspace tiny{};
  tiny.device_scratch = scratch;
  tiny.device_scratch_bytes = 8;
  KOPS_EXPECT_STATUS(kops::sample::sample(t_ids, t_logits, params, tiny, ctx),
                     kops::Status::kWorkspaceTooSmall, "sample tiny workspace");

  cudaFree(scratch);
  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
}

/// gumbel_max_sample() with top-k: the candidate threshold path (small boundary bin) and the
/// radix fallback (a tie group larger than the buffer) must both draw only from the exact k
/// tokens ordered by value descending, then id ascending.
void test_gumbel_topk() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const int64_t batch = 8;
  constexpr int64_t vocab = 8192;
  constexpr int k = 50;
  auto check_row = [&](const std::vector<float>& row, const char* name) {
    std::vector<int32_t> order(static_cast<size_t>(vocab));
    for (int64_t v = 0; v < vocab; ++v) order[static_cast<size_t>(v)] = static_cast<int32_t>(v);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      return row[static_cast<size_t>(a)] > row[static_cast<size_t>(b)] ||
             (row[static_cast<size_t>(a)] == row[static_cast<size_t>(b)] && a < b);
    });
    std::vector<char> in_top(static_cast<size_t>(vocab), 0);
    for (int i = 0; i < k; ++i) in_top[static_cast<size_t>(order[static_cast<size_t>(i)])] = 1;
    std::vector<float> all(static_cast<size_t>(batch * vocab));
    for (int64_t r = 0; r < batch; ++r) {
      std::copy(row.begin(), row.end(), all.begin() + static_cast<size_t>(r * vocab));
    }
    float* d = kops_test::device_alloc<float>(all.size());
    int32_t* ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(batch));
    kops_test::copy_in(d, all);
    const Tensor t = kops_test::tensor2d(d, DType::kFloat32, batch, vocab);
    const Tensor ti = tensor1d(ids, DType::kInt32, batch);
    kops::sample::SamplingParams p{};
    p.default_temperature = 1.0f;
    p.default_top_k = k;
    p.default_top_p = 1.0f;
    p.default_min_p = 0.0f;
    int bad = 0;
    for (uint64_t seed = 0; seed < 16; ++seed) {
      p.rng.global_seed = 900u + seed;
      KOPS_EXPECT_OK(kops::sample::gumbel_max_sample(ti, t, p, ctx), "gumbel top-k");
      const auto got = kops_test::copy_out(ids, static_cast<size_t>(batch));
      for (int32_t id : got) {
        if (id < 0 || id >= vocab || in_top[static_cast<size_t>(id)] == 0) ++bad;
      }
    }
    if (bad != 0) std::fprintf(stderr, "%s: %d outside the top-%d set\n", name, bad, k);
    KOPS_TEST_CHECK(bad == 0);
    kops_test::device_free(d);
    kops_test::device_free(ids);
  };
  check_row(uniform_row(vocab, -3.0f, 3.0f, 9u), "gumbel topk random");
  // One tie group of 2048 tokens: the boundary bin overflows kCand and takes the radix fallback.
  std::vector<float> flat(static_cast<size_t>(vocab), 2.0f);
  check_row(flat, "gumbel topk flat");
}

/// The sample() fast path runs greedy, unfiltered and top-p-only rows through the fused body.
/// Greedy must still resolve ties to the lowest id and write -1 for a row with no admissible
/// token; a top-p-only run must keep the truncated distribution and reproduce under one seed.
void test_sample_fast_paths() {
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const int64_t rows = 6;
  const int64_t vocab = 1001;  // non-vectorized tail
  std::vector<float> f(static_cast<size_t>(rows * vocab));
  std::mt19937 gen(17u);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  for (float& x : f) x = nd(gen);
  // Row 1: tie at the maximum (lowest id wins). Row 2: two +inf (lowest id wins).
  // Row 3: all -inf. Row 4: all NaN (treated as -inf). Row 5: maximum in the last slot.
  f[1 * vocab + 7] = 40.0f;
  f[1 * vocab + 300] = 40.0f;
  f[2 * vocab + 5] = inf;
  f[2 * vocab + 900] = inf;
  for (int64_t v = 0; v < vocab; ++v) {
    f[3 * vocab + v] = -inf;
    f[4 * vocab + v] = nan;
  }
  f[5 * vocab + vocab - 1] = 50.0f;

  float* d_logits = kops_test::device_alloc<float>(f.size());
  int32_t* d_ids = kops_test::device_alloc<int32_t>(static_cast<size_t>(rows));
  kops_test::copy_in(d_logits, f);
  const Tensor t_logits = kops_test::tensor2d(d_logits, DType::kFloat32, rows, vocab);
  const Tensor t_ids = tensor1d(d_ids, DType::kInt32, rows);

  kops::WorkspaceRequirements req{};
  KOPS_EXPECT_OK(kops::sample::sampling_workspace_requirements(
                     static_cast<int32_t>(rows), static_cast<int32_t>(vocab), 0, &req),
                 "sample fast workspace");
  void* scratch = nullptr;
  KOPS_TEST_CHECK(cudaMalloc(&scratch, req.device_scratch_bytes) == cudaSuccess);
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = req.device_scratch_bytes;

  kops::sample::SamplingParams greedy{};
  greedy.default_temperature = 0.0f;
  greedy.default_top_k = 0;
  greedy.default_top_p = 1.0f;
  greedy.default_min_p = 0.0f;
  greedy.rng.global_seed = 9001u;
  KOPS_EXPECT_OK(kops::sample::sample(t_ids, t_logits, greedy, ws, ctx), "sample greedy");
  const auto got = kops_test::copy_out(d_ids, static_cast<size_t>(rows));
  std::vector<int32_t> want(static_cast<size_t>(rows));
  for (int64_t r = 0; r < rows; ++r) {
    if (r == 3 || r == 4) {
      want[static_cast<size_t>(r)] = -1;
      continue;
    }
    std::vector<double> row_d(static_cast<size_t>(vocab));
    for (int64_t v = 0; v < vocab; ++v) {
      row_d[static_cast<size_t>(v)] = f[static_cast<size_t>(r * vocab + v)];
    }
    want[static_cast<size_t>(r)] = static_cast<int32_t>(argmax_lowest(row_d.data(), vocab));
  }
  for (int64_t r = 0; r < rows; ++r) {
    if (got[static_cast<size_t>(r)] != want[static_cast<size_t>(r)]) {
      std::fprintf(stderr, "sample greedy row %lld: %d/%d\n", static_cast<long long>(r),
                   got[static_cast<size_t>(r)], want[static_cast<size_t>(r)]);
    }
  }
  KOPS_TEST_CHECK(got == want);
  // Greedy ignores the RNG: the same calls with another seed draw the same ids.
  greedy.rng.global_seed = 123u;
  KOPS_EXPECT_OK(kops::sample::sample(t_ids, t_logits, greedy, ws, ctx), "sample greedy repeat");
  KOPS_TEST_CHECK(kops_test::copy_out(d_ids, static_cast<size_t>(rows)) == want);
  kops_test::device_free(d_logits);
  kops_test::device_free(d_ids);
  cudaFree(scratch);

  // Top-p only: the sampled histogram must match the renormalized nucleus, and one seed must
  // reproduce the same ids (the fused body is counter-RNG based).
  const int64_t pbatch = 1024;
  const int64_t pvocab = 32;
  std::vector<float> plogits(static_cast<size_t>(pvocab));
  for (int64_t v = 0; v < pvocab; ++v) plogits[static_cast<size_t>(v)] = 0.35f * static_cast<float>(v);
  std::vector<float> pall(static_cast<size_t>(pbatch * pvocab));
  for (int64_t r = 0; r < pbatch; ++r) {
    std::copy(plogits.begin(), plogits.end(), pall.begin() + static_cast<size_t>(r * pvocab));
  }
  float* d_plogits = kops_test::device_alloc<float>(pall.size());
  int32_t* d_pids = kops_test::device_alloc<int32_t>(static_cast<size_t>(pbatch));
  kops_test::copy_in(d_plogits, pall);
  const Tensor pt_logits = kops_test::tensor2d(d_plogits, DType::kFloat32, pbatch, pvocab);
  const Tensor pt_ids = tensor1d(d_pids, DType::kInt32, pbatch);
  kops::WorkspaceRequirements preq{};
  KOPS_EXPECT_OK(kops::sample::sampling_workspace_requirements(
                     static_cast<int32_t>(pbatch), static_cast<int32_t>(pvocab), 0, &preq),
                 "sample p workspace");
  kops::Workspace pws{};
  pws.device_scratch = kops_test::device_alloc<uint8_t>(preq.device_scratch_bytes);
  pws.device_scratch_bytes = preq.device_scratch_bytes;

  std::vector<double> p_d(static_cast<size_t>(pvocab));
  double pm = -1e300;
  for (float x : plogits) pm = std::max(pm, static_cast<double>(x));
  double pz = 0.0;
  for (int64_t v = 0; v < pvocab; ++v) {
    p_d[static_cast<size_t>(v)] = std::exp(static_cast<double>(plogits[static_cast<size_t>(v)]) - pm);
    pz += p_d[static_cast<size_t>(v)];
  }
  for (double& x : p_d) x /= pz;
  const std::vector<double> p_expect = truncate_kpp(p_d, static_cast<int>(pvocab), 0.6);

  kops::sample::SamplingParams po{};
  po.default_temperature = 1.0f;
  po.default_top_k = 0;
  po.default_top_p = 0.6f;
  po.default_min_p = 0.0f;
  const int launches = 256;
  std::vector<double> hist(static_cast<size_t>(pvocab), 0.0);
  for (int it = 0; it < launches; ++it) {
    po.rng.global_seed = 5000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::sample(pt_ids, pt_logits, po, pws, ctx), "sample top-p only");
    const auto ids = kops_test::copy_out(d_pids, static_cast<size_t>(pbatch));
    for (int32_t id : ids) {
      KOPS_TEST_CHECK(id >= 0 && id < pvocab);
      if (id >= 0 && id < pvocab) hist[static_cast<size_t>(id)] += 1.0;
    }
  }
  std::vector<double> expect_counts(static_cast<size_t>(pvocab));
  for (int64_t v = 0; v < pvocab; ++v) {
    expect_counts[static_cast<size_t>(v)] =
        p_expect[static_cast<size_t>(v)] * static_cast<double>(pbatch) * launches;
  }
  int df = 0;
  const double chi = chi_square(hist, expect_counts, &df);
  KOPS_TEST_CHECK(df >= 1);
  KOPS_TEST_CHECK(chi < 80.0);

  po.rng.global_seed = 777u;
  KOPS_EXPECT_OK(kops::sample::sample(pt_ids, pt_logits, po, pws, ctx), "sample top-p repeat 1");
  const auto pids1 = kops_test::copy_out(d_pids, static_cast<size_t>(pbatch));
  KOPS_EXPECT_OK(kops::sample::sample(pt_ids, pt_logits, po, pws, ctx), "sample top-p repeat 2");
  KOPS_TEST_CHECK(kops_test::copy_out(d_pids, static_cast<size_t>(pbatch)) == pids1);

  // Min-p only is also routed to the fused filtered body: every returned token must be in
  // `probs >= min_p * max(probs)`, with the sampled set renormalized.
  {
    kops::sample::SamplingParams mp{};
    mp.default_temperature = 1.0f;
    mp.default_top_k = 0;
    mp.default_top_p = 1.0f;
    mp.default_min_p = 0.25f;
    const double pmax = *std::max_element(p_d.begin(), p_d.end());
    std::vector<char> in_set(static_cast<size_t>(pvocab), 0);
    int kept = 0;
    for (int64_t v = 0; v < pvocab; ++v) {
      if (p_d[static_cast<size_t>(v)] >= 0.25 * pmax) {
        in_set[static_cast<size_t>(v)] = 1;
        ++kept;
      }
    }
    KOPS_TEST_CHECK(kept > 0);
    int bad = 0;
    for (int it = 0; it < 64; ++it) {
      mp.rng.global_seed = 3000u + static_cast<uint64_t>(it);
      KOPS_EXPECT_OK(kops::sample::sample(pt_ids, pt_logits, mp, pws, ctx), "sample min-p only");
      const auto ids = kops_test::copy_out(d_pids, static_cast<size_t>(pbatch));
      for (int32_t id : ids) {
        if (id < 0 || id >= pvocab || in_set[static_cast<size_t>(id)] == 0) ++bad;
      }
    }
    KOPS_TEST_CHECK(bad == 0);
  }

  kops_test::device_free(d_plogits);
  kops_test::device_free(d_pids);
  kops_test::device_free(static_cast<uint8_t*>(pws.device_scratch));

  // A vectorizable large vocab makes the auto plan split the row. sample() carves
  // the fused exchange area and barrier out of the fp32 row it does not write;
  // two calls must both produce a valid token (the barrier is re-zeroed), and a
  // greedy call must still take the one-pass fused body.
  {
    const int64_t big_rows = 1;
    const int64_t big_vocab = 151936;
    std::vector<float> lf(static_cast<size_t>(big_vocab));
    std::mt19937 big_gen(23u);
    std::normal_distribution<float> big_nd(0.0f, 3.0f);
    for (float& x : lf) x = big_nd(big_gen);
    lf[7777] = 50.0f;
    float* d_lf = kops_test::device_alloc<float>(lf.size());
    int32_t* d_id = kops_test::device_alloc<int32_t>(static_cast<size_t>(big_rows));
    kops_test::copy_in(d_lf, lf);
    const Tensor bt = kops_test::tensor2d(d_lf, DType::kFloat32, big_rows, big_vocab);
    const Tensor bid = tensor1d(d_id, DType::kInt32, big_rows);
    kops::WorkspaceRequirements breq{};
    KOPS_EXPECT_OK(kops::sample::sampling_workspace_requirements(
                       static_cast<int32_t>(big_rows), static_cast<int32_t>(big_vocab), 0, &breq),
                   "sample split workspace");
    kops::Workspace bws{};
    bws.device_scratch = kops_test::device_alloc<uint8_t>(breq.device_scratch_bytes);
    bws.device_scratch_bytes = breq.device_scratch_bytes;

    kops::sample::SamplingParams bp{};
    bp.default_temperature = 1.0f;
    bp.default_top_k = 0;
    bp.default_top_p = 0.9f;
    bp.default_min_p = 0.0f;
    for (uint64_t seed : {11u, 12u}) {
      bp.rng.global_seed = seed;
      KOPS_EXPECT_OK(kops::sample::sample(bid, bt, bp, bws, ctx), "sample split top-p");
      const int32_t id = kops_test::copy_out(d_id, 1)[0];
      KOPS_TEST_CHECK(id >= 0 && id < big_vocab);
    }
    bp.default_temperature = 0.0f;
    bp.default_top_p = 1.0f;
    KOPS_EXPECT_OK(kops::sample::sample(bid, bt, bp, bws, ctx), "sample split greedy");
    KOPS_TEST_CHECK(kops_test::copy_out(d_id, 1)[0] == 7777);

    kops_test::device_free(d_lf);
    kops_test::device_free(d_id);
    kops_test::device_free(static_cast<uint8_t*>(bws.device_scratch));
  }
}

// =============================================================================
// B10 / Mốc 5: speculative decoding (S03)
// =============================================================================
std::vector<float> spec_softmax(std::vector<float> logits) {
  const float m = *std::max_element(logits.begin(), logits.end());
  double z = 0.0;
  for (float& x : logits) {
    x = std::exp(x - m);
    z += x;
  }
  for (float& x : logits) x = static_cast<float>(x / z);
  return logits;
}

void fill_prob_rows(float* dst, int rows, int vocab, const std::vector<float>& row) {
  std::vector<float> all(static_cast<size_t>(rows) * vocab);
  for (int r = 0; r < rows; ++r) {
    std::copy(row.begin(), row.end(), all.begin() + static_cast<size_t>(r) * vocab);
  }
  kops_test::copy_in(dst, all);
}

void test_spec_chain_greedy() {
  constexpr int B = 4, K = 3, V = 8;
  const int total = 9;
  float* d_tp = kops_test::device_alloc<float>(static_cast<size_t>(total + B) * V);
  int32_t* d_cu = kops_test::device_alloc<int32_t>(B + 1);
  int32_t* d_draft = kops_test::device_alloc<int32_t>(total);
  int32_t* d_argmax = kops_test::device_alloc<int32_t>(total + B);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (K + 1));
  int32_t* d_num = kops_test::device_alloc<int32_t>(B);
  uint8_t* d_greedy = kops_test::device_alloc<uint8_t>(B);
  if (d_tp == nullptr || d_cu == nullptr || d_draft == nullptr || d_argmax == nullptr ||
      d_out == nullptr || d_num == nullptr || d_greedy == nullptr) {
    return;
  }
  std::vector<float> zeros(static_cast<size_t>(total + B) * V, 0.0f);
  kops_test::copy_in(d_tp, zeros);
  kops_test::copy_in(d_cu, std::vector<int32_t>{0, 3, 6, 6, 9});
  kops_test::copy_in(d_draft, std::vector<int32_t>{1, 2, 3, 1, 2, 3, 0, 0, 0});
  kops_test::copy_in(d_argmax, std::vector<int32_t>{1, 2, 3, 1, 7, 3, 5, 5, 5, 4, 8, 6, 7});
  kops_test::copy_in(d_out, std::vector<int32_t>(B * (K + 1), -99));
  kops_test::copy_in(d_num, std::vector<int32_t>(B, -99));
  kops_test::copy_in(d_greedy, std::vector<uint8_t>{1, 1, 1, 1});

  kops::sample::SpecVerifyParams p{};
  p.draft_token_ids = d_draft;
  p.cu_num_draft = d_cu;
  p.target_probs = d_tp;
  p.target_argmax = d_argmax;
  p.is_greedy = d_greedy;
  p.output_token_ids = d_out;
  p.num_accepted = d_num;
  p.batch_size = B;
  p.max_draft = K;
  p.vocab_size = V;
  p.rng.global_seed = 7;
  p.rng.stream_id = 1;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::sample::rejection_sample(p, ctx), "rejection_sample greedy");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  expect_i32(kops_test::copy_out(d_out, B * (K + 1)),
             std::vector<int32_t>{1, 2, 3, 4, 1, 7, -1, -1, 6, -1, -1, -1, 5, -1, -1, -1},
             "chain greedy output");
  expect_i32(kops_test::copy_out(d_num, B), std::vector<int32_t>{3, 1, 0, 0},
             "chain greedy accepted");

  kops_test::device_free(d_tp);
  kops_test::device_free(d_cu);
  kops_test::device_free(d_draft);
  kops_test::device_free(d_argmax);
  kops_test::device_free(d_out);
  kops_test::device_free(d_num);
  kops_test::device_free(d_greedy);
}

void test_spec_chain_distribution() {
  constexpr int B = 2048, K = 4, V = 16, launches = 512;
  const int total = B * K;
  std::vector<float> logits(V);
  fill(logits, 811u, -2.0f, 2.0f);
  const std::vector<float> p = spec_softmax(logits);

  float* d_tp = kops_test::device_alloc<float>(static_cast<size_t>(total + B) * V);
  float* d_dp = nullptr;  // target-only
  int32_t* d_cu = kops_test::device_alloc<int32_t>(B + 1);
  int32_t* d_draft = kops_test::device_alloc<int32_t>(total);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (K + 1));
  int32_t* d_num = kops_test::device_alloc<int32_t>(B);
  if (d_tp == nullptr || d_cu == nullptr || d_draft == nullptr || d_out == nullptr ||
      d_num == nullptr) return;
  fill_prob_rows(d_tp, total + B, V, p);
  std::vector<int32_t> cu(B + 1), draft(total);
  std::mt19937 rng(1919u);
  std::uniform_int_distribution<int32_t> tok(0, V - 1);
  for (int b = 0; b <= B; ++b) cu[static_cast<size_t>(b)] = b * K;
  for (int32_t& x : draft) x = tok(rng);
  kops_test::copy_in(d_cu, cu);
  kops_test::copy_in(d_draft, draft);

  kops::sample::SpecVerifyParams pms{};
  pms.draft_token_ids = d_draft;
  pms.cu_num_draft = d_cu;
  pms.target_probs = d_tp;
  pms.draft_probs = d_dp;
  pms.output_token_ids = d_out;
  pms.num_accepted = d_num;
  pms.batch_size = B;
  pms.max_draft = K;
  pms.vocab_size = V;
  pms.rng.global_seed = 1;
  pms.rng.stream_id = 1;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  std::vector<double> hist(V, 0.0);
  for (int it = 0; it < launches; ++it) {
    pms.rng.global_seed = 100000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::rejection_sample(pms, ctx), "rejection_sample target-only");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto out = kops_test::copy_out(d_out, B * (K + 1));
    for (int b = 0; b < B; ++b) {
      const int32_t id = out[static_cast<size_t>(b * (K + 1))];
      KOPS_TEST_CHECK(id >= 0 && id < V);
      if (id >= 0 && id < V) hist[static_cast<size_t>(id)] += 1.0;
    }
  }
  std::vector<double> expected(V);
  for (int v = 0; v < V; ++v) expected[static_cast<size_t>(v)] = p[static_cast<size_t>(v)] * B * launches;
  int df = 0;
  const double chi = chi_square(hist, expected, &df);
  KOPS_TEST_CHECK(df >= 1);
  KOPS_TEST_CHECK(chi < 80.0);

  // Same seed reproduces the same rows.
  pms.rng.global_seed = 424242u;
  KOPS_EXPECT_OK(kops::sample::rejection_sample(pms, ctx), "rejection_sample repeat 1");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto out1 = kops_test::copy_out(d_out, B * (K + 1));
  KOPS_EXPECT_OK(kops::sample::rejection_sample(pms, ctx), "rejection_sample repeat 2");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto out2 = kops_test::copy_out(d_out, B * (K + 1));
  KOPS_TEST_CHECK(out1 == out2);

  kops_test::device_free(d_tp);
  kops_test::device_free(d_cu);
  kops_test::device_free(d_draft);
  kops_test::device_free(d_out);
  kops_test::device_free(d_num);
}

void test_spec_chain_probabilistic() {
  constexpr int B = 1024, K = 4, V = 8, launches = 512;
  const int total = B * K;
  std::vector<float> pl(V), ql(V);
  fill(pl, 313u, -2.0f, 2.0f);
  fill(ql, 317u, -2.0f, 2.0f);
  const std::vector<float> p = spec_softmax(pl);
  const std::vector<float> q = spec_softmax(ql);

  float* d_tp = kops_test::device_alloc<float>(static_cast<size_t>(total + B) * V);
  float* d_dp = kops_test::device_alloc<float>(static_cast<size_t>(total) * V);
  int32_t* d_cu = kops_test::device_alloc<int32_t>(B + 1);
  int32_t* d_draft = kops_test::device_alloc<int32_t>(total);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (K + 1));
  int32_t* d_num = kops_test::device_alloc<int32_t>(B);
  if (d_tp == nullptr || d_dp == nullptr || d_cu == nullptr || d_draft == nullptr ||
      d_out == nullptr || d_num == nullptr) return;
  fill_prob_rows(d_tp, total + B, V, p);
  fill_prob_rows(d_dp, total, V, q);
  std::vector<int32_t> cu(B + 1), draft(total);
  std::mt19937 rng(929u);
  std::discrete_distribution<int32_t> pick(q.begin(), q.end());
  for (int b = 0; b <= B; ++b) cu[static_cast<size_t>(b)] = b * K;
  for (int32_t& x : draft) x = pick(rng);
  kops_test::copy_in(d_cu, cu);
  kops_test::copy_in(d_draft, draft);

  kops::sample::SpecVerifyParams pms{};
  pms.draft_token_ids = d_draft;
  pms.cu_num_draft = d_cu;
  pms.target_probs = d_tp;
  pms.draft_probs = d_dp;
  pms.output_token_ids = d_out;
  pms.num_accepted = d_num;
  pms.batch_size = B;
  pms.max_draft = K;
  pms.vocab_size = V;
  pms.rng.global_seed = 1;
  pms.rng.stream_id = 1;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  std::vector<double> hist(V, 0.0);
  for (int it = 0; it < launches; ++it) {
    for (int32_t& x : draft) x = pick(rng);
    kops_test::copy_in(d_draft, draft);
    pms.rng.global_seed = 700000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::rejection_sample(pms, ctx), "rejection_sample probabilistic");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto out = kops_test::copy_out(d_out, B * (K + 1));
    for (int b = 0; b < B; ++b) {
      const int32_t id = out[static_cast<size_t>(b * (K + 1))];
      KOPS_TEST_CHECK(id >= 0 && id < V);
      if (id >= 0 && id < V) hist[static_cast<size_t>(id)] += 1.0;
    }
  }
  std::vector<double> expected(V);
  for (int v = 0; v < V; ++v) expected[static_cast<size_t>(v)] = p[static_cast<size_t>(v)] * B * launches;
  int df = 0;
  const double chi = chi_square(hist, expected, &df);
  KOPS_TEST_CHECK(df >= 1);
  KOPS_TEST_CHECK(chi < 80.0);

  // q == p: every draft is accepted with certainty.
  constexpr int BS = 8;
  float* d_small = kops_test::device_alloc<float>(static_cast<size_t>(BS * K + BS) * V);
  int32_t* d_cus = kops_test::device_alloc<int32_t>(BS + 1);
  int32_t* d_ds = kops_test::device_alloc<int32_t>(BS * K);
  int32_t* d_os = kops_test::device_alloc<int32_t>(BS * (K + 1));
  int32_t* d_ns = kops_test::device_alloc<int32_t>(BS);
  if (d_small != nullptr && d_cus != nullptr && d_ds != nullptr && d_os != nullptr &&
      d_ns != nullptr) {
    fill_prob_rows(d_small, BS * K + BS, V, p);
    std::vector<int32_t> cus(BS + 1), ds(BS * K);
    for (int b = 0; b <= BS; ++b) cus[static_cast<size_t>(b)] = b * K;
    for (int i = 0; i < BS * K; ++i) ds[static_cast<size_t>(i)] = static_cast<int32_t>(i % V);
    kops_test::copy_in(d_cus, cus);
    kops_test::copy_in(d_ds, ds);
    pms.draft_token_ids = d_ds;
    pms.cu_num_draft = d_cus;
    pms.target_probs = d_small;
    pms.draft_probs = d_small;  // q == p
    pms.output_token_ids = d_os;
    pms.num_accepted = d_ns;
    pms.batch_size = BS;
    pms.rng.global_seed = 55;
    KOPS_EXPECT_OK(kops::sample::rejection_sample(pms, ctx), "rejection_sample q==p");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto ns = kops_test::copy_out(d_ns, BS);
    for (int32_t n : ns) KOPS_TEST_CHECK(n == K);
    const auto os = kops_test::copy_out(d_os, BS * (K + 1));
    for (int b = 0; b < BS; ++b) {
      for (int i = 0; i < K; ++i) {
        KOPS_TEST_CHECK(os[static_cast<size_t>(b * (K + 1) + i)] ==
                        ds[static_cast<size_t>(b * K + i)]);
      }
      KOPS_TEST_CHECK(os[static_cast<size_t>(b * (K + 1) + K)] >= 0);
    }
  }
  kops_test::device_free(d_small);
  kops_test::device_free(d_cus);
  kops_test::device_free(d_ds);
  kops_test::device_free(d_os);
  kops_test::device_free(d_ns);

  kops_test::device_free(d_tp);
  kops_test::device_free(d_dp);
  kops_test::device_free(d_cu);
  kops_test::device_free(d_draft);
  kops_test::device_free(d_out);
  kops_test::device_free(d_num);
}

void test_spec_acceptance_length() {
  constexpr int B = 1024, K = 8, V = 8, launches = 128;
  std::vector<float> pl(V), ql(V);
  fill(pl, 501u, -2.0f, 2.0f);
  fill(ql, 503u, -2.0f, 2.0f);
  const std::vector<float> p = spec_softmax(pl);
  const std::vector<float> q = spec_softmax(ql);
  double a = 0.0;
  for (int v = 0; v < V; ++v) a += std::min(p[static_cast<size_t>(v)], q[static_cast<size_t>(v)]);
  double expected = 0.0;
  double term = 1.0;
  for (int i = 0; i < K; ++i) {
    term *= a;
    expected += term;
  }

  const int total = B * K;
  float* d_tp = kops_test::device_alloc<float>(static_cast<size_t>(total + B) * V);
  float* d_dp = kops_test::device_alloc<float>(static_cast<size_t>(total) * V);
  int32_t* d_cu = kops_test::device_alloc<int32_t>(B + 1);
  int32_t* d_draft = kops_test::device_alloc<int32_t>(total);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (K + 1));
  int32_t* d_num = kops_test::device_alloc<int32_t>(B);
  if (d_tp == nullptr || d_dp == nullptr || d_cu == nullptr || d_draft == nullptr ||
      d_out == nullptr || d_num == nullptr) return;
  fill_prob_rows(d_tp, total + B, V, p);
  fill_prob_rows(d_dp, total, V, q);
  std::vector<int32_t> cu(B + 1), draft(total);
  std::mt19937 rng(777u);
  std::discrete_distribution<int32_t> pick(q.begin(), q.end());
  for (int b = 0; b <= B; ++b) cu[static_cast<size_t>(b)] = b * K;
  kops_test::copy_in(d_cu, cu);

  kops::sample::SpecVerifyParams pms{};
  pms.draft_token_ids = d_draft;
  pms.cu_num_draft = d_cu;
  pms.target_probs = d_tp;
  pms.draft_probs = d_dp;
  pms.output_token_ids = d_out;
  pms.num_accepted = d_num;
  pms.batch_size = B;
  pms.max_draft = K;
  pms.vocab_size = V;
  pms.rng.stream_id = 1;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  double sum = 0.0;
  for (int it = 0; it < launches; ++it) {
    for (int32_t& x : draft) x = pick(rng);
    kops_test::copy_in(d_draft, draft);
    pms.rng.global_seed = 550000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::rejection_sample(pms, ctx), "rejection_sample acceptance");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto ns = kops_test::copy_out(d_num, B);
    for (int32_t n : ns) {
      KOPS_TEST_CHECK(n >= 0 && n <= K);
      sum += n;
    }
  }
  const double mean = sum / static_cast<double>(B) / launches;
  std::fprintf(stderr, "[b10] chain accepted length measured %.4f, expected %.4f\n", mean, expected);
  KOPS_TEST_CHECK(std::fabs(mean - expected) < 0.08);
  kops_test::device_free(d_tp);
  kops_test::device_free(d_dp);
  kops_test::device_free(d_cu);
  kops_test::device_free(d_draft);
  kops_test::device_free(d_out);
  kops_test::device_free(d_num);
}

void test_spec_chain_contracts() {
  constexpr int B = 1, K = 1, V = 4;
  float* d_tp = kops_test::device_alloc<float>(static_cast<size_t>(2 * V));
  int32_t* d_cu = kops_test::device_alloc<int32_t>(2);
  int32_t* d_draft = kops_test::device_alloc<int32_t>(1);
  int32_t* d_out = kops_test::device_alloc<int32_t>(2);
  int32_t* d_num = kops_test::device_alloc<int32_t>(1);
  uint8_t* d_greedy = kops_test::device_alloc<uint8_t>(1);
  if (d_tp == nullptr || d_cu == nullptr || d_draft == nullptr || d_out == nullptr ||
      d_num == nullptr || d_greedy == nullptr) return;
  kops_test::copy_in(d_tp, std::vector<float>(2 * V, 0.25f));
  kops_test::copy_in(d_cu, std::vector<int32_t>{0, 1});
  kops_test::copy_in(d_draft, std::vector<int32_t>{1});
  kops_test::copy_in(d_out, std::vector<int32_t>{-77, -77});
  kops_test::copy_in(d_num, std::vector<int32_t>{-77});
  kops_test::copy_in(d_greedy, std::vector<uint8_t>{1});

  kops::sample::SpecVerifyParams p{};
  p.draft_token_ids = d_draft;
  p.cu_num_draft = d_cu;
  p.target_probs = d_tp;
  p.output_token_ids = d_out;
  p.num_accepted = d_num;
  p.batch_size = B;
  p.max_draft = K;
  p.vocab_size = V;
  p.is_greedy = d_greedy;
  p.rng.global_seed = 1;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_STATUS(kops::sample::rejection_sample(p, ctx), kops::Status::kInvalidArgument,
                     "chain greedy needs argmax");
  KOPS_EXPECT_STATUS(kops::sample::rejection_sample(p, ctx), kops::Status::kInvalidArgument,
                     "chain greedy needs argmax again");
  auto bad = p;
  bad.batch_size = -1;
  KOPS_EXPECT_STATUS(kops::sample::rejection_sample(bad, ctx), kops::Status::kInvalidArgument,
                     "chain negative batch");
  bad = p;
  bad.vocab_size = 0;
  KOPS_EXPECT_STATUS(kops::sample::rejection_sample(bad, ctx), kops::Status::kInvalidArgument,
                     "chain zero vocab");
  bad = p;
  bad.max_draft = -1;
  KOPS_EXPECT_STATUS(kops::sample::rejection_sample(bad, ctx), kops::Status::kInvalidArgument,
                     "chain negative draft");
  bad = p;
  bad.target_probs = nullptr;
  KOPS_EXPECT_STATUS(kops::sample::rejection_sample(bad, ctx), kops::Status::kInvalidArgument,
                     "chain null target");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto untouched = kops_test::copy_out(d_out, 2);
  KOPS_TEST_CHECK(untouched[0] == -77 && untouched[1] == -77);

  // Empty batch is a no-op even with null arrays.
  kops::sample::SpecVerifyParams empty{};
  empty.batch_size = 0;
  empty.vocab_size = V;
  KOPS_EXPECT_OK(kops::sample::rejection_sample(empty, ctx), "chain empty batch");

  kops::LaunchContext hip = ctx;
  hip.backend = kops::BackendKind::kHip;
  bad = p;
  bad.is_greedy = nullptr;
  KOPS_EXPECT_STATUS(kops::sample::rejection_sample(bad, hip), kops::Status::kUnsupportedBackend,
                     "chain hip backend");

  kops_test::device_free(d_tp);
  kops_test::device_free(d_cu);
  kops_test::device_free(d_draft);
  kops_test::device_free(d_out);
  kops_test::device_free(d_num);
  kops_test::device_free(d_greedy);
}

void test_spec_chain_graph() {
  constexpr int B = 2, K = 2, V = 4;
  float* d_tp = kops_test::device_alloc<float>(static_cast<size_t>(B * K + B) * V);
  int32_t* d_cu = kops_test::device_alloc<int32_t>(B + 1);
  int32_t* d_draft = kops_test::device_alloc<int32_t>(B * K);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (K + 1));
  int32_t* d_num = kops_test::device_alloc<int32_t>(B);
  if (d_tp == nullptr || d_cu == nullptr || d_draft == nullptr || d_out == nullptr ||
      d_num == nullptr) return;
  kops_test::copy_in(d_tp, std::vector<float>(static_cast<size_t>(B * K + B) * V, 0.25f));
  kops_test::copy_in(d_cu, std::vector<int32_t>{0, K, 2 * K});
  kops_test::copy_in(d_draft, std::vector<int32_t>{0, 1, 2, 3});
  kops::sample::SpecVerifyParams p{};
  p.draft_token_ids = d_draft;
  p.cu_num_draft = d_cu;
  p.target_probs = d_tp;
  p.output_token_ids = d_out;
  p.num_accepted = d_num;
  p.batch_size = B;
  p.max_draft = K;
  p.vocab_size = V;
  p.rng.global_seed = 17;

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return;
  kops::LaunchContext gc{};
  if (!kops_test::make_context(&gc)) return;
  gc.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  KOPS_EXPECT_OK(kops::sample::rejection_sample(p, gc), "graph rejection_sample");
  cudaStreamEndCapture(stream, &graph);
  if (graph != nullptr && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    const auto n1 = kops_test::copy_out(d_num, B);
    const auto o1 = kops_test::copy_out(d_out, B * (K + 1));
    // Replay with new draft data through the same captured pointers.
    kops_test::copy_in(d_draft, std::vector<int32_t>{1, 2, 3, 0});
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    const auto n2 = kops_test::copy_out(d_num, B);
    const auto o2 = kops_test::copy_out(d_out, B * (K + 1));
    for (int b = 0; b < B; ++b) {
      KOPS_TEST_CHECK(n1[static_cast<size_t>(b)] >= 0 && n1[static_cast<size_t>(b)] <= K);
      KOPS_TEST_CHECK(n2[static_cast<size_t>(b)] >= 0 && n2[static_cast<size_t>(b)] <= K);
      KOPS_TEST_CHECK(o1[static_cast<size_t>(b * (K + 1))] >= 0);
      KOPS_TEST_CHECK(o2[static_cast<size_t>(b * (K + 1))] >= 0);
    }
    cudaGraphExecDestroy(exec);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  kops_test::device_free(d_tp);
  kops_test::device_free(d_cu);
  kops_test::device_free(d_draft);
  kops_test::device_free(d_out);
  kops_test::device_free(d_num);
}

void test_spec_tree_greedy() {
  constexpr int B = 2, N = 5, D = 3, V = 8;
  int32_t* d_cand = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_first = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_next = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_argmax = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_acc_idx = kops_test::device_alloc<int32_t>(B * (D + 1));
  int32_t* d_acc_len = kops_test::device_alloc<int32_t>(B);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (D + 1));
  if (d_cand == nullptr || d_first == nullptr || d_next == nullptr || d_argmax == nullptr ||
      d_acc_idx == nullptr || d_acc_len == nullptr || d_out == nullptr) return;
  std::vector<int32_t> cand, first, next, argmax;
  for (int b = 0; b < B; ++b) {
    cand.insert(cand.end(), {0, 1, 2, 3, 4});
    first.insert(first.end(), {1, 3, 4, -1, -1});
    next.insert(next.end(), {-1, 2, -1, -1, -1});
  }
  argmax = {2, 3, 4, 7, 6, 2, 3, 5, 7, 6};
  kops_test::copy_in(d_cand, cand);
  kops_test::copy_in(d_first, first);
  kops_test::copy_in(d_next, next);
  kops_test::copy_in(d_argmax, argmax);
  kops_test::copy_in(d_acc_idx, std::vector<int32_t>(B * (D + 1), -99));
  kops_test::copy_in(d_acc_len, std::vector<int32_t>(B, -99));
  kops_test::copy_in(d_out, std::vector<int32_t>(B * (D + 1), -99));

  kops::sample::TreeVerifyParams p{};
  p.candidates = d_cand;
  p.first_child = d_first;
  p.next_sibling = d_next;
  p.target_argmax = d_argmax;
  p.accept_index = d_acc_idx;
  p.accept_length = d_acc_len;
  p.output_token_ids = d_out;
  p.batch_size = B;
  p.num_nodes = N;
  p.max_depth = D;
  p.vocab_size = V;
  p.rng.global_seed = 11;
  p.rng.stream_id = 1;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::sample::tree_verify(p, ctx), "tree_verify greedy");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  expect_i32(kops_test::copy_out(d_acc_len, B), std::vector<int32_t>{2, 1}, "tree greedy length");
  expect_i32(kops_test::copy_out(d_out, B * (D + 1)),
             std::vector<int32_t>{2, 4, 6, -1, 2, 5, -1, -1}, "tree greedy output");
  expect_i32(kops_test::copy_out(d_acc_idx, B * (D + 1)),
             std::vector<int32_t>{2, 4, -1, -1, 2, -1, -1, -1}, "tree greedy index");

  kops_test::device_free(d_cand);
  kops_test::device_free(d_first);
  kops_test::device_free(d_next);
  kops_test::device_free(d_argmax);
  kops_test::device_free(d_acc_idx);
  kops_test::device_free(d_acc_len);
  kops_test::device_free(d_out);
}

void test_spec_tree_distribution() {
  constexpr int B = 1024, N = 4, D = 3, V = 8, launches = 512;
  std::vector<float> pl(V);
  fill(pl, 123u, -2.0f, 2.0f);
  const std::vector<float> p = spec_softmax(pl);
  float* d_tp = kops_test::device_alloc<float>(static_cast<size_t>(B * N) * V);
  int32_t* d_cand = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_first = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_next = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_ridx = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_acc_idx = kops_test::device_alloc<int32_t>(B * (D + 1));
  int32_t* d_acc_len = kops_test::device_alloc<int32_t>(B);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (D + 1));
  if (d_tp == nullptr || d_cand == nullptr || d_first == nullptr || d_next == nullptr ||
      d_ridx == nullptr || d_acc_idx == nullptr || d_acc_len == nullptr || d_out == nullptr) return;
  fill_prob_rows(d_tp, B * N, V, p);
  std::vector<int32_t> cand, first, next, ridx;
  for (int b = 0; b < B; ++b) {
    cand.insert(cand.end(), {0, 1, 2, 3});
    first.insert(first.end(), {1, 2, 3, -1});
    next.insert(next.end(), {-1, -1, -1, -1});
    ridx.insert(ridx.end(), {0, 1, 2, 3});
  }
  kops_test::copy_in(d_cand, cand);
  kops_test::copy_in(d_first, first);
  kops_test::copy_in(d_next, next);
  kops_test::copy_in(d_ridx, ridx);

  kops::sample::TreeVerifyParams pms{};
  pms.candidates = d_cand;
  pms.retrieve_index = d_ridx;
  pms.first_child = d_first;
  pms.next_sibling = d_next;
  pms.target_probs = d_tp;
  pms.accept_index = d_acc_idx;
  pms.accept_length = d_acc_len;
  pms.output_token_ids = d_out;
  pms.batch_size = B;
  pms.num_nodes = N;
  pms.max_depth = D;
  pms.vocab_size = V;
  pms.rng.global_seed = 1;
  pms.rng.stream_id = 1;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  std::vector<double> hist(V, 0.0);
  for (int it = 0; it < launches; ++it) {
    pms.rng.global_seed = 900000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::tree_verify(pms, ctx), "tree_verify target-only");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto out = kops_test::copy_out(d_out, B * (D + 1));
    for (int b = 0; b < B; ++b) {
      const int32_t id = out[static_cast<size_t>(b * (D + 1))];
      KOPS_TEST_CHECK(id >= 0 && id < V);
      if (id >= 0 && id < V) hist[static_cast<size_t>(id)] += 1.0;
    }
  }
  std::vector<double> expected(V);
  for (int v = 0; v < V; ++v) expected[static_cast<size_t>(v)] = p[static_cast<size_t>(v)] * B * launches;
  int df = 0;
  const double chi = chi_square(hist, expected, &df);
  KOPS_TEST_CHECK(df >= 1);
  KOPS_TEST_CHECK(chi < 80.0);

  kops_test::device_free(d_tp);
  kops_test::device_free(d_cand);
  kops_test::device_free(d_first);
  kops_test::device_free(d_next);
  kops_test::device_free(d_ridx);
  kops_test::device_free(d_acc_idx);
  kops_test::device_free(d_acc_len);
  kops_test::device_free(d_out);
}

void test_spec_tree_probabilistic() {
  constexpr int B = 1024, N = 4, D = 1, V = 8, launches = 512;
  std::vector<float> pl(V);
  fill(pl, 321u, -2.0f, 2.0f);
  const std::vector<float> p = spec_softmax(pl);
  std::vector<float> q(V, 0.0f);
  q[1] = 0.2f;
  q[2] = 0.3f;
  q[3] = 0.5f;
  float* d_tp = kops_test::device_alloc<float>(static_cast<size_t>(B * N) * V);
  float* d_dp = kops_test::device_alloc<float>(static_cast<size_t>(B * N) * V);
  int32_t* d_cand = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_first = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_next = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_ridx = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_acc_idx = kops_test::device_alloc<int32_t>(B * (D + 1));
  int32_t* d_acc_len = kops_test::device_alloc<int32_t>(B);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (D + 1));
  if (d_tp == nullptr || d_dp == nullptr || d_cand == nullptr || d_first == nullptr ||
      d_next == nullptr || d_ridx == nullptr || d_acc_idx == nullptr || d_acc_len == nullptr ||
      d_out == nullptr) return;
  fill_prob_rows(d_tp, B * N, V, p);
  fill_prob_rows(d_dp, B * N, V, q);
  std::vector<int32_t> cand, first, next, ridx;
  for (int b = 0; b < B; ++b) {
    cand.insert(cand.end(), {0, 1, 2, 3});
    first.insert(first.end(), {1, -1, -1, -1});
    next.insert(next.end(), {-1, 2, 3, -1});
    ridx.insert(ridx.end(), {0, 1, 2, 3});
  }
  kops_test::copy_in(d_cand, cand);
  kops_test::copy_in(d_first, first);
  kops_test::copy_in(d_next, next);
  kops_test::copy_in(d_ridx, ridx);

  kops::sample::TreeVerifyParams pms{};
  pms.candidates = d_cand;
  pms.retrieve_index = d_ridx;
  pms.first_child = d_first;
  pms.next_sibling = d_next;
  pms.target_probs = d_tp;
  pms.draft_probs = d_dp;
  pms.accept_index = d_acc_idx;
  pms.accept_length = d_acc_len;
  pms.output_token_ids = d_out;
  pms.batch_size = B;
  pms.num_nodes = N;
  pms.max_depth = D;
  pms.vocab_size = V;
  pms.rng.global_seed = 1;
  pms.rng.stream_id = 1;

  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  std::vector<double> hist(V, 0.0);
  for (int it = 0; it < launches; ++it) {
    pms.rng.global_seed = 300000u + static_cast<uint64_t>(it);
    KOPS_EXPECT_OK(kops::sample::tree_verify(pms, ctx), "tree_verify probabilistic");
    KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const auto out = kops_test::copy_out(d_out, B * (D + 1));
    for (int b = 0; b < B; ++b) {
      const int32_t id = out[static_cast<size_t>(b * (D + 1))];
      KOPS_TEST_CHECK(id >= 0 && id < V);
      if (id >= 0 && id < V) hist[static_cast<size_t>(id)] += 1.0;
    }
  }
  std::vector<double> expected(V);
  for (int v = 0; v < V; ++v) expected[static_cast<size_t>(v)] = p[static_cast<size_t>(v)] * B * launches;
  int df = 0;
  const double chi = chi_square(hist, expected, &df);
  KOPS_TEST_CHECK(df >= 1);
  KOPS_TEST_CHECK(chi < 80.0);

  kops_test::device_free(d_tp);
  kops_test::device_free(d_dp);
  kops_test::device_free(d_cand);
  kops_test::device_free(d_first);
  kops_test::device_free(d_next);
  kops_test::device_free(d_ridx);
  kops_test::device_free(d_acc_idx);
  kops_test::device_free(d_acc_len);
  kops_test::device_free(d_out);
}

void test_spec_tree_contracts() {
  constexpr int B = 1, N = 3, D = 1, V = 4;
  int32_t* d_cand = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_first = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_next = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_argmax = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_acc_idx = kops_test::device_alloc<int32_t>(B * (D + 1));
  int32_t* d_acc_len = kops_test::device_alloc<int32_t>(B);
  int32_t* d_out = kops_test::device_alloc<int32_t>(B * (D + 1));
  if (d_cand == nullptr || d_first == nullptr || d_next == nullptr || d_argmax == nullptr ||
      d_acc_idx == nullptr || d_acc_len == nullptr || d_out == nullptr) return;
  kops_test::copy_in(d_cand, std::vector<int32_t>{0, 1, 2});
  kops_test::copy_in(d_first, std::vector<int32_t>{1, -1, -1});
  kops_test::copy_in(d_next, std::vector<int32_t>{-1, 2, -1});
  kops_test::copy_in(d_argmax, std::vector<int32_t>{1, 2, 3});
  kops::sample::TreeVerifyParams p{};
  p.candidates = d_cand;
  p.first_child = d_first;
  p.next_sibling = d_next;
  p.accept_index = d_acc_idx;
  p.accept_length = d_acc_len;
  p.output_token_ids = d_out;
  p.batch_size = B;
  p.num_nodes = N;
  p.max_depth = D;
  p.vocab_size = V;
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_STATUS(kops::sample::tree_verify(p, ctx), kops::Status::kInvalidArgument,
                     "tree needs a mode");
  auto bad = p;
  bad.target_argmax = d_argmax;
  bad.num_nodes = 0;
  KOPS_EXPECT_STATUS(kops::sample::tree_verify(bad, ctx), kops::Status::kInvalidArgument,
                     "tree zero nodes");
  bad = p;
  bad.target_argmax = d_argmax;
  bad.max_depth = N;
  KOPS_EXPECT_STATUS(kops::sample::tree_verify(bad, ctx), kops::Status::kInvalidArgument,
                     "tree max depth");
  bad = p;
  bad.target_argmax = d_argmax;
  bad.candidates = nullptr;
  KOPS_EXPECT_STATUS(kops::sample::tree_verify(bad, ctx), kops::Status::kInvalidArgument,
                     "tree null candidates");
  kops::sample::TreeVerifyParams empty{};
  empty.batch_size = 0;
  empty.num_nodes = N;
  empty.max_depth = D;
  empty.vocab_size = V;
  empty.target_argmax = d_argmax;
  KOPS_EXPECT_OK(kops::sample::tree_verify(empty, ctx), "tree empty batch");
  kops_test::device_free(d_cand);
  kops_test::device_free(d_first);
  kops_test::device_free(d_next);
  kops_test::device_free(d_argmax);
  kops_test::device_free(d_acc_idx);
  kops_test::device_free(d_acc_len);
  kops_test::device_free(d_out);
}

void test_spec_tree_mask() {
  constexpr int B = 2, N = 4;
  const int32_t parent[2][4] = {{-1, 0, 0, 2}, {-1, 0, 1, 2}};
  const int32_t prefix[2] = {5, 3};
  uint8_t* d_mask = kops_test::device_alloc<uint8_t>(16);
  int64_t* d_indptr = kops_test::device_alloc<int64_t>(B + 1);
  int32_t* d_pos = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_parent = kops_test::device_alloc<int32_t>(B * N);
  int32_t* d_prefix = kops_test::device_alloc<int32_t>(B);
  if (d_mask == nullptr || d_indptr == nullptr || d_pos == nullptr || d_parent == nullptr ||
      d_prefix == nullptr) return;
  std::vector<int32_t> pv, pfx;
  for (int b = 0; b < B; ++b) {
    pv.insert(pv.end(), parent[b], parent[b] + N);
    pfx.push_back(prefix[b]);
  }
  kops_test::copy_in(d_parent, pv);
  kops_test::copy_in(d_prefix, pfx);
  kops_test::copy_in(d_mask, std::vector<uint8_t>(16, 0xA5));
  kops_test::copy_in(d_indptr, std::vector<int64_t>(B + 1, -1));
  kops_test::copy_in(d_pos, std::vector<int32_t>(B * N, -1));

  const Tensor tm = tensor1d(d_mask, DType::kUInt8, 16);
  const Tensor ti = tensor1d(d_indptr, DType::kInt64, B + 1);
  const Tensor tp = kops_test::tensor2d(d_pos, DType::kInt32, B, N);
  const Tensor tpar = kops_test::tensor2d(d_parent, DType::kInt32, B, N);
  const Tensor tpre = tensor1d(d_prefix, DType::kInt32, B);
  kops::LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  KOPS_EXPECT_OK(kops::sample::build_tree_attention_mask(tm, ti, tp, tpar, tpre, ctx),
                 "build_tree_attention_mask");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<uint8_t> want(16, 0);
  int widths[2];
  int offset[2] = {0, 0};
  for (int b = 0; b < B; ++b) {
    widths[b] = prefix[b] + N;
    if (b > 0) offset[b] = offset[b - 1] + N * widths[b - 1];
    for (int node = 0; node < N; ++node) {
      for (int key = 0; key < widths[b]; ++key) {
        bool vis = key < prefix[b];
        if (!vis) {
          int t = key - prefix[b];
          int cur = node;
          int guard = 0;
          while (cur >= 0 && cur < N && guard++ <= N) {
            if (cur == t) {
              vis = true;
              break;
            }
            cur = parent[b][cur];
          }
        }
        if (vis) {
          const int bit = offset[b] + node * widths[b] + key;
          want[static_cast<size_t>(bit >> 3)] |= static_cast<uint8_t>(1u << (bit & 7));
        }
      }
    }
  }
  const auto got = kops_test::copy_out(d_mask, 16);
  bool same = true;
  for (int i = 0; i < 16; ++i) same = same && (got[static_cast<size_t>(i)] == want[static_cast<size_t>(i)]);
  KOPS_TEST_CHECK(same);
  expect_i32(kops_test::copy_out(d_pos, B * N), std::vector<int32_t>{0, 1, 1, 2, 0, 1, 2, 3},
             "tree mask depth");
  const auto indptr = kops_test::copy_out(d_indptr, B + 1);
  KOPS_TEST_CHECK(indptr[0] == 0 && indptr[1] == 36 && indptr[2] == 64);

  // Contract checks.
  const Tensor bad_dtype = tensor1d(d_mask, DType::kInt32, 4);
  KOPS_EXPECT_STATUS(kops::sample::build_tree_attention_mask(bad_dtype, ti, tp, tpar, tpre, ctx),
                     kops::Status::kUnsupportedDtype, "mask dtype");
  const Tensor bad_indptr = tensor1d(d_indptr, DType::kInt64, B);
  KOPS_EXPECT_STATUS(kops::sample::build_tree_attention_mask(tm, bad_indptr, tp, tpar, tpre, ctx),
                     kops::Status::kInvalidArgument, "mask indptr length");
  const Tensor bad_pos = kops_test::tensor2d(d_pos, DType::kInt32, B, N - 1);
  KOPS_EXPECT_STATUS(kops::sample::build_tree_attention_mask(tm, ti, bad_pos, tpar, tpre, ctx),
                     kops::Status::kInvalidArgument, "mask position shape");
  const Tensor bad_pre = tensor1d(d_prefix, DType::kInt32, B - 1);
  KOPS_EXPECT_STATUS(kops::sample::build_tree_attention_mask(tm, ti, tp, tpar, bad_pre, ctx),
                     kops::Status::kInvalidArgument, "mask prefix length");

  kops_test::device_free(d_mask);
  kops_test::device_free(d_indptr);
  kops_test::device_free(d_pos);
  kops_test::device_free(d_parent);
  kops_test::device_free(d_prefix);
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;

  test_softmax_fp32();
  test_softmax_half<__half>(DType::kFloat16, "softmax fp16");
  test_softmax_half<__nv_bfloat16>(DType::kBFloat16, "softmax bf16");
  test_vectorized_paths<float>(DType::kFloat32, "softmax vectorized fp32");
  test_vectorized_paths<__nv_bfloat16>(DType::kBFloat16, "softmax vectorized bf16");
  test_softmax_split();
  test_argmax_fp32();
  test_argmax_bf16();
  test_compute_logprobs<float>(DType::kFloat32, "compute_logprobs fp32", 5, 1e-5, 1e-4);
  test_compute_logprobs<__nv_bfloat16>(DType::kBFloat16, "compute_logprobs bf16", 3, 1e-4, 1e-4);
  test_token_bincount();
  test_penalties<float>(DType::kFloat32, "apply_penalties fp32", 1e-6, 1e-6);
  test_penalties<__nv_bfloat16>(DType::kBFloat16, "apply_penalties bf16", 2e-2, 2e-2);
  test_bitmask<float>(DType::kFloat32, "apply_token_bitmask fp32");
  test_bitmask<__nv_bfloat16>(DType::kBFloat16, "apply_token_bitmask bf16");
  test_bitmask_identity_rows();
  test_logit_bias<float>(DType::kFloat32, "apply_logit_bias fp32", 1e-6, 1e-6);
  test_logit_bias<__nv_bfloat16>(DType::kBFloat16, "apply_logit_bias bf16", 2e-2, 2e-2);
  test_contracts();
  test_truncation_probs();
  test_top_k_mask_16bit<__nv_bfloat16>(DType::kBFloat16, "top_k_mask bf16");
  test_top_k_mask_16bit<__half>(DType::kFloat16, "top_k_mask fp16");
  test_top_k_renorm_ties();
  test_fused_unfiltered();
  test_fused_greedy_edges<float>("fp32");
  test_fused_greedy_edges<__half>("fp16");
  test_fused_greedy_edges<__nv_bfloat16>("bf16");
  test_fused_processors();
  test_fused_categorical<__nv_bfloat16>("categorical bf16 vec", 4096, 1024, 64);
  test_fused_categorical<__half>("categorical fp16 vec", 4096, 1024, 64);
  test_fused_categorical<float>("categorical fp32 scalar", 4095, 1024, 64);
  test_fused_categorical<float>("categorical fp32 grouped records", 40001, 128, 64);
  test_fused_topk_minp();
  test_fused_topp();
  test_fused_split();
  test_fused_split_batches();
  test_fused_split_reuse();
  test_sampling_distribution();
  test_gumbel_topk();
  test_sample_fast_paths();
  test_spec_chain_greedy();
  test_spec_chain_distribution();
  test_spec_chain_probabilistic();
  test_spec_acceptance_length();
  test_spec_chain_contracts();
  test_spec_chain_graph();
  test_spec_tree_greedy();
  test_spec_tree_distribution();
  test_spec_tree_probabilistic();
  test_spec_tree_contracts();
  test_spec_tree_mask();
  return kops_test::finish("test_sampling");
}
