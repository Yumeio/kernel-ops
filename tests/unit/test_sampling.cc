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
#include <limits>
#include <random>
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
  const Tensor t_temp = tensor1d(d_temp, DType::kFloat32, batch);
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
  test_sampling_distribution();
  return kops_test::finish("test_sampling");
}
