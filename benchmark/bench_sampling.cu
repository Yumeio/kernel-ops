// =============================================================================
// kernel-ops | benchmark/bench_sampling.cu
// B1 sampling / logits-processor cases. rows x cols means [batch, vocab] for
// S01 and S02 alike; --dump-cases exposes the table to benchmark/compare.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "csrc/common/op_types.cuh"
#include "csrc/common/rng.cuh"
#include "csrc/sampling/sampling_kernels.cuh"
#include "csrc/sampling/fused_sampler_kernels.cuh"
#include "csrc/sampling/logits_processor_kernels.cuh"
#include "csrc/sampling/speculative_kernels.cuh"
#include "csrc/sampling/ops.h"

namespace {

using namespace kops_bench;

// Benchmarks must not run on all-zero data: every element ties, so selection, sort and routing
// kernels take their degenerate path. Values come from a hashed table of N(0, 1) samples.
const std::vector<float>& normal_table() {
  static const std::vector<float> table = [] {
    std::vector<float> v(size_t{1} << 20);
    std::mt19937 gen(1234u);
    std::normal_distribution<float> dist;
    for (auto& x : v) x = dist(gen);
    return v;
  }();
  return table;
}
inline float rand_normal(uint64_t i) {
  return normal_table()[static_cast<size_t>(((i + 1) * 0x9E3779B97F4A7C15ull) >> 44)];
}

inline void put(float& d, float v) { d = v; }
inline void put(__half& d, float v) { d = __float2half(v); }
inline void put(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

/// logits ~ N(0, sigma^2), the spread of a trained LM head.
template <typename T>
void fill_logits(T* dev, size_t n, float sigma) {
  std::vector<T> h(n);
  for (size_t i = 0; i < n; ++i) put(h[i], sigma * rand_normal(i));
  cudaMemcpy(dev, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
}

/// probabilities = row softmax of such logits.
void fill_probs(float* dev, size_t rows, size_t cols, float sigma) {
  std::vector<float> h(rows * cols);
  for (size_t r = 0; r < rows; ++r) {
    double sum = 0.0;
    for (size_t c = 0; c < cols; ++c) {
      const float e = std::exp(sigma * rand_normal(r * cols + c));
      h[r * cols + c] = e;
      sum += e;
    }
    for (size_t c = 0; c < cols; ++c) h[r * cols + c] = static_cast<float>(h[r * cols + c] / sum);
  }
  cudaMemcpy(dev, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
}

constexpr int kWarmup = 10;
constexpr int kIters = 80;

/// Mode token of a matrix case id: "sampling.<path>.<mode>.<dtype>.v<V>.b<B>".
std::string case_mode(const std::string& id) {
  const size_t a = id.find('.', 9) + 1;  // skip "sampling.<path>."
  const size_t dt = id.find(".bf16") != std::string::npos ? id.find(".bf16") : id.find(".fp32");
  return id.substr(a, dt - a);
}

/// Optional ".s<N>" suffix on a fused case id (the split sweep); 0 means "let the plan decide".
int case_splits(const std::string& id) {
  const size_t s = id.rfind(".s");
  if (s == std::string::npos || s + 2 >= id.size()) return 0;
  for (size_t i = s + 2; i < id.size(); ++i) {
    if (id[i] < '0' || id[i] > '9') return 0;
  }
  return std::atoi(id.c_str() + s + 2);
}

bool fused_supports(const std::string& mode) {
  (void)mode;  // every matrix mode runs on the fused path
  return true;
}

/// Split-row sweep: the same case at a forced split count; an invalid count is refused by the API.
/// Batch 1 is the shape the raised cap exists for (its auto plan now resolves above the old 16-way
/// ceiling), so the sweep carries it with the same 1..32 range.
void add_split_cases(std::vector<Case>* cases) {
  for (int64_t b : {1, 32, 64}) {
    for (const char* mode : {"k50", "p0.95"}) {
      for (int s : {1, 2, 4, 8, 16, 32}) {
        char id[160];
        std::snprintf(id, sizeof(id), "sampling.fused.%s.bf16.v151936.b%lld.s%d", mode,
                      static_cast<long long>(b), s);
        cases->push_back({id, "sample_fused", kops::DType::kBFloat16, b, 151936,
                          b * 151936 * 2 + b * 16, 0, 10, 80});
      }
    }
  }
}

kops::sample::SamplingParams mode_params(const std::string& mode) {
  kops::sample::SamplingParams p{};
  p.default_temperature = mode == "greedy" ? 0.0f : 1.0f;
  p.default_top_k = (mode == "k50" || mode == "k50p0.9") ? 50 : 0;
  p.default_top_p = mode == "k50p0.9" ? 0.9f
                    : mode == "p0.95" ? 0.95f
                    : mode == "p0.5" ? 0.5f
                    : mode == "p0.99" ? 0.99f
                                      : 1.0f;
  p.default_min_p = mode == "minp0.05" ? 0.05f : 0.0f;
  p.rng.global_seed = 7u;
  return p;
}

void add_case(std::vector<Case>* cases, const char* id, const char* op, kops::DType dtype,
              int64_t rows, int64_t cols, int64_t bytes) {
  cases->push_back({id, op, dtype, rows, cols, bytes, 0, kWarmup, kIters});
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto softmax_case = [&](const char* id, kops::DType dt, int64_t rows, int64_t cols) {
    const int es = dtype_size(dt);
    add_case(&cases, id, "softmax", dt, rows, cols, rows * cols * (es + 4) + rows * 4);
  };
  softmax_case("sampling.softmax.fp32.v131072.b32", kops::DType::kFloat32, 32, 131072);
  softmax_case("sampling.softmax.bf16.v131072.b32", kops::DType::kBFloat16, 32, 131072);
  softmax_case("sampling.softmax.fp16.v131072.b32", kops::DType::kFloat16, 32, 131072);
  softmax_case("sampling.softmax.bf16.v32768.b256", kops::DType::kBFloat16, 256, 32768);
  softmax_case("sampling.softmax.bf16.v262144.b1", kops::DType::kBFloat16, 1, 262144);
  softmax_case("sampling.softmax.fp32.v262144.b1", kops::DType::kFloat32, 1, 262144);
  softmax_case("sampling.softmax.bf16.v131072.b2", kops::DType::kBFloat16, 2, 131072);

  auto argmax_case = [&](const char* id, kops::DType dt, int64_t rows, int64_t cols) {
    const int es = dtype_size(dt);
    add_case(&cases, id, "argmax", dt, rows, cols, rows * cols * es + rows * 8);
  };
  argmax_case("sampling.argmax.fp32.v131072.b128", kops::DType::kFloat32, 128, 131072);
  argmax_case("sampling.argmax.bf16.v131072.b256", kops::DType::kBFloat16, 256, 131072);

  auto logprobs_case = [&](const char* id, kops::DType dt, int64_t rows, int64_t cols,
                           int64_t top_n) {
    const int es = dtype_size(dt);
    add_case(&cases, id, "compute_logprobs", dt, rows, cols,
             rows * cols * es + rows * (8 + top_n * 8));
  };
  logprobs_case("sampling.logprobs.bf16.v131072.b64.top5", kops::DType::kBFloat16, 64, 131072, 5);
  logprobs_case("sampling.logprobs.fp32.v32768.b128.top20", kops::DType::kFloat32, 128, 32768, 20);

  auto bincount_case = [&](const char* id, int64_t rows, int64_t cols, int64_t tokens) {
    const int64_t words = (cols + 31) / 32;
    // Touched: the ragged ids, the atomic count slots and prompt-mask words.
    add_case(&cases, id, "token_bincount", kops::DType::kInt32, rows, cols,
             tokens * 8 + rows * words * 4);
  };
  bincount_case("sampling.bincount.v32768.b256.t64", 256, 32768, 256 * 64);
  bincount_case("sampling.bincount.v131072.b32.t1024", 32, 131072, 32 * 1024);

  auto penalties_case = [&](const char* id, kops::DType dt, int64_t rows, int64_t cols) {
    const int es = dtype_size(dt);
    const int64_t words = (cols + 31) / 32;
    add_case(&cases, id, "apply_penalties", dt, rows, cols,
             rows * cols * (2 * es + 4) + rows * words * 4);
  };
  penalties_case("sampling.penalties.bf16.v131072.b128", kops::DType::kBFloat16, 128, 131072);
  penalties_case("sampling.penalties.fp32.v32768.b256", kops::DType::kFloat32, 256, 32768);

  auto bitmask_case = [&](const char* id, kops::DType dt, int64_t rows, int64_t cols) {
    const int es = dtype_size(dt);
    const int64_t words = (cols + 31) / 32;
    // The kernel only stores -inf into disallowed slots (all slots here).
    add_case(&cases, id, "apply_token_bitmask", dt, rows, cols,
             rows * cols * es + rows * words * 4);
  };
  bitmask_case("sampling.bitmask.bf16.v131072.b128", kops::DType::kBFloat16, 128, 131072);
  bitmask_case("sampling.bitmask.fp32.v32768.b256", kops::DType::kFloat32, 256, 32768);

  auto bias_case = [&](const char* id, kops::DType dt, int64_t rows, int64_t cols, int64_t nnz_row) {
    const int es = dtype_size(dt);
    // Touched: the sparse entries (read + write) and the row offsets.
    add_case(&cases, id, "apply_logit_bias", dt, rows, cols,
             rows * nnz_row * (2 * es + 8) + rows * 4);
  };
  bias_case("sampling.bias.bf16.v131072.b128.nnz8", kops::DType::kBFloat16, 128, 131072, 8);
  bias_case("sampling.bias.fp32.v32768.b256.nnz8", kops::DType::kFloat32, 256, 32768, 8);


  // Minimum DRAM traffic: one read of the row, plus one write when the op emits a full row
  // (renorm and mask). The multi-pass radix reads are not counted.
  auto sample_case = [&](const char* id, const char* op, kops::DType dt, int64_t rows, int64_t cols,
                         int64_t extra) {
    const int es = dtype_size(dt);
    const bool writes_row = std::string(op) == "topk_renorm" || std::string(op) == "topp_renorm" ||
                            std::string(op) == "topk_mask";
    add_case(&cases, id, op, dt, rows, cols, rows * cols * es * (writes_row ? 2 : 1) + rows * extra);
  };
  sample_case("sampling.topk_topp.bf16.v131072.b32.k50.p0.9", "sample", kops::DType::kBFloat16, 32,
              131072, 16);
  sample_case("sampling.topk_topp.bf16.v32768.b256.k20.p0.95", "sample", kops::DType::kBFloat16,
              256, 32768, 16);
  sample_case("sampling.gumbel.bf16.v131072.b32.k50", "gumbel", kops::DType::kBFloat16, 32, 131072,
              16);
  sample_case("sampling.topk_renorm.fp32.v131072.b128.k50", "topk_renorm", kops::DType::kFloat32,
              128, 131072, 16);
  sample_case("sampling.topp_renorm.fp32.v131072.b128.p0.9", "topp_renorm", kops::DType::kFloat32,
              128, 131072, 16);
  sample_case("sampling.topk_mask.bf16.v131072.b128.k50", "topk_mask", kops::DType::kBFloat16, 128,
              131072, 16);

  // sample() vs sample_fused() on the same rows (Qwen2.5 vocab). Bytes = one read of the logits,
  // so GB/s against the copy ceiling shows how many effective passes a path makes. The mode token
  // selects the parameters (mode_params); fused cases exist only for the modes it implements.
  static const char* const kModes[] = {"greedy", "t1", "k50", "k50p0.9", "p0.5", "p0.95", "p0.99",
                                       "minp0.05"};
  static const int64_t kBatches[] = {1, 32, 128, 512};
  static const int64_t kFusedOnlyBatches[] = {16, 64, 256};
  for (const char* mode : kModes) {
    for (int64_t b : kBatches) {
      for (const char* path : {"sample", "fused"}) {
        const bool fused = std::string(path) == "fused";
        if (fused && !fused_supports(mode)) continue;
        char id[128];
        std::snprintf(id, sizeof(id), "sampling.%s.%s.bf16.v151936.b%lld", path, mode,
                      static_cast<long long>(b));
        add_case(&cases, id, fused ? "sample_fused" : "sample_mode", kops::DType::kBFloat16, b,
                 151936, b * 151936 * 2 + b * 16);
      }
    }
    for (int64_t b : kFusedOnlyBatches) {
      char id[128];
      std::snprintf(id, sizeof(id), "sampling.fused.%s.bf16.v151936.b%lld", mode,
                    static_cast<long long>(b));
      add_case(&cases, id, "sample_fused", kops::DType::kBFloat16, b, 151936,
               b * 151936 * 2 + b * 16);
    }
  }
  // hpc-ops only accepts V=120832; keep a head-to-head subset on the same shape.
  for (const char* mode : {"greedy", "k50", "k50p0.9", "p0.95"}) {
    for (int64_t b : {32, 128, 512}) {
      char id[128];
      std::snprintf(id, sizeof(id), "sampling.fused.%s.bf16.v120832.b%lld", mode,
                    static_cast<long long>(b));
      add_case(&cases, id, "sample_fused", kops::DType::kBFloat16, b, 120832,
               b * 120832 * 2 + b * 16);
    }
  }
  for (const char* path : {"sample", "fused"}) {
    char id[128];
    std::snprintf(id, sizeof(id), "sampling.%s.t1.fp32.v151936.b128", path);
    add_case(&cases, id, std::string(path) == "fused" ? "sample_fused" : "sample_mode",
             kops::DType::kFloat32, 128, 151936, int64_t{128} * 151936 * 4 + 128 * 16);
  }

  // B10 speculative decoding.  bytes = the rows the verifier must inspect at
  // least once (target rows, plus draft rows for the p/q mode), matching the
  // dataflow the kernel and the eager baselines both pay.
  // The verifier stops at the first rejection, so the guaranteed traffic is one
  // target row per request (the recovery/bonus draw) plus the O(K) acceptance
  // probes.  Counting all K target rows would report a bandwidth above DRAM.
  cases.push_back({"spec.reject.chain.k8.v151936.b32", "spec_reject_chain",
                   kops::DType::kFloat32, 32, 151936,
                   int64_t{32} * 151936 * 4 + int64_t{32} * 9 * 8, 0, 5, 30});
  cases.push_back({"spec.reject.chain.k8.v32768.b128", "spec_reject_chain",
                   kops::DType::kFloat32, 128, 32768,
                   int64_t{128} * 32768 * 4 + int64_t{128} * 9 * 8, 0, 5, 30});
  cases.push_back({"spec.reject.target_only.k8.v151936.b32", "spec_reject_target",
                   kops::DType::kFloat32, 32, 151936,
                   int64_t{32} * 151936 * 4 + int64_t{32} * 9 * 8, 0, 5, 30});
  cases.push_back({"spec.reject.greedy.k8.v151936.b32", "spec_reject_greedy",
                   kops::DType::kFloat32, 32, 151936,
                   int64_t{32} * 9 * 8, 0, 10, 100});
  cases.push_back({"spec.reject.bonus.k0.v120832.b128", "spec_reject_bonus",
                   kops::DType::kFloat32, 128, 120832,
                   int64_t{32} * 120832 * 4, 0, 10, 100});
  cases.push_back({"spec.tree.q.n64.d8.v32000.b8", "spec_tree_q",
                   kops::DType::kFloat32, 8, 32000,
                   int64_t{8} * 32000 * 4 + int64_t{8} * 64 * 8, 0, 3, 20});
  cases.push_back({"spec.tree.target_only.n64.d8.v32000.b8", "spec_tree_target",
                   kops::DType::kFloat32, 8, 32000,
                   int64_t{8} * 32000 * 4 + int64_t{8} * 64 * 8, 0, 3, 20});
  cases.push_back({"spec.tree.greedy.n64.d8.v32000.b256", "spec_tree_greedy",
                   kops::DType::kFloat32, 256, 32000,
                   int64_t{8} * 64 * 4 * 3, 0, 10, 100});
  cases.push_back({"spec.mask.p2048.n64.b64", "spec_mask",
                   kops::DType::kFloat32, 64, 2048,
                   int64_t{64} * 64 * (2048 + 64) / 8 + int64_t{64} * 64 * 12, 0, 10, 100});
  return cases;
}

template <typename T>
bool run_softmax(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* logits = nullptr;
  float* probs = nullptr;
  float* temperature = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "malloc logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&probs), n * sizeof(float)), "malloc probs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temperature),
                          static_cast<size_t>(c.rows) * sizeof(float)), "malloc temperature")) {
    return false;
  }
  fill_logits(logits, n, 3.0f);
  cudaMemset(probs, 0, n * sizeof(float));
  std::vector<float> host_temperature(static_cast<size_t>(c.rows), 0.9f);
  cudaMemcpy(temperature, host_temperature.data(), host_temperature.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  const kops::Tensor t_logits = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_probs = tensor2d(probs, kops::DType::kFloat32, c.rows, c.cols);
  const kops::Tensor t_temperature = tensor1d(temperature, kops::DType::kFloat32, c.rows);
  kops::WorkspaceRequirements req{};
  if (!status_ok(kops::sample::sampling_workspace_requirements(
                     static_cast<int32_t>(c.rows), static_cast<int32_t>(c.cols), 0, &req),
                 "softmax workspace requirements")) {
    return false;
  }
  void* scratch = nullptr;
  if (req.device_scratch_bytes > 0 &&
      !cuda_ok(cudaMalloc(&scratch, req.device_scratch_bytes), "malloc softmax workspace")) {
    return false;
  }
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = req.device_scratch_bytes;

  bool ok = status_ok(kops::sample::softmax(t_probs, t_logits, t_temperature, ws, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(c,
                 [&] { (void)kops::sample::softmax(t_probs, t_logits, t_temperature, ws, ctx); },
                 st);
  }
  if (scratch != nullptr) cudaFree(scratch);
  cudaFree(logits);
  cudaFree(probs);
  cudaFree(temperature);
  return ok;
}

template <typename T>
bool run_argmax(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* logits = nullptr;
  int32_t* ids = nullptr;
  float* logprob = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "malloc logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ids), static_cast<size_t>(c.rows) * sizeof(int32_t)),
               "malloc ids") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logprob),
                          static_cast<size_t>(c.rows) * sizeof(float)), "malloc logprob")) {
    return false;
  }
  fill_logits(logits, n, 3.0f);

  const kops::Tensor t_logits = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_ids = tensor1d(ids, kops::DType::kInt32, c.rows);
  const kops::Tensor t_logprob = tensor1d(logprob, kops::DType::kFloat32, c.rows);

  bool ok = status_ok(kops::sample::argmax(t_ids, t_logprob, t_logits, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::sample::argmax(t_ids, t_logprob, t_logits, ctx); }, st);
  cudaFree(logits);
  cudaFree(ids);
  cudaFree(logprob);
  return ok;
}

template <typename T>
bool run_logprobs(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t top_n = c.id.find("top20") != std::string::npos ? 20 : 5;
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* logits = nullptr;
  int32_t* chosen = nullptr;
  float* chosen_logprob = nullptr;
  int32_t* rank = nullptr;
  int32_t* top_ids = nullptr;
  float* top_logprobs = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "malloc logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&chosen),
                          static_cast<size_t>(c.rows) * sizeof(int32_t)), "malloc chosen") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&chosen_logprob),
                          static_cast<size_t>(c.rows) * sizeof(float)), "malloc chosen logprob") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&rank),
                          static_cast<size_t>(c.rows) * sizeof(int32_t)), "malloc rank") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&top_ids),
                          static_cast<size_t>(c.rows * top_n) * sizeof(int32_t)), "malloc top ids") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&top_logprobs),
                          static_cast<size_t>(c.rows * top_n) * sizeof(float)),
               "malloc top logprobs")) {
    return false;
  }
  fill_logits(logits, n, 3.0f);
  cudaMemset(chosen, 0, static_cast<size_t>(c.rows) * sizeof(int32_t));

  const kops::Tensor t_logits = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_chosen = tensor1d(chosen, kops::DType::kInt32, c.rows);
  const kops::Tensor t_chosen_logprob = tensor1d(chosen_logprob, kops::DType::kFloat32, c.rows);
  const kops::Tensor t_rank = tensor1d(rank, kops::DType::kInt32, c.rows);
  const kops::Tensor t_top_ids = tensor2d(top_ids, kops::DType::kInt32, c.rows, top_n);
  const kops::Tensor t_top_logprobs = tensor2d(top_logprobs, kops::DType::kFloat32, c.rows, top_n);

  const auto call = [&] {
    return kops::sample::compute_logprobs(t_chosen_logprob, t_rank, t_top_ids, t_top_logprobs,
                                          t_logits, t_chosen, static_cast<int32_t>(top_n), ctx);
  };
  bool ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(logits);
  cudaFree(chosen);
  cudaFree(chosen_logprob);
  cudaFree(rank);
  cudaFree(top_ids);
  cudaFree(top_logprobs);
  return ok;
}

bool run_bincount(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t tokens = c.rows * 64;
  const int64_t words = (c.cols + 31) / 32;
  int32_t* counts = nullptr;
  uint32_t* mask = nullptr;
  int32_t* ids = nullptr;
  int32_t* indptr = nullptr;
  uint8_t* is_prompt = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&counts),
                          static_cast<size_t>(c.rows * c.cols) * sizeof(int32_t)),
               "malloc counts") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&mask),
                          static_cast<size_t>(c.rows * words) * sizeof(uint32_t)), "malloc mask") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ids),
                          static_cast<size_t>(tokens) * sizeof(int32_t)), "malloc ids") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&indptr),
                          static_cast<size_t>(c.rows + 1) * sizeof(int32_t)), "malloc indptr") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&is_prompt), static_cast<size_t>(tokens)),
               "malloc is_prompt")) {
    return false;
  }
  std::vector<int32_t> host_ids(static_cast<size_t>(tokens));
  std::vector<int32_t> host_indptr(static_cast<size_t>(c.rows + 1));
  std::vector<uint8_t> host_prompt(static_cast<size_t>(tokens), 0);
  for (int64_t r = 0; r <= c.rows; ++r) host_indptr[static_cast<size_t>(r)] = static_cast<int32_t>(r * 64);
  for (int64_t i = 0; i < tokens; ++i) host_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i % c.cols);
  cudaMemset(counts, 0, static_cast<size_t>(c.rows * c.cols) * sizeof(int32_t));
  cudaMemset(mask, 0, static_cast<size_t>(c.rows * words) * sizeof(uint32_t));
  cudaMemcpy(ids, host_ids.data(), host_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(indptr, host_indptr.data(), host_indptr.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(is_prompt, host_prompt.data(), host_prompt.size(), cudaMemcpyHostToDevice);

  const kops::Tensor t_counts = tensor2d(counts, kops::DType::kInt32, c.rows, c.cols);
  const kops::Tensor t_mask = tensor2d(mask, kops::DType::kUInt32, c.rows, words);
  const kops::Tensor t_ids = tensor1d(ids, kops::DType::kInt32, tokens);
  const kops::Tensor t_indptr = tensor1d(indptr, kops::DType::kInt32, c.rows + 1);
  const kops::Tensor t_prompt = tensor1d(is_prompt, kops::DType::kUInt8, tokens);

  const auto call = [&] { return kops::sample::token_bincount(t_counts, t_mask, t_ids, t_indptr, t_prompt, ctx); };
  bool ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(counts);
  cudaFree(mask);
  cudaFree(ids);
  cudaFree(indptr);
  cudaFree(is_prompt);
  return ok;
}

template <typename T>
bool run_penalties(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  const int64_t words = (c.cols + 31) / 32;
  T* logits = nullptr;
  int32_t* counts = nullptr;
  uint32_t* mask = nullptr;
  float* repetition = nullptr;
  float* frequency = nullptr;
  float* presence = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "malloc logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&counts), n * sizeof(int32_t)), "malloc counts") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&mask),
                          static_cast<size_t>(c.rows * words) * sizeof(uint32_t)), "malloc mask") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&repetition),
                          static_cast<size_t>(c.rows) * sizeof(float)), "malloc repetition") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&frequency),
                          static_cast<size_t>(c.rows) * sizeof(float)), "malloc frequency") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&presence),
                          static_cast<size_t>(c.rows) * sizeof(float)), "malloc presence")) {
    return false;
  }
  cudaMemset(logits, 0, n * sizeof(T));
  // 0x01 bytes: every count is 16843009 > 0, so the penalty branches run.
  cudaMemset(counts, 0x01, n * sizeof(int32_t));
  cudaMemset(mask, 0, static_cast<size_t>(c.rows * words) * sizeof(uint32_t));
  std::vector<float> host_rep(static_cast<size_t>(c.rows), 1.5f);
  std::vector<float> host_freq(static_cast<size_t>(c.rows), 0.1f);
  std::vector<float> host_pres(static_cast<size_t>(c.rows), 0.1f);
  cudaMemcpy(repetition, host_rep.data(), host_rep.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(frequency, host_freq.data(), host_freq.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(presence, host_pres.data(), host_pres.size() * sizeof(float), cudaMemcpyHostToDevice);

  const kops::Tensor t_logits = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_counts = tensor2d(counts, kops::DType::kInt32, c.rows, c.cols);
  const kops::Tensor t_mask = tensor2d(mask, kops::DType::kUInt32, c.rows, words);
  const kops::Tensor t_rep = tensor1d(repetition, kops::DType::kFloat32, c.rows);
  const kops::Tensor t_freq = tensor1d(frequency, kops::DType::kFloat32, c.rows);
  const kops::Tensor t_pres = tensor1d(presence, kops::DType::kFloat32, c.rows);

  const auto call = [&] {
    return kops::sample::apply_penalties(t_logits, t_counts, t_mask, t_rep, t_freq, t_pres, ctx);
  };
  bool ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(logits);
  cudaFree(counts);
  cudaFree(mask);
  cudaFree(repetition);
  cudaFree(frequency);
  cudaFree(presence);
  return ok;
}

template <typename T>
bool run_bitmask(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  const int64_t words = (c.cols + 31) / 32;
  T* logits = nullptr;
  int32_t* mask = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "malloc logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&mask),
                          static_cast<size_t>(c.rows * words) * sizeof(int32_t)), "malloc mask")) {
    return false;
  }
  cudaMemset(logits, 0, n * sizeof(T));
  cudaMemset(mask, 0, static_cast<size_t>(c.rows * words) * sizeof(int32_t));  // all disallowed

  const kops::Tensor t_logits = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_mask = tensor2d(mask, kops::DType::kInt32, c.rows, words);
  const auto call = [&] { return kops::sample::apply_token_bitmask(t_logits, t_mask, kops::Tensor{}, ctx); };
  bool ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(logits);
  cudaFree(mask);
  return ok;
}

template <typename T>
bool run_bias(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  const int64_t nnz_row = 8;
  const int64_t nnz = c.rows * nnz_row;
  T* logits = nullptr;
  int32_t* indptr = nullptr;
  int32_t* ids = nullptr;
  float* values = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "malloc logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&indptr),
                          static_cast<size_t>(c.rows + 1) * sizeof(int32_t)), "malloc indptr") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ids), static_cast<size_t>(nnz) * sizeof(int32_t)),
               "malloc ids") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&values), static_cast<size_t>(nnz) * sizeof(float)),
               "malloc values")) {
    return false;
  }
  std::vector<int32_t> host_indptr(static_cast<size_t>(c.rows + 1));
  std::vector<int32_t> host_ids(static_cast<size_t>(nnz));
  std::vector<float> host_values(static_cast<size_t>(nnz), 0.5f);
  for (int64_t r = 0; r <= c.rows; ++r) host_indptr[static_cast<size_t>(r)] = static_cast<int32_t>(r * nnz_row);
  for (int64_t r = 0; r < c.rows; ++r) {
    for (int64_t j = 0; j < nnz_row; ++j) {
      host_ids[static_cast<size_t>(r * nnz_row + j)] = static_cast<int32_t>((r * 977 + j * 131) % c.cols);
    }
  }
  cudaMemset(logits, 0, n * sizeof(T));
  cudaMemcpy(indptr, host_indptr.data(), host_indptr.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(ids, host_ids.data(), host_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(values, host_values.data(), host_values.size() * sizeof(float), cudaMemcpyHostToDevice);

  const kops::Tensor t_logits = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_indptr = tensor1d(indptr, kops::DType::kInt32, c.rows + 1);
  const kops::Tensor t_ids = tensor1d(ids, kops::DType::kInt32, nnz);
  const kops::Tensor t_values = tensor1d(values, kops::DType::kFloat32, nnz);
  const auto call = [&] { return kops::sample::apply_logit_bias(t_logits, t_indptr, t_ids, t_values, ctx); };
  bool ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(logits);
  cudaFree(indptr);
  cudaFree(ids);
  cudaFree(values);
  return ok;
}

template <typename T>
bool run_sample(const Case& c, const kops::LaunchContext& ctx, Stats* st);
template <typename T>
bool run_sample_matrix(const Case& c, const kops::LaunchContext& ctx, Stats* st);
bool run_renorm(const Case& c, const kops::LaunchContext& ctx, Stats* st);
template <typename T>
bool run_topk_mask(const Case& c, const kops::LaunchContext& ctx, Stats* st);
bool run_spec_reject(const Case& c, const kops::LaunchContext& ctx, Stats* st);
bool run_spec_tree(const Case& c, const kops::LaunchContext& ctx, Stats* st);
bool run_spec_mask(const Case& c, const kops::LaunchContext& ctx, Stats* st);

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op.rfind("spec_reject", 0) == 0) return run_spec_reject(c, ctx, st);
  if (c.op.rfind("spec_tree", 0) == 0) return run_spec_tree(c, ctx, st);
  if (c.op == "spec_mask") return run_spec_mask(c, ctx, st);
  if (c.op == "token_bincount") return run_bincount(c, ctx, st);
  if (c.op == "sample" || c.op == "gumbel") {
    if (c.dtype == kops::DType::kBFloat16) return run_sample<__nv_bfloat16>(c, ctx, st);
    if (c.dtype == kops::DType::kFloat16) return run_sample<__half>(c, ctx, st);
    return run_sample<float>(c, ctx, st);
  }
  if (c.op == "sample_mode" || c.op == "sample_fused") {
    if (c.dtype == kops::DType::kBFloat16) return run_sample_matrix<__nv_bfloat16>(c, ctx, st);
    return run_sample_matrix<float>(c, ctx, st);
  }
  if (c.op == "topk_renorm" || c.op == "topp_renorm") return run_renorm(c, ctx, st);
  if (c.op == "topk_mask") {
    if (c.dtype == kops::DType::kBFloat16) return run_topk_mask<__nv_bfloat16>(c, ctx, st);
    if (c.dtype == kops::DType::kFloat16) return run_topk_mask<__half>(c, ctx, st);
    return run_topk_mask<float>(c, ctx, st);
  }
  if (c.dtype == kops::DType::kFloat32) {
    if (c.op == "softmax") return run_softmax<float>(c, ctx, st);
    if (c.op == "argmax") return run_argmax<float>(c, ctx, st);
    if (c.op == "compute_logprobs") return run_logprobs<float>(c, ctx, st);
    if (c.op == "apply_penalties") return run_penalties<float>(c, ctx, st);
    if (c.op == "apply_token_bitmask") return run_bitmask<float>(c, ctx, st);
    if (c.op == "apply_logit_bias") return run_bias<float>(c, ctx, st);
  }
  if (c.dtype == kops::DType::kBFloat16) {
    if (c.op == "softmax") return run_softmax<__nv_bfloat16>(c, ctx, st);
    if (c.op == "argmax") return run_argmax<__nv_bfloat16>(c, ctx, st);
    if (c.op == "compute_logprobs") return run_logprobs<__nv_bfloat16>(c, ctx, st);
    if (c.op == "apply_penalties") return run_penalties<__nv_bfloat16>(c, ctx, st);
    if (c.op == "apply_token_bitmask") return run_bitmask<__nv_bfloat16>(c, ctx, st);
    if (c.op == "apply_logit_bias") return run_bias<__nv_bfloat16>(c, ctx, st);
  }
  if (c.dtype == kops::DType::kFloat16) {
    if (c.op == "softmax") return run_softmax<__half>(c, ctx, st);
  }
  std::fprintf(stderr, "[bench] %s: no runner for op '%s' dtype %s\n", c.id.c_str(), c.op.c_str(),
               dtype_label(c.dtype));
  return false;
}


template <typename T>
bool run_sample(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* logits = nullptr;
  int32_t* ids = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ids), static_cast<size_t>(c.rows) * 4), "ids")) {
    return false;
  }
  fill_logits(logits, n, 3.0f);
  const kops::Tensor t_logits = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_ids = tensor1d(ids, kops::DType::kInt32, c.rows);
  kops::sample::SamplingParams params{};
  params.default_temperature = 1.0f;
  const bool k20 = c.id.find("k20") != std::string::npos;
  const bool p95 = c.id.find("p0.95") != std::string::npos;
  params.default_top_k = c.op == "sample" ? (k20 ? 20 : 50) : 0;
  params.default_top_p = c.op == "sample" ? (p95 ? 0.95f : 0.9f) : 1.0f;
  params.default_min_p = 0.0f;
  params.rng.global_seed = 7u;
  kops::WorkspaceRequirements req{};
  if (!status_ok(kops::sample::sampling_workspace_requirements(
                     static_cast<int32_t>(c.rows), static_cast<int32_t>(c.cols), 0, &req),
                 "sample workspace")) {
    return false;
  }
  void* scratch = nullptr;
  if (!cuda_ok(cudaMalloc(&scratch, req.device_scratch_bytes), "sample scratch")) return false;
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = req.device_scratch_bytes;
  const bool gumbel = c.op == "gumbel";
  const auto call = [&] {
    return gumbel ? kops::sample::gumbel_max_sample(t_ids, t_logits, params, ctx)
                  : kops::sample::sample(t_ids, t_logits, params, ws, ctx);
  };
  const bool ok = status_ok(call(), c.id.c_str()) && measure(c, [&] { (void)call(); }, st);
  cudaFree(scratch);
  cudaFree(logits);
  cudaFree(ids);
  return ok;
}

template <typename T>
bool run_sample_matrix(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* logits = nullptr;
  int32_t* ids = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ids), static_cast<size_t>(c.rows) * 4), "ids")) {
    return false;
  }
  fill_logits(logits, n, 3.0f);
  const kops::Tensor t_logits = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_ids = tensor1d(ids, kops::DType::kInt32, c.rows);
  const kops::sample::SamplingParams params = mode_params(case_mode(c.id));
  const bool fused = c.op == "sample_fused";
  kops::sample::FusedSamplingParams fp{};
  fp.sampling = params;
  fp.default_repetition_penalty = 1.0f;
  kops::sample::SamplingPlan plan{};
  kops::WorkspaceRequirements req{};
  if (fused) {
    if (!status_ok(kops::sample::sampling_plan(t_logits, params, ctx, &plan), "sampling_plan")) {
      return false;
    }
    const int forced = case_splits(c.id);
    if (forced > 0) plan.splits = forced;
    if (!status_ok(kops::sample::sampling_fused_workspace_requirements(
                       static_cast<int32_t>(c.rows), static_cast<int32_t>(c.cols), plan, &req),
                   "fused workspace")) {
      return false;
    }
  } else if (!status_ok(kops::sample::sampling_workspace_requirements(
                            static_cast<int32_t>(c.rows), static_cast<int32_t>(c.cols), 0, &req),
                        "sample workspace")) {
    return false;
  }
  void* scratch = nullptr;
  if (req.device_scratch_bytes > 0 &&
      !cuda_ok(cudaMalloc(&scratch, req.device_scratch_bytes), "sample scratch")) {
    return false;
  }
  void* persistent = nullptr;
  if (req.device_persistent_bytes > 0 &&
      (!cuda_ok(cudaMalloc(&persistent, req.device_persistent_bytes), "sample persistent") ||
       !cuda_ok(cudaMemset(persistent, 0, req.device_persistent_bytes), "zero persistent"))) {
    return false;
  }
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = req.device_scratch_bytes;
  ws.device_persistent = persistent;
  ws.device_persistent_bytes = req.device_persistent_bytes;
  const auto call = [&] {
    return fused ? kops::sample::sample_fused(t_ids, t_logits, fp, plan, ws, ctx)
                 : kops::sample::sample(t_ids, t_logits, params, ws, ctx);
  };
  const kops::Status first = call();
  bool ok = true;
  if (first == kops::Status::kPlanInvalidated) {
    std::fprintf(stderr, "[bench] %s skipped: %s\n", c.id.c_str(), kops::last_error_message());
  } else {
    ok = status_ok(first, c.id.c_str()) && measure(c, [&] { (void)call(); }, st);
  }
  if (scratch != nullptr) cudaFree(scratch);
  if (persistent != nullptr) cudaFree(persistent);
  cudaFree(logits);
  cudaFree(ids);
  return ok;
}

bool run_renorm(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  float* probs = nullptr;
  float* out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&probs), n * 4), "probs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * 4), "out")) {
    return false;
  }
  fill_probs(probs, static_cast<size_t>(c.rows), static_cast<size_t>(c.cols), 3.0f);
  const kops::Tensor t_p = tensor2d(probs, kops::DType::kFloat32, c.rows, c.cols);
  const kops::Tensor t_o = tensor2d(out, kops::DType::kFloat32, c.rows, c.cols);
  bool ok = false;
  if (c.op == "topk_renorm") {
    const auto call = [&] { return kops::sample::top_k_renorm_probs(t_o, t_p, kops::Tensor{}, 50, ctx); };
    ok = status_ok(call(), c.id.c_str()) && measure(c, [&] { (void)call(); }, st);
  } else {
    const auto call = [&] { return kops::sample::top_p_renorm_probs(t_o, t_p, kops::Tensor{}, 0.9f, ctx); };
    ok = status_ok(call(), c.id.c_str()) && measure(c, [&] { (void)call(); }, st);
  }
  cudaFree(probs);
  cudaFree(out);
  return ok;
}

template <typename T>
bool run_topk_mask(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* logits = nullptr;
  T* out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&logits), n * sizeof(T)), "logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(T)), "out")) {
    return false;
  }
  fill_logits(logits, n, 3.0f);
  const kops::Tensor t_l = tensor2d(logits, c.dtype, c.rows, c.cols);
  const kops::Tensor t_o = tensor2d(out, c.dtype, c.rows, c.cols);
  const auto call = [&] { return kops::sample::top_k_mask_logits(t_o, t_l, kops::Tensor{}, 50, ctx); };
  const bool ok = status_ok(call(), c.id.c_str()) && measure(c, [&] { (void)call(); }, st);
  cudaFree(logits);
  cudaFree(out);
  return ok;
}

// ---- B10 speculative decoding ---------------------------------------------------------------
template <typename T>
void upload(T* dst, const std::vector<T>& src) {
  if (!src.empty()) {
    cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
}

int spec_int(const std::string& id, const char* key, int def) {
  const std::string marker = std::string(".") + key;
  const size_t pos = id.find(marker);
  if (pos == std::string::npos) return def;
  return std::atoi(id.c_str() + pos + marker.size());
}

bool run_spec_reject(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int B = static_cast<int>(c.rows);
  const int V = static_cast<int>(c.cols);
  const int K = spec_int(c.id, "k", 0);
  const bool greedy = c.op == "spec_reject_greedy";
  const bool target_only = c.op == "spec_reject_target" || c.op == "spec_reject_bonus";
  const int total = B * K;
  const int target_rows = total + B;
  float* tp = nullptr;
  float* dp = nullptr;
  int32_t* cu = nullptr;
  int32_t* draft = nullptr;
  int32_t* argmax = nullptr;
  int32_t* out = nullptr;
  int32_t* num = nullptr;
  uint8_t* flags = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&tp),
                          static_cast<size_t>(target_rows) * V * sizeof(float)), "spec target") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cu), (B + 1) * sizeof(int32_t)), "spec cu") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&draft),
                          std::max(1, total) * sizeof(int32_t)), "spec draft") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), B * (K + 1) * sizeof(int32_t)),
               "spec out") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&num), B * sizeof(int32_t)), "spec num")) {
    return false;
  }
  if (!target_only && !greedy) {
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dp),
                            std::max(1, total) * V * sizeof(float)), "spec draft probs")) {
      return false;
    }
  }
  if (greedy) {
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&argmax),
                            target_rows * sizeof(int32_t)), "spec argmax") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&flags), B * sizeof(uint8_t)), "spec flags")) {
      return false;
    }
  }
  fill_probs(tp, static_cast<size_t>(target_rows), V, 3.0f);
  if (dp != nullptr) fill_probs(dp, static_cast<size_t>(total), V, 2.5f);
  std::vector<int32_t> hcu(B + 1), hdraft(std::max(1, total)), hargmax(target_rows, 0);
  for (int b = 0; b <= B; ++b) hcu[static_cast<size_t>(b)] = b * K;
  for (int i = 0; i < total; ++i) hdraft[static_cast<size_t>(i)] = (i * 7919 + 13) % V;
  for (int i = 0; i < total; ++i) hargmax[static_cast<size_t>(i)] = hdraft[static_cast<size_t>(i)];
  upload(cu, hcu);
  if (total > 0) upload(draft, hdraft);
  if (greedy) {
    upload(argmax, hargmax);
    upload(flags, std::vector<uint8_t>(B, 1));
  }
  cudaMemset(out, 0, B * (K + 1) * sizeof(int32_t));
  cudaMemset(num, 0, B * sizeof(int32_t));

  kops::sample::SpecVerifyParams p{};
  p.draft_token_ids = draft;
  p.cu_num_draft = cu;
  p.target_probs = tp;
  p.draft_probs = dp;
  p.target_argmax = greedy ? argmax : nullptr;
  p.is_greedy = greedy ? flags : nullptr;
  p.output_token_ids = out;
  p.num_accepted = num;
  p.batch_size = B;
  p.max_draft = K;
  p.vocab_size = V;
  p.rng.global_seed = 7;
  p.rng.stream_id = 1;
  const bool ok = status_ok(kops::sample::rejection_sample(p, ctx), c.id.c_str()) &&
                  measure(c, [&] { (void)kops::sample::rejection_sample(p, ctx); }, st);
  cudaFree(tp);
  if (dp != nullptr) cudaFree(dp);
  cudaFree(cu);
  cudaFree(draft);
  if (argmax != nullptr) cudaFree(argmax);
  if (flags != nullptr) cudaFree(flags);
  cudaFree(out);
  cudaFree(num);
  return ok;
}

bool run_spec_tree(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int B = static_cast<int>(c.rows);
  const int V = static_cast<int>(c.cols);
  const int N = spec_int(c.id, "n", 64);
  const int D = spec_int(c.id, "d", 8);
  const bool greedy = c.op == "spec_tree_greedy";
  const bool qmode = c.op == "spec_tree_q";
  float* tp = nullptr;
  float* dp = nullptr;
  int32_t* cand = nullptr;
  int32_t* first = nullptr;
  int32_t* next = nullptr;
  int32_t* ridx = nullptr;
  int32_t* argmax = nullptr;
  int32_t* acc_idx = nullptr;
  int32_t* acc_len = nullptr;
  int32_t* out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&tp),
                          static_cast<size_t>(B) * N * V * sizeof(float)), "tree target") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cand), B * N * sizeof(int32_t)), "tree cand") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&first), B * N * sizeof(int32_t)), "tree first") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&next), B * N * sizeof(int32_t)), "tree next") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ridx), B * N * sizeof(int32_t)), "tree ridx") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&acc_idx),
                          B * (D + 1) * sizeof(int32_t)), "tree index") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&acc_len), B * sizeof(int32_t)), "tree len") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out),
                          B * (D + 1) * sizeof(int32_t)), "tree out")) {
    return false;
  }
  if (qmode) {
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dp),
                            static_cast<size_t>(B) * N * V * sizeof(float)), "tree draft")) {
      return false;
    }
  }
  if (greedy) {
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&argmax),
                            B * N * sizeof(int32_t)), "tree argmax")) {
      return false;
    }
  }
  fill_probs(tp, static_cast<size_t>(B) * N, V, 3.0f);
  if (dp != nullptr) fill_probs(dp, static_cast<size_t>(B) * N, V, 2.5f);
  std::vector<int32_t> cand1(N), first1(N, -1), next1(N, -1), ridx1(N), argmax1(N);
  for (int i = 0; i < N; ++i) {
    cand1[static_cast<size_t>(i)] = (i * 13 + 1) % V;
    ridx1[static_cast<size_t>(i)] = i;
    if (2 * i + 1 < N) {
      first1[static_cast<size_t>(i)] = 2 * i + 1;
      if (2 * i + 2 < N) next1[static_cast<size_t>(2 * i + 1)] = 2 * i + 2;
    }
    argmax1[static_cast<size_t>(i)] = first1[static_cast<size_t>(i)] >= 0
                                          ? cand1[static_cast<size_t>(first1[static_cast<size_t>(i)])]
                                          : 0;
  }
  std::vector<int32_t> hc, hf, hn, hr, ha;
  for (int b = 0; b < B; ++b) {
    hc.insert(hc.end(), cand1.begin(), cand1.end());
    hf.insert(hf.end(), first1.begin(), first1.end());
    hn.insert(hn.end(), next1.begin(), next1.end());
    hr.insert(hr.end(), ridx1.begin(), ridx1.end());
    ha.insert(ha.end(), argmax1.begin(), argmax1.end());
  }
  upload(cand, hc);
  upload(first, hf);
  upload(next, hn);
  upload(ridx, hr);
  if (greedy) upload(argmax, ha);
  cudaMemset(out, 0, B * (D + 1) * sizeof(int32_t));
  cudaMemset(acc_len, 0, B * sizeof(int32_t));

  kops::sample::TreeVerifyParams p{};
  p.candidates = cand;
  p.retrieve_index = greedy ? nullptr : ridx;
  p.first_child = first;
  p.next_sibling = next;
  p.target_probs = greedy ? nullptr : tp;
  p.draft_probs = qmode ? dp : nullptr;
  p.target_argmax = greedy ? argmax : nullptr;
  p.accept_index = acc_idx;
  p.accept_length = acc_len;
  p.output_token_ids = out;
  p.batch_size = B;
  p.num_nodes = N;
  p.max_depth = D;
  p.vocab_size = V;
  p.rng.global_seed = 7;
  p.rng.stream_id = 1;
  const bool ok = status_ok(kops::sample::tree_verify(p, ctx), c.id.c_str()) &&
                  measure(c, [&] { (void)kops::sample::tree_verify(p, ctx); }, st);
  cudaFree(tp);
  if (dp != nullptr) cudaFree(dp);
  cudaFree(cand);
  cudaFree(first);
  cudaFree(next);
  cudaFree(ridx);
  if (argmax != nullptr) cudaFree(argmax);
  cudaFree(acc_idx);
  cudaFree(acc_len);
  cudaFree(out);
  return ok;
}

bool run_spec_mask(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int B = static_cast<int>(c.rows);
  const int P = static_cast<int>(c.cols);
  const int N = spec_int(c.id, "n", 64);
  const int64_t bits = static_cast<int64_t>(B) * N * (P + N);
  const int64_t bytes = (bits + 7) / 8 + 16;
  uint8_t* mask = nullptr;
  int64_t* indptr = nullptr;
  int32_t* pos = nullptr;
  int32_t* parent = nullptr;
  int32_t* prefix = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&mask), bytes), "mask") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&indptr), (B + 1) * sizeof(int64_t)),
               "mask indptr") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pos), B * N * sizeof(int32_t)), "mask pos") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&parent), B * N * sizeof(int32_t)),
               "mask parent") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&prefix), B * sizeof(int32_t)),
               "mask prefix")) {
    return false;
  }
  std::vector<int32_t> hparent, hprefix(B, P);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < N; ++i) hparent.push_back(i == 0 ? -1 : (i - 1) / 2);
  }
  upload(parent, hparent);
  upload(prefix, hprefix);
  const kops::Tensor tm = tensor1d(mask, kops::DType::kUInt8, bytes);
  const kops::Tensor ti = tensor1d(indptr, kops::DType::kInt64, B + 1);
  const kops::Tensor tp = tensor2d(pos, kops::DType::kInt32, B, N);
  const kops::Tensor tpar = tensor2d(parent, kops::DType::kInt32, B, N);
  const kops::Tensor tpre = tensor1d(prefix, kops::DType::kInt32, B);
  const bool ok = status_ok(kops::sample::build_tree_attention_mask(tm, ti, tp, tpar, tpre, ctx),
                            c.id.c_str()) &&
                  measure(c,
                          [&] {
                            (void)kops::sample::build_tree_attention_mask(tm, ti, tp, tpar, tpre,
                                                                         ctx);
                          },
                          st);
  cudaFree(mask);
  cudaFree(indptr);
  cudaFree(pos);
  cudaFree(parent);
  cudaFree(prefix);
  return ok;
}
}

int main(int argc, char** argv) {
  using namespace kops_bench;

  const Options opt = parse_options(argc, argv);
  if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice")) return 1;

  DeviceInfo dev{};
  query_device(&dev);

  std::vector<Case> cases = make_cases();
  if (opt.split_sweep) add_split_cases(&cases);
  filter_cases(&cases, opt);
  if (cases.empty()) {
    std::fprintf(stderr, "[bench] no case matches the --only/--skip filters\n");
    return 1;
  }
  if (opt.dump_cases) {
    dump_cases(cases, dev);
    return 0;
  }

  kops::LaunchContext ctx{};
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) return 1;

  std::vector<Result> results;
  results.reserve(cases.size());
  for (const Case& c : cases) {
    Stats st{};
    if (!run_case(c, ctx, &st)) {
      std::fprintf(stderr, "[bench] %s failed\n", c.id.c_str());
      return 1;
    }
    results.push_back({&c, st});
  }

  print_table(results, dev);
  if (opt.json_path != nullptr) write_results(opt.json_path, results, dev);
  return 0;
}
