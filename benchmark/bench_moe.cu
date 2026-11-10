// =============================================================================
// kernel-ops | benchmark/bench_moe.cu
// B6 routing / metadata / permute / reduce cases plus the control-chain case.
// Rows/cols are (tokens, experts) unless the case id says otherwise. --dump-cases
// exposes the table to benchmark/compare/torch_baseline.py.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "csrc/common/quant.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/moe/permute_kernels.cuh"
#include "csrc/moe/fused_moe_kernels.cuh"
#include "csrc/moe/ops.h"

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

void fill_f32(float* dev, size_t n, float sigma) {
  std::vector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = sigma * rand_normal(i);
  cudaMemcpy(dev, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}
void fill_bf16(void* dev, size_t n, float sigma) {
  std::vector<__nv_bfloat16> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = __float2bfloat16(sigma * rand_normal(i));
  cudaMemcpy(dev, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
}
using kops::Tensor;

constexpr int kWarmup = 10;
constexpr int kIters = 100;

void add_case(std::vector<Case>* cases, const char* id, const char* op, kops::DType dtype,
              int64_t rows, int64_t cols, int64_t bytes) {
  cases->push_back({id, op, dtype, rows, cols, bytes, 0, kWarmup, kIters});
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto route = [&](const char* id, int64_t tokens, int64_t experts, int k) {
    add_case(&cases, id, "moe_route", kops::DType::kFloat32, tokens, experts,
             tokens * experts * 4 + tokens * k * 8);
  };
  route("moe.route.sigmoid.e256.k8.t4096", 4096, 256, 8);
  add_case(&cases, "moe.grouped.sigmoid.e256.g8.tg4.k8.t2048", "moe_grouped", kops::DType::kFloat32,
           2048, 256, 2048 * 256 * 4 + 2048 * 8 * 8);
  add_case(&cases, "moe.fused_router.sigmoid.e256.h7168.k8.t8", "moe_fused_router",
           kops::DType::kBFloat16, 8, 7168, 8 * 7168 * 2 + 256 * 7168 * 2 + 8 * 8 * 8);
  add_case(&cases, "moe.fused_router.sigmoid.e256.h7168.k8.t32", "moe_fused_router",
           kops::DType::kBFloat16, 32, 7168, 32 * 7168 * 2 + 256 * 7168 * 2 + 32 * 8 * 8);
  add_case(&cases, "moe.hash.e256.k8.t4096", "moe_hash", kops::DType::kInt32, 4096, 256,
           4096 * 8 * 8 + 4096 * 8 + 4096 * 8 * 4);
  add_case(&cases, "moe.count_offsets.e256.align128.slots32768", "moe_count_offsets",
           kops::DType::kInt32, 32768, 256, 32768 * 4 + 256 * 12);
  add_case(&cases, "moe.align.e256.block128.slots32768", "moe_align", kops::DType::kInt32, 32768,
           256, 32768 * 4 + (32768 + 256 * 64) * 4 + ((32768 + 256 * 64) / 128) * 4);
  add_case(&cases, "moe.permute.bf16.t256.h7168.k8.e256", "moe_permute", kops::DType::kBFloat16,
           256, 7168, 256 * 7168 * 2 * 9);
  add_case(&cases, "moe.permute_fp8.t256.h7168.k8.e256", "moe_permute_fp8", kops::DType::kBFloat16,
           256, 7168, 256 * 7168 * 2 + 256 * 8 * 7168 + 256 * 8 * (7168 / 128) * 4);
  add_case(&cases, "moe.unpermute.bf16.t1024.h7168.k8", "moe_unpermute", kops::DType::kBFloat16,
           1024, 7168, 1024 * 7168 * 2 * (8 + 3));
  add_case(&cases, "moe.sum.bf16.t1024.k8.h7168", "moe_sum", kops::DType::kBFloat16, 1024, 7168,
           (1024 * 8 + 1024) * 7168 * 2);  // read [T,K,H], write [T,H]
  add_case(&cases, "moe.latent_combine.bf16.t1024.k8.l3584", "moe_latent", kops::DType::kBFloat16,
           1024, 3584, 1024 * 3584 * 2 * (8 + 1));
  add_case(&cases, "chain_moe_route_permute.bf16.t4096.h7168.e256.k8", "chain_moe_route_permute",
           kops::DType::kBFloat16, 4096, 7168, 4096 * 256 * 4 + 4096 * 7168 * 2 * (1 + 8) + 4096 * 8 * 4 * 2);

  // B17 experimental W2 experts: E=8, N=2048, K=4096, 1024 tokens x top-2 slots.
  add_case(&cases, "moe.w2.pack.bf16.e8.n2048.k4096.g32", "moe_w2_pack", kops::DType::kBFloat16,
           2048, 4096, 8ll * 2048 * 4096 * 2 + 8ll * 2048 * 4096 / 4 + 8ll * 2048 * 4096 / 32 * 4);
  add_case(&cases, "moe.w2.gemm.bf16.e8.n2048.k4096.slots2048", "moe_w2_gemm",
           kops::DType::kBFloat16, 2048, 2048, 8ll * 2048 * 4096 / 4 +
           8ll * 2048 * 4096 / 32 * 4 + 2048ll * 4096 * 2 + 2048ll * 2048 * 2);
  cases.back().flops = 2ll * 2048 * 2048 * 4096;
  return cases;
}

struct Buffers {
  int64_t tokens = 0;
  int64_t experts = 0;
  int64_t hidden = 0;
  int topk = 0;
  float* logits = nullptr;
  float* bias = nullptr;
  int32_t* ids = nullptr;
  float* weights = nullptr;
  int32_t* topk_ids = nullptr;
  int32_t* counts = nullptr;
  int64_t* offsets = nullptr;
  int32_t* sorted = nullptr;
  int32_t* expert_ids = nullptr;
  int32_t* post = nullptr;
  int32_t* cursor = nullptr;
  int32_t* slot_row = nullptr;
  int32_t* row_slot = nullptr;
  void* hidden_in = nullptr;
  void* perm = nullptr;
  void* expert_out = nullptr;
  void* out = nullptr;
  void* shared = nullptr;
  void* residual = nullptr;
  int64_t* token_ids = nullptr;
  int32_t* table = nullptr;
  __nv_bfloat16* h_bf = nullptr;
  __nv_bfloat16* w_bf = nullptr;
  int32_t* ws = nullptr;
};

bool alloc(Buffers* b, const Case& c, bool grouped) {
  b->tokens = c.rows;
  b->experts = grouped ? 256 : 256;
  b->topk = 8;
  const int64_t slots = b->tokens * b->topk;
  const int64_t hidden = c.cols;
  b->hidden = hidden;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->logits), slots * 0 + c.rows * 256 * 4),
               "logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->bias), 256 * 4), "bias") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->ids), slots * 4), "ids") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->weights), slots * 4), "weights") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->topk_ids), slots * 4), "topk_ids") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->counts), 256 * 4), "counts") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->offsets), 257 * 8), "offsets") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->cursor), 256 * 4), "cursor")) {
    return false;
  }
  fill_f32(b->logits, static_cast<size_t>(c.rows) * 256, 2.0f);
  cudaMemset(b->bias, 0, 256 * 4);
  cudaMemset(b->topk_ids, 0, slots * 4);
  std::vector<int32_t> host_ids(static_cast<size_t>(slots));
  for (int64_t i = 0; i < slots; ++i) {
    host_ids[static_cast<size_t>(i)] = static_cast<int32_t>((i * 2654435761ull) % 256);
  }
  cudaMemcpy(b->topk_ids, host_ids.data(), host_ids.size() * 4, cudaMemcpyHostToDevice);
  return true;
}

void free_bufs(Buffers* b) {
  cudaFree(b->logits);
  cudaFree(b->bias);
  cudaFree(b->ids);
  cudaFree(b->weights);
  cudaFree(b->topk_ids);
  cudaFree(b->counts);
  cudaFree(b->offsets);
  cudaFree(b->cursor);
  cudaFree(b->sorted);
  cudaFree(b->expert_ids);
  cudaFree(b->post);
  cudaFree(b->slot_row);
  cudaFree(b->row_slot);
  cudaFree(b->hidden_in);
  cudaFree(b->perm);
  cudaFree(b->expert_out);
  cudaFree(b->out);
  cudaFree(b->shared);
  cudaFree(b->residual);
  cudaFree(b->token_ids);
  cudaFree(b->table);
  cudaFree(b->ws);
}

kops::moe::RouterParams router_params(const Buffers& b) {
  kops::moe::RouterParams p{};
  p.logits = b.logits;
  p.logits_stride = b.experts;
  p.correction_bias = b.bias;
  p.num_tokens = static_cast<int>(b.tokens);
  p.num_experts = static_cast<int>(b.experts);
  p.topk = b.topk;
  p.num_expert_groups = 1;
  p.topk_groups = 1;
  p.renormalize = true;
  p.routed_scaling_factor = 2.5f;
  p.topk_ids = b.ids;
  p.topk_weights = b.weights;
  return p;
}

bool setup_permute(Buffers* b, const Case& c, bool quant) {
  const int64_t slots = b->tokens * b->topk;
  const int64_t rows = slots;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->hidden_in),
                          b->tokens * b->hidden * 2), "hidden") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->perm), quant ? b->tokens * b->topk * b->hidden
                                                                   : rows * b->hidden * 2),
               "perm") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->slot_row), slots * 4), "slot_row") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b->row_slot), rows * 4), "row_slot")) {
    return false;
  }
  cudaMemset(b->hidden_in, 0, b->tokens * b->hidden * 2);
  (void)c;
  return true;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  Buffers b{};
  const bool grouped = c.op == "moe_grouped";
  const bool fused = c.op == "moe_fused_router";
  if (grouped) {
    b.tokens = c.rows;
    b.experts = 256;
    b.hidden = c.cols;
    b.topk = 8;
    const int64_t slots = b.tokens * b.topk;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.logits), slots * 0 + b.tokens * 256 * 4), "g logits") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.bias), 256 * 4), "g bias") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.ids), slots * 4), "g ids") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.weights), slots * 4), "g weights")) {
      return false;
    }
    fill_f32(b.logits, static_cast<size_t>(b.tokens) * 256, 2.0f);
    cudaMemset(b.bias, 0, 256 * 4);
    auto p = router_params(b);
    p.num_expert_groups = 8;
    p.topk_groups = 4;
    p.group_score = kops::moe::GroupScore::kTop2Sum;
    const auto call = [&] {
      (void)kops::moe::route(p, kops::moe::RouterScore::kSigmoid, kops::DType::kFloat32, ctx);
    };
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  if (fused) {
    b.tokens = c.rows;
    b.hidden = c.cols;
    b.experts = 256;
    b.topk = 8;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.h_bf), b.tokens * b.hidden * 2), "x") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.w_bf), b.experts * b.hidden * 2), "w") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.bias), 256 * 4), "bias") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.ids), b.tokens * b.topk * 4), "ids") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.weights), b.tokens * b.topk * 4), "weights")) {
      return false;
    }
    fill_bf16(b.h_bf, static_cast<size_t>(b.tokens * b.hidden), 1.0f);
    fill_bf16(b.w_bf, static_cast<size_t>(b.experts * b.hidden), 0.05f);
    cudaMemset(b.bias, 0, 256 * 4);
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.logits), b.tokens * b.experts * 4), "fused logits")) {
      return false;
    }
    cudaMemset(b.logits, 0, b.tokens * b.experts * 4);
    auto p = router_params(b);
    p.logits = b.logits;
    p.logits_stride = b.experts;
    const Tensor t_x = tensor2d(b.h_bf, kops::DType::kBFloat16, b.tokens, b.hidden);
    const Tensor t_w = tensor2d(b.w_bf, kops::DType::kBFloat16, b.experts, b.hidden);
    const Tensor t_bias = tensor1d(b.bias, kops::DType::kFloat32, 256);
    const auto call = [&] {
      (void)kops::moe::route_fused(p, t_x, t_w, t_bias, kops::moe::RouterScore::kSigmoid,
                                   kops::DType::kBFloat16, ctx);
    };
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  if (!alloc(&b, c, grouped)) {
    free_bufs(&b);
    return false;
  }
  if (c.op == "moe_route") {
    auto p = router_params(b);
    const auto call = [&] {
      (void)kops::moe::route(p, kops::moe::RouterScore::kSigmoid, kops::DType::kFloat32, ctx);
    };
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  if (c.op == "moe_hash") {
    const int64_t tokens = c.rows;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.token_ids), tokens * 8), "token ids") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.table), 32000 * 8 * 4), "table")) {
      free_bufs(&b);
      return false;
    }
    cudaMemset(b.token_ids, 0, tokens * 8);
    cudaMemset(b.table, 0, 32000 * 8 * 4);
    const Tensor t_w = tensor2d(b.weights, kops::DType::kFloat32, tokens, 8);
    const Tensor t_ids = tensor2d(b.ids, kops::DType::kInt32, tokens, 8);
    const Tensor t_tok = tensor1d(b.token_ids, kops::DType::kInt64, tokens);
    const Tensor t_table = tensor2d(b.table, kops::DType::kInt32, 32000, 8);
    const auto call = [&] {
      (void)kops::moe::hash_route(t_w, t_ids, t_tok, t_table, Tensor{}, 2.0f, 256, ctx);
    };
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  if (c.op == "moe_count_offsets") {
    const int64_t slots = c.rows;
    const Tensor t_ids = tensor1d(b.topk_ids, kops::DType::kInt32, slots);
    const Tensor t_counts = tensor1d(b.counts, kops::DType::kInt32, 256);
    const Tensor t_off = tensor1d(b.offsets, kops::DType::kInt64, 257);
    const auto call = [&] {
      (void)kops::moe::count_and_offsets(t_ids, t_counts, t_off, Tensor{}, 256, 128, ctx);
    };
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  if (c.op == "moe_align") {
    const int64_t slots = c.rows;
    const int64_t capacity = slots + 256 * 127;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.sorted), capacity * 4), "sorted") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.expert_ids), capacity * 4), "expert_ids") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.post), 4), "post") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.ws), 257 * 4), "ws")) {
      free_bufs(&b);
      return false;
    }
    const Tensor t_ids = tensor1d(b.topk_ids, kops::DType::kInt32, slots);
    const Tensor t_sorted = tensor1d(b.sorted, kops::DType::kInt32, capacity);
    const Tensor t_eids = tensor1d(b.expert_ids, kops::DType::kInt32, capacity);
    const Tensor t_post = tensor1d(b.post, kops::DType::kInt32, 1);
    kops::Workspace ws{};
    ws.device_scratch = b.ws;
    ws.device_scratch_bytes = 257 * 4;
    const auto call = [&] {
      (void)kops::moe::align_block_size(t_ids, t_sorted, t_eids, t_post, Tensor{}, 256, 128, ws, ctx);
    };
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  if (c.op == "moe_permute" || c.op == "moe_permute_fp8") {
    const bool quant = c.op == "moe_permute_fp8";
    if (!setup_permute(&b, c, quant)) {
      free_bufs(&b);
      return false;
    }
    kops::moe::PermuteParams p{};
    p.hidden = b.hidden_in;
    p.hidden_stride = b.hidden;
    p.topk_ids = b.topk_ids;
    p.topk_weights = nullptr;
    p.apply_weight_on_input = false;
    p.expert_offsets = b.offsets;
    p.write_cursor = b.cursor;
    p.num_local_experts = static_cast<int32_t>(b.experts);
    p.permuted = b.perm;
    p.permuted_stride = b.hidden;
    p.permuted_row_to_slot = b.row_slot;
    p.slot_to_permuted_row = b.slot_row;
    p.num_tokens = static_cast<int>(b.tokens);
    p.topk = static_cast<int>(b.topk);
    p.hidden_size = static_cast<int>(b.hidden);
    if (quant) {
      p.quant.q = b.perm;
      p.quant.q_row_stride = b.hidden;
      float* scales = nullptr;
      if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), b.tokens * b.topk * (b.hidden / 128) * 4),
                   "scales")) {
        free_bufs(&b);
        return false;
      }
      b.perm = scales;  // reuse the free() path
      p.quant.scales = scales;
      p.quant.scale_row_stride = b.hidden / 128;
      p.quant.q_dtype = kops::DType::kFloat8E4M3;
      p.quant.mode = kops::ActQuantMode::kPerGroup;
      p.quant.group_size = 128;
    }
    {
      const Tensor t_flat = tensor1d(b.topk_ids, kops::DType::kInt32, b.tokens * b.topk);
      const Tensor t_counts = tensor1d(b.counts, kops::DType::kInt32, 256);
      const Tensor t_off = tensor1d(b.offsets, kops::DType::kInt64, 257);
      (void)kops::moe::count_and_offsets(t_flat, t_counts, t_off, Tensor{}, 256, 0, ctx);
    }
    const auto call = [&] {
      cudaMemsetAsync(b.cursor, 0, 256 * 4, reinterpret_cast<cudaStream_t>(ctx.stream));
      (void)kops::moe::permute(p, kops::DType::kBFloat16, ctx);
    };
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  if (c.op == "moe_unpermute" || c.op == "moe_sum" || c.op == "moe_latent") {
    const int64_t tokens = c.rows;
    const int64_t hidden = c.cols;
    const int64_t slots = tokens * b.topk;
    const int64_t rows = slots;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.expert_out), rows * hidden * 2), "expert_out") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.out), tokens * hidden * 2), "out") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.shared), tokens * hidden * 2), "shared") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.residual), tokens * hidden * 2), "residual") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.slot_row), slots * 4), "slot_row")) {
      free_bufs(&b);
      return false;
    }
    cudaMemset(b.expert_out, 0, rows * hidden * 2);
    cudaMemset(b.shared, 0, tokens * hidden * 2);
    cudaMemset(b.residual, 0, tokens * hidden * 2);
    std::vector<int32_t> host_slots(static_cast<size_t>(slots));
    for (int64_t i = 0; i < slots; ++i) host_slots[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    cudaMemcpy(b.slot_row, host_slots.data(), host_slots.size() * 4, cudaMemcpyHostToDevice);
    const Tensor t_out = tensor2d(b.out, kops::DType::kBFloat16, tokens, hidden);
    const Tensor t_expert = tensor2d(b.expert_out, kops::DType::kBFloat16, rows, hidden);
    const Tensor t_slots = tensor1d(b.slot_row, kops::DType::kInt32, slots);
    const Tensor t_w = tensor1d(b.weights, kops::DType::kFloat32, slots);
    const Tensor t_shared = tensor2d(b.shared, kops::DType::kBFloat16, tokens, hidden);
    const Tensor t_res = tensor2d(b.residual, kops::DType::kBFloat16, tokens, hidden);
    cudaMemset(b.weights, 0, slots * 4);
    if (c.op == "moe_unpermute") {
      const auto call = [&] {
        (void)kops::moe::unpermute_reduce(t_out, t_expert, t_slots, t_w, t_shared, t_res, 1.0f, false,
                                          ctx);
      };
      const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
      free_bufs(&b);
      return ok;
    }
    if (c.op == "moe_sum") {
      const Tensor t_in = tensor3d(b.expert_out, kops::DType::kBFloat16, tokens, b.topk, hidden);
      const Tensor t_in2 = tensor3d(b.expert_out, kops::DType::kBFloat16, tokens, b.topk, hidden);
      (void)t_in2;
      const auto call = [&] { (void)kops::moe::moe_sum(t_out, t_in, 8, ctx); };
      const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
      free_bufs(&b);
      return ok;
    }
    const int latent = static_cast<int>(hidden);
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.h_bf), latent * 2), "norm weight")) {
      free_bufs(&b);
      return false;
    }
    cudaMemset(b.h_bf, 0, latent * 2);
    const Tensor t_norm = tensor1d(b.h_bf, kops::DType::kBFloat16, latent);
    const auto call = [&] {
      (void)kops::moe::latent_combine_norm(t_out, t_expert, t_slots, t_w, t_norm, 1.0f, 1e-5f, 8,
                                           kops::ActQuantOut{}, ctx);
    };
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  if (c.op == "moe_w2_pack" || c.op == "moe_w2_gemm") {
    const int E = 8, N = 2048, K = 4096, G = 32, block = 64;
    const int tokens = 1024, topk = 2, slots = tokens * topk;
    __nv_bfloat16* w = nullptr;
    __nv_bfloat16* a = nullptr;
    __nv_bfloat16* cdev = nullptr;
    uint8_t* codes = nullptr;
    float* scales = nullptr;
    int32_t *sorted = nullptr, *eids = nullptr, *post = nullptr;
    bool ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&w), static_cast<size_t>(E) * N * K * 2), "w") &&
              cuda_ok(cudaMalloc(reinterpret_cast<void**>(&a), static_cast<size_t>(slots) * K * 2), "a") &&
              cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cdev), static_cast<size_t>(slots) * N * 2), "c") &&
              cuda_ok(cudaMalloc(reinterpret_cast<void**>(&codes), static_cast<size_t>(E) * N * K / 4), "codes") &&
              cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), static_cast<size_t>(E) * N * K / G * 4), "scales");
    if (!ok) return false;
    fill_bf16(w, static_cast<size_t>(E) * N * K, 0.3f);
    fill_bf16(a, static_cast<size_t>(slots) * K, 1.0f);
    // Group slots by expert with block-8 padding; sentinel = slots.
    std::vector<std::vector<int32_t>> per(E);
    for (int s = 0; s < slots; ++s) per[s % E].push_back(s);
    std::vector<int32_t> hs, he;
    for (int e = 0; e < E; ++e) {
      const int n = static_cast<int>(per[e].size());
      const int nb = (n + block - 1) / block;
      for (int b = 0; b < nb; ++b) he.push_back(e);
      for (int i = 0; i < nb * block; ++i) hs.push_back(i < n ? per[e][i] : slots);
    }
    const int post_pad = static_cast<int>(hs.size());
    ok = cuda_ok(cudaMalloc(reinterpret_cast<void**>(&sorted), hs.size() * 4), "sorted") &&
         cuda_ok(cudaMalloc(reinterpret_cast<void**>(&eids), he.size() * 4), "eids") &&
         cuda_ok(cudaMalloc(reinterpret_cast<void**>(&post), 4), "post");
    if (!ok) return false;
    cudaMemcpy(sorted, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(eids, he.data(), he.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(post, &post_pad, 4, cudaMemcpyHostToDevice);
    const kops::Tensor tw = tensor3d(w, kops::DType::kBFloat16, E, N, K);
    const kops::Tensor tc = tensor3d(codes, kops::DType::kUInt8, E, N, K / 4);
    const kops::Tensor ts = tensor3d(scales, kops::DType::kFloat32, E, N, K / G);
    ok = status_ok(kops::experimental::moe_w2::pack(tc, ts, tw, kops::WeightPacking::kNone, G, ctx),
                   c.id.c_str());
    if (ok && c.op == "moe_w2_pack") {
      ok = measure(c, [&] {
        (void)kops::experimental::moe_w2::pack(tc, ts, tw, kops::WeightPacking::kNone, G, ctx);
      }, st);
    } else if (ok) {
      kops::moe::FusedMoEGemmParams p{};
      p.a = a;
      p.lda = K;
      p.a_rows_are_slots = true;
      p.w = codes;
      p.w_expert_stride = static_cast<int64_t>(N) * (K / 4);
      p.w_scale = scales;
      p.w_scale_expert_stride = static_cast<int64_t>(N) * (K / G);
      p.c = cdev;
      p.ldc = N;
      p.sorted_token_ids = sorted;
      p.expert_ids = eids;
      p.num_tokens_post_pad = post;
      p.top_k = topk;
      p.n = N;
      p.k = K;
      p.num_valid_slots = slots;
      p.w_quant.packing = kops::WeightPacking::kW2;
      p.w_quant.granularity = kops::ScaleGranularity::kPerGroup;
      p.w_quant.group_k = G;
      ok = status_ok(kops::experimental::moe_w2::gemm(p, E, block, kops::DType::kBFloat16,
                                                      kops::DType::kBFloat16, ctx),
                     c.id.c_str());
      if (ok) {
        ok = measure(c, [&] {
          (void)kops::experimental::moe_w2::gemm(p, E, block, kops::DType::kBFloat16,
                                                 kops::DType::kBFloat16, ctx);
        }, st);
      }
    }
    for (void* p : {static_cast<void*>(w), static_cast<void*>(a), static_cast<void*>(cdev),
                    static_cast<void*>(codes), static_cast<void*>(scales),
                    static_cast<void*>(sorted), static_cast<void*>(eids), static_cast<void*>(post)}) {
      cudaFree(p);
    }
    return ok;
  }
  if (c.op == "chain_moe_route_permute") {
    const int64_t tokens = c.rows;
    const int64_t hidden = c.cols;
    const int64_t slots = tokens * 8;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.hidden_in), tokens * hidden * 2), "hidden") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.perm), slots * hidden * 2), "perm") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.slot_row), slots * 4), "slot_row") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.row_slot), slots * 4), "row_slot") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.sorted), (slots + 256 * 8) * 4), "sorted") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.expert_ids), (slots + 256 * 8) * 4), "eids") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.post), 4), "post") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&b.ws), 257 * 4), "ws")) {
      free_bufs(&b);
      return false;
    }
    cudaMemset(b.hidden_in, 0, tokens * hidden * 2);
    auto p = router_params(b);
    const Tensor t_ids = tensor1d(b.topk_ids, kops::DType::kInt32, slots);
    const Tensor t_sorted = tensor1d(b.sorted, kops::DType::kInt32, slots + 256 * 8);
    const Tensor t_eids = tensor1d(b.expert_ids, kops::DType::kInt32, slots + 256 * 8);
    const Tensor t_post = tensor1d(b.post, kops::DType::kInt32, 1);
    kops::Workspace ws{};
    ws.device_scratch = b.ws;
    ws.device_scratch_bytes = 257 * 4;
    kops::moe::PermuteParams pp{};
    pp.hidden = b.hidden_in;
    pp.hidden_stride = hidden;
    pp.topk_ids = b.ids;
    pp.expert_offsets = b.offsets;
    pp.write_cursor = b.cursor;
    pp.num_local_experts = static_cast<int32_t>(b.experts);
    pp.permuted = b.perm;
    pp.permuted_stride = hidden;
    pp.permuted_row_to_slot = b.row_slot;
    pp.slot_to_permuted_row = b.slot_row;
    pp.num_tokens = static_cast<int>(tokens);
    pp.topk = 8;
    pp.hidden_size = static_cast<int>(hidden);
    const Tensor t_route_ids = tensor1d(b.ids, kops::DType::kInt32, slots);
    const Tensor t_counts = tensor1d(b.counts, kops::DType::kInt32, 256);
    const Tensor t_offsets = tensor1d(b.offsets, kops::DType::kInt64, 257);
    (void)t_ids;
    const auto call = [&] {
      (void)kops::moe::route(p, kops::moe::RouterScore::kSigmoid, kops::DType::kFloat32, ctx);
      (void)kops::moe::count_and_offsets(t_route_ids, t_counts, t_offsets, Tensor{}, 256, 0, ctx);
      (void)kops::moe::align_block_size(t_route_ids, t_sorted, t_eids, t_post, Tensor{}, 256, 8, ws, ctx);
      cudaMemsetAsync(b.cursor, 0, 256 * 4, reinterpret_cast<cudaStream_t>(ctx.stream));
      (void)kops::moe::permute(pp, kops::DType::kBFloat16, ctx);
    };
    // Pre-flight once so a bad shape reports the step before timing.
    const kops::Status pre0 = kops::moe::route(p, kops::moe::RouterScore::kSigmoid, kops::DType::kFloat32, ctx);
    const kops::Status pre1 = kops::moe::count_and_offsets(t_route_ids, t_counts, t_offsets, Tensor{}, 256, 0, ctx);
    const kops::Status pre2 = kops::moe::align_block_size(t_route_ids, t_sorted, t_eids, t_post, Tensor{}, 256, 8, ws, ctx);
    if (pre0 != kops::Status::kSuccess || pre1 != kops::Status::kSuccess || pre2 != kops::Status::kSuccess) {
      std::fputs("[bench] chain metadata failed: ", stderr);
      std::fputs(kops::last_error_message(), stderr);
      std::fputc(10, stderr);
      free_bufs(&b);
      return false;
    }
    cudaMemset(b.cursor, 0, 256 * 4);
    const kops::Status pre3 = kops::moe::permute(pp, kops::DType::kBFloat16, ctx);
    if (pre3 != kops::Status::kSuccess) {
      std::fputs("[bench] chain permute failed: ", stderr);
      std::fputs(kops::last_error_message(), stderr);
      std::fputc(10, stderr);
      free_bufs(&b);
      return false;
    }
    const bool ok = status_ok(kops::Status::kSuccess, c.id.c_str()) && measure(c, call, st);
    free_bufs(&b);
    return ok;
  }
  free_bufs(&b);
  std::fprintf(stderr, "[bench] %s: no runner for op '%s'\n", c.id.c_str(), c.op.c_str());
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace kops_bench;
  const Options opt = parse_options(argc, argv);
  if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice")) return 1;
  DeviceInfo dev{};
  query_device(&dev);
  std::vector<Case> cases = make_cases();
  filter_cases(&cases, opt);
  if (opt.dump_cases) {
    dump_cases(cases, dev);
    return 0;
  }
  kops::LaunchContext ctx{};
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) return 1;
  std::vector<Result> results;
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
