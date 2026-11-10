// =============================================================================
// kernel-ops | benchmark/bench_state.cu
// B2 L03 state cache: state_copy, state_gather, state_scatter, state_zero.
// Two pools with different slot sizes exercise the per-pool byte geometry.
// B9 L01-L02 recurrent kernels (conv1d, SSM, SSD, lightning, delta rule) on real layer shapes.
// =============================================================================
#include "benchmark/harness.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/state/state_cache_kernels.cuh"
#include "csrc/state/conv1d_kernels.cuh"
#include "csrc/state/ssm_kernels.cuh"
#include "csrc/state/linear_attention_kernels.cuh"
#include "csrc/state/delta_rule_kernels.cuh"
#include "csrc/state/ops.h"

namespace {

using namespace kops_bench;
using kops::DType;
using kops::Tensor;

Tensor t1(void* p, int64_t n) { return tensor1d(p, DType::kInt64, n); }

bool run_state_commit(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int B = static_cast<int>(c.rows);
  const int M = static_cast<int>(c.cols);
  constexpr int kSlots = 128;
  constexpr int kB0 = 4096, kB1 = 2048, kPools = 2;
  void* pool0 = nullptr;
  void* pool1 = nullptr;
  int64_t* ptrs = nullptr;
  int64_t* bytes = nullptr;
  int32_t* live = nullptr;
  int32_t* inter = nullptr;
  int32_t* num = nullptr;
  if (!cuda_ok(cudaMalloc(&pool0, static_cast<size_t>(kSlots) * kB0), "commit pool0") ||
      !cuda_ok(cudaMalloc(&pool1, static_cast<size_t>(kSlots) * kB1), "commit pool1") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ptrs), kPools * 8), "commit ptrs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&bytes), kPools * 8), "commit bytes") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&live), B * 4), "commit live") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&inter), static_cast<size_t>(B) * M * 4),
               "commit inter") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&num), B * 4), "commit num")) {
    return false;
  }
  const int64_t pv[kPools] = {reinterpret_cast<int64_t>(pool0), reinterpret_cast<int64_t>(pool1)};
  const int64_t bv[kPools] = {kB0, kB1};
  cudaMemcpy(ptrs, pv, sizeof(pv), cudaMemcpyHostToDevice);
  cudaMemcpy(bytes, bv, sizeof(bv), cudaMemcpyHostToDevice);
  cudaMemset(pool0, 0x3C, static_cast<size_t>(kSlots) * kB0);
  cudaMemset(pool1, 0xA5, static_cast<size_t>(kSlots) * kB1);
  std::vector<int32_t> hl(B), hi(static_cast<size_t>(B) * M), hn(B);
  for (int b = 0; b < B; ++b) {
    hl[b] = b % kSlots;
    hn[b] = b % (M - 1);
    for (int j = 0; j < M; ++j) hi[static_cast<size_t>(b) * M + j] = 4 + (b * M + j) % (kSlots - 4);
  }
  cudaMemcpy(live, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(inter, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(num, hn.data(), hn.size() * 4, cudaMemcpyHostToDevice);
  const Tensor tp = t1(ptrs, kPools);
  const Tensor tb = t1(bytes, kPools);
  const Tensor tl = tensor1d(live, DType::kInt32, B);
  const Tensor ti = tensor2d(inter, DType::kInt32, B, M);
  const Tensor tn = tensor1d(num, DType::kInt32, B);
  const auto call = [&] { return kops::state::state_commit_accepted(tp, tb, tl, ti, tn, ctx); };
  const bool ok = status_ok(call(), c.id.c_str()) && measure(c, [&] { (void)call(); }, st);
  cudaFree(pool0);
  cudaFree(pool1);
  cudaFree(ptrs);
  cudaFree(bytes);
  cudaFree(live);
  cudaFree(inter);
  cudaFree(num);
  return ok;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "state_commit") return run_state_commit(c, ctx, st);
  constexpr int kPools = 2, kSlots = 256, kB0 = 4096, kB1 = 2048, kItems = 64;
  const int64_t total = static_cast<int64_t>(kItems) * (kB0 + kB1);
  void* pool0 = nullptr;
  void* pool1 = nullptr;
  int64_t* ptrs = nullptr;
  int64_t* bytes = nullptr;
  int32_t* pairs = nullptr;
  int32_t* slots = nullptr;
  void* staging = nullptr;
  if (!cuda_ok(cudaMalloc(&pool0, static_cast<size_t>(kSlots) * kB0), "pool0") ||
      !cuda_ok(cudaMalloc(&pool1, static_cast<size_t>(kSlots) * kB1), "pool1") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ptrs), kPools * 8), "ptrs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&bytes), kPools * 8), "bytes") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pairs), static_cast<size_t>(kItems) * 8), "pairs") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&slots), static_cast<size_t>(kItems) * 4), "slots") ||
      !cuda_ok(cudaMalloc(&staging, static_cast<size_t>(total)), "staging")) {
    return false;
  }
  const int64_t pv[kPools] = {reinterpret_cast<int64_t>(pool0), reinterpret_cast<int64_t>(pool1)};
  const int64_t bv[kPools] = {kB0, kB1};
  std::vector<int32_t> qv(static_cast<size_t>(kItems) * 2);
  std::vector<int32_t> sv(static_cast<size_t>(kItems));
  for (int i = 0; i < kItems; ++i) {
    qv[static_cast<size_t>(i * 2)] = i;
    qv[static_cast<size_t>(i * 2 + 1)] = (i + 1) % kSlots;
    sv[static_cast<size_t>(i)] = i;
  }
  cudaMemcpy(ptrs, pv, sizeof(pv), cudaMemcpyHostToDevice);
  cudaMemcpy(bytes, bv, sizeof(bv), cudaMemcpyHostToDevice);
  cudaMemcpy(pairs, qv.data(), qv.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(slots, sv.data(), sv.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(pool0, 1, static_cast<size_t>(kSlots) * kB0);
  cudaMemset(pool1, 1, static_cast<size_t>(kSlots) * kB1);
  cudaMemset(staging, 1, static_cast<size_t>(total));

  const Tensor tp = t1(ptrs, kPools);
  const Tensor tb = t1(bytes, kPools);
  const Tensor tq = tensor2d(pairs, DType::kInt32, kItems, 2);
  const Tensor ts = tensor1d(slots, DType::kInt32, kItems);
  const Tensor tg = tensor2d(staging, DType::kUInt8, kItems, kB0 + kB1);

  bool ok = true;
  auto fn = [&] {
    if (c.op == "state_copy") {
      (void)kops::state::state_copy(tp, tb, tq, ctx);
    } else if (c.op == "state_gather") {
      (void)kops::state::state_gather(tg, tp, tb, ts, ctx);
    } else if (c.op == "state_scatter") {
      (void)kops::state::state_scatter(tp, tg, tb, ts, ctx);
    } else {
      (void)kops::state::state_zero(tp, tb, ts, ctx);
    }
  };
  if (ok) {
    if (c.op == "state_copy") {
      ok = status_ok(kops::state::state_copy(tp, tb, tq, ctx), c.id.c_str());
    } else if (c.op == "state_gather") {
      ok = status_ok(kops::state::state_gather(tg, tp, tb, ts, ctx), c.id.c_str());
    } else if (c.op == "state_scatter") {
      ok = status_ok(kops::state::state_scatter(tp, tg, tb, ts, ctx), c.id.c_str());
    } else {
      ok = status_ok(kops::state::state_zero(tp, tb, ts, ctx), c.id.c_str());
    }
  }
  if (ok) ok = measure(c, fn, st);
  cudaFree(pool0);
  cudaFree(pool1);
  cudaFree(ptrs);
  cudaFree(bytes);
  cudaFree(pairs);
  cudaFree(slots);
  cudaFree(staging);
  return ok;
}

// ---- B9 recurrent kernels ---------------------------------------------------------------------
// Shapes of real layers that fit 4 GB (sequences / batch reduced, head geometry kept):
//   Qwen3-Next GDN   Hk 16, Hv 32, Dk = Dv = 128, conv 4      Kimi Linear KDA  H 32, Dk = Dv = 128, conv 4
//   Mamba-2          H 32, P 64, N 128, G 8 (Nemotron-H-like)  Falcon-Mamba M1  dim 8192, N 16
//   MiniMax lightning H 64, Dk = Dv = 128
// Every buffer is filled with small values (bytes 0x3C / 0x3C3C), so no NaN or overflow occurs.

/// Device scratch owned by one case.
struct Bufs {
  std::vector<void*> p;
  ~Bufs() {
    for (void* q : p) cudaFree(q);
  }
  template <typename T>
  T* get(size_t n, int fill = 0) {
    void* q = nullptr;
    if (cudaMalloc(&q, std::max<size_t>(n, 1) * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemset(q, fill, std::max<size_t>(n, 1) * sizeof(T));
    p.push_back(q);
    return static_cast<T*>(q);
  }
};

using bf16 = __nv_bfloat16;

std::vector<int32_t> iota_i32(int n, int start = 0) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = start + i;
  return v;
}

template <typename T>
T* put(Bufs& b, const std::vector<T>& h) {
  T* d = b.get<T>(h.size());
  if (d) cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

/// cu_seqlens of `batch` sequences of `len` tokens.
std::vector<int32_t> uniform_cu(int batch, int len) {
  std::vector<int32_t> cu(static_cast<size_t>(batch) + 1);
  for (int b = 0; b <= batch; ++b) cu[static_cast<size_t>(b)] = b * len;
  return cu;
}

template <typename Fn>
bool run_b9(const Case& c, Fn&& fn, Stats* st) {
  if (!status_ok(fn(), c.id.c_str())) return false;
  if (!cuda_ok(cudaDeviceSynchronize(), c.id.c_str())) return false;
  return measure(c, [&] { (void)fn(); }, st);
}

// conv1d prefill / update, Qwen3-Next conv_dim = 2 * 16 * 128 + 32 * 128 = 8192, width 4.
bool case_conv(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  constexpr int kDim = 8192, kW = 4;
  Bufs b;
  if (c.op == "conv1d_fwd") {
    const int B = 2, L = 2048, T = B * L;
    kops::state::Conv1dFwdParams p{};
    p.x = b.get<bf16>(static_cast<size_t>(T) * kDim, 0x3C);
    p.x_token_stride = kDim;
    p.x_dim_stride = 1;
    p.weight = b.get<bf16>(static_cast<size_t>(kDim) * kW, 0x3C);
    p.weight_dim_stride = kW;
    p.weight_width_stride = 1;
    p.bias = b.get<bf16>(kDim);
    p.out = b.get<bf16>(static_cast<size_t>(T) * kDim);
    p.out_token_stride = kDim;
    p.out_dim_stride = 1;
    p.query_start_loc = put(b, uniform_cu(B, L));
    p.cache_indices = put(b, iota_i32(B));
    p.has_initial_state = b.get<uint8_t>(B, 1);
    p.conv_states = b.get<bf16>(static_cast<size_t>(B) * kDim * (kW - 1));
    p.conv_state_slot_stride = static_cast<int64_t>(kDim) * (kW - 1);
    p.conv_state_dim_stride = kW - 1;
    p.conv_state_len_stride = 1;
    p.num_slots = B;
    p.batch_size = B;
    p.dim = kDim;
    p.width = kW;
    p.total_tokens = T;
    p.max_seqlen = L;
    p.pad_slot_id = -1;
    p.act = kops::ActFn::kSiLU;
    p.weight_dtype = DType::kBFloat16;
    return run_b9(c, [&] { return kops::state::causal_conv1d_fwd(p, DType::kBFloat16, ctx); }, st);
  }
  const int B = 64;
  kops::state::Conv1dUpdateParams p{};
  bf16* x = b.get<bf16>(static_cast<size_t>(B) * kDim, 0x3C);
  p.x = x;
  p.x_batch_stride = kDim;
  p.x_token_stride = kDim;
  p.weight = b.get<bf16>(static_cast<size_t>(kDim) * kW, 0x3C);
  p.weight_dim_stride = kW;
  p.weight_width_stride = 1;
  p.out = b.get<bf16>(static_cast<size_t>(B) * kDim);
  p.out_batch_stride = kDim;
  p.out_token_stride = kDim;
  p.conv_states = b.get<bf16>(static_cast<size_t>(B) * kDim * (kW - 1));
  p.conv_state_slot_stride = static_cast<int64_t>(kDim) * (kW - 1);
  p.conv_state_dim_stride = kW - 1;
  p.conv_state_len_stride = 1;
  p.num_slots = B;
  p.cache_indices = put(b, iota_i32(B));
  p.batch_size = B;
  p.seqlen = 1;
  p.dim = kDim;
  p.width = kW;
  p.pad_slot_id = -1;
  p.act = kops::ActFn::kSiLU;
  p.weight_dtype = DType::kBFloat16;
  return run_b9(c, [&] { return kops::state::causal_conv1d_update(p, DType::kBFloat16, ctx); }, st);
}

// Mamba-2 decode (H 32, P 64, N 128, G 8) and Mamba-1 selective scan (dim 8192, N 16).
bool case_ssm(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  Bufs b;
  if (c.op == "ssm_update") {
    const int B = 64, H = 32, P = 64, N = 128, G = 8;
    kops::state::SSMStateUpdateParams p{};
    p.state = b.get<float>(static_cast<size_t>(B) * H * P * N);
    p.state_slot_stride = static_cast<int64_t>(H) * P * N;
    p.state_head_stride = static_cast<int64_t>(P) * N;
    p.state_p_stride = N;
    p.x = b.get<bf16>(static_cast<size_t>(B) * H * P, 0x3C);
    p.x_batch_stride = static_cast<int64_t>(H) * P;
    p.x_token_stride = p.x_batch_stride;
    p.x_head_stride = P;
    p.dt = b.get<bf16>(static_cast<size_t>(B) * H, 0x3C);
    p.dt_batch_stride = H;
    p.dt_token_stride = H;
    p.dt_head_stride = 1;
    p.A = b.get<float>(H);
    p.A_head_stride = 1;
    p.B = b.get<bf16>(static_cast<size_t>(B) * G * N, 0x3C);
    p.B_batch_stride = static_cast<int64_t>(G) * N;
    p.B_token_stride = p.B_batch_stride;
    p.B_group_stride = N;
    p.C = b.get<bf16>(static_cast<size_t>(B) * G * N, 0x3C);
    p.C_batch_stride = p.B_batch_stride;
    p.C_token_stride = p.B_batch_stride;
    p.C_group_stride = N;
    p.D = b.get<float>(H);
    p.D_head_stride = 1;
    p.z = b.get<bf16>(static_cast<size_t>(B) * H * P, 0x3C);
    p.z_batch_stride = p.x_batch_stride;
    p.z_token_stride = p.x_batch_stride;
    p.z_head_stride = P;
    p.out = b.get<bf16>(static_cast<size_t>(B) * H * P);
    p.out_batch_stride = p.x_batch_stride;
    p.out_token_stride = p.x_batch_stride;
    p.out_head_stride = P;
    p.state_batch_indices = put(b, iota_i32(B));
    p.dt_softplus = true;
    p.num_slots = B;
    p.num_steps = 1;
    p.batch_size = B;
    p.nheads = H;
    p.headdim = P;
    p.dstate = N;
    p.ngroups = G;
    p.pad_slot_id = -1;
    return run_b9(c, [&] { return kops::state::selective_state_update(p, DType::kBFloat16, ctx); }, st);
  }
  const int B = 2, L = 2048, T = B * L, D = 8192, N = 16;
  kops::state::SelectiveScanParams p{};
  p.u = b.get<bf16>(static_cast<size_t>(T) * D, 0x3C);
  p.u_dim_stride = 1;
  p.u_token_stride = D;
  p.delta = b.get<bf16>(static_cast<size_t>(T) * D, 0x3C);
  p.delta_dim_stride = 1;
  p.delta_token_stride = D;
  p.A = b.get<float>(static_cast<size_t>(D) * N);
  p.A_dim_stride = N;
  p.B = b.get<bf16>(static_cast<size_t>(T) * N, 0x3C);
  p.B_state_stride = 1;
  p.B_token_stride = N;
  p.C = b.get<bf16>(static_cast<size_t>(T) * N, 0x3C);
  p.C_state_stride = 1;
  p.C_token_stride = N;
  p.D = b.get<float>(D);
  p.z = b.get<bf16>(static_cast<size_t>(T) * D, 0x3C);
  p.z_dim_stride = 1;
  p.z_token_stride = D;
  p.delta_softplus = true;
  p.out = b.get<bf16>(static_cast<size_t>(T) * D);
  p.out_dim_stride = 1;
  p.out_token_stride = D;
  p.ssm_states = b.get<float>(static_cast<size_t>(B) * D * N);
  p.ssm_state_slot_stride = static_cast<int64_t>(D) * N;
  p.ssm_state_dim_stride = N;
  p.query_start_loc = put(b, uniform_cu(B, L));
  p.cache_indices = put(b, iota_i32(B));
  p.num_slots = B;
  p.batch_size = B;
  p.dim = D;
  p.dstate = N;
  p.ngroups = 1;
  p.total_tokens = T;
  p.pad_slot_id = -1;
  return run_b9(c, [&] { return kops::state::selective_scan_fwd(p, DType::kBFloat16, ctx); }, st);
}

// Mamba-2 SSD prefill: H 32, P 64, N 128, G 8, chunk 256, 2 x 2048 tokens.
bool case_ssd(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int B = 2, L = 2048, T = B * L, H = 32, P = 64, N = 128, G = 8, CH = 256, NC = T / CH;
  Bufs b;
  kops::state::SSDParams p{};
  p.x = b.get<bf16>(static_cast<size_t>(T) * H * P, 0x3C);
  p.x_token_stride = static_cast<int64_t>(H) * P;
  p.x_head_stride = P;
  p.dt = b.get<bf16>(static_cast<size_t>(T) * H, 0x3C);
  p.dt_token_stride = H;
  p.dt_head_stride = 1;
  p.A = b.get<float>(H);
  p.B = b.get<bf16>(static_cast<size_t>(T) * G * N, 0x3C);
  p.B_token_stride = static_cast<int64_t>(G) * N;
  p.B_group_stride = N;
  p.C = b.get<bf16>(static_cast<size_t>(T) * G * N, 0x3C);
  p.C_token_stride = p.B_token_stride;
  p.C_group_stride = N;
  p.D = b.get<float>(H);
  p.D_head_stride = 1;
  p.z = b.get<bf16>(static_cast<size_t>(T) * H * P, 0x3C);
  p.z_token_stride = p.x_token_stride;
  p.z_head_stride = P;
  p.out = b.get<bf16>(static_cast<size_t>(T) * H * P);
  p.out_token_stride = p.x_token_stride;
  p.out_head_stride = P;
  p.dt_softplus = true;
  p.cu_seqlens = put(b, uniform_cu(B, L));
  p.cu_chunks = put(b, uniform_cu(B, L / CH));
  p.cu_chunk_seqlens = put(b, uniform_cu(NC, CH));
  p.ssm_states = b.get<float>(static_cast<size_t>(B) * H * P * N);
  p.ssm_state_slot_stride = static_cast<int64_t>(H) * P * N;
  p.state_indices = put(b, iota_i32(B));
  p.num_slots = B;
  p.batch_size = B;
  p.num_chunks = NC;
  p.chunk_size = CH;
  p.nheads = H;
  p.headdim = P;
  p.dstate = N;
  p.ngroups = G;
  p.pad_slot_id = -1;
  kops::WorkspaceRequirements req{};
  if (!status_ok(kops::state::ssd_workspace_requirements(T, B, H, P, N, G, CH, &req), "ssd ws")) return false;
  kops::Workspace ws{};
  ws.device_scratch = b.get<uint8_t>(req.device_scratch_bytes);
  ws.device_scratch_bytes = req.device_scratch_bytes;
  return run_b9(c, [&] { return kops::state::ssd_chunked_scan(p, DType::kBFloat16, ws, ctx); }, st);
}

// MiniMax lightning attention: H 64, Dk = Dv = 128; prefill 2 x 2048 tokens, decode B 64.
bool case_linear(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const bool prefill = c.op == "lightning_prefill";
  const int H = 64, Dk = 128, Dv = 128, B = prefill ? 2 : 64, L = prefill ? 2048 : 1, T = B * L;
  Bufs b;
  kops::state::LinearAttentionParams p{};
  p.q = b.get<bf16>(static_cast<size_t>(T) * H * Dk, 0x3C);
  p.k = b.get<bf16>(static_cast<size_t>(T) * H * Dk, 0x3C);
  p.v = b.get<bf16>(static_cast<size_t>(T) * H * Dv, 0x3C);
  p.q_token_stride = static_cast<int64_t>(H) * Dk;
  p.q_head_stride = Dk;
  p.k_token_stride = p.q_token_stride;
  p.k_head_stride = Dk;
  p.v_token_stride = static_cast<int64_t>(H) * Dv;
  p.v_head_stride = Dv;
  p.slope = b.get<float>(H);
  p.scale = 0.088f;
  p.state_pool = b.get<float>(static_cast<size_t>(B) * H * Dk * Dv);
  p.state_slot_stride = static_cast<int64_t>(H) * Dk * Dv;
  p.state_indices = put(b, iota_i32(B));
  p.o = b.get<bf16>(static_cast<size_t>(T) * H * Dv);
  p.o_token_stride = p.v_token_stride;
  p.o_head_stride = Dv;
  p.cu_seqlens = put(b, uniform_cu(B, L));
  p.num_slots = B;
  p.total_tokens = T;
  p.batch_size = B;
  p.num_heads = H;
  p.head_dim_k = Dk;
  p.head_dim_v = Dv;
  p.pad_slot_id = -1;
  if (prefill) return run_b9(c, [&] { return kops::state::lightning_attention_prefill(p, DType::kBFloat16, ctx); }, st);
  return run_b9(c, [&] { return kops::state::linear_attention_decode(p, DType::kBFloat16, ctx); }, st);
}

/// Delta-rule operands: GDN = Qwen3-Next (Hk 16, Hv 32), KDA = Kimi Linear (Hk = Hv = 32); Dk = Dv = 128.
void fill_delta(Bufs& b, kops::state::DeltaRuleParams& p, bool kda, int B, int L) {
  const int Hk = kda ? 32 : 16, Hv = 32, D = 128, T = B * L;
  p = kops::state::DeltaRuleParams{};
  p.q = b.get<bf16>(static_cast<size_t>(T) * Hk * D, 0x3C);
  p.k = b.get<bf16>(static_cast<size_t>(T) * Hk * D, 0x3C);
  p.v = b.get<bf16>(static_cast<size_t>(T) * Hv * D, 0x3C);
  p.q_token_stride = static_cast<int64_t>(Hk) * D;
  p.q_head_stride = D;
  p.k_token_stride = p.q_token_stride;
  p.k_head_stride = D;
  p.v_token_stride = static_cast<int64_t>(Hv) * D;
  p.v_head_stride = D;
  p.g = b.get<float>(static_cast<size_t>(T) * Hv * (kda ? D : 1), 0xBC);  // small negative log-decay
  p.beta = b.get<float>(static_cast<size_t>(T) * Hv, 0x3C);
  p.scale = 0.088f;
  p.use_qk_l2norm = true;
  p.cu_seqlens = put(b, uniform_cu(B, L));
  p.state_indices = put(b, iota_i32(B));
  p.state_pool = b.get<float>(static_cast<size_t>(B) * Hv * D * D);
  p.state_slot_stride = static_cast<int64_t>(Hv) * D * D;
  p.o = b.get<bf16>(static_cast<size_t>(T) * Hv * D);
  p.o_token_stride = p.v_token_stride;
  p.o_head_stride = D;
  p.num_slots = B;
  p.total_tokens = T;
  p.batch_size = B;
  p.num_k_heads = Hk;
  p.num_v_heads = Hv;
  p.head_dim_k = D;
  p.head_dim_v = D;
  p.decay = kda ? kops::state::DecayGranularity::kPerChannel : kops::state::DecayGranularity::kPerHead;
  p.pad_slot_id = -1;
}

bool case_delta(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const bool kda = c.id.find(".kda.") != std::string::npos;
  Bufs b;
  kops::state::DeltaRuleParams p;
  if (c.op == "delta_decode") {
    fill_delta(b, p, kda, 64, 1);
    return run_b9(c, [&] { return kops::state::delta_rule_decode(p, DType::kBFloat16, ctx); }, st);
  }
  if (c.op == "delta_prefill" || c.op == "delta_prefill_tc") {
    fill_delta(b, p, kda, 2, 2048);
    p.math = c.op == "delta_prefill_tc" ? kops::state::ChunkMath::kTensorCore : kops::state::ChunkMath::kFp32;
    kops::WorkspaceRequirements req{};
    if (!status_ok(kops::state::delta_rule_workspace_requirements(p.total_tokens, p.batch_size, p.num_v_heads, 128, 128, 64,
                                                                  p.decay, &req), "delta ws")) return false;
    kops::Workspace ws{};
    ws.device_scratch = b.get<uint8_t>(req.device_scratch_bytes);
    ws.device_scratch_bytes = req.device_scratch_bytes;
    return run_b9(c, [&] { return kops::state::delta_rule_prefill(p, DType::kBFloat16, ws, ctx); }, st);
  }
  if (c.op == "delta_gating") {
    const int T = 4096, Hv = 32, D = kda ? 128 : 1;
    kops::state::DeltaGatingParams g{};
    g.g = b.get<float>(static_cast<size_t>(T) * Hv * D);
    g.beta = b.get<float>(static_cast<size_t>(T) * Hv);
    g.a = b.get<bf16>(static_cast<size_t>(T) * Hv * D, 0x3C);
    g.a_token_stride = static_cast<int64_t>(Hv) * D;
    g.a_head_stride = D;
    g.b = b.get<bf16>(static_cast<size_t>(T) * Hv, 0x3C);
    g.b_token_stride = Hv;
    g.A_log = b.get<float>(Hv);
    g.dt_bias = b.get<float>(static_cast<size_t>(Hv) * D);
    g.decay = kda ? kops::state::DecayGranularity::kPerChannel : kops::state::DecayGranularity::kPerHead;
    g.total_tokens = T;
    g.num_v_heads = Hv;
    g.head_dim_k = D;
    return run_b9(c, [&] { return kops::state::delta_rule_gating(g, DType::kBFloat16, ctx); }, st);
  }
  // fused short conv + delta rule decode, B 64
  const int B = 64, Hk = kda ? 32 : 16, Hv = 32, D = 128, W = 4, conv_dim = 2 * Hk * D + Hv * D;
  kops::state::FusedDeltaDecodeParams f{};
  fill_delta(b, f.rule, kda, B, 1);
  f.qkv = b.get<bf16>(static_cast<size_t>(B) * conv_dim, 0x3C);
  f.qkv_batch_stride = conv_dim;
  f.conv_weight = b.get<bf16>(static_cast<size_t>(conv_dim) * W, 0x3C);
  f.conv_weight_dim_stride = W;
  f.conv_weight_width_stride = 1;
  f.conv_states = b.get<bf16>(static_cast<size_t>(B) * conv_dim * (W - 1));
  f.conv_state_slot_stride = static_cast<int64_t>(conv_dim) * (W - 1);
  f.conv_state_dim_stride = W - 1;
  f.conv_state_len_stride = 1;
  f.conv_width = W;
  f.gate_a = b.get<bf16>(static_cast<size_t>(B) * Hv * (kda ? D : 1), 0x3C);
  f.gate_a_batch_stride = static_cast<int64_t>(Hv) * (kda ? D : 1);
  f.gate_b = b.get<bf16>(static_cast<size_t>(B) * Hv, 0x3C);
  f.gate_b_batch_stride = Hv;
  f.A_log = b.get<float>(Hv);
  f.dt_bias = b.get<float>(static_cast<size_t>(Hv) * (kda ? D : 1));
  f.out_gate = b.get<bf16>(static_cast<size_t>(B) * Hv * D, 0x3C);
  f.out_gate_batch_stride = static_cast<int64_t>(Hv) * D;
  f.out_gate_act = kda ? kops::ActFn::kSigmoid : kops::ActFn::kSiLU;
  f.out_norm_weight = b.get<float>(D);
  f.out_norm_eps = 1e-5f;
  f.out = b.get<bf16>(static_cast<size_t>(B) * Hv * D);
  f.out_batch_stride = static_cast<int64_t>(Hv) * D;
  return run_b9(c, [&] { return kops::state::fused_delta_decode(f, DType::kBFloat16, ctx); }, st);
}

bool run_b9_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "conv1d_fwd" || c.op == "conv1d_update") return case_conv(c, ctx, st);
  if (c.op == "ssm_update" || c.op == "selective_scan") return case_ssm(c, ctx, st);
  if (c.op == "ssd_chunked_scan") return case_ssd(c, ctx, st);
  if (c.op == "lightning_prefill" || c.op == "linear_decode") return case_linear(c, ctx, st);
  return case_delta(c, ctx, st);
}

void add_b9_cases(std::vector<Case>& cases) {
  // bytes: the operands each kernel must move at least once (state read + write for decode)
  const int64_t conv_T = 4096, conv_dim = 8192;
  cases.push_back({"conv1d_fwd.bf16.qwen3next.T4096", "conv1d_fwd", DType::kBFloat16, conv_T, conv_dim,
                   conv_T * conv_dim * 2 * 2, 2 * 4 * conv_T * conv_dim, 5, 50});
  cases.push_back({"conv1d_update.bf16.qwen3next.B64", "conv1d_update", DType::kBFloat16, 64, conv_dim,
                   64 * conv_dim * (2 + 2 + 2 * 3 * 2), 2 * 4 * 64 * conv_dim, 10, 200});
  const int64_t m2_state = 64LL * 32 * 64 * 128 * 4;
  cases.push_back({"ssm_update.bf16.mamba2.H32P64N128.B64", "ssm_update", DType::kBFloat16, 64, 32 * 64,
                   2 * m2_state, 64LL * 32 * 64 * 128 * 6, 10, 200});
  const int64_t sc_T = 4096, sc_D = 8192;
  cases.push_back({"selective_scan.bf16.falcon_mamba.D8192N16.T4096", "selective_scan", DType::kBFloat16, sc_T, sc_D,
                   sc_T * sc_D * 2 * 4 + sc_T * 16 * 2 * 2, sc_T * sc_D * 16 * 8, 3, 20});
  const int64_t ssd_T = 4096;
  cases.push_back({"ssd_chunked_scan.bf16.mamba2.H32P64N128.T4096", "ssd_chunked_scan", DType::kBFloat16, ssd_T, 32 * 64,
                   ssd_T * 32 * 64 * 2 * 3 + ssd_T * 8 * 128 * 2 * 2, ssd_T * 32 * (2LL * 256 * 64 + 6LL * 64 * 128), 3, 20});
  const int64_t la_T = 4096;
  cases.push_back({"lightning_prefill.bf16.minimax.H64D128.T4096", "lightning_prefill", DType::kBFloat16, la_T, 64 * 128,
                   la_T * 64 * 128 * 2 * 4, la_T * 64 * (4LL * 64 * 128 + 4LL * 128 * 128), 3, 20});
  cases.push_back({"linear_decode.bf16.minimax.H64D128.B64", "linear_decode", DType::kBFloat16, 64, 64 * 128,
                   64LL * 64 * 128 * 128 * 4 * 2, 64LL * 64 * 128 * 128 * 4, 10, 100});
  const int64_t dstate = 64LL * 32 * 128 * 128 * 4;
  for (const char* kind : {"gdn", "kda"}) {
    const std::string k = kind;
    const int64_t chunk_flops = 4096LL / 64 * 32 * (8192LL * 640 + 64LL * 16384 * 6);
    cases.push_back({"delta_decode." + k + ".bf16.B64", "delta_decode", DType::kBFloat16, 64, 32 * 128, 2 * dstate,
                     64LL * 32 * 128 * 128 * 6, 10, 100});
    cases.push_back({"fused_delta_decode." + k + ".bf16.B64", "fused_delta_decode", DType::kBFloat16, 64, 32 * 128,
                     2 * dstate, 64LL * 32 * 128 * 128 * 6, 10, 100});
    cases.push_back({"delta_prefill." + k + ".fp32math.bf16.T4096", "delta_prefill", DType::kBFloat16, 4096, 32 * 128,
                     4096LL * 32 * 128 * 2 * 4, chunk_flops, 2, 10});
    cases.push_back({"delta_prefill." + k + ".tc.bf16.T4096", "delta_prefill_tc", DType::kBFloat16, 4096, 32 * 128,
                     4096LL * 32 * 128 * 2 * 4, chunk_flops, 3, 20});
    cases.push_back({"delta_gating." + k + ".bf16.T4096", "delta_gating", DType::kBFloat16, 4096, 32,
                     4096LL * 32 * (k == "kda" ? 128 : 1) * (2 + 4) + 4096LL * 32 * (2 + 4), 0, 10, 100});
  }
}

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  const int64_t bytes = 64LL * (4096 + 2048) * 2;
  cases.push_back({"state_copy.2pools.64items", "state_copy", DType::kUInt8, 64, 6144, bytes, 0,
                   10, 100});
  cases.push_back({"state_gather.2pools.64items", "state_gather", DType::kUInt8, 64, 6144, bytes,
                   0, 10, 100});
  cases.push_back({"state_scatter.2pools.64items", "state_scatter", DType::kUInt8, 64, 6144, bytes,
                   0, 10, 100});
  cases.push_back({"state_zero.2pools.64items", "state_zero", DType::kUInt8, 64, 6144, bytes / 2,
                   0, 10, 100});
  // B10: commit the intermediate state of every accepted speculative position.
  cases.push_back({"state_commit.2pools.b64.d8", "state_commit", DType::kUInt8, 64, 8,
                   int64_t{64} * (4096 + 2048) * 2, 0, 10, 100});
  add_b9_cases(cases);
  return cases;
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
    const bool b2 = c.op.rfind("state_", 0) == 0;
    if (!(b2 ? run_case(c, ctx, &st) : run_b9_case(c, ctx, &st))) {
      std::fprintf(stderr, "[bench] %s failed\n", c.id.c_str());
      return 1;
    }
    results.push_back({&c, st});
  }
  print_table(results, dev);
  if (opt.json_path != nullptr) write_results(opt.json_path, results, dev);
  return 0;
}
