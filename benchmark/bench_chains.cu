// =============================================================================
// kernel-ops | benchmark/bench_chains.cu
// Inference-subgraph benchmarks used for the B0/B1 >=2x acceptance gate. Each
// case is a sequence of kernels that replaces a realistic PyTorch eager chain
// (pre-norm, QK-norm+RoPE, GLU, embedding+norm, logits pipeline, split-KV
// merge). Case ids are the shared spec with benchmark/compare/torch_ops.py.
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <vector>

#include "csrc/common/quant.cuh"
#include "csrc/quantization/ops.h"
#include "csrc/model_ops/hyper_connection_kernels.cuh"
#include "csrc/model_ops/attn_res_kernels.cuh"
#include "csrc/model_ops/ops.h"
#include "csrc/norm/norm_kernels.cuh"
#include "csrc/norm/ops.h"
#include "csrc/activation/activation_kernels.cuh"
#include "csrc/activation/ops.h"
#include "csrc/position/rope_kernels.cuh"
#include "csrc/position/ops.h"
#include "csrc/embedding/embedding_kernels.cuh"
#include "csrc/embedding/ops.h"
#include "csrc/common/rng.cuh"
#include "csrc/sampling/sampling_kernels.cuh"
#include "csrc/sampling/fused_sampler_kernels.cuh"
#include "csrc/sampling/logits_processor_kernels.cuh"
#include "csrc/sampling/speculative_kernels.cuh"
#include "csrc/sampling/ops.h"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/merge_kernels.cuh"

namespace kops::attn {
Status merge_split_kv(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                      const Tensor& sink_logits, const LaunchContext& ctx);
}

namespace {

using namespace kops_bench;

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  cases.push_back({"chain.pre_norm_residual.bf16.b1024.h4096", "chain_pre_norm_residual",
                   kops::DType::kBFloat16, 1024, 4096, 6 * 1024 * 4096 * 2, 0, 10, 100});
  cases.push_back({"chain.pre_norm_glu.bf16.t2048.i4096", "chain_pre_norm_glu",
                   kops::DType::kBFloat16, 2048, 4096, 9 * 2048 * 4096 * 2, 0, 10, 100});
  cases.push_back({"chain.qk_norm_rope.bf16.t2048.hq32.hkv8.d128", "chain_qk_norm_rope",
                   kops::DType::kBFloat16, 2048, 128, 5 * 2048 * 40 * 128 * 2, 0, 10, 100});
  cases.push_back({"chain.embedding_norm.bf16.t2048.h4096", "chain_embedding_norm",
                   kops::DType::kBFloat16, 2048, 4096, 8 * 2048 * 4096 * 2, 0, 10, 100});
  cases.push_back({"chain.logits_pipeline.bf16.b32.v131072", "chain_logits_pipeline",
                   kops::DType::kBFloat16, 32, 131072, 6 * 32 * 131072 * 4, 0, 10, 100});
  cases.push_back({"chain.pre_norm_quant.bf16.b4096.h4096", "chain_pre_norm_quant",
                   kops::DType::kBFloat16, 4096, 4096, 6LL * 4096 * 4096, 5LL * 4096 * 4096, 20,
                   200});
  cases.push_back({"chain.glu_quant.bf16.t2048.i4096", "chain_glu_quant", kops::DType::kBFloat16,
                   2048, 4096, 8LL * 2048 * 4096, 6LL * 2048 * 4096, 20, 200});
  cases.push_back({"chain.ref_pre_norm_quant.bf16.b4096.h4096", "chain_ref_pre_norm_quant",
                   kops::DType::kBFloat16, 4096, 4096, 9LL * 4096 * 4096, 6LL * 4096 * 4096, 20,
                   200});
  cases.push_back({"chain.ref_glu_quant.bf16.t2048.i4096", "chain_ref_glu_quant",
                   kops::DType::kBFloat16, 2048, 4096, 12LL * 2048 * 4096,
                   7LL * 2048 * 4096, 20, 200});
  cases.push_back({"chain.splitkv_merge.fp32.b64.h32.d128.s8", "chain_splitkv_merge",
                   kops::DType::kFloat32, 64, 128, 8 * 64 * 32 * 128 * 4, 0, 10, 100});
  return cases;
}

struct ChainCtx {
  kops::LaunchContext ctx{};
  cudaStream_t stream = nullptr;
};

template <typename T>
T* alloc(size_t n) {
  T* p = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)) != cudaSuccess) return nullptr;
  cudaMemset(p, 0, n * sizeof(T));
  return p;
}

template <typename T>
bool run_pre_norm(const Case& c, const ChainCtx& cc, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* input = alloc<T>(n);
  T* residual = alloc<T>(n);
  T* weight = alloc<T>(static_cast<size_t>(c.cols));
  T* glu_in = alloc<T>(2 * n);
  T* glu_out = alloc<T>(n);
  if (!input || !residual || !weight || !glu_in || !glu_out) return false;
  const kops::Tensor ti = tensor2d(input, c.dtype, c.rows, c.cols);
  const kops::Tensor tr = tensor2d(residual, c.dtype, c.rows, c.cols);
  const kops::Tensor tw = tensor1d(weight, c.dtype, c.cols);
  const kops::Tensor tg = tensor2d(glu_in, c.dtype, c.rows, 2 * c.cols);
  const kops::Tensor to = tensor2d(glu_out, c.dtype, c.rows, c.cols);
  const bool glu = c.op == "chain_pre_norm_glu";
  auto fn = [&]() -> kops::Status {
    kops::Status s =
        kops::norm::fused_add_rms_norm(ti, tr, tw, 1e-5f, 0.0f, 1.0f, cc.ctx);
    if (s != kops::Status::kSuccess || !glu) return s;
    return kops::act::act_and_mul(to, tg, kops::Tensor{}, kops::ActFn::kSiLU,
                                  kops::GateLayout::kGateUp, kops::GluParams{}, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  cudaFree(input); cudaFree(residual); cudaFree(weight); cudaFree(glu_in); cudaFree(glu_out);
  return ok;
}

template <typename T>
bool run_qk_norm_rope(const Case& c, const ChainCtx& cc, Stats* st) {
  const int64_t tokens = c.rows;
  const int64_t d = c.cols;
  const int32_t hq = 32, hkv = 8;
  const int64_t qn = tokens * hq * d, kn = tokens * hkv * d;
  T* q = alloc<T>(static_cast<size_t>(qn));
  T* k = alloc<T>(static_cast<size_t>(kn));
  T* qw = alloc<T>(static_cast<size_t>(d));
  T* kw = alloc<T>(static_cast<size_t>(d));
  int64_t* pos = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&pos), static_cast<size_t>(tokens) * sizeof(int64_t));
  std::vector<int64_t> hpos(static_cast<size_t>(tokens));
  for (int64_t i = 0; i < tokens; ++i) hpos[static_cast<size_t>(i)] = i;
  cudaMemcpy(pos, hpos.data(), hpos.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  const int64_t max_pos = 4096;
  T* cache = alloc<T>(static_cast<size_t>(max_pos * d));
  if (!q || !k || !qw || !kw || !pos || !cache) return false;
  const kops::Tensor tq = tensor3d(q, c.dtype, tokens, hq, d);
  const kops::Tensor tk = tensor3d(k, c.dtype, tokens, hkv, d);
  const kops::Tensor tqw = tensor1d(qw, c.dtype, d);
  const kops::Tensor tkw = tensor1d(kw, c.dtype, d);
  const kops::Tensor tp = tensor1d(pos, kops::DType::kInt64, tokens);
  const kops::Tensor tc = tensor2d(cache, c.dtype, max_pos, d);
  kops::rope::RopeConfig cfg{};
  cfg.theta = 1e6f;
  cfg.rotary_dim = static_cast<int32_t>(d);
  float inv[128], ms = 1.0f;
  if (kops::rope::compute_inv_freq(cfg, inv, &ms) != kops::Status::kSuccess) return false;
  // Fill the cache through the public builder so the timed chain uses a real table.
  if (kops::rope::build_cos_sin_cache(tc, cfg, static_cast<int32_t>(max_pos), cc.ctx) !=
      kops::Status::kSuccess) {
    return false;
  }
  auto fn = [&]() -> kops::Status {
    kops::Status s = kops::norm::qk_norm(
        tq, tk, tqw, tkw, hq, hkv, static_cast<int32_t>(d), kops::NormType::kRMSNorm, false,
        kops::Tensor{}, 1e-6f, 0.0f, cc.ctx);
    if (s != kops::Status::kSuccess) return s;
    return kops::rope::apply_rotary(tp, tq, tk, tc, kops::Tensor{}, static_cast<int32_t>(d), 0,
                                    kops::RopeStyle::kNeoX, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  cudaFree(q); cudaFree(k); cudaFree(qw); cudaFree(kw); cudaFree(pos); cudaFree(cache);
  return ok;
}

template <typename T>
bool run_embedding_norm(const Case& c, const ChainCtx& cc, Stats* st) {
  const int64_t tokens = c.rows, hidden = c.cols;
  const int64_t vocab = 32000;
  int32_t* token_ids = nullptr;
  int64_t* positions = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&token_ids), static_cast<size_t>(tokens) * sizeof(int32_t));
  cudaMalloc(reinterpret_cast<void**>(&positions), static_cast<size_t>(tokens) * sizeof(int64_t));
  std::vector<int32_t> hids(static_cast<size_t>(tokens));
  std::vector<int64_t> hpos(static_cast<size_t>(tokens));
  for (int64_t i = 0; i < tokens; ++i) {
    hids[static_cast<size_t>(i)] = static_cast<int32_t>(i % vocab);
    hpos[static_cast<size_t>(i)] = i % 1024;
  }
  cudaMemcpy(token_ids, hids.data(), hids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(positions, hpos.data(), hpos.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  T* tok_w = alloc<T>(static_cast<size_t>(vocab) * hidden);
  T* pos_w = alloc<T>(static_cast<size_t>(1024) * hidden);
  T* out = alloc<T>(static_cast<size_t>(tokens) * hidden);
  T* normed = alloc<T>(static_cast<size_t>(tokens) * hidden);
  T* nw = alloc<T>(static_cast<size_t>(hidden));
  if (!token_ids || !positions || !tok_w || !pos_w || !out || !normed || !nw) return false;
  const kops::Tensor tids = tensor1d(token_ids, kops::DType::kInt32, tokens);
  const kops::Tensor tpos = tensor1d(positions, kops::DType::kInt64, tokens);
  const kops::Tensor ttw = tensor2d(tok_w, c.dtype, vocab, hidden);
  const kops::Tensor tpw = tensor2d(pos_w, c.dtype, 1024, hidden);
  const kops::Tensor to = tensor2d(out, c.dtype, tokens, hidden);
  const kops::Tensor tn = tensor2d(normed, c.dtype, tokens, hidden);
  const kops::Tensor tnw = tensor1d(nw, c.dtype, hidden);
  auto fn = [&]() -> kops::Status {
    kops::Status s =
        kops::embed::embedding_with_position(to, tids, tpos, ttw, tpw, 0, vocab, 0, true, cc.ctx);
    if (s != kops::Status::kSuccess) return s;
    return kops::norm::rms_norm(tn, to, tnw, 1e-6f, 0.0f, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  cudaFree(token_ids); cudaFree(positions); cudaFree(tok_w); cudaFree(pos_w);
  cudaFree(out); cudaFree(normed); cudaFree(nw);
  return ok;
}

bool run_logits_pipeline(const Case& c, const ChainCtx& cc, Stats* st) {
  const int batch = static_cast<int>(c.rows), vocab = static_cast<int>(c.cols);
  const size_t n = static_cast<size_t>(batch) * vocab;
  __nv_bfloat16* raw = alloc<__nv_bfloat16>(n);
  float* proc = alloc<float>(n);
  float* probs = alloc<float>(n);
  int32_t* counts = alloc<int32_t>(n);
  uint32_t* prompt = alloc<uint32_t>(static_cast<size_t>(batch) * ((vocab + 31) / 32));
  int32_t* bitmask = alloc<int32_t>(static_cast<size_t>(batch) * ((vocab + 31) / 32));
  int32_t* ids = alloc<int32_t>(static_cast<size_t>(batch));
  float* temp = alloc<float>(static_cast<size_t>(batch));
  float* repetition = alloc<float>(static_cast<size_t>(batch));
  float* frequency = alloc<float>(static_cast<size_t>(batch));
  float* presence = alloc<float>(static_cast<size_t>(batch));
  int32_t* chosen = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&chosen), static_cast<size_t>(batch) * sizeof(int32_t));
  std::vector<float> htemp(static_cast<size_t>(batch), 1.0f);
  std::vector<float> hrep(static_cast<size_t>(batch), 1.1f);
  std::vector<float> hfreq(static_cast<size_t>(batch), 0.1f);
  std::vector<float> hpres(static_cast<size_t>(batch), 0.1f);
  cudaMemcpy(temp, htemp.data(), htemp.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(repetition, hrep.data(), hrep.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(frequency, hfreq.data(), hfreq.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(presence, hpres.data(), hpres.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(prompt, 0, static_cast<size_t>(batch) * ((vocab + 31) / 32) * sizeof(uint32_t));
  cudaMemset(bitmask, 0xff, static_cast<size_t>(batch) * ((vocab + 31) / 32) * sizeof(int32_t));
  if (!raw || !proc || !probs || !counts || !prompt || !bitmask || !ids || !temp ||
      !repetition || !frequency || !presence || !chosen) {
    return false;
  }
  const kops::Tensor traw = tensor2d(raw, kops::DType::kBFloat16, batch, vocab);
  const kops::Tensor tproc = tensor2d(proc, kops::DType::kFloat32, batch, vocab);
  const kops::Tensor tprobs = tensor2d(probs, kops::DType::kFloat32, batch, vocab);
  const kops::Tensor tcounts = tensor2d(counts, kops::DType::kInt32, batch, vocab);
  const kops::Tensor tprompt = tensor2d(prompt, kops::DType::kUInt32, batch, (vocab + 31) / 32);
  const kops::Tensor tmask = tensor2d(bitmask, kops::DType::kInt32, batch, (vocab + 31) / 32);
  const kops::Tensor tids = tensor1d(ids, kops::DType::kInt32, batch);
  const kops::Tensor ttemp = tensor1d(temp, kops::DType::kFloat32, batch);
  const kops::Tensor trep = tensor1d(repetition, kops::DType::kFloat32, batch);
  const kops::Tensor tfreq = tensor1d(frequency, kops::DType::kFloat32, batch);
  const kops::Tensor tpres = tensor1d(presence, kops::DType::kFloat32, batch);
  const kops::Tensor tchosen = tensor1d(chosen, kops::DType::kInt32, batch);
  kops::WorkspaceRequirements req{};
  if (!status_ok(kops::sample::sampling_workspace_requirements(batch, vocab, 0, &req),
                 "chain softmax workspace requirements")) {
    return false;
  }
  void* scratch = nullptr;
  if (req.device_scratch_bytes > 0 &&
      !cuda_ok(cudaMalloc(&scratch, req.device_scratch_bytes), "chain softmax workspace")) {
    return false;
  }
  kops::Workspace ws{};
  ws.device_scratch = scratch;
  ws.device_scratch_bytes = req.device_scratch_bytes;
  auto fn = [&]() -> kops::Status {
    kops::Status s = kops::embed::logits_postprocess(tproc, traw, 1.0f, 30.0f, 0, vocab, cc.ctx);
    if (s != kops::Status::kSuccess) return s;
    s = kops::sample::apply_penalties(tproc, tcounts, tprompt, trep, tfreq, tpres, cc.ctx);
    if (s != kops::Status::kSuccess) return s;
    s = kops::sample::apply_token_bitmask(tproc, tmask, kops::Tensor{}, cc.ctx);
    if (s != kops::Status::kSuccess) return s;
    s = kops::sample::softmax(tprobs, tproc, ttemp, ws, cc.ctx);
    if (s != kops::Status::kSuccess) return s;
    return kops::sample::argmax(tchosen, kops::Tensor{}, tprobs, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  if (scratch != nullptr) cudaFree(scratch);
  cudaFree(raw); cudaFree(proc); cudaFree(probs); cudaFree(counts);
  cudaFree(prompt); cudaFree(bitmask); cudaFree(ids); cudaFree(temp);
  cudaFree(repetition); cudaFree(frequency); cudaFree(presence); cudaFree(chosen);
  return ok;
}

bool run_splitkv_merge(const Case& c, const ChainCtx& cc, Stats* st) {
  const int batch = static_cast<int>(c.rows), heads = 32, d = static_cast<int>(c.cols);
  const int splits = 8;
  const int64_t partial_rows = static_cast<int64_t>(batch) * splits;
  float* o = alloc<float>(static_cast<size_t>(batch) * heads * d);
  float* lse = alloc<float>(static_cast<size_t>(batch) * heads);
  float* op = alloc<float>(static_cast<size_t>(partial_rows) * heads * d);
  float* lp = alloc<float>(static_cast<size_t>(partial_rows) * heads);
  int32_t* indptr = alloc<int32_t>(static_cast<size_t>(batch) + 1);
  float* sink = alloc<float>(static_cast<size_t>(heads));
  if (!o || !lse || !op || !lp || !indptr || !sink) return false;
  std::vector<int32_t> hind(static_cast<size_t>(batch) + 1);
  for (int i = 0; i <= batch; ++i) hind[static_cast<size_t>(i)] = i * splits;
  cudaMemcpy(indptr, hind.data(), hind.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  const kops::Tensor to = tensor3d(o, kops::DType::kFloat32, batch, heads, d);
  const kops::Tensor tl = tensor2d(lse, kops::DType::kFloat32, batch, heads);
  const kops::Tensor tsink = tensor1d(sink, kops::DType::kFloat32, heads);
  kops::attn::SplitKVBuffers split{op, lp, indptr, splits};
  auto fn = [&]() -> kops::Status {
    return kops::attn::merge_split_kv(to, tl, split, tsink, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  cudaFree(o); cudaFree(lse); cudaFree(op); cudaFree(lp); cudaFree(indptr); cudaFree(sink);
  return ok;
}

template <typename T>
bool run_norm_quant(const Case& c, const ChainCtx& cc, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* x = nullptr;
  T* res = nullptr;
  T* w = nullptr;
  uint8_t* q = nullptr;
  float* scales = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&x), n * sizeof(T)), "x") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&res), n * sizeof(T)), "res") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&w), c.cols * sizeof(T)), "w") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), n), "q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), c.rows * sizeof(float)), "scales")) {
    return false;
  }
  cudaMemset(x, 0, n * sizeof(T));
  cudaMemset(res, 0, n * sizeof(T));
  cudaMemset(w, 0, c.cols * sizeof(T));
  kops::norm::NormQuantParams p{};
  p.input = x;
  p.input_stride = c.cols;
  p.residual = res;
  p.input_scale = 1.0f;
  p.weight = w;
  p.epsilon = 1e-5f;
  p.num_tokens = static_cast<int32_t>(c.rows);
  p.hidden_size = static_cast<int32_t>(c.cols);
  p.quant.q = q;
  p.quant.q_row_stride = c.cols;
  p.quant.scales = scales;
  p.quant.scale_row_stride = 1;
  p.quant.q_dtype = kops::DType::kFloat8E4M3;
  p.quant.mode = kops::ActQuantMode::kDynamicPerToken;
  auto fn = [&]() -> kops::Status {
    return kops::norm::norm_quant(p, kops::NormType::kRMSNorm, kops::DType::kBFloat16,
                                  kops::DType::kBFloat16, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  cudaFree(x); cudaFree(res); cudaFree(w); cudaFree(q); cudaFree(scales);
  return ok;
}

template <typename T>
bool run_glu_quant(const Case& c, const ChainCtx& cc, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  uint8_t* q = nullptr;
  float* scales = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), 2 * n * sizeof(T)), "in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), n), "q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), c.rows * sizeof(float)), "scales")) {
    return false;
  }
  cudaMemset(in, 0, 2 * n * sizeof(T));
  kops::act::ActMulQuantParams p{};
  p.input = in;
  p.input_stride = 2 * c.cols;
  p.glu = kops::GluParams{};
  p.num_rows = static_cast<int32_t>(c.rows);
  p.d = static_cast<int32_t>(c.cols);
  p.quant.q = q;
  p.quant.q_row_stride = c.cols;
  p.quant.scales = scales;
  p.quant.scale_row_stride = 1;
  p.quant.q_dtype = kops::DType::kFloat8E4M3;
  p.quant.mode = kops::ActQuantMode::kDynamicPerToken;
  auto fn = [&]() -> kops::Status {
    return kops::act::act_and_mul_quant(p, kops::ActFn::kSiLU, kops::GateLayout::kGateUp,
                                        kops::DType::kBFloat16, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  cudaFree(in); cudaFree(q); cudaFree(scales);
  return ok;
}

template <typename T>
bool run_ref_norm_quant(const Case& c, const ChainCtx& cc, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* x = nullptr; T* res = nullptr; T* w = nullptr; T* norm = nullptr;
  uint8_t* q = nullptr; float* scales = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&x), n * sizeof(T)), "x") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&res), n * sizeof(T)), "res") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&w), c.cols * sizeof(T)), "w") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&norm), n * sizeof(T)), "norm") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), n), "q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), c.rows * sizeof(float)), "scales")) {
    return false;
  }
  cudaMemset(x, 0, n * sizeof(T));
  cudaMemset(res, 0, n * sizeof(T));
  cudaMemset(w, 0, c.cols * sizeof(T));
  const kops::Tensor tx = tensor2d(x, kops::DType::kBFloat16, c.rows, c.cols);
  const kops::Tensor tr = tensor2d(res, kops::DType::kBFloat16, c.rows, c.cols);
  const kops::Tensor tw = tensor1d(w, kops::DType::kBFloat16, c.cols);
  const kops::Tensor tn = tensor2d(norm, kops::DType::kBFloat16, c.rows, c.cols);
  auto fn = [&]() -> kops::Status {
    kops::Status s = kops::norm::fused_add_rms_norm(tx, tr, tw, 1e-5f, 0.0f, 1.0f, cc.ctx);
    if (s != kops::Status::kSuccess) return s;
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols;
    o.scales = scales;
    o.scale_row_stride = 1;
    o.q_dtype = kops::DType::kFloat8E4M3;
    o.mode = kops::ActQuantMode::kDynamicPerToken;
    return kops::quant::quantize_activation(o, tn, kops::Tensor{}, 0, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  cudaFree(x); cudaFree(res); cudaFree(w); cudaFree(norm); cudaFree(q); cudaFree(scales);
  return ok;
}

template <typename T>
bool run_ref_glu_quant(const Case& c, const ChainCtx& cc, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr; T* act = nullptr; uint8_t* q = nullptr; float* scales = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), 2 * n * sizeof(T)), "in") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&act), n * sizeof(T)), "act") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), n), "q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&scales), c.rows * sizeof(float)), "scales")) {
    return false;
  }
  cudaMemset(in, 0, 2 * n * sizeof(T));
  const kops::Tensor ti = tensor2d(in, kops::DType::kBFloat16, c.rows, 2 * c.cols);
  const kops::Tensor ta = tensor2d(act, kops::DType::kBFloat16, c.rows, c.cols);
  auto fn = [&]() -> kops::Status {
    kops::Status s = kops::act::act_and_mul(ta, ti, kops::Tensor{}, kops::ActFn::kSiLU,
                                            kops::GateLayout::kGateUp, kops::GluParams{}, cc.ctx);
    if (s != kops::Status::kSuccess) return s;
    kops::ActQuantOut o{};
    o.q = q;
    o.q_row_stride = c.cols;
    o.scales = scales;
    o.scale_row_stride = 1;
    o.q_dtype = kops::DType::kFloat8E4M3;
    o.mode = kops::ActQuantMode::kDynamicPerToken;
    return kops::quant::quantize_activation(o, ta, kops::Tensor{}, 0, cc.ctx);
  };
  bool ok = status_ok(fn(), c.id.c_str());
  if (ok) ok = measure(c, fn, st);
  cudaFree(in); cudaFree(act); cudaFree(q); cudaFree(scales);
  return ok;
}

bool run_case(const Case& c, const ChainCtx& cc, Stats* st) {
  if (c.op == "chain_pre_norm_quant") return run_norm_quant<__nv_bfloat16>(c, cc, st);
  if (c.op == "chain_glu_quant") return run_glu_quant<__nv_bfloat16>(c, cc, st);
  if (c.op == "chain_ref_pre_norm_quant") return run_ref_norm_quant<__nv_bfloat16>(c, cc, st);
  if (c.op == "chain_ref_glu_quant") return run_ref_glu_quant<__nv_bfloat16>(c, cc, st);
  if (c.op == "chain_pre_norm_residual" || c.op == "chain_pre_norm_glu") {
    return c.dtype == kops::DType::kBFloat16 ? run_pre_norm<__nv_bfloat16>(c, cc, st)
                                             : run_pre_norm<float>(c, cc, st);
  }
  if (c.op == "chain_qk_norm_rope") return run_qk_norm_rope<__nv_bfloat16>(c, cc, st);
  if (c.op == "chain_embedding_norm") return run_embedding_norm<__nv_bfloat16>(c, cc, st);
  if (c.op == "chain_logits_pipeline") return run_logits_pipeline(c, cc, st);
  if (c.op == "chain_splitkv_merge") return run_splitkv_merge(c, cc, st);
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
  ChainCtx cc{};
  if (!status_ok(kops::make_launch_context(0, nullptr, &cc.ctx), "make_launch_context")) return 1;
  cc.stream = reinterpret_cast<cudaStream_t>(cc.ctx.stream);
  std::vector<Result> results;
  for (const Case& c : cases) {
    Stats st{};
    if (!run_case(c, cc, &st)) {
      std::fprintf(stderr, "[bench] %s failed\n", c.id.c_str());
      return 1;
    }
    results.push_back({&c, st});
  }
  print_table(results, dev);
  if (opt.json_path != nullptr) write_results(opt.json_path, results, dev);
  return 0;
}
