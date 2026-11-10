// =============================================================================
// kernel-ops | benchmark/bench_position.cu
// B1 position family cases: apply_rotary (table and on-the-fly),
// fused_qk_norm_rope, apply_mrope and build_cos_sin_cache.
// Case geometry is fixed per op (benchmark/compare/torch_baseline.py reads the
// same table through --dump-cases): Hq = 32, Hkv = 8, rot_dim = head_dim, the
// cache is 4096 rows unless the op builds it (then rows = -max_position-).
//   apply_rotary            rows = tokens, cols = head_dim
//   apply_rotary_on_the_fly rows = tokens, cols = head_dim
//   fused_qk_norm_rope      rows = tokens, cols = head_dim
//   apply_mrope             rows = tokens, cols = head_dim
//   build_cos_sin_cache     rows = max_position, cols = rot_dim
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>

#include <cmath>
#include <vector>

#include "csrc/common/op_types.cuh"
#include "csrc/position/rope_kernels.cuh"
#include "csrc/position/ops.h"

namespace {

using namespace kops_bench;

constexpr int64_t kHeads = 32;
constexpr int64_t kKvHeads = 8;
constexpr int64_t kCacheRows = 4096;
constexpr float kTheta = 10000.0f;

std::vector<Case> make_cases() {
  std::vector<Case> cases;

  // q and k read + written, the cache row read once per token, 6 flops per pair.
  auto add_apply = [&](const char* id, const char* op, kops::DType dt, int64_t tokens,
                       int64_t head_dim) {
    const int64_t es = dtype_size(dt);
    const int64_t qk = tokens * kHeads * head_dim;
    const int64_t kv = tokens * kKvHeads * head_dim;
    const int64_t bytes = 2 * (qk + kv) * es + tokens * head_dim * es;
    const int64_t flops = 6 * tokens * kHeads * (head_dim / 2);
    cases.push_back({id, op, dt, tokens, head_dim, bytes, flops, 20, 200});
  };

  for (kops::DType dt : {kops::DType::kBFloat16, kops::DType::kFloat32}) {
    const char* d = dt == kops::DType::kBFloat16 ? "bf16" : "fp32";
    const std::string tag = std::string("position.apply_rotary.") + d;
    for (int64_t tokens : {1, 16, 512, 4096}) {
      const std::string id = tag + ".t" + std::to_string(tokens);
      add_apply(id.c_str(), "apply_rotary", dt, tokens, 128);
    }
    const std::string fly = std::string("position.apply_rotary_on_the_fly.") + d + ".t512";
    add_apply(fly.c_str(), "apply_rotary_on_the_fly", dt, 512, 128);
    const std::string mrope = std::string("position.apply_mrope.") + d + ".t1024";
    add_apply(mrope.c_str(), "apply_mrope", dt, 1024, 128);
  }

  // fused: qkv is (Hq + 2 Hkv) * head_dim wide, read + written, plus cos/sin.
  for (kops::DType dt : {kops::DType::kBFloat16, kops::DType::kFloat32}) {
    const char* d = dt == kops::DType::kBFloat16 ? "bf16" : "fp32";
    for (int64_t tokens : {64, 2048}) {
      const int64_t head_dim = 128;
      const int64_t es = dtype_size(dt);
      const int64_t width = (kHeads + 2 * kKvHeads) * head_dim;
      const int64_t bytes = 2 * tokens * width * es + tokens * head_dim * es +
                            2 * head_dim * es;
      const int64_t flops = 5 * tokens * (kHeads + kKvHeads) * head_dim +
                            6 * tokens * (kHeads + kKvHeads) * (head_dim / 2);
      const std::string id = std::string("position.fused_qk_norm_rope.") + d + ".t" +
                             std::to_string(tokens);
      cases.push_back({id, "fused_qk_norm_rope", dt, tokens, head_dim, bytes, flops, 20, 200});
    }
  }

  // Llama-4 NoPE temperature tuning: in-place Q scale.
  for (kops::DType dt : {kops::DType::kBFloat16, kops::DType::kFloat32}) {
    const char* d = dt == kops::DType::kBFloat16 ? "bf16" : "fp32";
    const int64_t tokens = 1024, head_dim = 128;
    const int64_t es = dtype_size(dt);
    const int64_t bytes = 2 * tokens * kHeads * head_dim * es;
    const std::string id = std::string("position.attn_temperature_tuning.") + d + ".t1024";
    cases.push_back({id, "attn_temperature_tuning", dt, tokens, head_dim, bytes,
                     2 * tokens * kHeads * head_dim, 20, 200});
  }

  // cache builder: one row of rot_dim cos + sin values, written once.
  for (kops::DType dt : {kops::DType::kBFloat16, kops::DType::kFloat32}) {
    const char* d = dt == kops::DType::kBFloat16 ? "bf16" : "fp32";
    for (int64_t rows : {4096, 32768}) {
      const int64_t rot = 128;
      const int64_t es = dtype_size(dt);
      const std::string id = std::string("position.build_cos_sin_cache.") + d + ".p" +
                             std::to_string(rows);
      cases.push_back({id, "build_cos_sin_cache", dt, rows, rot, rows * rot * es,
                       4 * rows * (rot / 2), 5, 20});
    }
  }

  return cases;
}

std::vector<float> make_inv_freq(int64_t rot_dim) {
  std::vector<float> inv(static_cast<size_t>(rot_dim / 2));
  for (size_t i = 0; i < inv.size(); ++i) {
    inv[i] = static_cast<float>(std::pow(double(kTheta), -2.0 * double(i) / double(rot_dim)));
  }
  return inv;
}

struct Buffers {
  void* q = nullptr;
  void* k = nullptr;
  void* cache = nullptr;
  void* inv = nullptr;
  void* pos = nullptr;
  void* weight = nullptr;

  ~Buffers() {
    if (q != nullptr) cudaFree(q);
    if (k != nullptr) cudaFree(k);
    if (cache != nullptr) cudaFree(cache);
    if (inv != nullptr) cudaFree(inv);
    if (pos != nullptr) cudaFree(pos);
    if (weight != nullptr) cudaFree(weight);
  }
};

template <typename T>
bool alloc(Buffers* b, const Case& c, bool with_key, bool on_the_fly) {
  const int64_t tokens = c.rows;
  const int64_t head_dim = c.cols;
  const size_t qn = static_cast<size_t>(tokens * kHeads * head_dim);
  const size_t kn = static_cast<size_t>(tokens * kKvHeads * head_dim);
  if (!cuda_ok(cudaMalloc(&b->q, qn * sizeof(T)), "cudaMalloc q")) return false;
  if (!cuda_ok(cudaMalloc(&b->pos, static_cast<size_t>(tokens) * sizeof(int64_t)),
               "cudaMalloc positions")) {
    return false;
  }
  cudaMemset(b->q, 0, qn * sizeof(T));
  cudaMemset(b->pos, 0, static_cast<size_t>(tokens) * sizeof(int64_t));
  if (with_key) {
    if (!cuda_ok(cudaMalloc(&b->k, kn * sizeof(T)), "cudaMalloc k")) return false;
    cudaMemset(b->k, 0, kn * sizeof(T));
  }
  if (on_the_fly) {
    if (!cuda_ok(cudaMalloc(&b->inv, static_cast<size_t>(head_dim / 2) * sizeof(float)),
                 "cudaMalloc inv_freq")) {
      return false;
    }
    const std::vector<float> inv = make_inv_freq(head_dim);
    cudaMemcpy(b->inv, inv.data(), inv.size() * sizeof(float), cudaMemcpyHostToDevice);
  } else if (!cuda_ok(cudaMalloc(&b->cache, static_cast<size_t>(kCacheRows * head_dim) * sizeof(T)),
                      "cudaMalloc cache")) {
    return false;
  }
  return true;
}

kops::rope::RopeConfig rope_config(int64_t rot_dim) {
  kops::rope::RopeConfig cfg;
  cfg.scaling = kops::rope::RopeScaling::kNone;
  cfg.theta = kTheta;
  cfg.rotary_dim = static_cast<int32_t>(rot_dim);
  return cfg;
}

template <typename T>
bool run_apply(const Case& c, const kops::LaunchContext& ctx, Stats* st, bool on_the_fly) {
  const int64_t tokens = c.rows;
  const int64_t head_dim = c.cols;
  const int32_t rot_dim = static_cast<int32_t>(head_dim);
  Buffers b;
  if (!alloc<T>(&b, c, true, on_the_fly)) return false;
  const kops::Tensor tq = tensor3d(b.q, c.dtype, tokens, kHeads, head_dim);
  const kops::Tensor tk = tensor3d(b.k, c.dtype, tokens, kKvHeads, head_dim);
  const kops::Tensor tpos = tensor1d(b.pos, kops::DType::kInt64, tokens);
  bool ok = true;
  if (on_the_fly) {
    const kops::Tensor tinv = tensor1d(b.inv, kops::DType::kFloat32, head_dim / 2);
    ok = status_ok(kops::rope::apply_rotary_on_the_fly(tpos, tq, tk, tinv, 1.0f, rot_dim, 0,
                                                       kops::RopeStyle::kNeoX, ctx),
                   c.id.c_str());
    if (ok) {
      ok = measure(
          c,
          [&] {
            (void)kops::rope::apply_rotary_on_the_fly(tpos, tq, tk, tinv, 1.0f, rot_dim, 0,
                                                      kops::RopeStyle::kNeoX, ctx);
          },
          st);
    }
  } else {
    const kops::Tensor tcache = tensor2d(b.cache, c.dtype, kCacheRows, head_dim);
    ok = status_ok(kops::rope::build_cos_sin_cache(tcache, rope_config(head_dim),
                                                   static_cast<int32_t>(kCacheRows), ctx),
                   "build cache");
    if (ok) {
      ok = status_ok(kops::rope::apply_rotary(tpos, tq, tk, tcache, kops::Tensor{}, rot_dim, 0,
                                              kops::RopeStyle::kNeoX, ctx),
                     c.id.c_str());
    }
    if (ok) {
      ok = measure(
          c,
          [&] {
            (void)kops::rope::apply_rotary(tpos, tq, tk, tcache, kops::Tensor{}, rot_dim, 0,
                                           kops::RopeStyle::kNeoX, ctx);
          },
          st);
    }
  }
  return ok;
}

template <typename T>
bool run_mrope(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t tokens = c.rows;
  const int64_t head_dim = c.cols;
  const int32_t rot_dim = static_cast<int32_t>(head_dim);
  Buffers b;
  if (!alloc<T>(&b, c, true, false)) return false;
  cudaFree(b.pos);
  b.pos = nullptr;
  if (!cuda_ok(cudaMalloc(&b.pos, static_cast<size_t>(3 * tokens) * sizeof(int64_t)),
               "cudaMalloc mrope positions")) {
    return false;
  }
  cudaMemset(b.pos, 0, static_cast<size_t>(3 * tokens) * sizeof(int64_t));
  const kops::Tensor tq = tensor3d(b.q, c.dtype, tokens, kHeads, head_dim);
  const kops::Tensor tk = tensor3d(b.k, c.dtype, tokens, kKvHeads, head_dim);
  const kops::Tensor tcache = tensor2d(b.cache, c.dtype, kCacheRows, head_dim);
  const kops::Tensor tpos = tensor2d(b.pos, kops::DType::kInt64, 3, tokens);
  if (!status_ok(kops::rope::build_cos_sin_cache(tcache, rope_config(head_dim),
                                                 static_cast<int32_t>(kCacheRows), ctx),
                 "build cache")) {
    return false;
  }
  const int32_t section[3] = {static_cast<int32_t>(rot_dim / 4), static_cast<int32_t>(rot_dim / 8),
                              static_cast<int32_t>(rot_dim / 8)};
  bool ok = status_ok(kops::rope::apply_mrope(tpos, tq, tk, tcache, section, false,
                                              kops::RopeStyle::kNeoX, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] {
          (void)kops::rope::apply_mrope(tpos, tq, tk, tcache, section, false,
                                        kops::RopeStyle::kNeoX, ctx);
        },
        st);
  }
  return ok;
}

template <typename T>
bool run_fused(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t tokens = c.rows;
  const int64_t head_dim = c.cols;
  const int64_t width = (kHeads + 2 * kKvHeads) * head_dim;
  Buffers b;
  if (!cuda_ok(cudaMalloc(&b.q, static_cast<size_t>(tokens * width) * sizeof(T)),
               "cudaMalloc qkv")) {
    return false;
  }
  cudaMemset(b.q, 0, static_cast<size_t>(tokens * width) * sizeof(T));
  if (!cuda_ok(cudaMalloc(&b.pos, static_cast<size_t>(tokens) * sizeof(int64_t)),
               "cudaMalloc positions")) {
    return false;
  }
  cudaMemset(b.pos, 0, static_cast<size_t>(tokens) * sizeof(int64_t));
  if (!cuda_ok(cudaMalloc(&b.weight, static_cast<size_t>(head_dim) * sizeof(T)),
               "cudaMalloc weight")) {
    return false;
  }
  cudaMemset(b.weight, 0, static_cast<size_t>(head_dim) * sizeof(T));
  if (!cuda_ok(cudaMalloc(&b.cache, static_cast<size_t>(kCacheRows * head_dim) * sizeof(T)),
               "cudaMalloc cache")) {
    return false;
  }
  const kops::Tensor tqkv = tensor2d(b.q, c.dtype, tokens, width);
  const kops::Tensor tpos = tensor1d(b.pos, kops::DType::kInt64, tokens);
  const kops::Tensor tw = tensor1d(b.weight, c.dtype, head_dim);
  const kops::Tensor tcache = tensor2d(b.cache, c.dtype, kCacheRows, head_dim);
  if (!status_ok(kops::rope::build_cos_sin_cache(tcache, rope_config(head_dim),
                                                 static_cast<int32_t>(kCacheRows), ctx),
                 "build cache")) {
    return false;
  }
  const int32_t rot_dim = static_cast<int32_t>(head_dim);
  bool ok = status_ok(kops::rope::fused_qk_norm_rope(
                          tqkv, tw, tw, tpos, tcache, static_cast<int32_t>(kHeads),
                          static_cast<int32_t>(kKvHeads), static_cast<int32_t>(head_dim), rot_dim,
                          0, kops::RopeStyle::kNeoX, 0.0f, 1e-6f, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] {
          (void)kops::rope::fused_qk_norm_rope(
              tqkv, tw, tw, tpos, tcache, static_cast<int32_t>(kHeads),
              static_cast<int32_t>(kKvHeads), static_cast<int32_t>(head_dim), rot_dim, 0,
              kops::RopeStyle::kNeoX, 0.0f, 1e-6f, ctx);
        },
        st);
  }
  return ok;
}

template <typename T>
bool run_build_cache(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t rows = c.rows;
  const int64_t rot = c.cols;
  T* cache = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&cache),
                          static_cast<size_t>(rows * rot) * sizeof(T)),
               "cudaMalloc cache")) {
    return false;
  }
  cudaMemset(cache, 0, static_cast<size_t>(rows * rot) * sizeof(T));
  const kops::Tensor tcache = tensor2d(cache, c.dtype, rows, rot);
  bool ok = status_ok(kops::rope::build_cos_sin_cache(tcache, rope_config(rot),
                                                      static_cast<int32_t>(rows), ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] {
          (void)kops::rope::build_cos_sin_cache(tcache, rope_config(rot),
                                                static_cast<int32_t>(rows), ctx);
        },
        st);
  }
  cudaFree(cache);
  return ok;
}

template <typename T>
bool run_temperature(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const int64_t tokens = c.rows, d = c.cols;
  const int64_t n = tokens * 32 * d;
  T* q = nullptr;
  int64_t* positions = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&q), static_cast<size_t>(n) * sizeof(T)),
               "malloc q") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&positions),
                          static_cast<size_t>(tokens) * sizeof(int64_t)), "malloc positions")) {
    return false;
  }
  cudaMemset(q, 0, static_cast<size_t>(n) * sizeof(T));
  std::vector<int64_t> pos(static_cast<size_t>(tokens));
  for (int64_t i = 0; i < tokens; ++i) pos[static_cast<size_t>(i)] = i;
  cudaMemcpy(positions, pos.data(), pos.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  const kops::Tensor tq = tensor3d(q, c.dtype, tokens, 32, d);
  const kops::Tensor tp = tensor1d(positions, kops::DType::kInt64, tokens);
  auto call = [&] {
    return kops::rope::attn_temperature_tuning(tq, tp, 8192.0f, 0.1f, ctx);
  };
  bool ok = status_ok(call(), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)call(); }, st);
  cudaFree(q);
  cudaFree(positions);
  return ok;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const bool bf16 = c.dtype == kops::DType::kBFloat16;
  if (c.op == "build_cos_sin_cache") {
    return bf16 ? run_build_cache<__nv_bfloat16>(c, ctx, st) : run_build_cache<float>(c, ctx, st);
  }
  if (c.op == "fused_qk_norm_rope") {
    return bf16 ? run_fused<__nv_bfloat16>(c, ctx, st) : run_fused<float>(c, ctx, st);
  }
  if (c.op == "apply_mrope") {
    return bf16 ? run_mrope<__nv_bfloat16>(c, ctx, st) : run_mrope<float>(c, ctx, st);
  }
  if (c.op == "apply_rotary_on_the_fly") {
    return bf16 ? run_apply<__nv_bfloat16>(c, ctx, st, true) : run_apply<float>(c, ctx, st, true);
  }
  if (c.op == "apply_rotary") {
    return bf16 ? run_apply<__nv_bfloat16>(c, ctx, st, false) : run_apply<float>(c, ctx, st, false);
  }
  if (c.op == "attn_temperature_tuning") {
    return bf16 ? run_temperature<__nv_bfloat16>(c, ctx, st) : run_temperature<float>(c, ctx, st);
  }
  std::fprintf(stderr, "[bench] unknown op %s\n", c.op.c_str());
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
