// =============================================================================
// kernel-ops | benchmark/bench_embedding.cu
// B1.2 embedding cases: lookup (vocab shard), token + learned position, row
// gather / scatter and logits post-processing. The case table is the shared
// spec read by benchmark/compare/torch_baseline.py through --dump-cases.
//
// Case columns: rows/cols mean tokens/hidden for the row kernels and
// rows/vocab-shard for logits_postprocess. `bytes` counts every tensor the
// kernel touches (table row read + output write, or logits read + fp32 write).
// =============================================================================
#include "benchmark/harness.h"

#include <cuda_bf16.h>

#include <vector>

#include "csrc/common/op_types.cuh"
#include "csrc/embedding/embedding_kernels.cuh"
#include "csrc/embedding/ops.h"

namespace {

using namespace kops_bench;

constexpr int64_t kVocab = 16384;    // weight table rows used by the row kernels
constexpr int64_t kHidden = 4096;
constexpr int64_t kPosRows = 2048;   // learned absolute position table
constexpr int64_t kLogitsVocab = 128256;  // real LM-head width (padded shard)

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto add = [&](const char* id, const char* op, kops::DType dt, int64_t rows, int64_t cols,
                 int64_t bytes, int64_t flops) {
    cases.push_back({id, op, dt, rows, cols, bytes, flops, 20, 200});
  };

  // Embedding lookup: one table row read plus one output row written.
  auto add_lookup = [&](const char* id, kops::DType dt, int64_t tokens) {
    const int64_t es = dtype_size(dt);
    add(id, "embedding_lookup", dt, tokens, kHidden, 2 * tokens * kHidden * es, 0);
  };
  add_lookup("embedding.lookup.bf16.t2048", kops::DType::kBFloat16, 2048);
  add_lookup("embedding.lookup.fp16.t2048", kops::DType::kFloat16, 2048);
  add_lookup("embedding.lookup.fp32.t2048", kops::DType::kFloat32, 2048);
  add_lookup("embedding.lookup.bf16.t1", kops::DType::kBFloat16, 1);  // decode step

  // Token embedding + learned position: two table reads (one may be masked),
  // one output write, one fp32 add per element.
  auto add_pos = [&](const char* id, kops::DType dt, int64_t tokens) {
    const int64_t es = dtype_size(dt);
    add(id, "embedding_with_position", dt, tokens, kHidden, 3 * tokens * kHidden * es,
        tokens * kHidden);
  };
  add_pos("embedding.with_position.bf16.t2048", kops::DType::kBFloat16, 2048);
  add_pos("embedding.with_position.fp16.t2048", kops::DType::kFloat16, 2048);
  add_pos("embedding.with_position.fp32.t2048", kops::DType::kFloat32, 2048);

  // Row gather / scatter: read + write of the same byte count.
  auto add_copy = [&](const char* id, const char* op, kops::DType dt, int64_t rows) {
    const int64_t es = dtype_size(dt);
    add(id, op, dt, rows, kHidden, 2 * rows * kHidden * es, 0);
  };
  add_copy("embedding.gather_rows.bf16.n512", "gather_rows", kops::DType::kBFloat16, 512);
  add_copy("embedding.scatter_rows.bf16.n512", "scatter_rows", kops::DType::kBFloat16, 512);

  // Logits fix-up: fp16/bf16 read (2 B) + fp32 write (4 B), scale/softcap/mask.
  auto add_logits = [&](const char* id, kops::DType dt, int64_t rows) {
    const int64_t es = dtype_size(dt);
    add(id, "logits_postprocess", dt, rows, kLogitsVocab, rows * kLogitsVocab * (es + 4),
        rows * kLogitsVocab * 10);
  };
  add_logits("embedding.logits_postprocess.bf16.r1", kops::DType::kBFloat16, 1);
  add_logits("embedding.logits_postprocess.bf16.r32", kops::DType::kBFloat16, 32);

  return cases;
}

template <typename T>
bool run_lookup(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n_out = static_cast<size_t>(c.rows * c.cols);
  const size_t n_w = static_cast<size_t>(kVocab * c.cols);
  T* w = nullptr;
  T* out = nullptr;
  int32_t* ids = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&w), n_w * sizeof(T)), "cudaMalloc weight") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n_out * sizeof(T)), "cudaMalloc out") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ids), static_cast<size_t>(c.rows) * sizeof(int32_t)),
               "cudaMalloc ids")) {
    return false;
  }
  cudaMemset(w, 0, n_w * sizeof(T));
  cudaMemset(out, 0, n_out * sizeof(T));
  std::vector<int32_t> host_ids(static_cast<size_t>(c.rows));
  for (size_t i = 0; i < host_ids.size(); ++i) host_ids[i] = static_cast<int32_t>(i % kVocab);
  if (!cuda_ok(cudaMemcpy(ids, host_ids.data(), host_ids.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy ids")) {
    return false;
  }

  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  const kops::Tensor tw = tensor2d(w, c.dtype, kVocab, c.cols);
  const kops::Tensor ti = tensor1d(ids, kops::DType::kInt32, c.rows);

  bool ok = status_ok(kops::embed::embedding_lookup(to, ti, tw, 0, kVocab, 1.0f, ctx),
                      c.id.c_str());
  if (ok) {
    ok = measure(c, [&] { (void)kops::embed::embedding_lookup(to, ti, tw, 0, kVocab, 1.0f, ctx); },
                 st);
  }

  cudaFree(w);
  cudaFree(out);
  cudaFree(ids);
  return ok;
}

template <typename T>
bool run_with_position(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n_out = static_cast<size_t>(c.rows * c.cols);
  const size_t n_tok = static_cast<size_t>(kVocab * c.cols);
  const size_t n_pos = static_cast<size_t>(kPosRows * c.cols);
  T* tok = nullptr;
  T* pos = nullptr;
  T* out = nullptr;
  int32_t* ids = nullptr;
  int64_t* positions = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&tok), n_tok * sizeof(T)), "cudaMalloc token") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pos), n_pos * sizeof(T)), "cudaMalloc pos") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n_out * sizeof(T)), "cudaMalloc out") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ids), static_cast<size_t>(c.rows) * sizeof(int32_t)),
               "cudaMalloc ids") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&positions),
                          static_cast<size_t>(c.rows) * sizeof(int64_t)),
               "cudaMalloc positions")) {
    return false;
  }
  cudaMemset(tok, 0, n_tok * sizeof(T));
  cudaMemset(pos, 0, n_pos * sizeof(T));
  cudaMemset(out, 0, n_out * sizeof(T));
  std::vector<int32_t> host_ids(static_cast<size_t>(c.rows));
  std::vector<int64_t> host_pos(static_cast<size_t>(c.rows));
  for (size_t i = 0; i < host_ids.size(); ++i) {
    host_ids[i] = static_cast<int32_t>(i % kVocab);
    host_pos[i] = static_cast<int64_t>(i % kPosRows);
  }
  cudaMemcpy(ids, host_ids.data(), host_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(positions, host_pos.data(), host_pos.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  const kops::Tensor tt = tensor2d(tok, c.dtype, kVocab, c.cols);
  const kops::Tensor tp = tensor2d(pos, c.dtype, kPosRows, c.cols);
  const kops::Tensor ti = tensor1d(ids, kops::DType::kInt32, c.rows);
  const kops::Tensor tpos = tensor1d(positions, kops::DType::kInt64, c.rows);

  bool ok = status_ok(
      kops::embed::embedding_with_position(to, ti, tpos, tt, tp, 0, kVocab, 0, true, ctx),
      c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] { (void)kops::embed::embedding_with_position(to, ti, tpos, tt, tp, 0, kVocab, 0, true, ctx); },
        st);
  }

  cudaFree(tok);
  cudaFree(pos);
  cudaFree(out);
  cudaFree(ids);
  cudaFree(positions);
  return ok;
}

template <typename T>
bool run_gather(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  T* out = nullptr;
  int32_t* rows = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), n * sizeof(T)), "cudaMalloc input") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(T)), "cudaMalloc out") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&rows), static_cast<size_t>(c.rows) * sizeof(int32_t)),
               "cudaMalloc rows")) {
    return false;
  }
  cudaMemset(in, 0, n * sizeof(T));
  cudaMemset(out, 0, n * sizeof(T));
  std::vector<int32_t> host_rows(static_cast<size_t>(c.rows));
  for (size_t i = 0; i < host_rows.size(); ++i) {
    host_rows[i] = static_cast<int32_t>((i * 7) % static_cast<size_t>(c.rows));
  }
  cudaMemcpy(rows, host_rows.data(), host_rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

  const kops::Tensor ti = tensor2d(in, c.dtype, c.rows, c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  const kops::Tensor tr = tensor1d(rows, kops::DType::kInt32, c.rows);

  bool ok = status_ok(kops::embed::gather_rows(to, ti, tr, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::embed::gather_rows(to, ti, tr, ctx); }, st);

  cudaFree(in);
  cudaFree(out);
  cudaFree(rows);
  return ok;
}

template <typename T>
bool run_scatter(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* src = nullptr;
  T* out = nullptr;
  int32_t* dst = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&src), n * sizeof(T)), "cudaMalloc src") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(T)), "cudaMalloc out") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&dst), static_cast<size_t>(c.rows) * sizeof(int32_t)),
               "cudaMalloc dst_rows")) {
    return false;
  }
  cudaMemset(src, 0, n * sizeof(T));
  cudaMemset(out, 0, n * sizeof(T));
  // A rotation is a permutation for every row count: duplicate destinations
  // are a write race, so they must never be benchmarked.
  std::vector<int32_t> host_dst(static_cast<size_t>(c.rows));
  for (size_t i = 0; i < host_dst.size(); ++i) {
    host_dst[i] = static_cast<int32_t>((i + 1) % static_cast<size_t>(c.rows));
  }
  cudaMemcpy(dst, host_dst.data(), host_dst.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

  const kops::Tensor ts = tensor2d(src, c.dtype, c.rows, c.cols);
  const kops::Tensor to = tensor2d(out, c.dtype, c.rows, c.cols);
  const kops::Tensor td = tensor1d(dst, kops::DType::kInt32, c.rows);

  bool ok = status_ok(kops::embed::scatter_rows(to, ts, td, ctx), c.id.c_str());
  if (ok) ok = measure(c, [&] { (void)kops::embed::scatter_rows(to, ts, td, ctx); }, st);

  cudaFree(src);
  cudaFree(out);
  cudaFree(dst);
  return ok;
}

template <typename T>
bool run_logits(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  const size_t n = static_cast<size_t>(c.rows * c.cols);
  T* in = nullptr;
  float* out = nullptr;
  if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&in), n * sizeof(T)), "cudaMalloc logits") ||
      !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&out), n * sizeof(float)), "cudaMalloc out")) {
    return false;
  }
  cudaMemset(in, 0, n * sizeof(T));
  cudaMemset(out, 0, n * sizeof(float));

  const kops::Tensor ti = tensor2d(in, c.dtype, c.rows, c.cols);
  const kops::Tensor to = tensor2d(out, kops::DType::kFloat32, c.rows, c.cols);
  // Gemma-2 softcap 30, Granite/uP scale: the tail 8 columns are TP padding.
  const float scale = 0.5f;
  const float softcap = 30.0f;
  const int64_t vocab_size = c.cols - 8;

  bool ok = status_ok(
      kops::embed::logits_postprocess(to, ti, scale, softcap, 0, vocab_size, ctx), c.id.c_str());
  if (ok) {
    ok = measure(
        c,
        [&] { (void)kops::embed::logits_postprocess(to, ti, scale, softcap, 0, vocab_size, ctx); },
        st);
  }

  cudaFree(in);
  cudaFree(out);
  return ok;
}

template <typename T>
bool run_typed(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  if (c.op == "embedding_lookup") return run_lookup<T>(c, ctx, st);
  if (c.op == "embedding_with_position") return run_with_position<T>(c, ctx, st);
  if (c.op == "gather_rows") return run_gather<T>(c, ctx, st);
  if (c.op == "scatter_rows") return run_scatter<T>(c, ctx, st);
  if (c.op == "logits_postprocess") return run_logits<T>(c, ctx, st);
  return false;
}

bool run_case(const Case& c, const kops::LaunchContext& ctx, Stats* st) {
  switch (c.dtype) {
    case kops::DType::kFloat32: return run_typed<float>(c, ctx, st);
    case kops::DType::kFloat16: return run_typed<__half>(c, ctx, st);
    case kops::DType::kBFloat16: return run_typed<__nv_bfloat16>(c, ctx, st);
    default: return false;
  }
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
  if (!status_ok(kops::make_launch_context(0, nullptr, &ctx), "make_launch_context")) {
    return 1;
  }

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
