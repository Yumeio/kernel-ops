// =============================================================================
// benchmark/compare/external/spec_trace_runner.cu               Dev-only runner
//
// Consumes the binary traces written by tools/dev/export_vllm_spec_trace.py and
// runs kernel-ops' B10 `rejection_sample` on the exact target/draft data that
// vLLM's rejection sampler used.  It reports accepted-length parity, greedy
// token parity and verifier latency.
//
// Build: bash tools/dev/build_external_baselines.sh traces
// Never built by CMake/ctest and never linked into the shipped library.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
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
#include "csrc/common/op_types.cuh"
#include "csrc/common/rng.cuh"
#include "csrc/sampling/sampling_kernels.cuh"
#include "csrc/sampling/fused_sampler_kernels.cuh"
#include "csrc/sampling/speculative_kernels.cuh"
#include "csrc/sampling/ops.h"

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kMagic = 0x42313054u;

struct Header {
  uint32_t magic, version, batch, num_tokens, vocab, max_spec, has_q;
};

struct Trace {
  Header h{};
  std::vector<int32_t> cu, draft, target_idx, bonus_idx, sampled, accepted;
  std::vector<float> target, draft_probs;
};

bool read_file(const fs::path& path, Trace* t) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  if (std::fread(&t->h, sizeof(t->h), 1, f) != 1 || t->h.magic != kMagic || t->h.version != 1) {
    std::fclose(f);
    return false;
  }
  const int B = static_cast<int>(t->h.batch);
  const int T = static_cast<int>(t->h.num_tokens);
  const int V = static_cast<int>(t->h.vocab);
  const int M = static_cast<int>(t->h.max_spec);
  t->cu.resize(B + 1);
  t->draft.resize(T);
  t->target_idx.resize(T);
  t->bonus_idx.resize(B);
  t->sampled.resize(static_cast<size_t>(B) * (M + 1));
  t->accepted.resize(B);
  t->target.resize(static_cast<size_t>(T + B) * V);
  auto rd = [f](void* p, size_t bytes) { return std::fread(p, 1, bytes, f) == bytes; };
  if (!rd(t->cu.data(), t->cu.size() * 4) || !rd(t->draft.data(), t->draft.size() * 4) ||
      !rd(t->target_idx.data(), t->target_idx.size() * 4) ||
      !rd(t->bonus_idx.data(), t->bonus_idx.size() * 4) ||
      !rd(t->target.data(), t->target.size() * 4)) {
    std::fclose(f);
    return false;
  }
  if (t->h.has_q != 0u) {
    t->draft_probs.resize(static_cast<size_t>(T) * V);
    if (!rd(t->draft_probs.data(), t->draft_probs.size() * 4)) {
      std::fclose(f);
      return false;
    }
  }
  if (!rd(t->sampled.data(), t->sampled.size() * 4) || !rd(t->accepted.data(), t->accepted.size() * 4)) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

template <typename T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(n, 1) * sizeof(T)) != cudaSuccess) {
    return nullptr;
  }
  return p;
}

void copy_in(void* dst, const void* src, size_t bytes) {
  cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const char* dir = nullptr;
  const char* json_path = nullptr;
  bool greedy = true;
  int iters = 50;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--dir") == 0) dir = argv[i + 1];
    if (std::strcmp(argv[i], "--json") == 0) json_path = argv[i + 1];
    if (std::strcmp(argv[i], "--prob") == 0) greedy = false;
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
  }
  if (dir == nullptr) {
    std::fprintf(stderr, "usage: %s --dir <trace-dir> [--json out.json] [--prob]\n", argv[0]);
    return 2;
  }
  kops::LaunchContext ctx{};
  if (kops::make_launch_context(0, nullptr, &ctx) != kops::Status::kSuccess) {
    std::fprintf(stderr, "make_launch_context failed\n");
    return 2;
  }

  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.is_regular_file() && e.path().extension() == ".bin") files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::fprintf(stderr, "no .bin traces in %s\n", dir);
    return 2;
  }

  int total_batches = 0;
  double sum_engine = 0.0, sum_kops = 0.0;
  int greedy_exact = 0, greedy_total = 0;
  std::vector<double> latencies;
  FILE* jf = json_path != nullptr ? std::fopen(json_path, "w") : nullptr;
  if (jf != nullptr) {
    std::fprintf(jf, "{\"dir\":\"%s\",\"mode\":\"%s\",\"traces\":[", dir, greedy ? "greedy" : "prob");
  }

  for (size_t fi = 0; fi < files.size(); ++fi) {
    Trace t{};
    if (!read_file(files[fi], &t)) {
      std::fprintf(stderr, "bad trace %s\n", files[fi].c_str());
      return 2;
    }
    const int B = static_cast<int>(t.h.batch);
    const int T = static_cast<int>(t.h.num_tokens);
    const int V = static_cast<int>(t.h.vocab);
    const int M = static_cast<int>(t.h.max_spec);
    const int rows = T + B;

    // target_argmax for greedy mode.
    std::vector<int32_t> argmax(rows, -1);
    for (int r = 0; r < rows; ++r) {
      int best = 0;
      for (int v = 1; v < V; ++v) {
        if (t.target[static_cast<size_t>(r) * V + v] > t.target[static_cast<size_t>(r) * V + best]) best = v;
      }
      argmax[static_cast<size_t>(r)] = best;
    }
    std::vector<uint8_t> flags(static_cast<size_t>(B), 1);
    int32_t* d_cu = dev_alloc<int32_t>(B + 1);
    int32_t* d_draft = dev_alloc<int32_t>(T);
    float* d_target = dev_alloc<float>(static_cast<size_t>(rows) * V);
    float* d_q = nullptr;
    int32_t* d_argmax = dev_alloc<int32_t>(rows);
    uint8_t* d_flags = dev_alloc<uint8_t>(B);
    int32_t* d_out = dev_alloc<int32_t>(static_cast<size_t>(B) * (M + 1));
    int32_t* d_num = dev_alloc<int32_t>(B);
    if (d_cu == nullptr || d_draft == nullptr || d_target == nullptr || d_argmax == nullptr ||
        d_flags == nullptr || d_out == nullptr || d_num == nullptr) return 2;
    if (!greedy) {
      d_q = dev_alloc<float>(static_cast<size_t>(T) * V);
      if (d_q == nullptr) return 2;
    }
    copy_in(d_cu, t.cu.data(), t.cu.size() * 4);
    copy_in(d_draft, t.draft.data(), t.draft.size() * 4);
    copy_in(d_target, t.target.data(), t.target.size() * 4);
    copy_in(d_argmax, argmax.data(), argmax.size() * 4);
    copy_in(d_flags, flags.data(), flags.size());
    if (!greedy) copy_in(d_q, t.draft_probs.data(), t.draft_probs.size() * 4);

    kops::sample::SpecVerifyParams p{};
    p.draft_token_ids = d_draft;
    p.cu_num_draft = d_cu;
    p.target_probs = d_target;
    p.draft_probs = greedy ? nullptr : d_q;
    p.target_argmax = d_argmax;
    p.is_greedy = greedy ? d_flags : nullptr;
    p.output_token_ids = d_out;
    p.num_accepted = d_num;
    p.batch_size = B;
    p.max_draft = M;
    p.vocab_size = V;
    p.rng.global_seed = 12345u;
    p.rng.stream_id = 1;

    const kops::Status st = kops::sample::rejection_sample(p, ctx);
    if (st != kops::Status::kSuccess) {
      std::fprintf(stderr, "%s: %s\n", files[fi].c_str(), kops::last_error_message());
      return 2;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return 2;
    const auto kops_num = [&] {
      std::vector<int32_t> h(B);
      cudaMemcpy(h.data(), d_num, B * 4, cudaMemcpyDeviceToHost);
      return h;
    }();
    const auto kops_out = [&] {
      std::vector<int32_t> h(static_cast<size_t>(B) * (M + 1));
      cudaMemcpy(h.data(), d_out, h.size() * 4, cudaMemcpyDeviceToHost);
      return h;
    }();

    int engine_sum = 0, kops_sum = 0;
    for (int b = 0; b < B; ++b) {
      engine_sum += t.accepted[static_cast<size_t>(b)];
      kops_sum += kops_num[static_cast<size_t>(b)];
      ++greedy_total;
      bool same = t.accepted[static_cast<size_t>(b)] == kops_num[static_cast<size_t>(b)];
      for (int j = 0; j < M + 1 && same; ++j) {
        const int32_t e = t.sampled[static_cast<size_t>(b) * (M + 1) + j];
        const int32_t k = kops_out[static_cast<size_t>(b) * (M + 1) + j];
        if (e >= 0 && e != k) same = false;
      }
      if (same) ++greedy_exact;
    }

    // Time the verifier.  Warm up, then one event pair per call.
    for (int i = 0; i < 5; ++i) (void)kops::sample::rejection_sample(p, ctx);
    cudaDeviceSynchronize();
    std::vector<double> ms;
    for (int i = 0; i < iters; ++i) {
      cudaEvent_t a = nullptr, b = nullptr;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      cudaEventRecord(a);
      (void)kops::sample::rejection_sample(p, ctx);
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float v = 0.0f;
      cudaEventElapsedTime(&v, a, b);
      ms.push_back(v);
      cudaEventDestroy(a);
      cudaEventDestroy(b);
    }
    const double med = median(ms);
    latencies.push_back(med);
    total_batches += B;
    sum_engine += engine_sum;
    sum_kops += kops_sum;

    if (jf != nullptr) {
      if (fi != 0) std::fprintf(jf, ",");
      std::fprintf(jf,
                   "{\"file\":\"%s\",\"batch\":%d,\"drafts\":%d,\"vocab\":%d,"
                   "\"engine_accepted\":%.4f,\"kops_accepted\":%.4f,\"kops_ms\":%.6f}",
                   files[fi].filename().c_str(), B, T / std::max(B, 1), V,
                   static_cast<double>(engine_sum) / B, static_cast<double>(kops_sum) / B, med);
    }
    cudaFree(d_cu);
    cudaFree(d_draft);
    cudaFree(d_target);
    if (d_q != nullptr) cudaFree(d_q);
    cudaFree(d_argmax);
    cudaFree(d_flags);
    cudaFree(d_out);
    cudaFree(d_num);
  }

  const double engine_mean = sum_engine / std::max(1, total_batches);
  const double kops_mean = sum_kops / std::max(1, total_batches);
  std::printf("traces=%zu batches=%d mode=%s engine_accept=%.4f kops_accept=%.4f match=%d/%d median_ms=%.6f\n",
              files.size(), total_batches, greedy ? "greedy" : "prob", engine_mean, kops_mean,
              greedy_exact, greedy_total, median(latencies));
  if (jf != nullptr) {
    std::fprintf(jf, "],\"engine_accept\":%.6f,\"kops_accept\":%.6f,"
                     "\"greedy_match\":%d,\"greedy_total\":%d,\"median_ms\":%.6f}\n",
                 engine_mean, kops_mean, greedy_exact, greedy_total, median(latencies));
    std::fclose(jf);
  }
  return 0;
}
