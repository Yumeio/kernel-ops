// =============================================================================
// kernel-ops | benchmark/compare/external/nccl_baseline.cu          (dev-only, never built by CMake)
// NCCL baseline for bench_communication on the same two GPUs, same case ids, same timing protocol
// (K calls per sample, CUDA graph per rank or eager launches; time on rank 0's stream / K).
//
//   bench_communication --dump-cases > out/cases.json
//   nccl_baseline --cases out/cases.json --json out/nccl.json [--mode graph|eager] [--only substr]
//   python benchmark/compare/compare_results.py out/kops-comm.json out/nccl.json
//
// Ops: allreduce, allgather, reduce_scatter (ncclAllReduce / ncclAllGather / ncclReduceScatter),
//      allreduce_fusion (NCCL all-reduce, then a fused residual-add + RMSNorm kernel written here with the
//      same arithmetic as csrc/norm), ep_dispatch_combine (ncclSend / ncclRecv all-to-all of the routed
//      tokens, DeepEP is not installed), allgather_gemm / gemm_reduce_scatter / gemm_allreduce (cuBLAS GEMM
//      then NCCL, sequential: the Gate of B12 compares against exactly this).
// Build: tools/dev/build_comm_baselines.sh
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(1);} } while (0)
#define NK(x) do { ncclResult_t e_ = (x); if (e_ != ncclSuccess) { std::fprintf(stderr, "NCCL %s @%d: %s\n", #x, __LINE__, ncclGetErrorString(e_)); std::exit(1);} } while (0)
#define BK(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cuBLAS %s @%d: %d\n", #x, __LINE__, (int)e_); std::exit(1);} } while (0)

struct CaseSpec { std::string id, op, dtype; int64_t rows = 0, cols = 0, bytes = 0, flops = 0; int warmup = 0, iters = 0; std::vector<int64_t> extra; };

static int64_t field_i(const std::string& obj, const char* key) {
  const std::string k = std::string("\"") + key + "\":";
  const size_t p = obj.find(k);
  return p == std::string::npos ? 0 : std::atoll(obj.c_str() + p + k.size());
}
static std::string field_s(const std::string& obj, const char* key) {
  const std::string k = std::string("\"") + key + "\":\"";
  const size_t p = obj.find(k);
  if (p == std::string::npos) return "";
  const size_t b = p + k.size();
  return obj.substr(b, obj.find('"', b) - b);
}
static std::vector<CaseSpec> load_cases(const char* path) {
  std::ifstream f(path);
  std::stringstream ss; ss << f.rdbuf();
  const std::string all = ss.str();
  std::vector<CaseSpec> out;
  size_t pos = all.find("\"cases\":[");
  while ((pos = all.find("{\"id\":", pos)) != std::string::npos) {
    const size_t end = all.find('}', pos);
    const std::string obj = all.substr(pos, end - pos + 1);
    CaseSpec c; c.id = field_s(obj, "id"); c.op = field_s(obj, "op"); c.dtype = field_s(obj, "dtype");
    c.rows = field_i(obj, "rows"); c.cols = field_i(obj, "cols"); c.bytes = field_i(obj, "bytes");
    c.flops = field_i(obj, "flops"); c.warmup = (int)field_i(obj, "warmup"); c.iters = (int)field_i(obj, "iters");
    out.push_back(c); pos = end;
  }
  return out;
}

// ---- fused residual-add + RMSNorm (the composite baseline after NCCL all-reduce) -----------------
__global__ void add_rmsnorm_kernel(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ res, const __nv_bfloat16* __restrict__ w, int hidden, float eps) {
  __shared__ float red[32];
  __nv_bfloat16* xr = x + (size_t)blockIdx.x * hidden; __nv_bfloat16* rr = res + (size_t)blockIdx.x * hidden;
  float ss = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) { float v = __bfloat162float(rr[i]) + __bfloat162float(xr[i]); rr[i] = __float2bfloat16_rn(v); ss += v * v; }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float t = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : 0.f;
  for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
  __shared__ float s_rstd; if (threadIdx.x == 0) s_rstd = rsqrtf(t / hidden + eps);
  __syncthreads();
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) xr[i] = __float2bfloat16_rn(__bfloat162float(rr[i]) * s_rstd * __bfloat162float(w[i]));
}

struct Ctx {
  int n = 2; ncclComm_t comm[2]; cudaStream_t s[2]; cublasHandle_t blas[2];
  void* a[2]; void* b[2]; void* c[2]; void* d[2]; void* e[2]; size_t cap = 0;
  void init(size_t bytes) {
    int devs[2] = {0, 1};
    NK(ncclCommInitAll(comm, 2, devs));
    cap = bytes;
    for (int k = 0; k < 2; ++k) {
      CK(cudaSetDevice(k)); CK(cudaStreamCreateWithFlags(&s[k], cudaStreamNonBlocking)); BK(cublasCreate(&blas[k])); BK(cublasSetStream(blas[k], s[k]));
      void** bufs[5] = {&a[k], &b[k], &c[k], &d[k], &e[k]};
      for (void** p : bufs) { CK(cudaMalloc(p, cap)); CK(cudaMemset(*p, 0x3c, cap)); }
    }
  }
};
static Ctx g;
static int g_calls = 16; static bool g_graph = true;

static bool measure(const CaseSpec& c, const std::function<void(int)>& issue, double out[5]) {
  std::vector<cudaGraphExec_t> exec(2, nullptr); std::vector<cudaGraph_t> graph(2, nullptr);
  // NCCL group calls spanning both devices must be issued from one thread: issue_all(call) wraps the group
  auto issue_all = [&]() { NK(ncclGroupStart()); for (int k = 0; k < 2; ++k) { CK(cudaSetDevice(k)); issue(k); } NK(ncclGroupEnd()); };
  if (g_graph) {
    for (int k = 0; k < 2; ++k) { CK(cudaSetDevice(k)); CK(cudaStreamBeginCapture(g.s[k], cudaStreamCaptureModeGlobal)); }
    for (int j = 0; j < g_calls; ++j) issue_all();
    for (int k = 0; k < 2; ++k) { CK(cudaSetDevice(k)); CK(cudaStreamEndCapture(g.s[k], &graph[k])); }
    for (int k = 0; k < 2; ++k) { CK(cudaSetDevice(k)); CK(cudaGraphInstantiate(&exec[k], graph[k], 0)); }
  }
  auto run_group = [&]() {
    if (g_graph) { for (int k = 0; k < 2; ++k) { CK(cudaSetDevice(k)); CK(cudaGraphLaunch(exec[k], g.s[k])); } }
    else { for (int j = 0; j < g_calls; ++j) issue_all(); }
  };
  auto sync = [&]() { for (int k = 0; k < 2; ++k) { CK(cudaSetDevice(k)); CK(cudaStreamSynchronize(g.s[k])); } };
  for (int i = 0; i < std::max(2, c.warmup / g_calls); ++i) run_group();
  sync();
  cudaEvent_t e0, e1; CK(cudaSetDevice(0)); cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int groups = std::max(8, c.iters / g_calls);
  std::vector<float> samples;
  for (int i = 0; i < groups; ++i) {
    CK(cudaSetDevice(0)); cudaEventRecord(e0, g.s[0]); run_group(); CK(cudaSetDevice(0)); cudaEventRecord(e1, g.s[0]); sync();
    float ms; cudaEventElapsedTime(&ms, e0, e1); samples.push_back(ms / g_calls);
  }
  std::sort(samples.begin(), samples.end()); const size_t n = samples.size();
  out[0] = samples[n / 2]; out[1] = samples[n / 10]; out[2] = samples[std::min(n - 1, n * 9 / 10)]; out[3] = samples[std::min(n - 1, n * 99 / 100)]; out[4] = samples[0];
  for (int k = 0; k < 2; ++k) { if (exec[k]) cudaGraphExecDestroy(exec[k]); if (graph[k]) cudaGraphDestroy(graph[k]); }
  return true;
}

int main(int argc, char** argv) {
  const char* cases_path = nullptr; const char* json_path = nullptr; const char* only = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--cases") && i + 1 < argc) cases_path = argv[++i];
    else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
    else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) g_graph = std::strcmp(argv[++i], "eager") != 0;
    else if (!std::strcmp(argv[i], "--calls") && i + 1 < argc) g_calls = std::max(1, std::atoi(argv[++i]));
  }
  if (!cases_path) { std::fprintf(stderr, "usage: nccl_baseline --cases cases.json [--json out.json] [--mode graph|eager]\n"); return 2; }
  std::vector<CaseSpec> cases = load_cases(cases_path);
  size_t maxb = 0; for (auto& c : cases) maxb = std::max<size_t>(maxb, (size_t)c.bytes * 4 + (size_t)c.rows * c.cols * 4);
  g.init(maxb + (1 << 20));
  FILE* f = json_path ? std::fopen(json_path, "w") : nullptr;
  if (f) std::fprintf(f, "{\"schema\":1,\"backend\":\"nccl\",\"results\":[");
  bool first = true;
  const cublasComputeType_t ct = CUBLAS_COMPUTE_32F;
  for (CaseSpec& c : cases) {
    if (only && c.id.find(only) == std::string::npos) continue;
    const size_t n = (size_t)c.rows * c.cols;
    std::function<void(int)> issue;
    if (c.op == "allreduce") {
      issue = [&](int k) { NK(ncclAllReduce(g.a[k], g.b[k], n, ncclBfloat16, ncclSum, g.comm[k], g.s[k])); };
    } else if (c.op == "allgather") {
      issue = [&](int k) { NK(ncclAllGather(g.a[k], g.b[k], n, ncclBfloat16, g.comm[k], g.s[k])); };
    } else if (c.op == "reduce_scatter") {
      issue = [&](int k) { NK(ncclReduceScatter(g.a[k], g.b[k], n, ncclBfloat16, ncclSum, g.comm[k], g.s[k])); };
    } else if (c.op == "allreduce_fusion") {
      issue = [&](int k) {
        NK(ncclAllReduce(g.a[k], g.b[k], n, ncclBfloat16, ncclSum, g.comm[k], g.s[k]));
        add_rmsnorm_kernel<<<(unsigned)c.rows, std::min<int>(512, (int)((c.cols / 8 + 31) / 32 * 32)), 0, g.s[k]>>>((__nv_bfloat16*)g.b[k], (__nv_bfloat16*)g.c[k], (__nv_bfloat16*)g.d[k], (int)c.cols, 1e-6f);
      };
    } else if (c.op == "allgather_gemm" || c.op == "gemm_reduce_scatter" || c.op == "gemm_allreduce") {
      // rows = M_total (tokens), cols = K, extra N via bytes field: bytes = 2 * M * N encodes the output size
      const int64_t M = c.rows, K = c.cols, N = c.bytes / (2 * c.rows);
      issue = [&, M, K, N](int k) {
        const float alpha = 1.f, beta = 0.f;
        // D[M, N] = A[M, K] * B[N, K]^T in row-major == column-major D^T[N, M] = B[N, K] (op T) * A^T
        if (c.op == "allgather_gemm") {
          const int64_t Ml = M / 2;
          NK(ncclAllGather(g.a[k], g.b[k], (size_t)(Ml * K), ncclBfloat16, g.comm[k], g.s[k]));  // gathered A in b
          BK(cublasGemmEx(g.blas[k], CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha, g.c[k], CUDA_R_16BF, (int)K, g.b[k], CUDA_R_16BF, (int)K, &beta, g.d[k], CUDA_R_16BF, (int)N, ct, CUBLAS_GEMM_DEFAULT));
        } else {
          BK(cublasGemmEx(g.blas[k], CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)(K), &alpha, g.c[k], CUDA_R_16BF, (int)K, g.a[k], CUDA_R_16BF, (int)K, &beta, g.d[k], CUDA_R_16BF, (int)N, ct, CUBLAS_GEMM_DEFAULT));
          if (c.op == "gemm_allreduce") NK(ncclAllReduce(g.d[k], g.e[k], (size_t)(M * N), ncclBfloat16, ncclSum, g.comm[k], g.s[k]));
          else NK(ncclReduceScatter(g.d[k], g.e[k], (size_t)(M * N / 2), ncclBfloat16, ncclSum, g.comm[k], g.s[k]));
        }
      };
    } else {
      std::fprintf(stderr, "[nccl] skipping %s (op %s)\n", c.id.c_str(), c.op.c_str());
      continue;
    }
    double st[5];
    measure(c, issue, st);
    std::printf("%-44s median %9.2f us\n", c.id.c_str(), st[0] * 1000.0);
    if (f) {
      std::fprintf(f, "%s{\"id\":\"%s\",\"op\":\"%s\",\"dtype\":\"%s\",\"rows\":%lld,\"cols\":%lld,\"bytes\":%lld,\"flops\":%lld,\"median_ms\":%.6f,\"p10_ms\":%.6f,\"p90_ms\":%.6f,\"p99_ms\":%.6f,\"min_ms\":%.6f,\"gbs\":%.2f,\"tflops\":0}",
                   first ? "" : ",", c.id.c_str(), c.op.c_str(), c.dtype.c_str(), (long long)c.rows, (long long)c.cols, (long long)c.bytes, (long long)c.flops, st[0], st[1], st[2], st[3], st[4], c.bytes / (st[0] * 1e6));
      first = false;
    }
  }
  if (f) { std::fprintf(f, "]}\n"); std::fclose(f); }
  return 0;
}
