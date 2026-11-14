// =============================================================================
// kernel-ops | tools/dev/check_grouped_spot.cu
// Dev-only fp64 spot check for grouped_gemm_wgmma_kernel at the bench_gemm shapes
// (contiguous + masked, G=8/32/128 including 1-row and empty groups).
//
// nvcc -std=c++17 -O2 -arch=sm_90a -I. tools/dev/check_grouped_spot.cu \
//   build/handoff-sm90a/libkops_cuda.a build/handoff-sm90a/libkops_host.a \
//   -lcudart -lcublas -lcuda -o /tmp/check_grouped && /tmp/check_grouped
// =============================================================================
// Builds A/B with small integers, runs the public grouped_gemm() entry with the
// same tactic the bench uses, and compares sampled outputs with a CPU reference.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/dense_gemm_kernels.cuh"
#include "csrc/gemm/fp8_gemm_kernels.cuh"
#include "csrc/gemm/int8_gemm_kernels.cuh"
#include "csrc/gemm/fp4_gemm_kernels.cuh"
#include "csrc/gemm/mixed_input_gemm_kernels.cuh"
#include "csrc/gemm/gguf_gemm_kernels.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/gemm/gemm_prep_kernels.cuh"
#include "csrc/gemm/ops.h"

namespace kg = kops::gemm;
using kops::DType;

static float hrand(uint64_t& s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<float>((s >> 33) % 3);
}

int main() {
  int fails = 0;
  struct Shape { int64_t g, r, n, k; };
  const Shape shapes[] = {{8, 1024, 4096, 4096}, {32, 4096, 2048, 2048}, {128, 100, 1024, 1024}};
  for (const Shape& s : shapes) {
    for (int masked = 0; masked < 2; ++masked) {
      const int64_t per = (s.r + s.g - 1) / s.g;
      const int64_t cap = masked ? per * s.g : s.r;
      std::vector<int64_t> rows(s.g);
      for (int64_t g = 0; g < s.g; ++g) rows[g] = s.r / s.g + (g < s.r % s.g ? 1 : 0);
      std::vector<__nv_bfloat16> a(static_cast<size_t>(cap * s.k));
      std::vector<__nv_bfloat16> b(static_cast<size_t>(s.g * s.n * s.k));
      uint64_t seed = 7;
      for (auto& v : a) v = __float2bfloat16(hrand(seed));
      for (auto& v : b) v = __float2bfloat16(hrand(seed));
      // NaN in the rows that belong to no group (contiguous spare rows are none here).
      void *da = nullptr, *db = nullptr, *dd = nullptr, *dmeta = nullptr;
      cudaMalloc(&da, a.size() * 2);
      cudaMalloc(&db, b.size() * 2);
      cudaMalloc(&dd, static_cast<size_t>(cap * s.n) * 2);
      cudaMalloc(&dmeta, static_cast<size_t>(s.g + 1) * 8);
      cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice);
      cudaMemcpy(db, b.data(), b.size() * 2, cudaMemcpyHostToDevice);
      if (masked) {
        std::vector<int32_t> m(static_cast<size_t>(s.g));
        for (int64_t g = 0; g < s.g; ++g) m[g] = static_cast<int32_t>(rows[g]);
        cudaMemcpy(dmeta, m.data(), m.size() * 4, cudaMemcpyHostToDevice);
      } else {
        std::vector<int64_t> off(static_cast<size_t>(s.g + 1));
        int64_t acc = 0;
        for (int64_t g = 0; g < s.g; ++g) { off[g] = acc; acc += rows[g]; }
        off[s.g] = acc;
        cudaMemcpy(dmeta, off.data(), off.size() * 8, cudaMemcpyHostToDevice);
      }
      kg::GroupedGemmParams p{};
      p.layout = masked ? kg::GroupedLayout::kMasked : kg::GroupedLayout::kContiguous;
      p.recipe = kg::GemmRecipe::kBF16;
      p.num_groups = static_cast<int32_t>(s.g);
      p.a = da; p.lda = s.k;
      p.b = db; p.ldb = s.k; p.b_group_stride = s.n * s.k;
      p.d = dd; p.ldd = s.n;
      p.n = static_cast<int32_t>(s.n); p.k = static_cast<int32_t>(s.k);
      p.max_m = static_cast<int32_t>(masked ? per : cap);
      if (masked) p.masked_m = static_cast<const int32_t*>(dmeta);
      else p.expert_offsets = static_cast<const int64_t*>(dmeta);
      kg::GemmTactic t{};
      t.tile_m = 128; t.tile_n = 128; t.tile_k = 64;
      t.mainloop = kg::MainloopSchedule::kTmaWarpSpecializedCooperative;
      t.epilogue = kg::EpilogueSchedule::kTmaStore;
      t.persistent = true;
      kops::LaunchContext ctx{};
      if (kops::make_launch_context(0, nullptr, &ctx) != kops::Status::kSuccess) return 2;
      const kops::Status st = kg::grouped_gemm(p, &t, kg::GemmBackend::kNative, kops::Workspace{}, ctx);
      if (ctx.stream == nullptr) cudaDeviceSynchronize();
      if (st != kops::Status::kSuccess) { std::printf("FAIL launch %d\n", static_cast<int>(st)); ++fails; continue; }
      cudaDeviceSynchronize();
      const cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) { std::printf("FAIL cuda %s\n", cudaGetErrorString(err)); ++fails; continue; }
      std::vector<__nv_bfloat16> d(static_cast<size_t>(cap * s.n));
      cudaMemcpy(d.data(), dd, d.size() * 2, cudaMemcpyDeviceToHost);
      // Sample: every group, first/last valid row, 8 columns.
      int64_t bad = 0, checked = 0;
      for (int64_t g = 0; g < s.g; ++g) {
        if (rows[g] == 0) continue;
        const int64_t base = masked ? g * per : 0;
        for (int64_t rr : {int64_t{0}, rows[g] - 1}) {
          const int64_t row = masked ? base + rr : [&] {
            int64_t acc = 0;
            for (int64_t gg = 0; gg < g; ++gg) acc += rows[gg];
            return acc + rr;
          }();
          for (int64_t j = 0; j < s.n; j += s.n / 8) {
            double ref = 0.0;
            const __nv_bfloat16* ar = &a[static_cast<size_t>(row * s.k)];
            const __nv_bfloat16* br = &b[static_cast<size_t>((g * s.n + j) * s.k)];
            for (int64_t kk = 0; kk < s.k; ++kk)
              ref += static_cast<double>(__bfloat162float(ar[kk])) * __bfloat162float(br[kk]);
            const double got = __bfloat162float(d[static_cast<size_t>(row * s.n + j)]);
            ++checked;
            if (std::fabs(got - ref) > 0.02 * std::max(1.0, std::fabs(ref))) {
              if (bad < 3) std::printf("  mismatch g%lld row%lld col%lld ref %.1f got %.1f\n",
                                       static_cast<long long>(g), static_cast<long long>(rr),
                                       static_cast<long long>(j), ref, got);
              ++bad;
            }
          }
        }
      }
      std::printf("G%-4lld R%-5lld N%-5lld K%-5lld %-8s checked %3lld bad %lld\n",
                  static_cast<long long>(s.g), static_cast<long long>(s.r), static_cast<long long>(s.n),
                  static_cast<long long>(s.k), masked ? "masked" : "contig",
                  static_cast<long long>(checked), static_cast<long long>(bad));
      fails += bad != 0;
      // timing: 20 launches, median
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      std::vector<float> ts;
      for (int i = 0; i < 25; ++i) {
        cudaEventRecord(e0);
        kg::grouped_gemm(p, &t, kg::GemmBackend::kNative, kops::Workspace{}, ctx);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        if (i >= 5) ts.push_back(ms);
      }
      std::sort(ts.begin(), ts.end());
      std::printf("  time %.4f ms -> %.2f TB/s (bytes %.1f MB)\n", ts[ts.size()/2],
                  (double)(s.r * s.k * 2 + s.g * s.n * s.k * 2 + s.r * s.n * 2) / (ts[ts.size()/2] * 1e9),
                  (double)(s.r * s.k * 2 + s.g * s.n * s.k * 2 + s.r * s.n * 2) / 1e6);
      cudaFree(da); cudaFree(db); cudaFree(dd); cudaFree(dmeta);
    }
  }
  std::printf(fails == 0 ? "PASS grouped spot check\n" : "FAIL grouped spot check\n");
  return fails == 0 ? 0 : 1;
}
