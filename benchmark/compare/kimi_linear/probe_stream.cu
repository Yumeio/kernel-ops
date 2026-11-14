// Weight-stream probe for the W4A8 GEMV loop of ayaka_mega.cu.
//
//   nvcc -O3 -arch=sm_90a -o probe_stream probe_stream.cu && ./probe_stream
//
// Measures GB/s of (a) a load-only pass and (b) the real dp4a GEMV loop over a
// warp-major arena, for several (CTAs per SM, warps per CTA, lookahead) shapes.
// The unit of work is the same as in the megakernel: one warp owns 128 output
// columns x one 128-element contraction group == 17 rows of 512 B
// (row 0 = packed {bf16 scale, int8 zero}, rows 1..16 = int4 weights).
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define ROWS 17
static __device__ __forceinline__ int dp4a_us(unsigned a, int b, int c) {
  int d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

template <int MODE, int LOOK>
__global__ void stream_kernel(const uint4* __restrict__ W0, long long units_all, int nkg,
                              const int* __restrict__ xq_g, float* __restrict__ out, long long su) {
  extern __shared__ int smem[];
  int* xq = smem;                       // [nkg][32] packed int8 activations
  float* xs = (float*)(smem + 64 * 32); // [nkg][2] (scale, sum)
  for (int i = threadIdx.x; i < nkg * 32; i += blockDim.x) xq[i] = xq_g[i];
  for (int i = threadIdx.x; i < nkg * 2; i += blockDim.x) xs[i] = 1.0f + i;
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int nw = blockDim.x >> 5;
  const long long gw = (long long)blockIdx.x * nw + (threadIdx.x >> 5);
  const long long tw = (long long)gridDim.x * nw;
  float v0 = 0, v1 = 0, v2 = 0, v3 = 0;
  for (long long sb = 0; sb < units_all; sb += su) {
  const uint4* W = W0 + sb * (ROWS * 32);
  const long long units = su;
  const long long u0 = units * gw / tw, u1 = units * (gw + 1) / tw;
  if (MODE == 0) {  // load only
    unsigned x = 0;
    for (long long u = u0; u < u1; ++u) {
      const uint4* p = W + u * (ROWS * 32) + lane;
#pragma unroll
      for (int m = 0; m < ROWS; ++m) {
        uint4 w = p[m * 32];
        x ^= w.x ^ w.y ^ w.z ^ w.w;
      }
    }
    v0 = (float)x;
  } else {
    uint4 nb[ROWS];
    if (LOOK) {
      const uint4* p = W + u0 * (ROWS * 32) + lane;
#pragma unroll
      for (int m = 0; m < ROWS; ++m) nb[m] = p[m * 32];
    }
    for (long long u = u0; u < u1; ++u) {
      uint4 cur[ROWS];
      if (LOOK) {
#pragma unroll
        for (int m = 0; m < ROWS; ++m) cur[m] = nb[m];
        if (u + 1 < u1) {
          const uint4* p = W + (u + 1) * (ROWS * 32) + lane;
#pragma unroll
          for (int m = 0; m < ROWS; ++m) nb[m] = p[m * 32];
        }
      } else {
        const uint4* p = W + u * (ROWS * 32) + lane;
#pragma unroll
        for (int m = 0; m < ROWS; ++m) cur[m] = p[m * 32];
      }
      const int kg = (int)(u % nkg);
      const int* xk = xq + kg * 32;
      int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll
      for (int m = 0; m < 16; ++m) {
        const uint4 w = cur[m + 1];
        const int xl = xk[2 * m], xh = xk[2 * m + 1];
        a0 = dp4a_us(w.x & 0x0F0F0F0Fu, xl, a0);
        a0 = dp4a_us((w.x >> 4) & 0x0F0F0F0Fu, xh, a0);
        a1 = dp4a_us(w.y & 0x0F0F0F0Fu, xl, a1);
        a1 = dp4a_us((w.y >> 4) & 0x0F0F0F0Fu, xh, a1);
        a2 = dp4a_us(w.z & 0x0F0F0F0Fu, xl, a2);
        a2 = dp4a_us((w.z >> 4) & 0x0F0F0F0Fu, xh, a2);
        a3 = dp4a_us(w.w & 0x0F0F0F0Fu, xl, a3);
        a3 = dp4a_us((w.w >> 4) & 0x0F0F0F0Fu, xh, a3);
      }
      const float f = xs[kg * 2], sm = xs[kg * 2 + 1];
      const uint4 sz = cur[0];
      v0 += (float)(sz.x & 0xffff) * f * ((float)a0 - (float)(int)(sz.x >> 16) * sm);
      v1 += (float)(sz.y & 0xffff) * f * ((float)a1 - (float)(int)(sz.y >> 16) * sm);
      v2 += (float)(sz.z & 0xffff) * f * ((float)a2 - (float)(int)(sz.z >> 16) * sm);
      v3 += (float)(sz.w & 0xffff) * f * ((float)a3 - (float)(int)(sz.w >> 16) * sm);
      if (MODE == 2 && kg == nkg - 1) {  // flush like the megakernel does at a tile edge
        float* o = out + ((u / nkg) & 127) * 128 + lane * 4;
        atomicAdd(o, v0); atomicAdd(o + 1, v1); atomicAdd(o + 2, v2); atomicAdd(o + 3, v3);
        v0 = v1 = v2 = v3 = 0;
      }
    }
  }
  }
  if (lane == 0 && threadIdx.x == 0) out[blockIdx.x & 1023] = v0 + v1 + v2 + v3;
}

__global__ void fill(unsigned* p, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += (long long)gridDim.x * blockDim.x) p[i] = (unsigned)(i * 2654435761u) ^ (unsigned)(i >> 7);
}

template <int MODE, int LOOK>
double run(const uint4* W, long long units, int nkg, const int* xq, float* out, int grid, int warps,
           int reps, long long su) {
  cudaFuncSetAttribute(stream_kernel<MODE, LOOK>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);
  for (int i = 0; i < 3; ++i) stream_kernel<MODE, LOOK><<<grid, warps * 32, 16384>>>(W, units, nkg, xq, out, su);
  cudaDeviceSynchronize();
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    stream_kernel<MODE, LOOK><<<grid, warps * 32, 16384>>>(W, units, nkg, xq, out, su);
    cudaDeviceSynchronize();
    double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    if (dt < best) best = dt;
  }
  if (cudaGetLastError() != cudaSuccess) return 1e30;
  return best;
}

int main(int argc, char** argv) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int nsm = prop.multiProcessorCount;
  const double gb = argc > 1 ? atof(argv[1]) : 2.0;
  const int nkg = 18;
  long long units = (long long)(gb * 1e9 / (ROWS * 512));
  const long long su = (argc > 2 ? atoll(argv[2]) : 2304);   // units per "stage" (2304 ~ 20 MB)
  units -= units % su;
  const size_t bytes = (size_t)units * ROWS * 512;
  uint4* W; int* xq; float* out;
  cudaMalloc(&W, bytes); cudaMalloc(&xq, 64 * 32 * 4); cudaMalloc(&out, 1 << 20);
  fill<<<nsm * 4, 256>>>((unsigned*)W, (long long)(bytes / 4));
  fill<<<1, 256>>>((unsigned*)xq, 64 * 32);
  cudaMemset(out, 0, 1 << 20);
  cudaDeviceSynchronize();
  printf("%s: %d SMs, arena %.2f GB, %lld units\n", prop.name, nsm, bytes / 1e9, units);
  printf("%-10s %-6s %-6s %10s %10s %10s %10s\n", "ctas/sm", "warps", "thr/sm", "load", "gemv", "gemv+look", "look+flush");
  const int cps[] = {1, 2, 4};
  const int wps[] = {4, 8, 16, 32};
  for (int c : cps)
    for (int w : wps) {
      if (c * w > 32) continue;
      const int grid = nsm * c;
      double a = run<0, 0>(W, units, nkg, xq, out, grid, w, 5, su);
      double b = run<1, 0>(W, units, nkg, xq, out, grid, w, 5, su);
      double d = run<1, 1>(W, units, nkg, xq, out, grid, w, 5, su);
      double e = run<2, 1>(W, units, nkg, xq, out, grid, w, 5, su);
      printf("%-10d %-6d %-6d %10.0f %10.0f %10.0f %10.0f\n", c, w, c * w * 32, bytes / a / 1e9,
             bytes / b / 1e9, bytes / d / 1e9, bytes / e / 1e9);
    }
  return 0;
}
