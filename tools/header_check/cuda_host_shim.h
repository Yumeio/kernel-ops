// =============================================================================
// kernel-ops | tools/header_check/cuda_host_shim.h
// Host-only stand-ins for CUDA keywords/types so the declaration headers can be
// syntax-checked with g++/clang++ on machines without nvcc:
//   g++ -std=c++17 -fsyntax-only -I. tools/header_check/check_all_headers.cc
// This file is a TOOL, not part of the library; the library headers themselves
// contain no #include directives.
// =============================================================================
#pragma once
#include <cstddef>
#include <cstdint>

#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__ __restrict
#define __grid_constant__
#define __launch_bounds__(...)

struct __half { unsigned short x; };
struct __nv_bfloat16 { unsigned short x; };
struct __nv_fp8_e4m3 { unsigned char x; };
struct __nv_fp8_e5m2 { unsigned char x; };
struct __nv_fp8_e8m0 { unsigned char x; };
struct __nv_fp4x2_e2m1 { unsigned char x; };

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUgraph_st* cudaGraph_t;
typedef struct CUgraphExec_st* cudaGraphExec_t;
typedef int cudaError_t;
struct alignas(64) CUtensorMap_st { unsigned long long opaque[16]; };
typedef CUtensorMap_st CUtensorMap;

struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct dim3 { unsigned x = 1, y = 1, z = 1; };
