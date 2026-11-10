// =============================================================================
// kernel-ops | csrc/quantization/weight_repack.cu
// B4 rows 1-4 native implementation (load-time weight conversion):
//   Q02/Q03  gptq_to_marlin, awq_to_marlin (tile repack, scale permutation, AWQ zero
//            points), awq_dequantize, gguf_dequantize
//   FP4/FP8  fp4_to_marlin, fp4_backend_shuffle, fp8_block_scale_to_ue8m0
//   Q04      smoothquant_compute_scales, smoothquant_fold
//
// Contract: csrc/quantization/ops.h + csrc/quantization/weight_repack_kernels.cuh.
// These run once after the checkpoint is loaded; nothing allocates, synchronizes or
// keeps workspace.  Every kernel is architecture-generic.
//
// Provenance (THIRD_PARTY_NOTICES.md, "Derived implementations"): the Marlin tile/lane mapping and
// pack orders are re-expressed from vLLM gptq_marlin_repack.cu / awq_marlin_repack.cu (Apache-2.0,
// 3rd/vllm, commit unknown); GGUF block layouts and dequantization formulas from ggml-quants.c
// (MIT) via 3rd/sglang sgl-kernel/csrc/quantization/gguf @ 96a04cb and, for the codebook formats
// (IQ1/IQ2/IQ3), straight from llama.cpp @ 0ec4027 (ggml/src/ggml-quants.c, ggml-common.h); UE8M0
// rounding from DeepGEMM deep_gemm/utils/math.py @ 5081f85 (MIT).  The TRT-LLM-gen row shuffle of
// fp4_backend_shuffle follows TensorRT-LLM @ 09198b7 (Apache-2.0): tensorrt_llm/quantization/utils/
// fp4_utils.py (shuffle_matrix_a), cpp/tensorrt_llm/thop/moe/trtllmGen/fp4BlockScaleMoe.cpp
// (shuffleMatrix) and cpp/tensorrt_llm/kernels/quantization.cu (block_scale_interleave).  No upstream
// source text is copied except the llama.cpp codebook tables in the block marked
// "adapted-from-llama.cpp" (MIT, notice in THIRD_PARTY_NOTICES.md).
//
// Marlin layout (the oracle in tests/unit/test_weight_repack.cc re-derives it from the
// numpy `get_perms` / `marlin_weights` formulation used by vLLM, independently of the
// per-lane formulation below):
//   * weights are cut into 16 (K) x 64 (N) tiles, tile order [k_tile][n_tile];
//   * lane (th_id, warp) of a tile owns n = warp*16 + th_id/4 (and +8) and the k rows
//     {2*(th_id%4) + 0, +1, +8, +9};
//   * 4-bit: one uint32 per (th_id, warp) = nibbles in order {0,2,4,6,1,3,5,7} of the eight
//     values; 8-bit: two uint32 per (th_id, warp) with byte order {0,2,1,3}.
//   * scales: 64-wide chunks permuted i + 8*j <- 8*i + j (grouped) or 32-wide chunks with
//     offsets {0,1,8,9,16,17,24,25} (per channel).
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/quantization/quant_kernels.cuh"
#include "csrc/quantization/weight_repack_kernels.cuh"
#include "csrc/quantization/ops.h"
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

namespace kops::quant {
namespace {

using dev::from_f;
using dev::to_f;

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

template <typename T>
T* mut_ptr(const Tensor& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

template <typename T>
const T* const_ptr(const Tensor& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data) + t.byte_offset);
}

const void* data_of(const Tensor& t) { return static_cast<const char*>(t.data) + t.byte_offset; }

cudaStream_t stream_of(const LaunchContext& ctx) {
  return reinterpret_cast<cudaStream_t>(ctx.stream);
}

Status backend_ok(const LaunchContext& ctx) {
  if (ctx.backend != BackendKind::kCuda) {
    return detail::set_error(Status::kUnsupportedBackend, "weight_repack: CUDA backend only");
  }
  return Status::kSuccess;
}

Status launch_error(const char* what, cudaError_t err) {
  return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(err), "%s: %s", what,
                                   cudaGetErrorString(err));
}

Status last_launch(const char* what) {
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? Status::kSuccess : launch_error(what, err);
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

bool aligned(const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

int byte_size(DType t) {
  const int bits = dtype_bits(t);
  return (bits > 0 && bits % 8 == 0) ? bits / 8 : 0;
}

bool is_float_scalar(DType t) {
  return t == DType::kFloat32 || t == DType::kFloat16 || t == DType::kBFloat16;
}

/// Contiguous tensor of the given rank whose dtype is `dtype` (kInvalid = any).
Status check_dense(const Tensor& t, DType dtype, int ndim, size_t align, const char* name) {
  Status s = check_tensor(t, dtype, ndim, align, name);
  if (s != Status::kSuccess) return s;
  if (!tensor_is_contiguous(t)) {
    return detail::set_error(Status::kUnsupportedLayout, "%s: contiguous tensor required", name);
  }
  return Status::kSuccess;
}

Status shape_is(const Tensor& t, std::initializer_list<int64_t> dims, const char* name) {
  int d = 0;
  for (const int64_t want : dims) {
    if (d >= t.ndim || t.shape[d] != want) {
      return detail::set_error(Status::kInvalidArgument,
                               "%s: shape[%d] = %lld, expected %lld", name, d,
                               static_cast<long long>(d < t.ndim ? t.shape[d] : -1),
                               static_cast<long long>(want));
    }
    ++d;
  }
  return Status::kSuccess;
}

// -----------------------------------------------------------------------------
// Marlin constants
// -----------------------------------------------------------------------------

constexpr int kMarlinTileK = 16;
constexpr int kMarlinTileN = 64;

__constant__ int kMarlinPack4[8] = {0, 2, 4, 6, 1, 3, 5, 7};
__constant__ int kMarlinPack8[4] = {0, 2, 1, 3};
__constant__ int kScalePerChannelOffsets[8] = {0, 1, 8, 9, 16, 17, 24, 25};

/// AWQ packs column c of an 8-column (4-bit) / 4-column (8-bit) word at nibble/byte
/// {0,2,4,6,1,3,5,7}^-1 = {0,4,1,5,2,6,3,7} / {0,2,1,3}.
template <int kBits>
__device__ __forceinline__ int awq_position(int c) {
  return kBits == 4 ? (((c & 1) << 2) | (c >> 1)) : (((c & 1) << 1) | (c >> 1));
}

// -----------------------------------------------------------------------------
// Weight sources: value(k, n) of the logical [K, N] b-bit matrix
// -----------------------------------------------------------------------------

/// GPTQ: qweight [K/pack, N] uint32, row k packed in word k/pack at position k%pack.
/// `perm` (optional) makes logical row k read original row perm[k] (act-order).
template <int kBits>
struct GptqSource {
  const uint32_t* qweight;
  const int32_t* perm;
  int64_t size_n;
  __device__ __forceinline__ uint32_t get(int k, int n) const {
    constexpr int kPack = 32 / kBits;
    constexpr uint32_t kMask = (1u << kBits) - 1u;
    const int kk = perm != nullptr ? perm[k] : k;
    const uint32_t w = qweight[static_cast<int64_t>(kk / kPack) * size_n + n];
    return (w >> (kBits * (kk % kPack))) & kMask;
  }
};

/// AWQ: qweight [K, N/pack] uint32, column n at nibble/byte awq_position(n % pack) of word n/pack.
template <int kBits>
struct AwqSource {
  const uint32_t* qweight;
  int64_t n_words;
  __device__ __forceinline__ uint32_t get(int k, int n) const {
    constexpr int kPack = 32 / kBits;
    constexpr uint32_t kMask = (1u << kBits) - 1u;
    const uint32_t w = qweight[static_cast<int64_t>(k) * n_words + n / kPack];
    return (w >> (kBits * awq_position<kBits>(n & (kPack - 1)))) & kMask;
  }
};

/// FP4: weight [N, K/2] bytes, element k of row n in nibble k&1 (low nibble = even k).
struct Fp4Source {
  const uint8_t* weight;
  int64_t row_bytes;
  __device__ __forceinline__ uint32_t get(int k, int n) const {
    const uint32_t b = weight[static_cast<int64_t>(n) * row_bytes + (k >> 1)];
    return (k & 1) ? (b >> 4) : (b & 0xFu);
  }
};

/// Work folded into a repack launch instead of a second kernel: the scale permutation and the
/// identity activation permutation.  Both are tiny next to the repack's own traffic (a 4096x4096
/// layer permutes 512 KB of scales and writes a 16 KB identity) but each costs a launch, and a
/// launch serializes behind the repack.  An all-default epilogue means "nothing fused".
/// The element access mirrors `launch_permute_scales` and `identity_perm_kernel`.
struct RepackEpilogue {
  void* scales_out = nullptr;
  const void* scales_in = nullptr;
  int64_t scale_total = 0;
  int scale_bytes = 0;
  bool per_channel = false;
  int32_t* perm_out = nullptr;
  int64_t perm_size = 0;
  __host__ __device__ bool empty() const { return scale_total <= 0 && perm_out == nullptr; }
};

/// out[i] = in[i - i % C + perm(i % C)]; see marlin_permute_scales_kernel for the two orders.
template <typename T>
__device__ __forceinline__ void marlin_permute_scales_slice(const RepackEpilogue& epi, const int64_t gtid,
                                                            const int64_t stride) {
  T* out = static_cast<T*>(epi.scales_out);
  const T* in = static_cast<const T*>(epi.scales_in);
  const int chunk = epi.per_channel ? 32 : 64;
  for (int64_t i = gtid; i < epi.scale_total; i += stride) {
    const int p = static_cast<int>(i % chunk);
    const int src = epi.per_channel ? 2 * (p >> 3) + kScalePerChannelOffsets[p & 7] : (p >> 3) + 8 * (p & 7);
    out[i] = in[i - p + src];
  }
}

/// Every thread of the launch takes its share by global thread id.
__device__ __forceinline__ void marlin_repack_epilogue(const RepackEpilogue& epi) {
  if (epi.empty()) return;
  const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  if (epi.perm_out != nullptr) {
    for (int64_t i = gtid; i < epi.perm_size; i += stride) epi.perm_out[i] = static_cast<int32_t>(i);
  }
  switch (epi.scale_bytes) {
    case 1: marlin_permute_scales_slice<uint8_t>(epi, gtid, stride); break;
    case 2: marlin_permute_scales_slice<uint16_t>(epi, gtid, stride); break;
    case 4: marlin_permute_scales_slice<uint32_t>(epi, gtid, stride); break;
    default: break;
  }
}

/// One thread per (tile, th_id): it produces the words of all four warps of its lane, which are
/// contiguous in the output (16 B for 4-bit, 32 B for 8-bit).  `kKFastest` walks the tiles of one
/// n-column strip first (better sector reuse for sources that are row-major in N, such as FP4).
template <int kBits, bool kKFastest, typename Source>
__global__ void marlin_repack_tiles_kernel(uint32_t* __restrict__ out, const Source src,
                                           const int k_tiles, const int n_tiles,
                                           const int64_t total_threads, const RepackEpilogue epi) {
  marlin_repack_epilogue(epi);
  constexpr int kTileWords = kMarlinTileK * kMarlinTileN / (32 / kBits);  // 128 or 256
  constexpr int kLaneWords = kTileWords / 32;                              // 4 or 8
  constexpr int kOffsets[4] = {0, 1, 8, 9};
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= total_threads) return;
  const int th_id = static_cast<int>(t & 31);
  const int64_t tile_iter = t >> 5;
  int kt;
  int nt;
  if (kKFastest) {
    kt = static_cast<int>(tile_iter % k_tiles);
    nt = static_cast<int>(tile_iter / k_tiles);
  } else {
    nt = static_cast<int>(tile_iter % n_tiles);
    kt = static_cast<int>(tile_iter / n_tiles);
  }
  const int k_base = kt * kMarlinTileK + (th_id & 3) * 2;
  const int n_base = nt * kMarlinTileN + (th_id >> 2);

  uint32_t res[kLaneWords];
  #pragma unroll
  for (int w = 0; w < 4; ++w) {
    uint32_t vals[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      vals[i] = src.get(k_base + kOffsets[i], n_base + w * 16);
      vals[4 + i] = src.get(k_base + kOffsets[i], n_base + w * 16 + 8);
    }
    if constexpr (kBits == 4) {
      constexpr int kPack4[8] = {0, 2, 4, 6, 1, 3, 5, 7};
      uint32_t r = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) r |= vals[kPack4[i]] << (i * 4);
      res[w] = r;
    } else {
      constexpr int kPack8[4] = {0, 2, 1, 3};
      uint32_t r1 = 0;
      uint32_t r2 = 0;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        r1 |= vals[kPack8[i]] << (i * 8);
        r2 |= vals[4 + kPack8[i]] << (i * 8);
      }
      res[2 * w] = r1;
      res[2 * w + 1] = r2;
    }
  }
  uint4* dst = reinterpret_cast<uint4*>(out + (static_cast<int64_t>(kt) * n_tiles + nt) * kTileWords +
                                        th_id * kLaneWords);
  dst[0] = make_uint4(res[0], res[1], res[2], res[3]);
  if constexpr (kBits == 8) dst[1] = make_uint4(res[4], res[5], res[6], res[7]);
}

/// Shared-memory staging for the 4-bit sources whose words run along K: a Marlin tile reads only
/// the two source rows 2*kt and 2*kt+1 at columns [nt*64, nt*64+64), so one warp can stage its own
/// 512 B with a single 16-byte load per lane (row 0 in lanes 0-15, row 1 in lanes 16-31) and pack
/// from there.  The 64 + 4 word stride puts the row-1 reads four banks after row 0's.
constexpr int kRepackStageStride = kMarlinTileN + 4;

/// One warp per tile for GPTQ-4bit without act-order.  Every source word feeds four output words,
/// so the previous eight scattered 4-byte loads per output word become one coalesced 16-byte load
/// per lane plus conflict-free shared-memory reads; `__ldcs` keeps the read-once source out of L2.
/// A tile's source region is always fully in bounds, so there is no tail path.
template <int kWarps>
__global__ void marlin_repack_gptq4_tiles_kernel(uint32_t* __restrict__ out, const GptqSource<4> src,
                                                 const int n_tiles, const int64_t total_tiles,
                                                 const RepackEpilogue epi) {
  constexpr int kTileWords = kMarlinTileK * kMarlinTileN / 8;  // 128
  __shared__ __align__(16) uint32_t stage[kWarps * 2 * kRepackStageStride];
  marlin_repack_epilogue(epi);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t tile = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  if (tile >= total_tiles) return;
  const int nt = static_cast<int>(tile % n_tiles);
  const int kt = static_cast<int>(tile / n_tiles);

  uint32_t* st = &stage[warp * 2 * kRepackStageStride];
  const int row = lane >> 4;        // the tile's low or high source row
  const int col = (lane & 15) * 4;  // 16 B of the 64 columns
  *reinterpret_cast<uint4*>(&st[row * kRepackStageStride + col]) =
      __ldcs(reinterpret_cast<const uint4*>(src.qweight +
                                            (static_cast<int64_t>(kt) * 2 + row) * src.size_n +
                                            nt * kMarlinTileN + col));
  __syncwarp();

  // Values of a tile word in `marlin_repack_tiles_kernel` order: (k0, n), (k0+1, n), (k0+8, n),
  // (k0+9, n), (k0, n+8), (k0+1, n+8), (k0+8, n+8), (k0+9, n+8) with k0 = 16*kt + 2*(lane & 3).
  // Rows 0/1 hold k%8 = 0..7 / 8..15, so one staged word supplies a whole kOffsets pair ({0,1} or
  // {8,9}) and four words per output word replace the eight scattered source reads.
  const int shift = 8 * (lane & 3);  // nibble pair 2*(lane&3), 2*(lane&3) + 1
  const int n_off = lane >> 2;       // column 0..7 of each 8-column group
  uint32_t res[4];
  #pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t w0 = st[n_off + w * 16] >> shift;                          // n,    k0
    const uint32_t w1 = st[kRepackStageStride + n_off + w * 16] >> shift;      // n,    k0 + 8
    const uint32_t w2 = st[n_off + w * 16 + 8] >> shift;                      // n+8,  k0
    const uint32_t w3 = st[kRepackStageStride + n_off + w * 16 + 8] >> shift;  // n+8,  k0 + 8
    const uint32_t vals[8] = {w0 & 0xFu, (w0 >> 4) & 0xFu,
                              w1 & 0xFu, (w1 >> 4) & 0xFu,
                              w2 & 0xFu, (w2 >> 4) & 0xFu,
                              w3 & 0xFu, (w3 >> 4) & 0xFu};
    constexpr int kPack4[8] = {0, 2, 4, 6, 1, 3, 5, 7};  // nibble i holds vals[kPack4[i]]
    uint32_t r = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) r |= vals[kPack4[i]] << (i * 4);
    res[w] = r;
  }
  reinterpret_cast<uint4*>(out + (static_cast<int64_t>(kt) * n_tiles + nt) * kTileWords + lane * 4)[0] =
      make_uint4(res[0], res[1], res[2], res[3]);
}

/// perm = stable argsort(g_idx): perm[rank(i)] = i with
/// rank(i) = #{j : g[j] < g[i]} + #{j < i : g[j] == g[i]}.
/// A CTA ranks 32 elements; its eight warps scan one eighth of g_idx each (all lanes of a warp
/// read the same g[j], one broadcast load per step) and the partial ranks are summed in shared memory.
__global__ void gptq_stable_argsort_kernel(int32_t* __restrict__ perm,
                                           const int32_t* __restrict__ g_idx, const int size_k) {
  __shared__ int partial[8][32];
  const int lane = threadIdx.x & 31;
  const int part = threadIdx.x >> 5;
  const int i = blockIdx.x * 32 + lane;
  const int32_t gi = i < size_k ? g_idx[i] : 0;
  const int chunk = (size_k + 7) / 8;
  const int j0 = part * chunk;
  const int j1 = min(size_k, j0 + chunk);
  int rank = 0;
  #pragma unroll 8
  for (int j = j0; j < j1; ++j) {
    const int32_t gj = g_idx[j];
    rank += (gj < gi) || (gj == gi && j < i);
  }
  partial[part][lane] = rank;
  __syncthreads();
  if (part == 0 && i < size_k) {
    int total = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) total += partial[p][lane];
    perm[total] = i;
  }
}

__global__ void identity_perm_kernel(int32_t* __restrict__ perm, const int size_k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size_k) perm[i] = i;
}

// -----------------------------------------------------------------------------
// Scale / zero-point permutations
// -----------------------------------------------------------------------------

/// out[r][c*C + p] = in[r][c*C + perm(p)], element size 1/2/4 bytes.  Grouped scales use C = 64
/// and perm(p) = p/8 + 8*(p%8); per-channel scales use C = 32 and perm(p) = 2*(p/8) + off[p%8].
template <typename T, bool kPerChannel>
__global__ void marlin_permute_scales_kernel(T* __restrict__ out, const T* __restrict__ in,
                                             const int64_t total) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  constexpr int kChunk = kPerChannel ? 32 : 64;
  const int p = static_cast<int>(i % kChunk);
  const int64_t base = i - p;
  const int src = kPerChannel ? 2 * (p >> 3) + kScalePerChannelOffsets[p & 7] : (p >> 3) + 8 * (p & 7);
  out[i] = in[base + src];
}

/// AWQ packed zeros [G, N/pack] -> Marlin zero layout [G, N/pack]: undo the AWQ column order,
/// apply the grouped scale permutation, interleave inside each word, repack.
template <int kBits>
__global__ void marlin_awq_zero_points_words_kernel(uint32_t* __restrict__ out,
                                                    const uint32_t* __restrict__ qzeros,
                                                    const int64_t total_words, const int n_words) {
  constexpr int kPack = 32 / kBits;
  constexpr uint32_t kMask = (1u << kBits) - 1u;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= total_words) return;
  const int64_t g = t / n_words;
  const int c = static_cast<int>(t - g * n_words);
  uint32_t res = 0;
  #pragma unroll
  for (int i = 0; i < kPack; ++i) {
    const int m = c * kPack + ((kBits == 4) ? kMarlinPack4[i] : kMarlinPack8[i]);
    const int src_n = (m & ~63) + ((m & 63) >> 3) + 8 * (m & 7);
    const uint32_t w = qzeros[g * n_words + src_n / kPack];
    const int pos = awq_position<kBits>(src_n & (kPack - 1));
    res |= ((w >> (kBits * pos)) & kMask) << (kBits * i);
  }
  out[t] = res;
}

/// FP4 scales [N, G] -> Marlin grouped scale layout [G, N] (transpose fused with the permutation).
__global__ void fp4_marlin_scales_kernel(uint8_t* __restrict__ out, const uint8_t* __restrict__ in,
                                         const int size_n, const int groups) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(size_n) * groups) return;
  const int g = static_cast<int>(i / size_n);
  const int n = static_cast<int>(i - static_cast<int64_t>(g) * size_n);
  const int p = n & 63;
  const int src_n = (n - p) + (p >> 3) + 8 * (p & 7);
  out[i] = in[static_cast<int64_t>(src_n) * groups + g];
}

// -----------------------------------------------------------------------------
// AWQ dequantize
// -----------------------------------------------------------------------------

/// One thread per packed word (eight output columns).  (q - z) is an exact integer and the
/// product with an fp16/bf16/fp32 scale is exact in fp32, so a single rounding to out_t
/// produces the correctly rounded result.
template <typename out_t>
__global__ void awq_dequantize_words_kernel(out_t* __restrict__ out, const int64_t out_ld,
                                            const uint32_t* __restrict__ qweight,
                                            const out_t* __restrict__ scales,
                                            const uint32_t* __restrict__ qzeros, const int size_k,
                                            const int size_n, const int group_size) {
  const int n_words = size_n / 8;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(size_k) * n_words) return;
  const int k = static_cast<int>(t / n_words);
  const int cw = static_cast<int>(t - static_cast<int64_t>(k) * n_words);
  const int g = group_size > 0 ? k / group_size : 0;
  const uint32_t qw = qweight[t];
  const uint32_t zw = qzeros[static_cast<int64_t>(g) * n_words + cw];
  const out_t* s = scales + static_cast<int64_t>(g) * size_n + cw * 8;
  out_t* o = out + static_cast<int64_t>(k) * out_ld + cw * 8;
  #pragma unroll
  for (int c = 0; c < 8; ++c) {
    const int pos = awq_position<4>(c);
    const int q = static_cast<int>((qw >> (4 * pos)) & 0xFu);
    const int z = static_cast<int>((zw >> (4 * pos)) & 0xFu);
    from_f<out_t>(o[c], static_cast<float>(q - z) * to_f<out_t>(s[c]));
  }
}

// -----------------------------------------------------------------------------
// GGUF dequantize (ggml block formats; semantics of ggml-quants.c dequantize_row_*, fp32 math)
// -----------------------------------------------------------------------------

__constant__ int8_t kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1,    13,   25,  38,  53,  69,  89,  113};

// ---- GGUF codebooks: BEGIN adapted-from-llama.cpp block -------------------------------------
// Ported from llama.cpp ggml/src/ggml-common.h @ 0ec4027 (MIT)
//   tables iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid, iq3s_grid, iq1s_grid_gpu
// Copyright (c) 2023-2026 The ggml authors; permission notice in third_party/licenses/llama.cpp.txt.
// Change : the numbers are extracted by script and re-emitted as __device__ const arrays with
//          fixed-width literals; the cpu iq1s_grid and ksigns_iq2xs/kmask_iq2xs are not copied (see
//          gguf_iq1s_value and gguf_signs below, which reproduce them).
// SHA-256 (first 16 hex digits) of each table as little-endian words, to detect edits:
//   iq2xxs 05826b5d3e472a3a  iq2xs 06e47aaca60b4dc1  iq2s e1aa1473412b0552
//   iq3xxs 46e35f5a997efdee  iq3s  bd1af4945a1717c6  iq1s_gpu 29bb0f4450fab4aa
__device__ const uint64_t kIq2xxsGrid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL,
    0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b2b08ULL,
    0x08080808082b2b2bULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL,
    0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL, 0x0808081908191919ULL,
    0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL,
    0x0808082b082b082bULL, 0x0808082b2b08082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL,
    0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL,
    0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL, 0x08081919082b0808ULL, 0x080819191908192bULL,
    0x08081919192b2b19ULL, 0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL,
    0x0808192b19080808ULL, 0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL,
    0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL, 0x08082b1919080808ULL,
    0x08082b2b0808082bULL, 0x08082b2b08191908ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL,
    0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL, 0x0819081919190808ULL,
    0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL, 0x0819082b19081919ULL, 0x0819190808080808ULL,
    0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL,
    0x0819191908192b08ULL, 0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL, 0x08192b1908080808ULL,
    0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL, 0x082b080808080808ULL, 0x082b08080808082bULL,
    0x082b080808082b2bULL, 0x082b080819081908ULL, 0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL,
    0x082b0819082b2b19ULL, 0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL, 0x082b191908080808ULL,
    0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL, 0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL,
    0x082b2b082b191908ULL, 0x082b2b2b19081908ULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL,
    0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL,
    0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL,
    0x1908082b08080819ULL, 0x1908082b08190808ULL, 0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL,
    0x1908190808080808ULL, 0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,
    0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b1908080808ULL, 0x19082b1919192b08ULL,
    0x19082b19192b0819ULL, 0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL,
    0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL, 0x1919082b2b190819ULL,
    0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL, 0x1919192b19191908ULL,
    0x19192b0808080808ULL, 0x19192b0808190819ULL, 0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL,
    0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL, 0x192b190808080808ULL,
    0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL, 0x192b19192b081908ULL, 0x192b2b081908082bULL,
    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL,
    0x2b08081908081908ULL, 0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL,
    0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL, 0x2b08192b08082b19ULL, 0x2b08192b19080808ULL,
    0x2b08192b192b0808ULL, 0x2b082b080808082bULL, 0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL,
    0x2b19080808190808ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL, 0x2b19190819081908ULL,
    0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL, 0x2b2b08080808082bULL, 0x2b2b080819190808ULL,
    0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL, 0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL,
    0x2b2b2b1908081908ULL,
};
__device__ const uint64_t kIq2xsGrid[512] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL,
    0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL, 0x0808080808192b19ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL,
    0x080808081908192bULL, 0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL,
    0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL,
    0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b192b19ULL,
    0x080808082b2b0808ULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL,
    0x0808081908190808ULL, 0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL,
    0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL,
    0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL, 0x08080819192b0808ULL, 0x08080819192b2b08ULL,
    0x080808192b080819ULL, 0x080808192b081908ULL, 0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL,
    0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL,
    0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL, 0x0808082b2b080808ULL,
    0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL,
    0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL,
    0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL, 0x080819082b080819ULL,
    0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL,
    0x0808191908082b08ULL, 0x0808191908190819ULL, 0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL,
    0x0808191919081908ULL, 0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL,
    0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL, 0x0808192b1908082bULL,
    0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL,
    0x08082b0808082b2bULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL,
    0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL,
    0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL, 0x08082b1908190808ULL,
    0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL, 0x08082b2b08080808ULL, 0x08082b2b082b0808ULL,
    0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL, 0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL,
    0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL,
    0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL, 0x081908081908082bULL,
    0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x08190808192b0808ULL,
    0x08190808192b2b2bULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL,
    0x081908190808082bULL, 0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL,
    0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL, 0x081908192b080808ULL,
    0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b0808192bULL,
    0x0819082b08190808ULL, 0x0819082b19080808ULL, 0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL,
    0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL,
    0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL, 0x08191908192b1908ULL,
    0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x0819191908190808ULL, 0x0819191919080808ULL,
    0x0819192b08080808ULL, 0x0819192b08191908ULL, 0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL,
    0x08192b0808190808ULL, 0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL, 0x08192b2b2b2b2b19ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL, 0x082b080808082b2bULL,
    0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL,
    0x082b080819190808ULL, 0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL,
    0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL, 0x082b082b08080808ULL,
    0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL,
    0x082b190808190808ULL, 0x082b1908082b2b19ULL, 0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL,
    0x082b19191919082bULL, 0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL,
    0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL, 0x082b2b0819191919ULL,
    0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL, 0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL,
    0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL,
    0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL,
    0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL,
    0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL,
    0x1908080819191908ULL, 0x19080808192b0808ULL, 0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL,
    0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL, 0x1908081919081908ULL,
    0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x190808192b2b082bULL, 0x1908082b08080819ULL,
    0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL,
    0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL,
    0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL, 0x1908190819081908ULL,
    0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL, 0x1908191908080819ULL, 0x1908191908081908ULL,
    0x1908191908190808ULL, 0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL,
    0x1908192b08082b2bULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,
    0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL, 0x19082b08192b082bULL,
    0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL, 0x19082b19192b2b19ULL,
    0x19082b2b08081908ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL,
    0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL, 0x1919081908081908ULL,
    0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL, 0x191908191908082bULL, 0x1919082b08080808ULL,
    0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL,
    0x19191908082b0819ULL, 0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL,
    0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL, 0x1919192b082b0819ULL,
    0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL, 0x19192b0808191908ULL, 0x19192b0819080819ULL,
    0x19192b0819190808ULL, 0x19192b082b192b19ULL, 0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL,
    0x19192b2b2b081919ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL,
    0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL, 0x192b081908080808ULL,
    0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL, 0x192b190808080808ULL, 0x192b19080819192bULL,
    0x192b191908190808ULL, 0x192b191919080808ULL, 0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL,
    0x192b2b08192b2b2bULL, 0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL,
    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL,
    0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL,
    0x2b08080819190808ULL, 0x2b0808082b080808ULL, 0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL,
    0x2b08081908080819ULL, 0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL,
    0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL, 0x2b08082b2b080808ULL,
    0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL,
    0x2b08190808190808ULL, 0x2b0819080819082bULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL,
    0x2b0819082b082b19ULL, 0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL,
    0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL, 0x2b082b0819192b2bULL,
    0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL, 0x2b082b190808192bULL, 0x2b082b2b082b082bULL,
    0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL, 0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL,
    0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL,
    0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL, 0x2b19082b2b082b19ULL,
    0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL, 0x2b19190819192b08ULL,
    0x2b191919082b2b19ULL, 0x2b1919192b190808ULL, 0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL,
    0x2b192b082b2b192bULL, 0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL,
    0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL, 0x2b2b0808082b2b2bULL,
    0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL, 0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL,
    0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL,
    0x2b2b190819080808ULL, 0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL,
    0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL, 0x2b2b2b082b2b2b08ULL,
    0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL,
    0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL,
};
__device__ const uint64_t kIq2sGrid[1024] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL,
    0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL, 0x0808080808192b19ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL,
    0x080808081908192bULL, 0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL,
    0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x08080808192b192bULL, 0x08080808192b2b19ULL,
    0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL,
    0x080808082b191908ULL, 0x080808082b2b0808ULL, 0x080808082b2b1919ULL, 0x080808082b2b2b2bULL, 0x0808081908080819ULL,
    0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL,
    0x0808081908191919ULL, 0x0808081908192b08ULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL,
    0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL,
    0x080808191919192bULL, 0x0808081919192b19ULL, 0x08080819192b0808ULL, 0x08080819192b1919ULL, 0x08080819192b2b08ULL,
    0x080808192b080819ULL, 0x080808192b081908ULL, 0x080808192b190808ULL, 0x080808192b19082bULL, 0x080808192b191919ULL,
    0x080808192b2b0819ULL, 0x080808192b2b1908ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL,
    0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b082b2b2bULL,
    0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b1908192bULL, 0x0808082b19082b19ULL, 0x0808082b19190808ULL,
    0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b081919ULL, 0x0808082b2b082b2bULL, 0x0808082b2b191908ULL,
    0x0808082b2b2b082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL,
    0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x08081908082b192bULL, 0x08081908082b2b19ULL, 0x0808190819080808ULL, 0x080819081908082bULL,
    0x0808190819081919ULL, 0x0808190819082b08ULL, 0x0808190819082b2bULL, 0x0808190819190819ULL, 0x0808190819191908ULL,
    0x080819081919192bULL, 0x0808190819192b19ULL, 0x08081908192b0808ULL, 0x08081908192b082bULL, 0x08081908192b1919ULL,
    0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b08192bULL, 0x080819082b082b19ULL, 0x080819082b190808ULL,
    0x080819082b191919ULL, 0x080819082b192b08ULL, 0x080819082b2b0819ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL,
    0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908082b2bULL, 0x0808191908190819ULL,
    0x0808191908191908ULL, 0x080819190819192bULL, 0x0808191908192b19ULL, 0x08081919082b0808ULL, 0x08081919082b1919ULL,
    0x08081919082b2b08ULL, 0x0808191919080819ULL, 0x0808191919081908ULL, 0x080819191908192bULL, 0x0808191919082b19ULL,
    0x0808191919190808ULL, 0x080819191919082bULL, 0x0808191919191919ULL, 0x0808191919192b08ULL, 0x08081919192b0819ULL,
    0x08081919192b1908ULL, 0x080819192b080808ULL, 0x080819192b08082bULL, 0x080819192b081919ULL, 0x080819192b082b08ULL,
    0x080819192b190819ULL, 0x080819192b191908ULL, 0x080819192b2b0808ULL, 0x0808192b08080819ULL, 0x0808192b08081908ULL,
    0x0808192b0808192bULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b08191919ULL, 0x0808192b19080808ULL,
    0x0808192b19081919ULL, 0x0808192b19082b08ULL, 0x0808192b19190819ULL, 0x0808192b19191908ULL, 0x0808192b192b0808ULL,
    0x0808192b2b080819ULL, 0x0808192b2b081908ULL, 0x0808192b2b190808ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL,
    0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b080819192bULL,
    0x08082b0808192b19ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b08082b2b2bULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b081908192bULL, 0x08082b0819082b19ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL,
    0x08082b0819191919ULL, 0x08082b0819192b08ULL, 0x08082b08192b0819ULL, 0x08082b08192b1908ULL, 0x08082b082b080808ULL,
    0x08082b082b081919ULL, 0x08082b082b191908ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL,
    0x08082b1908190808ULL, 0x08082b190819082bULL, 0x08082b1908191919ULL, 0x08082b1908192b08ULL, 0x08082b19082b0819ULL,
    0x08082b1919080808ULL, 0x08082b1919081919ULL, 0x08082b1919082b08ULL, 0x08082b1919190819ULL, 0x08082b1919191908ULL,
    0x08082b19192b0808ULL, 0x08082b192b080819ULL, 0x08082b192b190808ULL, 0x08082b2b08080808ULL, 0x08082b2b08190819ULL,
    0x08082b2b08191908ULL, 0x08082b2b082b082bULL, 0x08082b2b082b2b08ULL, 0x08082b2b082b2b2bULL, 0x08082b2b19190808ULL,
    0x08082b2b2b192b19ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL, 0x0819080808082b19ULL,
    0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL,
    0x08190808082b1908ULL, 0x08190808082b192bULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL,
    0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x081908081919192bULL, 0x0819080819192b19ULL,
    0x08190808192b0808ULL, 0x08190808192b082bULL, 0x08190808192b1919ULL, 0x08190808192b2b08ULL, 0x081908082b080819ULL,
    0x081908082b081908ULL, 0x081908082b08192bULL, 0x081908082b190808ULL, 0x081908082b191919ULL, 0x081908082b192b08ULL,
    0x081908082b2b0819ULL, 0x081908082b2b1908ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL,
    0x0819081908082b08ULL, 0x0819081908082b2bULL, 0x0819081908190819ULL, 0x0819081908191908ULL, 0x081908190819192bULL,
    0x0819081908192b19ULL, 0x08190819082b0808ULL, 0x08190819082b082bULL, 0x08190819082b1919ULL, 0x08190819082b2b08ULL,
    0x0819081919080819ULL, 0x0819081919081908ULL, 0x081908191908192bULL, 0x0819081919082b19ULL, 0x0819081919190808ULL,
    0x081908191919082bULL, 0x0819081919191919ULL, 0x0819081919192b08ULL, 0x08190819192b0819ULL, 0x08190819192b1908ULL,
    0x081908192b080808ULL, 0x081908192b08082bULL, 0x081908192b081919ULL, 0x081908192b082b08ULL, 0x081908192b190819ULL,
    0x081908192b191908ULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b08082b19ULL, 0x0819082b08190808ULL,
    0x0819082b08191919ULL, 0x0819082b082b0819ULL, 0x0819082b082b1908ULL, 0x0819082b19080808ULL, 0x0819082b19081919ULL,
    0x0819082b19190819ULL, 0x0819082b19191908ULL, 0x0819082b2b080819ULL, 0x0819082b2b081908ULL, 0x0819082b2b190808ULL,
    0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL,
    0x0819190808191908ULL, 0x081919080819192bULL, 0x0819190808192b19ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL,
    0x08191908082b2b08ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x081919081908192bULL, 0x0819190819082b19ULL,
    0x0819190819190808ULL, 0x081919081919082bULL, 0x0819190819191919ULL, 0x0819190819192b08ULL, 0x08191908192b0819ULL,
    0x08191908192b1908ULL, 0x081919082b080808ULL, 0x081919082b08082bULL, 0x081919082b081919ULL, 0x081919082b082b08ULL,
    0x081919082b190819ULL, 0x081919082b191908ULL, 0x081919082b2b0808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL,
    0x081919190808192bULL, 0x0819191908082b19ULL, 0x0819191908190808ULL, 0x081919190819082bULL, 0x0819191908191919ULL,
    0x0819191908192b08ULL, 0x08191919082b0819ULL, 0x08191919082b1908ULL, 0x0819191919080808ULL, 0x081919191908082bULL,
    0x0819191919081919ULL, 0x0819191919082b08ULL, 0x0819191919190819ULL, 0x0819191919191908ULL, 0x08191919192b0808ULL,
    0x081919192b080819ULL, 0x081919192b081908ULL, 0x081919192b190808ULL, 0x0819192b08080808ULL, 0x0819192b08081919ULL,
    0x0819192b08082b08ULL, 0x0819192b08190819ULL, 0x0819192b08191908ULL, 0x0819192b082b0808ULL, 0x0819192b19080819ULL,
    0x0819192b19081908ULL, 0x0819192b19190808ULL, 0x0819192b2b080808ULL, 0x0819192b2b2b2b2bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b080808192bULL, 0x08192b0808082b19ULL, 0x08192b0808190808ULL, 0x08192b0808191919ULL,
    0x08192b0808192b08ULL, 0x08192b08082b0819ULL, 0x08192b0819080808ULL, 0x08192b081908082bULL, 0x08192b0819081919ULL,
    0x08192b0819082b08ULL, 0x08192b0819190819ULL, 0x08192b0819191908ULL, 0x08192b08192b0808ULL, 0x08192b082b080819ULL,
    0x08192b082b081908ULL, 0x08192b1908080808ULL, 0x08192b190808082bULL, 0x08192b1908081919ULL, 0x08192b1908082b08ULL,
    0x08192b1908190819ULL, 0x08192b1908191908ULL, 0x08192b19082b0808ULL, 0x08192b1919080819ULL, 0x08192b1919081908ULL,
    0x08192b1919190808ULL, 0x08192b19192b2b19ULL, 0x08192b192b2b082bULL, 0x08192b2b08081908ULL, 0x08192b2b08190808ULL,
    0x08192b2b19080808ULL, 0x08192b2b1919192bULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL,
    0x082b080808082b08ULL, 0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b08080819192bULL, 0x082b080808192b19ULL,
    0x082b0808082b0808ULL, 0x082b0808082b1919ULL, 0x082b0808082b2b2bULL, 0x082b080819080819ULL, 0x082b080819081908ULL,
    0x082b080819190808ULL, 0x082b08081919082bULL, 0x082b080819191919ULL, 0x082b0808192b1908ULL, 0x082b08082b080808ULL,
    0x082b08082b082b2bULL, 0x082b08082b191908ULL, 0x082b08082b2b2b2bULL, 0x082b081908080819ULL, 0x082b081908081908ULL,
    0x082b081908190808ULL, 0x082b08190819082bULL, 0x082b081908191919ULL, 0x082b0819082b0819ULL, 0x082b081919080808ULL,
    0x082b08191908082bULL, 0x082b081919081919ULL, 0x082b081919190819ULL, 0x082b081919191908ULL, 0x082b0819192b0808ULL,
    0x082b08192b080819ULL, 0x082b08192b081908ULL, 0x082b08192b190808ULL, 0x082b082b08080808ULL, 0x082b082b08082b2bULL,
    0x082b082b082b082bULL, 0x082b082b082b2b08ULL, 0x082b082b082b2b2bULL, 0x082b082b19081908ULL, 0x082b082b19190808ULL,
    0x082b082b2b082b08ULL, 0x082b082b2b082b2bULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL,
    0x082b19080808192bULL, 0x082b190808082b19ULL, 0x082b190808190808ULL, 0x082b190808191919ULL, 0x082b190808192b08ULL,
    0x082b1908082b0819ULL, 0x082b1908082b1908ULL, 0x082b190819080808ULL, 0x082b19081908082bULL, 0x082b190819081919ULL,
    0x082b190819082b08ULL, 0x082b190819190819ULL, 0x082b190819191908ULL, 0x082b1908192b0808ULL, 0x082b19082b080819ULL,
    0x082b19082b081908ULL, 0x082b19082b190808ULL, 0x082b191908080808ULL, 0x082b191908081919ULL, 0x082b191908082b08ULL,
    0x082b191908190819ULL, 0x082b191908191908ULL, 0x082b1919082b0808ULL, 0x082b191919080819ULL, 0x082b191919081908ULL,
    0x082b191919190808ULL, 0x082b1919192b192bULL, 0x082b19192b080808ULL, 0x082b192b08080819ULL, 0x082b192b08081908ULL,
    0x082b192b08190808ULL, 0x082b192b19080808ULL, 0x082b192b19192b19ULL, 0x082b2b0808080808ULL, 0x082b2b0808081919ULL,
    0x082b2b0808190819ULL, 0x082b2b0808191908ULL, 0x082b2b0819080819ULL, 0x082b2b0819081908ULL, 0x082b2b0819190808ULL,
    0x082b2b082b082b2bULL, 0x082b2b082b2b2b2bULL, 0x082b2b1908080819ULL, 0x082b2b1908081908ULL, 0x082b2b1908190808ULL,
    0x082b2b192b191919ULL, 0x082b2b2b08082b2bULL, 0x082b2b2b082b082bULL, 0x082b2b2b192b1908ULL, 0x082b2b2b2b082b08ULL,
    0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL,
    0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x1908080808192b2bULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x19080808082b192bULL, 0x1908080819080808ULL, 0x190808081908082bULL,
    0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL,
    0x190808081919192bULL, 0x1908080819192b19ULL, 0x19080808192b0808ULL, 0x19080808192b082bULL, 0x19080808192b1919ULL,
    0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x190808082b191919ULL, 0x190808082b192b08ULL,
    0x190808082b2b0819ULL, 0x190808082b2b1908ULL, 0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL,
    0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL, 0x190808190819192bULL, 0x1908081908192b19ULL,
    0x19080819082b0808ULL, 0x19080819082b082bULL, 0x19080819082b1919ULL, 0x1908081919080819ULL, 0x1908081919081908ULL,
    0x190808191908192bULL, 0x1908081919082b19ULL, 0x1908081919190808ULL, 0x190808191919082bULL, 0x1908081919191919ULL,
    0x1908081919192b08ULL, 0x19080819192b0819ULL, 0x19080819192b1908ULL, 0x190808192b080808ULL, 0x190808192b08082bULL,
    0x190808192b081919ULL, 0x190808192b082b08ULL, 0x190808192b190819ULL, 0x190808192b191908ULL, 0x190808192b2b0808ULL,
    0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b08191919ULL,
    0x1908082b08192b08ULL, 0x1908082b082b1908ULL, 0x1908082b19080808ULL, 0x1908082b19081919ULL, 0x1908082b19082b08ULL,
    0x1908082b19190819ULL, 0x1908082b19191908ULL, 0x1908082b192b0808ULL, 0x1908082b2b080819ULL, 0x1908082b2b081908ULL,
    0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808082b2bULL,
    0x1908190808190819ULL, 0x1908190808191908ULL, 0x190819080819192bULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL,
    0x19081908082b082bULL, 0x19081908082b1919ULL, 0x19081908082b2b08ULL, 0x1908190819080819ULL, 0x1908190819081908ULL,
    0x190819081908192bULL, 0x1908190819082b19ULL, 0x1908190819190808ULL, 0x190819081919082bULL, 0x1908190819191919ULL,
    0x1908190819192b08ULL, 0x19081908192b0819ULL, 0x19081908192b1908ULL, 0x190819082b080808ULL, 0x190819082b08082bULL,
    0x190819082b081919ULL, 0x190819082b082b08ULL, 0x190819082b190819ULL, 0x190819082b191908ULL, 0x190819082b2b0808ULL,
    0x1908191908080819ULL, 0x1908191908081908ULL, 0x190819190808192bULL, 0x1908191908082b19ULL, 0x1908191908190808ULL,
    0x190819190819082bULL, 0x1908191908191919ULL, 0x1908191908192b08ULL, 0x19081919082b0819ULL, 0x19081919082b1908ULL,
    0x1908191919080808ULL, 0x190819191908082bULL, 0x1908191919081919ULL, 0x1908191919082b08ULL, 0x1908191919190819ULL,
    0x1908191919191908ULL, 0x19081919192b0808ULL, 0x19081919192b2b2bULL, 0x190819192b080819ULL, 0x190819192b081908ULL,
    0x190819192b190808ULL, 0x1908192b08080808ULL, 0x1908192b0808082bULL, 0x1908192b08081919ULL, 0x1908192b08082b08ULL,
    0x1908192b08190819ULL, 0x1908192b08191908ULL, 0x1908192b082b0808ULL, 0x1908192b19080819ULL, 0x1908192b19081908ULL,
    0x1908192b19190808ULL, 0x1908192b2b080808ULL, 0x1908192b2b2b1919ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,
    0x19082b0808082b19ULL, 0x19082b0808190808ULL, 0x19082b080819082bULL, 0x19082b0808191919ULL, 0x19082b0808192b08ULL,
    0x19082b08082b0819ULL, 0x19082b08082b1908ULL, 0x19082b0819080808ULL, 0x19082b081908082bULL, 0x19082b0819081919ULL,
    0x19082b0819082b08ULL, 0x19082b0819190819ULL, 0x19082b0819191908ULL, 0x19082b08192b0808ULL, 0x19082b082b081908ULL,
    0x19082b082b190808ULL, 0x19082b1908080808ULL, 0x19082b190808082bULL, 0x19082b1908081919ULL, 0x19082b1908082b08ULL,
    0x19082b1908190819ULL, 0x19082b1908191908ULL, 0x19082b19082b0808ULL, 0x19082b1919080819ULL, 0x19082b1919081908ULL,
    0x19082b1919190808ULL, 0x19082b192b080808ULL, 0x19082b192b19192bULL, 0x19082b2b08080819ULL, 0x19082b2b08081908ULL,
    0x19082b2b08190808ULL, 0x19082b2b19080808ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL,
    0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL, 0x191908080819192bULL, 0x1919080808192b19ULL,
    0x19190808082b0808ULL, 0x19190808082b082bULL, 0x19190808082b1919ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL,
    0x1919080819081908ULL, 0x191908081908192bULL, 0x1919080819082b19ULL, 0x1919080819190808ULL, 0x191908081919082bULL,
    0x1919080819191919ULL, 0x1919080819192b08ULL, 0x19190808192b0819ULL, 0x19190808192b1908ULL, 0x191908082b080808ULL,
    0x191908082b08082bULL, 0x191908082b081919ULL, 0x191908082b082b08ULL, 0x191908082b190819ULL, 0x191908082b191908ULL,
    0x1919081908080819ULL, 0x1919081908081908ULL, 0x191908190808192bULL, 0x1919081908082b19ULL, 0x1919081908190808ULL,
    0x191908190819082bULL, 0x1919081908191919ULL, 0x1919081908192b08ULL, 0x19190819082b0819ULL, 0x19190819082b1908ULL,
    0x1919081919080808ULL, 0x191908191908082bULL, 0x1919081919081919ULL, 0x1919081919082b08ULL, 0x1919081919190819ULL,
    0x1919081919191908ULL, 0x19190819192b0808ULL, 0x191908192b080819ULL, 0x191908192b081908ULL, 0x191908192b190808ULL,
    0x1919082b08080808ULL, 0x1919082b08081919ULL, 0x1919082b08082b08ULL, 0x1919082b08190819ULL, 0x1919082b08191908ULL,
    0x1919082b082b0808ULL, 0x1919082b19080819ULL, 0x1919082b19081908ULL, 0x1919082b19190808ULL, 0x1919082b192b2b19ULL,
    0x1919082b2b080808ULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x191919080808192bULL, 0x1919190808082b19ULL,
    0x1919190808190808ULL, 0x191919080819082bULL, 0x1919190808191919ULL, 0x1919190808192b08ULL, 0x19191908082b0819ULL,
    0x19191908082b1908ULL, 0x1919190819080808ULL, 0x191919081908082bULL, 0x1919190819081919ULL, 0x1919190819082b08ULL,
    0x1919190819190819ULL, 0x1919190819191908ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b081908ULL,
    0x191919082b190808ULL, 0x1919191908080808ULL, 0x191919190808082bULL, 0x1919191908081919ULL, 0x1919191908082b08ULL,
    0x1919191908190819ULL, 0x1919191908191908ULL, 0x19191919082b0808ULL, 0x1919191919080819ULL, 0x1919191919081908ULL,
    0x1919191919190808ULL, 0x191919192b080808ULL, 0x1919192b08080819ULL, 0x1919192b08081908ULL, 0x1919192b08190808ULL,
    0x1919192b082b192bULL, 0x1919192b19080808ULL, 0x19192b0808080808ULL, 0x19192b080808082bULL, 0x19192b0808081919ULL,
    0x19192b0808082b08ULL, 0x19192b0808190819ULL, 0x19192b0808191908ULL, 0x19192b08082b0808ULL, 0x19192b0819080819ULL,
    0x19192b0819081908ULL, 0x19192b0819190808ULL, 0x19192b0819192b2bULL, 0x19192b082b080808ULL, 0x19192b1908080819ULL,
    0x19192b1908081908ULL, 0x19192b1908190808ULL, 0x19192b1919080808ULL, 0x19192b2b08080808ULL, 0x19192b2b08192b19ULL,
    0x19192b2b2b081919ULL, 0x19192b2b2b2b2b08ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b08080808192bULL,
    0x192b080808190808ULL, 0x192b08080819082bULL, 0x192b080808191919ULL, 0x192b080808192b08ULL, 0x192b0808082b0819ULL,
    0x192b0808082b1908ULL, 0x192b080819080808ULL, 0x192b080819081919ULL, 0x192b080819082b08ULL, 0x192b080819190819ULL,
    0x192b080819191908ULL, 0x192b0808192b0808ULL, 0x192b08082b081908ULL, 0x192b08082b190808ULL, 0x192b081908080808ULL,
    0x192b08190808082bULL, 0x192b081908081919ULL, 0x192b081908082b08ULL, 0x192b081908190819ULL, 0x192b081908191908ULL,
    0x192b0819082b0808ULL, 0x192b081919080819ULL, 0x192b081919081908ULL, 0x192b081919190808ULL, 0x192b08192b080808ULL,
    0x192b08192b192b19ULL, 0x192b082b08081908ULL, 0x192b082b08190808ULL, 0x192b082b19080808ULL, 0x192b082b1919192bULL,
    0x192b082b2b2b0819ULL, 0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b190808082b08ULL, 0x192b190808190819ULL,
    0x192b190808191908ULL, 0x192b1908082b0808ULL, 0x192b190819080819ULL, 0x192b190819081908ULL, 0x192b190819190808ULL,
    0x192b19082b080808ULL, 0x192b191908080819ULL, 0x192b191908081908ULL, 0x192b191908190808ULL, 0x192b191919080808ULL,
    0x192b191919082b2bULL, 0x192b1919192b2b08ULL, 0x192b19192b19082bULL, 0x192b192b08080808ULL, 0x192b192b2b191908ULL,
    0x192b2b0808080819ULL, 0x192b2b0808081908ULL, 0x192b2b0808190808ULL, 0x192b2b08192b1919ULL, 0x192b2b082b192b08ULL,
    0x192b2b1908080808ULL, 0x192b2b19082b2b2bULL, 0x192b2b2b1908082bULL, 0x192b2b2b2b2b0819ULL, 0x2b08080808080808ULL,
    0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL, 0x2b08080808191908ULL,
    0x2b08080808192b19ULL, 0x2b080808082b0808ULL, 0x2b080808082b1919ULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL,
    0x2b08080819190808ULL, 0x2b0808081919082bULL, 0x2b08080819191919ULL, 0x2b08080819192b08ULL, 0x2b080808192b0819ULL,
    0x2b0808082b080808ULL, 0x2b0808082b081919ULL, 0x2b0808082b190819ULL, 0x2b0808082b191908ULL, 0x2b08081908080819ULL,
    0x2b08081908081908ULL, 0x2b08081908082b19ULL, 0x2b08081908190808ULL, 0x2b0808190819082bULL, 0x2b08081908191919ULL,
    0x2b08081908192b08ULL, 0x2b080819082b0819ULL, 0x2b080819082b1908ULL, 0x2b08081919080808ULL, 0x2b0808191908082bULL,
    0x2b08081919081919ULL, 0x2b08081919082b08ULL, 0x2b08081919190819ULL, 0x2b08081919191908ULL, 0x2b0808192b080819ULL,
    0x2b0808192b081908ULL, 0x2b0808192b190808ULL, 0x2b0808192b2b2b19ULL, 0x2b08082b08080808ULL, 0x2b08082b08081919ULL,
    0x2b08082b08082b2bULL, 0x2b08082b08190819ULL, 0x2b08082b08191908ULL, 0x2b08082b19080819ULL, 0x2b08082b19081908ULL,
    0x2b08082b19190808ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b0819080808192bULL, 0x2b08190808082b19ULL,
    0x2b08190808190808ULL, 0x2b0819080819082bULL, 0x2b08190808191919ULL, 0x2b08190808192b08ULL, 0x2b081908082b0819ULL,
    0x2b08190819080808ULL, 0x2b0819081908082bULL, 0x2b08190819081919ULL, 0x2b08190819082b08ULL, 0x2b08190819190819ULL,
    0x2b08190819191908ULL, 0x2b081908192b0808ULL, 0x2b0819082b080819ULL, 0x2b0819082b081908ULL, 0x2b0819082b190808ULL,
    0x2b08191908080808ULL, 0x2b0819190808082bULL, 0x2b08191908081919ULL, 0x2b08191908082b08ULL, 0x2b08191908190819ULL,
    0x2b08191908191908ULL, 0x2b081919082b0808ULL, 0x2b08191919080819ULL, 0x2b08191919081908ULL, 0x2b08191919190808ULL,
    0x2b0819192b080808ULL, 0x2b0819192b082b2bULL, 0x2b08192b08080819ULL, 0x2b08192b08081908ULL, 0x2b08192b08190808ULL,
    0x2b08192b082b2b19ULL, 0x2b08192b19080808ULL, 0x2b082b0808080808ULL, 0x2b082b0808081919ULL, 0x2b082b0808190819ULL,
    0x2b082b0808191908ULL, 0x2b082b0819080819ULL, 0x2b082b0819081908ULL, 0x2b082b0819190808ULL, 0x2b082b082b2b082bULL,
    0x2b082b1908080819ULL, 0x2b082b1908081908ULL, 0x2b082b1919080808ULL, 0x2b082b19192b1919ULL, 0x2b082b2b082b082bULL,
    0x2b082b2b19192b08ULL, 0x2b082b2b19192b2bULL, 0x2b082b2b2b08082bULL, 0x2b082b2b2b2b082bULL, 0x2b19080808080819ULL,
    0x2b19080808081908ULL, 0x2b19080808082b19ULL, 0x2b19080808190808ULL, 0x2b1908080819082bULL, 0x2b19080808191919ULL,
    0x2b19080808192b08ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908081908082bULL, 0x2b19080819081919ULL,
    0x2b19080819082b08ULL, 0x2b19080819190819ULL, 0x2b19080819191908ULL, 0x2b190808192b0808ULL, 0x2b1908082b080819ULL,
    0x2b1908082b081908ULL, 0x2b1908082b190808ULL, 0x2b19081908080808ULL, 0x2b19081908081919ULL, 0x2b19081908190819ULL,
    0x2b19081908191908ULL, 0x2b19081919080819ULL, 0x2b19081919081908ULL, 0x2b19081919190808ULL, 0x2b19081919192b2bULL,
    0x2b19082b08080819ULL, 0x2b19082b08081908ULL, 0x2b19082b08190808ULL, 0x2b19082b19080808ULL, 0x2b19082b2b2b192bULL,
    0x2b19190808080808ULL, 0x2b1919080808082bULL, 0x2b19190808081919ULL, 0x2b19190808082b08ULL, 0x2b19190808190819ULL,
    0x2b19190808191908ULL, 0x2b191908082b0808ULL, 0x2b19190819080819ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL,
    0x2b1919082b080808ULL, 0x2b1919082b19192bULL, 0x2b19191908080819ULL, 0x2b19191908081908ULL, 0x2b19191908190808ULL,
    0x2b19191919080808ULL, 0x2b1919192b192b08ULL, 0x2b1919192b2b0819ULL, 0x2b19192b08080808ULL, 0x2b19192b1908192bULL,
    0x2b19192b192b1908ULL, 0x2b192b0808080819ULL, 0x2b192b0808081908ULL, 0x2b192b0808190808ULL, 0x2b192b08082b192bULL,
    0x2b192b0819080808ULL, 0x2b192b082b2b2b19ULL, 0x2b192b1908080808ULL, 0x2b192b1919082b19ULL, 0x2b192b191919082bULL,
    0x2b192b2b2b190808ULL, 0x2b2b080808080808ULL, 0x2b2b080808081919ULL, 0x2b2b080808082b2bULL, 0x2b2b080808191908ULL,
    0x2b2b0808082b082bULL, 0x2b2b0808082b2b2bULL, 0x2b2b080819080819ULL, 0x2b2b080819081908ULL, 0x2b2b080819190808ULL,
    0x2b2b08082b2b082bULL, 0x2b2b08082b2b2b2bULL, 0x2b2b081919080808ULL, 0x2b2b0819192b1919ULL, 0x2b2b082b0808082bULL,
    0x2b2b082b08082b2bULL, 0x2b2b082b082b082bULL, 0x2b2b082b082b2b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b08082bULL,
    0x2b2b082b2b082b08ULL, 0x2b2b082b2b082b2bULL, 0x2b2b082b2b2b2b08ULL, 0x2b2b190808080819ULL, 0x2b2b190808081908ULL,
    0x2b2b190808190808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b082b19ULL, 0x2b2b19082b2b1908ULL, 0x2b2b191908080808ULL,
    0x2b2b191908192b19ULL, 0x2b2b192b19190819ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b082bULL,
    0x2b2b2b1919191908ULL, 0x2b2b2b192b08192bULL, 0x2b2b2b2b08082b08ULL, 0x2b2b2b2b08082b2bULL, 0x2b2b2b2b082b0808ULL,
    0x2b2b2b2b082b082bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b2b082b08ULL, 0x2b2b2b2b2b2b2b2bULL,
};
__device__ const uint32_t kIq3xxsGrid[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};
__device__ const uint32_t kIq3sGrid[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};
__device__ const uint32_t kIq1sGridGpu[2048] = {
    0x00000000u, 0x00000002u, 0x00000101u, 0x00000200u, 0x00000202u, 0x00010001u, 0x00010101u, 0x00020000u,
    0x00020002u, 0x00020200u, 0x00020202u, 0x01000101u, 0x01010001u, 0x01010100u, 0x01010102u, 0x01020101u,
    0x02000000u, 0x02000002u, 0x02000200u, 0x02000202u, 0x02010101u, 0x02020000u, 0x02020002u, 0x02020200u,
    0x02020202u, 0x00000110u, 0x00000111u, 0x00010011u, 0x00010110u, 0x00010112u, 0x00010211u, 0x00010212u,
    0x00020111u, 0x01000011u, 0x01000112u, 0x01000211u, 0x01010012u, 0x01010111u, 0x01010212u, 0x01020011u,
    0x01020110u, 0x01020112u, 0x01020210u, 0x02000111u, 0x02010011u, 0x02010110u, 0x02010112u, 0x02020111u,
    0x00000020u, 0x00000022u, 0x00000220u, 0x00000222u, 0x00010121u, 0x00020020u, 0x00020022u, 0x00020220u,
    0x00020222u, 0x01000121u, 0x01010021u, 0x01010221u, 0x01020120u, 0x01020221u, 0x02000020u, 0x02000022u,
    0x02000220u, 0x02000222u, 0x02010021u, 0x02010121u, 0x02010221u, 0x02020020u, 0x02020022u, 0x02020220u,
    0x02020222u, 0x00011001u, 0x00011100u, 0x00011102u, 0x00021101u, 0x01001001u, 0x01001201u, 0x01011101u,
    0x01011202u, 0x01021100u, 0x01021101u, 0x02011001u, 0x02011201u, 0x02021101u, 0x00001011u, 0x00001110u,
    0x00001111u, 0x00001112u, 0x00011111u, 0x00011210u, 0x00011212u, 0x00021211u, 0x01001010u, 0x01001111u,
    0x01001212u, 0x01011010u, 0x01011011u, 0x01011110u, 0x01011111u, 0x01011112u, 0x01011211u, 0x01021010u,
    0x01021012u, 0x01021111u, 0x01021210u, 0x01021212u, 0x02001011u, 0x02011011u, 0x02011111u, 0x02011210u,
    0x02011212u, 0x02021011u, 0x02021110u, 0x02021111u, 0x02021112u, 0x02021211u, 0x00011120u, 0x00011221u,
    0x01001021u, 0x01001120u, 0x01011020u, 0x01011022u, 0x01011121u, 0x01011220u, 0x01021020u, 0x01021021u,
    0x01021122u, 0x01021221u, 0x02001121u, 0x02011021u, 0x02011120u, 0x02011221u, 0x00002000u, 0x00002002u,
    0x00002200u, 0x00002202u, 0x00012101u, 0x00022000u, 0x00022002u, 0x00022200u, 0x00022202u, 0x01002101u,
    0x01012001u, 0x01012102u, 0x01022101u, 0x02002000u, 0x02002002u, 0x02002200u, 0x02002202u, 0x02012101u,
    0x02022000u, 0x02022002u, 0x02022200u, 0x02022202u, 0x00002111u, 0x00012011u, 0x00012110u, 0x00012211u,
    0x00022110u, 0x00022111u, 0x01002011u, 0x01012010u, 0x01012011u, 0x01012111u, 0x01022011u, 0x01022110u,
    0x01022211u, 0x02012011u, 0x02012110u, 0x02012112u, 0x02012211u, 0x02022111u, 0x00002020u, 0x00002022u,
    0x00002220u, 0x00002222u, 0x00012121u, 0x00022020u, 0x00022022u, 0x00022220u, 0x00022222u, 0x01002121u,
    0x01012021u, 0x01012221u, 0x01022021u, 0x01022121u, 0x02002020u, 0x02002022u, 0x02002121u, 0x02002220u,
    0x02002222u, 0x02012121u, 0x02022020u, 0x02022022u, 0x02022220u, 0x02022222u, 0x00110000u, 0x00110001u,
    0x00110100u, 0x00110201u, 0x00120100u, 0x00120101u, 0x01100001u, 0x01100100u, 0x01110000u, 0x01110101u,
    0x01110200u, 0x01120001u, 0x01120100u, 0x01120101u, 0x01120201u, 0x02110001u, 0x02110100u, 0x02110102u,
    0x02120001u, 0x02120101u, 0x00100011u, 0x00100110u, 0x00100112u, 0x00100211u, 0x00110010u, 0x00110012u,
    0x00110111u, 0x00110210u, 0x00120011u, 0x00120110u, 0x00120211u, 0x01100111u, 0x01100212u, 0x01110010u,
    0x01110011u, 0x01110012u, 0x01110110u, 0x01110111u, 0x01110112u, 0x01110211u, 0x01120010u, 0x01120111u,
    0x02100110u, 0x02110012u, 0x02110111u, 0x02120011u, 0x02120110u, 0x00110021u, 0x00110120u, 0x00110122u,
    0x00120121u, 0x01100020u, 0x01100122u, 0x01100221u, 0x01110022u, 0x01110121u, 0x01110220u, 0x01110222u,
    0x01120120u, 0x01120122u, 0x02100121u, 0x02110021u, 0x02110120u, 0x02110122u, 0x02120121u, 0x00101001u,
    0x00101102u, 0x00101201u, 0x00111100u, 0x00111101u, 0x00111200u, 0x00111201u, 0x00121001u, 0x00121102u,
    0x01101001u, 0x01101101u, 0x01101102u, 0x01101200u, 0x01101202u, 0x01111001u, 0x01111100u, 0x01111101u,
    0x01111102u, 0x01111201u, 0x01121002u, 0x01121101u, 0x01121200u, 0x02101100u, 0x02101201u, 0x02111000u,
    0x02111100u, 0x02111101u, 0x02111200u, 0x02111201u, 0x02111202u, 0x02121001u, 0x02121100u, 0x02121101u,
    0x02121201u, 0x00101012u, 0x00101111u, 0x00101212u, 0x00111011u, 0x00111110u, 0x00111111u, 0x00111112u,
    0x00111211u, 0x00121010u, 0x00121012u, 0x00121111u, 0x00121210u, 0x00121212u, 0x01101011u, 0x01101110u,
    0x01101111u, 0x01101112u, 0x01111011u, 0x01111012u, 0x01111110u, 0x01111111u, 0x01111112u, 0x01111211u,
    0x01111212u, 0x01121011u, 0x01121110u, 0x01121111u, 0x01121112u, 0x01121211u, 0x02101010u, 0x02101012u,
    0x02101110u, 0x02101111u, 0x02101210u, 0x02101212u, 0x02111010u, 0x02111011u, 0x02111110u, 0x02111111u,
    0x02111112u, 0x02111211u, 0x02111212u, 0x02121010u, 0x02121012u, 0x02121111u, 0x00101021u, 0x00101120u,
    0x00101121u, 0x00101122u, 0x00111121u, 0x00111122u, 0x00111220u, 0x00111222u, 0x00121021u, 0x00121122u,
    0x01101020u, 0x01101022u, 0x01101120u, 0x01101121u, 0x01101220u, 0x01101222u, 0x01111021u, 0x01111121u,
    0x01111122u, 0x01111220u, 0x01111221u, 0x01121021u, 0x01121120u, 0x01121121u, 0x01121220u, 0x01121221u,
    0x01121222u, 0x02101122u, 0x02101222u, 0x02111022u, 0x02111121u, 0x02121120u, 0x02121221u, 0x00112001u,
    0x00112102u, 0x00122101u, 0x01102001u, 0x01102100u, 0x01102102u, 0x01102201u, 0x01112000u, 0x01112101u,
    0x01112200u, 0x01112202u, 0x01122000u, 0x01122001u, 0x01122100u, 0x01122102u, 0x01122201u, 0x02102101u,
    0x02112001u, 0x02112100u, 0x02122101u, 0x00112010u, 0x00112012u, 0x00112111u, 0x00112212u, 0x00122011u,
    0x00122111u, 0x01102012u, 0x01102110u, 0x01102111u, 0x01102210u, 0x01112011u, 0x01112110u, 0x01112111u,
    0x01112112u, 0x01112211u, 0x01112212u, 0x01122010u, 0x01122111u, 0x01122212u, 0x02102211u, 0x02112011u,
    0x02112012u, 0x02112111u, 0x02112210u, 0x02122011u, 0x02122112u, 0x02122211u, 0x00102221u, 0x00112122u,
    0x00122120u, 0x00122122u, 0x01102120u, 0x01102122u, 0x01102221u, 0x01112020u, 0x01112022u, 0x01112121u,
    0x01112220u, 0x01122021u, 0x01122122u, 0x01122221u, 0x02102121u, 0x02112021u, 0x02112122u, 0x02112222u,
    0x00200000u, 0x00200002u, 0x00200200u, 0x00200202u, 0x00210101u, 0x00220000u, 0x00220002u, 0x00220101u,
    0x00220200u, 0x00220202u, 0x01200101u, 0x01210001u, 0x01210201u, 0x01220001u, 0x01220101u, 0x02200000u,
    0x02200002u, 0x02200200u, 0x02200202u, 0x02210101u, 0x02220000u, 0x02220002u, 0x02220101u, 0x02220200u,
    0x02220202u, 0x00200111u, 0x00210011u, 0x00210110u, 0x00210211u, 0x00220111u, 0x01200012u, 0x01200110u,
    0x01200211u, 0x01210111u, 0x01210210u, 0x01210212u, 0x01220011u, 0x01220110u, 0x01220111u, 0x01220112u,
    0x02200111u, 0x02210010u, 0x02210112u, 0x02210211u, 0x02220111u, 0x00200021u, 0x00200220u, 0x00200222u,
    0x00210021u, 0x00210121u, 0x00220020u, 0x00220022u, 0x00220220u, 0x00220222u, 0x01200121u, 0x01210021u,
    0x01210122u, 0x01210221u, 0x01220121u, 0x02200021u, 0x02200220u, 0x02200222u, 0x02210021u, 0x02210121u,
    0x02220020u, 0x02220022u, 0x02220220u, 0x02220222u, 0x00201101u, 0x00211100u, 0x00211102u, 0x00211201u,
    0x00221101u, 0x01201100u, 0x01201101u, 0x01201102u, 0x01201201u, 0x01211002u, 0x01211101u, 0x01211200u,
    0x01211202u, 0x01221102u, 0x02201101u, 0x02211001u, 0x02211100u, 0x02211201u, 0x02221001u, 0x02221101u,
    0x00201211u, 0x00211111u, 0x00221011u, 0x00221211u, 0x01201010u, 0x01201111u, 0x01201210u, 0x01211011u,
    0x01211110u, 0x01211111u, 0x01211211u, 0x01221012u, 0x01221111u, 0x01221210u, 0x02201211u, 0x02211010u,
    0x02211110u, 0x02211111u, 0x02211210u, 0x02211212u, 0x02221011u, 0x02221110u, 0x02221112u, 0x02221211u,
    0x00201121u, 0x00211020u, 0x00211022u, 0x00211221u, 0x00221121u, 0x01201021u, 0x01201221u, 0x01211121u,
    0x01221020u, 0x01221021u, 0x01221221u, 0x02201120u, 0x02201122u, 0x02211020u, 0x02211222u, 0x00202000u,
    0x00202002u, 0x00202200u, 0x00202202u, 0x00212101u, 0x00222000u, 0x00222002u, 0x00222200u, 0x00222202u,
    0x01202101u, 0x01212001u, 0x01212100u, 0x01222101u, 0x02202000u, 0x02202002u, 0x02202200u, 0x02202202u,
    0x02222000u, 0x02222002u, 0x02222200u, 0x02222202u, 0x00202211u, 0x00212011u, 0x00212110u, 0x00212211u,
    0x00222111u, 0x01202112u, 0x01202211u, 0x01212012u, 0x01212111u, 0x01222011u, 0x01222110u, 0x01222112u,
    0x01222211u, 0x02202111u, 0x02212010u, 0x02212112u, 0x02212211u, 0x02222110u, 0x02222111u, 0x00202020u,
    0x00202022u, 0x00202220u, 0x00202222u, 0x00222020u, 0x00222022u, 0x00222220u, 0x00222222u, 0x01202121u,
    0x01212021u, 0x01212122u, 0x01212221u, 0x01222121u, 0x02202020u, 0x02202022u, 0x02202220u, 0x02202222u,
    0x02212121u, 0x02222020u, 0x02222022u, 0x02222220u, 0x02222222u, 0x10000101u, 0x10010001u, 0x10010102u,
    0x10020101u, 0x11000201u, 0x11010002u, 0x11010101u, 0x11010200u, 0x11010202u, 0x11020001u, 0x11020100u,
    0x11020102u, 0x12010100u, 0x12010201u, 0x12020001u, 0x12020102u, 0x10000010u, 0x10000011u, 0x10000110u,
    0x10000112u, 0x10000211u, 0x10010012u, 0x10010111u, 0x10010112u, 0x10010210u, 0x10010212u, 0x10020011u,
    0x10020112u, 0x10020211u, 0x11000111u, 0x11000210u, 0x11000212u, 0x11010011u, 0x11010110u, 0x11010111u,
    0x11010112u, 0x11010211u, 0x11010212u, 0x11020111u, 0x11020210u, 0x11020212u, 0x12000011u, 0x12000110u,
    0x12000112u, 0x12010010u, 0x12010012u, 0x12010111u, 0x12020010u, 0x12020011u, 0x12020012u, 0x10000121u,
    0x10010021u, 0x10010120u, 0x10010122u, 0x10020121u, 0x11000021u, 0x11010022u, 0x11010121u, 0x11010222u,
    0x11020120u, 0x11020221u, 0x12000221u, 0x12010120u, 0x12020121u, 0x10001001u, 0x10011101u, 0x10011201u,
    0x10021201u, 0x11001101u, 0x11001200u, 0x11001202u, 0x11011001u, 0x11011100u, 0x11011101u, 0x11011102u,
    0x11021001u, 0x11021002u, 0x11021101u, 0x11021200u, 0x11021202u, 0x12001001u, 0x12001102u, 0x12001201u,
    0x12011000u, 0x12011002u, 0x12011101u, 0x12021000u, 0x12021001u, 0x12021201u, 0x10001011u, 0x10001012u,
    0x10001111u, 0x10001212u, 0x10011011u, 0x10011110u, 0x10011111u, 0x10011112u, 0x10011211u, 0x10021010u,
    0x10021111u, 0x10021212u, 0x11001011u, 0x11001110u, 0x11001111u, 0x11001112u, 0x11001211u, 0x11011010u,
    0x11011011u, 0x11011110u, 0x11011111u, 0x11011112u, 0x11011210u, 0x11011211u, 0x11021011u, 0x11021110u,
    0x11021111u, 0x11021112u, 0x11021211u, 0x12001012u, 0x12001110u, 0x12001111u, 0x12001210u, 0x12011011u,
    0x12011110u, 0x12011111u, 0x12011112u, 0x12011211u, 0x12011212u, 0x12021111u, 0x12021210u, 0x12021212u,
    0x10001021u, 0x10001121u, 0x10001221u, 0x10011120u, 0x10011121u, 0x10011220u, 0x10011222u, 0x10021021u,
    0x10021120u, 0x10021221u, 0x11001020u, 0x11001022u, 0x11001121u, 0x11001220u, 0x11011020u, 0x11011021u,
    0x11011022u, 0x11011121u, 0x11011122u, 0x11011221u, 0x11021022u, 0x11021121u, 0x11021220u, 0x12001021u,
    0x12001121u, 0x12001222u, 0x12011120u, 0x12011121u, 0x12021021u, 0x12021120u, 0x12021122u, 0x10002101u,
    0x10012001u, 0x10012101u, 0x10012202u, 0x10022101u, 0x11002002u, 0x11002201u, 0x11012000u, 0x11012101u,
    0x11012200u, 0x11022001u, 0x11022100u, 0x11022102u, 0x11022201u, 0x12002101u, 0x12012001u, 0x12012100u,
    0x12012102u, 0x12012201u, 0x12022101u, 0x10002011u, 0x10002111u, 0x10002112u, 0x10002212u, 0x10012010u,
    0x10012110u, 0x10012111u, 0x10012210u, 0x10022011u, 0x10022110u, 0x10022112u, 0x11002010u, 0x11002111u,
    0x11002212u, 0x11012011u, 0x11012012u, 0x11012110u, 0x11012111u, 0x11012112u, 0x11012211u, 0x11022010u,
    0x11022012u, 0x11022111u, 0x11022112u, 0x11022212u, 0x12002112u, 0x12002211u, 0x12012012u, 0x12012111u,
    0x12012112u, 0x12012210u, 0x12022011u, 0x12022110u, 0x12022112u, 0x12022211u, 0x10012122u, 0x11002120u,
    0x11002122u, 0x11002221u, 0x11012121u, 0x11012220u, 0x11012222u, 0x11022120u, 0x11022221u, 0x12012120u,
    0x12022121u, 0x10100001u, 0x10100100u, 0x10100101u, 0x10100102u, 0x10100201u, 0x10110002u, 0x10110101u,
    0x10110202u, 0x10120001u, 0x10120100u, 0x10120201u, 0x11100000u, 0x11100101u, 0x11100200u, 0x11110001u,
    0x11110100u, 0x11110101u, 0x11110102u, 0x11110201u, 0x11120101u, 0x11120200u, 0x12100102u, 0x12100201u,
    0x12110101u, 0x12110200u, 0x12120000u, 0x12120001u, 0x12120102u, 0x12120201u, 0x10100111u, 0x10100210u,
    0x10100211u, 0x10100212u, 0x10110011u, 0x10110110u, 0x10110111u, 0x10110112u, 0x10110210u, 0x10110211u,
    0x10120010u, 0x10120111u, 0x10120112u, 0x10120210u, 0x10120212u, 0x11100011u, 0x11100110u, 0x11100111u,
    0x11100112u, 0x11100211u, 0x11110010u, 0x11110011u, 0x11110012u, 0x11110110u, 0x11110111u, 0x11110112u,
    0x11110210u, 0x11110211u, 0x11110212u, 0x11120011u, 0x11120110u, 0x11120111u, 0x11120112u, 0x11120211u,
    0x12100012u, 0x12100111u, 0x12110011u, 0x12110110u, 0x12110111u, 0x12110112u, 0x12110211u, 0x12120010u,
    0x12120111u, 0x12120212u, 0x10100021u, 0x10100122u, 0x10110022u, 0x10110121u, 0x10110222u, 0x10120021u,
    0x10120120u, 0x11100022u, 0x11100121u, 0x11100222u, 0x11110021u, 0x11110120u, 0x11110121u, 0x11110122u,
    0x11110221u, 0x11120022u, 0x11120121u, 0x12100121u, 0x12110020u, 0x12110022u, 0x12110121u, 0x12110221u,
    0x12110222u, 0x12120120u, 0x10101100u, 0x10101101u, 0x10111001u, 0x10111100u, 0x10111101u, 0x10111102u,
    0x10111200u, 0x10111201u, 0x10121001u, 0x10121101u, 0x10121200u, 0x10121202u, 0x11101001u, 0x11101100u,
    0x11101101u, 0x11101102u, 0x11101201u, 0x11101202u, 0x11111000u, 0x11111001u, 0x11111100u, 0x11111101u,
    0x11111102u, 0x11111200u, 0x11111201u, 0x11111202u, 0x11121001u, 0x11121002u, 0x11121100u, 0x11121101u,
    0x11121102u, 0x11121201u, 0x12101000u, 0x12101200u, 0x12101202u, 0x12111001u, 0x12111100u, 0x12111101u,
    0x12111102u, 0x12111201u, 0x12121001u, 0x12121100u, 0x12121101u, 0x12121202u, 0x10101011u, 0x10101012u,
    0x10101110u, 0x10101111u, 0x10101112u, 0x10101211u, 0x10111010u, 0x10111011u, 0x10111012u, 0x10111110u,
    0x10111111u, 0x10111112u, 0x10111211u, 0x10111212u, 0x10121011u, 0x10121110u, 0x10121111u, 0x10121112u,
    0x10121211u, 0x11101010u, 0x11101011u, 0x11101012u, 0x11101110u, 0x11101111u, 0x11101112u, 0x11101210u,
    0x11101211u, 0x11111010u, 0x11111011u, 0x11111012u, 0x11111110u, 0x11111111u, 0x11111112u, 0x11111210u,
    0x11111211u, 0x11111212u, 0x11121010u, 0x11121011u, 0x11121110u, 0x11121111u, 0x11121112u, 0x11121210u,
    0x11121211u, 0x11121212u, 0x12101011u, 0x12101110u, 0x12101111u, 0x12101211u, 0x12101212u, 0x12111010u,
    0x12111011u, 0x12111110u, 0x12111111u, 0x12111112u, 0x12111210u, 0x12111211u, 0x12121011u, 0x12121110u,
    0x12121111u, 0x12121112u, 0x12121211u, 0x10101020u, 0x10101021u, 0x10101022u, 0x10101120u, 0x10101122u,
    0x10101220u, 0x10101221u, 0x10111021u, 0x10111120u, 0x10111121u, 0x10111220u, 0x10111221u, 0x10121020u,
    0x10121021u, 0x10121022u, 0x10121120u, 0x10121121u, 0x10121122u, 0x10121220u, 0x10121221u, 0x11101021u,
    0x11101121u, 0x11101122u, 0x11101220u, 0x11101221u, 0x11101222u, 0x11111020u, 0x11111021u, 0x11111022u,
    0x11111120u, 0x11111121u, 0x11111122u, 0x11111220u, 0x11111221u, 0x11111222u, 0x11121021u, 0x11121120u,
    0x11121121u, 0x11121221u, 0x12101022u, 0x12101121u, 0x12101122u, 0x12101220u, 0x12101221u, 0x12101222u,
    0x12111021u, 0x12111121u, 0x12111222u, 0x12121022u, 0x12121121u, 0x12121122u, 0x12121220u, 0x12121221u,
    0x10102100u, 0x10102101u, 0x10102102u, 0x10102201u, 0x10112000u, 0x10112101u, 0x10112200u, 0x10122001u,
    0x10122202u, 0x11102101u, 0x11102200u, 0x11102202u, 0x11112001u, 0x11112100u, 0x11112101u, 0x11112102u,
    0x11112200u, 0x11112201u, 0x11122000u, 0x11122002u, 0x11122100u, 0x11122101u, 0x12102002u, 0x12102201u,
    0x12112000u, 0x12112002u, 0x12112101u, 0x12112200u, 0x12122001u, 0x12122201u, 0x10102011u, 0x10102012u,
    0x10102111u, 0x10102212u, 0x10112011u, 0x10112110u, 0x10112111u, 0x10112112u, 0x10112211u, 0x10122111u,
    0x11102011u, 0x11102110u, 0x11102111u, 0x11102112u, 0x11102211u, 0x11112010u, 0x11112011u, 0x11112012u,
    0x11112110u, 0x11112111u, 0x11112112u, 0x11112210u, 0x11112211u, 0x11112212u, 0x11122011u, 0x11122110u,
    0x11122111u, 0x11122112u, 0x11122211u, 0x12102011u, 0x12102111u, 0x12102211u, 0x12112011u, 0x12112110u,
    0x12112111u, 0x12112112u, 0x12112210u, 0x12112211u, 0x12122111u, 0x10102120u, 0x10102220u, 0x10112121u,
    0x10112222u, 0x10122020u, 0x10122121u, 0x10122122u, 0x10122221u, 0x11102121u, 0x11102220u, 0x11102221u,
    0x11112021u, 0x11112121u, 0x11112122u, 0x11112220u, 0x11112221u, 0x11122022u, 0x11122121u, 0x11122220u,
    0x11122222u, 0x12102021u, 0x12102222u, 0x12112022u, 0x12112121u, 0x12112122u, 0x12112220u, 0x12112222u,
    0x12122021u, 0x10200101u, 0x10210100u, 0x10210102u, 0x10210201u, 0x10220101u, 0x11200100u, 0x11210000u,
    0x11210101u, 0x11210102u, 0x11210200u, 0x11210202u, 0x11220001u, 0x11220100u, 0x11220102u, 0x11220201u,
    0x12200001u, 0x12210102u, 0x12220101u, 0x10200011u, 0x10200110u, 0x10200112u, 0x10200211u, 0x10210012u,
    0x10210111u, 0x10220011u, 0x10220012u, 0x10220112u, 0x10220211u, 0x11200111u, 0x11200211u, 0x11210011u,
    0x11210111u, 0x11210112u, 0x11210211u, 0x11220111u, 0x11220112u, 0x11220212u, 0x12200110u, 0x12200212u,
    0x12210012u, 0x12210111u, 0x12220011u, 0x12220112u, 0x12220211u, 0x10210021u, 0x10210122u, 0x10210221u,
    0x11200020u, 0x11200021u, 0x11200122u, 0x11210121u, 0x11210122u, 0x11210220u, 0x11220020u, 0x12200121u,
    0x12210021u, 0x12210122u, 0x12220121u, 0x10211001u, 0x10211002u, 0x10211101u, 0x10211102u, 0x10211202u,
    0x10221001u, 0x10221102u, 0x10221201u, 0x11201000u, 0x11201002u, 0x11201101u, 0x11201200u, 0x11201202u,
    0x11211001u, 0x11211100u, 0x11211101u, 0x11211102u, 0x11211201u, 0x11211202u, 0x11221000u, 0x11221002u,
    0x11221101u, 0x12201100u, 0x12201101u, 0x12201201u, 0x12211000u, 0x12211002u, 0x12211100u, 0x12211101u,
    0x12211102u, 0x12211200u, 0x12211202u, 0x12221001u, 0x12221100u, 0x12221201u, 0x10201111u, 0x10201210u,
    0x10201212u, 0x10211011u, 0x10211111u, 0x10211112u, 0x10211211u, 0x11201110u, 0x11201111u, 0x11201112u,
    0x11201211u, 0x11211010u, 0x11211011u, 0x11211110u, 0x11211111u, 0x11211112u, 0x11211211u, 0x11221011u,
    0x11221110u, 0x11221111u, 0x11221112u, 0x11221211u, 0x12201112u, 0x12201211u, 0x12201212u, 0x12211011u,
    0x12211111u, 0x12211112u, 0x12211211u, 0x12211212u, 0x12221012u, 0x12221111u, 0x12221112u, 0x12221210u,
    0x10201022u, 0x10201221u, 0x10211121u, 0x10221020u, 0x10221122u, 0x10221220u, 0x10221221u, 0x11201020u,
    0x11201121u, 0x11201220u, 0x11201222u, 0x11211021u, 0x11211120u, 0x11211121u, 0x11211122u, 0x11211220u,
    0x11211222u, 0x11221020u, 0x11221121u, 0x11221220u, 0x12201020u, 0x12201022u, 0x12201121u, 0x12201222u,
    0x12211120u, 0x12211122u, 0x12211220u, 0x12211221u, 0x12221020u, 0x12221120u, 0x12221122u, 0x12221222u,
    0x10212102u, 0x10212201u, 0x10222101u, 0x11202001u, 0x11212002u, 0x11212101u, 0x11212202u, 0x11222001u,
    0x11222201u, 0x12202101u, 0x12212001u, 0x12212200u, 0x12222102u, 0x10202011u, 0x10202110u, 0x10212010u,
    0x10212111u, 0x10222011u, 0x10222110u, 0x10222112u, 0x10222211u, 0x11202010u, 0x11202011u, 0x11202111u,
    0x11202112u, 0x11202210u, 0x11212011u, 0x11212110u, 0x11212111u, 0x11212112u, 0x11212211u, 0x11222010u,
    0x11222111u, 0x11222212u, 0x12202012u, 0x12202110u, 0x12202212u, 0x12212111u, 0x12222011u, 0x12222110u,
    0x12222111u, 0x12222211u, 0x10212021u, 0x10212122u, 0x10212220u, 0x11202021u, 0x11202120u, 0x11202221u,
    0x11212020u, 0x11212121u, 0x11212220u, 0x11212222u, 0x11222120u, 0x11222121u, 0x11222221u, 0x12202122u,
    0x12212120u, 0x12212220u, 0x12212222u, 0x12222122u, 0x20000000u, 0x20000002u, 0x20000200u, 0x20000202u,
    0x20020000u, 0x20020002u, 0x20020200u, 0x20020202u, 0x21000101u, 0x21010000u, 0x21010001u, 0x21010100u,
    0x21010102u, 0x21010201u, 0x21020101u, 0x22000000u, 0x22000002u, 0x22000200u, 0x22000202u, 0x22010101u,
    0x22020000u, 0x22020002u, 0x22020200u, 0x22020202u, 0x20000111u, 0x20010011u, 0x20010110u, 0x20010112u,
    0x20010211u, 0x20020111u, 0x21000011u, 0x21000110u, 0x21000211u, 0x21010010u, 0x21010012u, 0x21010111u,
    0x21010112u, 0x21010210u, 0x21010211u, 0x21020110u, 0x21020112u, 0x21020211u, 0x22000111u, 0x22000211u,
    0x22010110u, 0x22010112u, 0x22010211u, 0x22020111u, 0x20000020u, 0x20000022u, 0x20000220u, 0x20000222u,
    0x20010121u, 0x20020020u, 0x20020022u, 0x20020220u, 0x20020222u, 0x21010021u, 0x21010120u, 0x21010221u,
    0x21020121u, 0x22000020u, 0x22000022u, 0x22000220u, 0x22000222u, 0x22010121u, 0x22020020u, 0x22020022u,
    0x22020220u, 0x22020222u, 0x20011100u, 0x20011201u, 0x21001001u, 0x21001100u, 0x21011001u, 0x21011101u,
    0x21011202u, 0x21021001u, 0x21021100u, 0x21021201u, 0x22011100u, 0x22011201u, 0x20001011u, 0x20001211u,
    0x20011012u, 0x20011111u, 0x20011212u, 0x20021112u, 0x20021211u, 0x21001010u, 0x21001011u, 0x21001111u,
    0x21001210u, 0x21011011u, 0x21011110u, 0x21011111u, 0x21011112u, 0x21011211u, 0x21011212u, 0x21021111u,
    0x21021112u, 0x21021210u, 0x21021212u, 0x22001011u, 0x22001110u, 0x22001112u, 0x22001211u, 0x22011010u,
    0x22011012u, 0x22011111u, 0x22011210u, 0x22021112u, 0x20011021u, 0x20011122u, 0x20011221u, 0x20021121u,
    0x21001021u, 0x21001120u, 0x21001221u, 0x21001222u, 0x21011020u, 0x21011121u, 0x21011221u, 0x21011222u,
    0x21021021u, 0x21021122u, 0x21021222u, 0x22001121u, 0x22011021u, 0x22011222u, 0x22021120u, 0x20002000u,
    0x20002002u, 0x20002200u, 0x20002202u, 0x20012101u, 0x20022000u, 0x20022002u, 0x20022200u, 0x20022202u,
    0x21002001u, 0x21002101u, 0x21012001u, 0x21012100u, 0x21012201u, 0x21022101u, 0x21022201u, 0x22002000u,
    0x22002002u, 0x22002200u, 0x22002202u, 0x22012101u, 0x22022000u, 0x22022002u, 0x22022200u, 0x22022202u,
    0x20002111u, 0x20002112u, 0x20012011u, 0x20012110u, 0x20012112u, 0x20022111u, 0x21002011u, 0x21002110u,
    0x21002112u, 0x21002211u, 0x21012010u, 0x21012012u, 0x21012111u, 0x21012212u, 0x21022011u, 0x21022110u,
    0x22002111u, 0x22012112u, 0x22012211u, 0x22022111u, 0x20002020u, 0x20002022u, 0x20002220u, 0x20002222u,
    0x20012121u, 0x20022020u, 0x20022022u, 0x20022220u, 0x20022222u, 0x21002121u, 0x21012021u, 0x21012120u,
    0x21012122u, 0x22002020u, 0x22002022u, 0x22002220u, 0x22002222u, 0x22012121u, 0x22022020u, 0x22022022u,
    0x22022220u, 0x22022222u, 0x20100101u, 0x20110001u, 0x20110102u, 0x20110200u, 0x20110201u, 0x20120101u,
    0x21100001u, 0x21100102u, 0x21100201u, 0x21110101u, 0x21110200u, 0x21110202u, 0x21120201u, 0x21120202u,
    0x22100101u, 0x22110001u, 0x22110100u, 0x22110102u, 0x22110201u, 0x22120101u, 0x20100011u, 0x20100110u,
    0x20100112u, 0x20100211u, 0x20110010u, 0x20110111u, 0x20110210u, 0x20110212u, 0x20120011u, 0x20120110u,
    0x20120112u, 0x20120211u, 0x21100010u, 0x21100111u, 0x21110010u, 0x21110011u, 0x21110110u, 0x21110111u,
    0x21110112u, 0x21110211u, 0x21120012u, 0x21120111u, 0x22100110u, 0x22100112u, 0x22110012u, 0x22110111u,
    0x22110210u, 0x22120011u, 0x22120110u, 0x22120112u, 0x22120211u, 0x20100121u, 0x20110021u, 0x20110120u,
    0x20110221u, 0x20120121u, 0x21100120u, 0x21100122u, 0x21100221u, 0x21110020u, 0x21110022u, 0x21110121u,
    0x21110220u, 0x21120122u, 0x21120221u, 0x22100121u, 0x22110120u, 0x22110122u, 0x22120221u, 0x20101001u,
    0x20101100u, 0x20101102u, 0x20111000u, 0x20111101u, 0x20111200u, 0x20121102u, 0x21101000u, 0x21101202u,
    0x21111001u, 0x21111100u, 0x21111101u, 0x21111102u, 0x21111200u, 0x21111201u, 0x21121000u, 0x21121001u,
    0x21121002u, 0x21121101u, 0x22101100u, 0x22101102u, 0x22111002u, 0x22111100u, 0x22111101u, 0x22111200u,
    0x22121001u, 0x22121201u, 0x20101010u, 0x20101111u, 0x20101210u, 0x20101212u, 0x20111010u, 0x20111011u,
    0x20111110u, 0x20111111u, 0x20111112u, 0x20111211u, 0x20121011u, 0x20121111u, 0x20121211u, 0x20121212u,
    0x21101011u, 0x21101110u, 0x21101111u, 0x21101112u, 0x21101211u, 0x21111010u, 0x21111011u, 0x21111012u,
    0x21111110u, 0x21111111u, 0x21111112u, 0x21111210u, 0x21111211u, 0x21111212u, 0x21121011u, 0x21121110u,
    0x21121111u, 0x21121112u, 0x21121211u, 0x22101011u, 0x22101111u, 0x22101210u, 0x22111011u, 0x22111012u,
    0x22111110u, 0x22111111u, 0x22111112u, 0x22111211u, 0x22111212u, 0x22121010u, 0x22121012u, 0x22121111u,
    0x22121210u, 0x22121212u, 0x20101021u, 0x20101120u, 0x20111020u, 0x20111121u, 0x20111221u, 0x20121020u,
    0x20121122u, 0x20121221u, 0x21101121u, 0x21101220u, 0x21101221u, 0x21111021u, 0x21111022u, 0x21111121u,
    0x21111122u, 0x21111221u, 0x21121121u, 0x21121220u, 0x22101022u, 0x22101120u, 0x22101221u, 0x22101222u,
    0x22111022u, 0x22111120u, 0x22111121u, 0x22121120u, 0x22121122u, 0x22121221u, 0x20102101u, 0x20112102u,
    0x20112201u, 0x20122101u, 0x21102001u, 0x21102102u, 0x21112000u, 0x21112002u, 0x21112101u, 0x21112102u,
    0x21112202u, 0x21122100u, 0x21122101u, 0x22102101u, 0x22112001u, 0x22112102u, 0x22112201u, 0x22122101u,
    0x20102110u, 0x20102112u, 0x20102211u, 0x20112010u, 0x20112012u, 0x20112111u, 0x20112210u, 0x20112212u,
    0x20122010u, 0x20122011u, 0x20122110u, 0x20122112u, 0x21102010u, 0x21102012u, 0x21102111u, 0x21102210u,
    0x21102212u, 0x21112011u, 0x21112110u, 0x21112111u, 0x21112112u, 0x21112211u, 0x21122012u, 0x21122111u,
    0x21122112u, 0x21122212u, 0x22102011u, 0x22102110u, 0x22112010u, 0x22112012u, 0x22112111u, 0x22112212u,
    0x22122011u, 0x22122112u, 0x20102121u, 0x20112121u, 0x20122121u, 0x21102120u, 0x21102122u, 0x21102221u,
    0x21112020u, 0x21112121u, 0x21112220u, 0x21122021u, 0x22102121u, 0x22112021u, 0x22112120u, 0x22112121u,
    0x22112122u, 0x20200000u, 0x20200002u, 0x20200200u, 0x20200202u, 0x20210101u, 0x20220000u, 0x20220002u,
    0x20220200u, 0x20220202u, 0x21200101u, 0x21210001u, 0x21210100u, 0x21210102u, 0x21210201u, 0x22200000u,
    0x22200002u, 0x22200200u, 0x22200202u, 0x22210101u, 0x22220000u, 0x22220002u, 0x22220200u, 0x22220202u,
    0x20200111u, 0x20200211u, 0x20210011u, 0x20210110u, 0x20210112u, 0x20210211u, 0x20210212u, 0x21200112u,
    0x21200211u, 0x21210011u, 0x21210111u, 0x21210210u, 0x21210212u, 0x21220011u, 0x21220110u, 0x22200111u,
    0x22210010u, 0x22210012u, 0x22210112u, 0x22210211u, 0x20200022u, 0x20200220u, 0x20200222u, 0x20210020u,
    0x20210221u, 0x20220022u, 0x20220220u, 0x20220222u, 0x21200121u, 0x21210021u, 0x21210122u, 0x21210221u,
    0x21220121u, 0x22200020u, 0x22200022u, 0x22200220u, 0x22200222u, 0x22210121u, 0x22220020u, 0x22220022u,
    0x22220220u, 0x22220222u, 0x20211201u, 0x20221101u, 0x21201001u, 0x21201100u, 0x21211000u, 0x21211100u,
    0x21211101u, 0x21211200u, 0x21211202u, 0x21221001u, 0x21221101u, 0x21221102u, 0x21221200u, 0x21221201u,
    0x22201101u, 0x20201112u, 0x20201211u, 0x20211010u, 0x20211012u, 0x20211111u, 0x20211210u, 0x20221112u,
    0x20221211u, 0x21201012u, 0x21201111u, 0x21211011u, 0x21211110u, 0x21211111u, 0x21211112u, 0x21211211u,
    0x21221111u, 0x21221212u, 0x22201011u, 0x22201110u, 0x22201111u, 0x22201112u, 0x22201211u, 0x22211012u,
    0x22211111u, 0x22211210u, 0x20201121u, 0x20211021u, 0x20211122u, 0x20211222u, 0x20221021u, 0x20221121u,
    0x21201120u, 0x21201122u, 0x21201222u, 0x21211022u, 0x21211121u, 0x21211122u, 0x21211220u, 0x21221020u,
    0x21221022u, 0x22201122u, 0x22211020u, 0x22211121u, 0x22211122u, 0x22211221u, 0x22221021u, 0x22221120u,
    0x22221122u, 0x20202000u, 0x20202002u, 0x20202200u, 0x20202202u, 0x20222000u, 0x20222002u, 0x20222200u,
    0x20222202u, 0x21212001u, 0x21212100u, 0x21212102u, 0x21212201u, 0x22202000u, 0x22202002u, 0x22202200u,
    0x22202202u, 0x22212101u, 0x22222000u, 0x22222002u, 0x22222200u, 0x22222202u, 0x20202111u, 0x20212110u,
    0x20212211u, 0x20222011u, 0x20222111u, 0x21202011u, 0x21212010u, 0x21212111u, 0x21212212u, 0x21222011u,
    0x21222112u, 0x21222211u, 0x22212010u, 0x22212112u, 0x20202020u, 0x20202022u, 0x20202220u, 0x20202222u,
    0x20222020u, 0x20222022u, 0x20222220u, 0x20222222u, 0x21212021u, 0x21212120u, 0x21212122u, 0x22202020u,
    0x22202022u, 0x22202220u, 0x22202222u, 0x22212121u, 0x22222020u, 0x22222022u, 0x22222220u, 0x22222222u,
};
// ---- GGUF codebooks: END adapted-from-llama.cpp block ---------------------------------------

__device__ __forceinline__ float load_half(const uint8_t* p) {
  const unsigned short raw = static_cast<unsigned short>(p[0] | (p[1] << 8));
  return __half2float(__ushort_as_half(raw));
}

// Blocks are 50..110 bytes, so wider fields are assembled bytewise.
__device__ __forceinline__ uint32_t load_u16(const uint8_t* p) { return p[0] | (static_cast<uint32_t>(p[1]) << 8); }

__device__ __forceinline__ uint32_t load_u32(const uint8_t* p) { return load_u16(p) | (load_u16(p + 2) << 16); }

/// ggml ksigns_iq2xs: seven sign bits and an even-parity bit in bit 7.
__device__ __forceinline__ uint32_t gguf_signs(uint32_t seven) { return seven | ((__popc(seven) & 1u) << 7); }

/// Sign of value `j` of a group (ggml kmask_iq2xs[j] = 1 << j).
__device__ __forceinline__ float gguf_sign(uint32_t signs, int j) { return ((signs >> j) & 1u) ? -1.0f : 1.0f; }

template <typename word_t>
__device__ __forceinline__ float grid_byte(word_t word, int j) {
  return static_cast<float>((word >> (8 * j)) & 0xFFu);
}

/// Entry `j` (-1, 0 or 1) of ggml iq1s_grid[idx]; iq1s_grid_gpu stores value + 1 per nibble, entries 0-3 in
/// the low nibbles and 4-7 in the high nibbles of the same four bytes.
__device__ __forceinline__ float gguf_iq1s_value(uint32_t idx, int j) {
  const uint32_t word = kIq1sGridGpu[idx];
  return static_cast<float>(static_cast<int>((word >> (8 * (j & 3) + 4 * (j >> 2))) & 0xFu) - 1);
}

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

/// Value `e` of one block.  Every product/difference is a separate IEEE fp32 operation.
template <GGUFType kType>
__device__ __forceinline__ float gguf_element(const uint8_t* b, int e) {
  if constexpr (kType == GGUFType::kQ4_0 || kType == GGUFType::kQ4_1) {
    constexpr bool kHasMin = kType == GGUFType::kQ4_1;
    const float d = load_half(b);
    const uint8_t* qs = b + (kHasMin ? 4 : 2);
    const int j = e & 15;
    const int q = e < 16 ? (qs[j] & 0xF) : (qs[j] >> 4);
    if constexpr (kHasMin) return __fadd_rn(__fmul_rn(static_cast<float>(q), d), load_half(b + 2));
    return __fmul_rn(static_cast<float>(q - 8), d);
  } else if constexpr (kType == GGUFType::kQ5_0 || kType == GGUFType::kQ5_1) {
    constexpr bool kHasMin = kType == GGUFType::kQ5_1;
    const float d = load_half(b);
    const uint8_t* qh_p = b + (kHasMin ? 4 : 2);
    const uint32_t qh = qh_p[0] | (qh_p[1] << 8) | (qh_p[2] << 16) | (static_cast<uint32_t>(qh_p[3]) << 24);
    const uint8_t* qs = qh_p + 4;
    const int j = e & 15;
    const int low = e < 16 ? (qs[j] & 0xF) : (qs[j] >> 4);
    const int q = low | static_cast<int>(((qh >> (e < 16 ? j : j + 16)) & 1u) << 4);
    if constexpr (kHasMin) return __fadd_rn(__fmul_rn(static_cast<float>(q), d), load_half(b + 2));
    return __fmul_rn(static_cast<float>(q - 16), d);
  } else if constexpr (kType == GGUFType::kQ8_0) {
    return __fmul_rn(static_cast<float>(static_cast<int8_t>(b[2 + e])), load_half(b));
  } else if constexpr (kType == GGUFType::kQ2_K) {
    // scales[16], qs[64], d, dmin
    const uint8_t* scales = b;
    const uint8_t* qs = b + 16;
    const float d = load_half(b + 80);
    const float dmin = load_half(b + 82);
    const int half_ = e >> 7;
    const int r = e & 127;
    const int j = r >> 5;
    const int rr = r & 31;
    const int sc = scales[half_ * 8 + j * 2 + (rr >> 4)];
    const int q = (qs[half_ * 32 + rr] >> (2 * j)) & 3;
    return __fsub_rn(__fmul_rn(__fmul_rn(d, static_cast<float>(sc & 0xF)), static_cast<float>(q)),
                     __fmul_rn(dmin, static_cast<float>(sc >> 4)));
  } else if constexpr (kType == GGUFType::kQ3_K) {
    // hmask[32], qs[64], scales[12], d
    const uint8_t* hmask = b;
    const uint8_t* qs = b + 32;
    const uint8_t* sc = b + 96;
    const float d_all = load_half(b + 108);
    const int half_ = e >> 7;
    const int r = e & 127;
    const int j = r >> 5;
    const int rr = r & 31;
    const int i = half_ * 8 + j * 2 + (rr >> 4);
    const int lo = i < 8 ? (sc[i] & 0xF) : (sc[i - 8] >> 4);
    const int hi = (sc[8 + (i & 3)] >> (2 * (i >> 2))) & 3;
    const int scale = (lo | (hi << 4)) - 32;
    const int q = ((qs[half_ * 32 + rr] >> (2 * j)) & 3) - ((hmask[rr] & (1 << (4 * half_ + j))) ? 0 : 4);
    return __fmul_rn(__fmul_rn(d_all, static_cast<float>(scale)), static_cast<float>(q));
  } else if constexpr (kType == GGUFType::kQ4_K || kType == GGUFType::kQ5_K) {
    constexpr bool kFive = kType == GGUFType::kQ5_K;
    const float d = load_half(b);
    const float dmin = load_half(b + 2);
    const uint8_t* scales = b + 4;
    [[maybe_unused]] const uint8_t* qh = b + 16;
    const uint8_t* qs = b + (kFive ? 48 : 16);
    const int j = e >> 6;
    const int r = e & 63;
    const int hi = r >> 5;
    const int l = r & 31;
    const int qb = qs[32 * j + l];
    int q = hi ? (qb >> 4) : (qb & 0xF);
    if constexpr (kFive) q += ((qh[l] >> (2 * j + hi)) & 1) << 4;
    int sc;
    int mn;
    scale_min_k4(2 * j + hi, scales, sc, mn);
    return __fsub_rn(__fmul_rn(__fmul_rn(d, static_cast<float>(sc)), static_cast<float>(q)),
                     __fmul_rn(dmin, static_cast<float>(mn)));
  } else if constexpr (kType == GGUFType::kQ6_K) {
    // ql[128], qh[64], scales[16], d
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
    const float d = load_half(b + 208);
    const int ip = e >> 7;
    const int r = e & 127;
    const int quad = r >> 5;
    const int l = r & 31;
    const int qlb = ql[64 * ip + l + ((quad & 1) ? 32 : 0)];
    const int low = quad >= 2 ? (qlb >> 4) : (qlb & 0xF);
    const int high = (qh[32 * ip + l] >> (2 * quad)) & 3;
    const int q = (low | (high << 4)) - 32;
    return __fmul_rn(__fmul_rn(d, static_cast<float>(sc[8 * ip + (l >> 4) + 2 * quad])),
                     static_cast<float>(q));
  } else if constexpr (kType == GGUFType::kQ8_K) {
    float d;
    memcpy(&d, b, sizeof(float));
    return __fmul_rn(d, static_cast<float>(static_cast<int8_t>(b[4 + e])));
  } else if constexpr (kType == GGUFType::kIQ4_NL) {
    const float d = load_half(b);
    const int j = e & 15;
    const int q = e < 16 ? (b[2 + j] & 0xF) : (b[2 + j] >> 4);
    return __fmul_rn(d, static_cast<float>(kIq4nlValues[q]));
  } else if constexpr (kType == GGUFType::kIQ2_XXS) {
    // d, qs[32] (uint16): per 32 values four grid indices (bytes 0-3) and a word of 4 x 7 sign bits + 4-bit scale
    const float d = load_half(b);
    const int ib32 = e >> 5;
    const int l = (e >> 3) & 3;
    const int j = e & 7;
    const uint8_t* q = b + 2 + 8 * ib32;
    const uint32_t aux = load_u32(q + 4);
    const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, static_cast<float>(aux >> 28))), 0.25f);
    const uint32_t signs = gguf_signs((aux >> (7 * l)) & 127u);
    return __fmul_rn(__fmul_rn(db, grid_byte(kIq2xxsGrid[q[l]], j)), gguf_sign(signs, j));
  } else if constexpr (kType == GGUFType::kIQ2_XS) {
    // d, qs[32] (uint16: 9-bit grid index, 7 sign bits), scales[8] (two 4-bit scales each)
    const float d = load_half(b);
    const int ib32 = e >> 5;
    const int l = (e >> 3) & 3;
    const int j = e & 7;
    const int sc = b[66 + ib32];
    const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, static_cast<float>(l < 2 ? (sc & 0xF) : (sc >> 4)))), 0.25f);
    const uint32_t q = load_u16(b + 2 + 2 * (4 * ib32 + l));
    return __fmul_rn(__fmul_rn(db, grid_byte(kIq2xsGrid[q & 511u], j)), gguf_sign(gguf_signs(q >> 9), j));
  } else if constexpr (kType == GGUFType::kIQ2_S) {
    // d, qs[64] (grid index low bits; the last 32 bytes are sign bytes), qh[8], scales[8]
    const float d = load_half(b);
    const int ib32 = e >> 5;
    const int l = (e >> 3) & 3;
    const int j = e & 7;
    const uint8_t* qs = b + 2;
    const int sc = b[74 + ib32];
    const float dl = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, static_cast<float>(l < 2 ? (sc & 0xF) : (sc >> 4)))), 0.25f);
    const uint32_t idx = qs[4 * ib32 + l] | ((static_cast<uint32_t>(b[66 + ib32]) << (8 - 2 * l)) & 0x300u);
    return __fmul_rn(__fmul_rn(dl, grid_byte(kIq2sGrid[idx], j)), gguf_sign(qs[32 + 4 * ib32 + l], j));
  } else if constexpr (kType == GGUFType::kIQ3_XXS) {
    // d, qs[64] (grid indices, two per group of 8), then per 32 values a word of 4 x 7 sign bits + 4-bit scale
    const float d = load_half(b);
    const int ib32 = e >> 5;
    const int l = (e >> 3) & 3;
    const int j = e & 7;
    const uint8_t* qs = b + 2;
    const uint32_t aux = load_u32(qs + 64 + 4 * ib32);
    const float db = __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, static_cast<float>(aux >> 28))), 0.5f);
    const uint32_t signs = gguf_signs((aux >> (7 * l)) & 127u);
    return __fmul_rn(__fmul_rn(db, grid_byte(kIq3xxsGrid[qs[8 * ib32 + 2 * l + (j >> 2)]], j & 3)),
                     gguf_sign(signs, j));
  } else if constexpr (kType == GGUFType::kIQ3_S) {
    // d, qs[64], qh[8], signs[32], scales[4]; two 32-value sub-blocks share one scale byte
    const float d = load_half(b);
    const int sub = e >> 5;
    const int l = (e >> 3) & 3;
    const int j = e & 7;
    const uint8_t* qs = b + 2;
    const int sc = (b[106 + (sub >> 1)] >> (4 * (sub & 1))) & 0xF;
    const float db = __fmul_rn(d, static_cast<float>(1 + 2 * sc));
    const uint32_t idx = qs[8 * sub + 2 * l + (j >> 2)] |
                         ((static_cast<uint32_t>(b[66 + sub]) << (8 - 2 * l - (j >> 2))) & 256u);
    return __fmul_rn(__fmul_rn(db, grid_byte(kIq3sGrid[idx], j & 3)), gguf_sign(b[74 + 4 * sub + l], j));
  } else if constexpr (kType == GGUFType::kIQ1_S) {
    // d, qs[32], qh[8] (uint16: 3 high index bits per group, 3-bit scale, delta sign)
    const float d = load_half(b);
    const int ib = e >> 5;
    const int l = (e >> 3) & 3;
    const int j = e & 7;
    const uint32_t qh = load_u16(b + 34 + 2 * ib);
    const float dl = __fmul_rn(d, static_cast<float>(2 * ((qh >> 12) & 7u) + 1));
    const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    const uint32_t idx = b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8);
    return __fmul_rn(dl, __fadd_rn(gguf_iq1s_value(idx, j), delta));
  } else if constexpr (kType == GGUFType::kIQ1_M) {
    // qs[32], qh[16], scales[8] (four uint16, each also carries one nibble of the fp16 block scale)
    const uint32_t s0 = load_u16(b + 48), s1 = load_u16(b + 50), s2 = load_u16(b + 52), s3 = load_u16(b + 54);
    const uint32_t d_bits = (s0 >> 12) | ((s1 >> 8) & 0x00F0u) | ((s2 >> 4) & 0x0F00u) | (s3 & 0xF000u);
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(d_bits)));
    const int ib = e >> 5;
    const int l = (e >> 3) & 3;
    const int j = e & 7;
    const uint32_t sc = load_u16(b + 48 + 2 * (ib >> 1));
    const float dl = __fmul_rn(d, static_cast<float>(2 * ((sc >> (6 * (ib & 1) + 3 * (l >> 1))) & 7u) + 1));
    const uint32_t qh = b[32 + 2 * ib + (l >> 1)];
    const uint32_t idx = b[4 * ib + l] | ((qh << (8 - 4 * (l & 1))) & 0x700u);
    const float delta = (qh & ((l & 1) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
    return __fmul_rn(dl, __fadd_rn(gguf_iq1s_value(idx, j), delta));
  } else {  // kIQ4_XS: d, scales_h(2), scales_l[4], qs[128]
    static_assert(kType == GGUFType::kIQ4_XS, "unhandled GGUF type");
    const float d = load_half(b);
    const int scales_h = b[2] | (b[3] << 8);
    const int ib = e >> 5;
    const int r = e & 31;
    const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    const int qb = b[8 + 16 * ib + (r & 15)];
    const int q = r < 16 ? (qb & 0xF) : (qb >> 4);
    return __fmul_rn(__fmul_rn(d, static_cast<float>(ls - 32)), static_cast<float>(kIq4nlValues[q]));
  }
}

template <typename out_t, GGUFType kType, int kBlockElems, int kBlockBytes>
__global__ void gguf_dequantize_blocks_kernel(out_t* __restrict__ out,
                                              const uint8_t* __restrict__ qweight,
                                              const int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t block = idx / kBlockElems;
  const int e = static_cast<int>(idx - block * kBlockElems);
  from_f<out_t>(out[idx], gguf_element<kType>(qweight + block * kBlockBytes, e));
}

// -----------------------------------------------------------------------------
// FP8 block scales -> UE8M0
// -----------------------------------------------------------------------------

__device__ __forceinline__ float e4m3_to_float(uint8_t x) { return dev::e4m3_to_float_fast(x); }

__device__ __forceinline__ uint8_t e4m3_from_float(float x) { return dev::e4m3_from_float_sat(x); }

/// One CTA per 128x128 block.  s_new = 2^ceil(log2 s_old) (DeepGEMM ceil_to_ue8m0: exponent
/// field + 1 on a non-zero mantissa, clamped to [1, 254]); the payload is requantized as
/// q' = e4m3(q * (s_old / s_new)) so q' * s_new stays within one e4m3 ulp of q * s_old.  A block
/// whose scale is already a power of two is left untouched, bit for bit.
__global__ void fp8_block_scale_to_ue8m0_blocks_kernel(uint8_t* __restrict__ weight,
                                                       const int64_t weight_ld,
                                                       float* __restrict__ scales,
                                                       const int64_t scales_ld,
                                                       uint8_t* __restrict__ ue8m0_out,
                                                       const int size_n, const int size_k) {
  const int bn = blockIdx.y;
  const int bk = blockIdx.x;
  const float s_old = scales[static_cast<int64_t>(bn) * scales_ld + bk];
  const uint32_t bits = __float_as_uint(s_old) & 0x7FFFFFFFu;
  uint32_t e = (bits >> 23) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
  e = e < 1u ? 1u : (e > 254u ? 254u : e);
  const float s_new = __uint_as_float(e << 23);
  __syncthreads();  // every thread has read s_old before thread 0 overwrites it
  if (threadIdx.x == 0) {
    scales[static_cast<int64_t>(bn) * scales_ld + bk] = s_new;
    if (ue8m0_out != nullptr) ue8m0_out[static_cast<int64_t>(bn) * scales_ld + bk] = static_cast<uint8_t>(e);
  }
  const float ratio = __fdiv_rn(s_old, s_new);
  if (ratio == 1.0f) return;
  const int n0 = bn * 128;
  const int k0 = bk * 128;
  const int n1 = min(n0 + 128, size_n);
  const int k1 = min(k0 + 128, size_k);
  const int rows = n1 - n0;
  const int cols = k1 - k0;
  const bool vec = ((cols & 15) == 0) && ((weight_ld & 15) == 0) && ((reinterpret_cast<uintptr_t>(weight) & 15) == 0);
  if (vec) {
    const int vpr = cols / 16;
    for (int i = threadIdx.x; i < rows * vpr; i += blockDim.x) {
      const int r = i / vpr;
      const int v = i - r * vpr;
      uint4* p = reinterpret_cast<uint4*>(weight + static_cast<int64_t>(n0 + r) * weight_ld + k0 + v * 16);
      uint4 raw = *p;
      uint8_t* bytes = reinterpret_cast<uint8_t*>(&raw);
      #pragma unroll
      for (int j = 0; j < 16; ++j) bytes[j] = e4m3_from_float(__fmul_rn(e4m3_to_float(bytes[j]), ratio));
      *p = raw;
    }
  } else {
    for (int i = threadIdx.x; i < rows * cols; i += blockDim.x) {
      const int r = i / cols;
      const int c = i - r * cols;
      uint8_t* p = weight + static_cast<int64_t>(n0 + r) * weight_ld + k0 + c;
      *p = e4m3_from_float(__fmul_rn(e4m3_to_float(*p), ratio));
    }
  }
}

// -----------------------------------------------------------------------------
// SmoothQuant
// -----------------------------------------------------------------------------

__global__ void smoothquant_compute_scales_elems_kernel(float* __restrict__ smooth,
                                                        const float* __restrict__ act_absmax,
                                                        const float* __restrict__ weight_absmax,
                                                        const float alpha, const int size_k) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= size_k) return;
  const float w = fmaxf(weight_absmax[k], 1e-5f);
  smooth[k] = fmaxf(powf(act_absmax[k], alpha) / powf(w, 1.0f - alpha), 1e-5f);
}

/// W[n, k] *= s[k] over an [N, K] matrix, eight elements per thread.
template <typename scalar_t>
__global__ void smoothquant_scale_rows_kernel(scalar_t* __restrict__ weight, const int64_t weight_ld,
                                              const float* __restrict__ smooth, const int size_n,
                                              const int size_k) {
  const int vpr = size_k / 8;
  const int64_t total = static_cast<int64_t>(size_n) * vpr;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t n = i / vpr;
    const int k = static_cast<int>(i - n * vpr) * 8;
    scalar_t* p = weight + n * weight_ld + k;
    dev::Pack<scalar_t, 8> v = dev::load_pack<8>(p);
    const float4 s0 = *reinterpret_cast<const float4*>(smooth + k);
    const float4 s1 = *reinterpret_cast<const float4*>(smooth + k + 4);
    const float s[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) from_f<scalar_t>(v.x[j], __fmul_rn(to_f<scalar_t>(v.x[j]), s[j]));
    dev::store_pack<8>(p, v);
  }
}

/// Scalar fallback for K % 8 != 0 or unaligned rows.
template <typename scalar_t>
__global__ void smoothquant_scale_rows_scalar_kernel(scalar_t* __restrict__ weight,
                                                     const int64_t weight_ld,
                                                     const float* __restrict__ smooth,
                                                     const int size_n, const int size_k) {
  const int64_t total = static_cast<int64_t>(size_n) * size_k;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t n = i / size_k;
    const int k = static_cast<int>(i - n * size_k);
    scalar_t* p = weight + n * weight_ld + k;
    from_f<scalar_t>(*p, __fmul_rn(to_f<scalar_t>(*p), smooth[k]));
  }
}

/// norm_weight[k] /= s[k]; norm_bias[k] /= s[k] (true fp32 division).
template <typename scalar_t>
__global__ void smoothquant_fold_norm_kernel(scalar_t* __restrict__ norm_weight,
                                             scalar_t* __restrict__ norm_bias,
                                             const float* __restrict__ smooth, const int size_k) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= size_k) return;
  const float s = smooth[k];
  from_f<scalar_t>(norm_weight[k], __fdiv_rn(to_f<scalar_t>(norm_weight[k]), s));
  if (norm_bias != nullptr) from_f<scalar_t>(norm_bias[k], __fdiv_rn(to_f<scalar_t>(norm_bias[k]), s));
}

/// Flat byte copy (16-byte vectors where possible).
__global__ void copy_bytes_kernel(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src,
                                  const int64_t bytes) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t vecs = bytes / 16;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < vecs; i += stride) {
    reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
  }
  for (int64_t i = vecs * 16 + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes; i += stride) {
    dst[i] = src[i];
  }
}

// TRT-LLM-gen row shuffle (shuffle_matrix_a with epilogue_tile_m = 128, the value every TensorRT-LLM call site
// passes): inside each block of 32 rows, row i moves to row (i % 4) * 8 + i / 4.
__device__ __forceinline__ int trtllm_dst_row(int i) { return (i & 3) * 8 + (i >> 2); }

__device__ __forceinline__ int trtllm_src_row(int j) { return (j & 7) * 4 + (j >> 3); }

/// One 16-byte word of a weight row per thread; rows are permuted inside their 32-row block.
__global__ void trtllm_shuffle_rows_kernel(uint4* __restrict__ out, const uint4* __restrict__ in,
                                           const int words_per_row, const int64_t total_words) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_words) return;
  const int64_t row = idx / words_per_row;
  const int64_t word = idx - row * words_per_row;
  const int64_t dst_row = (row & ~31LL) | trtllm_dst_row(static_cast<int>(row & 31));
  out[dst_row * words_per_row + word] = in[idx];
}

/// Scale factors: the same row shuffle followed by the 128x4 interleave (block_scale_interleave), written as one
/// gather.  One thread per output byte; output offset = (row/128 * cols/4 + col/4) * 512 + (row%32) * 16 +
/// ((row%128)/32) * 4 + col%4.
__global__ void trtllm_shuffle_scales_kernel(uint8_t* __restrict__ out, const uint8_t* __restrict__ in,
                                             const int rows, const int cols, const int64_t total) {
  const int64_t o = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (o >= total) return;
  const int64_t per_expert = static_cast<int64_t>(rows) * cols;
  const int64_t expert = o / per_expert;
  const int64_t t = o - expert * per_expert;
  const int64_t tile = t >> 9;
  const int within = static_cast<int>(t & 511);
  const int col_tiles = cols >> 2;
  const int64_t row_tile = tile / col_tiles;
  const int col = static_cast<int>(tile - row_tile * col_tiles) * 4 + (within & 3);
  const int64_t new_row = row_tile * 128 + ((within >> 2) & 3) * 32 + (within >> 4);
  const int64_t old_row = (new_row & ~31LL) | trtllm_src_row(static_cast<int>(new_row & 31));
  out[o] = in[(expert * rows + old_row) * cols + col];
}

// -----------------------------------------------------------------------------
// Shared Marlin launch helpers
// -----------------------------------------------------------------------------

Status check_marlin_dims(int64_t size_k, int64_t size_n, int32_t num_bits, const char* op) {
  if (num_bits != 4 && num_bits != 8) {
    return detail::set_error(Status::kUnsupportedShape, "%s: num_bits %d not in {4, 8}", op, num_bits);
  }
  if (size_k <= 0 || size_n <= 0 || size_k % kMarlinTileK != 0 || size_n % kMarlinTileN != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "%s: K = %lld must be a multiple of 16 and N = %lld of 64", op,
                             static_cast<long long>(size_k), static_cast<long long>(size_n));
  }
  return Status::kSuccess;
}

Status check_group_size(int32_t group_size, int64_t size_k, const char* op) {
  if (group_size != -1 && group_size != 32 && group_size != 64 && group_size != 128) {
    return detail::set_error(Status::kUnsupportedShape, "%s: group_size %d not in {-1, 32, 64, 128}",
                             op, group_size);
  }
  if (group_size > 0 && size_k % group_size != 0) {
    return detail::set_error(Status::kUnsupportedShape, "%s: K = %lld is not a multiple of group_size %d",
                             op, static_cast<long long>(size_k), group_size);
  }
  return Status::kSuccess;
}

/// Number of scale rows: K/group for grouped scales, 1 for per-channel (group -1 or == K).
int64_t scale_rows(int32_t group_size, int64_t size_k) {
  return (group_size < 0 || group_size >= size_k) ? 1 : size_k / group_size;
}

template <int kBits, bool kKFastest, typename Source>
Status launch_marlin_repack(const Tensor& out, const Source& src, int64_t size_k, int64_t size_n,
                            const LaunchContext& ctx, const RepackEpilogue& epi = RepackEpilogue{}) {
  const int k_tiles = static_cast<int>(size_k / kMarlinTileK);
  const int n_tiles = static_cast<int>(size_n / kMarlinTileN);
  const int64_t total = static_cast<int64_t>(k_tiles) * n_tiles * 32;  // one thread per (tile, lane)
  if (ceil_div(total, 256) > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "marlin repack: matrix too large");
  }
  if (!aligned(data_of(out), 16)) {
    return detail::set_error(Status::kMisaligned, "marlin repack: output needs 16-byte alignment");
  }
  marlin_repack_tiles_kernel<kBits, kKFastest, Source>
      <<<static_cast<unsigned>(ceil_div(total, 256)), 256, 0, stream_of(ctx)>>>(
          mut_ptr<uint32_t>(out), src, k_tiles, n_tiles, total, epi);
  return last_launch("marlin_repack");
}

/// Scale permutation folded into a repack launch, or an empty epilogue the caller must follow with
/// `launch_permute_scales`.  `fused` reports which one happened.
RepackEpilogue make_repack_epilogue(const Tensor& out, const Tensor& in, bool per_channel, bool* fuse) {
  const int bytes = byte_size(in.dtype);
  *fuse = (bytes == 1 || bytes == 2 || bytes == 4) && tensor_present(out) && tensor_present(in);
  if (!*fuse) return RepackEpilogue{};
  RepackEpilogue epi;
  epi.scales_out = static_cast<char*>(out.data) + out.byte_offset;
  epi.scales_in = data_of(in);
  epi.scale_total = tensor_numel(in);
  epi.scale_bytes = bytes;
  epi.per_channel = per_channel;
  return epi;
}

/// 4-bit GPTQ without act-order: one warp stages each tile through shared memory.  Requires two
/// whole source rows per 16-byte lane load, so it falls back to the generic kernel when the source
/// columns are not a multiple of 4 words or the buffer is not 16-byte aligned.  `fused` reports
/// whether the launch absorbed the epilogue.
Status launch_marlin_repack_gptq4(const Tensor& out, const GptqSource<4>& src, int64_t size_k,
                                  int64_t size_n, const RepackEpilogue& epi, const LaunchContext& ctx,
                                  bool* fused) {
  *fused = false;
  if (src.perm != nullptr || src.size_n % 4 != 0 || !aligned(src.qweight, 16)) {
    return launch_marlin_repack<4, false>(out, src, size_k, size_n, ctx);
  }
  if (!aligned(data_of(out), 16)) {
    return detail::set_error(Status::kMisaligned, "marlin repack: output needs 16-byte alignment");
  }
  constexpr int kWarps = 4;  // one tile per warp
  const int n_tiles = static_cast<int>(size_n / kMarlinTileN);
  const int64_t total_tiles = static_cast<int64_t>(size_k / kMarlinTileK) * n_tiles;
  if (ceil_div(total_tiles, kWarps) > INT32_MAX) {
    return detail::set_error(Status::kUnsupportedShape, "marlin repack: matrix too large");
  }
  marlin_repack_gptq4_tiles_kernel<kWarps>
      <<<static_cast<unsigned>(ceil_div(total_tiles, kWarps)), 32 * kWarps, 0, stream_of(ctx)>>>(
          mut_ptr<uint32_t>(out), src, n_tiles, total_tiles, epi);
  Status s = last_launch("marlin_repack");
  if (s != Status::kSuccess) return s;
  *fused = !epi.empty();
  return Status::kSuccess;
}

Status launch_permute_scales(const Tensor& out, const Tensor& in, bool per_channel,
                             const LaunchContext& ctx) {
  const int64_t total = tensor_numel(in);
  const unsigned grid = static_cast<unsigned>(ceil_div(total, 256));
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_PERMUTE(T)                                                                            \
  do {                                                                                             \
    if (per_channel)                                                                               \
      marlin_permute_scales_kernel<T, true><<<grid, 256, 0, stream>>>(mut_ptr<T>(out), const_ptr<T>(in), total); \
    else                                                                                           \
      marlin_permute_scales_kernel<T, false><<<grid, 256, 0, stream>>>(mut_ptr<T>(out), const_ptr<T>(in), total); \
  } while (0)
  switch (byte_size(in.dtype)) {
    case 1: KOPS_PERMUTE(uint8_t); break;
    case 2: KOPS_PERMUTE(uint16_t); break;
    case 4: KOPS_PERMUTE(uint32_t); break;
    default:
      return detail::set_error(Status::kUnsupportedDtype, "marlin scales: dtype %s", dtype_name(in.dtype));
  }
#undef KOPS_PERMUTE
  return last_launch("marlin_permute_scales");
}

}  // namespace

// =============================================================================
// Host launchers
// =============================================================================

Status awq_to_marlin(const Tensor& out_weight, const Tensor& out_scales, const Tensor& out_zeros,
                     const Tensor& qweight, const Tensor& scales, const Tensor& qzeros,
                     int32_t num_bits, int32_t group_size, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (num_bits != 4 && num_bits != 8) {
    return detail::set_error(Status::kUnsupportedShape, "awq_to_marlin: num_bits %d not in {4, 8}", num_bits);
  }
  const int pack = 32 / num_bits;
  s = check_dense(qweight, DType::kInt32, 2, 4, "awq_to_marlin.qweight");
  if (s != Status::kSuccess) return s;
  const int64_t size_k = qweight.shape[0];
  const int64_t size_n = qweight.shape[1] * pack;
  s = check_marlin_dims(size_k, size_n, num_bits, "awq_to_marlin");
  if (s != Status::kSuccess) return s;
  s = check_group_size(group_size, size_k, "awq_to_marlin");
  if (s != Status::kSuccess) return s;
  const int64_t groups = scale_rows(group_size, size_k);
  if (!is_float_scalar(scales.dtype) || scales.dtype == DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "awq_to_marlin: scales must be fp16 or bf16");
  }
  s = check_dense(scales, scales.dtype, 2, 2, "awq_to_marlin.scales");
  if (s != Status::kSuccess) return s;
  s = shape_is(scales, {groups, size_n}, "awq_to_marlin.scales");
  if (s != Status::kSuccess) return s;
  s = check_dense(qzeros, DType::kInt32, 2, 4, "awq_to_marlin.qzeros");
  if (s != Status::kSuccess) return s;
  s = shape_is(qzeros, {groups, size_n / pack}, "awq_to_marlin.qzeros");
  if (s != Status::kSuccess) return s;
  s = check_dense(out_weight, DType::kInt32, 2, 4, "awq_to_marlin.out_weight");
  if (s != Status::kSuccess) return s;
  s = shape_is(out_weight, {size_k / kMarlinTileK, size_n * kMarlinTileK / pack}, "awq_to_marlin.out_weight");
  if (s != Status::kSuccess) return s;
  s = check_dense(out_scales, scales.dtype, 2, 2, "awq_to_marlin.out_scales");
  if (s != Status::kSuccess) return s;
  s = shape_is(out_scales, {groups, size_n}, "awq_to_marlin.out_scales");
  if (s != Status::kSuccess) return s;
  s = check_dense(out_zeros, DType::kInt32, 2, 4, "awq_to_marlin.out_zeros");
  if (s != Status::kSuccess) return s;
  s = shape_is(out_zeros, {groups, size_n / pack}, "awq_to_marlin.out_zeros");
  if (s != Status::kSuccess) return s;

  const bool per_channel = group_size < 0 || group_size >= size_k;
  bool fuse = false;
  const RepackEpilogue epi = make_repack_epilogue(out_scales, scales, per_channel, &fuse);
  if (num_bits == 4) {
    s = launch_marlin_repack<4, false>(out_weight, AwqSource<4>{const_ptr<uint32_t>(qweight), size_n / pack},
                                       size_k, size_n, ctx, fuse ? epi : RepackEpilogue{});
  } else {
    s = launch_marlin_repack<8, false>(out_weight, AwqSource<8>{const_ptr<uint32_t>(qweight), size_n / pack},
                                       size_k, size_n, ctx, fuse ? epi : RepackEpilogue{});
  }
  if (s != Status::kSuccess) return s;
  if (!fuse) {
    s = launch_permute_scales(out_scales, scales, per_channel, ctx);
    if (s != Status::kSuccess) return s;
  }
  const int64_t words = groups * (size_n / pack);
  const unsigned grid = static_cast<unsigned>(ceil_div(words, 256));
  if (num_bits == 4) {
    marlin_awq_zero_points_words_kernel<4><<<grid, 256, 0, stream_of(ctx)>>>(
        mut_ptr<uint32_t>(out_zeros), const_ptr<uint32_t>(qzeros), words, static_cast<int>(size_n / pack));
  } else {
    marlin_awq_zero_points_words_kernel<8><<<grid, 256, 0, stream_of(ctx)>>>(
        mut_ptr<uint32_t>(out_zeros), const_ptr<uint32_t>(qzeros), words, static_cast<int>(size_n / pack));
  }
  return last_launch("awq_to_marlin");
}

Status gptq_to_marlin(const Tensor& out_weight, const Tensor& out_scales, const Tensor& perm_out,
                      const Tensor& qweight, const Tensor& scales, const Tensor& g_idx,
                      int32_t num_bits, int32_t group_size, bool is_sym, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (num_bits != 4 && num_bits != 8) {
    return detail::set_error(Status::kUnsupportedShape, "gptq_to_marlin: num_bits %d not in {4, 8}", num_bits);
  }
  if (!is_sym) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "gptq_to_marlin: asymmetric GPTQ (zero points) is not supported; use sym checkpoints");
  }
  const int pack = 32 / num_bits;
  s = check_dense(qweight, DType::kInt32, 2, 4, "gptq_to_marlin.qweight");
  if (s != Status::kSuccess) return s;
  const int64_t size_k = qweight.shape[0] * pack;
  const int64_t size_n = qweight.shape[1];
  s = check_marlin_dims(size_k, size_n, num_bits, "gptq_to_marlin");
  if (s != Status::kSuccess) return s;
  s = check_group_size(group_size, size_k, "gptq_to_marlin");
  if (s != Status::kSuccess) return s;
  const int64_t groups = scale_rows(group_size, size_k);
  if (!is_float_scalar(scales.dtype) || scales.dtype == DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "gptq_to_marlin: scales must be fp16 or bf16");
  }
  s = check_dense(scales, scales.dtype, 2, 2, "gptq_to_marlin.scales");
  if (s != Status::kSuccess) return s;
  s = shape_is(scales, {groups, size_n}, "gptq_to_marlin.scales");
  if (s != Status::kSuccess) return s;
  s = check_dense(out_weight, DType::kInt32, 2, 4, "gptq_to_marlin.out_weight");
  if (s != Status::kSuccess) return s;
  s = shape_is(out_weight, {size_k / kMarlinTileK, size_n * kMarlinTileK / pack}, "gptq_to_marlin.out_weight");
  if (s != Status::kSuccess) return s;
  s = check_dense(out_scales, scales.dtype, 2, 2, "gptq_to_marlin.out_scales");
  if (s != Status::kSuccess) return s;
  s = shape_is(out_scales, {groups, size_n}, "gptq_to_marlin.out_scales");
  if (s != Status::kSuccess) return s;
  const bool has_perm = tensor_present(perm_out);
  if (has_perm) {
    s = check_dense(perm_out, DType::kInt32, 1, 4, "gptq_to_marlin.perm_out");
    if (s != Status::kSuccess) return s;
    s = shape_is(perm_out, {size_k}, "gptq_to_marlin.perm_out");
    if (s != Status::kSuccess) return s;
  }
  const bool act_order = tensor_present(g_idx);
  if (act_order) {
    s = check_dense(g_idx, DType::kInt32, 1, 4, "gptq_to_marlin.g_idx");
    if (s != Status::kSuccess) return s;
    s = shape_is(g_idx, {size_k}, "gptq_to_marlin.g_idx");
    if (s != Status::kSuccess) return s;
    if (!has_perm) {
      return detail::set_error(Status::kInvalidArgument, "gptq_to_marlin: act-order needs perm_out for the activation permutation");
    }
    if (size_k > INT32_MAX) {
      return detail::set_error(Status::kUnsupportedShape, "gptq_to_marlin: K too large");
    }
    gptq_stable_argsort_kernel<<<static_cast<unsigned>(ceil_div(size_k, 32)), 256, 0, stream_of(ctx)>>>(
        mut_ptr<int32_t>(perm_out), const_ptr<int32_t>(g_idx), static_cast<int>(size_k));
    s = last_launch("gptq_to_marlin.argsort");
    if (s != Status::kSuccess) return s;
  }
  const int32_t* perm = act_order ? const_ptr<int32_t>(perm_out) : nullptr;
  // Without act-order the activation permutation is the identity: 4*K bytes of stores that the
  // 4-bit staged launch below absorbs instead of paying a launch of their own.
  const bool perm_pending = has_perm && !act_order;
  // With act-order the rows of one tile come from 16 unrelated qweight rows, so each qweight word is
  // needed by up to 8 tiles; walking one 64-column strip (128 KB of qweight) at a time keeps it in L2.
  const GptqSource<4> src4{const_ptr<uint32_t>(qweight), perm, size_n};
  const GptqSource<8> src8{const_ptr<uint32_t>(qweight), perm, size_n};
  const bool per_channel = group_size < 0 || group_size >= size_k;
  bool fuse_scales = false;
  bool staged = false;
  if (num_bits == 4 && !act_order) {
    RepackEpilogue epi = make_repack_epilogue(out_scales, scales, per_channel, &fuse_scales);
    if (!fuse_scales) epi.scale_total = 0;  // odd scale dtype: standalone kernel reports it
    if (perm_pending) {
      epi.perm_out = mut_ptr<int32_t>(perm_out);
      epi.perm_size = size_k;
    }
    s = launch_marlin_repack_gptq4(out_weight, src4, size_k, size_n, epi, ctx, &staged);
  } else if (num_bits == 4) {
    s = launch_marlin_repack<4, true>(out_weight, src4, size_k, size_n, ctx);
  } else if (act_order) {
    s = launch_marlin_repack<8, true>(out_weight, src8, size_k, size_n, ctx);
  } else {
    s = launch_marlin_repack<8, false>(out_weight, src8, size_k, size_n, ctx);
  }
  if (s != Status::kSuccess) return s;
  if (perm_pending && !staged) {
    identity_perm_kernel<<<static_cast<unsigned>(ceil_div(size_k, 256)), 256, 0, stream_of(ctx)>>>(
        mut_ptr<int32_t>(perm_out), static_cast<int>(size_k));
    s = last_launch("gptq_to_marlin.identity");
    if (s != Status::kSuccess) return s;
  }
  if (staged && fuse_scales) return Status::kSuccess;
  return launch_permute_scales(out_scales, scales, per_channel, ctx);
}

Status awq_dequantize(const Tensor& out, const Tensor& qweight, const Tensor& scales,
                      const Tensor& qzeros, int32_t group_size, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_dense(qweight, DType::kInt32, 2, 4, "awq_dequantize.qweight");
  if (s != Status::kSuccess) return s;
  const int64_t size_k = qweight.shape[0];
  const int64_t size_n = qweight.shape[1] * 8;
  if (!is_float_scalar(scales.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "awq_dequantize: scales dtype %s", dtype_name(scales.dtype));
  }
  if (group_size != -1 && (group_size <= 0 || size_k % group_size != 0)) {
    return detail::set_error(Status::kUnsupportedShape, "awq_dequantize: group_size %d does not divide K = %lld",
                             group_size, static_cast<long long>(size_k));
  }
  const int64_t groups = group_size > 0 ? size_k / group_size : 1;
  s = check_dense(scales, scales.dtype, 2, 2, "awq_dequantize.scales");
  if (s != Status::kSuccess) return s;
  s = shape_is(scales, {groups, size_n}, "awq_dequantize.scales");
  if (s != Status::kSuccess) return s;
  s = check_dense(qzeros, DType::kInt32, 2, 4, "awq_dequantize.qzeros");
  if (s != Status::kSuccess) return s;
  s = shape_is(qzeros, {groups, size_n / 8}, "awq_dequantize.qzeros");
  if (s != Status::kSuccess) return s;
  s = check_tensor(out, scales.dtype, 2, 2, "awq_dequantize.out");
  if (s != Status::kSuccess) return s;
  s = shape_is(out, {size_k, size_n}, "awq_dequantize.out");
  if (s != Status::kSuccess) return s;
  if (out.stride[1] != 1 || out.stride[0] < size_n) {
    return detail::set_error(Status::kUnsupportedLayout, "awq_dequantize.out: rows must be contiguous");
  }
  const int64_t words = size_k * (size_n / 8);
  const unsigned grid = static_cast<unsigned>(ceil_div(words, 256));
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_AWQ_DQ(T)                                                                             \
  awq_dequantize_words_kernel<T><<<grid, 256, 0, stream>>>(                                        \
      mut_ptr<T>(out), out.stride[0], const_ptr<uint32_t>(qweight), const_ptr<T>(scales),          \
      const_ptr<uint32_t>(qzeros), static_cast<int>(size_k), static_cast<int>(size_n), group_size)
  if (scales.dtype == DType::kFloat16) KOPS_AWQ_DQ(__half);
  else if (scales.dtype == DType::kBFloat16) KOPS_AWQ_DQ(__nv_bfloat16);
  else KOPS_AWQ_DQ(float);
#undef KOPS_AWQ_DQ
  return last_launch("awq_dequantize");
}

namespace {

struct GgufShape {
  int block_elems;
  int block_bytes;
};

bool gguf_shape(GGUFType t, GgufShape* out) {
  switch (t) {
    case GGUFType::kQ4_0: *out = {32, 18}; return true;
    case GGUFType::kQ4_1: *out = {32, 20}; return true;
    case GGUFType::kQ5_0: *out = {32, 22}; return true;
    case GGUFType::kQ5_1: *out = {32, 24}; return true;
    case GGUFType::kQ8_0: *out = {32, 34}; return true;
    case GGUFType::kIQ4_NL: *out = {32, 18}; return true;
    case GGUFType::kQ2_K: *out = {256, 84}; return true;
    case GGUFType::kQ3_K: *out = {256, 110}; return true;
    case GGUFType::kQ4_K: *out = {256, 144}; return true;
    case GGUFType::kQ5_K: *out = {256, 176}; return true;
    case GGUFType::kQ6_K: *out = {256, 210}; return true;
    case GGUFType::kQ8_K: *out = {256, 292}; return true;
    case GGUFType::kIQ4_XS: *out = {256, 136}; return true;
    case GGUFType::kIQ2_XXS: *out = {256, 66}; return true;
    case GGUFType::kIQ2_XS: *out = {256, 74}; return true;
    case GGUFType::kIQ2_S: *out = {256, 82}; return true;
    case GGUFType::kIQ3_XXS: *out = {256, 98}; return true;
    case GGUFType::kIQ3_S: *out = {256, 110}; return true;
    case GGUFType::kIQ1_S: *out = {256, 50}; return true;
    case GGUFType::kIQ1_M: *out = {256, 56}; return true;
    default: return false;
  }
}

template <typename out_t>
Status dispatch_gguf(const Tensor& out, const Tensor& qweight, GGUFType type, int64_t total,
                     const LaunchContext& ctx) {
  const unsigned grid = static_cast<unsigned>(ceil_div(total, 256));
  const cudaStream_t stream = stream_of(ctx);
#define KOPS_GGUF(TYPE, E, B)                                                                      \
  case GGUFType::TYPE:                                                                             \
    gguf_dequantize_blocks_kernel<out_t, GGUFType::TYPE, E, B><<<grid, 256, 0, stream>>>(          \
        mut_ptr<out_t>(out), const_ptr<uint8_t>(qweight), total);                                  \
    break
  switch (type) {
    KOPS_GGUF(kQ4_0, 32, 18);
    KOPS_GGUF(kQ4_1, 32, 20);
    KOPS_GGUF(kQ5_0, 32, 22);
    KOPS_GGUF(kQ5_1, 32, 24);
    KOPS_GGUF(kQ8_0, 32, 34);
    KOPS_GGUF(kIQ4_NL, 32, 18);
    KOPS_GGUF(kQ2_K, 256, 84);
    KOPS_GGUF(kQ3_K, 256, 110);
    KOPS_GGUF(kQ4_K, 256, 144);
    KOPS_GGUF(kQ5_K, 256, 176);
    KOPS_GGUF(kQ6_K, 256, 210);
    KOPS_GGUF(kQ8_K, 256, 292);
    KOPS_GGUF(kIQ4_XS, 256, 136);
    KOPS_GGUF(kIQ2_XXS, 256, 66);
    KOPS_GGUF(kIQ2_XS, 256, 74);
    KOPS_GGUF(kIQ2_S, 256, 82);
    KOPS_GGUF(kIQ3_XXS, 256, 98);
    KOPS_GGUF(kIQ3_S, 256, 110);
    KOPS_GGUF(kIQ1_S, 256, 50);
    KOPS_GGUF(kIQ1_M, 256, 56);
    default:
      return detail::set_error(Status::kInternal, "gguf_dequantize: unreachable type");
  }
#undef KOPS_GGUF
  return last_launch("gguf_dequantize");
}

}  // namespace

Status gguf_dequantize(const Tensor& out, const Tensor& qweight, GGUFType type,
                       const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  GgufShape shape{};
  if (!gguf_shape(type, &shape)) {
    return detail::set_error(Status::kNotImplemented,
                             "gguf_dequantize: GGUF type %d is not a quantized block format (F32/F16/BF16 need no "
                             "dequantization, Q8_1 is an activation format)",
                             static_cast<int>(type));
  }
  if (!is_float_scalar(out.dtype)) {
    return detail::set_error(Status::kUnsupportedDtype, "gguf_dequantize: out dtype %s", dtype_name(out.dtype));
  }
  s = check_dense(out, out.dtype, 2, 2, "gguf_dequantize.out");
  if (s != Status::kSuccess) return s;
  const int64_t size_n = out.shape[0];
  const int64_t size_k = out.shape[1];
  if (size_k % shape.block_elems != 0) {
    return detail::set_error(Status::kUnsupportedShape, "gguf_dequantize: K = %lld is not a multiple of the block size %d",
                             static_cast<long long>(size_k), shape.block_elems);
  }
  s = check_dense(qweight, DType::kUInt8, -1, 1, "gguf_dequantize.qweight");
  if (s != Status::kSuccess) return s;
  const int64_t want_bytes = size_n * (size_k / shape.block_elems) * shape.block_bytes;
  if (tensor_numel(qweight) != want_bytes) {
    return detail::set_error(Status::kInvalidArgument, "gguf_dequantize: qweight has %lld bytes, [%lld, %lld] needs %lld",
                             static_cast<long long>(tensor_numel(qweight)), static_cast<long long>(size_n),
                             static_cast<long long>(size_k), static_cast<long long>(want_bytes));
  }
  const int64_t total = size_n * size_k;
  if (total == 0) return Status::kSuccess;
  if (out.dtype == DType::kFloat32) return dispatch_gguf<float>(out, qweight, type, total, ctx);
  if (out.dtype == DType::kFloat16) return dispatch_gguf<__half>(out, qweight, type, total, ctx);
  return dispatch_gguf<__nv_bfloat16>(out, qweight, type, total, ctx);
}

Status fp4_to_marlin(const Tensor& out_weight, const Tensor& out_scales, const Tensor& weight,
                     const Tensor& scales, int32_t block_size, bool scale_is_ue8m0,
                     const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (block_size != (scale_is_ue8m0 ? 32 : 16)) {
    return detail::set_error(Status::kUnsupportedShape,
                             "fp4_to_marlin: %s uses block %d, got %d", scale_is_ue8m0 ? "MXFP4 (UE8M0)" : "NVFP4 (E4M3)",
                             scale_is_ue8m0 ? 32 : 16, block_size);
  }
  const bool packed_dtype = weight.dtype == DType::kFloat4E2M1x2;
  if (!packed_dtype && weight.dtype != DType::kUInt8) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_to_marlin: weight dtype %s", dtype_name(weight.dtype));
  }
  s = check_dense(weight, weight.dtype, 2, 1, "fp4_to_marlin.weight");
  if (s != Status::kSuccess) return s;
  const int64_t size_n = weight.shape[0];
  const int64_t size_k = packed_dtype ? weight.shape[1] : weight.shape[1] * 2;
  s = check_marlin_dims(size_k, size_n, 4, "fp4_to_marlin");
  if (s != Status::kSuccess) return s;
  if (size_k % block_size != 0) {
    return detail::set_error(Status::kUnsupportedShape, "fp4_to_marlin: K = %lld is not a multiple of block %d",
                             static_cast<long long>(size_k), block_size);
  }
  const int64_t groups = size_k / block_size;
  s = check_dense(scales, DType::kInvalid, 2, 1, "fp4_to_marlin.scales");
  if (s != Status::kSuccess) return s;
  if (byte_size(scales.dtype) != 1) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_to_marlin: scales dtype %s must be 1 byte", dtype_name(scales.dtype));
  }
  s = shape_is(scales, {size_n, groups}, "fp4_to_marlin.scales");
  if (s != Status::kSuccess) return s;
  s = check_dense(out_weight, DType::kInt32, 2, 4, "fp4_to_marlin.out_weight");
  if (s != Status::kSuccess) return s;
  s = shape_is(out_weight, {size_k / kMarlinTileK, size_n * kMarlinTileK / 8}, "fp4_to_marlin.out_weight");
  if (s != Status::kSuccess) return s;
  s = check_dense(out_scales, scales.dtype, 2, 1, "fp4_to_marlin.out_scales");
  if (s != Status::kSuccess) return s;
  s = shape_is(out_scales, {groups, size_n}, "fp4_to_marlin.out_scales");
  if (s != Status::kSuccess) return s;

  s = launch_marlin_repack<4, true>(out_weight, Fp4Source{const_ptr<uint8_t>(weight), size_k / 2},
                                    size_k, size_n, ctx);
  if (s != Status::kSuccess) return s;
  const int64_t total = size_n * groups;
  fp4_marlin_scales_kernel<<<static_cast<unsigned>(ceil_div(total, 256)), 256, 0, stream_of(ctx)>>>(
      mut_ptr<uint8_t>(out_scales), const_ptr<uint8_t>(scales), static_cast<int>(size_n), static_cast<int>(groups));
  return last_launch("fp4_to_marlin");
}

Status fp4_backend_shuffle(const Tensor& out_weight, const Tensor& out_scales, const Tensor& weight,
                           const Tensor& scales, int32_t block_size, WeightPacking target,
                           const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  if (target != WeightPacking::kFP4E2M1 && target != WeightPacking::kTrtllmShuffled) {
    return detail::set_error(Status::kNotImplemented,
                             "fp4_backend_shuffle: target packing %d is not a backend layout (kFP4E2M1: CUTLASS "
                             "SM100/SM120 interleaved scales, kTrtllmShuffled: TRT-LLM-gen)",
                             static_cast<int>(target));
  }
  if (block_size != 16 && block_size != 32) {
    return detail::set_error(Status::kUnsupportedShape, "fp4_backend_shuffle: block %d not in {16, 32}", block_size);
  }
  const bool packed_dtype = weight.dtype == DType::kFloat4E2M1x2;
  if (!packed_dtype && weight.dtype != DType::kUInt8) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_backend_shuffle: weight dtype %s", dtype_name(weight.dtype));
  }
  s = check_dense(weight, weight.dtype, 3, 1, "fp4_backend_shuffle.weight");
  if (s != Status::kSuccess) return s;
  s = check_dense(out_weight, weight.dtype, 3, 1, "fp4_backend_shuffle.out_weight");
  if (s != Status::kSuccess) return s;
  for (int d = 0; d < 3; ++d) {
    if (out_weight.shape[d] != weight.shape[d]) {
      return detail::set_error(Status::kInvalidArgument, "fp4_backend_shuffle: out_weight shape != weight shape");
    }
  }
  const int64_t size_k = packed_dtype ? weight.shape[2] : weight.shape[2] * 2;
  if (size_k % block_size != 0) {
    return detail::set_error(Status::kUnsupportedShape, "fp4_backend_shuffle: K = %lld is not a multiple of block %d",
                             static_cast<long long>(size_k), block_size);
  }
  s = check_dense(scales, DType::kInvalid, 3, 1, "fp4_backend_shuffle.scales");
  if (s != Status::kSuccess) return s;
  if (byte_size(scales.dtype) != 1) {
    return detail::set_error(Status::kUnsupportedDtype, "fp4_backend_shuffle: scales dtype %s must be 1 byte", dtype_name(scales.dtype));
  }
  s = shape_is(scales, {weight.shape[0], weight.shape[1], size_k / block_size}, "fp4_backend_shuffle.scales");
  if (s != Status::kSuccess) return s;
  const int64_t weight_bytes = packed_dtype ? tensor_numel(weight) / 2 : tensor_numel(weight);
  if (target == WeightPacking::kTrtllmShuffled) {
    // Layout of TensorRT-LLM's trtllm-gen MoE: weight rows and scale rows go through shuffle_matrix_a (blocks of
    // 32 rows), scales then through block_scale_interleave.  The fused gate/up weight additionally needs
    // reorder_rows_for_gated_act_gemm first, i.e. gemm::glu_weight_interleave(halves -> interleave 1).
    const int64_t experts = weight.shape[0];
    const int64_t rows = weight.shape[1];
    const int64_t groups = size_k / block_size;
    if (rows % 128 != 0 || groups % 4 != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "fp4_backend_shuffle: the TRT-LLM-gen layout needs N %% 128 == 0 and K / block %% 4 == 0 "
                               "(N = %lld, K / block = %lld)",
                               static_cast<long long>(rows), static_cast<long long>(groups));
    }
    s = check_dense(out_scales, scales.dtype, 3, 1, "fp4_backend_shuffle.out_scales");
    if (s != Status::kSuccess) return s;
    s = shape_is(out_scales, {experts, rows, groups}, "fp4_backend_shuffle.out_scales");
    if (s != Status::kSuccess) return s;
    if (data_of(out_weight) == data_of(weight) || data_of(out_scales) == data_of(scales)) {
      return detail::set_error(Status::kInvalidArgument, "fp4_backend_shuffle: the TRT-LLM-gen shuffle is out of place");
    }
    if (weight_bytes == 0) return Status::kSuccess;
    if (!aligned(data_of(weight), 16) || !aligned(data_of(out_weight), 16)) {
      return detail::set_error(Status::kMisaligned, "fp4_backend_shuffle: weight buffers need 16-byte alignment");
    }
    const int64_t total_words = weight_bytes / 16;
    trtllm_shuffle_rows_kernel<<<static_cast<unsigned>(ceil_div(total_words, 256)), 256, 0, stream_of(ctx)>>>(
        reinterpret_cast<uint4*>(mut_ptr<uint8_t>(out_weight)), reinterpret_cast<const uint4*>(const_ptr<uint8_t>(weight)),
        static_cast<int>(size_k / 2 / 16), total_words);
    s = last_launch("fp4_backend_shuffle.weight");
    if (s != Status::kSuccess) return s;
    const int64_t total_scales = experts * rows * groups;
    trtllm_shuffle_scales_kernel<<<static_cast<unsigned>(ceil_div(total_scales, 256)), 256, 0, stream_of(ctx)>>>(
        mut_ptr<uint8_t>(out_scales), const_ptr<uint8_t>(scales), static_cast<int>(rows), static_cast<int>(groups),
        total_scales);
    return last_launch("fp4_backend_shuffle.scales");
  }
  if (weight_bytes > 0) {
    if (!aligned(data_of(weight), 16) || !aligned(data_of(out_weight), 16)) {
      return detail::set_error(Status::kMisaligned, "fp4_backend_shuffle: weight buffers need 16-byte alignment");
    }
    copy_bytes_kernel<<<static_cast<unsigned>(std::min<int64_t>(ceil_div(weight_bytes, 16 * 256), 4096)), 256, 0,
                        stream_of(ctx)>>>(mut_ptr<uint8_t>(out_weight), const_ptr<uint8_t>(weight), weight_bytes);
    s = last_launch("fp4_backend_shuffle.weight");
    if (s != Status::kSuccess) return s;
  }
  return gemm::block_scale_swizzle(out_scales, scales, false, ctx);
}

Status fp8_block_scale_to_ue8m0(const Tensor& weight_inout, const Tensor& scales_inout,
                                const Tensor& ue8m0_out, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_tensor(weight_inout, DType::kFloat8E4M3, 2, 1, "fp8_block_scale_to_ue8m0.weight");
  if (s != Status::kSuccess) return s;
  if (weight_inout.stride[1] != 1 || weight_inout.stride[0] < weight_inout.shape[1]) {
    return detail::set_error(Status::kUnsupportedLayout, "fp8_block_scale_to_ue8m0: weight rows must be contiguous");
  }
  const int64_t size_n = weight_inout.shape[0];
  const int64_t size_k = weight_inout.shape[1];
  const int64_t sn = ceil_div(size_n, 128);
  const int64_t sk = ceil_div(size_k, 128);
  s = check_tensor(scales_inout, DType::kFloat32, 2, 4, "fp8_block_scale_to_ue8m0.scales");
  if (s != Status::kSuccess) return s;
  s = shape_is(scales_inout, {sn, sk}, "fp8_block_scale_to_ue8m0.scales");
  if (s != Status::kSuccess) return s;
  if (scales_inout.stride[1] != 1) {
    return detail::set_error(Status::kUnsupportedLayout, "fp8_block_scale_to_ue8m0: scale rows must be contiguous");
  }
  uint8_t* exps = nullptr;
  if (tensor_present(ue8m0_out)) {
    if (ue8m0_out.dtype != DType::kUInt8 && ue8m0_out.dtype != DType::kFloat8E8M0) {
      return detail::set_error(Status::kUnsupportedDtype, "fp8_block_scale_to_ue8m0: ue8m0_out dtype %s", dtype_name(ue8m0_out.dtype));
    }
    s = check_tensor(ue8m0_out, ue8m0_out.dtype, 2, 1, "fp8_block_scale_to_ue8m0.ue8m0_out");
    if (s != Status::kSuccess) return s;
    s = shape_is(ue8m0_out, {sn, sk}, "fp8_block_scale_to_ue8m0.ue8m0_out");
    if (s != Status::kSuccess) return s;
    if (ue8m0_out.stride[0] != scales_inout.stride[0] || ue8m0_out.stride[1] != 1) {
      return detail::set_error(Status::kUnsupportedLayout, "fp8_block_scale_to_ue8m0: ue8m0_out must share the scale row stride");
    }
    exps = mut_ptr<uint8_t>(ue8m0_out);
  }
  if (size_n == 0 || size_k == 0) return Status::kSuccess;
  const dim3 grid(static_cast<unsigned>(sk), static_cast<unsigned>(sn));
  fp8_block_scale_to_ue8m0_blocks_kernel<<<grid, 256, 0, stream_of(ctx)>>>(
      mut_ptr<uint8_t>(weight_inout), weight_inout.stride[0], mut_ptr<float>(scales_inout),
      scales_inout.stride[0], exps, static_cast<int>(size_n), static_cast<int>(size_k));
  return last_launch("fp8_block_scale_to_ue8m0");
}

Status smoothquant_compute_scales(const Tensor& smooth, const Tensor& act_absmax,
                                  const Tensor& weight_absmax, float alpha, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  s = check_dense(smooth, DType::kFloat32, 1, 4, "smoothquant_compute_scales.smooth");
  if (s != Status::kSuccess) return s;
  const int64_t size_k = smooth.shape[0];
  s = check_dense(act_absmax, DType::kFloat32, 1, 4, "smoothquant_compute_scales.act_absmax");
  if (s != Status::kSuccess) return s;
  s = check_dense(weight_absmax, DType::kFloat32, 1, 4, "smoothquant_compute_scales.weight_absmax");
  if (s != Status::kSuccess) return s;
  if (act_absmax.shape[0] != size_k || weight_absmax.shape[0] != size_k) {
    return detail::set_error(Status::kInvalidArgument, "smoothquant_compute_scales: statistics must have K = %lld entries",
                             static_cast<long long>(size_k));
  }
  if (!(alpha >= 0.0f && alpha <= 1.0f)) {
    return detail::set_error(Status::kInvalidArgument, "smoothquant_compute_scales: alpha %g outside [0, 1]", static_cast<double>(alpha));
  }
  if (size_k == 0) return Status::kSuccess;
  smoothquant_compute_scales_elems_kernel<<<static_cast<unsigned>(ceil_div(size_k, 256)), 256, 0, stream_of(ctx)>>>(
      mut_ptr<float>(smooth), const_ptr<float>(act_absmax), const_ptr<float>(weight_absmax), alpha,
      static_cast<int>(size_k));
  return last_launch("smoothquant_compute_scales");
}

Status smoothquant_fold(const Tensor& norm_weight, const Tensor& norm_bias, const Tensor& linear_weight,
                        const Tensor& smooth, const LaunchContext& ctx) {
  Status s = backend_ok(ctx);
  if (s != Status::kSuccess) return s;
  const DType dt = linear_weight.dtype;
  if (!is_float_scalar(dt)) {
    return detail::set_error(Status::kUnsupportedDtype, "smoothquant_fold: linear dtype %s", dtype_name(dt));
  }
  s = check_tensor(linear_weight, dt, 2, 1, "smoothquant_fold.linear_weight");
  if (s != Status::kSuccess) return s;
  if (linear_weight.stride[1] != 1 || linear_weight.stride[0] < linear_weight.shape[1]) {
    return detail::set_error(Status::kUnsupportedLayout, "smoothquant_fold: linear rows must be contiguous");
  }
  const int64_t size_n = linear_weight.shape[0];
  const int64_t size_k = linear_weight.shape[1];
  s = check_dense(smooth, DType::kFloat32, 1, 4, "smoothquant_fold.smooth");
  if (s != Status::kSuccess) return s;
  s = shape_is(smooth, {size_k}, "smoothquant_fold.smooth");
  if (s != Status::kSuccess) return s;
  s = check_dense(norm_weight, dt, 1, 2, "smoothquant_fold.norm_weight");
  if (s != Status::kSuccess) return s;
  s = shape_is(norm_weight, {size_k}, "smoothquant_fold.norm_weight");
  if (s != Status::kSuccess) return s;
  const bool has_bias = tensor_present(norm_bias);
  if (has_bias) {
    s = check_dense(norm_bias, dt, 1, 2, "smoothquant_fold.norm_bias");
    if (s != Status::kSuccess) return s;
    s = shape_is(norm_bias, {size_k}, "smoothquant_fold.norm_bias");
    if (s != Status::kSuccess) return s;
  }
  if (size_n == 0 || size_k == 0) return Status::kSuccess;
  const cudaStream_t stream = stream_of(ctx);
  const unsigned norm_grid = static_cast<unsigned>(ceil_div(size_k, 256));
  // dev::Pack<T, 8> is aligned to 8 * sizeof(T) bytes.
  const bool vec = (size_k % 8 == 0) && (linear_weight.stride[0] % 8 == 0) &&
                   aligned(data_of(linear_weight), static_cast<uintptr_t>(byte_size(dt)) * 8) &&
                   aligned(data_of(smooth), 16);
  const int sms = ctx.compute_units > 0 ? ctx.compute_units : 32;
  const int64_t work = vec ? size_n * (size_k / 8) : size_n * size_k;
  const unsigned grid = static_cast<unsigned>(std::min<int64_t>(ceil_div(work, 256), static_cast<int64_t>(sms) * 16));
#define KOPS_SQ_FOLD(T)                                                                            \
  do {                                                                                             \
    if (vec)                                                                                       \
      smoothquant_scale_rows_kernel<T><<<grid, 256, 0, stream>>>(                                  \
          mut_ptr<T>(linear_weight), linear_weight.stride[0], const_ptr<float>(smooth),            \
          static_cast<int>(size_n), static_cast<int>(size_k));                                     \
    else                                                                                           \
      smoothquant_scale_rows_scalar_kernel<T><<<grid, 256, 0, stream>>>(                           \
          mut_ptr<T>(linear_weight), linear_weight.stride[0], const_ptr<float>(smooth),            \
          static_cast<int>(size_n), static_cast<int>(size_k));                                     \
    smoothquant_fold_norm_kernel<T><<<norm_grid, 256, 0, stream>>>(                                \
        mut_ptr<T>(norm_weight), has_bias ? mut_ptr<T>(norm_bias) : nullptr, const_ptr<float>(smooth), \
        static_cast<int>(size_k));                                                                 \
  } while (0)
  if (dt == DType::kFloat16) KOPS_SQ_FOLD(__half);
  else if (dt == DType::kBFloat16) KOPS_SQ_FOLD(__nv_bfloat16);
  else KOPS_SQ_FOLD(float);
#undef KOPS_SQ_FOLD
  return last_launch("smoothquant_fold");
}

}  // namespace kops::quant
