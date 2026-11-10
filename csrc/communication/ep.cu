// =============================================================================
// kernel-ops | csrc/communication/ep.cu
// B12 rows 4-5: expert-parallel dispatch / combine over peer-mapped buffers.
//   ep_dispatch_layout_kernel -> ep_layout_kernel (counts + rank membership)
//   ep_dispatch_kernel        -> ep_dispatch_normal_kernel / ep_dispatch_ll_simple_kernel
//   ep_combine_kernel         -> ep_combine_normal_kernel / ep_combine_ll_kernel
//
// Transport. Every rank owns one NVLink-domain staging buffer (EPBuffers::nvl_buffers[r])
// and two signal words in the group's symmetrical signal array (EPBuffers::signals):
//   signals[2*r + 0]  arrival counter of rank r's send phase (rank-private atomic target)
//   signals[2*r + 1]  epoch at which rank r's send phase finished (read by every receiver)
// A sender writes records into the *destination's* buffer, region `source rank`, then the
// last local CTA publishes the epoch with a system-scope release. The receiver acquires
// the peer words and compacts the records in source-rank order (normal) or scatters them
// into the masked expert layout (low latency). The layout is deterministic in normal mode;
// the low-latency masked slots are allocated with atomics and are intentionally unordered,
// exactly like DeepEP's masked path.
//
// Records are fixed-size and epoch tagged, so stale records from a previous call (or a
// CUDA-graph replay with a newer epoch) are ignored without a host-side reset:
//   dispatch: [u32 epoch][i32 n_local][i64 src_info][i32 idx[topk]][f32 w[topk]][T x[hidden]]
//   combine : [u32 epoch][i32 token][f32 weight][pad][T x[hidden]]
// The per-source region is `nvl_bytes / num_ranks` rounded down to 128 B; both the sender
// and the receiver derive the same stride from the buffer descriptor.
//
// Numerics. The payload is copied bit for bit. The only arithmetic in dispatch/combine is
// the fp32 routing-weight sum used by combine when `weights_already_applied` is false.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/communication/allreduce_kernels.cuh"
#include "csrc/communication/ep_kernels.cuh"
#include "csrc/communication/comm_gemm_kernels.cuh"
#include "csrc/communication/ops.h"

namespace kops::comm {
namespace {

constexpr uint64_t kEpDefaultTimeoutNs = 20ull * 1000ull * 1000ull * 1000ull;
constexpr int kEpBlock = 256;
constexpr size_t kEpAlign = 128;

// ---- system-scope signalling (same discipline as allreduce.cu) -------------------------------
__device__ __forceinline__ uint32_t ep_ld_acq_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void ep_st_rel_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ep_atom_add_acqrel_gpu(uint32_t* p, uint32_t v) {
  uint32_t old;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ uint64_t ep_timer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---- sizes ------------------------------------------------------------------------------------
size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

/// Region stride of one source rank inside a destination's NVLink buffer.
__host__ __device__ __forceinline__ size_t ep_region_stride(const EPBuffers& b) {
  const size_t w = static_cast<size_t>(b.num_ranks > 0 ? b.num_ranks : 1);
  return (static_cast<size_t>(b.nvl_bytes) / w) / kEpAlign * kEpAlign;
}

constexpr int ep_elems_per_vec(DType t) { return t == DType::kFloat32 ? 4 : 8; }

size_t ep_dispatch_record_bytes(int hidden, int topk, DType dtype) {
  const size_t eb = dtype == DType::kFloat32 ? 4 : 2;
  return align_up(16 + static_cast<size_t>(topk) * 8 + static_cast<size_t>(hidden) * eb, 16);
}
size_t ep_combine_record_bytes(int hidden, DType dtype) {
  const size_t eb = dtype == DType::kFloat32 ? 4 : 2;
  return align_up(16 + static_cast<size_t>(hidden) * eb, 16);
}
template <typename T>
__host__ __device__ constexpr size_t ep_dispatch_record_bytes_t(int hidden, int topk) {
  return ((16 + static_cast<size_t>(topk) * 8 + static_cast<size_t>(hidden) * sizeof(T) + 15) / 16) * 16;
}
template <typename T>
__host__ __device__ constexpr size_t ep_combine_record_bytes_t(int hidden) {
  return ((16 + static_cast<size_t>(hidden) * sizeof(T) + 15) / 16) * 16;
}

/// ---- generic byte copy ------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void ep_copy_row(T* dst, const T* src, int n) {
  if (n <= 0) return;
  constexpr int kVec = sizeof(T) == 4 ? 4 : 8;
  if ((n % kVec) == 0) {
    const int nv = n / kVec;
    uint4* d = reinterpret_cast<uint4*>(dst);
    const uint4* s = reinterpret_cast<const uint4*>(src);
    for (int i = threadIdx.x; i < nv; i += blockDim.x) d[i] = s[i];
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
  }
}

/// ---- send-phase completion --------------------------------------------------------------------
/// All CTAs of a rank are done writing remote records; the last one publishes the epoch.
__device__ __forceinline__ void ep_post_ready(const EPBuffers& b, uint32_t epoch_tag) {
  __syncthreads();
  __threadfence();
  if (threadIdx.x == 0) {
    uint32_t* base = b.signals + static_cast<size_t>(b.rank) * 2;
    const uint32_t prev = ep_atom_add_acqrel_gpu(base + 0, 1u);
    if (prev + 1u == gridDim.x) {
      base[0] = 0u;
      __threadfence_system();
      ep_st_rel_sys(base + 1, epoch_tag);
    }
  }
}

/// Wait until every peer (and this rank) has published `epoch_tag`; returns false on timeout.
__device__ __forceinline__ bool ep_wait_ready(const EPBuffers& b, uint32_t epoch_tag) {
  if (b.num_ranks <= 1) return true;
  __shared__ uint32_t ok;
  if (threadIdx.x == 0) {
    const uint64_t t0 = ep_timer_ns();
    ok = 1u;
    for (int r = 0; r < b.num_ranks; ++r) {
      const uint32_t* p = b.signals + static_cast<size_t>(r) * 2 + 1;
      for (;;) {
        const uint32_t v = ep_ld_acq_sys(p);
        if (static_cast<int32_t>(v - epoch_tag) >= 0) break;
        if (ep_timer_ns() - t0 > kEpDefaultTimeoutNs) {
          ok = 0u;
          break;
        }
      }
      if (ok == 0u) break;
    }
  }
  __syncthreads();
  return ok != 0u;
}

// ---- layout -----------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ int ep_local_index(const int32_t* topk_idx, int t, int topk, int dest,
                                              int e_local, int* local, float* weight,
                                              const float* topk_weights) {
  int n = 0;
  for (int j = 0; j < topk; ++j) {
    const int g = topk_idx[static_cast<int64_t>(t) * topk + j];
    local[j] = -1;
    weight[j] = 0.0f;
    if (g < 0) continue;
    const int l = g - dest * e_local;
    if (l < 0 || l >= e_local) continue;
    local[j] = l;
    weight[j] = topk_weights != nullptr ? topk_weights[static_cast<int64_t>(t) * topk + j] : 1.0f;
    ++n;
  }
  (void)sizeof(T);
  return n;
}

template <typename T>
__device__ __forceinline__ void ep_write_dispatch_record(char* dst, uint32_t epoch_tag,
                                                         const int32_t* idx, const float* w,
                                                         int topk, int64_t src_info, const T* x,
                                                         int hidden) {
  const int off_idx = 16;
  const int off_w = off_idx + topk * 4;
  const int off_x = static_cast<int>((off_w + topk * 4 + 15) & ~15);
  *reinterpret_cast<uint32_t*>(dst) = epoch_tag;
  *reinterpret_cast<int32_t*>(dst + 4) = 0;
  *reinterpret_cast<int64_t*>(dst + 8) = src_info;
  int32_t* di = reinterpret_cast<int32_t*>(dst + off_idx);
  float* dw = reinterpret_cast<float*>(dst + off_w);
  for (int j = 0; j < topk; ++j) {
    di[j] = idx[j];
    dw[j] = w[j];
  }
  ep_copy_row<T>(reinterpret_cast<T*>(dst + off_x), x, hidden);
}

// ---- layout -----------------------------------------------------------------------------------
__global__ void __launch_bounds__(kEpBlock)
    ep_layout_kernel(int32_t* __restrict__ num_tokens_per_rank,
                     int32_t* __restrict__ num_tokens_per_expert,
                     uint8_t* __restrict__ is_token_in_rank,
                     const int32_t* __restrict__ topk_idx, const int num_tokens, const int topk,
                     const int num_experts, const int num_ranks) {
  for (int r = threadIdx.x; r < num_ranks; r += blockDim.x) num_tokens_per_rank[r] = 0;
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) num_tokens_per_expert[e] = 0;
  __syncthreads();
  for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
    uint32_t mask = 0u;
    for (int j = 0; j < topk; ++j) {
      const int e = topk_idx[static_cast<int64_t>(t) * topk + j];
      if (e < 0 || e >= num_experts) continue;
      atomicAdd(&num_tokens_per_expert[e], 1);
      const int r = e / (num_experts / num_ranks);
      mask |= 1u << r;
    }
    for (int r = 0; r < num_ranks; ++r) {
      const uint8_t in = (mask & (1u << r)) != 0u ? 1u : 0u;
      is_token_in_rank[static_cast<int64_t>(t) * num_ranks + r] = in;
      if (in != 0u) atomicAdd(&num_tokens_per_rank[r], 1);
    }
  }
}

// ---- dispatch ---------------------------------------------------------------------------------
template <typename T, int W>
__global__ void __launch_bounds__(kEpBlock)
    ep_dispatch_normal_kernel(const __grid_constant__ EPDispatchParams p) {
  const uint32_t epoch_tag = p.epoch + 1u;
  const int e_local = p.num_experts / p.buffers.num_ranks;
  const int rank = p.buffers.rank;
  const size_t rec = ep_dispatch_record_bytes_t<T>(p.hidden, p.topk);
  const size_t region = ep_region_stride(p.buffers);
  const bool send_phase = !p.hook_mode || p.phase == 0;
  const bool recv_phase = !p.hook_mode || p.phase == 1;

  if (send_phase) {
    for (int t = blockIdx.x; t < p.num_tokens; t += gridDim.x) {
      for (int d = 0; d < p.buffers.num_ranks; ++d) {
        if (p.is_token_in_rank[static_cast<int64_t>(t) * p.buffers.num_ranks + d] == 0) continue;
        int32_t idx[16];
        float w[16];
        const int n = ep_local_index<T>(p.topk_idx, t, p.topk, d, e_local, idx, w, p.topk_weights);
        if (n == 0) continue;
        char* dst = static_cast<char*>(p.buffers.nvl_buffers[d]) + region * static_cast<size_t>(rank) +
                    rec * static_cast<size_t>(t);
        const int64_t src_info = (static_cast<int64_t>(rank) << 32) | static_cast<uint32_t>(t);
        ep_write_dispatch_record<T>(dst, epoch_tag, idx, w, p.topk, src_info,
                                    static_cast<const T*>(p.x) + static_cast<int64_t>(t) * p.hidden,
                                    p.hidden);
        *reinterpret_cast<int32_t*>(dst + 4) = n;
      }
    }
    ep_post_ready(p.buffers, epoch_tag);
    if (!recv_phase) return;
  }

  if (blockIdx.x != 0) return;
  if (!ep_wait_ready(p.buffers, epoch_tag)) {
    if (threadIdx.x == 0) p.buffers.signals[0] = 0u;  // never used as an error channel; see ops.h
    return;
  }

  const char* own = static_cast<const char*>(p.buffers.nvl_buffers[rank]);
  int row = 0;
  for (int s = 0; s < p.buffers.num_ranks; ++s) {
    const char* region_base = own + region * static_cast<size_t>(s);
    for (int t = 0; t < p.num_tokens; ++t) {
      uint32_t re = 0;
      int32_t n_local = 0;
      int64_t src_info = 0;
      int32_t idx[16];
      float w[16];
      const char* src = region_base + rec * static_cast<size_t>(t);
      re = *reinterpret_cast<const uint32_t*>(src);
      if (re != epoch_tag) continue;
      n_local = *reinterpret_cast<const int32_t*>(src + 4);
      if (n_local <= 0) continue;
      src_info = *reinterpret_cast<const int64_t*>(src + 8);
      const int off_idx = 16, off_w = off_idx + p.topk * 4;
      const int off_x = (off_w + p.topk * 4 + 15) & ~15;
      const int32_t* si = reinterpret_cast<const int32_t*>(src + off_idx);
      const float* sw = reinterpret_cast<const float*>(src + off_w);
      for (int j = 0; j < p.topk; ++j) {
        idx[j] = si[j];
        w[j] = sw[j];
      }
      // Copy the payload directly from the record: X is caller memory and may be much
      // larger than the stack staging above.
      T* dstx = static_cast<T*>(p.recv_x) + static_cast<int64_t>(row) * p.hidden;
      ep_copy_row<T>(dstx, reinterpret_cast<const T*>(src + off_x), p.hidden);
      if (p.recv_src_info != nullptr) p.recv_src_info[row] = src_info;
      if (p.recv_topk_idx != nullptr) {
        for (int j = 0; j < p.topk; ++j) p.recv_topk_idx[static_cast<int64_t>(row) * p.topk + j] = idx[j];
      }
      if (p.recv_topk_weights != nullptr) {
        for (int j = 0; j < p.topk; ++j)
          p.recv_topk_weights[static_cast<int64_t>(row) * p.topk + j] = w[j];
      }
      ++row;
    }
  }
  // per-expert counts (normal mode); the caller can use them for the expert GEMM task map.
  if (p.recv_count_per_expert != nullptr && threadIdx.x == 0) {
    for (int e = 0; e < e_local; ++e) p.recv_count_per_expert[e] = 0;
    for (int r = 0; r < row; ++r) {
      for (int j = 0; j < p.topk; ++j) {
        const int e = p.recv_topk_idx != nullptr ? p.recv_topk_idx[static_cast<int64_t>(r) * p.topk + j] : -1;
        if (e >= 0 && e < e_local) ++p.recv_count_per_expert[e];
      }
    }
  }
}

template <typename T, int W>
__global__ void __launch_bounds__(kEpBlock)
    ep_dispatch_ll_kernel(const __grid_constant__ EPDispatchParams p) {
  const uint32_t epoch_tag = p.epoch + 1u;
  const int e_local = p.num_experts / p.buffers.num_ranks;
  const int rank = p.buffers.rank;
  const size_t rec = ep_dispatch_record_bytes_t<T>(p.hidden, p.topk);
  const size_t region = ep_region_stride(p.buffers);
  const bool send_phase = !p.hook_mode || p.phase == 0;
  const bool recv_phase = !p.hook_mode || p.phase == 1;

  if (send_phase) {
    for (int t = blockIdx.x; t < p.num_tokens; t += gridDim.x) {
      for (int d = 0; d < p.buffers.num_ranks; ++d) {
        if (p.is_token_in_rank[static_cast<int64_t>(t) * p.buffers.num_ranks + d] == 0) continue;
        int32_t idx[16];
        float w[16];
        const int n = ep_local_index<T>(p.topk_idx, t, p.topk, d, e_local, idx, w, p.topk_weights);
        if (n == 0) continue;
        char* dst = static_cast<char*>(p.buffers.nvl_buffers[d]) + region * static_cast<size_t>(rank) +
                    rec * static_cast<size_t>(t);
        const int64_t src_info = (static_cast<int64_t>(rank) << 32) | static_cast<uint32_t>(t);
        ep_write_dispatch_record<T>(dst, epoch_tag, idx, w, p.topk, src_info,
                                    static_cast<const T*>(p.x) + static_cast<int64_t>(t) * p.hidden,
                                    p.hidden);
        *reinterpret_cast<int32_t*>(dst + 4) = n;
      }
    }
    ep_post_ready(p.buffers, epoch_tag);
    if (!recv_phase) return;
  }

  if (blockIdx.x != 0) return;
  if (!ep_wait_ready(p.buffers, epoch_tag)) return;
  // The masked slot is allocated by thread 0 (one atomic per (token, local expert)); the payload is
  // copied by the whole CTA so the low-latency receive is not serialised behind one thread.
  __shared__ int32_t sh_n;
  __shared__ int32_t sh_idx[16];
  __shared__ float sh_w[16];
  __shared__ int64_t sh_src;
  __shared__ const T* sh_x;
  __shared__ int64_t sh_dstrow;
  __shared__ uint32_t sh_copy;
  if (threadIdx.x == 0) {
    for (int e = 0; e < e_local; ++e) p.recv_count_per_expert[e] = 0;
  }
  __syncthreads();
  const char* own = static_cast<const char*>(p.buffers.nvl_buffers[rank]);
  const int cap = p.max_tokens_per_rank;  // masked rows per local expert
  for (int s = 0; s < p.buffers.num_ranks; ++s) {
    const char* region_base = own + region * static_cast<size_t>(s);
    for (int t = 0; t < p.num_tokens; ++t) {
      const char* src = region_base + rec * static_cast<size_t>(t);
      if (threadIdx.x == 0) {
        sh_n = 0;
        const uint32_t re = *reinterpret_cast<const uint32_t*>(src);
        if (re == epoch_tag) {
          sh_n = *reinterpret_cast<const int32_t*>(src + 4);
          sh_src = *reinterpret_cast<const int64_t*>(src + 8);
          const int off_idx = 16, off_w = off_idx + p.topk * 4;
          const int off_x = (off_w + p.topk * 4 + 15) & ~15;
          const int32_t* si = reinterpret_cast<const int32_t*>(src + off_idx);
          const float* sw = reinterpret_cast<const float*>(src + off_w);
          for (int j = 0; j < p.topk; ++j) {
            sh_idx[j] = si[j];
            sh_w[j] = sw[j];
          }
          sh_x = reinterpret_cast<const T*>(src + off_x);
        }
      }
      __syncthreads();
      if (sh_n <= 0) continue;
      for (int j = 0; j < p.topk; ++j) {
        if (threadIdx.x == 0) {
          sh_copy = 0u;
          const int e = sh_idx[j];
          if (e >= 0 && e < e_local) {
            const int slot = atomicAdd(&p.recv_count_per_expert[e], 1);
            if (slot < cap) {
              const int64_t dstrow = static_cast<int64_t>(e) * cap + slot;
              sh_dstrow = dstrow;
              sh_copy = 1u;
              if (p.recv_src_info != nullptr) p.recv_src_info[dstrow] = sh_src;
              if (p.recv_topk_weights != nullptr) p.recv_topk_weights[dstrow] = sh_w[j];
            }
          }
        }
        __syncthreads();
        if (sh_copy != 0u) {
          ep_copy_row<T>(static_cast<T*>(p.recv_x) + sh_dstrow * p.hidden, sh_x, p.hidden);
        }
        __syncthreads();
      }
    }
  }
}

// ---- combine ----------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void ep_write_combine_record(char* dst, uint32_t epoch_tag, int token,
                                                        float weight, const T* x, int hidden) {
  *reinterpret_cast<uint32_t*>(dst) = epoch_tag;
  *reinterpret_cast<int32_t*>(dst + 4) = token;
  *reinterpret_cast<float*>(dst + 8) = weight;
  *reinterpret_cast<uint32_t*>(dst + 12) = 0u;
  ep_copy_row<T>(reinterpret_cast<T*>(dst + 16), x, hidden);
}

template <typename T, int W>
__global__ void __launch_bounds__(kEpBlock)
    ep_combine_normal_kernel(const __grid_constant__ EPCombineParams p) {
  const uint32_t epoch_tag = p.epoch + 1u;
  const int rank = p.buffers.rank;
  const size_t rec = ep_combine_record_bytes_t<T>(p.hidden);
  const size_t region = ep_region_stride(p.buffers);
  const size_t slots = static_cast<size_t>(p.num_tokens);
  const bool send_phase = !p.hook_mode || p.phase == 0;
  const bool recv_phase = !p.hook_mode || p.phase == 1;

  if (send_phase) {
    for (int i = blockIdx.x; i < p.num_tokens; i += gridDim.x) {
      if (p.src_info == nullptr || p.src_info[i] < 0) continue;
      const int64_t src_info = p.src_info[i];
      const int dest = static_cast<int>(static_cast<uint64_t>(src_info) >> 32);
      const int token = static_cast<int>(static_cast<uint64_t>(src_info) & 0xFFFFFFFFu);
      if (dest < 0 || dest >= p.buffers.num_ranks) continue;
      float weight = 1.0f;
      if (!p.weights_already_applied && p.topk_weights != nullptr) {
        weight = 0.0f;
        for (int j = 0; j < p.topk; ++j) weight += p.topk_weights[static_cast<int64_t>(i) * p.topk + j];
      }
      char* dst = static_cast<char*>(p.buffers.nvl_buffers[dest]) + region * static_cast<size_t>(rank) +
                  rec * static_cast<size_t>(i);
      ep_write_combine_record<T>(dst, epoch_tag, token, weight,
                                 static_cast<const T*>(p.expert_out) + static_cast<int64_t>(i) * p.hidden,
                                 p.hidden);
    }
    ep_post_ready(p.buffers, epoch_tag);
    if (!recv_phase) return;
  }

  if (blockIdx.x != 0) return;
  if (!ep_wait_ready(p.buffers, epoch_tag)) return;

  // Initialise out (residual copy or zero) before accumulating the peer contributions.
  const T* residual = static_cast<const T*>(p.residual);
  T* out = static_cast<T*>(p.out);
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < static_cast<int64_t>(p.num_tokens) * p.hidden;
       i += blockDim.x) {
    out[i] = residual != nullptr ? residual[i] : static_cast<T>(0.0f);
  }
  __syncthreads();

  const char* own = static_cast<const char*>(p.buffers.nvl_buffers[rank]);
  for (int s = 0; s < p.buffers.num_ranks; ++s) {
    const char* region_base = own + region * static_cast<size_t>(s);
    for (size_t i = 0; i < slots; ++i) {
      const char* src = region_base + rec * i;
      if (*reinterpret_cast<const uint32_t*>(src) != epoch_tag) continue;
      const int token = *reinterpret_cast<const int32_t*>(src + 4);
      const float w = *reinterpret_cast<const float*>(src + 8);
      if (token < 0 || token >= p.num_tokens) continue;
      T* dstrow = out + static_cast<int64_t>(token) * p.hidden;
      const T* srcrow = reinterpret_cast<const T*>(src + 16);
      for (int c = threadIdx.x; c < p.hidden; c += blockDim.x) {
        dstrow[c] = static_cast<T>(static_cast<float>(dstrow[c]) + static_cast<float>(srcrow[c]) * w);
      }
      __syncthreads();
    }
  }
}

template <typename T, int W>
__global__ void __launch_bounds__(kEpBlock)
    ep_combine_ll_kernel(const __grid_constant__ EPCombineParams p) {
  const uint32_t epoch_tag = p.epoch + 1u;
  const int rank = p.buffers.rank;
  const size_t rec = ep_combine_record_bytes_t<T>(p.hidden);
  const size_t region = ep_region_stride(p.buffers);
  const size_t slots = static_cast<size_t>(p.num_tokens);
  const bool send_phase = !p.hook_mode || p.phase == 0;
  const bool recv_phase = !p.hook_mode || p.phase == 1;

  if (send_phase) {
    for (size_t i = blockIdx.x; i < slots; i += gridDim.x) {
      if (p.src_info == nullptr || p.src_info[i] < 0) continue;
      const int64_t src_info = p.src_info[i];
      const int dest = static_cast<int>(static_cast<uint64_t>(src_info) >> 32);
      const int token = static_cast<int>(static_cast<uint64_t>(src_info) & 0xFFFFFFFFu);
      if (dest < 0 || dest >= p.buffers.num_ranks) continue;
      float weight = 1.0f;
      if (!p.weights_already_applied && p.topk_weights != nullptr) weight = p.topk_weights[i];
      char* dst = static_cast<char*>(p.buffers.nvl_buffers[dest]) + region * static_cast<size_t>(rank) +
                  rec * i;
      ep_write_combine_record<T>(dst, epoch_tag, token, weight,
                                 static_cast<const T*>(p.expert_out) + static_cast<int64_t>(i) * p.hidden,
                                 p.hidden);
    }
    ep_post_ready(p.buffers, epoch_tag);
    if (!recv_phase) return;
  }

  if (blockIdx.x != 0) return;
  if (!ep_wait_ready(p.buffers, epoch_tag)) return;

  const T* residual = static_cast<const T*>(p.residual);
  T* out = static_cast<T*>(p.out);
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < static_cast<int64_t>(p.num_tokens) * p.hidden;
       i += blockDim.x) {
    out[i] = residual != nullptr ? residual[i] : static_cast<T>(0.0f);
  }
  __syncthreads();

  const char* own = static_cast<const char*>(p.buffers.nvl_buffers[rank]);
  for (int s = 0; s < p.buffers.num_ranks; ++s) {
    const char* region_base = own + region * static_cast<size_t>(s);
    for (size_t i = 0; i < slots; ++i) {
      const char* src = region_base + rec * i;
      if (*reinterpret_cast<const uint32_t*>(src) != epoch_tag) continue;
      const int token = *reinterpret_cast<const int32_t*>(src + 4);
      const float w = *reinterpret_cast<const float*>(src + 8);
      if (token < 0 || token >= p.num_tokens) continue;
      T* dstrow = out + static_cast<int64_t>(token) * p.hidden;
      const T* srcrow = reinterpret_cast<const T*>(src + 16);
      for (int c = threadIdx.x; c < p.hidden; c += blockDim.x) {
        dstrow[c] = static_cast<T>(static_cast<float>(dstrow[c]) + static_cast<float>(srcrow[c]) * w);
      }
      __syncthreads();
    }
  }
}

// ---- host side ---------------------------------------------------------------------------------
template <typename T, int W>
Status launch_dispatch(const EPDispatchParams& p, cudaStream_t s) {
  const int ctas = std::max(1, std::min<int>(16, (p.num_tokens + 7) / 8));
  const EPDispatchParams* pp = &p;
  if (p.low_latency) {
    ep_dispatch_ll_kernel<T, W><<<ctas, kEpBlock, 0, s>>>(*pp);
  } else {
    ep_dispatch_normal_kernel<T, W><<<ctas, kEpBlock, 0, s>>>(*pp);
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e),
                                     "ep_dispatch: launch failed: %s", cudaGetErrorString(e));
  }
  return Status::kSuccess;
}

template <typename T, int W>
Status launch_combine(const EPCombineParams& p, cudaStream_t s) {
  const int ctas = std::max(1, std::min<int>(16, (p.num_tokens + 7) / 8));
  if (p.low_latency) {
    ep_combine_ll_kernel<T, W><<<ctas, kEpBlock, 0, s>>>(p);
  } else {
    ep_combine_normal_kernel<T, W><<<ctas, kEpBlock, 0, s>>>(p);
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e),
                                     "ep_combine: launch failed: %s", cudaGetErrorString(e));
  }
  return Status::kSuccess;
}

Status ep_check_buffers(const EPBuffers& b, const char* who) {
  if (b.num_ranks < 1 || b.num_ranks > kMaxRanks || b.rank < 0 || b.rank >= b.num_ranks) {
    return detail::set_error(Status::kInvalidArgument, "%s: rank %d / num_ranks %d", who, b.rank,
                             b.num_ranks);
  }
  if (!(b.num_ranks == 1 || b.num_ranks == 2 || b.num_ranks == 4 || b.num_ranks == 8)) {
    return detail::set_error(Status::kUnsupportedShape, "%s: num_ranks %d (compiled: 1, 2, 4, 8)", who,
                             b.num_ranks);
  }
  if (b.transport != EPTransport::kNvlink) {
    return detail::set_error(Status::kUnsupportedBackend,
                             "%s: transport %d; only the NVLink peer transport is implemented", who,
                             static_cast<int>(b.transport));
  }
  if (b.num_ranks == 1) return Status::kSuccess;
  if (b.signals == nullptr) return detail::set_error(Status::kInvalidArgument, "%s: signals is null", who);
  for (int r = 0; r < b.num_ranks; ++r) {
    if (b.nvl_buffers[r] == nullptr) {
      return detail::set_error(Status::kInvalidArgument, "%s: nvl_buffers[%d] is null", who, r);
    }
    if ((reinterpret_cast<uintptr_t>(b.nvl_buffers[r]) & 15u) != 0) {
      return detail::set_error(Status::kMisaligned, "%s: nvl_buffers[%d] must be 16 B aligned", who, r);
    }
  }
  return Status::kSuccess;
}

Status ep_check_sizes(const EPBuffers& b, size_t bytes, const char* who) {
  if (b.nvl_bytes < bytes) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "%s: nvl_bytes %zu < %zu needed for %d source regions", who, b.nvl_bytes,
                             bytes, b.num_ranks);
  }
  return Status::kSuccess;
}

}  // namespace

// ---- public -----------------------------------------------------------------------------------
Status validate_ep_buffers(const EPBuffers& buffers, int32_t hidden, int32_t max_tokens_per_rank,
                           int32_t num_experts) {
  if (hidden <= 0 || max_tokens_per_rank < 0 || num_experts <= 0 || num_experts % buffers.num_ranks != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "validate_ep_buffers: hidden %d, max_tokens_per_rank %d, num_experts %d, "
                             "num_ranks %d",
                             hidden, max_tokens_per_rank, num_experts, buffers.num_ranks);
  }
  Status s = ep_check_buffers(buffers, "validate_ep_buffers");
  if (s != Status::kSuccess) return s;
  if (buffers.num_ranks == 1) return Status::kSuccess;
  const size_t rec = ep_dispatch_record_bytes(hidden, 1, DType::kFloat32);
  const size_t need = static_cast<size_t>(buffers.num_ranks) *
                      align_up(static_cast<size_t>(max_tokens_per_rank) * rec, kEpAlign) * 2;
  s = ep_check_sizes(buffers, need, "validate_ep_buffers");
  if (s != Status::kSuccess) return s;
  if ((reinterpret_cast<uintptr_t>(buffers.signals) & 15u) != 0) {
    return detail::set_error(Status::kMisaligned, "validate_ep_buffers: signals must be 16 B aligned");
  }
  return Status::kSuccess;
}

Status ep_dispatch_layout(const Tensor& num_tokens_per_rank, const Tensor& num_tokens_per_expert,
                          const Tensor& is_token_in_rank, const Tensor& topk_idx,
                          int32_t num_experts, int32_t num_ranks, const LaunchContext& ctx) {
  Status s = check_tensor(num_tokens_per_rank, DType::kInt32, 1, 4, "ep_dispatch_layout.num_tokens_per_rank");
  if (s != Status::kSuccess) return s;
  s = check_tensor(num_tokens_per_expert, DType::kInt32, 1, 4, "ep_dispatch_layout.num_tokens_per_expert");
  if (s != Status::kSuccess) return s;
  s = check_tensor(is_token_in_rank, DType::kUInt8, 1, 1, "ep_dispatch_layout.is_token_in_rank");
  if (s != Status::kSuccess) return s;
  s = check_tensor(topk_idx, DType::kInt32, 2, 4, "ep_dispatch_layout.topk_idx");
  if (s != Status::kSuccess) return s;
  if (num_experts <= 0 || num_ranks <= 0 || num_ranks > kMaxRanks || num_experts % num_ranks != 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "ep_dispatch_layout: num_experts %d / num_ranks %d", num_experts, num_ranks);
  }
  if (!tensor_is_contiguous(topk_idx)) {
    return detail::set_error(Status::kUnsupportedLayout, "ep_dispatch_layout: topk_idx must be contiguous");
  }
  const int64_t num_tokens = topk_idx.shape[0];
  const int64_t topk = topk_idx.shape[1];
  if (tensor_numel(num_tokens_per_rank) != num_ranks ||
      tensor_numel(num_tokens_per_expert) != num_experts ||
      tensor_numel(is_token_in_rank) != num_tokens * num_ranks) {
    return detail::set_error(Status::kInvalidArgument, "ep_dispatch_layout: output tensor sizes");
  }
  const int ctas = num_tokens == 0 ? 0 : 1;
  if (ctas == 0) {
    // still zero the counters
    cudaMemsetAsync(num_tokens_per_rank.data, 0, sizeof(int32_t) * num_ranks,
                    reinterpret_cast<cudaStream_t>(ctx.stream));
    cudaMemsetAsync(num_tokens_per_expert.data, 0, sizeof(int32_t) * num_experts,
                    reinterpret_cast<cudaStream_t>(ctx.stream));
    return Status::kSuccess;
  }
  ep_layout_kernel<<<1, kEpBlock, 0, reinterpret_cast<cudaStream_t>(ctx.stream)>>>(
      static_cast<int32_t*>(num_tokens_per_rank.data), static_cast<int32_t*>(num_tokens_per_expert.data),
      static_cast<uint8_t*>(is_token_in_rank.data), static_cast<const int32_t*>(topk_idx.data),
      static_cast<int>(num_tokens), static_cast<int>(topk), num_experts, num_ranks);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(e),
                                     "ep_dispatch_layout: launch failed: %s", cudaGetErrorString(e));
  }
  return Status::kSuccess;
}

Status ep_dispatch(const EPDispatchParams& params, DType dtype, const LaunchContext& ctx) {
  if (dtype != DType::kBFloat16 && dtype != DType::kFloat16 && dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "ep_dispatch: dtype %s (bf16, fp16, fp32)",
                             dtype_name(dtype));
  }
  if (params.payload_quant != ActQuantMode::kNone) {
    return detail::set_error(Status::kNotImplemented,
                             "ep_dispatch: payload_quant %d (the NVLink path copies the payload as-is)",
                             static_cast<int>(params.payload_quant));
  }
  if (params.num_tokens < 0 || params.hidden <= 0 || params.topk <= 0 || params.topk > 16 ||
      params.num_experts <= 0) {
    return detail::set_error(Status::kInvalidArgument,
                             "ep_dispatch: num_tokens %d, hidden %d, topk %d, num_experts %d",
                             params.num_tokens, params.hidden, params.topk, params.num_experts);
  }
  if (params.num_tokens == 0) return Status::kSuccess;
  Status s = ep_check_buffers(params.buffers, "ep_dispatch");
  if (s != Status::kSuccess) return s;
  if (params.is_token_in_rank == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "ep_dispatch: is_token_in_rank is null");
  }
  if (params.buffers.num_ranks > 1) {
    if (params.num_experts % params.buffers.num_ranks != 0) {
      return detail::set_error(Status::kUnsupportedShape,
                               "ep_dispatch: num_experts %d is not divisible by num_ranks %d",
                               params.num_experts, params.buffers.num_ranks);
    }
    const size_t rec = ep_dispatch_record_bytes(params.hidden, params.topk, dtype);
    const size_t region = ep_region_stride(params.buffers);
    s = ep_check_sizes(params.buffers,
                       static_cast<size_t>(params.buffers.num_ranks) *
                           align_up(static_cast<size_t>(params.num_tokens) * rec, kEpAlign),
                       "ep_dispatch");
    if (s != Status::kSuccess) return s;
    if (region < static_cast<size_t>(params.num_tokens) * rec) {
      return detail::set_error(Status::kWorkspaceTooSmall,
                               "ep_dispatch: source region %zu B < %zu B needed", region,
                               static_cast<size_t>(params.num_tokens) * rec);
    }
  }
  if (params.x == nullptr || params.topk_idx == nullptr || params.recv_x == nullptr) {
    return detail::set_error(Status::kInvalidArgument, "ep_dispatch: x / topk_idx / recv_x is null");
  }
  if (params.hidden % ep_elems_per_vec(dtype) != 0) {
    return detail::set_error(Status::kUnsupportedShape, "ep_dispatch: hidden %d is not a multiple of %d",
                             params.hidden, ep_elems_per_vec(dtype));
  }
  if (params.hook_mode && params.phase != 0 && params.phase != 1) {
    return detail::set_error(Status::kInvalidArgument, "ep_dispatch: hook phase %d (0 or 1)", params.phase);
  }
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int w = params.buffers.num_ranks;
#define KOPS_EP_DISPATCH(T)                                                                        \
  do {                                                                                             \
    switch (w) {                                                                                   \
      case 1: return launch_dispatch<T, 1>(params, st);                                            \
      case 2: return launch_dispatch<T, 2>(params, st);                                            \
      case 4: return launch_dispatch<T, 4>(params, st);                                            \
      default: return launch_dispatch<T, 8>(params, st);                                           \
    }                                                                                              \
  } while (0)
  switch (dtype) {
    case DType::kBFloat16: KOPS_EP_DISPATCH(__nv_bfloat16);
    case DType::kFloat16: KOPS_EP_DISPATCH(__half);
    default: KOPS_EP_DISPATCH(float);
  }
#undef KOPS_EP_DISPATCH
}

Status ep_combine(const EPCombineParams& params, DType dtype, const LaunchContext& ctx) {
  if (dtype != DType::kBFloat16 && dtype != DType::kFloat16 && dtype != DType::kFloat32) {
    return detail::set_error(Status::kUnsupportedDtype, "ep_combine: dtype %s (bf16, fp16, fp32)",
                             dtype_name(dtype));
  }
  if (params.num_tokens < 0 || params.hidden <= 0 || params.topk <= 0 || params.topk > 16) {
    return detail::set_error(Status::kInvalidArgument,
                             "ep_combine: num_tokens %d, hidden %d, topk %d", params.num_tokens,
                             params.hidden, params.topk);
  }
  if (params.num_tokens == 0) return Status::kSuccess;
  Status s = ep_check_buffers(params.buffers, "ep_combine");
  if (s != Status::kSuccess) return s;
  if (params.src_info == nullptr || params.expert_out == nullptr || params.out == nullptr) {
    return detail::set_error(Status::kInvalidArgument,
                             "ep_combine: src_info / expert_out / out is null");
  }
  if (params.buffers.num_ranks > 1) {
    const size_t rec = ep_combine_record_bytes(params.hidden, dtype);
    const size_t region = ep_region_stride(params.buffers);
    const size_t slots = static_cast<size_t>(params.num_tokens);
    s = ep_check_sizes(params.buffers, slots * rec * params.buffers.num_ranks, "ep_combine");
    if (s != Status::kSuccess) return s;
    if (region < slots * rec) {
      return detail::set_error(Status::kWorkspaceTooSmall,
                               "ep_combine: source region %zu B < %zu B needed", region, slots * rec);
    }
  }
  if (params.hook_mode && params.phase != 0 && params.phase != 1) {
    return detail::set_error(Status::kInvalidArgument, "ep_combine: hook phase %d (0 or 1)", params.phase);
  }
  if (params.hidden % ep_elems_per_vec(dtype) != 0) {
    return detail::set_error(Status::kUnsupportedShape, "ep_combine: hidden %d is not a multiple of %d",
                             params.hidden, ep_elems_per_vec(dtype));
  }
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(ctx.stream);
  const int w = params.buffers.num_ranks;
#define KOPS_EP_COMBINE(T)                                                                         \
  do {                                                                                             \
    switch (w) {                                                                                   \
      case 1: return launch_combine<T, 1>(params, st);                                             \
      case 2: return launch_combine<T, 2>(params, st);                                             \
      case 4: return launch_combine<T, 4>(params, st);                                             \
      default: return launch_combine<T, 8>(params, st);                                            \
    }                                                                                              \
  } while (0)
  switch (dtype) {
    case DType::kBFloat16: KOPS_EP_COMBINE(__nv_bfloat16);
    case DType::kFloat16: KOPS_EP_COMBINE(__half);
    default: KOPS_EP_COMBINE(float);
  }
#undef KOPS_EP_COMBINE
}

}  // namespace kops::comm
