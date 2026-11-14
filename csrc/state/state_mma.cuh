#pragma once
// =============================================================================
// kernel-ops | csrc/state/state_mma.cuh
// Shared mma.sync m16n8k16 primitives of the state family (Mamba-2 SSD, lightning
// attention, chunked delta rule): bf16 / fp16 operands, fp32 accumulation.
// Device-only header: family translation units include it, host code does not.
//   frag_a       A (16 x 16) of a row-major tile tile[m * ld + k]
//   frag_a_trans A of a tile with the other axis contiguous, tile[k * ld + m]
//   frag_b2      B fragments of two adjacent n8 tiles at (n0, k0) of tile[n * ld + k]
//   frag_b2_trans the same of tile[k * ld + n] (n contiguous)
//   tc_stage_*   stage a 64-row input tile (16-bit or fp32) as a 16-byte-aligned tile
// Every helper keeps the operand layout of the family kernels (ld a multiple of 8,
// tiles 16-byte aligned); mma is only emitted for SM80 and newer, the callers pick
// the body at run time from the launch context.
// Requires: common/macros.cuh, common/dtype.cuh.
// =============================================================================

namespace kops::state {
namespace mma {

using bf16 = __nv_bfloat16;

/// Row stride (in 16-bit elements) of a staged tile of `cols` columns: 8 elements
/// (16 bytes) more than the data, so the eight rows of an ldmatrix hit distinct banks.
__host__ __device__ constexpr int tc_ld(int cols) { return cols + 8; }

// ---- 16-bit packing / conversion ----------------------------------------------------------------

namespace bits {

/// Bit reinterpretation without a CUDA intrinsic, valid in host and device code.
union F32U32 {
  float f;
  uint32_t u;
};

}  // namespace bits

/// Two floats as one bf16x2 register, `lo` in the low half (round to nearest even).
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
#if defined(__CUDA_ARCH__)
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
#else
  const bits::F32U32 l{lo}, h{hi};
  const uint32_t rl = (l.u + 0x7fffu + ((l.u >> 16) & 1u)) >> 16;
  const uint32_t rh = (h.u + 0x7fffu + ((h.u >> 16) & 1u)) >> 16;
  return rl | (rh << 16);
#endif
}

/// Two floats as one fp16x2 register, `lo` in the low half (round to nearest even).
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
#if defined(__CUDA_ARCH__)
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
#else
  return pack_bf16x2(lo, hi);  // host syntax check only: never executed
#endif
}

/// float -> 16-bit element (round to nearest even).
__device__ __forceinline__ void st16(__nv_bfloat16& d, float v) {
#if defined(__CUDA_ARCH__)
  d = __float2bfloat16_rn(v);
#else
  (void)d;
  (void)v;
#endif
}
__device__ __forceinline__ void st16(__half& d, float v) {
#if defined(__CUDA_ARCH__)
  d = __float2half_rn(v);
#else
  (void)d;
  (void)v;
#endif
}
__device__ __forceinline__ void st16(float& d, float v) { d = v; }

/// 16-bit element -> float.
__device__ __forceinline__ float ld16(__nv_bfloat16 v) {
#if defined(__CUDA_ARCH__)
  return __bfloat162float(v);
#else
  (void)v;
  return 0.0f;
#endif
}
__device__ __forceinline__ float ld16(__half v) {
#if defined(__CUDA_ARCH__)
  return __half2float(v);
#else
  (void)v;
  return 0.0f;
#endif
}
__device__ __forceinline__ float ld16(float v) { return v; }

/// Element type -> the pack used by the staging helpers.
template <typename T>
struct Bits16;

template <>
struct Bits16<__nv_bfloat16> {
  __device__ static __forceinline__ uint32_t pack(float lo, float hi) { return pack_bf16x2(lo, hi); }
  __device__ static __forceinline__ void unpack(float (&v)[2], uint32_t w) {
    bits::F32U32 lo, hi;
    lo.u = w << 16;
    hi.u = w & 0xffff0000u;
    v[0] = lo.f;
    v[1] = hi.f;
  }
};

template <>
struct Bits16<__half> {
  __device__ static __forceinline__ uint32_t pack(float lo, float hi) { return pack_f16x2(lo, hi); }
  __device__ static __forceinline__ void unpack(float (&v)[2], uint32_t w) {
#if defined(__CUDA_ARCH__)
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w));
    v[0] = f.x;
    v[1] = f.y;
#else
    v[0] = 0.0f;
    v[1] = 0.0f;
    (void)w;
#endif
  }
};

// ---- mma.sync m16n8k16 (fp32 accumulation) ------------------------------------------------------

/// d += a (16 x 16, bf16) * b (16 x 8, bf16), fp32 accumulation.
__device__ __forceinline__ void mma_bf16_16x8x16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  (void)d;
  (void)a;
  (void)b0;
  (void)b1;
#endif
}

/// d += a (16 x 16, fp16) * b (16 x 8, fp16), fp32 accumulation.
__device__ __forceinline__ void mma_f16_16x8x16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  (void)d;
  (void)a;
  (void)b0;
  (void)b1;
#endif
}

/// The m16n8k16 mma of one operand type: `Mma16x8x16<T>::run(d, a, b0, b1)`.
template <typename T>
struct Mma16x8x16;

template <>
struct Mma16x8x16<__nv_bfloat16> {
  __device__ static __forceinline__ void run(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    mma_bf16_16x8x16(d, a, b0, b1);
  }
};

template <>
struct Mma16x8x16<__half> {
  __device__ static __forceinline__ void run(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    mma_f16_16x8x16(d, a, b0, b1);
  }
};

// ---- ldmatrix fragment loads --------------------------------------------------------------------

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* smem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#else
  (void)r;
  (void)smem;
#endif
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void* smem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#else
  (void)r;
  (void)smem;
#endif
}

/// The A fragment (16 x 16) at (m0, k0) of the row-major tile tile[m * ld + k]; ld is a multiple of
/// 8 and the tile 16-byte aligned.
template <typename T>
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const T* tile, int ld, int m0, int k0) {
  const int l = threadIdx.x & 31;
  ldsm_x4(a, tile + (m0 + (l & 7) + ((l >> 3) & 1) * 8) * ld + k0 + (l >> 4) * 8);
}

/// The same A fragment from the transposed tile tile[k * ld + m] (m contiguous).
template <typename T>
__device__ __forceinline__ void frag_a_trans(uint32_t (&a)[4], const T* tile, int ld, int m0, int k0) {
  const int l = threadIdx.x & 31;
  ldsm_x4_trans(a, tile + (k0 + (l & 7) + (l >> 4) * 8) * ld + m0 + ((l >> 3) & 1) * 8);
}

/// The B fragments of two adjacent n8 tiles (b[0], b[1] for columns n0..n0 + 7, b[2], b[3] for
/// n0 + 8..) at k0 from tile[n * ld + k] (k contiguous).
template <typename T>
__device__ __forceinline__ void frag_b2(uint32_t (&b)[4], const T* tile, int ld, int n0, int k0) {
  const int l = threadIdx.x & 31;
  ldsm_x4(b, tile + (n0 + (l & 7) + (l >> 4) * 8) * ld + k0 + ((l >> 3) & 1) * 8);
}

/// The same two B fragments from tile[k * ld + n] (n contiguous).
template <typename T>
__device__ __forceinline__ void frag_b2_trans(uint32_t (&b)[4], const T* tile, int ld, int k0, int n0) {
  const int l = threadIdx.x & 31;
  ldsm_x4_trans(b, tile + (k0 + (l & 7) + ((l >> 3) & 1) * 8) * ld + n0 + (l >> 4) * 8);
}

// ---- staging ------------------------------------------------------------------------------------

/// Eight consecutive elements at `p` as fp32; `vec`: 16-bit input with a 16-byte aligned `p`.
template <typename T>
__device__ __forceinline__ void tc_load8(float (&v)[8], const T* p, bool vec) {
  if constexpr (sizeof(T) == 2) {
    if (vec) {
      const uint4 raw = *reinterpret_cast<const uint4*>(p);
      const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float pair[2];
        Bits16<T>::unpack(pair, w[i]);
        v[2 * i] = pair[0];
        v[2 * i + 1] = pair[1];
      }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = ld16(p[i]);
}

/// Eight consecutive fp32 at `p` (16-byte aligned when `vec`).
__device__ __forceinline__ void tc_load8_f32(float (&v)[8], const float* p, bool vec) {
  if (vec) {
    const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = p[i];
  }
}

/// dst[r * ld + c] = T(src[(t0 + r) * stride + c]) for the m valid rows r < m and c < limit (zero past
/// `limit`); the rows r in [m, 64) are zeroed. kThreads threads, eight columns per piece (a 16-byte
/// copy when `vec`: 16-bit input, 16-byte aligned base and strides). cols is a multiple of 8 and the
/// valid columns are [0, limit), limit <= cols (limit < 0: cols).
template <typename T, int kThreads>
__device__ __forceinline__ void tc_stage_input(T* dst, int ld, const T* src, int64_t stride, int t0, int m, int cols,
                                               bool vec, int limit = -1) {
  if (limit < 0) limit = cols;
  const int pieces = cols / 8;
  for (int idx = threadIdx.x; idx < 64 * pieces; idx += kThreads) {
    const int r = idx / pieces, c = (idx % pieces) * 8;
    uint4 out;
    out.x = 0, out.y = 0, out.z = 0, out.w = 0;
    if (r < m) {
      const T* row = src + int64_t(t0 + r) * stride + c;
      if constexpr (sizeof(T) == 2) {
        if (vec && c + 8 <= limit) {
          out = *reinterpret_cast<const uint4*>(row);
        } else {
          float v[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) v[i] = c + i < limit ? ld16(row[i]) : 0.0f;
          out.x = Bits16<T>::pack(v[0], v[1]);
          out.y = Bits16<T>::pack(v[2], v[3]);
          out.z = Bits16<T>::pack(v[4], v[5]);
          out.w = Bits16<T>::pack(v[6], v[7]);
        }
      } else {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = c + i < limit ? ld16(row[i]) : 0.0f;
        out.x = Bits16<T>::pack(v[0], v[1]);
        out.y = Bits16<T>::pack(v[2], v[3]);
        out.z = Bits16<T>::pack(v[4], v[5]);
        out.w = Bits16<T>::pack(v[6], v[7]);
      }
    }
    *reinterpret_cast<uint4*>(dst + r * ld + c) = out;
  }
}

/// dst[r * ld + c] = T(src[r * stride + c]) for r < rows, c < cols (the columns past `limit` and the
/// rows past `rows` are zero) from an fp32 workspace buffer; `vec`: stride and base are 16-byte
/// aligned and limit is a multiple of 4. cols is a multiple of 8.
template <typename T, int kThreads>
__device__ __forceinline__ void tc_stage_f32(T* dst, int ld, const float* src, int64_t stride, int rows, int nrows,
                                             int cols, int limit, bool vec) {
  const int pieces = cols / 8;
  for (int idx = threadIdx.x; idx < nrows * pieces; idx += kThreads) {
    const int r = idx / pieces, c = (idx % pieces) * 8;
    float v[8];
    if (r < rows) {
      const float* row = src + int64_t(r) * stride + c;
      if (vec && c + 8 <= limit) {
        const float4 a = *reinterpret_cast<const float4*>(row), b = *reinterpret_cast<const float4*>(row + 4);
        v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = c + i < limit ? row[i] : 0.0f;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = 0.0f;
    }
    uint4 out;
    out.x = Bits16<T>::pack(v[0], v[1]);
    out.y = Bits16<T>::pack(v[2], v[3]);
    out.z = Bits16<T>::pack(v[4], v[5]);
    out.w = Bits16<T>::pack(v[6], v[7]);
    *reinterpret_cast<uint4*>(dst + r * ld + c) = out;
  }
}

}  // namespace mma
}  // namespace kops::state
