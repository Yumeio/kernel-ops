#pragma once
// =============================================================================
// kernel-ops | csrc/common/device_impl.cuh
// Device bodies for the primitives declared in common/device.cuh, plus the small
// scalar/vector helpers shared by the B0/B1 kernel groups.
//
// Requires: common/macros.cuh, common/dtype.cuh, common/op_types.cuh,
//           <cuda_runtime.h>, <cuda_fp16.h>, <cuda_bf16.h>.
// =============================================================================

namespace kops {
namespace dev {

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T x[N];
};

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ void from_f(T& d, float s);
template <>
__device__ __forceinline__ void from_f<float>(float& d, float s) { d = s; }
template <>
__device__ __forceinline__ void from_f<__half>(__half& d, float s) { d = __float2half_rn(s); }
template <>
__device__ __forceinline__ void from_f<__nv_bfloat16>(__nv_bfloat16& d, float s) {
  d = __float2bfloat16_rn(s);
}

template <int N, typename T>
__device__ __forceinline__ Pack<T, N> load_pack(const T* __restrict__ p) {
  return *reinterpret_cast<const Pack<T, N>*>(p);
}

template <int N, typename T>
__device__ __forceinline__ void store_pack(T* __restrict__ p, const Pack<T, N>& v) {
  *reinterpret_cast<Pack<T, N>*>(p) = v;
}

/// fp32 -> e4m3fn bits, round to nearest even, saturating to +-448 (the result of
/// __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)).  sm_86 has no FP8 convert
/// instruction, so the toolkit intrinsic expands to ~40 integer instructions; this form is
/// about a quarter of that.  tests/unit/test_gemm.cc compares it with the intrinsic on all
/// 2^32 fp32 bit patterns.
__device__ __forceinline__ uint8_t e4m3_from_float_sat(float x) {
  const uint32_t u = __float_as_uint(x);
  const uint32_t sign = (u >> 24) & 0x80u;
  const uint32_t a = u & 0x7FFFFFFFu;
  uint32_t code;
  if (a > 0x7F800000u) {
    return 0x7Fu;                                   // NaN has no sign
  } else if (a >= 0x43E80000u) {
    code = 0x7Eu;                                   // >= 464 (and +-inf): saturate to 448
  } else if (a < 0x3C800000u) {
    code = static_cast<uint32_t>(__float2int_rn(__uint_as_float(a) * 512.0f));  // subnormal grid 2^-9
  } else {
    const uint32_t r = a + 0x7FFFFu + ((a >> 20) & 1u);  // keep 3 mantissa bits, ties to even
    code = (r >> 20) - 960u;                        // re-bias the exponent 127 -> 7
  }
  return static_cast<uint8_t>(sign | code);
}

/// e4m3fn bits -> fp32 (exact).  Normal codes rebias the exponent field in place; the
/// subnormals are m * 2^-9.  Both NaN codes (0x7F, 0xFF) decode to NaN.
__device__ __forceinline__ float e4m3_to_float_fast(uint8_t b) {
  const uint32_t m = b & 0x7Fu;
  float v = m < 8u ? __uint2float_rn(m) * (1.0f / 512.0f) : __uint_as_float((m << 20) + (120u << 23));
  if (m == 0x7Fu) v = __uint_as_float(0x7FC00000u);
  return (b & 0x80u) ? -v : v;
}

}  // namespace dev

// ---- Vector primitives (load_vec / store_vec are in vec.cuh) ----------------------------
template <typename T, int N>
KOPS_DEVICE Vec<T, N> load_vec_streaming(const T* __restrict__ ptr) {
  return *reinterpret_cast<const Vec<T, N>*>(ptr);
}

// ---- Fast math -------------------------------------------------------------
KOPS_DEVICE float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
KOPS_DEVICE float fast_log2(float x) {
  float y;
  asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
KOPS_DEVICE float fast_rcp(float x) { return __frcp_rn(x); }
KOPS_DEVICE float fast_rsqrt(float x) { return rsqrtf(x); }
KOPS_DEVICE float fast_tanh(float x) { return tanhf(x); }

KOPS_DEVICE float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
KOPS_DEVICE float softplus(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

template <ActFn kAct>
KOPS_DEVICE float apply_act(float x, const GluParams& p) {
  switch (kAct) {
    case ActFn::kIdentity: return x;
    case ActFn::kSiLU: return x * sigmoid(p.gate_alpha * x);
    case ActFn::kGeLUErf: return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    case ActFn::kGeLUTanh:
      return 0.5f * x *
             (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    case ActFn::kQuickGeLU: return x * sigmoid(1.702f * x);
    case ActFn::kReLU: return x > 0.0f ? x : 0.0f;
    case ActFn::kReLU2: return x > 0.0f ? x * x : 0.0f;
    case ActFn::kSigmoid: return sigmoid(x);
    case ActFn::kTanh: return tanhf(x);
    case ActFn::kSoftplus: return softplus(x);
    case ActFn::kSiTU:
      return p.situ_beta * tanhf(x / p.situ_beta) * sigmoid(x);
    case ActFn::kFatReLU: return x > p.fatrelu_threshold ? x : 0.0f;
  }
  return x;
}

template <ActFn kAct>
KOPS_DEVICE float apply_glu(float gate, float up, const GluParams& p) {
  if (p.gate_clamp_max > 0.0f && gate > p.gate_clamp_max) gate = p.gate_clamp_max;
  if (p.up_clamp > 0.0f) {
    up = up > p.up_clamp ? p.up_clamp : up;
    up = up < -p.up_clamp ? -p.up_clamp : up;
  }
  up += p.up_offset;
  if (p.situ_linear_beta > 0.0f) up = p.situ_linear_beta * tanhf(up / p.situ_linear_beta);
  return apply_act<kAct>(gate, p) * up;
}

KOPS_DEVICE float softcap(float x, float cap) {
  return cap > 0.0f ? cap * tanhf(x / cap) : x;
}

// ---- Reductions ------------------------------------------------------------
template <typename T>
KOPS_DEVICE T warp_reduce_sum(T v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return __shfl_sync(0xffffffffu, v, 0);
}

template <typename T>
KOPS_DEVICE T warp_reduce_max(T v) {
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return __shfl_sync(0xffffffffu, v, 0);
}

template <typename T>
KOPS_DEVICE T warp_reduce_min(T v) {
  for (int off = 16; off > 0; off >>= 1) v = fminf(v, __shfl_down_sync(0xffffffffu, v, off));
  return __shfl_sync(0xffffffffu, v, 0);
}

template <typename T, int kWidth>
KOPS_DEVICE T group_reduce_sum(T v) {
  for (int off = kWidth / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off, kWidth);
  return __shfl_sync(0xffffffffu, v, 0, kWidth);
}

template <typename T, int NUM_WARPS>
KOPS_DEVICE T block_reduce_sum(T v, T* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    T t = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : T{};
    t = warp_reduce_sum(t);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

template <typename T, int NUM_WARPS>
KOPS_DEVICE T block_reduce_max(T v, T* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    T t = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : __int_as_float(0xff800000);
    t = warp_reduce_max(t);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  return smem[0];
}

/// (value, index) argmax with deterministic tie-break to the lowest index.
template <typename T>
KOPS_DEVICE void warp_argmax(T& v, int32_t& idx) {
  for (int off = 16; off > 0; off >>= 1) {
    const T ov = __shfl_down_sync(0xffffffffu, v, off);
    const int32_t oi = __shfl_down_sync(0xffffffffu, idx, off);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
}

// ---- Online softmax --------------------------------------------------------
KOPS_DEVICE void softmax_state_init(SoftmaxState& s) {
  s.m = __int_as_float(0xff800000);
  s.d = 0.0f;
}

KOPS_DEVICE void softmax_state_update(SoftmaxState& s, float logit) {
  const float m = fmaxf(s.m, logit);
  s.d = s.d * __expf(s.m - m) + __expf(logit - m);
  s.m = m;
}

KOPS_DEVICE void softmax_state_merge(SoftmaxState& a, const SoftmaxState& b) {
  const float m = fmaxf(a.m, b.m);
  a.d = a.d * __expf(a.m - m) + b.d * __expf(b.m - m);
  a.m = m;
}

// ---- B5: CUDA-core GEMV primitives ----------------------------------------
namespace dev {

/// Weights are read exactly once per launch: load 16 B / 4 B without allocating in L1.
__device__ __forceinline__ uint4 ld_stream16(const void* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t ld_stream4(const void* p) {
  uint32_t v;
  asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

/// Exact float(q) for 0 <= q < 2^23 (two instructions, no int->float convert).
__device__ __forceinline__ float u23_to_f(uint32_t q) {
  return __uint_as_float(0x4B000000u | q) - 8388608.0f;
}

/// Two 16-bit values packed in one word -> two floats (exact).
template <typename T>
__device__ __forceinline__ float2 unpack2(uint32_t w);
template <>
__device__ __forceinline__ float2 unpack2<__nv_bfloat16>(uint32_t w) {
  return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u));
}
template <>
__device__ __forceinline__ float2 unpack2<__half>(uint32_t w) {
  __half2 h;
  *reinterpret_cast<uint32_t*>(&h) = w;
  return __half22float2(h);
}

/// 16 B = eight 16-bit activations -> eight floats.
template <typename T>
__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  float2 t = unpack2<T>(v.x); f[0] = t.x; f[1] = t.y;
  t = unpack2<T>(v.y); f[2] = t.x; f[3] = t.y;
  t = unpack2<T>(v.z); f[4] = t.x; f[5] = t.y;
  t = unpack2<T>(v.w); f[6] = t.x; f[7] = t.y;
}

/// e4m3fn byte (low 8 bits of b) -> fp32, exact and branch free.  The seven magnitude bits
/// placed under the fp32 exponent give value * 2^-120 for normals and subnormals alike; the
/// scaling by 2^120 is exact.  0x7F / 0xFF are the NaN codes.
__device__ __forceinline__ float e4m3_decode_finite(uint32_t b) {
  const float v = __uint_as_float((b & 0x7Fu) << 20) * 0x1p120f;
  return __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
}

__device__ __forceinline__ float e4m3_decode(uint32_t b) {
  return (b & 0x7Fu) == 0x7Fu ? __uint_as_float(0x7FC00000u) : e4m3_decode_finite(b);
}

/// True when one of the four e4m3 bytes of w is a NaN code (0x7F / 0xFF).
__device__ __forceinline__ bool e4m3x4_has_nan(uint32_t w) {
  const uint32_t x = (w & 0x7F7F7F7Fu) ^ 0x7F7F7F7Fu;
  return ((x - 0x01010101u) & ~x & 0x80808080u) != 0u;
}


/// Two adjacent e4m3 bytes (low 16 bits of w) -> two floats.  Sm89+ uses the hardware
/// convert; the software form is bit-identical (tests/unit/test_gemm.cc compares the
/// decoder with the toolkit on all 256 codes, and the sm89 handoff kit runs the cvt path).
__device__ __forceinline__ float2 e4m3x2_decode(uint32_t w) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint32_t h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(static_cast<unsigned short>(w)));
  __half2 hh;
  *reinterpret_cast<uint32_t*>(&hh) = h;
  return __half22float2(hh);
#else
  return make_float2(e4m3_decode(w), e4m3_decode(w >> 8));
#endif
}

/// Four packed e4m3 bytes -> four floats, NaN-exact.  Sm89+ converts in hardware; elsewhere the
/// finite-only decoder runs unless a NaN code is present (rare, taken as a whole-word detour).
__device__ __forceinline__ void e4m3x4_decode(uint32_t w, float (&o)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const float2 lo = e4m3x2_decode(w);
  const float2 hi = e4m3x2_decode(w >> 16);
  o[0] = lo.x; o[1] = lo.y; o[2] = hi.x; o[3] = hi.y;
#else
  if (e4m3x4_has_nan(w)) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = e4m3_decode(w >> (8 * i));
  } else {
    #pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = e4m3_decode_finite(w >> (8 * i));
  }
#endif
}

/// e2m1 code (low 4 bits of c) -> fp32, exact, no table: code bits (E1 E0 M) placed under the
/// fp32 exponent give value * 2^-126; the sign goes straight to bit 31.
__device__ __forceinline__ float e2m1_decode(uint32_t c) {
  const float v = __uint_as_float((c & 7u) << 22) * 0x1p126f;
  return __uint_as_float(__float_as_uint(v) | ((c & 8u) << 28));
}

/// ActFn chosen at run time (the epilogue is not a template parameter of the GEMV kernels).
__device__ __forceinline__ float act_runtime(ActFn fn, float x, const GluParams& p) {
  switch (fn) {
    case ActFn::kSiLU: return apply_act<ActFn::kSiLU>(x, p);
    case ActFn::kGeLUErf: return apply_act<ActFn::kGeLUErf>(x, p);
    case ActFn::kGeLUTanh: return apply_act<ActFn::kGeLUTanh>(x, p);
    case ActFn::kQuickGeLU: return apply_act<ActFn::kQuickGeLU>(x, p);
    case ActFn::kReLU: return apply_act<ActFn::kReLU>(x, p);
    case ActFn::kReLU2: return apply_act<ActFn::kReLU2>(x, p);
    case ActFn::kSigmoid: return apply_act<ActFn::kSigmoid>(x, p);
    case ActFn::kTanh: return apply_act<ActFn::kTanh>(x, p);
    case ActFn::kSoftplus: return apply_act<ActFn::kSoftplus>(x, p);
    case ActFn::kSiTU: return apply_act<ActFn::kSiTU>(x, p);
    case ActFn::kFatReLU: return apply_act<ActFn::kFatReLU>(x, p);
    default: return x;
  }
}

__device__ __forceinline__ float load_as_float(const void* p, DType t, int64_t i) {
  switch (t) {
    case DType::kFloat32: return static_cast<const float*>(p)[i];
    case DType::kFloat16: return __half2float(static_cast<const __half*>(p)[i]);
    default: return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
  }
}

/// Scalar GEMM epilogue of the CUDA-core kernels, applied to one fp32 accumulator:
///   *alpha -> *row_scale[m] -> *col_scale[n] -> +bias[n] -> clamp -> softcap -> act
///   -> +beta * residual[m, n].
/// EP is gemm::EpilogueParams (this header precedes gemm_params.cuh, so the EpilogueOp bits
/// are spelled out: RowScale 1<<0, ColScale 1<<1, Bias 1<<2, Softcap 1<<3, Act 1<<4,
/// ResidualAdd 1<<8, Clamp 1<<12).  The INT8 azp correction is applied by the INT8 kernels
/// in the integer domain before the accumulator reaches this function.  Residual is read in
/// the output dtype OutT.
template <typename OutT, typename EP>
__device__ __forceinline__ float gemm_epilogue(float acc, const EP& e, int64_t m, int64_t n) {
  const uint32_t ops = e.ops;
  acc *= e.alpha_ptr != nullptr ? *e.alpha_ptr : e.alpha;
  if (ops & (1u << 0)) acc *= e.row_scale[m];
  if (ops & (1u << 1)) acc *= e.col_scale[n];
  if (ops & (1u << 2)) acc += load_as_float(e.bias, e.bias_dtype, n);
  if (ops & (1u << 12)) acc = fminf(fmaxf(acc, -e.clamp_limit), e.clamp_limit);
  if (ops & (1u << 3)) acc = softcap(acc, e.softcap);
  if (ops & (1u << 4)) acc = act_runtime(e.act, acc, e.glu);
  if (ops & (1u << 8)) {
    acc = fmaf(e.beta, to_f(static_cast<const OutT*>(e.residual)[m * e.residual_stride + n]), acc);
  }
  return acc;
}

// ---- Marlin-tile GEMV on tensor cores (SM80+): weight = MMA operand A, tokens = operand B ----------
// Shared by the W4A16 integer body (gemm/mixed_input_gemm.cu) and the MXFP4 / NVFP4 body
// (gemm/fp4_gemm.cu); the arch guard is inside every asm, the hosts refuse arch < 80.
//
// quantization::{awq,fp4}_to_marlin write, per 16(k) x 64(n) tile, 128 words = 4 sub-tiles of 16
// output columns.  Lane (r = lane / 4, c = lane % 4) owns one uint4 whose word w is sub-tile w and
// whose nibbles {0,4} {1,5} {2,6} {3,7} are the pairs (k0,k1 @ n=r) (k8,k9 @ n=r) (k0,k1 @ n=r+8)
// (k8,k9 @ n=r+8), with k0 = 2c and n relative to 16 w.  Those are the A fragment registers
// {a0,a2,a1,a3} of mma.m16n8k16 (row r / r+8, k 2c.. / 2c+8..), so a word dequantises straight into
// A, and the mma rows are output columns: rows r / r+8 of sub-tile w = columns 16 w + r / 16 w + r + 8.
//
// Numerics.  The dequantised weight is exact in bf16/fp16 (integer q - z, or an e2m1 code); the
// products with the activations are exact in fp32; the block scale is applied in fp32 to the
// block's accumulator.  Blocks are added in a fixed order per warp and the 8 warps in a fixed order,
// so every output element depends only on its own token and weight column: a row alone and the same
// row inside a batch are bit-identical, but the result is not bit-equal to a CUDA-core body.

constexpr int kTcWarps = 8;      // warps splitting K inside one CTA (256 threads)
constexpr int kTcTileN = 64;     // output columns of one CTA = one Marlin n-tile

template <typename T>
struct TcFrag;

template <>
struct TcFrag<__nv_bfloat16> {
  static constexpr uint32_t kMagic = 0x43004300u;   // bf16x2 128.0 | q: 128 + q is exact for q < 128
  static constexpr uint32_t kBase = 0x4300u;        // bf16 128.0
  /// a - b for small integers, exact.  fma.rn.bf16x2 (SM80); sub.bf16x2 needs SM90.
  __device__ static __forceinline__ uint32_t sub2(uint32_t a, uint32_t b) {
    uint32_t d = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(0x3F803F80u), "r"(b ^ 0x80008000u));
#endif
    return d;
  }
  /// Two e2m1 codes (the low nibble of each 16-bit half of x, nothing else) -> two bf16, exact: the
  /// magnitude bits (E1 E0 M) placed under the bf16 exponent give code * 2^-126, the sign goes to
  /// bit 15, and the multiply by 2^126 (0x7E80) restores the subnormal code 0.5 as well.
  __device__ static __forceinline__ uint32_t e2m1x2(uint32_t x) {
    const uint32_t y = ((x << 6) & 0x01C001C0u) | ((x << 12) & 0x80008000u);
    uint32_t d = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(y), "r"(0x7E807E80u), "r"(0u));
#endif
    return d;
  }
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
  }
};

template <>
struct TcFrag<__half> {
  static constexpr uint32_t kMagic = 0x64006400u;   // f16x2 1024.0 | q: 1024 + q is exact for q < 1024
  static constexpr uint32_t kBase = 0x6400u;        // f16 1024.0
  __device__ static __forceinline__ uint32_t sub2(uint32_t a, uint32_t b) {
    uint32_t d = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("sub.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
#endif
    return d;
  }
  /// As the bf16 form: (E1 E0 M) under the f16 exponent (bit 9 = mantissa MSB) gives code * 2^-14,
  /// times 2^14 (0x7400).
  __device__ static __forceinline__ uint32_t e2m1x2(uint32_t x) {
    const uint32_t y = ((x << 9) & 0x0E000E00u) | ((x << 12) & 0x80008000u);
    uint32_t d = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("mul.f16x2 %0, %1, %2;" : "=r"(d) : "r"(y), "r"(0x74007400u));
#endif
    return d;
  }
  __device__ static __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
  }
};

/// One CTA = one 64-column n-tile; its 8 warps take units of BPI consecutive scale blocks
/// (u = warp, warp + 8, ...), so every lane keeps TG * BPI >= 4 weight loads in flight.  M_TILE is
/// 8 or 16 tokens; tokens past m re-read row m - 1 and are never stored.
///
/// Policy (scale blocks of Policy::kBlockK elements along K, a multiple of 16):
///   State                       what one block needs per lane (raw scale words, zero points)
///   State load(blk, nt, r)      issues the scale / zero loads of block blk for lane group r
///   void unpack_scales(st, sf)  the eight fp32 scales j = 2 w + h: row r (h = 0) / r + 8 (h = 1) of sub-tile w
///   void dequant(st, w, q, fa)  weight word q of sub-tile w -> A registers fa[0..3] (bf16x2 / f16x2)
/// Epi: operator()(float sum, int token, int column) finishes and stores one output element.
template <typename Policy, typename Epi, typename T, int M_TILE>
__device__ __forceinline__ void tc_gemv_body(const T* __restrict__ a, const int64_t lda,
                                             const uint32_t* __restrict__ qweight, const Policy& pol,
                                             const Epi& epi, const int m, const int n, const int k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using Tc = TcFrag<T>;
  constexpr int TT = M_TILE / 8;                          // token tiles (MMA N = 8)
  constexpr int TG = Policy::kBlockK / 16;                // k16 tiles per scale block
  constexpr int BPI = TG >= 4 ? 1 : 4 / TG;               // scale blocks per K-loop iteration
  static_assert(Policy::kBlockK % 16 == 0 && (TG == 1 || TG == 2 || TG == 4 || TG == 8), "block is 16, 32, 64 or 128");
  __shared__ float red[kTcWarps][TT * 16][32];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int r = lane >> 2;
  const int c = lane & 3;
  const int nt = blockIdx.x;
  const int nblocks = k / Policy::kBlockK;
  const int nunits = (nblocks + BPI - 1) / BPI;
  const int64_t kt_stride = static_cast<int64_t>(n) * 2;   // words per k16 row of tiles: [K/16, 2N]

  const T* arow[TT];
  #pragma unroll
  for (int t = 0; t < TT; ++t) arow[t] = a + static_cast<int64_t>(min(8 * t + r, m - 1)) * lda + 2 * c;

  float acc[TT][4][4];
  #pragma unroll
  for (int t = 0; t < TT; ++t) {
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc[t][w][i] = 0.0f;
    }
  }

  for (int u = warp; u < nunits; u += kTcWarps) {
    uint4 wv[BPI][TG];
    typename Policy::State st[BPI];
    #pragma unroll
    for (int b = 0; b < BPI; ++b) {
      const int blk = u * BPI + b;
      if (blk < nblocks) {
        #pragma unroll
        for (int i = 0; i < TG; ++i) wv[b][i] = ld_stream16(qweight + (blk * TG + i) * kt_stride + nt * 128 + lane * 4);
        st[b] = pol.load(blk, nt, r);
      }
    }
    #pragma unroll
    for (int b = 0; b < BPI; ++b) {
      const int blk = u * BPI + b;
      if (blk >= nblocks) break;
      float cg[TT][4][4];
      #pragma unroll
      for (int t = 0; t < TT; ++t) {
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
          #pragma unroll
          for (int i = 0; i < 4; ++i) cg[t][w][i] = 0.0f;
        }
      }
      #pragma unroll
      for (int i = 0; i < TG; ++i) {
        const int kk = (blk * TG + i) * 16;
        uint32_t ab[TT][2];
        #pragma unroll
        for (int t = 0; t < TT; ++t) {
          ab[t][0] = *reinterpret_cast<const uint32_t*>(arow[t] + kk);
          ab[t][1] = *reinterpret_cast<const uint32_t*>(arow[t] + kk + 8);
        }
        const uint32_t words[4] = {wv[b][i].x, wv[b][i].y, wv[b][i].z, wv[b][i].w};
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
          uint32_t fa[4];
          pol.dequant(st[b], w, words[w], fa);
          #pragma unroll
          for (int t = 0; t < TT; ++t) Tc::mma(cg[t][w], fa, ab[t][0], ab[t][1]);
        }
      }
      float sf[8];
      pol.unpack_scales(st[b], sf);
      #pragma unroll
      for (int t = 0; t < TT; ++t) {
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
          acc[t][w][0] = fmaf(sf[2 * w], cg[t][w][0], acc[t][w][0]);
          acc[t][w][1] = fmaf(sf[2 * w], cg[t][w][1], acc[t][w][1]);
          acc[t][w][2] = fmaf(sf[2 * w + 1], cg[t][w][2], acc[t][w][2]);
          acc[t][w][3] = fmaf(sf[2 * w + 1], cg[t][w][3], acc[t][w][3]);
        }
      }
    }
  }

  // Fixed-order reduction over the 8 K slices (warp 0 first, then 1..7).
  #pragma unroll
  for (int t = 0; t < TT; ++t) {
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][t * 16 + w * 4 + i][lane] = acc[t][w][i];
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < TT * 512; idx += blockDim.x) {
    const int ln = idx & 31;
    const int slot = idx >> 5;                    // t * 16 + w * 4 + i
    float sum = red[0][slot][ln];
    #pragma unroll
    for (int wi = 1; wi < kTcWarps; ++wi) sum += red[wi][slot][ln];
    const int t = slot >> 4;
    const int w = (slot >> 2) & 3;
    const int i = slot & 3;
    const int col = nt * kTcTileN + 16 * w + (ln >> 2) + ((i >> 1) << 3);
    const int tok = 8 * t + 2 * (ln & 3) + (i & 1);
    if (tok < m) epi(sum, tok, col);
  }
#else
  // Unreachable: the hosts refuse mma.sync bodies below SM80 (kUnsupportedArch).
  (void)a; (void)lda; (void)qweight; (void)pol; (void)epi; (void)m; (void)n; (void)k;
  if (threadIdx.x == 0) __trap();
#endif
}

// ---- B9 recurrent helpers (state/conv1d.cu, ssm.cu, linear_attention.cu, delta_rule*.cu) ----
// Scalar gates and decays in fp32 with the fast exponential (relative error ~2^-21 plus 2^-23
// per unit of |x|, far below the bf16/fp16 rounding they feed); softplus keeps log1p accurate.

/// 1 / (1 + exp(-x)); 0 for x < -88 and 1 for large x, never NaN for finite x.
__device__ __forceinline__ float sigmoid_f(float x) { return __frcp_rn(1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x * sigmoid_f(x); }
/// torch softplus (threshold 20).
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

/// Deterministic xor-butterfly fp32 sum over the `Width` lanes of an aligned group; every lane
/// of the group gets the same value.
template <int Width = 32>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int off = Width / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off, Width);
  return v;
}

/// Register-tiled product of two k-major shared-memory operands (a row of either holds one k):
/// acc[r][c] += sum_{k < K} A[k * lda + ra + r] * B[k * ldb + cb + c], in increasing k, fp32 FMA.
/// ra, cb and both leading dimensions are multiples of 4 (of 2 for RN = 2): one 16- or 8-byte
/// shared load per piece of a row.
template <int RM, int RN>
__device__ __forceinline__ void smem_gemm_kmajor(float (&acc)[RM][RN], const float* __restrict__ A,
                                                 int lda, int ra, const float* __restrict__ B,
                                                 int ldb, int cb, int K) {
  static_assert(RM % 4 == 0 && (RN == 2 || RN % 4 == 0), "tile sides map onto 16/8-byte loads");
#pragma unroll 4
  for (int k = 0; k < K; ++k) {
    float a[RM], b[RN];
    const float* ap = A + k * lda + ra;
    const float* bp = B + k * ldb + cb;
#pragma unroll
    for (int i = 0; i < RM; i += 4) {
      const float4 v = *reinterpret_cast<const float4*>(ap + i);
      a[i] = v.x, a[i + 1] = v.y, a[i + 2] = v.z, a[i + 3] = v.w;
    }
    if constexpr (RN == 2) {
      const float2 v = *reinterpret_cast<const float2*>(bp);
      b[0] = v.x, b[1] = v.y;
    } else {
#pragma unroll
      for (int i = 0; i < RN; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(bp + i);
        b[i] = v.x, b[i + 1] = v.y, b[i + 2] = v.z, b[i + 3] = v.w;
      }
    }
#pragma unroll
    for (int r = 0; r < RM; ++r) {
#pragma unroll
      for (int c = 0; c < RN; ++c) acc[r][c] = __fmaf_rn(a[r], b[c], acc[r][c]);
    }
  }
}

}  // namespace dev

// ---- Counter RNG (declared in common/rng.cuh) -------------------------------
// Philox-4x32-10, the reference block PRNG. The counter is a pure function of
// (seed, request_id, position, draw_index, stream_id), so replaying a graph with
// the same inputs reproduces the same draws without host state.
KOPS_DEVICE uint4 philox4x32_10(uint2 key, uint4 ctr) {
  uint32_t k0 = key.x;
  uint32_t k1 = key.y;
  const uint32_t m0 = 0xD2511F53u;
  const uint32_t m1 = 0xCD9E8D57u;
  const uint32_t w0 = 0x9E3779B9u;
  const uint32_t w1 = 0xBB67AE85u;
  uint32_t c0 = ctr.x;
  uint32_t c1 = ctr.y;
  uint32_t c2 = ctr.z;
  uint32_t c3 = ctr.w;
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const uint64_t p0 = static_cast<uint64_t>(m0) * c0;
    const uint64_t p1 = static_cast<uint64_t>(m1) * c2;
    const uint32_t hi0 = static_cast<uint32_t>(p0 >> 32);
    const uint32_t lo0 = static_cast<uint32_t>(p0);
    const uint32_t hi1 = static_cast<uint32_t>(p1 >> 32);
    const uint32_t lo1 = static_cast<uint32_t>(p1);
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += w0;
    k1 += w1;
  }
  return make_uint4(c0, c1, c2, c3);
}

KOPS_DEVICE float uniform_from_counter(uint64_t seed, uint64_t request_id, uint64_t position,
                                       uint32_t draw_index, uint32_t stream_id) {
  const uint2 key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
  const uint4 ctr = make_uint4(static_cast<uint32_t>(request_id),
                               static_cast<uint32_t>(request_id >> 32) ^ stream_id,
                               static_cast<uint32_t>(position),
                               static_cast<uint32_t>(position >> 32) ^ draw_index);
  const uint4 r = philox4x32_10(key, ctr);
  // 24 random bits plus one maps to (0, 1] with a resolution of 2^-24.
  return static_cast<float>((r.x >> 8) + 1u) * (1.0f / 16777216.0f);
}

KOPS_DEVICE float gumbel_from_counter(uint64_t seed, uint64_t request_id, uint64_t position,
                                      uint32_t draw_index, uint32_t stream_id) {
  const float u = uniform_from_counter(seed, request_id, position, draw_index, stream_id);
  return -logf(-logf(u));
}

KOPS_DEVICE float exponential_from_counter(uint64_t seed, uint64_t request_id, uint64_t position,
                                           uint32_t draw_index, uint32_t stream_id) {
  return -logf(uniform_from_counter(seed, request_id, position, draw_index, stream_id));
}

}  // namespace kops
