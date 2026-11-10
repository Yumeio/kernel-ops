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

// ---- Marlin GEMM on tensor cores (SM80+), B7 row 6 ----------------------------------------------
// Shared by gemm/marlin_dense.cu (marlin_gemm_kernel) and gemm/marlin_moe.cu (marlin_moe_gemm_kernel, the
// sorted_token_ids mode). Same tile layout and fragment mapping as tc_gemv_body: a lane's 4 (4-bit) or 8
// (8-bit) words of a 16(k) x 64(n) Marlin tile dequantise straight into mma A fragments (weight rows =
// output columns); the tokens are the mma B operand (ldmatrix from a swizzled shared tile). A CTA is WM x WN
// x WK warps: WM token groups of TT * 8 tokens, WN Marlin n-tiles (64 columns), WK warps splitting the k16
// tiles of every 64-K pipeline stage (summed in warp order at the end). Group scales are applied in fp32 to
// the group's partial accumulator, as in tc_gemv_body (exact weights x activations in the MMA, any scale
// exponent); per-channel scales and the NVFP4 global scale in the epilogue. WT: 0 kU4 (+ zeros), 1 kU4B8,
// 2 kU8B128, 3 kFE4M3, 4 kFE2M1 (quant::MarlinWeightType). P is gemm::MarlinParams.

constexpr int kMarlinStageK = 64;   // K elements per pipeline stage (four k16 tiles)
constexpr int kMarlinStages = 4;

template <typename T, int WT, int TT, int WM, int WN, int WK>
struct MarlinCfg {
  static constexpr int kBits = (WT == 2 || WT == 3) ? 8 : 4;
  static constexpr int kTileBytes = kBits == 4 ? 512 : 1024;   // one 16 x 64 tile
  static constexpr int kLaneBytes = kTileBytes / 32;
  static constexpr int kTok = TT * 8 * WM;                       // tokens per CTA
  static constexpr int kCols = 64 * WN;                          // columns per CTA
  static constexpr int kAStage = kTok * kMarlinStageK * 2;       // bytes
  static constexpr int kBStage = 4 * WN * kTileBytes;
  static constexpr int kPipe = kMarlinStages * (kAStage + kBStage);
  static constexpr int kRed = WK > 1 ? WK * WM * WN * 32 * 16 * TT * 4 : 0;   // k-split partial sums
  static constexpr int kSmem = (kPipe > kRed ? kPipe : kRed) + 8 * kTok;     // + row tables
  static constexpr int kThreads = 32 * WM * WN * WK;
  static_assert(WM * WN * WK == 4 && (TT == 2 || TT == 4), "four warps; token tiles in pairs");
};

__device__ __forceinline__ uint32_t marlin_smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void marlin_cp16(uint32_t dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
#endif
}

template <typename T>
__device__ __forceinline__ uint32_t marlin_pack2(float lo, float hi) {
  T a, b;
  from_f(a, lo);
  from_f(b, hi);
  uint16_t ua, ub;
  __builtin_memcpy(&ua, &a, 2);
  __builtin_memcpy(&ub, &b, 2);
  return static_cast<uint32_t>(ua) | (static_cast<uint32_t>(ub) << 16);
}

/// 8-bit weight byte -> fp32 (exact): WT 2 = uint8 with bias 128, WT 3 = e4m3.
template <int WT>
__device__ __forceinline__ float marlin_byte(uint32_t b) {
  if constexpr (WT == 2) return static_cast<float>(static_cast<int>(b & 0xFFu) - 128);
  else return e4m3_decode(b & 0xFFu);
}

/// Per-channel scale permutation of the Marlin repack (32-column chunks): column c sits at position
/// 8 ((c >> 1) & 3) + 2 (c >> 3) + (c & 1).
__device__ __forceinline__ int marlin_channel_pos(int n) {
  const int c = n & 31;
  return (n - c) + 8 * ((c >> 1) & 3) + 2 * (c >> 3) + (c & 1);
}

template <typename T, int WT, int TT, int WM, int WN, int WK, bool MOE, typename P>
__device__ __forceinline__ void marlin_body(const P& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using Cfg = MarlinCfg<T, WT, TT, WM, WN, WK>;
  using Tc = TcFrag<T>;
  constexpr int kTok = Cfg::kTok, kCols = Cfg::kCols, kThreads = Cfg::kThreads;
  constexpr int kTileWords = Cfg::kTileBytes / 4;
  extern __shared__ __align__(128) unsigned char marlin_smem[];
  unsigned char* const sA = marlin_smem;
  unsigned char* const sB = marlin_smem + kMarlinStages * Cfg::kAStage;
  int32_t* const s_arow = reinterpret_cast<int32_t*>(marlin_smem + (Cfg::kPipe > Cfg::kRed ? Cfg::kPipe : Cfg::kRed));
  int32_t* const s_crow = s_arow + kTok;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wn = warp % WN;
  const int wm = (warp / WN) % WM;
  const int wk = warp / (WN * WM);
  const int g = lane >> 2, c4 = lane & 3;
  const int N = p.prob_n, K = p.prob_k;
  const int n_tiles = N / 64;
  const int tiles_n = (N + kCols - 1) / kCols;
  const int rows = MOE ? *p.num_tokens_post_pad : p.prob_m;
  const int tiles_m = (rows + kTok - 1) / kTok;
  const int kt16 = K / 16;                                     // k16 tiles (K % 16 == 0)
  const int nst = (K + kMarlinStageK - 1) / kMarlinStageK;     // last stage may be partial
  const bool channelwise = WT != 4 && (p.group_size < 0 || p.num_groups <= 1);
  const bool has_zeros = WT == 0 && p.zeros != nullptr;
  const bool act_order = p.g_idx != nullptr;
  const int64_t w_expert = static_cast<int64_t>(K) * N * Cfg::kBits / 32;   // words per expert
  const T* const A = static_cast<const T*>(p.a);
  // Global K split (dense only, c_tmp + locks given): the host launches exactly tiles x splits CTAs, slice s
  // of a tile takes stages [s nst / splits, (s + 1) nst / splits) and the slices are summed in slice order
  // through the fp32 c_tmp under the tile's lock (serial, so deterministic; the last slice runs the epilogue
  // and re-arms the lock to 0).
  const int tiles = tiles_m * tiles_n;
  const int splits = (!MOE && p.c_tmp != nullptr && p.locks != nullptr && static_cast<int>(gridDim.x) >= 2 * tiles)
                         ? static_cast<int>(gridDim.x) / tiles : 1;

  for (int item = blockIdx.x; item < tiles * splits; item += gridDim.x) {
    const int tile = item / splits;
    const int slice = item - tile * splits;
    const int s_begin = static_cast<int>(static_cast<int64_t>(slice) * nst / splits);
    const int s_count = static_cast<int>(static_cast<int64_t>(slice + 1) * nst / splits) - s_begin;
    const int tm = tile % tiles_m;
    const int tn = tile / tiles_m;
    const int r0 = tm * kTok;
    const int c0 = tn * kCols;
    const int expert = MOE ? p.expert_ids[r0 / p.moe_block_size] : 0;
    if (expert < 0) continue;   // uniform: no shared state touched yet
    for (int i = tid; i < kTok; i += kThreads) {
      int ar = -1, cr = -1;
      if (MOE) {
        const int id = r0 + i < rows ? p.sorted_token_ids[r0 + i] : -1;
        if (id >= 0 && id < p.prob_m * p.top_k) {
          ar = id / p.top_k;
          cr = id;
        }
      } else if (r0 + i < rows) {
        ar = cr = r0 + i;
      }
      s_arow[i] = ar;
      s_crow[i] = cr;
    }
    __syncthreads();

    const uint32_t* const Bw = p.b + expert * w_expert;
    const int64_t scale_off = static_cast<int64_t>(expert) * p.num_groups * N;
    const uint32_t* const Zw = has_zeros ? static_cast<const uint32_t*>(p.zeros) + static_cast<int64_t>(expert) * p.num_groups * (N / 8) : nullptr;
    auto load_stage = [&](int st, int s) {
      const int k0 = s * kMarlinStageK;
      unsigned char* a_st = sA + st * Cfg::kAStage;
      for (int c = tid; c < kTok * 8; c += kThreads) {
        const int r = c >> 3, ch = c & 7;
        const int ar = s_arow[r];
        const bool ok = ar >= 0 && k0 + ch * 8 < K;
        const T* src = ok ? A + static_cast<int64_t>(ar) * p.lda + k0 + ch * 8 : A;
        marlin_cp16(marlin_smem_u32(a_st + r * 128 + ((ch ^ (r & 7)) << 4)), src, ok);
      }
      unsigned char* b_st = sB + st * Cfg::kBStage;
      constexpr int kChunks = Cfg::kTileBytes / 16;
      for (int c = tid; c < 4 * WN * kChunks; c += kThreads) {
        const int tl = c / kChunks, ch = c % kChunks;
        const int j = tl / WN, w = tl % WN;
        const int nt = c0 / 64 + w;
        const bool ok = nt < n_tiles && k0 / 16 + j < kt16;
        const uint32_t* src = ok ? Bw + (static_cast<int64_t>(k0 / 16 + j) * n_tiles + nt) * kTileWords + ch * 4 : Bw;
        marlin_cp16(marlin_smem_u32(b_st + tl * Cfg::kTileBytes + ch * 16), src, ok);
      }
      asm volatile("cp.async.commit_group;");
    };

    float acc[4][TT][4];
    float pacc[4][TT][4];
    #pragma unroll
    for (int w = 0; w < 4; ++w)
      #pragma unroll
      for (int t = 0; t < TT; ++t)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[w][t][i] = pacc[w][t][i] = 0.0f;
    float sf[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) sf[j] = 1.0f;
    uint32_t zlo[4], zhi[4];
    #pragma unroll
    for (int w = 0; w < 4; ++w) zlo[w] = zhi[w] = (Tc::kBase | (WT == 0 ? 0u : 8u)) * 0x00010001u;
    int cur_g = -1;
    const int nt_w = c0 / 64 + wn;           // this warp's Marlin n-tile
    const bool w_live = nt_w < n_tiles;

    auto promote = [&]() {
      #pragma unroll
      for (int w = 0; w < 4; ++w)
        #pragma unroll
        for (int t = 0; t < TT; ++t)
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            acc[w][t][i] = fmaf(sf[2 * w + (i >> 1)], pacc[w][t][i], acc[w][t][i]);
            pacc[w][t][i] = 0.0f;
          }
    };
    // Group scales / zeros: the raw words of the group after the current one are fetched into registers
    // when a group starts (groups are visited in increasing order, also with act-order's sorted g_idx), so
    // the switch normally finds them loaded; a different group (empty group in a K shard) loads directly.
    uint4 pf_s = make_uint4(0u, 0u, 0u, 0u);
    uint32_t pf_z = 0u;
    int pf_g = -1;
    auto fetch_group = [&](int grp, uint4& sv, uint32_t& zw) {
      const int64_t base = scale_off + static_cast<int64_t>(grp) * N + nt_w * 64 + 8 * g;
      if constexpr (WT == 4) {
        const uint2 v = __ldg(reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(p.scales) + base));
        sv = make_uint4(v.x, v.y, 0u, 0u);
      } else {
        sv = __ldg(reinterpret_cast<const uint4*>(static_cast<const T*>(p.scales) + base));
      }
      if (has_zeros) zw = __ldg(Zw + static_cast<int64_t>(grp) * (N / 8) + nt_w * 8 + g);
    };
    auto load_group = [&](int grp) {
      if (!w_live) return;
      uint4 sv;
      uint32_t zw = 0u;
      if (grp == pf_g) {
        sv = pf_s;
        zw = pf_z;
      } else {
        fetch_group(grp, sv, zw);
      }
      if (!channelwise && grp + 1 < p.num_groups) {
        fetch_group(grp + 1, pf_s, pf_z);
        pf_g = grp + 1;
      }
      if constexpr (WT == 4) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t byte = ((j < 4 ? sv.x : sv.y) >> (8 * (j & 3))) & 0xFFu;
          sf[j] = p.group_size == 16 ? e4m3_decode(byte) : (byte == 0xFFu ? __int_as_float(0x7FC00000) : __uint_as_float(byte << 23));
        }
      } else {
        unpack8<T>(sv, sf);
      }
      if (has_zeros) {
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
          zlo[w] = (Tc::kBase | ((zw >> (4 * w)) & 0xFu)) * 0x00010001u;
          zhi[w] = (Tc::kBase | ((zw >> (4 * (w + 4))) & 0xFu)) * 0x00010001u;
        }
      }
    };
    if (channelwise && has_zeros) load_group(0);
    // act-order: group of the warp's next k16 tile, loaded one tile ahead (g_idx[perm[k]])
    auto act_group = [&](int kt) { return p.g_idx[p.perm != nullptr ? p.perm[kt * 16] : kt * 16]; };
    int next_ag = (act_order && s_count > 0 && (s_begin * 4 + wk) < kt16) ? act_group(s_begin * 4 + wk) : 0;

    #pragma unroll
    for (int i = 0; i < kMarlinStages - 1; ++i) {
      if (i < s_count) load_stage(i, s_begin + i);
      else asm volatile("cp.async.commit_group;");
    }
    for (int i = 0; i < s_count; ++i) {
      const int s = s_begin + i;
      asm volatile("cp.async.wait_group %0;" ::"n"(kMarlinStages - 2));
      __syncthreads();
      const int nxt = i + kMarlinStages - 1;
      if (nxt < s_count) load_stage(nxt % kMarlinStages, s_begin + nxt);
      else asm volatile("cp.async.commit_group;");
      const int st = i % kMarlinStages;
      const unsigned char* a_st = sA + st * Cfg::kAStage;
      const unsigned char* b_st = sB + st * Cfg::kBStage;
      #pragma unroll
      for (int jj = 0; jj < 4 / WK; ++jj) {
        const int j = wk + jj * WK;
        const int kt = s * 4 + j;
        if (kt >= kt16) break;   // partial last stage (warp-uniform)
        if (!channelwise) {
          int grp;
          if (act_order) {
            grp = next_ag;
            if (kt + WK < kt16) next_ag = act_group(kt + WK);
          } else {
            grp = (kt * 16) / p.group_size;
          }
          if (grp != cur_g) {
            if (cur_g >= 0) promote();
            load_group(grp);
            cur_g = grp;
          }
        }
        // token fragments (mma B): TT n8 tiles of this warp's tokens
        uint32_t bt[TT][2];
        #pragma unroll
        for (int np = 0; np < TT / 2; ++np) {
          const int r = wm * TT * 8 + np * 16 + (lane & 7) + (lane >> 4) * 8;
          const int ch = j * 2 + ((lane >> 3) & 1);
          uint32_t rr[4];
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                       : "=r"(rr[0]), "=r"(rr[1]), "=r"(rr[2]), "=r"(rr[3])
                       : "r"(marlin_smem_u32(a_st + r * 128 + ((ch ^ (r & 7)) << 4))));
          bt[2 * np][0] = rr[0];
          bt[2 * np][1] = rr[1];
          bt[2 * np + 1][0] = rr[2];
          bt[2 * np + 1][1] = rr[3];
        }
        // weight fragments (mma A) of the 4 sub-tiles
        const unsigned char* wp = b_st + (j * WN + wn) * Cfg::kTileBytes + lane * Cfg::kLaneBytes;
        uint32_t words[Cfg::kBits == 4 ? 4 : 8];
        {
          const uint4 v0 = *reinterpret_cast<const uint4*>(wp);
          words[0] = v0.x; words[1] = v0.y; words[2] = v0.z; words[3] = v0.w;
          if constexpr (Cfg::kBits == 8) {
            const uint4 v1 = *reinterpret_cast<const uint4*>(wp + 16);
            words[4] = v1.x; words[5] = v1.y; words[6] = v1.z; words[7] = v1.w;
          }
        }
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
          uint32_t fa[4];
          if constexpr (WT == 0 || WT == 1) {
            const uint32_t q = words[w];
            fa[0] = Tc::sub2((q & 0x000F000Fu) | Tc::kMagic, zlo[w]);
            fa[2] = Tc::sub2(((q >> 4) & 0x000F000Fu) | Tc::kMagic, zlo[w]);
            fa[1] = Tc::sub2(((q >> 8) & 0x000F000Fu) | Tc::kMagic, zhi[w]);
            fa[3] = Tc::sub2(((q >> 12) & 0x000F000Fu) | Tc::kMagic, zhi[w]);
          } else if constexpr (WT == 4) {
            const uint32_t q = words[w];
            fa[0] = Tc::e2m1x2(q & 0x000F000Fu);
            fa[2] = Tc::e2m1x2((q >> 4) & 0x000F000Fu);
            fa[1] = Tc::e2m1x2((q >> 8) & 0x000F000Fu);
            fa[3] = Tc::e2m1x2((q >> 12) & 0x000F000Fu);
          } else {
            // bytes of word 2w: (k0, n) (k8, n) (k1, n) (k9, n); word 2w + 1: the same for n + 8
            const uint32_t lo = words[2 * w], hi = words[2 * w + 1];
            fa[0] = marlin_pack2<T>(marlin_byte<WT>(lo), marlin_byte<WT>(lo >> 16));
            fa[2] = marlin_pack2<T>(marlin_byte<WT>(lo >> 8), marlin_byte<WT>(lo >> 24));
            fa[1] = marlin_pack2<T>(marlin_byte<WT>(hi), marlin_byte<WT>(hi >> 16));
            fa[3] = marlin_pack2<T>(marlin_byte<WT>(hi >> 8), marlin_byte<WT>(hi >> 24));
          }
          #pragma unroll
          for (int t = 0; t < TT; ++t) {
            if (channelwise) Tc::mma(acc[w][t], fa, bt[t][0], bt[t][1]);
            else Tc::mma(pacc[w][t], fa, bt[t][0], bt[t][1]);
          }
        }
      }
    }
    asm volatile("cp.async.wait_group 0;");
    if (!channelwise && cur_g >= 0) promote();
    __syncthreads();   // pipeline buffers are dead

    // ---- k-split reduction in warp order ----
    if constexpr (WK > 1) {
      float* red = reinterpret_cast<float*>(marlin_smem);
      const int slot = (wm * WN + wn) * 32 + lane;
      if (wk > 0) {
        #pragma unroll
        for (int w = 0; w < 4; ++w)
          #pragma unroll
          for (int t = 0; t < TT; ++t)
            #pragma unroll
            for (int i = 0; i < 4; ++i) red[((wk * WM * WN * 32 + slot) * 4 + w) * TT * 4 + t * 4 + i] = acc[w][t][i];
      }
      __syncthreads();
      if (wk == 0) {
        for (int q = 1; q < WK; ++q) {
          #pragma unroll
          for (int w = 0; w < 4; ++w)
            #pragma unroll
            for (int t = 0; t < TT; ++t)
              #pragma unroll
              for (int i = 0; i < 4; ++i) acc[w][t][i] += red[((q * WM * WN * 32 + slot) * 4 + w) * TT * 4 + t * 4 + i];
        }
      }
    }
    // ---- global K split: add the earlier slices' sum (fixed slice order), pass it on or finish ----
    if (splits > 1) {
      int* const lock = p.locks + tile;
      if (slice > 0) {
        if (tid == 0) {
          int v;
          do {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(lock) : "memory");
          } while (v != slice);
        }
        __syncthreads();
      }
      if (wk == 0 && w_live) {
        #pragma unroll
        for (int w = 0; w < 4; ++w)
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int n = nt_w * 64 + 16 * w + 8 * h + g;
            #pragma unroll
            for (int t = 0; t < TT; ++t)
              #pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int cr = s_crow[wm * TT * 8 + t * 8 + 2 * c4 + e];
                if (cr < 0) continue;
                float* const ct = p.c_tmp + static_cast<int64_t>(cr) * N + n;
                float& v = acc[w][t][2 * h + e];
                if (slice > 0) v = __ldcg(ct) + v;
                if (slice + 1 < splits) __stcg(ct, v);
              }
          }
      }
      if (slice + 1 < splits) {
        __threadfence();
        __syncthreads();
        if (tid == 0) asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(lock), "r"(slice + 1) : "memory");
        continue;   // uniform; s_crow is rewritten only after the next item's first __syncthreads
      }
    }
    // ---- epilogue: (acc x channel scale x 1 / global + bias) x topk weight -> T ----
    if (wk == 0 && w_live) {
      const float inv_g = (WT == 4 && p.global_scale != nullptr) ? 1.0f / *p.global_scale : 1.0f;
      T* const C = static_cast<T*>(p.c);
      const T* const bias = static_cast<const T*>(p.bias);
      #pragma unroll
      for (int w = 0; w < 4; ++w) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int n = nt_w * 64 + 16 * w + 8 * h + g;
          float cs = inv_g;
          if (channelwise) cs *= to_f(static_cast<const T*>(p.scales)[scale_off + marlin_channel_pos(n)]);
          const float bv = bias != nullptr ? to_f(bias[static_cast<int64_t>(MOE ? expert : 0) * N + n]) : 0.0f;
          #pragma unroll
          for (int t = 0; t < TT; ++t) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int tok = wm * TT * 8 + t * 8 + 2 * c4 + e;
              const int cr = s_crow[tok];
              if (cr < 0) continue;
              float v = fmaf(acc[w][t][2 * h + e], cs, bv);
              if (MOE && p.mul_topk_weights) v *= p.topk_weights[cr];
              from_f(C[static_cast<int64_t>(cr) * p.ldc + n], v);
            }
          }
        }
      }
    }
    __syncthreads();   // row tables / reduction buffer are rewritten by the next tile
    if (splits > 1 && tid == 0) p.locks[tile] = 0;   // the tile's slices are all done: re-arm
  }
#else
  (void)p;
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
