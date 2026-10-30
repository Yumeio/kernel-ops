#pragma once
// =============================================================================
// kernel-ops | csrc/common/dtype.cuh
// Element types, numeric limits and conversion primitives.
// Requires: common/macros.cuh. Toolkit: <cuda_fp16.h>, <cuda_bf16.h>, <cuda_fp8.h> (>= 12.8).
// =============================================================================

namespace kops {

using f32_t = float;
using f16_t = __half;
using bf16_t = __nv_bfloat16;
using f8e4m3_t = __nv_fp8_e4m3;      // OCP FP8 E4M3FN, finite max 448, no inf
using f8e5m2_t = __nv_fp8_e5m2;      // OCP FP8 E5M2, max 57344
using ue8m0_t = __nv_fp8_e8m0;       // OCP MX scale: 2^(e-127), unsigned, no mantissa
using f4e2m1x2_t = __nv_fp4x2_e2m1;  // two FP4 E2M1 per byte; element 2i in low nibble

/// Logical element types used by every descriptor. ABI: append only.
enum class DType : uint8_t {
  kInvalid = 0,
  kFloat32 = 1,
  kFloat16 = 2,
  kBFloat16 = 3,
  kFloat64 = 4,
  kFloat8E4M3 = 5,     // e4m3fn (NVIDIA / OCP)
  kFloat8E5M2 = 6,
  kFloat8E8M0 = 7,     // UE8M0 scale factor (MX formats, DeepSeek UE8M0 FP8 scales)
  kFloat4E2M1x2 = 8,   // packed pair, 1 byte per 2 elements
  kInt8 = 10,
  kUInt8 = 11,
  kInt4x2 = 12,        // packed signed int4 pair, low nibble = even element
  kUInt4x2 = 13,       // packed unsigned int4 pair (asymmetric zero-point formats)
  kInt16 = 14,
  kInt32 = 15,
  kInt64 = 16,
  kUInt32 = 17,
  kBool = 18,
};

// ---- Numeric limits ----------------------------------------------------------
inline constexpr float kFp8E4M3Max = 448.0f;
inline constexpr float kFp8E5M2Max = 57344.0f;
inline constexpr float kFp4E2M1Max = 6.0f;
inline constexpr float kInt8Max = 127.0f;
inline constexpr float kInt4Max = 7.0f;
inline constexpr int kMxBlock = 32;     // OCP MX block size (MXFP8/MXFP6/MXFP4)
inline constexpr int kNvFp4Block = 16;  // NVFP4 block size (E4M3 scale per 16)

/// Bits per logical element (4 for packed fp4/int4, 0 for kInvalid).
KOPS_HOST_DEVICE int dtype_bits(DType t);
/// True for kFloat4E2M1x2 / kInt4x2 / kUInt4x2 (two elements per byte).
KOPS_HOST_DEVICE bool dtype_is_packed_subbyte(DType t);
/// Bytes needed for `num_elements` logical elements of type t (rounded up).
KOPS_HOST_DEVICE int64_t dtype_storage_bytes(DType t, int64_t num_elements);
/// Lowercase name ("bfloat16", "float8_e4m3fn", ...).
const char* dtype_name(DType t) noexcept;

/// Map a C++ element type to its DType at compile time: dtype_of<bf16_t>::value.
template <typename T> struct dtype_of;

// ---- Device conversions (all fp32 math, round-to-nearest-even) ---------------
/// Widen any supported scalar to fp32.
template <typename T> KOPS_DEVICE float to_float(T x);
/// Narrow fp32 to T. FP8 conversions saturate to the finite range (satfinite),
/// integer conversions round-half-even then clamp to [-max, max].
template <typename T> KOPS_DEVICE T from_float(float x);
/// Pack four fp32 into four e4m3 in one 32-bit word (cvt.rn.satfinite.e4m3x2.f32 x2).
KOPS_DEVICE uint32_t cvt_f32x4_to_e4m3x4(float a, float b, float c, float d);
/// Pack eight fp32 into eight e2m1 in one 32-bit word (sm_100a/sm_120a cvt.rn.satfinite.e2m1x2).
KOPS_DEVICE uint32_t cvt_f32x8_to_e2m1x8(const float (&v)[8]);
/// Unpack eight e2m1 nibbles into fp32 (lookup or cvt on supported arch).
KOPS_DEVICE void cvt_e2m1x8_to_f32x8(uint32_t packed, float (&out)[8]);
/// Encode a positive fp32 scale as UE8M0. round_up=true gives ceil(log2(s)), which
/// guarantees amax/scale <= format max (DeepSeek-V3.1+/Kimi-K3 MX recipes).
KOPS_DEVICE uint8_t float_to_ue8m0(float scale, bool round_up);
/// Decode UE8M0 to fp32 (2^(e-127)); 0xFF is NaN by spec and must never be produced.
KOPS_HOST_DEVICE float ue8m0_to_float(uint8_t e);

}  // namespace kops
