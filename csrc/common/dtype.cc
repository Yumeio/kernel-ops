// =============================================================================
// kernel-ops | csrc/common/dtype.cc
// Host-side element-type metadata. Device conversions live in device_impl.cuh
// (not part of this file) because they must be compiled per target.
// =============================================================================
#include <cstdint>

#include "csrc/common/macros.cuh"
#include "csrc/common/dtype.cuh"

namespace kops {

int dtype_bits(DType t) {
  switch (t) {
    case DType::kFloat64: return 64;
    case DType::kFloat32:
    case DType::kInt32:
    case DType::kUInt32: return 32;
    case DType::kFloat16:
    case DType::kBFloat16:
    case DType::kInt16: return 16;
    case DType::kFloat8E4M3:
    case DType::kFloat8E5M2:
    case DType::kFloat8E8M0:
    case DType::kInt8:
    case DType::kUInt8:
    case DType::kBool:
    case DType::kFloat8E4M3FNUZ:
    case DType::kFloat8E5M2FNUZ: return 8;
    case DType::kFloat4E2M1x2:
    case DType::kInt4x2:
    case DType::kUInt4x2: return 4;
    case DType::kInt64: return 64;
    case DType::kInvalid: return 0;
  }
  return 0;
}

bool dtype_is_packed_subbyte(DType t) {
  return t == DType::kFloat4E2M1x2 || t == DType::kInt4x2 || t == DType::kUInt4x2;
}

bool dtype_is_fp8(DType t) {
  return t == DType::kFloat8E4M3 || t == DType::kFloat8E5M2 || t == DType::kFloat8E4M3FNUZ ||
         t == DType::kFloat8E5M2FNUZ;
}

int64_t dtype_storage_bytes(DType t, int64_t num_elements) {
  const int bits = dtype_bits(t);
  if (bits <= 0 || num_elements < 0) return 0;
  return (num_elements * bits + 7) / 8;
}

const char* dtype_name(DType t) noexcept {
  switch (t) {
    case DType::kInvalid: return "invalid";
    case DType::kFloat32: return "float32";
    case DType::kFloat16: return "float16";
    case DType::kBFloat16: return "bfloat16";
    case DType::kFloat64: return "float64";
    case DType::kFloat8E4M3: return "float8_e4m3fn";
    case DType::kFloat8E5M2: return "float8_e5m2";
    case DType::kFloat8E8M0: return "float8_e8m0";
    case DType::kFloat4E2M1x2: return "float4_e2m1x2";
    case DType::kInt8: return "int8";
    case DType::kUInt8: return "uint8";
    case DType::kInt4x2: return "int4x2";
    case DType::kUInt4x2: return "uint4x2";
    case DType::kInt16: return "int16";
    case DType::kInt32: return "int32";
    case DType::kInt64: return "int64";
    case DType::kUInt32: return "uint32";
    case DType::kBool: return "bool";
    case DType::kFloat8E4M3FNUZ: return "float8_e4m3fnuz";
    case DType::kFloat8E5M2FNUZ: return "float8_e5m2fnuz";
  }
  return "unknown";
}

float fp8_max(DType t) {
  switch (t) {
    case DType::kFloat8E4M3: return kFp8E4M3Max;
    case DType::kFloat8E5M2: return kFp8E5M2Max;
    case DType::kFloat8E4M3FNUZ: return kFp8E4M3FNUZMax;
    case DType::kFloat8E5M2FNUZ: return kFp8E5M2FNUZMax;
    default: return 0.0f;
  }
}

}  // namespace kops
