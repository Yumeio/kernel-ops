#pragma once
// =============================================================================
// kernel-ops | csrc/common/quant.cuh
// Quantization scheme descriptor shared by GEMM, MoE, KV cache, norm/act epilogues.
// Convention used everywhere: scales are DEQUANT MULTIPLIERS, x ~= q * s
// (and x ~= (q - z) * s for asymmetric). A checkpoint storing 1/s must be
// converted at import time, never reinterpreted silently.
// Requires: common/macros.cuh, common/dtype.cuh.
// =============================================================================

namespace kops {

/// Axis/shape covered by one scale value.
enum class ScaleGranularity : uint8_t {
  kNone = 0,
  kPerTensor = 1,       // one scale
  kPerChannel = 2,      // one per output channel N of a weight [N, K]
  kPerToken = 3,        // one per activation row M
  kPerHead = 4,         // one per KV head
  kPerTokenHead = 5,    // one per (token, kv_head) KV entry
  kPerGroup = 6,        // 1 x group_k along K (DeepSeek FP8 activations: 1x128; AWQ/GPTQ: g=32/64/128)
  kPerBlock2D = 7,      // group_m x group_k tiles (DeepSeek FP8 weights: 128x128)
  kMX = 8,              // OCP MX: 1x32, UE8M0 scale (MXFP8 / MXFP6 / MXFP4)
  kNVFP4 = 9,           // 1x16 E4M3 scale + one FP32 global scale per tensor
};

/// Memory layout of the scale tensor.
enum class ScaleLayout : uint8_t {
  kRowMajor = 0,         // [rows, cols/group]
  kColMajorTma = 1,      // [cols/group, round_up(rows, 4)] MN-major, TMA aligned (DeepGEMM)
  kSwizzled128x4 = 2,    // 128x4 swizzled tile layout for tcgen05 / CUTLASS block-scaled MMA
  kMarlinPermuted = 3,   // Marlin scale permutation
  kPackedUE8M0x4 = 4,    // four UE8M0 packed per int32 (DeepGEMM sm100 recipe)
  kInlineWithData = 5,   // scale stored inside the data record (KV MLA FP8 656 B entries)
};

/// Physical packing of low-bit weights.
enum class WeightPacking : uint8_t {
  kNone = 0,
  kInt4LowNibbleFirst = 1,  // plain packed int4, element 2i in low nibble
  kAWQ = 2,                 // qweight [K, N/8] int32, nibble order {0,2,4,6,1,3,5,7}
  kGPTQ = 3,                // qweight [K/8, N] int32 (+ g_idx / act-order permutation)
  kMarlin = 4,              // Marlin tile-interleaved layout (repacked from AWQ/GPTQ/FP4)
  kMachete = 5,             // CUTLASS mixed-input SM90 layout
  kGGUF = 6,                // ggml block_q* structs, see GGUFType
  kFP4E2M1 = 7,             // packed e2m1 pairs (MXFP4 / NVFP4)
  kTrtllmShuffled = 8,      // TRTLLM-gen MoE shuffled FP4/FP8 weight layout
  kW2 = 9,                  // experimental 2-bit expert packing (catalog M04)
};

/// ggml block types; numeric values equal GGML_TYPE_* so they can be passed through.
enum class GGUFType : uint8_t {
  kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ4_1 = 3, kQ5_0 = 6, kQ5_1 = 7, kQ8_0 = 8, kQ8_1 = 9,
  kQ2_K = 10, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kQ8_K = 15,
  kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ1_S = 19, kIQ4_NL = 20, kIQ3_S = 21,
  kIQ2_S = 22, kIQ4_XS = 23, kIQ1_M = 29, kBF16 = 30,
};

/// Optional orthogonal transform applied before quantization.
enum class QuantTransform : uint16_t {
  kNone = 0,
  kHadamard = 1,        // normalized Walsh-Hadamard over transform_size (QuaRot/SpinQuant style)
  kSignedHadamard = 2,  // diag(sign(seed)) * H; seed convention recorded in transform_seed
  kSmoothFolded = 3,    // SmoothQuant scales already folded into previous op (metadata only)
};

/// Where the scheme came from (provenance only; never changes kernel math).
enum class QuantProvenance : uint16_t {
  kUnknown = 0, kRTN = 1, kAWQ = 2, kGPTQ = 3, kSmoothQuant = 4, kQAT = 5, kGGUF = 6,
  kHQQ = 7, kModelOpt = 8, kCompressedTensors = 9, kBitsAndBytesNF4 = 10,
};

/// Full description of one quantized operand (weight, activation, K or V cache).
struct QuantScheme {
  DType storage_dtype = DType::kInvalid;   // payload element type
  DType scale_dtype = DType::kFloat32;     // f32 | bf16 | e4m3 (NVFP4) | ue8m0 (MX)
  DType zero_dtype = DType::kInvalid;      // kInvalid = symmetric
  // FP8 interpretation of the payload. Only meaningful for kFloat8E4M3 /
  // kFloat8E5M2; FNUZ payloads set DType::kFloat8E4M3FNUZ and enc == kFNUZ.
  // Never infer the encoding from the backend: a gfx94x reader must be told.
  Fp8Encoding fp8 = Fp8Encoding::kOCP;
  ScaleGranularity granularity = ScaleGranularity::kNone;
  ScaleLayout scale_layout = ScaleLayout::kRowMajor;
  WeightPacking packing = WeightPacking::kNone;
  uint16_t packing_version = 0;
  int16_t group_m = 1;                     // rows per scale block (kPerBlock2D)
  int16_t group_k = 0;                     // cols per scale (kPerGroup/kPerBlock2D/MX/NVFP4)
  bool has_zero_point = false;
  bool zero_point_is_float = false;        // HQQ/AWQ-float style (x = (q - z) * s with float z)
  bool scale_round_pow2 = false;           // scales constrained to powers of two (UE8M0 recipe)
  QuantTransform transform = QuantTransform::kNone;
  uint16_t transform_size = 0;             // Hadamard length
  uint32_t transform_seed = 0;
  QuantProvenance provenance = QuantProvenance::kUnknown;
  GGUFType gguf_type = GGUFType::kF32;     // valid when packing == kGGUF
};

/// On-the-fly activation quantization modes used by norm/act/MoE/GEMM epilogues.
enum class ActQuantMode : uint8_t {
  kNone = 0,
  kStaticPerTensor = 1,    // precomputed scale (FP8 / INT8)
  kDynamicPerTensor = 2,   // two-pass: absmax reduce then quantize
  kDynamicPerToken = 3,    // one scale per row (FP8 / INT8 sym, INT8 asym with azp)
  kPerGroup = 4,           // 1 x group (128 for DeepSeek FP8; UE8M0 optional)
  kMXFP8 = 5,              // e4m3 payload + UE8M0 per 32 (Kimi-K3 activations)
  kMXFP4 = 6,              // e2m1 payload + UE8M0 per 32
  kNVFP4 = 7,              // e2m1 payload + E4M3 per 16 + global fp32
};

/// Arithmetic used for the quotient x / scale of a per-group activation.
///   kExact      the correctly rounded fp32 quotient, so the payload bytes are a function of the
///               data alone and never depend on the launch shape, vector width, alignment or
///               arithmetic unit. This is the contract, not an instruction choice: bf16/fp16
///               inputs reach that quotient with one RN reciprocal plus a Markstein correction
///               (byte-identical to a division; every 16-bit bit pattern is swept in the unit
///               test) and keep `__fdiv_rn` for scales outside the proved range (a group whose
///               amax is +inf or NaN). fp32 inputs always divide. Default.
///   kReciprocal approximate and cheap: one RN reciprocal per group plus a raw multiply, so the
///               payload may differ from kExact by at most one code (about one ulp of the
///               quotient). Opt-in, per-group only; never silently substituted for kExact.
/// Power-of-two scales (UE8M0) multiply exactly in both modes.
/// ABI: appended at the end of ActQuantOut, so the existing field offsets and the struct size are
/// unchanged; the enumerator itself is a new public contract (bump KOPS_ABI_VERSION).
enum class QuantMath : uint8_t {
  kExact = 0,
  kReciprocal = 1,
};

/// Output descriptor for a quantized activation written by a fused kernel.
struct ActQuantOut {
  void* q = nullptr;                    // [rows, cols] payload (packed for fp4)
  int64_t q_row_stride = 0;             // in storage elements (bytes for fp4)
  void* scales = nullptr;               // layout per `scale_layout`
  int64_t scale_row_stride = 0;         // for row-major / col-major TMA layouts
  int32_t* zero_points = nullptr;       // [rows] int32 azp for asymmetric int8, else null
  const float* static_scale = nullptr;  // [1] device scalar for kStaticPerTensor
  const float* scale_upper_bound = nullptr;  // [1] clamp for dynamic per-token FP8 (vLLM scale_ub)
  const float* global_scale = nullptr;  // [1] NVFP4 global (= 448*6/amax) supplied by caller
  float* amax_out = nullptr;            // [1] optional running absmax (for next-step static scales)
  DType q_dtype = DType::kInvalid;
  ActQuantMode mode = ActQuantMode::kNone;
  ScaleLayout scale_layout = ScaleLayout::kRowMajor;
  int32_t group_size = 0;               // 128 (per-group), 32 (MX), 16 (NVFP4)
  bool scale_ue8m0 = false;             // round per-group scales up to powers of two
  bool symmetric = true;
  // Encoding of the payload when q_dtype is an FP8 type (see common/dtype.cuh).
  Fp8Encoding fp8 = Fp8Encoding::kOCP;
  // Only ActQuantMode::kPerGroup reads this today; every other mode rejects kReciprocal rather
  // than silently running the exact loop. The per-tensor / per-token / MX / NVFP4 paths keep
  // exact arithmetic.
  QuantMath math = QuantMath::kExact;
};

/// Physical bytes for a quantized tensor of `rows x cols` incl. scales, zero points,
/// swizzle padding. This (not storage_dtype bits) is the truth for memory ledgers.
KOPS_HOST_DEVICE int64_t quant_physical_bytes(const QuantScheme& s, int64_t rows, int64_t cols);

}  // namespace kops
