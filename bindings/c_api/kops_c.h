/* =============================================================================
 * kernel-ops | bindings/c_api/kops_c.h
 * Stable C ABI for FFI consumers (Rust, Go, Zig, C) and for engines that dlopen
 * libkops_<backend>_<arch>.so. This is the ONLY header a non-C++ consumer needs.
 *
 * Rules:
 *   * The structs below mirror csrc/common/{tensor.h,status.h} field by field.
 *     tools/header_check/abi_layout_test.cc static_asserts every offset, so a
 *     change to the C++ contract that is not mirrored here fails the build.
 *   * Opaque handles (stream, graph, event) are void*; never dereferenced.
 *   * Status values are append-only and equal to kops::Status.
 *   * An engine MUST call kops_abi_manifest() and compare abi_version and the
 *     struct sizes before using any other entry point.
 * ========================================================================== */
#ifndef KOPS_C_H
#define KOPS_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KOPS_C_ABI_VERSION 9u

/* ---- Status (== kops::Status) ------------------------------------------- */
typedef int32_t kops_status_t;
enum {
  KOPS_OK = 0,
  KOPS_INVALID_ARGUMENT = 1,
  KOPS_UNSUPPORTED_DTYPE = 2,
  KOPS_UNSUPPORTED_SHAPE = 3,
  KOPS_UNSUPPORTED_LAYOUT = 4,
  KOPS_UNSUPPORTED_ARCH = 5,
  KOPS_MISALIGNED = 6,
  KOPS_WORKSPACE_TOO_SMALL = 7,
  KOPS_CAPACITY_EXCEEDED = 8,
  KOPS_PLAN_INVALIDATED = 9,
  KOPS_BACKEND_ERROR = 10,       /* kCudaError: also used for HIP */
  KOPS_NOT_IMPLEMENTED = 11,
  KOPS_COMM_ERROR = 12,
  KOPS_NUMERICAL_GUARD = 13,
  KOPS_INTERNAL = 14,
  KOPS_UNSUPPORTED_BACKEND = 15
};

/* ---- Enums (fixed-width, equal to the C++ enum values) ------------------- */
typedef uint8_t kops_backend_t;       /* 0 = CUDA, 1 = HIP */
typedef uint8_t kops_dtype_t;         /* kops::DType */
typedef uint8_t kops_stride_unit_t;   /* 0 = element, 1 = byte */
typedef uint8_t kops_fp8_encoding_t;  /* 0 = OCP (448), 1 = FNUZ (240) */
typedef uint8_t kops_exec_strategy_t; /* 0 modular, 1 graph, 2 pdl, 3 persistent, 4 megakernel */

/* NB: KOPS_BACKEND_CUDA/KOPS_BACKEND_HIP are compile-time macros in
 * csrc/common/macros.cuh. CUDA is the only built backend for now; the HIP
 * value below is reserved and must not be reported as supported. */
enum { KOPS_BE_CUDA = 0, KOPS_BE_HIP = 1 };
enum { KOPS_STRIDE_ELEMENT = 0, KOPS_STRIDE_BYTE = 1 };
enum { KOPS_FP8_OCP = 0, KOPS_FP8_FNUZ = 1 };

/* ---- Target ids ---------------------------------------------------------- */
/* NB: the C++ side names these KOPS_TARGET_* as macros; the C constants carry
 * an ARCH_ID_ infix so both headers can be included in the same translation unit. */
enum {
  KOPS_ARCH_ID_SM70 = 700,    KOPS_ARCH_ID_SM75 = 750,    KOPS_ARCH_ID_SM80 = 800,
  KOPS_ARCH_ID_SM86 = 860,    KOPS_ARCH_ID_SM89 = 890,    KOPS_ARCH_ID_SM90A = 900,
  KOPS_ARCH_ID_SM100A = 1000, KOPS_ARCH_ID_SM103A = 1030, KOPS_ARCH_ID_SM120A = 1200,
  KOPS_ARCH_ID_SM121A = 1210,
  KOPS_ARCH_ID_GFX90A = 9010, KOPS_ARCH_ID_GFX942 = 9420, KOPS_ARCH_ID_GFX950 = 9500,
  KOPS_ARCH_ID_GFX1100 = 11000, KOPS_ARCH_ID_GFX1200 = 12000
};

/* ---- Tensor (mirror of kops::Tensor) ------------------------------------- */
#define KOPS_MAX_TENSOR_DIMS 8
typedef struct kops_tensor {
  void* data;
  kops_dtype_t dtype;
  uint8_t pad0[3];
  int32_t ndim;
  int32_t device;                 /* -1 = host */
  int32_t pad1;
  int64_t shape[KOPS_MAX_TENSOR_DIMS];
  int64_t stride[KOPS_MAX_TENSOR_DIMS];
  int64_t byte_offset;
  kops_stride_unit_t stride_unit;
  uint8_t pad2[7];
} kops_tensor;

/* ---- Workspace (mirror of kops::Workspace / WorkspaceRequirements) ------- */
typedef struct kops_workspace_requirements {
  size_t device_scratch_bytes;
  size_t device_persistent_bytes;
  size_t host_pinned_bytes;
  size_t alignment;
} kops_workspace_requirements;

typedef struct kops_workspace {
  void* device_scratch;
  size_t device_scratch_bytes;
  void* device_persistent;
  size_t device_persistent_bytes;
  void* host_pinned;
  size_t host_pinned_bytes;
} kops_workspace;

/* ---- LaunchContext (mirror of kops::LaunchContext) ----------------------- */
typedef struct kops_launch_context {
  void* stream;                   /* kops_stream_t */
  kops_backend_t backend;
  uint8_t pad0[3];
  int32_t device;
  int32_t arch_id;                /* KOPS_TARGET_* */
  int32_t wave_size;              /* 32 CUDA, 64 CDNA, 0 = unknown */
  int32_t compute_units;
  int32_t max_ctas_per_cu;
  int32_t lds_bytes;
  uint8_t enable_pdl;
  uint8_t deterministic;
  uint8_t batch_invariant;
  uint8_t allow_tf32;
  uint32_t capture_generation;
  kops_exec_strategy_t strategy;
  uint8_t pad1[3];
} kops_launch_context;

/* ---- ABI negotiation ----------------------------------------------------- */
typedef struct kops_abi_manifest {
  uint32_t abi_version;           /* KOPS_C_ABI_VERSION of the loaded library */
  uint32_t backend;               /* kops_backend_t */
  uint32_t sizeof_tensor;
  uint32_t sizeof_workspace;
  uint32_t sizeof_workspace_requirements;
  uint32_t sizeof_launch_context;
  uint32_t kernarg_limit_bytes;   /* CUDA 32764 */
  uint32_t reserved;
} kops_abi_manifest;

/* ---- Diagnostics --------------------------------------------------------- */
const char* kops_status_name(kops_status_t s);
const char* kops_last_error_message(void);
const char* kops_last_error_backend(void);
int32_t kops_last_backend_error(void);
void kops_clear_last_error(void);

/* ---- Version / probing --------------------------------------------------- */
uint32_t kops_abi_version(void);
kops_status_t kops_abi_manifest_get(kops_abi_manifest* out);

/* Fill a context for `device` (host query; never call during graph capture). */
kops_status_t kops_make_launch_context(int32_t device, void* stream, kops_launch_context* out);

/* ---- Contract checks shared by every family ------------------------------ */
/* Validate device/dtype/rank/stride-unit/alignment of one tensor view. */
kops_status_t kops_check_tensor(const kops_tensor* t, kops_dtype_t expected_dtype,
                                int32_t expected_ndim, size_t alignment_bytes,
                                const char* arg_name);

/* ---------------------------------------------------------------------------
 * Family entry points are declared next to their C++ launchers and mirrored
 * here as they reach `implemented`. The four above are the only ones required
 * to bootstrap a consumer (version pinning + validation + device probe).
 * ------------------------------------------------------------------------- */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* KOPS_C_H */
