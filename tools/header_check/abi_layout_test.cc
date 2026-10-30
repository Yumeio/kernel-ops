// =============================================================================
// kernel-ops | tools/header_check/abi_layout_test.cc
// Pins the ABI that bindings/c_api/kops_c.h promises:
//   * public structs are standard-layout and trivially copyable;
//   * the C mirror has identical size and every field offset;
//   * enums have the documented width;
//   * kernel parameter blocks fit the backend kernarg limit.
// Run on the host without nvcc:
//   clang++ -std=c++17 -fsyntax-only -Wall -Wextra -I. tools/header_check/abi_layout_test.cc
// A change to any public struct that is not mirrored in kops_c.h fails here,
// which is the point: KOPS_ABI_VERSION must be bumped deliberately.
// =============================================================================
#include "tools/header_check/cuda_host_shim.h"

#include <cstddef>
#include <type_traits>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/sync.cuh"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/gemm/grouped_gemm_kernels.cuh"
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/moe/fused_moe_kernels.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/communication/allreduce_kernels.cuh"
#include "csrc/communication/ep_kernels.cuh"
#include "csrc/execution/persistent.cuh"
#include "csrc/execution/launch.h"
#include "csrc/mega/mega_plan.cuh"
#include "csrc/mega/mega_kernels.cuh"
#include "csrc/runtime/registry.h"
#include "csrc/memory/ops.h"
#include "csrc/load/ops.h"
#include "csrc/backends/ops.h"

#include "bindings/c_api/kops_c.h"

// ---- 1. Every public struct must be safe to pass across the C ABI -----------
#define KOPS_ABI_STANDARD_LAYOUT(T)                                     \
  static_assert(std::is_standard_layout<T>::value, #T " must be standard-layout"); \
  static_assert(std::is_trivially_copyable<T>::value, #T " must be trivially copyable")

KOPS_ABI_STANDARD_LAYOUT(kops::Tensor);
KOPS_ABI_STANDARD_LAYOUT(kops::Workspace);
KOPS_ABI_STANDARD_LAYOUT(kops::WorkspaceRequirements);
KOPS_ABI_STANDARD_LAYOUT(kops::LaunchContext);
KOPS_ABI_STANDARD_LAYOUT(kops::QuantScheme);
KOPS_ABI_STANDARD_LAYOUT(kops::ActQuantOut);
KOPS_ABI_STANDARD_LAYOUT(kops::GluParams);
KOPS_ABI_STANDARD_LAYOUT(kops::kv::PagedKVDesc);
KOPS_ABI_STANDARD_LAYOUT(kops::gemm::ScaledOperand);
KOPS_ABI_STANDARD_LAYOUT(kops::gemm::EpilogueParams);
KOPS_ABI_STANDARD_LAYOUT(kops::gemm::ScaledGemmArgs);
KOPS_ABI_STANDARD_LAYOUT(kops::gemm::GemmTactic);
KOPS_ABI_STANDARD_LAYOUT(kops::gemm::GemmProblem);
KOPS_ABI_STANDARD_LAYOUT(kops::gemm::GroupedGemmParams);
KOPS_ABI_STANDARD_LAYOUT(kops::moe::MoELayerParams);
KOPS_ABI_STANDARD_LAYOUT(kops::attn::AttentionSpec);
KOPS_ABI_STANDARD_LAYOUT(kops::attn::AttentionModifiers);
KOPS_ABI_STANDARD_LAYOUT(kops::attn::DecodeParams);
KOPS_ABI_STANDARD_LAYOUT(kops::attn::MLADecodeParams);
KOPS_ABI_STANDARD_LAYOUT(kops::mega::MegaPlanDevice);
KOPS_ABI_STANDARD_LAYOUT(kops::comm::EPBuffers);
KOPS_ABI_STANDARD_LAYOUT(kops::exec::LaunchConfig);
KOPS_ABI_STANDARD_LAYOUT(kops::mem::MemPoolDesc);
KOPS_ABI_STANDARD_LAYOUT(kops::mem::OffloadDesc);
KOPS_ABI_STANDARD_LAYOUT(kops::mem::GreenCtxDesc);
KOPS_ABI_STANDARD_LAYOUT(kops::load::CheckpointMeta);
KOPS_ABI_STANDARD_LAYOUT(kops::load::TensorRoute);
KOPS_ABI_STANDARD_LAYOUT(kops::load::LoadPlan);
KOPS_ABI_STANDARD_LAYOUT(kops::backend::ProviderInfo);
KOPS_ABI_STANDARD_LAYOUT(kops::backend::AdapterEntry);
KOPS_ABI_STANDARD_LAYOUT(kops::backend::AdapterTable);
KOPS_ABI_STANDARD_LAYOUT(kops::backend::ConformanceCase);
KOPS_ABI_STANDARD_LAYOUT(kops::backend::ConformanceResult);

// ---- 2. Enum width is ABI ------------------------------------------------
static_assert(sizeof(kops::Status) == 4, "Status is int32 in the C ABI");
static_assert(sizeof(kops::DType) == 1, "DType is uint8 in the C ABI");
static_assert(sizeof(kops::Fp8Encoding) == 1, "Fp8Encoding is uint8 in the C ABI");
static_assert(sizeof(kops::BackendKind) == 1, "BackendKind is uint8 in the C ABI");
static_assert(sizeof(kops::StrideUnit) == 1, "StrideUnit is uint8 in the C ABI");

// ---- 3. C mirror layout --------------------------------------------------
static_assert(sizeof(kops::Tensor) == sizeof(kops_tensor), "kops_tensor size drift");
static_assert(offsetof(kops::Tensor, data) == offsetof(kops_tensor, data), "kops_tensor.data");
static_assert(offsetof(kops::Tensor, dtype) == offsetof(kops_tensor, dtype), "kops_tensor.dtype");
static_assert(offsetof(kops::Tensor, ndim) == offsetof(kops_tensor, ndim), "kops_tensor.ndim");
static_assert(offsetof(kops::Tensor, device) == offsetof(kops_tensor, device), "kops_tensor.device");
static_assert(offsetof(kops::Tensor, shape) == offsetof(kops_tensor, shape), "kops_tensor.shape");
static_assert(offsetof(kops::Tensor, stride) == offsetof(kops_tensor, stride), "kops_tensor.stride");
static_assert(offsetof(kops::Tensor, byte_offset) == offsetof(kops_tensor, byte_offset),
              "kops_tensor.byte_offset");
static_assert(offsetof(kops::Tensor, stride_unit) == offsetof(kops_tensor, stride_unit),
              "kops_tensor.stride_unit");

static_assert(sizeof(kops::Workspace) == sizeof(kops_workspace), "kops_workspace size drift");
static_assert(offsetof(kops::Workspace, device_persistent) ==
                  offsetof(kops_workspace, device_persistent),
              "kops_workspace.device_persistent");

static_assert(sizeof(kops::WorkspaceRequirements) == sizeof(kops_workspace_requirements),
              "kops_workspace_requirements size drift");

static_assert(sizeof(kops::LaunchContext) == sizeof(kops_launch_context),
              "kops_launch_context size drift");
static_assert(offsetof(kops::LaunchContext, stream) == offsetof(kops_launch_context, stream), "");
static_assert(offsetof(kops::LaunchContext, backend) == offsetof(kops_launch_context, backend), "");
static_assert(offsetof(kops::LaunchContext, arch_id) == offsetof(kops_launch_context, arch_id), "");
static_assert(offsetof(kops::LaunchContext, wave_size) == offsetof(kops_launch_context, wave_size), "");
static_assert(offsetof(kops::LaunchContext, lds_bytes) == offsetof(kops_launch_context, lds_bytes), "");
static_assert(offsetof(kops::LaunchContext, enable_pdl) == offsetof(kops_launch_context, enable_pdl), "");
static_assert(offsetof(kops::LaunchContext, capture_generation) ==
                  offsetof(kops_launch_context, capture_generation), "");
static_assert(offsetof(kops::LaunchContext, strategy) == offsetof(kops_launch_context, strategy), "");

static_assert(KOPS_C_ABI_VERSION == KOPS_ABI_VERSION, "C and C++ ABI versions must match");

// ---- 4. Kernel parameter blocks fit the kernarg space --------------------
#define KOPS_FITS_KERNARG(T)                                                        \
  static_assert(sizeof(T) <= static_cast<size_t>(kops::kKernargLimitBytes),          \
                #T " exceeds KOPS_KERNARG_LIMIT_BYTES (CUDA 32764 / HIP ~4096)")

KOPS_FITS_KERNARG(kops::gemm::ScaledGemmArgs);
KOPS_FITS_KERNARG(kops::gemm::GemmTmaParams);
KOPS_FITS_KERNARG(kops::gemm::GroupedGemmParams);
KOPS_FITS_KERNARG(kops::moe::MoELayerParams);
KOPS_FITS_KERNARG(kops::attn::DecodeParams);
KOPS_FITS_KERNARG(kops::attn::MLADecodeParams);
KOPS_FITS_KERNARG(kops::mega::MegaAttentionBlockParams);
KOPS_FITS_KERNARG(kops::mega::MegaMoELocalParams);
KOPS_FITS_KERNARG(kops::comm::EPDispatchParams);
KOPS_FITS_KERNARG(kops::comm::EPCombineParams);

// ---- 5. Documented constants must not drift ------------------------------
static_assert(kops::kMaxTensorDims == KOPS_MAX_TENSOR_DIMS, "kMaxTensorDims");
static_assert(static_cast<int>(kops::Status::kUnsupportedBackend) == KOPS_UNSUPPORTED_BACKEND, "");
static_assert(static_cast<int>(kops::Status::kCudaError) == KOPS_BACKEND_ERROR, "");

int main() { return 0; }
