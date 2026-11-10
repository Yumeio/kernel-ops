/* =============================================================================
 * kernel-ops | bindings/c_api/kops_c_check.c
 * Proves kops_c.h is a self-contained C header (no C++ compiler, no CUDA/HIP
 * headers) and that the C view of the ABI has the expected field widths.
 *   clang -std=c11 -fsyntax-only -Wall -Wextra -I. bindings/c_api/kops_c_check.c
 * ========================================================================== */
#include "bindings/c_api/kops_c.h"

_Static_assert(KOPS_C_ABI_VERSION == 7u, "C ABI version");
_Static_assert(sizeof(kops_status_t) == 4, "kops_status_t is int32");
_Static_assert(sizeof(kops_dtype_t) == 1, "kops_dtype_t is uint8");
_Static_assert(sizeof(kops_tensor) == 168, "kops_tensor size (x86-64 / aarch64 LP64)");
_Static_assert(offsetof(kops_tensor, shape) == 24, "kops_tensor.shape offset");
_Static_assert(offsetof(kops_tensor, stride) == 88, "kops_tensor.stride offset");
_Static_assert(offsetof(kops_tensor, stride_unit) == 160, "kops_tensor.stride_unit offset");
_Static_assert(sizeof(kops_abi_manifest) == 32, "kops_abi_manifest size");

int main(void) { return 0; }
