# =============================================================================
# kernel-ops | CMakeLists.txt
# M1 build: host core + the first golden-path kernels, one module per target set.
#   cmake -S . -B build -DKOPS_CUDA_ARCHS=native
#   cmake --build build -j
#   ./build/kops_test_m1
# Design rules (docs/TREE.md, docs/ARCH_MATRIX.md):
#   * kops_host has no CUDA dependency: status, dtype metadata, reference, registry.
#   * kops_cuda holds kernels and anything that touches the runtime.
#   * exact-SM targets come from KOPS_CUDA_ARCHS; "native" builds for the local GPU.
#   * no CUTLASS / CCCL / torch dependency in the release path (KOPS_WITH_* off).
# =============================================================================
cmake_minimum_required(VERSION 3.24)
project(kernel_ops LANGUAGES CXX CUDA)

# ---- compiler cache ---------------------------------------------------------
# ccache/sccache makes a rebuild after touching one header ~10x cheaper, which
# matters because every family header is included by many TUs. Auto-detected;
# pass -DKOPS_CCACHE_PROGRAM=<path> when the binary is not on PATH.
option(KOPS_USE_CCACHE "Use ccache/sccache as the compiler launcher" ON)
set(KOPS_CCACHE_PROGRAM "" CACHE FILEPATH "ccache/sccache executable; auto-detected when empty")
if(KOPS_USE_CCACHE)
  if(NOT KOPS_CCACHE_PROGRAM)
    find_program(KOPS_CCACHE_PROGRAM NAMES ccache sccache HINTS ENV CCACHE_BIN)
  endif()
  if(KOPS_CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${KOPS_CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${KOPS_CCACHE_PROGRAM}")
    message(STATUS "kernel-ops: compiler cache = ${KOPS_CCACHE_PROGRAM}")
  else()
    message(STATUS "kernel-ops: compiler cache not found (set KOPS_CCACHE_PROGRAM or -DKOPS_USE_CCACHE=OFF)")
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# CMake >= 4 defaults single-config generators to Debug, which makes CUDA kernels
# ~10x slower and defeats ccache (MSVC adds /Zi). Kernel builds default to Release;
# use -DKOPS_BUILD_TYPE=Debug when you actually need symbols.
set(KOPS_BUILD_TYPE "Release" CACHE STRING "Release | RelWithDebInfo | Debug")
set(CMAKE_BUILD_TYPE "${KOPS_BUILD_TYPE}" CACHE STRING "kernel-ops build type" FORCE)

set(KOPS_CUDA_ARCHS "native" CACHE STRING
    "CUDA targets: native, or a ;-list such as 80;89;90a;100a;120a")
option(KOPS_BUILD_TESTS "Build the host/GPU smoke tests" ON)
option(KOPS_WITH_CUTLASS "Link CUTLASS-based adapters (not implemented yet)" OFF)
option(KOPS_WITH_TORCH "Build bindings/torch (needs libtorch; not in M1)" OFF)

find_package(CUDAToolkit REQUIRED)

set(KOPS_NVCC_THREADS "4" CACHE STRING "nvcc --threads for one TU (0 = single-threaded)")

include(cmake/arch.cmake)
kops_set_cuda_archs("${KOPS_CUDA_ARCHS}")

# ---- host core (no CUDA, no torch) -----------------------------------------
add_library(kops_host STATIC
  csrc/common/status.cc
  csrc/common/dtype.cc
  csrc/reference/reference.cc
  csrc/runtime/registry.cc)
target_include_directories(kops_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(MSVC)
  target_compile_options(kops_host PRIVATE /W4)
else()
  target_compile_options(kops_host PRIVATE -Wall -Wextra)
endif()

# ---- device core ------------------------------------------------------------
# tensor.cc touches cuda_runtime.h, so it is compiled as CUDA.
set_source_files_properties(csrc/common/tensor.cc PROPERTIES LANGUAGE CUDA)
add_library(kops_cuda STATIC
  csrc/common/tensor.cc
  csrc/norm/norm.cu
  csrc/activation/activation.cu
  csrc/attention/merge.cu)
target_link_libraries(kops_cuda PUBLIC kops_host CUDA::cudart)
target_include_directories(kops_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(kops_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(KOPS_NVCC_THREADS GREATER 0)
  target_compile_options(kops_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--threads=${KOPS_NVCC_THREADS}>)
endif()
if(MSVC)
  target_compile_options(kops_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
else()
  target_compile_options(kops_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endif()

# ---- tests ------------------------------------------------------------------
if(KOPS_BUILD_TESTS)
  add_executable(kops_test_m1 tests/unit/test_m1.cc)
  target_link_libraries(kops_test_m1 PRIVATE kops_cuda)
  add_test(NAME kops_test_m1 COMMAND kops_test_m1)
  enable_testing()
endif()
