# =============================================================================
# kernel-ops | CMakeLists.txt
# M1 build: host core + the first golden-path kernels, one module per target set.
#   cmake --preset dev          # native arch, tests on (see CMakePresets.json)
#   cmake --build --preset dev
#   ctest --preset dev
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
# The launcher only works with Ninja/Makefiles: CMake ignores it for the Visual
# Studio generator, so use -G Ninja there.
option(KOPS_USE_CCACHE "Use ccache/sccache as the compiler launcher" ON)
set(KOPS_CCACHE_PROGRAM "" CACHE FILEPATH "ccache/sccache executable; auto-detected when empty")
if(KOPS_USE_CCACHE)
  if(NOT KOPS_CCACHE_PROGRAM)
    # find_program() skips a variable that is already defined (even as an empty
    # string), so search into a temporary and promote the result to the cache.
    find_program(KOPS_CCACHE_PROGRAM_FOUND NAMES ccache sccache HINTS ENV CCACHE_BIN)
    if(KOPS_CCACHE_PROGRAM_FOUND)
      set(KOPS_CCACHE_PROGRAM "${KOPS_CCACHE_PROGRAM_FOUND}" CACHE FILEPATH
          "ccache/sccache executable; auto-detected when empty" FORCE)
    endif()
    unset(KOPS_CCACHE_PROGRAM_FOUND CACHE)
  endif()
  if(KOPS_CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${KOPS_CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${KOPS_CCACHE_PROGRAM}")
    message(STATUS "kernel-ops: compiler cache = ${KOPS_CCACHE_PROGRAM}")
    if(NOT CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
      message(WARNING
        "kernel-ops: the ${CMAKE_GENERATOR} generator ignores the compiler cache; "
        "configure with -G Ninja (see CMakePresets.json).")
    endif()
  else()
    message(STATUS
      "kernel-ops: compiler cache not found (install ninja+ccache, or set "
      "KOPS_CCACHE_PROGRAM / -DKOPS_USE_CCACHE=OFF)")
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(GNUInstallDirs)

set(KOPS_BUILD_TYPE "Release" CACHE STRING
    "Default CMAKE_BUILD_TYPE for single-config generators")
# CMake >= 4 leaves single-config generators without a build type, which makes
# CUDA kernels ~10x slower and adds MSVC /Zi that defeats ccache. Default to
# Release, but never override an explicit -DCMAKE_BUILD_TYPE. Multi-config
# generators (Visual Studio) carry the configuration at build time.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "${KOPS_BUILD_TYPE}" CACHE STRING "kernel-ops build type" FORCE)
endif()

set(KOPS_CUDA_ARCHS "native" CACHE STRING
    "CUDA targets: native, or a ;-list such as 80;89;90a;100a;120a")
option(KOPS_BUILD_TESTS "Build the host/GPU smoke tests" ON)
option(KOPS_BUILD_BENCHMARKS "Build the C++ benchmark targets" OFF)
option(KOPS_WITH_CUTLASS "Link CUTLASS-based adapters (not implemented yet)" OFF)
option(KOPS_WITH_TORCH "Build bindings/torch (needs libtorch; not in M1)" OFF)
option(KOPS_LINEINFO "Emit -lineinfo in CUDA objects (Nsight source view; slower build)" OFF)
option(KOPS_PTXAS_VERBOSE "Pass -Xptxas=-v and print per-kernel register/smem usage" OFF)

find_package(CUDAToolkit REQUIRED)

set(KOPS_NVCC_THREADS "4" CACHE STRING "nvcc --threads for one TU (0 = single-threaded)")

include(cmake/arch.cmake)
include(cmake/instantiate.cmake)
kops_set_cuda_archs("${KOPS_CUDA_ARCHS}")

# ---- shared compile settings ------------------------------------------------
set(KOPS_CUDA_OPTIONS "")
if(KOPS_LINEINFO)
  list(APPEND KOPS_CUDA_OPTIONS -lineinfo)
endif()
if(KOPS_PTXAS_VERBOSE)
  list(APPEND KOPS_CUDA_OPTIONS -Xptxas=-v)
endif()
if(KOPS_NVCC_THREADS GREATER 0)
  list(APPEND KOPS_CUDA_OPTIONS --threads=${KOPS_NVCC_THREADS})
endif()

function(kops_set_cuda_options target)
  if(KOPS_CUDA_OPTIONS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${KOPS_CUDA_OPTIONS}>)
  endif()
  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
  endif()
endfunction()

# ---- host core (no CUDA, no torch) -----------------------------------------
add_library(kops_host STATIC
  csrc/common/status.cc
  csrc/common/dtype.cc
  csrc/reference/reference.cc
  csrc/runtime/registry.cc)
target_include_directories(kops_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(kops_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  target_compile_options(kops_host PRIVATE /W4)
else()
  target_compile_options(kops_host PRIVATE -Wall -Wextra)
endif()

# The registry table is generated from registry/capabilities/*.yaml at configure
# time. A build without Python (or without YAML rows) keeps the weak empty table.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  set(KOPS_REGISTRY_GEN_DIR "${CMAKE_BINARY_DIR}/generated")
  set(KOPS_REGISTRY_TABLE "${KOPS_REGISTRY_GEN_DIR}/registry_table.inc")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/gen_registry.py"
            -o "${KOPS_REGISTRY_TABLE}" --dir "${CMAKE_SOURCE_DIR}/registry/capabilities"
    RESULT_VARIABLE KOPS_REGISTRY_RESULT
    OUTPUT_QUIET ERROR_QUIET)
  if(KOPS_REGISTRY_RESULT EQUAL 0)
    target_include_directories(kops_host PRIVATE "${KOPS_REGISTRY_GEN_DIR}")
    target_compile_definitions(kops_host PRIVATE KOPS_REGISTRY_HAS_TABLE=1)
  endif()
endif()

# ---- device core ------------------------------------------------------------
# tensor.cc calls the CUDA runtime but contains no device code, so it is compiled
# once as C++ instead of once per gencode target.
add_library(kops_cuda STATIC
  csrc/common/tensor.cc
  csrc/execution/launch.cc
  csrc/runtime/trace.cc
  csrc/model_ops/residual.cu
  csrc/norm/norm.cu
  csrc/activation/activation.cu
  csrc/position/rope.cu
  csrc/embedding/embedding.cu
  csrc/sampling/sampling.cu
  csrc/sampling/logits_processor.cu
  csrc/quantization/quant.cu
  csrc/quantization/weight_repack.cu
  csrc/gemm/gemm_prep.cu
  csrc/gemm/mixed_input_gemm.cu
  csrc/gemm/fp8_gemm.cu
  csrc/gemm/int8_gemm.cu
  csrc/gemm/fp4_gemm.cu
  csrc/gemm/dense_gemm.cu
  csrc/gemm/gguf_gemm.cu
  csrc/gemm/grouped_gemm.cu
  csrc/moe/fused_moe.cu
  csrc/moe/routing.cu
  csrc/moe/permute.cu
  csrc/attention/merge.cu
  csrc/attention/mla.cu
  csrc/kvcache/kvcache.cu
  csrc/state/state_cache.cu)
target_link_libraries(kops_cuda PUBLIC kops_host CUDA::cudart)
target_include_directories(kops_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(kops_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Shared)
kops_set_cuda_options(kops_cuda)

# ---- tests ------------------------------------------------------------------
function(kops_add_test name)
  cmake_parse_arguments(TEST "" "" "SOURCES;LIBS" ${ARGN})
  add_executable(${name} ${TEST_SOURCES})
  target_link_libraries(${name} PRIVATE ${TEST_LIBS})
  kops_set_cuda_options(${name})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 77)
endfunction()

if(KOPS_BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

if(KOPS_BUILD_TESTS)
  enable_testing()
  # ops.h currently exposes attention parameter types declared in the kernel
  # headers, so the test TU still needs device keywords. It is compiled by nvcc
  # until those parameter structs move to attention_params.cuh.
  set_source_files_properties(tests/unit/test_m1.cc PROPERTIES LANGUAGE CUDA)
  kops_add_test(kops_test_m1
    SOURCES tests/unit/test_m1.cc
    LIBS kops_cuda)
  set_source_files_properties(tests/unit/test_model_ops.cc PROPERTIES LANGUAGE CUDA)
  kops_add_test(kops_test_model_ops
    SOURCES tests/unit/test_model_ops.cc
    LIBS kops_cuda)
  set_source_files_properties(tests/unit/test_activation.cc PROPERTIES LANGUAGE CUDA)
  kops_add_test(kops_test_activation
    SOURCES tests/unit/test_activation.cc
    LIBS kops_cuda)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_embedding.cc")
    set_source_files_properties(tests/unit/test_embedding.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_embedding
      SOURCES tests/unit/test_embedding.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_position.cc")
    set_source_files_properties(tests/unit/test_position.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_position
      SOURCES tests/unit/test_position.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_sampling.cc")
    set_source_files_properties(tests/unit/test_sampling.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_sampling
      SOURCES tests/unit/test_sampling.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_merge.cc")
    set_source_files_properties(tests/unit/test_merge.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_merge
      SOURCES tests/unit/test_merge.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_norm.cc")
    set_source_files_properties(tests/unit/test_norm.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_norm
      SOURCES tests/unit/test_norm.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_kvcache.cc")
    set_source_files_properties(tests/unit/test_kvcache.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_kvcache
      SOURCES tests/unit/test_kvcache.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_state.cc")
    set_source_files_properties(tests/unit/test_state.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_state
      SOURCES tests/unit/test_state.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_quantization.cc")
    set_source_files_properties(tests/unit/test_quantization.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_quantization
      SOURCES tests/unit/test_quantization.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_gemm.cc")
    set_source_files_properties(tests/unit/test_gemm.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_gemm
      SOURCES tests/unit/test_gemm.cc
      LIBS kops_cuda)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_moe.cc")
    set_source_files_properties(tests/unit/test_moe.cc PROPERTIES LANGUAGE CUDA)
    kops_add_test(kops_test_moe
      SOURCES tests/unit/test_moe.cc
      LIBS kops_cuda)
  endif()
endif()

# ---- install (static libs now; module/wheel targets later) ------------------
install(TARGETS kops_host kops_cuda
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
