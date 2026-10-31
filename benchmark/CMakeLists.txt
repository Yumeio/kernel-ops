# =============================================================================
# kernel-ops | benchmark/CMakeLists.txt
# C++ benchmark executables (enabled with -DKOPS_BUILD_BENCHMARKS=ON). They link
# kops_cuda only: no torch, no CUTLASS. The PyTorch comparison scripts live in
# benchmark/compare/ and are run by hand, never built here.
# =============================================================================

# ---- provenance stamped into every result JSON ------------------------------
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE KOPS_BENCH_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT KOPS_BENCH_COMMIT)
  set(KOPS_BENCH_COMMIT "unknown")
endif()

execute_process(
  COMMAND git status --porcelain
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE KOPS_BENCH_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(KOPS_BENCH_STATUS STREQUAL "")
  set(KOPS_BENCH_DIRTY 0)
else()
  set(KOPS_BENCH_DIRTY 1)
endif()

set(KOPS_BENCH_FLAGS
  "${CMAKE_BUILD_TYPE};${CMAKE_CUDA_ARCHITECTURES};${CMAKE_CUDA_FLAGS};"
  "${CMAKE_CXX_FLAGS};${KOPS_CUDA_OPTIONS}")
string(SHA256 KOPS_BENCH_FLAGS_HASH "${KOPS_BENCH_FLAGS}")

add_library(kops_bench_provenance INTERFACE)
target_compile_definitions(kops_bench_provenance INTERFACE
  KOPS_BUILD_COMMIT="${KOPS_BENCH_COMMIT}"
  KOPS_BUILD_DIRTY=${KOPS_BENCH_DIRTY}
  KOPS_BUILD_FLAGS_HASH="${KOPS_BENCH_FLAGS_HASH}"
  KOPS_BUILD_ARCHS="${CMAKE_CUDA_ARCHITECTURES}"
  KOPS_TOOLKIT_VERSION="${CUDAToolkit_VERSION}")

function(kops_add_bench name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE kops_cuda kops_bench_provenance)
  kops_set_cuda_options(${name})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark")
endfunction()

kops_add_bench(bench_model_ops bench_model_ops.cu)
kops_add_bench(bench_norm bench_norm.cu)
kops_add_bench(bench_activation bench_activation.cu)
kops_add_bench(bench_merge bench_merge.cu)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_embedding.cu")
  kops_add_bench(bench_embedding bench_embedding.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_position.cu")
  kops_add_bench(bench_position bench_position.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_sampling.cu")
  kops_add_bench(bench_sampling bench_sampling.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_chains.cu")
  kops_add_bench(bench_chains bench_chains.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_kvcache.cu")
  kops_add_bench(bench_kvcache bench_kvcache.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_state.cu")
  kops_add_bench(bench_state bench_state.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_quantization.cu")
  kops_add_bench(bench_quantization bench_quantization.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_gemm.cu")
  kops_add_bench(bench_gemm bench_gemm.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_moe.cu")
  kops_add_bench(bench_moe bench_moe.cu)
endif()

message(STATUS "kernel-ops: benchmark targets -> ${CMAKE_BINARY_DIR}/benchmark")
