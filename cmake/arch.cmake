# =============================================================================
# kernel-ops | cmake/arch.cmake
# Exact-target selection. "native" builds for the local GPU; an explicit list
# builds only those targets. The "a" suffix (sm_90a/sm_100a/sm_103a/sm_120a/
# sm_121a) is required for WGMMA/tcgen05/block-scaled instructions and is never
# inferred from a larger numeric SM.
# =============================================================================

set(KOPS_KNOWN_ARCHS 70 75 80 86 89 90a 100a 103a 120a 121a)

function(kops_set_cuda_archs archs)
  if("${archs}" STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "" FORCE)
    message(STATUS "kernel-ops: CUDA archs = native (local GPU)")
    return()
  endif()

  set(_normalized "")
  foreach(a IN LISTS archs)
    string(STRIP "${a}" a)
    if(NOT a IN_LIST KOPS_KNOWN_ARCHS)
      message(FATAL_ERROR
        "kernel-ops: unknown CUDA target '${a}'. Known: ${KOPS_KNOWN_ARCHS}.")
    endif()
    list(APPEND _normalized "${a}")
  endforeach()
  list(REMOVE_DUPLICATES _normalized)
  set(CMAKE_CUDA_ARCHITECTURES "${_normalized}" CACHE STRING "" FORCE)
  message(STATUS "kernel-ops: CUDA archs = ${_normalized}")
endfunction()
