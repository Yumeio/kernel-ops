# =============================================================================
# kernel-ops | cmake/instantiate.cmake
# Explicit instantiation is intentionally kept in the .cu translation unit:
# each kernel group owns its X-macro list, so the ABI and the compiled target
# set stay visible next to the kernel body. This helper exists for build files
# that need to attach generated instantiation lists later.
# =============================================================================

function(kops_add_instantiation_list target)
  # ARGN: optional generated .inc files. They are added to the target's include
  # path; no source file is generated per kernel.
  if(ARGC GREATER 1)
    target_include_directories(${target} PRIVATE ${ARGN})
  endif()
endfunction()
