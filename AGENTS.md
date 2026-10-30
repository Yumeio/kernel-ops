# ayaka-kernel

## Overview

ayaka-kernel defines CUDA operator contracts for LLM inference. The C++ API is
framework-independent: an engine or future binding supplies tensor views, a
CUDA stream, device information, and workspace; host launchers return
`kops::Status`. The public C++ namespace remains `kops` (and its family
namespaces). The project name does not change those API names.

This checkout is at the **first-implementation stage (M1)**. `csrc/` has complete
declarations and data layouts for every family, and a first working slice:
`norm.cu` (`rms_norm`, `fused_add_rms_norm`), `activation.cu` (`act_kernel`,
`act_and_mul_kernel`), `attention/merge.cu` (`merge_state`), the host core
(`status.cc`, `dtype.cc`, `tensor.cc`, `registry.cc`) and the fp64 reference.
`CMakeLists.txt` builds `kops_host` + `kops_cuda` and `tests/unit/test_m1.cc`
runs numerics + error-contract checks on the local GPU. Everything else is still
a contract without a body; a declared kernel or a model row in
`docs/MODEL_COVERAGE.md` is not evidence that an operator runs. There is still no
Python package, generated registry table, torch binding, or benchmark harness.

The call path (implemented families follow it exactly):

```text
engine or binding (owns tensors, stream, workspace and cache)
  -> csrc/<family>/ops.h                 typed host API: validate, plan, run
  -> csrc/<family>/*.cu launcher         backend and exact-target selection
  -> csrc/<family>/*_kernels.cuh         device signature and parameter contract
  -> csrc/<family>/*_impl.cuh            compiled kernel body
```

Keep the host API, device ABI, and engine-owned resources separate. Do not copy
an extension layout, `torch.ops` registration, dispatch macros, or `make`
targets into this guide unless they are actually implemented here.

## Current repository layout

```text
csrc/common/             Status, dtype, tensor views, quantization, device helpers
csrc/{norm,activation,position,embedding,quantization,gemm,moe}/
csrc/{kvcache,attention,state,sampling,communication,model_ops}/
                         family kernel declarations and host `ops.h` APIs
csrc/execution/          persistent task contracts and launch/graph API
csrc/mega/               bounded megakernel plans and declarations
csrc/runtime/            capability registry and trace API declarations
csrc/{memory,load,backends,reference}/
                         host-only service contracts: allocator/sleep/IPC/greenctx,
                         checkpoint routing/repack, provider adapters, fp64 reference
tools/header_check/      host CUDA shim, canonical include order, Requires checker
docs/TREE.md             actual headers marked `✓`, proposed tree, catalog map,
                         and translation-unit include order
docs/MODEL_COVERAGE.md   model semantics representable by the headers
```

Read the directory tree before changing a family. One `ops.h` can expose
several operations; one `*_kernels.cuh` can declare several variants. Catalog
IDs and model coverage describe contracts, not binary availability. Use the
headers themselves as the source of truth for signatures and layouts.

## Header and ABI rules

- Library headers under `csrc/` use `#pragma once` and contain **no `#include`**.
  Headers with prerequisites document them in a `Requires:` comment. A `.cu` or
  `.cc` translation unit includes toolkit headers and project headers in the
  order recorded by `tools/header_check/check_all_headers.cc` and
  `docs/TREE.md`.
- `csrc/common/macros.cuh` defines qualifiers, shape notation, target ids
  (`KOPS_TARGET_*`), backend-neutral handle types, capability gates, and
  `KOPS_ABI_VERSION`. Treat public struct layouts, enum values, kernel template
  parameters, argument order, and launcher signatures as ABI. Update the ABI
  version when changing one of those contracts.
- `csrc/common/status.h` defines the error model. Append new `Status` values;
  never renumber existing values. Host APIs report validation and launch
  failures through `Status` and the thread-local error detail.
- Kernel source layout: one `.cu` per kernel group or per **named concern**
  (`host`, `dense`, `moe`, `chunk`, `recurrent`, `stats`, `select`, `sparse`,
  `lowbit`, `prep`). Ordinal suffixes (`_part2`, `_part3`, `_v2`) are banned:
  `tools/check_roadmap.py` rejects them and any `.cu` whose stem is not in
  `ROADMAP.md` Phụ lục E.2. If a translation unit exceeds ~90 s, move the
  architecture body into `<family>/<arch>_impl.cuh` before splitting anything.
- `csrc/common/tensor.h` defines non-owning `Tensor`, `LaunchContext`,
  `WorkspaceRequirements`, and `Workspace`. Family `ops.h` APIs use these
  types, not framework-specific tensors. The caller chooses the device and
  stream, reserves workspace, and owns its lifetime.
- A family `*_kernels.cuh` declares device parameters and `__global__`
  signatures; `ops.h` declares host entry points. Keep shapes, dtypes,
  strides, alignment, aliasing, mutations, and launch requirements explicit
  when changing either side. Add implementation files with matching
  declarations and focused validation.

The current host syntax check uses `cuda_host_shim.h` as a stand-in for CUDA
types and qualifiers. It does not compile device code, link launchers, or prove
numerical correctness.

## Ownership and execution contracts

- `kops::kv::PagedKVDesc` describes pages; the engine owns page allocation,
  refcounts, copy-on-write, prefix identity, and retirement. Kernels may touch
  only pages and slots named by the descriptor. Preserve the selected
  `KVLayout`, `PageTableFormat`, `KVCodec`, strides, scales, and `entry_bytes`;
  use `kv_physical_bytes` as the declared memory-accounting contract.
- Plan and workspace requirements are resolved before capture. `run()` APIs
  enqueue on `LaunchContext::stream` using caller-owned memory. Keep plan
  capacity, backing addresses, and `capture_generation` valid across CUDA
  Graph replay; replan or recapture after those resources change. Do not add
  allocation, device discovery, or JIT compilation to the replay path.
- Persistent and megakernel task graphs follow the progress rule in
  `csrc/execution/persistent.cuh`: a CTA must not wait on work that may never
  be scheduled. Preserve the declared acquire/release and epoch behavior when
  implementing queue counters or cross-CTA dependencies.
- `csrc/runtime/registry.h` declares capability status, provider, exact SM,
  predicates, mutations, and evidence. Its selection API allows automatic
  dispatch only among `kReleaseSupported` candidates. Do not mark a row as
  compiled, numerically validated, benchmarked, or release supported without
  the corresponding evidence and an implemented registry.

Numerical conventions are shared across families: quantization scales are
dequant multipliers (`x ≈ q * scale`); attention LSE is fp32 natural-log
log-sum-exp; fully masked rows produce zero output and `-inf` LSE; causal
attention aligns at the bottom right. Preserve mask, softcap, window, sink,
codec, and deterministic semantics when selecting or adding a variant.

## Architecture boundaries

`csrc/common/macros.cuh` declares target tags for NVIDIA SM 70, 75, 80, 86,
89, 90a, 100a, 103a, 120a, 121a and AMD gfx90a, gfx942, gfx950, gfx1100,
gfx1200. These are compile-time contracts, not a list of built modules or
supported operators. General feature gates (`KOPS_HAS_*`) and
architecture-specific `a` feature gates are distinct. A kernel requiring
WGMMA, tcgen05/TMEM, or SM120 block-scaled instructions must be compiled for
the matching target; do not infer compatibility from a larger numeric SM, and
do not infer one gfx target from another.

The public surface is backend-neutral: `kops_stream_t`, `kops_error_t`,
`kops_graph_t` and `KOPS_TMA_DESC` instead of CUDA types, `LaunchContext::arch_id`
(`KOPS_TARGET_*`) plus `wave_size` (32 CUDA / 64 CDNA) instead of `sm_version`,
and `Fp8Encoding` wherever an FP8 payload is described. `docs/ARCH_MATRIX.md`
records which target has which ISA capability and which Tier (A/B/C) it is in.
Anything that cannot run natively must go through the declared lowering ladder
in that document and report the effective recipe; never silently change
semantics.

The future build and registry may select different bodies or providers per
family and exact SM. Keep architecture-specific implementations in their
own paths when their instructions or algorithms differ. Add an architecture
to capability selection only after its kernels compile and run on that
hardware. Header syntax checks alone do not establish architecture support.

## Typical development workflow

1. Find the public host declaration in `csrc/<family>/ops.h` and the device
   signature or parameter struct in the same family. Follow referenced
   descriptors into `csrc/common/`, `csrc/kvcache/`, or another family.
2. Check `docs/TREE.md` for the catalog mapping and canonical include order.
   Check `docs/MODEL_COVERAGE.md` when changing model-facing semantics.
3. Change the smallest relevant contract. If a public signature or layout
   changes, search every use and update the ABI version as required.
4. If a new header or prerequisite is added, update its `Requires:` line and
   `tools/header_check/check_all_headers.cc` in dependency order. Update the
   actual/proposed distinction in `docs/TREE.md` when the tree changes.
5. Run the host declaration checks below. When implementations and tests
   exist, build the relevant exact SM and run focused numerical, graph,
   concurrency, and sanitizer tests for the changed behavior.

From the repository root, with a C++17 compiler and Python available:

```bash
clang++ -std=c++17 -fsyntax-only -Wall -Wextra -I. tools/header_check/check_all_headers.cc
python3 tools/header_check/check_requires.py --cxx clang++
clang++ -std=c++17 -fsyntax-only -Wall -Wextra -I. tools/header_check/abi_layout_test.cc
clang  -std=c11   -fsyntax-only -Wall -Wextra -I. bindings/c_api/kops_c_check.c
python3 tools/check_roadmap.py
```

The first three compile the declaration headers (the third pins every public struct
layout, the C mirror in `bindings/c_api/kops_c.h`, and the kernarg limit). The fourth
proves the C ABI header is self-contained. The last one checks that every declared
`__global__` kernel is assigned to exactly one ROADMAP step. `g++` can replace
`clang++`, and `gcc`/`cc` can replace `clang` for the C check.

Build and run the implemented kernels (needs CUDA toolkit; `native` targets the
local GPU, or pass an explicit exact-target list):

```bash
# Windows: vcvars64 + Ninja + ccache + ctest, one command.
tools\build-windows.bat

# Any platform, Ninja (required for ccache; the VS generator ignores it):
cmake -S . -B build -G Ninja -DKOPS_CUDA_ARCHS=native
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The test exits with the number of failed checks. It compares every implemented
kernel against `csrc/reference/` on device memory and asserts the `Status`
contract for bad dtype / shape / alignment.

### Build cache

`CMakeLists.txt` auto-detects `ccache` or `sccache` and sets both
`CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_CUDA_COMPILER_LAUNCHER`; pass
`-DKOPS_CCACHE_PROGRAM=<path>` when the binary is not on `PATH`, or
`-DKOPS_USE_CCACHE=OFF` to disable. This only works with the Makefile/Ninja
generators — CMake silently ignores the launcher with the Visual Studio
generator, which is why `tools/build-windows.bat` uses Ninja.

Measured on the dev machine (RTX 3050, CUDA 12.9, MSVC 19.44): cold build 15 s,
`ninja -t clean` + rebuild 4 s, 9/9 compilations served from cache (5 host
`.cc` + 4 CUDA `.cu`). Useful knobs:

```bash
ccache -M 5G          # cache size; also read from CCACHE_MAXSIZE
ccache -s             # hit/miss stats
```

`KOPS_BUILD_TYPE` defaults to `Release`. CMake >= 4 defaults single-config
generators to `Debug`, which makes kernels ~10x slower and adds MSVC `/Zi` that
ccache treats as uncacheable; use `-DKOPS_BUILD_TYPE=Debug` only when needed.
`KOPS_NVCC_THREADS` (default 4) passes `--threads` to nvcc for one TU.

## Adding functionality

- **New operation:** define its device parameters and kernel signature in the
  appropriate family header, then declare the host API in `ops.h`. Document
  inputs, outputs, mutation, workspace, unsupported cases, and exact-SM
  requirements. Add an implementation, registry row, and focused tests when
  those parts are introduced. A declaration alone stays `kPlanned`.
- **Existing operation:** preserve the shared `Tensor`/`LaunchContext`/
  `Workspace` contract and the family's layouts. Update callers and model
  mapping when semantics change. Do not silently substitute a backend with
  different numerical or mutation behavior.
- **New architecture or provider:** add target-specific code and predicates,
  then validate compilation, numerical behavior, graph/concurrency behavior,
  and performance on the target hardware before promoting capability status.
  Record the supporting evidence in the registry once it exists.

Keep project includes repository-relative in future translation units. Match
the style of adjacent headers and keep changes focused on the affected
contract. Treat `docs/TREE.md` as a design map: verify whether a path is
checked in before giving commands or editing it.

`@RTK.md`