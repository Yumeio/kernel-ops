# ayaka-kernel

## Overview

ayaka-kernel defines CUDA operator contracts for LLM inference. The C++ API is
framework-independent: an engine or future binding supplies tensor views, a
CUDA stream, device information, and workspace; host launchers return
`kops::Status`. The public C++ namespace remains `kops` (and its family
namespaces). The project name does not change those API names.

This checkout is at the **header-contract stage**. `csrc/` contains declarations
and data layouts, not kernel bodies or working launchers. `tools/header_check/`
checks declaration syntax and include prerequisites. `docs/TREE.md` includes a
proposed implementation tree; files without its `✓` mark are not present here.
There is currently no CMake/Make build, Python package, generated registry,
operator test suite, or GPU benchmark in this checkout. A declared kernel or a
model row in `docs/MODEL_COVERAGE.md` is not evidence that an operator runs.

The intended call path as implementation is added is:

```text
engine or binding (owns tensors, stream, workspace and cache)
  -> csrc/<family>/ops.h                 typed host API: validate, plan, run
  -> future .cc/.cu launcher            backend and exact-SM selection
  -> csrc/<family>/*_kernels.cuh         device signature and parameter contract
  -> future .cu / *_impl.cuh            compiled kernel body
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
- `csrc/common/macros.cuh` defines qualifiers, shape notation, exact-SM tags,
  feature gates, and `KOPS_ABI_VERSION`. Treat public struct layouts, enum
  values, kernel template parameters, argument order, and launcher signatures
  as ABI. Update the ABI version when changing one of those contracts.
- `csrc/common/status.h` defines the error model. Append new `Status` values;
  never renumber existing values. Host APIs report validation and launch
  failures through `Status` and the thread-local error detail.
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

`csrc/common/macros.cuh` currently declares SM 80, 86, 89, 90a, 100a, 103a,
120a, and 121a tags. These are compile-time contracts, not a list of built
modules or supported operators. General feature gates (`KOPS_HAS_*`) and
architecture-specific `a` feature gates are distinct. A kernel requiring
WGMMA, tcgen05/TMEM, or SM120 block-scaled instructions must be compiled for
the matching target; do not infer compatibility from a larger numeric SM.

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
```

`g++` can replace `clang++` in both commands. On Windows, use the installed
Python launcher in place of `python3` if necessary. The second command checks
that each header compiles with only its declared prerequisites. This checkout
has no `make test`, `make format`, wheel, CUDA sanitizer, or GPU test target;
add and document those commands when the corresponding build and tests exist.

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
