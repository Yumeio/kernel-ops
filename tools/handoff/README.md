# Handoff kits

A handoff kit packages the C++ tests, the C++ benchmarks and the golden manifest
for one **exact** GPU architecture, so a borrowed H100/H200 (`sm90a`) or
B200/GB200 (`sm100a`) can produce evidence without a rebuild on the target.
The dev machine (RTX 5060 Ti, SM120, 16 GB) cannot run either target; it only
cross-compiles and packages the kit.

There is no Python package, no torch import, no network access and no input
cache in the kit. A target run needs:

- Linux x86_64 with a GPU whose compute capability is exactly **9.0** or
  **10.0**;
- a driver new enough for that GPU (and for the toolkit used to build);
- a compatible CUDA runtime for the prebuilt binaries (`libcudart.so.<major>`);
- CMake/CTest 3.24+ on `PATH` (the kit contains no CTest installation);
- Python 3.8+ for `check_env.py` and `run_all.sh` (standard library only).

A CUDA toolkit with `nvcc` is required to rebuild on the target and is checked
when present. A runtime-only machine can still run the prebuilt kit; `check_env`
reports the missing toolkit as a warning, while the exact-SM cubin and `ldd`
checks verify that the binaries themselves are usable.

If glibc or the CUDA runtime does not match, rebuild from the same commit on the
target with the matching preset instead of forcing the prebuilt kit. The kit
manifest records the source commit and every file hash.

## Build a kit

From the repository root on the build machine:

```bash
bash tools/handoff/build_kit.sh sm90a      # -> out/handoff/sm90a/
bash tools/handoff/build_kit.sh sm100a     # -> out/handoff/sm100a/
```

The script configures and builds CMake preset `handoff-sm90a` or
`handoff-sm100a`. Those presets inherit the `base` preset and set
`KOPS_CUDA_ARCHS=90a` / `100a`, `KOPS_BUILD_TESTS=ON` and
`KOPS_BUILD_BENCHMARKS=ON`. `sm_90a` and `sm_100a` are exact targets: the `a`
suffix matters for `wgmma` / `tcgen05` instructions.

Rebuilding a kit replaces the whole `out/handoff/<arch>/` directory, including
its `results/`. Archive or copy previous evidence first, or write a new run to an
external `--json-dir`.

Useful options:

```bash
bash tools/handoff/build_kit.sh --arch sm100a --jobs 8
bash tools/handoff/build_kit.sh sm90a --golden /data/golden
bash tools/handoff/build_kit.sh sm90a --skip-configure --skip-build
```

Building does not need a GPU, but it does need a CUDA toolkit that can compile
the target (`compute_90a`, `compute_100a`) and Ninja. `ccache` is optional.

Only `out/handoff/<arch>/` is replaced, through a staging directory and a narrow
path guard. The script never deletes another path, never touches `build/`, and
never downloads anything. `out/` is gitignored.

## Kit layout

```text
out/handoff/sm90a/
  README.md              this document
  manifest.json          kit manifest: arch, preset, commit, file hashes
  check_env.py           preflight checks (stdlib only)
  run_all.sh             ctest + benchmarks + JSON summary
  CTestTestfile.cmake    generated test list for `ctest --test-dir`
  bin/
    kops_test_*          one executable per test family
    bench_*              one executable per benchmark family
  golden/
    manifest.json        copied from tests/golden when present
    manifest.schema.json
    cases/*.kopsgold     output-only golden files (small cases)
  results/               created empty; filled by run_all.sh
```

`manifest.json` at the kit root is **not** the golden manifest. It records the
source commit, dirty flag, host, toolkit, build type, exact arch, executables,
golden counts and SHA-256 of every file in the kit. `check_env.py` verifies it.

Archive and transfer:

```bash
tar -C out/handoff -czf out/handoff-sm90a.tar.gz sm90a
# on the target:
mkdir -p /data/kits && tar -C /data/kits -xzf handoff-sm90a.tar.gz
cd /data/kits/sm90a
```

## Preflight on the target

```bash
cd /data/kits/sm90a
./check_env.py --arch sm90a --kit .
```

`check_env.py` checks, in order:

1. kit manifest schema, arch, clean/dirty source and file hashes;
2. `nvidia-smi`: GPU name, compute capability, VRAM and driver; compute
   capability must be exactly 9.0 (`sm90a`) or 10.0 (`sm100a`). The driver floor
   also accounts for the CUDA major recorded in the kit manifest;
3. `nvcc`: release, exact `compute_90a` / `compute_100a` target support and a
   minimum version (12.0 for sm90a, 12.8 for sm100a);
4. `ctest` on `PATH`;
5. test/benchmark executables, their exact-SM cubin (`cuobjdump --list-elf`) and
   shared-library resolution (`ldd`);
6. the golden manifest, ready-case hashes and `KOPSGOLD` headers.

Every line is `PASS`, `WARN` or `FAIL`. A `WARN` does not fail the run; a `FAIL`
exits nonzero. Use `--require-golden` to turn a missing golden manifest into a
hard failure, and `--skip-gpu` only when validating the package on the build
machine. `--json out.json` writes the full report.

A green preflight is necessary but not sufficient: the kit still has to be built
from a clean commit and run on the exact GPU.

## Run everything

```bash
cd /data/kits/sm90a
./run_all.sh --arch sm90a --kit . --json-dir results
```

`run_all.sh`:

- exports `KOPS_GOLDEN_DIR=<kit>/golden` (or `--golden DIR`);
- runs `ctest --test-dir <kit>` and stores `ctest-<arch>.log`, `.xml` (JUnit) and
  `.json`;
- runs every `bin/bench_*` with `--json results/<arch>-<family>.json` and keeps
  a log per benchmark;
- writes `results/summary-<arch>.json` with the aggregate result, per-test
  status and per-benchmark exit code;
- prints a short summary and exits 0 only when CTest and every benchmark pass.

If `ctest` or `CTestTestfile.cmake` is missing, the script falls back to running
the test executables directly and records `mode: "direct"` in the JSON. That is
diagnostic evidence only; install CTest for a normal run.

Options: `--json-dir DIR`, `--golden DIR`, `--jobs N`, `--no-ctest`,
`--no-bench`. A relative `--json-dir` is resolved against the kit root. The
default is `<kit>/results/`.

Tests whose exact SM, GPU count, VRAM or golden data is unavailable return 77;
CTest records them as **Skipped**, not passed. The JSON keeps the skipped count
separate. A kit that skips a case does not produce evidence for it.

Benchmark JSON is emitted by `benchmark/harness.h` and carries provenance
(`commit`, `dirty`, `flags_hash`, `archs`, `toolkit`) plus the device name,
compute capability and per-case median/p10/p90. Attach the `results/` directory
or a tar of it to the PR; do not commit it.

## Golden data

The format and manifest rules are in
[`tests/golden/README.md`](../../tests/golden/README.md) in the repository
(inside a kit the same file is `golden/README.md`). In short:

- `tests/golden/manifest.json` indexes cases; ready cases have a real
  `.kopsgold` file and a matching `sha256`.
- `on_demand_input` cases regenerate inputs from `seed`; `stored_output` cases
  carry output/LSE only. Both are output-only on disk.
- 1M-token cases are output-only. Their KV cache is regenerated on the target;
  no 656 MB input cache is committed, copied or shipped in a kit.
- Without `KOPS_GOLDEN_DIR`, golden-dependent tests return 77. With a manifest
  that contains only `planned` cases, they also return 77.

If a case is still planned at handoff time, its test is a known skip and must be
reported as such; do not count it as validation.

## Exact-architecture rules

| Kit | Compute capability | GPUs | Not this kit |
| --- | ---: | --- | --- |
| `sm90a` | 9.0 | H100, H200, H20 | A100 (8.0), L40S (8.9) |
| `sm100a` | 10.0 | B200, GB200 | B300 (10.3), GB300, RTX 50 (12.0) |

Do not infer one target from a larger numeric SM, and do not reuse a `sm120a`
build (the dev GPU) for either kit. The preflight checks the exact compute
capability and the cubin architecture.

## Troubleshooting

| Symptom | Cause / action |
| --- | --- |
| `CMake preset 'handoff-sm90a' is not defined` | The handoff presets are missing from `CMakePresets.json`; add them before packaging. |
| `no test executables` | The build failed or `KOPS_BUILD_TESTS=OFF`; inspect the CMake output and rerun without `--skip-build`. |
| `compute_90a is missing` | The toolkit is too old for the exact target; install a newer CUDA toolkit. |
| `gpu.arch FAIL` | Wrong GPU for this kit (for example an RTX 50 is 12.0, not 10.0). Use the matching kit. |
| `kit.cubin FAIL` | The executable was built for another arch; rebuild with the matching handoff preset. |
| `libcudart.so.* => not found` | Install the matching CUDA runtime or rebuild on the target. |
| Golden tests skipped | No manifest, or only `planned` cases. Check `golden/manifest.json`; `--require-golden` makes this hard-fail in check_env. |
| `ctest` reports zero tests | `CTestTestfile.cmake` missing from the kit root; rebuild the kit. |

## What is deliberately absent

- No `torch`, no Python bindings, no PyTorch baseline.
- No network download, no package install, no corpus.
- No model weights and no input/KV caches; only output/LSE golden files.
- No multi-GPU kit yet. B12 and multi-node EP need a separate `multi` kit with
  NVLink/NVSwitch and real peer memory; virtual ranks on one GPU are not a
  substitute.
