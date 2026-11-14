# DeepEP baseline for the B12 EP cases

`benchmark/compare/deepep_baseline.py` runs the expert-parallel slice of
`bench_communication --dump-cases` through DeepEP v1.2.1 and writes the shared
baseline JSON that `compare_results.py` reads. It is dev-only: nothing here is
built, linked or shipped by CMake, and the numbers are comparisons, not registry
evidence.

- Upstream: <https://github.com/deepseek-ai/DeepEP>, tag `v1.2.1`
  (commit `9af0e0d`), MIT. The clone is pinned in `tools/dev/3rd_manifest.tsv`
  at `3rd/DeepEP`.
- NVSHMEM is **not** MIT: `csrc/kernels/ibgda_device.cuh` and
  `third-party/nvshmem.patch` are under the NVSHMEM SLA. Do not vendor them.

## Current status (2026-11-14)

**Not runnable on this machine.** The dev box has one `NVIDIA GeForce RTX 3050
Laptop GPU` (4 GiB, SM86, no P2P, single GPU), `torch 2.14.1+cu126`, Python
3.13.5. `deep_ep` is not installed, NVSHMEM is not installed, and an EP
baseline needs at least two ranks.

The script therefore takes its documented skip path:

```console
$ python benchmark/compare/deepep_baseline.py \
    --cases out/b12/cases.json --json out/b12/deepep-comm.json
deepep baseline skipped: needs >= 2 CUDA devices with NVLink, found 1
(NVIDIA GeForce RTX 3050 Laptop GPU). DeepEP's intranode EP runs over
NVLink/IPC; see reports/deepep-baseline.md
$ echo $?
77
```

77 is the repository skip code (`tests/unit/harness.h`, `CTest SKIP_RETURN_CODE`),
same as `communication_torch.py`. `--world-size` defaults to 2 and must be in
`[2, 8]`; on a host with fewer visible GPUs than requested the same 77 is
returned, and if `deep_ep` cannot be imported the message names the build doc.
The same 77 comes back when the only selected EP rows need a buffer the host
cannot create (for example an LL-only `--only` selection on a build without
NVSHMEM): the JSON is still written with the reason in `environment`.

Everything except the GPU path is validated on this machine:

- `python -m py_compile benchmark/compare/deepep_baseline.py` (Python 3.10 and
  the repo venv's 3.13);
- an offline smoke test with a stub `Buffer` (run from the session scratch dir,
  not committed) checks that the closures pass the shapes DeepEP v1.2.1 asserts
  (`[T, K]` int64 top-k, `[T, K]` fp32 weights, `[T, H]` BF16 x,
  `num_worst_tokens = T*world`, LL `[E/W, W*max_tokens, H]`);
- the case filter/world filter and buffer-size math are checked against the
  five real EP ids.

The C++ side currently reports the five EP cases with **no baseline**
(`docs/BENCH_RTX5060.md` §12, `ep | 5 | 0`):

| case | kops ms (1x3050, virtual ranks) |
|---|---:|
| `ep_layout_bf16_e64_t1024_k8` | 0.0062 |
| `ep_dispatch_combine_bf16_normal_w2_t128_h2048_e32_k4` | 0.5051 |
| `ep_dispatch_combine_bf16_normal_w4_t256_h4096_e64_k8` | 5.1592 |
| `ep_dispatch_combine_bf16_normal_w8_t64_h1024_e16_k4` | 1.2756 |
| `ep_dispatch_combine_bf16_ll_w4_t256_h4096_e64_k8` | 18.0247 |

Those kops numbers use 2/4/8 **virtual ranks on one GPU** with NVLink replaced
by device memory, so they cannot be compared to a real DeepEP run. They are
listed only to show what the baseline will be compared against once both sides
run on the same NVLink host.

## Hardware needed

One run drives one `--world-size`; repeat the run on the same host to collect
another world size.

| host | `ep_layout` | normal w2 | normal w4 | normal w8 | low-latency w4 |
|---|:---:|:---:|:---:|:---:|:---:|
| 2x A100 NVLink | yes | yes | skip | skip | no (SM80 build has no LL kernels) |
| 4x H100 NVLink | yes | skip | yes | skip | yes |
| 8x H100 NVLink | yes | skip | skip | yes | no (needs `--world-size 4`) |

All four dispatch cases are intranode (`num_ranks <= 8`), so "NVLink" means
`nvidia-smi nvlink -s` shows an active link between every pair in the group.
`Buffer.__init__` calls `check_nvlink_connections`, which rejects PCIe-only
groups larger than 2 (`3rd/DeepEP/deep_ep/utils.py:66`).

DeepEP's low-latency kernels are SM90-only. `setup.py:56-65` asserts
`DISABLE_SM90_FEATURES` **implies** `DISABLE_NVSHMEM`, and it is the A100 path;
so an A100 build cannot run `low_latency_dispatch` at all, even with NVSHMEM
installed. That is why the A100 column has no LL entry.

## Build NVSHMEM 3.3.9+

DeepEP v1.2.1 requires NVSHMEM 3.3.9 or later
(`3rd/DeepEP/third-party/README.md`). Either the tarball, the deb/rpm packages,
conda-forge, or `pip install nvidia-nvshmem-cu12`.

The low-latency path needs IBGDA. Pick one of the two supported setups:

1. Driver regkeys (traditional IBGDA), in `/etc/modprobe.d/nvidia.conf`:

   ```text
   options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
   ```

   then `sudo update-initramfs -u && sudo reboot`.

2. GDRCopy + the `gdrdrv` kernel module (CPU-assisted IBGDA). Slightly slower,
   usable when the driver regkeys cannot be changed.

Then:

```bash
export NVSHMEM_DIR=/opt/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH}"
export PATH="${NVSHMEM_DIR}/bin:${PATH}"
nvshmem-info -a    # must print the device list and version
```

Notes from the B12 harness side:

- `deep_ep.Buffer` sets `NVSHMEM_IB_ENABLE_IBGDA=1`, `NVSHMEM_IBGDA_NUM_RC_PER_PE`
  (= experts/world), `NVSHMEM_MAX_TEAMS=7`, `NVSHMEM_DISABLE_NVLS=1` and
  `NVSHMEM_CUMEM_GRANULARITY=2^29` before sync, matching DeepEP's own tests.
  Do not override them from the shell.
- One NVSHMEM instance is initialized per `Buffer` when `num_rdma_bytes > 0`.
  The normal buffer in this script uses `num_rdma_bytes=0`, so a host can run
  the normal cases **without** NVSHMEM at all.
- `check_nvlink_connections` needs `pynvml` only on PCIe-named devices; on
  A100/H100 SXM it returns immediately.

## Build DeepEP v1.2.1

The checkout already carries the tag; confirm it, then build in place:

```bash
cd 3rd/DeepEP
git describe --tags --always        # v1.2.1
git rev-parse --short HEAD          # 9af0e0d
```

**H100 (SM90, normal + low-latency):**

```bash
export NVSHMEM_DIR=/opt/nvshmem
export TORCH_CUDA_ARCH_LIST="9.0"
export DISABLE_AGGRESSIVE_PTX_INSTRS=0   # tuned path; 1 (the default) is the safe path
python setup.py bdist_wheel
pip install dist/*.whl              # install.sh does the same
```

`DISABLE_AGGRESSIVE_PTX_INSTRS` defaults to `1` on every arch
(`setup.py:79-81`), which disables the `.L1::no_allocate` load/store path; the
H800 numbers in DeepEP's README come from the `0` build. That path relies on
an undefined-behavior PTX read (README, "Undefined-behavior PTX usage"), so
record which value was used next to the result.

**A100 (SM80, normal only):**

```bash
unset NVSHMEM_DIR                    # and make sure the nvidia.nvshmem pip module is absent
export DISABLE_SM90_FEATURES=1
export TORCH_CUDA_ARCH_LIST="8.0"
python setup.py bdist_wheel
pip install dist/*.whl
```

`setup.py` finds NVSHMEM either through `NVSHMEM_DIR` or through the
`nvidia.nvshmem` wheel, and `DISABLE_SM90_FEATURES=1` asserts that NVSHMEM is
disabled (`setup.py:56-65`). If either one is present the build stops with an
assertion instead of silently producing an A100 wheel with SM90 kernels.

Verify and run DeepEP's own tests before touching the baseline:

```bash
python -c "import deep_ep; print(deep_ep.__file__, deep_ep.Buffer.is_sm90_compiled())"
cd 3rd/DeepEP && python tests/test_intranode.py --num-processes 2   # normal path
python tests/test_low_latency.py --num-processes 4                  # H100 only
```

`is_sm90_compiled()` is recorded in the JSON's `environment` object, together
with the `deep_ep` module path and `num_sms` when overridden.

## Running the baseline

```bash
build/bench/benchmark/bench_communication --dump-cases > out/b12/cases.json
build/bench/benchmark/bench_communication --mode eager --json out/b12/kops-comm.json

# 2x A100/H100: layout + the w2 normal row; w4/w8 rows land in "skipped".
python benchmark/compare/deepep_baseline.py \
    --cases out/b12/cases.json --json out/b12/deepep-comm.json \
    --world-size 2 --calls 8 --groups 8

python benchmark/compare/compare_results.py \
    out/b12/kops-comm.json out/b12/deepep-comm.json
```

On 4x H100 add `--world-size 4` to pick up the w4 normal and LL rows; on 8x
H100 `--world-size 8` picks up the w8 normal row instead. A single run only
drives one world size, so a full matrix is three runs (the kops side is
world-agnostic because it uses virtual ranks).

Options beyond the required `--cases/--json/--world-size/--calls/--groups/--warmup`:

| flag | meaning |
|---|---|
| `--only`, `--skip` | comma-separated case-id substrings, applied before the EP filter |
| `--num-sms N` | `Buffer.set_num_sms(N)` and the matching buffer-size hints (default: DeepEP's 20) |
| `--max-dispatch-tokens N` | low-latency per-rank capacity; 0 uses the case's token count, rounded up to a multiple of 4 |
| `--no-worst-tokens` | normal dispatch host-syncs the receive count instead of allocating `[T*world, H]` |
| `--no-nvlink-ll` | low-latency traffic over RDMA only (`allow_nvlink_for_low_latency_mode=False`) |

Output schema (the subset `compare_results.py` reads, plus DeepEP metadata):

```json
{"schema": 1, "backend": "deepep", "torch": "...", "device": {...},
 "protocol": {"calls": 8, "groups": 8, "warmup": 16, "world_size": 2,
              "mode": "eager", "worst_tokens": true},
 "environment": {"ok": true, "reason": "", "deepep": ".../deep_ep/__init__.py",
                 "version": "1.2.1+9af0e0d", "sm90_compiled": true},
 "notes": ["..."],
 "results": [{"id": "...", "op": "...", "dtype": "bfloat16", "rows": 128,
              "cols": 2048, "bytes": 1048576, "flops": 0, "median_ms": 0.0,
              "p10_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0,
              "gbs": 0.0, "tflops": 0.0, "world": 2, "note": "..."}],
 "skipped": [{"id": "...", "note": "..."}]}
```

## API mapping

DeepEP v1.2.1, `deep_ep/buffer.py` (the signatures this script calls; the
line numbers are from the pinned checkout):

| kops case | DeepEP call | Layouts |
|---|---|---|
| `ep_layout_bf16_e64_t1024_k8` | `Buffer.get_dispatch_layout(topk_idx, 64)` (`buffer.py:276`) | in `[1024, 8]` int64; out `[ranks]` int32, `[ranks]` int32 or `None`, `[64]` int32, `[1024, ranks]` bool |
| `ep_dispatch_combine_..._normal_...` | `Buffer.dispatch(x, topk_idx=..., topk_weights=..., num_tokens_per_rank=..., num_tokens_per_rdma_rank=..., is_token_in_rank=..., num_tokens_per_expert=..., num_worst_tokens=T*world)` then `Buffer.combine(recv_x, handle, topk_weights=recv_topk_weights)` (`buffer.py:305`, `:387`) | x `[T, H]` bf16; top-k `[T, K]` int64/fp32 with `-1`; recv `[T*world, H]` bf16 with `num_worst_tokens` |
| `ep_dispatch_combine_..._ll_...` | `Buffer.low_latency_dispatch(x, topk_idx, max_tokens, experts, use_fp8=False)` then `Buffer.low_latency_combine(recv_x, topk_idx, topk_weights, handle, out=out)` (`buffer.py:530`, `:600`) | recv `[E/world, world*max_tokens, H]` bf16 (masked); combine output `[T, H]` bf16 |

Matching details:

- **top-k**: `torch.topk` over random scores, `~1/13` of the entries replaced
  with `-1`, the same mask the C++ table applies
  (`bench_communication.cu:523-528`). DeepEP accepts `-1` on both paths.
- **layout world**: `ep_layout` carries no `_wN` (the C++ case runs one virtual
  rank), so this script runs it at `--world-size` and `is_token_in_rank` is
  `[T, world]`. The kernel is `T*K`-bound, but record the world next to the row.
- **weights**: one fp32 weight per top-k entry; dispatch returns the received
  weights and combine reduces with them, matching `ep_combine`'s
  `weights_already_applied=false` contract.
- **normal layout**: the C++ case sizes `recv_x` as `[T*world, H]` and passes
  `max_tokens_per_rank = cap_rows`; this script passes the same size through
  `num_worst_tokens`, which also removes DeepEP's implicit CPU wait and makes
  the timed region a pure GPU round trip. `--no-worst-tokens` gives the
  host-synced path (`num_worst_tokens=0`) for a service-like measurement.
- **low-latency layout**: `num_max_dispatch_tokens_per_rank` is the per-rank
  token count (`T`, rounded up to a multiple of 4; DeepEP asserts
  `num_ranks * max_tokens % 4 == 0`), because DeepEP requires
  `x.size(0) <= num_max_dispatch_tokens_per_rank`. `num_qps_per_rank` is
  `experts/world`; a buffer is built per distinct value.
- **hidden**: DeepEP requires `H % 128 == 0` for low-latency and
  `H*2 % 16 == 0` for normal; the four cases are 1024/2048/4096, so both hold.
  A future case that violates them is skipped with a reason, not silently
  altered.
- **FP8**: not used. The case ids are `bf16`, and `use_fp8=False` keeps the
  combine input BF16, matching the C++ table. DeepEP's FP8 dispatch would be a
  different case (and a different `bytes`).

## Protocol

`rank_time` follows `communication_torch.py`: `--calls` dispatch+combine round
trips per sample between one pair of CUDA events, `--groups` samples, each
sample reduced across ranks with `all_reduce(MAX)` **outside** the events, then
median/p10/p90/p99/min. `--warmup` round trips are issued before the first
event. A `dist.barrier()` + `torch.cuda.synchronize()` brackets every case.

Per case, outside the timed region:

1. allocate `x` (`[T, H]` bf16), top-k indices and weights;
2. call `get_dispatch_layout` (normal) or nothing (LL) once;
3. allocate the LL combine `out`.

Inside the timed region, one call = `dispatch` + `combine` (or
`low_latency_dispatch` + `low_latency_combine`). DeepEP allocates its receive
tensors on every dispatch, so allocator time is included; DeepEP's own
`tests/utils.py::bench` also times the whole call, allocations included.

## Byte accounting

`bytes` is copied from the kops case table so the ratio in
`compare_results.py` is a time ratio with identical metadata:

| case | kops `bytes` (rows x H x 2) | DeepEP allocation |
|---|---:|---:|
| normal w2 t128 h2048 | 1 MiB (256 rows) | 1 MiB (`num_worst_tokens=256`) |
| normal w4 t256 h4096 | 8 MiB (1024 rows) | 8 MiB (`num_worst_tokens=1024`) |
| normal w8 t64 h1024 | 1 MiB (512 rows) | 1 MiB (`num_worst_tokens=512`) |
| ll w4 t256 h4096 | 16 MiB (2048 rows = `(E/W) * ceil(T*K/(E/W))`) | 128 MiB (`(E/W) * W*T` rows) |

The LL row is the one exception: the C++ case uses an exact-capacity
`[E/W, cap, H]` buffer, while DeepEP's masked layout is
`[E/W, W*num_max_tokens, H]`, 8x larger for this case. The receive kernel only
touches the valid rows, so the time is still the real DeepEP LL round trip, but
`gbs` for that row is computed from the kops `bytes` and must not be read as
DeepEP's achieved bandwidth. The JSON `note` says so explicitly.

## Reading the result

`compare_results.py` computes `ratio = best_baseline_median / kops_median`, so
`ratio > 1` means kops is faster. For the B12 EP gate:

- `ep_layout`: compare `ep_layout_bf16_e64_t1024_k8` only; both sides run one
  layout kernel, so this should be near 1x on the same GPU.
- normal: the w2/w4/w8 rows are prefill-shaped. kops is expected to win the
  small-message rows on launch overhead and lose bandwidth at w4 unless the
  custom all-to-all matches DeepEP's NVLink chunking. Record the transport
  (NVLink vs the virtual-rank memory path) next to every ratio.
- low-latency: only meaningful on H100+. DeepEP's LL path is the hardest
  reference in the tree (NVSHMEM/IBGDA, two-buffer masked layout); a kops ratio
  below 1.0x here is the expected starting point and the gate to close.
- Do not quote a ratio from the 3050 virtual-rank numbers. The C++ EP rows and
  this baseline must be measured on the same NVLink host, same clocks, same
  `--calls/--groups`, and both eager (`bench_communication --mode eager`).

## What was not done

- NVSHMEM and DeepEP were **not** built on the 3050; that build is impossible
  for the LL path (SM86) and pointless for the normal path (one GPU).
- No A100/H100 host was available, so no DeepEP number is recorded in this
  report. The first real run should also record `nvidia-smi nvlink -s`,
  clocks and driver, per the fairness checklist in `benchmark/compare/README.md`.
- `thirdparty.py`, `run_thirdparty.py` and `thirdparty_matrix.py` are untouched;
  wiring DeepEP into `run_thirdparty.py` should reuse this script's CLI and
  needs a hardware gate like the `communication_torch` one
  (`run_thirdparty.py:452`).
