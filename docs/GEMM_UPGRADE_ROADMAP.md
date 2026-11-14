# Lộ trình nâng cấp GEMM / tensor-core

Trạng thái: **đề xuất, chờ thực thi**. Ngày: 2026-11-14.
Phạm vi: `csrc/gemm/` và các consumer trực tiếp (`attention/`, `moe/`, `mega/`,
`model_ops/`).
Liên quan: `ROADMAP.md` (§3.4 P4, Phụ lục E), `CONTRACTS.md`, `registry/capabilities/`.

## 0. Quyết định đã chốt

1. **P1 (rút core) làm trước** mọi việc khác. Chỉ có một thân SM90a nên đây là thời
   điểm rẻ nhất để tách; sau khi có FP8/INT8/grouped thì mỗi lần sửa core phải sửa
   N thân.
2. **Blackwell: học + compile-only.** Máy hiện tại không có SM100/103, nên P6 gồm
   track đọc nguồn trong `3rd/`, dựng skeleton build được bằng preset
   `ci-sm100a`, và SASS probe. Không đánh dấu support khi chưa có evidence.
3. **Fast-accum là opt-in** qua `ScaledGemmArgs::fp8_accum` (field đã tồn tại từ
   ABI v7). Mặc định giữ `kPromoted` (fp32 đầy đủ, promote mỗi 128 K).

Ba quyết định này là ràng buộc của toàn bộ lộ trình bên dưới.

---

## 1. Tóm tắt điều hành

- FP8 GEMM hiện tại (đường `mma.sync`) đạt **~9 % peak** trên H100 SXM, tức khoảng
  **1/3** con số 30,3 % geomean của solution tham chiếu, và **~0,34× cuBLAS** đo
  cùng máy.
- Nguyên nhân là **thân kernel**, không phải epilogue hay hợp đồng: `mma.sync`
  4–8 warp + `cp.async` 64 B/hàng + 3–4 stage, không TMA, không warp
  specialization, không fast-accum.
- Đã có sẵn một **cell SM90a TMA + wgmma** (bf16/fp16) đang staged, compile sạch
  với `nvcc -arch=sm_90a` — đó là móng để rút core và mở rộng theo dtype.
- Lộ trình: giữ **băng decode M ≤ 16** và **bao ngoài** (hợp đồng, registry,
  bench), thay phần thân bằng **một core dùng chung + các cell theo arch**.
- KPI: FP8 4096³ **≥ 25 % peak (≥ 495 TF)**, BF16 **≥ 50 % (≥ 495 TF)**, decode
  không hồi quy, không mất shape nào đang chạy được.

---

## 2. Hiện trạng đo được

### 2.1 Số đo (H100 80GB HBM3, 132 SM, peak dense FP8 1979 TF)

| case | kops (`mma.sync`) | % peak | cuBLAS cùng máy | % peak |
|---|---|---|---|---|
| fp8 tensorwise m256 4096×4096 | 158,4 TF | 8,0 % | 505,2 TF | 25,5 % |
| fp8 tensorwise m1024 | 202,2 TF | 10,2 % | 548,7 TF | 27,7 % |
| fp8 tensorwise m4096 | 181,8 TF | 9,2 % | 541,0 TF | 27,3 % |
| **geomean 3 shape** | **179,9 TF** | **9,1 %** | — | — |
| bf16 4096³ (cuBLAS) | chưa đo | — | 364,9 TF | 36,9 % |
| fp8 GEMV m1 4096×4096 | 11,1 µs, 1509 GB/s | 45 % DRAM | 82,2 µs, 204 GB/s | 6 % DRAM |

Con số reference do người dùng cung cấp: **30,3 % geomean** trên H100 PCIe (peak
1513 TF), tương đương ~458 TF tuyệt đối.

Lệnh tái lập:

```bash
./build/handoff-sm90a/benchmark/bench_gemm --only gemm.fp8_mma --rounds 3
./build/handoff-sm90a/benchmark/bench_gemm --only gemm.fp8_gemv --rounds 2
```

### 2.2 Cảnh báo về phép đo

Máy này **không giữ clock ổn định**: `nvidia-smi` trong lúc chạy tensor liên tục
cho các mẫu 345 MHz → 1980 MHz; bf16 `matmul` chạy 6 s liên tục chỉ đạt 220 TF
(22 % peak). Vì vậy:

- Mọi báo cáo `% peak` phải kèm **anchor cuBLAS đo cùng phiên**.
- Không so số tuyệt đối giữa hai phiên khác nhau.
- Ghi `clocks.sm` đầu/cuối mỗi vòng vào JSON (thêm ở P0).

### 2.3 Đường chạy hiện tại

```text
fp8_gemm (csrc/gemm/fp8_gemm.cu)
  ├─ M ≤ 16, không tactic           -> fp8_gemv_kernel (CUDA core, bandwidth-bound)
  └─ M > 16 (hoặc có tactic)        -> launch_gemm_mma_sync
        heuristic_tactic (gemm_mma.cu:1441) trên bảng 10 tile cố định
        -> gemm_mma_sync_kernel<Atom,...>:
             cp.async 16 B, ring 3–4 stage (8 stage cho 16-bit small-M),
             smem XOR-swizzle, ldmatrix.x4,
             mma.sync.m16n8k32 (e4m3) / m16n8k16 (bf16) / m16n8k32.s8,
             fp32 accumulator đầy đủ, epilogue trực tiếp.
```

Điểm mạnh: độ phủ (tensorwise / rowwise / blockwise 1×128×128×128 / UE8M0),
epilogue hợp nhất (GLU, residual, quant-out, scatter, atomic), grouped/MoE
schedule, deterministic, hợp đồng `Status` đầy đủ.
Điểm yếu: mọi thứ chạy trên một thân `mma.sync` viết cho decode/độ phủ.

### 2.4 Đường của solution tham chiếu

```text
Python (chọn cfg theo M/N) -> CUTLASS 3.x GemmUniversal
  TMA load (128 B, swizzle 128 B) -> smem ring
  wgmma.mma_async m64nNk32, fp32 accum, FastAccum cho tensorwise/rowwise
  EVT: Sm90RowBroadcast(scale) x AccFetch -> bf16 round-to-nearest
  persistent scheduler + swizzle/raster, stream-K cho shape thiếu wave, PDL
  K % 16 != 0 -> pad_rows vào buffer stride-aligned (weight pad cache theo
                 (data_ptr, _version)); CUDA graph cache; Triton fallback.
```

### 2.5 Tài sản đã có trong repo (đang staged)

| file | nội dung | trạng thái |
|---|---|---|
| `csrc/gemm/gemm_wgmma.cu` (340 dòng) | kernel definition, bảng 5 tile, validation, encode tensor map, launch, entry `wgmma_gemm` | compile sạch `nvcc -arch=sm_90a -c` |
| `csrc/gemm/sm90a_impl.cuh` (608 dòng) | TMA/mbarrier/cluster, atom wgmma, `TileSched` raster L2, `dense_tile<At,Dt,BM,BN,ST,MC>` | như trên, body chỉ trong `__CUDA_ARCH__ == 900 && __CUDA_ARCH_FEAT_SM90_ALL` |
| `tests/unit/test_sm90.cc` | oracle fp64, tile invariance, contract, graph | đã wire vào CMake |
| `csrc/common/macros.cuh` | `KOPS_ABI_VERSION 8` | additive |
| `bindings/c_api/kops_c.h` | `KOPS_C_ABI_VERSION 8` | mirror |

Cell hiện tại: 5 tile `{64×128×8, 64×256×4, 128×128×6, 128×256×3, 256×128×3}`,
`GemmTmaParams` đã có `tma_sfa/tma_sfb/tma_c/tma_d`, `tile_counter`,
`streamk_partials` — tức hợp đồng cho block-scale, stream-K và TMA store đã sẵn.
Giới hạn host đang chặn: `batch == 1`, không split-K, không `swap_ab`, cluster
1×1, epilogue ngoài dense subset trả `kNotImplemented`, `Pingpong` chưa có body.

### 2.6 Ràng buộc K hiện tại

Cả hai đường đều chặn K lẻ, nhưng ở hai chỗ khác nhau:

| đường | điều kiện | status |
|---|---|---|
| GEMV (M ≤ 16, `fp8_gemm.cu`) | `K % 16 == 0` | `kUnsupportedShape` |
| tensor-core (M > 16, `gemm_mma.cu:1869`) | `K % (16 / elem_bytes) == 0` (fp8 → 16, bf16 → 8) | `kMisaligned` |
| blockwise | thêm `K % 128 == 0` | `kUnsupportedShape` |

Vì vậy K = 4127 hiện **không chạy được** ở bất kỳ đường nào. TMA cũng cần pitch
16 B nên ràng buộc này không tự mất đi khi đổi mainloop; P3-1 (`pad_rows`) là
cách duy nhất để phủ shape đó. P0-4 thêm test khẳng định đúng status hiện tại
để tránh hồi quy ngầm.

---

## 3. Nguyên tắc thiết kế

1. **Hợp đồng ngoài bất biến.** `gemm/ops.h`, `ScaledGemmArgs`, `GemmTactic`,
   `GemmTmaParams`, `Status`, `Workspace` chỉ được **thêm**; mọi thay đổi layout
   phải bump `KOPS_ABI_VERSION` + mirror C.
2. **Giữ băng decode.** M ≤ 16 là chỗ mình thắng cuBLAS 7×; không đụng vào trừ khi
   có lý do bandwidth.
3. **Một core, nhiều body.** TMA/pipeline/work-source/epilogue viết một lần; body
   khác nhau theo arch (`sm90a`, `sm100a`, `sm120a`) và theo dtype.
4. **Evidence trước auto-dispatch.** Registry chỉ cho chọn tự động giữa row
   `kReleaseSupported`; row đó chỉ mở khi có test + bench.
5. **Mỗi bước một commit có gate.** Không gộp refactor với feature.

---

## 4. Kiến trúc đích

### 4.1 Sơ đồ lớp

```text
contract:   gemm/ops.h, gemm_params.cuh, common/tensor.h, common/status.h
dispatch:   scaled_gemm / fp8_gemm / int8_gemm / fp4_gemm / ... (host)
            + registry-generated table (tools/gen_registry.py)
            + tactic cache / autotune
core:       gemm/gemm_tma.cuh       TMA encode 2D/3D, mbarrier, cluster, PDL
            gemm/gemm_pipeline.cuh  ring stages, producer/consumer role, setmaxnreg
            gemm/gemm_work.cuh      work source: dense/persistent/raster/grouped/splitk
            gemm/gemm_epilogue.cuh  acc -> store (direct/TMA) + fusion
bodies:     gemm/sm90a_impl.cuh     wgmma (k16 16-bit, k32 fp8/int8)
            gemm/sm100a_impl.cuh    tcgen05/TMEM (mới, P6)
            gemm/sm120a_impl.cuh    mma.sync block-scaled (giữ, nâng sau)
            gemm/gemm_mma.cu        sàn SM80/86 + epilogue đầy đủ + grouped fallback
```

### 4.2 Danh mục file

| file | hành động | ghi chú |
|---|---|---|
| `csrc/gemm/gemm_tma.cuh` | **mới** (P1) | chuyển từ `sm90a_impl.cuh`: `encode_tma_2d`, `tma_load_2d[_multicast]`, `tma_load_3d`, `tma_store_*`, `mbar_*`, `cluster_*`, `smem_u32`, `elect_one`, `bar_sync`, `pdl_wait` |
| `csrc/gemm/gemm_pipeline.cuh` | **mới** (P1/P2a) | ring + role; tách khỏi `dense_tile` |
| `csrc/gemm/gemm_work.cuh` | **mới** (P1) | từ `TileSched`; thêm interface cho grouped ở P4 |
| `csrc/gemm/gemm_epilogue.cuh` | **mới** (P1) | `pack2`, map fragment, `dev::gemm_epilogue`, TMA store staging, direct fallback |
| `csrc/gemm/sm90a_impl.cuh` | **sửa** | còn atom wgmma + `dense_tile` (orchestration); sau P2 thêm nhánh FP8/INT8 |
| `csrc/gemm/gemm_wgmma.cu` | **sửa** | host SM90a; thêm tile table theo dtype ở P2 |
| `csrc/gemm/fp8_gemm.cu` | **sửa** | dispatch FP8 sang cell SM90a; validation `fp8_accum` |
| `csrc/gemm/gemm_mma.cu` | **giữ** (P1–P3), **thu hẹp** (P7) | sàn portable + epilogue + grouped |
| `csrc/gemm/gemm_prep.cu` | **sửa** (P3) | thêm `pad_rows` |
| `csrc/runtime/registry.cc` | **giữ** | đọc bảng sinh từ YAML; auto-dispatch dùng `select()` |
| `registry/capabilities/gemm.yaml` | **sửa** | thêm row cho tile/recipe SM90a |
| `tests/unit/test_sm90.cc` | **sửa** | mở rộng theo dtype/batch/cluster/P3 |
| `benchmark/bench_gemm.cu` | **sửa** | case wgmma + shape sweep |

### 4.3 Hợp đồng core (phác thảo chữ ký)

```cpp
// gemm_tma.cuh  (device + host encode; Requires: common/*, gemm/gemm_params.cuh, cuda.h)
Status encode_tma_2d(KOPS_TMA_DESC*, CUtensorMapDataType, const void* base,
                     uint64_t inner, uint64_t outer, uint64_t stride_bytes,
                     uint32_t box_inner, uint32_t box_outer,
                     CUtensorMapSwizzle, const char* who);
Status encode_tma_3d(...);                       // batch (P1-C5)

__device__ void mbar_init(uint64_t*, uint32_t count);
__device__ void mbar_wait(uint64_t*, uint32_t phase);
__device__ void mbar_arrive_expect_tx(uint64_t*, uint32_t bytes);
__device__ void tma_load_2d(const KOPS_TMA_DESC*, uint64_t* bar, void* smem,
                            int32_t c0, int32_t c1);
__device__ void tma_load_2d_multicast(..., uint16_t cta_mask);
__device__ void tma_store_2d(const KOPS_TMA_DESC*, const void* smem, int32_t, int32_t);

// gemm_pipeline.cuh  (P2a)
template <int STAGES, int CONSUMERS> struct Pipeline;   // full/empty ring, phase, arrive
// gemm_work.cuh
struct WorkDense { bool next(uint32_t& mb, uint32_t& nb); bool multicast; bool peer_alive; };
struct WorkGrouped { ... };                              // P4
// gemm_epilogue.cuh
template <typename Dt, int BM, int BN, int NACC>
__device__ void store_tile(const float* acc, const GemmTmaParams& P,
                           uint32_t mb, uint32_t nb, uint8_t* smem_d, int lane);
```

Ràng buộc header: **không `#include`**, ghi `Requires:`, thêm vào
`tools/header_check/check_all_headers.cc` theo đúng thứ tự phụ thuộc. `*_kernels.cuh`
vẫn là device-only; không đưa host encode vào đó.

### 4.4 Dispatch và registry

- `tools/gen_registry.py` sinh bảng từ YAML; `runtime/registry.cc::select()` chỉ
  trả candidate `kReleaseSupported`.
- Hiện **không có row nào** ở trạng thái `release_supported` cho `gemm.yaml`, nên
  chưa có auto-dispatch thật. P1-C6 phải thêm row theo từng tile/recipe với
  evidence `test_sm90;bench_gemm`, rồi mới bật chọn tự động.
- Thứ tự ưu tiên khi chọn: tactic tường minh → registry auto (SM90a) → heuristic
  mma.sync hiện tại → SIMT.
- `autotune()` hiện chỉ biết tile `mma.sync`; P2a mở rộng để sweep cả tile SM90a.

### 4.5 Chính sách số học

| chế độ | recipe hợp lệ | cộng dồn | bất biến | ghi chú |
|---|---|---|---|---|
| `kPromoted` (default) | tensorwise, rowwise, blockwise, UE8M0 | fp32, promote mỗi 128 K | theo grid và theo tile | an toàn, dùng cho test/contract |
| `kFastAccum` (opt-in) | tensorwise, rowwise | scale-d=1 trong wgmma | theo grid; **khác** `mma.sync` | dùng để so peak với CUTLASS/cuBLAS |

`kFastAccum` + blockwise trả `kInvalidArgument` (cần promote để áp scale
128×128). Test phải đo `max|Δ|/|ref|` thực nghiệm cho K lớn nhất trong bộ case và
ghi ngưỡng vào test, không được giả định.

---

## 5. Lộ trình

### P0 — Baseline (0,5–1 ngày) — *nguyên liệu đã có, chưa chạy*

| ID | Việc | File | Gate |
|---|---|---|---|
| P0-1 | Chạy `test_sm90` + toàn bộ `ctest` trên H100 | — | xanh, log lưu `out/` |
| P0-2 | Thêm `% peak` + `% cuBLAS` + clock đầu/cuối vào JSON và report | `benchmark/harness.h`, `benchmark/compare/report.py` | report sinh được cột mới |
| P0-3 | Bảng peak theo GPU/dtype (H100 SXM 1979/989/1979; PCIe 1513/756/1513) | `benchmark/compare/report.py` | geomean `%peak` tính được |
| P0-4 | Test K = 4127 cho fp8 tensorwise (M > 16) và GEMV | `tests/unit/test_gemm.cc` | xác nhận hành vi thật |

### P1 — Rút core (6–8 ngày) — **đang thực hiện**

Mục tiêu: không đổi hành vi, không đổi số đo quá ±2 %, chuẩn bị chỗ cho FP8.

| ID | Việc | File | Gate | Est |
|---|---|---|---|---|
| P1-C0 | Thêm case bench `gemm.wgmma.{bf16,fp16}`: 5 tile × m ∈ {256, 1024, 4096}; dựng `GemmTactic` (tile_m/n, stages theo bảng, mainloop suy ra) và gọi `gemm_with_tactic` | `benchmark/bench_gemm.cu` | 30 case JSON; lưu `out/p1-baseline.json` | 0,5 d |
| P1-C1 | Tách TMA/mbarrier/cluster/PDL sang `gemm_tma.cuh`; giữ atom wgmma + `reg_*` + `stsm` trong cell | như trên | 5 lệnh header/ABI/roadmap xanh; bench ±2 % | 1 d |
| P1-C2 | Tách epilogue: `pack2`, map fragment → (row, col), gọi `dev::gemm_epilogue`, staging TMA store, fallback direct-store; tham số hoá `use_tma` | `gemm_epilogue.cuh` | tile invariance **bitwise**; test ép cả hai nhánh store cho cùng bytes | 1 d |
| P1-C3 | Tách work source: `TileSched` → `WorkDense::next()` (giữ nguyên raster nhóm 8/16 và multicast flags) | `gemm_work.cuh` | cùng bytes; grid-invariance (persistent vs static) | 1 d |
| P1-C4 | `dense_tile` còn orchestration: prologue → role split → loop{work, mainloop, epilogue} | `sm90a_impl.cuh` | `test_sm90` xanh; bench bằng C0 | 1 d |
| P1-C5 | Tensor map 3D + batch > 1 (blockIdx.z, tập `num_tiles_k` theo batch) | `gemm_tma.cuh`, `gemm_wgmma.cu` | test batch mới; numerics | 1 d |
| P1-C6 | Registry rows cho từng tile/recipe + `select()` cho auto-dispatch (chỉ sau khi có evidence) | `registry/capabilities/gemm.yaml`, `dense_gemm.cu` | bench cho thấy auto chọn wgmma; `mma.sync` vẫn reachable; `explain()` đúng | 1 d |
| P1-C7 | Cập nhật `CONTRACTS.md` (ràng buộc TMA, tile table, epilogue subset), `ROADMAP.md`, report | docs | `check_roadmap.py`, `abi_layout_test`, `kops_c_check` | 0,5 d |

**Thứ tự commit**: C0 → C1 → C2 → C3 → C4 → C5 → C6 → C7. C1–C4 là refactor
thuần, mỗi commit phải chứng minh “không đổi số”. C5–C6 là feature, bắt đầu được
đổi hành vi.

**Rollback**: mỗi commit C1–C4 độc lập; nếu C4 vượt gate, revert C4 và giữ
C1–C3 (3 header đã tách vẫn dùng được).

### P2 — Dense peak trên SM90a (10–14 ngày)

| ID | Việc | Gate | Est |
|---|---|---|---|
| P2a | Hoàn tất 16-bit: mở cluster (bỏ chặn host, dùng instantiation MC=1/2 đã có), ping-pong mainloop, stream-K, `swap_ab`, split-K | bf16 4096³ ≥ 50 % peak | 3–4 d |
| P2b | **FP8** `fp8_gemm_wgmma_kernel<Traits,kRecipe,kAccum>`: `tile_k = 128`, atom `m64nNk32`, SS operand, tensorwise/rowwise `kFastAccum`, blockwise `kPromoted` (scale A 1×128 TMA cột, B 128×128, UE8M0) | 4096³ ≥ 25 %; m1024 ≥ 18 %; m256 ≥ 12 % | 4–5 d |
| P2c | **INT8** cùng khung (`s8 k32`), azp ở epilogue | ≥ 25 % | 2 d |
| P2d | **FP4/MXFP4 → FP8**: dequant trong register (LUT/`prmt`), MX scale 32 trước wgmma | ngang hoặc hơn đường software hiện tại | 3 d |
| P2e | **Mixed-input W4A8** trên SM90a (Marlin giữ cho SM86) | không hồi quy | 3 d |

Tham chiếu đọc: `3rd/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`,
`.../mma/sm90.cuh` (descriptor arithmetic), `.../sched/gemm.cuh` (raster/persistent).

### P3 — Phủ shape (4–6 ngày)

| ID | Việc | Gate |
|---|---|---|
| P3-1 | `pad_rows` trong `gemm_prep.cu` + entry `gemm_pad_operands` trả `WorkspaceRequirements`; cache weight pad ở tầng engine/binding, **không cấp phát trong `run()`** | K = 4127 chạy qua cell SM90a |
| P3-2 | Bảng band M: 1–16 GEMV; 17–64 đo `wgmma` 64-row + mask vs `mma.sync swap_ab`; 65+ wgmma | chọn band có số đo, ghi vào dispatch |
| P3-3 | N lẻ, `ld` lạ (TMA box clamp + epilogue predicate), batch > 1 | không case nào `kUnsupportedShape` trong bộ eval |
| P3-4 | Shape sweep 12 case (gồm 4127, 14336, 8192) + geomean `%peak` | report có geomean |

### P4 — Grouped / MoE / LoRA / bmm (8–12 ngày)

- Work source grouped (contiguous/masked/taskmap) chạy trên mainloop mới, thay vì
  giữ `gemm_mma_sync_kernel` cho mid/large M.
- MoE fused gate&up + down dùng pixel pipeline; bmm (MLA absorption) thành
  mainloop batched; LoRA SGMV giữ decode.
- Gate: grouped FP8 blockwise parity với DeepGEMM; MoE không chậm hơn `mma.sync`.

### P5 — Attention cell SM90a (10–15 ngày)

- QK^T / PV / absorption bằng TMA + wgmma cho MLA decode/prefill/sparse.
- Decode band giữ `mma.sync`; metadata/merge không đổi.
- Gate: không hồi quy các số trong `docs/B11_B15_STATUS.md`
  (MLA 1M: 4,04× bf16, 3,85× FP8).

### P6 — Blackwell (không có HW)

| ID | Việc | Gate | Est |
|---|---|---|---|
| P6A | Study: `3rd/DeepGEMM/.../mma/sm100.cuh`, `impls/sm100_bf16_gemm.cuh`, `sm100_fp8_fp4_gemm_1d1d.cuh`, `sm100_bf16_mega_moe.cuh`, `sm100_mqa_logits.cuh`; FlashMLA sm100; `3rd/vllm/vllm/cute_utils/_tcgen05.py` và các kernel CuTe DSL. Đầu ra: `research_notes/sm100-tcgen05-kernel-study.md` + bảng map khái niệm → API core | tài liệu review được, chỉ rõ chỗ core thiếu | 5 d |
| P6B | Skeleton `sm100a_impl.cuh` + `gemm_tcgen05_kernel` instantiation, build `ci-sm100a`, runtime trả `kUnsupportedArch`; SASS probe kernel tham chiếu cùng toolkit (`cuobjdump -sass`, lấy mnemonic họ `UTC*` / `UTMA*` từ SASS tham chiếu, không tự đoán) | build xanh, SASS có mnemonic kỳ vọng | 3 d |
| P6C | Chỉ khi có HW: numerics, graph, bench, mở registry | row `kReleaseSupported` | — |

P6A chạy song song từ tuần 1, không chặn P2–P5.

### P7 — Thu hẹp và bằng chứng (2–3 ngày)

- `gemm_mma.cu` chỉ còn: sàn SM80/86, epilogue đầy đủ, grouped fallback.
- Xoá `heuristic_tactic` cho mid/large M trên SM89+; giữ bảng tile SM120a.
- Hợp nhất tactic cache + autotune thành bảng dispatch có evidence.
- Cập nhật `ROADMAP.md`, `CONTRACTS.md`, ABI, commit report.

---

## 6. Ma trận recipe × arch

| nhóm | hiện tại | đích |
|---|---|---|
| GEMV decode (fp8/fp4/int4/gguf/marlin/router) | CUDA core, 45 % DRAM | **giữ** |
| dense bf16/fp16 | mma.sync + cell SM90a (staged) | **P1–P2a**: cell SM90a + auto-dispatch |
| dense fp8 e4m3 | mma.sync, 9 % peak | **P2b**: wgmma k32 |
| dense int8 | mma.sync s8 | **P2c** |
| fp4 / mxfp4 / nvfp4 | GEMV + software decode | **P2d** (SM90a), **P6** (SM100 native) |
| mixed-input w4a8 | marlin | **P2e** (SM90a) |
| grouped / MoE | mma.sync + taskmap | **P4** |
| LoRA / dual / bmm | mma.sync + split-K | **P4** |
| attention decode/prefill/MLA | mma.sync | **P5** |
| mega | planned | **P6/P7**, xây trên core |
| SM100a/103a | chưa có | **P6** |
| SM120a | mma.sync block-scaled | sau P6, ưu tiên thấp |

---

## 7. Lịch và phụ thuộc

```text
tuần 1      P0 -> P1-C0..C4        (core tách xong, bf16 vẫn đúng)
tuần 2      P1-C5..C7 + P2a        (batch, registry, auto-dispatch, bf16 peak)
tuần 1-4    P6A song song
tuần 3-4    P2b (FP8) + P2c (INT8) -> con số 25 %+
tuần 5      P3 shape + pad
tuần 6-7    P4 grouped/MoE
tuần 8-9    P5 attention
sau đó      P6B/P6C, P2d/2e, P7
```

Đường găng: P1-C1→C4 → P2a → P2b → P3. P6A không nằm trên đường găng.

---

## 8. Quy trình đo

1. Build đúng preset (`dev-sm90a` / `bench-sm90a`), không dùng `native` cho
   `sm_90a` (CMake báo `90` thiếu hậu tố `a`).
2. Mỗi case: `--rounds 3`, lấy median; ghi `clocks.sm` đầu/cuối.
3. Chạy anchor cuBLAS cùng phiên:
   `python benchmark/compare/torch_baseline.py --cases ... --json ...`.
4. Tính `%peak` theo bảng peak của GPU đang chạy; tính `%cuBLAS` để tách yếu tố
   máy khỏi yếu tố kernel.
5. Geomean trên **bộ shape cố định** (định nghĩa ở P3-4), không đổi bộ giữa các
   lần báo cáo.
6. Không so số tuyệt đối giữa hai phiên vì clock không ổn định.

---

## 9. Checklist kiểm tra (mọi commit)

```bash
clang++ -std=c++17 -fsyntax-only -Wall -Wextra -I. tools/header_check/check_all_headers.cc
python3 tools/header_check/check_requires.py --cxx clang++
clang++ -std=c++17 -fsyntax-only -Wall -Wextra -I. tools/header_check/abi_layout_test.cc
clang  -std=c11   -fsyntax-only -Wall -Wextra -I. bindings/c_api/kops_c_check.c
python3 tools/check_roadmap.py
cmake --preset dev-sm90a && cmake --build --preset dev-sm90a && ctest --preset dev-sm90a
```

Với commit đụng `.cu` của cell: thêm `nvcc -std=c++17 -arch=sm_90a -c -o /dev/null`
(không dùng được `-fsyntax-only` với nvcc) để bắt lỗi template trước khi link.

---

## 10. Definition of Done cho một kernel row

Một row chỉ được nâng status khi có đủ:

1. **Numerics**: oracle fp64 trên operand đã lưu; tile invariance (mọi tile cùng
   bytes); với blockwise: thứ tự promote 128 K được test riêng.
2. **Contract**: bad dtype / shape / alignment / null trả đúng `Status`.
3. **Graph**: capture + replay cho cùng kết quả; không cấp phát trong replay.
4. **Determinism**: cùng output giữa các grid/persistent khác nhau.
5. **Bench**: JSON có `median/p10/p90`, `%peak`, `%cuBLAS`.
6. **Registry**: row YAML với `status` và `evidence` tương ứng.

---

## 11. Rủi ro

| rủi ro | mức | biện pháp |
|---|---|---|
| Fast-accum phá hợp đồng “bit-identical với mma.sync” | cao | opt-in, default promoted, test tolerance ghi rõ |
| Clock máy không ổn định làm sai %peak | cao | anchor cuBLAS mỗi phiên; báo cáo kèm clock |
| TMA pitch 16 B làm vỡ shape K lẻ | trung bình | P3 pad prep; test 4127 |
| Compile time tăng khi thêm dtype/tile | trung bình | ≤ 6 tile/dtype, 1 TU/dtype, đo ccache |
| Auto-dispatch chọn cell khi chưa đủ evidence | cao | chỉ `kReleaseSupported`; default tactic tường minh |
| Không có HW Blackwell | cao | P6 = study + compile-only; không claim |
| Rút core làm lệch số đo | trung bình | gate ±2 % mỗi commit; rollback độc lập |
| Multicast/cluster lỗi trên H100 thật | trung bình | P2a bật dần theo tile, test `test_sm90` |

---

## 12. Việc cần duyệt

1. **Row `release_supported`**: hiện chưa có row nào; bật auto-dispatch cần duyệt
   policy này (P1-C6).
2. **Bộ shape chuẩn** cho geomean (P3-4): chốt danh sách để số báo cáo so sánh
   được giữa các lần.
3. **`kFastAccum`**: xác nhận chỉ dùng cho bench/opt-in, không đổi default.

---

## Phụ lục A — Baseline JSON cần lưu

- `out/p1-baseline.json`: 30 case wgmma (bf16/fp16 × 5 tile × 3 M).
- `out/p1-anchor-cublas-fp8.json`, `out/p1-anchor-cublas-bf16.json`: đo cùng phiên.
- Log `test_sm90` + `ctest` của P0.

## Phụ lục B — Nguồn tham chiếu trong `3rd/`

| chủ đề | file |
|---|---|
| FP8 SM90 (promote, multicast, persistent) | `3rd/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` |
| wgmma descriptor | `3rd/DeepGEMM/deep_gemm/include/deep_gemm/mma/sm90.cuh` |
| scheduler/raster | `3rd/DeepGEMM/deep_gemm/include/deep_gemm/sched/gemm.cuh` |
| tcgen05 wrapper | `3rd/DeepGEMM/deep_gemm/include/deep_gemm/mma/sm100.cuh` |
| dense SM100 | `.../impls/sm100_bf16_gemm.cuh`, `sm100_fp8_fp4_gemm_1d1d.cuh` |
| MoE/attention SM100 | `.../impls/sm100_bf16_mega_moe.cuh`, `sm100_mqa_logits.cuh` |
| CuTe DSL tcgen05 | `3rd/vllm/vllm/cute_utils/_tcgen05.py` |

## Phụ lục C — Ánh xạ status

`tools/gen_registry.py` ánh xạ YAML → `CapabilityStatus`:
`planned / implemented / compiled / numerically_validated / benchmarked /
release_supported`. Chỉ `release_supported` được auto-dispatch
(`csrc/runtime/registry.cc`). Row mới của P1-C6 bắt đầu ở `benchmarked`, lên
`release_supported` sau khi có đủ mục §10.
