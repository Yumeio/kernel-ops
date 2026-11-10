# Benchmark B0–B17 trên NVIDIA RTX 5060 Ti (SM120a)

Trạng thái: **đo ngày 2026-11-10**. Binary benchmark được build từ commit
`aa4e2b8` trên một cây làm việc **dirty** (provenance trong JSON:
`commit=aa4e2b8`, `dirty=1`). Trong lúc chạy, cùng checkout có session khác
commit tiếp; HEAD lúc viết báo cáo là `8ecaf00`, **không phải** bản đã đo.
Số liệu dưới đây ứng với binary đã build từ cây `aa4e2b8` + dirty, không ứng
với HEAD lúc viết. Theo `CONTRACTS.md` §2, kết quả từ build dirty là **diagnostic-only**,
không dùng để thăng hạng `release_supported`; đây là báo cáo hiệu năng và độ
phủ, không phải evidence phát hành.

> **Cập nhật sampler (2026-11-10 20:02).** `bench_sampling` được chạy lại 3
> round trên cùng GPU sau khi tối ưu `sample()`, top-p p≥0.95, gumbel top-k và
> argmax: **136/136 case thắng**, geomean **6.40x** (trước 3.63x, 27 case thua).
> Các bảng sampler trong §1, §5.1, §5.3, §5.4, §7.7 và các hàng `sampling.*`
> ở Phụ lục A là ảnh chụp 18:4x; số mới và phương pháp nằm trong
> [`docs/SAMPLING_BENCH.md`](SAMPLING_BENCH.md) và thay thế các hàng đó.

Mục tiêu: đo **toàn bộ kernel B0–B17 có thể chạy trên SM120** bằng một GPU
Blackwell consumer (RTX 5060 Ti), so với **PyTorch 2.14.1+cu130**, NCCL và các
baseline third-party build được tại chỗ (SGLang sampler, hpc-ops sampler,
llama.cpp GGUF, attention.rs low-bit). Báo cáo chia hai phần:

- **Phần I — B0–B10:** 13 target, 443 case, đo trực tiếp bằng
  `bench-sm120a` (mục 1–9).
- **Phần II — B11–B17:** 164 case, dùng lại các lần đo trong `out/`
  (`out/b11-b15`, `out/b12`, `out/b17`) cộng hai lần chạy mới cho B13/B16
  (mục 10–18).

B14 được cập nhật lại lúc **19:5x** sau khi kernel sparse thay đổi; các phần
khác giữ ảnh chụp **18:4x**. Số liệu thô nằm trong `out/rtx5060/` và các thư
mục `out/b*` (không commit).

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Phần cứng và phần mềm](#2-phần-cứng-và-phần-mềm)
3. [Phạm vi và phương pháp](#3-phạm-vi-và-phương-pháp)
4. [Cổng test và độ phủ](#4-cổng-test-và-độ-phủ)
5. [Tổng hợp kops vs PyTorch](#5-tổng-hợp-kops-vs-pytorch)
6. [So sánh third-party](#6-so-sánh-third-party)
7. [Đọc kết quả theo family](#7-đọc-kết-quả-theo-family)
8. [Tái lập](#8-tái-lập)
9. [Hạn chế](#9-hạn-chế)
10. [Phần II — B11–B17](#phần-ii--b11b17)
11. [B11 — MLA decode + prepack](#11-b11--mla-decode--prepack)
12. [B12 — Communication](#12-b12--communication)
13. [B13 — SM120 tile](#13-b13--sm120-tile)
14. [B14 — Sparse attention](#14-b14--sparse-attention)
15. [B15 — Hyper-connection + AttnRes](#15-b15--hyper-connection--attnres)
16. [B16 — Megakernel](#16-b16--megakernel)
17. [B17 — Low-bit KV + MoE W2](#17-b17--low-bit-kv--moe-w2)
18. [Kết luận Phần II](#18-kết-luận-phần-ii)
- [Phụ lục A — Bảng đầy đủ 443 case B0–B10](#phụ-lục-a--bảng-đầy-đủ-443-case)
- [Phụ lục B — 32 case chuyển từ Phần I sang Phần II](#phụ-lục-b--32-case-chuyển-từ-phần-i-sang-phần-ii)
- [Phụ lục C — Danh mục artifact](#phụ-lục-c--danh-mục-artifact)
- [Phụ lục D — Bảng đầy đủ 164 case B11–B17](#phụ-lục-d--bảng-đầy-đủ-164-case-b11b17)

## 1. Tóm tắt

- **Phủ kernel:** 174/174 kernel của B0–B10 có thân (`csrc/`), 13 target
  benchmark chạy được, **443 case** ở Phần I. Toàn bộ B0–B17 có **236 kernel
  khai báo** và **607 case** (443 + 164); 32 case từng nằm ngoài B0–B10 nay
  thuộc Phần II.
- **Cổng đúng đắn:** `ctest` 25/26 tại lần chạy 18:4x; test fail duy nhất là
  `kops_test_sparse` (B14). Sau bản cập nhật B14 (19:5x), cả 5 test B14 —
  `test_sparse`, `test_sparse_mla`, `test_compressed_attention`,
  `test_kv_compressor`, `test_indexer` — đều **PASS**; toàn bộ test B0–B10
  vẫn pass. Chưa chạy lại cả 26 test sau cập nhật.
- **kops vs PyTorch:** **340/443 case có baseline** (335 PyTorch + 5 SDPA).
  Geomean **6.79x** sau cập nhật sampler (ảnh chụp 18:4x: 5.41x; tính lại từ
  log-tỉ lệ, 136 case sampler đổi từ 3.63x lên 6.40x); median của 136 case
  sampler là **5.37x**. kops thắng 317, thua 17, hòa 6. So với eager kiểu HF
  (`--style hf`), norm đạt **8.56x** và chains **8.58x** (§5.8).
- **Sampler (B6):** 136 case sampling đạt geomean **6.40x**, **136/136 thắng**
  sau cập nhật 20:02 (trước: 3.63x, 27 thua). `sample_fused` 5.47x/69 case,
  `sample()` 4.09x/33 case; chỉ top-k còn dùng đường probability legacy, đã đổi
  sang candidate histogram. Xem `docs/SAMPLING_BENCH.md`.
- **GEMM tensor core (B7):** BF16 `mma_bf16_tuned` đạt **48.9 TFLOP/s** ở
  M=4096 (**1.09x** PyTorch matmul), FP8 tensorwise **86.4 TFLOP/s**, INT8
  **132.8 TFLOP/s**. Ở M=64, kops thua PyTorch (0.69–0.95x) — đúng điểm yếu đã
  ghi trong roadmap.
- **B2 (KV cache) và B9 (recurrent):** không có primitive PyTorch tương đương
  nên không có cột PyTorch; báo cáo dùng băng thông đạt được so với trần
  448 GB/s và thời gian thô. Các case copy lớn đạt 380–420 GB/s; case nhỏ vượt
  trần do nằm trong L2 (harness gắn cờ `over_peak`).
- **Third-party:**
  - B10 `spec.reject.bonus`: hpc-ops **0.1659 ms** vs kops **0.1781 ms**
    (hpc nhanh 1.07x); SGLang tree target-only **0.0246 ms** vs kops
    **0.0143 ms** (kops nhanh **1.71x**).
  - B5 GGUF MMVQ (cold cache 256 MB): llama.cpp nhanh hơn kops **1.1–2.5x**
    trên q4_0/q4_k/q8_0; kops vẫn nhanh hơn PyTorch dequant+matmul
    **24–131x** tùy shape và kiểu lượng tử.
  - B8 prefill: kops đạt 0.70–0.94x torch SDPA trên 5 case prefill (1 case paged đo bằng KV dense); decode
    không có baseline SDPA (math fallback).
- **Điểm yếu đã xác nhận:** prefill attention thua SDPA; GGUF MMVQ thua
  llama.cpp; M=64 GEMM thua PyTorch; gần như toàn bộ B2/B9 và 40 case GEMM
  lượng tử hóa không có baseline PyTorch. Sampler không còn case thua sau
  cập nhật 20:02.


### 1b. B11–B17 (Phần II)

- **B11 (MLA):** 37 case, 8 case có baseline torch, geomean **1.13x**,
  5 thắng / 3 thua. Split-KV là bắt buộc: `kv64k.q1` không split chỉ 0.29x
  torch, bật `s8` lên 1.13x; `kv1m.q1.s32` đạt **24.2 TFLOP/s / 100 GB/s**.
  FP8 software dequant chậm hơn BF16 (16.2 TF so với 24.2 TF).
- **B12 (communication):** 64 case, 57 case có baseline, geomean **1.59x**
  nhưng median **0.90x**: kops thắng message nhỏ và fusion residual+RMSNorm
  (4.40x), thua NCCL ở size lớn (allreduce t8192 0.87x) và comm-GEMM (0.56x).
- **B13 (SM120 tile):** 2 case BF16, **49.4 TFLOP/s** ở 4096², 1.07x PyTorch;
  11 kernel FP8/FP4/tcgen05 còn lại chưa có case trên SM120.
- **B14 (sparse):** 22 case, 22 có baseline, geomean **2.52x**, 20 thắng /
  2 thua. Top-k r1 2.3–3.7x, indexer 10–11.6x, compressed attention CSA/HCA
  1.25–3.00x; `sparse_mla.topk2048` và `sparse_topk.r4.kv1m` thua.
- **B15 (hyper-connection + AttnRes):** 24 case, 24 có baseline, geomean
  **13.47x**, thắng cả 24 (HC 3.59x, AttnRes 20.92x).
- **B16 (megakernel):** 4 case; mega **chậm hơn** modular 6.6x (MLP), 22.9x
  (attention), 15.8x (decoder) — chưa đạt gate "nhanh hơn modular + graph".
- **B17 (low-bit KV + W2):** 11 case, 6 có baseline attention.rs, geomean
  **0.64x**: attention.rs nhanh hơn 2.2–2.6x ở store, 1.17x ở prefill; kops
  chỉ nhỉnh 2% ở decode turbo4.
- Cộng hai phần: **607 case**, **459 case có baseline**; geomean toàn bộ
  **4.42x**. Riêng Phần II ngoài B16: geomean **2.47x**, median **1.92x**.

## 2. Phần cứng và phần mềm

| Hạng mục | Giá trị |
|---|---|
| GPU đo | NVIDIA GeForce RTX 5060 Ti, 16 311 MiB, cc 12.0 (SM120), 36 SM |
| GPU khác | Cùng máy có GPU 0 cùng model; **GPU 0 đang bận** bởi process khác, mọi phép đo dùng **GPU 1** |
| Trần băng thông | 448 GB/s (128-bit × 14 001 MHz × 2), harness tự tính |
| Xung SM | 2 692 MHz trong JSON; max của GPU 1 là 3 090 MHz |
| CPU / RAM | AMD Ryzen Threadripper PRO 3975WX (32C/64T), 251 GiB |
| OS | Ubuntu 24.04.5 LTS, kernel 6.8.0-138 |
| Driver / runtime | NVIDIA 615.71.09, CUDA UMD 13.4 |
| Toolkit build | CUDA 13.4.92 (`nvcc` V13.4.92) |
| PyTorch | 2.14.1+cu130, CUDA available, 2 GPU |
| Build | preset `bench-sm120a`, `KOPS_CUDA_ARCHS=120a`, Release, tests ON |
| Commit build | `aa4e2b8` + dirty; HEAD lúc viết `8ecaf00` (session khác commit sau khi đo) |
| ABI | `KOPS_ABI_VERSION=6` |

Ghi chú: GPU 0 có một process benchmark của phiên khác (dao động 100% util).
GPU 1 idle hoàn toàn tại thời điểm bắt đầu; mọi phép đo Phần I chạy trên GPU 1,
tuần tự, không chồng lấn. Phần II dùng lại các lần đo trong `out/` (cùng GPU
1); riêng B12 chạy 2 GPU với pinned-host staging, không P2P.

## 3. Phạm vi và phương pháp

### 3.1 Target và bộ lọc

| Target | Family | Case đo | B0–B10 | B11–B17 |
|---|---|---:|---:|---:|
| `bench_model_ops` | model_ops | 28 | 10 | 18 (B15 AttnRes) |
| `bench_norm` | norm | 16 | 16 | 0 |
| `bench_activation` | activation | 9 | 9 | 0 |
| `bench_position` | position | 22 | 22 | 0 |
| `bench_embedding` | embedding | 11 | 11 | 0 |
| `bench_merge` | merge | 30 | 21 | 9 (B17 lowbit) |
| `bench_sampling` | sampling | 136 | 136 | 0 |
| `bench_chains` | chains | 10 | 10 | 0 |
| `bench_kvcache` | kvcache | 28 | 28 | 0 |
| `bench_state` | state | 22 | 22 | 0 |
| `bench_quantization` | quantization | 22 | 22 | 0 |
| `bench_gemm` | gemm | 128 | 123 | 3 (B11 prepack) + 2 (B13 SM120) |
| `bench_moe` | moe | 15 | 13 | 2 (B17 W2) |
| `bench_mla` | mla | 34 | 0 | 34 (B11) |
| `bench_sparse` | sparse | 22 | 0 | 22 (B14) |
| `bench_hyper_connection` | hyper_connection | 6 | 0 | 6 (B15) |
| `bench_mega` | mega | 4 | 0 | 4 (B16) |
| `bench_communication` | communication | 64 | 0 | 64 (B12) |

Tổng: **607 case thô** trên 18 target — **443 case Phần I** (B0–B10) và
**164 case Phần II** (B11–B17). Không còn case nào bị loại: 32 case từng nằm
ngoài B0–B10 nay thuộc Phần II (B11 3, B15 18, B17 11) và có bảng đầy đủ ở
Phụ lục D. Các case `ds_mla_v40/v41/fp4` (codec V4.0/V4.1 của B14) vẫn được
giữ trong Phần I vì kernel gốc `ds_mla` thuộc B2; chúng được ghi chú riêng
trong bảng.

### 3.2 Giao thức đo

- **kops:** `build/bench-sm120a/benchmark/bench_<family> --rounds 3 --json`.
  Harness dùng CUDA event, mỗi mẫu là một nhóm 20 launch (hot-L2 mặc định),
  lấy median trên 3 round. Warmup/iters lấy từ bảng case của từng family.
- **PyTorch:** `python3 benchmark/compare/torch_baseline.py --cases ... --rounds 3`,
  style mặc định `builtin` — tức là baseline mạnh nhất có sẵn (ví dụ
  `F.rms_norm`, `F.silu`, `F.scaled_dot_product_attention`), không phải bản
  eager nhiều op. Riêng sampler fused dùng chuỗi eager softmax + multinomial
  vì PyTorch không có primitive tương đương.
- **Tỉ lệ:** `baseline_ms / kops_ms`; **>1 là kops nhanh hơn**. Geomean trên
  các case có baseline; median là median của phân bố tỉ lệ.
- **Cold cache:** các so sánh third-party dùng `--flush-l2 256` cho kops và
  `--flush-mb 256` cho baseline C++; llama.cpp luôn flush 256 MB. Chế độ cold
  tính cả overhead một cặp event cho mỗi launch (harness ghi rõ), nên không
  so trực tiếp với số hot-L2.
- **L2:** case nhỏ (vài MB) nằm trong L2 của GPU nên GB/s có thể vượt trần
  DRAM; JSON gắn `over_peak=true`. Bảng B2 ghi rõ % trần và không dùng các
  case đó để kết luận về DRAM.
- **Số round:** 3 cho cả kops và PyTorch. Không chạy song song hai phép đo.
- **Phần II:** B11/B15 lấy từ `out/b11-b15/opt-*.json` (bản đã tối ưu);
  B14 lấy `out/rtx5060/{kops/sparse-new,torch/sparse-new,torch/sparse-csa}.json`
  (bản cập nhật 19:5x); B12 lấy `out/b12/kops-opt-clean.json` +
  `torch-opt.json`; B17 lấy `out/b17/*.json`; B13/B16 do báo cáo này chạy
  bằng `bench-sm120a` (`--only mma_sm120`, `bench_mega`).

### 3.3 Cách gắn case với bước

Mỗi case được gắn một nhãn bước theo bảng kernel của `ROADMAP.md` (Phụ lục
A). Chain được gắn nhiều bước (`B0+B1`, `B1+B3`, `B1+B8`) và được đếm một lần
trong tổng. Nhãn chỉ dùng để nhóm thống kê; nguồn chân lý cho kernel vẫn là
`ROADMAP.md` + header.

## 4. Cổng test và độ phủ

### 4.1 ctest trên SM120a

`ctest --test-dir build/bench-sm120a` — **25/26 pass**, 200.6 s (trạng thái
18:4x). Sau bản cập nhật B14, chạy riêng 5 test B14: **5/5 PASS** (bao gồm
`kops_test_sparse`); chưa chạy lại toàn bộ suite.

| Test | Kết quả | Ghi chú |
|---|---|---|
| `kops_test_m1`, `model_ops`, `activation`, `embedding`, `position`, `sampling`, `merge`, `norm`, `kvcache`, `state`, `quantization`, `gemm`, `communication`, `moe`, `mega`, `mla`, `sparse_mla`, `indexer`, `compressed_attention`, `kv_compressor`, `hyper_connection`, `mla_metadata`, `mla_1m`, `sm120`, `conformance` | PASS | 25 test |
| `kops_test_sparse` | **PASS** (sau cập nhật B14) | Trước 19:5x fail illegal memory access; bản mới pass cả 122 check |

Log: `out/rtx5060/logs/ctest.log`.

### 4.2 Phủ theo bước

| bước | kernel | family / target | case đo |
|---|---:|---|---:|
| B0 | 2 | model_ops | 10 |
| B1 | 32 | norm, activation, position, embedding, sampling, attention/merge | 83 |
| B2 | 20 | kvcache, state cache, attention/mla | 31 |
| B3 | 21 | quantization (+ norm/act/kvcache fusion) | 18 |
| B4 | 18 | gemm prep, quantization/weight_repack | 20 |
| B5 | 16 | gemm GEMV/GEMM decode, moe/gemv | 53 |
| B6 | 18 | moe routing/permute, sampling, chains | 121 |
| B7 | 16 | gemm MMA, marlin, grouped, LoRA, GGUF, moe | 59 |
| B8 | 7 | attention prefill/decode | 15 |
| B9 | 20 | state conv1d/SSM/delta/lightning | 17 |
| B10 | 4 | sampling/speculative, state_commit | 10 |
| B11 | 12 | mla, machete/grouped prep (4 kernel chạy SM120) | 37 |
| B12 | 11 | communication: collectives, EP, comm-GEMM | 64 |
| B13 | 12 | gemm SM120 tile (1 cell có case) | 2 |
| B14 | 8 | sparse topk/indexer/sparse MLA/compressed attention (4 kernel có case) | 22 |
| B15 | 8 | hyper-connection + AttnRes | 24 |
| B16 | 6 | mega_mlp/attn/moe_local/decoder | 4 |
| B17 | 5 | lowbit KV + moe_w2 | 11 |
| **B0–B17** | **236** | 18 target benchmark | **607** (+6 case chain gắn nhiều bước) |

Ghi chú: cột "kernel" là số kernel khai báo của bước (B11–B17 gồm cả kernel
compile-only/handoff). Số case không tỉ lệ 1:1 với số kernel: B11 chỉ 4/12
kernel chạy được trên SM120, B13 1/12 cell có case, B14 4/8 kernel có case,
B16 4/6 kernel có case.

Ghi chú:

- B2 có 31 case nhưng 20 kernel: 7 case là codec `ds_mla_v40/v41/fp4` của B14
  chạy trên kernel B2, cộng 4 case state cache và 2 case
  `concat_mla_q`/`mla_context_gather`.
- B6 có 121 case vì ma trận sampler: 33 case `sample()`, 69 case
  `sample_fused`, 17 case spec/tree, cộng routing/permute.
- B8 chỉ có 15 case attention nhưng 5 case trong đó là prefill có SDPA.
- B1 có thêm 6 case chain gắn `B0+B1`/`B1+B3`/`B1+B8`.

## 5. Phần I — Tổng hợp kops vs PyTorch

### 5.1 Theo family

| family | case | có baseline | PyTorch | SDPA | không ref | geomean | median | thắng | thua | hòa | tốt nhất | tệ nhất |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model_ops | 10 | 10 | 10 | 0 | 0 | 2.40x | 2.09x | 7 | 0 | 3 | 6.93x | 1.00x |
| norm | 16 | 16 | 16 | 0 | 0 | 2.67x | 1.62x | 15 | 1 | 0 | 10.38x | 0.96x |
| activation | 9 | 9 | 9 | 0 | 0 | 2.09x | 1.77x | 8 | 1 | 0 | 8.13x | 0.91x |
| position | 22 | 22 | 22 | 0 | 0 | 13.74x | 19.29x | 22 | 0 | 0 | 145.46x | 1.35x |
| embedding | 11 | 11 | 11 | 0 | 0 | 3.43x | 2.79x | 11 | 0 | 0 | 15.73x | 1.05x |
| merge | 21 | 9 | 4 | 5 | 12 | 3.74x | 0.94x | 4 | 5 | 0 | 32.32x | 0.70x |
| sampling | 136 | 136 | 136 | 0 | 0 | 6.40x | 5.37x | 136 | 0 | 0 | 2475.5x | 1.02x |
| chains | 10 | 8 | 8 | 0 | 2 | 8.63x | 8.34x | 8 | 0 | 0 | 26.05x | 3.71x |
| kvcache | 28 | 0 | 0 | 0 | 28 | — | — | 0 | 0 | 0 | — | — |
| state | 22 | 1 | 1 | 0 | 21 | 972.46x | 972.46x | 1 | 0 | 0 | 972.46x | 972.46x |
| quantization | 22 | 22 | 22 | 0 | 0 | 12.16x | 9.63x | 22 | 0 | 0 | 229.64x | 2.57x |
| gemm | 123 | 83 | 83 | 0 | 40 | 9.09x | 6.72x | 72 | 10 | 1 | 184.97x | 0.69x |
| moe | 13 | 13 | 13 | 0 | 0 | 5.47x | 7.09x | 11 | 0 | 2 | 37.83x | 1.00x |
| **tổng** | **443** | **340** | **335** | **5** | **103** | **6.79x** | **—** | **317** | **17** | **6** | | |

Đọc bảng:

- `có baseline` = số case có ít nhất một baseline (PyTorch hoặc SDPA).
- `không ref` = case chỉ có số đo kops (B2, B9, GEMM lượng tử, attention
  decode).
- `thắng/thua/hòa` tính với ngưỡng ±2%. Tổng 290 thắng / 43 thua / 7 hòa.
- Geomean toàn bộ **6.79x** sau cập nhật sampler (tính lại từ log-tỉ lệ;
  ảnh chụp 18:4x là 5.41x). Median của 136 case sampler là **5.37x**; median
  tổng chưa tính lại. Geomean vẫn bị kéo bởi các case PyTorch eager cực chậm
  (`bincount` 2476x, `state_commit` 972x, `spec.tree.target_only` 776x).

### 5.2 15 case kops nhanh nhất

| case | family | kops ms | baseline ms | ratio |
|---|---|---:|---:|---:|
| `sampling.bincount.v32768.b256.t64` | sampling | 0.0048 | 11.7833 | 2475.48x |
| `state_commit.2pools.b64.d8` | state | 0.0028 | 2.7355 | 972.46x |
| `spec.tree.target_only.n64.d8.v32000.b8` | sampling | 0.0073 | 5.8636 | 803.89x |
| `sampling.bincount.v131072.b32.t1024` | sampling | 0.0029 | 1.6365 | 567.83x |
| `quant.gptq_marlin.4bit.g128.4096x4096` | quantization | 0.0123 | 2.8305 | 229.64x |
| `sampling.bias.bf16.v131072.b128.nnz8` | sampling | 0.0029 | 0.6287 | 219.90x |
| `gemm.w4a16_awq.g128.m1.14336x4096` | gemm | 0.0398 | 7.3636 | 184.97x |
| `gemm.nf4_lut.b64.m1.14336x4096` | gemm | 0.0385 | 6.9827 | 181.28x |
| `gemm.gguf_mmvq.q4_0.m1.14336x4096` | gemm | 0.0347 | 5.8396 | 168.12x |
| `gemm.nf4_lut.b64.m1.4096x14336` | gemm | 0.0397 | 6.5427 | 165.00x |
| `gemm.w4a16_gptq.g128.m1.14336x4096` | gemm | 0.0408 | 6.0744 | 148.93x |
| `spec.tree.greedy.n64.d8.v32000.b256` | sampling | 0.0110 | 1.6045 | 145.80x |
| `position.fused_qk_norm_rope.bf16.t64` | position | 0.0031 | 0.4454 | 145.46x |
| `gemm.nf4_lut.b64.m1.4096x4096` | gemm | 0.0128 | 1.7589 | 136.98x |
| `quant.awq_marlin.4bit.g128.4096x4096` | quantization | 0.0260 | 3.4616 | 133.39x |

### 5.3 15 case kops chậm nhất

| case | family | kops ms | baseline ms | ratio |
|---|---|---:|---:|---:|
| `gemm.dual_bf16.m64.4096x4096` | gemm | 0.2730 | 0.1886 | 0.69x |
| `prefill.ragged.bf16.d64.causal.s2048.h64x8` | merge | 1.1423 | 0.8048 | 0.70x |
| `gemm.mma_bf16.m64.4096x4096` | gemm | 0.0950 | 0.0676 | 0.71x |
| `prefill.ragged.bf16.d256.causal.s1024.h8x4` | merge | 0.1886 | 0.1459 | 0.77x |
| `prefill.paged.bf16.d128.chunk512.kv4096.ps16.h32x8` | merge | 1.0807 | 0.8541 | 0.79x |
| `gemm.mma_bf16.m128.4096x4096` | gemm | 0.1388 | 0.1124 | 0.81x |
| `prefill.ragged.bf16.d128.full.b4.s1024.h32x8` | merge | 1.8388 | 1.6124 | 0.88x |
| `activation.silu.bf16.n4m` | activation | 0.0208 | 0.0190 | 0.91x |
| `prefill.ragged.bf16.d128.causal.s2048.h32x8` | merge | 0.9459 | 0.8868 | 0.94x |
| `gemm.mma_bf16_tuned.m64.4096x4096` | gemm | 0.0714 | 0.0680 | 0.95x |
| `gemm.mma_bf16.m1024.4096x4096` | gemm | 0.7661 | 0.7323 | 0.96x |
| `norm.layer_norm.fp32.h4096.b1024` | norm | 0.0423 | 0.0405 | 0.96x |
| `gemm.dual_bf16.m512.4096x4096` | gemm | 0.8443 | 0.8126 | 0.96x |
| `gemm.mma_fp16.m256.4096x4096` | gemm | 0.2160 | 0.2107 | 0.98x |
| `gemm.mma_bf16.m256.4096x4096` | gemm | 0.2161 | 0.2114 | 0.98x |

Bảng này là ảnh chụp 18:4x của các family khác. Sau cập nhật sampler
20:02, **không còn case sampling nào dưới 1.02x**; 15 case chậm nhất còn lại
là GEMM M≤128, prefill SDPA và layer_norm fp32. Đây là các điểm cần xử lý nếu
muốn dùng các đường đó trong production.

### 5.4 Sampler (B6) — cập nhật 20:02

| chỉ số | giá trị |
|---|---:|
| case `sample_fused` có baseline | 69 |
| geomean PyTorch/kops | **5.47x** |
| kops thắng / thua | **69 / 0** (thấp nhất 1.02x) |
| case `sample()` có baseline | 33 |
| geomean `sample()` | **4.09x** |
| kops thắng / thua | **33 / 0** |
| toàn bộ sampling | **136 case, 6.40x, 136 / 0** |

Ba case fused có biên thắng nhỏ nhất (trước đây là ba case thua):

| case | kops ms | PyTorch ms | ratio |
|---|---:|---:|---:|
| `sampling.fused.p0.99.bf16.v151936.b16` | 0.1532 | 0.1771 | 1.16x |
| `sampling.fused.p0.99.bf16.v151936.b32` | 0.2181 | 0.3553 | 1.63x |
| `sampling.fused.p0.99.bf16.v151936.b64` | 0.3717 | 1.1924 | 3.21x |

Đường **fused** vẫn là đường nên dùng. `sample()` giờ chia sẻ body fused cho
greedy, unfiltered, top-p-only và min-p; chỉ top-k còn chạy đường probability
legacy, đã đổi sang candidate histogram (1 histogram + 1 gather thay 4 pass
radix) và bỏ pass `row_max`/pass z khi không cần. Vì vậy `sample()` không còn
là "reference chậm"; số cũ 0.59x chỉ còn trong ảnh chụp 18:4x.

### 5.5 GEMM tensor core (B7)

| case | kops ms | kops TFLOP/s | PyTorch TFLOP/s | tỉ lệ |
|---|---:|---:|---:|---:|
| `gemm.mma_bf16.m1024.4096x4096` | 0.7661 | 44.9 | 46.9 | 0.96x |
| `gemm.mma_bf16.m256.4096x4096` | 0.2161 | 39.7 | 40.6 | 0.98x |
| `gemm.mma_bf16.m4096.4096x4096` | 2.8697 | 47.9 | 45.1 | 1.06x |
| `gemm.mma_bf16.m64.4096x4096` | 0.0950 | 22.6 | 31.7 | 0.71x |
| `gemm.mma_bf16_tuned.m1024.4096x4096` | 0.7168 | 47.9 | 45.5 | 1.05x |
| `gemm.mma_bf16_tuned.m256.4096x4096` | 0.2150 | 39.9 | 40.8 | 0.98x |
| `gemm.mma_bf16_tuned.m4096.4096x4096` | 2.8087 | 48.9 | 45.1 | 1.09x |
| `gemm.mma_bf16_tuned.m64.4096x4096` | 0.0714 | 30.1 | 31.6 | 0.95x |
| `gemm.dual_bf16.m1024.14336x4096` | 5.3454 | 45.0 | 43.7 | 1.03x |
| `gemm.dual_bf16.m64.4096x4096` | 0.2730 | 15.7 | 22.8 | 0.69x |
| `gemm.fp8_mma.tensorwise.m4096.4096x4096` | 1.5913 | 86.4 | — | — |
| `gemm.int8_mma.m4096.4096x4096` | 1.0349 | 132.8 | — | — |

- BF16 `mma_bf16_tuned` đạt **48.9 TFLOP/s** ở M=4096, **1.09x** PyTorch
  matmul; M=1024 đạt 47.9 TF (1.05x); M=64 chỉ 30.1 TF (0.95x).
- FP8 tensorwise đạt **86.4 TFLOP/s** ở M=4096; INT8 **132.8 TOPS**. Hai
  đường này không có baseline PyTorch trong `torch_baseline.py` (không dùng
  `torch._scaled_mm`), nên chỉ so với roofline nội bộ.
- `dual_bf16` (gate/up + GLU) đạt 45–46 TF ở M≥1024, 1.03x PyTorch; M=64
  thua (0.69x).
- Các case `machete_prepack`/`grouped_prepare` là B11, đã loại khỏi bảng.

### 5.6 B2 — KV cache và page (không có baseline PyTorch)

| case | kops ms | GB/s | % trần 448 GB/s |
|---|---:|---:|---:|
| `copy_pages.bytes.p512.b4096` | 0.0112 | 1503.5 | 336% |
| `append.nhd.bf16.t2048.h4.d128` | 0.0056 | 1490.7 | 333% |
| `convert.bf16_to_f32.b256.s4096` | 0.0050 | 1246.7 | 278% |
| `reshape.nhd.bf16.t2048.h4.d128` | 0.0074 | 1140.2 | 255% |
| `zero_pages.bytes.p512.b4096` | 0.0080 | 1044.2 | 233% |
| `gather_dequant.t4096.e576` | 0.0103 | 917.3 | 205% |
| `cp_gather.bytes.t4096.e576` | 0.0053 | 895.6 | 200% |
| `gather_pages.nhd.bf16.t4096.h4.d128` | 0.0109 | 769.5 | 172% |
| `scatter_pages.nhd.bf16.t4096.h4.d128` | 0.0113 | 741.6 | 166% |
| `fused_qkv_append.bf16.t2048.hq32.hkv8.d128` | 0.1135 | 665.0 | 148% |
| `kv_scale_update.bf16.t2048.h8.d128` | 0.0155 | 542.9 | 121% |
| `ds_mla_v41_gather.bf16.t1024` | 0.0032 | 493.4 | 110% |
| `mla_rope_fused.bf16.t2048.hq16.r512.p64` | 0.0272 | 481.4 | 107% |
| `fused_kv_append.bf16.t2048.hkv4.d128` | 0.0176 | 477.8 | 107% |
| `ds_mla_v41_fp4_gather.bf16.t1024` | 0.0032 | 425.4 | 95% |
| `ds_mla.fp8.t2048.r512.p64` | 0.0085 | 404.9 | 90% |
| `relayout.nhd_to_hnd.p512` | 0.0494 | 339.3 | 76% |
| `concat_mla.t2048.r512.p64` | 0.0157 | 301.2 | 67% |
| `concat_mla_grouped.l2.t1024.r512.p64` | 0.0173 | 272.9 | 61% |
| `requantize.int8.t2048.h8.d128` | 0.0435 | 193.0 | 43% |
| `reshape_per_token_head.fp8.t2048.h8.d128` | 0.0440 | 190.8 | 43% |
| `ds_mla_v41.fp4.t1024.r448.p64` | 0.0080 | 168.3 | 38% |
| `indexer_fp8.t2048.d128` | 0.0053 | 152.5 | 34% |
| `ds_mla_v41.fp8.t1024.r448.p64` | 0.0122 | 130.0 | 29% |
| `ds_mla_v40.fp8.t256.r448.p64` | 0.0032 | 127.8 | 29% |
| `ring.bf16.t1024.h4.d128` | 0.0193 | 108.4 | 24% |
| `ds_mla_v40_gather.bf16.t256` | 0.0042 | 104.7 | 23% |
| `reshape.legacy.bf16.t2048.h4.d128` | 0.0904 | 92.8 | 21% |

Các case copy lớn (`copy_pages`, `append`, `convert`, `reshape`) đạt
1 000–1 500 GB/s vì working set 8–16 MB nằm trong L2; `gather_pages`,
`scatter_pages`, `gather_dequant` đạt 700–950 GB/s. Các case lớn hơn
(`relayout`, `concat_mla`, `fused_qkv_append`) đạt 270–665 GB/s, tức 61–148%
trần — phần vượt trần là do byte-count của harness tính cả đọc và ghi, và một
phần working set vẫn nằm trong L2. Bộ case hiện tại chưa có một case copy
thuần DRAM đủ lớn để tách hoàn toàn L2, nên bảng này là **thời gian thô +
roofline L2/DRAM**, không phải so sánh công bằng với một thư viện khác.

### 5.7 B9 — Recurrent (không có baseline PyTorch)

Family `state` có 22 case: 4 case state-cache của B2, 17 case B9, 1 case
`state_commit` của B10. Chỉ `state_commit` có baseline PyTorch; 21 case còn
lại không có primitive tương đương. Các case lớn đạt 350–380 GB/s
(`ssm_update`, `delta_decode`, `linear_decode`, `fused_delta_decode`); các case
nhỏ (`state_copy/gather/scatter/zero`, `delta_gating`) bị chi phối bởi launch
overhead. `state_commit` có baseline PyTorch (vòng lặp copy trên CPU/GPU) và
nhanh hơn 972x — con số này chỉ có nghĩa "kernel so với vòng lặp eager", không
phải so với một kernel tối ưu.

### 5.8 PyTorch eager kiểu HF (`--style hf`)

Bảng chính dùng `builtin` — baseline mạnh nhất. Với các op mà model code thật
chạy bằng chuỗi eager (RMSNorm tự viết, LayerNorm nhiều op), chạy thêm
`--style hf`:

| family | case có ref | geomean vs eager HF | geomean vs builtin |
|---|---:|---:|---:|
| norm | 16 | **8.56x** | 2.67x |
| chains | 8 | **8.58x** | 8.63x |

Với eager HF, norm đạt 8.56x thay vì 2.67x; chains gần như không đổi
(8.58x) vì chain vốn đã là chuỗi eager ở cả hai style. Đây là con số nên dùng
khi so với một model implementation thông thường, không phải khi so với
primitive đã fuse của PyTorch.

## 6. Phần I — So sánh third-party

### 6.1 B10 — speculative decoding (cold cache 256 MB)

| case | kops ms | PyTorch ms | hpc-ops ms | SGLang ms | kops vs tốt nhất |
|---|---:|---:|---:|---:|---:|
| `spec.mask.p2048.n64.b64` | 0.0737 | 2.2210 | — | — | 30.14x |
| `spec.reject.bonus.k0.v120832.b128` | 0.1781 | 1.0424 | 0.1659 | — | 0.93x |
| `spec.reject.chain.k8.v151936.b32` | 0.1187 | 0.4035 | — | — | 3.40x |
| `spec.reject.chain.k8.v32768.b128` | 0.1004 | 0.3246 | — | — | 3.23x |
| `spec.reject.greedy.k8.v151936.b32` | 0.0082 | 0.0215 | — | — | 2.62x |
| `spec.reject.target_only.k8.v151936.b32` | 0.0655 | 0.3236 | — | — | 4.94x |
| `spec.tree.greedy.n64.d8.v32000.b256` | 0.0184 | 0.1905 | — | — | 10.33x |
| `spec.tree.q.n64.d8.v32000.b8` | 0.0205 | 1.7465 | — | — | 85.28x |
| `spec.tree.target_only.n64.d8.v32000.b8` | 0.0143 | 5.7736 | — | 0.0246 | 1.71x |

- **hpc-ops** (`fused_sampler_temperature`, commit `2a2e265`): 0.1659 ms vs
  kops 0.1781 ms cho `spec.reject.bonus.k0.v120832.b128` — hpc nhanh **1.07x**.
  Đây là full-vocab draw đơn, không phải verifier.
- **SGLang** (`TreeSpeculativeSamplingTargetOnly`, sgl-kernel `96a04cb`):
  0.0246 ms vs kops 0.0143 ms cho `spec.tree.target_only` — kops nhanh
  **1.71x**. SGLang chỉ làm target-only, không làm p/q rejection.
- 7 case spec còn lại chỉ có PyTorch; kops nhanh 2.6–403x so với eager.

### 6.2 B5 — GGUF MMVQ vs llama.cpp (cold cache 256 MB)

| case | kops ms | llama.cpp ms | PyTorch ms | kops/llama.cpp | PyTorch/kops |
|---|---:|---:|---:|---:|---:|
| `gemm.gguf_mmvq.q4_0.m1.14336x4096` | 0.0963 | 0.0525 | 12.6136 | 1.83x | 131.0x |
| `gemm.gguf_mmvq.q4_0.m1.4096x4096` | 0.0328 | 0.0179 | 3.7038 | 1.83x | 113.0x |
| `gemm.gguf_mmvq.q4_k.m1.14336x4096` | 0.1004 | 0.0585 | 5.6248 | 1.72x | 56.1x |
| `gemm.gguf_mmvq.q4_k.m1.4096x4096` | 0.0369 | 0.0207 | 3.7389 | 1.78x | 101.4x |
| `gemm.gguf_mmvq.q8_0.m1.14336x4096` | 0.1720 | 0.1566 | 4.1687 | 1.10x | 24.2x |
| `gemm.gguf_mmvq.q8_0.m1.4096x4096` | 0.0573 | 0.0230 | 3.3034 | 2.49x | 57.6x |

llama.cpp (`3rd/llama.cpp` @ `90e3841d9`, build CUDA sm_120) nhanh hơn kops
**1.1–2.5x** trên cả 6 case. Đây là khoảng cách thật của đường `gguf_mmvq`
(luôn cold); ở chế độ hot-L2, kops trông nhanh hơn vì weight 8 MB nằm trong L2
— một ví dụ điển hình của việc không được so hot-L2 với baseline cold.

### 6.3 B8 — prefill attention vs torch SDPA

| case | kops ms | SDPA ms | kops/SDPA | kops TFLOP/s |
|---|---:|---:|---:|---:|
| `prefill.paged.bf16.d128.chunk512.kv4096.ps16.h32x8` | 1.0807 | 0.8541 | 1.27x | 29.8 |
| `prefill.ragged.bf16.d128.causal.s2048.h32x8` | 0.9459 | 0.8868 | 1.07x | 36.3 |
| `prefill.ragged.bf16.d128.full.b4.s1024.h32x8` | 1.8388 | 1.6124 | 1.14x | 37.4 |
| `prefill.ragged.bf16.d256.causal.s1024.h8x4` | 0.1886 | 0.1459 | 1.29x | 22.8 |
| `prefill.ragged.bf16.d64.causal.s2048.h64x8` | 1.1423 | 0.8048 | 1.42x | 30.1 |

kops chậm hơn SDPA **6–30%** trên 5 case prefill (0.70–0.94x). Đây là kết quả
kỳ vọng: B8 là thân native lớp SM80, còn SDPA trên SM120 dùng kernel
cuDNN/FlashAttention đã tối ưu cho Blackwell. Decode không có baseline SDPA
(SDPA decode là math fallback), nên 10 case decode chỉ có số đo kops.

### 6.4 Các baseline không đo được

| Baseline | Lý do |
|---|---|
| vLLM (`thirdparty.py --impl vllm`) | package `vllm` không có trong môi trường; wheel không tương thích torch 2.14+cu130 |
| FlashInfer (`--impl flashinfer`) | package `flashinfer` không có; SGLang baseline dùng header FlashInfer 0.6.14 tải riêng, không phải runtime Python |
| SGLang Python (`--impl sglang`) | `sgl_kernel` không có |
| FlashAttention-2 (`--impl fa2`) | `flash_attn` không có |
| bitsandbytes (`--impl bnb`) | `bitsandbytes` không có |
| NCCL / attention.rs lowbit | thuộc B12/B17, ngoài phạm vi B0–B10 |

## 7. Phần I — Đọc kết quả theo family

### 7.1 model_ops (B0)

10 case, geomean 2.40x, 7 thắng / 3 hòa. Ba case `residual_add` lớn
(`t4096.h8192`) hòa vì cả hai đều chạm trần DRAM; `residual_add` 3 ngả và
`cast` nhỏ nhanh 5.9–6.9x nhờ fuse/gộp launch.

### 7.2 norm (B1)

16 case, geomean 2.67x, 15 thắng / 1 thua. Case thua duy nhất là
`layer_norm.fp32.h4096.b1024` (0.96x) — PyTorch dùng kernel layer_norm rất
tốt cho fp32. Các case fuse `norm_quant`, `qk_norm`, `l2_norm` nhanh 9.8–10.4x.

### 7.3 activation (B1)

9 case, geomean 2.09x. `silu.bf16.n4m` thua nhẹ (0.91x) vì case 4 phần tử quá
nhỏ, launch overhead chi phối; `act_and_mul_quant` nhanh 8.1x.

### 7.4 position (B1)

22 case, geomean 13.74x, thắng cả 22. `build_cos_sin_cache` chỉ 1.35–1.63x vì
tính góc fp64; các case rope/fused qk-norm-rope rất nhỏ nên tỉ lệ 90–145x chủ
yếu phản ánh overhead của eager PyTorch, không phải throughput.

### 7.5 embedding (B1)

11 case, geomean 3.43x, thắng cả 11. `lookup.fp32` chỉ 1.05x (memory-bound);
`logits_postprocess.r1` nhanh 15.7x.

### 7.6 merge + attention (B1/B8)

21 case B1/B2/B8: 4 case merge thắng 15–32x; 5 case attention prefill thua
SDPA 0.70–0.94x; 12 case decode/MLA không có baseline (decode dùng SDPA math
fallback). Đây là family có median <1 vì attention prefill chiếm nhóm có
baseline.

### 7.7 sampling (B6/B10)

136 case, geomean **6.40x**, **136 thắng / 0 thua** sau cập nhật 20:02
(trước: 3.63x, 109/26). Toàn bộ case `sample()` cũ (greedy/top-p/min-p) đã
đi qua body fused; top-k chuyển sang candidate histogram; p≥0.95 dùng exact
mass40 histogram nên không còn rơi vào fallback. Xem §5.4 và
`docs/SAMPLING_BENCH.md`.

### 7.8 chains (B1/B3/B6)

10 case, 8 case có PyTorch, geomean 8.63x (3.7–26x). So với chain reference
của chính kops: `pre_norm_quant` nhanh 1.57x, `glu_quant` nhanh 1.31x. Chain
`logits_pipeline` và `embedding_norm` ở 3.7–4.9x.

### 7.9 kvcache (B2)

28 case, không có baseline PyTorch. Xem §5.6.

### 7.10 state (B2/B9/B10)

22 case, chỉ `state_commit` có baseline. Xem §5.7.

### 7.11 quantization (B3/B4)

22 case, geomean 12.16x, thắng cả 22. Pack weight (`gptq_to_marlin`,
`awq_to_marlin`, `fp4_to_marlin`) nhanh 80–230x so với đường PyTorch tương
đương; các case quantize runtime nhanh 2.6–9x.

### 7.12 gemm (B4/B5/B7)

123 case, 83 có baseline, geomean 9.09x, 72 thắng / 10 thua. Điểm mạnh: GEMV
lượng tử (100–185x), BF16/FP8/INT8 tensor core (xem §5.5). Điểm yếu: M=64
(0.69–0.95x), và 40 case không có baseline (marlin, GGUF MMQ, FP8/INT8 MMA
lớn, LoRA).

### 7.13 moe (B5/B6/B7)

13 case, geomean 5.47x, 11 thắng / 2 hòa. `moe.align` nhanh 37.8x,
`moe.route` 13.8x; các case memory-bound (`moe.sum`, `moe.permute`) hòa.

## 8. Tái lập

```bash
# 1. Build đúng SM120a, kèm test và benchmark.
cmake --preset bench-sm120a
cmake --build --preset bench-sm120a -j 8

# 2. Cổng đúng đắn (GPU 1 nếu GPU 0 bận).
CUDA_VISIBLE_DEVICES=1 ctest --test-dir build/bench-sm120a --output-on-failure

# 3. kops + PyTorch cho từng family, 3 round.
B=build/bench-sm120a/benchmark
for f in model_ops norm activation merge position embedding sampling chains \
         kvcache state quantization gemm moe; do
  $B/bench_$f --dump-cases > out/rtx5060/cases/$f.json
  $B/bench_$f --rounds 3 --json out/rtx5060/kops/$f.json
  python3 benchmark/compare/torch_baseline.py --cases out/rtx5060/cases/$f.json \
      --rounds 3 --json out/rtx5060/torch/$f.json
  python3 benchmark/compare/compare_results.py out/rtx5060/kops/$f.json \
      out/rtx5060/torch/$f.json --json out/rtx5060/compare/$f.json
done

# 4. Third-party B10 (cold cache).
$B/bench_sampling --only spec. --flush-l2 256 --rounds 3 \
    --json out/rtx5060/thirdparty/kops-spec-cold.json
python3 benchmark/compare/torch_baseline.py --cases out/rtx5060/cases/sampling.json \
    --only spec. --flush-l2 256 --rounds 3 \
    --json out/rtx5060/thirdparty/torch-spec-cold.json
out/rtx5060/bin/hpc_sampler --flush-mb 256 --json out/rtx5060/thirdparty/hpc-sampler.json
out/rtx5060/bin/sglang_tree --flush-mb 256 --json out/rtx5060/thirdparty/sglang-tree.json

# 5. Third-party B8 (dense prefill).
python3 benchmark/compare/thirdparty.py --cases out/rtx5060/cases/merge.json \
    --impl sdpa --rounds 3 --json out/rtx5060/thirdparty/sdpa-merge.json

# 6. Third-party B5 (llama.cpp GGUF, cold).
$B/bench_gemm --only gguf_mmvq --flush-l2 256 --rounds 3 \
    --json out/rtx5060/thirdparty/kops-gguf-cold.json
python3 benchmark/compare/torch_baseline.py --cases out/rtx5060/cases/gemm.json \
    --only gguf_mmvq --flush-l2 256 --rounds 3 \
    --json out/rtx5060/thirdparty/torch-gguf-cold.json
out/rtx5060/bin/llamacpp_mmvq > out/rtx5060/thirdparty/llamacpp-gguf.csv
```

### 8.2 Phần II (B11–B17)

```bash
# B11/B15: dùng lại JSON đã có trong out/b11-b15 (bản opt-*).
# Nếu muốn đo lại bằng harness hiện tại:
$B/bench_mla --rounds 3 --json out/rtx5060/kops/mla.json
$B/bench_sparse --dump-cases > out/rtx5060/cases/sparse-new.json
$B/bench_sparse --rounds 3 --json out/rtx5060/kops/sparse-new.json
python3 benchmark/compare/torch_baseline.py --cases out/rtx5060/cases/sparse-new.json \
    --rounds 3 --json out/rtx5060/torch/sparse-new.json
python3 benchmark/compare/csa_torch.py --cases out/rtx5060/cases/sparse-new.json \
    --only compressed_attention --rounds 3 --json out/rtx5060/torch/sparse-csa.json
$B/bench_hyper_connection --rounds 3 --json out/rtx5060/kops/hyper_connection.json
$B/bench_model_ops --only attn_res --rounds 3 --json out/rtx5060/kops/attn_res.json

# B13 (SM120 tile) và B16 (mega).
$B/bench_gemm --only mma_sm120 --rounds 3 --json out/rtx5060/kops/b13_gemm.json
$B/bench_mega --rounds 3 --json out/rtx5060/kops/mega.json

# B12 và B17: xem tools/dev/run_b10_bench.sh và out/b12/, out/b17/.
# B12 dùng 2 GPU (torch.distributed/NCCL); B17 dùng build riêng của
# attention.rs qua tools/dev/build_lowbit_baselines.sh.
```

Lưu ý: `torch_baseline.py` trong cây hiện tại có thêm nhánh
`sample_mode`/`sample_fused` (mục 5.4); nếu dùng bản chưa có nhánh này thì 102
case sampler sẽ bị bỏ qua. Baseline SGLang cần header FlashInfer; script dùng
bản 0.6.14 tải riêng, không thêm dependency vào repo.

## 9. Hạn chế

1. **Dirty tree:** binary build từ `aa4e2b8` + cây dirty, trong đó có
   `csrc/attention/sparse.cu` đang sửa. HEAD đã moved khỏi `aa4e2b8` (lúc viết là `8ecaf00`)
   bởi một session khác trong lúc đo; các file `csrc/gemm/*`, `tests/unit/test_sm120.cc`
   tiếp tục thay đổi sau đó. Số liệu là diagnostic-only cho đúng binary đã đo.
2. **Một GPU consumer:** RTX 5060 Ti không đại diện cho H100/B200; các kết
   luận về WGMMA/tcgen05 không suy ra được từ báo cáo này.
3. **Hot-L2 mặc định:** nhiều case nhỏ nằm trong L2, GB/s vượt trần DRAM.
   Chỉ các so sánh third-party dùng cold cache; B2/B9 đọc số thô cẩn thận.
4. **Không có baseline PyTorch cho B2/B9:** không có primitive tương đương;
   báo cáo dùng roofline/thời gian thô, không tuyên bố thắng/thua.
5. **Thiếu baseline cho 40 case GEMM lượng tử:** marlin, GGUF MMQ, FP8/INT8
   MMA lớn, LoRA; `torch_baseline.py` không có đường tương đương.
6. **Attention decode không có baseline:** SDPA decode là math fallback;
   FlashInfer/FA2 không có trong môi trường.
7. **GPU 0 bận:** đã dùng GPU 1; hai GPU cùng model nhưng clock/temperature
   có thể khác nhau vài %.
8. **Số round 3:** đủ để so sánh family, không đủ cho tuning tactic/autotune.
9. **Test B14:** bản 18:4x fail `kops_test_sparse`; bản cập nhật 19:5x pass
   cả 5 test B14. Các test ngoài B14 vẫn là ảnh chụp 18:4x, chưa chạy lại toàn
   bộ suite.
10. **B11–B17 không phủ 1:1 kernel:** B11 4/12 kernel chạy SM120, B13 1/12
    cell có case, B14 4/8 kernel, B16 4/6 kernel; phần còn lại là
    compile-only/handoff sm90a/sm100a, không có số đo SM120.
11. **B12 dùng driver `b12-manual`** (flags hash riêng) và transport
    pinned-host không P2P; số GB/s không phải NVLink, không so với H100.
12. **B16 megakernel chậm hơn modular 6.6–22.9x**; đây là kết quả âm, chưa
    đạt gate "nhanh hơn modular + CUDA graph".
13. **B17 là family thử nghiệm**, chậm hơn attention.rs ở store/prefill/pack;
    registry giữ experimental, chưa nên dùng thay thế.
14. **Số liệu B11/B12/B15/B17 lấy từ `out/` của session khác** (cùng
    GPU/commit nhưng có thể chạy ở thời điểm khác); B13/B14/B16 do báo cáo
    này chạy.
15. **Ảnh chụp theo thời điểm:** Phần I đo lúc 18:4x. Sampler được đo lại lúc
    **20:02** trên cùng GPU, 3 round, bằng binary sau tối ưu; các bảng sampler
    trong §1/§5.1/§5.4/§7.7 đã cập nhật, còn các hàng `sampling.*` ở Phụ lục A
    vẫn là ảnh chụp 18:4x. Muốn tái lập chính xác phải dùng đúng
    build/commit của từng lần đo.
16. **Sampler update là diagnostic-only:** binary sampler 20:02 build từ cây
    dirty (gồm `sample()` fast path, exact mass40 histogram p≥0.95, top-k
    candidate histogram, gumbel candidate histogram, argmax 1024 thread), chưa
    chạy lại toàn bộ 26 test ngoài `kops_test_sampling`. `kops_test_sampling`
    PASS; các family khác giữ số 18:4x.


## Phần II — B11–B17

Phần II dùng lại các lần đo đã có trong `out/` trên cùng GPU: B11/B14/B15 lấy
`out/b11-b15/opt-*.json` (provenance `aa4e2b8` + dirty), B12 lấy
`out/b12/kops-opt-clean.json` (driver `b12-manual`, provenance riêng), B17 lấy
`out/b17/*.json` (flags hash riêng). Hai phần chưa có dữ liệu được chạy mới
bằng `bench-sm120a`: `bench_mega` (B16) và `bench_gemm --only mma_sm120` (B13).
Tất cả vẫn là RTX 5060 Ti / SM120a / CUDA 13.4.

### 10. Tổng quan B11–B17

| bước | kernel | case | có ref | geomean | median | thắng / thua | ghi chú |
|---|---:|---:|---:|---:|---:|---:|---|
| B11 | 12 | 37 | 8 | 1.13x | 1.13x | 5 / 3 | MLA decode + metadata + prepack; 3/8 ref thua, split-KV thắng |
| B12 | 11 | 64 | 57 | 1.59x | 0.90x | 24 / 33 | collectives + EP + comm-GEMM; NCCL thắng ở size lớn |
| B13 | 12 | 2 | 2 | 1.05x | 1.05x | 2 / 0 | chỉ cell BF16 SM120 có case; FP8/FP4/grouped SM120 chưa có case |
| B14 | 8 | 22 | 22 | 2.52x | 2.42x | 20 / 2 | topk/indexer/sparse MLA/compressed attention; 2 case thua |
| B15 | 8 | 24 | 24 | 13.47x | 16.77x | 24 / 0 | hyper-connection + attn_res; thắng cả 24 |
| B16 | 6 | 4 | 3 | 13.37x | 15.81x | 3 / 0 | mega vs modular; mega chậm hơn 6.6–22.9x |
| B17 | 5 | 11 | 6 | 0.64x | 0.69x | 0 / 5 | lowbit + moe_w2; attention.rs nhanh hơn ở store/prefill |
| **B11–B17** | **62** | **164** | **119** | **2.47x** | **1.92x** | **75 / 43** | **B16 tách riêng: mega/modular 13.37x (mega chậm hơn)** |

Tổng Phần II: **164 case**, **119 case có baseline ngoài** (B16 không tính vì
baseline là chuỗi modular nội bộ). Geomean ngoài B16: **2.47x**, median
**1.92x**, 75 thắng / 43 thua. Cộng với Phần I: **607 case** trên toàn bộ
B0–B17, **459 case có baseline**.

Độ phủ không 1:1 với số kernel:

| bước | kernel khai báo | kernel có case | kernel chưa có case trên SM120 |
|---|---:|---:|---|
| B11 | 12 | 4 | 8: `gemm_wgmma`, `fp8_gemm_wgmma`, `int8_gemm_wgmma`, `machete_gemm`, `fp4_to_fp8_gemm_wgmma`, `grouped_gemm_wgmma`, `grouped_fp4_to_fp8_gemm_wgmma`, `grouped_mixed_input_gemm` |
| B12 | 11 | 11 | — |
| B13 | 12 | 1 | 11: 3 cell FP8/FP4/grouped SM120 + 8 kernel `tcgen05`/NVFP4 SM100/103 |
| B14 | 8 | 4 | 4: `kv_compressor`, `kv_block_summary`, `block_topk_select`, `block_sparse_prefill` |
| B15 | 8 | 8 (3 nhóm case) | — (5 kernel HC đi qua `hc_chain`, 2 kernel AttnRes, 1 `mega_mhc`) |
| B16 | 6 | 4 | 2: `mega_moe_ep`, `task_queue_reset` |
| B17 | 5 | 5 | — |

Các kernel `wgmma`/`tcgen05` còn lại là compile-only hoặc handoff kit
sm90a/sm100a, không chạy trên SM120.

### 11. B11 — MLA decode + prepack

37 case: 30 case `mla_decode` (BF16/FP8, KV 64k/256k/1m, q1/q4, split-KV
s8/s16/s32), 4 case `mla_metadata`, 3 case prepack (`machete_prepack` int4/int8,
`grouped_prepare`). 8 case có baseline torch MLA (64k/1m, q1, q1.s8).

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `mla_decode.bf16.kv64k.q1` | 3.0126 | torch-mla | 0.8617 | 0.29x | 25.1 | 6.06 |
| `mla_decode.fp8.kv64k.q1` | 4.6710 | torch-mla | 1.7161 | 0.37x | 9.3 | 3.91 |
| `mla_decode.bf16.kv1m.q1` | 48.2594 | torch-mla | 42.9691 | 0.89x | 25.0 | 6.05 |
| `mla_decode.fp8.kv1m.q1` | 72.4652 | torch-mla | 81.5731 | 1.13x | 9.5 | 4.03 |
| `mla_decode.bf16.kv64k.q1.s8` | 0.7627 | torch-mla | 0.8613 | 1.13x | 99.3 | 23.93 |
| `mla_decode.bf16.kv1m.q1.s8` | 12.0894 | torch-mla | 48.8590 | 4.04x | 99.9 | 24.16 |
| `mla_decode.fp8.kv64k.q1.s8` | 1.1545 | torch-mla | 1.7155 | 1.49x | 37.5 | 15.81 |
| `mla_decode.fp8.kv1m.q1.s8` | 18.1181 | torch-mla | 69.7230 | 3.85x | 38.0 | 16.12 |
| `mla_decode.bf16.kv64k.q1.s32` | 0.7663 | — | — | — | 98.9 | 23.82 |
| `mla_decode.bf16.kv1m.q1.s32` | 12.0748 | — | — | — | 100.1 | 24.19 |
| `mla_decode.fp8.kv64k.q1.s32` | 1.0579 | — | — | — | 40.9 | 17.25 |
| `mla_decode.fp8.kv1m.q1.s32` | 17.9832 | — | — | — | 38.3 | 16.24 |
| `mla_metadata.b8.parts132.block64` | 0.0197 | — | — | — | 0.2 | 0.00 |
| `mla_metadata.b32.parts132.block64` | 0.0213 | — | — | — | 0.2 | 0.00 |
| `mla_metadata.b128.parts192.block64` | 0.0339 | — | — | — | 0.2 | 0.00 |
| `mla_metadata.b32.parts132.block16` | 0.0213 | — | — | — | 0.2 | 0.00 |
| `gemm.machete_prepack.int4.4096x4096` | 0.0294 | — | — | — | 571.6 | 0.00 |
| `gemm.machete_prepack.int8.4096x4096` | 0.0251 | — | — | — | 1334.9 | 0.00 |
| `gemm.grouped_prepare.e32.1024x1024` | 0.0068 | — | — | — | 0.9 | 0.00 |

Đọc bảng:

- **Split-KV là bắt buộc.** Ở 64k, `q1` không split mất 3.01 ms (25 GB/s,
  6.1 TF) và chỉ đạt 0.29x torch; bật `s8` còn 0.76 ms (99 GB/s, 23.9 TF),
  nhanh 1.13x torch. Ở 1M, `s32` cho 12.07 ms / 100 GB/s / 24.2 TF, nhanh
  khoảng 4x torch.
- **FP8 chậm hơn BF16 trên SM120** ở cùng shape: 1M q1 s32 là 17.98 ms
  (38 GB/s, 16.2 TF) so với BF16 12.07 ms. Dequant FP8 bằng software là điểm
  nghẽn, đúng như ghi chú T1 trong roadmap.
- **q4 không giúp decode**: q4 không split vẫn 3.01 ms; q4 + s8 ở 1M là
  72.07 ms (chậm hơn q1 + s8 6x). Đường q4 chỉ có lợi khi phần sinh token
  chiếm ưu thế, không phải ở decode attention thuần.
- **Metadata/prepack rất nhẹ**: metadata 0.020–0.034 ms; machete prepack
  0.0294/0.0251 ms; grouped prepare 0.0068 ms.
- Geomean 8 case có ref là 1.13x, 5 thắng / 3 thua; 3 thua đều là q1 không
  split (64k bf16 0.29x, 64k fp8 0.37x, 1M bf16 0.89x).

### 12. B12 — Communication

64 case: 41 collectives (allreduce/allgather/reduce_scatter, kể cả two-shot),
12 fusion residual+RMSNorm (BF16 và FP8), 5 EP dispatch/combine (rank ảo),
5 comm-GEMM, 1 peer barrier. 57 case có baseline `pytorch-nccl`; EP và
peer barrier không có baseline.

| nhóm | case | có ref | geomean | thắng / thua |
|---|---:|---:|---:|---:|
| collectives | 41 | 41 | 1.31x | 16 / 25 |
| fusion | 12 | 12 | 4.40x | 8 / 4 |
| ep | 5 | 0 | 0.00x | 0 / 0 |
| comm_gemm | 5 | 4 | 0.56x | 0 / 4 |
| other | 1 | 0 | 0.00x | 0 / 0 |

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `allgather_bf16_t1_h4096` | 0.0073 | pytorch-nccl | 0.0578 | 7.87x | 1.1 | 0.00 |
| `allgather_bf16_t2048_h4096` | 3.4405 | pytorch-nccl | 3.1059 | 0.90x | 4.9 | 0.00 |
| `allgather_gemm_bf16_w2_m512_n1024_k1024_c8` | 0.3885 | pytorch-nccl | 0.2365 | 0.61x | 8.1 | 5.53 |
| `allgather_gemm_bf16_w4_m256_n1024_k1024_c8` | 0.7689 | — | — | — | 3.4 | 2.79 |
| `allreduce_bf16_t1_h4096` | 0.0071 | pytorch-nccl | 0.0657 | 9.32x | 1.2 | 0.00 |
| `allreduce_bf16_t2048_h4096` | 3.7528 | pytorch-nccl | 3.1193 | 0.83x | 4.5 | 0.00 |
| `allreduce_bf16_t8192_h4096` | 14.2032 | pytorch-nccl | 12.3499 | 0.87x | 4.7 | 0.00 |
| `allreduce_bf16_t8_h4096` | 0.0201 | pytorch-nccl | 0.0602 | 3.00x | 3.3 | 0.00 |
| `allreduce_fusion_bf16_t1_h4096` | 0.0055 | pytorch-nccl | 0.1038 | 18.90x | 1.5 | 0.00 |
| `allreduce_fusion_bf16_t2048_h4096` | 3.7056 | pytorch-nccl | 3.2060 | 0.87x | 4.5 | 0.00 |
| `allreduce_fusion_fp8_bf16_t1_h4096` | 0.0064 | pytorch-nccl | 3.8374 | 595.86x | 0.6 | 0.00 |
| `allreduce_two_shot_bf16_t2048_h4096` | 5.2204 | pytorch-nccl | 3.2602 | 0.62x | 3.2 | 0.00 |
| `ep_dispatch_combine_bf16_ll_w4_t256_h4096_e64_k8` | 18.0247 | — | — | — | 0.9 | 0.00 |
| `ep_dispatch_combine_bf16_normal_w2_t128_h2048_e32_k4` | 0.5051 | — | — | — | 2.1 | 0.00 |
| `ep_dispatch_combine_bf16_normal_w4_t256_h4096_e64_k8` | 5.1592 | — | — | — | 1.6 | 0.00 |
| `ep_dispatch_combine_bf16_normal_w8_t64_h1024_e16_k4` | 1.2756 | — | — | — | 0.8 | 0.00 |
| `ep_layout_bf16_e64_t1024_k8` | 0.0062 | — | — | — | 0.0 | 0.00 |
| `gemm_allreduce_bf16_w2_m1024_n1024_k1024_c8` | 0.4859 | pytorch-nccl | 0.4459 | 0.92x | 4.3 | 4.42 |
| `gemm_allreduce_norm_bf16_w2_m512_n4096_k4096_c8` | 3.6011 | pytorch-nccl | 1.1387 | 0.32x | 1.2 | 4.77 |
| `gemm_reduce_scatter_bf16_w2_m1024_n1024_k1024_c8` | 0.4573 | pytorch-nccl | 0.2470 | 0.54x | 4.6 | 4.70 |
| `peer_barrier_bf16_w2` | 0.0043 | — | — | — | 0.0 | 0.00 |
| `reduce_scatter_bf16_t1_h4096` | 0.0072 | pytorch-nccl | 0.0465 | 6.45x | 1.1 | 0.00 |
| `reduce_scatter_bf16_t2048_h4096` | 3.7429 | pytorch-nccl | 3.3676 | 0.90x | 4.5 | 0.00 |

Đọc bảng:

- **Geomean 1.59x nhưng median 0.90x**: kops thắng ở message nhỏ
  (launch-bound) và thua NCCL ở message lớn. Ví dụ allreduce t1 0.0071 ms
  vs 0.066 ms (9.3x), nhưng t2048 3.75 ms vs 3.12 ms (0.83x) và t8192
  14.2 ms vs 12.35 ms (0.87x). NCCL thắng 33/57 case.
- **Fusion residual+RMSNorm là điểm mạnh**: 4.40x geomean, 8/12 thắng; t1
  BF16 0.0055 ms vs 0.104 ms (18.9x), FP8 t1 0.0064 ms vs 3.84 ms (596x —
  baseline PyTorch làm cả chuỗi cast/quant nên rất chậm).
- **comm-GEMM thua**: 0.56x geomean, 0/4 thắng; `gemm_allreduce_norm`
  3.60 ms vs 1.14 ms (0.31x). Đây là đường cần tối ưu nếu dùng TP.
- **EP** 5 case không có baseline (DeepEP không có trên máy); chỉ báo cáo
  thời gian.
- **Transport**: 2 GPU không P2P, pinned-host staging; GB/s trong bảng là
  host-staged, không phải NVLink. Đừng so với H100/NVLink.

### 13. B13 — SM120 tile

2 case `gemm.mma_sm120` (tile 64×128 ở 2048², 128×128 ở 4096²), so với cùng
op `mma_bf16` của PyTorch:

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `gemm.mma_sm120.bf16.m2048.2048x2048.t64x128` | 0.3646 | pytorch | 0.3734 | 1.02x | 69.0 | 47.13 |
| `gemm.mma_sm120.bf16.m4096.4096x4096.t128x128` | 2.7837 | pytorch | 2.9681 | 1.07x | 36.2 | 49.37 |

- 2048²: 0.3646 ms, **47.1 TFLOP/s**, 1.02x PyTorch.
- 4096²: 2.7837 ms, **49.4 TFLOP/s**, 1.07x PyTorch.
- Đây là cell BF16 của B13. B13 còn 11 kernel (`fp8_gemm_sm120`,
  `fp4_gemm_sm120`, `grouped_gemm_sm120`, các `tcgen05` cho SM100/103,
  `concat_and_cache_ds_mla_nvfp4`) chưa có case benchmark trên SM120; chúng
  đang ở mức compiled/handoff, không nằm trong bảng này.

### 14. B14 — Sparse attention

22 case: 10 `sparse_topk`, 3 `lightning_indexer`, 2 `sparse_mla`, 7
`compressed_attention` (5 CSA + 2 HCA — phần mới của bản cập nhật B14 ngày
2026-11-10). **22/22 case có baseline**: torch cho topk/indexer/sparse MLA,
`benchmark/compare/csa_torch.py` cho CSA/HCA.

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `sparse_topk.fp32.r1.kv64k.topk512` | 0.0350 | torch | 0.0966 | 2.76x | 7.6 | 0.00 |
| `sparse_topk.fp32.r16.kv64k.topk512` | 0.0365 | torch | 0.1019 | 2.79x | 116.7 | 0.00 |
| `sparse_topk.fp32.r1.kv1m.topk512` | 0.0267 | torch | 0.0991 | 3.71x | 157.3 | 0.00 |
| `sparse_topk.fp32.r16.kv1m.topk512` | 0.7543 | torch | 0.9807 | 1.30x | 89.0 | 0.00 |
| `sparse_topk.fp32.r1.kv64k.topk2048` | 0.0407 | torch | 0.0951 | 2.34x | 6.8 | 0.00 |
| `sparse_topk.fp32.r16.kv64k.topk2048` | 0.0414 | torch | 0.1040 | 2.51x | 107.8 | 0.00 |
| `sparse_topk.fp32.r1.kv1m.topk2048` | 0.0268 | torch | 0.0991 | 3.70x | 157.1 | 0.00 |
| `sparse_topk.fp32.r16.kv1m.topk2048` | 0.7677 | torch | 0.9754 | 1.27x | 87.8 | 0.00 |
| `sparse_topk.fp32.r1.kv256k.topk2048` | 0.0186 | torch | 0.0949 | 5.11x | 57.4 | 0.00 |
| `sparse_topk.fp32.r4.kv1m.topk2048` | 0.4372 | torch | 0.2026 | 0.46x | 38.5 | 0.00 |
| `lightning_indexer.bf16.r1.kv64k` | 0.0358 | torch | 0.3581 | 10.01x | 476.6 | 0.00 |
| `lightning_indexer.bf16.r1.kv1m` | 0.6578 | torch | 7.6332 | 11.60x | 414.5 | 0.00 |
| `lightning_indexer.bf16.r16.kv64k` | 0.5033 | torch | 5.6181 | 11.16x | 42.2 | 0.00 |
| `sparse_mla.bf16.h128.topk512` | 0.0489 | torch | 0.1139 | 2.33x | 17.8 | 0.00 |
| `sparse_mla.bf16.h128.topk2048` | 0.1832 | torch | 0.1172 | 0.64x | 14.4 | 0.00 |
| `compressed_attention.bf16.csa.t1.h128.pool4k.topk512.w512.sink` | 0.0918 | torch | 0.2751 | 3.00x | 14.3 | 2.92 |
| `compressed_attention.bf16.csa.t1.h128.pool4k.topk2048.w512.sink` | 0.2218 | torch | 0.2783 | 1.26x | 13.0 | 3.03 |
| `compressed_attention.bf16.csa.t1.h128.pool64k.topk2048.w512.sink` | 0.2218 | torch | 0.2777 | 1.25x | 13.0 | 3.03 |
| `compressed_attention.bf16.csa.t16.h128.pool4k.topk2048.w512.sink` | 0.5732 | torch | 1.3128 | 2.29x | 80.5 | 18.73 |
| `compressed_attention.bf16.csa.t16.h128.pool64k.topk2048.w512.sink` | 0.5998 | torch | 1.4001 | 2.33x | 76.9 | 17.90 |
| `compressed_attention.bf16.hca.t1.h128.dense512.w512.sink` | 0.0935 | torch | 0.2549 | 2.73x | 14.0 | 2.87 |
| `compressed_attention.bf16.hca.t16.h128.dense512.w512.sink` | 0.2382 | torch | 0.5108 | 2.14x | 88.1 | 18.03 |

- Geomean **2.52x**, median **2.42x**, **20 thắng / 2 thua**.
- `sparse_topk`: r1 đạt 2.34–3.71x, `kv256k` 5.11x; r16 1.27–1.30x. Case thua
  là `r4.kv1m.topk2048` 0.46x (0.437 ms vs 0.203 ms).
- `lightning_indexer`: 10.0–11.6x.
- `sparse_mla.topk512` 2.33x; `topk2048` 0.64x — vẫn là trade-off của top-k lớn
  (gather/dequant gấp 4 lần, PyTorch chỉ chạy dense attention).
- `compressed_attention`: CSA 1.25–3.00x, HCA 2.14–2.73x; t16 đạt
  **17.9–18.7 TFLOP/s**. Đây là phần mới của B14, trước đây không có case.
- `kops_test_sparse` **PASS** sau bản cập nhật (`build/final`, 19:47); bản cũ
  fail illegal memory access. Số liệu lấy từ `out/rtx5060/kops/sparse-new.json`,
  `torch/sparse-new.json` và `torch/sparse-csa.json`.
- B14 còn 4/8 kernel chưa có case: `kv_compressor`, `kv_block_summary`,
  `block_topk_select`, `block_sparse_prefill` (chỉ có test/compile).

### 15. B15 — Hyper-connection + AttnRes

24 case: 6 `hc_chain`/`mega_mhc` (t1/t64/t1024) và 18 `attn_res`
(t1/t16/t256 × s1/s4/s8 × có/không norm). Cả 24 có baseline PyTorch.

Bảng HC:

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `model_ops.hc_chain.bf16.t1.h7168.n4` | 0.0790 | torch | 0.3502 | 4.44x | 38.3 | 0.02 |
| `model_ops.mega_mhc.bf16.t1.h7168.n4` | 0.0786 | torch | 0.3508 | 4.46x | 38.5 | 0.02 |
| `model_ops.hc_chain.bf16.t64.h7168.n4` | 0.1650 | torch | 0.3494 | 2.12x | 116.9 | 0.53 |
| `model_ops.mega_mhc.bf16.t64.h7168.n4` | 0.1706 | torch | 0.3500 | 2.05x | 113.1 | 0.52 |
| `model_ops.hc_chain.bf16.t1024.h7168.n4` | 1.0362 | torch | 5.2231 | 5.04x | 257.9 | 1.36 |
| `model_ops.mega_mhc.bf16.t1024.h7168.n4` | 1.0547 | torch | 5.2233 | 4.95x | 253.3 | 1.34 |

Bảng AttnRes (chọn):

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `model_ops.attn_res.bf16.t1.h7168.s1` | 0.0047 | torch | 0.1452 | 31.17x | 15.4 | 0.01 |
| `model_ops.attn_res.bf16.t1.h7168.s8` | 0.0111 | torch | 0.1550 | 14.02x | 15.6 | 0.02 |
| `model_ops.attn_res.bf16.t16.h7168.s1` | 0.0047 | torch | 0.1562 | 33.17x | 243.5 | 0.19 |
| `model_ops.attn_res.bf16.t16.h7168.s8` | 0.0119 | torch | 0.1564 | 13.16x | 231.7 | 0.35 |
| `model_ops.attn_res.bf16.t256.h7168.s1` | 0.0145 | torch | 0.1723 | 11.85x | 1262.0 | 1.01 |
| `model_ops.attn_res.bf16.t256.h7168.s8` | 0.0685 | torch | 1.8422 | 26.91x | 643.4 | 0.96 |

- HC geomean 3.59x, 6/6 thắng; t1024 đạt 258 GB/s (5.05x torch).
- AttnRes geomean 20.92x, 18/18 thắng; t256 s1 đạt 1262 GB/s. Các case lớn
  bị gắn `!byte-count` (băng thông > trần) vì byte-count của case tính cả
  đọc/ghi và working set nằm trong L2 — đọc tỉ lệ, đừng đọc GB/s tuyệt đối.
- B15 case phủ 3 nhóm kernel (`hc_chain` ghép 5 kernel HC, `mega_mhc`,
  `attn_res`/`attn_res_commit`), không phải 1 case/kernel.

### 16. B16 — Megakernel

4 case, mỗi case 4 biến thể: mega, mega trong CUDA graph, modular, modular
trong CUDA graph (MoE chỉ có 2 biến thể mega vì orchestrator modular chưa có
thân).

| case | mega ms | mega graph ms | modular ms | modular graph ms | mega/modular |
|---|---:|---:|---:|---:|---:|
| `mega.mlp.m8.h1024.i2048` | 0.1376 | 0.1420 | 0.0208 | 0.0226 | 6.6x |
| `mega.attn.m1.h1024.hq8.hkv2.d128.kv64` | 0.5800 | 0.5812 | 0.0254 | 0.0247 | 22.9x |
| `mega.moe.m8.h512.i512.e8.k2` | 0.3958 | 0.3973 | — | — | — |
| `mega.decoder.m1.h512.i1024` | 0.5100 | 0.5105 | 0.0323 | 0.0308 | 15.8x |

- **Megakernel chưa có lợi trên RTX 5060 Ti** ở các shape nhỏ: MLP chậm hơn
  modular **6.6x**, attention **22.9x**, decoder **15.8x**. CUDA graph không
  cứu được (mega graph tương đương mega).
- MoE local chỉ có mega (0.3958 ms), không có modular để so.
- Kết luận: B16 chưa đạt gate "nhanh hơn modular + CUDA graph"; cần batch lớn
  hơn, nhiều SM hơn hoặc sửa chi phí barrier/scheduler trước khi dùng.
- 6 kernel B16: case phủ `mega_mlp`, `mega_attention_block`,
  `mega_moe_local`, `mega_decoder`; `mega_moe_ep` (cần multi-rank) và
  `task_queue_reset` (nội bộ) không có case riêng.

### 17. B17 — Low-bit KV + MoE W2

11 case: 9 lowbit (store/decode/prefill, turbo3/turbo4/prod3) và 2 `moe_w2`
(pack/gemm). 6 case có baseline attention.rs (dev-only build từ
`3rd/attention.rs`, không link vào kernel-ops).

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `lowbit.store.bf16.t2048.h8.d128.turbo4` | 0.0574 | — | — | — | 184.8 | 0.00 |
| `lowbit.store.bf16.t2048.h8.d128.turbo3` | 0.0604 | — | — | — | 171.4 | 0.00 |
| `lowbit.store.bf16.t4096.h8.d128.turbo4` | 0.1124 | attention.rs | 0.0509 | 0.45x | 188.8 | 0.00 |
| `lowbit.store.bf16.t4096.h8.d128.turbo3` | 0.1188 | attention.rs | 0.0451 | 0.38x | 174.3 | 0.00 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4` | 0.6804 | attention.rs | 0.6923 | 1.02x | 52.6 | 0.00 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo3` | 0.5563 | attention.rs | 0.4843 | 0.87x | 56.8 | 0.00 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.prod3` | 0.5702 | — | — | — | 57.2 | 0.00 |
| `lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo4` | 2.9694 | attention.rs | 2.5425 | 0.86x | 0.0 | 10.85 |
| `lowbit.prefill.bf16.d128.t512.kv4096.h32x8.prod3` | 5.7487 | — | — | — | 0.0 | 5.60 |
| `moe.w2.pack.bf16.e8.n2048.k4096.g32` | 0.4273 | attention.rs | 0.2260 | 0.53x | 373.0 | 0.00 |
| `moe.w2.gemm.bf16.e8.n2048.k4096.slots2048` | 2.0442 | — | — | — | 24.6 | 16.81 |

So với attention.rs:

| case | kops ms | attention.rs ms | 3rd / kops | ghi chú |
|---|---:|---:|---:|---|
| `lowbit.store.bf16.t4096.h8.d128.turbo4` | 0.1124 | 0.0509 | 0.45x | attention.rs nhanh hơn |
| `lowbit.store.bf16.t4096.h8.d128.turbo3` | 0.1188 | 0.0451 | 0.38x | attention.rs nhanh hơn |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4` | 0.6804 | 0.6923 | 1.02x | kops nhanh hơn 2% |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo3` | 0.5563 | 0.4843 | 0.87x | attention.rs nhanh hơn |
| `lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo4` | 2.9694 | 2.5425 | 0.86x | attention.rs nhanh hơn |
| `moe.w2.pack` (kops bf16 → pack; 3rd đã packed MXFP4) | 0.4273 | 0.2260 | 0.53x | khác định dạng đầu vào |

- Geomean 6 case có ref là **0.64x** (kops chậm hơn); 1 hòa (decode turbo4,
  kops nhỉnh 2%), 0 thắng, 5 thua theo ngưỡng ±2%. Store chậm 2.2–2.6x;
  prefill chậm 1.17x; W2 pack chậm 1.89x (khác định dạng đầu vào: kops nhận
  BF16 và tự pack, attention.rs nhận MXFP4 đã pack).
- Đây là family thử nghiệm (B17), registry giữ ở mức experimental; kết quả
  này đủ để nói "chưa nên dùng thay attention.rs" cho low-bit KV.
- Chất lượng codec (NMSE/cosine/SQNR, exp(KL) proxy) nằm ở
  `out/b17/quality.md`; báo cáo này chỉ đo hiệu năng.

### 18. Kết luận Phần II

| bước | kết luận ngắn |
|---|---|
| B11 | Split-KV s8/s16/s32 rất hiệu quả; q1 không split thua torch ở 64k. FP8 software dequant là điểm nghẽn. |
| B12 | Cạnh tranh ở message nhỏ và fusion; NCCL thắng ở size lớn; comm-GEMM cần tối ưu. |
| B13 | Cell BF16 SM120 đạt 49.4 TF, 1.07x PyTorch; các cell FP8/FP4/tcgen05 chưa có case. |
| B14 | Top-k 2.3–5.1x, indexer 10–11.6x, compressed attention 1.25–3.0x; sparse_mla topk2048 và topk r4 thua. |
| B15 | Thắng cả 24 case; HC 3.6x, AttnRes 20.9x (cẩn thận byte-count). |
| B16 | Megakernel chậm hơn modular 6.6–22.9x — chưa dùng được. |
| B17 | Chậm hơn attention.rs ở store/prefill/pack; decode turbo4 gần hòa. |


## Phụ lục A — Bảng đầy đủ 443 case B0–B10

Cột `step` là nhãn bước theo ROADMAP; `—` ở cột baseline nghĩa là không có
PyTorch/SDPA cho case đó. Tất cả bảng dưới đây lấy từ JSON trong
`out/rtx5060/kops/` và `out/rtx5060/torch/` (merge dùng thêm
`thirdparty/sdpa-merge.json`).

### A.1 model_ops

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `model_ops.cast.bf16_to_fp32.t1024.h4096` | B0 | 0.0155 | 0.1071 | 6.91x | 1624.0 | 0.000 |
| `model_ops.cast.bf16_to_fp32.t4096.h8192` | B0 | 0.5394 | 1.2511 | 2.32x | 373.3 | 0.000 |
| `model_ops.cast.fp16_to_fp32.t1024.h4096` | B0 | 0.0155 | 0.1071 | 6.93x | 1628.2 | 0.000 |
| `model_ops.cast.fp32_to_bf16.t1024.h4096` | B0 | 0.0096 | 0.0326 | 3.39x | 2616.2 | 0.000 |
| `model_ops.residual_add.bf16.2.t1024.h4096` | B0 | 0.0097 | 0.0181 | 1.86x | 2588.7 | 0.863 |
| `model_ops.residual_add.bf16.2.t4096.h8192` | B0 | 0.5096 | 0.5108 | 1.00x | 395.1 | 0.132 |
| `model_ops.residual_add.bf16.3.t1024.h4096` | B0 | 0.0128 | 0.0764 | 5.95x | 2615.6 | 0.981 |
| `model_ops.residual_add.fp32.2.t1024.h4096` | B0 | 0.1040 | 0.1058 | 1.02x | 484.1 | 0.081 |
| `model_ops.residual_add.fp32.2.t4096.h8192` | B0 | 1.0168 | 1.0213 | 1.00x | 396.0 | 0.066 |
| `model_ops.residual_add.fp32.3.t1024.h4096` | B0 | 0.1479 | 0.2206 | 1.49x | 453.8 | 0.085 |


### A.2 norm

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `norm.fused_add_layer_norm.bf16.h4096.b1024` | B1 | 0.0316 | 0.0390 | 1.24x | 1063.2 | 1.063 |
| `norm.fused_add_rms_norm.bf16.h4096.b1024` | B1 | 0.0263 | 0.0330 | 1.25x | 1275.6 | 1.116 |
| `norm.fused_add_rms_norm.fp32.h4096.b1024` | B1 | 0.0822 | 0.1465 | 1.78x | 816.7 | 0.357 |
| `norm.fused_add_rms_norm_out.bf16.h4096.b1024` | B1 | 0.0239 | 0.2287 | 9.57x | 1405.1 | 1.229 |
| `norm.gated_rms_norm.bf16.g128.h4096.b1024` | B1 | 0.0535 | 0.4076 | 7.62x | 470.6 | 0.470 |
| `norm.l2_norm.bf16.h4096.b1024` | B1 | 0.0141 | 0.1446 | 10.28x | 1192.0 | 0.894 |
| `norm.layer_norm.bf16.h4096.b1024` | B1 | 0.0296 | 0.0323 | 1.09x | 567.5 | 0.851 |
| `norm.layer_norm.fp32.h4096.b1024` | B1 | 0.0423 | 0.0405 | 0.96x | 794.3 | 0.596 |
| `norm.norm_quant.bf16.h4096.b1024` | B3 | 0.0431 | 0.4236 | 9.83x | 584.1 | 0.584 |
| `norm.qk_norm.bf16.h32kv8.d128.b1024` | B1 | 0.0312 | 0.3234 | 10.38x | 1211.5 | 0.673 |
| `norm.rms_norm.bf16.h1024.b1024` | B1 | 0.0054 | 0.0236 | 4.39x | 781.5 | 0.976 |
| `norm.rms_norm.bf16.h4096.b1024` | B1 | 0.0199 | 0.0251 | 1.26x | 843.2 | 1.054 |
| `norm.rms_norm.bf16.h8192.b4096` | B1 | 0.3448 | 0.3565 | 1.03x | 389.3 | 0.487 |
| `norm.rms_norm.fp32.h1024.b1024` | B1 | 0.0082 | 0.0235 | 2.87x | 1026.5 | 0.641 |
| `norm.rms_norm.fp32.h4096.b1024` | B1 | 0.0255 | 0.0373 | 1.46x | 1315.2 | 0.822 |
| `norm.rms_norm.fp32.h8192.b4096` | B1 | 0.6939 | 0.8701 | 1.25x | 386.9 | 0.242 |


### A.3 activation

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `act_and_mul.silu.bf16.t2048.i4096` | B1 | 0.1007 | 0.2373 | 2.36x | 500.0 | 1.167 |
| `act_and_mul.silu.fp32.t2048.i4096` | B1 | 0.2557 | 0.4392 | 1.72x | 393.7 | 0.459 |
| `act_and_mul_quant.silu.bf16.t2048.i4096` | B3 | 0.1140 | 0.9276 | 8.13x | 294.3 | 1.103 |
| `activation.silu.bf16.n4m` | B1 | 0.0208 | 0.0190 | 0.91x | 768.0 | 1.920 |
| `activation.silu.fp32.n4m` | B1 | 0.0223 | 0.0250 | 1.12x | 1434.5 | 1.793 |
| `gate_mul.sigmoid.bf16.t2048.d4096.w1` | B1 | 0.0497 | 0.1369 | 2.75x | 674.8 | 0.506 |
| `gate_mul.sigmoid.bf16.t2048.d4096.w32.acc` | B1 | 0.1011 | 0.2737 | 2.71x | 499.1 | 0.249 |
| `gate_mul.sigmoid.bf16.t2048.d4096.w4096` | B1 | 0.1007 | 0.1780 | 1.77x | 500.0 | 0.250 |
| `gate_mul.sigmoid.fp32.t2048.d4096.w4096` | B1 | 0.2555 | 0.4334 | 1.70x | 394.0 | 0.099 |


### A.4 position

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `position.apply_mrope.bf16.t1024` | B1 | 0.0256 | 0.3711 | 14.51x | 830.2 | 0.492 |
| `position.apply_mrope.fp32.t1024` | B1 | 0.0252 | 0.5349 | 21.24x | 1686.4 | 0.500 |
| `position.apply_rotary.bf16.t1` | B1 | 0.0031 | 0.2794 | 89.94x | 6.7 | 0.004 |
| `position.apply_rotary.bf16.t16` | B1 | 0.0031 | 0.2763 | 90.11x | 108.2 | 0.064 |
| `position.apply_rotary.bf16.t4096` | B1 | 0.2295 | 1.2836 | 5.59x | 370.0 | 0.219 |
| `position.apply_rotary.bf16.t512` | B1 | 0.0110 | 0.2749 | 25.00x | 965.3 | 0.572 |
| `position.apply_rotary.fp32.t1` | B1 | 0.0030 | 0.2539 | 84.17x | 13.8 | 0.004 |
| `position.apply_rotary.fp32.t16` | B1 | 0.0031 | 0.2584 | 84.46x | 216.9 | 0.064 |
| `position.apply_rotary.fp32.t4096` | B1 | 0.4326 | 2.2929 | 5.30x | 392.7 | 0.116 |
| `position.apply_rotary.fp32.t512` | B1 | 0.0119 | 0.2848 | 24.01x | 1790.0 | 0.530 |
| `position.apply_rotary_on_the_fly.bf16.t512` | B1 | 0.0139 | 0.2743 | 19.71x | 762.8 | 0.452 |
| `position.apply_rotary_on_the_fly.fp32.t512` | B1 | 0.0154 | 0.2899 | 18.88x | 1383.0 | 0.410 |
| `position.attn_temperature_tuning.bf16.t1024` | B1 | 0.0205 | 0.0669 | 3.26x | 816.4 | 0.408 |
| `position.attn_temperature_tuning.fp32.t1024` | B1 | 0.0233 | 0.0460 | 1.97x | 1438.1 | 0.359 |
| `position.build_cos_sin_cache.bf16.p32768` | B1 | 0.3737 | 0.5054 | 1.35x | 22.4 | 0.022 |
| `position.build_cos_sin_cache.bf16.p4096` | B1 | 0.0500 | 0.0815 | 1.63x | 21.0 | 0.021 |
| `position.build_cos_sin_cache.fp32.p32768` | B1 | 0.3744 | 0.5495 | 1.47x | 44.8 | 0.022 |
| `position.build_cos_sin_cache.fp32.p4096` | B1 | 0.0500 | 0.0827 | 1.65x | 41.9 | 0.021 |
| `position.fused_qk_norm_rope.bf16.t2048` | B1 | 0.0635 | 1.3594 | 21.40x | 800.7 | 1.321 |
| `position.fused_qk_norm_rope.bf16.t64` | B1 | 0.0031 | 0.4454 | 145.46x | 519.1 | 0.856 |
| `position.fused_qk_norm_rope.fp32.t2048` | B1 | 0.2175 | 1.6089 | 7.40x | 467.7 | 0.386 |
| `position.fused_qk_norm_rope.fp32.t64` | B1 | 0.0032 | 0.3739 | 116.18x | 988.2 | 0.815 |


### A.5 embedding

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `embedding.gather_rows.bf16.n512` | B1 | 0.0051 | 0.0245 | 4.76x | 1632.8 | 0.000 |
| `embedding.logits_postprocess.bf16.r1` | B1 | 0.0030 | 0.0465 | 15.73x | 260.1 | 0.433 |
| `embedding.logits_postprocess.bf16.r32` | B1 | 0.0349 | 0.1459 | 4.18x | 705.4 | 1.176 |
| `embedding.lookup.bf16.t1` | B1 | 0.0031 | 0.0197 | 6.28x | 5.2 | 0.000 |
| `embedding.lookup.bf16.t2048` | B1 | 0.0203 | 0.0410 | 2.01x | 1650.0 | 0.000 |
| `embedding.lookup.fp16.t2048` | B1 | 0.0204 | 0.0426 | 2.08x | 1641.6 | 0.000 |
| `embedding.lookup.fp32.t2048` | B1 | 0.1727 | 0.1816 | 1.05x | 388.6 | 0.000 |
| `embedding.scatter_rows.bf16.n512` | B1 | 0.0051 | 0.0251 | 4.90x | 1642.0 | 0.000 |
| `embedding.with_position.bf16.t2048` | B1 | 0.1011 | 0.2826 | 2.79x | 497.6 | 0.083 |
| `embedding.with_position.fp16.t2048` | B1 | 0.1010 | 0.2775 | 2.75x | 498.1 | 0.083 |
| `embedding.with_position.fp32.t2048` | B1 | 0.2548 | 0.6039 | 2.37x | 395.1 | 0.033 |


### A.6 merge

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `concat_mla_q.bf16.t1024.h8.n512.p64` | B2 | 0.0117 | — | — | 1612.2 | 0.000 |
| `decode.dynamic.bf16.d128.skew1x32768+31x1024.h32x8` | B8 | 0.6382 | — | — | 414.1 | 0.000 |
| `decode.simt.bf16.d128.b8.kv4096.h32x32` | B8 | 1.5250 | — | — | 352.0 | 0.000 |
| `decode.simt.e4m3.d128.b8.kv4096.h32x32` | B8 | 1.0941 | — | — | 245.3 | 0.000 |
| `decode.tc.bf16.d128.b1.kv32768.h32x8` | B8 | 0.3290 | — | — | 408.0 | 0.000 |
| `decode.tc.bf16.d128.b32.kv2048.h32x8` | B8 | 0.6442 | — | — | 416.7 | 0.000 |
| `decode.tc.bf16.d128.b32.kv2048.h32x8.q4` | B8 | 0.6532 | — | — | 411.0 | 0.000 |
| `decode.tc.bf16.d128.skew1x32768+31x1024.h32x8` | B8 | 0.7166 | — | — | 368.7 | 0.000 |
| `decode.tc.e4m3.d128.b32.kv2048.h32x8` | B8 | 0.4359 | — | — | 307.9 | 0.000 |
| `merge_split_kv.fp32.b64.h32.d128.s8` | B1 | 0.0062 | 0.1583 | 25.38x | 1525.0 | 0.000 |
| `merge_state.fp32.b64.h32.d128` | B1 | 0.0031 | 0.0991 | 32.32x | 1034.2 | 0.513 |
| `merge_state.fp32.b8.h32.d128` | B1 | 0.0030 | 0.0974 | 32.12x | 130.7 | 0.065 |
| `merge_states.fp32.b64.h32.d128.n8` | B1 | 0.0072 | 0.1108 | 15.45x | 1326.9 | 0.000 |
| `mla_context_gather.bf16.t2048.r512.p64` | B2 | 0.0049 | — | — | 954.7 | 0.000 |
| `prefill.paged.bf16.d128.chunk512.kv4096.ps16.h32x8` | B8 | 1.0807 | 0.8541 | 0.79x | 0.0 | 29.811 |
| `prefill.paged.e4m3.d128.chunk512.kv4096.ps16.h32x8` | B8 | 1.3111 | — | — | 0.0 | 24.572 |
| `prefill.ragged.bf16.d128.causal.s2048.h32x8` | B8 | 0.9459 | 0.8868 | 0.94x | 0.0 | 36.342 |
| `prefill.ragged.bf16.d128.causal.s2048.h32x8.softcap_swa` | B8 | 1.0214 | — | — | 0.0 | 33.655 |
| `prefill.ragged.bf16.d128.full.b4.s1024.h32x8` | B8 | 1.8388 | 1.6124 | 0.88x | 0.0 | 37.372 |
| `prefill.ragged.bf16.d256.causal.s1024.h8x4` | B8 | 0.1886 | 0.1459 | 0.77x | 0.0 | 22.796 |
| `prefill.ragged.bf16.d64.causal.s2048.h64x8` | B8 | 1.1423 | 0.8048 | 0.70x | 0.0 | 30.095 |


### A.7 sampling

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `sampling.argmax.bf16.v131072.b256` | B1 | 0.2008 | 0.8493 | 4.23x | 334.2 | 0.000 |
| `sampling.argmax.fp32.v131072.b128` | B1 | 0.1719 | 0.1754 | 1.02x | 390.5 | 0.000 |
| `sampling.bias.bf16.v131072.b128.nnz8` | B1 | 0.0029 | 0.6287 | 219.90x | 4.5 | 0.000 |
| `sampling.bias.fp32.v32768.b256.nnz8` | B1 | 0.0029 | 0.1884 | 64.68x | 11.6 | 0.000 |
| `sampling.bincount.v131072.b32.t1024` | B1 | 0.0029 | 1.6365 | 567.83x | 272.9 | 0.000 |
| `sampling.bincount.v32768.b256.t64` | B1 | 0.0048 | 11.7833 | 2475.48x | 247.8 | 0.000 |
| `sampling.bitmask.bf16.v131072.b128` | B1 | 0.2061 | 1.1027 | 5.35x | 173.0 | 0.000 |
| `sampling.bitmask.fp32.v32768.b256` | B1 | 0.1142 | 0.4166 | 3.65x | 303.0 | 0.000 |
| `sampling.fused.greedy.bf16.v120832.b128` | B6 | 0.0238 | 0.0494 | 2.08x | 1298.7 | 0.000 |
| `sampling.fused.greedy.bf16.v120832.b32` | B6 | 0.0103 | 0.0272 | 2.63x | 748.2 | 0.000 |
| `sampling.fused.greedy.bf16.v120832.b512` | B6 | 0.2934 | 0.3054 | 1.04x | 421.8 | 0.000 |
| `sampling.fused.greedy.bf16.v151936.b1` | B6 | 0.0062 | 0.0239 | 3.83x | 48.7 | 0.000 |
| `sampling.fused.greedy.bf16.v151936.b128` | B6 | 0.0939 | 0.1150 | 1.22x | 414.0 | 0.000 |
| `sampling.fused.greedy.bf16.v151936.b16` | B6 | 0.0083 | 0.0261 | 3.15x | 587.0 | 0.000 |
| `sampling.fused.greedy.bf16.v151936.b256` | B6 | 0.1962 | 0.2051 | 1.05x | 396.5 | 0.000 |
| `sampling.fused.greedy.bf16.v151936.b32` | B6 | 0.0124 | 0.0303 | 2.45x | 785.8 | 0.000 |
| `sampling.fused.greedy.bf16.v151936.b512` | B6 | 0.3726 | 0.3816 | 1.02x | 417.6 | 0.000 |
| `sampling.fused.greedy.bf16.v151936.b64` | B6 | 0.0186 | 0.0424 | 2.28x | 1046.2 | 0.000 |
| `sampling.fused.k50.bf16.v120832.b128` | B6 | 0.1592 | 3.1695 | 19.91x | 194.3 | 0.000 |
| `sampling.fused.k50.bf16.v120832.b32` | B6 | 0.0801 | 0.4320 | 5.40x | 96.6 | 0.000 |
| `sampling.fused.k50.bf16.v120832.b512` | B6 | 0.6611 | 13.0371 | 19.72x | 187.2 | 0.000 |
| `sampling.fused.k50.bf16.v151936.b1` | B6 | 0.0513 | 0.2902 | 5.66x | 5.9 | 0.000 |
| `sampling.fused.k50.bf16.v151936.b128` | B6 | 0.2231 | 4.0731 | 18.26x | 174.3 | 0.000 |
| `sampling.fused.k50.bf16.v151936.b16` | B6 | 0.0699 | 0.3129 | 4.48x | 69.6 | 0.000 |
| `sampling.fused.k50.bf16.v151936.b256` | B6 | 0.4526 | 9.0474 | 19.99x | 171.9 | 0.000 |
| `sampling.fused.k50.bf16.v151936.b32` | B6 | 0.0902 | 0.5872 | 6.51x | 107.8 | 0.000 |
| `sampling.fused.k50.bf16.v151936.b512` | B6 | 0.8582 | 16.7446 | 19.51x | 181.3 | 0.000 |
| `sampling.fused.k50.bf16.v151936.b64` | B6 | 0.1421 | 1.8402 | 12.95x | 136.9 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v120832.b128` | B6 | 0.1928 | 3.1776 | 16.48x | 160.4 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v120832.b32` | B6 | 0.0801 | 0.4327 | 5.41x | 96.6 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v120832.b512` | B6 | 0.7935 | 13.0375 | 16.43x | 155.9 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v151936.b1` | B6 | 0.0514 | 0.2946 | 5.73x | 5.9 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v151936.b128` | B6 | 0.2646 | 4.0730 | 15.39x | 147.0 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v151936.b16` | B6 | 0.0700 | 0.3082 | 4.41x | 69.5 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v151936.b256` | B6 | 0.5287 | 8.3979 | 15.88x | 147.1 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v151936.b32` | B6 | 0.0904 | 0.5900 | 6.53x | 107.6 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v151936.b512` | B6 | 0.9874 | 16.7518 | 16.97x | 157.6 | 0.000 |
| `sampling.fused.k50p0.9.bf16.v151936.b64` | B6 | 0.1419 | 1.8614 | 13.12x | 137.1 | 0.000 |
| `sampling.fused.minp0.05.bf16.v151936.b1` | B6 | 0.0349 | 0.2218 | 6.35x | 8.7 | 0.000 |
| `sampling.fused.minp0.05.bf16.v151936.b128` | B6 | 0.1974 | 3.6940 | 18.72x | 197.1 | 0.000 |
| `sampling.fused.minp0.05.bf16.v151936.b16` | B6 | 0.0514 | 0.2352 | 4.58x | 94.6 | 0.000 |
| `sampling.fused.minp0.05.bf16.v151936.b256` | B6 | 0.4080 | 7.5083 | 18.40x | 190.7 | 0.000 |
| `sampling.fused.minp0.05.bf16.v151936.b32` | B6 | 0.0724 | 0.4686 | 6.47x | 134.3 | 0.000 |
| `sampling.fused.minp0.05.bf16.v151936.b512` | B6 | 0.7841 | 15.9066 | 20.29x | 198.4 | 0.000 |
| `sampling.fused.minp0.05.bf16.v151936.b64` | B6 | 0.1223 | 1.5520 | 12.70x | 159.1 | 0.000 |
| `sampling.fused.p0.5.bf16.v151936.b1` | B6 | 0.0493 | 0.1778 | 3.61x | 6.2 | 0.000 |
| `sampling.fused.p0.5.bf16.v151936.b128` | B6 | 0.3894 | 2.7555 | 7.08x | 99.9 | 0.000 |
| `sampling.fused.p0.5.bf16.v151936.b16` | B6 | 0.1102 | 0.1797 | 1.63x | 44.1 | 0.000 |
| `sampling.fused.p0.5.bf16.v151936.b256` | B6 | 0.7567 | 5.7948 | 7.66x | 102.8 | 0.000 |
| `sampling.fused.p0.5.bf16.v151936.b32` | B6 | 0.1442 | 0.3588 | 2.49x | 67.4 | 0.000 |
| `sampling.fused.p0.5.bf16.v151936.b512` | B6 | 1.4187 | 11.6002 | 8.18x | 109.7 | 0.000 |
| `sampling.fused.p0.5.bf16.v151936.b64` | B6 | 0.2207 | 1.2043 | 5.46x | 88.1 | 0.000 |
| `sampling.fused.p0.95.bf16.v120832.b128` | B6 | 0.4121 | 2.1051 | 5.11x | 75.1 | 0.000 |
| `sampling.fused.p0.95.bf16.v120832.b32` | B6 | 0.1347 | 0.2329 | 1.73x | 57.4 | 0.000 |
| `sampling.fused.p0.95.bf16.v120832.b512` | B6 | 1.6053 | 8.9815 | 5.59x | 77.1 | 0.000 |
| `sampling.fused.p0.95.bf16.v151936.b1` | B6 | 0.0622 | 0.1775 | 2.86x | 4.9 | 0.000 |
| `sampling.fused.p0.95.bf16.v151936.b128` | B6 | 0.5591 | 2.7534 | 4.92x | 69.6 | 0.000 |
| `sampling.fused.p0.95.bf16.v151936.b16` | B6 | 0.1131 | 0.1684 | 1.49x | 43.0 | 0.000 |
| `sampling.fused.p0.95.bf16.v151936.b256` | B6 | 1.1088 | 6.1024 | 5.50x | 70.2 | 0.000 |
| `sampling.fused.p0.95.bf16.v151936.b32` | B6 | 0.1698 | 0.3564 | 2.10x | 57.2 | 0.000 |
| `sampling.fused.p0.95.bf16.v151936.b512` | B6 | 2.0527 | 12.3826 | 6.03x | 75.8 | 0.000 |
| `sampling.fused.p0.95.bf16.v151936.b64` | B6 | 0.2932 | 1.1880 | 4.05x | 66.3 | 0.000 |
| `sampling.fused.p0.99.bf16.v151936.b1` | B6 | 0.0862 | 0.1763 | 2.04x | 3.5 | 0.000 |
| `sampling.fused.p0.99.bf16.v151936.b128` | B6 | 0.6728 | 2.7568 | 4.10x | 57.8 | 0.000 |
| `sampling.fused.p0.99.bf16.v151936.b16` | B6 | 0.1532 | 0.1771 | 1.16x | 31.7 | 0.000 |
| `sampling.fused.p0.99.bf16.v151936.b256` | B6 | 1.3033 | 6.0984 | 4.68x | 59.7 | 0.000 |
| `sampling.fused.p0.99.bf16.v151936.b32` | B6 | 0.2181 | 0.3553 | 1.63x | 44.6 | 0.000 |
| `sampling.fused.p0.99.bf16.v151936.b512` | B6 | 2.3747 | 12.3471 | 5.20x | 65.5 | 0.000 |
| `sampling.fused.p0.99.bf16.v151936.b64` | B6 | 0.3717 | 1.1924 | 3.21x | 52.3 | 0.000 |
| `sampling.fused.t1.bf16.v151936.b1` | B6 | 0.0145 | 0.1520 | 10.46x | 20.9 | 0.000 |
| `sampling.fused.t1.bf16.v151936.b128` | B6 | 0.2296 | 2.2223 | 9.68x | 169.4 | 0.000 |
| `sampling.fused.t1.bf16.v151936.b16` | B6 | 0.0309 | 0.1553 | 5.02x | 157.3 | 0.000 |
| `sampling.fused.t1.bf16.v151936.b256` | B6 | 0.4559 | 4.8158 | 10.56x | 170.7 | 0.000 |
| `sampling.fused.t1.bf16.v151936.b32` | B6 | 0.0514 | 0.2872 | 5.59x | 189.2 | 0.000 |
| `sampling.fused.t1.bf16.v151936.b512` | B6 | 0.8262 | 9.8737 | 11.95x | 188.3 | 0.000 |
| `sampling.fused.t1.bf16.v151936.b64` | B6 | 0.0935 | 0.9718 | 10.40x | 208.1 | 0.000 |
| `sampling.fused.t1.fp32.v151936.b128` | B6 | 0.3863 | 1.8843 | 4.88x | 201.4 | 0.000 |
| `sampling.gumbel.bf16.v131072.b32.k50` | B6 | 0.1801 | 0.1997 | 1.11x | 46.6 | 0.000 |
| `sampling.logprobs.bf16.v131072.b64.top5` | B1 | 0.1683 | 0.6914 | 4.11x | 99.7 | 0.000 |
| `sampling.logprobs.fp32.v32768.b128.top20` | B1 | 0.1795 | 0.2291 | 1.28x | 93.6 | 0.000 |
| `sampling.penalties.bf16.v131072.b128` | B1 | 0.3520 | 4.5596 | 12.95x | 387.3 | 0.000 |
| `sampling.penalties.fp32.v32768.b256` | B1 | 0.2537 | 2.1563 | 8.50x | 400.9 | 0.000 |
| `sampling.sample.greedy.bf16.v151936.b1` | B6 | 0.0064 | 0.0239 | 3.73x | 47.5 | 0.000 |
| `sampling.sample.greedy.bf16.v151936.b128` | B6 | 0.0940 | 0.1150 | 1.22x | 413.7 | 0.000 |
| `sampling.sample.greedy.bf16.v151936.b32` | B6 | 0.0125 | 0.0310 | 2.47x | 775.5 | 0.000 |
| `sampling.sample.greedy.bf16.v151936.b512` | B6 | 0.3727 | 0.3814 | 1.02x | 417.5 | 0.000 |
| `sampling.sample.k50.bf16.v151936.b1` | B6 | 0.1381 | 0.2882 | 2.09x | 2.2 | 0.000 |
| `sampling.sample.k50.bf16.v151936.b128` | B6 | 1.1363 | 4.0733 | 3.58x | 34.2 | 0.000 |
| `sampling.sample.k50.bf16.v151936.b32` | B6 | 0.1882 | 0.5873 | 3.12x | 51.7 | 0.000 |
| `sampling.sample.k50.bf16.v151936.b512` | B6 | 3.9213 | 18.0551 | 4.60x | 39.7 | 0.000 |
| `sampling.sample.k50p0.9.bf16.v151936.b1` | B6 | 0.2151 | 0.2964 | 1.38x | 1.4 | 0.000 |
| `sampling.sample.k50p0.9.bf16.v151936.b128` | B6 | 1.1303 | 4.0745 | 3.60x | 34.4 | 0.000 |
| `sampling.sample.k50p0.9.bf16.v151936.b32` | B6 | 0.2598 | 0.5891 | 2.27x | 37.4 | 0.000 |
| `sampling.sample.k50p0.9.bf16.v151936.b512` | B6 | 4.2407 | 16.7372 | 3.95x | 36.7 | 0.000 |
| `sampling.sample.minp0.05.bf16.v151936.b1` | B6 | 0.0352 | 0.2225 | 6.33x | 8.6 | 0.000 |
| `sampling.sample.minp0.05.bf16.v151936.b128` | B6 | 0.1977 | 3.5814 | 18.12x | 196.8 | 0.000 |
| `sampling.sample.minp0.05.bf16.v151936.b32` | B6 | 0.0739 | 0.4694 | 6.35x | 131.6 | 0.000 |
| `sampling.sample.minp0.05.bf16.v151936.b512` | B6 | 0.7841 | 16.2091 | 20.67x | 198.4 | 0.000 |
| `sampling.sample.p0.5.bf16.v151936.b1` | B6 | 0.0495 | 0.1798 | 3.63x | 6.1 | 0.000 |
| `sampling.sample.p0.5.bf16.v151936.b128` | B6 | 0.3893 | 2.7521 | 7.07x | 99.9 | 0.000 |
| `sampling.sample.p0.5.bf16.v151936.b32` | B6 | 0.1450 | 0.3551 | 2.45x | 67.0 | 0.000 |
| `sampling.sample.p0.5.bf16.v151936.b512` | B6 | 1.4192 | 11.5918 | 8.17x | 109.6 | 0.000 |
| `sampling.sample.p0.95.bf16.v151936.b1` | B6 | 0.0639 | 0.1798 | 2.81x | 4.8 | 0.000 |
| `sampling.sample.p0.95.bf16.v151936.b128` | B6 | 0.5551 | 2.7545 | 4.96x | 70.1 | 0.000 |
| `sampling.sample.p0.95.bf16.v151936.b32` | B6 | 0.1704 | 0.3550 | 2.08x | 57.1 | 0.000 |
| `sampling.sample.p0.95.bf16.v151936.b512` | B6 | 2.0511 | 11.6036 | 5.66x | 75.9 | 0.000 |
| `sampling.sample.p0.99.bf16.v151936.b1` | B6 | 0.0865 | 0.1769 | 2.04x | 3.5 | 0.000 |
| `sampling.sample.p0.99.bf16.v151936.b128` | B6 | 0.6729 | 2.7624 | 4.10x | 57.8 | 0.000 |
| `sampling.sample.p0.99.bf16.v151936.b32` | B6 | 0.2190 | 0.3539 | 1.62x | 44.4 | 0.000 |
| `sampling.sample.p0.99.bf16.v151936.b512` | B6 | 2.3734 | 12.2560 | 5.16x | 65.6 | 0.000 |
| `sampling.sample.t1.bf16.v151936.b1` | B6 | 0.0150 | 0.1515 | 10.10x | 20.3 | 0.000 |
| `sampling.sample.t1.bf16.v151936.b128` | B6 | 0.2290 | 2.2208 | 9.70x | 169.8 | 0.000 |
| `sampling.sample.t1.bf16.v151936.b32` | B6 | 0.0515 | 0.2873 | 5.58x | 188.8 | 0.000 |
| `sampling.sample.t1.bf16.v151936.b512` | B6 | 0.8252 | 9.3138 | 11.29x | 188.5 | 0.000 |
| `sampling.sample.t1.fp32.v151936.b128` | B6 | 0.3863 | 1.8835 | 4.88x | 201.4 | 0.000 |
| `sampling.softmax.bf16.v131072.b2` | B1 | 0.0154 | 0.0583 | 3.79x | 102.1 | 0.000 |
| `sampling.softmax.bf16.v131072.b32` | B1 | 0.0452 | 0.1325 | 2.93x | 556.3 | 0.000 |
| `sampling.softmax.bf16.v262144.b1` | B1 | 0.0113 | 0.0825 | 7.33x | 139.7 | 0.000 |
| `sampling.softmax.bf16.v32768.b256` | B1 | 0.1284 | 0.4844 | 3.77x | 391.9 | 0.000 |
| `sampling.softmax.fp16.v131072.b32` | B1 | 0.0455 | 0.1316 | 2.89x | 553.2 | 0.000 |
| `sampling.softmax.fp32.v131072.b32` | B1 | 0.0428 | 0.1329 | 3.11x | 784.4 | 0.000 |
| `sampling.softmax.fp32.v262144.b1` | B1 | 0.0121 | 0.0726 | 5.99x | 172.9 | 0.000 |
| `sampling.topk_mask.bf16.v131072.b128.k50` | B6 | 0.4338 | 1.1529 | 2.66x | 154.7 | 0.000 |
| `sampling.topk_renorm.fp32.v131072.b128.k50` | B6 | 0.6463 | 1.5989 | 2.47x | 207.7 | 0.000 |
| `sampling.topk_topp.bf16.v131072.b32.k50.p0.9` | B6 | 0.2303 | 2.7371 | 11.89x | 36.4 | 0.000 |
| `sampling.topk_topp.bf16.v32768.b256.k20.p0.95` | B6 | 0.3044 | 6.4454 | 21.17x | 55.1 | 0.000 |
| `sampling.topp_renorm.fp32.v131072.b128.p0.9` | B6 | 1.8452 | 11.6209 | 6.30x | 72.7 | 0.000 |
| `spec.mask.p2048.n64.b64` | B10 | 0.0654 | 3.5892 | 54.91x | 17.3 | 0.000 |
| `spec.reject.bonus.k0.v120832.b128` | B10 | 0.1607 | 1.0681 | 6.65x | 96.2 | 0.000 |
| `spec.reject.chain.k8.v151936.b32` | B10 | 0.0832 | 0.9224 | 11.09x | 233.8 | 0.000 |
| `spec.reject.chain.k8.v32768.b128` | B10 | 0.0387 | 0.8408 | 21.72x | 433.7 | 0.000 |
| `spec.reject.greedy.k8.v151936.b32` | B10 | 0.0035 | 0.1895 | 54.66x | 0.7 | 0.000 |
| `spec.reject.target_only.k8.v151936.b32` | B10 | 0.0236 | 0.6035 | 25.57x | 824.0 | 0.000 |
| `spec.tree.greedy.n64.d8.v32000.b256` | B10 | 0.0110 | 1.6045 | 145.80x | 0.6 | 0.000 |
| `spec.tree.q.n64.d8.v32000.b8` | B10 | 0.0144 | 1.7838 | 123.78x | 71.3 | 0.000 |
| `spec.tree.target_only.n64.d8.v32000.b8` | B10 | 0.0073 | 5.8636 | 803.89x | 140.9 | 0.000 |


### A.8 chains

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `chain.embedding_norm.bf16.t2048.h4096` | B1 | 0.1667 | 0.9834 | 5.90x | 805.2 | 0.000 |
| `chain.glu_quant.bf16.t2048.i4096` | B1+B3 | 0.1139 | 0.9256 | 8.13x | 589.2 | 0.442 |
| `chain.logits_pipeline.bf16.b32.v131072` | B1 | 0.2452 | 1.2088 | 4.93x | 410.5 | 0.000 |
| `chain.pre_norm_glu.bf16.t2048.i4096` | B1 | 0.2994 | 1.1110 | 3.71x | 504.4 | 0.000 |
| `chain.pre_norm_quant.bf16.b4096.h4096` | B1+B3 | 0.2995 | 3.4791 | 11.62x | 336.1 | 0.280 |
| `chain.pre_norm_residual.bf16.b1024.h4096` | B0+B1 | 0.0263 | 0.2251 | 8.56x | 1913.6 | 0.000 |
| `chain.qk_norm_rope.bf16.t2048.hq32.hkv8.d128` | B1 | 0.1009 | 1.3640 | 13.53x | 1039.7 | 0.000 |
| `chain.ref_glu_quant.bf16.t2048.i4096` | B1+B3 | 0.1492 | — | — | 674.6 | 0.394 |
| `chain.ref_pre_norm_quant.bf16.b4096.h4096` | B1+B3 | 0.4709 | — | — | 320.7 | 0.214 |
| `chain.splitkv_merge.fp32.b64.h32.d128.s8` | B1+B8 | 0.0062 | 0.1625 | 26.05x | 1345.0 | 0.000 |


### A.9 kvcache

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `append.nhd.bf16.t2048.h4.d128` | B2 | 0.0056 | — | — | 1490.7 | 0.000 |
| `concat_mla.t2048.r512.p64` | B2 | 0.0157 | — | — | 301.2 | 0.000 |
| `concat_mla_grouped.l2.t1024.r512.p64` | B2 | 0.0173 | — | — | 272.9 | 0.000 |
| `convert.bf16_to_f32.b256.s4096` | B2 | 0.0050 | — | — | 1246.7 | 0.000 |
| `copy_pages.bytes.p512.b4096` | B2 | 0.0112 | — | — | 1503.5 | 0.000 |
| `cp_gather.bytes.t4096.e576` | B2 | 0.0053 | — | — | 895.6 | 0.000 |
| `ds_mla.fp8.t2048.r512.p64` | B2 | 0.0085 | — | — | 404.9 | 0.000 |
| `ds_mla_v40.fp8.t256.r448.p64` | B2 | 0.0032 | — | — | 127.8 | 0.000 |
| `ds_mla_v40_gather.bf16.t256` | B2 | 0.0042 | — | — | 104.7 | 0.000 |
| `ds_mla_v41.fp4.t1024.r448.p64` | B2 | 0.0080 | — | — | 168.3 | 0.000 |
| `ds_mla_v41.fp8.t1024.r448.p64` | B2 | 0.0122 | — | — | 130.0 | 0.000 |
| `ds_mla_v41_fp4_gather.bf16.t1024` | B2 | 0.0032 | — | — | 425.4 | 0.000 |
| `ds_mla_v41_gather.bf16.t1024` | B2 | 0.0032 | — | — | 493.4 | 0.000 |
| `fused_kv_append.bf16.t2048.hkv4.d128` | B2 | 0.0176 | — | — | 477.8 | 0.000 |
| `fused_qkv_append.bf16.t2048.hq32.hkv8.d128` | B2 | 0.1135 | — | — | 665.0 | 0.000 |
| `gather_dequant.t4096.e576` | B2 | 0.0103 | — | — | 917.3 | 0.000 |
| `gather_pages.nhd.bf16.t4096.h4.d128` | B2 | 0.0109 | — | — | 769.5 | 0.000 |
| `indexer_fp8.t2048.d128` | B2 | 0.0053 | — | — | 152.5 | 0.000 |
| `kv_scale_update.bf16.t2048.h8.d128` | B3 | 0.0155 | — | — | 542.9 | 0.000 |
| `mla_rope_fused.bf16.t2048.hq16.r512.p64` | B2 | 0.0272 | — | — | 481.4 | 0.000 |
| `relayout.nhd_to_hnd.p512` | B2 | 0.0494 | — | — | 339.3 | 0.000 |
| `requantize.int8.t2048.h8.d128` | B3 | 0.0435 | — | — | 193.0 | 0.000 |
| `reshape.legacy.bf16.t2048.h4.d128` | B2 | 0.0904 | — | — | 92.8 | 0.000 |
| `reshape.nhd.bf16.t2048.h4.d128` | B2 | 0.0074 | — | — | 1140.2 | 0.000 |
| `reshape_per_token_head.fp8.t2048.h8.d128` | B3 | 0.0440 | — | — | 190.8 | 0.000 |
| `ring.bf16.t1024.h4.d128` | B2 | 0.0193 | — | — | 108.4 | 0.000 |
| `scatter_pages.nhd.bf16.t4096.h4.d128` | B2 | 0.0113 | — | — | 741.6 | 0.000 |
| `zero_pages.bytes.p512.b4096` | B2 | 0.0080 | — | — | 1044.2 | 0.000 |


### A.10 state

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `conv1d_fwd.bf16.qwen3next.T4096` | B9 | 0.3756 | — | — | 357.3 | 0.715 |
| `conv1d_update.bf16.qwen3next.B64` | B9 | 0.0625 | — | — | 134.2 | 0.067 |
| `delta_decode.gdn.bf16.B64` | B9 | 1.1932 | — | — | 225.0 | 0.169 |
| `delta_decode.kda.bf16.B64` | B9 | 0.7082 | — | — | 379.0 | 0.284 |
| `delta_gating.gdn.bf16.T4096` | B9 | 0.0030 | — | — | 529.7 | 0.000 |
| `delta_gating.kda.bf16.T4096` | B9 | 0.3807 | — | — | 266.4 | 0.000 |
| `delta_prefill.gdn.fp32math.bf16.T4096` | B9 | 9.2888 | — | — | 14.4 | 2.543 |
| `delta_prefill.gdn.tc.bf16.T4096` | B9 | 2.8500 | — | — | 47.1 | 8.289 |
| `delta_prefill.kda.fp32math.bf16.T4096` | B9 | 21.0130 | — | — | 6.4 | 1.124 |
| `delta_prefill.kda.tc.bf16.T4096` | B9 | 4.3909 | — | — | 30.6 | 5.380 |
| `fused_delta_decode.gdn.bf16.B64` | B9 | 0.7278 | — | — | 368.8 | 0.277 |
| `fused_delta_decode.kda.bf16.B64` | B9 | 0.7342 | — | — | 365.6 | 0.274 |
| `lightning_prefill.bf16.minimax.H64D128.T4096` | B9 | 6.7689 | — | — | 39.7 | 3.807 |
| `linear_decode.bf16.minimax.H64D128.B64` | B9 | 1.4441 | — | — | 371.8 | 0.186 |
| `selective_scan.bf16.falcon_mamba.D8192N16.T4096` | B9 | 3.8101 | — | — | 70.5 | 1.127 |
| `ssd_chunked_scan.bf16.mamba2.H32P64N128.T4096` | B9 | 2.0700 | — | — | 32.4 | 5.187 |
| `ssm_update.bf16.mamba2.H32P64N128.B64` | B9 | 0.3533 | — | — | 379.9 | 0.285 |
| `state_commit.2pools.b64.d8` | B10 | 0.0028 | 2.7355 | 972.46x | 279.6 | 0.000 |
| `state_copy.2pools.64items` | B2 | 0.0028 | — | — | 279.4 | 0.000 |
| `state_gather.2pools.64items` | B2 | 0.0028 | — | — | 276.9 | 0.000 |
| `state_scatter.2pools.64items` | B2 | 0.0029 | — | — | 274.4 | 0.000 |
| `state_zero.2pools.64items` | B2 | 0.0028 | — | — | 141.3 | 0.000 |


### A.11 quantization

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `quant.absmax.bf16.4096x4096` | B3 | 0.0225 | 0.2299 | 10.22x | 1491.6 | 0.746 |
| `quant.absmax.fp32.4096x4096` | B3 | 0.1595 | 0.4810 | 3.02x | 420.7 | 0.105 |
| `quant.awq_dequant.fp16.g128.4096x4096` | B4 | 0.1690 | 2.0916 | 12.38x | 250.6 | 0.099 |
| `quant.awq_marlin.4bit.g128.4096x4096` | B4 | 0.0260 | 3.4616 | 133.39x | 686.9 | 0.000 |
| `quant.dequant_per_token.fp8.fp32.4096x4096` | B3 | 0.2282 | 0.5873 | 2.57x | 367.5 | 0.073 |
| `quant.fp4_marlin.mxfp4.4096x4096` | B4 | 0.0366 | 2.9591 | 80.76x | 486.5 | 0.000 |
| `quant.fp8_to_ue8m0.4096x4096` | B4 | 0.0634 | 0.8681 | 13.69x | 529.2 | 0.265 |
| `quant.gguf_q4_k.fp16.4096x4096` | B4 | 0.2006 | 1.6193 | 8.07x | 214.4 | 0.084 |
| `quant.gguf_q8_0.fp16.4096x4096` | B4 | 0.1938 | 0.9894 | 5.11x | 265.1 | 0.087 |
| `quant.gptq_marlin.4bit.g128.4096x4096` | B4 | 0.0123 | 2.8305 | 229.64x | 1403.6 | 0.000 |
| `quant.gptq_marlin.4bit.g128.actorder.4096x4096` | B4 | 0.0520 | 3.1761 | 61.07x | 333.6 | 0.000 |
| `quant.hadamard.fp32.4096x4096` | B3 | 0.3481 | 10.0980 | 29.01x | 96.4 | 0.578 |
| `quant.mxfp4.bf16.4096x4096` | B3 | 0.3079 | 1.5242 | 4.95x | 81.7 | 0.164 |
| `quant.mxfp8.bf16.4096x4096` | B3 | 0.2937 | 1.5233 | 5.19x | 171.4 | 0.171 |
| `quant.nvfp4.bf16.4096x4096` | B3 | 0.5749 | 3.3698 | 5.86x | 43.8 | 0.087 |
| `quant.pack_int4.4096x4096` | B3 | 0.1145 | 0.5971 | 5.22x | 219.8 | 0.147 |
| `quant.per_group128.bf16.fp8.4096x4096` | B3 | 0.1408 | 1.2729 | 9.04x | 357.4 | 0.357 |
| `quant.per_token.bf16.fp8.4096x4096` | B3 | 0.1009 | 1.5437 | 15.29x | 498.6 | 0.499 |
| `quant.per_token.fp32.int8.4096x4096` | B3 | 0.1786 | 1.9304 | 10.81x | 469.7 | 0.282 |
| `quant.smoothquant_fold.bf16.4096x4096` | B4 | 0.0842 | 0.2173 | 2.58x | 796.7 | 0.199 |
| `quant.static.bf16.fp8.4096x4096` | B3 | 0.0991 | 1.1673 | 11.78x | 507.8 | 0.169 |
| `quant.unpack_int4.4096x4096` | B3 | 0.0289 | 0.2459 | 8.52x | 871.7 | 0.581 |


### A.12 gemm

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `gemm.argmax.bf16.m1.32000x4096` | B5 | 0.6186 | 0.6262 | 1.01x | 423.8 | 0.424 |
| `gemm.bmm_mla_uk.m1` | B7 | 0.0176 | 0.0468 | 2.67x | 964.7 | 0.955 |
| `gemm.bmm_mla_uk.m512` | B7 | 0.2831 | 0.5647 | 1.99x | 355.6 | 30.344 |
| `gemm.bmm_mla_uk.m64` | B7 | 0.0339 | 0.0847 | 2.50x | 803.4 | 31.640 |
| `gemm.bmm_mla_uk.m8` | B7 | 0.0174 | 0.0520 | 2.98x | 1037.7 | 7.700 |
| `gemm.bmm_mla_uv.m1` | B7 | 0.0174 | 0.0466 | 2.68x | 975.2 | 0.966 |
| `gemm.bmm_mla_uv.m512` | B7 | 0.2939 | 0.4909 | 1.67x | 342.5 | 29.228 |
| `gemm.bmm_mla_uv.m64` | B7 | 0.0320 | 0.0785 | 2.45x | 851.0 | 33.518 |
| `gemm.bmm_mla_uv.m8` | B7 | 0.0173 | 0.0491 | 2.83x | 1042.8 | 7.738 |
| `gemm.dense_gemv.bf16.m1.14336x4096` | B5 | 0.2771 | 0.2949 | 1.06x | 423.9 | 0.424 |
| `gemm.dense_gemv.bf16.m1.4096x4096` | B5 | 0.0182 | 0.0464 | 2.55x | 1844.9 | 1.844 |
| `gemm.dual_bf16.m1024.14336x4096` | B7 | 5.3454 | 5.4992 | 1.03x | 51.0 | 44.996 |
| `gemm.dual_bf16.m2048.4096x4096` | B7 | 2.9970 | 3.0920 | 1.03x | 33.6 | 45.859 |
| `gemm.dual_bf16.m512.4096x4096` | B7 | 0.8443 | 0.8126 | 0.96x | 89.4 | 40.695 |
| `gemm.dual_bf16.m64.4096x4096` | B7 | 0.2730 | 0.1886 | 0.69x | 249.7 | 15.734 |
| `gemm.fp4_gemv.mx.m1.14336x4096` | B5 | 0.0424 | 5.4329 | 128.06x | 736.2 | 2.768 |
| `gemm.fp4_gemv.mx.m1.4096x4096` | B5 | 0.0151 | 1.4411 | 95.52x | 591.9 | 2.224 |
| `gemm.fp4_gemv.nv.m1.14336x4096` | B5 | 0.0504 | 5.4777 | 108.59x | 655.5 | 2.328 |
| `gemm.fp4_gemv.nv.m1.4096x4096` | B5 | 0.0169 | 1.4580 | 86.32x | 559.7 | 1.987 |
| `gemm.fp4_marlin.mx.m1.14336x4096` | B5 | 0.0338 | — | — | 923.6 | 3.473 |
| `gemm.fp4_marlin.mx.m1.4096x4096` | B5 | 0.0116 | — | — | 772.1 | 2.901 |
| `gemm.fp4_marlin.mx.m16.4096x4096` | B5 | 0.0171 | — | — | 535.2 | 31.315 |
| `gemm.fp4_marlin.mx.m4.4096x4096` | B5 | 0.0119 | — | — | 756.4 | 11.307 |
| `gemm.fp4_marlin.nv.m1.14336x4096` | B5 | 0.0422 | — | — | 782.8 | 2.780 |
| `gemm.fp4_marlin.nv.m1.4096x4096` | B5 | 0.0136 | — | — | 696.2 | 2.471 |
| `gemm.fp4_marlin.nv.m16.4096x4096` | B5 | 0.0194 | — | — | 499.0 | 27.619 |
| `gemm.fp4_marlin.nv.m4.4096x4096` | B5 | 0.0139 | — | — | 685.4 | 9.680 |
| `gemm.fp8_block_quant.bf16.4096x4096` | B4 | 0.1560 | 1.0634 | 6.82x | 322.7 | 0.000 |
| `gemm.fp8_block_quant.ue8m0.bf16.4096x4096` | B4 | 0.1560 | 1.0717 | 6.87x | 322.7 | 0.000 |
| `gemm.fp8_gemv.blockwise.m1.14336x4096` | B5 | 0.1394 | 1.5861 | 11.38x | 421.5 | 0.842 |
| `gemm.fp8_gemv.blockwise.m1.4096x14336` | B5 | 0.1421 | 1.6015 | 11.27x | 413.4 | 0.826 |
| `gemm.fp8_gemv.blockwise.m1.4096x4096` | B5 | 0.0191 | 0.4531 | 23.73x | 879.9 | 1.758 |
| `gemm.fp8_gemv.rowwise.m1.14336x4096` | B5 | 0.1394 | 0.9332 | 6.70x | 422.0 | 0.843 |
| `gemm.fp8_gemv.rowwise.m1.4096x14336` | B5 | 0.1409 | 0.9427 | 6.69x | 417.1 | 0.833 |
| `gemm.fp8_gemv.rowwise.m1.4096x4096` | B5 | 0.0106 | 0.2482 | 23.32x | 1579.9 | 3.154 |
| `gemm.fp8_gemv.rowwise.m4.4096x14336` | B5 | 0.1434 | 0.9643 | 6.72x | 410.6 | 3.276 |
| `gemm.fp8_gemv.rowwise.m4.4096x4096` | B5 | 0.0222 | 0.2538 | 11.45x | 760.7 | 6.056 |
| `gemm.fp8_mma.blockwise.m1024.4096x4096` | B7 | 0.4856 | — | — | 60.8 | 70.764 |
| `gemm.fp8_mma.blockwise.m256.4096x4096` | B7 | 0.1523 | — | — | 131.0 | 56.393 |
| `gemm.fp8_mma.blockwise.m4096.4096x4096` | B7 | 1.9469 | — | — | 34.7 | 70.593 |
| `gemm.fp8_mma.rowwise.m1024.4096x4096` | B7 | 0.4256 | — | — | 69.0 | 80.731 |
| `gemm.fp8_mma.rowwise.m256.4096x4096` | B7 | 0.1170 | — | — | 170.4 | 73.401 |
| `gemm.fp8_mma.rowwise.m4096.4096x4096` | B7 | 1.6049 | — | — | 41.8 | 85.636 |
| `gemm.fp8_mma.tensorwise.m1024.4096x4096` | B7 | 0.4214 | — | — | 69.7 | 81.544 |
| `gemm.fp8_mma.tensorwise.m256.4096x4096` | B7 | 0.1155 | — | — | 172.5 | 74.365 |
| `gemm.fp8_mma.tensorwise.m4096.4096x4096` | B7 | 1.5913 | — | — | 42.2 | 86.369 |
| `gemm.gguf_mmq.q4_k.m2048.4096x4096` | B7 | 1.7763 | — | — | 34.8 | 38.686 |
| `gemm.gguf_mmq.q4_k.m512.4096x4096` | B7 | 0.4609 | — | — | 48.9 | 37.278 |
| `gemm.gguf_mmq.q4_k.m64.4096x4096` | B7 | 0.1172 | — | — | 94.5 | 18.320 |
| `gemm.gguf_mmq.q8_0.m2048.4096x4096` | B7 | 1.6188 | — | — | 43.4 | 42.450 |
| `gemm.gguf_mmq.q8_0.m512.4096x4096` | B7 | 0.4134 | — | — | 74.8 | 41.561 |
| `gemm.gguf_mmq.q8_0.m64.4096x4096` | B7 | 0.1096 | — | — | 177.6 | 19.589 |
| `gemm.gguf_mmvq.q4_0.m1.14336x4096` | B5 | 0.0347 | 5.8396 | 168.12x | 952.0 | 3.381 |
| `gemm.gguf_mmvq.q4_0.m1.4096x4096` | B5 | 0.0124 | 1.4564 | 117.63x | 763.6 | 2.710 |
| `gemm.gguf_mmvq.q4_k.m1.14336x4096` | B5 | 0.0541 | 5.6286 | 104.09x | 611.5 | 2.172 |
| `gemm.gguf_mmvq.q4_k.m1.4096x4096` | B5 | 0.0184 | 1.6877 | 91.62x | 513.2 | 1.822 |
| `gemm.gguf_mmvq.q8_0.m1.14336x4096` | B5 | 0.1504 | 3.8350 | 25.49x | 415.0 | 0.781 |
| `gemm.gguf_mmvq.q8_0.m1.4096x4096` | B5 | 0.0186 | 1.0301 | 55.31x | 958.0 | 1.802 |
| `gemm.glu_interleave.bf16.8192x4096` | B4 | 0.3488 | 0.3664 | 1.05x | 384.8 | 0.000 |
| `gemm.grouped_gemv.fp8.e32x2rows.2048x2048` | B7 | 0.3173 | 2.2117 | 6.97x | 424.7 | 1.692 |
| `gemm.int8_colsum.4096x4096` | B4 | 0.0088 | 0.3537 | 40.00x | 1897.5 | 0.000 |
| `gemm.int8_gemv.w8a16.m1.14336x4096` | B5 | 0.1399 | 0.8843 | 6.32x | 420.4 | 0.840 |
| `gemm.int8_gemv.w8a16.m1.4096x14336` | B5 | 0.1417 | 0.8871 | 6.26x | 414.9 | 0.829 |
| `gemm.int8_gemv.w8a16.m1.4096x4096` | B5 | 0.0115 | 0.2294 | 19.86x | 1455.4 | 2.905 |
| `gemm.int8_gemv.w8a8.m1.14336x4096` | B5 | 0.1391 | 0.8831 | 6.35x | 422.9 | 0.844 |
| `gemm.int8_gemv.w8a8.m1.4096x14336` | B5 | 0.1415 | 0.8981 | 6.35x | 415.2 | 0.830 |
| `gemm.int8_gemv.w8a8.m1.4096x4096` | B5 | 0.0100 | 0.2413 | 24.08x | 1676.8 | 3.348 |
| `gemm.int8_gemv.w8a8.m4.4096x14336` | B5 | 0.1418 | 0.9165 | 6.46x | 414.9 | 3.313 |
| `gemm.int8_gemv.w8a8.m4.4096x4096` | B5 | 0.0168 | 0.2496 | 14.83x | 1000.5 | 7.972 |
| `gemm.int8_mma.m1024.4096x4096` | B7 | 0.2856 | — | — | 102.9 | 120.296 |
| `gemm.int8_mma.m256.4096x4096` | B7 | 0.0771 | — | — | 258.6 | 111.377 |
| `gemm.int8_mma.m4096.4096x4096` | B7 | 1.0349 | — | — | 64.9 | 132.810 |
| `gemm.lora.rank16.t512.4096x4096` | B7 | 0.1287 | — | — | 73.6 | 1.042 |
| `gemm.lora.rank64.t512.4096x4096` | B7 | 0.1357 | — | — | 93.7 | 3.956 |
| `gemm.mma_bf16.m1024.4096x4096` | B7 | 0.7661 | 0.7323 | 0.96x | 65.7 | 44.850 |
| `gemm.mma_bf16.m128.4096x4096` | B7 | 0.1388 | 0.1124 | 0.81x | 256.8 | 30.939 |
| `gemm.mma_bf16.m2048.14336x4096` | B7 | 5.0422 | 5.1657 | 1.02x | 38.3 | 47.701 |
| `gemm.mma_bf16.m2048.4096x14336` | B7 | 5.0426 | 5.3050 | 1.05x | 38.3 | 47.698 |
| `gemm.mma_bf16.m256.4096x4096` | B7 | 0.2161 | 0.2114 | 0.98x | 174.7 | 39.747 |
| `gemm.mma_bf16.m32.4096x4096` | B7 | 0.0486 | 0.0664 | 1.37x | 700.6 | 22.076 |
| `gemm.mma_bf16.m4096.4096x4096` | B7 | 2.8697 | 3.0452 | 1.06x | 35.1 | 47.893 |
| `gemm.mma_bf16.m64.4096x4096` | B7 | 0.0950 | 0.0676 | 0.71x | 364.1 | 22.594 |
| `gemm.mma_bf16_tuned.m1024.4096x4096` | B7 | 0.7168 | 0.7556 | 1.05x | 70.2 | 47.938 |
| `gemm.mma_bf16_tuned.m256.4096x4096` | B7 | 0.2150 | 0.2105 | 0.98x | 175.6 | 39.947 |
| `gemm.mma_bf16_tuned.m4096.4096x4096` | B7 | 2.8087 | 3.0504 | 1.09x | 35.8 | 48.934 |
| `gemm.mma_bf16_tuned.m64.4096x4096` | B7 | 0.0714 | 0.0680 | 0.95x | 484.4 | 30.065 |
| `gemm.mma_fp16.m1024.4096x4096` | B7 | 0.7655 | 0.7492 | 0.98x | 65.8 | 44.883 |
| `gemm.mma_fp16.m256.4096x4096` | B7 | 0.2160 | 0.2107 | 0.98x | 174.7 | 39.764 |
| `gemm.moe_gemv.fp8.s32.2048x2048` | B5 | 0.3241 | 2.8702 | 8.85x | 414.9 | 0.828 |
| `gemm.nf4_lut.b64.m1.14336x4096` | B5 | 0.0385 | 6.9827 | 181.28x | 858.4 | 3.049 |
| `gemm.nf4_lut.b64.m1.4096x14336` | B5 | 0.0397 | 6.5427 | 165.00x | 833.9 | 2.962 |
| `gemm.nf4_lut.b64.m1.4096x4096` | B5 | 0.0128 | 1.7589 | 136.98x | 736.3 | 2.613 |
| `gemm.nf4_lut.b64.m2.4096x4096` | B5 | 0.0162 | 1.7689 | 109.18x | 584.5 | 4.142 |
| `gemm.nf4_lut.b64.m4.4096x4096` | B5 | 0.0262 | 1.7636 | 67.37x | 363.0 | 5.128 |
| `gemm.nf4_lut.b64.m8.4096x4096` | B5 | 0.0654 | 1.7679 | 27.04x | 146.3 | 4.105 |
| `gemm.router.fp32out.m8.256x7168` | B5 | 0.0102 | 0.0619 | 6.05x | 371.0 | 2.872 |
| `gemm.router_split.fp32.4096x4096` | B4 | 0.3501 | 1.6460 | 4.70x | 383.4 | 0.000 |
| `gemm.scale_swizzle.14336x256` | B4 | 0.0066 | 0.0684 | 10.38x | 1113.8 | 0.000 |
| `gemm.scale_tma.f32.32768x56` | B4 | 0.0087 | 0.0616 | 7.04x | 1678.6 | 0.000 |
| `gemm.scale_tma.ue8m0.32768x56` | B4 | 0.0129 | 0.0655 | 5.08x | 711.7 | 0.000 |
| `gemm.transpose.bf16.4096x4096` | B4 | 0.1770 | 0.4111 | 2.32x | 379.2 | 0.000 |
| `gemm.transpose.fp8.4096x4096` | B4 | 0.0323 | 0.3966 | 12.29x | 1039.6 | 0.000 |
| `gemm.transpose.int4.4096x4096` | B4 | 0.0348 | 0.6673 | 19.17x | 482.0 | 0.000 |
| `gemm.w4a16_awq.g128.m1.14336x4096` | B5 | 0.0398 | 7.3636 | 184.97x | 767.2 | 2.950 |
| `gemm.w4a16_awq.g128.m1.4096x14336` | B5 | 0.0666 | 7.4226 | 111.40x | 458.4 | 1.763 |
| `gemm.w4a16_awq.g128.m1.4096x4096` | B5 | 0.0175 | 2.1611 | 123.83x | 500.4 | 1.923 |
| `gemm.w4a16_awq.g128.m4.4096x14336` | B5 | 0.0730 | 7.4206 | 101.68x | 420.0 | 6.437 |
| `gemm.w4a16_awq.g128.m4.4096x4096` | B5 | 0.0224 | 2.1579 | 96.51x | 392.8 | 6.003 |
| `gemm.w4a16_awq_marlin.g128.m1.14336x4096` | B7 | 0.0267 | — | — | 1143.8 | 4.398 |
| `gemm.w4a16_awq_marlin.g128.m1.4096x14336` | B7 | 0.0283 | — | — | 1077.7 | 4.144 |
| `gemm.w4a16_awq_marlin.g128.m1.4096x4096` | B7 | 0.0091 | — | — | 958.9 | 3.684 |
| `gemm.w4a16_awq_marlin.g128.m16.4096x4096` | B7 | 0.0154 | — | — | 584.9 | 34.974 |
| `gemm.w4a16_awq_marlin.g128.m4.4096x14336` | B7 | 0.0291 | — | — | 1053.5 | 16.144 |
| `gemm.w4a16_awq_marlin.g128.m4.4096x4096` | B7 | 0.0093 | — | — | 949.1 | 14.506 |
| `gemm.w4a16_gptq.g128.m1.14336x4096` | B5 | 0.0408 | 6.0744 | 148.93x | 748.9 | 2.879 |
| `gemm.w4a16_gptq.g128.m1.4096x14336` | B5 | 0.0489 | 6.1313 | 125.46x | 625.0 | 2.403 |
| `gemm.w4a16_gptq.g128.m1.4096x4096` | B5 | 0.0148 | 1.7485 | 118.26x | 590.6 | 2.269 |
| `gemm.w4a8_chn.m1024.4096x4096` | B7 | 0.4147 | — | — | 50.6 | 82.856 |
| `gemm.w4a8_chn.m256.4096x4096` | B7 | 0.1312 | — | — | 88.0 | 65.451 |
| `gemm.w4a8_chn.m4096.4096x4096` | B7 | 1.6239 | — | — | 36.2 | 84.637 |
| `gemm.w4a8_g128.m1024.4096x4096` | B7 | 0.4488 | — | — | 47.4 | 76.563 |
| `gemm.w4a8_g128.m256.4096x4096` | B7 | 0.1337 | — | — | 88.4 | 64.250 |
| `gemm.w4a8_g128.m4096.4096x4096` | B7 | 1.7183 | — | — | 34.3 | 79.984 |


### A.13 moe

| case | step | kops ms | PyTorch ms | ratio | kops GB/s | kops TFLOP/s |
|---|---|---:|---:|---:|---:|---:|
| `chain_moe_route_permute.bf16.t4096.h7168.e256.k8` | B6 | 1.4592 | 1.4565 | 1.00x | 365.2 | 0.000 |
| `moe.align.e256.block128.slots32768` | B6 | 0.0088 | 0.3348 | 37.83x | 37.2 | 0.000 |
| `moe.count_offsets.e256.align128.slots32768` | B6 | 0.0086 | 0.0789 | 9.15x | 15.6 | 0.000 |
| `moe.fused_router.sigmoid.e256.h7168.k8.t32` | B6 | 0.0465 | 0.1062 | 2.28x | 88.8 | 0.000 |
| `moe.fused_router.sigmoid.e256.h7168.k8.t8` | B6 | 0.0155 | 0.1064 | 6.88x | 244.8 | 0.000 |
| `moe.grouped.sigmoid.e256.g8.tg4.k8.t2048` | B6 | 0.0178 | 0.2381 | 13.41x | 125.5 | 0.000 |
| `moe.hash.e256.k8.t4096` | B6 | 0.0029 | 0.0356 | 12.48x | 149.4 | 0.000 |
| `moe.latent_combine.bf16.t1024.k8.l3584` | B6 | 0.1677 | 1.2175 | 7.26x | 394.0 | 0.000 |
| `moe.permute.bf16.t256.h7168.k8.e256` | B6 | 0.0766 | 0.0780 | 1.02x | 431.1 | 0.000 |
| `moe.permute_fp8.t256.h7168.k8.e256` | B6 | 0.2048 | 1.1707 | 5.72x | 91.8 | 0.000 |
| `moe.route.sigmoid.e256.k8.t4096` | B6 | 0.0123 | 0.1701 | 13.80x | 361.4 | 0.000 |
| `moe.sum.bf16.t1024.k8.h7168` | B6 | 0.2956 | 0.3099 | 1.05x | 446.9 | 0.000 |
| `moe.unpermute.bf16.t1024.h7168.k8` | B6 | 0.3677 | 2.6078 | 7.09x | 439.1 | 0.000 |


## Phụ lục B — 32 case chuyển từ Phần I sang Phần II

32 case này từng nằm ngoài tổng hợp B0–B10; nay thuộc Phần II (B11 3, B15 18, B17 11) và có mặt trong Phụ lục D. Bảng giữ lại để đối chiếu:

| case | family | step | kops ms |
|---|---|---|---:|
| `model_ops.attn_res.bf16.t1.h7168.s1` | model_ops | B15 | 0.0048 |
| `model_ops.attn_res.bf16.t1.h7168.s1.norm` | model_ops | B15 | 0.0069 |
| `model_ops.attn_res.bf16.t1.h7168.s4` | model_ops | B15 | 0.0075 |
| `model_ops.attn_res.bf16.t1.h7168.s4.norm` | model_ops | B15 | 0.0096 |
| `model_ops.attn_res.bf16.t1.h7168.s8` | model_ops | B15 | 0.0115 |
| `model_ops.attn_res.bf16.t1.h7168.s8.norm` | model_ops | B15 | 0.0133 |
| `model_ops.attn_res.bf16.t16.h7168.s1` | model_ops | B15 | 0.0049 |
| `model_ops.attn_res.bf16.t16.h7168.s1.norm` | model_ops | B15 | 0.0070 |
| `model_ops.attn_res.bf16.t16.h7168.s4` | model_ops | B15 | 0.0081 |
| `model_ops.attn_res.bf16.t16.h7168.s4.norm` | model_ops | B15 | 0.0101 |
| `model_ops.attn_res.bf16.t16.h7168.s8` | model_ops | B15 | 0.0124 |
| `model_ops.attn_res.bf16.t16.h7168.s8.norm` | model_ops | B15 | 0.0141 |
| `model_ops.attn_res.bf16.t256.h7168.s1` | model_ops | B15 | 0.0153 |
| `model_ops.attn_res.bf16.t256.h7168.s1.norm` | model_ops | B15 | 0.0248 |
| `model_ops.attn_res.bf16.t256.h7168.s4` | model_ops | B15 | 0.0288 |
| `model_ops.attn_res.bf16.t256.h7168.s4.norm` | model_ops | B15 | 0.0386 |
| `model_ops.attn_res.bf16.t256.h7168.s8` | model_ops | B15 | 0.0720 |
| `model_ops.attn_res.bf16.t256.h7168.s8.norm` | model_ops | B15 | 0.1267 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.prod3` | merge | B17 | 0.5593 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo3` | merge | B17 | 0.5742 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4` | merge | B17 | 0.7045 |
| `lowbit.prefill.bf16.d128.t512.kv4096.h32x8.prod3` | merge | B17 | 5.9423 |
| `lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo4` | merge | B17 | 3.0658 |
| `lowbit.store.bf16.t2048.h8.d128.turbo3` | merge | B17 | 0.0624 |
| `lowbit.store.bf16.t2048.h8.d128.turbo4` | merge | B17 | 0.0590 |
| `lowbit.store.bf16.t4096.h8.d128.turbo3` | merge | B17 | 0.1228 |
| `lowbit.store.bf16.t4096.h8.d128.turbo4` | merge | B17 | 0.1163 |
| `gemm.grouped_prepare.e32.1024x1024` | gemm | B11 | 0.0068 |
| `gemm.machete_prepack.int4.4096x4096` | gemm | B11 | 0.0294 |
| `gemm.machete_prepack.int8.4096x4096` | gemm | B11 | 0.0251 |
| `moe.w2.gemm.bf16.e8.n2048.k4096.slots2048` | moe | B17 | 2.1043 |
| `moe.w2.pack.bf16.e8.n2048.k4096.g32` | moe | B17 | 0.4416 |

Ghi chú: `ds_mla_v40/v41/fp4` là codec của B14 nhưng chạy trên kernel B2 nên
được giữ trong phạm vi; `attn_res` (B15), `lowbit_*`/`moe_w2_*` (B17),
`machete_prepack`/`grouped_prepare` (B11) bị loại.

## Phụ lục C — Danh mục artifact

| Đường dẫn | Nội dung |
|---|---|
| `out/rtx5060/kops/<family>.json` | 13 file kops, 475 result |
| `out/rtx5060/torch/<family>.json` | 13 file PyTorch, 335 result |
| `out/rtx5060/torch/norm-hf.json`, `chains-hf.json` | PyTorch eager kiểu HF |
| `out/rtx5060/compare/<family>.json` | so sánh từng family |
| `out/rtx5060/cases/<family>.json` | case table dùng chung |
| `out/rtx5060/thirdparty/kops-spec-cold.json` | kops spec, cold 256 MB |
| `out/rtx5060/thirdparty/torch-spec-cold.json` | PyTorch spec, cold |
| `out/rtx5060/thirdparty/hpc-sampler.json` | hpc-ops sampler |
| `out/rtx5060/thirdparty/sglang-tree.json` | SGLang tree target-only |
| `out/rtx5060/thirdparty/sdpa-merge.json` | torch SDPA, 5 case prefill |
| `out/rtx5060/thirdparty/kops-gguf-cold.json` | kops GGUF MMVQ, cold |
| `out/rtx5060/thirdparty/torch-gguf-cold.json` | PyTorch GGUF dequant+matmul |
| `out/rtx5060/thirdparty/llamacpp-gguf.csv` | llama.cpp GGUF GEMV |
| `out/rtx5060/kops/b13_gemm.json`, `torch/b13_gemm.json` | B13 SM120 tile |
| `out/rtx5060/kops/mega.json` | B16 mega/modular/graph |
| `out/rtx5060/remaining/cases.json` | bảng chuẩn hoá B11–B17 |
| `out/b11-b15/opt-*.json`, `torch-*.json` | B11/B15 đã tối ưu |
| `out/rtx5060/kops/sparse-new.json` | B14 sau cập nhật (22 case) |
| `out/rtx5060/torch/sparse-new.json`, `torch/sparse-csa.json` | B14 torch + CSA/HCA |
| `out/b12/kops-opt-clean.json`, `torch-opt.json`, `nccl.json` | B12 |
| `out/b17/lowbit-bench.json`, `w2-bench.json`, `attention-rs-*.json` | B17 |
| `out/rtx5060/logs/` | log ctest, từng family, third-party |
| `out/rtx5060/bin/` | `hpc_sampler`, `sglang_tree`, `llamacpp_mmvq` |

Các JSON kops mang `provenance.commit`, `dirty=1`, `flags_hash`, `archs=120a`,
`toolkit=13.4.92`; `device.name` là RTX 5060 Ti, `peak_gbs=448.0`.

## Phụ lục D — Bảng đầy đủ 164 case B11–B17

Bảng dưới đây gộp các nguồn `out/b11-b15/opt-*.json`, `out/b12/kops-opt-clean.json`, `out/b17/*.json`, `out/rtx5060/kops/b13_gemm.json` và `out/rtx5060/kops/mega.json`. Cột baseline ghi tên baseline thực tế (torch, pytorch-nccl, attention.rs, mega/modular).

### D.1 B11 — MLA + prepack

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `mla_decode.bf16.kv64k.q1` | 3.0126 | torch-mla | 0.8617 | 0.29x | 25.1 | 6.06 |
| `mla_decode.bf16.kv64k.q4` | 3.0148 | — | — | — | 25.4 | 24.22 |
| `mla_decode.fp8.kv64k.q1` | 4.6710 | torch-mla | 1.7161 | 0.37x | 9.3 | 3.91 |
| `mla_decode.fp8.kv64k.q4` | 4.6830 | — | — | — | 9.4 | 15.59 |
| `mla_decode.bf16.kv256k.q1` | 12.0688 | — | — | — | 25.1 | 6.05 |
| `mla_decode.bf16.kv256k.q4` | 12.0385 | — | — | — | 25.2 | 24.26 |
| `mla_decode.fp8.kv256k.q1` | 18.1153 | — | — | — | 9.5 | 4.03 |
| `mla_decode.fp8.kv256k.q4` | 18.4316 | — | — | — | 9.4 | 15.85 |
| `mla_decode.bf16.kv1m.q1` | 48.2594 | torch-mla | 42.9691 | 0.89x | 25.0 | 6.05 |
| `mla_decode.bf16.kv1m.q4` | 48.2912 | — | — | — | 25.0 | 24.19 |
| `mla_decode.fp8.kv1m.q1` | 72.4652 | torch-mla | 81.5731 | 1.13x | 9.5 | 4.03 |
| `mla_decode.fp8.kv1m.q4` | 73.8265 | — | — | — | 9.3 | 15.82 |
| `mla_decode.bf16.kv64k.q1.s8` | 0.7627 | torch-mla | 0.8613 | 1.13x | 99.3 | 23.93 |
| `mla_decode.bf16.kv256k.q1.s8` | 3.0278 | — | — | — | 99.8 | 24.11 |
| `mla_decode.bf16.kv1m.q1.s8` | 12.0894 | torch-mla | 48.8590 | 4.04x | 99.9 | 24.16 |
| `mla_decode.fp8.kv64k.q1.s8` | 1.1545 | torch-mla | 1.7155 | 1.49x | 37.5 | 15.81 |
| `mla_decode.fp8.kv256k.q1.s8` | 4.5439 | — | — | — | 37.9 | 16.07 |
| `mla_decode.fp8.kv1m.q1.s8` | 18.1181 | torch-mla | 69.7230 | 3.85x | 38.0 | 16.12 |
| `mla_decode.bf16.kv1m.q4.s8` | 72.0729 | — | — | — | 16.8 | 16.21 |
| `mla_decode.fp8.kv1m.q4.s8` | 88.7380 | — | — | — | 7.8 | 13.16 |
| `mla_decode.bf16.kv64k.q1.s16` | 0.7646 | — | — | — | 99.1 | 23.87 |
| `mla_decode.bf16.kv64k.q1.s32` | 0.7663 | — | — | — | 98.9 | 23.82 |
| `mla_decode.bf16.kv256k.q1.s16` | 3.0281 | — | — | — | 99.8 | 24.11 |
| `mla_decode.bf16.kv256k.q1.s32` | 3.0280 | — | — | — | 99.8 | 24.11 |
| `mla_decode.bf16.kv1m.q1.s16` | 12.0819 | — | — | — | 100.0 | 24.17 |
| `mla_decode.bf16.kv1m.q1.s32` | 12.0748 | — | — | — | 100.1 | 24.19 |
| `mla_decode.fp8.kv64k.q1.s16` | 1.1613 | — | — | — | 37.3 | 15.72 |
| `mla_decode.fp8.kv64k.q1.s32` | 1.0579 | — | — | — | 40.9 | 17.25 |
| `mla_decode.fp8.kv1m.q1.s16` | 18.0871 | — | — | — | 38.0 | 16.15 |
| `mla_decode.fp8.kv1m.q1.s32` | 17.9832 | — | — | — | 38.3 | 16.24 |
| `mla_metadata.b8.parts132.block64` | 0.0197 | — | — | — | 0.2 | 0.00 |
| `mla_metadata.b32.parts132.block64` | 0.0213 | — | — | — | 0.2 | 0.00 |
| `mla_metadata.b128.parts192.block64` | 0.0339 | — | — | — | 0.2 | 0.00 |
| `mla_metadata.b32.parts132.block16` | 0.0213 | — | — | — | 0.2 | 0.00 |
| `gemm.machete_prepack.int4.4096x4096` | 0.0294 | — | — | — | 571.6 | 0.00 |
| `gemm.machete_prepack.int8.4096x4096` | 0.0251 | — | — | — | 1334.9 | 0.00 |
| `gemm.grouped_prepare.e32.1024x1024` | 0.0068 | — | — | — | 0.9 | 0.00 |

### D.2 B12 — communication

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `allgather_bf16_t128_h4096` | 0.2102 | pytorch-nccl | 0.2050 | 0.98x | 5.0 | 0.00 |
| `allgather_bf16_t128_h8192` | 0.4891 | pytorch-nccl | 0.3621 | 0.74x | 4.3 | 0.00 |
| `allgather_bf16_t1_h4096` | 0.0073 | pytorch-nccl | 0.0578 | 7.87x | 1.1 | 0.00 |
| `allgather_bf16_t1_h8192` | 0.0077 | pytorch-nccl | 0.0451 | 5.88x | 2.1 | 0.00 |
| `allgather_bf16_t2048_h4096` | 3.4405 | pytorch-nccl | 3.1059 | 0.90x | 4.9 | 0.00 |
| `allgather_bf16_t32_h4096` | 0.0720 | pytorch-nccl | 0.0628 | 0.87x | 3.6 | 0.00 |
| `allgather_bf16_t32_h8192` | 0.1062 | pytorch-nccl | 0.1345 | 1.27x | 4.9 | 0.00 |
| `allgather_bf16_t512_h4096` | 0.9157 | pytorch-nccl | 0.9348 | 1.02x | 4.6 | 0.00 |
| `allgather_bf16_t8_h4096` | 0.0261 | pytorch-nccl | 0.0502 | 1.92x | 2.5 | 0.00 |
| `allgather_bf16_t8_h8192` | 0.0269 | pytorch-nccl | 0.0495 | 1.84x | 4.9 | 0.00 |
| `allgather_gemm_bf16_w2_m512_n1024_k1024_c8` | 0.3885 | pytorch-nccl | 0.2365 | 0.61x | 8.1 | 5.53 |
| `allgather_gemm_bf16_w4_m256_n1024_k1024_c8` | 0.7689 | — | — | — | 3.4 | 2.79 |
| `allreduce_bf16_t128_h4096` | 0.2320 | pytorch-nccl | 0.2221 | 0.96x | 4.5 | 0.00 |
| `allreduce_bf16_t128_h8192` | 0.5264 | pytorch-nccl | 0.3011 | 0.57x | 4.0 | 0.00 |
| `allreduce_bf16_t1_h4096` | 0.0071 | pytorch-nccl | 0.0657 | 9.32x | 1.2 | 0.00 |
| `allreduce_bf16_t1_h8192` | 0.0078 | pytorch-nccl | 0.0645 | 8.31x | 2.1 | 0.00 |
| `allreduce_bf16_t2048_h4096` | 3.7528 | pytorch-nccl | 3.1193 | 0.83x | 4.5 | 0.00 |
| `allreduce_bf16_t2048_h8192` | 6.8398 | pytorch-nccl | 5.9522 | 0.87x | 4.9 | 0.00 |
| `allreduce_bf16_t32_h4096` | 0.1082 | pytorch-nccl | 0.0703 | 0.65x | 2.4 | 0.00 |
| `allreduce_bf16_t32_h8192` | 0.1219 | pytorch-nccl | 0.1286 | 1.05x | 4.3 | 0.00 |
| `allreduce_bf16_t512_h4096` | 0.9100 | pytorch-nccl | 0.7815 | 0.86x | 4.6 | 0.00 |
| `allreduce_bf16_t8192_h4096` | 14.2032 | pytorch-nccl | 12.3499 | 0.87x | 4.7 | 0.00 |
| `allreduce_bf16_t8_h4096` | 0.0201 | pytorch-nccl | 0.0602 | 3.00x | 3.3 | 0.00 |
| `allreduce_bf16_t8_h8192` | 0.0272 | pytorch-nccl | 0.0647 | 2.37x | 4.8 | 0.00 |
| `allreduce_fusion_bf16_t128_h4096` | 0.1855 | pytorch-nccl | 0.2157 | 1.16x | 5.7 | 0.00 |
| `allreduce_fusion_bf16_t1_h4096` | 0.0055 | pytorch-nccl | 0.1038 | 18.90x | 1.5 | 0.00 |
| `allreduce_fusion_bf16_t2048_h4096` | 3.7056 | pytorch-nccl | 3.2060 | 0.87x | 4.5 | 0.00 |
| `allreduce_fusion_bf16_t32_h4096` | 0.0495 | pytorch-nccl | 0.1113 | 2.25x | 5.3 | 0.00 |
| `allreduce_fusion_bf16_t512_h4096` | 0.7813 | pytorch-nccl | 0.8094 | 1.04x | 5.4 | 0.00 |
| `allreduce_fusion_bf16_t8_h4096` | 0.0106 | pytorch-nccl | 0.8195 | 77.19x | 6.2 | 0.00 |
| `allreduce_fusion_fp8_bf16_t128_h4096` | 0.3154 | pytorch-nccl | 0.2568 | 0.81x | 1.7 | 0.00 |
| `allreduce_fusion_fp8_bf16_t1_h4096` | 0.0064 | pytorch-nccl | 3.8374 | 595.86x | 0.6 | 0.00 |
| `allreduce_fusion_fp8_bf16_t2048_h4096` | 5.7896 | pytorch-nccl | 3.7577 | 0.65x | 1.4 | 0.00 |
| `allreduce_fusion_fp8_bf16_t32_h4096` | 0.0821 | pytorch-nccl | 0.2503 | 3.05x | 1.6 | 0.00 |
| `allreduce_fusion_fp8_bf16_t512_h4096` | 1.3223 | pytorch-nccl | 0.9329 | 0.71x | 1.6 | 0.00 |
| `allreduce_fusion_fp8_bf16_t8_h4096` | 0.0143 | pytorch-nccl | 0.3254 | 22.79x | 2.3 | 0.00 |
| `allreduce_two_shot_bf16_t128_h4096` | 0.2794 | pytorch-nccl | 0.2106 | 0.75x | 3.8 | 0.00 |
| `allreduce_two_shot_bf16_t128_h8192` | 0.5993 | pytorch-nccl | 0.3880 | 0.65x | 3.5 | 0.00 |
| `allreduce_two_shot_bf16_t2048_h4096` | 5.2204 | pytorch-nccl | 3.2602 | 0.62x | 3.2 | 0.00 |
| `allreduce_two_shot_bf16_t2048_h8192` | 10.3697 | pytorch-nccl | 6.4125 | 0.62x | 3.2 | 0.00 |
| `allreduce_two_shot_bf16_t32_h4096` | 0.1197 | pytorch-nccl | 0.0661 | 0.55x | 2.2 | 0.00 |
| `allreduce_two_shot_bf16_t32_h8192` | 0.1481 | pytorch-nccl | 0.1181 | 0.80x | 3.5 | 0.00 |
| `allreduce_two_shot_bf16_t512_h4096` | 1.1351 | pytorch-nccl | 0.7736 | 0.68x | 3.7 | 0.00 |
| `allreduce_two_shot_bf16_t8192_h4096` | 21.0227 | pytorch-nccl | 12.4691 | 0.59x | 3.2 | 0.00 |
| `allreduce_two_shot_bf16_t8_h8192` | 0.0299 | pytorch-nccl | 0.0684 | 2.29x | 4.4 | 0.00 |
| `ep_dispatch_combine_bf16_ll_w4_t256_h4096_e64_k8` | 18.0247 | — | — | — | 0.9 | 0.00 |
| `ep_dispatch_combine_bf16_normal_w2_t128_h2048_e32_k4` | 0.5051 | — | — | — | 2.1 | 0.00 |
| `ep_dispatch_combine_bf16_normal_w4_t256_h4096_e64_k8` | 5.1592 | — | — | — | 1.6 | 0.00 |
| `ep_dispatch_combine_bf16_normal_w8_t64_h1024_e16_k4` | 1.2756 | — | — | — | 0.8 | 0.00 |
| `ep_layout_bf16_e64_t1024_k8` | 0.0062 | — | — | — | 0.0 | 0.00 |
| `gemm_allreduce_bf16_w2_m1024_n1024_k1024_c8` | 0.4859 | pytorch-nccl | 0.4459 | 0.92x | 4.3 | 4.42 |
| `gemm_allreduce_norm_bf16_w2_m512_n4096_k4096_c8` | 3.6011 | pytorch-nccl | 1.1387 | 0.32x | 1.2 | 4.77 |
| `gemm_reduce_scatter_bf16_w2_m1024_n1024_k1024_c8` | 0.4573 | pytorch-nccl | 0.2470 | 0.54x | 4.6 | 4.70 |
| `peer_barrier_bf16_w2` | 0.0043 | — | — | — | 0.0 | 0.00 |
| `reduce_scatter_bf16_t128_h4096` | 0.2347 | pytorch-nccl | 0.1834 | 0.78x | 4.5 | 0.00 |
| `reduce_scatter_bf16_t128_h8192` | 0.4836 | pytorch-nccl | 0.3966 | 0.82x | 4.3 | 0.00 |
| `reduce_scatter_bf16_t1_h4096` | 0.0072 | pytorch-nccl | 0.0465 | 6.45x | 1.1 | 0.00 |
| `reduce_scatter_bf16_t1_h8192` | 0.0078 | pytorch-nccl | 0.0504 | 6.47x | 2.1 | 0.00 |
| `reduce_scatter_bf16_t2048_h4096` | 3.7429 | pytorch-nccl | 3.3676 | 0.90x | 4.5 | 0.00 |
| `reduce_scatter_bf16_t32_h4096` | 0.0736 | pytorch-nccl | 0.0660 | 0.90x | 3.6 | 0.00 |
| `reduce_scatter_bf16_t32_h8192` | 0.1259 | pytorch-nccl | 0.1128 | 0.90x | 4.2 | 0.00 |
| `reduce_scatter_bf16_t512_h4096` | 1.0936 | pytorch-nccl | 0.8015 | 0.73x | 3.8 | 0.00 |
| `reduce_scatter_bf16_t8_h4096` | 0.0229 | pytorch-nccl | 0.0472 | 2.06x | 2.9 | 0.00 |
| `reduce_scatter_bf16_t8_h8192` | 0.0266 | pytorch-nccl | 0.0569 | 2.14x | 4.9 | 0.00 |

### D.3 B13 — SM120 tile

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `gemm.mma_sm120.bf16.m2048.2048x2048.t64x128` | 0.3646 | pytorch | 0.3734 | 1.02x | 69.0 | 47.13 |
| `gemm.mma_sm120.bf16.m4096.4096x4096.t128x128` | 2.7837 | pytorch | 2.9681 | 1.07x | 36.2 | 49.37 |

### D.4 B14 — sparse

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `sparse_topk.fp32.r1.kv64k.topk512` | 0.0350 | torch | 0.0966 | 2.76x | 7.6 | 0.00 |
| `sparse_topk.fp32.r16.kv64k.topk512` | 0.0365 | torch | 0.1019 | 2.79x | 116.7 | 0.00 |
| `sparse_topk.fp32.r1.kv1m.topk512` | 0.0267 | torch | 0.0991 | 3.71x | 157.3 | 0.00 |
| `sparse_topk.fp32.r16.kv1m.topk512` | 0.7543 | torch | 0.9807 | 1.30x | 89.0 | 0.00 |
| `sparse_topk.fp32.r1.kv64k.topk2048` | 0.0407 | torch | 0.0951 | 2.34x | 6.8 | 0.00 |
| `sparse_topk.fp32.r16.kv64k.topk2048` | 0.0414 | torch | 0.1040 | 2.51x | 107.8 | 0.00 |
| `sparse_topk.fp32.r1.kv1m.topk2048` | 0.0268 | torch | 0.0991 | 3.70x | 157.1 | 0.00 |
| `sparse_topk.fp32.r16.kv1m.topk2048` | 0.7677 | torch | 0.9754 | 1.27x | 87.8 | 0.00 |
| `sparse_topk.fp32.r1.kv256k.topk2048` | 0.0186 | torch | 0.0949 | 5.11x | 57.4 | 0.00 |
| `sparse_topk.fp32.r4.kv1m.topk2048` | 0.4372 | torch | 0.2026 | 0.46x | 38.5 | 0.00 |
| `lightning_indexer.bf16.r1.kv64k` | 0.0358 | torch | 0.3581 | 10.01x | 476.6 | 0.00 |
| `lightning_indexer.bf16.r1.kv1m` | 0.6578 | torch | 7.6332 | 11.60x | 414.5 | 0.00 |
| `lightning_indexer.bf16.r16.kv64k` | 0.5033 | torch | 5.6181 | 11.16x | 42.2 | 0.00 |
| `sparse_mla.bf16.h128.topk512` | 0.0489 | torch | 0.1139 | 2.33x | 17.8 | 0.00 |
| `sparse_mla.bf16.h128.topk2048` | 0.1832 | torch | 0.1172 | 0.64x | 14.4 | 0.00 |
| `compressed_attention.bf16.csa.t1.h128.pool4k.topk512.w512.sink` | 0.0918 | torch | 0.2751 | 3.00x | 14.3 | 2.92 |
| `compressed_attention.bf16.csa.t1.h128.pool4k.topk2048.w512.sink` | 0.2218 | torch | 0.2783 | 1.26x | 13.0 | 3.03 |
| `compressed_attention.bf16.csa.t1.h128.pool64k.topk2048.w512.sink` | 0.2218 | torch | 0.2777 | 1.25x | 13.0 | 3.03 |
| `compressed_attention.bf16.csa.t16.h128.pool4k.topk2048.w512.sink` | 0.5732 | torch | 1.3128 | 2.29x | 80.5 | 18.73 |
| `compressed_attention.bf16.csa.t16.h128.pool64k.topk2048.w512.sink` | 0.5998 | torch | 1.4001 | 2.33x | 76.9 | 17.90 |
| `compressed_attention.bf16.hca.t1.h128.dense512.w512.sink` | 0.0935 | torch | 0.2549 | 2.73x | 14.0 | 2.87 |
| `compressed_attention.bf16.hca.t16.h128.dense512.w512.sink` | 0.2382 | torch | 0.5108 | 2.14x | 88.1 | 18.03 |

### D.5 B15 — hyper-connection + attn_res

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `model_ops.hc_chain.bf16.t1.h7168.n4` | 0.0790 | torch | 0.3502 | 4.44x | 38.3 | 0.02 |
| `model_ops.mega_mhc.bf16.t1.h7168.n4` | 0.0786 | torch | 0.3508 | 4.46x | 38.5 | 0.02 |
| `model_ops.hc_chain.bf16.t64.h7168.n4` | 0.1650 | torch | 0.3494 | 2.12x | 116.9 | 0.53 |
| `model_ops.mega_mhc.bf16.t64.h7168.n4` | 0.1706 | torch | 0.3500 | 2.05x | 113.1 | 0.52 |
| `model_ops.hc_chain.bf16.t1024.h7168.n4` | 1.0362 | torch | 5.2231 | 5.04x | 257.9 | 1.36 |
| `model_ops.mega_mhc.bf16.t1024.h7168.n4` | 1.0547 | torch | 5.2233 | 4.95x | 253.3 | 1.34 |
| `model_ops.attn_res.bf16.t1.h7168.s1` | 0.0047 | torch | 0.1452 | 31.17x | 15.4 | 0.01 |
| `model_ops.attn_res.bf16.t1.h7168.s1.norm` | 0.0067 | torch | 0.2191 | 32.87x | 14.1 | 0.01 |
| `model_ops.attn_res.bf16.t1.h7168.s4` | 0.0073 | torch | 0.1539 | 21.15x | 15.8 | 0.02 |
| `model_ops.attn_res.bf16.t1.h7168.s4.norm` | 0.0093 | torch | 0.2177 | 23.43x | 14.8 | 0.02 |
| `model_ops.attn_res.bf16.t1.h7168.s8` | 0.0111 | torch | 0.1550 | 14.02x | 15.6 | 0.02 |
| `model_ops.attn_res.bf16.t1.h7168.s8.norm` | 0.0128 | torch | 0.2209 | 17.26x | 15.2 | 0.02 |
| `model_ops.attn_res.bf16.t16.h7168.s1` | 0.0047 | torch | 0.1562 | 33.17x | 243.5 | 0.19 |
| `model_ops.attn_res.bf16.t16.h7168.s1.norm` | 0.0068 | torch | 0.2220 | 32.58x | 220.9 | 0.13 |
| `model_ops.attn_res.bf16.t16.h7168.s4` | 0.0078 | torch | 0.1566 | 20.11x | 235.7 | 0.29 |
| `model_ops.attn_res.bf16.t16.h7168.s4.norm` | 0.0097 | torch | 0.2215 | 22.77x | 225.5 | 0.24 |
| `model_ops.attn_res.bf16.t16.h7168.s8` | 0.0119 | torch | 0.1564 | 13.16x | 231.7 | 0.35 |
| `model_ops.attn_res.bf16.t16.h7168.s8.norm` | 0.0136 | torch | 0.2217 | 16.28x | 228.4 | 0.30 |
| `model_ops.attn_res.bf16.t256.h7168.s1` | 0.0145 | torch | 0.1723 | 11.85x | 1262.0 | 1.01 |
| `model_ops.attn_res.bf16.t256.h7168.s1.norm` | 0.0239 | torch | 0.2351 | 9.85x | 1009.4 | 0.62 |
| `model_ops.attn_res.bf16.t256.h7168.s4` | 0.0271 | torch | 0.9591 | 35.34x | 1082.0 | 1.35 |
| `model_ops.attn_res.bf16.t256.h7168.s4.norm` | 0.0370 | torch | 1.0002 | 27.06x | 949.4 | 0.99 |
| `model_ops.attn_res.bf16.t256.h7168.s8` | 0.0685 | torch | 1.8422 | 26.91x | 643.4 | 0.96 |
| `model_ops.attn_res.bf16.t256.h7168.s8.norm` | 0.1248 | torch | 1.8802 | 15.07x | 398.9 | 0.53 |

### D.6 B17 — low-bit KV + moe_w2

| case | kops ms | baseline | baseline ms | ratio | kops GB/s | kops TFLOP/s |
|---|---:|---|---:|---:|---:|---:|
| `lowbit.store.bf16.t2048.h8.d128.turbo4` | 0.0574 | — | — | — | 184.8 | 0.00 |
| `lowbit.store.bf16.t2048.h8.d128.turbo3` | 0.0604 | — | — | — | 171.4 | 0.00 |
| `lowbit.store.bf16.t4096.h8.d128.turbo4` | 0.1124 | attention.rs | 0.0509 | 0.45x | 188.8 | 0.00 |
| `lowbit.store.bf16.t4096.h8.d128.turbo3` | 0.1188 | attention.rs | 0.0451 | 0.38x | 174.3 | 0.00 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo4` | 0.6804 | attention.rs | 0.6923 | 1.02x | 52.6 | 0.00 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.turbo3` | 0.5563 | attention.rs | 0.4843 | 0.87x | 56.8 | 0.00 |
| `lowbit.decode.bf16.d128.b8.kv4096.h32x8.prod3` | 0.5702 | — | — | — | 57.2 | 0.00 |
| `lowbit.prefill.bf16.d128.t512.kv4096.h32x8.turbo4` | 2.9694 | attention.rs | 2.5425 | 0.86x | 0.0 | 10.85 |
| `lowbit.prefill.bf16.d128.t512.kv4096.h32x8.prod3` | 5.7487 | — | — | — | 0.0 | 5.60 |
| `moe.w2.pack.bf16.e8.n2048.k4096.g32` | 0.4273 | attention.rs | 0.2260 | 0.53x | 373.0 | 0.00 |
| `moe.w2.gemm.bf16.e8.n2048.k4096.slots2048` | 2.0442 | — | — | — | 24.6 | 16.81 |

### D.7 B16 — megakernel

| case | mega ms | mega graph ms | modular ms | modular graph ms | mega/modular |
|---|---:|---:|---:|---:|---:|
| `mega.mlp.m8.h1024.i2048` | 0.1376 | 0.1420 | 0.0208 | 0.0226 | 6.6x |
| `mega.attn.m1.h1024.hq8.hkv2.d128.kv64` | 0.5800 | 0.5812 | 0.0254 | 0.0247 | 22.9x |
| `mega.moe.m8.h512.i512.e8.k2` | 0.3958 | 0.3973 | — | — | — |
| `mega.decoder.m1.h512.i1024` | 0.5100 | 0.5105 | 0.0323 | 0.0308 | 15.8x |
