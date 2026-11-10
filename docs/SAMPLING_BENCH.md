# Sampling benchmark (B1/B6/B10)

Trạng thái: **cập nhật 2026-11-10 20:02** trên NVIDIA RTX 5060 Ti
(SM120a, 36 SM, trần 448 GB/s), PyTorch 2.14.1+cu130, CUDA 13.4.92,
preset `bench-sm120a`. Kết quả là **diagnostic-only** (cây làm việc dirty,
chưa chạy lại toàn bộ suite ngoài `kops_test_sampling`).

Tài liệu này thay thế các hàng `sampling.*` của
[`BENCH_RTX5060.md`](BENCH_RTX5060.md) (ảnh chụp 18:4x). Các family khác vẫn
dùng số trong báo cáo đó.

## 1. Hai đường, hai báo cáo

`bench_sampling` phát ra 136 case thuộc ba nhóm không được gộp geomean:

| nhóm | case | ý nghĩa |
|---|---:|---|
| `sampling.fused.*` | 69 | sampler một launch (`sample_fused`) |
| `sampling.sample.*` | 33 | entry point công khai `sample()` |
| ops lẻ (`softmax`, `argmax`, `bincount`, `spec.*`, …) | 34 | logits processor và speculative |

Baseline là **chuỗi eager** mà một model implementation chạy
(`softmax → top-k → sort/cumsum → multinomial`), không phải `torch.multinomial`
trên full vocab. Dùng `--groups 2` để in bảng theo nhóm:

```bash
python benchmark/compare/compare_results.py out/rtx5060/kops/sampling.json \
    out/rtx5060/torch/sampling.json --groups 2 \
    --json out/rtx5060/compare/sampling.json
python tools/dev/report_sampling.py out/rtx5060/compare/sampling.json \
    --write out/rtx5060/report-sampling.md
```

## 2. Kết quả

| chỉ số | trước (18:4x) | hiện tại (20:02) |
|---|---:|---:|
| geomean toàn bộ 136 case | 3.63x | **6.40x** |
| case thua | 27 | **0** |
| `sampling.fused.*` | 4.67x, 4 thua | **5.47x, 0 thua** |
| `sampling.sample.*` | 0.59x, 21 thua | **4.09x, 0 thua** |
| median tỉ lệ | 3.80x | **5.37x** |
| case thấp nhất | 0.07x | **1.02x** |
| case cao nhất | 2376.6x | 2475.5x |

`sample()` và `sample_fused()` giờ cho cùng kết quả trên các hàng
greedy/unfiltered/top-p-only/min-p (cùng body); top-k vẫn là đường probability
riêng vì fused top-k giữ cả tie group, khác contract của `sample()`.

## 3. Chín case từng thua

| case | trước | hiện tại | eager ms | kops ms |
|---|---:|---:|---:|---:|
| `fused.p0.99.b16` | 0.42x | **1.16x** | 0.1771 | 0.1532 |
| `fused.p0.99.b32` | 0.50x | **1.63x** | 0.3553 | 0.2181 |
| `fused.p0.99.b64` | 0.88x | **3.21x** | 1.1924 | 0.3717 |
| `sample.p0.99.b32` | 0.51x | **1.62x** | 0.3539 | 0.2190 |
| `sample.k50.b1` | 0.60x | **2.09x** | 0.2882 | 0.1381 |
| `sample.k50p0.9.b1` | 0.51x | **1.38x** | 0.2964 | 0.2151 |
| `sample.minp0.05.b1` | 0.70x | **6.33x** | 0.2225 | 0.0352 |
| `gumbel.k50.b32` | 0.82x | **1.11x** | 0.1997 | 0.1801 |
| `argmax.fp32.b128` | 0.96x | **1.02x** | 0.1754 | 0.1719 |

Case từng hòa `fused.p0.95.b16` (0.98x) hiện đạt **1.49x**
(0.1131 ms so với 0.1684 ms).

## 4. Đã thay đổi gì

- **`sample()` fast path** (`csrc/sampling/sampling.cu`): top-k tắt → chạy
  body `sample_fused()`. Auto split-row dùng exchange area + barrier carve từ
  vùng fp32 row không còn dùng, zero counter mỗi call; nếu không đủ chỗ thì
  `splits=1`. Top-k vẫn ở đường legacy.
- **P-only p ≥ 0.95** (`csrc/sampling/fused_sampler.cu`): dựng **exact mass40
  histogram** sau khi biết row max. Cut nằm gọn trong một bin nên candidate
  range không cần 2 bin slack; trước đây 3 bin biên của p0.99 chứa ~3145 token
  > `kCand=2048` nên rơi vào exact fallback.
- **Top-k legacy**: `topk_by_candidates` (1 histogram 2048 bin + 1 gather +
  rank) thay 4 pass radix; bỏ `row_max` khi không có min-p; dùng luôn `ksum`
  thay pass z khi không có top-p/min-p.
- **Gumbel top-k**: candidate histogram trên logits thay 4 pass radix; bỏ pass
  `row_max` khi không cần.
- **Argmax**: 1024 thread/block cho vocab ≥ 64k (390 GB/s).

Chi tiết test: `tests/unit/test_sampling.cc` thêm
`test_sample_fast_paths()` (greedy/+inf/all -inf, top-p distribution,
min-p membership, split carve ở V=151936) và `test_gumbel_topk()` (candidate
path + fallback tie group 2048 token).

## 5. Tái lập

```bash
cmake --preset bench-sm120a
cmake --build --preset bench-sm120a --target bench_sampling -j 8

build/bench-sm120a/benchmark/bench_sampling --rounds 3 \
    --json out/rtx5060/kops/sampling.json
build/bench-sm120a/benchmark/bench_sampling --dump-cases \
    > out/rtx5060/cases/sampling.json
python benchmark/compare/torch_baseline.py \
    --cases out/rtx5060/cases/sampling.json \
    --rounds 3 --json out/rtx5060/torch/sampling.json
python benchmark/compare/compare_results.py \
    out/rtx5060/kops/sampling.json out/rtx5060/torch/sampling.json \
    --groups 2 --json out/rtx5060/compare/sampling.json
```

`bench_sampling` dùng hot-L2 mặc định; mỗi mẫu là một nhóm 20 launch, median
trên 4 nhóm × 3 round (80 launch mỗi case mỗi round). PyTorch baseline chạy cùng
shape/dtype và `torch.inference_mode()`.

## 6. Hạn chế

- Một GPU consumer (RTX 5060 Ti); không suy ra cho H100/B200.
- Hot-L2: các case nhỏ có GB/s vượt trần DRAM; chỉ so tỉ lệ, không so GB/s
  tuyệt đối với baseline cold.
- Baseline eager không có `torch.compile`; nếu so với graph/compiled engine,
  tỉ lệ sẽ khác.
- `sample()` top-k vẫn giữ tie-group semantics của contract (khác fused);
  các case `sample.k50*` đo đường legacy đã tối ưu, không phải fused.
- Chưa chạy lại toàn bộ 26 test sau cập nhật; `kops_test_sampling` PASS
  (7.7 s) và các check header/ABI/roadmap PASS.
