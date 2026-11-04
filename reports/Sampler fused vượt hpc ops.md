# Một lần đọc logits, đánh bại hpc-ops

hpc-ops không đánh bại được bằng cách viết một kernel giống nó nhưng nhanh hơn: cấu trúc của nó đã chạm sàn traffic (1.04× mức tối thiểu) nhưng lại tiêu phần lớn thời gian vào một `cub::BlockRadixSort` chạy lại **bên trong** vòng load, nên toàn bộ dư địa nằm ở chỗ bỏ sort chứ không phải ở chỗ đọc ít byte hơn. Ngược lại, `sample()` hiện tại của kernel-ops đọc logits 2 lần rồi *materialize* một ma trận xác suất fp32 và đọc lại nó 7 lần, tức **36 B cho mỗi phần tử vocab của mỗi dòng** so với sàn 2 B — đo trên RTX 3050 cho thấy mô hình 7-pass khớp tới 93 % bandwidth đỉnh, nên đây là một khoản nợ traffic thật chứ không phải suy đoán. Thiết kế đề xuất trong báo cáo này — một launch, một CTA mỗi dòng khi B ≥ số SM, split-row khi B nhỏ, pass 1 dựng histogram 8-bit trên monotone key + online softmax, pass 2 gom candidate vào shared memory rồi rút thăm bằng exponential race — đưa traffic xuống **4 B/phần tử (lọc) và 2 B/phần tử (không lọc)**, giữ nguyên ngữ nghĩa exact full-vocab, và ước tính **2–3 µs ở B=1, ~4–5 µs ở B=32, ~15 µs ở B=128, ~45–90 µs ở B=512 trên H100**. Điều còn thiếu để biến "tốt hơn hpc-ops" thành một khẳng định có thể kiểm chứng là **một con số đo được của hpc-ops tại B=512**: nếu nó chạy ~40 µs ở đó thì thiết kế này chỉ thắng ở B ≤ 128 và thắng bằng ngữ nghĩa, không thắng bằng latency.

## Kết luận cho bạn

- **Đừng lấy `3rd/attention.rs/src/kernels/src/fast_topk.cu` làm mẫu.** Đường "bitonic" của nó thực chất là vòng argmax + invalidate `O(K·V)` (`3rd/attention.rs/src/kernels/src/fast_topk.cu:91-125`), còn đường radix đọc lại cả hàng tới **33 lần** vì không compact candidate (`fast_topk.cu:169-179`, `224-232`). Ở V=128256, B=512 đó là ~8.5 GB traffic, tức ~2.5 ms trên H100 — tệ hơn thiết kế SGLang khoảng 16×. Nó còn trả index **không sắp thứ tự**, tie không deterministic, và **không có caller nào** trong cây Rust.
- **Hai mẫu radix đáng copy là SGLang `topk.cu` và TensorRT-LLM `indexerTopK.cu`.** SGLang làm đúng 2 pass toàn hàng rồi 4 vòng 8-bit *trên shared memory* với ≤4096 candidate (`3rd/sglang/sgl-kernel/csrc/elementwise/topk.cu:96`, `186-251`); TRT-LLM chia 11/11/11/10 bit với 2048 bin và tuyên bố rõ "boundary bin ≤ 2048 thì kết thúc trong smem" (`3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/indexerTopK.cu:41-46`, `393`).
- **`csrc/sampling` hiện tại mạnh về ngữ nghĩa, yếu về dataflow.** Exact full-vocab, tie-break theo id nhỏ nhất, per-row params, Philox counter-based — tất cả đều đúng và nên giữ. Nhưng 4 launch, 36 B/phần tử, **247.5 MB workspace ở B=512**, và nhánh mặc định (top-p only) lại là nhánh **chậm nhất** (8 pass, đo được ~35 GB/s vì barrier/atomic-bound).
- **Mục tiêu đúng không phải "nhanh hơn hpc-ops" mà là "1 launch + 2 pass + exact".** Ở B=512 thiết kế 2-pass *không* thắng về traffic so với 1-pass của hpc-ops (4 B so với 2 B/phần tử); nó thắng vì hpc-ops trả giá bằng sort + 31 wave + 2 launch. Hãy nói rõ điều đó trong mọi tài liệu so sánh.
- **Hãy làm split-row theo kiểu "shard-resident"** (mục 7): khi B < số SM, chia hàng cho S CTA và **cache nguyên shard trong shared memory** ở pass 1. Với V=120832, S=32 thì shard chỉ 7.4 KiB, nên pass 2 đọc từ smem → **1 pass DRAM** ngay cả khi có lọc. Đây là điểm mà cả bốn note chưa khai thác và là lý do thiết kế này không thua B=1.
- **RNG phải là hàm thuần của `(seed, request, step, token_id)`**, với `step` đọc từ device tensor. hpc-ops dùng host atomic offset (`3rd/hpc-ops/src/sampler/sampler_rng.cuh:42-52`), nên offset bị **đóng băng vào graph** lúc capture và mỗi replay lặp lại đúng vector Gumbel cũ.
- **Thứ tự triển khai: M1 (1-pass, không lọc) trước, M2 (histogram + threshold) sau.** M1 đã là một sản phẩm: nó thay `gumbel_max_sample` + `sample` cho cấu hình greedy/temperature, bỏ 2 kernel và 5 pass, và có thể ship trong một ABI bump.
- **Nếu chỉ làm một việc:** thay `softmax(probs) → top_k_top_p_sampling_kernel` bằng **một kernel hai pass không materialize fp32 probs**, trong đó pass 1 vừa dựng histogram 8-bit vừa chạy online softmax, pass 2 gom candidate vào smem và rút thăm bằng exponential race. Chỉ riêng việc bỏ ma trận `B×V×4` đã cắt 36 B → 4 B/phần tử và 247 MB → ~10 MB workspace, tức **~9× traffic** và **~25× workspace**, trước khi nói tới chuyện bỏ 3 launch.

## Bối cảnh, phạm vi, giả định

"3rd" trong báo cáo này là cây vendored `3rd/` của repo tại `C:\Users\Tuand\Downloads\kernel-ops-headers\kernel-ops`, ở HEAD `0aa87a2` với `KOPS_ABI_VERSION 5` (`csrc/common/macros.cuh:42`). "fast_topk" được hiểu là `3rd/attention.rs/src/kernels/src/fast_topk.cu` cộng với các kernel top-k production cùng loại ở SGLang, vLLM, TensorRT-LLM và FlashInfer; tôi không coi riêng file của attention.rs là đại diện cho "fast top-k" vì nó là một kernel chưa được dùng, viết cho bài toán indexer chứ không phải sampler.

Workload mục tiêu: decode sampling của LLM, vocab 120832 / 128256 / 151936, batch 1..512, logits BF16 hoặc FP32, H100 (SM90) và B200 (SM100), decode loop được capture bằng CUDA Graph, có thể bật speculative decoding. "Tốt hơn hpc-ops" được định nghĩa falsifiable: latency thấp hơn ở B ≥ 8, **không regression ở B=1**, ngữ nghĩa exact full-vocab, hỗ trợ K > 64, không phụ thuộc vocab, replay-safe, deterministic, không allocate trong replay path. Ở bất kỳ chỗ nào khẳng định dựa vào một baseline hpc-ops chưa đo, báo cáo nói rõ.

Ba nhãn bằng chứng được dùng xuyên suốt: **[measured]** là số ai đó đã chạy và công bố (kể cả đo trong repo này trên RTX 3050); **[derived]** là số học từ một đại lượng đã cite, không đo; **[estimate]** là mô hình cần benchmark để xác nhận. Không có số [estimate] nào được trình bày như [measured].

Một điểm về chất lượng nguồn: **không tồn tại số đo sampler nào của hpc-ops trong cây** — không CSV, không JSONL, không `.nsys-rep`. README của nó chỉ có một dòng "Up to 8.5×" so với "vLLM-style PyTorch, FlashInfer" (`3rd/hpc-ops/README.md:169-181`), và baseline của benchmark đó là một pipeline torch ngây thơ có `work.sort(dim=-1)` trên 120832 fp32 giá trị mỗi dòng (`3rd/hpc-ops/benchmark/sampler/benchmark_sampler.py:134-170`). Con số 8.5× vì thế là phát biểu về **baseline chậm**, không phải về khoảng cách tới roofline.

## Giải phẫu hpc-ops: pipeline đúng, nhưng sort nằm trong vòng lặp

### K1 và K2 làm gì

hpc-ops có hai kernel. **K1** (`fused_scan_topk_kernel`) phủ toàn bộ vocab đúng **một lần**: load logits, áp repetition-penalty từ bitmask, nhân `1/T`, vừa chạy online max/sum vừa duy trì một "rolling top-K" trong register, rồi ghi `N` danh sách top-K cục bộ ra global memory. Grid là `(N, B)`, block **1024 thread**, template `<DType, kVocabSize, kMaxTopK, kSoftmaxPolicy>` (`3rd/hpc-ops/src/sampler/fused_sampler.cu:161-172`, `654-660`). **K2** (`stage2_kernel`) gộp `N` danh sách bằng một `BlockRadixSort` thứ hai, rồi **chỉ warp 0** làm softmax/top-p/Gumbel-max/writeback (`fused_sampler.cu:354-401`, `403-579`); 31 trong 32 warp thoát ngay sau `if (tid >= 32) return;`.

Cấu trúc then chốt: K1 chia thread thành **keeper** và **loader**. `kKeeperThreads = kMaxTopK/4` (8 với K=32, 16 với K=64), `kLoaderThreads = 1024 - kKeeperThreads` (1016 hoặc 1008); keeper giữ 4 giá trị top trong register (`fused_sampler.cu:176-180`, `203-214`). Sau mỗi vòng load, **cả 1024 thread sort lại 4096 item** để keeper nhận lại top-K đang giữ:

```text
__syncthreads(); BlockRadixSort.SortDescending(...); __syncthreads();
```

nằm **bên trong** `for (iloop < num_load_loops)` (`fused_sampler.cu:225-263`). Đó là toàn bộ câu chuyện hiệu năng: mỗi vòng load xử lý `N × 1016 × 4` phần tử nhưng lại sort 4096 item, tức **≈1.008 radix item cho mỗi phần tử vocab** (4096/4064 với K=32; 4096/4032 với K=64). Với `RADIX_BITS = 4` mặc định của CUB, một key float 32-bit tốn **8 radix pass**, mỗi pass là một block-wide rank + scatter trên 4096 cặp (key, value).

Hình học này có một tính chất phản trực giác: **tổng số sort item không phụ thuộc N**. Với V=120832, `N=32` cho 1 vòng load và 131072 sort item mỗi dòng; `N=8` cho 4 vòng load và `8 × 4096 × 4 = 131072` — y hệt. Nghĩa là `N` chỉ đổi *độ song song*, không đổi *tổng công việc sort*. Ở B=512, K=32, tổng là **67.1 M sort item**, và vì K1 được thiết kế 1 block/SM, 4096 block đó chạy **31 wave** trên H100 132 SM (`fused_sampler.cu:610-627`).

`N` được chọn `clamp(floor(SMs/B), 8, Nmax)` với `Nmax = min(1024/K, 32)` (`fused_sampler.cu:42-46`) — tức **32 với K=32 và 16 với K=64**. Hệ quả: **xin K lớn hơn làm latency batch nhỏ tệ hơn** (B=1 chỉ dùng 12 % số SM của H100 thay vì 24 %). Đây là một inversion mà không sampler nào nên có, và nó là hệ quả trực tiếp của việc buộc K2 phải là một block 1024 thread.

### Mô hình pass/byte: nó nhanh ở đâu

Traffic mỗi call (BF16, V=120832, K=32, Nmax=32) ước tính từ code:

| Hạng mục | Byte/dòng | B=1 | B=512 |
|---|---|---|---|
| logits đọc (1 lần) | 241,664 | 236 KiB | 118 MiB |
| bitmask penalty (nếu bật rp) | 15,104 | 14.8 KiB | 7.2 MiB |
| `mid_logits`+`mid_tokens` ghi | 8,192 | 8 KiB | 4.0 MiB |
| K2 đọc lại | 8,192 | 8 KiB | 4.0 MiB |
| partial max/sum (chỉ BEFORE_TOPK) | 256 | 0.25 KiB | 0.13 MiB |
| token ra | 4 | 4 B | 2 KiB |
| **tổng** | **~273 KB** | **~267 KiB** | **~134 MiB (~140 MB)** |

Ở 3.35 TB/s, 140 MB là **41.8 µs** [derived]. Đó là sàn tuyệt đối của một thiết kế đọc một lần. Mọi thứ vượt trên con số đó là sort, sync, launch và sóng không đủ lấp. Đây là con số quan trọng nhất của cả báo cáo: **hpc-ops đã đúng về traffic và sai về compute**, nên đối thủ của nó không phải là một kernel đọc ít byte hơn mà là một kernel *làm ít việc hơn trên mỗi byte*.

### "Sort-bound" là estimate, không phải measurement — và khoảng của nó rộng

Đây là chỗ hai note va nhau và cần phân xử. Note khảo sát nói hpc-ops "sort-bound" và cho B=512 khoảng **200–350 µs**; note thiết kế cho **120–370 µs** và tự cảnh báo rằng nếu đo được ~40 µs thì luận điểm sụp. Cả hai đều là **mô hình**, không phải phép đo: trong cây không có `ncu`, không có `-Xptxas -v`, không có register count, không có achieved occupancy.

Bằng chứng cấu trúc thì chắc chắn và đủ để nói **sort chiếm phần lớn thời gian nếu** giả định "~1.6–2.7 µs cho một sort 4096 item trên một SM" là đúng:

- 1.008 radix item/element, 8 pass/item → **~8 lượt register/smem traffic cho mỗi phần tử vocab**, so với 1 lượt max.
- Sort chạy lại mỗi vòng load, không phải mỗi launch (`fused_sampler.cu:260-263`).
- B=512 = 67.1 M sort item trên 4096 block / 31 wave.

Cái chưa biết là **hằng số thời gian**. Nếu một sort 4096 item tốn 1.0 µs thì B=512 là `31 wave × 4 sort × 1.0 µs ≈ 124 µs`; nếu 3.0 µs thì 372 µs; nếu chỉ 0.35 µs thì 43 µs và thiết kế gần như bandwidth-bound. Vì vậy phát biểu đúng là: **hpc-ops có 3–9× dư địa trên sàn byte, và dư địa đó gần như chắc chắn là sort, nhưng biên độ chưa được đo**. Con số 8.5× trong README không giúp thu hẹp khoảng này vì baseline của nó tự sort 120k fp32 mỗi dòng.

Ở chiều ngược lại, đường `fused_sampler_temperature.cu` là bằng chứng nội bộ rằng nhóm tác giả biết cách làm đúng: 256 thread/block, `grid=(ceil(SM/B), B)` nên B=1 dùng **toàn bộ** SM, **một** launch, không `cudaMallocAsync`, RNG và noise fuse vào cùng một pass, và một block-arrival counter thay cho kernel thứ hai (`fused_sampler_temperature.cu:147-217`, `268-326`, `435-445`). Đó chính là hình dạng mà thiết kế ở mục 7 nhắm tới — chỉ khác là nó không lọc.

### Ngữ nghĩa: bounded top-32/64, và năm cái bẫy

hpc-ops **luôn** thu vocab về `max_topk` (32 hoặc 64) trước khi rút thăm. Header nói thẳng: "`topk=0` (or unset) does NOT mean full-vocab sampling — it only means 'do not tighten below max_topk'. Low-probability tokens outside the top-`max_topk` are never drawn" (`3rd/hpc-ops/src/sampler/sampler.h:13-20`). Năm hệ quả:

1. **`topk=0` không phải full-vocab.** Ở temperature cao hoặc logits phẳng, sampler trả về gần như uniform trên 32 token thay vì gần uniform trên 128k. Một test chỉ kiểm tra "token nằm trong top-20" sẽ không bắt được (`3rd/hpc-ops/tests/test_sampler.py:326-337`).
2. **Top-p chỉ chạy trên ≤64 candidate** (`fused_sampler.cu:453-458`). Với `kSoftmaxPolicy = BEFORE_TOPK`, cumulative mass là mass **chưa chuẩn hoá của toàn vocab** giới hạn trong top-K, nên nếu top-32 giữ dưới `topp` thì **top-p không bao giờ cắt gì**. Với logits N(0,1) và V=128k, top-32 giữ ~0.83 % mass [derived: `32·e^4 ≈ 1748` so với `128256·1.6487 ≈ 211,470`], nên `topp=0.9` biến thành "giữ tất cả 32". Đây là **divergence ngữ nghĩa** so với vLLM/HF, nơi top-p chuẩn hoá lại trên phần top-k còn sống. Có thể đây là bug, có thể là lựa chọn có chủ ý; không có issue/PR nào trong cây nói rõ.
3. **`topk > max_topk` bị clamp im lặng** bằng `min(user_k, kMaxTopK)` trong K2 (`fused_sampler.cu:313-327`), không có lỗi host. Test reference cũng mirror đúng cái clamp đó (`tests/test_sampler.py:119-122`), nên test không phát hiện được.
4. **Không có min-p, typical-p, XTC, DRY, no-repeat-ngram.** Grep trên `src/sampler/*` và `hpc/sampler.py` không trả về gì. Penalty chỉ là **1 bit "đã thấy"** nhân `×1/rp` hoặc `×rp` (`fused_sampler.cu:94-106`) — không phân biệt frequency/presence, không có cửa sổ, không có count.
5. **Vocab là hằng số compile-time**: dispatcher chỉ nhận `vocab_size == 120832`, còn lại `throw` (`fused_sampler.cu:757-766`; đường temperature y hệt ở `fused_sampler_temperature.cu:538-547`). 128256 (Llama-3) và 151936 (Qwen) **không chạy được**.

Hai edge case nữa đáng ghi vào contract của mình: hàng toàn `-inf` hoặc NaN rơi về **token 0** (`tok = (best_tok < 0) ? 0 : best_tok`, `fused_sampler.cu:560-562`), không phải `-1` như kernel-ops; và `topp=0` là disable sạch, nhưng truyền một *tensor* top-p toàn 0 vẫn instantiate `kHasTopP=true` và chạy scan.

### RNG và CUDA Graph: hai lỗi kiến trúc

Offset Philox lấy từ một **`std::atomic` host-side, tăng một lần mỗi launch** (`3rd/hpc-ops/src/sampler/sampler_rng.cuh:30-52`), rồi `curand_init(seed, sequence, offset)` với `sequence = ibatch*32 + ilane` (`fused_sampler.cu:478-489`). Hai hệ quả, cả hai đều chí mạng với decode loop:

- **Reproducibility cross-process là bất khả**: kết quả phụ thuộc số lần sampler đã được gọi trước đó trong process. Chỉ tái lập được nếu *chuỗi* launch và batch size giống hệt.
- **Graph replay đóng băng offset**: lúc `cudaStreamBeginCapture`, offset được đọc một lần và nhúng vào kernel node. Mỗi `cudaGraphLaunch` phát lại **đúng vector Gumbel cũ** cho cùng (row, rank). Nếu logits lặp lại, token lặp lại; tổng quát hơn, noise stream không bao giờ tiến.

Đường temperature còn tệ hơn ở mặt concurrency: nó dùng `static WorkspaceCache g_ws[16]` per-device cộng một arrival counter, và header tự ghi "not safe to call concurrently on multiple streams of the same device" (`sampler.h:38-39`, `fused_sampler_temperature.cu:352-372`). Failure mode không phải crash mà là **token cũ**: nếu `atomicAdd` của hai stream xen nhau, không block nào thấy `s_prev == N-1`, final reduce không chạy, `token_ids_out[ibatch]` giữ giá trị trước đó. Đó đúng là lớp bug mà `csrc/sampling` phải loại trừ.

Phần bù lại: đường fused (2 kernel) **không** có static device state, mọi scratch là `cudaMallocAsync` per-call, nên nó stream-safe — nhưng trả giá 4 alloc + 4 free mỗi call (`fused_sampler.cu:585-607`, `648-651`), tức 8 node thừa trong graph và địa chỉ bị đóng băng. Và đường external-Gumbel (truyền buffer `[B,V]` fp32) là một ý tưởng **rất tốt** để test bit-exact: nó cho phép so sánh kernel với reference khi noise cố định (`sampler.h:32-34`, `tests/test_sampler.py:443-482`). Nên copy ý tưởng đó.

## Fast top-k / top-p: bốn trường phái và một bãi mìn

### attention.rs `fast_topk`: bài học phản diện

Kernel này chia hai đường theo `seq_len`. Với `seq_len ≤ 4096` nó dùng `fast_topk_bitonic_kernel` — tên là bitonic nhưng thân hàm là **argmax rồi invalidate**, lặp `topk` lần: mỗi route quét toàn bộ smem, reduce 8 bước, winner ghi index rồi poison slot bằng `-FLT_MAX` (`3rd/attention.rs/src/kernels/src/fast_topk.cu:91-125`). Đó là `O(topk · seq_len)` comparison cộng `topk × 9` `__syncthreads()`. Với `topk=256, seq_len=4096` là 256 lượt quét mỗi block.

Với `seq_len > 4096` nó chuyển sang radix 1 bit: duyệt bit 31 → 0, mỗi bit **quét lại toàn bộ hàng từ global memory** và chỉ test `(sv & prefix) == prefix_val`, **không có candidate list, không compaction** (`fast_topk.cu:169-179`). `remaining_k` chỉ giảm khi rơi vào nhánh 0, nên trường hợp thường là **đủ 32 vòng**; cộng một vòng collect cuối là **33 lần đọc toàn hàng**. Shared memory cho đường này chỉ 40 byte (`fast_topk.cu:275`) — nó tiết kiệm smem bằng cách trả bằng băng thông, đúng ngược lại với điều cần làm ở V=128K.

Collect dùng `atomicAdd` trên counter trong smem và chỉ ghi nếu `pos < topk`, nên khi có nhiều giá trị bằng nhau đúng tại ngưỡng thì **tập được chọn phụ thuộc thứ tự thread thắng race** (`fast_topk.cu:224-232`). Output **không sắp thứ tự**, pad bằng `-1`, chỉ nhận F32, và toàn bộ file **không có test, không có benchmark, không có caller Rust** — `fast_topk_select` chỉ xuất hiện trong `ffi.rs:4676-4683`. Cái MoE top-k mà attention.rs thực sự dùng là symbol khác (`moe/topk_softmax_kernels.cu:629`).

Kết luận sử dụng: **đọc để tránh, không đọc để copy**. Giá trị duy nhất của nó là một phản ví dụ rất rõ về "radix select không compaction thì tệ hơn radix select có compaction 16×".

### Bốn trường phái production

**SGLang `topk.cu`** là thiết kế gọn nhất và là mẫu tốt nhất cho pass 1 của mình. Nó histogram 256 bin trên key suy từ fp16 (giữ 8 bit cao), suffix-sum ngược để tìm bin ranh giới, rồi **một pass collect làm hai việc**: index trên ngưỡng ghi thẳng ra output, index trong bin ranh giới vừa được stage vào `s_input_idx[0]` vừa atomic-add vào histogram kế tiếp (`3rd/sglang/sgl-kernel/csrc/elementwise/topk.cu:110-181`). Sau đó 4 vòng 8-bit nữa chỉ chạy trên ≤4096 candidate trong smem (`topk.cu:186-251`). Tổng: **≈2·length + ≤16K gather**, với 32 KiB smem và 1024 thread. Hạn chế: `TopK = 2048` hard-wired, chỉ F32, và test chấp nhận tới 5 index sai (`tests/test_topk.py:129`) vì có nhánh `min(s_num_input, SMEM_INPUT_SIZE)` clip khi bin quá đông.

**TensorRT-LLM `indexerTopK.cu`** là bản "nhiều bin hơn, ít pass hơn": 2048 bin, chia **11/11/11/10 bit** trên fp32 sortable key, 512 thread/block, `kNumFinalItems = 2048`, và dispatch theo độ rộng hàng: `< 12288` dùng insertion sort, `< 200K` dùng radix, `≥ 200K` chỉ có đường fp32 multi-CTA (`3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/indexerTopK.cu:41-46`, `903-913`, `393`). Điểm đáng copy: **bin rộng 11 bit làm bin ranh giới chỉ còn ~V/2048 ≈ 63 phần tử** ở V=128K, tức với K=32/64 thì pass tinh chỉnh gần như miễn phí.

**vLLM** đi theo hướng khác hẳn: fast path của V1 **không phải radix select** mà là **dual-pivot rejection sampling** của FlashInfer, và đường Triton mới (Qrita) là pivot search với σ-truncation. Đáng chú ý nhất là một comment đo được trong source: "at small batch sizes nearly all SMs idle and every full-vocab sweep is latency-bound (**~25us each on SM100**)" (`3rd/vllm/vllm/v1/sample/ops/topk_topp_triton.py`, split-pipeline docstring). Đó là lý do vLLM thêm **split-row pipeline** cho top-p-only. Con số 25 µs cho *một* sweep toàn vocab trên SM100 là calibration point hữu ích: nó nói rằng một thiết kế 2 pass mà không split sẽ có sàn ~50 µs ở B=1 — không chấp nhận được.

**FlashInfer** tách bạch hai contract rất rõ: `top_k` (trả *tập*) dùng radix/FilteredTopK với **128 KiB dynamic smem** và `k ≤ 2048`, có `tie_break ∈ {none, smaller, larger}` và `deterministic=True`; còn *sampling* dùng dual-pivot rejection với bound `O(log(1/ε))` và bảo đảm `Pr[out=j] = p_j/Z`. Cả hai đều có knob `dsa_graph_safe` / `VEC_SIZE=1` cho CUDA Graph.

### Bảng so sánh thuật toán

| Họ thuật toán | Pass toàn hàng | Traffic @V=128K,B=512,F32 [derived] | Smem | Regime tốt | Exact | Tie deterministic |
|---|---|---|---|---|---|---|
| Full bitonic sort | 1 đọc + `O(V log²V)` smem | 501 KiB/dòng → **không vừa smem** | >228 KiB | V ≤ 2–4K | có | có |
| Iterative argmax (attention.rs nhỏ) | K lượt quét smem | n/a (V ≤ 4096) | 8 B/elem | V ≤ 4K, K ≤ 8 | có | có |
| Radix không compaction (attention.rs lớn) | **33** | ~8.5 GB → 2.5 ms H100 | 40 B | không có | có (trừ tie) | **không** |
| Radix + smem staging 8-bit (SGLang) | **≈2** + ≤4 vòng nhỏ | ~513 MB → 153 µs | 32 KiB | V ≥ 8K, K=2048 | có | không |
| Radix + smem staging 11-bit (TRT-LLM) | 2–4 | 513 MB–1.0 GB → 150–310 µs | 8 KiB hist + out | 12K ≤ V < 200K | có | không (có variant) |
| Mass-histogram / GVR V2 | **1** classify + refine | ~1.0–1.5 pass → 77–115 µs | hist + sample window | mọi V, B ≥ SM | có | có nếu collect có thứ tự |
| Pivot search / σ-truncation (Qrita) | 1 truncate + vài sweep trên hàng đã cắt | < 1 pass nếu fit tốt | thấp | sampling | có | có (claimed) |
| Dual-pivot rejection (FlashInfer) | `O(log 1/ε)` vòng, thường 3–5 | ~3–5 pass | thấp | B nhỏ, không cần k-list | có (chứng minh) | theo draw |
| Heap / `O(n log k)` | 1 | 1 pass | k entry | K rất nhỏ, V nhỏ | có | có |

Hai kết luận rút ra từ bảng. Thứ nhất, **mọi thiết kế khả dĩ đều quy về cùng một skeleton**: một pass histogram thô → bracket bin ranh giới → kết thúc trong shared memory. Khác biệt quyết định là (i) bao nhiêu bit mỗi pass, (ii) survivor được **compact** hay chỉ bị **filter**, (iii) bước cuối có sort hay không. Thứ hai, **rejection sampling thắng khi B nhỏ và không cần tập k**, nhưng mỗi vòng là một lần đọc toàn hàng, nên nó sai ở B=512 — đúng như TRT-LLM kết luận khi họ chỉ dùng rejection cho đường "tokens-only" sau khi đã narrow support.

Một calibration bằng số cho toàn bộ ngành: TensorRT-LLM công bố reachable rate của radix CUDA production chỉ **7.7–8.3 % trung bình, 16.4–17.8 % đỉnh** của sàn "một lần đọc", trong khi GVR V2 (self-sampling + multi-threshold) đạt **39–41.6 % trung bình, 66.5–77.8 % đỉnh** trên B200, kèm **5.05× geomean** so với radix. Nghĩa là: radix production đốt ~12 pass-equivalent, còn trần của một thiết kế streaming là ~1.3 pass. Đó là khoảng cách mà thiết kế ở mục 7 nhắm vào.

Phần học thuật đóng góp đúng một ý tưởng dùng được, và nó không phải ý tưởng mà tên bài báo gợi ra. **AIR TopK** (SC'23) là "iteration-fused radix" với chiến lược thích ứng để tự cắt traffic không cần thiết, công bố **1.98–21.48×** ở B=1 và **8.01–574.78×** ở B=100 so với radix trước đó, và đã có trong RAFT. **Dr. Top-k** (SC'21) dùng delegate để lọc hơn **99 %** workload. **RadiK** (ICS'24) là bản duy nhất giữ hiệu năng khi k lớn và "không phụ thuộc batch size". Nhưng cả ba đều nhắm **vector cỡ 2^20 với batch ≤ 100** — RTop-K nói thẳng điều này trong phần khảo sát — nên **không bài nào đo ở regime V ≈ 2^17, B = 512, K = 32/64 của mình**, và không có so sánh head-to-head nào giữa AIR/RadiK/FilteredTopK trên cùng grid. Giá trị thực dụng của chúng với mình là **thuật toán**, không phải con số: fuse iteration, lọc bằng delegate, và giữ chi phí không đổi theo batch đều là những tính chất thiết kế ở mục 7 đã có sẵn.

## Hiện trạng `csrc/sampling`: ngữ nghĩa tốt, dataflow sai

### Inventory

`csrc/sampling/ops.h` khai báo 16 entry point: **13 có body** (mười hai kernel op cộng `sampling_workspace_requirements` chỉ host) và **ba entry speculative** — `rejection_sample`, `tree_verify`, `build_tree_attention_mask` — **chỉ có khai báo**, không có định nghĩa nào trong cây, nên một engine link family này và gọi chúng sẽ fail ở link time. Ba entry bị "degraded" so với chính docstring của chúng: `sample()` không chứa rejection sampling, `gumbel_max_sample()` từ chối top-p dù header quảng cáo là "generalized" (`csrc/sampling/sampling.cu:2315-2318`), và `sample()` không dùng `topk_by_candidates` — selector 3-read đã tồn tại ngay trong cùng translation unit và đã được chứng minh tương đương (`sampling.cu:1338-1343`, dùng bởi `top_k_renorm_probs` ở `1637`).

`csrc/runtime/registry.h` **không có dòng sampling nào**, nên theo chính quy tắc trong `AGENTS.md`, family này nằm ngoài dispatchable surface dù body đã có.

### `sample()` đốt gì

Với cấu hình benchmark (BF16, `k ≤ 64`, `0 < p < 1`, `min_p = 0`, `joint = false`, V=120832), `sample()` enqueue:

1. `fill_temperature_kernel` (chỉ khi `params.temperature == nullptr`).
2. `softmax(...)` — với V=120832, `softmax_ctas_for_vocab` trả **4**, nên đây là **hai** launch: `softmax_stats_kernel<<<(B,4),512>>>` rồi `softmax_norm_kernel<<<(B,4),512>>>`.
3. `top_k_top_p_sampling_kernel<1,8><<<B,256>>>`.

Tức **4 launch** (3 nếu có tensor temperature per-row). Trong kernel selection, mỗi dòng đọc lại **7 lần** mảng xác suất fp32: `row_max` (1), `select_kth_key` 4 pass radix 8-bit (4), `boundary_equals` (1), candidate compaction (1). Cộng 2 lần đọc logits BF16 ở softmax và 1 lần ghi probs fp32:

| Đường | Đọc | Ghi | Byte/(dòng, phần tử) |
|---|---|---|---|
| sàn lý thuyết | 1 × 2 B | 5 B/dòng | **2** |
| `sample()`, fast path `k ≤ 64` | 2×2 + 7×4 | 4 | **36** |
| `sample()`, **top-p only** (mặc định) | 2×2 + 8×4 | 4 | **40** |
| `sample()`, `k > 64` + top-p | 2×2 + 13×4 | 4 | **60** |
| `gumbel_max_sample()` top-k | 7 × 2 | 4 | 14 + V Philox |
| `argmax` / `compute_logprobs` | 1 × 2 | 4–8 B/dòng | **2** |
| `top_k_mask_logits` (BF16) | 3 × 2 | 2 | 8 |

Workspace là `B·V·4 + B·4 + B·ctas·8` = **483,364 B/dòng** ở V=120832, tức **247.5 MB ở B=512** (`sampling.cu:2248-2260`, `858-868`). Ba vùng `probs`/`temp`/`state` **alias** nhau, và `temp_ptr` sau đó được reinterpret thành `out_valid` — nghĩa là hai `sample()` đồng thời trên hai stream dùng chung một `Workspace` sẽ phá nhau, dù không có gì trong `ops.h` nói vậy.

### Số đo RTX 3050 và projection H100/B200

[measured] trên RTX 3050 Laptop (16 SM, 192 GB/s đỉnh, 10 warmup + 80 iter):

| Case | median | traffic mô hình | GB/s suy ra | % đỉnh |
|---|---|---|---|---|
| `topk_topp.bf16.v131072.b32.k50.p0.9` | 1.3373 ms | 151 MB | 113 | 59 % |
| `topk_topp.bf16.v32768.b256.k20.p0.95` | **1.6852 ms** | 302 MB | **179** | **93 %** |
| `gumbel.bf16.v131072.b32` (k tắt) | 0.5461 ms | 16.8 MB | 31 | 16 % |
| `softmax.bf16.v131072.b32` | 0.2653 ms | 25.2 MB | 95 | 49 % |
| `topp_renorm.fp32.v131072.b128.p0.9` | **13.37 ms** | 470 MB | 35–40 | 18–21 % |

Case thứ hai là bằng chứng khoá: **93 % đỉnh dưới mô hình 7-pass**. Một mô hình 6-pass sẽ cho 154 GB/s, 8-pass cho 205 GB/s — nên phép đo *pin* số pass ở 7 và xác nhận toàn bộ byte model. Case `topp_renorm` là bằng chứng thứ hai, ngược lại: cùng thuật toán nucleus mà `sample()` dùng khi `top_k ≤ 0` chỉ đạt **~35 GB/s** trên một bài toán mà byte cho phép 192 GB/s, vì 4 pass histogram u64 cộng `block_exscan_u64` với **24 barrier** (`sampling.cu:1098-1113`) đắt hơn traffic. Đây chính là nhánh mặc định của vLLM/OpenAI (`top_k=-1, top_p=0.9`).

[estimate] projection lên H100/B200, dùng traffic 36 B/phần tử, bandwidth đạt 0.88× đỉnh, và 12 GB/s mỗi CTA 256-thread (calibrated từ 3.5 GB/s/CTA đo trên 3050):

| B | traffic | H100 (132 SM) | B200 (148 SM) |
|---|---|---|---|
| 1 | 4.4 MB | **0.36 ms** (1 CTA × 12 GB/s) | **0.22 ms** |
| 8 | 35 MB | 0.36 ms (CTA-bound) | 0.22 ms |
| 32 | 139 MB | 0.36 ms | 0.22 ms |
| 128 | 557 MB | 0.36 ms (1.5 TB/s, vẫn CTA-bound) | 0.22 ms |
| 512 | 2.23 GB | **0.76 ms** (2.23/2.95 TB/s) | 0.32 ms |

Điểm cần nhấn: latency gần như **phẳng từ B=1 đến B=128**. Đó không phải đường cong bandwidth mà là đường cong "một CTA mỗi dòng, 256 thread". Bất kỳ thiết kế nào giữ 1 CTA/dòng đều thừa hưởng cái plateau này.

### Điểm mạnh cần giữ, và khoảng trống cần lấp

Giữ nguyên: exact threshold trên toàn vocab (`select_kth_key` + `boundary_equals`, kể cả short-circuit "tie group fully kept" ở `sampling.cu:1194`); monotone key map `prob_key`/`logit_key` xử lý NaN không cần special case; **fixed-point mass 2^40 + uint64 scan** — exact nên order-independent, đây là giải pháp sẵn có cho bài toán mà FlashInfer cảnh báo là nguyên nhân sinh token invalid; per-row param resolution; counter RNG `(request_id, stream_id, position, draw_index)`; policy `-1` cho hàng không có token hợp lệ; caller-owned workspace và run path không allocate.

Khoảng trống: materialize fp32 probs (1 write + 7–13 read); 4 pass radix 8-bit trên key 32-bit dù đã có bản 3-read; pass `row_max` vô điều kiện (14 % traffic fast path khi `min_p == 0`); nucleus 4-pass u64 + 24 barrier; một CTA/dòng 256 thread; greedy không short-circuit (một hàng greedy vẫn ghi probs one-hot rồi chạy full selection, `sampling.cu:309-355`); và **serial bubble sort 64 phần tử trong thread 0** ở fast path (`sampling.cu:1819-1830`) — ở B=1 nó có thể là critical path.

Về test: coverage tốt cho softmax/argmax/logprobs/processors và có **chi-square test thật** với 1,048,576 draw trên oracle fp64, nhưng `sample()` **chỉ được test ở V=16**. Không có test nào chạm vào radix 4-pass, `select_nucleus`, fast path ≤64, hay nhánh `-1` ở vocab thật. Không có test CUDA Graph, multi-stream, min-p, joint mode, hay per-row params cho `sample()`. Benchmark thì hard-code `status_ok(kSuccess)` (`benchmark/bench_sampling.cu:509`) nên một regression trả `kCudaError` vẫn được báo là timing hợp lệ.

## So sánh trực diện: current vs hpc-ops vs đề xuất

| Thuộc tính | `sample()` hiện tại | hpc-ops fused | Đề xuất |
|---|---|---|---|
| Kernel/launch mỗi call | **4** (3 nếu có T tensor) | 2 | **1** |
| Số lần đọc logits | 2 (BF16) | **1** | 1 (không lọc) / 2 (có lọc) |
| Byte/(dòng, phần tử) BF16 | 36 / 40 / 60 | **2** + 0.26 (bitmask + scratch) | **2 / 4** |
| Workspace @B=512,V=120832 | **247.5 MB** | ~4 MB | **~10 MB** |
| CTAs/dòng | 1 × 256 thread | `clamp(SM/B, 8, Nmax≤32)` × 1024 | 1 × 1024, hoặc S = `clamp(⌈SM/B⌉, 1, 32)` |
| K tối đa | **V** (bất kỳ) | **32/64** (clamp im lặng) | **V** (exact), fast path ≤2048 |
| Vocab | bất kỳ (split tới 262144) | **chỉ 120832** | bất kỳ |
| Top-p | exact trên toàn hàng, hoặc trên tập top-k | chỉ trên ≤64 candidate, BEFORE_TOPK dùng mass chưa chuẩn hoá | exact trên toàn hàng, tie-group như vLLM |
| min-p | có | **không** | có |
| Tie-break | id nhỏ hơn, deterministic | id nhỏ hơn sau CUB sort (không ổn định ở biên K) | id nhỏ hơn, deterministic |
| RNG | counter `(seed,row,pos,draw,stream)` | host atomic offset → **graph-frozen** | counter `(seed,request,step,token)` |
| CUDA Graph | capture-clean, test cho softmax | capture được nhưng **sai ngữ nghĩa** khi replay | capture-clean, replay-safe |
| Multi-stream | cần workspace riêng (alias 3 vùng, không document) | fused OK, temperature **không** | OK, per-call state |
| Allocate trong replay | không | **8 allocator node** | không |
| Hàng all `-inf` | `out_ids = -1` | token 0 | `-1` |
| Latency B=1 (H100) | ~0.36 ms [estimate] | 6–10 µs (full) / 2–4 µs (temp) [estimate] | **~2–3 µs** [estimate] |
| Latency B=512 (H100) | ~0.76 ms [estimate] | **120–370 µs** (sort-bound?) [estimate, chưa đo] | ~45 µs (1 pass) / 80–90 µs (2 pass) [estimate] |

Đọc bảng này đúng cách: kernel-ops hiện tại thua hpc-ops **2–6× ở B=512 nhưng tới 36–60× ở B ≤ 8** — tức càng batch nhỏ càng tệ, đúng ngược với kỳ vọng, vì cả hai lý do thua (traffic multiplier 18× và parallelism 1-CTA-per-row) đều bị phóng đại khi không có đủ dòng để lấp SM. hpc-ops thua thiết kế đề xuất vì **sort + 2 launch + Nmax cap**, không vì byte. Thiết kế đề xuất thua hpc-ops ở đúng một ô: **traffic thuần ở B=512** (4 B so với 2 B/phần tử), và thắng lại nhờ bỏ sort.

## Thiết kế đề xuất: fused exact sampler, một launch

### Nguyên tắc

Một launch. Không materialize xác suất. Pass 1 đọc logits **một lần** và dựng mọi thứ cần cho threshold. Pass 2 đọc lại (hoặc đọc từ smem) và gom candidate. Rút thăm bằng **exponential race** trên tập candidate đã compact, với noise là hàm thuần của `(seed, request, step, token_id)` — nhờ vậy kết quả **không phụ thuộc thứ tự cộng, số split, hay hình dạng grid**, tức bitwise identical dưới graph replay và dưới mọi cách chia batch.

Hai biến thể chọn ở plan time, không ở runtime:

- **Path A — row-resident (B ≥ số SM).** Một CTA 1024 thread mỗi dòng. Pass 2 đọc lại từ DRAM; với B ≤ ~128 toàn bộ logits (≤31 MB) nằm trong L2 50 MB của H100 nên pass 2 gần như miễn phí.
- **Path B — shard-resident (B nhỏ).** `S = clamp(2^floor(log2(SM/B)), 4, 32)` CTA mỗi dòng. Mỗi CTA **cache nguyên shard vào shared memory** ngay trong pass 1: V=120832, S=32 → 3776 phần tử × 2 B = **7.4 KiB**; S=4 → 59 KiB, vẫn vừa ngân sách smem. Pass 2 đọc từ smem, nên **DRAM chỉ bị đọc một lần** ngay cả khi có lọc. Merge ở CTA cuối qua atomic ticket, không cần grid sync.

Path B là chỗ thiết kế này vượt cả bốn note: nó không đánh đổi traffic để lấy occupancy, mà lấy cả hai, vì shard nhỏ hơn nhiều so với smem. Điều kiện hợp lệ: `V/S × dtype_bytes ≤ smem_budget` (7.4 KiB với BF16, 14.8 KiB với FP32 ở S=32) — luôn đúng trong dải V mục tiêu.

### Dataflow

```
                    ┌─────────── caller-owned Workspace (không alloc khi replay) ───────────┐
 seed[B], step[B] ──┤ device tensor (engine cập nhật giữa các replay)                        │
 penalty bitmask ───┤ [B, ceil(V/8)] uint8  (1 atomicOr writeback / dòng)                     │
 topk/topp/minp/temp┤ [B] mỗi thứ (null → scalar default)                                     │
 logits [B,V] ──────┤ read-only                                                               │
                    │ row_state[B]  { m, Z, argmax, n_cand, flags }           32 B/dòng        │
                    │ split_state[B,S] { m, Z, argmax, cnt[256], mass_q[256] } 3.1 KiB/split   │
                    │ cand[B, C] { key u32, idx u32 }  C=2048                 16 KiB/dòng       │
                    └──────────────────────────────────────────────────────────────────────────┘

 grid = (S, B), block = 1024
 S = 1 nếu B ≥ SM/4, ngược lại clamp(2^floor(log2(SM/B)), 4, 32)
     (Path B cần V/S · e ≤ 64 KiB; với BF16 V=120832 nghĩa là S ≥ 4)

 ┌ CTA (b, s) — PASS 1: 1 lần đọc, 1 exp/phần tử ─────────────────────────────────────┐
 │ load shard (vector 16 B), áp bitmask penalty → x, rồi x *= rcp(T)                  │
 │ (Path B) copy x vào smem shard[]                                                   │
 │ online softmax per-thread: (m_t, s_t), combine kiểu associative                    │
 │ hist_cnt[256]++ và hist_mass_q[256] += fp32→fixed(exp(x - hi_bin))                 │
 │ argmax per-thread                                                                  │
 │ block-reduce (m, Z, argmax) theo cây cố định; ghi split_state[b,s]                 │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                    │  (chỉ Path B: atomicAdd ticket, CTA cuối merge S split)
 ┌ MERGE + DESCENT — số học trên 256 bin, không đọc vocab ────────────────────────────┐
 │ m = max(m_s), Z = Σ Z_s·exp(m_s - m); mass_bin = Σ mass_q·exp(hi_bin - m)          │
 │ top-k: suffix-sum cnt → bin b_k nơi count vượt K                                   │
 │ top-p: suffix-sum mass với bracket [cnt·exp(lo-m), cnt·exp(hi-m)] → b_p bảo thủ    │
 │ t_lo = min_key(thr_k, thr_p)  (key thấp hơn = support rộng hơn = superset)         │
 │ publish (t_lo, b*, m, Z)                                                           │
 └────────────────────────────────────────────────────────────────────────────────────┘
 ┌ CTA (b, s) — PASS 2: gom candidate ───────────────────────────────────────────────┐
 │ nguồn: smem shard (Path B) hoặc đọc lại DRAM slice (Path A)                        │
 │ nếu key ≥ t_lo: append (key, idx) vào cand[b] qua ticket CÓ THỨ TỰ (prefix-sum)    │
 │ đồng thời dựng sub-histogram 8-bit trên w = exp(x - m) của candidate               │
 └────────────────────────────────────────────────────────────────────────────────────┘
 ┌ FINISH — chỉ trong shared memory, ≤2048 candidate ────────────────────────────────┐
 │ 1–2 vòng 8-bit trên w → threshold chính xác, giữ nguyên tie group tại biên         │
 │ lọc thêm min_p (w ≥ min_p·w_max) và token mask                                     │
 │ exponential race: argmax_j  w_j / (−log u_j),  u_j = uniform(seed, req, step, tok_j)│
 │ tie-break: ratio lớn hơn thắng, bằng nhau thì token id nhỏ hơn                      │
 │ ghi out_ids[b]; atomicOr bit penalty; optional out_logprob / chosen_rank            │
 └────────────────────────────────────────────────────────────────────────────────────┘

 fallback xác định (hiếm): n_cand > C  →  một pass global thứ ba với t_lo' = threshold
 đã siết, hoặc rejection sampling có bound với CÙNG RNG (TRT-LLM pattern)
```

### Vì sao từng lựa chọn

**Pass 1 gộp penalty + temperature + histogram + online softmax.** Đây là điểm hpc-ops làm đúng và phải copy: penalty áp trên giá trị raw trước temperature (`fused_sampler.cu:94-112`), cả hai nằm trong cùng vòng load, nên chi phí biên là 0 byte. Registers giữ `(m_t, s_t)` và `(best, best_idx)`; block reduce theo cây cố định nên kết quả không phụ thuộc scheduling.

**Histogram 8-bit trên monotone key, 16 bản private.** Contention chứ không phải capacity là chi phí — đó là lý do TRT-LLM dùng `kHistCopies = 16` (`fusedSamplingKernelsCommon.cuh:48`) và ghi rõ "w ∈ (0,1] nên top 8 bit chỉ dùng ~64/256 giá trị, và một softmax dồn phần lớn hàng 128k vào vài bin" (`:322-331`). 256 bin × 4 B × 16 = 16 KiB cho count; mass fixed-point uint64 × 16 = 32 KiB. Tổng 48 KiB/CTA, chấp nhận được với 228 KiB/SM trên H100 (2–4 CTA/SM).

**Mass fixed-point thay vì float atomic.** Đây là chỗ tôi sửa lại đề xuất trong note thiết kế. `atomicAdd(float)` cho thứ tự cộng phụ thuộc scheduling → mất bitwise determinism, đúng cái mà `csrc/common/rng.cuh:4-7` bảo đảm. Giải pháp: mỗi phần tử đóng góp `exp(x − hi_bin)` (≤ 1, vì `hi_bin` là biên trên của bin), nhân `2^32`, cộng bằng **uint64 atomic**. Tổng của một bin ≤ `count × 2^32 ≤ 131072 × 2^32 = 2^49`, vừa uint64. Mass thật của bin = `mass_q × exp(hi_bin − m) / 2^32`, một phép nhân sau khi biết `m`. Nhờ vậy pass 1 cho **mass chính xác theo bin mà không cần biết max trước**, và toàn bộ là số nguyên nên deterministic tuyệt đối. Khi bin rất rộng (logit lớn sau khi chia cho T nhỏ), `exp(x − hi_bin)` có thể underflow về 0; điều đó chỉ làm bracket lỏng hơn chứ không làm sai, vì **cận trên vẫn được tính giải tích** là `count × exp(hi_bin − m)`, không lấy từ accumulator.

**Hai ngưỡng từ một histogram.** Top-k là ngưỡng theo *rank*: suffix-sum `cnt` tới khi vượt K. Top-p là ngưỡng theo *mass*: suffix-sum `mass_bin`, nhưng vì trong bin có phân bố, dùng bracket `[cnt_b·exp(lo_b−m), cnt_b·exp(hi_b−m)]` để chọn bin bảo thủ `b_p` sao cho mass upper-bound đã vượt `p·Z`. Vì upper-bound ≥ mass thật, `t_lo = lo(b_p)` **luôn ≤ ngưỡng thật** → tập gom được là superset, **không bao giờ phải rescan vì under-admit**. Đây là khác biệt so với GVR V2, nơi tác giả tự nhận V2 có thể chậm hơn V1 0.689× ở worst case vì phải rescan ở ngưỡng thấp hơn.

**Exponential race thay vì Gumbel hay inverse-CDF.** `argmax_j w_j / (−log u_j)` với `u_j ~ Uniform(0,1]` cho đúng phân bố categorical trên tập đã lọc, **không cần normalizer, không cần prefix sum, không cần `log p`** (nên không có `-inf` do underflow: `w` có thể 1e-38 và vẫn chia được). Quan trọng hơn: `argmax` là associative/commutative với tie-break toàn phần, nên kết quả **bất biến với số split và thứ tự thread**. Đó là điều kiện để D2 determinism (bitwise identical với mọi B) khả thi — điều mà một thiết kế inverse-CDF không làm được vì phụ thuộc thứ tự cộng.

**Chỉ 1 hash mỗi candidate, không phải mỗi token vocab.** Sau compaction, mỗi dòng cần ≤2048 hash thay vì 120832 — giảm 59× công RNG, và quan trọng hơn là làm kết quả độc lập với việc thread nào xử lý phần tử nào.

**Không sort.** Bước cuối là selection trên ≤2048 candidate: nếu cần thứ tự (cho logprobs/top-n) thì một warp bitonic trên 64–2048 phần tử; nếu chỉ cần rút thăm thì không cần thứ tự nào cả.

### Pseudocode của CTA

```text
template <DType, POLICY, int kSplits>  __global__ fused_sample(plan, params, logits, ws, out)
  b = blockIdx.y; s = blockIdx.x; row = logits + b*stride
  k, p, mp, T = row_params(params, b)                    // device reads, không sync
  if (!(T > 0)) return greedy_shard_path();              // 1 pass, không exp, merge như dưới

  // ---- PASS 1 ----
  for (v = s*chunk + tid; v < min((s+1)*chunk, V); v += blockDim) {
      float x = to_f(row[v]);
      if (bitmask) x = mask_bit ? (x>0 ? x/rp : x*rp) : x;
      if (counts)  x = seen ? x - freq*cnt - pres : x;
      x *= rcp_approx(T);
      if (kSplits > 1) smem_shard[v - s*chunk] = x;       // Path B
      key = monotone_key(x);
      online_max_sum(m_t, s_t, x);                        // 1 exp
      atomicAdd(&hist_cnt[key>>24][lane&15], 1);
      atomicAdd(&hist_mass[key>>24][lane&15], fixed(exp(x - hi(key>>24))));
      if (x > best) { best = x; best_idx = v; }
  }
  reduce_block(m_t, s_t, best, best_idx) -> (m, Z, argmax)
  ws.split[b][s] = {m, Z, argmax, hist_cnt, hist_mass}
  if (kSplits > 1) {
      if (atomicAdd(&ws.ticket[b], 1) == kSplits-1) { merge_splits(b); }  // CTA cuối
  } else { merge_splits(b); }                                             // S=1: tự làm

  // ---- DESCENT (số học trên 256 bin) ----
  t_lo = min_key(thresh_topk(k), thresh_topp_bracketed(p*Z))
  if (s == kSplits-1) { ws.thr[b] = t_lo; __threadfence(); ws.ready[b] = epoch; }
  if (kSplits > 1) spin_until(ws.ready[b] == epoch);      // mọi CTA đều resident

  // ---- PASS 2: gom candidate ----
  for (v = ...) { x = (kSplits>1) ? smem_shard[...] : to_f(row[v]);
                  if (monotone_key(x) >= t_lo) push_ordered(ws.cand[b], x, v); }

  // ---- FINISH (smem, ≤C) ----
  if (kSplits > 1) { atomicAdd(&ws.done[b], 1); spin_until(ws.done[b] == kSplits); }
  if (s == kSplits-1) {
      refine_8bit(w = exp(x-m)) x2  -> exact threshold, giữ nguyên tie group
      apply min_p: keep w >= mp * w_max
      winner = argmax_j  w_j / (-log u_j),  u_j = uniform(seed_b, b, step_b, tok_j)
      out[b] = winner; atomicOr(bitmask, 1u << winner);
  }
```

### Fusion processors và hook cho speculative decoding

Ba loại processor có ba chi phí khác nhau và không nên đối xử giống nhau. **Token bitmask** gần như miễn phí khi fuse: nó chỉ thêm `V/8` byte đọc và một phép nhân (`fused_sampler.cu:94-106`), và writeback chỉ là **một `atomicOr` 32-bit mỗi dòng**. **Logit bias** đã là `O(nnz)` in-place (`csrc/sampling/logits_processor.cu:171-192`), nên cứ để nó là kernel riêng trước sampler — nó không tạo thêm full pass, và một launch rẻ hơn nhiều so với việc scatter bias vào trong pass 1.

**Repetition/frequency/presence** là trường hợp đáng fuse nhất và cũng dễ sai nhất. `apply_penalties_kernel` đọc `counts[row*V + v]` cho **mọi** v, tức 4 B/phần tử, rồi ghi đè logits in-place (`logits_processor.cu:104-138`). Chạy nó như một kernel riêng nghĩa là **2 B đọc + 2 B ghi + 2 B đọc lại** = 6 B/phần tử trước khi sampler bắt đầu. Fuse vào pass 1 chỉ tốn **4 B/phần tử** (đọc counts) và bỏ luôn một launch — tiết kiệm 2 B/phần tử và một node graph. Điều kiện: engine phải cấp `counts` như `[B, V] int32` sẵn; nếu chỉ có danh sách token đã sinh thì đừng fuse, vì binary search `log2(L)` bước mỗi phần tử sẽ đắt hơn 4 B đọc.

**Speculative decoding** cần đúng một tính chất: noise của một token tại một step phải **giống hệt** giữa lượt draft và lượt verify, nếu không tỉ số `p_target/p_draft` không còn nghĩa. Vì noise ở đây là `f(seed, request, step, token_id)` — không có state, không có `draw_index` tuần tự — tính chất đó **miễn phí**. Chỉ cần tách miền bằng `stream_id`/salt: 0 cho sampling, 1 cho verify, 2 cho draft noise, đúng kiểu `_DRAFT_NOISE_SALT` của vLLM. Hook cần thêm là một output tuỳ chọn `chosen_logprob` tính bằng **một gather** trên token thắng (không phải cả hàng), và một biến thể `sample_from_probs_gathered` cho `rejection_sample` để nó không phải materialize `[num_draft, V]` như `speculative_kernels.cuh:17-18` đang specify.

### Ngữ nghĩa phải khớp

Exact full-vocab là mặc định: `top_k ≥ V` và `top_p ≥ 1` phải cho **đúng** phân bố full-vocab, không được âm thầm cắt. `top_k = 0` nghĩa là **tắt**, không phải "kẹp về 64". Thứ tự mặc định là top-k trước rồi top-p trên mass **đã renormalize theo top-k** (`sampling.cu:1265-1269`), với `joint_topk_topp = true` là mode thứ hai. Tie group tại biên K được giữ **toàn bộ** (giống `logits < top_k_mask` của vLLM), và thứ tự toàn phần là (key giảm dần, index tăng dần). NaN coi như `-inf`; `+inf` cho uniform trên các entry `+inf`; hàng không có token hợp lệ ghi `-1` + `out_valid = 0`; `min_p > 1` phải bị **reject trên host** thay vì lặng lẽ làm rỗng hàng, và `top_p = NaN` cũng vậy.

### Chi phí kỳ vọng

Công thức, V phần tử, B dòng, `e` = 2 B (BF16) hoặc 4 B (FP32):

```
traffic_unfiltered = B · V · e                       (1 pass)
traffic_filtered   = B · V · e · (1 + [Path A])      (Path B: pass 2 từ smem → vẫn 1×)
                    + B · V/8                        (bitmask, nếu bật penalty)
                    + B · C · 8                      (candidate, C ≤ 2048)
workspace          = B·32 + B·S·3104 + B·C·8
```

V=120832, BF16, C=2048:

| B | S | pass DRAM | traffic | H100 @2.95 TB/s | B200 @7.0 TB/s |
|---|---|---|---|---|---|
| 1 | 32 | **1** | 0.27 MB | **~2–3 µs** (launch-bound) | ~2–3 µs |
| 8 | 16 | **1** | 2.2 MB | **~2–3 µs** | ~2–3 µs |
| 32 | 4 | **1** | 8.7 MB | **~4–5 µs** | ~3 µs |
| 128 | 1 | 2 (pass 2 từ L2) | 61.9 MB → ~1.3 pass hiệu dụng | **~15 µs** | ~8 µs |
| 512 | 1 | 2 | 247 MB | **~85–90 µs** | ~35 µs |

Workspace: B=1 → 115 KiB; B=512 (S=1) → 1.6 MiB (split_state) + 8 MiB (cand) + 16 KiB ≈ **9.7 MiB**, so với 247.5 MB hiện tại. Toàn bộ là [estimate] từ byte model, với giả định bandwidth đạt 90 % đỉnh và launch overhead 1.5–2.5 µs; cả hai đều phải đo.

So với hpc-ops: ở B ≤ 32 thiết kế này **1 pass như hpc-ops nhưng không sort**; ở B=64..128 nó 1.3–2 pass nhưng không sort và không có 31 wave; ở B=512 nó 2 pass và **đây là ô duy nhất mà hpc-ops có lợi thế traffic**.

## Vì sao thắng hpc-ops — và chỗ nào chỉ hòa hoặc thua

Thắng có cấu trúc, không phải nhờ micro-optimization:

- **Bỏ 1.008 radix item/phần tử × 8 pass.** Đây là khoản lớn nhất. Đổi lại thiết kế mới thêm **1 exp + 2 atomic smem** mỗi phần tử ở pass 1 và **1 exp + 1 phép chia** mỗi candidate ở pass 2. Với V=120832, B=512, một exp/phần tử là **~16 µs** SFU trên H100 (132 SM × 16 MUFU/clk × 1.8 GHz ≈ 3.8e12 ops/s) [derived], tức nằm gọn trong 41 µs memory time; hai exp vẫn còn dư; ba exp thì bắt đầu SFU-bound. Thiết kế này dùng **2 exp/phần tử** ở pass 1+2, đúng ở biên.
- **Bỏ 31 wave.** hpc-ops ở B=512 chạy `N=8` block/dòng = 4096 block, 1 block/SM → 31 wave tuần tự. Thiết kế mới ở B=512 cũng 512 block nhưng mỗi block làm 2 pass độc lập, không sort, nên wave chỉ là bài toán bandwidth.
- **Bỏ launch thứ hai và 8 allocator node.**
- **Bỏ Nmax cap.** `N` không còn bị buộc bởi block size của kernel merge, nên B=1 dùng đủ SM thay vì 24 % (K=32) hay 12 % (K=64); và **K lớn không làm batch nhỏ chậm đi**.
- **Bỏ giới hạn vocab và K.** 128256/151936 chạy được; K > 64 chạy được; `top_k=0` là full-vocab thật.

Chỗ chỉ hòa, hoặc thua:

- **B=1 và B ≤ 8.** Đây là launch-latency regime. hpc-ops temperature path đã là 1 launch, ~2–4 µs [estimate], và thiết kế mới nhắm ~2–3 µs — tức **hòa, không thắng**. Cái thắng ở B=1 là *tính năng* (full-vocab, min-p, vocab bất kỳ), không phải latency. Bất kỳ ai hứa "10× ở B=1" đều đang so với `sample()` hiện tại, không phải so với hpc-ops temperature path.
- **K rất nhỏ (K ≤ 8) và hàng ngắn.** Với K ≤ 8, một heap/insertion 4-phần-tử trong register có thể rẻ hơn histogram 256 bin. Ngưỡng crossover nên chọn bằng đo, không bằng niềm tin; dự kiến K ≤ 8 và V ≤ 8192 thì register-top-k thắng.
- **B=512 nếu hpc-ops thực sự ~40 µs.** Khi đó thiết kế 2-pass (~85–90 µs) **thua**, và phương án là chuyển sang **1-pass có self-sampling** (GVR V2 style): sample ~1/64 số phần tử để đoán threshold, gom candidate theo một safety floor ngay trong pass 1, verify trên tập đã gom, rescan chỉ khi under-admit. Đó là roadmap M7, không phải M1.
- **Overhead hiện thực.** Split-row cần hai global barrier cho Path B; mỗi barrier là một vòng global memory + fence, có thể 0.5–2 µs. Ở B=1 với tổng 2–3 µs, barrier là 30–60 % thời gian. Nếu đo thấy vậy, phương án là **bỏ barrier thứ hai**: mỗi shard tự giữ top-C cục bộ đã sort, merge CTA đọc workspace và merge S-way, không cần publish ngược.
- **Workspace.** hpc-ops vẫn thắng ở đây (4 MB so với ~10 MB), và điều đó ổn — mục tiêu là *không thua 25×* như hiện tại.

## Kế hoạch triển khai trong repo

### File layout và ràng buộc header

Theo `AGENTS.md`, mỗi file `.cu` phải là một "named concern", không được có hậu tố ordinal, và phải có mặt trong `ROADMAP.md` Phụ lục E.2 nếu không `tools/check_roadmap.py` sẽ reject. Đề xuất:

```text
csrc/sampling/fused_sampler.cu            # host API + launcher + plan resolution
csrc/sampling/fused_sampler_kernels.cuh   # SamplingPlan, FusedParams, __global__ signature
csrc/sampling/fused_sampler_impl.cuh      # thân kernel (templated trên dtype/policy)
```

Cả ba header phải `#pragma once`, **không `#include`**, và khai báo `Requires:` (ví dụ `Requires: common/*, sampling/*.cuh`). `fused_sampler_kernels.cuh` là device-only vì chứa `__grid_constant__`, nên chỉ được include từ device pass hoặc từ host check có `tools/header_check/cuda_host_shim.h`. Cần thêm dòng tương ứng vào `tools/header_check/check_all_headers.cc` theo đúng thứ tự dependency, và cập nhật `docs/TREE.md` (nếu có trong checkout này — hiện `docs/` chỉ có `KV_CACHE_PARITY.md`, nên đây là một gap cần xử lý trước).

### API và ABI

Giữ nguyên signature `sample()` và `sampling_workspace_requirements()`; thêm một entry plan hoá tường minh:

```cpp
struct SamplingPlan {          // resolved before capture, stable across replay
  int32_t splits;              // S
  int32_t cand_capacity;       // C
  int32_t block_threads;       // 512 or 1024
  uint32_t flags;              // has_bitmask | has_minp | joint | exact_fallback
  uint64_t capture_generation; // bump khi S/C/threads đổi
};
Status sampling_plan(const Tensor& logits, const SamplingParams& params,
                     const LaunchContext& ctx, SamplingPlan* out);
Status sample_fused(const Tensor& out_ids, const Tensor& logits, const SamplingParams& params,
                    const SamplingPlan& plan, const Workspace& ws, const LaunchContext& ctx);
```

Bốn thay đổi là ABI-relevant và phải bump `KOPS_ABI_VERSION` 5 → 6: workspace layout mới (ba vùng thay vì alias), `SamplingPlan` như một tham số mới, tham số device mới (`step`, `seed` tensor), và template axis mới (`S`, `C`, policy). Theo `AGENTS.md`, thay đổi kernel template parameter, thứ tự argument, hay launcher signature đều thuộc diện bump.

`sampling_workspace_requirements` phải trả **công thức mới** và `ops.h` phải mô tả đúng ba vùng — hiện tại nó understate (nói chỉ có `num_tokens * vocab * 4` trong khi code còn reserve `B*4` cho temperature fill, `sampling.cu:863`).

### Workspace plan

```
offset 0                      : row_state[B]           32 B/dòng, 256-align
offset B*32                   : split_state[B][S]      3104 B/split
offset B*32 + B*S*3104        : cand[B][C]             8 B/candidate
```

`C = 2048` cho K ≤ 2048; nếu `params.default_top_k > 2048` thì plan chọn `C = next_pow2(K + 256)` và bật `exact_fallback`. Tổng ≤ ~10 MiB ở B=512. Mọi offset là hàm của `(B, V, S, C)` — không phụ thuộc pointer hay device property — nên replay-safe.

### Multi-stream

Mỗi `Workspace` chỉ dùng cho một call đang chạy. Ba vùng không alias nên hai call trên hai stream với hai workspace rời là an toàn tuyệt đối; điều này phải được **document trong `ops.h`** (hiện không có) và có test. Không có static device state, không có per-device scratch, nên không lặp lại lỗi của `fused_sampler_temperature.cu`.

### Milestones

| M | Deliverable | Exit criterion |
|---|---|---|
| **M0** | Bench harness dùng chung + baseline | hpc-ops K1/K2 time và DRAM bytes trên H100/B200 tại B ∈ {1,32,512}, V=120832, K=32/64; bảng estimate ở mục 7 được thay bằng số đo |
| **M1** | 1-pass exact path: greedy + temperature + unfiltered, exponential race, penalty từ bitmask | khớp `gumbel_max_sample` về phân bố; 1 launch; D1+D2 pass; nhanh hơn `sample()` ≥ 10× ở mọi B |
| **M2** | Pass 1 histogram + descent + threshold top-k | threshold **bitwise equal** `select_kth_key` trên toàn bộ hàng adversarial; 2 pass, 1 launch |
| **M3** | Compaction vào smem + race + fallback khi overflow | chi-square/TV pass cho `k ∈ {1,50,1000}`; TV trong nhiễu ở N=1e6 |
| **M4** | Top-p mass descent, min-p, joint mode | chi-square pass cho `p ∈ {0.9,0.95}`, `min_p ∈ {0.05}`; `top_p=1` là no-op |
| **M5** | Path B split-row + last-CTA merge + graph capture | 1 launch ở mọi B; replay bitwise stable; không `cudaMalloc` trong capture window |
| **M6** | Spec decode: rejection + tree verify dùng chung noise | acceptance rate khớp `Σ min(p,q)`; draft/verify noise trùng |
| **M7** | 1-pass self-sampling cho B lớn (GVR V2 style) + `outputProbs` opt-in | chỉ bật khi M0 chứng minh cần; không regression path mặc định |

M1 nên ship trước như một cải thiện độc lập: nó chỉ cần đường unfiltered, đã cắt 3 launch và toàn bộ ma trận fp32, và không phụ thuộc bất kỳ quyết định nào về histogram.

### Test matrix

Bắt buộc phải có, vì hiện tại `sample()` chỉ được test ở V=16:

- **Exactness có kiểm soát noise**: dùng external-Gumbel kiểu hpc-ops (một buffer `[B,V]` fp32 hoặc một bảng `uniform_from_counter` cố định), so token với argmax tính bằng fp64. Đây là test bắt được bug threshold mà chi-square cần 1e6 mẫu mới thấy.
- **Distribution**: 1e6 draw, oracle fp64, chi-square với bin gộp `expected < 5`, cộng TV distance ngưỡng `5·sqrt(K/(2N))`. Chạy ở V=120832 cho `top_k=50`, `top_p=0.9`, `min-p=0.05`, và **top-p-only** (nhánh chưa từng được test).
- **Threshold equivalence**: fast path và general path phải cho cùng kết quả trên cùng hàng (ví dụ k=50 vs k=65).
- **Determinism matrix**: hai lần trong cùng process, hai lần ở process mới, `B` ghép `{8}` vs `{4,4}` vs `{1}×8`, hai stream, và dưới graph replay. Kỳ vọng D1 luôn đúng; D2 đúng như đã declare.
- **Adversarial rows**: all-equal, one-hot, hai giá trị bằng nhau đúng biên K, dynamic range 1e6, `±inf`, NaN, V không chia hết vector width, `stride[0] ≠ V`.
- **Edge**: `top_k ≥ V` phải bằng full-vocab; `top_p = 1` no-op; `min_p = 1` chỉ còn argmax; `T → 0` greedy; all `-inf` → `-1`; `V = 1`.
- **Multi-stream**: hai `sample_fused` trên hai stream, hai workspace, so với chạy tuần tự.

## Benchmark và kiểm chứng

### Case matrix

Trục V: 120832, 128256, 151936 (thêm 32000 và 262144 cho robustness). Trục B: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512. Dtype: BF16, FP32. Params: greedy; T=0.7; T=1.0; top-k=1/50/1000; top-k=50 + top-p=0.9; **top-p=0.95 only**; min-p=0.05; joint vs sequential. State: không penalty; repetition + bitmask; frequency/presence; draft mask.

### Baseline

Bốn baseline, chạy **cùng process, cùng tensor, cùng allocator**:

1. `sample()` hiện tại (kernel-ops).
2. hpc-ops fused + temperature (cần build riêng, vocab 120832).
3. FlashInfer `sampling.top_k_top_p_sampling_from_logits` — cùng input, không qua serving layer.
4. vLLM `_C.fused_gumbel_argmax` / `apply_top_k_top_p` nếu build được.

So sánh service-level với vLLM/SGLang là **không công bằng** vì họ gồm cả scheduling; chỉ so kernel-level.

### Metric

p50/p90/p99 trên ≥1000 iteration với decode loop đã capture; `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum` mỗi launch; achieved GB/s và tỉ lệ so với sàn 1-pass; `sm__warps_active.avg.pct_of_peak`; số graph node (metric hạng nhất ở B ≤ 8); `sm__inst_executed_pipe_xu` để phát hiện SFU-bound — mô hình dự đoán crossover ở ~2 transcendental/phần tử. Hai công cụ, hai mục đích, không trộn: **`nsys`** với NVTX range quanh mỗi decode step cho latency end-to-end và per-node trên GPU timeline (đây là số dùng để so sánh), **`ncu`** cho traffic/occupancy/SFU (không dùng để đo latency vì nó serialize kernel). Với B=1 phải báo riêng **launch overhead** đo bằng một graph chỉ có một kernel rỗng, vì ở đó launch là 30–100 % thời gian.

### Fairness

Cùng logits tensor và cùng allocation, warmup ≥50, clock lock, cùng CUDA/driver, 5 run độc lập, báo median-of-p50 và worst-of-p99, và **luôn báo DRAM bytes thực tế đi kèm latency** để một "thắng" không thể đến từ việc bỏ việc. Với hpc-ops phải báo rõ `max_topk` và `softmax_policy`, vì hai tham số đó đổi hoàn toàn bài toán: so `max_topk=32` với một thiết kế exact là so hai contract khác nhau, không phải hai cách hiện thực cùng một contract.

### Định nghĩa "thắng" (falsifiable)

1. **Shared cases** (K ≤ 64, V=120832, BF16/FP32): p50 và p99 không tệ hơn ở **mọi** B, và **≥1.3× nhanh hơn ở B ≥ 64**. Không đạt được thì chưa thắng, chỉ mới "nhiều tính năng hơn".
2. **Launch count = 1** ở mọi B (hpc-ops: 2).
3. **Workspace ≤ 10 MiB ở B=512** (hiện tại 247.5 MB; hpc-ops ~4 MB — hòa ở mức chấp nhận được).
4. **Ngữ nghĩa**: exact full-vocab, min-p, 128256/151936, `top_k=0` là tắt.
5. **Determinism**: D1 luôn, D2 declare; bitwise identical qua graph replay và qua số stream.
6. **Không allocate, không D2H sync** trong replay path.

## Rủi ro và câu hỏi mở

**Rủi ro lớn nhất là baseline chưa đo.** Toàn bộ lập luận "nhanh hơn hpc-ops ở B=512" đứng trên một mô hình sort cost với biên ±2×. Nếu M0 đo được 40–60 µs ở B=512 thì thiết kế 2-pass **thua ở đúng batch lớn nhất**, và câu chuyện phải chuyển thành "thắng ở B ≤ 128 + thắng bằng ngữ nghĩa + 1 launch". Đây không phải rủi ro lý thuyết: 41.8 µs là sàn byte của chính hpc-ops, và một kernel sort-bound nhẹ có thể chỉ cách sàn 1.5×.

**Rủi ro thứ hai là overflow của crossing bin.** Nếu hàng phẳng (temperature cao, logits đều) thì bin ranh giới có thể chứa hàng nghìn phần tử và `C=2048` không đủ. Phải đo overflow rate trên logits thật ở T ∈ {0.7, 1.0} và trên các hàng adversarial; nếu >1 % thì tăng C hoặc chấp nhận pass 3 có xác suất. TRT-LLM chọn `kCandCap = 2048` nhưng đó là cho K=512..2048, không phải K=32.

**Rủi ro thứ ba là L2 residency.** Toàn bộ lập luận "2 pass gần như miễn phí ở B ≤ 128" dựa vào 50 MB L2 của H100; hit rate thực tế chưa đo, và B200/GB100 có L2 khác (một số nguồn nói ~126 MB, cần xác nhận đúng SKU). Nếu hit rate thấp, B=128 tụt từ ~15 µs xuống ~25 µs.

**Rủi ro thứ tư là SFU.** Hai exp/phần tử ở pass 1+2 là đúng ở biên 41 µs memory time trên H100. Trên B200, memory time chỉ 15.5 µs cho cùng khối lượng, nên **cùng một thiết kế sẽ SFU-bound trên B200 trước**. Nếu vậy, phương án là dùng xấp xỉ `exp2` với bảng tra, hoặc chuyển sang histogram chỉ-count cho top-k và chỉ tính mass trên tập candidate.

**Rủi ro thứ năm là nhiễu số học của race.** `−log(u)` với `u` tiến về 1 cho giá trị nhỏ, và tỉ số `w/(−log u)` có thể tràn nếu `u` cực nhỏ. Phải làm đúng như vLLM: lấy đuôi noise lớn (quyết định winner) từ `u → 0` nơi fp32 có độ phân giải mịn, và dùng series `log1p` bậc 8 cho `−log(1−u)` với sai số tuyệt đối < 6e-7 trên `[0, 0.25]`. Cần một chi-square test riêng cho bias của race ở V=120832.

**Rủi ro thứ sáu là ABI churn.** Workspace layout mới + plan struct + device tensor mới là ba thay đổi độc lập; gộp vào một bump ABI 6 và một migration note, thay vì ba bump.

Câu hỏi mở còn lại: (a) hpc-ops K1 sort chiếm bao nhiêu % — chưa ai đo; (b) crossing-bin overflow rate trên logits thật — chưa ai đo; (c) launch overhead bare và graph-replayed trên H100/B200 — chưa ai đo; (d) vLLM `gumbel.py` có apply top-k/top-p như mask upstream không — không thấy trong file, nên "vLLM bỏ top-p" là **chưa xác nhận**; (e) liệu một thiết kế 1-pass self-sampling có thắng được 2-pass ở B=512 đủ để đáng độ phức tạp không — câu trả lời phụ thuộc (a).

## Tài liệu tham khảo

**Trong repo**

- `3rd/hpc-ops/src/sampler/fused_sampler.cu`, `sampler.h`, `sampler_rng.cuh`, `fused_sampler_temperature.cu`, `entry.cc`, `tests/test_sampler.py`, `benchmark/sampler/benchmark_sampler.py`, `README.md`
- `3rd/attention.rs/src/kernels/src/fast_topk.cu`, `ffi.rs`, `moe/topk_softmax_kernels.cu`
- `3rd/sglang/sgl-kernel/csrc/elementwise/topk.cu`, `tests/test_topk.py`
- `3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/indexerTopK.cu`, `kernels/fusedSampling/*`, `docs/source/blogs/tech_blog/blog29_GVR_V2_Self_Sampling_Exact_TopK_for_Sparse_Attention.md`
- `3rd/vllm/vllm/v1/sample/ops/topk_topp_sampler.py`, `topk_topp_triton.py`, `vllm/v1/sample/rejection_sampler.py`, `csrc/libtorch_stable/persistent_topk.cuh`
- `csrc/sampling/ops.h`, `sampling.cu`, `sampling_kernels.cuh`, `speculative_kernels.cuh`, `logits_processor.cu`
- `csrc/common/tensor.h`, `csrc/common/rng.cuh`, `csrc/common/device_impl.cuh`, `csrc/common/macros.cuh`
- `tests/unit/test_sampling.cc`, `benchmark/bench_sampling.cu`, `AGENTS.md`, `ROADMAP.md`

**Ngoài repo**

- Tencent/hpc-ops — https://github.com/Tencent/hpc-ops
- CUDA Programming Guide, Graph Memory Nodes — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- FlashInfer, "Sampling under filtering, sorting-free" — https://flashinfer.ai/2025/03/10/sampling.html
- FlashInfer top-k API (FilteredTopK, `tie_break`, `dsa_graph_safe`) — https://docs.flashinfer.ai/api/topk.html
- FlashInfer sampling API (`deterministic`, seed/offset tensor) — https://docs.flashinfer.ai/api/sampling.html
- FlashInfer issue #710 (4–6 rejection rounds) — https://github.com/flashinfer-ai/flashinfer/issues/710
- vLLM PR #11394 (bỏ sort, ~20 % overhead) — https://github.com/vllm-project/vllm/pull/11394
- vLLM `gumbel.py` (murmur3 counter noise, log1p series) — https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py
- Qrita, pivot-based top-k/top-p — https://arxiv.org/abs/2602.01518
- AIR TopK, SC'23 — https://dl.acm.org/doi/abs/10.1145/3581784.3607062 ; https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap294.html
- Dr. Top-k, SC'21 — https://arxiv.org/abs/2109.08219
- RadiK, ICS'24 — https://dl.acm.org/doi/fullHtml/10.1145/3650200.3656596
- RTop-K (khảo sát regime của Dr. Top-k/AIR/RadiK) — https://arxiv.org/html/2409.00822v1
- NVIDIA, FlashInfer kernels blog — https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/
- NVIDIA H100 datasheet — https://www.nvidia.com/en-us/data-center/h100/
- So sánh H100/H200/B200/B300 (bandwidth B200) — https://openmetal.io/resources/blog/h100-vs-h200-vs-b200-vs-b300-ai-gpu-server

## Phụ lục: trạng thái triển khai M4/M5 (2026-11-02)

**M4 (split-row + graph).** Code đã có cooperative launch, trao đổi split-state qua
`device_scratch`, barrier `16/S` trong `device_persistent`, plan/workspace contract. Bổ sung test
ở batch mục tiêu (B=32/64/128, V=32768), test tái dùng một persistent buffer qua S=8→2→16→4, và
graph replay với splits do `sampling_plan` chọn. Trên RTX 3050 (16 SM, capacity 64 CTA) các grid
vượt capacity tự skip; H100 (132 SM) chạy B=32 với plan.splits=4 (256 CTA). Không làm
shard-resident smem: Path B hiện đọc lại DRAM ở pass 2, đã ghi nhận là deviation có chủ đích.

**M5 (B=512).** Đã thêm `bench-sm90a`, `--only/--iters/--split-sweep`, p99 trong JSON, case
B=16/64/256, V=120832, p=0.5/0.95/0.99, và `tools/run_sampling_bench.sh` +
`benchmark/compare/hpcops_sampler.py`. Kernel có thêm nhánh P_ONLY: template
`fused_filtered_kernel<in_t, SPLIT, P_ONLY>`; pass 1 dựng histogram mass fixed-point 2048 bin
(overlay count bin) để tìm biên top-p chính xác mà không cần exp toàn hàng/pass 3; pass 2 race
các bin trên biên và tinh chỉnh candidate biên. `cuobjdump` cho P_ONLY=true: REG 64, STACK 0
(general path: STACK 176).

Số đo dev RTX 3050, BF16, V=151936, B=512, median (ms):

| mode | fused | sample() cũ | ghi chú |
|---|---:|---:|---|
| greedy | 0.839 | 30.25 | 185 GB/s hiệu dụng (1 lần đọc) |
| t1 | 2.20 | 37.95 | 2 pass |
| k50 | 2.36 | 50.04 | 2 pass |
| k50p0.9 | 2.82 | 36.64 | |
| p0.5 | 4.76 | — | |
| p0.95 | 6.40 | 58.98 | 2.7× k50 |
| p0.99 | 29.8 | — | race-bound, nucleus ~37k token |
| minp0.05 | 2.06 | 42.29 | |

Đã thử inverse-CDF một uniform/row (mass per record + warp re-read) và **revert**: trên dev nó
chậm hơn race ở p0.95 (7.05 vs 6.40 ms) và p0.99 (28.8 vs 29.8 ms), vì chi phí fixed-point mass
mỗi token xấp xỉ chi phí Philox. Giữ race làm mặc định; p0.99 là trường hợp còn nợ, cần đo lại
trên H100 trước khi quyết định rejection sampling.

**Gate H100 chưa chạy trong phiên này.** Lệnh:

```bash
tools/run_sampling_bench.sh                 # kernel-ops matrix
RUN_HPC_OPS=1 tools/run_sampling_bench.sh   # + hpc-ops baseline V=120832
NCU=1 tools/run_sampling_bench.sh           # + ncu counters
```

So sánh hợp lệ với hpc-ops: V=120832, k=50, p=0.9, temperature; hpc-ops chỉ giữ tối đa 32/64
token và top-p trên ≤64 candidate, nên khác semantics. Gate “B=512 ≥1.3×” chỉ được đóng sau khi
có JSON của cả hai bên trên cùng H100.
