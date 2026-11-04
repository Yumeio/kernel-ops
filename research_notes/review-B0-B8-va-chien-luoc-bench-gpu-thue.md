# Review B0 → B8 và chiến lược benchmark third-party trên GPU thuê

Ngày: 2026-11-02. Nguồn: 7 transcript Claude Code của repo (B4 → B8, P1.x, fused sampler), `ROADMAP.md`,
`registry/capabilities/*.yaml`, `out/*summary.txt`, `benchmark/harness.h`, `benchmark/compare/`.
B0–B3 được làm trước các phiên này nên phần review của chúng dựa vào trạng thái repo, không dựa vào transcript.

---

## Phần 1. Token đã đi đâu (số đo, không phải ước đoán)

| Phiên | Việc | Lượt (turn) | Output | Cache read | Build ~ |
|---|---|---:|---:|---:|---:|
| ee952c7d | chiến lược + B4 + đầu B5 | 559 | 1.50M | 175M | 34 |
| 555074da | P1.5/P1.6/P1.6b | 486 | 1.09M | 156M | 39 |
| 3ac9be87 | B5 | 262 | 0.78M | 92M | 32 |
| c11912a6 | baseline B0–B5, B6 vec.cuh | 346 | 0.37M | 82M | 11 |
| fc2b8c97 | B7 nhịp 1 | 239 | 0.44M | 61M | 16 |
| f87af8f9 | fused sampler M0–M4 | 609 | 1.42M | 278M | 34 |
| 34d2c413 | B8 | 450 | 0.91M | 181M | 9 |
| **Tổng** | | **2 951** | **6.5M** | **~1.03B** | |

Context của phiên B8 tăng theo lượt: 133k token ở lượt 50, 410k ở lượt 200, 587k ở lượt 400.
Mỗi lượt đọc lại toàn bộ context (cache read), nên trung bình mỗi lượt tốn ~400k token đọc.

Quy đổi theo tỉ lệ giá tương đối của API (cache read = 0.1× input, ghi cache 1h = 2×, output = 5×):
**cache read ~65%, output ~21%, ghi cache ~14%**. Đòn bẩy lớn nhất là **độ dài phiên**, không phải độ dài câu trả lời.

Những thứ khác thấy trong transcript:
- `@ROADMAP.md` (297 KB) bị Claude Code đánh dấu `too_large` và **không được nạp**. Gõ `@ROADMAP.md` không tốn gì nhưng cũng không cho gì; phiên nào cũng phải tự đọc lại từng mục (2–7 lần mỗi phiên).
- Kết quả tool lớn nhất là `Read` file 15–50 KB (`test_merge.cc`, `fused_sampler.cu`, `weight_repack.cu` đọc 5–12 lần mỗi phiên).
- Mỗi phiên dựng lại script tạm trong scratchpad (`kenv.bat`, `wcycle.ps1`, `mutate.py`, `p9.py`) rồi mất khi hết phiên.
- 3 phiên (B7, sampler, B8) cùng sửa một working tree: build hỏng giữa chừng, một A/B benchmark bị hỏng vì phiên khác chạy test trên cùng GPU, phải dựng bản copy `work/` trong scratchpad.
- Hook RTK viết lại `grep`/`tail`/heredoc thành cú pháp sai ở git-bash (lỗi lặp lại ở 3 phiên).

## Phần 2. Nâng cấp quy trình để tiết kiệm token (xếp theo hiệu quả)

1. **Một nhịp (hoặc một mốc M) = một phiên.** Cuối nhịp: ghi memory handoff, `/clear`. Khi context vượt ~200k mà chưa xong thì `/compact` kèm trọng tâm. Nếu giữ trung bình ~100k/lượt thay vì ~400k, cache read giảm khoảng 4 lần (ước lượng từ bảng trên).
2. **Thẻ bước thay cho ROADMAP.** Thêm `tools/check_roadmap.py --slice B9`: in các hàng nhịp của bước + §1.1 vòng nhịp + mục CONTRACTS liên quan (~5–10k token). Prompt mở phiên chỉ cần: "làm B9 nhịp 1, đọc slice".
3. **Sửa ROADMAP §2.1 cho đúng máy thật** (build MSVC trên Windows, torch chỉ trong WSL `~/venv-ayaka`, 183 GB/s trần memcpy). Phiên nào cũng phải tự phát hiện lại điều này hoặc đọc memory.
4. **Một script dev cố định** (ví dụ `tools/dev/kops.ps1 build|test|bench|mutate|compare <family>`): gói vcvars + ninja `-j 2` + ccache; in **chỉ** lỗi, dòng `[n/n] Linking`, số test fail và bảng tóm tắt bench; log đầy đủ ghi ra file. Cần chủ repo duyệt thêm vào Phụ lục E.2 (hoặc để ngoài repo).
5. **Git worktree cho mỗi luồng song song + khoá GPU.** Mỗi phiên song song làm ở worktree riêng (Claude Code có `EnterWorktree`); benchmark lấy một lock file và kiểm `nvidia-smi --query-compute-apps` trước khi đo.
6. **Commit theo nhịp** (cần bạn cho phép). Hiện B7 + B8 + fused sampler + ABI 6 nằm chung một đống chưa commit: report luôn `dirty`, nên 30 dòng registry kẹt ở `numerically_validated` và không lên `benchmarked` được.
7. **Mutation test thành danh sách cố định** theo family (file + chuỗi cần đổi + test phải đỏ), chạy nền một lần cuối nhịp, thay vì viết lại `mutate.py` mỗi phiên.
8. **Đọc theo vùng.** Grep ký hiệu trước, `Read` với offset/limit; không đọc lại cả file 2 000+ dòng.
9. **Kiểm thời gian compile sớm.** Ba lần nổ compile time (B7 >8 phút, B8 183 s, sampler 139 s) đều chỉ lộ ra sau khi viết xong. Script ở mục 4 in thời gian mỗi TU, cảnh báo khi > 60 s (trần 90 s).
10. **Phiên cơ học ("chạy lại bench", "tiếp tục") dùng effort/model thấp hơn**; có lượt dùng tới 10k token thinking chỉ để đọc log.

## Phần 3. Review từng bước

Registry hiện tại: 104 dòng `benchmarked`, 30 dòng `numerically_validated` (gemm 18, attention 9, quantization 2, moe 1).

| Bước | Hiện trạng | Vấn đề / nợ thấy được | Nâng cấp lần sau |
|---|---|---|---|
| B0 hạ tầng | harness, registry, check_roadmap, preset | Harness bench **không xoá L2** giữa các lần đo và không có chế độ CUDA graph; không có kiểm tra "GB/s > trần đo được"; `env` thiếu clock/nhiệt/nguồn | Thêm `--flush-l2` (hoặc xoay buffer > 2× L2), `--graph`, sanity bound tự động, ghi `clocks.sm`, `power.draw`, `clocks_event_reasons` vào JSON |
| B1 elementwise (32) | benchmarked; so torch `builtin` + `hf` | Op copy-like đã ở trần 183 GB/s, ratio so torch builtin ~1× là đúng | Đối thủ thật: FlashInfer `rmsnorm`/`fused_add_rmsnorm`/rope, vLLM `_C`, Liger, `torch.compile` |
| B2 KV cache (20) | 23 dòng benchmarked | Nhiều case nhỏ hơn L2 của GPU lớn (H100 50 MB): số GB/s sẽ ảo nếu không xoá L2 | So với vLLM `reshape_and_cache_flash`, FlashInfer `append_paged_kv_cache` |
| B3 quant | geomean 4.75× vs torch | Baseline FP4 phải ép qua FP8 (bảo thủ) | So với vLLM `scaled_fp8_quant`, `per_token_group_quant_fp8`, FlashInfer `fp4_quantize` |
| B4 weight prep | geomean 8.23× | Marlin repack so với bản PyTorch thuần (hàng chục ×) chứ không phải op CUDA của vLLM; mục tiêu "≥3×" không thể đạt với op copy-like | Tính trần `torch_time / (bytes / ceiling)` **trước** khi đặt mục tiêu; so với `gptq_marlin_repack`/`awq_marlin_repack` của vLLM |
| B5 GEMV | dense 1.01–1.12× (đã ở trần), quant 8–95× | 8–95× là so với dequant eager + `F.linear` (đối thủ yếu); AWQ/FP4 M=16 chỉ 113–131 GB/s; NF4 M≥4 tụt | Đối thủ thật: Marlin của vLLM, llama.cpp `mmvq` (`test-backend-ops perf`), bitsandbytes NF4, exllamav2 |
| B6 routing/sampling | geomean 3.77×; fused sampler M1–M4 xong | Byte count từng sai (moe.sum đếm đôi; chain 241 GB/s > trần); test cũ ≤ 9 token chỉ chạy warp 0 | Sanity bound tự động (B0); case test có số token ≥ số warp × 32; so với FlashInfer sampling, vLLM sampler, sgl-kernel `moe_fused_gate`, vLLM `moe_align_block_size` |
| B7 mma.sync | nhịp 1/10, 12.4 TFLOP/s = 0.86× cuBLAS ở M lớn | Còn 9 nhịp; tài liệu §1.6 chưa có các hàng B7; compile time từng nổ | cuBLAS là thước đo trên mọi GPU; mục tiêu 0.9× trước khi mở nhịp mới |
| B8 attention (7) | đủ thân, numerically_validated; prefill 8.9–11.5 TFLOP/s, decode 187 GB/s | Decode TC FP8 bị giới hạn bởi chuyển đổi (73 GB/s); `test_merge.cc` gần trần 2 500 dòng; **lượt mutation cuối phiên B8 chưa có kết quả** | Kiểm lại mutation; so với FlashAttention-2 (SM80/86/89), FlashInfer prefill/decode, vLLM `paged_attention_v2` |

Bài học chung:
- **Đặt mục tiêu từ trần phần cứng, không từ tỉ lệ so với torch.** "≥3× baseline" vô nghĩa cho op đã ở trần, và quá dễ khi baseline là dequant eager.
- **Kế hoạch dán vào hay chẩn đoán sai** (P1.6b: không phải `wf[32]` mà là LSU pipe; sampler: histogram khối lượng bằng 0 khi x < 41.8). Giữ bước M0 "đo / ncu trước khi sửa".
- **Test phải chạy hết các warp/CTA**: shape test nhỏ che lỗi (B6).
- **Đo trên laptop**: cắm sạc, xen kẽ A/B, không có phiên khác dùng GPU.

## Phần 4. Chiến lược benchmark third-party trên GPU thuê

### 4.1 Mục tiêu

Mỗi kernel kops cần trả lời được ba câu trên từng exact SM:
1. **Đúng không?** Sai số so với oracle fp64, cùng thước đo cho kops và cho từng thư viện; có tất định không.
2. **Nhanh tới đâu so với trần?** % băng thông DRAM hoặc % TFLOP/s, với trần **đo trên chính máy thuê** (memcpy D2D, cuBLAS GEMM), không lấy số spec.
3. **So với thư viện tốt nhất hiện có thì sao?** Tỉ lệ so với *best-of-3rd* cho từng case, không phải so với torch eager.

Thêm một câu cho máy thuê: **máy này có đúng như quảng cáo không?** (bước hiệu chuẩn 4.4).

### 4.2 Làm miễn phí trước trên RTX 3050 (SM86)

FlashInfer, vLLM `_C`, FlashAttention-2 và llama.cpp đều chạy trên SM86. Cài chúng vào venv WSL (cùng GPU, không chạy song song với bench C++) và chạy toàn bộ pipeline ở máy nhà trước khi thuê. Việc này:
- cho số so sánh third-party đầu tiên mà không tốn tiền thuê;
- bắt gần hết lỗi script (adapter sai layout, case thiếu, JSON lệch) trước khi trả tiền theo giờ.

Theo ROADMAP §0.7 mục 7, việc này nằm trong `benchmark/compare/` (công cụ dev, không build, không ship) nên không phá quy tắc C++-only. Số third-party là **so sánh**, không phải evidence của registry.

### 4.3 Bộ công cụ cần có

| Thành phần | Nơi đặt | Nội dung |
|---|---|---|
| Nâng harness C++ | `benchmark/harness.h` | `--flush-l2`, `--graph` (capture N launch rồi replay), `--rounds 3`, sanity bound, ghi clock/nhiệt/nguồn/ECC/MIG vào `env` |
| Runner third-party | `benchmark/compare/thirdparty.py` | đọc `--dump-cases`, mỗi thư viện là một adapter `setup(case) -> fn`, `output(case) -> tensor`; kiểm đúng trước, rồi mới đo bằng **cùng phương pháp** (CUDA graph, nhóm 20 launch, xoá L2, median/p10/p90); xuất JSON cùng schema + trường `impl` |
| So sánh N chiều | `benchmark/compare/compare_results.py` | nhận nhiều JSON, cột: kops, best-of-3rd (tên thư viện), tỉ lệ, % trần, sai số; geomean theo family; danh sách thua |
| Hiệu chuẩn máy | trong runner (`--calibrate`) | memcpy D2D, cuBLAS bf16/fp8 GEMM lớn, chạy nóng 10 phút để xem throttle |
| Image cố định | Dockerfile ngoài repo | CUDA 12.9 + torch + vLLM + FlashInfer + flash-attn + llama.cpp `3rd/llama.cpp@0ec4027`; pin phiên bản; build sẵn trước khi thuê |

Ma trận đối thủ theo family:

| Family (bước) | Đối thủ |
|---|---|
| norm, activation, position (B1) | FlashInfer, vLLM `_C`, sgl-kernel, Liger, `torch.compile` |
| kvcache (B2) | vLLM `reshape_and_cache(_flash)`, FlashInfer `append_paged_kv_cache` |
| quantization (B3, B4) | vLLM fp8/int8 quant + Marlin repack, FlashInfer fp4 quant, TRT-LLM (nếu build được) |
| gemm (B5, B7) | cuBLAS/cuBLASLt (thước đo chính), vLLM Marlin / Machete (SM90), llama.cpp `mmvq`/`mmq`, bitsandbytes, DeepGEMM (SM90/100, FP8) |
| moe (B6) | vLLM `moe_align_block_size`, `topk_softmax`, sgl-kernel, fused_moe Triton |
| sampling (B6, fused) | FlashInfer sampling, vLLM sampler, hpc-ops (chỉ SM90) |
| attention (B8) | FlashAttention-2 (SM80/86/89), FA3 (SM90), FlashInfer prefill/decode, vLLM `paged_attention_v2` |
| state (B9, sau này) | flash-linear-attention, `causal_conv1d`, `mamba_ssm` |

### 4.4 Quy trình một lần thuê (mục tiêu ≤ 2–3 giờ mỗi loại GPU)

0. **Ở nhà:** commit sạch (report phải có `git.dirty = false`, theo ROADMAP §2.5); image đã build; pipeline đã chạy xanh trên SM86.
1. **Dấu vân tay máy** (2 phút): `nvidia-smi -q` (tên chính xác: H100 **SXM** hay **PCIe**, A100 40 hay 80 GB), driver, power limit, ECC, MIG, NVLink topo, CPU, có tiến trình khác trên GPU không.
2. **Hiệu chuẩn** (10–15 phút): memcpy D2D và cuBLAS GEMM so với spec của đúng biến thể; chạy nóng 10 phút, xem `clocks_event_reasons`. **Thấp hơn ~85% spec hoặc có throttle thì đổi máy** trước khi tiêu thêm giờ. Đây cũng là cách so chất lượng giữa các nhà cho thuê.
3. **Đúng trên exact SM** (handoff kit §2.5): `ctest -L "oracle|contract"`, `-L "arch|system"`, `compute-sanitizer` cho twin.
4. **Bench kops** `--flush-l2 --graph`, 3 vòng.
5. **Bench third-party**, 3 vòng, xen kẽ với bước 4 theo family (không chạy hết kops rồi mới tới 3rd).
6. **ncu** cho 3–5 case thua nặng nhất (để sửa ở nhà).
7. **Đóng gói** `out/<arch>-<gpu>-<date>/` + `sha256`, tải về, **xoá instance**. Ghi kết quả sau mỗi family để instance spot bị thu hồi cũng không mất hết.

Luật thống kê: median của 3 vòng xen kẽ; p90/p10 > 1.05 thì đo lại case đó; trong container thường không khoá được clock (`nvidia-smi -lgc` cần root) nên luôn ghi clock kèm số đo.

### 4.5 Thứ tự thuê (theo giá trị / chi phí)

| Đợt | GPU | Exact SM | Vì sao | Khi nào |
|---|---|---|---|---|
| 1 | A100 (40/80 GB) | sm80 | Rẻ nhất trong nhóm datacenter; nhận evidence sm80 cho toàn bộ kernel generic B0–B8; đối thủ đầy đủ | ngay khi 4.2 xanh |
| 2 | L4 hoặc L40S / RTX 4090 | sm89 | Nhánh FP8 `cvt` thật (B3, B5, KV FP8 của B8) chưa chạy được ở nhà | sau đợt 1 |
| 3 | H100 SXM | sm90a | Chỉ đáng khi B11 (wgmma/TMA) có thân; trước đó chỉ đo kernel generic so với FA3/FlashInfer Hopper để biết khoảng cách | sau B11 |
| 4 | RTX 5090 / RTX PRO 6000 | sm120a | `mma.sync` block-scaled, FP4 | sau B13 |
| 5 | B200 | sm100a | `tcgen05`/TMEM | cuối cùng |

Giá thuê thay đổi theo nhà cung cấp và thời điểm; xem giá lúc thuê, ưu tiên spot cho đợt 1–2.

### 4.6 Báo cáo

Mỗi đợt một file `reports/<arch>-<gpu>-<date>.md` gồm: dấu vân tay + hiệu chuẩn; bảng mỗi case (kops ms, best-of-3rd ms + tên, tỉ lệ, % trần, sai số max-abs/max-rel so với fp64, tất định có/không); geomean theo family; danh sách case thua kèm chẩn đoán ncu; dòng registry được nâng (`numerically_validated`/`benchmarked` cho đúng `arch_tag`).

## Phần 5. Quyết định của chủ repo (2026-11-02) và hệ quả

1. **Không commit theo nhịp.** Report vẫn `dirty = 1`, nên số đo trên GPU thuê chỉ là **so sánh**, không nâng được dòng registry (ROADMAP §2.5 đòi `git.dirty = false`). Lên máy thuê bằng tar của working tree (bỏ `build/`, `out/`, `3rd/`), không `git clone`.
2. **Duyệt script dev + `--slice`.** Đã làm: `tools/dev/kops.py`, `tools/dev/mutations.json` (75 mutation: quantization 38, gemm 12, sampling 17, merge/attention 8; 11 mutation cũ không còn khớp code nên bỏ), `check_roadmap.py --slice`, ghi vào Phụ lục E.2.
3. **Cài thư viện so sánh vào WSL.** Venv riêng `~/venv-3rd` (không đụng `~/venv-ayaka`): vLLM 0.33.0, torch 2.14.0+cu132, FlashInfer 0.7.2.post1, `vllm.vllm_flash_attn` (FA2). Không có `flash_attn` gốc và `sgl_kernel`. FlashInfer cần kernel dựng sẵn (`vllm download-kernels`) vì WSL không có nvcc.
4. **Ngân sách thuê: 6 USD.** Đủ cho khoảng 3 giờ A100 cộng 1–2 giờ L4/L40S ở chợ GPU giá rẻ, theo giá tham khảo; kiểm lại giá lúc thuê, chọn nơi tính tiền theo phút, tắt máy ngay khi xong. **Chưa thuê** cho tới khi 4.2 xanh ở nhà (harness `--flush-l2`/`--graph` + `benchmark/compare/thirdparty.py`), vì mọi lỗi script trên máy thuê đều tốn tiền.

## Phần 6. Thuê GPU cho B0 → B16: implement + so với vLLM, SGLang, TensorRT-LLM, hpc-ops

### 6.1 Nguyên tắc

1. **Viết code ở nhà, thuê GPU để chạy theo lô.** Ở SM86 làm được oracle, thân SM80-class, twin (T2), compile mọi exact SM, SASS, `--dryrun`. Máy thuê chỉ chạy script: test, bench, ncu, rồi tải report về. Ngoại lệ: **sprint tuning** của B11/B13, vì twin không đo được hiệu năng `wgmma`/`tcgen05`/TMA.
2. **Gom theo exact SM, không theo bước.** Một lần thuê sm80 chạy mọi bước đã có thân; thuê lại chỉ khi có bước mới cần SM đó.
3. **Compile không cần GPU.** kops cho `sm_90a`/`sm_100a`/`sm_120a`, hpc-ops, sgl-kernel build từ nguồn: build sẵn trên máy CPU (hoặc image dựng sẵn), không build trên đồng hồ GPU.
4. **Mỗi thư viện một môi trường riêng**, vì mỗi cái pin một bản torch khác: `venv-3rd`/image vLLM (kèm FlashInfer, `vllm_flash_attn` FA2/FA3), SGLang (`sgl-kernel`), TensorRT-LLM (container NGC, rất nặng), hpc-ops (build từ `3rd/hpc-ops`). Tất cả xuất cùng schema JSON theo case id; `compare_results.py` gộp lại và chọn best-of-3rd.
5. **Đường rẻ tới kernel TRT-LLM:** nhiều kernel trtllm-gen (FMHA decode, MoE FP4, GEMM FP4) đã có trong FlashInfer (`trtllm_*` backend, chủ yếu SM100). Chỉ kéo container TRT-LLM khi cần op không có trong FlashInfer.
6. **hpc-ops chỉ SM90** (H20/H100): so với nó ở các lần thuê sm90a.
7. **Hẹn giờ tắt máy** ngay khi vào (`sudo shutdown -h +<phút>`), ghi report sau mỗi family.
8. **Phiên Claude Code trên máy thuê** chỉ dùng cho sprint tuning có mục tiêu và timebox rõ; GPU vẫn tính tiền khi model đang suy nghĩ.

### 6.2 Từng bước

| Bước | Ở nhà (SM86) | Cần thuê | vLLM | SGLang (`sgl-kernel`) | TensorRT-LLM | hpc-ops | Khác |
|---|---|---|---|---|---|---|---|
| B0 nền | đủ | lô sm80/sm90 chung | — | — | — | — | torch |
| B1 elementwise/norm/rope | đủ | lô chung; sm120a cho vector 256-bit | `rms_norm`, `fused_add_rms_norm`, `rotary_embedding`, `silu_and_mul` | norm, rope, act | (plugin, khó gọi lẻ) | norm (SM90) | FlashInfer, Liger |
| B2 KV cache | đủ | sm89 (FP8 `cvt`) | `reshape_and_cache_flash` | store KV | — | — | FlashInfer `append_paged_kv_cache` |
| B3 quant runtime | nhánh software | sm89 (FP8), sm100a/sm120a (FP4 `cvt`) | `scaled_fp8_quant`, `per_token_group_quant` | `sgl_per_token_group_quant_fp8` | `fp4_quantize` | FP8 quant (SM90) | FlashInfer `fp4_quantize` |
| B4 weight prep | đủ | ít giá trị (chạy lúc load) | `gptq/awq_marlin_repack` | — | shuffle FP4 | — | — |
| B5 GEMV | đủ, trừ FP8 `cvt` | sm80, sm89 | Marlin `gptq_marlin_gemm` | — | weight-only GEMV (fpA_intB) | GEMM FP8 (SM90) | llama.cpp `mmvq`, cuBLAS |
| B6 routing/sampling | đủ | lô chung; sm90 để so hpc-ops | `moe_align_block_size`, `topk_softmax`, sampler | `moe_fused_gate`, top-k | routing `noaux_tc` | sampler, MoE (SM90) | FlashInfer sampling |
| B7 `mma.sync` | đủ, trừ FP8 mma | sm80, sm89 (FP8), sm120a | `cutlass_scaled_mm`, Machete (SM90) | `fp8_scaled_mm` | cutlass GEMM | — | **cuBLAS** (thước đo chính) |
| B8 attention native | đủ (FP8 KV software) | sm80, sm89, sm120a | FA2/FA3 (`vllm_flash_attn`), paged attn | qua FlashInfer/FA3 | XQA/FMHA (qua FlashInfer) | decode/prefill (SM90) | FlashInfer |
| B9 recurrent | đủ, chuỗi dài không vừa 4 GB | sm80 40–80 GB | mamba/conv1d ops | ops Mamba | — | — | FLA, `causal_conv1d`, `mamba_ssm` |
| B10 speculative | đủ | lô chung | rejection sampler | `tree_speculative_sampling` | — | — | FlashInfer chain speculative |
| B11 Hopper | compile + SASS + twin | **H100: sprint tuning** | FA3, Machete | FP8 blockwise GEMM | trtllm GEMM/FMHA | **chính sân nhà** | DeepGEMM, FlashMLA |
| B12 comm | rank ảo 1 GPU | **2× (rồi 8×) GPU NVLink** | custom allreduce | custom allreduce | allreduce + norm fusion | comm-compute fusion | NCCL, DeepEP |
| B13 Blackwell | compile + SASS + twin | **sm120a trước (RTX 5090), sm100a (B200)** | cutlass FP4/FP8 | FP4 GEMM | **trtllm-gen** (mạnh nhất trên Blackwell) | — | FlashInfer, DeepGEMM SM100 |
| B14 sparse attention | qua thân SM80-class | sm90a, sm100a | DSA/MLA ops | DSA/MLA | — | — | FlashMLA sparse |
| B15 DeepSeek-V4/Kimi K3 | generic | sm90a, sm100a | — | — | — | — | DeepGEMM mHC (nguồn chính thức) |
| B16 megakernel | trên mainloop B7/B8 | sm90a, sm100a, multi | — | — | — | — | DeepGEMM Mega MoE |

### 6.3 Lịch thuê

Giá chỉ là tham khảo ở chợ GPU giá rẻ, phải kiểm lại lúc thuê. Mốc giờ là ước lượng cho một lô có script sẵn.

| Đợt | Khi nào | GPU | Giờ | Chi phí ~ | Bao phủ |
|---|---|---|---:|---:|---|
| 1 | bây giờ, sau khi 4.2 xanh | A100 + L40S/L4 | 3 + 2 | ~6 USD | B0–B10: so với vLLM, SGLang, FlashInfer; FP8 sm89 |
| 2a | trước khi viết B11 | H100 | 2 | ~5 USD | đo hpc-ops, FA3, DeepGEMM, FlashMLA để đặt mục tiêu B11 |
| 2b | B11 twin xanh | H100 | 3–5 sprint × 4 | ~40–60 USD | tuning B11, so với hpc-ops/DeepGEMM/FA3 |
| 3 | B7/B8 SM120 + B13 phần SM120 | RTX 5090 | 10–20 | ~10–20 USD | sm120a, so với trtllm-gen/FlashInfer FP4 |
| 4 | B12 | 2× H100/A100 NVLink, cuối cùng 8× | 4 + 2 | ~20–40 USD | allreduce, EP, comm-GEMM |
| 5 | B13 SM100 → B16 | B200 | 3–4 sprint × 4 | ~60–100 USD | `tcgen05`, so với trtllm-gen, DeepGEMM SM100 |
| 6 | cuối lộ trình | mỗi SM một lô | 1–2 mỗi SM | ~20–30 USD | chạy lại toàn bộ để có số cuối |

Tổng cả lộ trình khoảng **150–250 USD**. Nếu sprint SM120 kéo dài (hàng trăm giờ), mua một card RTX 50-series (SM120) có thể rẻ hơn thuê; quyết định sau đợt 3.

### 6.4 Việc làm miễn phí ngay ở nhà

- Thêm venv `~/venv-sgl` (sgl-kernel) cạnh `~/venv-3rd`: phần lớn op norm/rope/MoE/sampling của SGLang chạy trên SM86.
- Mọi so sánh của B0–B10 với vLLM/FlashInfer/SGLang chạy trước ở SM86; đợt 1 chỉ xác nhận lại trên sm80/sm89.

### 6.5 Lỗi đã thấy khi chạy thử (2026-11-02)

Phát hiện khi chạy thử `kops.py bench norm`: 5 case của `bench_norm` (`layer_norm` bf16/fp32, `fused_add_layer_norm`, `fused_add_rms_norm_out`, `gated_rms_norm`) báo 216–250 GB/s, vượt trần lý thuyết 192 GB/s. Tensor 8–32 MB lớn hơn L2 2 MB nên đây là byte count đếm thừa, cùng loại lỗi với `moe.sum` ở B6.
