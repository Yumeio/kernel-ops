// =============================================================================
// kernel-ops | csrc/mega/mega.cu                                Catalog: E04-E08
// B16: persistent / megakernel family.
//
// This translation unit implements the B16 device bodies and the host plan/run API:
//
//   mega_mlp_kernel             E04  norm -> W13 -> GLU -> W2 -> residual
//   mega_attention_block_kernel E05  QKV -> QK-norm/RoPE -> KV append -> decode -> O-proj
//   mega_moe_local_kernel       E06  routed W13 -> GLU -> W2 -> combine (routes precomputed)
//   mega_moe_ep_kernel          E07  same body over the EP symmetric buffer (single rank only)
//   mega_decoder_layer_kernel   E08  attention block + dense MLP / local MoE in one DAG
//
// Scheduling model.  Every kernel is a persistent CTA loop over plan.queue tasks
// (exec::task_queue_next).  Cross-stage dependencies use the epoch-tagged counters
// of execution/persistent.cuh (task_wait / task_signal), never a grid barrier:
//
//   MLP:      gate/up tile (row, n-tile)  --signal[row]-->  down tile (row, h-tile)
//   attention append (row)                --counter 0---->  decode+O-proj (row)
//   decoder:  append (0) -> decode (1) -> FFN (2)      (counter 0, then counter 1)
//
// The task order is monotonic and every waiter depends only on tasks with a lower
// task index, so the progress rule of persistent.cuh holds for any grid size,
// including one CTA (no deadlock when few SMs are available).
//
// Graph safety.  task_queue_next uses a monotonic cursor and the counters are
// epoch-tagged.  prepare() zeroes that carve once (outside capture) and every
// megakernel resets it on exit from its last CTA (mega_finish), so a direct run and
// a CUDA-graph replay both start from a clean queue.  num_device_launches is 1.
//
// Scope of this body (reported in the B16 handoff):
//   * BF16 activations and BF16 weights, K-major (contiguous [out, in]) weights.
//   * Dense MLP, QKV and O-projection tiles run the B7 mma.sync mainloop
//     (cp.async 128-byte stage rows, ldmatrix.x4, m16n8k16 fp32 accumulators).
//     One CTA is 16 warps split into four K-groups; each group consumes one k16
//     step of a stage and the fp32 partials are summed in group order after the
//     loop.  GLU / residual / norm are fused into the tile epilogues.
//   * The MoE bodies and the arbitrary-K fallback still use the original fp32
//     SIMT dot product; K not a multiple of 64 selects that fallback.
//   * mega_moe_local / mega_moe_ep require precomputed routes (topk_ids/weights).
//     Fused router + top-k is a separate capability: MegaMoELocalParams carries
//     RouterParams but no RouterScore, so the score function is not declared here.
//   * mega_moe_ep supports num_ranks == 1 (virtual rank) only; multi-rank peer
//     dispatch/combine is not implemented.
//   * attention: native BF16 KV, kNHD, block-table or CSR page table, one request
//     (q_len = m).  FP8/HND/MLA/FP4 KV codecs return kUnsupportedLayout.
// =============================================================================
#include <atomic>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/device.cuh"
#include "csrc/common/device_impl.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/moe/routing_kernels.cuh"
#include "csrc/communication/allreduce_kernels.cuh"
#include "csrc/communication/ep_kernels.cuh"
#include "csrc/gemm/gemm_params.cuh"
#include "csrc/execution/persistent.cuh"
#include "csrc/execution/launch.h"
#include "csrc/mega/mega_plan.cuh"
#include "csrc/mega/mega_kernels.cuh"
#include "csrc/mega/ops.h"

namespace kops::mega {
namespace {

using Bf16 = __nv_bfloat16;
// The B16 body only needs SM80-class instructions (warp shuffles, fp32 FMA,
// bf16 convert).  Compile-time tag = the policy, not a claim about a narrower SM.
using MegaArch = Sm80;

constexpr int kMegaThreads = 512;
constexpr int kMegaWarps = kMegaThreads / 32;
// MLP mainloop tile: one m16 tensor-core row tile covers every decode row of this
// body (M <= 16 per tile; larger M is tiled over tile_m).  A stage holds 64 bytes
// of every operand row, i.e. BK = 32 bf16 K elements, and four stages of the
// weight ring keep cp.async busy across the whole K loop.
constexpr int kMlpBM = 16;    // rows per task (one m16 atom)
constexpr int kMlpBN1 = 64;   // GLU-output columns per gate/up task
constexpr int kMlpBN2 = 64;   // hidden columns per down task
constexpr int kMlpBK = 64;    // K elements per stage (128-byte operand rows)
constexpr int kMlpStages = 3; // cp.async ring depth
constexpr int kAttnBM = 16;   // rows per attention projection tile
constexpr int kAttnBN = 64;   // QKV / O-projection columns per task
constexpr int kAttnStages = 3;
constexpr int kMegaBN = 64;   // MoE tile width (metadata only, body loops whole rows)

// ---- small host helpers ------------------------------------------------------
size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

uint64_t fnv1a64(const char* s) {
  uint64_t h = 0x100000001b3ull;
  for (const char* p = s; *p != '\0'; ++p) {
    h ^= static_cast<uint8_t>(*p);
    h *= 0x100000001b3ull;
  }
  return h == 0 ? 1 : h;
}

// Runtime record of the host-side slot addresses / launch geometry of one plan.
// MegaPlan itself is the public ABI and cannot grow host-only fields without a
// version bump; this side table is keyed by the caller's plan address.  prepare()
// overwrites the entry, so re-using a stack slot for a new plan is safe.
struct HostPlan {
  MegaKind kind = MegaKind::kMLP;
  int32_t max_m = 0;
  int32_t threads = kMegaThreads;
  int32_t ctas = 0;
  int32_t dynamic_smem = 0;
  int32_t num_tasks = 0;
  int32_t num_counters = 0;
  uint32_t* cursor = nullptr;
  uint32_t* counters = nullptr;
  void* slots[4] = {nullptr, nullptr, nullptr, nullptr};
  size_t slot_bytes[4] = {0, 0, 0, 0};
  bool stream_bound = false;
  kops_stream_t bound_stream = nullptr;
  uint32_t generation = 0;
};

std::mutex g_plan_mutex;
std::unordered_map<const MegaPlan*, HostPlan> g_host_plans;
std::atomic<uint32_t> g_plan_generation{1};

struct Layout {
  int32_t m = 0;
  int32_t hidden = 0;
  int32_t intermediate = 0;
  int32_t num_tasks = 0;
  int32_t num_counters = 0;
  int32_t num_slots = 0;
  size_t slot_bytes[4] = {0, 0, 0, 0};
  int32_t dynamic_smem = 0;
};

Status bad(const char* what, const char* detail) {
  return detail::set_error(Status::kInvalidArgument, "mega: %s: %s", what, detail);
}

bool bf16_weights(const QuantScheme& q) {
  return (q.storage_dtype == DType::kInvalid || q.storage_dtype == DType::kBFloat16) &&
         q.granularity == ScaleGranularity::kNone && !q.has_zero_point &&
         q.packing == WeightPacking::kNone;
}

Status check_arch(const LaunchContext& ctx) {
  if (ctx.backend == BackendKind::kHip || target_is_amd(ctx.arch_id)) {
    return detail::set_error(Status::kUnsupportedBackend, "mega: CUDA only");
  }
  if (ctx.arch_id != 0 && ctx.arch_id < KOPS_TARGET_SM80) {
    return detail::set_error(Status::kUnsupportedArch, "mega: SM80+ required, got arch %d",
                             ctx.arch_id);
  }
  return Status::kSuccess;
}

// ---- parameter validation ----------------------------------------------------
Status validate_mlp(const MegaMLPParams* p) {
  if (p == nullptr) return bad("mlp", "null params");
  if (p->x == nullptr || p->w13 == nullptr || p->w2 == nullptr || p->out == nullptr) {
    return bad("mlp", "x / w13 / w2 / out must be non-null");
  }
  if (p->m <= 0 || p->hidden_size <= 0 || p->intermediate_size <= 0) {
    return bad("mlp", "m / hidden_size / intermediate_size must be positive");
  }
  if (p->ldx != 0 && p->ldx < p->hidden_size) return bad("mlp", "ldx < hidden_size");
  if (p->ldo != 0 && p->ldo < p->hidden_size) return bad("mlp", "ldo < hidden_size");
  if (!bf16_weights(p->w_quant) || p->w13_scale != nullptr || p->w2_scale != nullptr) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mega.mlp: BF16 x BF16 unscaled weights only in this body");
  }
  return Status::kSuccess;
}

Status validate_attention(const MegaAttentionBlockParams* p) {
  if (p == nullptr) return bad("attention", "null params");
  if (p->hidden == nullptr || p->w_qkv == nullptr || p->w_o == nullptr || p->out == nullptr) {
    return bad("attention", "hidden / w_qkv / w_o / out must be non-null");
  }
  if (p->m <= 0 || p->hidden_size <= 0 || p->head_dim <= 0) {
    return bad("attention", "m / hidden_size / head_dim must be positive");
  }
  if (p->num_q_heads <= 0 || p->num_kv_heads <= 0 ||
      p->num_q_heads % p->num_kv_heads != 0) {
    return bad("attention", "num_q_heads must be a positive multiple of num_kv_heads");
  }
  if (p->head_dim > 256 || (p->head_dim & 1) != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mega.attention: head_dim must be even and <= 256, got %d",
                             p->head_dim);
  }
  if (p->positions == nullptr) return bad("attention", "positions must be non-null");
  if (p->slot_mapping == nullptr) return bad("attention", "slot_mapping must be non-null");
  if (p->rot_dim < 0 || p->rot_dim > p->head_dim || (p->rot_dim & 1) != 0 ||
      p->rot_offset < 0 || p->rot_offset + p->rot_dim > p->head_dim) {
    return bad("attention", "rot_dim / rot_offset outside the head");
  }
  if (p->rot_dim > 0 && p->cos_sin_cache == nullptr) {
    return bad("attention", "cos_sin_cache required when rot_dim > 0");
  }
  if (p->mod.window_right >= 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mega.attention: mod.window_right is not implemented");
  }
  if (p->mod.chunk_size > 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mega.attention: mod.chunk_size is not implemented");
  }
  const kv::PagedKVDesc& d = p->kv;
  if (d.k_data == nullptr || d.v_data == nullptr) return bad("attention", "KV data null");
  if (d.layout != kv::KVLayout::kNHD) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "mega.attention: only KVLayout::kNHD is implemented");
  }
  if (d.entry_bytes != 0 || d.stride_unit != StrideUnit::kElement) {
    return detail::set_error(Status::kUnsupportedLayout,
                             "mega.attention: only native element-stride KV is implemented");
  }
  if (d.k_dtype != DType::kBFloat16 || d.v_dtype != DType::kBFloat16) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mega.attention: only BF16 KV payloads are implemented");
  }
  if (d.page_size <= 0 || d.num_kv_heads != p->num_kv_heads ||
      d.head_dim_k != p->head_dim || d.head_dim_v != p->head_dim) {
    return bad("attention", "KV descriptor heads/dims disagree with the block params");
  }
  if (d.table_format == kv::PageTableFormat::kBlockTable) {
    if (d.block_table == nullptr || d.block_table_stride <= 0 || d.seq_lens == nullptr) {
      return bad("attention", "block-table KV needs block_table / stride / seq_lens");
    }
  } else if (d.table_format == kv::PageTableFormat::kCSR) {
    if (d.kv_indptr == nullptr || d.kv_page_indices == nullptr || d.seq_lens == nullptr) {
      return bad("attention", "CSR KV needs kv_indptr / kv_page_indices / seq_lens");
    }
  } else {
    return detail::set_error(Status::kUnsupportedLayout,
                             "mega.attention: kSlotMapping is a write-only page table");
  }
  return Status::kSuccess;
}

Status validate_moe(const MegaMoELocalParams* p, bool require_routes) {
  if (p == nullptr) return bad("moe", "null params");
  if (p->hidden == nullptr || p->w13 == nullptr || p->w2 == nullptr || p->out == nullptr) {
    return bad("moe", "hidden / w13 / w2 / out must be non-null");
  }
  if (p->m <= 0 || p->hidden_size <= 0 || p->intermediate_size <= 0 ||
      p->num_local_experts <= 0 || p->topk <= 0) {
    return bad("moe", "m / hidden / intermediate / num_local_experts / topk must be positive");
  }
  if (p->latent_dim != 0) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mega.moe: latent_dim != 0 (LatentMoE) is follow-on work");
  }
  if (!bf16_weights(p->w_quant) || p->w13_scale != nullptr || p->w2_scale != nullptr) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mega.moe: BF16 x BF16 unscaled weights only in this body");
  }
  if (require_routes && (p->topk_ids == nullptr || p->topk_weights == nullptr)) {
    return detail::set_error(
        Status::kUnsupportedBackend,
        "mega.moe: fused router is a separate capability; precompute topk_ids/topk_weights");
  }
  return Status::kSuccess;
}

Status validate_ep(const MegaMoEEPParams* p) {
  Status st = validate_moe(&p->local, true);
  if (st != Status::kSuccess) return st;
  if (p->local.m > p->max_tokens_per_rank) {
    return bad("moe_ep", "m exceeds max_tokens_per_rank");
  }
  if (p->ep.num_ranks != 1 || p->ep.rank != 0) {
    return detail::set_error(Status::kUnsupportedBackend,
                             "mega.moe_ep: only num_ranks == 1 (virtual rank) is implemented");
  }
  if (p->ep.nvl_buffers[0] == nullptr) return bad("moe_ep", "nvl_buffers[0] is null");
  const size_t need = static_cast<size_t>(p->local.m) * p->local.hidden_size * sizeof(Bf16);
  if (p->ep.nvl_bytes < need) return bad("moe_ep", "nvl_bytes smaller than m * hidden * 2");
  if (p->payload_quant != ActQuantMode::kNone) {
    return detail::set_error(Status::kUnsupportedDtype,
                             "mega.moe_ep: only BF16 payloads (payload_quant == kNone)");
  }
  return Status::kSuccess;
}

// ---- workspace layout --------------------------------------------------------
Status layout_for(MegaKind kind, const void* params, const LaunchContext& ctx, Layout* out) {
  if (out == nullptr) return bad("layout", "null output");
  Status st = check_arch(ctx);
  if (st != Status::kSuccess) return st;
  Layout L;
  switch (kind) {
    case MegaKind::kMLP: {
      const auto* p = static_cast<const MegaMLPParams*>(params);
      st = validate_mlp(p);
      if (st != Status::kSuccess) return st;
      const int mt = (p->m + kMlpBM - 1) / kMlpBM;
      const int n1 = (p->intermediate_size + kMlpBN1 - 1) / kMlpBN1;
      const int n2 = (p->hidden_size + kMlpBN2 - 1) / kMlpBN2;
      L.m = p->m;
      L.hidden = p->hidden_size;
      L.intermediate = p->intermediate_size;
      L.num_tasks = mt * (n1 + n2);
      L.num_counters = mt;
      // slot 0: intermediate; slot 1: pre-norm scratch (unused by this kind)
      L.slot_bytes[0] = align_up(static_cast<size_t>(p->m) * p->intermediate_size * sizeof(Bf16), 256);
      L.slot_bytes[1] = align_up(static_cast<size_t>(p->m) * p->hidden_size * sizeof(Bf16), 256);
      L.num_slots = 2;
      // Gate tasks stage A (BM rows) plus gate and up B tiles; down tasks stage A
      // plus one B tile.  Both bodies share the same dynamic-smem carve.
      int gate_smem = kMlpStages * (kMlpBM * kMlpBK * 2 + 2 * kMlpBN1 * kMlpBK * 2);
      const int red_smem = 2 * 4 * kMlpBM * kMlpBN1 * static_cast<int>(sizeof(float));
      if (red_smem > gate_smem) gate_smem = red_smem;
      int down_smem = kMlpStages * (kMlpBM * kMlpBK * 2 + kMlpBN2 * kMlpBK * 2);
      const int red_down = 4 * kMlpBM * kMlpBN2 * static_cast<int>(sizeof(float));
      if (red_down > down_smem) down_smem = red_down;
      L.dynamic_smem = static_cast<int32_t>(align_up(static_cast<size_t>(
          gate_smem > down_smem ? gate_smem : down_smem), 16));
      break;
    }
    case MegaKind::kAttentionBlock: {
      const auto* p = static_cast<const MegaAttentionBlockParams*>(params);
      st = validate_attention(p);
      if (st != Status::kSuccess) return st;
      const int qkv_cols = (p->num_q_heads + 2 * p->num_kv_heads) * p->head_dim;
      const int mt = (p->m + kAttnBM - 1) / kAttnBM;
      const int nq = (qkv_cols + kAttnBN - 1) / kAttnBN;
      const int no = (p->hidden_size + kAttnBN - 1) / kAttnBN;
      L.m = p->m;
      L.hidden = p->hidden_size;
      L.intermediate = 0;
      L.num_tasks = mt * (nq + no) + 2 * p->m;
      L.num_counters = 3;
      L.slot_bytes[0] = align_up(static_cast<size_t>(p->m) * qkv_cols * sizeof(Bf16), 256);
      L.slot_bytes[1] = align_up(static_cast<size_t>(p->m) * p->num_q_heads * p->head_dim * sizeof(Bf16), 256);
      L.num_slots = 2;
      int tile_smem = kAttnStages * (kAttnBM * kMlpBK * 2 + kAttnBN * kMlpBK * 2);
      const int red_attn = 4 * kAttnBM * kAttnBN * static_cast<int>(sizeof(float));
      if (red_attn > tile_smem) tile_smem = red_attn;
      const int row_smem = static_cast<int>(p->hidden_size * sizeof(Bf16));
      L.dynamic_smem = static_cast<int32_t>(
          align_up(static_cast<size_t>(tile_smem > row_smem ? tile_smem : row_smem), 16));
      break;
    }
    case MegaKind::kMoELocal: {
      const auto* p = static_cast<const MegaMoELocalParams*>(params);
      st = validate_moe(p, true);
      if (st != Status::kSuccess) return st;
      L.m = p->m;
      L.hidden = p->hidden_size;
      L.intermediate = p->intermediate_size;
      L.num_tasks = p->m;
      L.num_counters = 0;
      L.num_slots = 0;
      L.dynamic_smem = static_cast<int32_t>(align_up(
          static_cast<size_t>(p->hidden_size) * sizeof(Bf16) +
              static_cast<size_t>(p->intermediate_size) * sizeof(Bf16) +
              static_cast<size_t>(p->hidden_size) * sizeof(float) + 64,
          16));
      break;
    }
    case MegaKind::kMoEEP: {
      const auto* p = static_cast<const MegaMoEEPParams*>(params);
      st = validate_ep(p);
      if (st != Status::kSuccess) return st;
      L.m = p->local.m;
      L.hidden = p->local.hidden_size;
      L.intermediate = p->local.intermediate_size;
      L.num_tasks = p->local.m;
      L.num_counters = 0;
      L.num_slots = 0;
      L.dynamic_smem = static_cast<int32_t>(align_up(
          static_cast<size_t>(p->local.hidden_size) * sizeof(Bf16) +
              static_cast<size_t>(p->local.intermediate_size) * sizeof(Bf16) +
              static_cast<size_t>(p->local.hidden_size) * sizeof(float) + 64,
          16));
      break;
    }
    case MegaKind::kDecoder: {
      // The decoder's parameter block is assembled from the three sub-blocks; the
      // layout only needs the attention and dense MLP shapes plus the MoE flag.
      const auto* p = static_cast<const MegaDecoderParams*>(params);
      if (p == nullptr) return bad("decoder", "null params");
      st = validate_attention(&p->attn);
      if (st != Status::kSuccess) return st;
      st = validate_mlp(&p->mlp);
      if (st != Status::kSuccess) return st;
      if (p->attn.m != p->mlp.m) return bad("decoder", "attn.m != mlp.m");
      const bool moe_active = p->moe.m > 0;
      if (moe_active) {
        st = validate_moe(&p->moe, true);
        if (st != Status::kSuccess) return st;
        if (p->moe.m != p->attn.m || p->moe.hidden_size != p->attn.hidden_size ||
            p->moe.intermediate_size != p->mlp.intermediate_size) {
          return bad("decoder", "MoE/attention/MLP shapes disagree");
        }
      }
      const int qkv_cols = (p->attn.num_q_heads + 2 * p->attn.num_kv_heads) * p->attn.head_dim;
      const int mt_a = (p->attn.m + kAttnBM - 1) / kAttnBM;
      const int nq = (qkv_cols + kAttnBN - 1) / kAttnBN;
      const int no = (p->attn.hidden_size + kAttnBN - 1) / kAttnBN;
      const int mt_f = (p->mlp.m + kMlpBM - 1) / kMlpBM;
      const int n1 = (p->mlp.intermediate_size + kMlpBN1 - 1) / kMlpBN1;
      const int n2 = (p->mlp.hidden_size + kMlpBN2 - 1) / kMlpBN2;
      L.m = p->attn.m;
      L.hidden = p->attn.hidden_size;
      L.intermediate = p->mlp.intermediate_size;
      L.num_tasks = mt_a * (nq + no) + 2 * p->attn.m +
                    (moe_active ? p->moe.m : mt_f * (n1 + n2));
      L.num_counters = 4 + mt_f;
      L.slot_bytes[0] = align_up(static_cast<size_t>(p->attn.m) * qkv_cols * sizeof(Bf16), 256);
      L.slot_bytes[1] = align_up(
          static_cast<size_t>(p->attn.m) * p->attn.num_q_heads * p->attn.head_dim * sizeof(Bf16), 256);
      L.slot_bytes[2] = align_up(static_cast<size_t>(p->mlp.m) * p->mlp.intermediate_size * sizeof(Bf16), 256);
      L.slot_bytes[3] = align_up(static_cast<size_t>(p->attn.m) * p->attn.hidden_size * sizeof(Bf16), 256);
      L.num_slots = 4;
      int tile_smem = kAttnStages * (kAttnBM * kMlpBK * 2 + kAttnBN * kMlpBK * 2);
      const int red_attn = 4 * kAttnBM * kAttnBN * static_cast<int>(sizeof(float));
      if (red_attn > tile_smem) tile_smem = red_attn;
      int ffn_smem = kMlpStages * (kMlpBM * kMlpBK * 2 + 2 * kMlpBN1 * kMlpBK * 2);
      const int red_ffn = 2 * 4 * kMlpBM * kMlpBN1 * static_cast<int>(sizeof(float));
      if (red_ffn > ffn_smem) ffn_smem = red_ffn;
      int row_smem = static_cast<int>(
          static_cast<size_t>(p->attn.hidden_size) * sizeof(Bf16) +
          static_cast<size_t>(p->mlp.intermediate_size) * sizeof(Bf16) +
          static_cast<size_t>(p->attn.hidden_size) * sizeof(float) + 64);
      if (ffn_smem > row_smem) row_smem = ffn_smem;
      if (tile_smem > row_smem) row_smem = tile_smem;
      L.dynamic_smem = static_cast<int32_t>(align_up(static_cast<size_t>(row_smem), 16));
      break;
    }
    default:
      return bad("layout", "unknown MegaKind");
  }
  *out = L;
  return Status::kSuccess;
}

size_t persistent_bytes(const Layout& L) {
  // Mirror prepare_common's carve exactly: every region starts on a 256-byte
  // boundary, so the requirement is the sum of the individually aligned sizes
  // (not one final align_up, which under-counts by up to 255 B per region).
  size_t total = 0;
  auto add = [&](size_t n) { total = align_up(total + n, 256); };
  add(4);                                       // cursor
  add(static_cast<size_t>(L.num_counters) * sizeof(uint32_t) + sizeof(uint32_t));
  add(static_cast<size_t>(L.num_tasks) * sizeof(exec::TileTask));
  add(8 * sizeof(StageDesc));                   // stage descriptors (at most 8)
  add(static_cast<size_t>(L.num_slots) * sizeof(void*));
  add(static_cast<size_t>(L.num_slots) * sizeof(int64_t));
  for (int i = 0; i < 4; ++i) add(L.slot_bytes[i]);
  return total + 256;                           // + worst-case base alignment pad
}

// ---- kernel instantiations ---------------------------------------------------
const void* kernel_for(MegaKind kind) {
  switch (kind) {
    case MegaKind::kMLP:
      return reinterpret_cast<const void*>(
          &mega_mlp_kernel<Bf16, Bf16, kMlpBM, kMlpBN1, kMlpBN2, kMlpBK, kMlpStages, MegaArch>);
    case MegaKind::kAttentionBlock:
      return reinterpret_cast<const void*>(
          &mega_attention_block_kernel<Bf16, 128, 0, 16, MegaArch>);
    case MegaKind::kMoELocal:
      return reinterpret_cast<const void*>(
          &mega_moe_local_kernel<Bf16, Bf16, 16, kMegaBN, kMegaBN, 64, false, MegaArch>);
    case MegaKind::kMoEEP:
      return reinterpret_cast<const void*>(
          &mega_moe_ep_kernel<Bf16, Bf16, 16, kMegaBN, kMegaBN, 1, MegaArch>);
    case MegaKind::kDecoder:
      return reinterpret_cast<const void*>(&mega_decoder_layer_kernel<Bf16, MegaArch>);
    default:
      return nullptr;
  }
}

Status prepare_common(MegaKind kind, const void* params, const Workspace& ws,
                      const LaunchContext& ctx, MegaPlan* plan, const Layout& L) {
  if (plan == nullptr) return bad("prepare", "null plan");
  if (ws.device_persistent == nullptr) return bad("prepare", "null device_persistent");
  if (ws.device_persistent_bytes < persistent_bytes(L)) {
    return detail::set_error(Status::kWorkspaceTooSmall,
                             "mega.prepare: persistent workspace %zu < required %zu",
                             ws.device_persistent_bytes, persistent_bytes(L));
  }
  const uintptr_t raw = reinterpret_cast<uintptr_t>(ws.device_persistent);
  unsigned char* base = reinterpret_cast<unsigned char*>((raw + 255u) & ~uintptr_t(255u));
  size_t off = 0;
  auto take = [&](size_t n) -> unsigned char* {
    unsigned char* p = base + off;
    off = align_up(off + n, 256);
    return p;
  };

  uint32_t* cursor = reinterpret_cast<uint32_t*>(take(4));
  uint32_t* counters = reinterpret_cast<uint32_t*>(take((L.num_counters + 1) * 4));
  auto* tasks = reinterpret_cast<exec::TileTask*>(take(L.num_tasks * sizeof(exec::TileTask)));
  auto* stages = reinterpret_cast<StageDesc*>(take(8 * sizeof(StageDesc)));
  auto** slot_ptrs = reinterpret_cast<void**>(take(L.num_slots * sizeof(void*)));
  auto* slot_bytes = reinterpret_cast<int64_t*>(take(L.num_slots * sizeof(int64_t)));

  std::vector<exec::TileTask> host_tasks;
  std::vector<StageDesc> host_stages;
  std::vector<void*> host_slot_ptrs(L.num_slots, nullptr);
  std::vector<int64_t> host_slot_bytes(L.num_slots, 0);
  HostPlan hp;
  hp.kind = kind;
  hp.max_m = L.m;
  hp.threads = kMegaThreads;
  hp.num_tasks = L.num_tasks;
  hp.num_counters = L.num_counters;
  hp.cursor = cursor;
  hp.counters = counters;
  hp.generation = g_plan_generation.fetch_add(1);
  for (int i = 0; i < L.num_slots; ++i) {
    hp.slots[i] = take(L.slot_bytes[i]);
    hp.slot_bytes[i] = L.slot_bytes[i];
    host_slot_ptrs[i] = hp.slots[i];
    host_slot_bytes[i] = static_cast<int64_t>(L.slot_bytes[i]);
  }

  host_stages.reserve(8);
  auto add_stage = [&](StageOp op, int first, int count) {
    StageDesc s{};
    s.op = op;
    s.num_inputs = 0;
    s.num_outputs = 0;
    s.input_slots[0] = s.input_slots[1] = s.input_slots[2] = s.input_slots[3] = -1;
    s.output_slots[0] = s.output_slots[1] = -1;
    s.first_task = first;
    s.num_tasks = count;
    host_stages.push_back(s);
  };

  switch (kind) {
    case MegaKind::kMLP: {
      const auto* p = static_cast<const MegaMLPParams*>(params);
      const int mt = (p->m + kMlpBM - 1) / kMlpBM;
      const int n1 = (p->intermediate_size + kMlpBN1 - 1) / kMlpBN1;
      const int n2 = (p->hidden_size + kMlpBN2 - 1) / kMlpBN2;
      host_tasks.reserve(static_cast<size_t>(L.num_tasks));
      for (int mi = 0; mi < mt; ++mi) {
        for (int t = 0; t < n1; ++t) {
          exec::TileTask k{};
          k.op_id = 0;
          k.tile_m = mi;
          k.tile_n = t;
          k.wait_counter = -1;
          k.signal_counter = mi;   // one counter per 16-row tile
          k.out_slot = 0;
          host_tasks.push_back(k);
        }
        for (int t = 0; t < n2; ++t) {
          exec::TileTask k{};
          k.op_id = 1;
          k.tile_m = mi;
          k.tile_n = t;
          k.wait_counter = mi;
          k.wait_target = n1;
          k.signal_counter = -1;
          k.out_slot = 0;
          host_tasks.push_back(k);
        }
      }
      add_stage(StageOp::kGemm, 0, mt * n1);
      add_stage(StageOp::kGemm, mt * n1, mt * n2);
      break;
    }
    case MegaKind::kAttentionBlock: {
      const auto* p = static_cast<const MegaAttentionBlockParams*>(params);
      const int qkv_cols = (p->num_q_heads + 2 * p->num_kv_heads) * p->head_dim;
      const int mt = (p->m + kAttnBM - 1) / kAttnBM;
      const int nq = (qkv_cols + kAttnBN - 1) / kAttnBN;
      const int no = (p->hidden_size + kAttnBN - 1) / kAttnBN;
      host_tasks.reserve(static_cast<size_t>(L.num_tasks));
      for (int mi = 0; mi < mt; ++mi) {
        for (int t = 0; t < nq; ++t) {
          exec::TileTask k{};
          k.op_id = 0;
          k.tile_m = mi;
          k.tile_n = t;
          k.wait_counter = -1;
          k.signal_counter = 0;
          k.out_slot = 0;
          host_tasks.push_back(k);
        }
      }
      for (int m = 0; m < p->m; ++m) {
        exec::TileTask k{};
        k.op_id = 1;
        k.tile_m = m;
        k.wait_counter = 0;
        k.wait_target = mt * nq;
        k.signal_counter = 1;
        k.out_slot = 0;
        host_tasks.push_back(k);
      }
      for (int m = 0; m < p->m; ++m) {
        exec::TileTask k{};
        k.op_id = 2;
        k.tile_m = m;
        k.wait_counter = 1;
        k.wait_target = p->m;
        k.signal_counter = 2;
        k.out_slot = 1;
        host_tasks.push_back(k);
      }
      for (int mi = 0; mi < mt; ++mi) {
        for (int t = 0; t < no; ++t) {
          exec::TileTask k{};
          k.op_id = 3;
          k.tile_m = mi;
          k.tile_n = t;
          k.wait_counter = 2;
          k.wait_target = p->m;
          k.signal_counter = -1;
          k.out_slot = -1;
          host_tasks.push_back(k);
        }
      }
      add_stage(StageOp::kGemm, 0, mt * nq);
      add_stage(StageOp::kKVAppend, mt * nq, p->m);
      add_stage(StageOp::kAttentionDecode, mt * nq + p->m, p->m);
      add_stage(StageOp::kGemm, mt * nq + 2 * p->m, mt * no);
      break;
    }
    case MegaKind::kMoELocal:
    case MegaKind::kMoEEP: {
      const int rows = kind == MegaKind::kMoELocal
                           ? static_cast<const MegaMoELocalParams*>(params)->m
                           : static_cast<const MegaMoEEPParams*>(params)->local.m;
      host_tasks.reserve(static_cast<size_t>(L.num_tasks));
      for (int m = 0; m < rows; ++m) {
        exec::TileTask k{};
        k.op_id = 0;
        k.tile_m = m;
        k.wait_counter = -1;
        k.signal_counter = -1;
        k.out_slot = -1;
        host_tasks.push_back(k);
      }
      add_stage(StageOp::kExpertGemm, 0, rows);
      break;
    }
    case MegaKind::kDecoder: {
      const auto* p = static_cast<const MegaDecoderParams*>(params);
      const bool moe_active = p->moe.m > 0;
      const int qkv_cols = (p->attn.num_q_heads + 2 * p->attn.num_kv_heads) * p->attn.head_dim;
      const int mt_a = (p->attn.m + kAttnBM - 1) / kAttnBM;
      const int nq = (qkv_cols + kAttnBN - 1) / kAttnBN;
      const int no = (p->attn.hidden_size + kAttnBN - 1) / kAttnBN;
      const int mt_f = (p->mlp.m + kMlpBM - 1) / kMlpBM;
      const int n1 = (p->mlp.intermediate_size + kMlpBN1 - 1) / kMlpBN1;
      const int n2 = (p->mlp.hidden_size + kMlpBN2 - 1) / kMlpBN2;
      host_tasks.reserve(static_cast<size_t>(L.num_tasks));
      for (int mi = 0; mi < mt_a; ++mi) {
        for (int t = 0; t < nq; ++t) {
          exec::TileTask k{};
          k.op_id = 0;
          k.tile_m = mi;
          k.tile_n = t;
          k.wait_counter = -1;
          k.signal_counter = 0;
          k.out_slot = 0;
          host_tasks.push_back(k);
        }
      }
      for (int m = 0; m < p->attn.m; ++m) {
        exec::TileTask k{};
        k.op_id = 1;
        k.tile_m = m;
        k.wait_counter = 0;
        k.wait_target = mt_a * nq;
        k.signal_counter = 1;
        k.out_slot = 0;
        host_tasks.push_back(k);
      }
      for (int m = 0; m < p->attn.m; ++m) {
        exec::TileTask k{};
        k.op_id = 2;
        k.tile_m = m;
        k.wait_counter = 1;
        k.wait_target = p->attn.m;
        k.signal_counter = 2;
        k.out_slot = 1;
        host_tasks.push_back(k);
      }
      for (int mi = 0; mi < mt_a; ++mi) {
        for (int t = 0; t < no; ++t) {
          exec::TileTask k{};
          k.op_id = 3;
          k.tile_m = mi;
          k.tile_n = t;
          k.wait_counter = 2;
          k.wait_target = p->attn.m;
          k.signal_counter = 3;
          k.out_slot = 1;
          host_tasks.push_back(k);
        }
      }
      if (moe_active) {
        for (int m = 0; m < p->moe.m; ++m) {
          exec::TileTask k{};
          k.op_id = 6;
          k.tile_m = m;
          k.wait_counter = 3;
          k.wait_target = mt_a * no;
          k.signal_counter = -1;
          k.out_slot = -1;
          host_tasks.push_back(k);
        }
      } else {
        for (int mi = 0; mi < mt_f; ++mi) {
          for (int t = 0; t < n1; ++t) {
            exec::TileTask k{};
            k.op_id = 4;
            k.tile_m = mi;
            k.tile_n = t;
            k.wait_counter = 3;
            k.wait_target = mt_a * no;
            k.signal_counter = 4 + mi;
            k.out_slot = 2;
            host_tasks.push_back(k);
          }
          for (int t = 0; t < n2; ++t) {
            exec::TileTask k{};
            k.op_id = 5;
            k.tile_m = mi;
            k.tile_n = t;
            k.wait_counter = 4 + mi;
            k.wait_target = n1;
            k.signal_counter = -1;
            k.out_slot = 2;
            host_tasks.push_back(k);
          }
        }
      }
      add_stage(StageOp::kGemm, 0, mt_a * nq);
      add_stage(StageOp::kKVAppend, mt_a * nq, p->attn.m);
      add_stage(StageOp::kAttentionDecode, mt_a * nq + p->attn.m, p->attn.m);
      add_stage(StageOp::kGemm, mt_a * nq + 2 * p->attn.m, mt_a * no);
      if (moe_active) {
        add_stage(StageOp::kExpertGemm, mt_a * (nq + no) + 2 * p->attn.m, p->moe.m);
      } else {
        add_stage(StageOp::kGemm, mt_a * (nq + no) + 2 * p->attn.m, mt_f * n1);
        add_stage(StageOp::kGemm, mt_a * (nq + no) + 2 * p->attn.m + mt_f * n1, mt_f * n2);
      }
      break;
    }
    default:
      return bad("prepare", "unknown MegaKind");
  }

  if (static_cast<int>(host_tasks.size()) != L.num_tasks) {
    return detail::set_error(Status::kInternal, "mega.prepare: task count mismatch");
  }

  plan->kind = kind;
  plan->device.stages = stages;
  plan->device.num_stages = static_cast<int32_t>(host_stages.size());
  plan->device.queue.tasks = tasks;
  plan->device.queue.num_tasks = L.num_tasks;
  plan->device.queue.cursor = cursor;
  plan->device.queue.counters = counters;
  plan->device.queue.num_counters = L.num_counters;
  // The last slot of the counter carve is the cross-CTA arrival count used by the
  // in-kernel reset below (never part of the task wait/signal namespace).
  plan->device.queue.epoch = 1u;
  plan->device.slots = slot_ptrs;
  plan->device.slot_bytes = slot_bytes;
  plan->device.num_slots = L.num_slots;
  plan->device.max_m = L.m;
  plan->device.capture_generation = ctx.capture_generation;
  plan->plan_generation = hp.generation;

  const void* kfn = kernel_for(kind);
  if (kfn == nullptr) return bad("prepare", "no kernel for kind");
  const cudaError_t smem_err =
      cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, L.dynamic_smem);
  if (smem_err != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(smem_err),
                                     "mega.prepare: dynamic smem attribute: %s",
                                     cudaGetErrorString(smem_err));
  }
  int smem_limit = 0;
  cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         ctx.device >= 0 ? ctx.device : 0);
  if (L.dynamic_smem > smem_limit) {
    return detail::set_error(Status::kUnsupportedShape,
                             "mega.prepare: %d B dynamic smem exceeds device limit %d B",
                             L.dynamic_smem, smem_limit);
  }
  cudaFuncAttributes attr{};
  const cudaError_t aerr = cudaFuncGetAttributes(&attr, kfn);
  if (aerr != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(aerr),
                                     "mega.prepare: func attributes: %s",
                                     cudaGetErrorString(aerr));
  }
  exec::LaunchConfig cfg{};
  Status st = exec::persistent_grid(kfn, kMegaThreads, static_cast<size_t>(L.dynamic_smem), ctx,
                                    L.num_tasks, &cfg);
  if (st != Status::kSuccess) return st;
  hp.ctas = static_cast<int32_t>(cfg.grid.x);
  hp.dynamic_smem = L.dynamic_smem;

  cudaError_t e1 = cudaSuccess, e2 = cudaSuccess, e3 = cudaSuccess, e4 = cudaSuccess;
  if (!host_tasks.empty()) {
    e1 = cudaMemcpy(tasks, host_tasks.data(), host_tasks.size() * sizeof(exec::TileTask),
                    cudaMemcpyHostToDevice);
  }
  if (!host_stages.empty()) {
    e2 = cudaMemcpy(stages, host_stages.data(), host_stages.size() * sizeof(StageDesc),
                    cudaMemcpyHostToDevice);
  }
  if (!host_slot_ptrs.empty()) {
    e3 = cudaMemcpy(slot_ptrs, host_slot_ptrs.data(), host_slot_ptrs.size() * sizeof(void*),
                    cudaMemcpyHostToDevice);
  }
  if (!host_slot_bytes.empty()) {
    e4 = cudaMemcpy(slot_bytes, host_slot_bytes.data(), host_slot_bytes.size() * sizeof(int64_t),
                    cudaMemcpyHostToDevice);
  }
  if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess || e4 != cudaSuccess) {
    return detail::set_backend_error(Status::kCudaError, 0, "mega.prepare: metadata upload failed");
  }
  {
    // Synchronous: prepare() is outside capture and may name a different stream than
    // the first run(), so the zeroed carve must be visible to every stream.
    cudaError_t ez = cudaMemset(counters, 0, (L.num_counters + 1) * sizeof(uint32_t));
    if (ez != cudaSuccess) {
      return detail::set_backend_error(Status::kCudaError, static_cast<int32_t>(ez),
                                       "mega.prepare: counter reset: %s", cudaGetErrorString(ez));
    }
  }

  const char* id_text = "mega.unknown";
  switch (kind) {
    case MegaKind::kMLP: id_text = "mega_mlp_kernel"; break;
    case MegaKind::kAttentionBlock: id_text = "mega_attention_block_kernel"; break;
    case MegaKind::kMoELocal: id_text = "mega_moe_local_kernel"; break;
    case MegaKind::kMoEEP: id_text = "mega_moe_ep_kernel"; break;
    case MegaKind::kDecoder: id_text = "mega_decoder_layer_kernel"; break;
    default: break;
  }
  plan->info.kernel_id = fnv1a64(id_text);
  plan->info.num_device_launches = 1;   // the kernel resets cursor/counters on exit
  plan->info.num_stages = static_cast<int32_t>(host_stages.size());
  plan->info.num_tasks = L.num_tasks;
  plan->info.ctas = hp.ctas;
  plan->info.regs_per_thread = attr.numRegs;
  plan->info.smem_per_cta = static_cast<int32_t>(attr.sharedSizeBytes) + L.dynamic_smem;
  plan->info.workspace.device_scratch_bytes = 0;
  plan->info.workspace.device_persistent_bytes = persistent_bytes(L);
  plan->info.workspace.host_pinned_bytes = 0;
  plan->info.workspace.alignment = 256;

  {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    g_host_plans[plan] = hp;
  }
  return Status::kSuccess;
}

// Look up the host record and enforce the one-plan-one-stream rule.  The caller
// gets a copy; the launch itself runs outside the lock.
Status plan_enter(const MegaPlan& plan, MegaKind kind, int32_t m, const LaunchContext& ctx,
                  HostPlan* out) {
  if (out == nullptr) return bad("run", "null host plan");
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  auto it = g_host_plans.find(&plan);
  if (it == g_host_plans.end()) {
    return detail::set_error(Status::kPlanInvalidated,
                             "mega.run: plan was not prepared by mega_prepare (or was copied)");
  }
  HostPlan hp = it->second;
  if (hp.kind != kind) {
    return detail::set_error(Status::kInvalidArgument, "mega.run: plan kind mismatch");
  }
  if (m < 0 || m > hp.max_m) {
    return detail::set_error(Status::kCapacityExceeded,
                             "mega.run: m=%d exceeds prepared capacity %d", m, hp.max_m);
  }
  if (!hp.stream_bound) {
    hp.stream_bound = true;
    hp.bound_stream = ctx.stream;
    it->second.stream_bound = true;
    it->second.bound_stream = ctx.stream;
  } else if (hp.bound_stream != ctx.stream) {
    return detail::set_error(Status::kInvalidArgument,
                             "mega.run: one plan is bound to one stream; prepare a second plan");
  }
  *out = hp;
  return Status::kSuccess;
}

Status launch_common(const HostPlan& hp, const LaunchContext& ctx, const void* kernel_fn,
                     void** args) {
  (void)cudaGetLastError();
  exec::LaunchConfig cfg{};
  cfg.grid = dim3(static_cast<unsigned>(hp.ctas));
  cfg.block = dim3(static_cast<unsigned>(hp.threads));
  cfg.dynamic_smem_bytes = static_cast<size_t>(hp.dynamic_smem);
  return exec::launch_kernel(kernel_fn, cfg, args, ctx);
}

}  // namespace
}  // namespace kops::mega

// =============================================================================
// Device bodies
// =============================================================================
namespace kops::mega {
namespace {

template <typename T>
__device__ __forceinline__ float to_f(T v) {
  return dev::to_f(v);
}

/// Warp-scoped dot product.  The empty asm consumes the reduced value on every
/// lane: without it nvcc can sink the whole reduction into the `lane == 0` store
/// and execute __shfl_down_sync with one active lane (observed on sm_120: the
/// result then depended on the launch grid).  The value is also stored by every
/// lane at the call sites where that is harmless, so the collective stays uniform.
template <typename A, typename B>
__device__ __forceinline__ float warp_dot(const A* a, const B* b, int k, int lane) {
  float s = 0.0f;
  for (int i = lane; i < k; i += 32) s = fmaf(to_f(a[i]), to_f(b[i]), s);
  // XOR butterfly: every lane ends with the full sum, so no final broadcast and no
  // opportunity for the compiler to leave the reduction on a single active lane.
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
  asm volatile("" ::"f"(s));
  return s;
}

template <typename T>
__device__ __forceinline__ void norm_row_to_smem(const T* __restrict__ x, const T* __restrict__ w,
                                                 float eps, T* s, int H, int tid, int lane,
                                                 int warp) {
  if (w == nullptr) {
    for (int i = tid; i < H; i += blockDim.x) s[i] = x[i];
    __syncthreads();
    return;
  }
  __shared__ float red[32];
  float ss = 0.0f;
  for (int i = tid; i < H; i += blockDim.x) {
    const float v = to_f(x[i]);
    ss = fmaf(v, v, ss);
  }
  ss = warp_reduce_sum(ss);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float t = lane < kMegaWarps ? red[lane] : 0.0f;
    t = warp_reduce_sum(t);
    if (lane == 0) red[0] = t;
  }
  __syncthreads();
  const float inv = rsqrtf(red[0] / static_cast<float>(H) + eps);
  for (int i = tid; i < H; i += blockDim.x) {
    const float v = to_f(x[i]) * inv * to_f(w[i]);
    dev::from_f(s[i], v);
  }
  __syncthreads();
}

__device__ __forceinline__ float glu_runtime(ActFn fn, float gate, float up, const GluParams& p) {
  if (p.gate_clamp_max > 0.0f && gate > p.gate_clamp_max) gate = p.gate_clamp_max;
  if (p.up_clamp > 0.0f) {
    up = up > p.up_clamp ? p.up_clamp : up;
    up = up < -p.up_clamp ? -p.up_clamp : up;
  }
  up += p.up_offset;
  if (p.situ_linear_beta > 0.0f) up = p.situ_linear_beta * tanhf(up / p.situ_linear_beta);
  return dev::act_runtime(fn, gate, p) * up;
}

__device__ __forceinline__ int gate_row(int i, int I, GateLayout layout) {
  switch (layout) {
    case GateLayout::kUpGate: return i + I;
    case GateLayout::kInterleaved: return 2 * i;
    default: return i;
  }
}

__device__ __forceinline__ int up_row(int i, int I, GateLayout layout) {
  switch (layout) {
    case GateLayout::kUpGate: return i;
    case GateLayout::kInterleaved: return 2 * i + 1;
    default: return i + I;
  }
}

// ---- E04 MLP tiles -----------------------------------------------------------
// B7 mma.sync mainloop for the megakernel, reusing the dense_tile frame of
// csrc/gemm/sm120a_impl.cuh: BM = 16 rows (one m16n8k16 atom), BN output columns,
// a stage holds 64 bytes (BK = 32 bf16 K elements) of every operand row, and a
// STAGES-deep cp.async ring uses the same XOR swizzle as the B7 body.  The K chain
// is the same increasing k16 sequence as the modular mma_gemm oracle, so the fp32
// accumulation order matches.  The MLP kernels are BF16-only (validate_mlp).
//
// The gate tile stages two B matrices (gate row / up row of W13) and applies GLU in
// the epilogue; the down tile stages one B matrix and adds the residual.  Normalized
// x is produced on the fly into the A stage (RMS scale per row + norm weight), so no
// separate norm launch/scratch is needed; rows past m are zero-filled.
__device__ __forceinline__ uint32_t mlp_smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mlp_cp_async16(uint32_t dst, const void* src, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
#else
  (void)dst; (void)src; (void)pred;
#endif
}
__device__ __forceinline__ void mlp_cp_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
__device__ __forceinline__ void mlp_cp_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}
__device__ __forceinline__ void mlp_ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#else
  (void)r; (void)addr;
#endif
}

/// Per-row RMS scale of the 16-row tile: 1/rsqrt(mean(x^2)+eps), or 1 when the layer
/// has no pre-norm weight (the contract of norm_row_to_smem).  Out-of-range rows -> 0.
template <int BM>
__device__ __forceinline__ void mlp_row_scales(const void* xp, const void* nwp, float eps,
                                               int64_t ldx, int m, int H, int r0, float* s_inv,
                                               int lane, int warp) {
  const Bf16* nw = static_cast<const Bf16*>(nwp);
  for (int r = warp; r < BM; r += kMegaWarps) {
    const int row = r0 + r;
    float ss = 0.0f;
    if (row < m && nw != nullptr) {
      const Bf16* xrow = static_cast<const Bf16*>(xp) + static_cast<int64_t>(row) * ldx;
      for (int k = lane; k < H; k += 32) {
        const float v = to_f(xrow[k]);
        ss = fmaf(v, v, ss);
      }
      ss = warp_reduce_sum(ss);
    }
    if (lane == 0) {
      s_inv[r] = row < m
                     ? (nw != nullptr ? rsqrtf(ss / static_cast<float>(H) + eps) : 1.0f)
                     : 0.0f;
    }
  }
}

// ---- K-split mma tile (BM = 16 rows, BN = 64 columns, 512 threads) --------------
// One CTA runs 16 warps as four K-groups of four warps.  A stage holds four k16 steps;
// group g consumes step g only, so the CTA keeps 16 warps of cp.async / ldmatrix in
// flight.  The first 4-warp body was limited to ~535 GB/s per SM (the one-CTA-per-SM
// drain rate), while 16 warps reach the L2 plateau; see the B16 handoff.  Each group
// accumulates a fp32 partial over its k-steps and the partials are summed in group
// order in shared memory after the loop: only fp32 summation order changes.
constexpr int kMegaTileThreads = 512;
constexpr int kMegaKGroups = 4;

template <int BM, int BN, int STAGES, int NB, bool A_RING, typename FillA, typename IssueB>
__device__ __forceinline__ void bf16_ks_tile(unsigned char* smem, int nk, FillA fill_a,
                                             IssueB issue_b, float (&acc)[NB][2][4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using Bf = __nv_bfloat16;
  using Tc = dev::TcFrag<Bf>;
  constexpr int BK = 64;
  constexpr int RB = BK * 2;
  constexpr int KS = BK / 16;
  static_assert(KS == kMegaKGroups, "one k-step per group per stage");
  static_assert(BN == 64, "K-split tile is 64 columns wide");
  constexpr int CW = BN / kMegaKGroups;   // 16 columns per group warp
  const uint32_t sA = mlp_smem_u32(smem);
  const uint32_t sB = sA + STAGES * BM * RB;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int kg = warp >> 2;
  const int wg = warp & 3;

  fill_a(0, 0);
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) issue_b(s, s * BK);
    mlp_cp_commit();
  }
  mlp_cp_wait<STAGES - 2>();
  __syncthreads();

  const int sw_l = lane & 7;
  const uint32_t a_lane = static_cast<uint32_t>(((lane & 7) + ((lane >> 3) & 1) * 8) * RB);
  const uint32_t b_lane =
      static_cast<uint32_t>((wg * CW + (lane & 7) + (lane >> 4) * 8) * RB);
  const uint32_t ak = static_cast<uint32_t>(((kg * 2 + (lane >> 4)) ^ sw_l) << 4);
  const uint32_t bk = static_cast<uint32_t>(((kg * 2 + ((lane >> 3) & 1)) ^ sw_l) << 4);

#pragma unroll
  for (int m = 0; m < NB; ++m)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[m][nt][q] = 0.0f;

  for (int it = 0; it < nk; ++it) {
    const int rs = it % STAGES;
    // Refill the slot consumed in the previous iteration before touching this one;
    // the single barrier at the end of the iteration ordered every warp's reads.
    const int ns = it + STAGES - 1;
    if (ns < nk) issue_b(ns % STAGES, ns * BK);
    mlp_cp_commit();
    if (it + 1 < nk) fill_a((it + 1) & 1, (it + 1) * BK);
    uint32_t af[4];
    const int as = A_RING ? rs : (it & 1);
    mlp_ldmatrix_x4(af, sA + static_cast<uint32_t>(as * BM * RB) + a_lane + ak);
    const uint32_t bs = sB + static_cast<uint32_t>(rs * BN * RB);
#pragma unroll
    for (int m = 0; m < NB; ++m) {
      uint32_t rb[4];
      mlp_ldmatrix_x4(rb, bs + static_cast<uint32_t>(m * STAGES * BN * RB) + b_lane + bk);
      Tc::mma(acc[m][0], af, rb[0], rb[1]);
      Tc::mma(acc[m][1], af, rb[2], rb[3]);
    }
    if (it + 1 < nk) {
      mlp_cp_wait<STAGES - 2>();
      __syncthreads();
    }
  }
  mlp_cp_wait<0>();
  __syncthreads();
#else
  (void)smem; (void)nk; (void)fill_a; (void)issue_b; (void)acc;
  if (threadIdx.x == 0) __trap();
#endif
}

/// One K-group's [NB, 16, 64] partials -> shared fp32 [NB][4 groups][16][64].
template <int BM, int BN, int NB>
__device__ __forceinline__ void ks_store_partials(float* s_red, const float (&acc)[NB][2][4],
                                                  int lane) {
  const int gl = lane >> 2;
  const int gc = lane & 3;
  const int wg = (threadIdx.x >> 5) & 3;
  const int kg = threadIdx.x >> 7;
#pragma unroll
  for (int m = 0; m < NB; ++m)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = (i >> 1) * 8 + gl;
        const int col = wg * (BN / 4) + nt * 8 + 2 * gc + (i & 1);
        s_red[((m * kMegaKGroups + kg) * BM + row) * BN + col] = acc[m][nt][i];
      }
}

/// Sum the four K-group partials of one [BM, BN] element block.
__device__ __forceinline__ float ks_sum(const float* s_red, int m, int row, int col, int BN) {
  float v = 0.0f;
#pragma unroll
  for (int g = 0; g < kMegaKGroups; ++g) v += s_red[((m * kMegaKGroups + g) * 16 + row) * BN + col];
  return v;
}

/// One gate/up tile: [BM, BN] of the GLU output.  Writes the bf16 activated values to
/// p.intermediate; the down stage consumes them after the counter reaches n1.
template <int BM, int BN, int STAGES>
__device__ __forceinline__ void mlp_gate_tile(const MegaMLPParams& p, int midx, int tidx,
                                              unsigned char* smem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using Bf = __nv_bfloat16;
  constexpr int BK = 64;
  constexpr int RB = BK * 2;
  constexpr int RW = kMegaTileThreads / 8;
  constexpr int BI = (BN + RW - 1) / RW;

  const int H = p.hidden_size;
  const int I = p.intermediate_size;
  const int r0 = midx * BM;
  const int n0 = tidx * BN;
  const int nk = H / BK;
  const Bf* const w13 = static_cast<const Bf*>(p.w13);
  const Bf* const x = static_cast<const Bf*>(p.x);
  const Bf* const nw = static_cast<const Bf*>(p.pre_norm_weight);
  const int64_t ldx = p.ldx != 0 ? p.ldx : H;
  Bf* const inter = static_cast<Bf*>(p.intermediate) + static_cast<int64_t>(r0) * I + n0;

  const uint32_t sBg = mlp_smem_u32(smem) + STAGES * BM * RB;
  const uint32_t sBu = sBg + STAGES * BN * RB;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float s_inv[BM];
  mlp_row_scales<BM>(p.x, p.pre_norm_weight, p.pre_norm_eps, ldx, p.m, H, r0, s_inv,
                     lane, warp);
  __syncthreads();

  auto fill_a = [&](int stage, int kbase) {
    // 512 threads: four 4-byte quarters of every 16-byte chunk, so the whole tile
    // is normalized with four times the load parallelism of a 128-thread fill.
    const int chunk = tid >> 2;
    const int sub = tid & 3;
    const int row = chunk >> 3;
    const int ci = chunk & 7;
    const int k0 = kbase + ci * 8 + sub * 2;
    const int grow = r0 + row;
    const float inv = s_inv[row];
    float v0 = 0.0f, v1 = 0.0f;
    if (grow < p.m) {
      if (k0 < H) {
        v0 = to_f(x[static_cast<int64_t>(grow) * ldx + k0]);
        if (nw != nullptr) v0 *= inv * to_f(nw[k0]);
      }
      if (k0 + 1 < H) {
        v1 = to_f(x[static_cast<int64_t>(grow) * ldx + k0 + 1]);
        if (nw != nullptr) v1 *= inv * to_f(nw[k0 + 1]);
      }
    }
    const __nv_bfloat162 p0 = __floats2bfloat162_rn(v0, v1);
    *reinterpret_cast<uint32_t*>(smem + stage * BM * RB + row * RB +
                                 ((ci ^ (row & 7)) << 4) + (sub << 2)) =
        *reinterpret_cast<const uint32_t*>(&p0);
  };

  const int ld_ch = tid & 7;
  const int ld_row = tid >> 3;
  const uint32_t ld_soff = static_cast<uint32_t>(ld_row * RB + ((ld_ch ^ (ld_row & 7)) << 4));
  auto issue_b = [&](int stage, int kbase) {
    const uint32_t bgoff = static_cast<uint32_t>(stage * BN * RB);
    const int kbyte = kbase * 2 + ld_ch * 16;
    const bool kok = kbyte + 16 <= H * 2;
    const Bf* src = w13 + kbase + ld_ch * 8;
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      const int c = ld_row + i * RW;
      if (BI * RW == BN || c < BN) {
        const int col = n0 + c;
        const int ccol = col < I ? col : I - 1;
        const bool ok = kok && col < I;
        const uint32_t off = bgoff + static_cast<uint32_t>(i * RW * RB) + ld_soff;
        mlp_cp_async16(sBg + off,
                       src + static_cast<int64_t>(gate_row(ccol, I, p.gate_layout)) * H, ok);
        mlp_cp_async16(sBu + off,
                       src + static_cast<int64_t>(up_row(ccol, I, p.gate_layout)) * H, ok);
      }
    }
  };

  float acc[2][2][4];
  bf16_ks_tile<BM, BN, STAGES, 2, false>(smem, nk, fill_a, issue_b, acc);
  float* const s_red = reinterpret_cast<float*>(smem);
  ks_store_partials<BM, BN, 2>(s_red, acc, lane);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += kMegaTileThreads) {
    const int row = idx / BN;
    const int col = idx - row * BN;
    if (r0 + row < p.m && n0 + col < I) {
      const float g = ks_sum(s_red, 0, row, col, BN);
      const float u = ks_sum(s_red, 1, row, col, BN);
      dev::from_f(inter[static_cast<int64_t>(row) * I + col], glu_runtime(p.act, g, u, p.glu));
    }
  }
  __syncthreads();
#else
  (void)p; (void)midx; (void)tidx; (void)smem;
  if (threadIdx.x == 0) __trap();
#endif
}

/// One down tile: [BM, BN] of the hidden output, K = I, residual added in bf16.
template <int BM, int BN, int STAGES>
__device__ __forceinline__ void mlp_down_tile(const MegaMLPParams& p, int midx, int tidx,
                                              unsigned char* smem, Bf16* publish = nullptr,
                                              int64_t publish_ld = 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using Bf = __nv_bfloat16;
  constexpr int BK = 64;
  constexpr int RB = BK * 2;
  constexpr int RW = kMegaTileThreads / 8;
  constexpr int BI = (BN + RW - 1) / RW;

  const int H = p.hidden_size;
  const int I = p.intermediate_size;
  const int r0 = midx * BM;
  const int n0 = tidx * BN;
  const int nk = I / BK;
  const Bf* const w2 = static_cast<const Bf*>(p.w2);
  const Bf* const src = static_cast<const Bf*>(p.intermediate);
  const int64_t stride = p.ldo != 0 ? p.ldo : H;
  const Bf* const residual = p.residual != nullptr
                                 ? static_cast<const Bf*>(p.residual) +
                                       static_cast<int64_t>(r0) * stride + n0
                                 : nullptr;
  Bf* const out = static_cast<Bf*>(p.out) + static_cast<int64_t>(r0) * stride + n0;

  const uint32_t sA = mlp_smem_u32(smem);
  const uint32_t sB = sA + STAGES * BM * RB;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int ld_ch = tid & 7;
  const int ld_row = tid >> 3;
  const uint32_t ld_soff = static_cast<uint32_t>(ld_row * RB + ((ld_ch ^ (ld_row & 7)) << 4));
  auto issue = [&](int stage, int kbase) {
    const int kbyte = kbase * 2 + ld_ch * 16;
    const bool kok = kbyte + 16 <= I * 2;
    if (ld_row < BM) {
      mlp_cp_async16(sA + static_cast<uint32_t>(stage * BM * RB) + ld_soff,
                     src + static_cast<int64_t>(r0 + ld_row) * I + kbase + ld_ch * 8,
                     kok && r0 + ld_row < p.m);
    }
    const uint32_t boff = static_cast<uint32_t>(stage * BN * RB);
    const Bf* wsrc = w2 + kbase + ld_ch * 8;
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      const int c = ld_row + i * RW;
      if (BI * RW == BN || c < BN) {
        const bool ok = kok && n0 + c < H;
        mlp_cp_async16(sB + boff + static_cast<uint32_t>(i * RW * RB) + ld_soff,
                       wsrc + static_cast<int64_t>(n0 + c) * I, ok);
      }
    }
  };
  auto no_fill = [](int, int) {};

  float acc[1][2][4];
  bf16_ks_tile<BM, BN, STAGES, 1, true>(smem, nk, no_fill, issue, acc);
  float* const s_red = reinterpret_cast<float*>(smem);
  ks_store_partials<BM, BN, 1>(s_red, acc, lane);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += kMegaTileThreads) {
    const int row = idx / BN;
    const int col = idx - row * BN;
    if (r0 + row < p.m && n0 + col < H) {
      float v = ks_sum(s_red, 0, row, col, BN);
      if (residual != nullptr) v += to_f(residual[static_cast<int64_t>(row) * stride + col]);
      dev::from_f(out[static_cast<int64_t>(row) * stride + col], v);
      if (publish != nullptr && publish != static_cast<Bf*>(p.out)) {
        dev::from_f(publish[static_cast<int64_t>(r0 + row) * publish_ld + n0 + col], v);
      }
    }
  }
  __syncthreads();
#else
  (void)p; (void)midx; (void)tidx; (void)smem; (void)publish; (void)publish_ld;
  if (threadIdx.x == 0) __trap();
#endif
}

// SIMT fallback for K not a multiple of the 64-element stage (every benchmark and
// test shape uses an aligned K; this keeps arbitrary H/I correct without a second
// mainloop).  Same per-row fp32 dot contract as the first body.
template <typename T, typename W>
__device__ void mlp_gate_tile_simt(const MegaMLPParams& p, int midx, int tidx, T* s_norm,
                                   int tid, int lane, int warp) {
  const int H = p.hidden_size;
  const int I = p.intermediate_size;
  const int r0 = midx * kMlpBM;
  const int rend = r0 + kMlpBM < p.m ? r0 + kMlpBM : p.m;
  const W* w13 = static_cast<const W*>(p.w13);
  const int64_t ldx = p.ldx != 0 ? p.ldx : H;
  for (int m = r0; m < rend; ++m) {
    const T* xrow = static_cast<const T*>(p.x) + static_cast<int64_t>(m) * ldx;
    norm_row_to_smem<T>(xrow, static_cast<const T*>(p.pre_norm_weight), p.pre_norm_eps, s_norm,
                        H, tid, lane, warp);
    T* inter = static_cast<T*>(p.intermediate) + static_cast<int64_t>(m) * I;
    const int begin = tidx * kMlpBN1;
    const int end = begin + kMlpBN1 < I ? begin + kMlpBN1 : I;
    for (int i = begin + warp; i < end; i += kMegaWarps) {
      const float g = warp_dot<T, W>(s_norm, w13 + static_cast<int64_t>(gate_row(i, I, p.gate_layout)) * H, H, lane);
      const float u = warp_dot<T, W>(s_norm, w13 + static_cast<int64_t>(up_row(i, I, p.gate_layout)) * H, H, lane);
      dev::from_f(inter[i], glu_runtime(p.act, g, u, p.glu));
    }
    __syncthreads();
  }
}

template <typename T, typename W>
__device__ void mlp_down_tile_simt(const MegaMLPParams& p, int midx, int tidx, int lane, int warp) {
  const int H = p.hidden_size;
  const int I = p.intermediate_size;
  const int r0 = midx * kMlpBM;
  const int rend = r0 + kMlpBM < p.m ? r0 + kMlpBM : p.m;
  const W* w2 = static_cast<const W*>(p.w2);
  const int64_t stride = p.ldo != 0 ? p.ldo : H;
  const int begin = tidx * kMlpBN2;
  const int end = begin + kMlpBN2 < H ? begin + kMlpBN2 : H;
  for (int m = r0; m < rend; ++m) {
    const T* inter = static_cast<const T*>(p.intermediate) + static_cast<int64_t>(m) * I;
    const T* residual = p.residual != nullptr
                            ? static_cast<const T*>(p.residual) + static_cast<int64_t>(m) * stride
                            : nullptr;
    T* out = static_cast<T*>(p.out) + static_cast<int64_t>(m) * stride;
    for (int h = begin + warp; h < end; h += kMegaWarps) {
      float v = warp_dot<T, W>(inter, w2 + static_cast<int64_t>(h) * I, I, lane);
      if (residual != nullptr) v += to_f(residual[h]);
      dev::from_f(out[h], v);
    }
    __syncthreads();
  }
}

// ---- E05 attention block -----------------------------------------------------
template <typename T>
__device__ __forceinline__ void head_norm_rope(T* head, const T* weight, float eps, int D,
                                               int rot_dim, int rot_offset, RopeStyle style,
                                               const float* cache_row, int lane) {
  // A null weight means the model has no QK norm for this side (e.g. Llama);
  // RoPE still runs.  A non-null weight is the per-head RMS affine.
  if (weight != nullptr) {
    float ss = 0.0f;
    for (int d = lane; d < D; d += 32) {
      const float v = to_f(head[d]);
      ss = fmaf(v, v, ss);
    }
    ss = warp_reduce_sum(ss);
    const float inv = rsqrtf(ss / static_cast<float>(D) + eps);
    for (int d = lane; d < D; d += 32) {
      dev::from_f(head[d], to_f(head[d]) * inv * to_f(weight[d]));
    }
    __syncwarp();
  }
  if (rot_dim <= 0 || cache_row == nullptr) return;
  if (style == RopeStyle::kNeoX) {
    const int half = rot_dim / 2;
    for (int e = lane; e < half; e += 32) {
      const int i0 = rot_offset + e;
      const int i1 = rot_offset + e + half;
      const float a = to_f(head[i0]);
      const float b = to_f(head[i1]);
      const float c = cache_row[e];
      const float s = cache_row[half + e];
      dev::from_f(head[i0], a * c - b * s);
      dev::from_f(head[i1], b * c + a * s);
    }
  } else {
    const int half = rot_dim / 2;
    for (int e = lane; e < half; e += 32) {
      const int i0 = rot_offset + 2 * e;
      const int i1 = i0 + 1;
      const float a = to_f(head[i0]);
      const float b = to_f(head[i1]);
      const float c = cache_row[e];
      const float s = cache_row[half + e];
      dev::from_f(head[i0], a * c - b * s);
      dev::from_f(head[i1], b * c + a * s);
    }
  }
  __syncwarp();
}

__device__ __forceinline__ int64_t kv_elem_offset(const kv::PagedKVDesc& d, int pos, int hk) {
  const int pg = pos / d.page_size;
  const int off = pos - pg * d.page_size;
  int page = 0;
  if (d.table_format == kv::PageTableFormat::kBlockTable) {
    page = d.block_table[static_cast<int64_t>(0) * d.block_table_stride + pg];
  } else {
    page = d.kv_page_indices[d.kv_indptr[0] + pg];
  }
  if (page < 0) return -1;
  return static_cast<int64_t>(page) * d.k_stride_page + static_cast<int64_t>(off) * d.k_stride_token +
         static_cast<int64_t>(hk) * d.k_stride_head;
}

template <typename T>
__device__ __forceinline__ int64_t kv_elem_offset_v(const kv::PagedKVDesc& d, int pos, int hk) {
  const int pg = pos / d.page_size;
  const int off = pos - pg * d.page_size;
  int page = 0;
  if (d.table_format == kv::PageTableFormat::kBlockTable) {
    page = d.block_table[static_cast<int64_t>(0) * d.block_table_stride + pg];
  } else {
    page = d.kv_page_indices[d.kv_indptr[0] + pg];
  }
  if (page < 0) return -1;
  return static_cast<int64_t>(page) * d.v_stride_page + static_cast<int64_t>(off) * d.v_stride_token +
         static_cast<int64_t>(hk) * d.v_stride_head;
}

// ---- E05 attention projection tiles (B7 mma.sync mainloop) --------------------
// QKV and O projections used to run as per-row SIMT dots inside one CTA, which is
// latency-bound on a single SM (3 MB of QKV weights per block).  They are now tiled
// over output columns exactly like the MLP mainloop, so 8-12 CTAs fetch the weights
// in parallel; the per-row QK-norm/RoPE/KV-append and paged decode stay row tasks.

template <int BM, int BN, int STAGES>
__device__ __forceinline__ void attention_proj_tile(const MegaAttentionBlockParams& p, int midx,
                                                    int tidx, unsigned char* smem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using Bf = __nv_bfloat16;
  constexpr int BK = 64;
  constexpr int RB = BK * 2;
  constexpr int RW = kMegaTileThreads / 8;
  constexpr int BI = (BN + RW - 1) / RW;

  const int H = p.hidden_size;
  const int total = (p.num_q_heads + 2 * p.num_kv_heads) * p.head_dim;
  const int r0 = midx * BM;
  const int n0 = tidx * BN;
  const int nk = H / BK;
  const Bf* const w = static_cast<const Bf*>(p.w_qkv);
  const Bf* const x = static_cast<const Bf*>(p.hidden);
  const Bf* const nw = static_cast<const Bf*>(p.in_norm_weight);
  const Bf* const bias = static_cast<const Bf*>(p.qkv_bias);
  Bf* const qkv = static_cast<Bf*>(p.plan.slots[0]) + static_cast<int64_t>(r0) * total + n0;

  const uint32_t sB = mlp_smem_u32(smem) + STAGES * BM * RB;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float s_inv[BM];
  mlp_row_scales<BM>(p.hidden, p.in_norm_weight, p.norm_eps, H, p.m, H, r0, s_inv, lane, warp);
  __syncthreads();

  auto fill_a = [&](int stage, int kbase) {
    const int chunk = tid >> 2;
    const int sub = tid & 3;
    const int row = chunk >> 3;
    const int ci = chunk & 7;
    const int k0 = kbase + ci * 8 + sub * 2;
    const int grow = r0 + row;
    const float inv = s_inv[row];
    float v0 = 0.0f, v1 = 0.0f;
    if (grow < p.m) {
      if (k0 < H) {
        v0 = to_f(x[static_cast<int64_t>(grow) * H + k0]);
        if (nw != nullptr) v0 *= inv * to_f(nw[k0]);
      }
      if (k0 + 1 < H) {
        v1 = to_f(x[static_cast<int64_t>(grow) * H + k0 + 1]);
        if (nw != nullptr) v1 *= inv * to_f(nw[k0 + 1]);
      }
    }
    const __nv_bfloat162 p0 = __floats2bfloat162_rn(v0, v1);
    *reinterpret_cast<uint32_t*>(smem + stage * BM * RB + row * RB +
                                 ((ci ^ (row & 7)) << 4) + (sub << 2)) =
        *reinterpret_cast<const uint32_t*>(&p0);
  };

  const int ld_ch = tid & 7;
  const int ld_row = tid >> 3;
  const uint32_t ld_soff = static_cast<uint32_t>(ld_row * RB + ((ld_ch ^ (ld_row & 7)) << 4));
  auto issue_b = [&](int stage, int kbase) {
    const uint32_t boff = static_cast<uint32_t>(stage * BN * RB);
    const int kbyte = kbase * 2 + ld_ch * 16;
    const bool kok = kbyte + 16 <= H * 2;
    const Bf* src = w + kbase + ld_ch * 8;
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      const int c = ld_row + i * RW;
      if (BI * RW == BN || c < BN) {
        const int col = n0 + c;
        const bool ok = kok && col < total;
        const int ccol = col < total ? col : total - 1;
        mlp_cp_async16(sB + boff + static_cast<uint32_t>(i * RW * RB) + ld_soff,
                       src + static_cast<int64_t>(ccol) * H, ok);
      }
    }
  };

  float acc[1][2][4];
  bf16_ks_tile<BM, BN, STAGES, 1, false>(smem, nk, fill_a, issue_b, acc);
  float* const s_red = reinterpret_cast<float*>(smem);
  ks_store_partials<BM, BN, 1>(s_red, acc, lane);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += kMegaTileThreads) {
    const int row = idx / BN;
    const int col = idx - row * BN;
    if (r0 + row < p.m && n0 + col < total) {
      float v = ks_sum(s_red, 0, row, col, BN);
      if (bias != nullptr) v += to_f(bias[n0 + col]);
      dev::from_f(qkv[static_cast<int64_t>(row) * total + col], v);
    }
  }
  __syncthreads();
#else
  (void)p; (void)midx; (void)tidx; (void)smem;
  if (threadIdx.x == 0) __trap();
#endif
}

template <int BM, int BN, int STAGES>
__device__ __forceinline__ void attention_oproj_tile(const MegaAttentionBlockParams& p, int midx,
                                                     int tidx, unsigned char* smem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using Bf = __nv_bfloat16;
  constexpr int BK = 64;
  constexpr int RB = BK * 2;
  constexpr int RW = kMegaTileThreads / 8;
  constexpr int BI = (BN + RW - 1) / RW;

  const int H = p.hidden_size;
  const int K = p.num_q_heads * p.head_dim;
  const int r0 = midx * BM;
  const int n0 = tidx * BN;
  const int nk = K / BK;
  const Bf* const w = static_cast<const Bf*>(p.w_o);
  const Bf* const src = static_cast<const Bf*>(p.plan.slots[1]);
  const Bf* const residual = p.residual != nullptr
                                 ? static_cast<const Bf*>(p.residual) +
                                       static_cast<int64_t>(r0) * H + n0
                                 : nullptr;
  const Bf* const gate = p.o_gate != nullptr
                             ? static_cast<const Bf*>(p.o_gate) + static_cast<int64_t>(r0) * H + n0
                             : nullptr;
  Bf* const out = static_cast<Bf*>(p.out) + static_cast<int64_t>(r0) * H + n0;
  Bf* const res_out = p.residual != nullptr
                          ? static_cast<Bf*>(const_cast<void*>(p.residual)) +
                                static_cast<int64_t>(r0) * H + n0
                          : nullptr;

  const uint32_t sA = mlp_smem_u32(smem);
  const uint32_t sB = sA + STAGES * BM * RB;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int ld_ch = tid & 7;
  const int ld_row = tid >> 3;
  const uint32_t ld_soff = static_cast<uint32_t>(ld_row * RB + ((ld_ch ^ (ld_row & 7)) << 4));
  auto issue = [&](int stage, int kbase) {
    const int kbyte = kbase * 2 + ld_ch * 16;
    const bool kok = kbyte + 16 <= K * 2;
    if (ld_row < BM) {
      mlp_cp_async16(sA + static_cast<uint32_t>(stage * BM * RB) + ld_soff,
                     src + static_cast<int64_t>(r0 + ld_row) * K + kbase + ld_ch * 8,
                     kok && r0 + ld_row < p.m);
    }
    const uint32_t boff = static_cast<uint32_t>(stage * BN * RB);
    const Bf* wsrc = w + kbase + ld_ch * 8;
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      const int c = ld_row + i * RW;
      if (BI * RW == BN || c < BN) {
        const bool ok = kok && n0 + c < H;
        mlp_cp_async16(sB + boff + static_cast<uint32_t>(i * RW * RB) + ld_soff,
                       wsrc + static_cast<int64_t>(n0 + c) * K, ok);
      }
    }
  };
  auto no_fill = [](int, int) {};

  float acc[1][2][4];
  bf16_ks_tile<BM, BN, STAGES, 1, true>(smem, nk, no_fill, issue, acc);
  float* const s_red = reinterpret_cast<float*>(smem);
  ks_store_partials<BM, BN, 1>(s_red, acc, lane);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += kMegaTileThreads) {
    const int row = idx / BN;
    const int col = idx - row * BN;
    if (r0 + row < p.m && n0 + col < H) {
      float v = ks_sum(s_red, 0, row, col, BN);
      if (residual != nullptr) v += to_f(residual[static_cast<int64_t>(row) * H + col]);
      if (gate != nullptr) v *= sigmoid(to_f(gate[static_cast<int64_t>(row) * H + col]));
      dev::from_f(out[static_cast<int64_t>(row) * H + col], v);
      if (res_out != nullptr && res_out != out) {
        dev::from_f(res_out[static_cast<int64_t>(row) * H + col], v);
      }
    }
  }
  __syncthreads();
#else
  (void)p; (void)midx; (void)tidx; (void)smem;
  if (threadIdx.x == 0) __trap();
#endif
}

// ---- E05 attention row work ---------------------------------------------------
template <typename T>
__device__ void attention_proj_simt(const MegaAttentionBlockParams& p, int midx, int tidx,
                                    T* s_norm, int tid, int lane, int warp) {
  const int H = p.hidden_size;
  const int total = (p.num_q_heads + 2 * p.num_kv_heads) * p.head_dim;
  const int r0 = midx * kAttnBM;
  const int rend = r0 + kAttnBM < p.m ? r0 + kAttnBM : p.m;
  const T* wqkv = static_cast<const T*>(p.w_qkv);
  const T* bias = static_cast<const T*>(p.qkv_bias);
  T* qkvb = static_cast<T*>(p.plan.slots[0]);
  const int begin = tidx * kAttnBN;
  const int end = begin + kAttnBN < total ? begin + kAttnBN : total;
  for (int m = r0; m < rend; ++m) {
    const T* xrow = static_cast<const T*>(p.hidden) + static_cast<int64_t>(m) * H;
    norm_row_to_smem<T>(xrow, static_cast<const T*>(p.in_norm_weight), p.norm_eps, s_norm, H, tid,
                        lane, warp);
    T* qkv = qkvb + static_cast<int64_t>(m) * total;
    for (int j = begin + warp; j < end; j += kMegaWarps) {
      float v = warp_dot<T, T>(s_norm, wqkv + static_cast<int64_t>(j) * H, H, lane);
      if (bias != nullptr) v += to_f(bias[j]);
      dev::from_f(qkv[j], v);
    }
    __syncthreads();
  }
}

/// Per-row QK RMS-norm + RoPE on the projected QKV row, then KV append.
template <typename T>
__device__ void attention_append_epilogue(const MegaAttentionBlockParams& p, int m, int tid,
                                          int lane, int warp) {
  const int D = p.head_dim;
  const int Hq = p.num_q_heads;
  const int Hkv = p.num_kv_heads;
  const int total = (Hq + 2 * Hkv) * D;
  T* qkv = static_cast<T*>(p.plan.slots[0]) + static_cast<int64_t>(m) * total;
  const int64_t pos = p.positions[m];
  const float* cache_row = p.cos_sin_cache != nullptr && p.rot_dim > 0
                               ? static_cast<const float*>(p.cos_sin_cache) + pos * p.rot_dim
                               : nullptr;
  for (int h = warp; h < Hq + 2 * Hkv; h += kMegaWarps) {
    T* head = qkv + static_cast<int64_t>(h) * D;
    if (h < Hq) {
      head_norm_rope<T>(head, static_cast<const T*>(p.q_norm_weight), p.norm_eps, D, p.rot_dim,
                        p.rot_offset, p.rope_style, cache_row, lane);
    } else if (h < Hq + Hkv) {
      head_norm_rope<T>(head, static_cast<const T*>(p.k_norm_weight), p.norm_eps, D, p.rot_dim,
                        p.rot_offset, p.rope_style, cache_row, lane);
    }
  }
  __syncthreads();
  const int64_t slot = p.slot_mapping[m];
  if (slot >= 0) {
    const int pg = slot / p.kv.page_size;
    const int off = slot - pg * p.kv.page_size;
    int page = 0;
    if (p.kv.table_format == kv::PageTableFormat::kBlockTable) {
      page = p.kv.block_table[static_cast<int64_t>(0) * p.kv.block_table_stride + pg];
    } else {
      page = p.kv.kv_page_indices[p.kv.kv_indptr[0] + pg];
    }
    if (page >= 0) {
      T* kbase = static_cast<T*>(p.kv.k_data);
      T* vbase = static_cast<T*>(p.kv.v_data);
      for (int hk = warp; hk < Hkv; hk += kMegaWarps) {
        const T* ksrc = qkv + static_cast<int64_t>(Hq + hk) * D;
        const T* vsrc = qkv + static_cast<int64_t>(Hq + Hkv + hk) * D;
        T* kdst = kbase + static_cast<int64_t>(page) * p.kv.k_stride_page +
                  static_cast<int64_t>(off) * p.kv.k_stride_token +
                  static_cast<int64_t>(hk) * p.kv.k_stride_head;
        T* vdst = vbase + static_cast<int64_t>(page) * p.kv.v_stride_page +
                  static_cast<int64_t>(off) * p.kv.v_stride_token +
                  static_cast<int64_t>(hk) * p.kv.v_stride_head;
        for (int d = lane; d < D; d += 32) {
          kdst[d] = ksrc[d];
          vdst[d] = vsrc[d];
        }
      }
    }
  }
  __syncthreads();
}

/// Paged decode of one row: attention output only (O-projection is a separate tile).
template <typename T>
__device__ void attention_decode_heads(const MegaAttentionBlockParams& p, int m, int tid, int lane,
                                       int warp) {
  const int D = p.head_dim;
  const int Hq = p.num_q_heads;
  const int Hkv = p.num_kv_heads;
  const int G = Hq / Hkv;
  const int total = (Hq + 2 * Hkv) * D;
  const T* qkv = static_cast<const T*>(p.plan.slots[0]) + static_cast<int64_t>(m) * total;
  T* attn = static_cast<T*>(p.plan.slots[1]) + static_cast<int64_t>(m) * Hq * D;
  const int pos = static_cast<int>(p.positions[m]);
  const int kv_len = p.kv.seq_lens[0];
  int hi = pos < kv_len - 1 ? pos : kv_len - 1;
  if (hi < 0) hi = -1;
  const float sm = p.mod.sm_scale != 0.0f ? p.mod.sm_scale : rsqrtf(static_cast<float>(D));
  const float cap = p.mod.logits_softcap;
  const float qscale = p.mod.q_scale != 0.0f ? p.mod.q_scale : 1.0f;
  const int wl = p.mod.window_left;
  const int nsink = p.mod.num_sink_tokens;

  for (int hq = warp; hq < Hq; hq += kMegaWarps) {
    const int hk = hq / G;
    float qreg[8];
    int nq = 0;
    for (int d = lane; d < D && nq < 8; d += 32) qreg[nq++] = to_f(qkv[static_cast<int64_t>(hq) * D + d]) * qscale;
    float oacc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) oacc[i] = 0.0f;
    const float neg_inf = __int_as_float(0xff800000);
    float mrun = neg_inf;
    float lrun = 0.0f;
    const float slope = p.mod.alibi_slopes != nullptr ? p.mod.alibi_slopes[hq] : 0.0f;
    const T* kbase = static_cast<const T*>(p.kv.k_data);
    const T* vbase = static_cast<const T*>(p.kv.v_data);
    for (int j = 0; j <= hi; ++j) {
      const bool visible = (wl < 0) || (j >= pos - wl) || (j < nsink);
      if (!visible) continue;
      const int64_t koff = kv_elem_offset(p.kv, j, hk);
      const int64_t voff = kv_elem_offset_v<T>(p.kv, j, hk);
      if (koff < 0 || voff < 0) continue;
      const T* krow = kbase + koff;
      const T* vrow = vbase + voff;
      float s = 0.0f;
      int qi = 0;
      for (int d = lane; d < D && qi < 8; d += 32) s = fmaf(qreg[qi++], to_f(krow[d]), s);
      s = warp_reduce_sum(s);
      s *= sm;
      if (cap > 0.0f) s = cap * tanhf(s / cap);
      if (p.mod.alibi_slopes != nullptr) s += slope * static_cast<float>(j - pos);
      const float mn = fmaxf(mrun, s);
      const float alpha = mrun == neg_inf ? 0.0f : __expf(mrun - mn);
      const float pj = __expf(s - mn);
      lrun = lrun * alpha + pj;
      mrun = mn;
      int vi = 0;
      for (int d = lane; d < D && vi < 8; d += 32) oacc[vi] = fmaf(pj, to_f(vrow[d]), oacc[vi] * alpha), ++vi;
    }
    if (p.mod.sink_logits != nullptr) {
      const float sk = p.mod.sink_logits[hq];
      const float mn = fmaxf(mrun, sk);
      const float alpha = mrun == neg_inf ? 0.0f : __expf(mrun - mn);
      lrun = lrun * alpha + __expf(sk - mn);
      mrun = mn;
    }
    const float inv = lrun > 0.0f ? 1.0f / lrun : 0.0f;
    int vi = 0;
    for (int d = lane; d < D && vi < 8; d += 32) {
      dev::from_f(attn[static_cast<int64_t>(hq) * D + d], oacc[vi] * inv);
      ++vi;
    }
  }
  __syncthreads();
}

template <typename T>
__device__ void attention_oproj_simt(const MegaAttentionBlockParams& p, int midx, int tidx,
                                     int lane, int warp) {
  const int H = p.hidden_size;
  const int K = p.num_q_heads * p.head_dim;
  const int r0 = midx * kAttnBM;
  const int rend = r0 + kAttnBM < p.m ? r0 + kAttnBM : p.m;
  const T* w_o = static_cast<const T*>(p.w_o);
  const int begin = tidx * kAttnBN;
  const int end = begin + kAttnBN < H ? begin + kAttnBN : H;
  for (int m = r0; m < rend; ++m) {
    const T* attn = static_cast<const T*>(p.plan.slots[1]) + static_cast<int64_t>(m) * K;
    const T* residual = p.residual != nullptr
                            ? static_cast<const T*>(p.residual) + static_cast<int64_t>(m) * H
                            : nullptr;
    const T* gate = p.o_gate != nullptr
                        ? static_cast<const T*>(p.o_gate) + static_cast<int64_t>(m) * H
                        : nullptr;
    T* out = static_cast<T*>(p.out) + static_cast<int64_t>(m) * H;
    T* res_out = p.residual != nullptr
                     ? static_cast<T*>(const_cast<void*>(p.residual)) + static_cast<int64_t>(m) * H
                     : nullptr;
    for (int h = begin + warp; h < end; h += kMegaWarps) {
      float v = warp_dot<T, T>(attn, w_o + static_cast<int64_t>(h) * K, K, lane);
      if (residual != nullptr) v += to_f(residual[h]);
      if (gate != nullptr) v *= sigmoid(to_f(gate[h]));
      dev::from_f(out[h], v);
      if (res_out != nullptr && res_out != out) dev::from_f(res_out[h], v);
    }
    __syncthreads();
  }
}

// ---- E06/E07 MoE -------------------------------------------------------------
template <typename T, typename W>
__device__ void moe_row(const MegaMoELocalParams& p, int m, const T* norm_w, float norm_eps,
                        T* s_norm, T* s_inter, float* s_acc, int tid, int lane, int warp,
                        T* ep_stage) {
  const int H = p.hidden_size;
  const int I = p.intermediate_size;
  const int E = p.num_local_experts;
  const int K = p.topk;
  const T* src = static_cast<const T*>(p.hidden) + static_cast<int64_t>(m) * H;
  if (ep_stage != nullptr) {
    T* dst = ep_stage + static_cast<int64_t>(m) * H;
    for (int i = tid; i < H; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
    src = dst;
  }
  if (norm_w != nullptr) {
    norm_row_to_smem<T>(src, norm_w, norm_eps, s_norm, H, tid, lane, warp);
  } else {
    for (int i = tid; i < H; i += blockDim.x) s_norm[i] = src[i];
    __syncthreads();
  }
  for (int h = tid; h < H; h += blockDim.x) s_acc[h] = 0.0f;
  __syncthreads();
  const W* w13 = static_cast<const W*>(p.w13);
  const W* w2 = static_cast<const W*>(p.w2);
  for (int kk = 0; kk < K; ++kk) {
    const int e = p.topk_ids[static_cast<int64_t>(m) * K + kk];
    if (e < 0 || e >= E) continue;
    const float w = p.topk_weights[static_cast<int64_t>(m) * K + kk];
    const W* w13e = w13 + static_cast<int64_t>(e) * 2 * I * H;
    const W* w2e = w2 + static_cast<int64_t>(e) * H * I;
    for (int i = warp; i < I; i += kMegaWarps) {
      const float g = warp_dot<T, W>(s_norm, w13e + static_cast<int64_t>(gate_row(i, I, p.gate_layout)) * H, H, lane);
      const float u = warp_dot<T, W>(s_norm, w13e + static_cast<int64_t>(up_row(i, I, p.gate_layout)) * H, H, lane);
      dev::from_f(s_inter[i], glu_runtime(p.act, g, u, p.glu));   // uniform value
    }
    __syncthreads();
    for (int h = warp; h < H; h += kMegaWarps) {
      const float v = warp_dot<T, W>(s_inter, w2e + static_cast<int64_t>(h) * I, I, lane);
      asm volatile("" ::"f"(v));   // keep the reduction uniform before the lane-0 update
      if (lane == 0) s_acc[h] = fmaf(w, v, s_acc[h]);
    }
    __syncthreads();
  }
  const T* shared = p.shared_out != nullptr ? static_cast<const T*>(p.shared_out) + static_cast<int64_t>(m) * H
                                            : nullptr;
  T* out = static_cast<T*>(p.out) + static_cast<int64_t>(m) * H;
  T* ep_out = ep_stage != nullptr ? ep_stage + static_cast<int64_t>(m) * H : nullptr;
  for (int h = warp; h < H; h += kMegaWarps) {
    float v = p.routed_scaling_factor * s_acc[h];
    if (shared != nullptr) v += to_f(shared[h]);
    if (lane == 0) {
      if (ep_out != nullptr) {
        dev::from_f(ep_out[h], v);
      } else {
        dev::from_f(out[h], v);
      }
    }
  }
  __syncthreads();
  if (ep_out != nullptr) {
    for (int h = tid; h < H; h += blockDim.x) out[h] = ep_out[h];
    __syncthreads();
  }
}

// ---- task loops --------------------------------------------------------------
__device__ __forceinline__ int next_task(const exec::TaskQueue& q) {
  __shared__ int s_idx;
  if (threadIdx.x == 0) s_idx = exec::task_queue_next(q);
  __syncthreads();
  return s_idx;
}

__device__ __forceinline__ void wait_task(const exec::TaskQueue& q, const exec::TileTask& t) {
  if (threadIdx.x == 0) {
    exec::task_wait(q, t);
    __threadfence();
  }
  __syncthreads();
}

__device__ __forceinline__ void signal_task(const exec::TaskQueue& q, const exec::TileTask& t) {
  __syncthreads();
  if (threadIdx.x == 0) exec::task_signal(q, t);
}

/// End-of-kernel reset: the last CTA to finish the task loop clears the cursor and the
/// epoch counters for the next launch (direct or CUDA-graph replay), then clears the
/// arrival slot.  The kernel is therefore one device launch; prepare() zeroes the carve
/// once, before capture.  Every CTA has stopped using the counters and has fenced its
/// own task stores before it arrives, so a winner may safely clear them.
__device__ __forceinline__ void mega_finish(const MegaPlanDevice& plan) {
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t* const done = plan.queue.counters + plan.queue.num_counters;
    __threadfence();
    const uint32_t old = atomicAdd(done, 1u);
    if (old + 1u >= gridDim.x) {
      __threadfence();
      *plan.queue.cursor = 0u;
      for (int i = 0; i < plan.queue.num_counters; ++i) plan.queue.counters[i] = 0u;
      *done = 0u;
      __threadfence();
    }
  }
}

}  // namespace
}  // namespace kops::mega

// =============================================================================
// Kernel definitions (must match mega_kernels.cuh exactly)
// =============================================================================
namespace kops::mega {

template <typename T, typename W, int BM, int BN1, int BN2, int BK, int STAGES, typename ArchTag>
__global__ void mega_mlp_kernel(const __grid_constant__ MegaMLPParams params) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const bool mma_ok = (params.hidden_size % kMlpBK == 0) && (params.intermediate_size % kMlpBK == 0);
  const MegaPlanDevice& plan = params.plan;
  for (;;) {
    const int idx = next_task(plan.queue);
    if (idx < 0) break;
    const exec::TileTask task = plan.queue.tasks[idx];
    if (task.op_id == 0) {
      if (mma_ok) {
        mlp_gate_tile<BM, BN1, STAGES>(params, task.tile_m, task.tile_n, smem_raw);
      } else {
        mlp_gate_tile_simt<T, W>(params, task.tile_m, task.tile_n,
                                 reinterpret_cast<T*>(smem_raw), tid, lane, warp);
      }
      signal_task(plan.queue, task);
    } else {
      wait_task(plan.queue, task);
      if (mma_ok) {
        mlp_down_tile<BM, BN2, STAGES>(params, task.tile_m, task.tile_n, smem_raw);
      } else {
        mlp_down_tile_simt<T, W>(params, task.tile_m, task.tile_n, lane, warp);
      }
    }
  }
  mega_finish(plan);
}

template <typename T, int HEAD_DIM, int GQA_GROUP, int PAGE_SIZE, typename ArchTag>
__global__ void mega_attention_block_kernel(const __grid_constant__ MegaAttentionBlockParams params) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const bool proj_mma = (params.hidden_size % kMlpBK == 0);
  const bool oproj_mma = ((params.num_q_heads * params.head_dim) % kMlpBK == 0);
  const MegaPlanDevice& plan = params.plan;
  for (;;) {
    const int idx = next_task(plan.queue);
    if (idx < 0) break;
    const exec::TileTask task = plan.queue.tasks[idx];
    if (task.op_id == 0) {
      if (proj_mma) {
        attention_proj_tile<kAttnBM, kAttnBN, kAttnStages>(params, task.tile_m, task.tile_n, smem_raw);
      } else {
        attention_proj_simt<T>(params, task.tile_m, task.tile_n, reinterpret_cast<T*>(smem_raw),
                               tid, lane, warp);
      }
      signal_task(plan.queue, task);
    } else if (task.op_id == 1) {
      wait_task(plan.queue, task);
      attention_append_epilogue<T>(params, task.tile_m, tid, lane, warp);
      signal_task(plan.queue, task);
    } else if (task.op_id == 2) {
      wait_task(plan.queue, task);
      attention_decode_heads<T>(params, task.tile_m, tid, lane, warp);
      signal_task(plan.queue, task);
    } else {
      wait_task(plan.queue, task);
      if (oproj_mma) {
        attention_oproj_tile<kAttnBM, kAttnBN, kAttnStages>(params, task.tile_m, task.tile_n, smem_raw);
      } else {
        attention_oproj_simt<T>(params, task.tile_m, task.tile_n, lane, warp);
      }
    }
  }
  mega_finish(plan);
}

template <typename T, typename W, int BM, int BN, int BK, int STAGES, bool kFusedRouter, typename ArchTag>
__global__ void mega_moe_local_kernel(const __grid_constant__ MegaMoELocalParams params) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int H = params.hidden_size;
  const int I = params.intermediate_size;
  T* const s_norm = reinterpret_cast<T*>(smem_raw);
  T* const s_inter = s_norm + H;
  float* const s_acc = reinterpret_cast<float*>(smem_raw + (((H + I) * static_cast<int>(sizeof(T)) + 3) & ~3));
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const MegaPlanDevice& plan = params.plan;
  for (;;) {
    const int idx = next_task(plan.queue);
    if (idx < 0) break;
    const exec::TileTask task = plan.queue.tasks[idx];
    moe_row<T, W>(params, task.tile_m, nullptr, 0.0f, s_norm, s_inter, s_acc, tid, lane, warp, nullptr);
  }
  mega_finish(plan);
}

template <typename T, typename W, int BM, int BN, int BK, int NUM_RANKS_PER_NODE, typename ArchTag>
__global__ void mega_moe_ep_kernel(const __grid_constant__ MegaMoEEPParams params) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int H = params.local.hidden_size;
  const int I = params.local.intermediate_size;
  T* const s_norm = reinterpret_cast<T*>(smem_raw);
  T* const s_inter = s_norm + H;
  float* const s_acc = reinterpret_cast<float*>(smem_raw + (((H + I) * static_cast<int>(sizeof(T)) + 3) & ~3));
  T* const ep_stage = static_cast<T*>(params.ep.nvl_buffers[params.ep.rank]);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const MegaPlanDevice& plan = params.local.plan;
  for (;;) {
    const int idx = next_task(plan.queue);
    if (idx < 0) break;
    const exec::TileTask task = plan.queue.tasks[idx];
    moe_row<T, W>(params.local, task.tile_m, nullptr, 0.0f, s_norm, s_inter, s_acc, tid, lane, warp,
                  ep_stage);
  }
  mega_finish(plan);
}

template <typename T, typename ArchTag>
__global__ void mega_decoder_layer_kernel(const __grid_constant__ MegaPlanDevice plan,
                                          const __grid_constant__ MegaAttentionBlockParams attn,
                                          const __grid_constant__ MegaMLPParams mlp,
                                          const __grid_constant__ MegaMoELocalParams moe) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int H = attn.hidden_size;
  const int I = mlp.intermediate_size;
  T* const s_norm = reinterpret_cast<T*>(smem_raw);
  T* const s_inter = s_norm + H;
  float* const s_acc = reinterpret_cast<float*>(smem_raw + (((H + I) * static_cast<int>(sizeof(T)) + 3) & ~3));
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const bool use_moe = moe.m > 0;
  const bool proj_mma = (H % kMlpBK == 0);
  const bool oproj_mma = ((attn.num_q_heads * attn.head_dim) % kMlpBK == 0);
  const bool ffn_mma = (mlp.hidden_size % kMlpBK == 0) && (I % kMlpBK == 0);
  const int64_t stream_ld = mlp.ldo != 0 ? mlp.ldo : H;
  const bool publish = !use_moe && mlp.residual != nullptr && mlp.residual != mlp.out;
  for (;;) {
    const int idx = next_task(plan.queue);
    if (idx < 0) break;
    const exec::TileTask task = plan.queue.tasks[idx];
    if (task.op_id == 0) {
      if (proj_mma) {
        attention_proj_tile<kAttnBM, kAttnBN, kAttnStages>(attn, task.tile_m, task.tile_n, smem_raw);
      } else {
        attention_proj_simt<T>(attn, task.tile_m, task.tile_n, s_norm, tid, lane, warp);
      }
      signal_task(plan.queue, task);
    } else if (task.op_id == 1) {
      wait_task(plan.queue, task);
      attention_append_epilogue<T>(attn, task.tile_m, tid, lane, warp);
      signal_task(plan.queue, task);
    } else if (task.op_id == 2) {
      wait_task(plan.queue, task);
      attention_decode_heads<T>(attn, task.tile_m, tid, lane, warp);
      signal_task(plan.queue, task);
    } else if (task.op_id == 3) {
      wait_task(plan.queue, task);
      if (oproj_mma) {
        attention_oproj_tile<kAttnBM, kAttnBN, kAttnStages>(attn, task.tile_m, task.tile_n, smem_raw);
      } else {
        attention_oproj_simt<T>(attn, task.tile_m, task.tile_n, lane, warp);
      }
      signal_task(plan.queue, task);
    } else if (task.op_id == 4) {
      wait_task(plan.queue, task);
      if (ffn_mma) {
        mlp_gate_tile<kMlpBM, kMlpBN1, kMlpStages>(mlp, task.tile_m, task.tile_n, smem_raw);
      } else {
        mlp_gate_tile_simt<T, T>(mlp, task.tile_m, task.tile_n, s_norm, tid, lane, warp);
      }
      signal_task(plan.queue, task);
    } else if (task.op_id == 5) {
      wait_task(plan.queue, task);
      if (ffn_mma) {
        mlp_down_tile<kMlpBM, kMlpBN2, kMlpStages>(
            mlp, task.tile_m, task.tile_n, smem_raw,
            publish ? static_cast<Bf16*>(const_cast<void*>(mlp.residual)) : nullptr, stream_ld);
      } else {
        mlp_down_tile_simt<T, T>(mlp, task.tile_m, task.tile_n, lane, warp);
        if (publish) {
          const int r0 = task.tile_m * kMlpBM;
          const int rend = r0 + kMlpBM < mlp.m ? r0 + kMlpBM : mlp.m;
          const T* src = static_cast<const T*>(mlp.out);
          T* dst = static_cast<T*>(const_cast<void*>(mlp.residual));
          for (int r = r0; r < rend; ++r) {
            for (int h = tid; h < H; h += blockDim.x) {
              dst[static_cast<int64_t>(r) * stream_ld + h] =
                  src[static_cast<int64_t>(r) * stream_ld + h];
            }
          }
          __syncthreads();
        }
      }
    } else {
      wait_task(plan.queue, task);
      moe_row<T, T>(moe, task.tile_m, static_cast<const T*>(mlp.pre_norm_weight), mlp.pre_norm_eps,
                    s_norm, s_inter, s_acc, tid, lane, warp, nullptr);
      if (moe.out != mlp.residual) {
        const T* src = static_cast<const T*>(moe.out) + static_cast<int64_t>(task.tile_m) * H;
        T* dst = static_cast<T*>(const_cast<void*>(mlp.residual)) + static_cast<int64_t>(task.tile_m) * H;
        for (int h = tid; h < H; h += blockDim.x) dst[h] = src[h];
        __syncthreads();
      }
    }
  }
  mega_finish(plan);
}

}  // namespace kops::mega

// =============================================================================
// Host entry points (csrc/mega/ops.h)
// =============================================================================
namespace kops::mega {

Status mega_workspace_requirements(MegaKind kind, const void* params, const LaunchContext& ctx,
                                   WorkspaceRequirements* out) {
  if (out == nullptr) return bad("workspace_requirements", "null output");
  Layout L;
  Status st = layout_for(kind, params, ctx, &L);
  if (st != Status::kSuccess) return st;
  *out = WorkspaceRequirements{};
  out->device_scratch_bytes = 0;
  out->device_persistent_bytes = persistent_bytes(L);
  out->host_pinned_bytes = 0;
  out->alignment = 256;
  return Status::kSuccess;
}

Status mega_prepare(MegaKind kind, const void* params, const Workspace& ws,
                    const LaunchContext& ctx, MegaPlan* plan) {
  Status st = check_arch(ctx);
  if (st != Status::kSuccess) return st;
  Layout L;
  st = layout_for(kind, params, ctx, &L);
  if (st != Status::kSuccess) return st;
  if (exec::stream_is_capturing(ctx.stream)) {
    return detail::set_error(Status::kInvalidArgument,
                             "mega.prepare: called while the stream is capturing");
  }
  return prepare_common(kind, params, ws, ctx, plan, L);
}

Status mega_run_mlp(const MegaPlan& plan, const MegaMLPParams& params, const LaunchContext& ctx) {
  Status st = check_arch(ctx);
  if (st != Status::kSuccess) return st;
  st = validate_mlp(&params);
  if (st != Status::kSuccess) return st;
  HostPlan hp;
  st = plan_enter(plan, MegaKind::kMLP, params.m, ctx, &hp);
  if (st != Status::kSuccess) return st;
  MegaMLPParams q = params;
  q.plan = plan.device;   // prepared tile schedule / counters / slots
  if (q.ldx == 0) q.ldx = q.hidden_size;
  if (q.ldo == 0) q.ldo = q.hidden_size;
  if (q.intermediate == nullptr) q.intermediate = hp.slots[0];
  if (q.intermediate == nullptr) return bad("run_mlp", "no intermediate buffer");
  void* args[] = {&q};
  const void* kfn = kernel_for(MegaKind::kMLP);
  return launch_common(hp, ctx, kfn, args);
}

Status mega_run_attention_block(const MegaPlan& plan, const MegaAttentionBlockParams& params,
                                const LaunchContext& ctx) {
  Status st = check_arch(ctx);
  if (st != Status::kSuccess) return st;
  st = validate_attention(&params);
  if (st != Status::kSuccess) return st;
  HostPlan hp;
  st = plan_enter(plan, MegaKind::kAttentionBlock, params.m, ctx, &hp);
  if (st != Status::kSuccess) return st;
  MegaAttentionBlockParams q = params;
  q.plan = plan.device;
  void* args[] = {&q};
  const void* kfn = kernel_for(MegaKind::kAttentionBlock);
  return launch_common(hp, ctx, kfn, args);
}

Status mega_run_moe_local(const MegaPlan& plan, const MegaMoELocalParams& params,
                          const LaunchContext& ctx) {
  Status st = check_arch(ctx);
  if (st != Status::kSuccess) return st;
  st = validate_moe(&params, true);
  if (st != Status::kSuccess) return st;
  HostPlan hp;
  st = plan_enter(plan, MegaKind::kMoELocal, params.m, ctx, &hp);
  if (st != Status::kSuccess) return st;
  MegaMoELocalParams q = params;
  q.plan = plan.device;
  void* args[] = {&q};
  const void* kfn = kernel_for(MegaKind::kMoELocal);
  return launch_common(hp, ctx, kfn, args);
}

Status mega_run_moe_ep(const MegaPlan& plan, const MegaMoEEPParams& params,
                       const LaunchContext& ctx) {
  Status st = check_arch(ctx);
  if (st != Status::kSuccess) return st;
  st = validate_ep(&params);
  if (st != Status::kSuccess) return st;
  HostPlan hp;
  st = plan_enter(plan, MegaKind::kMoEEP, params.local.m, ctx, &hp);
  if (st != Status::kSuccess) return st;
  MegaMoEEPParams q = params;
  q.local.plan = plan.device;
  void* args[] = {&q};
  const void* kfn = kernel_for(MegaKind::kMoEEP);
  return launch_common(hp, ctx, kfn, args);
}

Status mega_run_decoder(const MegaPlan& plan, const MegaAttentionBlockParams& attn,
                        const MegaMLPParams& mlp, const MegaMoELocalParams& moe,
                        const LaunchContext& ctx) {
  Status st = check_arch(ctx);
  if (st != Status::kSuccess) return st;
  st = validate_attention(&attn);
  if (st != Status::kSuccess) return st;
  st = validate_mlp(&mlp);
  if (st != Status::kSuccess) return st;
  if (attn.m != mlp.m) return bad("run_decoder", "attn.m != mlp.m");
  const bool use_moe = moe.m > 0;
  if (use_moe) {
    st = validate_moe(&moe, true);
    if (st != Status::kSuccess) return st;
    if (moe.m != attn.m || moe.hidden_size != attn.hidden_size ||
        moe.intermediate_size != mlp.intermediate_size) {
      return bad("run_decoder", "MoE/attention/MLP shapes disagree");
    }
  }
  HostPlan hp;
  st = plan_enter(plan, MegaKind::kDecoder, attn.m, ctx, &hp);
  if (st != Status::kSuccess) return st;

  MegaAttentionBlockParams a = attn;
  MegaMLPParams m = mlp;
  MegaMoELocalParams mo = use_moe ? moe : MegaMoELocalParams{};
  void* stream_residual = a.residual != nullptr ? a.residual : a.out;
  // The decoder DAG writes the post-attention residual stream back into
  // attn.residual (or attn.out when residual == null).  The FFN consumes that
  // same stream; mlp.pre_norm_weight is the layer's pre-FFN norm weight, also
  // used by the MoE body because MegaMoELocalParams has no norm field.
  a.plan = plan.device;   // attention helpers read the plan slots (QKV / attn buffers)
  m.plan = plan.device;
  mo.plan = plan.device;
  a.residual = stream_residual;
  m.x = stream_residual;
  m.residual = stream_residual;   // the kernel copies the FFN result into this stream
  // m.out stays the caller's output: the FFN writes it, then the kernel copies it to the stream.
  if (m.intermediate == nullptr) m.intermediate = hp.slots[2];
  if (m.intermediate == nullptr) return bad("run_decoder", "no MLP intermediate");
  if (use_moe) {
    mo.hidden = stream_residual;
    mo.out = m.out;               // the MoE writes the caller output; the kernel copies it to the stream
  }
  MegaPlanDevice pdev = plan.device;
  void* args[] = {&pdev, &a, &m, &mo};
  const void* kfn = kernel_for(MegaKind::kDecoder);
  return launch_common(hp, ctx, kfn, args);
}

}  // namespace kops::mega
