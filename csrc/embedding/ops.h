#pragma once
// =============================================================================
// kernel-ops | csrc/embedding/ops.h                                   Catalog: X02
// Requires: common/tensor.h, common/status.h, embedding/embedding_kernels.cuh.
// =============================================================================

namespace kops::embed {

Status embedding_lookup(const Tensor& out, const Tensor& token_ids, const Tensor& weight,
                        int64_t vocab_start, int64_t vocab_end, float scale,
                        const LaunchContext& ctx);
Status embedding_with_position(const Tensor& out, const Tensor& token_ids, const Tensor& positions,
                               const Tensor& token_weight, const Tensor& pos_weight,
                               int64_t vocab_start, int64_t vocab_end, int32_t position_offset,
                               bool is_first_tp_rank, const LaunchContext& ctx);
Status scatter_rows(const Tensor& out, const Tensor& src, const Tensor& dst_rows,
                    const LaunchContext& ctx);
Status gather_rows(const Tensor& out, const Tensor& input, const Tensor& rows,
                   const LaunchContext& ctx);
Status logits_postprocess(const Tensor& out, const Tensor& logits, float scale,
                          float final_softcap, int64_t vocab_start, int64_t vocab_size,
                          const LaunchContext& ctx);

}  // namespace kops::embed
