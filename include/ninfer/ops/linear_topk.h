#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the caller-owned transient capacity required by linear_topk for every column count in the
 * inclusive `[min_columns,max_columns]` interval. The registered profile is identified by
 * its weight format and exact `[head_rows,input_rows]` geometry. Column counts are positive.
 */
[[nodiscard]] std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                               std::int32_t input_rows,
                                                               std::int32_t min_columns,
                                                               std::int32_t max_columns);

/**
 * @brief Projects independent matrix columns through a full vocabulary head and returns
 * the stable top sixteen scores and global token ids per column.
 *
 * @details For any positive column count `U`, `hidden` is contiguous BF16 `[5120,U]`, `head` is
 * either W8G32_F16S or FP8_E4M3FN_ROW_BF16S `[248320,5120]`, and `valid_rows` is 248077. For every
 * `t in [0,U)` and valid vocabulary row `v`, the ideal score is
 *
 * @f[
 *   s_{v,t}=\sum_{k=0}^{5119}\mathrm{FP32Dequant}(head)_{v,k}
 *                              \mathrm{FP32}(hidden_{k,t}).
 * @f]
 *
 * `candidate_scores` is contiguous FP32 `[16,U]` and `candidate_ids` is contiguous I32
 * `[16,U]`. Rank is stored fastest; a caller may view the output as `[16,K,B]` when `U=K*B`.
 * Candidates are ordered by descending computed score, with exact score ties resolved by lower
 * global token id. Physical rows `[valid_rows,248320)` never participate. Projection uses the
 * registered A16 arithmetic profile; candidate scores are returned directly in FP32 and no dense
 * vocabulary-logit tensor is an observable intermediate.
 *
 * All tensors and every weight plane are preserved except that both output tensors are completely
 * overwritten. Inputs, outputs, weight planes, and workspace must not overlap. The Op has no
 * persistent state and performs no internal device allocation.
 */
void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream);

/**
 * @brief Projects through the optimized 131072-row proposal head and returns stable top sixteen
 * scores with shortlist rows mapped to global token ids.
 *
 * @details The tensor contract is the same as the full-head overload except that `head` is
 * Q4G64_F16S `[131072,5120]` and every head row participates. `row_to_global_ids` is contiguous
 * I32 `[131072]`; it contains distinct ids in `[0,248077)` and maps each head row to the id
 * used for output and tie-breaking. Artifact binding establishes the map's range and uniqueness.
 *
 * Under tensor parallelism the head is a ROW SHARD of that same logical head: `head.n` is the
 * shard's row count and `map_row_base` is the shard's first row in the logical head, so shard row
 * `i` is mapped through `row_to_global_ids[map_row_base + i]` and the candidates come out as
 * global token ids either way. The map itself is never sharded -- it is replicated on every
 * device, because the winning id can name a row held by any shard. `map_row_base` is 0 when the
 * head is not sharded.
 */
void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 std::int32_t map_row_base, Tensor& candidate_ids, Tensor& candidate_scores,
                 WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
