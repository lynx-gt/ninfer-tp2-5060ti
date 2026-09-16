#include "ninfer/ops/topk_pair_merge.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kTopK       = 16;
constexpr const char* kOp          = "topk_pair_merge";

__global__ void topk_pair_merge_kernel(const std::int32_t* __restrict__ ids_a,
                                       const float* __restrict__ scores_a,
                                       const std::int32_t* __restrict__ ids_b,
                                       const float* __restrict__ scores_b,
                                       std::int32_t* __restrict__ ids_out,
                                       float* __restrict__ scores_out, std::int32_t columns,
                                       std::int32_t remote_id_offset) {
    const std::int32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) { return; }

    float best_scores[kTopK];
    std::int32_t best_ids[kTopK];
    int count = 0;
    const auto consider = [&](std::int32_t id, float score) {
        if (count < kTopK) {
            int slot = count;
            while (slot > 0 && (best_scores[slot - 1] < score ||
                                (best_scores[slot - 1] == score && best_ids[slot - 1] > id))) {
                best_scores[slot] = best_scores[slot - 1];
                best_ids[slot]    = best_ids[slot - 1];
                --slot;
            }
            best_scores[slot] = score;
            best_ids[slot]    = id;
            ++count;
            return;
        }
        if (score < best_scores[kTopK - 1] ||
            (score == best_scores[kTopK - 1] && id >= best_ids[kTopK - 1])) {
            return;
        }
        int slot = kTopK - 1;
        while (slot > 0 && (best_scores[slot - 1] < score ||
                            (best_scores[slot - 1] == score && best_ids[slot - 1] > id))) {
            best_scores[slot] = best_scores[slot - 1];
            best_ids[slot]    = best_ids[slot - 1];
            --slot;
        }
        best_scores[slot] = score;
        best_ids[slot]    = id;
    };

    for (int row = 0; row < kTopK; ++row) {
        const std::int32_t index = row * columns + column;
        consider(ids_a[index], scores_a[index]);
        // 第二张头的行号是卡内行号，先搬到全局行号再比，否则同分次序和输出 id 都错。
        consider(ids_b[index] + remote_id_offset, scores_b[index]);
    }
    for (int row = 0; row < kTopK; ++row) {
        const std::int32_t index = row * columns + column;
        ids_out[index]           = best_ids[row];
        scores_out[index]        = best_scores[row];
    }
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t columns, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != kTopK || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string(kOp) + ": invalid " + name);
    }
}

} // namespace

void topk_pair_merge(const Tensor& ids_a, const Tensor& scores_a, const Tensor& ids_b,
                     const Tensor& scores_b, Tensor& ids_out, Tensor& scores_out,
                     std::int32_t remote_id_offset, cudaStream_t stream) {
    const std::int32_t columns = ids_a.ne[1];
    if (columns <= 0 || ids_a.ne[0] != kTopK) {
        throw std::invalid_argument("topk_pair_merge: invalid candidate geometry");
    }
    require_matrix(ids_a, DType::I32, columns, "ids_a");
    require_matrix(ids_b, DType::I32, columns, "ids_b");
    require_matrix(scores_a, DType::FP32, columns, "scores_a");
    require_matrix(scores_b, DType::FP32, columns, "scores_b");
    require_matrix(ids_out, DType::I32, columns, "ids_out");
    require_matrix(scores_out, DType::FP32, columns, "scores_out");

    constexpr int kThreads = 128;
    const int blocks       = (columns + kThreads - 1) / kThreads;
    topk_pair_merge_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids_a.data), static_cast<const float*>(scores_a.data),
        static_cast<const std::int32_t*>(ids_b.data), static_cast<const float*>(scores_b.data),
        static_cast<std::int32_t*>(ids_out.data), static_cast<float*>(scores_out.data), columns,
        remote_id_offset);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
