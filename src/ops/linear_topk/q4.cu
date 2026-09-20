#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cstdint>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

struct Q4KSplitTopKOutput {
    std::uint64_t* partial_keys;
    const std::int32_t* row_to_global_ids;
    // 本卡头片在**逻辑整表**里的首行行号：tp2 下优化头被行分片，映射表是复制的整表，
    // 所以卡内第 i 行取 row_to_global_ids[map_row_base + i]。单卡（未分片）为 0。
    std::int32_t map_row_base;
    std::int32_t producer_groups;
    std::int32_t columns;

    template <int Capacity>
    __device__ __forceinline__ void store(std::int32_t row0, std::int32_t column0,
                                          float4 values) const {
        const std::int32_t group = row0 / kLinearTopK;
        const std::int32_t rank0 = row0 % kLinearTopK;
        const auto put           = [&](std::int32_t row, std::int32_t rank, std::int32_t column,
                             float value) {
            if (column >= columns) { return; }
            const std::int64_t offset =
                (static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK + rank;
            partial_keys[offset] =
                score_id_order_key(value, row_to_global_ids[map_row_base + row]);
        };
        put(row0, rank0, column0, values.x);
        put(row0, rank0, column0 + 1, values.y);
        put(row0 + 8, rank0 + 8, column0, values.z);
        put(row0 + 8, rank0 + 8, column0 + 1, values.w);
    }
};

template <int Capacity>
void launch_ksplit(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                   std::int32_t map_row_base, const LinearTopKWorkspace& workspace,
                   cudaStream_t stream) {
    using Geometry             = Q4DraftHeadGeometry<kLinearTopKHidden>;
    using Schedule             = Q4DraftSmallTSchedule;
    constexpr int kTileColumns = ((Capacity + 7) / 8) * 8;
    // grid 按**本卡实际行数**铺：workspace.producer_groups 是运行期由 head.n 推出的组数，
    // 而 rows_per_producer == kLinearTopKDirectRows 时 1 个 CTA 恰好归约 1 组。写死整表的
    // Q4DraftHeadGeometry::kOutputRows 会让 tp2 的行分片头（每卡一半）多铺一倍 CTA：
    // 越界读权重平面、并且把 group 索引写到 producer_groups 之外，两侧都是非法访存。
    const std::int32_t blocks = workspace.producer_groups;
    const Q4KSplitTopKOutput output{static_cast<std::uint64_t*>(workspace.partial_keys.data),
                                    static_cast<const std::int32_t*>(row_to_global_ids.data),
                                    map_row_base,
                                    workspace.producer_groups, hidden.ne[1]};
    q4_small_t_mma_kernel<Geometry, kTileColumns, Capacity, Q4KSplitTopKOutput,
                          Q4SmallTMmaIdentityRows, true>
        <<<blocks, Schedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(hidden.data),
                                                     static_cast<const std::uint8_t*>(head.qdata),
                                                     static_cast<const std::uint8_t*>(head.scales),
                                                     nullptr, output, Q4SmallTMmaIdentityRows{}, hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

using Launch = void (*)(const Tensor&, const Weight&, const Tensor&, std::int32_t,
                        const LinearTopKWorkspace&, cudaStream_t);

template <std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{&launch_ksplit<8 * (1 + I)>...};
}

constexpr auto launchers = make_launchers(std::make_index_sequence<2>{});

} // namespace

void linear_topk_q4_launch(const Tensor& hidden, const Weight& head,
                           const Tensor& row_to_global_ids, std::int32_t map_row_base,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    if (workspace.rows_per_producer == kLinearTopKDirectRows) {
        launchers[(hidden.ne[1] - 1) / 8](hidden, head, row_to_global_ids, map_row_base, workspace,
                                          stream);
    } else {
        linear_topk_q4_m64_launch(hidden, head, row_to_global_ids, map_row_base, workspace, stream);
    }
}
} // namespace ninfer::ops::detail
