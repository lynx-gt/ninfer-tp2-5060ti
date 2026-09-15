#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include "core/cyclic_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct DFlashPersistentState {
    // 本 fork 没有上游的 StateImage 架构，draft 的滑窗 KV 由本结构自己持有（值成员）。
    CyclicKVCache local;
    std::optional<qwen3_6::PagedKVCache> full;
    Tensor prefill_features;
    Tensor prefill_positions;
    Tensor pending_features;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
    // 复用检查点：local 内 slot -> slot 的整槽拷贝（master 的 DFlash2 路径按此调用）。
    void save_rewrite_checkpoint(std::int32_t source_slot, std::int32_t destination_slot,
                                 cudaStream_t stream);
    // 本 fork 的调用形态：一个 lane 的 live 槽 <-> 它的检查点槽。
    // local 规划成 2 * max_concurrency 个 lane：前一半是 live，后一半是同 lane 的检查点。
    void save_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream);
    void restore_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream);
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
