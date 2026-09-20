#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/topk_pair_merge.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("masked draft schedule requires its weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        nvtx::ScopedRange append_range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                                       static_cast<std::uint64_t>(columns));
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }

        const int projected_width = Config::coherent_selector ? local_width : width;
        const Tensor input        = Config::coherent_selector && replace_local_window
                                        ? features.slice(1, local_offset, local_width)
                                        : features;

        const auto roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, projected_width * batch);
        Tensor projected = roots.projected;
        Tensor context   = roots.normalized;
        ops::linear(input.view({Config::feature_rows, projected_width * batch}),
                    state.execution.model.dflash->feature_projection, projected,
                    state.execution.device.stream);
        ops::rmsnorm(projected, state.execution.model.dflash->context_norm, Config::rms_epsilon,
                     false, context, state.execution.device.stream);

        if constexpr (Config::coherent_selector) {
            std::array<ops::ContextKVMaterializeLayerView, Config::layers> layers;
            for (int layer = 0; layer < Config::layers; ++layer) {
                const auto& weights = state.execution.model.dflash->layers[layer];
                layers[layer]       = {weights.context_key, weights.context_value, weights.key_norm,
                                       dflash_state(state).local_layer(layer)};
            }
            const Tensor local_positions =
                replace_local_window ? positions.slice(0, local_offset, local_width) : positions;
            ops::context_kv_materialize(context.view({Config::hidden, local_width, batch}),
                                        local_positions, local_counts, lanes, layers,
                                        {local_envelope.min_count, local_envelope.max_count},
                                        state.execution.work, state.execution.device.stream);
        } else {
            for (int layer = 0; layer < Config::layers; ++layer) {
                auto layer_scope = state.execution.work.scope();
                const auto& weight =
                    state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
                const bool local_layer  = layer < Config::local_layers;
                const int layer_width   = local_layer ? local_width : width;
                const int layer_columns = layer_width * batch;
                Tensor layer_context    = local_layer && replace_local_window
                                              ? context.slice(1, local_offset, local_width)
                                              : context;
                Tensor layer_positions  = local_layer && replace_local_window
                                              ? positions.slice(0, local_offset, local_width)
                                              : positions;
                auto layer_roots        = workspace_recipe::dflash_context_layer<Config>(
                    state.execution.work, layer_columns);
                Tensor key_raw =
                    layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
                Tensor value =
                    layer_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, layer_columns});
                Tensor value_flat = value.view({Config::kv_size, layer_columns});
                ops::linear_pair(layer_context, weight.context_key, weight.context_value, key_flat,
                                 value_flat, state.execution.device.stream);
                Tensor key =
                    layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(layer_positions.view({layer_columns}), Config::head_dim,
                          Config::rope_theta, key, state.execution.device.stream);
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, layer_width, batch});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, layer_width, batch});
                Tensor position_batch = layer_positions.view({layer_width, batch});
                if (local_layer) {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                        dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                        state.execution.device.stream);
                } else {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                        dflash_state(state).full_batch_layer(0), state.execution.device.stream);
                }
            }
        }
    }
}

void prepare_dynamic_branch(DeviceContext& device, WorkspaceArena& work, const Tensor& residual,
                            const Tensor& norm, float eps,
                            const qwen3_6::DFlash2DynamicConvWeights& weights,
                            workspace_recipe::DFlash2BranchRoots& branch) {
    auto scope      = work.scope();
    const int width = residual.ne[1], batch = residual.ne[2];
    WorkspaceArena scratch(work.alloc_bytes(
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(width, width, batch,
                                                                          batch)));
    ops::rmsnorm_dynamic_grouped_conv_prepare(
        residual, norm, eps, weights.base_kernel, weights.kernel_projection, branch.prepared,
        branch.finish_delta, scratch, device.stream);
}

void finish_dynamic_branch(DeviceContext& device, WorkspaceArena& work, const Tensor& input,
                           const Weight& projection,
                           const qwen3_6::DFlash2DynamicConvWeights& weights,
                           const Tensor& finish_delta, Tensor& residual) {
    auto scope      = work.scope();
    const int width = input.ne[1], batch = input.ne[2];
    WorkspaceArena scratch(work.alloc_bytes(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
        input.ne[0], width, width, batch, batch)));
    ops::linear_dynamic_grouped_conv_add(input, projection, weights.base_kernel, finish_delta,
                                         residual, scratch, device.stream);
}

// DFlash2 的候选选择段（tp1 与 tp2 草稿共用）：rank 0 用自己的半词表取 top-16，tp2 下再与
// rank 1 那半合并成全局 top-16，随后跑隐藏投影与 coherent selector。
//
// `peer_hidden` 为 tp2 草稿自算出的 rank 1 hidden（两卡逐位相同，省掉一次 82KB 拷贝与一对事件）；
// 传 nullptr 时按原路径把 rank 0 的 hidden 拷到 rank 1 再算（tp1 及混合路径保持不变）。
//
// 模板参必须是 Variant（而非具体权重类型）：`candidate_selector` 是 DFlash2 专有成员，经由 V
// 取得 weights 才让这些访问成为依赖名，35B 的 DFlash 变体因此不会在解析期被实例化检查。
template <class V>
void dflash2_select_candidates(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                               const Tensor& hidden, const Tensor* peer_hidden, int batch, int k) {
    const typename V::ModelView::DFlash& weights = *state.execution.model.dflash;
    auto& work                = state.execution.work;
    const cudaStream_t stream = state.execution.device.stream;
    const int mask_columns    = k * batch;
    Tensor candidates         = frame.candidate_ids.slice(2, 0, batch);
    Tensor ids_flat           = candidates.view({16, mask_columns});
    Tensor scores             = work.alloc(DType::FP32, {16, mask_columns});
    // tp2：词表头按行分片（每卡 124160 行），草稿的候选选择必须取到**全局** top-16：
    // 两卡各在自己那半词表上取 top-16，再把两路合并（分数降序、同分取更小 token id）。
    std::optional<TpExecution> tp = tp_execution(state.execution);
    const std::int32_t full_valid = TextConfig::token_domain;
    const std::int32_t shard_rows = state.execution.model.output_head.n;
    const std::int32_t local_valid = tp ? std::min(full_valid, shard_rows) : full_valid;
    if (state.execution.proposal_head == ProposalHead::Full) {
        ops::linear_topk(hidden, state.execution.model.output_head, local_valid, ids_flat, scores,
                         work, stream);
    } else {
        const auto& head = *state.execution.model.optimized_proposal;
        // 优化头在 tp2 下是同一张逻辑表的行分片（每卡 65536 行），而 `draft_head_token_ids` 是
        // **复制**的整表：本卡第 i 行取 row_to_global_ids[map_row_base + i]，输出即全局 token id。
        // rank 0 的 map_row_base = 0（与单卡逐位一致）。
        ops::linear_topk(hidden, head.head, head.token_ids, 0, ids_flat, scores, work, stream);
    }
    if (tp && state.execution.proposal_head != ProposalHead::Full) {
        // rank 1 那半张头走同一条逻辑表，map_row_base 取 rank 0 的片宽（= 本卡头片的全局行起点）。
        // 与 W8 整词表那路相反：那一路 `linear_topk` 返回的是卡内行号，所以合并要补 shard_rows；
        // 优化头这一路映射在核内完成，两卡出来的都是全局 token id，合并偏移必须给 0，否则会
        // 把已经全局化的 id 再加一次偏移，候选全部指错 token。
        const auto& head      = *state.execution.model.optimized_proposal;
        const auto& peer_head = *tp->weights->optimized_proposal;
        CUDA_CHECK(cudaEventRecord(tp->events->inputs_ready(0), stream));
        CUDA_CHECK(cudaSetDevice(tp->device->device));
        CUDA_CHECK(cudaStreamWaitEvent(tp->device->stream, tp->events->inputs_ready(0), 0));
        Tensor peer_ids    = tp->work->alloc(DType::I32, {16, mask_columns});
        Tensor peer_scores = tp->work->alloc(DType::FP32, {16, mask_columns});
        Tensor peer_hidden_owned;
        Tensor peer_input = {};
        if (peer_hidden != nullptr) {
            peer_input = *peer_hidden;
        } else {
            peer_hidden_owned = tp->work->alloc(DType::BF16, {TextConfig::hidden, mask_columns});
            CUDA_CHECK(cudaMemcpyAsync(peer_hidden_owned.data, hidden.data, peer_hidden_owned.bytes(),
                                       cudaMemcpyDeviceToDevice, tp->device->stream));
            peer_input = peer_hidden_owned;
        }
        auto peer_scope = tp->work->scope();
        ops::linear_topk(peer_input, peer_head.head, peer_head.token_ids, head.head.n, peer_ids,
                         peer_scores, *tp->work, tp->device->stream);
        CUDA_CHECK(cudaEventRecord(tp->events->inputs_ready(1), tp->device->stream));
        CUDA_CHECK(cudaSetDevice(state.execution.device.device));
        CUDA_CHECK(cudaStreamWaitEvent(stream, tp->events->inputs_ready(1), 0));
        Tensor remote_ids    = work.alloc(DType::I32, {16, mask_columns});
        Tensor remote_scores = work.alloc(DType::FP32, {16, mask_columns});
        CUDA_CHECK(cudaMemcpyAsync(remote_ids.data, peer_ids.data, remote_ids.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(remote_scores.data, peer_scores.data, remote_scores.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        ops::topk_pair_merge(ids_flat, scores, remote_ids, remote_scores, ids_flat, scores, 0,
                             stream);
    } else if (tp && state.execution.proposal_head == ProposalHead::Full) {
        const auto peer_valid = full_valid - local_valid;
        if (peer_valid > 0) {
            // 跨卡这段是手工排的，没有集合通信帮忙排序，必须自己用事件把两条流串起来，
            // 否则（1）rank1 拿 hidden 时 rank0 还没算完、（2）rank0 拷候选时 rank1 还没算完，
            // 两个方向都会读到上一轮的残留 —— 实测漏掉时 rank1 候选整片读到 0，草稿大面积失配。
            // 事件用 PeerEvents 的 inputs_ready：语义就是"某卡的数据已就绪"，与集合通信同一套
            // 复用纪律（先发 wait 再发 record）。
            CUDA_CHECK(cudaEventRecord(tp->events->inputs_ready(0), stream));
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            CUDA_CHECK(cudaStreamWaitEvent(tp->device->stream, tp->events->inputs_ready(0), 0));
            // rank 1：用它自己那半头、在同一份 hidden（自算或拷过去）上取 top-16
            Tensor peer_ids    = tp->work->alloc(DType::I32, {16, mask_columns});
            Tensor peer_scores = tp->work->alloc(DType::FP32, {16, mask_columns});
            Tensor peer_hidden_owned;
            Tensor peer_input = {};
            if (peer_hidden != nullptr) {
                peer_input = *peer_hidden;
            } else {
                peer_hidden_owned =
                    tp->work->alloc(DType::BF16, {TextConfig::hidden, mask_columns});
                CUDA_CHECK(cudaMemcpyAsync(peer_hidden_owned.data, hidden.data,
                                           peer_hidden_owned.bytes(), cudaMemcpyDeviceToDevice,
                                           tp->device->stream));
                peer_input = peer_hidden_owned;
            }
            auto peer_scope = tp->work->scope();
            ops::linear_topk(peer_input, tp->weights->output_head, peer_valid, peer_ids,
                             peer_scores, *tp->work, tp->device->stream);
            CUDA_CHECK(cudaEventRecord(tp->events->inputs_ready(1), tp->device->stream));
            CUDA_CHECK(cudaSetDevice(state.execution.device.device));
            CUDA_CHECK(cudaStreamWaitEvent(stream, tp->events->inputs_ready(1), 0));
            // 把 rank 1 的候选搬到 rank 0（peer 访问已在构造时打开），再合并
            Tensor remote_ids    = work.alloc(DType::I32, {16, mask_columns});
            Tensor remote_scores = work.alloc(DType::FP32, {16, mask_columns});
            CUDA_CHECK(cudaMemcpyAsync(remote_ids.data, peer_ids.data, remote_ids.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(remote_scores.data, peer_scores.data, remote_scores.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
            // 合并成全局 top-16。`shard_rows` 作为 rank1 候选的全局行偏移：linear_topk 返回的是
            // 本卡头内的行号，rank1 那半张头的行号必须加上它那一片的全局起点。
            ops::topk_pair_merge(ids_flat, scores, remote_ids, remote_scores, ids_flat, scores,
                                 shard_rows, stream);
        }
    }
    Tensor projected = work.alloc(DType::BF16, {256, mask_columns});
    ops::linear(hidden, weights.candidate_selector.hidden_projection, projected, stream);
    Tensor drafts     = frame.draft_tokens.slice(1, 0, batch);
    Tensor proposal_q = frame.proposal_q.slice(2, 0, batch);
    ops::candidate_selector_path(candidates, scores.view({16, k, batch}),
                                 projected.view({256, k, batch}), frame.anchors.slice(0, 0, batch),
                                 weights.candidate_selector.predecessor_codebook,
                                 weights.candidate_selector.successor_codebook,
                                 frame.execution_frontiers.slice(0, 0, batch), frame.sampling,
                                 drafts, proposal_q, work, stream);
}

// DFlash2 草稿的 TP2 分片：**只分 MLP**。
//
// - attention 段（qkv 投影 / 滑窗 attention / attention-output 投影 + 卷积残差）在 rank 0 上
//   原样运行，与单卡路径逐行一致：草稿的滑窗 KV 因此只有 rank 0 一份，它的 append / 检查点 /
//   清零全部保持单卡语义。做成两卡各持一份会引入 append 路径的跨卡读（rank 1 读 rank 0 的
//   compact_features，需要事件排序且白费 P2P 带宽），实测第 2 轮起两卡 cache 分叉、草稿全废。
// - MLP 段的两次投影（gate_up 列并、down 行并）在两卡上分片执行，各承担一半权重流量；
//   allreduce 之后两卡拿到逐位相同的残差，rank 1 于是能用自算 hidden 参与半词表 top-16
//   （省掉 82KB 的 hidden 拷贝与一对事件）。
// - 每层在 MLP 段之前把 rank 0 的残差逐位拷给 rank 1：这是两卡 lockstep 唯一的跨卡数据流
//   （5120×W×B BF16，W=8 时 82KB），attention 段留在 rank 0 的代价就是这一份拷贝。
//
// 数值合规性：草稿只产提案、接受由 target 验证决定，因此草稿侧的浮点重排（gate_up/down 的
// 半孪生 partial 求和位置）只影响接受率、不影响输出内容。
template <class V>
void propose_dflash2_batch_tp2(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                               const TpExecution& tp, int batch, int k, DFlashEnvelopes envelopes) {
    using Config = typename V::DFlashConfig;
    // 权重视图经由 V 取（依赖类型），否则 35B 的 DFlashLayerWeights 会在解析期被这里的
    // DFlash2 专有成员访问（attention_conv / mlp_conv）打中。
    using DFlashWeights = typename V::ModelView::DFlash;
    static_assert(Config::layers == Config::local_layers,
                  "DFlash2 TP2 前传只实现了全 local 层的草稿");
    static_assert(Config::intermediate % 2 == 0, "DFlash2 TP2 需要偶数 intermediate");
    const ExecutionContext& ec           = *tp.execution;
    WorkspaceArena* work[2]              = {&state.execution.work, tp.work};
    DeviceContext* device[2]             = {&state.execution.device, tp.device};
    const LoadedModelData* runtime[2]    = {&state.execution.model, tp.weights};
    const DFlashWeights* weights[2]      = {&*state.execution.model.dflash,
                                            &*tp.weights->dflash};

    const int width        = k + 1;
    const int columns      = width * batch;
    const int mask_columns = k * batch;

    // ---- 草稿输入只有 rank 0 一份：frame 展开 + 嵌入，两卡各自的残差缓冲 ----
    work[0]->reset();
    work[1]->reset();
    Tensor anchors   = frame.anchors.slice(0, 0, batch);
    Tensor frontiers = frame.execution_frontiers.slice(0, 0, batch);
    Tensor valid     = frame.proposal_valid_columns.slice(0, 0, batch);
    Tensor ids       = frame.proposal_ids.slice(1, 0, batch);
    Tensor positions = frame.proposal_positions.slice(1, 0, batch);
    ops::prepare_masked_block(anchors, frontiers, valid, Config::mask_token, ids, positions,
                              device[0]->stream);
    Tensor residual0 = work[0]->alloc(DType::BF16, {Config::hidden, width, batch});
    Tensor residual1 = work[1]->alloc(DType::BF16, {Config::hidden, width, batch});
    Tensor flat_residual = residual0.view({Config::hidden, columns});
    ops::embedding(ids.view({columns}), runtime[0]->token_embedding, flat_residual,
                   device[0]->stream);
    // 第 0 层的 attention 段之前两卡必须从同一份残差出发：把嵌入结果也拷给 rank 1。
    CUDA_CHECK(cudaEventRecord(tp.events->inputs_ready(0), device[0]->stream));
    CUDA_CHECK(cudaSetDevice(device[1]->device));
    CUDA_CHECK(cudaStreamWaitEvent(device[1]->stream, tp.events->inputs_ready(0), 0));
    CUDA_CHECK(cudaMemcpyAsync(residual1.data, residual0.data, residual0.bytes(),
                               cudaMemcpyDeviceToDevice, device[1]->stream));
    CUDA_CHECK(cudaSetDevice(device[0]->device));

    for (std::size_t layer_index = 0; layer_index < static_cast<std::size_t>(Config::layers);
         ++layer_index) {
        nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                      layer_index);
        // ---- attention 段：rank 0 原样跑（与单卡路径同构），滑窗 KV 只此一份 ----
        {
            const auto& layer = weights[0]->layers[layer_index];
            auto scope  = work[0]->scope();
            auto branch = workspace_recipe::dflash2_branch<Config>(*work[0], width, batch);
            prepare_dynamic_branch(state.execution.device, *work[0], residual0, layer.input_norm,
                                   Config::rms_epsilon, layer.attention_conv, branch);
            Tensor query =
                work[0]->alloc(DType::BF16, {Config::head_dim, Config::query_heads, width, batch});
            Tensor key =
                work[0]->alloc(DType::BF16, {Config::head_dim, Config::kv_heads, width, batch});
            Tensor value =
                work[0]->alloc(DType::BF16, {Config::head_dim, Config::kv_heads, width, batch});
            Tensor query_flat = query.view({Config::query_size, columns});
            Tensor key_flat   = key.view({Config::kv_size, columns});
            Tensor value_flat = value.view({Config::kv_size, columns});
            ops::attn_input_proj(branch.prepared.view({Config::hidden, columns}),
                                 layer.query_key_value, query_flat, key_flat, value_flat,
                                 device[0]->stream);
            ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, query, key,
                              device[0]->stream);
            Tensor attention =
                work[0]->alloc(DType::BF16, {Config::head_dim, Config::query_heads, width, batch});
            ops::swa(query, key, value, positions, valid,
                     frame.state_destination_slots.slice(0, 0, batch), Config::attention_scale,
                     dflash_state(state).local_layer(static_cast<std::uint32_t>(layer_index)),
                     envelopes.local, *work[0], attention, device[0]->stream);
            finish_dynamic_branch(state.execution.device, *work[0],
                                  attention.view({Config::query_size, width, batch}),
                                  layer.attention_output, layer.attention_conv,
                                  branch.finish_delta, residual0);
        }
        // ---- 残差跨卡拷贝：MLP 段两卡从同一份残差出发 ----
        CUDA_CHECK(cudaEventRecord(tp.events->inputs_ready(0), device[0]->stream));
        CUDA_CHECK(cudaSetDevice(device[1]->device));
        CUDA_CHECK(cudaStreamWaitEvent(device[1]->stream, tp.events->inputs_ready(0), 0));
        CUDA_CHECK(cudaMemcpyAsync(residual1.data, residual0.data, residual0.bytes(),
                                   cudaMemcpyDeviceToDevice, device[1]->stream));
        CUDA_CHECK(cudaSetDevice(device[0]->device));
        // ---- MLP 段：gate_up 列并（每卡 [17408,5120] shard，gate'|up'）+ down 行并 ----
        {
            std::array<Tensor, 2> prepared;
            std::array<Tensor, 2> delta;
            std::array<Tensor, 2> gate_up;
            std::array<Tensor, 2> residual = {residual0, residual1};
            std::array<WorkspaceArena::Scope, 2> scopes = {work[0]->scope(), work[1]->scope()};
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto& layer = weights[r]->layers[layer_index];
                auto branch = workspace_recipe::dflash2_branch<Config>(*work[r], width, batch);
                prepare_dynamic_branch(*device[r], *work[r], residual[r], layer.post_attention_norm,
                                       Config::rms_epsilon, layer.mlp_conv, branch);
                prepared[r] = branch.prepared;
                delta[r]    = branch.finish_delta;
                gate_up[r]  = work[r]->alloc(DType::BF16, {Config::intermediate, columns});
            });
            ops::linear_column_parallel(
                {prepared[0].view({Config::hidden, columns}),
                 prepared[1].view({Config::hidden, columns})},
                {weights[0]->layers[layer_index].gate_up,
                 weights[1]->layers[layer_index].gate_up},
                gate_up, ec);
            // 每卡的 gate' 与 up' 都是自己那半 shard 的两段：列切把 gate_up 切成两个独立块，
            // rank r 持有 gate 的第 r 个半块与 up 的第 r 个半块，SiLU 配对是卡内的，无需通信。
            std::array<Tensor, 2> activation;
            constexpr int kShardIntermediate = Config::intermediate / 2;
            for_each_rank(ec, [&](int rank) {
                const auto r = static_cast<std::size_t>(rank);
                activation[r] = work[r]->alloc(DType::BF16, {kShardIntermediate, columns});
                ops::silu_mul(gate_up[r].slice(0, 0, kShardIntermediate),
                              gate_up[r].slice(0, kShardIntermediate, kShardIntermediate),
                              activation[r], device[rank]->stream);
            });
            std::array<Tensor, 2> projected;
            std::array<Tensor, 2> staging;
            for (std::size_t r = 0; r < 2; ++r) {
                projected[r] = work[r]->alloc(DType::BF16, {Config::hidden, columns});
                staging[r]   = work[r]->alloc(DType::BF16, {Config::hidden, columns});
            }
            ops::linear_row_parallel(activation,
                                     {weights[0]->layers[layer_index].down,
                                      weights[1]->layers[layer_index].down},
                                     projected, staging, ec, *tp.events);
            // allreduce 之后 projected 两卡逐位相同，卷积+残差 tail 在两卡各自复制执行，
            // 因此下一层的残差仍然逐位一致。
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto& layer = weights[r]->layers[layer_index];
                ops::dynamic_grouped_conv_add_tail(projected[r], layer.mlp_conv.base_kernel,
                                                   delta[r], residual[r], device[rank]->stream);
            });
        }
    }
    std::array<Tensor, 2> hidden;
    for_each_rank(ec, [&](int rank) {
        const auto r = static_cast<std::size_t>(rank);
        hidden[r]    = work[r]->alloc(DType::BF16, {Config::hidden, mask_columns});
        ops::rmsnorm_pack_tail(rank == 0 ? residual0 : residual1, weights[r]->final_norm,
                               hidden[r], device[rank]->stream);
    });
    dflash2_select_candidates<V>(state, frame, hidden[0], &hidden[1], batch, k);
    work[0]->reset();
    work[1]->reset();
}

template <class V>
void propose_dflash2_batch(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame, int batch,
                           int k, DFlashEnvelopes envelopes) {
    if constexpr (V::DFlashConfig::coherent_selector) {
        // TP2：草稿权重按头/块分片，两卡各跑自己那半（见 propose_dflash2_batch_tp2）。
        // 单卡（tp == 1）走下面的原路径，一行未改。
        if (const std::optional<TpExecution> tp = tp_execution(state.execution)) {
            propose_dflash2_batch_tp2<V>(state, frame, *tp, batch, k, envelopes);
            return;
        }
        using Config                                 = typename V::DFlashConfig;
        const int width                              = k + 1;
        const int columns                            = width * batch;
        const int mask_columns                       = k * batch;
        auto& work                                   = state.execution.work;
        const cudaStream_t stream                    = state.execution.device.stream;
        const typename V::ModelView::DFlash& weights = *state.execution.model.dflash;
        Tensor anchors                               = frame.anchors.slice(0, 0, batch);
        Tensor frontiers                             = frame.execution_frontiers.slice(0, 0, batch);
        Tensor valid_columns = frame.proposal_valid_columns.slice(0, 0, batch);
        Tensor state_slots   = frame.state_destination_slots.slice(0, 0, batch);
        Tensor ids           = frame.proposal_ids.slice(1, 0, batch);
        Tensor positions     = frame.proposal_positions.slice(1, 0, batch);
        work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, stream);
        Tensor residual      = work.alloc(DType::BF16, {Config::hidden, width, batch});
        Tensor flat_residual = residual.view({Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, flat_residual,
                       stream);
        for (std::size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
            const auto& layer = weights.layers[layer_index];
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          layer_index);
            {
                auto scope  = work.scope();
                auto branch = workspace_recipe::dflash2_branch<Config>(work, width, batch);
                prepare_dynamic_branch(state.execution.device, work, residual, layer.input_norm,
                                       Config::rms_epsilon, layer.attention_conv, branch);
                Tensor query =
                    work.alloc(DType::BF16, {Config::head_dim, Config::query_heads, width, batch});
                Tensor key =
                    work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, width, batch});
                Tensor value =
                    work.alloc(DType::BF16, {Config::head_dim, Config::kv_heads, width, batch});
                Tensor query_flat = query.view({Config::query_size, columns});
                Tensor key_flat   = key.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::attn_input_proj(branch.prepared.view({Config::hidden, columns}),
                                     layer.query_key_value, query_flat, key_flat, value_flat,
                                     stream);
                ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, query, key, stream);
                Tensor attention =
                    work.alloc(DType::BF16, {Config::head_dim, Config::query_heads, width, batch});
                ops::swa(query, key, value, positions, valid_columns, state_slots,
                         Config::attention_scale,
                         state.dflash.local_layer(static_cast<std::uint32_t>(layer_index)),
                         envelopes.local, work, attention, stream);
                finish_dynamic_branch(
                    state.execution.device, work,
                    attention.view({Config::query_size, width, batch}), layer.attention_output,
                    layer.attention_conv, branch.finish_delta, residual);
            }
            {
                auto scope  = work.scope();
                auto branch = workspace_recipe::dflash2_branch<Config>(work, width, batch);
                prepare_dynamic_branch(state.execution.device, work, residual,
                                       layer.post_attention_norm, Config::rms_epsilon,
                                       layer.mlp_conv, branch);
                Tensor intermediate = work.alloc(DType::BF16, {Config::intermediate, width, batch});
                Tensor intermediate_flat = intermediate.view({Config::intermediate, columns});
                ops::linear_swiglu(branch.prepared.view({Config::hidden, columns}), layer.gate_up,
                                   intermediate_flat, work, stream);
                finish_dynamic_branch(state.execution.device, work, intermediate, layer.down,
                                      layer.mlp_conv, branch.finish_delta, residual);
            }
        }
        Tensor hidden = work.alloc(DType::BF16, {Config::hidden, mask_columns});
        ops::rmsnorm_pack_tail(residual, weights.final_norm, hidden, stream);
        dflash2_select_candidates<V>(state, frame, hidden, nullptr, batch, k);
        work.reset();
    }
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else if constexpr (V::DFlashConfig::coherent_selector) {
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(k + 1U) * batch_size);
        propose_dflash2_batch<V>(state, frame, batch_size, k, envelopes);
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        nvtx::ScopedRange proposal_range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                                         static_cast<std::uint64_t>(columns));
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor full_rows          = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions          = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                          static_cast<std::uint64_t>(layer));
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            {
                nvtx::ScopedRange attention_range(nvtx::Name::DFlashAttention,
                                                  nvtx::Category::Attention,
                                                  static_cast<std::uint64_t>(layer));
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value   = roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::attn_input_proj(roots.hidden, weight.query_key_value, query_flat, key_flat,
                                     value_flat, state.execution.device.stream);
                Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.stream);
                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                if (layer < Config::local_layers) {
                    ops::swa(query_batch, key_batch, value_batch, positions, valid_columns,
                             state_destinations, Config::attention_scale,
                             dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                             envelopes.local, state.execution.work, attention_batch,
                             state.execution.device.stream);
                } else {
                    ops::gqa_attention(query_batch, key_batch, value_batch,
                                       positions.view({width, batch_size}), valid_columns, full_rows,
                                       Config::attention_scale,
                                       dflash_state(state).full_batch_layer(0), envelopes.full,
                                       state.execution.work, attention_batch,
                                       state.execution.device.stream);
                }
                ops::linear_add(roots.attention.view({Config::query_size, columns}),
                                weight.attention_output, residual, state.execution.work,
                                state.execution.device.stream);
            }
            {
                nvtx::ScopedRange mlp_range(nvtx::Name::DFlashMlp, nvtx::Category::PostMixer,
                                            static_cast<std::uint64_t>(layer));
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                ops::linear_swiglu(roots.hidden, weight.gate_up, roots.intermediate,
                                   state.execution.work, state.execution.device.stream);
                ops::linear_add(roots.intermediate, weight.down, residual, state.execution.work,
                                state.execution.device.stream);
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.model.dflash->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
            ops::argmax(logits, flat_drafts, TextConfig::token_domain,
                        state.execution.device.stream);
        } else {
            if (!state.execution.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.model.optimized_proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {V::draft_head_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, proposal.head, logits, state.execution.device.stream);
            ops::argmax(logits, flat_drafts, V::draft_head_rows, state.execution.device.stream);
            ops::proposal_remap_token_ids(flat_drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          V::draft_head_rows, state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));


        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts     = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents            = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows        = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor active_lanes       = frame.active_lanes.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor append_positions   = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts      = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.verify_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, active_lanes,
                                   context_starts, frontiers, compact_features, append_positions,
                                   append_counts, state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     state_destinations, dflash_rows, envelopes.append);

        propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
        ops::speculative_prepare_verify_inputs(anchors, drafts, frontiers, extents, verify_ids,
                                               target_positions, state.execution.device.stream);

        // ---- DFlash2 tp2：草稿候选由 rank 0 的选择器产出（两卡隐藏态逐位相同，选择器只在
        // rank 0 跑一次），把 draft_tokens 拷给 rank 1，两卡各自验证自己那半 target ----
        std::optional<TpExecution> tp = tp_execution(state.execution);
        if (tp) {
            if (!tp->io->dflash_decode.has_value()) {
                throw std::logic_error("tensor-parallel DFlash2 decode requires a peer frame");
            }
            qwen3_6::DFlashDecodeState& peer_frame = *tp->io->dflash_decode;
            // 草稿是 rank 0 的选择器写在 rank 0 显存上的：必须先把 rank 0 流上到此为止的工作
            // （选择器写 frame.draft_tokens）与本卡的 D2D 拷贝/验证排好序，否则 rank 1 的拷贝可能
            // 读到上一轮的草稿 —— 实测症状是两卡 verify_ids 的草稿完全不同（`[vid]` 同 frontier
            // 两组不同草稿），rank 1 按另一批草稿裁决 ⇒ 两卡 KV 逐轮发散 ⇒ 下一轮 p̃ 来自错状态
            // ⇒ 采样输出丢 token（HTML 缺 `-scale` / 缺 `=`）。事件用法与 dflash2_select_candidates
            // 的 peer 路径一致（先 record 再 wait，复用 inputs_ready 这一对事件）。
            CUDA_CHECK(cudaEventRecord(tp->events->inputs_ready(0), state.execution.device.stream));
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            CUDA_CHECK(cudaStreamWaitEvent(tp->device->stream, tp->events->inputs_ready(0), 0));
            // rank 1 用同一份 ingress 记录（DFlash2 的验证默认 greedy，不读 token_counts 指针）
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            CUDA_CHECK(cudaMemcpyAsync(peer_frame.ingress.data, &state.host_ingress,
                                       sizeof(qwen3_6::DFlashDecodeIngress),
                                       cudaMemcpyHostToDevice, tp->device->stream));
            // 草稿是 rank 0 那套模型算出来的（两卡位置/嵌入相同），所以按构造就是同一批提案
            CUDA_CHECK(cudaMemcpyAsync(peer_frame.draft_tokens.data, frame.draft_tokens.data,
                                       frame.draft_tokens.bytes(), cudaMemcpyDeviceToDevice,
                                       tp->device->stream));
            Tensor p_anchors   = peer_frame.anchors.slice(0, 0, batch_size);
            Tensor p_frontiers = peer_frame.execution_frontiers.slice(0, 0, batch_size);
            Tensor p_extents   = peer_frame.proposal_extents.slice(0, 0, batch_size);
            Tensor p_drafts    = peer_frame.draft_tokens.slice(1, 0, batch_size);
            Tensor p_verify    = peer_frame.verify_ids.slice(1, 0, batch_size);
            Tensor p_positions = peer_frame.verify_positions.slice(1, 0, batch_size);
            ops::speculative_prepare_verify_inputs(p_anchors, p_drafts, p_frontiers, p_extents,
                                                   p_verify, p_positions, tp->device->stream);
            CUDA_CHECK(cudaSetDevice(state.execution.device.device));
        }

        // 本 fork 的 TextContext 多两个参数：per-rank 的 rope 频率表与 tp 执行上下文。
        // tp2 时**必须**把 tp 传进来：目标验证要跑两卡版本（rank 1 有自己的 KV/GDN 状态），
        // 否则 tp2() 为假、验证路径会退回单卡并抛异常。
        TextContext card(state.execution.device, state.execution.model, state.execution.work,
                         state.execution.rope_frequency, {}, state.execution.linear_attention,
                         state.execution.io, state.execution.prefill_hidden,
                         state.execution.prefill_chunk, 0, {}, &state.text_cache, nullptr,
                         tp ? &*tp : nullptr);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, active_lanes, valid_columns, width, batch_size);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeDFlashTarget, nvtx::Category::DFlash,
                                           static_cast<std::uint64_t>(width) * batch_size);
            TargetVerifyFrameView rank0_frame{
                    .ids                     = verify_ids,
                    .cache_positions         = target_positions,
                    .rope_positions          = target_rope,
                    .valid_columns           = valid_columns,
                    .kv_table_rows           = text_rows,
                    .state_source_slots      = state_sources,
                    .state_destination_slots = state_destinations,
                    .lanes                   = active_lanes,
                    .target_hidden           = target_hidden,
                    .target_logits           = target_logits,
                    .target_tokens           = target_tokens,
                    .drafts                  = drafts,
                    .current_extents         = extents,
                    .candidate_ids           = frame.candidate_ids.data
                                                   ? frame.candidate_ids.slice(2, 0, batch_size)
                                                   : Tensor{},
                    .proposal_q =
                        frame.proposal_q.data ? frame.proposal_q.slice(2, 0, batch_size) : Tensor{},
                    .frontiers       = frontiers,
                    .anchors         = anchors,
                    .licensed_tokens = licensed_tokens,
                    .licensed_counts = licensed_counts,
                    .accepted_drafts = accepted,
                    .selected_hidden = selected_hidden,
                    .replay_records  = state.execution.replay_records,
                    .sampling        = frame.sampling,
                    .feature_sink    = &sink,
            };
            if (tp) {
                qwen3_6::DFlashDecodeState& pf = *tp->io->dflash_decode;
                TargetVerifyFrameView peer_frame_view{
                    .ids             = pf.verify_ids.slice(1, 0, batch_size),
                    .cache_positions = pf.verify_positions.slice(1, 0, batch_size),
                    .rope_positions  = pf.target_rope_positions.slice(1, 0, batch_size),
                    .valid_columns   = pf.target_valid_columns.slice(0, 0, batch_size),
                    .kv_table_rows   = pf.text_kv_table_rows.slice(0, 0, batch_size),
                    .state_source_slots = pf.state_source_slots.slice(0, 0, batch_size),
                    .state_destination_slots = pf.state_destination_slots.slice(0, 0, batch_size),
                    .lanes           = pf.active_lanes.slice(0, 0, batch_size),
                    .target_hidden   = pf.target_hidden.slice(2, 0, batch_size),
                    .target_logits   = pf.target_logits.slice(2, 0, batch_size),
                    .target_tokens   = pf.target_argmax.slice(1, 0, batch_size),
                    .drafts          = pf.draft_tokens.slice(1, 0, batch_size),
                    .current_extents = pf.proposal_extents.slice(0, 0, batch_size),
                    .candidate_ids   = pf.candidate_ids.data
                                                   ? pf.candidate_ids.slice(2, 0, batch_size)
                                                   : Tensor{},
                    .proposal_q =
                        pf.proposal_q.data ? pf.proposal_q.slice(2, 0, batch_size) : Tensor{},
                    .frontiers       = pf.execution_frontiers.slice(0, 0, batch_size),
                    .anchors         = pf.anchors.slice(0, 0, batch_size),
                    .licensed_tokens = pf.licensed_tokens.slice(1, 0, batch_size),
                    .licensed_counts = pf.licensed_counts.slice(0, 0, batch_size),
                    .accepted_drafts = pf.accepted_drafts.slice(0, 0, batch_size),
                    .selected_hidden = pf.target_continuation_hidden.slice(1, 0, batch_size),
                    .replay_records  = tp->replay_records,
                    .sampling        = pf.sampling,
                    .feature_sink    = nullptr,
                };
                target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                     rank0_frame, peer_frame_view, target_envelope);
            } else {
                target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                     rank0_frame, target_envelope);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
