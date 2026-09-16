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

        // TP2：草稿按头分片，每卡用自己的 context_key/value shard 与自己的 local cache 物化
        // 本卡那半 KV。feature_projection/context_norm 是复制权重，两卡各算一份，输入相同 ⇒
        // 结果逐位相同；context_kv_materialize 只读 context 与本卡权重、只写本卡 cache。
        if (const std::optional<TpExecution> tp = tp_execution(state.execution)) {
            if constexpr (Config::coherent_selector) {
                if (!tp->dflash) {
                    throw std::logic_error(
                        "tensor-parallel DFlash2 append requires a peer draft state");
                }
                const Tensor local_positions = replace_local_window
                                                   ? positions.slice(0, local_offset, local_width)
                                                   : positions;
                std::array<WorkspaceArena*, 2> works = {&state.execution.work, tp->work};
                std::array<const LoadedModelData*, 2> models = {&state.execution.model,
                                                                tp->weights};
                DFlashPersistentState* states[2] = {&dflash_state(state), tp->dflash};
                std::array<Tensor, 2> projected2;
                std::array<Tensor, 2> context2;
                std::array<WorkspaceArena::Scope, 2> scopes = {works[0]->scope(),
                                                               works[1]->scope()};
                for_each_rank(*tp->execution, [&](int rank) {
                    const auto r      = static_cast<std::size_t>(rank);
                    cudaStream_t s    = tp->execution->dev[rank]->stream;
                    const auto roots  = workspace_recipe::dflash_context<Config>(
                        *works[r], projected_width * batch);
                    projected2[r] = roots.projected;
                    context2[r]   = roots.normalized;
                    ops::linear(input.view({Config::feature_rows, projected_width * batch}),
                                models[r]->dflash->feature_projection, projected2[r], s);
                    ops::rmsnorm(projected2[r], models[r]->dflash->context_norm,
                                 Config::rms_epsilon, false, context2[r], s);
                });
                for_each_rank(*tp->execution, [&](int rank) {
                    const auto r    = static_cast<std::size_t>(rank);
                    cudaStream_t s  = tp->execution->dev[rank]->stream;
                    std::array<ops::ContextKVMaterializeLayerView, Config::layers> layers;
                    for (int layer = 0; layer < Config::layers; ++layer) {
                        const auto& weights = models[r]->dflash->layers[layer];
                        layers[layer]       = {weights.context_key, weights.context_value,
                                               weights.key_norm,
                                               states[r]->local_layer(layer)};
                    }
                    ops::context_kv_materialize(
                        context2[r].view({Config::hidden, local_width, batch}), local_positions,
                        local_counts, lanes, layers,
                        {local_envelope.min_count, local_envelope.max_count}, *works[r], s);
                });
                return;
            }
        }

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
        ops::linear_topk(hidden, head.head, head.token_ids, ids_flat, scores, work, stream);
    }
    if (tp && state.execution.proposal_head == ProposalHead::Full) {
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

// DFlash2 草稿的 TP2 前传：权重按头/块分片（bindings.cpp 的 dflash2 规则），两卡各跑自己那半，
// 每层两次 allreduce（attention/output 与 mlp/down 的行并行投影），卷积+残差 tail 在两卡各自
// 复制执行。
//
// 数值合规性：草稿只产提案，接受由 target 验证决定，因此草稿侧的浮点重排/分片只影响接受率、
// 不影响输出内容。两卡 residual 经 allreduce 后逐位相同（IEEE 加法交换律），其后所有复制计算
// 保持逐位一致，草稿候选在两卡上按构造相同。
template <class V>
void propose_dflash2_batch_tp2(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                               const TpExecution& tp, int batch, int k, DFlashEnvelopes envelopes) {
    using Config = typename V::DFlashConfig;
    // 权重视图经由 V 取（依赖类型），否则 35B 的 DFlashLayerWeights 会在解析期被这里的
    // DFlash2 专有成员访问（attention_conv / mlp_conv）打中。
    using DFlashWeights = typename V::ModelView::DFlash;
    static_assert(Config::layers == Config::local_layers,
                  "DFlash2 TP2 前传只实现了全 local 层的草稿");
    if (!tp.dflash) {
        throw std::logic_error("tensor-parallel DFlash2 proposal requires a peer draft state");
    }
    const ExecutionContext& ec      = *tp.execution;
    WorkspaceArena* work[2]         = {&state.execution.work, tp.work};
    DeviceContext* device[2]        = {&state.execution.device, tp.device};
    const LoadedModelData* runtime[2]     = {&state.execution.model, tp.weights};
    qwen3_6::DFlashDecodeState* frames[2] = {&frame, &*tp.io->dflash_decode};
    DFlashPersistentState* draft_state[2] = {&dflash_state(state), tp.dflash};
    const DFlashWeights* weights[2]       = {&*state.execution.model.dflash,
                                             &*tp.weights->dflash};

    const int width        = k + 1;
    const int columns      = width * batch;
    const int mask_columns = k * batch;
    // attention/output 的行并行收缩维是草稿的 query_size(4096)，每卡一半。
    constexpr int kQRows = Config::query_size / 2;  // 2048
    static_assert(Config::query_size % 2 == 0 && Config::intermediate % 2 == 0,
                  "DFlash2 TP2 需要偶数收缩维");

    // ---- 每卡自己的 frame 展开 + 嵌入（同一份 ingress 各卡独立展开，结果逐位相同）----
    std::array<Tensor, 2> residual;
    for_each_rank(ec, [&](int rank) {
        const auto r    = static_cast<std::size_t>(rank);
        cudaStream_t s  = device[rank]->stream;
        auto& f         = *frames[r];
        Tensor anchors  = f.anchors.slice(0, 0, batch);
        Tensor frontiers = f.execution_frontiers.slice(0, 0, batch);
        Tensor valid    = f.proposal_valid_columns.slice(0, 0, batch);
        Tensor ids      = f.proposal_ids.slice(1, 0, batch);
        Tensor positions = f.proposal_positions.slice(1, 0, batch);
        work[r]->reset();
        ops::prepare_masked_block(anchors, frontiers, valid, Config::mask_token, ids, positions,
                                  s);
        residual[r] = work[r]->alloc(DType::BF16, {Config::hidden, width, batch});
        Tensor flat_residual = residual[r].view({Config::hidden, columns});
        ops::embedding(ids.view({columns}), runtime[rank]->token_embedding, flat_residual, s);
    });

    for (std::size_t layer_index = 0; layer_index < Config::layers; ++layer_index) {
        nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash,
                                      layer_index);
        // ---- attention 分支：qkv 与滑窗 attention 在两卡上**复制**运行（swa 的 kernel 把
        // 32Q/8KV 头写死），只有 attention/output 的行并行投影分片。复制的 attention 两卡逐位
        // 相同，行并的两半因此取自同一份张量。 ----
        {
            std::array<Tensor, 2> prepared;
            std::array<Tensor, 2> delta;
            std::array<Tensor, 2> attention;
            std::array<WorkspaceArena::Scope, 2> scopes = {work[0]->scope(), work[1]->scope()};
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                cudaStream_t s    = device[rank]->stream;
                auto& f           = *frames[r];
                const auto& layer = weights[r]->layers[layer_index];
                auto branch = workspace_recipe::dflash2_branch<Config>(*work[r], width, batch);
                prepare_dynamic_branch(*device[r], *work[r], residual[r], layer.input_norm,
                                       Config::rms_epsilon, layer.attention_conv, branch);
                prepared[r] = branch.prepared;
                delta[r]    = branch.finish_delta;
                Tensor query = work[r]->alloc(
                    DType::BF16, {Config::head_dim, Config::query_heads, width, batch});
                Tensor key = work[r]->alloc(
                    DType::BF16, {Config::head_dim, Config::kv_heads, width, batch});
                Tensor value = work[r]->alloc(
                    DType::BF16, {Config::head_dim, Config::kv_heads, width, batch});
                Tensor query_flat = query.view({Config::query_size, columns});
                Tensor key_flat   = key.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                Tensor positions  = f.proposal_positions.slice(1, 0, batch);
                ops::attn_input_proj(branch.prepared.view({Config::hidden, columns}),
                                     layer.query_key_value, query_flat, key_flat, value_flat, s);
                ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, query, key, s);
                attention[r] = work[r]->alloc(
                    DType::BF16, {Config::head_dim, Config::query_heads, width, batch});
                ops::swa(query, key, value, positions,
                         f.proposal_valid_columns.slice(0, 0, batch),
                         f.state_destination_slots.slice(0, 0, batch), Config::attention_scale,
                         draft_state[r]->local_layer(static_cast<std::uint32_t>(layer_index)),
                         envelopes.local, *work[r], attention[r], s);
            });
            std::array<Tensor, 2> projected;
            std::array<Tensor, 2> staging;
            for (std::size_t r = 0; r < 2; ++r) {
                projected[r] = work[r]->alloc(DType::BF16, {Config::hidden, columns});
                staging[r]   = work[r]->alloc(DType::BF16, {Config::hidden, columns});
            }
            // 行并行：rank r 取 attention 输出（两卡逐位相同，含全部 32 头）的第 r 个 2048 行块，
            // 各自算 [5120,2048] 权重 shard 的部分积，一次 allreduce 得到完整投影。
            Tensor context0 = attention[0].view({Config::query_size, columns});
            Tensor context1 = attention[1].view({Config::query_size, columns});
            ops::linear_row_parallel({context0.slice(0, 0, kQRows),
                                      context1.slice(0, kQRows, kQRows)},
                                     {weights[0]->layers[layer_index].attention_output,
                                      weights[1]->layers[layer_index].attention_output},
                                     projected, staging, ec, *tp.events);
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto& layer = weights[r]->layers[layer_index];
                ops::dynamic_grouped_conv_add_tail(projected[r], layer.attention_conv.base_kernel,
                                                   delta[r], residual[r], device[rank]->stream);
            });
        }
        // ---- MLP 分支：gate_up 列并行（每卡 [17408,5120] shard，gate'|up'），down 行并行 ----
        {
            std::array<Tensor, 2> prepared;
            std::array<Tensor, 2> delta;
            std::array<Tensor, 2> gate_up;
            std::array<WorkspaceArena::Scope, 2> scopes = {work[0]->scope(), work[1]->scope()};
            for_each_rank(ec, [&](int rank) {
                const auto r      = static_cast<std::size_t>(rank);
                const auto& layer = weights[r]->layers[layer_index];
                auto branch = workspace_recipe::dflash2_branch<Config>(*work[r], width, batch);
                prepare_dynamic_branch(*device[r], *work[r], residual[r],
                                       layer.post_attention_norm, Config::rms_epsilon,
                                       layer.mlp_conv, branch);
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
                activation[r] =
                    work[r]->alloc(DType::BF16, {kShardIntermediate, columns});
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
            ops::linear_row_parallel(
                activation,
                {weights[0]->layers[layer_index].down, weights[1]->layers[layer_index].down},
                projected, staging, ec, *tp.events);
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
        ops::rmsnorm_pack_tail(residual[r], weights[r]->final_norm, hidden[r],
                               device[rank]->stream);
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

        // TP2：草稿在两张卡上各自展开 frame 并前传（见 propose_dflash2_batch_tp2），因此 rank 1
        // 必须在提案**之前**拿到同一份 ingress —— 两个 frame 的全部输入张量都由它展开。
        std::optional<TpExecution> tp = tp_execution(state.execution);
        if (tp) {
            if (!tp->io->dflash_decode.has_value()) {
                throw std::logic_error("tensor-parallel DFlash2 decode requires a peer frame");
            }
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            CUDA_CHECK(cudaMemcpyAsync(tp->io->dflash_decode->ingress.data, &state.host_ingress,
                                       sizeof(qwen3_6::DFlashDecodeIngress),
                                       cudaMemcpyHostToDevice, tp->device->stream));
            CUDA_CHECK(cudaSetDevice(state.execution.device.device));
        }

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
        if (tp) {
            qwen3_6::DFlashDecodeState& peer_frame = *tp->io->dflash_decode;
            CUDA_CHECK(cudaSetDevice(tp->device->device));
            // 草稿是两卡各自那半模型算出来的（TP2）或 rank 0 算出来的（tp1）；两卡位置/嵌入
            // 相同，所以按构造就是同一批提案
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
