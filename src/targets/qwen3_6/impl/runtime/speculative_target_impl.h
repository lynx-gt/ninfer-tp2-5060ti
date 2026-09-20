#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens);
    }
    if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            TextConfig::token_domain, frame.sampling, {false}, execution.work,
            execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts, TextConfig::token_domain, frame.sampling, execution.work,
            execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          TargetVerifyFrameView peer, ops::GqaExecutionEnvelope envelope) {
    if (execution.peer == nullptr) {
        throw std::logic_error("tensor-parallel target verify requires a peer");
    }
    if (frame.replay_records == nullptr || peer.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    if (peer.feature_sink != nullptr && frame.feature_sink == nullptr) {
        throw std::logic_error("tensor-parallel target verify found a feature sink on rank 1 only");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    // DFlash2 的 feature sink 只在 rank 0 上存在：残差在两卡上逐位相同，而草稿（唯一消费者）
    // 只在 rank 0 上跑，rank 1 只负责最终那半张词表的 top-k。
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch({frame.ids, peer.ids},
                                 {frame.cache_positions, peer.cache_positions},
                                 {frame.rope_positions, peer.rope_positions},
                                 {frame.valid_columns, peer.valid_columns},
                                 {frame.kv_table_rows, peer.kv_table_rows},
                                 {frame.lanes, peer.lanes}, envelope,
                                 {frame.target_hidden, peer.target_hidden},
                                 {frame.target_logits, peer.target_logits},
                                 {frame.target_tokens, peer.target_tokens}, *frame.feature_sink);
    } else {
        card.target_verify_batch({frame.ids, peer.ids},
                                 {frame.cache_positions, peer.cache_positions},
                                 {frame.rope_positions, peer.rope_positions},
                                 {frame.valid_columns, peer.valid_columns},
                                 {frame.kv_table_rows, peer.kv_table_rows},
                                 {frame.lanes, peer.lanes}, envelope,
                                 {frame.target_hidden, peer.target_hidden},
                                 {frame.target_logits, peer.target_logits},
                                 {frame.target_tokens, peer.target_tokens});
    }
    // [FIX-SPARSE-TP2] DFlash2 的 candidate_ids / proposal_q 只由 rank 0 的 selector 写出（草稿全程只在 rank 0 跑），
    // 但两卡都要各自跑一次接受判定来推进自己的 state 与 anchor。必须先把这两张小表按 p2p 补给 peer：
    // 否则 rank 1 读到全 0 的表 ⇒ qd=0 ⇒ `pd >= qd` 恒真 ⇒ 草稿被全部接受 ⇒ 两卡的 KV/位置随轮次
    // 发散，而 tp2 注意力要两卡 KV 一起算 ⇒ 后续 logits 被污染、提交的 token 出错。
    // 实测症状：只有"dflash2 + 采样"这一格会出现缺 `=`/重复词；贪心不受影响，因为贪心核不读 q。
    if (frame.candidate_ids.data != nullptr && frame.proposal_q.data != nullptr &&
        peer.candidate_ids.data != nullptr && peer.proposal_q.data != nullptr &&
        frame.drafts.data != nullptr && peer.drafts.data != nullptr) {
        const TpPeerCore& peer_core = *execution.peer;
        CUDA_CHECK(cudaEventRecord(peer_core.events->inputs_ready(0), execution.device.stream));
        CUDA_CHECK(cudaSetDevice(peer_core.device->device));
        CUDA_CHECK(
            cudaStreamWaitEvent(peer_core.device->stream, peer_core.events->inputs_ready(0), 0));
        CUDA_CHECK(cudaMemcpyAsync(peer.candidate_ids.data, frame.candidate_ids.data,
                                   peer.candidate_ids.bytes(), cudaMemcpyDeviceToDevice,
                                   peer_core.device->stream));
        CUDA_CHECK(cudaMemcpyAsync(peer.proposal_q.data, frame.proposal_q.data,
                                   peer.proposal_q.bytes(), cudaMemcpyDeviceToDevice,
                                   peer_core.device->stream));
        // `drafts` / `current_extents` 同样是 rank 1 判定所需：实测 rank 1 的 drafts 会**落后一整轮**
        // （草稿只在 rank 0 产出），两卡因此对同一轮给出不同裁决，KV 回滚量不同 ⇒ 仍会发散。
        CUDA_CHECK(cudaMemcpyAsync(peer.drafts.data, frame.drafts.data, peer.drafts.bytes(),
                                   cudaMemcpyDeviceToDevice, peer_core.device->stream));
        CUDA_CHECK(cudaMemcpyAsync(peer.current_extents.data, frame.current_extents.data,
                                   peer.current_extents.bytes(), cudaMemcpyDeviceToDevice,
                                   peer_core.device->stream));
        CUDA_CHECK(cudaEventRecord(peer_core.events->inputs_ready(1), peer_core.device->stream));
        CUDA_CHECK(cudaSetDevice(execution.device.device));
        CUDA_CHECK(
            cudaStreamWaitEvent(execution.device.stream, peer_core.events->inputs_ready(1), 0));
    }

    const ExecutionContext& ec      = *execution.peer->execution;
    WorkspaceArena* work[2]         = {&execution.work, execution.peer->work};
    TargetVerifyFrameView* views[2] = {&frame, &peer};
    for_each_rank(ec, [&](int rank) {
        const auto r             = static_cast<std::size_t>(rank);
        TargetVerifyFrameView& v = *views[r];
        cudaStream_t stream      = ec.dev[rank]->stream;
        if (v.proposal_q.data != nullptr) {
            ops::speculative_accept_sparse_drafts(
                v.target_tokens, v.target_logits, v.drafts, v.candidate_ids, v.proposal_q,
                v.current_extents, v.frontiers, v.anchors, v.licensed_tokens, v.licensed_counts,
                v.accepted_drafts, TextConfig::token_domain, v.sampling, {false}, *work[r], stream);
        } else {
            ops::speculative_accept_greedy_drafts(v.target_tokens, v.target_logits, v.drafts,
                                                  v.current_extents, v.frontiers, v.anchors,
                                                  v.licensed_tokens, v.licensed_counts,
                                                  v.accepted_drafts, TextConfig::token_domain,
                                                  v.sampling, *work[r], stream);
        }
        ops::speculative_select_accepted_hidden(v.target_hidden, v.accepted_drafts,
                                                v.selected_hidden, stream);
    });
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
