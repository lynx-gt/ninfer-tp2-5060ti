#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * @brief 合并两路各自有序的 top-16 候选，输出全局 top-16。
 *
 * @details 用于 TP2 下被行分片的词表头：每张卡只持有半张头，各自在自己的词表半区上取
 * top-16（`linear_topk`），再在本 Op 里合并成全局候选。排序与 `linear_topk` 一致：分数降序，
 * 分数精确相同时取**更小的全局 token id**。
 *
 * Tensor 合同：`ids_a`/`ids_b` 是连续 I32 `[16,U]`，`scores_a`/`scores_b` 是连续 FP32 `[16,U]`，
 * `ids_out`/`scores_out` 形如 `[16,U]`。每一列独立合并；`*_b` 允许与 `*_a` 重叠（同源退化情形）。
 * 所有输出被完整覆盖，Op 无持久状态、不做内部设备分配。
 *
 * `remote_id_offset` 加到 `ids_b` 上再参与比较与输出：`linear_topk` 返回的是**本卡头内的行号**
 * （整词表头时行号==全局 token id），所以被行分片的第二张头的行号必须加上它那一片的全局行起点
 * 才是全局 token id —— 漏掉就是半个词表的候选全部指错 token。传 0 即无偏移。
 */
void topk_pair_merge(const Tensor& ids_a, const Tensor& scores_a, const Tensor& ids_b,
                     const Tensor& scores_b, Tensor& ids_out, Tensor& scores_out,
                     std::int32_t remote_id_offset, cudaStream_t stream);

} // namespace ninfer::ops
