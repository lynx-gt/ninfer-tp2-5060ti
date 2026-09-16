#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Keep the candidate domain separate from the measured production crossover so the two kernel
// families remain directly comparable in the benchmark overlap.
inline constexpr std::int32_t kBf16SmallTMinTokens         = 2;
inline constexpr std::int32_t kBf16SmallTMaxTokens         = 32;
inline constexpr std::int32_t kBf16LinearSmallTDispatchEnd = 27;

using Bf16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_bf16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
// DFlash2 candidate-selector rank projection（n=256, k=5120）：master 后期新增的专用 kernel，
// TP2 基底早于它，搬运时只带了 kernel 文件没带分派，所以这里补回入口。
void launch_bf16_n256_k5120(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
