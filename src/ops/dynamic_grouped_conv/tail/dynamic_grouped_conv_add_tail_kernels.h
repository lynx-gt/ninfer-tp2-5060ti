#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// 融合 op `linear_dynamic_grouped_conv_add` 的卷积+残差半边，独立成 op 供 TP2 草稿路径使用：
//
//   residual[h,i,b] += (base_kernel[h,0,1] + finish_delta[g,0,i,b]) * z[h,i,b]
//                    + I(i>0) * (base_kernel[h,1,1] + finish_delta[g,1,i,b]) * z[h,i-1,b]
//
// 其中 z（这里的 `projected`）是投影半边在融合实现里本来就要物化成 BF16 的中间量
// （见 w8_dynamic_grouped_conv_add_materialized.cu 的 finish_kernel 调用点），所以把这个
// tail 与投影拆开不引入新的舍入边界：TP2 下投影走"每卡半 K partial -> 一次 allreduce"，
// allreduce 结果两卡逐位相同，tail 在两卡各自复制执行即可保持残差逐位一致。
//
// 形状契约与融合 op 的 tail 完全一致：projected/residual 为连续 BF16 [5120, tokens]
// （dim-0 最快），base_kernel 为连续 BF16 [5120,2,2]，finish_delta 为连续 BF16
// [320,2,W,B]，tokens = W*B 且 W∈[2,16]、B∈[1,8]。调用方负责契约校验（见 ops 包装层）。
void dynamic_grouped_conv_add_tail_launch(const Tensor& projected, const Tensor& base_kernel,
                                          const Tensor& finish_delta, Tensor& residual,
                                          cudaStream_t stream);

} // namespace ninfer::ops::detail
