#pragma once

// ninfer::ops - INT4-G64 KV cache codec（K/V 同档，无 Hadamard 旋转）。
//
// 存储：每 (token, head) 128 B 对称 int4 打包码（每字节 2 个，低 nibble = 偶数维，二进制
// 补码，量化端对称钳位 ±7）+ 8 B fp16 scale（每 64 维 1 个，与 int8 档的 scale 平面逐项同形）。
// 量化/取整口径与 int8 档（gqa_attention_kv_quant.cuh）完全一致：8-lane 组内 xor 归约 absmax、
// fp16 舍入后的 scale 取倒数、round-to-nearest、对称钳位 —— 区别只有码宽 8→4 bit 和打包。
// scale 平面的索引直接复用 int8 的 gqa_kv_quant_scale_index（同 4 组、同 fp16）。

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheInt4HeadDim   = 256;
inline constexpr int kKVCacheInt4Group     = 64;
inline constexpr int kKVCacheInt4Groups    = kKVCacheInt4HeadDim / kKVCacheInt4Group;
inline constexpr int kKVCacheInt4CodeBytes = kKVCacheInt4HeadDim / 2;
inline constexpr float kKVCacheInt4MaxCode = 7.0F;

static_assert(kKVCacheInt4Group == kGqaKvQuantGroup && kKVCacheInt4Groups == kGqaKvQuantGroups,
              "int4-g64 shares the int8 scale-plane geometry");

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_int4_code_index(int physical_page, int kv_head,
                                                                 int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt4CodeBytes, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 1);
}

// 单个值的对称 int4 量化：inv_scale 是 fp16 舍入后 scale 的倒数（与 int8 档同一口径，
// 只是钳位边界 ±127 → ±7）。返回 4 bit 补码（未移位）。
__device__ __forceinline__ std::uint32_t kv_cache_int4_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return 0u; }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-7, min(7, q));
    return static_cast<std::uint32_t>(q) & 0xFu;
}

// 8 个值打包成 4 字节：dims [d, d+8) → byte i 的低 nibble 是 d+2i、高 nibble 是 d+2i+1。
__device__ __forceinline__ std::uint32_t kv_cache_int4_pack8(const float* values, float inv_scale) {
    std::uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { packed |= kv_cache_int4_code(values[i], inv_scale) << (4 * i); }
    return packed;
}

// 从（全局或共享）内存解量化 8 个连续维（8 对齐 ⇒ 落在同一 64 组内）：4 字节码 + 该组的
// fp16 scale → 8 个 bf16 打包成 int4。与 gqa_kv_dequant_i8x8_from 同构。
__device__ __forceinline__ int4 kv_cache_int4_dequant_x8_from(const std::uint8_t* codes4, float s) {
    const std::uint32_t packed = load_vec<std::uint32_t>(codes4);
    unsigned words[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t byte = (packed >> (8 * i)) & 0xFFu;
        // 补码 4 bit → [-8, 7]；量化端钳位在 ±7，-8 不会出现，解码仍按补码展开。
        const int lo  = static_cast<int>(byte & 0xFu);
        const int hi  = static_cast<int>(byte >> 4);
        const float x0 = static_cast<float>(lo >= 8 ? lo - 16 : lo) * s;
        const float x1 = static_cast<float>(hi >= 8 ? hi - 16 : hi) * s;
        words[i]       = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(words[0]), static_cast<int>(words[1]),
                     static_cast<int>(words[2]), static_cast<int>(words[3]));
}

} // namespace ninfer::ops
