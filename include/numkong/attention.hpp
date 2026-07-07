/**
 *  @brief C++ bindings for multi-target ragged scaled-dot-product attention kernels.
 *  @file include/numkong/attention.hpp
 *  @author Ash Vardanian
 *  @date July 7, 2026
 */
#ifndef NK_ATTENTION_HPP
#define NK_ATTENTION_HPP

#include <cstddef>

#include "numkong/attention.h"
#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Returns the packed KV-cache size in bytes for a ragged batch of segments.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`.
 */
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_PUBLIC std::size_t attention_packed_size(std::size_t num_kv_heads, std::size_t head_dim,
                                            nk_u32_t const *segment_lengths, std::size_t segment_count) {
    constexpr bool simd = allow_simd_ == prefer_simd_k;
    if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        return nk_attention_packed_size_bf16(num_kv_heads, head_dim, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        return nk_attention_packed_size_e4m3(num_kv_heads, head_dim, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        return nk_attention_packed_size_i8(num_kv_heads, head_dim, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return nk_attention_packed_size_bf16_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return nk_attention_packed_size_e4m3_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return nk_attention_packed_size_i8_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
    else return 0;
}

/**
 *  @brief Packs ragged K/V token matrices into a backend-opaque KV-cache blob.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`.
 */
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_PUBLIC void attention_pack(in_type_ const *k, in_type_ const *v, std::size_t num_kv_heads, std::size_t head_dim,
                              nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                              std::size_t segment_count, std::size_t k_stride_in_bytes, std::size_t v_stride_in_bytes,
                              void *kv_packed, std::size_t first_task = 0, std::size_t task_count = 0) {
    using raw_t = typename in_type_::raw_t;
    constexpr bool simd = allow_simd_ == prefer_simd_k;
    raw_t const *k_raw = reinterpret_cast<raw_t const *>(k);
    raw_t const *v_raw = reinterpret_cast<raw_t const *>(v);
    if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_attention_pack_bf16(k_raw, v_raw, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                               k_stride_in_bytes, v_stride_in_bytes, kv_packed, first_task, task_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_attention_pack_e4m3(k_raw, v_raw, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                               k_stride_in_bytes, v_stride_in_bytes, kv_packed, first_task, task_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_attention_pack_i8(k_raw, v_raw, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                             k_stride_in_bytes, v_stride_in_bytes, kv_packed, first_task, task_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        nk_attention_pack_bf16_serial(k_raw, v_raw, num_kv_heads, head_dim, segment_offsets, segment_lengths,
                                      segment_count, k_stride_in_bytes, v_stride_in_bytes, kv_packed, first_task,
                                      task_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        nk_attention_pack_e4m3_serial(k_raw, v_raw, num_kv_heads, head_dim, segment_offsets, segment_lengths,
                                      segment_count, k_stride_in_bytes, v_stride_in_bytes, kv_packed, first_task,
                                      task_count);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        nk_attention_pack_i8_serial(k_raw, v_raw, num_kv_heads, head_dim, segment_offsets, segment_lengths,
                                    segment_count, k_stride_in_bytes, v_stride_in_bytes, kv_packed, first_task,
                                    task_count);
}

/**
 *  @brief Ragged scaled-dot-product attention against a pre-packed KV-cache.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`.
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::attention_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
NK_PUBLIC void attention_packed(in_type_ const *queries, void const *kv_packed, result_type_ *output,
                                std::size_t num_heads, std::size_t num_kv_heads, std::size_t head_dim,
                                nk_u32_t const *query_offsets, std::size_t q_stride_in_bytes,
                                std::size_t o_stride_in_bytes, nk_f32_t scale, std::size_t first_task = 0,
                                std::size_t task_count = 0) {
    using raw_t = typename in_type_::raw_t;
    static_assert(std::is_same_v<result_type_, typename in_type_::attention_result_t>,
                  "Attention accumulates and normalizes in F32");
    constexpr bool simd = allow_simd_ == prefer_simd_k;
    raw_t const *q_raw = reinterpret_cast<raw_t const *>(queries);
    nk_f32_t *output_raw = reinterpret_cast<nk_f32_t *>(output);
    if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_attention_packed_bf16(q_raw, kv_packed, output_raw, num_heads, num_kv_heads, head_dim, query_offsets,
                                 q_stride_in_bytes, o_stride_in_bytes, scale, first_task, task_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_attention_packed_e4m3(q_raw, kv_packed, output_raw, num_heads, num_kv_heads, head_dim, query_offsets,
                                 q_stride_in_bytes, o_stride_in_bytes, scale, first_task, task_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_attention_packed_i8(q_raw, kv_packed, output_raw, num_heads, num_kv_heads, head_dim, query_offsets,
                               q_stride_in_bytes, o_stride_in_bytes, scale, first_task, task_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        nk_attention_packed_bf16_serial(q_raw, kv_packed, output_raw, num_heads, num_kv_heads, head_dim, query_offsets,
                                        q_stride_in_bytes, o_stride_in_bytes, scale, first_task, task_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        nk_attention_packed_e4m3_serial(q_raw, kv_packed, output_raw, num_heads, num_kv_heads, head_dim, query_offsets,
                                        q_stride_in_bytes, o_stride_in_bytes, scale, first_task, task_count);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        nk_attention_packed_i8_serial(q_raw, kv_packed, output_raw, num_heads, num_kv_heads, head_dim, query_offsets,
                                      q_stride_in_bytes, o_stride_in_bytes, scale, first_task, task_count);
}

} // namespace ashvardanian::numkong

#endif // NK_ATTENTION_HPP
