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
NK_API_COMPTIME std::size_t attention_pack_size(std::size_t key_value_head_count, std::size_t depth,
                                                nk_u32_t const *segment_lengths, std::size_t segment_count) {
    constexpr bool simd = allow_simd_ == prefer_simd_k;
    if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        return nk_attention_pack_size_bf16(key_value_head_count, depth, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        return nk_attention_pack_size_e4m3(key_value_head_count, depth, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        return nk_attention_pack_size_i8(key_value_head_count, depth, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return nk_attention_pack_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return nk_attention_pack_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return nk_attention_pack_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    else return 0;
}

/**
 *  @brief Packs ragged K/V token matrices into a backend-opaque KV-cache blob.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`.
 */
template <numeric_dtype in_type_, allow_simd_t allow_simd_ = prefer_simd_k>
NK_API_COMPTIME void attention_pack(in_type_ const *keys, in_type_ const *values, std::size_t key_value_head_count,
                                    std::size_t depth, nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                    std::size_t segment_count, std::size_t key_stride_bytes,
                                    std::size_t value_stride_bytes, void *key_value_packed, std::size_t begin = 0,
                                    std::size_t end = static_cast<std::size_t>(-1)) {
    using raw_t = typename in_type_::raw_t;
    constexpr bool simd = allow_simd_ == prefer_simd_k;
    raw_t const *keys_raw = reinterpret_cast<raw_t const *>(keys);
    raw_t const *values_raw = reinterpret_cast<raw_t const *>(values);
    if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_attention_pack_bf16(keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths,
                               segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, begin, end);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_attention_pack_e4m3(keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths,
                               segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, begin, end);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_attention_pack_i8(keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths,
                             segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, begin, end);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        nk_attention_pack_bf16_serial(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                      segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                      key_value_packed, begin, end);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        nk_attention_pack_e4m3_serial(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                      segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                      key_value_packed, begin, end);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        nk_attention_pack_i8_serial(keys_raw, values_raw, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, begin, end);
}

/**
 *  @brief Ragged scaled-dot-product attention against a pre-packed KV-cache.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`.
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::attention_result_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
NK_API_COMPTIME void attention_packed(in_type_ const *queries, void const *key_value_packed, result_type_ *output,
                                      std::size_t head_count, std::size_t key_value_head_count, std::size_t depth,
                                      nk_u32_t const *query_offsets, std::size_t query_stride_bytes,
                                      std::size_t output_stride_bytes, nk_f32_t scale, std::size_t begin = 0,
                                      std::size_t end = static_cast<std::size_t>(-1)) {
    using raw_t = typename in_type_::raw_t;
    static_assert(std::is_same_v<result_type_, typename in_type_::attention_result_t>,
                  "Attention accumulates and normalizes in F32");
    constexpr bool simd = allow_simd_ == prefer_simd_k;
    raw_t const *queries_raw = reinterpret_cast<raw_t const *>(queries);
    nk_f32_t *output_raw = reinterpret_cast<nk_f32_t *>(output);
    if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_attention_packed_bf16(queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                                 query_offsets, query_stride_bytes, output_stride_bytes, scale, begin, end);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_attention_packed_e4m3(queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                                 query_offsets, query_stride_bytes, output_stride_bytes, scale, begin, end);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_attention_packed_i8(queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                               query_offsets, query_stride_bytes, output_stride_bytes, scale, begin, end);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        nk_attention_packed_bf16_serial(queries_raw, key_value_packed, output_raw, head_count, key_value_head_count,
                                        depth, query_offsets, query_stride_bytes, output_stride_bytes, scale, begin,
                                        end);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        nk_attention_packed_e4m3_serial(queries_raw, key_value_packed, output_raw, head_count, key_value_head_count,
                                        depth, query_offsets, query_stride_bytes, output_stride_bytes, scale, begin,
                                        end);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        nk_attention_packed_i8_serial(queries_raw, key_value_packed, output_raw, head_count, key_value_head_count,
                                      depth, query_offsets, query_stride_bytes, output_stride_bytes, scale, begin, end);
}

} // namespace ashvardanian::numkong

#endif // NK_ATTENTION_HPP
