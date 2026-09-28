/**
 *  @file include/numkong/attention.hpp
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief C++ bindings for multi-target ragged scaled-dot-product attention kernels.
 */
#ifndef NUMKONG_ATTENTION_HPP
#define NUMKONG_ATTENTION_HPP

#include <cstddef>

#include "numkong/attention.h"
#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Returns the packed KV-cache size in bytes for a ragged batch of segments, or zero when no
 *      capability in @p capabilities packs @p in_type_. No capability runs the serial kernel, the reference.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 */
template <numeric_dtype in_type_>
std::size_t attention_pack_size(std::size_t key_value_head_count, std::size_t depth, nk_u32_t const *segment_lengths,
                                std::size_t segment_count, nk_capability_t capabilities = cpu_capabilities()) {
    nk_size_t bytes = 0;
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        (capabilities
             ? nk_attention_pack_size_bf16_best(key_value_head_count, depth, segment_lengths, segment_count,
                                                capabilities, &bytes)
             : nk_attention_pack_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count, &bytes));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        (capabilities
             ? nk_attention_pack_size_e4m3_best(key_value_head_count, depth, segment_lengths, segment_count,
                                                capabilities, &bytes)
             : nk_attention_pack_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count, &bytes));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        (capabilities
             ? nk_attention_pack_size_i8_best(key_value_head_count, depth, segment_lengths, segment_count, capabilities,
                                              &bytes)
             : nk_attention_pack_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count, &bytes));
    return bytes;
}

/**
 *  @brief Packs ragged K/V token matrices into a backend-opaque KV-cache blob.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 */
template <numeric_dtype in_type_>
nk_status_t attention_pack(in_type_ const *keys, in_type_ const *values, std::size_t key_value_head_count,
                           std::size_t depth, nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                           std::size_t segment_count, std::size_t key_stride_bytes, std::size_t value_stride_bytes,
                           void *key_value_packed, std::size_t task_begin = 0,
                           std::size_t task_end = static_cast<std::size_t>(-1),
                           nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) {
    using raw_t = typename in_type_::raw_t;
    raw_t const *keys_raw = reinterpret_cast<raw_t const *>(keys);
    raw_t const *values_raw = reinterpret_cast<raw_t const *>(values);
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return (capabilities
                    ? nk_attention_pack_bf16_best(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                                  segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                                  key_value_packed, task_begin, task_end, capabilities, stream)
                    : nk_attention_pack_bf16_serial(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                                    segment_lengths, segment_count, key_stride_bytes,
                                                    value_stride_bytes, key_value_packed, task_begin, task_end,
                                                    stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return (capabilities
                    ? nk_attention_pack_e4m3_best(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                                  segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                                  key_value_packed, task_begin, task_end, capabilities, stream)
                    : nk_attention_pack_e4m3_serial(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                                    segment_lengths, segment_count, key_stride_bytes,
                                                    value_stride_bytes, key_value_packed, task_begin, task_end,
                                                    stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return (capabilities
                    ? nk_attention_pack_i8_best(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                                segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                                key_value_packed, task_begin, task_end, capabilities, stream)
                    : nk_attention_pack_i8_serial(keys_raw, values_raw, key_value_head_count, depth, segment_offsets,
                                                  segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                                  key_value_packed, task_begin, task_end, stream));
    else return nk_missing_kernel_k;
}

/**
 *  @brief Ragged bidirectional scaled-dot-product attention against a pre-packed KV-cache.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::attention_result_t>
nk_status_t attention_bidirectional_packed(in_type_ const *queries, void const *key_value_packed, result_type_ *output,
                                           std::size_t head_count, std::size_t key_value_head_count, std::size_t depth,
                                           nk_u32_t const *query_offsets, std::size_t query_stride_bytes,
                                           std::size_t output_stride_bytes, nk_f32_t scale, std::size_t task_start = 0,
                                           std::size_t task_count = static_cast<std::size_t>(-1),
                                           nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) {
    using raw_t = typename in_type_::raw_t;
    static_assert(std::is_same_v<result_type_, typename in_type_::attention_result_t>,
                  "Attention accumulates and normalizes in F32");
    raw_t const *queries_raw = reinterpret_cast<raw_t const *>(queries);
    nk_f32_t *output_raw = reinterpret_cast<nk_f32_t *>(output);
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return (
            capabilities
                ? nk_attention_bidirectional_packed_bf16_best(
                      queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                      query_stride_bytes, output_stride_bytes, scale, task_start, task_count, capabilities, stream)
                : nk_attention_bidirectional_packed_bf16_serial(
                      queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                      query_stride_bytes, output_stride_bytes, scale, task_start, task_count, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return (
            capabilities
                ? nk_attention_bidirectional_packed_e4m3_best(
                      queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                      query_stride_bytes, output_stride_bytes, scale, task_start, task_count, capabilities, stream)
                : nk_attention_bidirectional_packed_e4m3_serial(
                      queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                      query_stride_bytes, output_stride_bytes, scale, task_start, task_count, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return (
            capabilities
                ? nk_attention_bidirectional_packed_i8_best(
                      queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                      query_stride_bytes, output_stride_bytes, scale, task_start, task_count, capabilities, stream)
                : nk_attention_bidirectional_packed_i8_serial(
                      queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                      query_stride_bytes, output_stride_bytes, scale, task_start, task_count, stream));
    else return nk_missing_kernel_k;
}

/**
 *  @brief Ragged causal, optionally sliding-window, attention against a pre-packed KV-cache.
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::attention_result_t>
nk_status_t attention_causal_packed(in_type_ const *queries, void const *key_value_packed, result_type_ *output,
                                    std::size_t head_count, std::size_t key_value_head_count, std::size_t depth,
                                    nk_u32_t const *query_offsets, std::size_t query_stride_bytes,
                                    std::size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset = 0,
                                    std::size_t window = static_cast<std::size_t>(-1), std::size_t task_start = 0,
                                    std::size_t task_count = static_cast<std::size_t>(-1),
                                    nk_capability_t capabilities = cpu_capabilities(), void *stream = nullptr) {
    using raw_t = typename in_type_::raw_t;
    static_assert(std::is_same_v<result_type_, typename in_type_::attention_result_t>,
                  "Attention accumulates and normalizes in F32");
    raw_t const *queries_raw = reinterpret_cast<raw_t const *>(queries);
    nk_f32_t *output_raw = reinterpret_cast<nk_f32_t *>(output);
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return (capabilities ? nk_attention_causal_packed_bf16_best(
                                   queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                                   query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                   window, task_start, task_count, capabilities, stream)
                             : nk_attention_causal_packed_bf16_serial(
                                   queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                                   query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                   window, task_start, task_count, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return (capabilities ? nk_attention_causal_packed_e4m3_best(
                                   queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                                   query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                   window, task_start, task_count, capabilities, stream)
                             : nk_attention_causal_packed_e4m3_serial(
                                   queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                                   query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                   window, task_start, task_count, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return (capabilities ? nk_attention_causal_packed_i8_best(
                                   queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                                   query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                   window, task_start, task_count, capabilities, stream)
                             : nk_attention_causal_packed_i8_serial(
                                   queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth,
                                   query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                   window, task_start, task_count, stream));
    else return nk_missing_kernel_k;
}

} // namespace ashvardanian::numkong

#endif // NUMKONG_ATTENTION_HPP
