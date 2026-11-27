/**
 *  @file cpp/numkong/attention.hpp
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief C++ bindings for multi-target ragged scaled-dot-product attention kernels.
 *
 *  Three layers, like the dots and maxsim bindings: raw-pointer wrappers that keep the task windows
 *  for sharded launches, the owning @c packed_attention from <tt>matrix.hpp</tt> with @c attention_pack,
 *  and view overloads that derive heads, depth and strides from @b [tokens,heads,depth] views.
 *
 *  @code{.cpp}
 *  auto [packed, packed_status] = nk::attention_pack<nk::bf16_t>(keys, values, offsets, lengths);
 *  if (nk::failed(packed_status)) return packed_status;
 *  std::size_t const all_keys = (std::numeric_limits<std::size_t>::max)(), causal = 0;
 *  nk::status_t status = nk::attention_packed<nk::bf16_t>(queries, packed, output, scale, all_keys, causal);
 *  auto [fresh, fresh_status] = nk::attention_packed<nk::bf16_t>(queries, packed, scale);
 *  @endcode
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_ATTENTION_HPP
#define NUMKONG_ATTENTION_HPP

#include <cstddef>
#include <limits>
#include <type_traits>

#include "numkong/attention.h"
#include "numkong/devices.hpp" // `default_capabilities`
#include "numkong/matrix.hpp"  // `packed_attention`

namespace ashvardanian::numkong {

/**
 *  @brief Bounds the packed KV-cache of a ragged batch of segments, however its tokens split.
 *  @param[in] key_value_head_count Number of K/V heads, a nonzero divisor of the query head count.
 *  @param[in] depth Head dimension; any value ≥ 1.
 *  @param[in] token_count Live tokens across all segments.
 *  @param[in] segment_count Number of segments packed together.
 *  @param[in] capabilities Capabilities to pick from.
 *  @return The size in bytes.
 *
 *  @tparam in_type_ Input element type (bf16_t, f16_t, e4m3_t, i8_t), or a block-scaled format.
 */
template <a_numeric_dtype in_type_>
expected<std::size_t> attention_pack_size(std::size_t key_value_head_count, std::size_t depth, std::size_t token_count,
                                          std::size_t segment_count,
                                          nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_attention_pack_size_bf16_best(key_value_head_count, depth, token_count, segment_count, capabilities,
                                                  &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_attention_pack_size_f16_best(key_value_head_count, depth, token_count, segment_count, capabilities,
                                                 &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_attention_pack_size_e4m3_best(key_value_head_count, depth, token_count, segment_count, capabilities,
                                                  &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_attention_pack_size_i8_best(key_value_head_count, depth, token_count, segment_count, capabilities,
                                                &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_attention_pack_size_nvfp4_best(key_value_head_count, depth, token_count, segment_count,
                                                   capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_attention_pack_size_mxfp4_best(key_value_head_count, depth, token_count, segment_count,
                                                   capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_attention_pack_size_mxfp6e2m3_best(key_value_head_count, depth, token_count, segment_count,
                                                       capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_attention_pack_size_mxfp6e3m2_best(key_value_head_count, depth, token_count, segment_count,
                                                       capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_attention_pack_size_mxfp8e4m3_best(key_value_head_count, depth, token_count, segment_count,
                                                       capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_attention_pack_size_mxfp8e5m2_best(key_value_head_count, depth, token_count, segment_count,
                                                       capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_attention_pack_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Packs ragged K/V token matrices into a backend-opaque KV-cache blob.
 *  @param[in] key_value_head_count K/V heads.
 *  @param[in] depth Head dimension.
 *  @param[in] key_offsets The @p segment_count + 1 slot boundaries, which must not decrease.
 *  @param[in] key_lengths Keys each segment holds, at most its slot's width; zeros mark padding
 *      slots. Null means every slot is full.
 *  @param[in] segment_count Number of segments in the batch.
 *  @param[in] keys Token-major matrix, element (token, head, d) at @p keys + token × @p key_stride
 *      + (head × depth + d) elements. Packing fuses no transposition, so depth-major K is
 *      transposed in a separate pass first.
 *  @param[in] values Token-major matrix laid out like @p keys over @p value_stride. Depth-major V
 *      is likewise transposed in a separate pass first.
 *  @param[in] key_stride Bytes between tokens of @p keys.
 *  @param[in] value_stride Bytes between tokens of @p values.
 *  @param[out] key_value_packed 64-byte-aligned buffer of @c attention_pack_size bytes.
 *  @param[in] tasks_begin First task of a window over the segments × K/V heads grid.
 *  @param[in] tasks_end End of that half-open window, clipped to the grid, so callers can shard.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type (bf16_t, f16_t, e4m3_t, i8_t), or a block-scaled format
 *      whose operands are @c operand_pointer references.
 */
template <a_numeric_dtype in_type_>
status_t attention_pack(std::size_t key_value_head_count, std::size_t depth, std::uint32_t const *key_offsets,
                        std::uint32_t const *key_lengths, std::size_t segment_count, operand_pointer<in_type_> keys,
                        std::size_t key_stride, operand_pointer<in_type_> values, std::size_t value_stride,
                        void *key_value_packed, std::size_t tasks_begin = 0,
                        std::size_t tasks_end = (std::numeric_limits<std::size_t>::max)(),
                        nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_pack_bf16_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, &keys->raw_, key_stride,
            &values->raw_, value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_attention_pack_f16_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, &keys->raw_, key_stride,
            &values->raw_, value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_attention_pack_e4m3_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, &keys->raw_, key_stride,
            &values->raw_, value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_attention_pack_i8_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, &keys->raw_, key_stride,
            &values->raw_, value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_attention_pack_nvfp4_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, keys, key_stride, values,
            value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_attention_pack_mxfp4_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, keys, key_stride, values,
            value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_attention_pack_mxfp6e2m3_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, keys, key_stride, values,
            value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_attention_pack_mxfp6e3m2_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, keys, key_stride, values,
            value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_attention_pack_mxfp8e4m3_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, keys, key_stride, values,
            value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_attention_pack_mxfp8e5m2_best(
            key_value_head_count, depth, key_offsets, key_lengths, segment_count, keys, key_stride, values,
            value_stride, key_value_packed, tasks_begin, tasks_end, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_attention_pack has no kernel for this dtype");
}

/**
 *  @brief Sizes, allocates and packs a ragged batch of K and V tokens into an owner the view
 *      overloads of @c attention_packed attend to.
 *  @param[in] keys @b [tokens,key_value_heads,depth] view, its heads back to back along depth.
 *  @param[in] values @b [tokens,key_value_heads,depth] view laid out like @p keys.
 *  @param[in] key_offsets The slot boundaries of the segments, one more than there are segments.
 *  @param[in] key_lengths Keys each segment holds; empty means every slot is full.
 *  @param[in] alloc Allocator of the pack.
 *  @param[in] capabilities Capabilities to pack for; attend with the same ones.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *  @return The pack, failing like @c packed_attention::make.
 */
template <a_numeric_dtype value_type_, typename allocator_type_ = aligned_allocator<char>, std::size_t max_rank_>
expected<packed_attention<value_type_, allocator_type_>> attention_pack(
    tensor_view<value_type_, max_rank_> keys, tensor_view<value_type_, max_rank_> values,
    vector_view<std::uint32_t> key_offsets, vector_view<std::uint32_t> key_lengths = {}, allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    auto const pack_size = [capabilities](nk_size_t heads, nk_size_t depth, nk_size_t tokens, nk_size_t segments,
                                          nk_size_t *bytes) noexcept {
        auto const [size, status] = attention_pack_size<value_type_>(heads, depth, tokens, segments, capabilities);
        *bytes = size;
        return status;
    };
    auto const pack = [capabilities, stream](
                          nk_size_t heads, nk_size_t depth, std::uint32_t const *offsets, std::uint32_t const *lengths,
                          nk_size_t segments, operand_pointer<value_type_> keys, nk_size_t key_stride,
                          operand_pointer<value_type_> values, nk_size_t value_stride, void *packed) noexcept {
        return attention_pack<value_type_>(heads, depth, offsets, lengths, segments, keys, key_stride, values,
                                           value_stride, packed, 0, (std::numeric_limits<std::size_t>::max)(),
                                           capabilities, stream);
    };
    return packed_attention<value_type_, allocator_type_>::make(keys, values, key_offsets, key_lengths, alloc,
                                                                pack_size, pack);
}

/** The slot boundaries @p pack was packed with, which the view overloads take as the query offsets;
 *  empty for an empty pack. */
template <a_numeric_dtype value_type_, typename allocator_type_>
vector_view<std::uint32_t> attention_key_offsets(packed_attention<value_type_, allocator_type_> const &pack) noexcept {
    std::uint32_t const *offsets = nullptr, *lengths = nullptr;
    if (pack.empty() || failed(static_cast<status_t>(
                            nk_attention_packed_segments(pack.data(), pack.segment_count(), &offsets, &lengths))))
        return {};
    return {offsets, pack.segment_count() + 1};
}

/** The keys each segment of @p pack holds; empty for an empty pack. */
template <a_numeric_dtype value_type_, typename allocator_type_>
vector_view<std::uint32_t> attention_key_lengths(packed_attention<value_type_, allocator_type_> const &pack) noexcept {
    std::uint32_t const *offsets = nullptr, *lengths = nullptr;
    if (pack.empty() || failed(static_cast<status_t>(
                            nk_attention_packed_segments(pack.data(), pack.segment_count(), &offsets, &lengths))))
        return {};
    return {lengths, pack.segment_count()};
}

/**
 *  @brief Ragged scaled-dot-product attention against a pre-packed KV-cache, under a band of keys.
 *  @param[in] head_count Query heads.
 *  @param[in] key_value_head_count K/V heads, at most @p head_count.
 *  @param[in] depth Head dimension.
 *  @param[in] query_offsets First query row of each segment, as segment count + 1 prefix sums.
 *  @param[in] query_token_count Query rows the call addresses, <tt>query_offsets[segments]</tt>; GPU
 *      kernels size tensor maps and workspaces from it without reading the offsets.
 *  @param[in] scale Score multiplier, typically 1 / √depth.
 *  @param[in] keys_before Keys visible before each query's position, all of them for @c max.
 *  @param[in] keys_after Keys visible after each query's position, all of them for @c max.
 *  @param[in] queries Token-major matrix, one row of @p head_count × @p depth elements per token.
 *  @param[in] queries_stride Row (token) stride of @p queries in bytes.
 *  @param[in] key_value_packed A buffer @c attention_pack filled with the same @p capabilities.
 *  @param[out] output Token-major matrix, one row of @p head_count × @p depth results per token.
 *  @param[in] output_stride Row (token) stride of @p output in bytes.
 *  @param[out] log_sum_exp Optional natural log-sum-exp per query token and head, for training.
 *  @param[in] tasks_begin First task of a window over the query tokens × heads grid of output rows.
 *  @param[in] tasks_end End of that half-open window, clipped to the grid. The window partitions
 *      the rows of @p output and @p log_sum_exp the call writes, per query token × head.
 *  @param[in] workspace Scratch memory of @c attention_packed_workspace_size bytes on the device of
 *      @p stream, or empty for the path without scratch.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  Queries align to the end of each segment's keys, as @c nk_attention_packed_bf16_best describes:
 *  causal sets @p keys_after to 0, and a sliding window of @c w keys sets @p keys_before to w − 1.
 *
 *  @tparam in_type_ Input element type (bf16_t, f16_t, e4m3_t, i8_t), or a block-scaled format
 *      whose queries are an @c operand_pointer reference.
 */
template <a_numeric_dtype in_type_>
status_t attention_packed(std::size_t head_count, std::size_t key_value_head_count, std::size_t depth,
                          std::uint32_t const *query_offsets, std::size_t query_token_count, f32_t scale,
                          std::size_t keys_before, std::size_t keys_after, operand_pointer<in_type_> queries,
                          std::size_t queries_stride, void const *key_value_packed,
                          typename in_type_::attention_result_t *output, std::size_t output_stride, f32_t *log_sum_exp,
                          std::size_t tasks_begin = 0,
                          std::size_t tasks_end = (std::numeric_limits<std::size_t>::max)(),
                          std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                          nk_stream_t stream = nullptr) noexcept {
    nk_f32_t *output_raw = reinterpret_cast<nk_f32_t *>(output);
    nk_f32_t *log_sum_exp_raw = reinterpret_cast<nk_f32_t *>(log_sum_exp);
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_packed_bf16_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, &queries->raw_, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_attention_packed_f16_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, &queries->raw_, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_attention_packed_e4m3_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, &queries->raw_, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_attention_packed_i8_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, &queries->raw_, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_attention_packed_nvfp4_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp4_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp6e2m3_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp6e3m2_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp8e4m3_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp8e5m2_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_attention_packed has no kernel for this dtype");
}

/**
 *  @brief Returns the workspace bytes @c attention_packed takes for its fast path on these shapes.
 *  @param[in] head_count Query heads.
 *  @param[in] key_value_head_count K/V heads, at most @p head_count.
 *  @param[in] depth Head dimension.
 *  @param[in] query_token_count Query rows the call addresses.
 *  @param[in] keys_before Keys visible before each query's position, all of them for @c max.
 *  @param[in] keys_after Keys visible after each query's position, all of them for @c max.
 *  @param[in] capabilities Capabilities to pick from.
 */
template <a_numeric_dtype in_type_>
expected<std::size_t> attention_packed_workspace_size(std::size_t head_count, std::size_t key_value_head_count,
                                                      std::size_t depth, std::size_t query_token_count,
                                                      std::size_t keys_before, std::size_t keys_after,
                                                      nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_attention_packed_workspace_size_bf16_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_attention_packed_workspace_size_f16_best(head_count, key_value_head_count, depth, query_token_count,
                                                             keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_attention_packed_workspace_size_e4m3_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_attention_packed_workspace_size_i8_best(head_count, key_value_head_count, depth, query_token_count,
                                                            keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_attention_packed_workspace_size_nvfp4_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_attention_packed_workspace_size_mxfp4_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_attention_packed_workspace_size_mxfp6e2m3_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_attention_packed_workspace_size_mxfp6e3m2_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_attention_packed_workspace_size_mxfp8e4m3_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_attention_packed_workspace_size_mxfp8e5m2_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else static_assert(sizeof(in_type_) == 0, "nk_attention_packed_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief Gradients of @c attention_packed with respect to its queries, keys and values.
 *  @param[in] head_count Query heads.
 *  @param[in] key_value_head_count K/V heads, at most @p head_count.
 *  @param[in] depth Head dimension.
 *  @param[in] query_offsets First query row of each segment, as in the forward pass.
 *  @param[in] query_token_count Query rows the call addresses, as in the forward pass.
 *  @param[in] scale Score multiplier of the forward pass.
 *  @param[in] keys_before Keys before each query that it saw in the forward pass.
 *  @param[in] keys_after Keys after each query that it saw in the forward pass.
 *  @param[in] queries The forward's queries, token-major.
 *  @param[in] queries_stride Row (token) stride of @p queries in bytes.
 *  @param[in] key_value_packed The pack the forward pass read.
 *  @param[in] output The forward's output for these queries and pack.
 *  @param[in] output_gradient Gradient of the loss with respect to @p output, laid out like it.
 *  @param[in] output_stride Row (token) stride of @p output and @p output_gradient in bytes.
 *  @param[in] log_sum_exp The forward's log-sum-exp for these queries and pack.
 *  @param[out] query_gradient Gradient with respect to @p queries, one row of
 *      @p head_count × @p depth per query token.
 *  @param[in] query_gradient_stride Bytes between the @p query_gradient rows of consecutive tokens.
 *  @param[out] key_gradient Gradient with respect to the packed keys, one row of
 *      @p key_value_head_count × @p depth per key token; rows past a segment's live keys are
 *      left unwritten by the kernel.
 *  @param[out] value_gradient Gradient of the packed values, shaped like @p key_gradient.
 *  @param[in] key_value_gradient_stride Bytes between the gradient rows of consecutive key tokens.
 *  @param[in] tasks_begin First task of a window over the segments × key-value heads grid.
 *  @param[in] tasks_end End of that half-open window, clipped to the grid. The window partitions
 *      the outputs the call writes: segments × K/V heads own their @p key_gradient and
 *      @p value_gradient rows and the @p query_gradient rows of their query heads.
 *
 *  @param[in] workspace Scratch memory of @c attention_packed_gradients_workspace_size bytes on the
 *      device of @p stream, or empty for the two-pass path.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  Every other parameter follows @c attention_packed; the band must be the forward's.
 *
 *  @tparam in_type_ Input element type, bf16_t, or a block-scaled format whose queries are an
 *      @c operand_pointer reference.
 */
template <a_numeric_dtype in_type_>
status_t attention_packed_gradients(
    std::size_t head_count, std::size_t key_value_head_count, std::size_t depth, std::uint32_t const *query_offsets,
    std::size_t query_token_count, f32_t scale, std::size_t keys_before, std::size_t keys_after,
    operand_pointer<in_type_> queries, std::size_t queries_stride, void const *key_value_packed, f32_t const *output,
    f32_t const *output_gradient, std::size_t output_stride, f32_t const *log_sum_exp, f32_t *query_gradient,
    std::size_t query_gradient_stride, f32_t *key_gradient, f32_t *value_gradient,
    std::size_t key_value_gradient_stride, std::size_t tasks_begin = 0,
    std::size_t tasks_end = (std::numeric_limits<std::size_t>::max)(), std::span<std::byte> workspace = {},
    nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    nk_f32_t const *output_raw = reinterpret_cast<nk_f32_t const *>(output);
    nk_f32_t const *output_gradient_raw = reinterpret_cast<nk_f32_t const *>(output_gradient);
    nk_f32_t const *log_sum_exp_raw = reinterpret_cast<nk_f32_t const *>(log_sum_exp);
    nk_f32_t *query_gradient_raw = reinterpret_cast<nk_f32_t *>(query_gradient);
    nk_f32_t *key_gradient_raw = reinterpret_cast<nk_f32_t *>(key_gradient);
    nk_f32_t *value_gradient_raw = reinterpret_cast<nk_f32_t *>(value_gradient);
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_bf16_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, &queries->raw_, queries_stride, key_value_packed, output_raw, output_gradient_raw,
            output_stride, log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw,
            value_gradient_raw, key_value_gradient_stride, tasks_begin, tasks_end, workspace.data(), workspace.size(),
            capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_nvfp4_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities,
            stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp4_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities,
            stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp6e2m3_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities,
            stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp6e3m2_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities,
            stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp8e4m3_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities,
            stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp8e5m2_best(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, workspace.data(), workspace.size(), capabilities,
            stream));
    else static_assert(sizeof(in_type_) == 0, "nk_attention_packed_gradients has no kernel for this dtype");
}

/**
 *  @brief Returns the workspace bytes @c attention_packed_gradients takes for its single pass on
 *      these shapes, or 0 where it has none.
 *  @param[in] head_count Query heads.
 *  @param[in] key_value_head_count K/V heads, at most @p head_count.
 *  @param[in] depth Head dimension.
 *  @param[in] query_token_count Query rows the call addresses.
 *  @param[in] keys_before Keys before each query that it saw in the forward pass.
 *  @param[in] keys_after Keys after each query that it saw in the forward pass.
 *  @param[in] capabilities Capabilities to pick from.
 */
template <a_numeric_dtype in_type_>
expected<std::size_t> attention_packed_gradients_workspace_size(
    std::size_t head_count, std::size_t key_value_head_count, std::size_t depth, std::size_t query_token_count,
    std::size_t keys_before, std::size_t keys_after, nk_capability_t capabilities = default_capabilities()) noexcept {
    nk_size_t bytes = 0;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_attention_packed_gradients_workspace_size_bf16_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        status = nk_attention_packed_gradients_workspace_size_nvfp4_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        status = nk_attention_packed_gradients_workspace_size_mxfp4_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        status = nk_attention_packed_gradients_workspace_size_mxfp6e2m3_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        status = nk_attention_packed_gradients_workspace_size_mxfp6e3m2_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        status = nk_attention_packed_gradients_workspace_size_mxfp8e4m3_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        status = nk_attention_packed_gradients_workspace_size_mxfp8e5m2_best(
            head_count, key_value_head_count, depth, query_token_count, keys_before, keys_after, capabilities, &bytes);
    else
        static_assert(sizeof(in_type_) == 0,
                      "nk_attention_packed_gradients_workspace_size has no kernel for this dtype");
    return {static_cast<std::size_t>(bytes), static_cast<status_t>(status)};
}

/**
 *  @brief NeoX split-half RoPE: rotates channel pairs of every head by per-token angles.
 *
 *  Rotates each pair <tt>(i, i + @p depth / 2)</tt> of every head by the <tt>[rows, @p depth / 2]</tt> grids.
 *
 *  @param[in] x Row-major matrix of shape @b [rows,channels]; channels = @p head_count · @p depth
 *  @param[in] cos <tt>[rows, @p depth / 2]</tt> per-token cosine grid, shared across heads
 *  @param[in] sin <tt>[rows, @p depth / 2]</tt> per-token sine grid, shared across heads
 *  @param[out] y Output, same shape and dtype as x; may alias x for in-place rotation
 *  @param[in] rows Token count
 *  @param[in] head_count Heads per token
 *  @param[in] depth Even channel count per head; channel @c i pairs with <tt>i + depth / 2</tt>
 *  @param[in] x_stride Row (token) stride of x in bytes
 *  @param[in] y_stride Row (token) stride of y in bytes
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 */
template <a_numeric_dtype in_type_>
status_t attention_rope(in_type_ const *x, f32_t const *cos, f32_t const *sin, in_type_ *y, std::size_t rows,
                        std::size_t head_count, std::size_t depth, std::size_t x_stride, std::size_t y_stride,
                        nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if (depth % 2) return status_t::unexpected_dimensions_k;
    if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(nk_attention_rope_f32_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                head_count, depth, x_stride, y_stride, capabilities,
                                                                stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_rope_bf16_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                 head_count, depth, x_stride, y_stride, capabilities,
                                                                 stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_attention_rope_e4m3_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                 head_count, depth, x_stride, y_stride, capabilities,
                                                                 stream));
    else static_assert(sizeof(in_type_) == 0, "nk_attention_rope has no kernel for this dtype");
}

#pragma region Attention Views

/** Checks that @p queries and @p output are matching @b [tokens,heads,depth] layouts over the
 *  @p key_value_packed depth, whose head count divides theirs, covering every packed segment. */
template <a_numeric_dtype value_type_, std::size_t max_rank_, typename allocator_type_, typename output_type_>
status_t attention_shapes_(tensor_view<value_type_, max_rank_> queries,
                           packed_attention<value_type_, allocator_type_> const &key_value_packed,
                           output_type_ const &output) noexcept {
    if (key_value_packed.empty() || !attention_rows_supported_(queries) || !attention_rows_supported_(output))
        return status_t::unexpected_dimensions_k;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (output.extent(axis) != queries.extent(axis)) return status_t::unexpected_dimensions_k;
    if (queries.extent(2) != key_value_packed.depth() ||
        queries.extent(1) % key_value_packed.key_value_head_count() != 0)
        return status_t::unexpected_dimensions_k;
    vector_view<std::uint32_t> const key_offsets = attention_key_offsets(key_value_packed);
    if (key_offsets[key_value_packed.segment_count()] > queries.extent(0)) return status_t::unexpected_dimensions_k;
    return status_t::success_k;
}

/** Self-attention of @b [tokens,heads,depth] @p queries against @p key_value_packed, whose
 *  pack-time key offsets split the query tokens too, under the @p keys_before, @p keys_after
 *  band, with the workspace, capabilities and stream of the pointer overload;
 *  @c unexpected_dimensions_k when the shapes disagree. The raw-pointer overload covers
 *  cross-attention and pooling. */
template <a_numeric_dtype value_type_, std::size_t max_rank_, typename allocator_type_>
status_t attention_packed(tensor_view<value_type_, max_rank_> queries,
                          packed_attention<value_type_, allocator_type_> const &key_value_packed,
                          tensor_span<typename value_type_::attention_result_t, max_rank_> output, f32_t scale,
                          std::size_t keys_before = (std::numeric_limits<std::size_t>::max)(),
                          std::size_t keys_after = (std::numeric_limits<std::size_t>::max)(),
                          std::span<std::byte> workspace = {}, nk_capability_t capabilities = default_capabilities(),
                          nk_stream_t stream = nullptr) noexcept {
    if (status_t status = attention_shapes_(queries, key_value_packed, output); failed(status)) return status;
    vector_view<std::uint32_t> const key_offsets = attention_key_offsets(key_value_packed);
    return attention_packed<value_type_>(
        queries.extent(1), key_value_packed.key_value_head_count(), key_value_packed.depth(), key_offsets.data(),
        key_offsets[key_value_packed.segment_count()], scale, keys_before, keys_after, queries.data(),
        static_cast<std::size_t>(queries.stride_bytes(0)), key_value_packed.data(), output.data(),
        static_cast<std::size_t>(output.stride_bytes(0)), nullptr, 0, (std::numeric_limits<std::size_t>::max)(),
        workspace, capabilities, stream);
}

/** Allocating self-attention returning a fresh @b [tokens,heads,depth] tensor, with any workspace
 *  in its block; @c unexpected_dimensions_k when the shapes disagree, or else the failure of the
 *  allocation or of the kernel itself. */
template <a_numeric_dtype value_type_, std::size_t max_rank_, typename packed_allocator_type_,
          typename allocator_type_ = aligned_allocator<typename value_type_::attention_result_t>>
expected<tensor<typename value_type_::attention_result_t, allocator_type_, max_rank_>> attention_packed(
    tensor_view<value_type_, max_rank_> queries,
    packed_attention<value_type_, packed_allocator_type_> const &key_value_packed, f32_t scale,
    std::size_t keys_before = (std::numeric_limits<std::size_t>::max)(),
    std::size_t keys_after = (std::numeric_limits<std::size_t>::max)(), allocator_type_ alloc = {},
    nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    using out_tensor_t = tensor<typename value_type_::attention_result_t, allocator_type_, max_rank_>;
    if (key_value_packed.empty() || !attention_rows_supported_(queries))
        return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    vector_view<std::uint32_t> const key_offsets = attention_key_offsets(key_value_packed);
    auto workspace_bytes = attention_packed_workspace_size<value_type_>(
        queries.extent(1), key_value_packed.key_value_head_count(), key_value_packed.depth(),
        key_offsets[key_value_packed.segment_count()], keys_before, keys_after, capabilities);
    if (!workspace_bytes) return {out_tensor_t(alloc), workspace_bytes.status};
    std::span<std::byte> workspace;
    auto result = uninitialized_with_workspace_<out_tensor_t>({queries.extent(0), queries.extent(1), queries.extent(2)},
                                                              workspace_bytes.value, alloc, workspace);
    if (!result) return result;
    if (status_t status = attention_packed<value_type_>(queries, key_value_packed, result.value.span(), scale,
                                                        keys_before, keys_after, workspace, capabilities, stream);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** NeoX split-half RoPE of a @b [rows,channels] matrix into a matching span, which may alias it,
 *  channels being @p head_count times an even @p depth; @c unexpected_dimensions_k when the shapes
 *  or the tables are too small. */
template <a_numeric_dtype value_type_>
status_t attention_rope(matrix_view<value_type_> x, vector_view<f32_t> cos, vector_view<f32_t> sin,
                        matrix_span<value_type_> y, std::size_t head_count, std::size_t depth,
                        nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1)) return status_t::unexpected_dimensions_k;
    if (x.stride_bytes(1) != sizeof(value_type_) || x.stride_bytes(0) < 0 || y.stride_bytes(1) != sizeof(value_type_) ||
        y.stride_bytes(0) < 0)
        return status_t::unexpected_dimensions_k;
    if (x.extent(1) < head_count * depth) return status_t::unexpected_dimensions_k;
    if (cos.size() < x.extent(0) * depth / 2 || sin.size() < x.extent(0) * depth / 2)
        return status_t::unexpected_dimensions_k;
    return numkong::attention_rope<value_type_>(x.data(), cos.data(), sin.data(), y.data(), x.extent(0), head_count,
                                                depth, static_cast<std::size_t>(x.stride_bytes(0)),
                                                static_cast<std::size_t>(y.stride_bytes(0)), capabilities, stream);
}

#pragma endregion Attention Views

} // namespace ashvardanian::numkong

#endif // NUMKONG_ATTENTION_HPP
