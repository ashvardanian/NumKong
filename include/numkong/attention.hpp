/**
 *  @file include/numkong/attention.hpp
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief C++ bindings for multi-target ragged scaled-dot-product attention kernels.
 *
 *  Three layers, like the dots and maxsim bindings: raw-pointer wrappers that keep the task windows
 *  for sharded launches, the owning @c packed_attention from `matrix.hpp` with @c attention_pack,
 *  and view overloads that derive heads, depth and strides from @b [tokens,heads,depth] views.
 *
 *  @code{.cpp}
 *  auto [packed, packed_status] = nk::packed_attention<nk::bf16_t>::make(keys, values, offsets, lengths);
 *  if (nk::failed(packed_status)) return packed_status;
 *  std::size_t const all_keys = std::numeric_limits<std::size_t>::max(), causal = 0;
 *  nk::status_t status = nk::attention_packed<nk::bf16_t>(queries, packed, output, scale, all_keys, causal);
 *  auto [fresh, fresh_status] = nk::attention_packed<nk::bf16_t>(queries, packed, scale);
 *  @endcode
 */
#ifndef NUMKONG_ATTENTION_HPP
#define NUMKONG_ATTENTION_HPP

#include <cstddef>
#include <limits>
#include <type_traits>

#include "numkong/attention.h"
#include "numkong/types.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Ragged scaled-dot-product attention against a pre-packed KV-cache, under a band of keys.
 *  @param[in] query_offsets First query row of each segment, as segment count + 1 prefix sums.
 *  @param[in] query_token_count Query rows the call addresses, `query_offsets[segments]`; GPU
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
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial reference.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  Queries align to the end of each segment's keys, as @c nk_attention_packed_bf16_best describes:
 *  causal sets @p keys_after to 0, and a sliding window of @c w keys sets @p keys_before to w − 1.
 *
 *  @tparam in_type_ Input element type (bf16_t, f16_t, e4m3_t, i8_t), or a block-scaled format
 *      whose queries are an @c operand_pointer reference.
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::attention_result_t>
status_t attention_packed(std::size_t head_count, std::size_t key_value_head_count, std::size_t depth,
                          std::uint32_t const *query_offsets, std::size_t query_token_count, f32_t scale,
                          std::size_t keys_before, std::size_t keys_after, operand_pointer<in_type_> queries,
                          std::size_t queries_stride, void const *key_value_packed, result_type_ *output,
                          std::size_t output_stride, f32_t *log_sum_exp, std::size_t tasks_begin = 0,
                          std::size_t tasks_end = std::numeric_limits<std::size_t>::max(),
                          nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) {
    using raw_t = typename in_type_::raw_t;
    static_assert(std::is_same_v<result_type_, typename in_type_::attention_result_t>,
                  "Attention accumulates and normalizes in F32");
    raw_t const *queries_raw = reinterpret_cast<raw_t const *>(queries);
    nk_f32_t *output_raw = reinterpret_cast<nk_f32_t *>(output);
    nk_f32_t *log_sum_exp_raw = reinterpret_cast<nk_f32_t *>(log_sum_exp);
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_attention_packed_bf16_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries_raw, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t>)
            return static_cast<status_t>(nk_attention_packed_f16_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries_raw, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_attention_packed_e4m3_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries_raw, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t>)
            return static_cast<status_t>(nk_attention_packed_i8_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries_raw, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
            return static_cast<status_t>(nk_attention_packed_nvfp4_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
            return static_cast<status_t>(nk_attention_packed_mxfp4_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
            return static_cast<status_t>(nk_attention_packed_mxfp6e2m3_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
            return static_cast<status_t>(nk_attention_packed_mxfp6e3m2_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
            return static_cast<status_t>(nk_attention_packed_mxfp8e4m3_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
            return static_cast<status_t>(nk_attention_packed_mxfp8e5m2_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
                tasks_begin, tasks_end, capabilities, stream));
    }
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_packed_bf16_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries_raw, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(nk_attention_packed_f16_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries_raw, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_attention_packed_e4m3_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries_raw, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_attention_packed_i8_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries_raw, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_attention_packed_nvfp4_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp4_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp6e2m3_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp6e3m2_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp8e4m3_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_attention_packed_mxfp8e5m2_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_stride, log_sum_exp_raw,
            tasks_begin, tasks_end, stream));
    else return status_t::missing_kernel_k;
}

/**
 *  @brief Gradients of @c attention_packed with respect to its queries, keys and values.
 *  @param[in] output The forward's output for these queries and pack.
 *  @param[in] output_gradient Gradient of the loss with respect to @p output, laid out like it.
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
 *  Every other parameter follows @c attention_packed; the band must be the forward's.
 *
 *  @tparam in_type_ Input element type, bf16_t, or a block-scaled format whose queries are an
 *      @c operand_pointer reference.
 */
template <numeric_dtype in_type_>
status_t attention_packed_gradients(std::size_t head_count, std::size_t key_value_head_count, std::size_t depth,
                                    std::uint32_t const *query_offsets, std::size_t query_token_count, f32_t scale,
                                    std::size_t keys_before, std::size_t keys_after, operand_pointer<in_type_> queries,
                                    std::size_t queries_stride, void const *key_value_packed, f32_t const *output,
                                    f32_t const *output_gradient, std::size_t output_stride, f32_t const *log_sum_exp,
                                    f32_t *query_gradient, std::size_t query_gradient_stride, f32_t *key_gradient,
                                    f32_t *value_gradient, std::size_t key_value_gradient_stride,
                                    std::size_t tasks_begin = 0,
                                    std::size_t tasks_end = std::numeric_limits<std::size_t>::max(),
                                    nk_capability_t capabilities = default_capabilities(),
                                    nk_stream_t stream = nullptr) {
    auto const *output_raw = reinterpret_cast<nk_f32_t const *>(output);
    auto const *output_gradient_raw = reinterpret_cast<nk_f32_t const *>(output_gradient);
    auto const *log_sum_exp_raw = reinterpret_cast<nk_f32_t const *>(log_sum_exp);
    auto *query_gradient_raw = reinterpret_cast<nk_f32_t *>(query_gradient);
    auto *key_gradient_raw = reinterpret_cast<nk_f32_t *>(key_gradient);
    auto *value_gradient_raw = reinterpret_cast<nk_f32_t *>(value_gradient);
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_attention_packed_gradients_bf16_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, reinterpret_cast<nk_bf16_t const *>(queries), queries_stride, key_value_packed, output_raw,
                output_gradient_raw, output_stride, log_sum_exp_raw, query_gradient_raw, query_gradient_stride,
                key_gradient_raw, value_gradient_raw, key_value_gradient_stride, tasks_begin, tasks_end, capabilities,
                stream));
        else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
            return static_cast<status_t>(nk_attention_packed_gradients_nvfp4_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
                log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
                key_value_gradient_stride, tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
            return static_cast<status_t>(nk_attention_packed_gradients_mxfp4_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
                log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
                key_value_gradient_stride, tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
            return static_cast<status_t>(nk_attention_packed_gradients_mxfp6e2m3_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
                log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
                key_value_gradient_stride, tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
            return static_cast<status_t>(nk_attention_packed_gradients_mxfp6e3m2_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
                log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
                key_value_gradient_stride, tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
            return static_cast<status_t>(nk_attention_packed_gradients_mxfp8e4m3_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
                log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
                key_value_gradient_stride, tasks_begin, tasks_end, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
            return static_cast<status_t>(nk_attention_packed_gradients_mxfp8e5m2_best(
                head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
                keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
                log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
                key_value_gradient_stride, tasks_begin, tasks_end, capabilities, stream));
    }
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_bf16_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, reinterpret_cast<nk_bf16_t const *>(queries), queries_stride, key_value_packed, output_raw,
            output_gradient_raw, output_stride, log_sum_exp_raw, query_gradient_raw, query_gradient_stride,
            key_gradient_raw, value_gradient_raw, key_value_gradient_stride, tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, nvfp4_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_nvfp4_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp4_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp4_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e2m3_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp6e2m3_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp6e3m2_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp6e3m2_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e4m3_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp8e4m3_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, stream));
    else if constexpr (std::is_same_v<in_type_, mxfp8e5m2_t>)
        return static_cast<status_t>(nk_attention_packed_gradients_mxfp8e5m2_serial(
            head_count, key_value_head_count, depth, query_offsets, query_token_count, scale.raw_, keys_before,
            keys_after, queries, queries_stride, key_value_packed, output_raw, output_gradient_raw, output_stride,
            log_sum_exp_raw, query_gradient_raw, query_gradient_stride, key_gradient_raw, value_gradient_raw,
            key_value_gradient_stride, tasks_begin, tasks_end, stream));
    else return status_t::missing_kernel_k;
}

/**
 *  @brief NeoX split-half RoPE: rotates channel pairs of every head by per-token angles.
 *
 *  Rotates each pair `(i, i + @p depth / 2)` of every head by the `[rows, @p depth / 2]` grids.
 *
 *  @param[in] x Row-major matrix of shape @b [rows,channels]; channels = @p head_count · @p depth
 *  @param[in] cos,sin `[rows, @p depth / 2]` per-token angle grids, shared across heads
 *  @param[out] y Output, same shape and dtype as x; may alias x for in-place rotation
 *  @param[in] rows Token count
 *  @param[in] head_count Heads per token
 *  @param[in] depth Even channel count per head; channel @c i pairs with `i + depth / 2`
 *  @param[in] x_stride Row (token) stride of x in bytes
 *  @param[in] y_stride Row (token) stride of y in bytes
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 */
template <numeric_dtype in_type_>
status_t attention_rope(in_type_ const *x, f32_t const *cos, f32_t const *sin, in_type_ *y, std::size_t rows,
                        std::size_t head_count, std::size_t depth, std::size_t x_stride, std::size_t y_stride,
                        nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if (depth % 2) return status_t::unexpected_dimensions_k;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f32_t>)
            return static_cast<status_t>(nk_attention_rope_f32_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                    head_count, depth, x_stride, y_stride, capabilities,
                                                                    stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_attention_rope_bf16_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                     head_count, depth, x_stride, y_stride,
                                                                     capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_attention_rope_e4m3_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                     head_count, depth, x_stride, y_stride,
                                                                     capabilities, stream));
    }
    // Scalar fallback for other numeric dtypes or a mask of no capability.
    std::size_t const half_depth = depth / 2;
    for (std::size_t row = 0; row < rows; ++row) {
        f32_t const *cos_row = cos + row * half_depth;
        f32_t const *sin_row = sin + row * half_depth;
        in_type_ const *x_row = reinterpret_cast<in_type_ const *>(reinterpret_cast<char const *>(x) + row * x_stride);
        in_type_ *y_row = reinterpret_cast<in_type_ *>(reinterpret_cast<char *>(y) + row * y_stride);
        for (std::size_t head = 0; head < head_count; ++head) {
            in_type_ const *x_base = x_row + head * depth;
            in_type_ *y_base = y_row + head * depth;
            for (std::size_t i = 0; i < half_depth; ++i) {
                float low = static_cast<float>(x_base[i]);
                float high = static_cast<float>(x_base[i + half_depth]);
                float cosine = static_cast<float>(cos_row[i]), sine = static_cast<float>(sin_row[i]);
                y_base[i] = f32_t(low * cosine - high * sine).template to<in_type_>();
                y_base[i + half_depth] = f32_t(low * sine + high * cosine).template to<in_type_>();
            }
        }
    }
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/matrix.hpp"

namespace ashvardanian::numkong {

#pragma region Attention Views

/** Checks that @p queries and @p output are matching @b [tokens,heads,depth] layouts over the
 *  @p key_value_packed depth, whose head count divides theirs, covering every packed segment. */
template <numeric_dtype value_type_, std::size_t max_rank_, typename allocator_type_, typename output_type_>
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
    if (key_value_packed.key_offsets()[key_value_packed.segment_count()] > queries.extent(0))
        return status_t::unexpected_dimensions_k;
    return status_t::success_k;
}

/** Self-attention of @b [tokens,heads,depth] @p queries against @p key_value_packed, whose
 *  pack-time key offsets split the query tokens too, under the @p keys_before, @p keys_after
 *  band; @c unexpected_dimensions_k when the shapes disagree. The raw-pointer overload covers
 *  cross-attention and pooling. */
template <numeric_dtype value_type_, std::size_t max_rank_, typename allocator_type_>
status_t attention_packed(tensor_view<value_type_, max_rank_> queries,
                          packed_attention<value_type_, allocator_type_> const &key_value_packed,
                          tensor_span<typename value_type_::attention_result_t, max_rank_> output, f32_t scale,
                          std::size_t keys_before = std::numeric_limits<std::size_t>::max(),
                          std::size_t keys_after = std::numeric_limits<std::size_t>::max(),
                          nk_capability_t capabilities = default_capabilities(),
                          nk_stream_t stream = nullptr) noexcept {
    if (status_t status = attention_shapes_(queries, key_value_packed, output); failed(status)) return status;
    return attention_packed<value_type_>(
        queries.extent(1), key_value_packed.key_value_head_count(), key_value_packed.depth(),
        key_value_packed.key_offsets().data(), key_value_packed.key_offsets()[key_value_packed.segment_count()], scale,
        keys_before, keys_after, queries.data(), static_cast<std::size_t>(queries.stride_bytes(0)),
        key_value_packed.data(), output.data(), static_cast<std::size_t>(output.stride_bytes(0)), nullptr, 0,
        std::numeric_limits<std::size_t>::max(), capabilities, stream);
}

/** Allocating self-attention returning a fresh @b [tokens,heads,depth] tensor;
 *  @c unexpected_dimensions_k when the shapes disagree, or else the failure of the allocation or
 *  of the kernel itself. */
template <numeric_dtype value_type_, std::size_t max_rank_, typename packed_allocator_type_,
          typename allocator_type_ = aligned_allocator<typename value_type_::attention_result_t>>
expected<tensor<typename value_type_::attention_result_t, allocator_type_, max_rank_>> attention_packed(
    tensor_view<value_type_, max_rank_> queries,
    packed_attention<value_type_, packed_allocator_type_> const &key_value_packed, f32_t scale,
    std::size_t keys_before = std::numeric_limits<std::size_t>::max(),
    std::size_t keys_after = std::numeric_limits<std::size_t>::max(), allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<typename value_type_::attention_result_t, allocator_type_, max_rank_>;
    if (!attention_rows_supported_(queries)) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    auto result = out_tensor_t::uninitialized({queries.extent(0), queries.extent(1), queries.extent(2)}, alloc);
    if (!result) return result;
    if (status_t status = attention_packed<value_type_>(queries, key_value_packed, result.value.span(), scale,
                                                        keys_before, keys_after);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** NeoX split-half RoPE of a @b [rows,channels] matrix into a matching span, which may alias it,
 *  channels being @p head_count times an even @p depth; @c unexpected_dimensions_k when the shapes
 *  or the tables are too small. */
template <numeric_dtype value_type_>
status_t attention_rope(matrix_view<value_type_> x, vector_view<f32_t> cos, vector_view<f32_t> sin,
                        matrix_span<value_type_> y, std::size_t head_count, std::size_t depth) noexcept {
    if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1)) return status_t::unexpected_dimensions_k;
    if (x.extent(1) < head_count * depth) return status_t::unexpected_dimensions_k;
    if (cos.size() < x.extent(0) * depth / 2 || sin.size() < x.extent(0) * depth / 2)
        return status_t::unexpected_dimensions_k;
    return numkong::attention_rope<value_type_>(x.data(), cos.data(), sin.data(), y.data(), x.extent(0), head_count,
                                                depth, static_cast<std::size_t>(x.stride_bytes(0)),
                                                static_cast<std::size_t>(y.stride_bytes(0)));
}

#pragma endregion Attention Views

} // namespace ashvardanian::numkong

#endif // NUMKONG_ATTENTION_HPP
