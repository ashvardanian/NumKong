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
 *  nk::status_t status = nk::attention_causal_packed<nk::bf16_t>(queries, packed, output, scale);
 *  auto [fresh, fresh_status] = nk::attention_bidirectional_packed<nk::bf16_t>(queries, packed, scale);
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
 *  @brief Ragged bidirectional scaled-dot-product attention against a pre-packed KV-cache.
 *  @param[in] queries Token-major matrix, one row of @p head_count × @p depth elements per token.
 *  @param[in] key_value_packed A buffer @c attention_pack filled with the same @p capabilities.
 *  @param[out] output Token-major matrix, one row of @p head_count × @p depth results per token.
 *  @param[in] query_offsets First query row of each segment, as segment count + 1 prefix sums.
 *  @param[in] queries_stride_in_bytes Row (token) stride of @p queries in bytes.
 *  @param[in] output_stride_in_bytes Row (token) stride of @p output in bytes.
 *  @param[in] scale Score multiplier, typically 1 / √depth.
 *  @param[in] task_start First task of a window over the segments × heads grid.
 *  @param[in] task_count Tasks in that window, clipped to the grid, for sharded launches.
 *  @param[in] capabilities Capabilities to pick from, or zero for the serial reference.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::attention_result_t>
status_t attention_bidirectional_packed(in_type_ const *queries, void const *key_value_packed, result_type_ *output,
                                        std::size_t head_count, std::size_t key_value_head_count, std::size_t depth,
                                        std::uint32_t const *query_offsets, std::size_t queries_stride_in_bytes,
                                        std::size_t output_stride_in_bytes, f32_t scale, std::size_t task_start = 0,
                                        std::size_t task_count = static_cast<std::size_t>(-1),
                                        nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) {
    using raw_t = typename in_type_::raw_t;
    static_assert(std::is_same_v<result_type_, typename in_type_::attention_result_t>,
                  "Attention accumulates and normalizes in F32");
    raw_t const *queries_raw = reinterpret_cast<raw_t const *>(queries);
    nk_f32_t *output_raw = reinterpret_cast<nk_f32_t *>(output);
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_attention_bidirectional_packed_bf16_best(
                queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, task_start, task_count, capabilities,
                stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_attention_bidirectional_packed_e4m3_best(
                queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, task_start, task_count, capabilities,
                stream));
        else if constexpr (std::is_same_v<in_type_, i8_t>)
            return static_cast<status_t>(nk_attention_bidirectional_packed_i8_best(
                queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, task_start, task_count, capabilities,
                stream));
    }
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_bidirectional_packed_bf16_serial(
            queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
            queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, task_start, task_count, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_attention_bidirectional_packed_e4m3_serial(
            queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
            queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, task_start, task_count, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_attention_bidirectional_packed_i8_serial(
            queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
            queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, task_start, task_count, stream));
    else return status_t::missing_kernel_k;
}

/**
 *  @brief Ragged causal, optionally sliding-window, attention against a pre-packed KV-cache.
 *  @param[in] diagonal_offset Position of query row 0: `0` for prefill, `length − query_count`
 *      against a cache.
 *  @param[in] window Keys each row sees, ending at its own position; unbounded by default.
 *
 *  Every other parameter follows the bidirectional @c attention_bidirectional_packed.
 *
 *  @tparam in_type_ Input element type (bf16_t, e4m3_t, i8_t).
 */
template <numeric_dtype in_type_, numeric_dtype result_type_ = typename in_type_::attention_result_t>
status_t attention_causal_packed(in_type_ const *queries, void const *key_value_packed, result_type_ *output,
                                 std::size_t head_count, std::size_t key_value_head_count, std::size_t depth,
                                 std::uint32_t const *query_offsets, std::size_t queries_stride_in_bytes,
                                 std::size_t output_stride_in_bytes, f32_t scale, std::int64_t diagonal_offset = 0,
                                 std::size_t window = static_cast<std::size_t>(-1), std::size_t task_start = 0,
                                 std::size_t task_count = static_cast<std::size_t>(-1),
                                 nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) {
    using raw_t = typename in_type_::raw_t;
    static_assert(std::is_same_v<result_type_, typename in_type_::attention_result_t>,
                  "Attention accumulates and normalizes in F32");
    raw_t const *queries_raw = reinterpret_cast<raw_t const *>(queries);
    nk_f32_t *output_raw = reinterpret_cast<nk_f32_t *>(output);
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_attention_causal_packed_bf16_best(
                queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, diagonal_offset, window, task_start,
                task_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_attention_causal_packed_e4m3_best(
                queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, diagonal_offset, window, task_start,
                task_count, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t>)
            return static_cast<status_t>(nk_attention_causal_packed_i8_best(
                queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
                queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, diagonal_offset, window, task_start,
                task_count, capabilities, stream));
    }
    if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(nk_attention_causal_packed_bf16_serial(
            queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
            queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, diagonal_offset, window, task_start,
            task_count, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(nk_attention_causal_packed_e4m3_serial(
            queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
            queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, diagonal_offset, window, task_start,
            task_count, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(nk_attention_causal_packed_i8_serial(
            queries_raw, key_value_packed, output_raw, head_count, key_value_head_count, depth, query_offsets,
            queries_stride_in_bytes, output_stride_in_bytes, scale.raw_, diagonal_offset, window, task_start,
            task_count, stream));
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
 *  @param[in] x_stride_bytes Row (token) stride of x in bytes
 *  @param[in] y_stride_bytes Row (token) stride of y in bytes
 *  @param[in] input_scale Scalar folded onto every loaded element (E4M3 descale; 1.0 for BF16/F32)
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Element type
 */
template <numeric_dtype in_type_>
status_t attention_rope(in_type_ const *x, f32_t const *cos, f32_t const *sin, in_type_ *y, std::size_t rows,
                        std::size_t head_count, std::size_t depth, std::size_t x_stride_bytes,
                        std::size_t y_stride_bytes, f32_t input_scale = 1.0f,
                        nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    if (depth % 2) return status_t::unexpected_dimensions_k;
    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f32_t>)
            return static_cast<status_t>(nk_attention_rope_f32_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                    head_count, depth, x_stride_bytes, y_stride_bytes,
                                                                    input_scale.raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t>)
            return static_cast<status_t>(nk_attention_rope_bf16_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                     head_count, depth, x_stride_bytes, y_stride_bytes,
                                                                     input_scale.raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t>)
            return static_cast<status_t>(nk_attention_rope_e4m3_best(&x->raw_, &cos->raw_, &sin->raw_, &y->raw_, rows,
                                                                     head_count, depth, x_stride_bytes, y_stride_bytes,
                                                                     input_scale.raw_, capabilities, stream));
    }
    // Scalar fallback for other numeric dtypes or a mask of no capability.
    std::size_t const half_depth = depth / 2;
    for (std::size_t row = 0; row < rows; ++row) {
        f32_t const *cos_row = cos + row * half_depth;
        f32_t const *sin_row = sin + row * half_depth;
        in_type_ const *x_row = reinterpret_cast<in_type_ const *>(reinterpret_cast<char const *>(x) +
                                                                   row * x_stride_bytes);
        in_type_ *y_row = reinterpret_cast<in_type_ *>(reinterpret_cast<char *>(y) + row * y_stride_bytes);
        for (std::size_t head = 0; head < head_count; ++head) {
            in_type_ const *x_base = x_row + head * depth;
            in_type_ *y_base = y_row + head * depth;
            for (std::size_t i = 0; i < half_depth; ++i) {
                float low = static_cast<float>(x_base[i]) * input_scale.raw_;
                float high = static_cast<float>(x_base[i + half_depth]) * input_scale.raw_;
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

/**
 *  @brief Which keys each causal query row sees: row @c r sits at position `r + diagonal_offset`
 *      and attends to the @c window keys ending there, inclusive.
 *
 *  The default is causal prefill with an unbounded window; `{.diagonal_offset = length - rows}`
 *  decodes against a longer cache, and `{.window = 4096}` gives sliding-window attention.
 */
struct causal_mask_t {

    /** Position of query row 0 among the keys. */
    std::int64_t diagonal_offset = 0;

    /** Keys each row sees, ending at its own position. */
    std::size_t window = std::numeric_limits<std::size_t>::max();
};

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
    if (key_value_packed.segment_offsets()[key_value_packed.segment_count()] > queries.extent(0))
        return status_t::unexpected_dimensions_k;
    return status_t::success_k;
}

/** Self-attention of @b [tokens,heads,depth] @p queries against @p key_value_packed, whose
 *  pack-time segment offsets split the query tokens too; @c unexpected_dimensions_k when the
 *  shapes disagree. The raw-pointer overload covers cross-attention and pooling. */
template <numeric_dtype value_type_, std::size_t max_rank_, typename allocator_type_>
status_t attention_bidirectional_packed(tensor_view<value_type_, max_rank_> queries,
                                        packed_attention<value_type_, allocator_type_> const &key_value_packed,
                                        tensor_span<typename value_type_::attention_result_t, max_rank_> output,
                                        f32_t scale, nk_capability_t capabilities = default_capabilities(),
                                        void *stream = nullptr) noexcept {
    if (status_t status = attention_shapes_(queries, key_value_packed, output); failed(status)) return status;
    return attention_bidirectional_packed<value_type_>(
        queries.data(), key_value_packed.data(), output.data(), queries.extent(1),
        key_value_packed.key_value_head_count(), key_value_packed.depth(), key_value_packed.segment_offsets().data(),
        static_cast<std::size_t>(queries.stride_bytes(0)), static_cast<std::size_t>(output.stride_bytes(0)), scale, 0,
        static_cast<std::size_t>(-1), capabilities, stream);
}

/** Causal self-attention, the view counterpart of the raw @c attention_causal_packed; fails like
 *  the bidirectional view overload. */
template <numeric_dtype value_type_, std::size_t max_rank_, typename allocator_type_>
status_t attention_causal_packed(tensor_view<value_type_, max_rank_> queries,
                                 packed_attention<value_type_, allocator_type_> const &key_value_packed,
                                 tensor_span<typename value_type_::attention_result_t, max_rank_> output, f32_t scale,
                                 causal_mask_t mask = {}, nk_capability_t capabilities = default_capabilities(),
                                 void *stream = nullptr) noexcept {
    if (status_t status = attention_shapes_(queries, key_value_packed, output); failed(status)) return status;
    return attention_causal_packed<value_type_>(
        queries.data(), key_value_packed.data(), output.data(), queries.extent(1),
        key_value_packed.key_value_head_count(), key_value_packed.depth(), key_value_packed.segment_offsets().data(),
        static_cast<std::size_t>(queries.stride_bytes(0)), static_cast<std::size_t>(output.stride_bytes(0)), scale,
        mask.diagonal_offset, mask.window, 0, static_cast<std::size_t>(-1), capabilities, stream);
}

/** Allocating bidirectional self-attention returning a fresh @b [tokens,heads,depth] tensor;
 *  @c unexpected_dimensions_k when the shapes disagree, or else the failure of the allocation or
 *  of the kernel itself. */
template <numeric_dtype value_type_, std::size_t max_rank_, typename packed_allocator_type_,
          typename allocator_type_ = aligned_allocator<typename value_type_::attention_result_t>>
expected<tensor<typename value_type_::attention_result_t, allocator_type_, max_rank_>> attention_bidirectional_packed(
    tensor_view<value_type_, max_rank_> queries,
    packed_attention<value_type_, packed_allocator_type_> const &key_value_packed, f32_t scale,
    allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<typename value_type_::attention_result_t, allocator_type_, max_rank_>;
    if (!attention_rows_supported_(queries)) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    auto result = out_tensor_t::uninitialized({queries.extent(0), queries.extent(1), queries.extent(2)}, alloc);
    if (!result) return result;
    if (status_t status = attention_bidirectional_packed<value_type_>(queries, key_value_packed, result.value.span(),
                                                                      scale);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** Allocating causal self-attention; fails like the allocating bidirectional overload. */
template <numeric_dtype value_type_, std::size_t max_rank_, typename packed_allocator_type_,
          typename allocator_type_ = aligned_allocator<typename value_type_::attention_result_t>>
expected<tensor<typename value_type_::attention_result_t, allocator_type_, max_rank_>> attention_causal_packed(
    tensor_view<value_type_, max_rank_> queries,
    packed_attention<value_type_, packed_allocator_type_> const &key_value_packed, f32_t scale, causal_mask_t mask = {},
    allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<typename value_type_::attention_result_t, allocator_type_, max_rank_>;
    if (!attention_rows_supported_(queries)) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    auto result = out_tensor_t::uninitialized({queries.extent(0), queries.extent(1), queries.extent(2)}, alloc);
    if (!result) return result;
    if (status_t status = attention_causal_packed<value_type_>(queries, key_value_packed, result.value.span(), scale,
                                                               mask);
        failed(status))
        return {out_tensor_t(alloc), status};
    return result;
}

/** NeoX split-half RoPE of a @b [rows,channels] matrix into a matching span, which may alias it,
 *  channels being @p head_count times an even @p depth; @c unexpected_dimensions_k when the shapes
 *  or the tables are too small. */
template <numeric_dtype value_type_>
status_t attention_rope(matrix_view<value_type_> x, vector_view<f32_t> cos, vector_view<f32_t> sin,
                        matrix_span<value_type_> y, std::size_t head_count, std::size_t depth,
                        f32_t input_scale = 1.0f) noexcept {
    if (x.extent(0) != y.extent(0) || x.extent(1) != y.extent(1)) return status_t::unexpected_dimensions_k;
    if (x.extent(1) < head_count * depth) return status_t::unexpected_dimensions_k;
    if (cos.size() < x.extent(0) * depth / 2 || sin.size() < x.extent(0) * depth / 2)
        return status_t::unexpected_dimensions_k;
    return numkong::attention_rope<value_type_>(x.data(), cos.data(), sin.data(), y.data(), x.extent(0), head_count,
                                                depth, static_cast<std::size_t>(x.stride_bytes(0)),
                                                static_cast<std::size_t>(y.stride_bytes(0)), input_scale);
}

#pragma endregion Attention Views

} // namespace ashvardanian::numkong

#endif // NUMKONG_ATTENTION_HPP
