/**
 *  @file include/numkong/reduce.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief Reduction kernels: reduce_moments (sum, sum-of-squares), reduce_minmax (min/max/indices).
 */
#ifndef NUMKONG_REDUCE_HPP
#define NUMKONG_REDUCE_HPP

#include <cstddef>     // `std::byte`, `std::size_t`
#include <cstdint>     // `std::uint32_t`
#include <memory>      // `std::allocator_traits`
#include <type_traits> // `std::is_same_v`

#include "numkong/reduce.h"

#include "numkong/types.hpp"
#include "numkong/vector.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Computes sum and sum of squares in a single pass: sum = Σ dataᵢ, sumsq = Σ dataᵢ².
 *  @param[in] data Input array.
 *  @param[in] count Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride_bytes Stride between elements in bytes, `sizeof(in_type_)` for contiguous.
 *  @param[out] sum Output sum.
 *  @param[out] sumsq Output sum of squares.
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input vector element type.
 *  @tparam sum_type_ Accumulator, defaults to @c in_type_::reduce_moments_sum_t, often widened.
 *  @tparam sumsq_type_ Sum-of-squares accumulator type, defaults to @c sum_type_.
 */
template <numeric_dtype in_type_, numeric_dtype sum_type_ = typename in_type_::reduce_moments_sum_t,
          numeric_dtype sumsq_type_ = typename in_type_::reduce_moments_sumsq_t>
status_t reduce_moments(in_type_ const *data, std::size_t count, std::size_t stride_bytes, sum_type_ *sum,
                        sumsq_type_ *sumsq, nk_capability_t capabilities = default_capabilities(),
                        void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<sum_type_, typename in_type_::reduce_moments_sum_t> &&
                              std::is_same_v<sumsq_type_, typename in_type_::reduce_moments_sumsq_t>;

    if (capabilities) {
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_f64_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_f32_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_f16_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_bf16_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                     &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_e4m3_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                     &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_e5m2_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                     &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_e2m3_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                     &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e2m1x2_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_e2m1_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                     &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_e3m2_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                     &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_i4_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                   &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_u4_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                   &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_u1_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                   &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_i8_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                   &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_u8_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                   &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i16_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_i16_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u16_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_u16_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i32_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_i32_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u32_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_u32_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, i64_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_i64_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
        else if constexpr (std::is_same_v<in_type_, u64_t> && dispatch)
            return static_cast<status_t>(nk_reduce_moments_u64_best(&data->raw_, count, stride_bytes, &sum->raw_,
                                                                    &sumsq->raw_, capabilities, stream));
    }
    // Sub-byte views yield raw FP4 codes, so pairs decode through their nibble accessors
    if constexpr (std::is_same_v<in_type_, e2m1x2_t>) {
        sum_type_ running_sum {};
        sumsq_type_ running_sumsq {};
        char const *bytes = reinterpret_cast<char const *>(data);
        for (std::size_t i = 0; i < count; ++i) {
            e2m1x2_t const pair = *reinterpret_cast<e2m1x2_t const *>(bytes + (i / 2) * stride_bytes);
            float const value = (i & 1) ? pair.second() : pair.first();
            running_sum = saturating_add(running_sum, sum_type_(value));
            running_sumsq = saturating_add(running_sumsq, sumsq_type_(value * value));
        }
        *sum = running_sum;
        *sumsq = running_sumsq;
    }
    // Scalar fallback
    else {
        sum_type_ running_sum {};
        sumsq_type_ running_sumsq {};
        vector_view<in_type_> values(reinterpret_cast<char const *>(data), count, stride_bytes);
        for (std::size_t i = 0; i < count; ++i) {
            auto val = values[i];
            running_sum = saturating_add(running_sum, val);
            running_sumsq = saturating_fma(val, val, running_sumsq);
        }
        *sum = running_sum;
        *sumsq = running_sumsq;
    }
    return status_t::success_k;
}

/**
 *  @brief Find minimum and maximum elements with their indices in a single pass.
 *  @param[in] data Input array
 *  @param[in] count Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] min_value Output minimum value
 *  @param[out] min_index Output index of minimum value, @c NUMKONG_SIZE_MAX if every value is NaN
 *  @param[out] max_value Output maximum value
 *  @param[out] max_index Output index of maximum value, @c NUMKONG_SIZE_MAX if every value is NaN
 *  @param[in] capabilities Capabilities to pick from, or zero for the C++ template
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam minmax_type_ Result type for min/max values, defaults to
 *      @c in_type_::reduce_minmax_value_t
 */
template <numeric_dtype in_type_, numeric_dtype minmax_type_ = typename in_type_::reduce_minmax_value_t>
status_t reduce_minmax(in_type_ const *data, std::size_t count, std::size_t stride_bytes, minmax_type_ *min_value,
                       std::size_t *min_index, minmax_type_ *max_value, std::size_t *max_index,
                       nk_capability_t capabilities = default_capabilities(), void *stream = nullptr) noexcept {
    constexpr bool dispatch = std::is_same_v<minmax_type_, typename in_type_::reduce_minmax_value_t>;
    static_assert(sizeof(std::size_t) == sizeof(nk_size_t), "std::size_t and nk_size_t must have the same width");
    nk_size_t min_offset = NUMKONG_SIZE_MAX, max_offset = NUMKONG_SIZE_MAX;
    nk_status_t status = nk_success_k;
    bool dispatched = false;

    // For types where minmax_type_ matches the C function output type directly,
    // dispatch to the C kernel and pass raw pointers through.
    if (capabilities) {
        dispatched = true;
        if constexpr (std::is_same_v<in_type_, f64_t> && dispatch)
            status = nk_reduce_minmax_f64_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, f32_t> && dispatch)
            status = nk_reduce_minmax_f32_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, i8_t> && dispatch)
            status = nk_reduce_minmax_i8_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                              &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, u8_t> && dispatch)
            status = nk_reduce_minmax_u8_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                              &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, i16_t> && dispatch)
            status = nk_reduce_minmax_i16_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, u16_t> && dispatch)
            status = nk_reduce_minmax_u16_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, i32_t> && dispatch)
            status = nk_reduce_minmax_i32_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, u32_t> && dispatch)
            status = nk_reduce_minmax_u32_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, i64_t> && dispatch)
            status = nk_reduce_minmax_i64_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, u64_t> && dispatch)
            status = nk_reduce_minmax_u64_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, e2m3_t> && dispatch)
            status = nk_reduce_minmax_e2m3_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                                &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, e3m2_t> && dispatch)
            status = nk_reduce_minmax_e3m2_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                                &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, f16_t> && dispatch)
            status = nk_reduce_minmax_f16_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                               &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, bf16_t> && dispatch)
            status = nk_reduce_minmax_bf16_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                                &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, e4m3_t> && dispatch)
            status = nk_reduce_minmax_e4m3_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                                &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, e5m2_t> && dispatch)
            status = nk_reduce_minmax_e5m2_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                                &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, i4x2_t> && dispatch)
            status = nk_reduce_minmax_i4_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                              &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, u4x2_t> && dispatch)
            status = nk_reduce_minmax_u4_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                              &max_value->raw_, &max_offset, capabilities, stream);
        else if constexpr (std::is_same_v<in_type_, u1x8_t> && dispatch)
            status = nk_reduce_minmax_u1_best(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset,
                                              &max_value->raw_, &max_offset, capabilities, stream);
        else dispatched = false;
    }
    // Scalar fallback, where the first value that isn't NaN takes both sides
    if (!dispatched) {
        minmax_type_ best_min = finite_max<minmax_type_>(), best_max = finite_min<minmax_type_>();
        if constexpr (infinity_capable_dtype<minmax_type_>)
            best_min = minmax_type_::positive_infinity(), best_max = minmax_type_::negative_infinity();
        vector_view<in_type_> values(reinterpret_cast<char const *>(data), count, stride_bytes);
        for (nk_size_t i = 0; i < count; ++i) {
            minmax_type_ v = minmax_type_(values[i]);
            if (is_nan(v)) continue;
            if (min_offset == NUMKONG_SIZE_MAX || v < best_min) best_min = v, min_offset = i;
            if (max_offset == NUMKONG_SIZE_MAX || v > best_max) best_max = v, max_offset = i;
        }
        *min_value = best_min, *max_value = best_max;
    }
    if (status != nk_success_k) return static_cast<status_t>(status);
    if (min_index) *min_index = static_cast<std::size_t>(min_offset);
    if (max_index) *max_index = static_cast<std::size_t>(max_offset);
    return status_t::success_k;
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Tensor Reduction Helpers

/** Result of detecting how many trailing dimensions form a single arithmetic progression:
 *  @c tail_dims collapsible trailing dimensions, @c element_count product of collapsed extents,
 *  @c stride_bytes absolute stride of the innermost collapsed dimension. */
struct uniform_stride_tail_result_t_ {
    std::size_t tail_dims;
    std::size_t element_count;
    std::size_t stride_bytes;
};

/**
 *  @brief Detects trailing dimensions where `stride[i] == stride[i+1] * extent[i+1]`.
 *
 *  When this holds, the tail is a single strided sequence and can be passed to a SIMD kernel in one
 *  call with @c element_count and @c stride_bytes.
 */
template <typename value_type_, std::size_t max_rank_>
uniform_stride_tail_result_t_ uniform_stride_tail_(tensor_view<value_type_, max_rank_> input) noexcept {
    if constexpr (dimensions_per_value<value_type_>() > 1) return {0, 0, 0};
    auto rank = input.rank();
    if (rank == 0) return {0, 1, sizeof(value_type_)};
    std::size_t tail = 1;
    auto innermost_stride = input.stride_bytes(rank - 1);
    auto expected_stride = innermost_stride;
    for (std::size_t i = rank - 1; i > 0; --i) {
        expected_stride *= static_cast<std::ptrdiff_t>(input.extent(i));
        if (input.stride_bytes(i - 1) != expected_stride) break;
        ++tail;
    }
    std::size_t count = 1;
    for (std::size_t i = rank - tail; i < rank; ++i) count *= input.extent(i);
    return {tail, count, static_cast<std::size_t>(innermost_stride < 0 ? -innermost_stride : innermost_stride)};
}

/** Collapses trailing `tail.tail_dims` dimensions into one, preserving outer dims and strides. */
template <typename value_type_, std::size_t max_rank_>
tensor_view<value_type_, max_rank_> collapse_uniform_tail_(tensor_view<value_type_, max_rank_> input,
                                                           uniform_stride_tail_result_t_ const &tail) noexcept {
    shape_storage_<max_rank_> s;
    s.rank = input.rank() - tail.tail_dims + 1;
    for (std::size_t i = 0; i + tail.tail_dims < input.rank(); ++i) {
        s.extents[i] = input.extent(i);
        s.strides[i] = input.stride_bytes(i);
    }
    s.extents[s.rank - 1] = tail.element_count;
    s.strides[s.rank - 1] = input.stride_bytes(input.rank() - 1);
    return {input.byte_data(), s};
}

/** Normalize a fully-collapsed tail for SIMD kernel consumption, handling negative strides. */
template <typename value_type_, std::size_t max_rank_>
normalized_rank1_lane_<value_type_, max_rank_> normalize_rank1_lane_from_tail_(
    tensor_view<value_type_, max_rank_> input, uniform_stride_tail_result_t_ const &tail) noexcept {
    normalized_rank1_lane_<value_type_, max_rank_> lane;
    lane.count = tail.element_count;
    lane.stride_bytes = tail.stride_bytes;
    auto innermost_stride = input.stride_bytes(input.rank() - 1);
    if (innermost_stride >= 0) {
        lane.data = input.data();
        lane.reversed = false;
    }
    else {
        lane.data = reinterpret_cast<value_type_ const *>(
            input.byte_data() + static_cast<std::ptrdiff_t>(lane.count - 1) * innermost_stride);
        lane.reversed = true;
    }
    return lane;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_rank1_moments_(tensor_view<value_type_, max_rank_> input,
                               typename value_type_::reduce_moments_sum_t &sum,
                               typename value_type_::reduce_moments_sumsq_t &sumsq) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    if (input.rank() != 1 || input.byte_data() == nullptr) return status_t::unexpected_dimensions_k;
    if (!tensor_layout_supported_(input)) return status_t::misaligned_k;
    if (can_reduce_rank1_with_kernel_(input)) {
        auto lane = normalize_rank1_lane_(input);
        return numkong::reduce_moments<value_type_>(lane.data, lane.count, lane.stride_bytes, &sum, &sumsq);
    }
    auto values = input.as_vector();
    sum = sum_t {};
    sumsq = sumsq_t {};
    for (std::size_t i = 0; i < values.size(); ++i) {
        auto value = values[i];
        sum = saturating_add(sum, value);
        sumsq = saturating_fma(value, value, sumsq);
    }
    return status_t::success_k;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_rank1_minmax_(tensor_view<value_type_, max_rank_> input,
                              minmax_result<typename value_type_::reduce_minmax_value_t> &result) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    if (input.rank() != 1 || input.byte_data() == nullptr) return status_t::unexpected_dimensions_k;
    if (!tensor_layout_supported_(input)) return status_t::misaligned_k;
    if (can_reduce_rank1_with_kernel_(input)) {
        auto lane = normalize_rank1_lane_(input);
        if (status_t status = numkong::reduce_minmax<value_type_>(lane.data, lane.count, lane.stride_bytes,
                                                                  &result.min_value, &result.min_index,
                                                                  &result.max_value, &result.max_index);
            failed(status))
            return status;
        if (lane.reversed) {
            result.min_index = lane.count - 1 - result.min_index;
            result.max_index = lane.count - 1 - result.max_index;
        }
        return status_t::success_k;
    }
    auto values = input.as_vector();
    return numkong::reduce_minmax<value_type_, minmax_t>(
        values.data(), values.size(), static_cast<std::size_t>(values.stride_bytes()), &result.min_value,
        &result.min_index, &result.max_value, &result.max_index, 0);
}

template <numeric_dtype value_type_, std::size_t max_rank_>
status_t accumulate_moments_tensor_(
    tensor_view<value_type_, max_rank_> input, tensor_span<typename value_type_::reduce_moments_sum_t, max_rank_> sums,
    tensor_span<typename value_type_::reduce_moments_sumsq_t, max_rank_> sumsqs) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    if (!tensor_layout_supported_(input)) return status_t::misaligned_k;
    if (!shapes_match_out_(input, sums) || !shapes_match_out_(input, sumsqs)) return status_t::unexpected_dimensions_k;
    if (input.rank() == 1) {
        auto src = input.as_vector();
        auto dst_sum = sums.as_vector();
        auto dst_sumsq = sumsqs.as_vector();
        for (std::size_t i = 0; i < src.size(); ++i) {
            auto value = src[i];
            dst_sum[i] = saturating_add(dst_sum[i], sum_t(value));
            dst_sumsq[i] = saturating_fma(value, value, sumsq_t(dst_sumsq[i]));
        }
        return status_t::success_k;
    }
    for (std::size_t i = 0; i < input.extent(0); ++i)
        if (status_t status = accumulate_moments_tensor_(input.slice_leading(i), sums.slice_leading(i),
                                                         sumsqs.slice_leading(i));
            failed(status))
            return status;
    return status_t::success_k;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
status_t update_minmax_tensor_(tensor_view<value_type_, max_rank_> input,
                               tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> mins,
                               tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> maxs) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    if (!tensor_layout_supported_(input)) return status_t::misaligned_k;
    if (!shapes_match_out_(input, mins) || !shapes_match_out_(input, maxs)) return status_t::unexpected_dimensions_k;
    if (input.rank() == 1) {
        auto src = input.as_vector();
        auto dst_min = mins.as_vector();
        auto dst_max = maxs.as_vector();
        for (std::size_t i = 0; i < src.size(); ++i) {
            minmax_t value = minmax_t(src[i]);
            if (value < dst_min[i]) dst_min[i] = value;
            if (value > dst_max[i]) dst_max[i] = value;
        }
        return status_t::success_k;
    }
    for (std::size_t i = 0; i < input.extent(0); ++i)
        if (status_t status = update_minmax_tensor_(input.slice_leading(i), mins.slice_leading(i),
                                                    maxs.slice_leading(i));
            failed(status))
            return status;
    return status_t::success_k;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_moments_axis_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                              typename value_type_::reduce_moments_sum_t *sums,
                              typename value_type_::reduce_moments_sumsq_t *sumsqs) noexcept {
    return for_each_axis_lane_(input, axis,
                               [&](tensor_view<value_type_, max_rank_> lane, std::size_t output_index) noexcept {
                                   typename value_type_::reduce_moments_sum_t sum {};
                                   typename value_type_::reduce_moments_sumsq_t sumsq {};
                                   if (status_t status = reduce_rank1_moments_(lane, sum, sumsq); failed(status))
                                       return status;
                                   if (sums) sums[output_index] = sum;
                                   if (sumsqs) sumsqs[output_index] = sumsq;
                                   return status_t::success_k;
                               });
}

template <numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_moments_axis_packed_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                                     tensor_span<typename value_type_::reduce_moments_sum_t, max_rank_> sums,
                                     tensor_span<typename value_type_::reduce_moments_sumsq_t, max_rank_> sumsqs,
                                     keep_dims_t keep_dims) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    if (!tensor_layout_supported_(input)) return status_t::misaligned_k;
    if (axis >= input.rank()) return status_t::unexpected_dimensions_k;
    if (axis == 0) {
        auto sum_target = keep_dims ? sums.slice_leading(0) : sums;
        auto sumsq_target = keep_dims ? sumsqs.slice_leading(0) : sumsqs;
        if (input.rank() == 1) {
            sum_t sum {};
            sumsq_t sumsq {};
            if (status_t status = reduce_rank1_moments_(input, sum, sumsq); failed(status)) return status;
            sum_target.scalar_ref() = sum;
            sumsq_target.scalar_ref() = sumsq;
            return status_t::success_k;
        }
        for (std::size_t i = 0; i < input.extent(0); ++i)
            if (status_t status = accumulate_moments_tensor_(input.slice_leading(i), sum_target, sumsq_target);
                failed(status))
                return status;
        return status_t::success_k;
    }
    if (input.rank() == 1) return status_t::unexpected_dimensions_k;
    for (std::size_t i = 0; i < input.extent(0); ++i)
        if (status_t status = reduce_moments_axis_packed_(input.slice_leading(i), axis - 1, sums.slice_leading(i),
                                                          sumsqs.slice_leading(i), keep_dims);
            failed(status))
            return status;
    return status_t::success_k;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_minmax_axis_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                             typename value_type_::reduce_minmax_value_t *mins, std::size_t *argmins,
                             typename value_type_::reduce_minmax_value_t *maxs, std::size_t *argmaxs) noexcept {
    return for_each_axis_lane_(input, axis,
                               [&](tensor_view<value_type_, max_rank_> lane, std::size_t output_index) noexcept {
                                   minmax_result<typename value_type_::reduce_minmax_value_t> result {};
                                   if (status_t status = reduce_rank1_minmax_(lane, result); failed(status))
                                       return status;
                                   if (mins) mins[output_index] = result.min_value;
                                   if (argmins) argmins[output_index] = result.min_index;
                                   if (maxs) maxs[output_index] = result.max_value;
                                   if (argmaxs) argmaxs[output_index] = result.max_index;
                                   return status_t::success_k;
                               });
}

template <numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_minmax_axis_packed_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                                    tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> mins,
                                    tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> maxs,
                                    keep_dims_t keep_dims) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    if (!tensor_layout_supported_(input)) return status_t::misaligned_k;
    if (axis >= input.rank()) return status_t::unexpected_dimensions_k;
    if (axis == 0) {
        auto min_target = keep_dims ? mins.slice_leading(0) : mins;
        auto max_target = keep_dims ? maxs.slice_leading(0) : maxs;
        if (input.rank() == 1) {
            minmax_result<minmax_t> result {};
            if (status_t status = reduce_rank1_minmax_(input, result); failed(status)) return status;
            min_target.scalar_ref() = result.min_value;
            max_target.scalar_ref() = result.max_value;
            return status_t::success_k;
        }
        for (std::size_t i = 0; i < input.extent(0); ++i)
            if (status_t status = update_minmax_tensor_(input.slice_leading(i), min_target, max_target); failed(status))
                return status;
        return status_t::success_k;
    }
    if (input.rank() == 1) return status_t::unexpected_dimensions_k;
    for (std::size_t i = 0; i < input.extent(0); ++i)
        if (status_t status = reduce_minmax_axis_packed_(input.slice_leading(i), axis - 1, mins.slice_leading(i),
                                                         maxs.slice_leading(i), keep_dims);
            failed(status))
            return status;
    return status_t::success_k;
}

#pragma endregion Tensor Reduction Helpers

#pragma region Scalar Reductions

/** Compute Σxᵢ and Σxᵢ² in a single pass. Returns zeroed result for empty tensors. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t> moments(
    tensor_view<value_type_, max_rank_> input) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    moments_result<sum_t, sumsq_t> result {};
    if (input.empty() || input.numel() == 0) return result;
    // A 0-D view is a single contiguous scalar: no axes to collapse and no innermost stride to read
    // (the rank>=1 path below would index `stride_bytes(rank() - 1)` == `stride_bytes(SIZE_MAX)`).
    if (input.rank() == 0) {
        if (failed(
                numkong::reduce_moments<value_type_>(input.data(), 1, sizeof(value_type_), &result.sum, &result.sumsq)))
            return {};
        return result;
    }
    if (!tensor_layout_supported_(input)) return result;
    auto tail = uniform_stride_tail_(input);
    if (tail.tail_dims == input.rank()) {
        auto lane = normalize_rank1_lane_from_tail_<value_type_, max_rank_>(input, tail);
        if (failed(numkong::reduce_moments<value_type_>(lane.data, lane.count, lane.stride_bytes, &result.sum,
                                                        &result.sumsq)))
            return {};
        return result;
    }
    if (tail.tail_dims >= 2) return moments<value_type_, max_rank_>(collapse_uniform_tail_(input, tail));
    // Sub-byte rank-1 fallback: uniform_stride_tail_ returns {0,0,0} for packed types.
    if (input.rank() == 1) {
        if (failed(reduce_rank1_moments_(input, result.sum, result.sumsq))) return {};
        return result;
    }
    for (std::size_t i = 0; i < input.extent(0); ++i) {
        auto slice_result = moments<value_type_, max_rank_>(input.slice_leading(static_cast<std::ptrdiff_t>(i)));
        result.sum = saturating_add(result.sum, slice_result.sum);
        result.sumsq = saturating_add(result.sumsq, slice_result.sumsq);
    }
    return result;
}

/** Find min and max values with their flat indices. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
minmax_result<typename value_type_::reduce_minmax_value_t> minmax(tensor_view<value_type_, max_rank_> input) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    minmax_result<minmax_t> result {};
    if (input.empty() || input.numel() == 0) return {{}, NUMKONG_SIZE_MAX, {}, NUMKONG_SIZE_MAX};
    // A 0-D view is a single contiguous scalar (index 0); the rank>=1 path would read
    // `stride_bytes(rank() - 1)` == `stride_bytes(SIZE_MAX)`.
    if (input.rank() == 0) {
        if (failed(numkong::reduce_minmax<value_type_>(input.data(), 1, sizeof(value_type_), &result.min_value,
                                                       &result.min_index, &result.max_value, &result.max_index)))
            return {{}, NUMKONG_SIZE_MAX, {}, NUMKONG_SIZE_MAX};
        return result;
    }
    if (!tensor_layout_supported_(input)) return result;
    auto tail = uniform_stride_tail_(input);
    if (tail.tail_dims == input.rank()) {
        auto lane = normalize_rank1_lane_from_tail_<value_type_, max_rank_>(input, tail);
        if (failed(numkong::reduce_minmax<value_type_>(lane.data, lane.count, lane.stride_bytes, &result.min_value,
                                                       &result.min_index, &result.max_value, &result.max_index)))
            return {{}, NUMKONG_SIZE_MAX, {}, NUMKONG_SIZE_MAX};
        if (lane.reversed) {
            result.min_index = tail.element_count - 1 - result.min_index;
            result.max_index = tail.element_count - 1 - result.max_index;
        }
        return result;
    }
    if (tail.tail_dims >= 2) return minmax<value_type_, max_rank_>(collapse_uniform_tail_(input, tail));
    // Sub-byte rank-1 fallback.
    if (input.rank() == 1) {
        if (failed(reduce_rank1_minmax_(input, result))) return {{}, NUMKONG_SIZE_MAX, {}, NUMKONG_SIZE_MAX};
        return result;
    }
    // Slices merge like the halves of a split kernel: an all-NaN slice has no index and never wins
    result = minmax<value_type_, max_rank_>(input.slice_leading(0));
    std::size_t base = 0;
    for (std::size_t i = 1; i < input.extent(0); ++i) {
        auto slice = input.slice_leading(static_cast<std::ptrdiff_t>(i));
        base += slice.numel();
        auto slice_result = minmax<value_type_, max_rank_>(slice);
        if (slice_result.min_index == NUMKONG_SIZE_MAX) continue;
        if (result.min_index == NUMKONG_SIZE_MAX || slice_result.min_value < result.min_value)
            result.min_value = slice_result.min_value, result.min_index = base + slice_result.min_index;
        if (result.max_index == NUMKONG_SIZE_MAX || slice_result.max_value > result.max_value)
            result.max_value = slice_result.max_value, result.max_index = base + slice_result.max_index;
    }
    return result;
}

/** Σ of all elements. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_moments_sum_t sum(tensor_view<value_type_, max_rank_> input) noexcept {
    return moments(input).sum;
}

/** Find the minimum element value. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_minmax_value_t min(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).min_value;
}

/** Find the maximum element value. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_minmax_value_t max(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).max_value;
}

/** Index of the minimum element, flat. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
std::size_t argmin(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).min_index;
}

/** Index of the maximum element, flat. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
std::size_t argmax(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).max_index;
}

/** Σxᵢ and Σxᵢ² over one run of @p in_type_; @c unexpected_dimensions_k for a strided run or one
 *  ending mid-value. */
template <numeric_dtype in_type_, numeric_dtype sum_type_ = typename in_type_::reduce_moments_sum_t,
          numeric_dtype sumsq_type_ = typename in_type_::reduce_moments_sumsq_t, vector_of<in_type_> input_type_>
expected<moments_result<sum_type_, sumsq_type_>> reduce_moments(input_type_ const &input,
                                                                nk_capability_t capabilities = default_capabilities(),
                                                                void *stream = nullptr) noexcept {
    auto values = contiguous_values_<in_type_ const>(input);
    std::size_t const dimensions = values.value.size() * dimensions_per_value<in_type_>();
    if (!values) return {{}, values.status};
    moments_result<sum_type_, sumsq_type_> result {};
    status_t status = reduce_moments<in_type_, sum_type_, sumsq_type_>(
        values.value.data(), dimensions, sizeof(in_type_), &result.sum, &result.sumsq, capabilities, stream);
    return {result, status};
}

/** Minimum and maximum, with their indices, over one run of @p in_type_; @c unexpected_dimensions_k
 *  for a strided run or one ending mid-value, and @c NUMKONG_SIZE_MAX indices when every value is
 *  NaN, which no minimum or maximum can index. */
template <numeric_dtype in_type_, numeric_dtype minmax_type_ = typename in_type_::reduce_minmax_value_t,
          vector_of<in_type_> input_type_>
expected<minmax_result<minmax_type_>> reduce_minmax(input_type_ const &input,
                                                    nk_capability_t capabilities = default_capabilities(),
                                                    void *stream = nullptr) noexcept {
    auto values = contiguous_values_<in_type_ const>(input);
    std::size_t const dimensions = values.value.size() * dimensions_per_value<in_type_>();
    if (!values) return {{}, values.status};
    minmax_result<minmax_type_> result {};
    status_t status = reduce_minmax<in_type_, minmax_type_>(values.value.data(), dimensions, sizeof(in_type_),
                                                            &result.min_value, &result.min_index, &result.max_value,
                                                            &result.max_index, capabilities, stream);
    return {result, status};
}

/** Compute Σxᵢ and Σxᵢ² over a vector view, strided ones included. */
template <numeric_dtype value_type_>
moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t> moments(
    vector_view<value_type_> input) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    moments_result<sum_t, sumsq_t> result {};
    if (input.size() == 0) return result;
    if (failed(reduce_moments<value_type_>(input.data(), input.size(), static_cast<std::size_t>(input.stride_bytes()),
                                           &result.sum, &result.sumsq)))
        return {};
    return result;
}

/** Find min and max values with their indices over a vector view, strided ones included. */
template <numeric_dtype value_type_>
minmax_result<typename value_type_::reduce_minmax_value_t> minmax(vector_view<value_type_> input) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    minmax_result<minmax_t> result {};
    if (failed(reduce_minmax<value_type_>(input.data(), input.size(), static_cast<std::size_t>(input.stride_bytes()),
                                          &result.min_value, &result.min_index, &result.max_value, &result.max_index)))
        return {{}, NUMKONG_SIZE_MAX, {}, NUMKONG_SIZE_MAX};
    return result;
}

/** Σ of all elements in a vector view. */
template <numeric_dtype value_type_>
typename value_type_::reduce_moments_sum_t sum(vector_view<value_type_> input) noexcept {
    return moments(input).sum;
}

/** Find the minimum element value in a vector view. */
template <numeric_dtype value_type_>
typename value_type_::reduce_minmax_value_t min(vector_view<value_type_> input) noexcept {
    return minmax(input).min_value;
}

/** Find the maximum element value in a vector view. */
template <numeric_dtype value_type_>
typename value_type_::reduce_minmax_value_t max(vector_view<value_type_> input) noexcept {
    return minmax(input).max_value;
}

/** Index of the minimum element in a vector view. */
template <numeric_dtype value_type_>
std::size_t argmin(vector_view<value_type_> input) noexcept {
    return minmax(input).min_index;
}

/** Index of the maximum element in a vector view. */
template <numeric_dtype value_type_>
std::size_t argmax(vector_view<value_type_> input) noexcept {
    return minmax(input).max_index;
}

/** Population count of a packed bit tensor. Wraps @c nk_reduce_moments_u1_best via @c sum;
 *  returns 0 for empty inputs. */
template <std::size_t max_rank_ = 8>
u64_t popcount(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return sum<u1x8_t, max_rank_>(input);
}

/** True if any bit of the packed bit tensor is set. Suffixed @c _set to avoid colliding with the
 *  @c all_t and @c all slice marker in `vector.hpp`. */
template <std::size_t max_rank_ = 8>
bool any_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return popcount<max_rank_>(input).raw_ != 0;
}

/** True if no bit of the packed bit tensor is set. */
template <std::size_t max_rank_ = 8>
bool none_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return !any_set<max_rank_>(input);
}

/** True if every bit of the packed bit tensor is set. */
template <std::size_t max_rank_ = 8>
bool all_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return popcount<max_rank_>(input).raw_ == input.numel();
}

/** Population count over a 1D bit-vector view. */
inline u64_t popcount(vector_view<u1x8_t> input) noexcept { return sum<u1x8_t>(input); }

/** True if any bit of the 1D bit-vector view is set. */
inline bool any_set(vector_view<u1x8_t> input) noexcept { return popcount(input).raw_ != 0; }

/** True if no bit of the 1D bit-vector view is set. */
inline bool none_set(vector_view<u1x8_t> input) noexcept { return !any_set(input); }

/** True if every bit of the 1D bit-vector view is set. */
inline bool all_set(vector_view<u1x8_t> input) noexcept { return popcount(input).raw_ == input.size(); }

#pragma endregion Scalar Reductions

#pragma region Axis Reductions

/** Σ along a single axis; empty for an empty @p input, @c unexpected_dimensions_k for an @p axis
 *  past the rank, @c misaligned_k for an unsupported layout, or the allocation's failure. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_moments_sum_t>>
expected<tensor<typename value_type_::reduce_moments_sum_t, allocator_type_, max_rank_>> sum(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
    allocator_type_ alloc = {}) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sum_tensor_t = tensor<sum_t, allocator_type_, max_rank_>;

    if (input.empty()) return {sum_tensor_t(alloc), status_t::success_k};
    if (axis >= input.rank()) return {sum_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (!tensor_layout_supported_(input)) return {sum_tensor_t(alloc), status_t::misaligned_k};

    auto out_shape = reduced_shape_<sum_t>(input.shape(), axis, keep_dims);
    auto sums = sum_tensor_t::zeros(out_shape.extents, out_shape.rank, alloc);
    if (!sums) return sums;
    status_t status;
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
        using sumsq_alloc_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<sumsq_t>;
        using sumsq_tensor_t = tensor<sumsq_t, sumsq_alloc_t, max_rank_>;
        auto scratch = sumsq_tensor_t::zeros(out_shape.extents, out_shape.rank, sumsq_alloc_t(alloc));
        if (!scratch) return {sum_tensor_t(alloc), scratch.status};
        status = reduce_moments_axis_packed_(input, axis, sums.value.span(), scratch.value.span(), keep_dims);
    }
    else status = reduce_moments_axis_(input, axis, sums.value.data(), nullptr);
    if (failed(status)) return {sum_tensor_t(alloc), status};
    return sums;
}

/** Moments along an axis, Σxᵢ and Σxᵢ² per slice; fails like the axis @c sum. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_moments_sum_t>>
expected<moments_result<tensor<typename value_type_::reduce_moments_sum_t, allocator_type_, max_rank_>,
                        tensor<typename value_type_::reduce_moments_sumsq_t,
                               typename std::allocator_traits<allocator_type_>::template rebind_alloc<
                                   typename value_type_::reduce_moments_sumsq_t>,
                               max_rank_>>>
moments(tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
        allocator_type_ alloc = {}) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    using sum_tensor_t = tensor<sum_t, allocator_type_, max_rank_>;
    using sumsq_alloc_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<sumsq_t>;
    using sumsq_tensor_t = tensor<sumsq_t, sumsq_alloc_t, max_rank_>;
    using result_t = moments_result<sum_tensor_t, sumsq_tensor_t>;
    auto empty_result = [&](status_t status) noexcept -> expected<result_t> {
        return {result_t {sum_tensor_t(alloc), sumsq_tensor_t(sumsq_alloc_t(alloc))}, status};
    };

    if (input.empty()) return empty_result(status_t::success_k);
    if (axis >= input.rank()) return empty_result(status_t::unexpected_dimensions_k);
    if (!tensor_layout_supported_(input)) return empty_result(status_t::misaligned_k);

    auto out_shape_sum = reduced_shape_<sum_t>(input.shape(), axis, keep_dims);
    auto out_shape_sq = reduced_shape_<sumsq_t>(input.shape(), axis, keep_dims);

    auto sums = sum_tensor_t::zeros(out_shape_sum.extents, out_shape_sum.rank, alloc);
    if (!sums) return empty_result(sums.status);
    auto sumsqs = sumsq_tensor_t::zeros(out_shape_sq.extents, out_shape_sq.rank, sumsq_alloc_t(alloc));
    if (!sumsqs) return empty_result(sumsqs.status);

    status_t status;
    if constexpr (dimensions_per_value<value_type_>() > 1)
        status = reduce_moments_axis_packed_(input, axis, sums.value.span(), sumsqs.value.span(), keep_dims);
    else status = reduce_moments_axis_(input, axis, sums.value.data(), sumsqs.value.data());
    if (failed(status)) return empty_result(status);

    return {result_t {std::move(sums.value), std::move(sumsqs.value)}, status_t::success_k};
}

/** Min and max along an axis; fails like the axis @c sum. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
expected<minmax_result<tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_>>> minmax(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
    allocator_type_ alloc = {}) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    using out_tensor_t = tensor<minmax_t, allocator_type_, max_rank_>;
    using result_t = minmax_result<out_tensor_t>;
    auto empty_result = [&](status_t status) noexcept -> expected<result_t> {
        return {result_t {out_tensor_t(alloc), 0, out_tensor_t(alloc), 0}, status};
    };

    if (input.empty()) return empty_result(status_t::success_k);
    if (axis >= input.rank()) return empty_result(status_t::unexpected_dimensions_k);
    if (!tensor_layout_supported_(input)) return empty_result(status_t::misaligned_k);

    auto out_shape = reduced_shape_<minmax_t>(input.shape(), axis, keep_dims);
    auto mins = out_tensor_t::full(out_shape.extents, out_shape.rank, finite_max<minmax_t>(), alloc);
    if (!mins) return empty_result(mins.status);
    auto maxs = out_tensor_t::full(out_shape.extents, out_shape.rank, finite_min<minmax_t>(), alloc);
    if (!maxs) return empty_result(maxs.status);

    status_t status;
    if constexpr (dimensions_per_value<value_type_>() > 1)
        status = reduce_minmax_axis_packed_(input, axis, mins.value.span(), maxs.value.span(), keep_dims);
    else status = reduce_minmax_axis_(input, axis, mins.value.data(), nullptr, maxs.value.data(), nullptr);
    if (failed(status)) return empty_result(status);
    return {result_t {std::move(mins.value), 0, std::move(maxs.value), 0}, status_t::success_k};
}

/** Argmin along an axis; fails like the axis @c sum, and with @c missing_kernel_k for sub-byte
 *  inputs. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<std::size_t>>
expected<tensor<std::size_t, allocator_type_, max_rank_>> argmin(tensor_view<value_type_, max_rank_> input,
                                                                 std::size_t axis,
                                                                 keep_dims_t keep_dims = collapse_dims_k,
                                                                 allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<std::size_t, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    if (axis >= input.rank()) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (!tensor_layout_supported_(input)) return {out_tensor_t(alloc), status_t::misaligned_k};
    if constexpr (dimensions_per_value<value_type_>() > 1) return {out_tensor_t(alloc), status_t::missing_kernel_k};

    auto out_shape = reduced_shape_<std::size_t>(input.shape(), axis, keep_dims);
    auto indices = out_tensor_t::zeros(out_shape.extents, out_shape.rank, alloc);
    if (!indices) return indices;
    if (status_t status = reduce_minmax_axis_(input, axis, nullptr, indices.value.data(), nullptr, nullptr);
        failed(status))
        return {out_tensor_t(alloc), status};
    return indices;
}

/** Argmax along an axis; fails like @c argmin. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<std::size_t>>
expected<tensor<std::size_t, allocator_type_, max_rank_>> argmax(tensor_view<value_type_, max_rank_> input,
                                                                 std::size_t axis,
                                                                 keep_dims_t keep_dims = collapse_dims_k,
                                                                 allocator_type_ alloc = {}) noexcept {
    using out_tensor_t = tensor<std::size_t, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    if (axis >= input.rank()) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (!tensor_layout_supported_(input)) return {out_tensor_t(alloc), status_t::misaligned_k};
    if constexpr (dimensions_per_value<value_type_>() > 1) return {out_tensor_t(alloc), status_t::missing_kernel_k};

    auto out_shape = reduced_shape_<std::size_t>(input.shape(), axis, keep_dims);
    auto indices = out_tensor_t::zeros(out_shape.extents, out_shape.rank, alloc);
    if (!indices) return indices;
    if (status_t status = reduce_minmax_axis_(input, axis, nullptr, nullptr, nullptr, indices.value.data());
        failed(status))
        return {out_tensor_t(alloc), status};
    return indices;
}

/** Min along an axis; fails like the axis @c minmax. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
expected<tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_>> min(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
    allocator_type_ alloc = {}) noexcept {
    auto result = minmax<value_type_, max_rank_, allocator_type_>(input, axis, keep_dims, alloc);
    return {std::move(result.value.min_value), result.status};
}

/** Max along an axis; fails like the axis @c minmax. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
expected<tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_>> max(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
    allocator_type_ alloc = {}) noexcept {
    auto result = minmax<value_type_, max_rank_, allocator_type_>(input, axis, keep_dims, alloc);
    return {std::move(result.value.max_value), result.status};
}

#pragma endregion Axis Reductions

} // namespace ashvardanian::numkong

#endif // NUMKONG_REDUCE_HPP
