/**
 *  @brief Reduction kernels: reduce_moments (sum + sum-of-squares), reduce_minmax (min + max with indices).
 *  @file include/numkong/reduce.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 */
#ifndef NK_REDUCE_HPP
#define NK_REDUCE_HPP

#include <cstddef>     // `std::byte`, `std::size_t`
#include <cstdint>     // `std::uint32_t`
#include <memory>      // `std::allocator_traits`
#include <type_traits> // `std::is_same_v`

#include "numkong/reduce.h"

#include "numkong/types.hpp"
#include "numkong/vector.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Compute sum and sum-of-squares in a single pass: sum = Sigma data_i, sumsq = Sigma data_i^2
 *  @param[in] data Input array
 *  @param[in] count Number of elements
 *  @param[in] stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] sum Output sum
 *  @param[out] sumsq Output sum of squares
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam sum_type_ Sum accumulator type, defaults to `in_type_::reduce_moments_sum_t` (often widened)
 *  @tparam sumsq_type_ Sum-of-squares accumulator type, defaults to `sum_type_`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype sum_type_ = typename in_type_::reduce_moments_sum_t,
          numeric_dtype sumsq_type_ = typename in_type_::reduce_moments_sumsq_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_moments(in_type_ const *data, std::size_t count, std::size_t stride_bytes, sum_type_ *sum,
                    sumsq_type_ *sumsq) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<sum_type_, typename in_type_::reduce_moments_sum_t> &&
                          std::is_same_v<sumsq_type_, typename in_type_::reduce_moments_sumsq_t>;

    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_reduce_moments_f64(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_reduce_moments_f32(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_reduce_moments_f16(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_reduce_moments_bf16(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_reduce_moments_e4m3(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_reduce_moments_e5m2(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_reduce_moments_e2m3(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_reduce_moments_e3m2(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_reduce_moments_i4(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_reduce_moments_u4(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && simd)
        nk_reduce_moments_u1(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_reduce_moments_i8(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_reduce_moments_u8(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_reduce_moments_i16(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_reduce_moments_u16(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_reduce_moments_i32(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_reduce_moments_u32(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_reduce_moments_i64(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_reduce_moments_u64(&data->raw_, count, stride_bytes, &sum->raw_, &sumsq->raw_);
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
}

/**
 *  @brief Find minimum and maximum elements with their indices in a single pass.
 *  @param[in] data Input array
 *  @param[in] count Number of elements
 *  @param[in] stride_bytes Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] min_value Output minimum value
 *  @param[out] min_index Output index of minimum value
 *  @param[out] max_value Output maximum value
 *  @param[out] max_index Output index of maximum value
 *
 *  @tparam in_type_ Input vector element type
 *  @tparam minmax_type_ Result type for min/max values, defaults to `in_type_::reduce_minmax_value_t`
 *  @tparam allow_simd_ Enable SIMD kernel dispatch when `prefer_simd_k`
 */
template <numeric_dtype in_type_, numeric_dtype minmax_type_ = typename in_type_::reduce_minmax_value_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_minmax(in_type_ const *data, std::size_t count, std::size_t stride_bytes, minmax_type_ *min_value,
                   std::size_t *min_index, minmax_type_ *max_value, std::size_t *max_index) noexcept {
    constexpr bool simd = allow_simd_ == prefer_simd_k &&
                          std::is_same_v<minmax_type_, typename in_type_::reduce_minmax_value_t>;
    static_assert(sizeof(std::size_t) == sizeof(nk_size_t), "size_t and nk_size_t must have the same width");
    nk_size_t min_offset = 0, max_offset = 0;

    // For types where minmax_type_ matches the C function output type directly,
    // dispatch to the C kernel and pass raw pointers through.
    if constexpr (std::is_same_v<in_type_, f64_t> && simd)
        nk_reduce_minmax_f64(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, f32_t> && simd)
        nk_reduce_minmax_f32(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, i8_t> && simd)
        nk_reduce_minmax_i8(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                            &max_offset);
    else if constexpr (std::is_same_v<in_type_, u8_t> && simd)
        nk_reduce_minmax_u8(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                            &max_offset);
    else if constexpr (std::is_same_v<in_type_, i16_t> && simd)
        nk_reduce_minmax_i16(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, u16_t> && simd)
        nk_reduce_minmax_u16(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, i32_t> && simd)
        nk_reduce_minmax_i32(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, u32_t> && simd)
        nk_reduce_minmax_u32(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, i64_t> && simd)
        nk_reduce_minmax_i64(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, u64_t> && simd)
        nk_reduce_minmax_u64(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, e2m3_t> && simd)
        nk_reduce_minmax_e2m3(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                              &max_offset);
    else if constexpr (std::is_same_v<in_type_, e3m2_t> && simd)
        nk_reduce_minmax_e3m2(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                              &max_offset);
    else if constexpr (std::is_same_v<in_type_, f16_t> && simd)
        nk_reduce_minmax_f16(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                             &max_offset);
    else if constexpr (std::is_same_v<in_type_, bf16_t> && simd)
        nk_reduce_minmax_bf16(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                              &max_offset);
    else if constexpr (std::is_same_v<in_type_, e4m3_t> && simd)
        nk_reduce_minmax_e4m3(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                              &max_offset);
    else if constexpr (std::is_same_v<in_type_, e5m2_t> && simd)
        nk_reduce_minmax_e5m2(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                              &max_offset);
    else if constexpr (std::is_same_v<in_type_, i4x2_t> && simd)
        nk_reduce_minmax_i4(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                            &max_offset);
    else if constexpr (std::is_same_v<in_type_, u4x2_t> && simd)
        nk_reduce_minmax_u4(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                            &max_offset);
    else if constexpr (std::is_same_v<in_type_, u1x8_t> && simd)
        nk_reduce_minmax_u1(&data->raw_, count, stride_bytes, &min_value->raw_, &min_offset, &max_value->raw_,
                            &max_offset);
    // Scalar fallback
    else {
        minmax_type_ best_min = finite_max<minmax_type_>();
        minmax_type_ best_max = finite_min<minmax_type_>();
        vector_view<in_type_> values(reinterpret_cast<char const *>(data), count, stride_bytes);
        for (nk_size_t i = 0; i < count; ++i) {
            minmax_type_ v = minmax_type_(values[i]);
            if (v < best_min) best_min = v, min_offset = i;
            if (v > best_max) best_max = v, max_offset = i;
        }
        *min_value = best_min, *max_value = best_max;
    }
    if (min_index) *min_index = static_cast<std::size_t>(min_offset);
    if (max_index) *max_index = static_cast<std::size_t>(max_offset);
}

/** @brief Compute sum and sum-of-squares over a vector view. */
template <numeric_dtype in_type_, numeric_dtype sum_type_ = typename in_type_::reduce_moments_sum_t,
          numeric_dtype sumsq_type_ = typename in_type_::reduce_moments_sumsq_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_moments(vector_view<in_type_> input, sum_type_ *sum, sumsq_type_ *sumsq) noexcept {
    reduce_moments<in_type_, sum_type_, sumsq_type_, allow_simd_>(
        input.data(), input.size(), static_cast<std::size_t>(input.stride_bytes()), sum, sumsq);
}

/** @brief Find minimum and maximum elements with their indices over a vector view. */
template <numeric_dtype in_type_, numeric_dtype minmax_type_ = typename in_type_::reduce_minmax_value_t,
          allow_simd_t allow_simd_ = prefer_simd_k>
void reduce_minmax(vector_view<in_type_> input, minmax_type_ *min_value, std::size_t *min_index,
                   minmax_type_ *max_value, std::size_t *max_index) noexcept {
    reduce_minmax<in_type_, minmax_type_, allow_simd_>(input.data(), input.size(),
                                                       static_cast<std::size_t>(input.stride_bytes()), min_value,
                                                       min_index, max_value, max_index);
}

} // namespace ashvardanian::numkong

#include "numkong/tensor.hpp"

namespace ashvardanian::numkong {

#pragma region Tensor Reduction Helpers

/** @brief Result of detecting how many trailing dimensions form a single arithmetic progression. */
struct uniform_stride_tail_result_t_ {
    std::size_t tail_dims;     ///< Number of collapsible trailing dimensions.
    std::size_t element_count; ///< Product of collapsed extents.
    std::size_t stride_bytes;  ///< Absolute stride of the innermost collapsed dimension.
};

/** @brief Detect trailing dimensions where stride[i] == stride[i+1] * extent[i+1].
 *  When this holds, the tail is a single strided sequence and can be passed to a SIMD
 *  kernel in one call with (element_count, stride_bytes). */
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

/** @brief Collapse the trailing `tail.tail_dims` dimensions into one, preserving outer dims and strides. */
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

/** @brief Normalize a fully-collapsed tail for SIMD kernel consumption, handling negative strides. */
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
bool reduce_rank1_moments_(tensor_view<value_type_, max_rank_> input, typename value_type_::reduce_moments_sum_t &sum,
                           typename value_type_::reduce_moments_sumsq_t &sumsq) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    if (input.rank() != 1 || !tensor_layout_supported_(input) || input.byte_data() == nullptr) return false;
    if (can_reduce_rank1_with_kernel_(input)) {
        auto lane = normalize_rank1_lane_(input);
        numkong::reduce_moments<value_type_>(lane.data, lane.count, lane.stride_bytes, &sum, &sumsq);
        return true;
    }
    auto values = input.as_vector();
    sum = sum_t {};
    sumsq = sumsq_t {};
    for (std::size_t i = 0; i < values.size(); ++i) {
        auto value = values[i];
        sum = saturating_add(sum, value);
        sumsq = saturating_fma(value, value, sumsq);
    }
    return true;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
bool reduce_rank1_minmax_(tensor_view<value_type_, max_rank_> input,
                          minmax_result<typename value_type_::reduce_minmax_value_t> &result) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    if (input.rank() != 1 || !tensor_layout_supported_(input) || input.byte_data() == nullptr) return false;
    if (can_reduce_rank1_with_kernel_(input)) {
        auto lane = normalize_rank1_lane_(input);
        numkong::reduce_minmax<value_type_>(lane.data, lane.count, lane.stride_bytes, &result.min_value,
                                            &result.min_index, &result.max_value, &result.max_index);
        if (lane.reversed) {
            result.min_index = lane.count - 1 - result.min_index;
            result.max_index = lane.count - 1 - result.max_index;
        }
        return true;
    }
    auto values = input.as_vector();
    result.min_value = finite_max<minmax_t>();
    result.max_value = finite_min<minmax_t>();
    result.min_index = 0;
    result.max_index = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        minmax_t value = minmax_t(values[i]);
        if (value < result.min_value) result.min_value = value, result.min_index = i;
        if (value > result.max_value) result.max_value = value, result.max_index = i;
    }
    return true;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
bool accumulate_moments_tensor_(tensor_view<value_type_, max_rank_> input,
                                tensor_span<typename value_type_::reduce_moments_sum_t, max_rank_> sums,
                                tensor_span<typename value_type_::reduce_moments_sumsq_t, max_rank_> sumsqs) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    if (!tensor_layout_supported_(input) || !shapes_match_out_(input, sums) || !shapes_match_out_(input, sumsqs))
        return false;
    if (input.rank() == 1) {
        auto src = input.as_vector();
        auto dst_sum = sums.as_vector();
        auto dst_sumsq = sumsqs.as_vector();
        for (std::size_t i = 0; i < src.size(); ++i) {
            auto value = src[i];
            dst_sum[i] = saturating_add(dst_sum[i], sum_t(value));
            dst_sumsq[i] = saturating_fma(value, value, sumsq_t(dst_sumsq[i]));
        }
        return true;
    }
    for (std::size_t i = 0; i < input.extent(0); ++i) {
        if (!accumulate_moments_tensor_(input.slice_leading(i), sums.slice_leading(i), sumsqs.slice_leading(i)))
            return false;
    }
    return true;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
bool update_minmax_tensor_(tensor_view<value_type_, max_rank_> input,
                           tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> mins,
                           tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> maxs) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    if (!tensor_layout_supported_(input) || !shapes_match_out_(input, mins) || !shapes_match_out_(input, maxs))
        return false;
    if (input.rank() == 1) {
        auto src = input.as_vector();
        auto dst_min = mins.as_vector();
        auto dst_max = maxs.as_vector();
        for (std::size_t i = 0; i < src.size(); ++i) {
            minmax_t value = minmax_t(src[i]);
            if (value < dst_min[i]) dst_min[i] = value;
            if (value > dst_max[i]) dst_max[i] = value;
        }
        return true;
    }
    for (std::size_t i = 0; i < input.extent(0); ++i) {
        if (!update_minmax_tensor_(input.slice_leading(i), mins.slice_leading(i), maxs.slice_leading(i))) return false;
    }
    return true;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
bool reduce_moments_axis_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                          typename value_type_::reduce_moments_sum_t *sums,
                          typename value_type_::reduce_moments_sumsq_t *sumsqs) noexcept {
    return for_each_axis_lane_(input, axis,
                               [&](tensor_view<value_type_, max_rank_> lane, std::size_t output_index) noexcept {
                                   typename value_type_::reduce_moments_sum_t sum {};
                                   typename value_type_::reduce_moments_sumsq_t sumsq {};
                                   if (!reduce_rank1_moments_(lane, sum, sumsq)) return false;
                                   if (sums) sums[output_index] = sum;
                                   if (sumsqs) sumsqs[output_index] = sumsq;
                                   return true;
                               });
}

template <numeric_dtype value_type_, std::size_t max_rank_>
bool reduce_moments_axis_packed_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                                 tensor_span<typename value_type_::reduce_moments_sum_t, max_rank_> sums,
                                 tensor_span<typename value_type_::reduce_moments_sumsq_t, max_rank_> sumsqs,
                                 keep_dims_t keep_dims) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    if (!tensor_layout_supported_(input) || axis >= input.rank()) return false;
    if (axis == 0) {
        auto sum_target = keep_dims ? sums.slice_leading(0) : sums;
        auto sumsq_target = keep_dims ? sumsqs.slice_leading(0) : sumsqs;
        if (input.rank() == 1) {
            sum_t sum {};
            sumsq_t sumsq {};
            if (!reduce_rank1_moments_(input, sum, sumsq)) return false;
            sum_target.scalar_ref() = sum;
            sumsq_target.scalar_ref() = sumsq;
            return true;
        }
        for (std::size_t i = 0; i < input.extent(0); ++i)
            if (!accumulate_moments_tensor_(input.slice_leading(i), sum_target, sumsq_target)) return false;
        return true;
    }
    if (input.rank() == 1) return false;
    for (std::size_t i = 0; i < input.extent(0); ++i)
        if (!reduce_moments_axis_packed_(input.slice_leading(i), axis - 1, sums.slice_leading(i),
                                         sumsqs.slice_leading(i), keep_dims))
            return false;
    return true;
}

template <numeric_dtype value_type_, std::size_t max_rank_>
bool reduce_minmax_axis_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                         typename value_type_::reduce_minmax_value_t *mins, std::size_t *argmins,
                         typename value_type_::reduce_minmax_value_t *maxs, std::size_t *argmaxs) noexcept {
    return for_each_axis_lane_(input, axis,
                               [&](tensor_view<value_type_, max_rank_> lane, std::size_t output_index) noexcept {
                                   minmax_result<typename value_type_::reduce_minmax_value_t> result {};
                                   if (!reduce_rank1_minmax_(lane, result)) return false;
                                   if (mins) mins[output_index] = result.min_value;
                                   if (argmins) argmins[output_index] = result.min_index;
                                   if (maxs) maxs[output_index] = result.max_value;
                                   if (argmaxs) argmaxs[output_index] = result.max_index;
                                   return true;
                               });
}

template <numeric_dtype value_type_, std::size_t max_rank_>
bool reduce_minmax_axis_packed_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                                tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> mins,
                                tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> maxs,
                                keep_dims_t keep_dims) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    if (!tensor_layout_supported_(input) || axis >= input.rank()) return false;
    if (axis == 0) {
        auto min_target = keep_dims ? mins.slice_leading(0) : mins;
        auto max_target = keep_dims ? maxs.slice_leading(0) : maxs;
        if (input.rank() == 1) {
            minmax_result<minmax_t> result {};
            if (!reduce_rank1_minmax_(input, result)) return false;
            min_target.scalar_ref() = result.min_value;
            max_target.scalar_ref() = result.max_value;
            return true;
        }
        for (std::size_t i = 0; i < input.extent(0); ++i)
            if (!update_minmax_tensor_(input.slice_leading(i), min_target, max_target)) return false;
        return true;
    }
    if (input.rank() == 1) return false;
    for (std::size_t i = 0; i < input.extent(0); ++i)
        if (!reduce_minmax_axis_packed_(input.slice_leading(i), axis - 1, mins.slice_leading(i), maxs.slice_leading(i),
                                        keep_dims))
            return false;
    return true;
}

#pragma endregion Tensor Reduction Helpers

#pragma region Scalar Reductions

/** @brief Compute Σxᵢ and Σxᵢ² in a single pass. Returns zeroed result for empty tensors. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t> moments(
    tensor_view<value_type_, max_rank_> input) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    moments_result<sum_t, sumsq_t> result {};
    if (input.empty() || input.numel() == 0 || !tensor_layout_supported_(input)) return result;
    auto tail = uniform_stride_tail_(input);
    if (tail.tail_dims == input.rank()) {
        auto lane = normalize_rank1_lane_from_tail_<value_type_, max_rank_>(input, tail);
        numkong::reduce_moments<value_type_>(lane.data, lane.count, lane.stride_bytes, &result.sum, &result.sumsq);
        return result;
    }
    if (tail.tail_dims >= 2) return moments<value_type_, max_rank_>(collapse_uniform_tail_(input, tail));
    // Sub-byte rank-1 fallback: uniform_stride_tail_ returns {0,0,0} for packed types.
    if (input.rank() == 1) {
        reduce_rank1_moments_(input, result.sum, result.sumsq);
        return result;
    }
    for (std::size_t i = 0; i < input.extent(0); ++i) {
        auto slice_result = moments<value_type_, max_rank_>(input.slice_leading(static_cast<std::ptrdiff_t>(i)));
        result.sum = saturating_add(result.sum, slice_result.sum);
        result.sumsq = saturating_add(result.sumsq, slice_result.sumsq);
    }
    return result;
}

/** @brief Find min and max values with their flat indices. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
minmax_result<typename value_type_::reduce_minmax_value_t> minmax(tensor_view<value_type_, max_rank_> input) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    minmax_result<minmax_t> result {};
    if (input.empty() || input.numel() == 0 || !tensor_layout_supported_(input)) return result;
    auto tail = uniform_stride_tail_(input);
    if (tail.tail_dims == input.rank()) {
        auto lane = normalize_rank1_lane_from_tail_<value_type_, max_rank_>(input, tail);
        numkong::reduce_minmax<value_type_>(lane.data, lane.count, lane.stride_bytes, &result.min_value,
                                            &result.min_index, &result.max_value, &result.max_index);
        if (lane.reversed) {
            result.min_index = tail.element_count - 1 - result.min_index;
            result.max_index = tail.element_count - 1 - result.max_index;
        }
        return result;
    }
    if (tail.tail_dims >= 2) return minmax<value_type_, max_rank_>(collapse_uniform_tail_(input, tail));
    // Sub-byte rank-1 fallback.
    if (input.rank() == 1) {
        reduce_rank1_minmax_(input, result);
        return result;
    }
    result.min_value = finite_max<minmax_t>();
    result.max_value = finite_min<minmax_t>();
    std::size_t base = 0;
    for (std::size_t i = 0; i < input.extent(0); ++i) {
        auto slice = input.slice_leading(static_cast<std::ptrdiff_t>(i));
        auto slice_result = minmax<value_type_, max_rank_>(slice);
        if (slice_result.min_value < result.min_value) {
            result.min_value = slice_result.min_value;
            result.min_index = base + slice_result.min_index;
        }
        if (slice_result.max_value > result.max_value) {
            result.max_value = slice_result.max_value;
            result.max_index = base + slice_result.max_index;
        }
        base += slice.numel();
    }
    return result;
}

/** @brief Σ of all elements. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_moments_sum_t sum(tensor_view<value_type_, max_rank_> input) noexcept {
    return moments(input).sum;
}

/** @brief Find the minimum element value. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_minmax_value_t min(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).min_value;
}

/** @brief Find the maximum element value. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_minmax_value_t max(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).max_value;
}

/** @brief Index of the minimum element (flat). */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
std::size_t argmin(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).min_index;
}

/** @brief Index of the maximum element (flat). */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8>
std::size_t argmax(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).max_index;
}

/** @brief Compute Σxᵢ and Σxᵢ² over a vector view. */
template <numeric_dtype value_type_>
moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t> moments(
    vector_view<value_type_> input) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    moments_result<sum_t, sumsq_t> result {};
    if (input.size() == 0) return result;
    reduce_moments<value_type_>(input, &result.sum, &result.sumsq);
    return result;
}

/** @brief Find min and max values with their indices over a vector view. */
template <numeric_dtype value_type_>
minmax_result<typename value_type_::reduce_minmax_value_t> minmax(vector_view<value_type_> input) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    minmax_result<minmax_t> result {};
    if (input.size() == 0) return result;
    reduce_minmax<value_type_>(input, &result.min_value, &result.min_index, &result.max_value, &result.max_index);
    return result;
}

/** @brief Σ of all elements in a vector view. */
template <numeric_dtype value_type_>
typename value_type_::reduce_moments_sum_t sum(vector_view<value_type_> input) noexcept {
    return moments(input).sum;
}

/** @brief Find the minimum element value in a vector view. */
template <numeric_dtype value_type_>
typename value_type_::reduce_minmax_value_t min(vector_view<value_type_> input) noexcept {
    return minmax(input).min_value;
}

/** @brief Find the maximum element value in a vector view. */
template <numeric_dtype value_type_>
typename value_type_::reduce_minmax_value_t max(vector_view<value_type_> input) noexcept {
    return minmax(input).max_value;
}

/** @brief Index of the minimum element in a vector view. */
template <numeric_dtype value_type_>
std::size_t argmin(vector_view<value_type_> input) noexcept {
    return minmax(input).min_index;
}

/** @brief Index of the maximum element in a vector view. */
template <numeric_dtype value_type_>
std::size_t argmax(vector_view<value_type_> input) noexcept {
    return minmax(input).max_index;
}

/** @brief Population count of a packed bit tensor — number of set bits.
 *  Wraps `nk_reduce_moments_u1` via `sum`; returns 0 for empty inputs. */
template <std::size_t max_rank_ = 8>
u64_t popcount(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return sum<u1x8_t, max_rank_>(input);
}

/** @brief True if any bit of the packed bit tensor is set. Suffixed `_set` to
 *  avoid colliding with the `all_t`/`all` slice marker in `vector.hpp`. */
template <std::size_t max_rank_ = 8>
bool any_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return popcount<max_rank_>(input).raw_ != 0;
}

/** @brief True if no bit of the packed bit tensor is set. */
template <std::size_t max_rank_ = 8>
bool none_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return !any_set<max_rank_>(input);
}

/** @brief True if every bit of the packed bit tensor is set. */
template <std::size_t max_rank_ = 8>
bool all_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return popcount<max_rank_>(input).raw_ == input.numel();
}

/** @brief Population count over a 1D bit-vector view. */
inline u64_t popcount(vector_view<u1x8_t> input) noexcept { return sum<u1x8_t>(input); }

/** @brief True if any bit of the 1D bit-vector view is set. */
inline bool any_set(vector_view<u1x8_t> input) noexcept { return popcount(input).raw_ != 0; }

/** @brief True if no bit of the 1D bit-vector view is set. */
inline bool none_set(vector_view<u1x8_t> input) noexcept { return !any_set(input); }

/** @brief True if every bit of the 1D bit-vector view is set. */
inline bool all_set(vector_view<u1x8_t> input) noexcept { return popcount(input).raw_ == input.size(); }

#pragma endregion Scalar Reductions

#pragma region Axis Reductions

/** @brief Σ along a single axis. Returns empty tensor on failure. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_moments_sum_t>>
tensor<typename value_type_::reduce_moments_sum_t, allocator_type_, max_rank_> try_sum(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sum_tensor_t = tensor<sum_t, allocator_type_, max_rank_>;

    if (input.empty() || axis >= input.rank() || !tensor_layout_supported_(input)) return sum_tensor_t {};

    auto out_shape = reduced_shape_<sum_t>(input.shape(), axis, keep_dims);
    auto sums = sum_tensor_t::try_zeros(out_shape.extents, out_shape.rank);
    if (sums.empty() || !shape_matches_(out_shape, sums.span())) return sum_tensor_t {};
    if constexpr (dimensions_per_value<value_type_>() > 1) {
        using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
        using sumsq_alloc_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<sumsq_t>;
        using sumsq_tensor_t = tensor<sumsq_t, sumsq_alloc_t, max_rank_>;
        auto scratch = sumsq_tensor_t::try_zeros(out_shape.extents, out_shape.rank);
        if (scratch.empty() || !shape_matches_(reduced_shape_<sumsq_t>(input.shape(), axis, keep_dims), scratch.span()))
            return sum_tensor_t {};
        if (!reduce_moments_axis_packed_(input, axis, sums.span(), scratch.span(), keep_dims)) return sum_tensor_t {};
    }
    else if (!reduce_moments_axis_(input, axis, sums.data(), nullptr)) return sum_tensor_t {};
    return sums;
}

/** @brief Moments along an axis (Σxᵢ and Σxᵢ² per slice). */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_moments_sum_t>>
moments_result<tensor<typename value_type_::reduce_moments_sum_t, allocator_type_, max_rank_>,
               tensor<typename value_type_::reduce_moments_sumsq_t,
                      typename std::allocator_traits<allocator_type_>::template rebind_alloc<
                          typename value_type_::reduce_moments_sumsq_t>,
                      max_rank_>>
try_moments(tensor_view<value_type_, max_rank_> input, std::size_t axis,
            keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    using sum_tensor_t = tensor<sum_t, allocator_type_, max_rank_>;
    using sumsq_alloc_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<sumsq_t>;
    using sumsq_tensor_t = tensor<sumsq_t, sumsq_alloc_t, max_rank_>;

    if (input.empty() || axis >= input.rank() || !tensor_layout_supported_(input))
        return {sum_tensor_t {}, sumsq_tensor_t {}};

    auto out_shape_sum = reduced_shape_<sum_t>(input.shape(), axis, keep_dims);
    auto out_shape_sq = reduced_shape_<sumsq_t>(input.shape(), axis, keep_dims);

    auto sums = sum_tensor_t::try_zeros(out_shape_sum.extents, out_shape_sum.rank);
    auto sumsqs = sumsq_tensor_t::try_zeros(out_shape_sq.extents, out_shape_sq.rank);
    if (sums.empty() || sumsqs.empty() || !shape_matches_(out_shape_sum, sums.span()) ||
        !shape_matches_(out_shape_sq, sumsqs.span()))
        return {sum_tensor_t {}, sumsq_tensor_t {}};

    if constexpr (dimensions_per_value<value_type_>() > 1) {
        if (!reduce_moments_axis_packed_(input, axis, sums.span(), sumsqs.span(), keep_dims))
            return {sum_tensor_t {}, sumsq_tensor_t {}};
    }
    else if (!reduce_moments_axis_(input, axis, sums.data(), sumsqs.data()))
        return {sum_tensor_t {}, sumsq_tensor_t {}};

    return {std::move(sums), std::move(sumsqs)};
}

/** @brief Min and max along an axis. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
minmax_result<tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_>> try_minmax(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    using out_tensor_t = tensor<minmax_t, allocator_type_, max_rank_>;
    if (input.empty() || axis >= input.rank() || !tensor_layout_supported_(input))
        return {out_tensor_t {}, 0, out_tensor_t {}, 0};

    auto out_shape = reduced_shape_<minmax_t>(input.shape(), axis, keep_dims);
    auto mins = out_tensor_t::try_full(out_shape.extents, out_shape.rank, finite_max<minmax_t>());
    auto maxs = out_tensor_t::try_full(out_shape.extents, out_shape.rank, finite_min<minmax_t>());
    if (mins.empty() || maxs.empty() || !shape_matches_(out_shape, mins.span()) ||
        !shape_matches_(out_shape, maxs.span()))
        return {out_tensor_t {}, 0, out_tensor_t {}, 0};

    if constexpr (dimensions_per_value<value_type_>() > 1) {
        if (!reduce_minmax_axis_packed_(input, axis, mins.span(), maxs.span(), keep_dims))
            return {out_tensor_t {}, 0, out_tensor_t {}, 0};
    }
    else if (!reduce_minmax_axis_(input, axis, mins.data(), nullptr, maxs.data(), nullptr))
        return {out_tensor_t {}, 0, out_tensor_t {}, 0};
    return {std::move(mins), 0, std::move(maxs), 0};
}

/** @brief Argmin along an axis. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<std::size_t>>
tensor<std::size_t, allocator_type_, max_rank_> try_argmin(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                                                           keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using out_tensor_t = tensor<std::size_t, allocator_type_, max_rank_>;
    if (input.empty() || axis >= input.rank() || !tensor_layout_supported_(input)) return out_tensor_t {};
    if constexpr (dimensions_per_value<value_type_>() > 1) return out_tensor_t {};

    auto out_shape = reduced_shape_<std::size_t>(input.shape(), axis, keep_dims);
    auto indices = out_tensor_t::try_zeros(out_shape.extents, out_shape.rank);
    if (indices.empty() || !shape_matches_(out_shape, indices.span())) return out_tensor_t {};
    if (!reduce_minmax_axis_(input, axis, nullptr, indices.data(), nullptr, nullptr)) return out_tensor_t {};
    return indices;
}

/** @brief Argmax along an axis. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<std::size_t>>
tensor<std::size_t, allocator_type_, max_rank_> try_argmax(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                                                           keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using out_tensor_t = tensor<std::size_t, allocator_type_, max_rank_>;
    if (input.empty() || axis >= input.rank() || !tensor_layout_supported_(input)) return out_tensor_t {};
    if constexpr (dimensions_per_value<value_type_>() > 1) return out_tensor_t {};

    auto out_shape = reduced_shape_<std::size_t>(input.shape(), axis, keep_dims);
    auto indices = out_tensor_t::try_zeros(out_shape.extents, out_shape.rank);
    if (indices.empty() || !shape_matches_(out_shape, indices.span())) return out_tensor_t {};
    if (!reduce_minmax_axis_(input, axis, nullptr, nullptr, nullptr, indices.data())) return out_tensor_t {};
    return indices;
}

/** @brief Min along an axis. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_> try_min(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k) noexcept {
    return try_minmax<value_type_, max_rank_, allocator_type_>(input, axis, keep_dims).min_value;
}

/** @brief Max along an axis. */
template <numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_> try_max(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k) noexcept {
    return try_minmax<value_type_, max_rank_, allocator_type_>(input, axis, keep_dims).max_value;
}

#pragma endregion Axis Reductions

} // namespace ashvardanian::numkong

#endif // NK_REDUCE_HPP
