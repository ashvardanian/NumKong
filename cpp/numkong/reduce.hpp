/**
 *  @file cpp/numkong/reduce.hpp
 *  @author Ash Vardanian
 *  @date February 5, 2026
 *  @brief Reduction kernels: reduce_moments (sum, sum-of-squares), reduce_minmax (min/max/indices).
 */
#if defined(NUMKONG_HEADER_ONLY) && NUMKONG_HEADER_ONLY
#error "The NumKong C++ SDK links the library; link numkong::cpp, not numkong::header"
#endif

#ifndef NUMKONG_REDUCE_HPP
#define NUMKONG_REDUCE_HPP

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <concepts>    // `std::totally_ordered`
#include <memory>      // `std::allocator_traits`
#include <type_traits> // `std::is_same_v`

#include "numkong/reduce.h"

#include "numkong/tensor.hpp"
#include "numkong/types.hpp"
#include "numkong/vector.hpp"
#include "numkong/devices.hpp" // `default_capabilities`
#include "numkong/scalar.hpp"  // `saturating_add_`

namespace ashvardanian::numkong {

/**
 *  @brief Computes sum and sum of squares in a single pass: sum = Σ dataᵢ, sumsq = Σ dataᵢ².
 *  @param[in] data Input array.
 *  @param[in] count Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride Stride between elements in bytes, <tt>sizeof(in_type_)</tt> for contiguous.
 *  @param[out] sum Output sum.
 *  @param[out] sumsq Output sum of squares.
 *  @param[in] capabilities Capabilities to pick from.
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes.
 *
 *  @tparam in_type_ Input vector element type.
 */
template <a_numeric_dtype in_type_>
status_t reduce_moments(in_type_ const *data, std::size_t count, std::size_t stride,
                        typename in_type_::reduce_moments_sum_t *sum, typename in_type_::reduce_moments_sumsq_t *sumsq,
                        nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    if constexpr (std::is_same_v<in_type_, f64_t>)
        return static_cast<status_t>(
            nk_reduce_moments_f64_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        return static_cast<status_t>(
            nk_reduce_moments_f32_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        return static_cast<status_t>(
            nk_reduce_moments_f16_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        return static_cast<status_t>(
            nk_reduce_moments_bf16_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        return static_cast<status_t>(
            nk_reduce_moments_e4m3_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        return static_cast<status_t>(
            nk_reduce_moments_e5m2_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        return static_cast<status_t>(
            nk_reduce_moments_e2m3_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e2m1x2_t>)
        return static_cast<status_t>(
            nk_reduce_moments_e2m1_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        return static_cast<status_t>(
            nk_reduce_moments_e3m2_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        return static_cast<status_t>(
            nk_reduce_moments_i4_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        return static_cast<status_t>(
            nk_reduce_moments_u4_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        return static_cast<status_t>(
            nk_reduce_moments_u1_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        return static_cast<status_t>(
            nk_reduce_moments_i8_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        return static_cast<status_t>(
            nk_reduce_moments_u8_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i16_t>)
        return static_cast<status_t>(
            nk_reduce_moments_i16_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u16_t>)
        return static_cast<status_t>(
            nk_reduce_moments_u16_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i32_t>)
        return static_cast<status_t>(
            nk_reduce_moments_i32_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u32_t>)
        return static_cast<status_t>(
            nk_reduce_moments_u32_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, i64_t>)
        return static_cast<status_t>(
            nk_reduce_moments_i64_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else if constexpr (std::is_same_v<in_type_, u64_t>)
        return static_cast<status_t>(
            nk_reduce_moments_u64_best(&data->raw_, count, stride, &sum->raw_, &sumsq->raw_, capabilities, stream));
    else static_assert(sizeof(in_type_) == 0, "nk_reduce_moments has no kernel for this dtype");
}

/**
 *  @brief Find minimum and maximum elements with their indices in a single pass.
 *  @param[in] data Input array
 *  @param[in] count Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride Stride between elements in bytes (use sizeof(in_type_) for contiguous)
 *  @param[out] min_value Output minimum value
 *  @param[out] min_index Output index of minimum value, @c NUMKONG_SIZE_MAX if every value is NaN
 *  @param[out] max_value Output maximum value
 *  @param[out] max_index Output index of maximum value, @c NUMKONG_SIZE_MAX if every value is NaN
 *  @param[in] capabilities Capabilities to pick from
 *  @param[in] stream Null on the CPU, or the stream of the device @p capabilities describes
 *
 *  @tparam in_type_ Input vector element type
 */
template <a_numeric_dtype in_type_>
status_t reduce_minmax(in_type_ const *data, std::size_t count, std::size_t stride,
                       typename in_type_::reduce_minmax_value_t *min_value, std::size_t *min_index,
                       typename in_type_::reduce_minmax_value_t *max_value, std::size_t *max_index,
                       nk_capability_t capabilities = default_capabilities(), nk_stream_t stream = nullptr) noexcept {
    static_assert(sizeof(std::size_t) == sizeof(nk_size_t), "std::size_t and nk_size_t must have the same width");
    nk_size_t min_offset = NUMKONG_SIZE_MAX, max_offset = NUMKONG_SIZE_MAX;
    nk_status_t status;
    if constexpr (std::is_same_v<in_type_, f64_t>)
        status = nk_reduce_minmax_f64_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, f32_t>)
        status = nk_reduce_minmax_f32_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, i8_t>)
        status = nk_reduce_minmax_i8_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                          &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, u8_t>)
        status = nk_reduce_minmax_u8_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                          &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, i16_t>)
        status = nk_reduce_minmax_i16_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, u16_t>)
        status = nk_reduce_minmax_u16_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, i32_t>)
        status = nk_reduce_minmax_i32_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, u32_t>)
        status = nk_reduce_minmax_u32_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, i64_t>)
        status = nk_reduce_minmax_i64_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, u64_t>)
        status = nk_reduce_minmax_u64_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, e2m3_t>)
        status = nk_reduce_minmax_e2m3_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                            &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, e3m2_t>)
        status = nk_reduce_minmax_e3m2_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                            &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, f16_t>)
        status = nk_reduce_minmax_f16_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                           &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, bf16_t>)
        status = nk_reduce_minmax_bf16_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                            &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, e4m3_t>)
        status = nk_reduce_minmax_e4m3_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                            &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, e5m2_t>)
        status = nk_reduce_minmax_e5m2_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                            &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, i4x2_t>)
        status = nk_reduce_minmax_i4_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                          &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, u4x2_t>)
        status = nk_reduce_minmax_u4_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                          &max_offset, capabilities, stream);
    else if constexpr (std::is_same_v<in_type_, u1x8_t>)
        status = nk_reduce_minmax_u1_best(&data->raw_, count, stride, &min_value->raw_, &min_offset, &max_value->raw_,
                                          &max_offset, capabilities, stream);
    else static_assert(sizeof(in_type_) == 0, "nk_reduce_minmax has no kernel for this dtype");
    if (status != nk_success_k) return static_cast<status_t>(status);
    if (min_index) *min_index = static_cast<std::size_t>(min_offset);
    if (max_index) *max_index = static_cast<std::size_t>(max_offset);
    return status_t::success_k;
}

#pragma region Tensor Reduction Helpers

/** Result of detecting how many trailing dimensions form a single arithmetic progression:
 *  @c tail_dims collapsible trailing dimensions, @c element_count product of collapsed extents,
 *  @c stride absolute stride of the innermost collapsed dimension. */
struct uniform_stride_tail_result_t_ {
    std::size_t tail_dims;
    std::size_t element_count;
    std::size_t stride;
};

/**
 *  @brief Detects trailing dimensions where <tt>stride[i] == stride[i+1] * extent[i+1]</tt>.
 *
 *  When this holds, the tail is a single strided sequence and can be passed to a SIMD kernel in one
 *  call with @c element_count and @c stride.
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

/** Collapses trailing <tt>tail.tail_dims</tt> dimensions into one, preserving outer dims and strides. */
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
    lane.stride = tail.stride;
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

template <a_numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_rank1_moments_(tensor_view<value_type_, max_rank_> input,
                               typename value_type_::reduce_moments_sum_t &sum,
                               typename value_type_::reduce_moments_sumsq_t &sumsq, nk_capability_t capabilities,
                               nk_stream_t stream) noexcept {
    if (input.rank() != 1 || input.byte_data() == nullptr) return status_t::unexpected_dimensions_k;
    if (!tensor_layout_supported_(input) || !can_reduce_rank1_with_kernel_(input)) return status_t::misaligned_k;
    normalized_rank1_lane_<value_type_, max_rank_> lane = normalize_rank1_lane_(input);
    return numkong::reduce_moments<value_type_>(lane.data, lane.count, lane.stride, &sum, &sumsq, capabilities, stream);
}

template <a_numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_rank1_minmax_(tensor_view<value_type_, max_rank_> input,
                              minmax_result<typename value_type_::reduce_minmax_value_t> &result,
                              nk_capability_t capabilities, nk_stream_t stream) noexcept {
    if (input.rank() != 1 || input.byte_data() == nullptr) return status_t::unexpected_dimensions_k;
    if (!tensor_layout_supported_(input) || !can_reduce_rank1_with_kernel_(input)) return status_t::misaligned_k;
    normalized_rank1_lane_<value_type_, max_rank_> lane = normalize_rank1_lane_(input);
    if (status_t status = numkong::reduce_minmax<value_type_>(lane.data, lane.count, lane.stride, &result.min_value,
                                                              &result.min_index, &result.max_value, &result.max_index,
                                                              capabilities, stream);
        failed(status))
        return status;
    if (lane.reversed) {
        result.min_index = lane.count - 1 - result.min_index;
        result.max_index = lane.count - 1 - result.max_index;
    }
    return status_t::success_k;
}

template <a_numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_moments_axis_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                              typename value_type_::reduce_moments_sum_t *sums,
                              typename value_type_::reduce_moments_sumsq_t *sumsqs, nk_capability_t capabilities,
                              nk_stream_t stream) noexcept {
    return for_each_axis_lane_(
        input, axis, [&](tensor_view<value_type_, max_rank_> lane, std::size_t output_index) noexcept {
            typename value_type_::reduce_moments_sum_t sum {};
            typename value_type_::reduce_moments_sumsq_t sumsq {};
            if (status_t status = reduce_rank1_moments_(lane, sum, sumsq, capabilities, stream); failed(status))
                return status;
            if (sums) sums[output_index] = sum;
            if (sumsqs) sumsqs[output_index] = sumsq;
            return status_t::success_k;
        });
}

template <a_numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_moments_axis_packed_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                                     tensor_span<typename value_type_::reduce_moments_sum_t, max_rank_> sums,
                                     tensor_span<typename value_type_::reduce_moments_sumsq_t, max_rank_> sumsqs,
                                     keep_dims_t keep_dims, nk_capability_t capabilities, nk_stream_t stream) noexcept {
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
            if (status_t status = reduce_rank1_moments_(input, sum, sumsq, capabilities, stream); failed(status))
                return status;
            sum_target.scalar_ref() = sum;
            sumsq_target.scalar_ref() = sumsq;
            return status_t::success_k;
        }
        // Each lane steps through one sub-byte value per row, which no kernel can stride over
        return status_t::misaligned_k;
    }
    if (input.rank() == 1) return status_t::unexpected_dimensions_k;
    for (std::size_t i = 0; i < input.extent(0); ++i)
        if (status_t status = reduce_moments_axis_packed_(input.slice_leading(i), axis - 1, sums.slice_leading(i),
                                                          sumsqs.slice_leading(i), keep_dims, capabilities, stream);
            failed(status))
            return status;
    return status_t::success_k;
}

template <a_numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_minmax_axis_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                             typename value_type_::reduce_minmax_value_t *mins, std::size_t *argmins,
                             typename value_type_::reduce_minmax_value_t *maxs, std::size_t *argmaxs,
                             nk_capability_t capabilities, nk_stream_t stream) noexcept {
    return for_each_axis_lane_(
        input, axis, [&](tensor_view<value_type_, max_rank_> lane, std::size_t output_index) noexcept {
            minmax_result<typename value_type_::reduce_minmax_value_t> result {};
            if (status_t status = reduce_rank1_minmax_(lane, result, capabilities, stream); failed(status))
                return status;
            if (mins) mins[output_index] = result.min_value;
            if (argmins) argmins[output_index] = result.min_index;
            if (maxs) maxs[output_index] = result.max_value;
            if (argmaxs) argmaxs[output_index] = result.max_index;
            return status_t::success_k;
        });
}

template <a_numeric_dtype value_type_, std::size_t max_rank_>
status_t reduce_minmax_axis_packed_(tensor_view<value_type_, max_rank_> input, std::size_t axis,
                                    tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> mins,
                                    tensor_span<typename value_type_::reduce_minmax_value_t, max_rank_> maxs,
                                    keep_dims_t keep_dims, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    if (!tensor_layout_supported_(input)) return status_t::misaligned_k;
    if (axis >= input.rank()) return status_t::unexpected_dimensions_k;
    if (axis == 0) {
        auto min_target = keep_dims ? mins.slice_leading(0) : mins;
        auto max_target = keep_dims ? maxs.slice_leading(0) : maxs;
        if (input.rank() == 1) {
            minmax_result<minmax_t> result {};
            if (status_t status = reduce_rank1_minmax_(input, result, capabilities, stream); failed(status))
                return status;
            min_target.scalar_ref() = result.min_value;
            max_target.scalar_ref() = result.max_value;
            return status_t::success_k;
        }
        return status_t::misaligned_k;
    }
    if (input.rank() == 1) return status_t::unexpected_dimensions_k;
    for (std::size_t i = 0; i < input.extent(0); ++i)
        if (status_t status = reduce_minmax_axis_packed_(input.slice_leading(i), axis - 1, mins.slice_leading(i),
                                                         maxs.slice_leading(i), keep_dims, capabilities, stream);
            failed(status))
            return status;
    return status_t::success_k;
}

#pragma endregion Tensor Reduction Helpers

#pragma region Scalar Reductions

/** Whether @p a sorts before @p b: the order kernel for a narrow float, the type's own comparison
 *  otherwise. */
template <typename value_type_>
bool sorts_before_(value_type_ a, value_type_ b, nk_capability_t) noexcept {
    return a < b;
}

/** @copydoc sorts_before_ */
template <typename value_type_>
    requires(!std::totally_ordered<value_type_>)
bool sorts_before_(value_type_ a, value_type_ b, nk_capability_t capabilities) noexcept {
    return order_(a, b, capabilities) < 0;
}

/** @p a + @p b clamped to the range of @p sum_type_: the saturating-add kernel for an integer sum,
 *  the clamp of a floating one otherwise. */
template <typename sum_type_>
sum_type_ saturating_sum_(sum_type_ a, sum_type_ b, nk_capability_t capabilities) noexcept {
    if constexpr (is_integral_dtype<sum_type_>()) return saturating_add_(a, b, capabilities);
    else return a.saturating_add(b);
}

/** Σxᵢ and Σxᵢ² in a single pass; zeros for an empty @p input, @c misaligned_k for an unsupported
 *  layout or a sub-byte lane whose step is not a whole byte. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t>>
moments(tensor_view<value_type_, max_rank_> input, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    using result_t =
        moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t>;
    result_t result {};
    if (input.empty() || input.numel() == 0) return {result, status_t::success_k};
    // A 0-D view is a single contiguous scalar: no axes to collapse and no innermost stride to read
    if (input.rank() == 0) {
        status_t status = numkong::reduce_moments<value_type_>(input.data(), 1, sizeof(value_type_), &result.sum,
                                                               &result.sumsq, capabilities, stream);
        return {result, status};
    }
    if (!tensor_layout_supported_(input)) return {result, status_t::misaligned_k};
    uniform_stride_tail_result_t_ const tail = uniform_stride_tail_(input);
    if (tail.tail_dims == input.rank()) {
        normalized_rank1_lane_<value_type_, max_rank_> lane = normalize_rank1_lane_from_tail_<value_type_, max_rank_>(
            input, tail);
        status_t status = numkong::reduce_moments<value_type_>(lane.data, lane.count, lane.stride, &result.sum,
                                                               &result.sumsq, capabilities, stream);
        return {result, status};
    }
    if (tail.tail_dims >= 2)
        return moments<value_type_, max_rank_>(collapse_uniform_tail_(input, tail), capabilities, stream);
    // Sub-byte rank-1 lanes: uniform_stride_tail_ returns {0,0,0} for packed types
    if (input.rank() == 1) {
        status_t status = reduce_rank1_moments_(input, result.sum, result.sumsq, capabilities, stream);
        return {result, status};
    }
    for (std::size_t i = 0; i < input.extent(0); ++i) {
        auto [slice_result, slice_status] = moments<value_type_, max_rank_>(
            input.slice_leading(static_cast<std::ptrdiff_t>(i)), capabilities, stream);
        if (failed(slice_status)) return {result, slice_status};
        result.sum = saturating_sum_(result.sum, slice_result.sum, capabilities);
        result.sumsq = saturating_sum_(result.sumsq, slice_result.sumsq, capabilities);
    }
    return {result, status_t::success_k};
}

/** @c moments with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t>>
moments(tensor_view<value_type_, max_rank_> input) noexcept {
    return moments<value_type_, max_rank_>(input, default_capabilities(), nullptr);
}

/** Min and max values with their flat indices, @c NUMKONG_SIZE_MAX for an empty or all-NaN
 *  @p input; fails like the scalar @c moments. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<minmax_result<typename value_type_::reduce_minmax_value_t>> minmax(tensor_view<value_type_, max_rank_> input,
                                                                            nk_capability_t capabilities,
                                                                            nk_stream_t stream) noexcept {
    using result_t = minmax_result<typename value_type_::reduce_minmax_value_t>;
    result_t result {{}, NUMKONG_SIZE_MAX, {}, NUMKONG_SIZE_MAX};
    if (input.empty() || input.numel() == 0) return {result, status_t::success_k};
    // A 0-D view is a single contiguous scalar (index 0) with no innermost stride to read
    if (input.rank() == 0) {
        status_t status = numkong::reduce_minmax<value_type_>(input.data(), 1, sizeof(value_type_), &result.min_value,
                                                              &result.min_index, &result.max_value, &result.max_index,
                                                              capabilities, stream);
        return {result, status};
    }
    if (!tensor_layout_supported_(input)) return {result, status_t::misaligned_k};
    uniform_stride_tail_result_t_ const tail = uniform_stride_tail_(input);
    if (tail.tail_dims == input.rank()) {
        normalized_rank1_lane_<value_type_, max_rank_> lane = normalize_rank1_lane_from_tail_<value_type_, max_rank_>(
            input, tail);
        if (status_t status = numkong::reduce_minmax<value_type_>(lane.data, lane.count, lane.stride, &result.min_value,
                                                                  &result.min_index, &result.max_value,
                                                                  &result.max_index, capabilities, stream);
            failed(status))
            return {result, status};
        if (lane.reversed) {
            result.min_index = tail.element_count - 1 - result.min_index;
            result.max_index = tail.element_count - 1 - result.max_index;
        }
        return {result, status_t::success_k};
    }
    if (tail.tail_dims >= 2)
        return minmax<value_type_, max_rank_>(collapse_uniform_tail_(input, tail), capabilities, stream);
    if (input.rank() == 1) {
        status_t status = reduce_rank1_minmax_(input, result, capabilities, stream);
        return {result, status};
    }
    // Slices merge like the halves of a split kernel: an all-NaN slice has no index and never wins
    auto [first_result, first_status] = minmax<value_type_, max_rank_>(input.slice_leading(0), capabilities, stream);
    if (failed(first_status)) return {first_result, first_status};
    result = first_result;
    std::size_t base = 0;
    for (std::size_t i = 1; i < input.extent(0); ++i) {
        tensor_view<value_type_, max_rank_> slice = input.slice_leading(static_cast<std::ptrdiff_t>(i));
        base += slice.numel();
        auto [slice_result, slice_status] = minmax<value_type_, max_rank_>(slice, capabilities, stream);
        if (failed(slice_status)) return {result, slice_status};
        if (slice_result.min_index == NUMKONG_SIZE_MAX) continue;
        if (result.min_index == NUMKONG_SIZE_MAX ||
            sorts_before_(slice_result.min_value, result.min_value, capabilities))
            result.min_value = slice_result.min_value, result.min_index = base + slice_result.min_index;
        if (result.max_index == NUMKONG_SIZE_MAX ||
            sorts_before_(result.max_value, slice_result.max_value, capabilities))
            result.max_value = slice_result.max_value, result.max_index = base + slice_result.max_index;
    }
    return {result, status_t::success_k};
}

/** @c minmax with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<minmax_result<typename value_type_::reduce_minmax_value_t>> minmax(
    tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax<value_type_, max_rank_>(input, default_capabilities(), nullptr);
}

/** Σ of all elements; fails like the scalar @c moments. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<typename value_type_::reduce_moments_sum_t> sum(tensor_view<value_type_, max_rank_> input,
                                                         nk_capability_t capabilities, nk_stream_t stream) noexcept {
    auto [result, status] = moments<value_type_, max_rank_>(input, capabilities, stream);
    return {result.sum, status};
}

/** @c sum with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<typename value_type_::reduce_moments_sum_t> sum(tensor_view<value_type_, max_rank_> input) noexcept {
    return sum<value_type_, max_rank_>(input, default_capabilities(), nullptr);
}

/** The minimum element value; fails like the scalar @c moments. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<typename value_type_::reduce_minmax_value_t>(min)(tensor_view<value_type_, max_rank_> input,
                                                           nk_capability_t capabilities, nk_stream_t stream) noexcept {
    auto [result, status] = minmax<value_type_, max_rank_>(input, capabilities, stream);
    return {result.min_value, status};
}

/** @c min with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<typename value_type_::reduce_minmax_value_t>(min)(tensor_view<value_type_, max_rank_> input) noexcept {
    return numkong::min<value_type_, max_rank_>(input, default_capabilities(), nullptr);
}

/** The maximum element value; fails like the scalar @c moments. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<typename value_type_::reduce_minmax_value_t>(max)(tensor_view<value_type_, max_rank_> input,
                                                           nk_capability_t capabilities, nk_stream_t stream) noexcept {
    auto [result, status] = minmax<value_type_, max_rank_>(input, capabilities, stream);
    return {result.max_value, status};
}

/** @c max with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<typename value_type_::reduce_minmax_value_t>(max)(tensor_view<value_type_, max_rank_> input) noexcept {
    return numkong::max<value_type_, max_rank_>(input, default_capabilities(), nullptr);
}

/** Flat index of the minimum element; fails like the scalar @c moments. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<std::size_t> argmin(tensor_view<value_type_, max_rank_> input, nk_capability_t capabilities,
                             nk_stream_t stream) noexcept {
    auto [result, status] = minmax<value_type_, max_rank_>(input, capabilities, stream);
    return {result.min_index, status};
}

/** @c argmin with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<std::size_t> argmin(tensor_view<value_type_, max_rank_> input) noexcept {
    return argmin<value_type_, max_rank_>(input, default_capabilities(), nullptr);
}

/** Flat index of the maximum element; fails like the scalar @c moments. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<std::size_t> argmax(tensor_view<value_type_, max_rank_> input, nk_capability_t capabilities,
                             nk_stream_t stream) noexcept {
    auto [result, status] = minmax<value_type_, max_rank_>(input, capabilities, stream);
    return {result.max_index, status};
}

/** @c argmax with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8>
expected<std::size_t> argmax(tensor_view<value_type_, max_rank_> input) noexcept {
    return argmax<value_type_, max_rank_>(input, default_capabilities(), nullptr);
}

/** Σxᵢ and Σxᵢ² over one run of @p in_type_; @c unexpected_dimensions_k for a strided run or one
 *  ending mid-value. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> input_type_>
expected<moments_result<typename in_type_::reduce_moments_sum_t, typename in_type_::reduce_moments_sumsq_t>>
reduce_moments(input_type_ const &input, nk_capability_t capabilities = default_capabilities(),
               nk_stream_t stream = nullptr) noexcept {
    auto [values, values_status] = contiguous_values_<in_type_ const>(input);
    std::size_t const dimensions = values.size() * dimensions_per_value<in_type_>();
    moments_result<typename in_type_::reduce_moments_sum_t, typename in_type_::reduce_moments_sumsq_t> result {};
    if (failed(values_status)) return {result, values_status};
    status_t status = reduce_moments<in_type_>(values.data(), dimensions, sizeof(in_type_), &result.sum, &result.sumsq,
                                               capabilities, stream);
    return {result, status};
}

/** Minimum and maximum, with their indices, over one run of @p in_type_; @c unexpected_dimensions_k
 *  for a strided run or one ending mid-value, and @c NUMKONG_SIZE_MAX indices when every value is
 *  NaN, which no minimum or maximum can index. */
template <a_numeric_dtype in_type_, a_vector_of<in_type_> input_type_>
expected<minmax_result<typename in_type_::reduce_minmax_value_t>> reduce_minmax(
    input_type_ const &input, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
    auto [values, values_status] = contiguous_values_<in_type_ const>(input);
    std::size_t const dimensions = values.size() * dimensions_per_value<in_type_>();
    minmax_result<typename in_type_::reduce_minmax_value_t> result {};
    if (failed(values_status)) return {result, values_status};
    status_t status = reduce_minmax<in_type_>(values.data(), dimensions, sizeof(in_type_), &result.min_value,
                                              &result.min_index, &result.max_value, &result.max_index, capabilities,
                                              stream);
    return {result, status};
}

/** Σxᵢ and Σxᵢ² over a vector view, strided ones included; zeros for an empty @p input. */
template <a_numeric_dtype value_type_>
expected<moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t>>
moments(vector_view<value_type_> input, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t> result {};
    if (input.size() == 0) return {result, status_t::success_k};
    status_t status = reduce_moments<value_type_>(input.data(), input.size(),
                                                  static_cast<std::size_t>(input.stride_bytes()), &result.sum,
                                                  &result.sumsq, capabilities, stream);
    return {result, status};
}

/** @c moments with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_>
expected<moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t>>
moments(vector_view<value_type_> input) noexcept {
    return moments<value_type_>(input, default_capabilities(), nullptr);
}

/** Min and max values with their indices over a vector view, strided ones included. */
template <a_numeric_dtype value_type_>
expected<minmax_result<typename value_type_::reduce_minmax_value_t>> minmax(vector_view<value_type_> input,
                                                                            nk_capability_t capabilities,
                                                                            nk_stream_t stream) noexcept {
    minmax_result<typename value_type_::reduce_minmax_value_t> result {{}, NUMKONG_SIZE_MAX, {}, NUMKONG_SIZE_MAX};
    status_t status = reduce_minmax<value_type_>(
        input.data(), input.size(), static_cast<std::size_t>(input.stride_bytes()), &result.min_value,
        &result.min_index, &result.max_value, &result.max_index, capabilities, stream);
    return {result, status};
}

/** @c minmax with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_>
expected<minmax_result<typename value_type_::reduce_minmax_value_t>> minmax(vector_view<value_type_> input) noexcept {
    return minmax<value_type_>(input, default_capabilities(), nullptr);
}

/** Σ of all elements in a vector view. */
template <a_numeric_dtype value_type_>
expected<typename value_type_::reduce_moments_sum_t> sum(vector_view<value_type_> input, nk_capability_t capabilities,
                                                         nk_stream_t stream) noexcept {
    auto [result, status] = moments<value_type_>(input, capabilities, stream);
    return {result.sum, status};
}

/** @c sum with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_>
expected<typename value_type_::reduce_moments_sum_t> sum(vector_view<value_type_> input) noexcept {
    return sum<value_type_>(input, default_capabilities(), nullptr);
}

/** The minimum element value in a vector view. */
template <a_numeric_dtype value_type_>
expected<typename value_type_::reduce_minmax_value_t>(min)(vector_view<value_type_> input, nk_capability_t capabilities,
                                                           nk_stream_t stream) noexcept {
    auto [result, status] = minmax<value_type_>(input, capabilities, stream);
    return {result.min_value, status};
}

/** @c min with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_>
expected<typename value_type_::reduce_minmax_value_t>(min)(vector_view<value_type_> input) noexcept {
    return numkong::min<value_type_>(input, default_capabilities(), nullptr);
}

/** The maximum element value in a vector view. */
template <a_numeric_dtype value_type_>
expected<typename value_type_::reduce_minmax_value_t>(max)(vector_view<value_type_> input, nk_capability_t capabilities,
                                                           nk_stream_t stream) noexcept {
    auto [result, status] = minmax<value_type_>(input, capabilities, stream);
    return {result.max_value, status};
}

/** @c max with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_>
expected<typename value_type_::reduce_minmax_value_t>(max)(vector_view<value_type_> input) noexcept {
    return numkong::max<value_type_>(input, default_capabilities(), nullptr);
}

/** Flat index of the minimum element in a vector view. */
template <a_numeric_dtype value_type_>
expected<std::size_t> argmin(vector_view<value_type_> input, nk_capability_t capabilities,
                             nk_stream_t stream) noexcept {
    auto [result, status] = minmax<value_type_>(input, capabilities, stream);
    return {result.min_index, status};
}

/** @c argmin with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_>
expected<std::size_t> argmin(vector_view<value_type_> input) noexcept {
    return argmin<value_type_>(input, default_capabilities(), nullptr);
}

/** Flat index of the maximum element in a vector view. */
template <a_numeric_dtype value_type_>
expected<std::size_t> argmax(vector_view<value_type_> input, nk_capability_t capabilities,
                             nk_stream_t stream) noexcept {
    auto [result, status] = minmax<value_type_>(input, capabilities, stream);
    return {result.max_index, status};
}

/** @c argmax with the default capabilities on the CPU. */
template <a_numeric_dtype value_type_>
expected<std::size_t> argmax(vector_view<value_type_> input) noexcept {
    return argmax<value_type_>(input, default_capabilities(), nullptr);
}

/** Population count of a packed bit tensor, through @c nk_reduce_moments_u1_best; 0 for an
 *  empty @p input. */
template <std::size_t max_rank_ = 8>
expected<u64_t> popcount(tensor_view<u1x8_t, max_rank_> input, nk_capability_t capabilities,
                         nk_stream_t stream) noexcept {
    return sum<u1x8_t, max_rank_>(input, capabilities, stream);
}

/** @c popcount with the default capabilities on the CPU. */
template <std::size_t max_rank_ = 8>
expected<u64_t> popcount(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return popcount<max_rank_>(input, default_capabilities(), nullptr);
}

/** Whether any bit of the packed bit tensor is set, suffixed @c _set to avoid the @c all slice
 *  marker of <tt>vector.hpp</tt>. */
template <std::size_t max_rank_ = 8>
expected<bool> any_set(tensor_view<u1x8_t, max_rank_> input, nk_capability_t capabilities,
                       nk_stream_t stream) noexcept {
    auto [count, status] = popcount<max_rank_>(input, capabilities, stream);
    return {count.raw_ != 0, status};
}

/** @c any_set with the default capabilities on the CPU. */
template <std::size_t max_rank_ = 8>
expected<bool> any_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return any_set<max_rank_>(input, default_capabilities(), nullptr);
}

/** Whether no bit of the packed bit tensor is set. */
template <std::size_t max_rank_ = 8>
expected<bool> none_set(tensor_view<u1x8_t, max_rank_> input, nk_capability_t capabilities,
                        nk_stream_t stream) noexcept {
    auto [count, status] = popcount<max_rank_>(input, capabilities, stream);
    return {count.raw_ == 0, status};
}

/** @c none_set with the default capabilities on the CPU. */
template <std::size_t max_rank_ = 8>
expected<bool> none_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return none_set<max_rank_>(input, default_capabilities(), nullptr);
}

/** Whether every bit of the packed bit tensor is set. */
template <std::size_t max_rank_ = 8>
expected<bool> all_set(tensor_view<u1x8_t, max_rank_> input, nk_capability_t capabilities,
                       nk_stream_t stream) noexcept {
    auto [count, status] = popcount<max_rank_>(input, capabilities, stream);
    return {count.raw_ == input.numel(), status};
}

/** @c all_set with the default capabilities on the CPU. */
template <std::size_t max_rank_ = 8>
expected<bool> all_set(tensor_view<u1x8_t, max_rank_> input) noexcept {
    return all_set<max_rank_>(input, default_capabilities(), nullptr);
}

/** Population count over a 1D bit-vector view. */
inline expected<u64_t> popcount(vector_view<u1x8_t> input, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    return sum<u1x8_t>(input, capabilities, stream);
}

/** @c popcount with the default capabilities on the CPU. */
inline expected<u64_t> popcount(vector_view<u1x8_t> input) noexcept {
    return popcount(input, default_capabilities(), nullptr);
}

/** Whether any bit of the 1D bit-vector view is set. */
inline expected<bool> any_set(vector_view<u1x8_t> input, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    auto [count, status] = popcount(input, capabilities, stream);
    return {count.raw_ != 0, status};
}

/** @c any_set with the default capabilities on the CPU. */
inline expected<bool> any_set(vector_view<u1x8_t> input) noexcept {
    return any_set(input, default_capabilities(), nullptr);
}

/** Whether no bit of the 1D bit-vector view is set. */
inline expected<bool> none_set(vector_view<u1x8_t> input, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    auto [count, status] = popcount(input, capabilities, stream);
    return {count.raw_ == 0, status};
}

/** @c none_set with the default capabilities on the CPU. */
inline expected<bool> none_set(vector_view<u1x8_t> input) noexcept {
    return none_set(input, default_capabilities(), nullptr);
}

/** Whether every bit of the 1D bit-vector view is set. */
inline expected<bool> all_set(vector_view<u1x8_t> input, nk_capability_t capabilities, nk_stream_t stream) noexcept {
    auto [count, status] = popcount(input, capabilities, stream);
    return {count.raw_ == input.size(), status};
}

/** @c all_set with the default capabilities on the CPU. */
inline expected<bool> all_set(vector_view<u1x8_t> input) noexcept {
    return all_set(input, default_capabilities(), nullptr);
}

#pragma endregion Scalar Reductions

#pragma region Axis Reductions

/** Σ along a single axis; empty for an empty @p input, @c unexpected_dimensions_k for an @p axis
 *  past the rank, @c misaligned_k for an unsupported layout, or the allocation's failure. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_moments_sum_t>>
expected<tensor<typename value_type_::reduce_moments_sum_t, allocator_type_, max_rank_>> sum(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
    allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
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
        status = reduce_moments_axis_packed_(input, axis, sums.value.span(), scratch.value.span(), keep_dims,
                                             capabilities, stream);
    }
    else status = reduce_moments_axis_(input, axis, sums.value.data(), nullptr, capabilities, stream);
    if (failed(status)) return {sum_tensor_t(alloc), status};
    return sums;
}

/** Moments along an axis, Σxᵢ and Σxᵢ² per slice; fails like the axis @c sum. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_moments_sum_t>>
expected<moments_result<tensor<typename value_type_::reduce_moments_sum_t, allocator_type_, max_rank_>,
                        tensor<typename value_type_::reduce_moments_sumsq_t,
                               typename std::allocator_traits<allocator_type_>::template rebind_alloc<
                                   typename value_type_::reduce_moments_sumsq_t>,
                               max_rank_>>>
moments(tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
        allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
        nk_stream_t stream = nullptr) noexcept {
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
        status = reduce_moments_axis_packed_(input, axis, sums.value.span(), sumsqs.value.span(), keep_dims,
                                             capabilities, stream);
    else status = reduce_moments_axis_(input, axis, sums.value.data(), sumsqs.value.data(), capabilities, stream);
    if (failed(status)) return empty_result(status);

    return {result_t {std::move(sums.value), std::move(sumsqs.value)}, status_t::success_k};
}

/** Min and max along an axis; fails like the axis @c sum. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
expected<minmax_result<tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_>>> minmax(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
    allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
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
        status = reduce_minmax_axis_packed_(input, axis, mins.value.span(), maxs.value.span(), keep_dims, capabilities,
                                            stream);
    else
        status = reduce_minmax_axis_(input, axis, mins.value.data(), nullptr, maxs.value.data(), nullptr, capabilities,
                                     stream);
    if (failed(status)) return empty_result(status);
    return {result_t {std::move(mins.value), 0, std::move(maxs.value), 0}, status_t::success_k};
}

/** Argmin along an axis; fails like the axis @c sum, and with @c missing_kernel_k for sub-byte
 *  inputs. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<std::size_t>>
expected<tensor<std::size_t, allocator_type_, max_rank_>> argmin(tensor_view<value_type_, max_rank_> input,
                                                                 std::size_t axis,
                                                                 keep_dims_t keep_dims = collapse_dims_k,
                                                                 allocator_type_ alloc = {},
                                                                 nk_capability_t capabilities = default_capabilities(),
                                                                 nk_stream_t stream = nullptr) noexcept {
    using out_tensor_t = tensor<std::size_t, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    if (axis >= input.rank()) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (!tensor_layout_supported_(input)) return {out_tensor_t(alloc), status_t::misaligned_k};
    if constexpr (dimensions_per_value<value_type_>() > 1) return {out_tensor_t(alloc), status_t::missing_kernel_k};

    auto out_shape = reduced_shape_<std::size_t>(input.shape(), axis, keep_dims);
    auto indices = out_tensor_t::zeros(out_shape.extents, out_shape.rank, alloc);
    if (!indices) return indices;
    if (status_t status = reduce_minmax_axis_(input, axis, nullptr, indices.value.data(), nullptr, nullptr,
                                              capabilities, stream);
        failed(status))
        return {out_tensor_t(alloc), status};
    return indices;
}

/** Argmax along an axis; fails like @c argmin. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<std::size_t>>
expected<tensor<std::size_t, allocator_type_, max_rank_>> argmax(tensor_view<value_type_, max_rank_> input,
                                                                 std::size_t axis,
                                                                 keep_dims_t keep_dims = collapse_dims_k,
                                                                 allocator_type_ alloc = {},
                                                                 nk_capability_t capabilities = default_capabilities(),
                                                                 nk_stream_t stream = nullptr) noexcept {
    using out_tensor_t = tensor<std::size_t, allocator_type_, max_rank_>;
    if (input.empty()) return {out_tensor_t(alloc), status_t::success_k};
    if (axis >= input.rank()) return {out_tensor_t(alloc), status_t::unexpected_dimensions_k};
    if (!tensor_layout_supported_(input)) return {out_tensor_t(alloc), status_t::misaligned_k};
    if constexpr (dimensions_per_value<value_type_>() > 1) return {out_tensor_t(alloc), status_t::missing_kernel_k};

    auto out_shape = reduced_shape_<std::size_t>(input.shape(), axis, keep_dims);
    auto indices = out_tensor_t::zeros(out_shape.extents, out_shape.rank, alloc);
    if (!indices) return indices;
    if (status_t status = reduce_minmax_axis_(input, axis, nullptr, nullptr, nullptr, indices.value.data(),
                                              capabilities, stream);
        failed(status))
        return {out_tensor_t(alloc), status};
    return indices;
}

/** Min along an axis; fails like the axis @c minmax. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
expected<tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_>>(min)(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
    allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
    auto result = minmax<value_type_, max_rank_, allocator_type_>(input, axis, keep_dims, alloc, capabilities, stream);
    return {std::move(result.value.min_value), result.status};
}

/** Max along an axis; fails like the axis @c minmax. */
template <a_numeric_dtype value_type_, std::size_t max_rank_ = 8,
          typename allocator_type_ = aligned_allocator<typename value_type_::reduce_minmax_value_t>>
expected<tensor<typename value_type_::reduce_minmax_value_t, allocator_type_, max_rank_>>(max)(
    tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k,
    allocator_type_ alloc = {}, nk_capability_t capabilities = default_capabilities(),
    nk_stream_t stream = nullptr) noexcept {
    auto result = minmax<value_type_, max_rank_, allocator_type_>(input, axis, keep_dims, alloc, capabilities, stream);
    return {std::move(result.value.max_value), result.status};
}

#pragma endregion Axis Reductions

} // namespace ashvardanian::numkong

#endif // NUMKONG_REDUCE_HPP
