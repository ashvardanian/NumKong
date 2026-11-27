/**
 *  @file test/reduce.hpp
 *  @author Ash Vardanian
 *  @date October 1, 2026
 *  @brief Backend-neutral reduction scenarios, run on the host and on devices alike.
 *
 *  Every scenario is a template over the input type and a backend owning where the operands live
 *  and when results become readable, @c host_backend_t by default. Inputs sit a random number of
 *  bytes apart, aligned or not.
 */
#pragma once
#ifndef NUMKONG_TEST_REDUCE_HPP
#define NUMKONG_TEST_REDUCE_HPP

#include <cmath> // `std::ldexp`, `std::ilogb`

#include <algorithm>   // `std::fill_n`
#include <random>      // `std::uniform_int_distribution`
#include <type_traits> // `std::is_same_v`

#include "numkong/reduce.hpp" // `nk::reduce_moments`, `nk::reduce_minmax`

#include "harness.hpp"

namespace ashvardanian::numkong::test {

/** Σ dataᵢ and Σ dataᵢ² over @p count values @p stride bytes apart, saturating in @p sum_type_ and
 *  @p sumsq_type_, the oracle of the moments kernels. */
template <typename in_type_, typename sum_type_, typename sumsq_type_>
void reduce_moments_reference(in_type_ const *data, std::size_t count, std::size_t stride, sum_type_ *sum,
                              sumsq_type_ *sumsq) noexcept {
    sum_type_ running_sum {};
    sumsq_type_ running_sumsq {};
    // Sub-byte views yield raw FP4 codes, so pairs decode through their nibble accessors
    if constexpr (std::is_same_v<in_type_, e2m1x2_t>) {
        char const *bytes = reinterpret_cast<char const *>(data);
        for (std::size_t i = 0; i < count; ++i) {
            e2m1x2_t const pair = *reinterpret_cast<e2m1x2_t const *>(bytes + (i / 2) * stride);
            float const value = (i & 1) ? pair.second() : pair.first();
            running_sum = saturating_add_reference(running_sum, sum_type_(value));
            running_sumsq = saturating_add_reference(running_sumsq, sumsq_type_(value * value));
        }
    }
    else {
        nk::vector_view<in_type_> values(reinterpret_cast<char const *>(data), count, stride);
        for (std::size_t i = 0; i < count; ++i) {
            running_sum = saturating_add_reference(running_sum, values[i]);
            running_sumsq = saturating_fma_reference(values[i], values[i], running_sumsq);
        }
    }
    *sum = running_sum;
    *sumsq = running_sumsq;
}

/** The smallest and largest of @p count values @p stride bytes apart in @p minmax_type_, with their
 *  indices, where the first value that is not NaN takes both sides and an all-NaN run has
 *  @c NUMKONG_SIZE_MAX indices, the oracle of the minmax kernels. */
template <typename in_type_, typename minmax_type_>
void reduce_minmax_reference(in_type_ const *data, std::size_t count, std::size_t stride, minmax_type_ *min_value,
                             std::size_t *min_index, minmax_type_ *max_value, std::size_t *max_index) noexcept {
    std::size_t min_offset = NUMKONG_SIZE_MAX, max_offset = NUMKONG_SIZE_MAX;
    minmax_type_ best_min = nk::finite_max<minmax_type_>(), best_max = nk::finite_min<minmax_type_>();
    if constexpr (nk::an_infinity_capable_dtype<minmax_type_>)
        best_min = minmax_type_::positive_infinity(), best_max = minmax_type_::negative_infinity();
    nk::vector_view<in_type_> values(reinterpret_cast<char const *>(data), count, stride);
    for (std::size_t i = 0; i < count; ++i) {
        minmax_type_ value = minmax_type_(values[i]);
        if (nk::is_nan(value)) continue;
        if (min_offset == NUMKONG_SIZE_MAX || sorts_before_reference(value, best_min)) best_min = value, min_offset = i;
        if (max_offset == NUMKONG_SIZE_MAX || sorts_before_reference(best_max, value)) best_max = value, max_offset = i;
    }
    *min_value = best_min, *max_value = best_max;
    *min_index = min_offset, *max_index = max_offset;
}

constexpr std::size_t max_stride_k = 50;

template <typename input_type_, typename backend_type_ = host_backend_t>
error_stats_t test_reduce_moments(typename input_type_::reduce_moments_kernel_t kernel) {
    using sum_t = typename input_type_::reduce_moments_sum_t;
    using sumsq_t = typename input_type_::reduce_moments_sumsq_t;
    using sum_reference_t = bounded_reference_for<input_type_, sum_t>;
    using sumsq_reference_t = bounded_reference_for<input_type_, sumsq_t>;
    backend_type_ backend = make_backend<backend_type_>();
    error_stats_t stats(nk_reduce_moments_error_bound(input_type_::dtype()));
    split_mix_generator_t generator(case_stream_key());
    std::uniform_int_distribution<std::size_t> stride_distribution(1, max_stride_k);
    std::size_t const dims_per_value = nk::dimensions_per_value<input_type_>();
    std::size_t const n = nk::divide_round_up(test_run.settings.dense_dimensions, dims_per_value) * dims_per_value;
    auto buffer = make_vector<input_type_>(backend, n * (max_stride_k + sizeof(input_type_)));
    auto sum = make_vector<sum_t>(backend, 1);
    auto sumsq = make_vector<sumsq_t>(backend, 1);
    auto compare = [&](backend_vector<input_type_, backend_type_> const &input, std::size_t count, std::size_t stride) {
        backend.call(kernel, input.raw_values_data(), count, stride, sum.raw_values_data(), sumsq.raw_values_data());
        if (!stats.expect_status(backend.synchronize())) return false;
        sum_reference_t sum_reference;
        sumsq_reference_t sumsq_reference;
        reduce_moments_reference<input_type_, sum_reference_t, sumsq_reference_t>(input.values_data(), count, stride,
                                                                                  &sum_reference, &sumsq_reference);
        stats.accumulate(sum[0], sum_reference);
        stats.accumulate(sumsq[0], sumsq_reference);
        return true;
    };
    // Long runs of the largest magnitude overflow any narrow accumulator that is not split in time.
    if constexpr (input_type_::bits_per_dimension() <= 16 && nk::dimensions_per_value<input_type_>() == 1) {
        std::size_t const count = 1'048'577;
        auto extremes = make_vector<input_type_>(backend, count * 8);
        input_type_ value = input_type_::finite_max();
        if constexpr (nk::is_integral_dtype<input_type_>() && nk::is_signed_dtype<input_type_>())
            value = input_type_::finite_min();
        else if constexpr (!nk::is_integral_dtype<input_type_>())
            value = narrow<input_type_>(std::ldexp(1.0, std::ilogb(to_double(value)))); // powers of two sum exactly
        std::fill_n(extremes.values_data(), extremes.size_values(), value);
        if (!compare(extremes, 33, sizeof(input_type_)) || !compare(extremes, count, sizeof(input_type_)) ||
            !compare(extremes, count, 8 * sizeof(input_type_)))
            return stats;
    }
    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, buffer);
        if (!compare(buffer, n, stride_distribution(generator))) return stats;
    }
    return stats;
}

template <typename input_type_, typename backend_type_ = host_backend_t>
error_stats_t test_reduce_minmax(typename input_type_::reduce_minmax_kernel_t kernel) {
    using output_t = typename input_type_::reduce_minmax_value_t;
    static_assert(sizeof(nk_size_t) <= sizeof(nk_u64_t), "indices are stored in U64 slots");
    backend_type_ backend = make_backend<backend_type_>();
    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());
    std::uniform_int_distribution<std::size_t> stride_distribution(1, max_stride_k);
    std::size_t const dims_per_value = nk::dimensions_per_value<input_type_>();
    std::size_t const n = nk::divide_round_up(test_run.settings.dense_dimensions, dims_per_value) * dims_per_value;
    auto buffer = make_vector<input_type_>(backend, n * (max_stride_k + sizeof(input_type_)));
    auto extrema = make_vector<output_t>(backend, 2);
    auto indices = make_vector<u64_t>(backend, 2);
    auto *index_values = reinterpret_cast<nk_size_t *>(indices.raw_values_data());
    auto compare = [&](std::size_t stride) {
        backend.call(kernel, buffer.raw_values_data(), n, stride, extrema.raw_values_data() + 0, index_values + 0,
                     extrema.raw_values_data() + 1, index_values + 1);
        if (!stats.expect_status(backend.synchronize())) return false;
        output_t reference_min, reference_max;
        std::size_t reference_min_index, reference_max_index;
        reduce_minmax_reference<input_type_, output_t>(buffer.values_data(), n, stride, &reference_min,
                                                       &reference_min_index, &reference_max, &reference_max_index);
        stats.accumulate(index_values[0], static_cast<nk_size_t>(reference_min_index));
        stats.accumulate(index_values[1], static_cast<nk_size_t>(reference_max_index));
        if (reference_min_index == NUMKONG_SIZE_MAX) return true; // No index, so the values are only sentinels
        stats.accumulate(extrema[0], reference_min);
        stats.accumulate(extrema[1], reference_max);
        return true;
    };
    // Uniform inputs never win a strict comparison, yet only an all-NaN one lacks an index
    std::fill_n(buffer.values_data(), buffer.size_values(), nk::finite_max<input_type_>());
    if (!compare(sizeof(input_type_))) return stats;
    if constexpr (nk::a_nan_capable_dtype<input_type_>) {
        std::fill_n(buffer.values_data(), buffer.size_values(), input_type_::quiet_nan());
        if (!compare(sizeof(input_type_))) return stats;
    }
    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        std::size_t stride = stride_distribution(generator);
        fill_random(generator, buffer);
        if (!compare(stride)) return stats;
    }
    return stats;
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_REDUCE_HPP
