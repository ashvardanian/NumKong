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

#include <algorithm> // `std::fill_n`
#include <random>    // `std::uniform_int_distribution`

#include "numkong/reduce.hpp" // `nk::reduce_moments`, `nk::reduce_minmax`

#include "harness.hpp"

namespace ashvardanian::numkong::test {

constexpr std::size_t max_stride_k = 50;

template <typename input_type_, typename backend_type_ = host_backend_t>
error_stats_t test_reduce_moments(settings_t const &settings, typename input_type_::reduce_moments_kernel_t kernel) {
    using sum_t = typename input_type_::reduce_moments_sum_t;
    using sumsq_t = typename input_type_::reduce_moments_sumsq_t;
    using sum_reference_t = bounded_reference_for<input_type_, sum_t>;
    using sumsq_reference_t = bounded_reference_for<input_type_, sumsq_t>;
    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_reduce_moments_error_bound(input_type_::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::uniform_int_distribution<std::size_t> stride_distribution(1, max_stride_k);
    std::size_t const dims_per_value = nk::dimensions_per_value<input_type_>();
    std::size_t const n = nk::divide_round_up(settings.dense_dimensions, dims_per_value) * dims_per_value;
    auto buffer = make_vector<input_type_>(backend, n * (max_stride_k + sizeof(input_type_)));
    auto sum = make_vector<sum_t>(backend, 1);
    auto sumsq = make_vector<sumsq_t>(backend, 1);
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        std::size_t stride = stride_distribution(generator);
        fill_random(settings, generator, buffer);
        backend.call(kernel, buffer.raw_values_data(), n, stride, sum.raw_values_data(), sumsq.raw_values_data());
        if (!stats.expect(backend.synchronize())) return stats;
        sum_reference_t sum_reference;
        sumsq_reference_t sumsq_reference;
        stats.expect(nk::reduce_moments<input_type_, sum_reference_t, sumsq_reference_t>(
            buffer.values_data(), n, stride, &sum_reference, &sumsq_reference, no_tiers_k));
        stats.accumulate(sum[0], sum_reference);
        stats.accumulate(sumsq[0], sumsq_reference);
    }
    return stats;
}

template <typename input_type_, typename backend_type_ = host_backend_t>
error_stats_t test_reduce_minmax(settings_t const &settings, typename input_type_::reduce_minmax_kernel_t kernel) {
    using output_t = typename input_type_::reduce_minmax_value_t;
    static_assert(sizeof(nk_size_t) <= sizeof(nk_u64_t), "indices are stored in U64 slots");
    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(settings.seed.value);
    std::uniform_int_distribution<std::size_t> stride_distribution(1, max_stride_k);
    std::size_t const dims_per_value = nk::dimensions_per_value<input_type_>();
    std::size_t const n = nk::divide_round_up(settings.dense_dimensions, dims_per_value) * dims_per_value;
    auto buffer = make_vector<input_type_>(backend, n * (max_stride_k + sizeof(input_type_)));
    auto extrema = make_vector<output_t>(backend, 2);
    auto indices = make_vector<u64_t>(backend, 2);
    auto *index_values = reinterpret_cast<nk_size_t *>(indices.raw_values_data());
    auto compare = [&](std::size_t stride) {
        backend.call(kernel, buffer.raw_values_data(), n, stride, extrema.raw_values_data() + 0, index_values + 0,
                     extrema.raw_values_data() + 1, index_values + 1);
        if (!stats.expect(backend.synchronize())) return false;
        output_t reference_min, reference_max;
        std::size_t reference_min_index, reference_max_index;
        stats.expect(nk::reduce_minmax<input_type_, output_t>(buffer.values_data(), n, stride, &reference_min,
                                                              &reference_min_index, &reference_max,
                                                              &reference_max_index, no_tiers_k));
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
    if constexpr (nk::nan_capable_dtype<input_type_>) {
        std::fill_n(buffer.values_data(), buffer.size_values(), input_type_::quiet_nan());
        if (!compare(sizeof(input_type_))) return stats;
    }
    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        std::size_t stride = stride_distribution(generator);
        fill_random(settings, generator, buffer);
        if (!compare(stride)) return stats;
    }
    return stats;
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_REDUCE_HPP
