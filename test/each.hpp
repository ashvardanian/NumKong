/**
 *  @file test/each.hpp
 *  @author Ash Vardanian
 *  @date September 30, 2026
 *  @brief Backend-neutral element-wise scenarios, run on the host and on devices alike.
 *
 *  Every scenario is a template over the scalar type, its kernel, and a backend owning where the
 *  operands live and when results become readable, @c host_backend_t by default.
 */
#pragma once
#ifndef NUMKONG_TEST_EACH_HPP
#define NUMKONG_TEST_EACH_HPP

#include <cmath>  // `std::sqrt`, `std::fabs`
#include <random> // `std::uniform_real_distribution`

#include "numkong/each.h" // `nk_each_rmsnorm_error_bound`

#include "harness.hpp"

namespace ashvardanian::numkong::test {

/** Grouped RMSNorm over padded rows against an F64 reference: one group with a learned γ, as a
 *  pre-norm, then three groups with none, as a QK-norm over heads. E4M3 rows fold a descale. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename rmsnorm_kernel_type_>
error_stats_t test_rmsnorm(rmsnorm_kernel_type_ rmsnorm_fn) {
    using scalar_t = scalar_type_;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using gains_t = nk::vector<f32_t, typename backend_type_::template allocator<f32_t>>;

    backend_type_ backend;
    error_stats_t stats(nk_each_rmsnorm_error_bound(scalar_t::dtype()));
    std::mt19937 generator(global_config.seed);
    std::uniform_real_distribution<float> gain_distribution(0.5f, 1.5f);
    nk_f32_t const input_scale = scalar_t::dtype() == nk_e4m3_k ? 0.25f : 1.0f, eps = 1e-6f;
    std::size_t const rows = 33, cols = 100;

    for (auto start = test_start_time(); within_time_budget(start);)
        for (std::size_t const groups : {1, 3}) {
            std::size_t const row_values = groups * cols + 8, row_bytes = row_values * sizeof(scalar_t);
            auto x = scalars_t::zeros(rows * row_values).value, y = scalars_t::zeros(rows * row_values).value;
            auto gamma = gains_t::zeros(cols).value;
            fill_random(generator, x);
            for (std::size_t col = 0; col < cols; col++) gamma.raw_values_data()[col] = gain_distribution(generator);
            nk_f32_t const *gains = groups == 1 ? gamma.raw_values_data() : nullptr;

            backend.call(rmsnorm_fn, x.raw_values_data(), gains, y.raw_values_data(), rows, groups, cols, row_bytes,
                         row_bytes, eps, input_scale);
            if (char const *failure = backend.synchronize()) stats.expect(false, failure);

            for (std::size_t row = 0; row < rows; row++)
                for (std::size_t group = 0; group < groups; group++) {
                    std::size_t const first = row * row_values + group * cols;
                    double sum_squares = 0;
                    for (std::size_t col = 0; col < cols; col++) {
                        double const value = static_cast<double>(x[first + col]) * input_scale;
                        sum_squares += value * value;
                    }
                    double const inverse_rms = 1 / std::sqrt(sum_squares / cols + eps);
                    for (std::size_t col = 0; col < cols; col++) {
                        double const expected = static_cast<double>(x[first + col]) * input_scale * inverse_rms *
                                                (gains ? gains[col] : 1.0f);
                        stats.accumulate_bounded(y[first + col], expected,
                                                 stats.term_error_bound * std::fabs(expected));
                    }
                }
        }
    return stats;
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_EACH_HPP
