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

#include <cmath>   // `std::sqrt`, `std::exp`, `std::fabs`, `std::isnan`
#include <cstring> // `std::memcmp`

#include <iterator> // `std::size`
#include <random>   // `std::uniform_real_distribution`

#include "numkong/each.hpp" // `nk::add`, `nk_each_rmsnorm_error_bound`, `nk_each_swiglu_error_bound`

#include "harness.hpp"

namespace ashvardanian::numkong::test {

/** A random α or β for @p scalar_type_, each part in [-2, 2]. */
template <typename scalar_type_, typename generator_type_>
typename scalar_type_::scale_t random_coef(generator_type_ &gen) {
    using scale_t = typename scalar_type_::scale_t;
    if constexpr (scalar_type_::is_complex()) {
        using component_raw_t = typename scalar_type_::component_t::raw_t;
        std::uniform_real_distribution<component_raw_t> dist(-2, 2);
        scale_t coef;
        coef.real = dist(gen), coef.imag = dist(gen);
        return coef;
    }
    else {
        std::uniform_real_distribution<scale_t> dist(scale_t(-2), scale_t(2));
        return dist(gen);
    }
}

/** Unified test for scale: result[i] = alpha * x[i] + beta, with α and β where the kernel reads. */
template <typename scalar_type_, typename backend_type_ = host_backend_t>
error_stats_t test_scale(settings_t const &settings, typename scalar_type_::scale_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using value_t = tracked<reference_for<scalar_t>>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = settings.dense_dimensions;
    auto const values = allocator_of<scalar_t>(backend);
    auto input = scalars_t::zeros(count, values).value, result = scalars_t::zeros(count, values).value;
    auto coefficients = bytes_t::zeros(2 * sizeof(scale_t), allocator_of<char>(backend)).value;
    scale_t *alpha = reinterpret_cast<scale_t *>(coefficients.raw_values_data()), *beta = alpha + 1;

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, input);
        *alpha = random_coef<scalar_t>(generator), *beta = random_coef<scalar_t>(generator);

        if (nk_status_t const status = backend.call(kernel, input.raw_values_data(), count, alpha, beta,
                                                    result.raw_values_data());
            status != nk_success_k) {
            stats.expect(status);
            stats.expect(backend.synchronize());
            return stats;
        }
        if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
            stats.expect(status);
            return stats;
        }
        for (std::size_t i = 0; i < count; i++)
            stats.accumulate(result[i], value_t(input[i]) * value_t(*alpha) + value_t(*beta));
    }
    return stats;
}

/** Unified test for blend: result[i] = alpha * a[i] + beta * b[i], with α and β where the kernel
 *  reads. */
template <typename scalar_type_, typename backend_type_ = host_backend_t>
error_stats_t test_blend(settings_t const &settings, typename scalar_type_::blend_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using value_t = tracked<reference_for<scalar_t>>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = settings.dense_dimensions;
    auto const values = allocator_of<scalar_t>(backend);
    auto a = scalars_t::zeros(count, values).value, b = scalars_t::zeros(count, values).value;
    auto result = scalars_t::zeros(count, values).value;
    auto coefficients = bytes_t::zeros(2 * sizeof(scale_t), allocator_of<char>(backend)).value;
    scale_t *alpha = reinterpret_cast<scale_t *>(coefficients.raw_values_data()), *beta = alpha + 1;

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);
        *alpha = random_coef<scalar_t>(generator), *beta = random_coef<scalar_t>(generator);

        if (nk_status_t const status = backend.call(kernel, a.raw_values_data(), b.raw_values_data(), count, alpha,
                                                    beta, result.raw_values_data());
            status != nk_success_k) {
            stats.expect(status);
            stats.expect(backend.synchronize());
            return stats;
        }
        if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
            stats.expect(status);
            return stats;
        }
        for (std::size_t i = 0; i < count; i++)
            stats.accumulate(result[i], value_t(a[i]) * value_t(*alpha) + value_t(b[i]) * value_t(*beta));
    }
    return stats;
}

/** Unified test for FMA: result[i] = alpha * a[i] * b[i] + beta * c[i], with α and β where the
 *  kernel reads. */
template <typename scalar_type_, typename backend_type_ = host_backend_t>
error_stats_t test_fma(settings_t const &settings, typename scalar_type_::fma_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using value_t = tracked<reference_for<scalar_t>>;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using bytes_t = nk::vector<char, typename backend_type_::template allocator<char>>;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = settings.dense_dimensions;
    auto const values = allocator_of<scalar_t>(backend);
    auto a = scalars_t::zeros(count, values).value, b = scalars_t::zeros(count, values).value;
    auto c = scalars_t::zeros(count, values).value, result = scalars_t::zeros(count, values).value;
    auto coefficients = bytes_t::zeros(2 * sizeof(scale_t), allocator_of<char>(backend)).value;
    scale_t *alpha = reinterpret_cast<scale_t *>(coefficients.raw_values_data()), *beta = alpha + 1;

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);
        fill_random(settings, generator, c);
        *alpha = random_coef<scalar_t>(generator), *beta = random_coef<scalar_t>(generator);

        if (nk_status_t const status = backend.call(kernel, a.raw_values_data(), b.raw_values_data(),
                                                    c.raw_values_data(), count, alpha, beta, result.raw_values_data());
            status != nk_success_k) {
            stats.expect(status);
            stats.expect(backend.synchronize());
            return stats;
        }
        if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
            stats.expect(status);
            return stats;
        }
        for (std::size_t i = 0; i < count; i++)
            stats.accumulate(result[i],
                             value_t(a[i]) * value_t(b[i]) * value_t(*alpha) + value_t(c[i]) * value_t(*beta));
    }
    return stats;
}

/** Element-wise sums against the C++ reference, exact for integers, as a residual add. */
template <typename scalar_type_, typename backend_type_ = host_backend_t>
error_stats_t test_sum(settings_t const &settings, typename scalar_type_::sum_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::approximate_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = settings.dense_dimensions;
    auto const values = allocator_of<scalar_t>(backend);
    auto a = scalars_t::zeros(count, values).value, b = scalars_t::zeros(count, values).value;
    auto result = scalars_t::zeros(count, values).value;
    auto reference = make_vector<scalar_t>(count);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);

        if (nk_status_t const status = backend.call(kernel, a.raw_values_data(), b.raw_values_data(), count,
                                                    result.raw_values_data());
            status != nk_success_k) {
            stats.expect(status);
            stats.expect(backend.synchronize());
            return stats;
        }
        if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
            stats.expect(status);
            return stats;
        }
        stats.expect(nk::add<scalar_t>(a.values_data(), b.values_data(), count, reference.values_data(), no_tiers_k));

        for (std::size_t i = 0; i < count; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/** Grouped RMSNorm of @p input_type_ rows into @p scalar_type_ ones, padded, against an F64
 *  reference: ragged vectors with a learned γ as a pre-norm and without one as a QK-norm over
 *  heads, then whole 16-byte chunks as a QK-norm, as a pre-norm, and past what registers keep.
 *  A downcast also meets rows whose exact results pin its rounding, saturation and non-finites. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename input_type_ = scalar_type_,
          typename rmsnorm_kernel_type_>
error_stats_t test_rmsnorm(settings_t const &settings, rmsnorm_kernel_type_ rmsnorm_fn) {
    using scalar_t = scalar_type_;
    using input_t = input_type_;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;
    using inputs_t = nk::vector<input_t, typename backend_type_::template allocator<input_t>>;
    using gains_t = nk::vector<f32_t, typename backend_type_::template allocator<f32_t>>;
    struct layout {
        std::size_t groups, columns;
    };

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_rmsnorm_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::uniform_real_distribution<float> gain_distribution(0.5f, 1.5f);
    nk_f32_t const epsilon = 1e-6f;
    std::size_t const rows = 33;

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (layout const vectors :
             {layout {1, 99}, layout {3, 16385}, layout {8, 128}, layout {1, 2048}, layout {1, 36864}}) {
            std::size_t const groups = vectors.groups, columns = vectors.columns;
            std::size_t const row_values = groups * columns + 16;
            auto x = inputs_t::zeros(rows * row_values, allocator_of<input_t>(backend)).value;
            auto y = scalars_t::zeros(rows * row_values, allocator_of<scalar_t>(backend)).value;
            auto gamma = gains_t::zeros(columns, allocator_of<f32_t>(backend)).value;
            fill_random(settings, generator, x);
            for (std::size_t col = 0; col < columns; col++) gamma[col] = gain_distribution(generator);
            nk_f32_t const *gains = groups == 1 ? gamma.raw_values_data() : nullptr;

            if (nk_status_t const status = backend.call(rmsnorm_fn, x.raw_values_data(), gains, y.raw_values_data(),
                                                        rows, groups, columns, row_values * sizeof(input_t),
                                                        row_values * sizeof(scalar_t), epsilon);
                status != nk_success_k) {
                stats.expect(status);
                stats.expect(backend.synchronize());
                return stats;
            }
            if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
                stats.expect(status);
                return stats;
            }

            for (std::size_t row = 0; row < rows; row++)
                for (std::size_t group = 0; group < groups; group++) {
                    std::size_t const first = row * row_values + group * columns;
                    double sum_squares = 0;
                    for (std::size_t col = 0; col < columns; col++) {
                        double const value = static_cast<double>(x[first + col]);
                        sum_squares += value * value;
                    }
                    double const inverse_rms = 1 / std::sqrt(sum_squares / columns + epsilon);
                    for (std::size_t col = 0; col < columns; col++) {
                        double const expected = static_cast<double>(x[first + col]) * inverse_rms *
                                                (gains ? gains[col] : 1.0f);
                        stats.accumulate_bounded(y[first + col], expected,
                                                 stats.term_error_bound * std::fabs(expected));
                    }
                }
        }

    // Values of ±8 give every backend an inverse RMS of exactly 1/8, so each output is the cast of
    // ±γ: ties rounding either way, saturation, overflow, a zero row, and rows a NaN or ∞ poisons
    if constexpr (!std::is_same_v<input_t, scalar_t>) {
        constexpr float edge_gains[] = {0.5f,        1.5f,        2.5f,           -2.5f,    3.5f,   126.5f,
                                        127.5f,      -128.5f,     254.5f,         255.5f,   300.0f, -300.0f,
                                        1.00390625f, 1.01171875f, 1.00048828125f, 1.0625f,  1.125f, 7.25f,
                                        26.0f,       57344.0f,    61440.0f,       65520.0f, 3.0e5f, -0.0f};
        constexpr bool non_finite = input_t::has_infinity() && scalar_t::has_nan();
        std::size_t const columns = 32, edge_rows = 4;
        auto x = inputs_t::zeros(edge_rows * columns, allocator_of<input_t>(backend)).value;
        auto y = scalars_t::zeros(edge_rows * columns, allocator_of<scalar_t>(backend)).value;
        auto gamma = gains_t::zeros(columns, allocator_of<f32_t>(backend)).value;
        for (std::size_t col = 0; col < columns; col++) {
            gamma[col] = edge_gains[col % std::size(edge_gains)];
            x[col] = input_t(input_t::is_signed() && col % 2 ? -8 : 8);
            x[2 * columns + col] = input_t(8), x[3 * columns + col] = input_t(8);
        }
        if constexpr (non_finite) x[2 * columns] = input_t::quiet_nan(), x[3 * columns] = input_t::positive_infinity();
        if (nk_status_t const status = backend.call(rmsnorm_fn, x.raw_values_data(), gamma.raw_values_data(),
                                                    y.raw_values_data(), edge_rows, 1, columns,
                                                    columns * sizeof(input_t), columns * sizeof(scalar_t), epsilon);
            status != nk_success_k) {
            stats.expect(status);
            stats.expect(backend.synchronize());
            return stats;
        }
        if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
            stats.expect(status);
            return stats;
        }
        for (std::size_t col = 0; col < columns; col++) {
            nk_f32_t const gain = edge_gains[col % std::size(edge_gains)];
            nk_f32_t const products[2] = {(input_t::is_signed() && col % 2 ? -gain : gain), 0.0f * gain};
            for (std::size_t row = 0; row < 2; row++) {
                scalar_t expected, actual = y[row * columns + col];
                stats.expect(nk_cast_serial(products + row, nk_f32_k, &expected, scalar_t::dtype(), 1, nullptr));
                stats.expect(std::memcmp(&expected, &actual, sizeof(scalar_t)) == 0,
                             row ? "a zero row stays a signed zero" : "±γ rounds like the serial cast");
            }
            if constexpr (non_finite)
                stats.expect(std::isnan(static_cast<double>(y[2 * columns + col])), "a NaN poisons its vector");
        }
        if constexpr (non_finite)
            stats.expect(std::isnan(static_cast<double>(y[3 * columns])), "an infinity normalizes to NaN");
    }
    return stats;
}

/** SwiGLU over a fused @b [rows,2×columns] buffer of gate and up rows against an F64 reference,
 *  then plain SiLU with a NULL @c up. Both carry a non-unit gate and output scale. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename swiglu_kernel_type_>
error_stats_t test_swiglu(settings_t const &settings, swiglu_kernel_type_ swiglu_fn) {
    using scalar_t = scalar_type_;
    using scalars_t = nk::vector<scalar_t, typename backend_type_::template allocator<scalar_t>>;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_swiglu_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    nk_f32_t const gate_scale = 0.25f, output_scale = 2.0f;
    std::size_t const rows = 37;

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;)
        for (bool const gated : {true, false}) {
            std::size_t const columns = gated ? 129 : 130;
            std::size_t const fused_values = 2 * columns + 3, output_values = columns + 5;
            auto const values = allocator_of<scalar_t>(backend);
            auto fused = scalars_t::zeros(rows * fused_values, values).value;
            auto y = scalars_t::zeros(rows * output_values, values).value;
            fill_random(settings, generator, fused);
            auto const *gate = fused.raw_values_data();
            auto const *up = gated ? gate + columns : nullptr;

            if (nk_status_t const status = backend.call(
                    swiglu_fn, gate, up, y.raw_values_data(), rows, columns, fused_values * sizeof(scalar_t),
                    fused_values * sizeof(scalar_t), output_values * sizeof(scalar_t), gate_scale, output_scale);
                status != nk_success_k) {
                stats.expect(status);
                stats.expect(backend.synchronize());
                return stats;
            }
            if (nk_status_t const status = backend.synchronize(); status != nk_success_k) {
                stats.expect(status);
                return stats;
            }

            for (std::size_t row = 0; row < rows; row++)
                for (std::size_t col = 0; col < columns; col++) {
                    double const gate_value = static_cast<double>(fused[row * fused_values + col]) * gate_scale;
                    double expected = gate_value / (1 + std::exp(-gate_value));
                    if (gated) expected *= static_cast<double>(fused[row * fused_values + columns + col]);
                    expected *= output_scale;
                    stats.accumulate_bounded(y[row * output_values + col], expected,
                                             stats.term_error_bound * std::fabs(expected));
                }
        }
    return stats;
}

} // namespace ashvardanian::numkong::test

#endif // NUMKONG_TEST_EACH_HPP
