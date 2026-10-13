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

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = settings.dense_dimensions;
    auto input = make_vector<scalar_t>(backend, count), result = make_vector<scalar_t>(backend, count);
    auto coefficients = make_vector<char>(backend, 2 * sizeof(scale_t));
    scale_t *alpha = reinterpret_cast<scale_t *>(coefficients.raw_values_data()), *beta = alpha + 1;

    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        fill_random(settings, generator, input);
        *alpha = random_coef<scalar_t>(generator), *beta = random_coef<scalar_t>(generator);

        backend.call(kernel, input.raw_values_data(), count, alpha, beta, result.raw_values_data());
        if (!stats.expect(backend.synchronize())) return stats;
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

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = settings.dense_dimensions;
    auto a = make_vector<scalar_t>(backend, count), b = make_vector<scalar_t>(backend, count);
    auto result = make_vector<scalar_t>(backend, count);
    auto coefficients = make_vector<char>(backend, 2 * sizeof(scale_t));
    scale_t *alpha = reinterpret_cast<scale_t *>(coefficients.raw_values_data()), *beta = alpha + 1;

    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);
        *alpha = random_coef<scalar_t>(generator), *beta = random_coef<scalar_t>(generator);

        backend.call(kernel, a.raw_values_data(), b.raw_values_data(), count, alpha, beta, result.raw_values_data());
        if (!stats.expect(backend.synchronize())) return stats;
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

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = settings.dense_dimensions;
    auto a = make_vector<scalar_t>(backend, count), b = make_vector<scalar_t>(backend, count);
    auto c = make_vector<scalar_t>(backend, count), result = make_vector<scalar_t>(backend, count);
    auto coefficients = make_vector<char>(backend, 2 * sizeof(scale_t));
    scale_t *alpha = reinterpret_cast<scale_t *>(coefficients.raw_values_data()), *beta = alpha + 1;

    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);
        fill_random(settings, generator, c);
        *alpha = random_coef<scalar_t>(generator), *beta = random_coef<scalar_t>(generator);

        backend.call(kernel, a.raw_values_data(), b.raw_values_data(), c.raw_values_data(), count, alpha, beta,
                     result.raw_values_data());
        if (!stats.expect(backend.synchronize())) return stats;
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

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::approximate_k);
    std::mt19937 generator(settings.seed.value);
    std::size_t const count = settings.dense_dimensions;
    auto a = make_vector<scalar_t>(backend, count), b = make_vector<scalar_t>(backend, count);
    auto result = make_vector<scalar_t>(backend, count);
    auto reference = make_vector<scalar_t>(count);

    for (deadline_t const deadline = settings.deadline(); deadline.ahead();) {
        fill_random(settings, generator, a);
        fill_random(settings, generator, b);

        backend.call(kernel, a.raw_values_data(), b.raw_values_data(), count, result.raw_values_data());
        if (!stats.expect(backend.synchronize())) return stats;
        stats.expect(nk::add<scalar_t>(a.values_data(), b.values_data(), count, reference.values_data(), no_tiers_k));

        for (std::size_t i = 0; i < count; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/** Grouped RMSNorm of @p input_type_ rows into @p scalar_type_ ones, padded, against an F64
 *  reference within the bound of the input's tier: ragged vectors with a learned γ as a pre-norm
 *  and without one as a QK-norm over heads, then whole 16-byte chunks as a QK-norm, as a pre-norm,
 *  and past what registers keep, and rows whose squares leave the tier below the reduction's. A
 *  downcast also meets rows whose exact results pin its rounding, saturation and non-finites. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename input_type_ = scalar_type_,
          typename rmsnorm_kernel_type_>
error_stats_t test_rmsnorm(settings_t const &settings, rmsnorm_kernel_type_ rmsnorm_fn) {
    using scalar_t = scalar_type_;
    using input_t = input_type_;
    struct layout {
        std::size_t groups, columns;
    };

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(comparison_family_t::bounded_k);
    std::mt19937 generator(settings.seed.value);
    std::uniform_real_distribution<float> gain_distribution(0.5f, 1.5f);
    nk_f32_t const epsilon = 1e-6f;
    std::size_t const rows = 33;

    for (deadline_t const deadline = settings.deadline(); deadline.ahead();)
        for (layout const vectors :
             {layout {1, 99}, layout {3, 16385}, layout {8, 128}, layout {1, 2048}, layout {1, 36864}}) {
            std::size_t const groups = vectors.groups, columns = vectors.columns;
            std::size_t const row_values = groups * columns + 16;
            nk_f64_t const bound = nk_each_rmsnorm_error_bound(input_t::dtype(), scalar_t::dtype(), columns);
            auto x = make_vector<input_t>(backend, rows * row_values);
            auto y = make_vector<scalar_t>(backend, rows * row_values);
            auto gamma = make_vector<f32_t>(backend, columns);
            fill_random(settings, generator, x);
            for (std::size_t col = 0; col < columns; col++) gamma[col] = gain_distribution(generator);
            nk_f32_t const *gains = groups == 1 ? gamma.raw_values_data() : nullptr;

            backend.call(rmsnorm_fn, x.raw_values_data(), gains, y.raw_values_data(), rows, groups, columns,
                         row_values * sizeof(input_t), row_values * sizeof(scalar_t), epsilon);
            if (!stats.expect(backend.synchronize())) return stats;

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
                        stats.accumulate_bounded(y[first + col], expected, bound * std::fabs(expected));
                    }
                }
        }

    // Values of ±8 give an inverse RMS within the tier bound of 1/8, so outputs are casts of ±γ or,
    // at a tie, of its neighbour: saturation, overflow, a zero row, and rows a NaN or ∞ poisons
    if constexpr (!std::is_same_v<input_t, scalar_t>) {
        constexpr float edge_gains[] = {0.5f,        1.5f,        2.5f,           -2.5f,    3.5f,   126.5f,
                                        127.5f,      -128.5f,     254.5f,         255.5f,   300.0f, -300.0f,
                                        1.00390625f, 1.01171875f, 1.00048828125f, 1.0625f,  1.125f, 7.25f,
                                        26.0f,       57344.0f,    61440.0f,       65520.0f, 3.0e5f, -0.0f};
        constexpr bool non_finite = input_t::has_infinity() && scalar_t::has_nan();
        std::size_t const columns = 32, edge_rows = 4;
        nk_f64_t const bound = nk_each_rmsnorm_error_bound(input_t::dtype(), scalar_t::dtype(), columns);
        auto x = make_vector<input_t>(backend, edge_rows * columns);
        auto y = make_vector<scalar_t>(backend, edge_rows * columns);
        auto gamma = make_vector<f32_t>(backend, columns);
        for (std::size_t col = 0; col < columns; col++) {
            gamma[col] = edge_gains[col % std::size(edge_gains)];
            x[col] = input_t(input_t::is_signed() && col % 2 ? -8 : 8);
            x[2 * columns + col] = input_t(8), x[3 * columns + col] = input_t(8);
        }
        if constexpr (non_finite) x[2 * columns] = input_t::quiet_nan(), x[3 * columns] = input_t::positive_infinity();
        backend.call(rmsnorm_fn, x.raw_values_data(), gamma.raw_values_data(), y.raw_values_data(), edge_rows, 1,
                     columns, columns * sizeof(input_t), columns * sizeof(scalar_t), epsilon);
        if (!stats.expect(backend.synchronize())) return stats;
        for (std::size_t col = 0; col < columns; col++) {
            nk_f32_t const gain = edge_gains[col % std::size(edge_gains)];
            nk_f64_t const signed_gain = input_t::is_signed() && col % 2 ? -gain : gain;
            nk_f32_t const products[3] = {(nk_f32_t)(signed_gain * (1 - bound)), (nk_f32_t)(signed_gain * (1 + bound)),
                                          0.0f * gain};
            scalar_t expected[3], normalized = y[col], zero = y[columns + col];
            stats.expect(nk_cast_serial(products, nk_f32_k, expected, scalar_t::dtype(), 3, nullptr));
            stats.expect(std::memcmp(&expected[0], &normalized, sizeof(scalar_t)) == 0 ||
                             std::memcmp(&expected[1], &normalized, sizeof(scalar_t)) == 0,
                         "±γ rounds like the serial cast of either bound around it");
            stats.expect(std::memcmp(&expected[2], &zero, sizeof(scalar_t)) == 0, "a zero row stays a signed zero");
            if constexpr (non_finite)
                stats.expect(std::isnan(static_cast<double>(y[2 * columns + col])), "a NaN poisons its vector");
        }
        if constexpr (non_finite)
            stats.expect(std::isnan(static_cast<double>(y[3 * columns])), "an infinity normalizes to NaN");
    }

    // Rows whose squares leave the tier below the reduction's: F32 RMSNorm and F64 RMSCast
    // normalize them like rows near one; F32-tier kernels overflow 2⁷⁰ squares into signed zeros
    constexpr nk_dtype_t reduction_dtype = nk_each_rmsnorm_reduction_dtype(input_t::dtype(), scalar_t::dtype());
    constexpr bool wide_tier = reduction_dtype == nk_f64_k;
    constexpr bool narrow_tier = reduction_dtype == nk_f32_k &&
                                 (std::is_same_v<input_t, bf16_t> || std::is_same_v<input_t, f32_t>);
    if constexpr (wide_tier || narrow_tier) {
        std::vector<double> const magnitudes = std::is_same_v<input_t, f64_t> ? std::vector {1e-160, 1e-170, 1e200}
                                               : wide_tier                    ? std::vector {1e-25, 1e-40, 1e30}
                                                                              : std::vector {0x1p70};
        std::size_t const columns = 77, rows = magnitudes.size();
        nk_f64_t const bound = nk_each_rmsnorm_error_bound(input_t::dtype(), scalar_t::dtype(), columns);
        auto x = make_vector<input_t>(backend, rows * columns);
        auto y = make_vector<scalar_t>(backend, rows * columns);
        std::uniform_real_distribution<double> unit_distribution(0.5, 1.5);
        for (std::size_t row = 0; row < rows; row++)
            for (std::size_t col = 0; col < columns; col++)
                x[row * columns + col] = input_t((col % 3 ? 1 : -1) * unit_distribution(generator) * magnitudes[row]);
        backend.call(rmsnorm_fn, x.raw_values_data(), nullptr, y.raw_values_data(), rows, 1, columns,
                     columns * sizeof(input_t), columns * sizeof(scalar_t), 0.0f);
        if (!stats.expect(backend.synchronize())) return stats;
        for (std::size_t row = 0; row < rows; row++) {
            double sum_squares = 0;
            for (std::size_t col = 0; col < columns; col++)
                sum_squares += std::pow(static_cast<double>(x[row * columns + col]) / magnitudes[row], 2);
            double const inverse_rms = 1 / std::sqrt(sum_squares / columns);
            for (std::size_t col = 0; col < columns; col++) {
                double const output = static_cast<double>(y[row * columns + col]);
                if constexpr (narrow_tier)
                    stats.expect(output == 0 && std::signbit(output) == (col % 3 == 0), "F32 squares overflow to zero");
                else {
                    double const expected = static_cast<double>(x[row * columns + col]) / magnitudes[row] * inverse_rms;
                    stats.accumulate_bounded(y[row * columns + col], expected, bound * std::fabs(expected));
                }
            }
        }
    }
    return stats;
}

/** SwiGLU over a fused @b [rows,2×columns] buffer of gate and up rows against an F64 reference,
 *  then plain SiLU with a NULL @c up. Both carry a non-unit gate and output scale. */
template <typename scalar_type_, typename backend_type_ = host_backend_t, typename swiglu_kernel_type_>
error_stats_t test_swiglu(settings_t const &settings, swiglu_kernel_type_ swiglu_fn) {
    using scalar_t = scalar_type_;

    backend_type_ backend = make_backend<backend_type_>(settings);
    error_stats_t stats(nk_each_swiglu_error_bound(scalar_t::dtype()));
    std::mt19937 generator(settings.seed.value);
    nk_f32_t const gate_scale = 0.25f, output_scale = 2.0f;
    std::size_t const rows = 37;

    for (deadline_t const deadline = settings.deadline(); deadline.ahead();)
        for (bool const gated : {true, false}) {
            std::size_t const columns = gated ? 129 : 130;
            std::size_t const fused_values = 2 * columns + 3, output_values = columns + 5;
            auto fused = make_vector<scalar_t>(backend, rows * fused_values);
            auto y = make_vector<scalar_t>(backend, rows * output_values);
            fill_random(settings, generator, fused);
            auto const *gate = fused.raw_values_data();
            auto const *up = gated ? gate + columns : nullptr;

            backend.call(swiglu_fn, gate, up, y.raw_values_data(), rows, columns, fused_values * sizeof(scalar_t),
                         fused_values * sizeof(scalar_t), output_values * sizeof(scalar_t), gate_scale, output_scale);
            if (!stats.expect(backend.synchronize())) return stats;

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
