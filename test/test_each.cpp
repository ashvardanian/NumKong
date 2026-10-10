/**
 *  @brief Elementwise operations tests.
 *  @file test/test_each.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/each.hpp"         // `nk::sum`, `nk::scale`, `nk::blend`, `nk::fma`
#include "numkong/trigonometry.hpp" // `nk::try_sin`, `nk::try_cos`, `nk::try_atan` wrappers

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

/**
 *  @brief Unified test for elementwise sum: result[i] = a[i] + b[i]
 */
template <typename scalar_type_>
error_stats_t test_sum(typename scalar_type_::sum_kernel_t kernel) {
    using scalar_t = scalar_type_;

    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, result.raw_values_data());
        nk::sum<scalar_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                         reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for scale: result[i] = alpha * x[i] + beta
 */
template <typename scalar_type_>
error_stats_t test_scale(typename scalar_type_::scale_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto input = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, input);
        scale_t alpha = random_coef<scalar_t>(generator);
        scale_t beta = random_coef<scalar_t>(generator);

        kernel(input.raw_values_data(), global_config.dense_dimensions, &alpha, &beta, result.raw_values_data());
        nk::scale<scalar_t, reference_t, nk::no_simd_k>(input.values_data(), global_config.dense_dimensions, &alpha,
                                                        &beta, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for blend: result[i] = alpha * a[i] + beta * b[i]
 */
template <typename scalar_type_>
error_stats_t test_blend(typename scalar_type_::blend_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        scale_t alpha = random_coef<scalar_t>(generator);
        scale_t beta = random_coef<scalar_t>(generator);

        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &alpha, &beta,
               result.raw_values_data());
        nk::blend<scalar_t, reference_t, nk::no_simd_k>(
            a.values_data(), b.values_data(), global_config.dense_dimensions, &alpha, &beta, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Unified test for FMA: result[i] = alpha * a[i] * b[i] + beta * c[i]
 */
template <typename scalar_type_>
error_stats_t test_fma(typename scalar_type_::fma_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using scale_t = typename scalar_t::scale_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(nk::is_integral_dtype<scalar_t>() ? comparison_family_t::exact_k
                                                          : comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto c = make_vector<scalar_t>(global_config.dense_dimensions);
    auto result = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, c);
        scale_t alpha = random_coef<scalar_t>(generator);
        scale_t beta = random_coef<scalar_t>(generator);

        kernel(a.raw_values_data(), b.raw_values_data(), c.raw_values_data(), global_config.dense_dimensions, &alpha,
               &beta, result.raw_values_data());
        nk::fma<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                      c.values_data(), &alpha, &beta, reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(result[i], reference[i]);
    }
    return stats;
}

std::uint64_t f16_ulp_distance(f16_t a, f16_t b) noexcept {
    float const a_f32 = a.to_f32(), b_f32 = b.to_f32();
    if (std::isnan(a_f32) || std::isnan(b_f32)) return std::numeric_limits<std::uint64_t>::max();
    auto ordered = [](std::uint16_t bits) noexcept {
        std::uint16_t const magnitude = bits & 0x7FFFu;
        return static_cast<std::uint16_t>((bits & 0x8000u) ? 0x8000u - magnitude : 0x8000u + magnitude);
    };
    std::uint16_t const ordered_a = ordered(a.to_bits());
    std::uint16_t const ordered_b = ordered(b.to_bits());
    return ordered_a >= ordered_b ? static_cast<std::uint64_t>(ordered_a - ordered_b)
                                  : static_cast<std::uint64_t>(ordered_b - ordered_a);
}

void accumulate_f16_comparison(error_stats_t &stats, f16_t actual, f16_t expected) noexcept {
    nk_f64_t const actual_f64 = actual.to_f32(), expected_f64 = expected.to_f32();
    std::uint64_t const ulps = f16_ulp_distance(actual, expected);
    if (ulps == std::numeric_limits<std::uint64_t>::max()) {
        stats.expect(false, "half comparison contains no NaN");
        return;
    }
    nk_f64_t const abs_error = actual_f64 == expected_f64 ? 0.0 : std::fabs(expected_f64 - actual_f64);
    nk_f64_t const rel_error = expected_f64 != 0.0 ? abs_error / std::fabs(expected_f64) : abs_error;

    stats.min_abs_err = std::min(stats.min_abs_err, abs_error);
    stats.max_abs_err = std::max(stats.max_abs_err, abs_error);
    stats.sum_abs_err += abs_error;
    stats.min_rel_err = std::min(stats.min_rel_err, rel_error);
    stats.max_rel_err = std::max(stats.max_rel_err, rel_error);
    stats.sum_rel_err += rel_error;
    stats.min_ulp = std::min(stats.min_ulp, ulps);
    stats.max_ulp = std::max(stats.max_ulp, ulps);
    stats.sum_ulp += f118_t(ulps);
    stats.saw_floating_distance = true;
    ++stats.count;
    if (ulps == 0) ++stats.exact_matches;
}

error_stats_t test_f16_ulp_distance() {
    error_stats_t stats(comparison_family_t::exact_k);
    stats.expect(f16_ulp_distance(f16_t::from_bits(0x3C00), f16_t::from_bits(0x3C01)) == 1,
                 "adjacent positive half values have distance one");
    stats.expect(f16_ulp_distance(f16_t::from_bits(0xBC00), f16_t::from_bits(0xBC01)) == 1,
                 "adjacent negative half values have distance one");
    stats.expect(f16_ulp_distance(f16_t::from_bits(0x0000), f16_t::from_bits(0x0001)) == 1,
                 "zero and the smallest positive half subnormal have distance one");
    stats.expect(f16_ulp_distance(f16_t::from_bits(0x8000), f16_t::from_bits(0x0000)) == 0,
                 "signed half zeros compare equal");
    stats.expect(f16_ulp_distance(f16_t::from_bits(0x8000), f16_t::from_bits(0x0001)) == 1,
                 "negative zero and the smallest positive half subnormal have distance one");
    stats.expect(f16_ulp_distance(f16_t::from_bits(0x0000), f16_t::from_bits(0x8001)) == 1,
                 "positive zero and the smallest negative half subnormal have distance one");
    stats.expect(f16_ulp_distance(f16_t::from_bits(0x8001), f16_t::from_bits(0x0001)) == 2,
                 "the smallest half subnormals across zero have distance two");
    return stats;
}

error_stats_t test_f16_comparison() {
    error_stats_t stats(comparison_family_t::exact_k);
    f16_t const finite = f16_t::from_bits(0x3C00), nan = f16_t::from_bits(0x7E00);

    error_stats_t output_nan(comparison_family_t::approximate_k);
    accumulate_f16_comparison(output_nan, nan, finite);
    stats.expect(should_fail("each_scale_f16", output_nan), "NaN output fails a finite reference");

    error_stats_t reference_nan(comparison_family_t::approximate_k);
    accumulate_f16_comparison(reference_nan, finite, nan);
    stats.expect(should_fail("each_scale_f16", reference_nan), "finite output fails a NaN reference");
    return stats;
}

error_stats_t test_scale_f16_f32_reference(f16_t::scale_kernel_t kernel, nk_size_t n) {
    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto input = make_vector<f16_t>(n);
    auto result = make_vector<f16_t>(n);
    std::vector<nk_f32_t> input_f32(n), reference_f32(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, input);
        nk_f32_t alpha = random_coef<f16_t>(generator);
        nk_f32_t beta = random_coef<f16_t>(generator);
        for (std::size_t i = 0; i != n; ++i) input_f32[i] = input[i].to_f32();

        kernel(input.raw_values_data(), n, &alpha, &beta, result.raw_values_data());
        nk_each_scale_f32(input_f32.data(), n, &alpha, &beta, reference_f32.data());

        for (std::size_t i = 0; i != n; ++i)
            accumulate_f16_comparison(stats, result[i], f16_t::from_f32(reference_f32[i]));
    }
    return stats;
}

error_stats_t test_blend_f16_f32_reference(f16_t::blend_kernel_t kernel, nk_size_t n) {
    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<f16_t>(n), b = make_vector<f16_t>(n);
    auto result = make_vector<f16_t>(n);
    std::vector<nk_f32_t> a_f32(n), b_f32(n), reference_f32(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        nk_f32_t alpha = random_coef<f16_t>(generator);
        nk_f32_t beta = random_coef<f16_t>(generator);
        for (std::size_t i = 0; i != n; ++i) {
            a_f32[i] = a[i].to_f32();
            b_f32[i] = b[i].to_f32();
        }

        kernel(a.raw_values_data(), b.raw_values_data(), n, &alpha, &beta, result.raw_values_data());
        nk_each_blend_f32(a_f32.data(), b_f32.data(), n, &alpha, &beta, reference_f32.data());

        for (std::size_t i = 0; i != n; ++i)
            accumulate_f16_comparison(stats, result[i], f16_t::from_f32(reference_f32[i]));
    }
    return stats;
}

error_stats_t test_fma_f16_f32_reference(f16_t::fma_kernel_t kernel, nk_size_t n) {
    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto a = make_vector<f16_t>(n), b = make_vector<f16_t>(n), c = make_vector<f16_t>(n);
    auto result = make_vector<f16_t>(n);
    std::vector<nk_f32_t> a_f32(n), b_f32(n), c_f32(n), reference_f32(n);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, c);
        nk_f32_t alpha = random_coef<f16_t>(generator);
        nk_f32_t beta = random_coef<f16_t>(generator);
        for (std::size_t i = 0; i != n; ++i) {
            a_f32[i] = a[i].to_f32();
            b_f32[i] = b[i].to_f32();
            c_f32[i] = c[i].to_f32();
        }

        kernel(a.raw_values_data(), b.raw_values_data(), c.raw_values_data(), n, &alpha, &beta,
               result.raw_values_data());
        nk_each_fma_f32(a_f32.data(), b_f32.data(), c_f32.data(), n, &alpha, &beta, reference_f32.data());

        for (std::size_t i = 0; i != n; ++i)
            accumulate_f16_comparison(stats, result[i], f16_t::from_f32(reference_f32[i]));
    }
    return stats;
}

/**
 *  @brief Smoke-test for the tensor-shaped trig wrappers (`nk::try_sin`/`cos`/`atan`).
 *  Runs allocating + into-span variants on a small zero tensor — just exercises the dispatch
 *  paths, not the numerical accuracy (the latter is covered by the kernel tests above).
 */
void test_each() {
    error_stats_section_t check;

    check.section("Elementwise Precision Helpers", nk_cap_serial_k);
    check("f16_ulp_distance", test_f16_ulp_distance);
    check("f16_comparison", test_f16_comparison);

    check.section("Elementwise Operations Serial", nk_cap_serial_k);
    check("each_scale_f32_serial", test_scale<f32_t>, nk_each_scale_f32_serial);
    check("each_sum_f32_serial", test_sum<f32_t>, nk_each_sum_f32_serial);
    check("each_blend_f32_serial", test_blend<f32_t>, nk_each_blend_f32_serial);
    check("each_fma_f32_serial", test_fma<f32_t>, nk_each_fma_f32_serial);
    check("each_scale_e4m3_serial", test_scale<e4m3_t>, nk_each_scale_e4m3_serial);
    check("each_scale_e5m2_serial", test_scale<e5m2_t>, nk_each_scale_e5m2_serial);
    check("each_sum_e4m3_serial", test_sum<e4m3_t>, nk_each_sum_e4m3_serial);
    check("each_sum_e5m2_serial", test_sum<e5m2_t>, nk_each_sum_e5m2_serial);
    check("each_blend_e4m3_serial", test_blend<e4m3_t>, nk_each_blend_e4m3_serial);
    check("each_blend_e5m2_serial", test_blend<e5m2_t>, nk_each_blend_e5m2_serial);
    check("each_fma_e4m3_serial", test_fma<e4m3_t>, nk_each_fma_e4m3_serial);
    check("each_fma_e5m2_serial", test_fma<e5m2_t>, nk_each_fma_e5m2_serial);
    check("each_sum_f32c_serial", test_sum<f32c_t>, nk_each_sum_f32c_serial);
    check("each_sum_f64c_serial", test_sum<f64c_t>, nk_each_sum_f64c_serial);
    check("each_scale_f32c_serial", test_scale<f32c_t>, nk_each_scale_f32c_serial);
    check("each_scale_f64c_serial", test_scale<f64c_t>, nk_each_scale_f64c_serial);
    check("each_blend_f32c_serial", test_blend<f32c_t>, nk_each_blend_f32c_serial);
    check("each_blend_f64c_serial", test_blend<f64c_t>, nk_each_blend_f64c_serial);
    check("each_fma_f32c_serial", test_fma<f32c_t>, nk_each_fma_f32c_serial);
    check("each_fma_f64c_serial", test_fma<f64c_t>, nk_each_fma_f64c_serial);
    check("each_blend_f16_serial", test_blend<f16_t>, nk_each_blend_f16_serial);
    check("each_blend_i8_serial", test_blend<i8_t>, nk_each_blend_i8_serial);
    check("each_blend_u8_serial", test_blend<u8_t>, nk_each_blend_u8_serial);
    check("each_fma_f16_serial", test_fma<f16_t>, nk_each_fma_f16_serial);
    check("each_fma_i8_serial", test_fma<i8_t>, nk_each_fma_i8_serial);
    check("each_fma_u8_serial", test_fma<u8_t>, nk_each_fma_u8_serial);
    check("each_sum_f64_serial", test_sum<f64_t>, nk_each_sum_f64_serial);
    check("each_scale_f64_serial", test_scale<f64_t>, nk_each_scale_f64_serial);
    check("each_blend_f64_serial", test_blend<f64_t>, nk_each_blend_f64_serial);
    check("each_fma_f64_serial", test_fma<f64_t>, nk_each_fma_f64_serial);
    check("each_sum_bf16_serial", test_sum<bf16_t>, nk_each_sum_bf16_serial);
    check("each_scale_bf16_serial", test_scale<bf16_t>, nk_each_scale_bf16_serial);
    check("each_blend_bf16_serial", test_blend<bf16_t>, nk_each_blend_bf16_serial);
    check("each_fma_bf16_serial", test_fma<bf16_t>, nk_each_fma_bf16_serial);
    check("each_sum_f16_serial", test_sum<f16_t>, nk_each_sum_f16_serial);
    check("each_scale_f16_serial", test_scale<f16_t>, nk_each_scale_f16_serial);

#if NK_DYNAMIC_DISPATCH
    check.section("Elementwise Operations Dynamic", nk_cap_serial_k);
    // Dynamic dispatch - only test the dispatcher itself
    check("each_scale_f32", test_scale<f32_t>, nk_each_scale_f32);
    check("each_sum_f32", test_sum<f32_t>, nk_each_sum_f32);
    check("each_blend_f32", test_blend<f32_t>, nk_each_blend_f32);
    check("each_fma_f32", test_fma<f32_t>, nk_each_fma_f32);
    check("each_scale_f16_dynamic", test_scale_f16_f32_reference, nk_each_scale_f16,
          static_cast<nk_size_t>(global_config.dense_dimensions));
    check("each_scale_f16_dynamic_tail_only", test_scale_f16_f32_reference, nk_each_scale_f16, nk_size_t(7));
    check("each_scale_f16_dynamic_tail_after_vector", test_scale_f16_f32_reference, nk_each_scale_f16, nk_size_t(9));
    check("each_blend_f16_dynamic", test_blend_f16_f32_reference, nk_each_blend_f16,
          static_cast<nk_size_t>(global_config.dense_dimensions));
    check("each_blend_f16_dynamic_tail_only", test_blend_f16_f32_reference, nk_each_blend_f16, nk_size_t(7));
    check("each_blend_f16_dynamic_tail_after_vector", test_blend_f16_f32_reference, nk_each_blend_f16, nk_size_t(9));
    check("each_fma_f16_dynamic", test_fma_f16_f32_reference, nk_each_fma_f16,
          static_cast<nk_size_t>(global_config.dense_dimensions));
    check("each_fma_f16_dynamic_tail_only", test_fma_f16_f32_reference, nk_each_fma_f16, nk_size_t(7));
    check("each_fma_f16_dynamic_tail_after_vector", test_fma_f16_f32_reference, nk_each_fma_f16, nk_size_t(9));
    check("each_scale_e4m3", test_scale<e4m3_t>, nk_each_scale_e4m3);
    check("each_scale_e5m2", test_scale<e5m2_t>, nk_each_scale_e5m2);
    check("each_sum_e4m3", test_sum<e4m3_t>, nk_each_sum_e4m3);
    check("each_sum_e5m2", test_sum<e5m2_t>, nk_each_sum_e5m2);
    check("each_blend_e4m3", test_blend<e4m3_t>, nk_each_blend_e4m3);
    check("each_blend_e5m2", test_blend<e5m2_t>, nk_each_blend_e5m2);
    check("each_fma_e4m3", test_fma<e4m3_t>, nk_each_fma_e4m3);
    check("each_fma_e5m2", test_fma<e5m2_t>, nk_each_fma_e5m2);
    check("each_sum_f32c", test_sum<f32c_t>, nk_each_sum_f32c);
    check("each_sum_f64c", test_sum<f64c_t>, nk_each_sum_f64c);
    check("each_scale_f32c", test_scale<f32c_t>, nk_each_scale_f32c);
    check("each_scale_f64c", test_scale<f64c_t>, nk_each_scale_f64c);
    check("each_blend_f32c", test_blend<f32c_t>, nk_each_blend_f32c);
    check("each_blend_f64c", test_blend<f64c_t>, nk_each_blend_f64c);
    check("each_fma_f32c", test_fma<f32c_t>, nk_each_fma_f32c);
    check("each_fma_f64c", test_fma<f64c_t>, nk_each_fma_f64c);
#endif

    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    check.section("Elementwise Operations NEON", nk_cap_neon_k);
    // f64
    check("each_sum_f64_neon", test_sum<f64_t>, nk_each_sum_f64_neon);
    check("each_scale_f64_neon", test_scale<f64_t>, nk_each_scale_f64_neon);
    check("each_blend_f64_neon", test_blend<f64_t>, nk_each_blend_f64_neon);
    check("each_fma_f64_neon", test_fma<f64_t>, nk_each_fma_f64_neon);
    // f32
    check("each_sum_f32_neon", test_sum<f32_t>, nk_each_sum_f32_neon);
    check("each_scale_f32_neon", test_scale<f32_t>, nk_each_scale_f32_neon);
    check("each_blend_f32_neon", test_blend<f32_t>, nk_each_blend_f32_neon);
    check("each_fma_f32_neon", test_fma<f32_t>, nk_each_fma_f32_neon);
    // e4m3, e5m2
    check("each_sum_e4m3_neon", test_sum<e4m3_t>, nk_each_sum_e4m3_neon);
    check("each_scale_e4m3_neon", test_scale<e4m3_t>, nk_each_scale_e4m3_neon);
    check("each_blend_e4m3_neon", test_blend<e4m3_t>, nk_each_blend_e4m3_neon);
    check("each_fma_e4m3_neon", test_fma<e4m3_t>, nk_each_fma_e4m3_neon);
    check("each_sum_e5m2_neon", test_sum<e5m2_t>, nk_each_sum_e5m2_neon);
    check("each_scale_e5m2_neon", test_scale<e5m2_t>, nk_each_scale_e5m2_neon);
    check("each_blend_e5m2_neon", test_blend<e5m2_t>, nk_each_blend_e5m2_neon);
    check("each_fma_e5m2_neon", test_fma<e5m2_t>, nk_each_fma_e5m2_neon);
    // u8, i8
    check("each_sum_u8_neon", test_sum<u8_t>, nk_each_sum_u8_neon);
    check("each_sum_i8_neon", test_sum<i8_t>, nk_each_sum_i8_neon);
    // i16, u16
    check("each_sum_i16_neon", test_sum<i16_t>, nk_each_sum_i16_neon);
    check("each_scale_i16_neon", test_scale<i16_t>, nk_each_scale_i16_neon);
    check("each_fma_i16_neon", test_fma<i16_t>, nk_each_fma_i16_neon);
    check("each_sum_u16_neon", test_sum<u16_t>, nk_each_sum_u16_neon);
    check("each_scale_u16_neon", test_scale<u16_t>, nk_each_scale_u16_neon);
    check("each_fma_u16_neon", test_fma<u16_t>, nk_each_fma_u16_neon);
    // i32, u32
    check("each_sum_i32_neon", test_sum<i32_t>, nk_each_sum_i32_neon);
    check("each_scale_i32_neon", test_scale<i32_t>, nk_each_scale_i32_neon);
    check("each_fma_i32_neon", test_fma<i32_t>, nk_each_fma_i32_neon);
    check("each_sum_u32_neon", test_sum<u32_t>, nk_each_sum_u32_neon);
    check("each_scale_u32_neon", test_scale<u32_t>, nk_each_scale_u32_neon);
    check("each_fma_u32_neon", test_fma<u32_t>, nk_each_fma_u32_neon);
    // i64, u64
    check("each_sum_i64_neon", test_sum<i64_t>, nk_each_sum_i64_neon);
    check("each_scale_i64_neon", test_scale<i64_t>, nk_each_scale_i64_neon);
    check("each_fma_i64_neon", test_fma<i64_t>, nk_each_fma_i64_neon);
    check("each_sum_u64_neon", test_sum<u64_t>, nk_each_sum_u64_neon);
    check("each_scale_u64_neon", test_scale<u64_t>, nk_each_scale_u64_neon);
    check("each_fma_u64_neon", test_fma<u64_t>, nk_each_fma_u64_neon);
    // complex
    check("each_scale_f32c_neon", test_scale<f32c_t>, nk_each_scale_f32c_neon);
    check("each_blend_f32c_neon", test_blend<f32c_t>, nk_each_blend_f32c_neon);
    check("each_fma_f32c_neon", test_fma<f32c_t>, nk_each_fma_f32c_neon);
    check("each_scale_f64c_neon", test_scale<f64c_t>, nk_each_scale_f64c_neon);
    check("each_blend_f64c_neon", test_blend<f64c_t>, nk_each_blend_f64c_neon);
    check("each_fma_f64c_neon", test_fma<f64c_t>, nk_each_fma_f64c_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
    check.section("Elementwise Operations NEON HALF", nk_cap_neonhalf_k);
    check("each_scale_f16_neonhalf", test_scale_f16_f32_reference, nk_each_scale_f16_neonhalf,
          static_cast<nk_size_t>(global_config.dense_dimensions));
    check("each_scale_f16_neonhalf_tail_only", test_scale_f16_f32_reference, nk_each_scale_f16_neonhalf, nk_size_t(7));
    check("each_scale_f16_neonhalf_tail_after_vector", test_scale_f16_f32_reference, nk_each_scale_f16_neonhalf,
          nk_size_t(9));
    check("each_sum_f16_neonhalf", test_sum<f16_t>, nk_each_sum_f16_neonhalf);
    check("each_blend_f16_neonhalf", test_blend_f16_f32_reference, nk_each_blend_f16_neonhalf,
          static_cast<nk_size_t>(global_config.dense_dimensions));
    check("each_blend_f16_neonhalf_tail_only", test_blend_f16_f32_reference, nk_each_blend_f16_neonhalf, nk_size_t(7));
    check("each_blend_f16_neonhalf_tail_after_vector", test_blend_f16_f32_reference, nk_each_blend_f16_neonhalf,
          nk_size_t(9));
    check("each_fma_f16_neonhalf", test_fma_f16_f32_reference, nk_each_fma_f16_neonhalf,
          static_cast<nk_size_t>(global_config.dense_dimensions));
    check("each_fma_f16_neonhalf_tail_only", test_fma_f16_f32_reference, nk_each_fma_f16_neonhalf, nk_size_t(7));
    check("each_fma_f16_neonhalf_tail_after_vector", test_fma_f16_f32_reference, nk_each_fma_f16_neonhalf,
          nk_size_t(9));
    check("each_scale_u8_neonhalf", test_scale<u8_t>, nk_each_scale_u8_neonhalf);
    check("each_blend_u8_neonhalf", test_blend<u8_t>, nk_each_blend_u8_neonhalf);
    check("each_scale_i8_neonhalf", test_scale<i8_t>, nk_each_scale_i8_neonhalf);
    check("each_blend_i8_neonhalf", test_blend<i8_t>, nk_each_blend_i8_neonhalf);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
    check.section("Elementwise Operations NEON BF16", nk_cap_neonbfdot_k);
    check("each_scale_bf16_neonbfdot", test_scale<bf16_t>, nk_each_scale_bf16_neonbfdot);
    check("each_sum_bf16_neonbfdot", test_sum<bf16_t>, nk_each_sum_bf16_neonbfdot);
    check("each_blend_bf16_neonbfdot", test_blend<bf16_t>, nk_each_blend_bf16_neonbfdot);
    check("each_fma_bf16_neonbfdot", test_fma<bf16_t>, nk_each_fma_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_HASWELL
    check.section("Elementwise Operations Haswell", nk_cap_haswell_k);
    check("each_scale_f32_haswell", test_scale<f32_t>, nk_each_scale_f32_haswell);
    check("each_sum_f32_haswell", test_sum<f32_t>, nk_each_sum_f32_haswell);
    check("each_blend_f32_haswell", test_blend<f32_t>, nk_each_blend_f32_haswell);
    check("each_fma_f32_haswell", test_fma<f32_t>, nk_each_fma_f32_haswell);
    check("each_scale_e4m3_haswell", test_scale<e4m3_t>, nk_each_scale_e4m3_haswell);
    check("each_scale_e5m2_haswell", test_scale<e5m2_t>, nk_each_scale_e5m2_haswell);
    check("each_sum_e4m3_haswell", test_sum<e4m3_t>, nk_each_sum_e4m3_haswell);
    check("each_sum_e5m2_haswell", test_sum<e5m2_t>, nk_each_sum_e5m2_haswell);
    check("each_blend_e4m3_haswell", test_blend<e4m3_t>, nk_each_blend_e4m3_haswell);
    check("each_blend_e5m2_haswell", test_blend<e5m2_t>, nk_each_blend_e5m2_haswell);
    check("each_fma_e4m3_haswell", test_fma<e4m3_t>, nk_each_fma_e4m3_haswell);
    check("each_fma_e5m2_haswell", test_fma<e5m2_t>, nk_each_fma_e5m2_haswell);
    check("each_scale_f32c_haswell", test_scale<f32c_t>, nk_each_scale_f32c_haswell);
    check("each_scale_f64c_haswell", test_scale<f64c_t>, nk_each_scale_f64c_haswell);
    check("each_blend_f32c_haswell", test_blend<f32c_t>, nk_each_blend_f32c_haswell);
    check("each_blend_f64c_haswell", test_blend<f64c_t>, nk_each_blend_f64c_haswell);
    check("each_fma_f32c_haswell", test_fma<f32c_t>, nk_each_fma_f32c_haswell);
    check("each_fma_f64c_haswell", test_fma<f64c_t>, nk_each_fma_f64c_haswell);
    check("each_blend_bf16_haswell", test_blend<bf16_t>, nk_each_blend_bf16_haswell);
    check("each_blend_f64_haswell", test_blend<f64_t>, nk_each_blend_f64_haswell);
    check("each_blend_i8_haswell", test_blend<i8_t>, nk_each_blend_i8_haswell);
    check("each_blend_u8_haswell", test_blend<u8_t>, nk_each_blend_u8_haswell);
    check("each_blend_f16_haswell", test_blend<f16_t>, nk_each_blend_f16_haswell);
    check("each_fma_bf16_haswell", test_fma<bf16_t>, nk_each_fma_bf16_haswell);
    check("each_fma_f64_haswell", test_fma<f64_t>, nk_each_fma_f64_haswell);
    check("each_fma_i16_haswell", test_fma<i16_t>, nk_each_fma_i16_haswell);
    check("each_fma_i8_haswell", test_fma<i8_t>, nk_each_fma_i8_haswell);
    check("each_fma_u16_haswell", test_fma<u16_t>, nk_each_fma_u16_haswell);
    check("each_fma_u8_haswell", test_fma<u8_t>, nk_each_fma_u8_haswell);
    check("each_fma_f16_haswell", test_fma<f16_t>, nk_each_fma_f16_haswell);
    check("each_scale_bf16_haswell", test_scale<bf16_t>, nk_each_scale_bf16_haswell);
    check("each_scale_f16_haswell", test_scale<f16_t>, nk_each_scale_f16_haswell);
    check("each_scale_f64_haswell", test_scale<f64_t>, nk_each_scale_f64_haswell);
    check("each_scale_i16_haswell", test_scale<i16_t>, nk_each_scale_i16_haswell);
    check("each_scale_i8_haswell", test_scale<i8_t>, nk_each_scale_i8_haswell);
    check("each_scale_u16_haswell", test_scale<u16_t>, nk_each_scale_u16_haswell);
    check("each_scale_u8_haswell", test_scale<u8_t>, nk_each_scale_u8_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check.section("Elementwise Operations Skylake", nk_cap_skylake_k);
    check("each_scale_f32_skylake", test_scale<f32_t>, nk_each_scale_f32_skylake);
    check("each_sum_f32_skylake", test_sum<f32_t>, nk_each_sum_f32_skylake);
    check("each_blend_f32_skylake", test_blend<f32_t>, nk_each_blend_f32_skylake);
    check("each_fma_f32_skylake", test_fma<f32_t>, nk_each_fma_f32_skylake);
    check("each_scale_e4m3_skylake", test_scale<e4m3_t>, nk_each_scale_e4m3_skylake);
    check("each_scale_e5m2_skylake", test_scale<e5m2_t>, nk_each_scale_e5m2_skylake);
    check("each_sum_e4m3_skylake", test_sum<e4m3_t>, nk_each_sum_e4m3_skylake);
    check("each_sum_e5m2_skylake", test_sum<e5m2_t>, nk_each_sum_e5m2_skylake);
    check("each_blend_e4m3_skylake", test_blend<e4m3_t>, nk_each_blend_e4m3_skylake);
    check("each_blend_e5m2_skylake", test_blend<e5m2_t>, nk_each_blend_e5m2_skylake);
    check("each_fma_e4m3_skylake", test_fma<e4m3_t>, nk_each_fma_e4m3_skylake);
    check("each_fma_e5m2_skylake", test_fma<e5m2_t>, nk_each_fma_e5m2_skylake);
    check("each_scale_f32c_skylake", test_scale<f32c_t>, nk_each_scale_f32c_skylake);
    check("each_scale_f64c_skylake", test_scale<f64c_t>, nk_each_scale_f64c_skylake);
    check("each_blend_f32c_skylake", test_blend<f32c_t>, nk_each_blend_f32c_skylake);
    check("each_blend_f64c_skylake", test_blend<f64c_t>, nk_each_blend_f64c_skylake);
    check("each_fma_f32c_skylake", test_fma<f32c_t>, nk_each_fma_f32c_skylake);
    check("each_fma_f64c_skylake", test_fma<f64c_t>, nk_each_fma_f64c_skylake);
    check("each_scale_f16_skylake", test_scale<f16_t>, nk_each_scale_f16_skylake);
    check("each_blend_f16_skylake", test_blend<f16_t>, nk_each_blend_f16_skylake);
    check("each_fma_f16_skylake", test_fma<f16_t>, nk_each_fma_f16_skylake);
    check("each_blend_bf16_skylake", test_blend<bf16_t>, nk_each_blend_bf16_skylake);
    check("each_blend_f64_skylake", test_blend<f64_t>, nk_each_blend_f64_skylake);
    check("each_fma_bf16_skylake", test_fma<bf16_t>, nk_each_fma_bf16_skylake);
    check("each_fma_f64_skylake", test_fma<f64_t>, nk_each_fma_f64_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
    check.section("Elementwise Operations Ice Lake", nk_cap_icelake_k);
    check("each_sum_i8_icelake", test_sum<i8_t>, nk_each_sum_i8_icelake);
    check("each_sum_u8_icelake", test_sum<u8_t>, nk_each_sum_u8_icelake);
    check("each_sum_i16_icelake", test_sum<i16_t>, nk_each_sum_i16_icelake);
    check("each_sum_u16_icelake", test_sum<u16_t>, nk_each_sum_u16_icelake);
    check("each_sum_i32_icelake", test_sum<i32_t>, nk_each_sum_i32_icelake);
    check("each_sum_u32_icelake", test_sum<u32_t>, nk_each_sum_u32_icelake);
    check("each_sum_i64_icelake", test_sum<i64_t>, nk_each_sum_i64_icelake);
    check("each_sum_u64_icelake", test_sum<u64_t>, nk_each_sum_u64_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_SAPPHIRE
    check.section("Elementwise Operations Sapphire", nk_cap_sapphire_k);
    check("each_sum_f16_sapphire", test_sum<f16_t>, nk_each_sum_f16_sapphire);
    check("each_scale_u8_sapphire", test_scale<u8_t>, nk_each_scale_u8_sapphire);
    check("each_blend_u8_sapphire", test_blend<u8_t>, nk_each_blend_u8_sapphire);
    check("each_scale_i8_sapphire", test_scale<i8_t>, nk_each_scale_i8_sapphire);
    check("each_blend_i8_sapphire", test_blend<i8_t>, nk_each_blend_i8_sapphire);
    check("each_sum_e4m3_sapphire", test_sum<e4m3_t>, nk_each_sum_e4m3_sapphire);
#endif // NK_TARGET_SAPPHIRE

#if NK_TARGET_RVV
    check.section("Elementwise Operations RVV", nk_cap_rvv_k);
    check("each_sum_f64_rvv", test_sum<f64_t>, nk_each_sum_f64_rvv);
    check("each_scale_f64_rvv", test_scale<f64_t>, nk_each_scale_f64_rvv);
    check("each_blend_f64_rvv", test_blend<f64_t>, nk_each_blend_f64_rvv);
    check("each_fma_f64_rvv", test_fma<f64_t>, nk_each_fma_f64_rvv);
    check("each_sum_f32_rvv", test_sum<f32_t>, nk_each_sum_f32_rvv);
    check("each_scale_f32_rvv", test_scale<f32_t>, nk_each_scale_f32_rvv);
    check("each_blend_f32_rvv", test_blend<f32_t>, nk_each_blend_f32_rvv);
    check("each_fma_f32_rvv", test_fma<f32_t>, nk_each_fma_f32_rvv);
    check("each_sum_f16_rvv", test_sum<f16_t>, nk_each_sum_f16_rvv);
    check("each_scale_f16_rvv", test_scale<f16_t>, nk_each_scale_f16_rvv);
    check("each_blend_f16_rvv", test_blend<f16_t>, nk_each_blend_f16_rvv);
    check("each_fma_f16_rvv", test_fma<f16_t>, nk_each_fma_f16_rvv);
    check("each_sum_bf16_rvv", test_sum<bf16_t>, nk_each_sum_bf16_rvv);
    check("each_scale_bf16_rvv", test_scale<bf16_t>, nk_each_scale_bf16_rvv);
    check("each_blend_bf16_rvv", test_blend<bf16_t>, nk_each_blend_bf16_rvv);
    check("each_fma_bf16_rvv", test_fma<bf16_t>, nk_each_fma_bf16_rvv);
    check("each_sum_e4m3_rvv", test_sum<e4m3_t>, nk_each_sum_e4m3_rvv);
    check("each_scale_e4m3_rvv", test_scale<e4m3_t>, nk_each_scale_e4m3_rvv);
    check("each_blend_e4m3_rvv", test_blend<e4m3_t>, nk_each_blend_e4m3_rvv);
    check("each_fma_e4m3_rvv", test_fma<e4m3_t>, nk_each_fma_e4m3_rvv);
    check("each_sum_e5m2_rvv", test_sum<e5m2_t>, nk_each_sum_e5m2_rvv);
    check("each_scale_e5m2_rvv", test_scale<e5m2_t>, nk_each_scale_e5m2_rvv);
    check("each_blend_e5m2_rvv", test_blend<e5m2_t>, nk_each_blend_e5m2_rvv);
    check("each_fma_e5m2_rvv", test_fma<e5m2_t>, nk_each_fma_e5m2_rvv);
    check("each_sum_i8_rvv", test_sum<i8_t>, nk_each_sum_i8_rvv);
    check("each_scale_i8_rvv", test_scale<i8_t>, nk_each_scale_i8_rvv);
    check("each_blend_i8_rvv", test_blend<i8_t>, nk_each_blend_i8_rvv);
    check("each_fma_i8_rvv", test_fma<i8_t>, nk_each_fma_i8_rvv);
    check("each_sum_u8_rvv", test_sum<u8_t>, nk_each_sum_u8_rvv);
    check("each_scale_u8_rvv", test_scale<u8_t>, nk_each_scale_u8_rvv);
    check("each_blend_u8_rvv", test_blend<u8_t>, nk_each_blend_u8_rvv);
    check("each_fma_u8_rvv", test_fma<u8_t>, nk_each_fma_u8_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    check.section("Elementwise Operations V128 Relaxed", nk_cap_v128relaxed_k);
    check("each_sum_f32_v128relaxed", test_sum<f32_t>, nk_each_sum_f32_v128relaxed);
    check("each_scale_f32_v128relaxed", test_scale<f32_t>, nk_each_scale_f32_v128relaxed);
    check("each_blend_f32_v128relaxed", test_blend<f32_t>, nk_each_blend_f32_v128relaxed);
    check("each_fma_f32_v128relaxed", test_fma<f32_t>, nk_each_fma_f32_v128relaxed);
    check("each_sum_f16_v128relaxed", test_sum<f16_t>, nk_each_sum_f16_v128relaxed);
    check("each_scale_f16_v128relaxed", test_scale<f16_t>, nk_each_scale_f16_v128relaxed);
    check("each_blend_f16_v128relaxed", test_blend<f16_t>, nk_each_blend_f16_v128relaxed);
    check("each_fma_f16_v128relaxed", test_fma<f16_t>, nk_each_fma_f16_v128relaxed);
    check("each_sum_bf16_v128relaxed", test_sum<bf16_t>, nk_each_sum_bf16_v128relaxed);
    check("each_scale_bf16_v128relaxed", test_scale<bf16_t>, nk_each_scale_bf16_v128relaxed);
    check("each_blend_bf16_v128relaxed", test_blend<bf16_t>, nk_each_blend_bf16_v128relaxed);
    check("each_fma_bf16_v128relaxed", test_fma<bf16_t>, nk_each_fma_bf16_v128relaxed);
    check("each_sum_i8_v128relaxed", test_sum<i8_t>, nk_each_sum_i8_v128relaxed);
    check("each_scale_i8_v128relaxed", test_scale<i8_t>, nk_each_scale_i8_v128relaxed);
    check("each_blend_i8_v128relaxed", test_blend<i8_t>, nk_each_blend_i8_v128relaxed);
    check("each_fma_i8_v128relaxed", test_fma<i8_t>, nk_each_fma_i8_v128relaxed);
    check("each_sum_u8_v128relaxed", test_sum<u8_t>, nk_each_sum_u8_v128relaxed);
    check("each_scale_u8_v128relaxed", test_scale<u8_t>, nk_each_scale_u8_v128relaxed);
    check("each_blend_u8_v128relaxed", test_blend<u8_t>, nk_each_blend_u8_v128relaxed);
    check("each_fma_u8_v128relaxed", test_fma<u8_t>, nk_each_fma_u8_v128relaxed);
#endif // NK_TARGET_V128RELAXED
}
