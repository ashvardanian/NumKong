/**
 *  @brief Trigonometry tests (sin, cos, atan).
 *  @file test/test_trigonometry.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/trigonometry.hpp"

/**
 *  @brief Test sine approximation kernel against `nk::sin<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_sin(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(global_config.dense_dimensions);
    auto outputs = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -scalar_t::two_pi_k(),
                         scalar_t::two_pi_k());

        kernel(inputs.raw_values_data(), global_config.dense_dimensions, outputs.raw_values_data());
        nk::sin<scalar_t, reference_t, nk::no_simd_k>(inputs.values_data(), global_config.dense_dimensions,
                                                      reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test cosine approximation kernel against `nk::cos<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_cos(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(global_config.dense_dimensions);
    auto outputs = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -scalar_t::two_pi_k(),
                         scalar_t::two_pi_k());

        kernel(inputs.raw_values_data(), global_config.dense_dimensions, outputs.raw_values_data());
        nk::cos<scalar_t, reference_t, nk::no_simd_k>(inputs.values_data(), global_config.dense_dimensions,
                                                      reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/**
 *  @brief Test atan approximation kernel against `nk::atan<scalar_t, f118_t, nk::no_simd_k>`.
 */
template <typename scalar_type_>
error_stats_t test_atan(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);
    auto inputs = make_vector<scalar_t>(global_config.dense_dimensions);
    auto outputs = make_vector<scalar_t>(global_config.dense_dimensions),
         reference = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), scalar_t(-10.0), scalar_t(10.0));

        kernel(inputs.raw_values_data(), global_config.dense_dimensions, outputs.raw_values_data());
        nk::atan<scalar_t, reference_t, nk::no_simd_k>(inputs.values_data(), global_config.dense_dimensions,
                                                       reference.values_data());

        for (std::size_t i = 0; i < global_config.dense_dimensions; i++) stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

void test_trigonometry() {
    error_stats_section_t check;

    check.section("Trigonometry Serial", nk_cap_serial_k);
    check("each_sin_f32_serial", test_sin<f32_t>, nk_each_sin_f32_serial);
    check("each_cos_f32_serial", test_cos<f32_t>, nk_each_cos_f32_serial);
    check("each_atan_f32_serial", test_atan<f32_t>, nk_each_atan_f32_serial);
    check("each_sin_f64_serial", test_sin<f64_t>, nk_each_sin_f64_serial);
    check("each_cos_f64_serial", test_cos<f64_t>, nk_each_cos_f64_serial);
    check("each_atan_f64_serial", test_atan<f64_t>, nk_each_atan_f64_serial);
    check("each_sin_f16_serial", test_sin<f16_t>, nk_each_sin_f16_serial);
    check("each_cos_f16_serial", test_cos<f16_t>, nk_each_cos_f16_serial);
    check("each_atan_f16_serial", test_atan<f16_t>, nk_each_atan_f16_serial);

#if NK_DYNAMIC_DISPATCH
    check.section("Trigonometry Dynamic", nk_cap_serial_k);
    check("each_sin_f32", test_sin<f32_t>, nk_each_sin_f32);
    check("each_cos_f32", test_cos<f32_t>, nk_each_cos_f32);
    check("each_atan_f32", test_atan<f32_t>, nk_each_atan_f32);
    check("each_sin_f64", test_sin<f64_t>, nk_each_sin_f64);
    check("each_cos_f64", test_cos<f64_t>, nk_each_cos_f64);
    check("each_atan_f64", test_atan<f64_t>, nk_each_atan_f64);
#endif

#if NK_TARGET_NEON
    check.section("Trigonometry NEON", nk_cap_neon_k);
    check("each_sin_f32_neon", test_sin<f32_t>, nk_each_sin_f32_neon);
    check("each_cos_f32_neon", test_cos<f32_t>, nk_each_cos_f32_neon);
    check("each_atan_f32_neon", test_atan<f32_t>, nk_each_atan_f32_neon);
    check("each_sin_f64_neon", test_sin<f64_t>, nk_each_sin_f64_neon);
    check("each_cos_f64_neon", test_cos<f64_t>, nk_each_cos_f64_neon);
    check("each_atan_f64_neon", test_atan<f64_t>, nk_each_atan_f64_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    check.section("Trigonometry Haswell", nk_cap_haswell_k);
    check("each_sin_f32_haswell", test_sin<f32_t>, nk_each_sin_f32_haswell);
    check("each_cos_f32_haswell", test_cos<f32_t>, nk_each_cos_f32_haswell);
    check("each_atan_f32_haswell", test_atan<f32_t>, nk_each_atan_f32_haswell);
    check("each_sin_f64_haswell", test_sin<f64_t>, nk_each_sin_f64_haswell);
    check("each_cos_f64_haswell", test_cos<f64_t>, nk_each_cos_f64_haswell);
    check("each_atan_f64_haswell", test_atan<f64_t>, nk_each_atan_f64_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check.section("Trigonometry Skylake", nk_cap_skylake_k);
    check("each_sin_f32_skylake", test_sin<f32_t>, nk_each_sin_f32_skylake);
    check("each_cos_f32_skylake", test_cos<f32_t>, nk_each_cos_f32_skylake);
    check("each_atan_f32_skylake", test_atan<f32_t>, nk_each_atan_f32_skylake);
    check("each_sin_f64_skylake", test_sin<f64_t>, nk_each_sin_f64_skylake);
    check("each_cos_f64_skylake", test_cos<f64_t>, nk_each_cos_f64_skylake);
    check("each_atan_f64_skylake", test_atan<f64_t>, nk_each_atan_f64_skylake);
    check("each_sin_f16_skylake", test_sin<f16_t>, nk_each_sin_f16_skylake);
    check("each_cos_f16_skylake", test_cos<f16_t>, nk_each_cos_f16_skylake);
    check("each_atan_f16_skylake", test_atan<f16_t>, nk_each_atan_f16_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_V128RELAXED
    check.section("Trigonometry V128 Relaxed", nk_cap_v128relaxed_k);
    check("each_sin_f32_v128relaxed", test_sin<f32_t>, nk_each_sin_f32_v128relaxed);
    check("each_cos_f32_v128relaxed", test_cos<f32_t>, nk_each_cos_f32_v128relaxed);
    check("each_atan_f32_v128relaxed", test_atan<f32_t>, nk_each_atan_f32_v128relaxed);
    check("each_sin_f64_v128relaxed", test_sin<f64_t>, nk_each_sin_f64_v128relaxed);
    check("each_cos_f64_v128relaxed", test_cos<f64_t>, nk_each_cos_f64_v128relaxed);
    check("each_atan_f64_v128relaxed", test_atan<f64_t>, nk_each_atan_f64_v128relaxed);
#endif // NK_TARGET_V128RELAXED
}
