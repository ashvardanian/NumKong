/**
 *  @file test/trigonometry.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Trigonometry tests: sin, cos, atan.
 */

#include "harness.hpp"
#include "numkong/trigonometry.hpp"

namespace ashvardanian::numkong::test {

/** F16 kernels round an F32 result, which can land on the far side of an F16 tie from the exact
 *  value, so each F16 result may sit one F16 ULP from the rounded reference; wider types are held
 *  to the ULP threshold. */
template <typename scalar_type_>
constexpr bool trigonometry_bounded = std::is_same_v<scalar_type_, f16_t>;

/** Test sine approximation kernel against the `nk::sin<scalar_t, f118_t>` template. */
template <typename scalar_type_>
error_stats_t test_sin(settings_t const &settings, typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(trigonometry_bounded<scalar_t> ? comparison_family_t::bounded_k
                                                       : comparison_family_t::approximate_k);
    std::mt19937 generator(settings.seed.value);
    auto inputs = make_vector<scalar_t>(settings.dense_dimensions);
    auto outputs = make_vector<scalar_t>(settings.dense_dimensions),
         reference = make_vector<scalar_t>(settings.dense_dimensions);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -scalar_t::two_pi_k(),
                         scalar_t::two_pi_k());

        stats.expect(kernel(inputs.raw_values_data(), settings.dense_dimensions, outputs.raw_values_data(), nullptr));
        stats.expect(nk::sin<scalar_t, reference_t>(inputs.values_data(), settings.dense_dimensions,
                                                    reference.values_data(), no_tiers_k));

        for (std::size_t i = 0; i < settings.dense_dimensions; i++)
            if constexpr (trigonometry_bounded<scalar_t>)
                stats.accumulate_bounded(outputs[i], reference[i], half_ulp<scalar_t>(reference[i]));
            else stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/** Test cosine approximation kernel against the `nk::cos<scalar_t, f118_t>` template. */
template <typename scalar_type_>
error_stats_t test_cos(settings_t const &settings, typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(trigonometry_bounded<scalar_t> ? comparison_family_t::bounded_k
                                                       : comparison_family_t::approximate_k);
    std::mt19937 generator(settings.seed.value);
    auto inputs = make_vector<scalar_t>(settings.dense_dimensions);
    auto outputs = make_vector<scalar_t>(settings.dense_dimensions),
         reference = make_vector<scalar_t>(settings.dense_dimensions);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -scalar_t::two_pi_k(),
                         scalar_t::two_pi_k());

        stats.expect(kernel(inputs.raw_values_data(), settings.dense_dimensions, outputs.raw_values_data(), nullptr));
        stats.expect(nk::cos<scalar_t, reference_t>(inputs.values_data(), settings.dense_dimensions,
                                                    reference.values_data(), no_tiers_k));

        for (std::size_t i = 0; i < settings.dense_dimensions; i++)
            if constexpr (trigonometry_bounded<scalar_t>)
                stats.accumulate_bounded(outputs[i], reference[i], half_ulp<scalar_t>(reference[i]));
            else stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/** Test atan approximation kernel against the `nk::atan<scalar_t, f118_t>` template. */
template <typename scalar_type_>
error_stats_t test_atan(settings_t const &settings, typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(trigonometry_bounded<scalar_t> ? comparison_family_t::bounded_k
                                                       : comparison_family_t::approximate_k);
    std::mt19937 generator(settings.seed.value);
    auto inputs = make_vector<scalar_t>(settings.dense_dimensions);
    auto outputs = make_vector<scalar_t>(settings.dense_dimensions),
         reference = make_vector<scalar_t>(settings.dense_dimensions);

    for (time_point_t const deadline = steady_clock_t::now() + settings.time_limit_per_kernel;
         steady_clock_t::now() < deadline;) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), scalar_t(-10.0), scalar_t(10.0));

        stats.expect(kernel(inputs.raw_values_data(), settings.dense_dimensions, outputs.raw_values_data(), nullptr));
        stats.expect(nk::atan<scalar_t, reference_t>(inputs.values_data(), settings.dense_dimensions,
                                                     reference.values_data(), no_tiers_k));

        for (std::size_t i = 0; i < settings.dense_dimensions; i++)
            if constexpr (trigonometry_bounded<scalar_t>)
                stats.accumulate_bounded(outputs[i], reference[i], half_ulp<scalar_t>(reference[i]));
            else stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

void test_trigonometry(error_stats_section_t &check) {

    check.section("Trigonometry Serial", nk_cap_serial_k);
    check("trig_sin_f32_serial", test_sin<f32_t>, nk_trig_sin_f32_serial);
    check("trig_cos_f32_serial", test_cos<f32_t>, nk_trig_cos_f32_serial);
    check("trig_atan_f32_serial", test_atan<f32_t>, nk_trig_atan_f32_serial);
    check("trig_sin_f64_serial", test_sin<f64_t>, nk_trig_sin_f64_serial);
    check("trig_cos_f64_serial", test_cos<f64_t>, nk_trig_cos_f64_serial);
    check("trig_atan_f64_serial", test_atan<f64_t>, nk_trig_atan_f64_serial);
    check("trig_sin_f16_serial", test_sin<f16_t>, nk_trig_sin_f16_serial);
    check("trig_cos_f16_serial", test_cos<f16_t>, nk_trig_cos_f16_serial);
    check("trig_atan_f16_serial", test_atan<f16_t>, nk_trig_atan_f16_serial);

#if !NUMKONG_HEADER_ONLY
    check.section("Trigonometry Runtime Dispatch", nk_cap_serial_k);
    check("trig_sin_f32", test_sin<f32_t>, cpu_best<nk_trig_sin_f32_best>);
    check("trig_cos_f32", test_cos<f32_t>, cpu_best<nk_trig_cos_f32_best>);
    check("trig_atan_f32", test_atan<f32_t>, cpu_best<nk_trig_atan_f32_best>);
    check("trig_sin_f64", test_sin<f64_t>, cpu_best<nk_trig_sin_f64_best>);
    check("trig_cos_f64", test_cos<f64_t>, cpu_best<nk_trig_cos_f64_best>);
    check("trig_atan_f64", test_atan<f64_t>, cpu_best<nk_trig_atan_f64_best>);
#endif

#if NUMKONG_TARGET_NEON
    check.section("Trigonometry NEON", nk_cap_neon_k);
    check("trig_sin_f32_neon", test_sin<f32_t>, nk_trig_sin_f32_neon);
    check("trig_cos_f32_neon", test_cos<f32_t>, nk_trig_cos_f32_neon);
    check("trig_atan_f32_neon", test_atan<f32_t>, nk_trig_atan_f32_neon);
    check("trig_sin_f64_neon", test_sin<f64_t>, nk_trig_sin_f64_neon);
    check("trig_cos_f64_neon", test_cos<f64_t>, nk_trig_cos_f64_neon);
    check("trig_atan_f64_neon", test_atan<f64_t>, nk_trig_atan_f64_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONHALF
    check.section("Trigonometry NEON HALF", nk_cap_neonhalf_k);
    check("trig_sin_f16_neonhalf", test_sin<f16_t>, nk_trig_sin_f16_neonhalf);
    check("trig_cos_f16_neonhalf", test_cos<f16_t>, nk_trig_cos_f16_neonhalf);
    check("trig_atan_f16_neonhalf", test_atan<f16_t>, nk_trig_atan_f16_neonhalf);
#endif // NUMKONG_TARGET_NEONHALF

#if NUMKONG_TARGET_SVEHALF
    check.section("Trigonometry SVE HALF", nk_cap_svehalf_k);
    check("trig_sin_f16_svehalf", test_sin<f16_t>, nk_trig_sin_f16_svehalf);
    check("trig_cos_f16_svehalf", test_cos<f16_t>, nk_trig_cos_f16_svehalf);
    check("trig_atan_f16_svehalf", test_atan<f16_t>, nk_trig_atan_f16_svehalf);
#endif // NUMKONG_TARGET_SVEHALF

#if NUMKONG_TARGET_HASWELL
    check.section("Trigonometry Haswell", nk_cap_haswell_k);
    check("trig_sin_f32_haswell", test_sin<f32_t>, nk_trig_sin_f32_haswell);
    check("trig_cos_f32_haswell", test_cos<f32_t>, nk_trig_cos_f32_haswell);
    check("trig_atan_f32_haswell", test_atan<f32_t>, nk_trig_atan_f32_haswell);
    check("trig_sin_f64_haswell", test_sin<f64_t>, nk_trig_sin_f64_haswell);
    check("trig_cos_f64_haswell", test_cos<f64_t>, nk_trig_cos_f64_haswell);
    check("trig_atan_f64_haswell", test_atan<f64_t>, nk_trig_atan_f64_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    check.section("Trigonometry Skylake", nk_cap_skylake_k);
    check("trig_sin_f32_skylake", test_sin<f32_t>, nk_trig_sin_f32_skylake);
    check("trig_cos_f32_skylake", test_cos<f32_t>, nk_trig_cos_f32_skylake);
    check("trig_atan_f32_skylake", test_atan<f32_t>, nk_trig_atan_f32_skylake);
    check("trig_sin_f64_skylake", test_sin<f64_t>, nk_trig_sin_f64_skylake);
    check("trig_cos_f64_skylake", test_cos<f64_t>, nk_trig_cos_f64_skylake);
    check("trig_atan_f64_skylake", test_atan<f64_t>, nk_trig_atan_f64_skylake);
    check("trig_sin_f16_skylake", test_sin<f16_t>, nk_trig_sin_f16_skylake);
    check("trig_cos_f16_skylake", test_cos<f16_t>, nk_trig_cos_f16_skylake);
    check("trig_atan_f16_skylake", test_atan<f16_t>, nk_trig_atan_f16_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_SAPPHIRE
    check.section("Trigonometry Sapphire", nk_cap_sapphire_k);
    check("trig_sin_f16_sapphire", test_sin<f16_t>, nk_trig_sin_f16_sapphire);
    check("trig_cos_f16_sapphire", test_cos<f16_t>, nk_trig_cos_f16_sapphire);
    check("trig_atan_f16_sapphire", test_atan<f16_t>, nk_trig_atan_f16_sapphire);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_V128RELAXED
    check.section("Trigonometry V128 Relaxed", nk_cap_v128relaxed_k);
    check("trig_sin_f32_v128relaxed", test_sin<f32_t>, nk_trig_sin_f32_v128relaxed);
    check("trig_cos_f32_v128relaxed", test_cos<f32_t>, nk_trig_cos_f32_v128relaxed);
    check("trig_atan_f32_v128relaxed", test_atan<f32_t>, nk_trig_atan_f32_v128relaxed);
    check("trig_sin_f64_v128relaxed", test_sin<f64_t>, nk_trig_sin_f64_v128relaxed);
    check("trig_cos_f64_v128relaxed", test_cos<f64_t>, nk_trig_cos_f64_v128relaxed);
    check("trig_atan_f64_v128relaxed", test_atan<f64_t>, nk_trig_atan_f64_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED
}

} // namespace ashvardanian::numkong::test
