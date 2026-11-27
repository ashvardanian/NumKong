/**
 *  @file test/trigonometry.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Trigonometry tests: sin, cos, atan.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "numkong/trigonometry.hpp"
#include "numkong/random.hpp" // `nk::fill_uniform`

namespace ashvardanian::numkong::test {

/** Elementwise sin evaluated in @p precision_type_ and rounded once into @p in_type_. */
template <typename in_type_, typename precision_type_>
void sin_reference(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    for (std::size_t i = 0; i < n; i++) out[i] = narrow<in_type_>(widen<precision_type_>(in[i]).sin());
}

/** Elementwise cos evaluated in @p precision_type_ and rounded once into @p in_type_. */
template <typename in_type_, typename precision_type_>
void cos_reference(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    for (std::size_t i = 0; i < n; i++) out[i] = narrow<in_type_>(widen<precision_type_>(in[i]).cos());
}

/** Elementwise arctan evaluated in @p precision_type_ and rounded once into @p in_type_. */
template <typename in_type_, typename precision_type_>
void atan_reference(in_type_ const *in, std::size_t n, in_type_ *out) noexcept {
    for (std::size_t i = 0; i < n; i++) out[i] = narrow<in_type_>(widen<precision_type_>(in[i]).atan());
}

/** F16 kernels round an F32 result, which can land on the far side of an F16 tie from the exact
 *  value, so each F16 result may sit one F16 ULP from the rounded reference; wider types are held
 *  to the ULP threshold. */
template <typename scalar_type_>
constexpr bool trigonometry_bounded = std::is_same_v<scalar_type_, f16_t>;

/** Test sine approximation kernel against the `nk::sin<scalar_t, f118_t>` template. */
template <typename scalar_type_>
error_stats_t test_sin(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(trigonometry_bounded<scalar_t> ? comparison_family_t::bounded_k
                                                       : comparison_family_t::approximate_k);
    split_mix_generator_t generator(case_stream_key());
    auto inputs = make_vector<scalar_t>(test_run.settings.dense_dimensions);
    auto outputs = make_vector<scalar_t>(test_run.settings.dense_dimensions),
         reference = make_vector<scalar_t>(test_run.settings.dense_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -f64_t::two_pi_k().raw_,
                         f64_t::two_pi_k().raw_);

        stats.expect_status(
            kernel(inputs.raw_values_data(), test_run.settings.dense_dimensions, outputs.raw_values_data(), nullptr));
        sin_reference<scalar_t, reference_t>(inputs.values_data(), test_run.settings.dense_dimensions,
                                             reference.values_data());

        for (std::size_t i = 0; i < test_run.settings.dense_dimensions; i++)
            if constexpr (trigonometry_bounded<scalar_t>)
                stats.accumulate_bounded(outputs[i], reference[i], half_ulp<scalar_t>(to_double(reference[i])));
            else stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/** Test cosine approximation kernel against the `nk::cos<scalar_t, f118_t>` template. */
template <typename scalar_type_>
error_stats_t test_cos(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(trigonometry_bounded<scalar_t> ? comparison_family_t::bounded_k
                                                       : comparison_family_t::approximate_k);
    split_mix_generator_t generator(case_stream_key());
    auto inputs = make_vector<scalar_t>(test_run.settings.dense_dimensions);
    auto outputs = make_vector<scalar_t>(test_run.settings.dense_dimensions),
         reference = make_vector<scalar_t>(test_run.settings.dense_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -f64_t::two_pi_k().raw_,
                         f64_t::two_pi_k().raw_);

        stats.expect_status(
            kernel(inputs.raw_values_data(), test_run.settings.dense_dimensions, outputs.raw_values_data(), nullptr));
        cos_reference<scalar_t, reference_t>(inputs.values_data(), test_run.settings.dense_dimensions,
                                             reference.values_data());

        for (std::size_t i = 0; i < test_run.settings.dense_dimensions; i++)
            if constexpr (trigonometry_bounded<scalar_t>)
                stats.accumulate_bounded(outputs[i], reference[i], half_ulp<scalar_t>(to_double(reference[i])));
            else stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

/** Test atan approximation kernel against the `nk::atan<scalar_t, f118_t>` template. */
template <typename scalar_type_>
error_stats_t test_atan(typename scalar_type_::trigonometry_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(trigonometry_bounded<scalar_t> ? comparison_family_t::bounded_k
                                                       : comparison_family_t::approximate_k);
    split_mix_generator_t generator(case_stream_key());
    auto inputs = make_vector<scalar_t>(test_run.settings.dense_dimensions);
    auto outputs = make_vector<scalar_t>(test_run.settings.dense_dimensions),
         reference = make_vector<scalar_t>(test_run.settings.dense_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        nk::fill_uniform(generator, inputs.values_data(), inputs.size_values(), -10.0, 10.0);

        stats.expect_status(
            kernel(inputs.raw_values_data(), test_run.settings.dense_dimensions, outputs.raw_values_data(), nullptr));
        atan_reference<scalar_t, reference_t>(inputs.values_data(), test_run.settings.dense_dimensions,
                                              reference.values_data());

        for (std::size_t i = 0; i < test_run.settings.dense_dimensions; i++)
            if constexpr (trigonometry_bounded<scalar_t>)
                stats.accumulate_bounded(outputs[i], reference[i], half_ulp<scalar_t>(to_double(reference[i])));
            else stats.accumulate(outputs[i], reference[i]);
    }
    return stats;
}

void run_trigonometry_cases() {

    run_kernel_case("nk_trig_sin_f32_serial", test_sin<f32_t>, nk_cap_serial_k, nk_trig_sin_f32_serial);
    run_kernel_case("nk_trig_cos_f32_serial", test_cos<f32_t>, nk_cap_serial_k, nk_trig_cos_f32_serial);
    run_kernel_case("nk_trig_atan_f32_serial", test_atan<f32_t>, nk_cap_serial_k, nk_trig_atan_f32_serial);
    run_kernel_case("nk_trig_sin_f64_serial", test_sin<f64_t>, nk_cap_serial_k, nk_trig_sin_f64_serial);
    run_kernel_case("nk_trig_cos_f64_serial", test_cos<f64_t>, nk_cap_serial_k, nk_trig_cos_f64_serial);
    run_kernel_case("nk_trig_atan_f64_serial", test_atan<f64_t>, nk_cap_serial_k, nk_trig_atan_f64_serial);
    run_kernel_case("nk_trig_sin_f16_serial", test_sin<f16_t>, nk_cap_serial_k, nk_trig_sin_f16_serial);
    run_kernel_case("nk_trig_cos_f16_serial", test_cos<f16_t>, nk_cap_serial_k, nk_trig_cos_f16_serial);
    run_kernel_case("nk_trig_atan_f16_serial", test_atan<f16_t>, nk_cap_serial_k, nk_trig_atan_f16_serial);

    run_kernel_case("nk_trig_sin_f32_best", test_sin<f32_t>, nk_cap_serial_k, cpu_best<nk_trig_sin_f32_best>);
    run_kernel_case("nk_trig_cos_f32_best", test_cos<f32_t>, nk_cap_serial_k, cpu_best<nk_trig_cos_f32_best>);
    run_kernel_case("nk_trig_atan_f32_best", test_atan<f32_t>, nk_cap_serial_k, cpu_best<nk_trig_atan_f32_best>);
    run_kernel_case("nk_trig_sin_f64_best", test_sin<f64_t>, nk_cap_serial_k, cpu_best<nk_trig_sin_f64_best>);
    run_kernel_case("nk_trig_cos_f64_best", test_cos<f64_t>, nk_cap_serial_k, cpu_best<nk_trig_cos_f64_best>);
    run_kernel_case("nk_trig_atan_f64_best", test_atan<f64_t>, nk_cap_serial_k, cpu_best<nk_trig_atan_f64_best>);

#if NUMKONG_TARGET_NEON
    run_kernel_case("nk_trig_sin_f32_neon", test_sin<f32_t>, nk_cap_neon_k, nk_trig_sin_f32_neon);
    run_kernel_case("nk_trig_cos_f32_neon", test_cos<f32_t>, nk_cap_neon_k, nk_trig_cos_f32_neon);
    run_kernel_case("nk_trig_atan_f32_neon", test_atan<f32_t>, nk_cap_neon_k, nk_trig_atan_f32_neon);
    run_kernel_case("nk_trig_sin_f64_neon", test_sin<f64_t>, nk_cap_neon_k, nk_trig_sin_f64_neon);
    run_kernel_case("nk_trig_cos_f64_neon", test_cos<f64_t>, nk_cap_neon_k, nk_trig_cos_f64_neon);
    run_kernel_case("nk_trig_atan_f64_neon", test_atan<f64_t>, nk_cap_neon_k, nk_trig_atan_f64_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONHALF
    run_kernel_case("nk_trig_sin_f16_neonhalf", test_sin<f16_t>, nk_cap_neonhalf_k, nk_trig_sin_f16_neonhalf);
    run_kernel_case("nk_trig_cos_f16_neonhalf", test_cos<f16_t>, nk_cap_neonhalf_k, nk_trig_cos_f16_neonhalf);
    run_kernel_case("nk_trig_atan_f16_neonhalf", test_atan<f16_t>, nk_cap_neonhalf_k, nk_trig_atan_f16_neonhalf);
#endif // NUMKONG_TARGET_NEONHALF

#if NUMKONG_TARGET_SVEHALF
    run_kernel_case("nk_trig_sin_f16_svehalf", test_sin<f16_t>, nk_cap_svehalf_k, nk_trig_sin_f16_svehalf);
    run_kernel_case("nk_trig_cos_f16_svehalf", test_cos<f16_t>, nk_cap_svehalf_k, nk_trig_cos_f16_svehalf);
    run_kernel_case("nk_trig_atan_f16_svehalf", test_atan<f16_t>, nk_cap_svehalf_k, nk_trig_atan_f16_svehalf);
#endif // NUMKONG_TARGET_SVEHALF

#if NUMKONG_TARGET_HASWELL
    run_kernel_case("nk_trig_sin_f32_haswell", test_sin<f32_t>, nk_cap_haswell_k, nk_trig_sin_f32_haswell);
    run_kernel_case("nk_trig_cos_f32_haswell", test_cos<f32_t>, nk_cap_haswell_k, nk_trig_cos_f32_haswell);
    run_kernel_case("nk_trig_atan_f32_haswell", test_atan<f32_t>, nk_cap_haswell_k, nk_trig_atan_f32_haswell);
    run_kernel_case("nk_trig_sin_f64_haswell", test_sin<f64_t>, nk_cap_haswell_k, nk_trig_sin_f64_haswell);
    run_kernel_case("nk_trig_cos_f64_haswell", test_cos<f64_t>, nk_cap_haswell_k, nk_trig_cos_f64_haswell);
    run_kernel_case("nk_trig_atan_f64_haswell", test_atan<f64_t>, nk_cap_haswell_k, nk_trig_atan_f64_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    run_kernel_case("nk_trig_sin_f32_skylake", test_sin<f32_t>, nk_cap_skylake_k, nk_trig_sin_f32_skylake);
    run_kernel_case("nk_trig_cos_f32_skylake", test_cos<f32_t>, nk_cap_skylake_k, nk_trig_cos_f32_skylake);
    run_kernel_case("nk_trig_atan_f32_skylake", test_atan<f32_t>, nk_cap_skylake_k, nk_trig_atan_f32_skylake);
    run_kernel_case("nk_trig_sin_f64_skylake", test_sin<f64_t>, nk_cap_skylake_k, nk_trig_sin_f64_skylake);
    run_kernel_case("nk_trig_cos_f64_skylake", test_cos<f64_t>, nk_cap_skylake_k, nk_trig_cos_f64_skylake);
    run_kernel_case("nk_trig_atan_f64_skylake", test_atan<f64_t>, nk_cap_skylake_k, nk_trig_atan_f64_skylake);
    run_kernel_case("nk_trig_sin_f16_skylake", test_sin<f16_t>, nk_cap_skylake_k, nk_trig_sin_f16_skylake);
    run_kernel_case("nk_trig_cos_f16_skylake", test_cos<f16_t>, nk_cap_skylake_k, nk_trig_cos_f16_skylake);
    run_kernel_case("nk_trig_atan_f16_skylake", test_atan<f16_t>, nk_cap_skylake_k, nk_trig_atan_f16_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_SAPPHIRE
    run_kernel_case("nk_trig_sin_f16_sapphire", test_sin<f16_t>, nk_cap_sapphire_k, nk_trig_sin_f16_sapphire);
    run_kernel_case("nk_trig_cos_f16_sapphire", test_cos<f16_t>, nk_cap_sapphire_k, nk_trig_cos_f16_sapphire);
    run_kernel_case("nk_trig_atan_f16_sapphire", test_atan<f16_t>, nk_cap_sapphire_k, nk_trig_atan_f16_sapphire);
#endif // NUMKONG_TARGET_SAPPHIRE

#if NUMKONG_TARGET_V128RELAXED
    auto const v128relaxed = nk_cap_v128relaxed_k;
    run_kernel_case("nk_trig_sin_f32_v128relaxed", test_sin<f32_t>, v128relaxed, nk_trig_sin_f32_v128relaxed);
    run_kernel_case("nk_trig_cos_f32_v128relaxed", test_cos<f32_t>, v128relaxed, nk_trig_cos_f32_v128relaxed);
    run_kernel_case("nk_trig_atan_f32_v128relaxed", test_atan<f32_t>, v128relaxed, nk_trig_atan_f32_v128relaxed);
    run_kernel_case("nk_trig_sin_f64_v128relaxed", test_sin<f64_t>, v128relaxed, nk_trig_sin_f64_v128relaxed);
    run_kernel_case("nk_trig_cos_f64_v128relaxed", test_cos<f64_t>, v128relaxed, nk_trig_cos_f64_v128relaxed);
    run_kernel_case("nk_trig_atan_f64_v128relaxed", test_atan<f64_t>, v128relaxed, nk_trig_atan_f64_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED
}

} // namespace ashvardanian::numkong::test
