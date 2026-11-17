/**
 *  @file test/dot.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Dot product precision tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "numkong/dot.hpp" // `nk::dot`

namespace ashvardanian::numkong::test {

/** Unified dot product test for all types: float, integer, and complex. Works with f32_t, f64_t,
 *  f16_t, bf16_t, e4m3_t, e5m2_t, i8_t, u8_t, f32c_t, f64c_t. */
template <typename scalar_type_>
error_stats_t test_dot(typename scalar_type_::dot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::dot_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_dot_error_bound(scalar_t::dtype()));
    split_mix_generator_t generator(case_stream_key());
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const n = nk::divide_round_up(test_run.settings.dense_dimensions, dims_per_value) * dims_per_value;
    auto a = make_vector<scalar_t>(n), b = make_vector<scalar_t>(n);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_, nullptr));

        reference_t reference;
        stats.expect_status(
            nk::dot<scalar_t, reference_t>(a.values_data(), b.values_data(), n, &reference, no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

/** Conjugate dot product test for complex types (vdot = conj(a) * b). */
template <typename scalar_type_>
error_stats_t test_vdot(typename scalar_type_::vdot_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::vdot_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_dot_error_bound(scalar_t::dtype()));
    split_mix_generator_t generator(case_stream_key());
    auto a = make_vector<scalar_t>(test_run.settings.dense_dimensions),
         b = make_vector<scalar_t>(test_run.settings.dense_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), test_run.settings.dense_dimensions,
                                   &result.raw_, nullptr));

        reference_t reference;
        stats.expect_status(nk::vdot<scalar_t, reference_t>(
            a.values_data(), b.values_data(), test_run.settings.dense_dimensions, &reference, no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

void run_dot_cases() {

    run_kernel_case("nk_dot_f64_serial", test_dot<f64_t>, nk_cap_serial_k, nk_dot_f64_serial);
    run_kernel_case("nk_dot_f32_serial", test_dot<f32_t>, nk_cap_serial_k, nk_dot_f32_serial);
    run_kernel_case("nk_dot_bf16_serial", test_dot<bf16_t>, nk_cap_serial_k, nk_dot_bf16_serial);
    run_kernel_case("nk_dot_f16_serial", test_dot<f16_t>, nk_cap_serial_k, nk_dot_f16_serial);
    run_kernel_case("nk_dot_e5m2_serial", test_dot<e5m2_t>, nk_cap_serial_k, nk_dot_e5m2_serial);
    run_kernel_case("nk_dot_e4m3_serial", test_dot<e4m3_t>, nk_cap_serial_k, nk_dot_e4m3_serial);
    run_kernel_case("nk_dot_e3m2_serial", test_dot<e3m2_t>, nk_cap_serial_k, nk_dot_e3m2_serial);
    run_kernel_case("nk_dot_e2m3_serial", test_dot<e2m3_t>, nk_cap_serial_k, nk_dot_e2m3_serial);
    run_kernel_case("nk_dot_e2m1_serial", test_dot<e2m1x2_t>, nk_cap_serial_k, nk_dot_e2m1_serial);
    run_kernel_case("nk_dot_i8_serial", test_dot<i8_t>, nk_cap_serial_k, nk_dot_i8_serial);
    run_kernel_case("nk_dot_i4_serial", test_dot<i4x2_t>, nk_cap_serial_k, nk_dot_i4_serial);
    run_kernel_case("nk_dot_u8_serial", test_dot<u8_t>, nk_cap_serial_k, nk_dot_u8_serial);
    run_kernel_case("nk_dot_u4_serial", test_dot<u4x2_t>, nk_cap_serial_k, nk_dot_u4_serial);
    run_kernel_case("nk_dot_u1_serial", test_dot<u1x8_t>, nk_cap_serial_k, nk_dot_u1_serial);
    run_kernel_case("nk_dot_f32c_serial", test_dot<f32c_t>, nk_cap_serial_k, nk_dot_f32c_serial);
    run_kernel_case("nk_vdot_f32c_serial", test_vdot<f32c_t>, nk_cap_serial_k, nk_vdot_f32c_serial);
    run_kernel_case("nk_dot_f64c_serial", test_dot<f64c_t>, nk_cap_serial_k, nk_dot_f64c_serial);
    run_kernel_case("nk_vdot_f64c_serial", test_vdot<f64c_t>, nk_cap_serial_k, nk_vdot_f64c_serial);
    run_kernel_case("nk_dot_f16c_serial", test_dot<f16c_t>, nk_cap_serial_k, nk_dot_f16c_serial);
    run_kernel_case("nk_vdot_f16c_serial", test_vdot<f16c_t>, nk_cap_serial_k, nk_vdot_f16c_serial);
    run_kernel_case("nk_dot_bf16c_serial", test_dot<bf16c_t>, nk_cap_serial_k, nk_dot_bf16c_serial);
    run_kernel_case("nk_vdot_bf16c_serial", test_vdot<bf16c_t>, nk_cap_serial_k, nk_vdot_bf16c_serial);

#if !NUMKONG_HEADER_ONLY
    run_kernel_case("nk_dot_f64_best", test_dot<f64_t>, nk_cap_serial_k, cpu_best<nk_dot_f64_best>);
    run_kernel_case("nk_dot_f32_best", test_dot<f32_t>, nk_cap_serial_k, cpu_best<nk_dot_f32_best>);
    run_kernel_case("nk_dot_bf16_best", test_dot<bf16_t>, nk_cap_serial_k, cpu_best<nk_dot_bf16_best>);
    run_kernel_case("nk_dot_f16_best", test_dot<f16_t>, nk_cap_serial_k, cpu_best<nk_dot_f16_best>);
    run_kernel_case("nk_dot_e5m2_best", test_dot<e5m2_t>, nk_cap_serial_k, cpu_best<nk_dot_e5m2_best>);
    run_kernel_case("nk_dot_e4m3_best", test_dot<e4m3_t>, nk_cap_serial_k, cpu_best<nk_dot_e4m3_best>);
    run_kernel_case("nk_dot_e3m2_best", test_dot<e3m2_t>, nk_cap_serial_k, cpu_best<nk_dot_e3m2_best>);
    run_kernel_case("nk_dot_e2m3_best", test_dot<e2m3_t>, nk_cap_serial_k, cpu_best<nk_dot_e2m3_best>);
    run_kernel_case("nk_dot_e2m1_best", test_dot<e2m1x2_t>, nk_cap_serial_k, cpu_best<nk_dot_e2m1_best>);
    run_kernel_case("nk_dot_i8_best", test_dot<i8_t>, nk_cap_serial_k, cpu_best<nk_dot_i8_best>);
    run_kernel_case("nk_dot_i4_best", test_dot<i4x2_t>, nk_cap_serial_k, cpu_best<nk_dot_i4_best>);
    run_kernel_case("nk_dot_u8_best", test_dot<u8_t>, nk_cap_serial_k, cpu_best<nk_dot_u8_best>);
    run_kernel_case("nk_dot_u4_best", test_dot<u4x2_t>, nk_cap_serial_k, cpu_best<nk_dot_u4_best>);
    run_kernel_case("nk_dot_u1_best", test_dot<u1x8_t>, nk_cap_serial_k, cpu_best<nk_dot_u1_best>);
    run_kernel_case("nk_dot_f32c_best", test_dot<f32c_t>, nk_cap_serial_k, cpu_best<nk_dot_f32c_best>);
    run_kernel_case("nk_vdot_f32c_best", test_vdot<f32c_t>, nk_cap_serial_k, cpu_best<nk_vdot_f32c_best>);
    run_kernel_case("nk_dot_f64c_best", test_dot<f64c_t>, nk_cap_serial_k, cpu_best<nk_dot_f64c_best>);
    run_kernel_case("nk_vdot_f64c_best", test_vdot<f64c_t>, nk_cap_serial_k, cpu_best<nk_vdot_f64c_best>);
    run_kernel_case("nk_dot_f16c_best", test_dot<f16c_t>, nk_cap_serial_k, cpu_best<nk_dot_f16c_best>);
    run_kernel_case("nk_vdot_f16c_best", test_vdot<f16c_t>, nk_cap_serial_k, cpu_best<nk_vdot_f16c_best>);
    run_kernel_case("nk_dot_bf16c_best", test_dot<bf16c_t>, nk_cap_serial_k, cpu_best<nk_dot_bf16c_best>);
    run_kernel_case("nk_vdot_bf16c_best", test_vdot<bf16c_t>, nk_cap_serial_k, cpu_best<nk_vdot_bf16c_best>);
#endif

#if NUMKONG_TARGET_NEON
    run_kernel_case("nk_dot_f64_neon", test_dot<f64_t>, nk_cap_neon_k, nk_dot_f64_neon);
    run_kernel_case("nk_dot_f32_neon", test_dot<f32_t>, nk_cap_neon_k, nk_dot_f32_neon);
    run_kernel_case("nk_dot_f32c_neon", test_dot<f32c_t>, nk_cap_neon_k, nk_dot_f32c_neon);
    run_kernel_case("nk_vdot_f32c_neon", test_vdot<f32c_t>, nk_cap_neon_k, nk_vdot_f32c_neon);
    run_kernel_case("nk_dot_f64c_neon", test_dot<f64c_t>, nk_cap_neon_k, nk_dot_f64c_neon);
    run_kernel_case("nk_vdot_f64c_neon", test_vdot<f64c_t>, nk_cap_neon_k, nk_vdot_f64c_neon);
    run_kernel_case("nk_dot_bf16_neon", test_dot<bf16_t>, nk_cap_neon_k, nk_dot_bf16_neon);
    run_kernel_case("nk_dot_f16_neon", test_dot<f16_t>, nk_cap_neon_k, nk_dot_f16_neon);
    run_kernel_case("nk_dot_e5m2_neon", test_dot<e5m2_t>, nk_cap_neon_k, nk_dot_e5m2_neon);
    run_kernel_case("nk_dot_e4m3_neon", test_dot<e4m3_t>, nk_cap_neon_k, nk_dot_e4m3_neon);
    run_kernel_case("nk_dot_e3m2_neon", test_dot<e3m2_t>, nk_cap_neon_k, nk_dot_e3m2_neon);
    run_kernel_case("nk_dot_e2m3_neon", test_dot<e2m3_t>, nk_cap_neon_k, nk_dot_e2m3_neon);
    run_kernel_case("nk_dot_f16c_neon", test_dot<f16c_t>, nk_cap_neon_k, nk_dot_f16c_neon);
    run_kernel_case("nk_vdot_f16c_neon", test_vdot<f16c_t>, nk_cap_neon_k, nk_vdot_f16c_neon);
    run_kernel_case("nk_dot_u1_neon", test_dot<u1x8_t>, nk_cap_neon_k, nk_dot_u1_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONSDOT
    run_kernel_case("nk_dot_e3m2_neonsdot", test_dot<e3m2_t>, nk_cap_neonsdot_k, nk_dot_e3m2_neonsdot);
    run_kernel_case("nk_dot_e2m3_neonsdot", test_dot<e2m3_t>, nk_cap_neonsdot_k, nk_dot_e2m3_neonsdot);
    run_kernel_case("nk_dot_e2m1_neonsdot", test_dot<e2m1x2_t>, nk_cap_neonsdot_k, nk_dot_e2m1_neonsdot);
    run_kernel_case("nk_dot_i8_neonsdot", test_dot<i8_t>, nk_cap_neonsdot_k, nk_dot_i8_neonsdot);
    run_kernel_case("nk_dot_i4_neonsdot", test_dot<i4x2_t>, nk_cap_neonsdot_k, nk_dot_i4_neonsdot);
    run_kernel_case("nk_dot_u8_neonsdot", test_dot<u8_t>, nk_cap_neonsdot_k, nk_dot_u8_neonsdot);
    run_kernel_case("nk_dot_u4_neonsdot", test_dot<u4x2_t>, nk_cap_neonsdot_k, nk_dot_u4_neonsdot);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_NEONFHM
    run_kernel_case("nk_dot_f16_neonfhm", test_dot<f16_t>, nk_cap_neonfhm_k, nk_dot_f16_neonfhm);
    run_kernel_case("nk_dot_f16c_neonfhm", test_dot<f16c_t>, nk_cap_neonfhm_k, nk_dot_f16c_neonfhm);
    run_kernel_case("nk_vdot_f16c_neonfhm", test_vdot<f16c_t>, nk_cap_neonfhm_k, nk_vdot_f16c_neonfhm);
    run_kernel_case("nk_dot_e5m2_neonfhm", test_dot<e5m2_t>, nk_cap_neonfhm_k, nk_dot_e5m2_neonfhm);
    run_kernel_case("nk_dot_e4m3_neonfhm", test_dot<e4m3_t>, nk_cap_neonfhm_k, nk_dot_e4m3_neonfhm);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONBFDOT
    run_kernel_case("nk_dot_bf16_neonbfdot", test_dot<bf16_t>, nk_cap_neonbfdot_k, nk_dot_bf16_neonbfdot);
    run_kernel_case("nk_dot_bf16c_neonbfdot", test_dot<bf16c_t>, nk_cap_neonbfdot_k, nk_dot_bf16c_neonbfdot);
    run_kernel_case("nk_vdot_bf16c_neonbfdot", test_vdot<bf16c_t>, nk_cap_neonbfdot_k, nk_vdot_bf16c_neonbfdot);
    run_kernel_case("nk_dot_e5m2_neonbfdot", test_dot<e5m2_t>, nk_cap_neonbfdot_k, nk_dot_e5m2_neonbfdot);
    run_kernel_case("nk_dot_e4m3_neonbfdot", test_dot<e4m3_t>, nk_cap_neonbfdot_k, nk_dot_e4m3_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONFP8
    run_kernel_case("nk_dot_e5m2_neonfp8", test_dot<e5m2_t>, nk_cap_neonfp8_k, nk_dot_e5m2_neonfp8);
    run_kernel_case("nk_dot_e4m3_neonfp8", test_dot<e4m3_t>, nk_cap_neonfp8_k, nk_dot_e4m3_neonfp8);
    run_kernel_case("nk_dot_e3m2_neonfp8", test_dot<e3m2_t>, nk_cap_neonfp8_k, nk_dot_e3m2_neonfp8);
    run_kernel_case("nk_dot_e2m3_neonfp8", test_dot<e2m3_t>, nk_cap_neonfp8_k, nk_dot_e2m3_neonfp8);
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_SVE
    run_kernel_case("nk_dot_f64_sve", test_dot<f64_t>, nk_cap_sve_k, nk_dot_f64_sve);
    run_kernel_case("nk_dot_f32_sve", test_dot<f32_t>, nk_cap_sve_k, nk_dot_f32_sve);
    run_kernel_case("nk_dot_f32c_sve", test_dot<f32c_t>, nk_cap_sve_k, nk_dot_f32c_sve);
    run_kernel_case("nk_vdot_f32c_sve", test_vdot<f32c_t>, nk_cap_sve_k, nk_vdot_f32c_sve);
    run_kernel_case("nk_dot_f64c_sve", test_dot<f64c_t>, nk_cap_sve_k, nk_dot_f64c_sve);
    run_kernel_case("nk_vdot_f64c_sve", test_vdot<f64c_t>, nk_cap_sve_k, nk_vdot_f64c_sve);
#endif // NUMKONG_TARGET_SVE

#if NUMKONG_TARGET_SVEHALF
    run_kernel_case("nk_dot_f16_svehalf", test_dot<f16_t>, nk_cap_svehalf_k, nk_dot_f16_svehalf);
    run_kernel_case("nk_dot_f16c_svehalf", test_dot<f16c_t>, nk_cap_svehalf_k, nk_dot_f16c_svehalf);
    run_kernel_case("nk_vdot_f16c_svehalf", test_vdot<f16c_t>, nk_cap_svehalf_k, nk_vdot_f16c_svehalf);
#endif // NUMKONG_TARGET_SVEHALF

#if NUMKONG_TARGET_SVEBFDOT
    run_kernel_case("nk_dot_bf16_svebfdot", test_dot<bf16_t>, nk_cap_svebfdot_k, nk_dot_bf16_svebfdot);
#endif // NUMKONG_TARGET_SVEBFDOT

#if NUMKONG_TARGET_SVESDOT
    run_kernel_case("nk_dot_i8_svesdot", test_dot<i8_t>, nk_cap_svesdot_k, nk_dot_i8_svesdot);
    run_kernel_case("nk_dot_u8_svesdot", test_dot<u8_t>, nk_cap_svesdot_k, nk_dot_u8_svesdot);
#endif // NUMKONG_TARGET_SVESDOT

#if NUMKONG_TARGET_HASWELL
    run_kernel_case("nk_dot_f64c_haswell", test_dot<f64c_t>, nk_cap_haswell_k, nk_dot_f64c_haswell);
    run_kernel_case("nk_vdot_f64c_haswell", test_vdot<f64c_t>, nk_cap_haswell_k, nk_vdot_f64c_haswell);
    run_kernel_case("nk_dot_f32c_haswell", test_dot<f32c_t>, nk_cap_haswell_k, nk_dot_f32c_haswell);
    run_kernel_case("nk_vdot_f32c_haswell", test_vdot<f32c_t>, nk_cap_haswell_k, nk_vdot_f32c_haswell);
    run_kernel_case("nk_dot_bf16c_haswell", test_dot<bf16c_t>, nk_cap_haswell_k, nk_dot_bf16c_haswell);
    run_kernel_case("nk_vdot_bf16c_haswell", test_vdot<bf16c_t>, nk_cap_haswell_k, nk_vdot_bf16c_haswell);
    run_kernel_case("nk_dot_f16c_haswell", test_dot<f16c_t>, nk_cap_haswell_k, nk_dot_f16c_haswell);
    run_kernel_case("nk_vdot_f16c_haswell", test_vdot<f16c_t>, nk_cap_haswell_k, nk_vdot_f16c_haswell);
    run_kernel_case("nk_dot_f64_haswell", test_dot<f64_t>, nk_cap_haswell_k, nk_dot_f64_haswell);
    run_kernel_case("nk_dot_f32_haswell", test_dot<f32_t>, nk_cap_haswell_k, nk_dot_f32_haswell);
    run_kernel_case("nk_dot_bf16_haswell", test_dot<bf16_t>, nk_cap_haswell_k, nk_dot_bf16_haswell);
    run_kernel_case("nk_dot_f16_haswell", test_dot<f16_t>, nk_cap_haswell_k, nk_dot_f16_haswell);
    run_kernel_case("nk_dot_e5m2_haswell", test_dot<e5m2_t>, nk_cap_haswell_k, nk_dot_e5m2_haswell);
    run_kernel_case("nk_dot_e4m3_haswell", test_dot<e4m3_t>, nk_cap_haswell_k, nk_dot_e4m3_haswell);
    run_kernel_case("nk_dot_e3m2_haswell", test_dot<e3m2_t>, nk_cap_haswell_k, nk_dot_e3m2_haswell);
    run_kernel_case("nk_dot_e2m3_haswell", test_dot<e2m3_t>, nk_cap_haswell_k, nk_dot_e2m3_haswell);
    run_kernel_case("nk_dot_e2m1_haswell", test_dot<e2m1x2_t>, nk_cap_haswell_k, nk_dot_e2m1_haswell);
    run_kernel_case("nk_dot_i8_haswell", test_dot<i8_t>, nk_cap_haswell_k, nk_dot_i8_haswell);
    run_kernel_case("nk_dot_i4_haswell", test_dot<i4x2_t>, nk_cap_haswell_k, nk_dot_i4_haswell);
    run_kernel_case("nk_dot_u8_haswell", test_dot<u8_t>, nk_cap_haswell_k, nk_dot_u8_haswell);
    run_kernel_case("nk_dot_u4_haswell", test_dot<u4x2_t>, nk_cap_haswell_k, nk_dot_u4_haswell);
    run_kernel_case("nk_dot_u1_haswell", test_dot<u1x8_t>, nk_cap_haswell_k, nk_dot_u1_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    run_kernel_case("nk_dot_f64_skylake", test_dot<f64_t>, nk_cap_skylake_k, nk_dot_f64_skylake);
    run_kernel_case("nk_dot_f32_skylake", test_dot<f32_t>, nk_cap_skylake_k, nk_dot_f32_skylake);
    run_kernel_case("nk_dot_bf16_skylake", test_dot<bf16_t>, nk_cap_skylake_k, nk_dot_bf16_skylake);
    run_kernel_case("nk_dot_f16_skylake", test_dot<f16_t>, nk_cap_skylake_k, nk_dot_f16_skylake);
    run_kernel_case("nk_dot_e5m2_skylake", test_dot<e5m2_t>, nk_cap_skylake_k, nk_dot_e5m2_skylake);
    run_kernel_case("nk_dot_e4m3_skylake", test_dot<e4m3_t>, nk_cap_skylake_k, nk_dot_e4m3_skylake);
    run_kernel_case("nk_dot_e3m2_skylake", test_dot<e3m2_t>, nk_cap_skylake_k, nk_dot_e3m2_skylake);
    run_kernel_case("nk_dot_e2m3_skylake", test_dot<e2m3_t>, nk_cap_skylake_k, nk_dot_e2m3_skylake);
    run_kernel_case("nk_dot_e2m1_skylake", test_dot<e2m1x2_t>, nk_cap_skylake_k, nk_dot_e2m1_skylake);
    run_kernel_case("nk_dot_i8_skylake", test_dot<i8_t>, nk_cap_skylake_k, nk_dot_i8_skylake);
    run_kernel_case("nk_dot_u8_skylake", test_dot<u8_t>, nk_cap_skylake_k, nk_dot_u8_skylake);
    run_kernel_case("nk_dot_f32c_skylake", test_dot<f32c_t>, nk_cap_skylake_k, nk_dot_f32c_skylake);
    run_kernel_case("nk_vdot_f32c_skylake", test_vdot<f32c_t>, nk_cap_skylake_k, nk_vdot_f32c_skylake);
    run_kernel_case("nk_dot_f64c_skylake", test_dot<f64c_t>, nk_cap_skylake_k, nk_dot_f64c_skylake);
    run_kernel_case("nk_vdot_f64c_skylake", test_vdot<f64c_t>, nk_cap_skylake_k, nk_vdot_f64c_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
    run_kernel_case("nk_dot_e4m3_icelake", test_dot<e4m3_t>, nk_cap_icelake_k, nk_dot_e4m3_icelake);
    run_kernel_case("nk_dot_e3m2_icelake", test_dot<e3m2_t>, nk_cap_icelake_k, nk_dot_e3m2_icelake);
    run_kernel_case("nk_dot_e2m3_icelake", test_dot<e2m3_t>, nk_cap_icelake_k, nk_dot_e2m3_icelake);
    run_kernel_case("nk_dot_i8_icelake", test_dot<i8_t>, nk_cap_icelake_k, nk_dot_i8_icelake);
    run_kernel_case("nk_dot_i4_icelake", test_dot<i4x2_t>, nk_cap_icelake_k, nk_dot_i4_icelake);
    run_kernel_case("nk_dot_u8_icelake", test_dot<u8_t>, nk_cap_icelake_k, nk_dot_u8_icelake);
    run_kernel_case("nk_dot_u4_icelake", test_dot<u4x2_t>, nk_cap_icelake_k, nk_dot_u4_icelake);
    run_kernel_case("nk_dot_u1_icelake", test_dot<u1x8_t>, nk_cap_icelake_k, nk_dot_u1_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_ALDER
    run_kernel_case("nk_dot_e2m3_alder", test_dot<e2m3_t>, nk_cap_alder_k, nk_dot_e2m3_alder);
    run_kernel_case("nk_dot_e2m1_alder", test_dot<e2m1x2_t>, nk_cap_alder_k, nk_dot_e2m1_alder);
    run_kernel_case("nk_dot_i8_alder", test_dot<i8_t>, nk_cap_alder_k, nk_dot_i8_alder);
    run_kernel_case("nk_dot_u8_alder", test_dot<u8_t>, nk_cap_alder_k, nk_dot_u8_alder);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA
    run_kernel_case("nk_dot_e2m3_sierra", test_dot<e2m3_t>, nk_cap_sierra_k, nk_dot_e2m3_sierra);
    run_kernel_case("nk_dot_e2m1_sierra", test_dot<e2m1x2_t>, nk_cap_sierra_k, nk_dot_e2m1_sierra);
    run_kernel_case("nk_dot_i8_sierra", test_dot<i8_t>, nk_cap_sierra_k, nk_dot_i8_sierra);
    run_kernel_case("nk_dot_u8_sierra", test_dot<u8_t>, nk_cap_sierra_k, nk_dot_u8_sierra);
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_GENOA
    run_kernel_case("nk_dot_bf16_genoa", test_dot<bf16_t>, nk_cap_genoa_k, nk_dot_bf16_genoa);
    run_kernel_case("nk_dot_e5m2_genoa", test_dot<e5m2_t>, nk_cap_genoa_k, nk_dot_e5m2_genoa);
    run_kernel_case("nk_dot_bf16c_genoa", test_dot<bf16c_t>, nk_cap_genoa_k, nk_dot_bf16c_genoa);
    run_kernel_case("nk_vdot_bf16c_genoa", test_vdot<bf16c_t>, nk_cap_genoa_k, nk_vdot_bf16c_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_DIAMOND
    run_kernel_case("nk_dot_f16_diamond", test_dot<f16_t>, nk_cap_diamond_k, nk_dot_f16_diamond);
    run_kernel_case("nk_dot_e5m2_diamond", test_dot<e5m2_t>, nk_cap_diamond_k, nk_dot_e5m2_diamond);
    run_kernel_case("nk_dot_e4m3_diamond", test_dot<e4m3_t>, nk_cap_diamond_k, nk_dot_e4m3_diamond);
#endif // NUMKONG_TARGET_DIAMOND

#if NUMKONG_TARGET_RVV
    run_kernel_case("nk_dot_f64c_rvv", test_dot<f64c_t>, nk_cap_rvv_k, nk_dot_f64c_rvv);
    run_kernel_case("nk_vdot_f64c_rvv", test_vdot<f64c_t>, nk_cap_rvv_k, nk_vdot_f64c_rvv);
    run_kernel_case("nk_dot_f32c_rvv", test_dot<f32c_t>, nk_cap_rvv_k, nk_dot_f32c_rvv);
    run_kernel_case("nk_vdot_f32c_rvv", test_vdot<f32c_t>, nk_cap_rvv_k, nk_vdot_f32c_rvv);
    run_kernel_case("nk_dot_f64_rvv", test_dot<f64_t>, nk_cap_rvv_k, nk_dot_f64_rvv);
    run_kernel_case("nk_dot_f32_rvv", test_dot<f32_t>, nk_cap_rvv_k, nk_dot_f32_rvv);
    run_kernel_case("nk_dot_bf16_rvv", test_dot<bf16_t>, nk_cap_rvv_k, nk_dot_bf16_rvv);
    run_kernel_case("nk_dot_f16_rvv", test_dot<f16_t>, nk_cap_rvv_k, nk_dot_f16_rvv);
    run_kernel_case("nk_dot_e5m2_rvv", test_dot<e5m2_t>, nk_cap_rvv_k, nk_dot_e5m2_rvv);
    run_kernel_case("nk_dot_e4m3_rvv", test_dot<e4m3_t>, nk_cap_rvv_k, nk_dot_e4m3_rvv);
    run_kernel_case("nk_dot_e3m2_rvv", test_dot<e3m2_t>, nk_cap_rvv_k, nk_dot_e3m2_rvv);
    run_kernel_case("nk_dot_e2m3_rvv", test_dot<e2m3_t>, nk_cap_rvv_k, nk_dot_e2m3_rvv);
    run_kernel_case("nk_dot_e2m1_rvv", test_dot<e2m1x2_t>, nk_cap_rvv_k, nk_dot_e2m1_rvv);
    run_kernel_case("nk_dot_i8_rvv", test_dot<i8_t>, nk_cap_rvv_k, nk_dot_i8_rvv);
    run_kernel_case("nk_dot_i4_rvv", test_dot<i4x2_t>, nk_cap_rvv_k, nk_dot_i4_rvv);
    run_kernel_case("nk_dot_u8_rvv", test_dot<u8_t>, nk_cap_rvv_k, nk_dot_u8_rvv);
    run_kernel_case("nk_dot_u4_rvv", test_dot<u4x2_t>, nk_cap_rvv_k, nk_dot_u4_rvv);
    run_kernel_case("nk_dot_u1_rvv", test_dot<u1x8_t>, nk_cap_rvv_k, nk_dot_u1_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
    run_kernel_case("nk_dot_f64_v128relaxed", test_dot<f64_t>, nk_cap_v128relaxed_k, nk_dot_f64_v128relaxed);
    run_kernel_case("nk_dot_f32_v128relaxed", test_dot<f32_t>, nk_cap_v128relaxed_k, nk_dot_f32_v128relaxed);
    run_kernel_case("nk_dot_bf16_v128relaxed", test_dot<bf16_t>, nk_cap_v128relaxed_k, nk_dot_bf16_v128relaxed);
    run_kernel_case("nk_dot_f16_v128relaxed", test_dot<f16_t>, nk_cap_v128relaxed_k, nk_dot_f16_v128relaxed);
    run_kernel_case("nk_dot_e5m2_v128relaxed", test_dot<e5m2_t>, nk_cap_v128relaxed_k, nk_dot_e5m2_v128relaxed);
    run_kernel_case("nk_dot_e4m3_v128relaxed", test_dot<e4m3_t>, nk_cap_v128relaxed_k, nk_dot_e4m3_v128relaxed);
    run_kernel_case("nk_dot_e3m2_v128relaxed", test_dot<e3m2_t>, nk_cap_v128relaxed_k, nk_dot_e3m2_v128relaxed);
    run_kernel_case("nk_dot_e2m3_v128relaxed", test_dot<e2m3_t>, nk_cap_v128relaxed_k, nk_dot_e2m3_v128relaxed);
    run_kernel_case("nk_dot_e2m1_v128relaxed", test_dot<e2m1x2_t>, nk_cap_v128relaxed_k, nk_dot_e2m1_v128relaxed);
    run_kernel_case("nk_dot_i8_v128relaxed", test_dot<i8_t>, nk_cap_v128relaxed_k, nk_dot_i8_v128relaxed);
    run_kernel_case("nk_dot_i4_v128relaxed", test_dot<i4x2_t>, nk_cap_v128relaxed_k, nk_dot_i4_v128relaxed);
    run_kernel_case("nk_dot_u8_v128relaxed", test_dot<u8_t>, nk_cap_v128relaxed_k, nk_dot_u8_v128relaxed);
    run_kernel_case("nk_dot_u4_v128relaxed", test_dot<u4x2_t>, nk_cap_v128relaxed_k, nk_dot_u4_v128relaxed);
    run_kernel_case("nk_dot_f32c_v128relaxed", test_dot<f32c_t>, nk_cap_v128relaxed_k, nk_dot_f32c_v128relaxed);
    run_kernel_case("nk_vdot_f32c_v128relaxed", test_vdot<f32c_t>, nk_cap_v128relaxed_k, nk_vdot_f32c_v128relaxed);
    run_kernel_case("nk_dot_f64c_v128relaxed", test_dot<f64c_t>, nk_cap_v128relaxed_k, nk_dot_f64c_v128relaxed);
    run_kernel_case("nk_vdot_f64c_v128relaxed", test_vdot<f64c_t>, nk_cap_v128relaxed_k, nk_vdot_f64c_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_V128
    run_kernel_case("nk_dot_bf16_v128", test_dot<bf16_t>, nk_cap_v128_k, nk_dot_bf16_v128);
    run_kernel_case("nk_dot_i8_v128", test_dot<i8_t>, nk_cap_v128_k, nk_dot_i8_v128);
    run_kernel_case("nk_dot_u8_v128", test_dot<u8_t>, nk_cap_v128_k, nk_dot_u8_v128);
    run_kernel_case("nk_dot_u1_v128", test_dot<u1x8_t>, nk_cap_v128_k, nk_dot_u1_v128);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_RVVHALF
    run_kernel_case("nk_dot_f16_rvvhalf", test_dot<f16_t>, nk_cap_rvvhalf_k, nk_dot_f16_rvvhalf);
    run_kernel_case("nk_dot_e5m2_rvvhalf", test_dot<e5m2_t>, nk_cap_rvvhalf_k, nk_dot_e5m2_rvvhalf);
    run_kernel_case("nk_dot_e4m3_rvvhalf", test_dot<e4m3_t>, nk_cap_rvvhalf_k, nk_dot_e4m3_rvvhalf);
#endif // NUMKONG_TARGET_RVVHALF

#if NUMKONG_TARGET_RVVBF16
    run_kernel_case("nk_dot_bf16_rvvbf16", test_dot<bf16_t>, nk_cap_rvvbf16_k, nk_dot_bf16_rvvbf16);
    run_kernel_case("nk_dot_e5m2_rvvbf16", test_dot<e5m2_t>, nk_cap_rvvbf16_k, nk_dot_e5m2_rvvbf16);
    run_kernel_case("nk_dot_e4m3_rvvbf16", test_dot<e4m3_t>, nk_cap_rvvbf16_k, nk_dot_e4m3_rvvbf16);
#endif // NUMKONG_TARGET_RVVBF16

#if NUMKONG_TARGET_RVVBB
    run_kernel_case("nk_dot_u1_rvvbb", test_dot<u1x8_t>, nk_cap_rvvbb_k, nk_dot_u1_rvvbb);
#endif // NUMKONG_TARGET_RVVBB

#if NUMKONG_TARGET_LOONGSONASX
    run_kernel_case("nk_dot_f64_loongsonasx", test_dot<f64_t>, nk_cap_loongsonasx_k, nk_dot_f64_loongsonasx);
    run_kernel_case("nk_dot_f32_loongsonasx", test_dot<f32_t>, nk_cap_loongsonasx_k, nk_dot_f32_loongsonasx);
    run_kernel_case("nk_dot_bf16_loongsonasx", test_dot<bf16_t>, nk_cap_loongsonasx_k, nk_dot_bf16_loongsonasx);
    run_kernel_case("nk_dot_i8_loongsonasx", test_dot<i8_t>, nk_cap_loongsonasx_k, nk_dot_i8_loongsonasx);
    run_kernel_case("nk_dot_u8_loongsonasx", test_dot<u8_t>, nk_cap_loongsonasx_k, nk_dot_u8_loongsonasx);
#endif // NUMKONG_TARGET_LOONGSONASX

#if NUMKONG_TARGET_POWERVSX
    run_kernel_case("nk_dot_f64_powervsx", test_dot<f64_t>, nk_cap_powervsx_k, nk_dot_f64_powervsx);
    run_kernel_case("nk_dot_f32_powervsx", test_dot<f32_t>, nk_cap_powervsx_k, nk_dot_f32_powervsx);
    run_kernel_case("nk_dot_bf16_powervsx", test_dot<bf16_t>, nk_cap_powervsx_k, nk_dot_bf16_powervsx);
    run_kernel_case("nk_dot_f16_powervsx", test_dot<f16_t>, nk_cap_powervsx_k, nk_dot_f16_powervsx);
    run_kernel_case("nk_dot_i8_powervsx", test_dot<i8_t>, nk_cap_powervsx_k, nk_dot_i8_powervsx);
    run_kernel_case("nk_dot_u8_powervsx", test_dot<u8_t>, nk_cap_powervsx_k, nk_dot_u8_powervsx);
    run_kernel_case("nk_dot_u1_powervsx", test_dot<u1x8_t>, nk_cap_powervsx_k, nk_dot_u1_powervsx);
#endif // NUMKONG_TARGET_POWERVSX
}

} // namespace ashvardanian::numkong::test
