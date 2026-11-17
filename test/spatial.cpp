/**
 *  @file test/spatial.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Spatial distance tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "numkong/spatial.hpp"

namespace ashvardanian::numkong::test {

/** Unified squared Euclidean distance test for float types. Works with f32_t, f64_t, f16_t, bf16_t
 *  wrapper types. */
template <typename scalar_type_>
error_stats_t test_sqeuclidean(typename scalar_type_::sqeuclidean_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::sqeuclidean_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_sqeuclidean_error_bound(scalar_t::dtype()), nk_spatial_sum_error_bound(scalar_t::dtype()));
    split_mix_generator_t generator(case_stream_key());
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const n = nk::divide_round_up(test_run.settings.dense_dimensions, dims_per_value) * dims_per_value;
    auto a = make_vector<scalar_t>(n), b = make_vector<scalar_t>(n);

    if constexpr (std::is_same_v<scalar_t, f64_t> || std::is_same_v<scalar_t, bf16_t>) {
        std::fill_n(a.raw_values_data(), n, scalar_t(std::is_same_v<scalar_t, f64_t> ? 1e154 : 1e19).raw_);
        std::fill_n(b.raw_values_data(), n, scalar_t(0.0).raw_);
        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_, nullptr));
        stats.expect_true(std::isinf(result.raw_) && result.raw_ > 0, "overflowing squares give +inf");
    }

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_, nullptr));

        reference_t reference;
        stats.expect_status(
            nk::sqeuclidean<scalar_t, reference_t>(a.values_data(), b.values_data(), n, &reference, no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

/** Unified angular, or cosine, distance test for float types. Works with f32_t, f64_t, f16_t,
 *  bf16_t wrapper types. */
template <typename scalar_type_>
error_stats_t test_angular(typename scalar_type_::angular_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::angular_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_angular_error_bound(scalar_t::dtype()), nk_spatial_sum_error_bound(scalar_t::dtype()));
    split_mix_generator_t generator(case_stream_key());
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const n = nk::divide_round_up(test_run.settings.dense_dimensions, dims_per_value) * dims_per_value;
    auto a = make_vector<scalar_t>(n), b = make_vector<scalar_t>(n);

    if constexpr (std::is_same_v<scalar_t, f32_t> || std::is_same_v<scalar_t, f64_t>) {
        fill_random(generator, a);
        fill_random(generator, b);
        a.raw_values_data()[0] = std::numeric_limits<raw_t>::quiet_NaN();
        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_, nullptr));
        stats.expect_true(std::isnan(result.raw_), "a NaN input gives NaN");
    }

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_, nullptr));

        reference_t reference;
        stats.expect_status(
            nk::angular<scalar_t, reference_t>(a.values_data(), b.values_data(), n, &reference, no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

/** Unified Euclidean distance test for float types. Works with f32_t, f64_t, f16_t, bf16_t, e2m3_t,
 *  e3m2_t wrapper types. */
template <typename scalar_type_>
error_stats_t test_euclidean(typename scalar_type_::euclidean_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::euclidean_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_euclidean_error_bound(scalar_t::dtype()), nk_spatial_sum_error_bound(scalar_t::dtype()));
    split_mix_generator_t generator(case_stream_key());
    std::size_t const dims_per_value = nk::dimensions_per_value<scalar_t>();
    std::size_t const n = nk::divide_round_up(test_run.settings.dense_dimensions, dims_per_value) * dims_per_value;
    auto a = make_vector<scalar_t>(n), b = make_vector<scalar_t>(n);

    if constexpr (std::is_same_v<scalar_t, f64_t> || std::is_same_v<scalar_t, bf16_t>) {
        std::fill_n(a.raw_values_data(), n, scalar_t(std::is_same_v<scalar_t, f64_t> ? 1e154 : 1e19).raw_);
        std::fill_n(b.raw_values_data(), n, scalar_t(0.0).raw_);
        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_, nullptr));
        stats.expect_true(std::isinf(result.raw_) && result.raw_ > 0, "overflowing squares give +inf");
    }
    if constexpr (std::is_same_v<scalar_t, f32_t> || std::is_same_v<scalar_t, f64_t>) {
        fill_random(generator, a);
        fill_random(generator, b);
        a.raw_values_data()[0] = std::numeric_limits<raw_t>::quiet_NaN();
        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_, nullptr));
        stats.expect_true(std::isnan(result.raw_), "a NaN input gives NaN");
    }

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), n, &result.raw_, nullptr));

        reference_t reference;
        stats.expect_status(
            nk::euclidean<scalar_t, reference_t>(a.values_data(), b.values_data(), n, &reference, no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

void run_spatial_cases() {
    auto const serial = nk_cap_serial_k;

    run_kernel_case("nk_sqeuclidean_f32_serial", test_sqeuclidean<f32_t>, serial, nk_sqeuclidean_f32_serial);
    run_kernel_case("nk_sqeuclidean_f64_serial", test_sqeuclidean<f64_t>, serial, nk_sqeuclidean_f64_serial);
    run_kernel_case("nk_sqeuclidean_f16_serial", test_sqeuclidean<f16_t>, serial, nk_sqeuclidean_f16_serial);
    run_kernel_case("nk_sqeuclidean_bf16_serial", test_sqeuclidean<bf16_t>, serial, nk_sqeuclidean_bf16_serial);
    run_kernel_case("nk_sqeuclidean_e2m3_serial", test_sqeuclidean<e2m3_t>, serial, nk_sqeuclidean_e2m3_serial);
    run_kernel_case("nk_sqeuclidean_e3m2_serial", test_sqeuclidean<e3m2_t>, serial, nk_sqeuclidean_e3m2_serial);
    run_kernel_case("nk_euclidean_f32_serial", test_euclidean<f32_t>, serial, nk_euclidean_f32_serial);
    run_kernel_case("nk_euclidean_f64_serial", test_euclidean<f64_t>, serial, nk_euclidean_f64_serial);
    run_kernel_case("nk_euclidean_f16_serial", test_euclidean<f16_t>, serial, nk_euclidean_f16_serial);
    run_kernel_case("nk_euclidean_bf16_serial", test_euclidean<bf16_t>, serial, nk_euclidean_bf16_serial);
    run_kernel_case("nk_euclidean_e2m3_serial", test_euclidean<e2m3_t>, serial, nk_euclidean_e2m3_serial);
    run_kernel_case("nk_euclidean_e3m2_serial", test_euclidean<e3m2_t>, serial, nk_euclidean_e3m2_serial);
    run_kernel_case("nk_angular_f32_serial", test_angular<f32_t>, serial, nk_angular_f32_serial);
    run_kernel_case("nk_angular_f64_serial", test_angular<f64_t>, serial, nk_angular_f64_serial);
    run_kernel_case("nk_angular_f16_serial", test_angular<f16_t>, serial, nk_angular_f16_serial);
    run_kernel_case("nk_angular_bf16_serial", test_angular<bf16_t>, serial, nk_angular_bf16_serial);
    run_kernel_case("nk_angular_e2m3_serial", test_angular<e2m3_t>, serial, nk_angular_e2m3_serial);
    run_kernel_case("nk_angular_e3m2_serial", test_angular<e3m2_t>, serial, nk_angular_e3m2_serial);
    run_kernel_case("nk_sqeuclidean_i4_serial", test_sqeuclidean<i4x2_t>, serial, nk_sqeuclidean_i4_serial);
    run_kernel_case("nk_sqeuclidean_u4_serial", test_sqeuclidean<u4x2_t>, serial, nk_sqeuclidean_u4_serial);
    run_kernel_case("nk_euclidean_i4_serial", test_euclidean<i4x2_t>, serial, nk_euclidean_i4_serial);
    run_kernel_case("nk_euclidean_u4_serial", test_euclidean<u4x2_t>, serial, nk_euclidean_u4_serial);
    run_kernel_case("nk_angular_i4_serial", test_angular<i4x2_t>, serial, nk_angular_i4_serial);
    run_kernel_case("nk_angular_u4_serial", test_angular<u4x2_t>, serial, nk_angular_u4_serial);
    run_kernel_case("nk_angular_e4m3_serial", test_angular<e4m3_t>, serial, nk_angular_e4m3_serial);
    run_kernel_case("nk_sqeuclidean_e4m3_serial", test_sqeuclidean<e4m3_t>, serial, nk_sqeuclidean_e4m3_serial);
    run_kernel_case("nk_euclidean_e4m3_serial", test_euclidean<e4m3_t>, serial, nk_euclidean_e4m3_serial);
    run_kernel_case("nk_angular_e5m2_serial", test_angular<e5m2_t>, serial, nk_angular_e5m2_serial);
    run_kernel_case("nk_sqeuclidean_e5m2_serial", test_sqeuclidean<e5m2_t>, serial, nk_sqeuclidean_e5m2_serial);
    run_kernel_case("nk_euclidean_e5m2_serial", test_euclidean<e5m2_t>, serial, nk_euclidean_e5m2_serial);
    run_kernel_case("nk_angular_i8_serial", test_angular<i8_t>, serial, nk_angular_i8_serial);
    run_kernel_case("nk_sqeuclidean_i8_serial", test_sqeuclidean<i8_t>, serial, nk_sqeuclidean_i8_serial);
    run_kernel_case("nk_euclidean_i8_serial", test_euclidean<i8_t>, serial, nk_euclidean_i8_serial);
    run_kernel_case("nk_angular_u8_serial", test_angular<u8_t>, serial, nk_angular_u8_serial);
    run_kernel_case("nk_sqeuclidean_u8_serial", test_sqeuclidean<u8_t>, serial, nk_sqeuclidean_u8_serial);
    run_kernel_case("nk_euclidean_u8_serial", test_euclidean<u8_t>, serial, nk_euclidean_u8_serial);

#if !NUMKONG_HEADER_ONLY
    run_kernel_case("nk_sqeuclidean_f32_best", test_sqeuclidean<f32_t>, serial, cpu_best<nk_sqeuclidean_f32_best>);
    run_kernel_case("nk_sqeuclidean_f64_best", test_sqeuclidean<f64_t>, serial, cpu_best<nk_sqeuclidean_f64_best>);
    run_kernel_case("nk_sqeuclidean_f16_best", test_sqeuclidean<f16_t>, serial, cpu_best<nk_sqeuclidean_f16_best>);
    run_kernel_case("nk_sqeuclidean_bf16_best", test_sqeuclidean<bf16_t>, serial, cpu_best<nk_sqeuclidean_bf16_best>);
    run_kernel_case("nk_sqeuclidean_e2m3_best", test_sqeuclidean<e2m3_t>, serial, cpu_best<nk_sqeuclidean_e2m3_best>);
    run_kernel_case("nk_sqeuclidean_e3m2_best", test_sqeuclidean<e3m2_t>, serial, cpu_best<nk_sqeuclidean_e3m2_best>);
    run_kernel_case("nk_euclidean_f32_best", test_euclidean<f32_t>, serial, cpu_best<nk_euclidean_f32_best>);
    run_kernel_case("nk_euclidean_f64_best", test_euclidean<f64_t>, serial, cpu_best<nk_euclidean_f64_best>);
    run_kernel_case("nk_euclidean_f16_best", test_euclidean<f16_t>, serial, cpu_best<nk_euclidean_f16_best>);
    run_kernel_case("nk_euclidean_bf16_best", test_euclidean<bf16_t>, serial, cpu_best<nk_euclidean_bf16_best>);
    run_kernel_case("nk_euclidean_e2m3_best", test_euclidean<e2m3_t>, serial, cpu_best<nk_euclidean_e2m3_best>);
    run_kernel_case("nk_euclidean_e3m2_best", test_euclidean<e3m2_t>, serial, cpu_best<nk_euclidean_e3m2_best>);
    run_kernel_case("nk_angular_f32_best", test_angular<f32_t>, serial, cpu_best<nk_angular_f32_best>);
    run_kernel_case("nk_angular_f64_best", test_angular<f64_t>, serial, cpu_best<nk_angular_f64_best>);
    run_kernel_case("nk_angular_f16_best", test_angular<f16_t>, serial, cpu_best<nk_angular_f16_best>);
    run_kernel_case("nk_angular_bf16_best", test_angular<bf16_t>, serial, cpu_best<nk_angular_bf16_best>);
    run_kernel_case("nk_angular_e2m3_best", test_angular<e2m3_t>, serial, cpu_best<nk_angular_e2m3_best>);
    run_kernel_case("nk_angular_e3m2_best", test_angular<e3m2_t>, serial, cpu_best<nk_angular_e3m2_best>);
    run_kernel_case("nk_sqeuclidean_i4_best", test_sqeuclidean<i4x2_t>, serial, cpu_best<nk_sqeuclidean_i4_best>);
    run_kernel_case("nk_sqeuclidean_u4_best", test_sqeuclidean<u4x2_t>, serial, cpu_best<nk_sqeuclidean_u4_best>);
    run_kernel_case("nk_angular_i4_best", test_angular<i4x2_t>, serial, cpu_best<nk_angular_i4_best>);
    run_kernel_case("nk_angular_u4_best", test_angular<u4x2_t>, serial, cpu_best<nk_angular_u4_best>);
#endif

#if NUMKONG_TARGET_NEON
    run_kernel_case("nk_sqeuclidean_f64_neon", test_sqeuclidean<f64_t>, nk_cap_neon_k, nk_sqeuclidean_f64_neon);
    run_kernel_case("nk_sqeuclidean_f32_neon", test_sqeuclidean<f32_t>, nk_cap_neon_k, nk_sqeuclidean_f32_neon);
    run_kernel_case("nk_sqeuclidean_bf16_neon", test_sqeuclidean<bf16_t>, nk_cap_neon_k, nk_sqeuclidean_bf16_neon);
    run_kernel_case("nk_sqeuclidean_f16_neon", test_sqeuclidean<f16_t>, nk_cap_neon_k, nk_sqeuclidean_f16_neon);
    run_kernel_case("nk_sqeuclidean_e5m2_neon", test_sqeuclidean<e5m2_t>, nk_cap_neon_k, nk_sqeuclidean_e5m2_neon);
    run_kernel_case("nk_sqeuclidean_e4m3_neon", test_sqeuclidean<e4m3_t>, nk_cap_neon_k, nk_sqeuclidean_e4m3_neon);
    run_kernel_case("nk_sqeuclidean_e3m2_neon", test_sqeuclidean<e3m2_t>, nk_cap_neon_k, nk_sqeuclidean_e3m2_neon);
    run_kernel_case("nk_sqeuclidean_e2m3_neon", test_sqeuclidean<e2m3_t>, nk_cap_neon_k, nk_sqeuclidean_e2m3_neon);
    run_kernel_case("nk_euclidean_f64_neon", test_euclidean<f64_t>, nk_cap_neon_k, nk_euclidean_f64_neon);
    run_kernel_case("nk_euclidean_f32_neon", test_euclidean<f32_t>, nk_cap_neon_k, nk_euclidean_f32_neon);
    run_kernel_case("nk_euclidean_bf16_neon", test_euclidean<bf16_t>, nk_cap_neon_k, nk_euclidean_bf16_neon);
    run_kernel_case("nk_euclidean_f16_neon", test_euclidean<f16_t>, nk_cap_neon_k, nk_euclidean_f16_neon);
    run_kernel_case("nk_euclidean_e5m2_neon", test_euclidean<e5m2_t>, nk_cap_neon_k, nk_euclidean_e5m2_neon);
    run_kernel_case("nk_euclidean_e4m3_neon", test_euclidean<e4m3_t>, nk_cap_neon_k, nk_euclidean_e4m3_neon);
    run_kernel_case("nk_euclidean_e3m2_neon", test_euclidean<e3m2_t>, nk_cap_neon_k, nk_euclidean_e3m2_neon);
    run_kernel_case("nk_euclidean_e2m3_neon", test_euclidean<e2m3_t>, nk_cap_neon_k, nk_euclidean_e2m3_neon);
    run_kernel_case("nk_angular_f64_neon", test_angular<f64_t>, nk_cap_neon_k, nk_angular_f64_neon);
    run_kernel_case("nk_angular_f32_neon", test_angular<f32_t>, nk_cap_neon_k, nk_angular_f32_neon);
    run_kernel_case("nk_angular_bf16_neon", test_angular<bf16_t>, nk_cap_neon_k, nk_angular_bf16_neon);
    run_kernel_case("nk_angular_f16_neon", test_angular<f16_t>, nk_cap_neon_k, nk_angular_f16_neon);
    run_kernel_case("nk_angular_e5m2_neon", test_angular<e5m2_t>, nk_cap_neon_k, nk_angular_e5m2_neon);
    run_kernel_case("nk_angular_e4m3_neon", test_angular<e4m3_t>, nk_cap_neon_k, nk_angular_e4m3_neon);
    run_kernel_case("nk_angular_e3m2_neon", test_angular<e3m2_t>, nk_cap_neon_k, nk_angular_e3m2_neon);
    run_kernel_case("nk_angular_e2m3_neon", test_angular<e2m3_t>, nk_cap_neon_k, nk_angular_e2m3_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
    auto const neonbfdot = nk_cap_neonbfdot_k;
    run_kernel_case("nk_sqeuclidean_bf16_neonbfdot", test_sqeuclidean<bf16_t>, neonbfdot,
                    nk_sqeuclidean_bf16_neonbfdot);
    run_kernel_case("nk_euclidean_bf16_neonbfdot", test_euclidean<bf16_t>, neonbfdot, nk_euclidean_bf16_neonbfdot);
    run_kernel_case("nk_angular_bf16_neonbfdot", test_angular<bf16_t>, neonbfdot, nk_angular_bf16_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONSDOT
    auto const neonsdot = nk_cap_neonsdot_k;
    run_kernel_case("nk_angular_i8_neonsdot", test_angular<i8_t>, neonsdot, nk_angular_i8_neonsdot);
    run_kernel_case("nk_sqeuclidean_i8_neonsdot", test_sqeuclidean<i8_t>, neonsdot, nk_sqeuclidean_i8_neonsdot);
    run_kernel_case("nk_euclidean_i8_neonsdot", test_euclidean<i8_t>, neonsdot, nk_euclidean_i8_neonsdot);
    run_kernel_case("nk_angular_u8_neonsdot", test_angular<u8_t>, neonsdot, nk_angular_u8_neonsdot);
    run_kernel_case("nk_sqeuclidean_u8_neonsdot", test_sqeuclidean<u8_t>, neonsdot, nk_sqeuclidean_u8_neonsdot);
    run_kernel_case("nk_euclidean_u8_neonsdot", test_euclidean<u8_t>, neonsdot, nk_euclidean_u8_neonsdot);
    run_kernel_case("nk_angular_i4_neonsdot", test_angular<i4x2_t>, neonsdot, nk_angular_i4_neonsdot);
    run_kernel_case("nk_sqeuclidean_i4_neonsdot", test_sqeuclidean<i4x2_t>, neonsdot, nk_sqeuclidean_i4_neonsdot);
    run_kernel_case("nk_euclidean_i4_neonsdot", test_euclidean<i4x2_t>, neonsdot, nk_euclidean_i4_neonsdot);
    run_kernel_case("nk_angular_u4_neonsdot", test_angular<u4x2_t>, neonsdot, nk_angular_u4_neonsdot);
    run_kernel_case("nk_sqeuclidean_u4_neonsdot", test_sqeuclidean<u4x2_t>, neonsdot, nk_sqeuclidean_u4_neonsdot);
    run_kernel_case("nk_euclidean_u4_neonsdot", test_euclidean<u4x2_t>, neonsdot, nk_euclidean_u4_neonsdot);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_NEONFP8
    auto const neonfp8 = nk_cap_neonfp8_k;
    run_kernel_case("nk_angular_e4m3_neonfp8", test_angular<e4m3_t>, neonfp8, nk_angular_e4m3_neonfp8);
    run_kernel_case("nk_sqeuclidean_e4m3_neonfp8", test_sqeuclidean<e4m3_t>, neonfp8, nk_sqeuclidean_e4m3_neonfp8);
    run_kernel_case("nk_euclidean_e4m3_neonfp8", test_euclidean<e4m3_t>, neonfp8, nk_euclidean_e4m3_neonfp8);
    run_kernel_case("nk_angular_e5m2_neonfp8", test_angular<e5m2_t>, neonfp8, nk_angular_e5m2_neonfp8);
    run_kernel_case("nk_sqeuclidean_e5m2_neonfp8", test_sqeuclidean<e5m2_t>, neonfp8, nk_sqeuclidean_e5m2_neonfp8);
    run_kernel_case("nk_euclidean_e5m2_neonfp8", test_euclidean<e5m2_t>, neonfp8, nk_euclidean_e5m2_neonfp8);
    run_kernel_case("nk_angular_e2m3_neonfp8", test_angular<e2m3_t>, neonfp8, nk_angular_e2m3_neonfp8);
    run_kernel_case("nk_sqeuclidean_e2m3_neonfp8", test_sqeuclidean<e2m3_t>, neonfp8, nk_sqeuclidean_e2m3_neonfp8);
    run_kernel_case("nk_euclidean_e2m3_neonfp8", test_euclidean<e2m3_t>, neonfp8, nk_euclidean_e2m3_neonfp8);
    run_kernel_case("nk_angular_e3m2_neonfp8", test_angular<e3m2_t>, neonfp8, nk_angular_e3m2_neonfp8);
    run_kernel_case("nk_sqeuclidean_e3m2_neonfp8", test_sqeuclidean<e3m2_t>, neonfp8, nk_sqeuclidean_e3m2_neonfp8);
    run_kernel_case("nk_euclidean_e3m2_neonfp8", test_euclidean<e3m2_t>, neonfp8, nk_euclidean_e3m2_neonfp8);
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_SVE
    run_kernel_case("nk_angular_f64_sve", test_angular<f64_t>, nk_cap_sve_k, nk_angular_f64_sve);
    run_kernel_case("nk_sqeuclidean_f64_sve", test_sqeuclidean<f64_t>, nk_cap_sve_k, nk_sqeuclidean_f64_sve);
    run_kernel_case("nk_euclidean_f64_sve", test_euclidean<f64_t>, nk_cap_sve_k, nk_euclidean_f64_sve);
    run_kernel_case("nk_angular_f32_sve", test_angular<f32_t>, nk_cap_sve_k, nk_angular_f32_sve);
    run_kernel_case("nk_sqeuclidean_f32_sve", test_sqeuclidean<f32_t>, nk_cap_sve_k, nk_sqeuclidean_f32_sve);
    run_kernel_case("nk_euclidean_f32_sve", test_euclidean<f32_t>, nk_cap_sve_k, nk_euclidean_f32_sve);
#endif // NUMKONG_TARGET_SVE

#if NUMKONG_TARGET_SVEHALF
    auto const svehalf = nk_cap_svehalf_k;
    run_kernel_case("nk_angular_f16_svehalf", test_angular<f16_t>, svehalf, nk_angular_f16_svehalf);
    run_kernel_case("nk_sqeuclidean_f16_svehalf", test_sqeuclidean<f16_t>, svehalf, nk_sqeuclidean_f16_svehalf);
    run_kernel_case("nk_euclidean_f16_svehalf", test_euclidean<f16_t>, svehalf, nk_euclidean_f16_svehalf);
#endif // NUMKONG_TARGET_SVEHALF

#if NUMKONG_TARGET_SVEBFDOT
    auto const svebfdot = nk_cap_svebfdot_k;
    run_kernel_case("nk_angular_bf16_svebfdot", test_angular<bf16_t>, svebfdot, nk_angular_bf16_svebfdot);
    run_kernel_case("nk_sqeuclidean_bf16_svebfdot", test_sqeuclidean<bf16_t>, svebfdot, nk_sqeuclidean_bf16_svebfdot);
    run_kernel_case("nk_euclidean_bf16_svebfdot", test_euclidean<bf16_t>, svebfdot, nk_euclidean_bf16_svebfdot);
#endif // NUMKONG_TARGET_SVEBFDOT

#if NUMKONG_TARGET_SVESDOT
    run_kernel_case("nk_angular_i8_svesdot", test_angular<i8_t>, nk_cap_svesdot_k, nk_angular_i8_svesdot);
    run_kernel_case("nk_sqeuclidean_i8_svesdot", test_sqeuclidean<i8_t>, nk_cap_svesdot_k, nk_sqeuclidean_i8_svesdot);
    run_kernel_case("nk_euclidean_i8_svesdot", test_euclidean<i8_t>, nk_cap_svesdot_k, nk_euclidean_i8_svesdot);
    run_kernel_case("nk_angular_u8_svesdot", test_angular<u8_t>, nk_cap_svesdot_k, nk_angular_u8_svesdot);
    run_kernel_case("nk_sqeuclidean_u8_svesdot", test_sqeuclidean<u8_t>, nk_cap_svesdot_k, nk_sqeuclidean_u8_svesdot);
    run_kernel_case("nk_euclidean_u8_svesdot", test_euclidean<u8_t>, nk_cap_svesdot_k, nk_euclidean_u8_svesdot);
#endif // NUMKONG_TARGET_SVESDOT

#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("nk_angular_f64_haswell", test_angular<f64_t>, haswell, nk_angular_f64_haswell);
    run_kernel_case("nk_sqeuclidean_f64_haswell", test_sqeuclidean<f64_t>, haswell, nk_sqeuclidean_f64_haswell);
    run_kernel_case("nk_euclidean_f64_haswell", test_euclidean<f64_t>, haswell, nk_euclidean_f64_haswell);
    run_kernel_case("nk_angular_f32_haswell", test_angular<f32_t>, haswell, nk_angular_f32_haswell);
    run_kernel_case("nk_sqeuclidean_f32_haswell", test_sqeuclidean<f32_t>, haswell, nk_sqeuclidean_f32_haswell);
    run_kernel_case("nk_euclidean_f32_haswell", test_euclidean<f32_t>, haswell, nk_euclidean_f32_haswell);
    run_kernel_case("nk_angular_bf16_haswell", test_angular<bf16_t>, haswell, nk_angular_bf16_haswell);
    run_kernel_case("nk_sqeuclidean_bf16_haswell", test_sqeuclidean<bf16_t>, haswell, nk_sqeuclidean_bf16_haswell);
    run_kernel_case("nk_euclidean_bf16_haswell", test_euclidean<bf16_t>, haswell, nk_euclidean_bf16_haswell);
    run_kernel_case("nk_angular_f16_haswell", test_angular<f16_t>, haswell, nk_angular_f16_haswell);
    run_kernel_case("nk_sqeuclidean_f16_haswell", test_sqeuclidean<f16_t>, haswell, nk_sqeuclidean_f16_haswell);
    run_kernel_case("nk_euclidean_f16_haswell", test_euclidean<f16_t>, haswell, nk_euclidean_f16_haswell);
    run_kernel_case("nk_angular_e5m2_haswell", test_angular<e5m2_t>, haswell, nk_angular_e5m2_haswell);
    run_kernel_case("nk_sqeuclidean_e5m2_haswell", test_sqeuclidean<e5m2_t>, haswell, nk_sqeuclidean_e5m2_haswell);
    run_kernel_case("nk_euclidean_e5m2_haswell", test_euclidean<e5m2_t>, haswell, nk_euclidean_e5m2_haswell);
    run_kernel_case("nk_angular_e4m3_haswell", test_angular<e4m3_t>, haswell, nk_angular_e4m3_haswell);
    run_kernel_case("nk_sqeuclidean_e4m3_haswell", test_sqeuclidean<e4m3_t>, haswell, nk_sqeuclidean_e4m3_haswell);
    run_kernel_case("nk_euclidean_e4m3_haswell", test_euclidean<e4m3_t>, haswell, nk_euclidean_e4m3_haswell);
    run_kernel_case("nk_angular_e3m2_haswell", test_angular<e3m2_t>, haswell, nk_angular_e3m2_haswell);
    run_kernel_case("nk_sqeuclidean_e3m2_haswell", test_sqeuclidean<e3m2_t>, haswell, nk_sqeuclidean_e3m2_haswell);
    run_kernel_case("nk_euclidean_e3m2_haswell", test_euclidean<e3m2_t>, haswell, nk_euclidean_e3m2_haswell);
    run_kernel_case("nk_angular_e2m3_haswell", test_angular<e2m3_t>, haswell, nk_angular_e2m3_haswell);
    run_kernel_case("nk_sqeuclidean_e2m3_haswell", test_sqeuclidean<e2m3_t>, haswell, nk_sqeuclidean_e2m3_haswell);
    run_kernel_case("nk_euclidean_e2m3_haswell", test_euclidean<e2m3_t>, haswell, nk_euclidean_e2m3_haswell);
    run_kernel_case("nk_angular_i8_haswell", test_angular<i8_t>, haswell, nk_angular_i8_haswell);
    run_kernel_case("nk_sqeuclidean_i8_haswell", test_sqeuclidean<i8_t>, haswell, nk_sqeuclidean_i8_haswell);
    run_kernel_case("nk_euclidean_i8_haswell", test_euclidean<i8_t>, haswell, nk_euclidean_i8_haswell);
    run_kernel_case("nk_angular_u8_haswell", test_angular<u8_t>, haswell, nk_angular_u8_haswell);
    run_kernel_case("nk_sqeuclidean_u8_haswell", test_sqeuclidean<u8_t>, haswell, nk_sqeuclidean_u8_haswell);
    run_kernel_case("nk_euclidean_u8_haswell", test_euclidean<u8_t>, haswell, nk_euclidean_u8_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    auto const skylake = nk_cap_skylake_k;
    run_kernel_case("nk_angular_f64_skylake", test_angular<f64_t>, skylake, nk_angular_f64_skylake);
    run_kernel_case("nk_sqeuclidean_f64_skylake", test_sqeuclidean<f64_t>, skylake, nk_sqeuclidean_f64_skylake);
    run_kernel_case("nk_euclidean_f64_skylake", test_euclidean<f64_t>, skylake, nk_euclidean_f64_skylake);
    run_kernel_case("nk_angular_f32_skylake", test_angular<f32_t>, skylake, nk_angular_f32_skylake);
    run_kernel_case("nk_sqeuclidean_f32_skylake", test_sqeuclidean<f32_t>, skylake, nk_sqeuclidean_f32_skylake);
    run_kernel_case("nk_euclidean_f32_skylake", test_euclidean<f32_t>, skylake, nk_euclidean_f32_skylake);
    run_kernel_case("nk_angular_f16_skylake", test_angular<f16_t>, skylake, nk_angular_f16_skylake);
    run_kernel_case("nk_sqeuclidean_f16_skylake", test_sqeuclidean<f16_t>, skylake, nk_sqeuclidean_f16_skylake);
    run_kernel_case("nk_euclidean_f16_skylake", test_euclidean<f16_t>, skylake, nk_euclidean_f16_skylake);
    run_kernel_case("nk_angular_e5m2_skylake", test_angular<e5m2_t>, skylake, nk_angular_e5m2_skylake);
    run_kernel_case("nk_sqeuclidean_e5m2_skylake", test_sqeuclidean<e5m2_t>, skylake, nk_sqeuclidean_e5m2_skylake);
    run_kernel_case("nk_euclidean_e5m2_skylake", test_euclidean<e5m2_t>, skylake, nk_euclidean_e5m2_skylake);
    run_kernel_case("nk_angular_e4m3_skylake", test_angular<e4m3_t>, skylake, nk_angular_e4m3_skylake);
    run_kernel_case("nk_sqeuclidean_e4m3_skylake", test_sqeuclidean<e4m3_t>, skylake, nk_sqeuclidean_e4m3_skylake);
    run_kernel_case("nk_euclidean_e4m3_skylake", test_euclidean<e4m3_t>, skylake, nk_euclidean_e4m3_skylake);
    run_kernel_case("nk_angular_e3m2_skylake", test_angular<e3m2_t>, skylake, nk_angular_e3m2_skylake);
    run_kernel_case("nk_sqeuclidean_e3m2_skylake", test_sqeuclidean<e3m2_t>, skylake, nk_sqeuclidean_e3m2_skylake);
    run_kernel_case("nk_euclidean_e3m2_skylake", test_euclidean<e3m2_t>, skylake, nk_euclidean_e3m2_skylake);
    run_kernel_case("nk_angular_e2m3_skylake", test_angular<e2m3_t>, skylake, nk_angular_e2m3_skylake);
    run_kernel_case("nk_sqeuclidean_e2m3_skylake", test_sqeuclidean<e2m3_t>, skylake, nk_sqeuclidean_e2m3_skylake);
    run_kernel_case("nk_euclidean_e2m3_skylake", test_euclidean<e2m3_t>, skylake, nk_euclidean_e2m3_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
    auto const icelake = nk_cap_icelake_k;
    run_kernel_case("nk_angular_i8_icelake", test_angular<i8_t>, icelake, nk_angular_i8_icelake);
    run_kernel_case("nk_sqeuclidean_i8_icelake", test_sqeuclidean<i8_t>, icelake, nk_sqeuclidean_i8_icelake);
    run_kernel_case("nk_euclidean_i8_icelake", test_euclidean<i8_t>, icelake, nk_euclidean_i8_icelake);
    run_kernel_case("nk_angular_u8_icelake", test_angular<u8_t>, icelake, nk_angular_u8_icelake);
    run_kernel_case("nk_sqeuclidean_u8_icelake", test_sqeuclidean<u8_t>, icelake, nk_sqeuclidean_u8_icelake);
    run_kernel_case("nk_euclidean_u8_icelake", test_euclidean<u8_t>, icelake, nk_euclidean_u8_icelake);
    run_kernel_case("nk_angular_i4_icelake", test_angular<i4x2_t>, icelake, nk_angular_i4_icelake);
    run_kernel_case("nk_sqeuclidean_i4_icelake", test_sqeuclidean<i4x2_t>, icelake, nk_sqeuclidean_i4_icelake);
    run_kernel_case("nk_euclidean_i4_icelake", test_euclidean<i4x2_t>, icelake, nk_euclidean_i4_icelake);
    run_kernel_case("nk_angular_u4_icelake", test_angular<u4x2_t>, icelake, nk_angular_u4_icelake);
    run_kernel_case("nk_sqeuclidean_u4_icelake", test_sqeuclidean<u4x2_t>, icelake, nk_sqeuclidean_u4_icelake);
    run_kernel_case("nk_euclidean_u4_icelake", test_euclidean<u4x2_t>, icelake, nk_euclidean_u4_icelake);
    run_kernel_case("nk_angular_e4m3_icelake", test_angular<e4m3_t>, icelake, nk_angular_e4m3_icelake);
    run_kernel_case("nk_sqeuclidean_e4m3_icelake", test_sqeuclidean<e4m3_t>, icelake, nk_sqeuclidean_e4m3_icelake);
    run_kernel_case("nk_euclidean_e4m3_icelake", test_euclidean<e4m3_t>, icelake, nk_euclidean_e4m3_icelake);
    run_kernel_case("nk_angular_e2m3_icelake", test_angular<e2m3_t>, icelake, nk_angular_e2m3_icelake);
    run_kernel_case("nk_sqeuclidean_e2m3_icelake", test_sqeuclidean<e2m3_t>, icelake, nk_sqeuclidean_e2m3_icelake);
    run_kernel_case("nk_euclidean_e2m3_icelake", test_euclidean<e2m3_t>, icelake, nk_euclidean_e2m3_icelake);
    run_kernel_case("nk_angular_e3m2_icelake", test_angular<e3m2_t>, icelake, nk_angular_e3m2_icelake);
    run_kernel_case("nk_sqeuclidean_e3m2_icelake", test_sqeuclidean<e3m2_t>, icelake, nk_sqeuclidean_e3m2_icelake);
    run_kernel_case("nk_euclidean_e3m2_icelake", test_euclidean<e3m2_t>, icelake, nk_euclidean_e3m2_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
    run_kernel_case("nk_angular_bf16_genoa", test_angular<bf16_t>, nk_cap_genoa_k, nk_angular_bf16_genoa);
    run_kernel_case("nk_sqeuclidean_bf16_genoa", test_sqeuclidean<bf16_t>, nk_cap_genoa_k, nk_sqeuclidean_bf16_genoa);
    run_kernel_case("nk_euclidean_bf16_genoa", test_euclidean<bf16_t>, nk_cap_genoa_k, nk_euclidean_bf16_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_DIAMOND
    auto const diamond = nk_cap_diamond_k;
    run_kernel_case("nk_angular_f16_diamond", test_angular<f16_t>, diamond, nk_angular_f16_diamond);
    run_kernel_case("nk_sqeuclidean_f16_diamond", test_sqeuclidean<f16_t>, diamond, nk_sqeuclidean_f16_diamond);
    run_kernel_case("nk_euclidean_f16_diamond", test_euclidean<f16_t>, diamond, nk_euclidean_f16_diamond);
    run_kernel_case("nk_angular_e4m3_diamond", test_angular<e4m3_t>, diamond, nk_angular_e4m3_diamond);
    run_kernel_case("nk_sqeuclidean_e4m3_diamond", test_sqeuclidean<e4m3_t>, diamond, nk_sqeuclidean_e4m3_diamond);
    run_kernel_case("nk_euclidean_e4m3_diamond", test_euclidean<e4m3_t>, diamond, nk_euclidean_e4m3_diamond);
    run_kernel_case("nk_angular_e5m2_diamond", test_angular<e5m2_t>, diamond, nk_angular_e5m2_diamond);
    run_kernel_case("nk_sqeuclidean_e5m2_diamond", test_sqeuclidean<e5m2_t>, diamond, nk_sqeuclidean_e5m2_diamond);
    run_kernel_case("nk_euclidean_e5m2_diamond", test_euclidean<e5m2_t>, diamond, nk_euclidean_e5m2_diamond);
#endif // NUMKONG_TARGET_DIAMOND

#if NUMKONG_TARGET_ALDER
    run_kernel_case("nk_angular_e3m2_alder", test_angular<e3m2_t>, nk_cap_alder_k, nk_angular_e3m2_alder);
    run_kernel_case("nk_sqeuclidean_e3m2_alder", test_sqeuclidean<e3m2_t>, nk_cap_alder_k, nk_sqeuclidean_e3m2_alder);
    run_kernel_case("nk_euclidean_e3m2_alder", test_euclidean<e3m2_t>, nk_cap_alder_k, nk_euclidean_e3m2_alder);
    run_kernel_case("nk_angular_e2m3_alder", test_angular<e2m3_t>, nk_cap_alder_k, nk_angular_e2m3_alder);
    run_kernel_case("nk_sqeuclidean_e2m3_alder", test_sqeuclidean<e2m3_t>, nk_cap_alder_k, nk_sqeuclidean_e2m3_alder);
    run_kernel_case("nk_euclidean_e2m3_alder", test_euclidean<e2m3_t>, nk_cap_alder_k, nk_euclidean_e2m3_alder);
    run_kernel_case("nk_angular_i8_alder", test_angular<i8_t>, nk_cap_alder_k, nk_angular_i8_alder);
    run_kernel_case("nk_sqeuclidean_i8_alder", test_sqeuclidean<i8_t>, nk_cap_alder_k, nk_sqeuclidean_i8_alder);
    run_kernel_case("nk_euclidean_i8_alder", test_euclidean<i8_t>, nk_cap_alder_k, nk_euclidean_i8_alder);
    run_kernel_case("nk_angular_u8_alder", test_angular<u8_t>, nk_cap_alder_k, nk_angular_u8_alder);
    run_kernel_case("nk_sqeuclidean_u8_alder", test_sqeuclidean<u8_t>, nk_cap_alder_k, nk_sqeuclidean_u8_alder);
    run_kernel_case("nk_euclidean_u8_alder", test_euclidean<u8_t>, nk_cap_alder_k, nk_euclidean_u8_alder);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_SIERRA
    auto const sierra = nk_cap_sierra_k;
    run_kernel_case("nk_angular_e2m3_sierra", test_angular<e2m3_t>, sierra, nk_angular_e2m3_sierra);
    run_kernel_case("nk_sqeuclidean_e2m3_sierra", test_sqeuclidean<e2m3_t>, sierra, nk_sqeuclidean_e2m3_sierra);
    run_kernel_case("nk_euclidean_e2m3_sierra", test_euclidean<e2m3_t>, sierra, nk_euclidean_e2m3_sierra);
    run_kernel_case("nk_angular_e3m2_sierra", test_angular<e3m2_t>, sierra, nk_angular_e3m2_sierra);
    run_kernel_case("nk_sqeuclidean_e3m2_sierra", test_sqeuclidean<e3m2_t>, sierra, nk_sqeuclidean_e3m2_sierra);
    run_kernel_case("nk_euclidean_e3m2_sierra", test_euclidean<e3m2_t>, sierra, nk_euclidean_e3m2_sierra);
    run_kernel_case("nk_angular_i8_sierra", test_angular<i8_t>, sierra, nk_angular_i8_sierra);
    run_kernel_case("nk_sqeuclidean_i8_sierra", test_sqeuclidean<i8_t>, sierra, nk_sqeuclidean_i8_sierra);
    run_kernel_case("nk_euclidean_i8_sierra", test_euclidean<i8_t>, sierra, nk_euclidean_i8_sierra);
    run_kernel_case("nk_angular_u8_sierra", test_angular<u8_t>, sierra, nk_angular_u8_sierra);
    run_kernel_case("nk_sqeuclidean_u8_sierra", test_sqeuclidean<u8_t>, sierra, nk_sqeuclidean_u8_sierra);
    run_kernel_case("nk_euclidean_u8_sierra", test_euclidean<u8_t>, sierra, nk_euclidean_u8_sierra);
#endif // NUMKONG_TARGET_SIERRA

#if NUMKONG_TARGET_RVV
    run_kernel_case("nk_angular_f64_rvv", test_angular<f64_t>, nk_cap_rvv_k, nk_angular_f64_rvv);
    run_kernel_case("nk_sqeuclidean_f64_rvv", test_sqeuclidean<f64_t>, nk_cap_rvv_k, nk_sqeuclidean_f64_rvv);
    run_kernel_case("nk_euclidean_f64_rvv", test_euclidean<f64_t>, nk_cap_rvv_k, nk_euclidean_f64_rvv);
    run_kernel_case("nk_angular_f32_rvv", test_angular<f32_t>, nk_cap_rvv_k, nk_angular_f32_rvv);
    run_kernel_case("nk_sqeuclidean_f32_rvv", test_sqeuclidean<f32_t>, nk_cap_rvv_k, nk_sqeuclidean_f32_rvv);
    run_kernel_case("nk_euclidean_f32_rvv", test_euclidean<f32_t>, nk_cap_rvv_k, nk_euclidean_f32_rvv);
    run_kernel_case("nk_angular_bf16_rvv", test_angular<bf16_t>, nk_cap_rvv_k, nk_angular_bf16_rvv);
    run_kernel_case("nk_sqeuclidean_bf16_rvv", test_sqeuclidean<bf16_t>, nk_cap_rvv_k, nk_sqeuclidean_bf16_rvv);
    run_kernel_case("nk_euclidean_bf16_rvv", test_euclidean<bf16_t>, nk_cap_rvv_k, nk_euclidean_bf16_rvv);
    run_kernel_case("nk_angular_f16_rvv", test_angular<f16_t>, nk_cap_rvv_k, nk_angular_f16_rvv);
    run_kernel_case("nk_sqeuclidean_f16_rvv", test_sqeuclidean<f16_t>, nk_cap_rvv_k, nk_sqeuclidean_f16_rvv);
    run_kernel_case("nk_euclidean_f16_rvv", test_euclidean<f16_t>, nk_cap_rvv_k, nk_euclidean_f16_rvv);
    run_kernel_case("nk_angular_e5m2_rvv", test_angular<e5m2_t>, nk_cap_rvv_k, nk_angular_e5m2_rvv);
    run_kernel_case("nk_sqeuclidean_e5m2_rvv", test_sqeuclidean<e5m2_t>, nk_cap_rvv_k, nk_sqeuclidean_e5m2_rvv);
    run_kernel_case("nk_euclidean_e5m2_rvv", test_euclidean<e5m2_t>, nk_cap_rvv_k, nk_euclidean_e5m2_rvv);
    run_kernel_case("nk_angular_e4m3_rvv", test_angular<e4m3_t>, nk_cap_rvv_k, nk_angular_e4m3_rvv);
    run_kernel_case("nk_sqeuclidean_e4m3_rvv", test_sqeuclidean<e4m3_t>, nk_cap_rvv_k, nk_sqeuclidean_e4m3_rvv);
    run_kernel_case("nk_euclidean_e4m3_rvv", test_euclidean<e4m3_t>, nk_cap_rvv_k, nk_euclidean_e4m3_rvv);
    run_kernel_case("nk_angular_i8_rvv", test_angular<i8_t>, nk_cap_rvv_k, nk_angular_i8_rvv);
    run_kernel_case("nk_sqeuclidean_i8_rvv", test_sqeuclidean<i8_t>, nk_cap_rvv_k, nk_sqeuclidean_i8_rvv);
    run_kernel_case("nk_euclidean_i8_rvv", test_euclidean<i8_t>, nk_cap_rvv_k, nk_euclidean_i8_rvv);
    run_kernel_case("nk_angular_u8_rvv", test_angular<u8_t>, nk_cap_rvv_k, nk_angular_u8_rvv);
    run_kernel_case("nk_sqeuclidean_u8_rvv", test_sqeuclidean<u8_t>, nk_cap_rvv_k, nk_sqeuclidean_u8_rvv);
    run_kernel_case("nk_euclidean_u8_rvv", test_euclidean<u8_t>, nk_cap_rvv_k, nk_euclidean_u8_rvv);
    run_kernel_case("nk_angular_i4_rvv", test_angular<i4x2_t>, nk_cap_rvv_k, nk_angular_i4_rvv);
    run_kernel_case("nk_sqeuclidean_i4_rvv", test_sqeuclidean<i4x2_t>, nk_cap_rvv_k, nk_sqeuclidean_i4_rvv);
    run_kernel_case("nk_euclidean_i4_rvv", test_euclidean<i4x2_t>, nk_cap_rvv_k, nk_euclidean_i4_rvv);
    run_kernel_case("nk_angular_u4_rvv", test_angular<u4x2_t>, nk_cap_rvv_k, nk_angular_u4_rvv);
    run_kernel_case("nk_sqeuclidean_u4_rvv", test_sqeuclidean<u4x2_t>, nk_cap_rvv_k, nk_sqeuclidean_u4_rvv);
    run_kernel_case("nk_euclidean_u4_rvv", test_euclidean<u4x2_t>, nk_cap_rvv_k, nk_euclidean_u4_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
    auto const v128relaxed = nk_cap_v128relaxed_k;
    run_kernel_case("nk_sqeuclidean_f32_v128relaxed", test_sqeuclidean<f32_t>, v128relaxed,
                    nk_sqeuclidean_f32_v128relaxed);
    run_kernel_case("nk_sqeuclidean_f64_v128relaxed", test_sqeuclidean<f64_t>, v128relaxed,
                    nk_sqeuclidean_f64_v128relaxed);
    run_kernel_case("nk_sqeuclidean_f16_v128relaxed", test_sqeuclidean<f16_t>, v128relaxed,
                    nk_sqeuclidean_f16_v128relaxed);
    run_kernel_case("nk_sqeuclidean_bf16_v128relaxed", test_sqeuclidean<bf16_t>, v128relaxed,
                    nk_sqeuclidean_bf16_v128relaxed);
    run_kernel_case("nk_euclidean_f32_v128relaxed", test_euclidean<f32_t>, v128relaxed, nk_euclidean_f32_v128relaxed);
    run_kernel_case("nk_euclidean_f64_v128relaxed", test_euclidean<f64_t>, v128relaxed, nk_euclidean_f64_v128relaxed);
    run_kernel_case("nk_euclidean_f16_v128relaxed", test_euclidean<f16_t>, v128relaxed, nk_euclidean_f16_v128relaxed);
    run_kernel_case("nk_euclidean_bf16_v128relaxed", test_euclidean<bf16_t>, v128relaxed,
                    nk_euclidean_bf16_v128relaxed);
    run_kernel_case("nk_angular_f32_v128relaxed", test_angular<f32_t>, v128relaxed, nk_angular_f32_v128relaxed);
    run_kernel_case("nk_angular_f64_v128relaxed", test_angular<f64_t>, v128relaxed, nk_angular_f64_v128relaxed);
    run_kernel_case("nk_angular_f16_v128relaxed", test_angular<f16_t>, v128relaxed, nk_angular_f16_v128relaxed);
    run_kernel_case("nk_angular_bf16_v128relaxed", test_angular<bf16_t>, v128relaxed, nk_angular_bf16_v128relaxed);
    run_kernel_case("nk_sqeuclidean_u8_v128relaxed", test_sqeuclidean<u8_t>, v128relaxed,
                    nk_sqeuclidean_u8_v128relaxed);
    run_kernel_case("nk_euclidean_u8_v128relaxed", test_euclidean<u8_t>, v128relaxed, nk_euclidean_u8_v128relaxed);
    run_kernel_case("nk_angular_u8_v128relaxed", test_angular<u8_t>, v128relaxed, nk_angular_u8_v128relaxed);
    run_kernel_case("nk_sqeuclidean_i8_v128relaxed", test_sqeuclidean<i8_t>, v128relaxed,
                    nk_sqeuclidean_i8_v128relaxed);
    run_kernel_case("nk_euclidean_i8_v128relaxed", test_euclidean<i8_t>, v128relaxed, nk_euclidean_i8_v128relaxed);
    run_kernel_case("nk_angular_i8_v128relaxed", test_angular<i8_t>, v128relaxed, nk_angular_i8_v128relaxed);
    run_kernel_case("nk_sqeuclidean_e4m3_v128relaxed", test_sqeuclidean<e4m3_t>, v128relaxed,
                    nk_sqeuclidean_e4m3_v128relaxed);
    run_kernel_case("nk_euclidean_e4m3_v128relaxed", test_euclidean<e4m3_t>, v128relaxed,
                    nk_euclidean_e4m3_v128relaxed);
    run_kernel_case("nk_angular_e4m3_v128relaxed", test_angular<e4m3_t>, v128relaxed, nk_angular_e4m3_v128relaxed);
    run_kernel_case("nk_sqeuclidean_e5m2_v128relaxed", test_sqeuclidean<e5m2_t>, v128relaxed,
                    nk_sqeuclidean_e5m2_v128relaxed);
    run_kernel_case("nk_euclidean_e5m2_v128relaxed", test_euclidean<e5m2_t>, v128relaxed,
                    nk_euclidean_e5m2_v128relaxed);
    run_kernel_case("nk_angular_e5m2_v128relaxed", test_angular<e5m2_t>, v128relaxed, nk_angular_e5m2_v128relaxed);
    run_kernel_case("nk_sqeuclidean_e2m3_v128relaxed", test_sqeuclidean<e2m3_t>, v128relaxed,
                    nk_sqeuclidean_e2m3_v128relaxed);
    run_kernel_case("nk_euclidean_e2m3_v128relaxed", test_euclidean<e2m3_t>, v128relaxed,
                    nk_euclidean_e2m3_v128relaxed);
    run_kernel_case("nk_angular_e2m3_v128relaxed", test_angular<e2m3_t>, v128relaxed, nk_angular_e2m3_v128relaxed);
    run_kernel_case("nk_sqeuclidean_e3m2_v128relaxed", test_sqeuclidean<e3m2_t>, v128relaxed,
                    nk_sqeuclidean_e3m2_v128relaxed);
    run_kernel_case("nk_euclidean_e3m2_v128relaxed", test_euclidean<e3m2_t>, v128relaxed,
                    nk_euclidean_e3m2_v128relaxed);
    run_kernel_case("nk_angular_e3m2_v128relaxed", test_angular<e3m2_t>, v128relaxed, nk_angular_e3m2_v128relaxed);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_V128
    run_kernel_case("nk_sqeuclidean_bf16_v128", test_sqeuclidean<bf16_t>, nk_cap_v128_k, nk_sqeuclidean_bf16_v128);
    run_kernel_case("nk_euclidean_bf16_v128", test_euclidean<bf16_t>, nk_cap_v128_k, nk_euclidean_bf16_v128);
    run_kernel_case("nk_angular_bf16_v128", test_angular<bf16_t>, nk_cap_v128_k, nk_angular_bf16_v128);
    run_kernel_case("nk_sqeuclidean_u8_v128", test_sqeuclidean<u8_t>, nk_cap_v128_k, nk_sqeuclidean_u8_v128);
    run_kernel_case("nk_euclidean_u8_v128", test_euclidean<u8_t>, nk_cap_v128_k, nk_euclidean_u8_v128);
    run_kernel_case("nk_angular_u8_v128", test_angular<u8_t>, nk_cap_v128_k, nk_angular_u8_v128);
    run_kernel_case("nk_sqeuclidean_i8_v128", test_sqeuclidean<i8_t>, nk_cap_v128_k, nk_sqeuclidean_i8_v128);
    run_kernel_case("nk_euclidean_i8_v128", test_euclidean<i8_t>, nk_cap_v128_k, nk_euclidean_i8_v128);
    run_kernel_case("nk_angular_i8_v128", test_angular<i8_t>, nk_cap_v128_k, nk_angular_i8_v128);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_RVVHALF
    auto const rvvhalf = nk_cap_rvvhalf_k;
    run_kernel_case("nk_sqeuclidean_f16_rvvhalf", test_sqeuclidean<f16_t>, rvvhalf, nk_sqeuclidean_f16_rvvhalf);
    run_kernel_case("nk_euclidean_f16_rvvhalf", test_euclidean<f16_t>, rvvhalf, nk_euclidean_f16_rvvhalf);
    run_kernel_case("nk_angular_f16_rvvhalf", test_angular<f16_t>, rvvhalf, nk_angular_f16_rvvhalf);
#endif // NUMKONG_TARGET_RVVHALF

#if NUMKONG_TARGET_RVVBF16
    auto const rvvbf16 = nk_cap_rvvbf16_k;
    run_kernel_case("nk_sqeuclidean_bf16_rvvbf16", test_sqeuclidean<bf16_t>, rvvbf16, nk_sqeuclidean_bf16_rvvbf16);
    run_kernel_case("nk_euclidean_bf16_rvvbf16", test_euclidean<bf16_t>, rvvbf16, nk_euclidean_bf16_rvvbf16);
    run_kernel_case("nk_angular_bf16_rvvbf16", test_angular<bf16_t>, rvvbf16, nk_angular_bf16_rvvbf16);
#endif // NUMKONG_TARGET_RVVBF16

#if NUMKONG_TARGET_LOONGSONASX
    auto const loongsonasx = nk_cap_loongsonasx_k;
    run_kernel_case("nk_angular_f64_loongsonasx", test_angular<f64_t>, loongsonasx, nk_angular_f64_loongsonasx);
    run_kernel_case("nk_sqeuclidean_f64_loongsonasx", test_sqeuclidean<f64_t>, loongsonasx,
                    nk_sqeuclidean_f64_loongsonasx);
    run_kernel_case("nk_euclidean_f64_loongsonasx", test_euclidean<f64_t>, loongsonasx, nk_euclidean_f64_loongsonasx);
    run_kernel_case("nk_angular_f32_loongsonasx", test_angular<f32_t>, loongsonasx, nk_angular_f32_loongsonasx);
    run_kernel_case("nk_sqeuclidean_f32_loongsonasx", test_sqeuclidean<f32_t>, loongsonasx,
                    nk_sqeuclidean_f32_loongsonasx);
    run_kernel_case("nk_euclidean_f32_loongsonasx", test_euclidean<f32_t>, loongsonasx, nk_euclidean_f32_loongsonasx);
    run_kernel_case("nk_angular_bf16_loongsonasx", test_angular<bf16_t>, loongsonasx, nk_angular_bf16_loongsonasx);
    run_kernel_case("nk_sqeuclidean_bf16_loongsonasx", test_sqeuclidean<bf16_t>, loongsonasx,
                    nk_sqeuclidean_bf16_loongsonasx);
    run_kernel_case("nk_euclidean_bf16_loongsonasx", test_euclidean<bf16_t>, loongsonasx,
                    nk_euclidean_bf16_loongsonasx);
    run_kernel_case("nk_angular_i8_loongsonasx", test_angular<i8_t>, loongsonasx, nk_angular_i8_loongsonasx);
    run_kernel_case("nk_sqeuclidean_i8_loongsonasx", test_sqeuclidean<i8_t>, loongsonasx,
                    nk_sqeuclidean_i8_loongsonasx);
    run_kernel_case("nk_euclidean_i8_loongsonasx", test_euclidean<i8_t>, loongsonasx, nk_euclidean_i8_loongsonasx);
    run_kernel_case("nk_angular_u8_loongsonasx", test_angular<u8_t>, loongsonasx, nk_angular_u8_loongsonasx);
    run_kernel_case("nk_sqeuclidean_u8_loongsonasx", test_sqeuclidean<u8_t>, loongsonasx,
                    nk_sqeuclidean_u8_loongsonasx);
    run_kernel_case("nk_euclidean_u8_loongsonasx", test_euclidean<u8_t>, loongsonasx, nk_euclidean_u8_loongsonasx);
#endif // NUMKONG_TARGET_LOONGSONASX

#if NUMKONG_TARGET_POWERVSX
    auto const powervsx = nk_cap_powervsx_k;
    run_kernel_case("nk_sqeuclidean_f32_powervsx", test_sqeuclidean<f32_t>, powervsx, nk_sqeuclidean_f32_powervsx);
    run_kernel_case("nk_euclidean_f32_powervsx", test_euclidean<f32_t>, powervsx, nk_euclidean_f32_powervsx);
    run_kernel_case("nk_angular_f32_powervsx", test_angular<f32_t>, powervsx, nk_angular_f32_powervsx);
    run_kernel_case("nk_sqeuclidean_f64_powervsx", test_sqeuclidean<f64_t>, powervsx, nk_sqeuclidean_f64_powervsx);
    run_kernel_case("nk_euclidean_f64_powervsx", test_euclidean<f64_t>, powervsx, nk_euclidean_f64_powervsx);
    run_kernel_case("nk_angular_f64_powervsx", test_angular<f64_t>, powervsx, nk_angular_f64_powervsx);
    run_kernel_case("nk_sqeuclidean_f16_powervsx", test_sqeuclidean<f16_t>, powervsx, nk_sqeuclidean_f16_powervsx);
    run_kernel_case("nk_euclidean_f16_powervsx", test_euclidean<f16_t>, powervsx, nk_euclidean_f16_powervsx);
    run_kernel_case("nk_angular_f16_powervsx", test_angular<f16_t>, powervsx, nk_angular_f16_powervsx);
    run_kernel_case("nk_sqeuclidean_bf16_powervsx", test_sqeuclidean<bf16_t>, powervsx, nk_sqeuclidean_bf16_powervsx);
    run_kernel_case("nk_euclidean_bf16_powervsx", test_euclidean<bf16_t>, powervsx, nk_euclidean_bf16_powervsx);
    run_kernel_case("nk_angular_bf16_powervsx", test_angular<bf16_t>, powervsx, nk_angular_bf16_powervsx);
    run_kernel_case("nk_sqeuclidean_i8_powervsx", test_sqeuclidean<i8_t>, powervsx, nk_sqeuclidean_i8_powervsx);
    run_kernel_case("nk_euclidean_i8_powervsx", test_euclidean<i8_t>, powervsx, nk_euclidean_i8_powervsx);
    run_kernel_case("nk_angular_i8_powervsx", test_angular<i8_t>, powervsx, nk_angular_i8_powervsx);
    run_kernel_case("nk_sqeuclidean_u8_powervsx", test_sqeuclidean<u8_t>, powervsx, nk_sqeuclidean_u8_powervsx);
    run_kernel_case("nk_euclidean_u8_powervsx", test_euclidean<u8_t>, powervsx, nk_euclidean_u8_powervsx);
    run_kernel_case("nk_angular_u8_powervsx", test_angular<u8_t>, powervsx, nk_angular_u8_powervsx);
#endif // NUMKONG_TARGET_POWERVSX
}

} // namespace ashvardanian::numkong::test
