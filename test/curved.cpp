/**
 *  @file test/curved.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief Bilinear and Mahalanobis tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "numkong/curved.hpp" // `nk::bilinear`

namespace ashvardanian::numkong::test {

/**
 *  @brief Makes a square matrix positive semi-definite via symmetrization and diagonal dominance.
 *
 *  Uses Gershgorin's circle theorem, so that (a − b)ᵀ × M × (a − b) ≥ 0 keeps the Mahalanobis
 *  square root from NaN: a symmetric matrix whose diagonal entries each exceed the absolute row sum
 *  of their off-diagonal entries is positive definite.
 */
template <typename scalar_type_>
void make_psd(scalar_type_ *data, nk_size_t n) {
    auto at = [=](nk_size_t row, nk_size_t column) { return static_cast<double>(data[row * n + column]); };
    // Symmetrize — m[i][j] = m[j][i] = average of original pair
    for (nk_size_t i = 0; i < n; ++i)
        for (nk_size_t j = i + 1; j < n; ++j)
            data[i * n + j] = data[j * n + i] = scalar_type_((at(i, j) + at(j, i)) / 2);
    // Strict diagonal dominance — set m[i][i] > sum_{j!=i} |m[i][j]|
    for (nk_size_t i = 0; i < n; ++i) {
        double row_sum = 0;
        for (nk_size_t j = 0; j < n; ++j)
            if (i != j) row_sum += std::fabs(at(i, j));
        data[i * n + i] = scalar_type_(row_sum + 1);
    }
}

/** Template for bilinear form test: a^T * M * b. */
template <typename scalar_type_>
error_stats_t test_bilinear(typename scalar_type_::curved_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::curved_result_t;
    using reference_t = tracked<reference_for<scalar_t>>;

    error_stats_t stats(nk_bilinear_error_bound(scalar_t::dtype()));
    split_mix_generator_t generator(case_stream_key());

    auto a = make_vector<scalar_t>(test_run.settings.curved_dimensions),
         b = make_vector<scalar_t>(test_run.settings.curved_dimensions);
    auto m = make_vector<scalar_t>(test_run.settings.curved_dimensions * test_run.settings.curved_dimensions);
    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, m);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), m.raw_values_data(),
                                   test_run.settings.curved_dimensions, &result.raw_, nullptr));

        reference_t reference;
        stats.expect_status(nk::bilinear<scalar_t, reference_t>(a.values_data(), b.values_data(), m.values_data(),
                                                                test_run.settings.curved_dimensions, &reference,
                                                                no_tiers_k));

        stats.accumulate(result, reference);
    }

    return stats;
}

/** Template for Mahalanobis distance test: sqrt((a-b)^T * M * (a-b)). */
template <typename scalar_type_>
error_stats_t test_mahalanobis(typename scalar_type_::curved_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::curved_result_t;
    using reference_t = tracked<reference_for<scalar_t>>;

    error_stats_t stats(nk_mahalanobis_error_bound(scalar_t::dtype()));
    split_mix_generator_t generator(case_stream_key());

    auto a = make_vector<scalar_t>(test_run.settings.curved_dimensions),
         b = make_vector<scalar_t>(test_run.settings.curved_dimensions);
    auto m = make_vector<scalar_t>(test_run.settings.curved_dimensions * test_run.settings.curved_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, m);
        make_psd(m.values_data(), test_run.settings.curved_dimensions);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), m.raw_values_data(),
                                   test_run.settings.curved_dimensions, &result.raw_, nullptr));

        reference_t reference;
        stats.expect_status(nk::mahalanobis<scalar_t, reference_t>(a.values_data(), b.values_data(), m.values_data(),
                                                                   test_run.settings.curved_dimensions, &reference,
                                                                   no_tiers_k));

        stats.accumulate(result, reference);
    }
    return stats;
}

void run_curved_cases() {
    auto const serial = nk_cap_serial_k;

    run_kernel_case("nk_bilinear_f32_serial", test_bilinear<f32_t>, serial, nk_bilinear_f32_serial);
    run_kernel_case("nk_bilinear_f64_serial", test_bilinear<f64_t>, serial, nk_bilinear_f64_serial);
    run_kernel_case("nk_bilinear_f32c_serial", test_bilinear<f32c_t>, serial, nk_bilinear_f32c_serial);
    run_kernel_case("nk_bilinear_f64c_serial", test_bilinear<f64c_t>, serial, nk_bilinear_f64c_serial);
    run_kernel_case("nk_mahalanobis_f32_serial", test_mahalanobis<f32_t>, serial, nk_mahalanobis_f32_serial);
    run_kernel_case("nk_mahalanobis_f64_serial", test_mahalanobis<f64_t>, serial, nk_mahalanobis_f64_serial);
    run_kernel_case("nk_bilinear_f16_serial", test_bilinear<f16_t>, serial, nk_bilinear_f16_serial);
    run_kernel_case("nk_bilinear_f16c_serial", test_bilinear<f16c_t>, serial, nk_bilinear_f16c_serial);
    run_kernel_case("nk_mahalanobis_f16_serial", test_mahalanobis<f16_t>, serial, nk_mahalanobis_f16_serial);
    run_kernel_case("nk_bilinear_bf16_serial", test_bilinear<bf16_t>, serial, nk_bilinear_bf16_serial);
    run_kernel_case("nk_bilinear_bf16c_serial", test_bilinear<bf16c_t>, serial, nk_bilinear_bf16c_serial);
    run_kernel_case("nk_mahalanobis_bf16_serial", test_mahalanobis<bf16_t>, serial, nk_mahalanobis_bf16_serial);

#if !NUMKONG_HEADER_ONLY
    run_kernel_case("nk_bilinear_f32_best", test_bilinear<f32_t>, serial, cpu_best<nk_bilinear_f32_best>);
    run_kernel_case("nk_bilinear_f64_best", test_bilinear<f64_t>, serial, cpu_best<nk_bilinear_f64_best>);
    run_kernel_case("nk_bilinear_f32c_best", test_bilinear<f32c_t>, serial, cpu_best<nk_bilinear_f32c_best>);
    run_kernel_case("nk_bilinear_f64c_best", test_bilinear<f64c_t>, serial, cpu_best<nk_bilinear_f64c_best>);
    run_kernel_case("nk_mahalanobis_f32_best", test_mahalanobis<f32_t>, serial, cpu_best<nk_mahalanobis_f32_best>);
    run_kernel_case("nk_mahalanobis_f64_best", test_mahalanobis<f64_t>, serial, cpu_best<nk_mahalanobis_f64_best>);
#endif

#if NUMKONG_TARGET_NEON
    run_kernel_case("nk_bilinear_f64_neon", test_bilinear<f64_t>, nk_cap_neon_k, nk_bilinear_f64_neon);
    run_kernel_case("nk_mahalanobis_f64_neon", test_mahalanobis<f64_t>, nk_cap_neon_k, nk_mahalanobis_f64_neon);
    run_kernel_case("nk_bilinear_f32_neon", test_bilinear<f32_t>, nk_cap_neon_k, nk_bilinear_f32_neon);
    run_kernel_case("nk_bilinear_f32c_neon", test_bilinear<f32c_t>, nk_cap_neon_k, nk_bilinear_f32c_neon);
    run_kernel_case("nk_mahalanobis_f32_neon", test_mahalanobis<f32_t>, nk_cap_neon_k, nk_mahalanobis_f32_neon);
    run_kernel_case("nk_bilinear_f16_neon", test_bilinear<f16_t>, nk_cap_neon_k, nk_bilinear_f16_neon);
    run_kernel_case("nk_bilinear_f16c_neon", test_bilinear<f16c_t>, nk_cap_neon_k, nk_bilinear_f16c_neon);
    run_kernel_case("nk_mahalanobis_f16_neon", test_mahalanobis<f16_t>, nk_cap_neon_k, nk_mahalanobis_f16_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
    auto const neonbfdot = nk_cap_neonbfdot_k;
    run_kernel_case("nk_bilinear_bf16_neonbfdot", test_bilinear<bf16_t>, neonbfdot, nk_bilinear_bf16_neonbfdot);
    run_kernel_case("nk_bilinear_bf16c_neonbfdot", test_bilinear<bf16c_t>, neonbfdot, nk_bilinear_bf16c_neonbfdot);
    run_kernel_case("nk_mahalanobis_bf16_neonbfdot", test_mahalanobis<bf16_t>, neonbfdot,
                    nk_mahalanobis_bf16_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_HASWELL
    auto const haswell = nk_cap_haswell_k;
    run_kernel_case("nk_bilinear_f64_haswell", test_bilinear<f64_t>, haswell, nk_bilinear_f64_haswell);
    run_kernel_case("nk_mahalanobis_f64_haswell", test_mahalanobis<f64_t>, haswell, nk_mahalanobis_f64_haswell);
    run_kernel_case("nk_bilinear_f32_haswell", test_bilinear<f32_t>, haswell, nk_bilinear_f32_haswell);
    run_kernel_case("nk_bilinear_f16_haswell", test_bilinear<f16_t>, haswell, nk_bilinear_f16_haswell);
    run_kernel_case("nk_bilinear_bf16_haswell", test_bilinear<bf16_t>, haswell, nk_bilinear_bf16_haswell);
    run_kernel_case("nk_mahalanobis_f32_haswell", test_mahalanobis<f32_t>, haswell, nk_mahalanobis_f32_haswell);
    run_kernel_case("nk_mahalanobis_f16_haswell", test_mahalanobis<f16_t>, haswell, nk_mahalanobis_f16_haswell);
    run_kernel_case("nk_mahalanobis_bf16_haswell", test_mahalanobis<bf16_t>, haswell, nk_mahalanobis_bf16_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    auto const skylake = nk_cap_skylake_k;
    run_kernel_case("nk_bilinear_f32_skylake", test_bilinear<f32_t>, skylake, nk_bilinear_f32_skylake);
    run_kernel_case("nk_bilinear_f64_skylake", test_bilinear<f64_t>, skylake, nk_bilinear_f64_skylake);
    run_kernel_case("nk_bilinear_f32c_skylake", test_bilinear<f32c_t>, skylake, nk_bilinear_f32c_skylake);
    run_kernel_case("nk_bilinear_f64c_skylake", test_bilinear<f64c_t>, skylake, nk_bilinear_f64c_skylake);
    run_kernel_case("nk_mahalanobis_f32_skylake", test_mahalanobis<f32_t>, skylake, nk_mahalanobis_f32_skylake);
    run_kernel_case("nk_mahalanobis_f64_skylake", test_mahalanobis<f64_t>, skylake, nk_mahalanobis_f64_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_GENOA
    run_kernel_case("nk_bilinear_bf16_genoa", test_bilinear<bf16_t>, nk_cap_genoa_k, nk_bilinear_bf16_genoa);
    run_kernel_case("nk_bilinear_bf16c_genoa", test_bilinear<bf16c_t>, nk_cap_genoa_k, nk_bilinear_bf16c_genoa);
    run_kernel_case("nk_mahalanobis_bf16_genoa", test_mahalanobis<bf16_t>, nk_cap_genoa_k, nk_mahalanobis_bf16_genoa);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SMEF64
    run_kernel_case("nk_bilinear_f32_smef64", test_bilinear<f32_t>, nk_cap_smef64_k, nk_bilinear_f32_smef64);
    run_kernel_case("nk_bilinear_f32c_smef64", test_bilinear<f32c_t>, nk_cap_smef64_k, nk_bilinear_f32c_smef64);
    run_kernel_case("nk_mahalanobis_f32_smef64", test_mahalanobis<f32_t>, nk_cap_smef64_k, nk_mahalanobis_f32_smef64);
    run_kernel_case("nk_bilinear_f64_smef64", test_bilinear<f64_t>, nk_cap_smef64_k, nk_bilinear_f64_smef64);
    run_kernel_case("nk_bilinear_f64c_smef64", test_bilinear<f64c_t>, nk_cap_smef64_k, nk_bilinear_f64c_smef64);
    run_kernel_case("nk_mahalanobis_f64_smef64", test_mahalanobis<f64_t>, nk_cap_smef64_k, nk_mahalanobis_f64_smef64);
#endif // NUMKONG_TARGET_SMEF64
}

} // namespace ashvardanian::numkong::test
