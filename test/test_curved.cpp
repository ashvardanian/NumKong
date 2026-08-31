/**
 *  @brief Bilinear and Mahalanobis tests.
 *  @file test/test_curved.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/curved.hpp" // `nk::bilinear`

/**
 *  @brief Makes a square matrix positive semi-definite in-place via symmetrization + diagonal dominance.
 *
 *  Uses Gershgorin's circle theorem: a symmetric matrix with each diagonal entry exceeding
 *  the absolute row sum of its off-diagonal entries is positive definite. This ensures
 *  `(a-b)^T M (a-b) >= 0`, preventing NaN from sqrt in Mahalanobis distance.
 */
template <typename scalar_type_>
void make_psd(scalar_type_ *data, nk_size_t n) {
    // Step 1: Symmetrize — m[i][j] = m[j][i] = average of original pair
    for (nk_size_t i = 0; i < n; ++i)
        for (nk_size_t j = i + 1; j < n; ++j) {
            double avg = ((double)data[i * n + j] + (double)data[j * n + i]) * 0.5;
            data[i * n + j] = scalar_type_(avg);
            data[j * n + i] = scalar_type_(avg);
        }
    // Step 2: Strict diagonal dominance — set m[i][i] > sum_{j!=i} |m[i][j]|
    for (nk_size_t i = 0; i < n; ++i) {
        double row_sum = 0;
        for (nk_size_t j = 0; j < n; ++j)
            if (i != j) row_sum += std::abs((double)data[i * n + j]);
        data[i * n + i] = scalar_type_(row_sum + 1.0);
    }
}

/**
 *  @brief Template for bilinear form test: a^T * M * b
 */
template <typename scalar_type_>
error_stats_t test_bilinear(typename scalar_type_::curved_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::curved_result_t;
    using reference_t = reference_for<scalar_t, result_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto m = make_vector<scalar_t>(global_config.dense_dimensions * global_config.dense_dimensions);
    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, m);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), m.raw_values_data(), global_config.dense_dimensions,
               &result.raw_);

        reference_t reference;
        nk::bilinear<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), m.values_data(),
                                                           global_config.dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/**
 *  @brief Template for Mahalanobis distance test: sqrt((a-b)^T * M * (a-b))
 */
template <typename scalar_type_>
error_stats_t test_mahalanobis(typename scalar_type_::curved_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::curved_result_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::approximate_k);
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);
    auto m = make_vector<scalar_t>(global_config.dense_dimensions * global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);
        fill_random(generator, m);
        make_psd(m.values_data(), global_config.dense_dimensions);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), m.raw_values_data(), global_config.dense_dimensions,
               &result.raw_);

        reference_t reference;
        nk::mahalanobis<scalar_t, reference_t, nk::no_simd_k>(a.values_data(), b.values_data(), m.values_data(),
                                                              global_config.dense_dimensions, &reference);

        stats.accumulate(result, reference);
    }
    return stats;
}

void test_curved() {
    error_stats_section_t check;

    check.section("Curved Kernels Serial", nk_cap_serial_k);
    check("bilinear_f32_serial", test_bilinear<f32_t>, nk_bilinear_f32_serial);
    check("bilinear_f64_serial", test_bilinear<f64_t>, nk_bilinear_f64_serial);
    check("bilinear_f32c_serial", test_bilinear<f32c_t>, nk_bilinear_f32c_serial);
    check("bilinear_f64c_serial", test_bilinear<f64c_t>, nk_bilinear_f64c_serial);
    check("mahalanobis_f32_serial", test_mahalanobis<f32_t>, nk_mahalanobis_f32_serial);
    check("mahalanobis_f64_serial", test_mahalanobis<f64_t>, nk_mahalanobis_f64_serial);
    check("bilinear_f16_serial", test_bilinear<f16_t>, nk_bilinear_f16_serial);
    check("bilinear_f16c_serial", test_bilinear<f16c_t>, nk_bilinear_f16c_serial);
    check("mahalanobis_f16_serial", test_mahalanobis<f16_t>, nk_mahalanobis_f16_serial);
    check("bilinear_bf16_serial", test_bilinear<bf16_t>, nk_bilinear_bf16_serial);
    check("bilinear_bf16c_serial", test_bilinear<bf16c_t>, nk_bilinear_bf16c_serial);
    check("mahalanobis_bf16_serial", test_mahalanobis<bf16_t>, nk_mahalanobis_bf16_serial);

#if NK_DYNAMIC_DISPATCH
    check.section("Curved Kernels Dynamic", nk_cap_serial_k);
    check("bilinear_f32", test_bilinear<f32_t>, nk_bilinear_f32);
    check("bilinear_f64", test_bilinear<f64_t>, nk_bilinear_f64);
    check("bilinear_f32c", test_bilinear<f32c_t>, nk_bilinear_f32c);
    check("bilinear_f64c", test_bilinear<f64c_t>, nk_bilinear_f64c);
    check("mahalanobis_f32", test_mahalanobis<f32_t>, nk_mahalanobis_f32);
    check("mahalanobis_f64", test_mahalanobis<f64_t>, nk_mahalanobis_f64);
#endif

#if NK_TARGET_NEON
    check.section("Curved Kernels NEON", nk_cap_neon_k);
    check("bilinear_f32_neon", test_bilinear<f32_t>, nk_bilinear_f32_neon);
    check("bilinear_f32c_neon", test_bilinear<f32c_t>, nk_bilinear_f32c_neon);
    check("mahalanobis_f32_neon", test_mahalanobis<f32_t>, nk_mahalanobis_f32_neon);
    check("bilinear_f16_neon", test_bilinear<f16_t>, nk_bilinear_f16_neon);
    check("bilinear_f16c_neon", test_bilinear<f16c_t>, nk_bilinear_f16c_neon);
    check("mahalanobis_f16_neon", test_mahalanobis<f16_t>, nk_mahalanobis_f16_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONBFDOT
    check.section("Curved Kernels NEON BF16", nk_cap_neonbfdot_k);
    check("bilinear_bf16_neonbfdot", test_bilinear<bf16_t>, nk_bilinear_bf16_neonbfdot);
    check("bilinear_bf16c_neonbfdot", test_bilinear<bf16c_t>, nk_bilinear_bf16c_neonbfdot);
    check("mahalanobis_bf16_neonbfdot", test_mahalanobis<bf16_t>, nk_mahalanobis_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_HASWELL
    check.section("Curved Kernels Haswell", nk_cap_haswell_k);
    check("bilinear_f32_haswell", test_bilinear<f32_t>, nk_bilinear_f32_haswell);
    check("bilinear_f16_haswell", test_bilinear<f16_t>, nk_bilinear_f16_haswell);
    check("bilinear_bf16_haswell", test_bilinear<bf16_t>, nk_bilinear_bf16_haswell);
    check("mahalanobis_f32_haswell", test_mahalanobis<f32_t>, nk_mahalanobis_f32_haswell);
    check("mahalanobis_f16_haswell", test_mahalanobis<f16_t>, nk_mahalanobis_f16_haswell);
    check("mahalanobis_bf16_haswell", test_mahalanobis<bf16_t>, nk_mahalanobis_bf16_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check.section("Curved Kernels Skylake", nk_cap_skylake_k);
    check("bilinear_f32_skylake", test_bilinear<f32_t>, nk_bilinear_f32_skylake);
    check("bilinear_f64_skylake", test_bilinear<f64_t>, nk_bilinear_f64_skylake);
    check("bilinear_f32c_skylake", test_bilinear<f32c_t>, nk_bilinear_f32c_skylake);
    check("bilinear_f64c_skylake", test_bilinear<f64c_t>, nk_bilinear_f64c_skylake);
    check("mahalanobis_f32_skylake", test_mahalanobis<f32_t>, nk_mahalanobis_f32_skylake);
    check("mahalanobis_f64_skylake", test_mahalanobis<f64_t>, nk_mahalanobis_f64_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_GENOA
    check.section("Curved Kernels Genoa", nk_cap_genoa_k);
    check("bilinear_bf16_genoa", test_bilinear<bf16_t>, nk_bilinear_bf16_genoa);
    check("bilinear_bf16c_genoa", test_bilinear<bf16c_t>, nk_bilinear_bf16c_genoa);
    check("mahalanobis_bf16_genoa", test_mahalanobis<bf16_t>, nk_mahalanobis_bf16_genoa);
#endif // NK_TARGET_GENOA

#if NK_TARGET_SMEF64
    check.section("Curved Kernels SME F64", nk_cap_smef64_k);
    check("bilinear_f32_smef64", test_bilinear<f32_t>, nk_bilinear_f32_smef64);
    check("bilinear_f32c_smef64", test_bilinear<f32c_t>, nk_bilinear_f32c_smef64);
    check("mahalanobis_f32_smef64", test_mahalanobis<f32_t>, nk_mahalanobis_f32_smef64);
    check("bilinear_f64_smef64", test_bilinear<f64_t>, nk_bilinear_f64_smef64);
    check("bilinear_f64c_smef64", test_bilinear<f64c_t>, nk_bilinear_f64c_smef64);
    check("mahalanobis_f64_smef64", test_mahalanobis<f64_t>, nk_mahalanobis_f64_smef64);
#endif // NK_TARGET_SMEF64
}
