/**
 *  @brief KL-divergence and Jensen-Shannon distance tests.
 *  @file test/test_probability.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 */

#include "test.hpp"
#include "numkong/probability.hpp" // `nk::kld`

/**
 *  @brief Template for KL divergence test.
 *  KLD requires probability distributions: all values > 0, sum to 1.
 */
template <typename scalar_type_>
error_stats_t test_kld(typename scalar_type_::probability_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::probability_result_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::probability_k);
    std::mt19937 generator(global_config.seed);
    auto p = make_vector<scalar_t>(global_config.dense_dimensions),
         q = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_probability(generator, p.values_data(), global_config.dense_dimensions);
        nk::fill_probability(generator, q.values_data(), global_config.dense_dimensions);

        result_t result;
        kernel(p.raw_values_data(), q.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        reference_t reference;
        nk::kld<scalar_t, reference_t, nk::no_simd_k>(p.values_data(), q.values_data(), global_config.dense_dimensions,
                                                      &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/**
 *  @brief Template for Jensen-Shannon distance test.
 *  JSD requires probability distributions: all values > 0, sum to 1.
 */
template <typename scalar_type_>
error_stats_t test_jsd(typename scalar_type_::probability_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::probability_result_t;
    using reference_t = reference_for<scalar_t>;

    error_stats_t stats(comparison_family_t::probability_k);
    std::mt19937 generator(global_config.seed);
    auto p = make_vector<scalar_t>(global_config.dense_dimensions),
         q = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        nk::fill_probability(generator, p.values_data(), global_config.dense_dimensions);
        nk::fill_probability(generator, q.values_data(), global_config.dense_dimensions);

        result_t result;
        kernel(p.raw_values_data(), q.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        reference_t reference;
        nk::jsd<scalar_t, reference_t, nk::no_simd_k>(p.values_data(), q.values_data(), global_config.dense_dimensions,
                                                      &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void test_probability() {
    error_stats_section_t check;

    check.section("Probability Divergences Serial", nk_cap_serial_k);
    check("kld_f32_serial", test_kld<f32_t>, nk_kld_f32_serial);
    check("kld_f64_serial", test_kld<f64_t>, nk_kld_f64_serial);
    check("kld_bf16_serial", test_kld<bf16_t>, nk_kld_bf16_serial);
    check("kld_f16_serial", test_kld<f16_t>, nk_kld_f16_serial);
    check("jsd_f32_serial", test_jsd<f32_t>, nk_jsd_f32_serial);
    check("jsd_f64_serial", test_jsd<f64_t>, nk_jsd_f64_serial);
    check("jsd_bf16_serial", test_jsd<bf16_t>, nk_jsd_bf16_serial);
    check("jsd_f16_serial", test_jsd<f16_t>, nk_jsd_f16_serial);

#if NK_DYNAMIC_DISPATCH
    check.section("Probability Divergences Dynamic", nk_cap_serial_k);
    check("kld_f32", test_kld<f32_t>, nk_kld_f32);
    check("kld_f64", test_kld<f64_t>, nk_kld_f64);
    check("kld_f16", test_kld<f16_t>, nk_kld_f16);
    check("kld_bf16", test_kld<bf16_t>, nk_kld_bf16);
    check("jsd_f32", test_jsd<f32_t>, nk_jsd_f32);
    check("jsd_f64", test_jsd<f64_t>, nk_jsd_f64);
    check("jsd_f16", test_jsd<f16_t>, nk_jsd_f16);
    check("jsd_bf16", test_jsd<bf16_t>, nk_jsd_bf16);
#endif

#if NK_TARGET_NEON
    check.section("Probability Divergences NEON", nk_cap_neon_k);
    check("kld_f32_neon", test_kld<f32_t>, nk_kld_f32_neon);
    check("jsd_f32_neon", test_jsd<f32_t>, nk_jsd_f32_neon);
    check("kld_f16_neon", test_kld<f16_t>, nk_kld_f16_neon);
    check("jsd_f16_neon", test_jsd<f16_t>, nk_jsd_f16_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    check.section("Probability Divergences Haswell", nk_cap_haswell_k);
    check("kld_f16_haswell", test_kld<f16_t>, nk_kld_f16_haswell);
    check("kld_f64_haswell", test_kld<f64_t>, nk_kld_f64_haswell);
    check("jsd_f16_haswell", test_jsd<f16_t>, nk_jsd_f16_haswell);
    check("jsd_f64_haswell", test_jsd<f64_t>, nk_jsd_f64_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
    check.section("Probability Divergences Skylake", nk_cap_skylake_k);
    check("kld_f32_skylake", test_kld<f32_t>, nk_kld_f32_skylake);
    check("kld_f64_skylake", test_kld<f64_t>, nk_kld_f64_skylake);
    check("jsd_f32_skylake", test_jsd<f32_t>, nk_jsd_f32_skylake);
    check("jsd_f64_skylake", test_jsd<f64_t>, nk_jsd_f64_skylake);
    check("kld_f16_skylake", test_kld<f16_t>, nk_kld_f16_skylake);
    check("jsd_f16_skylake", test_jsd<f16_t>, nk_jsd_f16_skylake);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_RVV
    check.section("Probability Divergences RVV", nk_cap_rvv_k);
    check("kld_f32_rvv", test_kld<f32_t>, nk_kld_f32_rvv);
    check("kld_f64_rvv", test_kld<f64_t>, nk_kld_f64_rvv);
    check("kld_f16_rvv", test_kld<f16_t>, nk_kld_f16_rvv);
    check("kld_bf16_rvv", test_kld<bf16_t>, nk_kld_bf16_rvv);
    check("jsd_f32_rvv", test_jsd<f32_t>, nk_jsd_f32_rvv);
    check("jsd_f64_rvv", test_jsd<f64_t>, nk_jsd_f64_rvv);
    check("jsd_f16_rvv", test_jsd<f16_t>, nk_jsd_f16_rvv);
    check("jsd_bf16_rvv", test_jsd<bf16_t>, nk_jsd_bf16_rvv);
#endif // NK_TARGET_RVV
}
