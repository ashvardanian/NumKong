/**
 *  @file test/probability.cpp
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *  @brief KL-divergence and Jensen-Shannon distance tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "numkong/probability.hpp" // `nk::kld`
#include "numkong/random.hpp"      // `nk::fill_probability`

namespace ashvardanian::numkong::test {

/** Σ pᵢ × log(pᵢ / qᵢ) in @p result_type_, the oracle of the KLD kernels. */
template <typename in_type_, typename result_type_>
void kld_reference(in_type_ const *p, in_type_ const *q, std::size_t d, result_type_ *r) noexcept {
    result_type_ sum {};
    for (std::size_t i = 0; i < d; i++) {
        result_type_ pi(p[i]), qi(q[i]);
        if (pi > result_type_(0)) sum = sum + pi * (pi / qi).log();
    }
    *r = sum;
}

/** √(½ × (KL(p‖m) + KL(q‖m))) in @p result_type_, the oracle of the JSD kernels. */
template <typename in_type_, typename result_type_>
void jsd_reference(in_type_ const *p, in_type_ const *q, std::size_t d, result_type_ *r) noexcept {
    result_type_ sum {};
    result_type_ half(0.5);
    for (std::size_t i = 0; i < d; i++) {
        result_type_ pi(p[i]), qi(q[i]);
        result_type_ mi = half * (pi + qi);
        if (pi > result_type_(0)) sum = sum + pi * (pi / mi).log();
        if (qi > result_type_(0)) sum = sum + qi * (qi / mi).log();
    }
    result_type_ divergence = half * sum;
    *r = divergence > result_type_(0) ? square_root_reference(divergence) : result_type_(0);
}

/** Template for the KL divergence test, whose inputs must be probability distributions: positive
 *  values summing to 1. */
template <typename scalar_type_>
error_stats_t test_kld(typename scalar_type_::probability_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::probability_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_probability_error_bound(scalar_t::dtype()),
                        nk_accumulation_error_bound(nk_probability_output_dtype(scalar_t::dtype())));
    split_mix_generator_t generator(case_stream_key());
    auto p = make_vector<scalar_t>(test_run.settings.dense_dimensions),
         q = make_vector<scalar_t>(test_run.settings.dense_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        nk::fill_probability(generator, p.values_data(), test_run.settings.dense_dimensions);
        nk::fill_probability(generator, q.values_data(), test_run.settings.dense_dimensions);

        result_t result;
        stats.expect_status(kernel(p.raw_values_data(), q.raw_values_data(), test_run.settings.dense_dimensions,
                                   &result.raw_, nullptr));

        reference_t reference;
        kld_reference<scalar_t, reference_t>(p.values_data(), q.values_data(), test_run.settings.dense_dimensions,
                                             &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/** Template for the Jensen-Shannon distance test, whose inputs must be probability distributions:
 *  positive values summing to 1. */
template <typename scalar_type_>
error_stats_t test_jsd(typename scalar_type_::probability_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using raw_t = typename scalar_t::raw_t;
    using result_t = typename scalar_t::probability_result_t;
    using reference_t = bounded_reference_for<scalar_t, result_t>;

    error_stats_t stats(nk_probability_error_bound(scalar_t::dtype()),
                        nk_accumulation_error_bound(nk_probability_output_dtype(scalar_t::dtype())));
    split_mix_generator_t generator(case_stream_key());
    auto p = make_vector<scalar_t>(test_run.settings.dense_dimensions),
         q = make_vector<scalar_t>(test_run.settings.dense_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        nk::fill_probability(generator, p.values_data(), test_run.settings.dense_dimensions);
        nk::fill_probability(generator, q.values_data(), test_run.settings.dense_dimensions);

        result_t result;
        stats.expect_status(kernel(p.raw_values_data(), q.raw_values_data(), test_run.settings.dense_dimensions,
                                   &result.raw_, nullptr));

        reference_t reference;
        jsd_reference<scalar_t, reference_t>(p.values_data(), q.values_data(), test_run.settings.dense_dimensions,
                                             &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void run_probability_cases() {

    run_kernel_case("nk_kld_f32_serial", test_kld<f32_t>, nk_cap_serial_k, nk_kld_f32_serial);
    run_kernel_case("nk_kld_f64_serial", test_kld<f64_t>, nk_cap_serial_k, nk_kld_f64_serial);
    run_kernel_case("nk_kld_bf16_serial", test_kld<bf16_t>, nk_cap_serial_k, nk_kld_bf16_serial);
    run_kernel_case("nk_kld_f16_serial", test_kld<f16_t>, nk_cap_serial_k, nk_kld_f16_serial);
    run_kernel_case("nk_jsd_f32_serial", test_jsd<f32_t>, nk_cap_serial_k, nk_jsd_f32_serial);
    run_kernel_case("nk_jsd_f64_serial", test_jsd<f64_t>, nk_cap_serial_k, nk_jsd_f64_serial);
    run_kernel_case("nk_jsd_bf16_serial", test_jsd<bf16_t>, nk_cap_serial_k, nk_jsd_bf16_serial);
    run_kernel_case("nk_jsd_f16_serial", test_jsd<f16_t>, nk_cap_serial_k, nk_jsd_f16_serial);

    run_kernel_case("nk_kld_f32_best", test_kld<f32_t>, nk_cap_serial_k, cpu_best<nk_kld_f32_best>);
    run_kernel_case("nk_kld_f64_best", test_kld<f64_t>, nk_cap_serial_k, cpu_best<nk_kld_f64_best>);
    run_kernel_case("nk_kld_f16_best", test_kld<f16_t>, nk_cap_serial_k, cpu_best<nk_kld_f16_best>);
    run_kernel_case("nk_kld_bf16_best", test_kld<bf16_t>, nk_cap_serial_k, cpu_best<nk_kld_bf16_best>);
    run_kernel_case("nk_jsd_f32_best", test_jsd<f32_t>, nk_cap_serial_k, cpu_best<nk_jsd_f32_best>);
    run_kernel_case("nk_jsd_f64_best", test_jsd<f64_t>, nk_cap_serial_k, cpu_best<nk_jsd_f64_best>);
    run_kernel_case("nk_jsd_f16_best", test_jsd<f16_t>, nk_cap_serial_k, cpu_best<nk_jsd_f16_best>);
    run_kernel_case("nk_jsd_bf16_best", test_jsd<bf16_t>, nk_cap_serial_k, cpu_best<nk_jsd_bf16_best>);

#if NUMKONG_TARGET_NEON
    run_kernel_case("nk_kld_f32_neon", test_kld<f32_t>, nk_cap_neon_k, nk_kld_f32_neon);
    run_kernel_case("nk_jsd_f32_neon", test_jsd<f32_t>, nk_cap_neon_k, nk_jsd_f32_neon);
    run_kernel_case("nk_kld_f64_neon", test_kld<f64_t>, nk_cap_neon_k, nk_kld_f64_neon);
    run_kernel_case("nk_jsd_f64_neon", test_jsd<f64_t>, nk_cap_neon_k, nk_jsd_f64_neon);
    run_kernel_case("nk_kld_f16_neon", test_kld<f16_t>, nk_cap_neon_k, nk_kld_f16_neon);
    run_kernel_case("nk_jsd_f16_neon", test_jsd<f16_t>, nk_cap_neon_k, nk_jsd_f16_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_HASWELL
    run_kernel_case("nk_kld_f32_haswell", test_kld<f32_t>, nk_cap_haswell_k, nk_kld_f32_haswell);
    run_kernel_case("nk_jsd_f32_haswell", test_jsd<f32_t>, nk_cap_haswell_k, nk_jsd_f32_haswell);
    run_kernel_case("nk_kld_f16_haswell", test_kld<f16_t>, nk_cap_haswell_k, nk_kld_f16_haswell);
    run_kernel_case("nk_kld_f64_haswell", test_kld<f64_t>, nk_cap_haswell_k, nk_kld_f64_haswell);
    run_kernel_case("nk_jsd_f16_haswell", test_jsd<f16_t>, nk_cap_haswell_k, nk_jsd_f16_haswell);
    run_kernel_case("nk_jsd_f64_haswell", test_jsd<f64_t>, nk_cap_haswell_k, nk_jsd_f64_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
    run_kernel_case("nk_kld_f32_skylake", test_kld<f32_t>, nk_cap_skylake_k, nk_kld_f32_skylake);
    run_kernel_case("nk_kld_f64_skylake", test_kld<f64_t>, nk_cap_skylake_k, nk_kld_f64_skylake);
    run_kernel_case("nk_jsd_f32_skylake", test_jsd<f32_t>, nk_cap_skylake_k, nk_jsd_f32_skylake);
    run_kernel_case("nk_jsd_f64_skylake", test_jsd<f64_t>, nk_cap_skylake_k, nk_jsd_f64_skylake);
    run_kernel_case("nk_kld_f16_skylake", test_kld<f16_t>, nk_cap_skylake_k, nk_kld_f16_skylake);
    run_kernel_case("nk_jsd_f16_skylake", test_jsd<f16_t>, nk_cap_skylake_k, nk_jsd_f16_skylake);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_RVV
    run_kernel_case("nk_kld_f32_rvv", test_kld<f32_t>, nk_cap_rvv_k, nk_kld_f32_rvv);
    run_kernel_case("nk_kld_f64_rvv", test_kld<f64_t>, nk_cap_rvv_k, nk_kld_f64_rvv);
    run_kernel_case("nk_kld_f16_rvv", test_kld<f16_t>, nk_cap_rvv_k, nk_kld_f16_rvv);
    run_kernel_case("nk_kld_bf16_rvv", test_kld<bf16_t>, nk_cap_rvv_k, nk_kld_bf16_rvv);
    run_kernel_case("nk_jsd_f32_rvv", test_jsd<f32_t>, nk_cap_rvv_k, nk_jsd_f32_rvv);
    run_kernel_case("nk_jsd_f64_rvv", test_jsd<f64_t>, nk_cap_rvv_k, nk_jsd_f64_rvv);
    run_kernel_case("nk_jsd_f16_rvv", test_jsd<f16_t>, nk_cap_rvv_k, nk_jsd_f16_rvv);
    run_kernel_case("nk_jsd_bf16_rvv", test_jsd<bf16_t>, nk_cap_rvv_k, nk_jsd_bf16_rvv);
#endif // NUMKONG_TARGET_RVV
}

} // namespace ashvardanian::numkong::test
