/**
 *  @file test/set.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Binary set operations tests: Hamming, Jaccard.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"
#include "numkong/set.hpp"

namespace ashvardanian::numkong::test {

/** Σ(aᵢ ⊕ bᵢ) counted in @p result_type_, the oracle of the Hamming kernels. */
template <typename in_type_, typename result_type_>
void hamming_reference(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    std::size_t const n = d / nk::dimensions_per_value<in_type_>();
    typename result_type_::raw_t count = 0;
    for (std::size_t i = 0; i < n; i++) count += count_differences(a[i], b[i]);
    *r = result_type_::from_raw(count);
}

/** 1 − |A ∩ B| / |A ∪ B| in @p result_type_, the oracle of the Jaccard kernels. */
template <typename in_type_, typename result_type_>
void jaccard_reference(in_type_ const *a, in_type_ const *b, std::size_t d, result_type_ *r) noexcept {
    std::size_t const n = d / nk::dimensions_per_value<in_type_>();
    std::uint32_t intersection_count = 0, union_count = 0;
    for (std::size_t i = 0; i < n; i++)
        intersection_count += count_intersection(a[i], b[i]), union_count += count_union(a[i], b[i]);
    if (union_count == 0) *r = result_type_();
    else *r = result_type_(1) - result_type_(intersection_count) / result_type_(union_count);
}

/** Test Hamming distance for binary or integer vectors. */
template <typename scalar_type_>
error_stats_t test_hamming(typename scalar_type_::hamming_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::hamming_result_t;

    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());

    auto a = make_vector<scalar_t>(test_run.settings.dense_dimensions),
         b = make_vector<scalar_t>(test_run.settings.dense_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), test_run.settings.dense_dimensions,
                                   &result.raw_, nullptr));

        result_t reference;
        hamming_reference<scalar_t, result_t>(a.values_data(), b.values_data(), test_run.settings.dense_dimensions,
                                              &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/** Test Jaccard distance for binary or integer vectors. */
template <typename scalar_type_>
error_stats_t test_jaccard(typename scalar_type_::jaccard_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::jaccard_result_t;

    error_stats_t stats(comparison_family_t::exact_k);
    split_mix_generator_t generator(case_stream_key());

    auto a = make_vector<scalar_t>(test_run.settings.dense_dimensions),
         b = make_vector<scalar_t>(test_run.settings.dense_dimensions);

    for (deadline_t const deadline = test_run.settings.deadline(); deadline.ahead();) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        stats.expect_status(kernel(a.raw_values_data(), b.raw_values_data(), test_run.settings.dense_dimensions,
                                   &result.raw_, nullptr));

        result_t reference;
        jaccard_reference<scalar_t, result_t>(a.values_data(), b.values_data(), test_run.settings.dense_dimensions,
                                              &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void run_set_cases() {

    run_kernel_case("nk_hamming_u1_serial", test_hamming<u1x8_t>, nk_cap_serial_k, nk_hamming_u1_serial);
    run_kernel_case("nk_jaccard_u1_serial", test_jaccard<u1x8_t>, nk_cap_serial_k, nk_jaccard_u1_serial);
    run_kernel_case("nk_hamming_u8_serial", test_hamming<u8_t>, nk_cap_serial_k, nk_hamming_u8_serial);
    run_kernel_case("nk_jaccard_u16_serial", test_jaccard<u16_t>, nk_cap_serial_k, nk_jaccard_u16_serial);
    run_kernel_case("nk_jaccard_u32_serial", test_jaccard<u32_t>, nk_cap_serial_k, nk_jaccard_u32_serial);

    run_kernel_case("nk_hamming_u1_best", test_hamming<u1x8_t>, nk_cap_serial_k, cpu_best<nk_hamming_u1_best>);
    run_kernel_case("nk_jaccard_u1_best", test_jaccard<u1x8_t>, nk_cap_serial_k, cpu_best<nk_jaccard_u1_best>);

#if NUMKONG_TARGET_NEON
    run_kernel_case("nk_hamming_u1_neon", test_hamming<u1x8_t>, nk_cap_neon_k, nk_hamming_u1_neon);
    run_kernel_case("nk_jaccard_u1_neon", test_jaccard<u1x8_t>, nk_cap_neon_k, nk_jaccard_u1_neon);
    run_kernel_case("nk_hamming_u8_neon", test_hamming<u8_t>, nk_cap_neon_k, nk_hamming_u8_neon);
    run_kernel_case("nk_jaccard_u16_neon", test_jaccard<u16_t>, nk_cap_neon_k, nk_jaccard_u16_neon);
    run_kernel_case("nk_jaccard_u32_neon", test_jaccard<u32_t>, nk_cap_neon_k, nk_jaccard_u32_neon);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_HASWELL
    run_kernel_case("nk_hamming_u1_haswell", test_hamming<u1x8_t>, nk_cap_haswell_k, nk_hamming_u1_haswell);
    run_kernel_case("nk_jaccard_u1_haswell", test_jaccard<u1x8_t>, nk_cap_haswell_k, nk_jaccard_u1_haswell);
    run_kernel_case("nk_hamming_u8_haswell", test_hamming<u8_t>, nk_cap_haswell_k, nk_hamming_u8_haswell);
    run_kernel_case("nk_jaccard_u16_haswell", test_jaccard<u16_t>, nk_cap_haswell_k, nk_jaccard_u16_haswell);
    run_kernel_case("nk_jaccard_u32_haswell", test_jaccard<u32_t>, nk_cap_haswell_k, nk_jaccard_u32_haswell);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ICELAKE
    run_kernel_case("nk_hamming_u1_icelake", test_hamming<u1x8_t>, nk_cap_icelake_k, nk_hamming_u1_icelake);
    run_kernel_case("nk_jaccard_u1_icelake", test_jaccard<u1x8_t>, nk_cap_icelake_k, nk_jaccard_u1_icelake);
    run_kernel_case("nk_hamming_u8_icelake", test_hamming<u8_t>, nk_cap_icelake_k, nk_hamming_u8_icelake);
    run_kernel_case("nk_jaccard_u16_icelake", test_jaccard<u16_t>, nk_cap_icelake_k, nk_jaccard_u16_icelake);
    run_kernel_case("nk_jaccard_u32_icelake", test_jaccard<u32_t>, nk_cap_icelake_k, nk_jaccard_u32_icelake);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_SVE
    run_kernel_case("nk_hamming_u1_sve", test_hamming<u1x8_t>, nk_cap_sve_k, nk_hamming_u1_sve);
    run_kernel_case("nk_jaccard_u1_sve", test_jaccard<u1x8_t>, nk_cap_sve_k, nk_jaccard_u1_sve);
    run_kernel_case("nk_hamming_u8_sve", test_hamming<u8_t>, nk_cap_sve_k, nk_hamming_u8_sve);
    run_kernel_case("nk_jaccard_u16_sve", test_jaccard<u16_t>, nk_cap_sve_k, nk_jaccard_u16_sve);
    run_kernel_case("nk_jaccard_u32_sve", test_jaccard<u32_t>, nk_cap_sve_k, nk_jaccard_u32_sve);
#endif // NUMKONG_TARGET_SVE

#if NUMKONG_TARGET_RVV
    run_kernel_case("nk_hamming_u1_rvv", test_hamming<u1x8_t>, nk_cap_rvv_k, nk_hamming_u1_rvv);
    run_kernel_case("nk_jaccard_u1_rvv", test_jaccard<u1x8_t>, nk_cap_rvv_k, nk_jaccard_u1_rvv);
    run_kernel_case("nk_hamming_u8_rvv", test_hamming<u8_t>, nk_cap_rvv_k, nk_hamming_u8_rvv);
    run_kernel_case("nk_jaccard_u16_rvv", test_jaccard<u16_t>, nk_cap_rvv_k, nk_jaccard_u16_rvv);
    run_kernel_case("nk_jaccard_u32_rvv", test_jaccard<u32_t>, nk_cap_rvv_k, nk_jaccard_u32_rvv);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128
    run_kernel_case("nk_hamming_u1_v128", test_hamming<u1x8_t>, nk_cap_v128_k, nk_hamming_u1_v128);
    run_kernel_case("nk_jaccard_u1_v128", test_jaccard<u1x8_t>, nk_cap_v128_k, nk_jaccard_u1_v128);
    run_kernel_case("nk_hamming_u8_v128", test_hamming<u8_t>, nk_cap_v128_k, nk_hamming_u8_v128);
    run_kernel_case("nk_jaccard_u16_v128", test_jaccard<u16_t>, nk_cap_v128_k, nk_jaccard_u16_v128);
    run_kernel_case("nk_jaccard_u32_v128", test_jaccard<u32_t>, nk_cap_v128_k, nk_jaccard_u32_v128);
#endif // NUMKONG_TARGET_V128

#if NUMKONG_TARGET_LOONGSONASX
    run_kernel_case("nk_hamming_u1_loongsonasx", test_hamming<u1x8_t>, nk_cap_loongsonasx_k, nk_hamming_u1_loongsonasx);
    run_kernel_case("nk_jaccard_u1_loongsonasx", test_jaccard<u1x8_t>, nk_cap_loongsonasx_k, nk_jaccard_u1_loongsonasx);
    run_kernel_case("nk_hamming_u8_loongsonasx", test_hamming<u8_t>, nk_cap_loongsonasx_k, nk_hamming_u8_loongsonasx);
#endif // NUMKONG_TARGET_LOONGSONASX

#if NUMKONG_TARGET_POWERVSX
    run_kernel_case("nk_hamming_u1_powervsx", test_hamming<u1x8_t>, nk_cap_powervsx_k, nk_hamming_u1_powervsx);
    run_kernel_case("nk_jaccard_u1_powervsx", test_jaccard<u1x8_t>, nk_cap_powervsx_k, nk_jaccard_u1_powervsx);
    run_kernel_case("nk_hamming_u8_powervsx", test_hamming<u8_t>, nk_cap_powervsx_k, nk_hamming_u8_powervsx);
#endif // NUMKONG_TARGET_POWERVSX
}

} // namespace ashvardanian::numkong::test
