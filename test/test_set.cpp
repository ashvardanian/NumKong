/**
 *  @brief Binary set operations tests (Hamming, Jaccard).
 *  @file test/test_set.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */

#include "test.hpp"
#include "numkong/set.hpp"

/**
 *  @brief Test Hamming distance for binary or integer vectors.
 */
template <typename scalar_type_>
error_stats_t test_hamming(typename scalar_type_::hamming_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::hamming_result_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        result_t reference;
        nk::hamming<scalar_t, result_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                       &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

/**
 *  @brief Test Jaccard distance for binary or integer vectors.
 */
template <typename scalar_type_>
error_stats_t test_jaccard(typename scalar_type_::jaccard_kernel_t kernel) {
    using scalar_t = scalar_type_;
    using result_t = typename scalar_t::jaccard_result_t;

    error_stats_t stats(comparison_family_t::exact_k);
    std::mt19937 generator(global_config.seed);

    auto a = make_vector<scalar_t>(global_config.dense_dimensions),
         b = make_vector<scalar_t>(global_config.dense_dimensions);

    for (auto start = test_start_time(); within_time_budget(start);) {
        fill_random(generator, a);
        fill_random(generator, b);

        result_t result;
        kernel(a.raw_values_data(), b.raw_values_data(), global_config.dense_dimensions, &result.raw_);

        result_t reference;
        nk::jaccard<scalar_t, result_t, nk::no_simd_k>(a.values_data(), b.values_data(), global_config.dense_dimensions,
                                                       &reference);

        stats.accumulate(result, reference);
    }

    return stats;
}

void test_set() {
    error_stats_section_t check;

    check.section("Binary Distances Serial", nk_cap_serial_k);
    check("hamming_u1_serial", test_hamming<u1x8_t>, nk_hamming_u1_serial);
    check("jaccard_u1_serial", test_jaccard<u1x8_t>, nk_jaccard_u1_serial);
    check("hamming_u8_serial", test_hamming<u8_t>, nk_hamming_u8_serial);
    check("jaccard_u16_serial", test_jaccard<u16_t>, nk_jaccard_u16_serial);
    check("jaccard_u32_serial", test_jaccard<u32_t>, nk_jaccard_u32_serial);

#if NK_DYNAMIC_DISPATCH
    check.section("Binary Distances Dynamic", nk_cap_serial_k);
    // Dynamic dispatch - only test the dispatcher itself
    check("hamming_u1", test_hamming<u1x8_t>, nk_hamming_u1);
    check("jaccard_u1", test_jaccard<u1x8_t>, nk_jaccard_u1);
#endif

    // Static compilation - test all available ISA variants

#if NK_TARGET_NEON
    check.section("Binary Distances NEON", nk_cap_neon_k);
    check("hamming_u1_neon", test_hamming<u1x8_t>, nk_hamming_u1_neon);
    check("jaccard_u1_neon", test_jaccard<u1x8_t>, nk_jaccard_u1_neon);
    check("hamming_u8_neon", test_hamming<u8_t>, nk_hamming_u8_neon);
    check("jaccard_u16_neon", test_jaccard<u16_t>, nk_jaccard_u16_neon);
    check("jaccard_u32_neon", test_jaccard<u32_t>, nk_jaccard_u32_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
    check.section("Binary Distances Haswell", nk_cap_haswell_k);
    check("hamming_u1_haswell", test_hamming<u1x8_t>, nk_hamming_u1_haswell);
    check("jaccard_u1_haswell", test_jaccard<u1x8_t>, nk_jaccard_u1_haswell);
    check("hamming_u8_haswell", test_hamming<u8_t>, nk_hamming_u8_haswell);
    check("jaccard_u16_haswell", test_jaccard<u16_t>, nk_jaccard_u16_haswell);
    check("jaccard_u32_haswell", test_jaccard<u32_t>, nk_jaccard_u32_haswell);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ICELAKE
    check.section("Binary Distances Ice Lake", nk_cap_icelake_k);
    check("hamming_u1_icelake", test_hamming<u1x8_t>, nk_hamming_u1_icelake);
    check("jaccard_u1_icelake", test_jaccard<u1x8_t>, nk_jaccard_u1_icelake);
    check("hamming_u8_icelake", test_hamming<u8_t>, nk_hamming_u8_icelake);
    check("jaccard_u16_icelake", test_jaccard<u16_t>, nk_jaccard_u16_icelake);
    check("jaccard_u32_icelake", test_jaccard<u32_t>, nk_jaccard_u32_icelake);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_SVE
    check.section("Binary Distances SVE", nk_cap_sve_k);
    check("hamming_u1_sve", test_hamming<u1x8_t>, nk_hamming_u1_sve);
    check("jaccard_u1_sve", test_jaccard<u1x8_t>, nk_jaccard_u1_sve);
    check("hamming_u8_sve", test_hamming<u8_t>, nk_hamming_u8_sve);
    check("jaccard_u16_sve", test_jaccard<u16_t>, nk_jaccard_u16_sve);
    check("jaccard_u32_sve", test_jaccard<u32_t>, nk_jaccard_u32_sve);
#endif // NK_TARGET_SVE

#if NK_TARGET_RVV
    check.section("Binary Distances RVV", nk_cap_rvv_k);
    check("hamming_u1_rvv", test_hamming<u1x8_t>, nk_hamming_u1_rvv);
    check("jaccard_u1_rvv", test_jaccard<u1x8_t>, nk_jaccard_u1_rvv);
    check("hamming_u8_rvv", test_hamming<u8_t>, nk_hamming_u8_rvv);
    check("jaccard_u16_rvv", test_jaccard<u16_t>, nk_jaccard_u16_rvv);
    check("jaccard_u32_rvv", test_jaccard<u32_t>, nk_jaccard_u32_rvv);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128RELAXED
    check.section("Binary Distances V128 Relaxed", nk_cap_v128relaxed_k);
    check("hamming_u1_v128relaxed", test_hamming<u1x8_t>, nk_hamming_u1_v128relaxed);
    check("jaccard_u1_v128relaxed", test_jaccard<u1x8_t>, nk_jaccard_u1_v128relaxed);
    check("hamming_u8_v128relaxed", test_hamming<u8_t>, nk_hamming_u8_v128relaxed);
    check("jaccard_u16_v128relaxed", test_jaccard<u16_t>, nk_jaccard_u16_v128relaxed);
    check("jaccard_u32_v128relaxed", test_jaccard<u32_t>, nk_jaccard_u32_v128relaxed);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_LOONGSONASX
    check.section("Binary Distances LoongArch LASX", nk_cap_loongsonasx_k);
    check("hamming_u1_loongsonasx", test_hamming<u1x8_t>, nk_hamming_u1_loongsonasx);
    check("jaccard_u1_loongsonasx", test_jaccard<u1x8_t>, nk_jaccard_u1_loongsonasx);
    check("hamming_u8_loongsonasx", test_hamming<u8_t>, nk_hamming_u8_loongsonasx);
#endif // NK_TARGET_LOONGSONASX

#if NK_TARGET_POWERVSX
    check.section("Binary Distances Power VSX", nk_cap_powervsx_k);
    check("hamming_u1_powervsx", test_hamming<u1x8_t>, nk_hamming_u1_powervsx);
    check("jaccard_u1_powervsx", test_jaccard<u1x8_t>, nk_jaccard_u1_powervsx);
    check("hamming_u8_powervsx", test_hamming<u8_t>, nk_hamming_u8_powervsx);
#endif // NK_TARGET_POWERVSX
}
