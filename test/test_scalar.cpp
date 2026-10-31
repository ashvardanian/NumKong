/**
 *  @brief Scalar precision and CPU execution-state tests.
 *  @file test/test_scalar.cpp
 */
#include "test.hpp"
#include "numkong/dots.h"

template <typename scalar_type_>
error_stats_t test_ulp_distance() {
    error_stats_t stats(comparison_family_t::exact_k);
    auto one = scalar_type_(1).to_bits();
    stats.accumulate(ulp_distance(scalar_type_::from_bits(one), scalar_type_::from_bits(one + 1)), std::uint64_t {1});
    stats.accumulate(ulp_distance(scalar_type_::from_bits(one | 0x8000), scalar_type_::from_bits((one + 1) | 0x8000)),
                     std::uint64_t {1});
    stats.accumulate(ulp_distance(scalar_type_::from_bits(0), scalar_type_::from_bits(0x8000)), std::uint64_t {0});
    stats.accumulate(ulp_distance(scalar_type_::from_bits(1), scalar_type_::from_bits(0x8001)), std::uint64_t {2});
    return stats;
}

error_stats_t test_thread_configuration() {
    error_stats_t stats(comparison_family_t::exact_k);
    stats.expect(nk_configure_thread(nk_capabilities()) != 0, "CPU thread configuration failed");
    return stats;
}

#if NK_TARGET_SME && !defined(_MSC_VER)
error_stats_t test_sme_register_preservation() {
    error_stats_t stats(comparison_family_t::exact_k);
    f16_t::dots_packed_size_kernel_t volatile size_fn = nk_dots_packed_size_f16_sme;
    __asm__ __volatile__("fmov d8, #8.5" ::: "v8");
    auto size = size_fn(4, 4);
    double preserved;
    __asm__ __volatile__("fmov %d0, d8" : "=w"(preserved));
    stats.accumulate(preserved, 8.5);
    stats.expect(size > 0, "SME packed size must be positive");

    nk_i4x2_t input[8] = {};
    auto packed = make_vector<char>(nk_dots_packed_size_i4_sme(4, 4));
    i4x2_t::dots_pack_kernel_t volatile pack_fn = nk_dots_pack_i4_sme;
    __asm__ __volatile__("fmov d8, #8.5" ::: "v8");
    pack_fn(input, 4, 4, 2, packed.raw_values_data());
    __asm__ __volatile__("fmov %d0, d8" : "=w"(preserved));
    stats.accumulate(preserved, 8.5);
    return stats;
}
#endif

void test_scalar() {
    error_stats_section_t check;
    check.section("Scalar", nk_cap_serial_k);
    check("ulp_distance_f16", test_ulp_distance<f16_t>);
    check("ulp_distance_bf16", test_ulp_distance<bf16_t>);
    check("thread_configuration", test_thread_configuration);
#if NK_TARGET_SME && !defined(_MSC_VER)
    check.section("SME state", nk_cap_sme_k);
    check("sme_register_preservation", test_sme_register_preservation);
#endif
}
