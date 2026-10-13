/**
 *  @file test/cast_metal.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Conversion tests on Metal.
 */
#include "harness.hpp"

#if NUMKONG_ARCH_METAL_
#include "cast.hpp"
#endif // NUMKONG_ARCH_METAL_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_METAL_
void test_cast_metal(error_stats_section_t &check) {
#if NUMKONG_TARGET_METAL
    check.section("Conversions Metal", nk_cap_metal_k);
    check("cast_pairs_metal", test_cast_pairs<metal_backend_t>, nk_cast_metal);
    check("cast_block_scaled_metal", test_cast_block_scaled<metal_backend_t>, nk_cast_metal);
#endif // NUMKONG_TARGET_METAL
}

#else  // !NUMKONG_ARCH_METAL_
void test_cast_metal(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test
