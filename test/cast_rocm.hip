/**
 *  @file test/cast_rocm.hip
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Conversion tests on ROCm.
 */
#include "harness.hpp"

#if NUMKONG_ARCH_ROCM_
#include "cast.hpp"
#endif // NUMKONG_ARCH_ROCM_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_ROCM_
void test_cast_rocm(error_stats_section_t &check) {
    check.section("Conversions ROCm", nk_cap_rocm_k);
    check("cast_pairs_rocm", test_cast_pairs<rocm_backend_t>, nk_cast_rocm);
    check("cast_block_scaled_rocm", test_cast_block_scaled<rocm_backend_t>, nk_cast_rocm);
}

#else  // !NUMKONG_ARCH_ROCM_
void test_cast_rocm(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::test
