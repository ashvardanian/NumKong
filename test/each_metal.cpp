/**
 *  @file test/each_metal.cpp
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Elementwise operation tests on Metal.
 */
#include "harness.hpp"

#if NUMKONG_ARCH_METAL_
#include "each.hpp"
#endif // NUMKONG_ARCH_METAL_

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_METAL_
void test_each_metal(error_stats_section_t &check) {
#if NUMKONG_TARGET_METAL
    check.section("Elementwise Operations Metal", nk_cap_metal_k);
    check("each_swiglu_f32_metal", test_swiglu<f32_t, metal_backend_t>, nk_each_swiglu_f32_metal);
    check("each_swiglu_f16_metal", test_swiglu<f16_t, metal_backend_t>, nk_each_swiglu_f16_metal);
    check("each_swiglu_bf16_metal", test_swiglu<bf16_t, metal_backend_t>, nk_each_swiglu_bf16_metal);
    check("each_swiglu_e4m3_metal", test_swiglu<e4m3_t, metal_backend_t>, nk_each_swiglu_e4m3_metal);
    check("each_rmsnorm_f32_metal", test_rmsnorm<f32_t, metal_backend_t>, nk_each_rmsnorm_f32_metal);
    check("each_rmsnorm_f16_metal", test_rmsnorm<f16_t, metal_backend_t>, nk_each_rmsnorm_f16_metal);
    check("each_rmsnorm_bf16_metal", test_rmsnorm<bf16_t, metal_backend_t>, nk_each_rmsnorm_bf16_metal);
    check("each_rmsnorm_e4m3_metal", test_rmsnorm<e4m3_t, metal_backend_t>, nk_each_rmsnorm_e4m3_metal);
    check("each_rmscast_bf16_metal", test_rmsnorm<bf16_t, metal_backend_t, f32_t>, nk_each_rmscast_bf16_metal);
    check("each_rmscast_f16_metal", test_rmsnorm<f16_t, metal_backend_t, f32_t>, nk_each_rmscast_f16_metal);
    check("each_rmscast_e4m3_metal", test_rmsnorm<e4m3_t, metal_backend_t, f32_t>, nk_each_rmscast_e4m3_metal);
    check("each_rmscast_e5m2_metal", test_rmsnorm<e5m2_t, metal_backend_t, f32_t>, nk_each_rmscast_e5m2_metal);
    check("each_rmscast_e2m3_metal", test_rmsnorm<e2m3_t, metal_backend_t, f32_t>, nk_each_rmscast_e2m3_metal);
    check("each_rmscast_e3m2_metal", test_rmsnorm<e3m2_t, metal_backend_t, f32_t>, nk_each_rmscast_e3m2_metal);
    check("each_rmscast_i8_metal", test_rmsnorm<i8_t, metal_backend_t, i32_t>, nk_each_rmscast_i8_metal);
    check("each_rmscast_u8_metal", test_rmsnorm<u8_t, metal_backend_t, u32_t>, nk_each_rmscast_u8_metal);
#endif // NUMKONG_TARGET_METAL
}

#else  // !NUMKONG_ARCH_METAL_
void test_each_metal(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_METAL_

} // namespace ashvardanian::numkong::test
