/**
 *  @brief Batch operation tests - Power ISA family (VSX).
 *  @file test/test_cross_power.cpp
 *  @author Ash Vardanian
 *  @date March 24, 2026
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_power() {
    [[maybe_unused]] error_stats_section_t check;
#if NK_TARGET_POWERVSX
    check.section("Cross Power VSX", nk_cap_powervsx_k);
    check("dots_packed_f64_powervsx", test_dots_packed<f64_t>, nk_dots_packed_size_f64_powervsx,
          nk_dots_pack_f64_powervsx, nk_dots_packed_f64_powervsx);
    check("dots_packed_f32_powervsx", test_dots_packed<f32_t>, nk_dots_packed_size_f32_powervsx,
          nk_dots_pack_f32_powervsx, nk_dots_packed_f32_powervsx);
    check("dots_packed_f16_powervsx", test_dots_packed<f16_t>, nk_dots_packed_size_f16_powervsx,
          nk_dots_pack_f16_powervsx, nk_dots_packed_f16_powervsx);
    check("dots_packed_bf16_powervsx", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_powervsx,
          nk_dots_pack_bf16_powervsx, nk_dots_packed_bf16_powervsx);
    check("dots_packed_i8_powervsx", test_dots_packed<i8_t>, nk_dots_packed_size_i8_powervsx, nk_dots_pack_i8_powervsx,
          nk_dots_packed_i8_powervsx);
    check("dots_packed_u8_powervsx", test_dots_packed<u8_t>, nk_dots_packed_size_u8_powervsx, nk_dots_pack_u8_powervsx,
          nk_dots_packed_u8_powervsx);
    check("dots_packed_u1_powervsx", test_dots_packed<u1x8_t>, nk_dots_packed_size_u1_powervsx,
          nk_dots_pack_u1_powervsx, nk_dots_packed_u1_powervsx);

    check("dots_symmetric_f64_powervsx", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_powervsx);
    check("dots_symmetric_f32_powervsx", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_powervsx);
    check("dots_symmetric_f16_powervsx", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_powervsx);
    check("dots_symmetric_bf16_powervsx", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_powervsx);
    check("dots_symmetric_i8_powervsx", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_powervsx);
    check("dots_symmetric_u8_powervsx", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_powervsx);
    check("dots_symmetric_u1_powervsx", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_powervsx);

    check("angulars_packed_f64_powervsx", test_angulars_packed<f64_t>, nk_dots_packed_size_f64_powervsx,
          nk_dots_pack_f64_powervsx, nk_angulars_packed_f64_powervsx);
    check("angulars_packed_f32_powervsx", test_angulars_packed<f32_t>, nk_dots_packed_size_f32_powervsx,
          nk_dots_pack_f32_powervsx, nk_angulars_packed_f32_powervsx);
    check("angulars_packed_f16_powervsx", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_powervsx,
          nk_dots_pack_f16_powervsx, nk_angulars_packed_f16_powervsx);
    check("angulars_packed_bf16_powervsx", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_powervsx,
          nk_dots_pack_bf16_powervsx, nk_angulars_packed_bf16_powervsx);

    check("angulars_symmetric_f64_powervsx", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_powervsx);
    check("angulars_symmetric_f32_powervsx", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_powervsx);
    check("angulars_symmetric_f16_powervsx", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_powervsx);
    check("angulars_symmetric_bf16_powervsx", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_powervsx);

    check("euclideans_packed_f64_powervsx", test_euclideans_packed<f64_t>, nk_dots_packed_size_f64_powervsx,
          nk_dots_pack_f64_powervsx, nk_euclideans_packed_f64_powervsx);
    check("euclideans_packed_f32_powervsx", test_euclideans_packed<f32_t>, nk_dots_packed_size_f32_powervsx,
          nk_dots_pack_f32_powervsx, nk_euclideans_packed_f32_powervsx);
    check("euclideans_packed_f16_powervsx", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_powervsx,
          nk_dots_pack_f16_powervsx, nk_euclideans_packed_f16_powervsx);
    check("euclideans_packed_bf16_powervsx", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_powervsx,
          nk_dots_pack_bf16_powervsx, nk_euclideans_packed_bf16_powervsx);

    check("euclideans_symmetric_f64_powervsx", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_powervsx);
    check("euclideans_symmetric_f32_powervsx", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_powervsx);
    check("euclideans_symmetric_f16_powervsx", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_powervsx);
    check("euclideans_symmetric_bf16_powervsx", test_euclideans_symmetric<bf16_t>,
          nk_euclideans_symmetric_bf16_powervsx);

    check("hammings_packed_u1_powervsx", test_hammings_packed<u1x8_t>, nk_dots_packed_size_u1_powervsx,
          nk_dots_pack_u1_powervsx, nk_hammings_packed_u1_powervsx);
    check("hammings_symmetric_u1_powervsx", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_powervsx);

    check("jaccards_packed_u1_powervsx", test_jaccards_packed<u1x8_t>, nk_dots_packed_size_u1_powervsx,
          nk_dots_pack_u1_powervsx, nk_jaccards_packed_u1_powervsx);
    check("jaccards_symmetric_u1_powervsx", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_powervsx);
#endif // NK_TARGET_POWERVSX
}
