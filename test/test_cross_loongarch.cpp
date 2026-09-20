/**
 *  @brief Batch operation tests - LoongArch LASX ISA family.
 *  @file test/test_cross_loongarch.cpp
 *  @author Ash Vardanian
 *  @date March 23, 2026
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_loongarch() {
    [[maybe_unused]] error_stats_section_t check;
#if NK_TARGET_LOONGSONASX
    check.section("Cross LoongArch LASX", nk_cap_loongsonasx_k);
    check("dots_packed_f64_loongsonasx", test_dots_packed<f64_t>, nk_dots_packed_size_f64_loongsonasx,
          nk_dots_pack_f64_loongsonasx, nk_dots_packed_f64_loongsonasx);
    check("dots_packed_f32_loongsonasx", test_dots_packed<f32_t>, nk_dots_packed_size_f32_loongsonasx,
          nk_dots_pack_f32_loongsonasx, nk_dots_packed_f32_loongsonasx);
    check("dots_packed_bf16_loongsonasx", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_loongsonasx,
          nk_dots_pack_bf16_loongsonasx, nk_dots_packed_bf16_loongsonasx);
    check("dots_packed_f16_loongsonasx", test_dots_packed<f16_t>, nk_dots_packed_size_f16_loongsonasx,
          nk_dots_pack_f16_loongsonasx, nk_dots_packed_f16_loongsonasx);
    check("dots_packed_i8_loongsonasx", test_dots_packed<i8_t>, nk_dots_packed_size_i8_loongsonasx,
          nk_dots_pack_i8_loongsonasx, nk_dots_packed_i8_loongsonasx);
    check("dots_packed_u8_loongsonasx", test_dots_packed<u8_t>, nk_dots_packed_size_u8_loongsonasx,
          nk_dots_pack_u8_loongsonasx, nk_dots_packed_u8_loongsonasx);
    check("dots_packed_u1_loongsonasx", test_dots_packed<u1x8_t>, nk_dots_packed_size_u1_loongsonasx,
          nk_dots_pack_u1_loongsonasx, nk_dots_packed_u1_loongsonasx);

    check("dots_symmetric_f64_loongsonasx", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_loongsonasx);
    check("dots_symmetric_f32_loongsonasx", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_loongsonasx);
    check("dots_symmetric_bf16_loongsonasx", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_loongsonasx);
    check("dots_symmetric_f16_loongsonasx", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_loongsonasx);
    check("dots_symmetric_i8_loongsonasx", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_loongsonasx);
    check("dots_symmetric_u8_loongsonasx", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_loongsonasx);
    check("dots_symmetric_u1_loongsonasx", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_loongsonasx);

    check("angulars_packed_f64_loongsonasx", test_angulars_packed<f64_t>, nk_dots_packed_size_f64_loongsonasx,
          nk_dots_pack_f64_loongsonasx, nk_angulars_packed_f64_loongsonasx);
    check("angulars_packed_f32_loongsonasx", test_angulars_packed<f32_t>, nk_dots_packed_size_f32_loongsonasx,
          nk_dots_pack_f32_loongsonasx, nk_angulars_packed_f32_loongsonasx);
    check("angulars_packed_bf16_loongsonasx", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_loongsonasx,
          nk_dots_pack_bf16_loongsonasx, nk_angulars_packed_bf16_loongsonasx);
    check("angulars_packed_f16_loongsonasx", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_loongsonasx,
          nk_dots_pack_f16_loongsonasx, nk_angulars_packed_f16_loongsonasx);
    check("angulars_packed_i8_loongsonasx", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_loongsonasx,
          nk_dots_pack_i8_loongsonasx, nk_angulars_packed_i8_loongsonasx);
    check("angulars_packed_u8_loongsonasx", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_loongsonasx,
          nk_dots_pack_u8_loongsonasx, nk_angulars_packed_u8_loongsonasx);

    check("angulars_symmetric_f64_loongsonasx", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_loongsonasx);
    check("angulars_symmetric_f32_loongsonasx", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_loongsonasx);
    check("angulars_symmetric_bf16_loongsonasx", test_angulars_symmetric<bf16_t>,
          nk_angulars_symmetric_bf16_loongsonasx);
    check("angulars_symmetric_f16_loongsonasx", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_loongsonasx);
    check("angulars_symmetric_i8_loongsonasx", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_loongsonasx);
    check("angulars_symmetric_u8_loongsonasx", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_loongsonasx);

    check("euclideans_packed_f64_loongsonasx", test_euclideans_packed<f64_t>, nk_dots_packed_size_f64_loongsonasx,
          nk_dots_pack_f64_loongsonasx, nk_euclideans_packed_f64_loongsonasx);
    check("euclideans_packed_f32_loongsonasx", test_euclideans_packed<f32_t>, nk_dots_packed_size_f32_loongsonasx,
          nk_dots_pack_f32_loongsonasx, nk_euclideans_packed_f32_loongsonasx);
    check("euclideans_packed_bf16_loongsonasx", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_loongsonasx,
          nk_dots_pack_bf16_loongsonasx, nk_euclideans_packed_bf16_loongsonasx);
    check("euclideans_packed_f16_loongsonasx", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_loongsonasx,
          nk_dots_pack_f16_loongsonasx, nk_euclideans_packed_f16_loongsonasx);
    check("euclideans_packed_i8_loongsonasx", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_loongsonasx,
          nk_dots_pack_i8_loongsonasx, nk_euclideans_packed_i8_loongsonasx);
    check("euclideans_packed_u8_loongsonasx", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_loongsonasx,
          nk_dots_pack_u8_loongsonasx, nk_euclideans_packed_u8_loongsonasx);

    check("euclideans_symmetric_f64_loongsonasx", test_euclideans_symmetric<f64_t>,
          nk_euclideans_symmetric_f64_loongsonasx);
    check("euclideans_symmetric_f32_loongsonasx", test_euclideans_symmetric<f32_t>,
          nk_euclideans_symmetric_f32_loongsonasx);
    check("euclideans_symmetric_bf16_loongsonasx", test_euclideans_symmetric<bf16_t>,
          nk_euclideans_symmetric_bf16_loongsonasx);
    check("euclideans_symmetric_f16_loongsonasx", test_euclideans_symmetric<f16_t>,
          nk_euclideans_symmetric_f16_loongsonasx);
    check("euclideans_symmetric_i8_loongsonasx", test_euclideans_symmetric<i8_t>,
          nk_euclideans_symmetric_i8_loongsonasx);
    check("euclideans_symmetric_u8_loongsonasx", test_euclideans_symmetric<u8_t>,
          nk_euclideans_symmetric_u8_loongsonasx);

    check("hammings_packed_u1_loongsonasx", test_hammings_packed<u1x8_t>, nk_dots_packed_size_u1_serial,
          nk_dots_pack_u1_serial, nk_hammings_packed_u1_loongsonasx);
    check("hammings_symmetric_u1_loongsonasx", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_loongsonasx);
    check("jaccards_packed_u1_loongsonasx", test_jaccards_packed<u1x8_t>, nk_dots_packed_size_u1_serial,
          nk_dots_pack_u1_serial, nk_jaccards_packed_u1_loongsonasx);
    check("jaccards_symmetric_u1_loongsonasx", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_loongsonasx);
#endif // NK_TARGET_LOONGSONASX
}
