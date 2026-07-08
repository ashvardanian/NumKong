/**
 *  @brief Batch operation tests - SME ISA.
 *  @file test/test_cross_sme.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_sme() {
    [[maybe_unused]] error_stats_section_t check;
#if NK_TARGET_SME
    check("dots_packed_f16_sme", test_dots_packed<f16_t>, nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
          nk_dots_packed_f16_sme);
    check("dots_packed_bf16_sme", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_sme, nk_dots_pack_bf16_sme,
          nk_dots_packed_bf16_sme);
    check("dots_packed_e4m3_sme", test_dots_packed<e4m3_t>, nk_dots_packed_size_e4m3_sme, nk_dots_pack_e4m3_sme,
          nk_dots_packed_e4m3_sme);
    check("dots_packed_e5m2_sme", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_sme, nk_dots_pack_e5m2_sme,
          nk_dots_packed_e5m2_sme);
    check("dots_packed_e2m3_sme", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_sme, nk_dots_pack_e2m3_sme,
          nk_dots_packed_e2m3_sme);
    check("dots_packed_e3m2_sme", test_dots_packed<e3m2_t>, nk_dots_packed_size_e3m2_sme, nk_dots_pack_e3m2_sme,
          nk_dots_packed_e3m2_sme);
    check("dots_packed_i8_sme", test_dots_packed<i8_t>, nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme,
          nk_dots_packed_i8_sme);
    check("dots_packed_u8_sme", test_dots_packed<u8_t>, nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme,
          nk_dots_packed_u8_sme);
    check("dots_packed_i4_sme", test_dots_packed<i4x2_t>, nk_dots_packed_size_i4_sme, nk_dots_pack_i4_sme,
          nk_dots_packed_i4_sme);
    check("dots_packed_u4_sme", test_dots_packed<u4x2_t>, nk_dots_packed_size_u4_sme, nk_dots_pack_u4_sme,
          nk_dots_packed_u4_sme);

    check("dots_symmetric_f16_sme", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_sme);
    check("dots_symmetric_bf16_sme", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_sme);
    check("dots_symmetric_e4m3_sme", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_sme);
    check("dots_symmetric_e5m2_sme", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_sme);
    check("dots_symmetric_e2m3_sme", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_sme);
    check("dots_symmetric_e3m2_sme", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_sme);
    check("dots_symmetric_i8_sme", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sme);
    check("dots_symmetric_u8_sme", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sme);
    check("dots_symmetric_i4_sme", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_sme);
    check("dots_symmetric_u4_sme", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_sme);

    check("angulars_packed_f16_sme", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
          nk_angulars_packed_f16_sme);
    check("angulars_packed_bf16_sme", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_sme, nk_dots_pack_bf16_sme,
          nk_angulars_packed_bf16_sme);
    check("angulars_packed_e4m3_sme", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_sme, nk_dots_pack_e4m3_sme,
          nk_angulars_packed_e4m3_sme);
    check("angulars_packed_e5m2_sme", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_sme, nk_dots_pack_e5m2_sme,
          nk_angulars_packed_e5m2_sme);
    check("angulars_packed_e2m3_sme", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_sme, nk_dots_pack_e2m3_sme,
          nk_angulars_packed_e2m3_sme);
    check("angulars_packed_e3m2_sme", test_angulars_packed<e3m2_t>, nk_dots_packed_size_e3m2_sme, nk_dots_pack_e3m2_sme,
          nk_angulars_packed_e3m2_sme);
    check("angulars_packed_i8_sme", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme,
          nk_angulars_packed_i8_sme);
    check("angulars_packed_u8_sme", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme,
          nk_angulars_packed_u8_sme);
    check("angulars_packed_i4_sme", test_angulars_packed<i4x2_t>, nk_dots_packed_size_i4_sme, nk_dots_pack_i4_sme,
          nk_angulars_packed_i4_sme);
    check("angulars_packed_u4_sme", test_angulars_packed<u4x2_t>, nk_dots_packed_size_u4_sme, nk_dots_pack_u4_sme,
          nk_angulars_packed_u4_sme);

    check("angulars_symmetric_f16_sme", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_sme);
    check("angulars_symmetric_bf16_sme", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_sme);
    check("angulars_symmetric_e4m3_sme", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_sme);
    check("angulars_symmetric_e5m2_sme", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_sme);
    check("angulars_symmetric_e2m3_sme", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_sme);
    check("angulars_symmetric_e3m2_sme", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_sme);
    check("angulars_symmetric_i8_sme", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_sme);
    check("angulars_symmetric_u8_sme", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_sme);
    check("angulars_symmetric_i4_sme", test_angulars_symmetric<i4x2_t>, nk_angulars_symmetric_i4_sme);
    check("angulars_symmetric_u4_sme", test_angulars_symmetric<u4x2_t>, nk_angulars_symmetric_u4_sme);

    check("euclideans_packed_f16_sme", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_sme, nk_dots_pack_f16_sme,
          nk_euclideans_packed_f16_sme);
    check("euclideans_packed_bf16_sme", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_sme,
          nk_dots_pack_bf16_sme, nk_euclideans_packed_bf16_sme);
    check("euclideans_packed_e4m3_sme", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_sme,
          nk_dots_pack_e4m3_sme, nk_euclideans_packed_e4m3_sme);
    check("euclideans_packed_e5m2_sme", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_sme,
          nk_dots_pack_e5m2_sme, nk_euclideans_packed_e5m2_sme);
    check("euclideans_packed_e2m3_sme", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_sme,
          nk_dots_pack_e2m3_sme, nk_euclideans_packed_e2m3_sme);
    check("euclideans_packed_e3m2_sme", test_euclideans_packed<e3m2_t>, nk_dots_packed_size_e3m2_sme,
          nk_dots_pack_e3m2_sme, nk_euclideans_packed_e3m2_sme);
    check("euclideans_packed_i8_sme", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_sme, nk_dots_pack_i8_sme,
          nk_euclideans_packed_i8_sme);
    check("euclideans_packed_u8_sme", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_sme, nk_dots_pack_u8_sme,
          nk_euclideans_packed_u8_sme);
    check("euclideans_packed_i4_sme", test_euclideans_packed<i4x2_t>, nk_dots_packed_size_i4_sme, nk_dots_pack_i4_sme,
          nk_euclideans_packed_i4_sme);
    check("euclideans_packed_u4_sme", test_euclideans_packed<u4x2_t>, nk_dots_packed_size_u4_sme, nk_dots_pack_u4_sme,
          nk_euclideans_packed_u4_sme);

    check("euclideans_symmetric_f16_sme", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_sme);
    check("euclideans_symmetric_bf16_sme", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_sme);
    check("euclideans_symmetric_e4m3_sme", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_sme);
    check("euclideans_symmetric_e5m2_sme", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_sme);
    check("euclideans_symmetric_e2m3_sme", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_sme);
    check("euclideans_symmetric_e3m2_sme", test_euclideans_symmetric<e3m2_t>, nk_euclideans_symmetric_e3m2_sme);
    check("euclideans_symmetric_i8_sme", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_sme);
    check("euclideans_symmetric_u8_sme", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_sme);
    check("euclideans_symmetric_i4_sme", test_euclideans_symmetric<i4x2_t>, nk_euclideans_symmetric_i4_sme);
    check("euclideans_symmetric_u4_sme", test_euclideans_symmetric<u4x2_t>, nk_euclideans_symmetric_u4_sme);

    check("attention_packed_bf16_sme", test_attention_packed<bf16_t>, nk_attention_packed_size_bf16_sme,
          nk_attention_pack_bf16_sme, nk_attention_packed_bf16_sme);
#endif

#if NK_TARGET_SMEBI32
    check("dots_packed_u1_smebi32", test_dots_packed<u1x8_t>, nk_dots_packed_size_u1_smebi32, nk_dots_pack_u1_smebi32,
          nk_dots_packed_u1_smebi32);
    check("dots_symmetric_u1_smebi32", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_smebi32);

    check("hammings_packed_u1_smebi32", test_hammings_packed<u1x8_t>, nk_dots_packed_size_u1_smebi32,
          nk_dots_pack_u1_smebi32, nk_hammings_packed_u1_smebi32);
    check("hammings_symmetric_u1_smebi32", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_smebi32);

    check("jaccards_packed_u1_smebi32", test_jaccards_packed<u1x8_t>, nk_dots_packed_size_u1_smebi32,
          nk_dots_pack_u1_smebi32, nk_jaccards_packed_u1_smebi32);
    check("jaccards_symmetric_u1_smebi32", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_smebi32);
#endif

#if NK_TARGET_SMEF64
    check("dots_packed_f64_smef64", test_dots_packed<f64_t>, nk_dots_packed_size_f64_smef64, nk_dots_pack_f64_smef64,
          nk_dots_packed_f64_smef64);
    check("dots_packed_f32_smef64", test_dots_packed<f32_t>, nk_dots_packed_size_f32_smef64, nk_dots_pack_f32_smef64,
          nk_dots_packed_f32_smef64);
    check("dots_symmetric_f64_smef64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_smef64);
    check("dots_symmetric_f32_smef64", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_smef64);

    check("angulars_packed_f32_smef64", test_angulars_packed<f32_t>, nk_dots_packed_size_f32_smef64,
          nk_dots_pack_f32_smef64, nk_angulars_packed_f32_smef64);
    check("angulars_packed_f64_smef64", test_angulars_packed<f64_t>, nk_dots_packed_size_f64_smef64,
          nk_dots_pack_f64_smef64, nk_angulars_packed_f64_smef64);

    check("angulars_symmetric_f32_smef64", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_smef64);
    check("angulars_symmetric_f64_smef64", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_smef64);

    check("euclideans_packed_f32_smef64", test_euclideans_packed<f32_t>, nk_dots_packed_size_f32_smef64,
          nk_dots_pack_f32_smef64, nk_euclideans_packed_f32_smef64);
    check("euclideans_packed_f64_smef64", test_euclideans_packed<f64_t>, nk_dots_packed_size_f64_smef64,
          nk_dots_pack_f64_smef64, nk_euclideans_packed_f64_smef64);

    check("euclideans_symmetric_f32_smef64", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_smef64);
    check("euclideans_symmetric_f64_smef64", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_smef64);
#endif
}
