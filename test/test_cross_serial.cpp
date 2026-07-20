/**
 *  @brief Batch operation tests - Serial fallback.
 *  @file test/test_cross_serial.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_serial() {
    [[maybe_unused]] error_stats_section_t check;

#if NK_RUNTIME_DISPATCH
    check("dots_packed_f64", test_dots_packed<f64_t>, nk_dots_pack_size_f64, nk_dots_pack_f64, nk_dots_packed_f64);
    check("dots_packed_f32", test_dots_packed<f32_t>, nk_dots_pack_size_f32, nk_dots_pack_f32, nk_dots_packed_f32);
    check("dots_packed_f16", test_dots_packed<f16_t>, nk_dots_pack_size_f16, nk_dots_pack_f16, nk_dots_packed_f16);
    check("dots_packed_bf16", test_dots_packed<bf16_t>, nk_dots_pack_size_bf16, nk_dots_pack_bf16, nk_dots_packed_bf16);
    check("dots_packed_i8", test_dots_packed<i8_t>, nk_dots_pack_size_i8, nk_dots_pack_i8, nk_dots_packed_i8);
    check("dots_packed_i4", test_dots_packed<i4x2_t>, nk_dots_pack_size_i4, nk_dots_pack_i4, nk_dots_packed_i4);
    check("dots_packed_u4", test_dots_packed<u4x2_t>, nk_dots_pack_size_u4, nk_dots_pack_u4, nk_dots_packed_u4);
    check("dots_packed_u1", test_dots_packed<u1x8_t>, nk_dots_pack_size_u1, nk_dots_pack_u1, nk_dots_packed_u1);

    check("dots_symmetric_f64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64);
    check("dots_symmetric_f32", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32);
    check("dots_symmetric_f16", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16);
    check("dots_symmetric_bf16", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16);
    check("dots_symmetric_e4m3", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3);
    check("dots_symmetric_e5m2", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2);
    check("dots_symmetric_i8", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8);
    check("dots_symmetric_u8", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8);
    check("dots_symmetric_i4", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4);
    check("dots_symmetric_u4", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4);
    check("dots_symmetric_u1", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1);

    check("hammings_packed_u1", test_hammings_packed<u1x8_t>, nk_dots_pack_size_u1, nk_dots_pack_u1,
          nk_hammings_packed_u1);
    check("hammings_symmetric_u1", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1);

    check("jaccards_packed_u1", test_jaccards_packed<u1x8_t>, nk_dots_pack_size_u1, nk_dots_pack_u1,
          nk_jaccards_packed_u1);
    check("jaccards_symmetric_u1", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1);
#endif

    // Serial always runs - baseline tests
    check("dots_packed_f64_serial", test_dots_packed<f64_t>, nk_dots_pack_size_f64_serial, nk_dots_pack_f64_serial,
          nk_dots_packed_f64_serial);
    check("dots_packed_f32_serial", test_dots_packed<f32_t>, nk_dots_pack_size_f32_serial, nk_dots_pack_f32_serial,
          nk_dots_packed_f32_serial);
    check("dots_packed_f16_serial", test_dots_packed<f16_t>, nk_dots_pack_size_f16_serial, nk_dots_pack_f16_serial,
          nk_dots_packed_f16_serial);
    check("dots_packed_bf16_serial", test_dots_packed<bf16_t>, nk_dots_pack_size_bf16_serial, nk_dots_pack_bf16_serial,
          nk_dots_packed_bf16_serial);
    check("dots_packed_e4m3_serial", test_dots_packed<e4m3_t>, nk_dots_pack_size_e4m3_serial, nk_dots_pack_e4m3_serial,
          nk_dots_packed_e4m3_serial);
    check("dots_packed_e5m2_serial", test_dots_packed<e5m2_t>, nk_dots_pack_size_e5m2_serial, nk_dots_pack_e5m2_serial,
          nk_dots_packed_e5m2_serial);
    check("dots_packed_e2m3_serial", test_dots_packed<e2m3_t>, nk_dots_pack_size_e2m3_serial, nk_dots_pack_e2m3_serial,
          nk_dots_packed_e2m3_serial);
    check("dots_packed_e3m2_serial", test_dots_packed<e3m2_t>, nk_dots_pack_size_e3m2_serial, nk_dots_pack_e3m2_serial,
          nk_dots_packed_e3m2_serial);
    check("dots_packed_i8_serial", test_dots_packed<i8_t>, nk_dots_pack_size_i8_serial, nk_dots_pack_i8_serial,
          nk_dots_packed_i8_serial);
    check("dots_packed_u8_serial", test_dots_packed<u8_t>, nk_dots_pack_size_u8_serial, nk_dots_pack_u8_serial,
          nk_dots_packed_u8_serial);
    check("dots_packed_i4_serial", test_dots_packed<i4x2_t>, nk_dots_pack_size_i4_serial, nk_dots_pack_i4_serial,
          nk_dots_packed_i4_serial);
    check("dots_packed_u4_serial", test_dots_packed<u4x2_t>, nk_dots_pack_size_u4_serial, nk_dots_pack_u4_serial,
          nk_dots_packed_u4_serial);
    check("dots_packed_u1_serial", test_dots_packed<u1x8_t>, nk_dots_pack_size_u1_serial, nk_dots_pack_u1_serial,
          nk_dots_packed_u1_serial);

    // Serial symmetric tests
    check("dots_symmetric_f64_serial", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_serial);
    check("dots_symmetric_f32_serial", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_serial);
    check("dots_symmetric_f16_serial", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_serial);
    check("dots_symmetric_bf16_serial", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_serial);
    check("dots_symmetric_e4m3_serial", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_serial);
    check("dots_symmetric_e5m2_serial", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_serial);
    check("dots_symmetric_e2m3_serial", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_serial);
    check("dots_symmetric_e3m2_serial", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_serial);
    check("dots_symmetric_i8_serial", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_serial);
    check("dots_symmetric_u8_serial", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_serial);
    check("dots_symmetric_i4_serial", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_serial);
    check("dots_symmetric_u4_serial", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_serial);
    check("dots_symmetric_u1_serial", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_serial);

    // Serial hammings tests
    check("hammings_packed_u1_serial", test_hammings_packed<u1x8_t>, nk_dots_pack_size_u1_serial,
          nk_dots_pack_u1_serial, nk_hammings_packed_u1_serial);
    check("hammings_symmetric_u1_serial", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_serial);

    // Serial jaccards tests
    check("jaccards_packed_u1_serial", test_jaccards_packed<u1x8_t>, nk_dots_pack_size_u1_serial,
          nk_dots_pack_u1_serial, nk_jaccards_packed_u1_serial);
    check("jaccards_symmetric_u1_serial", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_serial);

    // Serial angulars packed tests
    check("angulars_packed_f64_serial", test_angulars_packed<f64_t>, nk_dots_pack_size_f64_serial,
          nk_dots_pack_f64_serial, nk_angulars_packed_f64_serial);
    check("angulars_packed_f32_serial", test_angulars_packed<f32_t>, nk_dots_pack_size_f32_serial,
          nk_dots_pack_f32_serial, nk_angulars_packed_f32_serial);
    check("angulars_packed_f16_serial", test_angulars_packed<f16_t>, nk_dots_pack_size_f16_serial,
          nk_dots_pack_f16_serial, nk_angulars_packed_f16_serial);
    check("angulars_packed_bf16_serial", test_angulars_packed<bf16_t>, nk_dots_pack_size_bf16_serial,
          nk_dots_pack_bf16_serial, nk_angulars_packed_bf16_serial);
    check("angulars_packed_e4m3_serial", test_angulars_packed<e4m3_t>, nk_dots_pack_size_e4m3_serial,
          nk_dots_pack_e4m3_serial, nk_angulars_packed_e4m3_serial);

    // Serial angulars symmetric tests
    check("angulars_symmetric_f64_serial", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_serial);
    check("angulars_symmetric_f32_serial", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_serial);
    check("angulars_symmetric_f16_serial", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_serial);
    check("angulars_symmetric_bf16_serial", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_serial);
    check("angulars_symmetric_e4m3_serial", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_serial);

    // Serial euclideans packed tests
    check("euclideans_packed_f64_serial", test_euclideans_packed<f64_t>, nk_dots_pack_size_f64_serial,
          nk_dots_pack_f64_serial, nk_euclideans_packed_f64_serial);
    check("euclideans_packed_f32_serial", test_euclideans_packed<f32_t>, nk_dots_pack_size_f32_serial,
          nk_dots_pack_f32_serial, nk_euclideans_packed_f32_serial);
    check("euclideans_packed_f16_serial", test_euclideans_packed<f16_t>, nk_dots_pack_size_f16_serial,
          nk_dots_pack_f16_serial, nk_euclideans_packed_f16_serial);
    check("euclideans_packed_bf16_serial", test_euclideans_packed<bf16_t>, nk_dots_pack_size_bf16_serial,
          nk_dots_pack_bf16_serial, nk_euclideans_packed_bf16_serial);
    check("euclideans_packed_e4m3_serial", test_euclideans_packed<e4m3_t>, nk_dots_pack_size_e4m3_serial,
          nk_dots_pack_e4m3_serial, nk_euclideans_packed_e4m3_serial);

    // Serial euclideans symmetric tests
    check("euclideans_symmetric_f64_serial", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_serial);
    check("euclideans_symmetric_f32_serial", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_serial);
    check("euclideans_symmetric_f16_serial", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_serial);
    check("euclideans_symmetric_bf16_serial", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_serial);
    check("euclideans_symmetric_e4m3_serial", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_serial);

    check("angulars_packed_e5m2_serial", test_angulars_packed<e5m2_t>, nk_dots_pack_size_e5m2_serial,
          nk_dots_pack_e5m2_serial, nk_angulars_packed_e5m2_serial);
    check("angulars_packed_e2m3_serial", test_angulars_packed<e2m3_t>, nk_dots_pack_size_e2m3_serial,
          nk_dots_pack_e2m3_serial, nk_angulars_packed_e2m3_serial);
    check("angulars_packed_e3m2_serial", test_angulars_packed<e3m2_t>, nk_dots_pack_size_e3m2_serial,
          nk_dots_pack_e3m2_serial, nk_angulars_packed_e3m2_serial);
    check("angulars_packed_i8_serial", test_angulars_packed<i8_t>, nk_dots_pack_size_i8_serial, nk_dots_pack_i8_serial,
          nk_angulars_packed_i8_serial);
    check("angulars_packed_u8_serial", test_angulars_packed<u8_t>, nk_dots_pack_size_u8_serial, nk_dots_pack_u8_serial,
          nk_angulars_packed_u8_serial);
    check("angulars_packed_i4_serial", test_angulars_packed<i4x2_t>, nk_dots_pack_size_i4_serial,
          nk_dots_pack_i4_serial, nk_angulars_packed_i4_serial);
    check("angulars_packed_u4_serial", test_angulars_packed<u4x2_t>, nk_dots_pack_size_u4_serial,
          nk_dots_pack_u4_serial, nk_angulars_packed_u4_serial);

    check("angulars_symmetric_e5m2_serial", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_serial);
    check("angulars_symmetric_e2m3_serial", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_serial);
    check("angulars_symmetric_e3m2_serial", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_serial);
    check("angulars_symmetric_i8_serial", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_serial);
    check("angulars_symmetric_u8_serial", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_serial);
    check("angulars_symmetric_i4_serial", test_angulars_symmetric<i4x2_t>, nk_angulars_symmetric_i4_serial);
    check("angulars_symmetric_u4_serial", test_angulars_symmetric<u4x2_t>, nk_angulars_symmetric_u4_serial);

    check("euclideans_packed_e5m2_serial", test_euclideans_packed<e5m2_t>, nk_dots_pack_size_e5m2_serial,
          nk_dots_pack_e5m2_serial, nk_euclideans_packed_e5m2_serial);
    check("euclideans_packed_e2m3_serial", test_euclideans_packed<e2m3_t>, nk_dots_pack_size_e2m3_serial,
          nk_dots_pack_e2m3_serial, nk_euclideans_packed_e2m3_serial);
    check("euclideans_packed_e3m2_serial", test_euclideans_packed<e3m2_t>, nk_dots_pack_size_e3m2_serial,
          nk_dots_pack_e3m2_serial, nk_euclideans_packed_e3m2_serial);
    check("euclideans_packed_i8_serial", test_euclideans_packed<i8_t>, nk_dots_pack_size_i8_serial,
          nk_dots_pack_i8_serial, nk_euclideans_packed_i8_serial);
    check("euclideans_packed_u8_serial", test_euclideans_packed<u8_t>, nk_dots_pack_size_u8_serial,
          nk_dots_pack_u8_serial, nk_euclideans_packed_u8_serial);
    check("euclideans_packed_i4_serial", test_euclideans_packed<i4x2_t>, nk_dots_pack_size_i4_serial,
          nk_dots_pack_i4_serial, nk_euclideans_packed_i4_serial);
    check("euclideans_packed_u4_serial", test_euclideans_packed<u4x2_t>, nk_dots_pack_size_u4_serial,
          nk_dots_pack_u4_serial, nk_euclideans_packed_u4_serial);

    check("euclideans_symmetric_e5m2_serial", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_serial);
    check("euclideans_symmetric_e2m3_serial", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_serial);
    check("euclideans_symmetric_e3m2_serial", test_euclideans_symmetric<e3m2_t>, nk_euclideans_symmetric_e3m2_serial);
    check("euclideans_symmetric_i8_serial", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_serial);
    check("euclideans_symmetric_u8_serial", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_serial);
    check("euclideans_symmetric_i4_serial", test_euclideans_symmetric<i4x2_t>, nk_euclideans_symmetric_i4_serial);
    check("euclideans_symmetric_u4_serial", test_euclideans_symmetric<u4x2_t>, nk_euclideans_symmetric_u4_serial);

    check("attention_packed_bf16_serial", test_attention_packed<bf16_t>, nk_attention_pack_size_bf16_serial,
          nk_attention_pack_bf16_serial, nk_attention_packed_bf16_serial);
    check("attention_packed_e4m3_serial", test_attention_packed<e4m3_t>, nk_attention_pack_size_e4m3_serial,
          nk_attention_pack_e4m3_serial, nk_attention_packed_e4m3_serial);
    check("attention_packed_i8_serial", test_attention_packed<i8_t>, nk_attention_pack_size_i8_serial,
          nk_attention_pack_i8_serial, nk_attention_packed_i8_serial);
}
