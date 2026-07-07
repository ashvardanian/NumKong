/**
 *  @brief Batch operation tests - x86 ISA family.
 *  @file test/test_cross_x86.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  Covers Haswell, Skylake, Ice Lake, Genoa, Sapphire.
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_x86() {
    [[maybe_unused]] error_stats_section_t check;
#if NK_TARGET_HASWELL

    check("dots_packed_f64_haswell", test_dots_packed<f64_t>, nk_dots_packed_size_f64_haswell, nk_dots_pack_f64_haswell,
          nk_dots_packed_f64_haswell);
    check("dots_packed_f32_haswell", test_dots_packed<f32_t>, nk_dots_packed_size_f32_haswell, nk_dots_pack_f32_haswell,
          nk_dots_packed_f32_haswell);
    check("dots_packed_e4m3_haswell", test_dots_packed<e4m3_t>, nk_dots_packed_size_e4m3_haswell,
          nk_dots_pack_e4m3_haswell, nk_dots_packed_e4m3_haswell);
    check("dots_packed_e5m2_haswell", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_haswell,
          nk_dots_pack_e5m2_haswell, nk_dots_packed_e5m2_haswell);
    check("dots_packed_e2m3_haswell", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_haswell,
          nk_dots_pack_e2m3_haswell, nk_dots_packed_e2m3_haswell);
    check("dots_packed_e3m2_haswell", test_dots_packed<e3m2_t>, nk_dots_packed_size_e3m2_haswell,
          nk_dots_pack_e3m2_haswell, nk_dots_packed_e3m2_haswell);
    check("dots_packed_f16_haswell", test_dots_packed<f16_t>, nk_dots_packed_size_f16_haswell, nk_dots_pack_f16_haswell,
          nk_dots_packed_f16_haswell);
    check("dots_packed_bf16_haswell", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_haswell,
          nk_dots_pack_bf16_haswell, nk_dots_packed_bf16_haswell);
    check("dots_packed_i8_haswell", test_dots_packed<i8_t>, nk_dots_packed_size_i8_haswell, nk_dots_pack_i8_haswell,
          nk_dots_packed_i8_haswell);
    check("dots_packed_u8_haswell", test_dots_packed<u8_t>, nk_dots_packed_size_u8_haswell, nk_dots_pack_u8_haswell,
          nk_dots_packed_u8_haswell);
    check("dots_packed_u1_haswell", test_dots_packed<u1x8_t>, nk_dots_packed_size_u1_serial, nk_dots_pack_u1_serial,
          nk_dots_packed_u1_haswell);

    check("dots_symmetric_f64_haswell", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_haswell);
    check("dots_symmetric_f32_haswell", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_haswell);
    check("dots_symmetric_f16_haswell", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_haswell);
    check("dots_symmetric_bf16_haswell", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_haswell);
    check("dots_symmetric_e4m3_haswell", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_haswell);
    check("dots_symmetric_e5m2_haswell", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_haswell);
    check("dots_symmetric_e2m3_haswell", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_haswell);
    check("dots_symmetric_e3m2_haswell", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_haswell);
    check("dots_symmetric_i8_haswell", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_haswell);
    check("dots_symmetric_u8_haswell", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_haswell);
    check("dots_symmetric_u1_haswell", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_haswell);

    check("hammings_packed_u1_haswell", test_hammings_packed<u1x8_t>, nk_dots_packed_size_u1_serial,
          nk_dots_pack_u1_serial, nk_hammings_packed_u1_haswell);
    check("hammings_symmetric_u1_haswell", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_haswell);

    check("jaccards_packed_u1_haswell", test_jaccards_packed<u1x8_t>, nk_dots_packed_size_u1_serial,
          nk_dots_pack_u1_serial, nk_jaccards_packed_u1_haswell);
    check("jaccards_symmetric_u1_haswell", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_haswell);

    check("angulars_packed_f64_haswell", test_angulars_packed<f64_t>, nk_dots_packed_size_f64_haswell,
          nk_dots_pack_f64_haswell, nk_angulars_packed_f64_haswell);
    check("angulars_packed_f32_haswell", test_angulars_packed<f32_t>, nk_dots_packed_size_f32_haswell,
          nk_dots_pack_f32_haswell, nk_angulars_packed_f32_haswell);
    check("angulars_packed_f16_haswell", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_haswell,
          nk_dots_pack_f16_haswell, nk_angulars_packed_f16_haswell);
    check("angulars_packed_bf16_haswell", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_haswell,
          nk_dots_pack_bf16_haswell, nk_angulars_packed_bf16_haswell);
    check("angulars_packed_e4m3_haswell", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_haswell,
          nk_dots_pack_e4m3_haswell, nk_angulars_packed_e4m3_haswell);
    check("angulars_packed_e5m2_haswell", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_haswell,
          nk_dots_pack_e5m2_haswell, nk_angulars_packed_e5m2_haswell);
    check("angulars_packed_e2m3_haswell", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_haswell,
          nk_dots_pack_e2m3_haswell, nk_angulars_packed_e2m3_haswell);
    check("angulars_packed_e3m2_haswell", test_angulars_packed<e3m2_t>, nk_dots_packed_size_e3m2_haswell,
          nk_dots_pack_e3m2_haswell, nk_angulars_packed_e3m2_haswell);
    check("angulars_packed_i8_haswell", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_haswell,
          nk_dots_pack_i8_haswell, nk_angulars_packed_i8_haswell);
    check("angulars_packed_u8_haswell", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_haswell,
          nk_dots_pack_u8_haswell, nk_angulars_packed_u8_haswell);

    check("angulars_symmetric_f64_haswell", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_haswell);
    check("angulars_symmetric_f32_haswell", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_haswell);
    check("angulars_symmetric_f16_haswell", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_haswell);
    check("angulars_symmetric_bf16_haswell", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_haswell);
    check("angulars_symmetric_e4m3_haswell", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_haswell);
    check("angulars_symmetric_e5m2_haswell", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_haswell);
    check("angulars_symmetric_e2m3_haswell", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_haswell);
    check("angulars_symmetric_e3m2_haswell", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_haswell);
    check("angulars_symmetric_i8_haswell", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_haswell);
    check("angulars_symmetric_u8_haswell", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_haswell);

    check("euclideans_packed_f64_haswell", test_euclideans_packed<f64_t>, nk_dots_packed_size_f64_haswell,
          nk_dots_pack_f64_haswell, nk_euclideans_packed_f64_haswell);
    check("euclideans_packed_f32_haswell", test_euclideans_packed<f32_t>, nk_dots_packed_size_f32_haswell,
          nk_dots_pack_f32_haswell, nk_euclideans_packed_f32_haswell);
    check("euclideans_packed_f16_haswell", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_haswell,
          nk_dots_pack_f16_haswell, nk_euclideans_packed_f16_haswell);
    check("euclideans_packed_bf16_haswell", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_haswell,
          nk_dots_pack_bf16_haswell, nk_euclideans_packed_bf16_haswell);
    check("euclideans_packed_e4m3_haswell", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_haswell,
          nk_dots_pack_e4m3_haswell, nk_euclideans_packed_e4m3_haswell);
    check("euclideans_packed_e5m2_haswell", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_haswell,
          nk_dots_pack_e5m2_haswell, nk_euclideans_packed_e5m2_haswell);
    check("euclideans_packed_e2m3_haswell", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_haswell,
          nk_dots_pack_e2m3_haswell, nk_euclideans_packed_e2m3_haswell);
    check("euclideans_packed_e3m2_haswell", test_euclideans_packed<e3m2_t>, nk_dots_packed_size_e3m2_haswell,
          nk_dots_pack_e3m2_haswell, nk_euclideans_packed_e3m2_haswell);
    check("euclideans_packed_i8_haswell", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_haswell,
          nk_dots_pack_i8_haswell, nk_euclideans_packed_i8_haswell);
    check("euclideans_packed_u8_haswell", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_haswell,
          nk_dots_pack_u8_haswell, nk_euclideans_packed_u8_haswell);

    check("euclideans_symmetric_f64_haswell", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_haswell);
    check("euclideans_symmetric_f32_haswell", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_haswell);
    check("euclideans_symmetric_f16_haswell", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_haswell);
    check("euclideans_symmetric_bf16_haswell", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_haswell);
    check("euclideans_symmetric_e4m3_haswell", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_haswell);
    check("euclideans_symmetric_e5m2_haswell", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_haswell);
    check("euclideans_symmetric_e2m3_haswell", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_haswell);
    check("euclideans_symmetric_e3m2_haswell", test_euclideans_symmetric<e3m2_t>, nk_euclideans_symmetric_e3m2_haswell);
    check("euclideans_symmetric_i8_haswell", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_haswell);
    check("euclideans_symmetric_u8_haswell", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_haswell);

    check("attention_packed_bf16_haswell", test_attention_packed<bf16_t>, nk_attention_packed_size_bf16_haswell,
          nk_attention_pack_bf16_haswell, nk_attention_packed_bf16_haswell);
    check("attention_packed_e4m3_haswell", test_attention_packed<e4m3_t>, nk_attention_packed_size_e4m3_haswell,
          nk_attention_pack_e4m3_haswell, nk_attention_packed_e4m3_haswell);

#endif

#if NK_TARGET_SKYLAKE

    check("dots_packed_f64_skylake", test_dots_packed<f64_t>, nk_dots_packed_size_f64_skylake, nk_dots_pack_f64_skylake,
          nk_dots_packed_f64_skylake);
    check("dots_packed_f32_skylake", test_dots_packed<f32_t>, nk_dots_packed_size_f32_skylake, nk_dots_pack_f32_skylake,
          nk_dots_packed_f32_skylake);
    check("dots_packed_f16_skylake", test_dots_packed<f16_t>, nk_dots_packed_size_f16_skylake, nk_dots_pack_f16_skylake,
          nk_dots_packed_f16_skylake);
    check("dots_packed_bf16_skylake", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_skylake,
          nk_dots_pack_bf16_skylake, nk_dots_packed_bf16_skylake);
    check("dots_packed_e4m3_skylake", test_dots_packed<e4m3_t>, nk_dots_packed_size_e4m3_skylake,
          nk_dots_pack_e4m3_skylake, nk_dots_packed_e4m3_skylake);
    check("dots_packed_e5m2_skylake", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_skylake,
          nk_dots_pack_e5m2_skylake, nk_dots_packed_e5m2_skylake);
    check("dots_packed_e2m3_skylake", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_skylake,
          nk_dots_pack_e2m3_skylake, nk_dots_packed_e2m3_skylake);
    check("dots_packed_e3m2_skylake", test_dots_packed<e3m2_t>, nk_dots_packed_size_e3m2_skylake,
          nk_dots_pack_e3m2_skylake, nk_dots_packed_e3m2_skylake);

    check("dots_symmetric_f64_skylake", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_skylake);
    check("dots_symmetric_f32_skylake", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_skylake);
    check("dots_symmetric_f16_skylake", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_skylake);
    check("dots_symmetric_bf16_skylake", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_skylake);
    check("dots_symmetric_e4m3_skylake", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_skylake);
    check("dots_symmetric_e5m2_skylake", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_skylake);
    check("dots_symmetric_e2m3_skylake", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_skylake);
    check("dots_symmetric_e3m2_skylake", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_skylake);

    check("angulars_packed_f64_skylake", test_angulars_packed<f64_t>, nk_dots_packed_size_f64_skylake,
          nk_dots_pack_f64_skylake, nk_angulars_packed_f64_skylake);
    check("angulars_packed_f32_skylake", test_angulars_packed<f32_t>, nk_dots_packed_size_f32_skylake,
          nk_dots_pack_f32_skylake, nk_angulars_packed_f32_skylake);
    check("angulars_packed_f16_skylake", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_skylake,
          nk_dots_pack_f16_skylake, nk_angulars_packed_f16_skylake);
    check("angulars_packed_bf16_skylake", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_skylake,
          nk_dots_pack_bf16_skylake, nk_angulars_packed_bf16_skylake);
    check("angulars_packed_e4m3_skylake", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_skylake,
          nk_dots_pack_e4m3_skylake, nk_angulars_packed_e4m3_skylake);
    check("angulars_packed_e5m2_skylake", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_skylake,
          nk_dots_pack_e5m2_skylake, nk_angulars_packed_e5m2_skylake);
    check("angulars_packed_e2m3_skylake", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_skylake,
          nk_dots_pack_e2m3_skylake, nk_angulars_packed_e2m3_skylake);
    check("angulars_packed_e3m2_skylake", test_angulars_packed<e3m2_t>, nk_dots_packed_size_e3m2_skylake,
          nk_dots_pack_e3m2_skylake, nk_angulars_packed_e3m2_skylake);

    check("angulars_symmetric_f64_skylake", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_skylake);
    check("angulars_symmetric_f32_skylake", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_skylake);
    check("angulars_symmetric_f16_skylake", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_skylake);
    check("angulars_symmetric_bf16_skylake", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_skylake);
    check("angulars_symmetric_e4m3_skylake", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_skylake);
    check("angulars_symmetric_e5m2_skylake", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_skylake);
    check("angulars_symmetric_e2m3_skylake", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_skylake);
    check("angulars_symmetric_e3m2_skylake", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_skylake);

    check("euclideans_packed_f64_skylake", test_euclideans_packed<f64_t>, nk_dots_packed_size_f64_skylake,
          nk_dots_pack_f64_skylake, nk_euclideans_packed_f64_skylake);
    check("euclideans_packed_f32_skylake", test_euclideans_packed<f32_t>, nk_dots_packed_size_f32_skylake,
          nk_dots_pack_f32_skylake, nk_euclideans_packed_f32_skylake);
    check("euclideans_packed_f16_skylake", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_skylake,
          nk_dots_pack_f16_skylake, nk_euclideans_packed_f16_skylake);
    check("euclideans_packed_bf16_skylake", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_skylake,
          nk_dots_pack_bf16_skylake, nk_euclideans_packed_bf16_skylake);
    check("euclideans_packed_e4m3_skylake", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_skylake,
          nk_dots_pack_e4m3_skylake, nk_euclideans_packed_e4m3_skylake);
    check("euclideans_packed_e5m2_skylake", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_skylake,
          nk_dots_pack_e5m2_skylake, nk_euclideans_packed_e5m2_skylake);
    check("euclideans_packed_e2m3_skylake", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_skylake,
          nk_dots_pack_e2m3_skylake, nk_euclideans_packed_e2m3_skylake);
    check("euclideans_packed_e3m2_skylake", test_euclideans_packed<e3m2_t>, nk_dots_packed_size_e3m2_skylake,
          nk_dots_pack_e3m2_skylake, nk_euclideans_packed_e3m2_skylake);

    check("euclideans_symmetric_f64_skylake", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_skylake);
    check("euclideans_symmetric_f32_skylake", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_skylake);
    check("euclideans_symmetric_f16_skylake", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_skylake);
    check("euclideans_symmetric_bf16_skylake", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_skylake);
    check("euclideans_symmetric_e4m3_skylake", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_skylake);
    check("euclideans_symmetric_e5m2_skylake", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_skylake);
    check("euclideans_symmetric_e2m3_skylake", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_skylake);
    check("euclideans_symmetric_e3m2_skylake", test_euclideans_symmetric<e3m2_t>, nk_euclideans_symmetric_e3m2_skylake);

    check("attention_packed_bf16_skylake", test_attention_packed<bf16_t>, nk_attention_packed_size_bf16_skylake,
          nk_attention_pack_bf16_skylake, nk_attention_packed_bf16_skylake);
    check("attention_packed_e4m3_skylake", test_attention_packed<e4m3_t>, nk_attention_packed_size_e4m3_skylake,
          nk_attention_pack_e4m3_skylake, nk_attention_packed_e4m3_skylake);

#endif

#if NK_TARGET_ICELAKE
    check("dots_packed_i8_icelake", test_dots_packed<i8_t>, nk_dots_packed_size_i8_icelake, nk_dots_pack_i8_icelake,
          nk_dots_packed_i8_icelake);
    check("dots_packed_u8_icelake", test_dots_packed<u8_t>, nk_dots_packed_size_u8_icelake, nk_dots_pack_u8_icelake,
          nk_dots_packed_u8_icelake);
    check("dots_packed_i4_icelake", test_dots_packed<i4x2_t>, nk_dots_packed_size_i4_icelake, nk_dots_pack_i4_icelake,
          nk_dots_packed_i4_icelake);
    check("dots_packed_u4_icelake", test_dots_packed<u4x2_t>, nk_dots_packed_size_u4_icelake, nk_dots_pack_u4_icelake,
          nk_dots_packed_u4_icelake);
    check("dots_packed_u1_icelake", test_dots_packed<u1x8_t>, nk_dots_packed_size_u1_serial, nk_dots_pack_u1_serial,
          nk_dots_packed_u1_icelake);

    check("dots_symmetric_i8_icelake", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_icelake);
    check("dots_symmetric_u8_icelake", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_icelake);
    check("dots_symmetric_i4_icelake", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_icelake);
    check("dots_symmetric_u4_icelake", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_icelake);
    check("dots_symmetric_u1_icelake", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_icelake);

    check("hammings_packed_u1_icelake", test_hammings_packed<u1x8_t>, nk_dots_packed_size_u1_serial,
          nk_dots_pack_u1_serial, nk_hammings_packed_u1_icelake);
    check("hammings_symmetric_u1_icelake", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_icelake);

    check("jaccards_packed_u1_icelake", test_jaccards_packed<u1x8_t>, nk_dots_packed_size_u1_serial,
          nk_dots_pack_u1_serial, nk_jaccards_packed_u1_icelake);
    check("jaccards_symmetric_u1_icelake", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_icelake);

    check("angulars_packed_i8_icelake", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_icelake,
          nk_dots_pack_i8_icelake, nk_angulars_packed_i8_icelake);
    check("angulars_packed_u8_icelake", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_icelake,
          nk_dots_pack_u8_icelake, nk_angulars_packed_u8_icelake);
    check("angulars_packed_i4_icelake", test_angulars_packed<i4x2_t>, nk_dots_packed_size_i4_icelake,
          nk_dots_pack_i4_icelake, nk_angulars_packed_i4_icelake);
    check("angulars_packed_u4_icelake", test_angulars_packed<u4x2_t>, nk_dots_packed_size_u4_icelake,
          nk_dots_pack_u4_icelake, nk_angulars_packed_u4_icelake);

    check("angulars_symmetric_i8_icelake", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_icelake);
    check("angulars_symmetric_u8_icelake", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_icelake);
    check("angulars_symmetric_i4_icelake", test_angulars_symmetric<i4x2_t>, nk_angulars_symmetric_i4_icelake);
    check("angulars_symmetric_u4_icelake", test_angulars_symmetric<u4x2_t>, nk_angulars_symmetric_u4_icelake);

    check("euclideans_packed_i8_icelake", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_icelake,
          nk_dots_pack_i8_icelake, nk_euclideans_packed_i8_icelake);
    check("euclideans_packed_u8_icelake", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_icelake,
          nk_dots_pack_u8_icelake, nk_euclideans_packed_u8_icelake);
    check("euclideans_packed_i4_icelake", test_euclideans_packed<i4x2_t>, nk_dots_packed_size_i4_icelake,
          nk_dots_pack_i4_icelake, nk_euclideans_packed_i4_icelake);
    check("euclideans_packed_u4_icelake", test_euclideans_packed<u4x2_t>, nk_dots_packed_size_u4_icelake,
          nk_dots_pack_u4_icelake, nk_euclideans_packed_u4_icelake);

    check("euclideans_symmetric_i8_icelake", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_icelake);
    check("euclideans_symmetric_u8_icelake", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_icelake);
    check("euclideans_symmetric_i4_icelake", test_euclideans_symmetric<i4x2_t>, nk_euclideans_symmetric_i4_icelake);
    check("euclideans_symmetric_u4_icelake", test_euclideans_symmetric<u4x2_t>, nk_euclideans_symmetric_u4_icelake);
#endif

#if NK_TARGET_GENOA

    check("dots_packed_bf16_genoa", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_genoa, nk_dots_pack_bf16_genoa,
          nk_dots_packed_bf16_genoa);
    check("dots_packed_e4m3_genoa", test_dots_packed<e4m3_t>, nk_dots_packed_size_e4m3_genoa, nk_dots_pack_e4m3_genoa,
          nk_dots_packed_e4m3_genoa);
    check("dots_packed_e5m2_genoa", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_genoa, nk_dots_pack_e5m2_genoa,
          nk_dots_packed_e5m2_genoa);
    check("dots_symmetric_bf16_genoa", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_genoa);
    check("dots_symmetric_e4m3_genoa", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_genoa);
    check("dots_symmetric_e5m2_genoa", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_genoa);
    check("angulars_packed_bf16_genoa", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_genoa,
          nk_dots_pack_bf16_genoa, nk_angulars_packed_bf16_genoa);
    check("angulars_packed_e4m3_genoa", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_genoa,
          nk_dots_pack_e4m3_genoa, nk_angulars_packed_e4m3_genoa);
    check("angulars_packed_e5m2_genoa", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_genoa,
          nk_dots_pack_e5m2_genoa, nk_angulars_packed_e5m2_genoa);
    check("angulars_symmetric_bf16_genoa", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_genoa);
    check("angulars_symmetric_e4m3_genoa", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_genoa);
    check("angulars_symmetric_e5m2_genoa", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_genoa);
    check("euclideans_packed_bf16_genoa", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_genoa,
          nk_dots_pack_bf16_genoa, nk_euclideans_packed_bf16_genoa);
    check("euclideans_packed_e4m3_genoa", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_genoa,
          nk_dots_pack_e4m3_genoa, nk_euclideans_packed_e4m3_genoa);
    check("euclideans_packed_e5m2_genoa", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_genoa,
          nk_dots_pack_e5m2_genoa, nk_euclideans_packed_e5m2_genoa);
    check("euclideans_symmetric_bf16_genoa", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_genoa);
    check("euclideans_symmetric_e4m3_genoa", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_genoa);
    check("euclideans_symmetric_e5m2_genoa", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_genoa);

    check("attention_packed_bf16_genoa", test_attention_packed<bf16_t>, nk_attention_packed_size_bf16_genoa,
          nk_attention_pack_bf16_genoa, nk_attention_packed_bf16_genoa);
    check("attention_packed_e4m3_genoa", test_attention_packed<e4m3_t>, nk_attention_packed_size_e4m3_genoa,
          nk_attention_pack_e4m3_genoa, nk_attention_packed_e4m3_genoa);

#endif

#if NK_TARGET_DIAMOND
    check("dots_packed_e4m3_diamond", test_dots_packed<e4m3_t>, nk_dots_packed_size_e4m3_diamond,
          nk_dots_pack_e4m3_diamond, nk_dots_packed_e4m3_diamond);
    check("dots_packed_e5m2_diamond", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_diamond,
          nk_dots_pack_e5m2_diamond, nk_dots_packed_e5m2_diamond);
    check("dots_symmetric_e4m3_diamond", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_diamond);
    check("dots_symmetric_e5m2_diamond", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_diamond);
    check("angulars_packed_e4m3_diamond", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_diamond,
          nk_dots_pack_e4m3_diamond, nk_angulars_packed_e4m3_diamond);
    check("angulars_packed_e5m2_diamond", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_diamond,
          nk_dots_pack_e5m2_diamond, nk_angulars_packed_e5m2_diamond);
    check("angulars_symmetric_e4m3_diamond", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_diamond);
    check("angulars_symmetric_e5m2_diamond", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_diamond);
    check("euclideans_packed_e4m3_diamond", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_diamond,
          nk_dots_pack_e4m3_diamond, nk_euclideans_packed_e4m3_diamond);
    check("euclideans_packed_e5m2_diamond", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_diamond,
          nk_dots_pack_e5m2_diamond, nk_euclideans_packed_e5m2_diamond);
    check("euclideans_symmetric_e4m3_diamond", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_diamond);
    check("euclideans_symmetric_e5m2_diamond", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_diamond);
#endif

#if NK_TARGET_ALDER
    check("dots_packed_e2m3_alder", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_alder, nk_dots_pack_e2m3_alder,
          nk_dots_packed_e2m3_alder);
    check("dots_packed_i8_alder", test_dots_packed<i8_t>, nk_dots_packed_size_i8_alder, nk_dots_pack_i8_alder,
          nk_dots_packed_i8_alder);
    check("dots_packed_u8_alder", test_dots_packed<u8_t>, nk_dots_packed_size_u8_alder, nk_dots_pack_u8_alder,
          nk_dots_packed_u8_alder);

    check("dots_symmetric_e2m3_alder", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_alder);
    check("dots_symmetric_i8_alder", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_alder);
    check("dots_symmetric_u8_alder", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_alder);

    check("angulars_packed_i8_alder", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_alder, nk_dots_pack_i8_alder,
          nk_angulars_packed_i8_alder);
    check("angulars_symmetric_i8_alder", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_alder);

    check("euclideans_packed_i8_alder", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_alder,
          nk_dots_pack_i8_alder, nk_euclideans_packed_i8_alder);
    check("euclideans_symmetric_i8_alder", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_alder);

    check("angulars_packed_u8_alder", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_alder, nk_dots_pack_u8_alder,
          nk_angulars_packed_u8_alder);
    check("angulars_symmetric_u8_alder", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_alder);

    check("euclideans_packed_u8_alder", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_alder,
          nk_dots_pack_u8_alder, nk_euclideans_packed_u8_alder);
    check("euclideans_symmetric_u8_alder", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_alder);

    check("angulars_packed_e2m3_alder", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_alder,
          nk_dots_pack_e2m3_alder, nk_angulars_packed_e2m3_alder);
    check("angulars_symmetric_e2m3_alder", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_alder);

    check("euclideans_packed_e2m3_alder", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_alder,
          nk_dots_pack_e2m3_alder, nk_euclideans_packed_e2m3_alder);
    check("euclideans_symmetric_e2m3_alder", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_alder);
#endif

#if NK_TARGET_SIERRA
    check("dots_packed_e2m3_sierra", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_sierra,
          nk_dots_pack_e2m3_sierra, nk_dots_packed_e2m3_sierra);
    check("dots_packed_i8_sierra", test_dots_packed<i8_t>, nk_dots_packed_size_i8_sierra, nk_dots_pack_i8_sierra,
          nk_dots_packed_i8_sierra);
    check("dots_packed_u8_sierra", test_dots_packed<u8_t>, nk_dots_packed_size_u8_sierra, nk_dots_pack_u8_sierra,
          nk_dots_packed_u8_sierra);

    check("dots_symmetric_e2m3_sierra", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_sierra);
    check("dots_symmetric_i8_sierra", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sierra);
    check("dots_symmetric_u8_sierra", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sierra);

    check("angulars_packed_i8_sierra", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_sierra,
          nk_dots_pack_i8_sierra, nk_angulars_packed_i8_sierra);
    check("angulars_symmetric_i8_sierra", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_sierra);

    check("euclideans_packed_i8_sierra", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_sierra,
          nk_dots_pack_i8_sierra, nk_euclideans_packed_i8_sierra);
    check("euclideans_symmetric_i8_sierra", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_sierra);

    check("angulars_packed_u8_sierra", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_sierra,
          nk_dots_pack_u8_sierra, nk_angulars_packed_u8_sierra);
    check("angulars_symmetric_u8_sierra", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_sierra);

    check("euclideans_packed_u8_sierra", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_sierra,
          nk_dots_pack_u8_sierra, nk_euclideans_packed_u8_sierra);
    check("euclideans_symmetric_u8_sierra", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_sierra);

    check("angulars_packed_e2m3_sierra", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_sierra,
          nk_dots_pack_e2m3_sierra, nk_angulars_packed_e2m3_sierra);
    check("angulars_symmetric_e2m3_sierra", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_sierra);

    check("euclideans_packed_e2m3_sierra", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_sierra,
          nk_dots_pack_e2m3_sierra, nk_euclideans_packed_e2m3_sierra);
    check("euclideans_symmetric_e2m3_sierra", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_sierra);
#endif

#if NK_TARGET_SAPPHIREAMX
    check("angulars_packed_bf16_sapphireamx", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_sapphireamx,
          nk_dots_pack_bf16_sapphireamx, nk_angulars_packed_bf16_sapphireamx);
    check("angulars_packed_i8_sapphireamx", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_sapphireamx,
          nk_dots_pack_i8_sapphireamx, nk_angulars_packed_i8_sapphireamx);
    check("angulars_packed_u8_sapphireamx", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_sapphireamx,
          nk_dots_pack_u8_sapphireamx, nk_angulars_packed_u8_sapphireamx);
    check("angulars_packed_e4m3_sapphireamx", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_sapphireamx,
          nk_dots_pack_e4m3_sapphireamx, nk_angulars_packed_e4m3_sapphireamx);
    check("angulars_packed_e5m2_sapphireamx", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_sapphireamx,
          nk_dots_pack_e5m2_sapphireamx, nk_angulars_packed_e5m2_sapphireamx);
    check("angulars_packed_e2m3_sapphireamx", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_sapphireamx,
          nk_dots_pack_e2m3_sapphireamx, nk_angulars_packed_e2m3_sapphireamx);
    check("angulars_packed_e3m2_sapphireamx", test_angulars_packed<e3m2_t>, nk_dots_packed_size_e3m2_sapphireamx,
          nk_dots_pack_e3m2_sapphireamx, nk_angulars_packed_e3m2_sapphireamx);

    check("angulars_symmetric_bf16_sapphireamx", test_angulars_symmetric<bf16_t>,
          nk_angulars_symmetric_bf16_sapphireamx);
    check("angulars_symmetric_i8_sapphireamx", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_sapphireamx);
    check("angulars_symmetric_e2m3_sapphireamx", test_angulars_symmetric<e2m3_t>,
          nk_angulars_symmetric_e2m3_sapphireamx);
    check("angulars_symmetric_e3m2_sapphireamx", test_angulars_symmetric<e3m2_t>,
          nk_angulars_symmetric_e3m2_sapphireamx);
    check("angulars_symmetric_e4m3_sapphireamx", test_angulars_symmetric<e4m3_t>,
          nk_angulars_symmetric_e4m3_sapphireamx);
    check("angulars_symmetric_e5m2_sapphireamx", test_angulars_symmetric<e5m2_t>,
          nk_angulars_symmetric_e5m2_sapphireamx);
    check("angulars_symmetric_u8_sapphireamx", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_sapphireamx);

    check("euclideans_packed_bf16_sapphireamx", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_sapphireamx,
          nk_dots_pack_bf16_sapphireamx, nk_euclideans_packed_bf16_sapphireamx);
    check("euclideans_packed_i8_sapphireamx", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_sapphireamx,
          nk_dots_pack_i8_sapphireamx, nk_euclideans_packed_i8_sapphireamx);
    check("euclideans_packed_u8_sapphireamx", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_sapphireamx,
          nk_dots_pack_u8_sapphireamx, nk_euclideans_packed_u8_sapphireamx);
    check("euclideans_packed_e4m3_sapphireamx", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_sapphireamx,
          nk_dots_pack_e4m3_sapphireamx, nk_euclideans_packed_e4m3_sapphireamx);
    check("euclideans_packed_e5m2_sapphireamx", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_sapphireamx,
          nk_dots_pack_e5m2_sapphireamx, nk_euclideans_packed_e5m2_sapphireamx);
    check("euclideans_packed_e2m3_sapphireamx", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_sapphireamx,
          nk_dots_pack_e2m3_sapphireamx, nk_euclideans_packed_e2m3_sapphireamx);
    check("euclideans_packed_e3m2_sapphireamx", test_euclideans_packed<e3m2_t>, nk_dots_packed_size_e3m2_sapphireamx,
          nk_dots_pack_e3m2_sapphireamx, nk_euclideans_packed_e3m2_sapphireamx);

    check("euclideans_symmetric_bf16_sapphireamx", test_euclideans_symmetric<bf16_t>,
          nk_euclideans_symmetric_bf16_sapphireamx);
    check("euclideans_symmetric_i8_sapphireamx", test_euclideans_symmetric<i8_t>,
          nk_euclideans_symmetric_i8_sapphireamx);
    check("euclideans_symmetric_e2m3_sapphireamx", test_euclideans_symmetric<e2m3_t>,
          nk_euclideans_symmetric_e2m3_sapphireamx);
    check("euclideans_symmetric_e3m2_sapphireamx", test_euclideans_symmetric<e3m2_t>,
          nk_euclideans_symmetric_e3m2_sapphireamx);
    check("euclideans_symmetric_e4m3_sapphireamx", test_euclideans_symmetric<e4m3_t>,
          nk_euclideans_symmetric_e4m3_sapphireamx);
    check("euclideans_symmetric_e5m2_sapphireamx", test_euclideans_symmetric<e5m2_t>,
          nk_euclideans_symmetric_e5m2_sapphireamx);
    check("euclideans_symmetric_u8_sapphireamx", test_euclideans_symmetric<u8_t>,
          nk_euclideans_symmetric_u8_sapphireamx);
#endif
}
