/**
 *  @brief Batch operation tests - Arm ISA family.
 *  @file test/test_cross_arm.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  Covers NEON, NEONHALF, NEONFHM, NEONBFDOT, NEONSDOT.
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_arm() {
    [[maybe_unused]] error_stats_section_t check;
#if NK_TARGET_NEON
    check.section("Cross NEON", nk_cap_neon_k);
    check("dots_packed_f64_neon", test_dots_packed<f64_t>, nk_dots_packed_size_f64_neon, nk_dots_pack_f64_neon,
          nk_dots_packed_f64_neon);
    check("dots_packed_f32_neon", test_dots_packed<f32_t>, nk_dots_packed_size_f32_neon, nk_dots_pack_f32_neon,
          nk_dots_packed_f32_neon);
    check("dots_packed_bf16_neon", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_neon, nk_dots_pack_bf16_neon,
          nk_dots_packed_bf16_neon);
    check("dots_packed_u1_neon", test_dots_packed<u1x8_t>, nk_dots_packed_size_u1_serial, nk_dots_pack_u1_serial,
          nk_dots_packed_u1_neon);

    check("dots_symmetric_f64_neon", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_neon);
    check("dots_symmetric_f32_neon", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_neon);
    check("dots_symmetric_bf16_neon", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_neon);
    check("dots_symmetric_u1_neon", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_neon);

    check("angulars_packed_f64_neon", test_angulars_packed<f64_t>, nk_dots_packed_size_f64_neon, nk_dots_pack_f64_neon,
          nk_angulars_packed_f64_neon);
    check("angulars_packed_f32_neon", test_angulars_packed<f32_t>, nk_dots_packed_size_f32_neon, nk_dots_pack_f32_neon,
          nk_angulars_packed_f32_neon);
    check("angulars_packed_bf16_neon", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_neon,
          nk_dots_pack_bf16_neon, nk_angulars_packed_bf16_neon);

    check("angulars_symmetric_f64_neon", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_neon);
    check("angulars_symmetric_f32_neon", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_neon);
    check("angulars_symmetric_bf16_neon", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_neon);

    check("euclideans_packed_f64_neon", test_euclideans_packed<f64_t>, nk_dots_packed_size_f64_neon,
          nk_dots_pack_f64_neon, nk_euclideans_packed_f64_neon);
    check("euclideans_packed_f32_neon", test_euclideans_packed<f32_t>, nk_dots_packed_size_f32_neon,
          nk_dots_pack_f32_neon, nk_euclideans_packed_f32_neon);
    check("euclideans_packed_bf16_neon", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_neon,
          nk_dots_pack_bf16_neon, nk_euclideans_packed_bf16_neon);

    check("euclideans_symmetric_f64_neon", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_neon);
    check("euclideans_symmetric_f32_neon", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_neon);
    check("euclideans_symmetric_bf16_neon", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_neon);

    check("dots_packed_f16_neon", test_dots_packed<f16_t>, nk_dots_packed_size_f16_neon, nk_dots_pack_f16_neon,
          nk_dots_packed_f16_neon);
    check("dots_symmetric_f16_neon", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_neon);

    check("angulars_packed_f16_neon", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_neon, nk_dots_pack_f16_neon,
          nk_angulars_packed_f16_neon);
    check("angulars_symmetric_f16_neon", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_neon);

    check("euclideans_packed_f16_neon", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_neon,
          nk_dots_pack_f16_neon, nk_euclideans_packed_f16_neon);
    check("euclideans_symmetric_f16_neon", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_neon);

    check("hammings_packed_u1_neon", test_hammings_packed<u1x8_t>, nk_dots_packed_size_u1_serial,
          nk_dots_pack_u1_serial, nk_hammings_packed_u1_neon);
    check("hammings_symmetric_u1_neon", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_neon);

    check("jaccards_packed_u1_neon", test_jaccards_packed<u1x8_t>, nk_dots_packed_size_u1_serial,
          nk_dots_pack_u1_serial, nk_jaccards_packed_u1_neon);
    check("jaccards_symmetric_u1_neon", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_neon);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONBFDOT
    check.section("Cross NEON BF16", nk_cap_neonbfdot_k);
    check("dots_packed_bf16_neonbfdot", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_neonbfdot,
          nk_dots_pack_bf16_neonbfdot, nk_dots_packed_bf16_neonbfdot);
    check("dots_symmetric_bf16_neonbfdot", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_neonbfdot);

    check("angulars_packed_bf16_neonbfdot", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_neonbfdot,
          nk_dots_pack_bf16_neonbfdot, nk_angulars_packed_bf16_neonbfdot);
    check("angulars_symmetric_bf16_neonbfdot", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_neonbfdot);

    check("euclideans_packed_bf16_neonbfdot", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_neonbfdot,
          nk_dots_pack_bf16_neonbfdot, nk_euclideans_packed_bf16_neonbfdot);
    check("euclideans_symmetric_bf16_neonbfdot", test_euclideans_symmetric<bf16_t>,
          nk_euclideans_symmetric_bf16_neonbfdot);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONFHM
    check.section("Cross NEON FHM", nk_cap_neonfhm_k);
    check("dots_packed_f16_neonfhm", test_dots_packed<f16_t>, nk_dots_packed_size_f16_neonfhm, nk_dots_pack_f16_neonfhm,
          nk_dots_packed_f16_neonfhm);
    check("dots_symmetric_f16_neonfhm", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_neonfhm);

    check("angulars_packed_f16_neonfhm", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_neonfhm,
          nk_dots_pack_f16_neonfhm, nk_angulars_packed_f16_neonfhm);
    check("angulars_packed_e5m2_neonfhm", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_neonfhm,
          nk_dots_pack_e5m2_neonfhm, nk_angulars_packed_e5m2_neonfhm);
    check("angulars_packed_e4m3_neonfhm", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_neonfhm,
          nk_dots_pack_e4m3_neonfhm, nk_angulars_packed_e4m3_neonfhm);

    check("angulars_symmetric_f16_neonfhm", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_neonfhm);
    check("angulars_symmetric_e5m2_neonfhm", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_neonfhm);
    check("angulars_symmetric_e4m3_neonfhm", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_neonfhm);

    check("euclideans_packed_f16_neonfhm", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_neonfhm,
          nk_dots_pack_f16_neonfhm, nk_euclideans_packed_f16_neonfhm);
    check("euclideans_packed_e5m2_neonfhm", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_neonfhm,
          nk_dots_pack_e5m2_neonfhm, nk_euclideans_packed_e5m2_neonfhm);
    check("euclideans_packed_e4m3_neonfhm", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_neonfhm,
          nk_dots_pack_e4m3_neonfhm, nk_euclideans_packed_e4m3_neonfhm);

    check("euclideans_symmetric_f16_neonfhm", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_neonfhm);
    check("euclideans_symmetric_e5m2_neonfhm", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_neonfhm);
    check("euclideans_symmetric_e4m3_neonfhm", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_neonfhm);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_NEONSDOT
    check.section("Cross NEON I8", nk_cap_neonsdot_k);
    check("dots_packed_i8_neonsdot", test_dots_packed<i8_t>, nk_dots_packed_size_i8_neonsdot, nk_dots_pack_i8_neonsdot,
          nk_dots_packed_i8_neonsdot);
    check("dots_packed_u8_neonsdot", test_dots_packed<u8_t>, nk_dots_packed_size_u8_neonsdot, nk_dots_pack_u8_neonsdot,
          nk_dots_packed_u8_neonsdot);
    check("dots_packed_i4_neonsdot", test_dots_packed<i4x2_t>, nk_dots_packed_size_i4_neonsdot,
          nk_dots_pack_i4_neonsdot, nk_dots_packed_i4_neonsdot);
    check("dots_packed_u4_neonsdot", test_dots_packed<u4x2_t>, nk_dots_packed_size_u4_neonsdot,
          nk_dots_pack_u4_neonsdot, nk_dots_packed_u4_neonsdot);

    check("dots_symmetric_i8_neonsdot", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_neonsdot);
    check("dots_symmetric_u8_neonsdot", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_neonsdot);
    check("dots_symmetric_i4_neonsdot", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_neonsdot);
    check("dots_symmetric_u4_neonsdot", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_neonsdot);

    check("angulars_packed_i8_neonsdot", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_neonsdot,
          nk_dots_pack_i8_neonsdot, nk_angulars_packed_i8_neonsdot);
    check("angulars_packed_u8_neonsdot", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_neonsdot,
          nk_dots_pack_u8_neonsdot, nk_angulars_packed_u8_neonsdot);
    check("angulars_packed_i4_neonsdot", test_angulars_packed<i4x2_t>, nk_dots_packed_size_i4_neonsdot,
          nk_dots_pack_i4_neonsdot, nk_angulars_packed_i4_neonsdot);
    check("angulars_packed_u4_neonsdot", test_angulars_packed<u4x2_t>, nk_dots_packed_size_u4_neonsdot,
          nk_dots_pack_u4_neonsdot, nk_angulars_packed_u4_neonsdot);

    check("angulars_symmetric_i8_neonsdot", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_neonsdot);
    check("angulars_symmetric_u8_neonsdot", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_neonsdot);
    check("angulars_symmetric_i4_neonsdot", test_angulars_symmetric<i4x2_t>, nk_angulars_symmetric_i4_neonsdot);
    check("angulars_symmetric_u4_neonsdot", test_angulars_symmetric<u4x2_t>, nk_angulars_symmetric_u4_neonsdot);

    check("euclideans_packed_i8_neonsdot", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_neonsdot,
          nk_dots_pack_i8_neonsdot, nk_euclideans_packed_i8_neonsdot);
    check("euclideans_packed_u8_neonsdot", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_neonsdot,
          nk_dots_pack_u8_neonsdot, nk_euclideans_packed_u8_neonsdot);
    check("euclideans_packed_i4_neonsdot", test_euclideans_packed<i4x2_t>, nk_dots_packed_size_i4_neonsdot,
          nk_dots_pack_i4_neonsdot, nk_euclideans_packed_i4_neonsdot);
    check("euclideans_packed_u4_neonsdot", test_euclideans_packed<u4x2_t>, nk_dots_packed_size_u4_neonsdot,
          nk_dots_pack_u4_neonsdot, nk_euclideans_packed_u4_neonsdot);

    check("euclideans_symmetric_i8_neonsdot", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_neonsdot);
    check("euclideans_symmetric_u8_neonsdot", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_neonsdot);
    check("euclideans_symmetric_i4_neonsdot", test_euclideans_symmetric<i4x2_t>, nk_euclideans_symmetric_i4_neonsdot);
    check("euclideans_symmetric_u4_neonsdot", test_euclideans_symmetric<u4x2_t>, nk_euclideans_symmetric_u4_neonsdot);

    check("dots_packed_e2m3_neonsdot", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_neonsdot,
          nk_dots_pack_e2m3_neonsdot, nk_dots_packed_e2m3_neonsdot);
    check("dots_packed_e3m2_neonsdot", test_dots_packed<e3m2_t>, nk_dots_packed_size_e3m2_neonsdot,
          nk_dots_pack_e3m2_neonsdot, nk_dots_packed_e3m2_neonsdot);
    check("dots_symmetric_e2m3_neonsdot", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_neonsdot);
    check("dots_symmetric_e3m2_neonsdot", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_neonsdot);

    check("angulars_packed_e2m3_neonsdot", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_neonsdot,
          nk_dots_pack_e2m3_neonsdot, nk_angulars_packed_e2m3_neonsdot);
    check("angulars_packed_e3m2_neonsdot", test_angulars_packed<e3m2_t>, nk_dots_packed_size_e3m2_neonsdot,
          nk_dots_pack_e3m2_neonsdot, nk_angulars_packed_e3m2_neonsdot);
    check("angulars_symmetric_e2m3_neonsdot", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_neonsdot);
    check("angulars_symmetric_e3m2_neonsdot", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_neonsdot);

    check("euclideans_packed_e2m3_neonsdot", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_neonsdot,
          nk_dots_pack_e2m3_neonsdot, nk_euclideans_packed_e2m3_neonsdot);
    check("euclideans_packed_e3m2_neonsdot", test_euclideans_packed<e3m2_t>, nk_dots_packed_size_e3m2_neonsdot,
          nk_dots_pack_e3m2_neonsdot, nk_euclideans_packed_e3m2_neonsdot);
    check("euclideans_symmetric_e2m3_neonsdot", test_euclideans_symmetric<e2m3_t>,
          nk_euclideans_symmetric_e2m3_neonsdot);
    check("euclideans_symmetric_e3m2_neonsdot", test_euclideans_symmetric<e3m2_t>,
          nk_euclideans_symmetric_e3m2_neonsdot);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_NEONFP8
    check.section("Cross NEON FP8", nk_cap_neonfp8_k);
    check("dots_packed_e4m3_neonfp8", test_dots_packed<e4m3_t>, nk_dots_packed_size_e4m3_neonfp8,
          nk_dots_pack_e4m3_neonfp8, nk_dots_packed_e4m3_neonfp8);
    check("dots_packed_e5m2_neonfp8", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_neonfp8,
          nk_dots_pack_e5m2_neonfp8, nk_dots_packed_e5m2_neonfp8);
    check("dots_packed_e2m3_neonfp8", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_neonfp8,
          nk_dots_pack_e2m3_neonfp8, nk_dots_packed_e2m3_neonfp8);
    check("dots_packed_e3m2_neonfp8", test_dots_packed<e3m2_t>, nk_dots_packed_size_e3m2_neonfp8,
          nk_dots_pack_e3m2_neonfp8, nk_dots_packed_e3m2_neonfp8);

    check("dots_symmetric_e4m3_neonfp8", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_neonfp8);
    check("dots_symmetric_e5m2_neonfp8", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_neonfp8);
    check("dots_symmetric_e2m3_neonfp8", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_neonfp8);
    check("dots_symmetric_e3m2_neonfp8", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_neonfp8);

    check("angulars_packed_e4m3_neonfp8", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_neonfp8,
          nk_dots_pack_e4m3_neonfp8, nk_angulars_packed_e4m3_neonfp8);
    check("angulars_packed_e5m2_neonfp8", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_neonfp8,
          nk_dots_pack_e5m2_neonfp8, nk_angulars_packed_e5m2_neonfp8);
    check("angulars_packed_e2m3_neonfp8", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_neonfp8,
          nk_dots_pack_e2m3_neonfp8, nk_angulars_packed_e2m3_neonfp8);
    check("angulars_packed_e3m2_neonfp8", test_angulars_packed<e3m2_t>, nk_dots_packed_size_e3m2_neonfp8,
          nk_dots_pack_e3m2_neonfp8, nk_angulars_packed_e3m2_neonfp8);

    check("angulars_symmetric_e4m3_neonfp8", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_neonfp8);
    check("angulars_symmetric_e5m2_neonfp8", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_neonfp8);
    check("angulars_symmetric_e2m3_neonfp8", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_neonfp8);
    check("angulars_symmetric_e3m2_neonfp8", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_neonfp8);

    check("euclideans_packed_e4m3_neonfp8", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_neonfp8,
          nk_dots_pack_e4m3_neonfp8, nk_euclideans_packed_e4m3_neonfp8);
    check("euclideans_packed_e5m2_neonfp8", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_neonfp8,
          nk_dots_pack_e5m2_neonfp8, nk_euclideans_packed_e5m2_neonfp8);
    check("euclideans_packed_e2m3_neonfp8", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_neonfp8,
          nk_dots_pack_e2m3_neonfp8, nk_euclideans_packed_e2m3_neonfp8);
    check("euclideans_packed_e3m2_neonfp8", test_euclideans_packed<e3m2_t>, nk_dots_packed_size_e3m2_neonfp8,
          nk_dots_pack_e3m2_neonfp8, nk_euclideans_packed_e3m2_neonfp8);

    check("euclideans_symmetric_e4m3_neonfp8", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_neonfp8);
    check("euclideans_symmetric_e5m2_neonfp8", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_neonfp8);
    check("euclideans_symmetric_e2m3_neonfp8", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_neonfp8);
    check("euclideans_symmetric_e3m2_neonfp8", test_euclideans_symmetric<e3m2_t>, nk_euclideans_symmetric_e3m2_neonfp8);
#endif // NK_TARGET_NEONFP8
}
