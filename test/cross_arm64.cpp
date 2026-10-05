/**
 *  @file test/cross_arm64.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Batch operation tests - Arm64 ISA family.
 *
 *  Covers NEON, NEONBFDOT, NEONFHM, NEONSDOT, NEONFP8, SME, SMEBI32, SMEF64.
 */
#include "harness.hpp"
#include "cross.hpp"

namespace ashvardanian::numkong::test {

void test_cross_arm64([[maybe_unused]] error_stats_section_t &check) {
#if NUMKONG_TARGET_NEON
    check.section("Cross NEON", nk_cap_neon_k);
    check("dots_packed_f64_neon", test_dots_packed<f64_t>, nk_dots_pack_size_f64_neon, nk_dots_pack_f64_neon,
          nk_dots_packed_f64_neon);
    check("dots_pack_f64_neon", test_dots_pack_layout<f64_t, host_backend_t, nk_dots_pack_size_f64_neon,
                                                      nk_dots_packed_shape_f64_neon, nk_dots_pack_f64_neon>);
    check("dots_packed_f32_neon", test_dots_packed<f32_t>, nk_dots_pack_size_f32_neon, nk_dots_pack_f32_neon,
          nk_dots_packed_f32_neon);
    check("dots_pack_f32_neon", test_dots_pack_layout<f32_t, host_backend_t, nk_dots_pack_size_f32_neon,
                                                      nk_dots_packed_shape_f32_neon, nk_dots_pack_f32_neon>);
    check("dots_packed_bf16_neon", test_dots_packed<bf16_t>, nk_dots_pack_size_bf16_neon, nk_dots_pack_bf16_neon,
          nk_dots_packed_bf16_neon);
    check("dots_pack_bf16_neon", test_dots_pack_layout<bf16_t, host_backend_t, nk_dots_pack_size_bf16_neon,
                                                       nk_dots_packed_shape_bf16_neon, nk_dots_pack_bf16_neon>);
    check("dots_packed_u1_neon", test_dots_packed<u1x8_t>, nk_dots_pack_size_u1_neon, nk_dots_pack_u1_neon,
          nk_dots_packed_u1_neon);

    check("dots_symmetric_f64_neon", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_neon);
    check("dots_symmetric_f32_neon", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_neon);
    check("dots_symmetric_bf16_neon", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_neon);
    check("dots_symmetric_u1_neon", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_neon);

    check("angulars_packed_f64_neon", test_angulars_packed<f64_t>, nk_dots_pack_size_f64_neon, nk_dots_pack_f64_neon,
          nk_angulars_packed_f64_neon);
    check("angulars_packed_f32_neon", test_angulars_packed<f32_t>, nk_dots_pack_size_f32_neon, nk_dots_pack_f32_neon,
          nk_angulars_packed_f32_neon);
    check("angulars_packed_bf16_neon", test_angulars_packed<bf16_t>, nk_dots_pack_size_bf16_neon,
          nk_dots_pack_bf16_neon, nk_angulars_packed_bf16_neon);

    check("angulars_symmetric_f64_neon", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_neon);
    check("angulars_symmetric_f32_neon", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_neon);
    check("angulars_symmetric_bf16_neon", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_neon);

    check("euclideans_packed_f64_neon", test_euclideans_packed<f64_t>, nk_dots_pack_size_f64_neon,
          nk_dots_pack_f64_neon, nk_euclideans_packed_f64_neon);
    check("euclideans_packed_f32_neon", test_euclideans_packed<f32_t>, nk_dots_pack_size_f32_neon,
          nk_dots_pack_f32_neon, nk_euclideans_packed_f32_neon);
    check("euclideans_packed_bf16_neon", test_euclideans_packed<bf16_t>, nk_dots_pack_size_bf16_neon,
          nk_dots_pack_bf16_neon, nk_euclideans_packed_bf16_neon);

    check("euclideans_symmetric_f64_neon", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_neon);
    check("euclideans_symmetric_f32_neon", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_neon);
    check("euclideans_symmetric_bf16_neon", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_neon);

    check("dots_packed_f16_neon", test_dots_packed<f16_t>, nk_dots_pack_size_f16_neon, nk_dots_pack_f16_neon,
          nk_dots_packed_f16_neon);
    check("dots_pack_f16_neon", test_dots_pack_layout<f16_t, host_backend_t, nk_dots_pack_size_f16_neon,
                                                      nk_dots_packed_shape_f16_neon, nk_dots_pack_f16_neon>);
    check("dots_symmetric_f16_neon", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_neon);

    check("angulars_packed_f16_neon", test_angulars_packed<f16_t>, nk_dots_pack_size_f16_neon, nk_dots_pack_f16_neon,
          nk_angulars_packed_f16_neon);
    check("angulars_symmetric_f16_neon", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_neon);

    check("euclideans_packed_f16_neon", test_euclideans_packed<f16_t>, nk_dots_pack_size_f16_neon,
          nk_dots_pack_f16_neon, nk_euclideans_packed_f16_neon);
    check("euclideans_symmetric_f16_neon", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_neon);

    check("hammings_packed_u1_neon", test_hammings_packed<u1x8_t>, nk_dots_pack_size_u1_neon, nk_dots_pack_u1_neon,
          nk_hammings_packed_u1_neon);
    check("hammings_symmetric_u1_neon", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_neon);

    check("jaccards_packed_u1_neon", test_jaccards_packed<u1x8_t>, nk_dots_pack_size_u1_neon, nk_dots_pack_u1_neon,
          nk_jaccards_packed_u1_neon);
    check("jaccards_symmetric_u1_neon", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_neon);
    check("dots_packed_nvfp4_neon", test_dots_packed<nvfp4_t>, nk_dots_pack_size_nvfp4_neon, nk_dots_pack_nvfp4_neon,
          nk_dots_packed_nvfp4_neon);
    check("dots_symmetric_nvfp4_neon", test_dots_symmetric<nvfp4_t>, nk_dots_symmetric_nvfp4_neon);
    check("angulars_packed_nvfp4_neon", test_angulars_packed<nvfp4_t>, nk_dots_pack_size_nvfp4_neon,
          nk_dots_pack_nvfp4_neon, nk_angulars_packed_nvfp4_neon);
    check("angulars_symmetric_nvfp4_neon", test_angulars_symmetric<nvfp4_t>, nk_angulars_symmetric_nvfp4_neon);
    check("euclideans_packed_nvfp4_neon", test_euclideans_packed<nvfp4_t>, nk_dots_pack_size_nvfp4_neon,
          nk_dots_pack_nvfp4_neon, nk_euclideans_packed_nvfp4_neon);
    check("euclideans_symmetric_nvfp4_neon", test_euclideans_symmetric<nvfp4_t>, nk_euclideans_symmetric_nvfp4_neon);
    check("dots_packed_mxfp4_neon", test_dots_packed<mxfp4_t>, nk_dots_pack_size_mxfp4_neon, nk_dots_pack_mxfp4_neon,
          nk_dots_packed_mxfp4_neon);
    check("dots_symmetric_mxfp4_neon", test_dots_symmetric<mxfp4_t>, nk_dots_symmetric_mxfp4_neon);
    check("angulars_packed_mxfp4_neon", test_angulars_packed<mxfp4_t>, nk_dots_pack_size_mxfp4_neon,
          nk_dots_pack_mxfp4_neon, nk_angulars_packed_mxfp4_neon);
    check("angulars_symmetric_mxfp4_neon", test_angulars_symmetric<mxfp4_t>, nk_angulars_symmetric_mxfp4_neon);
    check("euclideans_packed_mxfp4_neon", test_euclideans_packed<mxfp4_t>, nk_dots_pack_size_mxfp4_neon,
          nk_dots_pack_mxfp4_neon, nk_euclideans_packed_mxfp4_neon);
    check("euclideans_symmetric_mxfp4_neon", test_euclideans_symmetric<mxfp4_t>, nk_euclideans_symmetric_mxfp4_neon);
    check("dots_packed_mxfp6e2m3_neon", test_dots_packed<mxfp6e2m3_t>, nk_dots_pack_size_mxfp6e2m3_neon,
          nk_dots_pack_mxfp6e2m3_neon, nk_dots_packed_mxfp6e2m3_neon);
    check("dots_symmetric_mxfp6e2m3_neon", test_dots_symmetric<mxfp6e2m3_t>, nk_dots_symmetric_mxfp6e2m3_neon);
    check("angulars_packed_mxfp6e2m3_neon", test_angulars_packed<mxfp6e2m3_t>, nk_dots_pack_size_mxfp6e2m3_neon,
          nk_dots_pack_mxfp6e2m3_neon, nk_angulars_packed_mxfp6e2m3_neon);
    check("angulars_symmetric_mxfp6e2m3_neon", test_angulars_symmetric<mxfp6e2m3_t>,
          nk_angulars_symmetric_mxfp6e2m3_neon);
    check("euclideans_packed_mxfp6e2m3_neon", test_euclideans_packed<mxfp6e2m3_t>, nk_dots_pack_size_mxfp6e2m3_neon,
          nk_dots_pack_mxfp6e2m3_neon, nk_euclideans_packed_mxfp6e2m3_neon);
    check("euclideans_symmetric_mxfp6e2m3_neon", test_euclideans_symmetric<mxfp6e2m3_t>,
          nk_euclideans_symmetric_mxfp6e2m3_neon);
    check("dots_packed_mxfp6e3m2_neon", test_dots_packed<mxfp6e3m2_t>, nk_dots_pack_size_mxfp6e3m2_neon,
          nk_dots_pack_mxfp6e3m2_neon, nk_dots_packed_mxfp6e3m2_neon);
    check("dots_symmetric_mxfp6e3m2_neon", test_dots_symmetric<mxfp6e3m2_t>, nk_dots_symmetric_mxfp6e3m2_neon);
    check("angulars_packed_mxfp6e3m2_neon", test_angulars_packed<mxfp6e3m2_t>, nk_dots_pack_size_mxfp6e3m2_neon,
          nk_dots_pack_mxfp6e3m2_neon, nk_angulars_packed_mxfp6e3m2_neon);
    check("angulars_symmetric_mxfp6e3m2_neon", test_angulars_symmetric<mxfp6e3m2_t>,
          nk_angulars_symmetric_mxfp6e3m2_neon);
    check("euclideans_packed_mxfp6e3m2_neon", test_euclideans_packed<mxfp6e3m2_t>, nk_dots_pack_size_mxfp6e3m2_neon,
          nk_dots_pack_mxfp6e3m2_neon, nk_euclideans_packed_mxfp6e3m2_neon);
    check("euclideans_symmetric_mxfp6e3m2_neon", test_euclideans_symmetric<mxfp6e3m2_t>,
          nk_euclideans_symmetric_mxfp6e3m2_neon);
    check("dots_packed_mxfp8e4m3_neon", test_dots_packed<mxfp8e4m3_t>, nk_dots_pack_size_mxfp8e4m3_neon,
          nk_dots_pack_mxfp8e4m3_neon, nk_dots_packed_mxfp8e4m3_neon);
    check("dots_symmetric_mxfp8e4m3_neon", test_dots_symmetric<mxfp8e4m3_t>, nk_dots_symmetric_mxfp8e4m3_neon);
    check("angulars_packed_mxfp8e4m3_neon", test_angulars_packed<mxfp8e4m3_t>, nk_dots_pack_size_mxfp8e4m3_neon,
          nk_dots_pack_mxfp8e4m3_neon, nk_angulars_packed_mxfp8e4m3_neon);
    check("angulars_symmetric_mxfp8e4m3_neon", test_angulars_symmetric<mxfp8e4m3_t>,
          nk_angulars_symmetric_mxfp8e4m3_neon);
    check("euclideans_packed_mxfp8e4m3_neon", test_euclideans_packed<mxfp8e4m3_t>, nk_dots_pack_size_mxfp8e4m3_neon,
          nk_dots_pack_mxfp8e4m3_neon, nk_euclideans_packed_mxfp8e4m3_neon);
    check("euclideans_symmetric_mxfp8e4m3_neon", test_euclideans_symmetric<mxfp8e4m3_t>,
          nk_euclideans_symmetric_mxfp8e4m3_neon);
    check("dots_packed_mxfp8e5m2_neon", test_dots_packed<mxfp8e5m2_t>, nk_dots_pack_size_mxfp8e5m2_neon,
          nk_dots_pack_mxfp8e5m2_neon, nk_dots_packed_mxfp8e5m2_neon);
    check("dots_symmetric_mxfp8e5m2_neon", test_dots_symmetric<mxfp8e5m2_t>, nk_dots_symmetric_mxfp8e5m2_neon);
    check("angulars_packed_mxfp8e5m2_neon", test_angulars_packed<mxfp8e5m2_t>, nk_dots_pack_size_mxfp8e5m2_neon,
          nk_dots_pack_mxfp8e5m2_neon, nk_angulars_packed_mxfp8e5m2_neon);
    check("angulars_symmetric_mxfp8e5m2_neon", test_angulars_symmetric<mxfp8e5m2_t>,
          nk_angulars_symmetric_mxfp8e5m2_neon);
    check("euclideans_packed_mxfp8e5m2_neon", test_euclideans_packed<mxfp8e5m2_t>, nk_dots_pack_size_mxfp8e5m2_neon,
          nk_dots_pack_mxfp8e5m2_neon, nk_euclideans_packed_mxfp8e5m2_neon);
    check("euclideans_symmetric_mxfp8e5m2_neon", test_euclideans_symmetric<mxfp8e5m2_t>,
          nk_euclideans_symmetric_mxfp8e5m2_neon);

#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
    check.section("Cross NEON BF16", nk_cap_neonbfdot_k);
    check("dots_packed_bf16_neonbfdot", test_dots_packed<bf16_t>, nk_dots_pack_size_bf16_neonbfdot,
          nk_dots_pack_bf16_neonbfdot, nk_dots_packed_bf16_neonbfdot);
    check("dots_pack_bf16_neonbfdot",
          test_dots_pack_layout<bf16_t, host_backend_t, nk_dots_pack_size_bf16_neonbfdot,
                                nk_dots_packed_shape_bf16_neonbfdot, nk_dots_pack_bf16_neonbfdot>);
    check("dots_symmetric_bf16_neonbfdot", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_neonbfdot);

    check("angulars_packed_bf16_neonbfdot", test_angulars_packed<bf16_t>, nk_dots_pack_size_bf16_neonbfdot,
          nk_dots_pack_bf16_neonbfdot, nk_angulars_packed_bf16_neonbfdot);
    check("angulars_symmetric_bf16_neonbfdot", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_neonbfdot);

    check("euclideans_packed_bf16_neonbfdot", test_euclideans_packed<bf16_t>, nk_dots_pack_size_bf16_neonbfdot,
          nk_dots_pack_bf16_neonbfdot, nk_euclideans_packed_bf16_neonbfdot);
    check("euclideans_symmetric_bf16_neonbfdot", test_euclideans_symmetric<bf16_t>,
          nk_euclideans_symmetric_bf16_neonbfdot);

    check("attention_bidirectional_packed_bf16_neonbfdot", test_attention_bidirectional_packed<bf16_t>,
          nk_attention_pack_size_bf16_neonbfdot, nk_attention_pack_bf16_neonbfdot,
          nk_attention_bidirectional_packed_bf16_neonbfdot);
    check("attention_causal_packed_bf16_neonbfdot", test_attention_causal_packed<bf16_t>,
          nk_attention_pack_size_bf16_neonbfdot, nk_attention_pack_bf16_neonbfdot,
          nk_attention_causal_packed_bf16_neonbfdot);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONFHM
    check.section("Cross NEON FHM", nk_cap_neonfhm_k);
    check("dots_packed_f16_neonfhm", test_dots_packed<f16_t>, nk_dots_pack_size_f16_neonfhm, nk_dots_pack_f16_neonfhm,
          nk_dots_packed_f16_neonfhm);
    check("dots_pack_f16_neonfhm", test_dots_pack_layout<f16_t, host_backend_t, nk_dots_pack_size_f16_neonfhm,
                                                         nk_dots_packed_shape_f16_neonfhm, nk_dots_pack_f16_neonfhm>);
    check("dots_symmetric_f16_neonfhm", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_neonfhm);

    check("angulars_packed_f16_neonfhm", test_angulars_packed<f16_t>, nk_dots_pack_size_f16_neonfhm,
          nk_dots_pack_f16_neonfhm, nk_angulars_packed_f16_neonfhm);
    check("angulars_packed_e5m2_neonfhm", test_angulars_packed<e5m2_t>, nk_dots_pack_size_e5m2_neonfhm,
          nk_dots_pack_e5m2_neonfhm, nk_angulars_packed_e5m2_neonfhm);
    check("angulars_packed_e4m3_neonfhm", test_angulars_packed<e4m3_t>, nk_dots_pack_size_e4m3_neonfhm,
          nk_dots_pack_e4m3_neonfhm, nk_angulars_packed_e4m3_neonfhm);

    check("angulars_symmetric_f16_neonfhm", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_neonfhm);
    check("angulars_symmetric_e5m2_neonfhm", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_neonfhm);
    check("angulars_symmetric_e4m3_neonfhm", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_neonfhm);

    check("euclideans_packed_f16_neonfhm", test_euclideans_packed<f16_t>, nk_dots_pack_size_f16_neonfhm,
          nk_dots_pack_f16_neonfhm, nk_euclideans_packed_f16_neonfhm);
    check("euclideans_packed_e5m2_neonfhm", test_euclideans_packed<e5m2_t>, nk_dots_pack_size_e5m2_neonfhm,
          nk_dots_pack_e5m2_neonfhm, nk_euclideans_packed_e5m2_neonfhm);
    check("euclideans_packed_e4m3_neonfhm", test_euclideans_packed<e4m3_t>, nk_dots_pack_size_e4m3_neonfhm,
          nk_dots_pack_e4m3_neonfhm, nk_euclideans_packed_e4m3_neonfhm);

    check("euclideans_symmetric_f16_neonfhm", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_neonfhm);
    check("euclideans_symmetric_e5m2_neonfhm", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_neonfhm);
    check("euclideans_symmetric_e4m3_neonfhm", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_neonfhm);

    check("attention_bidirectional_packed_e4m3_neonfhm", test_attention_bidirectional_packed<e4m3_t>,
          nk_attention_pack_size_e4m3_neonfhm, nk_attention_pack_e4m3_neonfhm,
          nk_attention_bidirectional_packed_e4m3_neonfhm);
    check("attention_causal_packed_e4m3_neonfhm", test_attention_causal_packed<e4m3_t>,
          nk_attention_pack_size_e4m3_neonfhm, nk_attention_pack_e4m3_neonfhm, nk_attention_causal_packed_e4m3_neonfhm);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONSDOT
    check.section("Cross NEON I8", nk_cap_neonsdot_k);
    check("dots_packed_i8_neonsdot", test_dots_packed<i8_t>, nk_dots_pack_size_i8_neonsdot, nk_dots_pack_i8_neonsdot,
          nk_dots_packed_i8_neonsdot);
    check("dots_pack_i8_neonsdot", test_dots_pack_layout<i8_t, host_backend_t, nk_dots_pack_size_i8_neonsdot,
                                                         nk_dots_packed_shape_i8_neonsdot, nk_dots_pack_i8_neonsdot>);
    check("dots_packed_i4_neonsdot", test_dots_packed<i4x2_t>, nk_dots_pack_size_i4_neonsdot, nk_dots_pack_i4_neonsdot,
          nk_dots_packed_i4_neonsdot);
    check("dots_pack_i4_neonsdot", test_dots_pack_layout<i4x2_t, host_backend_t, nk_dots_pack_size_i4_neonsdot,
                                                         nk_dots_packed_shape_i4_neonsdot, nk_dots_pack_i4_neonsdot>);
    check("dots_packed_u8_neonsdot", test_dots_packed<u8_t>, nk_dots_pack_size_u8_neonsdot, nk_dots_pack_u8_neonsdot,
          nk_dots_packed_u8_neonsdot);
    check("dots_pack_u8_neonsdot", test_dots_pack_layout<u8_t, host_backend_t, nk_dots_pack_size_u8_neonsdot,
                                                         nk_dots_packed_shape_u8_neonsdot, nk_dots_pack_u8_neonsdot>);
    check("dots_packed_u4_neonsdot", test_dots_packed<u4x2_t>, nk_dots_pack_size_u4_neonsdot, nk_dots_pack_u4_neonsdot,
          nk_dots_packed_u4_neonsdot);
    check("dots_pack_u4_neonsdot", test_dots_pack_layout<u4x2_t, host_backend_t, nk_dots_pack_size_u4_neonsdot,
                                                         nk_dots_packed_shape_u4_neonsdot, nk_dots_pack_u4_neonsdot>);

    check("dots_symmetric_i8_neonsdot", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_neonsdot);
    check("dots_symmetric_i4_neonsdot", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_neonsdot);
    check("dots_symmetric_u8_neonsdot", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_neonsdot);
    check("dots_symmetric_u4_neonsdot", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_neonsdot);

    check("angulars_packed_i8_neonsdot", test_angulars_packed<i8_t>, nk_dots_pack_size_i8_neonsdot,
          nk_dots_pack_i8_neonsdot, nk_angulars_packed_i8_neonsdot);
    check("angulars_packed_i4_neonsdot", test_angulars_packed<i4x2_t>, nk_dots_pack_size_i4_neonsdot,
          nk_dots_pack_i4_neonsdot, nk_angulars_packed_i4_neonsdot);
    check("angulars_packed_u8_neonsdot", test_angulars_packed<u8_t>, nk_dots_pack_size_u8_neonsdot,
          nk_dots_pack_u8_neonsdot, nk_angulars_packed_u8_neonsdot);
    check("angulars_packed_u4_neonsdot", test_angulars_packed<u4x2_t>, nk_dots_pack_size_u4_neonsdot,
          nk_dots_pack_u4_neonsdot, nk_angulars_packed_u4_neonsdot);

    check("angulars_symmetric_i8_neonsdot", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_neonsdot);
    check("angulars_symmetric_i4_neonsdot", test_angulars_symmetric<i4x2_t>, nk_angulars_symmetric_i4_neonsdot);
    check("angulars_symmetric_u8_neonsdot", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_neonsdot);
    check("angulars_symmetric_u4_neonsdot", test_angulars_symmetric<u4x2_t>, nk_angulars_symmetric_u4_neonsdot);

    check("euclideans_packed_i8_neonsdot", test_euclideans_packed<i8_t>, nk_dots_pack_size_i8_neonsdot,
          nk_dots_pack_i8_neonsdot, nk_euclideans_packed_i8_neonsdot);
    check("euclideans_packed_i4_neonsdot", test_euclideans_packed<i4x2_t>, nk_dots_pack_size_i4_neonsdot,
          nk_dots_pack_i4_neonsdot, nk_euclideans_packed_i4_neonsdot);
    check("euclideans_packed_u8_neonsdot", test_euclideans_packed<u8_t>, nk_dots_pack_size_u8_neonsdot,
          nk_dots_pack_u8_neonsdot, nk_euclideans_packed_u8_neonsdot);
    check("euclideans_packed_u4_neonsdot", test_euclideans_packed<u4x2_t>, nk_dots_pack_size_u4_neonsdot,
          nk_dots_pack_u4_neonsdot, nk_euclideans_packed_u4_neonsdot);

    check("euclideans_symmetric_i8_neonsdot", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_neonsdot);
    check("euclideans_symmetric_i4_neonsdot", test_euclideans_symmetric<i4x2_t>, nk_euclideans_symmetric_i4_neonsdot);
    check("euclideans_symmetric_u8_neonsdot", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_neonsdot);
    check("euclideans_symmetric_u4_neonsdot", test_euclideans_symmetric<u4x2_t>, nk_euclideans_symmetric_u4_neonsdot);

    check("dots_packed_e3m2_neonsdot", test_dots_packed<e3m2_t>, nk_dots_pack_size_e3m2_neonsdot,
          nk_dots_pack_e3m2_neonsdot, nk_dots_packed_e3m2_neonsdot);
    check("dots_pack_e3m2_neonsdot",
          test_dots_pack_layout<e3m2_t, host_backend_t, nk_dots_pack_size_e3m2_neonsdot,
                                nk_dots_packed_shape_e3m2_neonsdot, nk_dots_pack_e3m2_neonsdot>);
    check("dots_packed_e2m3_neonsdot", test_dots_packed<e2m3_t>, nk_dots_pack_size_e2m3_neonsdot,
          nk_dots_pack_e2m3_neonsdot, nk_dots_packed_e2m3_neonsdot);
    check("dots_pack_e2m3_neonsdot",
          test_dots_pack_layout<e2m3_t, host_backend_t, nk_dots_pack_size_e2m3_neonsdot,
                                nk_dots_packed_shape_e2m3_neonsdot, nk_dots_pack_e2m3_neonsdot>);
    check("dots_packed_e2m1_neonsdot", test_dots_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_neonsdot,
          nk_dots_pack_e2m1_neonsdot, nk_dots_packed_e2m1_neonsdot);
    check("dots_pack_e2m1_neonsdot",
          test_dots_pack_layout<e2m1x2_t, host_backend_t, nk_dots_pack_size_e2m1_neonsdot,
                                nk_dots_packed_shape_e2m1_neonsdot, nk_dots_pack_e2m1_neonsdot>);
    check("dots_symmetric_e3m2_neonsdot", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_neonsdot);
    check("dots_symmetric_e2m3_neonsdot", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_neonsdot);
    check("dots_symmetric_e2m1_neonsdot", test_dots_symmetric<e2m1x2_t>, nk_dots_symmetric_e2m1_neonsdot);

    check("angulars_packed_e3m2_neonsdot", test_angulars_packed<e3m2_t>, nk_dots_pack_size_e3m2_neonsdot,
          nk_dots_pack_e3m2_neonsdot, nk_angulars_packed_e3m2_neonsdot);
    check("angulars_packed_e2m3_neonsdot", test_angulars_packed<e2m3_t>, nk_dots_pack_size_e2m3_neonsdot,
          nk_dots_pack_e2m3_neonsdot, nk_angulars_packed_e2m3_neonsdot);
    check("angulars_packed_e2m1_neonsdot", test_angulars_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_neonsdot,
          nk_dots_pack_e2m1_neonsdot, nk_angulars_packed_e2m1_neonsdot);
    check("angulars_symmetric_e3m2_neonsdot", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_neonsdot);
    check("angulars_symmetric_e2m3_neonsdot", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_neonsdot);
    check("angulars_symmetric_e2m1_neonsdot", test_angulars_symmetric<e2m1x2_t>, nk_angulars_symmetric_e2m1_neonsdot);

    check("euclideans_packed_e3m2_neonsdot", test_euclideans_packed<e3m2_t>, nk_dots_pack_size_e3m2_neonsdot,
          nk_dots_pack_e3m2_neonsdot, nk_euclideans_packed_e3m2_neonsdot);
    check("euclideans_packed_e2m3_neonsdot", test_euclideans_packed<e2m3_t>, nk_dots_pack_size_e2m3_neonsdot,
          nk_dots_pack_e2m3_neonsdot, nk_euclideans_packed_e2m3_neonsdot);
    check("euclideans_packed_e2m1_neonsdot", test_euclideans_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_neonsdot,
          nk_dots_pack_e2m1_neonsdot, nk_euclideans_packed_e2m1_neonsdot);
    check("euclideans_symmetric_e3m2_neonsdot", test_euclideans_symmetric<e3m2_t>,
          nk_euclideans_symmetric_e3m2_neonsdot);
    check("euclideans_symmetric_e2m3_neonsdot", test_euclideans_symmetric<e2m3_t>,
          nk_euclideans_symmetric_e2m3_neonsdot);
    check("euclideans_symmetric_e2m1_neonsdot", test_euclideans_symmetric<e2m1x2_t>,
          nk_euclideans_symmetric_e2m1_neonsdot);

    check("attention_bidirectional_packed_i8_neonsdot", test_attention_bidirectional_packed<i8_t>,
          nk_attention_pack_size_i8_neonsdot, nk_attention_pack_i8_neonsdot,
          nk_attention_bidirectional_packed_i8_neonsdot);
    check("attention_causal_packed_i8_neonsdot", test_attention_causal_packed<i8_t>, nk_attention_pack_size_i8_neonsdot,
          nk_attention_pack_i8_neonsdot, nk_attention_causal_packed_i8_neonsdot);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_NEONFP8
    check.section("Cross NEON FP8", nk_cap_neonfp8_k);
    check("dots_packed_e5m2_neonfp8", test_dots_packed<e5m2_t>, nk_dots_pack_size_e5m2_neonfp8,
          nk_dots_pack_e5m2_neonfp8, nk_dots_packed_e5m2_neonfp8);
    check("dots_pack_e5m2_neonfp8",
          test_dots_pack_layout<e5m2_t, host_backend_t, nk_dots_pack_size_e5m2_neonfp8,
                                nk_dots_packed_shape_e5m2_neonfp8, nk_dots_pack_e5m2_neonfp8>);
    check("dots_packed_e4m3_neonfp8", test_dots_packed<e4m3_t>, nk_dots_pack_size_e4m3_neonfp8,
          nk_dots_pack_e4m3_neonfp8, nk_dots_packed_e4m3_neonfp8);
    check("dots_pack_e4m3_neonfp8",
          test_dots_pack_layout<e4m3_t, host_backend_t, nk_dots_pack_size_e4m3_neonfp8,
                                nk_dots_packed_shape_e4m3_neonfp8, nk_dots_pack_e4m3_neonfp8>);
    check("dots_packed_e3m2_neonfp8", test_dots_packed<e3m2_t>, nk_dots_pack_size_e3m2_neonfp8,
          nk_dots_pack_e3m2_neonfp8, nk_dots_packed_e3m2_neonfp8);
    check("dots_pack_e3m2_neonfp8",
          test_dots_pack_layout<e3m2_t, host_backend_t, nk_dots_pack_size_e3m2_neonfp8,
                                nk_dots_packed_shape_e3m2_neonfp8, nk_dots_pack_e3m2_neonfp8>);
    check("dots_packed_e2m3_neonfp8", test_dots_packed<e2m3_t>, nk_dots_pack_size_e2m3_neonfp8,
          nk_dots_pack_e2m3_neonfp8, nk_dots_packed_e2m3_neonfp8);
    check("dots_pack_e2m3_neonfp8",
          test_dots_pack_layout<e2m3_t, host_backend_t, nk_dots_pack_size_e2m3_neonfp8,
                                nk_dots_packed_shape_e2m3_neonfp8, nk_dots_pack_e2m3_neonfp8>);
    check("dots_packed_e2m1_neonfp8", test_dots_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_neonfp8,
          nk_dots_pack_e2m1_neonfp8, nk_dots_packed_e2m1_neonfp8);
    check("dots_pack_e2m1_neonfp8",
          test_dots_pack_layout<e2m1x2_t, host_backend_t, nk_dots_pack_size_e2m1_neonfp8,
                                nk_dots_packed_shape_e2m1_neonfp8, nk_dots_pack_e2m1_neonfp8>);

    check("dots_symmetric_e5m2_neonfp8", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_neonfp8);
    check("dots_symmetric_e4m3_neonfp8", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_neonfp8);
    check("dots_symmetric_e3m2_neonfp8", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_neonfp8);
    check("dots_symmetric_e2m3_neonfp8", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_neonfp8);
    check("dots_symmetric_e2m1_neonfp8", test_dots_symmetric<e2m1x2_t>, nk_dots_symmetric_e2m1_neonfp8);

    check("angulars_packed_e5m2_neonfp8", test_angulars_packed<e5m2_t>, nk_dots_pack_size_e5m2_neonfp8,
          nk_dots_pack_e5m2_neonfp8, nk_angulars_packed_e5m2_neonfp8);
    check("angulars_packed_e4m3_neonfp8", test_angulars_packed<e4m3_t>, nk_dots_pack_size_e4m3_neonfp8,
          nk_dots_pack_e4m3_neonfp8, nk_angulars_packed_e4m3_neonfp8);
    check("angulars_packed_e3m2_neonfp8", test_angulars_packed<e3m2_t>, nk_dots_pack_size_e3m2_neonfp8,
          nk_dots_pack_e3m2_neonfp8, nk_angulars_packed_e3m2_neonfp8);
    check("angulars_packed_e2m3_neonfp8", test_angulars_packed<e2m3_t>, nk_dots_pack_size_e2m3_neonfp8,
          nk_dots_pack_e2m3_neonfp8, nk_angulars_packed_e2m3_neonfp8);
    check("angulars_packed_e2m1_neonfp8", test_angulars_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_neonfp8,
          nk_dots_pack_e2m1_neonfp8, nk_angulars_packed_e2m1_neonfp8);

    check("angulars_symmetric_e5m2_neonfp8", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_neonfp8);
    check("angulars_symmetric_e4m3_neonfp8", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_neonfp8);
    check("angulars_symmetric_e3m2_neonfp8", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_neonfp8);
    check("angulars_symmetric_e2m3_neonfp8", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_neonfp8);
    check("angulars_symmetric_e2m1_neonfp8", test_angulars_symmetric<e2m1x2_t>, nk_angulars_symmetric_e2m1_neonfp8);

    check("euclideans_packed_e5m2_neonfp8", test_euclideans_packed<e5m2_t>, nk_dots_pack_size_e5m2_neonfp8,
          nk_dots_pack_e5m2_neonfp8, nk_euclideans_packed_e5m2_neonfp8);
    check("euclideans_packed_e4m3_neonfp8", test_euclideans_packed<e4m3_t>, nk_dots_pack_size_e4m3_neonfp8,
          nk_dots_pack_e4m3_neonfp8, nk_euclideans_packed_e4m3_neonfp8);
    check("euclideans_packed_e3m2_neonfp8", test_euclideans_packed<e3m2_t>, nk_dots_pack_size_e3m2_neonfp8,
          nk_dots_pack_e3m2_neonfp8, nk_euclideans_packed_e3m2_neonfp8);
    check("euclideans_packed_e2m3_neonfp8", test_euclideans_packed<e2m3_t>, nk_dots_pack_size_e2m3_neonfp8,
          nk_dots_pack_e2m3_neonfp8, nk_euclideans_packed_e2m3_neonfp8);
    check("euclideans_packed_e2m1_neonfp8", test_euclideans_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_neonfp8,
          nk_dots_pack_e2m1_neonfp8, nk_euclideans_packed_e2m1_neonfp8);

    check("euclideans_symmetric_e5m2_neonfp8", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_neonfp8);
    check("euclideans_symmetric_e4m3_neonfp8", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_neonfp8);
    check("euclideans_symmetric_e3m2_neonfp8", test_euclideans_symmetric<e3m2_t>, nk_euclideans_symmetric_e3m2_neonfp8);
    check("euclideans_symmetric_e2m3_neonfp8", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_neonfp8);
    check("euclideans_symmetric_e2m1_neonfp8", test_euclideans_symmetric<e2m1x2_t>,
          nk_euclideans_symmetric_e2m1_neonfp8);
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_SME
    check.section("Cross SME", nk_cap_sme_k);
    check("dots_packed_bf16_sme", test_dots_packed<bf16_t>, nk_dots_pack_size_bf16_sme, nk_dots_pack_bf16_sme,
          nk_dots_packed_bf16_sme);
    check("dots_pack_bf16_sme", test_dots_pack_layout<bf16_t, host_backend_t, nk_dots_pack_size_bf16_sme,
                                                      nk_dots_packed_shape_bf16_sme, nk_dots_pack_bf16_sme>);
    check("dots_packed_f16_sme", test_dots_packed<f16_t>, nk_dots_pack_size_f16_sme, nk_dots_pack_f16_sme,
          nk_dots_packed_f16_sme);
    check("dots_pack_f16_sme", test_dots_pack_layout<f16_t, host_backend_t, nk_dots_pack_size_f16_sme,
                                                     nk_dots_packed_shape_f16_sme, nk_dots_pack_f16_sme>);
    check("dots_packed_e5m2_sme", test_dots_packed<e5m2_t>, nk_dots_pack_size_e5m2_sme, nk_dots_pack_e5m2_sme,
          nk_dots_packed_e5m2_sme);
    check("dots_pack_e5m2_sme", test_dots_pack_layout<e5m2_t, host_backend_t, nk_dots_pack_size_e5m2_sme,
                                                      nk_dots_packed_shape_e5m2_sme, nk_dots_pack_e5m2_sme>);
    check("dots_packed_e4m3_sme", test_dots_packed<e4m3_t>, nk_dots_pack_size_e4m3_sme, nk_dots_pack_e4m3_sme,
          nk_dots_packed_e4m3_sme);
    check("dots_pack_e4m3_sme", test_dots_pack_layout<e4m3_t, host_backend_t, nk_dots_pack_size_e4m3_sme,
                                                      nk_dots_packed_shape_e4m3_sme, nk_dots_pack_e4m3_sme>);
    check("dots_packed_e3m2_sme", test_dots_packed<e3m2_t>, nk_dots_pack_size_e3m2_sme, nk_dots_pack_e3m2_sme,
          nk_dots_packed_e3m2_sme);
    check("dots_pack_e3m2_sme", test_dots_pack_layout<e3m2_t, host_backend_t, nk_dots_pack_size_e3m2_sme,
                                                      nk_dots_packed_shape_e3m2_sme, nk_dots_pack_e3m2_sme>);
    check("dots_packed_e2m3_sme", test_dots_packed<e2m3_t>, nk_dots_pack_size_e2m3_sme, nk_dots_pack_e2m3_sme,
          nk_dots_packed_e2m3_sme);
    check("dots_pack_e2m3_sme", test_dots_pack_layout<e2m3_t, host_backend_t, nk_dots_pack_size_e2m3_sme,
                                                      nk_dots_packed_shape_e2m3_sme, nk_dots_pack_e2m3_sme>);
    check("dots_packed_e2m1_sme", test_dots_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_sme, nk_dots_pack_e2m1_sme,
          nk_dots_packed_e2m1_sme);
    check("dots_pack_e2m1_sme", test_dots_pack_layout<e2m1x2_t, host_backend_t, nk_dots_pack_size_e2m1_sme,
                                                      nk_dots_packed_shape_e2m1_sme, nk_dots_pack_e2m1_sme>);
    check("dots_packed_i8_sme", test_dots_packed<i8_t>, nk_dots_pack_size_i8_sme, nk_dots_pack_i8_sme,
          nk_dots_packed_i8_sme);
    check("dots_pack_i8_sme", test_dots_pack_layout<i8_t, host_backend_t, nk_dots_pack_size_i8_sme,
                                                    nk_dots_packed_shape_i8_sme, nk_dots_pack_i8_sme>);
    check("dots_packed_i4_sme", test_dots_packed<i4x2_t>, nk_dots_pack_size_i4_sme, nk_dots_pack_i4_sme,
          nk_dots_packed_i4_sme);
    check("dots_pack_i4_sme", test_dots_pack_layout<i4x2_t, host_backend_t, nk_dots_pack_size_i4_sme,
                                                    nk_dots_packed_shape_i4_sme, nk_dots_pack_i4_sme>);
    check("dots_packed_u8_sme", test_dots_packed<u8_t>, nk_dots_pack_size_u8_sme, nk_dots_pack_u8_sme,
          nk_dots_packed_u8_sme);
    check("dots_pack_u8_sme", test_dots_pack_layout<u8_t, host_backend_t, nk_dots_pack_size_u8_sme,
                                                    nk_dots_packed_shape_u8_sme, nk_dots_pack_u8_sme>);
    check("dots_packed_u4_sme", test_dots_packed<u4x2_t>, nk_dots_pack_size_u4_sme, nk_dots_pack_u4_sme,
          nk_dots_packed_u4_sme);
    check("dots_pack_u4_sme", test_dots_pack_layout<u4x2_t, host_backend_t, nk_dots_pack_size_u4_sme,
                                                    nk_dots_packed_shape_u4_sme, nk_dots_pack_u4_sme>);

    check("dots_symmetric_bf16_sme", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_sme);
    check("dots_symmetric_f16_sme", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_sme);
    check("dots_symmetric_e5m2_sme", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_sme);
    check("dots_symmetric_e4m3_sme", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_sme);
    check("dots_symmetric_e3m2_sme", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_sme);
    check("dots_symmetric_e2m3_sme", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_sme);
    check("dots_symmetric_e2m1_sme", test_dots_symmetric<e2m1x2_t>, nk_dots_symmetric_e2m1_sme);
    check("dots_symmetric_i8_sme", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sme);
    check("dots_symmetric_i4_sme", test_dots_symmetric<i4x2_t>, nk_dots_symmetric_i4_sme);
    check("dots_symmetric_u8_sme", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sme);
    check("dots_symmetric_u4_sme", test_dots_symmetric<u4x2_t>, nk_dots_symmetric_u4_sme);

    check("angulars_packed_bf16_sme", test_angulars_packed<bf16_t>, nk_dots_pack_size_bf16_sme, nk_dots_pack_bf16_sme,
          nk_angulars_packed_bf16_sme);
    check("angulars_packed_f16_sme", test_angulars_packed<f16_t>, nk_dots_pack_size_f16_sme, nk_dots_pack_f16_sme,
          nk_angulars_packed_f16_sme);
    check("angulars_packed_e5m2_sme", test_angulars_packed<e5m2_t>, nk_dots_pack_size_e5m2_sme, nk_dots_pack_e5m2_sme,
          nk_angulars_packed_e5m2_sme);
    check("angulars_packed_e4m3_sme", test_angulars_packed<e4m3_t>, nk_dots_pack_size_e4m3_sme, nk_dots_pack_e4m3_sme,
          nk_angulars_packed_e4m3_sme);
    check("angulars_packed_e3m2_sme", test_angulars_packed<e3m2_t>, nk_dots_pack_size_e3m2_sme, nk_dots_pack_e3m2_sme,
          nk_angulars_packed_e3m2_sme);
    check("angulars_packed_e2m3_sme", test_angulars_packed<e2m3_t>, nk_dots_pack_size_e2m3_sme, nk_dots_pack_e2m3_sme,
          nk_angulars_packed_e2m3_sme);
    check("angulars_packed_e2m1_sme", test_angulars_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_sme, nk_dots_pack_e2m1_sme,
          nk_angulars_packed_e2m1_sme);
    check("angulars_packed_i8_sme", test_angulars_packed<i8_t>, nk_dots_pack_size_i8_sme, nk_dots_pack_i8_sme,
          nk_angulars_packed_i8_sme);
    check("angulars_packed_i4_sme", test_angulars_packed<i4x2_t>, nk_dots_pack_size_i4_sme, nk_dots_pack_i4_sme,
          nk_angulars_packed_i4_sme);
    check("angulars_packed_u8_sme", test_angulars_packed<u8_t>, nk_dots_pack_size_u8_sme, nk_dots_pack_u8_sme,
          nk_angulars_packed_u8_sme);
    check("angulars_packed_u4_sme", test_angulars_packed<u4x2_t>, nk_dots_pack_size_u4_sme, nk_dots_pack_u4_sme,
          nk_angulars_packed_u4_sme);

    check("angulars_symmetric_bf16_sme", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_sme);
    check("angulars_symmetric_f16_sme", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_sme);
    check("angulars_symmetric_e5m2_sme", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_sme);
    check("angulars_symmetric_e4m3_sme", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_sme);
    check("angulars_symmetric_e3m2_sme", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_sme);
    check("angulars_symmetric_e2m3_sme", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_sme);
    check("angulars_symmetric_e2m1_sme", test_angulars_symmetric<e2m1x2_t>, nk_angulars_symmetric_e2m1_sme);
    check("angulars_symmetric_i8_sme", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_sme);
    check("angulars_symmetric_i4_sme", test_angulars_symmetric<i4x2_t>, nk_angulars_symmetric_i4_sme);
    check("angulars_symmetric_u8_sme", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_sme);
    check("angulars_symmetric_u4_sme", test_angulars_symmetric<u4x2_t>, nk_angulars_symmetric_u4_sme);

    check("euclideans_packed_bf16_sme", test_euclideans_packed<bf16_t>, nk_dots_pack_size_bf16_sme,
          nk_dots_pack_bf16_sme, nk_euclideans_packed_bf16_sme);
    check("euclideans_packed_f16_sme", test_euclideans_packed<f16_t>, nk_dots_pack_size_f16_sme, nk_dots_pack_f16_sme,
          nk_euclideans_packed_f16_sme);
    check("euclideans_packed_e5m2_sme", test_euclideans_packed<e5m2_t>, nk_dots_pack_size_e5m2_sme,
          nk_dots_pack_e5m2_sme, nk_euclideans_packed_e5m2_sme);
    check("euclideans_packed_e4m3_sme", test_euclideans_packed<e4m3_t>, nk_dots_pack_size_e4m3_sme,
          nk_dots_pack_e4m3_sme, nk_euclideans_packed_e4m3_sme);
    check("euclideans_packed_e3m2_sme", test_euclideans_packed<e3m2_t>, nk_dots_pack_size_e3m2_sme,
          nk_dots_pack_e3m2_sme, nk_euclideans_packed_e3m2_sme);
    check("euclideans_packed_e2m3_sme", test_euclideans_packed<e2m3_t>, nk_dots_pack_size_e2m3_sme,
          nk_dots_pack_e2m3_sme, nk_euclideans_packed_e2m3_sme);
    check("euclideans_packed_e2m1_sme", test_euclideans_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_sme,
          nk_dots_pack_e2m1_sme, nk_euclideans_packed_e2m1_sme);
    check("euclideans_packed_i8_sme", test_euclideans_packed<i8_t>, nk_dots_pack_size_i8_sme, nk_dots_pack_i8_sme,
          nk_euclideans_packed_i8_sme);
    check("euclideans_packed_i4_sme", test_euclideans_packed<i4x2_t>, nk_dots_pack_size_i4_sme, nk_dots_pack_i4_sme,
          nk_euclideans_packed_i4_sme);
    check("euclideans_packed_u8_sme", test_euclideans_packed<u8_t>, nk_dots_pack_size_u8_sme, nk_dots_pack_u8_sme,
          nk_euclideans_packed_u8_sme);
    check("euclideans_packed_u4_sme", test_euclideans_packed<u4x2_t>, nk_dots_pack_size_u4_sme, nk_dots_pack_u4_sme,
          nk_euclideans_packed_u4_sme);

    check("euclideans_symmetric_bf16_sme", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_sme);
    check("euclideans_symmetric_f16_sme", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_sme);
    check("euclideans_symmetric_e5m2_sme", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_sme);
    check("euclideans_symmetric_e4m3_sme", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_sme);
    check("euclideans_symmetric_e3m2_sme", test_euclideans_symmetric<e3m2_t>, nk_euclideans_symmetric_e3m2_sme);
    check("euclideans_symmetric_e2m3_sme", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_sme);
    check("euclideans_symmetric_e2m1_sme", test_euclideans_symmetric<e2m1x2_t>, nk_euclideans_symmetric_e2m1_sme);
    check("euclideans_symmetric_i8_sme", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_sme);
    check("euclideans_symmetric_i4_sme", test_euclideans_symmetric<i4x2_t>, nk_euclideans_symmetric_i4_sme);
    check("euclideans_symmetric_u8_sme", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_sme);
    check("euclideans_symmetric_u4_sme", test_euclideans_symmetric<u4x2_t>, nk_euclideans_symmetric_u4_sme);

    check("attention_bidirectional_packed_bf16_sme", test_attention_bidirectional_packed<bf16_t>,
          nk_attention_pack_size_bf16_sme, nk_attention_pack_bf16_sme, nk_attention_bidirectional_packed_bf16_sme);
    check("attention_causal_packed_bf16_sme", test_attention_causal_packed<bf16_t>, nk_attention_pack_size_bf16_sme,
          nk_attention_pack_bf16_sme, nk_attention_causal_packed_bf16_sme);
    check("attention_bidirectional_packed_e4m3_sme", test_attention_bidirectional_packed<e4m3_t>,
          nk_attention_pack_size_e4m3_sme, nk_attention_pack_e4m3_sme, nk_attention_bidirectional_packed_e4m3_sme);
    check("attention_causal_packed_e4m3_sme", test_attention_causal_packed<e4m3_t>, nk_attention_pack_size_e4m3_sme,
          nk_attention_pack_e4m3_sme, nk_attention_causal_packed_e4m3_sme);
    check("attention_bidirectional_packed_i8_sme", test_attention_bidirectional_packed<i8_t>,
          nk_attention_pack_size_i8_sme, nk_attention_pack_i8_sme, nk_attention_bidirectional_packed_i8_sme);
    check("attention_causal_packed_i8_sme", test_attention_causal_packed<i8_t>, nk_attention_pack_size_i8_sme,
          nk_attention_pack_i8_sme, nk_attention_causal_packed_i8_sme);
#endif // NUMKONG_TARGET_SME

#if NUMKONG_TARGET_SMEBI32
    check.section("Cross SME BI32", nk_cap_smebi32_k);
    check("dots_packed_u1_smebi32", test_dots_packed<u1x8_t>, nk_dots_pack_size_u1_smebi32, nk_dots_pack_u1_smebi32,
          nk_dots_packed_u1_smebi32);
    check("dots_pack_u1_smebi32", test_dots_pack_layout<u1x8_t, host_backend_t, nk_dots_pack_size_u1_smebi32,
                                                        nk_dots_packed_shape_u1_smebi32, nk_dots_pack_u1_smebi32>);
    check("dots_symmetric_u1_smebi32", test_dots_symmetric<u1x8_t>, nk_dots_symmetric_u1_smebi32);

    check("hammings_packed_u1_smebi32", test_hammings_packed<u1x8_t>, nk_dots_pack_size_u1_smebi32,
          nk_dots_pack_u1_smebi32, nk_hammings_packed_u1_smebi32);
    check("hammings_symmetric_u1_smebi32", test_hammings_symmetric<u1x8_t>, nk_hammings_symmetric_u1_smebi32);

    check("jaccards_packed_u1_smebi32", test_jaccards_packed<u1x8_t>, nk_dots_pack_size_u1_smebi32,
          nk_dots_pack_u1_smebi32, nk_jaccards_packed_u1_smebi32);
    check("jaccards_symmetric_u1_smebi32", test_jaccards_symmetric<u1x8_t>, nk_jaccards_symmetric_u1_smebi32);
#endif // NUMKONG_TARGET_SMEBI32

#if NUMKONG_TARGET_SMEF64
    check.section("Cross SME F64", nk_cap_smef64_k);
    check("dots_packed_f64_smef64", test_dots_packed<f64_t>, nk_dots_pack_size_f64_smef64, nk_dots_pack_f64_smef64,
          nk_dots_packed_f64_smef64);
    check("dots_pack_f64_smef64", test_dots_pack_layout<f64_t, host_backend_t, nk_dots_pack_size_f64_smef64,
                                                        nk_dots_packed_shape_f64_smef64, nk_dots_pack_f64_smef64>);
    check("dots_packed_f32_smef64", test_dots_packed<f32_t>, nk_dots_pack_size_f32_smef64, nk_dots_pack_f32_smef64,
          nk_dots_packed_f32_smef64);
    check("dots_pack_f32_smef64", test_dots_pack_layout<f32_t, host_backend_t, nk_dots_pack_size_f32_smef64,
                                                        nk_dots_packed_shape_f32_smef64, nk_dots_pack_f32_smef64>);
    check("dots_symmetric_f64_smef64", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_smef64);
    check("dots_symmetric_f32_smef64", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_smef64);

    check("angulars_packed_f64_smef64", test_angulars_packed<f64_t>, nk_dots_pack_size_f64_smef64,
          nk_dots_pack_f64_smef64, nk_angulars_packed_f64_smef64);
    check("angulars_packed_f32_smef64", test_angulars_packed<f32_t>, nk_dots_pack_size_f32_smef64,
          nk_dots_pack_f32_smef64, nk_angulars_packed_f32_smef64);

    check("angulars_symmetric_f64_smef64", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_smef64);
    check("angulars_symmetric_f32_smef64", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_smef64);

    check("euclideans_packed_f64_smef64", test_euclideans_packed<f64_t>, nk_dots_pack_size_f64_smef64,
          nk_dots_pack_f64_smef64, nk_euclideans_packed_f64_smef64);
    check("euclideans_packed_f32_smef64", test_euclideans_packed<f32_t>, nk_dots_pack_size_f32_smef64,
          nk_dots_pack_f32_smef64, nk_euclideans_packed_f32_smef64);

    check("euclideans_symmetric_f64_smef64", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_smef64);
    check("euclideans_symmetric_f32_smef64", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_smef64);
#endif // NUMKONG_TARGET_SMEF64
}

} // namespace ashvardanian::numkong::test
