/**
 *  @brief Batch operation tests - RVV ISA family (RISC-V Vector).
 *  @file test/test_cross_rvv.cpp
 *  @author Ash Vardanian
 *  @date February 15, 2026
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_rvv() {
    [[maybe_unused]] error_stats_section_t check;

#if NK_TARGET_RVV

    check("dots_packed_f64_rvv", test_dots_packed<f64_t>, nk_dots_packed_size_f64_rvv, nk_dots_pack_f64_rvv,
          nk_dots_packed_f64_rvv);
    check("dots_packed_f32_rvv", test_dots_packed<f32_t>, nk_dots_packed_size_f32_rvv, nk_dots_pack_f32_rvv,
          nk_dots_packed_f32_rvv);
    check("dots_packed_f16_rvv", test_dots_packed<f16_t>, nk_dots_packed_size_f16_rvv, nk_dots_pack_f16_rvv,
          nk_dots_packed_f16_rvv);
    check("dots_packed_bf16_rvv", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_rvv, nk_dots_pack_bf16_rvv,
          nk_dots_packed_bf16_rvv);
    check("dots_packed_e4m3_rvv", test_dots_packed<e4m3_t>, nk_dots_packed_size_e4m3_rvv, nk_dots_pack_e4m3_rvv,
          nk_dots_packed_e4m3_rvv);
    check("dots_packed_e5m2_rvv", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_rvv, nk_dots_pack_e5m2_rvv,
          nk_dots_packed_e5m2_rvv);
    check("dots_packed_e2m3_rvv", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_rvv, nk_dots_pack_e2m3_rvv,
          nk_dots_packed_e2m3_rvv);
    check("dots_packed_e3m2_rvv", test_dots_packed<e3m2_t>, nk_dots_packed_size_e3m2_rvv, nk_dots_pack_e3m2_rvv,
          nk_dots_packed_e3m2_rvv);
    check("dots_packed_i8_rvv", test_dots_packed<i8_t>, nk_dots_packed_size_i8_rvv, nk_dots_pack_i8_rvv,
          nk_dots_packed_i8_rvv);
    check("dots_packed_u8_rvv", test_dots_packed<u8_t>, nk_dots_packed_size_u8_rvv, nk_dots_pack_u8_rvv,
          nk_dots_packed_u8_rvv);

    check("dots_symmetric_f64_rvv", test_dots_symmetric<f64_t>, nk_dots_symmetric_f64_rvv);
    check("dots_symmetric_f32_rvv", test_dots_symmetric<f32_t>, nk_dots_symmetric_f32_rvv);
    check("dots_symmetric_f16_rvv", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_rvv);
    check("dots_symmetric_bf16_rvv", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_rvv);
    check("dots_symmetric_e4m3_rvv", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_rvv);
    check("dots_symmetric_e5m2_rvv", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_rvv);
    check("dots_symmetric_e2m3_rvv", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_rvv);
    check("dots_symmetric_e3m2_rvv", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_rvv);
    check("dots_symmetric_i8_rvv", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_rvv);
    check("dots_symmetric_u8_rvv", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_rvv);

    check("angulars_packed_f64_rvv", test_angulars_packed<f64_t>, nk_dots_packed_size_f64_rvv, nk_dots_pack_f64_rvv,
          nk_angulars_packed_f64_rvv);
    check("angulars_packed_f32_rvv", test_angulars_packed<f32_t>, nk_dots_packed_size_f32_rvv, nk_dots_pack_f32_rvv,
          nk_angulars_packed_f32_rvv);
    check("angulars_packed_f16_rvv", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_rvv, nk_dots_pack_f16_rvv,
          nk_angulars_packed_f16_rvv);
    check("angulars_packed_bf16_rvv", test_angulars_packed<bf16_t>, nk_dots_packed_size_bf16_rvv, nk_dots_pack_bf16_rvv,
          nk_angulars_packed_bf16_rvv);
    check("angulars_packed_e4m3_rvv", test_angulars_packed<e4m3_t>, nk_dots_packed_size_e4m3_rvv, nk_dots_pack_e4m3_rvv,
          nk_angulars_packed_e4m3_rvv);
    check("angulars_packed_e5m2_rvv", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_rvv, nk_dots_pack_e5m2_rvv,
          nk_angulars_packed_e5m2_rvv);
    check("angulars_packed_e2m3_rvv", test_angulars_packed<e2m3_t>, nk_dots_packed_size_e2m3_rvv, nk_dots_pack_e2m3_rvv,
          nk_angulars_packed_e2m3_rvv);
    check("angulars_packed_e3m2_rvv", test_angulars_packed<e3m2_t>, nk_dots_packed_size_e3m2_rvv, nk_dots_pack_e3m2_rvv,
          nk_angulars_packed_e3m2_rvv);
    check("angulars_packed_i8_rvv", test_angulars_packed<i8_t>, nk_dots_packed_size_i8_rvv, nk_dots_pack_i8_rvv,
          nk_angulars_packed_i8_rvv);
    check("angulars_packed_u8_rvv", test_angulars_packed<u8_t>, nk_dots_packed_size_u8_rvv, nk_dots_pack_u8_rvv,
          nk_angulars_packed_u8_rvv);

    check("angulars_symmetric_f64_rvv", test_angulars_symmetric<f64_t>, nk_angulars_symmetric_f64_rvv);
    check("angulars_symmetric_f32_rvv", test_angulars_symmetric<f32_t>, nk_angulars_symmetric_f32_rvv);
    check("angulars_symmetric_f16_rvv", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_rvv);
    check("angulars_symmetric_bf16_rvv", test_angulars_symmetric<bf16_t>, nk_angulars_symmetric_bf16_rvv);
    check("angulars_symmetric_e4m3_rvv", test_angulars_symmetric<e4m3_t>, nk_angulars_symmetric_e4m3_rvv);
    check("angulars_symmetric_e5m2_rvv", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_rvv);
    check("angulars_symmetric_e2m3_rvv", test_angulars_symmetric<e2m3_t>, nk_angulars_symmetric_e2m3_rvv);
    check("angulars_symmetric_e3m2_rvv", test_angulars_symmetric<e3m2_t>, nk_angulars_symmetric_e3m2_rvv);
    check("angulars_symmetric_i8_rvv", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_rvv);
    check("angulars_symmetric_u8_rvv", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_rvv);

    check("euclideans_packed_f64_rvv", test_euclideans_packed<f64_t>, nk_dots_packed_size_f64_rvv, nk_dots_pack_f64_rvv,
          nk_euclideans_packed_f64_rvv);
    check("euclideans_packed_f32_rvv", test_euclideans_packed<f32_t>, nk_dots_packed_size_f32_rvv, nk_dots_pack_f32_rvv,
          nk_euclideans_packed_f32_rvv);
    check("euclideans_packed_f16_rvv", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_rvv, nk_dots_pack_f16_rvv,
          nk_euclideans_packed_f16_rvv);
    check("euclideans_packed_bf16_rvv", test_euclideans_packed<bf16_t>, nk_dots_packed_size_bf16_rvv,
          nk_dots_pack_bf16_rvv, nk_euclideans_packed_bf16_rvv);
    check("euclideans_packed_e4m3_rvv", test_euclideans_packed<e4m3_t>, nk_dots_packed_size_e4m3_rvv,
          nk_dots_pack_e4m3_rvv, nk_euclideans_packed_e4m3_rvv);
    check("euclideans_packed_e5m2_rvv", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_rvv,
          nk_dots_pack_e5m2_rvv, nk_euclideans_packed_e5m2_rvv);
    check("euclideans_packed_e2m3_rvv", test_euclideans_packed<e2m3_t>, nk_dots_packed_size_e2m3_rvv,
          nk_dots_pack_e2m3_rvv, nk_euclideans_packed_e2m3_rvv);
    check("euclideans_packed_e3m2_rvv", test_euclideans_packed<e3m2_t>, nk_dots_packed_size_e3m2_rvv,
          nk_dots_pack_e3m2_rvv, nk_euclideans_packed_e3m2_rvv);
    check("euclideans_packed_i8_rvv", test_euclideans_packed<i8_t>, nk_dots_packed_size_i8_rvv, nk_dots_pack_i8_rvv,
          nk_euclideans_packed_i8_rvv);
    check("euclideans_packed_u8_rvv", test_euclideans_packed<u8_t>, nk_dots_packed_size_u8_rvv, nk_dots_pack_u8_rvv,
          nk_euclideans_packed_u8_rvv);

    check("euclideans_symmetric_f64_rvv", test_euclideans_symmetric<f64_t>, nk_euclideans_symmetric_f64_rvv);
    check("euclideans_symmetric_f32_rvv", test_euclideans_symmetric<f32_t>, nk_euclideans_symmetric_f32_rvv);
    check("euclideans_symmetric_f16_rvv", test_euclideans_symmetric<f16_t>, nk_euclideans_symmetric_f16_rvv);
    check("euclideans_symmetric_bf16_rvv", test_euclideans_symmetric<bf16_t>, nk_euclideans_symmetric_bf16_rvv);
    check("euclideans_symmetric_e4m3_rvv", test_euclideans_symmetric<e4m3_t>, nk_euclideans_symmetric_e4m3_rvv);
    check("euclideans_symmetric_e5m2_rvv", test_euclideans_symmetric<e5m2_t>, nk_euclideans_symmetric_e5m2_rvv);
    check("euclideans_symmetric_e2m3_rvv", test_euclideans_symmetric<e2m3_t>, nk_euclideans_symmetric_e2m3_rvv);
    check("euclideans_symmetric_e3m2_rvv", test_euclideans_symmetric<e3m2_t>, nk_euclideans_symmetric_e3m2_rvv);
    check("euclideans_symmetric_i8_rvv", test_euclideans_symmetric<i8_t>, nk_euclideans_symmetric_i8_rvv);
    check("euclideans_symmetric_u8_rvv", test_euclideans_symmetric<u8_t>, nk_euclideans_symmetric_u8_rvv);

    check("attention_packed_bf16_rvv", test_attention_packed<bf16_t>, nk_attention_packed_size_bf16_rvv,
          nk_attention_pack_bf16_rvv, nk_attention_packed_bf16_rvv);
    check("attention_packed_e4m3_rvv", test_attention_packed<e4m3_t>, nk_attention_packed_size_e4m3_rvv,
          nk_attention_pack_e4m3_rvv, nk_attention_packed_e4m3_rvv);
    check("attention_packed_i8_rvv", test_attention_packed<i8_t>, nk_attention_packed_size_i8_rvv,
          nk_attention_pack_i8_rvv, nk_attention_packed_i8_rvv);
#endif
}
