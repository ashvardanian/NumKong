/**
 *  @file test/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Batch operation tests - ROCm capabilities.
 *
 *  Compiled with every capability that some listed architecture runs: the ROCm baseline, then
 *  CDNA3, CDNA4 and CDNA5.
 */
#include "cross_device.hpp"

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_ROCM_

void test_cross_rocm(error_stats_section_t &check) {
    check.section("Cross ROCm", nk_cap_rocm_k);
    check("dots_packed_f64_rocm", test_dots_packed<f64_t, rocm_backend_t>, nk_dots_pack_size_f64_rocm,
          nk_dots_pack_f64_rocm, nk_dots_packed_f64_rocm);
    check("dots_pack_f64_rocm", test_dots_pack_layout<f64_t, rocm_backend_t, nk_dots_pack_size_f64_rocm,
                                                      nk_dots_packed_shape_f64_rocm, nk_dots_pack_f64_rocm>);
    check("dots_contract_f64_rocm", test_dots_launch_contract<f64_t, rocm_backend_t, nk_dots_pack_size_f64_rocm,
                                                              nk_dots_packed_f64_rocm, nk_dots_symmetric_f64_rocm>);
    check("dots_symmetric_f64_rocm", test_dots_symmetric<f64_t, rocm_backend_t>, nk_dots_symmetric_f64_rocm);
    check("dots_packed_f32_rocm", test_dots_packed<f32_t, rocm_backend_t>, nk_dots_pack_size_f32_rocm,
          nk_dots_pack_f32_rocm, nk_dots_packed_f32_rocm);
    check("dots_pack_f32_rocm", test_dots_pack_layout<f32_t, rocm_backend_t, nk_dots_pack_size_f32_rocm,
                                                      nk_dots_packed_shape_f32_rocm, nk_dots_pack_f32_rocm>);
    check("dots_contract_f32_rocm", test_dots_launch_contract<f32_t, rocm_backend_t, nk_dots_pack_size_f32_rocm,
                                                              nk_dots_packed_f32_rocm, nk_dots_symmetric_f32_rocm>);
    check("dots_symmetric_f32_rocm", test_dots_symmetric<f32_t, rocm_backend_t>, nk_dots_symmetric_f32_rocm);
    check("dots_packed_bf16_rocm", test_dots_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_rocm,
          nk_dots_pack_bf16_rocm, nk_dots_packed_bf16_rocm);
    check("dots_pack_bf16_rocm", test_dots_pack_layout<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_rocm,
                                                       nk_dots_packed_shape_bf16_rocm, nk_dots_pack_bf16_rocm>);
    check("dots_contract_bf16_rocm", test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_rocm,
                                                               nk_dots_packed_bf16_rocm, nk_dots_symmetric_bf16_rocm>);
    check("dots_symmetric_bf16_rocm", test_dots_symmetric<bf16_t, rocm_backend_t>, nk_dots_symmetric_bf16_rocm);
    check("dots_packed_f16_rocm", test_dots_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_rocm,
          nk_dots_pack_f16_rocm, nk_dots_packed_f16_rocm);
    check("dots_pack_f16_rocm", test_dots_pack_layout<f16_t, rocm_backend_t, nk_dots_pack_size_f16_rocm,
                                                      nk_dots_packed_shape_f16_rocm, nk_dots_pack_f16_rocm>);
    check("dots_contract_f16_rocm", test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_rocm,
                                                              nk_dots_packed_f16_rocm, nk_dots_symmetric_f16_rocm>);
    check("dots_symmetric_f16_rocm", test_dots_symmetric<f16_t, rocm_backend_t>, nk_dots_symmetric_f16_rocm);
    check("dots_packed_e5m2_rocm", test_dots_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_rocm,
          nk_dots_pack_e5m2_rocm, nk_dots_packed_e5m2_rocm);
    check("dots_pack_e5m2_rocm", test_dots_pack_layout<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_rocm,
                                                       nk_dots_packed_shape_e5m2_rocm, nk_dots_pack_e5m2_rocm>);
    check("dots_contract_e5m2_rocm", test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_rocm,
                                                               nk_dots_packed_e5m2_rocm, nk_dots_symmetric_e5m2_rocm>);
    check("dots_symmetric_e5m2_rocm", test_dots_symmetric<e5m2_t, rocm_backend_t>, nk_dots_symmetric_e5m2_rocm);
    check("dots_packed_e4m3_rocm", test_dots_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_rocm,
          nk_dots_pack_e4m3_rocm, nk_dots_packed_e4m3_rocm);
    check("dots_pack_e4m3_rocm", test_dots_pack_layout<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_rocm,
                                                       nk_dots_packed_shape_e4m3_rocm, nk_dots_pack_e4m3_rocm>);
    check("dots_contract_e4m3_rocm", test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_rocm,
                                                               nk_dots_packed_e4m3_rocm, nk_dots_symmetric_e4m3_rocm>);
    check("dots_symmetric_e4m3_rocm", test_dots_symmetric<e4m3_t, rocm_backend_t>, nk_dots_symmetric_e4m3_rocm);
    check("dots_packed_e3m2_rocm", test_dots_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_rocm,
          nk_dots_pack_e3m2_rocm, nk_dots_packed_e3m2_rocm);
    check("dots_pack_e3m2_rocm", test_dots_pack_layout<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_rocm,
                                                       nk_dots_packed_shape_e3m2_rocm, nk_dots_pack_e3m2_rocm>);
    check("dots_contract_e3m2_rocm", test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_rocm,
                                                               nk_dots_packed_e3m2_rocm, nk_dots_symmetric_e3m2_rocm>);
    check("dots_symmetric_e3m2_rocm", test_dots_symmetric<e3m2_t, rocm_backend_t>, nk_dots_symmetric_e3m2_rocm);
    check("dots_packed_e2m3_rocm", test_dots_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_rocm,
          nk_dots_pack_e2m3_rocm, nk_dots_packed_e2m3_rocm);
    check("dots_pack_e2m3_rocm", test_dots_pack_layout<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_rocm,
                                                       nk_dots_packed_shape_e2m3_rocm, nk_dots_pack_e2m3_rocm>);
    check("dots_contract_e2m3_rocm", test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_rocm,
                                                               nk_dots_packed_e2m3_rocm, nk_dots_symmetric_e2m3_rocm>);
    check("dots_symmetric_e2m3_rocm", test_dots_symmetric<e2m3_t, rocm_backend_t>, nk_dots_symmetric_e2m3_rocm);
    check("dots_packed_e2m1_rocm", test_dots_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_rocm,
          nk_dots_pack_e2m1_rocm, nk_dots_packed_e2m1_rocm);
    check("dots_pack_e2m1_rocm", test_dots_pack_layout<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_rocm,
                                                       nk_dots_packed_shape_e2m1_rocm, nk_dots_pack_e2m1_rocm>);
    check("dots_contract_e2m1_rocm", test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_rocm,
                                                               nk_dots_packed_e2m1_rocm, nk_dots_symmetric_e2m1_rocm>);
    check("dots_symmetric_e2m1_rocm", test_dots_symmetric<e2m1x2_t, rocm_backend_t>, nk_dots_symmetric_e2m1_rocm);
    check("dots_packed_i8_rocm", test_dots_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_rocm,
          nk_dots_pack_i8_rocm, nk_dots_packed_i8_rocm);
    check("dots_pack_i8_rocm", test_dots_pack_layout<i8_t, rocm_backend_t, nk_dots_pack_size_i8_rocm,
                                                     nk_dots_packed_shape_i8_rocm, nk_dots_pack_i8_rocm>);
    check("dots_contract_i8_rocm", test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_rocm,
                                                             nk_dots_packed_i8_rocm, nk_dots_symmetric_i8_rocm>);
    check("dots_symmetric_i8_rocm", test_dots_symmetric<i8_t, rocm_backend_t>, nk_dots_symmetric_i8_rocm);
    check("dots_packed_u8_rocm", test_dots_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_rocm,
          nk_dots_pack_u8_rocm, nk_dots_packed_u8_rocm);
    check("dots_pack_u8_rocm", test_dots_pack_layout<u8_t, rocm_backend_t, nk_dots_pack_size_u8_rocm,
                                                     nk_dots_packed_shape_u8_rocm, nk_dots_pack_u8_rocm>);
    check("dots_contract_u8_rocm", test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_rocm,
                                                             nk_dots_packed_u8_rocm, nk_dots_symmetric_u8_rocm>);
    check("dots_symmetric_u8_rocm", test_dots_symmetric<u8_t, rocm_backend_t>, nk_dots_symmetric_u8_rocm);
    check("dots_packed_i4_rocm", test_dots_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_rocm,
          nk_dots_pack_i4_rocm, nk_dots_packed_i4_rocm);
    check("dots_pack_i4_rocm", test_dots_pack_layout<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_rocm,
                                                     nk_dots_packed_shape_i4_rocm, nk_dots_pack_i4_rocm>);
    check("dots_contract_i4_rocm", test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_rocm,
                                                             nk_dots_packed_i4_rocm, nk_dots_symmetric_i4_rocm>);
    check("dots_symmetric_i4_rocm", test_dots_symmetric<i4x2_t, rocm_backend_t>, nk_dots_symmetric_i4_rocm);
    check("dots_packed_u4_rocm", test_dots_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_rocm,
          nk_dots_pack_u4_rocm, nk_dots_packed_u4_rocm);
    check("dots_pack_u4_rocm", test_dots_pack_layout<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_rocm,
                                                     nk_dots_packed_shape_u4_rocm, nk_dots_pack_u4_rocm>);
    check("dots_contract_u4_rocm", test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_rocm,
                                                             nk_dots_packed_u4_rocm, nk_dots_symmetric_u4_rocm>);
    check("dots_symmetric_u4_rocm", test_dots_symmetric<u4x2_t, rocm_backend_t>, nk_dots_symmetric_u4_rocm);
    check("angulars_packed_f64_rocm", test_angulars_packed<f64_t, rocm_backend_t>, nk_dots_pack_size_f64_rocm,
          nk_dots_pack_f64_rocm, nk_angulars_packed_f64_rocm);
    check("angulars_symmetric_f64_rocm", test_angulars_symmetric<f64_t, rocm_backend_t>,
          nk_angulars_symmetric_f64_rocm);
    check("euclideans_packed_f64_rocm", test_euclideans_packed<f64_t, rocm_backend_t>, nk_dots_pack_size_f64_rocm,
          nk_dots_pack_f64_rocm, nk_euclideans_packed_f64_rocm);
    check("euclideans_symmetric_f64_rocm", test_euclideans_symmetric<f64_t, rocm_backend_t>,
          nk_euclideans_symmetric_f64_rocm);
    check("angulars_packed_f32_rocm", test_angulars_packed<f32_t, rocm_backend_t>, nk_dots_pack_size_f32_rocm,
          nk_dots_pack_f32_rocm, nk_angulars_packed_f32_rocm);
    check("angulars_symmetric_f32_rocm", test_angulars_symmetric<f32_t, rocm_backend_t>,
          nk_angulars_symmetric_f32_rocm);
    check("euclideans_packed_f32_rocm", test_euclideans_packed<f32_t, rocm_backend_t>, nk_dots_pack_size_f32_rocm,
          nk_dots_pack_f32_rocm, nk_euclideans_packed_f32_rocm);
    check("euclideans_symmetric_f32_rocm", test_euclideans_symmetric<f32_t, rocm_backend_t>,
          nk_euclideans_symmetric_f32_rocm);
    check("angulars_packed_bf16_rocm", test_angulars_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_rocm,
          nk_dots_pack_bf16_rocm, nk_angulars_packed_bf16_rocm);
    check("angulars_symmetric_bf16_rocm", test_angulars_symmetric<bf16_t, rocm_backend_t>,
          nk_angulars_symmetric_bf16_rocm);
    check("euclideans_packed_bf16_rocm", test_euclideans_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_rocm,
          nk_dots_pack_bf16_rocm, nk_euclideans_packed_bf16_rocm);
    check("euclideans_symmetric_bf16_rocm", test_euclideans_symmetric<bf16_t, rocm_backend_t>,
          nk_euclideans_symmetric_bf16_rocm);
    check("angulars_packed_f16_rocm", test_angulars_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_rocm,
          nk_dots_pack_f16_rocm, nk_angulars_packed_f16_rocm);
    check("angulars_symmetric_f16_rocm", test_angulars_symmetric<f16_t, rocm_backend_t>,
          nk_angulars_symmetric_f16_rocm);
    check("euclideans_packed_f16_rocm", test_euclideans_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_rocm,
          nk_dots_pack_f16_rocm, nk_euclideans_packed_f16_rocm);
    check("euclideans_symmetric_f16_rocm", test_euclideans_symmetric<f16_t, rocm_backend_t>,
          nk_euclideans_symmetric_f16_rocm);
    check("angulars_packed_e5m2_rocm", test_angulars_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_rocm,
          nk_dots_pack_e5m2_rocm, nk_angulars_packed_e5m2_rocm);
    check("angulars_symmetric_e5m2_rocm", test_angulars_symmetric<e5m2_t, rocm_backend_t>,
          nk_angulars_symmetric_e5m2_rocm);
    check("euclideans_packed_e5m2_rocm", test_euclideans_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_rocm,
          nk_dots_pack_e5m2_rocm, nk_euclideans_packed_e5m2_rocm);
    check("euclideans_symmetric_e5m2_rocm", test_euclideans_symmetric<e5m2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e5m2_rocm);
    check("angulars_packed_e4m3_rocm", test_angulars_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_rocm,
          nk_dots_pack_e4m3_rocm, nk_angulars_packed_e4m3_rocm);
    check("angulars_symmetric_e4m3_rocm", test_angulars_symmetric<e4m3_t, rocm_backend_t>,
          nk_angulars_symmetric_e4m3_rocm);
    check("euclideans_packed_e4m3_rocm", test_euclideans_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_rocm,
          nk_dots_pack_e4m3_rocm, nk_euclideans_packed_e4m3_rocm);
    check("euclideans_symmetric_e4m3_rocm", test_euclideans_symmetric<e4m3_t, rocm_backend_t>,
          nk_euclideans_symmetric_e4m3_rocm);
    check("angulars_packed_e3m2_rocm", test_angulars_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_rocm,
          nk_dots_pack_e3m2_rocm, nk_angulars_packed_e3m2_rocm);
    check("angulars_symmetric_e3m2_rocm", test_angulars_symmetric<e3m2_t, rocm_backend_t>,
          nk_angulars_symmetric_e3m2_rocm);
    check("euclideans_packed_e3m2_rocm", test_euclideans_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_rocm,
          nk_dots_pack_e3m2_rocm, nk_euclideans_packed_e3m2_rocm);
    check("euclideans_symmetric_e3m2_rocm", test_euclideans_symmetric<e3m2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e3m2_rocm);
    check("angulars_packed_e2m3_rocm", test_angulars_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_rocm,
          nk_dots_pack_e2m3_rocm, nk_angulars_packed_e2m3_rocm);
    check("angulars_symmetric_e2m3_rocm", test_angulars_symmetric<e2m3_t, rocm_backend_t>,
          nk_angulars_symmetric_e2m3_rocm);
    check("euclideans_packed_e2m3_rocm", test_euclideans_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_rocm,
          nk_dots_pack_e2m3_rocm, nk_euclideans_packed_e2m3_rocm);
    check("euclideans_symmetric_e2m3_rocm", test_euclideans_symmetric<e2m3_t, rocm_backend_t>,
          nk_euclideans_symmetric_e2m3_rocm);
    check("angulars_packed_e2m1_rocm", test_angulars_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_rocm,
          nk_dots_pack_e2m1_rocm, nk_angulars_packed_e2m1_rocm);
    check("angulars_symmetric_e2m1_rocm", test_angulars_symmetric<e2m1x2_t, rocm_backend_t>,
          nk_angulars_symmetric_e2m1_rocm);
    check("euclideans_packed_e2m1_rocm", test_euclideans_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_rocm,
          nk_dots_pack_e2m1_rocm, nk_euclideans_packed_e2m1_rocm);
    check("euclideans_symmetric_e2m1_rocm", test_euclideans_symmetric<e2m1x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e2m1_rocm);
    check("angulars_packed_i8_rocm", test_angulars_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_rocm,
          nk_dots_pack_i8_rocm, nk_angulars_packed_i8_rocm);
    check("angulars_symmetric_i8_rocm", test_angulars_symmetric<i8_t, rocm_backend_t>, nk_angulars_symmetric_i8_rocm);
    check("euclideans_packed_i8_rocm", test_euclideans_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_rocm,
          nk_dots_pack_i8_rocm, nk_euclideans_packed_i8_rocm);
    check("euclideans_symmetric_i8_rocm", test_euclideans_symmetric<i8_t, rocm_backend_t>,
          nk_euclideans_symmetric_i8_rocm);
    check("angulars_packed_u8_rocm", test_angulars_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_rocm,
          nk_dots_pack_u8_rocm, nk_angulars_packed_u8_rocm);
    check("angulars_symmetric_u8_rocm", test_angulars_symmetric<u8_t, rocm_backend_t>, nk_angulars_symmetric_u8_rocm);
    check("euclideans_packed_u8_rocm", test_euclideans_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_rocm,
          nk_dots_pack_u8_rocm, nk_euclideans_packed_u8_rocm);
    check("euclideans_symmetric_u8_rocm", test_euclideans_symmetric<u8_t, rocm_backend_t>,
          nk_euclideans_symmetric_u8_rocm);
    check("angulars_packed_i4_rocm", test_angulars_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_rocm,
          nk_dots_pack_i4_rocm, nk_angulars_packed_i4_rocm);
    check("angulars_symmetric_i4_rocm", test_angulars_symmetric<i4x2_t, rocm_backend_t>, nk_angulars_symmetric_i4_rocm);
    check("euclideans_packed_i4_rocm", test_euclideans_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_rocm,
          nk_dots_pack_i4_rocm, nk_euclideans_packed_i4_rocm);
    check("euclideans_symmetric_i4_rocm", test_euclideans_symmetric<i4x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_i4_rocm);
    check("angulars_packed_u4_rocm", test_angulars_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_rocm,
          nk_dots_pack_u4_rocm, nk_angulars_packed_u4_rocm);
    check("angulars_symmetric_u4_rocm", test_angulars_symmetric<u4x2_t, rocm_backend_t>, nk_angulars_symmetric_u4_rocm);
    check("euclideans_packed_u4_rocm", test_euclideans_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_rocm,
          nk_dots_pack_u4_rocm, nk_euclideans_packed_u4_rocm);
    check("euclideans_symmetric_u4_rocm", test_euclideans_symmetric<u4x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_u4_rocm);
    check("attention_packed_bf16_rocm", test_attention_packed<bf16_t, rocm_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_rocm, nk_attention_pack_bf16_rocm, nk_attention_packed_bf16_rocm);
    check("attention_packed_gradients_bf16_rocm", test_attention_packed_gradients<bf16_t, rocm_backend_t>,
          nk_attention_pack_size_bf16_rocm, nk_attention_pack_bf16_rocm, nk_attention_packed_gradients_bf16_rocm);
    check("attention_packed_e4m3_rocm", test_attention_packed<e4m3_t, rocm_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_rocm, nk_attention_pack_e4m3_rocm, nk_attention_packed_e4m3_rocm);
    check("attention_packed_i8_rocm", test_attention_packed<i8_t, rocm_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_rocm, nk_attention_pack_i8_rocm, nk_attention_packed_i8_rocm);

#if NUMKONG_TARGET_CDNA3
    check.section("Cross CDNA3", nk_cap_cdna3_k);
    check("dots_packed_f16_cdna3", test_dots_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna3,
          nk_dots_pack_f16_cdna3, nk_dots_packed_f16_cdna3);
    check("dots_pack_f16_cdna3", test_dots_pack_layout<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna3,
                                                       nk_dots_packed_shape_f16_cdna3, nk_dots_pack_f16_cdna3>);
    check("dots_contract_f16_cdna3", test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna3,
                                                               nk_dots_packed_f16_cdna3, nk_dots_symmetric_f16_cdna3>);
    check("dots_symmetric_f16_cdna3", test_dots_symmetric<f16_t, rocm_backend_t>, nk_dots_symmetric_f16_cdna3);
    check("dots_packed_e5m2_cdna3", test_dots_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna3,
          nk_dots_pack_e5m2_cdna3, nk_dots_packed_e5m2_cdna3);
    check("dots_pack_e5m2_cdna3", test_dots_pack_layout<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna3,
                                                        nk_dots_packed_shape_e5m2_cdna3, nk_dots_pack_e5m2_cdna3>);
    check("dots_contract_e5m2_cdna3",
          test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna3, nk_dots_packed_e5m2_cdna3,
                                    nk_dots_symmetric_e5m2_cdna3>);
    check("dots_symmetric_e5m2_cdna3", test_dots_symmetric<e5m2_t, rocm_backend_t>, nk_dots_symmetric_e5m2_cdna3);
    check("dots_packed_e4m3_cdna3", test_dots_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna3,
          nk_dots_pack_e4m3_cdna3, nk_dots_packed_e4m3_cdna3);
    check("dots_pack_e4m3_cdna3", test_dots_pack_layout<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna3,
                                                        nk_dots_packed_shape_e4m3_cdna3, nk_dots_pack_e4m3_cdna3>);
    check("dots_contract_e4m3_cdna3",
          test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna3, nk_dots_packed_e4m3_cdna3,
                                    nk_dots_symmetric_e4m3_cdna3>);
    check("dots_symmetric_e4m3_cdna3", test_dots_symmetric<e4m3_t, rocm_backend_t>, nk_dots_symmetric_e4m3_cdna3);
    check("dots_packed_e3m2_cdna3", test_dots_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna3,
          nk_dots_pack_e3m2_cdna3, nk_dots_packed_e3m2_cdna3);
    check("dots_pack_e3m2_cdna3", test_dots_pack_layout<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna3,
                                                        nk_dots_packed_shape_e3m2_cdna3, nk_dots_pack_e3m2_cdna3>);
    check("dots_contract_e3m2_cdna3",
          test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna3, nk_dots_packed_e3m2_cdna3,
                                    nk_dots_symmetric_e3m2_cdna3>);
    check("dots_symmetric_e3m2_cdna3", test_dots_symmetric<e3m2_t, rocm_backend_t>, nk_dots_symmetric_e3m2_cdna3);
    check("dots_packed_e2m3_cdna3", test_dots_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna3,
          nk_dots_pack_e2m3_cdna3, nk_dots_packed_e2m3_cdna3);
    check("dots_pack_e2m3_cdna3", test_dots_pack_layout<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna3,
                                                        nk_dots_packed_shape_e2m3_cdna3, nk_dots_pack_e2m3_cdna3>);
    check("dots_contract_e2m3_cdna3",
          test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna3, nk_dots_packed_e2m3_cdna3,
                                    nk_dots_symmetric_e2m3_cdna3>);
    check("dots_symmetric_e2m3_cdna3", test_dots_symmetric<e2m3_t, rocm_backend_t>, nk_dots_symmetric_e2m3_cdna3);
    check("dots_packed_e2m1_cdna3", test_dots_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_cdna3,
          nk_dots_pack_e2m1_cdna3, nk_dots_packed_e2m1_cdna3);
    check("dots_pack_e2m1_cdna3", test_dots_pack_layout<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna3,
                                                        nk_dots_packed_shape_e2m1_cdna3, nk_dots_pack_e2m1_cdna3>);
    check("dots_contract_e2m1_cdna3",
          test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna3, nk_dots_packed_e2m1_cdna3,
                                    nk_dots_symmetric_e2m1_cdna3>);
    check("dots_symmetric_e2m1_cdna3", test_dots_symmetric<e2m1x2_t, rocm_backend_t>, nk_dots_symmetric_e2m1_cdna3);
    check("dots_packed_i8_cdna3", test_dots_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna3,
          nk_dots_pack_i8_cdna3, nk_dots_packed_i8_cdna3);
    check("dots_pack_i8_cdna3", test_dots_pack_layout<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna3,
                                                      nk_dots_packed_shape_i8_cdna3, nk_dots_pack_i8_cdna3>);
    check("dots_contract_i8_cdna3", test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna3,
                                                              nk_dots_packed_i8_cdna3, nk_dots_symmetric_i8_cdna3>);
    check("dots_symmetric_i8_cdna3", test_dots_symmetric<i8_t, rocm_backend_t>, nk_dots_symmetric_i8_cdna3);
    check("dots_packed_i4_cdna3", test_dots_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna3,
          nk_dots_pack_i4_cdna3, nk_dots_packed_i4_cdna3);
    check("dots_pack_i4_cdna3", test_dots_pack_layout<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna3,
                                                      nk_dots_packed_shape_i4_cdna3, nk_dots_pack_i4_cdna3>);
    check("dots_contract_i4_cdna3", test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna3,
                                                              nk_dots_packed_i4_cdna3, nk_dots_symmetric_i4_cdna3>);
    check("dots_symmetric_i4_cdna3", test_dots_symmetric<i4x2_t, rocm_backend_t>, nk_dots_symmetric_i4_cdna3);
    check("angulars_packed_f16_cdna3", test_angulars_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna3,
          nk_dots_pack_f16_cdna3, nk_angulars_packed_f16_cdna3);
    check("angulars_symmetric_f16_cdna3", test_angulars_symmetric<f16_t, rocm_backend_t>,
          nk_angulars_symmetric_f16_cdna3);
    check("euclideans_packed_f16_cdna3", test_euclideans_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna3,
          nk_dots_pack_f16_cdna3, nk_euclideans_packed_f16_cdna3);
    check("euclideans_symmetric_f16_cdna3", test_euclideans_symmetric<f16_t, rocm_backend_t>,
          nk_euclideans_symmetric_f16_cdna3);
    check("angulars_packed_e5m2_cdna3", test_angulars_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna3,
          nk_dots_pack_e5m2_cdna3, nk_angulars_packed_e5m2_cdna3);
    check("angulars_symmetric_e5m2_cdna3", test_angulars_symmetric<e5m2_t, rocm_backend_t>,
          nk_angulars_symmetric_e5m2_cdna3);
    check("euclideans_packed_e5m2_cdna3", test_euclideans_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna3,
          nk_dots_pack_e5m2_cdna3, nk_euclideans_packed_e5m2_cdna3);
    check("euclideans_symmetric_e5m2_cdna3", test_euclideans_symmetric<e5m2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e5m2_cdna3);
    check("angulars_packed_e4m3_cdna3", test_angulars_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna3,
          nk_dots_pack_e4m3_cdna3, nk_angulars_packed_e4m3_cdna3);
    check("angulars_symmetric_e4m3_cdna3", test_angulars_symmetric<e4m3_t, rocm_backend_t>,
          nk_angulars_symmetric_e4m3_cdna3);
    check("euclideans_packed_e4m3_cdna3", test_euclideans_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna3,
          nk_dots_pack_e4m3_cdna3, nk_euclideans_packed_e4m3_cdna3);
    check("euclideans_symmetric_e4m3_cdna3", test_euclideans_symmetric<e4m3_t, rocm_backend_t>,
          nk_euclideans_symmetric_e4m3_cdna3);
    check("angulars_packed_e3m2_cdna3", test_angulars_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna3,
          nk_dots_pack_e3m2_cdna3, nk_angulars_packed_e3m2_cdna3);
    check("angulars_symmetric_e3m2_cdna3", test_angulars_symmetric<e3m2_t, rocm_backend_t>,
          nk_angulars_symmetric_e3m2_cdna3);
    check("euclideans_packed_e3m2_cdna3", test_euclideans_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna3,
          nk_dots_pack_e3m2_cdna3, nk_euclideans_packed_e3m2_cdna3);
    check("euclideans_symmetric_e3m2_cdna3", test_euclideans_symmetric<e3m2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e3m2_cdna3);
    check("angulars_packed_e2m3_cdna3", test_angulars_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna3,
          nk_dots_pack_e2m3_cdna3, nk_angulars_packed_e2m3_cdna3);
    check("angulars_symmetric_e2m3_cdna3", test_angulars_symmetric<e2m3_t, rocm_backend_t>,
          nk_angulars_symmetric_e2m3_cdna3);
    check("euclideans_packed_e2m3_cdna3", test_euclideans_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna3,
          nk_dots_pack_e2m3_cdna3, nk_euclideans_packed_e2m3_cdna3);
    check("euclideans_symmetric_e2m3_cdna3", test_euclideans_symmetric<e2m3_t, rocm_backend_t>,
          nk_euclideans_symmetric_e2m3_cdna3);
    check("angulars_packed_e2m1_cdna3", test_angulars_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_cdna3,
          nk_dots_pack_e2m1_cdna3, nk_angulars_packed_e2m1_cdna3);
    check("angulars_symmetric_e2m1_cdna3", test_angulars_symmetric<e2m1x2_t, rocm_backend_t>,
          nk_angulars_symmetric_e2m1_cdna3);
    check("euclideans_packed_e2m1_cdna3", test_euclideans_packed<e2m1x2_t, rocm_backend_t>,
          nk_dots_pack_size_e2m1_cdna3, nk_dots_pack_e2m1_cdna3, nk_euclideans_packed_e2m1_cdna3);
    check("euclideans_symmetric_e2m1_cdna3", test_euclideans_symmetric<e2m1x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e2m1_cdna3);
    check("angulars_packed_i8_cdna3", test_angulars_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna3,
          nk_dots_pack_i8_cdna3, nk_angulars_packed_i8_cdna3);
    check("angulars_symmetric_i8_cdna3", test_angulars_symmetric<i8_t, rocm_backend_t>, nk_angulars_symmetric_i8_cdna3);
    check("euclideans_packed_i8_cdna3", test_euclideans_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna3,
          nk_dots_pack_i8_cdna3, nk_euclideans_packed_i8_cdna3);
    check("euclideans_symmetric_i8_cdna3", test_euclideans_symmetric<i8_t, rocm_backend_t>,
          nk_euclideans_symmetric_i8_cdna3);
    check("angulars_packed_i4_cdna3", test_angulars_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna3,
          nk_dots_pack_i4_cdna3, nk_angulars_packed_i4_cdna3);
    check("angulars_symmetric_i4_cdna3", test_angulars_symmetric<i4x2_t, rocm_backend_t>,
          nk_angulars_symmetric_i4_cdna3);
    check("euclideans_packed_i4_cdna3", test_euclideans_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna3,
          nk_dots_pack_i4_cdna3, nk_euclideans_packed_i4_cdna3);
    check("euclideans_symmetric_i4_cdna3", test_euclideans_symmetric<i4x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_i4_cdna3);
#endif // NUMKONG_TARGET_CDNA3

#if NUMKONG_TARGET_CDNA4
    check.section("Cross CDNA4", nk_cap_cdna4_k);
    check("dots_packed_bf16_cdna4", test_dots_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_cdna4,
          nk_dots_pack_bf16_cdna4, nk_dots_packed_bf16_cdna4);
    check("dots_pack_bf16_cdna4", test_dots_pack_layout<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna4,
                                                        nk_dots_packed_shape_bf16_cdna4, nk_dots_pack_bf16_cdna4>);
    check("dots_contract_bf16_cdna4",
          test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna4, nk_dots_packed_bf16_cdna4,
                                    nk_dots_symmetric_bf16_cdna4>);
    check("dots_symmetric_bf16_cdna4", test_dots_symmetric<bf16_t, rocm_backend_t>, nk_dots_symmetric_bf16_cdna4);
    check("dots_packed_f16_cdna4", test_dots_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna4,
          nk_dots_pack_f16_cdna4, nk_dots_packed_f16_cdna4);
    check("dots_pack_f16_cdna4", test_dots_pack_layout<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna4,
                                                       nk_dots_packed_shape_f16_cdna4, nk_dots_pack_f16_cdna4>);
    check("dots_contract_f16_cdna4", test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna4,
                                                               nk_dots_packed_f16_cdna4, nk_dots_symmetric_f16_cdna4>);
    check("dots_symmetric_f16_cdna4", test_dots_symmetric<f16_t, rocm_backend_t>, nk_dots_symmetric_f16_cdna4);
    check("dots_packed_e5m2_cdna4", test_dots_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna4,
          nk_dots_pack_e5m2_cdna4, nk_dots_packed_e5m2_cdna4);
    check("dots_pack_e5m2_cdna4", test_dots_pack_layout<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna4,
                                                        nk_dots_packed_shape_e5m2_cdna4, nk_dots_pack_e5m2_cdna4>);
    check("dots_contract_e5m2_cdna4",
          test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna4, nk_dots_packed_e5m2_cdna4,
                                    nk_dots_symmetric_e5m2_cdna4>);
    check("dots_symmetric_e5m2_cdna4", test_dots_symmetric<e5m2_t, rocm_backend_t>, nk_dots_symmetric_e5m2_cdna4);
    check("dots_packed_e4m3_cdna4", test_dots_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna4,
          nk_dots_pack_e4m3_cdna4, nk_dots_packed_e4m3_cdna4);
    check("dots_pack_e4m3_cdna4", test_dots_pack_layout<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna4,
                                                        nk_dots_packed_shape_e4m3_cdna4, nk_dots_pack_e4m3_cdna4>);
    check("dots_contract_e4m3_cdna4",
          test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna4, nk_dots_packed_e4m3_cdna4,
                                    nk_dots_symmetric_e4m3_cdna4>);
    check("dots_symmetric_e4m3_cdna4", test_dots_symmetric<e4m3_t, rocm_backend_t>, nk_dots_symmetric_e4m3_cdna4);
    check("dots_packed_e3m2_cdna4", test_dots_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna4,
          nk_dots_pack_e3m2_cdna4, nk_dots_packed_e3m2_cdna4);
    check("dots_pack_e3m2_cdna4", test_dots_pack_layout<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna4,
                                                        nk_dots_packed_shape_e3m2_cdna4, nk_dots_pack_e3m2_cdna4>);
    check("dots_contract_e3m2_cdna4",
          test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna4, nk_dots_packed_e3m2_cdna4,
                                    nk_dots_symmetric_e3m2_cdna4>);
    check("dots_symmetric_e3m2_cdna4", test_dots_symmetric<e3m2_t, rocm_backend_t>, nk_dots_symmetric_e3m2_cdna4);
    check("dots_packed_e2m3_cdna4", test_dots_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna4,
          nk_dots_pack_e2m3_cdna4, nk_dots_packed_e2m3_cdna4);
    check("dots_pack_e2m3_cdna4", test_dots_pack_layout<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna4,
                                                        nk_dots_packed_shape_e2m3_cdna4, nk_dots_pack_e2m3_cdna4>);
    check("dots_contract_e2m3_cdna4",
          test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna4, nk_dots_packed_e2m3_cdna4,
                                    nk_dots_symmetric_e2m3_cdna4>);
    check("dots_symmetric_e2m3_cdna4", test_dots_symmetric<e2m3_t, rocm_backend_t>, nk_dots_symmetric_e2m3_cdna4);
    check("dots_packed_e2m1_cdna4", test_dots_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_cdna4,
          nk_dots_pack_e2m1_cdna4, nk_dots_packed_e2m1_cdna4);
    check("dots_pack_e2m1_cdna4", test_dots_pack_layout<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna4,
                                                        nk_dots_packed_shape_e2m1_cdna4, nk_dots_pack_e2m1_cdna4>);
    check("dots_contract_e2m1_cdna4",
          test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna4, nk_dots_packed_e2m1_cdna4,
                                    nk_dots_symmetric_e2m1_cdna4>);
    check("dots_symmetric_e2m1_cdna4", test_dots_symmetric<e2m1x2_t, rocm_backend_t>, nk_dots_symmetric_e2m1_cdna4);
    check("dots_packed_i8_cdna4", test_dots_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna4,
          nk_dots_pack_i8_cdna4, nk_dots_packed_i8_cdna4);
    check("dots_pack_i8_cdna4", test_dots_pack_layout<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna4,
                                                      nk_dots_packed_shape_i8_cdna4, nk_dots_pack_i8_cdna4>);
    check("dots_contract_i8_cdna4", test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna4,
                                                              nk_dots_packed_i8_cdna4, nk_dots_symmetric_i8_cdna4>);
    check("dots_symmetric_i8_cdna4", test_dots_symmetric<i8_t, rocm_backend_t>, nk_dots_symmetric_i8_cdna4);
    check("dots_packed_u8_cdna4", test_dots_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_cdna4,
          nk_dots_pack_u8_cdna4, nk_dots_packed_u8_cdna4);
    check("dots_pack_u8_cdna4", test_dots_pack_layout<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna4,
                                                      nk_dots_packed_shape_u8_cdna4, nk_dots_pack_u8_cdna4>);
    check("dots_contract_u8_cdna4", test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna4,
                                                              nk_dots_packed_u8_cdna4, nk_dots_symmetric_u8_cdna4>);
    check("dots_symmetric_u8_cdna4", test_dots_symmetric<u8_t, rocm_backend_t>, nk_dots_symmetric_u8_cdna4);
    check("dots_packed_i4_cdna4", test_dots_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna4,
          nk_dots_pack_i4_cdna4, nk_dots_packed_i4_cdna4);
    check("dots_pack_i4_cdna4", test_dots_pack_layout<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna4,
                                                      nk_dots_packed_shape_i4_cdna4, nk_dots_pack_i4_cdna4>);
    check("dots_contract_i4_cdna4", test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna4,
                                                              nk_dots_packed_i4_cdna4, nk_dots_symmetric_i4_cdna4>);
    check("dots_symmetric_i4_cdna4", test_dots_symmetric<i4x2_t, rocm_backend_t>, nk_dots_symmetric_i4_cdna4);
    check("dots_packed_u4_cdna4", test_dots_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_cdna4,
          nk_dots_pack_u4_cdna4, nk_dots_packed_u4_cdna4);
    check("dots_pack_u4_cdna4", test_dots_pack_layout<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna4,
                                                      nk_dots_packed_shape_u4_cdna4, nk_dots_pack_u4_cdna4>);
    check("dots_contract_u4_cdna4", test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna4,
                                                              nk_dots_packed_u4_cdna4, nk_dots_symmetric_u4_cdna4>);
    check("dots_symmetric_u4_cdna4", test_dots_symmetric<u4x2_t, rocm_backend_t>, nk_dots_symmetric_u4_cdna4);
    check("angulars_packed_bf16_cdna4", test_angulars_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_cdna4,
          nk_dots_pack_bf16_cdna4, nk_angulars_packed_bf16_cdna4);
    check("angulars_symmetric_bf16_cdna4", test_angulars_symmetric<bf16_t, rocm_backend_t>,
          nk_angulars_symmetric_bf16_cdna4);
    check("euclideans_packed_bf16_cdna4", test_euclideans_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_cdna4,
          nk_dots_pack_bf16_cdna4, nk_euclideans_packed_bf16_cdna4);
    check("euclideans_symmetric_bf16_cdna4", test_euclideans_symmetric<bf16_t, rocm_backend_t>,
          nk_euclideans_symmetric_bf16_cdna4);
    check("angulars_packed_f16_cdna4", test_angulars_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna4,
          nk_dots_pack_f16_cdna4, nk_angulars_packed_f16_cdna4);
    check("angulars_symmetric_f16_cdna4", test_angulars_symmetric<f16_t, rocm_backend_t>,
          nk_angulars_symmetric_f16_cdna4);
    check("euclideans_packed_f16_cdna4", test_euclideans_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna4,
          nk_dots_pack_f16_cdna4, nk_euclideans_packed_f16_cdna4);
    check("euclideans_symmetric_f16_cdna4", test_euclideans_symmetric<f16_t, rocm_backend_t>,
          nk_euclideans_symmetric_f16_cdna4);
    check("angulars_packed_e5m2_cdna4", test_angulars_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna4,
          nk_dots_pack_e5m2_cdna4, nk_angulars_packed_e5m2_cdna4);
    check("angulars_symmetric_e5m2_cdna4", test_angulars_symmetric<e5m2_t, rocm_backend_t>,
          nk_angulars_symmetric_e5m2_cdna4);
    check("euclideans_packed_e5m2_cdna4", test_euclideans_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna4,
          nk_dots_pack_e5m2_cdna4, nk_euclideans_packed_e5m2_cdna4);
    check("euclideans_symmetric_e5m2_cdna4", test_euclideans_symmetric<e5m2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e5m2_cdna4);
    check("angulars_packed_e4m3_cdna4", test_angulars_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna4,
          nk_dots_pack_e4m3_cdna4, nk_angulars_packed_e4m3_cdna4);
    check("angulars_symmetric_e4m3_cdna4", test_angulars_symmetric<e4m3_t, rocm_backend_t>,
          nk_angulars_symmetric_e4m3_cdna4);
    check("euclideans_packed_e4m3_cdna4", test_euclideans_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna4,
          nk_dots_pack_e4m3_cdna4, nk_euclideans_packed_e4m3_cdna4);
    check("euclideans_symmetric_e4m3_cdna4", test_euclideans_symmetric<e4m3_t, rocm_backend_t>,
          nk_euclideans_symmetric_e4m3_cdna4);
    check("angulars_packed_e3m2_cdna4", test_angulars_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna4,
          nk_dots_pack_e3m2_cdna4, nk_angulars_packed_e3m2_cdna4);
    check("angulars_symmetric_e3m2_cdna4", test_angulars_symmetric<e3m2_t, rocm_backend_t>,
          nk_angulars_symmetric_e3m2_cdna4);
    check("euclideans_packed_e3m2_cdna4", test_euclideans_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna4,
          nk_dots_pack_e3m2_cdna4, nk_euclideans_packed_e3m2_cdna4);
    check("euclideans_symmetric_e3m2_cdna4", test_euclideans_symmetric<e3m2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e3m2_cdna4);
    check("angulars_packed_e2m3_cdna4", test_angulars_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna4,
          nk_dots_pack_e2m3_cdna4, nk_angulars_packed_e2m3_cdna4);
    check("angulars_symmetric_e2m3_cdna4", test_angulars_symmetric<e2m3_t, rocm_backend_t>,
          nk_angulars_symmetric_e2m3_cdna4);
    check("euclideans_packed_e2m3_cdna4", test_euclideans_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna4,
          nk_dots_pack_e2m3_cdna4, nk_euclideans_packed_e2m3_cdna4);
    check("euclideans_symmetric_e2m3_cdna4", test_euclideans_symmetric<e2m3_t, rocm_backend_t>,
          nk_euclideans_symmetric_e2m3_cdna4);
    check("angulars_packed_e2m1_cdna4", test_angulars_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_cdna4,
          nk_dots_pack_e2m1_cdna4, nk_angulars_packed_e2m1_cdna4);
    check("angulars_symmetric_e2m1_cdna4", test_angulars_symmetric<e2m1x2_t, rocm_backend_t>,
          nk_angulars_symmetric_e2m1_cdna4);
    check("euclideans_packed_e2m1_cdna4", test_euclideans_packed<e2m1x2_t, rocm_backend_t>,
          nk_dots_pack_size_e2m1_cdna4, nk_dots_pack_e2m1_cdna4, nk_euclideans_packed_e2m1_cdna4);
    check("euclideans_symmetric_e2m1_cdna4", test_euclideans_symmetric<e2m1x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e2m1_cdna4);
    check("angulars_packed_i8_cdna4", test_angulars_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna4,
          nk_dots_pack_i8_cdna4, nk_angulars_packed_i8_cdna4);
    check("angulars_symmetric_i8_cdna4", test_angulars_symmetric<i8_t, rocm_backend_t>, nk_angulars_symmetric_i8_cdna4);
    check("euclideans_packed_i8_cdna4", test_euclideans_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna4,
          nk_dots_pack_i8_cdna4, nk_euclideans_packed_i8_cdna4);
    check("euclideans_symmetric_i8_cdna4", test_euclideans_symmetric<i8_t, rocm_backend_t>,
          nk_euclideans_symmetric_i8_cdna4);
    check("angulars_packed_u8_cdna4", test_angulars_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_cdna4,
          nk_dots_pack_u8_cdna4, nk_angulars_packed_u8_cdna4);
    check("angulars_symmetric_u8_cdna4", test_angulars_symmetric<u8_t, rocm_backend_t>, nk_angulars_symmetric_u8_cdna4);
    check("euclideans_packed_u8_cdna4", test_euclideans_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_cdna4,
          nk_dots_pack_u8_cdna4, nk_euclideans_packed_u8_cdna4);
    check("euclideans_symmetric_u8_cdna4", test_euclideans_symmetric<u8_t, rocm_backend_t>,
          nk_euclideans_symmetric_u8_cdna4);
    check("angulars_packed_i4_cdna4", test_angulars_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna4,
          nk_dots_pack_i4_cdna4, nk_angulars_packed_i4_cdna4);
    check("angulars_symmetric_i4_cdna4", test_angulars_symmetric<i4x2_t, rocm_backend_t>,
          nk_angulars_symmetric_i4_cdna4);
    check("euclideans_packed_i4_cdna4", test_euclideans_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna4,
          nk_dots_pack_i4_cdna4, nk_euclideans_packed_i4_cdna4);
    check("euclideans_symmetric_i4_cdna4", test_euclideans_symmetric<i4x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_i4_cdna4);
    check("angulars_packed_u4_cdna4", test_angulars_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_cdna4,
          nk_dots_pack_u4_cdna4, nk_angulars_packed_u4_cdna4);
    check("angulars_symmetric_u4_cdna4", test_angulars_symmetric<u4x2_t, rocm_backend_t>,
          nk_angulars_symmetric_u4_cdna4);
    check("euclideans_packed_u4_cdna4", test_euclideans_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_cdna4,
          nk_dots_pack_u4_cdna4, nk_euclideans_packed_u4_cdna4);
    check("euclideans_symmetric_u4_cdna4", test_euclideans_symmetric<u4x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_u4_cdna4);
    check("attention_packed_bf16_cdna4", test_attention_packed<bf16_t, rocm_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_cdna4, nk_attention_pack_bf16_cdna4, nk_attention_packed_bf16_cdna4);
    check("attention_packed_gradients_bf16_cdna4", test_attention_packed_gradients<bf16_t, rocm_backend_t>,
          nk_attention_pack_size_bf16_cdna4, nk_attention_pack_bf16_cdna4, nk_attention_packed_gradients_bf16_cdna4);
    check("attention_packed_e4m3_cdna4", test_attention_packed<e4m3_t, rocm_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_cdna4, nk_attention_pack_e4m3_cdna4, nk_attention_packed_e4m3_cdna4);
    check("attention_packed_i8_cdna4", test_attention_packed<i8_t, rocm_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_cdna4, nk_attention_pack_i8_cdna4, nk_attention_packed_i8_cdna4);
#endif // NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5
    check.section("Cross CDNA5", nk_cap_cdna5_k);
    check("dots_packed_bf16_cdna5", test_dots_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_cdna5,
          nk_dots_pack_bf16_cdna5, nk_dots_packed_bf16_cdna5);
    check("dots_pack_bf16_cdna5", test_dots_pack_layout<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna5,
                                                        nk_dots_packed_shape_bf16_cdna5, nk_dots_pack_bf16_cdna5>);
    check("dots_contract_bf16_cdna5",
          test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna5, nk_dots_packed_bf16_cdna5,
                                    nk_dots_symmetric_bf16_cdna5>);
    check("dots_symmetric_bf16_cdna5", test_dots_symmetric<bf16_t, rocm_backend_t>, nk_dots_symmetric_bf16_cdna5);
    check("dots_packed_f16_cdna5", test_dots_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna5,
          nk_dots_pack_f16_cdna5, nk_dots_packed_f16_cdna5);
    check("dots_pack_f16_cdna5", test_dots_pack_layout<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna5,
                                                       nk_dots_packed_shape_f16_cdna5, nk_dots_pack_f16_cdna5>);
    check("dots_contract_f16_cdna5", test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna5,
                                                               nk_dots_packed_f16_cdna5, nk_dots_symmetric_f16_cdna5>);
    check("dots_symmetric_f16_cdna5", test_dots_symmetric<f16_t, rocm_backend_t>, nk_dots_symmetric_f16_cdna5);
    check("dots_packed_e5m2_cdna5", test_dots_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna5,
          nk_dots_pack_e5m2_cdna5, nk_dots_packed_e5m2_cdna5);
    check("dots_pack_e5m2_cdna5", test_dots_pack_layout<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna5,
                                                        nk_dots_packed_shape_e5m2_cdna5, nk_dots_pack_e5m2_cdna5>);
    check("dots_contract_e5m2_cdna5",
          test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna5, nk_dots_packed_e5m2_cdna5,
                                    nk_dots_symmetric_e5m2_cdna5>);
    check("dots_symmetric_e5m2_cdna5", test_dots_symmetric<e5m2_t, rocm_backend_t>, nk_dots_symmetric_e5m2_cdna5);
    check("dots_packed_e4m3_cdna5", test_dots_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna5,
          nk_dots_pack_e4m3_cdna5, nk_dots_packed_e4m3_cdna5);
    check("dots_pack_e4m3_cdna5", test_dots_pack_layout<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna5,
                                                        nk_dots_packed_shape_e4m3_cdna5, nk_dots_pack_e4m3_cdna5>);
    check("dots_contract_e4m3_cdna5",
          test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna5, nk_dots_packed_e4m3_cdna5,
                                    nk_dots_symmetric_e4m3_cdna5>);
    check("dots_symmetric_e4m3_cdna5", test_dots_symmetric<e4m3_t, rocm_backend_t>, nk_dots_symmetric_e4m3_cdna5);
    check("dots_packed_e3m2_cdna5", test_dots_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna5,
          nk_dots_pack_e3m2_cdna5, nk_dots_packed_e3m2_cdna5);
    check("dots_pack_e3m2_cdna5", test_dots_pack_layout<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna5,
                                                        nk_dots_packed_shape_e3m2_cdna5, nk_dots_pack_e3m2_cdna5>);
    check("dots_contract_e3m2_cdna5",
          test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna5, nk_dots_packed_e3m2_cdna5,
                                    nk_dots_symmetric_e3m2_cdna5>);
    check("dots_symmetric_e3m2_cdna5", test_dots_symmetric<e3m2_t, rocm_backend_t>, nk_dots_symmetric_e3m2_cdna5);
    check("dots_packed_e2m3_cdna5", test_dots_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna5,
          nk_dots_pack_e2m3_cdna5, nk_dots_packed_e2m3_cdna5);
    check("dots_pack_e2m3_cdna5", test_dots_pack_layout<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna5,
                                                        nk_dots_packed_shape_e2m3_cdna5, nk_dots_pack_e2m3_cdna5>);
    check("dots_contract_e2m3_cdna5",
          test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna5, nk_dots_packed_e2m3_cdna5,
                                    nk_dots_symmetric_e2m3_cdna5>);
    check("dots_symmetric_e2m3_cdna5", test_dots_symmetric<e2m3_t, rocm_backend_t>, nk_dots_symmetric_e2m3_cdna5);
    check("dots_packed_e2m1_cdna5", test_dots_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_cdna5,
          nk_dots_pack_e2m1_cdna5, nk_dots_packed_e2m1_cdna5);
    check("dots_pack_e2m1_cdna5", test_dots_pack_layout<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna5,
                                                        nk_dots_packed_shape_e2m1_cdna5, nk_dots_pack_e2m1_cdna5>);
    check("dots_contract_e2m1_cdna5",
          test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna5, nk_dots_packed_e2m1_cdna5,
                                    nk_dots_symmetric_e2m1_cdna5>);
    check("dots_symmetric_e2m1_cdna5", test_dots_symmetric<e2m1x2_t, rocm_backend_t>, nk_dots_symmetric_e2m1_cdna5);
    check("dots_packed_i8_cdna5", test_dots_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna5,
          nk_dots_pack_i8_cdna5, nk_dots_packed_i8_cdna5);
    check("dots_pack_i8_cdna5", test_dots_pack_layout<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna5,
                                                      nk_dots_packed_shape_i8_cdna5, nk_dots_pack_i8_cdna5>);
    check("dots_contract_i8_cdna5", test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna5,
                                                              nk_dots_packed_i8_cdna5, nk_dots_symmetric_i8_cdna5>);
    check("dots_symmetric_i8_cdna5", test_dots_symmetric<i8_t, rocm_backend_t>, nk_dots_symmetric_i8_cdna5);
    check("dots_packed_u8_cdna5", test_dots_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_cdna5,
          nk_dots_pack_u8_cdna5, nk_dots_packed_u8_cdna5);
    check("dots_pack_u8_cdna5", test_dots_pack_layout<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna5,
                                                      nk_dots_packed_shape_u8_cdna5, nk_dots_pack_u8_cdna5>);
    check("dots_contract_u8_cdna5", test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna5,
                                                              nk_dots_packed_u8_cdna5, nk_dots_symmetric_u8_cdna5>);
    check("dots_symmetric_u8_cdna5", test_dots_symmetric<u8_t, rocm_backend_t>, nk_dots_symmetric_u8_cdna5);
    check("dots_packed_i4_cdna5", test_dots_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna5,
          nk_dots_pack_i4_cdna5, nk_dots_packed_i4_cdna5);
    check("dots_pack_i4_cdna5", test_dots_pack_layout<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna5,
                                                      nk_dots_packed_shape_i4_cdna5, nk_dots_pack_i4_cdna5>);
    check("dots_contract_i4_cdna5", test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna5,
                                                              nk_dots_packed_i4_cdna5, nk_dots_symmetric_i4_cdna5>);
    check("dots_symmetric_i4_cdna5", test_dots_symmetric<i4x2_t, rocm_backend_t>, nk_dots_symmetric_i4_cdna5);
    check("dots_packed_u4_cdna5", test_dots_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_cdna5,
          nk_dots_pack_u4_cdna5, nk_dots_packed_u4_cdna5);
    check("dots_pack_u4_cdna5", test_dots_pack_layout<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna5,
                                                      nk_dots_packed_shape_u4_cdna5, nk_dots_pack_u4_cdna5>);
    check("dots_contract_u4_cdna5", test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna5,
                                                              nk_dots_packed_u4_cdna5, nk_dots_symmetric_u4_cdna5>);
    check("dots_symmetric_u4_cdna5", test_dots_symmetric<u4x2_t, rocm_backend_t>, nk_dots_symmetric_u4_cdna5);
    check("angulars_packed_bf16_cdna5", test_angulars_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_cdna5,
          nk_dots_pack_bf16_cdna5, nk_angulars_packed_bf16_cdna5);
    check("angulars_symmetric_bf16_cdna5", test_angulars_symmetric<bf16_t, rocm_backend_t>,
          nk_angulars_symmetric_bf16_cdna5);
    check("euclideans_packed_bf16_cdna5", test_euclideans_packed<bf16_t, rocm_backend_t>, nk_dots_pack_size_bf16_cdna5,
          nk_dots_pack_bf16_cdna5, nk_euclideans_packed_bf16_cdna5);
    check("euclideans_symmetric_bf16_cdna5", test_euclideans_symmetric<bf16_t, rocm_backend_t>,
          nk_euclideans_symmetric_bf16_cdna5);
    check("angulars_packed_f16_cdna5", test_angulars_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna5,
          nk_dots_pack_f16_cdna5, nk_angulars_packed_f16_cdna5);
    check("angulars_symmetric_f16_cdna5", test_angulars_symmetric<f16_t, rocm_backend_t>,
          nk_angulars_symmetric_f16_cdna5);
    check("euclideans_packed_f16_cdna5", test_euclideans_packed<f16_t, rocm_backend_t>, nk_dots_pack_size_f16_cdna5,
          nk_dots_pack_f16_cdna5, nk_euclideans_packed_f16_cdna5);
    check("euclideans_symmetric_f16_cdna5", test_euclideans_symmetric<f16_t, rocm_backend_t>,
          nk_euclideans_symmetric_f16_cdna5);
    check("angulars_packed_e5m2_cdna5", test_angulars_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna5,
          nk_dots_pack_e5m2_cdna5, nk_angulars_packed_e5m2_cdna5);
    check("angulars_symmetric_e5m2_cdna5", test_angulars_symmetric<e5m2_t, rocm_backend_t>,
          nk_angulars_symmetric_e5m2_cdna5);
    check("euclideans_packed_e5m2_cdna5", test_euclideans_packed<e5m2_t, rocm_backend_t>, nk_dots_pack_size_e5m2_cdna5,
          nk_dots_pack_e5m2_cdna5, nk_euclideans_packed_e5m2_cdna5);
    check("euclideans_symmetric_e5m2_cdna5", test_euclideans_symmetric<e5m2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e5m2_cdna5);
    check("angulars_packed_e4m3_cdna5", test_angulars_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna5,
          nk_dots_pack_e4m3_cdna5, nk_angulars_packed_e4m3_cdna5);
    check("angulars_symmetric_e4m3_cdna5", test_angulars_symmetric<e4m3_t, rocm_backend_t>,
          nk_angulars_symmetric_e4m3_cdna5);
    check("euclideans_packed_e4m3_cdna5", test_euclideans_packed<e4m3_t, rocm_backend_t>, nk_dots_pack_size_e4m3_cdna5,
          nk_dots_pack_e4m3_cdna5, nk_euclideans_packed_e4m3_cdna5);
    check("euclideans_symmetric_e4m3_cdna5", test_euclideans_symmetric<e4m3_t, rocm_backend_t>,
          nk_euclideans_symmetric_e4m3_cdna5);
    check("angulars_packed_e3m2_cdna5", test_angulars_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna5,
          nk_dots_pack_e3m2_cdna5, nk_angulars_packed_e3m2_cdna5);
    check("angulars_symmetric_e3m2_cdna5", test_angulars_symmetric<e3m2_t, rocm_backend_t>,
          nk_angulars_symmetric_e3m2_cdna5);
    check("euclideans_packed_e3m2_cdna5", test_euclideans_packed<e3m2_t, rocm_backend_t>, nk_dots_pack_size_e3m2_cdna5,
          nk_dots_pack_e3m2_cdna5, nk_euclideans_packed_e3m2_cdna5);
    check("euclideans_symmetric_e3m2_cdna5", test_euclideans_symmetric<e3m2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e3m2_cdna5);
    check("angulars_packed_e2m3_cdna5", test_angulars_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna5,
          nk_dots_pack_e2m3_cdna5, nk_angulars_packed_e2m3_cdna5);
    check("angulars_symmetric_e2m3_cdna5", test_angulars_symmetric<e2m3_t, rocm_backend_t>,
          nk_angulars_symmetric_e2m3_cdna5);
    check("euclideans_packed_e2m3_cdna5", test_euclideans_packed<e2m3_t, rocm_backend_t>, nk_dots_pack_size_e2m3_cdna5,
          nk_dots_pack_e2m3_cdna5, nk_euclideans_packed_e2m3_cdna5);
    check("euclideans_symmetric_e2m3_cdna5", test_euclideans_symmetric<e2m3_t, rocm_backend_t>,
          nk_euclideans_symmetric_e2m3_cdna5);
    check("angulars_packed_e2m1_cdna5", test_angulars_packed<e2m1x2_t, rocm_backend_t>, nk_dots_pack_size_e2m1_cdna5,
          nk_dots_pack_e2m1_cdna5, nk_angulars_packed_e2m1_cdna5);
    check("angulars_symmetric_e2m1_cdna5", test_angulars_symmetric<e2m1x2_t, rocm_backend_t>,
          nk_angulars_symmetric_e2m1_cdna5);
    check("euclideans_packed_e2m1_cdna5", test_euclideans_packed<e2m1x2_t, rocm_backend_t>,
          nk_dots_pack_size_e2m1_cdna5, nk_dots_pack_e2m1_cdna5, nk_euclideans_packed_e2m1_cdna5);
    check("euclideans_symmetric_e2m1_cdna5", test_euclideans_symmetric<e2m1x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_e2m1_cdna5);
    check("angulars_packed_i8_cdna5", test_angulars_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna5,
          nk_dots_pack_i8_cdna5, nk_angulars_packed_i8_cdna5);
    check("angulars_symmetric_i8_cdna5", test_angulars_symmetric<i8_t, rocm_backend_t>, nk_angulars_symmetric_i8_cdna5);
    check("euclideans_packed_i8_cdna5", test_euclideans_packed<i8_t, rocm_backend_t>, nk_dots_pack_size_i8_cdna5,
          nk_dots_pack_i8_cdna5, nk_euclideans_packed_i8_cdna5);
    check("euclideans_symmetric_i8_cdna5", test_euclideans_symmetric<i8_t, rocm_backend_t>,
          nk_euclideans_symmetric_i8_cdna5);
    check("angulars_packed_u8_cdna5", test_angulars_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_cdna5,
          nk_dots_pack_u8_cdna5, nk_angulars_packed_u8_cdna5);
    check("angulars_symmetric_u8_cdna5", test_angulars_symmetric<u8_t, rocm_backend_t>, nk_angulars_symmetric_u8_cdna5);
    check("euclideans_packed_u8_cdna5", test_euclideans_packed<u8_t, rocm_backend_t>, nk_dots_pack_size_u8_cdna5,
          nk_dots_pack_u8_cdna5, nk_euclideans_packed_u8_cdna5);
    check("euclideans_symmetric_u8_cdna5", test_euclideans_symmetric<u8_t, rocm_backend_t>,
          nk_euclideans_symmetric_u8_cdna5);
    check("angulars_packed_i4_cdna5", test_angulars_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna5,
          nk_dots_pack_i4_cdna5, nk_angulars_packed_i4_cdna5);
    check("angulars_symmetric_i4_cdna5", test_angulars_symmetric<i4x2_t, rocm_backend_t>,
          nk_angulars_symmetric_i4_cdna5);
    check("euclideans_packed_i4_cdna5", test_euclideans_packed<i4x2_t, rocm_backend_t>, nk_dots_pack_size_i4_cdna5,
          nk_dots_pack_i4_cdna5, nk_euclideans_packed_i4_cdna5);
    check("euclideans_symmetric_i4_cdna5", test_euclideans_symmetric<i4x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_i4_cdna5);
    check("angulars_packed_u4_cdna5", test_angulars_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_cdna5,
          nk_dots_pack_u4_cdna5, nk_angulars_packed_u4_cdna5);
    check("angulars_symmetric_u4_cdna5", test_angulars_symmetric<u4x2_t, rocm_backend_t>,
          nk_angulars_symmetric_u4_cdna5);
    check("euclideans_packed_u4_cdna5", test_euclideans_packed<u4x2_t, rocm_backend_t>, nk_dots_pack_size_u4_cdna5,
          nk_dots_pack_u4_cdna5, nk_euclideans_packed_u4_cdna5);
    check("euclideans_symmetric_u4_cdna5", test_euclideans_symmetric<u4x2_t, rocm_backend_t>,
          nk_euclideans_symmetric_u4_cdna5);
    check("attention_packed_bf16_cdna5", test_attention_packed<bf16_t, rocm_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_cdna5, nk_attention_pack_bf16_cdna5, nk_attention_packed_bf16_cdna5);
    check("attention_packed_gradients_bf16_cdna5", test_attention_packed_gradients<bf16_t, rocm_backend_t>,
          nk_attention_pack_size_bf16_cdna5, nk_attention_pack_bf16_cdna5, nk_attention_packed_gradients_bf16_cdna5);
    check("attention_packed_e4m3_cdna5", test_attention_packed<e4m3_t, rocm_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_cdna5, nk_attention_pack_e4m3_cdna5, nk_attention_packed_e4m3_cdna5);
    check("attention_packed_i8_cdna5", test_attention_packed<i8_t, rocm_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_cdna5, nk_attention_pack_i8_cdna5, nk_attention_packed_i8_cdna5);
#endif // NUMKONG_TARGET_CDNA5

    test_cross_dispatch<rocm_backend_t>(check);
}

#else  // !NUMKONG_ARCH_ROCM_
void test_cross_rocm(error_stats_section_t &) {}
#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::test
