/**
 *  @file test/cross_amx.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Batch operation tests - AMX ISA family, Sapphire Rapids AMX.
 */
#include "harness.hpp"
#include "cross.cuh"

using namespace ashvardanian::numkong::test;

void test_cross_amx() {
    [[maybe_unused]] error_stats_section_t check;

#if NUMKONG_TARGET_SAPPHIREAMX
    check.section("Cross Sapphire AMX", nk_cap_sapphireamx_k);
    check("dots_packed_bf16_sapphireamx", test_dots_packed<bf16_t>, nk_dots_pack_size_bf16_sapphireamx,
          nk_dots_pack_bf16_sapphireamx, nk_dots_packed_bf16_sapphireamx);
    check("dots_pack_bf16_sapphireamx",
          test_dots_pack_layout<bf16_t, host_backend_t, nk_dots_pack_size_bf16_sapphireamx,
                                nk_dots_packed_shape_bf16_sapphireamx, nk_dots_pack_bf16_sapphireamx>);
    check("dots_packed_e5m2_sapphireamx", test_dots_packed<e5m2_t>, nk_dots_pack_size_e5m2_sapphireamx,
          nk_dots_pack_e5m2_sapphireamx, nk_dots_packed_e5m2_sapphireamx);
    check("dots_pack_e5m2_sapphireamx",
          test_dots_pack_layout<e5m2_t, host_backend_t, nk_dots_pack_size_e5m2_sapphireamx,
                                nk_dots_packed_shape_e5m2_sapphireamx, nk_dots_pack_e5m2_sapphireamx>);
    check("dots_packed_e4m3_sapphireamx", test_dots_packed<e4m3_t>, nk_dots_pack_size_e4m3_sapphireamx,
          nk_dots_pack_e4m3_sapphireamx, nk_dots_packed_e4m3_sapphireamx);
    check("dots_pack_e4m3_sapphireamx",
          test_dots_pack_layout<e4m3_t, host_backend_t, nk_dots_pack_size_e4m3_sapphireamx,
                                nk_dots_packed_shape_e4m3_sapphireamx, nk_dots_pack_e4m3_sapphireamx>);
    check("dots_packed_e3m2_sapphireamx", test_dots_packed<e3m2_t>, nk_dots_pack_size_e3m2_sapphireamx,
          nk_dots_pack_e3m2_sapphireamx, nk_dots_packed_e3m2_sapphireamx);
    check("dots_pack_e3m2_sapphireamx",
          test_dots_pack_layout<e3m2_t, host_backend_t, nk_dots_pack_size_e3m2_sapphireamx,
                                nk_dots_packed_shape_e3m2_sapphireamx, nk_dots_pack_e3m2_sapphireamx>);
    check("dots_packed_e2m3_sapphireamx", test_dots_packed<e2m3_t>, nk_dots_pack_size_e2m3_sapphireamx,
          nk_dots_pack_e2m3_sapphireamx, nk_dots_packed_e2m3_sapphireamx);
    check("dots_pack_e2m3_sapphireamx",
          test_dots_pack_layout<e2m3_t, host_backend_t, nk_dots_pack_size_e2m3_sapphireamx,
                                nk_dots_packed_shape_e2m3_sapphireamx, nk_dots_pack_e2m3_sapphireamx>);
    check("dots_packed_e2m1_sapphireamx", test_dots_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_sapphireamx,
          nk_dots_pack_e2m1_sapphireamx, nk_dots_packed_e2m1_sapphireamx);
    check("dots_pack_e2m1_sapphireamx",
          test_dots_pack_layout<e2m1x2_t, host_backend_t, nk_dots_pack_size_e2m1_sapphireamx,
                                nk_dots_packed_shape_e2m1_sapphireamx, nk_dots_pack_e2m1_sapphireamx>);
    check("dots_packed_i8_sapphireamx", test_dots_packed<i8_t>, nk_dots_pack_size_i8_sapphireamx,
          nk_dots_pack_i8_sapphireamx, nk_dots_packed_i8_sapphireamx);
    check("dots_pack_i8_sapphireamx",
          test_dots_pack_layout<i8_t, host_backend_t, nk_dots_pack_size_i8_sapphireamx,
                                nk_dots_packed_shape_i8_sapphireamx, nk_dots_pack_i8_sapphireamx>);
    check("dots_packed_u8_sapphireamx", test_dots_packed<u8_t>, nk_dots_pack_size_u8_sapphireamx,
          nk_dots_pack_u8_sapphireamx, nk_dots_packed_u8_sapphireamx);
    check("dots_pack_u8_sapphireamx",
          test_dots_pack_layout<u8_t, host_backend_t, nk_dots_pack_size_u8_sapphireamx,
                                nk_dots_packed_shape_u8_sapphireamx, nk_dots_pack_u8_sapphireamx>);

    check("dots_symmetric_bf16_sapphireamx", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_sapphireamx);
    check("dots_symmetric_e5m2_sapphireamx", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_sapphireamx);
    check("dots_symmetric_e4m3_sapphireamx", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_sapphireamx);
    check("dots_symmetric_e3m2_sapphireamx", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_sapphireamx);
    check("dots_symmetric_e2m3_sapphireamx", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_sapphireamx);
    check("dots_symmetric_e2m1_sapphireamx", test_dots_symmetric<e2m1x2_t>, nk_dots_symmetric_e2m1_sapphireamx);
    check("dots_symmetric_i8_sapphireamx", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sapphireamx);
    check("dots_symmetric_u8_sapphireamx", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sapphireamx);

    check("attention_bidirectional_packed_bf16_sapphireamx", test_attention_bidirectional_packed<bf16_t>,
          nk_attention_pack_size_bf16_sapphireamx, nk_attention_pack_bf16_sapphireamx,
          nk_attention_bidirectional_packed_bf16_sapphireamx);
    check("attention_causal_packed_bf16_sapphireamx", test_attention_causal_packed<bf16_t>,
          nk_attention_pack_size_bf16_sapphireamx, nk_attention_pack_bf16_sapphireamx,
          nk_attention_causal_packed_bf16_sapphireamx);
    check("attention_bidirectional_packed_e4m3_sapphireamx", test_attention_bidirectional_packed<e4m3_t>,
          nk_attention_pack_size_e4m3_sapphireamx, nk_attention_pack_e4m3_sapphireamx,
          nk_attention_bidirectional_packed_e4m3_sapphireamx);
    check("attention_causal_packed_e4m3_sapphireamx", test_attention_causal_packed<e4m3_t>,
          nk_attention_pack_size_e4m3_sapphireamx, nk_attention_pack_e4m3_sapphireamx,
          nk_attention_causal_packed_e4m3_sapphireamx);
    check("attention_bidirectional_packed_i8_sapphireamx", test_attention_bidirectional_packed<i8_t>,
          nk_attention_pack_size_i8_sapphireamx, nk_attention_pack_i8_sapphireamx,
          nk_attention_bidirectional_packed_i8_sapphireamx);
    check("attention_causal_packed_i8_sapphireamx", test_attention_causal_packed<i8_t>,
          nk_attention_pack_size_i8_sapphireamx, nk_attention_pack_i8_sapphireamx,
          nk_attention_causal_packed_i8_sapphireamx);
#endif // NUMKONG_TARGET_SAPPHIREAMX

#if NUMKONG_TARGET_GRANITEAMX
    check.section("Cross Granite AMX", nk_cap_graniteamx_k);
    check("dots_packed_f16_graniteamx", test_dots_packed<f16_t>, nk_dots_pack_size_f16_graniteamx,
          nk_dots_pack_f16_graniteamx, nk_dots_packed_f16_graniteamx);
    check("dots_pack_f16_graniteamx",
          test_dots_pack_layout<f16_t, host_backend_t, nk_dots_pack_size_f16_graniteamx,
                                nk_dots_packed_shape_f16_graniteamx, nk_dots_pack_f16_graniteamx>);
    check("dots_symmetric_f16_graniteamx", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_graniteamx);

    check("angulars_packed_f16_graniteamx", test_angulars_packed<f16_t>, nk_dots_pack_size_f16_graniteamx,
          nk_dots_pack_f16_graniteamx, nk_angulars_packed_f16_graniteamx);
    check("angulars_symmetric_f16_graniteamx", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_graniteamx);

    check("euclideans_packed_f16_graniteamx", test_euclideans_packed<f16_t>, nk_dots_pack_size_f16_graniteamx,
          nk_dots_pack_f16_graniteamx, nk_euclideans_packed_f16_graniteamx);
    check("euclideans_symmetric_f16_graniteamx", test_euclideans_symmetric<f16_t>,
          nk_euclideans_symmetric_f16_graniteamx);

    check("dots_packed_e5m2_graniteamx", test_dots_packed<e5m2_t>, nk_dots_pack_size_e5m2_graniteamx,
          nk_dots_pack_e5m2_graniteamx, nk_dots_packed_e5m2_graniteamx);
    check("dots_pack_e5m2_graniteamx",
          test_dots_pack_layout<e5m2_t, host_backend_t, nk_dots_pack_size_e5m2_graniteamx,
                                nk_dots_packed_shape_e5m2_graniteamx, nk_dots_pack_e5m2_graniteamx>);
    check("dots_symmetric_e5m2_graniteamx", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_graniteamx);

    check("angulars_packed_e5m2_graniteamx", test_angulars_packed<e5m2_t>, nk_dots_pack_size_e5m2_graniteamx,
          nk_dots_pack_e5m2_graniteamx, nk_angulars_packed_e5m2_graniteamx);
    check("angulars_symmetric_e5m2_graniteamx", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_graniteamx);

    check("euclideans_packed_e5m2_graniteamx", test_euclideans_packed<e5m2_t>, nk_dots_pack_size_e5m2_graniteamx,
          nk_dots_pack_e5m2_graniteamx, nk_euclideans_packed_e5m2_graniteamx);
    check("euclideans_symmetric_e5m2_graniteamx", test_euclideans_symmetric<e5m2_t>,
          nk_euclideans_symmetric_e5m2_graniteamx);
#endif // NUMKONG_TARGET_GRANITEAMX

#if NUMKONG_TARGET_DIAMONDAMX
    check.section("Cross Diamond AMX", nk_cap_diamondamx_k);
    check("attention_bidirectional_packed_e4m3_diamondamx", test_attention_bidirectional_packed<e4m3_t>,
          nk_attention_pack_size_e4m3_diamondamx, nk_attention_pack_e4m3_diamondamx,
          nk_attention_bidirectional_packed_e4m3_diamondamx);
    check("attention_causal_packed_e4m3_diamondamx", test_attention_causal_packed<e4m3_t>,
          nk_attention_pack_size_e4m3_diamondamx, nk_attention_pack_e4m3_diamondamx,
          nk_attention_causal_packed_e4m3_diamondamx);
#endif // NUMKONG_TARGET_DIAMONDAMX
}
