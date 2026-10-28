/**
 *  @file test/cross_x8664amx.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Batch operation tests - x86-64 ISA family, AMX tiles.
 *
 *  Covers Sapphire AMX, Granite AMX, Diamond AMX.
 */
#include "harness.hpp"
#include "cross.hpp"

namespace ashvardanian::numkong::test {

void test_cross_x8664amx([[maybe_unused]] error_stats_section_t &check) {
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
    check("dots_packed_nvfp4_sapphireamx", test_dots_packed<nvfp4_t>, nk_dots_pack_size_nvfp4_sapphireamx,
          nk_dots_pack_nvfp4_sapphireamx, nk_dots_packed_nvfp4_sapphireamx);
    check("dots_packed_mxfp4_sapphireamx", test_dots_packed<mxfp4_t>, nk_dots_pack_size_mxfp4_sapphireamx,
          nk_dots_pack_mxfp4_sapphireamx, nk_dots_packed_mxfp4_sapphireamx);
    check("dots_packed_mxfp8e4m3_sapphireamx", test_dots_packed<mxfp8e4m3_t>, nk_dots_pack_size_mxfp8e4m3_sapphireamx,
          nk_dots_pack_mxfp8e4m3_sapphireamx, nk_dots_packed_mxfp8e4m3_sapphireamx);
    check("dots_packed_mxfp8e5m2_sapphireamx", test_dots_packed<mxfp8e5m2_t>, nk_dots_pack_size_mxfp8e5m2_sapphireamx,
          nk_dots_pack_mxfp8e5m2_sapphireamx, nk_dots_packed_mxfp8e5m2_sapphireamx);
    check("dots_pack_e2m1_sapphireamx",
          test_dots_pack_layout<e2m1x2_t, host_backend_t, nk_dots_pack_size_e2m1_sapphireamx,
                                nk_dots_packed_shape_e2m1_sapphireamx, nk_dots_pack_e2m1_sapphireamx>);
    check("dots_pack_nvfp4_sapphireamx",
          test_dots_pack_layout<nvfp4_t, host_backend_t, nk_dots_pack_size_nvfp4_sapphireamx,
                                nk_dots_packed_shape_nvfp4_sapphireamx, nk_dots_pack_nvfp4_sapphireamx>);
    check("dots_pack_mxfp4_sapphireamx",
          test_dots_pack_layout<mxfp4_t, host_backend_t, nk_dots_pack_size_mxfp4_sapphireamx,
                                nk_dots_packed_shape_mxfp4_sapphireamx, nk_dots_pack_mxfp4_sapphireamx>);
    check("dots_pack_mxfp8e4m3_sapphireamx",
          test_dots_pack_layout<mxfp8e4m3_t, host_backend_t, nk_dots_pack_size_mxfp8e4m3_sapphireamx,
                                nk_dots_packed_shape_mxfp8e4m3_sapphireamx, nk_dots_pack_mxfp8e4m3_sapphireamx>);
    check("dots_pack_mxfp8e5m2_sapphireamx",
          test_dots_pack_layout<mxfp8e5m2_t, host_backend_t, nk_dots_pack_size_mxfp8e5m2_sapphireamx,
                                nk_dots_packed_shape_mxfp8e5m2_sapphireamx, nk_dots_pack_mxfp8e5m2_sapphireamx>);
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
    check("dots_symmetric_nvfp4_sapphireamx", test_dots_symmetric<nvfp4_t>, nk_dots_symmetric_nvfp4_sapphireamx);
    check("dots_symmetric_mxfp4_sapphireamx", test_dots_symmetric<mxfp4_t>, nk_dots_symmetric_mxfp4_sapphireamx);
    check("dots_symmetric_mxfp8e4m3_sapphireamx", test_dots_symmetric<mxfp8e4m3_t>,
          nk_dots_symmetric_mxfp8e4m3_sapphireamx);
    check("dots_symmetric_mxfp8e5m2_sapphireamx", test_dots_symmetric<mxfp8e5m2_t>,
          nk_dots_symmetric_mxfp8e5m2_sapphireamx);
    check("dots_symmetric_i8_sapphireamx", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sapphireamx);
    check("dots_symmetric_u8_sapphireamx", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sapphireamx);

    check("attention_packed_bf16_sapphireamx", test_attention_packed<bf16_t>, nk_attention_pack_size_bf16_sapphireamx,
          nk_attention_pack_bf16_sapphireamx, nk_attention_packed_bf16_sapphireamx);
    check("attention_packed_e4m3_sapphireamx", test_attention_packed<e4m3_t>, nk_attention_pack_size_e4m3_sapphireamx,
          nk_attention_pack_e4m3_sapphireamx, nk_attention_packed_e4m3_sapphireamx);
    check("attention_packed_i8_sapphireamx", test_attention_packed<i8_t>, nk_attention_pack_size_i8_sapphireamx,
          nk_attention_pack_i8_sapphireamx, nk_attention_packed_i8_sapphireamx);

    check("angulars_packed_bf16_sapphireamx", test_angulars_packed<bf16_t>, nk_dots_pack_size_bf16_sapphireamx,
          nk_dots_pack_bf16_sapphireamx, nk_angulars_packed_bf16_sapphireamx);
    check("angulars_packed_e5m2_sapphireamx", test_angulars_packed<e5m2_t>, nk_dots_pack_size_e5m2_sapphireamx,
          nk_dots_pack_e5m2_sapphireamx, nk_angulars_packed_e5m2_sapphireamx);
    check("angulars_packed_e4m3_sapphireamx", test_angulars_packed<e4m3_t>, nk_dots_pack_size_e4m3_sapphireamx,
          nk_dots_pack_e4m3_sapphireamx, nk_angulars_packed_e4m3_sapphireamx);
    check("angulars_packed_e3m2_sapphireamx", test_angulars_packed<e3m2_t>, nk_dots_pack_size_e3m2_sapphireamx,
          nk_dots_pack_e3m2_sapphireamx, nk_angulars_packed_e3m2_sapphireamx);
    check("angulars_packed_e2m3_sapphireamx", test_angulars_packed<e2m3_t>, nk_dots_pack_size_e2m3_sapphireamx,
          nk_dots_pack_e2m3_sapphireamx, nk_angulars_packed_e2m3_sapphireamx);
    check("angulars_packed_e2m1_sapphireamx", test_angulars_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_sapphireamx,
          nk_dots_pack_e2m1_sapphireamx, nk_angulars_packed_e2m1_sapphireamx);
    check("angulars_packed_nvfp4_sapphireamx", test_angulars_packed<nvfp4_t>, nk_dots_pack_size_nvfp4_sapphireamx,
          nk_dots_pack_nvfp4_sapphireamx, nk_angulars_packed_nvfp4_sapphireamx);
    check("angulars_packed_mxfp4_sapphireamx", test_angulars_packed<mxfp4_t>, nk_dots_pack_size_mxfp4_sapphireamx,
          nk_dots_pack_mxfp4_sapphireamx, nk_angulars_packed_mxfp4_sapphireamx);
    check("angulars_packed_mxfp8e4m3_sapphireamx", test_angulars_packed<mxfp8e4m3_t>,
          nk_dots_pack_size_mxfp8e4m3_sapphireamx, nk_dots_pack_mxfp8e4m3_sapphireamx,
          nk_angulars_packed_mxfp8e4m3_sapphireamx);
    check("angulars_packed_mxfp8e5m2_sapphireamx", test_angulars_packed<mxfp8e5m2_t>,
          nk_dots_pack_size_mxfp8e5m2_sapphireamx, nk_dots_pack_mxfp8e5m2_sapphireamx,
          nk_angulars_packed_mxfp8e5m2_sapphireamx);
    check("angulars_packed_i8_sapphireamx", test_angulars_packed<i8_t>, nk_dots_pack_size_i8_sapphireamx,
          nk_dots_pack_i8_sapphireamx, nk_angulars_packed_i8_sapphireamx);
    check("angulars_packed_u8_sapphireamx", test_angulars_packed<u8_t>, nk_dots_pack_size_u8_sapphireamx,
          nk_dots_pack_u8_sapphireamx, nk_angulars_packed_u8_sapphireamx);

    check("angulars_symmetric_bf16_sapphireamx", test_angulars_symmetric<bf16_t>,
          nk_angulars_symmetric_bf16_sapphireamx);
    check("angulars_symmetric_e5m2_sapphireamx", test_angulars_symmetric<e5m2_t>,
          nk_angulars_symmetric_e5m2_sapphireamx);
    check("angulars_symmetric_e4m3_sapphireamx", test_angulars_symmetric<e4m3_t>,
          nk_angulars_symmetric_e4m3_sapphireamx);
    check("angulars_symmetric_e3m2_sapphireamx", test_angulars_symmetric<e3m2_t>,
          nk_angulars_symmetric_e3m2_sapphireamx);
    check("angulars_symmetric_e2m3_sapphireamx", test_angulars_symmetric<e2m3_t>,
          nk_angulars_symmetric_e2m3_sapphireamx);
    check("angulars_symmetric_e2m1_sapphireamx", test_angulars_symmetric<e2m1x2_t>,
          nk_angulars_symmetric_e2m1_sapphireamx);
    check("angulars_symmetric_nvfp4_sapphireamx", test_angulars_symmetric<nvfp4_t>,
          nk_angulars_symmetric_nvfp4_sapphireamx);
    check("angulars_symmetric_mxfp4_sapphireamx", test_angulars_symmetric<mxfp4_t>,
          nk_angulars_symmetric_mxfp4_sapphireamx);
    check("angulars_symmetric_mxfp8e4m3_sapphireamx", test_angulars_symmetric<mxfp8e4m3_t>,
          nk_angulars_symmetric_mxfp8e4m3_sapphireamx);
    check("angulars_symmetric_mxfp8e5m2_sapphireamx", test_angulars_symmetric<mxfp8e5m2_t>,
          nk_angulars_symmetric_mxfp8e5m2_sapphireamx);
    check("angulars_symmetric_i8_sapphireamx", test_angulars_symmetric<i8_t>, nk_angulars_symmetric_i8_sapphireamx);
    check("angulars_symmetric_u8_sapphireamx", test_angulars_symmetric<u8_t>, nk_angulars_symmetric_u8_sapphireamx);

    check("euclideans_packed_bf16_sapphireamx", test_euclideans_packed<bf16_t>, nk_dots_pack_size_bf16_sapphireamx,
          nk_dots_pack_bf16_sapphireamx, nk_euclideans_packed_bf16_sapphireamx);
    check("euclideans_packed_e5m2_sapphireamx", test_euclideans_packed<e5m2_t>, nk_dots_pack_size_e5m2_sapphireamx,
          nk_dots_pack_e5m2_sapphireamx, nk_euclideans_packed_e5m2_sapphireamx);
    check("euclideans_packed_e4m3_sapphireamx", test_euclideans_packed<e4m3_t>, nk_dots_pack_size_e4m3_sapphireamx,
          nk_dots_pack_e4m3_sapphireamx, nk_euclideans_packed_e4m3_sapphireamx);
    check("euclideans_packed_e3m2_sapphireamx", test_euclideans_packed<e3m2_t>, nk_dots_pack_size_e3m2_sapphireamx,
          nk_dots_pack_e3m2_sapphireamx, nk_euclideans_packed_e3m2_sapphireamx);
    check("euclideans_packed_e2m3_sapphireamx", test_euclideans_packed<e2m3_t>, nk_dots_pack_size_e2m3_sapphireamx,
          nk_dots_pack_e2m3_sapphireamx, nk_euclideans_packed_e2m3_sapphireamx);
    check("euclideans_packed_e2m1_sapphireamx", test_euclideans_packed<e2m1x2_t>, nk_dots_pack_size_e2m1_sapphireamx,
          nk_dots_pack_e2m1_sapphireamx, nk_euclideans_packed_e2m1_sapphireamx);
    check("euclideans_packed_nvfp4_sapphireamx", test_euclideans_packed<nvfp4_t>, nk_dots_pack_size_nvfp4_sapphireamx,
          nk_dots_pack_nvfp4_sapphireamx, nk_euclideans_packed_nvfp4_sapphireamx);
    check("euclideans_packed_mxfp4_sapphireamx", test_euclideans_packed<mxfp4_t>, nk_dots_pack_size_mxfp4_sapphireamx,
          nk_dots_pack_mxfp4_sapphireamx, nk_euclideans_packed_mxfp4_sapphireamx);
    check("euclideans_packed_mxfp8e4m3_sapphireamx", test_euclideans_packed<mxfp8e4m3_t>,
          nk_dots_pack_size_mxfp8e4m3_sapphireamx, nk_dots_pack_mxfp8e4m3_sapphireamx,
          nk_euclideans_packed_mxfp8e4m3_sapphireamx);
    check("euclideans_packed_mxfp8e5m2_sapphireamx", test_euclideans_packed<mxfp8e5m2_t>,
          nk_dots_pack_size_mxfp8e5m2_sapphireamx, nk_dots_pack_mxfp8e5m2_sapphireamx,
          nk_euclideans_packed_mxfp8e5m2_sapphireamx);
    check("euclideans_packed_i8_sapphireamx", test_euclideans_packed<i8_t>, nk_dots_pack_size_i8_sapphireamx,
          nk_dots_pack_i8_sapphireamx, nk_euclideans_packed_i8_sapphireamx);
    check("euclideans_packed_u8_sapphireamx", test_euclideans_packed<u8_t>, nk_dots_pack_size_u8_sapphireamx,
          nk_dots_pack_u8_sapphireamx, nk_euclideans_packed_u8_sapphireamx);

    check("euclideans_symmetric_bf16_sapphireamx", test_euclideans_symmetric<bf16_t>,
          nk_euclideans_symmetric_bf16_sapphireamx);
    check("euclideans_symmetric_e5m2_sapphireamx", test_euclideans_symmetric<e5m2_t>,
          nk_euclideans_symmetric_e5m2_sapphireamx);
    check("euclideans_symmetric_e4m3_sapphireamx", test_euclideans_symmetric<e4m3_t>,
          nk_euclideans_symmetric_e4m3_sapphireamx);
    check("euclideans_symmetric_e3m2_sapphireamx", test_euclideans_symmetric<e3m2_t>,
          nk_euclideans_symmetric_e3m2_sapphireamx);
    check("euclideans_symmetric_e2m3_sapphireamx", test_euclideans_symmetric<e2m3_t>,
          nk_euclideans_symmetric_e2m3_sapphireamx);
    check("euclideans_symmetric_e2m1_sapphireamx", test_euclideans_symmetric<e2m1x2_t>,
          nk_euclideans_symmetric_e2m1_sapphireamx);
    check("euclideans_symmetric_nvfp4_sapphireamx", test_euclideans_symmetric<nvfp4_t>,
          nk_euclideans_symmetric_nvfp4_sapphireamx);
    check("euclideans_symmetric_mxfp4_sapphireamx", test_euclideans_symmetric<mxfp4_t>,
          nk_euclideans_symmetric_mxfp4_sapphireamx);
    check("euclideans_symmetric_mxfp8e4m3_sapphireamx", test_euclideans_symmetric<mxfp8e4m3_t>,
          nk_euclideans_symmetric_mxfp8e4m3_sapphireamx);
    check("euclideans_symmetric_mxfp8e5m2_sapphireamx", test_euclideans_symmetric<mxfp8e5m2_t>,
          nk_euclideans_symmetric_mxfp8e5m2_sapphireamx);
    check("euclideans_symmetric_i8_sapphireamx", test_euclideans_symmetric<i8_t>,
          nk_euclideans_symmetric_i8_sapphireamx);
    check("euclideans_symmetric_u8_sapphireamx", test_euclideans_symmetric<u8_t>,
          nk_euclideans_symmetric_u8_sapphireamx);
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
    check("attention_packed_e4m3_diamondamx", test_attention_packed<e4m3_t>, nk_attention_pack_size_e4m3_diamondamx,
          nk_attention_pack_e4m3_diamondamx, nk_attention_packed_e4m3_diamondamx);
#endif // NUMKONG_TARGET_DIAMONDAMX
}

} // namespace ashvardanian::numkong::test
