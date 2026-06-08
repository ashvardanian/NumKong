/**
 *  @brief Batch operation tests - AMX ISA family (Sapphire Rapids AMX).
 *  @file test/test_cross_amx.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2026
 */
#include "test.hpp"
#include "test_cross.hpp"

void test_cross_amx() {
    [[maybe_unused]] error_stats_section_t check;
#if NK_TARGET_SAPPHIREAMX
    check("dots_packed_bf16_sapphireamx", test_dots_packed<bf16_t>, nk_dots_packed_size_bf16_sapphireamx,
          nk_dots_pack_bf16_sapphireamx, nk_dots_packed_bf16_sapphireamx);
    check("dots_packed_e5m2_sapphireamx", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_sapphireamx,
          nk_dots_pack_e5m2_sapphireamx, nk_dots_packed_e5m2_sapphireamx);
    check("dots_packed_e4m3_sapphireamx", test_dots_packed<e4m3_t>, nk_dots_packed_size_e4m3_sapphireamx,
          nk_dots_pack_e4m3_sapphireamx, nk_dots_packed_e4m3_sapphireamx);
    check("dots_packed_e3m2_sapphireamx", test_dots_packed<e3m2_t>, nk_dots_packed_size_e3m2_sapphireamx,
          nk_dots_pack_e3m2_sapphireamx, nk_dots_packed_e3m2_sapphireamx);
    check("dots_packed_e2m3_sapphireamx", test_dots_packed<e2m3_t>, nk_dots_packed_size_e2m3_sapphireamx,
          nk_dots_pack_e2m3_sapphireamx, nk_dots_packed_e2m3_sapphireamx);
    check("dots_packed_i8_sapphireamx", test_dots_packed<i8_t>, nk_dots_packed_size_i8_sapphireamx,
          nk_dots_pack_i8_sapphireamx, nk_dots_packed_i8_sapphireamx);
    check("dots_packed_u8_sapphireamx", test_dots_packed<u8_t>, nk_dots_packed_size_u8_sapphireamx,
          nk_dots_pack_u8_sapphireamx, nk_dots_packed_u8_sapphireamx);

    check("dots_symmetric_bf16_sapphireamx", test_dots_symmetric<bf16_t>, nk_dots_symmetric_bf16_sapphireamx);
    check("dots_symmetric_e5m2_sapphireamx", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_sapphireamx);
    check("dots_symmetric_e4m3_sapphireamx", test_dots_symmetric<e4m3_t>, nk_dots_symmetric_e4m3_sapphireamx);
    check("dots_symmetric_e3m2_sapphireamx", test_dots_symmetric<e3m2_t>, nk_dots_symmetric_e3m2_sapphireamx);
    check("dots_symmetric_e2m3_sapphireamx", test_dots_symmetric<e2m3_t>, nk_dots_symmetric_e2m3_sapphireamx);
    check("dots_symmetric_i8_sapphireamx", test_dots_symmetric<i8_t>, nk_dots_symmetric_i8_sapphireamx);
    check("dots_symmetric_u8_sapphireamx", test_dots_symmetric<u8_t>, nk_dots_symmetric_u8_sapphireamx);
#endif
#if NK_TARGET_GRANITEAMX
    check("dots_packed_f16_graniteamx", test_dots_packed<f16_t>, nk_dots_packed_size_f16_graniteamx,
          nk_dots_pack_f16_graniteamx, nk_dots_packed_f16_graniteamx);
    check("dots_symmetric_f16_graniteamx", test_dots_symmetric<f16_t>, nk_dots_symmetric_f16_graniteamx);

    check("angulars_packed_f16_graniteamx", test_angulars_packed<f16_t>, nk_dots_packed_size_f16_graniteamx,
          nk_dots_pack_f16_graniteamx, nk_angulars_packed_f16_graniteamx);
    check("angulars_symmetric_f16_graniteamx", test_angulars_symmetric<f16_t>, nk_angulars_symmetric_f16_graniteamx);

    check("euclideans_packed_f16_graniteamx", test_euclideans_packed<f16_t>, nk_dots_packed_size_f16_graniteamx,
          nk_dots_pack_f16_graniteamx, nk_euclideans_packed_f16_graniteamx);
    check("euclideans_symmetric_f16_graniteamx", test_euclideans_symmetric<f16_t>,
          nk_euclideans_symmetric_f16_graniteamx);

    check("dots_packed_e5m2_graniteamx", test_dots_packed<e5m2_t>, nk_dots_packed_size_e5m2_graniteamx,
          nk_dots_pack_e5m2_graniteamx, nk_dots_packed_e5m2_graniteamx);
    check("dots_symmetric_e5m2_graniteamx", test_dots_symmetric<e5m2_t>, nk_dots_symmetric_e5m2_graniteamx);

    check("angulars_packed_e5m2_graniteamx", test_angulars_packed<e5m2_t>, nk_dots_packed_size_e5m2_graniteamx,
          nk_dots_pack_e5m2_graniteamx, nk_angulars_packed_e5m2_graniteamx);
    check("angulars_symmetric_e5m2_graniteamx", test_angulars_symmetric<e5m2_t>, nk_angulars_symmetric_e5m2_graniteamx);

    check("euclideans_packed_e5m2_graniteamx", test_euclideans_packed<e5m2_t>, nk_dots_packed_size_e5m2_graniteamx,
          nk_dots_pack_e5m2_graniteamx, nk_euclideans_packed_e5m2_graniteamx);
    check("euclideans_symmetric_e5m2_graniteamx", test_euclideans_symmetric<e5m2_t>,
          nk_euclideans_symmetric_e5m2_graniteamx);
#endif
}
