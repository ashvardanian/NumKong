/**
 *  @file test/cross_runtime.cpp
 *  @author Ash Vardanian
 *  @date January 14, 2025
 *  @brief Batch operation tests - runtime dispatch through the library.
 */
#include "harness.hpp"
#include "cross.hpp"

namespace ashvardanian::numkong::test {

void test_cross_runtime([[maybe_unused]] error_stats_section_t &check) {
#if !NUMKONG_HEADER_ONLY
    check.section("Cross Runtime Dispatch", nk_cap_serial_k);
    check("dots_packed_f64", test_dots_packed<f64_t>, cpu_best<nk_dots_pack_size_f64_best>,
          cpu_best<nk_dots_pack_f64_best>, cpu_best<nk_dots_packed_f64_best>);
    check("dots_pack_f64",
          test_dots_pack_layout<f64_t, host_backend_t, cpu_best<nk_dots_pack_size_f64_best>,
                                cpu_best<nk_dots_packed_shape_f64_best>, cpu_best<nk_dots_pack_f64_best>>);
    check("dots_packed_f32", test_dots_packed<f32_t>, cpu_best<nk_dots_pack_size_f32_best>,
          cpu_best<nk_dots_pack_f32_best>, cpu_best<nk_dots_packed_f32_best>);
    check("dots_pack_f32",
          test_dots_pack_layout<f32_t, host_backend_t, cpu_best<nk_dots_pack_size_f32_best>,
                                cpu_best<nk_dots_packed_shape_f32_best>, cpu_best<nk_dots_pack_f32_best>>);
    check("dots_packed_bf16", test_dots_packed<bf16_t>, cpu_best<nk_dots_pack_size_bf16_best>,
          cpu_best<nk_dots_pack_bf16_best>, cpu_best<nk_dots_packed_bf16_best>);
    check("dots_pack_bf16",
          test_dots_pack_layout<bf16_t, host_backend_t, cpu_best<nk_dots_pack_size_bf16_best>,
                                cpu_best<nk_dots_packed_shape_bf16_best>, cpu_best<nk_dots_pack_bf16_best>>);
    check("dots_packed_f16", test_dots_packed<f16_t>, cpu_best<nk_dots_pack_size_f16_best>,
          cpu_best<nk_dots_pack_f16_best>, cpu_best<nk_dots_packed_f16_best>);
    check("dots_pack_f16",
          test_dots_pack_layout<f16_t, host_backend_t, cpu_best<nk_dots_pack_size_f16_best>,
                                cpu_best<nk_dots_packed_shape_f16_best>, cpu_best<nk_dots_pack_f16_best>>);
    check("dots_packed_i8", test_dots_packed<i8_t>, cpu_best<nk_dots_pack_size_i8_best>, cpu_best<nk_dots_pack_i8_best>,
          cpu_best<nk_dots_packed_i8_best>);
    check("dots_pack_i8",
          test_dots_pack_layout<i8_t, host_backend_t, cpu_best<nk_dots_pack_size_i8_best>,
                                cpu_best<nk_dots_packed_shape_i8_best>, cpu_best<nk_dots_pack_i8_best>>);
    check("dots_packed_i4", test_dots_packed<i4x2_t>, cpu_best<nk_dots_pack_size_i4_best>,
          cpu_best<nk_dots_pack_i4_best>, cpu_best<nk_dots_packed_i4_best>);
    check("dots_pack_i4",
          test_dots_pack_layout<i4x2_t, host_backend_t, cpu_best<nk_dots_pack_size_i4_best>,
                                cpu_best<nk_dots_packed_shape_i4_best>, cpu_best<nk_dots_pack_i4_best>>);
    check("dots_packed_u4", test_dots_packed<u4x2_t>, cpu_best<nk_dots_pack_size_u4_best>,
          cpu_best<nk_dots_pack_u4_best>, cpu_best<nk_dots_packed_u4_best>);
    check("dots_pack_u4",
          test_dots_pack_layout<u4x2_t, host_backend_t, cpu_best<nk_dots_pack_size_u4_best>,
                                cpu_best<nk_dots_packed_shape_u4_best>, cpu_best<nk_dots_pack_u4_best>>);
    check("dots_packed_u1", test_dots_packed<u1x8_t>, cpu_best<nk_dots_pack_size_u1_best>,
          cpu_best<nk_dots_pack_u1_best>, cpu_best<nk_dots_packed_u1_best>);
    check("dots_pack_u1",
          test_dots_pack_layout<u1x8_t, host_backend_t, cpu_best<nk_dots_pack_size_u1_best>,
                                cpu_best<nk_dots_packed_shape_u1_best>, cpu_best<nk_dots_pack_u1_best>>);

    check("dots_symmetric_f64", test_dots_symmetric<f64_t>, cpu_best<nk_dots_symmetric_f64_best>);
    check("dots_symmetric_f32", test_dots_symmetric<f32_t>, cpu_best<nk_dots_symmetric_f32_best>);
    check("dots_symmetric_bf16", test_dots_symmetric<bf16_t>, cpu_best<nk_dots_symmetric_bf16_best>);
    check("dots_symmetric_f16", test_dots_symmetric<f16_t>, cpu_best<nk_dots_symmetric_f16_best>);
    check("dots_symmetric_e5m2", test_dots_symmetric<e5m2_t>, cpu_best<nk_dots_symmetric_e5m2_best>);
    check("dots_symmetric_e4m3", test_dots_symmetric<e4m3_t>, cpu_best<nk_dots_symmetric_e4m3_best>);
    check("dots_symmetric_i8", test_dots_symmetric<i8_t>, cpu_best<nk_dots_symmetric_i8_best>);
    check("dots_symmetric_i4", test_dots_symmetric<i4x2_t>, cpu_best<nk_dots_symmetric_i4_best>);
    check("dots_symmetric_u8", test_dots_symmetric<u8_t>, cpu_best<nk_dots_symmetric_u8_best>);
    check("dots_symmetric_u4", test_dots_symmetric<u4x2_t>, cpu_best<nk_dots_symmetric_u4_best>);
    check("dots_symmetric_u1", test_dots_symmetric<u1x8_t>, cpu_best<nk_dots_symmetric_u1_best>);

    check("hammings_packed_u1", test_hammings_packed<u1x8_t>, cpu_best<nk_dots_pack_size_u1_best>,
          cpu_best<nk_dots_pack_u1_best>, cpu_best<nk_hammings_packed_u1_best>);
    check("hammings_symmetric_u1", test_hammings_symmetric<u1x8_t>, cpu_best<nk_hammings_symmetric_u1_best>);

    check("jaccards_packed_u1", test_jaccards_packed<u1x8_t>, cpu_best<nk_dots_pack_size_u1_best>,
          cpu_best<nk_dots_pack_u1_best>, cpu_best<nk_jaccards_packed_u1_best>);
    check("jaccards_symmetric_u1", test_jaccards_symmetric<u1x8_t>, cpu_best<nk_jaccards_symmetric_u1_best>);

    check("dots_packed_e5m2", test_dots_packed<e5m2_t>, cpu_best<nk_dots_pack_size_e5m2_best>,
          cpu_best<nk_dots_pack_e5m2_best>, cpu_best<nk_dots_packed_e5m2_best>);
    check("dots_pack_e5m2",
          test_dots_pack_layout<e5m2_t, host_backend_t, cpu_best<nk_dots_pack_size_e5m2_best>,
                                cpu_best<nk_dots_packed_shape_e5m2_best>, cpu_best<nk_dots_pack_e5m2_best>>);
    check("dots_packed_e4m3", test_dots_packed<e4m3_t>, cpu_best<nk_dots_pack_size_e4m3_best>,
          cpu_best<nk_dots_pack_e4m3_best>, cpu_best<nk_dots_packed_e4m3_best>);
    check("dots_pack_e4m3",
          test_dots_pack_layout<e4m3_t, host_backend_t, cpu_best<nk_dots_pack_size_e4m3_best>,
                                cpu_best<nk_dots_packed_shape_e4m3_best>, cpu_best<nk_dots_pack_e4m3_best>>);
    check("dots_packed_e3m2", test_dots_packed<e3m2_t>, cpu_best<nk_dots_pack_size_e3m2_best>,
          cpu_best<nk_dots_pack_e3m2_best>, cpu_best<nk_dots_packed_e3m2_best>);
    check("dots_pack_e3m2",
          test_dots_pack_layout<e3m2_t, host_backend_t, cpu_best<nk_dots_pack_size_e3m2_best>,
                                cpu_best<nk_dots_packed_shape_e3m2_best>, cpu_best<nk_dots_pack_e3m2_best>>);
    check("dots_packed_e2m3", test_dots_packed<e2m3_t>, cpu_best<nk_dots_pack_size_e2m3_best>,
          cpu_best<nk_dots_pack_e2m3_best>, cpu_best<nk_dots_packed_e2m3_best>);
    check("dots_pack_e2m3",
          test_dots_pack_layout<e2m3_t, host_backend_t, cpu_best<nk_dots_pack_size_e2m3_best>,
                                cpu_best<nk_dots_packed_shape_e2m3_best>, cpu_best<nk_dots_pack_e2m3_best>>);
    check("dots_packed_e2m1", test_dots_packed<e2m1x2_t>, cpu_best<nk_dots_pack_size_e2m1_best>,
          cpu_best<nk_dots_pack_e2m1_best>, cpu_best<nk_dots_packed_e2m1_best>);
    check("dots_pack_e2m1",
          test_dots_pack_layout<e2m1x2_t, host_backend_t, cpu_best<nk_dots_pack_size_e2m1_best>,
                                cpu_best<nk_dots_packed_shape_e2m1_best>, cpu_best<nk_dots_pack_e2m1_best>>);
    check("dots_packed_u8", test_dots_packed<u8_t>, cpu_best<nk_dots_pack_size_u8_best>, cpu_best<nk_dots_pack_u8_best>,
          cpu_best<nk_dots_packed_u8_best>);
    check("dots_pack_u8",
          test_dots_pack_layout<u8_t, host_backend_t, cpu_best<nk_dots_pack_size_u8_best>,
                                cpu_best<nk_dots_packed_shape_u8_best>, cpu_best<nk_dots_pack_u8_best>>);

    check("dots_symmetric_e3m2", test_dots_symmetric<e3m2_t>, cpu_best<nk_dots_symmetric_e3m2_best>);
    check("dots_symmetric_e2m3", test_dots_symmetric<e2m3_t>, cpu_best<nk_dots_symmetric_e2m3_best>);
    check("dots_symmetric_e2m1", test_dots_symmetric<e2m1x2_t>, cpu_best<nk_dots_symmetric_e2m1_best>);

    check("angulars_packed_f64", test_angulars_packed<f64_t>, cpu_best<nk_dots_pack_size_f64_best>,
          cpu_best<nk_dots_pack_f64_best>, cpu_best<nk_angulars_packed_f64_best>);
    check("angulars_packed_f32", test_angulars_packed<f32_t>, cpu_best<nk_dots_pack_size_f32_best>,
          cpu_best<nk_dots_pack_f32_best>, cpu_best<nk_angulars_packed_f32_best>);
    check("angulars_packed_bf16", test_angulars_packed<bf16_t>, cpu_best<nk_dots_pack_size_bf16_best>,
          cpu_best<nk_dots_pack_bf16_best>, cpu_best<nk_angulars_packed_bf16_best>);
    check("angulars_packed_f16", test_angulars_packed<f16_t>, cpu_best<nk_dots_pack_size_f16_best>,
          cpu_best<nk_dots_pack_f16_best>, cpu_best<nk_angulars_packed_f16_best>);
    check("angulars_packed_e5m2", test_angulars_packed<e5m2_t>, cpu_best<nk_dots_pack_size_e5m2_best>,
          cpu_best<nk_dots_pack_e5m2_best>, cpu_best<nk_angulars_packed_e5m2_best>);
    check("angulars_packed_e4m3", test_angulars_packed<e4m3_t>, cpu_best<nk_dots_pack_size_e4m3_best>,
          cpu_best<nk_dots_pack_e4m3_best>, cpu_best<nk_angulars_packed_e4m3_best>);
    check("angulars_packed_e3m2", test_angulars_packed<e3m2_t>, cpu_best<nk_dots_pack_size_e3m2_best>,
          cpu_best<nk_dots_pack_e3m2_best>, cpu_best<nk_angulars_packed_e3m2_best>);
    check("angulars_packed_e2m3", test_angulars_packed<e2m3_t>, cpu_best<nk_dots_pack_size_e2m3_best>,
          cpu_best<nk_dots_pack_e2m3_best>, cpu_best<nk_angulars_packed_e2m3_best>);
    check("angulars_packed_e2m1", test_angulars_packed<e2m1x2_t>, cpu_best<nk_dots_pack_size_e2m1_best>,
          cpu_best<nk_dots_pack_e2m1_best>, cpu_best<nk_angulars_packed_e2m1_best>);
    check("angulars_packed_i8", test_angulars_packed<i8_t>, cpu_best<nk_dots_pack_size_i8_best>,
          cpu_best<nk_dots_pack_i8_best>, cpu_best<nk_angulars_packed_i8_best>);
    check("angulars_packed_i4", test_angulars_packed<i4x2_t>, cpu_best<nk_dots_pack_size_i4_best>,
          cpu_best<nk_dots_pack_i4_best>, cpu_best<nk_angulars_packed_i4_best>);
    check("angulars_packed_u8", test_angulars_packed<u8_t>, cpu_best<nk_dots_pack_size_u8_best>,
          cpu_best<nk_dots_pack_u8_best>, cpu_best<nk_angulars_packed_u8_best>);
    check("angulars_packed_u4", test_angulars_packed<u4x2_t>, cpu_best<nk_dots_pack_size_u4_best>,
          cpu_best<nk_dots_pack_u4_best>, cpu_best<nk_angulars_packed_u4_best>);

    check("angulars_symmetric_f64", test_angulars_symmetric<f64_t>, cpu_best<nk_angulars_symmetric_f64_best>);
    check("angulars_symmetric_f32", test_angulars_symmetric<f32_t>, cpu_best<nk_angulars_symmetric_f32_best>);
    check("angulars_symmetric_bf16", test_angulars_symmetric<bf16_t>, cpu_best<nk_angulars_symmetric_bf16_best>);
    check("angulars_symmetric_f16", test_angulars_symmetric<f16_t>, cpu_best<nk_angulars_symmetric_f16_best>);
    check("angulars_symmetric_e5m2", test_angulars_symmetric<e5m2_t>, cpu_best<nk_angulars_symmetric_e5m2_best>);
    check("angulars_symmetric_e4m3", test_angulars_symmetric<e4m3_t>, cpu_best<nk_angulars_symmetric_e4m3_best>);
    check("angulars_symmetric_e3m2", test_angulars_symmetric<e3m2_t>, cpu_best<nk_angulars_symmetric_e3m2_best>);
    check("angulars_symmetric_e2m3", test_angulars_symmetric<e2m3_t>, cpu_best<nk_angulars_symmetric_e2m3_best>);
    check("angulars_symmetric_e2m1", test_angulars_symmetric<e2m1x2_t>, cpu_best<nk_angulars_symmetric_e2m1_best>);
    check("angulars_symmetric_i8", test_angulars_symmetric<i8_t>, cpu_best<nk_angulars_symmetric_i8_best>);
    check("angulars_symmetric_i4", test_angulars_symmetric<i4x2_t>, cpu_best<nk_angulars_symmetric_i4_best>);
    check("angulars_symmetric_u8", test_angulars_symmetric<u8_t>, cpu_best<nk_angulars_symmetric_u8_best>);
    check("angulars_symmetric_u4", test_angulars_symmetric<u4x2_t>, cpu_best<nk_angulars_symmetric_u4_best>);

    check("euclideans_packed_f64", test_euclideans_packed<f64_t>, cpu_best<nk_dots_pack_size_f64_best>,
          cpu_best<nk_dots_pack_f64_best>, cpu_best<nk_euclideans_packed_f64_best>);
    check("euclideans_packed_f32", test_euclideans_packed<f32_t>, cpu_best<nk_dots_pack_size_f32_best>,
          cpu_best<nk_dots_pack_f32_best>, cpu_best<nk_euclideans_packed_f32_best>);
    check("euclideans_packed_bf16", test_euclideans_packed<bf16_t>, cpu_best<nk_dots_pack_size_bf16_best>,
          cpu_best<nk_dots_pack_bf16_best>, cpu_best<nk_euclideans_packed_bf16_best>);
    check("euclideans_packed_f16", test_euclideans_packed<f16_t>, cpu_best<nk_dots_pack_size_f16_best>,
          cpu_best<nk_dots_pack_f16_best>, cpu_best<nk_euclideans_packed_f16_best>);
    check("euclideans_packed_e5m2", test_euclideans_packed<e5m2_t>, cpu_best<nk_dots_pack_size_e5m2_best>,
          cpu_best<nk_dots_pack_e5m2_best>, cpu_best<nk_euclideans_packed_e5m2_best>);
    check("euclideans_packed_e4m3", test_euclideans_packed<e4m3_t>, cpu_best<nk_dots_pack_size_e4m3_best>,
          cpu_best<nk_dots_pack_e4m3_best>, cpu_best<nk_euclideans_packed_e4m3_best>);
    check("euclideans_packed_e3m2", test_euclideans_packed<e3m2_t>, cpu_best<nk_dots_pack_size_e3m2_best>,
          cpu_best<nk_dots_pack_e3m2_best>, cpu_best<nk_euclideans_packed_e3m2_best>);
    check("euclideans_packed_e2m3", test_euclideans_packed<e2m3_t>, cpu_best<nk_dots_pack_size_e2m3_best>,
          cpu_best<nk_dots_pack_e2m3_best>, cpu_best<nk_euclideans_packed_e2m3_best>);
    check("euclideans_packed_e2m1", test_euclideans_packed<e2m1x2_t>, cpu_best<nk_dots_pack_size_e2m1_best>,
          cpu_best<nk_dots_pack_e2m1_best>, cpu_best<nk_euclideans_packed_e2m1_best>);
    check("euclideans_packed_i8", test_euclideans_packed<i8_t>, cpu_best<nk_dots_pack_size_i8_best>,
          cpu_best<nk_dots_pack_i8_best>, cpu_best<nk_euclideans_packed_i8_best>);
    check("euclideans_packed_i4", test_euclideans_packed<i4x2_t>, cpu_best<nk_dots_pack_size_i4_best>,
          cpu_best<nk_dots_pack_i4_best>, cpu_best<nk_euclideans_packed_i4_best>);
    check("euclideans_packed_u8", test_euclideans_packed<u8_t>, cpu_best<nk_dots_pack_size_u8_best>,
          cpu_best<nk_dots_pack_u8_best>, cpu_best<nk_euclideans_packed_u8_best>);
    check("euclideans_packed_u4", test_euclideans_packed<u4x2_t>, cpu_best<nk_dots_pack_size_u4_best>,
          cpu_best<nk_dots_pack_u4_best>, cpu_best<nk_euclideans_packed_u4_best>);

    check("euclideans_symmetric_f64", test_euclideans_symmetric<f64_t>, cpu_best<nk_euclideans_symmetric_f64_best>);
    check("euclideans_symmetric_f32", test_euclideans_symmetric<f32_t>, cpu_best<nk_euclideans_symmetric_f32_best>);
    check("euclideans_symmetric_bf16", test_euclideans_symmetric<bf16_t>, cpu_best<nk_euclideans_symmetric_bf16_best>);
    check("euclideans_symmetric_f16", test_euclideans_symmetric<f16_t>, cpu_best<nk_euclideans_symmetric_f16_best>);
    check("euclideans_symmetric_e5m2", test_euclideans_symmetric<e5m2_t>, cpu_best<nk_euclideans_symmetric_e5m2_best>);
    check("euclideans_symmetric_e4m3", test_euclideans_symmetric<e4m3_t>, cpu_best<nk_euclideans_symmetric_e4m3_best>);
    check("euclideans_symmetric_e3m2", test_euclideans_symmetric<e3m2_t>, cpu_best<nk_euclideans_symmetric_e3m2_best>);
    check("euclideans_symmetric_e2m3", test_euclideans_symmetric<e2m3_t>, cpu_best<nk_euclideans_symmetric_e2m3_best>);
    check("euclideans_symmetric_e2m1", test_euclideans_symmetric<e2m1x2_t>,
          cpu_best<nk_euclideans_symmetric_e2m1_best>);
    check("euclideans_symmetric_i8", test_euclideans_symmetric<i8_t>, cpu_best<nk_euclideans_symmetric_i8_best>);
    check("euclideans_symmetric_i4", test_euclideans_symmetric<i4x2_t>, cpu_best<nk_euclideans_symmetric_i4_best>);
    check("euclideans_symmetric_u8", test_euclideans_symmetric<u8_t>, cpu_best<nk_euclideans_symmetric_u8_best>);
    check("euclideans_symmetric_u4", test_euclideans_symmetric<u4x2_t>, cpu_best<nk_euclideans_symmetric_u4_best>);
#endif
}

} // namespace ashvardanian::numkong::test
