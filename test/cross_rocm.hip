/**
 *  @file test/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date September 28, 2026
 *  @brief Batch operation tests - ROCm capabilities.
 *
 *  Compiled with every capability that some listed architecture runs: the ROCm baseline, then
 *  CDNA3, CDNA4 and CDNA5.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "cross_device.hpp"
#include "geospatial.hpp" // `test_geodesics_packed`, `test_terrain_packed`
#include "reduce.hpp"     // `test_reduce_moments`, `test_reduce_minmax`

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_ROCM_

void run_cross_rocm_cases() {
    auto const rocm = nk_cap_rocm_k;
    run_kernel_case("nk_dots_packed_f64_rocm", test_dots_packed<f64_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f64_rocm, nk_dots_pack_f64_rocm, nk_dots_packed_f64_rocm);
    run_kernel_case("nk_dots_pack_f64_rocm",
                    test_dots_pack_layout<f64_t, rocm_backend_t, nk_dots_pack_size_f64_rocm,
                                          nk_dots_packed_shape_f64_rocm, nk_dots_pack_f64_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<f64_t, rocm_backend_t, nk_dots_pack_size_f64_rocm, " //
                    "nk_dots_packed_f64_rocm, nk_dots_symmetric_f64_rocm>",
                    test_dots_launch_contract<f64_t, rocm_backend_t, nk_dots_pack_size_f64_rocm,
                                              nk_dots_packed_f64_rocm, nk_dots_symmetric_f64_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_f64_rocm", test_dots_symmetric<f64_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_f64_rocm);
    run_kernel_case("nk_dots_packed_f32_rocm", test_dots_packed<f32_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f32_rocm, nk_dots_pack_f32_rocm, nk_dots_packed_f32_rocm);
    run_kernel_case("nk_dots_pack_f32_rocm",
                    test_dots_pack_layout<f32_t, rocm_backend_t, nk_dots_pack_size_f32_rocm,
                                          nk_dots_packed_shape_f32_rocm, nk_dots_pack_f32_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<f32_t, rocm_backend_t, nk_dots_pack_size_f32_rocm, " //
                    "nk_dots_packed_f32_rocm, nk_dots_symmetric_f32_rocm>",
                    test_dots_launch_contract<f32_t, rocm_backend_t, nk_dots_pack_size_f32_rocm,
                                              nk_dots_packed_f32_rocm, nk_dots_symmetric_f32_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_f32_rocm", test_dots_symmetric<f32_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_f32_rocm);
    run_kernel_case("nk_dots_packed_bf16_rocm", test_dots_packed<bf16_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_bf16_rocm, nk_dots_pack_bf16_rocm, nk_dots_packed_bf16_rocm);
    run_kernel_case("nk_dots_pack_bf16_rocm",
                    test_dots_pack_layout<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_rocm,
                                          nk_dots_packed_shape_bf16_rocm, nk_dots_pack_bf16_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_rocm, " //
                    "nk_dots_packed_bf16_rocm, nk_dots_symmetric_bf16_rocm>",
                    test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_rocm,
                                              nk_dots_packed_bf16_rocm, nk_dots_symmetric_bf16_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_bf16_rocm", test_dots_symmetric<bf16_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_bf16_rocm);
    run_kernel_case("nk_dots_packed_f16_rocm", test_dots_packed<f16_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f16_rocm, nk_dots_pack_f16_rocm, nk_dots_packed_f16_rocm);
    run_kernel_case("nk_dots_pack_f16_rocm",
                    test_dots_pack_layout<f16_t, rocm_backend_t, nk_dots_pack_size_f16_rocm,
                                          nk_dots_packed_shape_f16_rocm, nk_dots_pack_f16_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_rocm, " //
                    "nk_dots_packed_f16_rocm, nk_dots_symmetric_f16_rocm>",
                    test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_rocm,
                                              nk_dots_packed_f16_rocm, nk_dots_symmetric_f16_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_f16_rocm", test_dots_symmetric<f16_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_f16_rocm);
    run_kernel_case("nk_dots_packed_e5m2_rocm", test_dots_packed<e5m2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e5m2_rocm, nk_dots_pack_e5m2_rocm, nk_dots_packed_e5m2_rocm);
    run_kernel_case("nk_dots_pack_e5m2_rocm",
                    test_dots_pack_layout<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_rocm,
                                          nk_dots_packed_shape_e5m2_rocm, nk_dots_pack_e5m2_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_rocm, " //
                    "nk_dots_packed_e5m2_rocm, nk_dots_symmetric_e5m2_rocm>",
                    test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_rocm,
                                              nk_dots_packed_e5m2_rocm, nk_dots_symmetric_e5m2_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_e5m2_rocm", test_dots_symmetric<e5m2_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_e5m2_rocm);
    run_kernel_case("nk_dots_packed_e4m3_rocm", test_dots_packed<e4m3_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e4m3_rocm, nk_dots_pack_e4m3_rocm, nk_dots_packed_e4m3_rocm);
    run_kernel_case("nk_dots_pack_e4m3_rocm",
                    test_dots_pack_layout<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_rocm,
                                          nk_dots_packed_shape_e4m3_rocm, nk_dots_pack_e4m3_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_rocm, " //
                    "nk_dots_packed_e4m3_rocm, nk_dots_symmetric_e4m3_rocm>",
                    test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_rocm,
                                              nk_dots_packed_e4m3_rocm, nk_dots_symmetric_e4m3_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_e4m3_rocm", test_dots_symmetric<e4m3_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_e4m3_rocm);
    run_kernel_case("nk_dots_packed_e3m2_rocm", test_dots_packed<e3m2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e3m2_rocm, nk_dots_pack_e3m2_rocm, nk_dots_packed_e3m2_rocm);
    run_kernel_case("nk_dots_pack_e3m2_rocm",
                    test_dots_pack_layout<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_rocm,
                                          nk_dots_packed_shape_e3m2_rocm, nk_dots_pack_e3m2_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_rocm, " //
                    "nk_dots_packed_e3m2_rocm, nk_dots_symmetric_e3m2_rocm>",
                    test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_rocm,
                                              nk_dots_packed_e3m2_rocm, nk_dots_symmetric_e3m2_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_e3m2_rocm", test_dots_symmetric<e3m2_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_e3m2_rocm);
    run_kernel_case("nk_dots_packed_e2m3_rocm", test_dots_packed<e2m3_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e2m3_rocm, nk_dots_pack_e2m3_rocm, nk_dots_packed_e2m3_rocm);
    run_kernel_case("nk_dots_pack_e2m3_rocm",
                    test_dots_pack_layout<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_rocm,
                                          nk_dots_packed_shape_e2m3_rocm, nk_dots_pack_e2m3_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_rocm, " //
                    "nk_dots_packed_e2m3_rocm, nk_dots_symmetric_e2m3_rocm>",
                    test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_rocm,
                                              nk_dots_packed_e2m3_rocm, nk_dots_symmetric_e2m3_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_e2m3_rocm", test_dots_symmetric<e2m3_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_e2m3_rocm);
    run_kernel_case("nk_dots_packed_e2m1_rocm", test_dots_packed<e2m1x2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e2m1_rocm, nk_dots_pack_e2m1_rocm, nk_dots_packed_e2m1_rocm);
    run_kernel_case("nk_dots_pack_e2m1_rocm",
                    test_dots_pack_layout<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_rocm,
                                          nk_dots_packed_shape_e2m1_rocm, nk_dots_pack_e2m1_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_rocm, " //
                    "nk_dots_packed_e2m1_rocm, nk_dots_symmetric_e2m1_rocm>",
                    test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_rocm,
                                              nk_dots_packed_e2m1_rocm, nk_dots_symmetric_e2m1_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_e2m1_rocm", test_dots_symmetric<e2m1x2_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_e2m1_rocm);
    run_kernel_case("nk_dots_packed_i8_rocm", test_dots_packed<i8_t, rocm_backend_t>, rocm, nk_dots_pack_size_i8_rocm,
                    nk_dots_pack_i8_rocm, nk_dots_packed_i8_rocm);
    run_kernel_case("nk_dots_pack_i8_rocm",
                    test_dots_pack_layout<i8_t, rocm_backend_t, nk_dots_pack_size_i8_rocm, nk_dots_packed_shape_i8_rocm,
                                          nk_dots_pack_i8_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_rocm, " //
                    "nk_dots_packed_i8_rocm, nk_dots_symmetric_i8_rocm>",
                    test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_rocm, nk_dots_packed_i8_rocm,
                                              nk_dots_symmetric_i8_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_i8_rocm", test_dots_symmetric<i8_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_i8_rocm);
    run_kernel_case("nk_dots_packed_u8_rocm", test_dots_packed<u8_t, rocm_backend_t>, rocm, nk_dots_pack_size_u8_rocm,
                    nk_dots_pack_u8_rocm, nk_dots_packed_u8_rocm);
    run_kernel_case("nk_dots_pack_u8_rocm",
                    test_dots_pack_layout<u8_t, rocm_backend_t, nk_dots_pack_size_u8_rocm, nk_dots_packed_shape_u8_rocm,
                                          nk_dots_pack_u8_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_rocm, " //
                    "nk_dots_packed_u8_rocm, nk_dots_symmetric_u8_rocm>",
                    test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_rocm, nk_dots_packed_u8_rocm,
                                              nk_dots_symmetric_u8_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_u8_rocm", test_dots_symmetric<u8_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_u8_rocm);
    run_kernel_case("nk_dots_packed_i4_rocm", test_dots_packed<i4x2_t, rocm_backend_t>, rocm, nk_dots_pack_size_i4_rocm,
                    nk_dots_pack_i4_rocm, nk_dots_packed_i4_rocm);
    run_kernel_case("nk_dots_pack_i4_rocm",
                    test_dots_pack_layout<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_rocm,
                                          nk_dots_packed_shape_i4_rocm, nk_dots_pack_i4_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_rocm, " //
                    "nk_dots_packed_i4_rocm, nk_dots_symmetric_i4_rocm>",
                    test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_rocm, nk_dots_packed_i4_rocm,
                                              nk_dots_symmetric_i4_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_i4_rocm", test_dots_symmetric<i4x2_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_i4_rocm);
    run_kernel_case("nk_dots_packed_u4_rocm", test_dots_packed<u4x2_t, rocm_backend_t>, rocm, nk_dots_pack_size_u4_rocm,
                    nk_dots_pack_u4_rocm, nk_dots_packed_u4_rocm);
    run_kernel_case("nk_dots_pack_u4_rocm",
                    test_dots_pack_layout<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_rocm,
                                          nk_dots_packed_shape_u4_rocm, nk_dots_pack_u4_rocm>,
                    rocm);
    run_kernel_case("test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_rocm, " //
                    "nk_dots_packed_u4_rocm, nk_dots_symmetric_u4_rocm>",
                    test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_rocm, nk_dots_packed_u4_rocm,
                                              nk_dots_symmetric_u4_rocm>,
                    rocm);
    run_kernel_case("nk_dots_symmetric_u4_rocm", test_dots_symmetric<u4x2_t, rocm_backend_t>, rocm,
                    nk_dots_symmetric_u4_rocm);
    run_kernel_case("nk_angulars_packed_f64_rocm", test_angulars_packed<f64_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f64_rocm, nk_dots_pack_f64_rocm, nk_angulars_packed_f64_rocm);
    run_kernel_case("nk_angulars_symmetric_f64_rocm", test_angulars_symmetric<f64_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_f64_rocm);
    run_kernel_case("nk_euclideans_packed_f64_rocm", test_euclideans_packed<f64_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f64_rocm, nk_dots_pack_f64_rocm, nk_euclideans_packed_f64_rocm);
    run_kernel_case("nk_euclideans_symmetric_f64_rocm", test_euclideans_symmetric<f64_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_f64_rocm);
    run_kernel_case("nk_angulars_packed_f32_rocm", test_angulars_packed<f32_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f32_rocm, nk_dots_pack_f32_rocm, nk_angulars_packed_f32_rocm);
    run_kernel_case("nk_angulars_symmetric_f32_rocm", test_angulars_symmetric<f32_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_f32_rocm);
    run_kernel_case("nk_euclideans_packed_f32_rocm", test_euclideans_packed<f32_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f32_rocm, nk_dots_pack_f32_rocm, nk_euclideans_packed_f32_rocm);
    run_kernel_case("nk_euclideans_symmetric_f32_rocm", test_euclideans_symmetric<f32_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_f32_rocm);
    run_kernel_case("nk_angulars_packed_bf16_rocm", test_angulars_packed<bf16_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_bf16_rocm, nk_dots_pack_bf16_rocm, nk_angulars_packed_bf16_rocm);
    run_kernel_case("nk_angulars_symmetric_bf16_rocm", test_angulars_symmetric<bf16_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_bf16_rocm);
    run_kernel_case("nk_euclideans_packed_bf16_rocm", test_euclideans_packed<bf16_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_bf16_rocm, nk_dots_pack_bf16_rocm, nk_euclideans_packed_bf16_rocm);
    run_kernel_case("nk_euclideans_symmetric_bf16_rocm", test_euclideans_symmetric<bf16_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_bf16_rocm);
    run_kernel_case("nk_angulars_packed_f16_rocm", test_angulars_packed<f16_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f16_rocm, nk_dots_pack_f16_rocm, nk_angulars_packed_f16_rocm);
    run_kernel_case("nk_angulars_symmetric_f16_rocm", test_angulars_symmetric<f16_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_f16_rocm);
    run_kernel_case("nk_euclideans_packed_f16_rocm", test_euclideans_packed<f16_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_f16_rocm, nk_dots_pack_f16_rocm, nk_euclideans_packed_f16_rocm);
    run_kernel_case("nk_euclideans_symmetric_f16_rocm", test_euclideans_symmetric<f16_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_f16_rocm);
    run_kernel_case("nk_angulars_packed_e5m2_rocm", test_angulars_packed<e5m2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e5m2_rocm, nk_dots_pack_e5m2_rocm, nk_angulars_packed_e5m2_rocm);
    run_kernel_case("nk_angulars_symmetric_e5m2_rocm", test_angulars_symmetric<e5m2_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_e5m2_rocm);
    run_kernel_case("nk_euclideans_packed_e5m2_rocm", test_euclideans_packed<e5m2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e5m2_rocm, nk_dots_pack_e5m2_rocm, nk_euclideans_packed_e5m2_rocm);
    run_kernel_case("nk_euclideans_symmetric_e5m2_rocm", test_euclideans_symmetric<e5m2_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_e5m2_rocm);
    run_kernel_case("nk_angulars_packed_e4m3_rocm", test_angulars_packed<e4m3_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e4m3_rocm, nk_dots_pack_e4m3_rocm, nk_angulars_packed_e4m3_rocm);
    run_kernel_case("nk_angulars_symmetric_e4m3_rocm", test_angulars_symmetric<e4m3_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_e4m3_rocm);
    run_kernel_case("nk_euclideans_packed_e4m3_rocm", test_euclideans_packed<e4m3_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e4m3_rocm, nk_dots_pack_e4m3_rocm, nk_euclideans_packed_e4m3_rocm);
    run_kernel_case("nk_euclideans_symmetric_e4m3_rocm", test_euclideans_symmetric<e4m3_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_e4m3_rocm);
    run_kernel_case("nk_angulars_packed_e3m2_rocm", test_angulars_packed<e3m2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e3m2_rocm, nk_dots_pack_e3m2_rocm, nk_angulars_packed_e3m2_rocm);
    run_kernel_case("nk_angulars_symmetric_e3m2_rocm", test_angulars_symmetric<e3m2_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_e3m2_rocm);
    run_kernel_case("nk_euclideans_packed_e3m2_rocm", test_euclideans_packed<e3m2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e3m2_rocm, nk_dots_pack_e3m2_rocm, nk_euclideans_packed_e3m2_rocm);
    run_kernel_case("nk_euclideans_symmetric_e3m2_rocm", test_euclideans_symmetric<e3m2_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_e3m2_rocm);
    run_kernel_case("nk_angulars_packed_e2m3_rocm", test_angulars_packed<e2m3_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e2m3_rocm, nk_dots_pack_e2m3_rocm, nk_angulars_packed_e2m3_rocm);
    run_kernel_case("nk_angulars_symmetric_e2m3_rocm", test_angulars_symmetric<e2m3_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_e2m3_rocm);
    run_kernel_case("nk_euclideans_packed_e2m3_rocm", test_euclideans_packed<e2m3_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e2m3_rocm, nk_dots_pack_e2m3_rocm, nk_euclideans_packed_e2m3_rocm);
    run_kernel_case("nk_euclideans_symmetric_e2m3_rocm", test_euclideans_symmetric<e2m3_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_e2m3_rocm);
    run_kernel_case("nk_angulars_packed_e2m1_rocm", test_angulars_packed<e2m1x2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e2m1_rocm, nk_dots_pack_e2m1_rocm, nk_angulars_packed_e2m1_rocm);
    run_kernel_case("nk_angulars_symmetric_e2m1_rocm", test_angulars_symmetric<e2m1x2_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_e2m1_rocm);
    run_kernel_case("nk_euclideans_packed_e2m1_rocm", test_euclideans_packed<e2m1x2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_e2m1_rocm, nk_dots_pack_e2m1_rocm, nk_euclideans_packed_e2m1_rocm);
    run_kernel_case("nk_euclideans_symmetric_e2m1_rocm", test_euclideans_symmetric<e2m1x2_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_e2m1_rocm);
    run_kernel_case("nk_angulars_packed_i8_rocm", test_angulars_packed<i8_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_i8_rocm, nk_dots_pack_i8_rocm, nk_angulars_packed_i8_rocm);
    run_kernel_case("nk_angulars_symmetric_i8_rocm", test_angulars_symmetric<i8_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_i8_rocm);
    run_kernel_case("nk_euclideans_packed_i8_rocm", test_euclideans_packed<i8_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_i8_rocm, nk_dots_pack_i8_rocm, nk_euclideans_packed_i8_rocm);
    run_kernel_case("nk_euclideans_symmetric_i8_rocm", test_euclideans_symmetric<i8_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_i8_rocm);
    run_kernel_case("nk_angulars_packed_u8_rocm", test_angulars_packed<u8_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_u8_rocm, nk_dots_pack_u8_rocm, nk_angulars_packed_u8_rocm);
    run_kernel_case("nk_angulars_symmetric_u8_rocm", test_angulars_symmetric<u8_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_u8_rocm);
    run_kernel_case("nk_euclideans_packed_u8_rocm", test_euclideans_packed<u8_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_u8_rocm, nk_dots_pack_u8_rocm, nk_euclideans_packed_u8_rocm);
    run_kernel_case("nk_euclideans_symmetric_u8_rocm", test_euclideans_symmetric<u8_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_u8_rocm);
    run_kernel_case("nk_angulars_packed_i4_rocm", test_angulars_packed<i4x2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_i4_rocm, nk_dots_pack_i4_rocm, nk_angulars_packed_i4_rocm);
    run_kernel_case("nk_angulars_symmetric_i4_rocm", test_angulars_symmetric<i4x2_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_i4_rocm);
    run_kernel_case("nk_euclideans_packed_i4_rocm", test_euclideans_packed<i4x2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_i4_rocm, nk_dots_pack_i4_rocm, nk_euclideans_packed_i4_rocm);
    run_kernel_case("nk_euclideans_symmetric_i4_rocm", test_euclideans_symmetric<i4x2_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_i4_rocm);
    run_kernel_case("nk_angulars_packed_u4_rocm", test_angulars_packed<u4x2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_u4_rocm, nk_dots_pack_u4_rocm, nk_angulars_packed_u4_rocm);
    run_kernel_case("nk_angulars_symmetric_u4_rocm", test_angulars_symmetric<u4x2_t, rocm_backend_t>, rocm,
                    nk_angulars_symmetric_u4_rocm);
    run_kernel_case("nk_euclideans_packed_u4_rocm", test_euclideans_packed<u4x2_t, rocm_backend_t>, rocm,
                    nk_dots_pack_size_u4_rocm, nk_dots_pack_u4_rocm, nk_euclideans_packed_u4_rocm);
    run_kernel_case("nk_euclideans_symmetric_u4_rocm", test_euclideans_symmetric<u4x2_t, rocm_backend_t>, rocm,
                    nk_euclideans_symmetric_u4_rocm);
    run_kernel_case("nk_attention_packed_bf16_rocm",
                    test_attention_packed<bf16_t, rocm_backend_t, attention_weights_t::bits_8_k>, rocm,
                    nk_attention_pack_size_bf16_rocm, nk_attention_pack_bf16_rocm, nk_attention_packed_bf16_rocm);
    run_kernel_case("nk_attention_packed_e4m3_rocm",
                    test_attention_packed<e4m3_t, rocm_backend_t, attention_weights_t::bits_11_k>, rocm,
                    nk_attention_pack_size_e4m3_rocm, nk_attention_pack_e4m3_rocm, nk_attention_packed_e4m3_rocm);
    run_kernel_case("nk_attention_packed_i8_rocm",
                    test_attention_packed<i8_t, rocm_backend_t, attention_weights_t::bits_8_k>, rocm,
                    nk_attention_pack_size_i8_rocm, nk_attention_pack_i8_rocm, nk_attention_packed_i8_rocm);
    run_kernel_case("nk_attention_packed_gradients_bf16_rocm", test_attention_packed_gradients<bf16_t, rocm_backend_t>,
                    rocm, nk_attention_pack_size_bf16_rocm, nk_attention_pack_bf16_rocm,
                    nk_attention_packed_gradients_bf16_rocm);
    run_kernel_case("nk_attention_rope_f32_rocm", test_attention_rope<f32_t, rocm_backend_t>, rocm,
                    nk_attention_rope_f32_rocm);
    run_kernel_case("nk_attention_rope_bf16_rocm", test_attention_rope<bf16_t, rocm_backend_t>, rocm,
                    nk_attention_rope_bf16_rocm);
    run_kernel_case("nk_attention_rope_e4m3_rocm", test_attention_rope<e4m3_t, rocm_backend_t>, rocm,
                    nk_attention_rope_e4m3_rocm);
    run_kernel_case("nk_haversines_packed_f64_rocm",
                    test_geodesics_packed<f64_t, nk_kernel_haversines_packed_k, rocm_backend_t>, rocm,
                    nk_haversines_pack_size_f64_rocm, nk_haversines_pack_f64_rocm, nk_haversines_packed_f64_rocm);
    run_kernel_case("nk_vincentys_packed_f64_rocm",
                    test_geodesics_packed<f64_t, nk_kernel_vincentys_packed_k, rocm_backend_t>, rocm,
                    nk_vincentys_pack_size_f64_rocm, nk_vincentys_pack_f64_rocm, nk_vincentys_packed_f64_rocm);
    run_kernel_case("nk_clearances_packed_f32_rocm", test_terrain_packed<f32_t, rocm_backend_t>, rocm,
                    nk_terrain_pack_size_f32_rocm, nk_terrain_pack_f32_rocm, nk_terrain_heights_packed_f32_rocm,
                    nk_clearances_packed_f32_rocm);
    run_kernel_case("nk_wassersteinprocrustes_packed_f32_rocm",
                    test_transport_packed<f32_t, nk_kernel_wassersteinprocrustes_packed_k, rocm_backend_t>, rocm,
                    nk_wassersteinprocrustes_workspace_size_f32_rocm, nk_wassersteinprocrustes_packed_f32_rocm);
    run_kernel_case("nk_wassersteinprocrustes_packed_f16_rocm",
                    test_transport_packed<f16_t, nk_kernel_wassersteinprocrustes_packed_k, rocm_backend_t>, rocm,
                    nk_wassersteinprocrustes_workspace_size_f16_rocm, nk_wassersteinprocrustes_packed_f16_rocm);
    run_kernel_case("nk_gromovwassersteins_packed_f32_rocm",
                    test_transport_packed<f32_t, nk_kernel_gromovwassersteins_packed_k, rocm_backend_t>, rocm,
                    nk_gromovwassersteins_workspace_size_f32_rocm, nk_gromovwassersteins_packed_f32_rocm);
    run_kernel_case("nk_gromovwassersteins_packed_f16_rocm",
                    test_transport_packed<f16_t, nk_kernel_gromovwassersteins_packed_k, rocm_backend_t>, rocm,
                    nk_gromovwassersteins_workspace_size_f16_rocm, nk_gromovwassersteins_packed_f16_rocm);

    run_kernel_case("nk_reduce_moments_f32_rocm", test_reduce_moments<f32_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_f32_rocm);
    run_kernel_case("nk_reduce_moments_f64_rocm", test_reduce_moments<f64_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_f64_rocm);
    run_kernel_case("nk_reduce_moments_i8_rocm", test_reduce_moments<i8_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_i8_rocm);
    run_kernel_case("nk_reduce_moments_u8_rocm", test_reduce_moments<u8_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_u8_rocm);
    run_kernel_case("nk_reduce_moments_i16_rocm", test_reduce_moments<i16_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_i16_rocm);
    run_kernel_case("nk_reduce_moments_u16_rocm", test_reduce_moments<u16_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_u16_rocm);
    run_kernel_case("nk_reduce_moments_i32_rocm", test_reduce_moments<i32_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_i32_rocm);
    run_kernel_case("nk_reduce_moments_u32_rocm", test_reduce_moments<u32_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_u32_rocm);
    run_kernel_case("nk_reduce_moments_i64_rocm", test_reduce_moments<i64_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_i64_rocm);
    run_kernel_case("nk_reduce_moments_u64_rocm", test_reduce_moments<u64_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_u64_rocm);
    run_kernel_case("nk_reduce_moments_f16_rocm", test_reduce_moments<f16_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_f16_rocm);
    run_kernel_case("nk_reduce_moments_bf16_rocm", test_reduce_moments<bf16_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_bf16_rocm);
    run_kernel_case("nk_reduce_moments_e4m3_rocm", test_reduce_moments<e4m3_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_e4m3_rocm);
    run_kernel_case("nk_reduce_moments_e5m2_rocm", test_reduce_moments<e5m2_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_e5m2_rocm);
    run_kernel_case("nk_reduce_moments_e2m3_rocm", test_reduce_moments<e2m3_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_e2m3_rocm);
    run_kernel_case("nk_reduce_moments_e3m2_rocm", test_reduce_moments<e3m2_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_e3m2_rocm);
    run_kernel_case("nk_reduce_moments_e2m1_rocm", test_reduce_moments<e2m1x2_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_e2m1_rocm);
    run_kernel_case("nk_reduce_moments_i4_rocm", test_reduce_moments<i4x2_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_i4_rocm);
    run_kernel_case("nk_reduce_moments_u4_rocm", test_reduce_moments<u4x2_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_u4_rocm);
    run_kernel_case("nk_reduce_moments_u1_rocm", test_reduce_moments<u1x8_t, rocm_backend_t>, rocm,
                    nk_reduce_moments_u1_rocm);
    run_kernel_case("nk_reduce_minmax_f32_rocm", test_reduce_minmax<f32_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_f32_rocm);
    run_kernel_case("nk_reduce_minmax_f64_rocm", test_reduce_minmax<f64_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_f64_rocm);
    run_kernel_case("nk_reduce_minmax_i8_rocm", test_reduce_minmax<i8_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_i8_rocm);
    run_kernel_case("nk_reduce_minmax_u8_rocm", test_reduce_minmax<u8_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_u8_rocm);
    run_kernel_case("nk_reduce_minmax_i16_rocm", test_reduce_minmax<i16_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_i16_rocm);
    run_kernel_case("nk_reduce_minmax_u16_rocm", test_reduce_minmax<u16_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_u16_rocm);
    run_kernel_case("nk_reduce_minmax_i32_rocm", test_reduce_minmax<i32_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_i32_rocm);
    run_kernel_case("nk_reduce_minmax_u32_rocm", test_reduce_minmax<u32_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_u32_rocm);
    run_kernel_case("nk_reduce_minmax_i64_rocm", test_reduce_minmax<i64_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_i64_rocm);
    run_kernel_case("nk_reduce_minmax_u64_rocm", test_reduce_minmax<u64_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_u64_rocm);
    run_kernel_case("nk_reduce_minmax_f16_rocm", test_reduce_minmax<f16_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_f16_rocm);
    run_kernel_case("nk_reduce_minmax_bf16_rocm", test_reduce_minmax<bf16_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_bf16_rocm);
    run_kernel_case("nk_reduce_minmax_e4m3_rocm", test_reduce_minmax<e4m3_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_e4m3_rocm);
    run_kernel_case("nk_reduce_minmax_e5m2_rocm", test_reduce_minmax<e5m2_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_e5m2_rocm);
    run_kernel_case("nk_reduce_minmax_e2m3_rocm", test_reduce_minmax<e2m3_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_e2m3_rocm);
    run_kernel_case("nk_reduce_minmax_e3m2_rocm", test_reduce_minmax<e3m2_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_e3m2_rocm);
    run_kernel_case("nk_reduce_minmax_i4_rocm", test_reduce_minmax<i4x2_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_i4_rocm);
    run_kernel_case("nk_reduce_minmax_u4_rocm", test_reduce_minmax<u4x2_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_u4_rocm);
    run_kernel_case("nk_reduce_minmax_u1_rocm", test_reduce_minmax<u1x8_t, rocm_backend_t>, rocm,
                    nk_reduce_minmax_u1_rocm);

#if NUMKONG_TARGET_CDNA3
    auto const cdna3 = nk_cap_cdna3_k;
    run_kernel_case("nk_dots_packed_bf16_cdna3", test_dots_packed<bf16_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_bf16_cdna3, nk_dots_pack_bf16_cdna3, nk_dots_packed_bf16_cdna3);
    run_kernel_case("nk_dots_pack_bf16_cdna3",
                    test_dots_pack_layout<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna3,
                                          nk_dots_packed_shape_bf16_cdna3, nk_dots_pack_bf16_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna3, " //
                    "nk_dots_packed_bf16_cdna3, nk_dots_symmetric_bf16_cdna3>",
                    test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna3,
                                              nk_dots_packed_bf16_cdna3, nk_dots_symmetric_bf16_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_bf16_cdna3", test_dots_symmetric<bf16_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_bf16_cdna3);
    run_kernel_case("nk_dots_packed_f16_cdna3", test_dots_packed<f16_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_f16_cdna3, nk_dots_pack_f16_cdna3, nk_dots_packed_f16_cdna3);
    run_kernel_case("nk_dots_pack_f16_cdna3",
                    test_dots_pack_layout<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna3,
                                          nk_dots_packed_shape_f16_cdna3, nk_dots_pack_f16_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna3, " //
                    "nk_dots_packed_f16_cdna3, nk_dots_symmetric_f16_cdna3>",
                    test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna3,
                                              nk_dots_packed_f16_cdna3, nk_dots_symmetric_f16_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_f16_cdna3", test_dots_symmetric<f16_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_f16_cdna3);
    run_kernel_case("nk_dots_packed_e5m2_cdna3", test_dots_packed<e5m2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e5m2_cdna3, nk_dots_pack_e5m2_cdna3, nk_dots_packed_e5m2_cdna3);
    run_kernel_case("nk_dots_pack_e5m2_cdna3",
                    test_dots_pack_layout<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna3,
                                          nk_dots_packed_shape_e5m2_cdna3, nk_dots_pack_e5m2_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna3, " //
                    "nk_dots_packed_e5m2_cdna3, nk_dots_symmetric_e5m2_cdna3>",
                    test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna3,
                                              nk_dots_packed_e5m2_cdna3, nk_dots_symmetric_e5m2_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_e5m2_cdna3", test_dots_symmetric<e5m2_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_e5m2_cdna3);
    run_kernel_case("nk_dots_packed_e4m3_cdna3", test_dots_packed<e4m3_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e4m3_cdna3, nk_dots_pack_e4m3_cdna3, nk_dots_packed_e4m3_cdna3);
    run_kernel_case("nk_dots_pack_e4m3_cdna3",
                    test_dots_pack_layout<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna3,
                                          nk_dots_packed_shape_e4m3_cdna3, nk_dots_pack_e4m3_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna3, " //
                    "nk_dots_packed_e4m3_cdna3, nk_dots_symmetric_e4m3_cdna3>",
                    test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna3,
                                              nk_dots_packed_e4m3_cdna3, nk_dots_symmetric_e4m3_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_e4m3_cdna3", test_dots_symmetric<e4m3_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_e4m3_cdna3);
    run_kernel_case("nk_dots_packed_e3m2_cdna3", test_dots_packed<e3m2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e3m2_cdna3, nk_dots_pack_e3m2_cdna3, nk_dots_packed_e3m2_cdna3);
    run_kernel_case("nk_dots_pack_e3m2_cdna3",
                    test_dots_pack_layout<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna3,
                                          nk_dots_packed_shape_e3m2_cdna3, nk_dots_pack_e3m2_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna3, " //
                    "nk_dots_packed_e3m2_cdna3, nk_dots_symmetric_e3m2_cdna3>",
                    test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna3,
                                              nk_dots_packed_e3m2_cdna3, nk_dots_symmetric_e3m2_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_e3m2_cdna3", test_dots_symmetric<e3m2_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_e3m2_cdna3);
    run_kernel_case("nk_dots_packed_e2m3_cdna3", test_dots_packed<e2m3_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e2m3_cdna3, nk_dots_pack_e2m3_cdna3, nk_dots_packed_e2m3_cdna3);
    run_kernel_case("nk_dots_pack_e2m3_cdna3",
                    test_dots_pack_layout<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna3,
                                          nk_dots_packed_shape_e2m3_cdna3, nk_dots_pack_e2m3_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna3, " //
                    "nk_dots_packed_e2m3_cdna3, nk_dots_symmetric_e2m3_cdna3>",
                    test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna3,
                                              nk_dots_packed_e2m3_cdna3, nk_dots_symmetric_e2m3_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_e2m3_cdna3", test_dots_symmetric<e2m3_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_e2m3_cdna3);
    run_kernel_case("nk_dots_packed_e2m1_cdna3", test_dots_packed<e2m1x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e2m1_cdna3, nk_dots_pack_e2m1_cdna3, nk_dots_packed_e2m1_cdna3);
    run_kernel_case("nk_dots_pack_e2m1_cdna3",
                    test_dots_pack_layout<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna3,
                                          nk_dots_packed_shape_e2m1_cdna3, nk_dots_pack_e2m1_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna3, " //
                    "nk_dots_packed_e2m1_cdna3, nk_dots_symmetric_e2m1_cdna3>",
                    test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna3,
                                              nk_dots_packed_e2m1_cdna3, nk_dots_symmetric_e2m1_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_e2m1_cdna3", test_dots_symmetric<e2m1x2_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_e2m1_cdna3);
    run_kernel_case("nk_dots_packed_i8_cdna3", test_dots_packed<i8_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_i8_cdna3, nk_dots_pack_i8_cdna3, nk_dots_packed_i8_cdna3);
    run_kernel_case("nk_dots_pack_i8_cdna3",
                    test_dots_pack_layout<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna3,
                                          nk_dots_packed_shape_i8_cdna3, nk_dots_pack_i8_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna3, " //
                    "nk_dots_packed_i8_cdna3, nk_dots_symmetric_i8_cdna3>",
                    test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna3, nk_dots_packed_i8_cdna3,
                                              nk_dots_symmetric_i8_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_i8_cdna3", test_dots_symmetric<i8_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_i8_cdna3);
    run_kernel_case("nk_dots_packed_i4_cdna3", test_dots_packed<i4x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_i4_cdna3, nk_dots_pack_i4_cdna3, nk_dots_packed_i4_cdna3);
    run_kernel_case("nk_dots_pack_i4_cdna3",
                    test_dots_pack_layout<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna3,
                                          nk_dots_packed_shape_i4_cdna3, nk_dots_pack_i4_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna3, " //
                    "nk_dots_packed_i4_cdna3, nk_dots_symmetric_i4_cdna3>",
                    test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna3,
                                              nk_dots_packed_i4_cdna3, nk_dots_symmetric_i4_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_i4_cdna3", test_dots_symmetric<i4x2_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_i4_cdna3);
    run_kernel_case("nk_dots_packed_u8_cdna3", test_dots_packed<u8_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_u8_cdna3, nk_dots_pack_u8_cdna3, nk_dots_packed_u8_cdna3);
    run_kernel_case("nk_dots_pack_u8_cdna3",
                    test_dots_pack_layout<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna3,
                                          nk_dots_packed_shape_u8_cdna3, nk_dots_pack_u8_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna3, " //
                    "nk_dots_packed_u8_cdna3, nk_dots_symmetric_u8_cdna3>",
                    test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna3, nk_dots_packed_u8_cdna3,
                                              nk_dots_symmetric_u8_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_u8_cdna3", test_dots_symmetric<u8_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_u8_cdna3);
    run_kernel_case("nk_dots_packed_u4_cdna3", test_dots_packed<u4x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_u4_cdna3, nk_dots_pack_u4_cdna3, nk_dots_packed_u4_cdna3);
    run_kernel_case("nk_dots_pack_u4_cdna3",
                    test_dots_pack_layout<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna3,
                                          nk_dots_packed_shape_u4_cdna3, nk_dots_pack_u4_cdna3>,
                    cdna3);
    run_kernel_case("test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna3, " //
                    "nk_dots_packed_u4_cdna3, nk_dots_symmetric_u4_cdna3>",
                    test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna3,
                                              nk_dots_packed_u4_cdna3, nk_dots_symmetric_u4_cdna3>,
                    cdna3);
    run_kernel_case("nk_dots_symmetric_u4_cdna3", test_dots_symmetric<u4x2_t, rocm_backend_t>, cdna3,
                    nk_dots_symmetric_u4_cdna3);
    run_kernel_case("nk_angulars_packed_bf16_cdna3", test_angulars_packed<bf16_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_bf16_cdna3, nk_dots_pack_bf16_cdna3, nk_angulars_packed_bf16_cdna3);
    run_kernel_case("nk_angulars_symmetric_bf16_cdna3", test_angulars_symmetric<bf16_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_bf16_cdna3);
    run_kernel_case("nk_euclideans_packed_bf16_cdna3", test_euclideans_packed<bf16_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_bf16_cdna3, nk_dots_pack_bf16_cdna3, nk_euclideans_packed_bf16_cdna3);
    run_kernel_case("nk_euclideans_symmetric_bf16_cdna3", test_euclideans_symmetric<bf16_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_bf16_cdna3);
    run_kernel_case("nk_angulars_packed_f16_cdna3", test_angulars_packed<f16_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_f16_cdna3, nk_dots_pack_f16_cdna3, nk_angulars_packed_f16_cdna3);
    run_kernel_case("nk_angulars_symmetric_f16_cdna3", test_angulars_symmetric<f16_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_f16_cdna3);
    run_kernel_case("nk_euclideans_packed_f16_cdna3", test_euclideans_packed<f16_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_f16_cdna3, nk_dots_pack_f16_cdna3, nk_euclideans_packed_f16_cdna3);
    run_kernel_case("nk_euclideans_symmetric_f16_cdna3", test_euclideans_symmetric<f16_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_f16_cdna3);
    run_kernel_case("nk_angulars_packed_e5m2_cdna3", test_angulars_packed<e5m2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e5m2_cdna3, nk_dots_pack_e5m2_cdna3, nk_angulars_packed_e5m2_cdna3);
    run_kernel_case("nk_angulars_symmetric_e5m2_cdna3", test_angulars_symmetric<e5m2_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_e5m2_cdna3);
    run_kernel_case("nk_euclideans_packed_e5m2_cdna3", test_euclideans_packed<e5m2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e5m2_cdna3, nk_dots_pack_e5m2_cdna3, nk_euclideans_packed_e5m2_cdna3);
    run_kernel_case("nk_euclideans_symmetric_e5m2_cdna3", test_euclideans_symmetric<e5m2_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_e5m2_cdna3);
    run_kernel_case("nk_angulars_packed_e4m3_cdna3", test_angulars_packed<e4m3_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e4m3_cdna3, nk_dots_pack_e4m3_cdna3, nk_angulars_packed_e4m3_cdna3);
    run_kernel_case("nk_angulars_symmetric_e4m3_cdna3", test_angulars_symmetric<e4m3_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_e4m3_cdna3);
    run_kernel_case("nk_euclideans_packed_e4m3_cdna3", test_euclideans_packed<e4m3_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e4m3_cdna3, nk_dots_pack_e4m3_cdna3, nk_euclideans_packed_e4m3_cdna3);
    run_kernel_case("nk_euclideans_symmetric_e4m3_cdna3", test_euclideans_symmetric<e4m3_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_e4m3_cdna3);
    run_kernel_case("nk_angulars_packed_e3m2_cdna3", test_angulars_packed<e3m2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e3m2_cdna3, nk_dots_pack_e3m2_cdna3, nk_angulars_packed_e3m2_cdna3);
    run_kernel_case("nk_angulars_symmetric_e3m2_cdna3", test_angulars_symmetric<e3m2_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_e3m2_cdna3);
    run_kernel_case("nk_euclideans_packed_e3m2_cdna3", test_euclideans_packed<e3m2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e3m2_cdna3, nk_dots_pack_e3m2_cdna3, nk_euclideans_packed_e3m2_cdna3);
    run_kernel_case("nk_euclideans_symmetric_e3m2_cdna3", test_euclideans_symmetric<e3m2_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_e3m2_cdna3);
    run_kernel_case("nk_angulars_packed_e2m3_cdna3", test_angulars_packed<e2m3_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e2m3_cdna3, nk_dots_pack_e2m3_cdna3, nk_angulars_packed_e2m3_cdna3);
    run_kernel_case("nk_angulars_symmetric_e2m3_cdna3", test_angulars_symmetric<e2m3_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_e2m3_cdna3);
    run_kernel_case("nk_euclideans_packed_e2m3_cdna3", test_euclideans_packed<e2m3_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e2m3_cdna3, nk_dots_pack_e2m3_cdna3, nk_euclideans_packed_e2m3_cdna3);
    run_kernel_case("nk_euclideans_symmetric_e2m3_cdna3", test_euclideans_symmetric<e2m3_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_e2m3_cdna3);
    run_kernel_case("nk_angulars_packed_e2m1_cdna3", test_angulars_packed<e2m1x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e2m1_cdna3, nk_dots_pack_e2m1_cdna3, nk_angulars_packed_e2m1_cdna3);
    run_kernel_case("nk_angulars_symmetric_e2m1_cdna3", test_angulars_symmetric<e2m1x2_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_e2m1_cdna3);
    run_kernel_case("nk_euclideans_packed_e2m1_cdna3", test_euclideans_packed<e2m1x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_e2m1_cdna3, nk_dots_pack_e2m1_cdna3, nk_euclideans_packed_e2m1_cdna3);
    run_kernel_case("nk_euclideans_symmetric_e2m1_cdna3", test_euclideans_symmetric<e2m1x2_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_e2m1_cdna3);
    run_kernel_case("nk_angulars_packed_i8_cdna3", test_angulars_packed<i8_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_i8_cdna3, nk_dots_pack_i8_cdna3, nk_angulars_packed_i8_cdna3);
    run_kernel_case("nk_angulars_symmetric_i8_cdna3", test_angulars_symmetric<i8_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_i8_cdna3);
    run_kernel_case("nk_euclideans_packed_i8_cdna3", test_euclideans_packed<i8_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_i8_cdna3, nk_dots_pack_i8_cdna3, nk_euclideans_packed_i8_cdna3);
    run_kernel_case("nk_euclideans_symmetric_i8_cdna3", test_euclideans_symmetric<i8_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_i8_cdna3);
    run_kernel_case("nk_angulars_packed_i4_cdna3", test_angulars_packed<i4x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_i4_cdna3, nk_dots_pack_i4_cdna3, nk_angulars_packed_i4_cdna3);
    run_kernel_case("nk_angulars_symmetric_i4_cdna3", test_angulars_symmetric<i4x2_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_i4_cdna3);
    run_kernel_case("nk_euclideans_packed_i4_cdna3", test_euclideans_packed<i4x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_i4_cdna3, nk_dots_pack_i4_cdna3, nk_euclideans_packed_i4_cdna3);
    run_kernel_case("nk_euclideans_symmetric_i4_cdna3", test_euclideans_symmetric<i4x2_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_i4_cdna3);
    run_kernel_case("nk_angulars_packed_u8_cdna3", test_angulars_packed<u8_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_u8_cdna3, nk_dots_pack_u8_cdna3, nk_angulars_packed_u8_cdna3);
    run_kernel_case("nk_angulars_symmetric_u8_cdna3", test_angulars_symmetric<u8_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_u8_cdna3);
    run_kernel_case("nk_euclideans_packed_u8_cdna3", test_euclideans_packed<u8_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_u8_cdna3, nk_dots_pack_u8_cdna3, nk_euclideans_packed_u8_cdna3);
    run_kernel_case("nk_euclideans_symmetric_u8_cdna3", test_euclideans_symmetric<u8_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_u8_cdna3);
    run_kernel_case("nk_angulars_packed_u4_cdna3", test_angulars_packed<u4x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_u4_cdna3, nk_dots_pack_u4_cdna3, nk_angulars_packed_u4_cdna3);
    run_kernel_case("nk_angulars_symmetric_u4_cdna3", test_angulars_symmetric<u4x2_t, rocm_backend_t>, cdna3,
                    nk_angulars_symmetric_u4_cdna3);
    run_kernel_case("nk_euclideans_packed_u4_cdna3", test_euclideans_packed<u4x2_t, rocm_backend_t>, cdna3,
                    nk_dots_pack_size_u4_cdna3, nk_dots_pack_u4_cdna3, nk_euclideans_packed_u4_cdna3);
    run_kernel_case("nk_euclideans_symmetric_u4_cdna3", test_euclideans_symmetric<u4x2_t, rocm_backend_t>, cdna3,
                    nk_euclideans_symmetric_u4_cdna3);
    run_kernel_case("nk_attention_packed_bf16_cdna3",
                    test_attention_packed<bf16_t, rocm_backend_t, attention_weights_t::bits_8_k>, cdna3,
                    nk_attention_pack_size_bf16_cdna3, nk_attention_pack_bf16_cdna3, nk_attention_packed_bf16_cdna3);
    run_kernel_case("nk_attention_packed_i8_cdna3",
                    test_attention_packed<i8_t, rocm_backend_t, attention_weights_t::bits_8_k>, cdna3,
                    nk_attention_pack_size_i8_cdna3, nk_attention_pack_i8_cdna3, nk_attention_packed_i8_cdna3);
    run_kernel_case("nk_attention_packed_gradients_bf16_cdna3", test_attention_packed_gradients<bf16_t, rocm_backend_t>,
                    cdna3, nk_attention_pack_size_bf16_cdna3, nk_attention_pack_bf16_cdna3,
                    nk_attention_packed_gradients_bf16_cdna3);
#endif // NUMKONG_TARGET_CDNA3

#if NUMKONG_TARGET_CDNA4
    auto const cdna4 = nk_cap_cdna4_k;
    run_kernel_case("nk_dots_packed_bf16_cdna4", test_dots_packed<bf16_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_bf16_cdna4, nk_dots_pack_bf16_cdna4, nk_dots_packed_bf16_cdna4);
    run_kernel_case("nk_dots_pack_bf16_cdna4",
                    test_dots_pack_layout<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna4,
                                          nk_dots_packed_shape_bf16_cdna4, nk_dots_pack_bf16_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna4, " //
                    "nk_dots_packed_bf16_cdna4, nk_dots_symmetric_bf16_cdna4>",
                    test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna4,
                                              nk_dots_packed_bf16_cdna4, nk_dots_symmetric_bf16_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_bf16_cdna4", test_dots_symmetric<bf16_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_bf16_cdna4);
    run_kernel_case("nk_dots_packed_f16_cdna4", test_dots_packed<f16_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_f16_cdna4, nk_dots_pack_f16_cdna4, nk_dots_packed_f16_cdna4);
    run_kernel_case("nk_dots_pack_f16_cdna4",
                    test_dots_pack_layout<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna4,
                                          nk_dots_packed_shape_f16_cdna4, nk_dots_pack_f16_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna4, " //
                    "nk_dots_packed_f16_cdna4, nk_dots_symmetric_f16_cdna4>",
                    test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna4,
                                              nk_dots_packed_f16_cdna4, nk_dots_symmetric_f16_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_f16_cdna4", test_dots_symmetric<f16_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_f16_cdna4);
    run_kernel_case("nk_dots_packed_e5m2_cdna4", test_dots_packed<e5m2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e5m2_cdna4, nk_dots_pack_e5m2_cdna4, nk_dots_packed_e5m2_cdna4);
    run_kernel_case("nk_dots_pack_e5m2_cdna4",
                    test_dots_pack_layout<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna4,
                                          nk_dots_packed_shape_e5m2_cdna4, nk_dots_pack_e5m2_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna4, " //
                    "nk_dots_packed_e5m2_cdna4, nk_dots_symmetric_e5m2_cdna4>",
                    test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna4,
                                              nk_dots_packed_e5m2_cdna4, nk_dots_symmetric_e5m2_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_e5m2_cdna4", test_dots_symmetric<e5m2_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_e5m2_cdna4);
    run_kernel_case("nk_dots_packed_e4m3_cdna4", test_dots_packed<e4m3_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e4m3_cdna4, nk_dots_pack_e4m3_cdna4, nk_dots_packed_e4m3_cdna4);
    run_kernel_case("nk_dots_pack_e4m3_cdna4",
                    test_dots_pack_layout<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna4,
                                          nk_dots_packed_shape_e4m3_cdna4, nk_dots_pack_e4m3_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna4, " //
                    "nk_dots_packed_e4m3_cdna4, nk_dots_symmetric_e4m3_cdna4>",
                    test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna4,
                                              nk_dots_packed_e4m3_cdna4, nk_dots_symmetric_e4m3_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_e4m3_cdna4", test_dots_symmetric<e4m3_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_e4m3_cdna4);
    run_kernel_case("nk_dots_packed_e3m2_cdna4", test_dots_packed<e3m2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e3m2_cdna4, nk_dots_pack_e3m2_cdna4, nk_dots_packed_e3m2_cdna4);
    run_kernel_case("nk_dots_pack_e3m2_cdna4",
                    test_dots_pack_layout<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna4,
                                          nk_dots_packed_shape_e3m2_cdna4, nk_dots_pack_e3m2_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna4, " //
                    "nk_dots_packed_e3m2_cdna4, nk_dots_symmetric_e3m2_cdna4>",
                    test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna4,
                                              nk_dots_packed_e3m2_cdna4, nk_dots_symmetric_e3m2_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_e3m2_cdna4", test_dots_symmetric<e3m2_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_e3m2_cdna4);
    run_kernel_case("nk_dots_packed_e2m3_cdna4", test_dots_packed<e2m3_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e2m3_cdna4, nk_dots_pack_e2m3_cdna4, nk_dots_packed_e2m3_cdna4);
    run_kernel_case("nk_dots_pack_e2m3_cdna4",
                    test_dots_pack_layout<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna4,
                                          nk_dots_packed_shape_e2m3_cdna4, nk_dots_pack_e2m3_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna4, " //
                    "nk_dots_packed_e2m3_cdna4, nk_dots_symmetric_e2m3_cdna4>",
                    test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna4,
                                              nk_dots_packed_e2m3_cdna4, nk_dots_symmetric_e2m3_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_e2m3_cdna4", test_dots_symmetric<e2m3_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_e2m3_cdna4);
    run_kernel_case("nk_dots_packed_e2m1_cdna4", test_dots_packed<e2m1x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e2m1_cdna4, nk_dots_pack_e2m1_cdna4, nk_dots_packed_e2m1_cdna4);
    run_kernel_case("nk_dots_pack_e2m1_cdna4",
                    test_dots_pack_layout<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna4,
                                          nk_dots_packed_shape_e2m1_cdna4, nk_dots_pack_e2m1_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna4, " //
                    "nk_dots_packed_e2m1_cdna4, nk_dots_symmetric_e2m1_cdna4>",
                    test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna4,
                                              nk_dots_packed_e2m1_cdna4, nk_dots_symmetric_e2m1_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_e2m1_cdna4", test_dots_symmetric<e2m1x2_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_e2m1_cdna4);
    run_kernel_case("nk_dots_packed_i8_cdna4", test_dots_packed<i8_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4, nk_dots_packed_i8_cdna4);
    run_kernel_case("nk_dots_pack_i8_cdna4",
                    test_dots_pack_layout<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna4,
                                          nk_dots_packed_shape_i8_cdna4, nk_dots_pack_i8_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna4, " //
                    "nk_dots_packed_i8_cdna4, nk_dots_symmetric_i8_cdna4>",
                    test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna4, nk_dots_packed_i8_cdna4,
                                              nk_dots_symmetric_i8_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_i8_cdna4", test_dots_symmetric<i8_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_i8_cdna4);
    run_kernel_case("nk_dots_packed_u8_cdna4", test_dots_packed<u8_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4, nk_dots_packed_u8_cdna4);
    run_kernel_case("nk_dots_pack_u8_cdna4",
                    test_dots_pack_layout<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna4,
                                          nk_dots_packed_shape_u8_cdna4, nk_dots_pack_u8_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna4, " //
                    "nk_dots_packed_u8_cdna4, nk_dots_symmetric_u8_cdna4>",
                    test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna4, nk_dots_packed_u8_cdna4,
                                              nk_dots_symmetric_u8_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_u8_cdna4", test_dots_symmetric<u8_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_u8_cdna4);
    run_kernel_case("nk_dots_packed_i4_cdna4", test_dots_packed<i4x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4, nk_dots_packed_i4_cdna4);
    run_kernel_case("nk_dots_pack_i4_cdna4",
                    test_dots_pack_layout<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna4,
                                          nk_dots_packed_shape_i4_cdna4, nk_dots_pack_i4_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna4, " //
                    "nk_dots_packed_i4_cdna4, nk_dots_symmetric_i4_cdna4>",
                    test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna4,
                                              nk_dots_packed_i4_cdna4, nk_dots_symmetric_i4_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_i4_cdna4", test_dots_symmetric<i4x2_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_i4_cdna4);
    run_kernel_case("nk_dots_packed_u4_cdna4", test_dots_packed<u4x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4, nk_dots_packed_u4_cdna4);
    run_kernel_case("nk_dots_pack_u4_cdna4",
                    test_dots_pack_layout<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna4,
                                          nk_dots_packed_shape_u4_cdna4, nk_dots_pack_u4_cdna4>,
                    cdna4);
    run_kernel_case("test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna4, " //
                    "nk_dots_packed_u4_cdna4, nk_dots_symmetric_u4_cdna4>",
                    test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna4,
                                              nk_dots_packed_u4_cdna4, nk_dots_symmetric_u4_cdna4>,
                    cdna4);
    run_kernel_case("nk_dots_symmetric_u4_cdna4", test_dots_symmetric<u4x2_t, rocm_backend_t>, cdna4,
                    nk_dots_symmetric_u4_cdna4);
    run_kernel_case("nk_angulars_packed_bf16_cdna4", test_angulars_packed<bf16_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_bf16_cdna4, nk_dots_pack_bf16_cdna4, nk_angulars_packed_bf16_cdna4);
    run_kernel_case("nk_angulars_symmetric_bf16_cdna4", test_angulars_symmetric<bf16_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_bf16_cdna4);
    run_kernel_case("nk_euclideans_packed_bf16_cdna4", test_euclideans_packed<bf16_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_bf16_cdna4, nk_dots_pack_bf16_cdna4, nk_euclideans_packed_bf16_cdna4);
    run_kernel_case("nk_euclideans_symmetric_bf16_cdna4", test_euclideans_symmetric<bf16_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_bf16_cdna4);
    run_kernel_case("nk_angulars_packed_f16_cdna4", test_angulars_packed<f16_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_f16_cdna4, nk_dots_pack_f16_cdna4, nk_angulars_packed_f16_cdna4);
    run_kernel_case("nk_angulars_symmetric_f16_cdna4", test_angulars_symmetric<f16_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_f16_cdna4);
    run_kernel_case("nk_euclideans_packed_f16_cdna4", test_euclideans_packed<f16_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_f16_cdna4, nk_dots_pack_f16_cdna4, nk_euclideans_packed_f16_cdna4);
    run_kernel_case("nk_euclideans_symmetric_f16_cdna4", test_euclideans_symmetric<f16_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_f16_cdna4);
    run_kernel_case("nk_angulars_packed_e5m2_cdna4", test_angulars_packed<e5m2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e5m2_cdna4, nk_dots_pack_e5m2_cdna4, nk_angulars_packed_e5m2_cdna4);
    run_kernel_case("nk_angulars_symmetric_e5m2_cdna4", test_angulars_symmetric<e5m2_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_e5m2_cdna4);
    run_kernel_case("nk_euclideans_packed_e5m2_cdna4", test_euclideans_packed<e5m2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e5m2_cdna4, nk_dots_pack_e5m2_cdna4, nk_euclideans_packed_e5m2_cdna4);
    run_kernel_case("nk_euclideans_symmetric_e5m2_cdna4", test_euclideans_symmetric<e5m2_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_e5m2_cdna4);
    run_kernel_case("nk_angulars_packed_e4m3_cdna4", test_angulars_packed<e4m3_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e4m3_cdna4, nk_dots_pack_e4m3_cdna4, nk_angulars_packed_e4m3_cdna4);
    run_kernel_case("nk_angulars_symmetric_e4m3_cdna4", test_angulars_symmetric<e4m3_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_e4m3_cdna4);
    run_kernel_case("nk_euclideans_packed_e4m3_cdna4", test_euclideans_packed<e4m3_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e4m3_cdna4, nk_dots_pack_e4m3_cdna4, nk_euclideans_packed_e4m3_cdna4);
    run_kernel_case("nk_euclideans_symmetric_e4m3_cdna4", test_euclideans_symmetric<e4m3_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_e4m3_cdna4);
    run_kernel_case("nk_angulars_packed_e3m2_cdna4", test_angulars_packed<e3m2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e3m2_cdna4, nk_dots_pack_e3m2_cdna4, nk_angulars_packed_e3m2_cdna4);
    run_kernel_case("nk_angulars_symmetric_e3m2_cdna4", test_angulars_symmetric<e3m2_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_e3m2_cdna4);
    run_kernel_case("nk_euclideans_packed_e3m2_cdna4", test_euclideans_packed<e3m2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e3m2_cdna4, nk_dots_pack_e3m2_cdna4, nk_euclideans_packed_e3m2_cdna4);
    run_kernel_case("nk_euclideans_symmetric_e3m2_cdna4", test_euclideans_symmetric<e3m2_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_e3m2_cdna4);
    run_kernel_case("nk_angulars_packed_e2m3_cdna4", test_angulars_packed<e2m3_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e2m3_cdna4, nk_dots_pack_e2m3_cdna4, nk_angulars_packed_e2m3_cdna4);
    run_kernel_case("nk_angulars_symmetric_e2m3_cdna4", test_angulars_symmetric<e2m3_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_e2m3_cdna4);
    run_kernel_case("nk_euclideans_packed_e2m3_cdna4", test_euclideans_packed<e2m3_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e2m3_cdna4, nk_dots_pack_e2m3_cdna4, nk_euclideans_packed_e2m3_cdna4);
    run_kernel_case("nk_euclideans_symmetric_e2m3_cdna4", test_euclideans_symmetric<e2m3_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_e2m3_cdna4);
    run_kernel_case("nk_angulars_packed_e2m1_cdna4", test_angulars_packed<e2m1x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e2m1_cdna4, nk_dots_pack_e2m1_cdna4, nk_angulars_packed_e2m1_cdna4);
    run_kernel_case("nk_angulars_symmetric_e2m1_cdna4", test_angulars_symmetric<e2m1x2_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_e2m1_cdna4);
    run_kernel_case("nk_euclideans_packed_e2m1_cdna4", test_euclideans_packed<e2m1x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_e2m1_cdna4, nk_dots_pack_e2m1_cdna4, nk_euclideans_packed_e2m1_cdna4);
    run_kernel_case("nk_euclideans_symmetric_e2m1_cdna4", test_euclideans_symmetric<e2m1x2_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_e2m1_cdna4);
    run_kernel_case("nk_angulars_packed_i8_cdna4", test_angulars_packed<i8_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4, nk_angulars_packed_i8_cdna4);
    run_kernel_case("nk_angulars_symmetric_i8_cdna4", test_angulars_symmetric<i8_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_i8_cdna4);
    run_kernel_case("nk_euclideans_packed_i8_cdna4", test_euclideans_packed<i8_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_i8_cdna4, nk_dots_pack_i8_cdna4, nk_euclideans_packed_i8_cdna4);
    run_kernel_case("nk_euclideans_symmetric_i8_cdna4", test_euclideans_symmetric<i8_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_i8_cdna4);
    run_kernel_case("nk_angulars_packed_u8_cdna4", test_angulars_packed<u8_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4, nk_angulars_packed_u8_cdna4);
    run_kernel_case("nk_angulars_symmetric_u8_cdna4", test_angulars_symmetric<u8_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_u8_cdna4);
    run_kernel_case("nk_euclideans_packed_u8_cdna4", test_euclideans_packed<u8_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_u8_cdna4, nk_dots_pack_u8_cdna4, nk_euclideans_packed_u8_cdna4);
    run_kernel_case("nk_euclideans_symmetric_u8_cdna4", test_euclideans_symmetric<u8_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_u8_cdna4);
    run_kernel_case("nk_angulars_packed_i4_cdna4", test_angulars_packed<i4x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4, nk_angulars_packed_i4_cdna4);
    run_kernel_case("nk_angulars_symmetric_i4_cdna4", test_angulars_symmetric<i4x2_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_i4_cdna4);
    run_kernel_case("nk_euclideans_packed_i4_cdna4", test_euclideans_packed<i4x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_i4_cdna4, nk_dots_pack_i4_cdna4, nk_euclideans_packed_i4_cdna4);
    run_kernel_case("nk_euclideans_symmetric_i4_cdna4", test_euclideans_symmetric<i4x2_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_i4_cdna4);
    run_kernel_case("nk_angulars_packed_u4_cdna4", test_angulars_packed<u4x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4, nk_angulars_packed_u4_cdna4);
    run_kernel_case("nk_angulars_symmetric_u4_cdna4", test_angulars_symmetric<u4x2_t, rocm_backend_t>, cdna4,
                    nk_angulars_symmetric_u4_cdna4);
    run_kernel_case("nk_euclideans_packed_u4_cdna4", test_euclideans_packed<u4x2_t, rocm_backend_t>, cdna4,
                    nk_dots_pack_size_u4_cdna4, nk_dots_pack_u4_cdna4, nk_euclideans_packed_u4_cdna4);
    run_kernel_case("nk_euclideans_symmetric_u4_cdna4", test_euclideans_symmetric<u4x2_t, rocm_backend_t>, cdna4,
                    nk_euclideans_symmetric_u4_cdna4);
    run_kernel_case("nk_attention_packed_bf16_cdna4",
                    test_attention_packed<bf16_t, rocm_backend_t, attention_weights_t::bits_8_k>, cdna4,
                    nk_attention_pack_size_bf16_cdna4, nk_attention_pack_bf16_cdna4, nk_attention_packed_bf16_cdna4);
    run_kernel_case("nk_attention_packed_e4m3_cdna4",
                    test_attention_packed<e4m3_t, rocm_backend_t, attention_weights_t::bits_11_k>, cdna4,
                    nk_attention_pack_size_e4m3_cdna4, nk_attention_pack_e4m3_cdna4, nk_attention_packed_e4m3_cdna4);
    run_kernel_case("nk_attention_packed_i8_cdna4",
                    test_attention_packed<i8_t, rocm_backend_t, attention_weights_t::bits_8_k>, cdna4,
                    nk_attention_pack_size_i8_cdna4, nk_attention_pack_i8_cdna4, nk_attention_packed_i8_cdna4);
    run_kernel_case("nk_attention_packed_gradients_bf16_cdna4", test_attention_packed_gradients<bf16_t, rocm_backend_t>,
                    cdna4, nk_attention_pack_size_bf16_cdna4, nk_attention_pack_bf16_cdna4,
                    nk_attention_packed_gradients_bf16_cdna4);
#endif // NUMKONG_TARGET_CDNA4

#if NUMKONG_TARGET_CDNA5
    auto const cdna5 = nk_cap_cdna5_k;
    run_kernel_case("nk_dots_packed_bf16_cdna5", test_dots_packed<bf16_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_bf16_cdna5, nk_dots_pack_bf16_cdna5, nk_dots_packed_bf16_cdna5);
    run_kernel_case("nk_dots_pack_bf16_cdna5",
                    test_dots_pack_layout<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna5,
                                          nk_dots_packed_shape_bf16_cdna5, nk_dots_pack_bf16_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna5, " //
                    "nk_dots_packed_bf16_cdna5, nk_dots_symmetric_bf16_cdna5>",
                    test_dots_launch_contract<bf16_t, rocm_backend_t, nk_dots_pack_size_bf16_cdna5,
                                              nk_dots_packed_bf16_cdna5, nk_dots_symmetric_bf16_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_bf16_cdna5", test_dots_symmetric<bf16_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_bf16_cdna5);
    run_kernel_case("nk_dots_packed_f16_cdna5", test_dots_packed<f16_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_f16_cdna5, nk_dots_pack_f16_cdna5, nk_dots_packed_f16_cdna5);
    run_kernel_case("nk_dots_pack_f16_cdna5",
                    test_dots_pack_layout<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna5,
                                          nk_dots_packed_shape_f16_cdna5, nk_dots_pack_f16_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna5, " //
                    "nk_dots_packed_f16_cdna5, nk_dots_symmetric_f16_cdna5>",
                    test_dots_launch_contract<f16_t, rocm_backend_t, nk_dots_pack_size_f16_cdna5,
                                              nk_dots_packed_f16_cdna5, nk_dots_symmetric_f16_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_f16_cdna5", test_dots_symmetric<f16_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_f16_cdna5);
    run_kernel_case("nk_dots_packed_e5m2_cdna5", test_dots_packed<e5m2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e5m2_cdna5, nk_dots_pack_e5m2_cdna5, nk_dots_packed_e5m2_cdna5);
    run_kernel_case("nk_dots_pack_e5m2_cdna5",
                    test_dots_pack_layout<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna5,
                                          nk_dots_packed_shape_e5m2_cdna5, nk_dots_pack_e5m2_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna5, " //
                    "nk_dots_packed_e5m2_cdna5, nk_dots_symmetric_e5m2_cdna5>",
                    test_dots_launch_contract<e5m2_t, rocm_backend_t, nk_dots_pack_size_e5m2_cdna5,
                                              nk_dots_packed_e5m2_cdna5, nk_dots_symmetric_e5m2_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_e5m2_cdna5", test_dots_symmetric<e5m2_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_e5m2_cdna5);
    run_kernel_case("nk_dots_packed_e4m3_cdna5", test_dots_packed<e4m3_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e4m3_cdna5, nk_dots_pack_e4m3_cdna5, nk_dots_packed_e4m3_cdna5);
    run_kernel_case("nk_dots_pack_e4m3_cdna5",
                    test_dots_pack_layout<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna5,
                                          nk_dots_packed_shape_e4m3_cdna5, nk_dots_pack_e4m3_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna5, " //
                    "nk_dots_packed_e4m3_cdna5, nk_dots_symmetric_e4m3_cdna5>",
                    test_dots_launch_contract<e4m3_t, rocm_backend_t, nk_dots_pack_size_e4m3_cdna5,
                                              nk_dots_packed_e4m3_cdna5, nk_dots_symmetric_e4m3_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_e4m3_cdna5", test_dots_symmetric<e4m3_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_e4m3_cdna5);
    run_kernel_case("nk_dots_packed_e3m2_cdna5", test_dots_packed<e3m2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e3m2_cdna5, nk_dots_pack_e3m2_cdna5, nk_dots_packed_e3m2_cdna5);
    run_kernel_case("nk_dots_pack_e3m2_cdna5",
                    test_dots_pack_layout<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna5,
                                          nk_dots_packed_shape_e3m2_cdna5, nk_dots_pack_e3m2_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna5, " //
                    "nk_dots_packed_e3m2_cdna5, nk_dots_symmetric_e3m2_cdna5>",
                    test_dots_launch_contract<e3m2_t, rocm_backend_t, nk_dots_pack_size_e3m2_cdna5,
                                              nk_dots_packed_e3m2_cdna5, nk_dots_symmetric_e3m2_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_e3m2_cdna5", test_dots_symmetric<e3m2_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_e3m2_cdna5);
    run_kernel_case("nk_dots_packed_e2m3_cdna5", test_dots_packed<e2m3_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e2m3_cdna5, nk_dots_pack_e2m3_cdna5, nk_dots_packed_e2m3_cdna5);
    run_kernel_case("nk_dots_pack_e2m3_cdna5",
                    test_dots_pack_layout<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna5,
                                          nk_dots_packed_shape_e2m3_cdna5, nk_dots_pack_e2m3_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna5, " //
                    "nk_dots_packed_e2m3_cdna5, nk_dots_symmetric_e2m3_cdna5>",
                    test_dots_launch_contract<e2m3_t, rocm_backend_t, nk_dots_pack_size_e2m3_cdna5,
                                              nk_dots_packed_e2m3_cdna5, nk_dots_symmetric_e2m3_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_e2m3_cdna5", test_dots_symmetric<e2m3_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_e2m3_cdna5);
    run_kernel_case("nk_dots_packed_e2m1_cdna5", test_dots_packed<e2m1x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e2m1_cdna5, nk_dots_pack_e2m1_cdna5, nk_dots_packed_e2m1_cdna5);
    run_kernel_case("nk_dots_pack_e2m1_cdna5",
                    test_dots_pack_layout<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna5,
                                          nk_dots_packed_shape_e2m1_cdna5, nk_dots_pack_e2m1_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna5, " //
                    "nk_dots_packed_e2m1_cdna5, nk_dots_symmetric_e2m1_cdna5>",
                    test_dots_launch_contract<e2m1x2_t, rocm_backend_t, nk_dots_pack_size_e2m1_cdna5,
                                              nk_dots_packed_e2m1_cdna5, nk_dots_symmetric_e2m1_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_e2m1_cdna5", test_dots_symmetric<e2m1x2_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_e2m1_cdna5);
    run_kernel_case("nk_dots_packed_i8_cdna5", test_dots_packed<i8_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5, nk_dots_packed_i8_cdna5);
    run_kernel_case("nk_dots_pack_i8_cdna5",
                    test_dots_pack_layout<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna5,
                                          nk_dots_packed_shape_i8_cdna5, nk_dots_pack_i8_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna5, " //
                    "nk_dots_packed_i8_cdna5, nk_dots_symmetric_i8_cdna5>",
                    test_dots_launch_contract<i8_t, rocm_backend_t, nk_dots_pack_size_i8_cdna5, nk_dots_packed_i8_cdna5,
                                              nk_dots_symmetric_i8_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_i8_cdna5", test_dots_symmetric<i8_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_i8_cdna5);
    run_kernel_case("nk_dots_packed_u8_cdna5", test_dots_packed<u8_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5, nk_dots_packed_u8_cdna5);
    run_kernel_case("nk_dots_pack_u8_cdna5",
                    test_dots_pack_layout<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna5,
                                          nk_dots_packed_shape_u8_cdna5, nk_dots_pack_u8_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna5, " //
                    "nk_dots_packed_u8_cdna5, nk_dots_symmetric_u8_cdna5>",
                    test_dots_launch_contract<u8_t, rocm_backend_t, nk_dots_pack_size_u8_cdna5, nk_dots_packed_u8_cdna5,
                                              nk_dots_symmetric_u8_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_u8_cdna5", test_dots_symmetric<u8_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_u8_cdna5);
    run_kernel_case("nk_dots_packed_i4_cdna5", test_dots_packed<i4x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5, nk_dots_packed_i4_cdna5);
    run_kernel_case("nk_dots_pack_i4_cdna5",
                    test_dots_pack_layout<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna5,
                                          nk_dots_packed_shape_i4_cdna5, nk_dots_pack_i4_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna5, " //
                    "nk_dots_packed_i4_cdna5, nk_dots_symmetric_i4_cdna5>",
                    test_dots_launch_contract<i4x2_t, rocm_backend_t, nk_dots_pack_size_i4_cdna5,
                                              nk_dots_packed_i4_cdna5, nk_dots_symmetric_i4_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_i4_cdna5", test_dots_symmetric<i4x2_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_i4_cdna5);
    run_kernel_case("nk_dots_packed_u4_cdna5", test_dots_packed<u4x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5, nk_dots_packed_u4_cdna5);
    run_kernel_case("nk_dots_pack_u4_cdna5",
                    test_dots_pack_layout<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna5,
                                          nk_dots_packed_shape_u4_cdna5, nk_dots_pack_u4_cdna5>,
                    cdna5);
    run_kernel_case("test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna5, " //
                    "nk_dots_packed_u4_cdna5, nk_dots_symmetric_u4_cdna5>",
                    test_dots_launch_contract<u4x2_t, rocm_backend_t, nk_dots_pack_size_u4_cdna5,
                                              nk_dots_packed_u4_cdna5, nk_dots_symmetric_u4_cdna5>,
                    cdna5);
    run_kernel_case("nk_dots_symmetric_u4_cdna5", test_dots_symmetric<u4x2_t, rocm_backend_t>, cdna5,
                    nk_dots_symmetric_u4_cdna5);
    run_kernel_case("nk_angulars_packed_bf16_cdna5", test_angulars_packed<bf16_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_bf16_cdna5, nk_dots_pack_bf16_cdna5, nk_angulars_packed_bf16_cdna5);
    run_kernel_case("nk_angulars_symmetric_bf16_cdna5", test_angulars_symmetric<bf16_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_bf16_cdna5);
    run_kernel_case("nk_euclideans_packed_bf16_cdna5", test_euclideans_packed<bf16_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_bf16_cdna5, nk_dots_pack_bf16_cdna5, nk_euclideans_packed_bf16_cdna5);
    run_kernel_case("nk_euclideans_symmetric_bf16_cdna5", test_euclideans_symmetric<bf16_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_bf16_cdna5);
    run_kernel_case("nk_angulars_packed_f16_cdna5", test_angulars_packed<f16_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_f16_cdna5, nk_dots_pack_f16_cdna5, nk_angulars_packed_f16_cdna5);
    run_kernel_case("nk_angulars_symmetric_f16_cdna5", test_angulars_symmetric<f16_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_f16_cdna5);
    run_kernel_case("nk_euclideans_packed_f16_cdna5", test_euclideans_packed<f16_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_f16_cdna5, nk_dots_pack_f16_cdna5, nk_euclideans_packed_f16_cdna5);
    run_kernel_case("nk_euclideans_symmetric_f16_cdna5", test_euclideans_symmetric<f16_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_f16_cdna5);
    run_kernel_case("nk_angulars_packed_e5m2_cdna5", test_angulars_packed<e5m2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e5m2_cdna5, nk_dots_pack_e5m2_cdna5, nk_angulars_packed_e5m2_cdna5);
    run_kernel_case("nk_angulars_symmetric_e5m2_cdna5", test_angulars_symmetric<e5m2_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_e5m2_cdna5);
    run_kernel_case("nk_euclideans_packed_e5m2_cdna5", test_euclideans_packed<e5m2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e5m2_cdna5, nk_dots_pack_e5m2_cdna5, nk_euclideans_packed_e5m2_cdna5);
    run_kernel_case("nk_euclideans_symmetric_e5m2_cdna5", test_euclideans_symmetric<e5m2_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_e5m2_cdna5);
    run_kernel_case("nk_angulars_packed_e4m3_cdna5", test_angulars_packed<e4m3_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e4m3_cdna5, nk_dots_pack_e4m3_cdna5, nk_angulars_packed_e4m3_cdna5);
    run_kernel_case("nk_angulars_symmetric_e4m3_cdna5", test_angulars_symmetric<e4m3_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_e4m3_cdna5);
    run_kernel_case("nk_euclideans_packed_e4m3_cdna5", test_euclideans_packed<e4m3_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e4m3_cdna5, nk_dots_pack_e4m3_cdna5, nk_euclideans_packed_e4m3_cdna5);
    run_kernel_case("nk_euclideans_symmetric_e4m3_cdna5", test_euclideans_symmetric<e4m3_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_e4m3_cdna5);
    run_kernel_case("nk_angulars_packed_e3m2_cdna5", test_angulars_packed<e3m2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e3m2_cdna5, nk_dots_pack_e3m2_cdna5, nk_angulars_packed_e3m2_cdna5);
    run_kernel_case("nk_angulars_symmetric_e3m2_cdna5", test_angulars_symmetric<e3m2_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_e3m2_cdna5);
    run_kernel_case("nk_euclideans_packed_e3m2_cdna5", test_euclideans_packed<e3m2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e3m2_cdna5, nk_dots_pack_e3m2_cdna5, nk_euclideans_packed_e3m2_cdna5);
    run_kernel_case("nk_euclideans_symmetric_e3m2_cdna5", test_euclideans_symmetric<e3m2_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_e3m2_cdna5);
    run_kernel_case("nk_angulars_packed_e2m3_cdna5", test_angulars_packed<e2m3_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e2m3_cdna5, nk_dots_pack_e2m3_cdna5, nk_angulars_packed_e2m3_cdna5);
    run_kernel_case("nk_angulars_symmetric_e2m3_cdna5", test_angulars_symmetric<e2m3_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_e2m3_cdna5);
    run_kernel_case("nk_euclideans_packed_e2m3_cdna5", test_euclideans_packed<e2m3_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e2m3_cdna5, nk_dots_pack_e2m3_cdna5, nk_euclideans_packed_e2m3_cdna5);
    run_kernel_case("nk_euclideans_symmetric_e2m3_cdna5", test_euclideans_symmetric<e2m3_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_e2m3_cdna5);
    run_kernel_case("nk_angulars_packed_e2m1_cdna5", test_angulars_packed<e2m1x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e2m1_cdna5, nk_dots_pack_e2m1_cdna5, nk_angulars_packed_e2m1_cdna5);
    run_kernel_case("nk_angulars_symmetric_e2m1_cdna5", test_angulars_symmetric<e2m1x2_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_e2m1_cdna5);
    run_kernel_case("nk_euclideans_packed_e2m1_cdna5", test_euclideans_packed<e2m1x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_e2m1_cdna5, nk_dots_pack_e2m1_cdna5, nk_euclideans_packed_e2m1_cdna5);
    run_kernel_case("nk_euclideans_symmetric_e2m1_cdna5", test_euclideans_symmetric<e2m1x2_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_e2m1_cdna5);
    run_kernel_case("nk_angulars_packed_i8_cdna5", test_angulars_packed<i8_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5, nk_angulars_packed_i8_cdna5);
    run_kernel_case("nk_angulars_symmetric_i8_cdna5", test_angulars_symmetric<i8_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_i8_cdna5);
    run_kernel_case("nk_euclideans_packed_i8_cdna5", test_euclideans_packed<i8_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_i8_cdna5, nk_dots_pack_i8_cdna5, nk_euclideans_packed_i8_cdna5);
    run_kernel_case("nk_euclideans_symmetric_i8_cdna5", test_euclideans_symmetric<i8_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_i8_cdna5);
    run_kernel_case("nk_angulars_packed_u8_cdna5", test_angulars_packed<u8_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5, nk_angulars_packed_u8_cdna5);
    run_kernel_case("nk_angulars_symmetric_u8_cdna5", test_angulars_symmetric<u8_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_u8_cdna5);
    run_kernel_case("nk_euclideans_packed_u8_cdna5", test_euclideans_packed<u8_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_u8_cdna5, nk_dots_pack_u8_cdna5, nk_euclideans_packed_u8_cdna5);
    run_kernel_case("nk_euclideans_symmetric_u8_cdna5", test_euclideans_symmetric<u8_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_u8_cdna5);
    run_kernel_case("nk_angulars_packed_i4_cdna5", test_angulars_packed<i4x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5, nk_angulars_packed_i4_cdna5);
    run_kernel_case("nk_angulars_symmetric_i4_cdna5", test_angulars_symmetric<i4x2_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_i4_cdna5);
    run_kernel_case("nk_euclideans_packed_i4_cdna5", test_euclideans_packed<i4x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_i4_cdna5, nk_dots_pack_i4_cdna5, nk_euclideans_packed_i4_cdna5);
    run_kernel_case("nk_euclideans_symmetric_i4_cdna5", test_euclideans_symmetric<i4x2_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_i4_cdna5);
    run_kernel_case("nk_angulars_packed_u4_cdna5", test_angulars_packed<u4x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5, nk_angulars_packed_u4_cdna5);
    run_kernel_case("nk_angulars_symmetric_u4_cdna5", test_angulars_symmetric<u4x2_t, rocm_backend_t>, cdna5,
                    nk_angulars_symmetric_u4_cdna5);
    run_kernel_case("nk_euclideans_packed_u4_cdna5", test_euclideans_packed<u4x2_t, rocm_backend_t>, cdna5,
                    nk_dots_pack_size_u4_cdna5, nk_dots_pack_u4_cdna5, nk_euclideans_packed_u4_cdna5);
    run_kernel_case("nk_euclideans_symmetric_u4_cdna5", test_euclideans_symmetric<u4x2_t, rocm_backend_t>, cdna5,
                    nk_euclideans_symmetric_u4_cdna5);
    run_kernel_case("nk_attention_packed_bf16_cdna5",
                    test_attention_packed<bf16_t, rocm_backend_t, attention_weights_t::bits_8_k>, cdna5,
                    nk_attention_pack_size_bf16_cdna5, nk_attention_pack_bf16_cdna5, nk_attention_packed_bf16_cdna5);
    run_kernel_case("nk_attention_packed_e4m3_cdna5",
                    test_attention_packed<e4m3_t, rocm_backend_t, attention_weights_t::bits_11_k>, cdna5,
                    nk_attention_pack_size_e4m3_cdna5, nk_attention_pack_e4m3_cdna5, nk_attention_packed_e4m3_cdna5);
    run_kernel_case("nk_attention_packed_i8_cdna5",
                    test_attention_packed<i8_t, rocm_backend_t, attention_weights_t::bits_8_k>, cdna5,
                    nk_attention_pack_size_i8_cdna5, nk_attention_pack_i8_cdna5, nk_attention_packed_i8_cdna5);
    run_kernel_case("nk_attention_packed_gradients_bf16_cdna5", test_attention_packed_gradients<bf16_t, rocm_backend_t>,
                    cdna5, nk_attention_pack_size_bf16_cdna5, nk_attention_pack_bf16_cdna5,
                    nk_attention_packed_gradients_bf16_cdna5);
#endif // NUMKONG_TARGET_CDNA5

    run_cross_dispatch_cases<rocm_backend_t>();
}

#else  // !NUMKONG_ARCH_ROCM_
void run_cross_rocm_cases() {}
#endif // NUMKONG_ARCH_ROCM_

} // namespace ashvardanian::numkong::test
